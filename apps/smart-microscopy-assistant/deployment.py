"""Smart Microscopy Assistant — VLM-backed analyst for microscopy images.

Accepts a microscopy image (Hypha artifact reference OR HTTPS URL) and either
a free-text instruction (describe-what-you-see) or the name of a previously-
defined "visual test" (few-shot verdict mode), and returns the VLM's textual
judgement.

A visual test is a re-usable definition of "what to look for" in an image.
Each test has:
  - a name
  - a PASS criterion (free text)
  - a FAIL criterion (free text)
  - 0–5 positive reference images and 0–5 negative reference images
  - an owner (Hypha user id) and a public/private flag

inspect() prepends the references (if any) and the PASS/FAIL criteria to
the prompt and asks the VLM to return one of three verdicts: PASSED,
FAILED, or UNSURE (when the visible evidence is ambiguous).

Tests are stored under $HOME/visual_tests/<test-id>/visual_test.json,
where <test-id> is a hash of owner + name. Each user can have a test
named "focus-quality" without colliding with another user's. Public
tests are visible to and usable by everyone; delete is owner-only.

Backed by Qwen2.5-VL-3B-Instruct via HuggingFace transformers on a single
NVIDIA A40-16C vGPU slice (Ampere, sm_86; 16 GB framebuffer time-shared
with co-tenants on the host A40).
"""

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import bioengine
from pydantic import Field
from pydantic.fields import FieldInfo

logger = bioengine.logger

_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
_DEFAULT_SERVER_URL = "https://hypha.aicell.io"

_MAX_IMAGE_BYTES = 25 * 1024 * 1024
_MAX_INSTRUCTION_CHARS = 4000
_MAX_PIXELS = 1280 * 28 * 28
_MAX_LONG_SIDE = 2048
_HARD_REJECT_PIXELS = 200 * 1024 * 1024
_DOWNLOAD_TIMEOUT_S = 30
_GENERATE_TIMEOUT_S = 180
_INSPECT_JOBS_TTL_SEC = 24 * 3600

_EXAMPLE_MAX_PIXELS = 512 * 512
_EXAMPLE_MAX_LONG_SIDE = 768
_MAX_EXAMPLES_PER_CLASS = 5
_MIN_EXAMPLES_PER_CLASS = 0  # text-only tests are allowed
_MAX_TEST_NAME_CHARS = 50
_MAX_TEST_DESC_CHARS = 800
_TEST_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,49}$")
_VERDICT_VALUES = ("passed", "failed", "unsure")

# Reserved owner id for the case where Hypha did not inject a user context
# (e.g. direct calls bypassing Hypha auth). Tests created under this id are
# treated as system-owned and cannot be deleted from the UI.
_ANON_OWNER = "anonymous"


def _resolve_tests_dir() -> Path:
    home = os.environ.get("HOME", "")
    if home and home != "/nonexistent":
        return Path(home) / "visual_tests"
    return Path("/tmp") / "smart-microscopy-assistant" / "visual_tests"


def _arg(value: Any, default: Any) -> Any:
    """Resolve an optional parameter the caller may have omitted.

    ``@bioengine.method(context=True)`` skips pydantic on the replica side,
    so an omitted parameter arrives as the raw ``FieldInfo`` sentinel from
    the signature instead of its declared default (bioengine 0.16.2).
    """
    return default if isinstance(value, FieldInfo) else value


def _owner_from_context(context: Optional[Dict[str, Any]]) -> str:
    """Resolve the caller's stable identity from the injected Hypha context.

    Email first, id second: `user.id` is the token's `sub`, which is a
    per-token synthetic id for API-token callers (it is only stable for
    browser logins), so keying a library on it would strand a user's tests
    every time they mint a new token. Anonymous callers share one bucket and
    therefore can never own a test.
    """
    user = (context or {}).get("user") or {}
    if user.get("is_anonymous"):
        return _ANON_OWNER
    for key in ("email", "id"):
        value = user.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _ANON_OWNER


def _readable_workspaces(context: Optional[Dict[str, Any]]) -> set:
    """Workspaces the caller may read, from the token scope Hypha injected.

    Hypha builds `context` server-side from the caller's own token, so a
    client can neither supply nor widen this. Permission letters are `r`,
    `rw` and `a` (admin); anything else is no read.
    """
    scope = ((context or {}).get("user") or {}).get("scope") or {}
    perms = scope.get("workspaces") or {}
    return {ws for ws, perm in perms.items() if perm in ("r", "rw", "a")}


def _test_id_for(owner: str, name: str) -> str:
    """Filesystem-safe per-owner test directory name.

    We include the user-visible name as a suffix so directory listings stay
    debuggable, but the leading hash guarantees per-owner namespacing so two
    users can both have a test called e.g. "focus-quality".
    """
    h = hashlib.sha1(f"{owner}:{name}".encode("utf-8")).hexdigest()[:10]
    safe_name = re.sub(r"[^a-z0-9-]", "-", name.lower())[:48]
    return f"{h}-{safe_name}"


def _read_pip(name: str) -> List[str]:
    """Load a ``requirements-*.txt`` file next to this module."""
    text = (Path(__file__).parent / name).read_text()
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@bioengine.app(
    num_cpus=4,
    gpu_memory_mb=-1,
    memory_mb=12 * 1024,
    pip=_read_pip("requirements-deployment.txt"),
    env_vars={
        # Triton's JIT cache wants a writable dir; runtime_env venv's
        # default $HOME is read-only on this Ray pod.
        "TRITON_CACHE_DIR": "/tmp/triton-cache",
        "HF_HOME": "/tmp/hf-home",
        "XDG_CACHE_HOME": "/tmp/xdg-cache",
    },
    # Generation is serialised by `_gpu_lock`, so this only has to be wide
    # enough that status polls are never queued behind running inspections.
    max_ongoing_requests=32,
    health_check_period_s=30.0,
    health_check_timeout_s=600.0,
    graceful_shutdown_timeout_s=120.0,
)
class SmartMicroscopyAssistant:
    def __init__(self) -> None:
        self.start_time = time.time()
        self._engine = None
        self._processor = None
        self._server = None
        self._artifact_manager = None
        # Second, token-less Hypha connection. Used to resolve artifact refs
        # the caller has no permission for, so those reach public artifacts
        # and nothing else — see `_resolve_to_url`.
        self._anon_server = None
        self._anon_am = None
        self._tests_dir: Optional[Path] = None
        # Submitted inspect jobs, keyed by run id. In-memory and per replica:
        # a restart drops every id, which `get_inspect_status` says out loud.
        self._inspect_jobs: Dict[str, dict] = {}
        # One generation at a time. Without it, concurrent `to_thread` calls
        # contend for the same GPU and no caller can be told where it is in
        # line — queue_position is only meaningful against a real queue.
        self._gpu_lock = asyncio.Lock()

    @bioengine.async_init
    async def _async_init(self) -> None:
        import os as _os
        for d in ("/tmp/triton-cache", "/tmp/hf-home", "/tmp/xdg-cache"):
            _os.makedirs(d, exist_ok=True)
        self._tests_dir = _resolve_tests_dir()
        self._tests_dir.mkdir(parents=True, exist_ok=True)
        existing = self._list_all_test_records()
        logger.info(
            "Visual-test library at %s (%d test%s)",
            self._tests_dir, len(existing), "" if len(existing) == 1 else "s",
        )

        from hypha_rpc import connect_to_server
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        import torch

        token = os.environ.get("HYPHA_TOKEN")
        if not token:
            raise RuntimeError("HYPHA_TOKEN environment variable is not set.")

        logger.info("Connecting to Hypha for artifact-manager access...")
        self._server = await connect_to_server({
            "server_url": _DEFAULT_SERVER_URL,
            "token": token,
        })
        self._artifact_manager = await self._server.get_service("public/artifact-manager")
        logger.info("Hypha artifact-manager connected.")

        logger.info("Loading Qwen2.5-VL processor (%s)...", _MODEL_ID)
        self._processor = AutoProcessor.from_pretrained(_MODEL_ID)

        logger.info("Loading Qwen2.5-VL-3B weights on cuda:0 (FP16)...")
        self._engine = await asyncio.to_thread(
            Qwen2_5_VLForConditionalGeneration.from_pretrained,
            _MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )
        self._engine.eval()
        logger.info("Qwen2.5-VL-3B ready on %s.", next(self._engine.parameters()).device)

    @bioengine.smoke_test
    async def _smoke_test(self) -> None:
        return None

    @bioengine.health_check
    async def _health_check(self) -> None:
        if self._engine is None or self._processor is None:
            raise RuntimeError("VLM not initialized.")
        if self._artifact_manager is None:
            raise RuntimeError("Hypha artifact-manager not connected.")

    # ---------------------------------------------------------------- helpers

    async def _ensure_artifact_manager(self, *, anonymous: bool = False) -> Any:
        """An artifact-manager handle, reconnecting if the websocket went stale.

        With ``anonymous=True`` the handle carries no token and so reaches
        only publicly readable artifacts.
        """
        am = self._anon_am if anonymous else self._artifact_manager
        if am is not None:
            try:
                await am.list(parent_id="public/applications")
                return am
            except Exception as e:
                msg = str(e)
                if "Connection is closed" not in msg and "WebSocket" not in msg:
                    raise
                logger.warning("Hypha WS appears stale (%s); reconnecting", msg[:120])

        from hypha_rpc import connect_to_server
        config: Dict[str, Any] = {"server_url": _DEFAULT_SERVER_URL}
        if not anonymous:
            token = os.environ.get("HYPHA_TOKEN")
            if not token:
                raise RuntimeError("HYPHA_TOKEN environment variable is not set.")
            config["token"] = token

        stale = self._anon_server if anonymous else self._server
        try:
            if stale and hasattr(stale, "disconnect"):
                await stale.disconnect()
        except Exception:
            pass

        server = await connect_to_server(config)
        am = await server.get_service("public/artifact-manager")
        if anonymous:
            self._anon_server, self._anon_am = server, am
        else:
            self._server, self._artifact_manager = server, am
        logger.info(
            "Connected Hypha artifact-manager (%s).",
            "anonymous" if anonymous else "app token",
        )
        return am

    async def _resolve_to_url(self, image_ref: str, context=None) -> str:
        """Turn an image reference into a fetchable URL.

        An artifact ref is resolved with the app's own token only when the
        caller could have read that workspace themselves; otherwise with no
        token at all, so the ref reaches public artifacts and nothing more.
        Resolving every caller-supplied ref with the app's token would make
        this app a confused deputy — any caller could name any file the
        app's token can read.
        """
        if image_ref.startswith(("http://", "https://")):
            return image_ref
        if ":" not in image_ref or "/" not in image_ref.split(":", 1)[0]:
            raise ValueError(
                "image_ref must be 'https://...' or '<workspace>/<alias>:<path>' "
                f"(got: {image_ref!r})"
            )
        artifact_id, file_path = image_ref.split(":", 1)
        workspace = artifact_id.split("/", 1)[0]
        privileged = workspace in _readable_workspaces(context)

        am = await self._ensure_artifact_manager(anonymous=not privileged)
        try:
            url = await am.get_file(artifact_id=artifact_id, file_path=file_path)
        except Exception as e:
            if privileged:
                raise
            raise PermissionError(
                f"{image_ref!r} is not publicly readable, and the caller has no "
                f"read permission on workspace {workspace!r}."
            ) from e
        if not url:
            raise RuntimeError(
                f"artifact-manager.get_file returned no URL for {image_ref!r}."
            )
        return url

    async def _download_image(
        self,
        url: str,
        max_pixels: int = _MAX_PIXELS,
        max_long_side: int = _MAX_LONG_SIDE,
    ) -> tuple["Image.Image", Optional[tuple[int, int]]]:
        import io
        import httpx
        from PIL import Image

        buf = bytearray()
        async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT_S, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                    buf.extend(chunk)
                    if len(buf) > _MAX_IMAGE_BYTES:
                        raise ValueError(
                            f"Image exceeds {_MAX_IMAGE_BYTES // (1024 * 1024)} MB limit."
                        )

        try:
            img = Image.open(io.BytesIO(bytes(buf)))
            img.load()
        except Exception as e:
            raise ValueError(f"Failed to decode image bytes ({len(buf)} B): {e}") from e

        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        if w * h > _HARD_REJECT_PIXELS:
            raise ValueError(
                f"Image is {w}x{h} ({w * h / 1e6:.0f} MP) which exceeds the "
                f"{_HARD_REJECT_PIXELS // 1024 // 1024} MP hard limit."
            )

        original_size = None
        long_side = max(w, h)
        needs_resize = (w * h > max_pixels) or (long_side > max_long_side)
        if needs_resize:
            t_resize = time.time()
            scale_pix  = (max_pixels / (w * h)) ** 0.5 if w * h > max_pixels else 1.0
            scale_side = max_long_side / long_side if long_side > max_long_side else 1.0
            scale = min(scale_pix, scale_side)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img = img.resize(new_size, Image.LANCZOS)
            original_size = (w, h)
            logger.info(
                "Downscaled %sx%s -> %sx%s (ratio %.3f, %.1f ms)",
                w, h, new_size[0], new_size[1], scale,
                (time.time() - t_resize) * 1000,
            )
        return img, original_size

    # --------------------------------------------- visual-test library helpers

    def _test_dir(self, owner: str, name: str) -> Path:
        if self._tests_dir is None:
            raise RuntimeError("visual-test library not initialized yet.")
        return self._tests_dir / _test_id_for(owner, name)

    def _test_json_path(self, owner: str, name: str) -> Path:
        return self._test_dir(owner, name) / "visual_test.json"

    def _load_test_record(self, owner: str, name: str) -> dict:
        path = self._test_json_path(owner, name)
        if not path.exists():
            raise ValueError(f"visual test {name!r} not found.")
        with open(path, "r") as f:
            return json.load(f)

    def _list_all_test_records(self) -> List[dict]:
        if self._tests_dir is None or not self._tests_dir.exists():
            return []
        out = []
        for child in sorted(self._tests_dir.iterdir()):
            if not child.is_dir():
                continue
            mj = child / "visual_test.json"
            if not mj.exists():
                continue
            try:
                with open(mj, "r") as f:
                    out.append(json.load(f))
            except Exception as e:
                logger.warning("Skipping corrupt visual test %s: %s", child.name, e)
        return out

    def _find_test_for_caller(self, name: str, caller_id: str) -> dict:
        """Return the most specific accessible record for a given test name.

        Resolution order:
          1. caller's own test (highest priority — your private one wins over
             a public one with the same name)
          2. any public test with that name
        """
        own = self._test_json_path(caller_id, name)
        if own.exists():
            with open(own, "r") as f:
                return json.load(f)
        for rec in self._list_all_test_records():
            if rec.get("name") == name and bool(rec.get("is_public")):
                return rec
        raise ValueError(f"visual test {name!r} not found or not accessible.")

    async def _save_example_images(
        self,
        test_dir: Path,
        side: str,                 # "positive" or "negative"
        image_refs: List[str],
        context=None,
    ) -> tuple[list[str], list[str]]:
        from PIL import Image  # noqa: F401

        side_dir = test_dir / side
        side_dir.mkdir(parents=True, exist_ok=True)
        rel_paths, source_urls = [], []
        for i, ref in enumerate(image_refs):
            url = await self._resolve_to_url(ref, context)
            img, _orig = await self._download_image(
                url,
                max_pixels=_EXAMPLE_MAX_PIXELS,
                max_long_side=_EXAMPLE_MAX_LONG_SIDE,
            )
            filename = f"{i:02d}.png"
            img.save(side_dir / filename, format="PNG")
            rel_paths.append(f"{side}/{filename}")
            source_urls.append(url)
        return rel_paths, source_urls

    _SYSTEM_PROMPT_DESCRIBE = (
        "You are a microscopy image analyst. Answer the user's question "
        "about the provided microscopy image, grounded strictly in what is "
        "visible. Be specific, do not invent details, and keep responses "
        "short."
    )
    _SYSTEM_PROMPT_VERDICT = (
        "You are a microscopy quality-control assistant. Your job is to "
        "decide whether a microscopy image meets a stated visual-test "
        "criterion. Base every judgement on visible evidence in the image. "
        "Possible verdicts are PASSED (the PASS condition is clearly met), "
        "FAILED (the FAIL condition applies), or UNSURE (the evidence is "
        "ambiguous or insufficient). Be precise, do not invent details, "
        "and keep responses short."
    )

    async def _run_vlm(
        self, image: "Image.Image", instruction: str, max_new_tokens: int
    ) -> tuple[str, int]:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self._SYSTEM_PROMPT_DESCRIBE}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ]},
        ]
        return await self._generate_with_images(messages, [image], max_new_tokens)

    async def _run_vlm_few_shot(
        self,
        new_image: "Image.Image",
        visual_test: dict,
        positive_images: List["Image.Image"],
        negative_images: List["Image.Image"],
        max_new_tokens: int,
    ) -> tuple[str, int]:
        """Few-shot verdict path; works with 0..N references on either side."""
        pass_text = (visual_test.get("pass_criterion") or "").strip()
        fail_text = (visual_test.get("fail_criterion") or "").strip()

        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text":
                f"Visual test: {visual_test['name']}\n"
                f"PASS condition: {pass_text or '(none specified)'}\n"
                f"FAIL condition: {fail_text or '(none specified)'}"},
        ]
        if positive_images:
            user_content.append({"type": "text", "text":
                "Reference images that PASS this criterion:"})
            for img in positive_images:
                user_content.append({"type": "image", "image": img})
        if negative_images:
            user_content.append({"type": "text", "text":
                "Reference images that FAIL this criterion:"})
            for img in negative_images:
                user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text":
            "Now evaluate this new image. Decide whether it PASSED, FAILED, "
            "or is UNSURE based on the conditions above (and the references "
            "if provided):"})
        user_content.append({"type": "image", "image": new_image})
        user_content.append({"type": "text", "text":
            "Reply on the first line with exactly `VERDICT: passed`, "
            "`VERDICT: failed`, or `VERDICT: unsure`. Use `unsure` only when "
            "the visible evidence is genuinely ambiguous or insufficient. "
            "Then on a second line write `REASON: ` followed by ONE short "
            "sentence grounded in the new image's visible content (not in "
            "the references)."})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self._SYSTEM_PROMPT_VERDICT}]},
            {"role": "user", "content": user_content},
        ]
        ordered_images = list(positive_images) + list(negative_images) + [new_image]
        return await self._generate_with_images(messages, ordered_images, max_new_tokens)

    async def _generate_with_images(
        self,
        messages: List[Dict[str, Any]],
        ordered_images: List["Image.Image"],
        max_new_tokens: int,
    ) -> tuple[str, int]:
        import torch

        def _generate():
            prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[prompt],
                images=ordered_images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
            in_len = int(inputs["input_ids"].shape[1])
            with torch.inference_mode():
                gen = self._engine.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            new_tokens = gen[:, in_len:]
            text = self._processor.batch_decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            return text, int(new_tokens.shape[1])

        return await asyncio.wait_for(
            asyncio.to_thread(_generate),
            timeout=_GENERATE_TIMEOUT_S,
        )

    @staticmethod
    def _parse_verdict(text: str) -> tuple[str, str, bool]:
        """Parse the ``VERDICT`` / ``REASON`` reply format.

        The third element is False when the model emitted no readable
        verdict line. Callers must not read that as a model-judged
        ``unsure``: an unparseable generation routes to ``unsure`` so a
        gate still holds rather than proceeding, but the two cases are
        different evidence. The raw generation is always returned
        separately as ``description``, so ``reason`` stays a single short
        sentence or empty.
        """
        m = re.search(r"VERDICT\s*:\s*(passed|failed|unsure)\b", text, flags=re.IGNORECASE)
        m2 = re.search(r"REASON\s*:\s*(.+?)(?:\n|$)", text, flags=re.IGNORECASE | re.DOTALL)
        reason = m2.group(1).strip() if m2 else ""
        if not m:
            return "unsure", reason, False
        return m.group(1).lower(), reason, True

    # ------------------------------------------------------------ job registry

    def _sweep_expired_jobs(self) -> None:
        """Drop finished jobs older than ``_INSPECT_JOBS_TTL_SEC``. Runs
        opportunistically on each new submission.
        """
        now = time.time()
        for run_id in [
            rid for rid, j in self._inspect_jobs.items()
            if j["completed_at"] is not None
            and (now - j["completed_at"]) > _INSPECT_JOBS_TTL_SEC
        ]:
            self._inspect_jobs.pop(run_id, None)

    def _new_job(self, owner: str) -> dict:
        self._sweep_expired_jobs()
        job = {
            "job_id": f"ij-{uuid.uuid4().hex[:12]}",
            "owner": owner,
            "state": "queued",
            "started_at": time.time(),
            "completed_at": None,
            "result": None,
            # Kept so `inspect` can re-raise what a caller would have seen
            # before the job split; `result["error"]` is the polled form.
            "exception": None,
            # Stage entry marks.
            "preprocess_ts": None,
            "running_ts": None,
            # Execution start (queue position #0 reached).
            "run_started_ts": None,
            "task": None,
        }
        self._inspect_jobs[job["job_id"]] = job
        return job

    def _run_queue_position(self, job: dict) -> Optional[int]:
        """0-based position in the GPU queue, or None when the job is not
        currently in the ``running`` stage.

        One ``_gpu_lock`` per replica, so this is a real FIFO queue: 0 = the
        job holding the lock and generating now, N = N jobs ahead of it.
        Rank by the *execution* signal (``run_started_ts``, stamped when the
        lock is acquired), not by raw entry order — an asyncio lock is not
        always granted in entry order, and the invariant that matters to a
        caller is that a start timestamp exists iff position == 0.
        """
        if job["state"] != "running":
            return None
        if job["run_started_ts"] is not None:
            return 0
        ts = job["running_ts"]
        ahead = 0
        for other in self._inspect_jobs.values():
            if other["state"] != "running":
                continue
            if other["run_started_ts"] is not None:
                ahead += 1  # generating now, ahead of us
            elif (
                other["running_ts"] is not None
                and ts is not None
                and other["running_ts"] < ts
            ):
                ahead += 1  # queued before us, also waiting for the lock
        return ahead

    def _job_progress(self, job: dict) -> dict:
        """Progress dict for an inspect job — a monotonic timeline bracketed
        by ``submitted_at`` / ``completed_at``.

        * ``state`` — ``queued``, ``preprocess``, ``running``, ``completed``
          or ``failed``.
        * ``result`` — the inspect result on success, ``{"error": str}`` on
          failure, else None.
        * ``stages`` — per-stage ``{start, end}`` map. ``preprocess`` (image
          fetch and decode) carries no ``queue_position``: it is pure I/O and
          runs concurrently. ``run`` does, and its ``start`` is the moment it
          reached position #0, not when it joined the queue.
        """
        run_start = job["run_started_ts"]
        if run_start is None and job["state"] in ("completed", "failed"):
            # Finished before any poll caught it at position #0; fall back to
            # its queue-entry mark so a terminal stage still reports a start.
            run_start = job["running_ts"]
        return {
            "state": job["state"],
            "submitted_at": job["started_at"],
            "completed_at": job["completed_at"],
            "result": job["result"],
            "stages": {
                "preprocess": {
                    "start": job["preprocess_ts"],
                    "end": job["running_ts"] or job["completed_at"],
                },
                "run": {
                    "start": run_start,
                    "end": job["completed_at"],
                    "queue_position": self._run_queue_position(job),
                },
            },
        }

    # ---------------------------------------------------------------- public API

    @bioengine.method
    async def ping(self) -> dict:
        """Liveness probe."""
        return {
            "status": "ok",
            "model": _MODEL_ID,
            "uptime_s": round(time.time() - self.start_time, 1),
        }

    @bioengine.method
    async def get_model_info(self) -> dict:
        """Describe the served model and the input/output contract."""
        return {
            "model": _MODEL_ID,
            "task": "vision-language",
            "engine": "huggingface-transformers",
            "dtype": "float16",
            "device": "cuda:0",
            "max_image_bytes": _MAX_IMAGE_BYTES,
            "max_instruction_chars": _MAX_INSTRUCTION_CHARS,
            "max_pixels": _MAX_PIXELS,
            "max_long_side": _MAX_LONG_SIDE,
            "hard_reject_pixels": _HARD_REJECT_PIXELS,
            "min_examples_per_class": _MIN_EXAMPLES_PER_CLASS,
            "max_examples_per_class": _MAX_EXAMPLES_PER_CLASS,
            "max_visual_test_name_chars": _MAX_TEST_NAME_CHARS,
            "max_visual_test_desc_chars": _MAX_TEST_DESC_CHARS,
            "verdicts": list(_VERDICT_VALUES),
            "license": "Qwen2.5-VL Apache 2.0 weights",
        }

    # ----------------------------------------------- visual-test management

    @bioengine.method(context=True)
    async def create_visual_test(
        self,
        name: str = Field(
            ...,
            description=(
                "Visual-test identifier. Lowercase letters, digits, and "
                "hyphens; max 50 chars; must start with a letter or digit. "
                "Two different users can each have a test with the same "
                "name without colliding. Re-using your own name overwrites."
            ),
        ),
        pass_criterion: str = Field(
            ...,
            description=(
                "Free-text description of what makes an image PASS this test "
                "(must hold for the verdict to be 'passed'). Max 800 chars."
            ),
        ),
        fail_criterion: str = Field(
            ...,
            description=(
                "Free-text description of what makes an image FAIL this test "
                "(must hold for the verdict to be 'failed'). Max 800 chars."
            ),
        ),
        positive_image_refs: list = Field(
            default_factory=list,
            description=(
                "Optional 0–5 image references that should PASS. Each entry "
                "can be an HTTPS URL or a Hypha artifact ref "
                "'<workspace>/<alias>:<path>', the latter resolvable only if "
                "public or readable by you. Omit for a text-only test."
            ),
        ),
        negative_image_refs: list = Field(
            default_factory=list,
            description=(
                "Optional 0–5 image references that should FAIL. Same "
                "accepted formats as positive_image_refs."
            ),
        ),
        is_public: bool = Field(
            False,
            description=(
                "When True, the test is visible to and usable by every "
                "user. When False (default) only the creator can list or "
                "use it. Delete is owner-only regardless."
            ),
        ),
        context=None,
    ) -> dict:
        """Define or replace one of your visual tests."""
        positive_image_refs = _arg(positive_image_refs, [])
        negative_image_refs = _arg(negative_image_refs, [])
        is_public = _arg(is_public, False)

        owner = _owner_from_context(context)
        if not _TEST_NAME_RE.match(name):
            raise ValueError(
                f"visual-test name must match {_TEST_NAME_RE.pattern} (got: {name!r})"
            )
        for label, value in (("pass_criterion", pass_criterion), ("fail_criterion", fail_criterion)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a non-empty string.")
            if len(value) > _MAX_TEST_DESC_CHARS:
                raise ValueError(
                    f"{label} exceeds {_MAX_TEST_DESC_CHARS}-char limit "
                    f"(got {len(value)})."
                )
        for side_name, refs in (
            ("positive", positive_image_refs),
            ("negative", negative_image_refs),
        ):
            if not isinstance(refs, list):
                raise ValueError(f"{side_name}_image_refs must be a list.")
            if len(refs) > _MAX_EXAMPLES_PER_CLASS:
                raise ValueError(
                    f"{side_name}_image_refs exceeds the "
                    f"{_MAX_EXAMPLES_PER_CLASS}-example cap (got {len(refs)})."
                )

        import shutil
        test_dir = self._test_dir(owner, name)
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            pos_paths, pos_urls = await self._save_example_images(
                test_dir, "positive", positive_image_refs, context,
            )
            neg_paths, neg_urls = await self._save_example_images(
                test_dir, "negative", negative_image_refs, context,
            )
        except Exception:
            shutil.rmtree(test_dir, ignore_errors=True)
            raise

        record = {
            "name": name,
            "pass_criterion": pass_criterion.strip(),
            "fail_criterion": fail_criterion.strip(),
            "positive_images": pos_paths,
            "negative_images": neg_paths,
            "positive_source_urls": pos_urls,
            "negative_source_urls": neg_urls,
            "n_positive": len(pos_paths),
            "n_negative": len(neg_paths),
            "is_public": bool(is_public),
            "created_by": owner,
            "created_at": time.time(),
        }
        with open(self._test_json_path(owner, name), "w") as f:
            json.dump(record, f, indent=2)
        logger.info(
            "Saved visual test %r (owner=%s, public=%s): %d pos / %d neg",
            name, owner, record["is_public"], record["n_positive"], record["n_negative"],
        )
        return record

    @bioengine.method(context=True)
    async def list_visual_tests(self, context=None) -> list:
        """List visual tests visible to the caller.

        Returns: the caller's own tests + every public test (regardless of
        owner). Each record carries `is_public`, `created_by`, and an
        `owned_by_you` boolean so the UI can branch on it without computing
        the comparison itself.
        """
        caller_id = _owner_from_context(context)
        out = []
        for rec in self._list_all_test_records():
            owner = rec.get("created_by", _ANON_OWNER)
            is_public = bool(rec.get("is_public"))
            if owner == caller_id or is_public:
                rec = dict(rec)
                rec["owned_by_you"] = (owner == caller_id)
                out.append(rec)
        return out

    @bioengine.method(context=True)
    async def get_visual_test(
        self,
        name: str = Field(..., description="Visual-test identifier."),
        context=None,
    ) -> dict:
        """Return one visual-test record visible to the caller."""
        caller_id = _owner_from_context(context)
        rec = dict(self._find_test_for_caller(name, caller_id))
        rec["owned_by_you"] = (rec.get("created_by") == caller_id)
        return rec

    @bioengine.method(context=True)
    async def delete_visual_test(
        self,
        name: str = Field(..., description="Visual-test identifier."),
        context=None,
    ) -> dict:
        """Delete one of YOUR visual tests. Refuses to delete another user's."""
        import shutil
        caller_id = _owner_from_context(context)
        if not _TEST_NAME_RE.match(name):
            raise ValueError(
                f"visual-test name must match {_TEST_NAME_RE.pattern} (got: {name!r})"
            )
        test_dir = self._test_dir(caller_id, name)
        if not test_dir.exists():
            # The name may exist as another user's test, but the caller has
            # no delete permission on it — message accordingly.
            other = any(
                rec.get("name") == name and rec.get("created_by") != caller_id
                for rec in self._list_all_test_records()
            )
            if other:
                raise PermissionError(
                    f"visual test {name!r} is owned by another user; only its "
                    f"creator can delete it."
                )
            raise ValueError(f"visual test {name!r} not found.")
        shutil.rmtree(test_dir)
        logger.info("Deleted visual test %r (owner=%s)", name, caller_id)
        return {"name": name, "deleted": True}
    # ------------------------------------------------------------ inspect path

    @staticmethod
    def _validate_inspect_args(
        instruction: Optional[str],
        visual_test_name: Optional[str],
        max_new_tokens: int,
    ) -> tuple[Optional[str], Optional[str], int]:
        """Normalise and check the caller-facing inspect arguments.

        Runs at submission time so a bad request fails on the submitting call
        rather than surfacing seconds later in a status poll.
        """
        instruction = _arg(instruction, None)
        visual_test_name = _arg(visual_test_name, None)
        max_new_tokens = _arg(max_new_tokens, 512)

        if not visual_test_name and not (isinstance(instruction, str) and instruction.strip()):
            raise ValueError(
                "Either `visual_test_name` or `instruction` must be provided."
            )
        if isinstance(instruction, str) and len(instruction) > _MAX_INSTRUCTION_CHARS:
            raise ValueError(
                f"instruction exceeds {_MAX_INSTRUCTION_CHARS}-char limit "
                f"(got {len(instruction)})."
            )
        return instruction, visual_test_name, max_new_tokens

    async def _execute_inspect(
        self,
        job: dict,
        image_ref: str,
        instruction: Optional[str],
        visual_test: Optional[dict],
        max_new_tokens: int,
        context=None,
    ) -> None:
        """Run one submitted inspection, recording its timeline on ``job``.

        Never raises: a failure is recorded as ``{"error": …}`` on the job and
        re-raised by ``inspect`` for callers that awaited it. A background
        submission has nobody to receive an exception, and an unretrieved task
        exception would only show up as noise in the replica log.
        """
        t0 = job["started_at"]
        try:
            job["state"] = "preprocess"
            job["preprocess_ts"] = time.time()
            url = await self._resolve_to_url(image_ref, context)
            image, original_size = await self._download_image(url)

            if visual_test is not None:
                from PIL import Image
                owner = visual_test.get("created_by", _ANON_OWNER)
                test_dir = self._test_dir(owner, visual_test["name"])
                pos_imgs = [Image.open(test_dir / p).convert("RGB")
                            for p in visual_test.get("positive_images", [])]
                neg_imgs = [Image.open(test_dir / p).convert("RGB")
                            for p in visual_test.get("negative_images", [])]

            job["state"] = "running"
            job["running_ts"] = time.time()
            async with self._gpu_lock:
                job["run_started_ts"] = time.time()
                t_gen0 = time.time()
                if visual_test is not None:
                    raw, n_tokens = await self._run_vlm_few_shot(
                        new_image=image,
                        visual_test=visual_test,
                        positive_images=pos_imgs,
                        negative_images=neg_imgs,
                        max_new_tokens=max_new_tokens,
                    )
                else:
                    raw, n_tokens = await self._run_vlm(image, instruction, max_new_tokens)
                gen_dt = time.time() - t_gen0

            if visual_test is not None:
                verdict, reason, verdict_parsed = self._parse_verdict(raw)
                result = {
                    "mode": "few-shot",
                    "visual_test_name": visual_test["name"],
                    "pass_criterion": visual_test.get("pass_criterion", ""),
                    "fail_criterion": visual_test.get("fail_criterion", ""),
                    "verdict": verdict,
                    "verdict_parsed": verdict_parsed,
                    "reason": reason,
                    "description": raw,
                    "n_positive_examples": visual_test.get("n_positive", 0),
                    "n_negative_examples": visual_test.get("n_negative", 0),
                }
            else:
                result = {"mode": "describe", "description": raw}

            result.update({
                "image_size": list(image.size),
                "source_url": url,
                "model": _MODEL_ID,
                "tokens_generated": n_tokens,
                "generation_time_s": round(gen_dt, 2),
                "tokens_per_second": round(n_tokens / gen_dt, 2) if gen_dt > 0 else None,
                "processing_time_s": round(time.time() - t0, 2),
                "run_id": job["job_id"],
            })
            if original_size is not None:
                result["downscaled_from"] = list(original_size)
                result["downscale_note"] = (
                    f"Image downscaled from {original_size[0]}x{original_size[1]} "
                    f"to {image.size[0]}x{image.size[1]} before VLM."
                )
            job["result"] = result
            job["state"] = "completed"
        except Exception as exc:
            job["exception"] = exc
            job["result"] = {"error": f"{type(exc).__name__}: {exc}"}
            job["state"] = "failed"
            logger.exception("Inspect job %s failed", job["job_id"])
        finally:
            job["completed_at"] = time.time()

    def _submit_inspect(
        self,
        image_ref: str,
        instruction: Optional[str],
        visual_test_name: Optional[str],
        max_new_tokens: int,
        context,
    ) -> dict:
        """Validate, register and start one inspect job. Returns the job."""
        instruction, visual_test_name, max_new_tokens = self._validate_inspect_args(
            instruction, visual_test_name, max_new_tokens
        )
        caller_id = _owner_from_context(context)
        visual_test = (
            self._find_test_for_caller(visual_test_name, caller_id)
            if visual_test_name else None
        )
        job = self._new_job(caller_id)
        job["task"] = asyncio.create_task(
            self._execute_inspect(
                job, image_ref, instruction, visual_test, max_new_tokens, context,
            )
        )
        return job

    @bioengine.method(context=True)
    async def submit_inspect(
        self,
        image_ref: str = Field(
            ...,
            description=(
                "Image to inspect. HTTPS URL (public or presigned) or "
                "Hypha artifact reference '<workspace>/<alias>:<path>'. An "
                "artifact ref resolves only if the artifact is public or "
                "you have read access to that workspace."
            ),
        ),
        instruction: Optional[str] = Field(
            None,
            description=(
                "Free-text instruction for describe mode. Required if "
                "`visual_test_name` is not given. Max 4000 chars."
            ),
        ),
        visual_test_name: Optional[str] = Field(
            None,
            description=(
                "Name of a visual test created via create_visual_test. The "
                "caller must either own the test or the test must be public."
            ),
        ),
        max_new_tokens: int = Field(
            512,
            description="Maximum response tokens (1-1024).",
            ge=1, le=1024,
        ),
        context=None,
    ) -> str:
        """
        Schedule an inspection and return a run id immediately.

        The inspection runs as a background job. This call returns right away
        with just the ``run_id`` string; poll ``get_inspect_status(run_id)``
        for the progress dict and, once the job finishes, the full result in
        ``result``::

            "ij-…"  # the returned run_id
            # then poll get_inspect_status(run_id) →
            # {"state": "running", "submitted_at": 1735689590.0,
            #  "completed_at": None, "result": None,
            #  "stages": {"preprocess": {...},
            #             "run": {"start": None, "end": None,
            #                     "queue_position": 2}}}

        Bad arguments and an unknown or inaccessible ``visual_test_name``
        raise here, not in the poll.

        Use ``inspect`` instead when you are happy to hold the connection open
        for the whole run — it submits and awaits with the same arguments.
        """
        return self._submit_inspect(
            image_ref, instruction, visual_test_name, max_new_tokens, context
        )["job_id"]

    @bioengine.method(context=True)
    async def get_inspect_status(
        self,
        run_id: str = Field(..., description="Id returned by `submit_inspect` ('ij-…')."),
        context=None,
    ) -> dict:
        """
        Progress and result for one submitted inspection.

        ``state`` moves ``queued`` → ``preprocess`` → ``running`` →
        ``completed`` / ``failed``. While a stage is queued its ``start`` is
        None and ``queue_position`` says how many jobs are ahead in that
        stage: 0 = generating now, N = N jobs ahead. Generation is serialised
        on one GPU, so exactly one job reports 0.

        Jobs are held for 24 hours after completion, then dropped. The
        registry is per replica and in-memory — a job submitted to one replica
        is unknown to the others, and a replica restart drops everything.

        In verdict mode the completed ``result`` carries ``verdict``
        (``passed`` / ``failed`` / ``unsure``) alongside ``verdict_parsed``.
        A False ``verdict_parsed`` means the model emitted no readable
        verdict line and the ``unsure`` is the safe fallback, not a judgement
        — check it before treating ``unsure`` as considered evidence. The
        full generation is always in ``description``; ``reason`` is the
        model's one-sentence reason, or empty if it emitted none.
        """
        job = self._inspect_jobs.get(run_id)
        if job is None:
            raise KeyError(
                f"Unknown run_id {run_id!r}. Jobs live in-memory per replica "
                f"and expire 24 hours after completion. Start a fresh run via "
                f"submit_inspect()."
            )
        caller_id = _owner_from_context(context)
        if job["owner"] != caller_id:
            # A result can carry the criteria of a private visual test, so a
            # job is only readable by the identity that submitted it.
            raise PermissionError(f"Run {run_id!r} belongs to a different caller.")
        return self._job_progress(job)

    @bioengine.method(context=True)
    async def inspect(
        self,
        image_ref: str = Field(
            ...,
            description=(
                "Image to inspect. HTTPS URL (public or presigned) or "
                "Hypha artifact reference '<workspace>/<alias>:<path>'. An "
                "artifact ref resolves only if the artifact is public or "
                "you have read access to that workspace."
            ),
        ),
        instruction: Optional[str] = Field(
            None,
            description=(
                "Free-text instruction for describe mode. Required if "
                "`visual_test_name` is not given. Max 4000 chars."
            ),
        ),
        visual_test_name: Optional[str] = Field(
            None,
            description=(
                "Name of a visual test created via create_visual_test. The "
                "caller must either own the test or the test must be public."
            ),
        ),
        max_new_tokens: int = Field(
            512,
            description="Maximum response tokens (1-1024).",
            ge=1, le=1024,
        ),
        context=None,
    ) -> dict:
        """Inspect a microscopy image and return a QC judgement.

        Submits the same job as ``submit_inspect`` and waits for it, so the
        connection stays open for the whole run — seconds to minutes when
        others are queued ahead. Prefer ``submit_inspect`` +
        ``get_inspect_status`` for anything that should see its queue
        position, and for callers that would otherwise time out.
        """
        job = self._submit_inspect(
            image_ref, instruction, visual_test_name, max_new_tokens, context
        )
        await job["task"]
        if job.get("exception") is not None:
            raise job["exception"]
        return job["result"]
