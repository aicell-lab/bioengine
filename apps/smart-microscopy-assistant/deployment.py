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
from pathlib import Path
from typing import Any, Dict, List, Optional

import bioengine
from pydantic import Field

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


def _owner_from_context(
    context: Optional[Dict[str, Any]],
    caller_user_id: Optional[str] = None,
) -> str:
    """Resolve the caller's stable user id.

    Today the BioEngine proxy consumes Hypha's `require_context` payload at
    its outer wrapper and does NOT forward it to the deployment method, so
    `context` here is almost always None. As a pragmatic fix we let the
    client also pass `caller_user_id` (the workspace name doubles as a
    stable Hypha identity, e.g. 'ws-user-github|49943582'). Resolution
    order: explicit caller_user_id -> context -> "anonymous".
    """
    if isinstance(caller_user_id, str) and caller_user_id.strip():
        return caller_user_id.strip()
    if isinstance(context, dict) and isinstance(context.get("user"), dict):
        uid = context["user"].get("id")
        if isinstance(uid, str) and uid:
            return uid
    return _ANON_OWNER


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
    num_gpus=1,
    memory_mb=12 * 1024,
    pip=_read_pip("requirements-deployment.txt"),
    env_vars={
        # Triton's JIT cache wants a writable dir; runtime_env venv's
        # default $HOME is read-only on this Ray pod.
        "TRITON_CACHE_DIR": "/tmp/triton-cache",
        "HF_HOME": "/tmp/hf-home",
        "XDG_CACHE_HOME": "/tmp/xdg-cache",
    },
    max_ongoing_requests=4,
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
        self._tests_dir: Optional[Path] = None

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

    async def _ensure_artifact_manager(self) -> Any:
        if self._artifact_manager is not None:
            try:
                await self._artifact_manager.list(parent_id="public/applications")
                return self._artifact_manager
            except Exception as e:
                msg = str(e)
                if "Connection is closed" not in msg and "WebSocket" not in msg:
                    raise
                logger.warning("Hypha WS appears stale (%s); reconnecting", msg[:120])

        from hypha_rpc import connect_to_server
        token = os.environ.get("HYPHA_TOKEN")
        if not token:
            raise RuntimeError("HYPHA_TOKEN environment variable is not set.")
        try:
            if self._server and hasattr(self._server, "disconnect"):
                await self._server.disconnect()
        except Exception:
            pass
        self._server = await connect_to_server({
            "server_url": _DEFAULT_SERVER_URL,
            "token": token,
        })
        self._artifact_manager = await self._server.get_service("public/artifact-manager")
        logger.info("Re-connected Hypha artifact-manager.")
        return self._artifact_manager

    async def _resolve_to_url(self, image_ref: str) -> str:
        if image_ref.startswith(("http://", "https://")):
            return image_ref
        if ":" not in image_ref or "/" not in image_ref.split(":", 1)[0]:
            raise ValueError(
                "image_ref must be 'https://...' or '<workspace>/<alias>:<path>' "
                f"(got: {image_ref!r})"
            )
        artifact_id, file_path = image_ref.split(":", 1)
        am = await self._ensure_artifact_manager()
        url = await am.get_file(artifact_id=artifact_id, file_path=file_path)
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
    ) -> tuple[list[str], list[str]]:
        from PIL import Image  # noqa: F401

        side_dir = test_dir / side
        side_dir.mkdir(parents=True, exist_ok=True)
        rel_paths, source_urls = [], []
        for i, ref in enumerate(image_refs):
            url = await self._resolve_to_url(ref)
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
    def _parse_verdict(text: str) -> tuple[str, str]:
        verdict = "unsure"
        reason = text.strip()
        m = re.search(r"VERDICT\s*:\s*(passed|failed|unsure)\b", text, flags=re.IGNORECASE)
        if m:
            verdict = m.group(1).lower()
        m2 = re.search(r"REASON\s*:\s*(.+?)(?:\n|$)", text, flags=re.IGNORECASE | re.DOTALL)
        if m2:
            reason = m2.group(1).strip()
        return verdict, reason

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

    @bioengine.method
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
                "'<workspace>/<alias>:<path>'. Omit for a text-only test."
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
        caller_user_id: Optional[str] = Field(
            None,
            description=(
                "Stable Hypha user / workspace id of the caller. The "
                "BioEngine proxy currently does not forward Hypha's auth "
                "context to deployment methods, so the client must pass "
                "this explicitly to participate in ownership/visibility. "
                "Falls back to context.user.id then 'anonymous'."
            ),
        ),
        context: Optional[Dict[str, Any]] = Field(
            None,
            description="Authentication context, automatically provided by Hypha.",
        ),
    ) -> dict:
        """Define or replace one of your visual tests."""
        owner = _owner_from_context(context, caller_user_id)
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
                test_dir, "positive", positive_image_refs,
            )
            neg_paths, neg_urls = await self._save_example_images(
                test_dir, "negative", negative_image_refs,
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

    @bioengine.method
    async def list_visual_tests(
        self,
        caller_user_id: Optional[str] = Field(
            None,
            description="Stable Hypha user / workspace id of the caller.",
        ),
        context: Optional[Dict[str, Any]] = Field(
            None,
            description="Authentication context, automatically provided by Hypha.",
        ),
    ) -> list:
        """List visual tests visible to the caller.

        Returns: the caller's own tests + every public test (regardless of
        owner). Each record carries `is_public`, `created_by`, and an
        `owned_by_you` boolean so the UI can branch on it without computing
        the comparison itself.
        """
        caller_id = _owner_from_context(context, caller_user_id)
        out = []
        for rec in self._list_all_test_records():
            owner = rec.get("created_by", _ANON_OWNER)
            is_public = bool(rec.get("is_public"))
            if owner == caller_id or is_public:
                rec = dict(rec)
                rec["owned_by_you"] = (owner == caller_id)
                out.append(rec)
        return out

    @bioengine.method
    async def get_visual_test(
        self,
        name: str = Field(..., description="Visual-test identifier."),
        caller_user_id: Optional[str] = Field(
            None,
            description="Stable Hypha user / workspace id of the caller.",
        ),
        context: Optional[Dict[str, Any]] = Field(
            None,
            description="Authentication context, automatically provided by Hypha.",
        ),
    ) -> dict:
        """Return one visual-test record visible to the caller."""
        caller_id = _owner_from_context(context, caller_user_id)
        rec = dict(self._find_test_for_caller(name, caller_id))
        rec["owned_by_you"] = (rec.get("created_by") == caller_id)
        return rec

    @bioengine.method
    async def delete_visual_test(
        self,
        name: str = Field(..., description="Visual-test identifier."),
        caller_user_id: Optional[str] = Field(
            None,
            description="Stable Hypha user / workspace id of the caller.",
        ),
        context: Optional[Dict[str, Any]] = Field(
            None,
            description="Authentication context, automatically provided by Hypha.",
        ),
    ) -> dict:
        """Delete one of YOUR visual tests. Refuses to delete another user's."""
        import shutil
        caller_id = _owner_from_context(context, caller_user_id)
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

    @bioengine.method
    async def inspect(
        self,
        image_ref: str = Field(
            ...,
            description=(
                "Image to inspect. HTTPS URL (public or presigned) or "
                "Hypha artifact reference '<workspace>/<alias>:<path>'."
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
        caller_user_id: Optional[str] = Field(
            None,
            description="Stable Hypha user / workspace id of the caller.",
        ),
        context: Optional[Dict[str, Any]] = Field(
            None,
            description="Authentication context, automatically provided by Hypha.",
        ),
    ) -> dict:
        """Inspect a microscopy image and return a QC judgement."""
        t0 = time.time()

        if not visual_test_name and not (isinstance(instruction, str) and instruction.strip()):
            raise ValueError(
                "Either `visual_test_name` or `instruction` must be provided."
            )
        if isinstance(instruction, str):
            if len(instruction) > _MAX_INSTRUCTION_CHARS:
                raise ValueError(
                    f"instruction exceeds {_MAX_INSTRUCTION_CHARS}-char limit "
                    f"(got {len(instruction)})."
                )

        url = await self._resolve_to_url(image_ref)
        image, original_size = await self._download_image(url)

        if visual_test_name:
            from PIL import Image
            caller_id = _owner_from_context(context, caller_user_id)
            visual_test = self._find_test_for_caller(visual_test_name, caller_id)
            owner = visual_test.get("created_by", _ANON_OWNER)
            test_dir = self._test_dir(owner, visual_test_name)
            pos_imgs = [Image.open(test_dir / p).convert("RGB") for p in visual_test.get("positive_images", [])]
            neg_imgs = [Image.open(test_dir / p).convert("RGB") for p in visual_test.get("negative_images", [])]

            t_gen0 = time.time()
            raw, n_tokens = await self._run_vlm_few_shot(
                new_image=image,
                visual_test=visual_test,
                positive_images=pos_imgs,
                negative_images=neg_imgs,
                max_new_tokens=max_new_tokens,
            )
            gen_dt = time.time() - t_gen0
            verdict, reason = self._parse_verdict(raw)
            result = {
                "mode": "few-shot",
                "visual_test_name": visual_test_name,
                "pass_criterion": visual_test.get("pass_criterion", ""),
                "fail_criterion": visual_test.get("fail_criterion", ""),
                "verdict": verdict,
                "reason": reason,
                "description": raw,
                "n_positive_examples": visual_test.get("n_positive", 0),
                "n_negative_examples": visual_test.get("n_negative", 0),
                "image_size": list(image.size),
                "source_url": url,
                "model": _MODEL_ID,
                "tokens_generated": n_tokens,
                "generation_time_s": round(gen_dt, 2),
                "tokens_per_second": round(n_tokens / gen_dt, 2) if gen_dt > 0 else None,
                "processing_time_s": round(time.time() - t0, 2),
            }
        else:
            t_gen0 = time.time()
            description, n_tokens = await self._run_vlm(image, instruction, max_new_tokens)
            gen_dt = time.time() - t_gen0
            result = {
                "mode": "describe",
                "description": description,
                "image_size": list(image.size),
                "source_url": url,
                "model": _MODEL_ID,
                "tokens_generated": n_tokens,
                "generation_time_s": round(gen_dt, 2),
                "tokens_per_second": round(n_tokens / gen_dt, 2) if gen_dt > 0 else None,
                "processing_time_s": round(time.time() - t0, 2),
            }

        if original_size is not None:
            result["downscaled_from"] = list(original_size)
            result["downscale_note"] = (
                f"Image downscaled from {original_size[0]}x{original_size[1]} "
                f"to {image.size[0]}x{image.size[1]} before VLM."
            )
        return result
