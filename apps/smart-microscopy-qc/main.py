"""Smart Microscopy QC — VLM-backed real-time quality-control inspector.

Accepts a microscopy image (Hypha artifact reference OR HTTPS URL) and either
a free-text QC instruction (describe-what-you-see mode) or the name of a
previously-defined QC metric (few-shot good/bad-verdict mode), and returns
the VLM's textual judgement.

Few-shot mode persists a small library of "metrics" on the replica's
local workspace under $HOME/metrics. Each metric records a name, a
free-text criterion, and a set of small reference images (up to 5 good
+ 5 bad) downloaded once at create_metric() time. inspect() then builds
a multi-image prompt prepending those reference images and asks the VLM
to classify the new image as GOOD or BAD against the criterion.

Backed by Qwen2.5-VL-3B-Instruct via HuggingFace transformers on a single
NVIDIA A40-16C vGPU slice (Ampere, sm_86; 16 GB framebuffer time-shared
with co-tenants on the host A40). ~6 GB FP16 weights + Qwen's vision
encoder + KV cache fit comfortably. The 7B-AWQ stack was tried first but
no viable AWQ kernel path serves Qwen2.5-VL on this cluster (vLLM V0
multimodal regression / inspector bug / autoawq-kernels lm_head dtype
mismatch); 3B FP16 lands cleanly without quantisation.
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from ray import serve

logger = logging.getLogger("ray.serve")

_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
_DEFAULT_SERVER_URL = "https://hypha.aicell.io"

_MAX_IMAGE_BYTES = 25 * 1024 * 1024      # 25 MB
_MAX_INSTRUCTION_CHARS = 4000
# Downscale targets: Qwen's native processor cap of 1280 * 28 * 28 pixels
# (≈ 1003520, about 1000×1000) is the largest tile the model will accept
# without internal downsampling. Plus a longest-side guard so a 1×10000
# strip is still reduced.
_MAX_PIXELS = 1280 * 28 * 28
_MAX_LONG_SIDE = 2048
# Hard reject above ~14000² (200 megapixel). At that scale we'd be holding
# 0.5+ GB of decoded pixels in RAM just to throw them away.
_HARD_REJECT_PIXELS = 200 * 1024 * 1024
_DOWNLOAD_TIMEOUT_S = 30
_GENERATE_TIMEOUT_S = 180

# Few-shot metric library.
# Reference images are aggressively downscaled before storage so the prompt
# stays within Qwen's working-set even when N is at the upper bound.
# 512x512 -> ~64 vision tokens after Qwen's 28-pixel patch + 2x2 merge,
# so 5+5 references add ~640 vision tokens on top of the new image.
_EXAMPLE_MAX_PIXELS = 512 * 512
_EXAMPLE_MAX_LONG_SIDE = 768
_MAX_EXAMPLES_PER_CLASS = 5
_MIN_EXAMPLES_PER_CLASS = 1
_MAX_METRIC_NAME_CHARS = 50
_MAX_METRIC_DESC_CHARS = 800
_METRIC_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,49}$")


def _resolve_metrics_dir() -> Path:
    """Pick a writable directory for the metric library at actor startup.

    The Ray runtime_env venv often boots with HOME=/nonexistent (no home
    for the service account); BioEngine injects the real per-deployment HOME
    after the process is up. Compute lazily so we never bake the
    module-import-time value into module state.
    """
    home = os.environ.get("HOME", "")
    candidate = Path(home) / "metrics" if home and home != "/nonexistent" else Path("/tmp") / "smart-microscopy-qc" / "metrics"
    return candidate


@serve.deployment(
    ray_actor_options={
        "num_cpus": 4,
        "num_gpus": 1,
        "memory": 12 * 1024**3,
        "runtime_env": {
            "pip": [
                # Frozen versions. Any change forces a 5-15 min env rebuild.
                # ray pinned to host: KTH Ray pod runs 2.55.1.
                "ray[serve]==2.55.1",
                # vLLM and AutoAWQ were both ruled out for serving
                # Qwen2.5-VL-7B on this cluster (see module docstring).
                # 3B FP16 uses the plain transformers loader and runs
                # without any quantisation kernel.
                "transformers==4.51.3",
                "accelerate==1.6.0",
                "torch==2.5.1",
                "torchvision==0.20.1",
                # numpy 1.26 keeps ABI in sync with the host Ray pod's pandas.
                "numpy==1.26.4",
                "pillow==10.4.0",
                "httpx==0.27.2",
                "hypha-rpc==0.20.54",
            ],
            "env_vars": {
                # Triton's JIT cache wants a writable dir; runtime_env venv's
                # default $HOME is read-only on this Ray pod.
                "TRITON_CACHE_DIR": "/tmp/triton-cache",
                "HF_HOME": "/tmp/hf-home",
                "XDG_CACHE_HOME": "/tmp/xdg-cache",
                "VLLM_LOGGING_LEVEL": "INFO",
            },
        },
    },
    max_ongoing_requests=4,
    health_check_period_s=30.0,
    health_check_timeout_s=600.0,
    graceful_shutdown_timeout_s=120.0,
)
class SmartMicroscopyQC:
    def __init__(self) -> None:
        self.start_time = time.time()
        self._engine = None
        self._processor = None
        self._server = None
        self._artifact_manager = None
        self._metrics_dir: Optional[Path] = None

    async def async_init(self) -> None:
        """Load the VLM and connect to Hypha for artifact resolution."""
        import os as _os
        for d in ("/tmp/triton-cache", "/tmp/hf-home", "/tmp/xdg-cache"):
            _os.makedirs(d, exist_ok=True)
        self._metrics_dir = _resolve_metrics_dir()
        self._metrics_dir.mkdir(parents=True, exist_ok=True)
        existing = self._list_metric_records()
        logger.info(
            "Metric library at %s (%d metric%s)",
            self._metrics_dir, len(existing), "" if len(existing) == 1 else "s",
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
        logger.info("Qwen2.5-VL-7B-AWQ ready on %s.", next(self._engine.parameters()).device)

    async def test_deployment(self) -> None:
        """No-op smoke test.

        The real first-request smoke happens lazily on the first inspect()
        call. Keeping this trivial avoids two known frustrations during
        BioEngine startup: (1) Qwen's processor rejects synthetic tiles
        below min_pixels, and (2) vLLM's async generate path is sensitive
        to the nested asyncio loop the BioEngine wrapper uses to launch
        test_deployment.
        """
        return None

    async def check_health(self) -> None:
        if self._engine is None or self._processor is None:
            raise RuntimeError("VLM not initialized.")
        if self._artifact_manager is None:
            raise RuntimeError("Hypha artifact-manager not connected.")

    # ---------------------------------------------------------------- helpers

    async def _resolve_to_url(self, image_ref: str) -> str:
        """Map either an HTTPS URL or '<workspace>/<alias>:<path>' to a fetchable URL."""
        if image_ref.startswith(("http://", "https://")):
            return image_ref
        if ":" not in image_ref or "/" not in image_ref.split(":", 1)[0]:
            raise ValueError(
                "image_ref must be 'https://...' or '<workspace>/<alias>:<path>' "
                f"(got: {image_ref!r})"
            )
        artifact_id, file_path = image_ref.split(":", 1)
        url = await self._artifact_manager.get_file(
            artifact_id=artifact_id,
            file_path=file_path,
        )
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
        """Stream-download up to _MAX_IMAGE_BYTES, decode to RGB PIL image,
        and downscale to ≤ max_pixels / max_long_side.

        Returns (image, original_size_or_none). original_size is None when no
        downscale was applied; otherwise it carries the pre-downscale (w, h)
        so callers can surface it.
        """
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

    # ------------------------------------------------ metric library helpers

    def _metric_dir(self, name: str) -> Path:
        if self._metrics_dir is None:
            raise RuntimeError("metric library not initialized yet.")
        return self._metrics_dir / name

    def _metric_json_path(self, name: str) -> Path:
        return self._metric_dir(name) / "metric.json"

    def _load_metric(self, name: str) -> dict:
        if not _METRIC_NAME_RE.match(name):
            raise ValueError(
                f"metric name must match {_METRIC_NAME_RE.pattern} "
                f"(got: {name!r})"
            )
        path = self._metric_json_path(name)
        if not path.exists():
            raise ValueError(f"metric {name!r} not found.")
        with open(path, "r") as f:
            return json.load(f)

    def _list_metric_records(self) -> List[dict]:
        if self._metrics_dir is None or not self._metrics_dir.exists():
            return []
        out = []
        for child in sorted(self._metrics_dir.iterdir()):
            if not child.is_dir():
                continue
            mj = child / "metric.json"
            if not mj.exists():
                continue
            try:
                with open(mj, "r") as f:
                    out.append(json.load(f))
            except Exception as e:
                logger.warning("Skipping corrupt metric %s: %s", child.name, e)
        return out

    async def _save_example_images(
        self,
        metric_dir: Path,
        side: str,                 # "good" or "bad"
        image_refs: List[str],
    ) -> tuple[list[str], list[str]]:
        """Download each ref, downscale tight, write PNG to disk.

        Returns (list of relative paths under metric_dir, list of source urls).
        """
        from PIL import Image  # noqa: F401  (touched here only to surface ImportError early)

        side_dir = metric_dir / side
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

    async def _run_vlm(
        self, image: "Image.Image", instruction: str, max_new_tokens: int
    ) -> tuple[str, int]:
        """Free-text describe path: single image + free-text instruction."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }]
        return await self._generate_with_images(messages, [image], max_new_tokens)

    async def _run_vlm_few_shot(
        self,
        new_image: "Image.Image",
        metric: dict,
        good_images: List["Image.Image"],
        bad_images: List["Image.Image"],
        instruction_override: Optional[str],
        max_new_tokens: int,
    ) -> tuple[str, int]:
        """Few-shot verdict path: prepend good/bad examples, ask for GOOD/BAD."""
        criterion = (instruction_override or metric["description"]).strip()
        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text":
                f"You are a microscopy QC assistant. Your task is to decide whether a new "
                f"image meets the following quality-control criterion.\n\n"
                f"Metric: {metric['name']}\n"
                f"Criterion: {criterion}\n\n"
                f"Reference images that DO meet the criterion (GOOD):"},
        ]
        for img in good_images:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text":
            "Reference images that DO NOT meet the criterion (BAD):"})
        for img in bad_images:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text":
            "Now evaluate this new image against the same criterion:"})
        user_content.append({"type": "image", "image": new_image})
        user_content.append({"type": "text", "text":
            "Reply on the first line with exactly `VERDICT: good` or `VERDICT: bad`, "
            "then on a second line `REASON: ` followed by ONE short sentence "
            "explaining your decision in terms of the criterion."})

        messages = [{"role": "user", "content": user_content}]
        ordered_images = list(good_images) + list(bad_images) + [new_image]
        return await self._generate_with_images(messages, ordered_images, max_new_tokens)

    async def _generate_with_images(
        self,
        messages: List[Dict[str, Any]],
        ordered_images: List["Image.Image"],
        max_new_tokens: int,
    ) -> tuple[str, int]:
        """Run the model on a chat-templated prompt with N images. The order in
        `ordered_images` MUST match the order of <|image_pad|> placeholders the
        chat template emits — which is the document order of `{"type":"image"}`
        entries inside `messages`.
        """
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
    def _parse_verdict(text: str) -> tuple[Optional[str], str]:
        """Parse 'VERDICT: good/bad' and 'REASON: ...' out of the model output.
        Returns (verdict, reason). verdict is None when the output didn't follow
        the schema closely enough."""
        verdict: Optional[str] = None
        reason = text.strip()
        m = re.search(r"VERDICT\s*:\s*(good|bad)\b", text, flags=re.IGNORECASE)
        if m:
            verdict = m.group(1).lower()
        m2 = re.search(r"REASON\s*:\s*(.+?)(?:\n|$)", text, flags=re.IGNORECASE | re.DOTALL)
        if m2:
            reason = m2.group(1).strip()
        return verdict, reason

    # ---------------------------------------------------------------- public API

    @schema_method
    async def ping(self) -> dict:
        """Liveness probe."""
        return {
            "status": "ok",
            "model": _MODEL_ID,
            "uptime_s": round(time.time() - self.start_time, 1),
        }

    @schema_method
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
            "max_metric_desc_chars": _MAX_METRIC_DESC_CHARS,
            "license": "Qwen2.5-VL Apache 2.0 weights",
        }

    # ----------------------------------------------------- metric management

    @schema_method
    async def create_metric(
        self,
        name: str = Field(
            ...,
            description=(
                "Metric identifier. Lowercase letters, digits, and hyphens; "
                "max 50 chars; must start with a letter or digit. Example: "
                "'focus-quality'. Re-using an existing name overwrites it."
            ),
        ),
        description: str = Field(
            ...,
            description=(
                "Free-text criterion describing what makes an image GOOD or "
                "BAD for this metric. Max 800 chars. Example: 'Sharp cell "
                "outlines, no motion blur, distinct staining patterns'."
            ),
        ),
        good_image_refs: list = Field(
            ...,
            description=(
                "List of 1–5 image references that exemplify the criterion. "
                "Each entry can be an HTTPS URL (public or presigned) or a "
                "Hypha artifact reference '<workspace>/<alias>:<file_path>'. "
                "Private artifacts must be presigned ahead of time or "
                "fetched server-side via artifact-manager."
            ),
        ),
        bad_image_refs: list = Field(
            ...,
            description=(
                "List of 1–5 image references that VIOLATE the criterion. "
                "Same accepted formats as good_image_refs."
            ),
        ),
    ) -> dict:
        """Define or replace a QC metric in the on-replica metric library.

        Downloads each reference image, downscales it to a small fixed budget
        (so the few-shot prompt stays within the model's working context),
        and persists the metric record on the replica's local disk under
        $HOME/metrics/<name>/. Subsequent inspect() calls can then reference
        the metric by name.

        Returns the saved metric record (without raw image bytes).
        """
        if not _METRIC_NAME_RE.match(name):
            raise ValueError(
                f"metric name must match {_METRIC_NAME_RE.pattern} (got: {name!r})"
            )
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string.")
        if len(description) > _MAX_METRIC_DESC_CHARS:
            raise ValueError(
                f"description exceeds {_MAX_METRIC_DESC_CHARS}-char limit "
                f"(got {len(description)})."
            )
        for side_name, refs in (("good", good_image_refs), ("bad", bad_image_refs)):
            if not isinstance(refs, list):
                raise ValueError(f"{side_name}_image_refs must be a list.")
            if len(refs) < _MIN_EXAMPLES_PER_CLASS:
                raise ValueError(
                    f"{side_name}_image_refs needs at least "
                    f"{_MIN_EXAMPLES_PER_CLASS} entr"
                    f"{'y' if _MIN_EXAMPLES_PER_CLASS == 1 else 'ies'} "
                    f"(got {len(refs)})."
                )
            if len(refs) > _MAX_EXAMPLES_PER_CLASS:
                raise ValueError(
                    f"{side_name}_image_refs exceeds the "
                    f"{_MAX_EXAMPLES_PER_CLASS}-example cap (got {len(refs)}). "
                    f"More examples eat the model's context budget without "
                    f"improving few-shot quality."
                )

        import shutil
        metric_dir = self._metric_dir(name)
        # Wipe any previous version so partial writes never get mixed in.
        if metric_dir.exists():
            shutil.rmtree(metric_dir)
        metric_dir.mkdir(parents=True, exist_ok=True)
        try:
            good_paths, good_urls = await self._save_example_images(
                metric_dir, "good", good_image_refs,
            )
            bad_paths, bad_urls = await self._save_example_images(
                metric_dir, "bad", bad_image_refs,
            )
        except Exception:
            shutil.rmtree(metric_dir, ignore_errors=True)
            raise

        record = {
            "name": name,
            "description": description.strip(),
            "good_images": good_paths,
            "bad_images": bad_paths,
            "good_source_urls": good_urls,
            "bad_source_urls": bad_urls,
            "n_good": len(good_paths),
            "n_bad": len(bad_paths),
            "created_at": time.time(),
        }
        with open(self._metric_json_path(name), "w") as f:
            json.dump(record, f, indent=2)
        logger.info(
            "Saved metric %r: %d good, %d bad examples at %s",
            name, record["n_good"], record["n_bad"], metric_dir,
        )
        return record

    @schema_method
    async def list_metrics(self) -> list:
        """List the QC metrics defined on this replica."""
        return self._list_metric_records()

    @schema_method
    async def get_metric(
        self,
        name: str = Field(..., description="Metric identifier."),
    ) -> dict:
        """Return one metric record by name."""
        return self._load_metric(name)

    @schema_method
    async def delete_metric(
        self,
        name: str = Field(..., description="Metric identifier."),
    ) -> dict:
        """Delete a metric and its reference images from the replica's disk."""
        import shutil
        if not _METRIC_NAME_RE.match(name):
            raise ValueError(
                f"metric name must match {_METRIC_NAME_RE.pattern} (got: {name!r})"
            )
        metric_dir = self._metric_dir(name)
        if not metric_dir.exists():
            raise ValueError(f"metric {name!r} not found.")
        shutil.rmtree(metric_dir)
        logger.info("Deleted metric %r at %s", name, metric_dir)
        return {"name": name, "deleted": True}

    @schema_method
    async def inspect(
        self,
        image_ref: str = Field(
            ...,
            description=(
                "Image to inspect. Either an HTTPS URL (public or presigned), "
                "or a Hypha artifact reference '<workspace>/<alias>:<file_path>' "
                "(e.g. 'ws-user-github|49943582/qc-samples:images/frame_001.tif'). "
                "Maximum decoded file size: 25 MB."
            ),
        ),
        instruction: Optional[str] = Field(
            None,
            description=(
                "Free-text QC instruction telling the VLM what to look for. "
                "Either `instruction` OR `metric_name` is required. If both are "
                "given the metric's pre-defined good/bad examples are used "
                "and `instruction` overrides the metric's stored description. "
                "Max 4000 characters."
            ),
        ),
        metric_name: Optional[str] = Field(
            None,
            description=(
                "Name of a previously-defined metric (see create_metric / "
                "list_metrics). Switches inspect() into few-shot verdict mode: "
                "the model is shown the metric's good/bad reference images "
                "before the new image and asked to classify it as GOOD or BAD."
            ),
        ),
        max_new_tokens: int = Field(
            512,
            description="Maximum response tokens (1-1024).",
            ge=1, le=1024,
        ),
    ) -> dict:
        """Inspect a microscopy image and return a QC judgement.

        Two modes:

        - Free-text describe (instruction set, metric_name unset):
          existing behaviour. Returns a natural-language description of
          what the model sees relative to `instruction`.

        - Few-shot verdict (metric_name set):
          looks up the named metric's reference images, prepends them
          (good then bad) to the prompt, asks the model to classify the
          new image as GOOD or BAD against the metric's criterion. The
          response includes parsed `verdict` and `reason` fields plus the
          raw text in `description`. If `instruction` is also supplied,
          it overrides the metric's stored description for this call only.
        """
        t0 = time.time()

        if not metric_name and not (isinstance(instruction, str) and instruction.strip()):
            raise ValueError(
                "Either `metric_name` or `instruction` must be provided."
            )
        if isinstance(instruction, str):
            if len(instruction) > _MAX_INSTRUCTION_CHARS:
                raise ValueError(
                    f"instruction exceeds {_MAX_INSTRUCTION_CHARS}-char limit "
                    f"(got {len(instruction)})."
                )

        url = await self._resolve_to_url(image_ref)
        image, original_size = await self._download_image(url)

        if metric_name:
            from PIL import Image
            metric = self._load_metric(metric_name)
            metric_dir = self._metric_dir(metric_name)
            good_imgs = [Image.open(metric_dir / p).convert("RGB") for p in metric["good_images"]]
            bad_imgs  = [Image.open(metric_dir / p).convert("RGB") for p in metric["bad_images"]]

            t_gen0 = time.time()
            raw, n_tokens = await self._run_vlm_few_shot(
                new_image=image,
                metric=metric,
                good_images=good_imgs,
                bad_images=bad_imgs,
                instruction_override=instruction,
                max_new_tokens=max_new_tokens,
            )
            gen_dt = time.time() - t_gen0
            verdict, reason = self._parse_verdict(raw)
            result = {
                "mode": "few-shot",
                "metric_name": metric_name,
                "metric_description": instruction.strip() if instruction else metric["description"],
                "verdict": verdict,                # "good" / "bad" / None if unparseable
                "reason": reason,
                "description": raw,                # raw model output
                "n_good_examples": metric["n_good"],
                "n_bad_examples": metric["n_bad"],
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
