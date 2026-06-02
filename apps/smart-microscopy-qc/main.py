"""Smart Microscopy QC — VLM-backed real-time quality-control inspector.

Accepts a microscopy image (Hypha artifact reference OR HTTPS URL) and a
free-text instruction describing the QC metrics to evaluate, and returns the
VLM's textual description of what it sees relative to that instruction.

Backed by Qwen2.5-VL-3B-Instruct served via HuggingFace transformers on a
single NVIDIA A40-16C vGPU slice (Ampere, sm_86; 16 GB framebuffer time-
shared with co-tenants on the host A40). ~6 GB FP16 weights + Qwen's vision
encoder + KV cache fit comfortably. KTH's Ray cluster exposes each A40 as a
1-GPU-per-pod vGPU profile, so a single replica owns one slice.

7B-AWQ was attempted first but every viable AWQ kernel path failed on this
cluster: vLLM 0.10's V0 multimodal regression on Qwen2.5-VL, vLLM 0.9.2's
inspector bug, autoawq + bundled Triton couldn't compile bf16 dot products,
and autoawq-kernels' CUDA gemm raises `expected scalar type Int but found
Half` on the lm_head linear. 3B FP16 lands cleanly without quantisation.
"""

import asyncio
import logging
import os
import time
from typing import Any, Optional

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
_GENERATE_TIMEOUT_S = 120


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

    async def async_init(self) -> None:
        """Load the VLM and connect to Hypha for artifact resolution."""
        import os as _os
        for d in ("/tmp/triton-cache", "/tmp/hf-home", "/tmp/xdg-cache"):
            _os.makedirs(d, exist_ok=True)

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

    async def _download_image(self, url: str) -> tuple["Image.Image", Optional[tuple[int, int]]]:
        """Stream-download up to _MAX_IMAGE_BYTES, decode to RGB PIL image,
        and downscale to ≤ _MAX_PIXELS / _MAX_LONG_SIDE.

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
        needs_resize = (w * h > _MAX_PIXELS) or (long_side > _MAX_LONG_SIDE)
        if needs_resize:
            t_resize = time.time()
            scale_pix  = (_MAX_PIXELS / (w * h)) ** 0.5 if w * h > _MAX_PIXELS else 1.0
            scale_side = _MAX_LONG_SIDE / long_side if long_side > _MAX_LONG_SIDE else 1.0
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

    async def _run_vlm(
        self, image: "Image.Image", instruction: str, max_new_tokens: int
    ) -> tuple[str, int]:
        import torch

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }]

        def _generate():
            prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[prompt],
                images=[image],
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
            "license": "Qwen2.5-VL Apache 2.0 weights",
        }

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
        instruction: str = Field(
            ...,
            description=(
                "Free-text QC instruction telling the VLM what to look for "
                "(sharpness, focus, illumination uniformity, dark/saturated regions, "
                "object counts/shapes, contamination, etc.). Max 4000 characters."
            ),
        ),
        max_new_tokens: int = Field(
            512,
            description="Maximum response tokens (1-1024).",
            ge=1, le=1024,
        ),
    ) -> dict:
        """Inspect a microscopy image and return a textual QC report.

        Returns:
          description     - VLM-generated text describing the image relative to instruction
          image_size      - [width, height] of the (possibly downscaled) image fed to the VLM
          source_url      - URL used to fetch the image
          model           - Model ID used
          tokens_generated, generation_time_s, tokens_per_second
          processing_time_s
        """
        t0 = time.time()

        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string.")
        if len(instruction) > _MAX_INSTRUCTION_CHARS:
            raise ValueError(
                f"instruction exceeds {_MAX_INSTRUCTION_CHARS}-char limit "
                f"(got {len(instruction)})."
            )

        url = await self._resolve_to_url(image_ref)
        image, original_size = await self._download_image(url)

        t_gen0 = time.time()
        description, n_tokens = await self._run_vlm(image, instruction, max_new_tokens)
        gen_dt = time.time() - t_gen0

        result = {
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
                f"to {image.size[0]}x{image.size[1]} before VLM. The QC verdict "
                f"applies to the resized image."
            )
        return result
