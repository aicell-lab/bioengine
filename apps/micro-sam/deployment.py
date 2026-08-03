"""micro-sam (μSAM) runner — one resident SAM encoder, three decoder consumers.

The paper's Act 4 flagship. A single fine-tuned SAM image encoder stays resident
on the GPU and backs three cheap consumers (Leg B):

  1. ``compute_image_embedding`` — run the encoder once per image, return the
     image embedding (and optionally the automatic masks). The embedding feeds
     the in-browser ONNX prompt decoder, so "draw a box → get a mask" needs no
     GPU round-trip per prompt.
  2. ``infer`` / ``segment_image`` — the μSAM AIS decoder (UNETR instance
     decoder on the ``*_lm`` models) produces a full instance label mask without
     prompts. This is the propose-and-prune pre-segmentation.
  3. ``get_onnx_model`` — export the lightweight interactive prompt decoder to
     ONNX bytes for onnxruntime-web in the annotation canvas.

Modelled on ``apps/cellpose4-runner`` (I/O transport, resident-model pattern,
GPU lock) and the deprecated ``../bioimageio-colab`` SAM backend (embedding /
ONNX-export / prompt mechanics).

Import rule: this module holds ``@bioengine.app`` and is introspected by the
worker with only ``bioengine[worker]`` + the standard library installed, so the
heavy deps (``torch``, ``micro_sam``, ``segment_anything``) are imported inside
method bodies. They ship to the replica via the decorator's ``pip=`` list.
"""

import asyncio
import gc
import os
import time
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import bioengine
import httpx
import numpy as np
from hypha_rpc import connect_to_server
from pydantic import Field

logger = bioengine.logger

SERVER_URL = "https://hypha.aicell.io"
SUPPORTED_FILE_TYPES = Literal[".npy", ".png", ".tiff", ".tif", ".jpeg", ".jpg"]

# LM generalists carry the AIS instance-segmentation decoder, so automatic
# segmentation works out of the box; vit_b is the speed/quality balance for
# brightfield, vit_l the higher-quality option.
ModelType = Literal["vit_b_lm", "vit_l_lm", "vit_t_lm", "vit_b", "vit_l", "vit_h"]


def _read_pip(name: str) -> List[str]:
    """Load a ``requirements-*.txt`` file next to this module."""
    text = (Path(__file__).parent / name).read_text()
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@bioengine.app(
    num_cpus=1,
    num_gpus=1,
    memory_mb=16 * 1024,
    pip=_read_pip("requirements-deployment.txt"),
    max_ongoing_requests=10,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 2,
        "target_num_ongoing_requests_per_replica": 3.0,
        "metrics_interval_s": 2.0,
        "look_back_period_s": 10.0,
        "downscale_delay_s": 600,
        "upscale_delay_s": 0.0,
    },
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    graceful_shutdown_timeout_s=120.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class MicroSAM:
    """Serves μSAM with one resident encoder backing three decoder consumers."""

    def __init__(self) -> None:
        self.start_time = time.time()
        self._hypha_token = os.getenv("HYPHA_TOKEN")
        if not self._hypha_token:
            raise RuntimeError("HYPHA_TOKEN environment variable is not set")

        # Serialises GPU work so one request touches the resident model at a
        # time; the RPCs queued on the lock are what the autoscaler measures.
        self._gpu_lock = asyncio.Lock()
        self._predictor = None
        self._segmenter = None
        self._loaded_model_type: Optional[str] = None
        self._onnx_cache: Dict[str, bytes] = {}

        self._hypha = None
        self._s3 = None
        self._http: Optional[httpx.AsyncClient] = None

    @bioengine.async_init
    async def _connect(self) -> None:
        self._hypha = await connect_to_server(
            {"server_url": SERVER_URL, "token": self._hypha_token}
        )
        self._s3 = await self._hypha.get_service("public/s3-storage")
        self._http = httpx.AsyncClient(timeout=60.0)
        logger.info(f"Connected to Hypha server at {SERVER_URL}")

    @bioengine.health_check
    async def _health(self) -> None:
        if self._s3 is None:
            raise RuntimeError("S3 storage service not connected")

    # === I/O transport (S3 + URL), mirrors cellpose4-runner ===

    @bioengine.method
    async def get_upload_url(
        self,
        file_type: SUPPORTED_FILE_TYPES = Field(
            ...,
            description='File type for the upload: ".npy" (NumPy array), ".png", '
            '".tiff"/".tif", or ".jpeg"/".jpg".',
        ),
    ) -> Dict[str, str]:
        """Request a presigned upload URL for an input image (1-hour TTL).

        Upload the file to ``upload_url`` via HTTP PUT, then pass the returned
        ``file_path`` as an image source to ``infer`` / ``compute_image_embedding``.
        """
        file_path = f"temp/{uuid.uuid4()}{file_type}"
        upload_url = await self._s3.put_file(file_path=file_path, ttl=3600)
        return {"upload_url": upload_url, "file_path": file_path}

    async def _get_download_url(self, file_path: str) -> str:
        return await self._s3.get_file(file_path=file_path, use_proxy=True)

    async def _load_image_from_source(self, source: str) -> np.ndarray:
        """Load an image from an http(s) URL or a ``get_upload_url`` file path."""
        ext = Path(source.split("?")[0]).suffix.lower()
        if ext not in SUPPORTED_FILE_TYPES.__args__:
            raise ValueError(
                f"Unsupported file extension '{ext}' in source '{source}'. "
                f"Supported: {SUPPORTED_FILE_TYPES.__args__}"
            )
        url = source if source.startswith(("http://", "https://")) else (
            await self._get_download_url(source)
        )
        resp = await self._http.get(url)
        if resp.status_code == 404:
            raise FileNotFoundError(f"Source '{source}' does not exist or has expired.")
        resp.raise_for_status()

        buffer = BytesIO(resp.content)
        if ext == ".npy":
            return await asyncio.to_thread(np.load, buffer)
        import imageio.v3 as iio

        return await asyncio.to_thread(iio.imread, buffer)

    async def _resolve_image(self, source: Union[np.ndarray, str]) -> np.ndarray:
        if isinstance(source, str):
            return await self._load_image_from_source(source)
        if isinstance(source, np.ndarray):
            return source
        return np.asarray(source)

    async def _save_npy_to_temp(self, array: np.ndarray) -> str:
        """Save an array to a temporary ``.npy`` in S3, return a download URL."""
        info = await self.get_upload_url(file_type=".npy")
        buffer = BytesIO()
        np.save(buffer, array)
        await self._http.put(info["upload_url"], content=buffer.getvalue())
        return await self._get_download_url(info["file_path"])

    # === Model residency (one encoder + AIS segmenter at a time) ===

    @staticmethod
    def _to_image_format(array: np.ndarray) -> np.ndarray:
        """Coerce input to HxWx3 uint8 RGB, as the SAM image encoder expects."""
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        if array.ndim == 2:
            array = np.concatenate([array[..., None]] * 3, axis=-1)
        elif array.ndim == 3 and array.shape[0] == 3:
            array = np.transpose(array, [1, 2, 0])
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(
                f"Invalid input image of shape {array.shape}. Expected 2D (HxW) "
                "grayscale or 3-channel (HxWx3 / 3xHxW) RGB."
            )
        if array.dtype != np.uint8:
            array = array.astype("float32") - array.min(axis=(0, 1))
            denom = array.max(axis=(0, 1))
            denom[denom == 0] = 1.0
            array = (array / denom * 255).astype("uint8")
        return array

    def _ensure_model(self, model_type: str, device: str):
        """Load (predictor, AIS segmenter), reusing the resident pair when the
        model_type is unchanged; a switch frees the previous model's VRAM.
        Blocking — call via ``asyncio.to_thread`` under ``self._gpu_lock``.
        """
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter

        if model_type != self._loaded_model_type:
            self._predictor = None
            self._segmenter = None
            self._loaded_model_type = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            logger.info(f"🔄 Loading μSAM model '{model_type}' on {device}...")
            predictor, segmenter = get_predictor_and_segmenter(
                model_type=model_type,
                device=device,
                # None → μSAM auto-selects AIS when the model ships a decoder
                # (all *_lm models do), else AMG.
                segmentation_mode=None,
            )
            self._predictor = predictor
            self._segmenter = segmenter
            self._loaded_model_type = model_type
            logger.info(f"✅ μSAM model '{model_type}' loaded.")
        return self._predictor, self._segmenter

    # === Consumer 2: automatic instance segmentation (AIS decoder) ===

    def _auto_segment(
        self,
        model_type: str,
        device: str,
        image: np.ndarray,
        generate_kwargs: Dict[str, Any],
    ) -> np.ndarray:
        """Run μSAM automatic instance segmentation → int32 [H,W] label mask.
        Blocking — call under the GPU lock.
        """
        from micro_sam.automatic_segmentation import automatic_instance_segmentation

        predictor, segmenter = self._ensure_model(model_type, device)
        image = self._to_image_format(image)
        labels = automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=image,
            # RGB must be flagged ndim=2, else the 3-channel axis is misread as z.
            ndim=2,
            verbose=False,
            **generate_kwargs,
        )
        return np.asarray(labels).astype(np.int32)

    @bioengine.method
    async def infer(
        self,
        input_arrays: List[Union[np.ndarray, str]] = Field(
            ...,
            description="List of input images. Each is a numpy array (HxW, HxWx3, "
            "or 3xHxW), an http(s) URL, or a get_upload_url file path. Cellpose "
            "drop-in shape: a list of [3,H,W] uint8 RGB ndarrays.",
        ),
        model_type: ModelType = Field(
            "vit_b_lm",
            description="μSAM model. LM generalists (vit_b_lm / vit_l_lm / "
            "vit_t_lm) carry the AIS decoder and suit brightfield/fluorescence "
            "cells; vit_b_lm is the speed/quality balance.",
        ),
        device: Literal["cuda", "cpu"] = Field("cuda", description="Compute device."),
        pred_iou_thresh: Optional[float] = Field(
            None, description="Post-processing IoU threshold forwarded to the segmenter."
        ),
        stability_score_thresh: Optional[float] = Field(
            None, description="Post-processing stability-score threshold."
        ),
        min_size: Optional[int] = Field(
            None, description="Minimum object size in pixels (smaller are dropped)."
        ),
        model: Optional[str] = Field(
            None, description="Ignored; accepted for Cellpose-API call-site compatibility."
        ),
        diameter: Optional[float] = Field(
            None, description="Ignored; accepted for Cellpose-API compatibility."
        ),
        flow_threshold: Optional[float] = Field(
            None, description="Ignored; accepted for Cellpose-API compatibility."
        ),
        cellprob_threshold: Optional[float] = Field(
            None, description="Ignored; accepted for Cellpose-API compatibility."
        ),
        niter: Optional[int] = Field(
            None, description="Ignored; accepted for Cellpose-API compatibility."
        ),
        enable_clahe: Optional[bool] = Field(
            None, description="Ignored; accepted for Cellpose-API compatibility."
        ),
    ) -> List[Dict[str, np.ndarray]]:
        """Automatic μSAM instance segmentation (propose-and-prune pre-seg).

        A true drop-in for the frontend's Cellpose ``infer`` reader: returns a
        **bare list**, one item per input, each ``{"output": <int32 [H,W] instance
        label mask>}`` (background 0, one positive integer per object).
        """
        generate_kwargs: Dict[str, Any] = {}
        if pred_iou_thresh is not None:
            generate_kwargs["pred_iou_thresh"] = pred_iou_thresh
        if stability_score_thresh is not None:
            generate_kwargs["stability_score_thresh"] = stability_score_thresh
        if min_size is not None:
            generate_kwargs["min_size"] = min_size

        images = [await self._resolve_image(src) for src in input_arrays]
        logger.info(f"🤖 μSAM AIS on {len(images)} image(s) with '{model_type}'...")
        results: List[Dict[str, np.ndarray]] = []
        async with self._gpu_lock:
            for image in images:
                labels = await asyncio.to_thread(
                    self._auto_segment, model_type, device, image, generate_kwargs
                )
                results.append({"output": labels})
        return results

    @bioengine.method
    async def segment_image(
        self,
        inputs: Union[np.ndarray, str] = Field(
            ..., description="A single input image: numpy array, http(s) URL, or "
            "get_upload_url file path."
        ),
        model_type: ModelType = Field("vit_b_lm", description="μSAM model."),
        device: Literal["cuda", "cpu"] = Field("cuda", description="Compute device."),
    ) -> List[Dict[str, np.ndarray]]:
        """Single-image alias of ``infer`` — same ``[{"output": int32 [H,W]}]`` shape."""
        return await self.infer(input_arrays=[inputs], model_type=model_type, device=device)

    # === Consumer 1: image embedding (feeds the in-browser prompt decoder) ===

    def _encode(self, model_type: str, device: str, image: np.ndarray) -> Dict[str, Any]:
        """Run the SAM image encoder → embedding payload for onnxruntime-web.
        Blocking — call under the GPU lock.
        """
        predictor, _ = self._ensure_model(model_type, device)
        image = self._to_image_format(image)
        original_image_shape = image.shape[:2]
        sam = predictor.model
        sam_scale = sam.image_encoder.img_size / max(original_image_shape)

        predictor.reset_image()
        predictor.set_image(image)
        features = predictor.get_image_embedding().cpu().numpy()
        predictor.reset_image()
        return {
            "features": features.astype(np.float32),
            "original_image_shape": [int(x) for x in original_image_shape],
            "sam_scale": float(sam_scale),
            "mask_threshold": float(sam.mask_threshold),
        }

    @bioengine.method
    async def compute_image_embedding(
        self,
        inputs: Union[np.ndarray, str] = Field(
            ..., description="A single input image: numpy array, http(s) URL, or "
            "get_upload_url file path."
        ),
        model_type: ModelType = Field("vit_b_lm", description="μSAM model."),
        output_mode: Literal["embedding", "embedding+masks"] = Field(
            "embedding",
            description="'embedding' returns only the encoder embedding "
            "(interactive-annotation mode). 'embedding+masks' also runs the AIS "
            "decoder and returns the automatic instance mask (propose-and-prune).",
        ),
        device: Literal["cuda", "cpu"] = Field("cuda", description="Compute device."),
        return_features_url: bool = Field(
            False,
            description="If True, the 4MB encoder features are saved to a temporary "
            ".npy in S3 and returned as 'features_url' (presigned, 1h) instead of the "
            "raw 'features' ndarray — for large batches / slow links.",
        ),
    ) -> Dict[str, Any]:
        """Run the resident encoder once and return its embedding.

        The embedding always comes back (interactive box→mask must always be
        available). Returns ``{features | features_url, original_image_shape,
        sam_scale, mask_threshold}`` and, when ``output_mode='embedding+masks'``,
        an additional ``masks`` (int32 [H,W] instance label mask).
        """
        image = await self._resolve_image(inputs)
        logger.info(f"🧠 μSAM embedding ({output_mode}) with '{model_type}'...")
        async with self._gpu_lock:
            payload = await asyncio.to_thread(self._encode, model_type, device, image)
            if output_mode == "embedding+masks":
                labels = await asyncio.to_thread(
                    self._auto_segment, model_type, device, image, {}
                )
                payload["masks"] = labels

        if return_features_url:
            payload["features_url"] = await self._save_npy_to_temp(payload.pop("features"))
        return payload

    # === Consumer 3: interactive prompt decoder → ONNX (browser decode) ===

    def _export_onnx(self, model_type: str, device: str, quantize: bool) -> bytes:
        """Export the lightweight SAM prompt decoder to (optionally quantized)
        ONNX bytes for onnxruntime-web. Blocking — call under the GPU lock.
        """
        import tempfile
        import warnings

        import torch
        from onnxruntime import InferenceSession
        from onnxruntime.quantization import QuantType
        from onnxruntime.quantization.quantize import quantize_dynamic
        from segment_anything.utils.onnx import SamOnnxModel

        predictor, _ = self._ensure_model(model_type, device)
        sam = predictor.model

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / f"{model_type}.onnx"
            sam.to("cpu")
            try:
                onnx_model = SamOnnxModel(
                    model=sam,
                    return_single_mask=True,
                    use_stability_score=False,
                    return_extra_metrics=False,
                )
                embed_dim = sam.prompt_encoder.embed_dim
                embed_size = sam.prompt_encoder.image_embedding_size
                mask_input_size = [4 * x for x in embed_size]
                dummy_inputs = {
                    "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
                    "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
                    "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
                    "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
                    "has_mask_input": torch.tensor([1], dtype=torch.float),
                    "orig_im_size": torch.tensor([1024, 1024], dtype=torch.float),
                }
                dynamic_axes = {
                    "point_coords": {1: "num_points"},
                    "point_labels": {1: "num_points"},
                }
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        str(onnx_path),
                        export_params=True,
                        verbose=False,
                        opset_version=17,
                        do_constant_folding=True,
                        input_names=list(dummy_inputs.keys()),
                        output_names=["masks", "iou_predictions", "low_res_masks"],
                        dynamic_axes=dynamic_axes,
                    )
                ort_inputs = {k: v.cpu().numpy() for k, v in dummy_inputs.items()}
                InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(
                    None, ort_inputs
                )
                if quantize:
                    q_path = onnx_path.with_stem(onnx_path.stem + "_quantized")
                    quantize_dynamic(
                        model_input=str(onnx_path),
                        model_output=str(q_path),
                        per_channel=False,
                        reduce_range=False,
                        weight_type=QuantType.QUInt8,
                    )
                    onnx_path = q_path
                return onnx_path.read_bytes()
            finally:
                sam.to(device)

    @bioengine.method
    async def get_onnx_model(
        self,
        model_type: ModelType = Field("vit_b_lm", description="μSAM model."),
        quantize: bool = Field(True, description="Quantize for faster browser runtime."),
        device: Literal["cuda", "cpu"] = Field("cuda", description="Compute device."),
    ) -> bytes:
        """Export the interactive prompt decoder to ONNX bytes (cached per model).

        The frontend fetches this once per session and runs it with
        onnxruntime-web, turning each user bounding box into a mask locally using
        the ``compute_image_embedding`` features — no GPU round-trip per prompt.
        """
        cache_key = f"{model_type}:{quantize}"
        if cache_key in self._onnx_cache:
            return self._onnx_cache[cache_key]
        logger.info(f"📦 Exporting ONNX prompt decoder for '{model_type}'...")
        async with self._gpu_lock:
            onnx_bytes = await asyncio.to_thread(
                self._export_onnx, model_type, device, quantize
            )
        self._onnx_cache[cache_key] = onnx_bytes
        return onnx_bytes

    @bioengine.method
    async def ping(self) -> Dict[str, Union[str, float, None]]:
        """Return service status and the currently resident model type."""
        return {
            "status": "ok",
            "loaded_model_type": self._loaded_model_type,
            "uptime": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat(),
        }
