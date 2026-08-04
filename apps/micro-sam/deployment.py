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
from concurrent.futures import ThreadPoolExecutor
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
    num_cpus=4,
    num_gpus=1,
    memory_mb=24 * 1024,
    pip=_read_pip("requirements-deployment.txt"),
    max_ongoing_requests=10,
    autoscaling_config={
        # Single replica: this app both serves and runs a long fine-tuning job
        # (in a background thread on the same GPU), so a second replica would
        # only duplicate the resident encoder without owning the training state.
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 1,
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
        self._loaded_key: Optional[tuple] = None
        self._onnx_cache: Dict[str, bytes] = {}

        # Fine-tuning sessions: one background asyncio task + executor each.
        self._training_tasks: Dict[str, asyncio.Task] = {}
        self._executors: Dict[str, ThreadPoolExecutor] = {}

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

    @staticmethod
    def _nd(arr: np.ndarray) -> Dict[str, Any]:
        """Encode an array as hypha-rpc's ndarray wire-dict.

        The worker's ProxyDeployment (numpy 1.x base env) cannot unpickle
        numpy-2.x arrays over the Ray hop, and this app must run numpy 2.x
        (python-elf). Returning the wire-dict crosses that hop as bytes and the
        hypha client still decodes it to a real ndarray — same format the
        frontend already sends inbound.
        """
        arr = np.ascontiguousarray(arr)
        return {
            "_rtype": "ndarray",
            "_rvalue": arr.tobytes(),
            "_rshape": list(arr.shape),
            "_rdtype": arr.dtype.name,
        }

    def _ensure_model(self, model_type: str, device: str, checkpoint: Optional[str] = None):
        """Load (predictor, AIS segmenter), reusing the resident pair when the
        (model_type, checkpoint) is unchanged; a switch frees the previous
        model's VRAM. ``checkpoint`` points at a fine-tuned ``best.pt`` to serve
        a just-trained model. Blocking — call via ``asyncio.to_thread`` under
        ``self._gpu_lock``.
        """
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter

        key = (model_type, checkpoint)
        if key != self._loaded_key:
            self._predictor = None
            self._segmenter = None
            self._loaded_key = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            label = f"{model_type}" + (f" (finetuned: {checkpoint})" if checkpoint else "")
            logger.info(f"🔄 Loading μSAM model '{label}' on {device}...")
            predictor, segmenter = get_predictor_and_segmenter(
                model_type=model_type,
                checkpoint=checkpoint,
                device=device,
                # None → μSAM auto-selects AIS when the model ships a decoder
                # (all *_lm models do), else AMG.
                segmentation_mode=None,
            )
            self._predictor = predictor
            self._segmenter = segmenter
            self._loaded_key = key
            self._loaded_model_type = model_type
            logger.info(f"✅ μSAM model '{label}' loaded.")
        return self._predictor, self._segmenter

    def _session_checkpoint(self, session_id: str) -> str:
        """Resolve a training session's ``best.pt``; raise if not yet available."""
        import training

        ckpt = training.checkpoint_path(session_id)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"No trained checkpoint for session '{session_id}' "
                "(best.pt not found — training may be unfinished or failed)."
            )
        return str(ckpt)

    # === Consumer 2: automatic instance segmentation (AIS decoder) ===

    def _auto_segment(
        self,
        model_type: str,
        device: str,
        image: np.ndarray,
        generate_kwargs: Dict[str, Any],
        checkpoint: Optional[str] = None,
    ) -> np.ndarray:
        """Run μSAM automatic instance segmentation → int32 [H,W] label mask.
        Blocking — call under the GPU lock.
        """
        from micro_sam.automatic_segmentation import automatic_instance_segmentation

        predictor, segmenter = self._ensure_model(model_type, device, checkpoint)
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
        session_id: Optional[str] = Field(
            None,
            description="If set, serve the fine-tuned model from this training "
            "session's checkpoint instead of the pretrained model_type.",
        ),
    ) -> List[Dict[str, Any]]:
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

        checkpoint = self._session_checkpoint(session_id) if session_id else None
        images = [await self._resolve_image(src) for src in input_arrays]
        tag = f"session {session_id}" if session_id else model_type
        logger.info(f"🤖 μSAM AIS on {len(images)} image(s) with '{tag}'...")
        results: List[Dict[str, Any]] = []
        async with self._gpu_lock:
            for image in images:
                labels = await asyncio.to_thread(
                    self._auto_segment, model_type, device, image, generate_kwargs, checkpoint
                )
                results.append({"output": self._nd(labels)})
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
        session_id: Optional[str] = Field(
            None, description="Serve a fine-tuned session checkpoint instead of model_type."
        ),
    ) -> List[Dict[str, Any]]:
        """Single-image alias of ``infer`` — same ``[{"output": int32 [H,W]}]`` shape."""
        return await self.infer(
            input_arrays=[inputs], model_type=model_type, device=device, session_id=session_id
        )

    # === Consumer 1: image embedding (feeds the in-browser prompt decoder) ===

    def _encode(self, model_type: str, device: str, image: np.ndarray,
                checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Run the SAM image encoder → embedding payload for onnxruntime-web.
        Blocking — call under the GPU lock.
        """
        predictor, _ = self._ensure_model(model_type, device, checkpoint)
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
        session_id: Optional[str] = Field(
            None, description="Serve a fine-tuned session checkpoint instead of model_type."
        ),
    ) -> Dict[str, Any]:
        """Run the resident encoder once and return its embedding.

        The embedding always comes back (interactive box→mask must always be
        available). Returns ``{features | features_url, original_image_shape,
        sam_scale, mask_threshold}`` and, when ``output_mode='embedding+masks'``,
        an additional ``masks`` (int32 [H,W] instance label mask).
        """
        checkpoint = self._session_checkpoint(session_id) if session_id else None
        image = await self._resolve_image(inputs)
        logger.info(f"🧠 μSAM embedding ({output_mode}) with "
                    f"'{session_id or model_type}'...")
        labels = None
        async with self._gpu_lock:
            payload = await asyncio.to_thread(self._encode, model_type, device, image, checkpoint)
            if output_mode == "embedding+masks":
                labels = await asyncio.to_thread(
                    self._auto_segment, model_type, device, image, {}, checkpoint
                )

        features = payload.pop("features")
        if return_features_url:
            payload["features_url"] = await self._save_npy_to_temp(features)
        else:
            payload["features"] = self._nd(features)
        if labels is not None:
            payload["masks"] = self._nd(labels)
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

    # === Fine-tuning (train + serve the just-trained model) ===

    async def _resolve_label_array(
        self, source: Union[np.ndarray, str], ref_shape: tuple
    ) -> np.ndarray:
        """Resolve a label source to a 2D instance-label array. GeoJSON polygon
        annotations are rasterized against the paired image's shape.
        """
        import training

        if isinstance(source, np.ndarray):
            return source
        if not isinstance(source, str):
            return np.asarray(source)
        ext = Path(source.split("?")[0]).suffix.lower()
        if ext in (".geojson", ".json"):
            url = source if source.startswith(("http://", "https://")) else (
                await self._get_download_url(source)
            )
            resp = await self._http.get(url)
            resp.raise_for_status()
            h, w = int(ref_shape[0]), int(ref_shape[1])
            return training.rasterize_geojson(resp.json(), width=w, height=h)
        return await self._load_image_from_source(source)

    async def _prepare_training_data(
        self, session_id, train_images, train_labels, val_images, val_labels, params
    ) -> Dict[str, Any]:
        import training

        if len(train_images) != len(train_labels):
            raise ValueError("train_images and train_labels must have equal length.")

        async def pairs(images, labels):
            out = []
            for img_src, lbl_src in zip(images, labels):
                img = await self._resolve_image(img_src)
                lbl = await self._resolve_label_array(lbl_src, np.asarray(img).shape[:2])
                out.append((img, lbl))
            return out

        train = await pairs(train_images, train_labels)
        val = await pairs(val_images, val_labels) if val_images and val_labels else None
        return await asyncio.to_thread(
            training.materialize_pairs, session_id, train, val,
            params["val_fraction"], params["patch_shape"],
        )

    async def _run_training_session(
        self, session_id, model_type, train_images, train_labels, val_images, val_labels, params
    ) -> None:
        import training

        try:
            data = await self._prepare_training_data(
                session_id, train_images, train_labels, val_images, val_labels, params
            )
            loop = asyncio.get_running_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            self._executors[session_id] = executor
            await loop.run_in_executor(
                executor, training.run_training_blocking, session_id, model_type, data, params
            )
        except asyncio.CancelledError:
            training.write_status(session_id, status="STOPPED", message="training task cancelled")
            raise
        except Exception as e:
            logger.error(f"Training session {session_id} failed: {e}")
            training.write_status(session_id, status="FAILED", message=str(e)[:800])
        finally:
            ex = self._executors.pop(session_id, None)
            if ex is not None:
                ex.shutdown(wait=False)
            self._training_tasks.pop(session_id, None)

    @bioengine.method
    async def start_training(
        self,
        train_images: List[Union[np.ndarray, str]] = Field(
            ..., description="Training images: numpy arrays, http(s) URLs, or "
            "get_upload_url file paths. Paired 1:1 with train_labels.",
        ),
        train_labels: List[Union[np.ndarray, str]] = Field(
            ..., description="Dense instance-label masks paired with train_images. "
            "Each is a numpy int mask, a .tif/.png/.npy label file, or a .geojson "
            "FeatureCollection of polygons (rasterized to instances). AIS "
            "fine-tuning needs DENSE labels — annotate all objects per image.",
        ),
        val_images: Optional[List[Union[np.ndarray, str]]] = Field(
            None, description="Optional validation images; if omitted a val split "
            "is taken from the training set (val_fraction).",
        ),
        val_labels: Optional[List[Union[np.ndarray, str]]] = Field(
            None, description="Validation labels paired with val_images."
        ),
        model_type: ModelType = Field(
            "vit_b_lm", description="Base μSAM model to fine-tune (LM generalist for cells)."
        ),
        n_epochs: int = Field(5, description="Number of training epochs."),
        n_objects_per_batch: int = Field(25, description="Objects sampled per batch."),
        patch_size: int = Field(
            512, description="Square training patch side (clamped to the smallest image)."
        ),
        batch_size: int = Field(1, description="Training batch size."),
        learning_rate: float = Field(1e-5, description="AdamW learning rate."),
        val_fraction: float = Field(
            0.2, description="Fraction of train pairs used for validation when val is omitted."
        ),
        n_samples: Optional[int] = Field(
            None, description="Patches sampled per epoch (auto if omitted)."
        ),
        label: str = Field("", description="Optional human-readable tag for this session."),
    ) -> Dict[str, Any]:
        """Start a μSAM fine-tuning session (with the AIS decoder) in the
        background and return immediately with the session status (incl.
        ``session_id``). Poll ``get_training_status`` and, once
        ``checkpoint_available`` is true, serve it via ``infer(session_id=...)``.
        """
        import training

        session_id = training.new_session_id()
        training.write_status(
            session_id, status="PREPARING", created_at=time.time(),
            model_type=model_type, label=label, n_train_inputs=len(train_images),
        )
        params = dict(
            n_epochs=n_epochs, n_objects_per_batch=n_objects_per_batch,
            batch_size=batch_size, learning_rate=learning_rate,
            val_fraction=val_fraction, n_samples=n_samples,
            patch_shape=(patch_size, patch_size), num_workers=0,
        )
        task = asyncio.create_task(self._run_training_session(
            session_id, model_type, train_images, train_labels, val_images, val_labels, params
        ))
        self._training_tasks[session_id] = task
        return training.get_status(session_id)

    @bioengine.method
    async def get_training_status(
        self, session_id: str = Field(..., description="Session id from start_training.")
    ) -> Dict[str, Any]:
        """Return a fine-tuning session's status (PREPARING/TRAINING/COMPLETED/
        FAILED/STOPPED), elapsed time, and whether its checkpoint is servable."""
        import training

        return training.get_status(session_id)

    @bioengine.method
    async def list_training_sessions(self) -> Dict[str, Any]:
        """Return all fine-tuning sessions on this worker, keyed by session_id."""
        import training

        return training.list_sessions()

    @bioengine.method
    async def stop_training(
        self, session_id: str = Field(..., description="Session id to stop.")
    ) -> Dict[str, Any]:
        """Request cancellation of a fine-tuning session. Note: an in-flight
        training epoch may run to completion before the task unwinds."""
        import training

        training.request_stop(session_id)
        task = self._training_tasks.get(session_id)
        if task is not None and not task.done():
            task.cancel()
        training.write_status(session_id, status="STOPPED", message="stop requested by user")
        return training.get_status(session_id)

    @bioengine.method
    async def ping(self) -> Dict[str, Union[str, float, None]]:
        """Return service status and the currently resident model type."""
        return {
            "status": "ok",
            "loaded_model_type": self._loaded_model_type,
            "uptime": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat(),
        }
