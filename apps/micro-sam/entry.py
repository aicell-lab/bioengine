"""micro-sam CPU entry — transport, routing, and fine-tuning orchestration.

The always-on CPU half of the two-deployment micro-sam app. It owns all Hypha/S3
transport and session orchestration, and composes the GPU ``RuntimeApp`` (in
``runtime.py``) by type hint — forwarding serving calls to it and firing a
long-running training call that drives the runtime's autoscaling so training and
inference can run on separate GPU replicas at the same time.

Only this entry is named in ``manifest.yaml``'s ``entry:``. Serving methods
return the runtime's ndarray wire-dicts unchanged (numpy-neutral across the Ray
hop and this entry's numpy-1.x proxy); the hypha client decodes them to real
ndarrays.
"""

import asyncio
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

from runtime import RuntimeApp

logger = bioengine.logger

SERVER_URL = "https://hypha.aicell.io"
SUPPORTED_FILE_TYPES = Literal[".npy", ".png", ".tiff", ".tif", ".jpeg", ".jpg"]
ModelType = Literal["vit_b_lm", "vit_l_lm", "vit_t_lm", "vit_b", "vit_l", "vit_h"]


def _read_pip(name: str) -> List[str]:
    text = (Path(__file__).parent / name).read_text()
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@bioengine.app(
    num_cpus=1,
    num_gpus=0,
    memory_mb=4 * 1024,
    pip=_read_pip("requirements-entry.txt"),
    max_ongoing_requests=10,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 1,
    },
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    graceful_shutdown_timeout_s=300.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class EntryApp:
    """CPU entry: transport + routing to the GPU runtime + session orchestration."""

    def __init__(self, runtime: RuntimeApp) -> None:
        self.runtime = runtime
        self.start_time = time.time()
        self._hypha_token = os.getenv("HYPHA_TOKEN")
        if not self._hypha_token:
            raise RuntimeError("HYPHA_TOKEN environment variable is not set")
        self._hypha = None
        self._s3 = None
        self._http: Optional[httpx.AsyncClient] = None
        self._training_tasks: Dict[str, asyncio.Task] = {}

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
        # Entry's own dependency only; the GPU runtime is checked per-call so
        # CPU-only methods stay served when the runtime is down/scaling.
        if self._s3 is None:
            raise RuntimeError("S3 storage service not connected")

    async def _check_runtime_available(self) -> None:
        try:
            await asyncio.wait_for(self.runtime.ping(), timeout=2.0)
        except Exception as e:
            raise RuntimeError(
                "GPU runtime is not available yet — inference, embedding, ONNX "
                "export, and training are unavailable until it starts."
            ) from e

    # === I/O transport (S3 + URL) ===

    @bioengine.method
    async def get_upload_url(
        self,
        file_type: SUPPORTED_FILE_TYPES = Field(
            ...,
            description='File type for the upload: ".npy" (NumPy array), ".png", '
            '".tiff"/".tif", or ".jpeg"/".jpg".',
        ),
    ) -> Dict[str, str]:
        """Presigned S3 PUT URL (1-hour TTL) for staging an input image."""
        file_path = f"temp/{uuid.uuid4()}{file_type}"
        upload_url = await self._s3.put_file(file_path=file_path, ttl=3600)
        return {"upload_url": upload_url, "file_path": file_path}

    async def _get_download_url(self, file_path: str) -> str:
        return await self._s3.get_file(file_path=file_path, use_proxy=True)

    async def _load_image_from_source(self, source: str) -> np.ndarray:
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

    async def _resolve_label_array(self, source: Union[np.ndarray, str], ref_shape: tuple) -> np.ndarray:
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

    async def _save_npy_to_temp(self, array: np.ndarray) -> str:
        info = await self.get_upload_url(file_type=".npy")
        buffer = BytesIO()
        np.save(buffer, array)
        await self._http.put(info["upload_url"], content=buffer.getvalue())
        return await self._get_download_url(info["file_path"])

    def _session_checkpoint(self, session_id: str) -> str:
        import training

        ckpt = training.checkpoint_path(session_id)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"No trained checkpoint for session '{session_id}' "
                "(best.pt not found — training may be unfinished or failed)."
            )
        return str(ckpt)

    # === serving (forwarded to the GPU runtime) ===

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
            "vit_t_lm) carry the AIS decoder and suit brightfield/fluorescence cells.",
        ),
        device: Literal["cuda", "cpu"] = Field("cuda", description="Compute device."),
        pred_iou_thresh: Optional[float] = Field(None, description="Post-processing IoU threshold."),
        stability_score_thresh: Optional[float] = Field(None, description="Post-processing stability-score threshold."),
        min_size: Optional[int] = Field(None, description="Minimum object size in pixels."),
        model: Optional[str] = Field(None, description="Ignored; Cellpose-API compatibility."),
        diameter: Optional[float] = Field(None, description="Ignored; Cellpose-API compatibility."),
        flow_threshold: Optional[float] = Field(None, description="Ignored; Cellpose-API compatibility."),
        cellprob_threshold: Optional[float] = Field(None, description="Ignored; Cellpose-API compatibility."),
        niter: Optional[int] = Field(None, description="Ignored; Cellpose-API compatibility."),
        enable_clahe: Optional[bool] = Field(None, description="Ignored; Cellpose-API compatibility."),
        session_id: Optional[str] = Field(
            None, description="Serve the fine-tuned model from this training session's checkpoint."
        ),
    ) -> List[Dict[str, Any]]:
        """Automatic μSAM instance segmentation. Cellpose ``infer`` drop-in:
        returns a bare list, one item per input, each ``{"output": <int32 [H,W]
        instance label mask>}``.
        """
        await self._check_runtime_available()
        generate_kwargs: Dict[str, Any] = {}
        if pred_iou_thresh is not None:
            generate_kwargs["pred_iou_thresh"] = pred_iou_thresh
        if stability_score_thresh is not None:
            generate_kwargs["stability_score_thresh"] = stability_score_thresh
        if min_size is not None:
            generate_kwargs["min_size"] = min_size
        checkpoint = self._session_checkpoint(session_id) if session_id else None
        images = [await self._resolve_image(src) for src in input_arrays]
        return await self.runtime.auto_segment(
            images=images, model_type=model_type, device=device,
            generate_kwargs=generate_kwargs, checkpoint=checkpoint,
        )

    @bioengine.method
    async def segment_image(
        self,
        inputs: Union[np.ndarray, str] = Field(..., description="A single input image."),
        model_type: ModelType = Field("vit_b_lm", description="μSAM model."),
        device: Literal["cuda", "cpu"] = Field("cuda", description="Compute device."),
        session_id: Optional[str] = Field(None, description="Serve a fine-tuned session checkpoint."),
    ) -> List[Dict[str, Any]]:
        """Single-image alias of ``infer`` — same ``[{"output": int32 [H,W]}]`` shape."""
        return await self.infer(
            input_arrays=[inputs], model_type=model_type, device=device, session_id=session_id
        )

    @bioengine.method
    async def compute_image_embedding(
        self,
        inputs: Union[np.ndarray, str] = Field(..., description="A single input image."),
        model_type: ModelType = Field("vit_b_lm", description="μSAM model."),
        output_mode: Literal["embedding", "embedding+masks"] = Field(
            "embedding",
            description="'embedding' returns only the encoder embedding; "
            "'embedding+masks' also returns the automatic AIS instance mask.",
        ),
        device: Literal["cuda", "cpu"] = Field("cuda", description="Compute device."),
        return_features_url: bool = Field(
            False,
            description="If True, the 4MB features are saved to a temporary .npy in S3 "
            "and returned as 'features_url' instead of the raw 'features' ndarray.",
        ),
        session_id: Optional[str] = Field(None, description="Serve a fine-tuned session checkpoint."),
    ) -> Dict[str, Any]:
        """Encoder embedding (+ optional AIS masks) for the in-browser prompt decoder."""
        await self._check_runtime_available()
        checkpoint = self._session_checkpoint(session_id) if session_id else None
        image = await self._resolve_image(inputs)
        payload = await self.runtime.encode(
            image=image, model_type=model_type, device=device,
            checkpoint=checkpoint, output_mode=output_mode,
        )
        if return_features_url:
            feat = payload.pop("features")
            arr = np.frombuffer(feat["_rvalue"], dtype=feat["_rdtype"]).reshape(feat["_rshape"])
            payload["features_url"] = await self._save_npy_to_temp(arr)
        return payload

    @bioengine.method
    async def get_onnx_model(
        self,
        model_type: ModelType = Field("vit_b_lm", description="μSAM model."),
        quantize: bool = Field(True, description="Quantize for faster browser runtime."),
        device: Literal["cuda", "cpu"] = Field("cuda", description="Compute device."),
    ) -> bytes:
        """The interactive prompt decoder as ONNX bytes for onnxruntime-web."""
        await self._check_runtime_available()
        return await self.runtime.export_onnx(model_type=model_type, device=device, quantize=quantize)

    # === fine-tuning orchestration (async-job model) ===

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

    async def _run_training(
        self, session_id, model_type, train_images, train_labels, val_images, val_labels, params
    ) -> None:
        import training

        try:
            data = await self._prepare_training_data(
                session_id, train_images, train_labels, val_images, val_labels, params
            )
            training.write_training_params(session_id, {
                **params, "model_type": model_type,
                "patch_shape": list(data["patch_shape"]),
                "train_images": data["train_images"], "train_labels": data["train_labels"],
                "val_images": data["val_images"], "val_labels": data["val_labels"],
            })
            # Long-running: this await holds one GPU runtime replica for the whole
            # training run, so a concurrent infer is routed to a second replica.
            await self.runtime.train(session_id=session_id, model_type=model_type, params=params)
        except asyncio.CancelledError:
            training.write_status(session_id, status="STOPPED", message="training task cancelled")
            raise
        except Exception as e:
            logger.error(f"Training session {session_id} failed: {e}")
            training.write_status(session_id, status="FAILED", message=str(e)[:800])
        finally:
            self._training_tasks.pop(session_id, None)

    @bioengine.method
    async def start_training(
        self,
        train_images: List[Union[np.ndarray, str]] = Field(
            ..., description="Training images: arrays, http(s) URLs, or get_upload_url paths."),
        train_labels: List[Union[np.ndarray, str]] = Field(
            ..., description="Dense instance-label masks paired 1:1 with train_images "
            "(.tif/.png/.npy, or a .geojson polygon FeatureCollection). AIS fine-tuning "
            "needs DENSE labels — annotate all objects per image."),
        val_images: Optional[List[Union[np.ndarray, str]]] = Field(None, description="Optional validation images."),
        val_labels: Optional[List[Union[np.ndarray, str]]] = Field(None, description="Validation labels."),
        model_type: ModelType = Field("vit_b_lm", description="Base μSAM model to fine-tune."),
        n_epochs: int = Field(5, description="Number of training epochs."),
        n_objects_per_batch: int = Field(
            8, description="Objects per batch — main GPU-memory knob; 8 fits vit_b on 24GB."),
        patch_size: int = Field(512, description="Square training patch side (clamped to the smallest image)."),
        batch_size: int = Field(1, description="Training batch size."),
        learning_rate: float = Field(1e-5, description="AdamW learning rate."),
        val_fraction: float = Field(0.2, description="Val split fraction when val is omitted."),
        n_samples: Optional[int] = Field(None, description="Patches sampled per epoch (auto if omitted)."),
        label: str = Field("", description="Optional human-readable tag for this session."),
    ) -> Dict[str, Any]:
        """Start a μSAM fine-tuning session (AIS decoder) and return immediately
        with the status incl. ``session_id``. Data prep runs on this CPU entry;
        the training itself runs in a subprocess on a GPU runtime replica. Poll
        ``get_training_status`` and, once ``checkpoint_available``, serve via
        ``infer(session_id=...)``.
        """
        import training

        await self._check_runtime_available()
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
        task = asyncio.create_task(self._run_training(
            session_id, model_type, train_images, train_labels, val_images, val_labels, params
        ))
        self._training_tasks[session_id] = task
        return training.get_status(session_id)

    @bioengine.method
    async def get_training_status(
        self, session_id: str = Field(..., description="Session id from start_training.")
    ) -> Dict[str, Any]:
        """Fine-tuning session status (PREPARING/TRAINING/COMPLETED/FAILED/STOPPED),
        elapsed time, and whether the checkpoint is servable."""
        import training

        return training.get_status(session_id)

    @bioengine.method
    async def list_training_sessions(self) -> Dict[str, Any]:
        """All fine-tuning sessions on this worker, keyed by session_id."""
        import training

        return training.list_sessions()

    @bioengine.method
    async def stop_training(
        self, session_id: str = Field(..., description="Session id to stop.")
    ) -> Dict[str, Any]:
        """Request cancellation (an in-flight training epoch may finish first)."""
        import training

        training.request_stop(session_id)
        task = self._training_tasks.get(session_id)
        if task is not None and not task.done():
            task.cancel()
        training.write_status(session_id, status="STOPPED", message="stop requested by user")
        return training.get_status(session_id)

    @bioengine.method
    async def ping(self) -> Dict[str, Any]:
        """Entry status + GPU runtime availability."""
        runtime_available = False
        try:
            await asyncio.wait_for(self.runtime.ping(), timeout=2.0)
            runtime_available = True
        except Exception:
            pass
        return {
            "status": "ok",
            "entry_uptime": time.time() - self.start_time,
            "runtime_available": runtime_available,
            "timestamp": datetime.now().isoformat(),
        }
