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
import json
import os
import re
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import bioengine
import httpx
import numpy as np
import yaml
from hypha_rpc import connect_to_server
from pydantic import Field

from runtime import RuntimeApp
from runtime_cellpose import CellposeRuntime

logger = bioengine.logger

SERVER_URL = "https://hypha.aicell.io"
SUPPORTED_FILE_TYPES = Literal[".npy", ".png", ".tiff", ".tif", ".jpeg", ".jpg"]
UPLOAD_FILE_TYPES = Literal[".npy", ".png", ".tiff", ".tif", ".jpeg", ".jpg", ".npz"]
ModelType = Literal[
    "vit_l_lm", "vit_b_lm", "vit_t_lm",
    "vit_l_em_organelles", "vit_b_em_organelles", "vit_t_em_organelles",
    "vit_b", "vit_l", "vit_h",
    "cpsam",
]
# Model types served by the isolated Cellpose-SAM runtime rather than micro-sam.
CELLPOSE_MODEL_TYPES = ("cpsam",)


def _read_pip(name: str) -> List[str]:
    text = (Path(__file__).parent / name).read_text()
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@bioengine.app(
    num_cpus=1,
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

    def __init__(self, runtime: RuntimeApp, cellpose_runtime: CellposeRuntime) -> None:
        self.runtime = runtime
        self.cellpose_runtime = cellpose_runtime
        self.start_time = time.time()
        self._hypha_token = os.getenv("HYPHA_TOKEN")
        if not self._hypha_token:
            raise RuntimeError("HYPHA_TOKEN environment variable is not set")
        self._hypha = None
        self._s3 = None
        self._http: Optional[httpx.AsyncClient] = None
        self._training_tasks: Dict[str, asyncio.Task] = {}
        # In-memory export registry (v1): export_id -> status record. Lost on
        # replica restart — the frontend re-exports if a handle goes missing.
        self._exports: Dict[str, Dict[str, Any]] = {}
        self._export_tasks: Dict[str, asyncio.Task] = {}
        # Serialize GPU fine-tuning across all runtime replicas: at most one
        # training runs at a time, so autoscaled replicas stay free to serve
        # inference during a long run. Mirrors model-runner's _env_build_lock.
        self._training_lock = asyncio.Lock()

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

    def _runtime_for(self, model_type: Optional[str]):
        """Route to the backend that owns ``model_type`` (cpsam → CellposeRuntime,
        vit_* / None → micro-sam RuntimeApp)."""
        return self.cellpose_runtime if model_type in CELLPOSE_MODEL_TYPES else self.runtime

    async def _check_runtime_available(self, model_type: Optional[str] = None) -> None:
        runtime = self._runtime_for(model_type)
        try:
            await asyncio.wait_for(runtime.ping(), timeout=2.0)
        except Exception as e:
            raise RuntimeError(
                "GPU runtime is not available yet — inference, embedding, ONNX "
                "export, and training are unavailable until it starts."
            ) from e

    # === I/O transport (S3 + URL) ===

    @bioengine.method
    async def get_upload_url(
        self,
        file_type: UPLOAD_FILE_TYPES = Field(
            ...,
            description='File type for the upload: ".npy" (NumPy array), ".png", '
            '".tiff"/".tif", ".jpeg"/".jpg", or ".npz" (a compute_embedding bundle '
            "to store yourself and feed to infer).",
        ),
    ) -> Dict[str, str]:
        """Presigned S3 PUT URL (1-hour TTL) for staging an input image or embedding."""
        file_path = f"temp/{uuid.uuid4()}{file_type}"
        upload_url = await self._s3.put_file(file_path=file_path, ttl=3600)
        return {"upload_url": upload_url, "file_path": file_path}

    async def _get_download_url(self, file_path: str) -> str:
        return await self._s3.get_file(file_path=file_path, use_proxy=True)

    # Exponential backoff (~30s total) covers a full broker commit + re-stage
    # cycle, during which a presigned URL can transiently 403/404/5xx. micro-sam
    # stays artifact-agnostic: it retries the raw HTTP transfer, never inspects
    # stage state, and never retries indefinitely.
    _HTTP_RETRY_BACKOFF = (1.0, 2.0, 4.0, 8.0, 15.0)

    async def _http_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Issue a request on a presigned URL, retrying transient failures.

        Retries on 403/404/408/429/5xx and httpx transport errors with the
        backoff above, then returns the final response WITHOUT raising for
        status — callers keep their own handling (e.g. 404 -> FileNotFoundError,
        raise_for_status on a PUT). An exhausted retry loop must surface loudly.

        If ``content`` is callable it is invoked fresh on every attempt to yield
        a new body — required for streamed uploads, whose async iterators are
        single-use and would be spent after a retry.
        """
        retry_status = {403, 404, 408, 429}
        body_factory = kwargs.pop("content", None) if callable(kwargs.get("content")) else None
        for delay in (*self._HTTP_RETRY_BACKOFF, None):
            if body_factory is not None:
                kwargs["content"] = body_factory()
            try:
                resp = await self._http.request(method, url, **kwargs)
            except httpx.TransportError:
                if delay is None:
                    raise
                await asyncio.sleep(delay)
                continue
            if delay is not None and (resp.status_code in retry_status or resp.status_code >= 500):
                await asyncio.sleep(delay)
                continue
            return resp

    @staticmethod
    async def _file_chunks(fpath: Path, size: int = 1 << 20):
        with open(fpath, "rb") as fh:
            while True:
                chunk = await asyncio.to_thread(fh.read, size)
                if not chunk:
                    break
                yield chunk

    async def _put_file(self, url: str, fpath: Path, timeout: float = 600.0) -> httpx.Response:
        """PUT a file to a presigned URL, streaming from disk so large weights
        (cpsam packages ship a ~1.2GB model_weights.pth) never load whole into
        RAM and OOM the memory-constrained EntryApp actor. Content-Length is set
        explicitly because S3 presigned PUTs reject chunked transfer encoding."""
        size = await asyncio.to_thread(lambda: fpath.stat().st_size)
        return await self._http_retry(
            "PUT", url,
            content=lambda: self._file_chunks(fpath),
            headers={"Content-Length": str(size)},
            timeout=timeout,
        )

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
        resp = await self._http_retry("GET", url)
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

    @staticmethod
    def _embedding_npz_bytes(features, input_size, original_size, model_type) -> bytes:
        buffer = BytesIO()
        np.savez(
            buffer, features=np.asarray(features),
            input_size=np.asarray(input_size), original_size=np.asarray(original_size),
            model_type=str(model_type),
        )
        return buffer.getvalue()

    async def _put_temp_embedding(self, content: bytes) -> str:
        file_path = f"temp/{uuid.uuid4()}.npz"
        upload_url = await self._s3.put_file(file_path=file_path, ttl=3600)
        resp = await self._http_retry("PUT", upload_url, content=content)
        resp.raise_for_status()
        return await self._get_download_url(file_path)

    async def _resolve_embedding(self, e: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve an infer ``embeddings`` item to ``{features, input_size,
        original_size, model_type}``. Accepts an embedding_url (.npz from
        compute_embedding) or an inline compute_embedding result dict.
        """
        if isinstance(e, str):
            url = e if e.startswith(("http://", "https://")) else await self._get_download_url(e)
            resp = await self._http_retry("GET", url)
            resp.raise_for_status()
            data = await asyncio.to_thread(np.load, BytesIO(resp.content), allow_pickle=False)
            return {
                "features": np.asarray(data["features"]),
                "input_size": [int(x) for x in np.asarray(data["input_size"]).tolist()],
                "original_size": [int(x) for x in np.asarray(data["original_size"]).tolist()],
                "model_type": str(data["model_type"]),
            }
        if isinstance(e, dict):
            original_size = e.get("original_image_shape") or e.get("original_size")
            input_size = e.get("input_size")
            if input_size is None:
                scale = float(e["sam_scale"])
                input_size = [round(original_size[0] * scale), round(original_size[1] * scale)]
            return {
                "features": np.asarray(e["features"]),
                "input_size": [int(x) for x in input_size],
                "original_size": [int(x) for x in original_size],
                "model_type": e.get("model_type"),
            }
        raise ValueError(
            "embeddings items must be an embedding_url (str) or a compute_embedding dict."
        )

    def _session_checkpoint(self, session_id: str) -> str:
        import training

        ckpt = training.checkpoint_path(session_id)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"No trained checkpoint for session '{session_id}' "
                "(training may be unfinished or failed)."
            )
        return str(ckpt)

    async def _fetch_model_package(
        self, model_id: str, model_token: Optional[str]
    ) -> Dict[str, Any]:
        """Download an exported bioimage.io micro-sam package (published or the
        caller's draft/staged) into a shared HOME cache and return
        ``{weights_path, model_type}``. The package is a combined
        ``{model_state, decoder_state}`` checkpoint served through the same AIS
        path as a fine-tuned session. ``model_token`` (the caller's own token)
        grants read access to a draft in their workspace; without it the app token
        is used, which only reaches published models.
        """
        cache_dir = Path.home() / ".bioengine" / "micro_sam_zoo_cache" / re.sub(r"[^\w.-]", "_", model_id)
        weights_path = cache_dir / "weights.pt"
        meta_path = cache_dir / "kwargs.json"
        if weights_path.exists() and meta_path.exists():
            return {"weights_path": str(weights_path), **json.loads(meta_path.read_text())}

        server = self._hypha
        if model_token:
            server = await connect_to_server({"server_url": SERVER_URL, "token": model_token})
        am = await server.get_service("public/artifact-manager")

        rdf_url = await am.get_file(model_id, file_path="rdf.yaml")
        rdf = yaml.safe_load((await self._http_retry("GET", rdf_url)).text)
        ps = rdf["weights"]["pytorch_state_dict"]
        model_type = ps["architecture"]["kwargs"]["model_type"]

        w_url = await am.get_file(model_id, file_path=ps["source"])
        content = (await self._http_retry("GET", w_url, timeout=600.0)).content
        cache_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(weights_path.write_bytes, content)
        meta = {"model_type": model_type}
        meta_path.write_text(json.dumps(meta))
        return {"weights_path": str(weights_path), **meta}

    # === serving (forwarded to the GPU runtime) ===

    @bioengine.method
    async def infer(
        self,
        input_arrays: Optional[List[Union[np.ndarray, str]]] = Field(
            None,
            description="List of input images (numpy HxW/HxWx3/3xHxW, an http(s) URL, "
            "or a get_upload_url file path). Provide this OR embeddings.",
        ),
        embeddings: Optional[List[Union[str, Dict[str, Any]]]] = Field(
            None,
            description="Run AIS on precomputed embeddings instead of images (skips "
            "the encoder). Each item is an embedding_url from "
            "compute_embedding(return_url=True), or a compute_embedding result dict. "
            "Provide this OR input_arrays.",
        ),
        model_type: ModelType = Field(
            "vit_l_lm",
            description="μSAM generalist. All six — LM (vit_t/b/l_lm) and EM organelles "
            "(vit_t/b/l_em_organelles) — carry the AIS decoder. For embeddings, the model "
            "must match the one that produced them (inferred from the embedding when available).",
        ),
        min_size: Optional[int] = Field(
            None, description="Minimum object size in pixels (smaller objects dropped)."
        ),
        session_id: Optional[str] = Field(
            None, description="Serve the fine-tuned model from this training session's checkpoint."
        ),
        model_id: Optional[str] = Field(
            None,
            description="Run an exported BioImage.IO micro-sam model by artifact id "
            "(e.g. 'bioimage-io/<alias>') — published, or your own draft/staged. "
            "Downloads the combined SAM+decoder package and runs its AIS decoder → "
            "instances. Requires input_arrays (not embeddings).",
        ),
        model_token: Optional[str] = Field(
            None,
            description="Hypha token with read access to model_id — needed for a "
            "draft/staged model in your workspace; omit for published models.",
        ),
    ) -> List[Dict[str, Any]]:
        """Automatic instance segmentation (μSAM AIS, or Cellpose-SAM when
        ``model_type='cpsam'`` / a cpsam session/model). Returns a bare list, one
        item per input, each ``{"output": <int32 [H,W] instance label mask>}``.
        Pass ``input_arrays`` (images) OR ``embeddings`` (micro-sam only —
        precomputed, runs the AIS decoder without re-encoding). Pass ``model_id``
        to run an exported BioImage.IO package instead of a built-in/finetuned model.
        """
        import training

        generate_kwargs: Dict[str, Any] = {}
        if min_size is not None:
            generate_kwargs["min_size"] = min_size
        if model_id is not None:
            if input_arrays is None:
                raise ValueError("model_id requires input_arrays (images), not embeddings.")
            pkg = await self._fetch_model_package(model_id, model_token)
            await self._check_runtime_available(pkg["model_type"])
            images = [await self._resolve_image(src) for src in input_arrays]
            return await self._runtime_for(pkg["model_type"]).auto_segment(
                images=images, model_type=pkg["model_type"],
                checkpoint=pkg["weights_path"], generate_kwargs=generate_kwargs,
            )
        if (input_arrays is None) == (embeddings is None):
            raise ValueError("Provide exactly one of 'input_arrays' or 'embeddings'.")
        checkpoint = self._session_checkpoint(session_id) if session_id else None
        if embeddings is not None:
            # Embeddings are a micro-sam-only path (cpsam has no encoder embedding).
            await self._check_runtime_available()
            embs = [await self._resolve_embedding(e) for e in embeddings]
            emb_model = next((e["model_type"] for e in embs if e.get("model_type")), None)
            return await self.runtime.auto_segment_from_embedding(
                embeddings=embs, model_type=emb_model or model_type,
                generate_kwargs=generate_kwargs, checkpoint=checkpoint,
            )
        # A fine-tuned session's backend fixes the runtime; otherwise route by model_type.
        served_type = (
            "cpsam" if session_id and training.session_backend(session_id) == "cellpose"
            else model_type
        )
        await self._check_runtime_available(served_type)
        if served_type == "cpsam" and session_id:
            # Serve at the diameter the session trained/exports at, so live GPU
            # masks match the exported package's CPU self-test.
            diam = training.read_training_params(session_id).get("diam_mean")
            if diam is not None:
                generate_kwargs["diameter"] = diam
        images = [await self._resolve_image(src) for src in input_arrays]
        return await self._runtime_for(served_type).auto_segment(
            images=images, model_type=served_type,
            generate_kwargs=generate_kwargs, checkpoint=checkpoint,
        )

    @bioengine.method
    async def compute_embedding(
        self,
        inputs: Union[np.ndarray, str] = Field(..., description="A single input image."),
        model_type: ModelType = Field("vit_l_lm", description="μSAM model."),
        return_url: bool = Field(
            False,
            description="If True, save the embedding as a self-contained .npz in a "
            "temporary S3 file and return its download URL as 'embedding_url' (feed it "
            "to infer(embeddings=[...])) instead of the inline 'features' ndarray.",
        ),
        embedding_upload_url: Optional[str] = Field(
            None,
            description="Presigned PUT URL (from get_upload_url('.npz')) to store the "
            "embedding .npz at, instead of a temporary S3 file. You keep the matching "
            "download URL to pass to infer(embeddings=[...]).",
        ),
        session_id: Optional[str] = Field(None, description="Serve a fine-tuned session checkpoint."),
    ) -> Dict[str, Any]:
        """Run the encoder once and return its embedding: ``{features |
        embedding_url, original_image_shape, input_size, sam_scale, mask_threshold,
        model_type}``. The embedding feeds the in-browser ONNX prompt decoder and can
        be passed to ``infer(embeddings=[...])`` to run AIS without re-encoding. With
        ``embedding_upload_url`` the .npz is PUT to your URL and ``features`` is
        dropped from the return.
        """
        if model_type in CELLPOSE_MODEL_TYPES:
            raise ValueError("compute_embedding is micro-sam only; cpsam has no embedding path.")
        await self._check_runtime_available()
        checkpoint = self._session_checkpoint(session_id) if session_id else None
        image = await self._resolve_image(inputs)
        payload = await self.runtime.encode(
            image=image, model_type=model_type, checkpoint=checkpoint,
        )
        payload["model_type"] = model_type
        if embedding_upload_url or return_url:
            feat = payload.pop("features")
            arr = np.frombuffer(feat["_rvalue"], dtype=feat["_rdtype"]).reshape(feat["_rshape"])
            bundle = self._embedding_npz_bytes(
                arr, payload["input_size"], payload["original_image_shape"], model_type
            )
            if embedding_upload_url:
                resp = await self._http_retry("PUT", embedding_upload_url, content=bundle)
                resp.raise_for_status()
            else:
                payload["embedding_url"] = await self._put_temp_embedding(bundle)
        return payload

    @bioengine.method
    async def get_onnx_model(
        self,
        model_type: ModelType = Field("vit_l_lm", description="μSAM model."),
        quantize: bool = Field(True, description="Quantize for faster browser runtime."),
    ) -> bytes:
        """The interactive prompt decoder as ONNX bytes for onnxruntime-web."""
        if model_type in CELLPOSE_MODEL_TYPES:
            raise ValueError("get_onnx_model is micro-sam only; cpsam has no ONNX prompt decoder.")
        await self._check_runtime_available()
        return await self.runtime.export_onnx(model_type=model_type, quantize=quantize)

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
            if data.get("val_reused_train"):
                training.write_status(session_id, val_reused_train=True)
            resolved = {}
            resume_id = params.get("resume_session_id")
            if resume_id:
                prior_model = training.read_training_params(resume_id).get("model_type")
                if prior_model and prior_model != model_type:
                    raise ValueError(
                        f"resume_session_id '{resume_id}' was trained as {prior_model}, "
                        f"not {model_type}; model_type must match to resume."
                    )
                resolved["checkpoint_path"] = self._session_checkpoint(resume_id)
            training.write_training_params(session_id, {
                **params, "model_type": model_type,
                "patch_shape": list(data["patch_shape"]),
                "train_images": data["train_images"], "train_labels": data["train_labels"],
                "val_images": data["val_images"], "val_labels": data["val_labels"],
                **resolved,
            })
            # Long-running: this await holds one GPU runtime replica for the whole
            # training run, so a concurrent infer is routed to a second replica.
            # The entry lock caps concurrency at one training regardless of how
            # many replicas exist, leaving the rest free for inference. QUEUED is
            # exempt from the PREPARING/TRAINING stale-window sweep, so a session
            # waiting here is not false-flagged STOPPED.
            if self._training_lock.locked():
                training.write_status(
                    session_id, status="QUEUED",
                    message="waiting for GPU — another fine-tuning is in progress",
                )
            async with self._training_lock:
                await self._runtime_for(model_type).train(
                    session_id=session_id, model_type=model_type, params=params
                )
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
        model_type: ModelType = Field(
            "vit_l_lm",
            description="Base model to fine-tune. vit_* → micro-sam (AIS decoder); "
            "'cpsam' → Cellpose-SAM (isolated runtime)."),
        n_epochs: int = Field(5, description="Number of training epochs."),
        n_objects_per_batch: int = Field(
            8, description="micro-sam only: objects per batch — main GPU-memory knob; 8 fits vit_b on 24GB."),
        patch_size: int = Field(512, description="micro-sam only: square training patch side (clamped to the smallest image)."),
        diam_mean: float = Field(30.0, description="cpsam only: mean object diameter in pixels."),
        batch_size: int = Field(1, description="Training batch size."),
        learning_rate: float = Field(1e-5, description="AdamW learning rate."),
        val_fraction: float = Field(0.2, description="Val split fraction when val is omitted."),
        n_samples: Optional[int] = Field(None, description="Patches sampled per epoch (auto if omitted)."),
        resume_session_id: Optional[str] = Field(
            None, description="Continue fine-tuning from a prior session's checkpoint (same model_type)."),
        label: str = Field("", description="Optional human-readable tag for this session."),
    ) -> Dict[str, Any]:
        """Start a μSAM fine-tuning session (AIS decoder) and return immediately
        with the status incl. ``session_id``. Data prep runs on this CPU entry;
        the training itself runs in a subprocess on a GPU runtime replica. Poll
        ``get_training_status`` and, once ``checkpoint_available``, serve via
        ``infer(session_id=...)``.
        """
        import training

        await self._check_runtime_available(model_type)
        backend = "cellpose" if model_type in CELLPOSE_MODEL_TYPES else "microsam"
        session_id = training.new_session_id()
        training.write_status(
            session_id, status="PREPARING", created_at=time.time(),
            model_type=model_type, backend=backend, label=label,
            n_train_inputs=len(train_images),
        )
        params = dict(
            n_epochs=n_epochs, n_objects_per_batch=n_objects_per_batch,
            batch_size=batch_size, learning_rate=learning_rate, diam_mean=diam_mean,
            val_fraction=val_fraction, n_samples=n_samples,
            resume_session_id=resume_session_id,
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
        """Fine-tuning session status (PREPARING/QUEUED/TRAINING/COMPLETED/FAILED/
        STOPPED), elapsed time, and whether the checkpoint is servable. QUEUED
        means another fine-tuning holds the single training slot."""
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
    async def export_model(
        self,
        session_id: str = Field(..., description="COMPLETED fine-tuning session to export."),
        name: str = Field(..., description="Model name for the BioImage.IO package."),
        description: str = Field("", description="Model description for the RDF."),
        authors: Optional[List[Dict[str, Any]]] = Field(
            None,
            description="Authors, each {name, affiliation?, email?, github_user?, orcid?}.",
        ),
        license: str = Field("CC-BY-4.0", description="SPDX license id for the model."),
        provenance: Optional[Dict[str, Any]] = Field(
            None,
            description="Baked server-side into RDF config.microsam_provenance, e.g. "
            "{dataset_artifact_id, label, split_name, session_lineage}.",
        ),
    ) -> Dict[str, Any]:
        """Start building a BioImage.IO model package from a COMPLETED fine-tuning
        session — a **combined SAM+decoder** package (interactive prompt head + AIS
        decoder, via ``micro_sam.bioimageio.export_sam_model``) for micro-sam
        sessions, or a **Cellpose-SAM pytorch_state_dict** package for cpsam
        sessions. Async, like start_training: returns immediately with an
        ``export_id`` — poll ``get_export_status``. The package is self-tested on
        CPU (``bioimageio.core.test_model``), built server-side, and staged on
        temporary storage; this method **publishes nothing**. The frontend creates
        the draft artifact with the user's own token and either downloads the zip
        from ``download_url`` or calls ``push_export`` to stream the package files
        straight into the draft. ``license`` applies to micro-sam packages only
        (cpsam packages carry Cellpose's BSD-3-Clause).
        """
        import training

        self._session_checkpoint(session_id)  # raises if no trained checkpoint
        served = "cpsam" if training.session_backend(session_id) == "cellpose" else None
        await self._check_runtime_available(served)
        export_id = uuid.uuid4().hex
        request = {
            "name": name,
            "description": description,
            "authors": authors,
            "license": license,
            "provenance": provenance,
        }
        self._exports[export_id] = {
            "export_id": export_id, "status": "PENDING", "progress": 0.0,
            "message": "queued", "download_url": None, "size_bytes": None, "error": None,
        }
        self._export_tasks[export_id] = asyncio.create_task(
            self._run_export(export_id, session_id, request)
        )
        return {"export_id": export_id, "status": "PENDING"}

    async def _run_export(self, export_id: str, session_id: str, request: Dict[str, Any]) -> None:
        import training

        rec = self._exports[export_id]
        try:
            rec.update(status="BUILDING", progress=0.1, message="building package")
            served = "cpsam" if training.session_backend(session_id) == "cellpose" else None
            res = await self._runtime_for(served).export_bioimageio(
                session_id=session_id, request=request
            )
            rec["_package_dir"] = res["package_dir"]
            rec.update(progress=0.8, message="staging package")
            zip_path = Path(res["zip"])
            file_path = f"temp/{export_id}/{zip_path.name}"
            upload_url = await self._s3.put_file(file_path=file_path, ttl=6 * 3600)
            resp = await self._put_file(upload_url, zip_path, timeout=600.0)
            resp.raise_for_status()
            download_url = await self._get_download_url(file_path)
            rec.update(
                status="READY", progress=1.0, message="ready",
                download_url=download_url, size_bytes=res["zip_size"], files=res["files"],
            )
        except Exception as e:
            logger.exception(f"Export {export_id} failed")
            rec.update(status="FAILED", progress=1.0, message="failed", error=str(e))

    @bioengine.method
    async def get_export_status(
        self, export_id: str = Field(..., description="export_id from export_model.")
    ) -> Dict[str, Any]:
        """Export progress. ``status`` ∈ {PENDING, BUILDING, READY, FAILED}.
        ``download_url`` (6h TTL) + ``size_bytes`` are set on READY; ``error`` on
        FAILED. ``files`` lists the package members (name + size) so the frontend
        can mirror them when creating the draft artifact and calling push_export."""
        rec = self._exports.get(export_id)
        if rec is None:
            raise KeyError(f"Unknown export_id '{export_id}'.")
        return {k: v for k, v in rec.items() if not k.startswith("_")}

    @bioengine.method
    async def push_export(
        self,
        export_id: str = Field(..., description="A READY export_id from export_model."),
        files: Dict[str, str] = Field(
            ...,
            description="Map of package file name -> presigned PUT URL. Keys must "
            "match the 'files' from get_export_status; mint the URLs via the draft "
            "artifact's put_file with the user's token.",
        ),
    ) -> Dict[str, Any]:
        """Stream the built package files straight into a draft artifact so the
        browser never moves the ~350 MB twice. The frontend creates the staged
        draft with the user's token, mints a presigned put_file URL per package
        file, then calls this. Draft-only: this app never creates, commits, or
        reads the artifact."""
        rec = self._exports.get(export_id)
        if rec is None:
            raise KeyError(f"Unknown export_id '{export_id}'.")
        if rec.get("status") != "READY":
            raise RuntimeError(f"Export '{export_id}' is not READY (status={rec.get('status')}).")
        pkg_dir = Path(rec["_package_dir"])
        pushed = []
        for rel, url in files.items():
            fpath = pkg_dir / rel
            if not fpath.is_file():
                raise FileNotFoundError(f"Package file '{rel}' not found for export '{export_id}'.")
            resp = await self._put_file(url, fpath, timeout=600.0)
            resp.raise_for_status()
            pushed.append(rel)
        return {"pushed": pushed, "n_files": len(pushed)}
