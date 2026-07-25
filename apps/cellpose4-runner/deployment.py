"""Cellpose4 Runner — run Cellpose-4 bioimage.io models via bioimageio.core.

A deliberately small, PyTorch-only alternative to ``model-runner``, tailored to
Cellpose-4 models (Cellpose-SAM, Cellpose-DINO). One model stays resident; a
new ``model_id`` frees the previous pipeline and loads the new one. Inference
runs in-process through ``bioimageio.core``'s prediction pipeline.

Import rule: this module holds ``@bioengine.app`` and is introspected by the
worker with only ``bioengine[worker]`` + the standard library installed, so the
heavy deps (``torch``, ``cellpose``, ``bioimageio.core``) are imported inside
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
from typing import Dict, List, Literal, Optional, Union

import bioengine
import httpx
import numpy as np
from hypha_rpc import connect_to_server
from pydantic import Field

logger = bioengine.logger

SERVER_URL = "https://hypha.aicell.io"
SUPPORTED_FILE_TYPES = Literal[".npy", ".png", ".tiff", ".tif", ".jpeg", ".jpg"]


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
    memory_mb=12 * 1024,
    pip=_read_pip("requirements-deployment.txt"),
    # Requests queue at the replica behind ``_gpu_lock`` so the Ray Serve
    # autoscaler can observe backlog (it only counts requests that reached
    # a replica, not those waiting at the router). Same pattern as
    # apps/model-runner/runtime.py.
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
class Cellpose4Runner:
    """Runs Cellpose-4 bioimage.io models, one resident at a time."""

    def __init__(self) -> None:
        self.start_time = time.time()
        self._hypha_token = os.getenv("HYPHA_TOKEN")
        if not self._hypha_token:
            raise RuntimeError("HYPHA_TOKEN environment variable is not set")

        # Serialises GPU work so only one request touches the GPU (and the
        # single resident pipeline) at a time. The extra RPCs queued on the
        # lock are what the autoscaler measures.
        self._gpu_lock = asyncio.Lock()
        self._pipeline = None
        self._loaded_model_id: Optional[str] = None

        self._hypha = None
        self._s3 = None
        self._http: Optional[httpx.AsyncClient] = None

        # bioimageio.spec/core loggers are disabled by default (library
        # convention); enable their loguru sink so per-weight-format progress
        # and weight downloads surface in the replica's stderr.
        try:
            from loguru import logger as _loguru_logger

            _loguru_logger.enable("bioimageio")
        except Exception as e:
            logger.warning(f"Could not enable bioimageio loguru sink: {e}")

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

    # === I/O transport (S3 + URL), mirrors model-runner ===

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
        ``file_path`` as the ``inputs`` parameter of ``infer``.

        Returns:
            Dict with ``upload_url`` (presigned PUT URL) and ``file_path``
            (reference to pass to ``infer``).
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
        if source.startswith(("http://", "https://")):
            url = source
        else:
            url = await self._get_download_url(source)

        resp = await self._http.get(url)
        if resp.status_code == 404:
            raise FileNotFoundError(f"Source '{source}' does not exist or has expired.")
        resp.raise_for_status()

        buffer = BytesIO(resp.content)
        if ext == ".npy":
            return await asyncio.to_thread(np.load, buffer)
        import imageio.v3 as iio

        return await asyncio.to_thread(iio.imread, buffer)

    async def _save_array_to_temp_file(self, array: np.ndarray) -> str:
        """Save an array to a temporary ``.npy`` in S3, return a download URL."""
        info = await self.get_upload_url(file_type=".npy")
        buffer = BytesIO()
        np.save(buffer, array)
        await self._http.put(info["upload_url"], content=buffer.getvalue())
        return await self._get_download_url(info["file_path"])

    # === Inference ===

    def _run_inference(
        self,
        model_id: str,
        device: str,
        inputs: np.ndarray,
        sample_id: str,
    ) -> Dict[str, np.ndarray]:
        """Load the model (reusing the resident pipeline when unchanged) and
        run the forward pass. Blocking — call via ``asyncio.to_thread`` while
        holding ``self._gpu_lock``.
        """
        from bioimageio.core import (
            create_prediction_pipeline,
            load_model_description,
        )
        from bioimageio.core.digest_spec import create_sample_for_model

        if model_id != self._loaded_model_id:
            # Free the previous model's VRAM before loading the next. A single
            # framework (torch) means empty_cache after dropping the refs is
            # enough — no subprocess isolation needed.
            self._pipeline = None
            self._loaded_model_id = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            logger.info(f"🔄 Loading model '{model_id}' on {device}...")
            descr = load_model_description(model_id)
            pipeline = create_prediction_pipeline(
                descr,
                weights_format="pytorch_state_dict",
                device=device,
            )
            pipeline.load()
            self._pipeline = pipeline
            self._loaded_model_id = model_id
            logger.info(f"✅ Model '{model_id}' loaded.")

        sample = create_sample_for_model(
            self._pipeline.model_description,
            inputs=inputs,
            sample_id=sample_id,
        )
        # Cellpose-4 models declare a fixed input size with a halo, so they
        # must be run tiled: blocking feeds each network call exactly the
        # model's input size (halo as tile overlap). Running the whole sample
        # unblocked pads it by the halo and breaks the ViT positional embed.
        result = self._pipeline.predict_sample_with_blocking(sample)
        return {str(key): member.data.data for key, member in result.members.items()}

    @bioengine.method
    async def infer(
        self,
        model_id: str = Field(
            ...,
            description="bioimage.io id (or full rdf URL) of a Cellpose-4 model to "
            "run. Supported published models: 'idealistic-eagle' (Cellpose-SAM, "
            "3-channel input, float32 label output). In review / staged (pass the "
            "staged rdf URL): 'passionate-bug' (Cellpose-DINO ViT-B, 1-channel "
            "input, uint16 label output). Any Cellpose-4 bioimage.io model with a "
            "pytorch_state_dict weight resolvable by this id should work.",
        ),
        inputs: Union[np.ndarray, str] = Field(
            ...,
            description="Input image as a numpy array, or a string: a direct "
            "http(s) URL, or a temporary file path from ``get_upload_url``. Shape "
            "and channel count must match the model's input spec (e.g. 3 channels "
            "for Cellpose-SAM, 1 for Cellpose-DINO).",
        ),
        device: Literal["cuda", "cpu"] = Field(
            "cuda", description="Computation device."
        ),
        sample_id: str = Field(
            "sample", description="Identifier for this request, used in logging."
        ),
        return_download_url: bool = Field(
            False,
            description="If True, each output array is saved to a temporary .npy "
            "in S3 and returned as a presigned download URL (valid 1 hour) instead "
            "of the raw array.",
        ),
    ) -> Dict[str, Union[np.ndarray, str]]:
        """Run a Cellpose-4 model's forward pass and return the label image(s).

        The model is loaded on first use and kept resident; passing a different
        ``model_id`` frees the previous model and loads the new one.

        Returns:
            Dict mapping each model output id (e.g. ``"labels"``) to the result
            array, or to a presigned download URL when ``return_download_url``.
        """
        if isinstance(inputs, str):
            inputs = await self._load_image_from_source(inputs)
        if not isinstance(inputs, np.ndarray):
            raise ValueError(
                "inputs must be a numpy array or a URL / get_upload_url file path."
            )

        logger.info(f"🤖 Running inference for model '{model_id}'...")
        async with self._gpu_lock:
            outputs = await asyncio.to_thread(
                self._run_inference, model_id, device, inputs, sample_id
            )

        if return_download_url:
            return {
                key: await self._save_array_to_temp_file(array)
                for key, array in outputs.items()
            }
        return outputs

    @bioengine.method
    async def ping(self) -> Dict[str, Union[str, float, None]]:
        """Return service status and the currently resident model id."""
        return {
            "status": "ok",
            "loaded_model_id": self._loaded_model_id,
            "uptime": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat(),
        }
