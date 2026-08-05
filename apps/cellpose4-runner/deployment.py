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

# bioimage.io ids this service will run. cellpose4-runner is an inference-only
# service for Cellpose-4 models; any id outside this set is rejected. Extend as
# new Cellpose-4 models are verified.
SUPPORTED_MODELS = ("idealistic-eagle",)


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
    gpu_memory_mb=-1,
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

    # Finished infer jobs are held this long (seconds) so a slow poller can
    # still read the result, then swept from the in-memory registry.
    _INFER_JOBS_TTL_SEC = 3600

    def __init__(self) -> None:
        self._hypha_token = os.getenv("HYPHA_TOKEN")
        if not self._hypha_token:
            raise RuntimeError("HYPHA_TOKEN environment variable is not set")

        # Serialises GPU work so only one request touches the GPU (and the
        # single resident pipeline) at a time. The extra RPCs queued on the
        # lock are what the autoscaler measures.
        self._gpu_lock = asyncio.Lock()
        self._pipeline = None
        self._loaded_model_id: Optional[str] = None
        self._loaded_overrides: Optional[tuple] = None

        # Async infer-job registry (mirrors model-runner's submit/poll API).
        # infer() returns a request_id and runs the work as a background task;
        # get_infer_status(request_id) polls this dict. In-memory per replica.
        self._infer_jobs: Dict[str, dict] = {}

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

    @staticmethod
    def _apply_postprocessing_overrides(descr, overrides: Dict[str, float]):
        """Return a deep copy of ``descr`` with Cellpose flow-dynamics
        postprocessing kwargs overridden. The on-disk / artifact RDF is never
        mutated — only the returned in-memory copy carries the overrides.

        ``overrides`` maps ``flow_threshold`` / ``cellprob_threshold`` /
        ``min_size`` to values; ``None`` entries are ignored. If nothing to
        override, ``descr`` is returned unchanged.
        """
        updates = {k: v for k, v in overrides.items() if v is not None}
        if not updates:
            return descr

        patched = descr.model_copy(deep=True)
        for output in patched.outputs:
            pps = getattr(output, "postprocessing", None) or []
            new_pps = list(pps)
            changed = False
            for i, pp in enumerate(new_pps):
                if "cellpose" in str(getattr(pp, "id", "")).lower():
                    new_kwargs = pp.kwargs.model_copy(update=updates)
                    new_pps[i] = pp.model_copy(update={"kwargs": new_kwargs})
                    changed = True
            if changed:
                object.__setattr__(output, "postprocessing", type(pps)(new_pps))
        return patched

    def _run_inference(
        self,
        model_id: str,
        inputs: np.ndarray,
        sample_id: str,
        overrides: Dict[str, float],
        return_flows: bool,
        two_pass: bool,
    ) -> Dict[str, np.ndarray]:
        """Load the model (reusing the resident pipeline when unchanged) and
        run the forward pass. Blocking — call via ``asyncio.to_thread`` while
        holding ``self._gpu_lock``.
        """
        device = "cuda"
        from bioimageio.core import (
            create_prediction_pipeline,
            load_model_description,
        )
        from bioimageio.core.digest_spec import create_sample_for_model

        # Postprocessing overrides change the pipeline, so they are part of the
        # resident-pipeline cache key: a new model_id OR a new override set
        # forces a reload.
        override_key = tuple(
            sorted((k, v) for k, v in overrides.items() if v is not None)
        )
        if model_id != self._loaded_model_id or override_key != self._loaded_overrides:
            # Free the previous model's VRAM before loading the next. A single
            # framework (torch) means empty_cache after dropping the refs is
            # enough — no subprocess isolation needed.
            self._pipeline = None
            self._loaded_model_id = None
            self._loaded_overrides = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            logger.info(
                f"🔄 Loading model '{model_id}' on {device} "
                f"(overrides={dict(override_key)})..."
            )
            descr = load_model_description(model_id)
            descr = self._apply_postprocessing_overrides(descr, overrides)
            pipeline = create_prediction_pipeline(
                descr,
                weights_format="pytorch_state_dict",
                device=device,
            )
            pipeline.load()
            self._pipeline = pipeline
            self._loaded_model_id = model_id
            self._loaded_overrides = override_key
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
        #
        # ``skip_postprocessing`` drops the samplewise ``cellpose_flow_dynamics``
        # op (which turns the stitched 3-channel flow+cellprob tensor into
        # instance labels), returning the raw flow field instead. It runs after
        # tile stitching, so skipping it yields a correctly stitched flow field.
        if two_pass:
            # First pass: image -> raw flow field, postprocessing skipped. Feed
            # that 3-channel flow field back through the model as the input of
            # the second pass, which carries the postprocessing (unless the
            # caller wants the raw second-pass flow field via return_flows).
            first = self._pipeline.predict_sample_with_blocking(
                sample, skip_postprocessing=True
            )
            flow = next(iter(first.members.values())).data.data
            sample = create_sample_for_model(
                self._pipeline.model_description,
                inputs=flow,
                sample_id=sample_id,
            )

        result = self._pipeline.predict_sample_with_blocking(
            sample, skip_postprocessing=return_flows
        )
        members = {
            str(key): member.data.data for key, member in result.members.items()
        }
        # The single output member id is "labels"; when returning flows it holds
        # the 3-channel field (2 flow components + cell probability), so relabel
        # it for callers.
        if return_flows and len(members) == 1:
            members = {"flows": next(iter(members.values()))}
        return members

    # === Async infer-job registry (mirrors model-runner submit/poll) ===

    def _sweep_expired_jobs(self) -> None:
        """Drop finished infer jobs older than ``_INFER_JOBS_TTL_SEC``. Runs
        opportunistically when a new job is created.
        """
        now = time.time()
        expired = [
            rid
            for rid, job in self._infer_jobs.items()
            if job["completed_at"] is not None
            and (now - job["completed_at"]) > self._INFER_JOBS_TTL_SEC
        ]
        for rid in expired:
            self._infer_jobs.pop(rid, None)

    def _new_job(self, model_id: str) -> dict:
        self._sweep_expired_jobs()
        request_id = f"ij-{uuid.uuid4().hex[:12]}"
        job = {
            "job_id": request_id,
            "model_id": model_id,
            "state": "queued",
            "submitted_at": time.time(),
            "running_ts": None,
            "completed_at": None,
            "result": None,
            # asyncio.Task handle, filled in by infer(); used by cancel_request
            # to drop a still-queued job before it acquires the GPU. Never
            # serialised out — _job_progress builds its own dict.
            "task": None,
        }
        self._infer_jobs[request_id] = job
        return job

    def _queue_position(self, job: dict) -> int:
        """0-based position in the GPU queue: 0 = running/done, N = N jobs
        ahead. The single ``_gpu_lock`` serialises work, so this is just the
        count of not-yet-finished jobs submitted before this one.
        """
        if job["state"] in ("completed", "failed", "cancelled"):
            return 0
        return sum(
            1
            for other in self._infer_jobs.values()
            if other["state"] not in ("completed", "failed", "cancelled")
            and other["submitted_at"] < job["submitted_at"]
        )

    def _job_progress(self, job: dict) -> dict:
        """Progress dict for an infer request. Same top-level shape as
        model-runner's ``get_infer_status`` so a single poller works against
        both services. ``model_download`` / ``env_setup`` are always None here
        (no separate download or per-model env-build stage) and kept for schema
        parity; the GPU forward is the only stage.
        """
        pos = self._queue_position(job)
        return {
            "queue_position": pos,
            "submitted_at": job["submitted_at"],
            "model_download": None,
            "env_setup": None,
            "running": job["running_ts"],
            "completed_at": job["completed_at"],
            "result": job["result"],
            "stages": {
                "model_download": {"start": None, "end": None},
                "env_setup": {"start": None, "end": None, "queue_position": None},
                "run": {
                    "start": job["running_ts"],
                    "end": job["completed_at"],
                    "queue_position": pos if job["state"] == "running" else None,
                },
            },
        }

    async def _execute_infer(
        self,
        job: dict,
        model_id: str,
        inputs: np.ndarray,
        sample_id: str,
        overrides: Dict[str, float],
        return_flows: bool,
        two_pass: bool,
        return_download_url: bool,
    ) -> None:
        """Background driver: wait for the GPU, run the forward pass, store the
        result on the job. Any exception is captured onto the job as an error
        rather than surfacing through asyncio's default handler.
        """
        request_id = job["job_id"]
        try:
            async with self._gpu_lock:
                job["state"] = "running"
                job["running_ts"] = time.time()
                outputs = await asyncio.to_thread(
                    self._run_inference,
                    model_id,
                    inputs,
                    sample_id,
                    overrides,
                    return_flows,
                    two_pass,
                )
            if return_download_url:
                outputs = {
                    key: await self._save_array_to_temp_file(array)
                    for key, array in outputs.items()
                }
            job["result"] = outputs
            job["state"] = "completed"
            job["completed_at"] = time.time()
            logger.info(f"✅ Inference completed for request {request_id!r}.")
        except asyncio.CancelledError:
            # cancel_request cancelled this job while it waited for the GPU
            # (it already set the terminal state). Swallow and don't run.
            logger.info(f"🚫 Infer request {request_id!r} cancelled before execution.")
            if job["completed_at"] is None:
                job["state"] = "cancelled"
                job["result"] = {"error": "cancelled"}
                job["completed_at"] = time.time()
            return
        except Exception as exc:
            logger.error(
                f"❌ Infer request {request_id!r} for '{model_id}' failed: {exc}"
            )
            job["result"] = {"error": str(exc)}
            job["state"] = "failed"
            job["completed_at"] = time.time()

    @bioengine.method
    async def list_supported_models(self) -> List[str]:
        """Return the bioimage.io model ids this service can run.

        cellpose4-runner is inference-only and accepts only these ids; any other
        ``model_id`` passed to ``infer`` is rejected. Use this to discover the
        current supported set (e.g. before routing a Cellpose-4 model here).
        """
        return list(SUPPORTED_MODELS)

    @bioengine.method
    async def infer(
        self,
        model_id: str = Field(
            ...,
            description="bioimage.io id of a supported Cellpose-4 model. Only the "
            "ids returned by ``list_supported_models`` are accepted (currently "
            "'idealistic-eagle' = Cellpose-SAM, 3-channel input, label output); "
            "any other id is rejected.",
        ),
        inputs: Union[np.ndarray, str] = Field(
            ...,
            description="Input image as a numpy array, or a string: a direct "
            "http(s) URL, or a temporary file path from ``get_upload_url``. Shape "
            "and channel count must match the model's input spec (3 channels for "
            "Cellpose-SAM).",
        ),
        sample_id: str = Field(
            "sample", description="Identifier for this request, used in logging."
        ),
        flow_threshold: Optional[float] = Field(
            None,
            description="Cellpose flow-dynamics postprocessing override. Max "
            "allowed error between predicted and reconstructed flows; higher "
            "keeps more (possibly lower-quality) masks. When None, the model's "
            "RDF default is used (0.4 for Cellpose-SAM). Only the in-memory RDF "
            "copy is patched; the artifact is never modified. Ignored when "
            "``return_flows`` is True.",
        ),
        cellprob_threshold: Optional[float] = Field(
            None,
            description="Cellpose flow-dynamics postprocessing override. Cell "
            "probability threshold; lower detects more (dimmer) cells, higher "
            "detects fewer. When None, the model's RDF default is used (0.0 for "
            "Cellpose-SAM). Ignored when ``return_flows`` is True.",
        ),
        min_size: Optional[int] = Field(
            None,
            description="Cellpose flow-dynamics postprocessing override. Minimum "
            "number of pixels per mask; smaller masks are discarded. When None, "
            "the model's RDF default is used (15 for Cellpose-SAM). Ignored when "
            "``return_flows`` is True.",
        ),
        return_flows: bool = Field(
            False,
            description="If True, skip the Cellpose flow-dynamics postprocessing "
            "and return the raw flow field instead of instance masks: a single "
            "``flows`` output with 3 channels (2 flow components + cell "
            "probability). The flow-dynamics overrides above do not apply in this "
            "mode.",
        ),
        two_pass: bool = Field(
            False,
            description="If True, run the model twice: the first pass maps the "
            "image to a raw flow field (postprocessing skipped) and the second "
            "pass feeds that flow field back through the model as input. The "
            "flow-dynamics postprocessing (with any overrides above) is applied "
            "on the second pass — unless ``return_flows`` is True, in which case "
            "the raw second-pass flow field is returned instead.",
        ),
        return_download_url: bool = Field(
            False,
            description="If True, each output array is saved to a temporary .npy "
            "in S3 and returned as a presigned download URL (valid 1 hour) instead "
            "of the raw array.",
        ),
    ) -> str:
        """Submit an inference request and return a ``request_id`` immediately.

        URL / file-path inputs are resolved to a numpy array up front (so a
        broken source surfaces synchronously), then the model download + GPU
        forward run as a background job. Poll ``get_infer_status(request_id)``
        for progress; once its ``result`` is populated it holds the output dict
        (or ``{"error": ...}`` on failure). The resident model is loaded on
        first use and kept; a different override set reloads the pipeline.

        Returns:
            The ``request_id`` string (e.g. ``"ij-…"``) to poll with
            ``get_infer_status``.

        Raises:
            ValueError: if ``model_id`` is not in ``list_supported_models``, or
                the resolved inputs are not a numpy array.
            FileNotFoundError: if a URL / file path does not exist or has expired.
        """
        if model_id not in SUPPORTED_MODELS:
            raise ValueError(
                f"model_id {model_id!r} is not supported. cellpose4-runner runs "
                f"only {list(SUPPORTED_MODELS)}; call list_supported_models() for "
                f"the current set."
            )
        if isinstance(inputs, str):
            inputs = await self._load_image_from_source(inputs)
        if not isinstance(inputs, np.ndarray):
            raise ValueError(
                "inputs must be a numpy array or a URL / get_upload_url file path."
            )

        overrides = {
            "flow_threshold": flow_threshold,
            "cellprob_threshold": cellprob_threshold,
            "min_size": min_size,
        }
        job = self._new_job(model_id)
        logger.info(
            f"🤖 Queued inference request {job['job_id']!r} for model "
            f"'{model_id}'..."
        )
        job["task"] = asyncio.create_task(
            self._execute_infer(
                job,
                model_id,
                inputs,
                sample_id,
                overrides,
                return_flows,
                two_pass,
                return_download_url,
            )
        )
        return job["job_id"]

    @bioengine.method
    async def get_infer_status(
        self,
        request_id: str = Field(
            ..., description="Request id returned by ``infer()``."
        ),
    ) -> Dict[str, Union[int, float, dict, None]]:
        """Return the progress dict for an infer request.

        Response shape (mirrors model-runner's ``get_infer_status``)::

            {
              "queue_position": int,          # 0 = running/done, N = N jobs ahead
              "submitted_at":   float,        # ts when queued
              "model_download": None,         # no separate stage here
              "env_setup":      None,         # no per-model env build here
              "running":        float | None, # ts when GPU work started
              "completed_at":   float | None, # ts when finished, None until then
              "result":         dict | None,  # output dict on success,
                                              # {"error": str} on failure
              "stages": {                     # per-stage timeline, schema parity
                "model_download": {"start": None, "end": None},
                "env_setup": {"start": None, "end": None, "queue_position": None},
                "run": {"start": float|None, "end": float|None,
                        "queue_position": int|None},
              },
            }

        Jobs live in-memory per replica and are held for 1 hour after
        completion, then swept. Unknown ids raise ``KeyError``.
        """
        job = self._infer_jobs.get(request_id)
        if job is None:
            raise KeyError(
                f"Unknown request_id {request_id!r}. Requests live in-memory per "
                f"replica and expire {self._INFER_JOBS_TTL_SEC // 60} minutes "
                f"after completion. Start a fresh request via infer()."
            )
        return self._job_progress(job)

    @bioengine.method
    async def cancel_request(
        self,
        request_id: str = Field(
            ..., description="Request id returned by ``infer()``."
        ),
    ) -> Dict[str, Union[int, float, dict, None]]:
        """Cancel a still-queued infer request; return its progress dict.

        Only a request still waiting in the GPU queue (``queue_position`` > 0,
        not yet started) is cancelled — its background task is dropped before it
        acquires the GPU. A request already running (the forward pass is short)
        or already finished is returned unchanged. The response shape matches
        ``get_infer_status``; a cancelled request reports ``completed_at`` set
        and ``result = {"error": "cancelled before execution"}``.

        Raises:
            KeyError: if ``request_id`` is unknown or has expired.
        """
        job = self._infer_jobs.get(request_id)
        if job is None:
            raise KeyError(
                f"Unknown request_id {request_id!r}. Requests live in-memory per "
                f"replica and expire {self._INFER_JOBS_TTL_SEC // 60} minutes "
                f"after completion."
            )
        if job["state"] == "queued":
            task = job.get("task")
            if task is not None:
                task.cancel()
            job["state"] = "cancelled"
            job["result"] = {"error": "cancelled before execution"}
            job["completed_at"] = time.time()
            logger.info(f"🚫 Cancelled queued infer request {request_id!r}.")
        return self._job_progress(job)
