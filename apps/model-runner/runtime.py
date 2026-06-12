"""GPU runtime for bioimage.io model inference.

The runtime is the GPU half of the model-runner app. ``EntryApp`` keeps a
type-hint reference to ``RuntimeApp`` so the v0.6 composition graph wires
them together; ``EntryApp`` then calls ``await self.runtime.ping()`` /
``await self.runtime.predict(...)`` / ``await self.runtime.test(...)`` to
delegate the heavy work.

Module-level imports stay deliberately lightweight (just stdlib + bioengine
+ numpy + ray) so the introspection task can load this file with only the
BioEngine baseline runtime_env. Heavy deps (``bioimageio.core``,
``careamics``, ``cellpose``, ``torch``, ``tensorflow``, …) are installed
by the ``@bioengine.app(pip=REQUIREMENTS)`` declaration and imported
inside method bodies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import bioengine
import numpy as np
import ray

logger = logging.getLogger("ray.serve")
logger.setLevel("INFO")

# Pinned versions of every heavy dep the runtime needs at replica boot.
# Mirrors the v0.5 runtime_deployment.py REQUIREMENTS list verbatim — same
# package set and same pins. ``xarray`` is a transitive dep of
# ``bioimageio.core`` but must be pinned explicitly here because pip's
# resolver doesn't otherwise lock it tightly enough.
REQUIREMENTS = [
    "bioimageio.core==0.10.0",
    "careamics==0.0.16",
    "cellpose==3.1.1.2",
    "nvidia-ml-py==12.555.43",
    "numpy==1.26.4",
    "onnxruntime==1.20.1",
    "psutil==6.1.1",
    "tensorflow==2.16.1",
    "torch==2.5.1",
    "torchvision==0.20.1",
    "xarray==2025.1.2",
]


@bioengine.app(
    num_cpus=1,
    num_gpus=1,
    memory_mb=12 * 1024,
    pip=REQUIREMENTS,
    max_ongoing_requests=1,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 2,
        "target_num_ongoing_requests_per_replica": 0.8,
        "metrics_interval_s": 2.0,
        "look_back_period_s": 10.0,
        "downscale_delay_s": 300,
        "upscale_delay_s": 0.0,
    },
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    graceful_shutdown_timeout_s=120.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class RuntimeApp:
    """GPU-resident bioimage.io model executor."""

    def __init__(self) -> None:
        self._kwargs_cache: Dict[str, dict] = {}

    # === Liveness ===

    @bioengine.method
    async def ping(self) -> str:
        """Fast liveness probe used by EntryApp before every GPU-bound method."""
        return "pong"

    # === Memory accounting (used by test/predict log lines) ===

    def _get_memory_usage(self) -> tuple:
        """Return current ``(cpu_bytes, gpu_bytes)`` for this process."""
        import psutil

        cpu_mem = psutil.Process().memory_info().rss
        gpu_mem = 0
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_mem += info.used
            finally:
                pynvml.nvmlShutdown()
        except (ImportError, Exception):
            pass
        return cpu_mem, gpu_mem

    # === Testing ===

    def _test(self, rdf_path: str) -> dict:
        """Run ``bioimageio.core.test_model`` on the given RDF path."""
        from bioimageio.core import test_model

        try:
            if not Path(rdf_path).exists():
                raise FileNotFoundError(f"RDF not found: {rdf_path}")
            validation_summary = test_model(rdf_path)
            return validation_summary.model_dump(mode="json")
        except Exception as e:
            logger.error(f"❌ Model test failed: {str(e)}")
            raise

    async def test(
        self,
        rdf_path: str,
        additional_requirements: Optional[List[str]] = None,
    ) -> dict:
        """Test model inference with optional additional pip requirements.

        When ``additional_requirements`` adds packages outside the baseline
        ``REQUIREMENTS`` set, the test runs as a *fresh remote Ray task*
        in a runtime_env that layers the extra packages on top — keeping
        the cached replica venv clean.
        """
        cpu_before, gpu_before = self._get_memory_usage()
        logger.info(
            f"📊 [test] Memory before: CPU: {cpu_before / (1024 * 1024):.2f} MB, "
            f"GPU: {gpu_before / (1024 * 1024):.2f} MB"
        )
        additional_packages: List[str] = []
        if additional_requirements:
            if not isinstance(additional_requirements, list):
                logger.error("❌ additional_requirements must be a list of strings")
                raise ValueError("additional_requirements must be a list of strings.")
            for ad_req in additional_requirements:
                ad_req = ad_req.strip()
                exists = False
                for req in REQUIREMENTS:
                    package, _ = req.split("==")
                    if ad_req.startswith(package):
                        exists = True
                        break
                if not exists:
                    additional_packages.append(ad_req)

        if additional_packages:
            logger.info(
                f"🚀 Running test with additional packages: {additional_packages}"
            )
            remote_function = ray.remote(self._test.__func__)
            remote_function = remote_function.options(
                num_cpus=1,
                num_gpus=0,
                memory=4 * 1024 * 1024 * 1024,
                runtime_env={"pip": REQUIREMENTS + additional_packages},
            )
            result_ref = remote_function.remote(None, rdf_path)
            logger.info("📋 Submitted remote test job")
            return result_ref

        test_report = self._test(rdf_path)
        cpu_after, gpu_after = self._get_memory_usage()
        logger.info(
            f"📊 [test] Memory after: CPU: {cpu_after / (1024 * 1024):.2f} MB, "
            f"GPU: {gpu_after / (1024 * 1024):.2f} MB"
        )
        return test_report

    # === Prediction pipeline cache key ===

    def _set_prediction_kwargs(
        self,
        rdf_path: str,
        weights_format: str,
        device: str,
        default_blocksize_parameter: int,
        latest_remote_modified: Optional[float] = None,
    ) -> str:
        """Generate cache key for prediction pipeline configuration."""
        pipeline_kwargs = {
            "rdf_path": rdf_path,
            "latest_remote_modified": latest_remote_modified,
            "create_kwargs": {
                "weights_format": weights_format,
                "device": device,
                "default_blocksize_parameter": default_blocksize_parameter,
            },
        }
        json_str = json.dumps(pipeline_kwargs, sort_keys=True)
        cache_key = hashlib.md5(json_str.encode()).hexdigest()
        self._kwargs_cache[cache_key] = pipeline_kwargs
        return cache_key

    # === Multiplexed pipeline (Ray Serve handles eviction by max_models) ===

    @bioengine.multiplexed(
        max_models=int(os.environ.get("PIPELINE_CACHE_SIZE", 10)),
    )
    async def _create_prediction_pipeline(self, cache_key: str):
        """Create + cache the prediction pipeline for ``cache_key``."""
        cpu_before, gpu_before = self._get_memory_usage()
        from bioimageio.core import create_prediction_pipeline, load_model_description

        pipeline_kwargs = self._kwargs_cache.get(cache_key)
        if not pipeline_kwargs:
            logger.error(f"❌ No pipeline kwargs found for cache key: {cache_key}")
            raise ValueError(f"No pipeline kwargs found for cache key: {cache_key}")

        rdf_path = pipeline_kwargs["rdf_path"]
        create_kwargs = pipeline_kwargs["create_kwargs"]

        try:
            model_description = load_model_description(rdf_path)
            pipeline = create_prediction_pipeline(model_description, **create_kwargs)
            pipeline.load()
            cpu_after, gpu_after = self._get_memory_usage()
            logger.info(
                f"✅ Created and loaded prediction pipeline for model at {rdf_path}"
            )
            logger.info(
                f"📊 [pipeline load] CPU: {cpu_after / (1024 * 1024):.2f} MB, "
                f"GPU: {gpu_after / (1024 * 1024):.2f} MB"
            )
            return pipeline
        except Exception as e:
            logger.error(f"❌ Failed to create prediction pipeline: {str(e)}")
            raise
        finally:
            self._kwargs_cache.pop(cache_key, None)

    # === Prediction ===

    async def predict(
        self,
        rdf_path: str,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
        weights_format: Optional[str] = None,
        device: Literal["cuda", "cpu"] = None,
        default_blocksize_parameter: Optional[int] = None,
        sample_id: str = "sample",
        latest_remote_modified: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Run inference using a cached bioimageio.core prediction pipeline."""
        cpu_before, gpu_before = self._get_memory_usage()
        logger.info(
            f"📊 [predict] Memory before: CPU: {cpu_before / (1024 * 1024):.2f} MB, "
            f"GPU: {gpu_before / (1024 * 1024):.2f} MB"
        )
        from bioimageio.core.digest_spec import create_sample_for_model

        try:
            if not Path(rdf_path).exists():
                raise FileNotFoundError(f"RDF not found: {rdf_path}")

            logger.info(
                f"🚀 Starting prediction for model at {rdf_path} with "
                f"device={device} and weights_format={weights_format}"
            )
            cache_key = self._set_prediction_kwargs(
                rdf_path=rdf_path,
                weights_format=weights_format,
                device=device,
                default_blocksize_parameter=default_blocksize_parameter,
                latest_remote_modified=latest_remote_modified,
            )
            pipeline = await self._create_prediction_pipeline(cache_key)

            sample = create_sample_for_model(
                pipeline.model_description,
                inputs=inputs,
                sample_id=sample_id,
            )

            if default_blocksize_parameter:
                result = pipeline.predict_sample_with_blocking(sample)
            else:
                result = pipeline.predict_sample_without_blocking(sample)

            cpu_after, gpu_after = self._get_memory_usage()
            logger.info(
                f"📊 [predict] CPU: {cpu_after / (1024 * 1024):.2f} MB, "
                f"GPU: {gpu_after / (1024 * 1024):.2f} MB"
            )
            return {str(k): v.data.data for k, v in result.members.items()}

        except Exception as e:
            # CUDA OOM names vary across PyTorch versions and may not be
            # importable on the receiving end of the RPC. Re-raise as a
            # plain RuntimeError so the deserialiser doesn't crash.
            import torch

            torch.cuda.empty_cache()
            err_type = type(e).__name__
            if err_type in ("OutOfMemoryError", "CUDAOutOfMemoryError") or (
                "out of memory" in str(e).lower()
            ):
                oom_msg = f"CUDA out of memory during inference: {str(e)}"
                logger.error(f"❌ {oom_msg}")
                raise RuntimeError(oom_msg) from None
            logger.error(f"❌ Prediction failed: {str(e)}")
            raise
