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


import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import bioengine
import numpy as np

logger = logging.getLogger("ray.serve")
logger.setLevel("INFO")


def _read_pip(name: str) -> List[str]:
    """Load a ``requirements-*.txt`` file next to this module.

    Keeps the heavy pin list out of the ``@bioengine.app`` decorator so
    the deps look like a real requirements file — Dependabot / pip-audit
    can point at the file directly and PR diffs isolate dep bumps.
    """
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
    pip=_read_pip("requirements-runtime.txt"),
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
        # Route bioimageio.spec + bioimageio.core log messages through
        # loguru. Their loggers are ``logger.disable()``-d by default
        # (standard convention for libraries), so per-weight-format
        # progress, conda subprocess spawns, and the like never surface
        # in the replica's stderr otherwise. Enabling once at replica
        # init is idempotent.
        try:
            from loguru import logger as _loguru_logger

            _loguru_logger.enable("bioimageio")
        except Exception as e:
            logger.warning(f"Could not enable bioimageio loguru sink: {e}")

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

    def _test(self, rdf_path: str, custom_environment: bool = False) -> dict:
        """Run ``bioimageio.core.test_description`` in a subprocess.

        **Both paths spawn a subprocess.** The point is GPU / CUDA
        context isolation: PyTorch caches allocator arenas inside
        its Python process's CUDA context and does not release them
        across calls even after ``torch.cuda.empty_cache()``. Running
        the model in the RuntimeApp's own process therefore leaves
        multi-GB of VRAM pinned across tests. Testing in a child
        process side-steps this — the OS reclaims the entire context
        (and all its VRAM) when the subprocess exits, no matter what
        state the model left behind.

        ``custom_environment=False`` (default):
            Spawn ``sys.executable -c <script>`` where the script
            imports ``test_description`` and runs it with
            ``runtime_env="currently-active"``. The child inherits
            the replica's venv via ``sys.executable`` + Ray's
            virtualenv layout, so the same package pins are in
            effect — this is the environment ``infer()`` will use in
            production. See ``_run_bioimageio_test_subprocess``.

        ``custom_environment=True``:
            Delegates to
            ``test_description(runtime_env="as-described")`` which
            already shells out through ``run_command`` (env create,
            env exec, ``bioimageio test`` inside the env). We swap
            ``conda`` → ``mamba`` on the first arg for the faster
            libmamba solver (both binaries ship in the replica image
            at ``/home/ray/anaconda3/bin/``). Every env name seen
            through ``-n <name>`` / ``--name=<name>`` is captured
            and removed after the call — success or failure —
            instead of leaving multi-GB envs on the pod disk.

        Enabling ``loguru`` output for the ``bioimageio`` namespace
        (spec + core) at replica init makes conda subprocess spawns
        and per-weight-format progress visible in the replica's
        stderr.
        """
        import subprocess

        if not Path(rdf_path).exists():
            raise FileNotFoundError(f"RDF not found: {rdf_path}")

        try:
            if not custom_environment:
                return self._run_bioimageio_test_subprocess(rdf_path)

            # custom_environment=True: run inside the model's declared
            # conda env via mamba. bioimageio.core spawns its own
            # subprocess for the actual test (conda run -n <env>
            # bioimageio test ...), so GPU/CUDA context is isolated
            # by process boundary just as in the standard-env path.
            # We track env names for cleanup.
            from bioimageio.core import test_description

            # ``/home/ray/anaconda3/envs/`` and ``/home/ray/.mamba/pkgs``
            # are on a read-only mount inside the Ray worker pod, so
            # mamba's default env prefix and package cache locations
            # fail with ``Read-only file system``. Redirect both to
            # this replica's writable HOME (set by
            # ``bioengine._app.replica_init`` to
            # ``<app_dir>/home/``, backed by the app's PVC). Same env
            # prefix across tests → mamba's pkg cache is reused between
            # calls, cutting env-create time from ~minutes to ~10s once
            # warm.
            mamba_root = Path(os.environ["HOME"]) / ".bioengine-conda"
            (mamba_root / "envs").mkdir(parents=True, exist_ok=True)
            (mamba_root / "pkgs").mkdir(parents=True, exist_ok=True)
            mamba_env_vars = os.environ.copy()
            mamba_env_vars["CONDA_ENVS_PATH"] = str(mamba_root / "envs")
            mamba_env_vars["CONDA_PKGS_DIRS"] = str(mamba_root / "pkgs")

            created_env_names: set[str] = set()

            def mamba_run_command(args):
                args = list(args)
                if args and args[0] == "conda":
                    args[0] = "mamba"
                for i, a in enumerate(args):
                    if a == "-n" and i + 1 < len(args):
                        created_env_names.add(args[i + 1])
                    elif a.startswith("--name="):
                        created_env_names.add(a[len("--name=") :])
                logger.info(f"🐍 [conda] running: {' '.join(args)}")
                proc = subprocess.run(
                    args, capture_output=True, text=True, env=mamba_env_vars
                )
                # Route child stdio through the replica logger — a compact
                # tail is enough to diagnose mamba failures without
                # flooding the log ring with libmamba trace lines.
                if proc.stdout:
                    for line in proc.stdout.rstrip().splitlines()[-20:]:
                        logger.info(f"[mamba:stdout] {line}")
                if proc.stderr:
                    for line in proc.stderr.rstrip().splitlines()[-20:]:
                        logger.info(f"[mamba:stderr] {line}")
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(
                        proc.returncode, args, output=proc.stdout,
                        stderr=proc.stderr,
                    )

            try:
                # Deliberately omit ``expected_type`` here — the model's
                # declared conda env often pins an older ``bioimageio.core``
                # whose ``bioimageio test`` CLI does not recognise
                # ``--expected-type=<type>``. ``test_description`` would
                # otherwise pass that flag into the subprocess and fail
                # with ``unrecognized arguments: --expected-type=model``.
                # We know ``model_id`` resolves to a model artifact
                # (the ``bioimage-io/model-runner`` service is scoped to
                # models), so losing the type assertion is not a real
                # gap.
                validation_summary = test_description(
                    rdf_path,
                    runtime_env="as-described",
                    run_command=mamba_run_command,
                )
                return validation_summary.model_dump(mode="json")
            finally:
                # Best-effort cleanup — remove any envs seen through
                # run_command so the pod doesn't accumulate multi-GB
                # per-model envs. Runs on both success and failure.
                for name in created_env_names:
                    try:
                        subprocess.run(
                            ["mamba", "env", "remove", "-n", name, "--yes"],
                            capture_output=True,
                            check=True,
                            env=mamba_env_vars,
                        )
                        logger.info(f"🧹 Removed conda env '{name}'")
                    except Exception as cleanup_err:
                        logger.warning(
                            f"⚠️ Failed to remove conda env '{name}': {cleanup_err}"
                        )
        except Exception as e:
            logger.error(f"❌ Model test failed: {str(e)}")
            raise

    def _run_bioimageio_test_subprocess(self, rdf_path: str) -> dict:
        """Run ``test_description(runtime_env="currently-active")`` in a
        child Python process and return the summary dict.

        We spawn via ``sys.executable`` so the child lands in the same
        Ray-managed virtualenv as this replica (same package pins),
        then drive ``test_description`` via a small inline script. The
        subprocess writes the ``ValidationSummary`` model-dump to a
        JSON temp file and we read that back — round-tripping through
        JSON keeps the data flat and avoids any cross-process
        cloudpickle. RDF path and output path are passed as
        ``argv[1:]`` so the script has no interpolated string content
        (safe against filenames with quotes / backslashes).
        """
        import json
        import subprocess
        import sys
        import tempfile

        script = (
            "import json, sys\n"
            "from bioimageio.core import test_description\n"
            "rdf_path, out_path = sys.argv[1], sys.argv[2]\n"
            "summary = test_description(\n"
            "    rdf_path, expected_type='model', runtime_env='currently-active'\n"
            ")\n"
            "with open(out_path, 'w') as f:\n"
            "    json.dump(summary.model_dump(mode='json'), f)\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = str(Path(tmpdir) / "summary.json")
            logger.info(
                f"🐍 [test] Spawning bioimageio subprocess for CUDA "
                f"context isolation: {sys.executable} -c <inline> "
                f"{rdf_path} {summary_path}"
            )
            result = subprocess.run(
                [sys.executable, "-c", script, str(rdf_path), summary_path],
                capture_output=True,
                text=True,
            )
            # Route child stdio through the replica's logger so
            # bioimageio's own log lines (progress, per-weight-format
            # status) surface here — without this, replica stderr is
            # silent about what the child actually did.
            if result.stdout:
                for line in result.stdout.rstrip().splitlines()[-40:]:
                    logger.info(f"[test:stdout] {line}")
            if result.stderr:
                for line in result.stderr.rstrip().splitlines()[-40:]:
                    logger.info(f"[test:stderr] {line}")

            summary_file = Path(summary_path)
            if not summary_file.exists():
                raise RuntimeError(
                    f"bioimageio test subprocess exited with code "
                    f"{result.returncode} without producing a summary "
                    f"(stderr tail: {(result.stderr or '')[-500:]!r})"
                )
            with summary_file.open() as f:
                return json.load(f)

    async def test(
        self,
        rdf_path: str,
        custom_environment: bool = False,
    ) -> dict:
        """Test model inference.

        When ``custom_environment=False`` (default), the test runs in
        the currently-active RuntimeApp venv — same interpreter that
        serves inference. Fast, and validates what the caller will
        actually use.

        When ``custom_environment=True``, the test runs inside the
        conda environment declared by the model's weights description
        (``bioimageio.core`` ``runtime_env="as-described"``). Env
        creation uses ``mamba`` (via a swapping ``run_command``) and
        the env is removed after the call.
        """
        cpu_before, gpu_before = self._get_memory_usage()
        logger.info(
            f"📊 [test] Memory before: CPU: {cpu_before / (1024 * 1024):.2f} MB, "
            f"GPU: {gpu_before / (1024 * 1024):.2f} MB"
        )

        # Free the GPU before ``bioimageio.core.test_description``
        # loads the tested model — any cached prediction pipelines
        # from prior ``predict()`` calls on this replica would
        # otherwise contend for VRAM against the tested model's
        # fresh load and OOM on foundation-scale weights. The
        # eviction path calls each cached model's ``__del__``, which
        # releases GPU memory eagerly (same code path Ray Serve uses
        # on natural LRU overflow). The ``max_ongoing_requests=1``
        # on this deployment already keeps every other call queued
        # behind us on the same replica, so once we've evicted the
        # tested model has the full GPU to itself for the duration
        # of the ``test_description`` call.
        evicted_count = await bioengine.multiplex.evict_all_models(self)
        if evicted_count:
            logger.info(
                f"🧹 Evicted {evicted_count} cached pipeline(s) to free "
                f"the GPU for test."
            )

        # Run the sync ``_test`` in a thread so it doesn't block this
        # replica's asyncio loop. ``_test`` blocks on
        # ``subprocess.run`` / ``check_call`` (bioimageio subprocess
        # for the standard path, mamba env-create + ``bioimageio
        # test`` inside the env for the custom path). A multi-minute
        # mamba env build would otherwise starve Ray Serve's health
        # probes on this replica — three consecutive
        # ``health_check_timeout_s=30`` misses and Ray Serve issues
        # ``ray.kill`` (observed on resourceful-lizard custom-env at
        # 372s). ``asyncio.to_thread`` keeps ``ping()`` responsive
        # for the duration of the subprocess wait.
        import asyncio

        test_report = await asyncio.to_thread(
            self._test, rdf_path, custom_environment
        )
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
