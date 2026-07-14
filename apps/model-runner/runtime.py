"""GPU runtime for bioimage.io model inference.

The runtime is the GPU half of the model-runner app. ``EntryApp`` keeps a
type-hint reference to ``RuntimeApp`` so the v0.6 composition graph wires
them together; ``EntryApp`` then calls ``await self.runtime.ping()`` /
``await self.runtime.predict_from_disk(...)`` /
``await self.runtime.test(...)`` to delegate the heavy work. Inputs and
outputs for ``predict_from_disk`` stream through the shared PVC-backed
inference dir instead of over the RPC, so large arrays don't sit in RAM
while the request queues on ``_gpu_lock``.

Module-level imports stay deliberately lightweight (just stdlib +
bioengine) so the introspection task can load this file with only the
BioEngine baseline runtime_env. Heavy deps (``bioimageio.core``,
``careamics``, ``cellpose``, ``torch``, ``tensorflow``, …) are installed
by the ``@bioengine.app(pip=REQUIREMENTS)`` declaration and imported
inside method bodies (or, for prediction, inside the child process).
"""


import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional

import bioengine

logger = logging.getLogger("ray.serve")
logger.setLevel("INFO")

# Filename key under which EntryApp.infer stages a bare (single, unnamed)
# input array. On read-back the runtime unwraps it to a bare array so
# bioimageio.core maps it to the model's sole input member id itself,
# rather than us guessing a member id that differs across spec versions.
SINGLE_INPUT_KEY = "__single_input__"


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
    # ``max_ongoing_requests=10`` intentionally lets requests queue at
    # the replica's asyncio queue behind ``_gpu_lock``. Ray Serve's
    # autoscaler only counts ongoing requests that reached a replica
    # (not those waiting at the router), so keeping this at 1 would
    # hide backlog and never trigger scale-up. With 10 and the internal
    # lock, the router forwards up to 10 concurrent RPCs per replica
    # and the lock serialises GPU work — autoscaler observes queue
    # depth. Precedent: apps/cell-image-search/main.py uses the same
    # pattern with max_ongoing_requests=10 and target=3.
    max_ongoing_requests=10,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 2,
        # Scale up when >3 ongoing per replica — with 1 replica and
        # 4 ongoing (1 running + 3 queued behind the lock), metric
        # 4/1 > 3 triggers a scale-up to 2. Then 4/2 = 2 ≤ 3 → stay.
        "target_num_ongoing_requests_per_replica": 3.0,
        "metrics_interval_s": 2.0,
        "look_back_period_s": 10.0,
        # 10 min after the last scale-up condition clears before
        # scaling down — avoids thrashing on bursty traffic.
        "downscale_delay_s": 600,
        "upscale_delay_s": 0.0,
    },
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    # Bumped from 120s: with ``max_ongoing_requests=10`` a full backlog
    # may need several minutes to drain through the lock-serialised
    # GPU work before the replica exits cleanly.
    graceful_shutdown_timeout_s=300.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class RuntimeApp:
    """GPU-resident bioimage.io model executor."""

    # Per-request scratch on the app's shared PVC-backed HOME. EntryApp
    # writes ``input/<key>.npy`` here on ``infer()`` receipt; this app
    # reads inputs from disk, deletes the input dir, writes outputs to
    # ``output/<key>.npy``, and records step timestamps in
    # ``state.json``. Kept in sync with ``EntryApp._INFERENCE_DIR_NAME``.
    _INFERENCE_DIR_NAME = ".model-runner-inference"

    def __init__(self) -> None:
        # Serialises GPU work — predict + test both acquire this lock
        # so only one request touches the GPU at a time even when
        # ``max_ongoing_requests=10`` lets multiple RPCs reach the
        # replica. That extra queue is intentional: Ray Serve's
        # autoscaler needs to see it to scale up.
        self._gpu_lock = asyncio.Lock()
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

    def _inference_dir(self) -> Path:
        """Base directory for per-request input/output/state files."""
        return Path(os.environ["HOME"]) / self._INFERENCE_DIR_NAME

    @staticmethod
    def _write_state_file(state_path: Path, state: Dict[str, float]) -> None:
        """Write ``state.json`` atomically via rename so the entry
        never observes a half-written file when polling.
        """
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(state_path)

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

    # === Subprocess env hardening ===

    _SENSITIVE_ENV_NEEDLES = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY")

    def _safe_subprocess_env(self) -> Dict[str, str]:
        """Return ``os.environ`` minus obviously-sensitive entries.

        Test subprocesses (bioimageio CLI, mamba) don't need the
        RuntimeApp's Hypha credentials. Denylist any env-var name
        containing ``TOKEN`` / ``SECRET`` / ``PASSWORD`` /
        ``CREDENTIAL`` / ``API_KEY`` — covers ``HYPHA_TOKEN``,
        ``BIOENGINE_ARTIFACT_TOKEN``, ``BIOIMAGE_IO_TOKEN``, and any
        generic cloud credentials leaking through. Everything else
        (``PATH``, ``HOME``, ``TMPDIR``, ``PYTHONPATH``, ``LANG``,
        ``CUDA_VISIBLE_DEVICES``, ``BIOENGINE_APP_DIR``,
        ``HYPHA_ARTIFACT_VERSION``) is preserved.
        """
        return {
            k: v
            for k, v in os.environ.items()
            if not any(needle in k.upper() for needle in self._SENSITIVE_ENV_NEEDLES)
        }

    def _mamba_env_vars(self) -> Dict[str, str]:
        """Scrubbed env with mamba's env-prefix + package cache
        redirected to the replica's writable HOME.

        ``/home/ray/anaconda3/envs/`` and ``/home/ray/.mamba/pkgs``
        are on a read-only mount inside the Ray worker pod. HOME is
        set by ``bioengine._app.replica_init`` to
        ``<app_dir>/home/`` on the app's PVC, which is
        cross-replica RWX under the bioengine layout, so envs built
        by EntryApp are visible to this RuntimeApp on the same
        path.
        """
        env_vars = self._safe_subprocess_env()
        mamba_root = Path(os.environ["HOME"]) / ".bioengine-conda"
        (mamba_root / "envs").mkdir(parents=True, exist_ok=True)
        (mamba_root / "pkgs").mkdir(parents=True, exist_ok=True)
        env_vars["CONDA_ENVS_PATH"] = str(mamba_root / "envs")
        env_vars["CONDA_PKGS_DIRS"] = str(mamba_root / "pkgs")
        return env_vars

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
            # conda env via mamba. Env creation is now owned by the
            # EntryApp (``_prebuild_conda_envs``) so this replica's
            # GPU isn't held during a multi-minute mamba solve.
            # ``bioimageio.core.test_description(runtime_env="as-described")``
            # still does its own existence probe
            # (``mamba run -n <hash> python --version``) — that lands
            # on the PVC-backed HOME that Entry populated → env
            # found → mamba env create is skipped → straight to the
            # actual ``bioimageio test`` subprocess.
            #
            # Envs are cached across calls in
            # ``$HOME/.bioengine-conda/envs/`` so a second test of
            # the same model reuses the same env in seconds. No
            # auto-cleanup here — Entry decides lifecycle.
            from bioimageio.core import test_description

            mamba_env_vars = self._mamba_env_vars()

            def mamba_run_command(args):
                args = list(args)
                if args and args[0] == "conda":
                    args[0] = "mamba"
                logger.info(f"🐍 [conda] running: {' '.join(args)}")
                proc = subprocess.run(
                    args, capture_output=True, text=True, env=mamba_env_vars
                )
                # Route child stdio through the replica logger — a
                # compact tail is enough to diagnose mamba failures
                # without flooding the log ring with libmamba trace
                # lines.
                if proc.stdout:
                    for line in proc.stdout.rstrip().splitlines()[-20:]:
                        logger.info(f"[mamba:stdout] {line}")
                if proc.stderr:
                    for line in proc.stderr.rstrip().splitlines()[-20:]:
                        logger.info(f"[mamba:stderr] {line}")
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(
                        proc.returncode,
                        args,
                        output=proc.stdout,
                        stderr=proc.stderr,
                    )

            # Deliberately omit ``expected_type`` here — the model's
            # declared conda env often pins an older ``bioimageio.core``
            # whose ``bioimageio test`` CLI does not recognise
            # ``--expected-type=<type>``. ``test_description`` would
            # otherwise pass that flag into the subprocess and fail
            # with ``unrecognized arguments: --expected-type=model``.
            # We know ``model_id`` resolves to a model artifact (the
            # ``bioimage-io/model-runner`` service is scoped to models),
            # so losing the type assertion is not a real gap.
            validation_summary = test_description(
                rdf_path,
                runtime_env="as-described",
                run_command=mamba_run_command,
            )
            return validation_summary.model_dump(mode="json")
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
                env=self._safe_subprocess_env(),
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
        # GPU serialisation: ``max_ongoing_requests=10`` allows requests
        # to queue at the replica so the autoscaler sees them, but only
        # one may hold the GPU at a time.
        async with self._gpu_lock:
            cpu_before, gpu_before = self._get_memory_usage()
            logger.info(
                f"📊 [test] Memory before: CPU: {cpu_before / (1024 * 1024):.2f} MB, "
                f"GPU: {gpu_before / (1024 * 1024):.2f} MB"
            )

            # Both the test and the infer path run the model in a child
            # process (CUDA-context isolation), so no GPU-resident model
            # from a prior call survives in this replica to contend for
            # VRAM — the tested model has the whole GPU to itself once we
            # hold ``_gpu_lock``. Nothing to evict here.

            # Run the sync ``_test`` in a thread so it doesn't block
            # this replica's asyncio loop. ``_test`` blocks on
            # ``subprocess.run`` / ``check_call`` (bioimageio subprocess
            # for the standard path, mamba env-create + ``bioimageio
            # test`` inside the env for the custom path). A multi-minute
            # mamba env build would otherwise starve Ray Serve's health
            # probes on this replica. ``asyncio.to_thread`` keeps
            # ``ping()`` responsive for the duration of the subprocess
            # wait (ping isn't guarded by the lock).
            test_report = await asyncio.to_thread(
                self._test, rdf_path, custom_environment
            )
            cpu_after, gpu_after = self._get_memory_usage()
            logger.info(
                f"📊 [test] Memory after: CPU: {cpu_after / (1024 * 1024):.2f} MB, "
                f"GPU: {gpu_after / (1024 * 1024):.2f} MB"
            )
            return test_report

    # === Prediction ===

    def _run_prediction_subprocess(
        self,
        rdf_path: str,
        input_dir: str,
        output_dir: str,
        params: Dict[str, object],
    ) -> None:
        """Load the model, run inference, and write outputs — all in a
        child Python process for CUDA-context isolation.

        Mirrors ``_run_bioimageio_test_subprocess``. A subprocess is the
        only framework-agnostic way to guarantee the model's VRAM is
        fully reclaimed: the OS tears down the entire CUDA context on
        exit, so torch's caching allocator, TensorFlow's greedy
        whole-GPU grab, and onnxruntime's CUDA arena are all released no
        matter what state the model left behind. An in-process pipeline
        cache plus ``torch.cuda.empty_cache()`` only ever freed torch
        memory and left TF / ONNX VRAM pinned until replica restart —
        the cause of OOM piling up across repeated infer calls.

        The child reads ``input_dir/<key>.npy`` and writes
        ``output_dir/<key>.npy`` itself so large arrays never cross the
        process boundary; ``params`` travels as a small JSON file.
        """
        import subprocess
        import sys
        import tempfile

        script = """
import json, sys
from pathlib import Path
import numpy as np
from bioimageio.core import create_prediction_pipeline, load_model_description
from bioimageio.core.digest_spec import create_sample_for_model

rdf_path, input_dir, output_dir, params_path = sys.argv[1:5]
with open(params_path) as f:
    params = json.load(f)
single_input_key = params["single_input_key"]

inputs = {}
for entry in sorted(Path(input_dir).iterdir()):
    if entry.is_file() and entry.suffix == ".npy":
        inputs[entry.stem] = np.load(str(entry))
if not inputs:
    raise ValueError("No .npy input files found under " + input_dir)

model_description = load_model_description(rdf_path)
pipeline = create_prediction_pipeline(
    model_description,
    weights_format=params["weights_format"],
    device=params["device"],
    default_blocksize_parameter=params["default_blocksize_parameter"],
)
pipeline.load()

# A bare single input arrives under the sentinel key; hand it to core as
# a bare array so it maps to the model's sole input member id. Explicit
# multi/named inputs pass through as-is.
sample_inputs = (
    inputs[single_input_key]
    if list(inputs) == [single_input_key]
    else inputs
)
sample = create_sample_for_model(
    pipeline.model_description,
    inputs=sample_inputs,
    sample_id=params["sample_id"],
)
if params["default_blocksize_parameter"]:
    result = pipeline.predict_sample_with_blocking(sample)
else:
    result = pipeline.predict_sample_without_blocking(sample)

out = Path(output_dir)
out.mkdir(parents=True, exist_ok=True)
for key, member in result.members.items():
    np.save(str(out / (str(key) + ".npy")), member.data.data)
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            params_path = str(Path(tmpdir) / "params.json")
            with open(params_path, "w") as f:
                json.dump(params, f)
            logger.info(
                f"🐍 [predict] Spawning bioimageio subprocess for CUDA "
                f"context isolation: {sys.executable} -c <inline> {rdf_path}"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    rdf_path,
                    input_dir,
                    output_dir,
                    params_path,
                ],
                capture_output=True,
                text=True,
                env=self._safe_subprocess_env(),
            )
            if result.stdout:
                for line in result.stdout.rstrip().splitlines()[-40:]:
                    logger.info(f"[predict:stdout] {line}")
            if result.stderr:
                for line in result.stderr.rstrip().splitlines()[-40:]:
                    logger.info(f"[predict:stderr] {line}")
            if result.returncode != 0:
                stderr_tail = (result.stderr or "")[-800:]
                if "out of memory" in stderr_tail.lower():
                    raise RuntimeError(
                        f"CUDA out of memory during inference: {stderr_tail}"
                    )
                raise RuntimeError(
                    f"Inference subprocess exited with code "
                    f"{result.returncode} (stderr tail: {stderr_tail!r})"
                )

    async def predict_from_disk(
        self,
        request_id: str,
        rdf_path: str,
        weights_format: Optional[str] = None,
        device: Literal["cuda", "cpu"] = None,
        default_blocksize_parameter: Optional[int] = None,
        sample_id: str = "sample",
    ) -> None:
        """Read inputs from disk, run inference in a subprocess, write
        outputs to disk.

        Called by ``EntryApp._execute_infer`` after it has staged
        ``<inference_dir>/<request_id>/input/<key>.npy`` on the shared
        PVC. This method:

        1. Acquires ``self._gpu_lock`` so only one request runs on the
           GPU at a time — the extra RPCs sitting on the lock are what
           the Ray Serve autoscaler measures.
        2. Writes ``state.json`` with ``runtime_started_at`` so the
           entry can distinguish "still queued at runtime" from
           "actively running" when computing ``queue_position``.
        3. Runs the model in a child process
           (``_run_prediction_subprocess``) which reads the inputs,
           predicts, and writes ``output/<key>.npy``. The subprocess exit
           reclaims all of the model's VRAM regardless of framework.
        4. Deletes the input dir and records ``runtime_completed_at``.

        The entry's poll (``get_infer_status``) reads the outputs off
        disk and deletes the request dir after the caller collects them.
        """
        request_dir = self._inference_dir() / request_id
        input_dir = request_dir / "input"
        output_dir = request_dir / "output"
        state_file = request_dir / "state.json"

        async with self._gpu_lock:
            cpu_before, gpu_before = self._get_memory_usage()
            logger.info(
                f"📊 [predict] Memory before: CPU: {cpu_before / (1024 * 1024):.2f} MB, "
                f"GPU: {gpu_before / (1024 * 1024):.2f} MB"
            )

            # Signal to the entry that this request has left the
            # (Ray Serve replica-side) queue and is now on the GPU.
            # Written before any expensive work so the ``running``
            # timestamp reflects the true start.
            state: Dict[str, float] = {"runtime_started_at": time.time()}
            await asyncio.to_thread(self._write_state_file, state_file, state)

            if not await asyncio.to_thread(input_dir.is_dir):
                raise FileNotFoundError(
                    f"Input directory missing for request {request_id!r}: "
                    f"{input_dir}"
                )
            if not await asyncio.to_thread(Path(rdf_path).exists):
                raise FileNotFoundError(f"RDF not found: {rdf_path}")

            logger.info(
                f"🚀 Starting prediction for model at {rdf_path} with "
                f"device={device} and weights_format={weights_format}"
            )
            params: Dict[str, object] = {
                "weights_format": weights_format,
                "device": device,
                "default_blocksize_parameter": default_blocksize_parameter,
                "sample_id": sample_id,
                "single_input_key": SINGLE_INPUT_KEY,
            }
            await asyncio.to_thread(
                self._run_prediction_subprocess,
                rdf_path,
                str(input_dir),
                str(output_dir),
                params,
            )

            # Free the input images now the child has consumed them.
            await asyncio.to_thread(
                shutil.rmtree, str(input_dir), ignore_errors=True
            )

            cpu_after, gpu_after = self._get_memory_usage()
            logger.info(
                f"📊 [predict] Memory after: CPU: {cpu_after / (1024 * 1024):.2f} MB, "
                f"GPU: {gpu_after / (1024 * 1024):.2f} MB"
            )

            state["runtime_completed_at"] = time.time()
            await asyncio.to_thread(self._write_state_file, state_file, state)
