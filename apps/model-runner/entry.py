"""@bioengine.app EntryApp — the public RPC surface of the model-runner.

The app provides bioimage.io model search, RDF/documentation retrieval,
RDF validation, end-to-end model testing, and inference. It delegates the
GPU-bound work (``predict`` and the heavy ``bioimageio.core.test_model``
call) to :class:`runtime.RuntimeApp` via the v0.6 type-hint composition.

Heavier helper modules:

* :mod:`model_cache.cache` — LRU cache of downloaded bioimage.io model
  packages, with on-disk markers for cross-replica coordination
* :mod:`model_cache.package` — per-use lock context manager around a
  cached package; the entry uses it via ``async with package: …`` to
  keep models from being evicted mid-inference
"""


import asyncio
import json
import logging
import os
import random
import shutil
import time
import traceback
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import bioengine
import httpx
import numpy as np
import yaml
from hypha_rpc import connect_to_server
from pydantic import Field

from bioengine import __version__

from model_cache import BioimageioPackage, ModelCache
from runtime import RuntimeApp


logger = logging.getLogger("ray.serve")
logger.setLevel("INFO")

SUPPORTED_FILES_TYPES = Literal[".npy", ".png", ".tiff", ".tif", ".jpeg", ".jpg"]


def _read_pip(name: str) -> List[str]:
    """Load a ``requirements-*.txt`` file next to this module.

    Same helper as ``runtime.py:_read_pip`` — duplicated instead of
    imported so each module is self-contained even if the composition
    graph changes.
    """
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
    max_queued_requests=30,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 1,
        "target_num_ongoing_requests_per_replica": 1,
    },
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    graceful_shutdown_timeout_s=300.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class EntryApp:
    """
    Ray Serve deployment for bioimage.io model operations.

    Handles model downloading, caching, validation, testing, and inference
    with cross-replica coordination and atomic filesystem operations.

    Concurrency Design:
    - Uses atomic filesystem operations (mkdir, rename) for replica coordination
    - Download markers prevent duplicate downloads across replicas
    - Access tracking (.last_access files) prevents eviction of active models
    - LRU eviction with retry logic handles cache space management
    - Context managers ensure proper access time tracking during model usage
    - Graceful error handling for filesystem race conditions and I/O errors
    """

    # === Conda env cache tuning ===
    #
    # Custom-environment tests build conda envs on the shared PVC
    # ``$HOME/.bioengine-conda/envs/``. Envs are ~5-10 GB apiece; a
    # given model has up to 3 (one per framework family: torch,
    # onnx, tensorflow) — models that declare multiple torch weight
    # formats (pytorch_state_dict + torchscript) can have more.
    # ``CONDA_ENV_CACHE_MAX_GB`` (env var, default 30) is the soft
    # ceiling — the LRU eviction step keeps the cache under it as
    # a rough estimate (``du -sk``, not exact). Minimum 30 so a
    # single 3-env model always fits without evicting envs it
    # itself needs.
    _CONDA_ENV_CACHE_MIN_GB = 30
    _CONDA_ENV_MAX_AGE_DAYS = 7  # weekly age-based sweep

    def __init__(
        self,
        runtime: RuntimeApp,
        cache_size_in_gb: float = 50.0,
    ) -> None:
        self.runtime = runtime

        # Set Hypha server and workspace
        self.server_url = "https://hypha.aicell.io"
        self._hypha_token = os.getenv("HYPHA_TOKEN")
        if not self._hypha_token:
            raise RuntimeError("HYPHA_TOKEN environment variable is not set")

        # Conda env cache soft ceiling
        raw = os.getenv("CONDA_ENV_CACHE_MAX_GB", str(self._CONDA_ENV_CACHE_MIN_GB))
        try:
            configured = float(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"CONDA_ENV_CACHE_MAX_GB must be a number, got {raw!r}"
            ) from exc
        if configured < self._CONDA_ENV_CACHE_MIN_GB:
            raise RuntimeError(
                f"CONDA_ENV_CACHE_MAX_GB={configured} is below the minimum "
                f"of {self._CONDA_ENV_CACHE_MIN_GB} GB — a single model with 3 "
                f"framework-family envs (torch, onnx, tensorflow) at ~10 GB "
                f"apiece would evict envs it itself needs."
            )
        self._conda_env_cache_max_gb = configured

        # Serializes ``mamba env create`` so only one custom-env test
        # can be building conda envs at a time on THIS replica. Two
        # concurrent solves would compete for memory + fight over
        # mamba's package cache locks. Cache-hit path
        # (all envs already exist) is fast; the lock is uncontended
        # in the common case.
        self._env_build_lock = asyncio.Lock()

        # Async test-job registry. Populated by every ``test()`` call
        # (both async_mode=True and async_mode=False so state can be
        # inspected via ``get_test_status`` for either). Keys are
        # opaque job IDs (``tj-<hex12>``); values track state,
        # metadata, and an asyncio.Future that resolves on
        # completion. Purely in-memory per replica; jobs from a
        # killed replica just disappear, ``get_test_status`` raises
        # KeyError for them and callers can retry / fall back to
        # sync mode.
        self._test_jobs: Dict[str, dict] = {}
        self._test_jobs_ttl_sec: int = 24 * 3600
        # Map from ``id(asyncio.Task)`` → job dict, so the recursive
        # ``test(async_mode=False)`` call inside the async-mode
        # wrapper finds the same job the wrapper allocated. Plain
        # dict — pickles fine (unlike ContextVar, which cloudpickle
        # trips on when Ray serialises the Serve deployment class).
        self._task_to_job: Dict[int, dict] = {}

        # Get replica identifier for logging
        try:
            from ray import serve as _serve
            self.replica_id = _serve.get_replica_context().replica_tag
        except Exception:
            self.replica_id = "unknown"

        # Set up model cache
        self.model_cache = ModelCache(
            cache_size_in_gb=cache_size_in_gb,
            replica_id=self.replica_id,
        )

        logger.info(
            f"🚀 {self.__class__.__name__} initialized with models directory: "
            f"{self.model_cache.cache_dir} (cache_size={self.model_cache.cache_size_bytes / (1024*1024*1024):.3f} GB)"
        )
        logger.info(
            f"🐍 Conda env cache: max {self._conda_env_cache_max_gb:.0f} GB, "
            f"age sweep every {self._CONDA_ENV_MAX_AGE_DAYS} days"
        )

    # === BioEngine App Method - will be called when the deployment is started ===

    @bioengine.async_init
    async def _async_init(self) -> None:
        self.hypha_client = await connect_to_server(
            {
                "server_url": self.server_url,
                "token": self._hypha_token,
            }
        )
        self.artifact_manager = await self.hypha_client.get_service(
            "public/artifact-manager"
        )
        self.s3_controller = await self.hypha_client.get_service("public/s3-storage")
        logger.info(f"Connected to Hypha Server at {self.server_url}")

    # === Ray Serve Health Check Method - will be called periodically to check the health of the deployment ===

    async def _check_runtime_available(self) -> None:
        """Ping the runtime deployment and raise immediately if it is not responding.

        Called at the top of every GPU method so callers get a fast, clear error
        instead of a 30 s+ Ray timeout when the GPU runtime is not running.
        """
        try:
            await asyncio.wait_for(
                self.runtime.ping(),
                timeout=2.0,
            )
        except Exception as e:
            raise RuntimeError(
                "GPU runtime deployment is not available. "
                "Inference, test, and validate are unavailable until the runtime starts."
            ) from e

    @bioengine.health_check
    async def _health_check(self) -> None:
        # Test connection to the Hypha server only — runtime availability is
        # checked per-call in GPU methods so partial registration is preserved
        # (service stays registered for CPU-only methods when GPU is down).
        await self.hypha_client.echo("ping")

    # === Internal Helper Methods ===

    def _get_bioimageio_versions(self) -> Dict[str, str]:
        """
        Return identities the test cache invalidates on.

        Includes installed versions of the bioimageio packages and the
        model-runner artifact's own version — so the existing env-mismatch
        check in ``test()`` re-runs when the runner itself is upgraded.
        """
        from importlib.metadata import PackageNotFoundError, version

        package_names = ["bioimageio.core", "bioimageio.spec"]
        versions: Dict[str, str] = {}

        for package_name in package_names:
            try:
                versions[package_name] = version(package_name)
            except PackageNotFoundError:
                versions[package_name] = "not-installed"

        versions["bioimage-io/model-runner"] = os.environ.get(
            "HYPHA_ARTIFACT_VERSION", "unknown"
        )

        logger.debug(f"📦 Bioimage.io package versions: {versions}")
        return versions

    def _stamp_runtime_versions_in_test_env(self, test_report: dict) -> dict:
        """Upsert runtime-identity rows in ``test_report['env']``.

        The test cache invalidation in ``test()`` reuses this env list to
        detect when a stored report was produced under different runtime
        versions; downstream consumers (e.g. bioimage.io CI) also read it
        from the published report to drive their own cache keys.
        """
        rows = [
            ("bioengine", __version__),
            (
                "bioimage-io/model-runner",
                os.environ.get("HYPHA_ARTIFACT_VERSION", "unknown"),
            ),
        ]

        env = test_report.get("env")
        if not isinstance(env, list):
            env = []
        else:
            env = list(env)

        for name, version_value in rows:
            row_index: Optional[int] = None
            for i, row in enumerate(env):
                if (
                    isinstance(row, (list, tuple))
                    and len(row) >= 1
                    and str(row[0]) == name
                ):
                    row_index = i
                    break

            new_row = [name, version_value, "", ""]
            if row_index is None:
                env.append(new_row)
            else:
                existing_row = env[row_index]
                if isinstance(existing_row, tuple):
                    existing_row = list(existing_row)
                while len(existing_row) < 4:
                    existing_row.append("")
                existing_row[1] = version_value
                env[row_index] = existing_row

        test_report["env"] = env
        return test_report

    # === Subprocess env hardening ===

    _SENSITIVE_ENV_NEEDLES = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY")

    def _safe_subprocess_env(self) -> Dict[str, str]:
        """Return ``os.environ`` minus obviously-sensitive entries.

        Test subprocesses (bioimageio CLI, mamba, the standard-env
        ``sys.executable -c`` wrapper) don't need the RuntimeApp's
        Hypha credentials. Denylist any env-var name containing
        ``TOKEN`` / ``SECRET`` / ``PASSWORD`` / ``CREDENTIAL`` /
        ``API_KEY`` — covers ``HYPHA_TOKEN``,
        ``BIOENGINE_ARTIFACT_TOKEN``, ``BIOIMAGE_IO_TOKEN``, and any
        generic cloud credentials leaking through. Everything else
        (``PATH``, ``HOME``, ``TMPDIR``, ``PYTHONPATH``, ``LANG``,
        ``CUDA_VISIBLE_DEVICES``, ``BIOENGINE_APP_DIR``,
        ``HYPHA_ARTIFACT_VERSION`` — anything the bioengine bootstrap
        or the bioimageio CLI needs to identify itself) is
        preserved.
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
        by EntryApp are visible to RuntimeApp on the same path.
        """
        env_vars = self._safe_subprocess_env()
        mamba_root = Path(os.environ["HOME"]) / ".bioengine-conda"
        (mamba_root / "envs").mkdir(parents=True, exist_ok=True)
        (mamba_root / "pkgs").mkdir(parents=True, exist_ok=True)
        env_vars["CONDA_ENVS_PATH"] = str(mamba_root / "envs")
        env_vars["CONDA_PKGS_DIRS"] = str(mamba_root / "pkgs")
        return env_vars

    # === Conda env pre-creation for custom_environment tests ===

    # === Conda env cache housekeeping (sweep + LRU eviction) ===

    def _conda_envs_dir(self) -> Path:
        return Path(os.environ["HOME"]) / ".bioengine-conda" / "envs"

    async def _du_bytes(self, path: Path) -> int:
        """Compact wrapper around ``du -sk`` — used for env sizing.

        Walking a 10 GB env with Python's ``rglob`` + ``stat`` takes
        several seconds per env; ``du`` finishes in ~100 ms. Returns
        0 on any error (missing path, du unavailable) so callers can
        continue safely.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "du",
                "-sk",
                str(path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                return 0
            return int(stdout.decode().split()[0]) * 1024
        except Exception:
            return 0

    async def _mamba_env_remove(self, env_name: str, env_vars: Dict[str, str]) -> None:
        """Non-blocking ``mamba env remove``. Best-effort — logs
        warnings on failure but never raises, since the caller
        (sweep + LRU eviction) shouldn't fail the whole test call
        because one stale env couldn't be reaped.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "mamba",
                "env",
                "remove",
                "-n",
                env_name,
                "--yes",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env_vars,
            )
            _, stderr = await proc.communicate()
            if proc.returncode == 0:
                logger.info(f"🧹 Removed conda env {env_name}")
            else:
                tail = (stderr or b"").decode(errors="replace")[-300:]
                logger.warning(
                    f"⚠️ Could not remove conda env {env_name}: rc="
                    f"{proc.returncode}, tail={tail!r}"
                )
        except Exception as e:
            logger.warning(f"⚠️ Exception removing conda env {env_name}: {e}")

    def _touch_env(self, env_name: str) -> None:
        """Bump the env directory's mtime so LRU tracks actual use
        (mamba's ``run -n <env>`` only reads, so mtime wouldn't
        move otherwise).
        """
        try:
            env_dir = self._conda_envs_dir() / env_name
            if env_dir.exists():
                os.utime(env_dir, None)
        except OSError as e:
            logger.debug(f"could not touch env {env_name}: {e}")

    async def _sweep_old_envs_if_due(self, env_vars: Dict[str, str]) -> None:
        """Age-based sweep: at most once per ``_CONDA_ENV_MAX_AGE_DAYS``.

        Uses a ``.last-sweep`` marker file next to the envs dir.
        First call on a fresh replica also runs (marker missing).
        """
        marker = self._conda_envs_dir().parent / ".last-sweep"
        now = time.time()
        try:
            age_hours = (now - marker.stat().st_mtime) / 3600.0
        except FileNotFoundError:
            age_hours = float("inf")
        if age_hours < self._CONDA_ENV_MAX_AGE_DAYS * 24:
            return

        cutoff = now - self._CONDA_ENV_MAX_AGE_DAYS * 86400
        envs_dir = self._conda_envs_dir()
        if not envs_dir.exists():
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            return

        candidates = []
        for entry in envs_dir.iterdir():
            if not entry.is_dir():
                continue
            try:
                if entry.stat().st_mtime < cutoff:
                    candidates.append(entry.name)
            except FileNotFoundError:
                continue
        if candidates:
            logger.info(
                f"🧹 Weekly conda env sweep: removing "
                f"{len(candidates)} env(s) older than "
                f"{self._CONDA_ENV_MAX_AGE_DAYS} days"
            )
            await asyncio.gather(
                *(self._mamba_env_remove(name, env_vars) for name in candidates)
            )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()

    async def _evict_lru_if_over_budget(
        self,
        protected: List[str],
        env_vars: Dict[str, str],
    ) -> None:
        """LRU eviction: while the on-disk cache exceeds
        ``_conda_env_cache_max_gb``, remove the least-recently-used
        env that isn't in ``protected`` (the envs the current test
        needs).

        Approximate: uses ``du -sk`` per env, not fsync-accurate,
        and ``mamba env remove`` on an in-use env silently fails.
        Both are acceptable — the ceiling is a soft target.
        """
        envs_dir = self._conda_envs_dir()
        if not envs_dir.exists():
            return
        max_bytes = int(self._conda_env_cache_max_gb * 1024**3)

        entries = []
        for entry in envs_dir.iterdir():
            if not entry.is_dir():
                continue
            try:
                mtime = entry.stat().st_mtime
            except FileNotFoundError:
                continue
            entries.append((entry.name, entry, mtime))

        sizes = await asyncio.gather(*(self._du_bytes(p) for _, p, _ in entries))
        total = sum(sizes)
        if total <= max_bytes:
            return

        logger.info(
            f"🧹 Conda env cache at {total / 1024**3:.1f} GB > "
            f"{self._conda_env_cache_max_gb:.0f} GB ceiling — LRU eviction"
        )
        protected_set = set(protected)
        ranked = sorted(
            [
                (mtime, name, size)
                for (name, _, mtime), size in zip(entries, sizes)
                if name not in protected_set
            ],
            key=lambda t: t[0],
        )
        for mtime, name, size in ranked:
            if total <= max_bytes:
                break
            await self._mamba_env_remove(name, env_vars)
            total -= size

    # === Async test-job registry ===

    def _sweep_expired_test_jobs(self) -> None:
        """Drop finished jobs older than ``_test_jobs_ttl_sec``.

        Runs opportunistically on each new job creation — cheaper than a
        background task and good enough for a low-traffic registry.
        """
        now = time.time()
        expired = [
            jid
            for jid, j in self._test_jobs.items()
            if j.get("completed_at") is not None
            and (now - j["completed_at"]) > self._test_jobs_ttl_sec
        ]
        for jid in expired:
            self._test_jobs.pop(jid, None)

    def _new_test_job(self, model_id: str, custom_environment: bool) -> dict:
        """Create a job record and return it. Sweeps expired jobs on the
        way in.
        """
        self._sweep_expired_test_jobs()
        job_id = f"tj-{uuid.uuid4().hex[:12]}"
        now = time.time()
        loop = asyncio.get_running_loop()
        job = {
            "job_id": job_id,
            "model_id": model_id,
            "custom_environment": custom_environment,
            "state": "queued",
            "started_at": now,
            "updated_at": now,
            "completed_at": None,
            "result": None,
            "error": None,
            "future": loop.create_future(),
        }
        self._test_jobs[job_id] = job
        return job

    def _update_test_job(
        self,
        job: dict,
        *,
        state: Optional[str] = None,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """Idempotent state update. Sets ``completed_at`` and resolves the
        future on terminal states.
        """
        if state is not None:
            job["state"] = state
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
        job["updated_at"] = time.time()
        if state in ("completed", "failed"):
            job["completed_at"] = job["updated_at"]
            fut = job.get("future")
            if fut is not None and not fut.done():
                if state == "failed":
                    fut.set_exception(RuntimeError(error or "test job failed"))
                else:
                    fut.set_result(result)

    def _job_public_view(self, job: dict) -> dict:
        """Serialize a job dict for the RPC response — strips the
        internal ``future`` handle and computes derived fields.
        """
        now = time.time()
        state = job["state"]
        queue_position = 0
        if job["custom_environment"] and state in ("queued", "env_setup"):
            queue_position = sum(
                1
                for other in self._test_jobs.values()
                if other is not job
                and other["custom_environment"]
                and other["state"] in ("queued", "env_setup")
                and other["started_at"] < job["started_at"]
            )
        completed_at = job["completed_at"]
        elapsed_end = completed_at if completed_at is not None else now
        return {
            "job_id": job["job_id"],
            "model_id": job["model_id"],
            "custom_environment": job["custom_environment"],
            "state": state,
            "queue_position": queue_position,
            "started_at": job["started_at"],
            "updated_at": job["updated_at"],
            "completed_at": completed_at,
            "elapsed_seconds": round(elapsed_end - job["started_at"], 2),
            "result": job["result"],
            "error": job["error"],
        }

    # === Conda env name computation (matches bioimageio.core) ===

    def _compute_conda_env_name(self, wf) -> tuple:
        """Return ``(env_name, encoded_yaml)`` for a weight-format entry.

        This is a MECHANICAL REPRODUCTION of the env-name algorithm inline
        in ``bioimageio.core 0.10.4`` at
        ``src/bioimageio/core/_resource_tests.py:425-435``:

            conda_env = get_conda_env(entry=wf)
            conda_env.name = None
            dumped_env = conda_env.model_dump(mode="json", exclude_none=True)
            env_io = StringIO()
            write_yaml(dumped_env, file=env_io)
            encoded_env = env_io.getvalue().encode()
            env_name = hashlib.sha256(encoded_env).hexdigest()

        Bit-identical hashes are load-bearing: the whole point of
        pre-building envs on EntryApp is that ``test_description``
        on RuntimeApp finds an env with the SAME name and skips its
        own ``mamba env create`` step. If any of the four upstream
        primitives changes (a new pydantic dump mode, a different YAML
        flow style, added metadata fields), the two silently diverge
        and every custom-env test starts paying the cold-build cost
        again. This function is the anchor — if hashes stop matching,
        this is the file to update.

        Reuses the helpers ``bioimageio.spec`` exports:
        ``get_conda_env``, ``write_yaml``, ``BioimageioCondaEnv``. The
        hash calculation itself has no upstream export — it is inline
        in ``_test_in_env`` — hence this local helper.
        """
        from io import StringIO
        import hashlib

        from bioimageio.spec import get_conda_env
        from bioimageio.spec._internal.io_utils import write_yaml

        conda_env = get_conda_env(entry=wf)
        conda_env.name = None
        dumped_env = conda_env.model_dump(mode="json", exclude_none=True)
        buf = StringIO()
        write_yaml(dumped_env, file=buf)
        encoded = buf.getvalue().encode()
        env_name = hashlib.sha256(encoded).hexdigest()
        return env_name, encoded

    # === Conda env pre-creation for custom_environment tests ===

    async def _prebuild_conda_envs(self, rdf_path: str, model_id: str) -> None:
        """CPU-side, non-blocking pre-creation of every conda env the
        model needs for ``custom_environment=True`` tests.

        Runs entirely on this EntryApp replica (no GPU held). Loads
        the model description, walks the present weight formats,
        computes each weight format's ``BioimageioCondaEnv`` spec
        via ``bioimageio.spec.get_conda_env``, and hashes the dumped
        YAML with SHA256 — matching ``bioimageio.core 0.10.4``'s
        env-name algorithm exactly so the pre-created envs are the
        same names ``test_description`` looks for on the RuntimeApp
        side.

        Each ``mamba env create`` is spawned via
        ``asyncio.create_subprocess_exec`` — a real async subprocess
        awaited through the event loop, so this call NEVER blocks
        the loop, no matter how long mamba takes. Multiple weight
        formats' envs build **concurrently** via ``asyncio.gather``
        rather than sequentially — a model with pytorch+onnx envs
        finishes in one env's wall-clock time, not two.

        Existence check runs first (also parallel) so already-cached
        envs on the PVC are skipped instantly — first call to a
        given model is slow, subsequent calls are near-free.
        """
        from bioimageio.spec import load_description

        # Load description off-loop — file I/O + validation.
        descr = await asyncio.to_thread(load_description, rdf_path)

        # Enumerate present weight formats. Skip ``tensorflow_js``
        # because ``bioimageio.core`` explicitly rejects testing
        # it; skip anything the descr doesn't carry at all (v0.4
        # pydantic-v2 raises AttributeError on missing fields, so
        # guard with getattr default).
        weight_formats = [
            "pytorch_state_dict",
            "torchscript",
            "keras_hdf5",
            "onnx",
            "tensorflow_saved_model_bundle",
            "keras_v3",
        ]
        env_specs: List[tuple] = []  # (wf, env_name, encoded_yaml)
        for wf_name in weight_formats:
            wf = getattr(descr.weights, wf_name, None)
            if not wf:
                continue
            env_name, encoded = self._compute_conda_env_name(wf)
            env_specs.append((wf_name, env_name, encoded))

        if not env_specs:
            logger.info(f"🐍 No custom conda envs to prebuild for '{model_id}'.")
            return

        env_vars = self._mamba_env_vars()
        needed_env_names = [name for _, name, _ in env_specs]

        # ``_env_build_lock`` serializes env-build across concurrent
        # ``test()`` calls on this Entry replica. Cache-hit path
        # (all envs already exist) still holds the lock briefly, but
        # that's OK — mamba probes take <1s and eviction/sweep are
        # cheap when there's nothing to do.
        async with self._env_build_lock:
            # Weekly age-based sweep — remove envs untouched for
            # more than ``_CONDA_ENV_MAX_AGE_DAYS`` days. Marker
            # file gates frequency so this is at most a per-week
            # cost.
            await self._sweep_old_envs_if_due(env_vars)

            # LRU eviction if we're already over the ceiling. Envs
            # this test itself needs (``needed_env_names``) are
            # protected regardless of age.
            await self._evict_lru_if_over_budget(needed_env_names, env_vars)

            logger.info(
                f"🐍 Prebuild: checking {len(env_specs)} conda env(s) for "
                f"'{model_id}' ({[wf for wf, _, _ in env_specs]})"
            )
            exists_results = await asyncio.gather(
                *(self._mamba_env_exists(name, env_vars) for _, name, _ in env_specs)
            )

            to_create = [
                (wf, name, enc)
                for (wf, name, enc), exists in zip(env_specs, exists_results)
                if not exists
            ]
            cached = [wf for (wf, _, _), e in zip(env_specs, exists_results) if e]
            if cached:
                logger.info(f"✅ Reusing cached conda env(s) for {cached}.")

            if to_create:
                logger.info(
                    f"🐍 Building {len(to_create)} missing conda env(s) concurrently: "
                    f"{[wf for wf, _, _ in to_create]}"
                )
                await asyncio.gather(
                    *(
                        self._mamba_env_create(name, enc, env_vars, wf=wf)
                        for wf, name, enc in to_create
                    )
                )
                logger.info(
                    f"✅ Prebuilt {len(to_create)} conda env(s) for '{model_id}'."
                )

            # Bump mtime on every env this test uses (cached or
            # just built) so subsequent LRU eviction treats them as
            # "recently used" — mamba's ``run -n <env>`` only reads,
            # so mtime wouldn't move otherwise.
            for name in needed_env_names:
                self._touch_env(name)

    async def _mamba_env_exists(self, env_name: str, env_vars: Dict[str, str]) -> bool:
        """Non-blocking existence check via
        ``mamba run -n <env_name> python --version``. Returns True
        iff the child exits 0.
        """
        proc = await asyncio.create_subprocess_exec(
            "mamba",
            "run",
            "-n",
            env_name,
            "python",
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env_vars,
        )
        await proc.communicate()
        return proc.returncode == 0

    async def _mamba_env_create(
        self,
        env_name: str,
        yaml_bytes: bytes,
        env_vars: Dict[str, str],
        wf: str = "",
    ) -> None:
        """Non-blocking ``mamba env create`` from encoded YAML bytes.

        Writes the env spec to a tempfile (mamba only accepts
        ``--file=<path>``, not stdin) and awaits the child. On
        non-zero exit, raises a compact error with the last ~1 KB of
        the child's stderr — enough to diagnose solver failures
        without dumping the full libmamba trace.
        """
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".yaml", delete=False
        ) as tmp:
            tmp.write(yaml_bytes)
            tmp_path = tmp.name
        try:
            logger.info(f"🐍 mamba env create ({wf!r}) → {env_name}")
            proc = await asyncio.create_subprocess_exec(
                "mamba",
                "env",
                "create",
                "--yes",
                f"--file={tmp_path}",
                f"--name={env_name}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env_vars,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                tail = (stderr or b"").decode(errors="replace")[-1000:]
                raise RuntimeError(
                    f"mamba env create failed for weight format "
                    f"{wf!r} (env {env_name}): {tail}"
                )
            logger.info(f"✅ Built conda env for {wf!r}: {env_name}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def _get_download_url(self, file_path: str) -> str:
        # Temporary S3 file path — resolve to a presigned download URL
        try:
            download_url = await self.s3_controller.get_file(
                file_path=file_path, use_proxy=True
            )
            return download_url
        except Exception as e:
            raise RuntimeError(
                f"Failed to get download URL for temporary file '{file_path}': {e}"
            ) from e

    async def _load_image_from_source(self, source: str) -> np.ndarray:
        """
        Load an image from a URL or a temporary S3 file path into a numpy array.

        Accepts either:
        - A direct HTTP/HTTPS URL (fetched as-is), or
        - A temporary file path returned by ``get_upload_url`` (resolved to a
          presigned S3 download URL via BioEngine S3 storage).

        The file content is decoded based on the file extension.

        Args:
            source: Direct URL (``http://…`` / ``https://…``) or temporary
                    file path returned by ``get_upload_url``

        Returns:
            np.ndarray: NumPy array containing the image data

        Raises:
            FileNotFoundError: If the remote resource does not exist or has expired
            ValueError: If the file extension is not supported
        """
        # Check file extension for supported formats
        ext = Path(
            source.split("?")[0]
        ).suffix.lower()  # strip query string for URL sources
        if ext not in SUPPORTED_FILES_TYPES.__args__:
            raise ValueError(
                f"Unsupported file extension '{ext}' in source '{source}'. "
                f"Supported extensions: {SUPPORTED_FILES_TYPES.__args__}"
            )

        logger.info(f"📥 Loading image from source '{source}'...")

        if source.startswith(("http://", "https://")):
            # Direct URL — fetch without S3 indirection
            download_url = source
        else:
            download_url = await self._get_download_url(source)

        # Download file content
        response = await self.model_cache._get_url_with_retry(download_url, params=None)

        if response.status_code == 404:
            raise FileNotFoundError(f"Source '{source}' does not exist or has expired.")
        try:
            response.raise_for_status()
        except Exception as e:
            raise FileNotFoundError(f"Failed to download source '{source}': {e}") from e

        # Parse and load based on file extension
        try:
            buffer = BytesIO(response.content)
            if ext == ".npy":
                array = await asyncio.to_thread(np.load, buffer)
            else:
                import imageio.v3 as iio

                array = await asyncio.to_thread(iio.imread, buffer)
        except Exception as e:
            raise ValueError(
                f"Failed to parse image from source '{source}': {e}"
            ) from e

        logger.info(
            f"✅ Loaded image from '{source}': shape={array.shape}, dtype={array.dtype}"
        )
        return array

    async def _save_array_to_temp_file(self, array: np.ndarray) -> str:
        """
        Save a NumPy array to a temporary ``.npy`` file in S3 and return a presigned download URL.

        The array is serialised with ``numpy.save`` and uploaded to BioEngine S3 storage using a
        presigned upload URL obtained from ``get_upload_url``. The file is given a 1-hour TTL.

        Args:
            array: NumPy array to save

        Returns:
            str: Presigned download URL for the uploaded ``.npy`` file (valid for 1 hour)

        Raises:
            RuntimeError: If saving the array to a temporary file fails
        """
        try:
            upload_info = await self.get_upload_url(file_type=".npy")
            logger.info(
                f"💾 Saving array (shape: {array.shape}, dtype: {array.dtype}) "
                f"to temporary file '{upload_info['file_path']}'..."
            )
            buffer = BytesIO()
            np.save(buffer, array)
            await self.model_cache.client.put(
                upload_info["upload_url"], data=buffer.getvalue()
            )
            logger.info(
                f"✅ Array saved to temporary file '{upload_info['file_path']}'"
            )

            download_url = await self._get_download_url(upload_info["file_path"])
        except Exception as e:
            raise RuntimeError(f"Failed to save array to temporary file: {e}") from e

        return download_url

    # === Exposed BioEngine App Methods - all methods decorated with @bioengine.method will be exposed as API endpoints ===
    # Note: Parameter type hints and docstrings will be used to generate the API documentation.

    @bioengine.method
    async def get_version(self) -> Dict[str, str]:
        """Return the artifact identity this replica was deployed from.

        Callers (e.g. bioimage.io CI cache invalidation) use this to detect
        when a stored test report was produced under a different runner
        version and should be re-tested.
        """
        return {
            "artifact_id": os.environ.get("HYPHA_ARTIFACT_ID", "unknown"),
            "version": os.environ.get("HYPHA_ARTIFACT_VERSION", "unknown"),
            "bioengine_version": __version__,
        }

    @bioengine.method
    async def search_models(
        self,
        keywords: Optional[List[str]] = Field(
            None,
            description="List of keywords to filter models by (e.g., ['cell', 'nuclei', 'segmentation']",
        ),
        limit: Optional[int] = Field(
            10, description="Maximum number of models to return in the search results"
        ),
        ignore_checks: Optional[bool] = Field(
            False,
            description="Whether to ignore bioengine inference checks and return all models (True) or only models that passed checks (False)",
        ),
    ) -> List[Dict[str, str]]:
        """
        Search for models in the bioimage.io collection.

        Returns a list of model identifiers with their descriptions that match the search query.
        """
        logger.info(f"🔍 Searching models with keywords={keywords}, limit={limit}")
        collection_id = "bioimage-io/bioimage.io"

        try:
            results = await self.artifact_manager.list(
                parent_id=collection_id,
                filters={"type": "model"},
                keywords=keywords,
                limit=limit,
                stage=False,
            )

            if not ignore_checks:
                collection = await self.artifact_manager.read(collection_id)
                bioengine_inference_results = collection["manifest"][
                    "bioengine_inference"
                ]
                runnable_models = {
                    model_id
                    for model_id, result in bioengine_inference_results.items()
                    if result.get("status") == "passed"
                }

            models = []
            for artifact in results:
                manifest = artifact["manifest"]
                if not ignore_checks and artifact["alias"] not in runnable_models:
                    continue
                models.append(
                    {
                        "model_id": artifact["alias"],
                        "description": manifest.get("description", ""),
                    }
                )

            logger.info(f"✅ Found {len(models)} models matching query.")
            return models

        except Exception as e:
            error_msg = f"Failed to search models: {e}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    @bioengine.method
    async def get_model_rdf(
        self,
        model_id: str = Field(
            ...,
            description="Unique identifier of the bioimage.io model (e.g., 'ambitious-ant')",
        ),
        stage: Optional[bool] = Field(
            False,
            description="Whether to get RDF from the staged version of the model (True) or the committed version (False)",
        ),
    ) -> Dict[str, Union[str, int, float, List, Dict]]:
        """
        Retrieve the Resource Description Framework (RDF) metadata for a bioimage.io model.

        Returns:
            Dictionary containing the complete RDF metadata structure with nested
            configuration for inputs, outputs, preprocessing, postprocessing, and model weights

        Raises:
            ValueError: If model_id is invalid or model not found
            RuntimeError: If download fails
        """
        logger.info(f"📋 Downloading RDF for model '{model_id}' (stage={stage}).")

        rdf_url = f"{self.server_url}/bioimage-io/artifacts/{model_id}/files/rdf.yaml"
        response = await self.model_cache._get_url_with_retry(
            rdf_url, params={"stage": str(stage).lower()}
        )

        if response.status_code == 404 and stage:
            # If staged version doesn't exist, try with stage=false
            logger.warning(
                f"⚠️ Staged RDF not found for model '{model_id}', trying committed version..."
            )
            response = await self.model_cache._get_url_with_retry(
                rdf_url, params={"stage": "false"}
            )

        try:
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to download RDF from {rdf_url}") from e

        model_rdf = await asyncio.to_thread(yaml.safe_load, response.text)

        logger.info(f"✅ Successfully downloaded RDF for model '{model_id}'.")
        return model_rdf

    @bioengine.method
    async def get_model_documentation(
        self,
        model_id: str = Field(
            ...,
            description="Unique identifier of the bioimage.io model (e.g., 'ambitious-ant')",
        ),
        stage: Optional[bool] = Field(
            False,
            description="Whether to fetch documentation from the staged version (True) or committed version (False)",
        ),
    ) -> Optional[str]:
        """
        Retrieve the documentation text for a bioimage.io model.

        Reads the 'documentation' field from the model RDF. If the field is set,
        downloads the referenced file from the artifact and returns its content.

        Returns:
            The documentation file content as a string, or None if the
            'documentation' field is absent/None or the file does not exist.
        """
        rdf = await self.get_model_rdf(model_id=model_id, stage=stage)

        doc_path = rdf.get("documentation")
        if not doc_path:
            logger.info(f"📄 No documentation field in RDF for model '{model_id}'.")
            return None

        doc_url = f"{self.server_url}/bioimage-io/artifacts/{model_id}/files/{doc_path}"
        logger.info(f"📄 Downloading documentation '{doc_path}' for model '{model_id}'.")

        response = await self.model_cache._get_url_with_retry(
            doc_url, params={"stage": str(stage).lower()}
        )

        if response.status_code == 404:
            logger.info(
                f"📄 Documentation file '{doc_path}' not found in artifact for model '{model_id}'."
            )
            return None

        try:
            response.raise_for_status()
        except Exception as e:
            logger.warning(f"⚠️ Failed to download documentation for model '{model_id}': {e}")
            return None

        logger.info(f"✅ Successfully downloaded documentation for model '{model_id}'.")
        return response.text

    @bioengine.method
    async def validate(
        self,
        rdf_dict: Dict[str, Union[str, int, float, List, Dict]] = Field(
            ..., description="Complete RDF dictionary structure to validate"
        ),
        known_files: Optional[Dict[str, str]] = Field(
            None,
            description="Mapping of relative file paths to their content hashes for validating file references within the RDF",
        ),
    ) -> Dict[str, Union[bool, str]]:
        """
        Validate a model Resource Description Framework (RDF) against bioimage.io specifications.

        Returns:
            Validation result containing:
            - success: Boolean indicating overall validation status
            - details: Detailed validation report with specific issues or confirmation

        Note:
            This method performs format validation only (perform_io_checks=False).
            File existence is not verified unless known_files mapping is provided.
        """
        from bioimageio.spec import ValidationContext, validate_format

        logger.info(
            f"🔬 Validating RDF (known_files: {len(known_files or {})} files)..."
        )

        ctx = ValidationContext(perform_io_checks=False, known_files=known_files or {})
        summary = await asyncio.to_thread(validate_format, rdf_dict, context=ctx)

        result = {
            "success": summary.status == "valid-format",
            "details": summary.format(),
        }

        logger.info(f"✅ RDF validation {'passed' if result['success'] else 'failed'}.")
        return result

    @bioengine.method
    async def test(
        self,
        model_id: str = Field(
            ..., description="Unique identifier of the bioimage.io model to test"
        ),
        stage: Optional[bool] = Field(
            False,
            description="Whether to get the staged version of the model (True) or the committed version (False)",
        ),
        custom_environment: Optional[bool] = Field(
            False,
            description="If True, run the test inside the conda environment declared by the model's own weights description (``bioimageio.core`` ``runtime_env='as-described'``, backed by ``mamba`` for env creation, env removed after the call). If False (default), run in the model-runner RuntimeApp's own venv — the same interpreter that serves inference.",
        ),
        skip_cache: Optional[bool] = Field(
            False,
            description="Force a complete model package re-download and bypass cached test results before testing",
        ),
        attach_test_report: Optional[bool] = Field(
            False,
            description="If True, upload ``test_report.json`` to the model artifact and add a compact ``test_summary`` entry to its manifest after testing. Renamed from the older ``publish_test_report`` to avoid confusion with the bioimage.io model zoo ``staged`` / ``published`` lifecycle — this parameter does NOT change the artifact's publication status.",
        ),
        hypha_token: Optional[str] = Field(
            None,
            description="Caller's personal Hypha token, required when ``attach_test_report=True``. All artifact writes for the attach are performed under this token — the runner does not use its own workspace credentials to edit user artifacts.",
        ),
        async_mode: Optional[bool] = Field(
            False,
            description="If True, schedule the test as a background job and return {job_id, state} immediately. Poll ``get_test_status(job_id)`` for progress and the final report. Job state is in-memory per replica and drops on replica restart. If False (default), block until the test completes and return the full report — behavior unchanged from previous versions.",
        ),
    ) -> Dict[str, Union[str, bool, List, Dict]]:
        """
        Execute comprehensive model testing using ``bioimageio.core.test_description``.

        Caching behavior:
        - Cached test reports are locally stored at ``<model_package>/.test_cache.json``.
        - Cached results are reused only when ``skip_cache=False`` AND the model
            package has not changed (same ``latest_remote_modified``) AND the cached
            ``test_report['env']`` versions for ``bioimageio.core`` and
            ``bioimageio.spec`` match the currently installed versions.
        - ``skip_cache=True`` forces a complete model package re-download,
            bypasses cached test results, and runs a fresh test.

        Environment mode:
        - ``custom_environment=False`` (default): the test runs in the
            RuntimeApp's own venv — the same interpreter that will serve
            ``infer()``. Fast, and validates the environment the caller
            will actually hit in production.
        - ``custom_environment=True``: the test runs inside the conda
            environment declared by the model's own weights description
            (``bioimageio.core`` ``runtime_env="as-described"``). Env
            creation is driven by ``mamba``; the env is removed after
            the call on both success and failure paths, so no multi-GB
            per-model envs accumulate on the pod.

        Attach-to-artifact behavior:
        - If ``attach_test_report=True``, a compact ``test_summary`` entry is
            written to the artifact manifest and ``test_report.json`` is
            uploaded to the artifact. This attaches the report to the
            artifact — it does NOT alter the artifact's ``staged`` /
            ``published`` lifecycle status.
        - **All artifact reads and writes on the attach path run under
            the caller's ``hypha_token``**, not the runner's workspace
            credentials. The caller must have write access to the
            target artifact; if the token is missing or under-privileged
            the attach fails with a clear ``PermissionError`` /
            hypha-side rejection. The runner's own token never touches
            user artifacts on the attach path.
        - **The runner commits only when the artifact was NOT already staged
            before the test call.** If a stage was already open — whoever put
            it there, whatever it contains, whatever ``manifest.status`` says —
            the runner adds the test report to the existing stage and leaves
            it open. The artifact owner / reviewer commits later, atomically,
            alongside whatever changes they already had in flight. This
            prevents the runner from accidentally publishing someone else's
            pending edits (bioimage.io #0006 amusing-angelfish incident).
        """
        import aiofiles

        await self._check_runtime_available()
        # Fail fast if the caller wants to attach without providing a
        # token — writes go through the caller's token, not the
        # runner's workspace credentials, so no token → no attach.
        # Doing this before compute avoids burning GPU time on a call
        # that can't complete.
        if attach_test_report and not hypha_token:
            raise PermissionError(
                f"Cannot attach test report to '{model_id}': "
                f"``attach_test_report=True`` requires a personal "
                f"``hypha_token`` with write access to the target "
                f"artifact's workspace. All artifact writes on the "
                f"attach path are performed under the caller's token "
                f"so edits are attributed to and authorized as the "
                f"actual user — the runner never uses its own "
                f"workspace credentials for user artifacts. Get a "
                f"personal token from https://hypha.aicell.io "
                f"(Profile → Access tokens) and pass it as "
                f"``hypha_token=<token>``, or call test() with "
                f"``attach_test_report=False``."
            )

        # Find (or allocate) the state-tracking job. When we're inside
        # the async-mode wrapper (``async_mode=True`` recursed here as
        # ``async_mode=False``), the wrapper's asyncio task carries a
        # ``_task_to_job`` entry the recursive call picks up — state
        # updates land in the right registry entry. Direct sync-mode
        # callers get a fresh best-effort job (they receive the report
        # directly on return; the job entry is available via
        # ``get_test_status`` if the caller wants queue/progress info
        # after the fact).
        current_task = asyncio.current_task()
        job = self._task_to_job.get(id(current_task)) if current_task else None
        if job is None:
            job = self._new_test_job(model_id, custom_environment)

        if async_mode:
            async def _bg_execute():
                bg_task = asyncio.current_task()
                self._task_to_job[id(bg_task)] = job
                try:
                    result = await self.test(
                        model_id=model_id,
                        stage=stage,
                        custom_environment=custom_environment,
                        skip_cache=skip_cache,
                        attach_test_report=attach_test_report,
                        hypha_token=hypha_token,
                        async_mode=False,
                    )
                    self._update_test_job(job, state="completed", result=result)
                except Exception as exc:
                    logger.error(
                        f"❌ Async test job {job['job_id']} for '{model_id}' failed: {exc}"
                    )
                    self._update_test_job(job, state="failed", error=str(exc))
                finally:
                    self._task_to_job.pop(id(bg_task), None)

            asyncio.create_task(_bg_execute())
            return self._job_public_view(job)

        logger.info(
            f"🧪 Testing model '{model_id}' (job {job['job_id']}, stage={stage}, "
            f"skip_cache={skip_cache}, "
            f"custom_environment={custom_environment}, "
            f"attach_test_report={attach_test_report})."
        )

        self._update_test_job(job, state="model_download")
        # Get model package with access tracking
        package = await self.model_cache.get_model_package(
            model_id=model_id,
            stage=stage,
            allow_unpublished=True,
            skip_cache=skip_cache,
        )

        # Use context manager to track access and prevent eviction during test
        async with package:
            logger.info(f"📍 Model source for '{model_id}': {package.source}")
            test_report: Optional[dict] = None
            tested_at: Optional[float] = None
            should_run_test = True
            should_cache_report = True

            # Check for cached test results
            test_report_path = package.package_path / ".test_cache.json"
            current_versions = self._get_bioimageio_versions()

            if not skip_cache and await asyncio.to_thread(test_report_path.exists):
                try:
                    # Load cached test results
                    async with aiofiles.open(test_report_path, "r") as f:
                        content = await f.read()
                        cached_data = await asyncio.to_thread(json.loads, content)

                    # Check if model files have changed since last test
                    cached_remote_modified = cached_data["latest_remote_modified"]
                    model_unchanged = (
                        package.latest_remote_modified == cached_remote_modified
                    )

                    # Validate cached environment versions against currently installed packages.
                    cached_test_report = cached_data["test_report"]
                    cached_env = cached_test_report.get("env", [])
                    cached_env_versions: Dict[str, str] = {}

                    for row in cached_env:
                        if isinstance(row, (list, tuple)) and len(row) >= 2:
                            pkg_name = str(row[0])
                            pkg_version = str(row[1])
                            if pkg_name in current_versions:
                                cached_env_versions[pkg_name] = pkg_version

                    env_versions_match = True
                    for pkg_name, installed_version in current_versions.items():
                        cached_version = cached_env_versions.get(pkg_name)
                        if cached_version != installed_version:
                            env_versions_match = False
                            logger.info(
                                f"🔄 Cached test env mismatch for '{model_id}' on {pkg_name}: "
                                f"cached={cached_version}, installed={installed_version}. "
                                f"Re-running tests."
                            )

                    if model_unchanged and env_versions_match:
                        # Model hasn't changed, return cached results
                        logger.info(
                            f"💾 Model '{model_id}' unchanged since last test, using cached results."
                        )
                        test_report = cached_data["test_report"]
                        tested_at = test_report["tested_at"]
                        should_run_test = False
                        should_cache_report = False
                    else:
                        if model_unchanged and not env_versions_match:
                            logger.info(
                                f"🔄 Model '{model_id}' unchanged but test environment changed, re-running tests."
                            )
                        elif not model_unchanged:
                            logger.info(
                                f"🔄 Model '{model_id}' has been updated, re-running tests "
                                f"(cached: {cached_remote_modified}, current: {package.latest_remote_modified})"
                            )
                except (json.JSONDecodeError, KeyError, OSError, IOError) as e:
                    logger.warning(
                        f"⚠️ Failed to load cached test results for '{model_id}': {e}. Running fresh test."
                    )

            # Run the test unless we already accepted a cached result
            if should_run_test:
                tested_at = time.time()
                try:
                    # For ``custom_environment=True``: build every conda
                    # env the model needs on THIS EntryApp replica
                    # first — Entry is CPU-only, so a ~10-min mamba
                    # solve doesn't hold the GPU-bound RuntimeApp
                    # replica. Multiple envs (one per weight format)
                    # are built concurrently. RuntimeApp then invokes
                    # ``bioimageio.core.test_description`` which does
                    # its own env-existence check (``mamba run -n
                    # <hash> python --version``) and finds the envs
                    # already present on the shared PVC-backed HOME,
                    # so ``mamba env create`` is a no-op there and the
                    # RuntimeApp only spends time on the actual test
                    # inference. If HOME is NOT shared cross-replica
                    # for some reason, RuntimeApp falls through to
                    # its normal env-create flow — graceful degrade
                    # to pre-1.10.0 behavior with no correctness gap.
                    if custom_environment:
                        self._update_test_job(job, state="env_setup")
                        await self._prebuild_conda_envs(
                            rdf_path=package.source,
                            model_id=model_id,
                        )

                    self._update_test_job(job, state="running")
                    test_report = await self.runtime.test(
                        rdf_path=package.source,
                        custom_environment=custom_environment,
                    )
                    logger.info(f"✅ Model test completed for '{model_id}'.")
                except Exception as e:
                    error_traceback = traceback.format_exc()
                    logger.warning(
                        f"⚠️ Model test failed for '{model_id}': {str(e)}. Generating fallback report."
                    )
                    should_cache_report = False

                    # Load RDF from package for fallback report
                    try:
                        async with aiofiles.open(package.source, "r") as f:
                            rdf_content = await f.read()
                            rdf = await asyncio.to_thread(yaml.safe_load, rdf_content)
                            artifact_type = rdf.get("type")
                    except Exception as rdf_error:
                        logger.error(
                            f"⚠️ Failed to load RDF for fallback report: {rdf_error}"
                        )
                        artifact_type = None

                    # Generate fallback test report
                    #! WARNING: Ensure that bioimageio.core and bioimageio.spec versions are the same in the EntryDeployment and RuntimeDeployment environments
                    test_report = {
                        "name": "bioimageio format validation",
                        "source_name": package.source,
                        "id": model_id,
                        "type": artifact_type,
                        "format_version": current_versions.get(
                            "bioimageio.spec", "unknown"
                        ),
                        "status": "failed",
                        "details": [{"errors": [{"msg": error_traceback}]}],
                        "env": [
                            [
                                "bioimageio.core",
                                current_versions.get("bioimageio.core", "unknown"),
                                "",
                                "",
                            ],
                            [
                                "bioimageio.spec",
                                current_versions.get("bioimageio.spec", "unknown"),
                                "",
                                "",
                            ],
                        ],
                        "saved_conda_list": "",
                    }

                    logger.warning(
                        f"⚠️ Generated fallback test report for '{model_id}' due to test error."
                    )

                test_report = self._stamp_runtime_versions_in_test_env(test_report)

                # Add tested_at timestamp to the test report so the report is self-contained.
                test_report["tested_at"] = tested_at

            # Save test results only for fresh, successful calculations.
            if should_cache_report:
                try:
                    cache_data = {
                        "test_report": test_report,
                        "latest_remote_modified": package.latest_remote_modified,
                        "custom_environment": custom_environment,
                    }
                    async with aiofiles.open(test_report_path, "w") as f:
                        await f.write(json.dumps(cache_data, indent=2))
                    logger.info(f"💾 Test report cached for model '{model_id}'")
                except (OSError, IOError) as e:
                    logger.warning(
                        f"⚠️ Failed to cache test report for '{model_id}': {e}"
                    )

            # Attach test report to artifact under the CALLER'S token —
            # not the runner's workspace credentials. The runner opens
            # a short-lived hypha client scoped to the caller and routes
            # every read + write in the attach block through it, so
            # every artifact edit is authorised as and attributed to
            # the actual user. Presence of ``hypha_token`` was already
            # validated at the top of ``test()``.
            if attach_test_report:
                artifact_id = f"bioimage-io/{model_id}"
                report_file_name = "test_report.json"
                should_attach_report = True

                caller_client = await connect_to_server(
                    {
                        "server_url": self.server_url,
                        "token": hypha_token,
                    }
                )
                try:
                    caller_am = await caller_client.get_service(
                        "public/artifact-manager"
                    )

                    try:
                        download_url = await caller_am.get_file(
                            artifact_id=artifact_id,
                            file_path=report_file_name,
                        )
                        async with httpx.AsyncClient(timeout=30) as client:
                            response = await client.get(download_url)
                            response.raise_for_status()

                        remote_test_report = await asyncio.to_thread(
                            json.loads, response.text
                        )
                        remote_tested_at = float(
                            remote_test_report.get("tested_at", 0.0)
                        )
                        local_tested_at = test_report["tested_at"]
                        should_attach_report = remote_tested_at != local_tested_at
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Failed to load remote test report for '{artifact_id}' before attaching: {e}"
                        )
                        should_attach_report = True

                    if not should_attach_report:
                        logger.info(
                            f"ℹ️ Existing test report for '{artifact_id}' is up to date; skipping attach."
                        )
                        return test_report

                    # Check current staging state. ``read()`` without
                    # ``stage=True`` returns the committed manifest and
                    # reports ``staging=None`` even when a stage exists,
                    # so we must explicitly read the staged view —
                    # that's the only surface that exposes an open
                    # stage. This was the latent bug: the original code
                    # called ``read()`` and trusted
                    # ``artifact.get("staging")``, which was always
                    # None, so any downstream "was staged?" branching
                    # was dead code and the commit path ran
                    # unconditionally.
                    stage_view = await caller_am.read(artifact_id, stage=True)
                    was_already_staged = bool(stage_view.get("staging"))

                    # Base the ``updated_manifest`` on the STAGE's
                    # manifest when one exists, so any pending edits (a
                    # reviewer's ``status`` change, RDF tweaks, doc
                    # updates) are preserved. Reading the committed
                    # manifest and calling
                    # ``edit(stage=True, manifest=...)`` would silently
                    # overwrite those changes — the bioimage.io #0006
                    # amusing-angelfish incident showed a
                    # ``status=deletion-requested`` stage being
                    # flattened back to ``status=published`` by exactly
                    # this path.
                    if was_already_staged:
                        artifact = stage_view
                    else:
                        artifact = await caller_am.read(artifact_id)

                    # Compact ``test_summary`` for the manifest (drops
                    # details + env to keep the manifest small); the
                    # full report is uploaded as ``test_report.json``.
                    test_report_summary = {
                        "status": test_report["status"],
                        "tested_at": test_report["tested_at"],
                        "env": test_report["env"],
                    }

                    # 'test_reports' and 'test_report' are legacy manifest keys.
                    updated_manifest = dict(artifact["manifest"])
                    updated_manifest.pop("test_reports", None)
                    updated_manifest.pop("test_report", None)
                    updated_manifest.pop("score", None)
                    updated_manifest["test_summary"] = test_report_summary

                    # Edit the artifact and stage the test-report
                    # additions. When a stage was already open,
                    # ``edit(stage=True)`` layers our changes onto that
                    # stage; when there was no stage, this opens a
                    # fresh one just for the report.
                    artifact = await caller_am.edit(
                        artifact_id=artifact.id,
                        manifest=updated_manifest,
                        stage=True,
                    )

                    upload_url = await caller_am.put_file(
                        artifact.id, file_path=report_file_name
                    )

                    async with httpx.AsyncClient(timeout=30) as client:
                        response = await client.put(
                            upload_url, data=json.dumps(test_report)
                        )
                        response.raise_for_status()

                    # 'test_reports.json' is a legacy file name
                    try:
                        existing_files = await caller_am.list_files(artifact.id)
                        if any(
                            file.name == "test_reports.json" for file in existing_files
                        ):
                            await caller_am.remove_file(
                                artifact.id, file_path="test_reports.json"
                            )
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Failed to remove legacy test report file for '{artifact_id}': {e}"
                        )

                    # Commit only if we opened the stage ourselves. If
                    # the artifact was already staged when the test
                    # call arrived — whoever put it there, whatever it
                    # contains, whatever ``manifest.status`` says —
                    # committing here would publish someone else's
                    # pending edits alongside the test report (the
                    # bioimage.io #0006 amusing-angelfish incident: a
                    # ``deletion-requested`` stage got published to v0
                    # when the user ran a test with
                    # ``attach_test_report=True`` because the runner
                    # unconditionally committed). Leave the stage open
                    # for the artifact owner / reviewer to commit
                    # atomically alongside their own changes.
                    if not was_already_staged:
                        await caller_am.commit(artifact_id=artifact.id)
                        logger.info(
                            f"📤 Attached test report for model '{model_id}' to "
                            f"artifact '{artifact_id}' (no prior stage — committed)."
                        )
                    else:
                        logger.info(
                            f"📋 Added test report for model '{model_id}' to the "
                            f"existing stage on '{artifact_id}' — leaving the "
                            f"stage open for the artifact owner / reviewer to "
                            f"commit alongside their own changes."
                        )
                finally:
                    try:
                        await caller_client.disconnect()
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Failed to close caller Hypha client after attach: {e}"
                        )

        # Terminal state for the sync-mode job. Async-mode wrapper
        # sees the same result via the return value and does its own
        # update (with completed_at set for the "wall clock stops on
        # background completion, not on return-to-caller" semantic).
        self._update_test_job(job, state="completed", result=test_report)
        return test_report

    @bioengine.method
    async def get_test_status(
        self,
        job_id: str = Field(
            ...,
            description="Opaque job identifier returned by ``test(..., async_mode=True)``.",
        ),
    ) -> Dict[str, Union[str, bool, int, float, list, dict, None]]:
        """Return the current state of an async test job.

        Response shape:
        - ``job_id`` — echoes the input.
        - ``model_id`` — the model this job is testing.
        - ``custom_environment`` — whether the test runs in the
          model's declared conda env (True) or the shared runtime
          (False).
        - ``state`` — one of ``queued``, ``model_download``,
          ``env_setup``, ``running``, ``completed``, ``failed``.
        - ``queue_position`` — for custom-env jobs still in
          ``queued`` / ``env_setup``, the count of custom-env jobs
          ahead of this one in start-time order. ``0`` otherwise.
        - ``started_at`` / ``updated_at`` / ``completed_at`` —
          Unix timestamps; ``completed_at`` is ``None`` until the
          job finishes.
        - ``elapsed_seconds`` — wall clock from ``started_at`` to
          either ``completed_at`` (terminal) or ``now`` (in-flight).
        - ``result`` — the full ``test_report`` dict on
          ``state=completed``, ``None`` otherwise.
        - ``error`` — human-readable message on ``state=failed``,
          ``None`` otherwise.

        Jobs are held for 24 hours after completion, then dropped.
        Registries are per-Entry replica and in-memory — jobs
        started on one replica are unknown to others, and replica
        restarts drop everything. Callers who need durability
        should use ``test(async_mode=False)`` and hold the RPC.
        """
        job = self._test_jobs.get(job_id)
        if job is None:
            raise KeyError(
                f"Unknown test job_id {job_id!r}. Jobs live in-memory per "
                f"Entry replica and expire 24 hours after completion. "
                f"Fresh call: retry via test(..., async_mode=True), or "
                f"fall back to test(..., async_mode=False) to block."
            )
        return self._job_public_view(job)

    @bioengine.method
    async def get_upload_url(
        self,
        file_type: SUPPORTED_FILES_TYPES = Field(
            ...,
            description='File type for the upload. Supported types: ".npy" (NumPy array), ".png" (PNG image), ".tiff"/".tif" (TIFF image), ",jpeg"/".jpg" (JPEG image)',
        ),
    ) -> Dict[str, str]:
        """
        Request a presigned upload URL for uploading an input image to temporary storage.

        Creates a unique temporary file in BioEngine S3 storage with a 1-hour TTL.
        Upload the file to the returned URL via an HTTP PUT request, then pass the
        returned ``file_path`` as the ``inputs`` parameter of the ``infer`` endpoint.

        Returns:
            Dictionary containing:
            - upload_url: Presigned URL for uploading the file via HTTP PUT
            - file_path: Unique temporary file path to reference the uploaded file

        Example::
            import httpx, imageio.v3 as iio, io

            result = await model_runner_service.get_upload_url(file_type=".png")
            buf = io.BytesIO()
            iio.imwrite(buf, image, extension=".png")
            async with httpx.AsyncClient() as client:
                await client.put(result["upload_url"], content=buf.getvalue())
            output = await model_runner_service.infer(model_id="...", inputs=result["file_path"])
        """
        unique_id = str(uuid.uuid4())
        file_path = f"temp/{unique_id}{file_type}"

        logger.info(f"📤 Requesting presigned upload URL for '{file_path}'...")

        try:
            upload_url = await self.s3_controller.put_file(
                file_path=file_path,
                ttl=3600,  # 1-hour TTL
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get upload URL for temporary file '{file_path}': {e}"
            ) from e

        logger.info(f"✅ Presigned upload URL generated for '{file_path}'.")
        return {"upload_url": upload_url, "file_path": file_path}

    @bioengine.method
    async def infer(
        self,
        model_id: str = Field(
            ..., description="Unique identifier of the published bioimage.io model"
        ),
        inputs: Union[np.ndarray, Dict[str, Union[np.ndarray, str]], str] = Field(
            ...,
            description="Input data as numpy array, dictionary mapping input names to arrays/strings, or a single string. "
            "Accepted string formats: a direct HTTP/HTTPS URL (fetched as-is) or a temporary file path returned by "
            "``get_upload_url`` (resolved via S3 storage). "
            "Must match the model's input specification for shape and data type. "
            "For single-input models, provide a np.ndarray or a string. "
            "For multi-input models, provide a dict with input names as keys; each value may be a np.ndarray or a string.",
        ),
        weights_format: Optional[str] = Field(
            None,
            description='Preferred model weights format ("pytorch_state_dict", "torchscript", "onnx", "tensorflow_saved_model"). If None, automatically selects best available.',
        ),
        device: Optional[Literal["cuda", "cpu"]] = Field(
            None,
            description='Target computation device. "cuda" for GPU acceleration, "cpu" for CPU-only. If None, automatically selects based on availability and model compatibility.',
        ),
        default_blocksize_parameter: Optional[int] = Field(
            None,
            description="Override default tiling block size for memory management. Larger values use more memory but may be faster. Only applicable for models supporting tiled inference.",
        ),
        sample_id: Optional[str] = Field(
            "sample",
            description="Identifier for this inference request, used for logging and debugging",
        ),
        skip_cache: Optional[bool] = Field(
            False, description="Force re-download of model package before inference"
        ),
        return_download_url: Optional[bool] = Field(
            False,
            description="If True, each array in the output will be saved to a temporary .npy file in S3 and the output value will be a presigned download URL (str) instead of the raw np.ndarray. The URL is valid for 1 hour.",
        ),
    ) -> Dict[str, Union[np.ndarray, str]]:
        """
        Execute inference on a bioimage.io model with provided input data.

        Performs end-to-end inference including:
        - Automatic input preprocessing according to model specification
        - Model execution with optimized framework backend
        - Output postprocessing and format standardization
        - Memory-efficient processing for large inputs using tiling if supported

        Returns:
            Dictionary mapping output names to inference results. By default each value is a
            ``np.ndarray`` whose shape and data type match the model's output specification
            (e.g. ``{"output": result_array}``). When ``return_download_url=True``, each value
            is instead a presigned S3 download URL (``str``) pointing to the result serialised
            as a ``.npy`` file; the URL is valid for 1 hour.

        Raises:
            ValueError: If model_id is a URL (only model IDs allowed) or inputs don't match specification
            FileNotFoundError: If a URL or temporary file path is provided but the resource does not exist or has expired
            RuntimeError: If model loading, preprocessing, inference, or postprocessing fails

        Note:
            Only published models from the bioimage.io model zoo are supported for inference.
            This method delegates to the model_inference deployment for optimized execution.
            String inputs are resolved via ``_load_image_from_source``: direct HTTP/HTTPS URLs are
            fetched as-is; all other strings are treated as temporary S3 file paths and resolved
            through BioEngine S3 storage. To upload large inputs, first call ``get_upload_url``
            to obtain a presigned URL, upload the file, then pass the returned ``file_path`` as ``inputs``.
        """
        from ray.exceptions import RayTaskError

        await self._check_runtime_available()
        logger.info(f"🤖 Running inference for model '{model_id}'...")

        # Resolve any URL or temporary file path strings to numpy arrays
        if isinstance(inputs, str):
            inputs = await self._load_image_from_source(inputs)
        elif isinstance(inputs, dict):
            resolved: Dict[str, np.ndarray] = {}
            for key, value in inputs.items():
                if isinstance(value, str):
                    array = await self._load_image_from_source(value)
                    resolved[key] = array
                else:
                    resolved[key] = value
            inputs = resolved

        try:
            # Get model package with access tracking
            package = await self.model_cache.get_model_package(
                model_id=model_id,
                stage=False,
                allow_unpublished=False,
                skip_cache=skip_cache,
            )

            # Use context manager to track access and prevent eviction during inference
            async with package:
                logger.info(
                    f"📍 Model source for '{model_id}': {package.source} "
                    f"(latest_remote_modified: {package.latest_remote_modified})"
                )

                result = await self.runtime.predict(
                    rdf_path=package.source,
                    inputs=inputs,
                    weights_format=weights_format,
                    device=device,
                    default_blocksize_parameter=default_blocksize_parameter,
                    sample_id=sample_id,
                    latest_remote_modified=package.latest_remote_modified,
                )
        except RayTaskError as e:
            error_msg = f"Failed to run inference for model '{model_id}': {e}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

        if return_download_url:
            new_result = {}
            for key, value in result.items():
                new_result[key] = await self._save_array_to_temp_file(value)
            result = new_result

        logger.info(f"✅ Inference completed for model '{model_id}'.")
        return result


