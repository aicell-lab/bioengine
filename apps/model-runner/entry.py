"""@bioengine.app EntryDeployment — the public RPC surface of the model-runner.

The app provides bioimage.io model search, RDF/documentation retrieval,
RDF validation, end-to-end model testing, and inference. It delegates the
GPU-bound work (``predict`` and the heavy ``bioimageio.core.test_model``
call) to :class:`runtime.RuntimeDeployment` via the v0.6 type-hint composition.

The GPU runtime ships Cellpose-4 (Cellpose-SAM / Cellpose-DINO) and runs
those models natively. It cannot run the older Cellpose-3 zoo models;
deploy the ``bioimage-io/cellpose3-runner`` app for those and call its
``list_supported_models()`` for the accepted ids.

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
from typing import Dict, List, Literal, Optional, Set, Union

import bioengine
import httpx
import numpy as np
import yaml
from hypha_rpc import connect_to_server
from pydantic import Field

from bioengine import __version__

from model_cache import BioimageioPackage, ModelCache
from runtime import SINGLE_INPUT_KEY, RuntimeDeployment, pin_bioimageio_conda_env


logger = logging.getLogger("ray.serve")
logger.setLevel("INFO")

SUPPORTED_FILES_TYPES = Literal[".npy", ".png", ".tiff", ".tif", ".jpeg", ".jpg"]

# Cellpose-3 zoo models. The runtime ships Cellpose 4, whose API these
# models' architectures do not survive, so ``infer`` refuses them and points
# at cellpose3-runner. Mirrors that app's ``SUPPORTED_MODELS``; ``test`` still
# accepts them because it runs in each model's own conda environment.
CELLPOSE3_MODELS = (
    "famous-fish",
    "happy-elephant",
    "merry-gorilla",
    "philosophical-panda",
    "thoughtful-chipmunk",
)


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
class EntryDeployment:
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

    # Test reports are published to per-model child artifacts under this
    # collection. Writes require the app token to be scoped to this
    # workspace; otherwise reports are cached locally but not published.
    _TEST_REPORTS_WORKSPACE = "bioimage-io"
    _TEST_REPORTS_COLLECTION = "bioimage-io/test-reports"

    # Test outcome mapped to a numeric quality score, surfaced on the
    # published test-report artifact's manifest for ranking.

    # Score at which a published test report says inference passed:
    # 1 (valid format) + 2 (default-env inference). See
    # ``_compute_report_score``.
    _INFERENCE_PASS_SCORE = 3.0

    # Async infer-request registry TTL — how long a completed/failed
    # infer job stays in memory after completion before being swept.
    # Also bounds how long the on-disk request dir survives if the
    # caller never polls after the runtime completes.
    _INFER_JOBS_TTL_SEC = 3600

    # Per-request scratch on the app's shared PVC-backed HOME. EntryDeployment
    # writes ``input/<key>.npy`` here on ``infer()`` receipt so large
    # images don't sit in RAM through the queue+download wait; RuntimeDeployment
    # reads inputs from the same directory, deletes them, writes outputs
    # to ``output/<key>.npy``, and records state timestamps in
    # ``state.json``. Kept in sync with ``RuntimeDeployment._INFERENCE_DIR_NAME``.
    _INFERENCE_DIR_NAME = ".model-runner-inference"

    def __init__(
        self,
        runtime: RuntimeDeployment,
        cache_size_in_gb: float = 50.0,
    ) -> None:
        self.runtime = runtime

        # Set Hypha server and workspace
        self.server_url = "https://hypha.aicell.io"
        # No token is allowed: the client then connects anonymously and the
        # app runs in public read-only mode (search / RDF / validate / test /
        # infer on public models via inline ndarray or public-URL inputs). The
        # anonymous workspace is read-only, so ``get_upload_url`` and test-report
        # publishing are unavailable — see ``_async_init`` and ``get_upload_url``.
        self._hypha_token = os.getenv("HYPHA_TOKEN") or None

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

        # Refcount of conda envs currently needed by a live test (env name →
        # number of in-flight tests using it). Env-build (this replica,
        # ``_env_build_lock``) is decoupled from the GPU run (RuntimeDeployment
        # ``_gpu_lock``), so a second test's prebuild could otherwise evict an
        # env the first test is still running against. Eviction + the weekly
        # sweep read this under ``_env_build_lock`` and never remove an env
        # with a positive count.
        self._env_inuse: Dict[str, int] = {}

        # Async test-run registry. Every ``test()`` call schedules a
        # background job keyed by an opaque run id (``tj-<hex12>``);
        # values track state, step timestamps and (on completion) the
        # test report. Purely in-memory per replica; runs from a killed
        # replica just disappear and ``get_test_status`` raises
        # KeyError for them.
        self._test_jobs: Dict[str, dict] = {}
        self._test_jobs_ttl_sec: int = 24 * 3600

        # Async infer-request registry — same shape as ``_test_jobs``.
        # Keyed by opaque request id (``ij-<hex12>``); each value
        # tracks state, step timestamps and (once completed) the
        # in-memory result cached after its first successful poll. Also
        # per-replica in-memory; a restart drops everything and orphan
        # on-disk request dirs are swept in ``_async_init``.
        self._infer_jobs: Dict[str, dict] = {}
        # Filled in ``_async_init`` — HOME isn't guaranteed to be the
        # final PVC path until then.
        self._inference_dir: Optional[Path] = None
        # Per-report-artifact locks serialize concurrent uploads to the
        # same ``test-report-<model>`` artifact (a published and a staged
        # test of one model would otherwise race on its single staging area).
        self._report_locks: Dict[str, asyncio.Lock] = {}
        # Both set in ``_async_init`` once the connected identity is known.
        self._test_reports_writable: bool = False
        self._is_anonymous: bool = False

        # Replica identifier for logging.
        try:
            from ray import serve as _serve
            _ctx = _serve.get_replica_context()
            self.replica_id = _ctx.replica_tag
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
        connect_config = {"server_url": self.server_url}
        if self._hypha_token:
            connect_config["token"] = self._hypha_token
        self.hypha_client = await connect_to_server(connect_config)
        self.artifact_manager = await self.hypha_client.get_service(
            "public/artifact-manager"
        )
        self.s3_controller = await self.hypha_client.get_service("public/s3-storage")
        logger.info(f"Connected to Hypha Server at {self.server_url}")

        # Per-request scratch directory on the shared PVC. On a fresh
        # replica life the in-memory registry is empty, so any lingering
        # request dirs belong to a previous replica and are safe to
        # reap (their in-memory job records went away with the replica).
        self._inference_dir = Path(os.environ["HOME"]) / self._INFERENCE_DIR_NAME
        self._inference_dir.mkdir(parents=True, exist_ok=True)
        try:
            leftovers = await asyncio.to_thread(
                lambda: [p for p in self._inference_dir.iterdir() if p.is_dir()]
            )
        except FileNotFoundError:
            leftovers = []
        for leftover in leftovers:
            await asyncio.to_thread(
                shutil.rmtree, str(leftover), ignore_errors=True
            )
        if leftovers:
            logger.info(
                f"🧹 Cleaned {len(leftovers)} stale inference request dir(s) "
                f"from a previous replica."
            )

        # Reports are published under the app's own token. Publishing
        # requires that token to carry at least read-write scope on the
        # test-reports workspace (``rw`` or ``a``); ``r`` or absent means
        # no publish. The startup app gets the bioimage-io workspace token
        # injected; manual deploys must pass it via ``--hypha-token``.
        # Otherwise reports are cached + returned only.
        user = dict(self.hypha_client.config.user or {})
        self._is_anonymous = bool(user.get("is_anonymous"))
        scope = dict(user.get("scope") or {})
        workspace_perms = dict(scope.get("workspaces") or {})
        token_permission = workspace_perms.get(self._TEST_REPORTS_WORKSPACE)
        self._test_reports_writable = token_permission in ("rw", "a")
        if self._is_anonymous:
            logger.warning(
                "⚠️ No HYPHA_TOKEN — connected anonymously in PUBLIC READ-ONLY "
                "mode. Search / RDF / validate / test / infer on public models "
                "work with inline ndarray or public-URL inputs, but get_upload_url "
                "(S3 upload) and test-report publishing are unavailable. Redeploy "
                "with --hypha-token for full functionality."
            )
        elif self._test_reports_writable:
            logger.info(
                f"📤 Test reports will be published to '{self._TEST_REPORTS_COLLECTION}'."
            )
        else:
            logger.warning(
                f"⚠️ App token lacks read-write scope on workspace "
                f"'{self._TEST_REPORTS_WORKSPACE}' (permission={token_permission!r}) — "
                f"test reports will be cached and returned but NOT published to "
                f"'{self._TEST_REPORTS_COLLECTION}'."
            )

    # === Ray Serve Health Check Method - will be called periodically to check the health of the deployment ===

    @bioengine.health_check
    async def _health_check(self) -> None:
        # Hypha connectivity: a failure here means we cannot serve at all.
        await self.hypha_client.echo("ping")

    # === Internal Helper Methods ===

    def _get_bioimageio_versions(self) -> Dict[str, str]:
        """
        Return the identities the test cache invalidates on: the installed
        ``bioimageio.core`` and ``bioimageio.spec`` versions.

        The bioengine version and the model-runner artifact version are
        deliberately NOT invalidation keys — a framework or runner upgrade
        should reuse an otherwise-current report. Both are still recorded in
        ``test_report['env']`` (see ``_stamp_runtime_versions_in_test_env``).
        """
        from importlib.metadata import PackageNotFoundError, version

        package_names = ["bioimageio.core", "bioimageio.spec"]
        versions: Dict[str, str] = {}

        for package_name in package_names:
            try:
                versions[package_name] = version(package_name)
            except PackageNotFoundError:
                versions[package_name] = "not-installed"

        logger.debug(f"📦 Bioimage.io package versions: {versions}")
        return versions

    def _report_is_current(
        self,
        report: dict,
        package,
        current_versions: Dict[str, str],
        custom_environment: bool,
    ) -> bool:
        """Whether ``report`` can be reused instead of re-running the test.

        A report is current when it ran in the requested environment mode,
        the model files have not changed since it was produced (same
        ``latest_remote_modified``) and every tracked runtime package matches
        the installed version. Same criteria used for the on-disk
        ``.test_cache.json`` and the collection fallback.

        ``custom_environment`` must be the *effective* mode (after the
        no-``environment.yaml`` downgrade in ``_execute_test``), not the raw
        request — otherwise a downgraded custom request would never match its
        own standard report. Reports written before ``test_environment``
        existed are treated as standard.
        """
        requested = "custom" if custom_environment else "standard"
        if report.get("test_environment", "standard") != requested:
            return False
        if report.get("latest_remote_modified") != package.latest_remote_modified:
            return False
        report_versions = {
            str(row[0]): str(row[1])
            for row in report.get("env", [])
            if isinstance(row, (list, tuple)) and len(row) >= 2
            and str(row[0]) in current_versions
        }
        return all(
            report_versions.get(name) == installed
            for name, installed in current_versions.items()
        )

    def _stamp_runtime_versions_in_test_env(
        self, test_report: dict, current_versions: Dict[str, str]
    ) -> dict:
        """Upsert runtime-identity rows in ``test_report['env']``.

        Records the ``bioimageio.core``/``.spec`` the RuntimeApp
        orchestrated the test under (``current_versions``), plus the
        bioengine and model-runner versions the report was produced under.

        ``bioimageio.core``/``.spec`` are the cache-invalidation keys and
        the versions the frontend renders. Upstream stamps ``env`` from the
        orchestrating interpreter, so a standard-env run already carries
        both; but an ``as-described`` custom-env run drops the
        ``bioimageio.core`` row (the outer summary is built by
        ``build_description`` and only merges ``details`` back from the
        in-env subprocess) and reports only the orchestrator's
        ``bioimageio.spec``. Backfill both here from the installed versions
        so every report — standard or custom — is keyed and displayed
        identically. bioengine and model-runner are NOT cache-invalidation
        keys; they are stamped for downstream consumers (e.g. bioimage.io
        CI) that read them from the published report.
        """
        rows = [
            ("bioimageio.core", current_versions.get("bioimageio.core", "unknown")),
            ("bioimageio.spec", current_versions.get("bioimageio.spec", "unknown")),
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
        ``sys.executable -c`` wrapper) don't need the RuntimeDeployment's
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
        by EntryDeployment are visible to RuntimeDeployment on the same path.
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

    def _acquire_inuse_envs(self, names: List[str]) -> None:
        """Mark envs as in use for the duration of a test. Call under
        ``_env_build_lock`` so eviction/sweep (which read the counter under
        the same lock) can never race an acquire.
        """
        for name in names:
            self._env_inuse[name] = self._env_inuse.get(name, 0) + 1

    def _release_inuse_envs(self, names: List[str]) -> None:
        """Release envs once a test no longer needs them. Synchronous (never
        awaits), so it can't interleave with a locked eviction read; no lock
        needed.
        """
        for name in names:
            remaining = self._env_inuse.get(name, 0) - 1
            if remaining > 0:
                self._env_inuse[name] = remaining
            else:
                self._env_inuse.pop(name, None)

    def _inuse_env_names(self) -> set:
        """Env names with a live test using them (positive refcount)."""
        return {name for name, count in self._env_inuse.items() if count > 0}

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

        in_use = self._inuse_env_names()
        candidates = []
        for entry in envs_dir.iterdir():
            if not entry.is_dir() or entry.name in in_use:
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
        env that isn't protected. Protected = ``protected`` (the envs the
        current test needs) plus every env a concurrently running test is
        using (``_inuse_env_names``), so a running test never loses its env.

        Approximate: uses ``du -sk`` per env, not fsync-accurate, so the
        ceiling is a soft target.
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
        protected_set = set(protected) | self._inuse_env_names()
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
        job = {
            "job_id": job_id,
            "model_id": model_id,
            "custom_environment": custom_environment,
            # Step timestamps surfaced through the shared progress
            # schema. ``env_setup_ts`` only ever fires when
            # ``custom_environment=True``; the other two fire for every
            # run.
            "model_download_ts": None,
            "env_setup_ts": None,
            "running_ts": None,
            # Execution-start timestamps (position #0 reached), distinct
            # from the ``*_ts`` queue-entry marks above. ``env_started_ts``
            # is stamped when the env-build lock is acquired;
            # ``run_started_ts`` when the GPU run actually starts.
            "env_started_ts": None,
            "run_started_ts": None,
            "state": "queued",
            "started_at": now,
            "completed_at": None,
            # For a test run this holds the full test report on
            # success, ``{"error": str}`` on failure, or None while
            # still running.
            "result": None,
            # Background asyncio.Task handle, set right after the job is
            # created so ``cancel_request`` can cancel it. Never surfaced
            # in the progress dict.
            "_task": None,
        }
        self._test_jobs[job_id] = job
        return job

    def _update_test_job(
        self,
        job: dict,
        *,
        state: Optional[str] = None,
        result: Optional[dict] = None,
    ) -> None:
        """Idempotent state update. Stamps the matching step timestamp
        when transitioning into ``model_download`` / ``env_setup`` /
        ``running`` and records ``completed_at`` on terminal states.
        """
        if state is not None:
            job["state"] = state
            now = time.time()
            if state == "model_download" and job["model_download_ts"] is None:
                job["model_download_ts"] = now
            elif state == "env_setup" and job["env_setup_ts"] is None:
                job["env_setup_ts"] = now
            elif state == "running" and job["running_ts"] is None:
                job["running_ts"] = now
            if state in ("completed", "failed", "cancelled"):
                job["completed_at"] = now
        if result is not None:
            job["result"] = result

    def _env_queue_position(self, job: dict) -> Optional[int]:
        """0-based position in the env-build queue, or None when the job
        is not currently in the ``env_setup`` stage.

        The single ``_env_build_lock`` serializes conda-env prebuild across
        test runs, so this is a real FIFO queue: 0 = building now, N = N
        env-setup jobs ahead of this one. Rank by the BUILD signal
        (``env_started_ts``, stamped when the lock is acquired), NOT raw
        entry order (``env_setup_ts``): the asyncio lock isn't always
        granted in entry order, so the job actually building must be the
        one reported at position 0 — keeping the position consistent with
        the ``env_setup`` start timestamp (start set iff position 0).
        Infer never enters ``env_setup``, so this is test-only.
        """
        if job["state"] != "env_setup":
            return None
        # The job holding the build lock (env_started_ts set) is building
        # now → position 0. A still-waiting job counts the current builder
        # plus any earlier-queued waiter ahead of it.
        if job["env_started_ts"] is not None:
            return 0
        ts = job["env_setup_ts"]
        ahead = 0
        for other in self._test_jobs.values():
            if other["state"] != "env_setup":
                continue
            if other["env_started_ts"] is not None:
                ahead += 1  # the current builder
            elif (
                other["env_setup_ts"] is not None
                and ts is not None
                and other["env_setup_ts"] < ts
            ):
                ahead += 1  # an earlier-queued waiter
        return ahead

    def _run_queue_position(self, job: dict) -> Optional[int]:
        """0-based position in the combined test+infer GPU queue, or None
        when the job is not currently in the ``running`` stage.

        Position 0 means the job is actually executing on a GPU replica,
        identified by a live execution start (``run_started_ts`` — infer
        reads ``runtime_started_at`` from state.json, test lazy-stamps at
        the front of the queue). Test and infer share one ``_gpu_lock``
        per replica and the runtime autoscales to 2 replicas, so up to 2
        jobs execute at once and BOTH report 0 — that's why position is
        keyed on "is it running", not on dispatch rank. A job still waiting
        for a lock counts every executing job plus every earlier-dispatched
        waiter ahead of it (ordered by ``running_ts``), so the first waiter
        behind two busy replicas is #2. This keeps the invariant that a
        start timestamp exists iff position == 0.
        """
        if job["state"] != "running":
            return None
        if job["run_started_ts"] is not None:
            return 0
        ts = job["running_ts"]
        ahead = 0
        for registry in (self._test_jobs, self._infer_jobs):
            for other in registry.values():
                if other["state"] != "running":
                    continue
                if other["run_started_ts"] is not None:
                    ahead += 1  # actively executing on a replica, ahead of us
                elif (
                    other["running_ts"] is not None
                    and ts is not None
                    and other["running_ts"] < ts
                ):
                    ahead += 1  # dispatched before us, also waiting for a lock
        return ahead

    def _stage_timeline(self, job: dict) -> dict:
        """Per-stage ``{start, end[, queue_position]}`` timeline, shared by
        the test and infer progress dicts.

        Each stage's ``start`` is the *execution* start — the moment it
        reaches queue position #0 and actually begins (env-build lock
        acquired / GPU run started), not when it joined the queue. So
        ``start`` is None while a stage is still queued (position > 0), and
        set once it runs. A stage's ``end`` is the next stage's queue-entry
        mark, falling back to ``completed_at`` once terminal.
        ``model_download`` carries no ``queue_position`` — distinct models
        download concurrently, so it is never queued.
        """
        download_ts = job["model_download_ts"]
        env_entry_ts = job["env_setup_ts"]
        run_entry_ts = job["running_ts"]
        done_ts = job["completed_at"]
        run_start = job["run_started_ts"]
        if run_start is None and job["state"] in ("completed", "failed", "cancelled"):
            # A run that finished before any poll caught it at position #0
            # never got a live execution stamp; fall back to its GPU-queue
            # entry mark so a terminal stage still reports a start.
            run_start = run_entry_ts
        return {
            "model_download": {
                "start": download_ts,
                "end": env_entry_ts or run_entry_ts or done_ts,
            },
            "env_setup": {
                "start": job["env_started_ts"],
                "end": run_entry_ts or done_ts,
                "queue_position": self._env_queue_position(job),
            },
            "run": {
                "start": run_start,
                "end": done_ts,
                "queue_position": self._run_queue_position(job),
            },
        }

    def _job_progress(self, job: dict) -> dict:
        """Return the progress dict for a test run.

        Schema is shared with ``_infer_job_progress`` so both endpoints
        speak the same shape — a monotonic timeline bracketed by
        ``submitted_at`` / ``completed_at``:

        * ``state`` — current job state (``queued``, ``model_download``,
          ``env_setup``, ``running``, ``completed``, ``failed``,
          ``cancelled``).
        * ``submitted_at`` — unix ts when the run was accepted (queued).
        * ``completed_at`` — unix ts when the run finished (result ready
          or failed), recorded server-side at completion so the elapsed
          time is accurate regardless of poll cadence; None until then.
        * ``result`` — the test report on success, ``{"error": str}``
          on failure, else None.
        * ``stages`` — per-stage ``{start, end[, queue_position]}`` map for
          model_download / env_setup / run. See ``_stage_timeline``.
        """
        # Lazy-stamp the run execution-start the first time this job is
        # observed at GPU-queue position #0. The test path has no runtime
        # state.json (infer reads ``runtime_started_at`` instead), so the
        # entry records the moment it sees the run reach the front. With 2
        # replicas this can only see the front runner: a second test running
        # concurrently on the other replica stays at #1 (no start) until the
        # first frees — invariant-preserving (no start above #0) but it
        # under-reports the rare co-running second test. Infer has a real
        # per-replica signal (state.json) and reports both at #0.
        if (
            job["run_started_ts"] is None
            and job["state"] == "running"
            and self._run_queue_position(job) == 0
        ):
            job["run_started_ts"] = time.time()
        return {
            "state": job["state"],
            "submitted_at": job["started_at"],
            "completed_at": job["completed_at"],
            "result": job["result"],
            "stages": self._stage_timeline(job),
        }

    # === Async infer-job registry ===

    def _sweep_expired_infer_jobs(self) -> None:
        """Drop finished infer jobs older than ``_INFER_JOBS_TTL_SEC``
        AND clean up any lingering on-disk request dir. Runs
        opportunistically on each new job creation.
        """
        now = time.time()
        expired = [
            rid
            for rid, j in self._infer_jobs.items()
            if j.get("completed_at") is not None
            and (now - j["completed_at"]) > self._INFER_JOBS_TTL_SEC
        ]
        for rid in expired:
            self._infer_jobs.pop(rid, None)
            if self._inference_dir is not None:
                request_dir = self._inference_dir / rid
                try:
                    shutil.rmtree(request_dir, ignore_errors=True)
                except OSError:
                    pass

    def _new_infer_job(
        self, model_id: str, return_download_url: bool
    ) -> dict:
        """Create an infer-job record and return it. Sweeps expired
        jobs on the way in.
        """
        self._sweep_expired_infer_jobs()
        request_id = f"ij-{uuid.uuid4().hex[:12]}"
        now = time.time()
        job = {
            "job_id": request_id,
            "model_id": model_id,
            "return_download_url": return_download_url,
            # Step timestamps that surface in the progress dict.
            # ``env_setup`` is never populated for infer (no per-model
            # env prebuild on the infer path) but is kept in the schema
            # for symmetry with test.
            "model_download_ts": None,
            "env_setup_ts": None,
            "running_ts": None,
            # Execution-start timestamps (position #0 reached).
            # ``env_started_ts`` is never set on the infer path (no env
            # prebuild); ``run_started_ts`` is filled from the runtime's
            # ``runtime_started_at`` marker.
            "env_started_ts": None,
            "run_started_ts": None,
            "state": "queued",
            "started_at": now,
            "completed_at": None,
            # Holds the inference result dict on success or
            # ``{"error": str}`` on failure. Materialized from disk on
            # the FIRST successful poll after completion; subsequent
            # polls read this cached value until the job TTL sweeps it.
            "result": None,
            # Only meaningful once ``state == "completed"``: True once
            # the first poll has read the outputs off disk and deleted
            # the request dir. Guards against re-reading a deleted dir.
            "_result_materialized": False,
            # Background asyncio.Task handle for ``cancel_request``. Never
            # surfaced in the progress dict.
            "_task": None,
        }
        self._infer_jobs[request_id] = job
        return job

    def _update_infer_job(
        self,
        job: dict,
        *,
        state: Optional[str] = None,
        result: Optional[dict] = None,
    ) -> None:
        """Idempotent state update for infer jobs — same semantics as
        ``_update_test_job`` (auto-stamp step timestamps, set
        ``completed_at`` on terminal states).
        """
        if state is not None:
            job["state"] = state
            now = time.time()
            if state == "model_download" and job["model_download_ts"] is None:
                job["model_download_ts"] = now
            elif state == "running" and job["running_ts"] is None:
                job["running_ts"] = now
            if state in ("completed", "failed", "cancelled"):
                job["completed_at"] = now
        if result is not None:
            job["result"] = result

    def _read_runtime_state_file(self, request_id: str) -> Optional[dict]:
        """Read ``<request_id>/state.json`` and return the parsed dict,
        or None if the file doesn't exist yet.

        Runtime writes ``state.json`` twice: once with just
        ``runtime_started_at`` right after it acquires ``_gpu_lock``,
        then again with both start and completion timestamps after the
        outputs are written. Entry uses this to refine the reported
        ``running`` timestamp to the moment GPU work actually started.
        """
        if self._inference_dir is None:
            return None
        state_file = self._inference_dir / request_id / "state.json"
        try:
            content = state_file.read_text()
        except (FileNotFoundError, IsADirectoryError):
            return None
        except OSError as e:
            logger.debug(f"Could not read state.json for {request_id}: {e}")
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Runtime writes atomically via rename, so a torn read is
            # only possible between .tmp creation and rename. Treat as
            # "not ready yet" — next poll will succeed.
            return None

    def _infer_job_progress(self, job: dict) -> dict:
        """Return the progress dict for an infer request.

        Same schema as ``_job_progress`` (test): ``state`` plus a monotonic
        timeline bracketed by ``submitted_at`` / ``completed_at`` and the
        per-stage ``stages`` map. ``env_setup`` is always None on the infer
        path since there's no per-model environment prebuild, so its
        ``stages`` entry never queues.
        """
        # Fill the run execution-start from ``state.json`` on the shared
        # PVC. Once the runtime has acquired ``_gpu_lock`` for this request
        # it writes ``runtime_started_at`` — the true "start of GPU work",
        # kept distinct from ``running_ts`` (the dispatch mark used for
        # queue ordering).
        if job["run_started_ts"] is None and job["state"] not in (
            "queued",
            "model_download",
            "completed",
            "failed",
            "cancelled",
        ):
            state_file = self._read_runtime_state_file(job["job_id"])
            if state_file and "runtime_started_at" in state_file:
                job["run_started_ts"] = float(state_file["runtime_started_at"])

        # Fallback: a fast infer can finish (and have its request dir swept)
        # before a poll catches ``runtime_started_at`` in state.json, leaving
        # a position-0 (executing) job with no start. Lazy-stamp it the first
        # time it's observed at the front of the GPU queue — mirrors the test
        # path (``_job_progress``) so a running job always carries a start and
        # a queued one (position > 0) never does.
        if (
            job["run_started_ts"] is None
            and job["state"] == "running"
            and self._run_queue_position(job) == 0
        ):
            job["run_started_ts"] = time.time()

        return {
            "state": job["state"],
            "submitted_at": job["started_at"],
            "completed_at": job["completed_at"],
            "result": job["result"],
            "stages": self._stage_timeline(job),
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
        pre-building envs on EntryDeployment is that ``test_description``
        on RuntimeDeployment finds an env with the SAME name and skips its
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

        ``pin_bioimageio_conda_env`` pins ``bioimageio.core``/``.spec`` to
        the installed versions; the RuntimeApp applies the identical pin
        (by wrapping ``get_conda_env``) before the same hash, so both sides
        still agree and the pre-built env is reused.
        """
        from io import StringIO
        import hashlib

        from bioimageio.spec import get_conda_env
        from bioimageio.spec._internal.io_utils import write_yaml

        versions = self._get_bioimageio_versions()
        conda_env = pin_bioimageio_conda_env(
            get_conda_env(entry=wf),
            versions["bioimageio.core"],
            versions["bioimageio.spec"],
        )
        conda_env.name = None
        dumped_env = conda_env.model_dump(mode="json", exclude_none=True)
        buf = StringIO()
        write_yaml(dumped_env, file=buf)
        encoded = buf.getvalue().encode()
        env_name = hashlib.sha256(encoded).hexdigest()
        return env_name, encoded

    # === Conda env pre-creation for custom_environment tests ===

    @staticmethod
    def _model_declares_custom_env(rdf_path: str) -> bool:
        """True iff any weight-format entry carries an explicit
        ``dependencies.source`` (an authored ``environment.yaml`` bundled
        with the model).

        When False, ``bioimageio.spec.get_conda_env`` builds a
        framework-default env from the framework name alone. Those
        defaults are frequently unsolvable in practice — the classic
        example is ``pytorch_state_dict`` without deps, where
        ``mkl==2024.*`` is pinned against ``pytorch<1.14`` which
        requires ``mkl<2023``. ``_execute_test`` uses this to force
        ``custom_environment=False`` silently rather than surfacing a
        confusing mamba-solve failure to the caller.
        """
        try:
            with open(rdf_path, "r") as f:
                rdf = yaml.safe_load(f)
        except (OSError, yaml.YAMLError):
            return False
        weights = (rdf or {}).get("weights") or {}
        if not isinstance(weights, dict):
            return False
        for wf in weights.values():
            if not isinstance(wf, dict):
                continue
            deps = wf.get("dependencies")
            if isinstance(deps, dict) and deps.get("source"):
                return True
        return False

    async def _prebuild_conda_envs(
        self, rdf_path: str, model_id: str, job: Optional[dict] = None
    ) -> List[str]:
        """CPU-side, non-blocking pre-creation of every conda env the
        model needs for ``custom_environment=True`` tests.

        Returns the env names this test needs, each marked in use so
        eviction/sweep won't reap them. The caller MUST pass the returned
        list to ``_release_inuse_envs`` once ``runtime.test`` returns.

        Runs entirely on this EntryDeployment replica (no GPU held). Loads
        the model description, walks the present weight formats,
        computes each weight format's ``BioimageioCondaEnv`` spec
        via ``bioimageio.spec.get_conda_env``, and hashes the dumped
        YAML with SHA256 — matching ``bioimageio.core 0.10.4``'s
        env-name algorithm exactly so the pre-created envs are the
        same names ``test_description`` looks for on the RuntimeDeployment
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
            return []

        env_vars = self._mamba_env_vars()
        needed_env_names = [name for _, name, _ in env_specs]

        # ``_env_build_lock`` serializes env-build across concurrent
        # ``test()`` calls on this Entry replica. Cache-hit path
        # (all envs already exist) still holds the lock briefly, but
        # that's OK — mamba probes take <1s and eviction/sweep are
        # cheap when there's nothing to do.
        async with self._env_build_lock:
            # Lock acquired ⇒ this job is now at env-queue position #0 and
            # the build actually begins; stamp the execution-start.
            if job is not None and job["env_started_ts"] is None:
                job["env_started_ts"] = time.time()

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

            # Mark in use before releasing the lock so a concurrent test's
            # eviction (which also holds the lock) can't reap these while the
            # GPU run below is using them. The caller MUST release via
            # ``_release_inuse_envs`` once ``runtime.test`` returns.
            self._acquire_inuse_envs(needed_env_names)

        return needed_env_names

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

    # === HuggingFace-gated weight localization (package preparation) ===

    # ``bioimageio.core`` resolves a remote weight ``source`` with a plain
    # ``httpx.get`` and no auth header, so a gated ``huggingface.co`` URL
    # (e.g. ``facebook/sam3``) returns 401 in the test/infer subprocess —
    # which is also where ``_safe_subprocess_env`` strips every ``*TOKEN*``
    # var. So we fetch such weights here instead, during package
    # preparation in the trusted EntryDeployment process, with the read-only HF
    # token, and repoint the local RDF ``source`` at the downloaded file.
    # The subprocess then loads a local file with no HF access and never
    # sees the token. Same shape as ``_prebuild_conda_envs``: EntryDeployment
    # prepares artifacts on the shared PVC-backed HOME, the RuntimeDeployment
    # subprocess consumes them.
    _HF_WEIGHTS_DIRNAME = ".hf_weights"

    @staticmethod
    def _sha256_file(path: Path) -> str:
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    async def _stream_download(
        self, url: str, dest: Path, headers: Dict[str, str]
    ) -> None:
        """Stream a (potentially multi-GB) file to ``dest`` off the loop.

        Never holds the whole body in memory — chunks are written through
        ``aiofiles`` so the event loop stays responsive during the download.
        """
        import aiofiles

        timeout = httpx.Timeout(60.0, read=300.0)
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True
        ) as client:
            async with client.stream("GET", url, headers=headers) as resp:
                resp.raise_for_status()
                async with aiofiles.open(dest, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                        await f.write(chunk)

    async def _ensure_hf_weight_file(
        self,
        url: str,
        local_path: Path,
        expected_sha: Optional[str],
        token: str,
        model_id: str,
        fmt: str,
    ) -> bool:
        """Download ``url`` to ``local_path`` (HF-authenticated) if not
        already present with the expected sha256. Returns True iff the file
        is in place and (when a sha is declared) verified.
        """
        if await asyncio.to_thread(local_path.exists):
            if not expected_sha or await asyncio.to_thread(
                self._sha256_file, local_path
            ) == expected_sha:
                logger.info(
                    f"✅ Reusing cached HF weight for '{model_id}' [{fmt}]: "
                    f"{local_path.name}"
                )
                return True
            await asyncio.to_thread(local_path.unlink, True)

        await asyncio.to_thread(
            lambda: local_path.parent.mkdir(parents=True, exist_ok=True)
        )
        tmp = local_path.with_name(local_path.name + ".partial")
        logger.info(
            f"🤗 Downloading gated HF weight for '{model_id}' [{fmt}] from {url}"
        )
        try:
            await self._stream_download(
                url, tmp, {"Authorization": f"Bearer {token}"}
            )
        except Exception as e:
            await asyncio.to_thread(tmp.unlink, True)
            logger.error(
                f"❌ Failed to download HF weight for '{model_id}' [{fmt}]: {e}"
            )
            return False

        if expected_sha:
            actual = await asyncio.to_thread(self._sha256_file, tmp)
            if actual != expected_sha:
                await asyncio.to_thread(tmp.unlink, True)
                logger.error(
                    f"❌ HF weight sha256 mismatch for '{model_id}' [{fmt}]: "
                    f"expected {expected_sha}, got {actual}"
                )
                return False

        await asyncio.to_thread(tmp.rename, local_path)
        logger.info(
            f"✅ Downloaded HF weight for '{model_id}' [{fmt}]: {local_path.name}"
        )
        return True

    async def _localize_hf_weights(self, package, model_id: str) -> None:
        """Replace gated ``huggingface.co`` weight sources in the cached RDF
        with locally-downloaded copies (see the section comment above).

        No-op for models without such a source. Idempotent — the weight file
        persists under a dot-dir the cache's file-delete step ignores, so a
        re-run only re-downloads after a real package refresh.
        """
        import aiofiles

        rdf_path = Path(package.source)
        try:
            async with aiofiles.open(rdf_path, "r") as f:
                content = await f.read()
            rdf = await asyncio.to_thread(yaml.safe_load, content)
        except (OSError, yaml.YAMLError) as e:
            logger.warning(
                f"⚠️ Could not read RDF for HF weight localization of "
                f"'{model_id}': {e}"
            )
            return

        weights = (rdf or {}).get("weights") or {}
        if not isinstance(weights, dict):
            return

        token = os.getenv("HF_READ_TOKEN")
        changed = False
        for fmt, entry in weights.items():
            if not isinstance(entry, dict):
                continue
            source = entry.get("source")
            if not isinstance(source, str) or not source.startswith(
                "https://huggingface.co/"
            ):
                continue
            basename = source.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
            if not basename:
                continue
            if not token:
                logger.warning(
                    f"⚠️ Model '{model_id}' weight '{fmt}' has gated HuggingFace "
                    f"source '{source}' but HF_READ_TOKEN is not set — leaving it "
                    f"unchanged; bioimageio will fail to download it."
                )
                continue
            local_path = package.package_path / self._HF_WEIGHTS_DIRNAME / basename
            if not await self._ensure_hf_weight_file(
                source, local_path, entry.get("sha256"), token, model_id, fmt
            ):
                continue
            entry["source"] = f"{self._HF_WEIGHTS_DIRNAME}/{basename}"
            changed = True

        if changed:
            rewritten = await asyncio.to_thread(
                yaml.safe_dump, rdf, sort_keys=False
            )
            async with aiofiles.open(rdf_path, "w") as f:
                await f.write(rewritten)
            logger.info(
                f"🤗 Localized gated HuggingFace weight source(s) for '{model_id}'."
            )

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
    async def get_status(self) -> Dict[str, Union[str, bool]]:
        """Report which capabilities this deployment actually has.

        The runner degrades gracefully by connected identity: a tokenless
        (anonymous) deploy can only read public models and run inference on
        inline/URL inputs, while full functionality (S3 upload, test-report
        publishing) needs a token with read-write scope on the bioimage-io
        workspace. Deployers verify a runner is fully functional by asserting
        ``test_reports_writable`` here. Artifact/version identity is NOT
        returned — read that from the worker's ``get_app_status`` instead.
        """
        return {
            "workspace": self.hypha_client.config.workspace,
            "is_anonymous": self._is_anonymous,
            "s3_upload_available": not self._is_anonymous,
            "test_reports_writable": self._test_reports_writable,
        }

    async def _runnable_model_ids(self) -> Set[str]:
        """Model aliases a BioEngine runner can actually serve.

        The per-model artifacts under ``test-reports`` are written by this
        runner as it tests, so they track the current runtime. They cannot
        vouch for the Cellpose models older than Cellpose 4: those need the
        incompatible Cellpose-3 runtime, so they score below the pass mark
        here however healthy they are. ``cellpose3-runner`` serves them, so
        they are added back from ``CELLPOSE3_MODELS`` — the same list
        ``infer`` uses to redirect them.
        """
        prefix = "test-report-"
        reports = await self.artifact_manager.list(
            parent_id=self._TEST_REPORTS_COLLECTION, limit=1000, stage=False
        )
        runnable = {
            report["alias"][len(prefix) :]
            for report in reports
            if report["alias"].startswith(prefix)
            and (report["manifest"].get("score") or 0) >= self._INFERENCE_PASS_SCORE
        }
        return runnable | set(CELLPOSE3_MODELS)

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

        Results span every Cellpose generation: the Cellpose-4 family
        (Cellpose-SAM, Cellpose-DINO) runs in this runner's default
        environment, while Cellpose models older than Cellpose 4 are served by
        the ``bioimage-io/cellpose3-runner`` app — query that app's
        ``list_supported_models()`` for its current accepted ids.
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
                runnable_models = await self._runnable_model_ids()

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

        # Prefer the new ``bioimageio.yaml`` name; fall back to legacy ``rdf.yaml``.
        response = None
        rdf_url = None
        for rdf_name in ("bioimageio.yaml", "rdf.yaml"):
            rdf_url = f"{self.server_url}/bioimage-io/artifacts/{model_id}/files/{rdf_name}"
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
            if response.status_code != 404:
                break

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

    @staticmethod
    def _resolve_cache(cache: str, skip_cache: Optional[bool]) -> str:
        """Resolve the effective cache policy from ``cache`` and the deprecated
        ``skip_cache`` alias. ``skip_cache`` (kept so existing callers keep
        working while ``cache`` rolls out) wins only when explicitly set:
        True→"skip", False→"check".
        """
        if skip_cache is not None:
            return "skip" if skip_cache else "check"
        return cache

    @bioengine.method
    async def test(
        self,
        model_id: str = Field(
            ..., description="Unique identifier of the bioimage.io model to test"
        ),
        stage: Optional[bool] = Field(
            False,
            description="Whether to test the staged version of the model (True) or the committed version (False). The report is published to the matching slot: staged/ or published/.",
        ),
        custom_environment: Optional[bool] = Field(
            None,
            description="If True, run the test inside the conda environment declared by the model's own weights description (``bioimageio.core`` ``runtime_env='as-described'``, backed by ``mamba`` for env creation; the env is cached on the shared PVC and LRU-evicted under a size ceiling, not removed per call). If False, run in the model-runner RuntimeDeployment's own venv — the same interpreter that serves inference. If left None (default), inherit the environment of the model's prior report for the matching slot (staged report for stage=True, published for stage=False) — so a model last tested in its custom env is re-tested in the custom env automatically — falling back to False when no prior report exists.",
        ),
        cache: Literal["skip", "check", "reuse"] = Field(
            "check",
            description="Model-cache policy (same meaning on ``infer`` and on "
            "cellpose3-runner): 'check' (default) verifies the cached package "
            "against upstream and re-downloads if stale, also re-checking "
            "cached test-report currency; 'skip' forces a complete re-download "
            "and bypasses cached test results; 'reuse' trusts the local cache "
            "as-is with no freshness round-trip.",
        ),
        skip_cache: Optional[bool] = Field(
            None,
            description="Deprecated alias for ``cache``; prefer ``cache``. When "
            "set it overrides ``cache``: True→'skip', False→'check'.",
        ),
    ) -> str:
        """
        Schedule comprehensive model testing and return a run id immediately.

        Testing (``bioimageio.core.test_description``) runs as a background
        job. This call returns right away with just the ``test_run_id``
        string; poll ``get_test_status(test_run_id)`` for the shared
        progress dict and, once the run finishes, the full report
        in ``result``::

            "tj-…"  # the returned test_run_id
            # then poll get_test_status(test_run_id) →
            # {"state": "running", "submitted_at": 1735689590.0,
            #  "completed_at": 1735689645.0, "result": {...test_report...},
            #  "stages": {...}}

        Caching behavior:
        - Cached test reports are locally stored at ``<model_package>/.test_cache.json``.
        - Cached results are reused only when ``cache != "skip"`` AND the model
            package has not changed (same ``latest_remote_modified``) AND the cached
            ``test_report['env']`` versions for ``bioimageio.core`` and
            ``bioimageio.spec`` match the currently installed versions AND the
            report's ``test_environment`` matches the environment requested for
            this run — a standard run never reuses a custom-env report, or
            vice versa.
        - When the local cache is absent (evicted / fresh package download),
            the published report in the ``bioimage-io/test-reports`` collection
            is reused instead — subject to the same currency check — and the
            local cache is warmed from it. This survives package eviction and
            replica restarts without re-running the test.
        - ``cache="skip"`` forces a complete model package re-download,
            bypasses cached test results, and runs a fresh test.

        Environment mode:
        - ``custom_environment=False``: the test runs in the
            RuntimeDeployment's own venv — the same interpreter that will serve
            ``infer()``. That venv ships Cellpose 4; Cellpose models older than
            Cellpose 4 need the incompatible Cellpose-3 runtime and only test
            green with ``custom_environment=True``.
        - ``custom_environment=True``: the test runs inside the conda
            environment declared by the model's own weights description
            (``bioimageio.core`` ``runtime_env="as-described"``). The env is
            cached on the shared PVC (LRU-evicted under a size ceiling, not
            removed per call) and protected from eviction while in use.
        - ``custom_environment=None`` (default): inherit the environment of the
            model's prior report for the matching slot (the staged report when
            ``stage=True``, the published report when ``stage=False``) — a model
            last tested ``custom`` re-tests ``custom``, otherwise ``standard``.
            Falls back to False when no prior report exists for that slot.

        Report publishing:
        - The report is published to the dedicated
            ``bioimage-io/test-report-<model-id>`` artifact under the
            ``bioimage-io/test-reports`` collection — ``staged/test_report.json``
            when ``stage=True`` else ``published/test_report.json`` — using the
            app's own bioimage-io workspace token. Model contributors have no
            write access to that collection; reports are world-readable.
        - Every report embeds ``latest_remote_modified`` (the model artifact's
            last file-change time) so consumers can tell whether a report is
            current with the artifact.
        - If the app token is not scoped to the ``bioimage-io`` workspace,
            publishing is skipped — the report is still cached locally and
            returned via ``get_test_status``.
        """
        # Resolve the default (None) env from the model's prior report for the
        # matching slot: a model last tested in its custom env re-tests custom,
        # else standard. Staged reads never influence a published run and vice
        # versa. No prior report → standard (False).
        cache = self._resolve_cache(cache, skip_cache)

        if custom_environment is None:
            model_alias = model_id.rsplit("/", 1)[-1]
            prior_report = await self._read_published_report(model_alias, stage)
            custom_environment = (
                (prior_report or {}).get("test_environment") == "custom"
            )

        job = self._new_test_job(model_id, custom_environment)

        async def _bg_execute():
            try:
                report = await self._execute_test(
                    job=job,
                    model_id=model_id,
                    stage=stage,
                    custom_environment=custom_environment,
                    cache=cache,
                )
                self._update_test_job(job, state="completed", result=report)
            except Exception as exc:
                logger.error(
                    f"❌ Test run {job['job_id']} for '{model_id}' failed: {exc}"
                )
                # Failure surfaces via ``result = {"error": ...}`` so the
                # progress dict shape stays uniform across success and
                # failure (no separate ``error`` field).
                self._update_test_job(
                    job, state="failed", result={"error": str(exc)}
                )

        job["_task"] = asyncio.create_task(_bg_execute())
        return job["job_id"]

    async def _execute_test(
        self,
        job: dict,
        model_id: str,
        stage: bool,
        custom_environment: bool,
        cache: str,
    ) -> dict:
        """Run the full test pipeline for a scheduled run and return the report.

        Advances ``job`` state through ``model_download`` → ``env_setup`` →
        ``running`` and publishes the report to the ``bioimage-io/test-reports``
        collection on completion. Raising here marks the run ``failed``; a test
        that runs but fails validation instead returns a report with
        ``status="failed"``.
        """
        import aiofiles

        logger.info(
            f"🧪 Testing model '{model_id}' (run {job['job_id']}, stage={stage}, "
            f"cache={cache}, custom_environment={custom_environment})."
        )

        # Publish-report invariant, enforced at the boundary: a stage=False
        # (published) test requires a committed version and a public status —
        # draft/in-review/in-revision never yield a published report even if a
        # committed version somehow exists.
        if not stage:
            model_alias = model_id.rsplit("/", 1)[-1]
            model_artifact = await self.artifact_manager.read(
                f"{self._TEST_REPORTS_WORKSPACE}/{model_alias}", silent=True
            )
            status = (model_artifact.get("manifest") or {}).get("status")
            if not model_artifact.get("versions") or status in (
                "draft",
                "in-review",
                "in-revision",
            ):
                raise RuntimeError(
                    f"Cannot test '{model_id}' as published (stage=False): the "
                    f"model must have a committed version and a public status "
                    f"(got versions={bool(model_artifact.get('versions'))}, "
                    f"status={status!r}). Use stage=True to test staged edits."
                )

        self._update_test_job(job, state="model_download")
        # Get model package with access tracking
        package = await self.model_cache.get_model_package(
            model_id=model_id,
            stage=stage,
            allow_unpublished=True,
            cache=cache,
        )

        # Silent fallback: models that declare no custom env yaml (no
        # ``dependencies.source`` on any weight format) get an
        # auto-generated framework-default env from bioimageio.spec
        # which is often unsolvable (mkl↔old-pytorch conflict on
        # pytorch_state_dict, etc.). Treat ``custom_environment=True``
        # as a no-op in that case rather than surfacing a mamba solver
        # error. Update the job flag too so the queue_position math
        # matches actual behaviour (only real custom-env runs queue).
        if custom_environment and not await asyncio.to_thread(
            self._model_declares_custom_env, package.source
        ):
            logger.info(
                f"🔁 '{model_id}' has no environment.yaml on any weight "
                f"format — falling back to standard environment "
                f"(custom_environment ignored)."
            )
            custom_environment = False
            job["custom_environment"] = False

        # Use context manager to track access and prevent eviction during test
        async with package:
            logger.info(f"📍 Model source for '{model_id}': {package.source}")

            # Package preparation: fetch any gated huggingface.co weight
            # source with the HF token and repoint the RDF at the local copy,
            # so the test subprocess loads it offline without the token.
            await self._localize_hf_weights(package, model_id)
            test_report: Optional[dict] = None
            tested_at: Optional[float] = None
            should_run_test = True
            should_cache_report = True
            # A loaded-but-stale local report means the model or env changed,
            # so the collection report is stale too — skip the fallback then.
            local_report_stale = False

            # Check for cached test results
            test_report_path = package.package_path / ".test_cache.json"
            current_versions = self._get_bioimageio_versions()

            if cache != "skip" and await asyncio.to_thread(test_report_path.exists):
                try:
                    async with aiofiles.open(test_report_path, "r") as f:
                        content = await f.read()
                        cached_data = await asyncio.to_thread(json.loads, content)

                    cached_report = cached_data["test_report"]
                    if self._report_is_current(
                        cached_report, package, current_versions, custom_environment
                    ):
                        logger.info(
                            f"💾 Model '{model_id}' unchanged since last test, using cached results."
                        )
                        test_report = cached_report
                        tested_at = test_report["tested_at"]
                        should_run_test = False
                        should_cache_report = False
                    else:
                        local_report_stale = True
                        cached_remote_modified = cached_data["latest_remote_modified"]
                        if package.latest_remote_modified != cached_remote_modified:
                            logger.info(
                                f"🔄 Model '{model_id}' has been updated, re-running tests "
                                f"(cached: {cached_remote_modified}, current: {package.latest_remote_modified})"
                            )
                        else:
                            logger.info(
                                f"🔄 Model '{model_id}' unchanged but test environment changed, re-running tests."
                            )
                except (json.JSONDecodeError, KeyError, OSError, IOError) as e:
                    logger.warning(
                        f"⚠️ Failed to load cached test results for '{model_id}': {e}. Running fresh test."
                    )

            # Fall back to the published report in the collection when the
            # local cache is absent (evicted / fresh package download).
            if should_run_test and cache != "skip" and not local_report_stale:
                model_alias = model_id.rsplit("/", 1)[-1]
                published = await self._read_published_report(model_alias, stage)
                if published and self._report_is_current(
                    published, package, current_versions, custom_environment
                ):
                    logger.info(
                        f"💾 Reusing published test report for '{model_id}' from "
                        f"'{self._TEST_REPORTS_COLLECTION}'."
                    )
                    test_report = published
                    tested_at = test_report["tested_at"]
                    should_run_test = False
                    should_cache_report = True

            # Run the test unless we already accepted a cached result
            if should_run_test:
                tested_at = time.time()
                needed_envs: List[str] = []
                try:
                    # For ``custom_environment=True``: build every conda
                    # env the model needs on THIS EntryDeployment replica
                    # first — Entry is CPU-only, so a ~10-min mamba
                    # solve doesn't hold the GPU-bound RuntimeDeployment
                    # replica. Multiple envs (one per weight format)
                    # are built concurrently. RuntimeDeployment then invokes
                    # ``bioimageio.core.test_description`` which does
                    # its own env-existence check (``mamba run -n
                    # <hash> python --version``) and finds the envs
                    # already present on the shared PVC-backed HOME,
                    # so ``mamba env create`` is a no-op there and the
                    # RuntimeDeployment only spends time on the actual test
                    # inference. If HOME is NOT shared cross-replica
                    # for some reason, RuntimeDeployment falls through to
                    # its normal env-create flow — graceful degrade
                    # to pre-1.10.0 behavior with no correctness gap.
                    if custom_environment:
                        self._update_test_job(job, state="env_setup")
                        needed_envs = await self._prebuild_conda_envs(
                            rdf_path=package.source,
                            model_id=model_id,
                            job=job,
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
                finally:
                    # Release conda envs once the GPU run is done (success or
                    # failure) so eviction can reclaim them again.
                    self._release_inuse_envs(needed_envs)

                test_report = self._stamp_runtime_versions_in_test_env(
                    test_report, current_versions
                )

                # Add tested_at timestamp to the test report so the report is self-contained.
                test_report["tested_at"] = tested_at

                # Label which environment the test actually ran in. Uses the
                # possibly-downgraded ``custom_environment`` (a model that
                # declares no environment.yaml falls back to standard above),
                # so the label reflects reality, not just the request.
                test_report["test_environment"] = (
                    "custom" if custom_environment else "standard"
                )

            # Record the model artifact's last file-change time in the report
            # so consumers can tell whether a stored report is current with the
            # artifact. Set on both the fresh and the cached path.
            test_report["latest_remote_modified"] = package.latest_remote_modified

            # ``id`` is optional in an RDF, so core emits null for models that
            # omit it and consumers keyed on the field drop the whole report.
            # Stamp it from the identity we were called with; the bare alias is
            # what core writes when the RDF does carry one.
            test_report["id"] = model_id.rsplit("/", 1)[-1]

            # Persist to disk for fresh successful runs and for reports
            # reused from the collection, so the next run loads from disk
            # instead of re-fetching the report from the collection.
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

            await self._upload_test_report(model_id, stage, test_report)

        return test_report

    async def _read_published_report(
        self, model_alias: str, stage: bool
    ) -> Optional[dict]:
        """Fetch the published test report for a model from the collection.

        The report artifact is world-readable, so this is a plain HTTP GET
        of the committed file — no artifact-manager RPC. Slot-aware:
        ``staged/test_report.json`` when ``stage=True`` else
        ``published/test_report.json``. Returns ``None`` when the artifact
        or slot does not exist yet, so callers fall through to a fresh test.
        """
        slot = "staged" if stage else "published"
        url = (
            f"{self.server_url}/{self._TEST_REPORTS_WORKSPACE}/artifacts/"
            f"test-report-{model_alias}/files/{slot}/test_report.json"
        )
        try:
            response = await self.model_cache._get_url_with_retry(url, params=None)
            if response.status_code != 200:
                return None
            return response.json()
        except Exception:
            return None

    def _compute_report_score(self, test_report: dict) -> float:
        """Additive ranking score for a published model's test report.

        0 unless the format is valid; then +1 (valid format) +2 (default-env
        inference passed) +4 (reproducibility passed) + metadata completeness
        (0..1). Tier gaps exceed the max metadata bonus, so a higher tier
        always outranks a lower one. Each tier presupposes the one below it.
        """
        status = test_report.get("status")
        if status not in ("valid-format", "passed"):
            return 0.0
        score = 1.0
        inference_check = test_report.get("inference_check") or {}
        if inference_check.get("status") == "passed":
            score += 2.0
        if status == "passed":
            score += 4.0
        try:
            score += float(test_report.get("metadata_completeness") or 0.0)
        except (TypeError, ValueError):
            pass
        return score

    async def _thumbnail_covers(
        self, model_alias: str, covers: list, stage: bool
    ) -> list:
        """Swap each cover for its thumbnail sibling when the model artifact
        ships one: ``<cover-stem>.thumbnail.<ext>`` (jpg/jpeg preferred over
        png), e.g. ``cover.png`` → ``cover.thumbnail.jpg``. Non-string covers
        and covers without a thumbnail are kept as-is. Returns the covers
        unchanged on any error — a report upload must never fail over cover
        cosmetics.
        """
        try:
            files = await self.artifact_manager.list_files(
                f"{self._TEST_REPORTS_WORKSPACE}/{model_alias}",
                version=("stage" if stage else None),
            )
            names = {(f.get("name") if isinstance(f, dict) else f) for f in files}
            resolved = []
            for cover in covers:
                if not isinstance(cover, str):
                    resolved.append(cover)
                    continue
                stem = cover.rsplit(".", 1)[0] if "." in cover else cover
                thumb = next(
                    (
                        c
                        for c in (
                            f"{stem}.thumbnail.jpg",
                            f"{stem}.thumbnail.jpeg",
                            f"{stem}.thumbnail.png",
                        )
                        if c in names
                    ),
                    cover,
                )
                resolved.append(thumb)
            return resolved
        except Exception as e:
            logger.warning(
                f"⚠️ Thumbnail cover resolution failed for '{model_alias}': {e}"
            )
            return covers

    async def _upload_test_report(
        self, model_id: str, stage: bool, test_report: dict
    ) -> None:
        """Publish ``test_report`` to the per-model artifact under the
        ``bioimage-io/test-reports`` collection.

        Writes ``staged/test_report.json`` when ``stage=True`` else
        ``published/test_report.json``, using the app's own workspace token.
        Skipped (report still cached + returned) when the app token is not
        scoped to the ``bioimage-io`` workspace. A per-artifact lock serializes
        a published and a staged upload for the same model so they don't race
        on the artifact's single staging area. Failures are logged and
        swallowed — the report is already cached and returned regardless.
        """
        if not self._test_reports_writable:
            return

        model_alias = model_id.rsplit("/", 1)[-1]
        report_artifact_id = (
            f"{self._TEST_REPORTS_WORKSPACE}/test-report-{model_alias}"
        )
        slot = "staged" if stage else "published"
        file_path = f"{slot}/test_report.json"

        lock = self._report_locks.setdefault(report_artifact_id, asyncio.Lock())
        async with lock:
            try:
                try:
                    artifact = await self.artifact_manager.read(
                        report_artifact_id, silent=True
                    )
                    manifest = dict(artifact.get("manifest") or {})
                except Exception:
                    manifest = None

                # Skip a redundant commit when the stored report is identical
                # (same ``tested_at`` — e.g. a cache-hit re-run).
                if manifest is not None:
                    try:
                        existing_url = await self.artifact_manager.get_file(
                            report_artifact_id, file_path=file_path
                        )
                        existing = (
                            await self.model_cache._get_url_with_retry(
                                existing_url, params=None
                            )
                        ).json()
                        if float(existing.get("tested_at", -1.0)) == float(
                            test_report.get("tested_at", 0.0)
                        ):
                            logger.info(
                                f"ℹ️ {slot} test report for '{model_alias}' is up to "
                                f"date; skipping upload."
                            )
                            return
                    except Exception:
                        pass

                # Mirror the model's identifying metadata from the bioimage.io
                # collection so the test-reports collection is self-describing,
                # and record whether the model currently has a committed public
                # version (a non-empty ``versions`` list). Fail closed to
                # unpublished on a read error.
                model_meta = {}
                model_is_published = False
                model_confirmed_unpublished = False
                try:
                    model_artifact = await self.artifact_manager.read(
                        f"{self._TEST_REPORTS_WORKSPACE}/{model_alias}", silent=True
                    )
                    model_is_published = bool(model_artifact.get("versions"))
                    model_confirmed_unpublished = not model_is_published
                    model_manifest = model_artifact.get("manifest") or {}
                    for key in (
                        "name",
                        "id",
                        "description",
                        "id_emoji",
                        "tags",
                        "covers",
                    ):
                        if key in model_manifest:
                            model_meta[key] = model_manifest[key]
                except Exception as e:
                    logger.warning(
                        f"⚠️ Could not copy model metadata for '{model_alias}': {e}"
                    )

                if "covers" in model_meta:
                    model_meta["covers"] = await self._thumbnail_covers(
                        model_alias, model_meta["covers"], stage
                    )

                if manifest is None:
                    manifest = await self._create_report_artifact(
                        report_artifact_id, model_alias, model_meta
                    )
                else:
                    manifest.update(model_meta)

                if not stage:
                    manifest["score"] = self._compute_report_score(test_report)

                # Type the report from the model's CURRENT published state — a
                # non-empty ``versions`` list, matching the website's
                # ``isPublished = versions.length > 0`` and the public grid's
                # ``type == published-model`` filter. Written whenever the model
                # read succeeds so it self-heals both ways: it stays
                # ``published-model`` across a staged re-test of a still-published
                # model (``versions`` keeps its committed entries while staging),
                # and drops back to ``staged-model`` once the model is
                # unpublished. Keying on the presence of a published report file
                # instead would strand the report at ``published-model`` after an
                # unpublish, since that file is never removed — leaking a phantom
                # card onto the grid. On a read error the type is left untouched
                # (neither state is confirmed); the next successful upload heals.
                edit_kwargs = {"manifest": manifest, "stage": True}
                if model_is_published:
                    edit_kwargs["type"] = "published-model"
                elif model_confirmed_unpublished:
                    edit_kwargs["type"] = "staged-model"
                await self.artifact_manager.edit(
                    report_artifact_id, **edit_kwargs
                )
                # A model with no committed version keeps no published report:
                # published uploads are blocked upstream, so a leftover published
                # file is never overwritten and would strand the report on the
                # public grid. Drop it (idempotent) only when the model read
                # positively confirmed an empty ``versions`` — never on a read
                # error, which must not destroy a real published report.
                if model_confirmed_unpublished:
                    try:
                        await self.artifact_manager.remove_file(
                            report_artifact_id,
                            file_path="published/test_report.json",
                        )
                    except Exception:
                        pass
                upload_url = await self.artifact_manager.put_file(
                    report_artifact_id, file_path=file_path
                )
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.put(
                        upload_url, data=json.dumps(test_report)
                    )
                    response.raise_for_status()
                await self.artifact_manager.commit(report_artifact_id)
                logger.info(
                    f"📤 Published {slot} test report for '{model_alias}' to "
                    f"'{report_artifact_id}'."
                )
            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to publish test report for '{model_alias}' to "
                    f"'{report_artifact_id}': {e}"
                )

    async def _create_report_artifact(
        self, report_artifact_id: str, model_alias: str, manifest: dict
    ) -> dict:
        """Create the per-model report artifact under the test-reports
        collection, tolerating a concurrent creator by re-reading on failure.

        ``manifest`` carries the model's identifying metadata copied by the
        caller; a minimal name is substituted only when that copy came back
        empty, since the artifact manager requires a non-empty manifest.
        Returns the manifest the artifact was created with.
        """
        manifest = dict(manifest) or {"name": f"test-report-{model_alias}"}
        try:
            await self.artifact_manager.create(
                parent_id=self._TEST_REPORTS_COLLECTION,
                alias=f"test-report-{model_alias}",
                type="staged-model",
                manifest=manifest,
            )
            logger.info(f"🆕 Created report artifact '{report_artifact_id}'.")
        except Exception:
            # A concurrent call likely created it first; confirm it now exists,
            # else re-raise so the caller logs a real failure.
            await self.artifact_manager.read(report_artifact_id, silent=True)
        return manifest

    @bioengine.method
    async def get_test_status(
        self,
        test_run_id: str = Field(
            ...,
            description="Run id returned by ``test()``.",
        ),
    ) -> Dict[str, Union[str, float, dict, None]]:
        """Return the shared progress dict for a test run.

        Response shape (same schema as ``get_infer_status``)::

            {
              "state":          str,          # queued / model_download / env_setup /
                                              # running / completed / failed / cancelled
              "submitted_at":   float,        # ts when the run was queued
              "completed_at":   float | None, # ts when finished, None until then
              "result":         dict | None,  # test report on success,
                                              # {"error": str} on failure
              "stages": {                     # per-stage timeline + queue position
                "model_download": {"start": float|None, "end": float|None},
                "env_setup": {"start": float|None, "end": float|None,
                              "queue_position": int|None},  # single-slot env-build queue
                "run":       {"start": float|None, "end": float|None,
                              "queue_position": int|None},  # combined test+infer GPU queue
              },
            }

        ``queue_position`` inside a stage is non-None only while the job is
        in that stage: 0 = executing, N = N jobs ahead (up to 2 GPU jobs
        may run at once, so ``run`` can drop 2→0).

        Runs are held for 24 hours after completion, then dropped. The
        registry is per-Entry replica and in-memory — a run started on
        one replica is unknown to others, and replica restarts drop
        everything.
        """
        job = self._test_jobs.get(test_run_id)
        if job is None:
            raise KeyError(
                f"Unknown test_run_id {test_run_id!r}. Runs live in-memory per "
                f"Entry replica and expire 24 hours after completion. Start a "
                f"fresh run via test()."
            )
        return self._job_progress(job)

    @bioengine.method
    async def cancel_request(
        self,
        request_id: str = Field(
            ...,
            description="Id returned by ``test()`` (``tj-…``) or ``infer()`` (``ij-…``).",
        ),
    ) -> Dict[str, Union[str, bool]]:
        """Cancel a pending or in-flight ``test()`` / ``infer()`` request.

        Best-effort: cancels the background orchestration task and marks the
        request ``cancelled`` (a terminal state, ``result={"error":
        "cancelled"}``). A GPU call already dispatched to RuntimeDeployment is
        not force-killed — it runs to completion on the replica and its result
        is discarded; the test path's conda-env hold is released by its
        ``finally`` and the infer path's staged inputs are deleted here.

        Routing is by id prefix (``tj-`` test, ``ij-`` infer). An already
        finished request is a no-op (``cancelled: False``). Unknown ids raise
        ``KeyError`` — runs live in-memory per Entry replica and expire after
        their TTL.

        Returns ``{"request_id", "state", "cancelled"}``.
        """
        if request_id.startswith("tj-"):
            registry = self._test_jobs
        elif request_id.startswith("ij-"):
            registry = self._infer_jobs
        else:
            registry = (
                self._test_jobs
                if request_id in self._test_jobs
                else self._infer_jobs
            )
        job = registry.get(request_id)
        if job is None:
            raise KeyError(
                f"Unknown request_id {request_id!r}. Requests live in-memory per "
                f"Entry replica and expire after completion. Start a fresh run."
            )

        if job["state"] in ("completed", "failed", "cancelled"):
            return {
                "request_id": request_id,
                "state": job["state"],
                "cancelled": False,
            }

        task = job.get("_task")
        if task is not None and not task.done():
            task.cancel()

        is_infer = registry is self._infer_jobs
        update = self._update_infer_job if is_infer else self._update_test_job
        update(job, state="cancelled", result={"error": "cancelled"})
        if is_infer and self._inference_dir is not None:
            await asyncio.to_thread(
                shutil.rmtree,
                str(self._inference_dir / request_id),
                ignore_errors=True,
            )

        logger.info(f"🛑 Cancelled request {request_id!r}.")
        return {"request_id": request_id, "state": "cancelled", "cancelled": True}

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
        if self._is_anonymous:
            raise RuntimeError(
                "Upload is unavailable: this model-runner was deployed without a "
                "HYPHA_TOKEN and runs in public read-only mode. Pass an ndarray or "
                "a public image URL to infer() directly, or redeploy the runner "
                "with --hypha-token to enable S3 uploads."
            )

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
        cache: Literal["skip", "check", "reuse"] = Field(
            "check",
            description="Model-cache policy (same meaning on ``test`` and on "
            "cellpose3-runner): 'check' (default) verifies the cached package "
            "against upstream and re-downloads if stale, re-checking a warm "
            "model too; 'skip' forces a complete re-download and reload even if "
            "the model is warm; 'reuse' trusts the local cache and any warm "
            "model as-is with no freshness round-trip.",
        ),
        skip_cache: Optional[bool] = Field(
            None,
            description="Deprecated alias for ``cache``; prefer ``cache``. When "
            "set it overrides ``cache``: True→'skip', False→'check'.",
        ),
        preprocessing: Optional[Dict[str, Optional[dict]]] = Field(
            None,
            description="Per-request overrides of the model's declared "
            "preprocessing, as {op_id: {kwarg: value}} (e.g. "
            '{"scale_range": {"min_percentile": 1.0, "max_percentile": 99.0}}). '
            "A value of None drops the op instead of patching it. Applied to an "
            "in-memory copy of the RDF; the published artifact is never "
            "modified. An op id the model does not declare is an error.",
        ),
        postprocessing: Optional[Dict[str, Optional[dict]]] = Field(
            None,
            description="Per-request overrides of the model's declared "
            "postprocessing, same form as ``preprocessing`` (e.g. "
            '{"cellpose_flow_dynamics": {"flow_threshold": 0.4, '
            '"cellprob_threshold": 0.0, "min_size": 15}}). Dropping an op with '
            "None returns the model's raw output — "
            '{"cellpose_flow_dynamics": None} yields Cellpose-4 flow fields.',
        ),
        return_download_url: Optional[bool] = Field(
            False,
            description="If True, each array in the output will be saved to a temporary .npy file in S3 and the output value will be a presigned download URL (str) instead of the raw np.ndarray. The URL is valid for 1 hour.",
        ),
    ) -> str:
        """
        Submit an inference request and return a ``request_id`` immediately.

        Cellpose-4 (Cellpose-SAM / Cellpose-DINO) models run natively here.
        Cellpose models older than Cellpose 4 are not
        supported — this runner's runtime ships Cellpose 4, whose API is
        incompatible. Run them via the ``bioimage-io/cellpose3-runner`` app
        instead.

        The call resolves URL / S3-path inputs to numpy arrays, spills
        them to disk under ``$HOME/.model-runner-inference/<request_id>/input/``
        so large images don't sit in RAM during the queue+download
        wait, then schedules the model download + GPU work as a
        background job and returns::

            "ij-…"  # the returned request_id

        Poll ``get_infer_status(request_id)`` for the shared
        progress dict::

            {"state": "queued", "submitted_at": 1735689590.0,
             "completed_at": None, "result": None, "stages": {...}}

        Once ``result`` is populated, the outputs are read off disk on
        that first poll and the request dir is deleted; subsequent
        polls return the cached result. Jobs are held for 1 hour after
        completion, then swept.

        When ``return_download_url=True`` the ``result`` maps output
        keys to presigned S3 URLs instead of raw arrays; the S3 upload
        happens on the poll that materialises the result.

        Raises:
            ValueError: if ``model_id`` is a URL, or the resolved
                inputs are empty / not decodeable.
            FileNotFoundError: if a URL / temporary file path is
                provided but the resource does not exist or has expired.
        """
        logger.info(f"🤖 Queuing inference for model '{model_id}'...")

        if model_id.rsplit("/", 1)[-1] in CELLPOSE3_MODELS:
            raise ValueError(
                f"model_id {model_id!r} is a Cellpose-3 model, which this "
                f"runner's Cellpose-4 runtime cannot run. Use the "
                f"'bioimage-io/cellpose3-runner' app instead."
            )

        # Resolve any URL or temporary file path strings to numpy
        # arrays BEFORE handing off — matches the user's spec ("save
        # the image the moment it's received") and surfaces broken
        # URLs to the caller synchronously instead of hiding them in
        # the background task.
        if isinstance(inputs, str):
            inputs = await self._load_image_from_source(inputs)
        elif isinstance(inputs, dict):
            resolved: Dict[str, np.ndarray] = {}
            for key, value in inputs.items():
                if isinstance(value, str):
                    resolved[key] = await self._load_image_from_source(value)
                else:
                    resolved[key] = value
            inputs = resolved

        # Normalize to Dict[str, np.ndarray] so the on-disk layout has
        # one file per input key. A bare ndarray is the "single unnamed
        # input" case: stage it under a sentinel key so the runtime can
        # hand it to bioimageio.core as a bare array, letting core map it
        # to the model's sole input member id. Inventing a real-looking
        # key here (e.g. "input") breaks every model whose input member
        # id differs — v0.4 models key on the input's ``name`` (e.g.
        # "input0"), so a hardcoded "input" is rejected as unexpected.
        if isinstance(inputs, np.ndarray):
            inputs = {SINGLE_INPUT_KEY: inputs}
        if not isinstance(inputs, dict) or not inputs:
            raise ValueError(
                "``inputs`` must resolve to a non-empty dict of numpy arrays."
            )
        for key, value in inputs.items():
            if not isinstance(value, np.ndarray):
                raise ValueError(
                    f"Input {key!r} did not resolve to a numpy array (got "
                    f"{type(value).__name__})."
                )

        cache = self._resolve_cache(cache, skip_cache)

        job = self._new_infer_job(
            model_id=model_id, return_download_url=bool(return_download_url)
        )
        request_id = job["job_id"]
        request_dir = self._inference_dir / request_id
        input_dir = request_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Write inputs as uncompressed .npy — spec is speed over disk
        # space. ``np.save`` writes atomically (single ``open`` +
        # ``write``), and the runtime only touches this dir after we
        # dispatch, so no partial-read race.
        def _save_inputs() -> None:
            for key, arr in inputs.items():
                np.save(str(input_dir / f"{key}.npy"), arr)

        await asyncio.to_thread(_save_inputs)
        logger.info(
            f"💾 Staged {len(inputs)} input tensor(s) for request "
            f"{request_id!r} at {input_dir}"
        )

        # Kick off the background task. Any exception it raises is
        # captured onto the job record — never propagates up through
        # the asyncio task's default exception handler.
        job["_task"] = asyncio.create_task(
            self._execute_infer(
                job=job,
                model_id=model_id,
                weights_format=weights_format,
                device=device,
                default_blocksize_parameter=default_blocksize_parameter,
                sample_id=sample_id,
                cache=cache,
                overrides={
                    "preprocessing": preprocessing or {},
                    "postprocessing": postprocessing or {},
                },
            )
        )
        return request_id

    async def _execute_infer(
        self,
        job: dict,
        model_id: str,
        weights_format: Optional[str],
        device: Optional[Literal["cuda", "cpu"]],
        default_blocksize_parameter: Optional[int],
        sample_id: Optional[str],
        cache: str,
        overrides: Dict[str, Dict[str, Optional[dict]]],
    ) -> None:
        """Background inference driver — model download → runtime dispatch.

        Mirrors ``_execute_test`` for the infer path. State transitions:

        * ``queued`` → ``model_download`` (ts stamped)
        * → ``running`` (entry timestamp — the true ``runtime_started_at``
          is filled lazily on poll from ``state.json``)
        * → ``completed`` on success (result NOT read here; the first
          poll after completion reads outputs off disk)
        * → ``failed`` on any exception; the on-disk request dir is
          cleaned up so we don't leak files.
        """
        from ray.exceptions import RayTaskError

        request_id = job["job_id"]
        try:
            self._update_infer_job(job, state="model_download")
            package = await self.model_cache.get_model_package(
                model_id=model_id,
                stage=False,
                allow_unpublished=False,
                cache=cache,
            )
            async with package:
                logger.info(
                    f"📍 Model source for request {request_id!r}: "
                    f"{package.source} "
                    f"(latest_remote_modified: {package.latest_remote_modified})"
                )

                # Package preparation: fetch any gated huggingface.co weight
                # source with the HF token and repoint the RDF at the local
                # copy, so the infer subprocess loads it offline without the
                # token.
                await self._localize_hf_weights(package, model_id)

                # Move to ``running`` before the RPC. The RPC itself
                # may queue at the runtime router; the poll refines
                # ``running_ts`` from the state.json marker.
                self._update_infer_job(job, state="running")
                await self.runtime.predict_from_disk(
                    request_id=request_id,
                    rdf_path=package.source,
                    weights_format=weights_format,
                    device=device,
                    default_blocksize_parameter=default_blocksize_parameter,
                    sample_id=sample_id,
                    remote_modified=package.latest_remote_modified,
                    force_reload=(cache == "skip"),
                    overrides=overrides,
                )

            self._update_infer_job(job, state="completed")
            logger.info(f"✅ Inference completed for request {request_id!r}.")
        except Exception as exc:
            # Wrap RayTaskError so the caller sees a plain RuntimeError
            # (matches the sync-path error surface of the previous API).
            if isinstance(exc, RayTaskError):
                exc = RuntimeError(
                    f"Runtime call failed for request {request_id!r}: {exc}"
                )
            logger.error(
                f"❌ Infer run {request_id!r} for '{model_id}' failed: {exc}"
            )
            # Clean up any staged on-disk data — we don't want it
            # sitting around after failure.
            if self._inference_dir is not None:
                await asyncio.to_thread(
                    shutil.rmtree,
                    str(self._inference_dir / request_id),
                    ignore_errors=True,
                )
            self._update_infer_job(
                job, state="failed", result={"error": str(exc)}
            )

    async def _materialize_infer_result(self, job: dict) -> None:
        """Read outputs off disk, apply ``return_download_url`` conversion
        if requested, cache the result on the job, and delete the
        request dir. Called at most once per job — guarded by
        ``job["_result_materialized"]``.
        """
        request_id = job["job_id"]
        request_dir = self._inference_dir / request_id
        output_dir = request_dir / "output"

        def _load_outputs() -> Dict[str, np.ndarray]:
            loaded: Dict[str, np.ndarray] = {}
            if not output_dir.is_dir():
                return loaded
            for entry in sorted(output_dir.iterdir()):
                if entry.is_file() and entry.suffix == ".npy":
                    loaded[entry.stem] = np.load(str(entry))
            return loaded

        outputs = await asyncio.to_thread(_load_outputs)
        if not outputs:
            # Runtime completed the RPC without writing any output —
            # surface as an error rather than an empty success.
            job["result"] = {
                "error": (
                    f"Runtime returned no outputs for request {request_id!r}. "
                    f"Expected .npy files under {output_dir}."
                )
            }
            job["_result_materialized"] = True
            await asyncio.to_thread(
                shutil.rmtree, str(request_dir), ignore_errors=True
            )
            return

        if job["return_download_url"]:
            converted: Dict[str, str] = {}
            for key, array in outputs.items():
                converted[key] = await self._save_array_to_temp_file(array)
            job["result"] = converted
        else:
            job["result"] = outputs

        job["_result_materialized"] = True
        # Free the disk immediately — user's spec: read the result,
        # delete it from disk, return the result in the ``result`` key.
        await asyncio.to_thread(
            shutil.rmtree, str(request_dir), ignore_errors=True
        )
        logger.info(
            f"📤 Materialized {len(outputs)} output(s) for request "
            f"{request_id!r} and cleaned up on-disk data."
        )

    @bioengine.method
    async def get_infer_status(
        self,
        request_id: str = Field(
            ...,
            description="Request id returned by ``infer()``.",
        ),
    ) -> Dict[str, Union[str, float, dict, None]]:
        """Return the shared progress dict for an infer request.

        Response shape (same schema as ``get_test_status``)::

            {
              "state":          str,          # queued / model_download /
                                              # running / completed / failed / cancelled
              "submitted_at":   float,        # ts when the request was queued
              "completed_at":   float | None, # ts when finished, None until then
              "result":         dict | None,  # inference dict on success,
                                              # {"error": str} on failure
              "stages": {                     # per-stage timeline + queue position
                "model_download": {"start": float|None, "end": float|None},
                "env_setup": {"start": None, "end": None,
                              "queue_position": None},  # unused on the infer path
                "run":       {"start": float|None, "end": float|None,
                              "queue_position": int|None},  # combined test+infer GPU queue
              },
            }

        On the FIRST poll after ``result`` becomes available the
        outputs are read off ``<request_id>/output/*.npy``, converted
        to presigned S3 URLs if ``return_download_url=True`` was set
        on submit, cached in memory, and the request dir is deleted.
        Subsequent polls return the cached result until the job TTL
        (1 hour after completion) sweeps it.

        Requests live in-memory per Entry replica and disappear on
        replica restart. Unknown ids raise ``KeyError``.
        """
        job = self._infer_jobs.get(request_id)
        if job is None:
            raise KeyError(
                f"Unknown request_id {request_id!r}. Requests live in-memory "
                f"per Entry replica and expire {self._INFER_JOBS_TTL_SEC // 60} "
                f"minutes after completion. Start a fresh request via infer()."
            )

        # Materialise the result on the first poll after completion —
        # reading a completed job's outputs off disk exactly once, then
        # holding them in memory. A ``failed`` state already has the
        # error stored via ``_execute_infer``, so nothing to materialise.
        if (
            job["state"] == "completed"
            and not job["_result_materialized"]
        ):
            await self._materialize_infer_result(job)

        return self._infer_job_progress(job)


