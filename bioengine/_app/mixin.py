"""Replica-side wiring installed by ``@bioengine.app``.

Replaces the worker-side ``_update_init`` / ``_update_async_init`` /
``_update_test_deployment`` / ``_update_health_check`` monkey-patching that
used to live in ``bioengine/apps/builder.py``. Everything in this module
runs *inside* a Ray Serve replica process — the worker never imports it.

The decorator (``bioengine/app/decorators.py``) calls ``_setup_replica`` from
inside the wrapped ``__init__``, then installs ``_make_check_health(...)`` as
the class's ``check_health`` method that Ray Serve polls.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def _setup_replica(instance: Any) -> None:
    """Configure the replica process before the user's ``__init__`` runs.

    Reads its configuration from environment variables set by the BioEngine
    worker in the replica's ``runtime_env``:

    * ``BIOENGINE_APPLICATION_ID`` — the deploy-time application id
    * ``BIOENGINE_PROXY_ACTOR_NAME`` — Ray named-actor for replica tracking
    * ``BIOENGINE_DATA_SERVER_URL`` — for ``bioengine.datasets``
    * ``HOME`` — the application's isolated working directory
    * keys prefixed with ``_BIOENGINE_SECRET_`` — secret env vars to unmask

    All state added to ``instance`` is namespaced under ``_bioengine_*`` so
    user code can use the rest of the attribute namespace freely.
    """
    os.environ["BIOENGINE_REPLICA"] = "1"

    logger = logging.getLogger("ray.serve")
    if os.environ.get("BIOENGINE_DEBUG") == "1":
        logger.setLevel(logging.DEBUG)

    _register_with_proxy_actor(logger)
    _ensure_working_directory(logger)
    _unmask_secret_env_vars()

    instance._bioengine_replica_initialized = False
    instance._bioengine_replica_test_failed = False
    instance._bioengine_test_task: Optional[asyncio.Task] = None
    instance._bioengine_health_check_lock = asyncio.Lock()


def _register_with_proxy_actor(logger: logging.Logger) -> None:
    """Register this replica with the worker's ``BioEngineProxyActor``.

    A best-effort call: if the actor isn't reachable (e.g. local unit test
    without a Ray cluster) we log and move on.
    """
    proxy_actor_name = os.environ.get("BIOENGINE_PROXY_ACTOR_NAME")
    application_id = os.environ.get("BIOENGINE_APPLICATION_ID")
    if not proxy_actor_name or not application_id:
        return

    try:
        import ray
        from ray import serve
    except ImportError:
        return

    try:
        proxy_actor_handle = ray.get_actor(name=proxy_actor_name, namespace="bioengine")
    except ValueError as e:
        logger.error(f"❌ BioEngineProxyActor '{proxy_actor_name}' not found: {e}")
        return
    except Exception as e:
        logger.error(
            f"❌ Unexpected error getting BioEngineProxyActor '{proxy_actor_name}': {e}",
            exc_info=True,
        )
        return

    try:
        replica_context = serve.get_replica_context()
        proxy_actor_handle.register_serve_replica.remote(
            application_id=application_id,
            deployment_name=replica_context.deployment,
            replica_id=replica_context.replica_tag,
            timezone=time.strftime("%Z"),
        )
        logger.info(
            f"✅ Registered replica '{replica_context.replica_tag}' "
            f"with BioEngineProxyActor."
        )
    except Exception as e:
        logger.error(
            f"❌ Unable to register replica with BioEngineProxyActor: {e}",
            exc_info=True,
        )


def _ensure_working_directory(logger: logging.Logger) -> None:
    """Anchor the replica process at the per-app directory.

    Reads ``BIOENGINE_APP_DIR`` populated by the worker — the replica setup
    hook (:mod:`bioengine._app.replica_init`) created the directory tree
    (``source/``, ``home/``, ``tmp/``) before this runs and pointed ``HOME``
    + ``TMPDIR`` at the right subdirs.

    Falls back to ``$HOME`` only on legacy single-machine deployments where
    no app dir was passed (the worker still injects ``HOME`` in that case,
    pointing at the v0.10-style ``apps_workdir/<app_id>`` layout).
    """
    app_dir_env = os.environ.get("BIOENGINE_APP_DIR")
    if app_dir_env:
        workdir = Path(app_dir_env).resolve()
    else:
        workdir = Path.home().resolve()
        os.environ["HOME"] = str(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(workdir)
    except OSError:
        pass
    logger.info(f"📁 Working directory: {workdir}/")


def _unmask_secret_env_vars() -> None:
    """Restore real values for env vars the worker stored under ``_BIOENGINE_SECRET_``.

    The worker writes secrets to ``runtime_env.env_vars`` as
    ``_BIOENGINE_SECRET_<KEY>=<value>`` so the unprefixed key shows as
    ``*****`` in Ray's actor config UI. We reverse that here.
    """
    prefix = "_BIOENGINE_SECRET_"
    for key, value in list(os.environ.items()):
        if key.startswith(prefix):
            os.environ[key[len(prefix) :]] = value


def _wrap_runtime_handles(
    user_cls: type, kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Replace incoming ``DeploymentHandle`` kwargs with ``BioEngineRuntimeHandle``.

    The decorator scans ``user_cls.__init__``'s type hints once at decoration
    time and records which parameter names are composition handles. At
    construction time we look them up and wrap the handles so user code never
    sees ``.remote()``.
    """
    from ray.serve.handle import DeploymentHandle

    from bioengine._app.runtime_handle import BioEngineRuntimeHandle

    composition_params: Dict[str, str] = getattr(
        user_cls, "_bioengine_composition_params", {}
    )
    if not composition_params:
        return kwargs

    wrapped = dict(kwargs)
    for param_name, target_cls_id in composition_params.items():
        if param_name not in wrapped:
            continue
        value = wrapped[param_name]
        if isinstance(value, DeploymentHandle):
            wrapped[param_name] = BioEngineRuntimeHandle(value, target_cls_id)
    return wrapped


def _make_check_health(
    user_cls: type, lifecycle: Dict[str, Any]
) -> Callable[..., Any]:
    """Build the orchestrating ``check_health`` method installed on the class.

    Ray Serve calls ``check_health`` repeatedly; on the first call we run
    one-shot ``async_init`` + ``smoke_test`` hooks, then on every call we
    run the user's ``health_check`` (if any). Concurrency-safe via an
    asyncio lock pre-installed by ``_setup_replica``.
    """
    async_init_name: Optional[str] = lifecycle.get("async_init")
    smoke_test_name: Optional[str] = lifecycle.get("smoke_test")
    health_check_name: Optional[str] = lifecycle.get("health_check")

    async def check_health(self: Any) -> None:
        logger = logging.getLogger("ray.serve")
        async with self._bioengine_health_check_lock:
            if not self._bioengine_replica_initialized and async_init_name:
                await _invoke_lifecycle(self, async_init_name, "async_init", logger)
                self._bioengine_replica_initialized = True
            elif not self._bioengine_replica_initialized:
                self._bioengine_replica_initialized = True

            if self._bioengine_test_task is None and smoke_test_name:
                logger.info("🚀 Launching smoke test in the background...")
                self._bioengine_test_task = asyncio.create_task(
                    _invoke_lifecycle(self, smoke_test_name, "smoke_test", logger)
                )

            if self._bioengine_test_task is not None and self._bioengine_test_task.done():
                exc = self._bioengine_test_task.exception()
                if exc is not None:
                    self._bioengine_replica_test_failed = True
                    raise RuntimeError(f"Smoke test failed: {exc}") from exc

            if self._bioengine_replica_test_failed:
                raise RuntimeError(
                    "Smoke test failed previously — replica is unhealthy"
                )

            if health_check_name:
                hook = getattr(self, health_check_name)
                if inspect.iscoroutinefunction(hook):
                    await hook()
                else:
                    await asyncio.to_thread(hook)

    return check_health


def _make_runtime_version(user_cls: type) -> Callable[..., Any]:
    """Build the ``bioengine_runtime_version`` method installed on the class.

    Returns the artifact identity this replica *actually booted with*, read
    from the process env the worker set in the replica's runtime_env. The
    worker queries it (through the ProxyDeployment) to verify that a running
    replica loaded the requested version rather than stale in-memory code
    left over from a reused replica.
    """

    async def bioengine_runtime_version(self: Any) -> Dict[str, Optional[str]]:
        return {
            "artifact_id": os.environ.get("BIOENGINE_ARTIFACT_ID"),
            "version": os.environ.get("BIOENGINE_ARTIFACT_VERSION"),
            "app_source_uri": os.environ.get("BIOENGINE_APP_SOURCE_URI"),
        }

    return bioengine_runtime_version


async def _invoke_lifecycle(
    instance: Any, method_name: str, label: str, logger: logging.Logger
) -> None:
    """Invoke a lifecycle hook (``async_init`` / ``smoke_test``) by name."""
    hook = getattr(instance, method_name)
    logger.info(f"⚡ Running @bioengine.{label} ({method_name})...")
    start = time.time()
    try:
        if inspect.iscoroutinefunction(hook):
            await hook()
        else:
            await asyncio.to_thread(hook)
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"❌ @bioengine.{label} failed after {elapsed:.2f}s: {e}")
        raise
    elapsed = time.time() - start
    logger.info(f"✅ @bioengine.{label} done in {elapsed:.2f}s")


def wrap_init(
    user_cls: type, orig_init: Callable[..., None]
) -> Callable[..., None]:
    """Wrap the user class's ``__init__`` so framework setup runs first."""

    @wraps(orig_init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        _setup_replica(self)
        kwargs = _wrap_runtime_handles(user_cls, kwargs)
        orig_init(self, *args, **kwargs)

    return __init__
