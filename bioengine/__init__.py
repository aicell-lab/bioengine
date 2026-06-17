"""BioEngine — execution and adaptation layer between curated bioimage AI and scalable compute.

The top-level ``bioengine`` namespace is the authoring surface for app
deployments. User code typically writes:

    import bioengine

    @bioengine.app(num_cpus=1, memory_mb=512, pip=["pandas"])
    class MyApp:
        @bioengine.async_init
        async def load(self): ...

        @bioengine.method
        async def predict(self, x): ...

All authoring symbols are resolved lazily via PEP 562 ``__getattr__`` so
that ``import bioengine`` stays cheap (no Ray import) and the heavy
imports only happen when the decorators are actually applied.
"""

from __future__ import annotations

import importlib.metadata as _md
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-checker-only re-exports so editors auto-complete bioengine.app etc.
    from bioengine._app.decorators import (
        app,
        async_init,
        health_check,
        method,
        multiplexed,
        smoke_test,
    )
    from bioengine._app.errors import (
        BioEngineUserError,
        CompositionCycleError,
        MissingDataServerError,
        ReservedMethodNameError,
    )
    from bioengine._app.runtime_handle import BioEngineRuntimeHandle


def _get_version() -> str | None:
    try:
        return _md.metadata("bioengine")["Version"]
    except Exception:
        print("Could not get version from package metadata. Is the package installed?")
        return None


__version__ = _get_version()


def _install_replica_bootstrap_finder() -> None:
    """Install a ``sys.meta_path`` finder that materialises user source
    on the first import Ray Serve issues against an unfindable module.

    Why a meta_path finder instead of a side effect on ``import bioengine``?

    Ray Serve replicas boot like this:

      ServeReplica.__init__
        → cloudpickle.loads(serialized_deployment_def)
            → importlib.import_module(entry_module_name)

    For an app whose entry is ``main:CellposeFinetune`` cloudpickle's
    very first move is ``import main`` — and at that point ``main`` is
    not on ``sys.path`` because the bootstrap hasn't run. Hooking
    ``import bioengine`` only worked for model-runner because its
    ``entry.py`` happens to ``import bioengine`` at line 31 (before any
    user-submodule import); cellpose's monolithic ``main.py`` doesn't
    hit ``import bioengine`` until after its own ``from training import
    …``, so it dies first.

    The finder is appended to ``sys.meta_path`` so it only fires when no
    other finder resolved the name — i.e. exactly the "module is not on
    sys.path yet" case. On the first such miss in a replica, it:

      1. Sets a sentinel so the bootstrap can't recurse if any of its
         own imports also miss.
      2. Runs ``setup_replica_environment()`` which calls
         ``_ensure_source`` (Ray-GCS download via
         ``BIOENGINE_APP_SOURCE_URI``) and ``sys.path.insert(0, source/)``.
      3. Delegates back to ``importlib.machinery.PathFinder`` to resolve
         the name against the now-populated ``sys.path`` — so the import
         that triggered us succeeds without the caller knowing.

    Guards: only active when ``BIOENGINE_APP_DIR`` is set (worker
    processes never have it); idempotent via a threading lock on the
    sentinel; failures fall through to the standard ModuleNotFoundError
    so users see the same error they would without the finder.
    """
    import os as _os
    import sys as _sys
    import threading as _threading

    if not _os.environ.get("BIOENGINE_APP_DIR"):
        return

    class _BioEngineImportBootstrap:
        _lock = _threading.Lock()
        _fired = False

        def find_spec(self, fullname, path, target=None):
            if self._fired:
                return None
            # bioengine and stdlib are never resolved here — by the time
            # this finder is consulted, earlier finders have already
            # answered for any module they own. Skip our own namespace
            # explicitly so we can't recurse on ``from bioengine._app
            # .replica_init import …`` below.
            if fullname.startswith("bioengine") or fullname.startswith("_"):
                return None
            with self._lock:
                if self._fired:
                    return None
                self._fired = True
                try:
                    from bioengine._app.replica_init import (
                        setup_replica_environment,
                    )

                    setup_replica_environment()
                except Exception:
                    import traceback

                    print(
                        "BioEngine: replica source bootstrap failed — user "
                        "source will not be on sys.path. Traceback:",
                        flush=True,
                    )
                    traceback.print_exc()
                    return None
            from importlib.machinery import PathFinder

            return PathFinder.find_spec(fullname, path, target)

    _sys.meta_path.append(_BioEngineImportBootstrap())


_install_replica_bootstrap_finder()


_LAZY_FROM_APP = {
    "app",
    "method",
    "async_init",
    "smoke_test",
    "health_check",
    "multiplexed",
    "BioEngineRuntimeHandle",
    "BioEngineUserError",
    "CompositionCycleError",
    "MissingDataServerError",
    "ReservedMethodNameError",
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute resolution for the ``bioengine`` package.

    Resolves decorator and error re-exports from ``bioengine._app`` lazily
    so that ``import bioengine`` stays cheap (no Ray import). The internal
    subpackage is ``_app`` (leading underscore) — naming it ``app`` would
    shadow ``bioengine.app`` (the decorator) once Python's import system
    set the submodule as an attribute on the package.

    ``bioengine.datasets`` triggers an import of the
    ``bioengine.datasets`` subpackage on first access; the subpackage's
    own ``__getattr__`` delegates instance-method calls to a lazy
    ``BioEngineDatasets`` singleton so ``bioengine.datasets.list_datasets()``
    Just Works without the user having to ``import bioengine.datasets``.
    Subsequent accesses bypass this hook because Python's import machinery
    has set the submodule as an attribute on the package.
    """
    if name == "datasets":
        import bioengine.datasets as _datasets_module

        return _datasets_module
    if name == "logger":
        from bioengine._app.accessors import _get_logger

        return _get_logger()
    if name in _LAZY_FROM_APP:
        import bioengine._app as _app_module

        return getattr(_app_module, name)
    raise AttributeError(f"module 'bioengine' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({"__version__", "datasets", "logger", *_LAZY_FROM_APP})
