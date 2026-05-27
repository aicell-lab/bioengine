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
