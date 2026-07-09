"""Replica-side authoring API for BioEngine apps.

This subpackage holds the user-facing decorators and the framework code that
runs inside a Ray Serve replica. The BioEngine worker (``bioengine/apps/``)
does not import any of this — it only knows how to ship the user's package
into a Ray ``runtime_env`` and call two introspection tasks defined in
``bioengine._app.bootstrap``.

The leading underscore is deliberate: users address these symbols as
``bioengine.app`` / ``bioengine.method`` (re-exported via PEP 562
``__getattr__`` in ``bioengine/__init__.py``), and naming the subpackage
``app`` would shadow ``bioengine.app`` (the decorator) once Python's import
machinery set the submodule as an attribute on the package.
"""

from bioengine._app.decorators import (
    app,
    async_init,
    cached,
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

__all__ = [
    "app",
    "async_init",
    "cached",
    "health_check",
    "method",
    "multiplexed",
    "smoke_test",
    "BioEngineRuntimeHandle",
    "BioEngineUserError",
    "CompositionCycleError",
    "MissingDataServerError",
    "ReservedMethodNameError",
]
