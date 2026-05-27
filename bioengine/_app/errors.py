"""Errors raised by the BioEngine app authoring framework.

These distinguish *user-code* problems (bad manifest, missing import in the
runtime_env, accessing a singleton outside a replica, etc.) from internal
BioEngine bugs. The worker catches BioEngineUserError, strips traceback noise,
and surfaces the message verbatim in the deployment status.
"""

from __future__ import annotations


class BioEngineUserError(Exception):
    """Raised when user code or the manifest is malformed.

    The worker surfaces the message to the deploying user; internal
    BioEngine code is responsible for adding enough context that the user
    can act on it (e.g. "move this import into a method body").
    """


class ReservedMethodNameError(BioEngineUserError):
    """A class decorated with @bioengine.app defined a method whose name
    is reserved by the framework (e.g. ``check_health``)."""


class CompositionCycleError(BioEngineUserError):
    """The type-hint composition graph contains a cycle."""


class MissingDataServerError(BioEngineUserError):
    """``bioengine.datasets`` accessed before BIOENGINE_DATA_SERVER_URL was set.

    Typically means the user touched ``bioengine.datasets`` at module
    import time, which runs inside the worker's introspection task where
    the data server is intentionally not configured.
    """
