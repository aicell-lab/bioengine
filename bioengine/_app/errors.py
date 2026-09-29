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


class DataServerError(Exception):
    """The data server answered a dataset request with a non-2xx status.

    Raised by ``bioengine.datasets`` in place of ``httpx.HTTPStatusError``.
    The httpx type cannot be reconstructed from its own ``args`` — its
    ``__init__`` requires keyword-only ``request`` and ``response`` — so it
    does not survive the pickle round trip Ray does when a replica method
    raises across the proxy boundary, and the caller loses the status code,
    the URL and the body. This one carries all three as plain values in
    ``args``, which is what makes it reconstructible.

    Attributes:
        status_code: HTTP status the data server returned.
        url: The URL that was requested.
        body: The response body, truncated to an excerpt.
    """

    def __init__(self, message: str, status_code: int, url: str, body: str) -> None:
        super().__init__(message, status_code, url, body)
        self.status_code = status_code
        self.url = url
        self.body = body

    def __str__(self) -> str:
        return self.args[0]
