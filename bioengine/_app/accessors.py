"""Lazy accessors exposed as ``bioengine.datasets`` / ``bioengine.logger``.

These are reached via the PEP 562 ``__getattr__`` on the ``bioengine`` package.
Each Ray Serve replica is its own process, so a process-global cache is the
right scope: one ``BioEngineDatasets`` (with its own ``httpx.AsyncClient`` and
chunk cache) per replica, never shared across replicas.

Both accessors are deliberately lazy. Importing ``bioengine`` must not connect
to anything — the BioEngine worker's introspection Ray task imports the user's
package before any data server URL is known, and a non-lazy accessor would
either blow up or silently bind to the wrong endpoint.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

from bioengine._app.errors import MissingDataServerError

if TYPE_CHECKING:
    from bioengine.datasets import BioEngineDatasets


_datasets_singleton: Optional["BioEngineDatasets"] = None


def _get_datasets() -> "BioEngineDatasets":
    """Return the process-local ``BioEngineDatasets`` instance.

    Constructed on first call from these environment variables, which the
    BioEngine worker sets in every replica's runtime_env:

    * ``BIOENGINE_DATA_SERVER_URL`` — the data server URL (``"auto"`` means
      "discover via Hypha service lookup"). If unset, the accessor raises
      ``MissingDataServerError`` rather than silently binding to the wrong
      endpoint.
    * ``HYPHA_TOKEN`` — optional bearer token for authenticated dataset access.
    """
    global _datasets_singleton
    if _datasets_singleton is not None:
        return _datasets_singleton

    data_server_url = os.environ.get("BIOENGINE_DATA_SERVER_URL")
    if not data_server_url:
        raise MissingDataServerError(
            "bioengine.datasets was accessed before BIOENGINE_DATA_SERVER_URL "
            "was set. This usually means user code touched bioengine.datasets "
            "at module import time (e.g. at class-body scope) — that runs in "
            "the BioEngine worker's introspection Ray task, which deliberately "
            "does not configure a data server. Move dataset access inside an "
            "instance method (@bioengine.async_init, @bioengine.method, ...)."
        )

    from bioengine.datasets import BioEngineDatasets

    _datasets_singleton = BioEngineDatasets(
        data_server_url=data_server_url,
        hypha_token=os.environ.get("HYPHA_TOKEN"),
        logger=_get_logger(),
    )
    return _datasets_singleton


def _get_logger() -> logging.Logger:
    """Return the logger for the current process.

    Inside a Ray Serve replica the appropriate logger is ``ray.serve`` —
    Ray installs handlers that route logs into the replica log files.
    Elsewhere (notably the worker's introspection Ray task) we fall back to
    ``bioengine.app``, configured on first use so the fallback is merely
    degraded rather than silent: unconfigured, it inherits the root level of
    ``WARNING`` and has no handler, so ``INFO`` records are dropped outright.

    Deliberately not cached. ``BIOENGINE_REPLICA`` is only true once the
    replica's environment is in place, and a cached fallback would outlive it.
    """
    if os.environ.get("BIOENGINE_REPLICA") == "1":
        return logging.getLogger("ray.serve")

    logger = logging.getLogger("bioengine.app")
    if not logger.handlers:
        from bioengine.utils import create_logger

        logger = create_logger("bioengine.app")
    return logger


class _LazyLogger:
    """Forwards every attribute to ``_get_logger()`` at the moment it is used.

    ``bioengine.logger`` cannot hand out a concrete ``Logger``: Ray ships a
    deployment class to its replica *by value*, so the user's module is imported
    in the build task and the replica never re-imports it. A module-scope
    ``logger = bioengine.logger`` would therefore keep the build task's logger,
    whose handler writes into a process that is gone by the time the replica
    logs — the records vanish. Resolving per call makes the placement of that
    assignment irrelevant.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(_get_logger(), name)

    def __reduce__(self):
        # Must pickle as the proxy; resolving here would re-freeze the logger.
        return (_LazyLogger, ())

    def __repr__(self) -> str:
        return f"<bioengine.logger → {_get_logger().name}>"


def _reset_for_tests() -> None:
    """Drop the cached datasets so tests can re-init under different env vars."""
    global _datasets_singleton
    _datasets_singleton = None
