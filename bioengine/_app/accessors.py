"""Lazy process-local singletons exposed as ``bioengine.datasets`` / ``bioengine.logger``.

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
from typing import TYPE_CHECKING, Optional

from bioengine._app.errors import MissingDataServerError

if TYPE_CHECKING:
    from bioengine.datasets import BioEngineDatasets


_datasets_singleton: Optional["BioEngineDatasets"] = None
_logger_singleton: Optional[logging.Logger] = None


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
    """Return the process-local logger.

    Inside a Ray Serve replica the appropriate logger is ``ray.serve`` —
    Ray installs handlers that route logs into the replica log files.
    Elsewhere we fall back to a plain ``bioengine.app`` logger.
    """
    global _logger_singleton
    if _logger_singleton is not None:
        return _logger_singleton

    if os.environ.get("BIOENGINE_REPLICA") == "1":
        _logger_singleton = logging.getLogger("ray.serve")
    else:
        _logger_singleton = logging.getLogger("bioengine.app")
    return _logger_singleton


def _reset_for_tests() -> None:
    """Drop cached singletons so tests can re-init under different env vars."""
    global _datasets_singleton, _logger_singleton
    _datasets_singleton = None
    _logger_singleton = None
