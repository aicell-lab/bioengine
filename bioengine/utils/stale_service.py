from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Iterable, Optional

if TYPE_CHECKING:
    from hypha_rpc.rpc import RemoteService


# A hypha_rpc service proxy is pinned to ONE client instance of the remote
# service. That instance can change client id at any time — the artifact manager
# re-registered under a new one with nothing restarted — and hypha_rpc's own
# websocket reconnect does not re-resolve cached proxies, so every later call
# addresses a client that no longer exists and hangs to timeout. Split by whether
# the call can have reached the server: a send-side failure provably did not, so
# retrying it is safe even for a write; a timeout or a mid-call disconnect may
# already have landed, so retrying those could double-execute a create or commit.
#
# hypha_rpc 0.21.40 does ship a stale-service retry, but it is installed by
# ``RPC.get_remote_service`` and so reaches only the workspace-manager proxy and
# services fetched by ``client_id:service_id`` URI. A service resolved by name
# through ``server.get_service`` arrives off the ordinary decode path as a plain
# ObjectProxy with no retry on it — every handle below included.
_PROXY_NEVER_SENT_MARKERS = (
    "failed to send the request",
    "websocket reconnection timed out",
)
_PROXY_MAYBE_SENT_MARKERS = (
    "client disconnected",
    "method call timed out",
    "service not found",
)


def stale_service_kind(exc: BaseException) -> Optional[str]:
    """``"never_sent"``, ``"maybe_sent"``, or ``None`` if not a stale-proxy failure."""
    # Deferred because this module is reachable from ``bioengine.utils``, which
    # has to stay importable on the Ray client server — a pod upstream of any
    # runtime_env that is not guaranteed to ship hypha_rpc (see the import note
    # in bioengine/_app/decorators.py).
    from hypha_rpc.rpc import RemoteException

    if isinstance(exc, RemoteException):
        # Its message is ``"RemoteError:" + value + "\n" + remote traceback``, and
        # the server runs hypha_rpc too, so any marker below can appear in that
        # traceback. Arriving at all proves the remote handler ran.
        return None
    message = str(exc).lower()
    if any(marker in message for marker in _PROXY_NEVER_SENT_MARKERS):
        return "never_sent"
    if isinstance(exc, asyncio.TimeoutError) or any(
        marker in message for marker in _PROXY_MAYBE_SENT_MARKERS
    ):
        return "maybe_sent"
    return None


class ReconnectingService:
    """A name-resolved hypha service handle, re-resolved when its client id goes stale.

    Built for the handle an app resolves once in ``async_init`` and caches on an
    attribute. Nothing else recovers it: a cached proxy pinned to a client id the
    server no longer knows is not ``None``, so a health check that tests for
    ``None`` reports healthy, and ``async_init`` runs once per replica and is
    never re-run in-process — so every call hangs to the full method timeout
    until the replica itself dies.

    Two axes decide whether the call is repeated against the fresh proxy:

    * a call that provably never left the process is always safe to repeat;
    * a call that may already have executed is repeated only if the caller
      named it in ``repeatable``.

    Anything else re-raises, so the caller sees one failure rather than an
    indefinite outage and the next call uses the refreshed handle. The retry is
    one attempt, not a loop: a genuinely absent service costs two timeouts and
    then surfaces.
    """

    def __init__(
        self,
        server: "RemoteService",
        service_id: str,
        proxy: Any,
        *,
        repeatable: Iterable[str] = (),
        logger: Optional[logging.Logger] = None,
    ):
        self._server = server
        self._service_id = service_id
        self._proxy = proxy
        self._repeatable = frozenset(repeatable)
        self._logger = logger or logging.getLogger(__name__)
        self._generation = 0
        self._lock = asyncio.Lock()
        self._resolved = asyncio.Event()
        self._resolved.set()

    async def _re_resolve(self, seen_generation: int) -> None:
        async with self._lock:
            # Concurrent callers all fail against the same dead proxy; only the
            # first needs to replace it.
            if self._generation != seen_generation:
                return
            self._resolved.clear()
            try:
                self._proxy = await self._server.get_service(self._service_id)
                self._generation += 1
            finally:
                self._resolved.set()
            self._logger.info(f"Re-resolved the '{self._service_id}' service proxy.")

    def __getattr__(self, name: str):
        attribute = getattr(self._proxy, name)
        if not callable(attribute):
            return attribute

        async def call(*args, **kwargs):
            # Dispatching while a replacement is being fetched would spend a
            # full method timeout on the handle already known to be dead.
            await self._resolved.wait()
            seen_generation = self._generation
            try:
                return await getattr(self._proxy, name)(*args, **kwargs)
            except Exception as exc:
                kind = stale_service_kind(exc)
                if kind is None:
                    raise
                self._logger.warning(
                    f"Call '{name}' on '{self._service_id}' failed against a "
                    f"stale service proxy: {exc}"
                )
                await self._re_resolve(seen_generation)
                if kind != "never_sent" and name not in self._repeatable:
                    raise
                return await getattr(self._proxy, name)(*args, **kwargs)

        return call


async def reconnecting_service(
    server: "RemoteService",
    service_id: str,
    *,
    repeatable: Iterable[str] = (),
    logger: Optional[logging.Logger] = None,
) -> ReconnectingService:
    """Resolve ``service_id`` by name and wrap it so a stale client id re-resolves.

    Drop-in for ``await server.get_service(service_id)``. ``repeatable`` names the
    methods safe to call twice when the first attempt may already have landed —
    for ``public/s3-storage`` that is ``("put_file", "get_file")``, since both
    only mint a presigned URL.
    """
    proxy = await server.get_service(service_id)
    return ReconnectingService(
        server, service_id, proxy, repeatable=repeatable, logger=logger
    )
