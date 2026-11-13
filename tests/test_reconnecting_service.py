"""An app's name-resolved service handle re-resolves instead of hanging forever.

The worker already protects its own artifact-manager handle
(``tests/apps/test_artifact_manager_reconnect.py``). An app resolving a service by
name in ``async_init`` has the same handle and none of that protection:

* the proxy is pinned to one *client instance*, which can re-register under a new
  client id with nothing having restarted;
* hypha_rpc's websocket reconnect does not re-resolve cached proxies, so every
  later call addresses a client that no longer exists and hangs to the full
  method timeout;
* a health check that tests ``handle is None`` still passes — the handle is not
  ``None``, only dead — so Ray Serve keeps the replica and ``async_init``, which
  runs once per replica and is never re-run in-process, never resolves it again.

``test_a_plain_cached_handle_never_recovers`` is the control: it is the fault as
apps hit it today, with no wrapper, and it passes either way. The rest exercise
the wrapper, so this module cannot even import on ``origin/main``.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from bioengine.utils.stale_service import ReconnectingService, reconnecting_service

logger = logging.getLogger("test-reconnecting-service")

# What the live fault produced: the hypha server answered a call against a client
# id it no longer knew, and a call to the dead client timed out.
_SERVICE_GONE = KeyError("Service not found: ws/dead-client-04319715:s3-storage@*")
_CALL_TIMED_OUT = RuntimeError(
    "Method call timed out: ws/dead-client-04319715:services.s3-storage.put_file"
)
_NEVER_SENT = RuntimeError("Failed to send the request when calling method")


class _S3Proxy:
    """One resolved client instance of ``public/s3-storage``.

    ``put_file``/``get_file`` mint a presigned URL and are safe to repeat;
    ``remove_file`` stands in for a call that is not.
    """

    id = "public/s3-storage"

    def __init__(self, name: str, fails_with: BaseException | None = None):
        self.name = name
        self._fails_with = fails_with
        self.calls: list[tuple] = []

    async def _answer(self, method, file_path):
        await asyncio.sleep(0)
        self.calls.append((method, file_path))
        if self._fails_with is not None:
            raise self._fails_with
        return f"https://{self.name}/{file_path}?signed"

    async def put_file(self, file_path, ttl=None):
        return await self._answer("put_file", file_path)

    async def get_file(self, file_path, use_proxy=None):
        return await self._answer("get_file", file_path)

    async def remove_file(self, file_path):
        return await self._answer("remove_file", file_path)


class _Server:
    """Hands out the next proxy each time the service is resolved by name."""

    def __init__(self, *proxies: _S3Proxy, fail_resolve_times: int = 0):
        self._proxies = list(proxies)
        self._fail_resolve_times = fail_resolve_times
        self.resolutions = 0

    async def get_service(self, service_id):
        assert service_id == "public/s3-storage"
        await asyncio.sleep(0)
        self.resolutions += 1
        if self._fail_resolve_times > 0:
            self._fail_resolve_times -= 1
            raise ConnectionRefusedError("hypha server unreachable")
        return self._proxies.pop(0)


def _wrap(server, proxy: _S3Proxy) -> ReconnectingService:
    return ReconnectingService(
        server,
        "public/s3-storage",
        proxy,
        repeatable=("put_file", "get_file"),
        logger=logger,
    )


# ===== the control: the fault as it exists today =====


@pytest.mark.asyncio
async def test_a_plain_cached_handle_never_recovers():
    """No wrapper: every call keeps addressing the client id that is gone.

    This is the arm that must pass on both branches — if it ever goes green by
    recovering, the harness has stopped modelling the fault.
    """
    dead = _S3Proxy("dead-client", fails_with=_SERVICE_GONE)
    server = _Server(_S3Proxy("fresh-client"))

    for _ in range(3):
        with pytest.raises(KeyError):
            await dead.put_file("runs/1/weights.pt")

    assert len(dead.calls) == 3
    assert server.resolutions == 0, "nothing re-resolves an app's cached handle"


# ===== pass-through =====


@pytest.mark.asyncio
async def test_a_healthy_call_is_forwarded_untouched():
    proxy = _S3Proxy("first")
    server = _Server()
    s3 = _wrap(server, proxy)

    url = await s3.put_file("runs/1/weights.pt")

    assert url == "https://first/runs/1/weights.pt?signed"
    assert server.resolutions == 0, "a healthy call must not re-resolve"


@pytest.mark.asyncio
async def test_an_application_error_propagates_without_re_resolving():
    proxy = _S3Proxy("first", fails_with=PermissionError("not your bucket"))
    server = _Server()
    s3 = _wrap(server, proxy)

    with pytest.raises(PermissionError):
        await s3.get_file("runs/1/weights.pt")

    assert server.resolutions == 0


@pytest.mark.asyncio
async def test_a_non_callable_attribute_is_read_off_the_proxy():
    proxy = _S3Proxy("first")
    s3 = _wrap(_Server(), proxy)

    assert s3.id == "public/s3-storage"


# ===== the fix =====


@pytest.mark.asyncio
async def test_a_stale_client_id_is_re_resolved_and_the_call_retried():
    dead = _S3Proxy("dead-client", fails_with=_SERVICE_GONE)
    server = _Server(_S3Proxy("fresh-client"))
    s3 = _wrap(server, dead)

    url = await s3.put_file("runs/1/weights.pt")

    assert url == "https://fresh-client/runs/1/weights.pt?signed"
    assert server.resolutions == 1


@pytest.mark.asyncio
async def test_a_timed_out_call_is_retried_for_a_repeatable_method():
    """The call that hung in the live fault: ``put_file``, timing out.

    It may already have landed, and minting a second presigned URL for the same
    path has no effect beyond the first, so it is named in ``repeatable``.
    """
    dead = _S3Proxy("dead-client", fails_with=_CALL_TIMED_OUT)
    server = _Server(_S3Proxy("fresh-client"))
    s3 = _wrap(server, dead)

    url = await s3.get_file("runs/1/weights.pt")

    assert url == "https://fresh-client/runs/1/weights.pt?signed"


@pytest.mark.asyncio
async def test_a_method_that_may_have_landed_is_not_replayed_but_the_handle_heals():
    """``remove_file`` is not repeatable: one visible failure, then recovery."""
    dead = _S3Proxy("dead-client", fails_with=_CALL_TIMED_OUT)
    fresh = _S3Proxy("fresh-client")
    server = _Server(fresh)
    s3 = _wrap(server, dead)

    with pytest.raises(RuntimeError):
        await s3.remove_file("runs/1/weights.pt")

    assert server.resolutions == 1, "the handle is refreshed even when not retried"
    assert fresh.calls == [], "the unsafe call is not replayed"
    assert await s3.put_file("runs/2/weights.pt") == (
        "https://fresh-client/runs/2/weights.pt?signed"
    )


@pytest.mark.asyncio
async def test_a_call_that_never_left_the_process_is_retried_whatever_it_is():
    """A send-side failure provably did not execute, so even ``remove_file`` is safe."""
    dead = _S3Proxy("dead-client", fails_with=_NEVER_SENT)
    server = _Server(_S3Proxy("fresh-client"))
    s3 = _wrap(server, dead)

    assert await s3.remove_file("runs/1/weights.pt") == (
        "https://fresh-client/runs/1/weights.pt?signed"
    )


@pytest.mark.asyncio
async def test_concurrent_callers_re_resolve_once():
    """N calls in flight all fail against the same dead proxy."""
    dead = _S3Proxy("dead-client", fails_with=_SERVICE_GONE)
    server = _Server(_S3Proxy("fresh-client"))
    s3 = _wrap(server, dead)

    urls = await asyncio.gather(
        s3.put_file("runs/1/a.pt"),
        s3.put_file("runs/1/b.pt"),
        s3.get_file("runs/1/c.pt"),
    )

    assert server.resolutions == 1
    assert all(url.startswith("https://fresh-client/") for url in urls)


@pytest.mark.asyncio
async def test_a_failed_re_resolve_surfaces_and_the_next_call_tries_again():
    """A genuinely absent service fails loudly rather than looping."""
    dead = _S3Proxy("dead-client", fails_with=_SERVICE_GONE)
    server = _Server(_S3Proxy("fresh-client"), fail_resolve_times=1)
    s3 = _wrap(server, dead)

    with pytest.raises(ConnectionRefusedError):
        await s3.put_file("runs/1/weights.pt")

    assert await s3.put_file("runs/2/weights.pt") == (
        "https://fresh-client/runs/2/weights.pt?signed"
    )
    assert server.resolutions == 2


@pytest.mark.asyncio
async def test_the_factory_resolves_by_name_and_wraps_the_result():
    fresh = _S3Proxy("fresh-client")
    server = _Server(_S3Proxy("dead-client", fails_with=_SERVICE_GONE), fresh)

    s3 = await reconnecting_service(
        server, "public/s3-storage", repeatable=("put_file",), logger=logger
    )
    url = await s3.put_file("runs/1/weights.pt")

    assert url == "https://fresh-client/runs/1/weights.pt?signed"
    assert server.resolutions == 2, "one resolve to construct, one to recover"
