"""The worker re-resolves its artifact-manager proxy instead of wedging forever.

A hypha_rpc service proxy is pinned to one *client instance* of the remote
service. The artifact manager can re-register under a new client id with nothing
having restarted — observed at KTH, hypha-server at restartCount 0 and two days
old — and hypha_rpc's websocket reconnect does not re-resolve cached proxies. The
worker resolves this one once in ``initialize()`` and holds it for the process
lifetime, so every artifact-backed call (``deploy_app``, ``list_apps``,
``get_app_manifest``) then hung to timeout while every local call stayed green.
Nothing recovered it: ``auto_redeploy`` needs the artifact read that is broken,
and ``run_code`` executes in Ray tasks so it cannot reach the in-process handle.
model-runner was down ~62 minutes and only a pod restart cleared it (#0074).

The retry rule is the delicate part, and it is not "retry transport errors":

* a **send-side** failure provably never reached the server, so retrying it is
  safe even for a write;
* a **timeout or mid-call disconnect** may already have executed, so retrying
  would risk double-running a ``create`` or ``commit``. Those re-raise, and only
  the handle is refreshed — one visible failure instead of an endless outage.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from bioengine.apps.manager import (
    _ReconnectingArtifactManager,
    _stale_proxy_kind,
)

logger = logging.getLogger("test-artifact-manager")


class _Proxy:
    """One resolved client instance of the artifact manager."""

    def __init__(self, name: str, fails_with: BaseException | None = None):
        self.name = name
        self._fails_with = fails_with
        self.calls: list[tuple] = []

    async def read(self, artifact_id, **kwargs):
        self.calls.append(("read", artifact_id))
        if self._fails_with is not None:
            raise self._fails_with
        return {"id": artifact_id, "served_by": self.name}

    async def commit(self, artifact_id, **kwargs):
        self.calls.append(("commit", artifact_id))
        if self._fails_with is not None:
            raise self._fails_with
        return {"id": artifact_id, "served_by": self.name}


class _Server:
    """Hands out a fresh proxy each time the service is resolved."""

    def __init__(self, *proxies: _Proxy):
        self._proxies = list(proxies)
        self.resolutions = 0

    async def get_service(self, service_id):
        assert service_id == "public/artifact-manager"
        self.resolutions += 1
        return self._proxies.pop(0)


def _wrap(server: _Server, proxy: _Proxy) -> _ReconnectingArtifactManager:
    return _ReconnectingArtifactManager(server=server, proxy=proxy, logger=logger)


# ===== the classifier =====


def test_send_side_failures_are_known_to_have_never_landed():
    assert (
        _stale_proxy_kind(RuntimeError("Failed to send the request when calling method"))
        == "never_sent"
    )
    assert (
        _stale_proxy_kind(RuntimeError("WebSocket reconnection timed out"))
        == "never_sent"
    )


def test_failures_that_may_have_landed_are_classified_separately():
    assert _stale_proxy_kind(asyncio.TimeoutError()) == "maybe_sent"
    assert (
        _stale_proxy_kind(ConnectionError("Client disconnected: ws/abc")) == "maybe_sent"
    )
    assert (
        _stale_proxy_kind(RuntimeError("Method call timed out: ws/abc:services.x.read"))
        == "maybe_sent"
    )


def test_ordinary_errors_are_not_stale_proxy_failures():
    assert _stale_proxy_kind(ValueError("artifact not found")) is None
    assert _stale_proxy_kind(PermissionError("denied")) is None


# ===== pass-through =====


@pytest.mark.asyncio
async def test_a_healthy_call_is_forwarded_untouched():
    proxy = _Proxy("first")
    server = _Server()
    am = _wrap(server, proxy)

    result = await am.read("ws/my-app")

    assert result["served_by"] == "first"
    assert server.resolutions == 0, "a healthy call must not re-resolve"


@pytest.mark.asyncio
async def test_an_application_error_propagates_without_re_resolving():
    """Only transport failures mean the handle is stale."""
    proxy = _Proxy("first", fails_with=ValueError("artifact not found"))
    server = _Server()
    am = _wrap(server, proxy)

    with pytest.raises(ValueError):
        await am.read("ws/missing")

    assert server.resolutions == 0


# ===== the repair =====


@pytest.mark.asyncio
async def test_a_send_side_failure_re_resolves_and_retries():
    dead = _Proxy("dead", fails_with=RuntimeError("Failed to send the request"))
    fresh = _Proxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    result = await am.read("ws/my-app")

    assert server.resolutions == 1
    assert result["served_by"] == "fresh", "the retry must use the new proxy"


@pytest.mark.asyncio
async def test_a_timeout_refreshes_the_handle_but_does_not_retry():
    """#0074's own signature, and the double-execute guard.

    A dead client id reached through a cached proxy hangs rather than erroring,
    so this is the path that mattered in production. The call may already have
    executed at the far end, so it must not be replayed.
    """
    dead = _Proxy("dead", fails_with=asyncio.TimeoutError())
    fresh = _Proxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    with pytest.raises(asyncio.TimeoutError):
        await am.commit("ws/my-app")

    assert server.resolutions == 1, "the handle must still be refreshed"
    assert fresh.calls == [], "a call that may have landed must not be replayed"


@pytest.mark.asyncio
async def test_the_next_call_after_a_timeout_succeeds():
    """The outage is one failed call, not an indefinite wedge."""
    dead = _Proxy("dead", fails_with=asyncio.TimeoutError())
    fresh = _Proxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    with pytest.raises(asyncio.TimeoutError):
        await am.commit("ws/my-app")

    result = await am.read("ws/my-app")
    assert result["served_by"] == "fresh"
    assert server.resolutions == 1, "the handle was already replaced"


@pytest.mark.asyncio
async def test_concurrent_failures_re_resolve_the_service_once():
    """Every in-flight call fails against the same dead proxy.

    Without the generation guard each one would resolve its own replacement,
    turning a single eviction into a burst of ``get_service`` calls.
    """
    dead = _Proxy("dead", fails_with=RuntimeError("Failed to send the request"))
    fresh = _Proxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    results = await asyncio.gather(*(am.read(f"ws/app-{i}") for i in range(5)))

    assert server.resolutions == 1
    assert all(r["served_by"] == "fresh" for r in results)


@pytest.mark.asyncio
async def test_a_failed_re_resolve_surfaces_rather_than_being_swallowed():
    """If the service genuinely cannot be resolved, say so."""

    class _BrokenServer:
        resolutions = 0

        async def get_service(self, service_id):
            raise RuntimeError("Hypha is returning 500")

    dead = _Proxy("dead", fails_with=RuntimeError("Failed to send the request"))
    am = _wrap(_BrokenServer(), dead)

    with pytest.raises(RuntimeError, match="500"):
        await am.read("ws/my-app")


# ===== the fix reaches the cached handle =====


def test_the_manager_wraps_the_proxy_it_caches():
    """The wrapper is useless unless the manager actually installs it.

    This is the step that makes the fix reach ``AppBuilder`` and every
    ``artifact_utils`` helper for free, since they are all handed this object.
    """
    import inspect

    from bioengine.apps.manager import AppsManager

    source = inspect.getsource(AppsManager.complete_initialization)

    assert "_ReconnectingArtifactManager(" in source
    assert "self.artifact_manager = _ReconnectingArtifactManager(" in source
