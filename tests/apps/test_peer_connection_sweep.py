"""Pin the WebRTC peer-connection sweep for long-running deployments.

The state-change handler in ``_on_webrtc_init`` cleans up when a peer
emits ``closed`` or ``failed``. Peers that vanish without either event
(browser tab killed, mobile network cut, kernel SIGKILL of the client)
would otherwise accumulate ghost RTCPeerConnection objects in
``_active_peer_connections`` indefinitely. Each one holds sockets, ICE
candidates, event handlers, and a queue of DTLS keying material — over
weeks of uptime that's a real leak.

``_sweep_stale_peer_connections`` runs from ``check_health`` and closes
peers that have been stuck too long in a non-``connected`` transient
state. These tests pin the timeouts, the state matrix, and the
integration hook into ``check_health``.
"""
from __future__ import annotations

import asyncio
import inspect
import time
import types

import pytest

from bioengine.apps import proxy_deployment as pd_module

_ProxyDeployment = pd_module.ProxyDeployment.func_or_class


class _StubPeerConnection:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _make_instance() -> _ProxyDeployment:
    """Skip ``__init__`` (it wants ~15 constructor args and a Ray Serve
    context) and stamp on just the attributes the sweep touches."""
    obj = _ProxyDeployment.__new__(_ProxyDeployment)
    obj.application_id = "test-app"
    obj._active_peer_connections = {}
    return obj


def _add_peer(obj: _ProxyDeployment, state: str, age_seconds: float) -> str:
    now = time.time()
    connection_id = f"peer-{state}-{int(age_seconds)}"
    obj._active_peer_connections[connection_id] = {
        "peer_connection": _StubPeerConnection(),
        "created_at": now - age_seconds,
        "state_changed_at": now - age_seconds,
        "state": state,
    }
    return connection_id


def test_handshake_timeout_is_at_least_a_minute() -> None:
    """Two minutes was tuned to give slow / captive-portal clients room
    to complete the ICE handshake. A future contributor may shorten it —
    guard against dropping it below the median mobile-network handshake
    time (~60 s)."""
    assert _ProxyDeployment._PEER_STUCK_HANDSHAKE_TIMEOUT_SECONDS >= 60


def test_disconnected_timeout_is_at_least_a_minute() -> None:
    """A disconnected peer that recovers usually does so within seconds
    via ICE restart. Five minutes is the conservative cutoff — well past
    every recovery we've ever observed. Guard against too-eager sweeps."""
    assert _ProxyDeployment._PEER_STUCK_DISCONNECTED_TIMEOUT_SECONDS >= 60


@pytest.mark.asyncio
async def test_sweeps_stuck_handshake() -> None:
    obj = _make_instance()
    cid = _add_peer(
        obj, "new", _ProxyDeployment._PEER_STUCK_HANDSHAKE_TIMEOUT_SECONDS + 10
    )
    pc = obj._active_peer_connections[cid]["peer_connection"]
    await obj._sweep_stale_peer_connections()
    assert cid not in obj._active_peer_connections
    assert pc.closed is True


@pytest.mark.asyncio
async def test_sweeps_stuck_connecting() -> None:
    obj = _make_instance()
    cid = _add_peer(
        obj, "connecting",
        _ProxyDeployment._PEER_STUCK_HANDSHAKE_TIMEOUT_SECONDS + 10,
    )
    await obj._sweep_stale_peer_connections()
    assert cid not in obj._active_peer_connections


@pytest.mark.asyncio
async def test_sweeps_stuck_disconnected() -> None:
    obj = _make_instance()
    cid = _add_peer(
        obj, "disconnected",
        _ProxyDeployment._PEER_STUCK_DISCONNECTED_TIMEOUT_SECONDS + 10,
    )
    pc = obj._active_peer_connections[cid]["peer_connection"]
    await obj._sweep_stale_peer_connections()
    assert cid not in obj._active_peer_connections
    assert pc.closed is True


@pytest.mark.asyncio
async def test_does_not_sweep_connected_peers_no_matter_the_age() -> None:
    """The whole point is that long-lived healthy peers stay alive.
    Never close a ``connected`` peer, even if it's been up for years."""
    obj = _make_instance()
    cid = _add_peer(obj, "connected", 86400 * 30)  # 30 days
    pc = obj._active_peer_connections[cid]["peer_connection"]
    await obj._sweep_stale_peer_connections()
    assert cid in obj._active_peer_connections
    assert pc.closed is False


@pytest.mark.asyncio
async def test_does_not_sweep_recently_started_handshake() -> None:
    obj = _make_instance()
    cid = _add_peer(obj, "new", 5)  # 5s old, well under timeout
    await obj._sweep_stale_peer_connections()
    assert cid in obj._active_peer_connections


@pytest.mark.asyncio
async def test_empty_map_is_a_noop() -> None:
    obj = _make_instance()
    obj._active_peer_connections = {}
    await obj._sweep_stale_peer_connections()  # must not raise


def test_check_health_calls_sweep() -> None:
    """The sweep hook must live inside check_health — that's the only
    periodic entry point aiortc replicas expose. Missing this call means
    the sweep never runs."""
    src = inspect.getsource(_ProxyDeployment.check_health)
    assert "_sweep_stale_peer_connections" in src


def test_state_changed_at_is_set_on_new_peer() -> None:
    """The sweep uses ``state_changed_at`` to compute how long a peer has
    been stuck. New peer records must initialise it or the sweep would
    key off ``created_at`` — usually right, but wrong the moment the
    peer transitions and we want to reset the clock."""
    src = inspect.getsource(_ProxyDeployment._on_webrtc_init)
    assert '"state_changed_at": current_time' in src


def test_state_changed_at_is_updated_on_state_change() -> None:
    """When a peer transitions, ``state_changed_at`` must reset to now.
    Without this, a peer that briefly disconnects and then reconnects
    would be swept as soon as it hit the disconnected timeout, even if
    the disconnect happened only seconds before the sweep."""
    src = inspect.getsource(_ProxyDeployment._on_webrtc_init)
    assert '"state_changed_at"' in src
    assert "time.time()" in src
