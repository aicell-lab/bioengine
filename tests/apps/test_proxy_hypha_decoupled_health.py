"""Pin that ProxyDeployment's health check is independent of Hypha.

``check_health`` used to make four Hypha round trips per tick — ``echo``,
a ``get_service`` resolve, then ``get_load`` and ``get_num_pcs`` routed
back into this replica's own service — and fail the replica when they
timed out. That put a *server-side* problem on the replica's condemnation
path: a slow Hypha killed healthy proxies, and each kill re-synced the
whole app through the same overloaded server.

The contract now:

* ``check_health`` reads local state only. The only Hypha-related reason
  it fails is a registration that can never succeed (bad workspace, bad
  client id, refused credentials) — surfaced via ``_registration_failure``.
* Connecting, registering and reconnecting live in a background
  maintenance task, where a failure costs a retry rather than a replica.
* Anything that might clear on its own is retried indefinitely, including
  the 'Client already exists and is active' collision that clears with
  Hypha's stale-client TTL.

The shape assertions read the source rather than driving Ray Serve's
health loop live; the behavioural ones drive ``_maintenance_tick``
directly, which is why it is factored out of the sleeping loop.
"""
from __future__ import annotations

import asyncio
import inspect
import time

import pytest

from bioengine.apps import proxy_deployment as pd_module

_ProxyCls = pd_module.ProxyDeployment.func_or_class


def _bare_proxy(**attrs):
    inst = object.__new__(_ProxyCls)
    inst.application_id = "app"
    inst.entry_deployment_ready = True
    inst.server = None
    inst.websocket_service_id = None
    inst._registration_lock = asyncio.Lock()
    inst._maintenance_task = None
    inst._connection_lost = False
    inst._registration_failure = None
    inst._probe_due_at = 0.0
    inst._next_register_at = 0.0
    for key, value in attrs.items():
        setattr(inst, key, value)
    return inst


class _FailingRegister:
    def __init__(self, exc: BaseException):
        self.exc = exc
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        raise self.exc


# ===== check_health touches no Hypha =====


def test_check_health_makes_no_hypha_calls() -> None:
    src = inspect.getsource(_ProxyCls.check_health)
    for call in ('echo("ping")', "get_service(", ".get_load(", ".get_num_pcs("):
        assert call not in src, (
            f"check_health must not call {call} — a slow Hypha would condemn a "
            f"replica that can still serve."
        )


def test_check_health_does_not_register() -> None:
    """Registration is retryable work; it belongs in the background task."""
    src = inspect.getsource(_ProxyCls.check_health)
    assert "_register_services" not in src
    assert "_ensure_maintenance_task" in src


def test_ping_tolerance_machinery_is_gone() -> None:
    """The consecutive-failure counter only existed to soften the Hypha
    calls that check_health no longer makes."""
    assert not hasattr(pd_module, "_MAX_CONSECUTIVE_PING_FAILURES")
    assert "_consecutive_ping_failures" not in inspect.getsource(_ProxyCls)


def test_probe_interval_is_far_above_the_health_check_period() -> None:
    """The one remaining Hypha round trip is background upkeep, not a
    per-tick tax: it must be rare compared to the 10s health check."""
    assert pd_module._REACHABILITY_PROBE_INTERVAL_S >= 60
    assert pd_module._MAINTENANCE_TICK_S < pd_module._REACHABILITY_PROBE_INTERVAL_S


def test_reregister_backoff_clears_the_stale_client_ttl() -> None:
    """A rebuild whose disconnect didn't land is refused with 'Client
    already exists and is active' until Hypha's TTL (~30-60s) sweeps it;
    retrying faster than that just burns attempts."""
    assert pd_module._REREGISTER_BACKOFF_S >= 30


# ===== permanent vs transient classification =====


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionAbortedError("Permission denied for workspace"),
        pd_module._PermanentRegistrationError("Workspace mismatch: 'a' != 'b'"),
        pd_module._PermanentRegistrationError("Client ID mismatch: 'a' != 'b'"),
        AssertionError("Connected to the wrong workspace: x, expected: y"),
        Exception("Connected to the wrong workspace: x, expected: y"),
    ],
)
def test_config_and_permission_errors_are_permanent(exc) -> None:
    assert _ProxyCls._is_permanent_registration_error(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionAbortedError("Client already exists and is active"),
        ConnectionRefusedError("[Errno 111] Connect call failed"),
        asyncio.TimeoutError(),
        OSError("network unreachable"),
        # hypha-rpc asserts this when a restarting server closes the socket
        # mid-handshake. Misreading it as permanent would kill healthy
        # replicas during exactly the outage this change protects against.
        AssertionError("Failed to connect to the server, no connection info obtained."),
        RuntimeError("Event loop is closed"),
    ],
)
def test_transient_errors_are_not_permanent(exc) -> None:
    assert _ProxyCls._is_permanent_registration_error(exc) is False


# ===== maintenance behaviour =====


@pytest.mark.asyncio
async def test_transient_registration_failure_keeps_replica_healthy() -> None:
    inst = _bare_proxy()
    inst._register_services = _FailingRegister(ConnectionRefusedError("refused"))

    await inst._maintenance_tick()

    assert inst._registration_failure is None


@pytest.mark.asyncio
async def test_permanent_registration_failure_fails_the_replica() -> None:
    inst = _bare_proxy()
    inst._register_services = _FailingRegister(
        ConnectionAbortedError("Permission denied")
    )

    await inst._maintenance_tick()

    assert isinstance(inst._registration_failure, ConnectionAbortedError)
    with pytest.raises(RuntimeError, match="failed permanently"):
        await _ProxyCls.check_health(inst)


@pytest.mark.asyncio
async def test_transient_failure_backs_off_instead_of_hammering() -> None:
    """A Hypha outage must not turn into a reconnect storm — that is the
    load pattern this whole change exists to remove."""
    inst = _bare_proxy()
    register = _FailingRegister(ConnectionRefusedError("refused"))
    inst._register_services = register

    await inst._maintenance_tick()
    await inst._maintenance_tick()
    await inst._maintenance_tick()

    assert register.calls == 1


@pytest.mark.asyncio
async def test_gate_closed_means_no_registration() -> None:
    """``_deregister_services`` re-gates by clearing ``entry_deployment_ready``;
    the maintenance task must respect that rather than re-registering a service
    for an app that cannot serve."""
    inst = _bare_proxy(entry_deployment_ready=False)
    register = _FailingRegister(ConnectionRefusedError("refused"))
    inst._register_services = register

    await inst._maintenance_tick()

    assert register.calls == 0


@pytest.mark.asyncio
async def test_unreachable_hypha_schedules_a_rebuild_without_raising() -> None:
    class _DeadServer:
        async def get_service_info(self, _sid):
            raise asyncio.TimeoutError()

    inst = _bare_proxy(server=_DeadServer(), websocket_service_id="ws")

    await inst._maintenance_tick()

    assert inst._connection_lost is True


@pytest.mark.asyncio
async def test_dropped_registration_on_a_live_socket_schedules_a_rebuild() -> None:
    """Observed live: a freeze long enough for Hypha to evict the client left
    the socket open, so ``echo`` kept succeeding for ten minutes while the
    service was unresolvable. The probe has to ask about the service."""

    class _AmnesiacServer:
        async def echo(self, _msg):
            return _msg

        async def get_service_info(self, sid):
            raise KeyError(f"Service not found: {sid}@*")

    inst = _bare_proxy(server=_AmnesiacServer(), websocket_service_id="ws")

    await inst._maintenance_tick()

    assert inst._connection_lost is True


def test_probe_asks_about_our_own_service() -> None:
    src = inspect.getsource(_ProxyCls._maintenance_tick)
    assert "get_service_info(self.websocket_service_id)" in src
    assert 'echo("ping")' not in src


@pytest.mark.asyncio
async def test_probe_is_skipped_until_the_interval_elapses() -> None:
    class _CountingServer:
        def __init__(self):
            self.pings = 0

        async def get_service_info(self, _sid):
            self.pings += 1

    server = _CountingServer()
    inst = _bare_proxy(server=server, websocket_service_id="ws")

    await inst._maintenance_tick()  # _probe_due_at is 0 -> probes once
    await inst._maintenance_tick()
    await inst._maintenance_tick()

    assert server.pings == 1


def test_disconnect_hook_is_synchronous_and_only_schedules_a_probe() -> None:
    """hypha-rpc calls ``on_disconnected`` without awaiting it, so a
    coroutine handler would never run.

    Observed on a live worker during a hypha.aicell.io wobble: the hook
    fires for drops the library then reconnects and re-registers itself.
    Rebuilding our client on that signal abandons the connection hypha-rpc
    is repairing, so the hook only brings the probe forward — the probe
    decides whether a rebuild is warranted.
    """
    assert not inspect.iscoroutinefunction(_ProxyCls._on_connection_lost)

    inst = _bare_proxy(server=object(), websocket_service_id="ws")
    inst._probe_due_at = time.time() + pd_module._REACHABILITY_PROBE_INTERVAL_S

    inst._on_connection_lost("closed")

    assert inst._connection_lost is False
    assert inst._probe_due_at <= time.time() + pd_module._RECONNECT_GRACE_S


def test_disconnect_hook_defers_a_probe_that_was_already_due() -> None:
    """Observed live: a 65s freeze left the probe overdue, so it ran 1ms
    after the drop, failed by definition, and rebuilt the client while
    hypha-rpc was reconnecting. A known-down socket resets the clock."""
    inst = _bare_proxy(server=object(), websocket_service_id="ws")
    inst._probe_due_at = 0.0

    inst._on_connection_lost("closed")

    assert inst._probe_due_at >= time.time() + pd_module._RECONNECT_GRACE_S - 1


def test_reconnect_grace_outlasts_the_libraries_own_retry() -> None:
    """hypha-rpc retries with its own backoff; probing before it has had a
    chance turns one drop into a rebuild storm."""
    assert pd_module._RECONNECT_GRACE_S >= 30


def test_register_services_attaches_the_disconnect_hook() -> None:
    src = inspect.getsource(_ProxyCls._register_services)
    assert "on_disconnected" in src


# ===== retained from the ping-flap fix (PR #135) =====


def test_register_services_calls_reset_before_connect() -> None:
    """_register_services must free the client_id before connect_to_server —
    otherwise a lingering registration causes 'Client already exists and is
    active' and the reconnect fails until Hypha's stale-client TTL fires."""
    import re

    src = inspect.getsource(_ProxyCls._register_services)
    reset_call = re.search(r"await self\._reset_server_connection\(\)", src)
    connect_call = re.search(r"connect_to_server\(", src)
    assert reset_call is not None, "expected an actual _reset_server_connection() call"
    assert connect_call is not None, "expected a connect_to_server(...) call"
    assert reset_call.start() < connect_call.start(), (
        "_reset_server_connection() must be awaited BEFORE connect_to_server()"
    )


def test_reset_server_connection_is_idempotent_and_bounded() -> None:
    src = inspect.getsource(_ProxyCls._reset_server_connection)
    assert "self.server is not None" in src or "if self.server" in src
    # Bounded: disconnect() wrapped in a timeout so a wedged transport
    # can't stall the caller (maintenance rebuild, __del__, etc.).
    assert "wait_for" in src
    assert "self.server = None" in src
