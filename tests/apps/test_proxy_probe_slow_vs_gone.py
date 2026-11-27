"""The reachability probe must tell a SLOW Hypha from a LOST registration.

#1061. ``_maintenance_tick`` probed with ``get_service_info`` and passed no
timeout, so it inherited ``hypha_rpc``'s 30 s default; then ANY exception set
``_connection_lost = True``, whose remedy is ``_register_services()`` ->
``_reset_server_connection()`` -> ``server.disconnect()``. So a slow server made
the proxy tear down a registration that was fine, the rebuild cost 23-44 s of
absence against the already-slow server, and 60 s later the probe timed out
again. On the live cluster that loop held a public app unreachable for 22
minutes while Ray Serve reported every deployment HEALTHY — Hypha is
deliberately off the replica condemnation path, so nothing else noticed.

The distinction is measurable, and these arms encode the measurement rather
than a guess. Against the hypha-server production image with **hypha_rpc
0.21.40**, 2026-11-27:

* absent service (never registered, or unregistered out from under us):
  ``hypha_rpc.rpc.RemoteException`` in ~0.0 s, message ending
  ``KeyError: 'Service not found: <id>@*'``
* server merely slow: ``asyncio.TimeoutError`` at the probe's bound

The ``Service not found`` text is a server-side message format and will rot
across Hypha versions. ``_is_definitive_absence`` therefore fails CLOSED — an
unrecognised error leaves the registration alone — so a format change degrades
to "slower to notice a real eviction", never to "destroys a live registration".
That degradation is silent, which is why the timeout counter is reported
through ``get_usage_stats`` and not only logged.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from bioengine.apps import proxy_deployment as pd_module
from tests.apps._proxy_double import PROXY_CLS as _ProxyCls
from tests.apps._proxy_double import ProxyDouble


def _bare_proxy(**attrs):
    inst = object.__new__(ProxyDouble)
    inst.application_id = "app"
    inst.entry_deployment_ready = True
    inst.server = None
    inst.websocket_service_id = None
    inst._registration_lock = asyncio.Lock()
    inst._maintenance_task = None
    inst._connection_lost = False
    inst._registration_absent = False
    inst._probe_timeouts = 0
    inst._probe_timeouts_consecutive = 0
    inst._registration_failure = None
    inst._usage_ledger = None
    inst._probe_due_at = 0.0
    inst._next_register_at = 0.0
    inst._proxy_actor_handle = None
    for key, value in attrs.items():
        setattr(inst, key, value)
    return inst


class _RecordingServer:
    """A live connection that answers the probe however the test wants."""

    def __init__(self, exc: BaseException | None = None):
        self._exc = exc
        self.disconnects = 0

    async def get_service_info(self, _sid):
        if self._exc is not None:
            raise self._exc
        return {"id": "ws"}

    async def disconnect(self):
        self.disconnects += 1


class _RecordingRegister:
    def __init__(self, exc: BaseException | None = None):
        self.calls: list[bool] = []
        self._exc = exc

    async def __call__(self, reuse_connection: bool = False):
        self.calls.append(reuse_connection)
        if self._exc is not None:
            raise self._exc


# --------------------------------------------------------------------------
# The arm that is RED on main: a timeout must not tear anything down.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_probe_timeout_does_not_disconnect_or_reregister() -> None:
    server = _RecordingServer(asyncio.TimeoutError())
    register = _RecordingRegister()
    inst = _bare_proxy(server=server, websocket_service_id="ws")
    inst._register_services = register

    await inst._maintenance_tick()  # the probe times out
    await inst._maintenance_tick()  # and the NEXT tick must not act on it

    assert server.disconnects == 0, "a slow server must not cost us the connection"
    assert register.calls == [], "a timeout is not grounds to re-register"
    assert inst._connection_lost is False
    assert inst._registration_absent is False


@pytest.mark.asyncio
async def test_repeated_timeouts_are_counted_and_do_not_escalate() -> None:
    """Constraint: repeated timeouts are logged loudly and counted, but never
    disconnect. Counting is the only trace this branch leaves, because the
    branch's whole point is to do nothing."""
    server = _RecordingServer(asyncio.TimeoutError())
    register = _RecordingRegister()
    inst = _bare_proxy(server=server, websocket_service_id="ws")
    inst._register_services = register

    for _ in range(5):
        inst._probe_due_at = 0.0  # make each probe due again
        await inst._maintenance_tick()

    assert inst._probe_timeouts == 5
    assert inst._probe_timeouts_consecutive == 5
    assert server.disconnects == 0
    assert register.calls == []


@pytest.mark.asyncio
async def test_a_successful_probe_resets_the_consecutive_counter() -> None:
    server = _RecordingServer(asyncio.TimeoutError())
    inst = _bare_proxy(server=server, websocket_service_id="ws")
    inst._register_services = _RecordingRegister()

    await inst._maintenance_tick()
    assert inst._probe_timeouts_consecutive == 1

    server._exc = None  # Hypha recovers
    inst._probe_due_at = 0.0
    await inst._maintenance_tick()

    assert inst._probe_timeouts_consecutive == 0
    # The running total is NOT reset: it is the operator's evidence that this
    # replica has been living with a slow server.
    assert inst._probe_timeouts == 1


# --------------------------------------------------------------------------
# Control: a definitive absence must still be repaired. Without this arm a
# "fix" that simply never re-registers would pass everything above.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_definitive_absence_reregisters_in_place() -> None:
    server = _RecordingServer(KeyError("Service not found: ws@*"))
    register = _RecordingRegister()
    inst = _bare_proxy(server=server, websocket_service_id="ws")
    inst._register_services = register

    await inst._maintenance_tick()  # probe answers: you are not there
    assert inst._registration_absent is True

    await inst._maintenance_tick()  # the remedy runs on the next tick

    assert register.calls == [True], "must re-assert on the EXISTING connection"
    assert server.disconnects == 0, "constraint 3: no unregistered window at all"
    assert inst._registration_absent is False, "cleared on success"


@pytest.mark.asyncio
async def test_a_failed_in_place_reassert_leaves_the_flag_set() -> None:
    """The failure path, which a success-path arm cannot see.

    ``_reset_server_connection`` is what clears ``_connection_lost``, and the
    in-place path deliberately does not call it — so ``_registration_absent``
    has no clearer except the explicit one on success. If that were the only
    place it were touched and the re-assert raised, the flag would be stuck
    SET and every later tick would re-register for the life of the replica;
    if it were cleared unconditionally, a failed repair would be forgotten and
    the service would stay unregistered. Neither is acceptable, and only
    driving the raise shows which one you built.
    """
    server = _RecordingServer()
    register = _RecordingRegister(RuntimeError("hypha said no"))
    inst = _bare_proxy(
        server=server, websocket_service_id="ws", _registration_absent=True
    )
    inst._register_services = register

    await inst._maintenance_tick()

    assert register.calls == [True]
    assert inst._registration_absent is True, "a failed repair must be retried"
    assert server.disconnects == 0
    # And it must not have escalated to a connection rebuild either.
    assert inst._connection_lost is False


@pytest.mark.asyncio
async def test_the_in_place_path_reports_the_registration() -> None:
    """Bookkeeping parity with the full-rebuild success path: an in-place
    re-assert that skips ``_report_service_registration`` leaves the record
    saying unregistered while the service is live."""
    reported: list[bool] = []
    server = _RecordingServer()
    inst = _bare_proxy(
        server=server, websocket_service_id="ws", _registration_absent=True
    )
    inst._register_services = _RecordingRegister()
    inst._report_service_registration = lambda ok: reported.append(ok)

    await inst._maintenance_tick()

    assert reported == [True]
    assert inst._next_register_at == 0.0
    assert inst._probe_due_at > 0.0


# --------------------------------------------------------------------------
# Shape assertions — read, not run.
# --------------------------------------------------------------------------


def test_the_probe_carries_its_own_bound() -> None:
    src = inspect.getsource(_ProxyCls._maintenance_tick)
    assert "get_service_info(self.websocket_service_id)" in src
    assert "_PROBE_TIMEOUT_S" in src, "the probe must not inherit the 30 s default"
    assert "asyncio.wait_for(" in src
    # Sized against the tick, because a blocking probe also stalls
    # _flush_usage and the sibling gate, which run ahead of it.
    assert pd_module._PROBE_TIMEOUT_S <= 2 * pd_module._MAINTENANCE_TICK_S


def test_absence_does_not_enter_the_connection_rebuild_branch() -> None:
    """``needs_registration`` calls ``_register_services()`` with no arguments
    — the destructive rebuild. ``_registration_absent`` must not be folded
    into it, or the in-place path would be bypassed by its own trigger."""
    src = inspect.getsource(_ProxyCls._maintenance_tick)
    head, _, _ = src.partition("if needs_registration:")
    needs_block = head.partition("needs_registration = (")[2]
    assert "_registration_absent" not in needs_block


def test_reuse_connection_skips_the_connection_reset() -> None:
    src = inspect.getsource(_ProxyCls._register_services)
    assert "reuse_connection" in src
    reset_at = src.index("await self._reset_server_connection()")
    guard_at = src.index("if not reuse_connection:")
    assert guard_at < reset_at, "the reset must sit behind the reuse guard"


def test_the_in_place_reassert_passes_overwrite() -> None:
    """The in-place repair MUST ask Hypha to overwrite, and the fresh path must not.

    Measured 2026-11-27 against the hypha-server production image with
    hypha_rpc 0.21.40: re-registering an id this client already holds, on the
    same live connection, is not an upsert — it raises

        Exception: Service already exists: <id>, please specify a different id
        (not <id>) or overwrite=True

    so an in-place re-assert without the flag fails outright and the service
    stays unregistered. With the flag it returns the same service id in 0.01 s
    and the service remains resolvable and callable. The fresh path must NOT
    pass it: a new connection has no prior registration of its own, and the id
    is namespaced by client_id, so the only thing overwrite can replace is this
    client's own earlier registration.
    """
    src = inspect.getsource(_ProxyCls._register_services)
    assert '{"overwrite": True} if reuse_connection else {}' in src


# --------------------------------------------------------------------------
# Round-1 review findings. Each of these is a case the first version of this
# fix got WRONG, so they are controls against repeating it, not decoration.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_closed_transport_still_rebuilds() -> None:
    """A transport that says it is CLOSED is not a slow server.

    Round-1 regression, caught in review. Narrowing the rebuild to "definitive
    absence only" left ``_connection_lost`` with no setter but token renewal,
    so once hypha_rpc marks its connection closed — every call then raising
    ``ConnectionError("Connection is closed")`` IMMEDIATELY, which is how it
    reports a refused reconnection — the proxy sat in the do-nothing branch
    for the life of the replica, logging "did not answer within 5.0s" about an
    error that came back in 0 s.

    Only ``asyncio.TimeoutError`` earns the do-nothing branch. Everything that
    is neither a timeout nor a definitive absence keeps the behaviour it had
    before #1061: rebuild.
    """
    server = _RecordingServer(ConnectionError("Connection is closed"))
    register = _RecordingRegister()
    inst = _bare_proxy(server=server, websocket_service_id="ws")
    inst._register_services = register

    await inst._maintenance_tick()  # probe fails, fast and definitively
    assert inst._connection_lost is True
    assert inst._registration_absent is False
    assert inst._probe_timeouts == 0, "a closed transport is not a timeout"

    await inst._maintenance_tick()  # and the rebuild actually happens
    assert register.calls == [False], "a full rebuild, not an in-place re-assert"


@pytest.mark.asyncio
async def test_repeated_in_place_failures_escalate_to_a_rebuild() -> None:
    """The in-place path assumes the transport is fine. When that assumption is
    wrong there is no other way back, because the rebuild branch is reached
    through ``_connection_lost``, which this path never sets. So it gives up."""
    server = _RecordingServer()
    register = _RecordingRegister(RuntimeError("hypha said no"))
    inst = _bare_proxy(
        server=server, websocket_service_id="ws", _registration_absent=True
    )
    inst._register_services = register

    for _ in range(pd_module._INPLACE_FAILURES_BEFORE_REBUILD):
        inst._next_register_at = 0.0  # make each attempt due
        await inst._maintenance_tick()

    assert inst._connection_lost is True, "must stop insisting the socket is fine"
    assert inst._registration_absent is False
    assert len(register.calls) == pd_module._INPLACE_FAILURES_BEFORE_REBUILD
    assert all(c is True for c in register.calls)


def test_the_in_place_path_clears_the_rtc_registration_first() -> None:
    """``register_rtc_service(server, service_id, config)`` takes no overwrite
    and calls ``register_service`` without one, so on the reuse path the
    client's own ``_services`` dict refuses the id.

    Measured 2026-11-27 on the throwaway hypha-server (production image,
    hypha_rpc 0.21.40): re-asserting the RTC id on a live connection raises
    "Service already exists: <id>-rtc ... or overwrite=True". That exception is
    swallowed by the warn-only handler, because WebRTC must never fail the
    deployment — so without clearing it first, an in-place repair leaves -rtc
    unregistered permanently and nothing retries it, since the probe only ever
    asks about the main id. After ``unregister_service`` the re-register
    succeeds and both ids resolve.
    """
    src = inspect.getsource(_ProxyCls._register_services)
    # Anchor on the CALLS, not on any mention: the source comment next to them
    # names register_rtc_service(...) too, and matching that instead silently
    # compares a comment's position with a call's.
    rtc_at = src.index("await register_rtc_service(")
    clear_at = src.index("await self.server.unregister_service(")
    assert clear_at < rtc_at, "the -rtc entry must be dropped BEFORE re-registering"
    head = src[:rtc_at]
    assert "if reuse_connection:" in head
