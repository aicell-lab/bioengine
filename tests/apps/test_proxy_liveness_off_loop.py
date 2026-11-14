"""A busy proxy replica must not be condemned for being busy.

Ray Serve dispatches a user-defined ``check_health`` onto the replica's
user-code event loop (``asyncio.run_coroutine_threadsafe`` in
``ray/serve/_private/replica.py``). That is the loop ``ProxyDeployment``
relays WebRTC bulk data on, so while a transfer is in flight the check cannot
be answered and Serve condemns the replica after a few missed windows —
measured on ray 2.55.1, and independent of what the check body does. With no
user health check defined Serve skips the user callable entirely.

So the proxy defines none, and the two conditions that genuinely mean "this
replica can never serve again" are read by a watchdog thread instead. Both are
latched or timestamped state, never "is the loop free right now", which is what
makes a saturated data path unable to trigger them.
"""

from __future__ import annotations

import asyncio
import inspect
import time

import pytest

from bioengine.apps import proxy_deployment as pd_module
from tests.apps._proxy_double import PROXY_CLS as _ProxyCls
from tests.apps._proxy_double import ProxyDouble


def _bare_proxy(**attrs):
    inst = object.__new__(ProxyDouble)
    inst.application_id = "app"
    inst.entry_deployment_ready = True
    inst._registration_failure = None
    inst._sibling_read_started_at = None
    inst._sibling_read_job_running = False
    for key, value in attrs.items():
        setattr(inst, key, value)
    return inst


# ===== the health check must be gone, and stay gone =====


def test_proxy_defines_no_check_health() -> None:
    assert not hasattr(_ProxyCls, "check_health"), (
        "ProxyDeployment must define no check_health. Serve dispatches a user "
        "health check onto the user-code event loop, so defining one re-couples "
        "liveness to the loop that relays WebRTC data and a bulk transfer "
        "condemns the replica."
    )


def test_watchdog_never_consults_the_event_loop() -> None:
    """The watchdog runs on a thread, so awaiting or touching the loop from it
    would either fail or reintroduce the coupling."""
    for name in ("_unrecoverable_reason", "_liveness_watchdog"):
        src = inspect.getsource(getattr(_ProxyCls, name))
        assert "await " not in src, f"{name} must not await — it runs off-loop"
        assert "get_event_loop" not in src and "get_running_loop" not in src, (
            f"{name} must not reach for the event loop"
        )
        assert not inspect.iscoroutinefunction(getattr(_ProxyCls, name))


def test_watchdog_is_started_from_init() -> None:
    """check_health used to start the maintenance task. Nothing else is called
    periodically without a client request, so __init__ must start both."""
    src = inspect.getsource(_ProxyCls.__init__)
    assert "_liveness_watchdog" in src
    assert "_ensure_maintenance_task" in src
    assert "daemon=True" in src, (
        "a non-daemon watchdog thread would keep a replica process alive "
        "after Serve has torn the actor down"
    )


# ===== a healthy replica is never condemned =====


def test_healthy_state_is_not_unrecoverable() -> None:
    assert _bare_proxy()._unrecoverable_reason() is None


def test_a_starved_loop_is_not_unrecoverable() -> None:
    """The whole point. The executor job has FINISHED while the coroutine has
    not resumed — that is a blocked event loop with a perfectly good replica,
    and it must read as recoverable no matter how long it has been."""
    inst = _bare_proxy(
        _sibling_read_started_at=time.monotonic() - 10_000.0,
        _sibling_read_job_running=False,
    )
    assert inst._unrecoverable_reason() is None, (
        "a long-starved loop must never be condemned — it is the legitimate "
        "bulk-transfer case this issue exists to protect"
    )


def test_a_slow_but_live_controller_read_is_not_unrecoverable() -> None:
    inst = _bare_proxy(
        _sibling_read_started_at=time.monotonic() - 1.0,
        _sibling_read_job_running=True,
    )
    assert inst._unrecoverable_reason() is None


# ===== the two conditions that do condemn =====


def test_permanent_registration_failure_is_unrecoverable() -> None:
    inst = _bare_proxy(_registration_failure=RuntimeError("wrong workspace"))
    reason = inst._unrecoverable_reason()
    assert reason is not None
    assert "failed permanently" in reason
    assert "wrong workspace" in reason, "the cause must reach the log line"


def test_wedged_controller_read_past_deadline_is_unrecoverable() -> None:
    inst = _bare_proxy(
        _sibling_read_started_at=time.monotonic() - (pd_module._SIBLING_READ_DEADLINE_S + 1.0),
        _sibling_read_job_running=True,
    )
    reason = inst._unrecoverable_reason()
    assert reason is not None
    assert "controller is not answering" in reason


def test_deadline_is_the_only_thing_separating_those_two_readings() -> None:
    """Same state, one variable: elapsed either side of the deadline. Without
    this the two tests above could both pass on an unconditional answer."""
    deadline = pd_module._SIBLING_READ_DEADLINE_S
    inside = _bare_proxy(
        _sibling_read_started_at=time.monotonic() - (deadline - 5.0),
        _sibling_read_job_running=True,
    )
    outside = _bare_proxy(
        _sibling_read_started_at=time.monotonic() - (deadline + 5.0),
        _sibling_read_job_running=True,
    )
    assert inside._unrecoverable_reason() is None
    assert outside._unrecoverable_reason() is not None


# ===== the work check_health used to do still happens =====


@pytest.mark.asyncio
async def test_maintenance_tick_drives_the_gate_and_the_rotation() -> None:
    """Both were reached from check_health. If the tick does not call them the
    app never becomes serviceable and TURN credentials never rotate."""
    src = inspect.getsource(_ProxyCls._maintenance_tick)
    assert "_apply_sibling_gate" in src
    assert "_rotate_webrtc_state" in src

    gate_src = inspect.getsource(_ProxyCls._maintenance_tick)
    assert gate_src.index("_apply_sibling_gate") < gate_src.index(
        "if not self.entry_deployment_ready"
    ), (
        "the gate has to run BEFORE the readiness early-return, or it can "
        "never flip entry_deployment_ready and the tick strands itself"
    )


@pytest.mark.asyncio
async def test_rotation_swallows_both_halves() -> None:
    """A stale ICE list or an un-swept peer must not propagate: the tick is
    documented as never raising, and the replica is fine either way."""
    calls = []

    async def boom_ice(self=None):
        calls.append("ice")
        raise RuntimeError("ice exploded")

    async def boom_sweep(self=None):
        calls.append("sweep")
        raise RuntimeError("sweep exploded")

    inst = _bare_proxy()
    inst._refresh_ice_if_expiring = boom_ice
    inst._sweep_stale_peer_connections = boom_sweep

    await _ProxyCls._rotate_webrtc_state(inst)

    assert calls == ["ice", "sweep"], (
        "the sweep must still run after the refresh raises — one failing half "
        "must not skip the other"
    )
