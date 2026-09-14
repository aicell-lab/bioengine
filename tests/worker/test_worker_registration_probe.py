"""The worker rebuilds its own Hypha client when the server evicts it.

Hypha can drop a client's registration while the socket stays open from the
container's side: hypha-rpc never sees a disconnect, never reconnects, and
``<workspace>/<client-id>:bioengine-worker`` stops resolving forever while the
process stays alive and every probe passes. ``echo("ping")`` cannot see this —
it kept succeeding for ten minutes against a server that had dropped the
registration — so the worker asks whether Hypha still serves its own service id
instead.

That probe is now the *only* Hypha liveness check in the monitoring loop; the
standalone ``echo`` step is gone. Anything that kills the socket also fails the
registration probe, so the echo step was strictly redundant as a detector. Two
things it did have to keep, and these tests pin both:

* **the repair.** ``echo`` survives as a diagnostic on the failure path, to pick
  the cheapest fix that can work — registration evicted but socket alive means
  re-register only; socket gone means reconnect first.
* **the escalation.** The echo step could propagate, which fed the monitoring
  loop's degraded counter and let the k8s liveness probe cycle a wedged pod.
  The registration probe now does the same, but only after
  ``_REGISTRATION_GRACE_S`` of continuous failure — Hypha serves nothing for a
  minute or two fairly often and recovers by itself, and cycling a pod through
  that costs more than waiting.

Note the throttle interacts with the counter: a check that can only fail once
per ``_REGISTRATION_PROBE_INTERVAL_S`` can never reach a *consecutive*-tick
threshold, because the ticks in between succeed and reset it. So the throttle is
dropped while failing, and the probe retries every tick.
"""

import asyncio
import inspect
import logging
import time

import pytest

from bioengine.worker import worker as worker_module
from bioengine.worker.worker import BioEngineWorker

SERVICE_ID = "my-ws/pod-1:bioengine-worker"


class _Server:
    """Stand-in for the worker's Hypha client connection."""

    def __init__(
        self,
        *,
        serves: bool = True,
        socket_alive: bool = True,
        disconnect_hangs: bool = False,
    ):
        self._serves = serves
        self._socket_alive = socket_alive
        self._disconnect_hangs = disconnect_hangs
        self.echo_calls = 0
        self.probe_calls = 0
        self.disconnected = False

    async def echo(self, message):
        self.echo_calls += 1
        if not self._socket_alive:
            raise RuntimeError("connection closed")
        return message

    async def get_service_info(self, service_id):
        self.probe_calls += 1
        if not self._socket_alive:
            raise RuntimeError("connection closed")
        if not self._serves:
            raise RuntimeError(f"Service not found: {service_id}")
        return {"id": service_id}

    async def disconnect(self):
        if self._disconnect_hangs:
            await asyncio.sleep(3600)
        self.disconnected = True


def _bare_worker(server, **attrs):
    worker = object.__new__(BioEngineWorker)
    worker.logger = logging.getLogger("test-bioengine-worker")
    worker.start_time = None  # keeps __del__ quiet on a hand-built instance
    worker.server = server
    worker.full_service_id = SERVICE_ID
    worker._registration_probe_due_at = 0.0
    worker._registration_failing = False
    worker._registration_ok_at = time.time()
    worker._monitor_consecutive_errors = 0
    worker.reconnects = 0
    worker.registrations = 0

    async def _connect():
        worker.reconnects += 1

    async def _register():
        worker.registrations += 1

    worker._connect_to_server = _connect
    worker._register_bioengine_worker_service = _register
    for key, value in attrs.items():
        setattr(worker, key, value)
    return worker


@pytest.mark.asyncio
async def test_a_served_registration_does_not_repair():
    worker = _bare_worker(_Server(serves=True))

    await worker._check_service_registration()

    assert worker.server.probe_calls == 1
    assert worker.server.echo_calls == 0, "echo must not cost a round trip when healthy"
    assert worker.reconnects == 0
    assert worker.registrations == 0


# ===== the repair the echo step used to do, now chosen by diagnosis =====


@pytest.mark.asyncio
async def test_an_evicted_registration_on_a_live_socket_only_re_registers():
    """The original failure mode, and the cheap repair for it.

    The socket answers, so there is nothing wrong with the connection —
    tearing it down and rebuilding it would be strictly more work than
    re-registering on the connection we already have.
    """
    worker = _bare_worker(_Server(serves=False, socket_alive=True))

    await worker._check_service_registration()

    assert worker.server.echo_calls == 1, "echo is the diagnostic that picks the repair"
    assert worker.reconnects == 0
    assert worker.registrations == 1


@pytest.mark.asyncio
async def test_a_dead_socket_reconnects_and_re_registers():
    """What the removed echo step covered: the connection itself is gone."""
    worker = _bare_worker(_Server(socket_alive=False))

    await worker._check_service_registration()

    assert worker.reconnects == 1
    assert worker.registrations == 1


@pytest.mark.asyncio
async def test_a_dead_socket_is_caught_without_a_separate_echo_step():
    """The probe subsumes echo as a *detector*: a dead socket fails it too."""
    server = _Server(socket_alive=False)
    worker = _bare_worker(server)

    await worker._check_service_registration()

    assert server.probe_calls == 1
    assert worker.reconnects == 1


# ===== the escalation the echo step used to do =====


@pytest.mark.asyncio
async def test_a_failed_repair_is_absorbed_inside_the_grace_period():
    worker = _bare_worker(_Server(serves=False))

    async def _register():
        raise RuntimeError("Hypha is returning 500")

    worker._register_bioengine_worker_service = _register

    # Must not raise: a Hypha blip is not a reason to cycle the pod.
    await worker._check_service_registration()

    assert worker._monitor_consecutive_errors == 0


@pytest.mark.asyncio
async def test_a_repair_that_keeps_failing_past_the_grace_period_raises():
    """The backstop. Without this, dropping the echo step would delete the only
    path from "Hypha is unreachable" to "cycle this pod"."""
    worker = _bare_worker(_Server(serves=False))

    async def _register():
        raise RuntimeError("Hypha is returning 500")

    worker._register_bioengine_worker_service = _register
    worker._registration_ok_at = (
        time.time() - worker_module._REGISTRATION_GRACE_S - 1
    )

    with pytest.raises(RuntimeError, match="unreachable"):
        await worker._check_service_registration()


@pytest.mark.asyncio
async def test_the_grace_is_measured_from_the_last_good_probe_not_the_first_failure():
    worker = _bare_worker(_Server(serves=False))

    async def _register():
        raise RuntimeError("Hypha is returning 500")

    worker._register_bioengine_worker_service = _register

    # Just inside the grace: absorbed.
    worker._registration_ok_at = time.time() - worker_module._REGISTRATION_GRACE_S + 30
    await worker._check_service_registration()

    # Same streak, now past it: raises.
    worker._registration_ok_at = time.time() - worker_module._REGISTRATION_GRACE_S - 1
    with pytest.raises(RuntimeError):
        await worker._check_service_registration()


@pytest.mark.asyncio
async def test_the_probe_reaches_the_degraded_threshold_one_tick_at_a_time():
    """A throttled check cannot feed a CONSECUTIVE-tick counter.

    The monitoring loop resets ``_monitor_consecutive_errors`` on every clean
    tick, so a probe that only runs once a minute would be cleared by the five
    ticks in between and could never reach the threshold. While failing, the
    throttle is dropped.
    """
    server = _Server(serves=False)
    worker = _bare_worker(server)

    async def _register():
        raise RuntimeError("Hypha is returning 500")

    worker._register_bioengine_worker_service = _register
    worker._registration_ok_at = time.time() - worker_module._REGISTRATION_GRACE_S - 1

    for _ in range(5):
        with pytest.raises(RuntimeError):
            await worker._check_service_registration()

    assert server.probe_calls == 5, (
        "a failing registration must be retried every tick, not once per "
        "probe interval — otherwise the degraded counter resets in between"
    )


@pytest.mark.asyncio
async def test_recovering_clears_the_failing_state_and_restores_the_throttle():
    server = _Server(serves=False)
    worker = _bare_worker(server)

    await worker._check_service_registration()  # repairs, marks healthy again
    assert worker.registrations == 1
    assert worker._registration_failing is False

    # The throttle was armed at the top of that same call, so a successful
    # repair does not buy another probe this interval.
    server._serves = True
    await worker._check_service_registration()
    assert server.probe_calls == 1

    # Next interval: probes, stays healthy, repairs nothing further.
    worker._registration_probe_due_at = 0.0
    await worker._check_service_registration()
    assert server.probe_calls == 2
    assert worker.registrations == 1
    assert worker.reconnects == 0


# ===== unchanged guarantees =====


@pytest.mark.asyncio
async def test_the_probe_costs_one_round_trip_per_interval():
    worker = _bare_worker(_Server(serves=True))

    await worker._check_service_registration()
    await worker._check_service_registration()
    await worker._check_service_registration()

    assert worker.server.probe_calls == 1


@pytest.mark.asyncio
async def test_the_probe_waits_until_the_service_is_registered():
    worker = _bare_worker(_Server(serves=True), full_service_id=None)

    await worker._check_service_registration()

    assert worker.server.probe_calls == 0
    assert worker.reconnects == 0


@pytest.mark.asyncio
async def test_a_worker_with_no_connection_is_a_no_op():
    worker = _bare_worker(None)

    await worker._check_service_registration()

    assert worker.reconnects == 0


@pytest.mark.asyncio
async def test_a_hung_probe_does_not_stall_the_monitoring_loop(monkeypatch):
    monkeypatch.setattr(worker_module, "_REGISTRATION_PROBE_TIMEOUT_S", 0.05)

    class _HangingServer(_Server):
        async def get_service_info(self, service_id):
            self.probe_calls += 1
            await asyncio.sleep(3600)

    worker = _bare_worker(_HangingServer())

    await asyncio.wait_for(worker._check_service_registration(), timeout=10)

    # Socket still answers, so the cheap repair is the right one.
    assert worker.registrations == 1
    assert worker.reconnects == 0


@pytest.mark.asyncio
async def test_a_hung_echo_does_not_stall_the_repair(monkeypatch):
    """The diagnostic must not become a new way to wedge the loop."""
    monkeypatch.setattr(worker_module, "_REGISTRATION_PROBE_TIMEOUT_S", 0.05)

    class _HangingEcho(_Server):
        async def echo(self, message):
            self.echo_calls += 1
            await asyncio.sleep(3600)

    worker = _bare_worker(_HangingEcho(serves=False))

    await asyncio.wait_for(worker._check_service_registration(), timeout=10)

    # A hung echo reads as a dead socket, so the full rebuild is taken.
    assert worker.reconnects == 1
    assert worker.registrations == 1


@pytest.mark.asyncio
async def test_a_hung_disconnect_does_not_stall_the_rebuild(monkeypatch):
    # The transport being closed on the rebuild path is the one already
    # suspected of being wedged, so the close has to be bounded.
    monkeypatch.setattr(worker_module, "_DISCONNECT_TIMEOUT_S", 0.05)

    connected = []

    async def _fake_connect_to_server(config):
        connected.append(config)
        return _Server()

    monkeypatch.setattr(worker_module, "connect_to_server", _fake_connect_to_server)

    worker = object.__new__(BioEngineWorker)
    worker.logger = logging.getLogger("test-bioengine-worker")
    worker.start_time = None
    worker.server = _Server(disconnect_hangs=True)
    worker.server_url = "https://hypha.example"
    worker._token = "token"
    worker.workspace = None
    worker.client_id = None

    with pytest.raises(Exception):
        # Stops at the first real step after the disconnect; what matters is
        # that the hung close did not swallow the call.
        await asyncio.wait_for(worker._connect_to_server(), timeout=10)

    assert connected, "the rebuild never reached connect_to_server"


def test_the_monitoring_loop_runs_the_registration_probe():
    source = inspect.getsource(BioEngineWorker._create_monitoring_task)

    assert "_check_service_registration()" in source


def test_the_standalone_echo_step_is_gone():
    """Pin the removal, so a future edit cannot quietly reinstate a redundant
    10s heartbeat alongside the probe that already subsumes it."""
    source = inspect.getsource(BioEngineWorker._create_monitoring_task)

    assert "_check_hypha_connection" not in source
    assert not hasattr(BioEngineWorker, "_check_hypha_connection")


def test_the_probe_can_reach_the_readiness_backstop():
    # The inverse of what this file used to assert. _monitor_consecutive_errors
    # drives the not-ready flip that lets the k8s liveness probe cycle the pod,
    # and with the echo step gone this is the only Hypha check that can feed it.
    body = inspect.getsource(BioEngineWorker._check_service_registration).split('"""')[2]

    assert "raise RuntimeError" in body
    assert "_REGISTRATION_GRACE_S" in body, "escalation must be gated on the grace"
