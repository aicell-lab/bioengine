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

* **the repair.** Disconnect, reconnect, register — unchanged, and deliberately
  unconditional. Re-registering on the client we already hold cannot work:
  ``hypha_rpc`` keeps a local service registry that a server-side eviction never
  touches, so ``register_service`` on that client raises *Service already
  exists*. ``test_a_repair_goes_through_the_real_hypha_service_registry`` pins
  that against the real library rather than a double.
* **the escalation.** The echo step could propagate, which feeds the monitoring
  loop's degraded counter and flips ``get_status`` to not-ready. The
  registration probe now does the same, but only after
  ``_REGISTRATION_GRACE_S`` of continuous failure — Hypha serves nothing for a
  minute or two fairly often and recovers by itself.

Note the throttle interacts with the counter: a check that can only fail once
per ``_REGISTRATION_PROBE_INTERVAL_S`` can never reach a *consecutive*-tick
threshold, because the ticks in between succeed and reset it. So the throttle is
dropped while failing, and the probe retries every tick.
"""

import asyncio
import inspect
import logging
import time
from types import SimpleNamespace

import pytest
import pytest_asyncio
from hypha_rpc.rpc import RPC

from bioengine.worker import worker as worker_module
from bioengine.worker.worker import BioEngineWorker

WORKSPACE = "my-ws"
CLIENT_ID = "pod-1"
SERVICE_ID = f"{WORKSPACE}/{CLIENT_ID}:bioengine-worker"


class _Server:
    """Stand-in for the worker's Hypha client connection.

    Deliberately has no ``echo``: the standalone heartbeat is gone, so a
    reinstated one fails every test in this file rather than passing silently.
    """

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
        self.probe_calls = 0
        self.disconnected = False

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
    worker._registration_window_start = worker._registration_ok_at
    worker._registration_window_failures = 0
    worker._monitor_consecutive_errors = 0
    worker.reconnects = 0
    worker.registrations = 0
    worker.repair_steps = []

    async def _connect():
        worker.reconnects += 1
        worker.repair_steps.append("connect")
        # A rebuilt client reaches a server that is answering again.
        worker.server._socket_alive = True

    async def _register():
        worker.registrations += 1
        worker.repair_steps.append("register")
        worker.server._serves = True

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
    assert worker.repair_steps == []


# ===== the repair the echo step used to do =====


@pytest.mark.asyncio
async def test_an_evicted_registration_is_repaired_by_rebuilding_the_client():
    """The original failure mode: the socket answers, the registration is gone.

    The repair reconnects before it registers even though the connection looks
    healthy — see ``test_a_repair_goes_through_the_real_hypha_service_registry``
    for why re-registering on the live client cannot work.
    """
    worker = _bare_worker(_Server(serves=False, socket_alive=True))

    await worker._check_service_registration()

    assert worker.repair_steps == ["connect", "register"]
    assert worker._registration_failing is False


@pytest.mark.asyncio
async def test_a_dead_socket_is_repaired_the_same_way():
    """What the removed echo step covered: the connection itself is gone."""
    server = _Server(socket_alive=False)
    worker = _bare_worker(server)

    await worker._check_service_registration()

    assert server.probe_calls >= 1, "the probe subsumes echo as a detector"
    assert worker.repair_steps == ["connect", "register"]


@pytest.mark.asyncio
async def test_a_repair_goes_through_the_real_hypha_service_registry(
    real_registration_worker,
):
    """The repair must survive the real ``hypha_rpc.register_service``.

    ``RPC.add_service`` refuses an id already in the client's local registry,
    and a server-side eviction never clears that registry — so re-registering
    on the client the worker already holds raises *Service already exists*
    every single time, and a worker that only did that could never recover.
    Both halves are asserted here against the real library, not a double.
    """
    worker = real_registration_worker

    await worker._register_bioengine_worker_service()  # the startup registration
    assert worker.server.serves is True

    with pytest.raises(Exception, match="already exists"):
        await worker._register_bioengine_worker_service()

    worker.server.serves = False  # Hypha evicts us; the socket is untouched

    await worker._check_service_registration()

    assert worker.server.serves is True
    assert worker._registration_failing is False
    assert worker.reconnects == 1, "the repair only works because it reconnects"


# ===== the escalation the echo step used to do =====


@pytest.mark.asyncio
async def test_a_failed_repair_is_absorbed_inside_the_grace_period():
    worker = _bare_worker(_Server(serves=False))
    worker._register_bioengine_worker_service = _failing_registration
    ok_at = worker._registration_ok_at

    # Must not raise: a Hypha blip is not a reason to report a worker degraded.
    await worker._check_service_registration()

    assert worker._registration_failing is True
    assert worker._registration_ok_at == ok_at, "an absorbed failure is not a recovery"


@pytest.mark.asyncio
async def test_a_repair_that_keeps_failing_past_the_grace_period_raises():
    """The backstop. Without this, dropping the echo step would delete the only
    path from "Hypha is unreachable" to a degraded worker."""
    worker = _bare_worker(_Server(serves=False))
    worker._register_bioengine_worker_service = _failing_registration
    worker._registration_ok_at = time.time() - worker_module._REGISTRATION_GRACE_S - 1

    with pytest.raises(RuntimeError, match="unreachable"):
        await worker._check_service_registration()


@pytest.mark.asyncio
async def test_a_repair_that_does_not_restore_reachability_is_not_a_recovery():
    """Reconnecting and registering can both succeed while the service stays
    unresolvable. Accepting that as recovery refills the grace clock on every
    tick, so the escalation above could never be reached."""
    server = _Server(serves=False)
    worker = _bare_worker(server)

    async def _register_without_restoring():
        worker.registrations += 1  # succeeds locally, service still unresolvable

    worker._register_bioengine_worker_service = _register_without_restoring
    ok_at = time.time() - worker_module._REGISTRATION_GRACE_S + 30
    worker._registration_ok_at = ok_at

    await worker._check_service_registration()

    assert worker.registrations == 1
    assert worker._registration_failing is True
    assert worker._registration_ok_at == ok_at, "the grace clock must keep running"

    worker._registration_ok_at = time.time() - worker_module._REGISTRATION_GRACE_S - 1
    with pytest.raises(RuntimeError, match="unreachable"):
        await worker._check_service_registration()


@pytest.mark.asyncio
async def test_the_grace_is_measured_from_the_last_good_probe_not_the_first_failure():
    worker = _bare_worker(_Server(serves=False))
    worker._register_bioengine_worker_service = _failing_registration

    # Just inside the grace: absorbed.
    worker._registration_ok_at = time.time() - worker_module._REGISTRATION_GRACE_S + 30
    await worker._check_service_registration()

    # Same streak, now past it: raises.
    worker._registration_ok_at = time.time() - worker_module._REGISTRATION_GRACE_S - 1
    with pytest.raises(RuntimeError):
        await worker._check_service_registration()


@pytest.mark.asyncio
async def test_one_good_probe_resets_the_grace_clock():
    """The grace measures CONTINUOUS unreachability, not cumulative.

    Deliberate, and worth pinning because "unreachable for 300s" reads as
    cumulative. A probe that answers means the service really was resolvable at
    that instant, so a flapping connection never escalates — the same way every
    other check in the monitoring loop resets on a clean tick.
    """
    server = _Server(serves=False)
    worker = _bare_worker(server)
    worker._register_bioengine_worker_service = _failing_registration
    worker._registration_ok_at = time.time() - worker_module._REGISTRATION_GRACE_S + 5

    # One tick short of escalating.
    await worker._check_service_registration()

    # The connection flaps back for a single probe.
    server._serves = True
    await worker._check_service_registration()
    assert worker._registration_failing is False

    # Now failing again, and past what would have been the original deadline.
    server._serves = False
    worker._registration_probe_due_at = 0.0
    await worker._check_service_registration()  # must not raise


@pytest.mark.asyncio
async def test_a_flapping_registration_is_reported_even_though_it_never_escalates(
    caplog,
):
    """The blind spot the reset-on-success property creates, closed by warning.

    A registration that answers one probe in ten recovers the grace clock every
    time, so it never reaches the degraded threshold — and today it produces no
    escalation, no page and no log anyone reads. The flap window counts on a
    clock that does not reset on success and only ever warns.
    """
    server = _Server(serves=True)
    worker = _bare_worker(server)
    worker._registration_window_start = time.time()

    for _ in range(worker_module._REGISTRATION_FLAP_THRESHOLD):
        server._serves = False
        worker._registration_probe_due_at = 0.0
        await worker._check_service_registration()  # fails, then repairs
        worker._registration_probe_due_at = 0.0
        await worker._check_service_registration()  # confirms it is healthy again

    # Nothing has escalated: every failure was followed by a recovery.
    assert worker._registration_failing is False
    assert (
        worker._registration_window_failures
        == worker_module._REGISTRATION_FLAP_THRESHOLD
    )

    # Roll the observation window.
    worker._registration_window_start = (
        time.time() - worker_module._REGISTRATION_FLAP_WINDOW_S - 1
    )
    worker._registration_probe_due_at = 0.0
    with caplog.at_level(logging.WARNING, logger="test-bioengine-worker"):
        await worker._check_service_registration()

    assert any(
        "flapping" in record.message for record in caplog.records
    ), "an intermittent registration must be reported even though it never escalates"


@pytest.mark.asyncio
async def test_one_continuous_outage_is_one_flap_episode_not_one_per_tick():
    """The window counts episodes, and the probe retries every tick while
    failing — so counting attempts would report a single 50s outage as a
    flapping connection, which is the opposite diagnosis."""
    worker = _bare_worker(_Server(serves=False))
    worker._register_bioengine_worker_service = _failing_registration

    for _ in range(worker_module._REGISTRATION_FLAP_THRESHOLD):
        await worker._check_service_registration()

    assert worker.server.probe_calls == worker_module._REGISTRATION_FLAP_THRESHOLD
    assert worker._registration_window_failures == 1


@pytest.mark.asyncio
async def test_the_flap_window_stays_quiet_below_its_threshold(caplog):
    server = _Server(serves=True)
    worker = _bare_worker(server)
    worker._registration_window_failures = (
        worker_module._REGISTRATION_FLAP_THRESHOLD - 1
    )
    worker._registration_window_start = (
        time.time() - worker_module._REGISTRATION_FLAP_WINDOW_S - 1
    )

    with caplog.at_level(logging.WARNING, logger="test-bioengine-worker"):
        await worker._check_service_registration()

    assert not any("flapping" in record.message for record in caplog.records)
    assert worker._registration_window_failures == 0, "the window must roll"


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
    worker._register_bioengine_worker_service = _failing_registration
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

    await worker._check_service_registration()  # repairs, confirms, marks healthy
    assert worker.registrations == 1
    assert worker._registration_failing is False
    assert server.probe_calls == 2, "the repair is confirmed by a second probe"

    # The throttle was armed at the top of that same call, so a successful
    # repair does not buy another probe this interval.
    await worker._check_service_registration()
    assert server.probe_calls == 2

    # Next interval: probes, stays healthy, repairs nothing further.
    worker._registration_probe_due_at = 0.0
    await worker._check_service_registration()
    assert server.probe_calls == 3
    assert worker.registrations == 1
    assert worker.reconnects == 1


# ===== the monitoring loop the check is wired into =====


@pytest.mark.asyncio
async def test_the_monitoring_loop_starts_the_grace_clock_it_does_not_inherit_it(
    tmp_path,
):
    """Construction is minutes before the first probe on a real worker — Ray
    start, connect, data-server discovery, dataset refresh, app recovery,
    startup apps. A grace clock started in ``__init__`` is already spent by the
    time the loop runs, so the first transient blip escalates immediately.
    """
    worker = _loop_worker(_Server(serves=False), tmp_path)
    worker._register_bioengine_worker_service = _failing_registration
    # As if the worker had been constructed an hour before the loop started.
    worker._registration_ok_at = time.time() - 3600
    worker._registration_window_start = worker._registration_ok_at

    await _run_monitoring_loop(worker, lambda: worker.server.probe_calls >= 2)

    assert worker._registration_failing is True
    assert worker._monitor_consecutive_errors == 0, (
        "a blip on the first tick after a slow startup must be absorbed, not "
        "escalated with zero grace"
    )


@pytest.mark.asyncio
async def test_a_condemned_registration_advances_the_degraded_counter(
    monkeypatch, tmp_path
):
    """The escalation's reachable half: the raise propagates out of the
    monitoring step and into the counter behind ``get_status``'s readiness.
    Nothing consumes that report yet — the deployed liveness probe is a local
    PID check — so this is where the escalation currently stops.
    """
    monkeypatch.setattr(worker_module, "_REGISTRATION_GRACE_S", 0)

    worker = _loop_worker(_Server(serves=False), tmp_path)
    worker._register_bioengine_worker_service = _failing_registration

    await _run_monitoring_loop(worker, lambda: worker._monitor_consecutive_errors >= 1)


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
    assert worker.repair_steps == []


@pytest.mark.asyncio
async def test_a_worker_with_no_connection_is_a_no_op():
    worker = _bare_worker(None)

    await worker._check_service_registration()

    assert worker.repair_steps == []


@pytest.mark.asyncio
async def test_a_hung_probe_does_not_stall_the_monitoring_loop(monkeypatch):
    monkeypatch.setattr(worker_module, "_REGISTRATION_PROBE_TIMEOUT_S", 0.05)

    class _HangingServer(_Server):
        async def get_service_info(self, service_id):
            self.probe_calls += 1
            await asyncio.sleep(3600)

    worker = _bare_worker(_HangingServer())

    await asyncio.wait_for(worker._check_service_registration(), timeout=10)

    assert worker.repair_steps == ["connect", "register"]
    assert worker._registration_failing is True, "a hung probe never confirms a repair"


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


def test_the_standalone_echo_step_is_gone():
    """Pin the removal, so a future edit cannot quietly reinstate a redundant
    10s heartbeat alongside the probe that already subsumes it. ``_Server``
    above has no ``echo`` either, so a reinstated call also breaks every
    behavioural test in this file."""
    assert not hasattr(BioEngineWorker, "_check_hypha_connection")
    assert "_check_hypha_connection" not in inspect.getsource(
        BioEngineWorker._create_monitoring_task
    )


# ===== helpers =====


async def _failing_registration():
    raise RuntimeError("Hypha is returning 500")


async def _noop(*args, **kwargs):
    return None


def _loop_worker(server, tmp_path, **attrs):
    """A worker whose monitoring loop runs for real, with every step but the
    registration check stubbed out."""
    worker = _bare_worker(server, **attrs)
    worker.heartbeat_file = tmp_path / "worker_heartbeat.json"
    worker.is_ready = asyncio.Event()
    worker.monitoring_interval_seconds = 0
    worker._last_monitoring = 0
    worker._monitor_degraded_threshold = 5
    worker._fetch_geo_location = _noop
    worker._check_token_expiry = _noop
    worker._ping_data_server = _noop
    worker._discover_data_server = _noop
    worker._refresh_datasets = _noop
    worker._cleanup = _noop
    worker.ray_cluster = SimpleNamespace(
        check_connection=_noop, monitor_cluster=_noop, mode="single-machine"
    )
    worker.apps_manager = SimpleNamespace(monitor_applications=_noop)
    return worker


async def _run_monitoring_loop(worker, until, timeout: float = 20.0):
    task = asyncio.create_task(worker._create_monitoring_task())
    try:
        deadline = time.time() + timeout
        while not until():
            if task.done():
                await task
                raise AssertionError("the monitoring loop exited early")
            if time.time() > deadline:
                raise AssertionError("the monitoring loop never reached the condition")
            await asyncio.sleep(0.02)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class _OfflineConnection:
    """Enough of a hypha-rpc connection for the client-side service registry."""

    workspace = WORKSPACE
    manager_id = "workspace-manager"

    def on_message(self, handler):
        pass

    def on_connected(self, handler):
        pass

    def on_disconnected(self, handler):
        pass

    async def emit_message(self, data):
        raise RuntimeError("offline")


class _RealRegistryServer:
    """A server double whose ``register_service`` is the real hypha-rpc one.

    ``notify=False`` skips the workspace-manager round trip and keeps
    ``RPC.add_service`` — the duplicate-id check that the repair has to get
    past — exactly as the library ships it.
    """

    def __init__(self):
        self.rpc = RPC(_OfflineConnection(), client_id=CLIENT_ID, workspace=WORKSPACE)
        self.serves = False

    async def register_service(self, api):
        service_info = await self.rpc.register_service(api, {"notify": False})
        self.serves = True
        return service_info

    async def get_service_info(self, service_id):
        if not self.serves:
            raise RuntimeError(f"Service not found: {service_id}")
        return {"id": service_id}

    async def disconnect(self):
        pass


@pytest_asyncio.fixture
async def real_registration_worker():
    """A worker using the real ``_register_bioengine_worker_service``.

    ``RPC`` starts a session-GC task whose first sleep is minutes long, and the
    test loop waits for it unless every client is closed.
    """
    worker = object.__new__(BioEngineWorker)
    worker.logger = logging.getLogger("test-bioengine-worker")
    worker.start_time = None
    worker.rpc_servers = []
    worker.service_id = "bioengine-worker"
    worker.worker_name = "test-worker"
    worker.full_service_id = SERVICE_ID
    worker._registration_probe_due_at = 0.0
    worker._registration_failing = False
    worker._registration_ok_at = time.time()
    worker._registration_window_start = worker._registration_ok_at
    worker._registration_window_failures = 0
    worker.reconnects = 0
    worker.ray_cluster = SimpleNamespace(mode="single-machine")
    worker.code_executor = SimpleNamespace(run_code=_noop)
    worker.apps_manager = SimpleNamespace(
        upload_app=_noop,
        list_app_directories=_noop,
        clear_app_directory=_noop,
        list_apps=_noop,
        get_app_manifest=_noop,
        delete_app=_noop,
        delete_app_version=_noop,
        deploy_app=_noop,
        stop_app=_noop,
        stop_all_apps=_noop,
        get_app_status=_noop,
    )

    def _new_server():
        server = _RealRegistryServer()
        worker.rpc_servers.append(server)
        return server

    async def _connect():
        worker.reconnects += 1
        worker.server = _new_server()

    worker.server = _new_server()
    worker._connect_to_server = _connect

    yield worker

    for server in worker.rpc_servers:
        server.rpc.close()
