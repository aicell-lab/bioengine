"""Startup must survive a transient dependency failure.

Two paths are covered: the initial connect (Hypha, Ray client), which used to
exit the worker on the first refusal, and startup applications, where one
application's failure used to abort the whole worker.
"""

import asyncio
import inspect
import time
from types import SimpleNamespace

import pytest

from bioengine.apps import manager as manager_module
from bioengine.apps.manager import AppsManager
from bioengine.cluster import ray_cluster as ray_cluster_module
from bioengine.cluster.ray_cluster import RayCluster
from bioengine.heartbeat import heartbeat_stale_after_seconds
from bioengine.utils import (
    RECONNECT_BUDGET_S,
    STARTUP_CONNECT_BUDGET_S,
    connect_with_retry,
    is_transient_connect_error,
)
from bioengine.utils import network as network_module
from bioengine.worker import worker as worker_module
from bioengine.worker.worker import BioEngineWorker


class _RecordingLogger:
    def __init__(self):
        self.messages = []

    def _record(self, message):
        self.messages.append(message)

    info = warning = error = debug = _record


def test_connection_failures_are_transient_but_auth_failures_are_not():
    assert is_transient_connect_error(
        ConnectionRefusedError(111, "Connect call failed ('10.43.210.131', 9520)")
    )
    assert is_transient_connect_error(
        Exception("server rejected WebSocket connection: HTTP 503")
    )
    # The checks the worker must still fail fast on.
    assert not is_transient_connect_error(
        ValueError("Provided token does not have admin permissions.")
    )
    assert not is_transient_connect_error(
        ValueError("Workspace mismatch: a (local) vs b (server)")
    )


@pytest.mark.parametrize(
    "message",
    [
        "Authentication error: Error decoding token headers.",
        "Failed to establish connection: Failed to authenticate user: Current "
        "workspace encoded in the token (ws-a) does not match the specified "
        "workspace (ws-b)",
        "Failed to establish connection: Client already exists and is active: "
        "ws-a/some-worker",
    ],
)
def test_hypha_rejections_fail_fast_despite_being_connection_errors(message):
    """The rejections hypha_rpc actually raises, verbatim.

    Every one arrives as ConnectionAbortedError, so classifying on the
    exception type alone retries a token that will never work for the whole
    budget. Measured against hypha.aicell.io with hypha-rpc 0.21.x.
    """
    assert not is_transient_connect_error(ConnectionAbortedError(message))


async def test_retries_a_refused_connection_until_it_succeeds():
    attempts = []

    async def connect():
        attempts.append(None)
        if len(attempts) < 3:
            raise ConnectionRefusedError(111, "Connect call failed")
        return "connected"

    result = await connect_with_retry(
        connect,
        description="Connection to Hypha server",
        logger=_RecordingLogger(),
        total_seconds=5.0,
        initial_delay=0.01,
        max_delay=0.01,
    )

    assert result == "connected"
    assert len(attempts) == 3


async def test_does_not_retry_an_authentication_failure():
    attempts = []

    async def connect():
        attempts.append(None)
        raise ValueError("Provided token does not have admin permissions.")

    with pytest.raises(ValueError):
        await connect_with_retry(
            connect,
            description="Connection to Hypha server",
            logger=_RecordingLogger(),
            total_seconds=5.0,
            initial_delay=0.01,
        )

    assert len(attempts) == 1


async def test_gives_up_once_the_budget_is_spent():
    async def connect():
        raise ConnectionRefusedError(111, "Connect call failed")

    with pytest.raises(ConnectionRefusedError):
        await connect_with_retry(
            connect,
            description="Connection to Hypha server",
            logger=_RecordingLogger(),
            total_seconds=0.05,
            initial_delay=0.01,
            max_delay=0.01,
        )


async def test_worker_retries_the_initial_hypha_connection(monkeypatch):
    """The retry has to be wired into _connect_to_server, not just available."""

    class _Connected(Exception):
        """Raised past the connect call to end the test early."""

    attempts = []

    async def fake_connect_to_server(config):
        attempts.append(config)
        if len(attempts) < 3:
            raise ConnectionRefusedError(111, "Connect call failed")

        async def generate_token():
            raise _Connected

        return SimpleNamespace(generate_token=generate_token)

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(worker_module, "connect_to_server", fake_connect_to_server)
    # network.py uses asyncio only for the backoff sleep.
    monkeypatch.setattr(network_module, "asyncio", SimpleNamespace(sleep=no_sleep))
    worker = SimpleNamespace(
        server=None,
        logger=_RecordingLogger(),
        server_url="http://hypha:9520",
        _token="token",
        workspace="ws",
        client_id="worker",
    )

    with pytest.raises(_Connected):
        await BioEngineWorker._connect_to_server(worker)

    assert len(attempts) == 3


def _startup_manager(startup_applications, deploy_app):
    """An AppsManager stand-in exposing only what the startup path touches."""
    deploy_app.__schema__ = {
        "parameters": {
            "properties": {
                "artifact_id": {},
                "version": {},
                "application_id": {},
                "hypha_token": {},
                "context": {},
            }
        }
    }

    async def generate_token(_config):
        return "startup-token"

    manager = SimpleNamespace(
        startup_applications=startup_applications,
        logger=_RecordingLogger(),
        server=SimpleNamespace(
            config=SimpleNamespace(workspace="ws"),
            generate_token=generate_token,
        ),
        admin_users=["admin@example.com"],
        deploy_app=deploy_app,
        _startup_retry_task=None,
        # Reporting-only helpers the startup path calls for their side effects.
        # Absorbed here so this test stays about retry, not about them.
        _warn_on_startup_pin_divergence=lambda _app_config: None,
    )
    manager._retry_startup_applications = (
        lambda configs: AppsManager._retry_startup_applications(manager, configs)
    )
    return manager


async def test_one_failing_startup_application_does_not_abort_the_others(monkeypatch):
    monkeypatch.setattr(manager_module, "_REDEPLOY_BACKOFF_INITIAL_SECONDS", 60.0)
    deployed = []

    async def deploy_app(**kwargs):
        artifact_id = kwargs["artifact_id"]
        if artifact_id == "ws/model-runner":
            raise OSError(37, "No locks available")
        deployed.append(artifact_id)
        return artifact_id.split("/")[-1]

    manager = _startup_manager(
        [
            {"artifact_id": "ws/model-runner"},
            {"artifact_id": "ws/annotation-broker"},
        ],
        deploy_app,
    )

    await AppsManager.deploy_startup_applications(manager)

    assert deployed == ["ws/annotation-broker"]
    assert manager._startup_retry_task is not None
    manager._startup_retry_task.cancel()


async def test_a_failed_startup_application_is_retried(monkeypatch):
    monkeypatch.setattr(manager_module, "_REDEPLOY_BACKOFF_INITIAL_SECONDS", 0.01)
    monkeypatch.setattr(manager_module, "_REDEPLOY_BACKOFF_MAX_SECONDS", 0.01)
    attempts = []

    async def deploy_app(**kwargs):
        attempts.append(kwargs["artifact_id"])
        if len(attempts) < 3:
            raise OSError(37, "No locks available")
        return "model-runner"

    manager = _startup_manager([], deploy_app)

    await AppsManager._retry_startup_applications(
        manager, [{"artifact_id": "ws/model-runner"}]
    )

    assert attempts == ["ws/model-runner"] * 3


class _VirtualClock:
    """Charges every bound the repair path waits on, without waiting."""

    def __init__(self):
        self.elapsed = 0.0

    def now(self) -> float:
        return self.elapsed

    async def sleep(self, seconds: float) -> None:
        self.elapsed += seconds


class _UnreachableHypha:
    """Answers nothing, charging the clock the timeouts the worker enforces."""

    def __init__(self, clock: _VirtualClock):
        self._clock = clock

    async def get_service_info(self, _service_id):
        await self._clock.sleep(worker_module._REGISTRATION_PROBE_TIMEOUT_S)
        raise asyncio.TimeoutError("probe timed out")

    async def disconnect(self):
        await self._clock.sleep(worker_module._DISCONNECT_TIMEOUT_S)
        raise asyncio.TimeoutError("disconnect timed out")


def _default(func, name):
    return inspect.signature(func).parameters[name].default


async def test_a_reconnect_pass_fits_inside_the_heartbeat_deadline(monkeypatch):
    """A monitoring-loop repair must finish before its own heartbeat goes stale.

    The repair shares ``_connect_to_server`` with startup, so it also shares
    whatever retry budget startup needs. Running the real probe and the real
    reconnect against a Hypha that answers nothing, with every bound they
    enforce charged to one clock, measures what one failing pass costs the
    monitoring loop.
    """
    clock = _VirtualClock()
    monkeypatch.setattr(network_module, "asyncio", SimpleNamespace(sleep=clock.sleep))
    monkeypatch.setattr(network_module, "time", SimpleNamespace(monotonic=clock.now))

    async def refused(_config):
        raise ConnectionRefusedError(111, "Connect call failed")

    monkeypatch.setattr(worker_module, "connect_to_server", refused)

    worker = BioEngineWorker.__new__(BioEngineWorker)
    worker.logger = _RecordingLogger()
    worker.start_time = None
    worker.server = _UnreachableHypha(clock)
    worker.server_url = "http://hypha:9520"
    worker._token = "token"
    worker.workspace = "ws"
    worker.client_id = "worker"
    worker.full_service_id = "ws/worker:bioengine-worker"
    worker._registration_failing = False
    worker._registration_probe_due_at = 0.0
    worker._registration_ok_at = time.time()
    worker._registration_window_start = time.time()
    worker._registration_window_failures = 0

    await worker._check_service_registration()

    deadline = heartbeat_stale_after_seconds(
        _default(BioEngineWorker.__init__, "monitoring_interval_seconds"),
        _default(BioEngineWorker._create_monitoring_task, "backoff_max_seconds"),
    )
    assert clock.elapsed < deadline, (
        f"one repair pass blocks the monitoring loop for {clock.elapsed:.0f}s, "
        f"past the {deadline:.0f}s heartbeat deadline"
    )


class _Reached(Exception):
    """Ends a start() once the value under test has been recorded."""


def _budget_recorder(budgets):
    """Stands in for a connect method, defaulting exactly as the real one does."""

    async def _connect(retry_budget_seconds: float = RECONNECT_BUDGET_S):
        budgets.append(retry_budget_seconds)
        raise _Reached

    return _connect


async def test_worker_startup_connects_on_the_startup_budget(tmp_path):
    """start() must pass the startup budget; inheriting the default is the bug.

    The reconnect default is deliberately short because the monitoring loop
    retries every tick. Startup has no loop yet, so a silent fall-back to it
    puts the worker back inside the hypha restart gap it exits on.
    """
    budgets = []

    async def _noop(*_args, **_kwargs):
        return None

    worker = BioEngineWorker.__new__(BioEngineWorker)
    worker.logger = _RecordingLogger()
    worker.start_time = None
    worker.heartbeat_file = tmp_path / "worker_heartbeat.json"
    worker._shutdown_event = asyncio.Event()
    worker.ray_cluster = SimpleNamespace(start=_noop)
    worker._connect_to_server = _budget_recorder(budgets)
    worker._stop = _noop

    with pytest.raises(_Reached):
        await worker.start(blocking=False)

    assert budgets == [STARTUP_CONNECT_BUDGET_S]


async def test_ray_startup_connects_on_the_startup_budget(monkeypatch):
    """The Ray client half of the same split, pinned at its own call site."""
    budgets = []

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(ray_cluster_module.ray, "is_initialized", lambda: False)

    cluster = RayCluster.__new__(RayCluster)
    cluster.logger = _RecordingLogger()
    cluster.start_time = None
    cluster.mode = "external-cluster"
    cluster.is_ready = asyncio.Event()
    cluster._set_head_node_address = lambda: None
    cluster._set_serve_http_url = lambda: None
    cluster._connect_to_cluster = _budget_recorder(budgets)
    cluster.stop = _noop

    with pytest.raises(_Reached):
        await cluster.start()

    assert budgets == [STARTUP_CONNECT_BUDGET_S]


async def test_cleanup_cancels_a_pending_startup_retry():
    """A teardown must not leave the startup retry loop deploying into it."""

    async def _never():
        await asyncio.Event().wait()

    task = asyncio.create_task(_never())
    await asyncio.sleep(0)
    manager = SimpleNamespace(_startup_retry_task=task)
    manager.cancel_startup_retry = lambda: AppsManager.cancel_startup_retry(manager)

    worker = BioEngineWorker.__new__(BioEngineWorker)
    worker.logger = _RecordingLogger()
    worker.start_time = None
    worker.apps_manager = manager
    worker.ray_cluster = SimpleNamespace(mode="external-cluster")
    worker._shutdown_event = asyncio.Event()

    await worker._cleanup()
    await asyncio.sleep(0)

    assert task.cancelled()


def test_the_startup_budget_outlasts_the_restart_window_it_exists_for():
    """The startup budget is pinned by value, not only by which symbol is passed.

    The call-site tests above compare against the imported constant, so they
    stay green however it is tuned. The hypha-server restart gap that produced
    the crash-loop was 20-45 s; a budget under that puts the worker straight
    back into it.
    """
    assert STARTUP_CONNECT_BUDGET_S >= 45.0
