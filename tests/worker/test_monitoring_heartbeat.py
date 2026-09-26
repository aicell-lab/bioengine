"""Liveness heartbeat written by the worker's monitoring loop.

A wedged asyncio event loop leaves PID 1 alive and still answering signals, so
a process-level liveness check cannot see it, and it can freeze before the
loop's own consecutive-error counter ever advances. What it always stops is the
monitoring loop completing passes — which is exactly what the heartbeat records.
"""

import asyncio
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import bioengine
from bioengine.heartbeat import (
    DEFAULT_HEARTBEAT_PATH,
    STARTUP_STALE_AFTER_SECONDS,
    check_heartbeat,
    heartbeat_stale_after_seconds,
    main,
    write_heartbeat,
)
from bioengine.worker.__main__ import create_parser
from bioengine.worker.worker import BioEngineWorker


# --------------------------------------------------------------------------
# Deadline derivation
# --------------------------------------------------------------------------


def test_deadline_scales_with_the_monitoring_interval():
    assert heartbeat_stale_after_seconds(20, 0) == 2 * heartbeat_stale_after_seconds(
        10, 0
    )


def test_deadline_tolerates_at_least_one_missed_pass():
    for interval in (1, 10, 60):
        assert heartbeat_stale_after_seconds(interval, 0) > interval


def test_deadline_clears_the_longest_error_backoff():
    # A loop failing every pass sleeps up to backoff_max between passes and is
    # still alive; the deadline must not treat that as frozen.
    assert heartbeat_stale_after_seconds(10, 600) > 600


def test_deadline_clears_a_slow_pass_on_top_of_the_longest_backoff():
    # The real gap under sustained failure is backoff_max, plus the loop's own
    # one-second tick, plus however long the pass takes — and a pass has no
    # bounded duration: the Hypha check alone allows ten seconds.
    interval, backoff_max = 10, 60
    deadline = heartbeat_stale_after_seconds(interval, backoff_max)
    for pass_duration in (10, 15, 30):
        assert deadline > backoff_max + 1 + pass_duration


# --------------------------------------------------------------------------
# Reading the heartbeat, as a liveness probe would
# --------------------------------------------------------------------------


def test_a_just_written_heartbeat_is_fresh(tmp_path):
    path = tmp_path / "worker_heartbeat.json"
    write_heartbeat(path, 30)

    is_fresh, reason = check_heartbeat(path)

    assert is_fresh is True, reason


def test_a_heartbeat_older_than_its_own_deadline_is_stale(tmp_path):
    path = tmp_path / "worker_heartbeat.json"
    write_heartbeat(path, 30)
    payload = json.loads(path.read_text())
    payload["timestamp"] -= payload["stale_after_seconds"] + 1
    path.write_text(json.dumps(payload))

    is_fresh, reason = check_heartbeat(path)

    assert is_fresh is False
    assert "deadline" in reason


def test_the_deadline_travels_inside_the_file(tmp_path):
    # The probe is given only a path, so the same age must be judged against
    # whatever deadline the worker wrote.
    lenient = tmp_path / "lenient.json"
    strict = tmp_path / "strict.json"
    for path, stale_after in ((lenient, 3600), (strict, 1)):
        payload = {"timestamp": time.time() - 60, "stale_after_seconds": stale_after}
        path.write_text(json.dumps(payload))

    assert check_heartbeat(lenient)[0] is True
    assert check_heartbeat(strict)[0] is False


def test_a_missing_heartbeat_is_not_fresh(tmp_path):
    assert check_heartbeat(tmp_path / "never_written.json")[0] is False


def test_a_truncated_heartbeat_is_not_fresh(tmp_path):
    path = tmp_path / "worker_heartbeat.json"
    path.write_text('{"timestamp": 17')

    assert check_heartbeat(path)[0] is False


def test_writing_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "state" / "worker_heartbeat.json"

    write_heartbeat(path, 30)

    assert check_heartbeat(path)[0] is True


def test_cli_exits_zero_when_fresh_and_one_when_missing(tmp_path, capsys):
    path = tmp_path / "worker_heartbeat.json"
    write_heartbeat(path, 30)

    assert main([str(path)]) == 0
    assert main([str(tmp_path / "never_written.json")]) == 1
    assert capsys.readouterr().out.strip() != ""


def test_the_probe_module_pulls_in_neither_ray_nor_hypha():
    # A liveness probe runs on a timeout of a second or two; importing the
    # worker package costs an order of magnitude more than that.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import bioengine.heartbeat, sys; "
            "print(sorted(m for m in ('ray', 'hypha_rpc') if m in sys.modules))",
        ],
        cwd=Path(bioengine.__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "[]"


def test_the_default_heartbeat_path_is_node_local():
    # Not the workspace directory: that is a network volume on at least one
    # cluster, and both the write (on the event loop) and the probe's read
    # would then wait on it.
    assert DEFAULT_HEARTBEAT_PATH.parent == Path(tempfile.gettempdir())


def test_worker_cli_accepts_a_heartbeat_path():
    args = create_parser().parse_args(
        ["--mode", "single-machine", "--heartbeat-file", "/var/run/hb.json"]
    )

    assert args.heartbeat_file == "/var/run/hb.json"


# --------------------------------------------------------------------------
# Where the touch sits in the monitoring loop
# --------------------------------------------------------------------------


# In the order the monitoring loop awaits them; the last one is the last thing
# that has to return before a pass counts as complete.
_STEP_NAMES = (
    "geo_location",
    "hypha_connection",
    "token_expiry",
    "ray_check_connection",
    "cluster_monitoring",
    "data_server_ping",
    "data_server_discover",
    "dataset_refresh",
    "applications_monitoring",
)


class _MonitoringSteps:
    """Stands in for every step the monitoring loop awaits, one mode each."""

    def __init__(self, mode: str = "healthy"):
        self.modes = {name: mode for name in _STEP_NAMES}
        self.calls = {name: 0 for name in _STEP_NAMES}

    def set_all(self, mode: str) -> None:
        self.modes = {name: mode for name in self.modes}

    def step(self, name: str):
        async def _run():
            self.calls[name] += 1
            if self.modes[name] == "hanging":
                await asyncio.Event().wait()
            if self.modes[name] == "failing":
                raise RuntimeError(f"monitoring step '{name}' failed")

        return _run


def _make_worker(tmp_path: Path, steps: _MonitoringSteps) -> BioEngineWorker:
    worker = BioEngineWorker.__new__(BioEngineWorker)
    worker.logger = logging.getLogger("test_monitoring_heartbeat")
    worker.start_time = None
    worker.monitoring_interval_seconds = 0.5
    worker.heartbeat_file = tmp_path / "worker_heartbeat.json"
    worker.is_ready = asyncio.Event()
    worker._last_monitoring = 0
    worker._monitor_consecutive_errors = 0
    worker._monitor_degraded_threshold = 2

    worker._fetch_geo_location = steps.step("geo_location")
    worker._check_hypha_connection = steps.step("hypha_connection")
    worker._check_token_expiry = steps.step("token_expiry")
    worker._ping_data_server = steps.step("data_server_ping")
    worker._discover_data_server = steps.step("data_server_discover")
    worker._refresh_datasets = steps.step("dataset_refresh")
    worker.ray_cluster = SimpleNamespace(
        check_connection=steps.step("ray_check_connection"),
        monitor_cluster=steps.step("cluster_monitoring"),
    )
    worker.apps_manager = SimpleNamespace(
        monitor_applications=steps.step("applications_monitoring")
    )

    async def _no_cleanup():
        return None

    worker._cleanup = _no_cleanup
    return worker


def _start_loop(worker: BioEngineWorker) -> asyncio.Task:
    return asyncio.create_task(
        worker._create_monitoring_task(
            backoff_initial_seconds=0.01, backoff_max_seconds=0.5
        )
    )


async def _stop_loop(task: asyncio.Task) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _wait_until(predicate, timeout: float, message: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    pytest.fail(message)


def _beat(worker: BioEngineWorker) -> dict:
    return json.loads(worker.heartbeat_file.read_text())


def test_startup_replaces_a_previous_runs_deadline(tmp_path):
    # A leftover file must neither be inherited (its deadline could be
    # anything) nor removed (a missing file reads as dead for the whole of a
    # startup that has not reached the monitoring loop yet).
    worker = _make_worker(tmp_path, _MonitoringSteps())
    write_heartbeat(worker.heartbeat_file, 24 * 3600)

    worker._touch_heartbeat(STARTUP_STALE_AFTER_SECONDS)

    assert check_heartbeat(worker.heartbeat_file)[0] is True
    assert _beat(worker)["stale_after_seconds"] == STARTUP_STALE_AFTER_SECONDS


async def test_a_hang_in_the_last_step_never_produces_a_beat(tmp_path):
    # Only the final step hangs, so a pass gets all the way to the end and
    # still never completes. Anything that beats before the last step has
    # returned — the top of the pass included — beats here.
    steps = _MonitoringSteps()
    steps.modes["applications_monitoring"] = "hanging"
    worker = _make_worker(tmp_path, steps)
    task = _start_loop(worker)
    try:
        await _wait_until(
            lambda: steps.calls["applications_monitoring"] > 0,
            10,
            "loop never reached the last monitoring step",
        )
        await asyncio.sleep(2)

        assert steps.calls["geo_location"] == 1
        assert steps.calls["applications_monitoring"] == 1
        assert worker.heartbeat_file.exists() is False
        assert check_heartbeat(worker.heartbeat_file)[0] is False
    finally:
        await _stop_loop(task)


async def test_a_wedged_pass_stops_the_heartbeat(tmp_path):
    steps = _MonitoringSteps()
    worker = _make_worker(tmp_path, steps)
    task = _start_loop(worker)
    try:
        await _wait_until(
            worker.heartbeat_file.exists, 10, "healthy loop wrote no heartbeat"
        )
        stale_after = _beat(worker)["stale_after_seconds"]

        steps.set_all("hanging")
        await asyncio.sleep(2)
        frozen_at = _beat(worker)["timestamp"]
        await asyncio.sleep(stale_after + 1)

        assert _beat(worker)["timestamp"] == frozen_at
        is_fresh, reason = check_heartbeat(worker.heartbeat_file)
        assert is_fresh is False, reason
        # The freeze is invisible to the loop's own error counter: it never
        # reaches the except branch, so nothing internal reports degradation.
        assert worker._monitor_consecutive_errors == 0
    finally:
        await _stop_loop(task)


async def test_failing_steps_keep_the_heartbeat_beating(tmp_path):
    steps = _MonitoringSteps(mode="failing")
    worker = _make_worker(tmp_path, steps)
    task = _start_loop(worker)
    try:
        await _wait_until(
            worker.heartbeat_file.exists, 10, "failing loop wrote no heartbeat"
        )
        first_beat = _beat(worker)["timestamp"]
        await _wait_until(
            lambda: _beat(worker)["timestamp"] > first_beat,
            10,
            "failing loop stopped beating",
        )

        # Every step failed on every pass, past the point where get_status
        # reports not-ready — and the loop is still alive, so it keeps beating.
        assert worker._monitor_consecutive_errors >= worker._monitor_degraded_threshold
        is_fresh, reason = check_heartbeat(worker.heartbeat_file)
        assert is_fresh is True, reason
    finally:
        await _stop_loop(task)
