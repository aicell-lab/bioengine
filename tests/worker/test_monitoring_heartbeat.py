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
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import bioengine
from bioengine.heartbeat import (
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


def test_worker_cli_accepts_a_heartbeat_path():
    args = create_parser().parse_args(
        ["--mode", "single-machine", "--heartbeat-file", "/var/run/hb.json"]
    )

    assert args.heartbeat_file == "/var/run/hb.json"


# --------------------------------------------------------------------------
# Where the touch sits in the monitoring loop
# --------------------------------------------------------------------------


class _MonitoringSteps:
    """Stands in for every step the monitoring loop awaits."""

    def __init__(self, mode: str = "healthy"):
        self.mode = mode

    async def __call__(self):
        if self.mode == "hanging":
            await asyncio.Event().wait()
        if self.mode == "failing":
            raise RuntimeError("monitoring step failed")


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

    worker._fetch_geo_location = steps
    worker._check_hypha_connection = steps
    worker._check_token_expiry = steps
    worker._ping_data_server = steps
    worker._discover_data_server = steps
    worker._refresh_datasets = steps
    worker.ray_cluster = SimpleNamespace(check_connection=steps, monitor_cluster=steps)
    worker.apps_manager = SimpleNamespace(monitor_applications=steps)

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


def test_a_previous_runs_heartbeat_does_not_survive_as_fresh(tmp_path):
    worker = _make_worker(tmp_path, _MonitoringSteps())
    write_heartbeat(worker.heartbeat_file, 3600)
    assert check_heartbeat(worker.heartbeat_file)[0] is True

    worker._clear_heartbeat()

    assert check_heartbeat(worker.heartbeat_file)[0] is False


async def test_a_wedged_pass_stops_the_heartbeat(tmp_path):
    steps = _MonitoringSteps()
    worker = _make_worker(tmp_path, steps)
    task = _start_loop(worker)
    try:
        await _wait_until(
            worker.heartbeat_file.exists, 10, "healthy loop wrote no heartbeat"
        )
        stale_after = _beat(worker)["stale_after_seconds"]

        steps.mode = "hanging"
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
