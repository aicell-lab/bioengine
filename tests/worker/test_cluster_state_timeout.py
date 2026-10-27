"""Bound on the monitoring loop's cluster-state reads.

A Ray head that stops answering without dying leaves
``proxy_actor_handle.get_cluster_state.remote()`` pending forever. The
monitoring tick awaits its steps in sequence, so an untimed read there starves
application monitoring and token renewal behind it — and because the loop never
reaches its error handler, nothing internal reports degradation either.

The harness for the loop-level case is the one in ``test_monitoring_heartbeat``;
pytest puts this directory on ``sys.path``, so it imports by module name.
"""

import asyncio
import inspect
import logging
import time
from pathlib import Path

import pytest
from test_monitoring_heartbeat import (
    _MonitoringSteps,
    _make_worker,
    _start_loop,
    _stop_loop,
    _wait_until,
)

from bioengine.cluster import ray_cluster
from bioengine.cluster.ray_cluster import RayCluster
from bioengine.heartbeat import heartbeat_stale_after_seconds
from bioengine.utils import CLUSTER_STATE_TIMEOUT_S, RECONNECT_BUDGET_S
from bioengine.worker.worker import BioEngineWorker


class _RemoteMethod:
    def __init__(self, handle: "_ProxyHandle"):
        self._handle = handle

    def remote(self):
        return self._handle._call()


class _ProxyHandle:
    """Stands in for the proxy actor handle the monitor holds."""

    def __init__(self, behaviour: str, state: dict = None):
        self.behaviour = behaviour
        self.state = state if state is not None else {"nodes": {}}
        self.calls = 0
        self.cancelled = 0
        self.get_cluster_state = _RemoteMethod(self)

    async def _call(self):
        self.calls += 1
        if self.behaviour == "silent":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        if self.behaviour == "stale":
            raise Exception("ray client doesn't have a handle for actor 0xdeadbeef")
        return self.state


def _make_cluster(handle: _ProxyHandle, after_reconnect: _ProxyHandle = None):
    cluster = RayCluster.__new__(RayCluster)
    cluster.logger = logging.getLogger("test_cluster_state_timeout")
    cluster.proxy_actor_handle = handle
    cluster.reconnects = 0

    async def _reconnect():
        cluster.reconnects += 1
        if after_reconnect is not None:
            cluster.proxy_actor_handle = after_reconnect

    cluster.reconnect = _reconnect
    return cluster


async def test_a_head_that_never_answers_is_cut_short(monkeypatch):
    monkeypatch.setattr(ray_cluster, "CLUSTER_STATE_TIMEOUT_S", 0.2)
    handle = _ProxyHandle("silent")
    cluster = _make_cluster(handle)

    started = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        # The outer bound keeps an unbounded read from hanging the suite, but it
        # raises TimeoutError itself — so the exception type proves nothing here.
        # What discriminates is returning early and the inner call being cancelled.
        await asyncio.wait_for(cluster._get_cluster_state_with_reconnect(), 5.0)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"the read was not bounded (took {elapsed:.1f}s)"
    assert handle.cancelled == 1
    # A head that has gone quiet is not a stale handle, so no reconnect is tried.
    assert cluster.reconnects == 0


async def test_the_retry_after_a_reconnect_is_bounded_too(monkeypatch):
    # Bounding only the first read leaves the recovery path able to hang, and
    # that is the path a recycled head actually takes.
    monkeypatch.setattr(ray_cluster, "CLUSTER_STATE_TIMEOUT_S", 0.2)
    silent = _ProxyHandle("silent")
    cluster = _make_cluster(_ProxyHandle("stale"), after_reconnect=silent)

    started = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(cluster._get_cluster_state_with_reconnect(), 5.0)
    elapsed = time.monotonic() - started

    assert cluster.reconnects == 1
    assert silent.cancelled == 1
    assert elapsed < 1.0, f"the retry was not bounded (took {elapsed:.1f}s)"


async def test_a_healthy_read_is_not_cut():
    # Discrimination control for the two tests above: with the shipped bound and
    # no monkeypatch, a normal read passes through untouched. Measured healthy
    # calls are milliseconds, including with pending resources to collect.
    handle = _ProxyHandle("healthy", state={"nodes": {"head": {"CPU": 8}}})
    cluster = _make_cluster(handle)

    assert await cluster._get_cluster_state_with_reconnect() == {
        "nodes": {"head": {"CPU": 8}}
    }
    assert handle.cancelled == 0


async def test_a_timed_out_read_does_not_starve_the_steps_behind_it(tmp_path):
    steps = _MonitoringSteps()
    worker = _make_worker(tmp_path, steps)

    async def _times_out():
        steps.calls["cluster_monitoring"] += 1
        raise asyncio.TimeoutError()

    worker.ray_cluster.monitor_cluster = _times_out
    task = _start_loop(worker)
    try:
        await _wait_until(
            lambda: worker._monitor_consecutive_errors
            >= worker._monitor_degraded_threshold,
            10,
            "a timed-out cluster read never reached the per-step error path",
        )

        # The point of the bound: the read gives up, the pass carries on to
        # application monitoring, and the streak degrades is_ready. An unbounded
        # read reaches none of the three.
        assert steps.calls["applications_monitoring"] > 0
        assert worker.heartbeat_file.exists() is True
    finally:
        await _stop_loop(task)


def test_the_bound_leaves_room_inside_the_staleness_deadline():
    # The worst bounded pass through the recovery path is two cut reads with a
    # reconnect between them; that has to fit inside the deadline the liveness
    # probe judges the heartbeat against, with room left for the rest of a pass.
    interval = (
        inspect.signature(BioEngineWorker.__init__)
        .parameters["monitoring_interval_seconds"]
        .default
    )
    backoff_max = (
        inspect.signature(BioEngineWorker._create_monitoring_task)
        .parameters["backoff_max_seconds"]
        .default
    )

    assert (
        2 * CLUSTER_STATE_TIMEOUT_S + RECONNECT_BUDGET_S
        < heartbeat_stale_after_seconds(interval, backoff_max)
    )


def test_every_cluster_state_read_in_the_module_is_bounded():
    # The read in _connect_to_cluster is reachable from the monitoring tick via
    # reconnect(), but exercising it needs a real Ray cluster — so it is covered
    # here at the source level, along with any read added later.
    lines = Path(ray_cluster.__file__).read_text().splitlines()
    unbounded = [
        number
        for number, line in enumerate(lines, start=1)
        if "get_cluster_state.remote()" in line
        and not any("asyncio.wait_for(" in lines[i] for i in range(number - 4, number - 1))
    ]

    assert not unbounded, f"unbounded get_cluster_state reads at lines {unbounded}"
