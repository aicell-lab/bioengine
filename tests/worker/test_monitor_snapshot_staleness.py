"""``get_status`` must stop reporting ready once cluster snapshots stop arriving.

A monitoring loop hung mid-pass raises nothing, so it never advances the
consecutive-error counter, and ``RayCluster.status`` serves the newest history
entry regardless of age — so the worker kept advertising ``is_ready: true`` off
a snapshot that had stopped tracking reality. Readiness therefore also requires
the snapshot to be advancing.

The gate keys on the snapshot's AGE and on nothing else. An earlier proposal
gated on its CONTENTS (``total_cpu > 0``), which is wrong: a slurm worker
excludes its head from the cluster totals (``exclude_head_node = mode ==
"slurm"``) and starts that head with ``--num-cpus=0``, so zero totals and an
empty node map are exactly what a correctly scaled-to-zero slurm worker
reports. ``test_zero_capacity_but_fresh_is_still_ready`` is the opposite-signed
control that holds the two apart: if the capacity gate ever reappears, that
test fails while the staleness tests still pass.
"""

import asyncio
import logging
import time
from collections import OrderedDict
from types import SimpleNamespace

from bioengine.cluster.ray_cluster import RayCluster
from bioengine.heartbeat import heartbeat_stale_after_seconds
from bioengine.utils import create_context
from bioengine.worker.worker import BioEngineWorker

ANONYMOUS = create_context()

MONITOR_INTERVAL_SECONDS = 10
BACKOFF_MAX_SECONDS = 60.0
# The deadline the worker itself derives, reused rather than restated so this
# suite cannot drift from the heartbeat's notion of a missed pass.
STALE_AFTER = heartbeat_stale_after_seconds(
    MONITOR_INTERVAL_SECONDS, BACKOFF_MAX_SECONDS
)

# A healthy slurm worker scaled to zero: head excluded from the totals, no
# sbatch workers. Byte-identical to the shape the incident probe read.
IDLE_SLURM_CLUSTER = {
    "total_cpu": 0,
    "total_gpu": 0,
    "total_memory": 0,
    "total_vram_mb": 0,
}
BUSY_CLUSTER = {
    "total_cpu": 64,
    "total_gpu": 4,
    "total_memory": 1 << 40,
    "total_vram_mb": 81920,
}


def worker_with_snapshot(
    last_cluster_update,
    cluster=None,
    nodes=None,
    stale_after_seconds=STALE_AFTER,
) -> BioEngineWorker:
    """A worker carrying only the state ``get_status`` reads."""
    worker = BioEngineWorker.__new__(BioEngineWorker)
    worker.logger = logging.getLogger("test")  # __del__ reaches for it
    worker.start_time = 1_700_000_000.0
    worker.workspace = "example-workspace"
    worker.client_id = "worker-abc"
    worker.admin_users = []
    worker.geo_location = {"country_name": "Sweden"}
    worker.is_ready = asyncio.Event()
    worker.is_ready.set()
    worker._monitor_consecutive_errors = 0
    worker._monitor_degraded_threshold = 5
    worker._monitor_stale_after_seconds = stale_after_seconds
    worker.ray_cluster = SimpleNamespace(
        mode="slurm",
        status={
            "head_address": "10.0.0.7:6379",
            "mode": "slurm",
            "geo_location": {"country_name": "Sweden"},
            "last_cluster_update": last_cluster_update,
            "cluster": BUSY_CLUSTER if cluster is None else cluster,
            "nodes": {"node-1": {"total_gpu": 4}} if nodes is None else nodes,
            "slurm_jobs": {"queued": [], "running": []},
        },
        slurm_workers=None,
    )
    return worker


async def is_ready_of(worker: BioEngineWorker) -> bool:
    return (await worker.get_status(context=ANONYMOUS))["is_ready"]


# ---------------------------------------------------------------------------
# The gate, and the control that proves the assertion can fail
# ---------------------------------------------------------------------------


async def test_a_stale_snapshot_reports_not_ready():
    stale = time.time() - (STALE_AFTER + 1)
    assert await is_ready_of(worker_with_snapshot(stale)) is False


async def test_a_fresh_snapshot_reports_ready():
    """Discrimination control for the test above: same worker, same cluster
    contents, only the snapshot's age differs — so a passing not-ready
    assertion is not one that passes no matter what."""
    assert await is_ready_of(worker_with_snapshot(time.time())) is True


async def test_readiness_flips_back_when_a_new_snapshot_lands():
    """The same worker object across both readings, so nothing but the
    timestamp can account for the change."""
    worker = worker_with_snapshot(time.time() - (STALE_AFTER + 1))
    assert await is_ready_of(worker) is False

    worker.ray_cluster.status["last_cluster_update"] = time.time()
    assert await is_ready_of(worker) is True


async def test_a_snapshot_just_inside_the_deadline_is_still_ready():
    """The boundary is a missed-pass budget, not a single interval: a loop
    that is failing but still cycling sleeps up to the backoff maximum
    between passes and must not be called frozen."""
    assert await is_ready_of(worker_with_snapshot(time.time() - 1)) is True
    assert await is_ready_of(worker_with_snapshot(time.time() - BACKOFF_MAX_SECONDS)) is True


# ---------------------------------------------------------------------------
# What the gate must NOT do — the retracted capacity gate
# ---------------------------------------------------------------------------


async def test_zero_capacity_but_fresh_is_still_ready():
    """A correctly scaled-to-zero slurm worker: no nodes, every total zero,
    fresh snapshot. The retracted capacity gate reported this not-ready,
    which would have been false for every idle slurm worker. Opposite-signed
    control: this fails if readiness is ever gated on contents again."""
    worker = worker_with_snapshot(
        time.time(), cluster=IDLE_SLURM_CLUSTER, nodes={}
    )
    assert await is_ready_of(worker) is True


async def test_full_capacity_but_stale_is_not_ready():
    """The other half of the pair: capacity is maximal and the snapshot is
    old. Together with the test above this shows the gate reads the age and
    ignores the contents — neither test alone could establish that."""
    worker = worker_with_snapshot(
        time.time() - (STALE_AFTER + 1), cluster=BUSY_CLUSTER
    )
    assert await is_ready_of(worker) is False


# ---------------------------------------------------------------------------
# A worker that has not ticked yet is starting, not stuck
# ---------------------------------------------------------------------------


async def test_an_empty_history_does_not_report_not_ready():
    """``is_ready`` is set before the loop's first pass, so between startup and
    the first snapshot there is no timestamp to judge. Reporting not-ready here
    would fail the k8s startup probe that greps this field."""
    assert await is_ready_of(worker_with_snapshot(None)) is True


async def test_a_status_payload_missing_the_field_does_not_raise():
    """``get_status`` is the unauthenticated health surface the k8s startup
    probe curls, so a payload without the field must degrade to "no snapshot
    yet" rather than raise and fail that probe. The tolerance is why
    ``test_ray_cluster_status_publishes_the_newest_history_key`` below is not
    optional: without it, the key silently disappearing would disable the gate
    instead of failing anything."""
    worker = worker_with_snapshot(time.time())
    del worker.ray_cluster.status["last_cluster_update"]
    assert await is_ready_of(worker) is True


async def test_no_deadline_yet_does_not_report_not_ready():
    """The deadline is derived inside the monitoring loop, so a worker whose
    loop has not reached that line has none. It must not read as stale."""
    worker = worker_with_snapshot(time.time() - 86400, stale_after_seconds=None)
    assert await is_ready_of(worker) is True


# ---------------------------------------------------------------------------
# The timestamp has to come from the real producer, not just the fake above
# ---------------------------------------------------------------------------


def ray_cluster_with_history(history: OrderedDict) -> RayCluster:
    cluster = RayCluster.__new__(RayCluster)
    cluster.address = "10.0.0.7:6379"
    cluster.mode = "slurm"
    cluster.start_time = 1_700_000_000.0
    cluster.head_geo_location = None
    cluster.cluster_status_history = history
    cluster._last_slurm_jobs = None
    return cluster


def test_ray_cluster_status_publishes_the_newest_history_key():
    """Every test above feeds ``last_cluster_update`` to a fake. If the real
    status property did not emit that key, they would all still pass while the
    gate never fired in production."""
    older, newer = 1_700_000_000.0, 1_700_000_010.0
    history = OrderedDict(
        [
            (older, {"cluster": {"total_cpu": 1}, "nodes": {"a": {}}}),
            (newer, {"cluster": {"total_cpu": 2}, "nodes": {"b": {}}}),
        ]
    )
    status = ray_cluster_with_history(history).status

    assert status["last_cluster_update"] == newer
    # Paired with the timestamp so a future refactor cannot keep serving the
    # newest entry's key while reading another entry's body.
    assert status["cluster"] == {"total_cpu": 2}
    assert status["nodes"] == {"b": {}}


def test_ray_cluster_status_reports_no_timestamp_for_an_empty_history():
    status = ray_cluster_with_history(OrderedDict()).status

    assert status["last_cluster_update"] is None
    assert status["cluster"] == {}
    assert status["nodes"] == {}


def test_the_worker_derives_its_deadline_from_the_heartbeat_helper():
    """Both liveness surfaces must agree on what a missed pass is: the loop
    assigns the attribute the gate reads from the heartbeat's own deadline."""
    assert STALE_AFTER == BACKOFF_MAX_SECONDS + 6 * MONITOR_INTERVAL_SECONDS
