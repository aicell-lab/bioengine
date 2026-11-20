"""Pins the shortfall report ``AppsManager._check_resources`` returns.

On an external cluster the method does not raise when no node has room — it
assumes Ray autoscaling will provide one. When that assumption is false the
deployment waits forever with no failure status and its service never
registers, so the report returned here is the only surface a caller has for
that state. These tests fail if the report stops being returned, loses a
field, or starts being computed cluster-wide instead of per node.

Exercises the method against a stub cluster so no Ray is needed.
"""

import logging

import pytest

from bioengine.apps.manager import PUBLIC_APP_STATUS_FIELDS, AppsManager

logger = logging.getLogger("test")

REQUEST = {"num_cpus": 1, "num_gpus": 1, "memory": 4 * 1024**3, "VRAM_MB": 8000}


class _Ready:
    async def wait(self):
        return


class _SlurmWorkers:
    """Every worker job registered and at max_workers, so neither headroom nor
    an in-flight job can rescue the request."""

    max_workers = 1
    default_num_cpus = 8
    default_num_gpus = 1
    default_mem_in_gb_per_cpu = 4

    async def get_num_worker_jobs(self):
        return 1


class _Cluster:
    def __init__(self, nodes, mode):
        self.mode = mode
        self.status = {"nodes": nodes}
        self.is_ready = _Ready()


class _Manager:
    def __init__(self, nodes, mode="external-cluster"):
        self.logger = logger
        self._deployed_applications = {}
        self.ray_cluster = _Cluster(nodes, mode)


def _node(cpu=8, gpu=1, memory=32 * 1024**3, used_cpu=0, used_gpu=0, used_memory=0):
    return {
        "total_cpu": cpu,
        "used_cpu": used_cpu,
        "total_gpu": gpu,
        "used_gpu": used_gpu,
        "total_memory": memory,
        "used_memory": used_memory,
    }


@pytest.mark.asyncio
async def test_a_node_with_room_reports_no_shortfall():
    mgr = _Manager(nodes={"n1": _node()})
    assert await AppsManager._check_resources(mgr, "app", REQUEST) is None


@pytest.mark.asyncio
async def test_a_full_cluster_returns_the_shortfall_instead_of_raising():
    """The deploy proceeds, so the report is the whole of what a caller gets."""
    mgr = _Manager(nodes={"n1": _node(used_gpu=1)})
    report = await AppsManager._check_resources(mgr, "app", REQUEST)

    assert report is not None
    assert report["requested"] == REQUEST
    assert report["free_per_node"]["n1"]["num_gpus"] == 0
    assert report["cluster_free"]["num_gpus"] == 0
    assert report["detected_at"] > 0


@pytest.mark.asyncio
async def test_a_cluster_scaled_to_zero_reports_an_empty_node_set():
    """The KubeRay case this exists for: no node to land on and, where in-tree
    autoscaling is off, none coming."""
    report = await AppsManager._check_resources(_Manager(nodes={}), "app", REQUEST)

    assert report is not None
    assert report["free_per_node"] == {}
    assert report["cluster_free"] == {"num_cpus": 0, "num_gpus": 0, "memory": 0}


@pytest.mark.asyncio
async def test_aggregate_headroom_is_reported_but_does_not_admit():
    """Two nodes hold four free GPUs between them and neither can host a
    four-GPU replica. ``cluster_free`` is diagnostic only — the moment it
    becomes the admission test, SLURM stops rejecting what it cannot place."""
    mgr = _Manager(nodes={"n1": _node(gpu=2), "n2": _node(gpu=2)})
    request = {**REQUEST, "num_gpus": 4}

    report = await AppsManager._check_resources(mgr, "app", request)

    assert report is not None
    assert report["cluster_free"]["num_gpus"] == 4
    assert report["requested"]["num_gpus"] == 4


@pytest.mark.asyncio
async def test_slurm_still_raises_on_the_same_cluster_state():
    """Clause 2's guard: the report is additive to external-cluster mode and
    changes nothing where the check actually gates a deployment."""
    node = {**_node(used_gpu=1), "slurm_job_id": "1"}
    mgr = _Manager(nodes={"n1": node}, mode="slurm")
    mgr.ray_cluster.slurm_workers = _SlurmWorkers()

    with pytest.raises(ValueError, match="Insufficient resources"):
        await AppsManager._check_resources(mgr, "app", REQUEST)


@pytest.mark.asyncio
async def test_an_aggregate_check_would_admit_what_slurm_must_reject():
    """The measured reason the per-node check stays per-node. Four free GPUs
    across two nodes, none of which can host a four-GPU replica: SLURM rejects
    it, and a cluster-wide sum would admit it onto a cluster that cannot place
    it — the silent wait this row exists to make visible, in the one mode that
    currently fails loudly instead."""
    nodes = {
        "n1": {**_node(gpu=2), "slurm_job_id": "1"},
        "n2": {**_node(gpu=2), "slurm_job_id": "2"},
    }
    mgr = _Manager(nodes=nodes, mode="slurm")
    mgr.ray_cluster.slurm_workers = _SlurmWorkers()

    with pytest.raises(ValueError, match="Insufficient resources"):
        await AppsManager._check_resources(mgr, "app", {**REQUEST, "num_gpus": 4})


def test_the_report_is_published_to_clients():
    """A field filtered out of the public payload is invisible to exactly the
    caller this row is about."""
    assert "resource_shortfall" in PUBLIC_APP_STATUS_FIELDS
