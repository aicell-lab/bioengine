"""Unit tests for the SLURM headroom check in ``AppsManager._check_resources``.

Exercises the method against a stub cluster so no Ray or SLURM is needed.
"""

import logging

import pytest

from bioengine.apps.manager import AppsManager

logger = logging.getLogger("test")

REQUEST = {"num_cpus": 1, "num_gpus": 0.01, "memory": 4 * 1024**3, "VRAM_MB": 8000}


class _Ready:
    async def wait(self):
        return


class _SlurmWorkers:
    def __init__(self, num_worker_jobs, max_workers=1):
        self._num_worker_jobs = num_worker_jobs
        self.max_workers = max_workers
        self.default_num_cpus = 16
        self.default_num_gpus = 1
        self.default_mem_in_gb_per_cpu = 8

    async def get_num_worker_jobs(self):
        return self._num_worker_jobs


class _Cluster:
    mode = "slurm"

    def __init__(self, nodes, slurm_workers):
        self.status = {"nodes": nodes}
        self.slurm_workers = slurm_workers
        self.is_ready = _Ready()


class _Manager:
    def __init__(self, nodes, num_worker_jobs, max_workers=1):
        self.logger = logger
        self._deployed_applications = {}
        self.ray_cluster = _Cluster(nodes, _SlurmWorkers(num_worker_jobs, max_workers))


def _node(job_id, cpu=16, gpu=1, memory=128 * 1024**3):
    return {
        "slurm_job_id": job_id,
        "total_cpu": cpu,
        "used_cpu": 0,
        "total_gpu": gpu,
        "used_gpu": 0,
        "total_memory": memory,
        "used_memory": 0,
    }


@pytest.mark.asyncio
async def test_empty_cluster_can_scale_up():
    mgr = _Manager(nodes={}, num_worker_jobs=0, max_workers=1)
    await AppsManager._check_resources(mgr, "app", REQUEST)


@pytest.mark.asyncio
async def test_in_flight_job_counts_as_capacity():
    """A submitted job that has not registered as a Ray node yet is capacity on
    the way. Deploying on a cluster scaled to zero triggers that scale-up
    itself, so counting the job against max_workers rejected the very
    deployment it was provisioned for."""
    mgr = _Manager(nodes={}, num_worker_jobs=1, max_workers=1)
    await AppsManager._check_resources(mgr, "app", REQUEST)


@pytest.mark.asyncio
async def test_registered_node_with_room_is_accepted():
    mgr = _Manager(nodes={"n1": _node("1")}, num_worker_jobs=1, max_workers=1)
    await AppsManager._check_resources(mgr, "app", REQUEST)


@pytest.mark.asyncio
async def test_full_cluster_rejects_oversized_request():
    """Every worker job is registered and none can fit the request, so there is
    no capacity in flight and no headroom to spawn more."""
    mgr = _Manager(nodes={"n1": _node("1")}, num_worker_jobs=1, max_workers=1)
    oversized = {**REQUEST, "num_cpus": 64}
    with pytest.raises(ValueError, match="Insufficient resources"):
        await AppsManager._check_resources(mgr, "app", oversized)
