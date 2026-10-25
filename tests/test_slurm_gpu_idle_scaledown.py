"""Unit tests for the hardware-side idle check in ``SlurmWorkers``.

``_check_is_idle`` reads Ray resource *reservations*, so a Ray Serve replica that
is alive but serving nothing keeps a GPU worker out of the scale-down candidate
set indefinitely while the physical GPU draws idle power — which is what HPC
sites reap allocations for. ``_sample_gpu_is_busy`` and ``_find_gpu_idle_nodes``
measure the hardware instead; these exercise both against stubs, so no Ray, no
SLURM and no GPU are needed.
"""

import asyncio
import logging
import time
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from bioengine.cluster import slurm_workers as slurm_workers_module
from bioengine.cluster.slurm_workers import SlurmWorkers

logger = logging.getLogger("test")


def _node(job_id, total_gpu=1, used_cpu=16, used_gpu=1):
    return {
        "slurm_job_id": job_id,
        "total_cpu": 16,
        "used_cpu": used_cpu,
        "total_gpu": total_gpu,
        "used_gpu": used_gpu,
    }


# --------------------------------------------------------------------------- #
# _sample_gpu_is_busy
# --------------------------------------------------------------------------- #


class _FakeObjectRef:
    def __init__(self, result):
        self.result = result

    def __await__(self):
        async def resolve():
            if isinstance(self.result, BaseException):
                raise self.result
            if self.result is _HANGS:
                await asyncio.sleep(3600)
            return self.result

        return resolve().__await__()


_HANGS = object()


class _FakeRay:
    """Stand-in for the ``ray`` module as ``_sample_gpu_is_busy`` uses it."""

    def __init__(self, result):
        self.result = result
        self.cancelled = []
        self.options = None

    def remote(self, **options):
        self.options = options
        return lambda fn: SimpleNamespace(remote=lambda: _FakeObjectRef(self.result))

    def cancel(self, obj_ref, force=False):
        self.cancelled.append(obj_ref)


async def _sample(monkeypatch, result, timeout=None):
    fake_ray = _FakeRay(result)
    monkeypatch.setattr(slurm_workers_module, "ray", fake_ray)
    if timeout is not None:
        monkeypatch.setattr(
            slurm_workers_module, "GPU_PROBE_TIMEOUT_SECONDS", timeout
        )
    worker = SimpleNamespace(logger=logger)
    is_busy = await SlurmWorkers._sample_gpu_is_busy(worker, "12345")
    return is_busy, fake_ray


async def test_busy_when_a_gpu_reports_utilization(monkeypatch):
    is_busy, _ = await _sample(monkeypatch, "37, 212.44\n")
    assert is_busy is True


async def test_busy_when_power_is_above_the_site_floor(monkeypatch):
    """Utilization samples a single instant and can read 0 between kernels. The
    power floor is the criterion the site itself reaps on, so staying above it
    is reason enough to keep the node."""
    is_busy, _ = await _sample(monkeypatch, "0, 95.00\n")
    assert is_busy is True


async def test_idle_when_utilization_and_power_are_both_low(monkeypatch):
    """55.70 W is one of the readings NSC terminated a Berzelius job for."""
    is_busy, _ = await _sample(monkeypatch, "0, 55.70\n")
    assert is_busy is False


async def test_busy_if_any_gpu_of_several_is_working(monkeypatch):
    is_busy, _ = await _sample(monkeypatch, "0, 55.70\n88, 301.10\n")
    assert is_busy is True


async def test_power_alone_decides_when_utilization_is_unavailable(monkeypatch):
    is_busy, _ = await _sample(monkeypatch, "[N/A], 61.20\n")
    assert is_busy is False


async def test_unmeasurable_when_no_field_parses(monkeypatch):
    """MIG instances report neither; an unreadable node must not read as idle."""
    is_busy, _ = await _sample(monkeypatch, "[N/A], [N/A]\n")
    assert is_busy is None


async def test_unmeasurable_when_the_probe_raises(monkeypatch):
    is_busy, fake_ray = await _sample(monkeypatch, RuntimeError("node is gone"))
    assert is_busy is None
    assert len(fake_ray.cancelled) == 1


async def test_unmeasurable_when_the_probe_never_returns(monkeypatch):
    is_busy, fake_ray = await _sample(monkeypatch, _HANGS, timeout=0.05)
    assert is_busy is None
    assert len(fake_ray.cancelled) == 1


async def test_probe_books_no_cpu_and_pins_to_the_job(monkeypatch):
    """A node whose CPUs are fully reserved is exactly the case being measured,
    so the probe must not queue behind those reservations."""
    _, fake_ray = await _sample(monkeypatch, "0, 55.70\n")
    assert fake_ray.options["num_cpus"] == 0
    assert "slurm_job_id:12345" in fake_ray.options["resources"]


# --------------------------------------------------------------------------- #
# _find_gpu_idle_nodes
# --------------------------------------------------------------------------- #


class _Sampler:
    def __init__(self, samples, scale_down_threshold=300, check_interval=60):
        self.logger = logger
        self.scale_down_threshold = scale_down_threshold
        self.scale_down_check_interval = check_interval
        self.last_gpu_sample_time = time.time()
        self.gpu_idle_since = {}
        self.worker_deletion_tasks = {}
        self.samples = samples
        self.probed = []

    async def _sample_gpu_is_busy(self, job_id):
        self.probed.append(job_id)
        return self.samples[job_id]


async def _find(worker, nodes):
    return await SlurmWorkers._find_gpu_idle_nodes(worker, nodes)


async def test_reserved_but_unused_gpu_node_is_released_after_the_threshold():
    """The headline case: Ray has booked every CPU and the GPU, so
    ``_check_is_idle`` says busy forever, while the GPU is doing nothing."""
    worker = _Sampler({"1": False})
    nodes = {"n1": _node("1", used_cpu=16, used_gpu=1)}

    assert await SlurmWorkers._check_is_idle(worker, nodes["n1"]) is False

    worker.gpu_idle_since["n1"] = time.time() - 400
    assert await _find(worker, nodes) == {"n1"}


async def test_not_released_before_the_threshold():
    worker = _Sampler({"1": False})
    nodes = {"n1": _node("1")}

    # First observation only starts the clock.
    assert await _find(worker, nodes) == set()

    worker.last_gpu_sample_time = time.time()
    worker.gpu_idle_since["n1"] = time.time() - 120
    assert await _find(worker, nodes) == set()


async def test_a_busy_gpu_restarts_the_clock():
    worker = _Sampler({"1": True})
    worker.gpu_idle_since["n1"] = time.time() - 400
    nodes = {"n1": _node("1")}

    assert await _find(worker, nodes) == set()
    assert "n1" not in worker.gpu_idle_since


async def test_an_unmeasurable_node_is_never_released():
    worker = _Sampler({"1": None})
    worker.gpu_idle_since["n1"] = time.time() - 400
    nodes = {"n1": _node("1")}

    assert await _find(worker, nodes) == set()
    assert "n1" not in worker.gpu_idle_since


async def test_cpu_only_nodes_are_left_to_the_reservation_check():
    """Without a GPU there is no GPU to measure, and nothing for a site to reap
    a GPU allocation over."""
    worker = _Sampler({"1": False})
    nodes = {"n1": _node("1", total_gpu=0, used_gpu=0)}
    worker.gpu_idle_since["n1"] = time.time() - 400

    assert await _find(worker, nodes) == set()
    assert worker.probed == []


async def test_a_node_already_being_closed_is_not_probed_again():
    worker = _Sampler({"1": False})
    worker.worker_deletion_tasks["n1"] = object()
    worker.gpu_idle_since["n1"] = time.time() - 400
    nodes = {"n1": _node("1")}

    assert await _find(worker, nodes) == set()
    assert worker.probed == []


async def test_a_gap_in_sampling_restarts_the_clock():
    """Scale-down checks are skipped while work is pending, so an old idle
    timestamp can span a period the GPU was busy for. Carrying it over would
    cancel a node that resumed work seconds ago."""
    worker = _Sampler({"1": False}, check_interval=60)
    worker.gpu_idle_since["n1"] = time.time() - 400
    worker.last_gpu_sample_time = time.time() - 500
    nodes = {"n1": _node("1")}

    assert await _find(worker, nodes) == set()


async def test_bookkeeping_is_dropped_for_nodes_that_are_gone():
    worker = _Sampler({"1": False})
    worker.gpu_idle_since["vanished"] = time.time() - 400

    assert await _find(worker, {"n1": _node("1")}) == set()
    assert "vanished" not in worker.gpu_idle_since


# --------------------------------------------------------------------------- #
# _check_scale_down
# --------------------------------------------------------------------------- #


class _Scaler:
    def __init__(self, nodes, gpu_idle_nodes=(), reservation_idle_age=600):
        self.logger = logger
        self.min_workers = 0
        self.scale_down_check_interval = 60
        self.scale_down_threshold = 300
        self.last_scale_down_check = 0
        self.worker_deletion_tasks = {}
        self.gpu_idle_nodes = set(gpu_idle_nodes)
        self.closed = []
        now = time.time()
        self.ray_cluster = SimpleNamespace(
            status={"nodes": nodes},
            cluster_status_history=OrderedDict(
                [
                    (now - reservation_idle_age, {"nodes": nodes}),
                    (now - 10, {"nodes": nodes}),
                ]
            ),
        )

    _check_is_idle = SlurmWorkers._check_is_idle
    _find_idle_nodes = SlurmWorkers._find_idle_nodes

    async def get_num_worker_jobs(self):
        return len(self.ray_cluster.status["nodes"])

    async def _find_gpu_idle_nodes(self, worker_nodes):
        return set(self.gpu_idle_nodes)

    async def _close_worker(self, node_id, job_id):
        self.closed.append((node_id, job_id))


async def _scale_down(worker):
    await SlurmWorkers._check_scale_down(worker)
    await asyncio.gather(*list(worker.worker_deletion_tasks.values()))


async def test_a_node_the_reservation_check_rejects_is_still_closed():
    worker = _Scaler({"n1": _node("1", used_cpu=16, used_gpu=1)}, gpu_idle_nodes=["n1"])
    await _scale_down(worker)
    assert worker.closed == [("n1", "1")]


async def test_reservation_idle_nodes_are_still_closed():
    worker = _Scaler({"n1": _node("1", used_cpu=0, used_gpu=0)})
    await _scale_down(worker)
    assert worker.closed == [("n1", "1")]


async def test_a_working_node_is_left_alone():
    worker = _Scaler({"n1": _node("1", used_cpu=16, used_gpu=1)})
    await _scale_down(worker)
    assert worker.closed == []


async def test_a_node_is_not_closed_twice():
    nodes = {"n1": _node("1", used_cpu=0, used_gpu=0)}
    worker = _Scaler(nodes, gpu_idle_nodes=["n1"])
    await _scale_down(worker)
    assert worker.closed == [("n1", "1")]


async def test_nothing_happens_before_the_check_interval_elapses():
    worker = _Scaler({"n1": _node("1", used_cpu=0, used_gpu=0)}, gpu_idle_nodes=["n1"])
    worker.last_scale_down_check = time.time()
    await _scale_down(worker)
    assert worker.closed == []


async def test_min_workers_blocks_scale_down():
    worker = _Scaler({"n1": _node("1", used_cpu=0, used_gpu=0)}, gpu_idle_nodes=["n1"])
    worker.min_workers = 1
    await _scale_down(worker)
    assert worker.closed == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
