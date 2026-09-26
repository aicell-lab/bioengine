"""Unit tests for the VRAM_MB reservation reported by ``get_cluster_state``
and rendered by ``bioengine cluster status``.

Drives the plain class behind the Ray actor decorator with stubbed node
resources, so the VRAM_MB branch is exercised without a GPU cluster.
"""

from click.testing import CliRunner

from bioengine.cli import cluster as cluster_cli
from bioengine.cluster.proxy_actor import BioEngineProxyActor

_PLAIN_CLASS = BioEngineProxyActor.__ray_metadata__.modified_class

#: One Europa-shaped head node: an RTX 3090 advertising its VRAM, with four
#: federated-unet replicas holding 5120 MB and a 0.01 GPU handle each.
_HEAD = {
    "node:172.17.0.3": 1.0,
    "node:__internal_head__": 0.001,
    "CPU": 8.0,
    "GPU": 1.0,
    "VRAM_MB": 24576.0,
    "memory": 32212254720.0,
    "object_store_memory": 10000000000.0,
}
_HEAD_AVAILABLE = {
    "CPU": 0.0,
    "GPU": 0.96,
    "VRAM_MB": 4096.0,
    "memory": 4294967296.0,
    "object_store_memory": 9999126230.0,
}


def _actor(total, available):
    actor = object.__new__(_PLAIN_CLASS)
    actor.exclude_head_node = False
    actor.check_pending_resources = False
    actor.node_gpu_memory = {}
    actor.global_state = type(
        "GlobalState",
        (),
        {
            "total_resources_per_node": staticmethod(lambda: {"n1": total}),
            "available_resources_per_node": staticmethod(lambda: {"n1": available}),
        },
    )()
    # No dashboard URL and no cached snapshot, so the real GPU-info fetch
    # returns "nothing known" instead of being stubbed out — a stub of that
    # helper goes stale silently when it is renamed.
    actor.dashboard_url = None
    actor._last_per_node_gpu_info = {}
    return actor


def test_vram_booking_is_reported_where_the_gpu_fraction_is_not():
    status = _actor(_HEAD, _HEAD_AVAILABLE).get_cluster_state()

    node = status["nodes"]["n1"]
    assert node["total_vram_mb"] == 24576.0
    assert node["used_vram_mb"] == 20480.0
    # The reason the field exists: 83% of the GPU is booked and used_gpu says 4%.
    assert round(node["used_gpu"], 2) == 0.04

    assert status["cluster"]["total_vram_mb"] == 24576.0
    assert status["cluster"]["used_vram_mb"] == 20480.0


def test_no_vram_resource_reports_zero_rather_than_inventing_capacity():
    total = {k: v for k, v in _HEAD.items() if k != "VRAM_MB"}
    available = {k: v for k, v in _HEAD_AVAILABLE.items() if k != "VRAM_MB"}
    available["GPU"] = 0.67

    status = _actor(total, available).get_cluster_state()

    node = status["nodes"]["n1"]
    assert node["total_vram_mb"] == 0
    assert node["used_vram_mb"] == 0
    # Where nothing advertises VRAM_MB the GPU fraction is the real reservation,
    # so a zero here must not read as an idle GPU.
    assert round(node["used_gpu"], 2) == 0.33


def _status_output(node_info, monkeypatch):
    async def _connect_worker(server_url, worker_service_id, token):
        class _Worker:
            async def get_status(self):
                return {
                    "ray_cluster": {
                        "cluster": {"used_cpu": 8, "total_cpu": 8},
                        "nodes": {"n1": node_info},
                    }
                }

        return _Worker()

    monkeypatch.setattr(cluster_cli, "get_server_url", lambda url: "https://example")
    monkeypatch.setattr(cluster_cli, "get_token", lambda token: "t")
    monkeypatch.setattr(cluster_cli, "connect_worker", _connect_worker)

    result = CliRunner().invoke(cluster_cli.cluster_status, ["--worker", "ws/svc"])
    assert result.exit_code == 0, result.output
    return result.output


_GPU_NODE = {
    "node_ip": "172.17.0.3",
    "head": True,
    "total_cpu": 8,
    "used_cpu": 8,
    "total_gpu": 1.0,
    "used_gpu": 0.04,
    "total_gpu_memory": "NA",
    "used_gpu_memory": "NA",
    "accelerator_type": "G",
    "gpu_device_name": "NVIDIA GeForce RTX 3090",
}


def test_status_line_shows_the_booking_and_the_real_device_name(monkeypatch):
    output = _status_output(
        {**_GPU_NODE, "total_vram_mb": 24576.0, "used_vram_mb": 20480.0}, monkeypatch
    )

    assert "booked: 20480/24576 MB" in output
    # A 0.01-handle booking rounds to 0.0 at one decimal, which is the reading
    # the booking figure exists to correct.
    assert "GPU: 0.04/1" in output
    assert "NVIDIA GeForce RTX 3090 (accelerator_type=G)" in output


def test_status_line_omits_the_booking_where_no_vram_is_advertised(monkeypatch):
    output = _status_output(
        {**_GPU_NODE, "used_gpu": 0.33, "total_vram_mb": 0, "used_vram_mb": 0},
        monkeypatch,
    )

    assert "booked:" not in output
    assert "GPU: 0.33/1" in output
