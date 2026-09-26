"""Unit tests for the VRAM_MB reservation reported by ``get_cluster_state``.

Drives the plain class behind the Ray actor decorator with stubbed node
resources, so the VRAM_MB branch is exercised without a GPU cluster.
"""

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
