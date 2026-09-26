"""Unit tests for how a node's GPU identity is reported in the cluster state.

Covers Ray's schedulable ``accelerator_type`` label and the separate,
untruncated ``gpu_device_name`` read from the Ray dashboard node summary.
Everything is exercised against plain dicts and a stubbed dashboard payload, so
no Ray cluster and no NVIDIA hardware are needed.
"""

import json
import urllib.error

import pytest

from bioengine.cli.cluster import _format_gpu_label
from bioengine.cluster import proxy_actor as proxy_actor_module
from bioengine.cluster.proxy_actor import BioEngineProxyActor

_ProxyActor = BioEngineProxyActor.__ray_metadata__.modified_class
_get_accelerator_type = _ProxyActor._get_accelerator_type


def test_reads_the_accelerator_type_resource():
    resources = {"CPU": 8.0, "GPU": 1.0, "accelerator_type:A40": 1.0}
    assert _get_accelerator_type(None, resources) == "A40"


def test_type_starting_with_a_prefix_character_survives():
    # str.lstrip("accelerator_type:") strips the character *set*, so a type
    # whose first characters all appear in the prefix loses them.
    resources = {"GPU": 1.0, "accelerator_type:tesla": 1.0}
    assert _get_accelerator_type(None, resources) == "tesla"


def test_returns_none_without_an_accelerator_resource():
    assert _get_accelerator_type(None, {"CPU": 8.0}) is None


class _FakeGlobalState:
    def __init__(self, totals):
        self._totals = totals

    def total_resources_per_node(self):
        return self._totals

    def available_resources_per_node(self):
        return {node_id: dict(res) for node_id, res in self._totals.items()}


def _actor(totals, dashboard_payload=None, dashboard_error=None, monkeypatch=None):
    actor = object.__new__(_ProxyActor)
    actor.global_state = _FakeGlobalState(totals)
    actor.exclude_head_node = False
    actor.check_pending_resources = False
    actor.node_gpu_memory = {}
    actor.dashboard_url = "http://127.0.0.1:8265"
    actor._last_per_node_gpu_info = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            return json.dumps(dashboard_payload).encode("utf-8")

    def _urlopen(request, timeout=None):
        if dashboard_error is not None:
            raise dashboard_error
        return _Response()

    monkeypatch.setattr(proxy_actor_module.urllib.request, "urlopen", _urlopen)
    return actor


def _payload(node_id, gpus):
    return {
        "result": True,
        "data": {"summary": [{"raylet": {"nodeId": node_id}, "gpus": gpus}]},
    }


def _gpu(name, memory_total=24576, memory_used=0):
    return {"name": name, "memoryTotal": memory_total, "memoryUsed": memory_used}


def test_consumer_gpu_reports_its_full_device_name(monkeypatch):
    totals = {"n1": {"CPU": 8.0, "GPU": 1.0, "accelerator_type:G": 1.0}}
    actor = _actor(
        totals,
        dashboard_payload=_payload("n1", [_gpu("NVIDIA GeForce RTX 3090")]),
        monkeypatch=monkeypatch,
    )

    node = actor.get_cluster_state()["nodes"]["n1"]

    assert node["gpu_device_name"] == "NVIDIA GeForce RTX 3090"


def test_the_schedulable_label_is_left_untouched(monkeypatch):
    totals = {"n1": {"CPU": 8.0, "GPU": 1.0, "accelerator_type:G": 1.0}}
    actor = _actor(
        totals,
        dashboard_payload=_payload("n1", [_gpu("NVIDIA GeForce RTX 3090")]),
        monkeypatch=monkeypatch,
    )

    node = actor.get_cluster_state()["nodes"]["n1"]

    assert node["accelerator_type"] == "G"


def test_identical_devices_collapse_to_one_name(monkeypatch):
    totals = {"n1": {"CPU": 8.0, "GPU": 2.0, "accelerator_type:A40": 1.0}}
    actor = _actor(
        totals,
        dashboard_payload=_payload("n1", [_gpu("NVIDIA A40"), _gpu("NVIDIA A40")]),
        monkeypatch=monkeypatch,
    )

    assert actor.get_cluster_state()["nodes"]["n1"]["gpu_device_name"] == "NVIDIA A40"


def test_mixed_devices_report_every_name(monkeypatch):
    totals = {"n1": {"CPU": 8.0, "GPU": 2.0, "accelerator_type:A40": 1.0}}
    actor = _actor(
        totals,
        dashboard_payload=_payload(
            "n1", [_gpu("NVIDIA A40"), _gpu("NVIDIA GeForce RTX 3090")]
        ),
        monkeypatch=monkeypatch,
    )

    assert (
        actor.get_cluster_state()["nodes"]["n1"]["gpu_device_name"]
        == "NVIDIA A40, NVIDIA GeForce RTX 3090"
    )


def test_cpu_only_node_reports_na(monkeypatch):
    totals = {"n1": {"CPU": 8.0}}
    actor = _actor(
        totals, dashboard_payload=_payload("n1", []), monkeypatch=monkeypatch
    )

    node = actor.get_cluster_state()["nodes"]["n1"]

    assert node["gpu_device_name"] == "NA"
    assert node["accelerator_type"] == "NA"


def test_gpu_node_without_nvidia_libraries_still_reports_status(monkeypatch):
    """No NVIDIA libraries on the node means the dashboard lists no GPU devices;
    the status must still come back, just without a device name."""
    totals = {"n1": {"CPU": 8.0, "GPU": 1.0, "accelerator_type:G": 1.0}}
    actor = _actor(
        totals, dashboard_payload=_payload("n1", []), monkeypatch=monkeypatch
    )

    node = actor.get_cluster_state()["nodes"]["n1"]

    assert node["gpu_device_name"] is None
    assert node["accelerator_type"] == "G"


def test_unreachable_dashboard_still_reports_status(monkeypatch):
    totals = {"n1": {"CPU": 8.0, "GPU": 1.0, "accelerator_type:G": 1.0}}
    actor = _actor(
        totals,
        dashboard_error=urllib.error.URLError("refused"),
        monkeypatch=monkeypatch,
    )

    node = actor.get_cluster_state()["nodes"]["n1"]

    assert node["gpu_device_name"] is None
    assert node["accelerator_type"] == "G"


@pytest.mark.parametrize(
    "info, expected",
    [
        (
            {"accelerator_type": "G", "gpu_device_name": "NVIDIA GeForce RTX 3090"},
            "NVIDIA GeForce RTX 3090 (accelerator_type=G)",
        ),
        ({"accelerator_type": "A40", "gpu_device_name": None}, "A40"),
        ({"accelerator_type": "A40"}, "A40"),
        ({}, "?"),
    ],
)
def test_cli_label_names_the_schedulable_value(info, expected):
    assert _format_gpu_label(info) == expected
