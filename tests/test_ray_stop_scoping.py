"""Unit tests for scoping the Ray shutdown to the cluster that owns it.

``ray stop`` takes no cluster selector and kills every Ray process the invoking
user owns on the node, so on a shared HPC login node stopping one BioEngine
worker silently kills every other worker of the same account.
``RayCluster._find_cluster_processes`` is what replaces it: it claims only the
processes belonging to this cluster. These tests drive it against a stubbed
process table, so no Ray cluster is needed.
"""

import logging

import pytest

from bioengine.cluster import ray_cluster as ray_cluster_module
from bioengine.cluster.ray_cluster import RayCluster

OUR_TEMP_DIR = "/home/u/.bioengine/ray"
OUR_SESSION = f"{OUR_TEMP_DIR}/session_2026-10-25_08-41-39_029528_27"
THEIR_TEMP_DIR = "/home/u/.bioengine/ray-other"
THEIR_SESSION = f"{THEIR_TEMP_DIR}/session_2026-10-25_02-34-02_118802_9"


class _FakeProcess:
    def __init__(self, pid, name, cmdline, ppid=1):
        self.pid = pid
        self.info = {"name": name, "cmdline": cmdline.split(" "), "ppid": ppid}

    def cmdline(self):
        return self.info["cmdline"]


def _cluster(table, temp_dir=OUR_TEMP_DIR, port=6379):
    cluster = RayCluster.__new__(RayCluster)
    cluster.logger = logging.getLogger("test")
    cluster.ray_cluster_config = {
        "ray_temp_dir": temp_dir,
        "head_node_address": "10.0.0.1",
        "ports": {"head_node": port},
    }
    return cluster


@pytest.fixture
def process_table(monkeypatch):
    """Install a stubbed process table and return the claimed pids."""

    def install(processes):
        monkeypatch.setattr(
            ray_cluster_module.psutil, "process_iter", lambda attrs: list(processes)
        )
        buckets = _cluster(processes)._find_cluster_processes()
        return buckets, {p.pid for bucket in buckets for p in bucket}

    return install


def _raylet(pid, session, ppid=1):
    return _FakeProcess(
        pid, "raylet", f"/sp/ray/core/src/ray/raylet/raylet --raylet_socket_name={session}/sockets/raylet", ppid
    )


def _gcs(pid, session):
    return _FakeProcess(
        pid, "gcs_server", f"/sp/ray/core/src/ray/gcs/gcs_server --log_dir={session}/logs"
    )


def test_claims_a_process_naming_our_temp_dir(process_table):
    _, claimed = process_table([_gcs(10, OUR_SESSION)])
    assert claimed == {10}


def test_leaves_another_clusters_processes_alone(process_table):
    """The whole point: a second worker on the same node must survive."""
    _, claimed = process_table(
        [_gcs(10, OUR_SESSION), _gcs(20, THEIR_SESSION), _raylet(21, THEIR_SESSION)]
    )
    assert claimed == {10}


def test_claims_the_client_server_by_gcs_address(process_table):
    """``ray.util.client.server`` names the GCS address, never the temp dir."""
    client = _FakeProcess(
        11,
        "python3.11",
        "/usr/bin/python3.11 -m ray.util.client.server --address=10.0.0.1:6379 --port=10001",
    )
    _, claimed = process_table([client])
    assert claimed == {11}


def test_does_not_claim_another_clusters_client_server(process_table):
    client = _FakeProcess(
        21,
        "python3.11",
        "/usr/bin/python3.11 -m ray.util.client.server --address=10.0.0.1:6380 --port=10001",
    )
    _, claimed = process_table([client])
    assert claimed == set()


def test_claims_renamed_workers_through_their_raylet(process_table):
    """``ray::`` workers name no cluster; only their parent raylet does."""
    processes = [
        _raylet(30, OUR_SESSION),
        _FakeProcess(31, "ray::Parked", "ray::Parked", ppid=30),
        _FakeProcess(32, "ray::IDLE", "ray::IDLE", ppid=30),
    ]
    _, claimed = process_table(processes)
    assert claimed == {30, 31, 32}


def test_does_not_claim_another_raylets_workers(process_table):
    processes = [
        _raylet(30, OUR_SESSION),
        _raylet(40, THEIR_SESSION),
        _FakeProcess(41, "ray::Parked", "ray::Parked", ppid=40),
    ]
    _, claimed = process_table(processes)
    assert claimed == {30}


def test_claims_a_grandchild_worker(process_table):
    """The adoption walk repeats, so a worker's own child is not left behind."""
    processes = [
        _raylet(30, OUR_SESSION),
        _FakeProcess(31, "ray::Parked", "ray::Parked", ppid=30),
        _FakeProcess(32, "ray::IDLE", "ray::IDLE", ppid=31),
    ]
    _, claimed = process_table(processes)
    assert claimed == {30, 31, 32}


def test_does_not_claim_the_worker_process_that_is_doing_the_stopping(process_table):
    """The BioEngine worker's own argv carries --ray-temp-dir, and killing it
    mid-shutdown would be this fix shooting its own caller. It survives because
    it matches no RAY_PROCESSES keyword, not because of a pid check."""
    worker = _FakeProcess(
        5,
        "python3.11",
        f"/usr/bin/python3.11 -m bioengine worker --mode=slurm --ray-temp-dir {OUR_TEMP_DIR}",
    )
    _, claimed = process_table([worker, _gcs(10, OUR_SESSION)])
    assert claimed == {10}


def test_orders_the_buckets_raylet_first_and_gcs_last(process_table):
    """Ray requires this order: the raylet must exit before the agents that
    fate-share with it, and the GCS must exit last."""
    processes = [
        _gcs(10, OUR_SESSION),
        _FakeProcess(
            11, "python3.11", f"/sp/ray/_private/log_monitor.py --session-dir={OUR_SESSION}"
        ),
        _raylet(12, OUR_SESSION),
    ]
    buckets, claimed = process_table(processes)
    assert claimed == {10, 11, 12}
    assert [p.pid for p in buckets[0]] == [12]
    assert [p.pid for p in buckets[1]] == [11]
    assert [p.pid for p in buckets[2]] == [10]


def test_claims_nothing_when_no_ray_process_is_running(process_table):
    unrelated = _FakeProcess(7, "sshd", "/usr/sbin/sshd -D")
    buckets, claimed = process_table([unrelated])
    assert claimed == set()
    assert buckets == [[], [], []]
