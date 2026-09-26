"""Unit tests for the startup warning about an unbackable head memory reservation.

All host figures are injected, so nothing here depends on the memory of the
machine running the tests.
"""

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

import pytest

from bioengine.cluster.ray_cluster import RayCluster
from bioengine.utils.host_memory import head_memory_budget_warning, read_meminfo

GIB = 1024**3

# The co-tenancy reported on a 62.49 GiB host: one worker reserved 40 GB and a
# build service with no container of its own held a further 17 GiB by the time
# the second worker started.
HOST_TOTAL = int(62.49 * GIB)
HOST_AVAILABLE_WITH_CO_TENANTS = HOST_TOTAL - int(57 * GIB)
HOST_AVAILABLE_WHEN_IDLE = HOST_TOTAL - int(2 * GIB)


def test_warns_when_reservation_plus_co_tenants_exceeds_budget():
    warning = head_memory_budget_warning(
        reserved_gb=30,
        mem_total_bytes=HOST_TOTAL,
        mem_available_bytes=HOST_AVAILABLE_WITH_CO_TENANTS,
        budget_fraction=0.9,
    )
    assert warning is not None


def test_no_warning_when_reservation_fits_the_budget():
    warning = head_memory_budget_warning(
        reserved_gb=30,
        mem_total_bytes=HOST_TOTAL,
        mem_available_bytes=HOST_AVAILABLE_WHEN_IDLE,
        budget_fraction=0.9,
    )
    assert warning is None


def test_no_warning_exactly_at_the_budget():
    warning = head_memory_budget_warning(
        reserved_gb=90,
        mem_total_bytes=100 * GIB,
        mem_available_bytes=100 * GIB,
        budget_fraction=0.9,
    )
    assert warning is None


def test_one_gib_past_the_budget_warns():
    warning = head_memory_budget_warning(
        reserved_gb=91,
        mem_total_bytes=100 * GIB,
        mem_available_bytes=100 * GIB,
        budget_fraction=0.9,
    )
    assert warning is not None


def test_default_budget_is_stricter_than_the_whole_host():
    """A reservation that fits MemTotal but leaves the host no headroom warns."""
    arguments = dict(
        reserved_gb=95,
        mem_total_bytes=100 * GIB,
        mem_available_bytes=100 * GIB,
    )
    assert head_memory_budget_warning(**arguments, budget_fraction=0.9) is not None
    assert head_memory_budget_warning(**arguments, budget_fraction=1.0) is None


def test_zero_fraction_disables_the_check():
    warning = head_memory_budget_warning(
        reserved_gb=1000,
        mem_total_bytes=HOST_TOTAL,
        mem_available_bytes=0,
        budget_fraction=0,
    )
    assert warning is None


def test_co_tenant_occupancy_alone_can_trip_the_budget():
    """A modest reservation on an already loaded host still warns."""
    warning = head_memory_budget_warning(
        reserved_gb=4,
        mem_total_bytes=100 * GIB,
        mem_available_bytes=10 * GIB,
        budget_fraction=0.9,
    )
    assert warning is not None


def test_warning_reports_the_reservation_the_occupancy_and_the_total():
    warning = head_memory_budget_warning(
        reserved_gb=30,
        mem_total_bytes=100 * GIB,
        mem_available_bytes=35 * GIB,
        budget_fraction=0.9,
    )
    assert "30.0 GiB" in warning
    assert "65.0 GiB" in warning
    assert "95.0 GiB" in warning
    assert "100.0 GiB" in warning
    assert "90.0 GiB" in warning
    assert "90%" in warning


def test_warning_describes_a_broken_guarantee_not_imminent_exhaustion():
    warning = head_memory_budget_warning(
        reserved_gb=30,
        mem_total_bytes=HOST_TOTAL,
        mem_available_bytes=HOST_AVAILABLE_WITH_CO_TENANTS,
        budget_fraction=0.9,
    )
    assert "guarantee" in warning
    assert "not a prediction that the host will run out of memory" in warning


def test_read_meminfo_converts_kilobytes_to_bytes(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       65523056 kB\n"
        "MemFree:            1234 kB\n"
        "MemAvailable:   40357352 kB\n"
        "HugePages_Total:       0\n"
        "DirectMap4k:        junk kB\n"
    )
    fields = read_meminfo(meminfo)
    assert fields["MemTotal"] == 65523056 * 1024
    assert fields["MemAvailable"] == 40357352 * 1024
    assert fields["HugePages_Total"] == 0
    assert "DirectMap4k" not in fields


class _WarningCollector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def short_temp_dir():
    """Ray rejects temp dirs long enough to push its socket path past 107 bytes."""
    path = Path(tempfile.mkdtemp(prefix="be-"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def make_cluster(short_temp_dir):
    def _make(**kwargs):
        cluster = RayCluster(
            mode="single-machine",
            head_node_address="127.0.0.1",
            ray_temp_dir=short_temp_dir,
            head_num_cpus=1,
            force_clean_up=False,
            **kwargs,
        )
        collector = _WarningCollector()
        cluster.logger.addHandler(collector)
        return cluster, collector

    return _make


def _patch_meminfo(monkeypatch, mem_total, mem_available):
    monkeypatch.setattr(
        "bioengine.cluster.ray_cluster.read_meminfo",
        lambda: {"MemTotal": mem_total, "MemAvailable": mem_available},
    )


def test_cluster_warns_on_an_oversubscribed_host(make_cluster, monkeypatch):
    cluster, collector = make_cluster(head_memory_in_gb=30)
    _patch_meminfo(monkeypatch, HOST_TOTAL, HOST_AVAILABLE_WITH_CO_TENANTS)

    cluster._check_head_memory_budget()

    assert len(collector.messages) == 1
    assert "exceeds this host's memory budget" in collector.messages[0]


def test_cluster_silent_when_within_budget(make_cluster, monkeypatch):
    cluster, collector = make_cluster(head_memory_in_gb=30)
    _patch_meminfo(monkeypatch, HOST_TOTAL, HOST_AVAILABLE_WHEN_IDLE)

    cluster._check_head_memory_budget()

    assert collector.messages == []


def test_cluster_honours_a_configured_fraction(make_cluster, monkeypatch):
    cluster, collector = make_cluster(
        head_memory_in_gb=30, head_memory_budget_fraction=0
    )
    _patch_meminfo(monkeypatch, HOST_TOTAL, HOST_AVAILABLE_WITH_CO_TENANTS)

    cluster._check_head_memory_budget()

    assert collector.messages == []


def test_cluster_skips_the_check_without_a_reservation(make_cluster, monkeypatch):
    cluster, collector = make_cluster()

    def _fail():
        raise AssertionError("meminfo must not be read without a reservation")

    monkeypatch.setattr("bioengine.cluster.ray_cluster.read_meminfo", _fail)

    cluster._check_head_memory_budget()

    assert collector.messages == []


def test_cluster_tolerates_an_unreadable_meminfo(make_cluster, monkeypatch):
    cluster, collector = make_cluster(head_memory_in_gb=30)

    def _raise():
        raise FileNotFoundError("meminfo")

    monkeypatch.setattr("bioengine.cluster.ray_cluster.read_meminfo", _raise)

    cluster._check_head_memory_budget()

    assert collector.messages == []


class _FakeProcess:
    returncode = 0

    async def communicate(self):
        return b"", b""


def _patch_cluster_startup(monkeypatch, cluster):
    """Stub out everything ``_start_cluster`` shells out to, recording the calls."""
    commands = []

    async def _fake_exec(program, *args, **kwargs):
        commands.append([str(program), *args])
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(
        cluster,
        "_set_cluster_ports",
        lambda: cluster.ray_cluster_config["ports"].update(
            {"min_worker": 10002, "max_worker": 10100}
        ),
    )
    monkeypatch.setattr(cluster, "_update_symlink", lambda ray_temp_dir: None)
    return commands


@pytest.mark.parametrize(
    "mem_available",
    [HOST_AVAILABLE_WITH_CO_TENANTS, HOST_AVAILABLE_WHEN_IDLE],
    ids=["oversubscribed", "within_budget"],
)
def test_startup_proceeds_whether_or_not_the_budget_is_exceeded(
    make_cluster, monkeypatch, mem_available
):
    cluster, _ = make_cluster(head_memory_in_gb=30)
    _patch_meminfo(monkeypatch, HOST_TOTAL, mem_available)
    commands = _patch_cluster_startup(monkeypatch, cluster)

    asyncio.run(cluster._start_cluster())

    assert any("--head" in command for command in commands)
    assert any(f"--memory={30 * GIB}" in command for command in commands)
