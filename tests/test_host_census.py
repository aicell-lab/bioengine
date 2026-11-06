"""Pin the host co-tenancy census: what it warns about, and what it refuses to.

The census exists because a worker cannot read a co-tenant's configured
resources from inside its own container — sibling cgroups and sibling argv are
both invisible, and a single-machine worker's ``ray.nodes()`` sees only its own
GCS. So the workers advertise their claims to each other, and these tests pin
the three decisions that make the result trustworthy:

* a claim counts what the worker was **promised**, not what it currently holds,
  which is the whole difference from the host-occupancy check that shipped
  first and stays silent on an idle-but-oversized sibling;
* shared memory counts as part of the memory claim, because Ray sizes the
  object store outside ``--memory``;
* CPU is deliberately never warned about.
"""

from __future__ import annotations

import json
import time

from bioengine.utils import host_census

_GIB = 1024**3


def _advert(worker_id: str, **resources) -> dict:
    return {"worker_id": worker_id, **resources}


def _published(
    directory, worker_id: str, stale_after_seconds: float = 600.0, **resources
) -> None:
    host_census.write_advertisement(
        directory=directory,
        worker_id=worker_id,
        stale_after_seconds=stale_after_seconds,
        resources=resources,
    )


# ===== the census itself =====


def test_roundtrip_excludes_self(tmp_path) -> None:
    _published(tmp_path, "worker-a", head_memory_in_gb=30)
    _published(tmp_path, "worker-b", head_memory_in_gb=30)

    assert [e["worker_id"] for e in host_census.read_co_tenants(tmp_path, "worker-a")] == [
        "worker-b"
    ]
    assert [e["worker_id"] for e in host_census.read_co_tenants(tmp_path, "worker-b")] == [
        "worker-a"
    ]


def test_withdrawn_claim_stops_counting(tmp_path) -> None:
    _published(tmp_path, "worker-b", head_memory_in_gb=30)
    host_census.remove_advertisement(tmp_path, "worker-b")
    assert host_census.read_co_tenants(tmp_path, "worker-a") == []
    # Withdrawing twice is how a crash-then-restart sequence can arrive.
    host_census.remove_advertisement(tmp_path, "worker-b")


def test_stale_claim_ages_out_on_its_own_deadline(tmp_path) -> None:
    """A worker killed without withdrawing must stop counting by itself.

    The deadline travels inside each file, so one worker's slow interval cannot
    make another's entry look alive — pinned here by giving the two different
    deadlines and expiring only one.
    """
    _published(tmp_path, "brief", stale_after_seconds=10.0, head_memory_in_gb=30)
    _published(tmp_path, "patient", stale_after_seconds=10_000.0, head_memory_in_gb=30)

    live = host_census.read_co_tenants(tmp_path, "self", now=time.time() + 100)
    assert [e["worker_id"] for e in live] == ["patient"]


def test_malformed_and_foreign_files_are_skipped(tmp_path) -> None:
    """Advisory throughout: nothing in the directory can break a reader."""
    _published(tmp_path, "good", head_memory_in_gb=30)
    (tmp_path / "truncated.json").write_text('{"worker_id": "x", "timesta')
    (tmp_path / "no-timestamp.json").write_text(json.dumps({"worker_id": "y"}))
    (tmp_path / "not-json.txt").write_text("not json at all")

    assert [e["worker_id"] for e in host_census.read_co_tenants(tmp_path, "self")] == [
        "good"
    ]


def test_missing_directory_reads_empty(tmp_path) -> None:
    """An un-mounted census is "no co-tenants known", not an error.

    This is the same answer as a host that genuinely has none, which is why the
    feature can default on without changing behaviour anywhere.
    """
    assert host_census.read_co_tenants(tmp_path / "never-created", "self") == []


def test_restart_overwrites_its_own_entry(tmp_path) -> None:
    """A restarted worker must not leave a second claim against itself."""
    _published(tmp_path, "worker-b", head_memory_in_gb=30)
    _published(tmp_path, "worker-b", head_memory_in_gb=10)

    co_tenants = host_census.read_co_tenants(tmp_path, "self")
    assert len(co_tenants) == 1
    assert co_tenants[0]["head_memory_in_gb"] == 10


# ===== memory: the configured claim, which is the point of the whole issue =====


def test_idle_oversized_sibling_still_warns(tmp_path) -> None:
    """The case the shipped occupancy check stays silent on.

    Three workers at 30 GiB on a 62.49 GiB host is the configuration that
    motivated this. Occupancy measures what a sibling *holds*, so an idle one
    contributes almost nothing; here it contributes everything it was promised.
    """
    own = _advert("worker-a", head_memory_in_gb=30)
    others = [
        _advert("worker-b", head_memory_in_gb=30),
        _advert("worker-c", head_memory_in_gb=30),
    ]
    warning = host_census.memory_co_tenancy_warning(
        own, others, mem_total_bytes=int(62.49 * _GIB), budget_fraction=0.9
    )
    assert warning is not None
    assert "90.0 GiB" in warning
    assert "worker-b" in warning and "worker-c" in warning


def test_shared_memory_counts_toward_the_claim() -> None:
    """``--object-store-memory`` is never passed, so shm is part of the claim.

    Two workers at 25 GiB fit inside 0.9 x 62.49 = 56.2 GiB on reservations
    alone; with 8 GiB of shm each they do not. Without counting shm this case
    reads as safe.
    """
    budget_args = {"mem_total_bytes": int(62.49 * _GIB), "budget_fraction": 0.9}

    reservations_only = host_census.memory_co_tenancy_warning(
        _advert("a", head_memory_in_gb=25),
        [_advert("b", head_memory_in_gb=25)],
        **budget_args,
    )
    assert reservations_only is None

    with_shm = host_census.memory_co_tenancy_warning(
        _advert("a", head_memory_in_gb=25, shm_size_bytes=8 * _GIB),
        [_advert("b", head_memory_in_gb=25, shm_size_bytes=8 * _GIB)],
        **budget_args,
    )
    assert with_shm is not None
    assert "66.0 GiB" in with_shm


def test_fitting_claims_do_not_warn() -> None:
    assert (
        host_census.memory_co_tenancy_warning(
            _advert("a", head_memory_in_gb=20),
            [_advert("b", head_memory_in_gb=20)],
            mem_total_bytes=int(62.49 * _GIB),
            budget_fraction=0.9,
        )
        is None
    )


def test_no_co_tenants_never_warns() -> None:
    """A single worker sized near MemTotal is the legitimate case, not a fault."""
    assert (
        host_census.memory_co_tenancy_warning(
            _advert("a", head_memory_in_gb=60),
            [],
            mem_total_bytes=int(62.49 * _GIB),
            budget_fraction=0.9,
        )
        is None
    )


# ===== GPU: exact device identity, no budget, no summing =====


def test_same_device_warns() -> None:
    """Measured live on europa: two workers both handed DeviceIDs ["1"]."""
    warning = host_census.gpu_co_tenancy_warning(
        _advert("worker-europa", gpu_device_ids=["1"]),
        [_advert("worker-tabula", gpu_device_ids=["1"])],
    )
    assert warning is not None
    assert "worker-tabula" in warning and "device 1" in warning


def test_distinct_devices_do_not_warn() -> None:
    assert (
        host_census.gpu_co_tenancy_warning(
            _advert("a", gpu_device_ids=["1"]),
            [_advert("b", gpu_device_ids=["2"])],
        )
        is None
    )


def test_all_collides_with_any_specific_device() -> None:
    """``all`` names every device, so it cannot be compared by intersection."""
    assert (
        host_census.gpu_co_tenancy_warning(
            _advert("a", gpu_device_ids=[host_census.ALL_GPUS]),
            [_advert("b", gpu_device_ids=["2"])],
        )
        is not None
    )
    assert (
        host_census.gpu_co_tenancy_warning(
            _advert("a", gpu_device_ids=["2"]),
            [_advert("b", gpu_device_ids=[host_census.ALL_GPUS])],
        )
        is not None
    )


def test_cpu_only_workers_never_collide() -> None:
    """A worker with no GPU cannot collide, and must not warn against one that has."""
    assert (
        host_census.gpu_co_tenancy_warning(
            _advert("a", gpu_device_ids=[]),
            [_advert("b", gpu_device_ids=["1"])],
        )
        is None
    )
    assert (
        host_census.gpu_co_tenancy_warning(
            _advert("a", gpu_device_ids=["1"]),
            [_advert("b", gpu_device_ids=[])],
        )
        is None
    )


# ===== what is deliberately not warned about =====


def test_cpu_oversubscription_is_not_a_warning() -> None:
    """CPU is time-shared: oversubscribing it costs throughput and kills nothing.

    europa runs 16 CPUs with per-worker defaults of 8 and has never died of it,
    so a warning here would fire on a correct configuration — which is what
    teaches people to ignore the memory and GPU warnings that do matter.
    """
    warnings = host_census.co_tenancy_warnings(
        own=_advert("a", head_num_cpus=8, head_memory_in_gb=5),
        co_tenants=[
            _advert("b", head_num_cpus=8, head_memory_in_gb=5),
            _advert("c", head_num_cpus=8, head_memory_in_gb=5),
        ],
        mem_total_bytes=int(62.49 * _GIB),
        budget_fraction=0.9,
    )
    assert warnings == [], "24 CPUs promised on 16 is not a warning"


def test_gpu_warning_outranks_memory() -> None:
    """Both can fire at once; the GPU one is first because it has no false mode."""
    warnings = host_census.co_tenancy_warnings(
        own=_advert("a", head_memory_in_gb=40, gpu_device_ids=["1"]),
        co_tenants=[_advert("b", head_memory_in_gb=40, gpu_device_ids=["1"])],
        mem_total_bytes=int(62.49 * _GIB),
        budget_fraction=0.9,
    )
    assert len(warnings) == 2
    assert "GPU device(s)" in warnings[0]
    assert "more memory than it has" in warnings[1]


# ===== the environment the census reads itself from =====


def test_census_is_inert_without_the_env_var(monkeypatch) -> None:
    monkeypatch.delenv(host_census.CENSUS_DIR_ENV, raising=False)
    assert host_census.census_dir() is None


def test_worker_id_prefers_the_container_name(monkeypatch) -> None:
    monkeypatch.setenv(host_census.WORKER_ID_ENV, "bioengine-worker-europa")
    assert host_census.worker_id() == "bioengine-worker-europa"
    monkeypatch.delenv(host_census.WORKER_ID_ENV)
    assert host_census.worker_id()  # hostname fallback, whatever it is


def test_gpu_env_parsing(monkeypatch) -> None:
    for value, expected in [
        ("1", ["1"]),
        ("0,2", ["0", "2"]),
        (" 1 , 2 ", ["1", "2"]),
        ("all", [host_census.ALL_GPUS]),
        ("none", []),
        ("void", []),
        ("", []),
        ("GPU-f3f75397-e6e1-a0da-cea9-5e200a6bcb9a", ["GPU-f3f75397-e6e1-a0da-cea9-5e200a6bcb9a"]),
    ]:
        monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", value)
        assert host_census.visible_gpu_device_ids() == expected, value


def test_nvidia_env_wins_over_cuda(monkeypatch) -> None:
    """The runtime sets NVIDIA_VISIBLE_DEVICES from --gpus/--device, so it is
    the authoritative statement of what this container was handed; app code can
    narrow CUDA_VISIBLE_DEVICES afterwards without changing the allocation."""
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert host_census.visible_gpu_device_ids() == ["1"]
