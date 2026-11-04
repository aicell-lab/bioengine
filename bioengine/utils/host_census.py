"""Census of the BioEngine workers sharing one host.

A worker cannot read a co-tenant's configured resources from inside its own
container: sibling memory cgroups are not visible, sibling process argv is not
visible, and a single-machine worker's ``ray.nodes()`` sees only its own GCS. So
the reservations only become readable if the workers advertise them, which is
what this module is.

Each worker writes one file into a directory every worker on the host mounts,
refreshes it on the monitoring loop's cadence, and removes it on clean shutdown.
The freshness deadline travels *inside* each file, as it does for the liveness
heartbeat, so a reader needs no knowledge of any other worker's interval — a
worker that died without removing its file ages out on its own terms.

The census is advisory throughout. No absent, stale, unreadable or malformed
advertisement is an error: an un-mounted census directory simply means no
co-tenants are known, which is also the correct answer on a host that has none.
"""

import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_GIB = 1024**3

# Set by the CLI launcher on the worker it starts, pointing at a host directory
# bind-mounted into every worker. Unset means the census is inert.
CENSUS_DIR_ENV = "BIOENGINE_HOST_CENSUS_DIR"
WORKER_ID_ENV = "BIOENGINE_WORKER_ID"

# The container runtime sets this from --gpus/--device and it is the
# authoritative statement of which devices this container was handed.
# CUDA_VISIBLE_DEVICES is the fallback for a native (uncontainerised) worker.
_GPU_ENV_VARS = ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")

# The runtime's two non-enumerating values. A worker holding every device
# collides with any worker holding any device, so ALL_GPUS is kept distinct
# from an empty list rather than expanded.
ALL_GPUS = "all"
_NO_GPUS = ("", "none", "void")

# Deadline carried by the advertisement written before the monitoring loop
# exists to refresh it. Starting a Ray cluster and deploying startup
# applications has no useful upper bound, so this only ages out a worker that
# never got anywhere at all — same reasoning as the liveness heartbeat's.
STARTUP_STALE_AFTER_SECONDS = 1800.0


def census_dir() -> Optional[Path]:
    """The shared advertisement directory, or None when the census is not wired up."""
    value = os.environ.get(CENSUS_DIR_ENV)
    return Path(value) if value else None


def worker_id() -> str:
    """Identity this worker advertises under.

    The CLI launcher sets this to the container name, which is what an operator
    recognises in a warning. The hostname fallback is a container id or a pod
    name — unique either way, just less legible.
    """
    return os.environ.get(WORKER_ID_ENV) or socket.gethostname()


def shm_size_bytes(path: Union[str, Path] = "/dev/shm") -> Optional[int]:
    """Size of the shared-memory mount, which is what bounds Ray's object store.

    Counted as part of a worker's memory claim because ``--object-store-memory``
    is never passed, so Ray auto-sizes the object store and it sits outside the
    ``--memory`` reservation entirely.
    """
    try:
        stats = os.statvfs(str(path))
    except OSError:
        return None
    return stats.f_blocks * stats.f_frsize


def visible_gpu_device_ids() -> List[str]:
    """Device ids this worker was given, as the runtime names them.

    The values may be indices or UUIDs, and a worker given index ``1`` and one
    given that card's UUID name the same device without matching as strings. So
    this detects collisions between workers specified the same way — which is
    the realistic case, since a site picks one convention — and can miss a
    collision expressed in two conventions at once. Indices additionally
    reorder across driver reloads, which is why the collision check compares
    them for equality only and never interprets them.
    """
    for name in _GPU_ENV_VARS:
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = raw.strip()
        if value.lower() in _NO_GPUS:
            return []
        if value.lower() == ALL_GPUS:
            return [ALL_GPUS]
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _advertisement_path(directory: Union[str, Path], worker_id: str) -> Path:
    # One file per worker id, so a restart overwrites its own entry instead of
    # accumulating a second claim against the same resources.
    safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in worker_id)
    return Path(directory) / f"{safe_id}.json"


def write_advertisement(
    directory: Union[str, Path],
    worker_id: str,
    stale_after_seconds: float,
    resources: Dict[str, Any],
) -> None:
    """Publish or refresh this worker's resource claim."""
    path = _advertisement_path(directory, worker_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "worker_id": worker_id,
            "timestamp": time.time(),
            "stale_after_seconds": stale_after_seconds,
            **resources,
        }
    )
    # Replace atomically so a co-tenant can never read a half-written file.
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(payload)
    os.replace(tmp_path, path)


def remove_advertisement(directory: Union[str, Path], worker_id: str) -> None:
    """Withdraw this worker's claim on clean shutdown."""
    _advertisement_path(directory, worker_id).unlink(missing_ok=True)


def read_co_tenants(
    directory: Union[str, Path],
    own_worker_id: str,
    now: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Every other worker's live advertisement, newest first.

    Stale entries are dropped against the deadline each file declares for
    itself, so a worker killed without a chance to withdraw stops counting
    without anyone having to clean up after it.
    """
    now = time.time() if now is None else now
    co_tenants: List[Dict[str, Any]] = []
    try:
        entries = sorted(Path(directory).glob("*.json"))
    except OSError:
        return []

    for entry in entries:
        try:
            payload = json.loads(entry.read_text())
            worker_id = str(payload["worker_id"])
            age = now - float(payload["timestamp"])
            stale_after = float(payload["stale_after_seconds"])
        except Exception:
            continue
        if worker_id == own_worker_id or age > stale_after:
            continue
        payload["age_seconds"] = age
        co_tenants.append(payload)

    co_tenants.sort(key=lambda entry: entry["age_seconds"])
    return co_tenants


def _claimed_gib(advertisement: Dict[str, Any]) -> float:
    """Host RAM one worker can claim: its head reservation plus its shared memory.

    ``--object-store-memory`` is never passed, so Ray auto-sizes the object
    store and it sits *outside* ``--memory``. What bounds it is the container's
    shared-memory size, which therefore has to be counted as part of the claim
    rather than alongside it.
    """
    reserved = advertisement.get("head_memory_in_gb") or 0
    shm_bytes = advertisement.get("shm_size_bytes") or 0
    return float(reserved) + float(shm_bytes) / _GIB


def _describe(advertisement: Dict[str, Any]) -> str:
    worker_id = str(advertisement.get("worker_id", "unknown"))
    details = []
    reserved = advertisement.get("head_memory_in_gb")
    if reserved:
        details.append(f"{float(reserved):.0f} GiB reserved")
    shm_bytes = advertisement.get("shm_size_bytes")
    if shm_bytes:
        details.append(f"{float(shm_bytes) / _GIB:.0f} GiB shm")
    gpu_ids = advertisement.get("gpu_device_ids")
    if gpu_ids:
        details.append(f"GPU {','.join(str(i) for i in gpu_ids)}")
    return f"{worker_id} ({', '.join(details)})" if details else worker_id


def memory_co_tenancy_warning(
    own: Dict[str, Any],
    co_tenants: List[Dict[str, Any]],
    mem_total_bytes: int,
    budget_fraction: float,
) -> Optional[str]:
    """Warn when the workers' *configured* claims cannot all be honoured.

    This is the complement of the host-occupancy check, not a replacement for
    it: occupancy measures what co-tenants currently *hold*, so an idle sibling
    configured at 40 GiB contributes almost nothing there. Here an idle sibling
    counts for everything it was promised, which is the whole point — Ray hands
    out the reservation as a guarantee whether or not it is being used yet.
    """
    if budget_fraction <= 0 or mem_total_bytes <= 0 or not co_tenants:
        return None

    own_gib = _claimed_gib(own)
    others_gib = sum(_claimed_gib(entry) for entry in co_tenants)
    total_gib = own_gib + others_gib
    budget_gib = mem_total_bytes * budget_fraction / _GIB
    if total_gib <= budget_gib:
        return None

    roster = "; ".join(_describe(entry) for entry in co_tenants)
    return (
        f"BioEngine workers on this host have together been promised more memory than it "
        f"has: this worker claims {own_gib:.1f} GiB and {len(co_tenants)} co-tenant "
        f"worker(s) claim {others_gib:.1f} GiB, a total of {total_gib:.1f} GiB against a "
        f"budget of {budget_gib:.1f} GiB ({budget_fraction:.0%} of this host's "
        f"{mem_total_bytes / _GIB:.1f} GiB). Each claim counts the worker's "
        f"--head-memory-in-gb plus its shared-memory size, because Ray sizes the object "
        f"store outside the memory reservation. Co-tenants: {roster}. Every claim is a "
        f"guarantee Ray schedules against, and the host cannot keep them all at once; if "
        f"the peaks coincide, the kernel or Ray's memory monitor kills a process that need "
        f"not belong to the worker that over-reserved."
    )


def gpu_co_tenancy_warning(
    own: Dict[str, Any],
    co_tenants: List[Dict[str, Any]],
) -> Optional[str]:
    """Warn when another worker on this host was handed the same GPU.

    Needs no summing and no budget: it is an exact match on device identity, so
    it has no false-positive mode beyond a device genuinely meant to be shared.
    It also matters more than the memory case — two single-machine workers are
    two separate Ray clusters, so neither scheduler can see the other's claim,
    and VRAM is not time-shared, so both filling the card is a CUDA allocation
    failure inside an app rather than anything a resource monitor arbitrates.
    """
    own_ids = {str(i) for i in (own.get("gpu_device_ids") or [])}
    if not own_ids:
        return None

    collisions = {}
    for entry in co_tenants:
        other_ids = {str(i) for i in (entry.get("gpu_device_ids") or [])}
        if not other_ids:
            continue
        # "all" names every device, so it collides with any non-empty claim
        # including another "all".
        if ALL_GPUS in own_ids or ALL_GPUS in other_ids:
            shared = sorted(own_ids | other_ids)
        else:
            shared = sorted(own_ids & other_ids)
        if shared:
            collisions[str(entry.get("worker_id", "unknown"))] = shared

    if not collisions:
        return None

    roster = "; ".join(
        f"{worker_id} (device {', '.join(ids)})" for worker_id, ids in collisions.items()
    )
    return (
        f"GPU device(s) handed to this worker are also held by another BioEngine worker on "
        f"this host: {roster}. Each single-machine worker is its own Ray cluster, so neither "
        f"scheduler can see the other's claim on the device, and the node-level VRAM a worker "
        f"advertises is read off the physical card — so N workers sharing one card advertise N "
        f"times its memory and apps pack against a figure the hardware cannot back. Unlike "
        f"CPU, VRAM is not time-shared: the second allocation fails inside the app. Give each "
        f"worker a distinct device, or make the sharing deliberate (MIG or time-slicing)."
    )


def co_tenancy_warnings(
    own: Dict[str, Any],
    co_tenants: List[Dict[str, Any]],
    mem_total_bytes: int,
    budget_fraction: float,
) -> List[str]:
    """Every co-tenancy warning that applies, in descending order of severity.

    CPU is deliberately absent. Oversubscribing it costs throughput and kills
    nothing, so a warning would fire on correct configurations — and a warning
    that fires on a correct configuration is what teaches people to ignore the
    two above, which do not.
    """
    warnings = [
        gpu_co_tenancy_warning(own, co_tenants),
        memory_co_tenancy_warning(own, co_tenants, mem_total_bytes, budget_fraction),
    ]
    return [warning for warning in warnings if warning]
