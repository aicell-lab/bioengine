"""Startup check for a head memory reservation that the host cannot back."""

from pathlib import Path
from typing import Dict, Optional, Union

_GIB = 1024**3

MEMINFO_PATH = "/proc/meminfo"


def read_meminfo(path: Union[str, Path] = MEMINFO_PATH) -> Dict[str, int]:
    """Parse ``/proc/meminfo`` into a field name -> bytes mapping.

    Container runtimes do not namespace ``/proc/meminfo``, so a worker inside a
    container reads the figures of the host it shares with its co-tenants.
    """
    fields: Dict[str, int] = {}
    for line in Path(path).read_text().splitlines():
        name, separator, value = line.partition(":")
        if not separator:
            continue
        parts = value.split()
        if not parts:
            continue
        try:
            amount = int(parts[0])
        except ValueError:
            continue
        if len(parts) > 1 and parts[1].lower() == "kb":
            amount *= 1024
        fields[name.strip()] = amount
    return fields


def head_memory_budget_warning(
    reserved_gb: float,
    mem_total_bytes: int,
    mem_available_bytes: int,
    budget_fraction: float,
) -> Optional[str]:
    """Return a warning when the head reservation does not fit the host, else None.

    ``mem_total_bytes - mem_available_bytes`` is what every other tenant of the
    host already holds, whether or not it is a container and whether or not it
    belongs to BioEngine. A reservation stacked on top of that is a guarantee
    the host has no way to keep.
    """
    if budget_fraction <= 0 or mem_total_bytes <= 0:
        return None

    reserved = reserved_gb * _GIB
    held_by_others = max(mem_total_bytes - mem_available_bytes, 0)
    total = reserved + held_by_others
    budget = mem_total_bytes * budget_fraction
    if total <= budget:
        return None

    headroom = budget - held_by_others
    if reserved > budget:
        cause = (
            "The reservation alone is over the budget, so this worker is oversized for the "
            "host whatever its co-tenants do. Lower --head-memory-in-gb, or raise "
            "--head-memory-budget-fraction if this host is deliberately oversubscribed."
        )
    elif headroom <= 0:
        cause = (
            "The reservation would fit an idle host; the memory already held by other tenants "
            "is over the budget on its own, so lowering --head-memory-in-gb cannot bring this "
            "back inside it. Free memory on the host, or raise "
            "--head-memory-budget-fraction if this host is deliberately oversubscribed."
        )
    else:
        cause = (
            f"The reservation would fit an idle host; it is the memory already held by other "
            f"tenants that leaves only {headroom / _GIB:.1f} GiB inside the budget. Free memory "
            f"on the host, reserve no more than {headroom / _GIB:.1f} GiB, or raise "
            f"--head-memory-budget-fraction if this host is deliberately oversubscribed."
        )

    return (
        f"Head memory reservation exceeds this host's memory budget: this worker reserves "
        f"{reserved / _GIB:.1f} GiB while {held_by_others / _GIB:.1f} GiB of the host's "
        f"{mem_total_bytes / _GIB:.1f} GiB is already held by other tenants, a total of "
        f"{total / _GIB:.1f} GiB against a budget of {budget / _GIB:.1f} GiB "
        f"({budget_fraction:.0%} of MemTotal). Ray schedules against the reservation as a "
        f"guarantee, and the host cannot keep every tenant's guarantee at once. This is not a "
        f"prediction that the host will run out of memory - the peaks may never coincide - but "
        f"if they do, the kernel or Ray's memory monitor kills a process that need not be one "
        f"of this worker's. {cause}"
    )
