"""A refusal must say how the ask was composed and what was free.

WHAT THIS PINS. The memory asked of Ray is the app's deployments plus the
ProxyDeployment's own reservation, so it exceeds the number the caller wrote. A
refusal quoting only the total prints a figure the caller never typed. The excess
is a flat ``proxy_memory_in_gb`` (default 0.5 GiB), not a percentage: at 4096 MiB
it is 12% and at 512 MiB it is 100%, so wording it as a percentage is wrong at
every size but one.
"""

from bioengine.apps.manager import AppsManager

MIB = 1024 * 1024
GIB = 1024**3


def _free(mem_mib, cpus=3, gpus=0):
    return {"node-a": {"memory": mem_mib * MIB, "num_cpus": cpus, "num_gpus": gpus}}


def _required(total_mib, cpus=1, gpus=0):
    return {"num_cpus": cpus, "num_gpus": gpus, "memory": total_mib * MIB, "VRAM_MB": 0}


def test_names_the_proxy_reservation_additively():
    # the reporter's case: asked 4096, Ray saw 4608
    msg = AppsManager._describe_resource_shortfall(
        _required(4608), _free(3891), proxy_memory_in_gb=0.5
    )
    assert "4096 MiB for the app's deployments" in msg
    assert "512 MiB for the BioEngine proxy" in msg
    assert "= 4608 MiB" in msg
    # the split must be ADDITIVE, never expressed as a percentage
    assert "%" not in msg
    assert "proxy_memory_in_gb" in msg


def test_names_what_was_free():
    msg = AppsManager._describe_resource_shortfall(
        _required(4608), _free(3891), proxy_memory_in_gb=0.5
    )
    assert "Free now" in msg
    assert "3891 MiB" in msg
    assert "node-a" in msg


def test_small_request_is_padded_proportionally_more():
    # 512 MiB asked -> 1024 MiB seen: the flat reservation DOUBLES a small ask
    msg = AppsManager._describe_resource_shortfall(
        _required(1024), _free(900), proxy_memory_in_gb=0.5
    )
    assert "512 MiB for the app's deployments" in msg
    assert "512 MiB for the BioEngine proxy" in msg
    assert "= 1024 MiB" in msg


def test_unknown_reservation_is_not_guessed():
    msg = AppsManager._describe_resource_shortfall(
        _required(4608), _free(3891), proxy_memory_in_gb=None
    )
    assert "4608 MiB total" in msg
    assert "for the BioEngine proxy" not in msg


def test_no_capacity_is_stated_not_omitted():
    msg = AppsManager._describe_resource_shortfall(
        _required(4608), {}, proxy_memory_in_gb=0.5
    )
    assert "no node reported capacity" in msg
