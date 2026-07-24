"""Pin that ProxyDeployment's entry health check tolerates saturation.

The proxy probes the entry in-band via ``check_health.remote()``. That call
is routed through the entry's router and counts against its
``max_ongoing_requests`` (default 10), so a saturated-but-healthy entry
head-of-line-blocks the probe and the 3s ``wait_for`` fires. Treating that
timeout as a health failure spuriously deregisters the Hypha service while
the entry is merely busy.

The fix classifies the two outcomes distinctly: an ``asyncio.TimeoutError``
is tolerated (entry busy, service stays registered), while any other raised
exception — the entry crashed, or a ``depends_on`` dependency is down and
its ``health_check`` raised — deregisters. Reading the entry's RUNNING
count from ``serve.status()`` instead does NOT work here: Ray Serve keeps a
health-check-failing replica at ``RUNNING`` (it is not evicted on a raised
check_health within the health window), so an out-of-band count is blind to
a ``depends_on`` outage. These tests pin the shape (check_health is
Ray-Serve glue that is impractical to exercise standalone).
"""
from __future__ import annotations

import inspect

from bioengine.apps import proxy_deployment as pd_module


def _check_health_src() -> str:
    return inspect.getsource(pd_module.ProxyDeployment.func_or_class.check_health)


def test_timeout_is_tolerated_not_deregistered() -> None:
    """A saturation timeout must be caught separately and NOT deregister."""
    src = _check_health_src()
    assert "except asyncio.TimeoutError" in src, (
        "check_health must catch asyncio.TimeoutError separately so a "
        "saturated entry is not mistaken for an unhealthy one."
    )
    timeout_block = src.split("except asyncio.TimeoutError", 1)[1].split(
        "except Exception", 1
    )[0]
    assert "_deregister_services" not in timeout_block, (
        "a saturation timeout must leave the Hypha service registered."
    )


def test_raised_health_error_deregisters() -> None:
    """A genuine health failure (crash / depends_on down) deregisters."""
    src = _check_health_src()
    error_block = src.split("except Exception", 1)[1]
    assert "_deregister_services" in error_block
    assert "raise RuntimeError" in error_block


def test_probe_stays_in_band() -> None:
    """The in-band probe is what propagates the entry's raised health error
    (incl. the depends_on gate) — an out-of-band serve.status() RUNNING count
    is blind to it, so the probe must stay in-band."""
    src = _check_health_src()
    assert "entry_deployment_handle.check_health.remote" in src
