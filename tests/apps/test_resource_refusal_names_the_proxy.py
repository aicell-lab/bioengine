"""A resource refusal must name the proxy reservation and what was free.

Two users deployed a CPU-only app, asked for the memory their node advertised, and were refused
with a number they had never typed:

    ValueError: Insufficient resources for application 'mito-em-quant'. Requested:
    {'num_cpus': 1, 'num_gpus': 0.0, 'memory': 4831838208, 'VRAM_MB': 0}

4831838208 B is 4608 MiB against a `memory_mb=4096` request. Both reporters independently read
the difference as a ~12.5% padding. It is not a percentage: `_sum_resources` adds the app's own
ProxyDeployment reservation, `int(proxy_memory_in_gb * 1024**3)`, which defaults to 0.5 GiB. It
is 12.5% of 4096 only because 512 is one eighth of 4096 — at 2048 the same constant is 25%, and
at 8192 it is 6.25%. A fix that "removed the 12.5%" would under-reserve for the proxy at every
other size, which is why these arms pin the ARITHMETIC rather than a ratio.

The refusal also never said what was free, though `free_per_node` is computed a few lines above
it and then discarded. That is the single fact that tells a caller whether to shrink the request
or add a node, and both reporters asked for it by name.
"""

import pytest

from bioengine.apps.manager import AppsManager

MIB = 1024**2
GIB = 1024**3

# The exact numbers from the report, so a regression reproduces the user's experience.
REPORTED_TOTAL = 4831838208  # what the refusal printed
REQUESTED_MB = 4096  # what the user wrote
PROXY_GB = 0.5  # the default proxy reservation

REQUIRED = {"num_cpus": 1, "num_gpus": 0.0, "memory": REPORTED_TOTAL, "VRAM_MB": 0}
FREE = {
    "node-a": {"num_cpus": 3.0, "num_gpus": 0.0, "memory": 2 * GIB},
    "node-b": {"num_cpus": 1.0, "num_gpus": 0.0, "memory": 512 * MIB},
}


def _format(**kw):
    return AppsManager._format_resource_refusal(
        kw.pop("application_id", "mito-em-quant"),
        kw.pop("required_resources", REQUIRED),
        kw.pop("free_per_node", FREE),
        kw.pop("proxy_memory_in_gb", PROXY_GB),
    )


def test_the_reported_arithmetic_closes_exactly() -> None:
    """The premise check. If this fails the rest of the file is testing a fiction."""
    assert REPORTED_TOTAL - REQUESTED_MB * MIB == int(PROXY_GB * GIB) == 512 * MIB


def test_the_refusal_names_what_was_asked_for_and_what_the_proxy_added() -> None:
    out = _format()
    assert "4096 MiB" in out, "the number the user actually wrote is not in the message"
    assert "512 MiB" in out, "the proxy's contribution is not named"
    assert "4608 MiB" in out, "the total is not given in the same unit as the request"
    assert (
        str(REPORTED_TOTAL) in out
    ), "the raw byte figure is gone, so logs cannot be matched"


def test_the_refusal_says_the_proxy_amount_is_FIXED_not_a_percentage() -> None:
    """The wording is load-bearing, because the wrong inference is the documented one.

    Two reporters derived a percentage from a single data point. A message that merely prints
    both numbers lets the next reader do the same, so it states which kind of quantity it is.
    """
    out = _format()
    assert (
        "fixed" in out.lower()
    ), "nothing stops the next reader inferring a percentage again"
    assert "proxy_memory_in_gb" in out, "the knob to change it is not named"


@pytest.mark.parametrize(
    "requested_mb,proxy_gb",
    [(4096, 0.5), (2048, 0.5), (8192, 0.5), (1024, 1.0), (16384, 0.25)],
)
def test_the_breakdown_is_arithmetic_not_a_ratio(requested_mb, proxy_gb) -> None:
    """The same message must be right at sizes where the ratio is nothing like 12.5%.

    At 2048 MiB with a 0.5 GiB proxy the overhead is 25%; at 16384 with 0.25 GiB it is 1.5%. An
    implementation that hardcoded a multiplier passes the 4096 case and fails here, which is the
    whole reason this is parametrized.
    """
    total = requested_mb * MIB + int(proxy_gb * GIB)
    out = _format(
        required_resources={**REQUIRED, "memory": total}, proxy_memory_in_gb=proxy_gb
    )
    assert f"{requested_mb} MiB" in out
    assert f"{int(proxy_gb * GIB) // MIB} MiB" in out
    assert f"{total // MIB} MiB" in out


def test_the_refusal_names_what_was_FREE_per_node() -> None:
    out = _format()
    for node in FREE:
        assert (
            node in out
        ), f"{node} is missing, so the caller cannot see which node was short"
    assert "2048 MiB" in out, "node-a's free memory is not reported"
    assert "512 MiB" in out, "node-b's free memory is not reported"


def test_no_nodes_is_DISTINGUISHED_from_nodes_with_nothing_free() -> None:
    """An empty section would read as "every node is full", which is a different problem.

    A cluster that reported no nodes at all and a cluster whose nodes are all full both leave
    `free_per_node` falsy at the call site if the section is simply omitted. They need different
    responses — wait for a node, versus shrink the request — so they must not render the same.
    """
    out = _format(free_per_node={})
    assert "NO NODES REPORTED" in out
    assert "node-a" not in out


def test_an_unknown_proxy_reservation_omits_the_breakdown_rather_than_guessing() -> (
    None
):
    """`proxy_memory_in_gb=None` means the caller could not tell us, not that it is zero.

    Printing a breakdown computed from an assumed default would be a confident wrong number in
    exactly the place this change exists to fix. The free figures are still reported, because
    those do not depend on it.
    """
    out = _format(proxy_memory_in_gb=None)
    assert "ProxyDeployment reserves" not in out
    assert (
        "node-a" in out
    ), "the free figures should survive an unknown proxy reservation"
    assert str(REPORTED_TOTAL) in out
