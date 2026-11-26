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

import ast
import inspect
import logging
from pathlib import Path
import pytest

from bioengine.apps.manager import AppsManager

MIB = 1024**2
GIB = 1024**3

# The exact numbers from the report, so a regression reproduces the user's experience.
REPORTED_TOTAL = 4831838208  # what the refusal printed
REQUESTED_MB = 4096  # what the user wrote
PROXY_GB = 0.5  # the default proxy reservation

REQUIRED = {"num_cpus": 1, "num_gpus": 0.0, "memory": REPORTED_TOTAL, "VRAM_MB": 0}
# node-b's free figure is deliberately a number that appears NOWHERE else in the message.
# It was 512 MiB, which is also the proxy reservation -- so the assertion "512 MiB is in the
# output" passed on the proxy line alone, and a formatter that dropped node-b entirely still
# satisfied it. A reviewer caught that; 333 MiB cannot be produced by any other part.
FREE = {
    "node-a": {"num_cpus": 3.0, "num_gpus": 0.0, "memory": 2 * GIB},
    "node-b": {"num_cpus": 1.0, "num_gpus": 0.0, "memory": 333 * MIB},
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
    assert "333 MiB" in out, "node-b's free memory is not reported"


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


# ───────────────────── the refusal a USER sees, not just the formatter ─────────────────────
#
# Everything above calls `_format_resource_refusal` directly. A reviewer showed that leaves the
# WIRING unpinned: deleting `proxy_memory_in_gb=proxy_memory_in_gb` from the `_check_resources`
# call in `deploy_app`, or restoring the old `raise ValueError` text, changed no test in this
# file. Both mutants reinstate the defect this row exists to fix, so the arms below drive the
# real `_check_resources` and read the exception a caller would actually get.


class _Ready:
    """`_check_resources` awaits `ray_cluster.is_ready.wait()` before reading the nodes."""

    async def wait(self):
        return True


class _Cluster:
    def __init__(self, nodes, mode):
        self.mode = mode
        self.status = {"nodes": nodes}
        self.is_ready = _Ready()


class _Mgr:
    """The smallest object `_check_resources` will run against.

    Deliberately NOT a real AppsManager: these arms must fail if the code starts depending on
    instance state that a caller cannot supply, and they must not drag a Ray cluster in.
    """

    def __init__(self, nodes, mode="single-machine"):
        self.logger = logging.getLogger("test-refusal")
        self._deployed_applications = {}
        self.ray_cluster = _Cluster(nodes, mode)


def _tiny_node():
    # 1 GiB free: smaller than the 4608 MiB the request totals, so the refusal must fire.
    return {
        "total_cpu": 8,
        "used_cpu": 0,
        "total_gpu": 0,
        "used_gpu": 0,
        "total_memory": 1 * GIB,
        "used_memory": 0,
    }


@pytest.mark.asyncio
async def test_the_RAISED_refusal_carries_the_split_and_the_free_figures() -> None:
    """The real path, end to end: `_check_resources` -> ValueError -> what the user reads."""
    mgr = _Mgr(nodes={"n1": _tiny_node()})
    with pytest.raises(ValueError) as caught:
        await AppsManager._check_resources(
            mgr, "mito-em-quant", dict(REQUIRED), proxy_memory_in_gb=PROXY_GB
        )
    msg = str(caught.value)
    assert "4096 MiB" in msg, "the raised error does not say what the user asked for"
    assert "512 MiB" in msg, "the raised error does not name the proxy's contribution"
    assert (
        "fixed" in msg.lower()
    ), "the raised error still invites the percentage misreading"
    assert "n1" in msg, "the raised error does not name the node that was short"
    assert "1024 MiB" in msg, "the raised error does not say how much was actually free"


@pytest.mark.asyncio
async def test_without_the_proxy_value_the_RAISED_refusal_omits_the_split() -> None:
    """The other half: the breakdown is reported only when it can be computed.

    This is what makes the arm above load-bearing rather than decorative -- it shows the split
    comes from the forwarded argument and not from something the formatter invents.
    """
    mgr = _Mgr(nodes={"n1": _tiny_node()})
    with pytest.raises(ValueError) as caught:
        await AppsManager._check_resources(mgr, "mito-em-quant", dict(REQUIRED))
    msg = str(caught.value)
    assert "ProxyDeployment reserves" not in msg
    assert "n1" in msg, "the free figures should survive an unknown proxy reservation"


def test_deploy_app_FORWARDS_the_resolved_proxy_reservation() -> None:
    """Structural, because the mutant it guards is a deletion.

    Dropping `proxy_memory_in_gb=` from the `_check_resources` call in `deploy_app` sends every
    production refusal down the None arm, silently losing the breakdown -- which is clause 1's
    original defect, reinstated. No behavioural arm short of a full deploy sees it, so this reads
    the call site out of the AST: find the `_check_resources` call and require that keyword.
    """
    src = Path(inspect.getsourcefile(AppsManager)).read_text()
    tree = ast.parse(src)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_check_resources"
    ]
    assert calls, "no _check_resources call site found; this guard is testing nothing"
    forwarding = [
        c for c in calls if any(k.arg == "proxy_memory_in_gb" for k in c.keywords)
    ]
    assert forwarding, (
        "no _check_resources call forwards proxy_memory_in_gb, so every production refusal "
        "takes the unknown-reservation path and loses the breakdown"
    )
