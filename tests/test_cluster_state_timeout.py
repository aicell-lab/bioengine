"""The ``get_cluster_state`` round trip is bounded, so a silent Ray head
cannot wedge the monitoring loop (#0137).

The monitoring steps are sequential awaits. An unbounded wait here stops the
whole pass, which is where application recovery and proxy-token renewal live,
so the failure mode is silent: the worker keeps advertising itself as healthy
while apps drift away and tokens expire.

Nothing here asserts anything about cluster capacity. ``total_cpu == 0`` and an
empty ``nodes`` map are the correct idle reading for a SLURM worker, and gating
on them was retracted on #0135.
"""

import ast
import asyncio
import logging
import time
from pathlib import Path

import pytest

from bioengine.cluster import ray_cluster as rc_mod
from bioengine.cluster.ray_cluster import RayCluster
from bioengine.heartbeat import MISSED_PASSES_BEFORE_STALE, heartbeat_stale_after_seconds

# The reconnect path's own bounded cost: a connect retry budget either side of
# a geo lookup (RECONNECT_BUDGET_S + 15.0 in ray_cluster._connect_to_cluster).
RECONNECT_BOUNDED_COST_S = 30.0
# Defaults of Worker.monitoring_interval_seconds and the monitor loop's
# backoff_max_seconds, which together set the liveness staleness deadline.
DEFAULT_MONITORING_INTERVAL_S = 10.0
DEFAULT_BACKOFF_MAX_S = 60.0
# Slowest healthy call measured on ray 2.55.1 — the cold first one, which
# dominates the warm p95 of 33 ms.
MEASURED_COLD_CALL_S = 0.322
# Outer guard so an unbounded helper fails the suite instead of hanging it.
# It must stay well above the patched bound for the elapsed assertions below
# to tell the two apart.
_GUARD_S = 5.0


class _Remote:
    def __init__(self, behaviour):
        self._behaviour = behaviour
        self.calls = 0

    def remote(self):
        self.calls += 1
        return self._behaviour()


class _Handle:
    """Stands in for ``proxy_actor_handle``; only the one method is used."""

    def __init__(self, behaviour):
        self.get_cluster_state = _Remote(behaviour)


def _never_answers():
    """A head that accepted the call and will never reply."""
    return asyncio.get_running_loop().create_future()


async def _answers(payload):
    return payload


def _cluster(handle):
    cluster = RayCluster.__new__(RayCluster)
    cluster.proxy_actor_handle = handle
    cluster.logger = logging.getLogger("test_cluster_state_timeout")
    return cluster


@pytest.mark.asyncio
async def test_a_head_that_stops_answering_raises_instead_of_hanging(monkeypatch):
    """DISCRIMINATION CONTROL for this whole file.

    Drop the ``asyncio.wait_for`` from ``_get_cluster_state_bounded`` and this
    fails on the elapsed assertion at 5.00 s, measured — the outer guard is
    what turns a never-resolving future into a failure instead of a hang.
    """
    monkeypatch.setattr(rc_mod, "CLUSTER_STATE_TIMEOUT_S", 0.1)
    cluster = _cluster(_Handle(_never_answers))

    started = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(cluster._get_cluster_state_bounded(), timeout=_GUARD_S)
    elapsed = time.monotonic() - started

    # The elapsed assertion is what makes this a control rather than a
    # tautology: the outer guard above raises the same TimeoutError, so
    # without it a helper with no bound of its own would still "pass".
    assert elapsed < _GUARD_S / 2


@pytest.mark.asyncio
async def test_a_healthy_call_is_returned_untouched(monkeypatch):
    """Positive control: the bound must not fire on a head that does answer.

    Without this, the test above passes just as well against a helper that
    raises unconditionally.
    """
    monkeypatch.setattr(rc_mod, "CLUSTER_STATE_TIMEOUT_S", 0.1)
    payload = {"cluster": {}, "nodes": {}}
    cluster = _cluster(_Handle(lambda: _answers(payload)))

    assert await cluster._get_cluster_state_bounded() is payload


@pytest.mark.asyncio
async def test_a_timeout_is_not_mistaken_for_a_stale_handle(monkeypatch):
    """A silent head is not a recycled actor, so it must not trigger a reconnect.

    Reconnecting would spend the reconnect budget and then await the same
    unanswerable call again; the error path is what we want instead.
    """
    monkeypatch.setattr(rc_mod, "CLUSTER_STATE_TIMEOUT_S", 0.1)
    cluster = _cluster(_Handle(_never_answers))
    reconnects = []

    async def _reconnect():
        reconnects.append(1)

    cluster.reconnect = _reconnect

    started = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            cluster._get_cluster_state_with_reconnect(), timeout=_GUARD_S
        )
    elapsed = time.monotonic() - started

    assert elapsed < _GUARD_S / 2
    assert reconnects == []
    assert cluster.proxy_actor_handle.get_cluster_state.calls == 1


@pytest.mark.asyncio
async def test_a_stale_handle_still_reconnects_and_retries_once(monkeypatch):
    """No regression: bounding the call must not cost the stale-handle recovery."""
    monkeypatch.setattr(rc_mod, "CLUSTER_STATE_TIMEOUT_S", 5.0)
    payload = {"cluster": {}, "nodes": {}}

    def _stale():
        raise Exception("ray client doesn't have a handle for actor id")

    cluster = _cluster(_Handle(_stale))

    async def _reconnect():
        cluster.proxy_actor_handle = _Handle(lambda: _answers(payload))

    cluster.reconnect = _reconnect

    assert await cluster._get_cluster_state_with_reconnect() is payload


def test_the_timeout_reaches_the_per_step_error_counter():
    """``Worker._step`` catches ``Exception`` and re-raises ``CancelledError``.

    A ``TimeoutError`` therefore lands in ``step_errors``, which makes the pass
    raise and ``_monitor_consecutive_errors`` increment. This pins the two type
    relationships that routing depends on.
    """
    assert issubclass(asyncio.TimeoutError, Exception)
    assert not issubclass(asyncio.TimeoutError, asyncio.CancelledError)


def test_the_bound_leaves_a_pass_inside_the_liveness_staleness_deadline():
    """The value is justified by the ceiling, not copied from the geo call.

    The ceiling is the binding constraint: the worst path through
    ``_get_cluster_state_with_reconnect`` crosses the bound three times — once
    before the reconnect, once inside the connection probe, once on the retry —
    around the reconnect's own bounded cost, and that has to still fit inside
    the heartbeat staleness deadline.

    There is no floor that also satisfies it. A healthy call measured p95 33 ms
    (max 67 ms, 322 ms cold) on Ray 2.55.1, but in slurm mode
    ``check_pending_resources`` adds three Ray state-API calls at
    DEFAULT_RPC_TIMEOUT, so a slow-but-healthy worst case is ~92 s — above
    anything that can be crossed three times inside the deadline. The bound
    therefore cuts a stalled GCS deliberately; that is the behaviour, not a
    gap in it.
    """
    bound = rc_mod.CLUSTER_STATE_TIMEOUT_S
    deadline = heartbeat_stale_after_seconds(
        DEFAULT_MONITORING_INTERVAL_S, DEFAULT_BACKOFF_MAX_S
    )

    assert deadline == DEFAULT_BACKOFF_MAX_S + MISSED_PASSES_BEFORE_STALE * 10.0
    assert bound > MEASURED_COLD_CALL_S * 10
    assert 3 * bound + RECONNECT_BOUNDED_COST_S < deadline


def test_no_unbounded_get_cluster_state_await_is_left_in_ray_cluster():
    """Every call site routes through the bounded helper.

    The reported defect named two call sites, but the connection probe reached
    from ``reconnect()`` is a third on the same monitoring pass. A line-oriented
    grep would not notice a fourth being added, so this walks the AST.
    """
    source = Path(rc_mod.__file__).read_text()
    tree = ast.parse(source)

    enclosing = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                enclosing.setdefault(id(child), node.name)

    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "remote"):
            continue
        inner = func.value
        if isinstance(inner, ast.Attribute) and inner.attr == "get_cluster_state":
            sites.append((enclosing.get(id(node), "<module>"), node.lineno))

    assert [name for name, _ in sites] == ["_get_cluster_state_bounded"], sites
