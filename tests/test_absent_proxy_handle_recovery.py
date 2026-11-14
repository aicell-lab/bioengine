"""A ``None`` proxy actor handle routes to the same recovery as a stale one (#0372).

``reconnect()`` and the external-cluster branch of ``check_connection()`` both
drop ``proxy_actor_handle`` immediately before re-acquiring it, so the ``None``
is deliberately observable to a concurrent monitor pass — and a monitor pass
that lands in that window used to dereference it. The resulting
``AttributeError: 'NoneType' object has no attribute 'get_cluster_state'``
contains none of the text ``_is_stale_handle_error`` matches on, so
``_get_cluster_state_with_reconnect`` re-raised instead of reconnecting, and
the real cause — an unreachable head — was reported as a worker bug.

Measured on a live external-cluster worker (``bioengine-worker-local:0.16.89``,
2026-11-12): ``_connect_to_cluster()`` raised, the handle stayed ``None``, and
the next pass logged the ``AttributeError`` in the same second, then once per
tick for 515 consecutive ticks.

WHICH FIX, AND WHY (#0372 clause 1): the call site refuses to dereference a
``None`` handle and raises ``ProxyHandleAbsentError``, which the recovery
predicate accepts by TYPE. The alternative — teaching
``_is_stale_handle_error`` to recognise the ``AttributeError`` — was rejected:
it is a substring match on a CPython message that is not a stable API, and it
would route *any* ``NoneType`` attribute error arising anywhere below this call
to a reconnect that cannot fix it. That is strictly worse than the defect.
"""

import ast
import asyncio
import logging
from pathlib import Path

import pytest

from bioengine.cluster import ray_cluster as rc_mod
from bioengine.cluster.ray_cluster import ProxyHandleAbsentError, RayCluster


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


async def _answers(payload):
    return payload


def _cluster(handle, mode="external-cluster"):
    cluster = RayCluster.__new__(RayCluster)
    cluster.proxy_actor_handle = handle
    cluster.mode = mode
    cluster.logger = logging.getLogger("test_absent_proxy_handle_recovery")
    return cluster


async def _prefix_bounded(cluster):
    """The body this fix replaced, verbatim in behaviour: a bare dereference.

    Used as the MUTANT arm below. ``test_the_source_still_carries_the_guard``
    is what keeps this honest — it asserts the shipping helper differs from
    this one in exactly the two ways claimed, so a revert cannot leave both
    arms passing.
    """
    return await asyncio.wait_for(
        cluster.proxy_actor_handle.get_cluster_state.remote(),
        timeout=rc_mod.CLUSTER_STATE_TIMEOUT_S,
    )


@pytest.mark.asyncio
async def test_a_none_handle_is_refused_rather_than_dereferenced():
    """The absent handle is reported as its own type, not as an AttributeError."""
    cluster = _cluster(None)

    with pytest.raises(ProxyHandleAbsentError):
        await cluster._get_cluster_state_bounded()


@pytest.mark.asyncio
async def test_a_monitor_pass_that_starts_with_no_handle_reconnects_and_succeeds():
    """Clause 2, FIXED arm: the recovery that used to be skipped now runs."""
    payload = {"cluster": {}, "nodes": {}}
    cluster = _cluster(None)
    reconnects = []

    async def _reconnect():
        reconnects.append(1)
        cluster.proxy_actor_handle = _Handle(lambda: _answers(payload))

    cluster.reconnect = _reconnect

    assert await cluster._get_cluster_state_with_reconnect() is payload
    assert reconnects == [1]
    assert cluster.proxy_actor_handle.get_cluster_state.calls == 1


@pytest.mark.asyncio
async def test_the_prefix_body_is_unclassifiable_and_skips_the_reconnect():
    """Clause 2, MUTANT arm — THE DISCRIMINATION CONTROL FOR THIS FILE.

    Restore the pre-fix bare dereference and the test above fails: the
    ``AttributeError`` is not recognised, no reconnect is attempted, and the
    payload is never fetched. This is the defect, reproduced on demand.
    """
    payload = {"cluster": {}, "nodes": {}}
    cluster = _cluster(None)
    reconnects = []

    async def _reconnect():
        reconnects.append(1)
        cluster.proxy_actor_handle = _Handle(lambda: _answers(payload))

    cluster.reconnect = _reconnect
    cluster._get_cluster_state_bounded = lambda: _prefix_bounded(cluster)

    with pytest.raises(AttributeError) as excinfo:
        await cluster._get_cluster_state_with_reconnect()

    # Both halves matter: the shape is an AttributeError about NoneType, and
    # the matcher in front of the recovery cannot see it.
    assert "get_cluster_state" in str(excinfo.value)
    assert not rc_mod._is_stale_handle_error(excinfo.value)
    assert reconnects == []


@pytest.mark.asyncio
async def test_recovery_is_exactly_one_reconnect_and_one_retry():
    """Clause 3: a reconnect that does not restore the handle must not loop.

    The error propagates to ``Worker._step``, which appends it to
    ``step_errors``; the tick then raises, ``_monitor_consecutive_errors``
    increments, the backoff doubles and the degraded threshold still applies.
    That existing degradation is the bound — nothing new retries here.
    """
    cluster = _cluster(None)
    reconnects = []

    async def _reconnect():
        reconnects.append(1)  # a reconnect that silently fails to re-acquire

    cluster.reconnect = _reconnect

    with pytest.raises(ProxyHandleAbsentError):
        await cluster._get_cluster_state_with_reconnect()
    assert reconnects == [1]


def test_the_absent_handle_error_reaches_the_per_step_error_counter():
    """Clause 3, the type relationships ``Worker._step`` routing depends on.

    ``_step`` re-raises ``CancelledError`` and catches ``Exception``, so this
    is what puts the error in ``step_errors`` rather than killing the loop.
    """
    assert issubclass(ProxyHandleAbsentError, Exception)
    assert not issubclass(ProxyHandleAbsentError, asyncio.CancelledError)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["single-machine", "slurm"])
async def test_a_non_client_worker_never_attempts_a_reconnect(mode):
    """Clause 4: outside external-cluster mode an absent handle propagates.

    ``reconnect()`` tears down a Ray Client session; in these modes it would
    stop the cluster this worker started, and the handle is set during startup
    and never dropped, so a ``None`` means the connection never happened —
    which no reconnect can fix.
    """
    cluster = _cluster(None, mode=mode)
    reconnects = []

    async def _reconnect():
        reconnects.append(1)

    cluster.reconnect = _reconnect

    with pytest.raises(ProxyHandleAbsentError):
        await cluster._get_cluster_state_with_reconnect()
    assert reconnects == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["single-machine", "slurm"])
async def test_reconnect_itself_still_refuses_outside_external_cluster_mode(mode):
    """Clause 4, second line of defence.

    The test above would also pass if someone loosened ``reconnect()`` instead
    of the predicate, so pin the refusal at its own site too.
    """
    cluster = _cluster(None, mode=mode)

    with pytest.raises(RuntimeError, match="only safe in external-cluster mode"):
        await cluster.reconnect()


@pytest.mark.asyncio
async def test_a_stale_handle_is_still_recovered():
    """No regression: the pre-existing path still routes through the predicate."""
    payload = {"cluster": {}, "nodes": {}}

    def _stale():
        raise Exception("ray client doesn't have a handle for actor id")

    cluster = _cluster(_Handle(_stale))

    async def _reconnect():
        cluster.proxy_actor_handle = _Handle(lambda: _answers(payload))

    cluster.reconnect = _reconnect

    assert await cluster._get_cluster_state_with_reconnect() is payload


def _bounded_helper_ast():
    """The shipping ``_get_cluster_state_bounded``, read off the installed file."""
    tree = ast.parse(Path(rc_mod.__file__).read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_get_cluster_state_bounded"
        ):
            return node
    return None


def test_the_source_still_carries_the_guard():
    """What keeps the MUTANT arm above from being a comparison against itself.

    The mutant is a hand-written copy of the pre-fix body, so if the shipping
    helper were reverted to that body both arms would pass and the file would
    certify nothing. This asserts the two differences by reading the source:
    the helper raises ``ProxyHandleAbsentError``, and the ``.remote()`` call is
    made on a LOCAL name rather than on ``self.proxy_actor_handle`` — the
    second being what makes the check and the dereference atomic against a
    concurrent ``reconnect()``.
    """
    fn = _bounded_helper_ast()
    assert fn is not None, "no _get_cluster_state_bounded found — update this test"

    raised = [
        n.exc.func.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Raise)
        and isinstance(n.exc, ast.Call)
        and isinstance(n.exc.func, ast.Name)
    ]
    assert "ProxyHandleAbsentError" in raised, raised

    derefs = [
        n.func.value
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "remote"
        and isinstance(n.func.value, ast.Attribute)
        and n.func.value.attr == "get_cluster_state"
    ]
    assert len(derefs) == 1, derefs
    assert isinstance(derefs[0].value, ast.Name), ast.dump(derefs[0])
