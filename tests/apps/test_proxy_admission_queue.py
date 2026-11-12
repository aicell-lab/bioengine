"""Pin that the proxy QUEUES at capacity and bounds how long a slot may be held.

Two changes land together here and neither is safe alone, which is why they are tested together.

Rejecting the instant the admission semaphore filled destroyed the only input Ray Serve's
autoscaler reads: a refused call never becomes an ongoing request, so a saturated app never asked
for a second replica. Waiting instead lets the call reach the entry handle and be counted.

But waiting alone is worse than refusing. Measured 2026-11-12 against the worker image's
hypha_rpc 0.21.40: a caller that times out does NOT cancel the handler coroutine, and neither does
a caller that disconnects outright — the slot stayed held for a 50 s observation window in both
cases, well past the heartbeat interval that could have cancelled it. So nothing outside the actor
ever ends a wedged call's hold on its slot. With queueing enabled and no timeout, saturation would
outlast every caller that caused it instead of draining as callers give up.

Each test below names which of the two it fails without.
"""
from __future__ import annotations

import asyncio

import pytest

from tests.apps._proxy_double import ProxyDouble


class _Handle:
    """Stands in for the entry DeploymentHandle. ``hold`` is how long a call takes."""

    def __init__(self, hold: float = 0.0):
        self.hold = hold
        self.calls = 0

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return _Method(self, name)


class _Method:
    def __init__(self, handle: _Handle, name: str):
        self.handle = handle
        self.name = name

    async def remote(self, *args, **kwargs):
        self.handle.calls += 1
        await asyncio.sleep(self.handle.hold)
        return "ok"


def _proxy(max_ongoing: int, max_queued: int, timeout_s, hold: float = 0.0) -> ProxyDouble:
    """A proxy with only the admission path wired up.

    Built with ``__new__`` rather than ``__init__`` on purpose: the real constructor wants a Hypha
    token, a TTL, an actor name and a live handle, none of which the admission path reads. Setting
    exactly the attributes under test keeps the arms from passing or failing for reasons that have
    nothing to do with admission.
    """
    p = ProxyDouble.__new__(ProxyDouble)
    p.application_id = "test-app"
    p.workspace = "ws"
    p.authorized_users = {"*": ["*"]}
    p.max_ongoing_requests = max_ongoing
    p.max_queued_requests = max_queued
    p.request_timeout_s = timeout_s
    p.service_semaphore = asyncio.Semaphore(max_ongoing)
    p._queued_requests = 0
    p.entry_deployment_handle = _Handle(hold=hold)
    p.usage = []

    async def _check_permissions(context, method_name=None):
        return None

    def _record_usage(method_name, caller_class, outcome):
        p.usage.append(outcome)

    p._check_permissions = _check_permissions
    p._record_usage = _record_usage
    return p


def _fn(p: ProxyDouble):
    return p._create_deployment_function({"name": "infer", "description": "", "parameters": {}})


CTX = {"user": {"id": "u1", "email": "u1@example.com"}}


@pytest.mark.asyncio
async def test_a_second_call_waits_instead_of_being_refused():
    """FAILS WITHOUT THE QUEUE: on the old code the second call raised at once.

    The discriminator is that both calls are *answered*. A test that only asserted "no exception"
    would also pass if the second call were silently dropped.
    """
    p = _proxy(max_ongoing=1, max_queued=1, timeout_s=None, hold=0.3)
    fn = _fn(p)
    results = await asyncio.gather(fn(context=CTX), fn(context=CTX))
    assert results == ["ok", "ok"]
    assert p.entry_deployment_handle.calls == 2, "the queued call never reached the entry handle"
    assert "rejected" not in p.usage, "a call that fits in the queue must not be recorded rejected"


@pytest.mark.asyncio
async def test_the_queue_is_bounded_and_the_overflow_is_still_refused():
    """FAILS IF THE QUEUE IS UNBOUNDED: a waiter holds its decoded payload in actor memory.

    cap 1 + queue 1 admits two callers; the third must be turned away rather than parked.
    """
    p = _proxy(max_ongoing=1, max_queued=1, timeout_s=None, hold=0.4)
    fn = _fn(p)
    settled = await asyncio.gather(
        fn(context=CTX), fn(context=CTX), fn(context=CTX), return_exceptions=True
    )
    errors = [r for r in settled if isinstance(r, BaseException)]
    assert len(errors) == 1, "exactly one of three should overflow, got %r" % (settled,)
    assert "queue" in str(errors[0]).lower(), (
        "the refusal must say the queue was full, not just the concurrency limit: %s" % errors[0]
    )
    assert p.usage.count("rejected") == 1


@pytest.mark.asyncio
async def test_a_wedged_call_releases_its_slot_on_the_timeout():
    """FAILS WITHOUT THE TIMEOUT: nothing else ever releases the slot.

    Two assertions, and the second is the one that matters. Raising TimeoutError is easy; what the
    change is for is that the NEXT caller can then be served. Without the release, the follow-up
    call hangs until the wedged work finishes.
    """
    p = _proxy(max_ongoing=1, max_queued=1, timeout_s=0.2, hold=30)
    fn = _fn(p)
    with pytest.raises(TimeoutError):
        await fn(context=CTX)
    assert p.usage.count("failed") == 1

    p.entry_deployment_handle.hold = 0.0
    assert await asyncio.wait_for(fn(context=CTX), timeout=5) == "ok", (
        "the slot was not released, so a later caller inherits the wedge"
    )


@pytest.mark.asyncio
async def test_the_queue_count_returns_to_zero_on_every_exit_path():
    """A leaked waiter count would refuse callers forever once it drifted up to the bound.

    Covers all three exits — answered, refused, timed out — because the decrement lives in a
    ``finally`` that only one of them exercises.
    """
    p = _proxy(max_ongoing=1, max_queued=1, timeout_s=0.2, hold=0.0)
    fn = _fn(p)
    await fn(context=CTX)
    assert p._queued_requests == 0, "answered path leaked a waiter"

    p.entry_deployment_handle.hold = 30
    with pytest.raises(TimeoutError):
        await fn(context=CTX)
    assert p._queued_requests == 0, "timeout path leaked a waiter"

    p.entry_deployment_handle.hold = 0.4
    settled = await asyncio.gather(
        fn(context=CTX), fn(context=CTX), fn(context=CTX), return_exceptions=True
    )
    assert any(isinstance(r, BaseException) for r in settled)
    assert p._queued_requests == 0, "refused path leaked a waiter"


@pytest.mark.asyncio
async def test_a_null_timeout_restores_the_unbounded_wait():
    """``request_timeout_s=None`` must mean "no bound", so the behaviour stays configurable.

    Without this arm the timeout could be hardcoded and every test above would still pass.
    """
    p = _proxy(max_ongoing=1, max_queued=1, timeout_s=None, hold=0.5)
    fn = _fn(p)
    assert await asyncio.wait_for(fn(context=CTX), timeout=10) == "ok"
    assert p.usage.count("failed") == 0
