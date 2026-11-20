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
from ray.exceptions import RayTaskError

from tests.apps._proxy_double import ProxyDouble


class _Handle:
    """Stands in for the entry DeploymentHandle. ``hold`` is how long a call takes."""

    def __init__(self, hold: float = 0.0):
        self.hold = hold
        self.calls = 0
        self.responses = []
        self.raises: BaseException | None = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return _Method(self, name)


class _Response:
    """Stands in for a Ray Serve ``DeploymentResponse``: awaitable, and cancellable.

    A plain coroutine will not do. The production path calls ``.cancel()`` on whatever ``remote()``
    returned, and a coroutine has no such method — so a coroutine-returning double makes the cancel
    path raise ``AttributeError`` instead of exercising it, and the arm guarding that path would be
    testing the double rather than the code. Measured 2026-11-12 on Ray Serve 2.55.1: the real
    object is awaitable, accepted by ``asyncio.wait_for``, and ``cancel()`` after a ``wait_for``
    timeout is safe and does reach the replica.
    """

    def __init__(self, handle: "_Handle", seconds: float):
        self.handle = handle
        self.cancelled = False
        self._task = asyncio.ensure_future(self._run(seconds))
        handle.responses.append(self)

    async def _run(self, seconds: float):
        await asyncio.sleep(seconds)
        if self.handle.raises is not None:
            raise self.handle.raises
        return "ok"

    def __await__(self):
        return self._task.__await__()

    def cancel(self):
        self.cancelled = True
        self._task.cancel()


class _Method:
    def __init__(self, handle: _Handle, name: str):
        self.handle = handle
        self.name = name

    def remote(self, *args, **kwargs):
        self.handle.calls += 1
        return _Response(self.handle, self.handle.hold)


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
async def test_the_timeout_cancels_the_call_and_not_only_the_wait():
    """The timeout must bound the WORK, not just the proxy's wait.

    Measured on Ray Serve 2.55.1: letting ``wait_for`` expire without cancelling leaves the replica
    running the abandoned work to completion. That interacts badly with the queue above — freeing
    admission slots faster admits more work onto a deployment still busy with calls nobody is
    waiting for — so the cancel is part of the fix rather than a tidy-up.
    """
    p = _proxy(max_ongoing=1, max_queued=1, timeout_s=0.2, hold=30)
    fn = _fn(p)
    with pytest.raises(TimeoutError):
        await fn(context=CTX)
    assert p.entry_deployment_handle.responses, "no response object was created"
    assert p.entry_deployment_handle.responses[-1].cancelled, (
        "the call was abandoned but never cancelled, so the replica keeps working"
    )


@pytest.mark.asyncio
async def test_a_completed_call_is_never_cancelled():
    """The discriminator for the arm above: cancel must fire on the timeout path ONLY.

    Without this, ``response.cancel()`` placed unconditionally would satisfy the previous test
    while cancelling every successful call.
    """
    p = _proxy(max_ongoing=1, max_queued=1, timeout_s=10, hold=0.0)
    fn = _fn(p)
    assert await fn(context=CTX) == "ok"
    assert not p.entry_deployment_handle.responses[-1].cancelled


def _remote_error(cause: BaseException):
    """What Ray actually hands the proxy when the app raises ``cause``.

    ``as_instanceof_cause`` is the real constructor rather than a hand-rolled subclass, because the
    whole hazard is the dynamic class it builds: bases ``(RayTaskError, type(cause))``, so the
    result satisfies ``isinstance`` for both — and on 3.11 ``asyncio.TimeoutError`` IS
    ``builtins.TimeoutError``. A double that merely raised ``TimeoutError`` would miss the half of
    the bug that comes from the RayTaskError side.

    The traceback string carries the cause's text because that is the only way the app's message
    reaches the caller: ``str(RayTaskError)`` returns ``traceback_str``, not the cause. A fixture
    with a placeholder there cannot tell "the message survived" from "the message was discarded",
    so it would report the bug fixed either way.
    """
    tb = "Traceback (most recent call last):\n  ...\n%s: %s" % (type(cause).__name__, cause)
    return RayTaskError("slow", tb, cause).as_instanceof_cause()


@pytest.mark.asyncio
async def test_an_apps_own_timeout_error_is_not_claimed_by_the_proxy_bound():
    """FAILS WITHOUT THE CLAUSE ORDER: ``except asyncio.TimeoutError`` first claims this one.

    An app whose upstream read times out raises TimeoutError; Ray delivers it as a subclass of both
    RayTaskError and TimeoutError, and Python takes the first matching clause. Written before the
    RayTaskError clause, the proxy-bound handler discards the app's message and tells the caller a
    bound expired — here 10 s, against a call that failed instantly. The caller then debugs the
    wrong system.
    """
    p = _proxy(max_ongoing=1, max_queued=1, timeout_s=10, hold=0.0)
    p.entry_deployment_handle.raises = _remote_error(TimeoutError("upstream read timed out"))
    fn = _fn(p)
    with pytest.raises(Exception) as got:
        await fn(context=CTX)
    assert isinstance(got.value, RayTaskError), (
        "the app's error was replaced by a bare TimeoutError: %r" % (got.value,)
    )
    assert "upstream read timed out" in str(got.value), "the app's own message was discarded"
    assert "did not complete within" not in str(got.value), (
        "the caller was told the proxy bound expired, but it never did"
    )


@pytest.mark.asyncio
async def test_an_apps_own_timeout_error_is_not_cancelled():
    """FAILS WITHOUT THE INNER GUARD, and passes on the clause reorder alone — so it pins the
    other half of the fix.

    The reorder fixes the message; it leaves the inner handler calling ``cancel()`` on a call that
    already finished, because that handler cannot tell the two kinds of TimeoutError apart either.
    Harmless today only because the response is done; the guard is what makes the cancel mean
    "our bound expired" rather than "something named TimeoutError happened".
    """
    p = _proxy(max_ongoing=1, max_queued=1, timeout_s=10, hold=0.0)
    p.entry_deployment_handle.raises = _remote_error(TimeoutError("upstream read timed out"))
    fn = _fn(p)
    with pytest.raises(Exception):
        await fn(context=CTX)
    assert not p.entry_deployment_handle.responses[-1].cancelled, (
        "a call that failed on its own was cancelled as if the proxy bound had expired"
    )


@pytest.mark.asyncio
async def test_a_non_timeout_app_error_is_unaffected():
    """The control for the two arms above: the chain must discriminate by TYPE.

    Without this, a change that simply stopped raising the bound message for everything would
    satisfy both — this arm is identical before and after the fix, so it isolates the regression
    to TimeoutError-shaped causes.
    """
    p = _proxy(max_ongoing=1, max_queued=1, timeout_s=10, hold=0.0)
    p.entry_deployment_handle.raises = _remote_error(ValueError("bad input"))
    fn = _fn(p)
    with pytest.raises(Exception) as got:
        await fn(context=CTX)
    assert isinstance(got.value, RayTaskError)
    assert "bad input" in str(got.value)
    assert not p.entry_deployment_handle.responses[-1].cancelled


@pytest.mark.asyncio
async def test_a_null_timeout_restores_the_unbounded_wait():
    """``request_timeout_s=None`` must mean "no bound", so the behaviour stays configurable.

    Without this arm the timeout could be hardcoded and every test above would still pass.
    """
    p = _proxy(max_ongoing=1, max_queued=1, timeout_s=None, hold=0.5)
    fn = _fn(p)
    assert await asyncio.wait_for(fn(context=CTX), timeout=10) == "ok"
    assert p.usage.count("failed") == 0
