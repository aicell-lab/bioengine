"""``stop_app`` must not replace the handle to an undeployment that is still running.

THE DEFECT. ``stop_app`` ends by assigning, unconditionally::

    undeployment_task = asyncio.create_task(self._undeploy_application(...))
    self._deployed_applications[application_id]["undeployment_task"] = undeployment_task

A second ``stop_app`` for the same application, arriving while the first teardown is still in
flight, therefore OVERWRITES the handle to the first. And because ``_undeploy_application``
correctly no-ops a stop of an application already marked ``stopping``, that second task
finishes almost at once — so the stored handle ends up pointing at a COMPLETED no-op while
the task actually tearing the application down is referenced by nothing.

THE HARM is to whoever awaits the stored handle to decide whether teardown finished:
``stop_all_apps`` awaits per-application undeployment with a timeout. It is told the work is
done while it is still running.

This is NOT about an orphan. The real teardown still completes and the no-op does not
interfere with it, so the guarantee the undeploy arms pin is untouched. It is about a handle
that lies, which is why these arms assert on the handle and on what awaiting it reports —
never on Serve's contents.

The no-op behaviour is the REAL one: these arms drive the shipped ``_undeploy_application``
through the stubs the undeploy arms already use, rather than a local imitation of it. An
imitation of the thing under test is how a fake ends up reporting its own gap as the defect.
"""

from __future__ import annotations

import asyncio

import pytest

from bioengine.apps.manager import AppsManager

from .test_undeploy_cannot_leave_an_orphan_in_serve import (
    APP,
    _ClusterStub,
    _Logger,
    _ManagerStub,
)

CTX = {"user": {"id": "tester"}}


class _BlockingCluster(_ClusterStub):
    """A Serve whose delete does not return until the arm lets it.

    This is what holds the first teardown open, so the second ``stop_app`` really does arrive
    mid-flight. Blocking inside the delete (rather than patching the undeploy) keeps the
    shipped undeploy code on the path.
    """

    def __init__(self, present_after_delete):
        super().__init__(present_after_delete)
        self.delete_reached = asyncio.Event()
        self.release = asyncio.Event()
        self.blocked_once = False

    async def check_connection(self):
        """``stop_app`` checks the Ray connection first; the undeploy arms never did.

        Present because the FIRST run of these arms failed on its absence — all four, with an
        AttributeError, which looks like a RED arm and is not one. A fake smaller than the
        surface reports its own gap as the defect.
        """
        return None

    async def call_with_reconnect(self, fn, *args):
        name = getattr(fn, "__name__", "")
        if name == "delete" and not self.blocked_once:
            self.blocked_once = True
            self.delete_reached.set()
            await self.release.wait()
        return await super().call_with_reconnect(fn, *args)


class _StopStub(_ManagerStub):
    """``_ManagerStub`` plus the few things ``stop_app`` touches before the assignment.

    ``stop_app`` calls ``_check_initialized()``, awaits ``ray_cluster.check_connection()`` and
    runs ``check_permissions`` against ``admin_users``. None of those is what these arms are
    about, so they are satisfied rather than exercised — but they are satisfied HONESTLY:
    ``admin_users`` really contains the caller, so the permission check really passes.
    """

    def __init__(self, cluster, logger):
        super().__init__(cluster, logger)
        self.admin_users = ["tester"]

    def _check_initialized(self):
        return None


def _stop_app():
    """The shipped ``stop_app`` body.

    ``stop_app`` is wrapped by ``@schema_method``, whose wrapper validates Field defaults and
    is not what these arms are about; ``__wrapped__`` is the function the wrapper calls. If the
    attribute ever disappears the arms FAIL with a reason rather than silently testing the
    wrapper.
    """
    fn = getattr(AppsManager, "stop_app", None)
    inner = getattr(fn, "__wrapped__", None) or fn
    if inner is None:
        pytest.fail(
            "AppsManager.stop_app does not exist, so nothing stores an undeployment handle"
        )
    return inner


async def _first_stop_in_flight():
    """Drive one stop_app and return (mgr, cluster, first_task) with teardown 1 still running."""
    cluster = _BlockingCluster(present_after_delete=[[]])
    mgr = _StopStub(cluster, _Logger())

    await _stop_app()(mgr, application_id=APP, context=CTX)

    first = mgr._deployed_applications[APP].get("undeployment_task")
    assert (
        first is not None
    ), "stop_app stored no undeployment_task at all, so nothing can await the teardown"
    # Let teardown 1 actually get going and reach the delete, so the second stop is genuinely
    # concurrent rather than arriving before the first task has started.
    await asyncio.wait_for(cluster.delete_reached.wait(), timeout=5)
    assert not first.done(), "teardown 1 finished before the arm could race it"
    return mgr, cluster, first


async def _drain(mgr, cluster, *tasks):
    cluster.release.set()
    for t in tasks:
        if t is not None:
            try:
                await asyncio.wait_for(t, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass


@pytest.mark.asyncio
async def test_a_second_stop_does_not_replace_the_handle_to_the_running_teardown() -> (
    None
):
    """RED today: the stored handle is the second call's task, not the in-flight one."""
    mgr, cluster, first = await _first_stop_in_flight()

    await _stop_app()(mgr, application_id=APP, context=CTX)
    stored = mgr._deployed_applications[APP].get("undeployment_task")

    try:
        assert stored is first, (
            "a second stop_app replaced the handle to the undeployment that is still "
            "running; whoever awaits this handle is now awaiting the wrong task"
        )
    finally:
        await _drain(mgr, cluster, first, stored)


@pytest.mark.asyncio
async def test_awaiting_the_stored_handle_does_not_report_a_teardown_that_is_still_running() -> (
    None
):
    """The HARM, asserted separately from the handle's identity.

    Keeping this apart from the arm above matters: storing the right object and reporting the
    right thing are two claims, and a fix that satisfied one while leaving the other would
    otherwise pass.
    """
    mgr, cluster, first = await _first_stop_in_flight()

    await _stop_app()(mgr, application_id=APP, context=CTX)
    stored = mgr._deployed_applications[APP].get("undeployment_task")

    # Teardown 1 is still blocked in the delete. Awaiting the stored handle must NOT come back
    # claiming the undeployment is over.
    finished_early = True
    try:
        await asyncio.wait_for(asyncio.shield(stored), timeout=1.0)
    except asyncio.TimeoutError:
        finished_early = False

    try:
        assert not finished_early, (
            "awaiting the stored undeployment_task returned while the real teardown was "
            "still blocked, so a caller such as stop_all_apps is told the application was "
            "undeployed when it was not"
        )
        assert (
            not first.done()
        ), "the real teardown ended on its own; this arm no longer measures what it claims"
    finally:
        await _drain(mgr, cluster, first, stored)


@pytest.mark.asyncio
async def test_the_second_stop_is_still_a_no_op_and_leaves_one_teardown() -> None:
    """Regression guard on the #437 behaviour this must not disturb.

    GREEN today, and it must stay green: the second stop must not start a second real
    teardown. Without this, the obvious wrong fix — refusing the second call outright, or
    letting it run a full teardown — would pass the two arms above.
    """
    mgr, cluster, first = await _first_stop_in_flight()

    await _stop_app()(mgr, application_id=APP, context=CTX)
    stored = mgr._deployed_applications[APP].get("undeployment_task")
    await _drain(mgr, cluster, first, stored)

    assert cluster.deletes == 1, (
        f"expected exactly one real teardown across two stop_app calls, saw "
        f"{cluster.deletes} delete(s)"
    )


@pytest.mark.asyncio
async def test_a_stop_after_the_first_teardown_finished_may_store_a_fresh_handle() -> (
    None
):
    """The fix must key on IN-FLIGHT, not on the key being present.

    Guards the lazy implementation that never overwrites once a handle exists: a later,
    unrelated stop of an application whose previous teardown has completed is entitled to
    store its own handle. GREEN today.
    """
    mgr, cluster, first = await _first_stop_in_flight()
    await _drain(mgr, cluster, first)
    assert (
        first.done()
    ), "teardown 1 did not finish, so this arm cannot test the settled case"

    # The app may have been dropped from tracking by the completed teardown; re-add it so the
    # second stop has something to stop. This is the "deployed again later" shape.
    mgr._deployed_applications.setdefault(APP, {"auto_redeploy": True})
    mgr._deployed_applications[APP].pop("stopping", None)
    cluster.blocked_once = True  # do not block this one
    cluster.release.set()

    await _stop_app()(mgr, application_id=APP, context=CTX)
    stored = mgr._deployed_applications[APP].get("undeployment_task")
    try:
        assert stored is not None and stored is not first, (
            "a stop issued after the previous teardown completed must be able to store its "
            "own handle, otherwise the handle goes stale for ever"
        )
    finally:
        await _drain(mgr, cluster, stored)
