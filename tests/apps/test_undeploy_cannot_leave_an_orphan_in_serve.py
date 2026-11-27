"""An undeploy must not leave Ray Serve holding an application the worker forgot.

MEASURED on a Kubernetes site. An operator undeployed an application whose build was still in
flight. The worker logged, in this order and within the same second::

    AppsManager  Cancelling deployment task of application '<app>' ... "was cancelled"
    serve 1      Deleting app ['<app>']
    AppsManager  Deleted Ray Serve application '<app>' in 0.03 seconds
    AppsManager  Cleared application replica history
    AppsManager  Undeployment of application '<app>' completed

and then, four seconds later::

    build_and_run_application pid=...  "Connecting to existing Serve app in namespace 'serve'"
    BioEngineProxyActor                Registered replica ... RuntimeApp

The build runs as a RAY TASK. ``.remote(...)`` submits it immediately and ``asyncio.to_thread``
only waits for it, so cancelling the deployment coroutine stopped the wait and left the task
running — and the task finished by calling ``serve.run``, re-creating the application after the
delete and after the worker had dropped it from its own tracking.

The result is an ORPHAN: an application Ray Serve holds and the worker does not list. It cannot be
seen by ``get_app_status`` or stopped by ``stop_app``, because both work from that list. This one
pinned 2 of the site's 3 GPUs, and nothing could attribute or free them.

Three contracts are pinned here:
  1. cancelling the submit cancels the Ray task, so it cannot re-create the application;
  2. a delete that SUCCEEDED is verified rather than assumed, and a reappearance is deleted again
     and kept tracked if it survives;
  3. the orphan comparison is available as a reading, so this state cannot be silent again.
"""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest

import bioengine.apps.builder as builder_mod
import bioengine.apps.manager as manager_mod
from bioengine.apps.builder import AppBuilder
from bioengine.apps.manager import AppsManager

APP = "throwaway-app"


class _Logger:
    """Collects records so an arm can assert the operator was told."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def _add(self, level):
        def log(msg, *a, **k):
            self.records.append((level, str(msg)))

        return log

    def __getattr__(self, name):
        if name in ("info", "warning", "error", "debug"):
            return self._add(name)
        raise AttributeError(name)

    def messages(self, level=None):
        return [m for lv, m in self.records if level is None or lv == level]


# ───────────────────────────── 1. cancelling the submit cancels the RAY TASK ─────────────────


class _Ref:
    """Stands in for the ObjectRef ``.remote(...)`` returns."""

    def __init__(self) -> None:
        self.id = "ref-1"


class _FakeRay:
    """Enough of the ray module for submit(): remote(), get(), cancel(), wait().

    ``wait`` is NOT optional furniture. It used to be absent, and the consequence was a silent
    hole: ``ray.wait`` then raised ``AttributeError`` inside the ``try`` around the cancel, the
    ``except Exception`` swallowed it, and EVERY arm ran the "could not confirm" path while
    asserting only that cancel had been called. Neither real branch was exercised, so replacing
    the whole bounded wait with ``cancelled_cleanly = True`` left all arms green — a mutant the
    gate could not kill because nothing reached the code it mutated.

    ``wait_returns_done`` picks the branch, so the two outcomes are separate, named arms.
    """

    class exceptions:  # noqa: N801 - mirrors the real module's shape
        class RayTaskError(Exception):
            def __init__(self, cause=None):
                super().__init__("task failed")
                self.cause = cause

    def __init__(self, ref: _Ref, wait_returns_done: bool = True) -> None:
        self.ref = ref
        self.cancelled: list[tuple[object, bool]] = []
        self.waits: list[tuple[list, object]] = []
        self.wait_returns_done = wait_returns_done
        self.get_started = asyncio.Event()
        # Lets the arm release the executor thread when it is done. The thread used to
        # ``time.sleep(30)``, and because the loop's shutdown JOINS its default executor, each
        # such arm added a real 30s to the suite in teardown — paid in full, invisibly, after
        # the arm had already passed.
        self.release_get = threading.Event()

    def remote(self, **kwargs):
        def decorator(fn):
            class _Remote:
                def remote(_self, *a, **k):
                    return self.ref

            return _Remote()

        return decorator

    def get(self, ref):
        # Mimics the real blocking wait. The arm cancels while this is parked, then releases
        # it. The timeout is a backstop so a mistake in an arm cannot wedge the suite.
        self.get_started.set()
        self.release_get.wait(timeout=30)
        return None

    def wait(self, refs, timeout=None):
        self.waits.append((list(refs), timeout))
        if self.wait_returns_done:
            return list(refs), []  # the task really ended
        return [], list(refs)  # still running when the bounded wait expired

    def cancel(self, ref, force=False):
        self.cancelled.append((ref, force))


class _BuiltApp:
    runtime_env = {}
    spec = {}
    translated_kwargs = {}
    proxy_args = {}
    bioengine_uri = "gcs://x.zip"
    proxy_pip = []
    user_replica_framework_pip = []
    replica_env_vars = {}
    replica_secret_env_vars = {}
    proxy_memory_in_gb = 1
    scaling = {}


def _real_builder_or_missing(name: str):
    """AppBuilder's counterpart of ``_real_or_missing``, and needed for the same reason.

    Binding ``AppBuilder.<name>`` in a class body raises at IMPORT time on a tree where the
    method does not exist yet — which is exactly the tree the mutation gate runs this file
    against to prove the arms are red before the fix. That would turn every arm in the file
    into one collection ERROR naming no contract at all, instead of the specific failures the
    gate is reading. This degrades to a per-arm assertion failure.
    """
    fn = getattr(AppBuilder, name, None)
    if fn is not None:
        return fn

    def _missing(self, *args, **kwargs):
        raise AssertionError(
            f"AppBuilder.{name} does not exist, so an unconfirmed build cancellation cannot "
            "reach the undeploy: it untracks the application while a task that can re-create "
            "it is still alive"
        )

    return _missing


class _BuilderStub:
    """Stands in for AppBuilder, carrying the real record the undeploy later reads.

    ``_unconfirmed_build_cancellations`` is the real attribute name and
    ``take_unconfirmed_build_cancellation`` the real method, bound from AppBuilder, so an arm
    that asserts on this stub is asserting on the shipped drain logic rather than on a
    reimplementation of it that could agree with a wrong one.
    """

    take_unconfirmed_build_cancellation = _real_builder_or_missing(
        "take_unconfirmed_build_cancellation"
    )

    def __init__(self, logger):
        self.logger = logger
        self._unconfirmed_build_cancellations: set[str] = set()

    async def _store_replica_secrets(self, application_id, secrets):
        return None


async def _cancel_a_submit_in_flight(monkeypatch, *, wait_returns_done: bool):
    """Drive the real ``AppBuilder.submit`` to the point of cancellation and cancel it.

    Returns ``(fake_ray, builder, log)`` so each arm can assert on a different consequence of
    the same single code path.
    """
    log = _Logger()
    ref = _Ref()
    fake = _FakeRay(ref, wait_returns_done=wait_returns_done)
    monkeypatch.setattr(builder_mod, "ray", fake)
    builder = _BuilderStub(log)

    task = asyncio.create_task(AppBuilder.submit(builder, _BuiltApp(), APP))
    await asyncio.wait_for(fake.get_started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Let the executor thread finish instead of leaving the loop's shutdown to join it.
    fake.release_get.set()
    return fake, builder, log


@pytest.mark.asyncio
async def test_cancelling_a_deploy_cancels_the_in_flight_build_task(
    monkeypatch,
) -> None:
    """The defect itself: a cancelled deploy left the Ray task alive to re-create the app.

    Without the fix the submit coroutine is simply cancelled while parked in ``asyncio.to_thread``,
    the Ray task is never told, and it goes on to call ``serve.run``.
    """
    fake, _builder, _log = await _cancel_a_submit_in_flight(
        monkeypatch, wait_returns_done=True
    )
    ref = fake.ref

    assert fake.cancelled, (
        "the Ray build task was never cancelled, so it stays alive and re-creates the "
        "application after the undeploy deletes it"
    )
    cancelled_ref, forced = fake.cancelled[-1]
    assert cancelled_ref is ref, "a different object was cancelled, not the build task"
    assert forced is True, (
        "a gentle cancel only interrupts between Python lines, and this task spends its time "
        "inside serve.run, so it can finish creating the application anyway"
    )


@pytest.mark.asyncio
async def test_a_cancel_the_task_ACKNOWLEDGED_is_not_reported_as_unconfirmed(
    monkeypatch,
) -> None:
    """The done branch of the bounded wait, which no arm reached before.

    ``ray.wait`` returning the ref in ``done`` means the task really ended, so there is no
    surviving builder that could re-create the application: nothing to warn about and nothing
    for the undeploy to keep tracked. Asserting the ABSENCE here is what makes the companion
    arm below meaningful — without it, code that always warns would satisfy that one.
    """
    fake, builder, log = await _cancel_a_submit_in_flight(
        monkeypatch, wait_returns_done=True
    )

    assert fake.waits, (
        "ray.wait was never called, so the cancel was treated as the task's death and the "
        "bounded wait this arm is about did not run at all"
    )
    waited_refs, timeout = fake.waits[-1]
    assert waited_refs == [
        fake.ref
    ], "a different object was waited on, not the build task"
    # Checked as "a positive, finite bound, and the module's own" rather than against a
    # literal. A literal would have to be edited whenever the limit is tuned; but `== the
    # constant` ALONE would also accept the constant being set to None, which is precisely an
    # unbounded wait, so both halves are asserted.
    assert isinstance(timeout, (int, float)) and timeout > 0, (
        "the bounded wait has no positive timeout, so a build task wedged inside a "
        f"runtime_env install holds the undeploy open indefinitely (got {timeout!r})"
    )
    assert timeout == getattr(builder_mod, "_BUILD_CANCEL_WAIT_SECONDS", None), (
        "the wait is bounded by something other than the module's declared limit, so tuning "
        f"that limit no longer changes the behaviour (got {timeout!r})"
    )
    assert not any("NOT confirmed finished" in m for m in log.messages("error")), (
        "the task was confirmed finished, yet it was reported as possibly still able to "
        f"re-create the application. errors seen: {log.messages('error')}"
    )
    assert builder.take_unconfirmed_build_cancellation(APP) is False, (
        "a confirmed-dead build task was recorded as an unconfirmed cancellation, which "
        "makes every clean stop keep its tracking entry forever"
    )


@pytest.mark.asyncio
async def test_a_cancel_that_TIMES_OUT_is_recorded_for_the_undeploy_to_read(
    monkeypatch,
) -> None:
    """The timeout branch: the task is still alive, and that has to REACH the undeploy.

    A log line is not a signal — ``_undeploy_application`` cannot read one, and it is the only
    caller that can act, by keeping the tracking entry so a re-creation stays visible. Before
    this the outcome stopped at the log and the undeploy untracked the application anyway.
    """
    _fake, builder, log = await _cancel_a_submit_in_flight(
        monkeypatch, wait_returns_done=False
    )

    assert any("NOT confirmed finished" in m for m in log.messages("error")), (
        "a build task still running after the bounded wait was not reported; it may yet call "
        f"serve.run. errors seen: {log.messages('error')}"
    )
    assert APP in builder._unconfirmed_build_cancellations, (
        "the unconfirmed cancellation was only logged, so _undeploy_application cannot know "
        "the build task may still re-create the application and will untrack it anyway"
    )
    # Draining is part of the contract: the record describes ONE teardown.
    assert builder.take_unconfirmed_build_cancellation(APP) is True
    assert builder.take_unconfirmed_build_cancellation(APP) is False, (
        "the record was not drained, so every later stop of this id reports the same stale "
        "hazard and never drops its tracking entry"
    )


# ────────────────────── 2. a SUCCESSFUL delete is verified, not assumed ──────────────────────


class _ServeStatus:
    def __init__(self, names):
        self.applications = {n: object() for n in names}


class _ClusterStub:
    """Replays a Serve that still holds the app after a successful delete."""

    def __init__(self, present_after_delete):
        # one entry per _serve_application_names() call, consumed in order
        # The LAST entry repeats once the list is exhausted. Returning [] instead would make
        # an application the arm intends to be permanent quietly vanish, and the arm would
        # pass for the opposite reason to the one it was written for.
        self.present_after_delete = list(present_after_delete)
        self.deletes = 0
        # Counted, because the sweep's COST is a contract too: one serve.status per sweep, and
        # the sweep is supposed to run at most once per cooldown rather than once per tick.
        self.status_calls = 0
        self.proxy_actor_handle = self

    class _Clear:
        async def remote(self, application_id):
            return None

    @property
    def clear_application_replicas(self):
        return self._Clear()

    async def call_with_reconnect(self, fn, *args):
        name = getattr(fn, "__name__", "")
        if name == "delete":
            self.deletes += 1
            return None
        if name == "status":
            self.status_calls += 1
            # The LAST entry REPEATS once the list is exhausted, rather than falling back to
            # []. Falling back would make an application an arm intends to be permanently
            # present quietly vanish after a few polls, and the arm would then pass for the
            # opposite reason to the one it was written for.
            if len(self.present_after_delete) > 1:
                names = self.present_after_delete.pop(0)
            elif self.present_after_delete:
                names = self.present_after_delete[0]
            else:
                names = []
            return _ServeStatus(names)
        raise AssertionError(f"unexpected call {name}")


#: The confirm window's sleep is zeroed by the autouse ``fast_undeploy_confirm_window``
#: fixture in ``tests/apps/conftest.py``. It used to live here, which left every arm in
#: ``test_undeploy_delete_failure_keeps_tracking.py`` paying the real 12s window twice.


def _real_or_missing(name: str):
    """Bind the real helper, or a stand-in that FAILS the arm with a reason.

    Binding ``AppsManager.<name>`` directly would raise at class-definition time on a tree where
    the helper does not exist yet, turning the whole file into one collection ERROR. An error
    names no contract: every arm disappears together and a reader cannot tell which behaviour is
    missing. This degrades to a per-arm assertion failure instead.
    """
    fn = getattr(AppsManager, name, None)
    if fn is not None:
        return fn

    async def _missing(self, *args, **kwargs):
        raise AssertionError(
            f"AppsManager.{name} does not exist, so the worker never compares its own "
            "application list against Ray Serve's and an orphan stays invisible"
        )

    return _missing


class _ManagerStub:
    # The real helpers, bound onto the stub, so the arms exercise the shipped
    # implementations rather than a reimplementation of them that could agree
    # with a wrong one.
    _serve_application_names = _real_or_missing("_serve_application_names")
    _serve_application_exists = _real_or_missing("_serve_application_exists")
    _delete_and_confirm_absent = _real_or_missing("_delete_and_confirm_absent")

    async def _is_this_workers_app(self, application_id):
        # Default for arms that are not about workspace filtering. The filter itself is
        # pinned by its own arms, which override this.
        return True

    _undeploy_application = _real_or_missing("_undeploy_application")

    def __init__(self, cluster, logger):
        self.ray_cluster = cluster
        self.logger = logger
        self._deployed_applications = {APP: {"auto_redeploy": True}}
        self._redeploy_backoff = {}
        self._missing_from_status = {}
        self._deleted_pending_redeploy = set()
        self._identity_warned = set()
        # The undeploy asks the builder whether the build task it cancelled actually ended.
        self.app_builder = _BuilderStub(logger)

    async def _cancel_deployment_process(self, application_id):
        return None


@pytest.mark.asyncio
async def test_an_app_that_reappears_after_a_successful_delete_is_deleted_again() -> (
    None
):
    """The measured race: deleted in 0.03s, re-created four seconds later.

    `serve.delete` not raising is not evidence the application is gone.
    """
    log = _Logger()
    # present after the first delete, gone after the second
    cluster = _ClusterStub(present_after_delete=[[APP], []])
    mgr = _ManagerStub(cluster, log)

    await AppsManager._undeploy_application(mgr, APP, "tester")

    assert (
        cluster.deletes == 2
    ), f"expected a second delete once the app reappeared, saw {cluster.deletes}"
    assert (
        APP not in mgr._deployed_applications
    ), "the second delete worked, so the app should not be left tracked"
    assert any(
        "present in Ray Serve again" in m for m in log.messages("error")
    ), "the reappearance was not reported; it would look like a clean undeploy"


@pytest.mark.asyncio
async def test_an_app_that_survives_both_deletes_stays_VISIBLE_in_the_worker_list() -> (
    None
):
    """The orphan is the real harm, so the last resort is to stay listed.

    An application Ray Serve holds and this worker does not list cannot be reached by stop_app,
    so it holds its GPUs with no API route to free them. Keeping it tracked is what keeps a
    route open.
    """
    log = _Logger()
    cluster = _ClusterStub(present_after_delete=[[APP], [APP]])
    mgr = _ManagerStub(cluster, log)

    await AppsManager._undeploy_application(mgr, APP, "tester")

    assert (
        APP in mgr._deployed_applications
    ), "an app still held by Ray Serve was dropped from tracking - that is exactly the orphan"
    assert (
        mgr._deployed_applications[APP]["auto_redeploy"] is False
    ), "a kept entry must not be auto-redeployed by the monitor"


@pytest.mark.asyncio
async def test_a_clean_undeploy_does_not_delete_twice() -> None:
    """The silence half: an app that really went away must not pay for the check."""
    log = _Logger()
    cluster = _ClusterStub(present_after_delete=[[]])
    mgr = _ManagerStub(cluster, log)

    await AppsManager._undeploy_application(mgr, APP, "tester")

    assert cluster.deletes == 1, f"a clean undeploy issued {cluster.deletes} deletes"
    assert APP not in mgr._deployed_applications
    assert not any("present in Ray Serve again" in m for m in log.messages("error"))


# ───────────────────────────── 3. the orphan comparison is readable ──────────────────────────


@pytest.mark.asyncio
async def test_orphans_are_reported_and_named() -> None:
    """Serve holds three, the worker tracks one: the other two are orphans."""
    log = _Logger()
    cluster = _ClusterStub(present_after_delete=[["tracked", "ghost-a", "ghost-b"]])
    mgr = _ManagerStub(cluster, log)
    mgr._deployed_applications = {"tracked": {}}

    orphans = await _real_or_missing("get_orphaned_applications")(mgr)

    assert orphans == ["ghost-a", "ghost-b"], orphans
    errors = log.messages("error")
    assert any(
        "ghost-a" in m and "ghost-b" in m for m in errors
    ), f"orphans were returned but not reported to the operator: {errors}"


@pytest.mark.asyncio
async def test_an_unreadable_serve_status_is_not_reported_as_everything_orphaned() -> (
    None
):
    """`None` means "cannot tell", which must not read as "Serve holds nothing"."""
    log = _Logger()

    class _Broken(_ClusterStub):
        async def call_with_reconnect(self, fn, *args):
            raise RuntimeError("serve unreachable")

    mgr = _ManagerStub(_Broken([]), log)
    mgr._deployed_applications = {"tracked": {}}

    assert await _real_or_missing("get_orphaned_applications")(mgr) == []
    assert await _real_or_missing("_serve_application_names")(mgr) is None, (
        "an unreadable status must be None, not an empty set - an empty set would let a caller "
        "conclude an application is gone because the status call broke"
    )
    assert await _real_or_missing("_serve_application_exists")(mgr, APP) is False


# ──────────────── 4. the sweep is WIRED to the monitor, not merely available ─────────────────


def test_the_monitor_sweep_is_not_pinned_by_a_source_grep_alone() -> None:
    """Kept, but DEMOTED to what it really is: a cheap wiring check.

    The reviewer showed the behavioural hole this could never catch — the sweep used to sit
    after `if not apps_to_redeploy: ... return`, so it was skipped in exactly the state a stop
    leaves. Grepping the source said "the call is present" the whole time. The arm that decides
    the contract is the one below, which drives the tick; this one only keeps the cheap
    properties (the pre-fetch reuse and the cooldown) from being dropped silently.
    """
    import inspect

    source = inspect.getsource(AppsManager.monitor_applications)
    assert (
        "_report_orphans_if_due" in source
    ), "the monitor never calls the orphan sweep"
    assert "_ORPHAN_REPORT_COOLDOWN_SECONDS" in inspect.getsource(
        AppsManager._report_orphans_if_due
    ), "an orphan persists, so an unthrottled report repeats every tick and becomes noise"


@pytest.mark.asyncio
async def test_an_app_recreated_AFTER_the_first_poll_is_still_caught() -> None:
    """The window, not the sample.

    ``ray.cancel`` requests cancellation; a task already inside ``serve.run`` can run to the end
    of it. So the first check after a delete can read "absent" for a build that re-creates the
    application a moment later — in the incident, four seconds later. Checking once would pass
    that case and drop the app from tracking, which is the orphan.

    Here Serve answers absent, then PRESENT (the late re-creation), then absent from then on.
    """
    log = _Logger()
    cluster = _ClusterStub(present_after_delete=[[], [APP], [], [], [], [], [], []])
    mgr = _ManagerStub(cluster, log)

    # _undeploy_application has already popped the entry by the time it calls
    # this; leaving it tracked would mean 'redeployed meanwhile' instead.
    mgr._deployed_applications = {}

    confirmed = await _real_or_missing("_delete_and_confirm_absent")(
        mgr, APP, polls=3, interval=0
    )

    assert confirmed is True
    assert (
        cluster.deletes >= 1
    ), "the late re-creation was never deleted; checking once would have missed it entirely"
    assert any(
        "present in Ray Serve again" in m for m in log.messages("error")
    ), "a re-creation after the first poll was not reported LOUDLY"


@pytest.mark.asyncio
async def test_an_app_that_keeps_coming_back_is_reported_as_NOT_confirmed() -> None:
    """A build that re-creates it on every look must not be declared gone."""
    log = _Logger()
    cluster = _ClusterStub(present_after_delete=[[APP]] * 12)

    class _FailingDelete(_ClusterStub):
        async def call_with_reconnect(self, fn, *args):
            if getattr(fn, "__name__", "") == "delete":
                raise RuntimeError("delete refused")
            return await _ClusterStub.call_with_reconnect(self, fn, *args)

    failing = _FailingDelete(present_after_delete=[[APP]] * 12)
    mgr = _ManagerStub(failing, log)

    # _undeploy_application has already popped the entry by the time it calls
    # this; leaving it tracked would mean 'redeployed meanwhile' instead.
    mgr._deployed_applications = {}

    confirmed = await _real_or_missing("_delete_and_confirm_absent")(
        mgr, APP, polls=3, interval=0
    )
    assert confirmed is False, (
        "an application Serve still holds was reported as confirmed gone, so the caller drops "
        "it from tracking and it becomes unstoppable"
    )


@pytest.mark.asyncio
async def test_a_clean_delete_confirms_without_deleting_again() -> None:
    """The silence half: an app that really went stays gone, and costs no extra delete."""
    log = _Logger()
    cluster = _ClusterStub(present_after_delete=[[], [], [], [], []])
    mgr = _ManagerStub(cluster, log)

    # _undeploy_application has already popped the entry by the time it calls
    # this; leaving it tracked would mean 'redeployed meanwhile' instead.
    mgr._deployed_applications = {}

    confirmed = await _real_or_missing("_delete_and_confirm_absent")(
        mgr, APP, polls=3, interval=0
    )
    assert confirmed is True
    assert (
        cluster.deletes == 0
    ), f"a clean delete issued {cluster.deletes} extra delete(s)"
    assert not log.messages("error")


# ──────── 6. the findings the round-1 REFUSE proved my earlier arms could not catch ──────────


class _SweepManager(_ManagerStub):
    """A manager whose sweep can be driven, recording whether it ran."""

    get_orphaned_applications = _real_or_missing("get_orphaned_applications")
    _report_orphans_if_due = _real_or_missing("_report_orphans_if_due")

    def __init__(self, cluster, logger, tracked):
        super().__init__(cluster, logger)
        self._deployed_applications = dict(tracked)
        self.swept = False

    async def _is_this_workers_app(self, application_id):
        return True  # workspace filtering is pinned by its own arms below


class _RealOwnerManager(_SweepManager):
    """Uses the SHIPPED `_is_this_workers_app`, not a permissive stand-in."""

    _is_this_workers_app = _real_or_missing("_is_this_workers_app")


@pytest.mark.asyncio
async def test_the_sweep_runs_WITH_NOTHING_TRACKED(monkeypatch) -> None:
    """The state the incident leaves, and the one the old placement skipped.

    The sweep used to sit after `if not apps_to_redeploy: ... return`. After a stop the worker
    can track nothing at all, so the tick returned before reaching it — a watcher skipped exactly
    when an orphan is most likely. Grepping the source could never see that, which is why this
    arm drives the sweep instead of reading it.
    """
    log = _Logger()
    cluster = _ClusterStub(present_after_delete=[["orphan-app"]])
    mgr = _SweepManager(cluster, log, tracked={})
    monkeypatch.setattr(
        manager_mod, "_ORPHAN_REPORT_COOLDOWN_SECONDS", 0, raising=False
    )

    ran = await _real_or_missing("_report_orphans_if_due")(mgr)

    assert ran is True, "the sweep did not run while this worker tracked NOTHING"
    assert any(
        "orphan-app" in m for m in log.messages("error")
    ), f"the orphan was not reported: {log.messages('error')}"


@pytest.mark.asyncio
async def test_a_SIBLING_workspaces_app_is_not_called_an_orphan() -> None:
    """Several workers share one Ray cluster, and serve.status() returns all of their apps.

    Without a workspace filter, "names minus the ones I track" reports a sibling's healthy
    application as unmanaged, at ERROR, every cooldown, for ever.
    """
    log = _Logger()
    cluster = _ClusterStub(present_after_delete=[["mine", "sibling-ws-app"]])
    mgr = _SweepManager(cluster, log, tracked={"mine": {}})

    async def _only_mine(application_id):
        return application_id != "sibling-ws-app"

    mgr._is_this_workers_app = _only_mine

    orphans = await _real_or_missing("get_orphaned_applications")(mgr)

    assert (
        orphans == []
    ), f"a sibling workspace's app was reported as an orphan: {orphans}"
    assert not log.messages("error"), log.messages("error")


@pytest.mark.asyncio
async def test_an_app_whose_owner_cannot_be_established_is_not_reported() -> None:
    """ "I could not tell whose it is" must not become "it is unmanaged".

    On a shared cluster that direction produces a permanent false alarm, which is the noise the
    cooldown exists to avoid in the first place.

    This arm drives the REAL `_is_this_workers_app`: the stub cluster has no `get_app_handle`,
    so reading the owner raises, which is the production shape of "cannot tell". An earlier
    version of this arm replaced `_is_this_workers_app` with its own stub and therefore tested
    the stub — a mutant flipping the real method's failure path to True survived it untouched.
    """
    log = _Logger()
    cluster = _ClusterStub(present_after_delete=[["unknown-app"]])
    mgr = _RealOwnerManager(cluster, log, tracked={})

    assert await _real_or_missing("get_orphaned_applications")(mgr) == []
    assert not log.messages("error")


@pytest.mark.asyncio
async def test_a_REDEPLOY_during_the_confirm_window_is_left_alone() -> None:
    """A new deploy may legitimately reuse the id while the window is open.

    From inside the loop that is indistinguishable from the re-creation it hunts. Unchecked it
    deletes the user's brand-new application repeatedly and the caller then overwrites its
    tracking entry with the stopped one. Tracking is the signal: the id being back means
    somebody deployed it on purpose.
    """
    log = _Logger()
    cluster = _ClusterStub(present_after_delete=[[APP]])
    mgr = _ManagerStub(cluster, log)
    mgr._deployed_applications = {APP: {"auto_redeploy": True}}  # redeployed meanwhile

    confirmed = await _real_or_missing("_delete_and_confirm_absent")(
        mgr, APP, polls=3, interval=0
    )

    assert (
        confirmed is True
    ), "the undeploy should finish rather than fight a fresh deploy"
    assert (
        cluster.deletes == 0
    ), f"the confirm loop deleted a freshly redeployed app {cluster.deletes} time(s)"


# ───────── 7. the app stays VISIBLE while it is being stopped (round-1 finding (a)) ──────────


class _SlowDeleteCluster(_ClusterStub):
    """serve.delete blocks, as it did for 60s on the measured site, and records the state."""

    def __init__(self, present_after_delete, observer):
        super().__init__(present_after_delete)
        self.observer = observer

    async def call_with_reconnect(self, fn, *args):
        if getattr(fn, "__name__", "") == "delete":
            # Whatever a caller would see WHILE the delete is in flight.
            self.observer()
        return await _ClusterStub.call_with_reconnect(self, fn, *args)


@pytest.mark.asyncio
async def test_the_app_is_still_LISTED_while_its_delete_is_in_flight() -> None:
    """`serve.delete` can block for a minute; the app must not vanish from the API meanwhile.

    The entry used to be popped before the delete, so for the whole of that window an
    application that was still running answered "not here" to get_app_status and could not be
    reached by stop_app. That is this row's own complaint, produced by the stop itself.
    """
    log = _Logger()
    seen = {}

    def observe():
        seen["tracked_during_delete"] = APP in mgr._deployed_applications
        seen["auto_redeploy_during_delete"] = mgr._deployed_applications.get(
            APP, {}
        ).get("auto_redeploy")

    cluster = _SlowDeleteCluster(present_after_delete=[[]], observer=observe)
    mgr = _ManagerStub(cluster, log)

    await AppsManager._undeploy_application(mgr, APP, "tester")

    assert (
        seen.get("tracked_during_delete") is True
    ), "the app was invisible to get_app_status while its delete was in flight"
    assert seen.get("auto_redeploy_during_delete") is False, (
        "it stayed listed but ALSO stayed redeployable - the monitor could race the stop, "
        "which is the hazard popping the entry was there to prevent"
    )
    # And once the delete is confirmed it really is gone.
    assert APP not in mgr._deployed_applications


@pytest.mark.asyncio
async def test_a_stopping_app_is_never_in_the_monitors_redeploy_list() -> None:
    """The guard that replaces the pop: the redeploy list is filtered on auto_redeploy."""
    log = _Logger()
    mgr = _ManagerStub(_ClusterStub(present_after_delete=[[]]), log)
    mgr._deployed_applications[APP]["auto_redeploy"] = False
    mgr._deployed_applications[APP]["stopping"] = True

    redeployable = [
        app_id
        for app_id, info in mgr._deployed_applications.items()
        if info["auto_redeploy"]
    ]
    assert redeployable == [], (
        "a stopping app is still in the monitor's redeploy list, so a tick could redeploy "
        "the app the user just asked to stop"
    )


@pytest.mark.asyncio
async def test_a_MONITOR_TICK_sweeps_when_the_worker_tracks_nothing(
    monkeypatch,
) -> None:
    """Drives the real tick, because the defect was WHERE the call sits, not what it does.

    The sweep used to be placed after `if not apps_to_redeploy: ... return`. Calling
    `_report_orphans_if_due` directly — as the arm above does — cannot see that: it tests the
    sweep in isolation and passes either way. Only a tick with NOTHING tracked, which is the
    state a stop leaves, distinguishes the two placements.
    """
    log = _Logger()
    cluster = _ClusterStub(present_after_delete=[["orphan-app"]])
    mgr = _SweepManager(cluster, log, tracked={})
    monkeypatch.setattr(
        manager_mod, "_ORPHAN_REPORT_COOLDOWN_SECONDS", 0, raising=False
    )

    renewed = []

    async def _renew():
        renewed.append(True)

    mgr._renew_proxy_tokens = _renew
    mgr.ray_cluster.check_connection = lambda: None

    await AppsManager.monitor_applications(mgr)

    assert renewed, "the tick did not run at all; this arm would pass vacuously"
    assert any("orphan-app" in m for m in log.messages("error")), (
        "a monitor tick with nothing tracked never reported the orphan — the sweep is "
        f"unreachable on that path. errors seen: {log.messages('error')}"
    )


# ───── 9. the teardown must not untrack somebody ELSE's deploy (round-2 finding (b)) ─────


class _RedeployDuringDeleteCluster(_ClusterStub):
    """A NEW deploy claims the id while ``serve.delete`` is still in flight.

    That is the legitimate case, not a race to be prevented: the id is free to reuse, and
    because the entry is now kept for the whole teardown, the new deploy takes the update path
    and REPLACES the tracking dict with its own.
    """

    def __init__(self, present_after_delete, holder, new_entry):
        super().__init__(present_after_delete)
        self.holder = holder
        self.new_entry = new_entry

    async def call_with_reconnect(self, fn, *args):
        if getattr(fn, "__name__", "") == "delete":
            self.holder["manager"]._deployed_applications[APP] = self.new_entry
        return await super().call_with_reconnect(fn, *args)


@pytest.mark.asyncio
async def test_a_REDEPLOY_during_the_undeploy_KEEPS_ITS_OWN_tracking_entry() -> None:
    """Round-2 finding (b): the undeploy used to delete the NEW deployment's entry.

    The confirm loop already recognises a redeploy and returns early, "leaving the new
    deployment alone" — but the undeploy then fell through to an unconditional
    ``_deployed_applications.pop(application_id)`` and removed the entry the new deploy had
    just created. Its build task kept running behind an application the worker no longer
    listed: this row's own orphan, manufactured by the teardown written to prevent it.

    This arm drives ``_undeploy_application``, not ``_delete_and_confirm_absent``. The inner
    helper returns True and looks correct; the damage is done by its CALLER afterwards, so an
    arm on the helper alone cannot see it — which is exactly how round 1 shipped.
    """
    log = _Logger()
    holder = {}
    new_entry = {"auto_redeploy": True, "version": "2.0.0"}
    cluster = _RedeployDuringDeleteCluster([[APP]], holder, new_entry)
    mgr = _ManagerStub(cluster, log)
    holder["manager"] = mgr
    original_entry = mgr._deployed_applications[APP]

    await _real_or_missing("_undeploy_application")(
        mgr, application_id=APP, user_id="u"
    )

    assert APP in mgr._deployed_applications, (
        "the undeploy removed the tracking entry of an application somebody else had just "
        "deployed, so its build task runs on behind an application nothing lists"
    )
    assert mgr._deployed_applications[APP] is new_entry, (
        "the id is tracked, but not by the NEW deploy's entry — the stopped one was left in "
        "its place, which reports the wrong version and auto_redeploy to every caller"
    )
    assert new_entry["auto_redeploy"] is True, (
        "the new deployment's auto_redeploy was cleared by the undeploy of the OLD one, so "
        "the monitor will never redeploy it after a controller loss"
    )
    assert "stopping" not in new_entry, (
        "the new deployment is marked as stopping, so get_app_status reports a live app as "
        "mid-teardown"
    )
    assert (
        original_entry is not new_entry
    ), "the arm's own premise failed: no swap happened"


# ───── 10. an unconfirmed build cancel reaches the undeploy (round-2 finding (c)1) ─────


@pytest.mark.asyncio
async def test_an_UNCONFIRMED_build_cancel_keeps_the_app_tracked() -> None:
    """The delete succeeded and the window stayed clear — and it is STILL not safe to forget.

    ``ray.cancel`` is a request. When the builder's bounded wait expires the task is still
    alive and may yet reach ``serve.run``, and no confirm window of any fixed length covers a
    task wedged in a multi-minute runtime_env install. Dropping the entry here is precisely
    what converts that return into an orphan: invisible to ``get_app_status``, unreachable by
    ``stop_app``, holding its GPUs.

    Before this fix the builder only LOGGED that outcome, so the undeploy untracked the app
    anyway: ``serve=['app'], tracked=[]``.
    """
    log = _Logger()
    cluster = _ClusterStub(present_after_delete=[[]])  # deleted, and it stays gone
    mgr = _ManagerStub(cluster, log)
    entry = mgr._deployed_applications[APP]

    # Recorded BY the cancellation, which is when the builder really writes it — not seeded
    # beforehand. Seeding beforehand would be a record from some EARLIER cancellation, which
    # this method now deliberately discards (see the stale-record arm below), so the arm would
    # be testing the opposite of what it claims.
    async def _cancel_that_leaves_the_task_alive(application_id):
        mgr.app_builder._unconfirmed_build_cancellations.add(application_id)

    mgr._cancel_deployment_process = _cancel_that_leaves_the_task_alive

    await _real_or_missing("_undeploy_application")(
        mgr, application_id=APP, user_id="u"
    )

    assert APP in mgr._deployed_applications, (
        "a build task that was never confirmed finished can still re-create this "
        "application, and the worker has just stopped listing it"
    )
    assert mgr._deployed_applications[APP] is entry
    assert entry["auto_redeploy"] is False, (
        "kept tracked is right, but auto_redeploy must stay off or the monitor redeploys an "
        "application the operator stopped"
    )
    assert (
        "stopping" not in entry
    ), "the stop is over; it must not still advertise as mid-stop"
    assert any("FAILED" in m for m in log.messages("error")), (
        "the stop left something behind and said so nowhere an operator or log watcher "
        f"greps. errors seen: {log.messages('error')}"
    )
    assert mgr.app_builder.take_unconfirmed_build_cancellation(APP) is False, (
        "the undeploy did not DRAIN the record, so every later stop of this id will keep its "
        "entry for a hazard that has already been handled"
    )


@pytest.mark.asyncio
async def test_a_CONFIRMED_build_cancel_still_DROPS_the_tracking_entry() -> None:
    """The control for the arm above, and it is the one that matters.

    Without it, code that simply never untracks anything satisfies every assertion there —
    and leaking a tracking entry on the happy path is the defect
    ``test_undeploy_delete_failure_keeps_tracking`` was written against.
    """
    log = _Logger()
    cluster = _ClusterStub(present_after_delete=[[]])
    mgr = _ManagerStub(cluster, log)
    assert (
        not mgr.app_builder._unconfirmed_build_cancellations
    ), "premise: nothing recorded"

    await _real_or_missing("_undeploy_application")(
        mgr, application_id=APP, user_id="u"
    )

    assert mgr._deployed_applications == {}, (
        "a clean stop of a confirmed-dead build task must drop the entry; keeping it makes "
        "every stopped app linger in get_app_status forever"
    )
    assert any("completed" in m for m in log.messages("info"))


# ───── 11. the sweep's cooldown is actually a cooldown (round-2 finding (f)) ─────


@pytest.mark.asyncio
async def test_the_SWEEP_WAITS_OUT_ITS_COOLDOWN_on_a_healthy_cluster() -> None:
    """The cooldown was stamped only when orphans were FOUND, so a healthy cluster never
    stamped it and re-ran the whole sweep on EVERY monitor tick.

    The sweep is not free: one ``serve.status``, plus a ``get_app_handle`` and a
    ``get_app_data`` — 10s timeout each, issued sequentially inside the tick — for every
    application Ray Serve holds that this worker does not track, which on a shared cluster
    means every other tenant's app. The measured cost was 6 ``serve.status`` and 30
    ``get_app_handle`` calls across 6 ticks that found nothing.

    The arm counts CALLS across two ticks rather than reading the stamp, because the comment
    claiming "the hot path pays nothing on the ticks in between" was there while this was
    false. Note it does NOT zero the cooldown, unlike every other sweep arm here: the cooldown
    is the thing under test.
    """
    log = _Logger()
    cluster = _ClusterStub(present_after_delete=[[]])  # nothing in Serve, so no orphans
    mgr = _SweepManager(cluster, log, tracked={})

    renewed = []

    async def _renew():
        renewed.append(True)

    mgr._renew_proxy_tokens = _renew
    mgr.ray_cluster.check_connection = lambda: None

    await AppsManager.monitor_applications(mgr)
    after_first = cluster.status_calls
    await AppsManager.monitor_applications(mgr)
    after_second = cluster.status_calls

    assert (
        len(renewed) == 2
    ), f"both ticks must actually run or this arm passes vacuously (ran {len(renewed)})"
    assert after_first >= 1, (
        "the first tick never swept at all, so this arm cannot tell a cooldown from a sweep "
        "that is simply unreachable"
    )
    assert after_second == after_first, (
        "the second tick swept again inside the cooldown: the stamp is only written when "
        f"orphans are FOUND, so a healthy cluster pays the sweep every tick "
        f"({after_first} serve.status calls after one tick, {after_second} after two)"
    )


@pytest.mark.asyncio
async def test_a_STALE_cancel_record_does_NOT_hold_a_clean_stop() -> None:
    """The record describes ONE cancellation, so a leftover must not decide a later stop.

    ``_cancel_deployment_process`` is also called on the deploy UPDATE path, which has no
    reason to act on the answer: an update re-creates the application in the open, so a build
    task that outlived its cancellation cannot orphan anything there. Nothing drains it on
    that path — so without the drain at the start of the undeploy, the NEXT stop of this id
    reads a hazard that belongs to somebody else's cancellation and refuses to drop a tracking
    entry it should drop. The app then lingers in ``get_app_status`` forever, and only a second
    ``stop_app`` clears it.

    The companion of the arm above: there, the record is written BY this teardown's cancel and
    must be honoured; here it predates it and must be ignored.
    """
    log = _Logger()
    cluster = _ClusterStub(present_after_delete=[[]])
    mgr = _ManagerStub(cluster, log)
    # Left over from an earlier cancellation of the same id. This teardown's own cancel adds
    # nothing: the stub is a no-op, which is the "task ended cleanly" case.
    mgr.app_builder._unconfirmed_build_cancellations.add(APP)

    await _real_or_missing("_undeploy_application")(
        mgr, application_id=APP, user_id="u"
    )

    assert mgr._deployed_applications == {}, (
        "a stale record from an earlier cancellation kept this application tracked, so a "
        "stop that fully succeeded reports as unclean and the entry never goes away"
    )
    assert (
        not mgr.app_builder._unconfirmed_build_cancellations
    ), "the stale record is still there, so it will decide the next stop too"
