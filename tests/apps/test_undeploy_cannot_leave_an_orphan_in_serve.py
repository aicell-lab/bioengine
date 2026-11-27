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

import pytest

import bioengine.apps.builder as builder_mod
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
    """Enough of the ray module for submit(): remote(), get(), cancel()."""

    class exceptions:  # noqa: N801 - mirrors the real module's shape
        class RayTaskError(Exception):
            def __init__(self, cause=None):
                super().__init__("task failed")
                self.cause = cause

    def __init__(self, ref: _Ref) -> None:
        self.ref = ref
        self.cancelled: list[tuple[object, bool]] = []
        self.get_started = asyncio.Event()

    def remote(self, **kwargs):
        def decorator(fn):
            class _Remote:
                def remote(_self, *a, **k):
                    return self.ref

            return _Remote()

        return decorator

    def get(self, ref):
        # Mimics the real blocking wait. The arm cancels while this is parked.
        self.get_started.set()
        import time

        time.sleep(30)  # never reached: the test cancels first
        return None

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


class _BuilderStub:
    def __init__(self, logger):
        self.logger = logger

    async def _store_replica_secrets(self, application_id, secrets):
        return None


@pytest.mark.asyncio
async def test_cancelling_a_deploy_cancels_the_in_flight_build_task(
    monkeypatch,
) -> None:
    """The defect itself: a cancelled deploy left the Ray task alive to re-create the app.

    Without the fix the submit coroutine is simply cancelled while parked in ``asyncio.to_thread``,
    the Ray task is never told, and it goes on to call ``serve.run``.
    """
    log = _Logger()
    ref = _Ref()
    fake = _FakeRay(ref)
    monkeypatch.setattr(builder_mod, "ray", fake)

    task = asyncio.create_task(AppBuilder.submit(_BuilderStub(log), _BuiltApp(), APP))
    await asyncio.wait_for(fake.get_started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

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


# ────────────────────── 2. a SUCCESSFUL delete is verified, not assumed ──────────────────────


class _ServeStatus:
    def __init__(self, names):
        self.applications = {n: object() for n in names}


class _ClusterStub:
    """Replays a Serve that still holds the app after a successful delete."""

    def __init__(self, present_after_delete):
        # one entry per _serve_application_names() call, consumed in order
        self.present_after_delete = list(present_after_delete)
        self.deletes = 0
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
            names = (
                self.present_after_delete.pop(0) if self.present_after_delete else []
            )
            return _ServeStatus(names)
        raise AssertionError(f"unexpected call {name}")


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

    def __init__(self, cluster, logger):
        self.ray_cluster = cluster
        self.logger = logger
        self._deployed_applications = {APP: {"auto_redeploy": True}}
        self._redeploy_backoff = {}
        self._missing_from_status = {}
        self._deleted_pending_redeploy = set()
        self._identity_warned = set()

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
        "present in Ray Serve again" in m for m in log.messages("warning")
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
    assert not any("present in Ray Serve again" in m for m in log.messages("warning"))


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


def test_the_monitor_actually_runs_the_orphan_sweep() -> None:
    """A method nobody calls reports nothing.

    The clause this pins is "a watcher reports any Serve app that is not in its app list" — so the
    comparison existing is not enough, something has to run it. Asserted structurally, against the
    monitor's source, because driving a full monitor tick needs a live Ray cluster.

    Anchored on the call and on the reuse of the status the tick already fetched: making the sweep
    do its own ``serve.status`` would double the RPCs on the hot loop, and a reviewer should see
    that choice break if someone changes it.
    """
    import inspect

    source = inspect.getsource(AppsManager.monitor_applications)
    assert "get_orphaned_applications(" in source, (
        "the monitor never calls the orphan sweep, so an orphan is still only visible to "
        "somebody who thinks to ask"
    )
    assert "serve_application_names=" in source, (
        "the sweep is called without the status the tick already fetched, so it makes a second "
        "serve.status RPC on every monitoring interval"
    )
    assert (
        "_ORPHAN_REPORT_COOLDOWN_SECONDS" in source
    ), "an orphan persists, so an unthrottled report repeats every tick and becomes noise"


def test_a_prefetched_status_is_used_instead_of_a_second_rpc() -> None:
    """The behavioural half of the arm above: given names, it must not call Serve again."""
    import asyncio

    log = _Logger()

    class _Exploding(_ClusterStub):
        async def call_with_reconnect(self, fn, *args):
            raise AssertionError("the sweep made its own serve.status call")

    mgr = _ManagerStub(_Exploding([]), log)
    mgr._deployed_applications = {"tracked": {}}

    orphans = asyncio.run(
        _real_or_missing("get_orphaned_applications")(
            mgr, serve_application_names={"tracked", "ghost"}
        )
    )
    assert orphans == ["ghost"], orphans
