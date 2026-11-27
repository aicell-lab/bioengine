"""Stopping an application must cancel the Ray task that is still building it.

THE INCIDENT. On a live site, with the worker pod at ``restartCount 0`` (so recovery never ran):

    03:06:44  the app is deployed
    03:07:17  stop_app deletes the Serve app and pops it from _deployed_applications
    03:07:21  the build_and_run_application Ray task — never cancelled — RE-CREATES it

The result was a Serve application holding two of three GPUs that no worker knew about, so nothing
could attribute or free it.

WHY THE EXISTING CANCEL DID NOT COVER IT. ``_cancel_deployment_process`` calls
``deployment_task.cancel()``. That cancels the *coroutine* awaiting the build. The coroutine is
blocked in ``asyncio.to_thread(ray.get, ref)``, so cancelling it abandons the await while the
worker thread keeps blocking and the **Ray task on the cluster keeps running** — through to
``serve.run``. Cancelling an awaiting coroutine is not cancelling the work it awaits.

These arms drive the manager's own teardown path with a recorded ObjectRef and assert the Ray task
is cancelled, which is the only thing that stops the re-creation.
"""

from __future__ import annotations

import contextlib
import threading
import asyncio
import logging
import sys
import types

import pytest


class _Ref:
    """Stands in for an ObjectRef; identity is all the assertions need."""

    def __init__(self, tag):
        self.tag = tag

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"<Ref {self.tag}>"


@pytest.fixture
def builder(monkeypatch):
    """A real AppBuilder with Ray's cancel recorded rather than called."""
    from bioengine.apps.builder import AppBuilder
    import bioengine.apps.builder as builder_mod

    cancelled = []

    def fake_cancel(ref, force=False, **kw):
        cancelled.append((ref, force))

    monkeypatch.setattr(builder_mod.ray, "cancel", fake_cancel)
    b = AppBuilder.__new__(AppBuilder)  # no Ray, no Hypha, no filesystem
    b.logger = logging.getLogger("test-builder")
    b._inflight_builds = {}
    return b, cancelled


def test_a_cancel_forces_the_ray_task(builder):
    """`force=True` matters: a graceful cancel only interrupts between calls, and this task spends
    its life inside serve.run."""
    b, cancelled = builder
    ref = _Ref("build-1")
    b._inflight_builds["app-1"] = ref

    assert b.cancel_inflight_build("app-1") is True
    assert cancelled == [(ref, True)], f"expected one forced cancel, got {cancelled!r}"


def test_b_a_finished_build_is_not_cancellable_later(builder):
    """A stale ref would make a later stop cancel someone else's build."""
    b, cancelled = builder
    b._inflight_builds["app-1"] = _Ref("build-1")
    b.cancel_inflight_build("app-1")
    cancelled.clear()

    assert b.cancel_inflight_build("app-1") is True, (
        "nothing registered means nothing can re-create the app, which is SAFE -- returning False "
        "there would make the ordinary case look dangerous"
    )
    assert cancelled == [], f"cancelled a build that was already gone: {cancelled!r}"


def test_c_a_failing_cancel_is_reported_not_raised(builder, monkeypatch):
    """Teardown must continue, but the operator has to hear it: the app may come back."""
    import bioengine.apps.builder as builder_mod

    b, _ = builder
    b._inflight_builds["app-1"] = _Ref("build-1")

    def boom(ref, force=False, **kw):
        raise RuntimeError("ray is unreachable")

    monkeypatch.setattr(builder_mod.ray, "cancel", boom)
    errors = []
    b.logger = types.SimpleNamespace(
        error=lambda msg, *a: errors.append(str(msg) % a if a else str(msg)),
        info=lambda *a, **k: None,
    )
    assert b.cancel_inflight_build("app-1") is False
    assert errors and "may be re-created" in " ".join(
        errors
    ), f"a failed cancel must say the app can come back; got {errors!r}"


def test_d_the_teardown_path_cancels_the_ray_task(builder):
    """The arm the review asked for: drive the manager's own cancel step and assert the RAY task
    was cancelled — not merely the coroutine.

    This is the one that is RED before the fix: on the base commit the teardown path calls only
    `deployment_task.cancel()`, so `ray.cancel` is never reached and `cancelled` stays empty.
    """
    from bioengine.apps.manager import AppsManager

    b, cancelled = builder
    ref = _Ref("build-1")
    b._inflight_builds["app-1"] = ref

    mgr = AppsManager.__new__(AppsManager)
    mgr.logger = logging.getLogger("test-manager")
    mgr.app_builder = b
    # The app is mid-build: tracked, with no asyncio task recorded yet. That is exactly the
    # window the incident fell into — a stop arriving while the build task is still running.
    mgr._deployed_applications = {"app-1": {"deployment_task": None}}

    asyncio.run(mgr._cancel_deployment_process(application_id="app-1"))

    assert cancelled == [(ref, True)], (
        "stopping an application must cancel the Ray build task, or that task will re-create the "
        f"application after teardown; cancels seen: {cancelled!r}"
    )


# ─────────── the chain, through the real submit() (review round 1, DONE WHEN 2) ───────────


def _built_app():
    """The minimum BuiltApp-shaped object submit() touches."""
    return types.SimpleNamespace(
        spec={},
        translated_kwargs={},
        proxy_args={},
        bioengine_uri="gcs://x",
        proxy_pip=[],
        user_replica_framework_pip=[],
        replica_env_vars={},
        proxy_memory_in_gb=1,
        scaling={},
        runtime_env={},
        replica_secret_env_vars={},
    )


@pytest.fixture
def submitting(monkeypatch):
    """A real AppBuilder whose Ray calls are recorded, with ray.get BLOCKING like the real one.

    This is the arm round 1 asked for. The earlier arms seeded `_inflight_builds` by hand, so a
    mutant that deleted the registration line -- making the fix a no-op in production -- left all
    of them green. Going through submit() is what makes the registration load-bearing.
    """
    import bioengine.apps.builder as bm
    from bioengine.apps.builder import AppBuilder

    ref = _Ref("build-live")
    cancelled, released = [], threading.Event()

    class _Fn:
        def remote(self, *a, **k):
            return ref

    monkeypatch.setattr(bm.ray, "remote", lambda *a, **k: (lambda fn: _Fn()))
    monkeypatch.setattr(bm.ray, "get", lambda r: released.wait(timeout=10))
    monkeypatch.setattr(
        bm.ray,
        "cancel",
        lambda r, force=False, **k: (cancelled.append((r, force)), released.set()),
    )

    b = AppBuilder.__new__(AppBuilder)
    b.logger = logging.getLogger("test-builder")
    b._inflight_builds = {}

    async def _no_secrets(*a, **k):
        return None

    monkeypatch.setattr(
        AppBuilder, "_store_replica_secrets", _no_secrets, raising=False
    )
    return b, ref, cancelled, released


def test_e_a_stop_during_a_real_submit_cancels_the_ray_task(submitting):
    """submit() registers the ref; a stop arriving while ray.get blocks must cancel it.

    RED under the mutant that drops the registration line, and RED on the base commit.
    """
    from bioengine.apps.manager import AppsManager

    b, ref, cancelled, released = submitting

    async def scenario():
        task = asyncio.create_task(b.submit(_built_app(), "app-1"))
        for _ in range(100):  # let submit reach the blocking ray.get
            await asyncio.sleep(0.01)
            if "app-1" in b._inflight_builds:
                break
        assert (
            "app-1" in b._inflight_builds
        ), "submit() never registered the build ref, so nothing could ever cancel it"
        mgr = AppsManager.__new__(AppsManager)
        mgr.logger = logging.getLogger("test-manager")
        mgr.app_builder = b
        # A RECORDED asyncio task, which is what the incident log shows
        # ("Cancelling deployment task ... was cancelled") -- not the None case.
        mgr._deployed_applications = {"app-1": {"deployment_task": task}}
        # `_cancel_deployment_process` cancels the deploy task and awaits it; the
        # CancelledError propagates out through `wait_for` into this coroutine. That is an
        # artefact of driving it from the same event loop, not of the behaviour under test —
        # the ray.cancel happens first, and that is what this arm asserts.
        with contextlib.suppress(asyncio.CancelledError):
            await mgr._cancel_deployment_process(application_id="app-1")
        released.set()
        # CancelledError derives from BaseException since 3.8, so suppress(Exception) does NOT
        # catch it -- which is why this arm failed with a CancelledError escaping `wait_for`
        # while the behaviour under test was already correct.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(scenario())
    assert cancelled == [
        (ref, True)
    ], f"a stop during an in-flight build must cancel the Ray task; cancels seen: {cancelled!r}"


def test_f_a_finished_build_leaves_nothing_to_cancel(submitting):
    """The control: a completed submit must not leave a cancellable ref behind, or a later stop
    would cancel a task that is not there -- or worse, someone else's."""
    b, ref, cancelled, released = submitting
    released.set()  # ray.get returns immediately

    asyncio.run(b.submit(_built_app(), "app-1"))
    assert (
        b._inflight_builds == {}
    ), f"a finished build stayed registered: {b._inflight_builds!r}"
    assert b.cancel_inflight_build("app-1") is True
    assert cancelled == [], f"cancelled something after a clean finish: {cancelled!r}"


# ───────── a build that could not be cancelled must stay VISIBLE (review round 1, (b)) ──────


def test_g_an_uncancellable_build_keeps_the_app_in_the_list(builder, monkeypatch):
    """`ray.cancel` is fire-and-forget, so a build can survive teardown and re-create the Serve
    app. Dropping the tracking entry in that case recreates the exact fault this row is about: an
    application holding GPUs that no worker lists. Visible-and-stoppable beats tidy-and-invisible.
    """
    import bioengine.apps.builder as bm
    from bioengine.apps.manager import AppsManager

    b, _ = builder
    b._inflight_builds["app-1"] = _Ref("build-1")
    monkeypatch.setattr(
        bm.ray,
        "cancel",
        lambda r, force=False, **k: (_ for _ in ()).throw(
            RuntimeError("ray is unreachable")
        ),
    )

    mgr = AppsManager.__new__(AppsManager)
    mgr.logger = logging.getLogger("test-manager")
    mgr.app_builder = b
    mgr._deployed_applications = {"app-1": {"deployment_task": None}}
    mgr._redeploy_backoff = {}
    mgr._missing_from_status = {}
    mgr._deleted_pending_redeploy = set()

    asyncio.run(mgr._cancel_deployment_process(application_id="app-1"))
    with contextlib.suppress(asyncio.CancelledError, Exception):
        asyncio.run(mgr._undeploy_application(application_id="app-1", user_id="u"))

    assert "app-1" in mgr._deployed_applications, (
        "a build that could not be cancelled may re-create the Serve app, so the application must "
        "stay in the worker's list rather than becoming invisible"
    )
