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
    b = AppBuilder.__new__(AppBuilder)          # no Ray, no Hypha, no filesystem
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

    assert b.cancel_inflight_build("app-1") is False
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
    assert errors and "may be re-created" in " ".join(errors), (
        f"a failed cancel must say the app can come back; got {errors!r}"
    )


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
