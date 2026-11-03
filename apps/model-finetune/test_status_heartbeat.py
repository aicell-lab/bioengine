"""Regression tests for the PREPARING heartbeat (the fix in 0.22.1).

``start_training`` writes PREPARING and then nothing writes status again until the
runtime's first TRAINING write. The training worker's own heartbeat only starts
*after* that write, so it cannot cover the span before it — input resolution, a cold
runtime replica coming up, the GPU lock. ``get_status`` turns PREPARING into STOPPED
once ``updated_at`` is ``STATUS_STALE_SECONDS`` (600) old, so on a cold instance a
healthy first run was reported STOPPED with no checkpoint and no message, and the GPU
time was lost. Measured on deNBI 2026-10-23: one cpsam run sat in PREPARING for over
ten minutes with ``updated_at`` never advancing past ``created_at`` and ended STOPPED,
while the same model on the same warm instance completed in 63.6 s.

``EntryApp._preparing_heartbeat`` closes that window. The subtle half is that it must
*stop* once the status leaves PREPARING: a wedged TRAINING run still has to be swept,
and a heartbeat that kept beating for the whole run would hide exactly the hang the
sweep exists to catch. That property is asserted here rather than left to the commit
message.

Plain pytest against a temporary ``HOME`` — no GPU, no Ray, no live Hypha. The defect
is status bookkeeping, so it is provable without any of them.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

import training


@pytest.fixture(autouse=True)
def _sandbox_home(monkeypatch, tmp_path):
    """``training.sessions_root()`` hangs off ``Path.home()`` — keep it in tmp_path."""
    monkeypatch.setenv("HOME", str(tmp_path))


def _backdate(session_id: str, seconds: float) -> None:
    """Age a session's ``updated_at`` without waiting for real time to pass."""
    record = training.read_status(session_id)
    record["updated_at"] = time.time() - seconds
    (training.session_dir(session_id) / "status.json").write_text(json.dumps(record))


def _heartbeat_of(instance) -> "object":
    """The real coroutine function off the Serve-wrapped class.

    ``entry.EntryApp`` is a ``ray.serve.deployment.Deployment`` once the
    ``@bioengine.app`` decorator has run, so its methods are not reachable through it;
    ``func_or_class`` is the class the decorator wrapped.
    """
    import entry

    return entry.EntryApp.func_or_class._preparing_heartbeat.__get__(instance)


def test_a_stale_preparing_session_is_swept_to_stopped() -> None:
    """The reported failure, reproduced: nothing wrote status, so the run 'stopped'."""
    training.write_status("s-stale", status="PREPARING", created_at=time.time())
    _backdate("s-stale", training.STATUS_STALE_SECONDS + 60)

    assert training.get_status("s-stale")["status"] == "STOPPED"


def test_a_field_only_write_moves_updated_at_and_prevents_the_sweep() -> None:
    """The property the heartbeat rests on: a write with no ``status`` refreshes the clock.

    Pinned separately because if this stops being true the heartbeat becomes a no-op
    that still looks like it is running.
    """
    training.write_status("s-beat", status="PREPARING", created_at=time.time())
    _backdate("s-beat", training.STATUS_STALE_SECONDS + 60)
    assert training.get_status("s-beat")["status"] == "STOPPED"

    training.write_status("s-beat")

    assert training.get_status("s-beat")["status"] == "PREPARING"


@pytest.mark.asyncio
async def test_the_heartbeat_keeps_a_preparing_session_alive() -> None:
    """It refreshes a PREPARING session that would otherwise go stale."""
    session_id = "s-preparing"
    training.write_status(session_id, status="PREPARING", created_at=time.time())
    _backdate(session_id, training.STATUS_STALE_SECONDS + 60)

    beat = asyncio.create_task(_heartbeat_of(object())(session_id, interval=0.05))
    await asyncio.sleep(0.25)
    status_while_beating = training.get_status(session_id)["status"]
    beat.cancel()

    assert status_while_beating == "PREPARING"


@pytest.mark.asyncio
async def test_the_heartbeat_stops_once_the_status_leaves_preparing() -> None:
    """The half that matters: a wedged TRAINING run must still be swept.

    A heartbeat that kept beating for the whole run would mask the hang the stale
    window exists to catch, so this asserts both that the coroutine *returns* and that
    the sweep still fires afterwards.
    """
    session_id = "s-wedged"
    training.write_status(session_id, status="PREPARING", created_at=time.time())
    training.write_status(session_id, status="TRAINING", message="training started")

    await asyncio.wait_for(
        _heartbeat_of(object())(session_id, interval=0.05), timeout=2.0
    )

    _backdate(session_id, training.STATUS_STALE_SECONDS + 60)
    assert training.get_status(session_id)["status"] == "STOPPED"


@pytest.mark.asyncio
async def test_run_training_starts_and_cancels_the_heartbeat(monkeypatch) -> None:
    """End-to-end on the real ``_run_training``, so the wiring itself is covered.

    What would regress is someone dropping the ``create_task``, and only a test that
    drives ``_run_training`` catches that. The stand-in ``self`` supplies the real
    heartbeat with a test interval — the 60 s default is not reachable from outside,
    but the coroutine under test is the shipped one.

    The matching ``preparing_hb.cancel()`` is deliberately NOT pinned. ``_run_training``
    always leaves a terminal status behind, so the heartbeat's own early return retires
    it anyway; deleting the ``cancel()`` changes no observable status and every test
    here still passes. It bounds the task's lifetime to one interval, nothing more, and
    an assertion that cannot fail would read as coverage without being any.
    """
    import entry

    # Shrink the stale window instead of backdating in the loop: backdating between
    # beats resets the clock faster than any heartbeat can refresh it, which fails the
    # test no matter how correct the fix is. Here real elapsed time makes it stale.
    monkeypatch.setattr(training, "STATUS_STALE_SECONDS", 0.15)

    session_id = "s-cold"
    training.write_status(session_id, status="PREPARING", created_at=time.time())
    seen = []

    class _ColdStart:
        _training_tasks: dict = {}

        async def _preparing_heartbeat(self, sid, interval=60.0):
            await _heartbeat_of(self)(sid, interval=0.05)

        async def _prepare_training_data(self, *_a, **_k):
            for _ in range(10):
                await asyncio.sleep(0.05)
                seen.append(training.get_status(session_id)["status"])
            raise RuntimeError("stop here — the PREPARING gap is what is under test")

    cold = _ColdStart()
    await entry.EntryApp.func_or_class._run_training(
        cold, session_id, "cpsam", [], [], [], [], {}
    )

    assert seen, "the prepare never ran"
    assert "STOPPED" not in seen, f"swept while still preparing: {seen}"
