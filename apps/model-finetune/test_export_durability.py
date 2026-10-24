"""Regression tests for making the entry deployment safe to scale to zero.

The entry was pinned at ``min_replicas: 1``. Letting it scale to zero means a
replica is torn down routinely rather than only on a crash or a redeploy, which
breaks two pieces of bookkeeping that were only ever correct because the replica
happened to be immortal:

* Export records lived in ``EntryApp._exports``, a plain dict on the instance —
  the source said so ("In-memory export registry (v1) ... Lost on replica
  restart"). After a teardown, ``get_export_status`` raised ``KeyError`` and a
  finished export looked as though it had never been requested. ``push_export``
  was worse: it needs the private ``_package_dir`` to find the built files.
* ``_run_training``'s ``CancelledError`` handler wrote ``STOPPED`` with
  ``terminated_by="user_stop"`` unconditionally. ``user_stop`` is rank 4 in
  ``training._TERMINATED_BY``, the highest, and ``write_status`` drops a differing
  terminal from an equal-or-lower-ranked writer — so the runtime child's
  authoritative ``COMPLETED`` (rank 3) could never correct it. The entry's task is
  also cancelled when the replica is torn down, while the runtime subprocess keeps
  running and owns the real outcome, so every scale-down during a healthy run
  would have recorded it as stopped-by-the-user, permanently.

Records now live under ``training.exports_root()`` on the same mounted
``~/.bioengine`` as the sessions, and the cancel handler consults
``training.stop_requested``.

Plain pytest against a temporary ``HOME`` — no GPU, no Ray, no live Hypha. Both
defects are bookkeeping, so both are provable without any of them.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

import training


@pytest.fixture(autouse=True)
def _sandbox_home(monkeypatch, tmp_path):
    """``training.exports_root()`` hangs off ``Path.home()`` — keep it in tmp_path."""
    monkeypatch.setenv("HOME", str(tmp_path))


def _backdate_export(export_id: str, seconds: float) -> None:
    """Age an export's ``updated_at`` without waiting for real time to pass."""
    record = training.read_export(export_id)
    record["updated_at"] = time.time() - seconds
    (training.exports_root() / f"{export_id}.json").write_text(json.dumps(record))


def _method_of(name: str, instance):
    """The real coroutine function off the Serve-wrapped class.

    ``entry.EntryApp`` is a ``ray.serve.deployment.Deployment`` once
    ``@bioengine.app`` has run, so its methods are not reachable through it;
    ``func_or_class`` is the class the decorator wrapped.
    """
    import entry

    return getattr(entry.EntryApp.func_or_class, name).__get__(instance)


# --- the records outlive the replica -------------------------------------


@pytest.mark.asyncio
async def test_get_export_status_answers_after_the_replica_is_gone() -> None:
    """The reported failure, inverted.

    The stand-in ``self`` deliberately carries no export state of any kind, so the
    only way this can pass is by reading the record off disk. Re-adding an
    instance-dict cache would not be caught here, but *removing* the persistence
    would be.
    """
    training.write_export(
        "e-ready", status="READY", progress=1.0,
        message="ready", download_url="https://example.invalid/pkg.zip",
        size_bytes=123, error=None,
    )

    class _FreshReplica:
        pass

    got = await _method_of("get_export_status", _FreshReplica())("e-ready")

    assert got["status"] == "READY"
    assert got["download_url"] == "https://example.invalid/pkg.zip"


@pytest.mark.asyncio
async def test_an_unknown_export_id_still_raises_keyerror() -> None:
    """The discrimination control for the test above.

    If ``get_export_status`` returned something for every id, the previous test
    would pass against a store that was not storing anything.
    """
    class _FreshReplica:
        pass

    with pytest.raises(KeyError):
        await _method_of("get_export_status", _FreshReplica())("e-never-existed")


@pytest.mark.asyncio
async def test_push_export_recovers_the_package_dir_after_a_restart(tmp_path) -> None:
    """``_package_dir`` is private and never returned to the caller, so nothing
    but the record itself can tell a fresh replica where the built files are."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "weights.pt").write_bytes(b"x" * 16)
    training.write_export(
        "e-push", status="READY",
        _package_dir=str(pkg), files=[{"name": "weights.pt", "size": 16}],
    )

    pushed_to = []

    class _FreshReplica:
        async def _put_file(self, url, path, timeout=600.0):
            pushed_to.append((url, path.name))

            class _Resp:
                def raise_for_status(self):
                    return None

            return _Resp()

    out = await _method_of("push_export", _FreshReplica())(
        "e-push", {"weights.pt": "https://example.invalid/put"}
    )

    assert out == {"pushed": ["weights.pt"], "n_files": 1}
    assert pushed_to == [("https://example.invalid/put", "weights.pt")]


def test_the_private_package_dir_is_not_leaked_to_the_caller() -> None:
    """Underscore-prefixed keys were filtered out of the old in-memory record;
    moving the store to disk must not start exposing the worker's filesystem."""
    training.write_export(
        "e-private", status="READY", _package_dir="/srv/secret",
    )

    assert "_package_dir" not in {
        k for k in training.get_export("e-private") if not k.startswith("_")
    }


# --- an export nobody is driving must not poll forever --------------------


def test_an_unattended_build_is_swept_to_failed() -> None:
    """The failure mode persistence introduces: the replica dies mid-build, so
    nothing ever advances the record off BUILDING."""
    training.write_export("e-orphan", status="BUILDING")
    _backdate_export("e-orphan", training.STATUS_STALE_SECONDS + 60)

    swept = training.get_export("e-orphan")

    assert swept["status"] == "FAILED"
    assert swept["error"]


def test_a_finished_export_is_never_swept() -> None:
    """Scope control for the sweep. A READY record is older than the stale window
    for as long as it exists; sweeping on age alone would expire every export."""
    training.write_export("e-old", status="READY")
    _backdate_export("e-old", training.STATUS_STALE_SECONDS * 10)

    assert training.get_export("e-old")["status"] == "READY"


@pytest.mark.asyncio
async def test_the_export_heartbeat_keeps_a_live_build_alive() -> None:
    """Building and uploading a package each run for minutes with no progress
    write, so without this a healthy export is swept out from under itself."""
    training.write_export("e-live", status="BUILDING")
    _backdate_export("e-live", training.STATUS_STALE_SECONDS + 60)

    beat = asyncio.create_task(_method_of("_export_heartbeat", object())("e-live", interval=0.05))
    await asyncio.sleep(0.25)
    status_while_beating = training.get_export("e-live")["status"]
    beat.cancel()

    assert status_while_beating == "BUILDING"


@pytest.mark.asyncio
async def test_the_export_heartbeat_stops_at_a_terminal_status() -> None:
    """The counterpart to the sweep: a heartbeat that outlived its build would
    keep a dead export reading BUILDING forever, which is the bug it is meant to
    prevent."""
    training.write_export("e-done", status="READY")

    await asyncio.wait_for(
        _method_of("_export_heartbeat", object())("e-done", interval=0.05), timeout=2.0
    )


# --- a teardown is not a user stop ----------------------------------------


async def _cancel_during_prepare(stop_requested_first: bool) -> str:
    """Drive the real ``_run_training`` to its ``CancelledError`` handler."""
    import entry

    session_id = "s-cancelled"
    training.write_status(session_id, status="TRAINING", message="in flight")
    if stop_requested_first:
        training.request_stop(session_id)

    class _Replica:
        _training_tasks: dict = {}

        async def _preparing_heartbeat(self, sid, interval=60.0):
            await asyncio.sleep(3600)

        async def _prepare_training_data(self, *_a, **_k):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await entry.EntryApp.func_or_class._run_training(
            _Replica(), session_id, "cpsam", [], [], [], [], {}
        )
    return session_id


@pytest.mark.asyncio
async def test_a_replica_teardown_does_not_claim_a_user_stop() -> None:
    """The defect. The runtime subprocess survives the entry and owns the outcome,
    so the entry must not stamp the record with the one rank nothing can correct."""
    session_id = await _cancel_during_prepare(stop_requested_first=False)

    assert training.read_status(session_id).get("terminated_by") != "user_stop"


@pytest.mark.asyncio
async def test_the_child_can_still_report_completed_after_a_teardown() -> None:
    """What the rank actually costs, stated as the user-visible outcome: a
    successful run must not be reported as stopped because the entry was scaled
    down while it was training."""
    session_id = await _cancel_during_prepare(stop_requested_first=False)

    training.write_status(session_id, status="COMPLETED", terminated_by="child")

    assert training.get_status(session_id)["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_a_real_user_stop_still_outranks_a_late_completed() -> None:
    """The discrimination control. ``stop_training`` writes the stop marker before
    cancelling the task, so gating on it must not weaken a genuine stop — if this
    passed with the gate removed *and* with it present, the gate would be untested.
    """
    session_id = await _cancel_during_prepare(stop_requested_first=True)
    assert training.read_status(session_id)["terminated_by"] == "user_stop"

    training.write_status(session_id, status="COMPLETED", terminated_by="child")

    assert training.get_status(session_id)["status"] == "STOPPED"
