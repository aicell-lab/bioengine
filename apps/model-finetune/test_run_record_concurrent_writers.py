"""Several processes write one run's record during every run. It must survive that.

The entry replica, both runtime replicas and the training subprocess all call
`write_status` for the same session, from different processes, for the whole
duration of a run. Before this change they shared one scratch file named after
the target (`status.json.tmp`), which cost two things at once:

* `write_text` truncates and writes, which is not atomic, so one writer could
  `os.replace` a document another was halfway through writing -- and a TORN
  record became the real one;
* the first `os.replace` consumed the shared file, so the second raised
  `FileNotFoundError`. Measured at 50 of 240 writes across six processes.

And then `read_status` answers `{}` for a record that will not parse, so the
next writer merged into an empty dict and wrote THAT back as the whole record.
Status, owner, created_at and the terminal-status provenance all went, and the
result was valid JSON, so nothing downstream could notice.

**The decisive test here spawns real processes.** A single-process test cannot
see any of it: `write_status` is synchronous, so within one thread each call
completes before the next begins and the shared scratch file never collides.
That is exactly why this survived having a test suite.
"""

import json
import multiprocessing as mp
import os
import re
import time
from pathlib import Path

import pytest

import training

SID = "2026-11-25-120000-abcdef12"


@pytest.fixture()
def run_dir(tmp_path, monkeypatch):
    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(training, "sessions_root", lambda: root)
    monkeypatch.setattr(training, "session_dir", lambda sid: root / sid)
    monkeypatch.setattr(training, "SESSION_ID_PATTERN", re.compile(r"\A[\w.-]+\Z"))
    d = root / SID
    d.mkdir()
    return d


# ── an unreadable record is not an empty one ─────────────────────────────────


def test_a_record_that_will_not_parse_is_REFUSED_not_overwritten(run_dir):
    """The destructive case, stated as the thing that must not happen.

    `read_status` answering {} is right for a reader and fatal for a writer: a
    merge into {} writes the empty dict back as the entire record.
    """
    (run_dir / "status.json").write_text('{"status": "TRAINING", "owner": "gh|1"')
    with pytest.raises(training.RunRecordUnreadable):
        training.write_status(SID, n_epochs_completed=3)


def test_the_unparseable_record_is_left_EXACTLY_as_it_was(run_dir):
    """Refusing is only useful if it also does not touch the file. A reader may
    still be able to recover something from it by hand."""
    broken = '{"status": "TRAINING", "owner": "gh|1"'
    (run_dir / "status.json").write_text(broken)
    with pytest.raises(training.RunRecordUnreadable):
        training.write_status(SID, n_epochs_completed=3)
    assert (run_dir / "status.json").read_text() == broken


def test_a_record_that_is_not_an_object_is_refused_too(run_dir):
    """`json.loads` succeeds on a bare list or string, and `.get` would then
    raise something far less legible deeper in."""
    (run_dir / "status.json").write_text("[1, 2, 3]")
    with pytest.raises(training.RunRecordUnreadable):
        training.write_status(SID, status="TRAINING")


def test_a_run_with_NO_record_yet_still_starts_normally(run_dir):
    """THE CONTROL. Both refusals above would pass if write_status simply
    refused everything. A first write has no file and must just work."""
    out = training.write_status(SID, status="PREPARING")
    assert out["status"] == "PREPARING"
    assert json.loads((run_dir / "status.json").read_text())["status"] == "PREPARING"


def test_a_normal_merge_still_preserves_what_was_there(run_dir):
    (run_dir / "status.json").write_text(
        json.dumps({"status": "TRAINING", "owner": "gh|1"})
    )
    training.write_status(SID, n_epochs_completed=2)
    rec = json.loads((run_dir / "status.json").read_text())
    assert rec["status"] == "TRAINING"
    assert rec["owner"] == "gh|1"
    assert rec["n_epochs_completed"] == 2


# ── the scratch file must belong to one writer ───────────────────────────────


def test_the_scratch_file_is_not_named_after_the_target(run_dir):
    """The shared name was the whole mechanism. Asserted structurally so the
    old pattern cannot come back without this failing, even if a concurrency
    test happens to get lucky on a given machine."""
    import inspect

    # BOTH functions: the merge moved into _write_status_locked when the lock
    # was added, so inspecting write_status alone would look in the wrong place
    # and pass for the wrong reason.
    src = inspect.getsource(training.write_status) + inspect.getsource(
        training._write_status_locked
    )
    assert 'with_suffix(".json.tmp")' not in src, (
        "the scratch file must not be derived from the target: every concurrent "
        "writer would share it"
    )
    assert "mkstemp" in src, "expected a per-writer scratch file"


def test_no_scratch_file_is_left_behind(run_dir):
    training.write_status(SID, status="PREPARING")
    leftovers = [
        p.name
        for p in run_dir.iterdir()
        if p.name not in ("status.json", training.RECORD_LOCK_NAME)
    ]
    assert leftovers == [], f"scratch files left in the run directory: {leftovers}"


def test_a_failed_write_leaves_no_scratch_file(run_dir, monkeypatch):
    """The cleanup path, which otherwise only runs on a day something else
    breaks."""

    def boom(*a, **k):
        raise OSError("disk went away")

    monkeypatch.setattr(training.os, "replace", boom)
    with pytest.raises(OSError):
        training.write_status(SID, status="PREPARING")
    leftovers = [
        p.name for p in run_dir.iterdir() if p.name != training.RECORD_LOCK_NAME
    ]
    assert leftovers == [], f"scratch files left after a failed write: {leftovers}"


# ── the one that needs real processes ────────────────────────────────────────


def _writer(idx, root_str, n, errors):
    import training as t

    root = Path(root_str)
    t.sessions_root = lambda: root
    t.session_dir = lambda sid: root / sid
    for i in range(n):
        try:
            t.write_status(SID, **{f"writer_{idx}": i})
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")


@pytest.mark.slow
def test_CONCURRENT_PROCESSES_do_not_lose_the_record(run_dir):
    """CLAUSE 4, and the only test in this file that could have caught the bug.

    A single-process test cannot see it: `write_status` is synchronous, so
    within one thread each call finishes before the next starts and the scratch
    file never collides. Real processes are required.

    Asserts the two things that were measured to fail: no writer raises, and the
    field that was there before all of them is still there after.
    """
    (run_dir / "status.json").write_text(json.dumps({"status": "TRAINING"}))

    writers, per_writer = 6, 40
    mgr = mp.Manager()
    errors = mgr.list()
    procs = [
        mp.Process(target=_writer, args=(i, str(run_dir.parent), per_writer, errors))
        for i in range(writers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)

    assert list(errors) == [], (
        f"{len(errors)} of {writers * per_writer} writes raised; first few: "
        f"{list(errors)[:3]}"
    )

    raw = (run_dir / "status.json").read_text()
    rec = json.loads(raw)  # must still parse
    assert rec.get("status") == "TRAINING", (
        "the status that existed before any of these writers is GONE -- a "
        f"concurrent write destroyed the record. Final keys: {sorted(rec)}"
    )
    # EVERY writer's LAST write must survive. The previous version of this test
    # asserted only that each writer appeared at all, under a comment calling
    # the rest "last-writer-wins on a shared field is expected". Each writer_i
    # is a DISTINCT field, so a stale value is not last-writer-wins -- it is one
    # writer erasing another's data, which is the defect this file is named for.
    # That comment excused the exact failure the test existed to catch.
    stale = {
        f"writer_{i}": rec.get(f"writer_{i}")
        for i in range(writers)
        if rec.get(f"writer_{i}") != per_writer - 1
    }
    assert stale == {}, (
        f"writers whose last write was lost: {stale} (each should be "
        f"{per_writer - 1}). A distinct field per writer means this is data "
        f"loss, not last-writer-wins."
    )

    leftovers = [
        p.name
        for p in run_dir.iterdir()
        if p.name not in ("status.json", training.RECORD_LOCK_NAME)
    ]
    assert leftovers == [], f"scratch files left behind: {leftovers}"


def _stopper(root_str, errors):
    """The user-stop path: one write of a terminal status, mid-run."""
    import time as _t

    import training as t

    root = Path(root_str)
    t.sessions_root = lambda: root
    t.session_dir = lambda sid: root / sid
    # Early, and the writers below are given enough iterations to still be
    # running when this lands. A stop that arrives after every writer has
    # finished cannot be erased by one, so a late stopper makes this test pass
    # on broken code -- measured: at 0.25s with 40 iterations per writer the
    # writers were already done and 10 trials all passed on the unlocked code.
    _t.sleep(0.05)
    try:
        t.write_status(SID, status="STOPPED", terminated_by="user_stop")
    except Exception as exc:
        errors.append(f"stopper: {type(exc).__name__}: {exc}")


@pytest.mark.slow
def test_a_USER_STOP_survives_concurrent_progress_writes(run_dir):
    """The outcome this row is named for, in its real shape.

    A user clicks stop while the training child is writing progress. The stop is
    one write; the child's are many. If the merge is an unlocked
    read-modify-write, a child write that read before the stop landed puts the
    record back to TRAINING -- and the terminal-status precedence logic cannot
    refuse an incumbent it could not see.

    Measured against the scratch-file-only fix: the stop was erased
    in 9 of 20 trials.
    """
    # REPEATED, because one trial is not a discriminator. Measured
    # the erasure in 9 of 20 trials on the scratch-file-only fix, so a single
    # run passes on broken code better than half the time -- and a guard that
    # misses the defect half the time is not a guard. Ten trials puts that
    # miss probability around one in a thousand. With the lock in place every
    # trial must pass, so this is not flaky in the other direction.
    TRIALS = 10
    for trial in range(TRIALS):
        (run_dir / "status.json").write_text(json.dumps({"status": "TRAINING"}))
        mgr = mp.Manager()
        errors = mgr.list()
        procs = [
            mp.Process(target=_writer, args=(i, str(run_dir.parent), 300, errors))
            for i in range(5)
        ]
        procs.append(mp.Process(target=_stopper, args=(str(run_dir.parent), errors)))
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)

        assert list(errors) == [], f"trial {trial}: writes raised: {list(errors)[:3]}"
        rec = json.loads((run_dir / "status.json").read_text())
        assert rec.get("status") == "STOPPED", (
            f"trial {trial} of {TRIALS}: the user's STOPPED was erased by a "
            f"concurrent progress write; the record now says "
            f"{rec.get('status')!r}, terminated_by={rec.get('terminated_by')!r}"
        )
        assert (
            rec.get("terminated_by") == "user_stop"
        ), f"trial {trial}: terminated_by={rec.get('terminated_by')!r}"


def test_the_mutex_is_RELEASED_and_leaves_no_residue(run_dir):
    """A link-based mutex is unlinked on release, unlike an flock sidecar which
    persists. So after a normal write the run directory holds the record and
    nothing else.

    This assertion was originally written the other way round -- expecting a
    persistent sidecar -- because the first implementation used flock. flock
    turned out to raise ENOLCK on the share these replicas use, so the design
    changed and this expectation was simply stale. Recorded because "the test
    says the file should be there" is a bad reason to put a file back.
    """
    training.write_status(SID, status="TRAINING", owner="gh|1")
    residue = [f.name for f in run_dir.iterdir() if f.name != "status.json"]
    assert residue == [], f"the mutex or its unique file was left behind: {residue}"


def test_the_mutex_does_not_disturb_the_listing(run_dir):
    """The mutex lives INSIDE a run directory while held, and the listing keys
    on directories containing status.json, so it cannot be mistaken for a run.
    Asserted rather than reasoned about."""
    training.write_status(SID, status="TRAINING", owner="gh|1")
    (run_dir / training.RECORD_LOCK_NAME).write_text("held")  # simulate one held
    # NOTE: on this branch `list_sessions` takes no context -- the per-caller
    # signature lands with the listing row, not here.
    seen = training.list_sessions()
    assert sorted(seen) == [SID], f"the listing returned {sorted(seen)}"


def _one_field(idx, root_str, stale_s, errors, overlaps, rename_delay=0.0):
    """Write one distinct field, holding the critical section briefly.

    *rename_delay* inserts a pause between the stat that judges a mutex stale
    and the rename that reaps it -- the window in which the file at that path
    can be replaced by a LIVE mutex. Nothing exotic produces that delay in
    production: an NFS round trip or being pre-empted between two syscalls is
    enough. It is injected here because a window that small is otherwise hit
    too rarely for a test to be a reliable RED.
    """
    import os as _os
    import time as _t

    import training as t

    root = Path(root_str)
    t.sessions_root = lambda: root
    t.session_dir = lambda sid: root / sid
    t.RECORD_LOCK_STALE_S = stale_s
    if rename_delay:
        # Only the training module's view of `os` is shimmed, so the delay
        # cannot reach tempfile, multiprocessing or the test harness itself.
        class _DelayedRenameOs:
            def __getattr__(self, name):
                return getattr(_os, name)

            def rename(self, src, dst):
                _t.sleep(rename_delay)
                return _os.rename(src, dst)

        t.os = _DelayedRenameOs()
    real = t._write_status_locked

    def slow(p, session_id, fields):
        # Count how many writers are inside the critical section at once. If the
        # mutex is doing its job this never exceeds 1.
        with overlaps.get_lock():
            overlaps.value += 1
            here = overlaps.value
        if here > 1:
            errors.append(f"OVERLAP: {here} writers inside the critical section")
        _t.sleep(0.05)
        out = real(p, session_id, fields)
        with overlaps.get_lock():
            overlaps.value -= 1
        return out

    t._write_status_locked = slow
    try:
        t.write_status(SID, **{f"f{idx}": idx})
    except Exception as exc:
        errors.append(f"f{idx}: {type(exc).__name__}: {exc}")


@pytest.mark.slow
def test_a_DEAD_HOLDERS_mutex_is_reaped_WITHOUT_breaking_exclusion(run_dir):
    """The case the reap exists for, and the one it used to break.

    A holder dies leaving a FRESH mutex. Every queued writer then waits out the
    staleness window. If the mutex inherits the waiter's QUEUE time rather than
    its ACQUIRE time, the winner links an already-stale lock and the next waiter
    reaps it while it is held -- so the reap destroys the exclusion it exists to
    restore.

    Measured on that version: up to six writers inside the
    critical section at once, and a write lost in four trials of five.

    The staleness window is scaled down so the test runs in seconds; the logic
    is identical at 30s.
    """
    (run_dir / "status.json").write_text(json.dumps({"status": "TRAINING"}))
    # A dead holder's mutex: present, fresh, and nobody is coming back for it.
    (run_dir / training.RECORD_LOCK_NAME).write_text("dead holder")

    N = 6
    for trial in range(3):
        mgr = mp.Manager()
        errors = mgr.list()
        overlaps = mp.Value("i", 0)
        procs = [
            mp.Process(
                target=_one_field, args=(i, str(run_dir.parent), 1.0, errors, overlaps)
            )
            for i in range(N)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)

        overlapping = [e for e in errors if e.startswith("OVERLAP")]
        assert overlapping == [], (
            f"trial {trial}: the mutex stopped excluding while reaping a dead "
            f"holder's lock: {overlapping[:3]}"
        )
        rec = json.loads((run_dir / "status.json").read_text())
        lost = [f"f{i}" for i in range(N) if f"f{i}" not in rec]
        assert lost == [], f"trial {trial}: fields lost during the reap: {lost}"
        assert rec.get("status") == "TRAINING", rec.get("status")
        # reset for the next trial
        (run_dir / "status.json").write_text(json.dumps({"status": "TRAINING"}))
        (run_dir / training.RECORD_LOCK_NAME).write_text("dead holder")


@pytest.mark.slow
def test_the_REAP_does_not_steal_a_LIVE_lock_when_the_rename_is_delayed(run_dir):
    """The reap was check-then-act: by rename rather than unlink, same defect.

    Reaping by rename makes exactly one of several writers win the reap. It does
    NOT make the winner's target the file it judged: rename acts on the path.
    Writer A judges L1 stale, reaps it, releases, links a fresh L2; writer B,
    still holding its verdict on L1, renames L2 away and enters A's live
    critical section.

    Measured against the removing reap, with the rename delayed by 20ms:
    25 overlaps and a write lost in 3 trials of 5. The delay is what makes the
    window reliably reachable -- see `_one_field`.

    This is the same scenario as the dead-holder test; the delay is the only
    difference, which is what makes it a test of the inode check specifically.
    """
    (run_dir / "status.json").write_text(json.dumps({"status": "TRAINING"}))
    (run_dir / training.RECORD_LOCK_NAME).write_text("dead holder")

    N = 6
    for trial in range(3):
        mgr = mp.Manager()
        errors = mgr.list()
        overlaps = mp.Value("i", 0)
        procs = [
            mp.Process(
                target=_one_field,
                args=(i, str(run_dir.parent), 1.0, errors, overlaps, 0.02),
            )
            for i in range(N)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)

        overlapping = [e for e in errors if e.startswith("OVERLAP")]
        assert overlapping == [], (
            f"trial {trial}: a delayed reap renamed away a LIVE mutex and let a "
            f"second writer in: {overlapping[:3]}"
        )
        rec = json.loads((run_dir / "status.json").read_text())
        lost = [f"f{i}" for i in range(N) if f"f{i}" not in rec]
        assert lost == [], f"trial {trial}: fields lost to a stolen lock: {lost}"
        assert rec.get("status") == "TRAINING", rec.get("status")
        (run_dir / "status.json").write_text(json.dumps({"status": "TRAINING"}))
        (run_dir / training.RECORD_LOCK_NAME).write_text("dead holder")


def test_the_RELEASE_does_not_delete_a_SUCCESSORS_mutex(run_dir, monkeypatch):
    """The release side of the same family: unlinking a path, not an inode.

    A holder that overruns `RECORD_LOCK_STALE_S` -- an NFS stall is enough --
    is reaped, and a successor takes the lock. That overlap is the accepted
    price of having a timeout at all. What is NOT acceptable is what the
    overrunning holder does on its way out: if the release unlinks the mutex
    PATH unconditionally it deletes the SUCCESSOR's mutex, and a third writer
    walks in while the successor is still inside. That part is avoidable, so it
    is a bug rather than a trade-off.

    No concurrency is needed to pin it: the reap-and-relink is performed by
    hand inside the critical section, which is exactly the state a reaped
    holder returns to.
    """
    mutex = run_dir / training.RECORD_LOCK_NAME
    monkeypatch.setattr(training, "RECORD_LOCK_STALE_S", 0.05)

    with training._record_mutex(run_dir, SID):
        assert mutex.exists(), "the holder should have the mutex"
        held_ino = mutex.stat().st_ino
        # The holder stalls past the window and is reaped; a successor links its
        # own mutex at the same path.
        successor = run_dir / ".mutex-successor"
        successor.write_text("successor")
        mutex.unlink()
        os.link(successor, mutex)
        successor_ino = mutex.stat().st_ino
        assert successor_ino != held_ino, "the fixture did not change the inode"

    assert mutex.exists(), (
        "the overrunning holder's release deleted the SUCCESSOR's mutex, which "
        "lets a third writer into the successor's critical section"
    )
    assert (
        mutex.stat().st_ino == successor_ino
    ), "the mutex at the path is no longer the successor's file"


def test_an_ABANDONED_lock_is_reaped_when_the_FILE_CLOCK_RUNS_AHEAD(
    run_dir, monkeypatch
):
    """The store's clock is not this process's clock, and the lock must not care.

    On an NFS share, mtime is set by the SERVER. Measured on the de.NBI export
    2026-11-25 from inside a pod: a file stamped with `os.utime(path, None)`
    came back with an mtime 78.6 seconds AHEAD of the caller's `time.time()`.

    A staleness test written as `time.time() - st_mtime` then reads every lock
    as taken in the future, so no abandoned lock is ever reaped and every
    subsequent writer waits out the whole timeout and raises. That is not
    theoretical: on the share, before this was fixed, all 18 writes of a 6
    writer x 3 trial run died with RunRecordLockTimeout -- one of them
    reporting the lock as "held for -18.5s" -- and the lock file was left
    behind. The same tests on local disk were green throughout, because there
    the two clocks are one clock.

    Simulated by moving the clock the CODE reads, which is the production shape:
    file mtimes come from the real clock, `time.time()` is behind it.
    """
    SKEW = 79.0
    real_time = time.time
    monkeypatch.setattr(training.time, "time", lambda: real_time() - SKEW)
    monkeypatch.setattr(training, "RECORD_LOCK_STALE_S", 1.0)
    monkeypatch.setattr(training, "RECORD_LOCK_WAIT_S", 8.0)

    (run_dir / "status.json").write_text(json.dumps({"status": "TRAINING"}))
    mutex = run_dir / training.RECORD_LOCK_NAME
    mutex.write_text("dead holder")
    # Abandoned well beyond the window, by the clock that stamps files.
    old = real_time() - 60.0
    os.utime(mutex, (old, old))

    training.write_status(SID, phase="after the reap")

    rec = json.loads((run_dir / "status.json").read_text())
    assert rec.get("phase") == "after the reap", (
        "the abandoned lock was never judged stale, so the write never "
        f"happened: {rec}"
    )
    assert rec.get("status") == "TRAINING", rec


# ── a heartbeat must not kill a run; a terminal write must still raise ───────


def _hold_the_mutex(run_dir):
    """Leave a FRESH mutex nobody will release, so the next writer times out."""
    mutex = run_dir / training.RECORD_LOCK_NAME
    mutex.write_text("a holder that is not coming back")
    os.utime(mutex, None)
    return mutex


def test_a_HEARTBEAT_under_a_held_lock_does_NOT_abort_the_run(run_dir, monkeypatch):
    """The lock must not be able to destroy hours of GPU work to save a timestamp.

    `RunRecordLockTimeout` is a new exception at call sites that never raised
    before. On a heartbeat it is the wrong answer: the write carries no
    outcome, the next one is a minute away, and killing the training process
    over it is strictly worse than the stale sweep the heartbeat exists to
    dodge.
    """
    monkeypatch.setattr(training, "RECORD_LOCK_WAIT_S", 0.3)
    monkeypatch.setattr(training, "RECORD_LOCK_STALE_S", 600.0)  # never reaped
    (run_dir / "status.json").write_text(json.dumps({"status": "TRAINING"}))
    _hold_the_mutex(run_dir)

    landed = training.write_status_best_effort(SID, n_epochs_completed=3)

    assert landed is False, "the write should have been skipped, not reported as done"
    rec = json.loads((run_dir / "status.json").read_text())
    assert rec == {"status": "TRAINING"}, f"the record was touched anyway: {rec}"


def test_a_TERMINAL_write_under_the_same_held_lock_STILL_RAISES(run_dir, monkeypatch):
    """THE OTHER HALF, and the one that makes the first half safe.

    If the catch were applied to every write, a terminal status could be
    swallowed and a finished run would look like it was still going -- which is
    the failure the lock was built to surface, reintroduced one layer up.
    """
    monkeypatch.setattr(training, "RECORD_LOCK_WAIT_S", 0.3)
    monkeypatch.setattr(training, "RECORD_LOCK_STALE_S", 600.0)
    (run_dir / "status.json").write_text(json.dumps({"status": "TRAINING"}))
    _hold_the_mutex(run_dir)

    with pytest.raises(training.RunRecordLockTimeout):
        training.write_status(SID, status="COMPLETED", terminated_by="child")


def test_an_UNREADABLE_record_is_also_survived_by_a_heartbeat(run_dir):
    """The second new exception, same rule.

    A record that will not parse is refused rather than overwritten. A
    heartbeat must not die of it either -- and the file must still be left
    exactly as it was, which is what makes the refusal worth having.
    """
    corrupt = "{ this is not json"
    (run_dir / "status.json").write_text(corrupt)

    landed = training.write_status_best_effort(SID, n_epochs_completed=1)

    assert landed is False
    assert (run_dir / "status.json").read_text() == corrupt, "the record was altered"


def test_a_heartbeat_that_CAN_take_the_lock_still_writes(run_dir):
    """THE CONTROL. A best-effort helper that silently never wrote would pass
    both tests above and would stop every heartbeat in production."""
    (run_dir / "status.json").write_text(json.dumps({"status": "TRAINING"}))

    landed = training.write_status_best_effort(SID, n_epochs_completed=7)

    assert landed is True
    rec = json.loads((run_dir / "status.json").read_text())
    assert rec["n_epochs_completed"] == 7, rec
    assert rec["status"] == "TRAINING", rec


def test_the_HEARTBEAT_CALL_SITES_use_the_best_effort_helper():
    """Pins the split at the places it actually has to hold.

    The helper is only worth having if the heartbeats call it, and the terminal
    writes do not. Read from the source, so moving a call to the wrong one
    fails here.
    """
    import pathlib

    here = pathlib.Path(__file__).parent
    for name, fn in (
        ("train_worker.py", "_heartbeat"),
        ("train_worker_cellpose.py", "_heartbeat"),
    ):
        src = (here / name).read_text()
        body = src.split(f"def {fn}(", 1)[1].split("\ndef ", 1)[0]
        assert "write_status_best_effort(" in body, f"{name}:{fn} does not use it"
        assert (
            "training.write_status(" not in body
        ), f"{name}:{fn} still calls the raising write_status"

    # ...and the terminal writes must NOT have been converted with them.
    for name in ("runtime.py", "runtime_cellpose.py"):
        src = (here / name).read_text()
        assert 'status="FAILED"' in src or "status='FAILED'" in src
        assert (
            "write_status_best_effort(" not in src
        ), f"{name} uses the best-effort helper; terminal writes must raise"


def test_the_CHILDS_OWN_TERMINAL_WRITES_still_raise():
    """The gap the call-site test left, and the likeliest mistake in this area.

    `train_worker*.py` contain BOTH the heartbeat that was just converted and
    the run's final COMPLETED/FAILED write, a few dozen lines apart. The
    earlier call-site test only asserted that the heartbeat uses the helper and
    that the two runtime files do not use it at all -- so converting the
    child's own terminal write, in the same file, went unnoticed: the suite
    stayed at 21 passed.

    If that were done, a lock timeout at completion would be logged and
    skipped, the record would stay TRAINING, and the stale sweep would later
    report a finished run as STOPPED -- the exact outcome the terminal half of
    this rule exists to prevent.

    So parse the calls and check every one carrying a terminal marker.
    """
    import ast
    import pathlib

    here = pathlib.Path(__file__).parent
    checked = 0
    for name in ("train_worker.py", "train_worker_cellpose.py"):
        src = (here / name).read_text()
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call):
                continue
            fn = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if fn not in ("write_status", "write_status_best_effort"):
                continue
            seg = ast.get_source_segment(src, node) or ""
            terminal = (
                ("terminated_by" in seg) or ("COMPLETED" in seg) or ("FAILED" in seg)
            )
            if not terminal:
                continue
            checked += 1
            assert fn == "write_status", (
                f"{name}:{node.lineno} writes a terminal status through "
                f"{fn}(), which swallows a lock timeout. A terminal write must "
                "raise: otherwise the record stays TRAINING and the stale "
                "sweep later reports a finished run as STOPPED."
            )
    assert checked >= 3, (
        f"only {checked} terminal write(s) found in the worker files; the "
        "parse is not seeing them, so this test would pass vacuously"
    )
