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

    src = inspect.getsource(training.write_status)
    assert 'with_suffix(".json.tmp")' not in src, (
        "the scratch file must not be derived from the target: every concurrent "
        "writer would share it"
    )
    assert "mkstemp" in src, "expected a per-writer scratch file"


def test_no_scratch_file_is_left_behind(run_dir):
    training.write_status(SID, status="PREPARING")
    leftovers = [p.name for p in run_dir.iterdir() if p.name != "status.json"]
    assert leftovers == [], f"scratch files left in the run directory: {leftovers}"


def test_a_failed_write_leaves_no_scratch_file(run_dir, monkeypatch):
    """The cleanup path, which otherwise only runs on a day something else
    breaks."""

    def boom(*a, **k):
        raise OSError("disk went away")

    monkeypatch.setattr(training.os, "replace", boom)
    with pytest.raises(OSError):
        training.write_status(SID, status="PREPARING")
    leftovers = [p.name for p in run_dir.iterdir()]
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
    # Every writer must have landed at least once. Individual values may be
    # stale -- last-writer-wins on a shared field is expected and is a different
    # question from the record being destroyed.
    missing = [i for i in range(writers) if f"writer_{i}" not in rec]
    assert missing == [], f"writers that lost every write: {missing}"

    leftovers = [p.name for p in run_dir.iterdir() if p.name != "status.json"]
    assert leftovers == [], f"scratch files left behind: {leftovers}"
