"""The permissions a run's status file is created with, pinned.

A run's `status.json` carries the run's OWNER, and the owner field is what every
permission check on that run is matched against — so the file's mode is part of
the access story and not a detail of how it is written.

WHAT IS ON PRODUCTION, measured on the de.NBI worker 2026-11-26, and it reads
like a defect until you look at the dates:

    2026-08    4 files   0o777
    2026-09   49 files   0o777
    2026-11  207 files   0o777,  9 files 0o644,  22 files 0o600

The newest files are `0o600` and the oldest are `0o777`. The spread is a HISTORY
of writer implementations, not something the current code produces: `write_status`
writes through `tempfile.mkstemp` and `os.replace`, and `mkstemp` creates at
`0o600` while `os.replace` carries the source's mode onto the target. So every
file the shipped writer creates is `0o600`, which is the restrictive end.

WHY PIN IT ANYWAY. That property is incidental — it is inherited from
`mkstemp`'s documented behaviour rather than chosen anywhere — so nothing stops
a future rewrite using `Path.write_text`, which creates at `0o666 & ~umask` and
would put a run's owner field back within reach of anything else running in that
container. A test that asserts the mode makes the restriction a decision.

This file says NOTHING about the 260 legacy `0o777` files already on disk.
Remedying those is a production data change, it needs its own clearance, and a
test cannot do it.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile

import pytest

import training

#: The mode the writer must produce. 0o600 and not "something without the world
#: bit": a single documented value is checkable, and "not world-writable" is the
#: kind of loose assertion that passes on 0o644 and then on 0o664.
#:
#: WRITTEN AGAIN HERE ON PURPOSE, so that a drift of `training.STATUS_FILE_MODE`
#: is caught rather than followed. This literal deliberately does NOT read the
#: writer's constant: a test whose expectation comes from the code under test
#: cannot see that code change, and `test_THE_WRITER_NAMES_THE_MODE_ITSELF`
#: below exists precisely to compare the two. Keeping the second copy is what
#: makes a silent edit of the writer's constant a test failure.
REQUIRED_MODE = 0o600


def test_THE_WRITER_NAMES_THE_MODE_ITSELF():
    """DONE WHEN 1's "one DOCUMENTED mode", and the document has to be where the
    next change is made. A review found 0o600 appeared only in this test file
    and in its docstring, while `training.py`'s long comment at the mkstemp call
    explained atomicity and said nothing about the mode.

    THIS TEST NO LONGER GREPS THE WRITER'S SOURCE, and the reason is a measured
    hole rather than a tidy-up. It used to assert the literal text
    `os.fchmod(fd, STATUS_FILE_MODE)` appeared in `training.py`; a review
    replaced that line with `# os.fchmod(fd, STATUS_FILE_MODE)` in a throwaway
    copy and every test in this file still passed, because the text was still
    there and the writer was back to inheriting its mode. A substring grep
    cannot tell code from a comment, so the guarantee now rests on the
    behavioural test below, which is immune to comments, re-wording and line
    moves. What remains here is the constant's VALUE, which no behavioural test
    can pin: it is the thing this file's expectation is compared against.
    """
    assert training.STATUS_FILE_MODE == REQUIRED_MODE, (
        f"the writer's constant is {training.STATUS_FILE_MODE:#o} and this test "
        f"expects {REQUIRED_MODE:#o}"
    )


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


# `_status_path` is private, and using it is deliberate rather than lazy: the
# PATH is what the assertion is about, there is no public accessor for it, and
# re-deriving `session_dir(sid) / "status.json"` here would be a second copy of
# the layout that could drift from the one the writer uses.
_status_path = training._status_path


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """The real session store, rooted in a tmp dir rather than the home dir."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(training.Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def test_a_NEW_status_file_is_created_0o600(store):
    sid = "2026-01-01-000000-aaaaaaaa"
    training.write_status(sid, state="RUNNING", owner={"id": "u1"})

    p = training._status_path(sid)
    assert p.exists(), p
    assert (
        _mode(p) == REQUIRED_MODE
    ), f"{p.name} was created {oct(_mode(p))}, not {oct(REQUIRED_MODE)}"


def test_THE_WRITER_SETS_THE_MODE_ITSELF_on_a_PERMISSIVE_temp_file(store, monkeypatch):
    """DONE WHEN 1's "the writer REQUESTS the mode", asserted behaviourally.

    This replaces a substring grep for `os.fchmod(fd, STATUS_FILE_MODE)` in the
    writer's source. A review defeated that grep by COMMENTING THE LINE OUT: the
    text was still in the file, every test passed, and the writer had silently
    gone back to inheriting whatever mode it was handed.

    So hand it a permissive one. `mkstemp` is wrapped so the writer's own temp
    file arrives at 0o666 — the mode `Path.write_text` would produce under a
    0o000 umask, and the one case where "inherited" and "requested" give
    different answers. `os.replace` carries the temp file's mode onto the
    target, so the installed `status.json` is 0o600 only if the writer set it
    between the two. A commented-out, deleted, re-worded or moved fchmod all
    fail this identically, because nothing here reads the source.

    The wrapper fires only on the writer's own `.status-` prefix, so nothing
    else that calls `mkstemp` while it is installed is affected, and `seen` is
    asserted non-empty: if the writer ever stops using that prefix the test
    would otherwise pass while exercising nothing at all.
    """
    real_mkstemp = tempfile.mkstemp
    seen = []

    def permissive_mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        if kwargs.get("prefix") == ".status-":
            os.fchmod(fd, 0o666)
            seen.append(stat.S_IMODE(os.fstat(fd).st_mode))
        return fd, name

    monkeypatch.setattr(training.tempfile, "mkstemp", permissive_mkstemp)

    sid = "2026-01-01-000004-dddddddd"
    training.write_status(sid, state="RUNNING", owner={"id": "u1"})

    assert seen == [0o666], (
        "the wrapper never saw a '.status-' temp file, so this test measured "
        f"NOTHING about the writer's mode (observed: {[oct(m) for m in seen]})"
    )
    p = training._status_path(sid)
    got = _mode(p)
    assert got == REQUIRED_MODE, (
        f"the temp file was handed over at 0o666 and the installed file is "
        f"{got:#o}: the writer INHERITED the mode instead of requesting "
        f"{REQUIRED_MODE:#o}"
    )


def test_a_REWRITTEN_status_file_KEEPS_0o600(store):
    """`os.replace` puts a NEW inode in place, so the mode is re-established on
    every write rather than preserved from the first one. A writer that stopped
    going through mkstemp would therefore change the mode of files it UPDATES,
    not only of files it creates."""
    sid = "2026-01-01-000001-bbbbbbbb"
    training.write_status(sid, state="RUNNING")
    first = _mode(training._status_path(sid))

    training.write_status(sid, state="DONE")
    p = training._status_path(sid)

    assert first == REQUIRED_MODE
    assert (
        _mode(p) == REQUIRED_MODE
    ), f"after a rewrite the mode is {oct(_mode(p))}, not {oct(REQUIRED_MODE)}"
    assert json.loads(p.read_text())["state"] == "DONE"


def test_THE_WORLD_BIT_IS_WHAT_MATTERS_and_the_test_can_see_it(store):
    """The control. A test that only ever observes the writer's own output
    cannot show it would notice a permissive mode, so this sets one by hand and
    asserts the check is the thing that catches it."""
    sid = "2026-01-01-000002-cccccccc"
    training.write_status(sid, state="RUNNING")
    p = training._status_path(sid)

    os.chmod(p, 0o777)
    assert (
        _mode(p) != REQUIRED_MODE
    ), "the mode did not change, so this control is inert"
    # NO assertion that 0o777 has the world bit: that would be a test of the
    # `stat` module, and a review was right that it reads as more evidence than
    # it is. The control is the two lines below.

    # The shipped writer puts it back, because os.replace re-creates the inode.
    training.write_status(sid, state="RUNNING")
    assert _mode(training._status_path(sid)) == REQUIRED_MODE


@pytest.mark.parametrize("umask", [0o000, 0o077])
def test_the_mode_is_NOT_dependent_on_the_process_umask(store, monkeypatch, umask):
    """The mode must not depend on the umask the process happens to carry.

    AND A CORRECTION, because the first version of this docstring stated a
    falsehood: "mkstemp ignores the umask". It does not. `mkstemp` calls
    `os.open(..., 0o600)` and the umask masks that like any other creation mode
    — it merely LOOKS immune because an ordinary umask never clears the owner
    bits (`umask 0o200` would give `0o400`). What makes the mode genuinely
    umask-independent is the explicit `os.fchmod` the writer now does.

    `0o222` WOULD BE the discriminating case — it clears the owner WRITE bit, so
    without the fchmod the file lands at 0o400 — and it is NOT in the list,
    which is worth stating rather than quietly omitting. Under that umask the
    writer cannot create its own `.mutex-*` file inside the session directory
    (`PermissionError` out of `Path.touch`), so the call never reaches the
    status file and the case measures a mutex problem instead of a mode one.
    Pre-creating the session directory does not help: the mutex is created per
    write, under whatever umask is in force.

    So the umask-independence claim rests on the explicit `os.fchmod` in the
    writer plus the mutation arm that swaps the writer for `Path.write_text`,
    not on an owner-clearing umask. Making 0o222 reachable would mean changing
    the mutex too, which is a different row.
    """
    # The suffix must be 8 chars to satisfy the id validator, so the umask is
    # padded rather than printed at its natural width.
    sid = f"2026-01-01-000003-{umask:08o}"
    # THE SESSION DIRECTORY IS CREATED FIRST, BEFORE THE UMASK IS APPLIED, and
    # that is not a way of dodging the hard case. Under `umask 0o222` a fresh
    # `mkdir` lands at 0o555 and the store is unwritable, so the call raises
    # SessionsRootUnavailable and never reaches the file this test is about —
    # a directory-permission effect standing in for the file-mode one. Creating
    # the directory first narrows the umask's reach to the status FILE, which
    # is the claim under test.
    training.session_dir(sid)

    old = os.umask(umask)
    try:
        training.write_status(sid, state="RUNNING")
        got = _mode(training._status_path(sid))
        assert (
            got == REQUIRED_MODE
        ), f"under umask {umask:#o} the file is {got:#o}, not {REQUIRED_MODE:#o}"
    finally:
        os.umask(old)
