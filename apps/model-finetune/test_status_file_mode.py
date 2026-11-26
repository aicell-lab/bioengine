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

import pytest

import training

#: The mode the writer must produce. 0o600 and not "something without the world
#: bit": a single documented value is checkable, and "not world-writable" is the
#: kind of loose assertion that passes on 0o644 and then on 0o664.
REQUIRED_MODE = 0o600


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
    assert _mode(p) != REQUIRED_MODE
    assert _mode(p) & stat.S_IWOTH, "0o777 must register as world-writable"

    # ...and the shipped writer puts it back, because os.replace re-creates it.
    training.write_status(sid, state="RUNNING")
    assert _mode(training._status_path(sid)) == REQUIRED_MODE


def test_the_mode_is_NOT_dependent_on_the_process_umask(store, monkeypatch):
    """`mkstemp` ignores the umask, which is the property being relied on. A
    writer switched to `write_text` would create at `0o666 & ~umask` and pass a
    test run under a strict umask while shipping 0o666 under a lax one."""
    old = os.umask(0o000)
    try:
        sid = "2026-01-01-000003-dddddddd"
        training.write_status(sid, state="RUNNING")
        assert _mode(training._status_path(sid)) == REQUIRED_MODE
    finally:
        os.umask(old)
