"""A failed source sync must NOT leave the replica with HOME on a read-only path.

WHAT BROKE, and why this is a regression guard and not a style preference. On a production
external-cluster worker the replica's source sync raised ``OSError: [Errno 37] No locks available``
from its ``flock`` on an NFS store. ``setup_replica_environment`` assigned ``HOME``/``TMPDIR``
*after* that sync, so the exception skipped the assignment; the caller — the ``sys.meta_path``
finder in ``bioengine/__init__.py`` — catches everything and lets the replica continue. The replica
registered, reported HEALTHY, and every training run then died far away with
``OSError: [Errno 30] Read-only file system: '/home/ray/.bioengine'``, because ``Path.home()`` fell
back to the image's own read-only home. It stayed hidden for eleven days.

The ordering is therefore the invariant: ``HOME`` and ``TMPDIR`` do not depend on the source sync,
so they are set first and a sync failure costs only the source.

WHAT THIS DOES NOT COVER, stated so the guard is not read as wider than it is:
  * that the finder keeps swallowing the exception. That is the caller's behaviour and these tests
    deliberately call ``setup_replica_environment`` directly, so they pin the ordering regardless of
    what the caller decides to do with the error.
  * whether the NFS export should serve locks. That is a cluster fact, not a code one.
"""

import errno
import os
from pathlib import Path

import pytest

from bioengine._app import replica_init


@pytest.fixture
def app_dir(tmp_path, monkeypatch):
    """A replica-shaped environment pointed at a throwaway app dir."""
    d = tmp_path / "apps" / "some-artifact"
    d.mkdir(parents=True)
    monkeypatch.setenv("BIOENGINE_APP_DIR", str(d))
    monkeypatch.setenv("BIOENGINE_ARTIFACT_VERSION", "1.2.3")
    # A HOME that is NOT the app dir, standing in for the image's own read-only
    # /home/ray. If the code under test never assigns HOME, this is what a later
    # Path.home() would return.
    stale = tmp_path / "image-home"
    stale.mkdir()
    monkeypatch.setenv("HOME", str(stale))
    return d


def test_home_is_set_even_when_the_source_sync_raises(app_dir, monkeypatch):
    """The exact production failure: ENOLCK out of the sync must not cost us HOME."""

    def _boom(*_a, **_kw):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(replica_init, "_ensure_source", _boom)

    with pytest.raises(OSError) as excinfo:
        replica_init.setup_replica_environment()

    # The error must still propagate out of this function -- swallowing it here
    # would hide a missing-source replica, which is a different bug.
    assert excinfo.value.errno == errno.ENOLCK

    home = Path(os.environ["HOME"])
    assert home == app_dir / "home", (
        f"HOME is {home}, not the per-app workspace {app_dir / 'home'}. The source sync failed and "
        "took the HOME assignment with it, so this replica would write to whatever HOME it "
        "inherited -- on a Ray worker image that is a read-only /home/ray and every write fails "
        "with Errno 30, nowhere near the real cause."
    )
    assert home.is_dir(), "HOME points at a path that was never created"
    assert Path(os.environ["TMPDIR"]) == app_dir / "tmp"


def test_home_is_still_set_on_the_happy_path(app_dir, monkeypatch):
    """A guard that only checks the failure path would pass if HOME were never set at all."""
    source = app_dir / "source"
    source.mkdir()
    monkeypatch.setattr(replica_init, "_ensure_source", lambda *a, **k: source)

    replica_init.setup_replica_environment()

    assert Path(os.environ["HOME"]) == app_dir / "home"
    assert Path(os.environ["TMPDIR"]) == app_dir / "tmp"
    assert str(source) in os.sys.path


def test_a_non_lock_oserror_from_flock_still_raises(app_dir, monkeypatch, tmp_path):
    """The ENOLCK tolerance must not become 'ignore every OSError from flock'.

    EBADF is the control: widening the no-lock path to it would silence a real defect.
    """
    import fcntl

    calls = {"n": 0}

    def _flock(fd, op):
        calls["n"] += 1
        raise OSError(errno.EBADF, "Bad file descriptor")

    monkeypatch.setattr(fcntl, "flock", _flock)
    monkeypatch.setenv("BIOENGINE_ARTIFACT_FILES_URL", "https://example.invalid/files")

    with pytest.raises(OSError) as excinfo:
        replica_init._ensure_source(app_dir, "1.2.3", __import__("logging").getLogger("t"))
    assert excinfo.value.errno == errno.EBADF
    assert calls["n"] >= 1, "flock was never called, so this asserted nothing"
