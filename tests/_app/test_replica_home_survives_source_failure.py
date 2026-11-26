"""Replica setup must survive a store that cannot serve advisory locks.

WHAT BROKE. On a production external-cluster worker the replica's source sync raised
``OSError: [Errno 37] No locks available`` from an ``fcntl.flock`` on the NFS store that holds the
app dirs. Two separate things then went wrong:

  * ``setup_replica_environment`` assigned ``HOME``/``TMPDIR`` *after* that sync, so the exception
    skipped them; and
  * the caller — the ``sys.meta_path`` finder in ``bioengine/__init__.py`` — catches everything and
    lets the replica continue.

The replica registered, reported HEALTHY, and every write to ``Path.home()`` then failed with
``OSError: [Errno 30] Read-only file system: '/home/ray/.bioengine'`` — two steps from the cause and
apparently inside the application. Measured across one site's running apps, three of eight
deployment reads carried the signature, so it was never specific to one deployment.

WHAT IS GUARDED HERE:
  1. ``HOME``/``TMPDIR`` are set BEFORE the source sync, so one fault cannot cost both.
  2. ``_ensure_source`` takes no advisory lock at all, so no store can answer ENOLCK at it.
  3. Concurrent replicas on one app dir do not both materialise ``source/``: exactly one syncs and
     the other reuses its tree. Dropping the old lock without this would trade a loud failure for a
     quiet torn tree, which is the worse bug.

WHAT IS NOT GUARDED, stated so this is not read as wider than it is:
  * that the finder keeps swallowing the exception. These tests call ``setup_replica_environment``
    directly, so they pin the ordering regardless of what the caller does with the error.
  * whether the NFS export should serve locks. That is a cluster fact, not a code one.
"""

import errno
import os
import threading
import time
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
    monkeypatch.setenv("BIOENGINE_ARTIFACT_ID", "ws/some-artifact")
    monkeypatch.setenv("BIOENGINE_ARTIFACT_FILES_URL", "https://example.invalid/files")
    # A HOME that is NOT the app dir, standing in for the image's own read-only
    # /home/ray. If the code under test never assigns HOME, this is what a later
    # Path.home() would return.
    stale = tmp_path / "image-home"
    stale.mkdir()
    monkeypatch.setenv("HOME", str(stale))
    return d


# --------------------------------------------------------------------------------------
# 1. the ordering invariant
# --------------------------------------------------------------------------------------


def test_home_is_set_even_when_the_source_sync_raises(app_dir, monkeypatch):
    """The exact production failure: an error out of the sync must not cost us HOME."""

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
    import sys as _sys

    assert str(source) in _sys.path


# --------------------------------------------------------------------------------------
# 2. no advisory lock -- the store can no longer break the setup with ENOLCK
# --------------------------------------------------------------------------------------


def test_ensure_source_takes_no_advisory_lock(app_dir, monkeypatch):
    """A store that cannot serve locks must be unable to fail this call.

    Asserted by making ``flock`` itself fatal: if ``_ensure_source`` still reaches for an advisory
    lock, this raises and the test fails. That is strictly stronger than catching ENOLCK, and it is
    why ENOLCK tolerance was not the fix -- proceeding unlocked would have left concurrent replicas
    materialising one tree.
    """
    import fcntl

    def _fatal(*_a, **_kw):
        raise AssertionError(
            "_ensure_source called fcntl.flock. The app dir lives on a shared store that may be "
            "an NFS export with no lock manager, where flock answers ENOLCK and the whole replica "
            "setup dies. Serialise with an atomically created claim instead."
        )

    monkeypatch.setattr(fcntl, "flock", _fatal)

    synced = []

    def _fake_sync(files_url, version, token, source, snapshot_path, logger):
        synced.append(version)
        source.mkdir(parents=True, exist_ok=True)
        (source / "main.py").write_text("x = 1\n")

    monkeypatch.setattr(replica_init, "_sync_source_from_hypha", _fake_sync)

    replica_init.setup_replica_environment()

    assert synced == ["1.2.3"], f"the sync ran {len(synced)} time(s), expected exactly 1"
    assert (app_dir / "source" / "main.py").is_file()
    assert Path(os.environ["HOME"]) == app_dir / "home"


def test_a_claim_that_cannot_be_created_fails_loudly_naming_the_path(app_dir, monkeypatch):
    """Silence is what made the original bug cost eleven days, so the claim must never pass quietly."""
    real_mkdir = os.mkdir

    def _mkdir(path, *a, **kw):
        if str(path).endswith(".materialising"):
            raise OSError(errno.EROFS, "Read-only file system")
        return real_mkdir(path, *a, **kw)

    monkeypatch.setattr(replica_init, "_sync_source_from_hypha", lambda *a, **k: None)
    monkeypatch.setattr(os, "mkdir", _mkdir)

    with pytest.raises(RuntimeError) as excinfo:
        replica_init._ensure_source(
            app_dir, "1.2.3", __import__("logging").getLogger("t")
        )
    msg = str(excinfo.value)
    assert ".materialising" in msg, f"the error does not name the path: {msg}"
    assert "errno=" in msg, f"the error does not name the errno: {msg}"


# --------------------------------------------------------------------------------------
# 3. concurrency -- exactly one replica materialises one app dir
# --------------------------------------------------------------------------------------

def test_two_concurrent_replicas_never_sync_the_same_app_dir_at_once(app_dir, monkeypatch):
    """The reason the old lock existed. Removing it without this is a quieter bug than the one fixed.

    THE INVARIANT IS NON-OVERLAP, NOT DEDUPLICATION. There is deliberately no fast path, so
    every start reconciles the tree against the artifact: two sequential syncs are correct and
    expected, and asserting a single call would fail against correct code, because the loser
    acquires the claim once the winner releases it and reconciles in its turn. What must never
    happen is two syncs *in flight at once* on one ``source/``, because that is what tears the
    tree, and it fails quietly.
    """
    import logging

    spans = []
    lock = threading.Lock()
    started = threading.Event()

    def _slow_sync(files_url, version, token, source, snapshot_path, logger):
        t0 = time.monotonic()
        started.set()
        time.sleep(1.0)
        source.mkdir(parents=True, exist_ok=True)
        (source / "main.py").write_text("x = 1\n")
        with lock:
            spans.append((t0, time.monotonic()))

    monkeypatch.setattr(replica_init, "_sync_source_from_hypha", _slow_sync)
    monkeypatch.setattr(replica_init, "_CLAIM_POLL_SECONDS", 0.05)

    results = {}
    errors = {}

    def _run(tag):
        try:
            results[tag] = replica_init._ensure_source(
                app_dir, "1.2.3", logging.getLogger(f"t{tag}")
            )
        except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
            errors[tag] = exc

    t1 = threading.Thread(target=_run, args=("a",))
    t1.start()
    assert started.wait(10), "the first thread never entered the sync; nothing was serialised"
    t2 = threading.Thread(target=_run, args=("b",))
    t2.start()
    t1.join(60)
    t2.join(60)

    assert not errors, f"a concurrent replica raised: {errors}"
    assert len(spans) == 2, f"expected both replicas to reconcile, got {len(spans)} sync(s)"
    spans.sort()
    (a_start, a_end), (b_start, b_end) = spans
    assert b_start >= a_end, (
        f"two syncs overlapped on one app dir: first ran {a_start:.2f}-{a_end:.2f}, second started "
        f"{b_start:.2f}. Concurrent writers on one source/ tree can tear it, and that failure is "
        "quiet -- worse than the loud one this change removed."
    )
    assert results["a"] == results["b"] == app_dir / "source"
    assert (app_dir / "source" / "main.py").is_file()
    assert not (app_dir / ".materialising").exists(), "the claim was not released"
