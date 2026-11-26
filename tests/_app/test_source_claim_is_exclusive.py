"""The source claim's two untested properties, plus the cause named in its failure message.

``test_replica_home_is_not_hook_dependent.py`` already covers the claim's lifecycle in
depth: that no byte-range lock is taken at all, that ENOLCK no longer stops a replica, that
the claim is held during the sync and released afterwards (including on the error path),
that an abandoned claim is broken and a live one fails loudly, that staleness survives a
skewed pod clock, and that no retry path spins past the deadline. This file does NOT repeat
any of that.

What it adds are the two properties that survived the merge untested, and one wording guard:

1. **An UNCREATABLE claim is not a held claim.** ``FileExistsError`` from the claim
   ``mkdir`` means "someone else holds it, wait". Any *other* ``OSError`` means the store
   will not accept a claim at all — a full disk, a read-only mount, a permission change, a
   vanished export. Those must raise, name the path and carry the errno. The load-bearing
   assertion is that the sync is **never called**: a replica that proceeds unclaimed looks
   perfectly healthy while two writers tear one shared tree.

2. **MUTUAL EXCLUSION**, the property the claim exists for. The invariant is
   **NON-OVERLAP**, not deduplication: two *sequential* syncs over one ``source/`` are
   correct and expected, two *in flight at once* corrupt it quietly. So the arm measures
   peak concurrency inside the sync rather than counting calls, because a call count cannot
   tell those two apart.

3. The bootstrap-failure message states where the cause is **not**, so the next reader does
   not spend days on the read-only-HOME defect again.

Each arm names the mutation it must die against in a ``MUTANT`` comment above it. Those
mutations are applied from OUTSIDE pytest, against a throwaway copy of the tree, so a mutant
can never be left applied by a crashing test; each is parse-checked before it runs, because a
mutant that fails to apply leaves the arm green and a mutant that does not parse makes every
test in the file error, and neither is the same as the arm detecting the change.
"""

import ast
import errno
import logging
import sys
import threading
import time
from pathlib import Path

import pytest

from bioengine._app import replica_init


class _Logger:
    """Collects what the module logged, so a warning can be asserted on."""

    def __init__(self) -> None:
        self.warnings = []
        self.infos = []

    def warning(self, msg, *a, **kw) -> None:
        self.warnings.append(str(msg))

    def info(self, msg, *a, **kw) -> None:
        self.infos.append(str(msg))

    def debug(self, *a, **kw) -> None:
        pass

    def error(self, msg, *a, **kw) -> None:
        self.warnings.append(str(msg))


def _claim_env(monkeypatch, tmp_path, sync):
    """Env + a patched sync, matching the seams the sibling file uses.

    ``HOME`` is deliberately seeded with the image's read-only default so that nothing
    here can pass by accident on an environment that already looks correct.
    """
    app_dir = tmp_path / "ws-test-my-app"
    app_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("BIOENGINE_APP_DIR", str(app_dir))
    monkeypatch.setenv("BIOENGINE_ARTIFACT_VERSION", "1.2.3")
    monkeypatch.setenv("BIOENGINE_ARTIFACT_ID", "ws-test/my-app")
    monkeypatch.setenv("BIOENGINE_ARTIFACT_FILES_URL", "https://example.invalid/files")
    monkeypatch.setenv("HOME", "/home/ray")
    monkeypatch.delenv("BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN", raising=False)
    monkeypatch.delenv("BIOENGINE_LOCAL_ARTIFACT_PATH", raising=False)
    monkeypatch.setattr(replica_init, "_sync_source_from_hypha", sync)
    return app_dir


# ───────────────────────── 1. an uncreatable claim ─────────────────────────
#
# MUTANT: in `_acquire_claim`, replace the `except OSError` body's `raise RuntimeError(...)`
# with `pass` (proceed unclaimed). This arm must go RED, and it must go red on the
# `calls == []` assertion rather than on the message assertions, because "proceeded
# unclaimed" is the defect and "worded the error differently" is not.


@pytest.mark.parametrize(
    "err,label",
    [
        (errno.EROFS, "a read-only store"),
        (errno.ENOSPC, "a full store"),
        (errno.EACCES, "a permission change"),
        (errno.ESTALE, "a vanished NFS export"),
    ],
)
def test_an_UNCREATABLE_claim_raises_and_NEVER_syncs_unclaimed(
    monkeypatch, tmp_path, err, label
) -> None:
    """Any non-FileExistsError from the claim mkdir must stop the replica.

    Four errnos rather than one: they arrive by different routes on this mount and the
    code must not distinguish them. Each is a real failure seen on an NFS app dir.
    """
    calls = []

    def sync(files_url, version, token, source, snapshot_path, logger):
        calls.append(version)
        Path(source).mkdir(parents=True, exist_ok=True)

    app_dir = _claim_env(monkeypatch, tmp_path, sync)

    real_mkdir = Path.mkdir

    def only_the_claim_fails(self, *a, **kw):
        # Scoped to the claim by NAME, so the app dir, source/ and the staleness probe
        # still behave. A blanket mkdir failure would make this pass for the wrong
        # reason: _ensure_source would die somewhere else entirely.
        if self.name == ".claim":
            raise OSError(err, "injected")
        return real_mkdir(self, *a, **kw)

    monkeypatch.setattr(Path, "mkdir", only_the_claim_fails)

    # NOT `with pytest.raises(...)`, and the order below is the point.
    #
    # Measured while building this: with the guard mutated to proceed unclaimed, a
    # `pytest.raises(RuntimeError)` wrapper fails with "DID NOT RAISE" *before* reaching
    # the `calls == []` assertion. The arm still goes red, so it discriminates -- but the
    # red it prints names the missing exception rather than the actual defect, which is
    # that a replica materialised source with no claim held. A reader of that failure
    # goes looking for an error-reporting bug instead of a corruption window.
    #
    # So the exception is captured, and the BEHAVIOURAL assertion is made first. The
    # report assertions follow, because a wrong message is a lesser fault than a wrong
    # action and should not be what a reader sees first.
    raised = None
    try:
        replica_init._ensure_source(app_dir, "1.2.3", _Logger())
    except BaseException as exc:  # noqa: BLE001
        raised = exc

    assert calls == [], (
        f"{label}: synced WITHOUT a claim, which is the defect itself -- a replica that "
        "proceeds unclaimed looks healthy while two writers tear one shared tree"
    )
    assert isinstance(raised, RuntimeError), (
        f"{label}: an unusable store must raise RuntimeError; got {raised!r}. A bare "
        "OSError escaping here is swallowed by the bootstrap finder."
    )
    msg = str(raised)
    assert (
        str(app_dir / ".claim") in msg
    ), f"{label}: the message does not name the path"
    assert f"[Errno {err}]" in msg, f"{label}: the message does not carry the errno"


def test_a_FileExistsError_is_NOT_treated_as_uncreatable(monkeypatch, tmp_path) -> None:
    """The control for the arm above, and it is not optional.

    Without it, `except OSError: raise` would satisfy every assertion in this file while
    breaking the wait entirely — FileExistsError IS an OSError, and it is the one case
    that must NOT raise. Here the claim exists and is then released by the "holder", so a
    correct implementation waits and proceeds.
    """
    calls = []

    def sync(files_url, version, token, source, snapshot_path, logger):
        calls.append(version)
        Path(source).mkdir(parents=True, exist_ok=True)

    app_dir = _claim_env(monkeypatch, tmp_path, sync)
    monkeypatch.setattr(replica_init, "_CLAIM_POLL_S", 0.01)

    claim = app_dir / ".claim"
    claim.mkdir()

    def release_soon():
        time.sleep(0.05)
        try:
            claim.rmdir()
        except OSError:
            pass

    t = threading.Thread(target=release_soon)
    t.start()
    try:
        replica_init._ensure_source(app_dir, "1.2.3", _Logger())
    finally:
        t.join()

    assert calls == [
        "1.2.3"
    ], "a released FileExistsError claim must be waited for, not raised on"


# ───────────────────────── 2. mutual exclusion ─────────────────────────
#
# MUTANT: replace `_acquire_claim`'s body with `return app_dir / ".claim", False`
# (no mkdir, no exclusion). This arm must go RED with peak concurrency > 1.


def test_concurrent_syncs_NEVER_OVERLAP(monkeypatch, tmp_path) -> None:
    """Non-overlap, which is the invariant — not "the sync ran once".

    Four threads enter `_ensure_source` on one store at the same moment. The fake sync
    holds for 60ms, long enough that an unserialised implementation overlaps reliably
    rather than by luck, and records the peak number of callers inside it at any instant.

    Why peak concurrency and not a call count: two SEQUENTIAL syncs are correct here. A
    second replica that waits, acquires, and reconciles is doing its job. Asserting
    `len(calls) == 1` would therefore forbid correct behaviour and would also pass on an
    implementation that overlapped two calls and happened to make only one of them sync.
    Only concurrency separates them.
    """
    inside = []
    peak = []
    guard = threading.Lock()

    def sync(files_url, version, token, source, snapshot_path, logger):
        with guard:
            inside.append(1)
            peak.append(len(inside))
        try:
            time.sleep(0.06)
            Path(source).mkdir(parents=True, exist_ok=True)
        finally:
            with guard:
                inside.pop()

    app_dir = _claim_env(monkeypatch, tmp_path, sync)
    monkeypatch.setattr(replica_init, "_CLAIM_POLL_S", 0.01)

    start = threading.Barrier(4)
    errors = []

    def run():
        try:
            start.wait(timeout=5)
            replica_init._ensure_source(app_dir, "1.2.3", _Logger())
        except BaseException as exc:  # noqa: BLE001
            # Collected and re-raised in the main thread. An exception that stays in a
            # worker thread is printed and otherwise ignored, so the test would pass
            # while every call failed.
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not [t for t in threads if t.is_alive()], "a worker thread never finished"
    assert not errors, f"a concurrent caller raised: {errors[:2]}"
    # Non-vacuous: if nothing ever entered the sync, peak is empty and max() would
    # raise, so state it as the assertion instead of discovering it as an error.
    assert peak, "the sync was never entered, so non-overlap was not measured"
    assert max(peak) == 1, (
        f"two syncs were in flight at once over one source tree (peak={max(peak)}); "
        "the claim is not providing mutual exclusion"
    )


def test_the_overlap_detector_CAN_see_an_overlap(monkeypatch, tmp_path) -> None:
    """The positive control for the arm above.

    An arm that reports "peak == 1" is worthless if its instrument cannot count past one.
    This runs the SAME counting code with the claim deliberately bypassed and requires it
    to observe a peak above 1. If this test fails, the non-overlap result above proves
    nothing and must not be read as a pass.
    """
    inside = []
    peak = []
    guard = threading.Lock()

    def sync(files_url, version, token, source, snapshot_path, logger):
        with guard:
            inside.append(1)
            peak.append(len(inside))
        try:
            time.sleep(0.06)
            Path(source).mkdir(parents=True, exist_ok=True)
        finally:
            with guard:
                inside.pop()

    app_dir = _claim_env(monkeypatch, tmp_path, sync)

    # The bypass is confined to this test: _acquire_claim returns a path without ever
    # creating the directory, which is exactly the rejected "unlocked sync" design.
    monkeypatch.setattr(
        replica_init,
        "_acquire_claim",
        lambda app_dir, logger: (app_dir / ".claim", False),
    )

    start = threading.Barrier(4)

    def run():
        try:
            start.wait(timeout=5)
            replica_init._ensure_source(app_dir, "1.2.3", _Logger())
        except BaseException:  # noqa: BLE001
            pass

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert peak, "the control never entered the sync, so it measured nothing"
    assert max(peak) > 1, (
        "the overlap detector could not observe an overlap even with the claim "
        "bypassed, so it cannot substantiate the non-overlap assertion"
    )


# ───────────────────────── 3. the failure message names the cause ─────────────────────────
#
# MUTANT: restore the previous wording ("source will not be on sys.path. Traceback:") in
# bioengine/__init__.py. Both arms below must go RED.


def _bootstrap_message_literals() -> str:
    """Every string constant in the finder's `except` handler, concatenated.

    Read from the AST rather than by importing and triggering the finder, so the wording
    can be asserted without installing anything into `sys.meta_path`.
    """
    import bioengine

    tree = ast.parse(Path(bioengine.__file__).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "_install_replica_bootstrap_finder":
            continue
        out = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                out.append(sub.value)
            # f-strings: the literal halves live in FormattedValue siblings
            elif isinstance(sub, ast.JoinedStr):
                for v in sub.values:
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        out.append(v.value)
        return "\n".join(out)
    raise AssertionError(
        "_install_replica_bootstrap_finder not found in bioengine/__init__.py"
    )


def test_the_failure_message_WORDING_cannot_be_dropped_silently() -> None:
    """Structural, so a future edit that removes the explanation fails here.

    This is the guard the row asks for. It is deliberately about the SOURCE, not about a
    captured run: a behavioural arm alone would still pass if the handler were rewritten
    to print the right words on one path and the old words on another.
    """
    text = _bootstrap_message_literals()
    for needle in ("HOME", "TMPDIR", "runtime_env", "NOT the read-only-HOME defect"):
        assert needle in text, (
            f"the bootstrap-failure message no longer mentions {needle!r}; it was added "
            "because this message was misread as the read-only-HOME defect for days"
        )


def test_the_failure_message_REPORTS_the_live_HOME_and_TMPDIR(
    monkeypatch, capfd
) -> None:
    """Behavioural: induce a real bootstrap failure and read what was printed.

    The values are printed rather than asserted-as-set because the hook can fail BEFORE
    its own HOME assignment (a missing BIOENGINE_ARTIFACT_VERSION and a raising mkdir both
    precede it), so a message claiming HOME is set would be false on exactly those paths.
    Here HOME is a recognisable sentinel, and the message must contain it.
    """
    import bioengine

    home = "/home/bioengine/apps/ws-test-my-app/home"
    tmp = "/home/bioengine/apps/ws-test-my-app/tmp"
    monkeypatch.setenv("BIOENGINE_APP_DIR", "/nonexistent-app-dir-for-this-test")
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("TMPDIR", tmp)

    def boom() -> None:
        raise RuntimeError("injected source sync failure")

    monkeypatch.setattr(replica_init, "setup_replica_environment", boom, raising=True)

    before = list(sys.meta_path)
    bioengine._install_replica_bootstrap_finder()
    finder = sys.meta_path[-1]
    assert (
        finder not in before
    ), "the finder was not installed, so nothing was exercised"
    try:
        # A module name the finder will not skip, and which does not exist, so the only
        # effect is firing the hook.
        finder.find_spec("a_module_that_does_not_exist_anywhere_in_this_tree", None)
    finally:
        sys.meta_path.remove(finder)

    out = capfd.readouterr()
    printed = out.out + out.err
    assert (
        "replica source bootstrap failed" in printed
    ), "the handler did not print at all"
    assert home in printed, "the message does not report the live HOME"
    assert tmp in printed, "the message does not report the live TMPDIR"
    assert "injected source sync failure" in printed, "the traceback was not printed"
