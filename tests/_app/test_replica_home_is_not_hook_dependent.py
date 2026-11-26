"""A replica's HOME must not depend on the setup hook having run, or having succeeded.

WHAT THIS PINS. Were HOME and TMPDIR set in only one place --
``replica_init.setup_replica_environment()``, and only *after* that function has
materialised the user's source over the network -- then two independent things have to go
right before a replica has a writable HOME:

  1. **The hook had to run at all.** Ray Serve does not honour
     ``runtime_env.worker_process_setup_hook`` for its replicas, so the hook is reached
     only through the ``sys.meta_path`` finder installed by ``bioengine/__init__.py``.
     That finder is *appended* to ``sys.meta_path``, so it is consulted only when every
     standard finder has missed — i.e. it fires as a side effect of the first
     **unresolved** import in the process. A replica whose imports all resolve never
     reaches it.
  2. **The source sync had to succeed.** ``_ensure_source`` reaches the network under a
     file lock. It ran *before* the HOME assignment, and the finder marks itself fired
     before calling the hook and swallows any exception to a printed traceback, never
     retrying. So a sync that raised left HOME untouched, permanently.

Either way HOME stays at the image's ``/home/ray``, which is read-only in the published
worker image, and the app's first write to its own store fails with
``OSError: [Errno 30] Read-only file system`` — a symptom two steps removed from its
cause. The failure is per-replica and nondeterministic: of four replicas of one app
created within half a second, one can log ``replica environment ready``, one
``replica source bootstrap failed``, and one neither — and the latter two both hit the
read-only error.

WHAT THE FIX IS, AND WHY BOTH LEGS ARE HERE. The paths are now composed into the
``runtime_env`` env_vars by ``AppBuilder._build_env_vars`` — applied by Ray before any
user code imports, so that leg cannot be skipped — *and* the hook sets them before
``_ensure_source`` rather than after. Testing one leg alone would leave the other free to
regress, and each on its own is enough to reintroduce a read-only HOME on some replica:
drop the builder leg and a replica whose hook never fires is back to ``/home/ray``; drop
the ordering and a replica whose source sync raises is too (the builder leg covers that
one, but only for as long as the builder leg exists).

WHAT THIS DOES NOT CLAIM. Not that the directory is writable — that is a property of the
volume, not of this code. Not that the source sync works. Not that an app reads ``HOME``
rather than hard-coding a path; ``os.path.expanduser`` and ``Path.home()`` both consult
it, which is what makes it the right lever, but an app is free to ignore it.
"""

from __future__ import annotations

import errno
import logging
import os
import time
from pathlib import Path

import pytest

from bioengine._app import replica_init
from bioengine.apps.builder import AppBuilder

# ─────────────────────────── leg 1: the builder ────────────────────────────


class _Logger:
    """Records warnings so the best-effort mkdir can be asserted on."""

    def __init__(self) -> None:
        self.warnings: list = []

    def warning(self, msg) -> None:
        self.warnings.append(str(msg))

    def info(self, msg) -> None:
        pass


def _builder(apps_workdir: Path, logger=None) -> AppBuilder:
    """An ``AppBuilder`` carrying only the attributes ``_build_env_vars`` reads.

    ``__init__`` wants a Ray cluster and a Hypha connection; the function under test
    touches seven attributes and nothing else, so the real constructor is bypassed
    rather than mocked. Calling the real function is the point — an AST assertion about
    where the keys are written would not notice them being overwritten further down.
    """
    b = object.__new__(AppBuilder)
    b.server = None
    b.server_url = "https://example.invalid"
    b.apps_workdir = apps_workdir
    b.worker_service_id = None
    b.proxy_actor_name = None
    b.data_server_url = None
    b.logger = logger if logger is not None else _Logger()
    return b


def _env(apps_workdir: Path, non_secret=None, logger=None) -> dict:
    return _builder(apps_workdir, logger)._build_env_vars(
        application_id="my-app",
        artifact_id="ws-test/my-app",
        version="1.2.3",
        non_secret_env_vars=non_secret or {},
        secret_env_vars={},
        hypha_token=None,
        download_token=None,
    )


def test_builder_puts_HOME_in_the_runtime_env(tmp_path) -> None:
    """The leg that does not depend on the hook firing."""
    env = _env(tmp_path)
    assert env["HOME"] == str(tmp_path / "local-my-app" / "home")


def test_builder_puts_TMPDIR_and_its_aliases_in_the_runtime_env(tmp_path) -> None:
    env = _env(tmp_path)
    tmp = str(tmp_path / "local-my-app" / "tmp")
    assert env["TMPDIR"] == tmp
    # TMP/TEMP are what a surprising amount of third-party code reads instead.
    assert env["TMP"] == tmp
    assert env["TEMP"] == tmp


def test_builder_HOME_is_under_the_app_dir_it_also_exports(tmp_path) -> None:
    """The two keys must agree, because the hook recomputes HOME from BIOENGINE_APP_DIR.

    If they disagreed, a replica would get one HOME before its first unresolved import
    and a different one after — which is worse than either alone.
    """
    env = _env(tmp_path)
    app_dir = Path(env["BIOENGINE_APP_DIR"])
    assert Path(env["HOME"]) == app_dir / "home"
    assert Path(env["TMPDIR"]) == app_dir / "tmp"


def test_builder_creates_the_directories(tmp_path) -> None:
    """Exported but absent would fail ENOENT instead of EROFS — no better for an app
    that does not mkdir its own parents."""
    env = _env(tmp_path)
    assert Path(env["HOME"]).is_dir()
    assert Path(env["TMPDIR"]).is_dir()


def test_builder_mkdir_failure_warns_but_does_not_refuse_the_deploy(tmp_path) -> None:
    """A venue where the worker cannot write apps_workdir must still deploy.

    The replica side creates the directories too, so refusing here would break
    deployments that work today. The warning is the compensating signal.
    """
    blocked = tmp_path / "blocked"
    blocked.write_text("I am a file, so mkdir underneath me raises")
    logger = _Logger()
    env = _env(blocked, logger=logger)
    # Still exported, so the replica-side mkdir has somewhere to aim.
    assert env["HOME"] == str(blocked / "local-my-app" / "home")
    assert any("Could not pre-create" in w for w in logger.warnings), logger.warnings


def test_builder_HOME_wins_over_a_manifest_supplied_one(tmp_path) -> None:
    """An app declaring HOME in its manifest must not get the read-only default back.

    The hook has always overridden HOME unconditionally; the two legs must not disagree
    about that, or which HOME a replica ends up with would again depend on whether the
    hook ran.
    """
    env = _env(tmp_path, non_secret={"HOME": "/home/ray"})
    assert env["HOME"] == str(tmp_path / "local-my-app" / "home")


def test_builder_still_exports_what_the_hook_needs(tmp_path) -> None:
    """A fix that passed by emptying the dict would be worthless."""
    env = _env(tmp_path)
    for required in (
        "BIOENGINE_APP_DIR",
        "BIOENGINE_ARTIFACT_VERSION",
        "BIOENGINE_ARTIFACT_FILES_URL",
        "BIOENGINE_ARTIFACT_ID",
        "BIOENGINE_APPLICATION_ID",
    ):
        assert env.get(required), f"{required} is missing"


# ──────────────────────── leg 2: the hook's ordering ───────────────────────


@pytest.fixture
def hook_env(monkeypatch, tmp_path):
    """Minimal env for ``setup_replica_environment``; sync is stubbed per test."""
    app_dir = tmp_path / "ws-test-my-app"
    monkeypatch.setenv("BIOENGINE_APP_DIR", str(app_dir))
    monkeypatch.setenv("BIOENGINE_ARTIFACT_VERSION", "1.2.3")
    monkeypatch.setenv("BIOENGINE_ARTIFACT_ID", "ws-test/my-app")
    monkeypatch.setenv("BIOENGINE_ARTIFACT_FILES_URL", "https://example.invalid/files")
    monkeypatch.setenv("HOME", "/home/ray")  # the read-only default being escaped
    monkeypatch.delenv("BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN", raising=False)
    monkeypatch.delenv("BIOENGINE_LOCAL_ARTIFACT_PATH", raising=False)
    return app_dir


def test_HOME_is_already_set_when_the_source_sync_is_CALLED(
    monkeypatch, hook_env
) -> None:
    """The ordering assertion, stated directly rather than through its consequence.

    Asserting only the end state would stay green if the sync were moved back in front,
    because a *successful* sync leaves the same end state either way. What makes the
    difference observable is what HOME is at the moment the fallible step runs.
    """
    app_dir = hook_env
    seen = {}

    def spy(app_dir_arg, version, logger):
        seen["HOME"] = os.environ.get("HOME")
        seen["TMPDIR"] = os.environ.get("TMPDIR")
        src = Path(app_dir_arg) / "source"
        src.mkdir(parents=True, exist_ok=True)
        return src

    monkeypatch.setattr(replica_init, "_ensure_source", spy)
    replica_init.setup_replica_environment()

    assert seen["HOME"] == str(app_dir / "home")
    assert seen["TMPDIR"] == str(app_dir / "tmp")


def test_HOME_survives_a_source_sync_that_RAISES(monkeypatch, hook_env) -> None:
    """The production failure, reproduced: the sync raised and HOME stayed read-only.

    The caller (the ``sys.meta_path`` finder) catches this and carries on with the user's
    source missing from ``sys.path``, which is a real but *separate* degradation. It must
    no longer also cost the app a writable HOME.
    """
    app_dir = hook_env

    def boom(app_dir_arg, version, logger):
        raise RuntimeError("artifact files endpoint returned 503")

    monkeypatch.setattr(replica_init, "_ensure_source", boom)
    with pytest.raises(RuntimeError, match="503"):
        replica_init.setup_replica_environment()

    assert os.environ["HOME"] == str(app_dir / "home")
    assert os.environ["TMPDIR"] == str(app_dir / "tmp")
    assert Path(os.environ["HOME"]).is_dir()
    assert Path(os.environ["TMPDIR"]).is_dir()


def test_the_fixture_really_starts_from_the_read_only_default(hook_env) -> None:
    """Without this, both tests above could pass on a machine where HOME happens to
    already point somewhere under tmp_path, and would be asserting nothing."""
    assert os.environ["HOME"] == "/home/ray"


# ─────────────────── leg 3: the lock that cannot work on NFS ───────────────────
#
# The measured production trigger. `_ensure_source` serialised concurrent replicas with
# `fcntl.flock(LOCK_EX)` on the app directory. That directory is an NFS mount with no lock
# manager, so the call raises `OSError: [Errno 37] No locks available` — six lines before
# HOME was set, escaping `setup_replica_environment`, swallowed by the bootstrap finder,
# leaving the replica serving with HOME=/home/ray, which is read-only in the published
# image. The assignment sits after the sync in the function, so an ENOLCK from the lock
# never reaches it.
#
# The lock is REPLACED, not dropped: `mkdir` is atomic on NFS. Degrading to an unlocked
# sync was rejected because ENOLCK is frequent on this mount, so the unlocked path would
# have become the normal path and concurrent writers into one tree the normal case.


def _sync_env(monkeypatch, tmp_path):
    app_dir = tmp_path / "ws-test-my-app"
    monkeypatch.setenv("BIOENGINE_APP_DIR", str(app_dir))
    monkeypatch.setenv("BIOENGINE_ARTIFACT_VERSION", "1.2.3")
    monkeypatch.setenv("BIOENGINE_ARTIFACT_ID", "ws-test/my-app")
    monkeypatch.setenv("BIOENGINE_ARTIFACT_FILES_URL", "https://example.invalid/files")
    monkeypatch.setenv("HOME", "/home/ray")
    monkeypatch.delenv("BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN", raising=False)
    monkeypatch.delenv("BIOENGINE_LOCAL_ARTIFACT_PATH", raising=False)
    calls = []
    held = []

    def fake_sync(files_url, version, token, source, snapshot_path, logger):
        calls.append(version)
        # Was the claim actually HELD while the sync ran? Without this the release
        # assertions below pass on code that never takes a claim at all — a vacuous
        # pass that reads exactly like a real one.
        held.append((app_dir / ".claim").is_dir())
        Path(source).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(replica_init, "_sync_source_from_hypha", fake_sync)
    return app_dir, calls, held


def test_the_module_takes_no_byte_range_lock_AT_ALL(monkeypatch, tmp_path):
    """Structural, so the lock cannot quietly come back.

    Asserted over the AST rather than the text: the docstrings deliberately *mention*
    `fcntl.flock` to explain why it was removed, and a grep would score that prose as the
    defect. This looks for an import of, or an attribute access on, `fcntl`.
    """
    import ast as _ast

    tree = _ast.parse(Path(replica_init.__file__).read_text())
    bad = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            bad += [a.name for a in node.names if a.name.split(".")[0] == "fcntl"]
        elif isinstance(node, _ast.ImportFrom) and (node.module or "").startswith(
            "fcntl"
        ):
            bad.append(node.module)
        elif isinstance(node, _ast.Attribute) and isinstance(node.value, _ast.Name):
            if node.value.id == "fcntl":
                bad.append(f"fcntl.{node.attr}")
    assert bad == [], f"replica_init still uses fcntl: {bad}"


def test_ENOLCK_no_longer_stops_a_replica_reaching_ready(monkeypatch, tmp_path):
    """THE REAL-WORLD ARM. flock raises ENOLCK as an NFS mount with no lock manager does.

    On the previous code this is the failure: `_ensure_source` propagates ENOLCK out of
    `setup_replica_environment` before HOME is assigned. Here the call must complete with
    a writable HOME, because no byte-range lock is taken at all.

    Patching `fcntl.flock` to raise is what makes this an arm rather than a restatement of
    the structural test above: it fails loudly if any code path still reaches a flock,
    including one added later in a helper this test does not know about.
    """
    import errno
    import fcntl

    app_dir, calls, held = _sync_env(monkeypatch, tmp_path)

    def enolck(*_a, **_k):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", enolck)
    replica_init.setup_replica_environment()

    assert os.environ["HOME"] == str(app_dir / "home")
    assert Path(os.environ["HOME"]).is_dir()
    assert calls == ["1.2.3"], "the source sync must still have happened"


def test_the_claim_is_HELD_during_the_sync_and_released_after(monkeypatch, tmp_path):
    """Both halves, so neither can pass vacuously.

    "Released afterwards" alone is true of code that never claims anything — it was, on
    the parent commit, where this assertion passed while measuring nothing.
    """
    app_dir, _calls, held = _sync_env(monkeypatch, tmp_path)
    replica_init.setup_replica_environment()
    assert held == [True], "the sync ran without the claim being held"
    assert not (app_dir / ".claim").exists(), "the claim was not released"


def test_the_claim_is_released_even_when_the_sync_RAISES(monkeypatch, tmp_path):
    """A claim leaked on the error path would stall every later replica for the full wait."""
    app_dir, _calls, held = _sync_env(monkeypatch, tmp_path)

    def boom(*_a, **_k):
        raise RuntimeError("files endpoint returned 503")

    seen = []

    def boom_seeing_claim(*_a, **_k):
        seen.append((app_dir / ".claim").is_dir())
        raise RuntimeError("files endpoint returned 503")

    monkeypatch.setattr(replica_init, "_sync_source_from_hypha", boom_seeing_claim)
    with pytest.raises(RuntimeError, match="503"):
        replica_init.setup_replica_environment()
    assert seen == [True], "the failing sync ran without the claim being held"
    assert not (app_dir / ".claim").exists(), "the claim leaked on the error path"


def test_a_second_replica_uses_the_winners_tree_instead_of_re_downloading(
    monkeypatch, tmp_path
):
    """The marker is only written after a completed sync, so trusting it inside the claim
    is safe — and it is what makes the LOSER cheap rather than a duplicate download.

    The loser is the point, so the loser is what this simulates: a claim is already held
    when we arrive, and it is released (with the winner's tree and marker published) on
    the first poll. Asserting this with two plain sequential calls instead would describe
    an ordinary restart, where skipping the sync is wrong — reconciliation is this
    module's contract on every start, and `test_missing_local_file_redownloaded`,
    `test_changed_timestamp_redownloads_only_that_file` and
    `test_removed_file_deleted_locally` all pin it.
    """
    app_dir, calls, held = _sync_env(monkeypatch, tmp_path)

    # The winner: materialises, publishes its marker, and leaves its claim in place.
    app_dir.mkdir(parents=True, exist_ok=True)
    claim = app_dir / ".claim"
    claim.mkdir()
    (app_dir / "source").mkdir(parents=True, exist_ok=True)
    (app_dir / ".version").write_text("ws-test/my-app@1.2.3")

    # The winner finishes while we are waiting: its claim disappears on the first poll.
    def release_on_first_poll(_seconds):
        try:
            claim.rmdir()
        except OSError:
            pass

    monkeypatch.setattr(replica_init.time, "sleep", release_on_first_poll)

    replica_init.setup_replica_environment()

    assert calls == [], f"the loser re-synced the winner's tree: {calls}"


def test_an_abandoned_claim_is_broken(monkeypatch, tmp_path):
    app_dir, calls, held = _sync_env(monkeypatch, tmp_path)
    app_dir.mkdir(parents=True, exist_ok=True)
    claim = app_dir / ".claim"
    claim.mkdir()
    old = time.time() - (replica_init._CLAIM_STALE_S + 600)
    os.utime(claim, (old, old))
    replica_init.setup_replica_environment()
    assert calls == ["1.2.3"]
    assert os.environ["HOME"] == str(app_dir / "home")


def test_a_LIVE_claim_makes_it_fail_LOUDLY_not_sync_unclaimed(monkeypatch, tmp_path):
    """The whole point of replacing the lock rather than dropping it.

    A fresh claim held by someone else must stop this replica with a raise. If it instead
    proceeded, two replicas would write one tree concurrently — which is exactly what the
    rejected "tolerate ENOLCK by skipping the lock" design would have done, routinely.
    """
    app_dir, calls, held = _sync_env(monkeypatch, tmp_path)
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / ".claim").mkdir()
    monkeypatch.setattr(replica_init, "_CLAIM_WAIT_S", 0)
    with pytest.raises(RuntimeError, match="claim"):
        replica_init.setup_replica_environment()
    assert calls == [], "it synced anyway, without a claim"


def test_staleness_survives_a_SKEWED_POD_CLOCK(monkeypatch, tmp_path):
    """A network file server's clock can run well ahead of its clients.

    So `time.time() - st_mtime` is the wrong subtraction, and at a large enough skew it
    reads a brand-new claim as abandoned and breaks it — reintroducing concurrent writers
    through the staleness path instead of the lock path. `_store_now` stats a probe it
    writes into the same directory, so both times come from the store's clock.

    Driven by moving the POD clock far forward: a live claim must still be respected.
    """
    app_dir, calls, held = _sync_env(monkeypatch, tmp_path)
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / ".claim").mkdir()  # fresh, so it must NOT be broken
    skew = replica_init._CLAIM_STALE_S * 10
    real_time = time.time  # captured BEFORE the patch, or the lambda calls itself
    monkeypatch.setattr(replica_init.time, "time", lambda: real_time() + skew)
    assert replica_init.time.time() - real_time() > skew / 2, "the skew patch is inert"
    monkeypatch.setattr(replica_init, "_CLAIM_WAIT_S", 0)
    with pytest.raises(RuntimeError, match="claim"):
        replica_init.setup_replica_environment()
    assert calls == [], "a skewed pod clock broke a live claim and synced anyway"


# ───────────────── leg 4: the claim's own bounds, both measured ─────────────────


def test_the_wait_bound_EXCEEDS_the_staleness_bound(monkeypatch, tmp_path):
    """The invariant, asserted on the shipped constants.

    A waiter that gives up before a claim can be judged abandoned can never break one. The
    replacement for a replica that died mid-sync arrives at age ~0 and is exactly the
    caller that must survive this, so `wait <= stale` makes the claim unrecoverable in the
    one case it exists for. Asserted on the constants because that is what cannot drift:
    the module derives the wait from the staleness bound rather than writing a second
    literal, and this fails if anyone unpicks that.
    """
    assert replica_init._CLAIM_WAIT_S > replica_init._CLAIM_STALE_S, (
        f"wait={replica_init._CLAIM_WAIT_S}s does not exceed "
        f"stale={replica_init._CLAIM_STALE_S}s, so an abandoned claim is never broken"
    )


def test_an_abandoned_YOUNG_claim_is_eventually_acquired(monkeypatch, tmp_path):
    """The behaviour the invariant buys, driven with the shipped *shape* scaled down.

    The claim is fresh on arrival and nobody ever releases it. With the wait exceeding the
    staleness bound the waiter outlives the threshold, breaks it and proceeds; with the
    bounds the other way round it raises.
    """
    app_dir, calls, held = _sync_env(monkeypatch, tmp_path)
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / ".claim").mkdir()
    monkeypatch.setattr(replica_init, "_CLAIM_STALE_S", 0.2)
    monkeypatch.setattr(replica_init, "_CLAIM_WAIT_S", 0.5)
    monkeypatch.setattr(replica_init, "_CLAIM_POLL_S", 0.02)
    replica_init.setup_replica_environment()
    assert calls == ["1.2.3"], "it never got the claim, so it never synced"
    assert os.environ["HOME"] == str(app_dir / "home")


def test_an_UNREADABLE_claim_does_not_spin_past_the_deadline(monkeypatch, tmp_path):
    """A retry path that skipped the deadline turned a bounded wait into a busy loop.

    `_store_now` writes a probe into the store, so it fails whenever the store does — a
    full quota, a read-only remount, a vanished mount. Each failure used to `continue`
    straight back to the mkdir with no sleep and no deadline check, so the loop spun as
    fast as the CPU allowed and never ended.

    Bounded here by COUNTING the attempts, not by wall clock: a test that only asserted
    "it raises" would pass on the spinning version too, since it does eventually raise
    when the deadline is checked — it just burns a core until then.
    """
    app_dir, calls, held = _sync_env(monkeypatch, tmp_path)
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / ".claim").mkdir()
    attempts = []

    def unreadable(_app_dir):
        attempts.append(1)
        if len(attempts) > 500:
            raise AssertionError(f"SPUN {len(attempts)} times past the deadline")
        raise OSError(122, "Disk quota exceeded")

    monkeypatch.setattr(replica_init, "_store_now", unreadable)
    monkeypatch.setattr(replica_init, "_CLAIM_WAIT_S", 0.2)
    monkeypatch.setattr(replica_init, "_CLAIM_POLL_S", 0.05)
    with pytest.raises(RuntimeError, match="claim"):
        replica_init.setup_replica_environment()
    # wait/poll is ~4 polls; allow generous slack but nothing like a spin.
    assert len(attempts) <= 40, f"{len(attempts)} attempts for a 0.2s/0.05s budget"
    assert calls == [], "it synced without a claim"


def test_a_FAILING_rmdir_on_a_stale_claim_does_not_spin(monkeypatch, tmp_path):
    """The stale-BREAK path, which the first spin fix did not route.

    `rmdir` failing is not "someone else got there first" — that case succeeds, or raises
    ENOENT, and both are fine. It is also EPERM in a sticky directory whose claim another
    uid created, ENOTEMPTY from an NFS silly-rename file left inside it, ESTALE, EACCES.
    Each of those recurs on the next pass, so retrying with no deadline and no sleep span
    the loop exactly as the stat path did.

    Counted, not timed, for the same reason as the stat arm: the spinning version does
    eventually raise once something checks the deadline, so "it raises" is not the test.
    """
    app_dir, calls, held = _sync_env(monkeypatch, tmp_path)
    app_dir.mkdir(parents=True, exist_ok=True)
    claim = app_dir / ".claim"
    claim.mkdir()
    old = time.time() - (replica_init._CLAIM_STALE_S + 600)
    os.utime(claim, (old, old))

    attempts = []
    real_rmdir = Path.rmdir

    def refusing_rmdir(self):
        if self.name == ".claim":
            attempts.append(1)
            if len(attempts) > 500:
                raise AssertionError(f"STALE PATH SPUN {len(attempts)} times")
            raise OSError(1, "Operation not permitted")
        return real_rmdir(self)

    monkeypatch.setattr(Path, "rmdir", refusing_rmdir)
    monkeypatch.setattr(replica_init, "_CLAIM_WAIT_S", 0)
    monkeypatch.setattr(replica_init, "_CLAIM_POLL_S", 0.01)
    with pytest.raises(RuntimeError, match="claim"):
        replica_init.setup_replica_environment()
    assert len(attempts) <= 10, f"{len(attempts)} rmdir attempts past a 0s deadline"
    assert calls == [], "it synced without a claim"


def test_an_ENOENT_rmdir_retries_IMMEDIATELY_without_sleeping(monkeypatch, tmp_path):
    """ENOENT from the stale-break rmdir genuinely IS "someone else got there first".

    The claim is already gone, so nothing is being waited for and the next `mkdir` can win
    straight away. Paying the poll interval for it delays a replica that is in fact
    unblocked — and with a production poll of 2 s and a cold cluster bringing replicas up
    together, that is a real stall rather than a rounding error.

    Driven by making the poll expensive and asserting it was never paid: a wall-clock
    bound would pass whether or not the sleep happened, on a machine fast enough.
    """
    app_dir, calls, held = _sync_env(monkeypatch, tmp_path)
    app_dir.mkdir(parents=True, exist_ok=True)
    claim = app_dir / ".claim"
    claim.mkdir()
    old = time.time() - (replica_init._CLAIM_STALE_S + 600)
    os.utime(claim, (old, old))

    real_rmdir = Path.rmdir
    slept = []

    def vanished_rmdir(self):
        if self.name == ".claim":
            real_rmdir(self)  # actually remove it, then report it was already gone
            raise OSError(errno.ENOENT, "No such file or directory")
        return real_rmdir(self)

    monkeypatch.setattr(Path, "rmdir", vanished_rmdir)
    monkeypatch.setattr(replica_init.time, "sleep", lambda s: slept.append(s))
    replica_init.setup_replica_environment()

    assert calls == ["1.2.3"], "it did not get the claim"
    assert slept == [], f"it slept {slept} for a claim that was already gone"


def test_the_timeout_message_CARRIES_the_errno_even_when_an_age_is_known(
    monkeypatch, tmp_path
):
    """The errno is the part a reader can act on.

    Reporting only "held by another replica for Ns" sends whoever reads the log looking
    for a busy peer, when the real answer is EPERM or ENOTEMPTY on the store. Both facts
    are known at that point, so both are reported.
    """
    app_dir, calls, held = _sync_env(monkeypatch, tmp_path)
    app_dir.mkdir(parents=True, exist_ok=True)
    claim = app_dir / ".claim"
    claim.mkdir()
    old = time.time() - (replica_init._CLAIM_STALE_S + 600)
    os.utime(claim, (old, old))

    real_rmdir = Path.rmdir

    def refusing(self):
        if self.name == ".claim":
            raise OSError(errno.ENOTEMPTY, "Directory not empty")
        return real_rmdir(self)

    monkeypatch.setattr(Path, "rmdir", refusing)
    monkeypatch.setattr(replica_init, "_CLAIM_WAIT_S", 0)
    monkeypatch.setattr(replica_init, "_CLAIM_POLL_S", 0.01)
    with pytest.raises(RuntimeError) as ei:
        replica_init.setup_replica_environment()
    msg = str(ei.value)
    assert "Directory not empty" in msg, f"the errno never reached the message: {msg}"
    assert "held by another replica" in msg, f"the age was dropped instead: {msg}"


# ───────── leg 5: two gaps the superseded sibling PR identified, adapted ─────────
#
# A sibling pull request proposed a different fix for the same defect and was closed
# unmerged once this one superseded it, with its ordering change folded in here. Two of
# its test ideas were NOT folded in, and neither is covered by anything above:
#
#   * the claim directory being UNCREATABLE at all (not held by a peer — mkdir itself
#     failing, as it would on a read-only or quota-exhausted store);
#   * genuine MUTUAL EXCLUSION between two concurrent replicas.
#
# Its own versions could not be applied: they assert a design that was rejected (a
# `.materialising` path, an `errno=` message format, a `_CLAIM_POLL_SECONDS` constant),
# so they would fail against the code that actually landed. These are the same two
# properties written against the landed implementation.


def test_a_claim_that_CANNOT_BE_CREATED_fails_loudly_naming_the_path(
    monkeypatch, tmp_path
):
    """mkdir failing outright is not the same as a peer holding the claim.

    `FileExistsError` means wait; every other `OSError` from that mkdir means the store
    will not accept a claim at all — a read-only remount, an exhausted quota, a vanished
    mount. That must raise, name the path, and carry the errno, because silence is what
    made the original defect expensive: a replica that proceeds unclaimed looks healthy
    and corrupts a shared tree instead.
    """
    import errno as _errno

    app_dir, calls, held = _sync_env(monkeypatch, tmp_path)
    app_dir.mkdir(parents=True, exist_ok=True)
    real_mkdir = Path.mkdir

    def refusing_mkdir(self, *a, **kw):
        if self.name == ".claim":
            raise OSError(_errno.EROFS, "Read-only file system")
        return real_mkdir(self, *a, **kw)

    monkeypatch.setattr(Path, "mkdir", refusing_mkdir)
    # Bound the wait. On correct code the raise is immediate, so this changes nothing --
    # but a regression that SWALLOWS the error does not proceed unclaimed, it spins until
    # the deadline, and with the shipped 1200 s bound that turns this arm into a
    # twenty-minute hang instead of a failure. Measured while building the mutant control
    # for this very test: the run had to be killed. A test that hangs on the defect it
    # guards is a test nobody will keep.
    monkeypatch.setattr(replica_init, "_CLAIM_WAIT_S", 1)
    monkeypatch.setattr(replica_init, "_CLAIM_POLL_S", 0.05)
    with pytest.raises(RuntimeError) as ei:
        replica_init.setup_replica_environment()
    msg = str(ei.value)
    assert ".claim" in msg, f"the error does not name the claim path: {msg}"
    assert "Read-only file system" in msg, f"the error does not carry the errno: {msg}"
    assert calls == [], "it synced despite being unable to claim"


def test_two_concurrent_replicas_NEVER_SYNC_AT_ONCE(monkeypatch, tmp_path):
    """The property the claim exists for, and the only arm here that is really concurrent.

    THE INVARIANT IS NON-OVERLAP, NOT DEDUPLICATION — and asserting the wrong one would
    fail against correct code. The loser acquires the claim once the winner releases it,
    and (since the marker fast path only applies to a replica that waited) a second sync
    is legitimate. What must never happen is two syncs IN FLIGHT AT ONCE over one
    `source/`, because that is what tears the tree, and it tears it quietly.

    So this records each sync's [start, end] and asserts no two intervals overlap.
    """
    import threading

    app_dir, _calls, _held = _sync_env(monkeypatch, tmp_path)
    spans, lock = [], threading.Lock()

    def slow_sync(files_url, version, token, source, snapshot_path, logger):
        t0 = time.monotonic()
        time.sleep(0.4)
        Path(source).mkdir(parents=True, exist_ok=True)
        with lock:
            spans.append((t0, time.monotonic()))

    monkeypatch.setattr(replica_init, "_sync_source_from_hypha", slow_sync)
    monkeypatch.setattr(replica_init, "_CLAIM_POLL_S", 0.02)
    # The wait MUST be bounded here. Left at the shipped value, a regression that never
    # releases the claim makes the loser wait the full interval: the arm then measured one
    # replica, passed, and kept a non-daemon thread alive for twenty minutes. A guard that
    # reports PASSED for a run whose second replica never finished is worse than no guard.
    monkeypatch.setattr(replica_init, "_CLAIM_WAIT_S", 5)
    errors = []

    def run():
        try:
            replica_init._ensure_source(app_dir, "1.2.3", logging.getLogger("t"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    # Both replicas must have FINISHED. join(timeout=...) returns whether or not the thread
    # ended, so without this a hung loser is indistinguishable from a fast one, and the
    # non-overlap assertion below would run over a single interval and trivially hold.
    alive = [t for t in threads if t.is_alive()]
    assert not alive, (
        f"{len(alive)} replica thread(s) were still running after join: the claim was never "
        "released, so this run measured fewer replicas than it started and proves nothing"
    )
    assert not errors, f"a replica raised: {errors}"
    # ONE span is legitimate and must stay allowed: `_ensure_source` trusts the version marker
    # only for a replica that actually WAITED, so the loser reuses the winner's tree instead of
    # re-syncing. Requiring two would assert against the design. Non-overlap is then trivially
    # true for a single interval — which is exactly why the liveness assertion above is what
    # carries this arm: it proves both replicas FINISHED, so one span means the loser reused
    # the marker rather than that it hung and was never measured.
    assert len(spans) >= 1, "neither replica synced, so nothing was measured"
    overlaps = [
        (a, b)
        for i, a in enumerate(spans)
        for b in spans[i + 1 :]
        if a[0] < b[1] and b[0] < a[1]
    ]
    assert (
        not overlaps
    ), f"two syncs were in flight at once over one source/: {overlaps}"


def test_the_bootstrap_failure_message_POINTS_AT_THE_REAL_CAUSE(monkeypatch, tmp_path):
    """The finder swallows a bootstrap failure into a printed traceback, so that print is the
    only thing a reader gets. It must report where the cause is.

    Without it the traceback reads as "source missing from sys.path" and the read-only write
    that follows looks like a separate fault — which is how the original defect stayed
    unexplained for eleven days.

    THIS ASSERTS THE PRINTED OUTPUT, not the source text, and the difference is the whole
    point. A structural check over the message literal is beaten by any rewrite that keeps the
    right tokens: one that says the variables are set AFTER the sync "never BEFORE" still
    contains "BEFORE" and still calls environ.get twice, and one that formats the values away
    with "%.0s" still has the environ.get calls while printing nothing. Both of those pass a
    structural guard and say the opposite of, or nothing about, the fact the clause asks for.

    So the arm drives the finder: it makes the bootstrap raise, calls find_spec for a module
    nothing can resolve, and reads stdout. HOME is set to a sentinel and TMPDIR is removed, so
    the assertions below can only hold if the message prints the values it read.

    The sentinel also guards against the wrong copy of bioengine: if the installed package is
    imported instead of the checkout, its older message cannot contain this sentinel and the
    arm fails rather than passing against code that is not under test.
    """
    import contextlib
    import io
    import sys as _sys

    import bioengine

    app_dir = tmp_path / "app"
    app_dir.mkdir()
    sentinel = str(tmp_path / "SENTINEL-home-c9f31a")
    monkeypatch.setenv("BIOENGINE_APP_DIR", str(app_dir))
    monkeypatch.setenv("HOME", sentinel)
    monkeypatch.delenv("TMPDIR", raising=False)

    def _raise() -> None:
        raise RuntimeError("deliberate bootstrap failure")

    monkeypatch.setattr(replica_init, "setup_replica_environment", _raise)

    before = list(_sys.meta_path)
    try:
        bioengine._install_replica_bootstrap_finder()
        assert len(_sys.meta_path) == len(before) + 1, (
            "the finder was not installed, so nothing below exercises the message. "
            "BIOENGINE_APP_DIR is set, which is its only precondition."
        )
        finder = _sys.meta_path[-1]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            finder.find_spec("definitely_not_a_module_xyz", None)
        out = buf.getvalue()
    finally:
        _sys.meta_path[:] = before

    assert (
        "replica source bootstrap failed" in out
    ), f"the finder printed no bootstrap-failure message at all; got {out[:200]!r}"
    # The VALUES must be printed. A message that reads os.environ and then discards the
    # values with a zero-width format is the mutant this catches.
    assert sentinel in out, (
        f"the message does not print the HOME it read (expected {sentinel!r}). A reader cannot "
        f"check an environment claim against a traceback, which never shows env values. Got: {out[:300]!r}"
    )
    assert "TMPDIR=<unset>" in out, (
        f"TMPDIR was removed from the environment and the message does not report it as unset, "
        f"so it is not reporting what it read. Got: {out[:300]!r}"
    )
    # …and the claim itself, so a message that states the OPPOSITE in other words is RED even
    # though it contains every keyword.
    assert "BEFORE syncing source" in out, (
        f"the message no longer states that the redirection happens BEFORE the source sync. That "
        f"is the fact that makes an unset HOME point at an earlier failure rather than at the "
        f"eventual read-only write. Got: {out[:300]!r}"
    )
