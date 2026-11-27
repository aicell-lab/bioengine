"""A replica whose HOME was never redirected must say so, loudly, and repair it.

THE FAILURE THIS COVERS, measured on a live external-cluster site: eight of eight
``ray::ServeReplica:*`` processes for a freshly deployed app served with ``HOME=/home/ray`` — the
image default, read-only — while the identical scan on another site showed six of six under the
per-app home. The app then refused every model type with
``[Errno 30] Read-only file system: '/home/ray/.bioengine'``.

Two legs were supposed to prevent that (the worker composes ``HOME`` into the replica's
``runtime_env.env_vars``, and the setup hook assigns it) and on that site neither delivered. Nothing
reported it: the bootstrap finder's loud path only fires when ``setup_replica_environment`` raises,
and the replica had even logged ``replica environment ready`` — a line printed by the hook about
what it intended in the process that ran it, not about the process now serving.

These arms cover the third leg: the check that runs in the serving process itself.

WHAT EACH ARM WOULD CATCH, so none of them is decorative:
  A  the bad HOME is repaired                        -> catches the check not running at all
  B  the repair is reported at ERROR level           -> catches a SILENT repair, which would hide
                                                        a real bootstrap defect
  C  a correct HOME is left alone and stays quiet    -> catches a check that fires always and so
                                                        says nothing about anything
  D  an unrepairable HOME is reported as such        -> catches the "repaired" line being printed
                                                        when the repair did not happen
  E  the legacy no-app-dir path is untouched         -> catches breaking single-machine deployments
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pytest

from bioengine._app.mixin import _ensure_working_directory


class _Recorder(logging.Logger):
    """A logger that records, so an arm can assert the LEVEL and not merely the text."""

    def __init__(self):
        super().__init__("rec")
        self.records = []

    def handle(self, record):  # pragma: no cover - not used
        self.records.append(record)

    def error(self, msg, *args, **kwargs):
        self.records.append(("ERROR", str(msg) % args if args else str(msg)))

    def info(self, msg, *args, **kwargs):
        self.records.append(("INFO", str(msg) % args if args else str(msg)))

    def warning(self, msg, *args, **kwargs):
        self.records.append(("WARNING", str(msg) % args if args else str(msg)))


@pytest.fixture(autouse=True)
def _no_tmpdir_leak():
    """Put TMPDIR *and* tempfile's cache back, whatever an arm or the code under test did.

    The function under test writes ``os.environ["TMPDIR"]`` directly and clears
    ``tempfile.tempdir``; neither goes through ``monkeypatch``, so neither is undone by it.
    Found by review: an arm that never touches TMPDIR was reading a PREVIOUS arm's
    ``ws-app/tmp`` as its "before" value, inside the same xdist worker.

    Deliberately NOT ``monkeypatch.delenv("TMPDIR", raising=False)``, which is the obvious
    form and does not work: ``delenv`` on a variable that is ABSENT registers nothing to undo,
    so whatever runs afterwards still leaks. That is measured — the identical fixture shape
    passed locally and failed in CI on an earlier row, because the bug is conditional on the
    variable starting unset, which is the usual case here.

    Autouse and WITHOUT a monkeypatch dependency, so it is set up first and therefore torn
    down last: after monkeypatch has replayed its own undo, not before it.

    The invariant is "unchanged", not "absent": a session that legitimately has TMPDIR set must
    not be disturbed either.
    """
    tmpdir_before = os.environ.get("TMPDIR")
    cache_before = tempfile.tempdir
    try:
        yield
    finally:
        if tmpdir_before is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = tmpdir_before
        tempfile.tempdir = cache_before


@pytest.fixture
def env(tmp_path, monkeypatch):
    app_dir = tmp_path / "ws-app"
    (app_dir / "home").mkdir(parents=True)
    monkeypatch.setenv("BIOENGINE_APP_DIR", str(app_dir))
    monkeypatch.chdir(tmp_path)
    return app_dir


def _errors(log):
    return [m for lvl, m in log.records if lvl == "ERROR"]


def test_a_bad_home_is_repaired_to_the_per_app_home(env, monkeypatch):
    monkeypatch.setenv("HOME", "/home/ray")
    log = _Recorder()
    _ensure_working_directory(log)
    assert (
        Path(os.environ["HOME"]).resolve() == (env / "home").resolve()
    ), f"HOME was left at {os.environ['HOME']!r}"


def test_b_the_repair_is_reported_at_error_level(env, monkeypatch):
    monkeypatch.setenv("HOME", "/home/ray")
    log = _Recorder()
    _ensure_working_directory(log)
    errs = _errors(log)
    assert errs, "a silent repair hides a real bootstrap defect"
    joined = " ".join(errs)
    assert "/home/ray" in joined, "the message must name what HOME actually was"
    assert str(env / "home") in joined, "the message must name what it should have been"


def test_c_a_correct_home_is_left_alone_and_says_nothing(env, monkeypatch):
    """Silence requires BOTH variables already correct: TMPDIR is part of the same per-app
    contract, so an arm that sets only HOME legitimately sees a TMPDIR repair and is not a test
    of 'says nothing'.

    It also requires ``tempfile``'s cache to agree. This arm used to set only the variables and
    leave the cache at whatever the test session had warmed — which is a process that WOULD
    write to the old directory, and since the detection step started asking ``tempfile``
    (review's R4-2) that is correctly reported rather than passed over. Clearing the cache here
    is what makes this arm a healthy replica instead of the stale-cache case its own sibling
    arm now pins.
    """
    good = env / "home"
    (env / "tmp").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(good))
    monkeypatch.setenv("TMPDIR", str(env / "tmp"))
    tempfile.tempdir = None  # resolve from TMPDIR, i.e. nothing stale to report
    log = _Recorder()
    _ensure_working_directory(log)
    assert Path(os.environ["HOME"]).resolve() == good.resolve()
    assert not _errors(log), (
        f"a healthy replica must stay quiet, got: {_errors(log)!r} — a check that fires "
        "always tells you nothing"
    )


def _fail_mkdir_for(monkeypatch, name):
    """Make Path.mkdir raise only for the directory called *name*."""
    real_mkdir = Path.mkdir

    def boom(self, *a, **k):
        if self.name == name:
            raise OSError(30, "Read-only file system")
        return real_mkdir(self, *a, **k)

    monkeypatch.setattr(Path, "mkdir", boom)


def test_d_an_unrepairable_home_is_reported_as_unrepairable(env, monkeypatch):
    """The 'repaired' line must not be printed when the repair did not happen."""
    monkeypatch.setenv("HOME", "/home/ray")
    _fail_mkdir_for(monkeypatch, "home")
    log = _Recorder()
    _ensure_working_directory(log)
    errs = " ".join(_errors(log))
    assert (
        "could NOT repair HOME" in errs
    ), f"an unrepairable HOME must say so, got: {errs!r}"
    assert (
        "HOME repaired to" not in errs
    ), "it must not claim a repair that did not happen"
    assert (
        os.environ["HOME"] == "/home/ray"
    ), "HOME must be left alone when it cannot be repaired"


def test_f_a_tmpdir_failure_does_not_slander_a_home_that_was_repaired(env, monkeypatch):
    """Found by review. HOME and TMPDIR are repaired independently and reported separately.

    The first version of this code assigned HOME, then created `tmp`; when the `tmp` creation
    failed it logged "could NOT repair HOME" while HOME had in fact been repaired, and TMPDIR had
    been pointed at a directory that did not exist. The message contradicted the state.
    """
    monkeypatch.setenv("HOME", "/home/ray")
    monkeypatch.delenv("TMPDIR", raising=False)
    _fail_mkdir_for(monkeypatch, "tmp")
    log = _Recorder()
    _ensure_working_directory(log)
    errs = " ".join(_errors(log))
    assert (
        Path(os.environ["HOME"]).resolve() == (env / "home").resolve()
    ), "HOME should be repaired"
    assert (
        "HOME repaired to" in errs
    ), "a HOME that WAS repaired must be reported as repaired"
    assert (
        "could NOT repair HOME" not in errs
    ), f"a tmp failure must not be reported as a HOME failure, got: {errs!r}"
    assert "could NOT repair TMPDIR" in errs, "the tmp failure itself must be reported"
    # And TMPDIR must not be left naming a directory that does not exist.
    tmp_env = os.environ.get("TMPDIR")
    assert (
        tmp_env is None or Path(tmp_env).exists()
    ), f"TMPDIR={tmp_env!r} names a directory that does not exist"


def test_g_a_bad_preexisting_tmpdir_is_repaired_not_left_alone(env, monkeypatch):
    """Found by review: `setdefault` would leave a present-but-unusable TMPDIR in place, and
    TMPDIR is part of the same per-app contract as HOME."""
    monkeypatch.setenv("HOME", str(env / "home"))
    monkeypatch.setenv("TMPDIR", "/nonexistent-ro")
    log = _Recorder()
    _ensure_working_directory(log)
    assert (
        Path(os.environ["TMPDIR"]).resolve() == (env / "tmp").resolve()
    ), f"TMPDIR was left at {os.environ['TMPDIR']!r}"
    assert "TMPDIR repaired to" in " ".join(_errors(log))


def test_h_a_bootstrap_that_fails_before_home_is_set_is_loud_and_then_repaired(
    env, monkeypatch, capsys
):
    """The path DONE WHEN 4 actually names, exercised rather than simulated.

    Earlier arms set HOME=/home/ray by hand. This one drives the real sequence: the import-time
    bootstrap finder calls `setup_replica_environment`, that call raises BEFORE assigning HOME (the
    ordering that exists at worker 0.16.130, where HOME is set after the source sync), the finder
    swallows the exception and prints its loud line, and the replica then starts up with HOME still
    at the image default — where this module's check must catch and repair it.
    """
    import bioengine

    monkeypatch.setenv("HOME", "/home/ray")

    def raises_before_setting_home():
        raise OSError(37, "No locks available")

    monkeypatch.setattr(
        "bioengine._app.replica_init.setup_replica_environment",
        raises_before_setting_home,
    )
    # The finder's loud branch: it must name HOME rather than failing silently.
    # The finder installs only when BIOENGINE_APP_DIR is set, at `import bioengine` time — which
    # already happened before this fixture ran. So install it here rather than skipping: a skipped
    # arm is not evidence, and this clause is specifically about that code path.
    import sys as _sys

    bioengine._install_replica_bootstrap_finder()
    installed = [
        f for f in _sys.meta_path if type(f).__name__ == "_BioEngineImportBootstrap"
    ]
    assert (
        installed
    ), "the bootstrap finder did not install even with BIOENGINE_APP_DIR set"
    finder = installed[-1]
    finder._fired = False
    finder.find_spec("some_user_module_that_does_not_exist", None, None)
    out = capsys.readouterr().out
    assert (
        "replica source bootstrap failed" in out
    ), f"the swallowed bootstrap failure must be loud, got: {out[:300]!r}"
    assert "/home/ray" in out, "the loud line must name the HOME it was left with"
    _sys.meta_path.remove(finder)

    # And the serving process still repairs itself.
    log = _Recorder()
    _ensure_working_directory(log)
    assert Path(os.environ["HOME"]).resolve() == (env / "home").resolve()
    assert "HOME repaired to" in " ".join(_errors(log))


def test_e_the_legacy_no_app_dir_path_is_untouched(tmp_path, monkeypatch):
    """Single-machine deployments pass no app dir; the worker injects HOME and this must keep
    honouring it rather than inventing a per-app path."""
    monkeypatch.delenv("BIOENGINE_APP_DIR", raising=False)
    legacy = tmp_path / "legacy-home"
    legacy.mkdir()
    monkeypatch.setenv("HOME", str(legacy))
    monkeypatch.chdir(tmp_path)
    log = _Recorder()
    _ensure_working_directory(log)
    assert Path(os.environ["HOME"]).resolve() == legacy.resolve()
    assert not _errors(log), f"the legacy path must not warn, got: {_errors(log)!r}"


def test_i_a_repair_that_does_not_take_is_reported_as_a_failure(env, monkeypatch):
    """The lead's requirement: verify the repair, do not assume it.

    If assigning the variable does not actually change what the process sees, saying "repaired"
    would put us back exactly where this function started — a replica that believes its HOME is
    fine and fails on the first write. The ERROR must name the expected and the observed value.
    """
    monkeypatch.setenv("HOME", "/home/ray")
    # Simulate a repair that does not take: Path.home() keeps answering the old value.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/ray")))
    log = _Recorder()
    _ensure_working_directory(log)
    errs = " ".join(_errors(log))
    assert (
        "could NOT repair HOME" in errs
    ), f"a repair that did not take must be reported: {errs!r}"
    assert (
        "HOME repaired to" not in errs
    ), "it must not claim a repair that did not take"
    assert (
        "/home/ray" in errs and str(env / "home") in errs
    ), "the message must name both the observed and the expected value"


def test_j_a_tmpdir_repair_is_verified_through_tempfile_not_its_own_write(
    env, monkeypatch
):
    """Found by review (N1): the re-read must ask what the process will USE.

    ``tempfile`` resolves its directory once, on first use, and caches it in
    ``tempfile.tempdir``. Every ``mkstemp``/``mkdtemp``/``NamedTemporaryFile`` afterwards goes
    there regardless of what the environment says — and any library that touched a temporary
    file at import time has already triggered that resolution, so the cache being warm is the
    ordinary case in a replica, not a corner.

    Re-reading ``os.environ`` reads back the value written one line earlier and therefore
    always agrees. Measured by review before this fix: the function logged "TMPDIR repaired to
    <app>/tmp" while ``tempfile.gettempdir()`` stayed ``/tmp`` and ``mkdtemp()`` returned
    ``/tmp/tmp…``. A reported repair that did not happen is the exact failure the verify step
    was added to prevent, so the verify step had to be able to see it.

    This arm WARMS the cache first, which is what makes it bite.
    """
    monkeypatch.setenv("HOME", str(env / "home"))
    monkeypatch.setenv("TMPDIR", "/tmp")
    tempfile.tempdir = None
    warmed = tempfile.gettempdir()
    assert (
        warmed == "/tmp"
    ), f"premise failed: the cache should start at /tmp, got {warmed!r}"

    log = _Recorder()
    _ensure_working_directory(log)

    effective = Path(tempfile.gettempdir())
    assert effective.resolve() == (env / "tmp").resolve(), (
        f"tempfile still hands out {effective}, so every temporary file this replica creates "
        f"lands outside the per-app tmp while the log says: {_errors(log)!r}"
    )
    # The consequence itself, not just the reading that predicts it.
    made = Path(tempfile.mkdtemp())
    try:
        assert made.resolve().is_relative_to(
            (env / "tmp").resolve()
        ), f"mkdtemp created {made}, outside the per-app tmp"
    finally:
        made.rmdir()
    assert "TMPDIR repaired to" in " ".join(_errors(log))


def test_k_a_repair_that_did_not_take_is_not_called_unchanged(env, monkeypatch):
    """Found by review (N2): the failure message has to be TRUE about what changed.

    When the re-read disagrees, the variable WAS written — something else did not follow. The
    old message said "HOME is unchanged", and review measured the opposite with ``Path.home``
    pinned: ``os.environ["HOME"]`` held the new value. The other failure mode, a directory that
    could not be created, leaves the variable genuinely untouched, so one fixed sentence could
    not be true for both.
    """
    monkeypatch.setenv("HOME", "/home/ray")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/ray")))
    log = _Recorder()
    _ensure_working_directory(log)
    errs = " ".join(_errors(log))

    assert "could NOT repair HOME" in errs
    assert "unchanged" not in errs, (
        f"the variable WAS changed; calling it unchanged sends the reader after the wrong "
        f"thing. message: {errs!r}"
    )
    assert os.environ["HOME"] == str(
        env / "home"
    ), "premise: the assignment really did happen, which is why 'unchanged' was false"
    assert "/home/ray" in errs and str(env / "home") in errs


def test_l_an_uncreatable_directory_IS_reported_as_unchanged(env, monkeypatch):
    """The companion of the arm above, and the reason it is not just a word filter.

    Here the variable really is untouched: the assignment sits after the ``mkdir``, so a
    failure to create leaves the old value in place and the reader should be told so. Without
    this arm, deleting the word "unchanged" everywhere would satisfy the previous arm while
    making THIS message less informative.
    """
    monkeypatch.setenv("HOME", "/home/ray")
    _fail_mkdir_for(monkeypatch, "home")
    log = _Recorder()
    _ensure_working_directory(log)
    errs = " ".join(_errors(log))

    assert "could NOT repair HOME" in errs
    assert "unchanged" in errs, (
        f"a directory that could not be created leaves HOME untouched, and the message should "
        f"say so: {errs!r}"
    )
    assert os.environ["HOME"] == "/home/ray", "premise: the assignment never ran"


def test_m_an_unwritable_target_is_refused_because_tempfile_is_what_is_ASKED(
    env, monkeypatch
):
    """Found by review (R4-1): the previous arm pinned the cache CLEAR, not the READ.

    Changing the re-read back to ``Path(os.environ[var])`` while keeping the clear left every
    arm green, because clearing alone makes the repair take in the ordinary case. So the arm
    has to use a case where the two answers DIFFER.

    This is that case, and it is reachable: the per-app ``tmp`` exists but is not writable.
    ``tempfile`` tests its candidates by actually writing, so it SKIPS an unwritable directory
    and falls back to ``/tmp`` — while ``os.environ["TMPDIR"]`` happily reads back the value
    just assigned. Reading the variable therefore reports "TMPDIR repaired to <app>/tmp" for a
    process whose every temporary file still lands in ``/tmp``: the false success this whole
    verify step exists to prevent.
    """
    monkeypatch.setenv("HOME", str(env / "home"))
    monkeypatch.setenv("TMPDIR", "/nonexistent-ro")
    target = env / "tmp"
    target.mkdir(parents=True, exist_ok=True)
    target.chmod(0o555)  # exists, and this process cannot write in it
    try:
        log = _Recorder()
        _ensure_working_directory(log)
        errs = " ".join(_errors(log))

        assert Path(tempfile.gettempdir()).resolve() != target.resolve(), (
            "premise failed: tempfile accepted an unwritable directory, so this arm is not in "
            f"the case it is about (gettempdir={tempfile.gettempdir()!r})"
        )
        assert "could NOT repair TMPDIR" in errs, (
            "the repair did NOT take — tempfile skips an unwritable candidate — and saying it "
            f"did is the false success this step exists to catch. errors: {errs!r}"
        )
        assert (
            "TMPDIR repaired to" not in errs
        ), f"a repair that did not happen was reported as done: {errs!r}"
        assert (
            str(Path(tempfile.gettempdir())) in errs
        ), f"the message must name what the process will actually use: {errs!r}"
    finally:
        target.chmod(0o755)


def test_n_a_correct_variable_with_a_STALE_cache_is_still_reported(env, monkeypatch):
    """Found by review (R4-2): the DETECTION step read the variable, so this case was silent.

    On the hook-only path ``replica_init`` assigns ``TMPDIR`` in-process, and if anything has
    already resolved a temporary directory, ``tempfile`` keeps handing out the old one. The
    variable then looks perfect, the old early-return matched, and nothing was logged at all —
    while every temporary file still went to ``/tmp``. The docstring claims this function
    covers "a future delivery path failing silently"; this was that path, failing silently.
    """
    monkeypatch.setenv("HOME", str(env / "home"))
    target = env / "tmp"
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TMPDIR", str(target))  # the variable is CORRECT
    tempfile.tempdir = "/tmp"  # ...and the cache is stale, as a warm process's would be
    assert tempfile.gettempdir() == "/tmp", "premise: the cache must start stale"

    log = _Recorder()
    _ensure_working_directory(log)
    errs = " ".join(_errors(log))

    assert Path(tempfile.gettempdir()).resolve() == target.resolve(), (
        f"the stale cache was never refreshed, so the replica still writes to "
        f"{tempfile.gettempdir()!r} despite a correct TMPDIR"
    )
    assert "TMPDIR" in errs, (
        "a replica that would write to the wrong directory was told nothing, because the "
        f"check compared the variable instead of what the process uses. errors: {errs!r}"
    )
    # The message must name BOTH values, because here they DISAGREE: the variable is correct
    # and the resolved directory is not. Naming only the variable produced "TMPDIR was
    # <target>, not <target>" — a sentence true about nothing, which sends the reader to look
    # at an environment variable that is already right.
    assert (
        str(target) in errs and "/tmp" in errs
    ), f"the message names only one of the two disagreeing values: {errs!r}"
    assert (
        "would resolve to" in errs
    ), f"the message does not say that the resolved value is the problem: {errs!r}"


def test_o_a_fully_healthy_process_is_still_silent(env, monkeypatch):
    """The control: when the variable AND what the process will use are both right, say nothing.

    A check that fires always tells you nothing, and both arms below assert a report, so
    without this one "report everything" would satisfy them.
    """
    good_home = env / "home"
    good_tmp = env / "tmp"
    good_tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(good_home))
    monkeypatch.setenv("TMPDIR", str(good_tmp))
    tempfile.tempdir = str(good_tmp)

    log = _Recorder()
    _ensure_working_directory(log)

    assert not _errors(
        log
    ), f"a healthy replica must say nothing, got: {_errors(log)!r}"
    assert tempfile.tempdir == str(good_tmp), (
        "the cache was cleared on a process that needed no repair, which discards a value an "
        f"app may have set on purpose (now {tempfile.tempdir!r})"
    )


def test_p_a_DELIBERATE_cache_pointing_elsewhere_is_reported_not_silently_dropped(
    env, monkeypatch
):
    """Enumerating the (variable, cache) states turned this up, and it decides a design choice.

    TMPDIR is correct and ``tempfile.tempdir`` points somewhere else — an app's own choice, or
    a cache warmed before the variable was set. The process WILL write to that other place, so
    this replica is not healthy and the per-app contract is not being met.

    Detection asks ``tempfile`` without clearing, so the mismatch is seen and REPORTED, then
    repaired. Clearing the cache at detection instead would re-resolve from the (correct)
    variable, conclude all is well, and discard the app's directory with no message at all —
    the same silence R4-2 was about, one level along.
    """
    good_tmp = env / "tmp"
    good_tmp.mkdir(parents=True, exist_ok=True)
    elsewhere = env / "deliberate"
    elsewhere.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(env / "home"))
    monkeypatch.setenv("TMPDIR", str(good_tmp))  # the variable is RIGHT
    tempfile.tempdir = str(elsewhere)  # ...and the process will use somewhere else

    log = _Recorder()
    _ensure_working_directory(log)
    errs = " ".join(_errors(log))

    assert "TMPDIR" in errs, (
        "the process would write outside the per-app directory and was told nothing; a silent "
        f"correction here is indistinguishable from a silent LOSS of the app's own choice"
    )
    assert (
        Path(tempfile.gettempdir()).resolve() == good_tmp.resolve()
    ), f"after the repair the process still uses {tempfile.gettempdir()!r}"


def test_q_a_WRONG_variable_masked_by_a_correct_cache_is_still_repaired(
    env, monkeypatch
):
    """The other blind spot the enumeration found, and it is mine, not review's.

    Checking only what the process will USE — review's R4-2 prescription taken literally —
    goes silent when TMPDIR is wrong but ``tempfile``'s cache happens to hold the right
    directory. This process looks fine; but ``os.environ["TMPDIR"]`` is what every CHILD
    process and every library reading the environment inherits, so it stays wrong for them.

    So the check requires both, and neither half subsumes the other.
    """
    good_tmp = env / "tmp"
    good_tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(env / "home"))
    monkeypatch.setenv("TMPDIR", "/nonexistent-wrong")  # the variable is WRONG
    tempfile.tempdir = str(good_tmp)  # ...and the cache masks it

    log = _Recorder()
    _ensure_working_directory(log)
    errs = " ".join(_errors(log))

    assert "TMPDIR" in errs, (
        "a wrong TMPDIR was passed over because the cache happened to be right, so every "
        f"child process inherits it. errors: {errs!r}"
    )
    assert (
        Path(os.environ["TMPDIR"]).resolve() == good_tmp.resolve()
    ), f"TMPDIR was left at {os.environ['TMPDIR']!r}, which children would inherit"
