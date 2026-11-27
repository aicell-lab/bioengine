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
    assert Path(os.environ["HOME"]).resolve() == (env / "home").resolve(), (
        f"HOME was left at {os.environ['HOME']!r}"
    )


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
    of 'says nothing'."""
    good = env / "home"
    (env / "tmp").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(good))
    monkeypatch.setenv("TMPDIR", str(env / "tmp"))
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
    assert "could NOT repair HOME" in errs, f"an unrepairable HOME must say so, got: {errs!r}"
    assert "HOME repaired to" not in errs, "it must not claim a repair that did not happen"
    assert os.environ["HOME"] == "/home/ray", "HOME must be left alone when it cannot be repaired"


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
    assert Path(os.environ["HOME"]).resolve() == (env / "home").resolve(), "HOME should be repaired"
    assert "HOME repaired to" in errs, "a HOME that WAS repaired must be reported as repaired"
    assert "could NOT repair HOME" not in errs, (
        f"a tmp failure must not be reported as a HOME failure, got: {errs!r}"
    )
    assert "could NOT repair TMPDIR" in errs, "the tmp failure itself must be reported"
    # And TMPDIR must not be left naming a directory that does not exist.
    tmp_env = os.environ.get("TMPDIR")
    assert tmp_env is None or Path(tmp_env).exists(), (
        f"TMPDIR={tmp_env!r} names a directory that does not exist"
    )


def test_g_a_bad_preexisting_tmpdir_is_repaired_not_left_alone(env, monkeypatch):
    """Found by review: `setdefault` would leave a present-but-unusable TMPDIR in place, and
    TMPDIR is part of the same per-app contract as HOME."""
    monkeypatch.setenv("HOME", str(env / "home"))
    monkeypatch.setenv("TMPDIR", "/nonexistent-ro")
    log = _Recorder()
    _ensure_working_directory(log)
    assert Path(os.environ["TMPDIR"]).resolve() == (env / "tmp").resolve(), (
        f"TMPDIR was left at {os.environ['TMPDIR']!r}"
    )
    assert "TMPDIR repaired to" in " ".join(_errors(log))


def test_h_a_bootstrap_that_fails_before_home_is_set_is_loud_and_then_repaired(env, monkeypatch, capsys):
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
    installed = [f for f in _sys.meta_path
                 if type(f).__name__ == "_BioEngineImportBootstrap"]
    assert installed, "the bootstrap finder did not install even with BIOENGINE_APP_DIR set"
    finder = installed[-1]
    finder._fired = False
    finder.find_spec("some_user_module_that_does_not_exist", None, None)
    out = capsys.readouterr().out
    assert "replica source bootstrap failed" in out, (
        f"the swallowed bootstrap failure must be loud, got: {out[:300]!r}"
    )
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
    assert "could NOT repair HOME" in errs, f"a repair that did not take must be reported: {errs!r}"
    assert "HOME repaired to" not in errs, "it must not claim a repair that did not take"
    assert "/home/ray" in errs and str(env / "home") in errs, (
        "the message must name both the observed and the expected value"
    )
