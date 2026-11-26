"""A replica's HOME must not depend on the setup hook having run, or having succeeded.

THE DEFECT THIS PINS. HOME and TMPDIR used to be set in exactly one place:
``replica_init.setup_replica_environment()``, and only *after* that function had
materialised the user's source over the network. Two independent things then had to go
right before a replica had a writable HOME:

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

Either way HOME stayed at the image's ``/home/ray``, which is read-only in the published
worker image, and the app's first write to its own store died with
``OSError: [Errno 30] Read-only file system`` — a symptom two steps removed from its
cause. Measured on a production worker: of four replicas of one app created within 0.5 s,
one logged ``replica environment ready``, one logged ``replica source bootstrap failed``,
and one logged neither; the latter two both hit the read-only error.

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

import os
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
