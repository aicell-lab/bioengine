"""A replica materialises staged source with NO download credential, or fails loudly.

This is the consumer half of taking ``BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN`` out of every Serve
replica's ``runtime_env``. The producer half — the worker's staging task, and the token travelling
to it as a task *argument* rather than in a ``runtime_env`` — is in
``tests/apps/test_replica_runtime_env_excludes_download_token.py``.

Both halves have to hold or the change is worse than no change: drop the token from the replica env
without a staged tree to read and every private-artifact replica dies; add the staged tree without
dropping the token and the credential is still in the per-node log.

WHAT MAKES THE "ALREADY STAGED" TEST NON-VACUOUS. ``_is_staged`` has to say *no* in the cases that
matter, not merely *yes* when everything is perfect, so each negative is asserted separately: no
marker, a marker naming a different version, a marker naming a different *artifact* at the same
version, and an empty source tree. The artifact case is the one a version-only check would miss —
an ``application_id`` reused across artifacts could otherwise serve the previous artifact's source.

Completeness comes from the WRITE ORDER, not from inspecting the tree: ``_sync_source_from_hypha``
writes every file and the caller writes ``.version`` only after it returns, so a matching marker
cannot coexist with a half-written tree. A test that counted files would be asserting the weaker
thing and would pass for an empty directory.

THE FAILURE PATH IS TESTED TOO, because it is the known gap. A node that joins the cluster after
the deploy has no staged source and — by design — no credential. On a shared filesystem it sees the
staged copy; on a per-node filesystem its replica must fail with a message that *names the missing
stage*, since the bare error is an HTTP 401/403 that reads like a broken token and sends the reader
hunting for a credential problem that does not exist. The replica cannot self-heal: this hook runs
before ``__init__``, which is exactly why this token is not delivered through the proxy actor the
way deploy-time secrets are.
"""

import logging

import pytest

from bioengine._app.replica_init import (
    _SOURCE_STAGED_KEY,
    _ensure_source,
    _is_staged,
)

ARTIFACT = "someworkspace/some-app"
VERSION = "1.2.3"
IDENTITY = f"{ARTIFACT}@{VERSION}"


def _stage(tmp_path, identity=IDENTITY, files=("entry.py",)):
    """Build a complete staged app_dir and return it."""
    app_dir = tmp_path / "ws-app"
    source = app_dir / "source"
    source.mkdir(parents=True)
    for name in files:
        (source / name).write_text("# staged\n")
    (app_dir / ".version").write_text(identity)
    return app_dir


# ───────────────────────────── _is_staged: the negatives first ────────────────────────────


def test_is_staged_false_without_marker(tmp_path):
    app_dir = _stage(tmp_path)
    (app_dir / ".version").unlink()
    assert _is_staged(app_dir / "source", app_dir / ".version", IDENTITY) is False


def test_is_staged_false_for_a_different_version(tmp_path):
    app_dir = _stage(tmp_path, identity=f"{ARTIFACT}@9.9.9")
    assert _is_staged(app_dir / "source", app_dir / ".version", IDENTITY) is False


def test_is_staged_false_for_a_different_artifact_at_the_same_version(tmp_path):
    """A version-only check would wrongly say yes here.

    ``application_id`` can be reused across artifacts, so identity keys on the artifact too.
    """
    app_dir = _stage(tmp_path, identity=f"otherworkspace/other-app@{VERSION}")
    assert _is_staged(app_dir / "source", app_dir / ".version", IDENTITY) is False


def test_is_staged_false_for_an_empty_source_tree(tmp_path):
    app_dir = _stage(tmp_path, files=())
    assert _is_staged(app_dir / "source", app_dir / ".version", IDENTITY) is False


def test_is_staged_false_when_source_is_missing_entirely(tmp_path):
    app_dir = _stage(tmp_path)
    for path in sorted((app_dir / "source").rglob("*"), reverse=True):
        path.unlink()
    (app_dir / "source").rmdir()
    assert _is_staged(app_dir / "source", app_dir / ".version", IDENTITY) is False


def test_is_staged_true_for_a_complete_stage(tmp_path):
    """The positive control for all five negatives above.

    Without this, every ``False`` could be coming from a function that always says no.
    """
    app_dir = _stage(tmp_path)
    assert _is_staged(app_dir / "source", app_dir / ".version", IDENTITY) is True


# ─────────────────────── _ensure_source: credential-free, and loud when not ───────────────


def test_ensure_source_returns_staged_tree_with_no_token_and_no_files_url(
    tmp_path, monkeypatch
):
    """The whole point: a replica holding NO credential still materialises.

    Both ``BIOENGINE_ARTIFACT_FILES_URL`` and ``BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN`` are removed, so
    if this returned by syncing from Hypha it could only be by reaching the network, which the
    missing URL makes impossible — the ``RuntimeError`` in that branch is what a regression here
    would raise.
    """
    app_dir = _stage(tmp_path)
    monkeypatch.setenv("BIOENGINE_ARTIFACT_ID", ARTIFACT)
    monkeypatch.setenv(_SOURCE_STAGED_KEY, "1")
    monkeypatch.delenv("BIOENGINE_ARTIFACT_FILES_URL", raising=False)
    monkeypatch.delenv("BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN", raising=False)
    monkeypatch.delenv("BIOENGINE_LOCAL_ARTIFACT_PATH", raising=False)

    source = _ensure_source(app_dir, VERSION, logging.getLogger("test"))

    assert source == app_dir / "source"
    assert (source / "entry.py").is_file()


def test_ensure_source_does_not_touch_the_network_when_staged(tmp_path, monkeypatch):
    """Asserted by making any sync attempt explode, not by inferring it from a return value.

    A ``files_url`` IS set here — which is the real replica's situation, since the URL stays in the
    replica env and only the token leaves — so "it returned the staged tree" has to mean it took the
    staged branch rather than that it had no other option.
    """
    app_dir = _stage(tmp_path)
    monkeypatch.setenv("BIOENGINE_ARTIFACT_ID", ARTIFACT)
    monkeypatch.setenv(_SOURCE_STAGED_KEY, "1")
    monkeypatch.setenv("BIOENGINE_ARTIFACT_FILES_URL", "https://example.invalid/files")
    monkeypatch.delenv("BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN", raising=False)
    monkeypatch.delenv("BIOENGINE_LOCAL_ARTIFACT_PATH", raising=False)

    import bioengine._app.replica_init as mod

    def _explode(*args, **kwargs):
        raise AssertionError("_sync_source_from_hypha must not be reached when staged")

    monkeypatch.setattr(mod, "_sync_source_from_hypha", _explode)

    assert _ensure_source(app_dir, VERSION, logging.getLogger("test")) == (
        app_dir / "source"
    )


def test_the_staged_branch_is_NOT_taken_without_the_worker_s_flag(
    tmp_path, monkeypatch
):
    """The flag is load-bearing, and this is the regression that proved it.

    Written first without a flag — "source exists and ``.version`` matches ⇒ use it" — this
    short-circuit broke three ``test_replica_init.py`` cases that assert incremental re-sync, and
    the reason generalises past the tests: a committed artifact version is immutable but a *staged*
    one is not, and a deploy at "latest" can resolve to the same version string with different
    content. An unconditional short-circuit therefore serves STALE code.

    So a process that can sync must keep syncing. Same complete stage as the test above, same
    absent token, only the flag removed — and the sync must run.
    """
    app_dir = _stage(tmp_path)
    monkeypatch.setenv("BIOENGINE_ARTIFACT_ID", ARTIFACT)
    monkeypatch.delenv(_SOURCE_STAGED_KEY, raising=False)
    monkeypatch.setenv("BIOENGINE_ARTIFACT_FILES_URL", "https://example.invalid/files")
    monkeypatch.setenv("BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN", "tok-not-a-real-token")
    monkeypatch.delenv("BIOENGINE_LOCAL_ARTIFACT_PATH", raising=False)

    called = []

    import bioengine._app.replica_init as mod

    monkeypatch.setattr(
        mod,
        "_sync_source_from_hypha",
        lambda *a, **k: called.append(True),
    )

    _ensure_source(app_dir, VERSION, logging.getLogger("test"))

    assert called == [True], (
        "the staged short-circuit fired without the worker declaring a stage; a re-staged "
        "version would be served stale"
    )


def test_ensure_source_names_the_missing_stage_when_there_is_no_credential(
    tmp_path, monkeypatch
):
    """The known gap: a node that joined after the deploy, on a per-node filesystem.

    The requirement is not merely "it raises" — a bare HTTP 401 raises too, and reads as a broken
    token. The message must point at the missing stage, so the reader does not go looking for a
    credential problem that does not exist by design.
    """
    app_dir = tmp_path / "ws-app"
    app_dir.mkdir()
    monkeypatch.setenv("BIOENGINE_ARTIFACT_ID", ARTIFACT)
    monkeypatch.setenv("BIOENGINE_ARTIFACT_FILES_URL", "https://example.invalid/files")
    monkeypatch.delenv("BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN", raising=False)
    monkeypatch.delenv("BIOENGINE_LOCAL_ARTIFACT_PATH", raising=False)

    import bioengine._app.replica_init as mod

    def _unauthorised(*args, **kwargs):
        raise OSError("HTTP Error 403: Forbidden")

    monkeypatch.setattr(mod, "_sync_source_from_hypha", _unauthorised)

    with pytest.raises(RuntimeError) as excinfo:
        _ensure_source(app_dir, VERSION, logging.getLogger("test"))

    message = str(excinfo.value)
    assert "no staged" in message
    assert "holds no download credential by design" in message
    assert "redeploy" in message
    # The underlying error is kept, so the diagnosis adds to it rather than hiding it.
    assert "403" in message
    assert isinstance(excinfo.value.__cause__, OSError)


def test_ensure_source_re_raises_unchanged_when_a_token_WAS_present(
    tmp_path, monkeypatch
):
    """The discriminating control for the test above.

    With a token in hand a sync failure is an ordinary failure and must NOT be re-labelled as a
    missing stage — otherwise the diagnosis would be printed for every network blip and would stop
    meaning anything. Same staged state, same exception, only the token differs.
    """
    app_dir = tmp_path / "ws-app"
    app_dir.mkdir()
    monkeypatch.setenv("BIOENGINE_ARTIFACT_ID", ARTIFACT)
    monkeypatch.setenv("BIOENGINE_ARTIFACT_FILES_URL", "https://example.invalid/files")
    monkeypatch.setenv("BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN", "tok-not-a-real-token")
    monkeypatch.delenv("BIOENGINE_LOCAL_ARTIFACT_PATH", raising=False)

    import bioengine._app.replica_init as mod

    def _boom(*args, **kwargs):
        raise OSError("HTTP Error 503: Service Unavailable")

    monkeypatch.setattr(mod, "_sync_source_from_hypha", _boom)

    with pytest.raises(OSError) as excinfo:
        _ensure_source(app_dir, VERSION, logging.getLogger("test"))
    assert "503" in str(excinfo.value)
    assert "no staged" not in str(excinfo.value)


def test_ensure_source_still_syncs_when_the_stage_is_for_another_version(
    tmp_path, monkeypatch
):
    """The staged branch must not pin a replica to stale source.

    A redeploy at a new version finds a marker for the OLD one, so the staged branch has to decline
    and let the sync run. This is the regression that would make an app version bump silently serve
    the previous code.
    """
    app_dir = _stage(tmp_path, identity=f"{ARTIFACT}@0.0.1")
    monkeypatch.setenv("BIOENGINE_ARTIFACT_ID", ARTIFACT)
    monkeypatch.setenv("BIOENGINE_ARTIFACT_FILES_URL", "https://example.invalid/files")
    monkeypatch.setenv("BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN", "tok-not-a-real-token")
    monkeypatch.delenv("BIOENGINE_LOCAL_ARTIFACT_PATH", raising=False)

    calls = []

    import bioengine._app.replica_init as mod

    def _record(files_url, version, token, source, snapshot_path, logger):
        calls.append({"version": version, "token": token})
        (source).mkdir(parents=True, exist_ok=True)
        (source / "entry.py").write_text("# fresh\n")

    monkeypatch.setattr(mod, "_sync_source_from_hypha", _record)

    _ensure_source(app_dir, VERSION, logging.getLogger("test"))

    assert calls == [{"version": VERSION, "token": "tok-not-a-real-token"}]
    # And the marker is advanced, so the NEXT start takes the credential-free path.
    assert (app_dir / ".version").read_text() == IDENTITY
