"""Tests for what ``create_file_list_from_directory`` refuses to ship.

A committed artifact version is immutable and is exactly what a deploy runs, so a
git-ignored local leftover swept into the file list is permanent and does not
appear in git. ``bioimage-io/model-finetune`` 0.27.3 shipped four
``.pytest_cache/`` files that way while 0.27.1 was clean, which is #0085.

Most of these exercise the pure helper against real temporary directories, so
they need no Hypha server. The last one is live-gated and goes the whole way to
a committed version, because the helper excluding a path and the *artifact* not
carrying it are two different claims. It creates its scratch artifact in the
test user's own workspace rather than in ``bioimage-io``, and deletes it again.

Run the live one with:
    source .env && pytest tests/test_artifact_file_list_exclusions.py --live
"""

import subprocess

import pytest

from bioengine.utils.artifact_utils import (
    create_application_from_files,
    create_file_list_from_directory,
)


def _names(directory):
    return {f["name"] for f in create_file_list_from_directory(directory)}


def _git_init(directory, gitignore=None):
    """A real work tree, since the helper shells out to ``git check-ignore``."""
    subprocess.run(["git", "init", "-q"], cwd=directory, check=True)
    if gitignore is not None:
        (directory / ".gitignore").write_text(gitignore)


def _app(parent):
    app = parent / "my-app"
    app.mkdir(parents=True)
    (app / "manifest.yaml").write_text("id: my-app\nversion: 1.0.0\n")
    (app / "entry.py").write_text("class EntryApp: ...\n")
    return app


@pytest.fixture
def loose_app(tmp_path):
    """An app directory with no work tree above it — the Ray-actor staging case."""
    return _app(tmp_path / "staging")


@pytest.fixture
def repo_app(tmp_path):
    """``apps/my-app`` inside a checkout, which is how this repo is laid out: the
    ``.git`` directory and ``.gitignore`` sit above the uploaded directory."""

    def _make(gitignore):
        _git_init(tmp_path, gitignore)
        return _app(tmp_path / "apps")

    return _make


def test_tracked_files_are_included(repo_app):
    app = repo_app(".pytest_cache/\n")
    assert _names(app) == {"entry.py", "manifest.yaml"}


def test_gitignored_directory_is_excluded(repo_app):
    """The 0.27.3 case: pytest leaves a cache dir behind and the upload ships it."""
    app = repo_app(".pytest_cache/\n")
    cache = app / ".pytest_cache" / "v" / "cache"
    cache.mkdir(parents=True)
    (cache / "nodeids").write_text("[]")
    (app / ".pytest_cache" / "CACHEDIR.TAG").write_text("Signature: 8a477f597d28d172")

    names = _names(app)
    assert not [n for n in names if n.startswith(".pytest_cache")]
    assert "entry.py" in names, "the fix must not drop real app content"


def test_gitignored_env_file_is_excluded(repo_app):
    """`.env` is git-ignored in this repo, so the leak path is the same mechanism."""
    app = repo_app(".env\n")
    (app / ".env").write_text("HYPHA_TOKEN=not-a-real-token\n")
    assert ".env" not in _names(app)


def test_nested_gitignored_file_is_excluded(repo_app):
    """Exclusion is by path, so a rule matching deeper than the top level counts."""
    app = repo_app("*.pt\n")
    weights = app / "model_cache"
    weights.mkdir()
    (weights / "keep.json").write_text("{}")
    (weights / "local.pt").write_bytes(b"\x80\x02")

    names = _names(app)
    assert "model_cache/keep.json" in names
    assert "model_cache/local.pt" not in names


def test_dot_git_is_excluded_when_the_directory_is_a_repository_root(tmp_path):
    """``git check-ignore`` never reports ``.git`` as ignored, so the git filter alone
    would publish ``.git/config`` — and a remote URL there can carry a token."""
    app = _app(tmp_path)
    _git_init(app)
    (app / ".git" / "config").write_text("[remote \"origin\"]\n")

    names = _names(app)
    assert not [n for n in names if n.startswith(".git/")]
    assert names == {"entry.py", "manifest.yaml"}


def test_pycache_excluded_without_a_work_tree(loose_app):
    """No work tree: the Ray-actor caller runs against a runtime_env staging dir,
    where git cannot answer. ``__pycache__`` must still be dropped on that path."""
    pycache = loose_app / "__pycache__"
    pycache.mkdir()
    (pycache / "entry.cpython-310.pyc").write_bytes(b"\x00\x01\x02")

    names = _names(loose_app)
    assert not [n for n in names if "__pycache__" in n]
    assert names == {"entry.py", "manifest.yaml"}


def test_without_a_work_tree_ordinary_files_still_ship(loose_app):
    """The discrimination control for the test above: absent a work tree the helper
    must fall back to shipping everything except ``__pycache__``, NOT to shipping
    nothing. A filter that silently emptied the bundle would pass the assertions
    above for the wrong reason."""
    (loose_app / "scratch.txt").write_text("not ignored by anything")
    assert _names(loose_app) == {"entry.py", "manifest.yaml", "scratch.txt"}


@pytest.mark.asyncio
async def test_a_committed_version_does_not_carry_a_gitignored_file(
    tmp_path, hypha_client, artifact_manager
):
    """The clause the unit tests above cannot settle: that the *artifact* is clean.

    The scratch artifact is created in the connected user's own workspace, not in
    ``bioimage-io``, and is deleted on the way out. The staged file is asserted to
    be git-ignored before the upload, so a silent no-op in the staging step cannot
    make the absence assertion pass for the wrong reason.
    """
    _git_init(tmp_path, ".pytest_cache/\n")
    alias = "gitignore-exclusion-test-pytest"
    app = _app(tmp_path / "apps")
    (app / "manifest.yaml").write_text(
        "name: Gitignore Exclusion Test\n"
        f"id: {alias}\n"
        'id_emoji: "🧪"\n'
        'description: "Temporary app used to test gitignored-file exclusion"\n'
        "type: ray-serve\n"
        "format_version: 0.6.0\n"
        "version: 1.0.0\n"
        "entry: entry:EntryApp\n"
        "authors:\n"
        "  - {name: \"Test\"}\n"
        "license: MIT\n"
        "authorized_users:\n"
        '  - "*"\n'
    )
    (app / "entry.py").write_text(
        "import bioengine\n\n\n"
        '@bioengine.app(ray_actor_options={"num_cpus": 0, "num_gpus": 0})\n'
        "class EntryApp:\n    pass\n"
    )
    cache = app / ".pytest_cache"
    cache.mkdir()
    (cache / "CACHEDIR.TAG").write_text("Signature: 8a477f597d28d172")

    staged = subprocess.run(
        ["git", "-C", str(app), "check-ignore", ".pytest_cache/CACHEDIR.TAG"],
        capture_output=True,
        text=True,
    )
    assert staged.returncode == 0, "the staging step did nothing: git does not ignore it"

    files = create_file_list_from_directory(app)
    workspace = hypha_client.config.workspace
    artifact_id = f"{workspace}/{alias}"

    try:
        await artifact_manager.delete(artifact_id)
    except Exception:
        pass

    try:
        saved_id = await create_application_from_files(
            artifact_manager=artifact_manager,
            files=files,
            workspace=workspace,
            user_id=hypha_client.config.user["id"],
        )
        committed = await artifact_manager.list_files(
            artifact_id=saved_id, version="1.0.0"
        )
        names = {f["name"] for f in committed}
        assert ".pytest_cache" not in names
        assert {"manifest.yaml", "entry.py"} <= names
    finally:
        try:
            await artifact_manager.delete(artifact_id)
        except Exception:
            pass
