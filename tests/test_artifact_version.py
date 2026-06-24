"""
Targeted tests for ``upload_app`` artifact version handling.

The worker enforces strict version monotonicity on upload: the manifest
``version`` must be strictly greater than every existing tag on the artifact.
This file verifies four cases:

1. New artifact → committed under the manifest version tag.
2. Bumping the version → both tags are independently readable.
3. Re-saving the same version → rejected (would overwrite a published release).
4. Saving an older version → rejected.

Run with:
    conda activate bioengine
    source .env
    pytest tests/test_artifact_version.py -v
"""

import os

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from hypha_rpc import connect_to_server
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")


# ── helpers ────────────────────────────────────────────────────────────────────

MANIFEST_TMPL = """\
name: Version Test App
id: {artifact_id}
id_emoji: "🧪"
description: "Temporary app used to test artifact version commits"
type: ray-serve
format_version: 0.6.0
version: {version}
entry: test_dep:TestDep
authors:
  - {{name: "Test"}}
license: MIT
authorized_users:
  - "*"
"""

DEPLOYMENT_SRC = """\
import bioengine


@bioengine.app(ray_actor_options={"num_cpus": 0, "num_gpus": 0})
class TestDep:
    pass
"""


def _make_files(artifact_id: str, version: str):
    return [
        {
            "name": "manifest.yaml",
            "content": MANIFEST_TMPL.format(artifact_id=artifact_id, version=version),
            "type": "text",
        },
        {"name": "test_dep.py", "content": DEPLOYMENT_SRC, "type": "text"},
    ]


# ── fixtures ────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="module")
async def hypha_client():
    token = os.environ.get("BIOIMAGE_IO_TOKEN") or os.environ.get("HYPHA_TOKEN")
    assert token, "No Hypha token found in environment"
    client = await connect_to_server(
        {"server_url": "https://hypha.aicell.io", "token": token}
    )
    yield client
    await client.disconnect()


@pytest_asyncio.fixture(scope="module")
async def worker(hypha_client):
    return await hypha_client.get_service("bioimage-io/bioengine-worker")


@pytest_asyncio.fixture(scope="module")
async def artifact_manager(hypha_client):
    return await hypha_client.get_service("public/artifact-manager")


# ── tests ───────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_app_commits_manifest_version(worker, artifact_manager):
    """upload_app must commit the artifact under the version in manifest.yaml."""
    artifact_alias = "version-test-app-pytest"
    artifact_id = f"bioimage-io/{artifact_alias}"
    test_version = "2.3.4"

    try:
        await artifact_manager.delete(artifact_id)
    except Exception:
        pass

    try:
        saved_id = await worker.upload_app(files=_make_files(artifact_alias, test_version))
        assert saved_id == artifact_id, f"Unexpected artifact ID: {saved_id}"

        artifact = await artifact_manager.read(artifact_id=artifact_id, version=test_version)
        assert artifact is not None
        assert artifact.manifest.get("version") == test_version
    finally:
        try:
            await artifact_manager.delete(artifact_id)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_upload_app_new_version_creates_isolated_snapshot(worker, artifact_manager):
    """Bumping the version creates a new isolated snapshot; old tag still readable."""
    artifact_alias = "version-bump-test-pytest"
    artifact_id = f"bioimage-io/{artifact_alias}"
    v1, v2 = "1.0.0", "1.1.0"

    try:
        await artifact_manager.delete(artifact_id)
    except Exception:
        pass

    try:
        await worker.upload_app(files=_make_files(artifact_alias, v1))
        a1 = await artifact_manager.read(artifact_id=artifact_id, version=v1)
        assert a1.manifest.get("version") == v1

        await worker.upload_app(files=_make_files(artifact_alias, v2))
        a2 = await artifact_manager.read(artifact_id=artifact_id, version=v2)
        assert a2.manifest.get("version") == v2

        a1_after = await artifact_manager.read(artifact_id=artifact_id, version=v1)
        assert a1_after.manifest.get("version") == v1
    finally:
        try:
            await artifact_manager.delete(artifact_id)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_upload_app_resave_same_version_rejected(worker, artifact_manager):
    """Re-saving the same version must be rejected (would overwrite a release)."""
    artifact_alias = "version-resave-pytest"
    artifact_id = f"bioimage-io/{artifact_alias}"
    version = "1.0.0"

    try:
        await artifact_manager.delete(artifact_id)
    except Exception:
        pass

    try:
        await worker.upload_app(files=_make_files(artifact_alias, version))
        with pytest.raises(Exception, match=r"strictly greater|already exists"):
            await worker.upload_app(files=_make_files(artifact_alias, version))
    finally:
        try:
            await artifact_manager.delete(artifact_id)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_upload_app_older_version_rejected(worker, artifact_manager):
    """Saving a lower version than the highest existing tag must be rejected."""
    artifact_alias = "version-older-pytest"
    artifact_id = f"bioimage-io/{artifact_alias}"
    v1, v2 = "1.0.0", "1.1.0"

    try:
        await artifact_manager.delete(artifact_id)
    except Exception:
        pass

    try:
        await worker.upload_app(files=_make_files(artifact_alias, v1))
        await worker.upload_app(files=_make_files(artifact_alias, v2))

        with pytest.raises(Exception, match=r"strictly greater|already exists"):
            await worker.upload_app(files=_make_files(artifact_alias, v1))
    finally:
        try:
            await artifact_manager.delete(artifact_id)
        except Exception:
            pass


pytest_plugins = ("pytest_asyncio",)
