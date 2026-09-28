"""Pin what ``deploy_app`` tells its caller it deployed.

``deploy_app`` used to return a bare application-id string. An RPC caller could
therefore not tell which version it got, and omitting ``version`` on an
already-running ``application_id`` silently redeploys that application's current
version instead of rolling forward — a deploy that reports success while
shipping nothing new.

The return is now a mapping, and the field that matters is ``version_source``:
it distinguishes "you named a version and got it" (``requested``) from "you
named none and got the artifact's newest" (``latest``) from "you named none and
got the one already running" (``inherited``). The third is the trap, and it has
to be readable *without* the caller remembering what it asked for and comparing
— a returned version nobody compares is still silent.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps.manager import AppsManager
from bioengine.utils.artifact_utils import latest_committed_version

WORKSPACE = "bioimage-io"
ARTIFACT_ID = f"{WORKSPACE}/nuclei-seg"
APP_ID = "nuclei-seg"
CONTEXT = {"user": {"id": "u-1", "email": "u@lab.test"}}

# The artifact's history: 1.0.1 is the newest committed version.
VERSIONS = [
    {"version": "1.0.0", "created_at": 1},
    {"version": "1.0.1", "created_at": 2},
]
LATEST = "1.0.1"


def _fake_app(resolved_version):
    """The minimum of ``AppBuilder.build``'s result that ``deploy_app`` reads."""
    return SimpleNamespace(
        metadata={
            "name": "Nuclei Seg",
            "description": "Segment nuclei.",
            "version": resolved_version,
            "application_kwargs": {},
            "application_env_vars": {},
            "resources": {},
            "authorized_users": {"*": ["*"]},
            "available_methods": ["segment"],
            "deployed_by_worker_client_id": "client-1",
            "proxy_service_token_issued_at": 0.0,
            "proxy_service_token_ttl_seconds": 3600,
        },
        spec={"classes": {}},
        scaling={},
    )


def _make_manager(*, running: dict | None = None) -> AppsManager:
    """An ``AppsManager`` that can be driven through ``deploy_app`` to its return.

    Everything ``deploy_app`` itself does is real — permission check, artifact-id
    qualification, the version-inheritance branch, the return. Only the things it
    *delegates to* are faked, and ``build`` resolves ``version=None`` exactly the
    way ``AppBuilder._load_manifest`` does, so a test can tell the resolved
    version apart from the argument that was passed.
    """
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.deploy_return")
    manager.admin_users = ["*"]
    manager._deployment_lock = asyncio.Lock()
    manager._check_initialized = lambda: None
    manager.server = MagicMock()
    manager.server.config.workspace = WORKSPACE

    manager.artifact_manager = MagicMock()
    manager.artifact_manager.read = AsyncMock(return_value={"versions": VERSIONS})

    manager.ray_cluster = MagicMock()
    manager.ray_cluster.check_connection = AsyncMock()
    manager.ray_cluster.call_with_reconnect = AsyncMock()

    async def _build(*, version, **_kwargs):
        resolved = version
        if resolved is None:
            resolved = latest_committed_version({"versions": VERSIONS})
        return _fake_app(resolved)

    manager.app_builder = SimpleNamespace(build=_build)
    manager._check_resources = AsyncMock()
    manager._cancel_deployment_process = AsyncMock()
    manager._deploy_application = AsyncMock()
    manager._generate_application_id = AsyncMock(return_value="wandering-cloud-1234")

    manager._deployed_applications = {}
    if running is not None:
        manager._deployed_applications[APP_ID] = {
            "started_at": 0.0,
            "version": running["version"],
            "artifact_id": ARTIFACT_ID,
            "application_kwargs": {},
            "application_env_vars": {},
            "hypha_token": "tok",
            "disable_gpu": False,
            "max_ongoing_requests": 10,
            "proxy_memory_in_gb": 0.5,
            "auto_redeploy": False,
            "debug": False,
            "scaling": {},
            "is_deployed": asyncio.Event(),
        }
    return manager


async def _deploy(manager, **kwargs):
    result = await manager.deploy_app(context=CONTEXT, **kwargs)
    task = manager._deployed_applications[result["application_id"]]["deployment_task"]
    await asyncio.gather(task, return_exceptions=True)
    return result


# ── the shape ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deploy_app_returns_a_mapping_not_a_bare_id() -> None:
    manager = _make_manager()

    result = await _deploy(
        manager,
        artifact_id=ARTIFACT_ID,
        application_id=APP_ID,
        version="1.0.0",
    )

    assert isinstance(result, dict), (
        "A bare id string is the whole defect: the caller cannot see which "
        "version it got."
    )
    assert set(result) == {
        "application_id",
        "artifact_id",
        "version",
        "version_source",
    }
    assert result["application_id"] == APP_ID


@pytest.mark.asyncio
async def test_a_short_artifact_id_comes_back_fully_qualified() -> None:
    # The worker rewrites a bare name against its own workspace, so the version
    # alone does not identify the deployed code — the pair does.
    manager = _make_manager()

    result = await _deploy(manager, artifact_id="nuclei-seg", version="1.0.0")

    assert result["artifact_id"] == ARTIFACT_ID


# ── the resolved version, and where it came from ──────────────────────────────


@pytest.mark.asyncio
async def test_an_explicit_version_is_reported_as_requested() -> None:
    manager = _make_manager()

    result = await _deploy(
        manager, artifact_id=ARTIFACT_ID, application_id=APP_ID, version="1.0.0"
    )

    assert result["version"] == "1.0.0"
    assert result["version_source"] == "requested"


@pytest.mark.asyncio
async def test_a_new_app_without_a_version_reports_the_resolved_latest() -> None:
    # The argument was None; the deployed version is not. Reporting the argument
    # back instead of what the builder resolved would hand the caller a None.
    manager = _make_manager()

    result = await _deploy(manager, artifact_id=ARTIFACT_ID, application_id=APP_ID)

    assert result["version"] == LATEST
    assert result["version_source"] == "latest"


@pytest.mark.asyncio
async def test_an_update_without_a_version_reports_the_inherited_version() -> None:
    # The case the whole change exists for: 1.0.1 is committed, 1.0.0 is
    # running, the caller asks for no version and gets 1.0.0 back. Without
    # version_source this is indistinguishable from a roll-forward.
    manager = _make_manager(running={"version": "1.0.0"})

    result = await _deploy(manager, artifact_id=ARTIFACT_ID, application_id=APP_ID)

    assert result["version"] == "1.0.0"
    assert result["version_source"] == "inherited"
    assert result["version"] != LATEST


@pytest.mark.asyncio
async def test_an_update_that_names_the_version_is_not_inherited() -> None:
    # A deliberate pin must not be reported as an inheritance, or a caller that
    # rejects "inherited" would reject every pinned deploy.
    manager = _make_manager(running={"version": "1.0.0"})

    result = await _deploy(
        manager, artifact_id=ARTIFACT_ID, application_id=APP_ID, version="1.0.0"
    )

    assert result["version"] == "1.0.0"
    assert result["version_source"] == "requested"


@pytest.mark.asyncio
async def test_an_update_of_an_unpinned_app_reports_latest_not_inherited() -> None:
    # A running app deployed as "latest" carries no version, so an update
    # without one resolves latest again and genuinely does roll forward.
    # Calling that "inherited" would be a false alarm.
    manager = _make_manager(running={"version": None})

    result = await _deploy(manager, artifact_id=ARTIFACT_ID, application_id=APP_ID)

    assert result["version"] == LATEST
    assert result["version_source"] == "latest"
