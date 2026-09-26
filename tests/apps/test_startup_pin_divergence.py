"""Pin the reporting of a startup version pin that reverts a running app.

A worker pod restart replays ``--startup-applications``, but an app rolled out
over the API only ever changed the worker's runtime state. When the pin still
carries the old version the restart quietly reinstates it: status RUNNING,
deployments HEALTHY, no diff against prior state — the app is simply running
older code. That happened on deNBI on 2026-09-01 (model-runner 2.7.2 -> 2.4.2)
and was found only because someone compared versions by hand.

``recover_deployed_applications()`` runs before ``deploy_startup_applications()``,
so the worker holds both numbers at that moment. Two pins here:

- ``deploy_startup_applications`` logs a WARNING naming both versions before it
  redeploys at the pin.
- ``get_app_status`` returns ``pinned_version`` alongside the running one, so
  the divergence is assertable over the API without reading cluster config.

Neither changes what gets deployed.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps.manager import AppsManager

RUNNING_VERSION = "2.7.2"
PINNED_VERSION = "2.4.2"
ARTIFACT_ID = "bioimage-io/model-runner"
APP_ID = "model-runner"


def _make_manager(
    *,
    startup_applications: list,
    running_version: str | None = RUNNING_VERSION,
    running_artifact_id: str = ARTIFACT_ID,
) -> AppsManager:
    """An AppsManager holding one recovered app plus the given startup pins."""
    is_deployed = asyncio.Event()
    is_deployed.set()

    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.startup_pin")
    manager.startup_applications = startup_applications
    manager._deployed_applications = {
        APP_ID: {
            "is_deployed": is_deployed,
            "display_name": "Model Runner",
            "description": "Run bioimage.io models.",
            "artifact_id": running_artifact_id,
            "version": running_version,
            "recovered_app": True,
            "application_kwargs": {},
            "application_env_vars": {},
            "disable_gpu": False,
            "application_resources": {},
            "authorized_users": ["*"],
            "available_methods": ["infer"],
            "max_ongoing_requests": 1,
            "scaling": {},
            "static_site_url": None,
            "started_at": 1_700_000_000.0,
            "last_updated_at": 1_700_000_000.0,
            "last_updated_by": "user@example.com",
            "auto_redeploy": False,
            "deployed_by_worker_client_id": "worker-abc",
            "proxy_service_token_issued_at": None,
            "proxy_service_token_ttl_seconds": None,
        }
    }

    server = MagicMock()
    server.config.workspace = "bioimage-io"
    server.generate_token = AsyncMock(return_value="startup-token")
    manager.server = server
    manager.admin_users = ["admin@example.com"]

    ray_cluster = MagicMock()
    ray_cluster.proxy_actor_handle.get_deployment_replicas.remote = AsyncMock(
        return_value={}
    )
    manager.ray_cluster = ray_cluster

    # deploy_app is the thing under observation, not under test: replace it but
    # keep the real __schema__, which deploy_startup_applications reads to
    # validate config keys.
    deploy_app = AsyncMock(return_value=APP_ID)
    deploy_app.__schema__ = AppsManager.deploy_app.__schema__
    manager.deploy_app = deploy_app

    return manager


@pytest.mark.asyncio
async def test_divergent_pin_warns_with_both_versions(caplog) -> None:
    manager = _make_manager(
        startup_applications=[
            {
                "artifact_id": ARTIFACT_ID,
                "application_id": APP_ID,
                "version": PINNED_VERSION,
            }
        ]
    )

    with caplog.at_level(logging.WARNING, logger="test.startup_pin"):
        await manager.deploy_startup_applications()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, (
        "A restart that reverts a running app to an older pin must say so. "
        "Without this line the revert is indistinguishable from a clean start."
    )
    message = warnings[0].getMessage()
    assert RUNNING_VERSION in message and PINNED_VERSION in message, message
    assert APP_ID in message

    # Observability only — the pinned version is still deployed.
    manager.deploy_app.assert_awaited_once()
    assert manager.deploy_app.await_args.kwargs["version"] == PINNED_VERSION


@pytest.mark.asyncio
async def test_matching_pin_is_silent(caplog) -> None:
    manager = _make_manager(
        startup_applications=[
            {
                "artifact_id": ARTIFACT_ID,
                "application_id": APP_ID,
                "version": RUNNING_VERSION,
            }
        ]
    )

    with caplog.at_level(logging.WARNING, logger="test.startup_pin"):
        await manager.deploy_startup_applications()

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


@pytest.mark.asyncio
async def test_pin_without_a_version_is_silent(caplog) -> None:
    # An unversioned pin inherits whatever is already running, so there is
    # nothing to revert and nothing to report.
    manager = _make_manager(
        startup_applications=[
            {"artifact_id": ARTIFACT_ID, "application_id": APP_ID},
        ]
    )

    with caplog.at_level(logging.WARNING, logger="test.startup_pin"):
        await manager.deploy_startup_applications()

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


@pytest.mark.asyncio
async def test_pin_for_a_different_artifact_is_silent(caplog) -> None:
    # Same application_id, different artifact: the versions belong to two
    # different release lines and comparing them says nothing.
    manager = _make_manager(
        startup_applications=[
            {
                "artifact_id": ARTIFACT_ID,
                "application_id": APP_ID,
                "version": PINNED_VERSION,
            }
        ],
        running_artifact_id="bioimage-io/other-app",
    )

    with caplog.at_level(logging.WARNING, logger="test.startup_pin"):
        await manager.deploy_startup_applications()

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


@pytest.mark.asyncio
async def test_status_surfaces_the_pin_next_to_the_running_version() -> None:
    manager = _make_manager(
        startup_applications=[
            {
                "artifact_id": ARTIFACT_ID,
                "application_id": APP_ID,
                "version": PINNED_VERSION,
            }
        ]
    )

    status = await manager._get_app_status(
        application_id=APP_ID,
        instance_details={"applications": {}},
        n_previous_replica=0,
        logs_tail=30,
    )

    assert status["version"] == RUNNING_VERSION
    assert status["pinned_version"] == PINNED_VERSION


@pytest.mark.asyncio
async def test_status_reports_no_pin_for_an_unpinned_app() -> None:
    manager = _make_manager(startup_applications=[])

    status = await manager._get_app_status(
        application_id=APP_ID,
        instance_details={"applications": {}},
        n_previous_replica=0,
        logs_tail=30,
    )

    assert status["pinned_version"] is None


@pytest.mark.asyncio
async def test_status_resolves_a_short_artifact_id_in_the_pin() -> None:
    # Startup configs may name the artifact without its workspace prefix; the
    # tracked app always carries the full form.
    manager = _make_manager(
        startup_applications=[
            {
                "artifact_id": "model-runner",
                "application_id": APP_ID,
                "version": PINNED_VERSION,
            }
        ]
    )

    status = await manager._get_app_status(
        application_id=APP_ID,
        instance_details={"applications": {}},
        n_previous_replica=0,
        logs_tail=30,
    )

    assert status["pinned_version"] == PINNED_VERSION
