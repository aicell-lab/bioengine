"""Pin that a failed ``serve.delete`` does not leave an app running untracked.

``_undeploy_application`` drops the tracking entry *before* calling
``serve.delete``, deliberately, so a monitor tick cannot re-adopt the app
mid-teardown. The delete exception was then caught, logged, and execution
carried on to log "Undeployment … completed" — so a delete that failed was
reported as a success and the app kept running with no tracking entry.

Measured consequence on the deNBI cluster, 2026-10-23: ``serve.delete``
timed out after 60 s, the app stayed RUNNING, and because
``get_app_status`` enumerates ``_deployed_applications`` it became invisible
to the worker API while still holding a GPU. ``stop_app`` could not reach it
either — its guard refuses any application_id absent from that same dict —
so there was no API route left to release the GPU. It took a Serve-level
delete from the Ray head to recover the hardware.

The fix restores the entry when the delete fails, with ``auto_redeploy``
forced off so the re-adoption race the pop exists to prevent stays
prevented.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps.manager import AppsManager

APP_ID = "model-finetune-metrics-check"
ENTRY = "CellposeRuntime"


def _make_manager(delete_raises: bool) -> AppsManager:
    is_deployed = asyncio.Event()
    is_deployed.set()

    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.undeploy")
    manager._redeploy_backoff = {APP_ID: {"consecutive_failures": 1}}
    manager._missing_from_status = {APP_ID: 2}
    manager._deleted_pending_redeploy = {APP_ID}
    manager._identity_warned = {APP_ID}
    manager._deployed_applications = {
        APP_ID: {
            "is_deployed": is_deployed,
            "artifact_id": "bioimage-io/model-finetune",
            "version": "0.16.1",
            "auto_redeploy": True,
            "entry_deployment_name": ENTRY,
            "built_app": SimpleNamespace(
                spec={"entry_id": f"deployment:{ENTRY}", "classes": {}}
            ),
            "deployment_task": None,
        }
    }

    ray_cluster = MagicMock()
    ray_cluster.check_connection = AsyncMock()
    if delete_raises:
        ray_cluster.call_with_reconnect = AsyncMock(
            side_effect=RuntimeError(
                "Some of these applications weren't deleted after 60s: "
                f"['{APP_ID}']"
            )
        )
    else:
        ray_cluster.call_with_reconnect = AsyncMock(return_value=None)
    ray_cluster.proxy_actor_handle.clear_application_replicas.remote = AsyncMock()
    manager.ray_cluster = ray_cluster

    manager._cancel_deployment_process = AsyncMock()
    manager._deploy_application = AsyncMock()

    return manager


def _messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_a_successful_delete_still_drops_all_tracking(caplog) -> None:
    # The control. Without it, a fix that simply never pops would pass every
    # assertion below while leaking a tracking entry on the happy path.
    manager = _make_manager(delete_raises=False)

    with caplog.at_level(logging.INFO, logger="test.undeploy"):
        await manager._undeploy_application(application_id=APP_ID, user_id="u")

    assert manager._deployed_applications == {}
    assert manager._redeploy_backoff == {}
    assert manager._missing_from_status == {}
    assert manager._deleted_pending_redeploy == set()
    assert manager._identity_warned == set()
    assert any("completed" in m for m in _messages(caplog))


@pytest.mark.asyncio
async def test_a_failed_delete_keeps_the_app_tracked() -> None:
    manager = _make_manager(delete_raises=True)

    await manager._undeploy_application(application_id=APP_ID, user_id="u")

    # This dict is exactly what get_app_status enumerates and exactly what
    # stop_app's guard tests, so presence here is what makes the still-running
    # app both visible and stoppable again.
    assert APP_ID in manager._deployed_applications


@pytest.mark.asyncio
async def test_a_failed_delete_is_not_reported_as_completed(caplog) -> None:
    manager = _make_manager(delete_raises=True)

    with caplog.at_level(logging.INFO, logger="test.undeploy"):
        await manager._undeploy_application(application_id=APP_ID, user_id="u")

    messages = _messages(caplog)
    assert not any("completed" in m for m in messages), (
        "A delete that failed must not log the success line — that is what "
        "made the deNBI leak invisible in the log for 33 hours."
    )
    assert any("FAILED" in m for m in messages)


@pytest.mark.asyncio
async def test_the_restored_app_is_not_auto_redeployed() -> None:
    # The pop-before-delete exists to stop a monitor tick re-adopting the app
    # mid-teardown. Restoring the entry must not reopen that race.
    manager = _make_manager(delete_raises=True)

    await manager._undeploy_application(application_id=APP_ID, user_id="u")
    assert manager._deployed_applications[APP_ID]["auto_redeploy"] is False

    await manager.monitor_applications()

    manager._deploy_application.assert_not_awaited()
    manager.ray_cluster.check_connection.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_second_stop_attempt_can_still_delete_it() -> None:
    # Recovery has to be reachable through the API, not only from the Ray head.
    manager = _make_manager(delete_raises=True)
    await manager._undeploy_application(application_id=APP_ID, user_id="u")
    assert APP_ID in manager._deployed_applications

    manager.ray_cluster.call_with_reconnect = AsyncMock(return_value=None)
    await manager._undeploy_application(application_id=APP_ID, user_id="u")

    assert manager._deployed_applications == {}
