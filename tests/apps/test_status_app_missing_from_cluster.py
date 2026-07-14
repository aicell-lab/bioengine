"""Pin the "tracked but gone from the cluster" status contract.

When the Ray head restarts it wipes every Ray Serve application record,
but the worker process keeps running and keeps the app in its in-memory
reference dict (``AppsManager._deployed_applications``). Such an app is
present in the reference dict yet absent from ``serve.status()`` /
``get_serve_instance_details()``.

The contract: it MUST still surface in ``get_app_status`` as ``UNHEALTHY``
(with a message pointing at the missing Ray Serve record) rather than be
dropped from the status output — otherwise a head restart makes deployed
apps silently vanish from the dashboard.

Two pins:
- ``_get_app_status`` returns a full ``UNHEALTHY`` status for a tracked
  app that is missing from ``instance_details`` (behavioral).
- ``get_app_status`` builds its work-list from ``_deployed_applications``,
  not from the Ray Serve response, so a missing app is still queried
  (source inspection — the public method is ``@schema_method``-wrapped and
  composes Ray + Hypha state that is impractical to reconstruct here).
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps.manager import AppsManager


def _make_app_info(*, is_deployed: bool) -> dict:
    """A fully-populated reference-dict entry for one deployed app."""
    deployed_event = asyncio.Event()
    if is_deployed:
        deployed_event.set()
    return {
        "is_deployed": deployed_event,
        "display_name": "Cellpose Finetuning",
        "description": "Finetune Cellpose models.",
        "artifact_id": "bioimage-io/cellpose-finetuning",
        "version": "1.0.1",
        "recovered_app": False,
        "application_kwargs": {},
        "application_env_vars": {},
        "disable_gpu": False,
        "application_resources": {},
        "authorized_users": ["*"],
        "available_methods": ["train"],
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


def _make_manager(app_id: str, app_info: dict) -> AppsManager:
    """An AppsManager wired with only what ``_get_app_status`` touches."""
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test")
    manager._deployed_applications = {app_id: app_info}

    ray_cluster = MagicMock()
    # No ProxyDeployment replicas exist once the head wiped Serve state;
    # get_deployment_replicas returns an empty mapping (not an error).
    ray_cluster.proxy_actor_handle.get_deployment_replicas.remote = AsyncMock(
        return_value={}
    )
    manager.ray_cluster = ray_cluster
    return manager


@pytest.mark.asyncio
async def test_tracked_app_missing_from_cluster_reports_unhealthy() -> None:
    app_id = "cellpose-finetuning"
    app_info = _make_app_info(is_deployed=True)
    manager = _make_manager(app_id, app_info)

    # instance_details has no record of the app — the head wiped Serve state.
    status = await manager._get_app_status(
        application_id=app_id,
        instance_details={"applications": {}},
        n_previous_replica=0,
        logs_tail=30,
    )

    assert status["status"] == "UNHEALTHY"
    assert "not found in Ray Serve status" in status["message"]

    # A full status payload is returned (not the simplified NOT_RUNNING
    # dict), so the app is still shown with its metadata rather than dropped.
    assert status["artifact_id"] == "bioimage-io/cellpose-finetuning"
    assert status["display_name"] == "Cellpose Finetuning"
    assert status["deployments"] == {}


@pytest.mark.asyncio
async def test_tracked_app_never_deployed_reports_not_started() -> None:
    # The counterpart branch: absent from the cluster because it was never
    # deployed (is_deployed not set) must read NOT_STARTED, not UNHEALTHY.
    app_id = "cellpose-finetuning"
    app_info = _make_app_info(is_deployed=False)
    manager = _make_manager(app_id, app_info)

    status = await manager._get_app_status(
        application_id=app_id,
        instance_details={"applications": {}},
        n_previous_replica=0,
        logs_tail=30,
    )

    assert status["status"] == "NOT_STARTED"


def test_get_app_status_iterates_reference_dict_not_cluster() -> None:
    # The work-list must come from the reference dict so an app that dropped
    # out of Ray Serve is still queried (and reported UNHEALTHY above),
    # rather than being iterated out of existence from the Serve response.
    src = inspect.getsource(AppsManager.get_app_status)
    assert "list(self._deployed_applications.keys())" in src, (
        "get_app_status must build its work-list from _deployed_applications; "
        "iterating the Ray Serve response instead would drop apps that the "
        "head restart wiped from the cluster."
    )
