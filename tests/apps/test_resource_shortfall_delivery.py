"""A computed shortfall must reach the caller: compute → store → publish (#0384).

``tests/test_external_cluster_shortfall.py`` pins what ``_check_resources``
*returns*, and that ``"resource_shortfall"`` is in ``PUBLIC_APP_STATUS_FIELDS``.
So the report's production is covered and its name is covered. Its **delivery**
was covered by nothing: measured on PR #309 head ``9b962117``, neutering either
the deploy record (``manager.py:3610``) or the status surface
(``manager.py:1628``) left the full suite at 1288 passed, 0 failed. The field
could be present, correctly named, publicly declared, and always empty.

Two things shape the assertions here.

The report's **contents** are compared, not its truthiness. A field arriving as
``{}``, ``True`` or a ``MagicMock`` satisfies ``is not None`` while telling a
caller nothing, and that exact stub mistake already cost this branch a cycle --
a bare ``AsyncMock()`` returns a ``MagicMock``, so the deployed record held a
Mock as its shortfall and no null assertion over the field could hold.

The store and publish steps are pinned **separately**, so a failure names which
hop broke, and the end-to-end case is pinned as well because two green hops do
not prove a connected path. The no-shortfall case is the control: without it,
every assertion below would also pass against an implementation that hard-coded
the report, and this file would read identically whether delivery worked or not.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps.manager import AppsManager

WORKSPACE = "bioimage-io"
APP_ID = "nuclei-seg"
ARTIFACT_ID = f"{WORKSPACE}/{APP_ID}"
VERSION = "0.2.4"
ENTRY_ID = "deployment:NucleiSeg"
ENTRY = "NucleiSeg"
WORKER_CLIENT_ID = "worker-1"
CONTEXT = {"user": {"id": "u-1", "email": "u@lab.test"}}

REQUESTED = {"num_cpus": 1, "num_gpus": 1, "memory": 4 * 1024**3, "VRAM_MB": 8000}

# The shape _check_resources returns for a cluster with no room, per
# tests/test_external_cluster_shortfall.py. Distinct values per field so a
# partial copy cannot pass as a whole one.
SHORTFALL = {
    "requested": REQUESTED,
    "free_per_node": {"n1": {"num_cpus": 7, "num_gpus": 0, "memory": 2 * 1024**3}},
    "cluster_free": {"num_cpus": 7, "num_gpus": 0, "memory": 2 * 1024**3},
    "detected_at": 1_700_000_123.5,
}

# A second, differently-valued report, used to prove the publish hop reads the
# stored record rather than recomputing or echoing the deploy-time argument.
STORED_LATER = {
    "requested": {"num_gpus": 4},
    "free_per_node": {"n2": {"num_gpus": 1}},
    "cluster_free": {"num_gpus": 1},
    "detected_at": 1_700_000_999.0,
}


def _instance_details() -> dict:
    return {
        "applications": {
            APP_ID: {
                "status": "RUNNING",
                "message": "",
                "deployments": {
                    ENTRY: {
                        "status": "HEALTHY",
                        "replicas": [{"replica_id": "r1", "state": "RUNNING"}],
                    }
                },
            }
        }
    }


def _built_app() -> SimpleNamespace:
    return SimpleNamespace(
        metadata={
            "name": "Nuclei Seg",
            "description": "Segment nuclei.",
            "version": VERSION,
            "source_signature": "0123456789abcdef",
            "application_kwargs": {},
            "application_env_vars": {},
            "resources": REQUESTED,
            "authorized_users": {"*": ["*"]},
            "available_methods": ["segment"],
            "frontend_entry": "frontend/index.html",
            "deployed_by_worker_client_id": WORKER_CLIENT_ID,
            "proxy_service_token_issued_at": 1_700_000_000.0,
            "proxy_service_token_ttl_seconds": 3600,
        },
        spec={"entry_id": ENTRY_ID, "classes": {ENTRY_ID: {"qualname": ENTRY}}},
        scaling={},
    )


async def _manager_that_deployed(shortfall: dict | None) -> AppsManager:
    """Drive the real ``deploy_app`` with ``_check_resources`` returning ``shortfall``."""
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.shortfall_delivery")
    # A named admin rather than "*": get_app_status projects identity-carrying
    # fields away for a caller it cannot authorize, and a wildcard authorizes
    # nobody, which would drop fields from the published status.
    manager.admin_users = [CONTEXT["user"]["email"]]
    manager.startup_applications = []

    server = MagicMock()
    server.config.workspace = WORKSPACE
    server.config.client_id = WORKER_CLIENT_ID
    server.config.public_base_url = "https://hypha.test"
    manager.server = server

    ray_cluster = MagicMock()
    ray_cluster.check_connection = AsyncMock()
    ray_cluster.call_with_reconnect = AsyncMock()
    proxy = ray_cluster.proxy_actor_handle
    proxy.get_serve_instance_details.remote = AsyncMock(
        return_value=_instance_details()
    )
    proxy.get_deployment_replicas.remote = AsyncMock(return_value=["r0"])
    proxy.get_service_registration.remote = AsyncMock(return_value=True)
    proxy.get_replica_identities.remote = AsyncMock(
        return_value={ENTRY: {"r1": {"version": VERSION, "artifact_id": ARTIFACT_ID}}}
    )
    proxy.get_deployment_logs.remote = AsyncMock(return_value={})
    manager.ray_cluster = ray_cluster

    manager._deployed_applications = {}
    manager._deployment_lock = asyncio.Lock()
    manager._check_initialized = lambda: None
    manager.artifact_manager = MagicMock()
    manager.app_builder = SimpleNamespace(build=AsyncMock(return_value=_built_app()))
    # return_value=, not a bare AsyncMock: an unset AsyncMock returns a MagicMock,
    # which is not None, so the record would hold a Mock and both the assertion
    # and its control would pass for the wrong reason.
    manager._check_resources = AsyncMock(return_value=shortfall)
    manager._cancel_deployment_process = AsyncMock()
    manager._deploy_application = AsyncMock()

    await manager.deploy_app(
        artifact_id=ARTIFACT_ID,
        application_id=APP_ID,
        version=VERSION,
        context=CONTEXT,
    )
    await asyncio.gather(
        manager._deployed_applications[APP_ID]["deployment_task"],
        return_exceptions=True,
    )
    return manager


async def _published(manager: AppsManager) -> dict:
    status = await manager.get_app_status(application_ids=[APP_ID], context=CONTEXT)
    return status[APP_ID]


@pytest.mark.asyncio
async def test_a_deploy_records_the_shortfall_check_resources_returned() -> None:
    """Hop 1, store: what the check computed reaches the worker's own record."""
    manager = await _manager_that_deployed(SHORTFALL)

    record = manager._deployed_applications[APP_ID]
    assert "resource_shortfall" in record
    assert record["resource_shortfall"] == SHORTFALL, (
        "deploy_app dropped or altered the report _check_resources returned, so "
        "nothing downstream can report that the cluster had no room. Got: "
        f"{record.get('resource_shortfall')!r}."
    )


@pytest.mark.asyncio
async def test_get_app_status_publishes_the_recorded_shortfall() -> None:
    """Hop 2, publish: the status surface reads the stored record.

    The stored value is replaced with a second, differently-valued report after
    the deploy, so this case stays red when only the publish hop is broken and
    green when only the store hop is -- it is the store case's complement, not a
    duplicate of it.
    """
    manager = await _manager_that_deployed(SHORTFALL)
    manager._deployed_applications[APP_ID]["resource_shortfall"] = STORED_LATER

    published = await _published(manager)
    assert published["resource_shortfall"] == STORED_LATER, (
        "get_app_status did not publish the shortfall on the worker's record, so "
        "the only surface a caller has for an unplaceable app reads empty. Got: "
        f"{published.get('resource_shortfall')!r}."
    )


@pytest.mark.asyncio
async def test_a_shortfall_reaches_a_caller_end_to_end() -> None:
    """The row's deliverable in one assertion: compute → store → publish.

    Pinned separately from the two hops because a path can break between two
    steps that each pass in isolation.
    """
    manager = await _manager_that_deployed(SHORTFALL)

    published = await _published(manager)
    arrived = published["resource_shortfall"]
    assert arrived == SHORTFALL, f"shortfall did not reach the caller: {arrived!r}"
    # Field-wise too: an equality that passed on a report rebuilt from different
    # numbers would be a different answer under the same key.
    assert arrived["cluster_free"]["num_gpus"] == 0
    assert arrived["requested"] == REQUESTED
    assert arrived["detected_at"] == SHORTFALL["detected_at"]


@pytest.mark.asyncio
async def test_a_deploy_with_room_publishes_no_shortfall() -> None:
    """The control. Without it every case above would pass against an
    implementation that hard-coded the report, and this file would read
    identically whether delivery worked or not."""
    manager = await _manager_that_deployed(None)

    assert manager._deployed_applications[APP_ID]["resource_shortfall"] is None
    published = await _published(manager)
    assert "resource_shortfall" in published
    assert published["resource_shortfall"] is None
