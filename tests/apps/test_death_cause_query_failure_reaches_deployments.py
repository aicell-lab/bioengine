"""A FAILED cluster-wide dead-actor query must reach every deployment as a flag.

``get_app_status`` fetches the dead-actor map once per call. When that fetch raises,
``death_causes`` is ``None`` — but ``None`` already means "no map supplied", so it
cannot say "the query failed". A separate ``death_cause_query_failed`` flag carries
that, and without it a failed hoist silently re-queries once per deployment: the
round-4 review of the change that landed this measured **5** cluster-wide queries for
one ``get_app_status``, where the fixed version issues 1.

**What was unpinned, and why a passing suite did not notice.** The proxy-side test
passes the flag by hand, so the manager layer — its ``try``/``except``, the
``death_causes is None`` branch, and the kwarg on the ``get_deployment_logs`` call —
was exercised by nothing. Deleting
``death_cause_query_failed=death_cause_query_failed`` from that call left the suite at
669 passed, rc=0.

**Which assertion below does the pinning, stated because they are not equal.**
``death_causes_unavailable`` is set from ``_get_deployment_status``'s own parameter,
not from the call it makes, so it survives that mutation. The assertion that goes RED
is the one reading the **kwargs the mock was called with**. The status-field assertion
is kept because it is what a consumer actually sees, not because it pins the defect.
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
ENTRY = "NucleiSeg"
ENTRY_ID = "deployment:NucleiSeg"
SECOND = "NucleiPost"  # a SECOND deployment in the same app
APP_ID_2 = "nuclei-seg-b"  # and a SECOND app
ARTIFACT_ID_2 = f"{WORKSPACE}/{APP_ID_2}"
WORKER_CLIENT_ID = "worker-1"
CONTEXT = {"user": {"id": "u-1", "email": "u@lab.test"}}

REQUESTED = {"num_cpus": 1, "num_gpus": 0, "memory": 1024**3, "VRAM_MB": 0}


class _QueryBlewUp(RuntimeError):
    """A distinct type, so a pass cannot come from some other exception."""


def _one_app(entry: str, second: str) -> dict:
    return {
        "status": "RUNNING",
        "message": "",
        "deployments": {
            entry: {
                "status": "HEALTHY",
                "replicas": [{"replica_id": "r1", "state": "RUNNING"}],
            },
            second: {
                "status": "HEALTHY",
                "replicas": [{"replica_id": "r2", "state": "RUNNING"}],
            },
        },
    }


def _instance_details() -> dict:
    """TWO applications with TWO deployments each — four deployments in total.

    Deliberately not one-and-one. The dead-actor query is hoisted to once per
    ``get_app_status`` CALL, and on a single-deployment fixture ``await_count == 1``
    is satisfied equally by one-per-call, one-per-app and one-per-deployment. A
    review mutant that moved the query into the per-deployment loop left such a
    fixture entirely green — the arm looked like a guard and guaranteed nothing,
    which is the exact failure this test file exists to close.
    """
    return {
        "applications": {
            APP_ID: _one_app(ENTRY, SECOND),
            APP_ID_2: _one_app(ENTRY, SECOND),
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


async def _manager(*, query_raises: bool) -> tuple[AppsManager, AsyncMock, AsyncMock]:
    """A manager whose cluster-wide dead-actor query either raises or returns a map."""
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.death_cause_query_failure")
    # A NAMED admin, not "*": get_app_status projects identity-carrying fields away
    # for a caller it cannot authorize, and a wildcard authorizes nobody.
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

    if query_raises:
        death_query = AsyncMock(side_effect=_QueryBlewUp("the proxy RPC failed"))
    else:
        death_query = AsyncMock(return_value={})
    proxy.get_all_death_causes.remote = death_query

    logs = AsyncMock(return_value={})
    proxy.get_deployment_logs.remote = logs
    proxy.get_serve_instance_details.remote = AsyncMock(
        return_value=_instance_details()
    )
    proxy.get_deployment_replicas.remote = AsyncMock(return_value=["r0"])
    proxy.get_service_registration.remote = AsyncMock(return_value=True)
    proxy.get_replica_identities.remote = AsyncMock(
        return_value={ENTRY: {"r1": {"version": VERSION, "artifact_id": ARTIFACT_ID}}}
    )
    manager.ray_cluster = ray_cluster

    manager._deployed_applications = {}
    manager._deployment_lock = asyncio.Lock()
    manager._check_initialized = lambda: None
    manager.artifact_manager = MagicMock()
    manager.app_builder = SimpleNamespace(build=AsyncMock(return_value=_built_app()))
    manager._check_resources = AsyncMock(return_value=None)
    manager._cancel_deployment_process = AsyncMock()
    manager._deploy_application = AsyncMock()

    for artifact, app_id in ((ARTIFACT_ID, APP_ID), (ARTIFACT_ID_2, APP_ID_2)):
        await manager.deploy_app(
            artifact_id=artifact,
            application_id=app_id,
            version=VERSION,
            context=CONTEXT,
        )
        await asyncio.gather(
            manager._deployed_applications[app_id]["deployment_task"],
            return_exceptions=True,
        )
    return manager, death_query, logs


def _logs_kwargs(logs: AsyncMock) -> dict:
    assert logs.await_args_list, "get_deployment_logs was never called — vacuous"
    return logs.await_args_list[-1].kwargs


@pytest.mark.asyncio
async def test_a_failed_query_reaches_get_deployment_logs_as_the_flag() -> None:
    """THE PIN. Delete ``death_cause_query_failed=`` from the call and this fails.

    Read off the kwargs the proxy was actually called with, because that argument is
    the whole mechanism: it is what stops each deployment re-querying for itself.
    """
    manager, death_query, logs = await _manager(query_raises=True)
    status = await manager.get_app_status(application_ids=[APP_ID], context=CONTEXT)

    kwargs = _logs_kwargs(logs)
    assert kwargs.get("death_cause_query_failed") is True, (
        "a failed cluster-wide query did not reach get_deployment_logs as a flag, "
        f"so each deployment will re-query for itself (kwargs: {sorted(kwargs)})"
    )
    # None, not {}: an empty map says "the query ran and found nothing", which the
    # caller turns into one unexplained-death entry per dead replica.
    assert kwargs.get("death_causes", "<ABSENT>") is None


@pytest.mark.asyncio
async def test_a_failed_query_is_reported_in_the_status_as_unavailable() -> None:
    """What a consumer sees. NOT the pin — this survives the mutation above.

    ``death_causes_unavailable`` is set from ``_get_deployment_status``'s own
    parameter, so removing the kwarg from the call it makes leaves this green. Kept
    because it is the field a reader acts on, and labelled so the two are not
    confused.
    """
    manager, _death_query, _logs = await _manager(query_raises=True)
    status = await manager.get_app_status(application_ids=[APP_ID], context=CONTEXT)

    dep = status[APP_ID]["deployments"][ENTRY]
    assert dep.get("death_causes_unavailable") is True


@pytest.mark.asyncio
async def test_the_query_is_issued_ONCE_per_CALL_across_apps_and_deployments() -> None:
    """The hoist itself, over FOUR deployments in TWO apps.

    One query per deployment would be 4 and one per app would be 2, so ``1`` can only
    mean per call. On a one-app one-deployment fixture all three collapse to 1 and this
    arm pins nothing — a review mutant that moved the query into the per-deployment loop
    passed such a fixture.
    """
    manager, death_query, logs = await _manager(query_raises=True)
    status = await manager.get_app_status(
        application_ids=[APP_ID, APP_ID_2], context=CONTEXT
    )

    # The fixture must actually have presented four deployments; otherwise the count
    # below is being compared against a smaller number than the docstring claims.
    seen = sum(len(status[a]["deployments"]) for a in (APP_ID, APP_ID_2))
    assert (
        seen == 4
    ), f"the fixture presented {seen} deployments, not 4 — arm is vacuous"

    assert death_query.await_count == 1, (
        f"the cluster-wide query ran {death_query.await_count} times for one "
        f"get_app_status over {seen} deployments in 2 apps; the hoist exists so it "
        "runs once per call"
    )


@pytest.mark.asyncio
async def test_a_failed_query_does_not_fail_the_whole_call() -> None:
    """Guarded, not propagated: the app status still comes back."""
    manager, _death_query, _logs = await _manager(query_raises=True)
    status = await manager.get_app_status(application_ids=[APP_ID], context=CONTEXT)

    assert APP_ID in status
    assert status[APP_ID]["status"] == "RUNNING"


class TestTheSucceedingDirection:
    """The control. Without it, code that hardcoded the flag to ``True`` would pass
    every assertion above."""

    @pytest.mark.asyncio
    async def test_a_SUCCESSFUL_query_reaches_the_call_as_False(self) -> None:
        manager, death_query, logs = await _manager(query_raises=False)
        await manager.get_app_status(application_ids=[APP_ID], context=CONTEXT)

        assert death_query.await_count == 1
        kwargs = _logs_kwargs(logs)
        assert kwargs.get("death_cause_query_failed") is False
        # The map, not None — the two are the thing being distinguished.
        assert kwargs.get("death_causes") == {}

    @pytest.mark.asyncio
    async def test_a_SUCCESSFUL_query_leaves_the_status_field_OFF(self) -> None:
        # An absent key and a False value read the same through ``.get()``, so this
        # compares against a sentinel rather than testing truthiness.
        manager, _death_query, _logs = await _manager(query_raises=False)
        status = await manager.get_app_status(application_ids=[APP_ID], context=CONTEXT)

        dep = status[APP_ID]["deployments"][ENTRY]
        assert dep.get("death_causes_unavailable", "<ABSENT>") == "<ABSENT>"
