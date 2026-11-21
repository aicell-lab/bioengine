"""The placement reading must reach the caller, not just exist as a helper.

``test_placement_shortfall.py`` pins what the helper computes. That is the easy
half, and on its own it is exactly the gap that let an earlier field in this same
status payload ship correct, correctly named, publicly declared and always empty:
neutering the call site left the whole suite green.

So this drives ``_get_deployment_status`` itself and reads the two keys off its
output. Both directions are asserted -- a deployment that cannot be placed reports
it, and a cold wake-up does not -- because an arm that only ever looks for the true
case would also pass against a projection that hardcoded ``True``.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps.manager import AppsManager


def _slow_allocation(required, available):
    """Serve's PENDING_ALLOCATION message, assembled as Serve assembles it."""
    return (
        "Deployment 'EntryApp' in application 'demo' has 1 replicas that have taken "
        "more than 30s to be scheduled. This may be due to waiting for the cluster "
        "to auto-scale or for a runtime environment to be installed. "
        f"Resources required for each replica: {json.dumps(required)}, "
        f"total resources available: {json.dumps(available)}. "
        "Use `ray status` for more details."
    )


GIB = 1024**3

# One 4-CPU, 1-GPU node. Both fixtures below are read against it.
NODES = {"n1": {"total_cpu": 4.0, "total_gpu": 1.0, "total_memory": 16 * GIB}}

# A 3-CPU replica waking on that node. Serve has already granted its lease, so the
# message reports only 1.0 free -- against the replica's own 3.0. It fits the node
# and it serves; comparing against the message's own figure called it stuck.
PLACEABLE_WAKE_UP = _slow_allocation({"CPU": 3.0}, {"CPU": 1.0})

# A requirement no node in this cluster could ever hold.
UNPLACEABLE = _slow_allocation({"CPU": 5.0}, {"CPU": 4.0})


def _make_manager() -> AppsManager:
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.placement_shortfall_delivery")
    manager.ray_cluster = MagicMock()
    manager.ray_cluster.status = {"nodes": NODES}
    manager.ray_cluster.proxy_actor_handle.get_deployment_logs.remote = AsyncMock(
        return_value={}
    )
    return manager


def _details(message: str):
    return {
        "deployments": {
            "EntryApp": {
                "status": "UPSCALING",
                "message": message,
                "replicas": [],
            }
        }
    }


async def _deployment(message: str):
    manager = _make_manager()
    out = await manager._get_deployment_status(
        application_id="demo",
        application_details=_details(message),
        n_previous_replica=0,
        logs_tail=0,
    )
    return out["EntryApp"]


@pytest.mark.asyncio
async def test_an_unplaceable_deployment_is_reported_through_get_app_status():
    dep = await _deployment(UNPLACEABLE)

    assert dep["unschedulable"] is True
    assert dep["placement_shortfall"] == {"CPU": {"required": 5.0, "largest_node": 4.0}}


@pytest.mark.asyncio
async def test_a_cold_wake_up_is_not():
    # The direction that makes the arm above a pin rather than a tautology, and the
    # direction the phrase-matching attempt got wrong.
    dep = await _deployment(PLACEABLE_WAKE_UP)

    assert dep["unschedulable"] is False
    assert dep["placement_shortfall"] == {}


@pytest.mark.asyncio
async def test_the_status_a_caller_cross_references_today_is_identical_in_both():
    """Why the new keys are needed at all.

    ``status`` and ``replica_states`` -- the two fields a caller has today -- cannot
    separate these cases. If this ever stops being true the row's premise has
    changed and this file should be revisited, so it is asserted rather than
    assumed.
    """
    stuck = await _deployment(UNPLACEABLE)
    waking = await _deployment(PLACEABLE_WAKE_UP)

    assert stuck["status"] == waking["status"] == "UPSCALING"
    assert stuck["replica_states"] == waking["replica_states"]


@pytest.mark.asyncio
async def test_both_keys_are_present_when_there_is_no_message_at_all():
    # An absent key and a false one read the same through ``.get()``.
    dep = await _deployment("")

    assert dep["unschedulable"] is False
    assert dep["placement_shortfall"] == {}
