"""A deployment that can never place its replica must stop reading as transitional.

Ray applies no deadline to a replica it cannot place: it stays
``PENDING_ALLOCATION`` for as long as the cluster is full, and the only mark is
a log warning every 30 s. The deployment status it produces depends on how the
replica count was raised — ``UPSCALING`` when a request wakes an app from
``min_replicas: 0``, ``UPDATING`` on a deploy — and neither value carries
elapsed time, so no consumer of the status payload could tell a wake-up that
will finish from one that never can.

``stalled_for_s`` is that missing bound, and it is deliberately keyed on replica
start times rather than on the status string: a check keyed on the string would
cover one of the two paths and silently miss the other.

The tests that matter most here are the negative ones. Holding zero replicas is
the *correct* idle state under ``min_replicas: 0``, and a cold start on this
platform includes a ``runtime_env`` venv build and takes minutes — so a field
that cried stalled during a legitimate wake-up would be worse than the defect
it reports.
"""

from __future__ import annotations

import logging
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps.manager import (
    PUBLIC_DEPLOYMENT_FIELDS,
    _REPLICA_STALL_SECONDS,
    AppsManager,
    public_app_status,
)

stalled_for_seconds = AppsManager._stalled_for_seconds

# Comfortably past the bound, so the assertions do not depend on how long the
# test itself takes to run.
WELL_PAST = _REPLICA_STALL_SECONDS + 600.0


def replica(state: str, age_s: float | None) -> dict:
    """One projected replica entry, aged ``age_s`` seconds (None => no time)."""
    return {
        "replica_id": f"r-{state}-{age_s}",
        "state": state,
        "start_time_s": None if age_s is None else time.time() - age_s,
    }


def test_idle_at_zero_replicas_is_not_stalled():
    """``min_replicas: 0`` with nothing running is healthy idle, not a stall."""
    assert stalled_for_seconds([]) is None


def test_wake_up_in_progress_is_not_stalled():
    assert stalled_for_seconds([replica("STARTING", 60.0)]) is None


def test_venv_build_just_under_the_bound_is_not_stalled():
    """The bound must clear a cold start, venv build included."""
    assert stalled_for_seconds([replica("STARTING", _REPLICA_STALL_SECONDS - 30.0)]) is None


def test_unplaceable_replica_past_the_bound_is_reported():
    stalled = stalled_for_seconds([replica("STARTING", WELL_PAST)])
    assert stalled is not None
    assert stalled == pytest.approx(WELL_PAST, abs=30.0)


def test_a_running_replica_is_never_stalled_however_old_the_others_are():
    """Serving is serving: an old pending replica alongside a RUNNING one is an
    upscale that has not landed, not an outage."""
    assert (
        stalled_for_seconds([replica("RUNNING", 10.0), replica("STARTING", WELL_PAST)])
        is None
    )


def test_epoch_start_time_is_not_an_ancient_stall():
    """Ray initialises a replica's details with ``start_time_s=0``; subtracting
    that from now would report every fresh deploy as stalled for decades."""
    assert stalled_for_seconds([{"state": "STARTING", "start_time_s": 0}]) is None
    assert stalled_for_seconds([replica("STARTING", None)]) is None


def test_oldest_pending_replica_sets_the_duration():
    stalled = stalled_for_seconds(
        [replica("STARTING", 60.0), replica("RECOVERING", WELL_PAST)]
    )
    assert stalled == pytest.approx(WELL_PAST, abs=30.0)


def test_a_downscale_to_zero_is_not_a_stall():
    """The false positive that would fire on every idle app.

    Under ``min_replicas: 0`` an idle deployment scales to zero, and its
    replicas pass through ``STOPPING`` on the way out. ``start_time_s`` dates
    from when the replica *started serving*, so a replica that was up for hours
    arrives in ``STOPPING`` with an hours-old start time and zero RUNNING
    alongside it — the exact shape of a stall, from the healthiest thing the
    autoscaler does. Any ``get_app_status`` landing in that window would report
    it.
    """
    assert stalled_for_seconds([replica("STOPPING", WELL_PAST)]) is None
    assert (
        stalled_for_seconds([replica("STOPPING", WELL_PAST), replica("STOPPING", 5.0)])
        is None
    )


def test_a_migrating_replica_is_serving_so_nothing_is_stalled():
    """Ray groups ``PENDING_MIGRATION`` with ``RUNNING`` for routing — see
    ``_BROADCAST_STATES`` and ``get_running_replica_infos`` — so a deployment
    at zero RUNNING that still holds one is answering requests."""
    assert stalled_for_seconds([replica("PENDING_MIGRATION", WELL_PAST)]) is None
    assert (
        stalled_for_seconds(
            [replica("PENDING_MIGRATION", 10.0), replica("STARTING", WELL_PAST)]
        )
        is None
    )


def test_both_stuck_paths_are_covered_whatever_ray_called_the_status():
    """The discrimination control for keying on start times instead of status.

    A never-placed replica reads ``UPDATING`` on a deploy and ``UPSCALING`` on a
    wake-up. Both must be reported, and the verdict must not move when only the
    status string changes — otherwise this is one more check that covers one
    path and is blind to the other.
    """
    stuck = [replica("STARTING", WELL_PAST)]
    waking = [replica("STARTING", 60.0)]
    for ray_status in ("UPDATING", "UPSCALING", "HEALTHY", "DOWNSCALING"):
        # The status is what a caller sees next to the field; the field itself
        # is computed without reading it.
        assert stalled_for_seconds(stuck) is not None, ray_status
        assert stalled_for_seconds(waking) is None, ray_status


@pytest.mark.asyncio
async def test_the_status_payload_actually_carries_the_field():
    """The wiring test, without which the rest of this file is vacuous.

    Every other assertion here either calls the helper directly or inspects the
    allowlist, so all of them would still pass with the field never written into
    the payload at all — a fix attached past the only caller that matters.
    """
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.stall")
    manager.ray_cluster = MagicMock()
    manager.ray_cluster.proxy_actor_handle.get_deployment_logs.remote = AsyncMock(
        return_value={}
    )

    details = {
        "deployments": {
            "Stuck": {
                "status": "UPSCALING",
                "replicas": [
                    {"replica_id": "r-stuck", "state": "STARTING", "start_time_s": time.time() - WELL_PAST}
                ],
            },
            "Waking": {
                "status": "UPSCALING",
                "replicas": [
                    {"replica_id": "r-wake", "state": "STARTING", "start_time_s": time.time() - 60.0}
                ],
            },
        }
    }

    info = await manager._get_deployment_status(
        application_id="app", application_details=details,
        n_previous_replica=0, logs_tail=0,
    )

    assert info["Stuck"]["stalled_for_s"] == pytest.approx(WELL_PAST, abs=30.0)
    # Both deployments report the same Ray status, so this pair is the whole
    # point: the field is what separates them.
    assert info["Waking"]["status"] == info["Stuck"]["status"] == "UPSCALING"
    assert info["Waking"]["stalled_for_s"] is None


def test_the_field_survives_the_public_projection():
    """A stalled deployment is readable by a caller outside the app's roster.

    ``PUBLIC_DEPLOYMENT_FIELDS`` is an allowlist, so a field nobody added to it
    is withheld — which would leave the only steady-state consumers of this
    payload reading the same green they read before.
    """
    assert "stalled_for_s" in PUBLIC_DEPLOYMENT_FIELDS

    status = {
        "status": "RUNNING",
        "authorized_users": {"*": ["someone@example.invalid"]},
        "deployments": {
            "EntryApp": {
                "status": "UPSCALING",
                "replica_states": {"STARTING": 1},
                "stalled_for_s": 4242.0,
                "logs": {"r-1": {"stdout": "PLANTED", "stderr": ""}},
            }
        },
    }
    public = public_app_status(status)
    deployment = public["deployments"]["EntryApp"]
    assert deployment["stalled_for_s"] == 4242.0
    # Positive control: the projection really did run and really does withhold.
    assert "logs" not in deployment
    assert "authorized_users" not in public
