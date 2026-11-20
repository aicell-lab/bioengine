"""Pin that a worker with no opted-in app still notices a lost Serve controller.

``auto_redeploy`` defaults to False, and ``monitor_applications`` used to build
the opted-in list and return as soon as it was empty. Everything that reads
Serve's state — the controller-loss detection, the recovery sweep, and the
adopted-app skip line that sits behind it — came *after* that return. So a
worker tracking only apps with the default setting could not observe that the
controller had died at all; it was undetectable for that worker rather than
merely unactioned, and a head restart left nothing in the one log an operator
reads.

Detection is reported on the actor lookup alone rather than by hoisting
``serve.status``: the lookup is the question being asked, and it costs a
fraction of the status RPC. The cooldown then gates the lookup itself, so the
workers that previously made no call at all gain one per cooldown rather than
one per tick — which bounds the added rate and, more importantly, how often a
tick can block on an unreachable head. No recovery follows — an app that asked
not to be redeployed still is not.

The two paths are mutually exclusive by construction: this branch runs only
when the opted-in list is empty, which is precisely when the recovery sweep is
not reached. That is what stops one controller-loss episode producing both this
line and the sweep's own.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bioengine.apps import manager as manager_module
from bioengine.apps.manager import (
    _CONTROLLER_RECOVERY_COOLDOWN_SECONDS,
    AppsManager,
)

OPTED_OUT_ID = "model-finetune-soup"
OPTED_OUT_ARTIFACT = "bioimage-io/model-finetune"
OPTED_IN_ID = "model-runner"
OPTED_IN_ARTIFACT = "bioimage-io/model-runner"

# The sentence this row is about, as the operator would grep for it.
DETECTION_MARK = "none with auto_redeploy enabled"


def _app(artifact_id: str, auto_redeploy: bool) -> dict:
    is_deployed = asyncio.Event()
    is_deployed.set()
    return {
        "is_deployed": is_deployed,
        "artifact_id": artifact_id,
        "version": "1.0.0",
        "auto_redeploy": auto_redeploy,
        "entry_deployment_name": "Entry",
        "built_app": SimpleNamespace(spec={"entry_id": "deployment:Entry", "classes": {}}),
        "deployment_task": None,
    }


def _make_manager(apps: dict) -> AppsManager:
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.controller_loss_detection")
    manager._redeploy_backoff = {}
    manager._missing_from_status = {}
    manager._deleted_pending_redeploy = set()
    manager._last_controller_recovery = 0.0
    manager._deployed_applications = apps
    manager._renew_proxy_tokens = AsyncMock()
    manager._deploy_application = AsyncMock()
    # Touching the cluster at all on the opted-out path is a failure: the whole
    # point is that the probe answers without a serve.status round trip.
    manager.ray_cluster = SimpleNamespace(
        check_connection=AsyncMock(side_effect=AssertionError("cluster touched")),
        call_with_reconnect=AsyncMock(side_effect=AssertionError("serve.status called")),
    )
    return manager


def _probe(monkeypatch, missing: bool) -> list[int]:
    """Patch the controller probe, counting how often it is asked."""
    calls: list[int] = []

    async def _fake() -> bool:
        calls.append(1)
        return missing

    monkeypatch.setattr(manager_module, "_controller_is_missing_off_loop", _fake)
    return calls


def _at(caplog, level: int) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == level]


@pytest.mark.asyncio
async def test_a_lost_controller_is_reported_with_no_opted_in_app(
    caplog, monkeypatch
) -> None:
    manager = _make_manager({OPTED_OUT_ID: _app(OPTED_OUT_ARTIFACT, False)})
    _probe(monkeypatch, missing=True)

    with caplog.at_level(logging.DEBUG):
        await manager.monitor_applications()

    errors = _at(caplog, logging.ERROR)
    assert len(errors) == 1, f"expected exactly one detection line, got {errors}"
    assert "Ray Serve controller lost" in errors[0]
    # Naming what it knows is the DONE WHEN, so the app has to be in the line.
    assert OPTED_OUT_ID in errors[0]
    assert DETECTION_MARK in errors[0]
    # Detection is not redeployment.
    manager._deploy_application.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_present_controller_stays_silent(caplog, monkeypatch) -> None:
    manager = _make_manager({OPTED_OUT_ID: _app(OPTED_OUT_ARTIFACT, False)})
    calls = _probe(monkeypatch, missing=False)

    with caplog.at_level(logging.DEBUG):
        await manager.monitor_applications()

    assert len(calls) == 1, "the probe is the one added call per tick"
    assert caplog.records == [], f"a healthy tick must say nothing: {caplog.records}"


@pytest.mark.asyncio
async def test_a_worker_tracking_nothing_does_not_even_probe(
    caplog, monkeypatch
) -> None:
    """The added cost is zero for a worker with no apps, which is most of them."""
    manager = _make_manager({})
    calls = _probe(monkeypatch, missing=True)

    with caplog.at_level(logging.DEBUG):
        await manager.monitor_applications()

    assert calls == [], "nothing is tracked, so there is nothing to report about"
    assert caplog.records == []


@pytest.mark.asyncio
async def test_the_cooldown_bounds_how_often_detection_speaks(
    caplog, monkeypatch
) -> None:
    manager = _make_manager({OPTED_OUT_ID: _app(OPTED_OUT_ARTIFACT, False)})
    calls = _probe(monkeypatch, missing=True)

    with caplog.at_level(logging.DEBUG):
        for _ in range(5):
            await manager.monitor_applications()

    assert len(_at(caplog, logging.ERROR)) == 1, (
        "five ticks inside one cooldown must produce one line, not five — at the "
        "default 10s interval an unbounded branch would emit 360 lines an hour"
    )
    # The cooldown gates the probe, not just the line. That matters beyond noise:
    # the lookup blocks a thread for up to the lookup timeout against an
    # unreachable head, so a per-tick probe would put that stall on a path that
    # previously had none.
    assert len(calls) == 1, f"the probe itself must be gated too, got {len(calls)}"
    assert _CONTROLLER_RECOVERY_COOLDOWN_SECONDS > 0


@pytest.mark.asyncio
async def test_one_opted_in_app_sends_the_episode_down_the_sweep_instead(
    caplog, monkeypatch
) -> None:
    """Mutual exclusion: the two paths cannot both report one episode."""
    manager = _make_manager(
        {
            OPTED_OUT_ID: _app(OPTED_OUT_ARTIFACT, False),
            OPTED_IN_ID: _app(OPTED_IN_ARTIFACT, True),
        }
    )
    _probe(monkeypatch, missing=True)
    # With one opted-in app the real path is taken, which does touch the
    # cluster; let it get as far as serve.status and fail there.
    boom = Exception("Failed to submit task to actor: Can't find actor")
    manager.ray_cluster = SimpleNamespace(
        check_connection=AsyncMock(),
        call_with_reconnect=AsyncMock(side_effect=boom),
    )
    manager._recover_from_controller_loss = AsyncMock(return_value=True)

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception):
            await manager.monitor_applications()

    manager._recover_from_controller_loss.assert_awaited_once()
    assert all(DETECTION_MARK not in m for m in _at(caplog, logging.ERROR)), (
        "the opted-out detection line must not fire when the sweep handles it"
    )
