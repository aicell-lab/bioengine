"""Pin that controller-loss recovery says something when it gives up on an app.

A Ray head restart destroys the Serve controller, and
``_recover_from_controller_loss`` walks every tracked app to re-establish it.
An app the worker *adopted* rather than deployed carries no cached
``built_app``, so it cannot be rebuilt from here — that part is by design.

What was not by design is that it used to be dropped by a bare ``continue``
placed *above* the log call, so an adopted app was recognised, discarded, and
never mentioned. The 2026-11-13 deNBI outage destroyed an adopted app and the
worker's own logs recorded nothing at all, which is why nobody could say from
the logs what had happened or to which app.

The sibling path through ``_fire_redeploy`` already had exactly the right
message for this case, naming the app, the artifact, why it was skipped and
what to do about it. Recovery now routes through it instead of returning early.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bioengine.apps.manager import (
    _CONTROLLER_RECOVERY_COOLDOWN_SECONDS,
    AppsManager,
)

ADOPTED_ID = "model-finetune-soup"
ADOPTED_ARTIFACT = "bioimage-io/model-finetune"
BUILT_ID = "model-runner"
BUILT_ARTIFACT = "bioimage-io/model-runner"


def _app(artifact_id: str, built: bool) -> dict:
    is_deployed = asyncio.Event()
    is_deployed.set()
    return {
        "is_deployed": is_deployed,
        "artifact_id": artifact_id,
        "version": "1.0.0",
        "auto_redeploy": True,
        "entry_deployment_name": "Entry",
        # None is what recover_deployed_applications stores for every adopted
        # app; a real built_app is whatever deploy_app cached.
        "built_app": (
            SimpleNamespace(spec={"entry_id": "deployment:Entry", "classes": {}})
            if built
            else None
        ),
        "deployment_task": None,
    }


def _make_manager() -> AppsManager:
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.controller_loss")
    manager._redeploy_backoff = {}
    manager._missing_from_status = {}
    manager._deleted_pending_redeploy = set()
    manager._last_controller_recovery = 0.0
    manager._deployed_applications = {
        ADOPTED_ID: _app(ADOPTED_ARTIFACT, built=False),
        BUILT_ID: _app(BUILT_ARTIFACT, built=True),
    }
    # Let the real _fire_redeploy run — its log line is the thing under test.
    manager._deploy_application = AsyncMock()
    return manager


def _at(caplog, level: int) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == level]


@pytest.mark.asyncio
async def test_an_adopted_app_is_named_when_recovery_gives_up(caplog) -> None:
    manager = _make_manager()

    with caplog.at_level(logging.INFO, logger="test.controller_loss"):
        await manager._recover_from_controller_loss([ADOPTED_ID])
        await asyncio.sleep(0)

    warnings = _at(caplog, logging.WARNING)
    assert len(warnings) == 1, (
        "Controller-loss recovery dropped an adopted app without saying so. "
        "This is the whole defect: the loss has to be readable from the "
        f"worker's own logs afterwards. Got: {warnings!r}"
    )
    # Both identifiers, because an incident is reconstructed from them: the
    # application_id says which instance went and the artifact_id says what to
    # redeploy to bring it back.
    assert ADOPTED_ID in warnings[0]
    assert ADOPTED_ARTIFACT in warnings[0]
    manager._deploy_application.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_adopted_app_is_not_claimed_to_be_re_established(caplog) -> None:
    manager = _make_manager()

    with caplog.at_level(logging.INFO, logger="test.controller_loss"):
        await manager._recover_from_controller_loss([ADOPTED_ID])
        await asyncio.sleep(0)

    # Logging the skip must not come at the price of also logging the
    # "re-establishing in-place" line, which would be false for an app nothing
    # can rebuild — a reader would conclude it came back.
    assert _at(caplog, logging.ERROR) == []


@pytest.mark.asyncio
async def test_a_built_app_still_redeploys_and_still_logs(caplog) -> None:
    manager = _make_manager()

    with caplog.at_level(logging.INFO, logger="test.controller_loss"):
        await manager._recover_from_controller_loss([BUILT_ID])
        await asyncio.sleep(0)

    errors = _at(caplog, logging.ERROR)
    assert len(errors) == 1 and BUILT_ID in errors[0]
    assert "re-establish" in errors[0]
    manager._deploy_application.assert_awaited()


@pytest.mark.asyncio
async def test_an_untracked_id_is_still_skipped_silently(caplog) -> None:
    manager = _make_manager()

    with caplog.at_level(logging.INFO, logger="test.controller_loss"):
        await manager._recover_from_controller_loss(["never-deployed"])
        await asyncio.sleep(0)

    # An id with no record is not an app that was lost — there is nothing to
    # report and nothing to look up for a message.
    assert caplog.records == []


@pytest.mark.asyncio
async def test_the_cooldown_still_bounds_how_often_this_can_speak(caplog) -> None:
    manager = _make_manager()

    with caplog.at_level(logging.INFO, logger="test.controller_loss"):
        await manager._recover_from_controller_loss([ADOPTED_ID])
        await manager._recover_from_controller_loss([ADOPTED_ID])
        await asyncio.sleep(0)

    # The reason this needs no separate dedupe: a sweep that runs inside the
    # cooldown returns before the loop, so a persistent head outage costs at
    # most one line per app per cooldown rather than one per monitor tick.
    assert len(_at(caplog, logging.WARNING)) == 1
    assert _CONTROLLER_RECOVERY_COOLDOWN_SECONDS > 0
