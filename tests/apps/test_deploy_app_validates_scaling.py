"""``deploy_app`` must actually CALL the scaling validator (#0212).

``_validate_scaling_map`` is thoroughly unit-tested — ``tests/test_scaling_validation.py``
names it eleven times — but nothing drove ``deploy_app`` and asserted that an invalid
map is refused. Measured on ``origin/main`` at 1ecea12: deleting the call site in
``deploy_app`` left all 16 cases across the two scaling suites green, so the wiring was
reachable, correct, and unpinned. A future edit to that one line would have shipped
silently.

The shape is worth naming because it recurred three times across one PR ladder: the
surrounding tests import the helper directly, which is faster to write and is exactly
what skips the call site.

Both directions are asserted, and that is what makes this a pin rather than a
tautology. An invalid map must RAISE at the validator; a valid map must get PAST it
and reach the next step. Testing only the first would also pass against a
``deploy_app`` that rejected every scaling map it was given.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps.manager import AppsManager

ARTIFACT_ID = "bioimage-io/demo-app"
APP_ID = "demo-app"
USER_EMAIL = "admin@example.com"
CONTEXT = {"user": {"id": "admin", "email": USER_EMAIL}}

# Raised by the stubbed step that follows the validator. Reaching it is how a
# valid map proves it got past the call site, rather than merely not raising —
# "no exception" would also describe a deploy_app that returned early.
PAST_THE_VALIDATOR = "reached _check_resources"


def _make_manager(class_names=("Entry", "Runtime")) -> AppsManager:
    """An AppsManager wired with only what deploy_app touches before the validator."""
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.deploy_app_scaling")
    manager._deployment_lock = asyncio.Lock()
    manager._deployed_applications = {}
    manager.admin_users = [USER_EMAIL]

    manager._check_initialized = lambda: None
    manager._get_full_artifact_id = lambda aid: aid if "/" in aid else f"bioimage-io/{aid}"

    server = MagicMock()
    server.config.workspace = "bioimage-io"
    manager.server = server

    ray_cluster = MagicMock()
    ray_cluster.check_connection = AsyncMock()
    manager.ray_cluster = ray_cluster

    # The build is what discovers the user classes the validator checks against,
    # so the names here are the whole input to the decision under test.
    app = SimpleNamespace(
        spec={
            "classes": {
                name: {"qualname": f"module.{name}"} for name in class_names
            }
        },
        metadata={
            "version": "1.0.0",
            "source_signature": "sig",
            # Read at the _check_resources call, which is the step after the
            # validator and therefore the marker this test aims at.
            "resources": {},
        },
    )
    app_builder = MagicMock()
    app_builder.build = AsyncMock(return_value=app)
    manager.app_builder = app_builder

    async def _check_resources(*_args, **_kwargs):
        raise RuntimeError(PAST_THE_VALIDATOR)

    manager._check_resources = _check_resources
    return manager


async def _deploy(manager, scaling):
    return await manager.deploy_app(
        artifact_id=ARTIFACT_ID,
        version="1.0.0",
        application_id=APP_ID,
        scaling=scaling,
        context=CONTEXT,
    )


@pytest.mark.asyncio
async def test_an_unknown_deployment_name_is_refused_by_deploy_app():
    """THE PIN. Delete the ``_validate_scaling_map`` call from ``deploy_app`` and
    this fails: the bad map sails through and the run dies at ``_check_resources``
    instead, which is a different exception with a different message."""
    manager = _make_manager()

    with pytest.raises(Exception) as caught:
        await _deploy(manager, {"NoSuchClass": {"num_replicas": 1}})

    message = str(caught.value)
    assert PAST_THE_VALIDATOR not in message, (
        "the deploy got past the scaling validator — the call site is missing"
    )
    assert "NoSuchClass" in message, message


@pytest.mark.asyncio
async def test_a_zero_replica_entry_is_refused_by_deploy_app():
    """A second invalid shape, so the pin does not rest on the unknown-name branch
    alone — that one is also reachable from the artifact's class list changing."""
    manager = _make_manager()

    with pytest.raises(Exception) as caught:
        await _deploy(manager, {"Entry": {"num_replicas": 0}})

    message = str(caught.value)
    assert PAST_THE_VALIDATOR not in message, (
        "the deploy got past the scaling validator — the call site is missing"
    )
    assert "Entry" in message, message


@pytest.mark.asyncio
async def test_a_valid_map_gets_past_the_validator():
    """POSITIVE CONTROL. Without this, both tests above pass just as well against a
    deploy_app that refuses every scaling map, or one that never builds at all."""
    manager = _make_manager()

    with pytest.raises(RuntimeError, match=PAST_THE_VALIDATOR):
        await _deploy(manager, {"Entry": {"num_replicas": 1}})


@pytest.mark.asyncio
async def test_an_empty_map_gets_past_the_validator():
    """The default. ``scaling`` is optional, and the validator must not turn an
    absent map into a refusal — the decorator's own config stands in that case."""
    manager = _make_manager()

    with pytest.raises(RuntimeError, match=PAST_THE_VALIDATOR):
        await _deploy(manager, {})
