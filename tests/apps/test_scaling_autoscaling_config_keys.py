"""Pin that an unknown ``autoscaling_config`` key is rejected rather than dropped.

Ray Serve's ``AutoscalingConfig`` is a pydantic model that ignores keys it does
not declare. A renamed or misspelled field therefore reaches Serve, is
discarded, and the deployment runs on Serve's default for that field — with no
warning anywhere, a healthy ``get_app_status``, and the declared value still
sitting in the manifest where every reader assumes it is in force.

That is not hypothetical: ``target_num_ongoing_requests_per_replica`` was
Serve's name for the steady-state target before it was renamed to
``target_ongoing_requests``, the old name appears in zero files of the vendored
ray 2.33 tree, and eight call sites in this repo were still passing it — all of
them silently running at the default of 2 instead of the value they declared.
``deploy_app``'s own documented ``scaling`` example taught the dead name.

So the ``scaling`` validator has to inspect the *contents* of
``autoscaling_config``, not just its shape: the structural checks it already
performs cannot see a key that is spelled wrong.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from ray.serve.config import AutoscalingConfig

from bioengine.apps.manager import AppsManager

WORKSPACE = "bioimage-io"
ARTIFACT_ID = f"{WORKSPACE}/nuclei-seg"
APP_ID = "nuclei-seg"
DEPLOYMENT = "RuntimeDeployment"
CONTEXT = {"user": {"id": "u-1", "email": "u@lab.test"}}

DEAD_KEY = "target_num_ongoing_requests_per_replica"
LIVE_KEY = "target_ongoing_requests"


def _fake_app(scaling):
    """The minimum of ``AppBuilder.build``'s result that ``deploy_app`` reads.

    ``scaling`` is echoed back the way ``AppBuilder.build`` does it
    (``builder.py:897`` → ``BuiltApp.scaling``), because what ``deploy_app``
    registers is the *built* app's map, not the argument it was handed.
    """
    return SimpleNamespace(
        metadata={
            "name": "Nuclei Seg",
            "description": "Segment nuclei.",
            "version": "1.0.0",
            "application_kwargs": {},
            "application_env_vars": {},
            "resources": {},
            "authorized_users": {"*": ["*"]},
            "available_methods": ["segment"],
            "deployed_by_worker_client_id": "client-1",
            "proxy_service_token_issued_at": 0.0,
            "proxy_service_token_ttl_seconds": 3600,
        },
        spec={"classes": {"c0": {"qualname": f"runtime.{DEPLOYMENT}"}}},
        scaling=dict(scaling or {}),
    )


def _make_manager() -> AppsManager:
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.scaling_keys")
    manager.admin_users = ["*"]
    manager._deployment_lock = asyncio.Lock()
    manager._check_initialized = lambda: None
    manager.server = MagicMock()
    manager.server.config.workspace = WORKSPACE

    manager.artifact_manager = MagicMock()
    manager.artifact_manager.read = AsyncMock(
        return_value={"versions": [{"version": "1.0.0", "created_at": 1}]}
    )

    manager.ray_cluster = MagicMock()
    manager.ray_cluster.check_connection = AsyncMock()
    manager.ray_cluster.call_with_reconnect = AsyncMock()

    async def _build(*, scaling=None, **_kwargs):
        return _fake_app(scaling)

    manager.app_builder = SimpleNamespace(build=_build)
    manager._check_resources = AsyncMock()
    manager._cancel_deployment_process = AsyncMock()
    manager._deploy_application = AsyncMock()
    manager._generate_application_id = AsyncMock(return_value="wandering-cloud-1234")
    manager._deployed_applications = {}
    return manager


async def _deploy(manager, autoscaling_config):
    result = await manager.deploy_app(
        context=CONTEXT,
        artifact_id=ARTIFACT_ID,
        application_id=APP_ID,
        version="1.0.0",
        scaling={DEPLOYMENT: {"autoscaling_config": autoscaling_config}},
    )
    task = manager._deployed_applications[result["application_id"]]["deployment_task"]
    await asyncio.gather(task, return_exceptions=True)
    return result


def test_the_dead_key_is_absent_from_ray_and_the_live_one_is_present() -> None:
    # Without this, the rejection below could pass for the wrong reason — a
    # validator checking against a hand-written key list would reject the dead
    # name even if Ray had kept it, and would keep rejecting it after a future
    # rename. The check reads Ray's own field set, so this is what makes the
    # rejection mean "Serve would drop it".
    assert DEAD_KEY not in AutoscalingConfig.model_fields
    assert LIVE_KEY in AutoscalingConfig.model_fields


@pytest.mark.asyncio
async def test_the_renamed_target_field_is_rejected_not_dropped() -> None:
    manager = _make_manager()

    with pytest.raises(ValueError) as excinfo:
        await _deploy(manager, {"min_replicas": 1, "max_replicas": 3, DEAD_KEY: 1.0})

    message = str(excinfo.value)
    assert DEAD_KEY in message, (
        "The error has to name the offending key: the whole failure mode is "
        "that the author cannot tell which value went missing."
    )
    assert APP_ID not in manager._deployed_applications, (
        "A rejected scaling map must not leave a half-registered application."
    )


@pytest.mark.asyncio
async def test_the_live_target_field_with_the_same_value_is_accepted() -> None:
    # The discrimination control for the test above: same numeric value, same
    # sibling keys, only the spelling differs. Were the validator rejecting
    # autoscaling_config wholesale, this would fail too.
    manager = _make_manager()

    result = await _deploy(
        manager, {"min_replicas": 1, "max_replicas": 3, LIVE_KEY: 1.0}
    )

    assert result["application_id"] == APP_ID
    assert manager._deployed_applications[APP_ID]["scaling"] == {
        DEPLOYMENT: {"autoscaling_config": {"min_replicas": 1, "max_replicas": 3, LIVE_KEY: 1.0}}
    }


@pytest.mark.asyncio
async def test_a_non_dict_autoscaling_config_is_left_to_serve() -> None:
    # The key check is `set(asc) - valid`, and `set()` on a non-mapping either
    # raises TypeError or enumerates characters as key names — so an int would
    # have escaped as an uncaught TypeError and a string would have been
    # reported as the unknown keys 'f' and 'o'. Neither is this check's job;
    # a malformed autoscaling_config reaches Serve exactly as it did before.
    for bad in (5, "foo"):
        manager = _make_manager()
        await _deploy(manager, bad)
        assert APP_ID in manager._deployed_applications


@pytest.mark.asyncio
async def test_every_unknown_key_is_listed_so_one_fix_round_is_enough() -> None:
    manager = _make_manager()

    with pytest.raises(ValueError) as excinfo:
        await _deploy(manager, {DEAD_KEY: 1.0, "min_replicass": 1, "max_replicas": 3})

    message = str(excinfo.value)
    assert DEAD_KEY in message and "min_replicass" in message
    assert "max_replicas'" not in message.split("key(s)")[1].split(";")[0], (
        "The valid sibling key must not be reported as unknown."
    )
