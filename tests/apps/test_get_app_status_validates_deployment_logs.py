"""``get_app_status`` must CALL the deployment_logs validator, and still pass overrides through (#0406).

Two separate things, because the unit tests in ``tests/test_deployment_logs_validation.py``
import the helper directly and so cover neither:

1. The call site. Delete ``_validate_deployment_logs_overrides(deployment_logs)`` from
   ``get_app_status`` and the first two cases fail. This is the shape that was named in
   ``tests/apps/test_deploy_app_validates_scaling.py`` as having recurred three times
   across one PR ladder — unit-testing the helper is what skips the call site.

2. The pass-through. A validator is a net gain only if the feature it guards still
   works, so the effective ``tail`` actually handed to Ray is asserted for the three
   cases the row cares about: no map (30), an empty map (30), and a real override (5).
   Without this arm, a "fix" that validated correctly and then dropped the overrides on
   the floor would be indistinguishable from a working one.

The validator runs before the empty-app early return deliberately: a caller who gets the
parameter wrong should be told so whether or not anything is deployed. ``get_app_status``
performs no permission check, so nothing is validated ahead of an authorization gate.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps.manager import AppsManager

CONTEXT = {"user": {"id": "admin", "email": "admin@example.com"}}
DEPLOYMENT = "RuntimeApp"


def _make_manager() -> AppsManager:
    """An AppsManager wired with only what get_app_status touches."""
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.get_app_status_deployment_logs")
    manager._deployed_applications = {}

    ray_cluster = MagicMock()
    ray_cluster.check_connection = AsyncMock()
    manager.ray_cluster = ray_cluster
    return manager


@pytest.mark.parametrize(
    "bad, expected",
    [
        (True, "got bool"),
        (0, "got int"),
        ("RuntimeApp", "got str"),
        ({DEPLOYMENT: "nonsense"}, "got str"),
    ],
    ids=["truthy-non-dict", "falsy-non-dict", "string", "non-dict-entry"],
)
def test_a_bad_map_is_refused_by_get_app_status_itself(bad, expected):
    """Driven through the public method, not the helper."""
    manager = _make_manager()

    with pytest.raises(ValueError) as excinfo:
        asyncio.run(manager.get_app_status(deployment_logs=bad, context=CONTEXT))

    assert expected in str(excinfo.value)


def test_a_bad_map_is_refused_before_any_ray_call():
    """It must not take a cluster round-trip to learn the argument was malformed.

    ``check_connection`` is the first thing ``get_app_status`` awaits on the happy
    path, so asserting it was never awaited is what places the validator ahead of Ray.
    """
    manager = _make_manager()

    with pytest.raises(ValueError):
        asyncio.run(manager.get_app_status(deployment_logs=True, context=CONTEXT))

    manager.ray_cluster.check_connection.assert_not_awaited()
    manager.ray_cluster.proxy_actor_handle.get_serve_instance_details.remote.assert_not_called()


def _tails_requested(deployment_logs, logs_tail=30):
    """Run ``_get_deployment_status`` against a stubbed proxy; return the tails asked for."""
    manager = _make_manager()
    get_logs = AsyncMock(return_value={"replica_1": {"stdout": [], "stderr": []}})
    manager.ray_cluster.proxy_actor_handle.get_deployment_logs.remote = get_logs
    manager._project_replicas = lambda replicas: []

    asyncio.run(
        manager._get_deployment_status(
            application_id="demo-app",
            application_details={"deployments": {DEPLOYMENT: {"status": "HEALTHY"}}},
            n_previous_replica=0,
            logs_tail=logs_tail,
            deployment_logs_overrides=deployment_logs,
        )
    )
    return [call.kwargs["tail"] for call in get_logs.await_args_list]


@pytest.mark.parametrize("empty", [None, {}])
def test_no_override_still_requests_the_top_level_default(empty):
    """30 and 30 — the two cases that must be unaffected by the new check."""
    assert _tails_requested(empty) == [30]


def test_a_real_override_still_reaches_ray():
    """5 — the feature the validator guards must still work."""
    assert _tails_requested({DEPLOYMENT: {"logs_tail": 5}}) == [5]


def test_an_override_naming_another_deployment_is_still_ignored_silently():
    """Documented behaviour, and the one case the validator must NOT start rejecting.

    The docstring promises overrides naming a deployment absent from the queried apps
    are ignored, because the dashboard may carry stale overrides. A validator checking
    keys against the live deployment set — the obvious symmetry with
    ``_validate_scaling_map``, which does exactly that for ``scaling`` — would break it.
    """
    assert _tails_requested({"SomeOtherClass": {"logs_tail": 5}}) == [30]
