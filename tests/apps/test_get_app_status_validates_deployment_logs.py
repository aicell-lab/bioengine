"""``get_app_status`` must actually CALL the deployment_logs validator (#0406).

``tests/test_deployment_logs_validation.py`` covers the helper thoroughly, but
every case there imports it directly and none drives ``get_app_status`` — which
is exactly the gap that let the original defect ship: the shape was correctly
annotated on the public method and never checked in its body.

This is the same lesson as ``test_deploy_app_validates_scaling.py``, which was
written after the same omission recurred three times across one PR ladder:
testing the helper is faster to write and is precisely what skips the call site.

Both directions are asserted, which is what makes this a pin rather than a
tautology. A bad map must raise AT the validator; a valid map must get PAST it.
Asserting only the first would also pass against a ``get_app_status`` that
refused every call.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps.manager import AppsManager

CONTEXT = {"user": {"id": "admin", "email": "admin@example.com"}}


def _make_manager() -> AppsManager:
    """An AppsManager wired with only what get_app_status touches before it
    returns for an empty application set."""
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.get_app_status_deployment_logs")
    # Empty, so a call that gets past the validator returns {} immediately and
    # never needs a Ray cluster. That early return is the "past" marker.
    manager._deployed_applications = {}
    manager.ray_cluster = MagicMock()
    return manager


async def _status(manager, deployment_logs):
    return await manager.get_app_status(
        application_ids=None,
        deployment_logs=deployment_logs,
        context=CONTEXT,
    )


@pytest.mark.asyncio
async def test_a_non_dict_map_is_refused_by_get_app_status():
    """THE PIN. Delete the ``_validate_deployment_logs_map`` call from
    ``get_app_status`` and this fails: ``True`` sails through and the call
    returns ``{}`` instead of raising."""
    manager = _make_manager()

    with pytest.raises(ValueError) as caught:
        await _status(manager, True)

    assert "deployment_logs must be a dict" in str(caught.value)
    assert "bool" in str(caught.value)


@pytest.mark.asyncio
async def test_a_bad_entry_is_refused_by_get_app_status():
    """A second invalid shape, so the pin does not rest on the top-level branch
    alone — an entry-level fault is the more likely real-world mistake."""
    manager = _make_manager()

    with pytest.raises(ValueError, match="must be a dict"):
        await _status(manager, {"RuntimeApp": 3})


@pytest.mark.asyncio
async def test_an_unknown_key_is_refused_by_get_app_status():
    manager = _make_manager()

    with pytest.raises(ValueError, match="unknown key"):
        await _status(manager, {"RuntimeApp": {"logs_tails": 5}})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "good",
    [None, {}, {"RuntimeApp": {"logs_tail": 200, "n_previous_replica": 3}}],
)
async def test_a_valid_map_gets_past_the_validator(good):
    """POSITIVE CONTROL. Without this, the cases above pass just as well against
    a ``get_app_status`` that raises for every ``deployment_logs`` it is given —
    including the ``None`` that every existing caller passes."""
    manager = _make_manager()

    assert await _status(manager, good) == {}


@pytest.mark.asyncio
async def test_the_validator_runs_before_the_ray_connection():
    """Where the guard sits is part of the fix. The original crash happened
    after ``check_connection`` and after a ``get_serve_instance_details`` round
    trip, so a bad argument cost a cluster call before failing. Pins that a
    refusal is now free."""
    manager = _make_manager()
    manager.ray_cluster = MagicMock()

    with pytest.raises(ValueError):
        await _status(manager, True)

    manager.ray_cluster.check_connection.assert_not_called()


def test_module_imports_without_asyncio_warnings():
    """Guard against the suite's asyncio marker being absent — if
    pytest-asyncio is not configured, the coroutine cases above would be
    collected, skipped-as-passed, and this file would assert nothing."""
    assert asyncio.iscoroutinefunction(AppsManager.get_app_status)


# --------------------------------------------------------------------------
# The overrides must still REACH Ray, not merely survive the validator.
#
# The positive control above asserts ``_status(...) == {}``, which is the
# empty-application early return: it proves a valid map got PAST the check,
# not that an override still changes anything. A fix that validated correctly
# and then dropped the overrides on the floor would be indistinguishable from
# a working one. These drive the real ``_get_deployment_status`` against a
# stubbed proxy handle and assert the ``tail`` actually requested.
# --------------------------------------------------------------------------

DEPLOYMENT = "RuntimeApp"


def _tails_requested(deployment_logs, logs_tail=30):
    """Return the ``tail`` values ``_get_deployment_status`` asked the proxy for."""
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
    """tail=30 for both — the two cases that must be unaffected by the check."""
    assert _tails_requested(empty) == [30]


def test_a_real_override_still_reaches_ray():
    """tail=5 — the feature the validator guards must still work."""
    assert _tails_requested({DEPLOYMENT: {"logs_tail": 5}}) == [5]


def test_an_override_naming_another_deployment_falls_back_to_the_default():
    """The documented silent-ignore, asserted on the tail rather than on the map.

    ``test_an_unknown_deployment_name_is_accepted`` pins that the validator lets
    such a key through. This pins what then happens: the deployment that IS
    present still gets the top-level default, so a stale dashboard override
    changes nothing rather than suppressing logs.
    """
    assert _tails_requested({"SomeOtherClass": {"logs_tail": 5}}) == [30]
