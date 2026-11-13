"""A non-dict ``scaling`` must be refused before ``build()`` runs (#0406).

``tests/apps/test_deploy_app_validates_scaling.py`` pins that ``deploy_app``
calls ``_validate_scaling_map``. That validator cannot be the whole boundary for
the TYPE of ``scaling``, because it also checks the map's keys against the user
``@bioengine.app`` class names, and those only exist after ``app_builder.build``
has minted a Hypha token (``builder.py``, "Mint a read-only token for the
introspect task") and submitted the introspect Ray task
(``ray.remote(...).remote(introspect_app_in_ray_task)``). So a validator placed
there refuses a wrong type only after paying for all of it.

What makes the ordering testable without a cluster: ``build`` is the first thing
in ``deploy_app`` that touches Ray or Hypha at all, so "build was not called" is
a sufficient stand-in for "no Ray call was made", and it is the strongest
assertion available to a unit test.

Both directions are asserted. A bad type must raise WITHOUT building; a valid
map must still reach the build and get past it. Asserting only the first would
pass just as well against a ``deploy_app`` that never builds anything.
"""

from __future__ import annotations

import logging

import pytest

from tests.apps.test_deploy_app_validates_scaling import (
    PAST_THE_VALIDATOR,
    _deploy,
    _make_manager,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad, type_name",
    [
        (True, "bool"),
        ("Entry", "str"),
        (["Entry"], "list"),
        (2, "int"),
    ],
)
async def test_a_truthy_non_dict_scaling_is_refused_before_the_build(bad, type_name):
    """THE PIN, and the half the old ``scaling or {}`` idiom could never catch:
    a truthy non-dict is not rescued by ``or {}`` and used to reach ``.items()``.
    """
    manager = _make_manager()

    with pytest.raises(ValueError) as caught:
        await _deploy(manager, bad)

    message = str(caught.value)
    assert "scaling must be a dict" in message, message
    assert type_name in message, message
    manager.app_builder.build.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("falsy", [False, 0, "", []])
async def test_a_falsy_non_dict_scaling_is_refused_before_the_build(falsy):
    """``or {}`` silently accepted every one of these and deployed on the
    decorator's own config, so a caller who meant to set replicas got none of
    what they asked for and no error. ``None`` is the one absent value that is
    legitimately a default, and it is the positive control below.
    """
    manager = _make_manager()

    with pytest.raises(ValueError, match="scaling must be a dict"):
        await _deploy(manager, falsy)

    manager.app_builder.build.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("good", [None, {}, {"Entry": {"num_replicas": 1}}])
async def test_a_valid_scaling_still_reaches_the_build(good):
    """POSITIVE CONTROL. Without this, the cases above pass against a
    ``deploy_app`` that refuses every ``scaling`` it is given, ``None``
    included — which is what every existing caller passes.
    """
    manager = _make_manager()

    with pytest.raises(RuntimeError, match=PAST_THE_VALIDATOR):
        await _deploy(manager, good)

    manager.app_builder.build.assert_called_once()


@pytest.mark.asyncio
async def test_the_per_deployment_name_check_still_runs_after_the_build():
    """CONTROL on the split itself: hoisting the type check must not displace
    the key check, which still needs the built class list and so still runs
    after the build. A refusal naming the unknown key, with the build having
    happened, is what distinguishes the two halves.
    """
    manager = _make_manager()

    with pytest.raises(ValueError, match="NoSuchClass") as caught:
        await _deploy(manager, {"NoSuchClass": {"num_replicas": 1}})

    assert PAST_THE_VALIDATOR not in str(caught.value)
    manager.app_builder.build.assert_called_once()


def test_the_imported_harness_is_the_one_being_asserted_against():
    """Guard against the shared fixtures being renamed out from under this file:
    an import that silently resolved to something else would make every pin
    above vacuous.
    """
    manager = _make_manager()

    assert manager.app_builder.build.call_count == 0
    assert isinstance(manager.logger, logging.Logger)
    assert PAST_THE_VALIDATOR == "reached _check_resources"
