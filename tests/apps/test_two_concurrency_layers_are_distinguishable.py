"""``max_ongoing_requests`` names two independent limits; status must separate them.

An app author writes ``@bioengine.app(max_ongoing_requests=...)`` and gets a
Ray Serve cap on one replica of one deployment. A deployer passes
``max_ongoing_requests=...`` to ``deploy_app`` and gets the number of callers
the application's proxy admits across the whole instance. Nothing connects the
two, and ``get_app_status`` reported only the second one — under the bare name
``max_ongoing_requests``, which is also what the author typed.

So the author who declared 32 and read back 10 had no way to learn that the 10
was somebody else's number. Measured 2026-11-12 on the live KTH worker:
``omezarr-view`` 0.2.4 declares 32 and reported 10; ``annotation-broker``
0.12.2 declares 20 and reported 10; ``model-runner`` 2.11.6 declares 10 and
reported 150, which no app source anywhere declares.

The fix reports the decorator's value too, one level down under the class's own
name, so the two layers are told apart by *where* they appear. These tests pin
that both numbers survive into one payload, that they are not conflated, and
that nothing else rides out of the Serve controller's ``deployment_config``
alongside the one integer now read from it.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from bioengine.apps.manager import PUBLIC_DEPLOYMENT_FIELDS, AppsManager

from .test_app_status_identity_projection import (
    ADMIN,
    APP_ID,
    ANONYMOUS,
    ENTRY_DEPLOYMENT,
    context_for,
    make_manager,
    status_for,
)

# Distinct, non-default, and not each other: a test in which the two layers
# carried the same number could not tell a correct report from a conflated one.
DECORATOR_CAP = 32
PROXY_CAP = 150
PROXY_SERVE_CAP = 10

RUNTIME_DEPLOYMENT = "RuntimeApp"

SECRET_IN_DEPLOYMENT_CONFIG = "PLANTED-SECRET-INSIDE-DEPLOYMENT-CONFIG"


def manager_with_both_layers(
    *,
    decorator_cap: object = DECORATOR_CAP,
    proxy_cap: int = PROXY_CAP,
) -> AppsManager:
    """A manager whose two layers hold deliberately different numbers."""
    manager = make_manager(authorized_users={"*": [ADMIN]})
    manager._deployed_applications[APP_ID]["max_ongoing_requests"] = proxy_cap
    manager.ray_cluster.proxy_actor_handle.get_serve_instance_details.remote = (
        AsyncMock(
            return_value={
                "applications": {
                    APP_ID: {
                        "status": "RUNNING",
                        "message": "",
                        "deployments": {
                            ENTRY_DEPLOYMENT: {
                                "status": "HEALTHY",
                                "message": "",
                                "deployment_config": {
                                    "max_ongoing_requests": decorator_cap,
                                    # Planted beside it: whatever is copied out
                                    # of deployment_config must be that one
                                    # integer and not its neighbours.
                                    "runtime_env": {
                                        "env_vars": {
                                            "_BIOENGINE_SECRET_HYPHA_TOKEN": (
                                                SECRET_IN_DEPLOYMENT_CONFIG
                                            )
                                        }
                                    },
                                },
                                "replicas": [],
                            },
                            RUNTIME_DEPLOYMENT: {
                                "status": "HEALTHY",
                                "message": "",
                                "deployment_config": {"max_ongoing_requests": 5},
                                "replicas": [],
                            },
                            "ProxyDeployment": {
                                "status": "HEALTHY",
                                "message": "",
                                "deployment_config": {
                                    "max_ongoing_requests": PROXY_SERVE_CAP
                                },
                                "replicas": [],
                            },
                        },
                    }
                }
            }
        )
    )
    return manager


@pytest.mark.asyncio
async def test_both_limits_reach_one_payload_without_being_conflated() -> None:
    status = await status_for(manager_with_both_layers(), context_for(email=ADMIN))
    assert status["max_ongoing_requests"] == PROXY_CAP
    assert status["deployments"][ENTRY_DEPLOYMENT]["max_ongoing_requests"] == (
        DECORATOR_CAP
    )
    # Two deployments, two decorators: the field is per deployment, not one
    # number copied across the map.
    assert status["deployments"][RUNTIME_DEPLOYMENT]["max_ongoing_requests"] == 5


@pytest.mark.asyncio
async def test_the_proxys_own_serve_cap_is_not_reported_as_a_third_number() -> None:
    """``ProxyDeployment`` carries no author decorator, and its Serve-level cap
    is not the limit that governs client calls — the top-level field is. A 10
    beside a top-level 150 would read as a contradiction."""
    status = await status_for(manager_with_both_layers(), context_for(email=ADMIN))
    assert status["deployments"]["ProxyDeployment"]["max_ongoing_requests"] is None
    assert status["max_ongoing_requests"] == PROXY_CAP


@pytest.mark.asyncio
async def test_an_unset_cap_is_nulled_rather_than_breaking_the_whole_payload() -> None:
    """Ray dumps an unset ``DeploymentSchema`` field as the ``DEFAULT.VALUE``
    enum member, which ``json.dumps`` refuses. Passing it through would not
    mis-report one field — it would fail the serialization of every
    application's status at once."""
    from ray.serve.schema import DeploymentSchema

    # Taken from Ray rather than named, so the arm follows a renamed sentinel
    # instead of silently testing a value Ray stopped producing.
    sentinel = DeploymentSchema(name="unset").model_dump()["max_ongoing_requests"]
    assert not isinstance(sentinel, int), f"Ray now dumps an unset cap as {sentinel!r}"

    manager = manager_with_both_layers(decorator_cap=sentinel)
    status = await status_for(manager, context_for(email=ADMIN))
    assert status["deployments"][ENTRY_DEPLOYMENT]["max_ongoing_requests"] is None
    json.dumps(status["deployments"])


@pytest.mark.asyncio
async def test_nothing_else_rides_out_of_deployment_config() -> None:
    """The one integer read out of the controller's dump is read as an integer.
    A future edit that copied the dict, or widened the read, would carry
    ``runtime_env.env_vars`` — every raw _BIOENGINE_SECRET_* value — into a
    response an app's whole roster can read."""
    status = await status_for(manager_with_both_layers(), context_for(email=ADMIN))
    entry = status["deployments"][ENTRY_DEPLOYMENT]
    assert isinstance(entry["max_ongoing_requests"], int)
    assert SECRET_IN_DEPLOYMENT_CONFIG not in repr(status)


@pytest.mark.asyncio
async def test_the_plant_is_findable_when_it_is_not_projected_away() -> None:
    """Positive control for the assertion above: the planted value is really in
    the dump this call reads, so its absence is the projection working rather
    than a search that never matched."""
    manager = manager_with_both_layers()
    details = (
        await manager.ray_cluster.proxy_actor_handle.get_serve_instance_details.remote()
    )
    assert SECRET_IN_DEPLOYMENT_CONFIG in repr(details)


@pytest.mark.asyncio
async def test_an_author_watching_their_own_app_anonymously_sees_both() -> None:
    """The decorator's value is in the app's published source, and the limit it
    sits beside is already public, so withholding it would only hide it from
    the author who declared it."""
    status = await status_for(manager_with_both_layers(), ANONYMOUS)
    assert status["max_ongoing_requests"] == PROXY_CAP
    assert status["deployments"][ENTRY_DEPLOYMENT]["max_ongoing_requests"] == (
        DECORATOR_CAP
    )
    assert "max_ongoing_requests" in PUBLIC_DEPLOYMENT_FIELDS


def test_the_two_parameters_each_say_which_layer_they_are() -> None:
    """Whichever of the two an author is reading, the text has to name its own
    layer AND point at the other — naming only one leaves the reader exactly
    where they started."""
    import inspect

    from bioengine._app.decorators import app as app_decorator

    parameter = inspect.signature(AppsManager.deploy_app).parameters[
        "max_ongoing_requests"
    ]
    proxy_text = getattr(parameter.default, "description", "") or ""
    assert "ProxyDeployment" in proxy_text and "whole instance" in proxy_text, (
        "deploy_app's max_ongoing_requests description no longer says which "
        f"layer it bounds: {proxy_text!r}"
    )
    assert "@bioengine.app" in proxy_text, "it does not point at the other layer"
    assert "per-replica" in proxy_text

    decorator_text = app_decorator.__doc__ or ""
    assert "per replica" in decorator_text
    assert "deploy_app" in decorator_text, "it does not point at the other layer"
    assert "independent" in decorator_text


def test_independence_is_not_stated_as_a_claim_about_the_other_layers_value() -> None:
    """Independence means neither knob writes the other, not that the other is
    sitting at its default.

    The description used to end that sentence with "and setting either leaves
    the other at its default", which is false in both directions. On an update
    the proxy cap is *preserved* rather than reset to 10 — ``deploy_app`` reads
    the running application's value when the parameter is omitted — so a cap an
    earlier call set to 150 stays 150. And the per-replica cap is whatever the
    author wrote in source, which is the default only if they wrote nothing:
    ``omezarr-view`` 0.2.4 declares 32.
    """
    import inspect

    parameter = inspect.signature(AppsManager.deploy_app).parameters[
        "max_ongoing_requests"
    ]
    text = getattr(parameter.default, "description", "") or ""

    # Positive control, first: an emptied or renamed description satisfies every
    # absence assertion below while saying nothing at all.
    assert "independent" in text, f"the independence claim itself is gone: {text!r}"
    assert "preserves the previous value if updating" in text, (
        f"the update behaviour is no longer stated: {text!r}"
    )

    assert "at its default" not in text, (
        "the description again promises a value for the other layer rather than "
        f"only that the two do not affect each other: {text!r}"
    )
