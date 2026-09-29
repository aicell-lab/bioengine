"""The public status surface must not publish anyone's identity.

The worker's Hypha service is registered ``"visibility": "public"`` and
``get_app_status`` carries no permission check, so every field it returned was
readable by anyone who could reach the worker — including each application's
``authorized_users`` (the addresses of third parties who never interacted with
this worker), ``last_updated_by``, the names of its secret environment
variables, and its replica logs verbatim.

The health half stays anonymous: an app author watches a deployment come up
through this call, and a user checks whether an app is answering through it.
What is withheld is who may use the application, what the deployer configured
it with, and what its replicas printed.

Asserted on the response *shape*, not on the one field we know about today: a
denylist would leave every field nobody has thought of on the public side.

Every address here is a fake at example.invalid, which by RFC 2606 can never be
a real domain.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps.manager import (
    PUBLIC_APP_STATUS_FIELDS,
    PUBLIC_DEPLOYMENT_FIELDS,
    AppsManager,
    public_app_status,
)

APP_ID = "cellpose-finetuning"
ENTRY_DEPLOYMENT = "CellposeApp"

ADMIN = "admin@example.invalid"
MEMBER = "member@example.invalid"
MEMBER_ID = "user:member"
OUTSIDER = "outsider@example.invalid"
DEPLOYER = "deployer@example.invalid"

# One marker per withheld input, so a leak names which one leaked.
SECRET_VALUE = "PLANTED-SECRET-VALUE-NOT-A-CREDENTIAL"
SECRET_KEY = "_MODEL_API_KEY"
PLAIN_ENV_VALUE = "https://internal.example.invalid/v1"
KWARG_VALUE = "PLANTED-KWARG-VALUE"
LOG_VALUE = "PLANTED-LOG-LINE"
# The positive control: planted in a field that IS public, so an assertion that
# a marker is absent means the projection dropped it rather than that the
# search never worked.
PUBLIC_MARKER = "PLANTED-PUBLIC-DESCRIPTION"


def context_for(*, user_id: str | None = None, email: str | None = None) -> dict:
    return {"user": {"id": user_id or "anonymous-user", "email": email or "no-email"}}


ANONYMOUS = context_for()


def make_manager(
    *,
    authorized_users: dict | None = None,
    admin_users: list | None = None,
    controller_env_vars: dict | None = None,
    proxy_replicas: list | None = None,
) -> AppsManager:
    """An AppsManager wired with only what ``get_app_status`` touches."""
    deployed = asyncio.Event()
    deployed.set()
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test")
    manager.startup_applications = []
    manager.admin_users = admin_users if admin_users is not None else [ADMIN]
    manager._deployed_applications = {
        APP_ID: {
            "is_deployed": deployed,
            "display_name": "Cellpose Finetuning",
            "description": f"Finetune Cellpose models. {PUBLIC_MARKER}",
            "artifact_id": "bioimage-io/cellpose-finetuning",
            "version": "1.0.1",
            "recovered_app": False,
            "application_kwargs": {"notes": KWARG_VALUE},
            "application_env_vars": {
                ENTRY_DEPLOYMENT: {
                    SECRET_KEY: SECRET_VALUE,
                    "MODEL_ENDPOINT": PLAIN_ENV_VALUE,
                }
            },
            "disable_gpu": False,
            "application_resources": {"num_cpus": 1},
            "authorized_users": (
                authorized_users
                if authorized_users is not None
                else {"*": [MEMBER, MEMBER_ID, ADMIN]}
            ),
            "available_methods": ["train"],
            "max_ongoing_requests": 1,
            "scaling": {},
            "static_site_url": None,
            "started_at": 1_700_000_000.0,
            "last_updated_at": 1_700_000_000.0,
            "last_updated_by": DEPLOYER,
            "auto_redeploy": False,
            "deployed_by_worker_client_id": "worker-abc",
            "proxy_service_token_issued_at": None,
            "proxy_service_token_ttl_seconds": None,
            "entry_deployment_name": ENTRY_DEPLOYMENT,
            "source_signature": None,
        }
    }

    server = MagicMock()
    server.config.workspace = "example-workspace"
    server.config.client_id = "worker-abc"
    server.config.public_base_url = "https://hypha.example.invalid"
    manager.server = server

    ray_cluster = MagicMock()
    handle = ray_cluster.proxy_actor_handle
    # The Serve controller's dump carries deployment_config, and that is where
    # runtime_env.env_vars — every raw _BIOENGINE_SECRET_* value, including the
    # worker's Hypha token — lives on the way through this process.
    handle.get_serve_instance_details.remote = AsyncMock(
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
                                "runtime_env": {
                                    "env_vars": controller_env_vars
                                    or {
                                        "_BIOENGINE_SECRET_HYPHA_TOKEN": SECRET_VALUE,
                                    }
                                }
                            },
                            "replicas": [
                                {
                                    "replica_id": "r-1",
                                    "node_id": "node-1",
                                    "node_ip": "10.0.0.1",
                                    "node_instance_id": "pod-1",
                                    "state": "RUNNING",
                                    "pid": 4321,
                                    "start_time_s": 1_700_000_000.0,
                                    "log_file_path": "/tmp/replica.log",
                                    "actor_id": "actor-1",
                                }
                            ],
                        }
                    },
                }
            }
        }
    )
    handle.get_deployment_logs.remote = AsyncMock(
        return_value={"r-1": {"stdout": f"booting… {LOG_VALUE}", "stderr": ""}}
    )
    handle.get_deployment_replicas.remote = AsyncMock(
        return_value=proxy_replicas if proxy_replicas is not None else []
    )
    handle.get_service_registration.remote = AsyncMock(return_value=True)
    handle.get_replica_identities.remote = AsyncMock(return_value={})
    ray_cluster.check_connection = AsyncMock(return_value=None)
    manager.ray_cluster = ray_cluster
    return manager


async def status_for(manager: AppsManager, context: dict) -> dict:
    result = await manager.get_app_status(context=context)
    return result[APP_ID]


# ---------------------------------------------------------------------------
# The unauthenticated response, asserted on its SHAPE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_anonymous_response_carries_no_field_outside_the_allowlist():
    """The point of the allowlist: a field added to the status payload next
    year must not become public by nobody having thought about it."""
    status = await status_for(make_manager(), ANONYMOUS)
    unexpected = set(status) - set(PUBLIC_APP_STATUS_FIELDS)
    assert not unexpected, f"published {sorted(unexpected)}"


@pytest.mark.asyncio
async def test_the_anonymous_response_carries_no_deployment_field_outside_the_allowlist():
    status = await status_for(make_manager(), ANONYMOUS)
    for name, deployment in status["deployments"].items():
        unexpected = set(deployment) - set(PUBLIC_DEPLOYMENT_FIELDS)
        assert not unexpected, f"'{name}' published {sorted(unexpected)}"


@pytest.mark.asyncio
async def test_no_planted_identity_survives_into_the_anonymous_response():
    """Searched over the whole serialized response rather than field by field,
    so a value that reappears somewhere unexpected is still caught."""
    status = await status_for(make_manager(), ANONYMOUS)
    rendered = repr(status)
    for planted in (
        MEMBER,
        MEMBER_ID,
        ADMIN,
        DEPLOYER,
        SECRET_VALUE,
        SECRET_KEY.lstrip("_"),
        PLAIN_ENV_VALUE,
        KWARG_VALUE,
        LOG_VALUE,
    ):
        assert planted not in rendered, planted


@pytest.mark.asyncio
async def test_the_search_finds_a_value_that_is_meant_to_come_through():
    """Positive control for the assertions above: absence means the projection
    dropped it, not that searching the response never worked."""
    status = await status_for(make_manager(), ANONYMOUS)
    assert PUBLIC_MARKER in repr(status)


@pytest.mark.asyncio
async def test_every_application_is_projected_not_just_the_first():
    manager = make_manager()
    second = dict(manager._deployed_applications[APP_ID])
    manager._deployed_applications["other-app"] = second
    result = await manager.get_app_status(context=ANONYMOUS)
    assert set(result) == {APP_ID, "other-app"}
    for app_id, status in result.items():
        assert "authorized_users" not in status, app_id


# ---------------------------------------------------------------------------
# What the anonymous half is for: health, versions, and how to call the app
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_anonymous_caller_still_sees_whether_the_app_is_running():
    status = await status_for(make_manager(), ANONYMOUS)
    assert status["status"] == "RUNNING"
    assert status["artifact_id"] == "bioimage-io/cellpose-finetuning"
    assert status["version"] == "1.0.1"
    assert status["deployments"][ENTRY_DEPLOYMENT]["status"] == "HEALTHY"
    assert status["deployments"][ENTRY_DEPLOYMENT]["replica_states"] == {"RUNNING": 1}


@pytest.mark.asyncio
async def test_the_anonymous_caller_still_gets_the_service_id_to_call():
    """The website's logged-out apps page reads exactly two things off this
    call — the websocket service id and the status — and drops any app without
    one. Withholding it would empty that page."""
    manager = make_manager(proxy_replicas=["CellposeApp#abc123"])
    status = await status_for(manager, ANONYMOUS)
    assert status["service_ids"]["websocket_service_id"].endswith(f":{APP_ID}")
    assert status["service_registered"] is True


@pytest.mark.asyncio
async def test_an_application_that_is_not_running_still_answers_anonymously():
    manager = make_manager()
    result = await manager.get_app_status(
        application_ids=["never-deployed"], context=ANONYMOUS
    )
    assert result["never-deployed"]["status"] == "NOT_RUNNING"
    assert "deploy_app" in result["never-deployed"]["message"]


# ---------------------------------------------------------------------------
# The roster goes to whoever is named on it, and to a worker admin
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_named_member_sees_the_roster():
    status = await status_for(
        make_manager(authorized_users={"*": [MEMBER]}), context_for(email=MEMBER)
    )
    assert status["authorized_users"] == {"*": [MEMBER]}
    assert status["last_updated_by"] == DEPLOYER


@pytest.mark.asyncio
async def test_a_user_id_entry_also_earns_the_roster():
    status = await status_for(
        make_manager(authorized_users={"*": [MEMBER_ID]}),
        context_for(user_id=MEMBER_ID),
    )
    assert status["authorized_users"] == {"*": [MEMBER_ID]}


@pytest.mark.asyncio
async def test_a_method_specific_entry_earns_the_roster():
    """authorized_users is keyed by method name; being on any rule counts."""
    status = await status_for(
        make_manager(authorized_users={"*": [], "train": [MEMBER]}),
        context_for(email=MEMBER),
    )
    assert status["authorized_users"] == {"*": [], "train": [MEMBER]}


@pytest.mark.asyncio
async def test_a_worker_admin_sees_the_roster_of_a_public_app():
    """A public app's roster is {"*": ["*"]}, which no wildcard-free check can
    pass — the admin list is what carries an operator through."""
    status = await status_for(
        make_manager(authorized_users={"*": ["*"]}), context_for(email=ADMIN)
    )
    assert status["authorized_users"] == {"*": ["*"]}


@pytest.mark.asyncio
async def test_a_logged_in_outsider_sees_only_the_public_view():
    status = await status_for(make_manager(), context_for(email=OUTSIDER))
    assert "authorized_users" not in status
    assert "last_updated_by" not in status


@pytest.mark.asyncio
async def test_membership_is_per_application():
    """Being on one app's roster must not reveal another app's."""
    manager = make_manager(authorized_users={"*": [MEMBER]})
    theirs = dict(manager._deployed_applications[APP_ID])
    theirs["authorized_users"] = {"*": [OUTSIDER]}
    manager._deployed_applications["theirs"] = theirs
    result = await manager.get_app_status(context=context_for(email=MEMBER))
    assert result[APP_ID]["authorized_users"] == {"*": [MEMBER]}
    assert "authorized_users" not in result["theirs"]


# ---------------------------------------------------------------------------
# A wildcard grants the app, not the roster
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_wildcard_does_not_hand_the_roster_to_anonymous():
    """An app listing both "*" and named addresses would otherwise publish
    those addresses to everyone — the wildcard authorizes calling the app, not
    reading who else may call it."""
    status = await status_for(
        make_manager(authorized_users={"*": ["*", MEMBER]}), ANONYMOUS
    )
    assert "authorized_users" not in status
    assert MEMBER not in repr(status)


@pytest.mark.asyncio
async def test_a_named_member_of_a_wildcard_app_still_sees_the_roster():
    status = await status_for(
        make_manager(authorized_users={"*": ["*", MEMBER]}), context_for(email=MEMBER)
    )
    assert status["authorized_users"] == {"*": ["*", MEMBER]}


@pytest.mark.asyncio
async def test_a_wildcard_admin_list_does_not_hand_the_roster_to_anonymous():
    status = await status_for(
        make_manager(authorized_users={"*": [MEMBER]}, admin_users=["*"]), ANONYMOUS
    )
    assert "authorized_users" not in status


@pytest.mark.asyncio
async def test_a_caller_with_no_usable_context_gets_the_public_view():
    """Fails closed: an unparseable context must not be read as entitlement."""
    for context in (None, {}, {"user": "not-a-dict"}, {"user": {}}):
        status = await status_for(make_manager(), context)
        assert "authorized_users" not in status


# ---------------------------------------------------------------------------
# Secret NAMES and replica logs, the two disclosures adjacent to the roster
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_secret_variable_names_are_not_public():
    """The values were already masked; the names were not, and they say which
    third-party services an application talks to."""
    status = await status_for(make_manager(), ANONYMOUS)
    assert "application_env_vars" not in status
    assert "MODEL_API_KEY" not in repr(status)


@pytest.mark.asyncio
async def test_replica_logs_are_not_public():
    """Unfiltered process output: its content is whatever the application
    decided to print, so it can only be withheld, not projected."""
    status = await status_for(make_manager(), ANONYMOUS)
    assert "logs" not in status["deployments"][ENTRY_DEPLOYMENT]
    assert LOG_VALUE not in repr(status)


@pytest.mark.asyncio
async def test_an_entitled_caller_still_gets_the_logs_to_debug_with():
    status = await status_for(
        make_manager(authorized_users={"*": [MEMBER]}), context_for(email=MEMBER)
    )
    assert LOG_VALUE in repr(status["deployments"][ENTRY_DEPLOYMENT]["logs"])


@pytest.mark.asyncio
async def test_deploy_time_configuration_is_not_public():
    status = await status_for(make_manager(), ANONYMOUS)
    assert "application_kwargs" not in status


# ---------------------------------------------------------------------------
# Raw secret values reach nobody — two independent barriers, pinned separately
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_serve_controller_dump_does_not_cross_into_the_response():
    """First barrier: ``_get_deployment_status`` names the fields it copies and
    never reads ``deployment_config``, which is where ``runtime_env.env_vars``
    — every raw _BIOENGINE_SECRET_* value, including the Hypha token — sits on
    its way through this process. Planted ONLY in the controller dump here, so
    the masking barrier cannot account for its absence."""
    manager = make_manager(
        controller_env_vars={"_BIOENGINE_SECRET_HYPHA_TOKEN": SECRET_VALUE}
    )
    manager._deployed_applications[APP_ID]["application_env_vars"] = {}
    status = await status_for(manager, context_for(email=ADMIN))
    assert SECRET_VALUE not in repr(status)


@pytest.mark.asyncio
async def test_the_secret_record_is_masked_independently_of_that():
    """Second barrier: the record the worker keeps is masked by
    ``_filter_secret_env_vars``. Planted ONLY there, so barrier one cannot
    account for its absence. Neither failing alone would leak."""
    manager = make_manager(controller_env_vars={})
    status = await status_for(manager, context_for(email=ADMIN))
    env = status["application_env_vars"][ENTRY_DEPLOYMENT]
    assert env["MODEL_API_KEY"] == "*****"
    assert SECRET_VALUE not in repr(status)


@pytest.mark.asyncio
async def test_the_probe_finds_a_plain_value_on_the_same_path():
    """Positive control for both barriers: a non-secret variable does reach the
    entitled response, so the two absences above are real absences rather than
    a search that never matched anything."""
    status = await status_for(make_manager(), context_for(email=ADMIN))
    assert status["application_env_vars"][ENTRY_DEPLOYMENT]["MODEL_ENDPOINT"] == (
        PLAIN_ENV_VALUE
    )


# ---------------------------------------------------------------------------
# The projection helper on its own
# ---------------------------------------------------------------------------


def test_the_allowlist_does_not_contain_the_withheld_fields():
    for field in (
        "authorized_users",
        "last_updated_by",
        "application_env_vars",
        "application_kwargs",
    ):
        assert field not in PUBLIC_APP_STATUS_FIELDS, field
    assert "logs" not in PUBLIC_DEPLOYMENT_FIELDS


def test_public_app_status_omits_absent_fields_rather_than_nulling_them():
    assert public_app_status({"status": "NOT_RUNNING", "message": "gone"}) == {
        "status": "NOT_RUNNING",
        "message": "gone",
    }
