"""An adopted app must be tracked with the same fields as a deployed one.

A worker that restarts against a Ray cluster it does not own adopts the apps
still running there instead of redeploying them. Adoption rebuilds the
worker's record of the app from a blob the running proxy hands back, and that
rebuild used to populate fewer fields than a deploy does — silently, because
nothing compared the two. The visible symptom was an app whose reported
``running_version`` and ``version_verified`` were null while the app itself was
healthy, so a monitor keying on either read a fine app as unknown-version. An
earlier instance of the same shape was a missing auth token.

These are two instances of one class, so the assertion here is set-equality
over the whole record rather than a check of the fields that happened to be
noticed. Fields an adopting worker genuinely cannot know are listed in
``LEGITIMATELY_NULL_AFTER_ADOPTION`` with the reason; they are still *present*,
so the set-equality assertion keeps holding and a newly missing field fails.

What set-equality cannot see is a key that is present on both paths but whose
*value* adoption only partly reconstructs: the builder strips secrets from the
blob, so ``application_env_vars`` comes back without the author's ``_``-prefixed
entries and every assertion here still passes. Closing that needs a value-level
check, not this one.

The adoption path is only reachable in external-cluster mode — single-machine
and SLURM workers tear down the Ray head on cleanup, so nothing survives to be
adopted. A unit test over the path is therefore the only coverage available.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import logging
import textwrap
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps import builder as builder_module
from bioengine.apps import manager as manager_module
from bioengine.apps.manager import AppsManager

WORKSPACE = "bioimage-io"
APP_ID = "nuclei-seg"
ARTIFACT_ID = f"{WORKSPACE}/{APP_ID}"
VERSION = "0.2.4"
ENTRY_ID = "deployment:NucleiSeg"
ENTRY = "NucleiSeg"
WORKER_CLIENT_ID = "worker-1"
CONTEXT = {"user": {"id": "u-1", "email": "u@lab.test"}}

# Values an adopting worker cannot reconstruct. Each stays None on a recovered
# record; the key itself must still exist so the parity assertion stays honest.
LEGITIMATELY_NULL_AFTER_ADOPTION = {
    # The builder strips secrets from the recovery blob, so the token the app
    # was deployed with is simply not retrievable from the running app.
    "hypha_token",
    # The built Ray Serve application object lives in the worker process that
    # deployed it; an adopting worker cannot reconstruct it, which is why
    # auto-redeploy skips adopted apps.
    "built_app",
    # A ProxyDeployment constructor argument rather than part of the recovery
    # blob, and any TURN credential in it would have expired. None means
    # "fetch a fresh list", which is what a later redeploy wants anyway.
    "ice_servers",
}


# ── the cluster both managers see ────────────────────────────────────────────


def _instance_details() -> dict:
    return {
        "applications": {
            APP_ID: {
                "status": "RUNNING",
                "message": "",
                "deployments": {
                    ENTRY: {
                        "status": "HEALTHY",
                        "replicas": [{"replica_id": "r1", "state": "RUNNING"}],
                    }
                },
            }
        }
    }


def _replica_identities() -> dict:
    return {ENTRY: {"r1": {"version": VERSION, "artifact_id": ARTIFACT_ID}}}


def _base_manager() -> AppsManager:
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.field_parity")
    manager.admin_users = ["*"]
    manager.startup_applications = []

    server = MagicMock()
    server.config.workspace = WORKSPACE
    server.config.client_id = WORKER_CLIENT_ID
    server.config.public_base_url = "https://hypha.test"
    manager.server = server

    ray_cluster = MagicMock()
    ray_cluster.check_connection = AsyncMock()
    proxy = ray_cluster.proxy_actor_handle
    proxy.get_serve_instance_details.remote = AsyncMock(
        return_value=_instance_details()
    )
    proxy.get_deployment_replicas.remote = AsyncMock(return_value=["r0"])
    proxy.get_service_registration.remote = AsyncMock(return_value=True)
    proxy.get_replica_identities.remote = AsyncMock(return_value=_replica_identities())
    proxy.get_deployment_logs.remote = AsyncMock(return_value={})
    manager.ray_cluster = ray_cluster

    manager._deployed_applications = {}
    return manager


# ── a manager that deployed the app ──────────────────────────────────────────


def _built_app() -> SimpleNamespace:
    return SimpleNamespace(
        metadata={
            "name": "Nuclei Seg",
            "description": "Segment nuclei.",
            "version": VERSION,
            "application_kwargs": {},
            "application_env_vars": {},
            "resources": {"num_cpus": 1},
            "authorized_users": {"*": ["*"]},
            "available_methods": ["segment"],
            "frontend_entry": "frontend/index.html",
            "deployed_by_worker_client_id": WORKER_CLIENT_ID,
            "proxy_service_token_issued_at": 1_700_000_000.0,
            "proxy_service_token_ttl_seconds": 3600,
        },
        spec={
            "entry_id": ENTRY_ID,
            "classes": {ENTRY_ID: {"qualname": ENTRY}},
        },
        scaling={},
    )


async def _deployed_manager(**deploy_kwargs) -> AppsManager:
    """Drive the real ``deploy_app`` until it has recorded the application."""
    manager = _base_manager()
    manager._deployment_lock = asyncio.Lock()
    manager._check_initialized = lambda: None
    manager.artifact_manager = MagicMock()
    manager.app_builder = SimpleNamespace(build=AsyncMock(return_value=_built_app()))
    manager._check_resources = AsyncMock()
    manager._cancel_deployment_process = AsyncMock()
    manager._deploy_application = AsyncMock()
    manager.ray_cluster.call_with_reconnect = AsyncMock()

    await manager.deploy_app(
        artifact_id=ARTIFACT_ID,
        application_id=APP_ID,
        version=VERSION,
        context=CONTEXT,
        **deploy_kwargs,
    )
    task = manager._deployed_applications[APP_ID]["deployment_task"]
    await asyncio.gather(task, return_exceptions=True)
    return manager


# ── a manager that adopted the app ───────────────────────────────────────────


def _app_data() -> dict:
    """The blob a running ProxyDeployment hands back, as the builder writes it."""
    return {
        "format_version": "0.6.0",
        "worker_workspace": WORKSPACE,
        "deployed_by_worker_client_id": WORKER_CLIENT_ID,
        "proxy_service_token_issued_at": 1_700_000_000.0,
        "proxy_service_token_ttl_seconds": 3600,
        "entry": ENTRY_ID,
        "spec_hash": "abc123",
        "display_name": "Nuclei Seg",
        "description": "Segment nuclei.",
        "frontend_entry": "frontend/index.html",
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "application_kwargs": {},
        "application_env_vars": {},
        "disable_gpu": False,
        "max_ongoing_requests": 10,
        "proxy_memory_in_gb": 0.5,
        "scaling": {},
        "application_resources": {"num_cpus": 1},
        "authorized_users": {"*": ["*"]},
        "available_methods": ["segment"],
        "started_at": 1_700_000_000.0,
        "last_updated_at": 1_700_000_000.0,
        "last_updated_by": "u-1",
        "auto_redeploy": False,
        "debug": False,
    }


async def _recovered_manager(monkeypatch, app_data: dict | None = None) -> AppsManager:
    """Drive the real ``recover_deployed_applications`` against a live-looking app."""
    manager = _base_manager()

    app_handle = MagicMock()
    app_handle.get_app_data.remote = AsyncMock(
        return_value=_app_data() if app_data is None else app_data
    )
    app_handle.update_authorized_users.remote = AsyncMock()

    serve_status = SimpleNamespace(
        applications={
            APP_ID: SimpleNamespace(
                deployments={"ProxyDeployment": MagicMock(), ENTRY: MagicMock()}
            )
        }
    )
    fake_serve = SimpleNamespace(
        status=lambda: serve_status,
        get_app_handle=lambda application_id: app_handle,
    )
    monkeypatch.setattr(manager_module, "serve", fake_serve)

    async def call_with_reconnect(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    manager.ray_cluster.call_with_reconnect = call_with_reconnect

    await manager.recover_deployed_applications()
    return manager


# ── the class-closing assertion ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_adopted_app_is_tracked_with_the_same_fields_as_a_deployed_one(
    monkeypatch,
) -> None:
    deployed = await _deployed_manager()
    recovered = await _recovered_manager(monkeypatch)

    deployed_fields = set(deployed._deployed_applications[APP_ID])
    recovered_fields = set(recovered._deployed_applications[APP_ID])

    assert recovered_fields == deployed_fields, (
        "Adoption must record every field a deploy records. Missing from "
        f"adoption: {sorted(deployed_fields - recovered_fields)}; extra: "
        f"{sorted(recovered_fields - deployed_fields)}. A field only one path "
        "writes is read as 'unknown' on the other, with nothing to flag it."
    )


@pytest.mark.asyncio
async def test_only_the_documented_fields_are_null_after_adoption(
    monkeypatch,
) -> None:
    # The set-equality above is satisfied by writing None everywhere, so pin
    # which fields may legitimately be None — anything else going null is the
    # same defect wearing a different hat.
    recovered = await _recovered_manager(monkeypatch)
    record = recovered._deployed_applications[APP_ID]

    nulled = {
        key
        for key, value in record.items()
        if value is None and key not in {"deployment_task", "undeployment_task"}
    }

    assert nulled == LEGITIMATELY_NULL_AFTER_ADOPTION, (
        "Fields that go null on adoption must each have a recorded reason. "
        f"Unexplained: {sorted(nulled - LEGITIMATELY_NULL_AFTER_ADOPTION)}; "
        f"now recoverable: "
        f"{sorted(LEGITIMATELY_NULL_AFTER_ADOPTION - nulled)}."
    )


# ── what the class cost, observed through the API ────────────────────────────


@pytest.mark.asyncio
async def test_status_of_an_adopted_app_is_null_wherever_a_deployed_one_is(
    monkeypatch,
) -> None:
    deployed = await _deployed_manager()
    recovered = await _recovered_manager(monkeypatch)

    deployed_status = (
        await deployed.get_app_status(application_ids=[APP_ID], context=CONTEXT)
    )[APP_ID]
    recovered_status = (
        await recovered.get_app_status(application_ids=[APP_ID], context=CONTEXT)
    )[APP_ID]

    assert set(recovered_status) == set(deployed_status)

    deployed_nulls = {k for k, v in deployed_status.items() if v is None}
    recovered_nulls = {k for k, v in recovered_status.items() if v is None}
    assert recovered_nulls == deployed_nulls, (
        "A status field that carries a value for a deployed app and None for an "
        "adopted one reads as 'cannot tell' on an app that is fine. Extra nulls "
        f"after adoption: {sorted(recovered_nulls - deployed_nulls)}."
    )


@pytest.mark.asyncio
async def test_an_adopted_app_reports_the_version_its_replicas_are_running(
    monkeypatch,
) -> None:
    # The reported symptom, asserted on the values rather than on presence.
    recovered = await _recovered_manager(monkeypatch)

    status = (
        await recovered.get_app_status(application_ids=[APP_ID], context=CONTEXT)
    )[APP_ID]

    assert status["recovered_app"] is True
    assert status["running_version"] == VERSION
    assert status["version_verified"] is True


@pytest.mark.asyncio
async def test_a_stale_replica_is_still_detected_on_an_adopted_app(
    monkeypatch,
) -> None:
    # Verification must be a real check, not a constant True: a replica baked
    # at another version has to read as unverified here too.
    recovered = await _recovered_manager(monkeypatch)
    recovered.ray_cluster.proxy_actor_handle.get_replica_identities.remote = AsyncMock(
        return_value={ENTRY: {"r1": {"version": "0.1.0"}}}
    )

    status = (
        await recovered.get_app_status(application_ids=[APP_ID], context=CONTEXT)
    )[APP_ID]

    assert status["running_version"] == "0.1.0"
    assert status["version_verified"] is False


@pytest.mark.asyncio
async def test_an_adopted_app_keeps_its_static_site_link(monkeypatch) -> None:
    # Same class again: whether the app has a frontend was not in the recovery
    # blob, so an adopted app lost the URL the dashboard links its UI from.
    # Read this with the AST test below, not alone: ``_app_data()`` hardcodes
    # ``frontend_entry``, so this covers only the read side and still passes
    # against a builder that never writes the key. The AST test is what fails
    # on the write side.
    recovered = await _recovered_manager(monkeypatch)

    status = (
        await recovered.get_app_status(application_ids=[APP_ID], context=CONTEXT)
    )[APP_ID]

    assert status["static_site_url"], (
        "An adopted app with a frontend must still advertise its static site."
    )
    assert APP_ID in status["static_site_url"]


@pytest.mark.asyncio
async def test_adoption_refuses_a_blob_without_an_entry_id(
    monkeypatch, caplog
) -> None:
    # The entry id is what names the Serve deployment whose replica reports the
    # running version. A blob without it cannot be adopted into a complete
    # record, so it must be refused — and say which key was missing, rather
    # than surfacing a bare KeyError.
    blob = _app_data()
    del blob["entry"]

    with caplog.at_level(logging.WARNING, logger="test.field_parity"):
        recovered = await _recovered_manager(monkeypatch, app_data=blob)

    assert APP_ID not in recovered._deployed_applications
    assert any(
        "Missing required app_data keys" in r.getMessage()
        and "entry" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


# ── the contract the recovery blob has to satisfy ────────────────────────────


def _app_data_keys_written_by_the_builder() -> set[str]:
    source = textwrap.dedent(inspect.getsource(builder_module.AppBuilder.build))
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Dict)
            and any(getattr(t, "id", None) == "app_data" for t in node.targets)
        ):
            return {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant)
            }
    raise AssertionError("no `app_data = {...}` literal in AppBuilder.build")


def _app_data_keys_read_by_recovery() -> set[str]:
    source = textwrap.dedent(
        inspect.getsource(AppsManager.recover_deployed_applications)
    )
    keys: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Subscript)
            and getattr(node.value, "id", None) == "app_data"
            and isinstance(node.slice, ast.Constant)
        ):
            keys.add(node.slice.value)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and getattr(node.func.value, "id", None) == "app_data"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            keys.add(node.args[0].value)
        elif (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Set)
            and any(getattr(t, "id", None) == "required_keys" for t in node.targets)
        ):
            keys |= {
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant)
            }
    return keys


def test_every_key_adoption_reads_is_one_the_builder_writes() -> None:
    # The blob is the whole contract between a deploying worker and an
    # adopting one, and a key read but never written fails silently: the
    # ``.get`` returns None and the adopted app is quietly missing a value.
    # That is how an adopted app stopped advertising its frontend.
    written = _app_data_keys_written_by_the_builder()
    read = _app_data_keys_read_by_recovery()

    assert read <= written, (
        "Recovery reads app_data keys the builder never writes: "
        f"{sorted(read - written)}. Either write them at build time or stop "
        "reading them — a key that is only ever read reads as None forever."
    )


@pytest.mark.asyncio
async def test_an_update_keeps_debug_on_without_being_asked_again() -> None:
    # The same class on the deploy side: debug was never recorded, so an update
    # that did not name it silently turned it off.
    manager = await _deployed_manager(debug=True)
    assert manager._deployed_applications[APP_ID]["debug"] is True

    await manager.deploy_app(
        artifact_id=ARTIFACT_ID,
        application_id=APP_ID,
        version=VERSION,
        context=CONTEXT,
    )
    task = manager._deployed_applications[APP_ID]["deployment_task"]
    await asyncio.gather(task, return_exceptions=True)

    assert manager.app_builder.build.await_args.kwargs["debug"] is True
    assert manager._deployed_applications[APP_ID]["debug"] is True
