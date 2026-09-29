"""An adopted app must not be redeployed with fewer secrets than it is running.

A worker that attaches to a Ray cluster already running apps adopts them rather
than redeploying them, and rebuilds its record of each app from a blob the
running proxy hands back. The builder deliberately keeps credentials out of that
blob — it is readable by anything in the cluster holding a Serve handle — so the
``application_env_vars`` an adopting worker records is the author's dict minus
every ``_``-prefixed entry.

The record therefore held a dict that was present, plausible-looking and
silently reduced. A later ``deploy_app(application_id=…)`` that did not name
env vars inherited that reduced dict and restarted the app with fewer secrets
than its own replicas were running with, and nothing anywhere said so.

The key-set parity tests in ``test_recovered_app_field_parity`` cannot see this:
``application_env_vars`` is present on both paths and is a dict on both, so set
equality holds. This module is the value-level companion, and its fixture
carries ``_``-prefixed keys precisely so there is something to lose.

Measured while writing this: the values are not actually unreachable — the Serve
controller returns each deployment's ``ray_actor_options.runtime_env.env_vars``
verbatim, secrets included. Restoring from there was rejected as the remedy
because it would make the Ray control plane a credential store for the worker,
and because that fetch returns ``{}`` on failure, which would fail back into the
exact silent reduction this closes. The blob names what it dropped instead, and
an update that would be lossy is refused.
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
from bioengine.apps.builder import AppBuilder
from bioengine.apps.manager import AppsManager

WORKSPACE = "bioimage-io"
APP_ID = "nuclei-seg"
ARTIFACT_ID = f"{WORKSPACE}/{APP_ID}"
VERSION = "0.2.4"
NEXT_VERSION = "0.2.5"
ENTRY_ID = "deployment:NucleiSeg"
ENTRY = "NucleiSeg"
WORKER_CLIENT_ID = "worker-1"
CONTEXT = {"user": {"id": "u-1", "email": "u@lab.test"}}

SECRET_KEY = "_MODEL_API_KEY"
SECRET_VALUE = "placeholder-not-a-real-secret"

# What the author deployed with: one plain variable and one secret.
RAW_ENV_VARS = {
    ENTRY: {"MODEL_URL": "https://models.test/nuclei", SECRET_KEY: SECRET_VALUE}
}
PLAIN_ONLY_ENV_VARS = {ENTRY: {"MODEL_URL": "https://models.test/nuclei"}}


# ── the worker-side doubles ──────────────────────────────────────────────────


def _base_manager() -> AppsManager:
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.env_var_loss")
    manager.admin_users = ["*"]
    manager.startup_applications = []

    server = MagicMock()
    server.config.workspace = WORKSPACE
    server.config.client_id = WORKER_CLIENT_ID
    server.config.public_base_url = "https://hypha.test"
    manager.server = server

    ray_cluster = MagicMock()
    ray_cluster.check_connection = AsyncMock()
    manager.ray_cluster = ray_cluster

    manager._deployed_applications = {}
    manager._deployment_lock = asyncio.Lock()
    manager._check_initialized = lambda: None
    manager.artifact_manager = MagicMock()
    manager._check_resources = AsyncMock()
    manager._cancel_deployment_process = AsyncMock()
    manager._deploy_application = AsyncMock()
    manager._warn_if_inherited_version_is_stale = AsyncMock()
    return manager


def _built_app(application_env_vars: dict) -> SimpleNamespace:
    return SimpleNamespace(
        metadata={
            "name": "Nuclei Seg",
            "description": "Segment nuclei.",
            "version": VERSION,
            "source_signature": "0123456789abcdef",
            "application_kwargs": {},
            "application_env_vars": application_env_vars,
            "resources": {"num_cpus": 1},
            "authorized_users": {"*": ["*"]},
            "available_methods": ["segment"],
            "frontend_entry": None,
            "deployed_by_worker_client_id": WORKER_CLIENT_ID,
            "proxy_service_token_issued_at": 1_700_000_000.0,
            "proxy_service_token_ttl_seconds": 3600,
        },
        spec={"entry_id": ENTRY_ID, "classes": {ENTRY_ID: {"qualname": ENTRY}}},
        scaling={},
    )


def _attach_builder(manager: AppsManager, application_env_vars: dict) -> None:
    manager.app_builder = SimpleNamespace(
        build=AsyncMock(return_value=_built_app(application_env_vars))
    )


async def _settle(manager: AppsManager) -> None:
    task = manager._deployed_applications[APP_ID]["deployment_task"]
    await asyncio.gather(task, return_exceptions=True)


# ── a worker that adopted the app ────────────────────────────────────────────


def _app_data(raw_env_vars: dict) -> dict:
    """The blob a running proxy hands back, written by the real sanitiser."""
    kept, redacted = AppBuilder._sanitize_recovery_env_vars(raw_env_vars)
    return {
        "format_version": "0.6.0",
        "worker_workspace": WORKSPACE,
        "deployed_by_worker_client_id": WORKER_CLIENT_ID,
        "proxy_service_token_issued_at": 1_700_000_000.0,
        "proxy_service_token_ttl_seconds": 3600,
        "entry": ENTRY_ID,
        "spec_hash": "abc123",
        "source_signature": "0123456789abcdef",
        "display_name": "Nuclei Seg",
        "description": "Segment nuclei.",
        "frontend_entry": None,
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "application_kwargs": {},
        "application_env_vars": kept,
        "redacted_env_var_keys": redacted,
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


async def _adopted_manager(monkeypatch, raw_env_vars: dict) -> AppsManager:
    """Drive the real ``recover_deployed_applications`` against a live-looking app."""
    manager = _base_manager()

    app_handle = MagicMock()
    app_handle.get_app_data.remote = AsyncMock(return_value=_app_data(raw_env_vars))
    app_handle.update_authorized_users.remote = AsyncMock()

    serve_status = SimpleNamespace(
        applications={
            APP_ID: SimpleNamespace(
                deployments={"ProxyDeployment": MagicMock(), ENTRY: MagicMock()}
            )
        }
    )
    monkeypatch.setattr(
        manager_module,
        "serve",
        SimpleNamespace(
            status=lambda: serve_status,
            get_app_handle=lambda application_id: app_handle,
        ),
    )

    async def call_with_reconnect(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    manager.ray_cluster.call_with_reconnect = call_with_reconnect

    await manager.recover_deployed_applications()
    assert APP_ID in manager._deployed_applications, "adoption did not happen"
    _attach_builder(manager, raw_env_vars)
    return manager


# ── a worker that deployed the app itself ────────────────────────────────────


async def _deploying_manager() -> AppsManager:
    manager = _base_manager()
    _attach_builder(manager, RAW_ENV_VARS)
    manager.ray_cluster.call_with_reconnect = AsyncMock()

    await manager.deploy_app(
        artifact_id=ARTIFACT_ID,
        application_id=APP_ID,
        version=VERSION,
        application_env_vars=RAW_ENV_VARS,
        context=CONTEXT,
    )
    await _settle(manager)
    return manager


# ── the defect, observed at the point it costs something ─────────────────────


@pytest.mark.asyncio
async def test_updating_an_adopted_app_without_env_vars_is_refused(
    monkeypatch,
) -> None:
    # The whole class in one assertion: the adopted record's env var dict is
    # short a secret, and an update that does not name env vars would ship that
    # reduced dict to the builder. Asserting on the record instead would pass —
    # the record is exactly where the reduced value still looks like a value.
    manager = await _adopted_manager(monkeypatch, RAW_ENV_VARS)

    with pytest.raises(ValueError) as excinfo:
        await manager.deploy_app(
            artifact_id=ARTIFACT_ID,
            application_id=APP_ID,
            version=NEXT_VERSION,
            context=CONTEXT,
        )

    assert SECRET_KEY in str(excinfo.value), (
        "The refusal has to name the secrets that are missing, or the caller "
        f"cannot act on it: {excinfo.value}"
    )
    assert manager.app_builder.build.await_count == 0, (
        "Refusing must happen before the build, so the app is never redeployed "
        "with fewer environment variables than its replicas are running."
    )


@pytest.mark.asyncio
async def test_updating_an_adopted_app_that_names_its_secrets_again_is_allowed(
    monkeypatch,
) -> None:
    # The positive control for the refusal above: a blanket "adopted apps can
    # never be updated" would also pass that test.
    manager = await _adopted_manager(monkeypatch, RAW_ENV_VARS)

    await manager.deploy_app(
        artifact_id=ARTIFACT_ID,
        application_id=APP_ID,
        version=NEXT_VERSION,
        application_env_vars=RAW_ENV_VARS,
        context=CONTEXT,
    )
    await _settle(manager)

    built_with = manager.app_builder.build.await_args.kwargs["application_env_vars"]
    assert built_with[ENTRY][SECRET_KEY] == SECRET_VALUE


@pytest.mark.asyncio
async def test_an_adopted_app_that_lost_nothing_still_updates_unprompted(
    monkeypatch,
) -> None:
    # An app that declared no secrets loses nothing to the sanitiser, so its
    # adopted record is complete and inheriting it is correct. The refusal must
    # be scoped to records that are actually short something.
    manager = await _adopted_manager(monkeypatch, PLAIN_ONLY_ENV_VARS)

    await manager.deploy_app(
        artifact_id=ARTIFACT_ID,
        application_id=APP_ID,
        version=NEXT_VERSION,
        context=CONTEXT,
    )
    await _settle(manager)

    built_with = manager.app_builder.build.await_args.kwargs["application_env_vars"]
    assert built_with == PLAIN_ONLY_ENV_VARS


@pytest.mark.asyncio
async def test_updating_a_self_deployed_app_still_inherits_its_secrets() -> None:
    # The path that was never broken has to stay unbroken: a worker that
    # deployed the app holds the raw dict and must keep passing it on.
    manager = await _deploying_manager()

    await manager.deploy_app(
        artifact_id=ARTIFACT_ID,
        application_id=APP_ID,
        version=NEXT_VERSION,
        context=CONTEXT,
    )
    await _settle(manager)

    built_with = manager.app_builder.build.await_args.kwargs["application_env_vars"]
    assert built_with[ENTRY][SECRET_KEY] == SECRET_VALUE


# ── the contract the recovery blob has to satisfy ────────────────────────────


def test_the_recovery_blob_strips_every_secret_and_names_what_it_stripped() -> None:
    # Both halves in one assertion, because either alone is a defect: keeping a
    # secret is a leak into an actor-readable blob, and dropping one without
    # recording the name is the silent reduction this module exists for.
    raw = {
        ENTRY: {
            "MODEL_URL": "https://models.test/nuclei",
            SECRET_KEY: SECRET_VALUE,
            "HYPHA_TOKEN": SECRET_VALUE,
        },
        "Sidecar": {"LOG_LEVEL": "info"},
    }

    kept, redacted = AppBuilder._sanitize_recovery_env_vars(raw)

    assert kept == {
        ENTRY: {"MODEL_URL": "https://models.test/nuclei"},
        "Sidecar": {"LOG_LEVEL": "info"},
    }
    assert redacted == {ENTRY: ["HYPHA_TOKEN", SECRET_KEY]}, (
        "Every key the sanitiser removes must be named, for every class it "
        "removed one from — and classes that lost nothing must not appear."
    )
    assert SECRET_VALUE not in str(redacted), (
        "The names are what travel in the blob; the values must not follow them."
    )


def test_the_builder_writes_the_redacted_key_names_into_the_recovery_blob() -> None:
    # The fixtures above build the blob themselves, so they would still pass
    # against a builder that never writes the key. This is the write side.
    source = textwrap.dedent(inspect.getsource(builder_module.AppBuilder.build))
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Dict)
            and any(getattr(t, "id", None) == "app_data" for t in node.targets)
        ):
            keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
            break
    else:
        raise AssertionError("no `app_data = {...}` literal in AppBuilder.build")

    assert "redacted_env_var_keys" in keys, (
        "Without this key in the blob an adopting worker cannot tell a reduced "
        "env var dict from a complete one, and the refusal never fires."
    )
