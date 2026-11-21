"""What ``deploy_app`` actually STORES for ``auto_redeploy``, by running it.

Every other arm for this setting is an AST or source-text assertion over ``deploy_app``. Those
constrain the shape of the code; none of them observes the value that ends up in
``self._deployed_applications[application_id]["auto_redeploy"]``, which is what the monitor loop
later reads and what ``get_app_status`` publishes. Two kinds of edit satisfy every static arm
while storing the wrong thing:

  * a write by a route that does not name the variable, e.g. a computed subscript key
  * anything that changes the value between the guard and the record

So drive the real method against stubs and read the record back, for the three cases that differ:

    omitted on a NEW application     -> True   (the default under test)
    explicit False on a NEW one      -> False  (the opt-out)
    omitted on an UPDATE             -> whatever the running application had (inherit)

Only the collaborators ``deploy_app`` actually reaches are stubbed -- the resource check, the
builder, the deployment task and a few id helpers -- so the guard, the record construction and
everything between them is the production code.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps.manager import AppsManager

APP_ID = "stub-app"
ARTIFACT = "bioimage-io/stub-artifact"
VERSION = "1.2.3"
CONTEXT = {"user": {"id": "tester", "email": "tester@example.org"}}


def _built_app():
    """The minimum a BuiltApp needs to get through the record construction.

    The keys below are the ones ``deploy_app`` reads by SUBSCRIPT, where a missing key is a
    ``KeyError``; the three it reads with ``.get`` are optional and omitted. Because the set is
    exactly the subscripted ones, a key added to the record later fails this stub loudly rather
    than being quietly absent.
    """
    return SimpleNamespace(
        metadata={
            "name": "Stub App",
            "description": "a stub",
            "version": VERSION,
            "resources": {"num_cpus": 1, "num_gpus": 0, "memory": 1 << 30},
            "application_kwargs": {},
            "application_env_vars": {},
            "authorized_users": ["*"],
            "available_methods": ["ping"],
            "deployed_by_worker_client_id": "stub-worker",
            "proxy_service_token_issued_at": 1.0,
            "proxy_service_token_ttl_seconds": 3600,
            "source_signature": "sig-1",
            # frontend_entry is absent on purpose: with it set, deploy_app calls
            # get_static_site_url, which wants a real server config.
        },
        spec={"entry_id": "entry:EntryApp", "classes": {}},
        scaling={},
        runtime_env={},
        replica_env_vars={},
    )


def _manager() -> AppsManager:
    m = object.__new__(AppsManager)
    m.logger = MagicMock()
    m.admin_users = ["tester"]
    m._deployed_applications = {}
    m._deployment_lock = asyncio.Lock()
    m._check_initialized = MagicMock()
    m._get_full_artifact_id = MagicMock(
        side_effect=lambda artifact_id, **_: artifact_id
    )
    m._generate_application_id = MagicMock(return_value=APP_ID)
    m._check_resources = AsyncMock(return_value=None)
    m._deploy_application = AsyncMock()
    m._cancel_deployment_process = AsyncMock()
    m._warn_if_inherited_version_is_stale = MagicMock()

    builder = MagicMock()
    builder.build = AsyncMock(side_effect=lambda **kw: _built_app())
    m.app_builder = builder

    ray_cluster = MagicMock()
    ray_cluster.check_connection = AsyncMock()
    ray_cluster.call_with_reconnect = AsyncMock()
    m.ray_cluster = ray_cluster

    m.server = SimpleNamespace(
        config=SimpleNamespace(
            workspace="ws-stub", client_id="stub-worker", user={"id": "tester"}
        )
    )
    return m


async def _deploy(manager, **overrides):
    kwargs = dict(
        artifact_id=ARTIFACT,
        version=VERSION,
        application_id=APP_ID,
        application_kwargs=None,
        application_env_vars=None,
        hypha_token=None,
        disable_gpu=None,
        max_ongoing_requests=None,
        proxy_memory_in_gb=None,
        auto_redeploy=None,
        debug=None,
        ice_servers=None,
        authorized_users=None,
        scaling=None,
        context=CONTEXT,
    )
    kwargs.update(overrides)
    return await manager.deploy_app(**kwargs)


def _stored(manager, app_id=APP_ID):
    """Read the field exactly where the monitor loop and get_app_status read it."""
    entry = manager._deployed_applications.get(app_id)
    assert entry is not None, (
        f"deploy_app stored no record for '{app_id}', so this arm would be vacuous -- the "
        f"records present are {sorted(manager._deployed_applications)}"
    )
    # Sentinel, not .get(None): an ABSENT key and a stored False are indistinguishable
    # through .get(), and False is the value this arm is about.
    return entry.get("auto_redeploy", "<ABSENT>")


@pytest.mark.asyncio
async def test_a_NEW_application_with_the_argument_OMITTED_stores_TRUE():
    manager = _manager()
    await _deploy(manager)
    stored = _stored(manager)
    assert stored is True, (
        f"a new application deployed with auto_redeploy omitted stored {stored!r}; the monitor "
        f"loop reads this field, so anything but True means the default did not take effect"
    )


@pytest.mark.asyncio
async def test_a_NEW_application_with_an_explicit_FALSE_stores_FALSE():
    manager = _manager()
    await _deploy(manager, auto_redeploy=False)
    stored = _stored(manager)
    assert stored is False, (
        f"an explicit auto_redeploy=False stored {stored!r}; the opt-out is what makes the new "
        f"default acceptable, and this is the value that enforces it"
    )


@pytest.mark.asyncio
async def test_an_UPDATE_that_OMITS_the_argument_stores_THE_RUNNING_VALUE():
    """Both directions, because inheriting only one of them is the same bug twice."""
    for running in (True, False):
        manager = _manager()
        await _deploy(manager, auto_redeploy=running)
        assert _stored(manager) is running

        await _deploy(manager)  # update, argument omitted
        stored = _stored(manager)
        assert stored is running, (
            f"an update omitting auto_redeploy stored {stored!r} for an application that was "
            f"running with {running!r}; an omitted argument on an update must preserve, not "
            f"re-default"
        )


@pytest.mark.asyncio
async def test_an_UPDATE_with_an_EXPLICIT_value_overrides_the_running_one():
    """The control for the arm above: inherit-on-omitted must not become ignore-the-argument."""
    manager = _manager()
    await _deploy(manager, auto_redeploy=True)
    assert _stored(manager) is True
    await _deploy(manager, auto_redeploy=False)
    stored = _stored(manager)
    assert stored is False, (
        f"an update passing auto_redeploy=False stored {stored!r}; the update path is "
        f"preserving the old value instead of honouring an explicit one"
    )


@pytest.mark.asyncio
async def test_the_three_cases_do_not_all_store_the_same_value():
    """Instrument control. Each arm above asserts one value; if the stub somehow stored a
    constant, each would still pass or fail on its own without saying so. Require the cases
    to DIFFER, which is what proves the argument reaches the record at all."""
    seen = []
    for kwargs in ({}, {"auto_redeploy": False}, {"auto_redeploy": True}):
        manager = _manager()
        await _deploy(manager, **kwargs)
        seen.append(_stored(manager))
    assert len(set(seen)) >= 2, (
        f"every case stored the same value {seen!r}, so this file cannot distinguish the "
        f"argument having an effect from it having none"
    )
