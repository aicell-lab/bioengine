"""Pin the three holes through which a stale actor kept serving.

The worker's guard against a reused Ray Serve replica was a version-string
comparison, made only for apps this worker had itself built. That left three
ways for an actor running old code to be reported as healthy:

1. A version string cannot see content that changed underneath it. Re-staging
   the same version with different files took the in-place ``serve.run`` path,
   which may reuse the replica, and nothing compared what the replica had
   actually loaded.
2. ``_verify_running_identities`` returned early when the app carried no
   ``built_app`` — exactly the apps recovered from a previous worker, which are
   the most likely to be running code the version pin no longer describes.
3. The recovered app that *was* found stale was deleted anyway, even though the
   redeploy path has nothing to resubmit for it, so the delete stranded it.

The counterpart constraint: a content mismatch is reported, never acted on. If
the two hashes ever disagreed systematically, deleting on it would turn the
monitor into a redeploy loop against a healthy app.

Closing hole 1 also moves *when* the pre-redeploy ``serve.delete`` happens: only
the build knows the content hash of the files it synced, so the delete has to
run after it. That reorder crosses an error boundary — a deploy whose build
fails now leaves the previously-healthy app untouched instead of deleted — so
the ordering is pinned here too.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine._app.bootstrap import hash_source_tree
from bioengine.apps.manager import AppsManager

APP_ID = "model-runner"
VERSION = "2.8.0"
SIGNATURE = "0123456789abcdef"


def _built_app(entry_deployment: str = "ModelRunner"):
    return SimpleNamespace(
        spec={"entry_id": "cid0", "classes": {"cid0": {"qualname": entry_deployment}}}
    )


def _details(*replicas: tuple[str, str, str]) -> dict:
    """Serve instance details for one app: (deployment, replica_id, state)."""
    deployments: dict = {}
    for deployment, replica_id, state in replicas:
        deployments.setdefault(deployment, {"replicas": []})["replicas"].append(
            {"replica_id": replica_id, "state": state}
        )
    return {"deployments": deployments}


def _make_manager(
    *,
    built_app=_built_app(),
    source_signature=SIGNATURE,
    entry_deployment_name: str | None = "ModelRunner",
) -> AppsManager:
    is_deployed = asyncio.Event()
    is_deployed.set()

    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.holes")
    manager._redeploy_backoff = {}
    manager._identity_warned = set()
    # Bookkeeping the monitor keeps for its other branches; empty is the
    # nothing-pending state these tests want.
    manager._missing_from_status = {}
    manager._deleted_pending_redeploy = set()
    manager._deployed_applications = {
        APP_ID: {
            "is_deployed": is_deployed,
            "artifact_id": "bioimage-io/model-runner",
            "version": VERSION,
            "source_signature": source_signature,
            "auto_redeploy": True,
            # Both the deploy and the adoption path store this, so the entry
            # deployment is nameable without a spec.
            "entry_deployment_name": entry_deployment_name,
            "built_app": built_app,
            "deployment_task": None,
        }
    }

    ray_cluster = MagicMock()
    ray_cluster.check_connection = AsyncMock()
    ray_cluster.proxy_actor_handle.get_serve_instance_details.remote = AsyncMock(
        return_value={}
    )
    ray_cluster.proxy_actor_handle.get_replica_identities.remote = AsyncMock(
        return_value={}
    )
    manager.ray_cluster = ray_cluster
    manager._deploy_application = AsyncMock()

    return manager


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


# ───────────────────── hole 2: verification without a spec ─────────────────────


@pytest.mark.asyncio
async def test_a_recovered_app_is_verified_even_without_a_built_app() -> None:
    # The whole point: an app that survived a worker restart has no built_app,
    # and used to skip verification entirely. Its replicas are precisely the
    # ones that have been alive longest.
    manager = _make_manager(built_app=None)
    manager.ray_cluster.proxy_actor_handle.get_replica_identities.remote.return_value = {
        "ModelRunner": {"r1": {"version": "2.7.0", "code_hash": SIGNATURE}}
    }

    running_version, verified, _ = await manager._verify_running_identities(
        APP_ID, _details(("ModelRunner", "r1", "RUNNING")), VERSION
    )

    assert verified is False, (
        "A recovered app running the wrong version must be detectable; "
        "returning None here is what let it serve stale code unnoticed."
    )
    # The entry deployment is named from the stored entry_deployment_name, not
    # from the absent spec, so a recovered app reports its running version too.
    assert running_version == "2.7.0"


@pytest.mark.asyncio
async def test_a_stale_replica_of_a_non_entry_deployment_fails_the_check() -> None:
    # Serve may reuse a replica of any deployment, not just the entry one.
    manager = _make_manager()
    manager.ray_cluster.proxy_actor_handle.get_replica_identities.remote.return_value = {
        "ModelRunner": {"r1": {"version": VERSION, "code_hash": SIGNATURE}},
        "Helper": {"r2": {"version": "2.7.0", "code_hash": SIGNATURE}},
    }

    running_version, verified, _ = await manager._verify_running_identities(
        APP_ID,
        _details(("ModelRunner", "r1", "RUNNING"), ("Helper", "r2", "RUNNING")),
        VERSION,
    )

    assert running_version == VERSION  # the entry replica alone looks fine
    assert verified is False


@pytest.mark.asyncio
async def test_nothing_to_check_yields_unknown_not_verified() -> None:
    # A replica that has not yet pushed its identity must not read as proof.
    manager = _make_manager()

    _, verified, code_verified = await manager._verify_running_identities(
        APP_ID, _details(("ModelRunner", "r1", "STARTING")), VERSION
    )

    assert verified is None
    assert code_verified is None


# ───────────────────── hole 1: same version, different files ─────────────────────


@pytest.mark.asyncio
async def test_a_restaged_version_is_caught_by_content_not_by_version() -> None:
    manager = _make_manager()
    manager.ray_cluster.proxy_actor_handle.get_replica_identities.remote.return_value = {
        "ModelRunner": {"r1": {"version": VERSION, "code_hash": "deadbeefdeadbeef"}}
    }

    _, verified, code_verified = await manager._verify_running_identities(
        APP_ID, _details(("ModelRunner", "r1", "RUNNING")), VERSION
    )

    assert verified is True, "The version string genuinely matches — that is the trap."
    assert code_verified is False


@pytest.mark.asyncio
async def test_matching_content_verifies() -> None:
    manager = _make_manager()
    manager.ray_cluster.proxy_actor_handle.get_replica_identities.remote.return_value = {
        "ModelRunner": {"r1": {"version": VERSION, "code_hash": SIGNATURE}}
    }

    _, verified, code_verified = await manager._verify_running_identities(
        APP_ID, _details(("ModelRunner", "r1", "RUNNING")), VERSION
    )

    assert (verified, code_verified) == (True, True)


@pytest.mark.asyncio
async def test_an_app_from_an_older_worker_has_no_signature_to_compare() -> None:
    # An app recovered from a worker released before source signatures existed
    # carries none. That must read as unknown, not as a mismatch, or every one
    # of them would warn on the first tick after an upgrade.
    manager = _make_manager(built_app=None, source_signature=None)
    manager.ray_cluster.proxy_actor_handle.get_replica_identities.remote.return_value = {
        "ModelRunner": {"r1": {"version": VERSION, "code_hash": "deadbeefdeadbeef"}}
    }

    _, verified, code_verified = await manager._verify_running_identities(
        APP_ID, _details(("ModelRunner", "r1", "RUNNING")), VERSION
    )

    assert verified is True
    assert code_verified is None


# ───────────────────── hole 3: what the monitor does about it ─────────────────────


def _status(state: str = "RUNNING"):
    return SimpleNamespace(
        applications={APP_ID: SimpleNamespace(status=SimpleNamespace(value=state))}
    )


def _wire_monitor(manager: AppsManager, verdict: tuple, state: str = "RUNNING") -> list:
    """Route serve.status through call_with_reconnect; record serve.delete calls."""
    deletes: list = []

    async def call_with_reconnect(fn, *args):
        name = getattr(fn, "__name__", "")
        if name == "delete":
            deletes.append(args[0])
            return None
        return _status(state)

    manager.ray_cluster.call_with_reconnect = AsyncMock(side_effect=call_with_reconnect)
    manager._verify_running_identities = AsyncMock(return_value=verdict)
    return deletes


@pytest.mark.asyncio
async def test_a_recovered_app_with_a_version_mismatch_is_reported_not_deleted(
    caplog,
) -> None:
    manager = _make_manager(built_app=None)
    deletes = _wire_monitor(manager, (None, False, None))

    with caplog.at_level(logging.WARNING, logger="test.holes"):
        await manager.monitor_applications()

    assert deletes == [], (
        "Deleting a recovered app strands it: _fire_redeploy has no built "
        "application to resubmit, so nothing would bring it back."
    )
    assert "recovered from a previous worker" in _warnings(caplog)[0]


@pytest.mark.asyncio
async def test_a_managed_app_with_a_version_mismatch_is_still_deleted() -> None:
    # The pre-existing self-heal must not regress: an app this worker built can
    # be deleted, because the next tick redeploys it from the cached built_app.
    manager = _make_manager()
    deletes = _wire_monitor(manager, (None, False, None))

    await manager.monitor_applications()

    assert deletes == [APP_ID]


@pytest.mark.asyncio
async def test_a_content_mismatch_warns_but_never_deletes(caplog) -> None:
    manager = _make_manager()
    deletes = _wire_monitor(manager, (VERSION, True, False))

    with caplog.at_level(logging.WARNING, logger="test.holes"):
        await manager.monitor_applications()

    assert deletes == [], (
        "A hash disagreement that turned out to be systematic would delete the "
        "app on every tick. Report only."
    )
    assert "different source content" in _warnings(caplog)[0]


@pytest.mark.asyncio
async def test_the_identity_warning_does_not_repeat_every_tick(caplog) -> None:
    # The condition persists until a human redeploys and the monitor ticks every
    # ~10 s, so without de-duplication this is one line per tick for the life of
    # the worker.
    manager = _make_manager()
    _wire_monitor(manager, (VERSION, True, False))

    with caplog.at_level(logging.WARNING, logger="test.holes"):
        for _ in range(5):
            await manager.monitor_applications()

    assert len(_warnings(caplog)) == 1


@pytest.mark.asyncio
async def test_the_warning_rearms_once_the_app_verifies_again(caplog) -> None:
    # Suppression may not be permanent: a later, genuinely new mismatch after a
    # clean period has to be visible.
    manager = _make_manager()
    _wire_monitor(manager, (VERSION, True, False))

    with caplog.at_level(logging.WARNING, logger="test.holes"):
        await manager.monitor_applications()
        manager._verify_running_identities = AsyncMock(return_value=(VERSION, True, True))
        await manager.monitor_applications()
        manager._verify_running_identities = AsyncMock(
            return_value=(VERSION, True, False)
        )
        await manager.monitor_applications()

    assert len(_warnings(caplog)) == 2


@pytest.mark.asyncio
async def test_a_recovered_app_is_never_auto_redeployed(caplog) -> None:
    # _deploy_application would only raise on the None built_app; say so once
    # per attempt instead of scheduling a task that cannot succeed.
    manager = _make_manager(built_app=None)
    _wire_monitor(manager, (None, None, None), state="UNHEALTHY")

    with caplog.at_level(logging.WARNING, logger="test.holes"):
        await manager.monitor_applications()
        await asyncio.sleep(0)

    manager._deploy_application.assert_not_awaited()
    assert "skipping auto-redeploy" in _warnings(caplog)[0]


@pytest.mark.asyncio
async def test_undeploying_clears_the_warning_marker() -> None:
    # Otherwise an application_id redeployed under the same name would inherit
    # the previous instance's suppression and never warn.
    manager = _make_manager()
    manager._identity_warned.add(APP_ID)
    _wire_monitor(manager, (VERSION, True, True))

    await manager.monitor_applications()

    assert APP_ID not in manager._identity_warned


# ───────────────────── the delete must happen after the build ─────────────────────


class _StopAfterDelete(Exception):
    """Sentinel: deploy_app got past the delete decision, stop it going further."""


def _make_deploy_manager(build_fails: bool) -> tuple[AppsManager, list]:
    """Drive ``deploy_app``'s update branch and record build/delete order."""
    events: list = []

    is_deployed = asyncio.Event()
    is_deployed.set()

    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.holes")
    manager.server = MagicMock()
    manager.admin_users = ["u-1"]
    manager._deployment_lock = asyncio.Lock()
    manager._check_initialized = lambda: None
    manager._deployed_applications = {
        APP_ID: {
            "started_at": 0.0,
            "is_deployed": is_deployed,
            "version": VERSION,
            "source_signature": SIGNATURE,
            "artifact_id": "bioimage-io/model-runner",
            "application_kwargs": {},
            "application_env_vars": {},
            "hypha_token": "tok",
            "disable_gpu": False,
            "max_ongoing_requests": 10,
            "proxy_memory_in_gb": 0.5,
            "auto_redeploy": False,
            "debug": False,
            "scaling": {},
        }
    }

    async def call_with_reconnect(fn, *args):
        if getattr(fn, "__name__", "") == "delete":
            events.append(("delete", args[0]))
        return None

    manager.ray_cluster = MagicMock()
    manager.ray_cluster.check_connection = AsyncMock()
    manager.ray_cluster.call_with_reconnect = AsyncMock(side_effect=call_with_reconnect)
    manager._cancel_deployment_process = AsyncMock()

    async def build(**_kwargs):
        events.append(("build", None))
        if build_fails:
            raise RuntimeError("artifact 2.9.0 does not import")
        return SimpleNamespace(
            spec={"classes": {"cid0": {"qualname": "ModelRunner"}}},
            metadata={
                "version": "2.9.0",
                "source_signature": "fedcba9876543210",
                "resources": {},
            },
        )

    manager.app_builder = MagicMock()
    manager.app_builder.build = AsyncMock(side_effect=build)

    async def _check_resources(**_kwargs):
        raise _StopAfterDelete

    manager._check_resources = _check_resources

    return manager, events


DEPLOY_CONTEXT = {"user": {"id": "u-1", "email": "u@lab.test"}}


@pytest.mark.asyncio
async def test_a_version_bump_deletes_the_old_app_only_after_the_build() -> None:
    manager, events = _make_deploy_manager(build_fails=False)

    with pytest.raises(_StopAfterDelete):
        await manager.deploy_app(
            artifact_id="bioimage-io/model-runner",
            application_id=APP_ID,
            version="2.9.0",
            context=DEPLOY_CONTEXT,
        )

    assert events == [("build", None), ("delete", APP_ID)], (
        "The delete still has to happen — it is what frees the replica slot — "
        "but only once the build has produced the signature it is keyed on."
    )


@pytest.mark.asyncio
async def test_a_failed_build_leaves_the_running_app_alone() -> None:
    # Deleting first meant a version bump whose build then failed took the
    # previously-healthy app down with it, and nothing redeployed the old one.
    manager, events = _make_deploy_manager(build_fails=True)

    with pytest.raises(RuntimeError, match="does not import"):
        await manager.deploy_app(
            artifact_id="bioimage-io/model-runner",
            application_id=APP_ID,
            version="2.9.0",
            context=DEPLOY_CONTEXT,
        )

    assert events == [("build", None)]
    assert manager._deployed_applications[APP_ID]["is_deployed"].is_set(), (
        "The old app is still serving, so it must still read as deployed."
    )


# ───────────────────── the fingerprint both sides must agree on ─────────────────────


def test_the_source_hash_tracks_content_and_ignores_bytecode(tmp_path: Path) -> None:
    # The submit task bakes this onto each replica and the introspect task
    # returns it to the worker; the two are only comparable if they hash the
    # same way, which is why there is one function rather than two copies.
    source = tmp_path / "src"
    (source / "pkg").mkdir(parents=True)
    (source / "pkg" / "main.py").write_text("x = 1\n")

    baseline = hash_source_tree(source)

    (source / "pkg" / "__pycache__").mkdir()
    (source / "pkg" / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"\x00\x01")
    assert hash_source_tree(source) == baseline

    (source / "pkg" / "main.py").write_text("x = 2\n")
    assert hash_source_tree(source) != baseline, (
        "A changed method body leaves qualnames and schemas identical — the "
        "content hash is the only thing that sees it."
    )


def test_the_source_hash_tracks_file_names_not_just_bytes(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "one.py").write_text("x = 1\n")
    (b / "two.py").write_text("x = 1\n")

    assert hash_source_tree(a) != hash_source_tree(b)
