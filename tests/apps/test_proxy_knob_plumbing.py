"""The proxy's queue depth and request timeout must be settable per deployment.

Both knobs existed, and both worked, but only as ``ProxyDeployment`` constructor
arguments — so every app on every cluster ran with the built-in defaults, and
raising the request bound for an app whose calls legitimately run long meant
cutting a worker release. These tests pin the path from ``deploy_app``'s
arguments down to the values the builder is handed.

Two of them are about a distinction that is easy to collapse and silent when it
is. ``request_timeout_s=None`` at the proxy means *no bound at all*; as a
``deploy_app`` argument, ``None`` means *not specified*. Passing the argument
straight through would therefore remove the bound this parameter exists to
provide, in exactly the case where the caller said nothing. ``0`` is how a
caller asks for no bound on purpose, so ``0`` must survive — a guard written
``if not request_timeout_s`` would silently turn it back into 600.

The last test is structural rather than behavioural. ``proxy_args`` is splatted
into ``ProxyDeployment.bind(**proxy_args)``, so a key the constructor does not
accept is a ``TypeError`` raised at deploy time on a live cluster, and a
required constructor parameter nobody plumbs is the same. Neither is reachable
from a unit test of either file alone, which is why this compares the two.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps import proxy_deployment as pd_module
from bioengine.apps.manager import AppsManager

WORKSPACE = "bioimage-io"
APP_ID = "nuclei-seg"
ARTIFACT_ID = f"{WORKSPACE}/{APP_ID}"
VERSION = "1.0.0"
CONTEXT = {"user": {"id": "u-1", "email": "u@lab.test"}}

# Names bootstrap supplies at bind time rather than carrying through proxy_args.
BOUND_SEPARATELY = {"entry_deployment_handle", "method_schemas"}


def _fake_app():
    """The minimum of ``AppBuilder.build``'s result that ``deploy_app`` reads."""
    return SimpleNamespace(
        metadata={
            "name": "Nuclei Seg",
            "description": "Segment nuclei.",
            "version": VERSION,
            "application_kwargs": {},
            "application_env_vars": {},
            "resources": {},
            "authorized_users": {"*": ["*"]},
            "available_methods": ["segment"],
            "deployed_by_worker_client_id": "client-1",
            "proxy_service_token_issued_at": 0.0,
            "proxy_service_token_ttl_seconds": 3600,
        },
        spec={"classes": {}},
        scaling={},
    )


def _make_manager(*, running: dict | None = None):
    """A manager drivable through ``deploy_app``, capturing the builder's kwargs.

    The resolution under test happens in ``deploy_app`` and is handed to
    ``AppBuilder.build``, so the build kwargs are the observation point: they are
    what the proxy is ultimately constructed from.
    """
    captured: dict = {}

    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.proxy_knobs")
    manager.admin_users = ["*"]
    manager._deployment_lock = asyncio.Lock()
    manager._check_initialized = lambda: None
    manager.server = MagicMock()
    manager.server.config.workspace = WORKSPACE

    manager.artifact_manager = MagicMock()
    manager.artifact_manager.read = AsyncMock(
        return_value={"versions": [{"version": VERSION, "created_at": 1}]}
    )

    manager.ray_cluster = MagicMock()
    manager.ray_cluster.check_connection = AsyncMock()
    manager.ray_cluster.call_with_reconnect = AsyncMock()

    async def _build(**kwargs):
        captured.update(kwargs)
        return _fake_app()

    manager.app_builder = SimpleNamespace(build=_build)
    manager._check_resources = AsyncMock()
    manager._cancel_deployment_process = AsyncMock()
    manager._deploy_application = AsyncMock()
    manager._generate_application_id = AsyncMock(return_value=APP_ID)

    manager._deployed_applications = {}
    if running is not None:
        manager._deployed_applications[APP_ID] = {
            "started_at": 0.0,
            "version": VERSION,
            "artifact_id": ARTIFACT_ID,
            "application_kwargs": {},
            "application_env_vars": {},
            "hypha_token": "tok",
            "disable_gpu": False,
            "max_ongoing_requests": 10,
            "proxy_memory_in_gb": 0.5,
            "auto_redeploy": False,
            "debug": False,
            "scaling": {},
            "is_deployed": asyncio.Event(),
            **running,
        }
    return manager, captured


async def _deploy(manager, **kwargs):
    result = await manager.deploy_app(
        context=CONTEXT, artifact_id=ARTIFACT_ID, application_id=APP_ID, **kwargs
    )
    task = manager._deployed_applications[result["application_id"]]["deployment_task"]
    await asyncio.gather(task, return_exceptions=True)
    return result


# ── the values a caller names arrive ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_both_knobs_reach_the_builder_as_given() -> None:
    manager, captured = _make_manager()

    await _deploy(manager, version=VERSION, max_queued_requests=7, request_timeout_s=42.5)

    assert captured["max_queued_requests"] == 7
    assert captured["request_timeout_s"] == 42.5


@pytest.mark.asyncio
async def test_zero_survives_as_an_explicit_request_for_no_bound() -> None:
    # The proxy reads this as falsy and skips the wait entirely, which is the
    # documented way to ask for no bound. A guard written `if not
    # request_timeout_s` would coerce it back to 600 and silently impose one.
    manager, captured = _make_manager()

    await _deploy(manager, version=VERSION, request_timeout_s=0)

    assert captured["request_timeout_s"] == 0


# ── what "not specified" resolves to ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_unspecified_timeout_resolves_to_the_default_not_to_none() -> None:
    # None reaching the proxy means no bound at all — the opposite of the
    # default. This is the one resolution that cannot be left to the proxy.
    manager, captured = _make_manager()

    await _deploy(manager, version=VERSION)

    assert captured["request_timeout_s"] == 600.0


@pytest.mark.asyncio
async def test_an_unspecified_queue_depth_stays_none_for_the_proxy_to_resolve() -> None:
    # Here None IS the wanted default: the proxy resolves it to
    # max_ongoing_requests, so repeating that number here would be a second
    # definition of one default, free to drift from the first.
    manager, captured = _make_manager()

    await _deploy(manager, version=VERSION)

    assert captured["max_queued_requests"] is None


# ── updates ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_update_preserves_knobs_it_was_not_given() -> None:
    manager, captured = _make_manager(
        running={"max_queued_requests": 3, "request_timeout_s": 12.0}
    )

    await _deploy(manager, version=VERSION)

    assert captured["max_queued_requests"] == 3
    assert captured["request_timeout_s"] == 12.0


@pytest.mark.asyncio
async def test_an_update_of_an_app_deployed_before_these_existed_defaults_them() -> None:
    # An app already running when this shipped has neither key in the record the
    # worker preserved, so a subscript lookup raises KeyError and the update
    # fails on an app that was fine. The absent keys must read as unspecified.
    manager, captured = _make_manager(running={})
    assert "request_timeout_s" not in manager._deployed_applications[APP_ID], (
        "control: the stored record must genuinely lack the key, or this test "
        "passes without exercising the absence"
    )

    await _deploy(manager, version=VERSION)

    assert captured["max_queued_requests"] is None
    assert captured["request_timeout_s"] == 600.0


# ── proxy_args against the constructor it is splatted into ───────────────────


def _proxy_args_keys() -> set:
    """The string keys of the ``proxy_args`` literal in ``builder.py``."""
    source = Path(inspect.getsourcefile(pd_module)).with_name("builder.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Dict)
            and any(
                isinstance(t, ast.Name) and t.id == "proxy_args" for t in node.targets
            )
        ):
            return {
                k.value
                for k in node.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
    raise AssertionError("no proxy_args dict literal found in builder.py")


def test_every_proxy_arg_is_a_proxy_constructor_parameter() -> None:
    keys = _proxy_args_keys()
    assert "max_ongoing_requests" in keys, (
        "control: a key known to be there must be found, or an AST walk that "
        "silently matched nothing would pass every assertion below"
    )

    params = set(
        inspect.signature(pd_module.ProxyDeployment.func_or_class.__init__).parameters
    ) - {"self"}

    # A key the constructor does not accept is a TypeError from bind(), raised
    # on a live cluster at deploy time and nowhere earlier.
    assert keys <= params, f"not constructor parameters: {sorted(keys - params)}"

    required = {
        name
        for name, p in inspect.signature(
            pd_module.ProxyDeployment.func_or_class.__init__
        ).parameters.items()
        if p.default is inspect.Parameter.empty and name != "self"
    }
    assert required <= keys | BOUND_SEPARATELY, (
        f"required but never plumbed: {sorted(required - keys - BOUND_SEPARATELY)}"
    )

    assert {"max_queued_requests", "request_timeout_s"} <= keys & params
