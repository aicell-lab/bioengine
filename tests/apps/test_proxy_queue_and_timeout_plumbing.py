"""The proxy's queue bound and request timeout are settable per deployment.

Both knobs existed on ``ProxyDeployment``'s constructor but nothing upstream
declared them, so the only way to change either was to edit worker source and
redeploy the worker. These tests pin the whole chain — ``deploy_app`` →
``AppBuilder.build`` → the ``proxy_args`` dict → ``proxy_cls.bind``.

A break in the middle of that chain is the quiet one. The *outer* hop is loud:
``deploy_app`` takes no ``**kwargs``, so a caller passing a parameter the worker
does not declare gets a ``RemoteException`` (measured against a 0.16.1 worker).
But once the keyword is accepted, dropping it from ``proxy_args`` raises
nothing at all — the proxy keeps its hardcoded default and the deploy reports
success, which is why the ``proxy_args`` assertion below exists separately from
the signature one.

The second test is about the *name*. Ray Serve already owns
``max_queued_requests``, and owns it **on this very object**:
``ProxyDeployment._deployment_config.max_queued_requests`` is Serve's router
queue ahead of the proxy's replicas, measured -1 (unbounded). The proxy's own
in-actor cap on callers parked holding a decoded payload is a different thing
bounding different memory, and when both wore one name on one object a reader
had no way to tell which an assertion or a grep meant.

The behavioural tests below pin what "not specified" resolves to, which differs
between the two parameters and is silent when collapsed. ``request_timeout_s``
is ``None`` at the proxy to mean *no bound at all*, while ``None`` as a
``deploy_app`` argument means *not specified* — so passing it straight through
would remove the bound in exactly the case where the caller said nothing. ``0``
is how a caller asks for no bound on purpose and must survive as ``0``; a guard
written ``if not request_timeout_s`` would turn it back into 600.
``max_queued_payloads`` is the opposite: ``None`` is the wanted default, because
the proxy resolves it to ``max_ongoing_requests``.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import logging
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps import proxy_deployment as pd_module
from bioengine.apps.builder import AppBuilder
from bioengine.apps.manager import AppsManager
from bioengine.apps.proxy_deployment import ProxyDeployment

# ProxyDeployment is itself a Serve Deployment object, so the class the
# constructor belongs to is reached through .func_or_class; the Deployment
# wrapper has no __init__ of the proxy's own.
PROXY_CLASS = ProxyDeployment.func_or_class

PARAMETERS = ("max_queued_payloads", "request_timeout_s")

WORKSPACE = "bioimage-io"
APP_ID = "nuclei-seg"
ARTIFACT_ID = f"{WORKSPACE}/{APP_ID}"
VERSION = "1.0.0"
CONTEXT = {"user": {"id": "u-1", "email": "u@lab.test"}}

# Names bootstrap supplies at bind time rather than carrying through proxy_args.
BOUND_SEPARATELY = {"entry_deployment_handle", "method_schemas"}


def _proxy_args_keys() -> set[str]:
    """The literal keys of the ``proxy_args`` dict the builder hands to bind()."""
    source = textwrap.dedent(inspect.getsource(AppBuilder.build))
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Dict)
            and any(
                isinstance(t, ast.Name) and t.id == "proxy_args" for t in node.targets
            )
        ):
            return {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    raise AssertionError("no `proxy_args = {...}` literal found in AppBuilder.build")


def test_both_parameters_are_declared_at_every_hop_to_the_proxy() -> None:
    hops = {
        "AppsManager.deploy_app": AppsManager.deploy_app,
        "AppBuilder.build": AppBuilder.build,
        "ProxyDeployment.__init__": PROXY_CLASS.__init__,
    }
    missing = {
        name: [p for p in PARAMETERS if p not in inspect.signature(fn).parameters]
        for name, fn in hops.items()
    }
    missing = {name: gaps for name, gaps in missing.items() if gaps}
    assert not missing, (
        f"a hop does not declare the parameter it is handed: {missing}. An "
        "undeclared keyword is dropped rather than rejected, so the deploy "
        "succeeds and keeps the hardcoded default."
    )

    absent = [p for p in PARAMETERS if p not in _proxy_args_keys()]
    assert not absent, (
        f"{absent} never reach proxy_args, so the constructor's default wins "
        "no matter what the caller passed to deploy_app."
    )


def test_the_proxy_queue_bound_no_longer_shares_serves_name() -> None:
    # Serve's bound on this same object still holds the name it always had —
    # the collision is resolved by renaming ours, not by Serve's disappearing.
    assert hasattr(ProxyDeployment._deployment_config, "max_queued_requests"), (
        "Serve's own router-queue bound is gone from _deployment_config; this "
        "test's premise about which name belongs to whom needs rechecking."
    )
    assert not hasattr(PROXY_CLASS, "max_queued_requests"), (
        "the proxy's in-actor payload cap is back under Ray Serve's name, on "
        "the same object that carries Serve's own max_queued_requests. One "
        "word, two bounds, different memory — rename it again."
    )
    assert hasattr(PROXY_CLASS, "max_queued_payloads")


# ── what a caller named, and what "not specified" resolves to ────────────────


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


@pytest.mark.asyncio
async def test_both_knobs_reach_the_builder_as_given() -> None:
    manager, captured = _make_manager()

    await _deploy(manager, version=VERSION, max_queued_payloads=7, request_timeout_s=42.5)

    assert captured["max_queued_payloads"] == 7
    assert captured["request_timeout_s"] == 42.5


@pytest.mark.asyncio
async def test_zero_survives_as_an_explicit_request_for_no_bound() -> None:
    # The proxy reads this as falsy and skips the wait entirely, which is the
    # documented way to ask for no bound. A guard written `if not
    # request_timeout_s` would coerce it back to 600 and silently impose one.
    manager, captured = _make_manager()

    await _deploy(manager, version=VERSION, request_timeout_s=0)

    assert captured["request_timeout_s"] == 0


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
    # definition of one default, free to drift from the first — and it would
    # freeze the depth of an app whose cap a later update raises.
    manager, captured = _make_manager()

    await _deploy(manager, version=VERSION)

    assert captured["max_queued_payloads"] is None


@pytest.mark.asyncio
async def test_an_update_preserves_knobs_it_was_not_given() -> None:
    manager, captured = _make_manager(
        running={"max_queued_payloads": 3, "request_timeout_s": 12.0}
    )

    await _deploy(manager, version=VERSION)

    assert captured["max_queued_payloads"] == 3
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

    assert captured["max_queued_payloads"] is None
    assert captured["request_timeout_s"] == 600.0


# ── the two duplicated rules this change creates ─────────────────────────────


def test_the_proxy_still_falls_back_to_the_admission_cap() -> None:
    # get_app_status reports max_ongoing_requests as the effective depth when a
    # caller named none, which is only true while the proxy resolves it that
    # way. Two expressions of one rule, in two files, free to drift apart
    # silently — a fallback changed to a constant here would make the status
    # payload report a bound nothing enforces.
    source = Path(inspect.getsourcefile(pd_module)).read_text()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Attribute) and t.attr == "max_queued_payloads"
                for t in node.targets
            )
            and isinstance(node.value, ast.IfExp)
        ):
            fallback = ast.get_source_segment(source, node.value)
            assert "max_ongoing_requests" in fallback, (
                "The proxy no longer defaults its queue depth to the admission "
                f"cap: {fallback}. AppsManager._get_app_status reports it as "
                "though it did."
            )
            return
    raise AssertionError(
        "no conditional assignment to self.max_queued_payloads in "
        "proxy_deployment.py — the fallback this pins has moved or gone"
    )


def test_every_proxy_arg_is_a_proxy_constructor_parameter() -> None:
    keys = _proxy_args_keys()
    assert "max_ongoing_requests" in keys, (
        "control: a key known to be there must be found, or an AST walk that "
        "silently matched nothing would pass every assertion below"
    )

    signature = inspect.signature(PROXY_CLASS.__init__)
    params = set(signature.parameters) - {"self"}

    # A key the constructor does not accept is a TypeError from bind(), raised
    # on a live cluster at deploy time and nowhere earlier.
    assert keys <= params, f"not constructor parameters: {sorted(keys - params)}"

    required = {
        name
        for name, p in signature.parameters.items()
        if p.default is inspect.Parameter.empty and name != "self"
    }
    assert required <= keys | BOUND_SEPARATELY, (
        f"required but never plumbed: {sorted(required - keys - BOUND_SEPARATELY)}"
    )

    assert set(PARAMETERS) <= keys & params
