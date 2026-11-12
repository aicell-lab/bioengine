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
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from bioengine.apps.builder import AppBuilder
from bioengine.apps.manager import AppsManager
from bioengine.apps.proxy_deployment import ProxyDeployment

# ProxyDeployment is itself a Serve Deployment object, so the class the
# constructor belongs to is reached through .func_or_class; the Deployment
# wrapper has no __init__ of the proxy's own.
PROXY_CLASS = ProxyDeployment.func_or_class

PARAMETERS = ("max_queued_payloads", "request_timeout_s")


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
