"""Lock in the baseline-import invariant for ``bioengine._app``.

Importing ``bioengine._app`` (and any of its submodules that the cloudpickle
unpickler may walk through on the Ray client server) must NOT require
``hypha_rpc`` or ``ray.serve`` to be installed. Those land via the user's
runtime_env on the actor where the decorator is *applied*; they are not
expected to be present on whatever pod the Ray client server runs on
upstream of the actor boundary.

This was the root cause of the 0.11.4 deploy_app regression — a top-level
``from hypha_rpc.utils.schema import schema_method`` in
``bioengine/_app/decorators.py`` crashed the head pod's unpickler with
``ModuleNotFoundError: No module named 'hypha_rpc'`` before the actor's
runtime_env was ever materialised. These tests pin the fix so the
regression doesn't sneak back in.
"""
from __future__ import annotations

import builtins
import importlib
import sys


_BIOENGINE_APP_MODULES = (
    "bioengine._app",
    "bioengine._app.decorators",
    "bioengine._app.mixin",
    "bioengine._app.runtime_handle",
    "bioengine._app.errors",
    "bioengine._app.accessors",
    "bioengine._app.bootstrap",
)


def _snapshot_relevant_modules() -> dict:
    return {
        k: v
        for k, v in sys.modules.items()
        if k.startswith("bioengine._app")
        or k.startswith("hypha_rpc")
        or k.startswith("ray.serve")
        or k.startswith("ray._private.deploy")
    }


def _drop_relevant_modules() -> None:
    for k in list(sys.modules):
        if (
            k.startswith("bioengine._app")
            or k.startswith("hypha_rpc")
            or k.startswith("ray.serve")
        ):
            sys.modules.pop(k, None)


def _patched_import(banned: tuple[str, ...]):
    """Return a replacement ``builtins.__import__`` that raises
    ImportError for any module whose name starts with one of ``banned``."""
    real = builtins.__import__

    def fake(name, globals=None, locals=None, fromlist=(), level=0):
        if any(name == b or name.startswith(b + ".") for b in banned):
            raise ImportError(f"simulated missing module: {name}")
        return real(name, globals, locals, fromlist, level)

    return fake


def test_app_subpackage_imports_without_hypha_rpc() -> None:
    """Replicates the head-pod environment: hypha_rpc absent."""
    saved = _snapshot_relevant_modules()
    try:
        _drop_relevant_modules()
        original_import = builtins.__import__
        builtins.__import__ = _patched_import(("hypha_rpc",))
        try:
            importlib.import_module("bioengine._app")
        finally:
            builtins.__import__ = original_import
    finally:
        _drop_relevant_modules()
        sys.modules.update(saved)


def test_app_subpackage_imports_without_ray_serve() -> None:
    """Same shape: ray-core only, no ray[serve]."""
    saved = _snapshot_relevant_modules()
    try:
        _drop_relevant_modules()
        original_import = builtins.__import__
        builtins.__import__ = _patched_import(("ray.serve",))
        try:
            importlib.import_module("bioengine._app")
        finally:
            builtins.__import__ = original_import
    finally:
        _drop_relevant_modules()
        sys.modules.update(saved)


def test_bootstrap_imports_without_hypha_rpc_or_ray_serve() -> None:
    """The introspection module the Ray client server unpickles must not
    pull in hypha_rpc or ray.serve transitively either."""
    saved = _snapshot_relevant_modules()
    try:
        _drop_relevant_modules()
        original_import = builtins.__import__
        builtins.__import__ = _patched_import(("hypha_rpc", "ray.serve"))
        try:
            importlib.import_module("bioengine._app.bootstrap")
        finally:
            builtins.__import__ = original_import
    finally:
        _drop_relevant_modules()
        sys.modules.update(saved)


def test_method_decorator_still_works_when_invoked() -> None:
    """The lazy import inside ``method()`` must succeed at decoration time."""
    from bioengine._app.decorators import method

    @method
    async def ping(self) -> dict:
        return {"ok": True}

    assert hasattr(ping, "__schema__")
    assert getattr(ping, "_bioengine_kind", None) == "method"


def test_app_decorator_still_works_when_invoked() -> None:
    """The lazy ``from ray import serve`` inside ``app()``'s inner
    ``decorator(cls)`` must succeed at decoration time."""
    from ray.serve import Deployment

    from bioengine._app.decorators import app, method

    @app(num_cpus=0)
    class Demo:
        @method
        async def ping(self) -> dict:
            return {"ok": True}

    assert isinstance(Demo, Deployment)
