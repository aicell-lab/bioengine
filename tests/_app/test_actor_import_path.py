"""Pin the invariant that ``bioengine._app.bootstrap.build_and_run_application``
imports only modules importable in the Ray actor's environment.

The actor lives in a runtime_env whose pip layer ships bioengine's *deps*
(``hypha-rpc``, ``pydantic``, ``httpx``, …), and whose py_modules layer
ships bioengine's source. Worker-only dependencies (``haikunator``, ``uv``,
``aiortc``) are deliberately NOT installed there — they're worker-side
plumbing for the BioEngineWorker that lives only on the bioengine-worker
pod, not on the Ray actor pod.

If any module the actor walks at unpickle / call time happens to do a
top-level import of one of those worker-only deps, ``serve.run`` raises
``ModuleNotFoundError`` and the deploy_app finishes with no deployments
registered, which is how this regression was caught in 0.11.7.

These tests use the same pattern as test_decorators_baseline_imports.py:
intercept ``builtins.__import__`` to simulate a missing module, then drive
the import chain the actor actually walks and assert it survives.
"""
from __future__ import annotations

import builtins
import importlib
import sys


_WORKER_ONLY_MODULES = ("haikunator",)


def _snapshot_relevant_modules() -> dict:
    return {
        k: v
        for k, v in sys.modules.items()
        if k.startswith("bioengine")
        or any(k == m or k.startswith(m + ".") for m in _WORKER_ONLY_MODULES)
    }


def _drop_relevant_modules() -> None:
    for k in list(sys.modules):
        if k.startswith("bioengine") or any(
            k == m or k.startswith(m + ".") for m in _WORKER_ONLY_MODULES
        ):
            sys.modules.pop(k, None)


def _patched_import(banned: tuple[str, ...]):
    real = builtins.__import__

    def fake(name, globals=None, locals=None, fromlist=(), level=0):
        if any(name == b or name.startswith(b + ".") for b in banned):
            raise ImportError(f"simulated missing module: {name}")
        return real(name, globals, locals, fromlist, level)

    return fake


def test_apps_package_init_does_not_pull_haikunator() -> None:
    """Importing ``bioengine.apps`` is the first thing Python does when a
    submodule under it is referenced. The package ``__init__`` must not
    transitively load ``haikunator`` (it's a worker-only dep)."""
    saved = _snapshot_relevant_modules()
    try:
        _drop_relevant_modules()
        original_import = builtins.__import__
        builtins.__import__ = _patched_import(_WORKER_ONLY_MODULES)
        try:
            importlib.import_module("bioengine.apps")
        finally:
            builtins.__import__ = original_import
    finally:
        _drop_relevant_modules()
        sys.modules.update(saved)


def test_proxy_deployment_importable_without_haikunator() -> None:
    """``build_and_run_application`` imports
    ``bioengine.apps.proxy_deployment`` inside its function body, which
    cascades through ``bioengine.apps.__init__``. Both must work in an
    environment that has bioengine's actor-side deps but no worker-only
    deps like haikunator."""
    saved = _snapshot_relevant_modules()
    try:
        _drop_relevant_modules()
        original_import = builtins.__import__
        builtins.__import__ = _patched_import(_WORKER_ONLY_MODULES)
        try:
            importlib.import_module("bioengine.apps.proxy_deployment")
        finally:
            builtins.__import__ = original_import
    finally:
        _drop_relevant_modules()
        sys.modules.update(saved)
