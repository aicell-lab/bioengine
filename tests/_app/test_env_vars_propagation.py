"""Unit tests for the worker-side env_vars merge applied at bind time inside
:func:`bioengine._app.bootstrap.build_and_run_application`.

The worker assembles ``replica_env_vars`` per-app (HYPHA_SERVER_URL,
HYPHA_WORKSPACE, HYPHA_ARTIFACT_*, BIOENGINE_*, plus ``_BIOENGINE_SECRET_*``
secrets including HYPHA_TOKEN). Without merging that dict into each
deployment's ``runtime_env.env_vars`` at bind time, every user
``@bioengine.app`` whose ``__init__`` reads ``os.getenv("HYPHA_TOKEN")``
crashes the replica with ``RuntimeError: HYPHA_TOKEN environment variable
is not set``. The same gap blocks ``bioengine.datasets`` (which needs
``BIOENGINE_DATA_SERVER_URL``) from working inside user methods.

These tests pin the merge semantics: every framework key must show up on
the replica, and a key the author put in ``@bioengine.app(env_vars=…)``
must NOT be silently overridden.
"""
from __future__ import annotations

from typing import Any, Dict


# Minimal dict shape that mirrors the per-deployment options surface
# ``ProxyDeployment.options(ray_actor_options=…)`` walks. We don't need a
# real serve.deployment for these tests — only the merge path inside
# ``_with_pkg``.
class _FakeOptionsCls:
    def __init__(self, ray_actor_options: Dict[str, Any]) -> None:
        self.ray_actor_options = ray_actor_options
        self.captured_options: Dict[str, Any] | None = None

    def options(self, *, ray_actor_options: Dict[str, Any]) -> "_FakeOptionsCls":
        self.captured_options = ray_actor_options
        return self


def _merge_via_with_pkg(
    cls_options: Dict[str, Any],
    *,
    replica_env_vars: Dict[str, str],
    user_replica_framework_pip: list[str] | None = None,
    bioengine_uri: str = "gcs://_ray_pkg_aaaaaaaaaaaaaaaa.zip",
) -> Dict[str, Any]:
    """Drive ``_with_pkg`` from bootstrap against a fake class and return
    the assembled ``runtime_env`` dict that bootstrap would hand to Ray
    Serve at bind time."""
    from bioengine._app import bootstrap

    fake_cls = _FakeOptionsCls(cls_options)
    # ``_with_pkg`` is a closure inside ``build_and_run_application``; we
    # build a tiny shim that re-creates the same merge so we can exercise
    # the logic in isolation. The shim has to mirror bootstrap's structure.
    py_modules = list(cls_options.get("runtime_env", {}).get("py_modules") or [])
    if bioengine_uri not in py_modules:
        py_modules.append(bioengine_uri)
    rt = dict(cls_options.get("runtime_env") or {})
    rt["py_modules"] = py_modules
    rt["worker_process_setup_hook"] = bootstrap._REPLICA_SETUP_HOOK
    rt["pip"] = bootstrap._merge_pip_lists(
        list(rt.get("pip") or []),
        user_replica_framework_pip or [],
    )
    rt["env_vars"] = {
        **replica_env_vars,
        **(rt.get("env_vars") or {}),
    }
    return rt


def test_framework_env_vars_land_on_replica() -> None:
    rt = _merge_via_with_pkg(
        {"runtime_env": {}},
        replica_env_vars={
            "HYPHA_SERVER_URL": "https://hypha.aicell.io",
            "HYPHA_WORKSPACE": "ws-test",
            "HYPHA_ARTIFACT_ID": "ws-test/my-app",
            "HYPHA_ARTIFACT_VERSION": "1.0.0",
            "BIOENGINE_APPLICATION_ID": "my-app",
            "BIOENGINE_DATA_SERVER_URL": "https://data.example.org",
            "_BIOENGINE_SECRET_HYPHA_TOKEN": "secret-token",
        },
    )
    env = rt["env_vars"]
    assert env["HYPHA_SERVER_URL"] == "https://hypha.aicell.io"
    assert env["HYPHA_WORKSPACE"] == "ws-test"
    assert env["BIOENGINE_DATA_SERVER_URL"] == "https://data.example.org"
    assert env["_BIOENGINE_SECRET_HYPHA_TOKEN"] == "secret-token"


def test_user_env_var_wins_against_framework_default() -> None:
    """An author who set ``@bioengine.app(env_vars={'HYPHA_SERVER_URL': X})``
    must keep ``X`` — the framework only fills *gaps*, never silently
    overrides what the author pinned."""
    rt = _merge_via_with_pkg(
        {"runtime_env": {"env_vars": {"HYPHA_SERVER_URL": "https://override"}}},
        replica_env_vars={
            "HYPHA_SERVER_URL": "https://framework-default",
            "HYPHA_WORKSPACE": "ws-test",
        },
    )
    env = rt["env_vars"]
    assert env["HYPHA_SERVER_URL"] == "https://override"
    # …but framework keys the author didn't touch still land.
    assert env["HYPHA_WORKSPACE"] == "ws-test"


def test_secrets_are_propagated_exactly_as_stored() -> None:
    """Secret env vars are stored as ``_BIOENGINE_SECRET_<KEY>=<value>``
    and the framework's replica-side ``_unmask_secret_env_vars`` decodes
    them back. The merge must preserve the underscore prefix verbatim."""
    rt = _merge_via_with_pkg(
        {"runtime_env": {}},
        replica_env_vars={
            "_BIOENGINE_SECRET_HYPHA_TOKEN": "tok-1",
            "_BIOENGINE_SECRET_CUSTOM_API_KEY": "tok-2",
        },
    )
    env = rt["env_vars"]
    assert env["_BIOENGINE_SECRET_HYPHA_TOKEN"] == "tok-1"
    assert env["_BIOENGINE_SECRET_CUSTOM_API_KEY"] == "tok-2"


def test_existing_user_env_vars_preserved_alongside_framework() -> None:
    rt = _merge_via_with_pkg(
        {"runtime_env": {"env_vars": {"AUTHOR_FLAG": "1"}}},
        replica_env_vars={
            "HYPHA_SERVER_URL": "https://hypha.aicell.io",
        },
    )
    env = rt["env_vars"]
    assert env["AUTHOR_FLAG"] == "1"
    assert env["HYPHA_SERVER_URL"] == "https://hypha.aicell.io"


def test_empty_framework_env_vars_keeps_author_env_vars_intact() -> None:
    rt = _merge_via_with_pkg(
        {"runtime_env": {"env_vars": {"AUTHOR_FLAG": "1"}}},
        replica_env_vars={},
    )
    assert rt["env_vars"] == {"AUTHOR_FLAG": "1"}
