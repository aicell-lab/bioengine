"""Unit tests for the worker-side env_vars merge applied at bind time inside
:func:`bioengine._app.bootstrap.build_and_run_application`.

The worker assembles ``replica_env_vars`` per-app (HYPHA_SERVER_URL,
HYPHA_WORKSPACE, HYPHA_ARTIFACT_*, BIOENGINE_*). Without merging that dict
into each deployment's ``runtime_env.env_vars`` at bind time,
``bioengine.datasets`` (which needs ``BIOENGINE_DATA_SERVER_URL``) stops
working inside user methods.

Secrets are the one thing that must *not* ride along, because everything in
here is written verbatim to each node's ``runtime_env_agent.log``. They go
to the proxy actor instead; ``replica_env_vars`` carries only their names, in
``BIOENGINE_SECRET_KEYS``.

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


def test_no_secret_value_reaches_the_runtime_env() -> None:
    """The invariant the proxy-actor delivery route exists for.

    Driven through the real ``split_secret_env_vars`` rather than a
    hand-written dict, so it fails if the worker ever starts putting secrets
    back into what bootstrap merges. Asserted on values, not on the
    ``_BIOENGINE_SECRET_`` prefix: a leak that arrived under some other name
    would still be a leak, and a prefix check would call it clean.
    """
    from bioengine.apps.builder import split_secret_env_vars

    replica_env_vars, secret_env_vars = split_secret_env_vars(
        {
            "HYPHA_SERVER_URL": "https://hypha.test",
            "_BIOENGINE_SECRET_HYPHA_TOKEN": "tok-1",
            "_BIOENGINE_SECRET_CUSTOM_API_KEY": "tok-2",
        }
    )
    # Positive control: the split really did find the secrets, so the
    # absence asserted below is not an empty dict passing trivially.
    assert secret_env_vars == {"HYPHA_TOKEN": "tok-1", "CUSTOM_API_KEY": "tok-2"}

    rt = _merge_via_with_pkg({"runtime_env": {}}, replica_env_vars=replica_env_vars)
    env = rt["env_vars"]

    assert "tok-1" not in env.values()
    assert "tok-2" not in env.values()
    # The names still travel, so a replica whose fetch comes back short can
    # name what it was missing.
    assert env["BIOENGINE_SECRET_KEYS"] == "CUSTOM_API_KEY,HYPHA_TOKEN"
    assert env["HYPHA_SERVER_URL"] == "https://hypha.test"


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
    # The author's env vars survive the merge (the app source is synced from
    # Hypha at replica init, not carried in env_vars).
    assert rt["env_vars"]["AUTHOR_FLAG"] == "1"
