"""``hypha_token=''`` must leave the replica with no credential, not merely un-injected.

The two are the same thing only where the replica starts from an empty
environment. On ``--mode single-machine`` Ray runs inside the worker's own
container, so the replica process inherits the worker's environment and the
worker's own ``HYPHA_TOKEN`` with it: measured on europa (#0526), an app
deployed with ``hypha_token=''`` came up reporting
``test_reports_writable: true`` with the variable present in
``/proc/<pid>/environ`` for all four replicas, and an unreviewed app version
then overwrote a published report on a shared public artifact that had one
version, so there was nothing to roll back to.

On ``--mode external-cluster`` the replica is a pod in another namespace which
inherits nothing, which is why the same empty token already fenced correctly
there. So the fix is an instruction to REMOVE, carried to the replica.

Two things have to stay true while that is added, and they are what the
controls here are for:

* ``hypha_token=None`` (the argument omitted) means "preserve whatever this
  instance had" and must NOT fence — conflating the two would silently strip
  the token off every ordinary redeploy;
* a ``HYPHA_TOKEN`` the caller passed themselves through
  ``application_env_vars`` legitimately reaches the replica when
  ``hypha_token`` carries no token, so fencing that would discard something
  explicitly asked for.
"""

from __future__ import annotations

import logging
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace

from bioengine.apps.builder import AppBuilder

FENCE_KEY = "BIOENGINE_FENCED_ENV_KEYS"
SECRET_PREFIX = "_BIOENGINE_SECRET_"


def _builder() -> AppBuilder:
    """An AppBuilder carrying exactly what ``_build_env_vars`` reads.

    The attribute list is derived from the method itself rather than guessed,
    so a new ``self.<attr>`` read there fails here loudly instead of being
    silently absent.
    """
    builder = object.__new__(AppBuilder)
    builder.server = SimpleNamespace(config=SimpleNamespace(workspace="ws-test"))
    builder.server_url = "https://hypha.example.org"
    builder.data_server_url = "https://hypha.example.org"
    builder.apps_workdir = Path("/tmp/bioengine-apps")
    builder.worker_service_id = "ws-test/worker:bioengine-worker"
    builder.proxy_actor_name = "bioengine_proxy"
    return builder


def test_the_fixture_covers_every_attribute_the_method_reads():
    """CONTROL on the fixture, not on the fix.

    An absent attribute raises AttributeError mid-method, which reads exactly
    like the behaviour under test failing — it already cost one round here. If
    `_build_env_vars` grows a `self.<attr>` read, this says so directly.
    """
    import ast
    import inspect

    tree = ast.parse(textwrap.dedent(inspect.getsource(AppBuilder._build_env_vars)))
    needed = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }
    have = set(vars(_builder()))
    assert not (needed - have), f"fixture is missing {sorted(needed - have)}"


def _env(hypha_token, non_secret=None, secret=None):
    return _builder()._build_env_vars(
        application_id="app1",
        artifact_id="ws-test/my-app",
        version="1.0.0",
        non_secret_env_vars=dict(non_secret or {}),
        secret_env_vars=dict(secret or {}),
        hypha_token=hypha_token,
        download_token=None,
    )


# --------------------------------------------------------------------------
# the worker side: what the replica is TOLD
# --------------------------------------------------------------------------


def test_an_empty_token_asks_the_replica_to_REMOVE_the_inherited_one():
    """THE DEFECT: this previously produced no instruction at all."""
    env = _env("")
    assert env.get(FENCE_KEY) == "HYPHA_TOKEN"
    # and still injects nothing of its own
    assert f"{SECRET_PREFIX}HYPHA_TOKEN" not in env


def test_an_OMITTED_token_does_not_fence():
    """CONTROL. ``None`` means preserve; fencing it would strip the credential
    off every ordinary redeploy that leaves the argument out."""
    env = _env(None)
    assert FENCE_KEY not in env
    assert f"{SECRET_PREFIX}HYPHA_TOKEN" not in env


def test_a_real_token_injects_and_does_not_fence():
    """CONTROL."""
    env = _env("a-token-value")
    assert env[f"{SECRET_PREFIX}HYPHA_TOKEN"] == "a-token-value"
    assert FENCE_KEY not in env


def test_a_caller_supplied_plain_token_is_NOT_fenced_away():
    """CONTROL, and the subtle one.

    When ``hypha_token`` carries no token a plain ``application_env_vars``
    value does reach the replica. Fencing it here would delete something the
    caller explicitly passed, so only the INHERITED case is scrubbed.
    """
    env = _env("", non_secret={"HYPHA_TOKEN": "callers-own"})
    assert FENCE_KEY not in env
    assert env["HYPHA_TOKEN"] == "callers-own"


def test_a_caller_supplied_SECRET_token_is_NOT_fenced_away():
    """CONTROL. Same claim through the underscore-prefixed route."""
    env = _env("", secret={"HYPHA_TOKEN": "callers-own"})
    assert FENCE_KEY not in env
    assert env[f"{SECRET_PREFIX}HYPHA_TOKEN"] == "callers-own"


# --------------------------------------------------------------------------
# the replica side: whether it actually loses the variable
# --------------------------------------------------------------------------


def _hydrate(environ: dict) -> dict:
    """Run the replica's env install against a copy of ``environ``."""
    from bioengine._app import mixin

    saved = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(environ)
        mixin._install_secret_env_vars(
            SimpleNamespace(_bioengine_proxy_handle=None),
            logging.getLogger("test"),
        )
        return dict(os.environ)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_the_replica_DROPS_an_inherited_token_when_told_to():
    """The half that actually closes the hole: the worker's instruction has to
    take effect in the process that runs the user's ``__init__``."""
    out = _hydrate({"HYPHA_TOKEN": "inherited-from-the-worker",
                    FENCE_KEY: "HYPHA_TOKEN"})
    assert "HYPHA_TOKEN" not in out, "the replica kept the worker's credential"


def test_the_replica_KEEPS_an_inherited_token_with_no_instruction():
    """CONTROL, and the one that makes the test above a measurement.

    Without this, a bug that dropped HYPHA_TOKEN unconditionally would pass —
    and that bug breaks every app that legitimately reads the worker's env.
    """
    out = _hydrate({"HYPHA_TOKEN": "inherited-from-the-worker"})
    assert out["HYPHA_TOKEN"] == "inherited-from-the-worker"


def test_the_fence_removes_only_the_named_keys():
    """CONTROL. A fence that cleared the environment would also pass the test
    above, so the blast radius is pinned too."""
    out = _hydrate({
        "HYPHA_TOKEN": "inherited",
        "HF_READ_TOKEN": "a-different-credential",
        "BIOENGINE_APP_DIR": "/tmp/x",
        FENCE_KEY: "HYPHA_TOKEN",
    })
    assert "HYPHA_TOKEN" not in out
    assert out["HF_READ_TOKEN"] == "a-different-credential"
    assert out["BIOENGINE_APP_DIR"] == "/tmp/x"


def test_an_absent_fence_list_is_a_no_op_for_an_older_worker():
    """CONTROL. A pre-fence worker names no keys; the replica must not fail or
    strip anything on that path."""
    out = _hydrate({"HYPHA_TOKEN": "inherited", FENCE_KEY: ""})
    assert out["HYPHA_TOKEN"] == "inherited"


def test_a_fenced_key_that_is_absent_anyway_does_not_raise():
    """The external-cluster case: nothing was inherited, so there is nothing to
    remove and the instruction is harmless."""
    out = _hydrate({FENCE_KEY: "HYPHA_TOKEN"})
    assert "HYPHA_TOKEN" not in out
