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

import ast
import inspect
import logging
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import tests.apps.test_adopted_app_env_var_loss as adoption_rig
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


# --------------------------------------------------------------------------
# the documented CLI combination, and the one edge case worth disclosing
# --------------------------------------------------------------------------


def _via_application_env_vars(application_env_vars, hypha_token):
    """Run the builder's own flattening, then `_build_env_vars`.

    `--env K=V` arrives as ``application_env_vars`` keyed by deployment class
    and is flattened ACROSS deployments before the secret/non-secret split.
    These four lines are copied from that site rather than imported, because
    the flattening is inline in a long async method; the tripwire below asserts
    the real one still looks like this.
    """
    flat = {}
    for env_dict in application_env_vars.values():
        for key, value in env_dict.items():
            flat[key] = value
    secret = {k[1:]: v for k, v in flat.items() if k.startswith("_")}
    non_secret = {k: v for k, v in flat.items() if not k.startswith("_")}
    return _builder()._build_env_vars(
        application_id="app1",
        artifact_id="ws-test/my-app",
        version="1.0.0",
        non_secret_env_vars=non_secret,
        secret_env_vars=secret,
        hypha_token=hypha_token,
        download_token=None,
    )


def test_an_empty_token_WITH_an_env_supplied_one_still_delivers_it():
    """The combination the published skill documents, so it must not change.

    `--hypha-token '' --env HYPHA_TOKEN=x` is the one invocation where an
    env-supplied token reaches the replica rather than being overwritten. The
    fence must not take it: the caller passed it explicitly, and a reader
    following the skill would be standing a deployment up on this.
    """
    env = _via_application_env_vars({"Entry": {"HYPHA_TOKEN": "x"}}, "")
    assert env["HYPHA_TOKEN"] == "x"
    assert FENCE_KEY not in env


def test_the_fence_is_decided_ACROSS_deployments_not_per_deployment():
    """DISCLOSED LIMITATION, pinned so it is a known shape and not a surprise.

    The flattening merges every deployment's env before the fence is decided,
    so `--env HYPHA_TOKEN=x` on ONE deployment class suppresses the fence for
    ALL of them — and the value reaches all of them too, which it already did
    before this change.

    Per-deployment fencing would need the decision moved inside the
    per-deployment loop, which is a wider change than this fix. Asserting it
    here means the next person reads it off a test rather than discovering it
    on a worker.
    """
    env = _via_application_env_vars(
        {"Entry": {"HYPHA_TOKEN": "x"}, "Runtime": {}}, ""
    )
    assert env["HYPHA_TOKEN"] == "x"
    assert FENCE_KEY not in env, (
        "the fence is global; if this starts passing, fencing became "
        "per-deployment and the skill's account of it needs rewriting"
    )


def test_the_builders_flattening_still_matches_what_the_helper_copies():
    """TRIPWIRE. The two tests above copy the builder's flattening; if the real
    one changes shape they would keep passing against a stale copy."""
    import inspect

    source = inspect.getsource(AppBuilder)
    assert "for env_dict in application_env_vars.values():" in source, (
        "the builder no longer flattens application_env_vars this way, so the "
        "helper in this file is testing a path that no longer exists"
    )
    assert 'k[1:]: v for k, v in flat_env_vars.items() if k.startswith("_")' in source


# --------------------------------------------------------------------------
# adoption must not silently drop the fence — DRIVEN, not grepped
# --------------------------------------------------------------------------
#
# WHY THESE ARMS ARE DRIVEN RATHER THAN ASSERTED OVER SOURCE TEXT. A substring
# assertion over ``build``, ``deploy_app`` or ``AppsManager`` answers "is this
# text present", which is not the question the fence needs answered. At least
# two defect shapes keep every such string in place while inverting the
# behaviour:
#
#   - the update gate written as ``hypha_token is None and not
#     existing_app.get("hypha_token_fenced")`` UNFENCES fenced apps and fences
#     unfenced ones. The key is still read, and ``hypha_token = ""`` is still
#     there a line below it, so a text check cannot tell the two apart.
#   - recovery reading ``app_data.get("hypha_token_fence")``, one letter short,
#     makes the flag always False, so adoption silently drops the fence. A
#     check anchored on the WRITE site never looks at this.
#
# So the arms below drive the real ``recover_deployed_applications`` and then
# the real ``deploy_app``, and assert the ``hypha_token`` value the builder was
# actually handed. Both shapes above make them fail. The adoption rig they
# build on is ``tests/apps/test_adopted_app_env_var_loss.py``.
#
# The WRITE side has no value to observe, so it stays a source check — but it
# parses the ``app_data`` dict literal with ``ast`` and asserts the value
# EXPRESSION rather than matching text, following the precedent of
# ``test_the_builder_writes_the_redacted_key_names_into_the_recovery_blob``.

def _recover_with_fence_flag(monkeypatch, flag):
    """Make the blob a running proxy hands back carry ``hypha_token_fenced``.

    ``flag=None`` means the key is ABSENT — which is what an app deployed by a
    worker older than this change recovers as, and the case that must not be
    read as "fenced".
    """
    original = adoption_rig._app_data

    def with_flag(raw_env_vars):
        blob = original(raw_env_vars)
        if flag is not None:
            blob["hypha_token_fenced"] = flag
        return blob

    monkeypatch.setattr(adoption_rig, "_app_data", with_flag)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_flag,expected_token",
    [
        (True, ""),  # adopted AND fenced -> the fence is re-applied
        (False, None),  # adopted, never fenced -> left alone
        (None, None),  # key absent (older worker) -> left alone
    ],
    ids=["fenced", "not-fenced", "flag-absent"],
)
async def test_an_update_that_omits_the_token_RE_FENCES_an_adopted_fenced_app(
    monkeypatch, stored_flag, expected_token
):
    """THE DEFECT, driven end to end.

    ``app_data`` deliberately holds no secret, so a worker adopting an app
    recovers it with ``hypha_token: None`` — and an update that omits the
    argument inherits that None, which emits NO fence. On a single-machine
    worker the new replicas then hold the worker's own credential again: a
    plain restart plus a routine update is enough, which is the shape of the
    incident the fence exists to prevent.

    The token cannot survive in the blob. The INTENT can, and is not secret.
    """
    _recover_with_fence_flag(monkeypatch, stored_flag)
    manager = await adoption_rig._adopted_manager(
        monkeypatch, adoption_rig.PLAIN_ONLY_ENV_VARS
    )
    assert manager._deployed_applications[adoption_rig.APP_ID][
        "hypha_token_fenced"
    ] is bool(stored_flag), (
        "recovery did not carry the stored fence intent onto the tracked app, "
        "so the update path has nothing to gate on"
    )

    await manager.deploy_app(
        artifact_id=adoption_rig.ARTIFACT_ID,
        application_id=adoption_rig.APP_ID,
        version=adoption_rig.NEXT_VERSION,
        context=adoption_rig.CONTEXT,
    )
    await adoption_rig._settle(manager)

    handed = manager.app_builder.build.await_args.kwargs["hypha_token"]
    # `is type(...)` and not `==`, because "" == None is False but a sloppy
    # falsy comparison would let one stand in for the other, and only "" emits
    # a fence while only None means "inherit".
    assert handed == expected_token and type(handed) is type(expected_token), (
        f"an adopted app whose stored fence intent was {stored_flag!r} handed "
        f"the builder {handed!r}; expected {expected_token!r}. "
        '"" emits the fence, None emits nothing.'
    )
    assert manager._deployed_applications[adoption_rig.APP_ID][
        "hypha_token_fenced"
    ] is (expected_token == ""), "the record did not carry the intent forward"


@pytest.mark.asyncio
async def test_an_adopted_fenced_app_handed_a_REAL_token_is_UNFENCED(monkeypatch):
    """CONTROL on over-reach: the fence must not be sticky.

    Without this, a gate that fenced an adopted app unconditionally — ignoring
    the caller entirely — would pass the arm above for the ``fenced`` case. An
    operator who supplies a credential on the update must get it.
    """
    _recover_with_fence_flag(monkeypatch, True)
    manager = await adoption_rig._adopted_manager(
        monkeypatch, adoption_rig.PLAIN_ONLY_ENV_VARS
    )

    await manager.deploy_app(
        artifact_id=adoption_rig.ARTIFACT_ID,
        application_id=adoption_rig.APP_ID,
        version=adoption_rig.NEXT_VERSION,
        hypha_token="real-credential",
        context=adoption_rig.CONTEXT,
    )
    await adoption_rig._settle(manager)

    assert (
        manager.app_builder.build.await_args.kwargs["hypha_token"]
        == "real-credential"
    ), "an explicit token on the update was swallowed by the fence"
    assert (
        manager._deployed_applications[adoption_rig.APP_ID]["hypha_token_fenced"]
        is False
    ), "the app is still recorded as fenced after being handed a real token"


def test_a_NON_fenced_app_gets_no_fence_from_the_builder_either():
    """CONTROL at the env-var layer, which the driven arms do not reach."""
    env = _env(None)
    assert FENCE_KEY not in env


def test_the_builder_RECORDS_the_fence_intent_in_the_recovery_blob():
    """The WRITE side, where there is no value to observe.

    Driven arms monkeypatch the blob, so they would still pass against a
    builder that never writes the flag — adoption would then always read it
    absent. Parsed with ``ast`` rather than grepped, and the VALUE expression is
    asserted too: a constant ``False``, or a comparison against something other
    than the empty string, is the same defect as the key being missing.
    """
    from bioengine.apps import builder as builder_mod

    source = textwrap.dedent(inspect.getsource(builder_mod.AppBuilder.build))
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Dict)
            and any(getattr(t, "id", None) == "app_data" for t in node.targets)
        ):
            entries = {
                k.value: v
                for k, v in zip(node.value.keys, node.value.values)
                if isinstance(k, ast.Constant)
            }
            break
    else:
        raise AssertionError("no `app_data = {...}` literal in AppBuilder.build")

    assert "hypha_token_fenced" in entries, (
        "the builder does not record the fence intent in app_data, so adoption "
        "has nothing to carry and the fence dies at the next worker restart"
    )
    value = entries["hypha_token_fenced"]
    assert isinstance(value, ast.Compare) and isinstance(
        value.ops[0], ast.Eq
    ), f"expected `hypha_token == \"\"`, got {ast.dump(value)[:120]}"
    assert (
        isinstance(value.left, ast.Name) and value.left.id == "hypha_token"
    ), "the flag is computed from something other than hypha_token"
    assert (
        isinstance(value.comparators[0], ast.Constant)
        and value.comparators[0].value == ""
    ), (
        "the flag is not a comparison against the empty string, which is the "
        "only value that means FENCED"
    )
