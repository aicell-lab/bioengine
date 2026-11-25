"""The artifact download token must not ride in any per-replica ``runtime_env`` either.

Sibling of ``test_submit_runtime_env_excludes_download_token.py``, which closed the two task-side
copies and said of this one: *"The per-replica surface. Every Serve deployment's ``runtime_env``
still carries the token via ``BuiltApp.replica_env_vars``; a replica has no argument channel.
Removing it there needs a credential-free source-materialisation path and is tracked separately."*
This is that removal, so that paragraph no longer describes the code — see the same file's updated
text.

``BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN`` is a 30-day workspace-scoped **read** credential. Ray's
``runtime_env_agent`` writes every ``runtime_env`` it materialises verbatim into a plaintext
per-node log, and ``replica_env_vars`` is merged into *every* deployment's ``runtime_env`` at bind
time — so that one dict put the credential in the log of every node a replica ever landed on, with a
lifetime far longer than the replica's.

A Serve replica genuinely has no argument channel: its setup hook reads ``os.environ`` before
``__init__``, which is also why this key cannot be served by the proxy actor the way deploy-time
secrets are. So the token could not simply be dropped — something else had to put the source where
the replica could read it. That something is ``_stage_source_on_nodes``, and it is a Ray **task**,
which does have an argument channel.

THE TWO HALVES ONLY MAKE SENSE TOGETHER, which is why they are asserted together:

  1. ``split_secret_env_vars`` must DROP the key, so no bind-time ``runtime_env`` carries it.
  2. ``_stage_source_on_nodes`` must still DELIVER it — as a task argument, never in the task's
     ``runtime_env`` — and must stage on **every alive node**, waiting for all of them.

Break 1 and the credential is still in the log. Break 2 and every private-artifact replica starves,
while a public-artifact smoke test stays green because anonymous read succeeds — the same silent
failure mode the sibling file warns about.

THE KNOWN RED-MAKER, carried over because the reviewer wrote it against the sibling change and the
identical mistake is available here. Replacing ``_stage_source_on_nodes``'s ``task_env_vars``
comprehension with::

    task_env_vars = dict(submit_runtime_env["env_vars"])

is one line, leaves every assertion about half 1 green, and re-derives the staging env from the dict
the exclusion just built — so staging would sync anonymously and private artifacts would fail.
``test_the_staging_task_ARGUMENT_carries_the_token`` is the assertion that goes red for it. The
lesson the sibling file draws applies unchanged: guard the CONSUMER, not the producer, and walk
every hop between them.

WHAT THIS DOES NOT CATCH, stated so it is not read as wider than it is:

  * Whether the staged source is *correct* — only that staging was asked for on every node and that
    the task got the credential it needs. The content path is
    ``tests/_app/test_staged_source_needs_no_credential.py``.
  * ``/proc/<pid>/environ`` in the staging task. A value applied to ``os.environ`` is visible there;
    this is about Ray's log, not about hiding a credential from the process using it.
  * A node that joins the cluster AFTER the deploy. By design it has neither staged source nor a
    credential; on a shared filesystem it reads the staged copy, on a per-node one its replica fails
    loudly. That boundary is asserted in the consumer-side file.
"""

import asyncio

import pytest

from bioengine.apps.builder import (
    _ARTIFACT_DOWNLOAD_TOKEN_KEY,
    _SECRET_ENV_VAR_PREFIX,
    AppBuilder,
    split_secret_env_vars,
)

TOKEN = "not-a-real-token-abc123"
SECRET = "not-a-real-secret-xyz789"


def _env_vars_fixture() -> dict:
    """A realistic ``env_vars`` dict as ``_build_env_vars`` composes one."""
    return {
        "BIOENGINE_APP_DIR": "/home/.bioengine/apps/ws-test-my-app",
        "BIOENGINE_ARTIFACT_ID": "ws-test/my-app",
        "BIOENGINE_ARTIFACT_VERSION": "1.2.3",
        "BIOENGINE_ARTIFACT_FILES_URL": "https://example.invalid/ws-test/artifacts/my-app/files",
        _ARTIFACT_DOWNLOAD_TOKEN_KEY: TOKEN,
        "HYPHA_SERVER_URL": "https://example.invalid",
        f"{_SECRET_ENV_VAR_PREFIX}HYPHA_TOKEN": SECRET,
    }


def test_the_fixture_actually_carries_both_credentials():
    """Control. If the fixture stopped carrying them, every assertion below passes vacuously."""
    fixture = _env_vars_fixture()
    assert fixture[_ARTIFACT_DOWNLOAD_TOKEN_KEY] == TOKEN
    assert f"{_SECRET_ENV_VAR_PREFIX}HYPHA_TOKEN" in fixture


# ──────────── half 1: no bind-time runtime_env carries the download token ─────────────


def test_split_secret_env_vars_drops_the_download_token():
    replica_env_vars, _ = split_secret_env_vars(_env_vars_fixture())
    assert _ARTIFACT_DOWNLOAD_TOKEN_KEY not in replica_env_vars


def test_the_token_VALUE_is_gone_from_replica_env_vars_under_any_key():
    """Absent by name is not enough — the log leak is about the value appearing at all."""
    replica_env_vars, _ = split_secret_env_vars(_env_vars_fixture())
    assert TOKEN not in replica_env_vars.values()
    assert TOKEN not in repr(replica_env_vars)


def test_the_token_is_not_smuggled_into_the_secret_half_instead():
    """Moving it to the secrets would not work and would look like it had.

    The secret channel is the proxy actor, fetched in ``__init__``; the setup hook needs this value
    *before* that. So "it is not in the replica env" must not be true merely because it was
    relocated somewhere equally unusable.
    """
    _, secret_env_vars = split_secret_env_vars(_env_vars_fixture())
    assert TOKEN not in secret_env_vars.values()
    assert _ARTIFACT_DOWNLOAD_TOKEN_KEY not in secret_env_vars


def test_the_keys_the_replicas_NEED_still_survive():
    """A guard that passed by emptying the dict would be worthless."""
    replica_env_vars, _ = split_secret_env_vars(_env_vars_fixture())
    for required in (
        "BIOENGINE_APP_DIR",
        "BIOENGINE_ARTIFACT_ID",
        "BIOENGINE_ARTIFACT_VERSION",
        "BIOENGINE_ARTIFACT_FILES_URL",
        "HYPHA_SERVER_URL",
    ):
        assert replica_env_vars.get(required), f"{required} was dropped with the token"


def test_the_secret_split_itself_is_unchanged():
    """The pre-existing behaviour must not regress while a second exclusion is added."""
    replica_env_vars, secret_env_vars = split_secret_env_vars(_env_vars_fixture())
    assert not [k for k in replica_env_vars if k.startswith(_SECRET_ENV_VAR_PREFIX)]
    assert secret_env_vars == {"HYPHA_TOKEN": SECRET}
    assert replica_env_vars["BIOENGINE_SECRET_KEYS"] == "HYPHA_TOKEN"


# ──────── half 2: the staging task still gets it, by argument, on every node ───────────


class _FakeRemoteFunction:
    def __init__(self, recorder, options):
        self._recorder = recorder
        self._options = dict(options)

    def options(self, **kwargs):
        return _FakeRemoteFunction(self._recorder, {**self._options, **kwargs})

    def remote(self, *args, **kwargs):
        ref = f"ref-{len(self._recorder)}"
        self._recorder.append(
            {"args": args, "kwargs": kwargs, "options": self._options, "ref": ref}
        )
        return ref


class _FakeRay:
    """Just enough of ``ray`` to drive ``_stage_source_on_nodes`` in-process."""

    class exceptions:
        class RayTaskError(Exception):
            pass

    def __init__(self, recorder, nodes, results):
        self._recorder = recorder
        self._nodes = nodes
        self._results = results

    def nodes(self):
        return self._nodes

    def remote(self, **options):
        return lambda fn: _FakeRemoteFunction(self._recorder, {**options, "fn": fn})

    def get(self, refs):
        # The whole list must be handed to ray.get: staging that is not waited on is
        # indistinguishable from staging that did not happen.
        assert isinstance(refs, list), f"ray.get got {type(refs).__name__}, not a list"
        return [self._results[r] for r in refs]


class _Logger:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(msg)

    warning = info
    error = info


def _drive_staging(monkeypatch, env_vars, nodes, files_per_node=3):
    """Call the real ``_stage_source_on_nodes`` and return (calls, logger)."""
    from bioengine.apps import builder as builder_mod

    calls = []
    results = {}

    class _Recorder(list):
        pass

    alive = [n["NodeID"] for n in nodes if n.get("Alive")]
    fake = _FakeRay(calls, nodes, results)
    monkeypatch.setattr(builder_mod, "ray", fake)

    # Results are keyed by ref, which only exists once .remote() is called, so fill them in
    # lazily via the same dict the fake reads.
    for index, node_id in enumerate(alive):
        results[f"ref-{index}"] = {
            "node_id": node_id,
            "source": f"/apps/ws-test-my-app/source",
            "files": files_per_node,
        }

    builder = AppBuilder.__new__(AppBuilder)
    builder.logger = _Logger()
    asyncio.run(
        AppBuilder._stage_source_on_nodes(
            builder, env_vars, {"pip": [], "env_vars": {}}, "my-app"
        )
    )
    return calls, builder.logger


# Real node ids, not placeholders: ``NodeAffinitySchedulingStrategy`` validates its argument as a
# 28-byte hex string and rejects anything else. The strategy object is deliberately NOT faked —
# faking it would have let a wrong-shaped id through and the assertion about ``soft`` would then be
# checking a stub rather than the thing Ray will actually be handed.
NODE_A = "a1" * 28
NODE_B = "b2" * 28
NODE_DEAD = "c3" * 28

NODES = [
    {"NodeID": NODE_A, "Alive": True},
    {"NodeID": NODE_B, "Alive": True},
    {"NodeID": NODE_DEAD, "Alive": False},
]


def test_the_staging_task_ARGUMENT_carries_the_token(monkeypatch):
    """THE red-maker assertion. Deriving the staging env from the runtime_env fails here.

    This is the consumer guard: half 1 above only removes the token from one place, and that is
    only safe if the staging task still receives it through the channel that does not reach the
    log.
    """
    calls, _ = _drive_staging(monkeypatch, _env_vars_fixture(), NODES)
    assert calls, "no .remote() call was made"
    for call in calls:
        (task_env,) = call["args"]
        assert task_env.get(_ARTIFACT_DOWNLOAD_TOKEN_KEY) == TOKEN


def test_the_token_is_NOT_in_the_staging_tasks_runtime_env(monkeypatch):
    """The other side of the same call: argument yes, runtime_env no."""
    calls, _ = _drive_staging(monkeypatch, _env_vars_fixture(), NODES)
    for call in calls:
        runtime_env = call["options"]["runtime_env"]
        assert TOKEN not in repr(runtime_env)


def test_secrets_do_not_travel_to_the_staging_task_at_all(monkeypatch):
    """The staging task needs the download token; it has no business with deploy-time secrets."""
    calls, _ = _drive_staging(monkeypatch, _env_vars_fixture(), NODES)
    for call in calls:
        (task_env,) = call["args"]
        assert not [k for k in task_env if k.startswith(_SECRET_ENV_VAR_PREFIX)]
        assert SECRET not in repr(task_env)


def test_one_task_per_ALIVE_node_with_hard_affinity(monkeypatch):
    """Every alive node, the dead one skipped, and pinned — not merely preferred.

    ``soft=True`` would let Ray stage a node's copy somewhere else, leaving the node it was meant
    for without source and the failure deferred to replica start.
    """
    calls, _ = _drive_staging(monkeypatch, _env_vars_fixture(), NODES)
    assert len(calls) == 2, f"expected one task per alive node, got {len(calls)}"
    targeted = set()
    for call in calls:
        strategy = call["options"]["scheduling_strategy"]
        assert strategy.soft is False, "staging must pin to its node, not prefer it"
        targeted.add(strategy.node_id)
    assert targeted == {NODE_A, NODE_B}
    assert NODE_DEAD not in targeted


def test_a_node_that_staged_nothing_is_refused(monkeypatch):
    """"The task returned" and "source is on disk" are different claims.

    Only the second is what replicas depend on, so an empty tree must fail the deploy here rather
    than being discovered by each replica in turn.
    """
    with pytest.raises(RuntimeError) as excinfo:
        _drive_staging(monkeypatch, _env_vars_fixture(), NODES, files_per_node=0)
    assert "empty tree" in str(excinfo.value)


def test_no_alive_node_is_an_error_not_a_no_op(monkeypatch):
    """Staging nowhere must not silently succeed.

    A cluster with no alive node cannot run the replicas either; returning quietly would turn a
    clear failure here into a confusing replica-side one later.
    """
    dead_only = [{"NodeID": "node-dead", "Alive": False}]
    with pytest.raises(RuntimeError) as excinfo:
        _drive_staging(monkeypatch, _env_vars_fixture(), dead_only)
    assert "no alive node" in str(excinfo.value)


def test_the_staging_result_is_logged_with_a_file_count(monkeypatch):
    """So an operator reading worker logs can tell staging ran, and on how many nodes."""
    _, logger = _drive_staging(monkeypatch, _env_vars_fixture(), NODES)
    staged = [line for line in logger.lines if "Staged source" in line]
    assert len(staged) == 1, logger.lines
    assert "2 Ray node(s)" in staged[0]
    assert "3 files" in staged[0]
    assert TOKEN not in staged[0]


def test_the_build_task_argument_no_longer_carries_the_token(monkeypatch):
    """``submit`` hands ``replica_env_vars`` to ``build_and_run_application``.

    That dict is now token-free, which is correct *because* source is staged before submit is
    reached — the build task's own ``_ensure_source`` takes the credential-free branch. Pinned so
    the two changes cannot drift apart: if someone re-adds the token to ``replica_env_vars`` to
    "fix" the build task, half 1 breaks and this records why that is not the repair.
    """
    replica_env_vars, _ = split_secret_env_vars(_env_vars_fixture())
    assert _ARTIFACT_DOWNLOAD_TOKEN_KEY not in replica_env_vars
    assert replica_env_vars["BIOENGINE_ARTIFACT_FILES_URL"]
