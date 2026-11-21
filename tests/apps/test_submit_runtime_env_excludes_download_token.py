"""The artifact download token must not ride in the introspect/build task runtime_env.

``BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN`` is a 30-day workspace-scoped **read** credential the worker
mints per deploy. Ray's ``runtime_env_agent`` writes every ``runtime_env`` it materialises verbatim
into a plaintext per-node log, so anything placed in one is readable by whatever can read that log
and outlives the process that used it. ``split_secret_env_vars`` cannot help here: that key is
deliberately unprefixed, because the replica setup hook reads it from ``os.environ`` *before*
``__init__`` and so cannot be served by the proxy actor.
``test_bind_time_runtime_env_excludes_secrets.py`` says as much and defers this key to here.

WHAT THIS GUARDS, in three legs that only make sense together:

  1. ``_build_submit_runtime_env`` must not put the key in the dict it returns. That dict is the
     ``runtime_env`` of the introspect Ray task *and* — it is the same object — of the
     ``build_and_run_application`` task.
  2. The token must still reach each task by its other channel: the env mapping passed as a task
     **argument**. For introspect that is ``task_env_vars``, which must be derived from the full
     ``env_vars`` and *not* from ``submit_runtime_env["env_vars"]``; for the build task it is
     ``BuiltApp.replica_env_vars``, which ``split_secret_env_vars`` must keep the key in **and**
     which ``submit`` must be the thing actually handed to ``.remote()``.
  3. Both tasks must keep applying that mapping onto ``os.environ`` *before* calling
     ``_ensure_source``.

Leg 1 is only safe because of legs 2 and 3, so all three live in one file. A task argument travels
through the object store rather than through runtime_env materialisation, which is why the token
still reaches every process that reads it. Break leg 2 or 3 and private-artifact source sync fails
silently — it falls back to anonymous read, which succeeds for public artifacts, so a public-only
smoke test would not notice.

TWO REVIEW ROUNDS SHAPED LEG 2, and both found the same kind of gap — a guard on the thing that
changed, with nothing guarding the thing that makes the change safe:

  * Round 1: leg 2 was absent entirely. The red-maker ``task_env_vars =
    dict(submit_runtime_env["env_vars"])`` passed every test here while breaking every
    private-artifact introspect.
  * Round 2: leg 2's build side tested the **helper** and not the **call site**. Asserting that
    ``split_secret_env_vars`` keeps the key says nothing about whether anyone calls it with the
    right dict, or passes its result on. Two call-site mutants survived green:
    ``split_secret_env_vars(runtime_env["env_vars"])`` in ``build()``, and handing
    ``built_app.runtime_env["env_vars"]`` instead of ``built_app.replica_env_vars`` at ``submit``'s
    ``.remote()``. Either removes the token from the build task's only channel.

Legs 1 and 2 are behavioural wherever the function can be driven: ``_build_submit_runtime_env``
touches no instance state, ``_introspect_via_ray_task`` touches none either, and ``submit`` needs
only a logger and a stubbed secret store — so all three are called for real with ``ray`` faked. The
one exception is ``build()`` itself, a ~290-line async method no test in this suite drives (the
sibling file says the same of it): the argument it passes to ``split_secret_env_vars`` is pinned
structurally instead, by AST, which is enough to fail the mutant above. Leg 3 is structural for the
same reason — the two task functions only run inside a Ray worker.

WHAT THIS DOES NOT CATCH, stated so nobody reads it as wider than it is:

  * The per-replica surface. Every Serve deployment's ``runtime_env`` still carries the token via
    ``BuiltApp.replica_env_vars``; a replica has no argument channel. Removing it there needs a
    credential-free source-materialisation path and is tracked separately.
  * Whether the token is *correct* or scoped as intended — only where it is placed.
  * ``/proc/<pid>/environ``. A value applied to ``os.environ`` is visible there; this is about
    Ray's log, not about hiding a credential from the process using it.
"""

import ast
import pathlib

import pytest

from bioengine.apps.builder import (
    _ARTIFACT_DOWNLOAD_TOKEN_KEY,
    _SECRET_ENV_VAR_PREFIX,
    AppBuilder,
)

BOOTSTRAP = (
    pathlib.Path(__file__).resolve().parents[2] / "bioengine" / "_app" / "bootstrap.py"
)

#: (function name, name of its env-mapping parameter) for every task composed from
#: ``_build_submit_runtime_env``'s dict.
TASKS = [
    ("introspect_app_in_ray_task", "env_vars"),
    ("build_and_run_application", "replica_env_vars"),
]


def _env_vars_fixture() -> dict:
    """A realistic ``env_vars`` dict as ``_build_env_vars`` composes one."""
    return {
        "BIOENGINE_APP_DIR": "/home/.bioengine/apps/ws-test-my-app",
        "BIOENGINE_ARTIFACT_ID": "ws-test/my-app",
        "BIOENGINE_ARTIFACT_VERSION": "1.2.3",
        "BIOENGINE_ARTIFACT_FILES_URL": "https://example.invalid/ws-test/artifacts/my-app/files",
        _ARTIFACT_DOWNLOAD_TOKEN_KEY: "not-a-real-token-abc123",
        "HYPHA_SERVER_URL": "https://example.invalid",
        f"{_SECRET_ENV_VAR_PREFIX}HYPHA_TOKEN": "not-a-real-secret-xyz789",
    }


def _submit_runtime_env(env_vars: dict) -> dict:
    # No instance state is read, so an unbound call is the whole function under test.
    return AppBuilder._build_submit_runtime_env(None, env_vars)


# ───────────────────────── leg 1: the exclusion ───────────────────────────────


def test_the_download_token_is_absent_from_the_task_runtime_env():
    env = _submit_runtime_env(_env_vars_fixture())["env_vars"]
    assert _ARTIFACT_DOWNLOAD_TOKEN_KEY not in env


def test_the_token_VALUE_is_absent_too_under_any_key():
    """Absent by name is not enough — the log leak is about the value appearing at all."""
    fixture = _env_vars_fixture()
    value = fixture[_ARTIFACT_DOWNLOAD_TOKEN_KEY]
    env = _submit_runtime_env(fixture)["env_vars"]
    assert value not in env.values()
    # And nowhere in the composed runtime_env, not just its env_vars.
    assert value not in repr(_submit_runtime_env(fixture))


def test_the_keys_the_tasks_NEED_still_survive():
    """A guard that passes by emptying the dict would be worthless."""
    env = _submit_runtime_env(_env_vars_fixture())["env_vars"]
    for required in (
        "BIOENGINE_APP_DIR",
        "BIOENGINE_ARTIFACT_VERSION",
        "BIOENGINE_ARTIFACT_FILES_URL",
        "BIOENGINE_ARTIFACT_ID",
    ):
        assert env.get(required), f"{required} was dropped along with the token"


def test_secret_prefixed_keys_are_still_excluded():
    """The pre-existing exclusion must not regress while a second one is added."""
    env = _submit_runtime_env(_env_vars_fixture())["env_vars"]
    assert not [k for k in env if k.startswith(_SECRET_ENV_VAR_PREFIX)]


def test_the_fixture_actually_carries_the_token():
    """Control. If the fixture stopped carrying it, every assertion above passes vacuously."""
    assert _ARTIFACT_DOWNLOAD_TOKEN_KEY in _env_vars_fixture()


# ─────── leg 2: the token still reaches each task by its OTHER channel ────────


class _FakeRemoteFunction:
    def __init__(self, recorder, options):
        self._recorder = recorder
        self._options = options

    def remote(self, *args, **kwargs):
        self._recorder.append({"args": args, "kwargs": kwargs, "options": self._options})
        return "object-ref-sentinel"


class _FakeRay:
    """Just enough of ``ray`` to drive ``_introspect_via_ray_task`` in-process."""

    class exceptions:
        class RayTaskError(Exception):
            pass

    def __init__(self, recorder, result):
        self._recorder = recorder
        self._result = result

    def remote(self, **options):
        return lambda fn: _FakeRemoteFunction(self._recorder, dict(options, fn=fn))

    def get(self, ref):
        assert ref == "object-ref-sentinel", f"unexpected ref {ref!r}"
        return self._result


def _drive_introspect(monkeypatch, env_vars, submit_runtime_env):
    """Call the real ``_introspect_via_ray_task`` and return what ``.remote()`` got."""
    import asyncio

    from bioengine.apps import builder as builder_mod

    calls = []
    monkeypatch.setattr(
        builder_mod, "ray", _FakeRay(calls, {"spec": {}, "source_signature": "sig"})
    )
    out = asyncio.run(
        builder_mod.AppBuilder._introspect_via_ray_task(
            None, "mod:Entry", env_vars, submit_runtime_env
        )
    )
    assert out == {"spec": {}, "source_signature": "sig"}
    assert len(calls) == 1, f"expected one .remote() call, got {len(calls)}"
    return calls[0]


def test_the_introspect_task_ARGUMENT_still_carries_the_token(monkeypatch):
    """Leg 2. The task argument is the channel; without it, private sync goes anonymous.

    The ``submit_runtime_env`` handed in is the REAL one, built by the function under test in
    leg 1 — so it genuinely lacks the token. Deriving ``task_env_vars`` from it (the shape a
    reviewer's red-maker used) therefore fails here while leaving leg 1 and leg 3 green.
    """
    fixture = _env_vars_fixture()
    submit_runtime_env = _submit_runtime_env(fixture)
    assert _ARTIFACT_DOWNLOAD_TOKEN_KEY not in submit_runtime_env["env_vars"], (
        "control: the runtime_env must already lack the token, or this arm cannot "
        "distinguish the two possible sources of task_env_vars"
    )

    call = _drive_introspect(monkeypatch, fixture, submit_runtime_env)
    entry_id, task_env_vars = call["args"]
    assert entry_id == "mod:Entry"
    assert task_env_vars.get(_ARTIFACT_DOWNLOAD_TOKEN_KEY) == (
        fixture[_ARTIFACT_DOWNLOAD_TOKEN_KEY]
    ), "the introspect task would sync source anonymously and fail on a private artifact"


def test_the_introspect_task_argument_still_excludes_the_SECRETS(monkeypatch):
    """The pre-existing property of that same dict, so fixing leg 2 cannot regress it."""
    fixture = _env_vars_fixture()
    call = _drive_introspect(monkeypatch, fixture, _submit_runtime_env(fixture))
    _, task_env_vars = call["args"]
    assert not [k for k in task_env_vars if k.startswith(_SECRET_ENV_VAR_PREFIX)]


def test_the_token_is_NOT_in_the_runtime_env_the_task_is_submitted_with(monkeypatch):
    """Ties the two channels together at the one call site that uses both."""
    fixture = _env_vars_fixture()
    call = _drive_introspect(monkeypatch, fixture, _submit_runtime_env(fixture))
    submitted = call["options"]["runtime_env"]
    assert _ARTIFACT_DOWNLOAD_TOKEN_KEY not in submitted["env_vars"]
    assert fixture[_ARTIFACT_DOWNLOAD_TOKEN_KEY] not in repr(submitted)


def test_split_secret_env_vars_KEEPS_the_token_for_the_build_task():
    """Leg 2, build side, HELPER only. Not sufficient on its own — see the two tests below.

    This says the helper keeps the key. It says nothing about whether anyone calls it with the
    right dict, or passes its result to the task. Round 2 found both of those unguarded.
    """
    from bioengine.apps.builder import split_secret_env_vars

    fixture = _env_vars_fixture()
    replica_env_vars, secret_env_vars = split_secret_env_vars(fixture)
    assert replica_env_vars.get(_ARTIFACT_DOWNLOAD_TOKEN_KEY) == (
        fixture[_ARTIFACT_DOWNLOAD_TOKEN_KEY]
    ), "the build task would sync source anonymously and fail on a private artifact"
    assert _ARTIFACT_DOWNLOAD_TOKEN_KEY not in secret_env_vars, (
        "routing it through the proxy actor would not work: the setup hook reads "
        "os.environ before __init__"
    )


def _built_app(replica_env_vars: dict, runtime_env: dict):
    """A BuiltApp carrying only what ``submit`` reads off it."""
    from bioengine.apps.builder import BuiltApp

    return BuiltApp(
        metadata={},
        spec={"classes": {}},
        translated_kwargs={},
        proxy_args={},
        runtime_env=runtime_env,
        bioengine_uri="gcs://_ray_pkg_deadbeef.zip",
        proxy_pip=[],
        user_replica_framework_pip=[],
        replica_env_vars=replica_env_vars,
        replica_secret_env_vars={"HYPHA_TOKEN": "not-a-real-secret-xyz789"},
        proxy_memory_in_gb=1.0,
        scaling={},
    )


def _drive_submit(monkeypatch, built_app):
    """Call the real ``AppBuilder.submit`` and return what ``.remote()`` received.

    ``submit`` reads only ``self.logger`` and ``self._store_replica_secrets``, so an uninitialised
    instance with those two supplied drives the real method — no AppBuilder construction, no Hypha,
    no Ray.
    """
    import asyncio
    import logging

    from bioengine.apps import builder as builder_mod

    calls = []
    monkeypatch.setattr(builder_mod, "ray", _FakeRay(calls, {"status": "ok"}))

    inst = builder_mod.AppBuilder.__new__(builder_mod.AppBuilder)
    inst.logger = logging.getLogger("test-0795")
    stored = []

    async def _fake_store(application_id, secret_env_vars):
        stored.append((application_id, secret_env_vars))

    inst._store_replica_secrets = _fake_store

    asyncio.run(builder_mod.AppBuilder.submit(inst, built_app, "my-app"))
    assert len(calls) == 1, f"expected one .remote() call, got {len(calls)}"
    assert stored, "submit did not store the replica secrets -- the method did not really run"
    return calls[0]


def test_submit_hands_the_task_the_dict_that_CARRIES_the_token(monkeypatch):
    """Leg 2, build side, CALL SITE. The mutant round 2 found lives here.

    The ``runtime_env`` given to the BuiltApp deliberately LACKS the token, exactly as
    ``_build_submit_runtime_env`` now produces it. So handing ``built_app.runtime_env["env_vars"]``
    to ``.remote()`` instead of ``built_app.replica_env_vars`` fails here — and that swap is
    otherwise silent, because a token-less sync falls back to anonymous read.

    The token is located by VALUE across every positional argument rather than by index: an index
    would break on any reordering of a 11-argument call and would then be fixed by renumbering
    rather than by noticing.
    """
    fixture = _env_vars_fixture()
    token = fixture[_ARTIFACT_DOWNLOAD_TOKEN_KEY]
    runtime_env = _submit_runtime_env(fixture)
    assert _ARTIFACT_DOWNLOAD_TOKEN_KEY not in runtime_env["env_vars"], (
        "control: the runtime_env must already lack the token, or this arm cannot tell the "
        "two candidate dicts apart"
    )
    replica_env_vars = {k: v for k, v in fixture.items()
                        if not k.startswith(_SECRET_ENV_VAR_PREFIX)}

    call = _drive_submit(monkeypatch, _built_app(replica_env_vars, runtime_env))

    carriers = [a for a in call["args"]
                if isinstance(a, dict) and a.get(_ARTIFACT_DOWNLOAD_TOKEN_KEY) == token]
    assert carriers, (
        "no argument handed to the build task carries the download token; the task would sync "
        "source anonymously and every private-artifact deploy would fail"
    )


def test_submit_does_NOT_leak_the_secrets_into_the_task_runtime_env(monkeypatch):
    """The property that was already there, so fixing the call site cannot regress it."""
    fixture = _env_vars_fixture()
    runtime_env = _submit_runtime_env(fixture)
    replica_env_vars = {k: v for k, v in fixture.items()
                        if not k.startswith(_SECRET_ENV_VAR_PREFIX)}
    call = _drive_submit(monkeypatch, _built_app(replica_env_vars, runtime_env))

    submitted = call["options"]["runtime_env"]
    assert _ARTIFACT_DOWNLOAD_TOKEN_KEY not in submitted.get("env_vars", {})
    secret_value = "not-a-real-secret-xyz789"
    assert secret_value not in repr(submitted), "a secret reached the task runtime_env"


def _build_fn() -> ast.AST:
    builder_src = (
        pathlib.Path(__file__).resolve().parents[2] / "bioengine" / "apps" / "builder.py"
    )
    tree = ast.parse(builder_src.read_text())
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "build"),
        None,
    )
    assert fn is not None, "builder.py has no `build` -- the anchor moved, not the fix"
    return fn


def test_build_passes_the_HELPERS_runtime_env_through_UNTOUCHED():
    """Leg 1 at its CALL SITE. Asserts an ALLOW-LIST of uses, not a list of forbidden shapes.

    Excluding the token inside ``_build_submit_runtime_env`` is worthless if ``build()`` then
    rebinds or edits the dict before handing it on. Round 1 of this review supplied the rebinding
    mutant (``runtime_env = {**runtime_env, ...}``), which passed 19/19 before any guard existed.

    The first guard enumerated the write shapes it forbade — subscript assignment and
    ``.update()``/``.setdefault()``/``.pop()``/``.clear()`` on the name. Round 2 defeated it twice
    with one-liners acting ONE LEVEL DOWN, where the receiver is an ``ast.Subscript`` rather than
    an ``ast.Name``, so neither check could see them:

        runtime_env["env_vars"][_ARTIFACT_DOWNLOAD_TOKEN_KEY] = download_token
        runtime_env["env_vars"].update({...unprefixed env_vars...})

    An alias (``ev = runtime_env["env_vars"]; ev[KEY] = ...``) evades a shape list the same way.
    Enumerating bad shapes can only ever cover the ones imagined; so this asserts the good ones
    instead.

    THE RULE: the bound name is loaded EXACTLY at its two consumers and nowhere else — as a direct
    positional argument of ``_introspect_via_ray_task(...)`` and as the ``runtime_env=`` keyword of
    ``BuiltApp(...)``. Any other load — under a ``Subscript``, under an ``Attribute``, into another
    variable, into an f-string — fails, whether or not anyone thought of it. Nothing can reach the
    dict without first loading the name.
    """
    build = _build_fn()

    bindings = [
        (t.id, n.lineno)
        for n in ast.walk(build)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "attr", None) == "_build_submit_runtime_env"
        for t in n.targets
        if isinstance(t, ast.Name)
    ]
    assert len(bindings) == 1, (
        f"expected exactly one `<name> = self._build_submit_runtime_env(...)`, found {bindings}"
    )
    name, bind_line = bindings[0]

    # Parent map: ast gives no parent links, and the whole assertion is about what encloses
    # each use.
    parents = {}
    for node in ast.walk(build):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    stores, loads, others = [], [], []
    for n in ast.walk(build):
        if not (isinstance(n, ast.Name) and n.id == name):
            continue
        if isinstance(n.ctx, ast.Store):
            stores.append(n.lineno)
        elif isinstance(n.ctx, ast.Load):
            loads.append(n)
        else:
            others.append((n.lineno, type(n.ctx).__name__))

    # Bound once, never rebound, never deleted.
    assert stores == [bind_line], (
        f"{name!r} is stored at {stores}; it must be bound once, at line {bind_line}, and never "
        "reassigned — a rebinding after the helper returns undoes the token exclusion"
    )
    assert not others, f"{name!r} appears in an unexpected context: {others}"
    walrus = [n.lineno for n in ast.walk(build)
              if isinstance(n, ast.NamedExpr) and getattr(n.target, "id", None) == name]
    assert not walrus, f"{name!r} is rebound by a walrus at {walrus}"

    # Every LOAD must be one of exactly two consumers.
    def classify(node):
        parent = parents.get(id(node))
        if isinstance(parent, ast.Call):
            fname = getattr(parent.func, "attr", None) or getattr(parent.func, "id", None)
            if fname == "_introspect_via_ray_task" and any(a is node for a in parent.args):
                return "introspect"
        if isinstance(parent, ast.keyword) and parent.arg == "runtime_env":
            grand = parents.get(id(parent))
            if isinstance(grand, ast.Call) and (
                getattr(grand.func, "id", None) or getattr(grand.func, "attr", None)
            ) in ("BuiltApp", "BuiltApplication"):
                return "builtapp"
        return None

    seen, illegal = {}, []
    for node in loads:
        kind = classify(node)
        if kind is None:
            parent = parents.get(id(node))
            illegal.append((node.lineno, type(parent).__name__))
        else:
            seen.setdefault(kind, []).append(node.lineno)

    assert not illegal, (
        f"{name!r} is loaded somewhere other than its two consumers, at "
        f"{[f'line {ln} under {p}' for ln, p in illegal]}. Every use must be a direct argument of "
        "_introspect_via_ray_task(...) or the runtime_env= keyword of BuiltApp(...); anything else "
        "can reach the dict and undo the exclusion"
    )
    assert "introspect" in seen, f"the introspect task is never submitted with {name!r}"
    assert "builtapp" in seen, f"BuiltApp is never constructed with runtime_env={name}"
    # Control: if the allow-list matched nothing at all the assertions above would be vacuous.
    assert loads, f"{name!r} is never loaded -- the helper's result goes nowhere"

    # The allow-list above guards the NAME. It does not, on its own, stop the OBJECT being
    # reached through the thing that now holds it:
    #     built = BuiltApp(..., runtime_env=runtime_env)
    #     built.runtime_env["env_vars"].update({...})      # name still loaded only twice
    #     return built
    # So require the BuiltApp(...) call to BE the returned value: constructed and handed off in
    # one expression, with no name bound to it that anything could go through.
    builtapp_calls = [
        n for n in ast.walk(build)
        if isinstance(n, ast.Call)
        and (getattr(n.func, "id", None) or getattr(n.func, "attr", None))
        in ("BuiltApp", "BuiltApplication")
    ]
    assert len(builtapp_calls) == 1, (
        f"expected one BuiltApp(...) construction in build(), found {len(builtapp_calls)}"
    )
    call = builtapp_calls[0]
    parent = parents.get(id(call))
    assert isinstance(parent, ast.Return), (
        f"BuiltApp(...) at line {call.lineno} is not the value of a `return` (its parent is "
        f"{type(parent).__name__}). Bound to a name first, the runtime_env dict can be reached "
        "through that name — `built.runtime_env[\"env_vars\"].update(...)` — without ever "
        "loading the guarded name again"
    )


def test_build_derives_replica_env_vars_from_the_FULL_env_vars(monkeypatch):
    """Leg 2, build side, the other call site — structural, because ``build()`` is not drivable.

    ``build()`` must call ``split_secret_env_vars`` on the dict ``_build_env_vars`` returned, not on
    ``runtime_env["env_vars"]`` (which no longer carries the token). The sibling file already
    asserts that ``replica_env_vars`` comes *from* ``split_secret_env_vars``; this pins WHICH dict
    goes in, which is what the round-2 mutant changed.
    """
    build = _build_fn()

    # The local name bound from self._build_env_vars(...)
    bound = [
        t.id
        for n in ast.walk(build)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "attr", None) == "_build_env_vars"
        for t in n.targets
        if isinstance(t, ast.Name)
    ]
    assert len(bound) == 1, f"expected one assignment from _build_env_vars, found {bound}"

    calls = [
        n for n in ast.walk(build)
        if isinstance(n, ast.Call)
        and (getattr(n.func, "id", None) or getattr(n.func, "attr", None))
        == "split_secret_env_vars"
    ]
    assert len(calls) == 1, f"expected one split_secret_env_vars call in build(), found {len(calls)}"
    arg = calls[0].args[0]
    assert isinstance(arg, ast.Name) and arg.id == bound[0], (
        f"build() calls split_secret_env_vars on {ast.dump(arg)[:60]} rather than on the "
        f"{bound[0]!r} that _build_env_vars returned. runtime_env['env_vars'] no longer carries "
        "the download token, so splitting that instead silently strips it from every replica."
    )


# ──────────── leg 3: the ordering that makes the exclusion safe ────────────────


def _fn(name: str) -> ast.AST:
    tree = ast.parse(BOOTSTRAP.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"bootstrap.py has no `{name}` -- the anchor moved, not the fix")


def _apply_loop_lines(fn: ast.AST, param: str) -> list:
    """Lines of ``for k, v in <param>.items(): os.environ[k] = v`` loops in `fn`.

    Keyed on the PARAMETER name: the point of the assertion is that the mapping arrives as an
    argument. A loop over something read back out of ``os.environ`` would prove nothing.
    """
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.For):
            continue
        over_param = any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "items"
            and isinstance(c.func.value, ast.Name)
            and c.func.value.id == param
            for c in ast.walk(node.iter)
        )
        writes_environ = any(
            isinstance(a, ast.Assign)
            and any(
                isinstance(t, ast.Subscript)
                and isinstance(t.value, ast.Attribute)
                and t.value.attr == "environ"
                for t in a.targets
            )
            for a in ast.walk(node)
        )
        if over_param and writes_environ:
            out.append(node.lineno)
    return out


def _ensure_source_call_lines(fn: ast.AST) -> list:
    return [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_ensure_source"
    ]


@pytest.mark.parametrize("task_name,param", TASKS)
def test_the_task_applies_its_env_ARGUMENT_onto_os_environ(task_name, param):
    fn = _fn(task_name)
    assert param in [a.arg for a in fn.args.args], (
        f"{task_name} no longer takes an env mapping called {param!r}; if it is delivered some "
        "other way, the exclusion in _build_submit_runtime_env needs re-justifying"
    )
    assert _apply_loop_lines(fn, param), (
        f"{task_name} no longer applies its {param} argument onto os.environ. The download token "
        "is excluded from this task's runtime_env *because* this loop delivers it; without the "
        "loop, source sync from a private artifact fails."
    )


@pytest.mark.parametrize("task_name,param", TASKS)
def test_the_apply_loop_runs_BEFORE_ensure_source(task_name, param):
    fn = _fn(task_name)
    loops = _apply_loop_lines(fn, param)
    calls = _ensure_source_call_lines(fn)
    assert calls, f"{task_name} no longer calls _ensure_source -- this assertion would be vacuous"
    assert loops, f"{task_name} has no apply loop -- covered by the test above"
    assert min(loops) < min(calls), (
        f"{task_name} calls _ensure_source at line {min(calls)} before applying {param} at "
        f"{min(loops)}; the token would not be in os.environ yet"
    )
