"""The artifact download token must not ride in the introspect/build task runtime_env.

``BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN`` is a 30-day workspace-scoped **read** credential the worker
mints per deploy. Ray's ``runtime_env_agent`` writes every ``runtime_env`` it materialises verbatim
into a plaintext per-node log, so anything placed in one is readable by whatever can read that log
and outlives the process that used it. ``split_secret_env_vars`` cannot help here: that key is
deliberately unprefixed, because the replica setup hook reads it from ``os.environ`` *before*
``__init__`` and so cannot be served by the proxy actor.
``test_bind_time_runtime_env_excludes_secrets.py`` says as much and defers this key to here.

WHAT THIS GUARDS, in two halves that only make sense together:

  1. ``_build_submit_runtime_env`` must not put the key in the dict it returns. That dict is the
     ``runtime_env`` of the introspect Ray task *and* — it is the same object — of the
     ``build_and_run_application`` task.
  2. Both of those tasks must keep applying their env mapping, which arrives as a task
     **argument**, onto ``os.environ`` *before* calling ``_ensure_source``. Half 1 is only safe
     because of half 2: a task argument travels through the object store rather than through
     runtime_env materialisation, so the token still reaches every process that reads it. Drop an
     apply loop and source sync breaks for private artifacts — which is precisely the regression a
     reviewer of half 1 would be looking for, so the two assertions live in one file.

Half 1 is behavioural: ``_build_submit_runtime_env`` touches no instance state, so it is called
directly. Half 2 is structural (AST), for the reason the sibling file states: ``build()`` is a
large async method no test in this suite drives, and the two task functions run only inside a Ray
worker.

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


# ───────────────────────── half 1: the exclusion ──────────────────────────────


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


# ──────────── half 2: the ordering that makes the exclusion safe ──────────────


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
