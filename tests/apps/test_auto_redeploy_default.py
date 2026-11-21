"""``auto_redeploy`` defaults to TRUE for a new application.

A sub-minute Hypha outage permanently kills a replica that is still STARTING,
while one already RUNNING rides it out. So the app that dies is the one nobody
is watching yet, and it stays dead until a human notices — the row this came
from was filed on a 19-hour outage of that exact shape.

Defaulting it on used to be a trade: a dead app for indefinite retries. It is
not any more, because attempts are bounded and the worker reports
``auto_redeploy_exhausted_at`` when it gives up. **This default must not land
without that bound**, which is why one test here asserts the bound exists.

THE OVER-REACH ARMS ARE THE POINT. A flip that also overrode an explicit
``auto_redeploy=False``, or that forced an UPDATE's inherited value to True,
would be worse than the defect it fixes: the first ignores an operator who
said no, and the second silently changes the behaviour of every running app on
the next redeploy. Both are pinned below.
"""

from __future__ import annotations

import ast
import textwrap
import inspect
import re

from bioengine.apps import manager as manager_mod
from bioengine.apps.manager import AppsManager


# Read from the module that is actually IMPORTED, not from a guessed path: a path read can
# silently answer for a different checkout than the one under test.
_MANAGER_SRC = inspect.getsource(manager_mod)

def _deploy_app_source() -> str:
    return inspect.getsource(AppsManager.deploy_app)


def _default_for(name: str, src: str) -> str | None:
    """The literal a `if <name> is None:` branch assigns, read from the AST.

    Read structurally rather than by substring: a comment mentioning
    ``auto_redeploy = True`` anywhere in the method would satisfy a grep while
    the code still assigned False.
    """
    tree = ast.parse(re.sub(r"^\s+", "", src.split("\n")[0]) and inspect.cleandoc(src))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        t = node.test
        if (
            isinstance(t, ast.Compare)
            and isinstance(t.left, ast.Name)
            and t.left.id == name
            and len(node.body) == 1
            and isinstance(node.body[0], ast.Assign)
        ):
            val = node.body[0].value
            if isinstance(val, ast.Constant):
                return repr(val.value)
    return None


def test_a_new_application_defaults_to_auto_redeploy_TRUE():
    """THE CHANGE."""
    got = _default_for("auto_redeploy", _deploy_app_source())
    assert got == "True", f"the fresh-deploy default assigns {got}, not True"


def test_the_sibling_defaults_were_not_swept_along():
    """CONTROL on blast radius. The edit touched one branch; a sed over the
    method would have flipped `debug` and `disable_gpu` too, and nothing else
    in this file would have noticed."""
    src = _deploy_app_source()
    assert _default_for("debug", src) == "False", "debug's default changed"
    assert _default_for("disable_gpu", src) == "False", "disable_gpu's default changed"


def test_an_explicit_FALSE_still_opts_out():
    """OVER-REACH ARM. The default must apply only when the argument is None.

    If this ever fails, the worker is auto-redeploying an app whose operator
    explicitly asked it not to.
    """
    src = _deploy_app_source()
    tree = ast.parse(inspect.cleandoc(src))
    guards = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Name)
        and n.test.left.id == "auto_redeploy"
        and any(isinstance(o, ast.Is) for o in n.test.ops)
    ]
    assert guards, "the default is not behind an `auto_redeploy is None` guard"


def test_an_UPDATE_that_omits_it_still_PRESERVES_the_running_value():
    """OVER-REACH ARM, and the one with the widest consequence.

    An update with the argument omitted must inherit from the running instance,
    not take the new default — otherwise every redeploy of an app that opted
    out silently turns redeployment back on.
    """
    src = _deploy_app_source()
    assert 'auto_redeploy = existing_app["auto_redeploy"]' in src, (
        "the update path no longer inherits the running value"
    )


def test_an_app_whose_UNDEPLOY_FAILED_is_still_forced_OFF():
    """CONTROL on a deliberate False elsewhere in the module.

    When `serve.delete` fails the app is kept tracked but auto_redeploy is
    forced off, so the monitor ignores something that is still running in Ray
    Serve. The flip must not have reached that line.
    """
    src = inspect.getsource(manager_mod)
    assert 'application_info["auto_redeploy"] = False' in src, (
        "the undeploy-failure path no longer forces auto_redeploy off"
    )


def test_the_bound_this_default_depends_on_EXISTS():
    """THE PRECONDITION. The ruling allows this default only together with or
    after the retry bound. Without the bound, defaulting on means every broken
    app retries forever — which is the objection the decision rested on being
    answered."""
    src = inspect.getsource(manager_mod)
    assert "_REDEPLOY_MAX_ATTEMPTS" in src, "the attempt bound is absent"
    assert "auto_redeploy_exhausted_at" in src, "the give-up marker is absent"


def test_the_schema_a_caller_reads_says_TRUE():
    """The description is the only account most callers see; leaving it saying
    False would make the API lie about its own default."""
    src = _deploy_app_source()
    block = src[src.index("auto_redeploy: bool = Field(") :]
    block = block[: block.index("debug: bool = Field(")]
    assert "uses True for a new application" in block, "the schema still advertises False"
    assert "startup-applications" in block, "the schema does not mention startup applications"
    assert "auto_redeploy_exhausted_at" in block, "the schema does not mention the bound"


def test_startup_applications_inherit_the_default_rather_than_setting_their_own():
    """The ruling covers --startup-applications entries explicitly. They are
    covered BY the fresh-deploy default: the startup path forwards its config
    to deploy_app without naming auto_redeploy. If that ever changes, this
    default stops reaching them and the ruling is only half implemented."""
    src = inspect.getsource(AppsManager.deploy_startup_applications)
    assert "auto_redeploy" not in src, (
        "the startup path now sets auto_redeploy itself, so it no longer "
        "inherits the new default"
    )
    assert "deploy_app(**deploy_app_kwargs)" in src.replace("await ", "")

# ─────────────────── behavioural: the decision is EXECUTED ───────────────────
#
# Review round 1 showed every arm above to be a source-text or AST assertion, and defeated them
# with two one-line mutants that both passed 8/8:
#   M1  an unconditional `auto_redeploy = True` after the guard, so an explicit False is overridden
#   M2  storing `"auto_redeploy": False` in the deployed-app record whatever the argument said
# The arms below run the real decision and pin its assignment sites, so both go red.


def _deploy_app_fn() -> ast.AST:
    tree = ast.parse(_MANAGER_SRC)
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "deploy_app"),
        None,
    )
    assert fn is not None, "manager.py has no `deploy_app` -- the anchor moved, not the fix"
    return fn


def _auto_redeploy_guards(fn):
    """Every `if auto_redeploy is None:` whose body assigns auto_redeploy, with its source."""
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        t = node.test
        if not (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                and t.left.id == "auto_redeploy"
                and len(t.ops) == 1 and isinstance(t.ops[0], ast.Is)
                and isinstance(t.comparators[0], ast.Constant)
                and t.comparators[0].value is None):
            continue
        assigns = [n for n in ast.walk(node) if isinstance(n, ast.Assign)
                   and any(isinstance(x, ast.Name) and x.id == "auto_redeploy" for x in n.targets)]
        if assigns:
            out.append((node, ast.get_source_segment(_MANAGER_SRC, node)))
    return out


def _run_guard(src: str, *, auto_redeploy, existing_app=None):
    """Execute one real guard block with controlled inputs and return the resulting value."""
    ns = {"auto_redeploy": auto_redeploy, "existing_app": existing_app or {}}
    exec(textwrap.dedent(src), {}, ns)   # the production source, not a copy of it
    return ns["auto_redeploy"]


def test_the_NEW_path_assigns_True_and_an_explicit_value_SURVIVES_it():
    """Executes the real new-application guard with each of the three inputs."""
    guards = _auto_redeploy_guards(_deploy_app_fn())
    new = [src for _, src in guards if "existing_app" not in src]
    assert len(new) == 1, f"expected one new-application guard, found {len(new)}"
    src = new[0]
    assert _run_guard(src, auto_redeploy=None) is True, (
        "a fresh deploy with the argument omitted does not come out True"
    )
    assert _run_guard(src, auto_redeploy=False) is False, (
        "an explicit auto_redeploy=False does NOT survive the new-application guard"
    )
    assert _run_guard(src, auto_redeploy=True) is True


def test_the_UPDATE_path_INHERITS_and_an_explicit_value_SURVIVES_it():
    """Executes the real update guard: omitted inherits, explicit wins."""
    guards = _auto_redeploy_guards(_deploy_app_fn())
    upd = [src for _, src in guards if "existing_app" in src]
    assert len(upd) == 1, f"expected one update guard, found {len(upd)}"
    src = upd[0]
    assert _run_guard(src, auto_redeploy=None, existing_app={"auto_redeploy": True}) is True
    assert _run_guard(src, auto_redeploy=None, existing_app={"auto_redeploy": False}) is False, (
        "an UPDATE that omits the argument does not preserve a running False"
    )
    assert _run_guard(src, auto_redeploy=False, existing_app={"auto_redeploy": True}) is False, (
        "an explicit False does NOT survive the update guard"
    )


def test_EVERY_assignment_to_auto_redeploy_in_deploy_app_is_BEHIND_a_None_guard():
    """Allow-list, not a blacklist. This is what kills the override mutant.

    An unconditional `auto_redeploy = True` anywhere in deploy_app overrides an explicit False, and
    no execution of the guarded blocks can see it. So: enumerate every assignment to the name and
    require each to sit inside an `if auto_redeploy is None:` block.
    """
    fn = _deploy_app_fn()
    guarded = set()
    for node, _ in _auto_redeploy_guards(fn):
        for n in ast.walk(node):
            if isinstance(n, ast.Assign) and any(
                isinstance(x, ast.Name) and x.id == "auto_redeploy" for x in n.targets
            ):
                guarded.add(n.lineno)
    all_assigns = [
        n.lineno for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and any(isinstance(x, ast.Name) and x.id == "auto_redeploy" for x in n.targets)
    ]
    assert all_assigns, "deploy_app assigns auto_redeploy nowhere -- this arm would be vacuous"
    unguarded = sorted(set(all_assigns) - guarded)
    assert not unguarded, (
        f"deploy_app assigns auto_redeploy outside an `is None` guard at line(s) {unguarded}; "
        "such an assignment overrides an explicit auto_redeploy=False"
    )


def test_the_deployed_record_STORES_THE_VARIABLE_not_a_constant():
    """What the monitor later reads. This is what kills the record mutant.

    `"auto_redeploy": False` in the record passes every guard test above while telling the monitor
    the opposite of what the caller asked for. So require the value to be the NAME.
    """
    fn = _deploy_app_fn()
    entries = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == "auto_redeploy":
                entries.append((getattr(k, "lineno", -1), v))
    assert entries, "deploy_app builds no record entry for auto_redeploy -- arm would be vacuous"
    bad = [(ln, type(v).__name__) for ln, v in entries
           if not (isinstance(v, ast.Name) and v.id == "auto_redeploy")]
    assert not bad, (
        f"the record stores auto_redeploy as {bad} rather than the resolved variable; "
        "a constant there makes the monitor read something the caller never asked for"
    )
