"""``auto_redeploy`` defaults to TRUE for a new application.

A sub-minute Hypha outage permanently kills a replica that is still STARTING,
while one already RUNNING rides it out. So the app that dies is the one nobody
is watching yet, and it stays dead until a human notices. This was observed in
production as a 19-hour outage of that exact shape.

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
    """THE PRECONDITION. This default is only safe together with, or after, the
    retry bound. Without the bound, defaulting on means every broken app
    retries forever, which is the one real objection to turning it on."""
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
    """--startup-applications entries are covered deliberately. They are
    covered BY the fresh-deploy default: the startup path forwards its config
    to deploy_app without naming auto_redeploy. If that ever changes, this
    default stops reaching them and only half of this change is in effect."""
    src = inspect.getsource(AppsManager.deploy_startup_applications)
    assert "auto_redeploy" not in src, (
        "the startup path now sets auto_redeploy itself, so it no longer "
        "inherits the new default"
    )
    assert "deploy_app(**deploy_app_kwargs)" in src.replace("await ", "")

# ─────────────────── behavioural: the decision is EXECUTED ───────────────────
#
# Every arm above is a source-text or AST assertion, and two one-line mutants defeat them
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


def _auto_redeploy_stores(scope):
    """Every node that WRITES the name ``auto_redeploy`` anywhere in ``scope``.

    Keyed on the Name node's context being a Store, NOT on the statement's type. That choice is
    what makes this an allow-list rather than a list of shapes somebody thought of: one predicate
    covers plain assignment, a tuple or list target (``auto_redeploy, _ = True, None``), an
    annotated assignment (``auto_redeploy: bool = True``), an augmented assignment, a walrus
    (``(auto_redeploy := True)``), a ``for`` target, a ``with ... as`` target and a comprehension
    target, because every one of them puts a Store-context Name in the tree.

    An earlier version of this helper matched ``ast.Assign`` with ``ast.Name`` targets. Two
    one-line mutants beat it while every test stayed green: a tuple target, because the target is
    then a Tuple and not a Name, and an annotated assignment, because that is an AnnAssign and
    never an Assign at all. Both override an explicit ``auto_redeploy=False``.

    ``except ... as auto_redeploy`` is covered separately: it carries the name as a plain string
    on the handler, with no Name node to find.
    """
    stores = [
        n for n in ast.walk(scope)
        if isinstance(n, ast.Name) and n.id == "auto_redeploy"
        and isinstance(n.ctx, ast.Store)
    ]
    stores += [
        n for n in ast.walk(scope)
        if isinstance(n, ast.ExceptHandler) and n.name == "auto_redeploy"
    ]
    return stores


def test_EVERY_write_of_auto_redeploy_in_deploy_app_is_BEHIND_a_None_guard():
    """Allow-list, not a blacklist. This is what kills the override mutants.

    An unconditional write to ``auto_redeploy`` anywhere in deploy_app overrides an explicit
    False, and no execution of the guarded blocks can see it. So: enumerate every WRITE of the
    name, in any syntactic form, and require each to sit inside an ``if auto_redeploy is None:``
    block. Identity, not line number — two writes can share a line.
    """
    fn = _deploy_app_fn()
    guarded = {id(n) for node, _ in _auto_redeploy_guards(fn)
               for n in _auto_redeploy_stores(node)}
    all_stores = _auto_redeploy_stores(fn)
    assert all_stores, "deploy_app writes auto_redeploy nowhere -- this arm would be vacuous"
    unguarded = sorted(n.lineno for n in all_stores if id(n) not in guarded)
    assert not unguarded, (
        f"deploy_app writes auto_redeploy outside an `is None` guard at line(s) {unguarded}; "
        "such a write overrides an explicit auto_redeploy=False"
    )


def test_the_guard_finder_SEES_every_write_form_a_mutant_could_use():
    """Guards the guard. If `_auto_redeploy_stores` stopped seeing a form, the arm above would
    pass for a mutant using it -- the failure that produced this test. So assert the finder
    counts a write in each form, on synthetic source rather than on production code."""
    forms = {
        "plain": "auto_redeploy = True",
        "tuple": "auto_redeploy, _ = True, None",
        "list": "[auto_redeploy, _] = [True, None]",
        "annotated": "auto_redeploy: bool = True",
        "augmented": "auto_redeploy += 1",
        "walrus": "x = (auto_redeploy := True)",
        "for-target": "for auto_redeploy in [True]:\n    pass",
        "with-target": "with open('/dev/null') as auto_redeploy:\n    pass",
        "comprehension": "y = [auto_redeploy for auto_redeploy in [True]]",
        "except-as": "try:\n    pass\nexcept ValueError as auto_redeploy:\n    pass",
    }
    missed = [name for name, src in forms.items()
              if not _auto_redeploy_stores(ast.parse(textwrap.dedent(src)))]
    assert not missed, (
        f"_auto_redeploy_stores does not see these write forms: {missed}; a mutant using one "
        "would pass the allow-list above"
    )
    # Negative control: a READ must not be counted, or every guard test becomes vacuous.
    assert not _auto_redeploy_stores(ast.parse("if auto_redeploy is None:\n    pass")), (
        "a READ of auto_redeploy is being counted as a write"
    )


def test_EVERY_auto_redeploy_ARGUMENT_deploy_app_passes_ON_is_the_resolved_VARIABLE():
    """The builder call, which is what an adopted app is rebuilt from after a worker restart.

    The resolved value has to reach the built application, not just the status record. A constant
    at that call site passes every test above: the record keeps the caller's False while the built
    app carries True, so an opted-out app comes back redeploying after a restart. Require every
    `auto_redeploy=` keyword in every call inside deploy_app to pass the NAME.
    """
    fn = _deploy_app_fn()
    kwargs = [
        (kw, call) for call in ast.walk(fn) if isinstance(call, ast.Call)
        for kw in call.keywords if kw.arg == "auto_redeploy"
    ]
    assert kwargs, (
        "deploy_app passes auto_redeploy on to nothing -- this arm would be vacuous, and the "
        "builder call it is written for has moved"
    )
    bad = [(kw.value.lineno, type(kw.value).__name__) for kw, _ in kwargs
           if not (isinstance(kw.value, ast.Name) and kw.value.id == "auto_redeploy")]
    assert not bad, (
        f"deploy_app passes auto_redeploy on as {bad} rather than the resolved variable; "
        "a constant there survives a restart and redeploys an app that opted out"
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


def test_the_RECORD_is_written_ONLY_by_the_dict_literal_and_nothing_else():
    """Allow-list over the STRING, because an allow-list over write SHAPES keeps losing.

    Enumerating the ways a value can be bound to the NAME ``auto_redeploy`` catches an
    assignment, a tuple target, an annotation, an augmented assignment and a walrus -- and
    still misses a write that never touches the name at all:

        self._deployed_applications[application_id]["auto_redeploy"] = True

    That is a Subscript target, not a Name, so no amount of Name-write enumeration sees it,
    and it overrides the recorded value after the dict literal has honoured the caller. It was
    measured surviving the whole file. ``.update({"auto_redeploy": True})``,
    ``setdefault`` and ``setattr`` are the same defect one step further along, and adding each
    to a list of shapes is the mistake that produced this test.

    So invert it: count the string constants equal to ``"auto_redeploy"`` inside ``deploy_app``
    and require each to be one of the two legitimate uses. Any route that writes the key --
    subscript, update, setdefault -- must mention the string to do it, so it shows up here
    whatever syntax it uses.

    The two legitimate uses:
      1. the key of the record dict literal, whose value must be the resolved NAME
      2. a READ of the previous value on the update path (``existing_app["auto_redeploy"]``)
    """
    fn = _deploy_app_fn()
    occurrences = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Constant) and n.value == "auto_redeploy"
    ]
    assert occurrences, (
        "deploy_app never mentions the string 'auto_redeploy' -- this arm would be vacuous "
        "and the record it is written for has moved"
    )

    # Classify each occurrence by the role it plays.
    dict_keys, subscript_reads, other = [], [], []
    for node in ast.walk(fn):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "auto_redeploy":
                    dict_keys.append((k, v))
        elif isinstance(node, ast.Subscript):
            s = node.slice
            if isinstance(s, ast.Constant) and s.value == "auto_redeploy":
                # A LOAD is a read; a STORE is a write to the record and is never legitimate.
                (subscript_reads if isinstance(node.ctx, ast.Load) else other).append(node)

    accounted = len(dict_keys) + len(subscript_reads) + len(other)
    assert accounted == len(occurrences), (
        f"{len(occurrences) - accounted} occurrence(s) of the string 'auto_redeploy' in "
        f"deploy_app are in neither a dict literal nor a subscript -- an unclassified "
        f"mention is exactly what this arm exists to refuse, so it fails rather than "
        f"ignoring them (lines "
        f"{sorted(n.lineno for n in occurrences) })"
    )

    assert not other, (
        f"deploy_app WRITES the record key by subscript at line(s) "
        f"{sorted(n.lineno for n in other)}; that overrides the value the dict literal "
        f"recorded for the caller, so an application that opted out is redeployed anyway"
    )

    assert len(dict_keys) == 1, (
        f"expected exactly ONE record entry for auto_redeploy, found {len(dict_keys)} at "
        f"lines {sorted(k.lineno for k, _ in dict_keys)}"
    )
    key, value = dict_keys[0]
    assert isinstance(value, ast.Name) and value.id == "auto_redeploy", (
        f"the record entry at line {key.lineno} stores "
        f"{type(value).__name__} rather than the resolved variable"
    )

    assert len(subscript_reads) <= 1, (
        f"more than one subscript READ of the key at lines "
        f"{sorted(n.lineno for n in subscript_reads)}; only the update path's read of the "
        f"previous value is expected"
    )
