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
import inspect
import re

from bioengine.apps import manager as manager_mod
from bioengine.apps.manager import AppsManager


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
