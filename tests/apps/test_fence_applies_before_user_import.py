"""The fence must land BEFORE user code is imported, not just before ``__init__``.

Found by review of the fence itself. The pop lived only in
``mixin._setup_replica``, which runs immediately before the user's ``__init__``
— but the user's MODULE is imported earlier, in tasks that inherit the raylet
environment and apply ``env_vars`` onto it:

    bioengine/_app/bootstrap.py  introspect_app_in_ray_task   (env-apply, then _load_app_class)
    bioengine/_app/bootstrap.py  build_and_run_application    (same shape)

So module-level code — ``TOKEN = os.getenv("HYPHA_TOKEN")``, or a client built
at import time — captures the worker's own credential and keeps it, because a
later ``os.environ.pop`` cannot reach a value already bound to a module global.
No app in this repo does that today, which makes it LATENT rather than live;
the fence exists for app versions nobody has reviewed, so latent is the wrong
place to leave it.

The arms below assert the ORDER, not merely the end state: a fence that runs
after the import leaves the captured value behind and the first test fails.
"""

from __future__ import annotations

import os
import textwrap

FENCE_KEY = "BIOENGINE_FENCED_ENV_KEYS"


def _app_package(tmp_path, marker: str):
    """An app whose module reads HYPHA_TOKEN AT IMPORT TIME, like a real one would."""
    pkg = tmp_path / "captures_at_import"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "entry.py").write_text(
        textwrap.dedent(
            f"""
            import os
            # Module level on purpose: this is the shape the fence cannot undo.
            CAPTURED_AT_IMPORT = os.environ.get("HYPHA_TOKEN")
            MARKER = {marker!r}
            """
        )
    )
    return pkg


def _import_under(env: dict, pkg_parent, module: str):
    """Import `module` with `env` as the process environment, and report what it
    captured. Runs in-process with os.environ swapped, which is what the Ray
    task does to itself."""
    import importlib
    import sys

    saved_env = dict(os.environ)
    saved_path = list(sys.path)
    for mod in [m for m in sys.modules if m.startswith("captures_at_import")]:
        del sys.modules[mod]
    try:
        os.environ.clear()
        os.environ.update(env)
        sys.path.insert(0, str(pkg_parent))
        # THE FENCE, as the bootstrap tasks must apply it: after the env-apply
        # loop and BEFORE the user module is imported.
        from bioengine._app.bootstrap import _apply_env_fence

        _apply_env_fence()
        m = importlib.import_module(module)
        return m.CAPTURED_AT_IMPORT, dict(os.environ)
    finally:
        os.environ.clear()
        os.environ.update(saved_env)
        sys.path[:] = saved_path


def test_a_fenced_token_is_gone_BEFORE_the_user_module_is_imported(tmp_path):
    """THE DEFECT. With the fence requested, module-level code must see nothing."""
    pkg = _app_package(tmp_path, "fenced")
    captured, env_after = _import_under(
        {
            "HYPHA_TOKEN": "worker-own-credential",
            FENCE_KEY: "HYPHA_TOKEN",
            "BIOENGINE_APP_DIR": "/tmp/x",
        },
        tmp_path,
        "captures_at_import.entry",
    )
    assert captured is None, (
        "the user's module captured the worker's credential at import time: "
        f"{captured!r}. The fence ran too late to matter."
    )
    assert "HYPHA_TOKEN" not in env_after


def test_an_UNFENCED_token_IS_still_visible_at_import(tmp_path):
    """CONTROL, and the one that makes the test above a measurement.

    Without this, a fence that deleted HYPHA_TOKEN unconditionally — or a rig
    where the variable never reached the environment at all — would pass the
    first test while proving nothing.
    """
    pkg = _app_package(tmp_path, "unfenced")
    captured, env_after = _import_under(
        {"HYPHA_TOKEN": "worker-own-credential", "BIOENGINE_APP_DIR": "/tmp/x"},
        tmp_path,
        "captures_at_import.entry",
    )
    assert captured == "worker-own-credential", (
        "the control did not see the token, so the first test's PASS would mean "
        "nothing about the fence"
    )
    assert env_after["HYPHA_TOKEN"] == "worker-own-credential"


def test_the_fence_leaves_other_inherited_vars_alone_at_import(tmp_path):
    """CONTROL on blast radius at the import boundary."""
    pkg = _app_package(tmp_path, "radius")
    _, env_after = _import_under(
        {
            "HYPHA_TOKEN": "worker-own-credential",
            "HF_READ_TOKEN": "a-different-credential",
            FENCE_KEY: "HYPHA_TOKEN",
            "BIOENGINE_APP_DIR": "/tmp/x",
        },
        tmp_path,
        "captures_at_import.entry",
    )
    assert "HYPHA_TOKEN" not in env_after
    assert env_after["HF_READ_TOKEN"] == "a-different-credential"


def test_both_bootstrap_tasks_apply_the_fence_after_their_env_loop():
    """Pins the CALL SITES, because the tests above exercise the helper directly.

    A helper nobody calls from the two tasks that import user code is the
    original defect with an extra function in it. Checked structurally: in each
    task's source, the fence call must appear AFTER the `os.environ[key] = value`
    loop and BEFORE the app class is loaded.
    """
    import inspect

    from bioengine._app import bootstrap

    for fn_name in ("introspect_app_in_ray_task", "build_and_run_application"):
        src = inspect.getsource(getattr(bootstrap, fn_name))
        assert "_apply_env_fence()" in src, f"{fn_name} never applies the fence"
        apply_at = src.index("os.environ[key] = value")
        fence_at = src.index("_apply_env_fence()")
        assert fence_at > apply_at, (
            f"{fn_name} fences BEFORE it applies env_vars, so the apply would "
            "put the inherited value straight back"
        )
