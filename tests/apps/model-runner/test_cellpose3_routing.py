"""Cellpose-3 models are served by the one GPU deployment, in their own venv.

Cellpose 3 and 4 are one distribution name at two versions and cannot share a
site-packages. The runner resolves that with a second *interpreter*, not a
second Ray deployment — see ``_python_env_for`` and ``RuntimeImpl._interpreter``.
A second GPU deployment would be scheduled independently by Ray Serve, which
defaults to SPREAD, so "both halves share one card" would be unenforced.
"""

import ast
import pathlib
import typing

import pytest

APP = pathlib.Path(__file__).resolve().parents[3] / "apps" / "model-runner"
ENTRY = APP / "entry.py"
RUNTIME = APP / "runtime.py"


def _load(path, *names):
    """Exec just the named top-level defs/assignments out of ``path``.

    Importing the module would pull in bioengine, ray and bioimageio; these
    tests want the real function bodies without any of that.
    """
    tree = ast.parse(path.read_text())
    wanted = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            wanted.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            wanted.append(node)
    if len(wanted) != len(names):
        found = sorted(
            n.name if hasattr(n, "name") else n.targets[0].id for n in wanted
        )
        pytest.fail(f"missing {sorted(set(names) - set(found))} in {path.name}")
    ns = {"Optional": typing.Optional}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(path), "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def entry_ns():
    return _load(ENTRY, "CELLPOSE3_MODELS", "_python_env_for")


def test_cellpose3_model_selects_the_cellpose3_venv(entry_ns):
    assert entry_ns["_python_env_for"]("bioimage-io/famous-fish") == "cellpose3"


def test_other_models_use_the_runtime_venv(entry_ns):
    """The negative control: without it, ``return "cellpose3"`` would pass."""
    assert entry_ns["_python_env_for"]("bioimage-io/affable-shark") is None


def test_every_listed_cellpose3_model_routes(entry_ns):
    for alias in entry_ns["CELLPOSE3_MODELS"]:
        assert entry_ns["_python_env_for"](f"bioimage-io/{alias}") == "cellpose3"


def test_infer_passes_the_env_to_the_runtime():
    """``infer`` must hand the chosen venv to ``predict_from_disk``."""
    src = ENTRY.read_text()
    call = src[src.index("predict_from_disk(") :]
    call = call[: call.index(")\n")]
    assert "python_env=_python_env_for(model_id)" in call, call


def test_infer_no_longer_refuses_cellpose3():
    """The whole point of the change: these models are served, not rejected."""
    assert "cellpose3-runner' app instead" not in ENTRY.read_text()


def _gpu_deployments(text):
    """Classes decorated with ``@bioengine.app(..., gpu_memory_mb=...)``."""
    out = []
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.ClassDef):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            name = getattr(dec.func, "attr", getattr(dec.func, "id", ""))
            if name == "app" and any(k.arg == "gpu_memory_mb" for k in dec.keywords):
                out.append(node.name)
    return out


def test_the_app_declares_exactly_one_gpu_deployment():
    """Two GPU deployments would be placed independently.

    Ray Serve leaves ``RAY_SERVE_USE_PACK_SCHEDULING_STRATEGY`` off, so
    replicas launch with SPREAD and prefer *different* nodes; nothing in
    bioengine builds a placement group. A second GPU deployment would also
    get its own ``_gpu_lock``, letting two models touch one card at once.
    """
    found = [d for p in APP.glob("*.py") for d in _gpu_deployments(p.read_text())]
    assert found == ["RuntimeDeployment"], found


def test_gpu_deployment_count_check_would_catch_a_second_one():
    """Positive control for the check above — it must not be vacuous."""
    second = (
        "import bioengine\n"
        "@bioengine.app(num_cpus=1, gpu_memory_mb=3072)\n"
        "class Cellpose3Deployment:\n"
        "    pass\n"
    )
    assert _gpu_deployments(second) == ["Cellpose3Deployment"]


def test_runtime_threads_the_env_through_to_the_child():
    """The interpreter swap must reach the actual spawn, not stop at the API."""
    src = RUNTIME.read_text()
    assert "python_env: Optional[str] = None," in src
    spawn = src[src.index("proc = subprocess.Popen(") :][:400]
    assert "self._interpreter(python_env)" in spawn, spawn
    assert "sys.executable" not in spawn, spawn


def test_warm_child_key_includes_the_env():
    """Otherwise a model switch could reuse a child on the wrong interpreter."""
    src = RUNTIME.read_text()
    key = src[src.index("key = (") :]
    assert "python_env," in key[: key.index(")\n")]
