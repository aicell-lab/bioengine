"""Regression test for the cold-replica pickle-by-reference bug.

Since 0.12.0 stopped shipping app source on Ray ``py_modules`` (it syncs per
file from Hypha instead), a Serve replica no longer has ``<app_dir>/source`` on
``sys.path`` when Ray runs ``cloudpickle.loads(serialized_deployment_def)`` in
``ServeReplica.__init__``. If the entry class's methods reference a module-level
symbol of their own source module (a helper function / class — any monolith like
cellpose has many), ``cloudpickle`` pickles that reference BY NAME, so the cold
replica must ``import main`` inside the unpickle — before any bioengine code runs
(so before the meta_path finder exists) — and dies with ``ModuleNotFoundError``.

``build_and_run_application`` fixes this by registering every materialised
app-source module for pickle-by-value before ``serve.run`` serializes, so the
deployment carries the code and the replica reconstructs it without importing
user modules. This test reproduces the exact failure and the fix at the
``ray.cloudpickle`` layer, loading in a CLEAN subprocess that mimics the cold
replica (source dir absent from ``sys.path``).
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("ray")

_ENTRY = textwrap.dedent(
    """
    # Module-level helper referenced by a method — the by-name pickle trigger.
    def compute():
        return 42

    class Entry:
        def marker(self):
            return compute()
    """
)


def _dump_script(src_dir: str, register: bool) -> str:
    return textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {src_dir!r})
        import entry
        from ray import cloudpickle as cp
        if {register!r}:
            cp.register_pickle_by_value(sys.modules["entry"])
        sys.stdout.buffer.write(cp.dumps(entry.Entry))
        """
    )


def _load_script(blob_path: str) -> str:
    # A cold replica: the source dir is NOT on sys.path, no bioengine finder.
    return textwrap.dedent(
        f"""
        from ray import cloudpickle as cp
        cls = cp.loads(open({blob_path!r}, "rb").read())
        print("OK", cls().marker())
        """
    )


def _dump(code: str, blob):
    with open(blob, "wb") as fh:
        return subprocess.run(
            [sys.executable, "-c", code], stdout=fh, stderr=subprocess.PIPE
        )


def _load(code: str, cwd: str):
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, cwd=cwd
    )


def test_by_reference_pickle_fails_cold_but_by_value_loads(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "entry.py").write_text(_ENTRY)

    byref = tmp_path / "byref.pkl"
    byval = tmp_path / "byval.pkl"
    assert _dump(_dump_script(str(src), register=False), byref).returncode == 0
    assert _dump(_dump_script(str(src), register=True), byval).returncode == 0

    # Cold replica: run from tmp_path so the source dir is not importable and
    # is absent from sys.path.
    ref = _load(_load_script(str(byref)), cwd=str(tmp_path))
    assert ref.returncode != 0
    assert b"No module named 'entry'" in ref.stderr, ref.stderr

    val = _load(_load_script(str(byval)), cwd=str(tmp_path))
    assert val.returncode == 0, val.stderr
    assert val.stdout.strip() == b"OK 42", (val.stdout, val.stderr)
