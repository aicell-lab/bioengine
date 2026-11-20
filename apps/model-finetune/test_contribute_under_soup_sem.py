"""#0087: the contribute path's state-dict inspection must run under ``_soup_sem``.

`prepare_contribution` holds up to TWO full cpsam state dicts while it inspects a finished
session's checkpoint. Until Option A (ruled 2026-11-15) that work ran outside every lock and
past every admission gate, so a deposit's peak could sum with a concurrent merge's and the
replica could breach ``memory_mb`` with no merge of its own running. Option A takes the
runtime's existing single-permit ``_soup_sem`` around the inspection, so the replica's peak
becomes ``max(merge, contribute)`` instead of their sum.

**THESE TESTS RUN WITH TORCH STUBBED, AND THAT IS THE POINT.** An earlier version opened with
``pytest.importorskip("torch")``. It passed locally and SKIPPED in CI: `test-suite.yml` runs
`pytest apps/model-finetune` inside `bioengine-worker`, where torch is not installed at all
(`requirements*.txt` mention it zero times). Three skipped cases cannot be caught by the
app-bundle floor either — model-finetune's floor is 90 against a measured 466, so a suite can
lose three cases and stay far above it. The invariant would have landed with nothing guarding
it, and a later refactor dropping the ``async with`` would have gone unnoticed.

So the heavy dependencies are stubbed the way `bench_soup_concurrency.py` stubs them — the
evidence for `#0083`, the row that put `_soup_sem` around the merge in the first place. What is
under test is the **semaphore around the allocating region**, not torch: the state dicts here
are three-key dicts, because ordering is what is being asserted, not bytes.

Scoping matters: `prepare_contribution` does `import torch` INSIDE the function body, so the
stubs have to be live when it is CALLED, not merely when this module was imported. They are
installed with `monkeypatch.setitem`, which restores them afterwards, so a sibling test file in
the same pytest process still sees the real torch.
"""
import asyncio
import importlib.util
import logging
import sys
import threading
import types
from pathlib import Path

import pytest

# Set by the inspection stub. Module-level so a test can wait on it from an executor thread.
_INSPECTION_RAN = threading.Event()


def _fake_tensor(value=0.0):
    """Enough of a tensor for the code paths these tests reach."""
    class _T:
        def __init__(self, v):
            self._v = v

        def detach(self):
            return self

        def clone(self):
            return self

        def reshape(self, *a):
            return [self._v]

        def __getitem__(self, i):
            return self._v
    return _T(value)


def _tiny_sd():
    """A state dict with the real KEY SHAPE and no size — this asserts ordering, not bytes."""
    return {
        "encoder.weight": _fake_tensor(1.0),
        "diam_labels": _fake_tensor(30.0),
        "diam_mean": _fake_tensor(30.0),
    }


def _install_stubs(monkeypatch, ckpt: Path, sdir: Path):
    """Stub every heavy dependency, for the duration of ONE test.

    Mirrors `bench_soup_concurrency.py:_load_runtime`, which is `#0083`'s own harness, rather
    than inventing a second way to do the same thing.
    """
    fake_be = types.ModuleType("bioengine")
    fake_be.app = lambda **kw: (lambda cls: cls)
    fake_be.async_init = lambda f: f
    fake_be.smoke_test = lambda f: f
    fake_be.health_check = lambda f: f
    fake_be.logger = logging.getLogger("model-finetune-test")
    monkeypatch.setitem(sys.modules, "bioengine", fake_be)

    fake_np = types.ModuleType("numpy")
    for attr in ("ndarray", "floating", "integer"):
        setattr(fake_np, attr, type(attr, (), {}))
    monkeypatch.setitem(sys.modules, "numpy", fake_np)

    fake_torch = types.ModuleType("torch")
    fake_torch.load = lambda path, map_location=None, weights_only=None: _tiny_sd()
    fake_torch.save = lambda obj, path: Path(path).write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_pool = types.ModuleType("pool")
    fake_pool.architecture_signature = lambda sd: "sig-under-test"
    fake_pool.state_dict_l2_from_base = lambda a, b: 0.5
    fake_pool.sha256_file = lambda p: "0" * 64
    fake_pool.base_commitment = lambda sd, source: {
        "non_learned": {"diam_labels": 30.0}, "key_set_sha256": "c" * 64,
        "key_count": len(sd), "source": source,
    }
    fake_pool.L2_BASIS_STOCK = "stock"
    fake_pool.L2_BASIS_SESSION_INIT = "session_init"
    fake_pool.is_non_weight_param = lambda k: k.rsplit(".", 1)[-1] in {
        "diam_labels", "diam_mean"}
    monkeypatch.setitem(sys.modules, "pool", fake_pool)

    fake_training = types.ModuleType("training")
    fake_training.checkpoint_path = lambda sid: ckpt
    fake_training.session_dir = lambda sid: sdir
    fake_training.cellpose_base_model = lambda mt: "cpsam_v2"
    monkeypatch.setitem(sys.modules, "training", fake_training)


def _load_runtime_module():
    """Load `runtime_cellpose.py` under a PRIVATE name, with the stubs already installed.

    A private name because importing it as `runtime_cellpose` would poison that entry in
    `sys.modules` for every other test file in this pytest process.
    """
    path = Path(__file__).with_name("runtime_cellpose.py")
    spec = importlib.util.spec_from_file_location("_rtc_isolated", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rtc_isolated"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """A no-``__init__`` runtime with only what `prepare_contribution` touches.

    The session is STOCK-INITIALISED on purpose — no `init_checkpoint.pt` — because that is the
    `else` branch `#0070` added, the one that loads a second state dict and so the one whose
    peak this row regressed on.
    """
    sdir = tmp_path / "sess"
    (sdir / "models").mkdir(parents=True)
    ckpt = sdir / "models" / "cp.pt"
    ckpt.write_bytes(b"not-really-a-checkpoint")
    assert not (sdir / "init_checkpoint.pt").exists(), "fixture must be stock-initialised"

    _install_stubs(monkeypatch, ckpt, sdir)
    mod = _load_runtime_module()
    monkeypatch.delitem(sys.modules, "_rtc_isolated", raising=False)

    cls = getattr(mod.CellposeRuntime, "func_or_class", mod.CellposeRuntime)
    rt = object.__new__(cls)
    rt._soup_sem = asyncio.Semaphore(1)
    rt._unwrap_sd = cls._unwrap_sd
    rt._upload_weights = lambda url, path: None

    _INSPECTION_RAN.clear()
    started = []

    def _base_state_dict(model_type="cpsam"):
        started.append(model_type)
        _INSPECTION_RAN.set()
        return _tiny_sd()

    rt._base_state_dict = _base_state_dict
    rt._started = started
    return rt


@pytest.mark.asyncio
async def test_the_inspection_cannot_start_while_the_merge_holds_the_semaphore(runtime):
    """THE arm this row exists for. Hold ``_soup_sem``; the deposit must wait.

    A mutant that drops the ``async with`` fails HERE and by name.
    """
    await runtime._soup_sem.acquire()                  # stand in for a merge in flight
    task = asyncio.create_task(
        runtime.prepare_contribution("sess", "https://example.invalid/put"))

    # Wait on an EVENT the inspection sets, not on event-loop yields.
    #
    # An earlier version spun `await asyncio.sleep(0)` twenty times and asserted the inspection
    # had not run. That passed against the UNFIXED tree too, which makes it worthless: `_inspect`
    # is dispatched with `asyncio.to_thread`, so it runs in a worker thread, and yielding the
    # loop does not wait for a thread to be scheduled. The assertion held because the thread had
    # not started, not because a lock stopped it.
    loop = asyncio.get_running_loop()
    fired = await loop.run_in_executor(None, _INSPECTION_RAN.wait, 3.0)
    assert not fired, (
        "the inspection ran while _soup_sem was held — the deposit's state dicts can still "
        "sum with a concurrent merge's (#0087)"
    )
    assert runtime._started == []
    assert not task.done()

    runtime._soup_sem.release()                        # the merge finishes
    info = await asyncio.wait_for(task, timeout=30)
    assert runtime._started == ["cpsam"], "the inspection never ran even after the release"
    assert info["architecture_signature"] == "sig-under-test"


@pytest.mark.asyncio
async def test_the_same_call_completes_when_the_semaphore_is_free(runtime):
    """NEGATIVE CONTROL for the arm above.

    Without this, a `prepare_contribution` that raised immediately, hung, or never reached the
    inspection would satisfy `_started == []` and the first test would pass for the wrong reason.
    """
    info = await asyncio.wait_for(
        runtime.prepare_contribution("sess", "https://example.invalid/put"), timeout=30)
    assert runtime._started == ["cpsam"], "the inspection did not run even with the lock free"
    assert info["base_commitment"] is not None


@pytest.mark.asyncio
async def test_the_semaphore_is_released_even_when_the_inspection_raises(runtime):
    """A lock taken on the deposit path must not be able to wedge the merge path.

    `async with` gives this for free, so the test is cheap; it is here because the failure it
    rules out — a replica that stops merging entirely after one bad checkpoint — is worse than
    the breach this row is about.
    """
    def _boom(model_type="cpsam"):
        runtime._started.append(model_type)
        _INSPECTION_RAN.set()
        raise RuntimeError("deliberate: a corrupt checkpoint")

    runtime._base_state_dict = _boom
    with pytest.raises(RuntimeError, match="deliberate"):
        await asyncio.wait_for(
            runtime.prepare_contribution("sess", "https://example.invalid/put"), timeout=30)

    assert runtime._started == ["cpsam"]
    await asyncio.wait_for(runtime._soup_sem.acquire(), timeout=5)   # merge path still usable
    runtime._soup_sem.release()


def test_these_tests_do_not_need_real_torch():
    """The refusal's own condition, asserted rather than assumed.

    If someone reintroduces `pytest.importorskip("torch")` or a module-level `import torch`,
    this file silently returns to skipping in CI — which is exactly how the invariant came to
    be unguarded the first time. Asserting it here means the regression is visible in the file
    that caused it.
    """
    import ast

    tree = ast.parse(Path(__file__).read_text())

    # Checked over the AST, not the raw text. A text grep for "importorskip" matches this
    # file's OWN docstring, which explains the mistake by naming it — the first version of
    # this guard failed on its own prose. Code is what matters, so code is what is parsed.
    skips = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "attr", getattr(n.func, "id", None)) == "importorskip"]
    assert not skips, (
        "this file must run where torch is absent; CI's bioengine-worker image has no torch "
        "and an importorskip turns all of #0087's evidence back into skips"
    )

    # A module-level `import torch` would fail collection outright in that image, taking the
    # whole file with it — a different failure, same outcome: no guard on the invariant.
    top_level_imports = [
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [])
    ]
    assert "torch" not in top_level_imports, (
        "a module-level `import torch` makes this file uncollectable where torch is absent"
    )

    # CONTROL: the finder must be able to see things that ARE there, or both assertions above
    # are vacuous — an AST walk that matched nothing would pass them for the wrong reason.
    assert "asyncio" in top_level_imports
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "attr", getattr(n.func, "id", None)) == "_install_stubs"]
    assert calls, "the Call finder matched nothing at all, so the guards above prove nothing"
