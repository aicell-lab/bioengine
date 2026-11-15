"""#0087: the contribute path's state-dict inspection must run under ``_soup_sem``.

`prepare_contribution` holds up to TWO full cpsam state dicts while it inspects a finished
session's checkpoint. Until Option A (ruled 2026-11-15) that work ran outside every lock
and past every admission gate, so a deposit's peak could sum with a concurrent merge's and
the replica could breach ``memory_mb`` with no merge of its own running.

Option A takes the runtime's existing single-permit ``_soup_sem`` around the inspection, so
the replica's peak becomes ``max(merge, contribute)`` instead of their sum.

**What these tests assert is MUTUAL EXCLUSION, not memory.** Peak RSS is not observable from
a unit test, and a test that claimed to measure it here would be measuring the harness. The
bound follows from exclusion plus this row's FU1 figures, and the close states it as a
number; what is mechanically checkable is that the inspection cannot start while the
semaphore is held, and that is what is checked.

The negative arm is the point: with the semaphore free the same call completes, so the
positive arm is not passing because the coroutine is simply broken.
"""
import asyncio
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Set by the inspection stub below. Module-level so a test can wait on it from an
# executor thread; `_runtime` clears it so one test cannot see another's signal.
_INSPECTION_RAN = threading.Event()

import pool  # noqa: E402


def _tiny_sd():
    """A state dict with the real key SHAPE but no size — this tests ordering, not bytes."""
    return {
        "encoder.weight": torch.zeros(2, dtype=torch.float32),
        "diam_labels": torch.tensor([30.0], dtype=torch.float32),
        "diam_mean": torch.tensor([30.0], dtype=torch.float32),
    }


def _runtime(tmp_path: Path, started: list, finished: list):
    """A no-``__init__`` runtime wired with only what ``prepare_contribution`` touches.

    ``_base_state_dict`` records entry and exit so a test can see WHEN the inspection ran
    relative to the semaphore, rather than only that it eventually did.
    """
    import runtime_cellpose

    rt = object.__new__(runtime_cellpose.CellposeRuntime.func_or_class)
    rt._soup_sem = asyncio.Semaphore(1)
    _INSPECTION_RAN.clear()

    def _base_state_dict(model_type="cpsam"):
        started.append(model_type)
        finished.append(model_type)
        _INSPECTION_RAN.set()
        return _tiny_sd()

    rt._base_state_dict = _base_state_dict
    rt._unwrap_sd = runtime_cellpose.CellposeRuntime.func_or_class._unwrap_sd
    rt._upload_weights = lambda url, path: None
    return rt


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A COMPLETED stock-initialised session on disk, with no ``init_checkpoint.pt``.

    Stock-initialised on purpose: that is the `else` branch #0070 added, the one that
    loads a second state dict and so the one this row's bound regressed on.
    """
    import sys
    import types

    sdir = tmp_path / "sess"
    sdir.mkdir()
    ckpt = sdir / "models" / "cp.pt"
    ckpt.parent.mkdir(parents=True)
    torch.save(_tiny_sd(), str(ckpt))
    assert not (sdir / "init_checkpoint.pt").exists(), "fixture must be stock-initialised"

    mod = types.ModuleType("training")
    mod.checkpoint_path = lambda sid: ckpt
    mod.session_dir = lambda sid: sdir
    mod.cellpose_base_model = lambda mt: "cpsam_v2"
    monkeypatch.setitem(sys.modules, "training", mod)
    return ckpt


@pytest.mark.asyncio
async def test_the_inspection_cannot_start_while_the_merge_holds_the_semaphore(
    tmp_path, session
):
    """THE arm this row exists for. Hold ``_soup_sem``; the deposit must wait."""
    started, finished = [], []
    rt = _runtime(tmp_path, started, finished)

    await rt._soup_sem.acquire()                      # stand in for a merge in flight
    task = asyncio.create_task(
        rt.prepare_contribution("sess", "https://example.invalid/put"))

    # Wait on an EVENT the inspection sets, not on event-loop yields.
    #
    # My first version of this spun `await asyncio.sleep(0)` twenty times and asserted
    # `started == []`. That passed against the UNFIXED tree as well, which makes it
    # worthless: `_inspect` is dispatched with `asyncio.to_thread`, so it runs in a worker
    # thread, and yielding the loop does not wait for a thread to be scheduled. The
    # assertion held because the thread had not got going yet, not because a lock stopped
    # it. Waiting on the event gives the unlocked code every chance to run and still
    # returns promptly when it does.
    loop = asyncio.get_running_loop()
    fired = await loop.run_in_executor(None, _INSPECTION_RAN.wait, 3.0)
    assert not fired, (
        "the inspection ran while _soup_sem was held — the deposit's state dicts can "
        "still sum with a concurrent merge's (#0087)"
    )
    assert started == []
    assert not task.done()

    rt._soup_sem.release()                            # the merge finishes
    info = await asyncio.wait_for(task, timeout=30)
    assert started == ["cpsam"], "the inspection never ran even after the release"
    assert info["l2_from_base_basis"] == pool.L2_BASIS_STOCK


@pytest.mark.asyncio
async def test_the_same_call_completes_when_the_semaphore_is_free(tmp_path, session):
    """NEGATIVE CONTROL for the arm above.

    Without this, a `prepare_contribution` that raised immediately, hung forever, or never
    reached the inspection at all would satisfy `started == []` and the first test would
    pass for the wrong reason.
    """
    started, finished = [], []
    rt = _runtime(tmp_path, started, finished)

    info = await asyncio.wait_for(
        rt.prepare_contribution("sess", "https://example.invalid/put"), timeout=30)

    assert started == ["cpsam"], "the inspection did not run even with the lock free"
    assert info["base_commitment"] is not None
    assert info["architecture_signature"]


@pytest.mark.asyncio
async def test_the_semaphore_is_released_even_when_the_inspection_raises(tmp_path, session):
    """A lock taken on the deposit path must not be able to wedge the merge path.

    `async with` guarantees this, so the test is cheap; it is here because the failure it
    rules out is a replica that stops merging entirely after one bad checkpoint, which is
    worse than the breach this row is about.
    """
    started, finished = [], []
    rt = _runtime(tmp_path, started, finished)

    def _boom(model_type="cpsam"):
        started.append(model_type)
        _INSPECTION_RAN.set()
        raise RuntimeError("deliberate: a corrupt checkpoint")

    rt._base_state_dict = _boom
    with pytest.raises(RuntimeError, match="deliberate"):
        await asyncio.wait_for(
            rt.prepare_contribution("sess", "https://example.invalid/put"), timeout=30)

    assert started == ["cpsam"]
    # The merge path must still be able to take it.
    await asyncio.wait_for(rt._soup_sem.acquire(), timeout=5)
    rt._soup_sem.release()
