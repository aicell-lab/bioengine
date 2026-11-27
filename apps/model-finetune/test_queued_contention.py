"""A caller must be able to tell "the slot is busy" from "the run broke".

WHAT WAS OBSERVED, by somebody driving the public app end to end. A second
`start_training` submitted while another run held the single training slot did
not become QUEUED. It went PREPARING -> FAILED after about 90 seconds with
`terminated_by: "entry"` and

    message: "Method call timed out: */manager-...:services.default.get_service"

`get_training_status` documents QUEUED as "another fine-tuning holds the single
training slot", and the concurrency cap is documented as serialising runs, so the
reporter reasonably expected QUEUED and got a raw RPC fault instead. Their words:
a caller cannot tell contention from a real failure.

WHY QUEUED WAS NEVER REACHED. `_run_training` prepares the whole dataset first --
resolving every image and label, downloading an init checkpoint, materialising the
pairs -- and only then looks at the semaphore. So for the entire preparation phase
a contended run is indistinguishable from an uncontended one, and anything that
fails during it reports whatever error it hit, with nothing saying a slot was busy.

WHAT IS NOT CLAIMED HERE. That contention CAUSED that `get_service` timeout. The
same report describes a worker pod roll and a transient "Target peer is not
connected" during the session, either of which explains an RPC timeout on its own,
and no test here reproduces one. What is reproduced is the part that made the
failure unreadable: contention was invisible, so the caller had no way to attribute
it. That is the defect these tests pin.

WHY THE STATUS STAYS `PREPARING` WHILE PREPARING, rather than being set to QUEUED
early, which would be the obvious way to "reach QUEUED sooner". `get_status`
sweeps a quiet run to STOPPED only from PREPARING or TRAINING -- QUEUED is
deliberately exempt (training.py, the status sweep), so a run marked QUEUED while
its replica dies stays QUEUED for ever. Writing QUEUED before preparation would
therefore hand every stuck preparation that same immunity, and preparation is the
phase that does the downloading. The contention is reported in its own field
instead, which costs nothing and keeps the sweep. One test below pins that
ordering so a later change has to read this paragraph before undoing it.
"""

from __future__ import annotations

import asyncio
import time

import pytest

import training

# `validate_session_id` enforces YYYY-MM-DD-HHMMSS-xxxxxxxx, so a short
# placeholder id is refused before any status is written.
FREE = "2026-11-27-100000-aaaaaaaa"
BUSY = "2026-11-27-100001-bbbbbbbb"


class _Runtime:
    def __init__(self, hold=None):
        self._hold = hold
        self.trained = []

    async def train(self, session_id=None, model_type=None, params=None):
        self.trained.append(session_id)
        if self._hold is not None:
            await self._hold


class _Stub:
    """Enough of the app to drive the real `_run_training`, and nothing else."""

    def __init__(self, prepare, runtime=None, slots=1):
        self._training_tasks = {}
        self._training_sem = asyncio.Semaphore(slots)
        self._prepare = prepare
        self._runtime = runtime or _Runtime()

    async def _preparing_heartbeat(self, session_id, interval=60.0):
        return None

    async def _prepare_training_data(self, session_id, *_a, **_kw):
        return await self._prepare(session_id)

    def _runtime_for(self, _model_type):
        return self._runtime


def _data():
    return {
        "patch_shape": [256, 256],
        "train_images": ["a.png"],
        "train_labels": ["a.geojson"],
        "val_images": [],
        "val_labels": [],
    }


async def _run(stub, session_id, params=None):
    import entry

    # `start_training` writes PREPARING before it creates the task; the harness
    # has to do the same or the run begins at UNKNOWN / "no such session" and
    # every status assertion below measures the harness instead of the app.
    # Measured: without this, the status during preparation reads
    # {"status": "UNKNOWN", "message": "no such session"}.
    training.write_status(session_id, status="PREPARING", created_at=time.time())
    await entry.EntryApp.func_or_class._run_training(
        stub, session_id, "cpsam", ["a.png"], ["a.geojson"], [], [], params or {}
    )


@pytest.fixture(autouse=True)
def sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "sessions_root", lambda: tmp_path / "s")
    yield


# ── the slot being busy has to be visible from the moment of submission ──────


@pytest.mark.asyncio
async def test_a_contended_run_says_the_slot_is_busy_WHILE_it_prepares():
    """THE case. The status is sampled DURING preparation, which is the window
    the reporter spent 90 seconds in with nothing to go on."""
    seen = {}
    gate = asyncio.Event()

    async def prepare(session_id):
        seen.update(training.get_status(session_id))
        gate.set()
        return _data()

    stub = _Stub(prepare)
    await stub._training_sem.acquire()  # another run holds the slot
    task = asyncio.create_task(_run(stub, BUSY))
    await asyncio.wait_for(gate.wait(), timeout=5)
    stub._training_sem.release()
    await asyncio.wait_for(task, timeout=5)

    assert seen.get("slot_busy") is True, (
        "while another run held the slot, the status carried no indication of "
        f"contention, so a caller cannot attribute a failure: {seen!r}"
    )
    assert "another" in str(seen.get("slot_busy_message", "")).lower(), (
        f"contention is flagged but not explained: {seen!r}"
    )


@pytest.mark.asyncio
async def test_an_UNCONTENDED_run_is_not_flagged():
    """The control. A flag set unconditionally would pass the test above while
    telling every caller their run is queued behind something."""
    seen = {}

    async def prepare(session_id):
        seen.update(training.get_status(session_id))
        return _data()

    await _run(_Stub(prepare), FREE)
    assert seen.get("slot_busy", False) is False, (
        f"an uncontended run was flagged as waiting for the slot: {seen!r}"
    )


@pytest.mark.asyncio
async def test_a_FAILURE_while_the_slot_was_busy_says_so_in_the_message():
    """The reporter's actual complaint. The raw error must survive -- it is the
    diagnostic -- with the contention added, not substituted."""

    async def prepare(_session_id):
        raise RuntimeError(
            "Method call timed out: */manager-abc:services.default.get_service"
        )

    stub = _Stub(prepare)
    await stub._training_sem.acquire()
    await _run(stub, BUSY)

    st = training.get_status(BUSY)
    assert st["status"] == "FAILED"
    msg = st["message"]
    assert "get_service" in msg, f"the underlying error was discarded: {msg}"
    assert "slot" in msg.lower() or "another" in msg.lower(), (
        f"a failure under contention does not mention the busy slot: {msg}"
    )


@pytest.mark.asyncio
async def test_a_FAILURE_with_the_slot_FREE_keeps_the_bare_message():
    """The other control. Appending the contention note unconditionally would
    tell every caller a slot was busy when none was."""

    async def prepare(_session_id):
        raise RuntimeError("cellpose ran out of memory on a 4096x4096 tile")

    await _run(_Stub(prepare), FREE)
    st = training.get_status(FREE)
    assert st["status"] == "FAILED"
    assert st["message"] == "cellpose ran out of memory on a 4096x4096 tile"


# ── the state machine itself must not drift ──────────────────────────────────


@pytest.mark.asyncio
async def test_the_status_is_still_PREPARING_while_preparing_not_QUEUED():
    """DELIBERATE, and the module docstring says why: QUEUED is exempt from the
    quiet-run sweep, so marking a run QUEUED before it has prepared would make
    every stuck preparation immune to being swept. If you are changing this,
    read that paragraph first -- the contention is reported in `slot_busy`
    precisely so the status does not have to carry it."""
    seen = {}
    gate = asyncio.Event()

    async def prepare(session_id):
        seen.update(training.get_status(session_id))
        gate.set()
        return _data()

    stub = _Stub(prepare)
    await stub._training_sem.acquire()
    task = asyncio.create_task(_run(stub, BUSY))
    await asyncio.wait_for(gate.wait(), timeout=5)
    stub._training_sem.release()
    await asyncio.wait_for(task, timeout=5)
    assert seen.get("status") == "PREPARING", (
        f"the status during preparation changed to {seen.get('status')!r}; the "
        f"quiet-run sweep does not act on QUEUED, so this is not a free change"
    )


@pytest.mark.asyncio
async def test_a_contended_run_STILL_reaches_QUEUED_before_it_waits():
    """The pre-existing behaviour, which the fix must not displace: once
    preparation is done and the slot is still held, the run is QUEUED."""
    states = []
    prepared = asyncio.Event()
    release = asyncio.Event()

    async def prepare(_session_id):
        prepared.set()
        await release.wait()
        return _data()

    stub = _Stub(prepare)
    await stub._training_sem.acquire()
    task = asyncio.create_task(_run(stub, BUSY))
    await asyncio.wait_for(prepared.wait(), timeout=5)
    release.set()
    # Let the run get past preparation and reach the semaphore wait.
    for _ in range(200):
        await asyncio.sleep(0.005)
        states.append(training.get_status(BUSY).get("status"))
        if "QUEUED" in states:
            break
    assert "QUEUED" in states, (
        f"a run waiting for a held slot never reported QUEUED: {sorted(set(states))}"
    )
    stub._training_sem.release()
    await asyncio.wait_for(task, timeout=5)
