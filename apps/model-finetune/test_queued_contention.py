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


async def _done(value):
    """`_Stub._prepare` is awaited, so a plain lambda has to return a
    coroutine; this keeps the arms below to one line each."""
    return value


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
    assert (
        seen.get("slot_busy_message") == _slot_busy_message()
    ), f"the flag is set but not explained in the module's own words: {seen!r}"


@pytest.mark.asyncio
async def test_an_UNCONTENDED_run_is_not_flagged():
    """The control. A flag set unconditionally would pass the test above while
    telling every caller their run is queued behind something."""
    seen = {}

    async def prepare(session_id):
        seen.update(training.get_status(session_id))
        return _data()

    await _run(_Stub(prepare), FREE)
    assert (
        seen.get("slot_busy", False) is False
    ), f"an uncontended run was flagged as waiting for the slot: {seen!r}"


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
    assert (
        "slot" in msg.lower() or "another" in msg.lower()
    ), f"a failure under contention does not mention the busy slot: {msg}"


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
    assert (
        "QUEUED" in states
    ), f"a run waiting for a held slot never reported QUEUED: {sorted(set(states))}"
    stub._training_sem.release()
    await asyncio.wait_for(task, timeout=5)


# ── the annotation must not be able to cost more than itself ─────────────────
#
# The contention write is a LABEL: it carries no outcome, and the run is the
# thing that matters. Two ways that balance can be got wrong, one arm each.
#
# The record lock made both reachable. `write_status` raises
# `RunRecordLockTimeout` when the mutex cannot be taken and
# `RunRecordUnreadable` when the file will not parse -- correct for a terminal
# write, wrong for this one -- and a second writer for the record genuinely
# exists in the process (`_runs_sync_loop`).


@pytest.mark.asyncio
async def test_a_lost_contention_write_does_not_disturb_the_run(monkeypatch):
    """THE degraded path. `write_status_best_effort` returns False when the
    mutex cannot be taken; the run must simply train, unlabelled."""
    calls = []

    def refuse(session_id, **fields):
        calls.append(fields)
        return False  # exactly what a lock timeout yields

    monkeypatch.setattr(training, "write_status_best_effort", refuse)

    runtime = _Runtime()
    stub = _Stub(lambda _sid: _done(_data()), runtime=runtime)
    await stub._training_sem.acquire()
    task = asyncio.create_task(_run(stub, BUSY))
    await asyncio.sleep(0)
    stub._training_sem.release()
    await asyncio.wait_for(task, timeout=5)

    assert calls, "the contention write never happened, so this arm proves nothing"
    assert runtime.trained == [
        BUSY
    ], "a contention write that could not land stopped the run from training"


@pytest.mark.asyncio
async def test_a_RAISING_contention_write_is_still_recorded_as_a_failure(monkeypatch):
    """THE fence. This arm is about WHERE the write sits, not which function it
    calls: the write used to sit before `_run_training`'s `try`, so an exception
    from it escaped with no handler in scope -- no FAILED record, the `finally`
    skipped, the heartbeat task dangling, the session left in `_training_tasks`,
    and the run stuck at PREPARING until the quiet-run sweep called it STOPPED.
    A label must not be able to do that. Whatever the write raises, the run's
    outcome has to be written somewhere a caller can read it."""

    def boom(session_id, **fields):
        raise RuntimeError("the record mutex could not be taken")

    monkeypatch.setattr(training, "write_status_best_effort", boom)

    stub = _Stub(lambda _sid: _done(_data()))
    await stub._training_sem.acquire()
    # BOUNDED, and not as hygiene. The slot is held and never released here, so
    # a change that let the run past this write would reach `async with
    # self._training_sem` and wait for ever -- and a mutant that HANGS is not a
    # red arm, it is a gate that never reports. Measured: without this bound the
    # mutation arm that restores the raising write sat in a container for nine
    # minutes instead of failing in two seconds.
    # No `pytest.raises`: under the fix `_run_training` CATCHES the write's
    # exception and records FAILED, so it returns normally. A mutant that hangs
    # instead trips the bound and the arm goes red, which is the point.
    await asyncio.wait_for(_run(stub, BUSY), timeout=5)

    st = training.get_status(BUSY)
    assert st["status"] == "FAILED", (
        f"the run left no terminal record, so it is indistinguishable from one "
        f"that is still preparing: {st!r}"
    )
    assert st.get("terminated_by") == "entry"
    assert BUSY not in stub._training_tasks


@pytest.mark.asyncio
async def test_the_contention_note_is_still_true_after_the_run_finishes():
    """`slot_busy` records the state AT SUBMISSION and is never written back, so
    whatever it says is still on the record of a COMPLETED run. The live answer
    to "is it waiting?" is `status`, which reaches QUEUED on its own; a
    present-tense note here reads as that and contradicts a finished run."""
    runtime = _Runtime()
    stub = _Stub(lambda _sid: _done(_data()), runtime=runtime)
    await stub._training_sem.acquire()
    task = asyncio.create_task(_run(stub, BUSY))
    await asyncio.sleep(0)
    stub._training_sem.release()
    await asyncio.wait_for(task, timeout=5)

    note = str(training.get_status(BUSY).get("slot_busy_message", ""))
    assert note, "no contention note was recorded"
    assert "started" in note, f"the note does not say WHEN the slot was held: {note}"
    for present_tense in (" is preparing", "will train"):
        assert present_tense not in note, (
            f"the note is present-tense, so it still claims {present_tense.strip()!r} "
            f"on a run that has finished: {note}"
        )


# ── a signal a caller cannot find is not a signal ───────────────────────────


def test_the_status_method_documents_the_contention_fields():
    """The defect being fixed is that a caller cannot tell contention from a
    failure. A new key they have never heard of does not fix that: the RPC
    schema is what a caller reads, so `get_training_status`'s own description
    has to name the fields."""
    import entry

    doc = getattr(entry.EntryApp.func_or_class, "get_training_status").__doc__ or ""
    assert "QUEUED" in doc, "the wrong method was read; this is not the status doc"
    assert "slot_busy" in doc, (
        "get_training_status does not mention slot_busy, so the only way to find "
        "the field is to already know it exists"
    )
    assert "slot_busy_message" in doc


def test_the_README_documents_the_contention_fields():
    from pathlib import Path

    text = (Path(__file__).with_name("README.md")).read_text(encoding="utf-8")
    assert len(text) > 5000, f"README read implausibly short ({len(text)})"
    assert "slot_busy" in text, "the README does not document the contention fields"


# ── the note has to survive the field's length limit ─────────────────────────
#
# Found in review, and it was the ordinary case rather than an edge one. The
# note was appended and the whole string THEN clipped to the status field's
# limit, so the note -- sitting at the clipped end -- vanished for any
# underlying error longer than about 630 characters, while the README and
# get_training_status both promise a failure under contention carries it. A
# Hypha RemoteException's traceback is routinely longer than that.


@pytest.mark.asyncio
@pytest.mark.parametrize("n", [10, 700, 790, 2000, 20000])
async def test_the_contention_note_survives_however_long_the_error_is(n):
    """Parametrised over the boundary, not just past it.

    10 and 700 passed before the fix and must keep passing; 790 and up are the
    regression. Covering both sides is what stops a future change trading one
    for the other.
    """

    async def prepare(_session_id):
        raise RuntimeError("E" * n)

    stub = _Stub(prepare)
    await stub._training_sem.acquire()
    await _run(stub, BUSY)

    st = training.get_status(BUSY)
    assert st["status"] == "FAILED"
    msg = st["message"]
    # A WORDING-STABLE FRAGMENT, NOT THE CONSTANT, and that is deliberate.
    # This arm's whole value is being RED against the pre-fix code, which
    # carried a differently worded note ("held the training slot"). Asserting
    # the current constant made all five lengths fail there on wording instead
    # of on truncation -- uniform red, which says nothing about WHICH length
    # loses the note. "training slot" is in both wordings, so the arm measures
    # presence across the boundary. Exact wording is pinned separately, by the
    # field test, where drift is the thing under test.
    assert "training slot" in msg, (
        f"a {n}-character error lost the contention note entirely; the note is "
        f"what the README and get_training_status promise is there "
        f"(message was {len(msg)} chars)"
    )
    # The field limit is still respected -- the fix must not have bought the
    # note by writing an unbounded message.
    assert (
        len(msg) <= _status_limit()
    ), f"the message is {len(msg)} chars, over the limit"
    # And the error itself still leads, so the diagnostic is not sacrificed.
    assert msg.startswith("EEE"), f"the underlying error no longer leads: {msg[:60]!r}"


def _slot_busy_message():
    """The module's own contention sentence, asserted by reference.

    A duplicated sentence is what drifted in the first place: the status field
    said "every training slot" while both notes still said "the training slot",
    and no test could see the disagreement because each one quoted whichever
    copy it was written against.
    """
    import entry

    return entry._SLOT_BUSY_MESSAGE


def _slot_busy_note():
    """The bracketed form appended to a failure message, derived from the
    sentence above so the two cannot disagree."""
    import entry

    return entry._SLOT_BUSY_NOTE


def _status_limit():
    """The limit, tolerating its absence.

    Read with a default because this test must be runnable against the code
    BEFORE the constant existed -- that is the only way its red check means
    anything. Reading it as a hard attribute made every parametrisation fail on
    AttributeError against the pre-fix tree, including the two cases that were
    supposed to PASS there, which turned a precise red check into a uniform one
    that proved nothing.
    """
    import entry

    return getattr(entry, "_STATUS_MESSAGE_LIMIT", 800)


@pytest.mark.asyncio
async def test_an_uncontended_long_error_is_clipped_and_carries_no_note():
    """The control. Keeping the note must not mean inventing one, and an
    uncontended failure must still be clipped to the limit."""

    async def prepare(_session_id):
        raise RuntimeError("E" * 2000)

    await _run(_Stub(prepare), FREE)
    st = training.get_status(FREE)
    msg = st["message"]
    assert "training slot" not in msg
    assert len(msg) == _status_limit()
