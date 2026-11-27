"""A run that cannot have the GPU slot must say QUEUED *before* it prepares.

``MODEL_FINETUNE_MAX_CONCURRENT_TRAINING`` serialises training correctly — the
mutual exclusion was never broken. What was wrong is WHEN the caller is told.
``_run_training`` resolved the dataset and downloaded ``init_checkpoint`` first
and only then, as the last step before training, sampled
``self._training_sem.locked()`` to decide whether to write QUEUED. So a second
``start_training`` spent that whole window reporting PREPARING, and any transient
fault in it — every step there is a Hypha or S3 round trip, and the checkpoint
download carries a 600 s ceiling — surfaced as FAILED with an RPC message instead
of as contention. A caller could not tell "the slot is busy" from "something
broke".

WHY THE TEST IS SHAPED LIKE THIS, because two obvious shapes do not work.

*It must not reach the semaphore.* The slot is deliberately held for the whole
test, so letting ``_run_training`` arrive at ``async with self._training_sem``
would block for ever. A hanging test is not a red arm: pytest-timeout is not
installed in this repo, so it would hang CI rather than report, and the assertion
it would have made is never reached either way. The probe therefore RAISES as
soon as it has recorded what it came to observe, which makes
``_run_training``'s own ``except Exception`` the exit and keeps the test bounded
with no timeout anywhere.

*It must observe the status AT the moment preparation begins*, not afterwards.
Asserting on the final status cannot discriminate: the fix and the defect both
end at FAILED here, because the probe raises in both. What differs is only what
the status document said when preparation was entered, so the probe reads it
there and the assertion is on that snapshot.

Plain pytest against a temporary HOME — no GPU, no Ray, no live Hypha.
"""

from __future__ import annotations

import asyncio

import pytest

import training

# The shape `training.session_dir` accepts (YYYY-MM-DD-HHMMSS-xxxxxxxx); a
# descriptive id cannot be a directory name here.
SESSION = "2026-11-27-150000-0000a048"


@pytest.fixture(autouse=True)
def _sandbox_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))


def _method_of(name: str, instance):
    """The real coroutine function off the Serve-wrapped class."""
    import entry

    return getattr(entry.EntryApp.func_or_class, name).__get__(instance)


class _Probe:
    """Enough of EntryApp to enter preparation, and nothing past it."""

    def __init__(self, sem: asyncio.Semaphore):
        self._training_tasks = {}
        self._training_sem = sem
        self.status_when_preparation_began = None
        self.prepared = False

    async def _preparing_heartbeat(self, *a, **k):
        return None

    async def _prepare_training_data(self, session_id, *a, **k):
        # Record, then refuse to go further. See the module docstring: raising is
        # what keeps this bounded while the slot is held.
        self.prepared = True
        self.status_when_preparation_began = training.read_status(session_id).get("status")
        raise RuntimeError("probe: stop at preparation")

    async def _http_retry(self, *a, **k):  # pragma: no cover - not reached
        raise AssertionError("the probe raises before any download")

    def _bump_transport(self, **k):  # pragma: no cover - not reached
        return None


async def _run_with_slot_held(occupied: bool):
    sem = asyncio.Semaphore(1)
    probe = _Probe(sem)
    training.write_status(SESSION, status="PREPARING", message="submitted")
    if occupied:
        await sem.acquire()  # somebody else is training
    await _method_of("_run_training", probe)(
        session_id=SESSION,
        model_type="vit_b_lm",
        train_images=[],
        train_labels=[],
        val_images=None,
        val_labels=None,
        params={},
    )
    return probe


@pytest.mark.asyncio
async def test_a_BUSY_slot_is_reported_as_QUEUED_BEFORE_the_dataset_is_prepared():
    """The clause: a second concurrent start reaches QUEUED.

    This is the arm that was RED before the gate moved. With the slot held, the
    status document must already read QUEUED by the time preparation starts —
    otherwise the caller sees PREPARING for the whole resolve-and-download
    window and learns nothing about contention.
    """
    probe = await _run_with_slot_held(occupied=True)
    assert probe.prepared, (
        "preparation never ran, so this test observed nothing — it is not evidence "
        "either way"
    )
    assert probe.status_when_preparation_began == "QUEUED", (
        "with the GPU slot held, the run still said "
        f"{probe.status_when_preparation_began!r} while it prepared; a caller cannot "
        "tell contention from a real failure during that window"
    )


@pytest.mark.asyncio
async def test_a_FREE_slot_still_reports_PREPARING_and_is_not_mislabelled_QUEUED():
    """The other direction, which is what stops the fix being 'always say QUEUED'.

    A mutation that writes QUEUED unconditionally would satisfy the test above
    and be plainly wrong — nothing is queued when the slot is free. Without this
    arm that mutation passes, so this is the one that makes the pair meaningful.
    """
    probe = await _run_with_slot_held(occupied=False)
    assert probe.prepared, "preparation never ran, so this observed nothing"
    assert probe.status_when_preparation_began == "PREPARING", (
        "with the slot FREE the run reported "
        f"{probe.status_when_preparation_began!r}; only a run that actually waits "
        "may be labelled QUEUED"
    )
