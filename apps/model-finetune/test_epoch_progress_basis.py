"""The status this backend writes must satisfy the rule the page applies to it.

Clause 3 of this work is "epoch progress reported FOR THE UI", and that is not the
same claim as "epoch progress written somewhere". The Colab page refuses to make
any progress claim unless the basis field says exactly ``measured``:

    // src/utils/trainingProgress.ts, epochProgressOf()
    if (session.n_epochs_completed_basis !== MEASURED_BASIS) return { kind: 'none' };

with ``MEASURED_BASIS = 'measured'``. An earlier version of the live writer sent a
count with NO basis, which that rule discards — so a correct, per-epoch, honestly
reported count rendered as no progress at all, and nothing on this side failed.

So these tests assert on the basis field and then feed the whole status through a
Python transcription of the page's own rule. The transcription is the point: a
test that only checked "the field says measured" would pass a backend that got the
count, the total or the types wrong in a way the page silently rejects.

**The transcription is a copy and copies rot.** It is pinned to the constant and
the file it came from, and `test_the_transcribed_rule_still_matches_its_source`
states what would have to change for it to drift. If the page's rule moves, this
file is the thing that should have to be edited.
"""

from __future__ import annotations

import epoch_progress
import pytest

# ── the page's rule, transcribed ──────────────────────────────────────────────
# From src/utils/trainingProgress.ts on the Colab page branch:
#   export const MEASURED_BASIS = 'measured';
MEASURED_BASIS = "measured"


def _as_count(v):
    """The page's asCount: a non-negative integer, or None.

    Written to reject the same things: booleans are not counts, floats that are
    not whole are not counts, and a negative number is not a count.
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if float(v) != int(v) or int(v) < 0:
        return None
    return int(v)


def epoch_progress_of(session):
    """A transcription of ``epochProgressOf``. Returns a dict, not a shape."""
    if not session:
        return {"kind": "none"}
    if session.get("n_epochs_completed_basis") != MEASURED_BASIS:
        return {"kind": "none"}
    completed = _as_count(session.get("n_epochs_completed"))
    if completed is None:
        return {"kind": "none"}
    total = _as_count(session.get("n_epochs"))
    if total is None or total <= 0:
        return {"kind": "measured", "completed": completed, "total": None,
                "fraction": None}
    return {"kind": "measured", "completed": completed, "total": total,
            "fraction": min(1.0, completed / total)}


# ── what the live writer puts on the status ───────────────────────────────────


class _Recorder:
    """Stands in for training.write_status, capturing the kwargs written."""

    def __init__(self):
        self.writes = []

    def __call__(self, session_id, **kw):
        self.writes.append(kw)


@pytest.fixture()
def live_writes(monkeypatch):
    """Drive the SHIPPED _publish_epoch callback and collect what it writes."""
    import train_worker_cellpose as twc

    rec = _Recorder()
    monkeypatch.setattr(twc.training, "write_status", rec)
    on_epoch = twc._publish_epoch("sess-1", has_val=False)
    on_epoch(1, [0.9])
    on_epoch(2, [0.9, 0.7])
    return rec.writes


def test_the_live_writer_labels_its_count_measured(live_writes):
    """THE FINDING. Without the basis the page discards the count entirely."""
    assert live_writes, "the callback wrote nothing"
    for w in live_writes:
        assert w.get("n_epochs_completed_basis") == MEASURED_BASIS, (
            f"a live epoch write carries basis {w.get('n_epochs_completed_basis')!r}; "
            f"the page discards any status whose basis is not {MEASURED_BASIS!r}, so "
            f"this is a count nobody can display"
        )
        assert isinstance(w.get("n_epochs_completed"), int)


def test_a_mid_run_status_reads_as_progress_through_the_PAGES_rule(live_writes):
    """The end-to-end claim of clause 3: not "we wrote a number" but "the page
    can show it". Two of three epochs done must come out as measured 2/3."""
    status = {"status": "TRAINING", "n_epochs": 3, **live_writes[-1]}
    got = epoch_progress_of(status)
    assert got["kind"] == "measured", (
        f"the page's own rule yields {got!r} for a mid-run status, i.e. it would "
        f"display nothing"
    )
    assert got["completed"] == 2 and got["total"] == 3
    assert got["fraction"] == pytest.approx(2 / 3)


def test_the_rule_rejects_the_status_this_writer_used_to_produce(live_writes):
    """THE CONTROL on the transcription. If the rule accepted a basis-less status
    too, the test above would pass whatever the writer did."""
    without = {k: v for k, v in live_writes[-1].items()
               if k != "n_epochs_completed_basis"}
    assert epoch_progress_of({"n_epochs": 3, **without}) == {"kind": "none"}


@pytest.mark.parametrize(
    "bad",
    [
        {"n_epochs_completed_basis": "floor_if_completed", "n_epochs_completed": 3},
        {"n_epochs_completed_basis": None, "n_epochs_completed": 3},
        {"n_epochs_completed_basis": "measured", "n_epochs_completed": None},
        {"n_epochs_completed_basis": "measured", "n_epochs_completed": "3"},
        {"n_epochs_completed_basis": "measured", "n_epochs_completed": True},
        {"n_epochs_completed_basis": "measured", "n_epochs_completed": 2.5},
        {"n_epochs_completed_basis": "measured", "n_epochs_completed": -1},
    ],
)
def test_the_rule_yields_no_claim_for_each_way_a_status_can_be_wrong(bad):
    """Each rejection path separately, so "it renders" cannot rest on one shape."""
    assert epoch_progress_of({"n_epochs": 3, **bad})["kind"] == "none"


def test_a_missing_or_zero_total_still_reports_the_count():
    """The page reports the count alone rather than dividing by nothing. Pinned
    because a backend that omits n_epochs should still show progress."""
    for total in (None, 0):
        got = epoch_progress_of(
            {"n_epochs": total, "n_epochs_completed_basis": "measured",
             "n_epochs_completed": 4}
        )
        assert got["kind"] == "measured" and got["completed"] == 4
        assert got["total"] is None and got["fraction"] is None


def test_a_count_above_the_total_is_reported_not_hidden():
    """The page clamps the BAR but not the number, on the grounds that a count
    above the requested total is a backend fact worth seeing."""
    got = epoch_progress_of(
        {"n_epochs": 3, "n_epochs_completed_basis": "measured",
         "n_epochs_completed": 5}
    )
    assert got["completed"] == 5 and got["fraction"] == 1.0


def test_the_transcribed_rule_still_matches_its_source():
    """A guard on the COPY. It cannot read the other repository, so it states the
    two facts the transcription depends on; if either changes, this file is what
    has to be edited rather than the backend quietly drifting out of contract."""
    assert MEASURED_BASIS == "measured", (
        "the page's MEASURED_BASIS constant is transcribed here; if it changed, "
        "every basis this backend writes has to change with it"
    )
    assert epoch_progress.BASIS_MEASURED == MEASURED_BASIS, (
        "epoch_progress.BASIS_MEASURED and the page's MEASURED_BASIS have drifted "
        "apart; the terminal write keys the basis off the former and the page "
        "gates on the latter, so they must be the same string"
    )


# ── the terminal write must not downgrade a measured count ───────────────────


def test_a_reconciled_curve_keeps_the_measured_basis_at_the_end():
    """It used to write `floor_if_completed` with the REQUESTED total, so a run
    the page had been showing progress for lost the claim on success."""
    live = [0.9, 0.7, 0.5]
    assert epoch_progress.agreement(live, [0.9, 0.7, 0.5]) == \
        epoch_progress.BASIS_MEASURED
    status = {"n_epochs": 3, "n_epochs_completed": len(live),
              "n_epochs_completed_basis": "measured"}
    assert epoch_progress_of(status)["kind"] == "measured"


def test_an_unreconciled_curve_does_NOT_claim_measured():
    """The fallback has to stay a fallback: a live curve that cannot be
    reproduced from train_seg's own curve is not a measurement to advertise."""
    assert epoch_progress.agreement([0.9, 0.7], [0.9, 0.4]) == \
        epoch_progress.BASIS_UNRECONCILED
    assert epoch_progress.agreement([], [0.9]) == epoch_progress.BASIS_UNAVAILABLE
    # and the page makes no claim for the floor basis the fallback writes
    assert epoch_progress_of(
        {"n_epochs": 3, "n_epochs_completed": 3,
         "n_epochs_completed_basis": "floor_if_completed"}
    ) == {"kind": "none"}


# ── the terminal decision itself, now that it is a function ──────────────────


def _fields(ok, basis, curve, n_epochs=3):
    import train_worker_cellpose as twc

    return twc.terminal_epoch_fields(ok, basis, curve, n_epochs)


def test_a_reconciled_curve_is_reported_as_measured_with_the_OBSERVED_count():
    """Not the requested total: if cellpose ended early the observed count is the
    truer number, and it is the one the page will display."""
    assert _fields(True, epoch_progress.BASIS_MEASURED, [0.9, 0.7], 3) == (2, "measured")
    assert _fields(True, epoch_progress.BASIS_MEASURED, [0.9, 0.7, 0.5], 3) == (
        3, "measured")


@pytest.mark.parametrize(
    "basis, curve",
    [
        (epoch_progress.BASIS_UNRECONCILED, [0.9, 0.7]),
        (epoch_progress.BASIS_UNAVAILABLE, []),
        (epoch_progress.BASIS_MEASURED, []),  # basis says measured, nothing observed
    ],
)
def test_anything_short_of_a_reconciled_curve_falls_back_to_the_floor(basis, curve):
    """The fallback must stay a fallback, and it must use the REQUESTED total,
    which is what `floor_if_completed` means."""
    assert _fields(True, basis, curve, 3) == (3, "floor_if_completed")


def test_a_run_with_no_checkpoint_claims_nothing():
    """`ok` false means no checkpoint was produced, so there is no completed-epoch
    claim to make at all -- not a zero, which would read as a real measurement."""
    for basis in (epoch_progress.BASIS_MEASURED, epoch_progress.BASIS_UNAVAILABLE):
        assert _fields(False, basis, [0.9, 0.7], 3) == (None, None)


def test_the_terminal_fields_render_through_the_PAGES_rule():
    """End to end for the terminal write: the pair the helper returns, put on a
    status, must come out as progress the page will show."""
    completed, basis = _fields(True, epoch_progress.BASIS_MEASURED, [0.9, 0.7, 0.5], 3)
    got = epoch_progress_of(
        {"n_epochs": 3, "n_epochs_completed": completed,
         "n_epochs_completed_basis": basis}
    )
    assert got == {"kind": "measured", "completed": 3, "total": 3, "fraction": 1.0}
