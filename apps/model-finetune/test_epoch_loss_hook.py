"""The per-epoch loss hook must reproduce ``train_seg``'s own arithmetic.

Why a hook at all, and why these arms: cellpose's log record cannot carry
per-epoch progress (it fires only on ``iepoch == 5 or iepoch % 10 == 0``, so the
app's default 5-epoch run reports epoch 0 and nothing else), and the printed
``train_loss`` on the records that do fire is a mean over every epoch since the
previous report, not that epoch's loss. The hook exists to produce the real
per-epoch number, which means the only question worth testing is whether its
arithmetic is cellpose's. Quoted from ``cellpose/train.py`` at the pinned
4.2.1.1::

    for k in range(0, nimg_per_epoch, batch_size):
        ...
        train_loss = loss.item()
        train_loss *= len(X)
        train_losses[iepoch] += train_loss
    train_losses[iepoch] /= nimg_per_epoch

Note the divisor: ``nimg_per_epoch``, not the number of samples actually summed.
They differ whenever ``batch_size`` does not divide ``nimg_per_epoch``, and one
arm below is built on exactly that case — without it a mean-of-batches
implementation would pass every other arm.

No torch, no cellpose, no GPU: the accumulator is arithmetic, and the hook takes
its loss functions and its ``grad_enabled`` predicate as arguments.
"""

from __future__ import annotations

import math
import sys

import pytest

import epoch_progress as ep

_ABSENT = object()


class _Tensor:
    """Only what the hook touches: ``.item()`` and ``.shape``."""

    def __init__(self, value: float, batch: int = 1) -> None:
        self._value = float(value)
        self.shape = (batch,)

    def item(self) -> float:
        return self._value


def _cellpose_epoch_loss(batch_losses, nimg_per_epoch):
    """train_seg's own per-epoch figure, computed independently of the module
    under test from the quoted source. *batch_losses* is ``[(loss, len(X)), ...]``."""
    return sum(loss * n for loss, n in batch_losses) / nimg_per_epoch


def test_the_epoch_loss_is_train_segs_epoch_loss():
    seen = []
    tracker = ep.EpochLossTracker(
        nimg_per_epoch=4, on_epoch=lambda done, curve: seen.append((done, curve[-1]))
    )
    batches = [(2.0, 1), (1.0, 1), (3.0, 1), (0.5, 1)]
    for loss, n in batches:
        tracker.observe_batch(loss, n)

    assert len(seen) == 1, "one epoch of batches did not close exactly one epoch"
    done, loss = seen[0]
    assert done == 1
    assert math.isclose(loss, _cellpose_epoch_loss(batches, 4))
    assert math.isclose(loss, 1.625)


def test_each_batch_is_weighted_by_its_sample_count():
    """THE DISCRIMINATING ARM, and the one that pins ``train_loss *= len(X)``.

    With batch_size=2 and nimg_per_epoch=3 the last batch holds ONE sample, so
    the two batches carry different weight: cellpose's figure is
    (1.0*2 + 4.0*1)/3 = 2.0, while dropping the weighting gives (1.0 + 4.0)/3 =
    1.67 and a plain mean over the two batches gives 2.5. All three are 'an
    average loss' and every other arm in this file is satisfied by any of them,
    because they only ever use equal-sized batches.

    What is NOT observable, and so is not claimed here: the divisor itself.
    cellpose divides by ``nimg_per_epoch`` rather than by the samples it summed,
    but its batches tile the epoch exactly (``range(0, nimg_per_epoch,
    batch_size)`` with a truncated last batch), so the two quantities are equal
    at every reachable epoch boundary. The code follows cellpose's spelling
    because a future caller passing ``nimg_per_epoch`` explicitly would separate
    them; a test cannot tell them apart today and does not pretend to.
    """
    seen = []
    tracker = ep.EpochLossTracker(
        nimg_per_epoch=3, on_epoch=lambda done, curve: seen.append(curve[-1])
    )
    batches = [(1.0, 2), (4.0, 1)]
    for loss, n in batches:
        tracker.observe_batch(loss, n)

    assert len(seen) == 1
    assert math.isclose(seen[0], _cellpose_epoch_loss(batches, 3))
    assert math.isclose(seen[0], 2.0)
    # The mean-over-batches answer, which this must NOT be.
    assert not math.isclose(seen[0], (1.0 + 4.0) / 2)


def test_epochs_close_on_the_sample_boundary_not_on_a_batch_count():
    ticks = []
    tracker = ep.EpochLossTracker(
        nimg_per_epoch=4, on_epoch=lambda done, curve: ticks.append(done)
    )
    for _ in range(12):
        tracker.observe_batch(1.0, 1)
    assert ticks == [1, 2, 3], "12 samples at 4 per epoch is 3 epochs"
    assert tracker.epochs_completed == 3
    assert tracker.epoch_losses == [1.0, 1.0, 1.0]


def test_a_batch_of_no_samples_cannot_close_an_epoch():
    """``_batch_size_of`` returns 0 for a label batch it cannot read. Counting
    that as one sample would let an unreadable observation tick the counter."""
    ticks = []
    tracker = ep.EpochLossTracker(
        nimg_per_epoch=2, on_epoch=lambda done, curve: ticks.append(done)
    )
    for _ in range(10):
        tracker.observe_batch(1.0, 0)
    assert ticks == []
    assert tracker.epochs_completed == 0


def test_a_tracker_without_a_boundary_is_refused():
    with pytest.raises(ValueError, match="nimg_per_epoch"):
        ep.EpochLossTracker(nimg_per_epoch=0)


# --- the hook: which calls it counts, and which it must not ------------------


def _hooked(nimg_per_epoch, grad_enabled, class_loss=None):
    seen = []
    tracker = ep.EpochLossTracker(
        nimg_per_epoch=nimg_per_epoch,
        on_epoch=lambda done, curve: seen.append((done, curve[-1])),
    )
    hook = ep.CellposeLossHook(tracker, grad_enabled=grad_enabled)
    seg = hook.wrap_seg(lambda lbl, y, device=None: y)
    cls = hook.wrap_class(lambda lbl, y, class_weights=None: class_loss)
    return tracker, hook, seg, cls, seen


def test_the_class_loss_term_is_added_to_the_seg_term():
    """cellpose does ``loss = _loss_fn_seg(...)`` then, for a net with more than
    three output channels, ``loss += _loss_fn_class(...)`` and only then takes
    ``.item()``. A hook reading the seg term alone would under-report every
    epoch of exactly the models that have the extra head."""
    tracker, hook, seg, cls, seen = _hooked(
        1, grad_enabled=lambda: True, class_loss=_Tensor(0.25)
    )
    seg(_Tensor(0.0, batch=1), _Tensor(1.0, batch=1))
    cls(_Tensor(0.0, batch=1), _Tensor(1.0, batch=1))
    hook.close()
    assert seen == [(1, 1.25)]


def test_validation_batches_are_not_counted():
    """CONTROL for the train/validation split. cellpose's validation pass calls
    the SAME loss function inside ``torch.no_grad()``. Counting those would both
    close epochs early and mix a validation number into a training curve — so
    this arm feeds two validation batches per training batch and asserts the
    epoch boundary is unmoved."""
    grad = {"on": True}
    tracker, hook, seg, cls, seen = _hooked(2, grad_enabled=lambda: grad["on"])

    for _ in range(2):
        seg(_Tensor(0.0, batch=1), _Tensor(1.0, batch=1))  # training batch
        grad["on"] = False
        seg(_Tensor(0.0, batch=1), _Tensor(9.0, batch=1))  # validation batch
        seg(_Tensor(0.0, batch=1), _Tensor(9.0, batch=1))
        grad["on"] = True
    hook.close()

    assert hook.skipped_no_grad == 4, "the validation batches were not recognised"
    assert seen == [(1, 1.0)], "validation batches moved the epoch boundary"


def test_the_last_batch_of_a_run_is_not_lost():
    """A batch is only complete once the next seg call arrives, so the final
    epoch exists only if the caller closes the hook. Without this the run's last
    epoch silently never reports."""
    tracker, hook, seg, cls, seen = _hooked(1, grad_enabled=lambda: True)
    seg(_Tensor(0.0, batch=1), _Tensor(2.0, batch=1))
    assert seen == [], "the batch was counted before it could be completed"
    hook.close()
    assert seen == [(1, 2.0)]


def test_a_missing_loss_function_installs_nothing(monkeypatch):
    """FAIL-SOFT. A cellpose that renames the loss functions must leave the run
    untouched and claim nothing, rather than patch half of it."""
    saved = {}
    for name, module in (
        ("torch", _FakeTorch()),
        ("cellpose", _FakeModule()),
        ("cellpose.train", _FakeModule()),  # no _loss_fn_seg on it
    ):
        saved[name] = sys.modules.get(name, _ABSENT)
        sys.modules[name] = module
    sys.modules["cellpose"].train = sys.modules["cellpose.train"]
    try:
        assert ep.install_cellpose_hook(ep.EpochLossTracker(1)) is None
    finally:
        for name, prev in saved.items():
            if prev is _ABSENT:
                del sys.modules[name]
            else:
                sys.modules[name] = prev


def test_the_hook_installs_over_both_loss_functions_when_they_are_there():
    """The positive half of the arm above: without it, "returns None" could be
    the only behaviour this function ever has and the fail-soft test would pass
    for the wrong reason."""
    saved = {}
    train_mod = _FakeModule()
    train_mod._loss_fn_seg = lambda lbl, y, device=None: y
    train_mod._loss_fn_class = lambda lbl, y, class_weights=None: y
    originals = (train_mod._loss_fn_seg, train_mod._loss_fn_class)
    for name, module in (
        ("torch", _FakeTorch()),
        ("cellpose", _FakeModule()),
        ("cellpose.train", train_mod),
    ):
        saved[name] = sys.modules.get(name, _ABSENT)
        sys.modules[name] = module
    sys.modules["cellpose"].train = train_mod
    try:
        hook = ep.install_cellpose_hook(ep.EpochLossTracker(1))
        assert hook is not None
        assert train_mod._loss_fn_seg is not originals[0]
        assert train_mod._loss_fn_class is not originals[1]
    finally:
        for name, prev in saved.items():
            if prev is _ABSENT:
                del sys.modules[name]
            else:
                sys.modules[name] = prev


class _FakeModule:
    pass


class _FakeTorch:
    @staticmethod
    def is_grad_enabled():
        return True


# --- reconciliation ----------------------------------------------------------


def test_a_matching_curve_earns_the_measured_basis():
    assert ep.agreement([1.0, 2.0], [1.0, 2.0]) == ep.BASIS_MEASURED


def test_a_curve_that_does_not_match_is_not_called_measured():
    assert ep.agreement([1.0, 2.0], [1.0, 2.5]) == ep.BASIS_UNRECONCILED
    assert ep.agreement([1.0], [1.0, 2.0]) == ep.BASIS_UNRECONCILED


def test_an_unchecked_curve_is_not_called_measured_either():
    """A run whose returned curve is absent (a train_seg whose signature moved)
    leaves the live curve UNCHECKED, which is not the same statement as checked
    and equal — so it must not inherit the stronger word."""
    assert ep.agreement([1.0, 2.0], []) == ep.BASIS_UNRECONCILED
    assert ep.agreement([1.0, 2.0], None) == ep.BASIS_UNRECONCILED


def test_no_observed_epoch_is_reported_as_unavailable():
    assert ep.agreement([], [1.0]) == ep.BASIS_UNAVAILABLE


def test_float_association_alone_does_not_break_the_match():
    """The hook and train_seg sum the same terms, so only association differs;
    the tolerance exists for that and must not be wide enough to accept a real
    divergence."""
    assert ep.agreement([0.1 + 0.2], [0.30000000000000004]) == ep.BASIS_MEASURED
    assert ep.agreement([0.3001], [0.3]) == ep.BASIS_UNRECONCILED


# --- the recorder: one row per epoch, from two sources ----------------------


def test_the_hook_and_the_log_land_in_one_row_per_epoch():
    writes = []
    rec = ep.EpochRecorder(lambda **fields: writes.append(fields))
    rec.record_epoch_loss(1, 2.0)  # hook: epoch 0 completed
    rec.annotate(0, val_loss=1.9, lr=1e-4)  # log: epoch 0's validation
    rec.record_epoch_loss(2, 1.5)  # hook: epoch 1

    rows = rec.rows()
    assert [r["epoch"] for r in rows] == [0, 1]
    assert rows[0] == {"epoch": 0, "train_loss": 2.0, "val_loss": 1.9, "lr": 1e-4}
    # Epoch 1 is off cellpose's reporting cadence, so it has a training loss and
    # no validation loss. That asymmetry is the normal case, not a gap.
    assert rows[1]["train_loss"] == 1.5 and rows[1]["val_loss"] is None
    assert len(writes) == 3, "progress was not published on every observation"


def test_an_annotation_never_erases_the_hooks_train_loss():
    """The log record carries no per-epoch training loss, so its absence must not
    overwrite the one the hook measured."""
    rec = ep.EpochRecorder(lambda **fields: None)
    rec.record_epoch_loss(1, 2.0)
    rec.annotate(0, val_loss=None, lr=None, train_loss_window_mean=1.75)
    row = rec.rows()[0]
    assert row["train_loss"] == 2.0
    assert row["train_loss_window_mean"] == 1.75


def test_the_measured_basis_is_claimed_only_once_an_epoch_is_observed():
    rec = ep.EpochRecorder(lambda **fields: None)
    assert "n_epochs_completed" not in rec.status_fields()
    rec.annotate(0, val_loss=1.0, lr=1e-4)
    assert "n_epochs_completed" not in rec.status_fields(), (
        "a log record is not an epoch count: cellpose logs epoch 0 before this "
        "worker can know the epoch's batches all ran"
    )
    rec.record_epoch_loss(1, 2.0)
    fields = rec.status_fields()
    assert fields["n_epochs_completed"] == 1
    assert fields["n_epochs_completed_basis"] == ep.BASIS_MEASURED


def test_withholding_the_loss_still_reports_the_epoch_number():
    """``metrics`` selects which metrics are COMPUTED; progress is not a metric."""
    rec = ep.EpochRecorder(lambda **fields: None)
    rec.record_epoch_loss(1, 2.0, publish_loss=False)
    fields = rec.status_fields()
    assert fields["n_epochs_completed"] == 1
    assert fields["epochs"][0]["train_loss"] is None
