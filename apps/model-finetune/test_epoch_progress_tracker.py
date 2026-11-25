"""The per-epoch loss curve reproduces train_seg's own, batch by batch.

This covers the accumulator and the loss hook, which exist because cellpose offers
no callback and no per-epoch counter and returns its curves only on completion. The
approach is to recompute what cellpose computes from the same per-batch terms, so
the test that matters replays cellpose's call pattern — ``_loss_fn_seg``, sometimes
``_loss_fn_class``, a validation pass under ``no_grad`` — and compares the live
curve against the one ``train_seg`` would have returned.

Everything here runs without cellpose, torch or a GPU: the hook takes its loss
functions and its ``grad_enabled`` predicate as arguments, and only
``install_cellpose_hook`` touches the real modules.
"""

import pytest

import epoch_progress as ep


class FakeTensor:
    """Stands in for a loss tensor: the hook only calls ``.item()`` on it."""

    def __init__(self, value):
        self._value = float(value)

    def item(self):
        return self._value


class FakeBatch:
    """Stands in for ``lbl``: the hook reads its leading dimension as len(X)."""

    def __init__(self, n):
        self.shape = (n,)


def test_an_epoch_closes_at_nimg_per_epoch_and_divides_by_it():
    """cellpose divides the epoch loss by nimg_per_epoch, NOT by the samples it
    actually summed. The two differ whenever batch_size does not divide it, and
    matching cellpose is the whole point of recomputing rather than averaging."""
    seen = []
    t = ep.EpochLossTracker(nimg_per_epoch=10, on_epoch=lambda n, curve: seen.append((n, list(curve))))
    # Four batches of 3 = 12 samples, so the epoch closes on the fourth and the
    # divisor is 10, not 12.
    for _ in range(4):
        t.observe_batch(1.0, 3)
    assert t.epochs_completed == 1
    assert t.epoch_losses == [pytest.approx(4 * 1.0 * 3 / 10)]
    assert seen == [(1, [pytest.approx(1.2)])]


def test_a_zero_sample_batch_cannot_close_an_epoch():
    """A malformed observation must not advance the epoch boundary; a batch of
    nothing is not a batch."""
    t = ep.EpochLossTracker(nimg_per_epoch=4)
    t.observe_batch(5.0, 0)
    t.observe_batch(5.0, -1)
    assert t.epochs_completed == 0
    t.observe_batch(2.0, 4)
    assert t.epochs_completed == 1
    assert t.epoch_losses == [pytest.approx(2.0)]


def test_a_nonpositive_nimg_per_epoch_is_refused():
    """Without an epoch length there is no boundary to count, and silently using 1
    would report every batch as an epoch."""
    for bad in (0, -5):
        with pytest.raises(ValueError):
            ep.EpochLossTracker(nimg_per_epoch=bad)


def test_the_callback_receives_a_copy_it_may_keep():
    kept = []
    t = ep.EpochLossTracker(2, on_epoch=lambda n, curve: kept.append(curve))
    t.observe_batch(1.0, 2)
    t.observe_batch(3.0, 2)
    assert kept[0] == [pytest.approx(1.0)], "the first callback saw a later mutation"
    assert kept[1] == [pytest.approx(1.0), pytest.approx(3.0)]


def _replay(hook, batches, grad):
    """Replay cellpose's per-batch call pattern through the wrapped functions."""
    seg = hook.wrap_seg(lambda lbl, y: FakeTensor(y))
    cls = hook.wrap_class(lambda lbl, y, class_weights=None: FakeTensor(y))
    for loss, n, extra in batches:
        grad["on"] = True
        seg(FakeBatch(n), loss)
        if extra is not None:
            cls(FakeBatch(n), extra)
    hook.close()
    return seg, cls


def test_the_live_curve_equals_what_train_seg_would_have_returned():
    """GROUND TRUTH. cellpose sums loss.item()*len(X) per batch and divides by
    nimg_per_epoch; this replays that and compares."""
    grad = {"on": True}
    nimg = 6
    t = ep.EpochLossTracker(nimg)
    hook = ep.CellposeLossHook(t, grad_enabled=lambda: grad["on"])
    # Two epochs' worth: batches of 2 samples, losses chosen to be distinguishable.
    batches = [(1.5, 2, None), (2.5, 2, None), (3.5, 2, None),
               (0.5, 2, None), (1.0, 2, None), (1.5, 2, None)]
    _replay(hook, batches, grad)
    expect = [
        (1.5 * 2 + 2.5 * 2 + 3.5 * 2) / nimg,
        (0.5 * 2 + 1.0 * 2 + 1.5 * 2) / nimg,
    ]
    assert t.epoch_losses == [pytest.approx(e) for e in expect]
    assert ep.agreement(t.epoch_losses, expect) == ep.BASIS_MEASURED


def test_the_class_term_is_added_to_the_same_batch():
    """cellpose adds the class loss to the seg loss for the same batch when the net
    has more than three output channels. Dropping it would under-report every loss
    on a multi-class net."""
    grad = {"on": True}
    t = ep.EpochLossTracker(2)
    hook = ep.CellposeLossHook(t, grad_enabled=lambda: grad["on"])
    _replay(hook, [(1.0, 2, 0.25)], grad)
    assert t.epoch_losses == [pytest.approx((1.0 + 0.25) * 2 / 2)]
    assert hook.class_calls == 1


def test_a_validation_batch_is_not_counted():
    """THE SUBTLETY. The validation pass calls the same loss functions inside
    no_grad. Counting those would close epochs early and report a loss cellpose
    never computed."""
    grad = {"on": True}
    t = ep.EpochLossTracker(4)
    hook = ep.CellposeLossHook(t, grad_enabled=lambda: grad["on"])
    seg = hook.wrap_seg(lambda lbl, y: FakeTensor(y))
    seg(FakeBatch(2), 1.0)          # a training batch
    grad["on"] = False
    for _ in range(5):
        seg(FakeBatch(2), 99.0)     # a validation pass: must be ignored entirely
    grad["on"] = True
    seg(FakeBatch(2), 3.0)          # the training batch that closes the epoch
    hook.close()
    assert hook.skipped_no_grad == 5
    assert t.epochs_completed == 1
    # 99.0 appears nowhere: the curve is the two training batches only.
    assert t.epoch_losses == [pytest.approx((1.0 * 2 + 3.0 * 2) / 4)]


def test_a_batch_is_flushed_lazily_so_the_class_term_can_still_arrive():
    """A batch is only complete once the next seg call arrives or the run ends,
    because the class term comes after the seg term."""
    grad = {"on": True}
    t = ep.EpochLossTracker(2)
    hook = ep.CellposeLossHook(t, grad_enabled=lambda: grad["on"])
    seg = hook.wrap_seg(lambda lbl, y: FakeTensor(y))
    seg(FakeBatch(2), 1.0)
    assert t.epochs_completed == 0, "the batch was counted before it was complete"
    hook.close()
    assert t.epochs_completed == 1


def test_agreement_distinguishes_checked_from_merely_unchecked():
    live = [1.0, 2.0]
    assert ep.agreement(live, [1.0, 2.0]) == ep.BASIS_MEASURED
    assert ep.agreement(live, [1.0, 2.5]) == ep.BASIS_UNRECONCILED
    assert ep.agreement(live, [1.0]) == ep.BASIS_UNRECONCILED, "a length mismatch is not agreement"
    # Absent or empty is UNRECONCILED, not MEASURED: unchecked is not checked-and-equal.
    assert ep.agreement(live, []) == ep.BASIS_UNRECONCILED
    assert ep.agreement(live, None) == ep.BASIS_UNRECONCILED
    # Nothing observed at all is its own answer, not a claim about a curve.
    assert ep.agreement([], [1.0]) == ep.BASIS_UNAVAILABLE


def test_agreement_tolerates_float_association_but_not_a_real_difference():
    """The two sum the same terms in the same order, so only float association can
    differ. The tolerance must be loose enough for that and tight enough to catch a
    genuine divergence."""
    live = [1.0000000001, 2.0]
    assert ep.agreement(live, [1.0, 2.0]) == ep.BASIS_MEASURED
    assert ep.agreement([1.001, 2.0], [1.0, 2.0]) == ep.BASIS_UNRECONCILED


def test_installing_the_hook_without_cellpose_claims_nothing():
    """Fail-soft: in an environment with no cellpose or torch, nothing is patched
    and no progress is claimed. A wrong number is worse than no number."""
    assert ep.install_cellpose_hook(ep.EpochLossTracker(4)) is None
