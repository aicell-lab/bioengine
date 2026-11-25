"""Live epoch progress is parsed out of cellpose's own log records.

The parsing is the whole feature, so these tests exist to make a FORMAT DRIFT
visible. ``train_seg`` offers no callback and no per-epoch counter and returns its
loss arrays only on completion, so if the regex stops matching there is no second
source for the validation loss and learning rate: those fields simply stay null,
which is indistinguishable from a run that was never validated. The fixtures below
are therefore built with the SAME f-string expression cellpose uses, quoted from
``cellpose/train.py`` at the pinned 4.2.1.1:

    f"{iepoch}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {...:.2f}s"

and one test asserts the regex matched at all, so a pattern that silently matches
nothing fails here rather than shipping a progress display that is always empty.

What this handler must NOT do is publish the printed ``train_loss`` as the epoch's
training loss — see ``test_the_printed_train_loss_is_a_window_mean_and_is_named_one``
and ``test_epoch_loss_hook.py``, which cover the per-epoch value's real source.
"""

import logging

import epoch_progress
import train_worker_cellpose as twc


def _cellpose_epoch_line(iepoch, train_loss, test_loss, lr=1e-4, secs=12.34):
    """Format exactly as cellpose does, so the fixture cannot drift from the source
    independently of the parser under test."""
    return (
        f"{iepoch}, train_loss={train_loss:.4f}, test_loss={test_loss:.4f}, "
        f"LR={lr:.6f}, time {secs:.2f}s"
    )


def _handler(has_val=True, want_loss=True):
    """A handler over a recorder whose status writes are captured, not written."""
    writes = []
    recorder = epoch_progress.EpochRecorder(lambda **fields: writes.append(fields))
    return twc._EpochProgressHandler(recorder, has_val, want_loss), recorder, writes


def test_the_real_cellpose_line_parses_and_the_pattern_is_not_inert():
    h, _rec, _w = _handler()
    entry = h._entry(_cellpose_epoch_line(10, 0.8125, 0.9031))
    # If this is None the pattern has gone inert and every other assertion about
    # "no entry" below would pass for the wrong reason.
    assert entry is not None, "the pinned cellpose format no longer parses"
    assert entry["epoch"] == 10
    assert abs(entry["train_loss_window_mean"] - 0.8125) < 1e-9
    assert abs(entry["val_loss"] - 0.9031) < 1e-9
    assert abs(entry["lr"] - 1e-4) < 1e-9
    assert h.unparsed == 0


def test_the_printed_train_loss_is_a_window_mean_and_is_named_one():
    """Measured in cellpose 4.2.1.1: ``lavg``/``nsum`` accumulate ACROSS epochs and
    are reset only inside the cadence branch that logs, so the printed
    ``train_loss`` at the epoch-10 report is the mean over epochs 6..10. Publishing
    it as ``train_loss`` would put points on a per-epoch curve that are not
    per-epoch values, so the row names it for what it is and leaves ``train_loss``
    to the hook that measures each epoch."""
    h, rec, _w = _handler()
    h.emit(
        logging.LogRecord(
            "cellpose.train",
            logging.INFO,
            "",
            0,
            _cellpose_epoch_line(10, 0.8125, 0.9031),
            (),
            None,
        )
    )
    row = rec.rows()[0]
    assert row["train_loss_window_mean"] == 0.8125
    assert (
        row["train_loss"] is None
    ), "the window mean was published as the epoch's training loss"


def test_a_zero_test_loss_is_reported_as_null_not_as_zero():
    """cellpose validates under the same hardcoded cadence guard that emits the line
    (`iepoch == 5 or iepoch % 10 == 0`) but only when a test set exists; off-cadence
    and no-test-set epochs keep the initialised 0.0. A real segmentation loss is never
    exactly 0.0, so a 0.0 must not reach a loss curve as a point on the axis."""
    h, _rec, _w = _handler(has_val=True)
    entry = h._entry(_cellpose_epoch_line(1, 1.5, 0.0))
    assert entry is not None
    assert entry["train_loss_window_mean"] == 1.5
    assert entry["val_loss"] is None


def test_without_a_validation_set_the_test_loss_is_never_believed():
    h, _rec, _w = _handler(has_val=False)
    entry = h._entry(_cellpose_epoch_line(0, 2.0, 7.5))
    assert entry is not None
    assert entry["train_loss_window_mean"] == 2.0
    # 7.5 is a number cellpose printed, but with no test set it cannot be a
    # measurement of anything, so it is withheld rather than surfaced.
    assert entry["val_loss"] is None


def test_epoch_number_is_reported_even_when_loss_metrics_were_not_requested():
    """``metrics`` chooses which metrics are COMPUTED; the epoch number is progress,
    not a metric, so a run asking for 'pixel' alone still shows where it has got to."""
    h, _rec, _w = _handler(want_loss=False)
    entry = h._entry(_cellpose_epoch_line(20, 0.5, 0.6))
    assert entry is not None
    assert entry["epoch"] == 20
    assert entry["train_loss_window_mean"] is None
    assert entry["val_loss"] is None


def test_cellposes_other_info_records_are_not_mistaken_for_epochs():
    """NEGATIVE control: the pattern must be narrow enough that cellpose's other INFO
    output cannot manufacture epochs. These are real lines from the same logger."""
    h, _rec, _w = _handler()
    for line in (
        ">>> n_epochs=5, n_train=3, n_test=1, bsize=256, batch_size=1, nimg_per_epoch=3",
        ">>> AdamW, learning_rate=0.00010, weight_decay=0.10000",
        ">>> computing diameters",
        ">>> loading images and labels",
        "saving network parameters to /x/models/model",
        ">>> saving model to /x/models/model",
    ):
        assert h._entry(line) is None, f"non-epoch line parsed as an epoch: {line}"
    assert h.unparsed == 0, "a non-epoch line was counted as a drifted epoch record"


def test_a_changed_upstream_format_is_counted_rather_than_silently_dropped():
    """A line that claims to be an epoch record but does not fit is the signature of
    an upstream format change. Counting it is what stops that failure being invisible:
    an empty ``epochs`` list looks identical whether the format moved or the run was
    simply short."""
    h, _rec, _w = _handler()
    drifted = "epoch 3 | train_loss=0.5000 | test_loss=0.6000 | lr=0.0001"
    assert h._entry(drifted) is None
    assert h.unparsed == 1


def test_emit_writes_the_accumulated_epochs_and_replaces_a_repeat():
    """The handler's job is to make progress visible DURING the run, so it must write
    on each record rather than accumulate and flush at the end."""
    h, _rec, writes = _handler()
    for line in (
        _cellpose_epoch_line(0, 2.0, 1.9),
        _cellpose_epoch_line(10, 1.0, 0.9),
        # cellpose re-reporting an epoch must replace it, not append a duplicate.
        _cellpose_epoch_line(10, 0.8, 0.7),
    ):
        h.emit(logging.LogRecord("cellpose.train", logging.INFO, "", 0, line, (), None))

    assert len(writes) == 3, "progress was not written on every record"
    kw = writes[-1]
    assert kw["last_reported_epoch"] == 10
    epochs = kw["epochs"]
    assert [e["epoch"] for e in epochs] == [0, 10], "a repeated epoch was duplicated"
    assert epochs[-1]["val_loss"] == 0.7, "the repeat did not replace the earlier value"
    assert kw["epoch_progress_unparsed"] == 0


def test_a_broken_write_cannot_fail_the_training_run():
    """A progress display must never be able to kill a training run that is working."""

    def boom(**kw):
        raise RuntimeError("status write failed")

    recorder = epoch_progress.EpochRecorder(boom)
    h = twc._EpochProgressHandler(recorder, has_val=True)
    h.emit(
        logging.LogRecord(
            "cellpose.train",
            logging.INFO,
            "",
            0,
            _cellpose_epoch_line(0, 1.0, 1.0),
            (),
            None,
        )
    )


def test_the_declared_cadence_matches_the_guard_it_describes():
    """The cadence string is a promise to clients about which epochs report. It is
    prose, so nothing else would catch it drifting from the guard it documents."""
    assert twc.EPOCH_REPORT_CADENCE == "0,5,then every 10"
    reported = [i for i in range(40) if i == 5 or i % 10 == 0]
    assert reported == [0, 5, 10, 20, 30]
    # The app's default run is 5 epochs -> iepoch 0..4 -> exactly ONE reported epoch,
    # which is why the per-epoch training loss cannot come from these records.
    assert [i for i in range(5) if i == 5 or i % 10 == 0] == [0]
