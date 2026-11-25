"""Per-epoch progress for a cellpose fine-tuning run, measured while it runs.

Why this exists at all: ``cellpose.train.train_seg`` reports nothing per epoch
that a caller can observe. Measured against the pinned ``cellpose==4.2.1.1``
(``requirements-runtime-cellpose.txt``), reading that release's own
``cellpose/train.py``:

* its signature carries **no callback and no validation interval** — it ends
  ``min_train_masks=5, model_name=None, class_weights=None``;
* its one per-epoch log line is inside ``if iepoch == 5 or iepoch % 10 == 0``,
  so a 5-epoch run logs epoch 0 and nothing else. **Parsing the log cannot give
  per-epoch progress**, which is why this module hooks the arithmetic instead;
* ``save_every`` defaults to 100 and a checkpoint is ~1.2 GB for cpsam, so the
  per-epoch-file trick ``train_worker.py`` uses for micro-sam's ``latest.pt`` is
  not affordable here.

What is observable is the loss itself. Per batch ``train_seg`` does::

    loss = _loss_fn_seg(lbl, y, device)
    if y.shape[1] > 3:
        loss += _loss_fn_class(lbl, y, class_weights=class_weights)
    train_loss = loss.item() * len(X)
    train_losses[iepoch] += train_loss

and after the batch loop ``train_losses[iepoch] /= nimg_per_epoch``, with the
batches running over ``range(0, nimg_per_epoch, batch_size)``. So wrapping the
two loss functions yields the same numbers cellpose will later return, and the
sample count reaches exactly ``nimg_per_epoch`` at every epoch boundary — which
is where the epoch counter ticks.

Three properties this file is built for:

**Fail-soft.** If a later cellpose renames either loss function, the hook does
not install and nothing claims to be measured: the run proceeds exactly as
before and the UI makes no per-epoch claim (``trainingProgress.ts`` only renders
a ``measured`` basis). A wrong number is worse than no number.

**Reconciled, not trusted.** ``train_seg`` returns its own per-epoch curve at the
end. :func:`agreement` compares what was reported live against it, so "the live
curve is the real curve" is a measurement rather than an assumption, and a drift
in cellpose's arithmetic is reported instead of being shipped as truth.

**Testable without cellpose, torch or a GPU.** The accumulator is plain
arithmetic and the hook takes its two loss functions and its
``grad_enabled`` predicate as arguments; only :func:`install_cellpose_hook`
touches the real modules.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

# Relative tolerance for "the live curve is train_seg's curve". The two sum the
# same per-batch terms in the same order, so only float association differs;
# 1e-6 is far above that and far below any real divergence.
AGREEMENT_RTOL = 1e-6

BASIS_MEASURED = "measured"
BASIS_UNRECONCILED = "measured_unreconciled"
BASIS_UNAVAILABLE = "unavailable"


class EpochLossTracker:
    """Reproduces ``train_seg``'s per-epoch loss from its per-batch terms.

    *on_epoch* is called as ``on_epoch(n_epochs_completed, epoch_losses)`` once
    per completed epoch, with ``epoch_losses`` the whole curve so far (a fresh
    list, so a consumer may keep it).
    """

    def __init__(
        self,
        nimg_per_epoch: int,
        on_epoch: Optional[Callable[[int, List[float]], None]] = None,
    ) -> None:
        if int(nimg_per_epoch) <= 0:
            raise ValueError(
                f"nimg_per_epoch must be positive, got {nimg_per_epoch!r}; "
                "without it there is no epoch boundary to count."
            )
        self.nimg_per_epoch = int(nimg_per_epoch)
        self._on_epoch = on_epoch
        self.epoch_losses: List[float] = []
        self._weighted = 0.0
        self._samples = 0

    @property
    def epochs_completed(self) -> int:
        return len(self.epoch_losses)

    def observe_batch(self, loss_value: float, n_samples: int) -> None:
        """One training batch, as ``loss.item()`` and ``len(X)``."""
        n_samples = int(n_samples)
        if n_samples <= 0:
            # A batch of nothing cannot move an epoch boundary, and counting it
            # would let a malformed observation close an epoch early.
            return
        self._weighted += float(loss_value) * n_samples
        self._samples += n_samples
        if self._samples < self.nimg_per_epoch:
            return
        # cellpose divides by nimg_per_epoch, NOT by the samples it actually
        # summed. The two differ whenever batch_size does not divide
        # nimg_per_epoch, and matching cellpose is the whole point.
        self.epoch_losses.append(self._weighted / self.nimg_per_epoch)
        self._weighted = 0.0
        self._samples = 0
        if self._on_epoch is not None:
            self._on_epoch(len(self.epoch_losses), list(self.epoch_losses))


class CellposeLossHook:
    """Wraps cellpose's two loss functions and feeds a tracker.

    Per batch cellpose calls ``_loss_fn_seg`` and then, only when the net has
    more than three output channels, ``_loss_fn_class``; the second term is added
    to the first. A batch is therefore only complete once the *next* ``seg`` call
    arrives (or the run ends), which is why the pending batch is flushed lazily.

    The validation pass calls the same functions, inside ``torch.no_grad()`` —
    so *grad_enabled* is what separates a training batch from a validation one.
    Counting a validation batch would close epochs early and report a loss
    cellpose never computed.
    """

    def __init__(
        self,
        tracker: EpochLossTracker,
        grad_enabled: Callable[[], bool],
        scalar: Callable[[Any], float] = lambda t: float(t.item()),
    ) -> None:
        self._tracker = tracker
        self._grad_enabled = grad_enabled
        self._scalar = scalar
        self._pending: Optional[Dict[str, float]] = None
        self.seg_calls = 0
        self.class_calls = 0
        self.skipped_no_grad = 0

    def wrap_seg(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def _seg(*args: Any, **kwargs: Any) -> Any:
            out = fn(*args, **kwargs)
            if not self._grad_enabled():
                self.skipped_no_grad += 1
                return out
            self.flush()
            lbl = args[0] if args else kwargs.get("lbl")
            self._pending = {
                "value": self._scalar(out),
                "n": float(_batch_size_of(lbl)),
            }
            self.seg_calls += 1
            return out

        return _seg

    def wrap_class(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def _class(*args: Any, **kwargs: Any) -> Any:
            out = fn(*args, **kwargs)
            if self._grad_enabled() and self._pending is not None:
                self._pending["value"] += self._scalar(out)
                self.class_calls += 1
            return out

        return _class

    def flush(self) -> None:
        """Hand the pending batch to the tracker, if there is one."""
        if self._pending is None:
            return
        pending, self._pending = self._pending, None
        self._tracker.observe_batch(pending["value"], int(pending["n"]))

    def close(self) -> None:
        self.flush()


def _batch_size_of(lbl: Any) -> int:
    """``len(X)`` for the batch, read off the label batch's leading dimension.

    ``X`` and ``lbl`` come out of one ``random_rotate_and_resize`` call, so they
    share it. Returns 0 for anything unreadable, which
    :meth:`EpochLossTracker.observe_batch` then ignores rather than guessing 1.
    """
    shape = getattr(lbl, "shape", None)
    try:
        return int(shape[0]) if shape is not None else len(lbl)
    except Exception:
        return 0


def install_cellpose_hook(
    tracker: EpochLossTracker,
) -> Optional[CellposeLossHook]:
    """Install the hook into the live ``cellpose.train`` module.

    Returns the hook, or ``None`` when the anchors are not there — a later
    cellpose that renames either loss function, or an environment with no torch.
    Nothing is patched in that case and no progress is claimed.
    """
    try:
        import torch
        from cellpose import train as cptrain
    except Exception:
        return None

    seg = getattr(cptrain, "_loss_fn_seg", None)
    cls = getattr(cptrain, "_loss_fn_class", None)
    if not callable(seg):
        return None

    hook = CellposeLossHook(tracker, grad_enabled=torch.is_grad_enabled)
    cptrain._loss_fn_seg = hook.wrap_seg(seg)
    if callable(cls):
        cptrain._loss_fn_class = hook.wrap_class(cls)
    return hook


class EpochRecorder:
    """The one place a run's per-epoch rows live, and the only writer of them.

    Two signals describe the same epochs and neither covers it alone:

    * the loss hook above sees **every** epoch's own training loss;
    * cellpose's log record, on its ``iepoch == 5 or iepoch % 10 == 0`` cadence,
      carries that epoch's **validation** loss and learning rate, which the hook
      cannot see (the validation pass runs under ``torch.no_grad()`` and is
      skipped on purpose).

    So they are merged into one row per epoch rather than published as two
    competing lists. ``epoch`` is cellpose's own 0-based ``iepoch``, so a row
    sourced from the hook and a row sourced from the log address the same epoch
    by the same number.

    *write* is called with keyword fields to merge into the session status.
    """

    def __init__(self, write: Callable[..., Any]) -> None:
        self._write = write
        self._rows: Dict[int, Dict[str, Any]] = {}
        self.unparsed = 0
        self.measured_epochs = 0

    def _row(self, iepoch: int) -> Dict[str, Any]:
        row = self._rows.get(iepoch)
        if row is None:
            row = {
                "epoch": int(iepoch),
                "train_loss": None,
                "val_loss": None,
                "lr": None,
            }
            self._rows[iepoch] = row
        return row

    def rows(self) -> List[Dict[str, Any]]:
        return [self._rows[k] for k in sorted(self._rows)]

    @property
    def last_epoch(self) -> Optional[int]:
        return max(self._rows) if self._rows else None

    def record_epoch_loss(
        self, epochs_completed: int, loss: float, publish_loss: bool = True
    ) -> None:
        """One completed epoch, from the hook. *epochs_completed* is 1-based."""
        iepoch = int(epochs_completed) - 1
        self.measured_epochs = max(self.measured_epochs, int(epochs_completed))
        if publish_loss:
            self._row(iepoch)["train_loss"] = float(loss)
        else:
            # The epoch NUMBER is progress, not a metric, so it is recorded even
            # when the caller did not ask for the loss curve.
            self._row(iepoch)
        self.publish()

    def annotate(self, iepoch: int, **fields: Any) -> None:
        """Merge fields into an epoch's row, from the log record."""
        row = self._row(int(iepoch))
        for key, value in fields.items():
            if value is not None or key not in row:
                row[key] = value
        self.publish()

    def status_fields(self) -> Dict[str, Any]:
        fields: Dict[str, Any] = {
            "epochs": [dict(r) for r in self.rows()],
            "last_reported_epoch": self.last_epoch,
            "epoch_progress_unparsed": self.unparsed,
        }
        if self.measured_epochs:
            # The UI only renders a per-epoch claim for a `measured` basis
            # (trainingProgress.ts), and this count IS an observation: it is the
            # number of epochs whose batches were seen, not the number requested.
            fields["n_epochs_completed"] = self.measured_epochs
            fields["n_epochs_completed_basis"] = BASIS_MEASURED
        return fields

    def publish(self, **extra: Any) -> None:
        self._write(**self.status_fields(), **extra)


def agreement(live: List[float], final: Any) -> str:
    """Which basis the live per-epoch curve has earned.

    ``unavailable`` when nothing was observed, ``measured`` when the live curve
    is train_seg's own curve to within :data:`AGREEMENT_RTOL`, and
    ``measured_unreconciled`` when it is not — the caller then keeps the
    authoritative curve and says so, rather than publishing a curve that cannot
    be reproduced.

    A final curve that is absent or empty leaves the live curve
    ``measured_unreconciled``: unchecked is not the same as checked and equal.
    """
    if not live:
        return BASIS_UNAVAILABLE
    try:
        final_list = [float(v) for v in final]
    except Exception:
        return BASIS_UNRECONCILED
    if len(final_list) != len(live):
        return BASIS_UNRECONCILED
    for a, b in zip(live, final_list):
        if abs(a - b) > AGREEMENT_RTOL * max(1.0, abs(b)):
            return BASIS_UNRECONCILED
    return BASIS_MEASURED
