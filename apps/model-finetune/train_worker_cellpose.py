"""Cellpose (cpsam / cpdino) fine-tuning subprocess.

Launched by ``CellposeRuntime.train`` as ``python train_worker_cellpose.py
<session_id>`` so training runs in a child process — the OS reclaims all
training VRAM when it exits, keeping the runtime replica's serving state clean.
Reads the materialized TIFF-pair paths + hyperparameters from the session dir
(written by the entry via ``training.materialize_pairs``), runs
``cellpose.train.train_seg`` on the raw Cellpose Transformer (``model.net``) —
the base is picked from the session's ``model_type`` (cpsam or a cpdino variant) —
and writes the terminal status. Runs in the CellposeRuntime pip env (cellpose /
numpy-1.x installed).

``train_seg`` with ``save_path=<session_dir>, model_name='model'`` writes the
fine-tuned net to ``<session_dir>/models/model`` — the cellpose checkpoint layout
``training.checkpoint_path`` expects for a 'cellpose'-backend session.
"""

import logging
import re
import sys
import threading
import time
import traceback

import training


def _measure_diameters(label_files):
    """Min/max/median cell diameter over the training label masks, using cellpose's
    own per-ROI definition (equivalent-circle diameter 2*sqrt(area/pi), area = pixel
    count) — so the median equals what ``cellpose.utils.diameters`` would report and
    is the right value to rescale inference to. Returns None if no labelled objects
    are found. Rescaling stays OFF during training (Cellpose-SAM paper); this only
    records the size the model was trained at."""
    import numpy as np
    from cellpose import io as cpio

    diams = []
    for lf in label_files:
        lbl = np.squeeze(np.asarray(cpio.imread(lf)))
        if lbl.ndim != 2:
            continue
        _ids, counts = np.unique(lbl[lbl > 0], return_counts=True)
        if counts.size:
            diams.extend((2.0 * np.sqrt(counts / np.pi)).tolist())
    if not diams:
        return None
    a = np.asarray(diams, dtype=float)
    return {"min": float(a.min()), "max": float(a.max()), "median": float(np.median(a))}


def _loss_list(x):
    """Coerce train_seg's returned per-epoch losses to a plain list of floats
    (None preserved for skipped-validation epochs); [] if not array-like."""
    import numpy as np

    if x is None:
        return []
    try:
        return [None if v is None else float(v) for v in np.asarray(x).ravel().tolist()]
    except Exception:
        return []


def _safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def _pixel_scores(gts, preds):
    """Foreground/background scorecard pooled over the validation split: any
    labelled pixel is foreground, 0 is background. NOT an instance metric — a
    high iou here says nothing about whether touching cells were separated,
    which is what the instance metrics answer."""
    import numpy as np

    tp = fp = fn = tn = 0
    for gt, pred in zip(gts, preds):
        g, p = gt > 0, pred > 0
        tp += int(np.count_nonzero(g & p))
        fp += int(np.count_nonzero(~g & p))
        fn += int(np.count_nonzero(g & ~p))
        tn += int(np.count_nonzero(~g & ~p))
    precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    return {
        "pixel_accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "iou": _safe_div(tp, tp + fp + fn),
    }


def _instance_scores(gts, preds, thresholds):
    """Instance AP and F1 per IoU threshold, via cellpose's Hungarian matching.
    AP is averaged over images (nanmean — an image cellpose scores as NaN drops
    out); F1 is micro-averaged, pooling tp/fp/fn over images, so one sparse image
    cannot outweigh a crowded one."""
    import numpy as np
    from cellpose import metrics as cpmetrics

    thr = list(thresholds)
    ap, tp, fp, fn = (
        np.atleast_2d(np.asarray(x, dtype=float))
        for x in cpmetrics.average_precision(gts, preds, threshold=thr)
    )
    out = {"thresholds": thr, "mean_ap": {}, "f1": {}, "tp": {}, "fp": {}, "fn": {}}
    for i, t in enumerate(thr):
        key = str(t)
        s_tp, s_fp, s_fn = tp[:, i].sum(), fp[:, i].sum(), fn[:, i].sum()
        precision, recall = _safe_div(s_tp, s_tp + s_fp), _safe_div(s_tp, s_tp + s_fn)
        out["mean_ap"][key] = float(np.nanmean(ap[:, i]))
        out["f1"][key] = _safe_div(2 * precision * recall, precision + recall)
        out["tp"][key], out["fp"][key], out["fn"][key] = int(s_tp), int(s_fp), int(s_fn)
    return out


def _evaluate_validation(
    model, val_image_files, val_label_files, want_pixel, want_instance
):
    """Score the fine-tuned net on the validation split, returning
    ``(pixel_metrics, instance_metrics)`` — each None when not requested and
    ``{"error": ...}`` when it could not be computed. One inference pass feeds
    both families, so asking for both costs the same forward passes as either
    alone. Best-effort throughout: a run that produced a checkpoint must not be
    reported FAILED because a metric could not be scored."""
    import numpy as np
    from cellpose import io as cpio

    if not (want_pixel or want_instance):
        return None, None

    def both(value):
        return (value if want_pixel else None), (value if want_instance else None)

    if not val_image_files or not val_label_files:
        return both({"error": "needs validation data (val_images/val_labels)"})
    try:
        imgs = [np.asarray(cpio.imread(f)) for f in val_image_files]
        gts = [
            np.squeeze(np.asarray(cpio.imread(f))).astype(np.int32)
            for f in val_label_files
        ]
        preds = [
            np.asarray(m).astype(np.int32)
            for m in model.eval(imgs, normalize=True, rescale=False)[0]
        ]
    except Exception as exc:
        return both({"error": f"validation inference failed: {str(exc)[:240]}"})

    counts = {
        "n_val": len(gts),
        "n_true": sum(int(g.max()) for g in gts),
        "n_pred": sum(int(p.max()) for p in preds),
    }
    pixel = instance = None
    if want_pixel:
        try:
            pixel = dict(counts, **_pixel_scores(gts, preds))
        except Exception as exc:
            pixel = {"error": str(exc)[:300]}
    if want_instance:
        try:
            instance = dict(
                counts, **_instance_scores(gts, preds, training.INSTANCE_IOU_THRESHOLDS)
            )
        except Exception as exc:
            instance = {"error": str(exc)[:300]}
    return pixel, instance


def _heartbeat(session_id: str, stop: threading.Event, interval: float = 60.0) -> None:
    """Refresh status.json's ``updated_at`` while train_seg runs, so a long epoch
    doesn't trip the stale-window check (get_status marks TRAINING → STOPPED after
    STATUS_STALE_SECONDS of no update). Carries no ``status`` so the refresh lands
    even under a sticky terminal record, keeping a live orphan visibly live."""
    while not stop.wait(interval):
        training.write_status(session_id)


# cellpose emits its per-epoch line under a HARDCODED cadence guard
# (`if iepoch == 5 or iepoch % 10 == 0`, cellpose/train.py at 4.2.1.1), so epochs
# 0, 5, 10, 20, 30 … report and no others. Declared to the client because the gaps
# are expected: a consumer that assumes one entry per epoch would render a 5-epoch
# run (our default) as a single point and conclude the rest were lost.
EPOCH_REPORT_CADENCE = "0,5,then every 10"

# The exact line cellpose formats:
#   f"{iepoch}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {...:.2f}s"
# Anchored through LR so no other INFO record can match by accident.
_EPOCH_LINE = re.compile(
    r"^(?P<epoch>\d+),\s+train_loss=(?P<train>[-+0-9.eEnan]+),"
    r"\s+test_loss=(?P<test>[-+0-9.eEnan]+),\s+LR=(?P<lr>[-+0-9.eEnan]+)"
)
# Anything carrying this token is meant to be an epoch record; if it does not parse,
# the upstream format moved and progress reporting has gone silent. Counted so the
# drift is VISIBLE in the status rather than looking like a run that reported nothing.
_EPOCH_TOKEN = "train_loss="


class _EpochProgressHandler(logging.Handler):
    """Turn cellpose's per-epoch log record into a status.json entry as it happens.

    This is the ONLY live progress signal available. ``train_seg`` takes no callback,
    exposes no per-epoch counter, and returns its loss arrays only on completion, so
    without this a run is opaque from start to finish. The records are already routed
    to stdout by ``_surface_cellpose_logs``; this reads them in-process instead of
    tailing the log file, so no parsing of interleaved subprocess output is needed.

    Never raises: a progress display must not be able to fail a training run.
    """

    def __init__(
        self, session_id: str, has_val: bool, want_loss: bool = True
    ) -> None:
        super().__init__(level=logging.INFO)
        self._session_id = session_id
        self._has_val = has_val
        # ``metrics`` governs which quantitative metrics are COMPUTED, and the loss
        # curve is one of them. The epoch NUMBER is not a metric — it is progress —
        # so it is reported either way and only the loss values are withheld. A run
        # asking for 'pixel' alone therefore still shows where it has got to.
        self._want_loss = want_loss
        self._epochs: list = []
        self.unparsed = 0

    def _entry(self, msg: str):
        """Parse one record. Returns an entry dict, or None if this is not an epoch
        line. Increments ``unparsed`` when a line claims to be one but does not fit."""
        m = _EPOCH_LINE.match(msg)
        if m is None:
            if _EPOCH_TOKEN in msg:
                self.unparsed += 1
            return None
        test = float(m.group("test"))
        if not self._want_loss:
            return {"epoch": int(m.group("epoch")), "train_loss": None,
                    "val_loss": None, "lr": float(m.group("lr"))}
        return {
            "epoch": int(m.group("epoch")),
            "train_loss": float(m.group("train")),
            # cellpose runs validation INSIDE the same cadence guard that emits this
            # line, but only when a test set exists. With none, the logged value is
            # the initialised 0.0 — not a measurement. A real segmentation loss is
            # never exactly 0.0, so both cases collapse to null rather than to a
            # point that would drag a loss curve to the axis.
            "val_loss": test if (self._has_val and test != 0.0) else None,
            "lr": float(m.group("lr")),
        }

    def emit(self, record: "logging.LogRecord") -> None:
        try:
            entry = self._entry(record.getMessage().strip())
            if entry is None:
                return
            # Re-reporting an epoch replaces it rather than appending a duplicate.
            self._epochs = [e for e in self._epochs if e["epoch"] != entry["epoch"]]
            self._epochs.append(entry)
            self._epochs.sort(key=lambda e: e["epoch"])
            training.write_status(
                self._session_id,
                epochs=list(self._epochs),
                last_reported_epoch=entry["epoch"],
                epoch_progress_unparsed=self.unparsed,
            )
        except Exception:
            pass


def _attach_epoch_progress(
    session_id: str, has_val: bool, want_loss: bool = True
) -> "_EpochProgressHandler":
    """Attach the progress handler to the ``cellpose`` logger.

    cellpose's own logger is ``cellpose.train`` (``logging.getLogger(__name__)``), a
    child of ``cellpose``, so a handler on the parent receives it by propagation —
    the same mechanism ``_surface_cellpose_logs`` relies on."""
    handler = _EpochProgressHandler(session_id, has_val, want_loss)
    logging.getLogger("cellpose").addHandler(handler)
    return handler


def _surface_cellpose_logs() -> None:
    """Route cellpose's per-epoch train/test-loss trace to stdout so the subprocess
    redirect captures it (into train.log / the session record). cellpose attaches a
    NullHandler and its loggers sit at NOTSET, so the effective level resolves to root
    WARNING and its INFO per-epoch records are never emitted; because a handler IS
    present, logging.lastResort never fires either, so even WARNING/ERROR are swallowed.
    Setting the level + adding an explicit stdout handler is immune to root-handler
    state (unlike basicConfig) and has no filesystem side effect (unlike
    cellpose.io.logger_setup, which mkdirs ~/.cellpose and clears handlers)."""
    lg = logging.getLogger("cellpose")
    lg.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in lg.handlers):
        lg.addHandler(logging.StreamHandler(sys.stdout))


def main(session_id: str) -> None:
    import torch
    from cellpose import models as cpmodels
    from cellpose.train import train_seg

    _surface_cellpose_logs()

    p = training.read_training_params(session_id)
    sdir = training.session_dir(session_id)
    gpu = torch.cuda.is_available()
    # Re-check here, before any GPU time is spent: the entry already validated,
    # but failing a bad selection after a full training run would be the worst
    # moment to discover it.
    metrics_requested = training.validate_metrics(p.get("metrics"))

    # Live progress. Attached before train_seg runs so the first reported epoch is
    # captured, and declared in the status so a client knows the cadence to expect
    # instead of reading the gaps as lost records.
    progress = _attach_epoch_progress(
        session_id,
        has_val=bool(p.get("val_images")),
        want_loss="loss" in metrics_requested,
    )
    training.write_status(
        session_id,
        status="TRAINING",
        message="training started",
        epochs=[],
        last_reported_epoch=None,
        epoch_report_cadence=EPOCH_REPORT_CADENCE,
        n_epochs=p["n_epochs"],
    )
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat, args=(session_id, stop), daemon=True
    )
    heartbeat.start()

    try:
        resume = p.get("checkpoint_path")
        if resume:
            # A fine-tuned checkpoint carries its architecture — cellpose
            # auto-detects the backbone (cpsam or cpdino) from it.
            model = cpmodels.CellposeModel(gpu=gpu, pretrained_model=resume)
        else:
            base = training.cellpose_base_model(p.get("model_type", "cpsam"))
            model = cpmodels.CellposeModel(gpu=gpu, pretrained_model=base)
        net = model.net
        # bf16 precision is insufficient for weight updates at lr<=1e-4; train and
        # save in float32 (matches cellpose-finetuning's train_seg_with_callbacks).
        if net.dtype == torch.bfloat16:
            net.dtype = torch.float32
            net.to(torch.float32)

        result = train_seg(
            net,
            train_files=p["train_images"],
            train_labels_files=p["train_labels"],
            test_files=p["val_images"],
            test_labels_files=p["val_labels"],
            save_path=str(sdir),
            model_name="model",
            n_epochs=p["n_epochs"],
            learning_rate=p["learning_rate"],
            weight_decay=p.get("weight_decay", 0.1),
            batch_size=p.get("batch_size", 1),
            min_train_masks=p.get("min_train_masks", 1),
            # None keeps cellpose's own per-backbone default: 256 for cpsam,
            # 384 for cpdino. cpsam rejects any other value outright.
            bsize=p.get("train_tile_size"),
            normalize=True,
            rescale=False,
        )
        stop.set()
        # Stock train_seg returns (model_path, train_losses, test_losses); capture
        # defensively so a signature change can't break the run.
        train_losses, test_losses = [], []
        if (
            "loss" in metrics_requested
            and isinstance(result, (tuple, list))
            and len(result) >= 3
        ):
            train_losses, test_losses = _loss_list(result[1]), _loss_list(result[2])
            # Stock train_seg validates under a HARDCODED cadence guard — not a
            # configurable interval: `if iepoch == 5 or iepoch % 10 == 0` in
            # cellpose/train.py at 4.2.1.1, with no parameter to change it. Off-cadence
            # epochs keep the initialised 0.0, so a real segmentation test loss is
            # never exactly 0.0 and those surface as null — a consumer must not read a
            # non-validated epoch as a test loss of 0.0 (verified on deNBI: a 4-epoch
            # run gave [11.05, 0, 0, 0] = validated at epoch 0 only, which is that
            # guard matching 0 % 10 and nothing else).
            test_losses = [
                None if (v is not None and v == 0.0) else v for v in test_losses
            ]
        ok = training.checkpoint_path(session_id).exists()
        cell_diameters = None
        pixel_metrics = instance_metrics = None
        if ok:
            try:
                cell_diameters = _measure_diameters(p.get("train_labels") or [])
            except Exception:
                cell_diameters = None
            pixel_metrics, instance_metrics = _evaluate_validation(
                model,
                p.get("val_images"),
                p.get("val_labels"),
                want_pixel="pixel" in metrics_requested,
                want_instance="instance" in metrics_requested,
            )
        # train_seg has no early stopping and only checkpoints after the full
        # range(n_epochs) loop, so a COMPLETED cellpose run ran exactly the
        # requested epochs; a truncated run never checkpoints → never COMPLETED.
        # Hence the count is a floor guaranteed only by COMPLETED, not a measured
        # value (basis="floor_if_completed") — cellpose exposes no per-epoch counter.
        training.write_status(
            session_id,
            status="COMPLETED" if ok else "FAILED",
            message=(
                "checkpoint saved"
                if ok
                else "training finished but no checkpoint was produced"
            ),
            end_time=time.time(),
            terminated_by="child",
            # Non-zero means cellpose logged lines that CLAIMED to be epoch records
            # and did not fit the format this worker parses — i.e. the upstream format
            # moved and live progress has gone silent. Recorded on every run so that
            # drift is visible here rather than looking like a run that simply never
            # reported, which is the same empty `epochs` either way.
            epoch_progress_unparsed=progress.unparsed,
            n_epochs_completed=p["n_epochs"] if ok else None,
            n_epochs_completed_basis="floor_if_completed" if ok else None,
            cell_diameters=cell_diameters,
            metrics_requested=metrics_requested,
            train_losses=train_losses,
            test_losses=test_losses,
            pixel_metrics=pixel_metrics,
            instance_metrics=instance_metrics,
        )
    except Exception as e:
        stop.set()
        training.write_status(
            session_id,
            status="FAILED",
            message=str(e)[:800],
            traceback=traceback.format_exc()[-2500:],
            end_time=time.time(),
            terminated_by="child",
        )
        raise


if __name__ == "__main__":
    main(sys.argv[1])
