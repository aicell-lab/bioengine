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


def _instance_ap(model, val_image_files, val_label_files, thresholds=(0.5, 0.75, 0.9)):
    """Mean instance AP (Hungarian-matched, cellpose.metrics.average_precision) over
    the validation split, per IoU threshold. The expensive opt-in metric: runs a full
    inference pass on the held-out set. Best-effort — returns {error: ...} on any
    failure so a metrics request never fails the training run itself."""
    import numpy as np
    from cellpose import io as cpio
    from cellpose import metrics as cpmetrics

    if not val_image_files or not val_label_files:
        return None
    try:
        imgs = [np.asarray(cpio.imread(f)) for f in val_image_files]
        gts = [np.squeeze(np.asarray(cpio.imread(f))) for f in val_label_files]
        preds = model.eval(imgs, normalize=True, rescale=False)[0]
        ap, _tp, _fp, _fn = cpmetrics.average_precision(
            gts, preds, threshold=list(thresholds))
        ap = np.asarray(ap, dtype=float)
        if ap.ndim == 1:
            ap = ap[None, :]
        return {
            "n_val": len(gts),
            "thresholds": list(thresholds),
            "mean_ap": {f"{t}": float(ap[:, i].mean()) for i, t in enumerate(thresholds)},
        }
    except Exception as exc:
        return {"error": str(exc)[:300]}


def _heartbeat(session_id: str, stop: threading.Event, interval: float = 60.0) -> None:
    """Refresh status.json's ``updated_at`` while train_seg runs, so a long epoch
    doesn't trip the stale-window check (get_status marks TRAINING → STOPPED after
    STATUS_STALE_SECONDS of no update). Carries no ``status`` so the refresh lands
    even under a sticky terminal record, keeping a live orphan visibly live."""
    while not stop.wait(interval):
        training.write_status(session_id)


def _surface_cellpose_logs() -> None:
    """Route cellpose's per-epoch train/test-loss trace to stdout so the subprocess
    redirect captures it (into train.log / the session record). cellpose attaches a
    NullHandler and its loggers sit at NOTSET, so the effective level resolves to root
    WARNING and its INFO per-epoch records are never emitted; because a handler IS
    present, logging.lastResort never fires either, so even WARNING/ERROR are swallowed.
    Setting the level + adding an explicit stdout handler is immune to root-handler
    state (unlike basicConfig) and has no filesystem side effect (unlike
    cellpose.io.logger_setup, which mkdirs ~/.cellpose and clears handlers)."""
    import logging

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

    training.write_status(session_id, status="TRAINING", message="training started")
    stop = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat, args=(session_id, stop), daemon=True)
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
            train_files=p["train_images"], train_labels_files=p["train_labels"],
            test_files=p["val_images"], test_labels_files=p["val_labels"],
            save_path=str(sdir), model_name="model",
            n_epochs=p["n_epochs"], learning_rate=p["learning_rate"],
            weight_decay=p.get("weight_decay", 0.1), batch_size=p.get("batch_size", 1),
            min_train_masks=p.get("min_train_masks", 1),
            normalize=True, rescale=False,
        )
        stop.set()
        # Stock train_seg returns (model_path, train_losses, test_losses); capture
        # defensively so a signature change can't break the run.
        train_losses, test_losses = [], []
        if isinstance(result, (tuple, list)) and len(result) >= 3:
            train_losses, test_losses = _loss_list(result[1]), _loss_list(result[2])
        ok = training.checkpoint_path(session_id).exists()
        cell_diameters = None
        instance_metrics = None
        metrics_requested = [m for m in (p.get("metrics") or ["loss"])]
        if ok:
            try:
                cell_diameters = _measure_diameters(p.get("train_labels") or [])
            except Exception:
                cell_diameters = None
            if "instance_ap" in metrics_requested:
                instance_metrics = _instance_ap(
                    model, p.get("val_images"), p.get("val_labels"))
        # train_seg has no early stopping and only checkpoints after the full
        # range(n_epochs) loop, so a COMPLETED cellpose run ran exactly the
        # requested epochs; a truncated run never checkpoints → never COMPLETED.
        # Hence the count is a floor guaranteed only by COMPLETED, not a measured
        # value (basis="floor_if_completed") — cellpose exposes no per-epoch counter.
        training.write_status(
            session_id,
            status="COMPLETED" if ok else "FAILED",
            message="checkpoint saved" if ok else "training finished but no checkpoint was produced",
            end_time=time.time(), terminated_by="child",
            n_epochs_completed=p["n_epochs"] if ok else None,
            n_epochs_completed_basis="floor_if_completed" if ok else None,
            cell_diameters=cell_diameters,
            metrics_requested=metrics_requested,
            train_losses=train_losses,
            test_losses=test_losses,
            instance_metrics=instance_metrics,
        )
    except Exception as e:
        stop.set()
        training.write_status(
            session_id, status="FAILED", message=str(e)[:800],
            traceback=traceback.format_exc()[-2500:], end_time=time.time(),
            terminated_by="child",
        )
        raise


if __name__ == "__main__":
    main(sys.argv[1])
