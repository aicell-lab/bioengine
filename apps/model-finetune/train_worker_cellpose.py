"""Cellpose-SAM (cpsam) fine-tuning subprocess.

Launched by ``CellposeRuntime.train`` as ``python train_worker_cellpose.py
<session_id>`` so training runs in a child process — the OS reclaims all
training VRAM when it exits, keeping the runtime replica's serving state clean.
Reads the materialized TIFF-pair paths + hyperparameters from the session dir
(written by the entry via ``training.materialize_pairs``), runs
``cellpose.train.train_seg`` on the raw cpsam Transformer (``model.net``), and
writes the terminal status. Runs in the CellposeRuntime pip env (cellpose /
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


def _heartbeat(session_id: str, stop: threading.Event, interval: float = 60.0) -> None:
    """Refresh status.json's ``updated_at`` while train_seg runs, so a long epoch
    doesn't trip the stale-window check (get_status marks TRAINING → STOPPED after
    STATUS_STALE_SECONDS of no update)."""
    while not stop.wait(interval):
        training.write_status(session_id, status="TRAINING", message="training in progress")


def main(session_id: str) -> None:
    import torch
    from cellpose import models as cpmodels
    from cellpose.train import train_seg

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
            model = cpmodels.CellposeModel(gpu=gpu, pretrained_model=resume)
        else:
            model = cpmodels.CellposeModel(gpu=gpu, model_type="cpsam")
        net = model.net
        # bf16 precision is insufficient for weight updates at lr<=1e-4; train and
        # save in float32 (matches cellpose-finetuning's train_seg_with_callbacks).
        if net.dtype == torch.bfloat16:
            net.dtype = torch.float32
            net.to(torch.float32)

        train_seg(
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
        ok = training.checkpoint_path(session_id).exists()
        training.write_status(
            session_id,
            status="COMPLETED" if ok else "FAILED",
            message="checkpoint saved" if ok else "training finished but no checkpoint was produced",
            end_time=time.time(),
        )
    except Exception as e:
        stop.set()
        training.write_status(
            session_id, status="FAILED", message=str(e)[:800],
            traceback=traceback.format_exc()[-2500:], end_time=time.time(),
        )
        raise


if __name__ == "__main__":
    main(sys.argv[1])
