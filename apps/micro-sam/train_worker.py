"""μSAM fine-tuning subprocess.

Launched by ``RuntimeApp.train`` as ``python train_worker.py <session_id>`` so
training runs in a child process — the OS reclaims all training VRAM when it
exits, keeping the runtime replica's serving state clean. Reads the materialized
data paths + hyperparameters from the session dir (written by the entry), runs
``micro_sam.training.train_sam`` with the AIS decoder, and writes the terminal
status. Runs in the runtime's pip env (micro-sam / torch-em installed).
"""

import sys
import time
import traceback

import training


def main(session_id: str) -> None:
    import torch
    from micro_sam.training import default_sam_loader, train_sam
    from torch_em.data import MinInstanceSampler

    p = training.read_training_params(session_id)
    sdir = training.session_dir(session_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch = tuple(p["patch_shape"])

    common = dict(
        raw_key=None, label_key=None, patch_shape=patch,
        with_segmentation_decoder=True, batch_size=p["batch_size"],
        sampler=MinInstanceSampler(min_num_instances=1),
        num_workers=p.get("num_workers", 0),
    )
    try:
        train_loader = default_sam_loader(
            raw_paths=p["train_images"], label_paths=p["train_labels"],
            is_train=True, shuffle=True, n_samples=p.get("n_samples"), **common,
        )
        val_loader = default_sam_loader(
            raw_paths=p["val_images"], label_paths=p["val_labels"],
            is_train=False, shuffle=False, **common,
        )
        train_sam(
            name=session_id, model_type=p["model_type"],
            train_loader=train_loader, val_loader=val_loader,
            n_epochs=p["n_epochs"], n_objects_per_batch=p.get("n_objects_per_batch", 8),
            with_segmentation_decoder=True, save_root=str(sdir),
            device=device, lr=p["learning_rate"],
        )
        ok = training.checkpoint_path(session_id).exists()
        training.write_status(
            session_id,
            status="COMPLETED" if ok else "FAILED",
            message="checkpoint saved" if ok else "training finished but no checkpoint was produced",
            end_time=time.time(),
        )
    except Exception as e:
        training.write_status(
            session_id, status="FAILED", message=str(e)[:800],
            traceback=traceback.format_exc()[-2500:], end_time=time.time(),
        )
        raise


if __name__ == "__main__":
    main(sys.argv[1])
