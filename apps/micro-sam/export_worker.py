"""BioImage.IO export subprocess.

Launched by ``RuntimeApp.export_bioimageio`` as
``python export_worker.py <session_id> <model_name>`` with
``CUDA_VISIBLE_DEVICES=""`` so it runs on CPU (export is trace-only — no GPU
contention with serving/training). Builds a standard BioImage.IO package from a
trained session's ``best.pt`` via ``micro_sam.bioimageio.export_sam_model``,
unzips it into ``<session>/export/package/`` (rdf.yaml + weights at root), and
writes ``export_result.json`` for the entry to upload to Hypha.
"""

import json
import shutil
import sys
import zipfile
from pathlib import Path

import training


def main(session_id: str, model_name: str) -> None:
    import tifffile
    from micro_sam.bioimageio import export_sam_model

    p = training.read_training_params(session_id)
    ckpt = training.checkpoint_path(session_id)
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint for session '{session_id}'.")

    image = tifffile.imread(p["train_images"][0])
    label = tifffile.imread(p["train_labels"][0])

    export_dir = training.session_dir(session_id) / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    zip_path = export_dir / f"{model_name}.zip"

    # export_sam_model builds the base SAM via segment_anything's
    # sam_model_registry, which only knows base architectures — strip the
    # micro-sam suffix (vit_b_lm -> vit_b); the fine-tuned weights come from the
    # checkpoint.
    base_arch = "_".join(p["model_type"].split("_")[:2])
    export_sam_model(
        image=image, label_image=label, model_type=base_arch,
        name=model_name, output_path=str(zip_path), checkpoint_path=str(ckpt),
    )

    pkg_dir = export_dir / "package"
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(pkg_dir)

    (export_dir / "export_result.json").write_text(
        json.dumps({"package_dir": str(pkg_dir), "zip": str(zip_path)})
    )
    print("EXPORT_OK", pkg_dir, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
