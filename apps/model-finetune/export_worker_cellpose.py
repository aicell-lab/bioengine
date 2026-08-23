"""Cellpose-SAM (cpsam) BioImage.IO export subprocess.

Launched by ``CellposeRuntime.export_bioimageio`` as
``python export_worker_cellpose.py <session_id> <export_dir>`` with
``CUDA_VISIBLE_DEVICES=""`` so it runs on CPU (no GPU contention with
serving/training). Reads ``<export_dir>/request.json`` and the session's
fine-tuned cellpose checkpoint (``<session>/models/model``), then hand-rolls a
BioImage.IO ``pytorch_state_dict`` package around ``cellpose_model.py``'s
``CellposeSAMWrapper`` (bundled as ``model.py``).

Two deliberate departures from ``apps/cellpose-finetuning``'s exporter:
  * ``output_sample.npy`` is a **real forward pass** of the wrapper (CPU,
    float32), not the ground-truth annotation — so the package self-validates.
  * we run ``bioimageio.core.test_model`` on the built package and **fail the
    export** unless it passes. The RDF kwargs therefore bake ``gpu=False,
    use_bfloat16=False`` for a deterministic CPU reproduction (integer masks →
    exact match). GPU serving of the checkpoint bypasses this RDF entirely
    (``CellposeRuntime`` loads the weights via ``CellposeModel`` directly).

This exporter **publishes nothing** — the entry streams the package into a draft
artifact the caller owns.
"""

import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import yaml

import training

HERE = Path(__file__).parent


def _load_array(path: str) -> np.ndarray:
    import imageio.v3 as iio

    ext = Path(str(path).split("?")[0]).suffix.lower()
    arr = np.load(path) if ext == ".npy" else iio.imread(path)
    return np.asarray(arr)


def _to_chw3(image: np.ndarray) -> np.ndarray:
    """Coerce an image to (3, H, W) float32."""
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=0)
    elif img.ndim == 3:
        if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3):
            pass  # already CHW
        else:
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        if img.shape[0] == 1:
            img = np.concatenate([img, img, img], axis=0)
        elif img.shape[0] == 2:
            img = np.concatenate([img, img[:1]], axis=0)
        img = img[:3]
    else:
        raise ValueError(f"Unsupported image ndim {img.ndim} (shape {img.shape}).")
    return img.astype(np.float32)


def _center_crop(chw: np.ndarray, size: int = 256) -> np.ndarray:
    """Center-crop to <=size on H and W to bound CPU test cost while keeping cell
    scale (so the reproduced masks are representative)."""
    _, h, w = chw.shape
    ch, cw = min(size, h), min(size, w)
    top, left = (h - ch) // 2, (w - cw) // 2
    return chw[:, top:top + ch, left:left + cw]


def _cite():
    return [
        {"text": "Stringer, C., Wang, T., Michaelos, M. et al. Cellpose: a generalist "
                 "algorithm for cellular segmentation. Nat Methods 18, 100–106 (2021).",
         "doi": "10.1038/s41592-020-01018-x"},
        {"text": "Pachitariu, M., Stringer, C. Cellpose 2.0: how to train your own "
                 "model. Nat Methods 19, 1634–1641 (2022).",
         "doi": "10.1038/s41592-022-01663-4"},
    ]


def _authors(raw):
    fields = ("name", "affiliation", "email", "github_user", "orcid")
    out = []
    for a in raw or []:
        author = {k: a[k] for k in fields if a.get(k)}
        if author.get("name"):
            out.append(author)
    return out or [{"name": "BioEngine user"}]


def _build_rdf(name, description, authors, license_id, diam_mean, provenance,
               input_shape, output_shape):
    import torch

    desc = (f"Fine-tuned Cellpose-SAM model. {description}".strip()
            if description else "Fine-tuned Cellpose-SAM model.")
    rdf = {
        "name": name,
        "description": desc,
        "authors": _authors(authors),
        "cite": _cite(),
        "license": license_id,
        "tags": ["Cellpose", "Cellpose-SAM", "Cell Segmentation", "Segmentation", "Fine-tuned"],
        "version": "0.1.0",
        "format_version": "0.5.6",
        "type": "model",
        "id_emoji": "🔬",
        "documentation": "documentation.md",
        "inputs": [{
            "id": "input",
            "axes": [
                {"type": "batch"},
                {"type": "channel", "channel_names": ["r", "g", "b"]},
                {"size": int(input_shape[1]), "id": "y", "type": "space"},
                {"size": int(input_shape[2]), "id": "x", "type": "space"},
            ],
            "test_tensor": {"source": "input_sample.npy"},
        }],
        "outputs": [{
            "id": "masks",
            "axes": [
                {"type": "batch"},
                {"size": int(output_shape[0]), "id": "y", "type": "space"},
                {"size": int(output_shape[1]), "id": "x", "type": "space"},
            ],
            "test_tensor": {"source": "output_sample.npy"},
        }],
        "weights": {
            "pytorch_state_dict": {
                "source": "model_weights.pth",
                "architecture": {
                    "source": "model.py",
                    "callable": "CellposeSAMWrapper",
                    "kwargs": {
                        "model_type": "cpsam",
                        "diam_mean": float(diam_mean),
                        "cp_batch_size": 8,
                        "channels": [0, 0],
                        "flow_threshold": 0.4,
                        "cellprob_threshold": 0.0,
                        "stitch_threshold": 0.0,
                        "estimate_diam": False,
                        "normalize": True,
                        "do_3D": False,
                        # CPU + float32 so bioimageio.core reproduces the packaged
                        # output_sample deterministically; GPU serving bypasses the RDF.
                        "gpu": False,
                        "use_bfloat16": False,
                    },
                },
                "pytorch_version": str(torch.__version__),
            }
        },
        "config": {
            "bioimageio": {
                "reproducibility_tolerance": [
                    {"relative_tolerance": 0.01, "absolute_tolerance": 0.001,
                     "mismatched_elements_per_million": 20},
                ]
            }
        },
    }
    if provenance:
        rdf["config"]["cellpose_provenance"] = provenance
    return rdf


def _doc(name: str, session_id: str, diam_mean: float) -> str:
    return f"""# {name}

Cellpose-SAM model fine-tuned in BioEngine (session `{session_id}`).

## Model
- Architecture: Cellpose-SAM (Transformer), wrapped as `CellposeSAMWrapper`
- Mean diameter: {diam_mean} pixels
- Flow threshold: 0.4 · Cell-probability threshold: 0.0

## Usage
```python
import torch
from model import CellposeSAMWrapper

model = CellposeSAMWrapper()
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()
masks = model(input_tensor)  # (B, 3, H, W) -> (B, H, W) instance labels
```

## License
BSD-3-Clause (Cellpose license).
"""


def _cover(pkg_dir: Path, input_chw: np.ndarray, mask_hw: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    disp = np.transpose(input_chw, (1, 2, 0))
    disp = (disp - disp.min()) / (disp.ptp() or 1.0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5))
    a1.imshow(disp); a1.set_title("input"); a1.axis("off")
    a2.imshow(mask_hw, cmap="tab20"); a2.set_title(f"{len(np.unique(mask_hw)) - 1} objects"); a2.axis("off")
    plt.tight_layout()
    plt.savefig(pkg_dir / "cover.png", dpi=120, bbox_inches="tight")
    plt.close()


def main(session_id: str, export_dir: str) -> None:
    import torch

    t0 = time.time()
    export_path = Path(export_dir)
    request = json.loads((export_path / "request.json").read_text())
    name = request["name"]
    provenance = request.get("provenance")
    # cpsam-derived weights inherit Cellpose's BSD-3-Clause; the entry's SPDX
    # default (CC-BY-4.0, for micro-sam) does not apply to this backend.
    license_id = "BSD-3-Clause"

    params = training.read_training_params(session_id)
    ckpt = training.checkpoint_path(session_id)
    if not ckpt.exists():
        raise FileNotFoundError(f"No cellpose checkpoint for session '{session_id}' at {ckpt}.")
    diam_mean = float(params.get("diam_mean", 30.0))

    pkg_dir = export_path / "package"
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True)

    shutil.copy(str(ckpt), pkg_dir / "model_weights.pth")
    shutil.copy(str(HERE / "cellpose_model.py"), pkg_dir / "model.py")

    input_chw = _center_crop(_to_chw3(_load_array(params["train_images"][0])))
    input_sample = input_chw[None].astype(np.float32)  # (1, 3, H, W)

    sys.path.insert(0, str(HERE))
    from cellpose_model import CellposeSAMWrapper

    wrapper = CellposeSAMWrapper(
        model_type="cpsam", diam_mean=diam_mean, cp_batch_size=8, channels=[0, 0],
        flow_threshold=0.4, cellprob_threshold=0.0, stitch_threshold=0.0,
        estimate_diam=False, normalize=True, do_3D=False, gpu=False, use_bfloat16=False,
    )
    wrapper.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    wrapper.eval()
    with torch.no_grad():
        out = wrapper(torch.from_numpy(input_sample).float())
    output_sample = out.detach().cpu().numpy().astype(np.float32)  # (1, H, W)

    np.save(pkg_dir / "input_sample.npy", input_sample)
    np.save(pkg_dir / "output_sample.npy", output_sample)

    rdf = _build_rdf(
        name, request.get("description", ""), request.get("authors"), license_id,
        diam_mean, provenance, input_sample.shape[1:], output_sample.shape[1:],
    )
    (pkg_dir / "rdf.yaml").write_text(yaml.safe_dump(rdf, sort_keys=False))
    (pkg_dir / "documentation.md").write_text(_doc(name, session_id, diam_mean))
    try:
        _cover(pkg_dir, input_chw, output_sample[0])
    except Exception as e:  # cover is best-effort
        print(f"cover skipped: {e}", flush=True)

    from bioimageio.core import test_model

    summary = test_model(pkg_dir / "rdf.yaml")
    status = getattr(summary, "status", None)
    if status != "passed":
        detail = ""
        try:
            detail = summary.format()
        except Exception:
            detail = str(status)
        raise RuntimeError(f"bioimageio test_model did not pass (status={status}):\n{detail[-3000:]}")

    zip_path = export_path / f"{name.strip().lower().replace(' ', '-') or 'cellpose-sam'}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pkg_dir.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(pkg_dir).as_posix())

    files = [
        {"name": f.relative_to(pkg_dir).as_posix(), "size": f.stat().st_size}
        for f in sorted(pkg_dir.rglob("*")) if f.is_file()
    ]
    result = {
        "package_dir": str(pkg_dir),
        "zip": str(zip_path),
        "zip_size": zip_path.stat().st_size,
        "files": files,
        "total_bytes": sum(f["size"] for f in files),
        "model_type": "cpsam",
        "test_model_status": status,
        "build_seconds": round(time.time() - t0, 1),
    }
    (export_path / "export_result.json").write_text(json.dumps(result))
    print("EXPORT_OK", pkg_dir, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
