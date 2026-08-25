"""BioImage.IO export subprocess (upstream micro-sam ``export_sam_model``).

Launched by ``RuntimeApp.export_bioimageio`` as
``python export_worker.py <session_id> <export_dir>`` with ``CUDA_VISIBLE_DEVICES=""``
so it runs on CPU (no GPU contention with serving/training). Reads
``<export_dir>/request.json`` ({name, description, authors, license, provenance})
and builds a **standard combined SAM+decoder** BioImage.IO package — the
interactive prompt head *and* the AIS decoder in one ``{"model_state",
"decoder_state"}`` checkpoint, run without prompts for automatic instance
segmentation — from the session's ``best.pt`` via
``micro_sam.bioimageio.export_sam_model``. micro-sam 1.8.11 (PR #1348) emits the
image-only AIS RDF natively — image-derived y/x minima, the embeddings
``reproducibility_tolerance``, and an eval/CPU-deterministic self-test — so we
just unzip, optionally bake provenance into ``rdf.yaml``'s ``config``, gate the
result on ``bioimageio.core.test_model``, and re-zip; no package rewrite.

``license`` in the request is not honoured — ``export_sam_model`` hard-codes
CC-BY-4.0 on the RDF.
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


def _load_array(path: str) -> np.ndarray:
    import imageio.v3 as iio

    ext = Path(str(path).split("?")[0]).suffix.lower()
    arr = np.load(path) if ext == ".npy" else iio.imread(path)
    return np.asarray(arr)


def _pick_label_with_two_instances(label_paths):
    """export_sam_model derives its decoder test data from label ids 1 and 2, so
    the label image must carry at least two instances; return the first that does."""
    for lp in label_paths:
        lbl = _load_array(lp)
        ids = np.unique(lbl)
        if ids[ids > 0].size >= 2:
            return lbl
    raise ValueError(
        "Export needs a training label with at least two instances to derive "
        "test data, but none of the session's labels qualify."
    )


def _authors(raw):
    from bioimageio.spec.model import v0_5 as m

    fields = ("name", "affiliation", "email", "github_user", "orcid")
    out = []
    for a in raw or []:
        kwargs = {k: a[k] for k in fields if a.get(k)}
        if kwargs.get("name"):
            out.append(m.Author(**kwargs))
    return out


def _bake_provenance(pkg_dir: Path, provenance) -> None:
    """Add ``config.microsam_provenance`` to the packaged RDF and re-zip so the
    download zip and the streamed package files stay identical."""
    rdf_path = pkg_dir / "rdf.yaml"
    rdf = yaml.safe_load(rdf_path.read_text())
    rdf.setdefault("config", {})["microsam_provenance"] = provenance
    rdf_path.write_text(yaml.safe_dump(rdf, sort_keys=False))


def _rezip(pkg_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pkg_dir.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(pkg_dir).as_posix())


def _slug(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_" else "-" for c in name.strip().lower())
    return "-".join(p for p in keep.split("-") if p) or "micro-sam"


def _test_model_gate(pkg_dir: Path) -> None:
    from bioimageio.core import test_model

    summary = test_model(pkg_dir / "rdf.yaml")
    if summary.status != "passed":
        raise RuntimeError("AIS export self-test failed:\n" + summary.format())


def main(session_id: str, export_dir: str) -> None:
    from micro_sam.bioimageio import export_sam_model

    t0 = time.time()
    export_path = Path(export_dir)
    request = json.loads((export_path / "request.json").read_text())
    name = request["name"]
    provenance = request.get("provenance")

    params = training.read_training_params(session_id)
    ckpt = training.checkpoint_path(session_id)
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint for session '{session_id}'.")

    model_type = params["model_type"]
    image = _load_array(params["train_images"][0])
    label_image = _pick_label_with_two_instances(params["train_labels"])

    export_kwargs = {}
    if request.get("description"):
        export_kwargs["description"] = request["description"]
    authors = _authors(request.get("authors"))
    if authors:
        export_kwargs["authors"] = authors

    zip_path = export_path / f"{_slug(name)}.zip"
    export_sam_model(
        image=image, label_image=label_image, model_type=model_type,
        name=name, output_path=str(zip_path), checkpoint_path=str(ckpt),
        **export_kwargs,
    )

    pkg_dir = export_path / "package"
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(pkg_dir)

    if provenance:
        _bake_provenance(pkg_dir, provenance)
    _test_model_gate(pkg_dir)
    _rezip(pkg_dir, zip_path)

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
        "model_type": model_type,
        "build_seconds": round(time.time() - t0, 1),
    }
    (export_path / "export_result.json").write_text(json.dumps(result))
    print("EXPORT_OK", pkg_dir, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
