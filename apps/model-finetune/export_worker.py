"""BioImage.IO export subprocess (upstream micro-sam ``export_sam_model``).

Launched by ``RuntimeApp.export_bioimageio`` as
``python export_worker.py <session_id> <export_dir>`` with ``CUDA_VISIBLE_DEVICES=""``
so it runs on CPU (no GPU contention with serving/training). Reads
``<export_dir>/request.json`` ({name, description, authors, license, provenance})
and builds a **standard combined SAM+decoder** BioImage.IO package — the
interactive prompt head *and* the AIS decoder in one ``{"model_state",
"decoder_state"}`` checkpoint, run without prompts for automatic instance
segmentation — from the session's ``best.pt`` via
``micro_sam.bioimageio.export_sam_model``. We then rewrite the package so its
self-test exercises the **AIS decoder, not the promptable head** (see
``_aisify_package``): ``export_sam_model`` declares five optional prompt inputs
each carrying a ``test_tensor``, so bioimageio tests the prompt path — whose mask
is all-zero (and rejected by spec's dynamic-range check) on data where the
prompts miss. We drop those inputs to an image-only RDF, regenerate the outputs
from a no-prompt ``PredictorAdaptor`` call (the AIS path), and gate the result on
``bioimageio.core.test_model``. We unzip into ``<export_dir>/package/``, optionally
bake provenance into ``rdf.yaml``'s ``config``, and write ``export_result.json``
for the entry to stage/upload.

``license`` in the request is not honoured — ``export_sam_model`` hard-codes
CC-BY-4.0 on the RDF.
"""

import hashlib
import json
import shutil
import sys
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

import training

HERE = Path(__file__).parent

# Image-only AIS RDF shape, matched byte-for-byte to the published zoo models
# (micro-sam branch fix/adaptor-embedding-recompute). test_model probes the model
# at y/x = min, min+step, min+2*step; AIS must segment the resized test image at
# each. min is raised well above the spec default (ParameterizedSize(min=1,
# step=1)): below 192 px AIS collapses to 0-1 instances across all six zoo models,
# 256 is the floor with usable margin. step is 1 so any native test-image size
# >= min stays a valid parametrized size (the test tensor is the user's arbitrary-
# sized training image).
_AIS_IMAGE_MIN = 256
_AIS_IMAGE_STEP = 1
# Embeddings are raw float features; a CPU-generated reference vs a CUDA (or MPS)
# test_model run drifts ~255 ppm cross-device class. masks/scores stay strict.
# Matches the published zoo AIS RDFs (round 3) and upstream export PR #1348.
_AIS_EMBEDDINGS_MISMATCH_PPM = 1000


def _load_array(path: str) -> np.ndarray:
    import imageio.v3 as iio

    ext = Path(str(path).split("?")[0]).suffix.lower()
    arr = np.load(path) if ext == ".npy" else iio.imread(path)
    return np.asarray(arr)


def _pick_label_with_two_instances(label_paths):
    """export_sam_model derives prompt test data from label ids 1 and 2, so the
    label image must carry at least two instances; return the first that does."""
    for lp in label_paths:
        lbl = _load_array(lp)
        ids = np.unique(lbl)
        if ids[ids > 0].size >= 2:
            return lbl
    raise ValueError(
        "Export needs a training label with at least two instances to derive "
        "prompt test data, but none of the session's labels qualify."
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def _suppress_degenerate_output_check():
    """``export_sam_model`` self-tests the promptable head while building; where
    the prompts miss, the prompted mask is all-zero and spec's dynamic-range check
    aborts the build. We ship only the AIS path (regenerated in ``_aisify_package``),
    so swallow just that one check during the build, then restore it so the final
    ``test_model`` gate validates honestly."""
    from bioimageio.spec.model import v0_5 as v

    orig = v.validate_tensors

    def patched(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        except ValueError as e:
            if "too small for reliable testing" in str(e):
                return None
            raise

    v.validate_tensors = patched
    try:
        yield
    finally:
        v.validate_tensors = orig


def _run_ais(pkg_dir: Path) -> dict:
    """Predict the package image-only through the bioimageio pipeline (AIS branch)
    and return ``{masks, scores, embeddings}``.

    Run through the same machinery ``test_model`` uses so the regenerated test
    tensors reproduce exactly under the gate; run twice and assert equality so a
    non-deterministic AIS watershed can never ship a self-test that flakes.
    """
    import xarray
    import bioimageio.core
    from bioimageio.core import load_description
    from bioimageio.core.digest_spec import create_sample_for_model

    # perform_io_checks=False: the patched adaptor is already overlaid (arch sha
    # patched below), but the rdf still carries the prompt-path output shas here.
    desc = load_description(pkg_dir / "rdf.yaml", perform_io_checks=False)
    image = np.load(pkg_dir / "input.npy")
    image_da = xarray.DataArray(image, dims=("batch", "channel", "y", "x"))
    runs = []
    with bioimageio.core.create_prediction_pipeline(desc) as pipeline:
        for _ in range(2):
            sample = create_sample_for_model(model=desc, inputs={"image": image_da})
            result = pipeline.predict_sample_without_blocking(sample)
            runs.append({k: np.asarray(v.data) for k, v in result.members.items()})
    for key in ("masks", "scores", "embeddings"):
        if not np.array_equal(runs[0][key], runs[1][key]):
            raise RuntimeError(f"AIS output {key} is non-deterministic; export unsafe.")
    return runs[0]


def _aisify_package(pkg_dir: Path) -> None:
    """Rewrite the built package so its self-test runs AIS, not the prompt head.

    Overlays the vendored embedding-cache-fixed adaptor and patches the RDF
    architecture sha (so ``import_callable`` accepts it at pipeline creation),
    regenerates the ``masks``/``scores``/``embeddings`` test tensors from a
    no-prompt run through the bioimageio pipeline, then trims the inputs to
    ``image`` only (raising its y/x ``min`` so test_model's size probes land where
    AIS segments) and drops the now-unreferenced prompt test tensors. Converges
    byte-for-byte on the published zoo models' AIS RDF shape.
    """
    rdf_path = pkg_dir / "rdf.yaml"
    rdf = yaml.safe_load(rdf_path.read_text())

    arch = rdf["weights"]["pytorch_state_dict"]["architecture"]
    shutil.copyfile(HERE / "predictor_adaptor.py", pkg_dir / arch["source"])
    arch["sha256"] = _sha256(pkg_dir / arch["source"])
    rdf_path.write_text(yaml.safe_dump(rdf, sort_keys=False))

    outs = _run_ais(pkg_dir)
    if outs["masks"].shape[1] == 0:
        raise RuntimeError(
            "AIS produced 0 instances on the export test image; the self-test "
            "output would be degenerate. Train on data the decoder can segment."
        )

    outputs = {d["id"]: d for d in rdf["outputs"]}
    saves = {"masks": outs["masks"].astype("uint8"),
             "scores": outs["scores"].astype("float32"),
             "embeddings": outs["embeddings"].astype("float32")}
    for oid, arr in saves.items():
        desc = outputs[oid]
        src = pkg_dir / desc["test_tensor"]["source"]
        np.save(src, arr)
        desc["test_tensor"]["sha256"] = _sha256(src)

    inputs = {d["id"]: d for d in rdf["inputs"]}
    image_desc = inputs["image"]
    for ax in image_desc["axes"]:
        size = ax.get("size")
        if ax.get("id") in ("x", "y") and isinstance(size, dict) and "min" in size:
            size["min"] = _AIS_IMAGE_MIN
            size["step"] = _AIS_IMAGE_STEP

    keep = {image_desc["test_tensor"]["source"]}
    keep |= {outputs[o]["test_tensor"]["source"] for o in ("masks", "scores", "embeddings")}
    for d in rdf["inputs"]:
        src = d["test_tensor"]["source"]
        if d["id"] != "image" and src not in keep:
            (pkg_dir / src).unlink(missing_ok=True)
    rdf["inputs"] = [image_desc]

    cfg = rdf.setdefault("config", {}).setdefault("bioimageio", {})
    cfg.setdefault("reproducibility_tolerance", []).append(
        {"output_ids": ["embeddings"],
         "mismatched_elements_per_million": _AIS_EMBEDDINGS_MISMATCH_PPM}
    )

    rdf_path.write_text(yaml.safe_dump(rdf, sort_keys=False))


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
    with _suppress_degenerate_output_check():
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

    _aisify_package(pkg_dir)
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
