"""BioImage.IO export subprocess (prompt-free AIS package).

Launched by ``RuntimeApp.export_bioimageio`` as
``python export_worker.py <session_id> <export_dir>`` with ``CUDA_VISIBLE_DEVICES=""``
so it runs on CPU (no GPU contention with serving/training). Reads
``<export_dir>/request.json`` ({name, description, authors, license, provenance}),
builds a standard **prompt-free** BioImage.IO model package from the session's
``best.pt`` — RGB image -> the three micro-SAM AIS maps (foreground, center
distance, boundary distance) — via the custom ``MicroSAMAIS`` architecture
(``ais_adapter.py``, which ships inside the package), unzips it into
``<export_dir>/package/``, and writes ``export_result.json`` for the entry to
serve/upload. The watershed that turns the maps into instances is downstream
(model-runner / the app's own consumer path), not part of the package.
"""

import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

import training

HERE = Path(__file__).parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rgb_float(path: str) -> np.ndarray:
    import imageio.v3 as iio

    ext = Path(str(path).split("?")[0]).suffix.lower()
    arr = np.load(path) if ext == ".npy" else iio.imread(path)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr[..., :3].astype("float32")


def _authors(raw):
    from bioimageio.spec.model import v0_5 as m

    fields = ("name", "affiliation", "email", "github_user", "orcid")
    out = []
    for a in raw or []:
        kwargs = {k: a[k] for k in fields if a.get(k)}
        out.append(m.Author(**kwargs))
    if not out:
        out.append(m.Author(name="micro-sam user"))
    return out


def main(session_id: str, export_dir: str) -> None:
    from bioimageio.spec import save_bioimageio_package
    from bioimageio.spec.model import v0_5 as m
    from micro_sam import util
    from micro_sam.instance_segmentation import get_decoder

    sys.path.insert(0, str(HERE))
    from ais_adapter import MicroSAMAIS

    t0 = time.time()
    export_path = Path(export_dir)
    request = json.loads((export_path / "request.json").read_text())
    name = request["name"]
    provenance = request.get("provenance")

    params = training.read_training_params(session_id)
    ckpt = training.checkpoint_path(session_id)
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint for session '{session_id}'.")

    # export_sam_model's registry only knows base architectures: strip the
    # micro-sam suffix (vit_b_lm -> vit_b); the fine-tuned weights come from the
    # checkpoint, not the registry.
    base_arch = "_".join(params["model_type"].split("_")[:2])
    predictor, state = util.get_sam_model(
        model_type=base_arch, checkpoint_path=str(ckpt), device="cpu", return_state=True,
    )
    if "decoder_state" not in state:
        raise ValueError(
            f"Session '{session_id}' checkpoint has no decoder_state — it was not "
            "trained with the segmentation decoder, so there is no AIS model to export."
        )
    dstate = state["decoder_state"]
    use_conv_transpose = any(
        ".block." in k for k in dstate if k.startswith("decoder.samplers")
    )

    mod = MicroSAMAIS(model_type=base_arch, use_conv_transpose=use_conv_transpose)
    mod.image_encoder.load_state_dict(predictor.model.image_encoder.state_dict())
    ref_dec = get_decoder(predictor.model.image_encoder, dstate, "cpu")
    mod.decoder.load_state_dict(ref_dec.state_dict())
    # eval() AFTER loading: TinyViT (vit_t) derives its non-persistent ``attn.ab``
    # buffer from ``attention_biases`` at eval() time and never saves it, so eval()
    # must run on the finetuned weights or the baked test output uses stale base
    # biases and fails core's reproduce check (bioimageio.core load-then-eval).
    mod.eval()

    build_dir = export_path / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    weights_path = build_dir / "weights.pt"
    import torch

    torch.save(mod.state_dict(), weights_path)

    arch_path = build_dir / "ais_adapter.py"
    shutil.copyfile(HERE / "ais_adapter.py", arch_path)

    img = _load_rgb_float(params["train_images"][0])
    inp = torch.from_numpy(img).permute(2, 0, 1)[None].contiguous()
    input_npy = build_dir / "input.npy"
    output_npy = build_dir / "output.npy"
    np.save(input_npy, inp.numpy().astype("float32"))
    with torch.no_grad():
        out = mod(inp)
    np.save(output_npy, out.numpy().astype("float32"))

    readme = build_dir / "README.md"
    readme.write_text(
        f"# {name}\n\n"
        "Prompt-free micro-SAM automatic instance segmentation model.\n\n"
        "Input: a single RGB image `[batch, 3, y, x]` (float32).\n"
        "Output: three dense maps `[batch, 3, y, x]` — channel 0 foreground, "
        "channel 1 center distance, channel 2 boundary distance.\n\n"
        "Instance labels are produced downstream by the micro-SAM watershed "
        "(`watershed_from_center_and_boundary_distances`); it is not part of this "
        "package because it is not expressible as a spec postprocessing op.\n\n"
        "Note: SAM's 1024px resize is baked into the architecture using a torch "
        "bilinear (antialiased) interpolation rather than micro-SAM's PIL resize, "
        "so the maps differ from native micro-SAM by ~1e-1 before watershed; this "
        "rarely changes the final instance labels.\n"
    )

    def make_descr():
        inputs = [m.InputTensorDescr(
            id=m.TensorId("image"),
            axes=[
                m.BatchAxis(),
                m.ChannelAxis(channel_names=[m.Identifier("r"), m.Identifier("g"), m.Identifier("b")]),
                m.SpaceInputAxis(id=m.AxisId("y"), size=int(img.shape[0])),
                m.SpaceInputAxis(id=m.AxisId("x"), size=int(img.shape[1])),
            ],
            test_tensor=m.FileDescr(source=input_npy),
            data=m.IntervalOrRatioDataDescr(),
        )]
        outputs = [m.OutputTensorDescr(
            id=m.TensorId("maps"),
            axes=[
                m.BatchAxis(),
                m.ChannelAxis(channel_names=[
                    m.Identifier("foreground"),
                    m.Identifier("center_distance"),
                    m.Identifier("boundary_distance"),
                ]),
                m.SpaceOutputAxis(id=m.AxisId("y"), size=m.SizeReference(
                    tensor_id=m.TensorId("image"), axis_id=m.AxisId("y"))),
                m.SpaceOutputAxis(id=m.AxisId("x"), size=m.SizeReference(
                    tensor_id=m.TensorId("image"), axis_id=m.AxisId("x"))),
            ],
            test_tensor=m.FileDescr(source=output_npy),
            data=m.IntervalOrRatioDataDescr(),
        )]
        weights = m.WeightsDescr(pytorch_state_dict=m.PytorchStateDictWeightsDescr(
            source=weights_path,
            sha256=_sha256(weights_path),
            architecture=m.ArchitectureFromFileDescr(
                source=arch_path,
                sha256=_sha256(arch_path),
                callable=m.Identifier("MicroSAMAIS"),
                kwargs={"model_type": base_arch, "use_conv_transpose": use_conv_transpose},
            ),
            pytorch_version=m.Version(torch.__version__.split("+")[0]),
            dependencies=None,
        ))
        # Reproduction across torch/library versions deviates from the stored
        # forward pass by up to ~4e-3 on the sigmoid-range maps (far below the
        # 0.5 watershed thresholds, so instances are unaffected); core's default
        # 1e-3 tolerance is tighter than that, so widen it here.
        cfg_kwargs = {"bioimageio": m.BioimageioConfig(reproducibility_tolerance=[
            m.ReproducibilityTolerance(
                absolute_tolerance=0.006,
                relative_tolerance=1e-3,
                mismatched_elements_per_million=5000,
                output_ids=(m.TensorId("maps"),),
            )
        ])}
        if provenance:
            cfg_kwargs["microsam_provenance"] = provenance
        config = m.Config(**cfg_kwargs)
        return m.ModelDescr(
            name=name,
            description=request.get("description") or "Prompt-free micro-SAM automatic instance segmentation.",
            authors=_authors(request.get("authors")),
            cite=[m.CiteEntry(
                text="Segment Anything for Microscopy",
                doi=m.Doi("10.1038/s41592-024-02580-4"),
            )],
            license=request.get("license") or "CC-BY-4.0",
            documentation=m.FileDescr(source=readme),
            inputs=inputs,
            outputs=outputs,
            weights=weights,
            config=config,
        )

    model = make_descr()

    zip_path = export_path / f"{_slug(name)}.zip"
    save_bioimageio_package(model, output_path=zip_path)

    pkg_dir = export_path / "package"
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(pkg_dir)

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
        "use_conv_transpose": use_conv_transpose,
        "model_type": base_arch,
        "build_seconds": round(time.time() - t0, 1),
    }
    (export_path / "export_result.json").write_text(json.dumps(result))
    print("EXPORT_OK", pkg_dir, flush=True)


def _slug(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_" else "-" for c in name.strip().lower())
    return "-".join(p for p in keep.split("-") if p) or "micro-sam-ais"


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
