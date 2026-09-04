# BioImageIO Fine-tune (μSAM + Cellpose) 🔬

Serves [micro-sam](https://github.com/computational-cell-analytics/micro-sam)
(μSAM) for microscopy segmentation and interactive annotation, and adds in-app
fine-tuning of **[Cellpose](https://github.com/MouseLand/cellpose)** — both the
SAM backbone (`cpsam`) and the DINOv3 backbone (`cpdino` / `cpdino-vitb`) — as an
isolated second backend (see *Architecture* below). On the micro-sam side,
**one fine-tuned SAM image encoder stays resident on the GPU and backs three cheap
consumers** (the paper's "Leg B" — one resident encoder, many lightweight
decoders):

1. **Image embedding** — run the encoder once per image; the embedding feeds an
   in-browser ONNX prompt decoder, so "draw a bounding box → get a cell mask"
   needs no GPU round-trip per prompt.
2. **Automatic instance segmentation (AIS decoder)** — the UNETR instance
   decoder on the generalist models produces a full instance label mask without
   prompts. This is the propose-and-prune pre-segmentation.
3. **ONNX prompt decoder** — the lightweight interactive decoder exported to
   ONNX bytes for `onnxruntime-web`.

Motivated by annotation ergonomics and low-contrast brightfield (where drawing
masks from scratch is slow); **not** an accuracy claim over Cellpose-SAM.

## Architecture (CPU entry + two GPU runtimes)

Like `model-runner`, the app is split into a **CPU `EntryApp`** (`entry.py`) and
GPU runtimes; only the entry is named in the manifest and it composes the
runtimes by type hint. The client always talks to the entry.

- **EntryApp** (CPU, 1 replica) — Hypha/S3 transport, training data
  materialization, session orchestration, and request routing to the runtimes.
- **RuntimeApp** (GPU, autoscaling `min=1 / max=3`, `runtime.py`) — the micro-sam
  backend: resident SAM encoder, a single `asyncio` **GPU lock over every GPU op**
  (serving *and* training), and fine-tuning in a **subprocess** (VRAM fully
  reclaimed on exit; the resident inference model is evicted first).
- **CellposeRuntime** (GPU, same shape, `runtime_cellpose.py`) — the Cellpose
  backend (`cpsam`, `cpdino`, `cpdino-vitb`), in its **own pip env**: cellpose
  pins `numpy==1.26.4` while micro-sam's `python-elf` needs `numpy>=2`, so the two
  backends cannot share a deployment. The env also carries `dinov3` for the cpdino
  ViT backbone. The entry routes by `model_type` (`cpsam`/`cpdino*` →
  CellposeRuntime, `vit_*` → RuntimeApp). Only one fine-tuning runs at a time
  across **both** backends (a single entry-side training lock).

Because each GPU runtime autoscales, a long training run holds one GPU replica's
lock while a concurrent inference request spins up and runs on the **second GPU
replica** — training and inference at the same time. Returned arrays are
hypha-rpc ndarray wire-dicts, which decode to real ndarrays on the client.

## Models

Pick a generalist for your modality — all six carry the AIS decoder, so automatic
segmentation works out of the box. Light-microscopy (`*_lm`) for
brightfield/fluorescence cells; EM-organelles (`*_em_organelles`) for organelles
in electron microscopy. Their weights load from **version-pinned bioimage.io
records** (`runtime.py` `ZOO_MODELS`), so a zoo-side update never changes what is
served without an app release.

| `model_type` | Notes |
|---|---|
| `vit_l_lm` (default) | Best quality for cells & nuclei in LM (~5 GB resident) |
| `vit_b_lm`, `vit_t_lm` | Lighter/faster LM, lower quality |
| `vit_l_em_organelles`, `vit_b_em_organelles`, `vit_t_em_organelles` | Organelles in EM (AIS decoder) |
| `vit_b`, `vit_l`, `vit_h` | Base SAM — no AIS decoder → AMG fallback |
| `cpsam` | Cellpose-SAM (SAM ViT-L backbone) — routed to the isolated CellposeRuntime; segmentation only (no embedding/ONNX). Best for fine-tuning cell segmentation with dense masks. |
| `cpdino`, `cpdino-vitb` | Cellpose-DINO (DINOv3 ViT-L / ViT-B backbone) — same CellposeRuntime, segmentation only. `cpdino-vitb` is the lighter ViT-B variant. |

Switching `model_type` frees the previous model's VRAM and loads the new one
(one resident at a time, per runtime).

## Methods

- **`infer(input_arrays=None, embeddings=None, model_type="vit_l_lm", min_size=None, session_id=None)`**
  Automatic μSAM instance segmentation (propose-and-prune). Returns a **bare
  list**, one item per input, each `{"output": <int32 [H,W] instance label
  mask>}` (bg 0, one positive int per object). Pass **`input_arrays`** (numpy
  arrays HxW/HxWx3/3xHxW, http(s) URLs, or `get_upload_url` file paths) **or**
  **`embeddings`** — a list of precomputed embeddings (an `embedding_url` from
  `compute_embedding(return_url=True)`, or a `compute_embedding` result dict).
  With `embeddings` the AIS decoder runs on the stored embedding **without
  re-encoding** (the model is inferred from the embedding). `min_size` drops
  smaller objects. `session_id` serves a fine-tuned checkpoint (micro-sam **or**
  cellpose — the session's backend picks the runtime). `model_type` in
  `{cpsam, cpdino, cpdino-vitb}` (or a cellpose `session_id`/`model_id`) routes to
  the Cellpose runtime; `embeddings` are micro-sam only.
- **`compute_embedding(inputs, model_type="vit_l_lm", return_url=False, embedding_upload_url=None, session_id=None)`**
  Run the resident encoder once. Returns `{features (1,256,64,64) f32,
  original_image_shape [H,W], input_size [h,w], sam_scale, mask_threshold,
  model_type}`. With `return_url=True` the embedding is saved as a
  self-contained `.npz` in a temporary S3 file and returned as `embedding_url`
  (feed straight to `infer(embeddings=[…])`). With `embedding_upload_url` (a
  presigned PUT URL from `get_upload_url('.npz')`) the `.npz` is stored at your
  own URL instead — you keep the matching download URL for `infer`.
- **`get_onnx_model(model_type="vit_l_lm", quantize=True)`**
  The interactive prompt decoder as ONNX bytes (cached per model). Fetch once
  per session, run with `onnxruntime-web`, decode each box locally using the
  `compute_embedding` features. `compute_embedding` and `get_onnx_model` are
  **micro-sam only** — cpsam has no encoder embedding or ONNX prompt decoder.
- **`get_upload_url(file_type)`** Presigned S3 PUT URL (1-hour TTL) for staging
  an input image (`.npy`/`.png`/`.tif`/`.jpg`) or an embedding bundle (`.npz`).

**Embedding reuse (Leg B).** Encode once, reuse the embedding for both the
in-browser interactive decoder and automatic segmentation without a second
encoder pass:
`image → get_upload_url → compute_embedding(return_url=True) → embedding_url →
infer(embeddings=[embedding_url]) → mask`.

> `device` is chosen internally (CUDA, CPU fallback) — not an API parameter.

### Fine-tuning (train → serve)

Retrain μSAM **or Cellpose** on your own annotated pairs and serve the
just-trained model — no export step needed. Both backends need **dense** labels
(annotate *all* objects per image): micro-sam's AIS decoder
(`with_segmentation_decoder=True`) and cellpose both learn from full instance
masks. Pick the backend with `model_type` (`vit_*` vs `cpsam`/`cpdino`/`cpdino-vitb`);
the same annotated-pair inputs feed both. The cellpose types differ only in
backbone — `cpsam` (SAM ViT-L), `cpdino` (DINOv3 ViT-L), `cpdino-vitb` (DINOv3
ViT-B) — and share the same training/serve/export path; a fine-tuned checkpoint
self-identifies its backbone, so serving is identical.

- **`start_training(train_images, train_labels, val_images=None, val_labels=None, model_type="vit_l_lm", n_epochs=5, n_objects_per_batch=8, patch_size=512, diam_mean=30.0, batch_size=1, learning_rate=1e-5, val_fraction=0.2, n_samples=None, resume_session_id=None, label="")`**
  Starts a background fine-tuning session and returns immediately with the
  status (incl. `session_id`). `train_images` are arrays / URLs / `get_upload_url`
  paths; `train_labels` are dense instance masks (`.tif`/`.png`/`.npy`) or a
  `.geojson` FeatureCollection of polygons (rasterized to instances).
  `n_objects_per_batch`/`patch_size` are micro-sam knobs; `diam_mean` (mean object
  diameter, px) is the cellpose knob (cpsam and cpdino alike). Only one session
  trains at a time across both backends (`QUEUED` while another holds the slot).
- **`get_training_status(session_id)`** → `{status, elapsed_s, n_epochs, checkpoint_available, message, ...}`. `status` ∈ `PREPARING | TRAINING | COMPLETED | FAILED | STOPPED`.
- **`list_training_sessions()`** → all sessions on this worker.
- **`stop_training(session_id)`** Request cancellation (an in-flight epoch may finish first).
- **`export_model(session_id, name, description="", authors=None, license="CC-BY-4.0", provenance=None)`**
  Build a BioImage.IO package from a COMPLETED session and **self-test it on CPU**
  (`bioimageio.core.test_model`) before saving, so a returned package is already
  spec-valid and reproducible. Async, like `start_training`: returns immediately
  with an `export_id` — poll `get_export_status`. The package shape depends on the
  session's backend:
  - **micro-sam** — a **standard combined SAM+decoder** package: the interactive
    prompt head *and* the AIS decoder in one `{model_state, decoder_state}`
    checkpoint (run without prompts for AIS), via
    `micro_sam.bioimageio.export_sam_model`. `provenance` → `config.microsam_provenance`;
    `license` is **ignored** (`export_sam_model` hard-codes `CC-BY-4.0`). Needs a
    training label with **≥ 2 instances** (test data uses label ids 1 & 2).
  - **cpsam / cpdino / cpdino-vitb** — a `pytorch_state_dict` package wrapping the
    fine-tuned Cellpose net (`CellposeSAMWrapper` bundled as `model.py`, which
    rebuilds the SAM or DINOv3 backbone from the session's `model_type`, with
    cellpose flow-dynamics postprocessing). The RDF's `output_sample` is a **real
    CPU forward pass** of the wrapper (so `test_model` reproduces it
    deterministically — the served GPU path bypasses this RDF). `provenance` →
    `config.cellpose_provenance`; license is Cellpose's `BSD-3-Clause`.

  **Draft-only.** The package is staged on temporary storage; `export_model`
  **publishes nothing**. The frontend creates the draft artifact with the user's
  own token, then either downloads the zip from `download_url` or calls
  `push_export(export_id, files)` to stream the package files straight into the
  draft.

**Serve the just-trained model:** once `checkpoint_available` is true, pass
`session_id` to any serving method — the fine-tuned checkpoint flows through the
exact same path as the pretrained model:

```python
sid = (await svc.start_training(train_images=imgs, train_labels=lbls, n_epochs=10))["session_id"]
# poll get_training_status(sid) until status == "COMPLETED"
out = await svc.infer(input_arrays=[img], session_id=sid)          # AIS masks from the fine-tuned model
emb = await svc.compute_embedding(inputs=img, session_id=sid) # embedding from the fine-tuned encoder
```

Sessions live under `~/.bioengine/micro_sam_sessions/<session_id>/`. Use
`export_model(session_id, ...)` to build a standard BioImage.IO package of the
fine-tuned model (draft-only — see the method note above).

## Interactive annotation loop (Option A — in-browser decode)

```
once per session:  onnx = get_onnx_model("vit_l_lm", quantize=True)
once per image:    emb  = compute_embedding(img)
per user box:      decode locally via onnxruntime-web using onnx + emb.features   # no GPU
auto pre-seg:      infer([img], "vit_l_lm") -> [{"output": int32 [H,W]}]          # propose-and-prune
```

## Deployment

The app reads `HYPHA_TOKEN` at startup (Hypha connection + S3 upload/download),
so it must be deployed with the token injected:

```python
await worker.deploy_app(
    artifact_id="bioimage-io/model-finetune",
    version="0.13.0",
    application_id="model-finetune",
    hypha_token=HYPHA_TOKEN,
)
```

Notes:
- First deploy is slow — **two** GPU runtime envs pip-install in parallel: the
  micro-sam env (`micro-sam` + `torch-em`, `segment-anything`, `bioimage-cpp`,
  `onnxruntime`) and the Cellpose env (`cellpose==4.2.1.1`, `numpy==1.26.4`, plus
  `dinov3` for the cpdino backbone).
- Requires a GPU replica; SAM ViT encoders are large. `vit_l_lm` (default) needs ~5 GB; `vit_b_lm` is lighter
  in a few GB of VRAM.
- micro-sam is pip-installable (no conda/mamba) — `bioimage-cpp` supplies the
  C++ pieces `python-elf` used to need from conda-forge.
