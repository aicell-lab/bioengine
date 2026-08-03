# micro-sam (μSAM) 🔬

Serves [micro-sam](https://github.com/computational-cell-analytics/micro-sam)
(μSAM) for microscopy segmentation and interactive annotation. **One fine-tuned
SAM image encoder stays resident on the GPU and backs three cheap consumers**
(the paper's "Leg B" — one resident encoder, many lightweight decoders):

1. **Image embedding** — run the encoder once per image; the embedding feeds an
   in-browser ONNX prompt decoder, so "draw a bounding box → get a cell mask"
   needs no GPU round-trip per prompt.
2. **Automatic instance segmentation (AIS decoder)** — the UNETR instance
   decoder on the `*_lm` models produces a full instance label mask without
   prompts. This is the propose-and-prune pre-segmentation.
3. **ONNX prompt decoder** — the lightweight interactive decoder exported to
   ONNX bytes for `onnxruntime-web`.

Motivated by annotation ergonomics and low-contrast brightfield (where drawing
masks from scratch is slow); **not** an accuracy claim over Cellpose-SAM.

## Models

Use a light-microscopy (LM) generalist for brightfield/fluorescence cells — the
`*_lm` models carry the AIS decoder so automatic segmentation works out of the
box.

| `model_type` | Notes |
|---|---|
| `vit_b_lm` (default) | Speed/quality balance for cells & nuclei in LM |
| `vit_l_lm` | Higher quality, slower |
| `vit_t_lm`, `vit_b`, `vit_l`, `vit_h` | Also selectable (base SAM has no AIS decoder → AMG fallback) |

Switching `model_type` frees the previous model's VRAM and loads the new one
(one resident at a time).

## Methods

- **`infer(input_arrays, model_type="vit_b_lm", device="cuda", ...)`**
  Automatic μSAM instance segmentation (propose-and-prune). Cellpose `infer`
  drop-in: returns a **bare list**, one item per input, each
  `{"output": <int32 [H,W] instance label mask>}` (bg 0, one positive int per
  object). `input_arrays` items are numpy arrays (HxW, HxWx3, or 3xHxW), http(s)
  URLs, or `get_upload_url` file paths. Optional `pred_iou_thresh`,
  `stability_score_thresh`, `min_size` tune post-processing; Cellpose-only
  params (`diameter`, `flow_threshold`, …) are accepted and ignored.
- **`segment_image(inputs, model_type="vit_b_lm", device="cuda")`**
  Single-image alias of `infer` (same `[{"output": …}]` shape).
- **`compute_image_embedding(inputs, model_type="vit_b_lm", output_mode="embedding", return_features_url=False)`**
  Run the resident encoder once. Returns `{features (1,256,64,64) f32,
  original_image_shape [H,W], sam_scale, mask_threshold}`. With
  `output_mode="embedding+masks"` also returns `masks` (int32 [H,W]). With
  `return_features_url=True` the 4MB `features` come back as a presigned `.npy`
  URL (`features_url`) instead of raw — for large batches / slow links.
- **`get_onnx_model(model_type="vit_b_lm", quantize=True)`**
  The interactive prompt decoder as ONNX bytes (cached per model). Fetch once
  per session, run with `onnxruntime-web`, decode each box locally using the
  `compute_image_embedding` features.
- **`get_upload_url(file_type)`** Presigned S3 PUT URL (1-hour TTL) for staging
  an input image.
- **`ping()`** Status + the currently resident `model_type`.

## Interactive annotation loop (Option A — in-browser decode)

```
once per session:  onnx = get_onnx_model("vit_b_lm", quantize=True)
once per image:    emb  = compute_image_embedding(img, output_mode="embedding")
per user box:      decode locally via onnxruntime-web using onnx + emb.features   # no GPU
auto pre-seg:      infer([img], "vit_b_lm") -> [{"output": int32 [H,W]}]          # propose-and-prune
```

## Deployment

The app reads `HYPHA_TOKEN` at startup (Hypha connection + S3 upload/download),
so it must be deployed with the token injected:

```python
await worker.deploy_app(
    artifact_id="bioimage-io/micro-sam",
    version="0.1.0",
    application_id="micro-sam",
    hypha_token=HYPHA_TOKEN,
)
```

Notes:
- First deploy is slow — the runtime env pip-installs `micro-sam` (+ `torch-em`,
  `segment-anything`, `bioimage-cpp`) and `onnxruntime`.
- Requires a GPU replica; SAM ViT encoders are large. `vit_b_lm` fits comfortably
  in a few GB of VRAM.
- micro-sam is pip-installable (no conda/mamba) — `bioimage-cpp` supplies the
  C++ pieces `python-elf` used to need from conda-forge.
