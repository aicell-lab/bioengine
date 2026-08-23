# Cellpose3 Runner 🔬

Inference-only service for the **Cellpose-3 bioimage.io models** through
[`bioimageio.core`](https://github.com/bioimage-io/core-bioimage-io-python).
PyTorch-only — a small companion to the general `model-runner`, whose runtime
ships Cellpose 4 and therefore cannot run these models.

The app keeps **one model resident** and reloads (freeing the previous model's
memory) when a different model or load parameter set is requested.

Supported models are an explicit allow-list — any other `model_id` is rejected.
Query the live set with `list_supported_models()`:

| Model | Type | Output |
|---|---|---|
| `famous-fish` | CellPose (cyto3) | instance labels |
| `happy-elephant` | CellPose | instance labels |
| `merry-gorilla` | CellPose | instance labels |
| `philosophical-panda` | Cellpose Plant Nuclei ResNet | flows / style |
| `thoughtful-chipmunk` | CellPose | instance labels |

## Methods

- **`list_supported_models()`** Returns the bioimage.io ids this service accepts.
  Any other id passed to `infer` is rejected; route it to `model-runner`.
- **`infer(model_id, inputs, weights_format=None, device=None, default_blocksize_parameter=None, sample_id="sample", preprocessing=None, postprocessing=None, return_download_url=False, cache="check")`**
  Submits an inference request and returns a `request_id` **immediately**. Poll
  `get_infer_status(request_id)` for progress and the result. The parameter
  surface matches `model-runner`'s `infer`. `inputs` is a numpy array, a direct
  http(s) URL, or a `get_upload_url` file path.
  - `preprocessing` / `postprocessing` override the kwargs of the model's
    declared processing ops, as `{op_id: {kwarg: value}}`. They patch only an
    in-memory copy of the RDF — the published artifact is never modified — and a
    changed override set reloads the resident pipeline. An op id the model does
    not declare is an error. Passing `None` instead of a kwarg dict drops the op,
    which is how a model's raw output is obtained:
    `postprocessing={"cellpose_flow_dynamics": None}` returns the flow field
    instead of instance labels. `bioimageio.core` skips a processing stage as a
    whole, so dropping one op drops all of them on that side; mixing a drop with
    a kwarg patch in the same direction is an error.
  - `return_download_url=True` returns each output as a presigned S3 `.npy` URL
    (1-hour TTL) instead of the raw array.
  - `cache` controls the freshness round-trip: `"check"` (default) reloads only
    if the artifact changed, `"skip"` always reloads, `"reuse"` trusts the
    resident pipeline.
- **`get_infer_status(request_id)`** Poll an infer request: returns a progress dict
  (`queue_position`, `submitted_at`, `running`, `completed_at`, `result`, `stages`).
  Once `result` is populated it holds the output dict (or `{"error": ...}` on
  failure). Jobs are held in-memory for 1 hour after completion.
- **`cancel_request(request_id)`** Cancel a still-queued request (drops its
  background task before it reaches the GPU) and return its progress dict. A
  request already running or finished is returned unchanged.
- **`get_upload_url(file_type)`** Presigned S3 PUT URL (1-hour TTL) for staging an
  input image. Upload via HTTP PUT, then pass the returned `file_path` to `infer`.

## Usage

```python
import asyncio
import numpy as np
from hypha_rpc import connect_to_server

server = await connect_to_server({"server_url": "https://hypha.aicell.io", "token": TOKEN})
svc = await server.get_service("<workspace>/cellpose3-runner")  # concrete per-replica id

img = np.random.rand(1, 1, 256, 256).astype("float32")
request_id = await svc.infer(model_id="famous-fish", inputs=img)

# poll until the result is ready
while True:
    status = await svc.get_infer_status(request_id=request_id)
    if status["completed_at"] is not None:
        break
    await asyncio.sleep(1)
labels = status["result"]["labels"]
```

## Deployment

The app reads `HYPHA_TOKEN` at startup (Hypha connection + S3 upload/download),
so it must be deployed with the token injected:

```python
await worker.deploy_app(
    artifact_id="bioimage-io/cellpose3-runner",
    version="0.1.0",
    application_id="cellpose3-runner",
    hypha_token=HYPHA_TOKEN,
)
```

Notes:
- First deploy is slow — the runtime env pip-installs torch + `cellpose`.
- The app defaults to a whole GPU (`gpu_memory_mb=-1`). Deploy with
  `disable_gpu=True` (CLI `--no-gpu`) on GPU-tight clusters to force it onto
  CPU; the device follows what the replica actually got. The KTH deployment
  runs CPU-only this way because both of its GPUs are held by model-runner.
