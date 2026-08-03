# Cellpose4 Runner 🔬

Inference-only service for **supported Cellpose-4 bioimage.io models** through
[`bioimageio.core`](https://github.com/bioimage-io/core-bioimage-io-python).
PyTorch-only — a small alternative to the general `model-runner` (which ships
Cellpose-3 and cannot run Cellpose-4 models).

The app keeps **one model resident** and reloads (freeing the previous model's
GPU memory) when a different override set is requested.

Supported models are an explicit allow-list — any other `model_id` is rejected.
Query the live set with `list_supported_models()`:

| Model | Type | Input | Output |
|---|---|---|---|
| `idealistic-eagle` | Cellpose-SAM | 3-channel 2D | instance labels |

## Methods

- **`list_supported_models()`** Returns the bioimage.io ids this service accepts
  (currently `["idealistic-eagle"]`). Any other id passed to `infer` is rejected.
- **`infer(model_id, inputs, sample_id="sample", flow_threshold=None, cellprob_threshold=None, min_size=None, return_flows=False, return_download_url=False)`**
  Submits an inference request and returns a `request_id` **immediately**. Poll
  `get_infer_status(request_id)` for progress and the result (same submit/poll
  contract as `model-runner`). `inputs` is a numpy array, a direct http(s) URL,
  or a `get_upload_url` file path.
  - `flow_threshold` / `cellprob_threshold` / `min_size` override the Cellpose
    flow-dynamics postprocessing; when left `None` the model's RDF defaults apply
    (Cellpose-SAM: `0.4` / `0.0` / `15`). Overrides patch only an in-memory copy
    of the RDF — the published artifact is never modified — and a changed override
    set reloads the resident pipeline.
  - `return_flows=True` skips the flow-dynamics postprocessing and returns the raw
    flow field (`{"flows": array}`, 3 channels = 2 flow components + cell
    probability) instead of instance masks. The overrides above do not apply in
    this mode.
  - `return_download_url=True` returns each output as a presigned S3 `.npy` URL
    (1-hour TTL) instead of the raw array.
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
svc = await server.get_service("<workspace>/cellpose4-runner")  # concrete per-replica id

# 3-channel input for Cellpose-SAM (idealistic-eagle)
img = np.random.rand(1, 3, 256, 256).astype("float32")
request_id = await svc.infer(model_id="idealistic-eagle", inputs=img)

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
    artifact_id="bioimage-io/cellpose4-runner",
    version="0.3.0",
    application_id="cellpose4-runner",
    hypha_token=HYPHA_TOKEN,
)
```

Notes:
- First deploy is slow — the runtime env pip-installs torch + `cellpose`.
- Requires a GPU replica; Cellpose-4 (SAM) models are large.
```
