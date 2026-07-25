# Cellpose4 Runner 🔬

Runs **Cellpose-4-based bioimage.io models** for instance segmentation through
[`bioimageio.core`](https://github.com/bioimage-io/core-bioimage-io-python).
Tailored and PyTorch-only — a small alternative to the general `model-runner`.

Switch models by their bioimage.io id: the app keeps **one model resident** and
reloads (freeing the previous model's GPU memory) when a different `model_id` is
requested.

Supported models (see the `infer` `model_id` schema for the live list):

| Model | Type | Input | Output |
|---|---|---|---|
| `idealistic-eagle` (published) | Cellpose-SAM | 3-channel 2D | float32 labels |
| `passionate-bug` (in review) | Cellpose-DINO ViT-B | 1-channel 2D | uint16 labels |

Any Cellpose-4 model exposing a `pytorch_state_dict` weight that resolves from
its bioimage.io id (or full rdf URL) should run.

## Methods

- **`infer(model_id, inputs, device="cuda", sample_id="sample", return_download_url=False)`**
  Run the forward pass. `inputs` is a numpy array, a direct http(s) URL, or a
  `get_upload_url` file path. Returns `{output_id: array}` (e.g. `{"labels": ...}`),
  or `{output_id: presigned_url}` when `return_download_url=True`.
- **`get_upload_url(file_type)`** Presigned S3 PUT URL (1-hour TTL) for staging an
  input image. Upload via HTTP PUT, then pass the returned `file_path` to `infer`.
- **`ping()`** Status + the currently resident `model_id`.

## Usage

```python
import numpy as np
from hypha_rpc import connect_to_server

server = await connect_to_server({"server_url": "https://hypha.aicell.io", "token": TOKEN})
svc = await server.get_service("<workspace>/cellpose4-runner")  # concrete per-replica id

# 3-channel input for Cellpose-SAM (idealistic-eagle)
img = np.random.rand(1, 3, 256, 256).astype("float32")
out = await svc.infer(model_id="idealistic-eagle", inputs=img)
labels = out["labels"]
```

## Deployment

The app reads `HYPHA_TOKEN` at startup (Hypha connection + S3 upload/download),
so it must be deployed with the token injected:

```python
await worker.deploy_app(
    artifact_id="bioimage-io/cellpose4-runner",
    version="0.1.0",
    application_id="cellpose4-runner",
    hypha_token=HYPHA_TOKEN,
)
```

Notes:
- First deploy is slow — the runtime env pip-installs torch + `cellpose` +
  `segment-anything` + DINOv3 (from git).
- Requires a GPU replica; Cellpose-4 (SAM/DINO) models are large.
