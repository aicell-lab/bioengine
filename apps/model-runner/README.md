# BioImage.IO ModelRunner – Deployment Overview

## Overview

**ModelRunner** is a BioEngine application built on **Ray Serve** that provides standardized loading, validation, and execution of BioImage.IO models.
It is implemented as a **two-deployment architecture**:

1. **Entry Deployment (CPU)** – ingress, metadata handling, validation, caching, and request routing
2. **Runtime Deployment (GPU)** – unified execution environment for model testing and inference

This separation enables clear responsibility boundaries, efficient resource utilization, and isolation of heavy ML dependencies to GPU-backed workers.

## Core BioEngine Base Packages

These packages are shared by all BioEngine applications. They are intentionally kept **minimal** to maximize compatibility across environments and deployments.

```
httpx==0.28.1
hypha-rpc==0.21.40
pydantic==2.11.9
```

### Package Roles

* **httpx**
  Used for web communication, including model downloads and external service calls.

* **hypha-rpc**
  Provides communication with **Hypha** services and infrastructure components.

* **pydantic**
  Used for schema definitions of services.

## ModelRunner Architecture

ModelRunner consists of two Ray Serve deployments:

```
[ Client / User ]
        |
        v
+-----------------------+
| Entry Deployment (CPU)|
+-----------------------+
        |
        v
+------------------------+
| Runtime Deployment (GPU)|
+------------------------+
```

## ModelRunner – Entry Deployment (CPU)

**Responsibilities**

The entry deployment acts as the ingress and control plane for ModelRunner. It:

* Receives and validates user requests
* Downloads models and manages the local model cache
* Extracts and serves model metadata (RDF)
* Validates model metadata via `bioimageio.core`
* Forwards model testing and inference requests to the runtime deployment

This deployment is designed to run **without GPU** and remain lightweight.

**Dependencies**

```text
bioimageio.core==0.9.5
numpy==1.26.4
tqdm>=4.64.0
aiofiles>=23.0.0
```

## ModelRunner – Runtime Deployment (GPU)

**Responsibilities**

The runtime deployment is responsible for **model execution**. It:

* Runs model testing
* Executes inference
* Selects the appropriate backend (PyTorch, TensorFlow, ONNX, etc.) based on the model format

This deployment is GPU-backed and contains all heavy ML dependencies.

**Design Note – Universal Runtime**

All supported frameworks are installed together to form a **single universal runtime image**.
This avoids the need for model-type-specific runtimes and simplifies scheduling, deployment, and maintenance at the cost of a larger runtime environment.

**Dependencies**

```text
bioimageio.core==0.9.5
careamics==0.0.16
cellpose==3.1.1.2
numpy==1.26.4
onnxruntime==1.20.1
tensorflow==2.16.1
torch==2.5.1
torchvision==0.20.1
xarray==2025.1.2
```

Notes:

* This environment supports:

  * PyTorch models
  * TensorFlow models
  * ONNX models
  * Cellpose-based workflows
  * CAREamics-based workflows
* All versions are **strictly pinned** to ensure reproducibility and avoid runtime incompatibilities.

## Using the Service

**Service ID**: `bioimage-io/model-runner` · **Server**: `https://hypha.aicell.io`

```python
import asyncio
from hypha_rpc import connect_to_server

server = await connect_to_server({"server_url": "https://hypha.aicell.io", "token": token})
svc = await server.get_service("bioimage-io/model-runner")

# Search for models
models = await svc.search_models(keywords=["nuclei", "segmentation"], limit=10)

# Get model metadata
rdf = await svc.get_model_rdf(model_id="affable-shark")

# Run BioImage.IO compliance tests (async: test() returns a run id, then poll)
test_run_id = await svc.test(model_id="affable-shark")
while True:
    status = await svc.get_test_status(test_run_id=test_run_id)
    if status["completed_at"] is not None:   # terminal; queue_position == 0
        break
    await asyncio.sleep(2)
report = status["result"]                    # the test report, or {"error": ...} on failure
print(report["status"])                      # "passed", "valid-format", or "failed"

# Get model documentation (README) to verify domain compatibility
doc = await svc.get_model_documentation(model_id="affable-shark")
```

Inference is also asynchronous — `infer()` returns a `request_id`; poll `get_infer_status`
until the result is ready. `inputs` accepts an HTTPS URL or a file path from `get_upload_url`:

```python
import asyncio

request_id = await svc.infer(model_id="affable-shark", inputs="<url-or-upload-file-path>")
while True:
    status = await svc.get_infer_status(request_id=request_id)
    if status["completed_at"] is not None:
        break
    await asyncio.sleep(1)
result = status["result"]                    # dict keyed by output id, or {"error": ...}
```

### Key pitfalls

- **Output key varies by model**: Read `outputs[0].id` from the RDF — do not assume `"output0"`.
- **Domain mismatch**: Always call `get_model_documentation()` before running inference — many models are domain-specific (e.g. trained on histology, not fluorescence microscopy).
- **Search keywords are AND-matched**: If `"denoising"` returns few results, try synonyms: `"restoration"`, `"noise"`.
