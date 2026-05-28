<div align="center">
  <img src="https://raw.githubusercontent.com/aicell-lab/bioengine/main/docs/assets/bioengine-icon.svg" alt="BioEngine Logo" width="200"/>

  # BioEngine

  **The execution and adaptation layer between curated bioimage AI and scalable compute**

  [![GitHub](https://img.shields.io/badge/github-aicell--lab%2Fbioengine-black?logo=github)](https://github.com/aicell-lab/bioengine)
  [![Docker Image](https://img.shields.io/badge/docker-ghcr.io-blue)](https://github.com/orgs/aicell-lab/packages/container/package/bioengine)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
  [![Paper](https://img.shields.io/badge/paper-bioRxiv-red)](https://www.biorxiv.org/content/10.64898/2026.04.19.719496v1)

</div>

---

**[Use BioEngine now](#use-bioengine-now--no-setup-required)** · **[What it enables](#what-bioengine-enables)** · **[Deploy your own worker](#deploy-your-own-bioengine-worker)** · **[Developer docs](#developer-documentation)** · **[Paper](#paper)**

---

Foundation models and curated repositories have transformed bioimage AI, yet most biologists cannot readily run, adapt, or extend them on available hardware. **BioEngine fills this gap** — deployable on a laptop, workstation, or cluster. Scientists screen models, fine-tune from the browser, enable real-time smart microscopy, and deploy analysis applications, all by describing their goal to an AI agent.

> *"The user does not need to know how BioEngine works internally. They describe what they want and receive the result. Infrastructure management stays with whoever runs the hardware. Scientific focus returns to the biologist."*

---

## Use BioEngine now — no setup required

The community BioEngine instance runs at [BioImage.IO](https://bioimage.io). Use it through the web dashboard or any AI agent — no installation, no IT ticket.

### 🌐 Web interface

| | |
|--|--|
| [**BioEngine Dashboard**](https://bioimage.io/#/bioengine) | Browse all available BioEngine workers and open their management panels |
| [**Worker Dashboard**](https://bioimage.io/#/bioengine/worker?service_id=bioimage-io/bioengine-worker) | Deploy and manage apps on the public worker — cluster resources, running deployments, available apps |
| [**Deployment Wizard**](https://bioimage.io/#/bioengine/worker?service_id=bioimage-io/bioengine-worker) | Interactive setup guide for deploying your own BioEngine worker (Docker, SLURM, Kubernetes) |

### 🤖 AI agent

Load the BioEngine skill in any AI agent (Claude, GPT-4, etc.) by providing this link:

**[`https://bioimage.io/skills/bioengine/SKILL.md`](https://bioimage.io/skills/bioengine/SKILL.md)**

The skill gives the agent a complete, plain-text description of every available service — inputs, outputs, and usage examples. The agent selects the appropriate workflow and dispatches it to the GPU hardware. No command-line access, software installation, or specialist knowledge required.

---

## What BioEngine enables

| | Description |
|-|-------------|
| **Model screening** | Agent queries the BioImage Model Zoo, filters by domain compatibility, runs inference across candidates, and ranks by mAP |
| **Real-time inference** | Live images stream from the microscope to BioEngine; per-frame statistics return to the controlling agent for closed-loop smart microscopy |
| **Collaborative fine-tuning** | Browser-based annotation against foundation model pre-segmentations; fine-tuning triggered with one click; models published back to the BioImage Model Zoo |
| **Agent-built applications** | Agent generates deployment manifest, GPU workflow, and web UI from a plain-language prompt; the resulting app is immediately callable by other agents |

BioEngine exposes all capabilities through a **SKILL.md contract** — a plain-text file designed for general-purpose AI agents to acquire domain knowledge and invoke GPU services directly. Any agent that reads the contract can screen models, trigger fine-tuning, or deploy a custom application, without any BioEngine-specific programming.

---

## Deploy your own BioEngine worker

Facility managers and system administrators can deploy a private worker on any hardware. The [interactive deployment wizard](https://bioimage.io/#/bioengine/worker?service_id=bioimage-io/bioengine-worker) on BioImage.IO walks through the full setup for all supported modes.

| Mode | Use case |
|------|----------|
| **Single machine** | Workstation, development, small-scale inference |
| **SLURM / HPC** | Auto-scaling on institutional HPC clusters (Apptainer) |
| **Kubernetes** | Production deployment with KubeRay |

```bash
# Docker — single machine quickstart
git clone https://github.com/aicell-lab/bioengine.git
cd bioengine
mkdir -p .bioengine data
UID=$(id -u) GID=$(id -g) docker compose up
```

#### Building against a specific Ray version

The published `ghcr.io/aicell-lab/bioengine-worker:0.9.0` image ships **Ray 2.55.1**. When connecting to a managed Ray cluster (KubeRay, Anyscale, etc.) Ray Client enforces an **exact** version match between driver and cluster — so you may need an image built against the version your cluster runs. Two ways:

**Overlay on the published image (fast — pulls the image, rebuilds only the Ray install + env layers, ~1-2 min):**

```bash
docker build \
    --build-arg BIOENGINE_IMAGE=ghcr.io/aicell-lab/bioengine-worker:0.9.0 \
    --build-arg RAY_VERSION=2.54.1 \
    -f docker/worker-ray-overlay.Dockerfile \
    -t bioengine-worker:0.9.0-ray2.54.1 .
```

**Full build from source (~5-10 min):**

```bash
docker build \
    --build-arg RAY_VERSION=2.54.1 \
    -f docker/worker.Dockerfile \
    -t bioengine-worker:0.9.0-ray2.54.1 .
```

Both paths produce equivalent images. The active Ray version is exposed inside the image as `$BIOENGINE_RAY_VERSION` for diagnostics. The supported Ray range is `>=2.33.0, <3.0.0` (set in `pyproject.toml`).

See [Deployment Guide](docs/deployment-guide.md) for full instructions for all modes.

---

## Developer documentation

### Python SDK

```python
from hypha_rpc import connect_to_server

server = await connect_to_server({"server_url": "https://hypha.aicell.io", "token": token})
worker = await server.get_service("bioimage-io/bioengine-worker")

status = await worker.get_status()
app_id = await worker.deploy_app(
    artifact_id="bioimage-io/cellpose-finetuning",
    application_id="cellpose-finetuning",
)
```

### CLI

```bash
pip install "bioengine[cli] @ git+https://github.com/aicell-lab/bioengine.git"

bioengine call bioimage-io/bioengine-worker get_status
bioengine apps list --worker bioimage-io/bioengine-worker
```

### Worker service API

| Method | Admin | Description |
|--------|:-----:|-------------|
| `get_status()` | | Worker and cluster status |
| `deploy_app(artifact_id, ...)` | ✓ | Deploy an application |
| `stop_app(application_id)` | ✓ | Stop a running application |
| `get_app_status(application_ids)` | | Status of specific applications |
| `list_apps()` | ✓ | All deployed applications |
| `upload_app(files, workspace, hypha_token)` | | Create/update application artifact |
| `list_app_directories()` | ✓ | List app working directories on disk |
| `clear_app_directory(application_id)` | ✓ | Delete a stopped app's working directory |
| `run_code(code, ...)` | ✓ | Run Python in a Ray task |
| `list_datasets()` | | Available datasets |

### Applications

BioEngine applications are self-contained deployable units: a `manifest.yaml` + Python deployment code + optional web frontend. They can compose multiple AI models, wrap models with custom pre/post-processing, and expose arbitrary web UIs. Once deployed, they register as Hypha services immediately discoverable and callable by other AI agents.

**Reference apps:**
- [`apps/demo-app/`](apps/demo-app/) — minimal single-deployment app
- [`apps/cellpose-finetuning/`](apps/cellpose-finetuning/) — browser-based collaborative fine-tuning
- [`apps/model-runner/`](apps/model-runner/) — production BioImage Model Zoo inference

See [Applications Guide](docs/apps-guide.md).

### Architecture

```
┌────────────────────────────────────────┐
│            Hypha Server                │
│   (RPC, service discovery, artifacts)  │
└────────────┬───────────────────────────┘
             │ WebSocket / RPC
┌────────────▼───────────────────────────┐
│         BioEngineWorker                │
│  ┌─────────────────────────────────┐   │
│  │  Ray Cluster                    │   │
│  │  (SLURM / single / Kubernetes)  │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Applications Manager           │   │
│  │  (Ray Serve lifecycle +         │   │
│  │   artifact management)          │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Datasets Manager               │   │
│  │  (Zarr HTTP streaming)          │   │
│  └─────────────────────────────────┘   │
└────────────────────────────────────────┘
```

**Stack:** [Ray](https://www.ray.io/) + [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) · [Hypha](https://docs.amun.ai/#/) · [BioImage Model Zoo](https://bioimage.io/#/models)

### Development setup

```bash
git clone https://github.com/aicell-lab/bioengine.git
cd bioengine
pip install -e ".[worker,cli,dev]"
source .env   # loads HYPHA_TOKEN

python -m bioengine.worker \
    --mode single-machine \
    --head-num-gpus 1 \
    --workspace-dir ~/.bioengine \
    --debug

pytest tests/end_to_end/ -v
```

### Documentation

- [Applications Guide](docs/apps-guide.md) — build and deploy BioEngine applications
- [Datasets Guide](docs/datasets-guide.md) — share and stream large scientific datasets
- [Deployment Guide](docs/deployment-guide.md) — single-machine, Kubernetes, and SLURM setup

---

## Paper

> Mechtel N, Dettner Källander H, Cheng S, Zhang H, AI4Life Consortium, Ouyang W.
> **BioEngine: scalable execution and adaptation of bioimage AI through agent-readable interfaces.**
> *bioRxiv* (2026). https://doi.org/10.64898/2026.04.19.719496

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

BioEngine is built on [Ray](https://www.ray.io/), [Hypha](https://docs.amun.ai/#/), and [Zarr](https://zarr.readthedocs.io/).
Supported by the SciLifeLab & Wallenberg Data Driven Life Science Program (KAW 2020.0239), the Göran Gustafsson Prize (2317), AI4Life (EU Horizon Europe grant 101057970), and RI-SCALE (EU grant 101188168).
