# BioEngine Worker — Deployment Guide

BioEngine supports three deployment modes. The easiest way to generate deployment commands for your environment is the **[BioEngine Dashboard](https://bioimage.io/#/bioengine)**, which provides an interactive configuration wizard.

---

## Prerequisites (all modes)

- A **Hypha account** — sign in at [hypha.aicell.io](https://hypha.aicell.io) to get a token
- The worker registers itself as a Hypha service on startup; your workspace and service ID are printed in the logs

---

## Who can control the worker

`--admin-users` takes a space-separated list of emails or user IDs and defaults to the account whose token started the worker. Admins can perform every admin operation on the worker, in every deployment mode.

> **`--admin-users '*'` is remote code execution open to the internet.** The worker's Hypha service is registered with public visibility, and a service's authorization gates invocation, not discovery. Almost every admin operation is gated on the admin list with the wildcard honoured, so with `*` in the list **any caller that can reach the Hypha server — including an unauthenticated, anonymous one — is a full admin of your worker**. They can run arbitrary Python on the deployment through `run_code`, `deploy_app` or `upload_app`, with whatever filesystem, credentials and network access its Ray cluster is given — in single-machine mode that is the host you started the worker on, as the user who started it. They can also destroy it through `stop_worker`, `stop_all_apps` or `delete_app`. The wildcard is not "skip maintaining an admin list"; it is "this deployment runs untrusted code from strangers". Use named emails unless that is genuinely what you want. The worker logs a warning at startup whenever the wildcard is in effect.

The one exception is editing the admin list itself: a caller who is only covered by `*` cannot call `add_admin_user` or `remove_admin_user`, so a stranger cannot make the grant permanent or lock you out.

---

## Mode 1: Single Machine

Runs a local Ray cluster on one machine. Good for workstations, development, and small-scale analysis.

### The `bioengine` CLI (recommended)

```bash
pip install "bioengine[cli]"

bioengine worker start -- \
  --mode single-machine \
  --head-num-cpus 4 \
  --head-num-gpus 1
```

This runs the worker image in a container. It picks the first of `docker`, `podman` and `apptainer` on your `PATH`, mounts `~/.bioengine` as the workspace, passes `HYPHA_TOKEN` through the environment, and pins the image tag to the installed `bioengine` version. Everything after `--` is forwarded verbatim to `python -m bioengine.worker` inside the container — see `bioengine worker start -- --help` for the full list.

```bash
bioengine worker start --dry-run -- --mode single-machine  # print the command, run nothing
bioengine worker start -d -- --mode single-machine          # run in the background
bioengine worker logs -f
bioengine worker stop
```

| Option | Default | Description |
|---|---|---|
| `--runtime` | `auto` | `docker`, `podman`, `apptainer`, or `native` to run the worker in the current environment instead of a container |
| `--image` | `ghcr.io/aicell-lab/bioengine-worker:<version>` | Worker image |
| `--workspace-dir` | `~/.bioengine` | Host directory mounted at `/.bioengine` |
| `--name` | `bioengine-worker` | Container name. Starting a second worker while this name is taken is refused — give it a different `--name` |
| `--gpus` / `--no-gpus` | on when `nvidia-smi` is present | Whether to give the container GPUs |
| `--shm-size` | `8g` | Shared memory size |
| `--detach` / `-d` | off | Run in the background |
| `--dry-run` | off | Print the command instead of running it |

`--gpus` only decides whether the *container* sees GPUs; tell Ray to use them with `--head-num-gpus` after the `--`.

### Running the container directly

The CLI is a thin wrapper — the underlying commands work on their own:

```bash
docker run --rm -it \
  --user $(id -u):$(id -g) \
  --shm-size=8g \
  --gpus=all \
  -v $HOME/.bioengine:/.bioengine \
  ghcr.io/aicell-lab/bioengine-worker:latest \
  python -m bioengine.worker \
    --mode single-machine \
    --head-num-cpus 4 \
    --head-num-gpus 1
```

**GPU support:**
- Docker: add `--gpus=all` (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- Podman: use `--device nvidia.com/gpu=all` instead
- No GPU: omit the GPU flag; CPU-only inference still works for most models

**Apptainer / Singularity** (HPC login nodes without Docker):

```bash
apptainer exec \
  --nv \
  --bind $HOME/.bioengine:/.bioengine \
  docker://ghcr.io/aicell-lab/bioengine-worker:latest \
  python -m bioengine.worker \
    --mode single-machine \
    --head-num-cpus 4 \
    --head-num-gpus 1
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | — | Must be `single-machine` |
| `--head-num-cpus` | 2 | CPU cores for the Ray head node |
| `--head-num-gpus` | 0 | GPUs for the Ray head node |
| `--workspace` | auto | Hypha workspace name (auto-detected from token) |
| `--server-url` | `https://hypha.aicell.io` | Hypha server URL |
| `--token` | prompt | Hypha authentication token |
| `--admin-users` | current user | Space-separated emails, or `*` for all — see [Who can control the worker](#who-can-control-the-worker) |
| `--client-id` | auto | Unique service identifier |

The workspace directory defaults to `~/.bioengine` and is mounted into the container at `/.bioengine`.

---

## Mode 2: External Cluster (Kubernetes / KubeRay)

Connects to a pre-existing Ray cluster instead of creating one. Suited for Kubernetes environments managed with [KubeRay](https://ray-project.github.io/kuberay/).

### 1. Deploy a Ray cluster with KubeRay

```bash
helm install kuberay-operator kuberay/kuberay-operator
kubectl apply -f raycluster.yaml
```

### 2. Run the BioEngine worker

```bash
python -m bioengine.worker \
  --mode external-cluster \
  --connection-address ray://raycluster-head-svc:10001
```

Or as a Kubernetes Deployment (example):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bioengine-worker
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: bioengine-worker
          image: ghcr.io/aicell-lab/bioengine-worker:latest
          args:
            - python
            - -m
            - bioengine.worker
            - --mode=external-cluster
            - --connection-address=ray://raycluster-head-svc:10001
          resources:
            requests: { memory: "2Gi", cpu: "1" }
            limits:   { memory: "4Gi", cpu: "2" }
          volumeMounts:
            - name: bioengine-storage
              mountPath: /.bioengine
      volumes:
        - name: bioengine-storage
          persistentVolumeClaim:
            claimName: bioengine-pvc  # 10Gi PVC
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | — | Must be `external-cluster` |
| `--connection-address` | — | Ray cluster address, e.g. `ray://host:10001` |
| `--client-server-port` | 10001 | Ray client connection port |
| `--serve-port` | 8000 | Ray Serve HTTP endpoint port |
| `--workspace` | auto | Hypha workspace name |
| `--server-url` | `https://hypha.aicell.io` | Hypha server URL |
| `--token` | prompt | Hypha authentication token |

---

## Mode 3: SLURM / HPC

> **Note:** SLURM support is under active development. The information below reflects the current state; details may change.

Runs BioEngine on an HPC cluster managed by SLURM. The worker submits Ray worker jobs via `sbatch` and auto-scales based on demand.

### Quick start (from a login node)

```bash
bash <(curl -s https://raw.githubusercontent.com/aicell-lab/bioengine/refs/heads/main/scripts/start_hpc_worker.sh)
```

The script handles container image download (via Singularity/Apptainer), Ray cluster setup, and SLURM job management automatically.

### Prerequisites

- SLURM commands (`sbatch`, `squeue`, `scancel`) available in `$PATH`
- Singularity or Apptainer installed on compute nodes
- A shared filesystem accessible from all nodes
- Network access from compute nodes (to pull the container image on first run)

### Key SLURM parameters

| Parameter | Description |
|-----------|-------------|
| `--workspace-dir PATH` | Shared filesystem path used by all nodes |
| `--image IMAGE` | Container image for worker jobs |
| `--default-num-gpus N` | GPUs requested per SLURM job |
| `--default-num-cpus N` | CPUs requested per SLURM job |
| `--default-mem-in-gb-per-cpu GB` | Memory per CPU for SLURM jobs |
| `--default-time-limit HH:MM:SS` | Time limit per worker job |
| `--min-workers N` | Minimum number of worker nodes |
| `--max-workers N` | Maximum number of worker nodes |
| `--further-slurm-args ...` | Extra arguments passed to `sbatch` |

Monitor submitted jobs with `squeue -u $USER`.

---

## After deployment

Once the worker is running, it prints its **service ID** (e.g. `ws-user-abc123/bioengine-worker`). Use this to connect from Python:

```python
from hypha_rpc import connect_to_server, login

token = await login({"server_url": "https://hypha.aicell.io"})
server = await connect_to_server({"server_url": "https://hypha.aicell.io", "token": token})
worker = await server.get_service("your-workspace/bioengine-worker")

status = await worker.get_status()
print(status)
```

You can also manage your worker from the **[BioEngine Dashboard](https://bioimage.io/#/bioengine)**.
