# BioEngine Worker — Deployment Guide

BioEngine supports three deployment modes. The easiest way to generate deployment commands for your environment is the **[BioEngine Dashboard](https://bioimage.io/#/bioengine)**, which provides an interactive configuration wizard.

---

## Prerequisites (all modes)

- A **Hypha account** — sign in at [hypha.aicell.io](https://hypha.aicell.io) to get a token
- The worker registers itself as a Hypha service on startup; your workspace and service ID are printed in the logs

---

## Who can control the worker

`--admin-users` takes a space-separated list of emails or user IDs and defaults to the account whose token started the worker. Admins can perform every admin operation on the worker, in every deployment mode.

**The users you name in `--admin-users` can never lose admin permissions on a running worker.** No other admin can remove them, and neither can they themselves. This closes a lockout that had no recovery: the runtime admin list is persisted to `<workspace-dir>/admin_users.json` and that file overrides the startup flag, so once a starting user had been removed, editing `--admin-users` and restarting did not bring them back. To demote a starting user, drop them from `--admin-users` and restart. If the persisted file is missing a starting user — because an older worker allowed the removal, or because it was edited by hand — the worker restores them at startup and logs that it did.

> **`--admin-users '*'` is no longer honoured.** A `*` entry is dropped at startup with a warning, whether it came from the flag or from the persisted admin list. It used to mean remote code execution open to the internet: the worker's Hypha service is registered with public visibility, a service's authorization gates invocation rather than discovery, and almost every admin operation is gated on the admin list — so `*` made **any caller that could reach the Hypha server, including an unauthenticated anonymous one, a full admin of the worker**. They could run arbitrary Python on the deployment through `run_code`, `deploy_app` or `upload_app`, with whatever filesystem, credentials and network access its Ray cluster is given, and destroy it through `stop_worker`, `stop_all_apps` or `delete_app`.
>
> **Migrating off it:** the account whose token starts the worker is always an admin, so a worker that relied on `*` keeps working for its operator and stops working for everyone else. Name the others in `--admin-users`, add them on the running worker with `add_admin_user`, or turn on `--enable-access-requests` and let them ask.

### Letting users ask for access

`--enable-access-requests` is off by default. Turning it on adds two public methods to the worker service — `request_admin_access`, which any logged-in caller may invoke, and `get_admin_access_request`, which returns only the caller's own request — plus `list_access_requests` and `resolve_access_request` for admins. With the flag off none of the four is registered, so there is no request surface at all.

A request is keyed on the caller's email address, and each account gets exactly one. A second request is refused rather than queued or replacing the first, so a decision already taken cannot be reset by asking again. Requests are persisted next to the admin list at `<workspace-dir>/access_requests.json`, so a pod restart does not silently discard them.

Admins resolve a request with `resolve_access_request(user, decision)`:

| Decision | Effect |
|---|---|
| `grant` | Adds the requester to the admin users, immediately and across restarts. A granted admin is *not* a starting user, so they can be removed again later. |
| `deny` | Records the refusal and leaves it in place. The requester cannot re-file. |
| `clear` | Deletes the record, so the requester may ask again. This is how a denial is lifted. |

Unauthenticated callers are refused: Hypha reports no email address for them and mints a fresh random user id per anonymous connection, so an anonymous request names no account that could be granted. Note that one-request-per-account bounds requests per *account*, not per person — anyone able to register additional Hypha accounts can file additional requests. Treat the flag as a convenience for a known user community, not as a hardened public endpoint.

Editing the admin list still requires a caller *named* in it: a caller covered only by a `*` entry in a deployed app's `authorized_users` cannot call `add_admin_user`, `remove_admin_user` or `resolve_access_request`.

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
| `--admin-users` | current user | Space-separated emails; `*` is not honoured — see [Who can control the worker](#who-can-control-the-worker) |
| `--enable-access-requests` | off | Let non-admins ask to become admins — see [Letting users ask for access](#letting-users-ask-for-access) |
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
