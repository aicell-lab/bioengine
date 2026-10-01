# BioEngine Worker — Deployment Guide

BioEngine supports three deployment modes. The easiest way to generate deployment commands for your environment is the **[BioEngine Dashboard](https://bioimage.io/#/bioengine)**, which provides an interactive configuration wizard.

---

## Prerequisites (all modes)

- A **Hypha account** — sign in at [hypha.aicell.io](https://hypha.aicell.io) to get a token
- The worker registers itself as a Hypha service on startup; your workspace and service ID are printed in the logs

---

## Supplying the token

The worker takes its Hypha token from the first of these that is set:

1. `--token-file PATH` — the file's contents, whitespace-stripped
2. `--token VALUE`
3. the `HYPHA_TOKEN` environment variable

With none of them set the worker prompts for an interactive login, which is fine on a workstation and useless in a container.

**Prefer `--token-file` or `HYPHA_TOKEN` for anything unattended.** A token passed with `--token` lands in `/proc/<pid>/cmdline`, which is world-readable: any user who can run a process next to the worker can read it, and any routine "what is this running with?" diagnostic copies it into its own output. `/proc/<pid>/environ` is readable only by the process owner, and a mounted file is readable only by whoever the mount permits. The worker prints a warning to stderr when `--token` is used.

On Kubernetes, mount the Secret and point `--token-file` at it rather than expanding it into an argument:

```yaml
spec:
  securityContext:
    fsGroup: 65534            # must match the container's runAsGroup
  containers:
    - name: bioengine-worker
      securityContext:
        runAsNonRoot: true
        runAsUser: 65534
        runAsGroup: 65534
      args:
        - "--token-file=/var/run/secrets/hypha/token"
      volumeMounts:
        - name: hypha-token
          mountPath: /var/run/secrets/hypha
          readOnly: true
  volumes:
    - name: hypha-token
      secret:
        secretName: bioengine-worker
        defaultMode: 0440
        items:
          - key: token
            path: token
```

**Set the mode and `fsGroup` together, or the worker will not start.** A Secret volume is mounted `0644` by default — world-readable inside the container, which is the same exposure this is meant to remove. But kubelet writes those files as `root:root` unless `fsGroup` is set, so tightening the mode without it locks out the very process that needs to read the file, and the worker then exits with `Could not read --token-file`. Two working combinations:

| Container runs as | Mode | `fsGroup` | Who can read |
|---|---|---|---|
| non-root (`runAsUser: 65534`) | `0440` | `65534` | root and the worker's group |
| root | `0400` | — | root only |

The worker image sets no `USER`, so a plain `docker run` is root and `0400` works. Production Kubernetes deployments generally are not root — the KTH chart runs `runAsNonRoot: true, runAsUser: 65534` — which is why `0400` alone is the wrong default to copy.

Substitute your own secret name and key. `items` maps a key in the Secret to a filename in the mount, so a Secret holding the token under `HYPHA_TOKEN` needs `- key: HYPHA_TOKEN`. Get it wrong and kubelet refuses to set the volume up at all (`FailedMount`, `references non-existent secret key`) and the container never starts; mark the volume `optional: true` and the file is simply absent, at which point the worker exits at startup with `Could not read --token-file`.

`--token=$(HYPHA_TOKEN)` is the shape to avoid: Kubernetes expands it at render time, so the Deployment spec keeps only the placeholder and looks clean under `kubectl get deploy -o yaml` while the live process carries the value.

---

## Who can control the worker

`--admin-users` takes a space-separated list of emails or user IDs and defaults to the account whose token started the worker. Admins can perform every admin operation on the worker, in every deployment mode.

**The users you name in `--admin-users` can never lose admin permissions on a running worker.** No other admin can remove them, and neither can they themselves. This closes a lockout that had no recovery: the runtime admin list is persisted to `<workspace-dir>/admin_users.json` and that file overrides the startup flag, so once a starting user had been removed, editing `--admin-users` and restarting did not bring them back. If the persisted file is missing a starting user — because an older worker allowed the removal, or because it was edited by hand — the worker restores them at startup and logs that it did.

> **Demoting a starting user takes two steps, and the first one alone looks like it worked.** Drop them from `--admin-users` and restart: that stops them being a starting user, but does **not** revoke their access, because the persisted admin list overrides the startup flag and still lists them. Then call `remove_admin_user` for them, which now succeeds. If you stop after the restart they remain a full admin.

> **`--admin-users '*'` is no longer honoured.** A `*` entry is dropped at startup with a warning, whether it came from the flag or from the persisted admin list. It used to mean remote code execution open to the internet: the worker's Hypha service is registered with public visibility, a service's authorization gates invocation rather than discovery, and almost every admin operation is gated on the admin list — so `*` made **any caller that could reach the Hypha server, including an unauthenticated anonymous one, a full admin of the worker**. They could run arbitrary Python on the deployment through `run_code`, `deploy_app` or `upload_app`, with whatever filesystem, credentials and network access its Ray cluster is given, and destroy it through `stop_worker`, `stop_all_apps` or `delete_app`.
>
> **Migrating off it:** the account whose token starts the worker is always an admin, so a worker that relied on `*` keeps working for its operator and stops working for everyone else. Name the others in `--admin-users`, add them on the running worker with `add_admin_user`, or turn on `--enable-access-requests` and let them ask.
>
> **Redeploy any app that was deployed while the wildcard was in effect.** The admin list is injected into each app's `authorized_users` once, at deploy time, so an app deployed under a wildcard admin list has `*` baked into its own rules — and a rule that already contains `*` is left alone by later injection, including the re-injection that happens when the worker recovers running apps after a restart. Those apps therefore stay callable by anyone after this upgrade, even though the worker itself no longer honours the wildcard. Upgrading the worker does not fix them; redeploying them does.

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

Editing the admin list still requires a caller *named* in it. `add_admin_user`, `remove_admin_user` and `resolve_access_request` are checked with the wildcard refused, so a `*` entry in the admin list would not authorize them even before it was dropped — they are the methods that widen or narrow the authorized set itself, and a caller who is only an admin by wildcard must not be able to make that permanent.

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
  --gpus=all $(for d in /dev/nvidia* /dev/nvidia-caps/*; do [ -c "$d" ] && printf -- '--device=%s ' "$d"; done) \
  -v $HOME/.bioengine:/.bioengine \
  ghcr.io/aicell-lab/bioengine-worker:latest \
  python -m bioengine.worker \
    --mode single-machine \
    --head-num-cpus 4 \
    --head-num-gpus 1
```

**GPU support:**
- Docker: `--gpus=all` (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)) **plus an explicit `--device` for each NVIDIA node** — see below
- Podman: use `--device nvidia.com/gpu=all` instead; its CDI reference already names the devices in the spec
- No GPU: omit the GPU flag; CPU-only inference still works for most models

> **Why `--gpus=all` is not enough on its own.** It records only a *DeviceRequest*: the nvidia-container-runtime prestart hook injects the device nodes and patches the device cgroup afterwards, so the container's OCI spec never names them. Anything that makes runc re-apply the cgroup — classically a `systemctl daemon-reload`, including ones fired by package managers — rebuilds the allowlist **from that spec** and silently drops the hook's patch. The container keeps running, and applications already holding the GPU keep working, so nothing looks wrong until the next process that needs a fresh CUDA context fails with `CUDA_ERROR_NO_DEVICE`. Listing the nodes with `--device` puts them where the rebuild looks. Preconditions are cgroup v2 with Docker's systemd cgroup driver; check yours with `docker info --format '{{.CgroupDriver}} {{.CgroupVersion}}'`.
>
> The `--device` flags are **additional**, never a replacement. The driver libraries come from the same hook, so a container given the nodes without `--gpus` fails at `libcuda.so.1: cannot open shared object file`.
>
> `bioengine worker start` adds these for you on Docker; the expansion above is for hand-written commands.

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
| `--token-file` | — | File holding the Hypha authentication token — see [Supplying the token](#supplying-the-token) |
| `--token` | prompt | Hypha authentication token, visible in the process table — see [Supplying the token](#supplying-the-token) |
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
      securityContext:
        fsGroup: 65534           # lets runAsGroup read the mounted Secret
      containers:
        - name: bioengine-worker
          image: ghcr.io/aicell-lab/bioengine-worker:latest
          securityContext:
            runAsNonRoot: true
            runAsUser: 65534
            runAsGroup: 65534
          args:
            - python
            - -m
            - bioengine.worker
            - --mode=external-cluster
            - --connection-address=ray://raycluster-head-svc:10001
            - --token-file=/var/run/secrets/hypha/token
          resources:
            requests: { memory: "2Gi", cpu: "1" }
            limits:   { memory: "4Gi", cpu: "2" }
          volumeMounts:
            - name: bioengine-storage
              mountPath: /.bioengine
            - name: hypha-token
              mountPath: /var/run/secrets/hypha
              readOnly: true
      volumes:
        - name: bioengine-storage
          persistentVolumeClaim:
            claimName: bioengine-pvc  # 10Gi PVC
        - name: hypha-token
          secret:
            secretName: bioengine-worker
            defaultMode: 0440
            items:
              - key: token
                path: token
```

See [Supplying the token](#supplying-the-token) for why the token is mounted rather than passed as `--token=$(HYPHA_TOKEN)`.

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | — | Must be `external-cluster` |
| `--connection-address` | — | Ray cluster address, e.g. `ray://host:10001` |
| `--client-server-port` | 10001 | Ray client connection port |
| `--serve-port` | 8000 | Ray Serve HTTP endpoint port |
| `--workspace` | auto | Hypha workspace name |
| `--server-url` | `https://hypha.aicell.io` | Hypha server URL |
| `--token-file` | — | File holding the Hypha authentication token — see [Supplying the token](#supplying-the-token) |
| `--token` | prompt | Hypha authentication token, visible in the process table — see [Supplying the token](#supplying-the-token) |

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
