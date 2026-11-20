---
name: bioengine-worker-clusters
description: How to deploy and validate the BioEngine worker on the two Kubernetes sites and on the SLURM site — the exact helm surface a worker change is allowed to touch, the release and namespace names, and the measured traps that have already cost a rollback. Read BEFORE any helm upgrade, kubectl rollout, or sbatch.
---

# BioEngine worker — cluster deploy & validation

Scope: the two non-local run modes of the worker, `external-cluster` and
`slurm`. Single-machine validation is not covered here.

Site coordinates — kubeconfig paths, context names, jump hosts, which
checkout holds each chart — are deployment-environment facts, not repo
facts, so they are deliberately not in this file. Get them from the
deploy environment you are working in. Everything below is true
regardless of which machine you reach the cluster from.

| Mode | Site | Namespace | Release |
|---|---|---|---|
| `external-cluster` | Kubernetes site A | `hypha` | `bioengine-worker-kth` |
| `external-cluster` | Kubernetes site B | `bioengine` | `bioengine-worker-denbi` |
| `slurm` | HPC site | — | `scripts/start_hpc_worker.sh` |

The two Kubernetes sites use **different namespaces**. Do not carry a
`-n <namespace>` flag from one to the other.

## The blast-radius rule — read this first

**Only three Helm releases are in scope for a worker change:
`bioengine-worker*`, `turn-server`, `turn-credentials`.**

Everything else in the shared namespace (the Hypha server itself, redis,
the postgres releases, and the various sibling apps) belongs to other
owners. Do not upgrade, roll, scale, or "reconcile" them — not even when
one of them looks like the cause. Problems with the cluster itself (node
and GPU faults, wedged Ray workers, KubeRay health) are not worker
problems and route to whoever owns the cluster.

`kuberay-operator` and the `raycluster-*` releases are **out of scope**
too, despite the worker depending on them.

## Kubernetes, both sites

- **A namespace you cannot see is reported as `Forbidden`, not
  `NotFound`.** So `helm list -n <wrong-namespace>` returning Forbidden
  reads exactly like revoked RBAC, and is not. Before concluding access
  was taken away, check `kubectl get ns -o name` and `kubectl auth can-i
  list pods -n <namespace>`.
- `--workspace-dir=/home/bioengine` in the worker args is a path **inside
  the container**, not a namespace.
- Ray runs on its own pods in `external-cluster` mode, so the worker pod
  holds no raylets. Its own resource footprint is small, and Ray's
  requests and limits are not this chart's concern.
- The chart carries a `values-dev.yaml` alongside `values.yaml`. Use the
  dev values for a `-devN` image, never the prod values file.
- The two sites are kept in **lock-step on the worker image tag**. If you
  bump one, bump the other in the same piece of work, or say explicitly
  why not.
- If a cluster credential has expired, **ask for a refresh**; do not
  attempt to re-auth.

## SLURM

- Entry point is `scripts/start_hpc_worker.sh`. It derives the image tag
  from `pyproject.toml` — or from `main` on GitHub when run outside a
  clone — so **the image tag follows the checkout's version**, and a stale
  checkout silently validates a stale worker.
- Apptainer, not Docker. SIF builds break where the host sets
  `yama.ptrace_scope=2`, which blocks the proot/mksquashfs path. The
  working path is `apptainer build --sandbox` to a directory image; the
  script accepts a sandbox directory as well as a `.sif`.
- The worker is spun up **on demand**, not persistently.

## Dos

- **Diff the LIVE args against the chart before you roll.** Revisions
  have been applied out-of-band, so `values.yaml` has at times been
  *behind* the running pod — a `helm upgrade` then silently reverts things
  nobody meant to revert. Compare `kubectl get deploy -n <ns> <name> -o
  yaml` against the rendered chart first, and reconcile the file before
  using it.
- **Read the `values.yaml` comments before changing a value.** Both charts
  carry long, dated, measured rationales (liveness probe design, resource
  requests, why `--token` was removed). They encode past incidents. If you
  are about to contradict one, say so out loud.
- **Keep `image.tag` and the startup args as one change** where the
  comments say they are coupled.
- Record which helm revision you created, so the next reader can tell
  chart state from live state.

## Don'ts

- **Never put a token on the command line.** `$(VAR)` is expanded by
  Kubernetes at render time and lands in `/proc/<pid>/cmdline`, which is
  world-readable. `HYPHA_TOKEN` is already injected as a pod env var from
  the secret, and 0.16.37+ resolves `--token-file > --token >
  HYPHA_TOKEN`. A mounted secret *file* does not work either: the pod runs
  `runAsUser: 65534` with no `fsGroup`, secret files are root-owned mode
  0400, and an unreadable `--token-file` is a hard startup error.
- **Do not assume `--startup-applications` protects an app.** It does the
  opposite of what the name suggests on restart: a worker that restarts
  re-adopts live Ray Serve apps at their *live* version, but any app
  listed in `--startup-applications` is force-**reset** to the version
  pinned there. A pinned-but-stale version has already rolled an app back
  six published releases. Before a roll, read the live versions from
  `get_app_status` and make the pins match.
- **Do not infer `disable_gpu` from `num_gpus`.** They are independent:
  `gpu_enabled` is permission, `num_gpus` is what the app's manifest
  requests. Read `gpu_enabled` from `get_app_status`.
- **Do not redeploy an app by RPC without passing `hypha_token`.** An app
  recovered from Serve state is stored with `hypha_token: None`, so an RPC
  redeploy that omits it drops a credential the running replica still
  holds. For `model-runner` this fails *silently* — it falls back to
  anonymous read-only and merely stops publishing test reports while still
  reporting RUNNING.
- **Do not `stop_app`/`delete_app` a failed deployment while diagnosing.**
  `get_app_status(application_ids=[...])` → `deployments.<name>.logs` is
  the primary signal, and `stop_app` discards it.
- **`RUNNING` is not `working`.** Check the replica logs for the thing you
  changed, not the status field.

## Reporting back

Each validation writes down, per site: the image tag that actually ran,
the helm revision or job id, the pod or replica name, and the specific log
line or RPC result that proves the behaviour. "Deployed and healthy" is
not evidence.
