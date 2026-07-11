---
name: bioengine-maintainer
description: Reference material for agents helping maintain the BioEngine codebase — architecture map, dev image testing workflow, cleanup rules, Hypha platform invariants, PR description style. Loaded on demand by Claude Code in addition to the always-on rules in CLAUDE.md.
---

# BioEngine maintainer skill

Reference material for agents helping maintain `aicell-lab/bioengine`. The always-on hard rules (sync-first, PR-vs-push gates, version bumps, the `application_id` vs `artifact_id` distinction, `hypha_token` parameter, cleanup) live in `CLAUDE.md` at the repo root. This skill holds the slower-moving reference content.

For the **user-facing** agent skill (for agents *using* BioEngine to deploy apps) see `../bioimage.io/public/skills/bioengine/SKILL.md` (cloned as a sibling of this repo) or the published copy at <https://bioimage.io/skills/bioengine/SKILL.md>.

## Repository ecosystem

BioEngine spans three GitHub repositories:

| Repository | Role | Status |
|---|---|---|
| `aicell-lab/bioengine` | **This repo** — Python package, worker, CLI, apps, Docker image. Canonical implementation. Previously named `aicell-lab/bioengine-worker`. | Active |
| `bioimage-io/bioimage.io` | BioImage.IO website — UIs (worker dashboard, deployment wizard, app manager, `BioEngineGuide.tsx`) and agent skills under `public/skills/bioengine/`. | Active |
| `bioimage-io/bioengine` | Original Triton-based BioEngine. All links redirect here. | Archived |

The website repo is expected as a sibling directory:

```
workspace/
├── bioengine/          ← this repo
└── bioimage.io/        ← bioimage-io/bioimage.io
```

Also added as a git remote `bioimage-io`:

```bash
git remote add bioimage-io git@github.com:bioimage-io/bioimage.io.git
```

What lives where:

- **Python package, worker, CLI, apps, Docker image** → this repo
- **Worker dashboard, app manager, deployment wizard, agent skills** → `../bioimage.io/` (specifically `src/components/bioengine/` and `public/skills/bioengine/`)
- **Production Hypha service** → workspace `bioimage-io` on <https://hypha.aicell.io>
- **BioEngine UI entry points**: <https://bioimage.io/#/bioengine> and <https://bioimage.io/#/bioengine/worker?service_id=...>

## Architecture

```
┌────────────────────────────────────────┐
│            Hypha Server                │
│   (RPC, service discovery, artifacts)  │
└────────────┬───────────────────────────┘
             │ WebSocket / RPC
┌────────────▼───────────────────────────┐
│         BioEngineWorker                │  ← bioengine/worker/worker.py
│  ┌─────────────────────────────────┐   │
│  │  RayCluster                     │   │  ← bioengine/cluster/ray_cluster.py
│  │  (SLURM / single / external)    │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  AppsManager                    │   │  ← bioengine/apps/manager.py
│  │  (Ray Serve lifecycle +         │   │
│  │   artifact management)          │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  BioEngineDatasets              │   │  ← bioengine/datasets/
│  │  (Zarr HTTP streaming)          │   │
│  └─────────────────────────────────┘   │
└────────────────────────────────────────┘
```

| Component | File | Responsibility |
|-----------|------|----------------|
| `BioEngineWorker` | `bioengine/worker/worker.py` | Main orchestrator; Hypha service registration |
| `AppsManager` | `bioengine/apps/manager.py` | Application lifecycle (deploy/stop/status) |
| `AppBuilder` | `bioengine/apps/builder.py` | Build Ray Serve apps from artifacts |
| `RayCluster` | `bioengine/cluster/ray_cluster.py` | Ray cluster lifecycle (SLURM / single-machine / external) |
| `BioEngineDatasets` | `bioengine/datasets/datasets.py` | Zarr dataset streaming |
| Artifact utilities | `bioengine/utils/artifact_utils.py` | Hypha artifact CRUD helpers |

The end-to-end deploy_app flow is documented step-by-step in `docs/deploy_app_flow.md`.

## Key file locations

- `bioengine/cli/` — `bioengine` CLI; entry point `bioengine.cli.cli:main`
- `bioengine/utils/artifact_utils.py` — Hypha artifact CRUD helpers
- `bioengine/apps/manager.py` — `deploy_app`, `upload_app`, lifecycle
- `bioengine/apps/builder.py` — `build()` constructs Ray Serve app from artifact
- `bioengine/_app/bootstrap.py` — `introspect_app_in_ray_task`, `build_and_run_application`
- `bioengine/_app/replica_init.py` — replica-side source materialisation
- `bioengine/_app/cache.py` — `PipelineCache` + `bioengine.cache` module (`@bioengine.cached` backend)
- `bioengine/_app/decorators.py` — `@bioengine.app` / `@bioengine.method` / `@bioengine.cached` / lifecycle hooks + `_scan_class`
- `apps/demo-app/` — reference single-deployment app (keep at version 1.0.0)
- `apps/composition-demo/` — reference multi-deployment app (keep at version 1.0.0)
- `apps/model-runner/` — production model-runner
- `apps/cellpose-finetuning/` — Cellpose fine-tuning
- `pyproject.toml` — package version + deps
- `../bioimage.io/public/skills/bioengine/` — user-facing agent skill (sibling repo)

## Development setup

```bash
conda activate bioengine
pip install -e ".[worker,cli,dev]"
```

`HYPHA_TOKEN`: get one from <https://hypha.aicell.io>, then `echo "HYPHA_TOKEN=<your-token>" > .env && source .env`.

### Run locally

```bash
python -m bioengine.worker \
    --mode single-machine \
    --head-num-gpus 1 \
    --head-num-cpus 4 \
    --workspace-dir ~/.bioengine \
    --debug
```

Local artifact development: `export BIOENGINE_LOCAL_ARTIFACT_PATH=/path/to/bioengine/tests`.

### Run tests

```bash
pytest tests/end_to_end/ -v
```

Test organisation:
- `tests/end_to_end/` — integration tests for the core worker
- `tests/apps/` — per-app tests, one subdirectory per app (`tests/apps/cellpose/`, …)

## Application manifest (v0.11 / `format_version: 0.6.0`)

Every BioEngine app is a folder containing `manifest.yaml` and the app's Python files at the root. The whole folder is uploaded to the Hypha artifact; the worker ships the same root via `runtime_env.py_modules`, excluding non-Python content.

```
my-app/                       ← artifact root + Python module directory
├── manifest.yaml
├── README.md                 ← stays in artifact (excluded from py_modules)
├── frontend/index.html       ← Hypha hosts statically (excluded from py_modules)
└── deployment.py             ← @bioengine.app class
```

Required manifest fields:

```yaml
name: My Application
id: my-application                       # Unique lowercase ID (hyphens only)
id_emoji: "🔬"
description: "..."
type: ray-serve
format_version: 0.6.0                    # v0.11 gate
entry: deployment:MyApp                  # module:Class
```

Optional fields: `frontend_entry`, `version`, `authorized_users` (`['*']` or `{method_name: [users], "*": [users]}`), `authors`, `license`, `documentation`, `tutorial`, `tags`.

Multi-file apps put files at the artifact root and use plain top-level imports (`apps/composition-demo/` is the reference).

**The decorator-module import rule.** Modules containing `@bioengine.app` (and what they transitively import at top level) MUST be importable with just `bioengine[worker]` and the standard library. The worker introspects them in a clean baseline `runtime_env`. Heavy deps go in `@bioengine.app(pip=…)`; their imports live inside method bodies or in helper modules the decorator file lazy-imports. See `apps/composition-demo/numpy_ops.py` + `runtimes/b.py`.

Local validation:

```bash
bioengine apps validate ./my-app
```

The CLI runs the same validator the worker uses — legacy manifests fail fast with a migration hint without a round-trip.

## Deployment modes

| Mode | Description |
|------|-------------|
| `single-machine` | Local Ray cluster (dev, small-scale) |
| `external-cluster` | Connect to existing Ray cluster (e.g. KubeRay) |
| `slurm` | Auto-scaling via SLURM job scheduler (HPC) |

## Production deployments (external-cluster mode) — example targets

**Production BioEngine only runs in external-cluster mode.** The active production deployments as of `0.11.22` are on KTH (SciLifeLab 2) and deNBI. The `bioimage.io` UI pins its canonical worker to KTH. NOT every maintainer will have credentials to these clusters — the notes below describe the shape so a maintainer with access can act, and let a maintainer without access hand off cleanly.

Different maintainers may run production on entirely different clusters — the pattern (KubeRay + helm chart with `values.yaml` `image.tag` + startup-application pins) is what generalizes; the specific repos below are the current KTH/deNBI examples.

| Target | Compute backend | Worker service ID pattern | Helm chart | Notes |
|---|---|---|---|---|
| **KTH (SciLifeLab 2 K8s)** | KubeRay on `scilifelab-2-dev` cluster, `hypha` namespace + `ray-cluster` namespace | `bioimage-io/bioengine-worker-kth-<hash>:bioengine-worker` | `aicell-lab/kth-k8s` repo, chart at `bioengine-worker/`. `values.yaml` `image.tag` + startup-app pins; `Chart.yaml` `appVersion` mirrors the bioengine version. `helm upgrade bioengine-worker-kth <chart> -f values.yaml -n hypha` | Split PVCs (hypha `bioengine-pvc` 10 GiB, ray-cluster `bioengine-pvc` 500 GiB Trident RWX NFS) — Entry↔Runtime can share HOME cross-node. **Canonical production endpoint the bioimage.io UI pins to `bioimage-io/bioengine-worker-kth-*:bioengine-worker`** — flip only when necessary. |
| **deNBI cloud K8s + Ray** | deNBI K8s | `bioimage-io/bioengine-worker-denbi-<hash>:bioengine-worker` | `denbi-k8s` repo (NOT `aicell-lab/kth-k8s`). Uses the **startup-application** mechanism — apps are pinned in `values.yaml` and cycled with the worker. `helm upgrade -f values.yaml` is REQUIRED (a plain `helm upgrade` silently reuses stored user-supplied values and drops the new pins) | Manila NFS RWX for HOME — Entry↔Runtime env sharing works cross-node. If the maintainer doesn't own `denbi-k8s`, the change is a coordination task: describe the values.yaml bump (image tag + startup-app version) and hand off. |

**Coordinated framework + app upgrades.** A bioengine version bump that changes app-facing API (0.11.22's rename `@bioengine.multiplexed → @bioengine.cached` was one) requires a matching app version bump landing at the same helm upgrade. On a single-commit helm upgrade the worker pod cycles → boots the new worker image → deploys the new app version → all in one atomic hop. Bumping the worker image alone would boot on 0.11.22 and immediately `AttributeError` while trying to deploy an app that still uses the removed API. The clean-break style used in 0.11.22 (no deprecation shim) makes this coordination mandatory rather than merely advisable.

## Other deployment modes — dev and HPC examples

Beyond the production external-cluster deployments, the other two supported modes each have their own workflow. Concrete targets vary per maintainer — the examples below happen to be the current maintainer's setup.

- **`single-machine` mode** — local Ray cluster on a workstation. Used for dev validation of the `ray.init(...)` path (different code path from KubeRay-mode; regressions in one won't necessarily show in the other). Launch with `python -m bioengine.worker --mode single-machine` or via docker-compose. Example: the current maintainer runs this on a machine called Europa; other maintainers will have their own dev box.
- **`slurm` mode** — on-demand worker per SLURM job on an HPC cluster, launched via apptainer. The bioengine version is picked up from `bioengine[worker]==<version>` in the job's env at submission time — no persistent helm, no active push required for framework bumps. Watch out for cluster-specific apptainer gotchas: e.g. `apptainer build sif` may fail on newer apptainer versions with `yama.ptrace_scope=2` — `apptainer build --sandbox` is the workaround (and the `start_hpc_worker.sh` script accepts sandbox dirs from 0.9.8+). Example: the current maintainer runs this on NSC's Berzelius (A100-SXM4-80GB); other maintainers will have their own HPC.

## Framework model cache (`bioengine._app.cache`)

`@bioengine.cached(max_models=N)` is the user-facing decorator; the details are in the sibling user-facing skill. This section describes the internals a maintainer needs when touching the code.

**What lives where:**
- `bioengine/_app/cache.py` — `PipelineCache` class + module-level helpers (`evict_all_models`, `evict_lru_model`, `evict_model`, `cached_model_ids`) + `_release_gpu_caches` (the `gc.collect()` + optional `torch.cuda.empty_cache()`).
- `bioengine/_app/decorators.py::cached` — the decorator marker. `_scan_class` collects `"cached"`-marked methods into `lifecycle["cached"]`; `@bioengine.app` wraps each with `_make_cached_wrapper` which lazy-instantiates the cache on first call.
- Cache lives on the deployment instance at `self._bioengine_caches: Dict[str, PipelineCache]` — one entry per decorated method, keyed by method name.

**Key invariants (regressions to look out for):**
- Every eviction path must call `_release_gpu_caches` **inside** the `asyncio.Lock`. If it's called after releasing the lock, a concurrent `get_or_load` sees a torch pool still holding the evicted VRAM and allocates on top — pynvml reports growth. Tests in `tests/_app/test_cache.py` mock `_release_gpu_caches` to assert it fires exactly once per eviction.
- `_release_gpu_caches` runs `gc.collect()` before `torch.cuda.empty_cache()`. Python may still hold refs via frames / weak refs / `__del__` closures; skipping the GC leaves the allocator's blocks unavailable to be returned.
- `torch` is imported inside the try — the framework does not hard-require torch. Apps without GPU/torch skip the empty_cache but still get correct cache semantics.
- `bioengine/_app/cache.py` MUST stay importable with just `bioengine[worker]` + stdlib (no torch, no numpy imports at module top). The introspection Ray task loads any module that touches `@bioengine.app` in a clean baseline `runtime_env`.

**What the refactor replaced (context for git-archaeology).** Before 0.11.22 `@bioengine.multiplexed` forwarded to `ray.serve.multiplexed(max_num_models_per_replica=N)` and `bioengine.multiplex.*` reached into Ray's private `__serve_multiplex_wrapper.unload_model_lru` for manual eviction. That path dropped Python refs but did not call `torch.cuda.empty_cache()` — pipelines evicted from the cache left ~200 MB of VRAM stuck per model, observable via `pynvml` and accumulating across sequential test calls (~626 MB pinned indefinitely on the model-runner in a common 3-model test scenario). The clean break (no shim) means older apps must migrate their decorator + module names before their worker upgrades.

## Worker service API

The BioEngineWorker registers as a Hypha service. Key methods:

| Method | Admin | Description |
|--------|:-----:|-------------|
| `get_status` | | Overall worker status |
| `check_access` | | Check caller permissions |
| `list_apps` | ✓ | List deployed applications |
| `deploy_app` | ✓ | Deploy an application from artifact |
| `stop_app` | ✓ | Stop a running application |
| `get_app_status` | | Status of specific application |
| `upload_app` | ✓ | Create/update application artifact |
| `get_app_manifest` | ✓ | Get manifest for an application |
| `delete_app` | ✓ | Delete an application artifact |
| `run_code` | ✓ | Run Python code in Ray task |
| `list_datasets` | | Available datasets |

## Dev image testing workflow

For deployment-side PRs (changes under `bioengine/**`, `bioengine/cluster/**`, `bioengine/apps/builder.py`, `bioengine/_app/bootstrap.py`, runtime_env transport, Ray client/server plumbing, Hypha service registration in the worker), validate on a live cluster before marking the PR ready. CI catches unit failures; it doesn't catch deployment topology issues. Validation must cover **both** modes — external-cluster and single-machine — because they exercise different code paths (the KubeRay-mode worker delegates to a remote Ray cluster carrying its own deps, the single-machine worker runs `ray.init` in-process). A change that's green on one mode can be broken on the other (e.g. an in-process `ray.init` exposes pip-resolved dep skew the kuberay pod hides).

1. Pick the next bioengine version (one above current `pyproject.toml` on `main` AND the latest published GHCR tag — bump from the higher) and append `-devN`. Example: next release `0.11.5` → `ghcr.io/aicell-lab/bioengine-worker:0.11.5-dev1`. Bump the suffix (`-dev2`, …) for each follow-up fix in the PR cycle.
2. Build + push directly with `docker buildx build --builder multiarch-builder --platform linux/amd64 -f docker/worker.Dockerfile -t ghcr.io/aicell-lab/bioengine-worker:<version>-devN --push .` using `GITHUB_PAT`. Bypasses `docker-publish-worker.yml` (which enforces strictly-increasing canonical tags). **Build single-arch only** (`--platform linux/amd64`) — both KTH and the single-machine test on europa are amd64. Multi-arch (`linux/amd64,linux/arm64`) adds ~25 min of QEMU cross-compile to the dev cycle for zero validation value; the canonical CI build covers arm64 once the PR is merged.
3. **External-cluster validation** (KTH): `helm upgrade` the test worker to the dev tag. The dev tag lives **only** on a feature branch of `kth-k8s` (or a transient local edit) — never on `kth-k8s` main. Smoke-verify via the live worker — deploy production apps at canonical versions, confirm RUNNING + ping/inference.
4. **Single-machine validation** (local docker): run the dev image as `docker run … python -m bioengine.worker --mode single-machine …` against the maintainer's personal Hypha workspace. Deploy a small CPU-only `@bioengine.app` and confirm it reaches RUNNING and serves over the Hypha proxy. This catches `ray.init` / replica-startup regressions the external-cluster path doesn't exercise.
5. Only after **both** modes are green: bump `pyproject.toml` to the canonical version in a separate commit, push, `gh pr ready <PR>`. CI publishes the canonical image.
6. After merge + canonical image published:
   - `helm upgrade` the test worker from `<version>-devN` → canonical. Commit the values.yaml bump back to canonical on `kth-k8s` main.
   - Delete every dev image tag from GHCR. The PAT has `delete:packages`.

### Cleaning up test GHCR images

After a canonical release, list every dev tag and DELETE via GHCR API. Pattern:

```bash
curl -s -H "Authorization: Bearer $GITHUB_PAT" \
  "https://api.github.com/orgs/aicell-lab/packages/container/bioengine-worker/versions?per_page=100" \
  | python3 -c "import json,sys; [print(v['id'], t) for v in json.load(sys.stdin) for t in v.get('metadata',{}).get('container',{}).get('tags',[]) if t.startswith('<version>-dev')]"
# Then for each id:
curl -X DELETE -H "Authorization: Bearer $GITHUB_PAT" \
  "https://api.github.com/orgs/aicell-lab/packages/container/bioengine-worker/versions/<id>"
```

Don't blanket-delete: dev tags + `:latest` move-tags are intentional. Only purge what you explicitly created.

## Dev app testing workflow

For app-side changes (anything under `apps/<name>/**` that changes the deployment's API surface or behaviour — e.g. a new method signature, changed return shape, a new dependency), validate on a live cluster by running the dev version SIDE-BY-SIDE with production, not by replacing it. The pattern:

1. **One artifact, many versions.** `bioimage-io/<app>` holds every version the artifact has ever shipped. `worker.upload_app` appends a new committed version to the same artifact — it does not create a separate artifact per version. Never make a `bioimage-io/<app>-v2` or `bioimage-io/<app>-dev` artifact; that fragments history and breaks the version-comparison stories consumers rely on.

2. **Distinct application_id for the dev deployment.** `application_id` is a *deployment* identity (Ray Serve service name + Hypha service alias), not an artifact identity. Production keeps `application_id="<app>"` pinned to the current released version. The dev deployment uses a distinct id — the canonical name is `application_id="<app>-dev"` — pinned to the new version. Both come from the same artifact.

   ```python
   # Production — pinned via the startup-app list in kth-k8s / denbi-k8s
   # values.yaml. Untouched during dev.
   {"artifact_id": "bioimage-io/model-runner",
    "application_id": "model-runner", "version": "1.14.0", ...}

   # Dev — deployed manually, side-by-side, same artifact, new version
   await worker.deploy_app(
       artifact_id="bioimage-io/model-runner",
       application_id="model-runner-dev",   # ← distinct
       version="1.15.0",                    # ← new
       hypha_token=os.environ["BIOIMAGE_IO_TOKEN"],
   )
   ```

3. **Registration.** The dev app registers as `bioimage-io/<app>-dev` on Hypha. Consumers that call `bioimage-io/<app>` stay on the production version until you promote. Iterate against `bioimage-io/<app>-dev` directly.

4. **Iteration is a fresh upload + redeploy.** For each dev iteration bump the `apps/<app>/manifest.yaml` version (the artifact rejects re-uploading the same version), `worker.upload_app` again, then `worker.deploy_app(artifact_id, application_id="<app>-dev", version=<new>)`. The application_id is stable across iterations, so `deploy_app` updates the existing dev deployment in place rather than spawning a new random-name instance (see the `application_id ≠ artifact_id` warning in CLAUDE.md).

5. **Promoting to production.** When the dev version is validated:
   - Bump the pin in `kth-k8s/bioengine-worker/values.yaml` (and `denbi-k8s/…/values.yaml`) to the new version. Commit + `helm upgrade -f values.yaml`. Per `model_runner_kth_denbi_sync.md` these two sites must ship in the same session.
   - After the production restart picks up the new version, `worker.stop_app(application_id="<app>-dev")` to reclaim cluster resources. The dev version stays in the artifact history — do NOT delete versions off the artifact.

6. **Cleanup on abandon.** If the dev version is abandoned rather than promoted, still `worker.stop_app("<app>-dev")` to free the replica slot. Leave the version on the artifact; artifact-version deletes rewrite history and break consumers of `worker.list_apps(...).artifact_versions`.

Compare with the worker-side dev image workflow above: that pattern varies the **image tag** (`0.11.5-dev1`), this pattern varies the **application_id** (`<app>-dev`). Both let you drive a live-cluster smoke test without disturbing the running production surface, at different layers of the stack.

## Cleanup rules

### Test deployments

After testing an app is complete, stop and delete temporary apps on the live worker:

```python
await worker.stop_app(application_id=app_id)   # stops the Ray Serve deployment
await worker.delete_app(artifact_id=app_id)    # deletes the Hypha artifact
```

Don't leave test/throwaway deployments on `bioimage-io/bioengine-worker` — they consume shared cluster resources.

### Failed deployments

When a deploy fails, **leave it in place while diagnosing** — `get_app_status(application_ids=[...])` → `deployments.<name>.logs` is the primary debug signal, and `stop_app` discards that history. Only after the fix is verified working: `stop_app` the failed instance (and `delete_app` if it was a throwaway test artifact; for production artifacts only `stop_app` — leave the artifact). Failed deploys consume Ray Serve retry capacity and clutter `list_apps`.

### Orphan apps before helm upgrade

Ray Serve state is cluster-wide and survives worker pod restarts. A new worker pod, on boot, **recovers every app it finds in `serve.status()`** that's stamped with its workspace marker. Before any `helm upgrade` that changes the deployed app set: list current apps, identify orphans (apps not in the new bootstrap config), `stop_app` them. Otherwise the new pod adopts them and they reappear in `list_apps` after the upgrade, surprising everyone.

## Hypha platform invariants

### Re-resolve service handles per call, never cache

A long-captured `server.get_service(...)` handle expires mid-session with `Method expired or not found` — even when the service is healthy in the registry. Resolve a fresh handle on **every method call** from any long-lived client (browser tab, dashboard, agent that stays connected for more than a few minutes). Only use the connect-time resolution as a probe (set `xxxAvailable` for UI gating).

Why: Hypha proxies each call through the server. Each `get_service(...)` returns a current-valid proxy stub. Re-resolution is cheap (no network beyond the lookup) and immune to whatever caches its expiry.

### Pin replica per session for stateful services

When a Hypha service maintains **per-replica state on local disk** (cellpose-finetuning training sessions in the actor's workdir, model checkpoints, downloaded artifact caches), the default `mode:'random'` / `mode:'last'` bouncing calls between replicas is silently wrong:

- `start_training(session_id=...)` lands on worker A → A writes `/home/.bioengine/sessions/<id>/...`
- `get_training_status(session_id=...)` lands on worker B → B sees nothing → "session not found"

Fix shape:

1. First call: `server.getService('<workspace>/<service>', {mode: 'random'})` → returns a handle whose `.id` is the concrete per-replica id.
2. **Stash that concrete id** in `sessionStorage` keyed by service name.
3. Subsequent calls: `server.getService(<pinned-id>)` — exact-id lookup, lands the same replica every time.
4. **Stale-pin recovery:** if the call throws (replica gone — Ray Serve roll, worker eviction), clear the pin and let the next call re-resolve a fresh random replica. Don't get stuck on a dead pin.

`sessionStorage` per tab is the right storage scope: long-running training + polling stay on the same replica across page navigations; each tab gets its own pin; pin clears with the tab. `localStorage` is wrong (cross-tab + persists past worker death).

Apply to: services that maintain per-session disk state. Inference-only services with no per-session state are fine on default routing.

## PR description style

PR bodies on `aicell-lab/*` and `bioimage-io/*` repos are standalone documentation read months later by maintainers and contributors who didn't follow the discussion that produced them. Assume the reader knows only how BioEngine is built.

**Include:**
- Problem / motivation
- Solution
- New features or behavioural changes
- Migration notes (if breaking)
- Test plan
- File-level summary

**Do not include:**
- "Design choice — alternative considered …"
- Pending follow-ups
- After-merge / deployment steps
- "Pairs with PR #N …" beyond a single `Depends on #N` line when there's a real ordering constraint
- "Per the recent discussion …", "As we agreed", "After I tested X" — any direct reference to the conversation flow

Previously-merged PRs that violate this style stay as-is; apply going forward only.

## BioEngine skills

Skills live in `../bioimage.io/public/skills/bioengine/` (sibling repo). They're Markdown documents describing BioEngine to a user-facing AI agent, published at <https://bioimage.io/skills/bioengine/SKILL.md>.

```
bioimage.io/public/skills/bioengine/
├── SKILL.md                        # Main entry point — load first
├── references/
│   ├── manifest_reference.md
│   └── cli_reference.md
└── apps/                           # App-specific subskills
    ├── model-runner/
    ├── cellpose-finetuning.md
    └── cell-image-search.md
```

CLI lives in `bioengine/cli/` in this repo. Install with `pip install "bioengine[cli] @ git+https://github.com/aicell-lab/bioengine.git"` or `pip install -e ".[cli]"` for development.

When skill content changes:

- **Main skill (`SKILL.md`)** — update when the worker API, CLI commands, manifest format, or deployment rules change.
- **App subskills** — update when the corresponding app's service API changes (parameters, status fields, export options, known pitfalls discovered during testing).
- **Adding a new app skill**: create `apps/<app-name>.md` (or `apps/<app-name>/` for multi-file) and add an entry to the app skills table in `SKILL.md`.
