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
