# BioEngine — CLAUDE.md

## Repository Ecosystem

BioEngine spans three GitHub repositories with distinct roles:

| Repository | GitHub | Role | Status |
|------------|--------|------|--------|
| **`aicell-lab/bioengine`** | https://github.com/aicell-lab/bioengine | **This repo** — Python package, worker, CLI, apps, Docker image. The canonical implementation. Previously named `aicell-lab/bioengine-worker`. | Active |
| **`bioimage-io/bioimage.io`** | https://github.com/bioimage-io/bioimage.io | BioImage.IO website — contains all BioEngine UIs: worker dashboard, deployment configuration wizard, cluster monitor, app manager, interactive setup guide (`BioEngineGuide.tsx`), and agent skills (`public/skills/bioengine/`). | Active |
| **`bioimage-io/bioengine`** | https://github.com/bioimage-io/bioengine | **Archived/deprecated** — the original Triton-based BioEngine (pre-Ray). Kept for historical reference only; all links there redirect here. | Archived |

### Directory layout convention

The `bioimage-io/bioimage.io` repo is expected to be cloned as a sibling directory named `bioimage.io` (matching the actual repo name):

```
workspace/
├── bioengine/          ← this repo (aicell-lab/bioengine)
└── bioimage.io/        ← bioimage-io/bioimage.io
```

To set up on a new machine:

```bash
# Clone this repo
git clone git@github.com:aicell-lab/bioengine.git

# Clone the BioImage.IO website repo as ../bioimage.io
git clone git@github.com:bioimage-io/bioimage.io.git
```

The git remote `bioimage-io` is also added to this repo so agents and developers can reference the website repo without knowing its local path:

```bash
git remote add bioimage-io git@github.com:bioimage-io/bioimage.io.git
```

To add it to an existing clone:
```bash
git remote add bioimage-io git@github.com:bioimage-io/bioimage.io.git
```

### What lives where

- **Python package, worker, CLI, apps, Docker image** → `aicell-lab/bioengine` (this repo)
- **Worker dashboard & app manager UI** → `bioimage-io/bioimage.io/src/components/bioengine/`
  - `BioEngineHome.tsx` — lists available worker instances
  - `BioEngineWorker.tsx` — per-worker dashboard (deploy/stop apps, cluster resources)
  - `BioEngineGuide.tsx` — interactive deployment wizard for all modes (Docker, SLURM, K8s)
  - `BioEngineAppManager.tsx` — browser-based app file editor
  - `DeploymentConfigModal.tsx` — deployment configuration form
- **Agent skills** → `bioimage-io/bioimage.io/public/skills/bioengine/` (published at https://bioimage.io/skills/bioengine/SKILL.md)
- **BioEngine UI entry points**:
  - https://bioimage.io/#/bioengine — worker service listing
  - https://bioimage.io/#/bioengine/worker?service_id=... — worker dashboard
- **Production Hypha service** → workspace `bioimage-io` on https://hypha.aicell.io

---

## Project Overview

BioEngine is the **execution and adaptation layer between curated bioimage AI and scalable compute**. It enables deployment and serving of AI models and applications at any scale — from a lab workstation to a multi-node GPU cluster. The platform runs on [Ray](https://www.ray.io/) and [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) for distributed inference and integrates with [Hypha](https://hypha.aicell.io/) for RPC service discovery, artifact management, and authentication.

**Top-level goals:**
- Deploy and serve AI models and applications at scale (single-machine, SLURM HPC, Kubernetes/external Ray clusters)
- Provide a unified API for remote management of model deployments via Hypha RPC
- Stream large scientific datasets with privacy-preserving access control
- Support both compute backends (Ray Serve deployments) and static frontends (artifact-hosted web UIs)

---

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

### Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `BioEngineWorker` | `bioengine/worker/worker.py` | Main orchestrator; Hypha service registration |
| `AppsManager` | `bioengine/apps/manager.py` | Application lifecycle (deploy/stop/status) |
| `AppBuilder` | `bioengine/apps/builder.py` | Build Ray Serve apps from artifacts |
| `RayCluster` | `bioengine/cluster/ray_cluster.py` | Ray cluster lifecycle (SLURM/local/external) |
| `BioEngineDatasets` | `bioengine/datasets/datasets.py` | Zarr dataset streaming |
| Artifact utilities | `bioengine/utils/artifact_utils.py` | Hypha artifact CRUD helpers |

---

## Application Manifest (v0.11 / format_version 0.6.0)

Every BioEngine application is a folder containing `manifest.yaml` and the app's Python files at the root. The whole folder is uploaded to the Hypha artifact; the worker ships the same root to Ray Serve replicas as `runtime_env.py_modules`, excluding non-Python content (`manifest.yaml`, `README*`, `*.md`, `*.ipynb`, `frontend/`, images).

```
my-app/                       ← artifact root + Python module directory
├── manifest.yaml
├── README.md                 ← stays in artifact (excluded from py_modules)
├── frontend/index.html       ← Hypha hosts statically (excluded from py_modules)
└── deployment.py             ← @bioengine.app class
```

For multi-file apps:

```
composition-demo/
├── manifest.yaml
├── frontend/index.html
├── entry.py                  ← @bioengine.app entry — type-hints reference RuntimeA/B/C
├── utils.py
└── runtimes/
    ├── __init__.py
    ├── a.py
    ├── b.py
    └── c.py
```

### Required Fields

```yaml
name: My Application
id: my-application                       # Unique lowercase ID (hyphens only)
id_emoji: "🔬"
description: "..."
type: ray-serve                          # Kept
format_version: 0.6.0                    # v0.11 gate
entry: deployment:MyApp                  # module:Class — module is the .py filename
```

### Optional Fields

```yaml
frontend_entry: "frontend/index.html"    # Static frontend hosting
version: 1.0.0
authorized_users: ['*']                  # Or {method_name: [users], "*": [users]}
authors:
  - {name: "...", affiliation: "...", github_user: "..."}
license: MIT
documentation: README.md
tutorial: tutorial.ipynb
tags: [bioengine, image-analysis]
```

When `frontend_entry` is set, BioEngine configures a `view_config` on the Hypha artifact during `upload_app` (while the artifact is staged). The `frontend_entry` determines `root_directory` and `index` (e.g., `frontend/index.html` → `root_directory: "frontend"`, `index: "index.html"`). The resulting URL is:
```
https://hypha.aicell.io/{workspace}/view/{artifact-id}/
```

### Authoring model

User code uses the decorators in the `bioengine` package — `@bioengine.app`, `@bioengine.method`, `@bioengine.async_init`, `@bioengine.smoke_test`, `@bioengine.health_check`, `@bioengine.multiplexed` — and accesses datasets/logger via the module-level `bioengine.datasets` / `bioengine.logger` accessors. Multi-deployment composition is declared by `__init__` type hints; `.remote()` is hidden by `BioEngineRuntimeHandle`. See `docs/migration/v0.11.md` for the full mapping from the legacy decorators and an end-to-end migration walkthrough; see `apps/demo-app/` and `apps/composition-demo/` for reference apps.

### Local validation

```bash
bioengine apps validate ./my-app
```

The CLI runs the same validator that the worker uses, so legacy manifests fail fast with a migration hint without a round-trip.

---

## Deployment Modes

| Mode | Description |
|------|-------------|
| `single-machine` | Local Ray cluster (dev, small-scale) |
| `external-cluster` | Connect to existing Ray cluster (Kubernetes) |
| `slurm` | Auto-scaling via SLURM job scheduler (HPC) |

---

## Worker Service API

The BioEngine worker registers as a Hypha service. Key methods:

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

---

## Development Setup

```bash
conda activate bioengine
pip install -e ".[worker,cli,dev]"
```

**HYPHA_TOKEN:** Create a `.env` file in the repo root with your Hypha token if not already present, then source it:
```bash
echo "HYPHA_TOKEN=<your-token>" > .env   # obtain a token from https://hypha.aicell.io
source .env
```
If `.env` already exists, just `source .env` before running the worker or tests.

**bioimage-io/bioimage.io sibling repo:** Clone the BioImage.IO website repo as a sibling of this repo (required for editing skills and UI components):
```bash
git clone git@github.com:bioimage-io/bioimage.io.git ../bioimage.io
```

### Run Locally

```bash
python -m bioengine.worker \
    --mode single-machine \
    --head-num-gpus 1 \
    --head-num-cpus 4 \
    --workspace-dir ~/.bioengine \
    --debug
```

For local artifact development, set:
```bash
export BIOENGINE_LOCAL_ARTIFACT_PATH=/path/to/bioengine/tests
```

### Run Tests

```bash
pytest tests/end_to_end/ -v
```

### Test Organization

- `tests/end_to_end/` — Integration tests for the core worker (applications manager, datasets, code executor)
- `tests/apps/` — Tests for individual BioEngine apps; one subfolder per app (e.g. `tests/apps/cellpose/`)
- `tests/apps/<app-name>/` — **All tests specific to a BioEngine app must go here**, not in `tests/end_to_end/`

---

## Code Conventions

- **App authorized_users**: When deploying an app, the worker's `admin_users` and the deploying user are always injected into every key of `authorized_users` (including `"*"`). This guarantees admins can always call any app method regardless of the app's access rules.
- **Permissions**: Use `check_permissions(context, authorized_users, resource_name)` from `bioengine.utils`
- **Schema methods**: Decorate public API methods with `@schema_method` and use `pydantic.Field` for parameter descriptions
- **Logging**: Use `create_logger("ComponentName", ...)` from `bioengine.utils`
- **Artifact IDs**: Always fully qualified as `workspace/alias`
- **Artifact config**: Use `{"permissions": {"*": "r"}}` for public read; `{"website_root": "<dir>"}` for static hosting

## Key File Locations

- `bioengine/cli/` — BioEngine CLI (`bioengine` command); entry point is `bioengine.cli.cli:main`
- `bioengine/utils/artifact_utils.py` — All Hypha artifact CRUD helpers
- `bioengine/apps/manager.py` — `deploy_app`, `upload_app`, lifecycle
- `bioengine/apps/builder.py` — `build()` constructs Ray Serve app from artifact
- `apps/demo-app/` — Reference BioEngine app (single deployment + frontend; ping, ascii_art, list_datasets, reverse_text); **always keep version at 1.0.0**
- `apps/composition-demo/` — Multi-deployment composition app (entry + 3 runtimes, reference for composition pattern); **always keep version at 1.0.0**
- `apps/model-runner/` — Production model-runner app
- `apps/cellpose-finetuning/` — Cellpose fine-tuning app
- `pyproject.toml` — Package version and dependencies; install with `pip install -e ".[cli]"` for CLI use
- `../bioimage.io/public/skills/bioengine/` — Agent skills for working with BioEngine (in `bioimage-io/bioimage.io` repo)

---

## BioEngine Skills

Skills live in the **`../bioimage.io/public/skills/bioengine/`** directory (in the `bioimage-io/bioimage.io` repo). They are Markdown documents that describe BioEngine capabilities to an AI agent and are published at https://bioimage.io/skills/bioengine/SKILL.md.

### Skill structure

```
bioimage.io/public/skills/bioengine/
├── SKILL.md                        # Main entry-point — load this first
├── references/
│   ├── manifest_reference.md       # Full manifest.yaml field reference
│   └── cli_reference.md            # CLI command reference
└── apps/                           # App-specific subskills
    ├── model-runner/               # BioImage.IO model inference
    ├── cellpose-finetuning.md      # Cellpose fine-tuning
    └── cell-image-search.md        # Cell image search
```

The CLI source lives in `bioengine/cli/` in this repo. Install with `pip install "bioengine[cli] @ git+https://github.com/aicell-lab/bioengine.git"` (or `pip install -e ".[cli]"` for development).

### How skills are used

- **`SKILL.md`** is the single entry-point skill. It covers app deployment, CLI, and all platform concepts, and references app subskills for deeper detail.
- **App subskills** (`apps/model-runner/`, `apps/cellpose-finetuning.md`, etc.) are referenced from `SKILL.md` — agents load them on demand when the task requires a specific service.

### Working on skills

- **Main skill** (`SKILL.md`): Update when the worker API, CLI commands, manifest format, or deployment rules change.
- **Model runner skill** (`apps/model-runner/`): Update when `apps/model-runner/` service API changes.
- **Cellpose fine-tuning skill** (`apps/cellpose-finetuning.md`): Update when `apps/cellpose-finetuning/main.py` service API changes (new training parameters, new status fields, new export options, known pitfalls discovered during testing).

### Adding a new app skill

1. Create `apps/<app-name>.md` (or `apps/<app-name>/` for multi-file) in the bioimage.io skills directory.
2. Add an entry to the app skills table in `SKILL.md`.

---

## External Skills

| Skill | URL | Purpose |
|-------|-----|---------|
| **BioEngine** | `../bioimage.io/public/skills/bioengine/SKILL.md` (fallback: https://bioimage.io/skills/bioengine/SKILL.md) | Deploy apps, call services, use the CLI — load this first when working with BioEngine |
| Hypha | https://hypha.aicell.io/ws/agent-skills/SKILL.md | Connect to the Hypha distributed computing platform — obtain tokens, discover workspaces, call services via RPC or HTTP, manage artifacts, deploy apps |

---

## Agent Workflow Guidelines

- **Sync with remote BEFORE doing anything**: BioEngine is developed across multiple machines (each tests a different deployment mode), so the *local* clone is almost always stale. Before editing code, before bumping a version, before opening a PR — sync first:
  - **On `main`**: `git fetch origin && git log --oneline HEAD..origin/main` — if any commits are listed, `git pull --ff-only origin main` before touching anything else. If the fast-forward fails, stop and ask the user; never `git pull` with a merge.
  - **On an existing feature branch** (the branch may have new commits pushed from another machine): `git fetch origin && git log --oneline HEAD..origin/<branch>` — if commits exist, `git pull --rebase origin <branch>` before adding your own. Never push without fetching first; force-push only with explicit user approval.
  - **Before bumping `version` in `pyproject.toml`**: do NOT trust the local file. Another machine may already have merged a higher version. Check the latest published image tag on GHCR — the CI version-strictly-greater check uses this, not the local file:
    ```bash
    source .env && curl -s -H "Authorization: Bearer $GITHUB_PAT" \
      "https://api.github.com/orgs/aicell-lab/packages/container/bioengine-worker/versions?per_page=20" \
      | python3 -c "import json,sys; print('\n'.join(t for v in json.load(sys.stdin) for t in v.get('metadata',{}).get('container',{}).get('tags',[]) if t and t!='latest'))" \
      | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1
    ```
    Bump from whichever is higher: `origin/main`'s `pyproject.toml` or the latest GHCR tag.
  - **Before bumping `version` in `apps/<name>/manifest.yaml`**: check `get_app_status` on the live worker for the currently deployed version, AND check `origin/main`'s copy of the manifest. Bump from the higher of the two so the next CI publish strictly increases.
- **Simplicity First**: Make every change as minimal as possible.
- **No Regressions**: Only change what's necessary; read before modifying.
- **Prove It Works**: Test and verify before marking done.
- **Comments — write almost none**: code should explain *what* via good names; comments only earn their place when they explain *why* something non-obvious is true. Concretely:
  - **Do not narrate** what the next line does (`# fetch the user`, `# loop over items`, `# now connect`). The line says it.
  - **Do not describe the PR / current task** in a comment (`# added for the geo location PR`, `# this fixes the timeout we hit yesterday`). That belongs in the commit message; in the file it just rots.
  - **Do not write multi-paragraph comments** unless the reader genuinely cannot reconstruct the reasoning. One short sentence is almost always enough; two if there's a real footgun. Prefer a one-line docstring over a wall of `#`.
  - **Do write a comment** when it captures a hidden constraint a future reader would otherwise re-discover the hard way: a workaround for a known bug, a non-obvious invariant, an external system's quirk, a reason for a counter-intuitive default. Make the comment as short as possible while still naming the *why*.
  - **Do not bake in machine-specific or user-identifying paths/identifiers** (`/proj/aicell/users/x_nilme/...`, `/home/<user>/...`, a SLURM allocation id, a private cluster name). The repo ships to other clusters; those strings leak local layout and rot the moment paths change. Use a generic placeholder (`<workspace>/...`, `the cluster's shared filesystem`) or drop the path entirely if the surrounding code already explains the shape.
  - When in doubt: delete the comment, run the diff again, and ask whether the code is materially harder to understand. If not, the comment was not pulling its weight.
- Planning lives in model context — do NOT create planning files in the repo.
- **This file is also exposed as `.github/copilot-instructions.md`** (a symlink to `../CLAUDE.md`) so GitHub Copilot agents see the same rules as Claude Code agents. Edit `CLAUDE.md`; the symlink follows.
- **Test on the live worker**: When working on a BioEngine app, test and debug by deploying to the live `bioimage-io/bioengine-worker` service on https://hypha.aicell.io and calling the service directly. Do not write standalone test scripts for app behaviour — use the live service. Deploy with a stable `application_id` matching the artifact alias so the service is consistently addressable:
  ```python
  app_id = await worker.deploy_app(
      artifact_id='bioimage-io/my-app',
      version='1.2.3',
      application_id='my-app',   # gives stable service ID, not a random name
  )
  svc = await client.get_service(f'bioimage-io/{app_id}')
  ```
- **CRITICAL — artifact ID ≠ app ID, omitting `application_id` always creates a NEW instance**: One artifact can be deployed multiple times with different `application_id`s. `deploy_app(artifact_id)` without `application_id` **always spawns a brand-new instance with a random ID** — it never updates an existing one. To update a running app, you MUST pass its `application_id` AND the new `version` explicitly:
  ```python
  # WRONG — creates a new random instance, does NOT update cellpose-finetuning:
  await worker.deploy_app('bioimage-io/cellpose-finetuning')
  
  # CORRECT — updates the running 'cellpose-finetuning' instance to the new version:
  await worker.deploy_app(
      'bioimage-io/cellpose-finetuning',
      application_id='cellpose-finetuning',
      version='0.0.28',
  )
  ```
  Before deploying, always check `list_apps()` or `get_app_status(None)` to find the correct running `application_id`.
- **Commit after live deploy**: Once an app in `apps/` is verified working on the live worker, commit the source to git so the deployed version is always reproducible:
  ```bash
  git add apps/my-app/
  git commit -m "feat(my-app): describe change, bump version to X.Y.Z"
  git push
  ```
  The version in `manifest.yaml` must be bumped whenever app code changes.
- **Version bump rules**:
  - **`deploy-applications.yml`** is manual-dispatch only (push trigger disabled — agents deploy directly via the worker API). Always bump `version` in the affected app's `manifest.yaml` when app code changes.
  - **`docker-publish.yml`** triggers on changes to any of these paths: `bioengine/**`, `requirements*.txt`, `pyproject.toml`, `docker/**`, `.dockerignore`. It enforces that `version` in `pyproject.toml` is strictly greater than the latest published image tag — CI will fail if not bumped. **Always create a PR** (never push directly to `main`) and **bump `version` in `pyproject.toml`** before opening the PR whenever any of those paths are touched.
- **PRs are only required for changes that trigger `docker-publish.yml`** (i.e. changes under `bioengine/**`, `requirements*.txt`, `pyproject.toml`, `docker/**`, `.dockerignore`). Changes to `apps/**` only — push directly to `main`, no PR needed.
- **NEVER push directly to `main` for worker/package code.** Always use a feature branch and open a PR for any change that touches the paths above. If the user asks you to push directly to main for those paths, refuse and create a PR instead.
- **Always open the PR immediately after pushing the branch** using the GitHub PAT from `.env` so the user can see and review it without having to navigate to GitHub manually. Never merge a PR — that is always left to the user.
- **PR descriptions are standalone documentation, not conversation summaries.** A PR body on any `aicell-lab/*` or `bioimage-io/*` repo is read months later by maintainers, contributors, and automated review tools, who did not follow the discussion that produced it. Assume the reader knows only how BioEngine is built.
  - **Include**: problem / motivation, solution, new features or behavioural changes, migration notes (if breaking), test plan, file-level summary.
  - **Do not include**: "design choice — alternative considered ...", pending follow-ups, after-merge / deployment steps, "pairs with PR #N ..." beyond a single `Depends on #N` line when there's a real ordering constraint, "per the recent discussion ...", "as we agreed", "after I tested X" — any direct reference to the conversation flow that produced the PR.
  - Previously-merged PRs that violate this style stay as-is — apply going forward only.
- **Clean up test deployments**: After testing is complete, stop and delete any temporary apps deployed to the live worker:
  ```python
  await worker.stop_app(application_id=app_id)   # stops the Ray Serve deployment
  await worker.delete_app(artifact_id=app_id)    # deletes the Hypha artifact
  ```
  Do not leave test/throwaway deployments running on the live `bioimage-io/bioengine-worker` Hypha service — they consume shared cluster resources.
