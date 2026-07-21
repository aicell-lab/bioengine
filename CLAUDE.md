# BioEngine — CLAUDE.md

Universal behavioural rules for agents working on this repo. **Hard
constraints; ignore at your peril.** Reference material (architecture,
file map, manifest reference, dev image workflow, cleanup rules, Hypha
platform invariants, PR description style) lives in
`.claude/skills/bioengine-maintainer/SKILL.md` and is loaded on demand.

For agents *using* BioEngine to deploy apps (rather than maintaining
the codebase), the user-facing skill is at
`../bioimage.io/public/skills/bioengine/SKILL.md` (sibling repo) or
<https://bioimage.io/skills/bioengine/SKILL.md>.

Also exposed as `.github/copilot-instructions.md` (a symlink to this
file) so GitHub Copilot agents see the same rules. Edit this file; the
symlink follows.

## Sync with remote BEFORE doing anything

BioEngine is developed across multiple machines. The local clone is
almost always stale. Before editing code, before bumping a version,
before opening a PR — sync first:

- **On `main`**: `git fetch origin && git log --oneline HEAD..origin/main` — if any commits are listed, `git pull --ff-only origin main` before touching anything else. If the fast-forward fails, stop and ask; never `git pull` with a merge.
- **On an existing feature branch** (may have commits pushed from another machine): `git fetch origin && git log --oneline HEAD..origin/<branch>` — if commits exist, `git pull --rebase origin <branch>` before adding your own. Never push without fetching first; force-push only with explicit user approval.
- **Before bumping `version` in `pyproject.toml`**: do NOT trust the local file. Another machine may already have merged a higher version. Check the latest published image tag on GHCR — the CI version-strictly-greater check uses this, not the local file:
  ```bash
  source .env && curl -s -H "Authorization: Bearer $GITHUB_PAT" \
    "https://api.github.com/orgs/aicell-lab/packages/container/bioengine-worker/versions?per_page=20" \
    | python3 -c "import json,sys; print('\n'.join(t for v in json.load(sys.stdin) for t in v.get('metadata',{}).get('container',{}).get('tags',[]) if t and t!='latest'))" \
    | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1
  ```
  Bump from whichever is higher: `origin/main`'s `pyproject.toml` or the latest GHCR tag.
- **Before bumping `version` in `apps/<name>/manifest.yaml`**: check `get_app_status` on the live worker for the currently deployed version AND check `origin/main`'s copy. Bump from the higher of the two so the next CI publish strictly increases.

## Code conventions

- **Simplicity First.** Make every change as minimal as possible.
- **No Regressions.** Only change what's necessary; read before modifying.
- **Prove It Works.** Test and verify before marking done.
- **Don't add features, refactors, abstractions, or error handling beyond what the task requires.** No designing for hypothetical futures. Three similar lines is better than a premature abstraction. No half-finished implementations.
- **Trust internal code and framework guarantees.** Only validate at system boundaries (user input, external APIs). Don't add backwards-compat shims when you can just change the code.
- **Don't bake in machine-specific paths/identifiers** (`/proj/aicell/users/x_nilme/...`, `/home/<user>/...`, SLURM allocation ids, private cluster names). The repo ships to other clusters; those strings leak local layout and rot. Use a generic placeholder or drop the path entirely if the surrounding code already explains the shape.

### Comments — write almost none

Code should explain *what* via good names; comments earn their place only when they explain *why* something non-obvious is true.

- **Do not narrate** what the next line does (`# fetch the user`, `# loop over items`, `# now connect`). The line says it.
- **Do not describe the PR / current task** in a comment (`# added for the geo location PR`, `# this fixes the timeout we hit yesterday`). That belongs in the commit message; in the file it just rots.
- **Do not write multi-paragraph comments** unless the reader genuinely cannot reconstruct the reasoning. One short sentence is almost always enough; two if there's a real footgun. Prefer a one-line docstring over a wall of `#`.
- **Do write a comment** when it captures a hidden constraint a future reader would otherwise re-discover the hard way: a workaround for a known bug, a non-obvious invariant, an external system's quirk, a reason for a counter-intuitive default. Make it as short as possible while still naming the *why*.
- When in doubt: delete the comment, run the diff again, and ask whether the code is materially harder to understand. If not, the comment was not pulling its weight.

## When PRs are required vs direct-push to main

| Path | Action |
|---|---|
| `bioengine/**`, `requirements*.txt`, `pyproject.toml`, `docker/**`, `.dockerignore` | **PR required.** These trigger `docker-publish-worker.yml` and `version-check.yml`. Never push directly to `main`. If the user asks you to, refuse and create a PR instead. |
| `apps/**` | **Push direct to `main`** — no PR. Then re-upload via the running worker so the Hypha artifact tracks main. |
| `docs/**`, `.github/**` (CI workflows), `.claude/skills/**`, `CLAUDE.md`, `README.md` | **Push direct to `main`** — no PR. |

## PR workflow (for paths above that require one)

1. Push the feature branch with the substantive changes only (NO `pyproject.toml` version edit).
2. `gh pr create --draft …` so it's visible for review but cannot be auto-merged. **Open as DRAFT.**
3. **Bump pyproject.toml just before marking ready.** Bumping early causes lost-bump collisions when a different PR merges first and ships the version you reserved. Workflow:
   - Re-check the latest published image tag on GHCR AND `origin/main`'s `pyproject.toml`.
   - Bump from the higher of the two as a separate atomic commit: `chore(release): bump version to X.Y.Z`.
   - Push, then `gh pr ready <number>`.
4. `version-check.yml` listens for `ready_for_review`, so marking a draft PR ready fires the required status check without any manual re-trigger.
5. **Always open the PR immediately after pushing the branch** using the GitHub PAT from `.env` so the user can see and review it without navigating to GitHub manually.
6. **Never merge a PR** — that is always left to the user.

For deployment-side PRs (changes under `bioengine/**`, `bioengine/cluster/**`, `bioengine/_app/**`, `bioengine/apps/builder.py`, runtime_env transport, Ray client/server plumbing) the **dev image testing workflow** in `.claude/skills/bioengine-maintainer/SKILL.md` applies: build `<next-version>-devN`, validate on a live cluster, only then mark the PR ready.

## App version bumps

- Always bump `version` in `apps/<name>/manifest.yaml` when app code changes.
- A committed artifact version is **immutable** and its content is exactly what a deploy runs — so a version string must map to one bundle forever. Never delete-and-recreate a version to change its code, and never deploy a *staged* (uncommitted) version; deploy pinned, committed versions only. `upload_app` already rejects any version that isn't strictly greater than every existing one (PEP 440).
- To iterate without inflating the release history, use pre-releases: upload `X.Y.Z-devN`, test, then publish the verified bundle once as `X.Y.Z` and drop the `-dev*` pre-releases. See the user-facing skill's dev-iteration workflow + `scripts/upload_app.py` (`../bioimage.io/public/skills/bioengine/`).
- `deploy-applications.yml` is manual-dispatch only (push trigger disabled). Agents deploy directly via the worker API.

## Testing apps on the live worker

When working on a BioEngine app, **test by deploying to the live worker** at `bioimage-io/bioengine-worker` on <https://hypha.aicell.io> and calling the service directly. Do not write standalone test scripts for app behaviour. Deploy with a stable `application_id` matching the artifact alias so the service is consistently addressable:

```python
app_id = await worker.deploy_app(
    artifact_id='bioimage-io/my-app',
    version='1.2.3',
    application_id='my-app',   # gives stable service ID, not a random name
)
svc = await client.get_service(f'bioimage-io/{app_id}')
```

Once an app is verified working, commit the source to git so the deployed version is reproducible:

```bash
git add apps/my-app/
git commit -m "feat(my-app): describe change, bump version to X.Y.Z"
git push
```

The version in `manifest.yaml` must be bumped whenever app code changes.

## CRITICAL — artifact_id ≠ application_id

One artifact can be deployed multiple times with different `application_id`s. **`deploy_app(artifact_id)` without `application_id` always spawns a brand-new instance with a random ID — it never updates an existing one.** To update a running app, you MUST pass its `application_id` AND the new `version` explicitly:

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

## `hypha_token` on `deploy_app` — read the parameter carefully

Apps whose code reads `HYPHA_TOKEN` at startup need it injected into the Ray actor; this is done via the `hypha_token` parameter on `deploy_app` (CLI: `--hypha-token $HYPHA_TOKEN`). When `application_id` matches an existing running instance, omitting `hypha_token` is safe — the previously stored token is reused. **On a fresh instance (no prior running app with that `application_id`), omitting it injects nothing and the app's `__init__` raises `RuntimeError: HYPHA_TOKEN environment variable is not set`** (or similar). Cross-check the app source before deploying — if `HYPHA_TOKEN` is referenced, pass the parameter. The `--env HYPHA_TOKEN=...` flag is silently ignored by the app builder.

## Cleanup after testing

After testing is complete, **stop and delete temporary apps** on the live worker:

```python
await worker.stop_app(application_id=app_id)   # stops the Ray Serve deployment
await worker.delete_app(artifact_id=app_id)    # deletes the Hypha artifact
```

When a deploy fails, **leave it in place while diagnosing** — `get_app_status(application_ids=[...])` → `deployments.<name>.logs` is the primary debug signal, and `stop_app` discards that history. After a successful redeploy of the fix is verified working: `stop_app` the failed instance (and `delete_app` if throwaway; for production artifacts only `stop_app`, leave the artifact).

## Reference material lives elsewhere

Don't duplicate this content here:

| Topic | Where |
|---|---|
| Repository ecosystem, architecture, file map, manifest reference, dev setup | `.claude/skills/bioengine-maintainer/SKILL.md` |
| Dev image testing workflow (build, push, helm, GHCR cleanup) | `.claude/skills/bioengine-maintainer/SKILL.md` |
| Cleanup rules (orphan apps, GHCR test images) | `.claude/skills/bioengine-maintainer/SKILL.md` |
| Hypha platform invariants (handle expiry, replica pinning) | `.claude/skills/bioengine-maintainer/SKILL.md` |
| PR description style | `.claude/skills/bioengine-maintainer/SKILL.md` |
| BioEngine skills overview (user-facing skill structure) | `.claude/skills/bioengine-maintainer/SKILL.md` |
| End-to-end deploy_app flow (14-step v0.11.4 walkthrough) | `docs/deploy_app_flow.md` |
| App authoring (`@bioengine.app`, manifest fields, deploying apps) | `../bioimage.io/public/skills/bioengine/SKILL.md` |
| Hypha platform basics (tokens, workspaces, RPC, artifacts) | <https://hypha.aicell.io/ws/agent-skills/SKILL.md> |

## Planning / files

- Planning lives in model context — do NOT create planning files in the repo.
- If the user asks for help or wants to give feedback inform them:
  - `/help` — get help with using Claude Code
  - Feedback: <https://github.com/anthropics/claude-code/issues>
