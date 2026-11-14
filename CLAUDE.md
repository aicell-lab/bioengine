# BioEngine — CLAUDE.md

Universal behavioural rules for agents working on this repo. **Hard
constraints; ignore at your peril.** Reference material (architecture,
file map, manifest reference, dev image workflow, cleanup rules, Hypha
platform invariants, issue and PR writing style) lives in
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
- **A worker version bump is TWO files**: `version` in `pyproject.toml` and `__version__` in `bioengine/_version.py`, edited in lock-step in the same commit. `version-check.yml` has a dedicated step that fails the required check when they differ. (`_version.py` exists because Ray Serve replicas import `bioengine` via `runtime_env.py_modules`, which ships module source but no `.dist-info`, so `importlib.metadata` has nothing to read replica-side.)
- **Whether you need a bump at all is a narrower question than whether the path needs a PR.** `version-check.yml` only compares versions when the PR's diff matches `^(bioengine/|docker/worker\.Dockerfile$|docker/datasets\.Dockerfile$|requirements[^/]*\.txt$|pyproject\.toml$|\.dockerignore$)`. So under `docker/` only `worker.Dockerfile` and `datasets.Dockerfile` count — the overlay recipe and the per-app Dockerfiles do not — and the `requirements` pattern is anchored, so it matches root-level files only and never `apps/<name>/requirements-*.txt`. A PR matching none of them reports `check-version` as a **success** ("No relevant files changed. Skipping version check."), indistinguishable from a passed comparison, so nothing tells you the bump was unnecessary — and an unnecessary bump consumes the next free version, forcing a second bump on every other open PR that already reserved one. The lock-step assertion above is **not** conditional: `bioengine/_version.py` must match `pyproject.toml` on every PR, including one that needs no bump.
- **Before bumping**: do NOT trust the local files. Another machine may already have merged a higher version. Two different checks compare against two different baselines, and only one of them runs on your PR:
  - `version-check.yml` — the required PR check — compares your branch's `pyproject.toml` against **`origin/main`'s**. GHCR is not involved.
  - `docker-publish-worker.yml` compares against the **newest published GHCR tag**, and it triggers on push to `main` — i.e. *after* merge. A version higher than `origin/main` but not higher than the newest GHCR tag passes the PR gate and then fails the publish on `main`.

  So bump from whichever is higher: `origin/main`'s `pyproject.toml`, or the latest GHCR tag:
  ```bash
  source .env && curl -s -H "Authorization: Bearer $GITHUB_PAT" \
    "https://api.github.com/orgs/aicell-lab/packages/container/bioengine-worker/versions?per_page=20" \
    | python3 -c "import json,sys; print('\n'.join(t for v in json.load(sys.stdin) for t in v.get('metadata',{}).get('container',{}).get('tags',[]) if t and t!='latest'))" \
    | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1
  ```
  One page is enough to answer *"what is the newest tag"*; it is not enough to answer *"is version X published"*. That API paginates over **digests**, most of them untagged, so a short page drops tagged versions out of the middle while still showing the newest. Measured 2026-11-06 against this exact URL: `per_page=40` returned 8 semver tags and **no `0.16.75`**; `per_page=100` returned 20 and `0.16.75` was on page one. Both reported the same newest tag. So a short page never misleads you about the newest and will readily mislead you about **membership** — use `per_page=100` whenever the question is "does this version exist", and never read a short page's silence as absence.
- **Before bumping `version` in `apps/<name>/manifest.yaml`**: check `get_app_status` on the live worker for the currently deployed version AND check `origin/main`'s copy. Bump from the higher of the two so the next CI publish strictly increases.

## Environment facts that would mislead you

- **Do not bake in machine-specific paths or identifiers** (`/proj/aicell/users/...`, `/home/<user>/...`, SLURM allocation ids, private cluster names). This repo ships to other clusters; those strings leak local layout and rot.

## When PRs are required vs direct-push to main

**This table is enforceable as a check — install it instead of remembering it:** `git config core.hooksPath .github/hooks` refuses a push to `main` that touches any PR-only path below. Read `.github/hooks/README.md` first; `core.hooksPath` is shared by every worktree of a clone, so arming it in a clone other sessions are working in affects them too. The repository's own `Require PR` ruleset cannot express this policy — it is scoped by ref with no path component, so it forbids direct pushes to `main` for *every* path, and it carries an `admin`-role bypass, which is why direct pushes succeed and only a `remote: Bypassed rule violations` line reports them.

**Which surface enforces versions.** `version-check.yml` and `app-version-check.yml` are both `pull_request`-triggered, so neither one sees a commit that reaches `main` without a PR. `push-version-check.yml` covers that path: it is `push`-triggered on `main` and runs the same two gates over `github.event.before..github.event.after` — `.github/scripts/check_worker_version.py` for `pyproject.toml` plus `bioengine/_version.py`, and the *same* `.github/scripts/check_app_artifact_version.py` the PR gate calls, so the published-artifact membership leg is identical on both paths. It is a **backstop, not a gate**: by the time it runs the commit is already on `main`, so it can only make a bad version loud.

⚠ **And on europa it is currently the *only* thing running, because the pre-push hook above never fires.** `core.hooksPath` is set repo-locally in `/data/nmechtel/bioengine/.git/config` to `.git/hooks`, which contains `commit-msg` and `pre-commit` and **no `pre-push`** — so `git push` never invokes `.github/hooks/pre-push`, and `match-pr-only.sh` / `pr-only-paths.txt` are dead code on that clone. Measured 2026-11-14 behaviourally, not from the config alone: a `git push --dry-run` of a commit touching `bioengine/__init__.py` to `main` exits **0** with **no** hook output, while piping the same ref line into the hook directly prints its `REFUSED` banner. That is the failure mode to watch for — running a hook's script by hand demonstrates the gate working while proving nothing about whether `git push` calls it. Test it with `git push --dry-run` and assert the hook's own output appears, or assert `$(git rev-parse --git-path hooks)/pre-push` exists. Those two hooks carry a `mycolab-crew-guard` header and were installed by tooling that sets `core.hooksPath` as a side effect, so a repoint at `.github/hooks` can be silently undone later. Until that is fixed and defended, treat the server-side workflow as the enforcement surface and the hook as aspirational.

| Path | Action |
|---|---|
| `bioengine/**`, `requirements*.txt`, `pyproject.toml`, `docker/**`, `.dockerignore` | **PR required.** Never push directly to `main`. If the user asks you to, refuse and create a PR instead. This is the *routing* rule only — both `version-check.yml` and `docker-publish-worker.yml` gate strictly narrower path sets than this list, so a PR here does not necessarily need a version bump. See "Whether you need a bump at all" above before bumping. A commit that reaches `main` without a PR is still version-checked, after the fact, by `push-version-check.yml`. |
| `apps/model-runner/**`, `apps/model-finetune/**` | **PR required** (backlog #0124). These two apps are published as Hypha artifacts and are gated by `app-version-check.yml`: on a non-draft PR, `version` in that app's `manifest.yaml` must be greater (PEP 440) than `origin/main`'s **and** must already exist in the published artifact's version list. So upload before you mark ready: open a draft PR, iterate on the live worker with `X.Y.Z-devN` pre-releases, upload the real `X.Y.Z`, then `gh pr ready`. Both legs, membership included, also run on the push path via `push-version-check.yml`. |
| `apps/**` (every other app) | **Push direct to `main`** — no PR. Then re-upload via the running worker so the Hypha artifact tracks main. An active GitHub ruleset actually forbids this and our PAT bypasses it silently (backlog #0112, still open for these paths). |
| `docs/**`, `.github/**` (CI workflows), `.claude/skills/**`, `CLAUDE.md`, `README.md` | **Push direct to `main`** — no PR. |

## PR workflow (for paths above that require one)

**A PR stays a DRAFT until the issue is cleared to land.** Ready does not mean "the code is finished"; it means "this goes to production now". A ready PR has to carry a version bump to pass `check-version`, so every ready PR reserves a rung and forces a merge order on all the others. Keep the number of ready PRs at one, or near it.

1. Push the feature branch with the substantive changes only (NO version edit).
2. `gh pr create --draft …` — **open as DRAFT**, and leave it there for the whole of the work.
3. **Get the clearance the change needs.** An independent reviewer agent's APPROVE is enough for a fix, a test or a doc change. A new feature, anything user-facing on the bioimage.io pages, and anything added to the BioEngine workspace or the model zoo needs the user's go as well.
4. **Only then** bump, mark ready and merge — in one sitting, not days apart:
   - Re-check the newest published GHCR tag AND `origin/main`'s `pyproject.toml`; bump from the higher.
   - Edit **both** `pyproject.toml` and `bioengine/_version.py` in one commit: `chore(release): bump version to X.Y.Z`.
   - Push, `gh pr ready <number>`, wait for the required checks, merge.
5. A draft reports every required check as `completed/skipped` — a green-looking conclusion that ran nothing. While drafting, the evidence is the issue's own harness, never the PR's checks.
6. **Always open the PR immediately after pushing the branch** using the GitHub PAT from `.env` so the user can see and review it without navigating to GitHub manually.
7. **Never merge your own PR.** A pull request is merged only after an **independent** agent has reviewed it and approved — never by the agent that wrote the code, and never on one agent's own judgement. The reviewer re-runs the evidence rather than reading it, and a review that cannot name what would have made the result RED is a refusal. Who plays which part is deployment-specific and is not recorded in this file: it is set by whatever agent team owns the checkout. If no such reviewer exists where you are, the merge is the user's.
8. **Whoever does merge waits for every required check to REPORT.** "No checks reported" means *not yet queued*, not *nothing to wait for* — the run can take over a minute to appear. The PAT does NOT bypass every rule — measured 2026-11-06, `gh pr merge` refuses a branch that is not up to date with `main` outright — but nothing stops a merge issued while a required check is merely absent. A merge that races its own gate is unreviewed no matter what the versions happen to be. The evidence is `gh api repos/<org>/<repo>/commits/<head-sha>/check-runs` returning a non-zero `total_count` with each required check `completed/success`; `gh pr checks` alone does not distinguish *pending* from *absent*. This repo also sets `delete_branch_on_merge: true`, so a check that queues after the merge dies with `fatal: bad object <head-sha>` and leaves a permanent red run that never evaluated anything.
9. **A green `check-version` is a point-in-time comparison, so re-read the versions yourself at merge time.** The check compares the branch against `origin/main` *as it was when the run executed*; it is never re-run when `main` advances, so a branch whose reserved rung `main` has since passed keeps a **stale green** that attests to a comparison which is no longer true. Merging on it passes the PR gate and then fails `docker-publish-worker.yml` on `main`, because that compares against the newest GHCR tag. Before merging, read the branch's `pyproject.toml` version and confirm it is strictly greater than **both** `origin/main`'s and the newest published GHCR tag — the two baselines of the previous bullet. Measured 2026-11-04 with `main` at 0.16.67: 8 of 18 open non-draft PRs reported `check-version: success` while sitting at or below it, the oldest reserving 0.16.52.

For deployment-side PRs (changes under `bioengine/**`, `bioengine/cluster/**`, `bioengine/_app/**`, `bioengine/apps/builder.py`, runtime_env transport, Ray client/server plumbing) the **dev image testing workflow** in `.claude/skills/bioengine-maintainer/SKILL.md` applies: build `<next-version>-devN`, validate on a live cluster, only then mark the PR ready.

## App version bumps

- Always bump `version` in `apps/<name>/manifest.yaml` when app code changes.
- A committed artifact version is **immutable** and its content is exactly what a deploy runs — so a version string must map to one bundle forever. Never delete-and-recreate a version to change its code, and never deploy a *staged* (uncommitted) version; deploy pinned, committed versions only. `upload_app` already rejects any version that isn't strictly greater than every existing one (PEP 440).
- To iterate without inflating the release history, use pre-releases: upload `X.Y.Z-devN`, test, then publish the verified bundle once as `X.Y.Z` and drop the `-dev*` pre-releases. See the user-facing skill's dev-iteration workflow + `scripts/upload_app.py` (`../bioimage.io/public/skills/bioengine/`).
- **Nothing in CI deploys an app — there is no CI deploy path at all.** Agents upload and deploy via the worker API *before* committing, so a commit records a version that is already published, and merging to `main` never changes what is running on a worker. CI only *checks* that: for `model-runner` and `model-finetune`, `app-version-check.yml` requires the committed `manifest.yaml` version to be greater (PEP 440) than `main`'s **and** to already exist in the published artifact's version list.

## Testing apps on the live worker

When working on a BioEngine app, **test by deploying to the live worker** at `bioimage-io/bioengine-worker` on <https://hypha.aicell.io> and calling the service directly. Do not write standalone test scripts for app behaviour. Deploy with a stable `application_id` matching the artifact alias so the service is consistently addressable:

```python
deployed = await worker.deploy_app(
    artifact_id='bioimage-io/my-app',
    version='1.2.3',
    application_id='my-app',   # gives stable service ID, not a random name
)
app_id = deployed['application_id']
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

## `deploy_app` returns a dict — and `version_source` is the bit that matters

`deploy_app` returns `{'application_id', 'artifact_id', 'version', 'version_source'}`, **not** a bare id string. `version_source` says how the deployed version was chosen:

| value | meaning |
|---|---|
| `requested` | you passed `version=` and got exactly that |
| `latest` | you passed no `version` and got the artifact's newest committed version |
| `inherited` | you passed no `version` and got the version the running `application_id` was already on |

`inherited` is the trap. Omitting `version` on an already-running `application_id` redeploys the code that is already there — a deploy that reports success while shipping nothing new. If you upload and then deploy, either pass the version you uploaded or treat `inherited` as a failure:

```python
deployed = await worker.deploy_app(artifact_id='bioimage-io/my-app', application_id='my-app')
if deployed['version_source'] == 'inherited':
    raise RuntimeError(f"redeployed the running {deployed['version']}, not the new code")
```

**Don't guard on `version_source != 'requested'` after passing an explicit `version`.** That branch is unreachable: an explicit version either yields `requested` or raises. `requested` echoes your own argument back — it is not independent confirmation that the artifact served that version.

## `hypha_token` on `deploy_app` — read the parameter carefully

Apps whose code reads `HYPHA_TOKEN` at startup need it injected into the Ray actor; this is done via the `hypha_token` parameter on `deploy_app` (CLI: `--hypha-token $HYPHA_TOKEN`). When `application_id` matches an existing instance, omitting `hypha_token` preserves whatever that instance stored — the token if one was passed, `None` if one was not, and always `None` for an app the worker *adopted* rather than deployed. Preserving `None` injects nothing, and the update then fails exactly like a fresh instance. Passing an empty string is the explicit opposite of omitting: it injects nothing *and* clears whatever the instance had stored, and it is the only way to say that, since an omitted argument and an explicit `None` are indistinguishable over RPC. Pass `hypha_token` on every update that needs one rather than inferring that you can omit it. `get_app_status` reports `recovered_app: true` for an adopted app — after a worker restart, or when the worker attaches to a Ray cluster already running apps — and `true` guarantees no token is stored, but `false` does **not** guarantee one is. The flag is diagnostic, not a decision input. **On a fresh instance (no prior running app with that `application_id`), omitting it injects nothing and the app's `__init__` raises `RuntimeError: HYPHA_TOKEN environment variable is not set`** (or similar). Cross-check the app source before deploying — if `HYPHA_TOKEN` is referenced, pass the parameter. Delivering it as a plain environment variable instead does not work: `--env HYPHA_TOKEN=...` / `application_env_vars` is overwritten by `hypha_token`, which the CLI defaults to `--token`. The plain value only survives when `hypha_token` carries no token — over RPC by omitting it or passing `''`, on the CLI via `--hypha-token ''` — and on that path the worker strips it from its recovery state, so a redeploy of an adopted app loses it. Nothing raises either way, so the app silently runs as the wrong identity.

**`hypha_token=''` is NOT a read-only fence, and which venue you are on decides whether it fences anything at all.** It controls only what the worker *injects*; it cannot take away what the replica *inherits*. On a `--mode single-machine` worker Ray runs inside the worker's own container, so a Serve replica inherits the worker process's environment — and an app reading `os.getenv("HYPHA_TOKEN")` (model-runner does, `apps/model-runner/entry.py`) comes up holding the worker's own credential. Measured 2026-11-14 on europa (#0526): deployed with `hypha_token=''` into a container started with `-e HYPHA_TOKEN`, `get_status()` reported `test_reports_writable: true` and all four replica processes carried the variable in `/proc/<pid>/environ`; an unreviewed app version then overwrote a published report on a shared public artifact. On `--mode=external-cluster` the replica is a KubeRay pod in another namespace which inherits nothing from the worker pod, and there the fence does hold — measured the same day on KTH: `hypha_token=''` came up in `ws-user-anonymouz-…` with `is_anonymous: true, s3_upload_available: false, test_reports_writable: false`. **So on a single-machine worker the fence is the container's environment, not the parameter**: start the validation worker with no `HYPHA_TOKEN`, or with a credential that holds no `rw`/`a` permission on the workspace the app writes to. Narrowing an existing token is not an option (previous paragraph). Assert it rather than assuming — read `is_anonymous` from the app's own `get_status()`, before the call you are fencing. `test_reports_writable` is computed from the permission on **one** workspace, so it reads `false` for a live credential scoped elsewhere and is not the same statement as "no credential"; and an absent key and a `false` value are indistinguishable through `.get()`, so compare `st.get(k, '<ABSENT>') is False`.

**It is per deployment, not per worker.** Each `deploy_app` call carries its own token and each is independently revocable, so many instances of one artifact can hold distinct credentials — deploying several sites with one shared workspace-admin token is a provisioning choice, not a platform limit. What this does *not* buy is a narrower token: Hypha permissions come from workspace scope, so any token that can write a shared artifact in workspace `W` carries `W`'s reach, and scope narrowing is blocked in Hypha (`extra_scopes` is additive, grants nothing, and makes the token unusable for the RPC connection an app opens at startup) rather than in BioEngine. Details and examples: `docs/apps-guide.md` → Per-deployment credentials.

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
| Issue and PR writing style (lede first, then detail) | `.claude/skills/bioengine-maintainer/SKILL.md` |
| BioEngine skills overview (user-facing skill structure) | `.claude/skills/bioengine-maintainer/SKILL.md` |
| End-to-end deploy_app flow (14-step walkthrough) | `docs/deploy_app_flow.md` |
| App authoring (`@bioengine.app`, manifest fields, deploying apps) | `../bioimage.io/public/skills/bioengine/SKILL.md` |
| Hypha platform basics (tokens, workspaces, RPC, artifacts) | <https://hypha.aicell.io/ws/agent-skills/SKILL.md> |
