# `deploy_app` flow: v0.11.3 vs v0.11.4

A side-by-side walkthrough of what the BioEngine worker does between the
moment a Hypha client calls `worker.deploy_app(...)` and the moment the
deployed app's first replica answers a request. The two versions reach the
same outcome through very different code paths; this document is the
reference for why the v0.11.4 redesign happened and what changed underneath
the public API.

The public API is unchanged. App authors do not need to migrate anything.

## Why v0.11.4

v0.11.3 worked only when the worker pod and the Ray cluster nodes shared
a filesystem. On the KTH KubeRay cluster they do not — the bioengine
worker runs in the `hypha` Kubernetes namespace with its own Trident NFS
export, the Ray pods run in `ray-cluster` namespace with a different
Trident NFS export, and the `file://` URIs the worker handed Ray Serve
pointed at paths the Ray nodes could not see. v0.11.4 removes the shared
filesystem assumption by making the worker filesystem-thin: it never
downloads source, never writes zips, and never hands Ray a `file://`
URI. Everything happens inside Ray tasks the worker submits.

## The flow, step by step

### 1. `deploy_app()` RPC arrives at the worker

| | v0.11.3 | v0.11.4 |
|---|---|---|
| **Where** | worker process | worker process |
| **Action** | Receives the call, looks up or creates the per-app tracking entry in `AppsManager._deployed_applications`. | Identical. |

No difference.

### 2. Manifest load

| | v0.11.3 | v0.11.4 |
|---|---|---|
| **Where** | worker process | worker process |
| **Action** | `artifact_manager.read(artifact_id, version)` returns the manifest dict. Metadata only — no files. | Identical. |

No difference.

### 3. Mint short-TTL Hypha download token

| | v0.11.3 | v0.11.4 |
|---|---|---|
| **Where** | — | worker process |
| **Action** | Not done. The worker authenticates via its long-lived `HYPHA_TOKEN` for the per-file downloads below. | `server.generate_token({permission: "read", workspace, expires_in: 600})` mints a 10-minute, read-only, single-workspace token. Passed to the introspect Ray task. By the time replicas boot the token has expired — they pull source from Ray's internal package store instead. |

### 4. Materialise the artifact source

| | v0.11.3 | v0.11.4 |
|---|---|---|
| **Where** | worker pod's local PVC | inside the introspect Ray task on the Ray cluster |
| **Action** | `_materialize_artifact` walks the artifact tree and downloads every file via per-file `artifact_manager.get_file()` to `<apps_workdir>/<app_id>/source/`. Accumulates on the worker's PVC across deploys. | The Ray task calls `replica_init._ensure_source` with `BIOENGINE_ARTIFACT_DOWNLOAD_URL` (Hypha `create-zip-file` endpoint) and the short-TTL token. One HTTP GET, one zip, extracted into `<app_dir>/source/` on the Ray node's filesystem. `fcntl.flock` on `<app_dir>/.lock` serialises concurrent same-node starts. |

### 5. Package the source for Ray

| | v0.11.3 | v0.11.4 |
|---|---|---|
| **Where** | worker pod | inside the introspect Ray task |
| **Action** | `_write_pkg_to_runtime_env_dir` walks the downloaded tree, content-hashes it (SHA-256), zips it (excluding `manifest.yaml`, `README*`, `*.md`, `*.ipynb`, `frontend/`, etc.) into `<apps_workdir>/_runtime_env_packages/bioengine_pkg_<hash>.zip`. Same again for the worker's own `bioengine/` source → `bioengine_runtime_<hash>.zip` (with a `_bioengine_wrap/` arc-prefix so Ray's `remove_top_level_directory=True` doesn't strip `bioengine/`). Both files are referenced by `file://` URIs in `runtime_env.py_modules`. | `ray._private.runtime_env.packaging.get_uri_for_directory` + `upload_package_if_needed` hash the source tree and upload it to Ray's content-addressed GCS as `gcs://_ray_pkg_<hash>.zip`. The URI is what's returned to the worker as `app_source_uri`. The worker never sees the bytes. |

### 6. Introspect the user's class graph

| | v0.11.3 | v0.11.4 |
|---|---|---|
| **Where** | a Ray task that runs in the app's `runtime_env` (so `pip` deps are installed) | the same Ray task that did steps 4 + 5 |
| **Action** | Worker submits `introspect_app` as a separate Ray task with `runtime_env={py_modules: [bioengine_runtime_uri, bioengine_pkg_uri], pip: app_pip}`. The task imports the user's entry class, walks `__init__` type hints for composition references, and returns a JSON spec. | The same introspect task that downloaded + GCS-packaged the source then walks the type-hint graph in-process via the same `introspect_app` helper. Returns `{spec, app_source_uri}` so the worker has the spec and the URI for the build task. |

The shape of the returned `AppSpec` is identical between versions.

### 7. Validate kwargs + check resources

| | v0.11.3 | v0.11.4 |
|---|---|---|
| **Where** | worker process | worker process |
| **Action** | `validate_kwargs_against_spec` + `_check_resources` against the returned spec. | Identical. |

No difference.

### 8. Mint the proxy service token

| | v0.11.3 | v0.11.4 |
|---|---|---|
| **Where** | worker process | worker process |
| **Action** | `server.generate_token({workspace, permission: "read_write", expires_in: 30 days})` so the per-app ProxyDeployment can re-register its Hypha service across replica restarts. | Identical. |

### 9. Build `proxy_args` + `app_data`

| | v0.11.3 | v0.11.4 |
|---|---|---|
| **Where** | worker process | worker process |
| **Action** | Assemble the dict the `ProxyDeployment` will keep around to identify the app: `display_name`, `description`, `artifact_id`, `version`, `application_kwargs`, `application_env_vars`, `disable_gpu`, `max_ongoing_requests`, `application_resources`, `authorized_users`, `available_methods`, `started_at`, `last_updated_at`, `last_updated_by`, `auto_redeploy`, `debug`. | Identical structure. |

### 10. Submit `build_and_run_application`

| | v0.11.3 | v0.11.4 |
|---|---|---|
| **Where** | worker submits, runs on the Ray head | worker submits, runs on the Ray head |
| **Action** | Ray task launched with `runtime_env={py_modules: [bioengine_runtime_uri, bioengine_pkg_uri], pip}`. The task imports the user class, calls `cls.bind(...)` to assemble the Ray Serve graph (composition handles wired up via type hints), wraps the entry deployment in `ProxyDeployment`, and calls `serve.run(blocking=False)`. | Ray task launched with `runtime_env={env_vars: {BIOENGINE_APP_SOURCE_URI: …, BIOENGINE_APP_DIR: …, …}, pip}`. The task calls `replica_init._ensure_source` first — on shared-FS clusters this no-ops because `<app_dir>/source/` already exists from step 4; on non-shared clusters it pulls from Ray's GCS using the URI. Then the same `cls.bind` + `serve.run` as v0.11.3. |

The `serve.run` call returns immediately because `blocking=False`. The
worker's RPC reply happens *after* the build task returns, while Ray
Serve continues to spin up replicas asynchronously.

### 11. Ray Serve schedules replicas

| | v0.11.3 | v0.11.4 |
|---|---|---|
| **Where** | each replica = a new actor process on some Ray node | identical |
| **Action** | Each actor's `runtime_env` carries the same two `file://` URIs from step 5. Ray's `runtime_env_agent` on each node runs `download_and_unpack_package` against those URIs — which on `file://` schemes opens the path directly. If the worker pod's `<apps_workdir>` is not visible on this Ray node, this is `FileNotFoundError`. Replica `__init__` aborts; Ray Serve retries up to 3 times then marks the deployment `DEPLOY_FAILED`. | Each actor's `runtime_env` carries `env_vars: {BIOENGINE_APP_SOURCE_URI, BIOENGINE_APP_DIR, …}`. No `py_modules` URI for the user source. Ray's standard runtime_env setup does not materialise the source — that's the meta-path finder's job. |

This is the step that fails on KTH under v0.11.3.

### 12. User source on `sys.path`

| | v0.11.3 | v0.11.4 |
|---|---|---|
| **Where** | each replica process | each replica process |
| **Action** | Ray's `runtime_env_agent` already extracted the zip and prepended the unzip directory to `sys.path`. `cloudpickle.loads(serialized_deployment_def)` then resolves the user's `module:qualname` reference, finds the module on `sys.path`, and instantiates the class. | `bioengine/__init__.py` installs a `sys.meta_path` finder at import time. When `cloudpickle.loads` tries to import the user's module and the standard import machinery returns `ImportError`, the meta-path finder catches that, calls `replica_init.setup_replica_environment` (which calls `_ensure_source` to materialise the source via the `BIOENGINE_APP_SOURCE_URI` GCS download), and prepends `<app_dir>/source/` to `sys.path`. The original import then succeeds. |

The meta-path finder is what makes the cloudpickle round-trip work
without the user's module having to import `bioengine` itself first.
v0.11.3 didn't need this because Ray Serve put the source on `sys.path`
before any user code ran.

### 13. ProxyDeployment registers the Hypha service

| | v0.11.3 | v0.11.4 |
|---|---|---|
| **Where** | the per-app ProxyDeployment replica | identical |
| **Action** | Uses the long-lived proxy service token (step 8) to register `workspace/<application_id>` as a Hypha service. From here, all client RPCs route through this proxy. | Identical. |

## Side-by-side summary

| Step | v0.11.3 location | v0.11.4 location | Same? |
|---|---|---|---|
| 1. `deploy_app()` arrives | worker | worker | ✓ |
| 2. Manifest load | worker | worker | ✓ |
| 3. Mint download token | — | worker | new in 0.11.4 |
| 4. Download artifact source | worker (per-file) → worker PVC | introspect Ray task (single zip) → Ray node PVC | moved + simplified |
| 5. Hash + package source | worker → `_runtime_env_packages/*.zip` (file://) | introspect Ray task → Ray GCS (gcs://) | moved |
| 6. Introspect user classes | separate Ray task with `py_modules=[…]` | same introspect Ray task | merged |
| 7. Validate kwargs + resources | worker | worker | ✓ |
| 8. Mint proxy service token | worker | worker | ✓ |
| 9. Assemble proxy_args + app_data | worker | worker | ✓ |
| 10. Submit `build_and_run_application` | Ray task with `py_modules=[…]` | Ray task with `env_vars={BIOENGINE_APP_SOURCE_URI}` | same task, different inputs |
| 11. Ray Serve spawns replicas | each replica's runtime_env pulls source from `file://` URI | each replica's runtime_env carries env_vars only; no source pull at runtime_env layer | the bug-fix |
| 12. User source on `sys.path` | Ray's runtime_env_agent puts it there before any user code runs | bioengine's `sys.meta_path` finder puts it there on the first `ImportError` cloudpickle hits | new mechanism |
| 13. ProxyDeployment registers Hypha service | per-app proxy replica | per-app proxy replica | ✓ |

## What this fixes and what it costs

**Fixes:**

- Cluster topology assumption is gone. v0.11.4 works on KubeRay clusters
  where the worker pod and the Ray pods are in different Kubernetes
  namespaces with different PVCs, on single-machine deployments where
  the worker shells out to a local Ray, on SLURM clusters with one
  shared `$HOME`, and on any other topology where Hypha and Ray's GCS
  are reachable.
- The worker's PVC stops accumulating per-deploy artifact downloads
  and `_runtime_env_packages/*.zip`. The PVC contains only logs and
  per-app tracking dirs that survive across version bumps.
- No more "the worker's `bioengine_runtime_*.zip` URI from the previous
  deploy was reused but the file is on the wrong PVC" partial failures.

**Costs:**

- The introspect Ray task is now responsible for the download. Failure
  surface is different — a network blip between the Ray cluster and
  Hypha now fails the task; previously it would have failed the worker
  RPC.
- A meta-path finder runs on every replica import that fails through
  the standard finders. Negligible overhead in practice (one extra
  `find_spec` call per failed import), but it's new code in the hot
  path for `cloudpickle.loads`.
- The short-TTL download token is one extra `generate_token` RPC per
  `deploy_app` call.

## Pointers

- The introspect + build Ray tasks live in
  [`bioengine/_app/bootstrap.py`](../bioengine/_app/bootstrap.py).
- The replica-side source materialisation lives in
  [`bioengine/_app/replica_init.py`](../bioengine/_app/replica_init.py).
- The meta-path finder is installed in
  [`bioengine/__init__.py`](../bioengine/__init__.py).
- The worker-side orchestration (manifest load, token mint, task
  submission) lives in [`bioengine/apps/builder.py`](../bioengine/apps/builder.py).
