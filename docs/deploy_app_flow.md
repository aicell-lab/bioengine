# How `deploy_app` works

Step-by-step walkthrough of what the BioEngine worker does between the
moment a Hypha client calls `worker.deploy_app(...)` and the moment the
deployed app's first replica answers a request.

The worker is filesystem-thin. It never downloads source, never writes
zips, and never hands Ray a `file://` URI. Everything that touches the
artifact happens inside Ray tasks the worker submits.

## 1. The RPC arrives

The worker receives the call, validates the caller has admin permission
on this worker, and looks up (or creates) the per-app tracking entry in
`AppsManager._deployed_applications`. The entry is keyed by
`application_id` — the deploy-time name, distinct from the underlying
`artifact_id`.

If an entry already exists, the call is treated as an *update*. The
old `is_deployed` event is cleared so any in-flight RPCs can be
short-circuited, but the entry itself is kept until the new deployment
takes over.

## 2. Manifest load

`artifact_manager.read(artifact_id, version)` returns the manifest dict.
Metadata only — no files yet. The worker validates `format_version`
(must be `0.6.0`) and resolves the entry class id (`entry: "main:MyApp"`
or similar).

If the caller did not pin a version, the manifest read also surfaces
the latest committed version, which the worker stamps into the
deployment record so it can be reproduced exactly on a recovery
restart.

## 3. Mint a short-TTL Hypha download token

The worker calls `server.generate_token({permission: "read", workspace, expires_in: 600})` to mint a 10-minute, read-only, single-workspace
token. It is the only credential the introspect Ray task needs to
reach Hypha. By the time replicas boot, this token has expired —
replicas pull source from Ray's internal package store instead, so
they don't need a Hypha credential at all.

## 4. Submit the introspect Ray task

The worker submits `bioengine._app.bootstrap.introspect_app_in_ray_task`
as a short Ray task. The task's `runtime_env` carries:

- `pip` — the framework baseline (hypha-rpc, pydantic, …) plus the app's
  declared `@bioengine.app(pip=…)` dependencies
- `env_vars` — `BIOENGINE_APP_DIR`, `BIOENGINE_ARTIFACT_DOWNLOAD_URL`
  (Hypha `create-zip-file` URL), `BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN`,
  `BIOENGINE_ARTIFACT_VERSION`

No `py_modules`. The worker doesn't ship the source — the task fetches
it itself.

## 5. The introspect task materialises the source

The task calls `replica_init._ensure_source`, which:

1. Acquires an `fcntl.flock` on `<app_dir>/.lock`. Two replicas booting on
   the same Ray node at the same time block here briefly; the second
   reads the up-to-date `.version` marker written by the first and
   skips the download.
2. Streams `<BIOENGINE_ARTIFACT_DOWNLOAD_URL>?token=<…>` (the Hypha
   `create-zip-file` endpoint) into a tempfile, then extracts it into
   `<app_dir>/source/` — excluding `manifest.yaml`, `README*`, `*.md`,
   `*.ipynb`, `frontend/`, images, `__pycache__/`, and dotfiles.
3. Writes the resolved version into `<app_dir>/.version` so subsequent
   ensure-source calls on the same node can short-circuit.

## 6. The introspect task packages the source into Ray's GCS

Once `source/` exists, the task hashes it with
`ray._private.runtime_env.packaging.get_uri_for_directory` and uploads
it via `upload_package_if_needed` to Ray's content-addressed package
store as `gcs://_ray_pkg_<hash>.zip`.

This is the `app_source_uri` the worker hands the build task and the
replicas later. Source bytes never travel back to the worker — only the
URI does.

## 7. The introspect task walks the type-hint composition graph

With `source/` on `sys.path`, the task imports the user's entry class
and walks its `__init__` type hints to discover composition references
(`def __init__(self, runtime: RuntimeA, batch_size: int = 32)`). It
returns a JSON-compatible `AppSpec` dict containing the class graph,
method schemas, resource requirements, and lifecycle method names.

## 8. The worker checks resources and kwargs

The worker receives `{spec, app_source_uri}` from the introspect task.
It runs `validate_kwargs_against_spec` against the spec, computes the
total resource requirements, and gates against
`AppsManager._check_resources` (on SLURM clusters, this may queue or
reject the deploy if the cluster is saturated).

## 9. Mint the long-lived proxy service token

The worker calls `server.generate_token({workspace, permission: "read_write", expires_in: 30 days})`. This is the credential the
per-app `ProxyDeployment` actor uses to register `workspace/<application_id>`
as a Hypha service and to keep that registration alive across replica
restarts.

## 10. Assemble `proxy_args` and `app_data`

The worker assembles the dict the `ProxyDeployment` actor keeps for the
lifetime of the app:

```python
app_data = {
    "format_version": "0.6.0",
    "entry": "<module>:<ClassName>",
    "spec_hash": "<sha256>",
    "display_name": …, "description": …,
    "artifact_id": …, "version": …,
    "application_kwargs": …, "application_env_vars": …,  # sanitised
    "disable_gpu": …, "max_ongoing_requests": …,
    "application_resources": …,
    "authorized_users": …, "available_methods": …,
    "started_at": …, "last_updated_at": …, "last_updated_by": …,
    "auto_redeploy": …, "debug": …,
}
proxy_args = {
    "application_id": …,
    "app_data": app_data,
    "server_url": …, "workspace": …,
    "proxy_service_token": <30-day token from step 9>,
    "authorized_users": …,
    ...
}
```

`app_data` is what `AppsManager.recover_deployed_applications` reads
back from the live actor on a worker restart, so it has to be
self-describing.

## 11. Submit `build_and_run_application`

The worker submits a second Ray task,
`bioengine._app.bootstrap.build_and_run_application`. Its `runtime_env`
carries:

- `pip` — same baseline + app deps as in step 4
- `env_vars` — same as step 4 plus `BIOENGINE_APP_SOURCE_URI` (the
  `gcs://…` URI from step 6)

No `py_modules`. The task calls `replica_init._ensure_source` first —
on shared-filesystem clusters this no-ops because `<app_dir>/source/`
already exists from step 5; on non-shared clusters it pulls the bytes
back via `download_and_unpack_package` against the Ray-GCS URI. Then
it imports the user classes, calls `cls.bind(…)` to assemble the Ray
Serve graph (composition handles wired up via the type hints), wraps
the entry deployment in `ProxyDeployment`, and calls
`serve.run(blocking=False)`.

The `blocking=False` is intentional: the task returns immediately,
the worker's RPC reply happens shortly after, and Ray Serve continues
spinning up replicas asynchronously. The client knows the call
succeeded as soon as the build task returned, not when replicas
finished warming up.

## 12. Ray Serve schedules replicas

For each `Deployment` in the bound graph, Ray Serve allocates one or
more actor processes on Ray nodes that satisfy the resource
requirements. Each actor's `runtime_env` carries the same `env_vars`
the build task had, including `BIOENGINE_APP_SOURCE_URI`. No
`py_modules` is set at this layer — the source materialisation is
the meta-path finder's job, next step.

## 13. Replica startup and `sys.path`

A Ray Serve replica process starts. Ray runtime_env_agent installs
the `pip` deps and sets the env vars, then hands off to Ray Serve's
replica wrapper. That wrapper calls
`cloudpickle.loads(serialized_deployment_def)` to reconstitute the
user's deployment class. cloudpickle tries to import the user's
`module:qualname`. The standard `sys.meta_path` finders look on
`sys.path` and find nothing (because the source hasn't been
materialised yet) — and `ImportError` propagates.

At this point a `sys.meta_path` finder installed by
`bioengine/__init__.py` at framework-import time catches the failed
lookup. The finder:

1. Calls `replica_init.setup_replica_environment`, which calls
   `_ensure_source` — on Ray Serve replicas, the URI backend is
   `BIOENGINE_APP_SOURCE_URI` (the `gcs://` URI), so the source is
   pulled from Ray's package store with no Hypha auth.
2. Prepends `<app_dir>/source/` to `sys.path`.
3. Re-delegates to the standard import machinery, which now finds
   the module.

cloudpickle's import succeeds, the class is reconstituted, and
`Deployment.__init__` runs.

The finder fires at most once per replica process. Subsequent
imports use the populated `sys.path` directly.

## 14. ProxyDeployment registers the Hypha service

The per-app `ProxyDeployment` replica calls
`server.register_service` using the 30-day proxy service token. From
here, all client RPCs to `workspace/<application_id>` route through
this proxy. Per-replica WebSocket service IDs are derived under it.

## What lives where

| Concern | Code |
|---|---|
| Worker-side orchestration (steps 1–4, 8–11) | [`bioengine/apps/builder.py`](../bioengine/apps/builder.py) |
| Introspect + build Ray tasks (steps 4–7, 11) | [`bioengine/_app/bootstrap.py`](../bioengine/_app/bootstrap.py) |
| Replica + build-task source materialisation (steps 5, 11, 13) | [`bioengine/_app/replica_init.py`](../bioengine/_app/replica_init.py) |
| `sys.meta_path` finder (step 13) | [`bioengine/__init__.py`](../bioengine/__init__.py) |
| `_deployed_applications` tracking, recovery, monitoring | [`bioengine/apps/manager.py`](../bioengine/apps/manager.py) |
| ProxyDeployment actor (step 14) | [`bioengine/apps/proxy_deployment.py`](../bioengine/apps/proxy_deployment.py) |
