"""Build BioEngine apps for Ray Serve from v0.6.0 artifacts.

In the v0.11.4 model the worker never imports user code AND never touches
the filesystem. The worker only handles:

* manifest loading + ``format_version`` / ``authorized_users`` gate
* runtime_env composition (base pip + env_vars for the introspect task)
* mint short-TTL Hypha download token
* submit :func:`bioengine._app.bootstrap.introspect_app_in_ray_task` —
  that Ray task downloads source, walks the type-hint composition graph,
  packages source to Ray's internal GCS, returns ``{spec, app_source_uri}``
* kwargs validation against the returned spec
* resource accounting + ``_check_resources`` (the SLURM-aware gate)
* authorisation-rule resolution + ``ProxyDeployment`` argument assembly
* submit :func:`bioengine._app.bootstrap.build_and_run_application` — that
  Ray task does ``cls.bind(...)`` + ``serve.run(blocking=False)``

Every replica receives ``BIOENGINE_APP_SOURCE_URI`` in its ``env_vars``
and uses Ray-internal GCS to materialise source — no Hypha auth on the
replica side, so the short-TTL token never matters after step 4.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect as _inspect
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ray
import yaml
from hypha_rpc.rpc import RemoteService
from hypha_rpc.utils import ObjectProxy
from ray import serve
from ray._private.runtime_env.packaging import get_uri_for_directory

import bioengine
from bioengine._app.bootstrap import (
    SPEC_FORMAT_VERSION,
    _ensure_jsonable,
    build_and_run_application,
    introspect_app_in_ray_task,
    validate_kwargs_against_spec,
)
from bioengine._app.errors import BioEngineUserError
from bioengine.apps.proxy_deployment import ProxyDeployment
from bioengine.utils import (
    create_logger,
    get_pip_requirements,
    update_requirements,
    validate_manifest,
)

#: TTL of the short-lived read-only token the worker mints per deploy and
#: hands the replica setup hook for its create-zip-file download. Bounded
#: well above the time it takes a typical Ray Serve replica to pull the
#: ~MB-scale artifact and well under any plausible session-reuse window.
_DOWNLOAD_TOKEN_TTL_SECONDS = 600


class AppBuilder:
    """Build BioEngine apps from artifact storage for Ray Serve.

    See module docstring for the v0.6.0 design. The public surface
    (``__init__``, ``complete_initialization``, ``update_data_server_url``,
    ``build``) is unchanged from v0.5; the implementation has been replaced.
    """

    def __init__(
        self,
        apps_workdir: Union[str, Path],
        server_url: str,
        log_file: Optional[str] = None,
        proxy_actor_name: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        self.logger = create_logger(
            name="AppBuilder",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )
        self.apps_workdir = Path(apps_workdir)
        self.server: Optional[RemoteService] = None
        self.artifact_manager: Optional[ObjectProxy] = None
        self.worker_service_id: Optional[str] = None
        self.proxy_actor_name: Optional[str] = proxy_actor_name
        self.server_url: str = server_url
        self.data_server_url: Optional[str] = None

    def complete_initialization(
        self,
        server: RemoteService,
        artifact_manager: ObjectProxy,
        worker_service_id: str,
    ) -> None:
        self.server = server
        self.artifact_manager = artifact_manager
        self.worker_service_id = worker_service_id

    def update_data_server_url(self, data_server_url: str) -> None:
        self.data_server_url = data_server_url

    # ────────────────────────── manifest ─────────────────────────────────

    async def _load_manifest(
        self, artifact_id: str, version: Optional[str] = None
    ) -> tuple[Dict[str, Any], Optional[str]]:
        """Load the manifest from the artifact (or local dev path) and validate."""
        manifest: Optional[Dict[str, Any]] = None
        resolved_version: Optional[str] = version

        if os.environ.get("BIOENGINE_LOCAL_ARTIFACT_PATH"):
            artifact_folder = artifact_id.split("/")[1]
            local_path = (
                Path(os.environ["BIOENGINE_LOCAL_ARTIFACT_PATH"])
                / artifact_folder
                / "manifest.yaml"
            )
            if local_path.exists():
                with open(local_path, "r") as f:
                    manifest = yaml.safe_load(f)
            else:
                self.logger.warning(
                    f"Local manifest file not found: {local_path}. "
                    f"Fetching from remote artifact manager."
                )

        if manifest is None:
            artifact = await self.artifact_manager.read(artifact_id, version=version)
            manifest = artifact.get("manifest")
            if manifest is None:
                raise ValueError(f"Manifest not found in artifact {artifact_id}.")
            if version is None:
                versions = artifact.get("versions") or []
                if versions:
                    latest = max(versions, key=lambda v: v["created_at"])
                    resolved_version = latest["version"]

        validate_manifest(manifest)
        return manifest, resolved_version

    # ─────────────────────── runtime_env compose ─────────────────────────

    def _build_env_vars(
        self,
        application_id: str,
        artifact_id: str,
        version: Optional[str],
        non_secret_env_vars: Dict[str, str],
        secret_env_vars: Dict[str, str],
        hypha_token: Optional[str],
        download_token: Optional[str],
    ) -> Dict[str, str]:
        """Compose the env_vars dict the replica's runtime_env will receive.

        The replica setup hook (``bioengine._app.replica_init``) reads
        ``BIOENGINE_APP_DIR`` + ``BIOENGINE_ARTIFACT_DOWNLOAD_URL`` +
        ``BIOENGINE_ARTIFACT_VERSION`` + ``BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN``
        from this dict to materialise the user source. HOME and TMPDIR are
        deliberately *not* set here — the setup hook computes them under
        ``BIOENGINE_APP_DIR`` so they survive app version bumps and stay
        per-app-per-node (shared across replicas of the same app on the
        same node by design; see :doc:`v0.11.4 release notes`).
        """
        env_vars: Dict[str, str] = {}
        env_vars.update(non_secret_env_vars)
        for key, value in secret_env_vars.items():
            env_vars[f"_BIOENGINE_SECRET_{key}"] = value

        if hypha_token is not None:
            env_vars["_BIOENGINE_SECRET_HYPHA_TOKEN"] = hypha_token

        worker_workspace = (
            self.server.config.workspace if self.server is not None else "local"
        )
        # Worker-workspace prefix so two workers serving the same Ray
        # cluster but different Hypha workspaces don't share state on disk.
        app_dir_name = f"{worker_workspace}-{application_id}"
        app_dir = self.apps_workdir / app_dir_name
        env_vars["BIOENGINE_APP_DIR"] = str(app_dir)

        if self.server is not None:
            env_vars["HYPHA_SERVER_URL"] = self.server_url
            env_vars["HYPHA_WORKSPACE"] = worker_workspace
        env_vars["HYPHA_ARTIFACT_ID"] = artifact_id
        env_vars["BIOENGINE_ARTIFACT_ID"] = artifact_id
        if version is not None:
            env_vars["HYPHA_ARTIFACT_VERSION"] = version
            env_vars["BIOENGINE_ARTIFACT_VERSION"] = version
            artifact_workspace, alias = artifact_id.split("/", 1)
            env_vars["BIOENGINE_ARTIFACT_DOWNLOAD_URL"] = (
                f"{self.server_url.rstrip('/')}/{artifact_workspace}"
                f"/artifacts/{alias}/create-zip-file?version={version}"
            )
            # Files-listing endpoint (metadata only) — the introspect uses it
            # to compute a content signature and skip the full re-download when
            # the artifact is unchanged.
            env_vars["BIOENGINE_ARTIFACT_FILES_URL"] = (
                f"{self.server_url.rstrip('/')}/{artifact_workspace}"
                f"/artifacts/{alias}/files"
            )
            if download_token:
                env_vars["BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN"] = download_token

        if self.worker_service_id:
            env_vars["BIOENGINE_WORKER_SERVICE_ID"] = self.worker_service_id
        if self.proxy_actor_name:
            env_vars["BIOENGINE_PROXY_ACTOR_NAME"] = self.proxy_actor_name
        if self.data_server_url:
            env_vars["BIOENGINE_DATA_SERVER_URL"] = self.data_server_url
        env_vars["BIOENGINE_APPLICATION_ID"] = application_id

        for key, value in list(env_vars.items()):
            if not isinstance(value, str):
                env_vars[key] = str(value)
        return env_vars

    @staticmethod
    def _bioengine_gcs_uri() -> str:
        """Content-hashed ``gcs://_ray_pkg_<hash>.zip`` URI for the worker's
        ``bioengine`` package.

        :func:`get_uri_for_directory` is a pure-Python hash — no upload
        happens here. The bytes were already pushed to the Ray cluster's
        internal GCS by the worker's ``ray.init`` job-level py_modules
        (:meth:`bioengine.cluster.ray_cluster.RayCluster.connect`); we
        just re-derive the same URI so Ray Serve replica runtime_envs
        can point at the cached package without going back through the
        ray-client gRPC bridge that wedges on per-deploy uploads.

        ``include_gitignore`` switched from optional to required between
        Ray patch versions; the inspect gate keeps us cross-compatible.
        """
        import inspect

        bioengine_dir = str(Path(bioengine.__file__).parent.resolve())
        kwargs: Dict[str, Any] = {}
        if "include_gitignore" in inspect.signature(get_uri_for_directory).parameters:
            kwargs["include_gitignore"] = False
        return get_uri_for_directory(bioengine_dir, **kwargs)

    def _build_submit_runtime_env(
        self,
        env_vars: Dict[str, str],
    ) -> Dict[str, Any]:
        """Compose the runtime_env for the ``build_and_run_application`` Ray task.

        No ``py_modules`` is set here: the task imports ``bioengine`` via
        Ray's per-field inheritance of the worker's job-level py_modules
        (set by :meth:`bioengine.cluster.ray_cluster.RayCluster.connect`).
        It materialises the user app source itself via
        :func:`bioengine._app.replica_init._ensure_source` using the
        ``BIOENGINE_*`` env_vars below — same flow as Ray Serve replicas.

        Task-level ``py_modules`` would need a ``gcs://`` URI Ray could
        accept; producing one requires ``upload_package_if_needed``, which
        in external-cluster mode wedges the worker's asyncio loop talking
        to the ray-client gRPC bridge. Hypha downloads sidestep that.

        ``pip`` is the BioEngine baseline only — every module containing
        ``@bioengine.app`` (and anything they transitively import at top
        level) is required to be importable with just ``bioengine[worker]``
        and the standard library. Heavy application dependencies belong in
        the decorator's ``pip=`` arg (where Ray Serve installs them once
        per replica) and are imported lazily inside method bodies or from
        sibling modules that aren't imported during introspection.
        """
        base_pip = update_requirements(
            [],
            select=["httpx", "hypha-rpc", "pydantic"],
            extras=["worker"],
        )
        task_env = {
            k: v for k, v in env_vars.items() if not k.startswith("_BIOENGINE_SECRET_")
        }
        return {
            "pip": base_pip,
            "env_vars": task_env,
        }

    async def _introspect_via_ray_task(
        self,
        entry_id: str,
        env_vars: Dict[str, str],
        submit_runtime_env: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Submit :func:`introspect_app_in_ray_task` as a Ray task.

        The task downloads the user package (Hypha, short-TTL token in
        ``env_vars``), walks the type-hint composition graph, packages
        the source to Ray's internal GCS, and returns the
        ``{spec, app_source_uri}`` payload. We never touch the worker's
        filesystem.

        Strips the ``_BIOENGINE_SECRET_*`` keys from ``env_vars`` before
        passing into the task to keep secrets out of Ray's logs; the
        introspect task only needs the download URL/token + APP_DIR +
        VERSION, none of which is secret-prefixed.
        """
        task_env_vars = {
            k: v for k, v in env_vars.items() if not k.startswith("_BIOENGINE_SECRET_")
        }
        try:
            return await asyncio.to_thread(
                ray.get,
                # max_calls=1 forces a fresh worker process per introspect:
                # Ray pools task workers, and _load_app_class imports the user
                # modules into that process's sys.modules. A reused worker
                # would return a stale cached module (Python resolves imports
                # from sys.modules before sys.path), so the spec — and the
                # by-value-pickled deployment class — would capture old code
                # even after the on-disk source is refreshed.
                ray.remote(num_cpus=0, max_calls=1, runtime_env=submit_runtime_env)(
                    introspect_app_in_ray_task
                ).remote(entry_id, task_env_vars),
            )
        except ray.exceptions.RayTaskError as exc:
            cause = exc.cause if isinstance(exc.cause, Exception) else exc
            raise BioEngineUserError(str(cause)) from exc

    # ───────────────────────── kwargs translation ────────────────────────

    @staticmethod
    def _translate_kwargs_keys(
        spec: Dict[str, Any], kwargs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Accept ``{ClassName: ...}`` or ``{module:qualname: ...}`` and
        return the canonical ``{module:qualname: ...}`` form.

        Class names must be unambiguous; ambiguous names raise so the user
        switches to the full form.
        """
        if not kwargs:
            return {}

        by_qualname: Dict[str, Optional[str]] = {}
        for cid, meta in spec["classes"].items():
            q = meta["qualname"]
            by_qualname[q] = cid if q not in by_qualname else None

        translated: Dict[str, Dict[str, Any]] = {}
        for key, value in kwargs.items():
            if ":" in key:
                if key not in spec["classes"]:
                    raise BioEngineUserError(
                        f"Unknown class id {key!r}. Known: {sorted(spec['classes'])}."
                    )
                translated[key] = value
                continue
            target = by_qualname.get(key)
            if target is None:
                if key in by_qualname:
                    raise BioEngineUserError(
                        f"Class name {key!r} is ambiguous in this app. "
                        f"Use the full 'module:qualname' form."
                    )
                raise BioEngineUserError(
                    f"Unknown class name {key!r}. Known: "
                    f"{sorted(q for q in by_qualname if by_qualname[q] is not None)}."
                )
            translated[target] = value
        return translated

    # ──────────────────────────── resources ──────────────────────────────

    @staticmethod
    def _sum_resources(
        spec: Dict[str, Any], proxy_memory_in_gb: float
    ) -> Dict[str, int]:
        totals: Dict[str, Union[int, float]] = {"num_cpus": 0, "num_gpus": 0, "memory": 0}
        for meta in spec["classes"].values():
            opts = meta.get("ray_actor_options", {})
            totals["num_cpus"] += opts.get("num_cpus", 0)
            totals["num_gpus"] += opts.get("num_gpus", 0)
            totals["memory"] += opts.get("memory", 0)
        # ProxyDeployment overhead — CPU/GPU come from the decorator;
        # memory reservation is the deploy-time ``proxy_memory_in_gb``.
        proxy_opts = getattr(ProxyDeployment, "ray_actor_options", {}) or {}
        totals["num_cpus"] += proxy_opts.get("num_cpus", 0)
        totals["num_gpus"] += proxy_opts.get("num_gpus", 0)
        totals["memory"] += int(proxy_memory_in_gb * (1024**3))
        return {k: int(v) if isinstance(v, (int, float)) else v for k, v in totals.items()}

    # ────────────────────────── auth resolution ──────────────────────────

    @staticmethod
    def _resolve_authorized_users(
        manifest: Dict[str, Any],
        override: Optional[Union[Dict[str, List[str]], List[str]]],
        deploying_user: Optional[tuple],
        admin_users: Optional[List[str]],
    ) -> Dict[str, List[str]]:
        """Merge deploy-time override → manifest → injected admin/deploying user."""
        if override is not None:
            effective = override if isinstance(override, dict) else {"*": override}
        else:
            users = manifest.get("authorized_users", ["*"])
            effective = users if isinstance(users, dict) else {"*": users}

        deploying_entries: List[str] = []
        if deploying_user:
            dep_id, dep_email = deploying_user
            deploying_entries = [v for v in [dep_id, dep_email] if v]

        for key in list(effective):
            rule = list(effective[key])
            if "*" not in rule:
                additions = list(deploying_entries)
                if admin_users:
                    additions.extend(admin_users)
                for user in additions:
                    if user not in rule:
                        rule.append(user)
            seen: set[str] = set()
            effective[key] = [u for u in rule if not (u in seen or seen.add(u))]

        if "*" not in effective:
            fallback = list(deploying_entries)
            if admin_users:
                fallback.extend(admin_users)
            seen2: set[str] = set()
            effective["*"] = [u for u in fallback if not (u in seen2 or seen2.add(u))]
        return effective

    # ──────────────────────────── env sanitiser ──────────────────────────

    @staticmethod
    def _sanitize_recovery_env_vars(
        application_env_vars: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, str]]:
        """Strip secret-like keys so they don't end up in proxy app_data."""
        out: Dict[str, Dict[str, str]] = {}
        for cls_key, env in application_env_vars.items():
            out[cls_key] = {
                k: v
                for k, v in env.items()
                if not k.startswith("_") and k != "HYPHA_TOKEN"
            }
        return out

    # ────────────────────────────── build ────────────────────────────────

    async def build(
        self,
        application_id: str,
        artifact_id: str,
        version: str,
        application_kwargs: Dict[str, Dict[str, Any]],
        application_env_vars: Dict[str, Dict[str, Any]],
        hypha_token: Optional[str],
        disable_gpu: bool,
        max_ongoing_requests: int,
        proxy_memory_in_gb: float,
        debug: bool,
        started_at: Optional[float] = None,
        last_updated_at: Optional[float] = None,
        last_updated_by: Optional[str] = None,
        auto_redeploy: bool = False,
        ice_servers: Optional[List[Dict[str, Any]]] = None,
        authorized_users: Optional[
            Union[Dict[str, List[str]], List[str]]
        ] = None,
        deploying_user: Optional[tuple] = None,
        admin_users: Optional[List[str]] = None,
        scaling: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> "BuiltApp":
        """Construct the Ray Serve application for a v0.6.0 BioEngine app."""
        self.logger.info(
            f"Building application '{application_id}' from artifact "
            f"'{artifact_id}' (version: {version})"
        )

        manifest, resolved_version = await self._load_manifest(artifact_id, version)
        version = resolved_version
        entry_id = manifest["entry"]
        self.logger.info(
            f"Resolved application '{application_id}' artifact '{artifact_id}' "
            f"to version '{version}', entry '{entry_id}'."
        )

        # Default per-deployment dicts to empty so downstream stays consistent.
        application_kwargs = dict(application_kwargs or {})
        application_env_vars = dict(application_env_vars or {})

        # 1. Mint a short-TTL read-only token the introspect Ray task will
        # attach to its single Hypha create-zip-file download. The token
        # never crosses to Ray Serve replicas — they pull source from Ray's
        # internal GCS via ``BIOENGINE_APP_SOURCE_URI`` (uploaded inside the
        # introspect task), so its TTL only matters for the next ~10 s.
        download_token: Optional[str] = None
        artifact_workspace = artifact_id.split("/", 1)[0]
        try:
            download_token = await self.server.generate_token(
                {
                    "workspace": artifact_workspace,
                    "permission": "read",
                    "expires_in": _DOWNLOAD_TOKEN_TTL_SECONDS,
                }
            )
        except Exception as exc:
            # The common expected case: this worker's token has no permission
            # on the artifact's workspace (e.g. a personal-workspace worker
            # deploying a public artifact from a shared collection). Hypha
            # wraps the server-side PermissionError in a RemoteException, so
            # we match on the message text — it's stable across recent Hypha
            # versions and the alternative (matching on RemoteError attrs) is
            # more fragile. Public artifacts still work via anonymous read.
            if "any permission for workspace" in str(exc):
                self.logger.info(
                    f"No cross-workspace grant on '{artifact_workspace}'; "
                    f"downloading '{artifact_id}' anonymously "
                    f"(works for public artifacts)."
                )
            else:
                self.logger.warning(
                    f"generate_token for '{artifact_workspace}' failed ({exc}); "
                    "proceeding without a download token (works for public "
                    "artifacts)."
                )

        # 2. Compose env_vars + the introspect-task runtime_env.
        flat_env_vars: Dict[str, str] = {}
        for env_dict in application_env_vars.values():
            for k, v in env_dict.items():
                flat_env_vars[k] = v
        secret_env_vars = {
            k[1:]: v for k, v in flat_env_vars.items() if k.startswith("_")
        }
        non_secret_env_vars = {
            k: v for k, v in flat_env_vars.items() if not k.startswith("_")
        }
        env_vars = self._build_env_vars(
            application_id, artifact_id, version,
            non_secret_env_vars, secret_env_vars, hypha_token, download_token,
        )
        runtime_env = self._build_submit_runtime_env(env_vars)

        # 3. Introspect the user package via a Ray task — the task
        # downloads from Hypha, walks @bioengine.app composition, and
        # packages the source to Ray-internal GCS. Returns spec + URI.
        introspect_result = await self._introspect_via_ray_task(
            entry_id, env_vars, runtime_env
        )
        spec = introspect_result["spec"]
        app_source_uri = introspect_result["app_source_uri"]
        self.logger.info(
            f"Introspect task returned for '{application_id}' — "
            f"app_source_uri={app_source_uri}"
        )

        # Sanity check: format_version round-trip.
        if spec.get("format_version") != SPEC_FORMAT_VERSION:
            raise RuntimeError(
                f"AppSpec format_version mismatch: bootstrap produced "
                f"{spec.get('format_version')!r}, worker expects "
                f"{SPEC_FORMAT_VERSION!r}."
            )

        # 4. Translate user-friendly kwargs keys and validate against spec.
        translated_kwargs = self._translate_kwargs_keys(spec, application_kwargs)
        validate_kwargs_against_spec(spec, translated_kwargs)

        # 5. Resource totals and disable_gpu override.
        if disable_gpu:
            for meta in spec["classes"].values():
                opts = meta.get("ray_actor_options", {})
                if opts.get("num_gpus"):
                    opts["num_gpus"] = 0
        required_resources = self._sum_resources(spec, proxy_memory_in_gb)

        # 6. Generate the proxy service token.
        proxy_service_token_ttl_seconds = 3600 * 24 * 30
        proxy_service_token_issued_at = time.time()
        proxy_service_token = await self.server.generate_token(
            {
                "workspace": self.server.config.workspace,
                "permission": "read_write",
                "expires_in": proxy_service_token_ttl_seconds,
            }
        )

        # 7. Authorisation resolution.
        effective_authorized_users = self._resolve_authorized_users(
            manifest, authorized_users, deploying_user, admin_users
        )

        # 8. Build the proxy_args; submit happens later in AppBuilder.submit().
        method_schemas = spec["classes"][spec["entry_id"]]["method_schemas"]
        available_methods = [m["name"] for m in method_schemas]
        spec_hash = hashlib.sha256(
            json.dumps(spec, sort_keys=True, default=str).encode()
        ).hexdigest()

        # worker_workspace stamps the app with the Hypha workspace of the
        # worker that deployed it. On recovery after a worker restart,
        # AppsManager.recover_deployed_applications skips apps whose
        # worker_workspace differs from this worker's — so multiple
        # workers from different workspaces sharing a Ray cluster don't
        # adopt each other's apps.
        app_data = {
            "format_version": SPEC_FORMAT_VERSION,
            "worker_workspace": self.server.config.workspace,
            "deployed_by_worker_client_id": self.server.config.client_id,
            "proxy_service_token_issued_at": proxy_service_token_issued_at,
            "proxy_service_token_ttl_seconds": proxy_service_token_ttl_seconds,
            "entry": entry_id,
            "spec_hash": spec_hash,
            "display_name": manifest["name"],
            "description": manifest["description"],
            "artifact_id": artifact_id,
            "version": version,
            "application_kwargs": application_kwargs,
            "application_env_vars": self._sanitize_recovery_env_vars(
                application_env_vars
            ),
            "disable_gpu": disable_gpu,
            "max_ongoing_requests": max_ongoing_requests,
            "proxy_memory_in_gb": proxy_memory_in_gb,
            "scaling": dict(scaling or {}),
            "application_resources": required_resources,
            "authorized_users": effective_authorized_users,
            "available_methods": available_methods,
            "started_at": started_at if started_at is not None else time.time(),
            "last_updated_at": (
                last_updated_at if last_updated_at is not None else time.time()
            ),
            "last_updated_by": (
                last_updated_by
                if last_updated_by is not None
                else self.server.config.user["id"]
            ),
            "auto_redeploy": auto_redeploy,
            "debug": debug,
        }

        proxy_args = {
            "application_id": application_id,
            "application_name": manifest["name"],
            "application_description": manifest["description"],
            "app_data": app_data,
            "max_ongoing_requests": max_ongoing_requests,
            "server_url": self.server_url,
            "workspace": self.server.config.workspace,
            "worker_client_id": self.server.config.client_id,
            "proxy_service_token": proxy_service_token,
            "authorized_users": effective_authorized_users,
            "proxy_actor_name": self.proxy_actor_name,
            "debug": debug,
            "ice_servers": ice_servers,
        }

        metadata = {
            "name": manifest["name"],
            "description": manifest["description"],
            "version": version,
            "resources": required_resources,
            "authorized_users": effective_authorized_users,
            "available_methods": available_methods,
            "application_kwargs": application_kwargs,
            "application_env_vars": application_env_vars,
            "frontend_entry": manifest.get("frontend_entry"),
            "deployed_by_worker_client_id": self.server.config.client_id,
            "proxy_service_token_issued_at": proxy_service_token_issued_at,
            "proxy_service_token_ttl_seconds": proxy_service_token_ttl_seconds,
        }
        self.logger.info(
            f"Introspected '{application_id}' "
            f"(methods: {available_methods})"
        )
        # Compute the proxy + user-replica pip lists here on the worker.
        # ``get_pip_requirements`` uses ``importlib.metadata`` which only
        # finds bioengine on a host where it was pip-installed; on the
        # Ray actor bioengine arrives as raw source via ``py_modules``,
        # so ``importlib.metadata`` raises. The lists are passed through
        # ``BuiltApp`` to bootstrap, where they are applied at bind time.
        proxy_pip = get_pip_requirements(
            select=["aiortc", "httpx", "hypha-rpc", "pydantic"],
            extras=["worker"],
        )
        # User deployments' ``runtime_env.pip`` only contains what the
        # author declared via ``@bioengine.app(pip=…)``. But ``@bioengine
        # .method`` wraps each method with ``hypha_rpc.utils.schema
        # .schema_method``, and that wrapping creates a cloudpickle
        # reference into ``hypha_rpc``. When Ray Serve cold-starts a
        # replica it ``cloudpickle.loads`` the deployment definition,
        # which re-imports any module the references point at —
        # so without ``hypha-rpc`` (and the ``pydantic`` it pulls in
        # via ``schema_method``) on the replica's venv, the replica
        # crashes at ``__init__`` with ``ModuleNotFoundError: No module
        # named 'hypha_rpc'``. Same story as Fix #7, just one layer
        # deeper. Inject both at bind time.
        user_replica_framework_pip = get_pip_requirements(
            select=["hypha-rpc", "pydantic"],
            extras=[],
        )
        # The env_vars dict the worker assembled above (HYPHA_SERVER_URL,
        # HYPHA_WORKSPACE, HYPHA_ARTIFACT_*, BIOENGINE_*, plus the
        # ``_BIOENGINE_SECRET_*`` secrets including HYPHA_TOKEN) only
        # made it onto the introspect Ray task's ``runtime_env``. The
        # per-replica deployments don't carry it: the @bioengine.app
        # decorator preserves whatever static ``env_vars`` the author
        # put in ``ray_actor_options`` but cannot see anything computed
        # at deploy time. Without this, user code that reads
        # ``os.getenv('HYPHA_TOKEN')`` at ``__init__`` crashes the
        # replica. Pass the worker-side env_vars through so bootstrap
        # merges them into every deployment's runtime_env at bind time.
        replica_env_vars = env_vars

        return BuiltApp(
            metadata=metadata,
            spec=spec,
            translated_kwargs=translated_kwargs,
            proxy_args=proxy_args,
            runtime_env=runtime_env,
            bioengine_uri=self._bioengine_gcs_uri(),
            app_source_uri=app_source_uri,
            proxy_pip=proxy_pip,
            user_replica_framework_pip=user_replica_framework_pip,
            replica_env_vars=replica_env_vars,
            proxy_memory_in_gb=proxy_memory_in_gb,
            scaling=dict(scaling or {}),
        )

    async def submit(self, built_app: "BuiltApp", application_id: str) -> None:
        """Fire the ``build_and_run_application`` Ray task.

        ``build()`` returns metadata only (after the introspect Ray task);
        the manager runs ``_check_resources`` against that metadata and
        then calls this to actually claim the cluster resources. Splitting
        the steps means the resource check happens *before* ``serve.run``,
        not after — and preserves the SLURM-scaling fallback.

        No worker-side cleanup needed: the worker never wrote anything to
        its own filesystem. Source bytes live in Ray's GCS package store
        (uploaded by the introspect task); replicas fetch from there.
        """
        # The Ray Client server unpickles ``.remote(...)`` args in a
        # per-client runtime_env that ships bioengine via ``py_modules``
        # but no ``pip`` — so any class ref in the pickle stream whose
        # module top-level imports ``hypha_rpc`` crashes the unpickle.
        # Sanitising dict args to primitives here keeps the
        # worker↔cluster RPC contract pure-data and cluster-env-agnostic.
        try:
            await asyncio.to_thread(
                ray.get,
                # max_calls=1: same reason as the introspect task — a fresh
                # process guarantees the deployment class is imported from the
                # refreshed source, not a stale sys.modules entry cached on a
                # reused worker (the class is cloudpickled by value to the
                # replicas, so the build worker's import is what they run).
                ray.remote(num_cpus=0, max_calls=1, runtime_env=built_app.runtime_env)(
                    build_and_run_application
                ).remote(
                    _ensure_jsonable(built_app.spec),
                    _ensure_jsonable(built_app.translated_kwargs),
                    _ensure_jsonable(built_app.proxy_args),
                    application_id,
                    f"/{application_id}",
                    built_app.bioengine_uri,
                    built_app.app_source_uri,
                    built_app.proxy_pip,
                    built_app.user_replica_framework_pip,
                    _ensure_jsonable(built_app.replica_env_vars),
                    built_app.proxy_memory_in_gb,
                    _ensure_jsonable(built_app.scaling),
                ),
            )
        except ray.exceptions.RayTaskError as exc:
            cause = exc.cause if isinstance(exc.cause, Exception) else exc
            raise BioEngineUserError(str(cause)) from exc
        self.logger.info(f"Submitted '{application_id}' to Ray Serve")


class BuiltApp:
    """Carries the introspected spec + metadata until the manager submits.

    ``AppBuilder.build()`` previously returned a ``serve.Application`` that
    the manager passed to ``serve.run``. In v0.11, ``serve.run`` is invoked
    inside a Ray task (see :func:`bioengine._app.bootstrap.
    build_and_run_application`) because Ray Serve's
    ``Deployment.__getattr__`` falls into unbounded recursion when an
    unpickled deployment is re-accessed; the graph can't ride back to the
    worker. The Built object stashes everything the submit task needs so
    the manager can do ``build → check_resources → submit`` and keep the
    resource check on the right side of the actual claim.

    ``app_source_uri`` is the ``gcs://_ray_pkg_<hash>.zip`` URI returned
    by the introspect Ray task — the build task plus every replica use
    it to materialise source via Ray's internal package store, no Hypha
    auth needed.
    """

    __slots__ = (
        "metadata",
        "spec",
        "translated_kwargs",
        "proxy_args",
        "runtime_env",
        "bioengine_uri",
        "app_source_uri",
        "proxy_pip",
        "user_replica_framework_pip",
        "replica_env_vars",
        "proxy_memory_in_gb",
        "scaling",
    )

    def __init__(
        self,
        metadata: Dict[str, Any],
        spec: Dict[str, Any],
        translated_kwargs: Dict[str, Dict[str, Any]],
        proxy_args: Dict[str, Any],
        runtime_env: Dict[str, Any],
        bioengine_uri: str,
        app_source_uri: str,
        proxy_pip: List[str],
        user_replica_framework_pip: List[str],
        replica_env_vars: Dict[str, str],
        proxy_memory_in_gb: float,
        scaling: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.metadata = metadata
        self.spec = spec
        self.translated_kwargs = translated_kwargs
        self.proxy_args = proxy_args
        self.runtime_env = runtime_env
        self.bioengine_uri = bioengine_uri
        self.app_source_uri = app_source_uri
        self.proxy_pip = proxy_pip
        self.user_replica_framework_pip = user_replica_framework_pip
        self.replica_env_vars = replica_env_vars
        self.proxy_memory_in_gb = proxy_memory_in_gb
        # Map of {user_deployment_class_name: {num_replicas, autoscaling_config}}.
        # Empty default means every user deployment runs at Ray Serve's default
        # of one replica with no autoscaling.
        self.scaling = dict(scaling or {})
