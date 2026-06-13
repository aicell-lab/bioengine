"""Build BioEngine apps for Ray Serve from v0.6.0 artifacts.

In the v0.6.0 model the worker never imports user code. It ships the user's
Python package as ``runtime_env.py_modules`` and delegates introspection and
binding to two Ray tasks defined in :mod:`bioengine._app.bootstrap`. The
worker only handles:

* manifest loading and validation
* artifact materialisation (download the *package directory*, nothing else)
* runtime_env composition (base pip + user pip extracted via AST)
* pydantic-core preflight (so cross-process pickling failures surface early)
* kwargs validation against the spec
* authorisation-rule resolution
* ``ProxyDeployment`` argument assembly

Everything else — class loading, lifecycle wiring, schema collection,
``cls.bind(...)`` composition — runs inside the app's runtime_env.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
import ray
import yaml
from hypha_rpc.rpc import RemoteService
from hypha_rpc.utils import ObjectProxy
from ray import serve

from bioengine._app.bootstrap import (
    SPEC_FORMAT_VERSION,
    build_and_run_application,
    introspect_app,
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

        self._sweep_runtime_env_tmp_files()

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

    # ───────────────────────── artifact files ────────────────────────────

    async def _materialize_artifact(
        self,
        artifact_id: str,
        version: Optional[str],
        application_id: str,
    ) -> Path:
        """Download the whole artifact root into a clean per-app directory.

        In the v0.6.0 layout the artifact root *is* the Python module dir:
        ``manifest.yaml`` and any ``.py`` files live side by side at the
        top level. The whole directory is later shipped as ``py_modules``;
        non-Python content (``manifest.yaml``, ``README.md``,
        ``frontend/``, notebooks) is excluded by
        :meth:`_write_pkg_to_runtime_env_dir`.
        """
        target_dir = self.apps_workdir / application_id / "source"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        # The app workdir doubles as HOME for the replica (see _build_env_vars).
        # On shared-NFS workers with root_squash the worker pod's mkdir lands
        # owned by nobody:nogroup (65534), so the replica's UID can't create
        # subdirs under Path.home(). Sticky + world-writable lets any replica
        # UID create state while preventing cross-UID deletes.
        app_workdir = target_dir.parent
        try:
            app_workdir.chmod(0o1777)
        except OSError as exc:
            self.logger.warning(
                f"Could not chmod app workdir {app_workdir}: {exc}. "
                "Replicas may fail to create subdirectories under Path.home()."
            )

        local_root = self._local_artifact_root(artifact_id)
        if local_root and local_root.is_dir():
            self.logger.info(
                f"Materialising '{artifact_id}' from local path "
                f"{local_root} → {target_dir}"
            )
            shutil.copytree(
                local_root,
                target_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", ".git"),
            )
            return target_dir

        self.logger.info(
            f"Downloading artifact root '{artifact_id}' (version={version}) "
            f"→ {target_dir}"
        )
        files = await self.artifact_manager.list_files(
            artifact_id=artifact_id, version=version
        )
        await self._download_files_recursive(
            artifact_id, version, target_dir, "", files
        )
        return target_dir

    def _local_artifact_root(self, artifact_id: str) -> Optional[Path]:
        local_root_env = os.environ.get("BIOENGINE_LOCAL_ARTIFACT_PATH")
        if not local_root_env:
            return None
        artifact_folder = artifact_id.split("/")[1]
        candidate = Path(local_root_env) / artifact_folder
        return candidate if candidate.is_dir() else None

    async def _download_files_recursive(
        self,
        artifact_id: str,
        version: Optional[str],
        target_dir: Path,
        prefix: str,
        files: List[Dict[str, Any]],
    ) -> None:
        """Walk ``files`` from ``artifact_manager.list_files`` and download to disk."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            for entry in files:
                name = entry.get("name")
                if name is None:
                    continue
                rel_path = f"{prefix}/{name}" if prefix else name
                if entry.get("type") == "directory":
                    sub = await self.artifact_manager.list_files(
                        artifact_id=artifact_id,
                        version=version,
                        dir_path=rel_path,
                    )
                    await self._download_files_recursive(
                        artifact_id, version, target_dir, rel_path, sub
                    )
                    continue
                url = await self.artifact_manager.get_file(
                    artifact_id=artifact_id,
                    version=version,
                    file_path=rel_path,
                )
                response = await client.get(url)
                response.raise_for_status()
                out_path = target_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(response.content)

    # ─────────────────────── runtime_env compose ─────────────────────────

    def _build_env_vars(
        self,
        application_id: str,
        artifact_id: str,
        version: Optional[str],
        non_secret_env_vars: Dict[str, str],
        secret_env_vars: Dict[str, str],
        hypha_token: Optional[str],
    ) -> Dict[str, str]:
        """Compose the env_vars dict the replica's runtime_env will receive."""
        env_vars: Dict[str, str] = {}
        env_vars.update(non_secret_env_vars)
        for key, value in secret_env_vars.items():
            env_vars[f"_BIOENGINE_SECRET_{key}"] = value

        if hypha_token is not None:
            env_vars["_BIOENGINE_SECRET_HYPHA_TOKEN"] = hypha_token

        app_workdir = self.apps_workdir / application_id
        env_vars["HOME"] = str(app_workdir)
        tmp_dir = str(app_workdir / "tmp")
        env_vars["TMPDIR"] = tmp_dir
        env_vars["TEMP"] = tmp_dir
        env_vars["TMP"] = tmp_dir

        if self.server is not None:
            env_vars["HYPHA_SERVER_URL"] = self.server_url
            env_vars["HYPHA_WORKSPACE"] = self.server.config.workspace
        env_vars["HYPHA_ARTIFACT_ID"] = artifact_id
        if version is not None:
            env_vars["HYPHA_ARTIFACT_VERSION"] = version

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

    def _build_introspect_runtime_env(
        self,
        pkg_uri: str,
        bioengine_uri: str,
        env_vars: Dict[str, str],
    ) -> Dict[str, Any]:
        """Compose the runtime_env for the introspection + build Ray tasks.

        The pip list is the BioEngine baseline only — every module
        containing ``@bioengine.app`` (and anything they transitively
        import at top level) is required to be importable with just
        ``bioengine[worker]`` and the standard library. Heavy
        application dependencies belong in the decorator's ``pip=`` arg
        (where Ray Serve installs them once per replica) and are
        imported lazily inside method bodies or from sibling modules
        that aren't imported during introspection.

        ``py_modules`` ships two URIs: the worker's own ``bioengine``
        source (the actor's venv only carries bioengine's *deps* — see
        :meth:`_write_bioengine_source_to_runtime_env_dir`) and the
        user app package. Both URIs are ``file://...zip`` returned by
        :meth:`_write_pkg_to_runtime_env_dir`. The caller is responsible
        for running the zip walks off the asyncio loop.
        """
        base_pip = update_requirements(
            [],
            select=["httpx", "hypha-rpc", "pydantic"],
            extras=["worker"],
        )
        introspect_env = {
            k: v for k, v in env_vars.items() if not k.startswith("_BIOENGINE_SECRET_")
        }
        introspect_env.pop("BIOENGINE_DATA_SERVER_URL", None)

        return {
            "py_modules": [bioengine_uri, pkg_uri],
            "pip": base_pip,
            "env_vars": introspect_env,
        }

    #: Files inside the artifact root that are *not* Python source and
    #: should not ship to Ray Serve replicas. ``manifest.yaml`` is already
    #: stored as the artifact's native manifest field; ``frontend/`` is
    #: served by Hypha statically; the others are author-facing docs.
    _PY_MODULES_EXCLUDES = [
        "manifest.yaml",
        "manifest.yml",
        "README*",
        "*.md",
        "*.ipynb",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.gif",
        "*.svg",
        "*.pdf",
        "*.pyc",
        "*.pyo",
        "*.so",
        "frontend/**",
        "__pycache__/**",
        ".git/**",
        ".github/**",
    ]

    #: Subdirectory of ``apps_workdir`` where content-addressed
    #: ``runtime_env.py_modules`` zips live. On a worker whose
    #: ``apps_workdir`` is on a shared filesystem with the Ray cluster
    #: (e.g. NFS) this is exactly the path Ray sees through the
    #: ``file://`` URI we hand it.
    _RUNTIME_ENV_PKG_SUBDIR = "_runtime_env_packages"

    @staticmethod
    def _is_excluded(rel_path: str, excludes: List[str]) -> bool:
        """Match ``rel_path`` against the AppBuilder exclude patterns.

        Two pattern shapes are supported, mirroring the subset of
        gitignore semantics ``_PY_MODULES_EXCLUDES`` uses:

        * ``foo`` / ``foo*`` / ``*.ext`` — fnmatch against the basename.
        * ``dir/**`` — match if ``dir`` appears as any directory
          component along the path.
        """
        import fnmatch

        components = rel_path.split("/")
        basename = components[-1]
        for pat in excludes:
            if pat.endswith("/**"):
                if pat[:-3] in components[:-1]:
                    return True
            elif fnmatch.fnmatch(basename, pat):
                return True
        return False

    def _sweep_runtime_env_tmp_files(self) -> None:
        """Unlink ``*.tmp.<pid>`` files left in ``_runtime_env_packages/``
        by a prior crashed write (worker SIGKILL'd mid-zip)."""
        pkg_dir = self.apps_workdir / self._RUNTIME_ENV_PKG_SUBDIR
        if not pkg_dir.is_dir():
            return
        count = 0
        orphans = list(pkg_dir.glob("bioengine_pkg_*.zip.tmp.*")) + list(
            pkg_dir.glob("bioengine_runtime_*.zip.tmp.*")
        )
        for orphan in orphans:
            try:
                orphan.unlink()
                count += 1
            except OSError as exc:
                self.logger.warning(
                    f"Could not remove orphan tmp file {orphan}: {exc}"
                )
        if count:
            self.logger.warning(
                f"Swept {count} orphan tmp runtime_env package(s) "
                f"from {pkg_dir} — signal of a prior unclean shutdown"
            )

    def _write_pkg_to_runtime_env_dir(
        self,
        pkg_root_dir: Path,
        *,
        filename_prefix: str = "bioengine_pkg",
        arc_prefix: str = "",
    ) -> str:
        """Build a content-hashed zip of ``pkg_root_dir`` and return a
        ``file://`` URI Ray accepts in task-level ``py_modules``.

        Why not ``gcs://`` via Ray's ``upload_package_if_needed``? In
        external-cluster mode the upload routes through the ray-client
        gRPC bridge to a server-side runtime-env handler that has been
        observed to never reply, wedging the worker's asyncio loop. A
        local zip on a path the Ray cluster can read directly (e.g. the
        shared NFS mount that backs ``apps_workdir`` on the deNBI
        staging cluster) takes the bridge out of the upload path.

        Same input bytes → same SHA-256 → same destination path, so
        repeat calls return without re-writing. Atomic publish via
        ``tmp.<pid>`` + ``os.replace`` keeps readers from ever seeing a
        partial file. Two writers racing the same content_hash is safe
        because the bytes are by construction identical.

        The directory ``<apps_workdir>/_runtime_env_packages/`` is
        content-addressed across every app and grows monotonically; a
        GC pass keyed on currently-deployed apps will be a follow-up.
        """
        import hashlib
        import os as _os
        import zipfile

        pkg_root = pkg_root_dir.resolve()
        included: List[tuple[str, Path]] = []
        for abs_path in sorted(pkg_root.rglob("*")):
            if not abs_path.is_file():
                continue
            rel = abs_path.relative_to(pkg_root).as_posix()
            if self._is_excluded(rel, self._PY_MODULES_EXCLUDES):
                continue
            included.append((rel, abs_path))

        if not included:
            raise RuntimeError(
                f"No Python source files to ship from {pkg_root_dir} "
                f"after applying excludes."
            )

        h = hashlib.sha256()
        h.update(arc_prefix.encode("utf-8"))
        h.update(b"\x00")
        total_src_bytes = 0
        for rel, abs_path in included:
            h.update(rel.encode("utf-8"))
            h.update(b"\x00")
            size = abs_path.stat().st_size
            total_src_bytes += size
            h.update(size.to_bytes(8, "big"))
            h.update(b"\x00")
            with open(abs_path, "rb") as f:
                while True:
                    chunk = f.read(64 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
            h.update(b"\x00")
        digest = h.hexdigest()[:16]

        pkg_dir = self.apps_workdir / self._RUNTIME_ENV_PKG_SUBDIR
        pkg_dir.mkdir(parents=True, exist_ok=True)
        out_path = pkg_dir / f"{filename_prefix}_{digest}.zip"
        if out_path.is_file():
            return out_path.as_uri()

        tmp_path = pkg_dir / f"{out_path.name}.tmp.{_os.getpid()}"
        try:
            with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for rel, abs_path in included:
                    arc = f"{arc_prefix}/{rel}" if arc_prefix else rel
                    zf.write(abs_path, arcname=arc)
            _os.replace(tmp_path, out_path)
        except BaseException:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

        self.logger.info(
            f"Wrote runtime_env package {out_path.name} "
            f"({len(included)} files, {total_src_bytes} src bytes, "
            f"{out_path.stat().st_size} zipped bytes)"
        )
        return out_path.as_uri()

    def _build_py_modules_uris(self, pkg_root_dir: Path) -> tuple[str, str]:
        """Write both runtime_env zips and return ``(pkg_uri, bioengine_uri)``.

        Wrapped in a single ``asyncio.to_thread`` call by ``build()`` so
        both blocking zip walks happen off the asyncio loop together.
        """
        pkg_uri = self._write_pkg_to_runtime_env_dir(pkg_root_dir)
        bioengine_uri = self._write_bioengine_source_to_runtime_env_dir()
        return pkg_uri, bioengine_uri

    def _write_bioengine_source_to_runtime_env_dir(self) -> str:
        """Bundle the worker's installed ``bioengine`` source as a
        ``file://`` zip and return its URI for ``runtime_env.py_modules``.

        Why this exists: in external-cluster mode the Ray actor that
        runs ``introspect_app`` (and every user deployment replica)
        spins up in a fresh process. ``runtime_env.pip`` installs
        bioengine's *dependencies* into a venv, but bioengine itself is
        not on PyPI, so the venv does not contain it. Without bioengine
        on the actor's ``sys.path``, ``pickle.loads`` of any function
        defined under ``bioengine._app`` fails immediately with
        ``ModuleNotFoundError: No module named 'bioengine'``.

        Why the ``_bioengine_wrap/`` prefix: Ray 2.55.1 calls
        ``unzip_package(remove_top_level_directory=True)`` when the URI
        protocol is ``file://`` (and every other "remote" scheme); a zip
        whose entries all live under a single top-level directory has
        that directory stripped on extraction. If we zipped entries as
        ``bioengine/__init__.py`` directly, Ray would strip ``bioengine/``
        and place ``__init__.py`` at the extract dir's root — and
        ``import bioengine`` would fail. By zipping under
        ``_bioengine_wrap/bioengine/`` we give Ray a throwaway top-level
        directory to strip, leaving ``bioengine/__init__.py`` at the
        extract dir's root, which is what gets added to ``sys.path``.
        ``gcs://`` URIs do the opposite (no strip) but we cannot use
        ``gcs://`` here — see PR #106 for the Ray-client wedge that
        forced the move to ``file://``.
        """
        import bioengine

        bioengine_dir = Path(bioengine.__file__).parent
        return self._write_pkg_to_runtime_env_dir(
            bioengine_dir,
            filename_prefix="bioengine_runtime",
            arc_prefix="_bioengine_wrap/bioengine",
        )

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
    ) -> serve.Application:
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

        # 1. Materialise the artifact root.
        pkg_root_dir = await self._materialize_artifact(
            artifact_id, version, application_id
        )

        # 2. Compose env_vars and the introspection runtime_env.
        # The CLI-supplied env_vars are flattened across deployments today;
        # in the v0.6 model the framework only sees a single env_vars dict
        # (per-deployment runtime_env extension is via the @app decorator's
        # env_vars kwarg). We collapse them here.
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
            non_secret_env_vars, secret_env_vars, hypha_token,
        )
        # Off-loop: the zip walk hashes and compresses every file in
        # the artifact root and could block the asyncio loop for large
        # apps. 120s keeps us under the 150s liveness window so a slow
        # disk fails cleanly instead of triggering a pod SIGKILL. We
        # also bundle the worker's own bioengine source here — the
        # actor's runtime_env.pip installs bioengine's *deps* into a
        # venv but bioengine itself is not on PyPI, so without this the
        # Ray actor crashes at ``pickle.loads(introspect_app)`` with
        # ``ModuleNotFoundError: No module named 'bioengine'``.
        try:
            pkg_uri, bioengine_uri = await asyncio.wait_for(
                asyncio.to_thread(self._build_py_modules_uris, pkg_root_dir),
                timeout=120,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "Timed out (120s) writing runtime_env package zips into "
                "the runtime_env packages directory."
            ) from exc
        runtime_env = self._build_introspect_runtime_env(
            pkg_uri, bioengine_uri, env_vars
        )

        # 3. Introspect the user package via a Ray task in the app's env.
        try:
            spec = await asyncio.to_thread(
                ray.get,
                ray.remote(num_cpus=0, runtime_env=runtime_env)(introspect_app).remote(
                    entry_id
                ),
            )
        except ray.exceptions.RayTaskError as exc:
            cause = exc.cause if isinstance(exc.cause, Exception) else exc
            raise BioEngineUserError(str(cause)) from exc

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
        proxy_service_token = await self.server.generate_token(
            {
                "workspace": self.server.config.workspace,
                "permission": "read_write",
                "expires_in": 3600 * 24 * 30,
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

        app_data = {
            "format_version": SPEC_FORMAT_VERSION,
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
            pkg_uri=pkg_uri,
            bioengine_uri=bioengine_uri,
            proxy_pip=proxy_pip,
            user_replica_framework_pip=user_replica_framework_pip,
            replica_env_vars=replica_env_vars,
            proxy_memory_in_gb=proxy_memory_in_gb,
        )

    async def submit(self, built_app: "BuiltApp", application_id: str) -> None:
        """Fire the ``build_and_run_application`` Ray task.

        ``build()`` returns metadata only (after introspection); the
        manager runs ``_check_resources`` against that metadata and then
        calls this to actually claim the cluster resources. Splitting
        the steps means the resource check happens *before* ``serve.run``,
        not after.
        """
        try:
            await asyncio.to_thread(
                ray.get,
                ray.remote(num_cpus=0, runtime_env=built_app.runtime_env)(
                    build_and_run_application
                ).remote(
                    built_app.spec,
                    built_app.translated_kwargs,
                    built_app.proxy_args,
                    application_id,
                    f"/{application_id}",
                    built_app.pkg_uri,
                    built_app.bioengine_uri,
                    built_app.proxy_pip,
                    built_app.user_replica_framework_pip,
                    built_app.replica_env_vars,
                    built_app.proxy_memory_in_gb,
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
    """

    __slots__ = (
        "metadata",
        "spec",
        "translated_kwargs",
        "proxy_args",
        "runtime_env",
        "pkg_uri",
        "bioengine_uri",
        "proxy_pip",
        "user_replica_framework_pip",
        "replica_env_vars",
        "proxy_memory_in_gb",
    )

    def __init__(
        self,
        metadata: Dict[str, Any],
        spec: Dict[str, Any],
        translated_kwargs: Dict[str, Dict[str, Any]],
        proxy_args: Dict[str, Any],
        runtime_env: Dict[str, Any],
        pkg_uri: str,
        bioengine_uri: str,
        proxy_pip: List[str],
        user_replica_framework_pip: List[str],
        replica_env_vars: Dict[str, str],
        proxy_memory_in_gb: float,
    ) -> None:
        self.metadata = metadata
        self.spec = spec
        self.translated_kwargs = translated_kwargs
        self.proxy_args = proxy_args
        self.runtime_env = runtime_env
        self.pkg_uri = pkg_uri
        self.bioengine_uri = bioengine_uri
        self.proxy_pip = proxy_pip
        self.user_replica_framework_pip = user_replica_framework_pip
        self.replica_env_vars = replica_env_vars
        self.proxy_memory_in_gb = proxy_memory_in_gb
