"""Replica-side (and build-task-side) artifact materialisation.

Materialises the user's app source into ``<app_dir>/source/`` before any
user code is imported. Two backends:

1. **Ray-internal GCS** (``BIOENGINE_APP_SOURCE_URI`` set) — preferred on
   replicas. The :mod:`bioengine._app.bootstrap` introspect task uploaded
   the source bytes to Ray's content-addressed package store; replicas
   re-materialise them by URI via
   :func:`ray._private.runtime_env.packaging.download_and_unpack_package`.
   No Hypha auth involved — the short-TTL token has typically expired by
   replica launch time.

2. **Hypha** (``BIOENGINE_ARTIFACT_DOWNLOAD_URL`` + ``_TOKEN`` set) — used
   by the introspect Ray task which has a fresh short-TTL token. Single
   ``fcntl`` lock per node serialises concurrent same-node starts; the
   second caller sees the up-to-date ``.version`` marker and skips.

Reads its configuration from env_vars set by the BioEngine worker:

* ``BIOENGINE_APP_DIR``                — ``<apps_workdir>/<worker-ws>-<app-id>``
* ``BIOENGINE_ARTIFACT_VERSION``       — cache-invalidation marker
* ``BIOENGINE_APP_SOURCE_URI``         — ``gcs://_ray_pkg_<hash>.zip`` (replicas)
* ``BIOENGINE_ARTIFACT_DOWNLOAD_URL``  — Hypha ``create-zip-file`` URL (introspect)
* ``BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN``— short-TTL read-only Hypha token
"""

from __future__ import annotations

import fnmatch
import logging
import os
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional

#: Excludes mirror v0.11.3's ``AppBuilder._PY_MODULES_EXCLUDES`` with one
#: addition (``.dotfiles`` per user request) — the artifact zip carries
#: everything in the artifact, but only Python/config content should be
#: extracted to ``source/``.
_SOURCE_EXCLUDES = [
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
    "frontend/**",
    "__pycache__/**",
]


def _is_excluded(rel_path: str) -> bool:
    components = rel_path.split("/")
    if any(comp.startswith(".") for comp in components):
        return True
    basename = components[-1]
    for pat in _SOURCE_EXCLUDES:
        if pat.endswith("/**"):
            if pat[:-3] in components[:-1]:
                return True
        elif fnmatch.fnmatch(basename, pat):
            return True
    return False


def _authed_url(url: str, token: Optional[str]) -> str:
    if not token:
        return url
    sep = "&" if urllib.parse.urlparse(url).query else "?"
    return f"{url}{sep}token={urllib.parse.quote(token, safe='')}"


def _download_and_extract(
    url: str, token: Optional[str], dest: Path, logger: logging.Logger
) -> None:
    """Download the artifact zip to a temp file, extract filtered entries, atomic rename into dest."""
    url = _authed_url(url, token)

    app_root = dest.parent
    tmp_zip = app_root / f".source.tmp.{os.getpid()}.zip"
    tmp_dir = app_root / f".source.new.{os.getpid()}"
    try:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        logger.info(f"BioEngine: downloading artifact zip → {tmp_zip}")
        with urllib.request.urlopen(url, timeout=120) as r:
            with open(tmp_zip, "wb") as out:
                shutil.copyfileobj(r, out, length=1024 * 1024)

        kept = 0
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if _is_excluded(info.filename):
                    continue
                zf.extract(info, tmp_dir)
                kept += 1

        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp_dir, dest)
        logger.info(
            f"BioEngine: extracted {kept} source files → {dest}"
        )
    finally:
        try:
            tmp_zip.unlink(missing_ok=True)
        except OSError:
            pass
        if tmp_dir.exists():
            try:
                shutil.rmtree(tmp_dir)
            except OSError:
                pass


def _download_from_ray_gcs(uri: str, dest: Path, logger: logging.Logger) -> None:
    """Materialise ``dest`` from a ``gcs://_ray_pkg_<hash>.zip`` URI.

    Wraps :func:`ray._private.runtime_env.packaging.download_and_unpack_package`
    (which is async) in a synchronous facade since both the replica
    bootstrap and the build Ray task are sync callers. The unpack target
    Ray picks is its own scratch dir; we copy the result into ``dest``
    with the same filter logic ``_download_and_extract`` uses for Hypha
    zips so the on-disk layout is identical across backends.
    """
    import asyncio
    import tempfile

    from ray._private.runtime_env.packaging import download_and_unpack_package

    logger.info(f"BioEngine: downloading source from Ray-GCS {uri} → {dest}")
    with tempfile.TemporaryDirectory(prefix="bioengine-rayfetch-") as scratch:
        unpacked = asyncio.run(download_and_unpack_package(uri, scratch))
        unpacked_path = Path(unpacked)

        app_root = dest.parent
        tmp_dir = app_root / f".source.new.{os.getpid()}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        kept = 0
        for root, _dirs, files in os.walk(unpacked_path):
            root_p = Path(root)
            rel_root = root_p.relative_to(unpacked_path)
            for fname in files:
                rel = (rel_root / fname).as_posix()
                if _is_excluded(rel):
                    continue
                out = tmp_dir / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root_p / fname, out)
                kept += 1
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp_dir, dest)
        logger.info(f"BioEngine: extracted {kept} source files → {dest}")


def _ensure_source(app_dir: Path, version: str, logger: logging.Logger) -> Path:
    """Atomically populate ``app_dir/source`` so it matches ``version``.

    Three backends, picked in order:

    1. ``BIOENGINE_LOCAL_ARTIFACT_PATH`` — dev override, short-circuits to a
       locally-mounted artifact root.
    2. ``BIOENGINE_APP_SOURCE_URI`` — Ray-internal GCS download via
       :func:`_download_from_ray_gcs`. Preferred on replicas because the
       Hypha short-TTL token has typically expired by replica launch.
    3. ``BIOENGINE_ARTIFACT_DOWNLOAD_URL`` (+ optional ``_TOKEN``) — direct
       Hypha pull, used by the introspect Ray task.

    Single ``fcntl`` lock per node serialises concurrent same-node starts;
    the second caller sees the up-to-date ``.version`` marker and skips.
    """
    source = app_dir / "source"
    version_marker = app_dir / ".version"
    app_dir.mkdir(parents=True, exist_ok=True)

    # Dev override: ``BIOENGINE_LOCAL_ARTIFACT_PATH`` points at a directory
    # holding ``<artifact_alias>/`` subdirs with the raw app sources.
    local_root_env = os.environ.get("BIOENGINE_LOCAL_ARTIFACT_PATH")
    artifact_id = os.environ.get("BIOENGINE_ARTIFACT_ID", "")
    if local_root_env and artifact_id:
        alias = artifact_id.split("/")[-1]
        candidate = Path(local_root_env) / alias
        if candidate.is_dir():
            if source.exists():
                shutil.rmtree(source)
            shutil.copytree(
                candidate, source, ignore=shutil.ignore_patterns("__pycache__", ".git")
            )
            version_marker.write_text(version)
            logger.info(f"BioEngine: source mirrored from local path {candidate}")
            return source

    import fcntl

    lock_path = app_dir / ".lock"
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            current = version_marker.read_text().strip() if version_marker.exists() else None
            if current == version and source.is_dir():
                logger.info(
                    f"BioEngine: source already at version {version} — skip download"
                )
                return source

            gcs_uri = os.environ.get("BIOENGINE_APP_SOURCE_URI")
            if gcs_uri:
                _download_from_ray_gcs(gcs_uri, source, logger)
                version_marker.write_text(version)
                return source

            url = os.environ.get("BIOENGINE_ARTIFACT_DOWNLOAD_URL")
            if not url:
                raise RuntimeError(
                    "Neither BIOENGINE_APP_SOURCE_URI nor "
                    "BIOENGINE_ARTIFACT_DOWNLOAD_URL is set; cannot "
                    "materialise the app source. The worker is expected to "
                    "populate one of these env_vars when constructing the "
                    "runtime_env for the task or deployment."
                )
            token = os.environ.get("BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN") or None
            _download_and_extract(url, token, source, logger)
            version_marker.write_text(version)
            return source
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def setup_replica_environment() -> None:
    """``worker_process_setup_hook`` entry point.

    Runs *before* Ray Serve loads the deployment class, so by the time the
    user's module is imported its ``sys.path`` already contains ``source/``
    and the ``HOME`` / ``TMPDIR`` / CWD env are pointed at the per-app
    workspace.
    """
    logger = logging.getLogger("ray.serve")

    app_dir_env = os.environ.get("BIOENGINE_APP_DIR")
    if not app_dir_env:
        # Workers don't run this hook (they don't get BIOENGINE_APP_DIR);
        # presence guards against accidental invocation outside a replica.
        return

    app_dir = Path(app_dir_env).resolve()
    version = os.environ.get("BIOENGINE_ARTIFACT_VERSION", "")
    if not version:
        raise RuntimeError(
            "BIOENGINE_ARTIFACT_VERSION not set; replica cannot determine "
            "which version of the source to materialise."
        )

    source = _ensure_source(app_dir, version, logger)

    home = app_dir / "home"
    tmp = app_dir / "tmp"
    home.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(home)
    os.environ["TMPDIR"] = str(tmp)
    os.environ["TMP"] = str(tmp)
    os.environ["TEMP"] = str(tmp)

    src_str = str(source)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    logger.info(
        f"BioEngine: replica environment ready "
        f"(source={source}, HOME={home}, TMPDIR={tmp})"
    )


__all__ = ["setup_replica_environment"]
