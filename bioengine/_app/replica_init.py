"""Replica-side (and build-task-side) artifact materialisation.

Materialises the user's app source into ``<app_dir>/source/`` before any user
code is imported, by downloading it **per file directly from Hypha** (the
durable artifact store) with an incremental sync keyed on each file's remote
``last_modified``. A snapshot at ``<app_dir>/.source_snapshot.json`` records
``{relpath: last_modified}`` so a redeploy re-downloads only changed/new files
and deletes files removed from the artifact — big unchanged files (e.g. model
weights) are never re-fetched. There is no Ray-GCS intermediary: the Ray head's
in-memory package store is neither an OOM sink (a package per app×version) nor a
single point of failure across a head restart.

Reads its configuration from env_vars set by the BioEngine worker:

* ``BIOENGINE_APP_DIR``                 — ``<apps_workdir>/<worker-ws>-<app-id>``
* ``BIOENGINE_ARTIFACT_ID``             — artifact id (cache identity)
* ``BIOENGINE_ARTIFACT_VERSION``        — committed version to materialise
* ``BIOENGINE_ARTIFACT_FILES_URL``      — Hypha ``/files`` listing endpoint
* ``BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN`` — read token, **only ever set in a Ray
  task's env, never in a Serve replica's**: the worker's staging task receives it
  as a task *argument* and applies it onto its own ``os.environ``. A replica
  materialises from the already-staged directory instead and holds no
  credential. Absent and a public artifact ⇒ anonymous read.

Hypha is already a hard dependency (an app cannot register its service without
it), so pulling source from Hypha at replica start adds no new availability
surface: if Hypha is unreachable the replica fails to materialise, the
deployment fails, and ``auto_redeploy`` retries.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

#: Excludes mirror v0.11.3's ``AppBuilder._PY_MODULES_EXCLUDES`` with one
#: addition (``.dotfiles`` per user request) — the artifact carries everything,
#: but only Python/config content should be materialised into ``source/``.
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


def _artifact_url(
    files_url: str, subpath: str, version: str, token: Optional[str]
) -> str:
    """Build a Hypha ``/files`` URL for a listing subpath or a single file."""
    url = f"{files_url}/{subpath}"
    query = []
    if version:
        query.append(f"version={urllib.parse.quote(version)}")
    if token:
        query.append(f"token={urllib.parse.quote(token, safe='')}")
    if query:
        url = f"{url}?{'&'.join(query)}"
    return url


def _list_source_files(
    files_url: str, version: str, token: Optional[str], logger: logging.Logger
) -> Dict[str, str]:
    """Recursively list the artifact's non-excluded source files.

    The Hypha ``/files`` endpoint is not recursive — it returns, per item,
    ``name`` / ``type`` (``"directory"`` vs file) / ``size`` / ``last_modified``
    — so we walk directories ourselves by re-requesting each subpath. Returns
    ``{relpath: last_modified}`` for exactly the files that would land in
    ``source/`` (excluded files are never listed or fetched). Raises on any
    fetch/parse failure so the caller fails the deployment and ``auto_redeploy``
    retries rather than silently serving a partial sync.
    """

    def _list(subpath: str) -> Optional[List[dict]]:
        url = _artifact_url(files_url, subpath, version, token)
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode())

    files: Dict[str, str] = {}

    def _walk(subpath: str) -> None:
        for item in _list(subpath) or []:
            name = item.get("name")
            if not name:
                continue
            rel = f"{subpath}{name}"
            if item.get("type") == "directory":
                if _is_excluded(f"{rel}/_"):
                    continue
                _walk(f"{rel}/")
            elif not _is_excluded(rel):
                files[rel] = str(item.get("last_modified"))

    _walk("")
    return files


def _download_file(
    files_url: str,
    relpath: str,
    version: str,
    token: Optional[str],
    dest: Path,
    logger: logging.Logger,
) -> None:
    """Stream one artifact file to ``dest`` (temp file + atomic rename)."""
    url = _artifact_url(files_url, relpath, version, token)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.tmp.{os.getpid()}"
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            with open(tmp, "wb") as out:
                shutil.copyfileobj(r, out, length=1024 * 1024)
        os.replace(tmp, dest)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _prune_empty_dirs(start: Path, stop_at: Path) -> None:
    """Remove now-empty directories from ``start`` up to (not incl.) ``stop_at``."""
    d = start
    while d != stop_at and stop_at in d.parents:
        try:
            d.rmdir()
        except OSError:
            break
        d = d.parent


def _sync_source_from_hypha(
    files_url: str,
    version: str,
    token: Optional[str],
    source: Path,
    snapshot_path: Path,
    logger: logging.Logger,
) -> None:
    """Incrementally sync ``source/`` to the artifact's committed ``version``.

    Downloads files whose remote ``last_modified`` differs from the local
    snapshot (or that are missing on disk), deletes files that vanished from the
    artifact, and rewrites the snapshot only after every download succeeds so a
    partial failure re-syncs cleanly on the next attempt.
    """
    remote = _list_source_files(files_url, version, token, logger)

    snapshot: Dict[str, str] = {}
    if snapshot_path.exists():
        try:
            snapshot = json.loads(snapshot_path.read_text())
        except (OSError, ValueError):
            snapshot = {}

    source.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for relpath, last_modified in remote.items():
        if snapshot.get(relpath) != last_modified or not (source / relpath).is_file():
            _download_file(files_url, relpath, version, token, source / relpath, logger)
            downloaded += 1

    # Reconcile against the whole source tree, not just the snapshot: delete any
    # file no longer in the artifact — covers files removed between versions AND
    # leftovers from a prior transport scheme on a migrated app_dir.
    removed = 0
    for path in list(source.rglob("*")):
        if path.is_file():
            relpath = path.relative_to(source).as_posix()
            if relpath not in remote:
                path.unlink()
                _prune_empty_dirs(path.parent, source)
                removed += 1

    snapshot_path.write_text(json.dumps(remote))
    logger.info(
        f"BioEngine: synced source from Hypha "
        f"({downloaded} downloaded, {removed} removed, {len(remote)} total) → {source}"
    )


#: Set by the worker on the env it gives REPLICAS, once its staging tasks have materialised source
#: on every alive node. Spelled out here rather than imported from the worker, for the same reason
#: as the other keys in this module: the replica gets bioengine as raw source and must not import
#: worker modules to learn one string.
#:
#: IT HAS TO BE AN EXPLICIT SIGNAL, not inferred from "source looks present". A committed artifact
#: version is immutable, but a *staged* version is not and a deploy at "latest" can resolve to the
#: same version string with changed content — which is why the worker also fingerprints the synced
#: tree (``source_signature``). An unconditional "source exists and .version matches ⇒ use it"
#: short-circuit therefore serves STALE code on a re-staged version, and that is not hypothetical:
#: it broke three ``tests/_app/test_replica_init.py`` cases that assert incremental re-sync
#: (changed timestamp, removed file, missing local file) the first time it was written that way.
#:
#: With the flag, the division is clean. A process that CAN sync (the worker's staging task, which
#: holds the download token as a task argument) is never given the flag and always re-syncs. A
#: process that CANNOT sync (a replica, which by design holds no credential) is given it and reads
#: what the worker just staged for this exact deploy.
_SOURCE_STAGED_KEY = "BIOENGINE_SOURCE_STAGED"


def _is_staged(source: Path, version_marker: Path, identity: str) -> bool:
    """Is ``source`` already a *complete* materialisation of ``identity``?

    True iff the directory exists, is non-empty, and ``.version`` holds exactly
    this ``<artifact_id>@<version>``.

    **Completeness comes from the write ORDER, not from inspecting the tree.**
    :func:`_sync_source_from_hypha` writes every file first and its caller
    writes ``.version`` only after it returns, so a matching marker cannot
    coexist with a half-populated tree. Checking file *counts* instead would be
    the vacuous version of this test — an empty directory and a complete one
    both "exist".

    The non-empty check is belt-and-braces for a tree emptied by hand after the
    marker was written; it is a cheap `any()`, not a full walk.
    """
    if not version_marker.is_file() or not source.is_dir():
        return False
    try:
        if version_marker.read_text().strip() != identity:
            return False
    except OSError:
        return False
    return any(source.iterdir())


def _ensure_source(app_dir: Path, version: str, logger: logging.Logger) -> Path:
    """Populate ``app_dir/source`` so it matches the artifact's ``version``.

    Backends, picked in order:

    1. **Already staged**, and only when :data:`_SOURCE_STAGED_KEY` says the
       worker staged it for this deploy. Needs **no credential**, which is the
       whole point: it is what lets the per-replica ``runtime_env`` stop
       carrying a download token (see
       :meth:`bioengine.apps.builder.AppBuilder._stage_source_on_nodes`).
    2. ``BIOENGINE_LOCAL_ARTIFACT_PATH`` — dev override, short-circuits to a
       locally-mounted artifact root.
    3. ``BIOENGINE_ARTIFACT_FILES_URL`` (+ optional ``_DOWNLOAD_TOKEN``) —
       per-file incremental Hypha sync (:func:`_sync_source_from_hypha`).

    Backend 1 is checked first so the credential-free path wins where it is
    offered, and the flag is what stops it winning where it must not: without
    it every caller keeps the previous behaviour of always re-syncing, so a
    re-staged version or a redeploy at "latest" cannot be served stale. See
    :data:`_SOURCE_STAGED_KEY`.

    A single ``fcntl`` lock per node serialises concurrent same-node starts.
    """
    source = app_dir / "source"
    version_marker = app_dir / ".version"
    snapshot_path = app_dir / ".source_snapshot.json"
    app_dir.mkdir(parents=True, exist_ok=True)

    artifact_id_for_identity = os.environ.get("BIOENGINE_ARTIFACT_ID", "")
    identity_for_stage = f"{artifact_id_for_identity}@{version}"
    staged_declared = bool(os.environ.get(_SOURCE_STAGED_KEY))
    if staged_declared and _is_staged(source, version_marker, identity_for_stage):
        logger.info(f"BioEngine: source already staged for {version} → {source}")
        return source

    # Dev override: ``BIOENGINE_LOCAL_ARTIFACT_PATH`` points at a directory
    # holding ``<artifact_alias>/`` subdirs with the raw app sources.
    local_root_env = os.environ.get("BIOENGINE_LOCAL_ARTIFACT_PATH")
    artifact_id = os.environ.get("BIOENGINE_ARTIFACT_ID", "")
    # Identity keys on the artifact too, not just the version string, so a
    # reused application_id across artifacts can't serve stale source.
    identity = f"{artifact_id}@{version}"
    if local_root_env and artifact_id:
        alias = artifact_id.split("/")[-1]
        candidate = Path(local_root_env) / alias
        if candidate.is_dir():
            if source.exists():
                shutil.rmtree(source)
            shutil.copytree(
                candidate, source, ignore=shutil.ignore_patterns("__pycache__", ".git")
            )
            version_marker.write_text(identity)
            logger.info(f"BioEngine: source mirrored from local path {candidate}")
            return source

    import fcntl

    lock_path = app_dir / ".lock"
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            # Another holder of the lock may have staged it while we waited.
            if staged_declared and _is_staged(source, version_marker, identity):
                logger.info(
                    f"BioEngine: source staged by a concurrent start for "
                    f"{version} → {source}"
                )
                return source
            files_url = os.environ.get("BIOENGINE_ARTIFACT_FILES_URL")
            if not files_url:
                raise RuntimeError(
                    "BIOENGINE_ARTIFACT_FILES_URL is not set, and no staged "
                    "source was found at "
                    f"{source} for {identity}; cannot materialise the app "
                    "source. A replica reaches this only if the worker's "
                    "staging step did not run for this node."
                )
            token = os.environ.get("BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN") or None
            try:
                _sync_source_from_hypha(
                    files_url, version, token, source, snapshot_path, logger
                )
            except Exception as exc:
                if token:
                    raise
                # No staged source AND no credential. For a PUBLIC artifact the
                # anonymous read above succeeds and we never get here, so this
                # branch means a private artifact on a node the worker's staging
                # step did not reach — a node that joined the cluster after the
                # deploy, on a filesystem that is per-node rather than shared.
                #
                # It is raised with that diagnosis attached because the bare
                # HTTP error is a 401/403 that reads like a broken token rather
                # than like a missing stage, and the replica CANNOT self-heal:
                # this hook runs before ``__init__``, which is exactly why the
                # download token is not delivered through the proxy actor the
                # way deploy-time secrets are.
                raise RuntimeError(
                    f"Could not materialise source for {identity}: no staged "
                    f"copy at {source}, and the anonymous Hypha read failed "
                    f"({exc}). The replica holds no download credential by "
                    "design. If this node joined after the deploy and "
                    "/home/bioengine is per-node rather than shared, redeploy "
                    "the application so the worker stages source on it."
                ) from exc
            version_marker.write_text(identity)
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
