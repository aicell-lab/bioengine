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
* ``BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN`` — read token (absent ⇒ anonymous/public)

Hypha is already a hard dependency (an app cannot register its service without
it), so pulling source from Hypha at replica start adds no new availability
surface: if Hypha is unreachable the replica fails to materialise, the
deployment fails, and ``auto_redeploy`` retries.
"""

from __future__ import annotations

import fnmatch
import json
import time
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


# How long a replica waits for a concurrent replica to publish the same identity before it
# either takes over a stale claim or refuses. A full cold materialisation of a large app is
# minutes, not seconds, so the wait is generous and the takeover threshold is longer still.
_CLAIM_WAIT_SECONDS = 300.0
_CLAIM_STALE_SECONDS = 900.0
_CLAIM_POLL_SECONDS = 1.0


def _ensure_source(app_dir: Path, version: str, logger: logging.Logger) -> Path:
    """Populate ``app_dir/source`` so it matches the artifact's ``version``.

    Backends, picked in order:

    1. ``BIOENGINE_LOCAL_ARTIFACT_PATH`` — dev override, short-circuits to a
       locally-mounted artifact root.
    2. ``BIOENGINE_ARTIFACT_FILES_URL`` (+ optional ``_DOWNLOAD_TOKEN``) —
       per-file incremental Hypha sync (:func:`_sync_source_from_hypha`).

    Concurrent starts on one ``app_dir`` are serialised by an atomically created
    claim directory, not by an advisory lock: the store is shared and may be an
    NFS export whose lock manager is not serving, where ``flock`` answers
    ``ENOLCK`` rather than blocking. A caller that does not win the claim waits
    for the winner's version marker and reuses its tree.
    """
    source = app_dir / "source"
    version_marker = app_dir / ".version"
    snapshot_path = app_dir / ".source_snapshot.json"
    app_dir.mkdir(parents=True, exist_ok=True)

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

    files_url = os.environ.get("BIOENGINE_ARTIFACT_FILES_URL")
    if not files_url:
        raise RuntimeError(
            "BIOENGINE_ARTIFACT_FILES_URL is not set; cannot materialise "
            "the app source. The worker is expected to populate it (and "
            "optionally BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN) when "
            "constructing the runtime_env for the task or deployment."
        )
    token = os.environ.get("BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN") or None

    # NOTE: there is deliberately NO fast path that skips the sync when the version marker
    # already matches. The sync is not only a first-time materialisation -- it reconciles the
    # tree against the artifact on every start, re-downloading a file whose timestamp changed,
    # restoring one deleted locally, and removing one dropped from the artifact. Short-circuiting
    # on the identity broke exactly those three behaviours, which this suite already covered.

    # MUTUAL EXCLUSION WITHOUT ADVISORY LOCKS.
    #
    # Advisory locking is unusable here. ``app_dir`` lives on a shared store that
    # may be an NFS export whose lock manager is not serving, and there
    # ``fcntl.flock`` answers ``ENOLCK`` instead of blocking. Since the caller of
    # ``setup_replica_environment`` swallows exceptions, such a failure takes out
    # the whole replica environment setup and leaves ``HOME`` on a read-only path.
    #
    # ``os.mkdir`` is atomic on NFS and is not part of the advisory-lock
    # protocol, so it cannot answer ENOLCK: exactly one caller creates the
    # directory and everyone else gets ``FileExistsError``. The loser does NOT
    # proceed unlocked — it waits for the winner's version marker and uses the
    # winner's tree, so two replicas never materialise one ``source/`` at once.
    claim = app_dir / ".materialising"
    wait_deadline = time.monotonic() + _CLAIM_WAIT_SECONDS
    while True:
        try:
            os.mkdir(claim)
            break
        except FileExistsError:
            # Someone else holds the claim. Wait for their marker rather than
            # racing them.
            if source.is_dir() and version_marker.is_file():
                try:
                    if version_marker.read_text() == identity:
                        logger.info(
                            f"BioEngine: source materialised concurrently by "
                            f"another replica → {source}"
                        )
                        return source
                except OSError:
                    pass
            if time.monotonic() >= wait_deadline:
                # Take over a claim that is older than any plausible sync: a
                # replica killed mid-materialise would otherwise wedge every
                # later start on this app dir forever.
                age = None
                try:
                    age = time.time() - claim.stat().st_mtime
                except OSError:
                    pass
                if age is not None and age > _CLAIM_STALE_SECONDS:
                    logger.warning(
                        f"BioEngine: taking over a stale materialisation claim "
                        f"{claim} (age {age:.0f}s > {_CLAIM_STALE_SECONDS}s)"
                    )
                    break
                raise TimeoutError(
                    f"another replica has held the materialisation claim {claim} "
                    f"for more than {_CLAIM_WAIT_SECONDS}s without publishing "
                    f"{identity}; refusing to materialise concurrently"
                )
            time.sleep(_CLAIM_POLL_SECONDS)
        except OSError as exc:
            # Never silent, and never mistaken for "no lock available": name the
            # path and the errno so the next reader does not have to guess.
            raise RuntimeError(
                f"cannot create the materialisation claim {claim}: "
                f"{exc.strerror} (errno={exc.errno})"
            ) from exc

    try:
        _sync_source_from_hypha(
            files_url, version, token, source, snapshot_path, logger
        )
        version_marker.write_text(identity)
        return source
    finally:
        try:
            claim.rmdir()
        except OSError:
            # Losing the claim directory would wedge later starts until the
            # staleness takeover above fires, so say so rather than passing.
            logger.warning(
                f"BioEngine: could not release the materialisation claim {claim}"
            )


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

    # HOME/TMPDIR FIRST, before the source sync, because they do not depend on
    # it and the ordering is load-bearing.
    #
    # The caller of this function (the ``sys.meta_path`` finder installed in
    # ``bioengine/__init__.py``) catches every exception, reports that user
    # source will be missing from ``sys.path``, and lets the replica continue.
    # When these assignments came *after* ``_ensure_source``, any failure in the
    # sync skipped them, so the replica ran on with HOME still pointing at the
    # image's own ``/home/ray`` — which is read-only. The replica then registered
    # with the proxy, reported HEALTHY, and died at the first write to
    # ``Path.home()``, far from the real cause.
    #
    # Measured on a production external-cluster worker: ``_ensure_source`` raised
    # ``OSError: [Errno 37] No locks available`` from its ``flock`` on an NFS
    # store, and every training run on that replica then failed with
    # ``OSError: [Errno 30] Read-only file system: '/home/ray/.bioengine'``.
    # Setting these first turns that into a missing-source error, which is loud,
    # immediate and points at what actually went wrong.
    home = app_dir / "home"
    tmp = app_dir / "tmp"
    home.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(home)
    os.environ["TMPDIR"] = str(tmp)
    os.environ["TMP"] = str(tmp)
    os.environ["TEMP"] = str(tmp)

    source = _ensure_source(app_dir, version, logger)

    src_str = str(source)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    logger.info(
        f"BioEngine: replica environment ready "
        f"(source={source}, HOME={home}, TMPDIR={tmp})"
    )


__all__ = ["setup_replica_environment"]
