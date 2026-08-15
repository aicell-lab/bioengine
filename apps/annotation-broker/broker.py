"""annotation-broker — standing authority for shared annotation datasets.

CPU-only BioEngine app. Holds the ``bioimage-io`` workspace token and owns
role/permission metadata, presigned URL handout, label folder creation, and
Hypha ACL mirroring for every dataset artifact under
``bioimage-io/colab-annotations``. Historically this logic lived in the host
browser's Pyodide kernel (``colab_service.py``) and died with the host's
tab; this app makes annotation sessions survive independent of any single
browser tab.

This module is the thin transport layer: it owns the Hypha connection and
artifact-manager calls. All decision logic (role resolution, path building,
metadata read/write, the ACL-permissions mirror, the stage-mode retry
wrapper) lives in ``broker_core.py``, which is plain Python and unit-tested
without Ray or a live Hypha connection.
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import bioengine
import httpx
from hypha_rpc import connect_to_server
from pydantic import Field

import broker_core as core

logger = bioengine.logger

SERVER_URL = "https://hypha.aicell.io"
COLLECTION_ID = "bioimage-io/colab-annotations"
# PVC-persistent in the deployed environment, so registered datasets survive
# app restarts.
STATE_ROOT = Path.home() / "annotation_broker"
MANIFEST_CACHE_TTL_S = 60.0


def _read_pip(name: str) -> List[str]:
    text = (Path(__file__).parent / name).read_text()
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@bioengine.app(
    num_cpus=1,
    memory_mb=512,
    pip=_read_pip("requirements-broker.txt"),
    max_ongoing_requests=20,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 1,
    },
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    graceful_shutdown_timeout_s=60.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class AnnotationBroker:
    """Role-gated authority for ``bioimage-io/colab-annotations`` datasets."""

    def __init__(self) -> None:
        self._hypha_token = os.getenv("HYPHA_TOKEN")
        if not self._hypha_token:
            raise RuntimeError("HYPHA_TOKEN environment variable is not set")
        self._hypha = None
        self._am = None
        self._http: Optional[httpx.AsyncClient] = None
        # artifact_id -> (cached_at, name)
        self._manifest_name_cache: Dict[str, tuple] = {}

    @bioengine.async_init
    async def _connect(self) -> None:
        self._hypha = await connect_to_server(
            {"server_url": SERVER_URL, "token": self._hypha_token}
        )
        self._am = await self._hypha.get_service("public/artifact-manager")
        self._http = httpx.AsyncClient(timeout=60.0)
        logger.info(f"annotation-broker connected to Hypha server at {SERVER_URL}")

    @bioengine.health_check
    async def _health(self) -> None:
        if self._am is None:
            raise RuntimeError("artifact-manager service not connected")

    # === context / role helpers ===

    @staticmethod
    def _ctx_user(context: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        user = (context or {}).get("user") or {}
        # Hypha marks anonymous connections with is_anonymous and gives them
        # ids like "http-anonymous"; treat them as identity-less so they can
        # never hold a role or a user folder.
        if user.get("is_anonymous"):
            return {"id": None, "email": None}
        return {"id": user.get("id"), "email": user.get("email")}

    @staticmethod
    def _canonical_id(artifact_id: str) -> str:
        """Frontend URLs carry the bare alias (no 'bioimage-io/' prefix);
        canonicalize to the full artifact id for all broker state and
        artifact-manager calls."""
        return artifact_id if "/" in artifact_id else f"bioimage-io/{artifact_id}"

    def _metadata_or_raise(self, artifact_id: str) -> Dict[str, Any]:
        meta = core.read_metadata(artifact_id, root=STATE_ROOT)
        if meta is None:
            raise ValueError(f"Dataset '{artifact_id}' is not registered with the broker.")
        return meta

    def _require_role(
        self, context: Optional[Dict[str, Any]], artifact_id: str, minimum: str
    ) -> tuple:
        """Load metadata, resolve the caller's role, and enforce *minimum*.

        Returns ``(metadata, caller, role)``. Raises ``PermissionError`` if
        the caller's role is below *minimum*.
        """
        meta = self._metadata_or_raise(artifact_id)
        caller = self._ctx_user(context)
        role = core.resolve_role(meta, caller["id"], caller["email"])
        if not core.role_at_least(role, minimum):
            who = caller["id"] or caller["email"] or "anonymous"
            raise PermissionError(
                f"User '{who}' has role '{role}' on '{artifact_id}'; "
                f"'{minimum}' or higher is required."
            )
        return meta, caller, role

    # === artifact-manager helpers ===
    # Call-shape notes (see colab_service.py / micro-sam's entry.py):
    #   - get_file needs KEYWORD args (artifact_id=..., file_path=...) — this
    #     bit a previous implementer.
    #   - put_file/list_files/commit take artifact_id positionally.
    #   - read/edit read+write the STAGED version via stage=True.

    async def _ensure_staged(self, artifact_id: str, fn) -> Any:
        async def call():
            return await fn()

        async def restage():
            await self._am.edit(artifact_id=artifact_id, stage=True)

        return await core.ensure_staged(call, restage)

    async def _list_files_safe(self, artifact_id: str, dir_path: str) -> List[Dict[str, Any]]:
        try:
            files = await self._am.list_files(artifact_id, dir_path=dir_path, stage=True)
            return files or []
        except Exception:
            return []

    async def _file_exists(self, artifact_id: str, file_path: str) -> bool:
        # Presigning does not check object existence, so a get_file probe can
        # succeed for a key that was never written. List the parent dir instead.
        dir_path, _, name = file_path.rpartition("/")
        entries = await self._list_files_safe(artifact_id, dir_path)
        return any(
            (e.get("name") if isinstance(e, dict) else str(e)) == name for e in entries
        )

    async def _read_json_file(self, artifact_id: str, file_path: str, default: Any) -> Any:
        try:
            url = await self._am.get_file(artifact_id=artifact_id, file_path=file_path, stage=True)
        except Exception:
            return default
        resp = await self._http.get(url)
        if resp.status_code == 404:
            return default
        resp.raise_for_status()
        return resp.json()

    async def _write_json_file(self, artifact_id: str, file_path: str, data: Any) -> None:
        url = await self._am.put_file(artifact_id, file_path=file_path)
        resp = await self._http.put(url, content=json.dumps(data).encode("utf-8"))
        resp.raise_for_status()

    async def _put_bytes(self, artifact_id: str, file_path: str, content: bytes) -> None:
        url = await self._am.put_file(artifact_id, file_path=file_path)
        resp = await self._http.put(url, content=content)
        resp.raise_for_status()

    async def _read_artifact_manifest(self, artifact_id: str) -> Dict[str, Any]:
        artifact = await self._am.read(artifact_id=artifact_id, stage=True)
        if isinstance(artifact, dict):
            return dict(artifact.get("manifest") or {})
        return dict(getattr(artifact, "manifest", {}) or {})

    async def _cached_manifest_name(self, artifact_id: str) -> str:
        cached = self._manifest_name_cache.get(artifact_id)
        now = time.time()
        if cached is not None and core.is_cache_fresh(cached[0], ttl_s=MANIFEST_CACHE_TTL_S, now=now):
            return cached[1]
        try:
            manifest = await self._read_artifact_manifest(artifact_id)
            name = manifest.get("name") or artifact_id
        except Exception as exc:
            logger.warning(f"annotation-broker: failed to read manifest for '{artifact_id}': {exc}")
            name = artifact_id
        self._manifest_name_cache[artifact_id] = (now, name)
        return name

    _MISSING_FILE_RE = re.compile(r"File '([^']+)' does not exist")

    async def _commit_tolerating_pending_uploads(self, artifact_id: str) -> None:
        """Commit, tolerating staged files that were minted via put_file but
        not (yet) uploaded to S3.

        put_file registers the file in the staged manifest immediately, and
        Hypha's commit validates every staged file against S3 — so a single
        in-flight or abandoned annotation/embedding upload would make every
        set_role/set_public fail. Strategy: retry with short waits so
        in-flight uploads can land, then prune entries that never arrived
        (an annotator whose upload lands after the prune sees the pair
        missing on the next index refresh and can simply re-save).
        """
        delays = [5.0, 10.0]
        prunes = 0
        while True:
            try:
                await self._am.commit(artifact_id)
                return
            except Exception as exc:
                match = self._MISSING_FILE_RE.search(str(exc))
                if not match:
                    raise
                if delays:
                    await asyncio.sleep(delays.pop(0))
                    continue
                if prunes >= 20:
                    raise
                missing = match.group(1)
                logger.warning(
                    f"annotation-broker: pruning never-uploaded staged file "
                    f"'{missing}' from '{artifact_id}' to unblock commit"
                )
                await self._am.remove_file(artifact_id=artifact_id, file_path=missing)
                prunes += 1

    async def _apply_permissions(self, artifact_id: str, meta: Dict[str, Any]) -> None:
        """Mirror broker-metadata roles into the Hypha ACL.

        Cycle (owned here, per the architecture plan): read current
        permissions -> edit(config=...) writes the new ACL -> commit ->
        edit(stage=True) re-stages, all wrapped in the stage-mode retry.
        """
        perms = core.build_acl_permissions(meta)

        async def _do_edit():
            current = await self._am.read(artifact_id=artifact_id, stage=True)
            current_config = (
                dict(current.get("config") or {})
                if isinstance(current, dict)
                else dict(getattr(current, "config", {}) or {})
            )
            current_config["permissions"] = perms
            await self._am.edit(artifact_id=artifact_id, config=current_config, stage=True)
            await self._commit_tolerating_pending_uploads(artifact_id)
            await self._am.edit(artifact_id=artifact_id, stage=True)

        await self._ensure_staged(artifact_id, _do_edit)

    # === public service API ===

    @bioengine.method(context=True)
    async def register_dataset(
        self,
        artifact_id: str = Field(..., description="Full artifact id, e.g. 'bioimage-io/annotation-abc123'."),
        context=None,
    ) -> Dict[str, Any]:
        """Register a freshly-created dataset artifact with the broker.

        Only the artifact's own owner (``manifest.owner`` or ``created_by``)
        may register it. Idempotent: calling it again on an already
        registered dataset just returns the existing record.
        """
        artifact_id = self._canonical_id(artifact_id)
        caller = self._ctx_user(context)
        existing = core.read_metadata(artifact_id, root=STATE_ROOT)
        if existing is not None:
            return existing

        artifact = await self._am.read(artifact_id=artifact_id, stage=True)
        if isinstance(artifact, dict):
            manifest = dict(artifact.get("manifest") or {})
            created_by = artifact.get("created_by")
        else:
            manifest = dict(getattr(artifact, "manifest", {}) or {})
            created_by = getattr(artifact, "created_by", None)
        # colab_service.py historically wrote created_by into the manifest,
        # not the artifact top level.
        created_by = created_by or manifest.get("created_by")

        if not core.caller_matches_artifact_owner(manifest, created_by, caller["id"], caller["email"]):
            who = caller["id"] or caller["email"] or "anonymous"
            raise PermissionError(f"User '{who}' does not own artifact '{artifact_id}'; cannot register it.")

        owner = manifest.get("owner") or {"id": caller["id"], "email": caller["email"]}
        meta = core.new_metadata(artifact_id, owner=owner)
        return core.write_metadata(meta, root=STATE_ROOT)

    @bioengine.method(context=True)
    async def list_my_datasets(
        self, context=None
    ) -> Dict[str, Any]:
        """Datasets where the caller is a manager or annotator (NOT owner —
        the frontend gets the caller's own datasets from the collection)."""
        caller = self._ctx_user(context)
        shared: List[Dict[str, Any]] = []
        for artifact_id in core.list_dataset_ids(root=STATE_ROOT):
            meta = core.read_metadata(artifact_id, root=STATE_ROOT)
            if meta is None:
                continue
            role = core.resolve_role(meta, caller["id"], caller["email"])
            if role not in ("manager", "annotator"):
                continue
            name = await self._cached_manifest_name(artifact_id)
            shared.append(
                {
                    "artifact_id": artifact_id,
                    "name": name,
                    "role": role,
                    "labels": meta.get("labels", []),
                }
            )
        return {"shared": shared}

    @bioengine.method(context=True)
    async def get_dataset(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        context=None,
    ) -> Dict[str, Any]:
        """Broker metadata + the caller's resolved role, for gating the
        dataset-overview route. Annotators get labels and their role but not
        the member lists (those are for the manager-side sharing panel)."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, role = self._require_role(context, artifact_id, "annotator")
        result = dict(meta)
        result["role"] = role
        if not core.role_at_least(role, "manager"):
            result.pop("managers", None)
            result.pop("annotators", None)
            result.pop("access_requests", None)
        return result

    @bioengine.method(context=True)
    async def delete_label(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        name: str = Field(..., description="Label name to delete."),
        context=None,
    ) -> Dict[str, Any]:
        """Delete a label: remove it from the broker metadata AND delete the
        whole ``label_<name>/`` folder (metadata.json, users.json, every
        user's annotation files) via the workspace token. Frontend deletion
        with the user token was unreliable and left the broker's label list
        stale, which made deleted labels reappear."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, _role = self._require_role(context, artifact_id, "manager")
        folder = core.label_folder(name)

        failed: List[str] = []

        async def _rm_dir(dir_path: str) -> None:
            entries = await self._list_files_safe(artifact_id, dir_path)
            for entry in entries:
                n = entry.get("name") if isinstance(entry, dict) else str(entry)
                if not n:
                    continue
                is_dir = isinstance(entry, dict) and (
                    entry.get("type") == "directory" or entry.get("is_dir")
                )
                path = f"{dir_path}/{n}"
                if is_dir:
                    await _rm_dir(path)
                else:
                    try:
                        await self._am.remove_file(artifact_id=artifact_id, file_path=path)
                    except Exception as exc:
                        failed.append(path)
                        logger.warning(
                            f"annotation-broker: failed to remove '{path}' from "
                            f"'{artifact_id}': {exc}"
                        )

        async def _do():
            await _rm_dir(folder)

        await self._ensure_staged(artifact_id, _do)
        meta["labels"] = [l for l in meta.get("labels", []) or [] if l.get("name") != name]
        meta = core.write_metadata(meta, root=STATE_ROOT)
        result = dict(meta)
        result["failed_files"] = failed
        return result

    @bioengine.method(context=True)
    async def update_sharing(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        add: Optional[List[Dict[str, Any]]] = Field(
            None,
            description="Users to add/move: [{'user': {'id'|'email': ...}, 'role': 'annotator'|'manager'}].",
        ),
        remove: Optional[List[Dict[str, Any]]] = Field(
            None, description="Users to remove: [{'id'|'email': ...}]."
        ),
        set_public: Optional[bool] = Field(
            None, description="New public flag, or omit to leave unchanged."
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Apply a batch of sharing changes with a SINGLE ACL commit +
        re-stage cycle (the Share dialog's Apply button). Same rules as
        set_role/remove_user: only the owner may add or remove managers.
        Granted users' pending access requests are cleared. A call with no
        changes returns the metadata without touching the artifact."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, caller_role = self._require_role(context, artifact_id, "manager")
        # Omitted optional params arrive as pydantic FieldInfo (truthy!), not
        # None — the classic bioengine decorator trap. Normalize by type.
        add = add if isinstance(add, list) else []
        remove = remove if isinstance(remove, list) else []
        if not isinstance(set_public, bool):
            set_public = None

        # Validate everything before mutating anything.
        for entry in add:
            role = entry.get("role")
            user = entry.get("user") or {}
            if role not in core.ROLE_LIST_KEYS:
                raise ValueError(f"Invalid role '{role}'; expected 'manager' or 'annotator'.")
            if not user.get("id") and not user.get("email"):
                raise ValueError("Each add entry needs a user with an id or email.")
            target_role = core.resolve_role(meta, user.get("id"), user.get("email"))
            if caller_role != "owner" and (role == "manager" or target_role == "manager"):
                raise PermissionError("Only the dataset owner may add or remove managers.")
        for user in remove:
            target_role = core.resolve_role(meta, user.get("id"), user.get("email"))
            if caller_role != "owner" and target_role == "manager":
                raise PermissionError("Only the dataset owner may remove a manager.")

        changed = bool(add or remove) or (
            set_public is not None and bool(set_public) != bool(meta.get("public"))
        )
        for entry in add:
            core.set_user_role(meta, entry["user"], entry["role"])
            core.remove_access_request(meta, entry["user"])
        for user in remove:
            core.remove_user_role(meta, user)
        if set_public is not None:
            meta["public"] = bool(set_public)

        if not changed:
            return meta
        meta = core.write_metadata(meta, root=STATE_ROOT)
        await self._apply_permissions(artifact_id, meta)
        return meta

    @bioengine.method(context=True)
    async def request_access(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        role: str = Field("annotator", description="Requested role: 'annotator' or 'manager'."),
        context=None,
    ) -> Dict[str, Any]:
        """Register an access request from a logged-in user without a role
        (the annotate page offers this when the broker denies access). The
        dataset owner sees pending requests in the Share dialog and grants
        them via set_role, which clears the request."""
        artifact_id = self._canonical_id(artifact_id)
        meta = self._metadata_or_raise(artifact_id)
        caller = self._ctx_user(context)
        if not caller["id"] and not caller["email"]:
            raise PermissionError("Log in to request access to this dataset.")
        current = core.resolve_role(meta, caller["id"], caller["email"])
        if core.role_at_least(current, "annotator"):
            return {"status": "already_has_access", "role": current}
        if not isinstance(role, str):
            role = "annotator"
        core.add_access_request(meta, caller, role, core.now_iso())
        core.write_metadata(meta, root=STATE_ROOT)
        return {"status": "requested", "requested_role": role}

    @bioengine.method(context=True)
    async def dismiss_access_request(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        user: Dict[str, Any] = Field(..., description="{'id': ...} or {'email': ...}."),
        context=None,
    ) -> Dict[str, Any]:
        """Reject a pending access request without granting a role."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, _role = self._require_role(context, artifact_id, "manager")
        core.remove_access_request(meta, user)
        meta = core.write_metadata(meta, root=STATE_ROOT)
        return meta

    @bioengine.method(context=True)
    async def set_role(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        user: Dict[str, Any] = Field(..., description="{'id': ...} or {'email': ...}."),
        role: str = Field(..., description="'manager' or 'annotator'."),
        context=None,
    ) -> Dict[str, Any]:
        """Grant *user* the given role. Only the dataset owner may add or
        remove the manager role (either direction)."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, caller_role = self._require_role(context, artifact_id, "manager")
        if role not in core.ROLE_LIST_KEYS:
            raise ValueError(f"Invalid role '{role}'; expected 'manager' or 'annotator'.")

        target_role = core.resolve_role(meta, user.get("id"), user.get("email"))
        if caller_role != "owner" and (role == "manager" or target_role == "manager"):
            raise PermissionError("Only the dataset owner may add or remove managers.")

        core.set_user_role(meta, user, role)
        core.remove_access_request(meta, user)
        meta = core.write_metadata(meta, root=STATE_ROOT)
        await self._apply_permissions(artifact_id, meta)
        return meta

    @bioengine.method(context=True)
    async def remove_user(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        user: Dict[str, Any] = Field(..., description="{'id': ...} or {'email': ...}."),
        context=None,
    ) -> Dict[str, Any]:
        """Remove *user* from the dataset's managers/annotators. Only the
        owner may remove a manager."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, caller_role = self._require_role(context, artifact_id, "manager")
        target_role = core.resolve_role(meta, user.get("id"), user.get("email"))
        if caller_role != "owner" and target_role == "manager":
            raise PermissionError("Only the dataset owner may remove a manager.")

        core.remove_user_role(meta, user)
        meta = core.write_metadata(meta, root=STATE_ROOT)
        await self._apply_permissions(artifact_id, meta)
        return meta

    @bioengine.method(context=True)
    async def set_public(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        is_public: bool = Field(..., description="Whether anyone can read (and, if logged in, annotate) this dataset."),
        context=None,
    ) -> Dict[str, Any]:
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, _role = self._require_role(context, artifact_id, "manager")
        meta["public"] = bool(is_public)
        meta = core.write_metadata(meta, root=STATE_ROOT)
        await self._apply_permissions(artifact_id, meta)
        return meta

    @bioengine.method(context=True)
    async def create_label(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        name: str = Field(..., description="Label name, must match ^[a-z0-9._-]+$."),
        description: str = Field("", description="Human-readable label description."),
        context=None,
    ) -> Dict[str, Any]:
        """Create a new annotation label: broker metadata + a
        ``label_<name>/metadata.json`` file carrying the description (and any
        future per-label info). The artifact manifest is NOT touched, and no
        empty placeholder files are written."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, _role = self._require_role(context, artifact_id, "manager")
        core.add_label(meta, name, description)  # raises ValueError on a bad name
        meta = core.write_metadata(meta, root=STATE_ROOT)

        async def _put_label_metadata():
            await self._write_json_file(
                artifact_id,
                core.label_metadata_path(name),
                {"name": name, "description": description, "created_at": core.now_iso()},
            )

        await self._ensure_staged(artifact_id, _put_label_metadata)
        return meta

    @bioengine.method(context=True)
    async def get_dataset_index(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        context=None,
    ) -> Dict[str, Any]:
        """One-shot payload for the annotate page: images, embeddings, labels,
        and the caller's own latest annotation per (label, stem). Callers
        should never have to reason about stage/permission edge cases
        themselves.

        Role ``public`` (anonymous visitor on a public dataset) gets the
        read-only payload with ``my_annotations`` omitted — annotating always
        requires a login.
        """
        artifact_id = self._canonical_id(artifact_id)
        meta, caller, role = self._require_role(context, artifact_id, "public")

        # Presigned-URL minting is one RPC round trip per file; run them with
        # bounded concurrency so large datasets don't take N sequential trips.
        sem = asyncio.Semaphore(16)

        async def _mint(file_path: str) -> str:
            async with sem:
                return await self._am.get_file(
                    artifact_id=artifact_id, file_path=file_path, stage=True
                )

        image_entries = await self._list_files_safe(artifact_id, "images")
        image_names = []
        for entry in image_entries:
            name = entry.get("name", "") if isinstance(entry, dict) else str(entry)
            if not name or (isinstance(entry, dict) and (entry.get("type") == "directory" or entry.get("is_dir"))):
                continue
            image_names.append(name)
        image_urls = await asyncio.gather(*[_mint(f"images/{n}") for n in image_names])
        images = [
            {"stem": Path(n).stem, "read_url": url}
            for n, url in zip(image_names, image_urls)
        ]

        embedding_entries = await self._list_files_safe(artifact_id, "embeddings")
        parsed_embeddings = []
        for entry in embedding_entries:
            name = entry.get("name", "") if isinstance(entry, dict) else str(entry)
            parsed = core.parse_embedding_filename(name)
            if parsed:
                parsed_embeddings.append((name, parsed))
        embedding_urls = await asyncio.gather(
            *[_mint(f"embeddings/{n}") for n, _ in parsed_embeddings]
        )
        embeddings: Dict[str, Any] = {
            parsed["stem"]: {"model_type": parsed["model_type"], "read_url": url}
            for (_, parsed), url in zip(parsed_embeddings, embedding_urls)
        }

        my_annotations: Dict[str, Any] = {}
        if role != "public":
            for label in meta.get("labels", []):
                label_name = label.get("name")
                if not label_name:
                    continue
                dir_path = core.user_label_dir(label_name, caller["id"])
                entries = await self._list_files_safe(artifact_id, dir_path)
                filenames = [e.get("name", "") if isinstance(e, dict) else str(e) for e in entries]
                pairs = core.latest_pairs_by_stem(filenames)
                if not pairs:
                    continue
                stems = list(pairs.keys())
                geojson_urls = await asyncio.gather(
                    *[_mint(f"{dir_path}/{pairs[s]['geojson']}") for s in stems]
                )
                my_annotations[label_name] = {
                    stem: {"latest_ts": pairs[stem]["timestamp"], "geojson_read_url": url}
                    for stem, url in zip(stems, geojson_urls)
                }

        result = {
            "images": images,
            "embeddings": embeddings,
            "labels": meta.get("labels", []),
            "my_annotations": my_annotations,
        }
        result["role"] = role
        return result

    @bioengine.method(context=True)
    async def get_image_url(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        image_stem: str = Field(..., description="Image stem (filename without extension)."),
        context=None,
    ) -> Dict[str, Any]:
        """Single presigned GET URL for one image. Featherweight first call
        for the annotate page so the image renders before the full dataset
        index (which mints URLs for every file) has been assembled."""
        artifact_id = self._canonical_id(artifact_id)
        self._require_role(context, artifact_id, "public")
        read_url = await self._am.get_file(
            artifact_id=artifact_id, file_path=core.image_path(image_stem), stage=True
        )
        return {"stem": image_stem, "read_url": read_url}

    @bioengine.method(context=True)
    async def get_embedding_urls(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        image_stem: str = Field(..., description="Image stem (filename without extension)."),
        model_type: str = Field(..., description="μSAM model type, e.g. 'vit_l_lm'."),
        context=None,
    ) -> Dict[str, Any]:
        artifact_id = self._canonical_id(artifact_id)
        """Existing embedding read URL, or URLs for the micro-sam app to
        compute and upload a new one."""
        self._require_role(context, artifact_id, "annotator")
        emb_path = core.embedding_path(image_stem, model_type)
        if await self._file_exists(artifact_id, emb_path):
            read_url = await self._am.get_file(artifact_id=artifact_id, file_path=emb_path, stage=True)
            return {"exists": True, "read_url": read_url}

        image_read_url = await self._am.get_file(
            artifact_id=artifact_id, file_path=core.image_path(image_stem), stage=True
        )
        embedding_put_url = await self._am.put_file(artifact_id, file_path=emb_path)
        return {
            "exists": False,
            "image_read_url": image_read_url,
            "embedding_put_url": embedding_put_url,
        }

    @bioengine.method(context=True)
    async def get_save_urls(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: str = Field(..., description="Label name."),
        image_stem: str = Field(..., description="Image stem (filename without extension)."),
        context=None,
    ) -> Dict[str, Any]:
        artifact_id = self._canonical_id(artifact_id)
        """Presigned PUT URLs for a new (png, geojson) annotation pair,
        sharing one server-generated UTC timestamp. Also upserts
        ``label_<label>/users.json`` with the caller's identity."""
        meta, caller, _role = self._require_role(context, artifact_id, "annotator")
        if not any(l.get("name") == label for l in meta.get("labels", [])):
            raise ValueError(f"Unknown label '{label}' for dataset '{artifact_id}'.")

        timestamp = core.new_timestamp()
        paths = core.annotation_save_paths(label, caller["id"], image_stem, timestamp)
        png_put_url = await self._am.put_file(artifact_id, file_path=paths["png"])
        geojson_put_url = await self._am.put_file(artifact_id, file_path=paths["geojson"])

        safe_user = core.sanitize_user_id(caller["id"])
        users_path = core.label_users_path(label)

        async def _upsert_users():
            current = await self._read_json_file(artifact_id, users_path, default={})
            updated = core.upsert_label_user(current, safe_user, caller["id"], caller["email"])
            await self._write_json_file(artifact_id, users_path, updated)

        await self._ensure_staged(artifact_id, _upsert_users)

        return {
            "timestamp": timestamp,
            "png_put_url": png_put_url,
            "geojson_put_url": geojson_put_url,
        }

    @bioengine.method(context=True)
    async def delete_dataset_record(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        context=None,
    ) -> Dict[str, Any]:
        artifact_id = self._canonical_id(artifact_id)
        """Broker-side cleanup after the frontend deletes the underlying
        artifact."""
        self._require_role(context, artifact_id, "owner")
        deleted = core.delete_metadata(artifact_id, root=STATE_ROOT)
        self._manifest_name_cache.pop(artifact_id, None)
        return {"deleted": deleted}
