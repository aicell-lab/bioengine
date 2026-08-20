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

    async def _cached_manifest_info(self, artifact_id: str) -> Dict[str, str]:
        cached = self._manifest_name_cache.get(artifact_id)
        now = time.time()
        if cached is not None and core.is_cache_fresh(cached[0], ttl_s=MANIFEST_CACHE_TTL_S, now=now):
            return cached[1]
        try:
            manifest = await self._read_artifact_manifest(artifact_id)
            info = {
                "name": manifest.get("name") or artifact_id,
                "description": str(manifest.get("description") or ""),
            }
        except Exception as exc:
            logger.warning(f"annotation-broker: failed to read manifest for '{artifact_id}': {exc}")
            info = {"name": artifact_id, "description": ""}
        self._manifest_name_cache[artifact_id] = (now, info)
        return info

    async def _cached_manifest_name(self, artifact_id: str) -> str:
        return (await self._cached_manifest_info(artifact_id))["name"]

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
            info = await self._cached_manifest_info(artifact_id)
            shared.append(
                {
                    "artifact_id": artifact_id,
                    "name": info["name"],
                    "description": info["description"],
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

    # ------------------------------------------------------------------
    # Per-label train/test splits (add-only snapshots, see broker_core)
    # ------------------------------------------------------------------

    async def _image_stems(self, artifact_id: str) -> set:
        """Stems of every file under images/."""
        entries = await self._list_files_safe(artifact_id, "images")
        stems = set()
        for e in entries:
            name = e.get("name", "") if isinstance(e, dict) else str(e)
            if not name or (isinstance(e, dict) and (e.get("type") == "directory" or e.get("is_dir"))):
                continue
            stems.add(Path(name).stem)
        return stems

    async def _annotation_counts(self, artifact_id: str, label: str) -> Dict[str, int]:
        """Complete (png+geojson) annotation saves per stem, summed across
        every user folder of *label*."""
        label_dir = core.label_folder(label)
        entries = await self._list_files_safe(artifact_id, label_dir)
        user_dirs = [
            e.get("name") for e in entries
            if isinstance(e, dict)
            and (e.get("type") == "directory" or e.get("is_dir"))
            and str(e.get("name", "")).startswith("user-")
        ]
        totals: Dict[str, int] = {}
        for user_dir in user_dirs:
            files = await self._list_files_safe(artifact_id, f"{label_dir}/{user_dir}")
            filenames = [f.get("name", "") if isinstance(f, dict) else str(f) for f in files]
            for stem, n in core.count_pairs_by_stem(filenames).items():
                totals[stem] = totals.get(stem, 0) + n
        return totals

    def _require_label(self, meta: Dict[str, Any], label: str) -> None:
        if not any(l.get("name") == label for l in meta.get("labels", []) or []):
            raise ValueError(f"Unknown label '{label}' on this dataset.")

    async def _read_split_or_raise(
        self, artifact_id: str, label: str, name: str
    ) -> Dict[str, Any]:
        doc = await self._read_json_file(
            artifact_id, core.split_path(label, name), default=None
        )
        if not isinstance(doc, dict):
            raise ValueError(f"No split named '{name}' for label '{label}'.")
        return doc

    async def _validate_split_stems_are_images(
        self, artifact_id: str, stems: List[str]
    ) -> None:
        image_stems = await self._image_stems(artifact_id)
        unknown = [s for s in stems if s not in image_stems]
        if unknown:
            raise ValueError(
                "Not images of this dataset: " + ", ".join(sorted(unknown))
            )

    @bioengine.method(context=True)
    async def create_split(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: str = Field(..., description="Label the split belongs to."),
        name: str = Field(..., description="Split name (^[a-z0-9._-]+$)."),
        train: List[str] = Field(..., description="Annotated image stems for training."),
        test: Optional[List[str]] = Field(None, description="Annotated image stems for testing."),
        ratio: Optional[float] = Field(
            None,
            description="Target train fraction (0..1] used for later auto-distribution; defaults to the actual fraction at creation.",
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Create a per-label split: an add-only snapshot of annotation
        progress. Only images with at least one complete annotation save for
        *label* may enter. Fails if the split name already exists."""
        artifact_id = self._canonical_id(artifact_id)
        meta, caller, _role = self._require_role(context, artifact_id, "manager")
        self._require_label(meta, label)
        train = train if isinstance(train, list) else []
        test = test if isinstance(test, list) else []
        ratio = ratio if isinstance(ratio, (int, float)) else None
        if not core.is_valid_split_name(name if isinstance(name, str) else ""):
            raise ValueError(f"Invalid split name {name!r}: must match ^[a-z0-9._-]+$")
        existing = await self._read_json_file(
            artifact_id, core.split_path(label, name), default=None
        )
        if isinstance(existing, dict):
            raise ValueError(
                f"Split '{name}' already exists for label '{label}'. "
                "Extend it with update_split or pick another name."
            )
        stems = [str(s) for s in train] + [str(s) for s in test]
        await self._validate_split_stems_are_images(artifact_id, stems)
        counts = await self._annotation_counts(artifact_id, label)
        doc = core.new_split(
            label, name, train, test, counts,
            caller.get("id") or caller.get("email"), ratio,
        )

        async def _write():
            await self._write_json_file(artifact_id, core.split_path(label, name), doc)

        await self._ensure_staged(artifact_id, _write)
        return doc

    @bioengine.method(context=True)
    async def update_split(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: str = Field(..., description="Label the split belongs to."),
        name: str = Field(..., description="Split name."),
        add_train: Optional[List[str]] = Field(None, description="Annotated stems to add to train."),
        add_test: Optional[List[str]] = Field(None, description="Annotated stems to add to test."),
        context=None,
    ) -> Dict[str, Any]:
        """Add-only extension of an existing split. Added stems must be
        annotated and new to both sets; an image that was ever in train can
        never enter test. Refreshes the annotation-count snapshot for every
        member."""
        artifact_id = self._canonical_id(artifact_id)
        meta, caller, _role = self._require_role(context, artifact_id, "manager")
        self._require_label(meta, label)
        add_train = add_train if isinstance(add_train, list) else []
        add_test = add_test if isinstance(add_test, list) else []
        doc = await self._read_split_or_raise(artifact_id, label, name)
        added = [str(s) for s in add_train] + [str(s) for s in add_test]
        await self._validate_split_stems_are_images(artifact_id, added)
        counts = await self._annotation_counts(artifact_id, label)
        doc = core.extend_split(
            doc, add_train, add_test, counts,
            caller.get("id") or caller.get("email"),
        )

        async def _write():
            await self._write_json_file(artifact_id, core.split_path(label, name), doc)

        await self._ensure_staged(artifact_id, _write)
        return doc

    @bioengine.method(context=True)
    async def set_split_checkpoint(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: str = Field(..., description="Label the split belongs to."),
        name: str = Field(..., description="Split name."),
        checkpoint: Dict[str, Any] = Field(
            ...,
            description="Training lineage record, e.g. {'session_id': ..., 'model_type': ...}.",
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Record the fine-tuning checkpoint a split's training produced.
        A non-null checkpoint locks the split's lineage: the frontend shows
        continued fine-tuning from this checkpoint instead of a base-model
        selector, and the split can no longer be deleted."""
        artifact_id = self._canonical_id(artifact_id)
        _meta, caller, _role = self._require_role(context, artifact_id, "manager")
        if not isinstance(checkpoint, dict) or not checkpoint:
            raise ValueError("checkpoint must be a non-empty object.")
        doc = await self._read_split_or_raise(artifact_id, label, name)
        doc["checkpoint"] = dict(checkpoint)
        doc["checkpoint"]["recorded_at"] = core.now_iso()
        doc["checkpoint"]["recorded_by"] = caller.get("id") or caller.get("email")
        doc["updated_at"] = core.now_iso()
        doc["updated_by"] = caller.get("id") or caller.get("email")

        async def _write():
            await self._write_json_file(artifact_id, core.split_path(label, name), doc)

        await self._ensure_staged(artifact_id, _write)
        return doc

    @bioengine.method(context=True)
    async def get_split(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: str = Field(..., description="Label the split belongs to."),
        name: str = Field(..., description="Split name."),
        context=None,
    ) -> Dict[str, Any]:
        """Read one split document (full: members, counts, history,
        checkpoint)."""
        artifact_id = self._canonical_id(artifact_id)
        self._require_role(context, artifact_id, "annotator")
        return await self._read_split_or_raise(artifact_id, label, name)

    @bioengine.method(context=True)
    async def list_splits(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: Optional[str] = Field(
            None, description="Label to list splits for; omit for ALL labels."
        ),
        context=None,
    ) -> List[Dict[str, Any]]:
        """Compact summaries of the splits of one label, or of every label
        when *label* is omitted (the image-delete guard wants all of them)."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, _role = self._require_role(context, artifact_id, "annotator")
        label = label if isinstance(label, str) else None
        labels = (
            [label]
            if label
            else [l.get("name") for l in meta.get("labels", []) or [] if l.get("name")]
        )
        summaries: List[Dict[str, Any]] = []
        for lbl in labels:
            entries = await self._list_files_safe(artifact_id, core.splits_dir_path(lbl))
            for e in entries:
                fname = e.get("name", "") if isinstance(e, dict) else str(e)
                if not fname.endswith(".json"):
                    continue
                doc = await self._read_json_file(
                    artifact_id, f"{core.splits_dir_path(lbl)}/{fname}", default=None
                )
                if isinstance(doc, dict):
                    summaries.append(core.split_summary(doc))
        return summaries

    @bioengine.method(context=True)
    async def delete_split(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: str = Field(..., description="Label the split belongs to."),
        name: str = Field(..., description="Split name."),
        context=None,
    ) -> Dict[str, Any]:
        """Delete a split that never produced a checkpoint. A trained
        split's lineage is permanent and cannot be deleted."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, _role = self._require_role(context, artifact_id, "manager")
        doc = await self._read_split_or_raise(artifact_id, label, name)
        if doc.get("checkpoint"):
            raise ValueError(
                f"Split '{name}' has a trained checkpoint and cannot be deleted."
            )

        async def _rm():
            await self._am.remove_file(
                artifact_id=artifact_id, file_path=core.split_path(label, name)
            )

        await self._ensure_staged(artifact_id, _rm)
        # Persist the deletion into the committed version (same staleness
        # trap as delete_label: a staged-only removal reappears on restage).
        await self._apply_permissions(artifact_id, meta)
        return {"deleted": True, "label": label, "name": name}

    @bioengine.method(context=True)
    async def get_training_urls(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: str = Field(..., description="Label whose annotations feed the training."),
        split_name: str = Field(..., description="Split that defines the train/test membership."),
        context=None,
    ) -> Dict[str, Any]:
        """Training inputs for micro-sam fine-tuning: for the LATEST pair per
        (user, stem) under the label, presigned image + geojson URLs. ONLY
        stems that are members of the split are included, partitioned by its
        train/test lists."""
        artifact_id = self._canonical_id(artifact_id)
        self._require_role(context, artifact_id, "manager")
        split = await self._read_split_or_raise(artifact_id, label, split_name)
        train_stems = set(split.get("train") or [])
        test_stems = set(split.get("test") or [])

        label_dir = core.label_folder(label)
        entries = await self._list_files_safe(artifact_id, label_dir)
        user_dirs = [
            e.get("name") for e in entries
            if isinstance(e, dict)
            and (e.get("type") == "directory" or e.get("is_dir"))
            and str(e.get("name", "")).startswith("user-")
        ]

        pairs: List[Dict[str, str]] = []
        for user_dir in user_dirs:
            dir_path = f"{label_dir}/{user_dir}"
            files = await self._list_files_safe(artifact_id, dir_path)
            filenames = [f.get("name", "") if isinstance(f, dict) else str(f) for f in files]
            for stem, pair in core.latest_pairs_by_stem(filenames).items():
                if stem not in train_stems and stem not in test_stems:
                    continue
                pairs.append({
                    "stem": stem,
                    "user": user_dir,
                    "geojson_path": f"{dir_path}/{pair['geojson']}",
                })

        sem = asyncio.Semaphore(16)

        async def _mint(file_path: str) -> str:
            async with sem:
                return await self._am.get_file(
                    artifact_id=artifact_id, file_path=file_path, stage=True
                )

        image_urls, geojson_urls = await asyncio.gather(
            asyncio.gather(*[_mint(core.image_path(p["stem"])) for p in pairs]),
            asyncio.gather(*[_mint(p["geojson_path"]) for p in pairs]),
        )
        train_out: List[Dict[str, str]] = []
        test_out: List[Dict[str, str]] = []
        for p, img_url, geo_url in zip(pairs, image_urls, geojson_urls):
            row = {
                "stem": p["stem"],
                "user": p["user"],
                "image_url": img_url,
                "geojson_url": geo_url,
            }
            (test_out if p["stem"] in test_stems else train_out).append(row)
        return {"train": train_out, "test": test_out, "split": core.split_summary(split)}

    @bioengine.method(context=True)
    async def delete_annotation(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: str = Field(..., description="Label the annotation belongs to."),
        user_folder: str = Field(..., description="Sanitized annotator folder, e.g. 'user-github-12345'."),
        stem: str = Field(..., description="Image stem the annotation belongs to."),
        timestamp: str = Field(..., description="Timestamp of the save pair (YYYYMMDD-HHMMSS)."),
        context=None,
    ) -> Dict[str, Any]:
        """Delete one annotation save (its png + geojson pair). Allowed for
        managers/owners, and for the annotation's own author. The removal is
        persisted through a commit cycle (a staged-only removal would
        resurface on the next re-stage)."""
        artifact_id = self._canonical_id(artifact_id)
        meta, caller, role = self._require_role(context, artifact_id, "annotator")
        if not core.is_valid_timestamp(timestamp):
            raise ValueError(f"Invalid timestamp {timestamp!r}; expected YYYYMMDD-HHMMSS.")
        is_own = core.sanitize_user_id(caller.get("id")) == user_folder
        if not core.role_at_least(role, "manager") and not is_own:
            raise PermissionError(
                "Only managers or the annotation's author may delete an annotation."
            )
        base = f"{core.label_folder(label)}/{user_folder}/{stem}-{timestamp}"
        removed = []
        for ext in ("png", "geojson"):
            path = f"{base}.{ext}"

            async def _rm(p=path):
                await self._am.remove_file(artifact_id=artifact_id, file_path=p)

            try:
                await self._ensure_staged(artifact_id, _rm)
                removed.append(path)
            except Exception as exc:
                logger.warning(
                    f"annotation-broker: could not remove '{path}' from '{artifact_id}': {exc}"
                )
        if not removed:
            raise ValueError(f"No annotation files found for {base}.*")
        await self._apply_permissions(artifact_id, meta)
        return {"deleted": removed}

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
        # Persist the deletion into the committed version as well: staged
        # removals alone leave the last committed snapshot holding the label
        # files. Reuses the ACL mirror cycle (commit + re-stage, tolerant of
        # pending uploads); permissions are rebuilt unchanged.
        await self._apply_permissions(artifact_id, meta)
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

        Contract since 0.5.0: NO presigned URLs in the index. Minting one URL
        per file made an 86-image index take ~30 s; the index now returns
        stems and presence only (a few list_files calls), and callers fetch
        URLs on demand via get_image_url / get_embedding_urls /
        get_my_annotation_url.
        """
        artifact_id = self._canonical_id(artifact_id)
        meta, caller, role = self._require_role(context, artifact_id, "public")

        image_entries = await self._list_files_safe(artifact_id, "images")
        images = []
        for entry in image_entries:
            name = entry.get("name", "") if isinstance(entry, dict) else str(entry)
            if not name or (isinstance(entry, dict) and (entry.get("type") == "directory" or entry.get("is_dir"))):
                continue
            images.append({"stem": Path(name).stem})

        # Per-image width/height (0.8.0) so the frontend can flag images
        # below the segmentation models' input minimum without loading
        # pixels. Each stem costs one 24-byte ranged read of the PNG IHDR
        # header, done once and cached in the dataset's broker metadata;
        # failed reads stay uncached and retry on a later index call.
        dims_cache: Dict[str, Any] = dict(meta.get("image_dims") or {})
        missing_dims = [img["stem"] for img in images if img["stem"] not in dims_cache]
        if missing_dims:
            dims_sem = asyncio.Semaphore(16)

            async def _fetch_dims(stem: str):
                async with dims_sem:
                    try:
                        url = await self._am.get_file(
                            artifact_id=artifact_id, file_path=core.image_path(stem), stage=True
                        )
                        resp = await self._http.get(url, headers={"Range": "bytes=0-23"})
                        resp.raise_for_status()
                        return stem, core.parse_png_dims(resp.content[:24])
                    except Exception:
                        return stem, None

            dims_results = await asyncio.gather(*(_fetch_dims(s) for s in missing_dims))
            cache_grew = False
            for stem, dims in dims_results:
                if dims is not None:
                    dims_cache[stem] = dims
                    cache_grew = True
            if cache_grew:
                meta["image_dims"] = dims_cache
                core.write_metadata(meta, root=STATE_ROOT)
        for img in images:
            dims = dims_cache.get(img["stem"])
            if dims:
                img["width"], img["height"] = int(dims[0]), int(dims[1])

        embedding_entries = await self._list_files_safe(artifact_id, "embeddings")
        embedding_names = [
            entry.get("name", "") if isinstance(entry, dict) else str(entry)
            for entry in embedding_entries
        ]
        # Per-stem entries carry every stored model in ``model_types`` (0.9.0,
        # for the per-model embedding badges) plus the single backward-
        # compatible ``model_type`` older clients read.
        embeddings: Dict[str, Any] = core.collect_embeddings(embedding_names)

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
                if pairs:
                    my_annotations[label_name] = {
                        stem: {"latest_ts": pairs[stem]["timestamp"]} for stem in pairs
                    }

        return {
            "images": images,
            "embeddings": embeddings,
            "labels": meta.get("labels", []),
            "my_annotations": my_annotations,
            "role": role,
        }

    @bioengine.method(context=True)
    async def get_my_annotation_url(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: str = Field(..., description="Label name."),
        image_stem: str = Field(..., description="Image stem."),
        context=None,
    ) -> Dict[str, Any]:
        """Presigned GET URL for the caller's LATEST geojson annotation of
        one image under one label (the refine flow), minted on demand."""
        artifact_id = self._canonical_id(artifact_id)
        _meta, caller, _role = self._require_role(context, artifact_id, "annotator")
        dir_path = core.user_label_dir(label, caller["id"])
        entries = await self._list_files_safe(artifact_id, dir_path)
        filenames = [e.get("name", "") if isinstance(e, dict) else str(e) for e in entries]
        pairs = core.latest_pairs_by_stem(filenames)
        pair = pairs.get(image_stem)
        if not pair:
            return {"exists": False}
        url = await self._am.get_file(
            artifact_id=artifact_id, file_path=f"{dir_path}/{pair['geojson']}", stage=True
        )
        return {"exists": True, "latest_ts": pair["timestamp"], "geojson_read_url": url}

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
