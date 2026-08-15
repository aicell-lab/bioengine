"""Pure business logic for the annotation-broker BioEngine app.

Deliberately free of Ray, ``bioengine``, and ``hypha_rpc`` imports so it can
be unit-tested with plain ``pytest`` (no live Hypha connection, no Ray
runtime). ``broker.py`` is the thin transport layer that wires these
functions to actual Hypha artifact-manager calls.

Covers:
    * role resolution (owner / manager / annotator / public / none)
    * path building (label folders, the per-user sanitizer, timestamped
      annotation filenames, embedding/image paths)
    * timestamp generation/formatting
    * per-dataset metadata JSON: read / atomic write / new / delete
    * "latest pair per (user, stem)" lookup over a flat filename listing
    * the Hypha ACL permissions mirror built from broker metadata
    * a generic async "retry after re-staging" wrapper
    * a small time-based cache freshness check
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

# Ordered weakest -> strongest. "public" sits between "none" and "annotator":
# a public dataset grants read access to anyone, and (per the architecture
# plan) annotate access to anyone who is *logged in* — see resolve_role().
ROLE_ORDER: Dict[str, int] = {
    "none": 0,
    "public": 1,
    "annotator": 2,
    "manager": 3,
    "owner": 4,
}

ROLE_LIST_KEYS: Dict[str, str] = {
    "manager": "managers",
    "annotator": "annotators",
}

_ANONYMOUS_IDS = {"", "anonymous", "http-anonymous"}


def role_at_least(role: str, minimum: str) -> bool:
    """True if *role* meets or exceeds *minimum* in the role hierarchy."""
    return ROLE_ORDER.get(role, -1) >= ROLE_ORDER.get(minimum, 999)


def _user_matches(
    entry: Optional[Dict[str, Any]],
    user_id: Optional[str],
    user_email: Optional[str],
) -> bool:
    """Match a stored user record (``{"id": ..} `` and/or ``{"email": ..}``)
    against a caller's id/email, mirroring
    ``bioengine.utils.permissions.check_permissions``'s id-or-email match.
    """
    if not isinstance(entry, dict):
        return False
    entry_id = entry.get("id")
    entry_email = entry.get("email")
    if user_id and entry_id and str(entry_id) == str(user_id):
        return True
    if user_email and entry_email and str(entry_email).lower() == str(user_email).lower():
        return True
    return False


def _is_logged_in(user_id: Optional[str]) -> bool:
    return bool(user_id) and str(user_id).strip().lower() not in _ANONYMOUS_IDS


def resolve_role(
    metadata: Optional[Dict[str, Any]],
    user_id: Optional[str],
    user_email: Optional[str],
) -> str:
    """Resolve a caller's role for a dataset from its broker metadata.

    Precedence: owner > manager > annotator > public > none. A public
    dataset resolves to "annotator" for a logged-in caller (public datasets
    can be annotated by anyone with an identity, per the architecture plan)
    and to "public" (read-only) for an anonymous caller.
    """
    if not metadata:
        return "none"

    if _user_matches(metadata.get("owner"), user_id, user_email):
        return "owner"

    if any(
        _user_matches(m, user_id, user_email)
        for m in metadata.get("managers", []) or []
    ):
        return "manager"

    if any(
        _user_matches(a, user_id, user_email)
        for a in metadata.get("annotators", []) or []
    ):
        return "annotator"

    if metadata.get("public"):
        return "annotator" if _is_logged_in(user_id) else "public"

    return "none"


def set_user_role(metadata: Dict[str, Any], user: Dict[str, Any], role: str) -> Dict[str, Any]:
    """Add/move *user* into the ``managers``/``annotators`` list for *role*.

    A user has at most one non-owner role at a time, so they're removed from
    both lists first. Mutates and returns *metadata*.
    """
    if role not in ROLE_LIST_KEYS:
        raise ValueError(f"Invalid role {role!r}; expected one of {sorted(ROLE_LIST_KEYS)}")
    remove_user_role(metadata, user)
    metadata.setdefault(ROLE_LIST_KEYS[role], []).append(dict(user))
    return metadata


def remove_user_role(metadata: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """Remove *user* from both the ``managers`` and ``annotators`` lists.

    Mutates and returns *metadata*.
    """
    user_id = user.get("id")
    user_email = user.get("email")
    for key in ROLE_LIST_KEYS.values():
        metadata[key] = [
            u for u in metadata.get(key, []) or [] if not _user_matches(u, user_id, user_email)
        ]
    return metadata


# ---------------------------------------------------------------------------
# Access requests (annotators knocking on the door via the QR link)
# ---------------------------------------------------------------------------


def add_access_request(
    metadata: Dict[str, Any], user: Dict[str, Any], requested_role: str, requested_at: str
) -> Dict[str, Any]:
    """Record that *user* asked for *requested_role* on this dataset.

    Idempotent per user (matched by id or email): a repeat request just
    updates the requested role and timestamp. Mutates and returns *metadata*.
    Older metadata files without the ``access_requests`` key are upgraded in
    place.
    """
    if requested_role not in ROLE_LIST_KEYS:
        raise ValueError(
            f"Invalid requested_role {requested_role!r}; expected one of {sorted(ROLE_LIST_KEYS)}"
        )
    requests = metadata.setdefault("access_requests", [])
    remove_access_request(metadata, user)
    metadata["access_requests"].append(
        {
            "id": user.get("id"),
            "email": user.get("email"),
            "requested_role": requested_role,
            "requested_at": requested_at,
        }
    )
    return metadata


def remove_access_request(metadata: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """Drop *user*'s pending access request, if any. Mutates and returns
    *metadata*."""
    user_id = user.get("id")
    user_email = user.get("email")
    metadata["access_requests"] = [
        r
        for r in metadata.get("access_requests", []) or []
        if not _user_matches(r, user_id, user_email)
    ]
    return metadata


# ---------------------------------------------------------------------------
# Path building
# ---------------------------------------------------------------------------

LABEL_NAME_RE = re.compile(r"^[a-z0-9._-]+$")


def sanitize_user_id(user_id: Optional[str]) -> str:
    """Sanitize a caller id into a filesystem/URL-safe folder name.

    Ported verbatim from ``colab_service.py``'s ``AnnotationSession._user_folder``:
    strip to ``[A-Za-z0-9._-]``, replace everything else with ``-``, and
    prefix with ``user-``. Empty/missing input falls back to ``"anonymous"``.
    """
    raw = (user_id or "anonymous").strip()
    if not raw:
        raw = "anonymous"
    safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in raw)
    return f"user-{safe}"


def is_valid_label_name(name: str) -> bool:
    return bool(name) and bool(LABEL_NAME_RE.match(name))


def label_folder(label: str) -> str:
    """``label_<label>`` — note the underscore, not a colon (colons break
    URL/path layers downstream)."""
    return f"label_{label}"


def label_users_path(label: str) -> str:
    return f"{label_folder(label)}/users.json"


def label_metadata_path(label: str) -> str:
    return f"{label_folder(label)}/metadata.json"


def user_label_dir(label: str, user_id: Optional[str]) -> str:
    return f"{label_folder(label)}/{sanitize_user_id(user_id)}"


def image_path(stem: str, ext: str = ".png") -> str:
    return f"images/{stem}{ext}"


def embedding_path(stem: str, model_type: str) -> str:
    return f"embeddings/{stem}_{model_type}.npz"


def annotation_save_paths(
    label: str, user_id: Optional[str], stem: str, timestamp: str
) -> Dict[str, str]:
    """Return the ``{"png": ..., "geojson": ...}`` paths for one save,
    sharing a single server-generated timestamp across both files."""
    base = f"{user_label_dir(label, user_id)}/{stem}-{timestamp}"
    return {"png": f"{base}.png", "geojson": f"{base}.geojson"}


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
TIMESTAMP_RE = re.compile(r"^\d{8}-\d{6}$")


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format a UTC datetime as ``YYYYMMDD-HHMMSS`` (lexicographic sort ==
    chronological sort)."""
    dt = dt or datetime.now(timezone.utc)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime(TIMESTAMP_FORMAT)


def new_timestamp() -> str:
    return format_timestamp(datetime.now(timezone.utc))


def is_valid_timestamp(value: str) -> bool:
    return bool(TIMESTAMP_RE.match(value or ""))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Annotation filename parsing + "latest pair per (user, stem)"
# ---------------------------------------------------------------------------

# {stem}-{YYYYMMDD-HHMMSS}.{png|geojson}. The timestamp itself contains a
# hyphen, so we anchor on the fixed-width digit groups at the end rather than
# a naive rsplit("-", 1).
ANNOTATION_FILENAME_RE = re.compile(
    r"^(?P<stem>.+)-(?P<timestamp>\d{8}-\d{6})\.(?P<ext>png|geojson)$"
)


def parse_annotation_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parse ``{stem}-{timestamp}.{png,geojson}`` -> ``{stem, timestamp, ext}``,
    or ``None`` if *filename* doesn't match the pattern."""
    m = ANNOTATION_FILENAME_RE.match(filename)
    if not m:
        return None
    return {
        "stem": m.group("stem"),
        "timestamp": m.group("timestamp"),
        "ext": m.group("ext"),
    }


def latest_pairs_by_stem(filenames: List[str]) -> Dict[str, Dict[str, str]]:
    """Given a flat filename listing from one user's label folder, return
    ``{stem: {"timestamp": ts, "png": name, "geojson": name}}`` for every
    stem that has at least one timestamp with BOTH a png and a geojson —
    i.e. an "annotated" (user, stem) pair — keeping only the lexicographically
    (== chronologically) greatest such timestamp per stem.
    """
    by_stem_ts: Dict[str, Dict[str, Dict[str, str]]] = {}
    for filename in filenames:
        parsed = parse_annotation_filename(filename)
        if not parsed:
            continue
        stem, ts, ext = parsed["stem"], parsed["timestamp"], parsed["ext"]
        by_stem_ts.setdefault(stem, {}).setdefault(ts, {})[ext] = filename

    result: Dict[str, Dict[str, str]] = {}
    for stem, ts_map in by_stem_ts.items():
        complete = [ts for ts, files in ts_map.items() if "png" in files and "geojson" in files]
        if not complete:
            continue
        latest = max(complete)
        result[stem] = {
            "timestamp": latest,
            "png": ts_map[latest]["png"],
            "geojson": ts_map[latest]["geojson"],
        }
    return result


def is_annotated(filenames: List[str]) -> bool:
    """True if *filenames* (one user's label folder listing) contains at
    least one complete (png + geojson) pair for any stem."""
    return bool(latest_pairs_by_stem(filenames))


# ---------------------------------------------------------------------------
# Embedding filename parsing
# ---------------------------------------------------------------------------

# Longest-model-type-name-first so "vit_l_lm" isn't mis-split as "vit_l".
KNOWN_MODEL_TYPES = ("vit_b_lm", "vit_l_lm", "vit_t_lm", "vit_b", "vit_l", "vit_h")


def parse_embedding_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parse ``{stem}_{model_type}.npz`` -> ``{stem, model_type}`` by
    matching a known μSAM model-type suffix (model types themselves contain
    underscores, so a generic split is ambiguous)."""
    if not filename.endswith(".npz"):
        return None
    base = filename[: -len(".npz")]
    for model_type in sorted(KNOWN_MODEL_TYPES, key=len, reverse=True):
        suffix = f"_{model_type}"
        if base.endswith(suffix):
            stem = base[: -len(suffix)]
            if stem:
                return {"stem": stem, "model_type": model_type}
    return None


# ---------------------------------------------------------------------------
# Dataset metadata: paths, new/read/write/delete, atomic replace
# ---------------------------------------------------------------------------

DATASETS_SUBDIR = "datasets"


def default_state_root() -> Path:
    return Path.home() / "annotation_broker"


def datasets_dir(root: Optional[Path] = None) -> Path:
    base = root if root is not None else default_state_root()
    d = Path(base) / DATASETS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def artifact_id_to_filename(artifact_id: str) -> str:
    return artifact_id.replace("/", "__") + ".json"


def filename_to_artifact_id(filename: str) -> str:
    stem = filename[:-5] if filename.endswith(".json") else filename
    return stem.replace("__", "/")


def metadata_path(artifact_id: str, root: Optional[Path] = None) -> Path:
    return datasets_dir(root) / artifact_id_to_filename(artifact_id)


def new_metadata(artifact_id: str, owner: Dict[str, Any]) -> Dict[str, Any]:
    ts = now_iso()
    return {
        "artifact_id": artifact_id,
        "owner": dict(owner),
        "managers": [],
        "annotators": [],
        "access_requests": [],
        "public": False,
        "labels": [],
        "created_at": ts,
        "updated_at": ts,
    }


def read_metadata(artifact_id: str, root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    p = metadata_path(artifact_id, root)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def write_metadata(metadata: Dict[str, Any], root: Optional[Path] = None) -> Dict[str, Any]:
    """Atomically write *metadata* (write to a ``.tmp`` file, then
    ``os.replace``) — same pattern as ``training.write_status`` in the
    micro-sam app. Stamps ``updated_at`` and returns the written dict."""
    artifact_id = metadata["artifact_id"]
    metadata = dict(metadata)
    metadata["updated_at"] = now_iso()
    p = metadata_path(artifact_id, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metadata, indent=2))
    os.replace(tmp, p)
    return metadata


def delete_metadata(artifact_id: str, root: Optional[Path] = None) -> bool:
    p = metadata_path(artifact_id, root)
    if p.exists():
        p.unlink()
        return True
    return False


def list_dataset_ids(root: Optional[Path] = None) -> List[str]:
    """All artifact ids with a metadata file on disk, sorted."""
    d = datasets_dir(root)
    ids = [
        filename_to_artifact_id(p.name)
        for p in d.glob("*.json")
        if not p.name.endswith(".tmp")
    ]
    return sorted(ids)


# ---------------------------------------------------------------------------
# Label management (broker-metadata side; the artifact-manifest side is
# mirrored by broker.py)
# ---------------------------------------------------------------------------


def add_label(metadata: Dict[str, Any], name: str, description: str = "") -> Dict[str, Any]:
    if not is_valid_label_name(name):
        raise ValueError(f"Invalid label name {name!r}: must match ^[a-z0-9._-]+$")
    labels = metadata.setdefault("labels", [])
    if any(l.get("name") == name for l in labels):
        return metadata  # idempotent: already present
    labels.append({"name": name, "description": description})
    return metadata


# ---------------------------------------------------------------------------
# Per-label users.json (sanitized-id -> {id, email})
# ---------------------------------------------------------------------------


def upsert_label_user(
    users_map: Optional[Dict[str, Any]],
    sanitized_id: str,
    user_id: Optional[str],
    user_email: Optional[str],
) -> Dict[str, Any]:
    """Idempotent read-modify-write step for ``label_<label>/users.json``."""
    users_map = dict(users_map or {})
    users_map[sanitized_id] = {"id": user_id, "email": user_email}
    return users_map


# ---------------------------------------------------------------------------
# Hypha ACL permissions mirror
# ---------------------------------------------------------------------------


def _acl_key(user: Dict[str, Any]) -> Optional[str]:
    # Prefer id over email as the ACL key when both are present.
    return user.get("id") or user.get("email")


def build_acl_permissions(metadata: Dict[str, Any]) -> Dict[str, str]:
    """Derive the Hypha ``config.permissions`` dict from broker metadata:
    owner/managers -> ``"*"``, annotators -> ``"r+"``, plus ``"*": "r+"``
    when the dataset is public."""
    perms: Dict[str, str] = {}

    owner_key = _acl_key(metadata.get("owner") or {})
    if owner_key:
        perms[owner_key] = "*"

    for manager in metadata.get("managers", []) or []:
        key = _acl_key(manager)
        if key:
            perms[key] = "*"

    for annotator in metadata.get("annotators", []) or []:
        key = _acl_key(annotator)
        if key and key not in perms:
            perms[key] = "r+"

    if metadata.get("public"):
        perms["*"] = "r+"

    return perms


# ---------------------------------------------------------------------------
# Generic "retry after re-staging" wrapper
# ---------------------------------------------------------------------------


def is_stage_mode_error(exc: BaseException) -> bool:
    """True if *exc*'s message looks like Hypha's "artifact not in stage
    mode" error (matched loosely on the substring "stage", per the
    architecture plan, since the exact wording isn't pinned down)."""
    return "stage" in str(exc).lower()


async def ensure_staged(
    call: Callable[[], Awaitable[Any]],
    restage: Callable[[], Awaitable[Any]],
    max_attempts: int = 3,
    backoff_s: float = 1.0,
    sleep: Callable[[float], Awaitable[Any]] = None,
) -> Any:
    """Call ``call()``; if it raises a stage-mode error, call ``restage()``
    and retry, up to *max_attempts* attempts total, sleeping *backoff_s*
    between retries. Non-stage-mode errors propagate immediately.

    ``sleep`` is injectable (defaults to ``asyncio.sleep``) so tests don't
    have to wait out the real backoff.
    """
    if sleep is None:
        import asyncio

        sleep = asyncio.sleep

    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await call()
        except Exception as exc:  # noqa: BLE001 - re-raised below when not retryable
            last_exc = exc
            if not is_stage_mode_error(exc) or attempt == max_attempts:
                raise
            await restage()
            await sleep(backoff_s)
    raise last_exc  # pragma: no cover - unreachable, loop always returns or raises


# ---------------------------------------------------------------------------
# Small TTL cache freshness check (used for the 60s manifest-name cache)
# ---------------------------------------------------------------------------


def is_cache_fresh(cached_at: float, ttl_s: float = 60.0, now: Optional[float] = None) -> bool:
    now = time.time() if now is None else now
    return (now - cached_at) < ttl_s


# ---------------------------------------------------------------------------
# register_dataset ownership check
# ---------------------------------------------------------------------------


def caller_matches_artifact_owner(
    artifact_manifest: Optional[Dict[str, Any]],
    created_by: Optional[str],
    user_id: Optional[str],
    user_email: Optional[str],
) -> bool:
    """True if the caller matches the artifact's ``manifest.owner`` (id or
    email) or its ``created_by`` field."""
    manifest = artifact_manifest or {}
    if _user_matches(manifest.get("owner"), user_id, user_email):
        return True
    if created_by and user_id and str(created_by) == str(user_id):
        return True
    return False
