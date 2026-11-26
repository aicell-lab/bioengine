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

import hashlib
import json
import math
import os
import re
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

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
    if (
        user_email
        and entry_email
        and str(entry_email).lower() == str(user_email).lower()
    ):
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


def set_user_role(
    metadata: Dict[str, Any], user: Dict[str, Any], role: str
) -> Dict[str, Any]:
    """Add/move *user* into the ``managers``/``annotators`` list for *role*.

    A user has at most one non-owner role at a time, so they're removed from
    both lists first. Mutates and returns *metadata*.
    """
    if role not in ROLE_LIST_KEYS:
        raise ValueError(
            f"Invalid role {role!r}; expected one of {sorted(ROLE_LIST_KEYS)}"
        )
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
            u
            for u in metadata.get(key, []) or []
            if not _user_matches(u, user_id, user_email)
        ]
    return metadata


# ---------------------------------------------------------------------------
# Access requests (annotators knocking on the door via the QR link)
# ---------------------------------------------------------------------------


def add_access_request(
    metadata: Dict[str, Any],
    user: Dict[str, Any],
    requested_role: str,
    requested_at: str,
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


def remove_access_request(
    metadata: Dict[str, Any], user: Dict[str, Any]
) -> Dict[str, Any]:
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
    """LEGACY. Only correct for datasets whose images are all ``.png``, which
    stopped being guaranteed in 0.10.0. Every live call site now reads
    ``entry["path"]`` from the image manifest instead; this remains for the
    manifest-less fallback and for tests."""
    return f"images/{stem}{ext}"


# ---------------------------------------------------------------------------
# Image manifest (0.10.0)
# ---------------------------------------------------------------------------
#
# Until 0.10.0 the image list *was* the ``images/`` directory listing, in two
# independent places. That makes an image with no file under ``images/``
# impossible to represent, which is exactly what a remote (not-yet-
# materialised) image is. The manifest becomes the authoritative list and
# ``images/`` becomes the materialised subset.
#
# Two rules govern it, and they are deliberately asymmetric:
#
#   * READ is a union. Stored entries plus any file in the listing that no
#     entry covers. A file therefore never disappears from the dataset
#     because a sync call was missed or the manifest was never written --
#     which is also what makes the manifest-less legacy fallback the *same*
#     code path rather than a second one.
#   * WRITE (sync_image_manifest) is the union plus pruning of ``local``
#     entries whose file has gone. ``remote`` entries are never pruned: they
#     are not expected to have a file.

IMAGE_MANIFEST_PATH = "images/manifest.json"
IMAGE_MANIFEST_VERSION = 1

IMAGE_STATE_LOCAL = "local"
IMAGE_STATE_REMOTE = "remote"

# The manifest lives inside the directory it describes, so the listing must
# never be read as if it were an image.
_IMAGE_MANIFEST_NAME = IMAGE_MANIFEST_PATH.split("/")[-1]

# Derived from the manifest path rather than written out again, so the directory
# an image path is built from and the directory the manifest lives in cannot
# drift apart. A marker records a full path (`images/<name>`) and the listing
# yields bare names, so something has to join them; doing it from one constant
# is what keeps the two sides of that comparison in agreement.
IMAGE_DIR = IMAGE_MANIFEST_PATH.rsplit("/", 1)[0]


def local_image_entry(
    name: str, dims: Optional[List[int]] = None, **extra: Any
) -> Dict[str, Any]:
    """A ``local`` manifest entry for the file ``images/<name>``.

    *dims* is the ``[width, height]`` shape the 0.8.0 ``image_dims`` cache
    already uses, so a cache can be poured straight in.
    """
    entry: Dict[str, Any] = {"state": IMAGE_STATE_LOCAL, "path": f"images/{name}"}
    if dims and len(dims) == 2 and dims[0] > 0 and dims[1] > 0:
        entry["width"], entry["height"] = int(dims[0]), int(dims[1])
    entry.update(extra)
    return entry


MATERIALISED_IMAGE_EXT = ".tif"


def materialised_image_name(stem: str) -> str:
    """Filename a materialised *stem* gets under ``images/``."""
    return f"{stem}{MATERIALISED_IMAGE_EXT}"


def materialise_image_entry(
    entry: Dict[str, Any], name: str, levels: Optional[int] = None
) -> Dict[str, Any]:
    """*entry* rewritten as a ``local`` entry backed by ``images/<name>``.

    The ``source`` block is carried over unchanged. Provenance has to survive
    materialisation (design §2), and it is also what a later re-check against
    ``content_hint`` would need to say "this copy is stale" rather than just
    "this copy exists".

    *levels* is the pyramid the encoder actually wrote, which is a property of
    the file and not of the source, so a stale value from the remote entry is
    dropped when the caller does not supply one.
    """
    materialised = dict(entry)
    materialised["state"] = IMAGE_STATE_LOCAL
    materialised["path"] = f"images/{name}"
    if levels:
        materialised["levels"] = int(levels)
    else:
        materialised.pop("levels", None)
    return materialised


def parse_image_manifest(doc: Any) -> Dict[str, Dict[str, Any]]:
    """Entries out of a raw ``manifest.json`` document.

    Returns ``{}`` for anything unusable -- absent, malformed, or written by
    a newer broker than this one. Never raises: an unreadable manifest must
    degrade to the listing-only fallback, not take the dataset down.
    """
    if not isinstance(doc, dict):
        return {}
    if doc.get("manifest_version") != IMAGE_MANIFEST_VERSION:
        return {}
    images = doc.get("images")
    if not isinstance(images, dict):
        return {}
    return {
        stem: dict(entry)
        for stem, entry in images.items()
        if stem
        and isinstance(entry, dict)
        and entry.get("state")
        in (
            IMAGE_STATE_LOCAL,
            IMAGE_STATE_REMOTE,
        )
    }


def build_image_manifest(entries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The full document to write back to ``images/manifest.json``."""
    return {
        "manifest_version": IMAGE_MANIFEST_VERSION,
        "images": {stem: entries[stem] for stem in sorted(entries)},
    }


def image_names_from_listing(entries: List[Any]) -> List[str]:
    """Image filenames out of a ``list_files("images")`` result: no
    directories, and never the manifest itself.

    **The withholding is NOT applied here, and it used to be.** This function
    answers "what files are in ``images/``", which is a fact about the store;
    whether a file may be treated as a dataset image is a different question,
    and :func:`merge_image_manifest` now answers it in one place. A review
    measured what splitting it cost: the withheld set was passed at the two
    READ call sites and omitted at the two manifest WRITERS, so a manager
    running `sync_image_manifest` -- or importing a remote image -- during
    somebody's open upload recorded that upload as a permanent dataset image
    and committed it. Nothing in the signature made the omission visible,
    because an optional argument that changes correctness reads as a tuning
    knob.
    """
    names = []
    for entry in entries:
        name = entry.get("name", "") if isinstance(entry, dict) else str(entry)
        if not name or name == _IMAGE_MANIFEST_NAME:
            continue
        if isinstance(entry, dict) and (
            entry.get("type") == "directory" or entry.get("is_dir")
        ):
            continue
        names.append(name)
    return names


def merge_image_manifest(
    stored: Dict[str, Dict[str, Any]],
    listing_names: List[str],
    dims: Optional[Dict[str, List[int]]] = None,
    prune: bool = False,
    *,
    withheld_paths: Any,
) -> tuple:
    """Reconcile *stored* entries against the ``images/`` listing.

    Returns ``(entries, changed)``. See the module comment above for why the
    read (``prune=False``) and write (``prune=True``) rules differ.

    Stems are unique, so two files sharing one stem (``a.png`` and
    ``a.tif``) collapse to a single entry, the alphabetically last. Before
    0.10.0 they produced two identical-stem rows in the index, which no
    caller handled sensibly.

    **``withheld_paths`` is REQUIRED and has no default**, which is the point
    of it rather than an inconvenience. It holds artifact paths
    (``images/<name>``) belonging to uploads that have not been finished -- see
    :func:`withheld_upload_paths` -- and it governs BOTH halves of this
    reconciliation, in opposite directions:

    * a withheld name is never ADDED. A PUT object becomes a dataset image
      through this function, so an upload that was never finished would
      otherwise be an image with no size check ever applied to it;
    * a withheld path is treated as PRESENT when pruning, so a stored entry on
      that path is never DROPPED while an upload to it is open.

    Both are needed and neither implies the other, which is exactly how the
    defect arose: the rule used to live in :func:`image_names_from_listing`,
    where it could only express the first half, so the pruning writer could not
    use it -- and the two manifest writers therefore passed NO withheld set at
    all. A review measured the consequence: one `sync_image_manifest` during
    somebody's open upload wrote that upload into the stored manifest and
    committed it, making an unchecked object a permanent image, which the
    expiry sweep then deleted and left a dangling entry behind.

    Required rather than defaulted because an omission here is silent and
    changes correctness. A caller that has nothing withheld says so with
    ``frozenset()``; a caller that forgot gets a `TypeError`.
    """
    dims = dims or {}
    withheld = frozenset(withheld_paths or ())
    entries = {stem: dict(entry) for stem, entry in stored.items()}
    changed = False

    covered = {
        entry["path"].split("/", 1)[-1]
        for entry in entries.values()
        if entry.get("state") == IMAGE_STATE_LOCAL and entry.get("path")
    }
    for name in sorted(listing_names):
        if name in covered:
            continue
        if f"{IMAGE_DIR}/{name}" in withheld:
            # NEVER ADDED. The object is at `images/<name>` and the listing
            # reports it, but no finishing call has vouched for it, so nothing
            # has checked its size or its signature. Keyed by full path rather
            # than by name because that is what a marker records; re-deriving
            # it would be a second place for the two to disagree.
            continue
        stem = Path(name).stem
        if not stem:
            continue
        prior = entries.get(stem)
        if isinstance(prior, dict) and prior.get("state") == IMAGE_STATE_REMOTE:
            # A file appeared under a stem that is still recorded as remote:
            # a materialisation whose upload landed but whose manifest write
            # did not. Promote rather than replace, so the ``source`` block,
            # dtype and channel metadata survive the crash that produced this
            # state. (A minted-but-unuploaded put_file does not show up in the
            # listing, so a name here always means real bytes.)
            entries[stem] = materialise_image_entry(prior, name)
        else:
            entries[stem] = local_image_entry(name, dims.get(stem))
        changed = True

    # Fill dims for entries that predate a successful probe.
    for stem, entry in entries.items():
        if "width" in entry or stem not in dims:
            continue
        d = dims[stem]
        if d and len(d) == 2 and d[0] > 0 and d[1] > 0:
            entry["width"], entry["height"] = int(d[0]), int(d[1])
            changed = True

    if prune:
        present = set(listing_names)
        for stem in list(entries):
            entry = entries[stem]
            if entry.get("state") != IMAGE_STATE_LOCAL:
                continue
            if entry.get("path", "") in withheld:
                # TREATED AS PRESENT, the opposite direction from the add
                # above, and the reason this set is passed here rather than
                # filtered out of `listing_names` before the call. A filtered
                # list would make a withheld path look ABSENT to this loop and
                # drop a stored entry while an upload to it is open -- turning
                # the fix for the writer into a different data loss.
                continue
            if entry.get("path", "").split("/", 1)[-1] not in present:
                del entries[stem]
                changed = True

    return entries, changed


# ---------------------------------------------------------------------------
# Remote NGFF (OME-Zarr) sources (0.10.0)
# ---------------------------------------------------------------------------
#
# An import records where the pixels live and nothing else: no copy, no pixel
# read beyond one preflight chunk. Everything below is pure so the parsing and
# the accept/reject rules can be tested without a network.
#
# The frontend mirrors part of this in ``ngffLayout.ts``. Where both parse the
# same thing (``multiscales``, axis roles, level dimensions) they must agree,
# because the broker's recorded width/height is what the UI shows before a
# single chunk is fetched. The tests on either side pin the same cases.

NGFF_SOURCE_KIND = "ngff"

# What zarrita 0.6.1 can actually decode in a browser: its built-in codecs
# plus the three numcodecs WASM builds it lazy-loads. Anything else (a
# jpeg2k-compressed store, an ``imagecodecs``-only filter, a delta filter)
# would import cleanly and then fail on every chunk, so it is rejected at
# import instead. Keep this in step with zarrita's `create_default_registry`.
NGFF_DECODABLE_CODECS = frozenset(
    {"blosc", "lz4", "zstd", "gzip", "zlib", "transpose", "bytes", "crc32c"}
)

# zarrita maps these v2 dtype strings to JS typed arrays. i8/u8 (64-bit) are
# deliberately absent: they decode to BigInt64Array, which the RGBA conversion
# cannot scale. Same for the string dtypes.
NGFF_NUMERIC_DTYPES = frozenset(
    {"b1", "i1", "u1", "i2", "u2", "i4", "u4", "f2", "f4", "f8"}
)

# v3 spells a dtype out where v2 abbreviates it. Everything downstream works
# in the v2 vocabulary, so this is the one translation table.
NGFF_V3_TO_V2_DTYPE = {
    "bool": "b1",
    "int8": "i1",
    "uint8": "u1",
    "int16": "i2",
    "uint16": "u2",
    "int32": "i4",
    "uint32": "u4",
    "int64": "i8",
    "uint64": "u8",
    "float16": "f2",
    "float32": "f4",
    "float64": "f8",
}

# Axis order implied by OME-NGFF before v0.4, which made `axes` optional.
NGFF_IMPLICIT_AXES = ["t", "c", "z", "y", "x"]


class NgffSourceError(ValueError):
    """A remote source that cannot be used, with a message aimed at the person
    who pasted the URL. Distinct from ValueError only so the transport layer
    can tell "your URL is wrong" from "your request is wrong"."""


def normalise_store_root(url: Any) -> str:
    """Validate a pasted store root and return it with exactly one trailing
    slash.

    A Zarr client builds every chunk URL by appending the key to the root, so
    a root carrying a query string would put ``?token=...`` *before* the key
    and every chunk would 404 (design §7). That is a silent, per-chunk failure
    rather than an obvious one, so it is refused here with the reason.
    """
    if not isinstance(url, str) or not url.strip():
        raise NgffSourceError("Store root is required.")
    root = url.strip()
    if not root.lower().startswith(("http://", "https://")):
        raise NgffSourceError(f"Store root must be an http(s) URL, got {root!r}.")
    if "?" in root or "#" in root:
        raise NgffSourceError(
            "Store root must not contain a query string or fragment: a Zarr "
            "client appends the chunk key to the root, so anything after '?' "
            "would land in the middle of every chunk URL. Use a path-form "
            "credential instead."
        )
    return root.rstrip("/") + "/"


# omezarr-view's path-form credential: /t/<token>/zarr/<id>/
_TOKEN_SEGMENT = re.compile(r"/t/([^/]+)/")


def split_store_credential(root: str) -> tuple:
    """``(untokened_root, credential)`` for a store root.

    The manifest lives in an artifact that can be made public
    (``set_public``), so a credential must never be written into it (design
    §12, decided 2026-09-10). This splits the path-form token out; the broker
    keeps it in its own PVC-backed metadata and splices it back in at
    ``get_image_url`` time, after the role check.

    Returns ``(root, None)`` unchanged for a store root with no credential,
    which is every plain S3/IDR-style bucket.
    """
    match = _TOKEN_SEGMENT.search(root)
    if not match:
        return root, None
    untokened = root[: match.start()] + "/" + root[match.end() :]
    return untokened, match.group(1)


def apply_store_credential(root: str, credential: Optional[str]) -> str:
    """Inverse of :func:`split_store_credential`.

    Inserts ``/t/<credential>/`` ahead of the last path segment group, which
    is where omezarr-view expects it. A falsy credential returns *root*
    unchanged, so the local/public case costs nothing.
    """
    if not credential:
        return root
    scheme, _, rest = root.partition("://")
    host, slash, path = rest.partition("/")
    if not slash:
        return f"{scheme}://{host}/t/{credential}/"
    return f"{scheme}://{host}/t/{credential}/{path}"


def parse_ngff_multiscales(
    zattrs: Any, multiscale_path: Optional[str] = None
) -> Dict[str, Any]:
    """``{"name", "dataset_paths", "axes"}`` from a group's attributes.

    Mirrors ``parseMultiscales`` in ``ngffLayout.ts``. *multiscale_path*
    selects one multiscale by name when a group holds several; the first is
    used otherwise, which is what every OME-Zarr writer produces.
    """
    attrs = zattrs if isinstance(zattrs, dict) else {}
    # A v3 store nests OME metadata one level down under "ome".
    if isinstance(attrs.get("ome"), dict) and "multiscales" in attrs["ome"]:
        attrs = attrs["ome"]
    multiscales = attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise NgffSourceError(
            "Not an OME-Zarr group: no 'multiscales' in its attributes. Point "
            "at the group that holds the pyramid, not at the array or the "
            "bucket root."
        )

    chosen = None
    if multiscale_path:
        for entry in multiscales:
            if isinstance(entry, dict) and str(entry.get("name")) == multiscale_path:
                chosen = entry
                break
        if chosen is None:
            names = [str(e.get("name")) for e in multiscales if isinstance(e, dict)]
            raise NgffSourceError(
                f"No multiscale named {multiscale_path!r}; this store has: "
                + (", ".join(n for n in names if n and n != "None") or "(unnamed)")
            )
    else:
        chosen = next((e for e in multiscales if isinstance(e, dict)), None)
    if chosen is None:
        raise NgffSourceError("OME-Zarr 'multiscales' holds no usable entry.")

    dataset_paths = []
    for dataset in chosen.get("datasets") or []:
        if isinstance(dataset, dict) and dataset.get("path"):
            dataset_paths.append(str(dataset["path"]))
        elif isinstance(dataset, str) and dataset:
            dataset_paths.append(dataset)
    if not dataset_paths:
        raise NgffSourceError(
            "OME-Zarr multiscale has no datasets, so there are no pyramid "
            "levels to read."
        )

    raw_axes = chosen.get("axes")
    if isinstance(raw_axes, list) and raw_axes:
        axes = [
            str(a.get("name", "")).lower() if isinstance(a, dict) else str(a).lower()
            for a in raw_axes
        ]
    else:
        # Pre-v0.4: `axes` was optional and the order was fixed.
        axes = list(NGFF_IMPLICIT_AXES)

    return {
        "name": chosen.get("name"),
        "dataset_paths": dataset_paths,
        "axes": axes,
    }


def ngff_axis_roles(axes: List[str], rank: int) -> Dict[str, Any]:
    """``{"y", "x", "c"}`` axis indices for an array of *rank* dimensions.

    Mirrors ``axisRoles`` in ``ngffLayout.ts``, including the fallback: when
    the declared axes cannot be trusted (wrong length, no y/x) Y and X are
    taken to be the two fastest-varying dimensions, which is true of every
    image layout in practice.
    """
    y = axes.index("y") if "y" in axes else -1
    x = axes.index("x") if "x" in axes else -1
    c = axes.index("c") if "c" in axes else -1
    if not (len(axes) == rank and y >= 0 and x >= 0):
        if rank < 2:
            raise NgffSourceError(
                f"OME-Zarr array has rank {rank}; need at least 2 dimensions."
            )
        y, x, c = rank - 2, rank - 1, -1
    return {"y": y, "x": x, "c": c if c >= 0 else None}


def ngff_level_dims(shape: List[int], roles: Dict[str, Any]) -> Dict[str, int]:
    """``{"width", "height"}`` for one pyramid level."""
    return {"width": int(shape[roles["x"]]), "height": int(shape[roles["y"]])}


def parse_zarr_array_meta(doc: Any) -> Dict[str, Any]:
    """Normalise a ``.zarray`` (v2) or ``zarr.json`` (v3) into
    ``{"shape", "chunks", "dtype", "codecs", "separator", "zarr_format"}``.

    The codec names come out in zarrita's vocabulary for both versions, so
    :func:`undecodable_ngff_codecs` has one thing to check rather than two.
    """
    if not isinstance(doc, dict):
        raise NgffSourceError("Array metadata is missing or not JSON.")

    if doc.get("zarr_format") == 3:
        grid = (doc.get("chunk_grid") or {}).get("configuration") or {}
        encoding = doc.get("chunk_key_encoding") or {}
        enc_config = encoding.get("configuration") or {}
        key_encoding = str(encoding.get("name") or "default")
        codecs = [
            str(c.get("name"))
            for c in doc.get("codecs") or []
            if isinstance(c, dict) and c.get("name")
        ]
        return {
            "zarr_format": 3,
            "shape": [int(s) for s in doc.get("shape") or []],
            "chunks": [int(s) for s in grid.get("chunk_shape") or []],
            "dtype": str(doc.get("data_type") or ""),
            "codecs": codecs,
            # The default separator differs per encoding: "/" for the v3
            # default, "." for a v3 store written with v2-style keys.
            "separator": str(
                enc_config.get("separator") or ("." if key_encoding == "v2" else "/")
            ),
            "key_encoding": key_encoding,
        }

    shape = [int(s) for s in doc.get("shape") or []]
    chunks = [int(s) for s in doc.get("chunks") or []]
    if not shape or not chunks:
        raise NgffSourceError("Array metadata has no shape or chunk shape.")
    codecs = []
    for f in doc.get("filters") or []:
        if isinstance(f, dict) and f.get("id"):
            codecs.append(str(f["id"]))
    compressor = doc.get("compressor")
    if isinstance(compressor, dict) and compressor.get("id"):
        codecs.append(str(compressor["id"]))
    return {
        "zarr_format": 2,
        "shape": shape,
        "chunks": chunks,
        "dtype": str(doc.get("dtype") or ""),
        "codecs": codecs,
        "separator": str(doc.get("dimension_separator") or "."),
        "key_encoding": "v2",
    }


def undecodable_ngff_codecs(meta: Dict[str, Any]) -> List[str]:
    """Codec names in *meta* that the browser reader cannot handle."""
    return sorted(
        {c for c in meta.get("codecs") or [] if c not in NGFF_DECODABLE_CODECS}
    )


def normalise_ngff_dtype(dtype: Any) -> str:
    """A Zarr dtype string in the v2 vocabulary (``u2``), byte order dropped.

    Accepts either spelling: v2 dtypes carry a byte-order prefix (``<u2``),
    v3 spells the type out (``uint16``). Returns ``""`` for anything empty
    or non-string, so callers can treat "unknown" as one case.
    """
    if not dtype or not isinstance(dtype, str):
        return ""
    return NGFF_V3_TO_V2_DTYPE.get(dtype, dtype.lstrip("<>|="))


def ngff_dtype_is_readable(dtype: str) -> bool:
    """True if the frontend can turn this dtype into pixels.

    Both spellings are accepted, and both reject 64-bit integers, which
    decode to BigInt arrays the RGBA conversion cannot scale.
    """
    return normalise_ngff_dtype(dtype) in NGFF_NUMERIC_DTYPES


def first_chunk_key(meta: Dict[str, Any]) -> str:
    """Key of chunk (0, 0, ...) for the array described by *meta*.

    This is what the preflight fetches: the coarsest level's first chunk is
    the cheapest object in the store that proves the whole read path works
    (reachable, CORS-open, decodable).
    """
    rank = max(1, len(meta.get("chunks") or meta.get("shape") or [1]))
    zeros = ["0"] * rank
    if meta.get("key_encoding") == "v2" or meta.get("zarr_format") == 2:
        return meta.get("separator", ".").join(zeros)
    return "c" + meta.get("separator", "/") + meta.get("separator", "/").join(zeros)


def ngff_channel_names(zattrs: Any) -> Optional[List[str]]:
    """Channel labels from an OMERO rendering block, when the store has one.

    Purely cosmetic: the manifest records them so the UI can name channels
    without a second fetch. Absent for most stores, and that is fine.
    """
    attrs = zattrs if isinstance(zattrs, dict) else {}
    if isinstance(attrs.get("ome"), dict):
        attrs = {**attrs, **attrs["ome"]}
    channels = (
        (attrs.get("omero") or {}).get("channels")
        if isinstance(attrs.get("omero"), dict)
        else None
    )
    if not isinstance(channels, list) or not channels:
        return None
    names = [
        str(c.get("label") or c.get("name") or "").strip()
        for c in channels
        if isinstance(c, dict)
    ]
    names = [n for n in names if n]
    return names or None


_IMAGE_STEM_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def is_valid_image_stem(stem: Any) -> bool:
    """A stem safe to use as a filename and as a manifest key.

    Deliberately narrow. The stem becomes ``images/<stem>.tif`` at
    materialisation and a path component in every annotation filename, so
    slashes, spaces and dot-dot have to be impossible rather than merely
    unlikely.
    """
    if not isinstance(stem, str) or not _IMAGE_STEM_RE.match(stem):
        return False
    return stem not in (".", "..")


def default_stem_from_store_root(root: str) -> str:
    """A stem guessed from the last meaningful segment of a store root.

    ``…/zarr/my-sample.zarr/`` gives ``my-sample``. Returns ``""`` when
    nothing usable is left, in which case the caller must ask for a name
    rather than invent one.
    """
    path = root.split("://", 1)[-1]
    segments = [s for s in path.split("/") if s]
    # Drop the host, and any purely numeric level directory a user may have
    # pasted along with the group root.
    for segment in reversed(segments[1:]):
        candidate = segment[:-5] if segment.lower().endswith(".zarr") else segment
        candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip("-")
        if candidate and not candidate.isdigit() and is_valid_image_stem(candidate):
            return candidate
    return ""


def normalise_planes(
    planes: Any,
    axes: Optional[List[str]],
    level0_shape: Optional[List[int]],
    axis_roles: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, int]]:
    """A validated ``{axis_name: index}``, or ``None`` when none was given.

    Recorded at import because the annotator knows which slice the region was
    drawn on and nothing downstream can recover it. A reader refuses an axis of
    size greater than one with no plane recorded -- deliberately, since picking
    index 0 pairs the polygons with a different slice's pixels, leaves the mask
    non-empty and the run reporting success. See ``ngff_region._plane_index``.

    **Checked HERE as well as at read time, and that is not redundant.** The
    reader's refusal protects the pixels; this one protects the RECORD. A plane
    naming an axis the store does not have, or an index past its end, would
    otherwise be written into the manifest and fail only when someone trained
    on it -- long after the annotation was saved, and reported as a training
    failure rather than as the bad import it was.

    ``None`` rather than ``{}`` for an empty mapping, so an absent plane and an
    empty one cannot mean two things to a consumer.

    **Only NON-SPATIAL axes may carry a plane, and that is enforced here rather
    than left to the reader.** Measured: the reader keeps y and x whole (they are
    the crop) and keeps the CHANNEL axis whole on purpose, so a recorded
    ``{"c": 2}`` is silently ignored -- ``region_index`` with
    ``planes={"z": 4, "c": 2, "y": 5}`` returns
    ``(0, slice(None), 4, slice(0, 10), slice(0, 10))``. The record would then
    say "channel 2" while training reads all three channels, with nothing
    reporting the difference. That is precisely the "record that says something
    training will not do" this validator exists to keep out of the manifest, so
    a plane on y, x or c is REFUSED.

    This matters concretely because the annotator sends the viewer's current
    selection, and that selection can include a channel. Refusing it tells the
    frontend to drop c from what it sends; accepting it would have the record
    and the pixels disagree forever.
    """
    # Only `None` and an EMPTY MAPPING mean "no plane". `0`, `""`, `False` and
    # `[]` are all falsy and used to be swallowed as "none given", so a caller
    # sending one got a silent no-op instead of being told its argument was not
    # a mapping -- the quiet shape of wrongness this validator exists to refuse.
    if planes is None:
        return None
    if not isinstance(planes, dict):
        raise ValueError(
            f"planes must be an object of axis name to index, not "
            f"{type(planes).__name__}."
        )
    if not planes:
        return None
    names = [str(a) for a in (axes or [])]
    shape = [int(n) for n in (level0_shape or [])]
    if not names or not shape:
        raise ValueError(
            "planes was given but this source records no axis names or shape, so "
            "an index cannot be checked against an axis; re-import the source."
        )
    if len(names) != len(shape):
        raise ValueError(
            f"this source records {len(names)} axis name(s) {names!r} and a shape "
            f"of {len(shape)} dimension(s) {shape!r}; planes cannot be resolved "
            f"against a layout that does not describe itself."
        )
    # Axis roles map a role to an axis INDEX, so the forbidden NAMES are the
    # names sitting at those indices. Absent roles forbid nothing: a source that
    # does not say which axis is y cannot have a plane refused for being y.
    forbidden: Dict[str, str] = {}
    for role, idx in (axis_roles or {}).items():
        if isinstance(idx, bool) or not isinstance(idx, int):
            continue
        if 0 <= idx < len(names):
            forbidden[names[idx]] = str(role)

    out: Dict[str, int] = {}
    for key, value in planes.items():
        name = str(key)
        if name in forbidden:
            role = forbidden[name]
            raise ValueError(
                f"planes names axis {name!r}, which is this source's "
                f"{role!r} axis. A plane selects one slice of a NON-SPATIAL "
                f"axis such as z or t; y and x are the crop, and the channel "
                f"axis is read whole, so a plane on {name!r} would be recorded "
                f"and then ignored -- the record would say one thing and "
                f"training would do another."
            )
        if name not in names:
            raise ValueError(
                f"planes names axis {name!r}, which this source does not have. "
                f"Its axes are {names!r}."
            )
        # STRICT on purpose: `int()` would turn 1.5 into slice 1 and True into
        # slice 1, silently, which is the wrong-slice bug this field exists to
        # prevent wearing a different hat. A plane must arrive as a whole
        # number, and a caller sending anything else is told so rather than
        # rounded into a plausible-looking answer.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"planes[{name!r}] is {value!r}, which is not a number. An index "
                f"must be a whole number, not a string or a flag."
            )
        if not math.isfinite(float(value)):
            raise ValueError(
                f"planes[{name!r}] is {value!r}, which is not a finite number. "
                f"`inf` raises OverflowError and `nan` a message naming neither "
                f"the axis nor the problem, so both are refused here instead."
            )
        if float(value) != int(value):
            raise ValueError(
                f"planes[{name!r}] is {value!r}, which is not a whole number. "
                f"There is no slice between two slices, and rounding one into "
                f"the other would pick the wrong pixels silently."
            )
        index = int(value)
        size = shape[names.index(name)]
        if not 0 <= index < size:
            raise ValueError(
                f"planes[{name!r}]={index} is outside axis {name!r}, which has "
                f"{size} element(s) (0..{size - 1})."
            )
        out[name] = index

    # N4: a PARTIAL mapping used to be accepted -- {"t": 0} on a store whose z is
    # 9 was recorded, and the reader then refused it at TRAINING time, which is
    # exactly the "fails only when somebody trained on it" this validator claims
    # to prevent. So once a caller names any plane, it must name them all: every
    # axis that is neither spatial nor channel and has more than one element.
    # An empty mapping is still fine -- that is "no plane recorded", which the
    # reader refuses immediately and legibly.
    # Only when the ROLES are known. Without them nothing is forbidden, so y, x
    # and c would count as non-spatial and the rule would demand a plane for the
    # crop axes themselves -- it broke six existing cases that way. The check is
    # only as good as the layout, which is the same reason absent roles forbid
    # nothing above.
    if out and forbidden:
        missing = [
            n
            for i, n in enumerate(names)
            if n not in forbidden and n not in out and shape[i] > 1
        ]
        if missing:
            raise ValueError(
                f"planes names {sorted(out)} but leaves {missing!r} unset, and "
                f"{'those axes have' if len(missing) > 1 else 'that axis has'} "
                f"more than one element. A partial record is refused here rather "
                f"than at training time: give every non-spatial axis a plane, or "
                f"none at all."
            )
    return out or None


def normalise_region(
    region: Any, store_width: int, store_height: int
) -> Optional[Dict[str, int]]:
    """A validated ``{x, y, width, height}`` patch, or ``None`` for the whole
    store (design §2, §13 step 6).

    ``None`` is returned for a region that covers the store entirely, not just
    for an absent one. A whole-store patch and a whole-store import have to
    produce byte-identical manifest entries, or every consumer grows a second
    case that means the same thing as the first.

    The far edge is **clamped** to the store's bounds while the origin is
    **rejected** outside them. The asymmetry is deliberate: a picker drawing on
    a scaled-down preview routinely rounds the far edge a pixel past the end,
    and refusing that would make the last patch of a slide unselectable, but an
    origin outside the store is a caller that has the wrong coordinate system
    and must not be silently repaired into a valid-looking patch somewhere else.
    """
    if region is None:
        return None
    if not isinstance(region, dict):
        raise ValueError("A region must be an object with x, y, width and height.")
    try:
        x, y = int(region["x"]), int(region["y"])
        width, height = int(region["width"]), int(region["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "A region needs integer x, y, width and height, in pixels of the "
            "store's full-resolution level."
        ) from exc

    if width <= 0 or height <= 0:
        raise ValueError(f"Region is empty: {width}x{height} pixels.")
    if x < 0 or y < 0:
        raise ValueError(f"Region starts outside the image at ({x}, {y}).")
    if (store_width > 0 and x >= store_width) or (
        store_height > 0 and y >= store_height
    ):
        raise ValueError(
            f"Region starts at ({x}, {y}), outside the "
            f"{store_width}x{store_height} image."
        )

    if store_width > 0:
        width = min(width, store_width - x)
    if store_height > 0:
        height = min(height, store_height - y)
    if x == 0 and y == 0 and width >= store_width and height >= store_height:
        return None
    return {"x": x, "y": y, "width": width, "height": height}


def remote_image_entry(
    store_root: str,
    multiscale_path: Optional[str],
    width: int,
    height: int,
    dtype: str = "",
    channels: Optional[List[str]] = None,
    levels: Optional[int] = None,
    content_hint: Optional[str] = None,
    fetched_at: Optional[str] = None,
    region: Optional[Dict[str, int]] = None,
    layout: Optional[Dict[str, Any]] = None,
    planes: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """A ``remote`` manifest entry.

    No ``path``: there is no object in the artifact yet, and
    ``_image_file_path`` raising on that is what keeps every "give me the
    file" caller honest until materialisation (design §8) lands.

    *store_root* must already be untokened -- see
    :func:`split_store_credential`.
    """
    entry: Dict[str, Any] = {
        "state": IMAGE_STATE_REMOTE,
        "width": int(width),
        "height": int(height),
        "source": {
            "kind": NGFF_SOURCE_KIND,
            "store_root": store_root,
            "multiscale_path": multiscale_path,
            "region": region,
            "fetched_at": fetched_at or now_iso(),
            "content_hint": content_hint,
        },
    }
    if planes:
        # Which slice of each non-spatial axis the annotation belongs to, by
        # axis NAME. A z-stack has no single answer and a pixel reader must not
        # invent one: see ngff_region.region_index, which refuses an axis of
        # size > 1 with nothing recorded here.
        entry["source"]["planes"] = {str(k): int(v) for k, v in planes.items()}
    if layout:
        # The store's shape, recorded at import because the probe is already in
        # hand here. A reader that wants pixels needs the level-0 dataset path
        # and which axis is y and which is x; without this it would have to
        # re-probe the store, or parse `multiscales` itself, and that parsing is
        # already duplicated between this module and ngffLayout.ts.
        #
        # Captured ONCE, at the moment the source was proved readable, rather
        # than re-derived per training run. ``content_hint`` beside it is what
        # tells a later reader whether the store's metadata has changed since.
        entry["source"]["layout"] = {
            "dataset_paths": [str(p) for p in (layout.get("dataset_paths") or [])],
            "axis_roles": dict(layout.get("axis_roles") or {}),
            "level0_shape": [int(n) for n in (layout.get("level0_shape") or [])],
            # The axis NAMES, not just the y/x/c roles. Without them a reader
            # can see that an axis is neither y, x nor c but cannot tell
            # whether it is z, t or something else, so it can neither name the
            # axis in a refusal nor accept a plane for it.
            "axes": [str(a) for a in (layout.get("axes") or [])],
        }
    if dtype:
        entry["dtype"] = dtype
    if channels:
        entry["channels"] = list(channels)
    if levels:
        entry["levels"] = int(levels)
    return entry


def image_entry_is_local(entry: Optional[Dict[str, Any]]) -> bool:
    """True only for a materialised image with a file behind it.

    ``state`` and ``path`` are checked together on purpose: a ``local`` entry
    with no path is a bug, and treating it as local would push the failure
    down to whoever tried to presign it.
    """
    if not isinstance(entry, dict):
        return False
    return entry.get("state") == IMAGE_STATE_LOCAL and bool(entry.get("path"))


def non_local_stems(entries: Dict[str, Dict[str, Any]], stems: List[str]) -> List[str]:
    """Sorted stems in *stems* that are known but not materialised."""
    return sorted(
        {s for s in stems if s in entries and not image_entry_is_local(entries.get(s))}
    )


def ngff_region_source(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A reference a pixel reader can use for a remote NGFF stem, or ``None``.

    This is the ONE place the manifest's stored source becomes the shape a
    trainer consumes, so the two cannot drift into two descriptions of a crop.

    ``None`` whenever the entry cannot be read without materialising: a local
    entry (it has a file, so the file is the answer), a non-NGFF source, or an
    NGFF source imported before the layout was recorded. That last case is why
    this returns None rather than a partial reference — a reader handed a
    reference with no ``dataset_paths`` cannot know which level is full
    resolution, and guessing ``"0"`` is wrong for the stores that name it
    ``"s0"``.
    """
    if not isinstance(entry, dict) or image_entry_is_local(entry):
        return None
    source = entry.get("source")
    if not isinstance(source, dict) or source.get("kind") != NGFF_SOURCE_KIND:
        return None
    layout = source.get("layout")
    if not isinstance(layout, dict):
        return None
    if not layout.get("dataset_paths") or not layout.get("level0_shape"):
        return None
    roles = layout.get("axis_roles")
    if not isinstance(roles, dict) or not isinstance(roles.get("y"), int):
        return None
    if not isinstance(roles.get("x"), int):
        return None
    return {
        "kind": NGFF_SOURCE_KIND,
        "store_root": source.get("store_root"),
        "multiscale_path": source.get("multiscale_path"),
        "region": source.get("region"),
        "dataset_paths": list(layout["dataset_paths"]),
        "axis_roles": dict(roles),
        "level0_shape": list(layout["level0_shape"]),
        # Absent for a stem imported before axis names were recorded. Passed
        # through as [] rather than omitted so the reader's refusal can say the
        # names are missing, which is the actionable form of the problem.
        "axes": [str(a) for a in (layout.get("axes") or [])],
        "planes": source.get("planes") or None,
    }


# ---------------------------------------------------------------------------
# Materialisation (0.11.0, design §8)
# ---------------------------------------------------------------------------
#
# A remote entry becomes local by writing one OME-TIFF into the artifact and
# flipping ``state``. The transition is one-way and the flip is the only
# mutation, which is what makes a half-finished materialisation harmless: the
# object sits at a deterministic path and its bytes are a pure function of the
# source, so the next attempt simply overwrites it.

# Bytes per sample, keyed by the normalised v2 dtype.
_DTYPE_ITEMSIZE = {
    "b1": 1,
    "i1": 1,
    "u1": 1,
    "i2": 2,
    "u2": 2,
    "f2": 2,
    "i4": 4,
    "u4": 4,
    "f4": 4,
    "i8": 8,
    "u8": 8,
    "f8": 8,
}

# What one level-0 read is allowed to cost the annotator's browser, in bytes
# of decoded samples.
#
# Materialisation reads level 0 whole, in the browser, and copies it into the
# Pyodide kernel to be encoded; a gigapixel slide would exhaust the tab long
# before it produced a file. 64 MiB covers 4096x4096 uint16 or 5792x5792
# uint8, far above every dataset in use (median 0.26 MP, max 1.02 MP) and
# still leaving room for the copy into Python and the encoder's own pyramid.
#
# Refusing here is not a failure mode. The image stays remote and stays
# annotatable; only training is blocked (design §8, §9). A whole-slide source
# is meant to reach this path as a region (§13 step 6), not whole.
MAX_MATERIALISE_BYTES = 64 * 1024 * 1024

# Widest image the OME-TIFF profile is verified for. ``_photometric_for`` in
# colab_service.py picks ``rgb`` at 3 and 4 samples and ``minisblack`` below,
# and was checked against both readers we depend on at each of those counts.
# Nothing above 4 has a tested spelling.
MAX_MATERIALISE_CHANNELS = 4


def dtype_itemsize(dtype: Any) -> int:
    """Bytes per sample for a Zarr dtype in either spelling, ``0`` if unknown."""
    return _DTYPE_ITEMSIZE.get(normalise_ngff_dtype(dtype), 0)


def materialisation_bytes(entry: Optional[Dict[str, Any]]) -> int:
    """Decoded size of *entry*'s level 0, in bytes. ``0`` when unknown.

    A **lower bound**, deliberately. A store with no omero metadata reports no
    channel names, and this counts one channel rather than guessing; an
    unrecognised dtype yields 0 rather than an invented itemsize. It exists to
    refuse the obviously impossible before the browser tries, not to predict
    the allocation exactly, so a zero has to read as "no opinion" and let the
    attempt proceed.
    """
    if not isinstance(entry, dict):
        return 0
    width, height = entry.get("width") or 0, entry.get("height") or 0
    itemsize = dtype_itemsize(entry.get("dtype"))
    if width <= 0 or height <= 0 or itemsize <= 0:
        return 0
    channels = entry.get("channels")
    samples = len(channels) if isinstance(channels, list) and channels else 1
    return int(width) * int(height) * itemsize * samples


def check_materialisable(
    stem: str,
    entry: Optional[Dict[str, Any]],
    max_bytes: int = MAX_MATERIALISE_BYTES,
) -> None:
    """Raise unless *entry* is a remote image small enough to materialise.

    Says nothing about a ``local`` entry: "already done" is the caller's to
    report, not an error.
    """
    if not isinstance(entry, dict):
        raise FileNotFoundError(f"'{stem}' is not an image of this dataset.")
    if entry.get("state") != IMAGE_STATE_REMOTE:
        raise ValueError(
            f"'{stem}' is not a remote image, so there is nothing to " f"materialise."
        )
    channels = entry.get("channels")
    count = len(channels) if isinstance(channels, list) else 0
    if count > MAX_MATERIALISE_CHANNELS:
        # The OME-TIFF profile's photometric choice is verified for 1 to 4
        # samples and nothing wider (design §4). Writing the first few
        # channels of a 40-channel source instead would produce a file that
        # quietly disagrees with the store it claims to copy, and the
        # disagreement would only surface once a model was trained on it.
        raise ValueError(
            f"'{stem}' has {count} channels, over the "
            f"{MAX_MATERIALISE_CHANNELS} the OME-TIFF profile covers. It "
            f"stays browsable and annotatable as a remote source; only "
            f"training needs a materialised copy."
        )
    size = materialisation_bytes(entry)
    if size and size > max_bytes:
        raise ValueError(
            f"'{stem}' is {size // (1024 * 1024)} MB at full resolution, over "
            f"the {max_bytes // (1024 * 1024)} MB the browser can materialise "
            f"in one piece. It stays browsable and annotatable as a remote "
            f"source; only training needs a materialised copy."
        )


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def parse_png_dims(header: bytes) -> Optional[List[int]]:
    """``[width, height]`` from the first bytes of a PNG file.

    The IHDR chunk is mandatory and always first: 8-byte signature, 4-byte
    chunk length, ``b'IHDR'``, then width and height as big-endian uint32 —
    so 24 bytes are enough. Returns ``None`` for anything that is not a
    plausible PNG header (the caller treats that as "dimensions unknown",
    never an error)."""
    if (
        len(header) < 24
        or not header.startswith(PNG_SIGNATURE)
        or header[12:16] != b"IHDR"
    ):
        return None
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    if width <= 0 or height <= 0:
        return None
    return [width, height]


# Byte order and version, in the four combinations TIFF allows: "II" little /
# "MM" big endian, then magic 42 (classic) or 43 (BigTIFF). The encoder emits
# classic below 4 GiB and BigTIFF above it, so both have to pass.
TIFF_SIGNATURES = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def looks_like_tiff(header: bytes) -> bool:
    """True if *header* starts like a TIFF file.

    Only a signature check. It exists to catch the case where a presigned PUT
    silently stored something other than the encoder's output (an error page,
    a truncated body, a stale object), before the manifest is flipped to say a
    readable image lives there. It cannot tell a valid pyramid from a corrupt
    one, and does not try.
    """
    return isinstance(header, (bytes, bytearray)) and bytes(header).startswith(
        TIFF_SIGNATURES
    )


# ---------------------------------------------------------------------------
# Direct upload of one local image
#
# A browser without the File System Access API cannot mount a folder, and the
# mounted folder is what registers the data-provider service every other
# upload route goes through. Such a browser therefore had no way to put a
# local image into a dataset at all. These rules exist so an image that
# arrives by presigned PUT is indistinguishable, afterwards, from one that
# arrived through a mount: the object lands at ``images/<stem><ext>``, and the
# manifest entry is then produced by the ordinary listing reconciliation
# rather than written by the upload path.
#
# The extension list is the one the frontend already applies when it reads a
# mounted folder. It is restated HERE, server-side, deliberately: a presigned
# PUT is handed to the client, so a rule that lives only in the client is a
# suggestion. Before this there was no server-side format rule, because no
# server-side upload route existed to need one.
LOCAL_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# Same ceiling as a materialisation: both end up as one object in the same
# bucket, read by the same viewer, and there is no reason for two numbers.
MAX_LOCAL_IMAGE_BYTES = MAX_MATERIALISE_BYTES


def file_size_from_listing(entries: Any, name: str) -> Optional[int]:
    """The stored size of *name* in a ``list_files`` result, or None.

    MEASURED before this was written, because the whole enforcement rests on
    it: a STAGED listing already reports the size of an object that has just
    been PUT, with no commit and no download. On a staged artifact,
    `list_files(..., stage=True)` returned
    ``[{"type": "file", "name": "probe.bin", "size": 12345,
    "last_modified": ...}]`` for a 12345-byte body — the exact length. So the
    size can be read where the signature check already looks, rather than by
    fetching the object.

    None means "no size for that name here", which is not the same as "zero":
    the file may be absent, or a store may not report sizes. The caller decides
    what to do about it, and :func:`oversized_image_refusal` refuses rather
    than waving it through.
    """
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("name") != name:
            continue
        if entry.get("type") == "directory" or entry.get("is_dir"):
            continue
        size = entry.get("size")
        # A bool is an int in Python and `True` would compare as 1. Rejected
        # explicitly, because a store reporting a flag where a size belongs
        # would otherwise read as a one-byte file.
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            return None
        return size
    return None


def oversized_image_refusal(
    name: str, size: Optional[int], max_bytes: int
) -> Optional[str]:
    """A refusal naming BOTH the ceiling and the size found, or None to allow.

    The ceiling was advertised to the client by the minting call and enforced
    nowhere, so an object of any size that merely STARTED with the declared
    format's signature became a dataset image. This is the enforcement.

    An UNREADABLE size refuses too, and that direction is deliberate: the
    alternative admits exactly the object the ceiling exists to keep out, and
    it would do so silently. The refusal says which case it is, so a store that
    stops reporting sizes is a loud failure rather than a quiet hole.

    The boundary is INCLUSIVE: a file of exactly *max_bytes* is allowed, since
    the number is advertised as a maximum and a client that honours it to the
    byte must not be refused.
    """
    if size is None:
        return (
            f"The size of '{name}' could not be read, so the {max_bytes}-byte "
            "limit cannot be checked. The upload was not accepted. Re-run with "
            "done=False and PUT again; if this repeats, the store is not "
            "reporting object sizes and the limit cannot be enforced."
        )
    if size > max_bytes:
        return (
            f"'{name}' is {size} bytes, over the {max_bytes}-byte limit for a "
            "local image upload. It was not accepted. The limit is reported as "
            "'max_bytes' by the call that mints the upload URL."
        )
    return None


# ---------------------------------------------------------------------------
# The upload marker: provenance for the discard
# ---------------------------------------------------------------------------
#
# Discarding an object that is too big needs an answer to exactly one question
# first: did THIS route put it there? Nothing readable in the store answers it.
# Measured against the live artifact manager, an object that has just been PUT
# into an open stage is returned by EVERY listing view there is -- `stage=True`,
# `stage` unset, `stage=False`, `version="v0"` and `version="latest"` all listed
# it. So the committed listing cannot distinguish "the caller uploaded this a
# moment ago" from "this was already an image in the dataset", and a guard built
# on that distinction does not merely fail to protect the pre-existing image: it
# refuses every legitimate upload, because every legitimate upload also appears
# in the committed view.
#
# So the route records the fact at the moment it creates it, instead of trying
# to infer it afterwards. Minting an upload URL writes a marker OUTSIDE
# `images/`; finishing requires one, and the marker is the only thing that
# authorises touching the object. That bounds the discard to objects this route
# minted -- and because the mint refuses a filename that is already an image, an
# object with a marker is provably not a pre-existing image. The guarantee is
# the conjunction of those two rules, so neither may be removed alone.

UPLOAD_MARKER_DIR = ".uploads"
UPLOAD_MARKER_VERSION = 1

# An hour. Long enough that a slow client pushing the 64 MB ceiling over a poor
# link is not refused in mid-flight, short enough that a crashed client's marker
# does not sit in the artifact indefinitely.
UPLOAD_MARKER_TTL_S = 3600.0

# How many markers one sweep will REMOVE per call. Bounded so that a caller
# cannot turn a mint into an unbounded series of deletions; the sweep is
# housekeeping, and doing less of it per call is merely slower, never wrong --
# the authorisation check treats an expired marker as no marker whether or not
# it was ever swept, and the withholding holds a path until its object is gone.
#
# ⚠ THIS DOES NOT BOUND THE READ, and it used to. A review measured both ways
# that cost: the finish consumed the markers for its path out of a capped list,
# so a marker past the 50th entry -- including the finishing upload's own -- was
# never seen and the fail-closed check passed on a list that did not contain it;
# and the withholding was computed from the same capped list, so with more than
# 50 markers in flight an unfinished upload past the cap was a dataset image
# again. A cap on a read that a correctness decision depends on is not a bound,
# it is a silent wrong answer. Both now read every marker; only the removals
# below are capped.
UPLOAD_MARKER_SWEEP_LIMIT = 50

# How long a minted upload URL stays writable. Passed explicitly to `put_file`,
# whose own default is an HOUR.
#
# WHY IT IS SHORT, AND WHAT IT DOES NOT FIX. The size check is point-in-time: it
# reads the object once, when the upload is finished. The presigned URL stays
# writable for its whole life and, under edit-staging, writes the same key the
# image lives at -- so after a successful finish the client can PUT an oversized
# body to the same URL and nothing checks it again. Measured in review: finish
# 32 bytes, accepted; PUT 10,000 bytes to the same URL; the listing size became
# 10,000. The enforcement point is the FINISH, and it is a statement about the
# bytes at that moment, not a guarantee about the stored object forever.
#
# This app cannot close that without support from the storage service. What it
# can do is bound the window, which is the whole purpose of this constant: five
# minutes is ample for the 64 MB ceiling over a poor link while leaving far less
# time than an hour in which the trick is available. Deliberately much shorter
# than UPLOAD_MARKER_TTL_S above: the marker has to outlive the URL so that a
# client whose PUT lands just before expiry can still finish.
UPLOAD_URL_TTL_S = 300

# Hex only, fixed length. A mint id is compared against the one in a stored
# marker document, and a fixed charset keeps that comparison a comparison: an
# id that could contain `/` or `..` must never be usable to aim anything, here
# or in any future caller, so the shape is refused at the edge rather than
# trusted because today's lookup happens not to use it as a filename.
MINT_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def new_mint_id() -> str:
    """A fresh mint id: 32 lowercase hex characters, matching MINT_ID_RE."""
    return uuid.uuid4().hex


def upload_marker_path_for_object(object_path: Any) -> Optional[str]:
    """``.uploads/<sha256(object_path)>.json``, or None if that is not a path.

    **The marker is keyed by the OBJECT PATH, and it used to be keyed by the
    mint id.** That is not a naming preference; it is what makes "one path, one
    marker" a property of the store rather than something a scan has to
    establish. A review measured what the mint-id keying cost, over four rounds:
    two mints for one path left two marker files, so the finish had to find the
    others by LISTING ``.uploads/`` and consuming every marker for its path --
    and that listing is paginated, so every bound on it was a cap that decided
    correctness. A short read let a duplicate marker survive, which withheld the
    committed image and then had the sweep delete it; a bound tight enough to
    terminate against a server that ignores ``offset`` refused valid uploads
    once a few staged removals sat in the directory. With the path as the key
    the second marker cannot exist: a re-mint OVERWRITES the one file, and the
    finish reads and removes exactly that file, found by name, with no listing
    in the path at all.

    **The filename confers nothing.** Authorisation is unchanged and still comes
    from the document -- it must name this exact path AND this exact mint id and
    be unexpired (:func:`marker_authorises_discard`) -- so the hash is a lookup
    key, not a credential. That is what answers the obvious objection to putting
    the path in the name: a caller who can compute the digest of a path learns
    nothing and gains nothing, because knowing which file to read is not
    permission to use it. Hashing also means no caller-supplied character
    reaches the filename, so the old mint-id charset rule is not weakened by
    this: a path can hold ``/`` or ``..`` and its digest cannot.

    **What a lost mint answer now gets, said plainly:** a fresh mint for the
    same path replaces the marker, so a finish presenting the EARLIER mint id is
    refused -- the mint id in the document no longer matches. That is correct
    rather than unfortunate: the superseded URL and the live one write the same
    key, so the bytes a stale finish would publish are whichever PUT landed
    last, and nothing in the request says which. The refusal touches nothing,
    the path stays withheld because the marker is still there, and a retry with
    ``done=false`` mints again. Under the mint-id keying either id could finish,
    which is exactly how two markers for one path arose.

    None rather than a raise, for the reason the mint-id version gave: a path
    this cannot key is answered by the same refusal as a path with no marker.
    """
    if not isinstance(object_path, str) or not object_path:
        return None
    digest = hashlib.sha256(object_path.encode("utf-8")).hexdigest()
    return f"{UPLOAD_MARKER_DIR}/{digest}.json"


def upload_marker_document(
    *,
    path: str,
    mint_id: str,
    max_bytes: int,
    minted_at: float,
    ttl_s: float = UPLOAD_MARKER_TTL_S,
) -> Dict[str, Any]:
    """The marker to store for one minted upload.

    ``expires_at`` is an ABSOLUTE instant, written from the same clock that will
    later be compared against it -- this app's own. That is the whole reason the
    expiry is recorded at mint time rather than derived at read time from the
    object's modification time: on the production NFS filer the server's clock
    runs about 79 seconds ahead of the pods, measured, so `time.time()` compared
    against a stored `st_mtime` or `last_modified` compares two different clocks.
    A marker written a moment ago can then look stale, or dated in the future,
    with nothing in either value saying which clock set it. Two stamps taken
    from one clock cannot do that.
    """
    minted_at = float(minted_at)
    return {
        "marker_version": UPLOAD_MARKER_VERSION,
        "mint_id": mint_id,
        "path": path,
        "max_bytes": int(max_bytes),
        "minted_at": minted_at,
        "expires_at": minted_at + float(ttl_s),
    }


def parse_upload_marker(doc: Any) -> Optional[Dict[str, Any]]:
    """A stored marker as a validated record, or None if it cannot be used.

    Fail-closed on every doubt. An unusable marker has to read as NO marker, so
    that the route refuses and touches nothing, rather than authorising a
    deletion on the strength of a document it does not understand.
    """
    # Mapping, not dict: a document that arrived over hypha-rpc is a `dotdict`,
    # which is NOT a dict subclass, so `isinstance(doc, dict)` is False for it
    # and the whole marker would read as unusable.
    if not isinstance(doc, Mapping):
        return None
    if doc.get("marker_version") != UPLOAD_MARKER_VERSION:
        return None
    mint_id = doc.get("mint_id")
    if not isinstance(mint_id, str) or not MINT_ID_RE.match(mint_id):
        return None
    path = doc.get("path")
    if not isinstance(path, str) or not path:
        return None
    expires_at = doc.get("expires_at")
    # A bool is an int in Python, so `True` would pass a numeric check and then
    # compare as 1.0 -- an expiry in 1970, i.e. always stale.
    if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
        return None
    max_bytes = doc.get("max_bytes")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        return None
    if max_bytes <= 0:
        return None
    return {
        "marker_version": UPLOAD_MARKER_VERSION,
        "mint_id": mint_id,
        "path": path,
        "max_bytes": max_bytes,
        "expires_at": float(expires_at),
    }


def upload_marker_is_expired(marker: Mapping, *, now: float) -> bool:
    """True once *now* has passed the marker's OWN recorded expiry.

    *now* must come from this app's clock -- the clock that wrote ``expires_at``.
    Nothing in here reads a file time or a listing's ``last_modified``; see
    :func:`upload_marker_document` for the measurement that forbids it.
    """
    return float(now) > float(marker["expires_at"])


def marker_authorises_discard(
    marker: Any, *, path: str, mint_id: str, now: float
) -> bool:
    """Whether this marker permits touching the object at *path*.

    Every condition must hold: the marker parses, it was minted for exactly this
    path, it is the marker for exactly this mint id, and it has not expired.

    A false answer means the route refuses and leaves the object alone. That is
    the behaviour this route had before any discard existed, and it is the safe
    direction: an object no marker vouches for is an object whose provenance is
    unknown, and the one thing the discard must never do is delete an image the
    dataset already had.
    """
    parsed = parse_upload_marker(marker)
    if parsed is None:
        return False
    if parsed["path"] != path:
        return False
    if parsed["mint_id"] != mint_id:
        return False
    return not upload_marker_is_expired(parsed, now=now)


def expired_marker_removals(
    markers: Any, *, now: float, limit: int = UPLOAD_MARKER_SWEEP_LIMIT
) -> List[str]:
    """Which marker paths the housekeeping sweep should remove.

    *markers* is an iterable of ``(marker_path, document)`` pairs as read from
    :data:`UPLOAD_MARKER_DIR`. A document that does not parse is swept too: it
    authorises nothing, so it is litter, and leaving it would mean the only
    thing that ever cleans it up is a human.

    Returns at most *limit* paths, so a mint cannot turn into an unbounded
    series of removals.
    """
    out: List[str] = []
    for item in markers or []:
        if len(out) >= max(0, int(limit)):
            break
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue
        marker_path, doc = item
        if not isinstance(marker_path, str) or not marker_path:
            continue
        parsed = parse_upload_marker(doc)
        if parsed is None or upload_marker_is_expired(parsed, now=now):
            out.append(marker_path)
    return out


def abandoned_upload_removals(
    markers: Any, *, now: float, limit: int = UPLOAD_MARKER_SWEEP_LIMIT
) -> List[tuple]:
    """``(marker_path, object_path)`` for uploads abandoned without finishing.

    The companion to :func:`expired_marker_removals`, and the reason it is a
    separate function: sweeping the MARKER is tidiness, while sweeping the
    OBJECT it vouches for is the enforcement. A caller that mints, PUTs an
    oversized body and never finishes is not refused by the finishing check --
    it never makes that call -- so the ceiling has to be applied to what the
    abandonment leaves behind.

    An EXPIRED marker is the warrant for deleting its object, and it is a good
    warrant for exactly the reason the finish-time discard is safe: the marker
    says this app minted that path for an upload that never completed, so the
    object at it was never a dataset image. Nothing else is touched.

    An UNPARSEABLE document is swept as litter by the sibling function but
    yields NO object removal here -- it names no path, so there is nothing it
    could authorise, and guessing one from the filename would be the one way
    this sweep could delete an image the dataset already had.

    Bounded by *limit* for the same reason as the sibling.
    """
    out: List[tuple] = []
    for item in markers or []:
        if len(out) >= max(0, int(limit)):
            break
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue
        marker_path, doc = item
        if not isinstance(marker_path, str) or not marker_path:
            continue
        parsed = parse_upload_marker(doc)
        if parsed is None:
            continue
        if not upload_marker_is_expired(parsed, now=now):
            continue
        out.append((marker_path, parsed["path"]))
    return out


def withheld_upload_paths(markers: Any) -> frozenset:
    """Object paths that must NOT be treated as dataset images yet.

    Every marker that parses withholds its path -- **open or expired**. The
    dataset's image list is built by unioning the stored manifest with a listing
    of ``images/``, so an object that has merely been PUT appears as an image
    whether or not the finishing call was ever made; withholding is what makes
    the finish a gate rather than a formality.

    **Expiry does not stop the withholding, and an earlier version of this was
    wrong about that.** It withheld only unexpired markers, reasoning that an
    expired one's object is about to be deleted by the sweep. A review measured
    the gap: the sweep ran only at the END of a successful mint, which a mint of
    the same name never reaches because the no-overwrite check answers
    ``already`` first -- so on a dataset where nobody uploads anything else, the
    object stayed an image indefinitely and the next commit from any route would
    publish it. The gap was not bounded by anything.

    What makes withholding-until-swept consistent is that the sweep now runs at
    the START of a mint, before the no-overwrite check reads the listing. An
    abandoned object is therefore deleted before that check can answer
    ``already`` about it, so "the image list hides a file" and "a re-mint says
    it already exists" can no longer both be true for the same path.

    An unparseable marker withholds nothing: it names no path. A successful
    finish removes THE marker for its path before committing, so a completed
    image is never withheld -- that is what makes this safe, and it is why the
    finish must fail closed if that removal fails. **"The" marker, singular, is
    a property of the keying**: a marker's filename is the digest of its object
    path, so a re-mint overwrites the one file rather than adding a second.
    Before that, a finish had to find the duplicates by listing this directory,
    and a listing it could not read in full left one open -- which withheld the
    image it had just committed and then had the sweep delete it.
    """
    out = set()
    for item in markers or []:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue
        _marker_path, doc = item
        parsed = parse_upload_marker(doc)
        if parsed is None:
            continue
        out.add(parsed["path"])
    return frozenset(out)


# `markers_for_path` USED TO BE HERE, and its removal is the point of the
# path-keyed marker rather than a tidy-up. It answered "which marker files
# record an upload of this object?", a question only a LISTING of `.uploads/`
# can answer -- and the finish asked it, so the finish's correctness depended on
# reading a paginated directory in full. Four review rounds went on that: a
# short read left a duplicate marker, which withheld the committed image and had
# the sweep delete it; a bound tight enough to terminate against a server that
# ignores `offset` refused valid uploads once staged removals sat in the
# directory. Keying the marker by its object path removes the question instead
# of bounding the answer: there is one marker per path, its name is computable
# from the path, and :func:`upload_marker_path_for_object` finds it without a
# listing. The listing survives only in the two HOUSEKEEPING readers -- the
# withholding, which is fail-open, and the sweep, which is best-effort -- where
# reading less per call costs a later pass rather than a wrong answer.


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"


def split_local_image_filename(filename: Any) -> tuple:
    """``(stem, ext)`` for an uploadable local image, or raise.

    The extension is lowercased so ``Scan.TIF`` and ``scan.tif`` land on one
    rule, but the STEM keeps its case: it is the manifest key and the name
    the user sees, and silently changing it would make the uploaded image a
    different image from the one they picked.
    """
    if not isinstance(filename, str) or not filename:
        raise ValueError("An image filename is required.")
    # Reject a path rather than taking its basename. A caller that sent
    # "a/b.png" has a different idea of where this goes than the broker does,
    # and quietly flattening it would store the image somewhere they do not
    # expect under a stem they did not choose.
    if "/" in filename or "\\" in filename:
        raise ValueError(
            f"'{filename}' is a path, not a filename; send only the final "
            f"component."
        )
    dot = filename.rfind(".")
    if dot <= 0:
        raise ValueError(
            f"'{filename}' has no extension; one of "
            f"{', '.join(LOCAL_IMAGE_EXTS)} is required."
        )
    stem, ext = filename[:dot], filename[dot:].lower()
    if ext not in LOCAL_IMAGE_EXTS:
        raise ValueError(
            f"'{ext}' is not a supported image format. Supported: "
            f"{', '.join(LOCAL_IMAGE_EXTS)}."
        )
    if not is_valid_image_stem(stem):
        raise ValueError(
            f"'{stem}' is not a usable image name. Use letters, digits, "
            f"dot, dash or underscore only."
        )
    return stem, ext


def looks_like_png(header: bytes) -> bool:
    """True if *header* starts like a PNG file."""
    return isinstance(header, (bytes, bytearray)) and bytes(header).startswith(
        PNG_SIGNATURE
    )


def looks_like_jpeg(header: bytes) -> bool:
    """True if *header* starts like a JPEG file."""
    return isinstance(header, (bytes, bytearray)) and bytes(header).startswith(
        JPEG_SIGNATURE
    )


def header_matches_ext(ext: str, header: bytes) -> bool:
    """True if *header*'s signature agrees with the declared *ext*.

    Checked because a presigned PUT reports success from S3, not from this
    app: "the client says it uploaded a PNG" is not evidence that a PNG is
    there. An error page or a truncated body stored under a .png name would
    otherwise be recorded as a readable image.

    An unknown extension is False rather than True. Reaching here with one
    means ``split_local_image_filename`` has been bypassed, and the safe
    reading of a rule that did not run is that it did not pass.
    """
    ext = (ext or "").lower()
    if ext == ".png":
        return looks_like_png(header)
    if ext in (".jpg", ".jpeg"):
        return looks_like_jpeg(header)
    if ext in (".tif", ".tiff"):
        return looks_like_tiff(header)
    return False


def embedding_path(stem: str, model_type: str) -> str:
    return f"embeddings/{stem}_{model_type}.npz"


def embedding_removal_paths(stem: str, model_type: Optional[str] = None) -> List[str]:
    """Candidate cache paths to delete for one image: a single model's
    ``.npz`` when ``model_type`` is given, otherwise every known model type
    plus the legacy suffix-less ``{stem}.npz``."""
    if model_type:
        return [embedding_path(stem, model_type)]
    paths = [embedding_path(stem, mt) for mt in KNOWN_MODEL_TYPES]
    paths.append(f"embeddings/{stem}.npz")
    return paths


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
        complete = [
            ts for ts, files in ts_map.items() if "png" in files and "geojson" in files
        ]
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


def count_pairs_by_stem(filenames: List[str]) -> Dict[str, int]:
    """Given a flat filename listing from one user's label folder, return
    ``{stem: n}`` where *n* is the number of timestamps with BOTH a png and
    a geojson — i.e. how many complete annotation saves that user made for
    the stem. Stems with zero complete pairs are omitted.
    """
    by_stem_ts: Dict[str, Dict[str, set]] = {}
    for filename in filenames:
        parsed = parse_annotation_filename(filename)
        if not parsed:
            continue
        stem, ts, ext = parsed["stem"], parsed["timestamp"], parsed["ext"]
        by_stem_ts.setdefault(stem, {}).setdefault(ts, set()).add(ext)

    result: Dict[str, int] = {}
    for stem, ts_map in by_stem_ts.items():
        n = sum(1 for exts in ts_map.values() if {"png", "geojson"} <= exts)
        if n:
            result[stem] = n
    return result


# ---------------------------------------------------------------------------
# Per-label train/test splits (add-only snapshots of annotation progress)
# ---------------------------------------------------------------------------

# A split is a per-label, named, ADD-ONLY snapshot: only annotated images may
# enter, images never move between sets, and an image that was ever in train
# can never later be in test (fine-tuning lineage integrity). Multiple named
# splits per label support k-fold-style reuse of one dataset.

SPLIT_NAME_RE = LABEL_NAME_RE  # same charset rules as label names


def is_valid_split_name(name: str) -> bool:
    return bool(name) and bool(SPLIT_NAME_RE.match(name))


def splits_dir_path(label: str) -> str:
    return f"{label_folder(label)}/splits"


def split_path(label: str, name: str) -> str:
    return f"{splits_dir_path(label)}/{name}.json"


def _clean_stems(stems: Any) -> List[str]:
    if not isinstance(stems, list):
        return []
    seen, out = set(), []
    for s in stems:
        s = str(s)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _require_annotated(
    stems: List[str], annotation_counts: Dict[str, int], what: str
) -> None:
    missing = [s for s in stems if annotation_counts.get(s, 0) < 1]
    if missing:
        raise ValueError(
            f"Cannot put unannotated images into the {what} set: {', '.join(sorted(missing))}. "
            "A split is a snapshot of annotation progress; annotate these images first."
        )


def new_split(
    label: str,
    name: str,
    train: List[str],
    test: List[str],
    annotation_counts: Dict[str, int],
    created_by: Optional[str],
    ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a fresh split document. *annotation_counts* maps every currently
    annotated stem of the label to its number of complete annotation saves;
    membership is validated against it (snapshot semantics)."""
    if not is_valid_split_name(name):
        raise ValueError(f"Invalid split name {name!r}: must match ^[a-z0-9._-]+$")
    train = _clean_stems(train)
    test = _clean_stems(test)
    if not train:
        raise ValueError("A split needs at least one training image.")
    overlap = set(train) & set(test)
    if overlap:
        raise ValueError(
            f"Stems cannot be in both train and test: {', '.join(sorted(overlap))}"
        )
    _require_annotated(train, annotation_counts, "train")
    _require_annotated(test, annotation_counts, "test")
    members = set(train) | set(test)
    if ratio is None:
        ratio = len(train) / (len(train) + len(test))
    ratio = float(ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    ts = now_iso()
    return {
        "name": name,
        "label": label,
        "created_at": ts,
        "created_by": created_by,
        "updated_at": ts,
        "updated_by": created_by,
        "ratio": ratio,
        "train": train,
        "test": test,
        "annotation_counts": {s: int(annotation_counts[s]) for s in sorted(members)},
        "history": [],
        "checkpoint": None,
    }


def extend_split(
    split: Dict[str, Any],
    add_train: List[str],
    add_test: List[str],
    annotation_counts: Dict[str, int],
    updated_by: Optional[str],
) -> Dict[str, Any]:
    """Add-only extension of an existing split document.

    Guards: added stems must be annotated and new to BOTH sets, and nothing
    that was ever in train may enter test (leakage guard; add-only sets mean
    the current train list IS the full train history). Refreshes the
    annotation-count snapshot for every member and appends a history entry.
    Mutates and returns *split*.
    """
    add_train = _clean_stems(add_train)
    add_test = _clean_stems(add_test)
    if not add_train and not add_test:
        raise ValueError("Nothing to add: both add_train and add_test are empty.")
    cur_train = list(split.get("train") or [])
    cur_test = list(split.get("test") or [])
    leaked = set(add_test) & set(cur_train)
    if leaked:
        raise ValueError(
            f"Refusing to move ever-trained images into test: {', '.join(sorted(leaked))}. "
            "Images that were part of a training set would leak into the evaluation."
        )
    already = (set(add_train) | set(add_test)) & (set(cur_train) | set(cur_test))
    if already:
        raise ValueError(
            f"Already in the split (images never move between sets): {', '.join(sorted(already))}"
        )
    both = set(add_train) & set(add_test)
    if both:
        raise ValueError(
            f"Stems cannot be in both train and test: {', '.join(sorted(both))}"
        )
    _require_annotated(add_train, annotation_counts, "train")
    _require_annotated(add_test, annotation_counts, "test")

    split["train"] = cur_train + add_train
    split["test"] = cur_test + add_test
    members = set(split["train"]) | set(split["test"])
    # Refresh the snapshot for every member so the frontend's "+N new
    # annotations" indicator resets after each extension. A member that lost
    # its files entirely keeps count 0 rather than erroring.
    split["annotation_counts"] = {
        s: int(annotation_counts.get(s, 0)) for s in sorted(members)
    }
    ts = now_iso()
    split["updated_at"] = ts
    split["updated_by"] = updated_by
    split.setdefault("history", []).append(
        {"at": ts, "by": updated_by, "added_train": add_train, "added_test": add_test}
    )
    return split


def split_summary(split: Dict[str, Any]) -> Dict[str, Any]:
    """Compact listing row for list_splits."""
    return {
        "name": split.get("name"),
        "label": split.get("label"),
        "n_train": len(split.get("train") or []),
        "n_test": len(split.get("test") or []),
        "ratio": split.get("ratio"),
        "created_at": split.get("created_at"),
        "updated_at": split.get("updated_at"),
        "checkpoint": split.get("checkpoint"),
    }


# ---------------------------------------------------------------------------
# Embedding filename parsing
# ---------------------------------------------------------------------------

# Longest-model-type-name-first so "vit_l_lm" isn't mis-split as "vit_l".
KNOWN_MODEL_TYPES = (
    "vit_b_lm",
    "vit_l_lm",
    "vit_t_lm",
    "vit_b_em_organelles",
    "vit_l_em_organelles",
    "vit_t_em_organelles",
    "vit_b",
    "vit_l",
    "vit_h",
)

# The annotate page's pinned default (frontend MICRO_SAM_MODEL_TYPE). Used to
# pick the backward-compatible single ``model_type`` an index entry reports
# when a stem has embeddings from more than one model.
DEFAULT_MODEL_TYPE = "vit_l_lm"


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


def collect_embeddings(filenames: List[str]) -> Dict[str, Dict[str, Any]]:
    """Aggregate ``embeddings/`` folder listings into per-stem index entries.

    Returns ``{stem: {"model_type": ..., "model_types": [...]}}`` where
    ``model_types`` is the sorted list of every model with a stored embedding
    for that stem, and ``model_type`` is the single backward-compatible pick
    older clients read: ``DEFAULT_MODEL_TYPE`` when present, else the first
    entry of the sorted list."""
    by_stem: Dict[str, set] = {}
    for name in filenames:
        parsed = parse_embedding_filename(name)
        if parsed:
            by_stem.setdefault(parsed["stem"], set()).add(parsed["model_type"])
    result: Dict[str, Dict[str, Any]] = {}
    for stem, models in by_stem.items():
        model_types = sorted(models)
        preferred = (
            DEFAULT_MODEL_TYPE if DEFAULT_MODEL_TYPE in models else model_types[0]
        )
        result[stem] = {"model_type": preferred, "model_types": model_types}
    return result


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


# ---------------------------------------------------------------------------
# The public gallery
# ---------------------------------------------------------------------------

# What an anonymous visitor is told about a public dataset.
#
# An ALLOW-LIST, not a strip-list. Broker metadata holds the owner as
# ``{"id", "email"}``, and the artifact manifests carry ``created_by`` with an
# address in it, so a listing built by removing known-sensitive fields has to
# be right about every field that exists now and every field added later. This
# only has to be right about what a gallery card needs.
PUBLIC_DATASET_FIELDS = (
    "artifact_id",
    "name",
    "description",
    "image_count",
    "owner_display",
)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def iter_public_metadata(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Every dataset whose BROKER flag says public, in a stable order.

    Reads the broker's own metadata rather than the artifacts, for two
    reasons. The flag lives here and nowhere else -- ``set_public`` writes it
    here -- so this is the same source ``resolve_role`` consults when it
    decides whether an anonymous caller may read a dataset at all. A gallery
    built on any other source could advertise a dataset the read path then
    refuses, or hide one it would serve.

    A file that cannot be parsed is skipped rather than fatal: one bad record
    must not empty the gallery.
    """
    out: List[Dict[str, Any]] = []
    for path in sorted(datasets_dir(root).glob("*.json")):
        try:
            meta = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(meta, dict) or not meta.get("public"):
            continue
        if not meta.get("artifact_id"):
            continue
        out.append(meta)
    return out


def owner_display(metadata: Dict[str, Any]) -> Optional[str]:
    """A display string for the owner that is NEVER an email address.

    Broker metadata stores ``{"id", "email"}`` and nothing resembling a name,
    so the id is the only non-email identifier available. When the id is
    itself an address -- some logins record one -- there is nothing publishable
    and this returns ``None`` rather than inventing something.
    """
    owner = metadata.get("owner") or {}
    ident = owner.get("id")
    if isinstance(ident, str) and ident and not _EMAIL_RE.search(ident):
        return ident
    return None


def public_dataset_entry(
    metadata: Dict[str, Any],
    *,
    name: Optional[str],
    description: Optional[str],
    image_count: Optional[int],
) -> Dict[str, Any]:
    """Project one public dataset down to its gallery card.

    Every value is scrubbed for email addresses on the way out, not just the
    owner field: ``name`` and ``description`` are free text an owner typed and
    could contain one.
    """
    entry = {
        "artifact_id": metadata.get("artifact_id"),
        "name": _scrub_emails(name),
        "description": _scrub_emails(description),
        "image_count": image_count,
        "owner_display": _scrub_emails(owner_display(metadata)),
    }
    return {k: entry.get(k) for k in PUBLIC_DATASET_FIELDS}


def _scrub_emails(value: Any) -> Any:
    """Replace any email address in *value* with a placeholder."""
    if not isinstance(value, str):
        return value
    return _EMAIL_RE.sub("[email removed]", value)


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


def read_metadata(
    artifact_id: str, root: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    p = metadata_path(artifact_id, root)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def write_metadata(
    metadata: Dict[str, Any], root: Optional[Path] = None
) -> Dict[str, Any]:
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


def add_label(
    metadata: Dict[str, Any], name: str, description: str = ""
) -> Dict[str, Any]:
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
    mode" error.

    Matched loosely, because the exact wording is not pinned down by any
    contract. The prefix is ``stag`` rather than ``stage`` deliberately:
    Hypha's actual message is ``Artifact must be in staging mode.``, and
    "staging" does *not* contain the substring "stage". Matching on the
    longer form silently disabled the whole restage-and-retry path for the
    one message it exists to catch, which surfaces as a hard failure on the
    first write after a commit.

    A false positive costs one redundant ``edit(stage=True)`` and a retry,
    so erring wide is the cheap direction.
    """
    return "stag" in str(exc).lower()


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


def is_cache_fresh(
    cached_at: float, ttl_s: float = 60.0, now: Optional[float] = None
) -> bool:
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
