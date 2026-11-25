"""μSAM fine-tuning core for the micro-sam app.

Ingests annotated image/label pairs, builds SAM training loaders, and runs
``micro_sam.training.train_sam`` with the AIS instance-segmentation decoder. The
session/status machinery (session dir + atomic file-backed ``status.json`` +
cooperative stop marker) mirrors the pattern in
``apps/cellpose-finetuning/main.py`` — the pattern is reused, not the code.

Imported lazily inside ``MicroSAM`` method bodies: heavy deps (``torch``,
``micro_sam``, ``torch_em``) live here, so the ``@bioengine.app`` decorator
module stays introspectable with only ``bioengine[worker]`` + the stdlib.
"""

import json
import os
import re
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

STATUS_STALE_SECONDS = 600


# === session paths + file-backed status ===


def sessions_root() -> Path:
    # Under the container's mounted ~/.bioengine so sessions survive restarts.
    root = Path.home() / ".bioengine" / "micro_sam_sessions"
    root.mkdir(parents=True, exist_ok=True)
    return root


# The exact shape ``new_session_id`` generates, and the ONLY shape accepted from
# a caller. An allow-list, not a sanitiser: there is no attempt to clean a bad id
# up, because every cleanup is a guess at what the caller meant and the guesses
# are what get escaped.
#
# ``\A``/``\Z`` and not ``^``/``$``: in Python ``$`` also matches immediately
# before a trailing newline, so ``re.match(r"^...$", "<valid-id>\n")`` SUCCEEDS.
# An id with a newline on the end would then be accepted and used as a directory
# name. There is a test for exactly that.
SESSION_ID_PATTERN = re.compile(r"\A\d{4}-\d{2}-\d{2}-\d{6}-[0-9a-f]{8}\Z")


class InvalidSessionId(ValueError):
    """A session_id that this server could not have generated.

    Raised BEFORE any filesystem call. It is a ``ValueError`` so that callers
    which already report bad arguments keep reporting them the same way.
    """


class SessionNotFound(ValueError):
    """A well-formed session_id for a session that does not exist here.

    Deliberately distinct from :class:`InvalidSessionId`: one says the caller
    sent something the server could not have issued, the other says the caller
    sent something plausible that this worker does not have. Collapsing them
    would make a legitimate "wrong worker" mistake read like an attack.
    """


def validate_session_id(session_id: Any) -> str:
    """The session_id, or raise.

    Checked against the generator's own shape, which was verified to cover every
    run in production: all 261 live session ids match it, so nothing existing
    becomes unreachable by enforcing it.
    """
    if not isinstance(session_id, str):
        raise InvalidSessionId(
            f"session_id must be a string, got {type(session_id).__name__}"
        )
    if not SESSION_ID_PATTERN.match(session_id):
        # The value is NOT echoed in full: it is attacker-controlled and this
        # message reaches logs. The length and a short prefix are enough to
        # debug a genuine mistake.
        raise InvalidSessionId(
            f"session_id is not a valid session identifier "
            f"(length {len(session_id)}, starts {session_id[:12]!r}). "
            f"Expected the form YYYY-MM-DD-HHMMSS-xxxxxxxx."
        )
    return session_id


def session_dir(session_id: str) -> Path:
    """The directory for *session_id*, guaranteed to be inside the sessions root.

    Two independent layers, deliberately not one:

    1. :func:`validate_session_id` refuses anything the server would not itself
       have generated, before any path is built.
    2. The built path is resolved and asserted to be inside the resolved root.

    Layer 2 is not redundant with layer 1 — it is what still holds if layer 1 is
    ever loosened, and loosening an id format is a normal thing to want to do.
    Softening either one alone leaves the other refusing, which is the whole
    point: each has its own failing test.

    This function is the chokepoint. ``_status_path``, ``_stop_path``,
    ``_params_path`` and ``checkpoint_path`` all build on it, so guarding here
    covers every method that takes a session_id rather than one of them.
    """
    validate_session_id(session_id)
    root = sessions_root()
    candidate = root / session_id
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    try:
        # `Path.is_relative_to` is 3.9+, and this app's suite also runs on 3.8.
        resolved.relative_to(resolved_root)
    except ValueError:
        raise InvalidSessionId(
            f"session_id resolves outside the sessions root "
            f"(length {len(session_id)}, starts {session_id[:12]!r})"
        )
    return candidate


def session_backend(session_id: str) -> str:
    """Fine-tuning backend a session used — 'microsam' (default) or 'cellpose'.
    Recorded in status.json at session start; the two backends save their
    checkpoint in different places, so this drives ``checkpoint_path``."""
    return read_status(session_id).get("backend", "microsam")


# Our public cpsam/cpdino model_type strings → the cellpose ``pretrained_model``
# name to initialise a base net from. cellpose 4.x ignores its ``model_type``
# kwarg and selects the architecture by this string (or auto-detects the backbone
# when given a fine-tuned checkpoint path). "cpsam" maps to "cpsam_v2" to preserve
# the base the validated cpsam path already trains/serves from.
_CELLPOSE_BASE = {"cpsam": "cpsam_v2", "cpdino": "cpdino", "cpdino-vitb": "cpdino-vitb"}


def cellpose_base_model(model_type: str) -> str:
    """cellpose ``pretrained_model`` name for one of our cpsam/cpdino model types."""
    return _CELLPOSE_BASE.get(model_type, "cpsam_v2")


# === training metrics: a fixed set the caller opts into ===

# What ``start_training(metrics=[...])`` accepts. "loss" is the per-epoch
# train/validation loss curves ``train_seg`` already returns — free, hence the
# default. "pixel" and "instance" both score the finished net on the validation
# split and share one held-out inference pass; "instance" adds Hungarian
# matching at several IoU thresholds, which is the expensive part and why
# neither is on by default.
TRAINING_METRICS = ("loss", "pixel", "instance")
DEFAULT_TRAINING_METRICS = ("loss",)
# Metrics scored against held-out data, so they need val_images/val_labels.
METRICS_NEEDING_VALIDATION = ("pixel", "instance")
INSTANCE_IOU_THRESHOLDS = (0.5, 0.75, 0.9)


def validate_metrics(metrics: Optional[List[str]]) -> List[str]:
    """Normalise a caller's metric selection against ``TRAINING_METRICS``.

    Unknown names are rejected rather than dropped, because in the returned
    status a typo that silently computed nothing is indistinguishable from a
    metric that legitimately produced nothing."""
    if metrics is None:
        return list(DEFAULT_TRAINING_METRICS)
    unknown = sorted({m for m in metrics if m not in TRAINING_METRICS})
    if unknown:
        raise ValueError(
            f"Unknown training metric(s) {unknown}. Choose from "
            f"{list(TRAINING_METRICS)}."
        )
    return [m for m in TRAINING_METRICS if m in metrics]


# Minimum total GPU VRAM (MB) a model needs to fine-tune AT ALL, at the smallest
# viable config (micro-sam: n_objects_per_batch=1, patch 256; cellpose:
# batch_size=1). The training gate compares detected device VRAM against this to
# offer/reject a model per-hardware — it does NOT bound n_objects_per_batch, which
# stays user-tunable. Anchored empirically on deNBI's T4 (~14.9 GiB usable): ViT-L
# OOMs there, ViT-B/T fit. cpsam/cpdino tiers set from the same calibration sweep;
# vit_h has no cluster that fits it, which the gate should reflect.
MODEL_TRAIN_VRAM_MB = {
    "vit_t_lm": 6000,
    "vit_t_em_organelles": 6000,
    "vit_b_lm": 12000,
    "vit_b_em_organelles": 12000,
    "vit_b": 12000,
    "vit_l_lm": 24000,
    "vit_l_em_organelles": 24000,
    "vit_l": 24000,
    "vit_h": 40000,
    "cpsam": 14000,
    "cpdino": 14000,
    "cpdino-vitb": 10000,
}

_MAX_VRAM_MB = max(MODEL_TRAIN_VRAM_MB.values())


def min_training_vram_mb(model_type: str) -> int:
    """Minimum total GPU VRAM (MB) needed to fine-tune ``model_type``. Unknown
    types fail closed at the largest known requirement."""
    return MODEL_TRAIN_VRAM_MB.get(model_type, _MAX_VRAM_MB)


# Machine-readable grouping keys the training UI keys off instead of string-matching
# model_type: ``family`` groups by architecture, ``size`` is the capacity tier.
MODEL_FAMILY = {
    "vit_t_lm": "lm",
    "vit_b_lm": "lm",
    "vit_l_lm": "lm",
    "vit_t_em_organelles": "em_organelles",
    "vit_b_em_organelles": "em_organelles",
    "vit_l_em_organelles": "em_organelles",
    "vit_b": "sam",
    "vit_l": "sam",
    "vit_h": "sam",
    "cpsam": "cpsam",
    "cpdino": "cpdino",
    "cpdino-vitb": "cpdino",
}
MODEL_SIZE = {
    "vit_t_lm": "tiny",
    "vit_t_em_organelles": "tiny",
    "vit_b_lm": "base",
    "vit_b_em_organelles": "base",
    "vit_b": "base",
    "cpdino-vitb": "base",
    "vit_l_lm": "large",
    "vit_l_em_organelles": "large",
    "vit_l": "large",
    "cpsam": "large",
    "cpdino": "large",
    "vit_h": "huge",
}


def model_family(model_type: str) -> str:
    return MODEL_FAMILY.get(model_type, "unknown")


def model_size(model_type: str) -> str:
    return MODEL_SIZE.get(model_type, "unknown")


def detect_gpu_memory() -> Dict[str, Any]:
    """Total/free VRAM of the runtime's GPU. Runs GPU-side (torch imported lazily
    so the CPU entry never loads it). ``available`` is False when no CUDA device."""
    import torch

    if not torch.cuda.is_available():
        return {"available": False, "total_mb": 0, "free_mb": 0, "device_name": ""}
    free_b, total_b = torch.cuda.mem_get_info()
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "total_mb": int(props.total_memory / (1024 * 1024)),
        "free_mb": int(free_b / (1024 * 1024)),
        "device_name": props.name,
    }


def checkpoint_path(session_id: str) -> Path:
    """Servable checkpoint for a finished session, by backend:
    micro-sam's torch_em writes ``<dir>/checkpoints/<name>/best.pt``; cellpose's
    ``train_seg`` (save_path=<dir>, model_name='model') writes ``<dir>/models/model``.
    """
    sdir = session_dir(session_id)
    if session_backend(session_id) == "cellpose":
        return sdir / "models" / "model"
    return sdir / "checkpoints" / session_id / "best.pt"


def _status_path(session_id: str) -> Path:
    return session_dir(session_id) / "status.json"


def _stop_path(session_id: str) -> Path:
    return session_dir(session_id) / "stop.requested"


def caller_identity(context: Any) -> Dict[str, Any]:
    """``{"owner", "owner_anonymous"}`` from Hypha's injected caller context.

    Hypha injects ``context`` at the service boundary and the ProxyDeployment
    forwards it, so the identity here is the live caller's and a client cannot
    set it. That is the whole reason a run can be attributed at all.

    **Three states, and the third is not the second.** A known user gives an id
    and ``owner_anonymous`` False; a known anonymous caller gives its id and
    True; and a call that arrives with no usable context gives ``None`` for
    both. The last is NOT "anonymous" — it is "not recorded", which is the shape
    every run written before this field existed also has. A reader that collapses
    the two would either hide a user's own runs or expose an anonymous one, so
    the distinction is kept here rather than flattened for convenience.

    Never raises. A malformed context must not fail a training run that is
    otherwise valid: the run is the user's work, the attribution is bookkeeping,
    and losing the second is not worth losing the first.
    """
    unknown = {"owner": None, "owner_anonymous": None}
    try:
        return _caller_identity(context, unknown)
    except Exception:
        # TOTAL on purpose, and the docstring's promise depends on it. Type
        # checks alone are not enough: a mapping whose own `get` raises slips
        # past every isinstance and takes the training run down with it. That is
        # not hypothetical hardening -- the test for this caught the unguarded
        # version returning an exception instead of "unknown".
        return unknown


def _caller_identity(context: Any, unknown: Dict[str, Any]) -> Dict[str, Any]:
    """The reading itself; :func:`caller_identity` owns the never-raise promise."""
    if not isinstance(context, dict):
        return unknown
    user = context.get("user")
    if not isinstance(user, dict):
        return unknown
    uid = user.get("id")
    if not isinstance(uid, str) or not uid:
        return unknown
    flag = user.get("is_anonymous")
    # Read the flag; do NOT infer anonymity from the shape of the id. An
    # anonymous workspace happens to be named recognisably today and that is a
    # convention, not a contract -- keying on it would silently mislabel the day
    # it changes.
    return {
        "owner": uid,
        "owner_anonymous": bool(flag) if isinstance(flag, bool) else None,
    }


def new_session_id() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


def read_status(session_id: str) -> Dict[str, Any]:
    p = _status_path(session_id)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


_TERMINAL = ("COMPLETED", "FAILED", "STOPPED")

# A terminal status may only be changed by an equal-or-higher-ranked writer.
# user stop must outrank a late child COMPLETED (the user asked it to stop); the
# child (the training outcome) outranks the supervisor (which only sees the
# subprocess rc) and the entry bare-except (which cannot see the child at all).
_TERMINATED_BY = {"entry": 1, "supervisor": 2, "child": 3, "user_stop": 4}


def write_status(session_id: str, **fields) -> Dict[str, Any]:
    """Atomically merge fields into the session's status.json.

    A terminal status is provenance-ranked, not blanket-sticky: once
    COMPLETED/FAILED/STOPPED is recorded, a write to a *different* status is
    dropped only when its writer (``terminated_by``) ranks equal-or-lower than
    the incumbent's. This lets the child's authoritative COMPLETED correct a
    bystander's premature terminal (e.g. the entry bare-except FAILED) while a
    user's STOPPED still survives a late child COMPLETED. A same-status write or
    a field-only write (no ``status``) always lands, so heartbeats and end_time
    refinements are unaffected. An unknown/absent ``terminated_by`` ranks 0."""
    p = _status_path(session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = read_status(session_id)
    cur_status, new_status = cur.get("status"), fields.get("status")
    if cur_status in _TERMINAL and new_status is not None and new_status != cur_status:
        cur_prec = _TERMINATED_BY.get(cur.get("terminated_by"), 0)
        new_prec = _TERMINATED_BY.get(fields.get("terminated_by"), 0)
        if new_prec <= cur_prec:
            return cur
    cur.update(fields)
    cur["updated_at"] = time.time()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cur))
    os.replace(tmp, p)
    return cur


# Fields recorded on a run but NOT returned by the public read path.
#
# `list_training_sessions` is now per-caller (see `list_sessions`), so a run is
# only returned to the identity it was recorded under. That is NOT why these
# fields are withheld. The owner is what the filter COMPARES, so it is read on
# every listing; publishing it would turn a history that says WHAT was trained
# into one that says WHO trained it, for every run the caller can see. Recording
# the owner and keeping it out of the response therefore have to be one change,
# not two: "record now, filter later" publishes the identity in between.
#
# These come back only for a caller that asks with include_private=True, which is
# the hook the per-caller filter uses to compare a request against the run's owner.
# The owner is therefore READ on every listing and PUBLISHED on none.
PRIVATE_STATUS_FIELDS = ("owner", "owner_anonymous")

# Runs that carry no owner are shown ONLY to a caller holding rw or admin on
# this exact workspace. Three narrowings, each deliberate:
#
#   * "r" is EXCLUDED. Read permission is not authority over other people's run
#     history. ``usage_ledger.classify_caller`` counts {"r", "rw", "a"} as
#     operator grants, which is right for its own question — is this caller an
#     insider, for a usage figure meant to be a floor — and wrong for this one.
#     Measured: that function buckets a caller holding only "r" on this
#     workspace as `internal`, where this one refuses it.
#   * the workspace is NAMED, and the caller's OWN workspace does not count.
#     Measured 2026-11-25: every token-less caller is minted a fresh identity
#     holding "a" on its own personal workspace (`ws-user-anonymouz-...`). This
#     function deliberately does NOT short-circuit on `is_anonymous`, because an
#     anonymous caller is a real identity that owns its own runs — so an
#     implementation reading "does this caller hold an operator grant anywhere"
#     would be true for every caller alive, anonymous ones included, and hand
#     the whole unowned history to exactly the callers it must exclude. A mutant
#     doing that turns three tests red.
#   * a WILDCARD grant does NOT count, even carrying rw or admin. A global
#     operator arguably does hold authority here, so accepting "*" would be
#     defensible — but the rule this implements names one workspace, and
#     widening a privacy filter past its rule is not a call to make in passing.
#     Refusing "*" costs a genuine global operator nothing but the ownerless
#     rows; accepting it would show 261 other people's runs to an identity
#     nobody authorised. If "*" is ever wanted, it should arrive as a decision
#     with a name on it.
#
# NB `classify_caller` would NOT have leaked to anonymous callers if reused: it
# returns `anonymous` before ever reading the grants (measured, including with
# `*: a`). The leak above is a hazard of a naive implementation of THIS
# function, which has no such short-circuit. Recorded because an earlier note
# here attributed it to that function, and that was wrong.
OWNERLESS_WORKSPACE = "bioimage-io"
OWNERLESS_GRANTS = frozenset({"rw", "a"})


def caller_is_workspace_admin(
    context: Any, *, workspace: str = OWNERLESS_WORKSPACE
) -> bool:
    """True iff the caller holds rw or admin on *workspace* — ``bioimage-io``.

    Named for what it TESTS, not for one of its uses. Two different rules ask
    this same question: an ownerless run is visible only to such a caller, and
    a mutating call on somebody else's run is allowed only for such a caller.
    Naming it after the first use made the second one read like a mistake.

    A run that carries no owner cannot be attributed to anybody, and there are
    261 of them written before the field existed. They are not guessed at, not
    made public and not deleted: they are shown only to a caller with rw or
    admin on ``bioimage-io`` specifically.

    **Not "the workspace hosting the app".** The name is the constant above, so
    an instance of this app deployed into some other workspace — a dev worker —
    does not treat its own hosting workspace as authoritative. That is the
    intended reading of the rule and not an oversight; it is also why the
    workspace is a keyword argument, so a test can state the name it means.

    Decided from the INJECTED context and from nothing else: there is no
    parameter, so a caller cannot ask to be treated as an administrator. A
    wildcard grant does not qualify — see the note above the constant.

    Never raises: an unusable context is "no authority", which is the closed
    direction. A caller that cannot be read must not inherit the open one.
    """
    if not isinstance(context, dict):
        return False
    user = context.get("user")
    if not isinstance(user, dict):
        return False
    scope = user.get("scope")
    if not isinstance(scope, dict):
        return False
    grants = scope.get("workspaces")
    if not isinstance(grants, dict):
        return False
    return grants.get(workspace) in OWNERLESS_GRANTS


class NotTheOwner(PermissionError):
    """A caller asked to change a run that is not theirs.

    A ``PermissionError`` rather than a ``ValueError``: the request was
    well-formed and the session exists, and the only thing wrong is who is
    asking. Callers that already distinguish "bad argument" from "not allowed"
    keep distinguishing them.
    """


def caller_may_mutate(session_id: str, context: Any) -> bool:
    """True iff *context* identifies the run's owner, or a ``bioimage-io`` admin.

    A run with a recorded owner is matched EXACTLY against the caller's injected
    identity. A run with no usable owner cannot be attributed to anybody, so it
    is admin-only — the same rule the listing applies to the 261 runs recorded
    before the field existed, and for the same reason: guessing an owner is worse
    than admitting there is none.

    Reads nothing but the status that is already on disk and CREATES NOTHING, so
    asking the question about a session that does not exist is safe. It answers
    False for one, because an absent status has no owner and the ownerless rule
    then applies — a non-admin is refused before any write can happen.

    Fails closed: an unreadable context yields no identity and so no match, and
    `caller_is_workspace_admin` is False for it too.
    """
    owner = read_status(session_id).get("owner")
    if isinstance(owner, str) and owner:
        caller = caller_identity(context).get("owner")
        if caller is not None and caller == owner:
            return True
        # NOT a fall-through to "refuse". The rule is "the owner OR an admin",
        # and an earlier version reached the admin test only for OWNERLESS runs
        # -- so an administrator could not touch an owned run, while this
        # function's own docstring said they could. The tests missed it because
        # their admin fixture used the SAME id as the owner, so "the admin can"
        # passed as "the owner can".
        return caller_is_workspace_admin(context)
    return caller_is_workspace_admin(context)


def require_caller_may_mutate(session_id: str, context: Any) -> None:
    """:func:`caller_may_mutate`, or raise :class:`NotTheOwner`.

    The message deliberately does NOT say whether the run exists, who owns it,
    or whether the caller merely lacks admin — a refusal that explains itself
    tells an unauthorised caller which session ids are real.
    """
    if not caller_may_mutate(session_id, context):
        raise NotTheOwner(
            "this training run belongs to another caller; only its owner or a "
            "workspace administrator can change it"
        )


def redact_private(status: Dict[str, Any]) -> Dict[str, Any]:
    """Drop every private field, in place, and return the same dict.

    One definition, used by both the single-run read and the listing. Two copies
    of this loop is how a field added to :data:`PRIVATE_STATUS_FIELDS` ends up
    stripped on one path and published on the other.
    """
    for field in PRIVATE_STATUS_FIELDS:
        status.pop(field, None)
    return status


def get_status(session_id: str, *, include_private: bool = False) -> Dict[str, Any]:
    st = read_status(session_id)
    if not st:
        return {
            "session_id": session_id,
            "status": "UNKNOWN",
            "message": "no such session",
        }
    st["session_id"] = session_id
    st["checkpoint_available"] = checkpoint_path(session_id).exists()
    # The config a run used, so a history listing can say WHAT was trained and not
    # merely that something was. It was already persisted per run and reachable only
    # from inside this app; `{}` for a run that died before the file was written.
    st["config"] = run_config(session_id)
    if st.get("start_time"):
        end = st.get("end_time") or time.time()
        st["elapsed_s"] = round(end - st["start_time"], 1)
    # A run with no live thread that hasn't updated in a while is stale → stopped.
    if (
        st.get("status") in ("PREPARING", "TRAINING")
        and time.time() - st.get("updated_at", 0) > STATUS_STALE_SECONDS
    ):
        st["status"] = "STOPPED"
        st["message"] = "no status update within the stale window"
    if not include_private:
        redact_private(st)

    return st


def list_sessions(context: Any) -> Dict[str, Dict[str, Any]]:
    """The runs this caller may see, keyed by session id, owner redacted.

    *context* is REQUIRED and has no default, deliberately. There is no
    unfiltered listing to call by mistake: the previous `list_sessions()`
    returned every run on the worker to whoever asked, and leaving that
    signature available with a filtered sibling beside it would make the open
    behaviour one forgotten argument away.

    A run is visible iff its recorded owner IS this caller, or it carries no
    owner and the caller may see ownerless runs. Both legs fail closed: an
    unreadable context yields no identity, no authority, and so an empty
    listing. The owner is read here and redacted from everything returned, so
    the field decides visibility and is never itself published.

    **A directory that is not a session is SKIPPED, not fatal.** `get_status`
    validates the id, so listing a directory whose name is not a session id
    raises -- and one such directory would then make the listing raise for
    EVERY caller, turning a stray directory into a total outage of the run
    history. The rule is: validate what a CALLER hands us, tolerate what we
    FIND. A name that cannot be a session id was not produced by this server,
    so leaving it out of the history is honest; refusing to show the history at
    all is not. Skipped names are logged with a bounded sample, since they are
    attacker-controlled.
    """
    identity = caller_identity(context)
    caller = identity.get("owner")
    see_ownerless = caller_is_workspace_admin(context)

    root = sessions_root()
    out: Dict[str, Dict[str, Any]] = {}
    skipped: List[str] = []
    for d in sorted(root.iterdir()) if root.exists() else []:
        if not (d.is_dir() and (d / "status.json").exists()):
            continue
        if not SESSION_ID_PATTERN.match(d.name):
            skipped.append(d.name)
            continue
        try:
            status = get_status(d.name, include_private=True)
        except InvalidSessionId:
            # The name passed but a deeper layer refused it -- today that means
            # a directory whose NAME is a valid session id but which is a
            # SYMLINK out of the sessions root. Skipped like a bad name, so one
            # bad link cannot take the whole history down.
            skipped.append(d.name)
            continue
        owner = status.get("owner")
        if isinstance(owner, str) and owner:
            # A recorded owner is matched EXACTLY, and only against a caller we
            # could actually read. `caller is None` must never match a run.
            visible = caller is not None and caller == owner
        else:
            # No owner, or one unusable as an identity. Both are unattributable.
            visible = see_ownerless
        if visible:
            out[d.name] = redact_private(status)
    if skipped:
        print(
            f"[model-finetune] skipped {len(skipped)} session directory(ies) that "
            f"are not a session on this worker: {skipped[:5]}",
            flush=True,
        )
    return out


def request_stop(session_id: str) -> None:
    if session_dir(session_id).exists():
        _stop_path(session_id).write_text("stop")


def stop_requested(session_id: str) -> bool:
    return _stop_path(session_id).exists()


def exports_root() -> Path:
    # Beside the sessions under the mounted ~/.bioengine, so an export handle
    # outlives the entry replica that minted it and the entry can scale to zero.
    root = Path.home() / ".bioengine" / "exports"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _export_path(export_id: str) -> Path:
    return exports_root() / f"{export_id}.json"


def read_export(export_id: str) -> Dict[str, Any]:
    p = _export_path(export_id)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def write_export(export_id: str, **fields) -> Dict[str, Any]:
    """Atomically merge fields into the export's record."""
    cur = read_export(export_id)
    cur.update(fields)
    cur["export_id"] = export_id
    cur["updated_at"] = time.time()
    p = _export_path(export_id)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cur))
    os.replace(tmp, p)
    return cur


def get_export(export_id: str) -> Dict[str, Any]:
    """The export record, with an unattended one swept to FAILED.

    Only the entry that is building an export advances its record, and nothing
    awaits that build, so the entry can be scaled away mid-export. Without the
    sweep such a record would poll at BUILDING forever; the entry heartbeats it
    while the build is genuinely alive.
    """
    rec = read_export(export_id)
    if (
        rec
        and rec.get("status") in ("PENDING", "BUILDING")
        and time.time() - rec.get("updated_at", 0) > STATUS_STALE_SECONDS
    ):
        rec["status"] = "FAILED"
        rec["message"] = "failed"
        rec["error"] = "the export was interrupted before the package was staged"
    return rec


def _set_pdeathsig() -> None:
    """preexec_fn: ask the kernel to SIGKILL this child when its parent (the runtime
    replica) dies, so a training subprocess cannot outlive an OOM-killed supervisor
    and orphan-hold the GPU. Linux-only (prctl PR_SET_PDEATHSIG=1); a no-op
    elsewhere. Runs in the forked child before exec."""
    try:
        import ctypes
        import signal as _signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, _signal.SIGKILL)
    except Exception:
        pass


def run_cancellable_subprocess(
    argv: List[str],
    session_id: str,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    poll: float = 2.0,
    grace: float = 10.0,
) -> Tuple[int, str, bool]:
    """Run a training subprocess that a concurrent ``stop_training`` can kill.

    Polls the session's stop marker while the child runs; on stop it terminates
    the child (SIGTERM, then SIGKILL after ``grace``). The stop marker is written
    by the entry, which shares this session dir, so this works without the caller
    knowing which runtime replica holds the process. Child stdout+stderr go to a
    log file (not a pipe) so the poll loop never deadlocks on a full pipe buffer.
    Returns ``(returncode, log_tail, stopped)``."""
    import subprocess

    log_path = session_dir(session_id) / "train.log"
    stopped = False
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=_set_pdeathsig,
        )
        while True:
            try:
                proc.wait(timeout=poll)
                break
            except subprocess.TimeoutExpired:
                if stop_requested(session_id):
                    stopped = True
                    proc.terminate()
                    try:
                        proc.wait(timeout=grace)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    break
    try:
        tail = log_path.read_text()[-800:]
    except Exception:
        tail = ""
    return proc.returncode, tail, stopped


# === annotation interpretation ===


def polygon_coordinate_bounds(payload: Any) -> Optional[Dict[str, float]]:
    """``{"x_min","x_max","y_min","y_max","features"}`` over a FeatureCollection's
    polygon rings, or ``None`` when it carries no polygon at all.

    Used to say WHY a mask came back empty. A polygon list whose x runs to 40000
    against a 512-wide canvas is a coordinate-frame mismatch, and printing the
    range turns a silent all-background training set into a diagnosis.
    """
    feats = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(feats, list):
        return None
    xs: List[float] = []
    ys: List[float] = []
    n = 0

    def _walk(coords: Any, depth: int = 0) -> None:
        # Polygon is [[ [x,y], ... ]]; MultiPolygon one level deeper. Walk until
        # a pair of numbers appears rather than trusting the declared type.
        if (
            isinstance(coords, (list, tuple))
            and len(coords) == 2
            and all(isinstance(v, (int, float)) for v in coords)
        ):
            xs.append(float(coords[0]))
            ys.append(float(coords[1]))
            return
        if isinstance(coords, (list, tuple)) and depth < 6:
            for item in coords:
                _walk(item, depth + 1)

    for feat in feats:
        geom = feat.get("geometry") if isinstance(feat, dict) else None
        if not isinstance(geom, dict):
            continue
        if str(geom.get("type") or "") not in ("Polygon", "MultiPolygon"):
            continue
        n += 1
        _walk(geom.get("coordinates"))
    if not n or not xs:
        return None
    return {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
        "features": float(n),
    }


def rasterize_geojson(payload: Dict[str, Any], width: int, height: int) -> np.ndarray:
    """Rasterize a GeoJSON FeatureCollection of polygons to a uint16 instance mask.

    One instance per Polygon/MultiPolygon feature, ids from 1 in document order;
    coordinates are image-pixel (x, y). Ported from cellpose-finetuning's
    ``_rasterize_geojson_to_tiff`` (the bioimage.io colab polygon convention).
    """
    from PIL import Image as PILImage, ImageDraw

    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        features = []
    label_img = PILImage.new("I;16", (width, height), 0)
    draw = ImageDraw.Draw(label_img)
    instance_id = 0
    for feature in features:
        if not isinstance(feature, dict):
            continue
        geom = feature.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        polygons = (
            [coords]
            if gtype == "Polygon"
            else (list(coords) if gtype == "MultiPolygon" else [])
        )
        for poly in polygons:
            if not poly:
                continue
            outer = [(float(p[0]), float(p[1])) for p in poly[0] if len(p) >= 2]
            if len(outer) < 3:
                continue
            instance_id += 1
            draw.polygon(outer, fill=instance_id)
    return np.array(label_img, dtype=np.uint16)


def _to_hwc_or_hw(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    # micro_sam's training loader requires inputs in [0, 255]; normalize the same
    # way inference does (runtime._to_image_format) so train and serve preprocess
    # identically for non-uint8 substrates (e.g. uint16 brightfield).
    if image.dtype != np.uint8:
        image = image.astype("float32")
        axis = (0, 1) if image.ndim == 3 else None
        keepdims = image.ndim == 3
        image = image - image.min(axis=axis, keepdims=keepdims)
        denom = image.max(axis=axis, keepdims=keepdims)
        denom = np.where(denom == 0, 1.0, denom)
        image = (image / denom * 255).astype("uint8")
    return image


def coerce_label_array(label: np.ndarray) -> np.ndarray:
    """Coerce a loaded label mask to a 2D integer instance map.

    The annotation tool saves per-user masks as RGB(A) PNGs, not single-channel
    label images, so those masks arrive here as H×W×3(4). A 2D array passes
    through unchanged. A single-channel array is squeezed. A multi-channel array
    is collapsed: if every channel is identical it is a grayscale label stored in
    RGB (take one channel); otherwise it is a colour-encoded instance map, so each
    distinct colour becomes a distinct id in raster order with pure black treated
    as background (0). Ids are per-image and need not match across images.
    """
    arr = np.asarray(label)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        chan = arr[..., :3]
        if np.all(chan == chan[..., :1]):
            return chan[..., 0]
        flat = chan.reshape(-1, chan.shape[-1])
        colours, inv = np.unique(flat, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        ids = np.zeros(len(colours), dtype=np.int64)
        next_id = 0
        for k, col in enumerate(colours):
            if np.any(col):
                next_id += 1
                ids[k] = next_id
        return ids[inv].reshape(chan.shape[:2])
    raise ValueError(
        f"Unsupported label shape {arr.shape}: expected a 2D mask or an "
        "RGB(A) instance PNG."
    )


def materialize_pairs(
    session_id: str,
    train: List[Tuple[np.ndarray, np.ndarray]],
    val: Optional[List[Tuple[np.ndarray, np.ndarray]]],
    val_fraction: float,
    patch_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """Write (image, instance-label) arrays to per-split TIFFs, drop empty-label
    pairs, split train/val when val is not supplied, and clamp the patch shape to
    the smallest image so training never requests a crop larger than an image.
    """
    import tifffile

    sdir = session_dir(session_id)

    def keep(pairs):
        return [
            (im, lb)
            for im, lb in pairs
            if lb is not None and int(np.asarray(lb).max()) > 0
        ]

    train = keep(train)
    val = keep(val) if val else []
    if not train:
        raise ValueError("No training pairs with foreground labels after ingestion.")
    val_reused_train = False
    if not val:
        n_val = max(1, int(round(len(train) * val_fraction))) if len(train) > 1 else 0
        if n_val:
            val = train[:n_val]
            train = train[n_val:]
        if not val:  # single-image (or otherwise empty) split: reuse train as val
            val = list(train)
            val_reused_train = True

    def dump(split, pairs):
        img_dir = sdir / "data" / split / "images"
        lbl_dir = sdir / "data" / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        imgs, lbls = [], []
        for i, (im, lb) in enumerate(pairs):
            im = _to_hwc_or_hw(np.asarray(im))
            lb = coerce_label_array(np.asarray(lb)).astype(np.uint16)
            ip = img_dir / f"{i:04d}.tif"
            lp = lbl_dir / f"{i:04d}.tif"
            tifffile.imwrite(str(ip), im)
            tifffile.imwrite(str(lp), lb)
            imgs.append(str(ip))
            lbls.append(str(lp))
        return imgs, lbls

    train_imgs, train_lbls = dump("train", train)
    val_imgs, val_lbls = dump("val", val)

    min_h = min(np.asarray(im).shape[0] for im, _ in train + val)
    min_w = min(np.asarray(im).shape[1] for im, _ in train + val)
    ph = (min(patch_shape[0], min_h), min(patch_shape[1], min_w))

    return {
        "train_images": train_imgs,
        "train_labels": train_lbls,
        "val_images": val_imgs,
        "val_labels": val_lbls,
        "patch_shape": ph,
        "n_train": len(train_imgs),
        "n_val": len(val_imgs),
        "val_reused_train": val_reused_train,
    }


# === training params (written by the entry, read by train_worker.py) ===


def _params_path(session_id: str) -> Path:
    return session_dir(session_id) / "training_params.json"


def write_training_params(session_id: str, params: Dict[str, Any]) -> None:
    p = _params_path(session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(params))


def read_training_params(session_id: str) -> Dict[str, Any]:
    return json.loads(_params_path(session_id).read_text())


# What a client may see of a run's configuration, as an ALLOW-LIST rather than a
# redaction list. ``training_params.json`` is written by spreading ``**params`` plus a
# ``**resolved`` block, so its contents are whatever those carry at the time: an
# allow-list stays safe when a field is added there later, while a blocklist would have
# to anticipate the new name. Today nothing in it is a credential — ``pool_token`` is
# consumed during checkpoint resolution and never stored — but a read surface should not
# depend on that staying true.
#
# Deliberately NOT included: ``train_images`` / ``train_labels`` / ``val_images`` /
# ``val_labels``, which are local materialisation paths — useless to a client and
# needless detail about the worker's filesystem — and ``init_checkpoint``, which is a
# path; ``resume_session_id`` and ``pool_provenance`` carry the same meaning as ids.
RUN_CONFIG_FIELDS = (
    "model_type",
    "n_epochs",
    "batch_size",
    "learning_rate",
    "diam_mean",
    "val_fraction",
    "n_samples",
    "train_tile_size",
    "patch_shape",
    "metrics",
    "resume_session_id",
    "pool_provenance",
)


def run_config(session_id: str) -> Dict[str, Any]:
    """The run's configuration for display, restricted to :data:`RUN_CONFIG_FIELDS`.

    Returns ``{}`` rather than raising when the params file is absent or unreadable: a
    run that failed during preparation can terminate before the file is written, and
    those are exactly the sessions a history page still has to list. ``get_status`` is
    on every read path, so it must not acquire a new way to throw."""
    try:
        params = read_training_params(session_id)
    except Exception:
        return {}
    if not isinstance(params, dict):
        return {}
    return {k: params[k] for k in RUN_CONFIG_FIELDS if k in params}
