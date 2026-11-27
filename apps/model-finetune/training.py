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

import contextlib
import json
import os
import re
import tempfile
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

STATUS_STALE_SECONDS = 600


# === session paths + file-backed status ===


class SessionsRootUnavailable(RuntimeError):
    """This replica cannot create, or cannot write, the directory runs are kept in.

    Raised from two places, because they are two different facts: `sessions_root`
    raises it when the store cannot be CREATED, and `assert_sessions_root_writable`
    when the store exists but refuses a WRITE. A store that exists read-only
    satisfies the first and fails the second.

    A ``RuntimeError`` and not a ``ValueError``: nothing the caller asked for is
    wrong. The replica cannot serve ANY run operation, and that is a property of
    where it is running rather than of the request.

    Measured 2026-11-25 on a de.NBI replica whose ``Path.home()`` was
    ``/home/ray`` on a read-only filesystem: `start_training`,
    `list_training_sessions` and `get_training_status` all raised an identical
    bare ``OSError`` from the ``mkdir`` below, while the app reported RUNNING
    and every deployment HEALTHY. Nothing said which of those failures was the
    same failure, or that the replica could never serve a run at all.
    """


def sessions_root() -> Path:
    # Under the container's mounted ~/.bioengine so sessions survive restarts.
    root = Path.home() / ".bioengine" / "micro_sam_sessions"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # NAMED, and says what it means for the app rather than for the syscall.
        # Every run-touching path funnels through here, so a bare mkdir error
        # surfaces identically at a dozen call sites with nothing tying them
        # together -- which is exactly how a replica came to look healthy while
        # being unable to do anything with runs. `_health` in entry.py calls
        # this so the condition reports as UNHEALTHY and Serve replaces the
        # replica, rather than it staying in the routing table forever.
        raise SessionsRootUnavailable(
            f"cannot create the training-run store at {root}: {exc}. This "
            "replica cannot serve any training-run operation while that path "
            "cannot be created."
        ) from exc
    return root


def assert_sessions_root_writable() -> Path:
    """Prove the run store accepts a WRITE, not merely that it exists.

    `sessions_root` only establishes that the directory could be CREATED, and
    `mkdir(exist_ok=True)` succeeds on a directory that already exists however
    read-only it is: Linux reports EEXIST before EROFS and `exist_ok` swallows
    it. So a store that exists and refuses every write left this replica
    HEALTHY while every run write failed -- the precise symptom this check was
    added for. Measured two ways: a pre-created store chmod'ed 0o555, where
    `sessions_root()` returns the path and writing into it raises EACCES; and a
    real read-only mount, where `mkdir(exist_ok=True)` on the existing
    directory succeeds while creating anything inside it raises EROFS.

    So write something. A create-and-unlink is the cheapest probe that cannot
    be satisfied by a path that merely exists.
    """
    root = sessions_root()
    probe = root / f".writable-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        probe.write_text("")
    except OSError as exc:
        raise SessionsRootUnavailable(
            f"the training-run store at {root} exists but refuses writes: "
            f"{exc}. This replica cannot serve any training-run operation."
        ) from exc
    finally:
        try:
            probe.unlink()
        except OSError:
            pass
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


def session_backend(session_id: str, *, status: Optional[Dict[str, Any]] = None) -> str:
    """Fine-tuning backend a session used — 'microsam' (default) or 'cellpose'.

    Recorded in status.json at session start; the two backends save their
    checkpoint in different places, so this drives ``checkpoint_path``.

    *status* lets a caller that has ALREADY read the record answer from it
    instead of reading it again. `get_status` does, and that one re-read was
    half the cost of the run listing: it is called once per run, so a history
    of N runs read 2N records where N would do.
    """
    if status is None:
        status = read_status(session_id)
    return status.get("backend", "microsam")


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


def base_checkpoint_refusal(model_type: Any, status: Any) -> Optional[str]:
    """The refusal text for a base checkpoint this site cannot produce, or None.

    *status* is what `RuntimeApp.base_checkpoint_status` returns: the PARTS,
    not a verdict. The decision lives here, away from the network, so the rule
    can be tested without one and so a reviewer can read the rule in one place.

    FAILS CLOSED ON ITS OWN IGNORANCE, deliberately. A probe that errored, a
    part with no URL and no cached copy, or a part that is neither cached nor
    reachable all refuse. The alternative -- admitting the run when the check
    could not tell -- restores exactly the behaviour this guard exists to
    remove: the caller waits for a GPU slot and finds out minutes later.

    THE MODEL TYPE IS NAMED IN EVERY REFUSAL. The message a caller sees has to
    say which model they asked for, because the fix on their side is to pick a
    different one.

    Returns None when every published part is cached or reachable, so a site
    that CAN train a type still starts -- the control that stops this becoming
    a check which refuses everything.
    """
    if not isinstance(status, dict):
        return (
            f"the base checkpoint for '{model_type}' could not be checked on this "
            f"site, so the run is refused rather than started: the availability "
            f"probe returned {type(status).__name__}, not a result"
        )
    err = status.get("probe_error")
    if err:
        return (
            f"the base checkpoint for '{model_type}' could not be checked on this "
            f"site, so the run is refused rather than started ({err})"
        )
    parts = status.get("parts")
    if not isinstance(parts, dict) or not parts:
        return (
            f"the base checkpoint for '{model_type}' could not be checked on this "
            f"site, so the run is refused rather than started: the probe reported "
            f"no parts for it"
        )
    missing = []
    for name, info in sorted(parts.items()):
        if not isinstance(info, dict):
            missing.append(f"{name} (unreadable probe result)")
            continue
        if info.get("cached"):
            continue
        if info.get("reachable") is True:
            continue
        # AN EXPLICIT REASON WINS over anything derived from the fields. A
        # caller that already knows WHY a part is unobtainable -- a probe that
        # timed out, say -- must be able to say so, because the derived text
        # would otherwise describe the wrong thing entirely: a part with no URL
        # reads as "micro-sam publishes no URL for it", which sends a reader to
        # the registry when the real trouble is this site's network.
        reason = info.get("reason")
        if isinstance(reason, str) and reason:
            where = reason
        elif not info.get("url"):
            where = "not cached here and micro-sam publishes no URL for it"
        else:
            where = "not cached here and not reachable from here"
        missing.append(f"{name} ({where})")
    if not missing:
        return None
    return (
        f"'{model_type}' cannot be trained at this site: its base checkpoint is "
        f"unavailable — {', '.join(missing)}. Nothing was started and no GPU was "
        f"claimed. Pick a model type whose weights this site can reach, or ask an "
        f"administrator to mirror this one."
    )


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


def checkpoint_path(
    session_id: str, *, status: Optional[Dict[str, Any]] = None
) -> Path:
    """Servable checkpoint for a finished session, by backend:
    micro-sam's torch_em writes ``<dir>/checkpoints/<name>/best.pt``; cellpose's
    ``train_seg`` (save_path=<dir>, model_name='model') writes ``<dir>/models/model``.

    *status* is passed straight to `session_backend`, so a caller holding the
    record already does not pay for a second read of it. `get_status` calls
    this on every run it reports.
    """
    sdir = session_dir(session_id)
    if session_backend(session_id, status=status) == "cellpose":
        return sdir / "models" / "model"
    return sdir / "checkpoints" / session_id / "best.pt"


#: The mode a run's `status.json` must carry. Named here, beside the writer,
#: because the writer is what a later change would touch -- see the block in
#: `write_status` for why 0o600 and not merely "nothing world-writable".
STATUS_FILE_MODE = 0o600


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


def write_status_best_effort(session_id: str, **fields: Any) -> bool:
    """`write_status` for a write the RUN MUST SURVIVE. True if it landed.

    The record lock introduced two exceptions that never existed before:
    `RunRecordLockTimeout` when the mutex cannot be taken, and
    `RunRecordUnreadable` when the file on disk will not parse. Both are right
    to raise at a write whose whole point is to be durable -- a terminal status,
    a failure record -- because an unlocked merge is what silently erases a
    record.

    They are wrong to raise at a HEARTBEAT. A heartbeat exists only to refresh
    `updated_at` so a long epoch does not trip the stale window; it carries no
    outcome, and the next one is a minute away.

    **What an unhandled heartbeat exception actually does**, stated precisely
    because an earlier version of this docstring overstated it: the heartbeat
    runs on a daemon thread, so the exception ends THAT THREAD and the training
    process carries on. It does not kill the run directly. What it does is stop
    `updated_at` advancing, silently -- and once `STATUS_STALE_SECONDS` passes,
    `get_status` sweeps a LIVE run from TRAINING to STOPPED. So a contended
    lock would make a healthy run report itself dead, which is the same
    stale-window failure the heartbeat exists to prevent, arrived at from the
    other side.

    Progress writes are the same shape: an epoch curve that misses one update
    is repaired by the next, and by the reconciliation at the end.

    So this is for heartbeat and progress ONLY. Terminal writes keep calling
    `write_status` and keep raising. The split is deliberate and is tested both
    ways round, because a catch-all here would hide exactly the failure the
    lock was built to surface.
    """
    try:
        write_status(session_id, **fields)
        return True
    except (RunRecordLockTimeout, RunRecordUnreadable) as exc:
        # Logged with the session id so a run whose timestamps go quiet can be
        # tied to the reason. Not re-raised: see above.
        print(
            f"[model-finetune] run {session_id}: a non-terminal status write was "
            f"skipped ({type(exc).__name__}: {exc}). The run continues; the next "
            "update will carry the state.",
            flush=True,
        )
        return False


def read_status(session_id: str) -> Dict[str, Any]:
    p = _status_path(session_id)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def read_status_strict(session_id: str) -> Dict[str, Any]:
    """The status document, RAISING if it cannot be read.

    :func:`read_status` answers ``{}`` for a status document that is missing or
    will not parse, which is right for a listing or a poll: one damaged run
    must not take out the operation around it, and the next write repairs it.

    It is WRONG for anything that copies a run somewhere permanent. There,
    ``{}`` is indistinguishable from "this run genuinely has no status", so a
    corrupt document would be faithfully preserved as an EMPTY record -- the
    run's history replaced by nothing, with no error anywhere. A backfill has
    one chance to read each run, so it asks the question that can fail.

    Raises ``OSError`` if the document is absent and ``ValueError`` if it will
    not parse.
    """
    p = _status_path(session_id)
    if not p.exists():
        raise OSError(f"no status document for {validate_session_id(session_id)}")
    text = p.read_text()
    try:
        document = json.loads(text)
    except Exception as exc:
        raise ValueError(
            f"the status document for {validate_session_id(session_id)} is not "
            f"readable JSON ({type(exc).__name__}); {len(text)} byte(s) on disk"
        ) from exc
    if not isinstance(document, dict):
        raise ValueError(
            f"the status document for {validate_session_id(session_id)} is a "
            f"{type(document).__name__}, not an object"
        )
    return document


# The mutex a writer holds while merging a record.
#
# ⚠ NOT flock, and not fcntl.lockf -- and the reason is portability across
# mounts, not that they always fail. Two measurements on the same de.NBI NFSv3
# export disagree, and both are reported here because the disagreement is the
# point:
#
#     flock  -> ENOLCK on one mount, SUCCEEDED on the worker pod's mount
#     lockf  -> ENOLCK on one mount, SUCCEEDED on the worker pod's mount
#     link   -> 600/600, and EEXIST when held, on both
#     none   -> 301/600   (the control: the processes genuinely race)
#
# Whether the NFS client was mounted with locking is a property of the pod spec,
# and a Serve replica is a KubeRay pod in another namespace with its own mount
# -- so the replica's answer is not the worker's answer. A lock that RAISES
# ENOLCK on every write would take out every run operation on that replica, and
# which replicas those are is not knowable from here.
#
# os.link() needs no lock manager: creating a unique file and linking it to a
# well-known name is atomic over NFSv3 and fails with EEXIST when somebody holds
# it. It is the classic NFS-safe mutex, it held on both mounts, and it works on
# a local filesystem too, so there is one code path rather than two.
RECORD_LOCK_NAME = ".status.mutex"


# ⚠ ACROSS NODES, the reap's judgement rests on a `stat` another NFS client may
# answer from its attribute cache, and that cache can outlive this window (the
# usual client defaults reach 60 s, twice the 30 s below). So a reaper on one
# node can in principle act on a stale view of a lock held on another. Measured
# only within ONE pod, where a single client's cache is coherent; the
# cross-client case is NOT measured. Accepted rather than fixed: it can only
# bite after a holder has died, which is the case the reap exists for and
# already concedes an overlap. The ordinary acquire path is unaffected --
# `os.link`'s EEXIST is decided on the server, not from any cache.
# How long a mutex may be held before another writer treats it as abandoned.
# A link-based lock has no kernel to release it if the holder dies, so without
# this one crashed writer wedges a run's record forever. Generous relative to a
# merge, which is a small read and a rename: anything near this means the holder
# is gone, not slow.
RECORD_LOCK_STALE_S = 30.0

# How long to keep trying before giving up and raising. Raising is the safe
# direction: a caller that cannot take the mutex must NOT merge anyway, because
# merging unlocked is the data loss this exists to prevent.
RECORD_LOCK_WAIT_S = 60.0


class RunRecordLockTimeout(RuntimeError):
    """Could not take a run's record mutex, so the merge was refused.

    Refusing is the safe direction: merging without the mutex is precisely the
    lost update this lock exists to prevent, so a caller that cannot acquire it
    must not proceed.
    """


class RunRecordUnreadable(RuntimeError):
    """A run's record exists but cannot be parsed, so a merge would destroy it.

    Raised only by `write_status`, deliberately. A READER must stay tolerant --
    `read_status` answers {} so one bad record cannot take the whole history
    down -- but a WRITER that reads {} and merges into it writes the empty dict
    back as the entire record. The tolerant read and the destructive write look
    identical in the code and are opposite in effect, which is why the writer
    now reads the file itself rather than going through `read_status`.
    """


_TERMINAL = ("COMPLETED", "FAILED", "STOPPED")

# A terminal status may only be changed by an equal-or-higher-ranked writer.
# user stop must outrank a late child COMPLETED (the user asked it to stop); the
# child (the training outcome) outranks the supervisor (which only sees the
# subprocess rc) and the entry bare-except (which cannot see the child at all).
_TERMINATED_BY = {"entry": 1, "supervisor": 2, "child": 3, "user_stop": 4}


def _store_age(reference: Path, target: os.stat_result) -> float:
    """Seconds between *target*'s mtime and now, measured by the STORE's clock.

    Never `time.time() - st_mtime`. On the NFS share these replicas use, mtime
    is set by the SERVER, and the server's clock is not this process's.
    Measured on the de.NBI share 2026-11-25: a file stamped with
    `os.utime(path, None)` came back with an mtime **78.6 seconds in the
    future** relative to the caller's `time.time()`.

    That skew is fatal to a staleness test built on the local clock, in both
    directions. Ahead, as measured, every lock looks like it was taken in the
    future, no abandoned lock is EVER reaped, and every later writer waits out
    `RECORD_LOCK_WAIT_S` and raises. Behind, every live lock looks ancient and
    the mutex excludes nothing. Measured on the share before this fix: 18 of 18
    writes lost to `RunRecordLockTimeout`, one reporting the mutex as "held for
    -18.5s", and the lock file left behind.

    *reference* must be a file on the SAME store whose mtime was just stamped,
    so both sides of the subtraction come from one clock and the skew cancels.
    The acquire loop stamps its unique file immediately before each attempt, so
    it already has one.
    """
    return reference.stat().st_mtime - target.st_mtime


def _reap_stale_mutex(
    run_dir: Path, mutex: Path, unique: Path, judged_ino: int
) -> bool:
    """Replace a stale mutex with THIS writer's lock. True if we now hold it.

    Reaping by removing the stale file cannot be made safe. Two designs tried
    it and failed the same way:

    * stat-then-unlink let a writer that judged mutex L1 stale unlink a FRESH
      L2 somebody had linked in the meantime -- unlink acts on the path, not on
      the inode that was judged;
    * stat-then-RENAME has the identical defect, and renaming into a scratch
      name so the inode can be checked afterwards only moves the damage: the
      path is EMPTY between the rename and the restore, and a third writer
      links into that gap while the victim is still inside its critical
      section. Measured, with a 20ms delay on the rename: still 1-4 overlapping
      writers in 2 runs of 4.

    So this never removes the mutex at all. It REPLACES it, with `os.rename`,
    which overwrites atomically -- the path never goes empty, so there is no
    gap to lose a race in. Two things make that safe:

    * **one reaper per inode.** `.reap-<ino>` is claimed with the same atomic
      `os.link`, so of several writers that judged one mutex stale exactly one
      proceeds and the rest back off while the path still holds the stale
      inode, which they will correctly judge stale again. A claim abandoned by
      a dead reaper is itself stale after `RECORD_LOCK_STALE_S`; it is linked
      from *our* freshly stamped file, not from the mutex, so its mtime says
      when it was claimed rather than inheriting the stale inode's.
    * **the inode is re-checked after the claim is won**, so a mutex that was
      replaced between the stat and the claim is left strictly alone.

    What remains, stated rather than papered over: if the presumed-dead holder
    of `judged_ino` releases AND another writer acquires in the window between
    that re-check and the rename, the rename overwrites that writer's lock.
    That requires the holder judged dead to be alive after all, which is the
    stale timeout's accepted trade-off, not a race this function adds.
    """
    claim = run_dir / f".reap-{judged_ino}"
    os.utime(unique, None)
    try:
        os.link(unique, claim)
    except FileExistsError:
        try:
            if _store_age(unique, claim.stat()) > RECORD_LOCK_STALE_S:
                claim.unlink()  # the reaper that claimed it died mid-reap
        except (FileNotFoundError, OSError):
            pass
        return False
    except OSError:
        return False
    try:
        try:
            if mutex.stat().st_ino != judged_ino:
                return False  # already replaced; not ours to touch
        except FileNotFoundError:
            return False
        spare = run_dir / f".mutex-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            os.link(unique, spare)
        except OSError:
            return False
        os.utime(unique, None)  # the lock's mtime must say when it was TAKEN
        try:
            os.rename(spare, mutex)
        except OSError:
            try:
                spare.unlink()
            except FileNotFoundError:
                pass
            return False
        return True
    finally:
        try:
            claim.unlink()
        except FileNotFoundError:
            pass


@contextlib.contextmanager
def _record_mutex(run_dir: Path, session_id: str):
    """Hold the run's mutex for the duration of the block.

    `os.link` of a unique file onto a well-known name: atomic on NFSv3, where
    unlike flock and lockf, it needs no lock manager -- which one mount of this
    share does not offer. See the note above `RECORD_LOCK_NAME`.

    An abandoned mutex older than `RECORD_LOCK_STALE_S` is reaped and the
    acquisition retried, because nothing releases a link-based lock when its
    holder dies.

    TWO things about that path are load-bearing, and the first version of this
    function got both wrong:

    * **The unique file is stamped immediately before `os.link`, not when it is
      created.** `os.link` gives the mutex the unique file's mtime, so stamping
      at queue time meant a writer that waited N seconds acquired a mutex that
      already looked N seconds old. After a holder dies every queued writer
      waits about `RECORD_LOCK_STALE_S`, so the winner linked an
      already-stale mutex and the next waiter reaped it WHILE IT WAS HELD.
      Measured on that version: up to six writers inside the critical section
      at once and a write lost in four trials of five -- the lock broken
      exactly in the case the reap exists for.
    * **Reaping REPLACES the mutex, it never removes it.** Removal cannot be
      made safe here: unlink and rename both act on the PATH, not on the inode
      that was judged stale, so a writer holding an old verdict destroys a
      fresh lock somebody linked in the meantime -- and renaming into a scratch
      name to check the inode afterwards only moves the damage, because the
      path is EMPTY between the rename and the restore and a third writer links
      into that gap. `_reap_stale_mutex` instead overwrites the stale inode
      with this writer's own, atomically, so the path is never empty; see its
      docstring for the per-inode arbitration that keeps two reapers apart and
      for what the stale timeout still concedes.
    * **The release deletes the mutex only while it is still ours**, comparing
      `st_ino` against the holder's own unique link. A holder that overran the
      window has already been reaped, and unlinking on the way out would delete
      its SUCCESSOR's mutex and admit a third writer.

      Measured on this machine, 6 writers against a dead holder's lock, three
      trials per run: stat-then-unlink and stat-then-rename both overlap (4 runs
      of 4 red, up to 4 writers at once); rename-then-restore, 2 runs of 4 red;
      replace-in-place, 10 runs of 10 green with 0 overlap events. Two earlier
      versions of this docstring claimed guarantees the code did not have, so
      those numbers are what this one rests on.
    """
    mutex = run_dir / RECORD_LOCK_NAME
    unique = run_dir / f".mutex-{os.getpid()}-{uuid.uuid4().hex}"
    unique.write_text(str(time.time()))
    deadline = time.monotonic() + RECORD_LOCK_WAIT_S
    try:
        while True:
            try:
                # STAMP IMMEDIATELY BEFORE LINKING. The mutex inherits this
                # mtime, and it must say when the lock was TAKEN, not when this
                # writer joined the queue -- see the docstring.
                os.utime(unique, None)
                os.link(unique, mutex)
                break
            except FileExistsError:
                try:
                    judged = mutex.stat()
                except FileNotFoundError:
                    continue  # released between the link and the stat
                age = _store_age(unique, judged)
                if age > RECORD_LOCK_STALE_S:
                    if _reap_stale_mutex(run_dir, mutex, unique, judged.st_ino):
                        break  # the reap replaced it with OUR lock
                    time.sleep(0.01)
                    continue
                if time.monotonic() > deadline:
                    raise RunRecordLockTimeout(
                        f"could not take the record mutex for run {session_id} "
                        f"within {RECORD_LOCK_WAIT_S:.0f}s ({mutex} held for "
                        f"{age:.1f}s). Refusing to merge without it, because an "
                        "unlocked merge is what silently erases a record."
                    )
                time.sleep(0.01)
        try:
            yield
        finally:
            # Unlink the mutex ONLY while it is still ours. Holding it means
            # `mutex` and `unique` are links to one inode, so comparing them
            # identifies it. A holder that overran the staleness window has
            # already been reaped and somebody else may hold the path -- that
            # overlap is the accepted cost of having a timeout at all, but
            # deleting the NEXT holder's mutex on the way out would let a THIRD
            # writer in, and that part is avoidable.
            try:
                if os.stat(mutex).st_ino == os.stat(unique).st_ino:
                    mutex.unlink()
            except FileNotFoundError:
                pass
    finally:
        try:
            unique.unlink()
        except FileNotFoundError:
            pass


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

    # EXCLUSIVE for the whole read-merge-replace. A private scratch file stops a
    # TORN document being installed; it does nothing about a LOST UPDATE, which
    # is the other half and the one that erases a user's stop.
    #
    # Measured on the scratch-file-only version: five writers plus one user-stop,
    # the stop erased and the record back to TRAINING on the first trial, and
    # every writer's last write lost in 20 of 20 trials. A read that happened
    # before another writer's replace writes its stale copy back over it, and the
    # terminal-status precedence below cannot refuse an incumbent it never saw.
    #
    # The run directory is the only place guaranteed to be on the same
    # filesystem as the record, which link() requires.
    #
    # NOTE ON CLOCK SKEW, unmeasured: staleness compares the SERVER-set mtime of
    # the mutex with this client's time.time(). On the NFS share those are
    # different machines, so skew shifts the window by however far the clocks
    # differ. A 30s window tolerates a few seconds of skew; it would not
    # tolerate minutes. Not measured on this cluster.
    with _record_mutex(p.parent, session_id):
        return _write_status_locked(p, session_id, fields)


def _write_status_locked(p, session_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """The merge itself. Call ONLY with the record lock held.

    Split out so the lock scope is one `with` block that a reader can see whole,
    rather than an early `return` inside it leaving the reader to check whether
    every path is covered.
    """
    # An UNREADABLE record is not an empty one. `read_status` answers {} for
    # both "no file yet" and "the file would not parse", which is right for a
    # reader -- a history page should not break on one bad record -- and
    # catastrophic here: starting a merge from {} writes the empty dict back as
    # the whole record, so status, owner, created_at and the terminated_by
    # provenance are all discarded in favour of whatever this one call is
    # setting. Measured: six concurrent writers reduced a record to their own
    # fields with the pre-existing `status` key gone, and the result was valid
    # JSON, so nothing downstream could tell.
    if p.exists():
        try:
            cur = json.loads(p.read_text())
        except Exception as exc:
            raise RunRecordUnreadable(
                f"the record for run {session_id} exists but does not parse "
                f"({exc}); refusing to overwrite it with a fresh one, which "
                "would discard whatever it still holds."
            ) from exc
        if not isinstance(cur, dict):
            raise RunRecordUnreadable(
                f"the record for run {session_id} is a {type(cur).__name__}, "
                "not an object; refusing to overwrite it."
            )
    else:
        cur = {}

    cur_status, new_status = cur.get("status"), fields.get("status")
    if cur_status in _TERMINAL and new_status is not None and new_status != cur_status:
        cur_prec = _TERMINATED_BY.get(cur.get("terminated_by"), 0)
        new_prec = _TERMINATED_BY.get(fields.get("terminated_by"), 0)
        if new_prec <= cur_prec:
            return cur
    cur.update(fields)
    cur["updated_at"] = time.time()

    # A scratch file UNIQUE to this writer. The name used to be derived from the
    # target -- `status.json.tmp` -- which every concurrent writer therefore
    # shared, and that is two bugs in one:
    #
    #   * `write_text` truncates and writes, which is not atomic, so one writer
    #     could `os.replace` a document another writer was halfway through
    #     writing. A TORN record then becomes the real one.
    #   * the first `os.replace` consumes the shared file, so the second raised
    #     FileNotFoundError. Measured at 50 of 240 writes across six processes.
    #
    # This is not a hypothetical second replica: the entry replica, both runtime
    # replicas and the training subprocess all write this file during a run,
    # from different processes.
    #
    # `mkstemp` in the SAME directory, because `os.replace` is only atomic
    # within one filesystem and the run directory is the only place guaranteed
    # to be on the same one as the target.
    fd, tmp_name = tempfile.mkstemp(dir=str(p.parent), prefix=".status-", suffix=".tmp")
    try:
        # THE MODE IS 0o600 AND IT IS REQUESTED, NOT INHERITED.
        #
        # This file carries the run's `owner`, and the owner field is what every
        # permission check on the run is matched against, so the mode is part of
        # the access story rather than a detail of how the write is done. A
        # world-writable status file puts "whose run is this" within reach of
        # anything else running in the container.
        #
        # `mkstemp` already creates at 0o600, so this looks redundant. It is
        # not, for two reasons. It makes the restriction a DECISION that
        # survives a rewrite of this function -- `Path.write_text` creates at
        # `0o666 & ~umask`, and whoever rewrites this reads this comment, not
        # the test. And `mkstemp`'s 0o600 is NOT umask-proof: it calls
        # `os.open(..., 0o600)` and the umask masks that too; it merely looks
        # immune because an ordinary umask never clears the OWNER bits. An
        # explicit `fchmod` is what actually makes the mode independent of the
        # umask the process happens to carry.
        #
        # `os.replace` installs this inode over the target, so the mode is
        # re-established on every write rather than preserved from the first
        # one. Pinned by apps/model-finetune/test_status_file_mode.py; the
        # legacy 0o777 files already on the production store are a separate
        # matter and no code change reaches them.
        os.fchmod(fd, STATUS_FILE_MODE)
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(cur))
        os.replace(tmp_name, p)
    except BaseException:
        # Never leave scratch files behind in a directory the listing walks.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
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


# The artifact the app keeps run history in.
#
# FIXED HERE, and deliberately NOT a parameter of any method. The app writes
# this artifact with its OWN credential, so a caller who could name the
# artifact could make the app write anywhere that credential reaches. There is
# no allow-list to get wrong because there is nothing to validate: the id is
# not an input. Measured on creation 2026-11-25: standalone (no `parent_id`,
# which is what makes "private" mean anything -- a parent decides a child's
# visibility), permissions `{<app>: "*"}` with no `"*"` or `"@"` key, and an
# anonymous caller refused on RPC `read`, RPC `list_files` and HTTP (403).
#
# A *dataset* write on a caller's behalf is the opposite case and keeps its own
# rule: it needs a caller token, because it acts as the caller. Stated here
# because the two rules sit next to each other in this app and a reader should
# not have to infer which applies.
RUNS_ARTIFACT_ID = "bioimage-io/model-finetune-runs"

# One run's entry in that artifact, under the DEPLOYMENT it belongs to.
#
# The application id in that path is load-bearing and was missing from the
# first version of this, which a review caught before it could do any damage.
# One artifact is shared by every deployment of this app, and a worker runs
# several at once: measured on one worker on 2026-11-25, `model-finetune`
# (production) held 261 runs, `model-finetune-soup` 37, and the throwaways
# `model-finetune-dev`, `model-finetune-dev-metrics`,
# `model-finetune-dev0181` and `mf-verify` held 19, 6, 2 and 1. Sharing one
# flat `runs/` would have merged every throwaway's test runs into the
# production history, and the issue's own "the count before equals the count
# after" check could then only pass by being counted on the wrong surface.
#
# The sharing unit is therefore the application id: replicas of ONE deployment
# share a history, which is exactly the split that was reported, and two
# deployments never see each other's.
RUN_RECORD_PATH_TEMPLATE = "runs/{application_id}/{session_id}.json"

#: The environment variable BioEngine sets on every app replica
#: (`bioengine/apps/builder.py` sets it unconditionally at deploy time).
APPLICATION_ID_ENV = "BIOENGINE_APPLICATION_ID"

_APPLICATION_ID_PATTERN = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def record_store_id() -> Optional[str]:
    """Which deployment's run history this replica belongs to, or None.

    None means sharing is OFF for this process, and that is deliberate rather
    than a degraded guess: a replica that cannot tell which deployment it is
    must not write into another deployment's history, and a fallback bucket
    shared by every such replica is precisely that. The local listing still
    works; the shared half is simply absent and says so.

    The value is validated even though it comes from the environment and not
    from a caller. It becomes a path component inside the artifact, and
    "it cannot be hostile" is an argument about today's deployment path, not
    about the next one.
    """
    raw = os.environ.get(APPLICATION_ID_ENV)
    if not isinstance(raw, str) or not _APPLICATION_ID_PATTERN.match(raw):
        return None
    return raw


def run_record_dir(application_id: str) -> str:
    """The directory holding one deployment's records."""
    if not _APPLICATION_ID_PATTERN.match(application_id or ""):
        raise InvalidSessionId(
            f"not a usable application id (length {len(application_id or '')})"
        )
    return RUN_RECORD_PATH_TEMPLATE.format(
        application_id=application_id, session_id="x"
    ).rsplit("/", 1)[0]


RUN_RECORD_VERSION = 1


def redact_urls(value: Any) -> Any:
    """Strip the query string from any http(s) URL, recursively.

    A run's params carry the image and label references it was given, and those
    can be PRESIGNED: the query string is the credential. A run record is meant
    to outlive the worker and be readable by whoever opens the run history, so
    it must not be the place a signature gets archived. The path is kept
    because it identifies the object; only what authorises access is dropped.

    Nothing else is filtered. The record is otherwise verbatim -- see
    :func:`run_record`.
    """
    if isinstance(value, str):
        if value.startswith(("http://", "https://")) and "?" in value:
            return value.split("?", 1)[0] + "?<redacted>"
        return value
    if isinstance(value, dict):
        return {k: redact_urls(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_urls(v) for v in value]
    return value


def run_record(
    session_id: str, status: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """One run, as a document that can outlive the worker that produced it.

    ``config`` is the training params and ``status`` is the status document,
    both VERBATIM apart from two subtractions. That is deliberate and is the
    whole design: per-epoch metrics are written into the status document by the
    training workers, so a record that enumerated known fields would have to be
    edited every time a worker learns to report something new -- by whoever
    happened to add it, in a second place. Copying the document means a new
    metric appears here the moment it appears there, with nothing to keep in
    step. The cost, stated so a reader is not surprised: a consumer sees
    whatever the worker wrote, including keys it does not know. Tolerating
    unknown keys is a better failure than a schema that silently drops a
    metric.

    The one subtraction is :func:`redact_urls`: a presigned URL loses its query
    string, because the record outlives the handle and the query string IS the
    credential.

    **The recorded owner STAYS, and that is a correction to an earlier plan of
    mine rather than an oversight.** I intended to strip
    :data:`PRIVATE_STATUS_FIELDS` here. Measured against the landed per-caller
    listing, that cannot work: :func:`list_sessions` decides visibility by
    comparing the caller to the run's recorded owner, so a record arriving from
    ANOTHER replica with no owner in it is unattributable, and the listing
    would either show it to everybody or to nobody. Two replicas answering the
    same caller differently is the exact failure this record exists to fix, so
    the owner has to travel with the record.

    What keeps the owner out of a caller's hands is therefore not this
    function. It is two things, both of which can be checked:

    * the record lives in a PRIVATE, standalone artifact
      (:data:`RUNS_ARTIFACT_ID`) that only this app's credential can read --
      measured on creation: anonymous `read`, anonymous `list_files` and an
      anonymous HTTP GET are all refused;
    * everything returned to a caller goes through :func:`redact_private`,
      which iterates :data:`PRIVATE_STATUS_FIELDS` so that adding a field there
      is enough. That is the single place the strip happens, rather than two
      places that can drift.

    *status* is accepted so a caller that has just written the document does
    not read it back; omitted, it is read from disk. An unknown session gives a
    record rather than raising, because a run history asking about a vanished
    id wants an answer, and an absent config is recorded as ``None`` rather
    than ``{}``, which would read as "trained with no settings".
    """
    sid = validate_session_id(session_id)
    document = read_status(sid) if status is None else status
    try:
        params = read_training_params(sid)
    except (OSError, ValueError):
        params = None
    # THE ALLOW-LIST, applied HERE as well as on the way out. A review found
    # this missing and measured what it cost: the local listing filters the
    # params through RUN_CONFIG_FIELDS, so one run read locally showed
    # {model_type, n_epochs} while the same run read through a record showed
    # train_images and init_checkpoint -- worker-local paths -- plus any field
    # a later version adds to the params. The allow-list held on one replica
    # and was bypassed on the other.
    #
    # Filtering at the WRITE means an excluded field never reaches the artifact
    # at all, which matters because a published record outlives the code that
    # wrote it. `status_from_record` filters again at the READ, so a record
    # already written by an unfiltered version is contained too: one hop is not
    # enough when the hops can be a year apart.
    return {
        "record_version": RUN_RECORD_VERSION,
        "session_id": sid,
        "config": (
            redact_urls(config_for_display(params)) if params is not None else None
        ),
        "status": redact_urls(document or {}),
    }


def run_record_path(session_id: str, application_id: str) -> str:
    """Where one run's record lives inside the runs artifact.

    Validated, not formatted blindly: both parts become path components, so a
    value that is not a session id must be impossible rather than unlikely.
    `validate_session_id` already refuses traversal and separators, and is the
    same check every on-disk path goes through -- reused rather than restated,
    because a second copy of the rule is how the two drift.
    """
    return RUN_RECORD_PATH_TEMPLATE.format(
        application_id=run_record_dir(application_id).rsplit("/", 1)[-1],
        session_id=validate_session_id(session_id),
    )


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


# The dataset a run was trained on, recorded at start_training. The field a
# RUN carries, not the artifact a run is STORED in — those are different things
# and the names are close enough to confuse.
#
# Present only on runs started from version 0.30.42 onwards. Everything older
# has no dataset, which is not a defect to repair: nothing recorded one, so
# inferring it now would be the same guess that was refused for `owner`. A run
# with no dataset simply has no co-managers and falls back to owner-or-admin.
DATASET_FIELD = "dataset_artifact_id"


# The ACL value an owner or manager of a dataset holds, as written by the
# annotation broker's `build_acl_permissions`: owner and managers get "*",
# annotators get "r+", and a PUBLIC dataset additionally gets the wildcard KEY
# "*" mapped to "r+".
#
# ⚠ The manager VALUE and the wildcard KEY are the same string, "*". They mean
# opposite things, and conflating them hands co-manager rights to every caller
# alive on any public dataset:
#
#     perms.get("*")            -> the PUBLIC grant, "r+"      NOT a manager test
#     perms.get(caller) == "*"  -> this caller is a manager     the real test
#
# So the caller's own key is looked up and the wildcard key is never consulted.
# This is the same narrowing `caller_is_workspace_admin` makes for workspace
# grants, for the same reason, and a mutant that reads the wildcard turns the
# public-dataset test red.
#
# Annotators hold "r+" and are deliberately NOT co-managers: a run's
# collaborators are its dataset's MANAGERS, and the page draws the same line
# (`role === 'owner' || role === 'manager'`). "r+" lets someone add annotations
# to a dataset; it is not authority over models trained from it. Requiring
# exactly "*" is what enforces that, so there is no second constant listing the
# permissions that do NOT confer management -- an enumeration of the rejected
# values would be a second place for the rule to live, and the wrong one to
# trust, since a new Hypha permission string absent from it would read as a
# manager.
DATASET_MANAGER_PERMISSION = "*"


def caller_manages_dataset(permissions: Any, context: Any) -> bool:
    """True iff *context*'s caller owns or manages the artifact *permissions* came from.

    *permissions* is a Hypha artifact's ``config.permissions`` mapping, already
    fetched by the caller of this function -- kept separate from the fetch so
    the rule is synchronous, pure, and testable without a server.

    Matches the caller's id first and then their email, mirroring the broker's
    own ``_acl_key``, which prefers id over email when writing the ACL. Checking
    only one of the two would silently miss datasets whose ACL was keyed by the
    other.

    Never consults the wildcard key -- see the note above the constant.
    """
    if not isinstance(permissions, dict) or not isinstance(context, dict):
        return False
    user = context.get("user")
    if not isinstance(user, dict):
        return False
    for key in (user.get("id"), user.get("email")):
        if not isinstance(key, str) or not key or key == DATASET_MANAGER_PERMISSION:
            # A caller whose own id is literally "*" would otherwise read the
            # public grant as its own. No such id should exist; refusing costs
            # nothing and removes the question.
            continue
        if permissions.get(key) == DATASET_MANAGER_PERMISSION:
            return True
    return False


def _owner_or_comanager(
    status: Dict[str, Any], context: Any, writable_datasets: Any
) -> bool:
    """The two legs SHARED by seeing and resuming: owner match, or co-manager.

    Takes the run's STATUS rather than its id, so a caller that has already
    read the record does not read it again. The listing reads every run once
    through `get_status` and then asks this question about it; resolving the id
    here instead meant the same file was opened three times per listed run --
    261 runs became 783 reads on the production store. Passing the status is
    not the same as passing the ANSWER: the rule still lives here and in one
    place.

    False means "neither leg applies", which leaves the caller's own
    administrator rule to decide. Those rules are NOT the same for seeing and
    resuming, which is why this stops here rather than answering.

    *writable_datasets* is computed SERVER-SIDE from the caller's injected
    context before this is reached; it is not a parameter a caller can supply.
    It is passed in rather than resolved here because resolving it is an async
    artifact-manager round trip while this is synchronous and runs once per run.
    """
    owner = status.get("owner")
    has_owner = isinstance(owner, str) and bool(owner)
    if has_owner:
        caller = caller_identity(context).get("owner")
        if caller is not None and caller == owner:
            return True
    # THE CO-MANAGER LEG REQUIRES THE RUN TO HAVE AN OWNER, and without this
    # line it did not. A run with no recorded owner on a dataset the caller
    # manages returned True here, which made an OWNERLESS run visible to a
    # co-manager and resumable by one.
    #
    # Two things said that is wrong, and they agreed with each other before the
    # code did. The lead's ruling on this row: the widening applies to OWNED
    # runs only -- a co-manager sees the runs of other MANAGERS on that
    # dataset -- and ownerless runs stay admin-only. And the shipped README
    # already said "resuming an ownerless run requires `rw`/admin on
    # `bioimage-io`, because with no recorded owner there is nobody for the
    # ownership check to match you against".
    #
    # So the rule is not "a manager of the dataset may act on anything filed
    # under it": it is "a manager may act on a COLLEAGUE's run", and an
    # ownerless run has no colleague. The administrator leg in each caller --
    # `caller_may_see_run`, `caller_may_resume_run` -- is what reaches those,
    # and it is deliberately not widened here.
    if not has_owner:
        return False
    dataset = status.get(DATASET_FIELD)
    if isinstance(dataset, str) and dataset and dataset in writable_datasets:
        return True
    return False


def caller_may_see_run(
    session_id: str,
    context: Any,
    *,
    writable_datasets: Any = frozenset(),
    status: Any = None,
) -> bool:
    """True iff this run belongs in the caller's listing.

    The caller owns it, or manages its dataset, or it has no owner at all and
    the caller administers ``bioimage-io``.

    **An administrator does NOT see other people's owned runs**, and that is not
    an oversight to tidy up later. Admin rights grant the unattributable rows,
    not everybody's history. Stopping somebody else's run is a different rule --
    see `caller_may_mutate` -- and the two are deliberately not the same width.
    A test pins this exact asymmetry; if it ever becomes wrong, it should change
    because somebody decided to widen a privacy boundary, not because a
    refactor unified two rules that only looked alike.

    *status* lets a caller that has ALREADY read this run's record hand it over
    instead of having it read again -- the listing does, once per run. It is the
    record, never the verdict: the rule below is the only one, and supplying a
    status cannot change what it decides.

    Fails closed: an unreadable context yields no identity and no authority, and
    *writable_datasets* defaults to empty.
    """
    st = read_status(session_id) if status is None else status
    if _owner_or_comanager(st, context, writable_datasets):
        return True
    owner = st.get("owner")
    if isinstance(owner, str) and owner:
        # Somebody else's, and attributable. Not an administrator's to browse.
        return False
    return caller_is_workspace_admin(context)


def caller_may_resume_run(
    session_id: str, context: Any, *, writable_datasets: Any = frozenset()
) -> bool:
    """True iff the caller may continue training from this run's checkpoint.

    The caller owns it, or manages its dataset, or administers ``bioimage-io``.

    Wider than `caller_may_see_run` on the administrator leg ONLY, and that
    matches what resume did before co-managers existed: it was gated by
    `caller_may_mutate`, under which an administrator could resume any run. This
    keeps that, adds the dataset manager, and changes nothing else.

    So an administrator can resume a run it cannot list. That reads odd and it
    is the PRE-EXISTING shape, preserved on purpose: narrowing it here would be
    a second unrequested change riding along with this one.
    """
    if _owner_or_comanager(read_status(session_id), context, writable_datasets):
        return True
    return caller_is_workspace_admin(context)


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


def run_names_on_disk() -> "tuple[List[str], List[str]]":
    """``(run names, skipped names)`` under the sessions root.

    THE ONLY place that iterates the sessions root. Both the listing and the
    dataset scan need the same walk with the same "is this actually a run"
    rules, and a second copy of it is a second place for those rules to drift.
    The containment guard counts the root-resolving call sites and refused this
    when it started life as a duplicate loop, which was the right answer.

    (Mind the prose here: that guard greps SOURCE LINES, so naming the helper it
    counts -- even inside a docstring -- is itself a third match and fails it.
    Measured, after exactly that happened.)

    A name that is not a session id goes to *skipped* rather than raising: the
    rule is validate what a CALLER hands us, tolerate what we FIND. A directory
    this server could not have created does not belong in the history, but
    neither does it get to take the history down for everybody.
    """
    root = sessions_root()
    names: List[str] = []
    skipped: List[str] = []
    for d in sorted(root.iterdir()) if root.exists() else []:
        if not (d.is_dir() and (d / "status.json").exists()):
            continue
        if not SESSION_ID_PATTERN.match(d.name):
            skipped.append(d.name)
            continue
        names.append(d.name)
    return names, skipped


# What an operator is shown about somebody else's run.
#
# An ALLOW-LIST, not a list of things to strip. `redact_private` removes `owner`
# and `owner_anonymous`, and an earlier version of this view used it and then
# claimed "nothing about whose it is". That was false: the record also carries
# `dataset_artifact_id`, and a dataset the caller manages lives in the caller's
# own workspace, so its id names them as surely as the owner field does. The
# free-text `label` is a second such channel, and the next field somebody adds
# would be a third. A strip-list has to be right about every field that exists
# now and every field added later; an allow-list only has to be right about what
# an operator needs.
#
# What an operator needs is: which run, what state, how long, and on which
# backend -- enough to decide whether to stop it.
OPERATOR_FIELDS = (
    "session_id",  # how an operator stops it; the mapping key is not enough
    #               for a caller holding one record on its own
    "status",
    "start_time",
    "updated_at",
    "end_time",
    "elapsed_s",
    "model_type",
    "backend",
    "n_epochs",
    "n_epochs_completed",
    "terminated_by",
)


def quiet_seconds(record: Dict[str, Any]) -> float:
    """How long since *record* was last written, by its own ``updated_at``.

    Returns infinity when there is no usable timestamp, so a record with none
    reads as maximally stale -- the safe direction for every caller here, all
    of which use this to decide whether something has gone quiet.

    It exists because the arithmetic was written inline in three places as
    ``time.time() - record.get("updated_at", 0)``, which raises TypeError on a
    non-numeric value and takes the whole read down with it. `get_status` is on
    every run-reading path, so one malformed record would have made the run
    listing, the status call and the operator view all raise rather than
    skipping the bad record. Booleans are rejected explicitly: `True` is an
    `int` to `isinstance` and would otherwise be read as a timestamp of 1.
    """
    stamped = record.get("updated_at")
    if isinstance(stamped, bool) or not isinstance(stamped, (int, float)):
        return float("inf")
    return time.time() - stamped


def operator_view(status: Dict[str, Any]) -> Dict[str, Any]:
    """Project a run record down to what an operator may see, plus staleness.

    `stale` is the second half of this, and it exists because a non-terminal
    status is NOT the same as a running process. `get_status` sweeps a run to
    STOPPED when it has gone quiet, but only from PREPARING or TRAINING --
    QUEUED is deliberately exempt, so a run that was QUEUED when its replica
    restarted stays QUEUED for ever. That is exactly the outlived-its-process
    case this view exists for, and without a staleness mark an operator reading
    the list sees a busy worker that is in fact idle: the inverse of the
    blindness being fixed. A record whose status cannot be read at all is in
    the same position.

    Marked rather than dropped. A stale QUEUED record may be a genuinely stuck
    run that somebody needs to see, and hiding it would reproduce the original
    complaint in a new place.
    """
    out = {k: status[k] for k in OPERATOR_FIELDS if k in status}
    state = status.get("status")
    # A non-numeric `updated_at` must not take the whole operator listing down
    # with a TypeError. This view exists to stay readable when things are
    # wrong, and the listing it sits in already promises that a junk directory
    # does not break it. An unusable timestamp is treated as no timestamp,
    # which reads as stale -- the safe direction for an operator.
    quiet_for = quiet_seconds(status)
    out["stale"] = bool(
        state is None or state == "UNKNOWN" or quiet_for > STATUS_STALE_SECONDS
    )
    if out["stale"]:
        out["stale_for_s"] = None if quiet_for == float("inf") else round(quiet_for, 1)
    return out


def non_terminal_runs() -> Dict[str, Dict[str, Any]]:
    """Every run on this worker that has NOT finished, keyed by session id.

    The OPERATOR question -- "what is this worker currently working on" -- which
    the per-caller listing deliberately cannot answer. A run is returned to its
    owner, to a manager of its dataset, or (ownerless only) to an administrator;
    so a run started by a token-less caller whose connection has since closed is
    visible to nobody, while still occupying the worker. This reports those by
    LIFECYCLE rather than by ownership.

    **Each record is projected through `operator_view`**, an allow-list of the
    fields an operator needs, which is the whole point of it being a separate
    function rather than a widened listing. Stripping the owner fields is not
    enough on its own -- `dataset_artifact_id` names a workspace, and so names
    its owner -- which is why this projects rather than redacts. Widening
    `list_sessions` instead would hand every owned run to every administrator
    and undo a privacy rule that has a test pinning it.

    Read from the DISK, not from any in-memory task registry. A replica's
    `_training_tasks` dict is empty after a restart, which would miss precisely
    the run that outlived its process -- the case most worth catching.

    Uses the one existing walk of the sessions root, so it adds no new way to
    enumerate runs and no new I/O pattern.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for name in run_names_on_disk()[0]:
        try:
            status = get_status(name, include_private=True)
        except InvalidSessionId:
            continue
        if status.get("status") not in _TERMINAL:
            out[name] = operator_view(status)
    return out


def require_caller_may_resume_run(
    session_id: str, context: Any, *, writable_datasets: Any = frozenset()
) -> None:
    """:func:`caller_may_resume_run`, or raise :class:`NotTheOwner`.

    Raises the SAME class as the mutate gate, deliberately. A caller refused a
    resume and a caller refused an export have learned exactly the same thing --
    "not yours" -- and giving the two refusals different types would let a
    caller tell which of the two rules it fell foul of, which is a readout of
    whether they are a co-manager of a dataset they cannot otherwise see.

    Like its sibling, the message says nothing about whether the run exists, who
    owns it, or which dataset it belongs to.
    """
    if not caller_may_resume_run(
        session_id, context, writable_datasets=writable_datasets
    ):
        raise NotTheOwner(
            "this training run belongs to another caller; only its owner, a "
            "manager of its dataset, or a workspace administrator can use it"
        )


class InvalidExportId(ValueError):
    """An export_id the server could not have issued."""


class InvalidPackageMember(ValueError):
    """A package file name that is not one this export announced."""


#: What `export_model` mints is `uuid4().hex`. The check is deliberately a
#: little wider than that so an id minted by an older version still resolves,
#: but narrow enough that it cannot be a path: no separators, no `..`.
EXPORT_ID_PATTERN = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def validate_export_id(export_id: Any) -> str:
    """The export_id, or raise.

    The id becomes a file name under the exports root, exactly as a session id
    becomes a directory name under the sessions root -- and that one was a
    traversal until it was checked. The value is NOT echoed in full: it is
    caller-controlled and this message reaches logs.
    """
    if not isinstance(export_id, str):
        raise InvalidExportId(
            f"export_id must be a string, got {type(export_id).__name__}"
        )
    if ".." in export_id or not EXPORT_ID_PATTERN.match(export_id):
        raise InvalidExportId(
            f"export_id is not a valid export identifier "
            f"(length {len(export_id)}, starts {export_id[:12]!r})"
        )
    return export_id


def package_member_path(package_dir: Any, name: Any, *, announced: Any) -> Path:
    """The path of one member of a built package, or raise.

    TWO independent layers, deliberately not one, and the same shape as
    `session_dir`:

    1. the name must be one the export's own record ANNOUNCED. That is an
       allow-list of the files this package actually has, which is a stronger
       statement than any pattern over names;
    2. the resolved path must still be inside the resolved package directory.
       Not redundant with layer 1 -- it is what still holds if the allow-list
       is ever loosened, and it also catches a symlink inside the package that
       points out of it, which layer 1 cannot see.

    The hole this closes, measured on the method before it existed: the caller
    supplies the file names, they were joined to the package directory
    unchecked, and `push_export(my_export, {"../../.bioengine/sessions/<other
    run>/best.pt": "https://attacker/put"})` streamed ANOTHER run's checkpoint
    to a URL of the caller's choosing. An absolute name did the same, because
    joining an absolute path discards the directory it was joined to. Binding
    the handle to its creator without bounding the names would only have moved
    the hole from "somebody else's export id" to "your own".
    """
    names = set()
    for entry in announced or []:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            names.add(entry["name"])
        elif isinstance(entry, str):
            names.add(entry)
    if not isinstance(name, str) or name not in names:
        raise InvalidPackageMember(
            f"'{name if isinstance(name, str) else type(name).__name__}' is not "
            "a file of this export; only the names it reported can be pushed"
        )
    root = Path(package_dir).resolve()
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise InvalidPackageMember(
            "a package file name resolves outside the package directory"
        )
    return candidate


def caller_may_use_export(export_id: str, context: Any) -> bool:
    """True iff *context* identifies the export's creator, or a ``bioimage-io``
    admin.

    An export id is a handle to a run's WEIGHTS: the READY record carries a
    presigned `download_url` with six hours of life in it, plus the name of the
    resumable checkpoint inside the package, and `push_export` streams the whole
    package to any presigned URL a caller supplies. `export_model` is already
    owner-or-admin, so this closes the gap between "who may start an export" and
    "who may use its handle" -- which were not the same question.

    Deliberately the SAME shape as :func:`caller_may_mutate`, including the two
    properties that one learned the hard way:

    * the admin test is reached for an OWNED export too, not only an ownerless
      one, so an administrator is not refused their own stated authority;
    * an export with no usable creator is admin-only. That covers an export
      minted before this field existed, which cannot be attributed to anybody.
      Those expire within hours, so the window in which a non-admin is refused
      their own in-flight export is bounded and known rather than permanent.

    Reads what is on disk through :func:`read_export` and creates nothing, so
    asking about an export that does not exist is safe and answers False for a
    non-admin -- which is what lets the refusal be identical for "not yours"
    and "no such export". (An earlier version of this note said
    :func:`get_export` WRITES the swept record; it does not. It changes its
    own in-memory copy and returns it. The gate still runs first, because the
    refusal must not depend on what the record says.)
    """
    owner = read_export(export_id).get("_owner")
    if isinstance(owner, str) and owner:
        caller = caller_identity(context).get("owner")
        if caller is not None and caller == owner:
            return True
        return caller_is_workspace_admin(context)
    return caller_is_workspace_admin(context)


def require_caller_may_use_export(export_id: str, context: Any) -> None:
    """:func:`caller_may_use_export`, or raise :class:`NotTheOwner`.

    The message says nothing about whether the export exists, who created it,
    or whether the caller merely lacks admin. That is the point: an id is a
    bearer capability, and a refusal that distinguished "not yours" from "no
    such export" would turn this check into an oracle for guessing ids.
    """
    if not caller_may_use_export(export_id, context):
        raise NotTheOwner(
            "this export belongs to another caller; only the identity that "
            "created it or a workspace administrator can use it"
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


def shape_status(
    st: Dict[str, Any],
    session_id: str,
    *,
    config: Dict[str, Any],
    checkpoint_available: bool,
) -> Dict[str, Any]:
    """The reported shape of one run, from its stored document.

    ONE definition, because there are now two sources: this worker's own disk
    and a record shared by another instance of the app. A shared listing that
    returns the stored document raw instead costs the difference below -- the
    same run read locally and through a record then comes back:

        LOCAL  STOPPED  [checkpoint_available, config, elapsed_s, message,
                         session_id, start_time, status, updated_at]
        REMOTE TRAINING [start_time, status, updated_at]

    So a run whose instance died reported TRAINING forever everywhere else,
    because the stale sweep lives here and the shared path skipped it, and the
    page's other four fields were simply missing. Both sources go through this
    function now, and a test lists one run both ways and requires the dicts to
    be EQUAL.

    *config* and *checkpoint_available* are arguments rather than reads because
    they are facts about a PLACE: the params file and the checkpoint are on the
    worker that ran the training. For a record from elsewhere the config travels
    with it and the checkpoint is not here.
    """
    st["session_id"] = session_id
    st["checkpoint_available"] = checkpoint_available
    # The config a run used, so a history listing can say WHAT was trained and not
    # merely that something was. It was already persisted per run and reachable only
    # from inside this app; `{}` for a run that died before the file was written.
    st["config"] = config
    if st.get("start_time"):
        end = st.get("end_time") or time.time()
        st["elapsed_s"] = round(end - st["start_time"], 1)
    # A run with no live thread that hasn't updated in a while is stale → stopped.
    if (
        st.get("status") in ("PREPARING", "TRAINING")
        and quiet_seconds(st) > STATUS_STALE_SECONDS
    ):
        st["status"] = "STOPPED"
        st["message"] = "no status update within the stale window"
    return st


def get_status(session_id: str, *, include_private: bool = False) -> Dict[str, Any]:
    st = read_status(session_id)
    if not st:
        return {
            "session_id": session_id,
            "status": "UNKNOWN",
            "message": "no such session",
        }
    shape_status(
        st,
        session_id,
        config=run_config(session_id),
        # `status=st` is NOT optional bookkeeping: `checkpoint_path` hands it to
        # `session_backend`, and without it the record this function has just
        # parsed is read a SECOND time, once per run in every listing. The read
        # count is pinned by a test, which is what caught it.
        checkpoint_available=checkpoint_path(session_id, status=st).exists(),
    )
    if not include_private:
        redact_private(st)

    return st


def status_from_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One shared record as this instance reports it, or None if unusable.

    `checkpoint_available` is False by construction: the weights are on the
    instance that trained the run, and this one cannot serve them. Saying False
    is the honest answer to "can this be served from here", and claiming the
    record's value would make the page offer a model no replica here holds.
    """
    if not isinstance(record, dict):
        return None
    st = record.get("status")
    session_id = record.get("session_id")
    if not isinstance(st, dict) or not isinstance(session_id, str) or not session_id:
        return None
    # Filtered again on the way out. A record written by a version without the
    # write-side filter still carries worker-local paths, and a reader that
    # trusted the stored config would publish them.
    return shape_status(
        dict(st),
        session_id,
        config=config_for_display(record.get("config")),
        checkpoint_available=False,
    )


def scan_runs_for_listing() -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
    """ONE walk of the sessions root, for the listing that needs both halves.

    Returns ``(records, dataset_ids, skipped)``: every run's full record keyed
    by session id, the distinct ``dataset_artifact_id`` values across them in
    first-seen order, and the names that were skipped.

    Exists because the published listing needs the dataset ids BEFORE it can
    filter -- resolving which of them the caller manages is one artifact-manager
    round trip each, and must happen once per call rather than once per run --
    and then needs the records themselves. Asking for those separately read
    every record twice over, on top of the read inside `get_status`: measured
    at 3 reads per run, so 783 of them for the 261 runs on the production
    store. A previous commit claimed to have removed that and had not; this is
    the combined scan that actually does, and `test_the_PUBLISHED_listing_reads_each_record_ONCE`
    measures the path the caller really takes rather than an inner helper.
    """
    records: Dict[str, Dict[str, Any]] = {}
    datasets: Dict[str, None] = {}
    names, skipped = run_names_on_disk()
    for name in names:
        try:
            status = get_status(name, include_private=True)
        except InvalidSessionId:
            skipped.append(name)
            continue
        records[name] = status
        dataset = status.get(DATASET_FIELD)
        if isinstance(dataset, str) and dataset:
            datasets.setdefault(dataset, None)
    return records, list(datasets), skipped


def merge_run_sources(
    local_records: Dict[str, Dict[str, Any]],
    local_dataset_ids: List[str],
    shared_records: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], frozenset, List[str]]:
    """The two run sources as ONE listing input: ``(records, shared_ids, dataset_ids)``.

    This is the only place the collision between the two sources is decided.
    It is a named function so that the published entry point and the tests run
    the SAME code: a test that composed the two sources itself could agree with
    itself while disagreeing with what a caller actually gets.

    LOCAL WINS, and it wins by ID rather than by visibility. A run this
    instance holds is taken from the local record even when the caller cannot
    see it. Resolving by visibility instead would make the shared copy a second
    route to a run the local rule just refused -- a bypass rather than a
    fallback. Local is also the better record: it has the checkpoint and the
    full config, and its status was read on this call, while a shared record is
    only as new as the last sync.

    *dataset_ids* comes back as the UNION. A shared run names a dataset too,
    and omitting it would leave a dataset's manager unrecognised for exactly
    the runs they did not start themselves -- which is the case co-manager
    access exists for.

    A shared record that is not a mapping contributes no dataset and is still
    passed through: `list_sessions` is where a record is judged unusable, so
    that the count of dropped records is made in one place and logged once.
    """
    records = dict(local_records)
    dataset_ids = list(local_dataset_ids)
    shared_ids = []
    for session_id, record in (shared_records or {}).items():
        if session_id in records:
            continue
        records[session_id] = record
        shared_ids.append(session_id)
        status = record.get("status") if isinstance(record, dict) else None
        dataset = status.get(DATASET_FIELD) if isinstance(status, dict) else None
        if isinstance(dataset, str) and dataset and dataset not in dataset_ids:
            dataset_ids.append(dataset)
    return records, frozenset(shared_ids), dataset_ids


def list_sessions(
    context: Any,
    *,
    writable_datasets: Any = frozenset(),
    records: Optional[Dict[str, Dict[str, Any]]] = None,
    shared_ids: Any = frozenset(),
    skipped: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """The runs this caller may see, keyed by session id, owner redacted.

    *records* is EVERY run to be listed, keyed by session id, already read by
    the caller so that this function does not read them again. Passing None
    means "read this worker's disk yourself", which is what every direct caller
    and every test that does not care about the shared history does.

    *shared_ids* names which of *records* came from ANOTHER instance rather
    than from this worker's disk, and it is the whole of the difference between
    the two sources. A local entry is a status this worker produced and is used
    as it is; a shared entry is the stored record as another instance persisted
    it, is unwrapped through `status_from_record`, and reports
    ``checkpoint_available=False`` because the weights are on that instance's
    disk and this one cannot serve them.

    ONE PARAMETER, ONE MEANING. These two are deliberately not a single
    "records" argument carrying a mixture: the caller knows which source each
    id came from and this function cannot tell by looking, so making it guess
    -- by shape, or by trying the unwrap and falling back -- would turn a
    malformed local status into a silently shared one. The caller also decides
    the collision: an id in BOTH sources belongs to the local record and is not
    in *shared_ids*, because the local copy has the checkpoint, the full config
    and a status re-read on this call rather than one as old as the last sync.

    *context* is REQUIRED and has no default, deliberately. There is no
    unfiltered listing to call by mistake: the previous `list_sessions()`
    returned every run on the worker to whoever asked, and leaving that
    signature available with a filtered sibling beside it would make the open
    behaviour one forgotten argument away.

    Visibility is exactly `caller_may_see_run`: the caller owns the run, or
    the run names a dataset the caller manages, or the run has no owner and the
    caller administers ``bioimage-io``. The listing does NOT re-implement that
    test -- it calls it -- because a listing carrying its own private copy of
    the rule is how a caller comes to be shown a run it is then refused.

    *writable_datasets* is resolved ONCE per call by the async entry point and
    passed in; see its docstring for why it is not resolved here. Defaulting it
    to empty keeps every existing caller -- and every test -- on the
    owner-or-admin behaviour unless co-manager access is deliberately supplied.

    All legs fail closed: an unreadable context yields no identity, no
    authority, and so an empty listing. The owner is read here and redacted from
    everything returned, so the field decides visibility and is never itself
    published.

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
    out: Dict[str, Dict[str, Any]] = {}
    shared = frozenset(shared_ids or ())
    unusable = 0
    if records is not None:
        # Already read by `scan_runs_for_listing`; re-reading them here is the
        # cost this parameter exists to avoid. The names IT skipped have to be
        # passed in with them: this function logs them, and a caller that
        # supplies records but drops the skip list silently removes the only
        # trace a stray directory leaves. That regressed once, when the single
        # scan was introduced and the published entry point discarded them.
        names = list(records)
        skipped = list(skipped or [])
    else:
        names, found_skipped = run_names_on_disk()
        skipped = list(skipped or []) + found_skipped
    for name in names:
        try:
            if records is None:
                status = get_status(name, include_private=True)
            elif name in shared:
                # A SHARED-ONLY run: `records[name]` is the stored record as
                # persisted by the instance that trained it, not a status this
                # worker produced, so it is unwrapped HERE rather than by the
                # caller. That keeps one shaping site for both sources -- see
                # `shape_status` -- and it is why the caller passes the raw
                # record through instead of shaping it on the way in.
                #
                # THE ID IS VALIDATED BEFORE IT IS USED AS ONE. These keys come
                # from the shared artifact rather than from this worker's own
                # directory listing, so they are not already known to be
                # session ids; a key that is not one is counted and dropped.
                if not SESSION_ID_PATTERN.match(name):
                    unusable += 1
                    continue
                status = status_from_record(records[name])
                if status is None:
                    # A record whose status is missing or not a mapping says
                    # nothing about who may see it, and the gate below would
                    # read a missing owner as "ownerless" and show it to an
                    # administrator. Counted and dropped instead: fail closed,
                    # and say how many.
                    unusable += 1
                    continue
            else:
                status = records[name]
        except InvalidSessionId:
            # The name passed but a deeper layer refused it -- today that means
            # a directory whose NAME is a valid session id but which is a
            # SYMLINK out of the sessions root. Skipped like a bad name, so one
            # bad link cannot take the whole history down.
            skipped.append(name)
            continue
        # ONE test, shared with resume, rather than a second copy of the rule
        # here: a listing that decides visibility by its own private copy of the
        # rule is exactly how a caller comes to be shown a run that resume then
        # refuses, and the drift is silent when it happens.
        #
        # The status just read is handed over rather than re-read. An earlier
        # version resolved the id inside the gate, which opened the same file
        # three times per listed run -- 261 runs became 783 reads on the
        # production store, for a listing that used to do one each. Passing the
        # RECORD is not passing the ANSWER: the rule is still the gate's.
        #
        # ORDER IS LOAD-BEARING: `redact_private` mutates this dict in place and
        # strips `owner`, which is the field the gate compares. Redacting first
        # would make every owned run invisible to its own owner.
        if caller_may_see_run(
            name, context, writable_datasets=writable_datasets, status=status
        ):
            out[name] = redact_private(status)
    if skipped:
        print(
            f"[model-finetune] skipped {len(skipped)} session directory(ies) that "
            f"are not a session on this worker: {skipped[:5]}",
            flush=True,
        )
    if unusable:
        print(
            f"[model-finetune] dropped {unusable} shared run record(s) that carry "
            "no usable status or no usable id; they are not listed to anybody",
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
    """The record file for *export_id*, guaranteed to be under the exports root.

    Validated AND contained, for the same reason `session_dir` is both: the id
    arrives from a caller, and softening either check alone leaves the other
    refusing.
    """
    root = exports_root()
    candidate = root / f"{validate_export_id(export_id)}.json"
    resolved_root = root.resolve()
    try:
        candidate.resolve().relative_to(resolved_root)
    except ValueError:
        raise InvalidExportId("export_id resolves outside the exports root")
    return candidate


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
        and quiet_seconds(rec) > STATUS_STALE_SECONDS
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

    One instance per POLYGON PART, ids from 1 in document order — not one per
    feature. A MultiPolygon of two parts yields two ids (measured: ids [1, 2]),
    because the counter advances per part, and interior rings are ignored, so a
    polygon with a hole is filled solid. Both matter to a caller counting
    instances: a two-part feature is two objects here and one object in the
    file it came from.

    THE AXIS CONVENTION, which GeoJSON itself does not fix and which nothing
    here can detect a violation of. Coordinates are IMAGE PIXELS as ``[x, y]``
    with the origin at the TOP-LEFT and **y increasing DOWNWARD** -- the raster
    convention, so ``y`` is the row index and the pair maps to ``mask[y][x]``.
    That is what the pairs below mean when they reach PIL, whose drawing origin
    is the top-left, and it is the bioimage.io colab polygon convention this was
    ported from (cellpose-finetuning's ``_rasterize_geojson_to_tiff``).

    It is NOT the y-up convention that GIS GeoJSON readers assume. A y-up file
    rasterises VERTICALLY MIRRORED, and that is silent: every polygon still
    lands inside the canvas, the instance count is right, the mask is non-empty,
    so the empty-mask guard in the caller does not fire and training reports
    success having learnt mirrored cells. A caller reported exactly that risk
    and rasterised locally to avoid it. There is no way to tell the two apart
    from one file -- a mirrored annotation is a perfectly plausible annotation
    -- so the convention has to be stated rather than validated.
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


def config_for_display(params: Any) -> Dict[str, Any]:
    """The allow-listed view of a run's params. One definition, three callers.

    `run_config` reads the params off local disk and projects them; the run
    record projects them at the WRITE; `status_from_record` projects again at
    the READ. All three go through this function, so the allow-list cannot hold
    on one path and be bypassed on another -- which is exactly what happened
    when the shared listing reported `record["config"]` unfiltered.
    """
    if not isinstance(params, dict):
        return {}
    return {k: params[k] for k in RUN_CONFIG_FIELDS if k in params}


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
    # REDACTED here too, so the local and shared paths are identical by
    # construction rather than by the current contents of the params. A review
    # measured the difference: a presigned URL inside an allow-listed field was
    # stripped on the shared path and shown in full on the local one. It cannot
    # happen today -- the only free-form allow-listed field is the pool
    # provenance, and the one place that builds it uses two ids and no URL --
    # but "cannot happen today" is a fact about today's callers, and the
    # equality the shared listing rests on should not depend on it.
    return redact_urls(config_for_display(params))
