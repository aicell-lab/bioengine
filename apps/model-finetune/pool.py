"""Async model-soup pool — storage layout + record semantics (backend-owned).

The community pool is a Hypha artifact (one per model family; cpsam first) that
holds contributed weights, per-contribution metadata, and versioned community
checkpoints. Contributors fine-tune on their own local data at their own pace and
upload only weights — the data never moves. A coordinator averages selected
contributions into a new community checkpoint when the ``aggregate`` step is
invoked. That call comes from outside this module, through the skill; there is
no trigger in this module. ``aggregate`` lives in the entry because it does
torch + artifact I/O; the pure combine math — ``uniform_soup`` — and the
community schema live here.

This module owns the STORAGE layout and the record SEMANTICS — what fills each
field. The campaign page owns the wire format it renders; every record here
carries ``SCHEMA_VERSION`` so a consumer can detect a backend/page mismatch. The
functions are pure (no RPC client, no network): the entry does the async artifact
I/O and calls these to build/mutate the JSON documents, which keeps the schema
unit-testable and free of transport concerns.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

# Tracks the campaign-page wire contract this backend maps onto. Echoed on every
# record and every read method so the page can flag a backend/page mismatch
# instead of silently mis-rendering. Bump in lock-step with the page's fixtures.
SCHEMA_VERSION = "0.15.0-draft"

POOL_INDEX = "pool_index.json"
# Resume journal for aggregate(): scores already paid for, so a killed merge
# restarts from where it died instead of from nothing. See usable_merge_journal.
MERGE_JOURNAL = "merge_journal.json"
WEIGHTS_NAME = "weights.pt"
METADATA_NAME = "metadata.json"
MANIFEST_NAME = "manifest.json"

# The gate and the witness curve are scored with the SAME metric+matcher
# (instance_f1 below), differing only in which split they run on — so the metric
# NAME is identical on both role-tagged records; only aggregate_basis names the
# split. A holdout-scoped metric turns off the participant-pool floor/completeness
# gate on the page, which is correct: these splits are curated holdouts, not a
# pooled per-site average.
METRIC_NAME = "mean instance F1 at IoU 0.5"
_HOLDOUT = "campaign_holdout"
# A basis string is published provenance, so it may only state what this app can
# check. Split A and B arrive as caller-supplied arrays; their disjointness is a
# precondition of aggregate(), never something verified here. Name the corpus by the
# gate_split/witness_split hash on the same record and attribute the rest.
SELECTION_BASIS = (
    "the score the greedy gate admitted contributions against, on split A, the corpus "
    "recorded as gate_split on this record and declared by the coordinator disjoint "
    "from every contributor's training data"
)
WITNESS_BASIS = (
    "scored on split B, the corpus recorded as witness_split on this record and declared "
    "by the coordinator held back from split A and from every contributor, the only "
    "number the improvement curve plots"
)
EMPTY_SELECTION_BASIS = (
    "the score the greedy gate required a contribution to beat on split A, which none "
    "of the assessed contributions did"
)
BASELINE_BASIS = (
    "Cellpose-SAM 0.2.0 as published (cpsam_v2, pytorch_state_dict sha256 "
    "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667), "
    "scored on the witness split before the first merge"
)

# The two wordings 0.27.15 (commit e762a17b5af5) retracted, verbatim. Both asserted
# disjointness as FACT — "a public benchmark disjoint from…", "the disjoint witness
# split B, held back from…" — which this app cannot check, because split A and B
# arrive as caller-supplied arrays. Records written before that release still carry
# them and are still shown to contributors, so the read path below maps them onto the
# current wording. They live here as constants, not as a substring search, because a
# substring that drifts stops matching and a search that stops matching reports clean.
#
# Keep these byte-exact. They are historical record, not editable prose: changing one
# character stops it matching the stored data it exists to recognise, and nothing
# announces that — the projection just quietly resumes publishing the retracted claim.
RETRACTED_SELECTION_BASIS = (
    "the score the greedy gate admitted contributions against, on split A, a public "
    "benchmark disjoint from every contributor's training data"
)
RETRACTED_WITNESS_BASIS = (
    "scored on the disjoint witness split B, held back from the selection split and "
    "from every contributor, the only number the improvement curve plots"
)
# Stored wording -> what it is corrected to on the way out. The value is the current
# constant, so the two cannot drift apart: re-word SELECTION_BASIS and the correction
# follows automatically rather than silently continuing to serve the older rewrite.
_RETRACTED_BASIS = {
    RETRACTED_SELECTION_BASIS: SELECTION_BASIS,
    RETRACTED_WITNESS_BASIS: WITNESS_BASIS,
}
# Appended to a corrected record so the substitution is visible rather than seamless.
# A contributor comparing what the page shows against a stored record they fetched
# themselves would otherwise find a difference with no explanation, which is the same
# "is this corrupt?" dead end #0417 describes for global_sha256.
BASIS_CORRECTION_NOTE = (
    "[corrected on read: the record as stored asserts this split's disjointness as "
    "fact; that claim was withdrawn in app 0.27.15 because the splits arrive as "
    "caller-supplied arrays and this app cannot verify them. The stored record is "
    "unchanged — only this wording is restated as what the app can actually check.]"
)


def round_metric(
    *,
    role: str,
    aggregate: Optional[float],
    aggregate_basis: Optional[str],
    name: str = METRIC_NAME,
    aggregate_scope: Optional[str] = _HOLDOUT,
    higher_is_better: bool = True,
    n_sites_scored: Optional[int] = None,
    n_datasets_scored: Optional[int] = None,
    per_site: Optional[Any] = None,
    per_site_basis: Optional[str] = None,
    aggregate_withheld: Optional[Any] = None,
) -> Dict[str, Any]:
    """One role-tagged metric record — the page's RoundMetric (11 fields).

    ``role`` is the load-bearing tag: the page refuses to plot anything whose
    ``role != 'witness'`` on the improvement curve, EVEN if it arrived through the
    ``witness_metric`` field. The field NAME and this tag are two carriers that must
    agree; keeping both makes a selection/witness swap detectable rather than silent.
    A ``campaign_holdout`` scope must not carry a ``per_site`` map (the page refuses
    that combination), so both stay null for our curated splits."""
    if role not in ("witness", "selection"):
        raise ValueError(f"role must be 'witness' or 'selection', not {role!r}")
    if aggregate_scope == _HOLDOUT and per_site is not None:
        raise ValueError("a campaign_holdout metric must not carry a per_site map")
    # Refusing at the WRITE site, not just correcting at the read site. Correcting on
    # read fixes the records that already exist; this stops a new one being written
    # with wording the project has withdrawn — a caller passing a basis string copied
    # from an old record, which is exactly how the retracted text would come back.
    if aggregate_basis in _RETRACTED_BASIS:
        raise ValueError(
            "aggregate_basis is wording withdrawn in app 0.27.15 (commit e762a17b5af5): "
            "it asserts the split's disjointness as fact, which this app cannot verify "
            "because split A and B arrive as caller-supplied arrays. Use "
            f"{'SELECTION_BASIS' if role == 'selection' else 'WITNESS_BASIS'} instead."
        )
    return {
        "role": role,
        "name": name,
        "aggregate": aggregate,
        "aggregate_basis": aggregate_basis,
        "aggregate_scope": aggregate_scope,
        "higher_is_better": higher_is_better,
        "n_sites_scored": n_sites_scored,
        "n_datasets_scored": n_datasets_scored,
        "per_site": per_site,
        "per_site_basis": per_site_basis,
        "aggregate_withheld": aggregate_withheld,
    }


def merge_trigger() -> Dict[str, Any]:
    """The campaign's merge-cadence policy record (5 fields). All null: no
    merge-cadence policy is declared here and the merge provenance is unpopulated,
    pending the campaign's trigger ruling. A merge-on-N-contributions policy would
    set ``kind`` and ``contributions_per_merge``; a named decider would set
    ``decided_by``/``agent``."""
    return {
        "kind": None,
        "contributions_per_merge": None,
        "next_merge_at": None,
        "decided_by": None,
        "agent": None,
    }


def contribution_paths(contribution_id: str) -> Dict[str, str]:
    base = f"contributions/{contribution_id}"
    return {"weights": f"{base}/{WEIGHTS_NAME}", "metadata": f"{base}/{METADATA_NAME}"}


def community_paths(community_version: str) -> Dict[str, str]:
    base = f"community/{community_version}"
    return {"weights": f"{base}/{WEIGHTS_NAME}", "manifest": f"{base}/{MANIFEST_NAME}"}


def architecture_signature(state_dict: Dict[str, Any]) -> str:
    """sha256 over the sorted ``(name, shape, dtype)`` triples of a state_dict.

    Identifies the exact parameter layout independent of the weight values, so a
    cpdino checkpoint (or a different cpsam revision) is refused before it is
    averaged into a cpsam pool. Two checkpoints share a signature iff a plain
    key-wise mean over them is well-defined.
    """
    items = sorted(
        (name, list(tuple(t.shape)), str(t.dtype)) for name, t in state_dict.items()
    )
    payload = json.dumps(items, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def state_dict_digest(state_dict: Dict[str, Any]) -> str:
    """sha256 over a state_dict's keys, shapes, dtypes and raw weight bytes.

    Digests the TENSORS rather than a file ``torch.save`` wrote: torch's zip
    container embeds the target file's stem in every record name, so hashing the
    saved bytes makes the digest a function of the filename and two merges of
    identical members can never agree. This is reproducible by construction and
    independent of the container format.
    """
    h = hashlib.sha256()
    for name in sorted(state_dict):
        t = state_dict[name]
        h.update(f"{name}|{tuple(t.shape)}|{t.dtype}|".encode("utf-8"))
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ── which of the two digests a stored value is ───────────────────────────────
# The published head digest (``merges[].global_sha256``, the
# ``current_community.sha256`` copy of it and the ``weights_sha256`` copy in each
# community manifest — one value written once, copied three ways) digested the FILE
# ``torch.save`` wrote before commit d3040c4f8165 (app 0.27.12) and digests the
# TENSORS after it. Same field name, same type, same 64-hex shape, so the value
# alone does not say which. A contributor who re-merges an identical member set
# gets a tensor digest, compares it against a stored file digest, and the only
# conclusion available to them is that the community model is corrupt. It is not:
# the two numbers were never answers to the same question. These constants make
# both meanings nameable and ``digest_provenance`` attributes a stored value to one
# of them — or declines to, which is the point.

DIGEST_BASIS_TENSORS = "state_dict_tensors"
DIGEST_BASIS_FILE = "torch_save_file"
DIGEST_BASIS_UNDETERMINED = "undetermined"
# A record with no digest at all (a merge that admitted nothing publishes no head).
# Kept apart from "undetermined" on purpose: "there is no value to attribute" and
# "the value cannot be attributed" are different answers and collapsing them would
# hide the second behind the first.
DIGEST_BASIS_ABSENT = "no_digest"

# What ``state_dict_digest`` above computes TODAY, stamped onto every new record so
# that a reader after the NEXT change to it never has to date-reason the way this
# module has to for pre-0.27.12 records. Change this in the SAME commit that changes
# ``state_dict_digest``: test_digest_basis.py pins that function's output against a
# golden vector, so a change which forgets this constant fails there.
CURRENT_DIGEST_BASIS = DIGEST_BASIS_TENSORS

# What a CONTRIBUTION's ``weights_sha256`` is — and it is NOT CURRENT_DIGEST_BASIS.
# The cut-over above moved the COMMUNITY head digest only. A contribution's digest is
# computed by ``sha256_file`` over the checkpoint the client uploaded, and it always
# has been. Measured 2026-11-14 over runtime_cellpose.py with `git log -G`:
#
#   :586  sha = await asyncio.to_thread(pool.state_dict_digest, soup)   <- the soup
#   :634  weights_sha256 = await asyncio.to_thread(pool.sha256_file, ckpt)  <- this
#
#   the soup line:         changed by d3040c4f ("digest the soup tensors, not the
#                          file torch wrote") — the cut-over itself
#   the contribution line: 6eec9581 (app 0.20.0) only, i.e. the commit that
#                          INTRODUCED it. d3040c4f did not touch it; it reads
#                          sha256_file at 6eec9581 and at d3040c4f alike.
#
# So this surface has exactly one basis over its whole history and no cut-over
# instant applies to it. Two consequences, both load-bearing:
#   * stamping CURRENT_DIGEST_BASIS here would assert "sha256 over keys, shapes,
#     dtypes and raw weight bytes" over a number produced by hashing a file. Worse
#     than the absence it replaces: digest_basis honours an explicit basis above
#     every other signal, so a wrong explicit value is BELIEVED, while an absent one
#     at least resolves to a refusal.
#   * an absent field on a contribution means the file basis as a PROOF, not as a
#     dated inference — the code that could have written anything else never existed
#     on this path. That is why contribution_digest_basis passes ``absent_basis``
#     instead of date-reasoning: one unchanged code path must not answer two ways
#     depending on when a deposit happened.
# Lock-step rule, same as the constant above: change this in the SAME commit that
# changes which digest function ``prepare_contribution`` calls. A grep for
# ``sha256_file, ckpt`` in runtime_cellpose.py is the thing that must still match.
CONTRIBUTION_DIGEST_BASIS = DIGEST_BASIS_FILE

# 2026-10-26T08:43:26Z — commit d3040c4f8165, which introduced state_dict_digest.
# Strictly BEFORE this instant the tensor-digest code did not exist anywhere, so a
# record written then carries a file digest with certainty. At or after it nothing
# follows: a pool can be served by a worker still running an older app build, so a
# late timestamp is UNDETERMINED rather than evidence of the new meaning. The
# asymmetry is deliberate — resolving the late side to the new meaning would be
# exactly the silent default this exists to remove, one version later.
# Read with `git show -s --format=%ct d3040c4f8165`, not by hand: a value derived by
# re-parsing the date is a year out at a glance and still resolves every real stored
# record to "undetermined", i.e. it looks like a working fix that attributes nothing.
# test_digest_basis.py keeps an independent copy and pins the eight live records
# against it, so a wrong constant here fails there rather than in production.
TENSOR_DIGEST_SINCE = 1793004206.0
TENSOR_DIGEST_SINCE_APP_VERSION = "0.27.12"

# Published next to the digest. This is the text that stands between a contributor
# and the conclusion that an intact community model is corrupt, so it says what the
# number is, whether re-merging reproduces it, and what to do when it does not.
DIGEST_BASIS_NOTES = {
    DIGEST_BASIS_TENSORS: (
        "sha256 over the checkpoint's keys, shapes, dtypes and raw weight bytes in "
        "sorted key order. Reproducible by construction: re-merging the same members "
        "in the same order yields this value again, so a mismatch is evidence about "
        "the weights."
    ),
    DIGEST_BASIS_FILE: (
        "sha256 over the .pt file torch.save wrote, which this app named "
        "head-<random>.pt before version "
        f"{TENSOR_DIGEST_SINCE_APP_VERSION}. torch's zip container embeds that random "
        "stem in every record name, so the digest is a function of the filename: "
        "re-merging the same members CANNOT reproduce it, and a mismatch against it "
        "is not evidence about the weights. Compare the weights themselves, or "
        "re-merge and compare against a digest recorded as "
        f"'{DIGEST_BASIS_TENSORS}'."
    ),
    DIGEST_BASIS_UNDETERMINED: (
        "this record does not say which digest the value is, and it was written late "
        "enough that the timestamp cannot settle it either — the app may have been "
        "running either build. Treat a mismatch against it as UNEXPLAINED, not as "
        "evidence about the weights."
    ),
    DIGEST_BASIS_ABSENT: (
        "no head digest on this record: this merge admitted nothing, so no community "
        "checkpoint was published and there is no value to attribute."
    ),
}

# Whether re-merging an identical member set reproduces the stored value. None is a
# third state, not a false: "we do not know" must not render as "no".
_DIGEST_REPRODUCIBLE = {
    DIGEST_BASIS_TENSORS: True,
    DIGEST_BASIS_FILE: False,
    DIGEST_BASIS_UNDETERMINED: None,
    DIGEST_BASIS_ABSENT: None,
}


def digest_basis(
    record: Optional[Dict[str, Any]],
    *,
    digest_key: str = "global_sha256",
    basis_key: Optional[str] = None,
    created_at_key: str = "created_at",
    absent_basis: Optional[str] = None,
) -> str:
    """Which of the two digests ``record[digest_key]`` is, from the record alone.

    Resolution order, each step refusing rather than guessing:

    1. no digest stored                       -> ``DIGEST_BASIS_ABSENT``
    2. an explicit basis field this build
       recognises                             -> that basis
    3. an explicit basis field it does NOT
       recognise (written by a newer app)     -> ``DIGEST_BASIS_UNDETERMINED``
    4. no basis field and ``absent_basis``
       given by the caller                    -> ``absent_basis``
    5. no basis field, ``created_at`` finite
       and strictly before TENSOR_DIGEST_SINCE-> ``DIGEST_BASIS_FILE``
    6. anything else                          -> ``DIGEST_BASIS_UNDETERMINED``

    Step 5 is one-directional by construction. A timestamp before the cutoff proves
    the file meaning because the tensor-digest code did not yet exist; a timestamp
    after it proves nothing, because deployment lags the commit. There is no input
    for which this returns ``DIGEST_BASIS_TENSORS`` on timestamp evidence alone.

    ``absent_basis`` is for a surface whose digest function NEVER changed, where the
    date is not evidence of anything because there was never a second meaning to
    choose between — see ``CONTRIBUTION_DIGEST_BASIS``. It is deliberately not the
    default: passing it asserts that a code path's whole history was checked, which
    is true of the contribution path and false of the community head. It cannot
    override a basis the record states (steps 2 and 3 come first), so it adds no way
    for a caller to contradict stored data; it only says what an absence means.
    An unrecognised value raises rather than resolving to "undetermined" — that
    would be a typo'd constant reading exactly like the honest refusal this module
    exists to give.
    """
    if absent_basis is not None and absent_basis not in (
        DIGEST_BASIS_TENSORS,
        DIGEST_BASIS_FILE,
    ):
        raise ValueError(
            f"absent_basis={absent_basis!r} is not one of {DIGEST_BASIS_TENSORS!r} "
            f"or {DIGEST_BASIS_FILE!r}. It states what a MISSING field means, so an "
            f"unrecognised value would silently become 'undetermined' and read as "
            f"this function working."
        )
    if not isinstance(record, dict):
        return DIGEST_BASIS_UNDETERMINED
    if basis_key is None:
        basis_key = f"{digest_key}_basis"
    if not record.get(digest_key):
        return DIGEST_BASIS_ABSENT
    stated = record.get(basis_key)
    if stated in (DIGEST_BASIS_TENSORS, DIGEST_BASIS_FILE):
        return stated
    if stated is not None:
        return DIGEST_BASIS_UNDETERMINED
    if absent_basis is not None:
        return absent_basis
    try:
        written_at = float(record.get(created_at_key))
    except (TypeError, ValueError):
        return DIGEST_BASIS_UNDETERMINED
    if not math.isfinite(written_at):
        return DIGEST_BASIS_UNDETERMINED
    if written_at < TENSOR_DIGEST_SINCE:
        return DIGEST_BASIS_FILE
    return DIGEST_BASIS_UNDETERMINED


def digest_provenance(
    record: Optional[Dict[str, Any]], **kwargs: Any
) -> Dict[str, Any]:
    """``digest_basis`` plus the two things a reader needs to act on it: whether a
    re-merge reproduces the value, and the sentence that explains a mismatch.

    Published alongside every digest the app hands out, so the explanation travels
    with the number instead of living in a document the reader has to already know
    about. Takes the same keyword arguments as ``digest_basis``.
    """
    basis = digest_basis(record, **kwargs)
    return {
        "basis": basis,
        "reproducible_by_re_merge": _DIGEST_REPRODUCIBLE[basis],
        "note": DIGEST_BASIS_NOTES[basis],
    }


def contribution_digest_basis(record: Optional[Dict[str, Any]]) -> str:
    """Which of the two digests a CONTRIBUTION's ``weights_sha256`` is.

    Always the FILE basis unless the record itself says otherwise, because
    ``prepare_contribution`` has computed this value with ``sha256_file`` on every
    build that ever wrote one — the history is quoted at
    ``CONTRIBUTION_DIGEST_BASIS``. So an absent field is a proof here, and no
    timestamp is consulted: ``uploaded_at`` is NOT passed, and that omission is the
    point rather than an oversight. Dating a record whose code path never changed
    would make one deposit resolve "file" and another "undetermined" on nothing but
    when it arrived.

    Named for this surface rather than spelled out at each call site because a
    contribution is the one record shape whose digest key differs from the default,
    so a plain ``digest_basis(record)`` is wrong in a way that does not raise: it
    looks up ``global_sha256``, finds nothing, and answers "no_digest" — "this
    record stores no digest", said of a record that stores one. The half-right call
    (``digest_key`` corrected, ``absent_basis`` forgotten) answers "undetermined"
    for a record whose code path PROVES the file meaning, and that one is the
    dangerous one: it is indistinguishable from the honest refusal this module
    exists to give, so it reads as the fix working. Both are pinned in
    test_digest_basis.py, which is what keeps this function from being an alias.
    """
    return digest_basis(
        record,
        digest_key="weights_sha256",
        absent_basis=CONTRIBUTION_DIGEST_BASIS,
    )


# Entries that live in a cpsam state_dict but are data statistics rather than
# learned weights. cellpose's own CPSAM/CPDINO do not declare these at all —
# cellpose_model.py registers them, as nn.Parameter(requires_grad=False), so they
# are float32 scalars with matching shapes and pass every dtype/shape filter.
# requires_grad is not serialised, so a state_dict carries no way to tell them
# apart — the name is the only handle, and it must come from the key (storage keys
# inside a .pt are bare positional integers and position is not stable across
# writers).
#
# ⚠ THIS IS THE SINGLE DECLARATION AND cellpose_model.py REGISTERS BY ITERATING IT
# (#0070 requirement 1: classify from the model definition, not from a second
# hardcoded list of names). It lives here, in the torch-free module, because
# pool.py's import must stay stdlib-only — test_pool.py's contract — so the
# dependency runs model -> pool and never the other way. Adding a non-learned
# Parameter therefore means adding it HERE, which is what makes it excluded from
# the soup in the same edit that registers it; test_soup_math.py asserts the two
# agree, so the pair cannot drift.
NON_WEIGHT_PARAMS = frozenset({"diam_labels", "diam_mean"})


def is_non_weight_param(key: str) -> bool:
    """True for a state_dict key holding a data statistic, not a learned weight.

    Matches on the final dot-component because the same checkpoint reaches us
    both bare and module-prefixed depending on how it was wrapped when saved.
    """
    return key.rsplit(".", 1)[-1] in NON_WEIGHT_PARAMS


def state_dict_l2_from_base(current: Dict[str, Any], base: Dict[str, Any]) -> float:
    """L2 distance between two state_dicts over shared learned weights — how far a
    contribution's weights moved from the checkpoint it started from. A drift proxy
    recorded on every contribution; its use as an acceptance BOUND is a pending
    aggregation decision, so this only measures, it does not gate.

    Non-weight entries are excluded: a site's median object diameter differs from
    the base's by tens of pixels while real weight motion is single digits, so
    including it made this field ~99.5% diameter metadata and inverted the
    per-site ordering it implies (it reported one site as having moved 17x
    further than another where the weights say 1.3x).
    """
    import torch

    total = 0.0
    for key, cur in current.items():
        ref = base.get(key)
        if ref is None or not torch.is_floating_point(cur) or cur.shape != ref.shape:
            continue
        if is_non_weight_param(key):
            continue
        total += float(torch.sum((cur.detach().float() - ref.detach().float()) ** 2).item())
    return total ** 0.5


# Resident set of a merge, as a multiple of ONE member's on-disk size. uniform_soup
# holds a float32 accumulator for the whole model plus the single member being added;
# on a member's FIRST use _fetch_member also holds the downloaded bytes while
# torch.load materialises them from the same file. Three member-equivalents on first
# use, two on every later one.
#
# bench_soup_memory.py is the re-runnable gate for the flatness this relies on. At a
# real cpsam member size (348 keys, float32, 1.1348 GiB) it holds: `--mode after
# --member-mb 1162` is 2.68 GiB at every n in 2/4/8/16, growth +0.00x, and `--mode
# before` is +13.98x over the same sweep, so the flat rows are a property of the
# combine and not of the rig.
#
# What picks the factor is a different denominator, because the formula adds to a LIVE
# base_bytes: growth above the process baseline, which is 2.00x a member warm and
# 3.00x on a member's first use. So 3 is the cold peak exactly, not slack — lowering
# it under-predicts first use, which is the common case for a newly contributed
# member. (The cold path is the one thing the bench does not cover; measured
# separately on the same members, flat in n to 0.6 MiB.)
SOUP_RESIDENT_FACTOR = 3


def soup_admission(
    member_bytes: Sequence[Optional[int]], budget_bytes: int, base_bytes: int,
) -> Optional[str]:
    """Reject a merge that cannot fit before it burns any GPU, or None to proceed.

    Returns the reason as a string so the caller owns the exception type.

    Peak no longer depends on member count — the combine streams and the member
    cache is bounded — so this sizes the ONE member that has to be resident, not the
    soup. Fails OPEN: a pool that has not recorded n_bytes for its contributions
    tells us nothing about the peak, and refusing to merge on missing metadata would
    be a worse failure than the one this guards against.
    """
    known = [int(n) for n in member_bytes if n is not None]
    if not known:
        return None
    predicted = base_bytes + SOUP_RESIDENT_FACTOR * max(known)
    if predicted <= budget_bytes:
        return None
    return (
        f"this merge needs about {predicted / 1024 ** 3:.1f} GiB resident "
        f"({base_bytes / 1024 ** 3:.1f} GiB base + {SOUP_RESIDENT_FACTOR}x the "
        f"largest member at {max(known) / 1024 ** 3:.1f} GiB) but the cpsam runtime "
        f"declares {budget_bytes / 1024 ** 3:.1f} GiB to Ray. Raise memory_mb on "
        f"CellposeRuntime to at least {predicted / 1024 ** 2:.0f} MiB and redeploy. "
        f"Refusing up front: exceeding the declaration gets the replica OOM-killed "
        f"mid-merge, which costs the whole run's GPU time instead of none of it."
    )


def split_provenance_error(
    gate_split: Dict[str, Any], witness_split: Dict[str, Any],
) -> Optional[str]:
    """Reject split provenance that cannot safely identify a split, or None to proceed.

    aggregate() is an RPC entry point and its ``gate_split``/``witness_split`` are
    ``Dict[str, Any] = Field(...)``: the dict is required, its CONTENTS are not. A
    caller that omits ``sha256`` gets the identity ``"<artifact_id>@None"`` for BOTH
    splits, and two things go wrong silently:

    * entry.py maps identity -> images. One identity means one entry, the witness
      overwrites the gate, and every selection score is then computed on the witness
      split — the circularity the two-split design exists to prevent.
    * a journal written for a different split that also lacked ``sha256`` compares
      equal on all four stamps and is judged usable, so scores measured on other
      validation data decide which members are admitted.

    Neither raises on its own, so this is checked once at the boundary rather than
    defended at each call site.
    """
    for name, split in (("gate_split", gate_split), ("witness_split", witness_split)):
        if not isinstance(split, dict):
            return f"{name} must be a dict of {{artifact_id, sha256}}, got {type(split).__name__}."
        for field in ("artifact_id", "sha256"):
            if not str(split.get(field) or "").strip():
                return (
                    f"{name} is missing '{field}'. Both splits need "
                    f"{{artifact_id, sha256}} provenance: it is what identifies a "
                    f"split, and without it the gate and witness splits collapse to "
                    f"one identity and selection silently reads the witness data."
                )
    if split_identity(gate_split) == split_identity(witness_split):
        return (
            f"gate_split and witness_split both resolve to "
            f"'{split_identity(gate_split)}'. Selection reads split A and the "
            f"reported curve is split B; scoring both on one split makes the "
            f"witness curve circular."
        )
    return None


def split_identity(split: Dict[str, Any]) -> str:
    """Stable id for a validation split, from the provenance every merge already
    records. Two splits with the same sha256 score identically, so a score may be
    reused across merges; two with different sha256 may not."""
    return f"{split.get('artifact_id')}@{split.get('sha256')}"


def score_key(split_id: str, member_paths: Sequence[str]) -> str:
    """Identity of one scoring pass: which soup, on which split.

    Sorted, because a soup is a uniform MEAN and so is defined by the SET of its
    members — ``[a, b]`` and ``[b, a]`` are the same checkpoint and must not be
    scored twice. That is also what lets the greedy admission pass reuse the
    best-first ordering pass's score for the candidate it ends up taking first.
    """
    canonical = "\x1f".join([split_id, *sorted(member_paths)])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def new_merge_journal(gate_split, witness_split, diam_mean: float) -> Dict[str, Any]:
    """An empty resume journal stamped with everything a recorded score depends on."""
    return {
        "schema_version": SCHEMA_VERSION,
        "gate_split": split_identity(gate_split),
        "witness_split": split_identity(witness_split),
        "diam_mean": float(diam_mean),
        "scores": {},
    }


def usable_merge_journal(
    journal: Optional[Dict[str, Any]], gate_split, witness_split, diam_mean: float,
) -> Dict[str, Any]:
    """Return ``journal`` if every score in it is still valid for this merge's
    inputs, else a fresh empty one.

    A recorded score is ``mean_instance_f1`` of a soup on a split, so it stays true
    only while the splits and the segmentation diameter are the ones it was measured
    with. Those are stamped on the journal and compared here; a mismatch discards
    the whole file rather than trying to salvage entries, because the stamp is
    per-journal. ``schema_version`` is included so a draft bump that changes the
    metric also invalidates it — anything else that changes what a score MEANS,
    without moving one of these four, would have to invalidate it by hand.
    """
    fresh = new_merge_journal(gate_split, witness_split, diam_mean)
    if not isinstance(journal, dict) or not isinstance(journal.get("scores"), dict):
        return fresh
    if any(journal.get(field) != fresh[field]
           for field in ("schema_version", "gate_split", "witness_split", "diam_mean")):
        return fresh
    return journal


def record_score(journal: Dict[str, Any], key: str, value: float, cap: int = 512) -> None:
    """Add a score, evicting oldest-first so the journal cannot grow without bound.

    One merge records at most ``2 * max_batch + 3`` scores (67 at the default
    max_batch of 32), so the cap holds several merges' worth — enough for deferred
    candidates, which aggregate() re-scores on the next merge, to come back free.
    """
    journal["scores"][key] = round(float(value), 6)
    while len(journal["scores"]) > cap:
        del journal["scores"][next(iter(journal["scores"]))]


def evict_lru_to_cap(
    directory: Path, pattern: str, cap_bytes: int, incoming_bytes: int = 0,
) -> int:
    """Delete least-recently-modified files matching ``pattern`` until the directory
    holds at most ``cap_bytes``, counting ``incoming_bytes`` as already present.

    Returns the number of files deleted. Callers treat a cache miss as a re-fetch,
    so losing a file is a cost and never an error; a file that vanishes underneath
    (a concurrent replica sharing the directory) is skipped rather than raised on.
    """
    entries = []
    for path in directory.glob(pattern):
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((stat.st_mtime, stat.st_size, path))

    total = sum(size for _, size, _ in entries) + incoming_bytes
    deleted = 0
    for _, size, path in sorted(entries, key=lambda e: e[0]):
        if total <= cap_bytes:
            break
        try:
            path.unlink()
        except OSError:
            continue
        total -= size
        deleted += 1
    return deleted


def uniform_soup(
    loaders: Sequence[Callable[[], Dict[str, Any]]],
    *,
    base: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    """Key-wise uniform mean of LEARNED tensors, with non-learned values taken from
    the base checkpoint — the AXIS-1 combine rule.

    Takes one zero-arg loader per member rather than materialised state dicts, and
    accumulates member-by-member, so peak memory is the running sum plus a single
    member regardless of member count. Taking a list and looping key-outer instead
    pins every member in RAM at once: at 16 cpsam members that measured ~27 GB
    against the runtime's 12 GiB reservation and the replica was OOM-killed.

    ``base`` is a loader for the checkpoint the pool started from, and it is
    REQUIRED rather than defaulted, which is the whole point of #0070 requirement 2:
    a caller that cannot say what the base is must fail, not silently average. It is
    read once, reduced to its key set plus its non-learned values, and released
    before the accumulation loop allocates, so it costs one transient member and
    does not raise the steady-state peak.

    #0070 requirement 3, RULED 2026-11-14: a non-learned key (``NON_WEIGHT_PARAMS``,
    e.g. ``diam_labels``) is carried through from ``base`` UNCHANGED and is never
    averaged. Averaging them produced 57.7405 for a four-member pool whose real
    per-site diameters were 29.711 / 54.986 / 28.912 / 117.353 — a field asserting a
    measurement nobody made, wrong by about a factor of two for three of the four.
    The base value is the only traceable answer: it is the prior the soup inherited,
    and the soup did not learn a diameter. The four real per-site numbers are not
    discarded; they belong in the community manifest, because they are statistics
    about contributors' DATA rather than parameters of the model.

    #0070 requirement 2: classification is TOTAL and fail-closed. Every member key
    must appear in ``base``, whose key set IS the architecture's own inventory, and a
    key in neither the base nor the declared non-learned set raises rather than being
    averaged. That is the property that stops this recurring: a future non-learned
    ``nn.Parameter`` added to the architecture is a key the base of an existing pool
    does not have, so it breaks the merge loudly instead of being quietly souped into
    every community checkpoint. A name blacklist could not do this — the reason the
    original defect survived review is that the docstring asked "are there persistent
    buffers to reset?", correctly got zero, and read clean, while the thing needing
    exclusion was a non-learned *Parameter*.

    Every member must share one parameter layout (``assert_compatible`` guards this
    at contribute + aggregate time), so the mean is well-defined per key. cpsam is
    LayerNorm-only with zero persistent buffers, so there is no running statistic to
    reset; a plain arithmetic mean over the selected members is the soup. Averaging
    is done in float32 for numerical stability, then cast back to each key's original
    dtype so the community checkpoint round-trips into cellpose unchanged."""
    import torch

    n = len(loaders)
    if not n:
        raise ValueError("uniform_soup requires at least one state_dict")

    base_sd = base()
    if not base_sd:
        raise ValueError("uniform_soup base loaded an empty state_dict")
    base_keys = frozenset(base_sd)
    # Detached clones, so the returned soup never aliases the base checkpoint's
    # tensors and keeps each key's own dtype without a cast.
    carried: Dict[str, Any] = {
        key: value.detach().clone()
        for key, value in base_sd.items()
        if is_non_weight_param(key)
    }
    del base_sd

    acc: Dict[str, Any] = {}
    dtypes: Dict[str, Any] = {}
    for index, load in enumerate(loaders):
        sd = load()
        if not sd:
            raise ValueError(
                f"uniform_soup member {index} loaded an empty state_dict"
            )
        keys = frozenset(sd)
        unclassifiable = sorted(keys - base_keys)
        if unclassifiable:
            raise ValueError(
                f"uniform_soup member {index} carries {len(unclassifiable)} key(s) "
                f"absent from the base checkpoint, so they can be classified neither "
                f"learned nor non-learned and MUST NOT be averaged: "
                f"{unclassifiable[:8]}"
            )
        missing = sorted(base_keys - keys)
        if missing:
            raise ValueError(
                f"uniform_soup member {index} is missing {len(missing)} key(s) the "
                f"base checkpoint declares: {missing[:8]}"
            )
        if not index:
            dtypes = {key: value.dtype for key, value in sd.items()
                      if key not in carried}
            acc = {key: torch.zeros_like(value, dtype=torch.float32)
                   for key, value in sd.items() if key not in carried}
        for key in acc:
            acc[key] += sd[key].detach().float()
        del sd
    for key in acc:
        acc[key] = (acc[key] / n).to(dtypes[key])
    acc.update(carried)
    return acc


# ── instance-segmentation metric (the gate + witness score) ──────────────────
# Ported verbatim from the bioengine-paper analysis (single source of truth for
# every figure): the greedy descending-IoU matcher is
# analysis/scripts/utils.py:compute_instance_f1, and the O(H*W) joint-histogram
# IoU matrix is analysis/scripts/uc3_finetune_public.py:fast_iou_matrix, whose
# selftest() proves it numerically identical to utils._iou_matrix. The same
# metric+matcher scores Figs 2/5/S2 and the UC3 fine-tuning eval, so the community
# soup MUST be gated + reported with this exact code, not a reimplementation. It is
# copied (not imported) because the paper repo is not on a deployed Ray actor's
# path; keep it byte-faithful to that source.

def _iou_matrix(pred: "np.ndarray", gt: "np.ndarray") -> "np.ndarray":
    """IoU for every (pred_id, gt_id) pair via a joint histogram — rows are pred
    instances, columns GT instances (background 0 excluded). O(H*W)."""
    import numpy as np

    pred = pred.astype(np.int64, copy=False)
    gt = gt.astype(np.int64, copy=False)
    pred_ids = np.unique(pred[pred > 0])
    gt_ids = np.unique(gt[gt > 0])
    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)

    p_index = np.zeros(int(pred.max()) + 1, dtype=np.int64)
    p_index[pred_ids] = np.arange(1, len(pred_ids) + 1)
    g_index = np.zeros(int(gt.max()) + 1, dtype=np.int64)
    g_index[gt_ids] = np.arange(1, len(gt_ids) + 1)

    pi = p_index[pred].ravel()
    gi = g_index[gt].ravel()
    n_p, n_g = len(pred_ids) + 1, len(gt_ids) + 1
    hist = np.bincount(pi * n_g + gi, minlength=n_p * n_g).reshape(n_p, n_g)

    inter = hist[1:, 1:].astype(np.float64)
    p_area = hist[1:, :].sum(axis=1).astype(np.float64)[:, None]
    g_area = hist[:, 1:].sum(axis=0).astype(np.float64)[None, :]
    union = p_area + g_area - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


def instance_f1(pred_labels: "np.ndarray", gt_labels: "np.ndarray", iou_thresh: float = 0.5) -> Dict[str, Any]:
    """Instance-level F1 at IoU >= iou_thresh via greedy descending-IoU matching,
    over two 2-D integer label maps (0 = background). Returns
    ``{f1, precision, recall, tp, fp, fn, n_pred, n_gt, mean_iou}``. The gate and
    the witness curve read ``["f1"]``; mean instance F1 over a split = the mean of
    per-image f1."""
    import numpy as np

    iou = _iou_matrix(pred_labels.astype(np.int32), gt_labels.astype(np.int32))
    n_pred = iou.shape[0]
    n_gt = iou.shape[1]

    matched_ious: List[float] = []
    used_pred = set()
    used_gt = set()

    if n_pred > 0 and n_gt > 0:
        order = np.argsort(-iou.ravel())
        for idx in order:
            i, j = divmod(idx, n_gt)
            if iou[i, j] < iou_thresh:
                break
            if i in used_pred or j in used_gt:
                continue
            matched_ious.append(float(iou[i, j]))
            used_pred.add(i)
            used_gt.add(j)

    tp = len(matched_ious)
    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = float(sum(matched_ious) / n_gt) if n_gt > 0 else 0.0

    return {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "mean_iou": round(mean_iou, 4),
    }


def mean_instance_f1(pred_maps: List["np.ndarray"], gt_maps: List["np.ndarray"], iou_thresh: float = 0.5) -> float:
    """Mean of per-image instance F1 over a split — the scalar the gate compares and
    the witness curve plots. Empty split is undefined, so raise rather than invent 0."""
    if not gt_maps:
        raise ValueError("mean_instance_f1 requires at least one image")
    scores = [instance_f1(p, g, iou_thresh=iou_thresh)["f1"] for p, g in zip(pred_maps, gt_maps)]
    return float(sum(scores) / len(scores))


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_contribution_metadata(
    *,
    contribution_id: str,
    model_type: str,
    app_version: Optional[str],
    cellpose_version: str,
    architecture_signature: str,
    base_checkpoint_id: str,
    base_checkpoint_sha256: Optional[str],
    samples_seen: int,
    training_params: Dict[str, Any],
    l2_from_base: Optional[float],
    dataset_fingerprint: Optional[str],
    solo_val_metric: Optional[float],
    weights_sha256: str,
    uploaded_at: float,
) -> Dict[str, Any]:
    """A contribution's metadata.json. ``samples_seen`` and ``l2_from_base`` are
    recorded on every contribution but only become load-bearing under specific
    aggregation families (sample-weighting, drift bound) — recording them now
    keeps the pool auditable regardless of which family Nils picks.

    ``app_version`` is written unconditionally, null included: a field that is
    absent when unknown cannot be told apart from one written before versions
    were stamped at all, which is the ambiguity it exists to remove.
    ``schema_version`` does not cover it — 0.27.6 moved every ``l2_from_base``
    value without changing the record's shape.

    ``weights_sha256_basis`` says what ``weights_sha256`` is a digest OF, from
    ``CONTRIBUTION_DIGEST_BASIS``, so the number is reproducible by whoever reads
    this file. Stamped here as well as on the index entry on purpose: both records
    carry the same digest, and stamping only one would make the absence on the other
    newly ambiguous — a reader would have to know which writer ran to know whether
    "no field" means "old record" or "this writer does not stamp". Additive, so
    ``SCHEMA_VERSION`` is unchanged: no existing field moves or changes meaning."""
    return {
        "schema_version": SCHEMA_VERSION,
        "contribution_id": contribution_id,
        "model_type": model_type,
        "app_version": app_version,
        "cellpose_version": cellpose_version,
        "architecture_signature": architecture_signature,
        "base_checkpoint_id": base_checkpoint_id,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "samples_seen": samples_seen,
        "n_epochs": training_params.get("n_epochs"),
        "batch_size": training_params.get("batch_size"),
        "learning_rate": training_params.get("learning_rate"),
        "diam_mean": training_params.get("diam_mean"),
        "l2_from_base": l2_from_base,
        "dataset_fingerprint": dataset_fingerprint,
        "solo_val_metric": solo_val_metric,
        "weights_sha256": weights_sha256,
        "weights_sha256_basis": CONTRIBUTION_DIGEST_BASIS,
        "uploaded_at": uploaded_at,
    }


def empty_index(model_type: str, architecture_signature: str) -> Dict[str, Any]:
    """A fresh pool_index.json for a newly created pool artifact.

    ``baseline`` anchors the witness curve at stock cpsam (set on the first merge);
    ``merges`` is the append-only per-merge timeline — the single source for both the
    plotted witness curve and the rejected-contribution list, because a merge that
    admits nothing publishes no checkpoint yet must still record its rejections."""
    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": model_type,
        "architecture_signature": architecture_signature,
        "current_community": None,
        "contributions": [],
        "decided_contribution_ids": [],
        "baseline": None,
        "merges": [],
    }


def append_merge(index: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    """Append one merge event to the timeline. ``entry`` carries
    ``{published_version|None, baseline_version, members[{contribution_id,
    marginal_score, gate_score, included}] (admission-ordered, see
    ``community_manifest``), selection_metric (RoundMetric role=selection),
    witness_metric (RoundMetric role=witness, or None when nothing was admitted),
    global_sha256|None, transport, deferred[], created_at}``. A merge with
    ``published_version=None`` admitted nothing and projects to an empty_merge
    (selection bar only, no witness — nothing was re-measured). Append-only: past
    merge records are immutable."""
    index.setdefault("merges", []).append(entry)
    return index


def undecided_contributions(index: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Contributions that have arrived but no merge event has yet weighed — sorted
    by arrival (``uploaded_at``). Candidate order ACROSS merges is chronological, so
    a merge sees only what has arrived and is not yet decided; overflow deferred at a
    merge stays undecided and is re-considered (and re-scored) at the next one."""
    decided = set(index.get("decided_contribution_ids", []))
    pending = [c for c in index.get("contributions", []) if c.get("contribution_id") not in decided]
    return sorted(pending, key=lambda c: c.get("uploaded_at", 0.0))


def next_community_version(index: Dict[str, Any]) -> str:
    """Monotonic ``vN`` for the next community checkpoint, derived from the current
    pointer so no separate counter can drift out of sync with it."""
    cur = index.get("current_community")
    if not cur or not cur.get("version"):
        return "v1"
    try:
        return f"v{int(str(cur['version']).lstrip('v')) + 1}"
    except ValueError:
        raise ValueError(f"cannot derive next version from '{cur.get('version')}'")


def find_contribution_by_digest(
    index: Dict[str, Any], weights_sha256: str
) -> Optional[Dict[str, Any]]:
    """The contribution already holding ``weights_sha256``, or None.

    Identity is the digest ALONE — not the digest plus any metadata. What a soup
    averages is the bytes, so two members carrying one digest weight that
    checkpoint 2:1 in the uniform mean whatever metadata they carry; metadata can
    only make the double-weighting harder to see. Comparing the digest against a
    caller-supplied field as well would let a retry defeat de-duplication by
    passing a different ``dataset_fingerprint``, which a retrying client can do by
    accident.

    Deliberately does NOT consult ``decided_contribution_ids``: a re-deposit of a
    checkpoint that is already inside a published community soup is the case where
    appending is *worst*, since the next merge would compound a weight that has
    already shipped. The first member is returned there too.

    Earliest match wins, so a pool that somehow already holds a duplicate (one
    created before this check existed) resolves to the same member on every call
    rather than drifting with list order.
    """
    if not weights_sha256:
        return None
    for entry in index.get("contributions", []) or []:
        if entry.get("weights_sha256") == weights_sha256:
            return entry
    return None


def append_contribution(index: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    """Append a contribution stub to the index. ``entry`` carries the pointers +
    the audit fields a reader needs without opening each metadata.json:
    ``{contribution_id, weights_path, metadata_path, weights_sha256,
    architecture_signature, samples_seen, uploaded_at}``. Append-only: existing
    entries are immutable.

    Stamps ``weights_sha256_basis`` from ``CONTRIBUTION_DIGEST_BASIS`` — the FILE
    basis, NOT ``CURRENT_DIGEST_BASIS``, which is the community head's and would
    assert a tensor digest over a number this process produced by hashing a file.
    That is what makes the stored record attributable from itself; read it back with
    ``contribution_digest_basis``. Stamped here rather than accepted as a field on
    ``entry`` for the reason ``build_community_manifest`` records at its own stamp:
    the app that writes the record is the only thing that knows which digest it just
    computed (``prepare_contribution`` computes ``weights_sha256`` in this process,
    so the client never supplies it), and a caller-supplied basis could disagree
    with the bytes — and would be believed, since an explicit field outranks every
    other signal.

    Stamped only when the entry carries a digest. A basis on a record with no digest
    would be an assertion about a value that is not there, and ``digest_basis``
    answers ``no_digest`` for it before ever reading the field, so writing one buys
    nothing and claims something.

    Written onto a COPY, so the caller's dict is not mutated: entries already in the
    index are never touched and nothing stored is back-filled. A pre-existing record
    with no basis keeps having none and resolves by date, which is the one direction
    that is a proof rather than a guess.

    REFUSES a second member for a digest the index already holds (#0376). A client
    that retries a deposit after a gateway 504 ate the response — the correct thing
    for it to do, and 4 of 92 deposits on the r2 corpus needed it — otherwise
    becomes two members and that site is silently weighted 2:1 in the uniform mean.
    The refusal is the invariant stated where the write happens, so a later caller
    cannot re-create the defect by forgetting to look first; on the retry path
    ``contribute`` recognises the duplicate and returns before reaching here, so
    this raise is a backstop rather than the mechanism.
    """
    existing = find_contribution_by_digest(index, entry.get("weights_sha256", ""))
    if existing is not None:
        raise ValueError(
            f"contribution '{entry.get('contribution_id')}' carries weights_sha256 "
            f"{entry.get('weights_sha256')}, which this pool already holds as "
            f"'{existing.get('contribution_id')}' — appending it would weight that "
            f"checkpoint twice in the uniform soup. Recognise the re-deposit and "
            f"return the existing member instead."
        )
    stored = dict(entry)
    if stored.get("weights_sha256"):
        stored["weights_sha256_basis"] = CONTRIBUTION_DIGEST_BASIS
    index.setdefault("contributions", []).append(stored)
    return index


def build_community_manifest(
    *,
    community_version: str,
    combine: str,
    selected_contribution_ids: List[str],
    gate_split: Dict[str, Any],
    witness_split: Dict[str, Any],
    selection_metric: Dict[str, Any],
    witness_metric: Dict[str, Any],
    members: List[Dict[str, Any]],
    architecture_signature: str,
    weights_sha256: str,
    created_at: float,
) -> Dict[str, Any]:
    """A community checkpoint's manifest.json (one per ``community/<version>/``).

    ``selection_metric`` (role=selection) is the score on split A that admitted this
    soup; ``witness_metric`` (role=witness) is the score on the disjoint split B and
    is the only number the improvement curve plots — reporting the selection metric
    as the curve would be circular. Both are role-tagged RoundMetric records, so the
    role is a property of the stored artifact, not an assignment the page makes.
    ``members`` records every candidate weighed as ``{contribution_id,
    marginal_score, gate_score, included}`` for internal audit (a rejected candidate
    reads "evaluated, not included", never a failure), **in the order the gate
    weighed them** — the admitted subsequence is the order ``uniform_soup``
    accumulated in, and float addition is not associative, so reordering this list
    loses the only record of which checkpoint was published. ``marginal_score`` is
    what that ordering was sorted on (the candidate's score against the head as it
    stood); ``gate_score`` is the cumulative score of the soup including it, so the
    two are not interconvertible. Both are audit-only and stripped from the wire
    projection. ``combine`` is "uniform" unless sample-weighting was empirically
    justified for this version."""
    return {
        "schema_version": SCHEMA_VERSION,
        "community_version": community_version,
        "combine": combine,
        "selected_contribution_ids": selected_contribution_ids,
        "gate_split": gate_split,
        "witness_split": witness_split,
        "selection_metric": selection_metric,
        "witness_metric": witness_metric,
        "members": members,
        "architecture_signature": architecture_signature,
        "weights_sha256": weights_sha256,
        # Stamped from the constant rather than taken as an argument: the app that
        # writes the record is the only thing that knows which digest its own
        # materialize_soup just computed, and a caller-supplied basis could disagree
        # with the bytes. See CURRENT_DIGEST_BASIS for the lock-step rule.
        "weights_sha256_basis": CURRENT_DIGEST_BASIS,
        "created_at": created_at,
    }


def set_current_community(
    index: Dict[str, Any],
    *,
    community_version: str,
    weights_path: str,
    sha256: str,
    selection_metric: Optional[Dict[str, Any]] = None,
    witness_metric: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Point the index at a newly published community checkpoint. Written LAST by
    the aggregate step (after the version's weights + manifest are committed) so a
    reader never sees the pointer aimed at a half-written version. Carries both
    role-tagged metrics so the page reads the reportable (witness) curve without
    opening each manifest; ``witness_metric`` is the plotted number, ``selection_metric``
    is internal."""
    index["current_community"] = {
        "version": community_version,
        "path": weights_path,
        "sha256": sha256,
        # This pointer is a copy of the merge's digest and carries no created_at of
        # its own, so without this stamp it is the one surface of the three that
        # cannot be attributed from the record alone. Written unconditionally.
        "sha256_basis": CURRENT_DIGEST_BASIS,
        "selection_metric": selection_metric,
        "witness_metric": witness_metric,
    }
    return index


def current_community(index: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return index.get("current_community")


def assert_compatible(index: Dict[str, Any], *, model_type: str, signature: str) -> None:
    """Refuse to read/write a pool whose model_type or parameter layout does not
    match the caller's — the guard that keeps a cpdino contribution out of the
    cpsam pool and flags a cellpose-revision drift before averaging."""
    idx_model = index.get("model_type")
    if idx_model != model_type:
        raise ValueError(
            f"pool model_type '{idx_model}' does not match '{model_type}'."
        )
    idx_sig = index.get("architecture_signature")
    if idx_sig and idx_sig != signature:
        raise ValueError(
            "architecture signature mismatch: this checkpoint's parameter layout "
            f"({signature[:12]}…) differs from the pool's ({idx_sig[:12]}…); "
            "averaging is only defined over an identical layout."
        )


# ── wire projection ──────────────────────────────────────────────────────────
# The backend stores pool_index.json in its own shape; the campaign page reads the
# subset below. This projection is where the page's data model is produced: it
# partitions the merge timeline into published soups and empty merges, strips the
# audit-only per-member scores (a per-candidate score in the wire payload is a
# contract violation — assessed is IDs only), and surfaces the role-tagged metrics
# straight from storage. Pure and torch-free so the contract is unit-testable.

def _iso(ts: Optional[float]) -> Optional[str]:
    """Epoch seconds → the page's ISO-8601 UTC millisecond format, or None."""
    import datetime

    if ts is None:
        return None
    dt = datetime.datetime.fromtimestamp(float(ts), tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _current_community_provenance(index: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Attribute ``current_community.sha256``, which carries no timestamp of its own.

    The pointer is a copy of one merge's digest written by ``set_current_community``,
    and before this change it recorded neither a basis nor a ``created_at`` — so it is
    the one surface of the three that cannot be dated from itself. Borrow both from
    the merge that published this version. If no merge matches, nothing is borrowed
    and ``digest_basis`` reaches its own undetermined branch; the join is never
    allowed to invent a basis the timeline does not support.
    """
    cur = index.get("current_community")
    if not isinstance(cur, dict):
        return None
    record = dict(cur)
    if record.get("sha256_basis") is None or record.get("created_at") is None:
        for merge in index.get("merges") or []:
            published = merge.get("published_version")
            if published and published == record.get("version"):
                if record.get("created_at") is None:
                    record["created_at"] = merge.get("created_at")
                if record.get("sha256_basis") is None:
                    record["sha256_basis"] = merge.get("global_sha256_basis")
                break
    return digest_provenance(record, digest_key="sha256")


def _assessed_ids(merge: Dict[str, Any]) -> List[str]:
    return [m["contribution_id"] for m in merge.get("members", [])]


def _taken_ids(merge: Dict[str, Any]) -> List[str]:
    return [m["contribution_id"] for m in merge.get("members", []) if m.get("included")]


def corrected_metric(metric: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A metric record whose ``aggregate_basis`` is retracted wording, restated — or
    the record exactly as given.

    Returns a **copy** when it corrects and the **same object** when it does not, so
    "was this corrected?" is answerable by identity (``out is metric``) rather than by
    re-comparing strings. Never mutates the argument: the stored index is the
    historical record of what the campaign published, and this is the read path.

    Anything that is not a dict carrying a known-retracted string passes through
    untouched, including ``None`` — an empty or absent metric is a normal state here
    (``witness_metric`` is None on a merge that admitted nothing) and must not become
    an error on a path whose job is to render.
    """
    if not isinstance(metric, dict):
        return metric
    stored = metric.get("aggregate_basis")
    if not isinstance(stored, str):
        return metric
    current = _RETRACTED_BASIS.get(stored)
    if current is None:
        return metric
    out = dict(metric)
    out["aggregate_basis"] = f"{current} {BASIS_CORRECTION_NOTE}"
    return out


def _corrected_community(community: Any) -> Any:
    """``current_community`` with its two metric records corrected.

    This one is easy to miss: ``current_community`` is projected through
    ``pool_wire_state`` verbatim, so it carries its own ``selection_metric`` and
    ``witness_metric`` that are NOT the merge records corrected alongside the soups.
    Four of the six live pools have one.
    """
    if not isinstance(community, dict):
        return community
    out = dict(community)
    changed = False
    for field in ("selection_metric", "witness_metric"):
        before = community.get(field)
        after = corrected_metric(before)
        if after is not before:
            out[field] = after
            changed = True
    return out if changed else community


def pool_wire_state(index: Dict[str, Any], *, pool_artifact_id: str) -> Dict[str, Any]:
    """Project a stored pool_index.json onto the campaign-page wire contract.

    Returns the subset this backend owns — ``merge_trigger``, ``baseline_metric``
    (role=witness), ``soups[]`` (published merges), ``empty_merges[]`` (merges that
    admitted nothing: selection bar only, no witness), and enriched
    ``contributions[]`` stubs (disposition + which soup they merged into). Campaign-
    registry metadata the page owns (contributor demographics, licences, stewards,
    description) is composed on the page, not here.

    A contribution's ``weights_sha256`` is published because ``contribute()`` returns
    it exactly once: a deposit whose response is lost — measured at 4 of 92 against a
    1.2 GiB PUT — otherwise leaves the client unable to tell a landed upload from one
    it must retry, and a retry duplicates the member and silently weights that site
    2:1 in the uniform mean. Reading it back makes that question answerable."""
    alias = pool_artifact_id.split("/")[-1]
    merges = index.get("merges", [])

    soups: List[Dict[str, Any]] = []
    # Which of the two digests each published head's global_sha256 is. Kept BESIDE
    # soups[] rather than inside it: the soup record's field set is the page's
    # 0.13.0-draft contract, pinned key-for-key against its fixture corpus
    # (test_pool.py::test_soup_key_set), and a backend that quietly widens it is the
    # mismatch SCHEMA_VERSION exists to make visible. This is an additive top-level
    # key the page can adopt without a draft bump, joined on soup_id.
    digest_provenance_by_soup: Dict[str, Any] = {}
    empty_merges: List[Dict[str, Any]] = []
    merged_into: Dict[str, str] = {}
    cumulative = 0
    for merge in merges:
        transport = merge.get("transport") or {
            "bytes_in": 0, "bytes_out": 0, "n_transfers": 0, "sources_complete": True,
        }
        if merge.get("published_version"):
            soup_id = f"soup-{len(soups)}"
            digest_provenance_by_soup[soup_id] = digest_provenance(merge)
            taken = _taken_ids(merge)
            cumulative += len(taken)
            for cid in taken:
                merged_into[cid] = soup_id
            version = merge["published_version"]
            soups.append({
                "soup_id": soup_id,
                "index": len(soups),
                "merged_at": _iso(merge.get("created_at")),
                "assessed": _assessed_ids(merge),
                "contributions": taken,
                "selection_metric": corrected_metric(merge.get("selection_metric")),
                "witness_metric": corrected_metric(merge.get("witness_metric")),
                "global_sha256": merge.get("global_sha256"),
                "weights": None,
                "transport": transport,
                "community_model": {
                    "artifact_id": pool_artifact_id,
                    "version": version,
                    "url": f"#/models/{alias}?version={version}",
                    "n_contributions_cumulative": cumulative,
                },
            })
        else:
            empty_merges.append({
                "merged_at": _iso(merge.get("created_at")),
                "assessed": _assessed_ids(merge),
                "selection_metric": corrected_metric(merge.get("selection_metric")),
                "transport": transport,
            })

    decided = set(index.get("decided_contribution_ids", []))
    contributions = []
    for c in index.get("contributions", []):
        cid = c.get("contribution_id")
        if cid in merged_into:
            disposition = "included"
        elif cid in decided:
            disposition = "excluded"
        else:
            disposition = "pending"
        contributions.append({
            "contribution_id": cid,
            "received_at": _iso(c.get("uploaded_at")),
            "weights_sha256": c.get("weights_sha256"),
            # Published under the storage name on purpose: soup_admission fails
            # open on a missing n_bytes, so the audit question is "did the pool
            # record n_bytes?" and the view has to answer it under that word.
            # bytes_out stays reserved for transport, one level up.
            "n_bytes": c.get("n_bytes"),
            "n_train_images": c.get("n_train_images"),
            "samples_seen": c.get("samples_seen"),
            "disposition": disposition,
            "merged_into": merged_into.get(cid),
        })

    baseline = index.get("baseline") or {}
    # Two independent corrections land on this one projection and both have to survive:
    # _corrected_community (#0418) rewrites the pointer's own selection_metric and
    # witness_metric, which are NOT the merge records corrected alongside the soups,
    # while sha256_provenance (#0417) says which of the two digests the pointer's sha256
    # is. They are orthogonal, and the order between them is interchangeable — measured,
    # not assumed: reversing these two statements leaves the app suite identical at 229
    # passed, 2 skipped, because _corrected_community copies with dict(community) and so
    # carries an already-attributed key through. The order below is the one that needs a
    # single isinstance guard: _corrected_community returns a non-dict unchanged, so
    # correcting first lets one guard cover the spread, whereas attributing first needs
    # its own guard before the spread and another pass afterwards.
    current_community = _corrected_community(index.get("current_community"))
    if isinstance(current_community, dict):
        current_community = {
            **current_community,
            "sha256_provenance": _current_community_provenance(index),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": index.get("model_type"),
        "architecture_signature": index.get("architecture_signature"),
        "current_community": current_community,
        "merge_trigger": merge_trigger(),
        "baseline_metric": corrected_metric(baseline.get("witness_metric")),
        "soups": soups,
        "empty_merges": empty_merges,
        "contributions": contributions,
        # Additive, outside every field set the page pins. ``soups`` is keyed by
        # soup_id; empty merges are absent from it because they publish no head and
        # their projection carries no digest to attribute.
        "digest_provenance": {
            "soups": digest_provenance_by_soup,
            "current_community": _current_community_provenance(index),
        },
    }
