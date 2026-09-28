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
from typing import Any, Dict, List, Optional

# Tracks the campaign-page wire contract this backend maps onto. Echoed on every
# record and every read method so the page can flag a backend/page mismatch
# instead of silently mis-rendering. Bump in lock-step with the page's fixtures.
SCHEMA_VERSION = "0.13.0-draft"

POOL_INDEX = "pool_index.json"
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
SELECTION_BASIS = (
    "the score the greedy gate admitted contributions against, on split A, a public "
    "benchmark disjoint from every contributor's training data"
)
WITNESS_BASIS = (
    "scored on the disjoint witness split B, held back from the selection split and "
    "from every contributor, the only number the improvement curve plots"
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


def state_dict_l2_from_base(current: Dict[str, Any], base: Dict[str, Any]) -> float:
    """L2 distance between two state_dicts over shared floating-point params — how
    far a contribution's weights moved from the checkpoint it started from. A drift
    proxy recorded on every contribution; its use as an acceptance BOUND is a
    pending aggregation decision, so this only measures, it does not gate."""
    import torch

    total = 0.0
    for key, cur in current.items():
        ref = base.get(key)
        if ref is None or not torch.is_floating_point(cur) or cur.shape != ref.shape:
            continue
        total += float(torch.sum((cur.detach().float() - ref.detach().float()) ** 2).item())
    return total ** 0.5


def uniform_soup(state_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Key-wise uniform mean of full weight tensors — the AXIS-1 combine rule.

    Every member must share one parameter layout (``assert_compatible`` guards this
    at contribute + aggregate time), so the mean is well-defined per key. cpsam is
    LayerNorm-only with zero persistent buffers, so there is no running statistic to
    reset; a plain arithmetic mean over the selected members is the soup. Averaging
    is done in float32 for numerical stability, then cast back to each key's original
    dtype so the community checkpoint round-trips into cellpose unchanged."""
    import torch

    if not state_dicts:
        raise ValueError("uniform_soup requires at least one state_dict")
    keys = list(state_dicts[0].keys())
    out: Dict[str, Any] = {}
    n = len(state_dicts)
    for key in keys:
        ref = state_dicts[0][key]
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for sd in state_dicts:
            acc += sd[key].detach().float()
        out[key] = (acc / n).to(ref.dtype)
    return out


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
    keeps the pool auditable regardless of which family Nils picks."""
    return {
        "schema_version": SCHEMA_VERSION,
        "contribution_id": contribution_id,
        "model_type": model_type,
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
    gate_score, included}], selection_metric (RoundMetric role=selection),
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


def append_contribution(index: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    """Append a contribution stub to the index. ``entry`` carries the pointers +
    the audit fields a reader needs without opening each metadata.json:
    ``{contribution_id, weights_path, metadata_path, weights_sha256,
    architecture_signature, samples_seen, uploaded_at}``. Append-only: existing
    entries are immutable."""
    index.setdefault("contributions", []).append(entry)
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
    ``members`` records every candidate weighed as ``{contribution_id, gate_score,
    included}`` for internal audit (a rejected candidate reads "evaluated, not
    included", never a failure); the per-member gate_score is audit-only and is
    stripped from the wire projection. ``combine`` is "uniform" unless sample-
    weighting was empirically justified for this version."""
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
# audit-only per-member gate_score (a per-candidate score in the wire payload is a
# contract violation — assessed is IDs only), and surfaces the role-tagged metrics
# straight from storage. Pure and torch-free so the contract is unit-testable.

def _iso(ts: Optional[float]) -> Optional[str]:
    """Epoch seconds → the page's ISO-8601 UTC millisecond format, or None."""
    import datetime

    if ts is None:
        return None
    dt = datetime.datetime.fromtimestamp(float(ts), tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _assessed_ids(merge: Dict[str, Any]) -> List[str]:
    return [m["contribution_id"] for m in merge.get("members", [])]


def _taken_ids(merge: Dict[str, Any]) -> List[str]:
    return [m["contribution_id"] for m in merge.get("members", []) if m.get("included")]


def pool_wire_state(index: Dict[str, Any], *, pool_artifact_id: str) -> Dict[str, Any]:
    """Project a stored pool_index.json onto the campaign-page wire contract.

    Returns the subset this backend owns — ``merge_trigger``, ``baseline_metric``
    (role=witness), ``soups[]`` (published merges), ``empty_merges[]`` (merges that
    admitted nothing: selection bar only, no witness), and enriched
    ``contributions[]`` stubs (disposition + which soup they merged into). Campaign-
    registry metadata the page owns (contributor demographics, licences, stewards,
    description) is composed on the page, not here."""
    alias = pool_artifact_id.split("/")[-1]
    merges = index.get("merges", [])

    soups: List[Dict[str, Any]] = []
    empty_merges: List[Dict[str, Any]] = []
    merged_into: Dict[str, str] = {}
    cumulative = 0
    for merge in merges:
        transport = merge.get("transport") or {
            "bytes_in": 0, "bytes_out": 0, "n_transfers": 0, "sources_complete": True,
        }
        if merge.get("published_version"):
            soup_id = f"soup-{len(soups)}"
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
                "selection_metric": merge.get("selection_metric"),
                "witness_metric": merge.get("witness_metric"),
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
                "selection_metric": merge.get("selection_metric"),
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
            "bytes_out": c.get("n_bytes"),
            "n_train_images": c.get("n_train_images"),
            "samples_seen": c.get("samples_seen"),
            "disposition": disposition,
            "merged_into": merged_into.get(cid),
        })

    baseline = index.get("baseline") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": index.get("model_type"),
        "architecture_signature": index.get("architecture_signature"),
        "current_community": index.get("current_community"),
        "merge_trigger": merge_trigger(),
        "baseline_metric": baseline.get("witness_metric"),
        "soups": soups,
        "empty_merges": empty_merges,
        "contributions": contributions,
    }
