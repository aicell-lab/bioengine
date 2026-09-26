"""Async model-soup pool — storage layout + record semantics (backend-owned).

The community pool is a Hypha artifact (one per model family; cpsam first) that
holds contributed weights, per-contribution metadata, and versioned community
checkpoints. Contributors fine-tune on their own local data at their own pace and
upload only weights — the data never moves. A coordinator periodically averages
selected contributions into a new community checkpoint (the ``aggregate`` step,
which lives in the entry because it does torch + artifact I/O; the pure combine
math — ``uniform_soup`` — and the community schema live here).

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

# Bumped when the shape of pool_index.json / metadata.json / manifest.json
# changes. Echoed on every record and every read method so the page can flag a
# contract mismatch instead of silently mis-rendering.
SCHEMA_VERSION = "1"

POOL_INDEX = "pool_index.json"
WEIGHTS_NAME = "weights.pt"
METADATA_NAME = "metadata.json"
MANIFEST_NAME = "manifest.json"


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
    gate_score, included}], gate_metric, witness_metric, deferred[], created_at}``.
    Append-only: past merge records are immutable."""
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
    gate_metric: Optional[float],
    witness_metric: Optional[float],
    metric_name: str,
    members: List[Dict[str, Any]],
    architecture_signature: str,
    weights_sha256: str,
    created_at: float,
) -> Dict[str, Any]:
    """A community checkpoint's manifest.json (one per ``community/<version>/``).

    ``gate_metric`` is the score on split A that admitted this soup (internal /
    selection); ``witness_metric`` is the score on the disjoint split B and is the
    only number the paper's improvement curve plots — reporting the gate metric as
    the curve would be circular. ``members`` records every candidate weighed at this
    merge as ``{contribution_id, gate_score, included}`` so a rejected contribution
    reads as "evaluated, not included" rather than a failure. ``combine`` is
    "uniform" unless sample-weighting was empirically justified for this version."""
    return {
        "schema_version": SCHEMA_VERSION,
        "community_version": community_version,
        "combine": combine,
        "metric_name": metric_name,
        "selected_contribution_ids": selected_contribution_ids,
        "gate_split": gate_split,
        "witness_split": witness_split,
        "gate_metric": gate_metric,
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
    gate_metric: Optional[float] = None,
    witness_metric: Optional[float] = None,
) -> Dict[str, Any]:
    """Point the index at a newly published community checkpoint. Written LAST by
    the aggregate step (after the version's weights + manifest are committed) so a
    reader never sees the pointer aimed at a half-written version. Carries both
    metrics so the page reads the reportable (witness) curve without opening each
    manifest; ``witness_metric`` is the plotted number, ``gate_metric`` is internal."""
    index["current_community"] = {
        "version": community_version,
        "path": weights_path,
        "sha256": sha256,
        "gate_metric": gate_metric,
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
