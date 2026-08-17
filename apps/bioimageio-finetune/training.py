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


def session_dir(session_id: str) -> Path:
    return sessions_root() / session_id


def checkpoint_path(session_id: str) -> Path:
    # torch_em DefaultTrainer writes <save_root>/checkpoints/<name>/best.pt
    return session_dir(session_id) / "checkpoints" / session_id / "best.pt"


def _status_path(session_id: str) -> Path:
    return session_dir(session_id) / "status.json"


def _stop_path(session_id: str) -> Path:
    return session_dir(session_id) / "stop.requested"


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


def write_status(session_id: str, **fields) -> Dict[str, Any]:
    """Atomically merge fields into the session's status.json."""
    p = _status_path(session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = read_status(session_id)
    cur.update(fields)
    cur["updated_at"] = time.time()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cur))
    os.replace(tmp, p)
    return cur


def get_status(session_id: str) -> Dict[str, Any]:
    st = read_status(session_id)
    if not st:
        return {"session_id": session_id, "status": "UNKNOWN", "message": "no such session"}
    st["session_id"] = session_id
    st["checkpoint_available"] = checkpoint_path(session_id).exists()
    if st.get("start_time"):
        end = st.get("end_time") or time.time()
        st["elapsed_s"] = round(end - st["start_time"], 1)
    # A run with no live thread that hasn't updated in a while is stale → stopped.
    if st.get("status") in ("PREPARING", "TRAINING") and \
            time.time() - st.get("updated_at", 0) > STATUS_STALE_SECONDS:
        st["status"] = "STOPPED"
        st["message"] = "no status update within the stale window"
    return st


def list_sessions() -> Dict[str, Dict[str, Any]]:
    root = sessions_root()
    out: Dict[str, Dict[str, Any]] = {}
    for d in sorted(root.iterdir()) if root.exists() else []:
        if d.is_dir() and (d / "status.json").exists():
            out[d.name] = get_status(d.name)
    return out


def request_stop(session_id: str) -> None:
    if session_dir(session_id).exists():
        _stop_path(session_id).write_text("stop")


def stop_requested(session_id: str) -> bool:
    return _stop_path(session_id).exists()


# === annotation interpretation ===

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
        polygons = [coords] if gtype == "Polygon" else (list(coords) if gtype == "MultiPolygon" else [])
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
        return [(im, lb) for im, lb in pairs if lb is not None and int(np.asarray(lb).max()) > 0]

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
            lb = np.asarray(lb).astype(np.uint16)
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
        "train_images": train_imgs, "train_labels": train_lbls,
        "val_images": val_imgs, "val_labels": val_lbls,
        "patch_shape": ph, "n_train": len(train_imgs), "n_val": len(val_imgs),
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
