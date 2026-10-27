"""
Cell Morphology Search Engine — self-contained Ray Serve deployment.

All helper code (normalizer, embedder, index_manager, ingestion) is inlined here
because BioEngine only executes the single entry-point file via exec().

Features:
  1. Auto-ingest preconfigured datasets on first launch
  2. Dataset registry — add Zarr/JUMP-CP datasets by URL
  3. Real-time streaming ingestion progress (polled by UI every 1s)
  4. DINOv2 ViT-B/14 embedding + FAISS similarity search (<100ms)
  5. UMAP visualisation of the full embedding space
  6. Drag-and-drop web UI
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

logger = logging.getLogger("CellImageSearch")


def _try_import(module_name: str) -> bool:
    """Return True if the module can be imported."""
    import importlib
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

# ===========================================================================
# NORMALIZER — microscopy image normalization
# ===========================================================================

# JUMP Cell Painting channel order (0-based)
# ch1=DNA(DAPI)=0, ch2=ER=1, ch3=RNA(SYTO)=2, ch4=AGP=3, ch5=Mito=4
_JUMP_CH_DNA = 0
_JUMP_CH_ER = 1
_JUMP_CH_AGP = 3

# Standard Cell Painting RGB composite: R=AGP, G=ER, B=DNA
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _percentile_stretch(img: np.ndarray, plow: float = 1.0, phigh: float = 99.0) -> np.ndarray:
    lo = np.percentile(img, plow)
    hi = np.percentile(img, phigh)
    if hi <= lo:
        hi = lo + 1.0
    s = (img.astype(np.float32) - lo) / (hi - lo)
    return (np.clip(s, 0.0, 1.0) * 255.0).astype(np.uint8)


def _to_rgb_uint8(
    img: np.ndarray,
    rgb_channels: list[int] | None = None,
    plow: float = 1.0,
    phigh: float = 99.0,
) -> np.ndarray:
    """Convert any microscopy image to (H, W, 3) uint8 RGB."""
    if img.ndim == 3 and img.shape[0] <= 7 and img.shape[0] < img.shape[2]:
        img = np.moveaxis(img, 0, -1)  # (C, H, W) → (H, W, C)

    if img.ndim == 2:
        ch = _percentile_stretch(img, plow, phigh)
        return np.stack([ch, ch, ch], axis=-1)

    n_ch = img.shape[-1]
    if rgb_channels is not None:
        selected = np.stack([img[..., c] for c in rgb_channels], axis=-1)
    elif n_ch == 1:
        ch = _percentile_stretch(img[..., 0], plow, phigh)
        return np.stack([ch, ch, ch], axis=-1)
    elif n_ch == 2:
        selected = np.stack([
            img[..., 0], img[..., 1],
            ((img[..., 0].astype(np.float32) + img[..., 1]) / 2).astype(img.dtype)
        ], axis=-1)
    elif n_ch == 3:
        selected = img
    elif n_ch >= 5:
        selected = np.stack([img[..., _JUMP_CH_AGP], img[..., _JUMP_CH_ER], img[..., _JUMP_CH_DNA]], axis=-1)
    else:
        selected = img[..., :3]

    rgb = np.zeros((*selected.shape[:2], 3), dtype=np.uint8)
    for c in range(3):
        rgb[..., c] = _percentile_stretch(selected[..., c], plow, phigh)
    return rgb


def _to_dinov2_tensor(img_rgb_uint8: np.ndarray, size: int = 224):
    """Convert (H, W, 3) uint8 RGB to normalised torch float32 tensor (1, 3, H, W)."""
    import torch
    from PIL import Image as _PIL_Image
    pil = _PIL_Image.fromarray(img_rgb_uint8, mode="RGB").resize((size, size), _PIL_Image.BICUBIC)
    arr = np.array(pil, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode bytes (TIFF/PNG/JPG) into a numpy array."""
    import io
    try:
        import tifffile
        return tifffile.imread(io.BytesIO(data))
    except Exception:
        pass
    from PIL import Image as _PIL_Image
    return np.array(_PIL_Image.open(io.BytesIO(data)))


# ===========================================================================
# EMBEDDER — DINOv2 ViT-B/14
# ===========================================================================

class _DINOv2Embedder:
    """DINOv2 ViT-B/14 wrapper (768-dim, L2-normalised output).

    Falls back to ViT-S/14 on CPU if ViT-B is too slow; override via MODEL_NAME.
    """

    MODEL_NAME = "dinov2_vitb14"
    EMBED_DIM = 768

    def __init__(self, device: str = "cuda", fp16: bool = True) -> None:
        self.device = device
        self.fp16 = fp16
        self._model: Any = None

    def load(self) -> None:
        import torch
        logger.info("Loading DINOv2 %s on %s (fp16=%s)", self.MODEL_NAME, self.device, self.fp16)
        model = torch.hub.load("facebookresearch/dinov2", self.MODEL_NAME, pretrained=True)
        model.eval()
        if self.fp16 and self.device != "cpu":
            model = model.half()
        model = model.to(self.device)
        if self.device == "cpu":
            try:
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Applied INT8 dynamic quantization (CPU speedup)")
            except Exception as qe:
                logger.warning("INT8 quantization failed (%s), using FP32.", qe)
        self._model = model
        logger.info("DINOv2 loaded, embed_dim=%d", self.EMBED_DIM)

    def embed_batch(self, images_rgb: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        import torch
        if self._model is None:
            self.load()
        all_embeddings = []
        for i in range(0, len(images_rgb), batch_size):
            batch_imgs = images_rgb[i:i + batch_size]
            tensors = [_to_dinov2_tensor(img) for img in batch_imgs]
            batch = torch.cat(tensors, dim=0).to(self.device)
            if self.fp16 and self.device != "cpu":
                batch = batch.half()
            with torch.no_grad():
                feats = self._model(batch).float().cpu().numpy()
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            all_embeddings.append(feats / np.maximum(norms, 1e-9))
        return np.vstack(all_embeddings).astype(np.float32)

    def embed_single(self, image_rgb: np.ndarray) -> np.ndarray:
        return self.embed_batch([image_rgb])[0]


# ===========================================================================
# INDEX MANAGER — FAISS
# ===========================================================================

def _index_dir(workspace_dir: str) -> Path:
    return Path(workspace_dir) / "cell_search"


def _build_index(
    embeddings: np.ndarray,
    metadata_df: Any,
    workspace_dir: str,
    n_cells_total: int | None = None,
) -> dict[str, Any]:
    import faiss
    t0 = time.time()
    n, d = embeddings.shape
    n_target = n_cells_total or n
    out_dir = _index_dir(workspace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if n_target < 100_000:
        index = faiss.IndexFlatIP(d)
        index_type = "FlatIP"
    elif n_target < 5_000_000:
        nlist = min(4096, max(64, int(np.sqrt(n_target))))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings[:min(n, 256 * nlist)])
        index_type = f"IVFFlat(nlist={nlist})"
    else:
        nlist = min(65536, max(4096, int(np.sqrt(n_target))))
        m, nbits = 96, 8  # 768/96=8 ✓
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        train_n = min(n, max(256 * nlist, 1_000_000))
        index.train(embeddings[:train_n])
        index_type = f"IVFPQ(nlist={nlist},m={m})"

    index.add(embeddings)
    elapsed = time.time() - t0

    index_path = out_dir / "cell_search.index"
    faiss.write_index(index, str(index_path))
    metadata_df.to_parquet(out_dir / "metadata.parquet", index=False)

    stats = {
        "n_cells": n,
        "embed_dim": d,
        "index_type": index_type,
        "index_size_mb": index_path.stat().st_size / 1024**2,
        "build_seconds": elapsed,
        "build_time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "index_info.json").write_text(json.dumps(stats, indent=2))
    logger.info("Built FAISS index: n=%d type=%s in %.1fs", n, index_type, elapsed)
    return stats


def _load_index(workspace_dir: str):
    import sys, os
    # Ensure pip_packages (installed during ingestion) are on path
    _pip_dir = str(Path(workspace_dir) / "pip_packages")
    if os.path.isdir(_pip_dir) and _pip_dir not in sys.path:
        sys.path.insert(0, _pip_dir)
    import faiss
    import pandas as pd
    out_dir = _index_dir(workspace_dir)
    index_path = out_dir / "cell_search.index"
    if not index_path.exists():
        raise FileNotFoundError(f"No FAISS index at {index_path}")
    index = faiss.read_index(str(index_path))
    if hasattr(index, "nprobe"):
        index.nprobe = 64
    meta_path = out_dir / "metadata.parquet"
    metadata_df = pd.read_parquet(meta_path) if meta_path.exists() else None
    info_path = out_dir / "index_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    logger.info("Loaded index: %d vectors, type=%s", index.ntotal, info.get("index_type"))
    return index, metadata_df, info


def _search_index(
    index: Any,
    metadata_df: Any,
    query_embedding: np.ndarray,
    top_k: int = 20,
) -> list[dict]:
    query = query_embedding.reshape(1, -1).astype(np.float32)
    scores, indices = index.search(query, top_k)
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0:
            continue
        meta = {}
        if metadata_df is not None and idx < len(metadata_df):
            meta = metadata_df.iloc[idx].to_dict()
        results.append({"rank": rank + 1, "score": float(score), "faiss_idx": int(idx), **meta})
    return results


def _compute_umap(
    workspace_dir: str,
    n_samples: int = 10_000,
    random_state: int = 42,
    force_recompute: bool = False,
    color_by: str = "compound",
) -> dict[str, Any]:
    import sys, os
    _pip_dir = str(Path(workspace_dir) / "pip_packages")
    if os.path.isdir(_pip_dir) and _pip_dir not in sys.path:
        sys.path.insert(0, _pip_dir)
    import faiss
    import pandas as pd
    out_dir = _index_dir(workspace_dir)
    # Per-color_by cache so switching doesn't require full recompute of PCA/UMAP
    safe_cb = color_by.replace("/", "_")
    cache_path = out_dir / f"umap_cache_{safe_cb}.npz"

    if cache_path.exists() and not force_recompute:
        data = np.load(cache_path, allow_pickle=True)
        return {
            "x": data["x"].tolist(), "y": data["y"].tolist(),
            "labels": data["labels"].tolist(), "colors": data["colors"].tolist(),
            "n_total": int(data["n_total"]),
        }

    index_path = out_dir / "cell_search.index"
    if not index_path.exists():
        return {"x": [], "y": [], "labels": [], "colors": [], "n_total": 0}

    index = faiss.read_index(str(index_path))
    n_total = index.ntotal
    n_samples = min(n_samples, n_total)
    rng = np.random.default_rng(random_state)
    sample_idx = np.sort(rng.choice(n_total, size=n_samples, replace=False))

    try:
        vecs = index.reconstruct_batch(sample_idx.tolist())
    except Exception:
        vecs = np.vstack([index.reconstruct(int(i)) for i in sample_idx])

    try:
        from umap import UMAP
        logger.info("Computing UMAP on %d vectors…", n_samples)
        t0 = time.time()
        coords = UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                      random_state=random_state).fit_transform(vecs)
        logger.info("UMAP done in %.1fs", time.time() - t0)
    except ImportError:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2).fit_transform(vecs)

    labels = ["unknown"] * n_samples
    colors = ["#888888"] * n_samples
    meta_path = out_dir / "metadata.parquet"
    if meta_path.exists():
        df = pd.read_parquet(meta_path)
        # Use requested color_by column; fall back to compound, then moa_class
        if color_by in df.columns:
            col = color_by
        elif "compound" in df.columns:
            col = "compound"
        elif "moa_class" in df.columns:
            col = "moa_class"
        else:
            col = None
        if col:
            unique_labels = df[col].unique().tolist()
            palette = _color_palette(len(unique_labels))
            cmap = {lbl: palette[i % len(palette)] for i, lbl in enumerate(unique_labels)}
            for i, idx in enumerate(sample_idx):
                if idx < len(df):
                    lbl = str(df.iloc[idx][col]) if col in df.columns else "unknown"
                    labels[i] = lbl
                    colors[i] = cmap.get(lbl, "#888888")

    result = {
        "x": coords[:, 0].tolist(), "y": coords[:, 1].tolist(),
        "labels": labels, "colors": colors, "n_total": n_total,
    }
    np.savez(cache_path, x=coords[:, 0], y=coords[:, 1],
             labels=np.array(labels), colors=np.array(colors), n_total=np.array(n_total))
    return result


def _color_palette(n: int) -> list[str]:
    base = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
        "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
        "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
    ]
    if n <= len(base):
        return base[:n]
    import colorsys
    extra = list(base)
    for i in range(n - len(base)):
        r, g, b = colorsys.hsv_to_rgb(i / (n - len(base)), 0.7, 0.85)
        extra.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return extra


# ===========================================================================
# INGESTION STATUS
# ===========================================================================

class IngestionStatus(str, Enum):
    WAITING = "waiting"
    PREPARING = "preparing"
    RUNNING = "running"
    BUILDING_INDEX = "building_index"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


def _session_dir(workspace_dir: str, session_id: str) -> Path:
    return Path(workspace_dir) / "sessions" / session_id


def write_status(
    workspace_dir: str,
    session_id: str,
    status: IngestionStatus,
    message: str,
    n_embedded: int = 0,
    n_total: int = 0,
    throughput_per_sec: float = 0.0,
    elapsed_seconds: float = 0.0,
    dataset_name: str = "",
    log_lines: list[str] | None = None,
    **extra: Any,
) -> None:
    path = _session_dir(workspace_dir, session_id) / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            pass
    prev_log = existing.get("log_tail", [])
    if log_lines:
        prev_log = (prev_log + log_lines)[-20:]
    data = {
        **existing,
        "status": status.value,
        "message": message,
        "dataset_name": dataset_name or existing.get("dataset_name", ""),
        "n_embedded": n_embedded,
        "n_total": n_total,
        "progress_pct": round(100.0 * n_embedded / max(n_total, 1), 1),
        "throughput_per_sec": round(throughput_per_sec, 1),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "eta_seconds": round((n_total - n_embedded) / max(throughput_per_sec, 0.1)),
        "log_tail": prev_log,
        "updated_at": time.time(),
        **extra,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def read_status(workspace_dir: str, session_id: str) -> dict:
    path = _session_dir(workspace_dir, session_id) / "status.json"
    if not path.exists():
        return {"status": IngestionStatus.WAITING.value, "message": "Not started"}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"status": "unknown", "message": "Error reading status"}


def is_stop_requested(workspace_dir: str, session_id: str) -> bool:
    return (_session_dir(workspace_dir, session_id) / "stop_requested").exists()


def request_stop(workspace_dir: str, session_id: str) -> None:
    p = _session_dir(workspace_dir, session_id) / "stop_requested"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("1")


# ===========================================================================
# DATASET LISTING
# ===========================================================================

def _rowcol_to_alpha(well_str: str) -> str:
    """Convert JUMP well format 'r01c29' to plate-map format 'A29'."""
    try:
        row = int(well_str[1:3])
        col = int(well_str[4:6])
        return chr(ord("A") + row - 1) + f"{col:02d}"
    except Exception:
        return well_str


def _fetch_jump_compound_names() -> dict:
    """Download JUMP perturbation control names (tiny file, 740 bytes).

    Returns dict: JCP2022_ID → compound_name for the 18 control compounds.
    """
    import urllib.request, io
    import pandas as pd
    url = (
        "https://raw.githubusercontent.com/jump-cellpainting/datasets"
        "/main/metadata/perturbation_control.csv"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            df = pd.read_csv(io.BytesIO(resp.read()))
        return dict(zip(df["Metadata_JCP2022"], df["Metadata_Name"]))
    except Exception:
        return {}


def _fetch_jump_well_lookup() -> dict:
    """Download JUMP well→compound lookup from GitHub (well.csv.gz, ~5.7 MB).

    Returns dict keyed by (source, plate_id, well_alpha) → compound name
    (JCP2022 ID resolved to common name for controls, raw JCP2022 ID otherwise).
    ``plate_id`` is the part before ``__`` in the full measurement folder name.
    """
    import gzip, io, urllib.request
    import pandas as pd
    url = (
        "https://raw.githubusercontent.com/jump-cellpainting/datasets"
        "/main/metadata/well.csv.gz"
    )
    logger.info("Fetching JUMP compound metadata from GitHub…")
    try:
        compound_names = _fetch_jump_compound_names()
        with urllib.request.urlopen(url, timeout=60) as resp:
            df = pd.read_csv(gzip.GzipFile(fileobj=io.BytesIO(resp.read())))
        lookup = {
            (row.Metadata_Source, row.Metadata_Plate, row.Metadata_Well):
                compound_names.get(row.Metadata_JCP2022, row.Metadata_JCP2022)
            for row in df.itertuples(index=False)
        }
        logger.info("JUMP compound lookup loaded: %d well entries (%d named controls)",
                    len(lookup), len(compound_names))
        return lookup
    except Exception as e:
        logger.warning("Could not fetch JUMP compound lookup: %s", e)
        return {}


def _list_jump_image_paths(n_plates: int = 10, compound_lookup: dict | None = None) -> list[dict]:
    """List JUMP CP image paths from S3.

    Actual structure:
      s3://cellpainting-gallery/cpg0016-jump/{source}/images/{batch}/images/{plate}/Images/*.tiff
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    BUCKET = "cellpainting-gallery"
    PREFIX = "cpg0016-jump"
    s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))

    # Level 1: list sources
    sources_resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/", Delimiter="/")
    sources = [p["Prefix"].split("/")[1] for p in sources_resp.get("CommonPrefixes", [])]
    if not sources:
        raise RuntimeError("No sources found in JUMP CP bucket")

    records = []
    plates_found = 0

    for source in sources:
        if plates_found >= n_plates:
            break
        # Level 2: source has a fixed 'images/' subfolder
        batches_resp = s3.list_objects_v2(
            Bucket=BUCKET, Prefix=f"{PREFIX}/{source}/images/", Delimiter="/"
        )
        batches = [p["Prefix"].split("/")[-2] for p in batches_resp.get("CommonPrefixes", [])]

        for batch in batches:
            if plates_found >= n_plates:
                break
            # Level 3: batch/{batch}/images/ → plate measurement folders
            plates_resp = s3.list_objects_v2(
                Bucket=BUCKET,
                Prefix=f"{PREFIX}/{source}/images/{batch}/images/",
                Delimiter="/",
            )
            plate_prefixes = [p["Prefix"] for p in plates_resp.get("CommonPrefixes", [])]

            for plate_prefix in plate_prefixes:
                if plates_found >= n_plates:
                    break
                plate = plate_prefix.rstrip("/").split("/")[-1]
                # Level 4: Images/ (uppercase) contains TIFFs
                img_resp = s3.list_objects_v2(
                    Bucket=BUCKET, Prefix=f"{plate_prefix}Images/", MaxKeys=5000
                )
                added = 0
                for obj in img_resp.get("Contents", []):
                    key = obj["Key"]
                    if not (key.endswith(".tiff") or key.endswith(".tif")):
                        continue
                    fname = key.split("/")[-1]
                    try:
                        # Filename format: r{row}c{col}f{site}p01-ch{ch}sk1fk1fl1.tiff
                        row = int(fname[1:3])
                        col = int(fname[4:6])
                        site = int(fname[7:9])
                        ch = int(fname.split("-ch")[1].split("s")[0])
                    except Exception:
                        continue
                    well_rc = f"r{row:02d}c{col:02d}"
                    well_alpha = _rowcol_to_alpha(well_rc)
                    plate_id = plate.split("__")[0]
                    jcp_id = (compound_lookup or {}).get(
                        (source, plate_id, well_alpha), "unknown"
                    )
                    records.append({
                        "s3_key": key, "bucket": BUCKET,
                        "plate": plate, "well": well_rc,
                        "site": site, "channel": ch,
                        "source": source, "batch": batch,
                        "image_path": f"s3://{BUCKET}/{key}",
                        "compound": jcp_id, "moa_class": "unknown",
                    })
                    added += 1
                if added > 0:
                    plates_found += 1
                    logger.info("Plate %d/%d: %s — %d images", plates_found, n_plates, plate, added)

    logger.info("Listed %d images across %d plates", len(records), plates_found)
    return records


def _list_zarr_paths(s3_url: str, n_volumes: int = 5, slices_per_volume: int = 200) -> list[dict]:
    import zarr, s3fs
    fs = s3fs.S3FileSystem(anon=True)
    store = zarr.open(s3fs.S3Map(s3_url, s3=fs), mode="r")
    records = []
    for key in list(store.keys())[:n_volumes]:
        try:
            arr = store[key]
            if arr.ndim < 3:
                continue
            n_z = arr.shape[0] if arr.ndim == 3 else arr.shape[1]
            step = max(1, n_z // slices_per_volume)
            for z in range(0, n_z, step):
                records.append({
                    "zarr_url": s3_url, "array_key": key, "z_slice": int(z),
                    "source": "zarr", "image_path": f"{s3_url}/{key}[{z}]",
                    "compound": "unknown", "moa_class": "organelle",
                })
        except Exception as e:
            logger.warning("Skipping zarr key %s: %s", key, e)
    return records[:n_volumes * slices_per_volume]


def _load_jump_multichannel(records_for_well: list[dict], s3_client: Any = None):
    import io, tifffile, boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    if s3_client is None:
        s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    records_for_well = sorted(records_for_well, key=lambda r: r["channel"])
    channels = []
    meta = {}
    for rec in records_for_well:
        try:
            obj = s3_client.get_object(Bucket=rec["bucket"], Key=rec["s3_key"])
            img = tifffile.imread(io.BytesIO(obj["Body"].read()))
            channels.append(img)
            meta = {k: v for k, v in rec.items() if k not in ("s3_key", "channel")}
        except Exception as e:
            logger.warning("Failed to load %s: %s", rec.get("s3_key"), e)
            return None, {}
    if not channels:
        return None, {}
    return np.stack(channels, axis=-1), meta


def _extract_cell_crops(image: np.ndarray, crop_size: int = 224, n_crops: int = 100) -> list[np.ndarray]:
    H, W = image.shape[:2]
    half = crop_size // 2
    centroids = []
    try:
        from skimage.filters import threshold_otsu
        from skimage.measure import label, regionprops
        dna = image[..., _JUMP_CH_DNA].astype(np.float32) if image.ndim == 3 else image.astype(np.float32)
        dna_norm = _percentile_stretch(dna)
        mask = dna_norm > threshold_otsu(dna_norm)
        props = [p for p in regionprops(label(mask)) if p.area > 200]
        props.sort(key=lambda p: p.area, reverse=True)
        centroids = [(int(p.centroid[0]), int(p.centroid[1])) for p in props[:n_crops]]
    except Exception:
        pass
    if len(centroids) < 10:
        stride = max(crop_size, min(H, W) // max(1, int(np.sqrt(n_crops))))
        centroids = [
            (y + half, x + half)
            for y in range(half, H - half, stride)
            for x in range(half, W - half, stride)
        ][:n_crops]
    crops = []
    for cy, cx in centroids[:n_crops]:
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        if y0 < 0 or y1 > H or x0 < 0 or x1 > W:
            continue
        crops.append(image[y0:y1, x0:x1, :] if image.ndim == 3 else image[y0:y1, x0:x1])
    return crops


# ===========================================================================
# INGESTION RUNNER
# ===========================================================================

async def run_ingestion(
    workspace_dir: str,
    session_id: str,
    dataset: str = "jump-cp",
    n_plates: int = 10,
    zarr_url: str | None = None,
    n_crops_per_image: int = 80,
    embed_batch_size: int = 64,
    n_gpu_workers: int = 4,
    rebuild_index: bool = False,
    _embedder: Any = None,
) -> None:
    """Main ingestion coroutine. Runs the full pipeline asynchronously.

    When _embedder is provided (a _DINOv2Embedder instance loaded on the head node),
    embedding runs locally in a thread pool instead of via Ray GPU workers.
    This is the preferred path when cluster GPUs are fully allocated to other services.
    """
    import subprocess, sys, importlib
    import pandas as pd

    # Ensure required packages are available.
    # The cluster filesystem is read-only for the default pip site-packages.
    # Install missing packages to workspace_dir/pip_packages (writable) and add to sys.path.
    # JUMP Cell Painting TIFFs use LZW compression.
    # Use PIL (Pillow) for TIFF reading — it has native LZW support without extra packages.
    # tifffile requires 'imagecodecs' which needs the cluster filesystem to be writable.
    _pip_dir = str(Path(workspace_dir) / "pip_packages")
    Path(_pip_dir).mkdir(parents=True, exist_ok=True)
    if _pip_dir not in sys.path:
        sys.path.insert(0, _pip_dir)

    _pkg_needed = [
        ("boto3", "boto3"),
        ("scikit-image", "skimage"),
        ("Pillow", "PIL"),
        ("faiss-cpu", "faiss"),
        ("umap-learn", "umap"),
    ]
    _missing = [pkg for pkg, mod in _pkg_needed if not _try_import(mod)]
    if _missing:
        logger.info("Installing missing packages to %s: %s", _pip_dir, _missing)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--target", _pip_dir, "-q"] + _missing,
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pip install failed: {result.stderr[:500]}")
        logger.info("Packages installed successfully")
        for _, mod in _pkg_needed:
            if mod in sys.modules:
                del sys.modules[mod]

    t0 = time.time()

    def _upd(status, msg, n_emb=0, n_tot=0, throughput=0.0, log_lines=None, **extra):
        write_status(workspace_dir, session_id, status, msg,
                     n_embedded=n_emb, n_total=n_tot,
                     throughput_per_sec=throughput,
                     elapsed_seconds=time.time() - t0,
                     log_lines=log_lines or [f"[{time.strftime('%H:%M:%S')}] {msg}"],
                     **extra)

    try:
        _upd(IngestionStatus.PREPARING, "Listing dataset images…")

        if dataset == "jump-cp":
            _loop = asyncio.get_event_loop()
            compound_lookup = await _loop.run_in_executor(None, _fetch_jump_well_lookup)
            records = _list_jump_image_paths(n_plates=n_plates, compound_lookup=compound_lookup)
        elif dataset == "zarr" and zarr_url:
            records = _list_zarr_paths(zarr_url)
        else:
            raise ValueError(f"Unknown dataset type: {dataset!r}")

        # Group CP records by (plate, well, site)
        if dataset == "jump-cp":
            from itertools import groupby
            key_fn = lambda r: (r["plate"], r["well"], r["site"])
            image_groups = [list(v) for _, v in groupby(sorted(records, key=key_fn), key=key_fn)]
        else:
            image_groups = [[r] for r in records]

        n_total_images = len(image_groups)
        n_total_expected = n_total_images * n_crops_per_image
        _upd(IngestionStatus.PREPARING,
             f"Found {n_total_images} images (~{n_total_expected:,} cells). Starting embedding…",
             n_tot=n_total_expected)

        # ----------------------------------------------------------------
        # Embedding strategy:
        #   If _embedder is provided → run on head node in thread pool (no Ray workers).
        #   This is preferred when cluster GPUs are fully allocated to other services.
        # ----------------------------------------------------------------
        loop = asyncio.get_event_loop()

        def _process_group_local(group_records: list, n_crops: int, embedder: Any) -> list | None:
            """Process one image group on the head node: load → crop → embed."""
            import io, base64, time as _time
            _gt0 = _time.time()
            try:
                if len(group_records) == 1 and group_records[0].get("zarr_url"):
                    rec = group_records[0]
                    import zarr, s3fs
                    fs = s3fs.S3FileSystem(anon=True)
                    store = zarr.open(s3fs.S3Map(rec["zarr_url"], s3=fs), mode="r")
                    arr = store[rec["array_key"]]
                    z = rec["z_slice"]
                    img = np.array(arr[z] if arr.ndim == 3 else arr[:, z, :])
                    crp = _extract_cell_crops(img, n_crops=n_crops) or [img]
                    rgb_crops = [_to_rgb_uint8(c) for c in crp]
                    meta_base = {k: v for k, v in rec.items() if k not in ("zarr_url", "array_key", "z_slice")}
                else:
                    import io as _io, boto3
                    from botocore import UNSIGNED
                    from botocore.config import Config
                    from PIL import Image as _PILRead
                    from concurrent.futures import ThreadPoolExecutor as _TPE
                    # Download all channels in parallel to maximize S3 bandwidth utilization
                    recs = sorted(group_records, key=lambda r: r["channel"])
                    meta_base = {k: v for k, v in recs[0].items() if k not in ("s3_key", "channel")} if recs else {}

                    def _dl_chan(rec):
                        _s3 = boto3.client("s3", region_name="us-east-1",
                                           config=Config(signature_version=UNSIGNED))
                        obj = _s3.get_object(Bucket=rec["bucket"], Key=rec["s3_key"])
                        # Use PIL (Pillow) which has native LZW TIFF support
                        # (tifffile requires 'imagecodecs' package for LZW decoding)
                        return np.array(_PILRead.open(_io.BytesIO(obj["Body"].read())))

                    with _TPE(max_workers=len(recs)) as pool:
                        chans = list(pool.map(_dl_chan, recs))
                    _t_dl = _time.time() - _gt0

                    if not chans:
                        return None
                    mc_img = np.stack(chans, axis=-1)
                    _t_stack = _time.time() - _gt0
                    crp = _extract_cell_crops(mc_img, n_crops=n_crops)
                    if not crp:
                        return None
                    rgb_crops = [_to_rgb_uint8(c) for c in crp]
                    _t_crop = _time.time() - _gt0

                embeddings = embedder.embed_batch(rgb_crops, batch_size=64)
                _t_emb = _time.time() - _gt0
                logger.debug("Timing: dl=%.1fs, crop=%.1fs, emb=%.1fs, total=%.1fs, n_crops=%d",
                             _t_dl, _t_crop - _t_dl, _t_emb - _t_crop, _t_emb, len(rgb_crops))
                from PIL import Image as _PIL
                results = []
                for i, (emb, crop_rgb) in enumerate(zip(embeddings, rgb_crops)):
                    thumb = _PIL.fromarray(crop_rgb).resize((96, 96))
                    buf = io.BytesIO(); thumb.save(buf, format="PNG")
                    results.append({
                        **meta_base, "crop_idx": i,
                        "embedding": emb.tolist(),
                        "thumbnail_b64": base64.b64encode(buf.getvalue()).decode(),
                    })
                return results
            except Exception as e:
                import traceback as _tb
                err_short = str(e)[:120]
                logger.warning("Group processing failed: %s\n%s", e, _tb.format_exc()[-800:])
                return ("ERROR", err_short)

        _upd(IngestionStatus.PREPARING,
             f"Embedding {n_total_images} image groups on head node (DINOv2)…",
             n_tot=n_total_expected)

        # Set up embed worker pool (supports both single embedder and list)
        _workers = _embedder if isinstance(_embedder, list) else [_embedder]
        _pool_q: asyncio.Queue = asyncio.Queue()
        for _w in _workers:
            _pool_q.put_nowait(_w)

        async def _run_group(group: list, idx: int):
            """Acquire an embedder from the pool, process group, release embedder."""
            emb = await _pool_q.get()
            try:
                return (idx, await loop.run_in_executor(
                    None, _process_group_local, group, n_crops_per_image, emb))
            finally:
                _pool_q.put_nowait(emb)

        # Launch all group tasks up front; pool_q bounds actual concurrency to len(_workers).
        all_tasks = [asyncio.create_task(_run_group(g, i)) for i, g in enumerate(image_groups)]

        all_results = []
        first_error: str | None = None
        n_done = 0

        for task in all_tasks:
            if is_stop_requested(workspace_dir, session_id):
                for t in all_tasks[n_done:]:
                    t.cancel()
                _upd(IngestionStatus.STOPPED, "Stopped by user.",
                     n_emb=len(all_results), n_tot=n_total_expected)
                return

            _idx, res = await task
            if isinstance(res, tuple) and res[0] == "ERROR":
                if first_error is None:
                    first_error = res[1]
                    logger.error("First group error: %s", first_error)
                if _idx < 5:
                    _upd(IngestionStatus.RUNNING,
                         f"Group {_idx+1} error: {first_error}",
                         n_emb=0, n_tot=n_total_expected,
                         log_lines=[f"[{time.strftime('%H:%M:%S')}] ERROR: {first_error}"])
            elif res:
                all_results.extend(res)
                first_error = None

            n_done += 1
            if n_done % 5 == 0 or n_done == n_total_images:
                elapsed = time.time() - t0
                tput = len(all_results) / max(elapsed, 1)
                log_line = (
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"{n_done}/{n_total_images} images · "
                    f"{len(all_results):,} cells · "
                    f"{tput:.1f} cells/s"
                )
                _upd(IngestionStatus.RUNNING,
                     f"Embedded {len(all_results):,} cells ({n_done}/{n_total_images} images)",
                     n_emb=len(all_results), n_tot=n_total_expected,
                     throughput=tput, log_lines=[log_line])

        if not all_results:
            _upd(IngestionStatus.FAILED, "No cells successfully embedded.")
            return

        # Build FAISS index
        _upd(IngestionStatus.BUILDING_INDEX,
             f"Building FAISS index for {len(all_results):,} cells…",
             n_emb=len(all_results), n_tot=len(all_results))

        emb_arr = np.array([r["embedding"] for r in all_results], dtype=np.float32)
        meta_rows = [{k: v for k, v in r.items() if k not in ("embedding", "thumbnail_b64")}
                     for r in all_results]
        thumbs = [r.get("thumbnail_b64", "") for r in all_results]

        df = pd.DataFrame(meta_rows)
        df["idx"] = np.arange(len(df))

        out_dir = Path(workspace_dir) / "cell_search"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "thumbnails.npy", np.array(thumbs, dtype=object))

        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, _build_index, emb_arr, df, workspace_dir)

        elapsed = time.time() - t0
        _upd(IngestionStatus.COMPLETED,
             f"Done! {len(all_results):,} cells in {elapsed/60:.1f} min. "
             f"Index: {stats['index_type']}, {stats['index_size_mb']:.0f} MB",
             n_emb=len(all_results), n_tot=len(all_results),
             throughput=len(all_results) / elapsed,
             index_stats=stats)

    except Exception as e:
        logger.exception("Ingestion failed: %s", e)
        write_status(workspace_dir, session_id, IngestionStatus.FAILED, f"Ingestion failed: {e}",
                     elapsed_seconds=time.time() - t0)
        raise


# ===========================================================================
# DATASET REGISTRY
# ===========================================================================

def _registry_path(workspace_dir: str) -> Path:
    return Path(workspace_dir) / "datasets.json"


def _load_registry(workspace_dir: str) -> list[dict]:
    p = _registry_path(workspace_dir)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def _save_registry(workspace_dir: str, registry: list[dict]) -> None:
    p = _registry_path(workspace_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(registry, indent=2))
    tmp.replace(p)


def _upsert_registry(workspace_dir: str, entry: dict) -> None:
    registry = _load_registry(workspace_dir)
    for i, e in enumerate(registry):
        if e.get("name") == entry["name"]:
            registry[i] = {**e, **entry}
            _save_registry(workspace_dir, registry)
            return
    registry.append(entry)
    _save_registry(workspace_dir, registry)


# ===========================================================================
# PRECONFIGURED DATASETS
# ===========================================================================

PRECONFIGURED_DATASETS: list[dict] = [
    {
        "name": "JUMP Cell Painting — Demo",
        "type": "jump-cp",
        "description": "JUMP-CP (cpg0016): 10 plates, ~80K cells. "
                       "Public AWS S3. Standard for drug MOA profiling.",
        "config": {"n_plates": 10, "n_crops_per_image": 80, "n_gpu_workers": 1},
    },
]


# ===========================================================================
# RAY SERVE DEPLOYMENT
# ===========================================================================

def _make_deployment():
    from ray import serve
    from bioengine.utils import create_logger
    from hypha_rpc.utils.schema import schema_method
    from pydantic import Field

    @serve.deployment(
        ray_actor_options={
            "num_cpus": 4,
            "num_gpus": 1,
            "memory": int(8 * 1024**3),
        },
        max_ongoing_requests=10,
        max_queued_requests=100,
        autoscaling_config={
            "min_replicas": 1,
            "initial_replicas": 1,
            "max_replicas": 1,
            "target_ongoing_requests": 3,
        },
        health_check_period_s=60.0,
        health_check_timeout_s=60.0,
        graceful_shutdown_timeout_s=600.0,
    )
    class CellImageSearch:
        """Large-scale cell morphology similarity search engine."""

        def __init__(
            self,
            workspace_dir: str = "/tmp/cell-image-search",
            auto_ingest: bool = True,
        ) -> None:
            self._workspace_dir = workspace_dir
            self._auto_ingest = auto_ingest
            self._start_time = time.time()
            self._logger = create_logger("CellImageSearch")
            self._embedder: Any = None
            self._index: Any = None
            self._metadata_df: Any = None
            self._index_info: dict = {}
            self._thumbnails: Any = None
            self._tasks: dict[str, asyncio.Task] = {}
            self._session_lock = asyncio.Lock()
            self._session_dataset_map: dict[str, str] = {}

        async def async_init(self) -> None:
            """Load DINOv2, load existing index, then auto-ingest if needed."""
            import torch, os

            # Use home dir (set by BioEngine to apps_workdir/app-id) as workspace
            home = os.environ.get("HOME", os.getcwd())
            self._workspace_dir = os.path.join(home, "cell_search_data")
            Path(self._workspace_dir).mkdir(parents=True, exist_ok=True)

            # Set TORCH_HOME to workspace so workers share the same model cache
            # (avoids re-downloading DINOv2 on every worker if PVC is shared)
            torch_hub = os.path.join(self._workspace_dir, "torch_hub")
            os.makedirs(torch_hub, exist_ok=True)
            os.environ.setdefault("TORCH_HOME", torch_hub)
            os.environ.setdefault("HF_HOME", os.path.join(self._workspace_dir, "hf_home"))

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._embedder = _DINOv2Embedder(device=device, fp16=(device == "cuda"))
            self._logger.info("Loading DINOv2 on %s…", device)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._embedder.load)
            self._embed_workers = [self._embedder]
            self._logger.info("DINOv2 ready.")

            index_loaded = await self._try_load_index()
            if self._auto_ingest and not index_loaded:
                self._logger.info("No index — auto-ingesting %d preconfigured dataset(s)…",
                                  len(PRECONFIGURED_DATASETS))
                for ds in PRECONFIGURED_DATASETS:
                    await self._start_dataset_ingestion(ds)

        async def _try_load_index(self) -> bool:
            try:
                loop = asyncio.get_event_loop()
                self._index, self._metadata_df, self._index_info = \
                    await loop.run_in_executor(None, _load_index, self._workspace_dir)
                thumb_path = Path(self._workspace_dir) / "cell_search" / "thumbnails.npy"
                if thumb_path.exists():
                    self._thumbnails = np.load(thumb_path, allow_pickle=True)
                self._logger.info("Index loaded: %d cells", self._index.ntotal)
                return True
            except FileNotFoundError:
                return False
            except Exception as e:
                self._logger.warning("Could not load index: %s", e)
                return False

        async def test_deployment(self) -> None:
            result = await self.ping()
            assert result["status"] == "ok"

        async def check_health(self) -> None:
            if self._embedder is None:
                raise RuntimeError("DINOv2 not loaded")

        # ----------------------------------------------------------------
        # Internal helpers
        # ----------------------------------------------------------------

        async def _start_dataset_ingestion(self, ds: dict) -> str:
            now = datetime.now(timezone.utc)
            session_id = now.strftime("%Y%m%d-%H%M%S") + "-" + str(uuid4())[:8]
            cfg = ds.get("config", {})
            ds_type = ds.get("type", "jump-cp")
            ds_name = ds.get("name", "unnamed")

            write_status(self._workspace_dir, session_id, IngestionStatus.WAITING,
                         f"Queued: {ds_name}", dataset_name=ds_name)
            _upsert_registry(self._workspace_dir, {
                "name": ds_name, "type": ds_type,
                "description": ds.get("description", ""),
                "zarr_url": ds.get("zarr_url", ""),
                "config": cfg, "status": "indexing",
                "session_id": session_id,
                "date_added": now.isoformat(), "n_cells": 0,
            })

            async def _run():
                try:
                    await run_ingestion(
                        workspace_dir=self._workspace_dir, session_id=session_id,
                        dataset=ds_type,
                        n_plates=cfg.get("n_plates", 10),
                        zarr_url=ds.get("zarr_url") or None,
                        n_crops_per_image=cfg.get("n_crops_per_image", 80),
                        _embedder=self._embed_workers,  # pass full worker pool for concurrency
                        n_gpu_workers=cfg.get("n_gpu_workers", 4),
                    )
                    await self._try_load_index()
                    final = read_status(self._workspace_dir, session_id)
                    _upsert_registry(self._workspace_dir, {
                        "name": ds_name,
                        "status": "indexed" if final.get("status") == "completed" else final.get("status", "unknown"),
                        "n_cells": final.get("n_embedded", 0),
                        "date_indexed": datetime.now(timezone.utc).isoformat(),
                    })
                except Exception as e:
                    self._logger.exception("Ingestion failed for %s: %s", ds_name, e)
                    _upsert_registry(self._workspace_dir, {"name": ds_name, "status": "failed", "error": str(e)})

            async with self._session_lock:
                task = asyncio.create_task(_run())
                self._tasks[session_id] = task
                self._session_dataset_map[session_id] = ds_name
            return session_id

        # ----------------------------------------------------------------
        # Public API
        # ----------------------------------------------------------------

        @schema_method
        async def ping(self) -> dict:
            """Check connectivity and get deployment status."""
            n_cells = self._index.ntotal if self._index is not None else 0
            active = [sid for sid, t in self._tasks.items() if not t.done()]
            return {
                "status": "ok",
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "model": f"DINOv2 {_DINOv2Embedder.MODEL_NAME} ({_DINOv2Embedder.EMBED_DIM}-dim)",
                "index_loaded": self._index is not None,
                "n_cells_indexed": n_cells,
                "index_type": self._index_info.get("index_type", "none"),
                "workspace_dir": self._workspace_dir,
                "active_sessions": active,
            }

        @schema_method
        async def get_index_stats(self) -> dict:
            """Return detailed statistics about the current vector index."""
            if self._index is None:
                return {"indexed": False, "n_cells": 0}
            n_cells = self._index.ntotal
            n_compounds = 0
            if self._metadata_df is not None and "compound" in self._metadata_df.columns:
                n_compounds = int(self._metadata_df["compound"].nunique())
            return {
                "indexed": True, "n_cells": n_cells, "n_compounds": n_compounds,
                **self._index_info, "workspace_dir": self._workspace_dir,
            }

        @schema_method
        async def list_datasets(self) -> dict:
            """List all registered datasets and their indexing status."""
            registry = _load_registry(self._workspace_dir)
            active_sids = {sid for sid, t in self._tasks.items() if not t.done()}
            for entry in registry:
                if entry.get("session_id") in active_sids:
                    entry["status"] = "indexing"
            return {"datasets": registry, "active_sessions": list(active_sids)}

        @schema_method
        async def add_dataset(
            self,
            name: str = Field(..., description="Human-readable name for this dataset."),
            zarr_url: str = Field(..., description="S3 or HTTP URL to the Zarr store."),
            description: str = Field("", description="Optional description."),
            n_slices_per_volume: int = Field(200, ge=10, le=5000,
                                              description="2D slices per 3D volume."),
            n_gpu_workers: int = Field(4, ge=1, le=64,
                                       description="Parallel GPU workers for embedding."),
            n_crops_per_slice: int = Field(50, ge=1, le=500,
                                           description="Crops per 2D slice."),
        ) -> dict:
            """Register a Zarr dataset and start indexing it immediately."""
            ds = {
                "name": name, "type": "zarr", "zarr_url": zarr_url,
                "description": description,
                "config": {"n_slices_per_volume": n_slices_per_volume,
                           "n_gpu_workers": n_gpu_workers,
                           "n_crops_per_image": n_crops_per_slice},
            }
            session_id = await self._start_dataset_ingestion(ds)
            return {"session_id": session_id, "name": name, "zarr_url": zarr_url,
                    "status": "queued",
                    "message": f"Ingestion started. Poll get_ingestion_status('{session_id}')."}

        @schema_method
        async def add_jump_cp_dataset(
            self,
            name: str = Field("JUMP Cell Painting", description="Dataset name."),
            n_plates: int = Field(10, ge=1, le=500,
                                  description="JUMP CP plates to index (≈5K cells each)."),
            n_gpu_workers: int = Field(4, ge=1, le=64,
                                       description="Parallel GPU workers."),
        ) -> dict:
            """Add and start indexing JUMP Cell Painting plates.

            Dataset source: s3://cellpainting-gallery/cpg0016-jump/ (public).
            """
            ds = {
                "name": name, "type": "jump-cp",
                "description": f"JUMP CP (cpg0016), {n_plates} plates.",
                "config": {"n_plates": n_plates, "n_gpu_workers": n_gpu_workers,
                           "n_crops_per_image": 80},
            }
            session_id = await self._start_dataset_ingestion(ds)
            return {"session_id": session_id, "name": name, "status": "queued",
                    "message": f"Ingestion started. Poll get_ingestion_status('{session_id}')."}

        @schema_method
        async def remove_dataset(
            self,
            name: str = Field(..., description="Dataset name to remove."),
        ) -> dict:
            """Remove a dataset from the registry (does not delete the FAISS index)."""
            registry = [e for e in _load_registry(self._workspace_dir) if e.get("name") != name]
            _save_registry(self._workspace_dir, registry)
            return {"removed": name, "remaining": len(registry)}

        @schema_method
        async def start_ingestion(
            self,
            dataset: str = Field("jump-cp", description="'jump-cp' or 'zarr'."),
            n_plates: int = Field(10, ge=1, le=500),
            zarr_url: str = Field(""),
            n_crops_per_image: int = Field(80, ge=1, le=500),
            n_gpu_workers: int = Field(4, ge=1, le=64),
            workspace_dir: str = Field(""),
            rebuild_index: bool = Field(False),
            dataset_name: str = Field(""),
        ) -> dict:
            """Start a background ingestion job (low-level API). Prefer add_dataset()."""
            wdir = workspace_dir or self._workspace_dir
            now = datetime.now(timezone.utc)
            session_id = now.strftime("%Y%m%d-%H%M%S") + "-" + str(uuid4())[:8]
            name = dataset_name or f"{dataset} {now.strftime('%Y-%m-%d %H:%M')}"

            write_status(wdir, session_id, IngestionStatus.WAITING, f"Queued: {name}",
                         dataset_name=name)
            _upsert_registry(wdir, {"name": name, "type": dataset, "status": "indexing",
                                     "session_id": session_id,
                                     "date_added": now.isoformat(), "n_cells": 0})

            async def _run():
                await run_ingestion(workspace_dir=wdir, session_id=session_id, dataset=dataset,
                                    n_plates=n_plates, zarr_url=zarr_url or None,
                                    n_crops_per_image=n_crops_per_image,
                                    n_gpu_workers=n_gpu_workers, rebuild_index=rebuild_index,
                                    _embedder=self._embedder)
                await self._try_load_index()
                final = read_status(wdir, session_id)
                _upsert_registry(wdir, {
                    "name": name,
                    "status": "indexed" if final.get("status") == "completed" else final.get("status", "unknown"),
                    "n_cells": final.get("n_embedded", 0),
                    "date_indexed": datetime.now(timezone.utc).isoformat(),
                })

            async with self._session_lock:
                task = asyncio.create_task(_run())
                self._tasks[session_id] = task
            return {"session_id": session_id, "status": "waiting", "dataset": dataset}

        @schema_method
        async def get_ingestion_status(
            self,
            session_id: str = Field(..., description="Session ID from start_ingestion or add_dataset."),
        ) -> dict:
            """Get real-time status of an ingestion job."""
            status = read_status(self._workspace_dir, session_id)
            task = self._tasks.get(session_id)
            if task is not None and task.done() and \
               status.get("status") in ("running", "preparing", "building_index"):
                exc = task.exception()
                if exc:
                    status["status"] = "failed"
                    status["message"] = str(exc)
            return status

        @schema_method
        async def stop_ingestion(
            self,
            session_id: str = Field(..., description="Session ID to cancel."),
        ) -> dict:
            """Cancel a running ingestion job."""
            request_stop(self._workspace_dir, session_id)
            task = self._tasks.get(session_id)
            if task and not task.done():
                task.cancel()
            sid_name = self._session_dataset_map.get(session_id)
            if sid_name:
                _upsert_registry(self._workspace_dir, {"name": sid_name, "status": "stopped"})
            return read_status(self._workspace_dir, session_id)

        @schema_method
        async def get_active_sessions(self) -> dict:
            """Return all running ingestion session IDs and their status.

            Call on page load to resume monitoring in-progress jobs.
            """
            sessions = []
            for sid, task in list(self._tasks.items()):
                status = read_status(self._workspace_dir, sid)
                sessions.append({
                    "session_id": sid,
                    "dataset_name": self._session_dataset_map.get(sid, ""),
                    "is_running": not task.done(),
                    **status,
                })
            return {"sessions": sessions}

        @schema_method
        async def search(
            self,
            image_b64: str = Field(..., description="Base64-encoded image (PNG/JPG/TIFF)."),
            top_k: int = Field(20, ge=1, le=100),
            plow: float = Field(1.0, ge=0.0, le=10.0),
            phigh: float = Field(99.0, ge=90.0, le=100.0),
        ) -> dict:
            """Find the top-K morphologically similar cells in the indexed database."""
            import io as _io
            from PIL import Image as _PIL

            if self._index is None:
                return {"error": "No index loaded. Add a dataset first.", "results": []}

            t0 = time.time()
            loop = asyncio.get_event_loop()
            img_bytes = base64.b64decode(image_b64)
            img_raw = await loop.run_in_executor(None, _decode_image_bytes, img_bytes)
            img_rgb = await loop.run_in_executor(None, _to_rgb_uint8, img_raw, None, plow, phigh)

            thumb = _PIL.fromarray(img_rgb).resize((224, 224))
            buf = _io.BytesIO()
            thumb.save(buf, format="PNG")
            query_thumb_b64 = base64.b64encode(buf.getvalue()).decode()

            query_emb = await loop.run_in_executor(None, self._embedder.embed_single, img_rgb)
            results_raw = await loop.run_in_executor(
                None, _search_index, self._index, self._metadata_df, query_emb, top_k
            )

            results = []
            for r in results_raw:
                idx = r.get("faiss_idx", -1)
                thumb_b64 = ""
                if self._thumbnails is not None and 0 <= idx < len(self._thumbnails):
                    thumb_b64 = str(self._thumbnails[idx])
                results.append({**r, "thumbnail_b64": thumb_b64})

            return {
                "results": results,
                "query_thumbnail_b64": query_thumb_b64,
                "elapsed_ms": round((time.time() - t0) * 1000, 1),
                "n_cells_searched": self._index.ntotal,
                "top_k": top_k,
            }

        @schema_method
        async def get_umap_preview(
            self,
            n_samples: int = Field(10_000, ge=100, le=100_000),
            color_by: str = Field("compound"),
            force_recompute: bool = Field(False),
        ) -> dict:
            """Compute or return cached UMAP projection of the indexed cells."""
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, _compute_umap, self._workspace_dir, n_samples, 42, force_recompute, color_by
            )

        @schema_method
        async def project_query_onto_umap(
            self,
            image_b64: str = Field(..., description="Base64-encoded query image."),
        ) -> dict:
            """Project a query image onto the pre-computed UMAP space."""
            if self._index is None:
                return {"error": "No index loaded."}

            loop = asyncio.get_event_loop()
            img_bytes = base64.b64decode(image_b64)
            img_raw = await loop.run_in_executor(None, _decode_image_bytes, img_bytes)
            img_rgb = await loop.run_in_executor(None, _to_rgb_uint8, img_raw)
            query_emb = await loop.run_in_executor(None, self._embedder.embed_single, img_rgb)
            nn_results = await loop.run_in_executor(
                None, _search_index, self._index, self._metadata_df, query_emb, 1
            )

            umap_cache = Path(self._workspace_dir) / "cell_search" / "umap_cache.npz"
            umap_x, umap_y = 0.0, 0.0
            if umap_cache.exists() and nn_results:
                nn_idx = nn_results[0].get("faiss_idx", -1)
                data = np.load(umap_cache, allow_pickle=True)
                if nn_idx >= 0:
                    umap_x = float(data["x"][nn_idx % len(data["x"])])
                    umap_y = float(data["y"][nn_idx % len(data["y"])])

            nn = nn_results[0] if nn_results else {}
            return {
                "umap_x": umap_x, "umap_y": umap_y,
                "nearest_score": nn.get("score", 0.0),
                "nearest_compound": nn.get("compound", "unknown"),
                "nearest_moa": nn.get("moa_class", "unknown"),
            }

        @schema_method
        async def enrich_metadata_with_compounds(self) -> dict:
            """Download JUMP compound metadata and enrich the existing index metadata.

            Fetches well.csv.gz from the JUMP datasets GitHub repo and updates
            the ``compound`` column in metadata.parquet using JCP2022 compound IDs.
            Safe to call on a running index — does not rebuild the FAISS index.
            """
            import pandas as pd

            meta_path = Path(self._workspace_dir) / "cell_search" / "metadata.parquet"
            if not meta_path.exists():
                return {"error": "No metadata.parquet found. Run ingestion first."}

            loop = asyncio.get_event_loop()
            lookup = await loop.run_in_executor(None, _fetch_jump_well_lookup)
            if not lookup:
                return {"error": "Could not download JUMP compound metadata."}

            df = pd.read_parquet(meta_path)
            n_before = int((df["compound"] != "unknown").sum()) if "compound" in df.columns else 0

            def _enrich(row):
                plate_id = str(row.get("plate", "")).split("__")[0]
                well_rc = str(row.get("well", ""))
                source = str(row.get("source", ""))
                well_alpha = _rowcol_to_alpha(well_rc)
                return lookup.get((source, plate_id, well_alpha), "unknown")

            df["compound"] = [_enrich(r) for r in df.to_dict("records")]
            df.to_parquet(meta_path, index=False)

            # Reload in-memory metadata
            self._metadata_df = df

            # Invalidate all UMAP caches so they get recomputed with new compound labels
            umap_dir = Path(self._workspace_dir) / "cell_search"
            for umap_cache in umap_dir.glob("umap_cache*.npz"):
                umap_cache.unlink()

            n_after = int((df["compound"] != "unknown").sum())
            n_unique = int(df["compound"].nunique())
            return {
                "status": "ok",
                "n_total": len(df),
                "n_enriched": n_after,
                "n_unknown": len(df) - n_after,
                "n_unique_compounds": n_unique,
                "enriched_pct": round(100 * n_after / len(df), 1),
            }

    return CellImageSearch


CellImageSearch = _make_deployment()
