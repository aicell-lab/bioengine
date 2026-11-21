"""
Distributed ingestion pipeline for large-scale biological imaging datasets.

Supports:
  - JUMP Cell Painting (s3://cellpainting-gallery/cpg0016-jump/)
    116K wells, 5-channel fluorescence TIFF, ~47 TB raw
  - OpenOrganelle/CellMap (s3://janelia-cosem-datasets/)
    3D FIB-SEM volumes in Zarr format, 2D slices extracted
  - Custom Zarr datasets (any S3/HTTP/local path)

Pipeline (per worker GPU):
  1. List image paths from S3/metadata CSV
  2. Download → decode → crop cells (~500/image for CP)
  3. Normalise (percentile stretch) → RGB uint8
  4. Embed with DINOv2 ViT-B/14 (fp16, batch=64)
  5. L2-normalise → accumulate (embedding, metadata) pairs
  6. Build / update FAISS index

Ray Data enables automatic load balancing across N GPU workers.
Status is written to {workspace_dir}/sessions/{session_id}/status.json.
"""

from __future__ import annotations

import json
import logging
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Status helpers (mirrors cellpose-finetuning pattern)
# ---------------------------------------------------------------------------


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


def _status_path(workspace_dir: str, session_id: str) -> Path:
    return _session_dir(workspace_dir, session_id) / "status.json"


def _stop_path(workspace_dir: str, session_id: str) -> Path:
    return _session_dir(workspace_dir, session_id) / "stop_requested"


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
    path = _status_path(workspace_dir, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve existing log lines and append new ones
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            pass
    prev_log = existing.get("log_tail", [])
    if log_lines:
        prev_log = (prev_log + log_lines)[-20:]  # keep last 20 lines

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
    path = _status_path(workspace_dir, session_id)
    if not path.exists():
        return {"status": IngestionStatus.WAITING.value, "message": "Not started"}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"status": "unknown", "message": "Error reading status"}


def is_stop_requested(workspace_dir: str, session_id: str) -> bool:
    return _stop_path(workspace_dir, session_id).exists()


def request_stop(workspace_dir: str, session_id: str) -> None:
    p = _stop_path(workspace_dir, session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("1")


# ---------------------------------------------------------------------------
# JUMP Cell Painting dataset helpers
# ---------------------------------------------------------------------------

JUMP_S3_BUCKET = "cellpainting-gallery"
JUMP_S3_PREFIX = "cpg0016-jump"


def list_jump_image_paths(
    n_plates: int = 10,
    channels: list[int] | None = None,
) -> list[dict]:
    """List JUMP CP image paths and metadata from S3.

    Returns list of dicts with keys: path, plate, well, site, channel, source.
    Downloads metadata CSV to discover plates/wells/sites.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    # List sources
    sources_resp = s3.list_objects_v2(
        Bucket=JUMP_S3_BUCKET,
        Prefix=f"{JUMP_S3_PREFIX}/",
        Delimiter="/",
    )
    sources = [
        p["Prefix"].split("/")[1] for p in sources_resp.get("CommonPrefixes", [])
    ]
    if not sources:
        raise RuntimeError("No sources found in JUMP CP dataset")

    records = []
    plates_found = 0

    for source in sources[:3]:  # limit sources for demo
        if plates_found >= n_plates:
            break
        batches_resp = s3.list_objects_v2(
            Bucket=JUMP_S3_BUCKET,
            Prefix=f"{JUMP_S3_PREFIX}/{source}/",
            Delimiter="/",
        )
        batches = [
            p["Prefix"].split("/")[-2] for p in batches_resp.get("CommonPrefixes", [])
        ]

        for batch in batches[:2]:
            if plates_found >= n_plates:
                break
            plates_resp = s3.list_objects_v2(
                Bucket=JUMP_S3_BUCKET,
                Prefix=f"{JUMP_S3_PREFIX}/{source}/{batch}/images/",
                Delimiter="/",
            )
            plate_prefixes = [
                p["Prefix"] for p in plates_resp.get("CommonPrefixes", [])
            ]

            for plate_prefix in plate_prefixes[: n_plates - plates_found]:
                plate = plate_prefix.rstrip("/").split("/")[-1]
                # List image files in this plate
                img_resp = s3.list_objects_v2(
                    Bucket=JUMP_S3_BUCKET,
                    Prefix=f"{plate_prefix}Images/",
                    MaxKeys=5000,
                )
                for obj in img_resp.get("Contents", []):
                    key = obj["Key"]
                    if not key.endswith(".tiff") and not key.endswith(".tif"):
                        continue
                    fname = key.split("/")[-1]
                    # Parse filename: r{row}c{col}f{site}p01-ch{ch}sk1fk1fl1.tiff
                    try:
                        row = int(fname[1:3])
                        col = int(fname[4:6])
                        site = int(fname[7:9])
                        ch = int(fname.split("-ch")[1].split("s")[0])
                    except Exception:
                        continue
                    if channels and ch not in channels:
                        continue
                    well = f"r{row:02d}c{col:02d}"
                    records.append(
                        {
                            "s3_key": key,
                            "bucket": JUMP_S3_BUCKET,
                            "plate": plate,
                            "well": well,
                            "site": site,
                            "channel": ch,
                            "source": source,
                            "batch": batch,
                            "image_path": f"s3://{JUMP_S3_BUCKET}/{key}",
                        }
                    )
                plates_found += 1
                if plates_found >= n_plates:
                    break

    logger.info("Listed %d images across %d plates", len(records), plates_found)
    return records


# ---------------------------------------------------------------------------
# Zarr dataset helpers (OpenOrganelle / CellMap)
# ---------------------------------------------------------------------------


def list_zarr_paths(
    s3_url: str,
    n_volumes: int = 5,
    slices_per_volume: int = 200,
) -> list[dict]:
    """List 2D slice paths from a Zarr volume store on S3.

    Extracts XY slices at regular Z intervals for 2D embedding.
    """
    import zarr
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    store = zarr.open(s3fs.S3Map(s3_url, s3=fs), mode="r")

    records = []
    z_arr = None

    # Find the highest-resolution array
    for key in list(store.keys())[:n_volumes]:
        try:
            arr = store[key]
            if arr.ndim >= 3:
                z_arr = arr
                n_z = arr.shape[0] if arr.ndim == 3 else arr.shape[1]
                step = max(1, n_z // slices_per_volume)
                for z in range(0, n_z, step):
                    records.append(
                        {
                            "zarr_url": s3_url,
                            "array_key": key,
                            "z_slice": int(z),
                            "source": "zarr",
                            "image_path": f"{s3_url}/{key}[{z}]",
                            "compound": "unknown",
                            "moa_class": "organelle",
                        }
                    )
        except Exception as e:
            logger.warning("Skipping %s: %s", key, e)

    return records[: n_volumes * slices_per_volume]


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------


def load_jump_image_multichannel(
    records_for_well: list[dict],
    s3_client: Any = None,
) -> tuple[np.ndarray | None, dict]:
    """Load all channels for one well/site and stack into (H, W, C) array.

    Returns (image_array, metadata_dict) or (None, {}) on error.
    """
    import io
    import tifffile

    if s3_client is None:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    # Sort by channel
    records_for_well = sorted(records_for_well, key=lambda r: r["channel"])
    channels = []
    meta = {}

    for rec in records_for_well:
        try:
            obj = s3_client.get_object(Bucket=rec["bucket"], Key=rec["s3_key"])
            data = obj["Body"].read()
            img = tifffile.imread(io.BytesIO(data))
            channels.append(img)
            meta = {k: v for k, v in rec.items() if k not in ("s3_key", "channel")}
        except Exception as e:
            logger.warning("Failed to load %s: %s", rec.get("s3_key"), e)
            return None, {}

    if not channels:
        return None, {}

    stacked = np.stack(channels, axis=-1)  # (H, W, n_channels)
    return stacked, meta


def extract_cell_crops(
    image: np.ndarray,
    crop_size: int = 224,
    n_crops: int = 100,
    dna_channel: int = 0,
) -> list[np.ndarray]:
    """Extract cell crops from a whole-well image by finding nuclei centroids.

    Uses Otsu thresholding on the DNA channel for nucleus detection.
    Falls back to a regular grid if segmentation fails.

    Args:
        image: (H, W, C) uint8 or uint16 array.
        crop_size: Size of square crops in pixels.
        n_crops: Maximum number of crops to return.
        dna_channel: Index of DNA/DAPI channel.

    Returns:
        List of (crop_size, crop_size, C) arrays.
    """
    from normalizer import percentile_stretch

    H, W = image.shape[:2]
    half = crop_size // 2

    # Try nucleus detection via Otsu threshold on DNA channel
    try:
        from skimage.filters import threshold_otsu
        from skimage.measure import label, regionprops

        if image.ndim == 3:
            dna = image[..., dna_channel].astype(np.float32)
        else:
            dna = image.astype(np.float32)

        dna_norm = percentile_stretch(dna)
        thresh = threshold_otsu(dna_norm)
        mask = dna_norm > thresh

        labeled = label(mask)
        props = regionprops(labeled)
        # Sort by area descending, take largest (actual nuclei, not noise)
        props = [p for p in props if p.area > 200]
        props = sorted(props, key=lambda p: p.area, reverse=True)

        centroids = [(int(p.centroid[0]), int(p.centroid[1])) for p in props[:n_crops]]
    except Exception:
        centroids = []

    # Fallback: regular grid
    if len(centroids) < 10:
        stride = max(crop_size, min(H, W) // int(np.sqrt(n_crops)))
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
        if image.ndim == 3:
            crop = image[y0:y1, x0:x1, :]
        else:
            crop = image[y0:y1, x0:x1]
        crops.append(crop)

    return crops


# ---------------------------------------------------------------------------
# Main ingestion runner (called as asyncio task from main.py)
# ---------------------------------------------------------------------------


async def run_ingestion(
    workspace_dir: str,
    session_id: str,
    dataset: str = "jump-cp",
    n_plates: int = 10,
    zarr_url: str | None = None,
    n_crops_per_image: int = 100,
    embed_batch_size: int = 64,
    n_gpu_workers: int = 4,
    rebuild_index: bool = False,
) -> None:
    """Main ingestion coroutine. Runs the full pipeline asynchronously.

    Spawns Ray tasks for distributed embedding, then builds FAISS index.
    Status updates are written to the session status file.
    """
    import asyncio
    import ray
    import pandas as pd
    from index_manager import build_index

    t0 = time.time()

    def _update(status, msg, n_emb=0, n_tot=0, throughput=0.0, log_lines=None):
        write_status(
            workspace_dir,
            session_id,
            status,
            msg,
            n_embedded=n_emb,
            n_total=n_tot,
            throughput_per_sec=throughput,
            elapsed_seconds=time.time() - t0,
            log_lines=log_lines or [f"[{time.strftime('%H:%M:%S')}] {msg}"],
        )

    try:
        _update(IngestionStatus.PREPARING, "Listing dataset images...")

        # Step 1: list images
        if dataset == "jump-cp":
            records = list_jump_image_paths(n_plates=n_plates)
        elif dataset == "zarr" and zarr_url:
            records = list_zarr_paths(zarr_url)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # Group CP records by (plate, well, site) for multi-channel loading
        if dataset == "jump-cp":
            from itertools import groupby

            key_fn = lambda r: (r["plate"], r["well"], r["site"])
            sorted_records = sorted(records, key=key_fn)
            grouped = {k: list(v) for k, v in groupby(sorted_records, key=key_fn)}
            image_groups = list(grouped.values())
        else:
            image_groups = [[r] for r in records]

        n_total_images = len(image_groups)
        n_total_expected = n_total_images * n_crops_per_image
        _update(
            IngestionStatus.PREPARING,
            f"Found {n_total_images} images ({n_total_expected:,} cells expected). Starting embedding...",
            n_tot=n_total_expected,
        )

        # Step 2: Distributed embedding with Ray
        @ray.remote(num_gpus=1, num_cpus=2)
        def embed_image_group(
            image_group_records: list[dict], n_crops: int
        ) -> list[dict] | None:
            """Ray remote task: load → crop → embed one image group."""
            import sys, os

            # Ensure local modules are importable inside the Ray task
            app_dir = os.path.dirname(__file__)
            if app_dir not in sys.path:
                sys.path.insert(0, app_dir)

            from normalizer import to_rgb_uint8
            from embedder import DINOv2Embedder
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            embedder = DINOv2Embedder(device=device, fp16=(device == "cuda"))
            embedder.load()

            # Load image
            if len(image_group_records) == 1 and image_group_records[0].get("zarr_url"):
                rec = image_group_records[0]
                try:
                    import zarr, s3fs

                    fs = s3fs.S3FileSystem(anon=True)
                    store = zarr.open(s3fs.S3Map(rec["zarr_url"], s3=fs), mode="r")
                    arr = store[rec["array_key"]]
                    z = rec["z_slice"]
                    img = np.array(arr[z] if arr.ndim == 3 else arr[:, z, :])
                except Exception:
                    return None
                rgb = to_rgb_uint8(img)
                crops = extract_cell_crops(img, n_crops=n_crops)
                if not crops:
                    crops = [img]
                rgb_crops = [to_rgb_uint8(c) for c in crops]
                meta_base = {
                    k: v
                    for k, v in rec.items()
                    if k not in ("zarr_url", "array_key", "z_slice")
                }
            else:
                mc_img, meta_base = load_jump_image_multichannel(image_group_records)
                if mc_img is None:
                    return None
                crops = extract_cell_crops(mc_img, n_crops=n_crops)
                if not crops:
                    return None
                rgb_crops = [to_rgb_uint8(c) for c in crops]

            embeddings = embedder.embed_batch(rgb_crops)

            results = []
            for i, (emb, crop_rgb) in enumerate(zip(embeddings, rgb_crops)):
                from PIL import Image
                import base64, io

                thumb = Image.fromarray(crop_rgb).resize((96, 96))
                buf = io.BytesIO()
                thumb.save(buf, format="PNG")
                thumb_b64 = base64.b64encode(buf.getvalue()).decode()

                results.append(
                    {
                        **meta_base,
                        "crop_idx": i,
                        "embedding": emb.tolist(),
                        "thumbnail_b64": thumb_b64,
                    }
                )
            return results

        # Launch Ray tasks in batches to control memory
        all_results = []
        futures = []
        batch_size = max(1, n_gpu_workers * 4)

        for i, group in enumerate(image_groups):
            if is_stop_requested(workspace_dir, session_id):
                _update(
                    IngestionStatus.STOPPED,
                    "Stopped by user.",
                    n_emb=len(all_results),
                    n_tot=n_total_expected,
                )
                return

            futures.append(embed_image_group.remote(group, n_crops_per_image))

            # Collect when we have a full batch
            if len(futures) >= batch_size or i == len(image_groups) - 1:
                done_results = await asyncio.get_event_loop().run_in_executor(
                    None, ray.get, futures
                )
                futures = []
                for res in done_results:
                    if res:
                        all_results.extend(res)

                elapsed = time.time() - t0
                throughput = len(all_results) / max(elapsed, 1)
                eta = (n_total_expected - len(all_results)) / max(throughput, 0.1)
                log_line = (
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"{i+1}/{n_total_images} images · "
                    f"{len(all_results):,} cells · "
                    f"{throughput:.0f} cells/s · "
                    f"ETA {int(eta//60)}m{int(eta%60):02d}s"
                )
                _update(
                    IngestionStatus.RUNNING,
                    f"Embedded {len(all_results):,} cells from {i+1}/{n_total_images} images",
                    n_emb=len(all_results),
                    n_tot=n_total_expected,
                    throughput=throughput,
                    log_lines=[log_line],
                )

        if not all_results:
            _update(IngestionStatus.FAILED, "No cells successfully embedded.")
            return

        # Step 3: Build FAISS index
        _update(
            IngestionStatus.BUILDING_INDEX,
            f"Building FAISS index for {len(all_results):,} cells...",
            n_emb=len(all_results),
            n_tot=len(all_results),
        )

        embeddings_arr = np.array(
            [r["embedding"] for r in all_results], dtype=np.float32
        )
        metadata_rows = [
            {k: v for k, v in r.items() if k not in ("embedding", "thumbnail_b64")}
            for r in all_results
        ]
        # Store thumbnails separately (large)
        thumbnails = [r.get("thumbnail_b64", "") for r in all_results]

        df = pd.DataFrame(metadata_rows)
        df["idx"] = np.arange(len(df))

        # Save thumbnails as a compressed numpy file (separately from parquet for speed)
        out_dir = Path(workspace_dir) / "cell_search"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "thumbnails.npy", np.array(thumbnails, dtype=object))

        stats = await asyncio.get_event_loop().run_in_executor(
            None, build_index, embeddings_arr, df, workspace_dir
        )

        elapsed = time.time() - t0
        _update(
            IngestionStatus.COMPLETED,
            f"Done! {len(all_results):,} cells indexed in {elapsed/60:.1f} min. "
            f"Index: {stats['index_type']}, {stats['index_size_mb']:.0f} MB",
            n_emb=len(all_results),
            n_tot=len(all_results),
            throughput=len(all_results) / elapsed,
            index_stats=stats,
        )

    except Exception as e:
        logger.exception("Ingestion failed: %s", e)
        _update(IngestionStatus.FAILED, f"Ingestion failed: {e}")
        raise
