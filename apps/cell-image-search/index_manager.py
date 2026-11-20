"""
FAISS vector index management for cell image search.

Index selection strategy:
  < 100K cells   → IndexFlatIP (exact, cosine via inner product on L2-normed vectors)
  100K–5M cells  → IndexIVFFlat (approximate, ~5× faster, <1% recall loss)
  > 5M cells     → IndexIVFPQ  (compressed, 768-dim → 96 bytes, ~100× memory reduction)

All vectors must be L2-normalised before insertion (use cosine via inner product).

Storage layout at workspace_dir/cell_search/:
  cell_search.index       — FAISS binary index
  metadata.parquet        — per-cell metadata (compound, plate, well, image_path, ...)
  umap_cache.npz          — pre-computed UMAP coords for a sample
  index_info.json         — build stats (n_cells, dim, index_type, build_time, ...)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

EMBED_DIM = 768


def _index_dir(workspace_dir: str) -> Path:
    return Path(workspace_dir) / "cell_search"


def build_index(
    embeddings: np.ndarray,
    metadata_df: "pd.DataFrame",
    workspace_dir: str,
    n_cells_total: int | None = None,
) -> dict[str, Any]:
    """Build and save FAISS index from embeddings.

    Automatically selects IndexFlatIP / IVFFlat / IVFPQ based on n_cells.

    Args:
        embeddings: (N, 768) float32, L2-normalised.
        metadata_df: DataFrame with N rows, columns: compound, plate, well, etc.
        workspace_dir: Root dir to store index files.
        n_cells_total: If building incrementally, pass expected final count
                       so the correct index type is selected upfront.

    Returns:
        dict with build stats.
    """
    import faiss
    import pandas as pd

    t0 = time.time()
    n, d = embeddings.shape
    n_target = n_cells_total or n
    out_dir = _index_dir(workspace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building FAISS index: n=%d, d=%d", n, d)

    if n_target < 100_000:
        index = faiss.IndexFlatIP(d)
        index_type = "FlatIP"
    elif n_target < 5_000_000:
        nlist = min(4096, max(64, int(np.sqrt(n_target))))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        logger.info("Training IVFFlat nlist=%d on %d vectors...", nlist, n)
        index.train(embeddings[: min(n, 256 * nlist)])
        index_type = f"IVFFlat(nlist={nlist})"
    else:
        # IVFPQ: 96 subquantizers × 8 bits → 96 bytes per vector
        nlist = min(65536, max(4096, int(np.sqrt(n_target))))
        m = 96  # must divide d=768; 768/96=8 ✓
        nbits = 8
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        logger.info("Training IVFPQ nlist=%d, m=%d on %d vectors...", nlist, m, n)
        train_n = min(n, max(256 * nlist, 1_000_000))
        index.train(embeddings[:train_n])
        index_type = f"IVFPQ(nlist={nlist},m={m})"

    index.add(embeddings)
    elapsed = time.time() - t0

    # Save index
    index_path = out_dir / "cell_search.index"
    faiss.write_index(index, str(index_path))
    logger.info("Saved FAISS index to %s (%.1fs)", index_path, elapsed)

    # Save metadata
    meta_path = out_dir / "metadata.parquet"
    metadata_df.to_parquet(meta_path, index=False)
    logger.info("Saved metadata (%d rows) to %s", len(metadata_df), meta_path)

    stats = {
        "n_cells": n,
        "embed_dim": d,
        "index_type": index_type,
        "index_size_mb": index_path.stat().st_size / 1024**2,
        "build_seconds": elapsed,
        "build_time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "index_info.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_index(workspace_dir: str) -> tuple["faiss.Index", "pd.DataFrame", dict]:
    """Load FAISS index + metadata from workspace_dir.

    Returns:
        (index, metadata_df, info_dict)
    Raises:
        FileNotFoundError if index not yet built.
    """
    import faiss
    import pandas as pd

    out_dir = _index_dir(workspace_dir)
    index_path = out_dir / "cell_search.index"
    meta_path = out_dir / "metadata.parquet"
    info_path = out_dir / "index_info.json"

    if not index_path.exists():
        raise FileNotFoundError(f"No FAISS index at {index_path}")

    index = faiss.read_index(str(index_path))
    # Set nprobe for IVF indices
    if hasattr(index, "nprobe"):
        index.nprobe = 64
    elif hasattr(index, "quantizer"):
        pass

    metadata_df = pd.read_parquet(meta_path) if meta_path.exists() else None
    info = json.loads(info_path.read_text()) if info_path.exists() else {}

    logger.info(
        "Loaded index: %d vectors, type=%s", index.ntotal, info.get("index_type")
    )
    return index, metadata_df, info


def search_index(
    index: "faiss.Index",
    metadata_df: "pd.DataFrame",
    query_embedding: np.ndarray,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Search the index for nearest neighbours.

    Args:
        index: Loaded FAISS index.
        metadata_df: Metadata DataFrame (row order matches index).
        query_embedding: (768,) float32, L2-normalised.
        top_k: Number of results to return.

    Returns:
        List of result dicts with score + metadata fields.
    """
    query = query_embedding.reshape(1, -1).astype(np.float32)
    scores, indices = index.search(query, top_k)
    scores = scores[0]
    indices = indices[0]

    results = []
    for rank, (score, idx) in enumerate(zip(scores, indices)):
        if idx < 0:
            continue  # FAISS returns -1 for empty slots
        meta = {}
        if metadata_df is not None and idx < len(metadata_df):
            meta = metadata_df.iloc[idx].to_dict()
        results.append(
            {
                "rank": rank + 1,
                "score": float(score),
                "faiss_idx": int(idx),
                **meta,
            }
        )
    return results


def compute_umap(
    workspace_dir: str,
    n_samples: int = 10_000,
    random_state: int = 42,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Compute or load cached UMAP projection of a random sample.

    Returns:
        Dict with keys: x (list), y (list), labels (list), colors (list), n_total.
    """
    import faiss
    import pandas as pd

    out_dir = _index_dir(workspace_dir)
    cache_path = out_dir / "umap_cache.npz"
    meta_path = out_dir / "metadata.parquet"
    index_path = out_dir / "cell_search.index"

    if cache_path.exists() and not force_recompute:
        data = np.load(cache_path, allow_pickle=True)
        return {
            "x": data["x"].tolist(),
            "y": data["y"].tolist(),
            "labels": data["labels"].tolist(),
            "colors": data["colors"].tolist(),
            "n_total": int(data["n_total"]),
        }

    if not index_path.exists():
        return {"x": [], "y": [], "labels": [], "colors": [], "n_total": 0}

    index = faiss.read_index(str(index_path))
    n_total = index.ntotal
    n_samples = min(n_samples, n_total)

    # Extract embeddings for a random sample
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(n_total, size=n_samples, replace=False)
    sample_idx.sort()

    # Reconstruct vectors (works for Flat and IVFPQ indices)
    try:
        vecs = index.reconstruct_batch(sample_idx.tolist())
    except Exception:
        # Fallback: reconstruct one at a time
        vecs = np.vstack([index.reconstruct(int(i)) for i in sample_idx])

    # UMAP
    try:
        from umap import UMAP

        logger.info("Computing UMAP on %d vectors...", n_samples)
        t0 = time.time()
        reducer = UMAP(
            n_neighbors=15, min_dist=0.1, metric="cosine", random_state=random_state
        )
        coords = reducer.fit_transform(vecs)
        logger.info("UMAP done in %.1fs", time.time() - t0)
    except ImportError:
        logger.warning("umap-learn not installed; returning PCA instead")
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2).fit_transform(vecs)

    # Labels from metadata
    labels = ["unknown"] * n_samples
    colors = ["#888888"] * n_samples
    if meta_path.exists():
        df = pd.read_parquet(meta_path)
        moa_col = (
            "moa_class"
            if "moa_class" in df.columns
            else ("compound" if "compound" in df.columns else None)
        )
        if moa_col:
            unique_labels = df[moa_col].unique().tolist()
            palette = _generate_palette(len(unique_labels))
            color_map = {
                lbl: palette[i % len(palette)] for i, lbl in enumerate(unique_labels)
            }
            for i, idx in enumerate(sample_idx):
                if idx < len(df):
                    lbl = str(df.iloc[idx].get(moa_col, "unknown"))
                    labels[i] = lbl
                    colors[i] = color_map.get(lbl, "#888888")

    result = {
        "x": coords[:, 0].tolist(),
        "y": coords[:, 1].tolist(),
        "labels": labels,
        "colors": colors,
        "n_total": n_total,
    }
    np.savez(
        cache_path,
        x=coords[:, 0],
        y=coords[:, 1],
        labels=np.array(labels),
        colors=np.array(colors),
        n_total=np.array(n_total),
    )
    return result


def _generate_palette(n: int) -> list[str]:
    """Generate n visually distinct hex colors."""
    colors = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#a65628",
        "#f781bf",
        "#999999",
        "#66c2a5",
        "#fc8d62",
        "#8da0cb",
        "#e78ac3",
        "#a6d854",
        "#ffd92f",
        "#e5c494",
        "#b3b3b3",
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
    ]
    if n <= len(colors):
        return colors[:n]
    import colorsys

    extended = list(colors)
    for i in range(n - len(colors)):
        h = i / (n - len(colors))
        r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.85)
        extended.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return extended
