"""
DINOv2 ViT-B/14 embedding model for biological images.

DINOv2 is a self-supervised vision transformer pretrained on 142M curated images.
It produces high-quality 768-dimensional feature vectors without any task-specific
fine-tuning, making it ideal for unsupervised morphological profiling.

Reference: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision"
https://arxiv.org/abs/2304.07193

Throughput: ~500 images/sec on A100 GPU (batch_size=64, fp16)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DINOv2Embedder:
    """Wraps DINOv2 ViT-B/14 for batch image embedding.

    Loaded lazily on first use. Supports fp16 for 2× throughput on modern GPUs.

    Usage:
        embedder = DINOv2Embedder(device="cuda", fp16=True)
        embedder.load()
        embeddings = embedder.embed_batch(list_of_rgb_uint8_arrays)
        # → np.ndarray shape (N, 768), dtype float32, L2-normalised
    """

    MODEL_NAME = "dinov2_vitb14"  # 768-dim; vitl14 = 1024-dim (slower, better)
    EMBED_DIM = 768
    INPUT_SIZE = 224

    def __init__(self, device: str = "cuda", fp16: bool = True) -> None:
        self.device = device
        self.fp16 = fp16
        self._model: Any = None

    def load(self) -> None:
        """Download (first time) and load the DINOv2 model."""
        import torch

        logger.info(
            "Loading DINOv2 %s on %s (fp16=%s)", self.MODEL_NAME, self.device, self.fp16
        )
        self._model = torch.hub.load(
            "facebookresearch/dinov2",
            self.MODEL_NAME,
            pretrained=True,
        )
        self._model.eval()
        self._model = self._model.to(self.device)
        if self.fp16 and self.device != "cpu":
            self._model = self._model.half()
        logger.info("DINOv2 loaded, embed_dim=%d", self.EMBED_DIM)

    def embed_batch(
        self,
        images_rgb: list[np.ndarray],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Embed a list of RGB uint8 images.

        Args:
            images_rgb: List of (H, W, 3) uint8 numpy arrays (any resolution).
            batch_size: GPU mini-batch size.

        Returns:
            np.ndarray shape (N, 768), float32, L2-normalised unit vectors.
        """
        import torch
        from normalizer import to_dinov2_tensor

        if self._model is None:
            self.load()

        all_embeddings = []
        for i in range(0, len(images_rgb), batch_size):
            batch_imgs = images_rgb[i : i + batch_size]
            tensors = [
                to_dinov2_tensor(img, size=self.INPUT_SIZE) for img in batch_imgs
            ]
            batch = torch.cat(tensors, dim=0).to(self.device)  # (B, 3, 224, 224)
            if self.fp16 and self.device != "cpu":
                batch = batch.half()
            with torch.no_grad():
                feats = self._model(batch)  # (B, 768)
            feats = feats.float().cpu().numpy()
            # L2-normalise for cosine similarity via inner product
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-9)
            feats = feats / norms
            all_embeddings.append(feats)

        return np.vstack(all_embeddings).astype(np.float32)

    def embed_single(self, image_rgb: np.ndarray) -> np.ndarray:
        """Embed one image. Returns (768,) float32 array."""
        return self.embed_batch([image_rgb])[0]


# ---------------------------------------------------------------------------
# Ray actor wrapper — used by ingestion pipeline as a stateful worker
# ---------------------------------------------------------------------------


class EmbeddingActor:
    """Ray actor that holds the model in GPU memory and processes batches.

    Designed to be used with ray.data.map_batches(..., concurrency=n_gpus).
    """

    def __init__(self) -> None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._embedder = DINOv2Embedder(device=device, fp16=(device == "cuda"))
        self._embedder.load()

    def __call__(self, batch: dict) -> dict:
        """Process a batch from ray.data pipeline.

        Expects batch["image_rgb"] = list of (H, W, 3) uint8 arrays.
        Returns batch with added "embedding" column.
        """
        images = batch["image_rgb"]
        embeddings = self._embedder.embed_batch(list(images), batch_size=64)
        batch["embedding"] = list(embeddings)
        return batch
