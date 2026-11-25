"""
Mitochondria Analysis BioEngine App

Segments mitochondria in 2D EM images and returns instance labels + morphological
properties. Inference is delegated to the existing `bioimage-io/model-runner` service
(which runs the `poisonous-spider` model on GPU). This deployment runs on CPU only and
requires no ML framework dependencies.

Pipeline:
  1. Tile input image into 512×512 chunks (with overlap)
  2. Upload each tile to S3, call model-runner.infer(), download probability map
  3. Stitch tiles with Gaussian-blended probability accumulation
  4. Threshold → morphological closing → remove small objects → connected-component labels
  5. Per-instance shape analysis via skimage.regionprops

Required environment variable (set via --env at deploy time):
  HYPHA_TOKEN — Hypha auth token (typically already set in the worker environment)
"""

import asyncio
import io
import logging
import os
import time
from datetime import datetime
from typing import Dict, Union

import numpy as np
from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from ray import serve

logger = logging.getLogger("ray.serve")

_MODEL_ID = "poisonous-spider"
_SERVER_URL = "https://hypha.aicell.io"
_WORKSPACE = "bioimage-io"
_MODEL_RUNNER = "bioimage-io/model-runner"


@serve.deployment(
    ray_actor_options={
        "num_cpus": 2,
        "num_gpus": 0,
        "memory": 4 * 1024**3,
    }
)
class MitoAnalysisDeployment:
    def __init__(self) -> None:
        self.start_time = time.time()
        self._model_runner = None  # set in async_init

    async def async_init(self) -> None:
        """Connect to the BioEngine model-runner service."""
        from hypha_rpc import connect_to_server

        token = os.environ.get("HYPHA_TOKEN")
        if not token:
            raise RuntimeError("HYPHA_TOKEN environment variable is not set.")

        logger.info(f"Connecting to {_SERVER_URL} / {_WORKSPACE} ...")
        server = await connect_to_server(
            {
                "server_url": _SERVER_URL,
                "token": token,
                "workspace": _WORKSPACE,
            }
        )
        self._model_runner = await server.get_service(_MODEL_RUNNER)
        logger.info(f"Connected to model-runner. Using model: {_MODEL_ID}")

    async def test_deployment(self) -> None:
        """Smoke-test: run inference on a 64×64 synthetic tile."""
        test_img = np.random.rand(64, 64).astype(np.float32)
        prob = await self._infer_full(test_img)
        assert prob.shape == (64, 64), f"Unexpected prob shape: {prob.shape}"
        logger.info("test_deployment passed.")

    async def check_health(self) -> None:
        if self._model_runner is None:
            raise RuntimeError("model-runner not connected.")

    # ── Inference helpers ─────────────────────────────────────────────────────

    async def _infer_full(self, img_norm: np.ndarray) -> np.ndarray:
        """
        Submit a single image (H×W float32) to the model-runner.
        Uploads via presigned S3 URL, calls infer(), downloads result.
        Returns a squeezed float32 probability map (H, W).
        """
        import requests as _requests

        inp = img_norm[np.newaxis, np.newaxis].astype(np.float32)  # (1,1,H,W)

        upload = await self._model_runner.get_upload_url(file_type=".npy")
        buf = io.BytesIO()
        np.save(buf, inp)
        _requests.put(upload["upload_url"], data=buf.getvalue(), timeout=30)

        result = await self._model_runner.infer(
            model_id=_MODEL_ID,
            inputs=upload["file_path"],
            return_download_url=True,
        )
        dl_url = list(result.values())[0] if isinstance(result, dict) else result
        resp = _requests.get(dl_url, timeout=30)
        prob = np.load(io.BytesIO(resp.content)).squeeze()
        return prob.astype(np.float32)

    async def _infer_tiled(
        self,
        image_norm: np.ndarray,
        tile_size: int = 512,
        overlap: int = 64,
        max_concurrent: int = 3,
    ) -> np.ndarray:
        """
        Tile a large image, run parallel inference on each tile, stitch with
        Gaussian-blended probability accumulation.
        Returns a float32 probability map the same shape as image_norm.
        """
        H, W = image_norm.shape
        stride = tile_size - overlap
        sem = asyncio.Semaphore(max_concurrent)

        # Gaussian blend window
        yy = np.linspace(-1, 1, tile_size)
        xx = np.linspace(-1, 1, tile_size)
        weight_win = np.outer(np.exp(-2 * yy**2), np.exp(-2 * xx**2)).astype(np.float32)

        prob_acc = np.zeros((H, W), dtype=np.float64)
        weight_acc = np.zeros((H, W), dtype=np.float64)
        lock = asyncio.Lock()

        async def process_tile(y0, x0):
            y1, x1 = min(y0 + tile_size, H), min(x0 + tile_size, W)
            tile = image_norm[y0:y1, x0:x1].copy()
            th, tw = tile.shape
            if th < tile_size or tw < tile_size:
                tile = np.pad(
                    tile, ((0, tile_size - th), (0, tile_size - tw)), mode="reflect"
                )

            async with sem:
                prob = await self._infer_full(tile)

            prob = prob[:th, :tw]
            w = weight_win[:th, :tw]
            async with lock:
                prob_acc[y0:y1, x0:x1] += prob * w
                weight_acc[y0:y1, x0:x1] += w

        tile_coords = [
            (y0, x0) for y0 in range(0, H, stride) for x0 in range(0, W, stride)
        ]
        await asyncio.gather(*[process_tile(y0, x0) for y0, x0 in tile_coords])

        return np.divide(prob_acc, weight_acc, where=weight_acc > 0).astype(np.float32)

    # ── Post-processing ───────────────────────────────────────────────────────

    @staticmethod
    def _prob_to_instances(prob: np.ndarray) -> np.ndarray:
        """Threshold → closing → remove small → watershed → instance labels."""
        import scipy.ndimage as ndi
        from skimage import morphology, measure, segmentation, feature

        binary = morphology.remove_small_objects(prob > 0.5, min_size=300)
        if not binary.any():
            return np.zeros(prob.shape, dtype=np.int32)
        closed = ndi.binary_closing(binary, structure=morphology.disk(4))
        dist = ndi.distance_transform_edt(closed)
        coords = feature.peak_local_max(dist, min_distance=8, labels=closed)
        m = np.zeros(closed.shape, dtype=bool)
        if len(coords):
            m[tuple(coords.T)] = True
        labels = segmentation.watershed(-dist, measure.label(m), mask=closed)
        return labels.astype(np.int32)

    # ── Public API ────────────────────────────────────────────────────────────

    @schema_method
    async def ping(self) -> dict:
        """Return service status."""
        return {
            "status": "ok",
            "model": _MODEL_ID,
            "model_runner": _MODEL_RUNNER,
            "uptime_s": round(time.time() - self.start_time, 1),
            "timestamp": datetime.now().isoformat(),
        }

    @schema_method
    async def analyze(
        self,
        image: list = Field(
            ...,
            description=(
                "2D grayscale EM image as a nested list (H×W). "
                "Values may be uint8 (0-255) or float. Use numpy_array.tolist()."
            ),
        ),
        pixel_size_nm: float = Field(
            5.0,
            description="Pixel size in nm used for physical area measurements.",
        ),
        tile_size: int = Field(
            512,
            description="Tile edge length in pixels for tiled inference.",
        ),
        overlap: int = Field(
            64,
            description="Overlap between adjacent tiles in pixels.",
        ),
    ) -> dict:
        """
        Segment mitochondria in a 2D EM image and return instance labels with
        morphological properties.

        Args:
          image: 2D grayscale EM image as a nested list (H×W). Values may be
                 uint8 (0-255) or float. Use numpy_array.tolist().
          pixel_size_nm: Pixel size in nm used for physical area measurements.
          tile_size: Tile edge length in pixels for tiled inference.
          overlap: Overlap between adjacent tiles in pixels.

        Returns:
          labels          - H×W integer list; each non-zero value is one instance
          properties      - per-instance dict: area_um2, aspect_ratio, eccentricity, centroids
          n_mitochondria  - number of detected instances
          image_shape     - [H, W]
          pixel_size_nm   - echo of input parameter
          model           - model ID used
          processing_time_s
        """
        t0 = time.time()

        image_np = np.array(image, dtype=np.float32)
        if image_np.ndim != 2:
            raise ValueError(f"Expected 2-D image, got shape {image_np.shape}.")

        H, W = image_np.shape
        p1, p99 = np.percentile(image_np, [1, 99])
        image_norm = np.clip((image_np - p1) / (p99 - p1 + 1e-6), 0, 1).astype(
            np.float32
        )

        logger.info(f"Analyzing {image_np.shape} image, tile_size={tile_size}")

        # Run tiled inference (or single call for small images)
        if H <= tile_size and W <= tile_size:
            prob = await self._infer_full(image_norm)
        else:
            prob = await self._infer_tiled(
                image_norm, tile_size=tile_size, overlap=overlap
            )

        labels = self._prob_to_instances(prob)
        n_mito = int(labels.max())
        logger.info(f"Found {n_mito} mitochondria instances.")

        pixel_um = pixel_size_nm / 1000.0
        if n_mito > 0:
            from skimage.measure import regionprops

            labels_l, area_l, ar_l, ecc_l, cy_l, cx_l = [], [], [], [], [], []
            for rp in regionprops(labels):
                labels_l.append(int(rp.label))
                area_l.append(float(rp.area) * pixel_um**2)
                major, minor = rp.axis_major_length, rp.axis_minor_length
                ar_l.append(float(major / (minor + 1e-6)))
                ecc_l.append(float(rp.eccentricity))
                cy, cx = rp.centroid
                cy_l.append(float(cy))
                cx_l.append(float(cx))
            properties = {
                "label": labels_l,
                "area_um2": area_l,
                "aspect_ratio": ar_l,
                "eccentricity": ecc_l,
                "centroid_y": cy_l,
                "centroid_x": cx_l,
            }
        else:
            properties = {
                "label": [],
                "area_um2": [],
                "aspect_ratio": [],
                "eccentricity": [],
                "centroid_y": [],
                "centroid_x": [],
            }

        return {
            "labels": labels.tolist(),
            "properties": properties,
            "n_mitochondria": n_mito,
            "image_shape": [H, W],
            "pixel_size_nm": pixel_size_nm,
            "model": _MODEL_ID,
            "processing_time_s": round(time.time() - t0, 2),
        }
