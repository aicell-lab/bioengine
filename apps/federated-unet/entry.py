"""One site of a federated U-Net experiment.

The same app is deployed once per participating site. Each instance downloads
its own dataset straight from the public source into its replica, trains
locally, and exchanges only ``state_dict`` bytes with the other sites through a
Hypha artifact. Aggregation is driven from outside by ``run_federated.py``,
which never sees an image — it reads checkpoints, averages them, and writes the
average back.

Three instances make up a run: two sites holding disjoint domains, and a third
"pooled" instance holding both, which is the deliberate premise violation used
as an upper bound. The pooled instance is a control, not a participant.
"""

import asyncio
import base64
import io
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import bioengine
from hypha_rpc import connect_to_server
from pydantic import Field

logger = bioengine.logger

SERVER_URL = "https://hypha.aicell.io"


def _read_pip(name: str) -> List[str]:
    text = (Path(__file__).parent / name).read_text()
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@bioengine.app(
    num_cpus=2,
    gpu_memory_mb=6144,
    memory_mb=12 * 1024,
    pip=_read_pip("requirements-entry.txt"),
    max_ongoing_requests=4,
    autoscaling_config={"min_replicas": 1, "initial_replicas": 1, "max_replicas": 1},
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    graceful_shutdown_timeout_s=120.0,
)
class FederatedUNetSite:
    """A single federation site: local data, local training, weights-only egress."""

    def __init__(
        self,
        site_name: str = "unnamed-site",
        datasets: Optional[List[str]] = None,
        role: str = "participant",
    ) -> None:
        self.site_name = site_name
        # A pooled-oracle instance is handed both domains; a real site gets one.
        self.dataset_names = list(datasets or ["dsb2018-fluo"])
        self.role = role
        self.start_time = time.time()
        self._hypha_token = os.getenv("HYPHA_TOKEN")
        if not self._hypha_token:
            raise RuntimeError("HYPHA_TOKEN environment variable is not set")
        self._hypha = None
        self._artifact_manager = None
        self._lock = asyncio.Lock()
        self._data: Dict[str, Any] = {}
        self._model = None
        self._device = None
        self._log = None
        self._history: List[Dict[str, Any]] = []
        self._cache_dir = Path.home() / "federated-unet-data"

    @bioengine.async_init
    async def load(self) -> None:
        import torch

        from checkpoints import TransportLog

        self._log = TransportLog()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._hypha = await connect_to_server(
            {"server_url": SERVER_URL, "token": self._hypha_token}
        )
        self._artifact_manager = await self._hypha.get_service("public/artifact-manager")
        logger.info(f"[{self.site_name}] ready on {self._device}")

    @bioengine.smoke_test
    async def smoke(self) -> None:
        status = await self.get_status()
        assert status["torch"]["device"] in ("cuda", "cpu")

    def _store(self, run_artifact_id: str):
        from checkpoints import CheckpointStore

        return CheckpointStore(self._artifact_manager, run_artifact_id, self._log)

    @bioengine.method
    async def get_status(self) -> Dict[str, Any]:
        """Site identity, hardware and software provenance, and current state."""
        import torch

        return {
            "site_name": self.site_name,
            "role": self.role,
            "uptime_s": time.time() - self.start_time,
            "datasets_configured": self.dataset_names,
            "datasets_loaded": {
                name: {
                    "n_train": len(d["train"]),
                    "n_val": len(d["val"]),
                    "n_test": len(d["test"]),
                    "n_available": d["n_available"],
                    "objects": d["objects"],
                    "source": d["source"],
                    "licence": d["licence"],
                    "citation": d["citation"],
                    "split_fingerprint": d["split_fingerprint"],
                    "mean_foreground_fraction": d["mean_foreground_fraction"],
                }
                for name, d in self._data.items()
            },
            "model_initialised": self._model is not None,
            "torch": {
                # str(): torch.__version__ is a torch-defined str subclass, and
                # the proxy that unpickles this reply has no torch installed.
                "version": str(torch.__version__),
                "device": self._device.type,
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "cuda_version": torch.version.cuda,
            },
            "host": {
                "hostname": socket.gethostname(),
                "node_ip": socket.gethostbyname(socket.gethostname()),
                "python": platform.python_version(),
            },
            "bioengine_version": getattr(bioengine, "__version__", "unknown"),
            "history_entries": len(self._history),
        }

    @bioengine.method
    async def prepare_data(
        self,
        n_train: int = Field(55, description="Training images per dataset"),
        n_val: int = Field(15, description="Validation images per dataset"),
        n_test: int = Field(25, description="Held-out test images per dataset"),
        split_seed: int = Field(20260905, description="Seed for the train/val/test partition; identical across sites and arms"),
    ) -> Dict[str, Any]:
        """Download this site's dataset(s) into the replica and cut the splits.

        Images are fetched from their public source directly into the replica
        and never leave it again.
        """
        from datasets import load_dataset

        def _work() -> Dict[str, Any]:
            return {
                name: load_dataset(
                    name,
                    cache_dir=self._cache_dir,
                    n_train=n_train,
                    n_val=n_val,
                    n_test=n_test,
                    split_seed=split_seed,
                )
                for name in self.dataset_names
            }

        async with self._lock:
            loop = asyncio.get_running_loop()
            self._data = await loop.run_in_executor(None, _work)
        status = await self.get_status()
        return status["datasets_loaded"]

    @bioengine.method
    async def init_model(
        self,
        seed: int = Field(0, description="Seed for parameter initialisation; identical across sites so round 0 starts from one common model"),
    ) -> Dict[str, Any]:
        """Build a freshly initialised model, discarding any current weights."""
        from unet import ARCH, build_model, signature

        async with self._lock:
            self._model = build_model(seed=seed)
            self._history = []
        params = sum(p.numel() for p in self._model.parameters())
        return {
            "architecture": ARCH,
            "n_parameters": params,
            "n_tensors": len(signature(self._model.state_dict())),
            "init_seed": seed,
        }

    @bioengine.method
    async def train(
        self,
        steps: int = Field(100, description="Optimiser steps to run in this call"),
        lr: float = Field(1e-3, description="Adam learning rate"),
        batch_size: int = Field(8, description="Crops per step"),
        crop: int = Field(256, description="Training crop size in pixels"),
        seed: int = Field(0, description="Seed for crop sampling and augmentation"),
        tag: str = Field("", description="Free-form label recorded in this site's history"),
    ) -> Dict[str, Any]:
        """Run local training on this site's data. Nothing leaves the site."""
        from training import train_steps

        if self._model is None:
            raise RuntimeError("call init_model or pull_weights before train")
        if not self._data:
            raise RuntimeError("call prepare_data before train")

        pairs = [pair for d in self._data.values() for pair in d["train"]]

        def _work() -> Dict[str, Any]:
            return train_steps(
                self._model, pairs, steps=steps, lr=lr, batch_size=batch_size,
                crop=crop, device=self._device, seed=seed,
            )

        async with self._lock:
            started = time.time()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, _work)
            result.update(
                site=self.site_name,
                tag=tag,
                n_train_images=len(pairs),
                datasets=list(self._data),
                wall_time_s=time.time() - started,
                lr=lr, batch_size=batch_size, crop=crop, seed=seed,
            )
            self._history.append(result)
        return result

    @bioengine.method
    async def evaluate(
        self,
        split: str = Field("test", description="Which split to score: train, val or test"),
    ) -> Dict[str, Any]:
        """Score the current weights on a local split, per dataset and per image."""
        from training import evaluate as run_eval

        if self._model is None:
            raise RuntimeError("no model loaded")
        if not self._data:
            raise RuntimeError("call prepare_data before evaluate")

        def _work() -> Dict[str, Any]:
            return {
                name: run_eval(self._model, d[split], device=self._device)
                for name, d in self._data.items()
            }

        async with self._lock:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, _work)
        for name, result in results.items():
            result["split"] = split
            result["split_fingerprint"] = self._data[name]["split_fingerprint"]
            result["site"] = self.site_name
        return results

    @bioengine.method
    async def push_weights(
        self,
        run_artifact_id: str = Field(..., description="Artifact holding this run's checkpoints"),
        path: str = Field(..., description="File path inside the artifact, e.g. round_01/site-a.pt"),
        note: str = Field("", description="Recorded in the transport log"),
    ) -> Dict[str, Any]:
        """Serialise the current weights and write them to the run artifact.

        This is the only outbound path in the app. Everything it can send is a
        ``state_dict``; there is no code path that writes image data out.
        """
        import torch

        if self._model is None:
            raise RuntimeError("no model to push")
        buffer = io.BytesIO()
        torch.save(
            {k: v.detach().cpu() for k, v in self._model.state_dict().items()}, buffer
        )
        payload = buffer.getvalue()
        n_train = sum(len(d["train"]) for d in self._data.values())
        entry = await self._store(run_artifact_id).put(
            path, payload, kind="model_weights", note=note or f"{self.site_name} weights"
        )
        return {**entry, "n_train_images": n_train, "site": self.site_name}

    @bioengine.method
    async def pull_weights(
        self,
        run_artifact_id: str = Field(..., description="Artifact holding this run's checkpoints"),
        path: str = Field(..., description="File path inside the artifact to load"),
        note: str = Field("", description="Recorded in the transport log"),
    ) -> Dict[str, Any]:
        """Load a checkpoint from the run artifact into this site's model."""
        import torch

        from unet import build_model, signature

        payload = await self._store(run_artifact_id).get(
            path, kind="model_weights", note=note or f"{self.site_name} sync"
        )
        state_dict = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        async with self._lock:
            if self._model is None:
                self._model = build_model(seed=0)
            incoming = signature(state_dict)
            current = signature(self._model.state_dict())
            if incoming != current:
                raise RuntimeError(
                    f"checkpoint architecture does not match this site's model "
                    f"({len(incoming)} vs {len(current)} tensors)"
                )
            self._model.load_state_dict(state_dict)
        return {"loaded": path, "bytes": len(payload), "n_tensors": len(incoming), "site": self.site_name}

    @bioengine.method
    async def get_transport_log(self) -> Dict[str, Any]:
        """Every payload that crossed this site's boundary, with sha256 digests.

        The evidence behind "no image ever left the site" — a caller can check
        it rather than take it on trust.
        """
        dump = self._log.dump()
        dump["site"] = self.site_name
        dump["datasets_ingested_from_public_source"] = [
            {"name": name, "source": d["source"], "licence": d["licence"]}
            for name, d in self._data.items()
        ]
        return dump

    @bioengine.method
    async def preview(
        self,
        dataset: str = Field(..., description="Which loaded dataset to render"),
        split: str = Field("test", description="Which split to render"),
        n: int = Field(3, description="How many images to render"),
    ) -> Dict[str, Any]:
        """Render image / ground truth / prediction triplets as one PNG.

        Returns base64 so a plain RPC client can write the file out; it is a
        picture of this site's own data, so it is only ever pulled by the
        operator, never pushed between sites.
        """
        import numpy as np
        import torch
        from PIL import Image

        if dataset not in self._data:
            raise ValueError(f"{dataset!r} not loaded here; have {sorted(self._data)}")
        pairs = self._data[dataset][split][:n]

        def _work() -> bytes:
            tiles = []
            self._model.to(self._device).eval()
            for image, mask in pairs:
                height, width = image.shape
                padded = np.pad(image, ((0, (-height) % 16), (0, (-width) % 16)), mode="reflect")
                with torch.no_grad():
                    tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(self._device)
                    probs = torch.sigmoid(self._model(tensor))[0, 0, :height, :width].cpu().numpy()
                row = np.concatenate([image, mask, probs], axis=1)
                tiles.append((row * 255).clip(0, 255).astype(np.uint8))
            widthmax = max(t.shape[1] for t in tiles)
            padded_tiles = [
                np.pad(t, ((0, 0), (0, widthmax - t.shape[1])), constant_values=0) for t in tiles
            ]
            buffer = io.BytesIO()
            Image.fromarray(np.concatenate(padded_tiles, axis=0)).save(buffer, format="PNG")
            return buffer.getvalue()

        if self._model is None:
            raise RuntimeError("no model loaded")
        loop = asyncio.get_running_loop()
        png = await loop.run_in_executor(None, _work)
        return {
            "dataset": dataset,
            "split": split,
            "n": len(pairs),
            "layout": "columns are image | ground truth | prediction, one row per image",
            "png_base64": base64.b64encode(png).decode(),
        }

    @bioengine.method
    async def get_history(self) -> List[Dict[str, Any]]:
        """Every training call this site has run since the last init_model."""
        return self._history
