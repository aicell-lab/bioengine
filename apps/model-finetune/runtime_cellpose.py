"""Cellpose GPU runtime — resident cpsam/cpdino model + shared GPU lock + training.

The second, isolated GPU deployment of the model-finetune app. numpy is
irreconcilable between the two backends — micro-sam's python-elf needs numpy>=2
while cellpose pins numpy==1.26.4 — so the Cellpose backend lives in its own Ray
deployment with its own pip env (``requirements-runtime-cellpose.txt``) rather
than sharing the micro-sam ``RuntimeApp``. It serves both Cellpose-SAM (cpsam)
and Cellpose-DINO (cpdino / cpdino-vitb); the CPU ``EntryApp`` composes both
deployments by type hint and routes by ``model_type`` (cellpose types → here,
``vit_*`` → RuntimeApp).

Mirrors ``RuntimeApp``'s contract: a single ``asyncio.Lock`` serialises all GPU
work (serving + training); fine-tuning and export run in **subprocesses** so the
OS reclaims their VRAM on exit; export is CPU-only. Heavy deps (``torch``,
``cellpose``) are imported inside method bodies so the ``@bioengine.app``
decorator module stays introspectable with only ``bioengine[worker]`` + stdlib.
"""

import asyncio
import gc
import os
import subprocess
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import bioengine
import numpy as np

logger = bioengine.logger

# Match CellposeSAMWrapper's eval kwargs so served masks reproduce the exported
# package's self-test output; diameter is cpsam's mean-diameter prior.
_SERVE_DIAMETER = 30.0


def _read_pip(name: str) -> List[str]:
    text = (Path(__file__).parent / name).read_text()
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@bioengine.app(
    num_cpus=2,
    gpu_memory_mb=-1,
    memory_mb=12 * 1024,
    pip=_read_pip("requirements-runtime-cellpose.txt"),
    max_ongoing_requests=10,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 3,
        "target_num_ongoing_requests_per_replica": 1.0,
        "metrics_interval_s": 2.0,
        "look_back_period_s": 10.0,
        "downscale_delay_s": 600,
        "upscale_delay_s": 0.0,
    },
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    graceful_shutdown_timeout_s=600.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class CellposeRuntime:
    """GPU compute for Cellpose-SAM: resident model, shared GPU lock, subprocess
    fine-tuning + CPU export."""

    def __init__(self) -> None:
        self.start_time = time.time()
        self._gpu_lock = asyncio.Lock()
        self._model = None
        self._loaded_key: Optional[str] = None
        self._device_cached: Optional[str] = None
        self._gpu_mem_cached: Optional[Dict[str, Any]] = None
        # Model-soup member state_dicts, keyed by their immutable pool weights_path
        # (content-addressed → no staleness); populated on demand during aggregate.
        self._soup_cache: Dict[str, Dict[str, Any]] = {}
        self._soup_dl_bytes = 0

    async def ping(self) -> Dict[str, Any]:
        """Internal liveness for the entry's readiness check (mirrors RuntimeApp)."""
        return {
            "status": "ok",
            "loaded_checkpoint": self._loaded_key,
            "gpu_busy": self._gpu_lock.locked(),
            "uptime": time.time() - self.start_time,
        }

    async def gpu_memory_info(self) -> Dict[str, Any]:
        """Total/free VRAM of this runtime's GPU, for the entry's training gate
        (mirrors RuntimeApp). Cached after the first sample; total VRAM is static."""
        if self._gpu_mem_cached is None:
            import training

            self._gpu_mem_cached = training.detect_gpu_memory()
        return self._gpu_mem_cached

    def _device(self) -> str:
        if self._device_cached is None:
            try:
                import torch

                self._device_cached = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self._device_cached = "cpu"
        return self._device_cached

    @staticmethod
    def _nd(arr: np.ndarray) -> Dict[str, Any]:
        """Encode an array as hypha-rpc's ndarray wire-dict (numpy-neutral bytes)."""
        arr = np.ascontiguousarray(arr)
        return {
            "_rtype": "ndarray",
            "_rvalue": arr.tobytes(),
            "_rshape": list(arr.shape),
            "_rdtype": arr.dtype.name,
        }

    @staticmethod
    def _to_hwc3(array: np.ndarray) -> np.ndarray:
        """Coerce input to HxWx3 float32 (cellpose normalizes internally)."""
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        elif array.ndim == 3:
            if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3):
                array = np.transpose(array, (1, 2, 0))  # CHW -> HWC
            if array.shape[-1] == 1:
                array = np.concatenate([array, array, array], axis=-1)
            elif array.shape[-1] == 2:
                array = np.concatenate([array, array[..., :1]], axis=-1)
            array = array[..., :3]
        else:
            raise ValueError(
                f"Invalid input image of shape {array.shape}. Expected 2D (HxW) or "
                "3-channel (HxWx3 / 3xHxW)."
            )
        return array.astype(np.float32)

    def _ensure_model(self, checkpoint: Optional[str], model_type: str = "cpsam"):
        """Load a resident CellposeModel, reusing it when the target is unchanged.
        A ``checkpoint`` path serves a fine-tuned session's bare-net state dict
        (cellpose auto-detects cpsam vs cpdino from it); ``None`` loads the base
        model for ``model_type``. Blocking — call via ``asyncio.to_thread`` under
        lock."""
        import training
        from cellpose import models as cpmodels

        # Two different base models both have checkpoint=None, so the reuse key
        # must include the base identity.
        key = checkpoint if checkpoint else f"base:{model_type}"
        if key != self._loaded_key:
            self._release_model()
            gpu = self._device() == "cuda"
            base = training.cellpose_base_model(model_type)
            label = f"finetuned: {checkpoint}" if checkpoint else f"base {base}"
            logger.info(f"🔄 Loading Cellpose model ({label}) gpu={gpu}...")
            pretrained = checkpoint if checkpoint else base
            self._model = cpmodels.CellposeModel(gpu=gpu, pretrained_model=pretrained)
            self._loaded_key = key
            logger.info(f"✅ Cellpose model ({label}) loaded.")
        return self._model

    def _release_model(self) -> None:
        self._model = None
        self._loaded_key = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _segment(self, image, generate_kwargs, checkpoint, model_type="cpsam"):
        """Cellpose instance segmentation → int32 [H,W] mask. Blocking."""
        model = self._ensure_model(checkpoint, model_type)
        img = self._to_hwc3(image)
        eval_kwargs = dict(
            channels=[0, 0], channel_axis=None, diameter=_SERVE_DIAMETER,
            flow_threshold=0.4, cellprob_threshold=0.0, stitch_threshold=0.0,
            batch_size=8, normalize=True, do_3D=False,
        )
        if generate_kwargs.get("min_size") is not None:
            eval_kwargs["min_size"] = generate_kwargs["min_size"]
        if generate_kwargs.get("diameter") is not None:
            eval_kwargs["diameter"] = generate_kwargs["diameter"]
        masks, _flows, _styles = model.eval([img], **eval_kwargs)
        mask = masks[0] if isinstance(masks, list) else masks
        return np.asarray(mask).astype(np.int32)

    # === composition endpoints (called by EntryApp via the runtime handle) ===

    async def auto_segment(
        self, images: List[np.ndarray], model_type: str = "cpsam",
        generate_kwargs: Optional[Dict[str, Any]] = None,
        checkpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Cellpose masks for a batch of (already-resolved) images → wire-dict list.
        ``model_type`` selects the base (cpsam / cpdino / cpdino-vitb) when no
        ``checkpoint`` is given; a fine-tuned ``checkpoint`` self-identifies its
        backbone regardless."""
        generate_kwargs = generate_kwargs or {}
        results: List[Dict[str, Any]] = []
        async with self._gpu_lock:
            for image in images:
                labels = await asyncio.to_thread(
                    self._segment, image, generate_kwargs, checkpoint, model_type
                )
                results.append({"output": self._nd(labels)})
        return results

    # === model-soup aggregation compute (torch + cellpose live here, not the CPU
    # EntryApp; member weights are fetched from the pool's presigned URLs, so no
    # cross-actor local file is ever shared) ===

    def _soup_dir(self) -> Path:
        d = Path.home() / ".bioengine" / "pool_soup"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _unwrap_sd(obj):
        if isinstance(obj, dict) and "model_state_dict" in obj:
            return obj["model_state_dict"]
        if isinstance(obj, dict) and "model_state" in obj:
            return obj["model_state"]
        return obj

    def _fetch_member(self, key: str, url: str) -> Dict[str, Any]:
        """Download + cache a member state_dict by its immutable pool weights_path."""
        import torch
        from urllib.request import urlopen

        if key not in self._soup_cache:
            with urlopen(url, timeout=600) as resp:
                content = resp.read()
            self._soup_dl_bytes += len(content)
            self._soup_cache[key] = self._unwrap_sd(
                torch.load(BytesIO(content), map_location="cpu", weights_only=False)
            )
        return self._soup_cache[key]

    def _build_soup(self, members: List[Dict[str, str]]):
        """Uniform mean of member state_dicts; None when empty (= stock base)."""
        import pool

        if not members:
            return None
        sds = [self._fetch_member(m["key"], m["url"]) for m in members]
        return pool.uniform_soup(sds)

    async def score_soup(
        self, members: List[Dict[str, str]], images: List[np.ndarray],
        labels: List[np.ndarray], diam_mean: float = 30.0,
    ) -> float:
        """Mean instance F1 (IoU>=0.5) of the uniform soup of ``members`` on a split.
        ``members`` is a list of ``{"key": weights_path, "url": presigned_get}``;
        empty scores the stock cpsam base. Downloads happen off-lock; segmentation
        holds the shared GPU lock."""
        import pool

        soup = await asyncio.to_thread(self._build_soup, members)
        ckpt = None
        async with self._gpu_lock:
            if soup is not None:
                import torch

                ckpt = self._soup_dir() / f"soup-{uuid.uuid4().hex}.pt"
                await asyncio.to_thread(lambda: torch.save(soup, str(ckpt)))
            try:
                gk = {"diameter": diam_mean}
                masks = []
                for image in images:
                    masks.append(await asyncio.to_thread(
                        self._segment, image, gk, str(ckpt) if ckpt else None, "cpsam"))
            finally:
                if ckpt and ckpt.exists():
                    ckpt.unlink()
        return pool.mean_instance_f1(masks, [np.asarray(lbl) for lbl in labels])

    async def materialize_soup(self, members: List[Dict[str, str]], put_url: str) -> Dict[str, Any]:
        """Build the uniform soup of ``members``, PUT it to the pool's presigned URL,
        and return its sha256 + byte size — so the head weights never traverse the
        CPU entry. ``members`` must be non-empty (a published head always has one)."""
        import torch
        from urllib.request import Request, urlopen

        import pool

        soup = await asyncio.to_thread(self._build_soup, members)
        path = self._soup_dir() / f"head-{uuid.uuid4().hex}.pt"
        await asyncio.to_thread(lambda: torch.save(soup, str(path)))
        sha = await asyncio.to_thread(pool.sha256_file, str(path))
        n_bytes = path.stat().st_size

        def _put():
            data = path.read_bytes()
            with urlopen(Request(put_url, data=data, method="PUT"), timeout=600) as r:
                r.read()

        try:
            await asyncio.to_thread(_put)
        finally:
            path.unlink()
        return {"sha256": sha, "n_bytes": n_bytes}

    def pop_soup_dl_bytes(self) -> int:
        """Return + zero the bytes downloaded for soup members since the last call,
        so the entry can fold pool traffic into its durable transport counter."""
        b = self._soup_dl_bytes
        self._soup_dl_bytes = 0
        return b

    def _subprocess_env(self) -> Dict[str, str]:
        return {
            k: v for k, v in os.environ.items()
            if "TOKEN" not in k.upper() and "SECRET" not in k.upper()
        }

    def _run_train_subprocess(self, session_id: str):
        import training

        worker = str(Path(__file__).parent / "train_worker_cellpose.py")
        return training.run_cancellable_subprocess(
            [sys.executable, worker, session_id], session_id,
            cwd=str(Path(__file__).parent), env=self._subprocess_env(),
        )

    def _run_export_subprocess(self, session_id: str, export_dir: str):
        worker = str(Path(__file__).parent / "export_worker_cellpose.py")
        env = self._subprocess_env()
        env["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only export → deterministic + no GPU contention
        return subprocess.run(
            [sys.executable, worker, session_id, export_dir],
            cwd=str(Path(__file__).parent),
            capture_output=True, text=True, env=env,
        )

    async def export_bioimageio(self, session_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Build a draft cpsam BioImage.IO package (pytorch_state_dict + bundled
        CellposeSAMWrapper, self-tested with bioimageio.core.test_model) from a
        trained session in a CPU subprocess. Publishes nothing — returns the export
        result (package dir, zip path, file listing) for the entry to serve/upload."""
        import json

        import training

        export_dir = training.session_dir(session_id) / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "request.json").write_text(json.dumps(request))

        proc = await asyncio.to_thread(self._run_export_subprocess, session_id, str(export_dir))
        res_path = export_dir / "export_result.json"
        if proc.returncode != 0 or not res_path.exists():
            raise RuntimeError(
                f"Cellpose-SAM export failed (rc={proc.returncode}): {(proc.stderr or '')[-3000:]}"
            )
        return json.loads(res_path.read_text())

    async def train(self, session_id: str, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fine-tune cpsam in a subprocess under the shared GPU lock. Evicts the
        resident inference model first so the subprocess owns the GPU; the
        subprocess (``train_worker_cellpose.py``) writes COMPLETED/FAILED."""
        import training

        async with self._gpu_lock:
            await asyncio.to_thread(self._release_model)
            training.write_status(
                session_id, status="TRAINING", start_time=time.time(),
                n_epochs=params.get("n_epochs"), device=self._device(),
            )
            rc, tail, stopped = await asyncio.to_thread(self._run_train_subprocess, session_id)

        if stopped:
            training.write_status(
                session_id, status="STOPPED", message="stopped by user",
                end_time=time.time(), terminated_by="user_stop",
            )
        else:
            st = training.read_status(session_id)
            if st.get("status") not in ("COMPLETED", "FAILED", "STOPPED"):
                training.write_status(
                    session_id, status="FAILED",
                    message=f"training subprocess exited rc={rc}: {tail}",
                    end_time=time.time(), terminated_by="supervisor",
                )
        return {"session_id": session_id, "returncode": rc}
