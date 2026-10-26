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
work (serving + training), and a separate semaphore keeps the federated merge
single-flight because it is bounded by host RAM rather than by the GPU;
fine-tuning and export run in **subprocesses** so the OS reclaims their VRAM on
exit; export is CPU-only. Heavy deps (``torch``,
``cellpose``) are imported inside method bodies so the ``@bioengine.app``
decorator module stays introspectable with only ``bioengine[worker]`` + stdlib.
"""

import asyncio
import gc
import hashlib
import os
import subprocess
import sys
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import bioengine
import busy
import numpy as np

logger = bioengine.logger

# Match CellposeSAMWrapper's eval kwargs so served masks reproduce the exported
# package's self-test output; diameter is cpsam's mean-diameter prior.
_SERVE_DIAMETER = 30.0

# A merge weighs up to a full candidate batch of 1.1 GiB members and is
# all-or-nothing, so one truncated read loses every completed gate pass.
_MEMBER_DOWNLOAD_ATTEMPTS = 4
_MEMBER_DOWNLOAD_BACKOFF_S = 2.0

# Disk budget for cached soup members; see _evict_members. Override with
# BIOENGINE_SOUP_CACHE_MAX_BYTES.
_SOUP_CACHE_MAX_BYTES = 24 * 1024 ** 3

# What this deployment promises Ray. Named rather than inlined in the decorator so
# soup_memory_budget() reports the figure actually in force: aggregate() rejects a
# merge that would exceed it, and a second copy of the number would let the check
# pass while the scheduler used a different one.
_RUNTIME_MEMORY_MB = 12 * 1024


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
    memory_mb=_RUNTIME_MEMORY_MB,
    pip=_read_pip("requirements-runtime-cellpose.txt"),
    max_ongoing_requests=10,
    # One replica, not autoscaled. Serve chooses which replica to stop from node
    # occupancy and replica age only, never from work in progress, so a second replica
    # makes the one holding a live merge removable — and min_replicas bounds the count,
    # not the identity (#0088). A burst is absorbed by one replica regardless, since
    # max_ongoing_requests admits ten of them before an upscale can take any.
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 1,
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
        self._soup_dl_bytes = 0
        # The entry's admission check sizes ONE resident merge, but Serve's
        # max_ongoing_requests admits ten into this replica — so merges must be
        # single-flight here for that model to hold.
        self._soup_sem = asyncio.Semaphore(1)
        self._soup_idle_rss: Optional[int] = None
        # Both bounds, because a concurrent caller waits at both and which one it
        # is waiting at is the whole diagnosis.
        self._busy = busy.BusyLog(
            {
                "cpsam soup build (_soup_sem)": self._soup_sem.locked,
                "cpsam GPU work (_gpu_lock)": self._gpu_lock.locked,
            }
        )

    @bioengine.async_init
    async def _start_busy_log(self) -> None:
        self._busy.start()

    async def ping(self) -> Dict[str, Any]:
        """Internal liveness for the entry's readiness check (mirrors RuntimeApp)."""
        return {
            "status": "ok",
            "loaded_checkpoint": self._loaded_key,
            "gpu_busy": self._gpu_lock.locked(),
            "uptime": time.time() - self.start_time,
        }

    async def soup_memory_budget(self) -> Dict[str, Any]:
        """What this replica promised Ray, and what it is using — the two numbers
        aggregate()'s up-front admission check compares a merge against.

        ``rss_bytes`` is the footprint the merge will actually start from, so the
        check sizes it against the space left rather than against an empty replica;
        it is None where /proc is unavailable. While another merge holds
        ``_soup_sem`` the instantaneous RSS includes *that* merge, and the caller
        asking will run alone once it releases — so the last idle sample is
        reported instead, or the check would refuse a merge that fits. Ray kills on
        the NODE's total, so fitting the declaration is necessary and not
        sufficient — a co-tenant can still take the node over the threshold.
        """
        rss = None
        try:
            with open("/proc/self/statm") as fh:
                rss = int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            pass
        in_flight = self._soup_sem.locked()
        if rss is not None and not in_flight:
            self._soup_idle_rss = rss
        return {
            "declared_bytes": _RUNTIME_MEMORY_MB * 1024 ** 2,
            "rss_bytes": self._soup_idle_rss if in_flight else rss,
            "soup_in_flight": in_flight,
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
        post = generate_kwargs.get("postprocessing") or {}
        for _k in ("flow_threshold", "cellprob_threshold", "niter"):
            if post.get(_k) is not None:
                eval_kwargs[_k] = post[_k]
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

    def _evict_members(self, incoming_bytes: int = 0) -> None:
        """Keep the on-disk member cache under its byte cap, dropping LRU first.

        The cap exists because the cache is filled by candidates WEIGHED, not by
        members admitted: the best-first ordering pass scores every candidate in the
        batch, so a merge touches up to ``max_batch`` (default 32) members whatever
        it ends up admitting. Unbounded, that accumulates across merges for the
        replica's whole lifetime — the same shape as the RAM cache whose saturation
        at the batch bound OOM-killed this replica (two kills, on a 4-member and a
        16-member merge, with peaks within 0.8% of each other).

        Overflowing the cap costs a re-download, never a failure, so the default is
        a disk budget and NOT sized to a full batch. At the measured cpsam member
        size of 1.135 GiB, 24 GiB holds about 21 members; a full 32-candidate batch
        would want 36.3 GiB and will re-download its oldest members. That is the
        deliberate trade — the volume behind ``$HOME`` on the worker has not been
        measured, and an unbounded cache is what this is fixing. Raise it if
        re-downloads show up mid-merge and the volume has room.
        """
        import pool

        pool.evict_lru_to_cap(
            self._soup_dir(), "member-*.pt",
            cap_bytes=int(os.environ.get(
                "BIOENGINE_SOUP_CACHE_MAX_BYTES", _SOUP_CACHE_MAX_BYTES)),
            incoming_bytes=incoming_bytes,
        )

    @staticmethod
    def _replica_id() -> str:
        """Which replica is serving this call.

        A merge issues ~2n+3 gate passes as separate Serve calls, each routed
        independently, while the member cache is ordinary instance state — so the
        caller cannot account for a merge's transport, or notice a cold replica
        re-downloading the member set, without knowing who answered each pass.
        """
        from ray import serve

        return serve.get_replica_context().replica_id.unique_id

    @staticmethod
    def _unwrap_sd(obj):
        if isinstance(obj, dict) and "model_state_dict" in obj:
            return obj["model_state_dict"]
        if isinstance(obj, dict) and "model_state" in obj:
            return obj["model_state"]
        return obj

    @staticmethod
    def _download_member(url: str) -> bytes:
        """GET a member checkpoint, retrying a truncated or dropped read.

        The URL is a presigned GET for an immutable pool ``weights_path``, so a
        retry cannot return different bytes. ``read()`` raises ``IncompleteRead``
        on a response the server cut short whether or not a Content-Length was
        sent (measured: a truncated *chunked* body raises too), so the explicit
        length check below is a second line of defence, not the only one.
        An ``HTTPError`` is a verdict (403 on an expired presign, 404 on a wrong
        key) and will not heal, so it is not retried.
        """
        from http.client import HTTPException, IncompleteRead
        from urllib.error import HTTPError, URLError
        from urllib.request import urlopen

        for attempt in range(_MEMBER_DOWNLOAD_ATTEMPTS):
            try:
                with urlopen(url, timeout=600) as resp:
                    declared = resp.headers.get("Content-Length")
                    content = resp.read()
                if declared is not None and len(content) != int(declared):
                    raise IncompleteRead(content, int(declared) - len(content))
                return content
            except HTTPError:
                raise
            except (HTTPException, URLError, OSError) as e:
                if attempt == _MEMBER_DOWNLOAD_ATTEMPTS - 1:
                    raise
                logger.warning(
                    f"⚠️ Member download attempt {attempt + 1}/"
                    f"{_MEMBER_DOWNLOAD_ATTEMPTS} failed ({type(e).__name__}: {e}); "
                    f"retrying..."
                )
                time.sleep(_MEMBER_DOWNLOAD_BACKOFF_S * 2**attempt)

    def _fetch_member(self, key: str, url: str) -> Dict[str, Any]:
        """Load a member state_dict, downloading on first use.

        The download is cached as a file, not as tensors: one download per member
        per replica, but no member stays resident between uses. Keyed by the
        immutable pool weights_path (content-addressed → no staleness). Written
        via a temporary name so an interrupted download cannot leave a truncated
        file that every later aggregate would then fail to unpickle."""
        import torch

        cached = self._soup_dir() / f"member-{hashlib.sha256(key.encode()).hexdigest()}.pt"
        if cached.exists():
            os.utime(cached)  # mark as most-recently-used for _evict_members
        else:
            content = self._download_member(url)
            self._soup_dl_bytes += len(content)
            self._evict_members(incoming_bytes=len(content))
            partial_file = cached.with_suffix(".part")
            partial_file.write_bytes(content)
            partial_file.rename(cached)
        return self._unwrap_sd(
            torch.load(cached, map_location="cpu", weights_only=False)
        )

    def _build_soup(self, members: List[Dict[str, str]]):
        """Uniform mean of member state_dicts; None when empty (= stock base)."""
        import pool

        if not members:
            return None
        return pool.uniform_soup(
            [partial(self._fetch_member, m["key"], m["url"]) for m in members]
        )

    async def score_soup(
        self, members: List[Dict[str, str]], images: List[np.ndarray],
        labels: List[np.ndarray], diam_mean: float = 30.0,
    ) -> Dict[str, Any]:
        """Mean instance F1 (IoU>=0.5) of the uniform soup of ``members`` on a split,
        with the serving replica and the bytes this one call downloaded.
        ``members`` is a list of ``{"key": weights_path, "url": presigned_get}``;
        empty scores the stock cpsam base. Downloads happen off-lock; the soup build
        is single-flight per replica, and segmentation holds the shared GPU lock — so
        a concurrent caller waits twice, once per bound.

        ``soup_dl_bytes`` is this call's delta, not the replica's running total, so
        the caller can sum across passes without double-counting a warm replica or
        carrying a previous merge's bytes into this one."""
        import pool

        dl_before = self._soup_dl_bytes
        ckpt = None
        try:
            # ``_build_soup`` allocates the same soup the merge does, and ten of these
            # can be admitted at once, so it is bounded by the same semaphore. The
            # checkpoint is written here rather than under ``_gpu_lock`` because
            # ``_segment`` loads it from disk: once it is on disk the resident copy is
            # dead weight, and the lock wait below can be minutes.
            async with self._soup_sem:
                soup = await asyncio.to_thread(self._build_soup, members)
                if soup is not None:
                    import torch

                    ckpt = self._soup_dir() / f"soup-{uuid.uuid4().hex}.pt"
                    await asyncio.to_thread(lambda: torch.save(soup, str(ckpt)))
                soup = None

            async with self._gpu_lock:
                gk = {"diameter": diam_mean}
                masks = []
                for image in images:
                    masks.append(await asyncio.to_thread(
                        self._segment, image, gk, str(ckpt) if ckpt else None, "cpsam"))
        finally:
            if ckpt and ckpt.exists():
                ckpt.unlink()
        return {
            "f1": pool.mean_instance_f1(masks, [np.asarray(lbl) for lbl in labels]),
            "replica": self._replica_id(),
            "soup_dl_bytes": self._soup_dl_bytes - dl_before,
        }

    async def materialize_soup(self, members: List[Dict[str, str]], put_url: str) -> Dict[str, Any]:
        """Build the uniform soup of ``members``, PUT it to the pool's presigned URL,
        and return its sha256 + byte size — so the head weights never traverse the
        CPU entry. ``members`` must be non-empty (a published head always has one)."""
        import torch
        from urllib.request import Request, urlopen

        import pool

        dl_before = self._soup_dl_bytes
        # Returns from inside the semaphore so the soup and the PUT's own copy of it
        # are both released before the next merge starts allocating.
        async with self._soup_sem:
            soup = await asyncio.to_thread(self._build_soup, members)
            # Digest the tensors, not the saved file: the random stem below lands in
            # every torch zip record name, so a file digest would differ on every
            # merge of the same members. The stem stays random for #0083.
            sha = await asyncio.to_thread(pool.state_dict_digest, soup)
            path = self._soup_dir() / f"head-{uuid.uuid4().hex}.pt"
            await asyncio.to_thread(lambda: torch.save(soup, str(path)))
            n_bytes = path.stat().st_size

            def _put():
                data = path.read_bytes()
                with urlopen(Request(put_url, data=data, method="PUT"), timeout=600) as r:
                    r.read()

            try:
                await asyncio.to_thread(_put)
            finally:
                path.unlink()
            return {
                "sha256": sha, "n_bytes": n_bytes,
                "replica": self._replica_id(),
                "soup_dl_bytes": self._soup_dl_bytes - dl_before,
            }

    async def prepare_contribution(self, session_id: str, weights_put_url: str) -> Dict[str, Any]:
        """Inspect a finished session's local cpsam checkpoint (architecture
        signature + L2 drift from the base it started from), PUT its weights to the
        pool's presigned URL, and return the signature, drift, sha256s, byte size and
        cellpose version. The checkpoint is torch + a runtime-local file, neither of
        which the torch-free CPU entry can touch, so all of it happens here."""
        import torch
        from urllib.request import Request, urlopen

        import pool
        import training

        ckpt = str(training.checkpoint_path(session_id))
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"no servable checkpoint for session '{session_id}' at {ckpt}.")
        base_path = training.session_dir(session_id) / "init_checkpoint.pt"

        def _inspect() -> Dict[str, Any]:
            def _state(path):
                return self._unwrap_sd(torch.load(path, map_location="cpu", weights_only=False))

            sd = _state(ckpt)
            out: Dict[str, Any] = {
                "architecture_signature": pool.architecture_signature(sd),
                "l2_from_base": None,
                "base_checkpoint_sha256": None,
            }
            if base_path.exists():
                base_sd = _state(str(base_path))
                out["l2_from_base"] = pool.state_dict_l2_from_base(sd, base_sd)
                out["base_checkpoint_sha256"] = pool.sha256_file(str(base_path))
            return out

        info = await asyncio.to_thread(_inspect)
        weights_sha256 = await asyncio.to_thread(pool.sha256_file, ckpt)
        n_bytes = Path(ckpt).stat().st_size

        def _put():
            data = Path(ckpt).read_bytes()
            with urlopen(Request(weights_put_url, data=data, method="PUT"), timeout=600) as r:
                r.read()

        await asyncio.to_thread(_put)

        try:
            import importlib.metadata as _im

            cellpose_version = _im.version("cellpose")
        except Exception:
            cellpose_version = None

        return {
            **info,
            "weights_sha256": weights_sha256,
            "n_bytes": int(n_bytes),
            "cellpose_version": cellpose_version,
        }

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
