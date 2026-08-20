"""micro-sam GPU runtime — resident SAM encoder + shared GPU lock + training.

The GPU half of the two-deployment micro-sam app (the CPU ``EntryApp`` in
``entry.py`` owns transport/orchestration and composes this runtime). One
resident SAM encoder backs three serving consumers (Leg B) — embedding, AIS
masks, and the ONNX prompt decoder — and a single ``asyncio.Lock`` serialises
**every** GPU operation (serving *and* training), exactly like model-runner's
predict+test. Fine-tuning runs in a **subprocess** so the OS reclaims all
training VRAM on exit; the resident inference model is evicted first so the
subprocess gets the whole GPU.

Autoscaling (``max_replicas=3``, ``target=1``) lets a long training run hold one
GPU replica's lock while concurrent inference requests spin up and run on the
other GPU replicas — training and inference at the same time.

Import rule: heavy deps (``torch``, ``micro_sam``, ``segment_anything``) are
imported inside method bodies so the ``@bioengine.app`` decorator module stays
introspectable with only ``bioengine[worker]`` + the stdlib.
"""

import asyncio
import gc
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import bioengine
import numpy as np

logger = bioengine.logger

# The six micro-SAM generalists, each pinned to an exact bioimage.io record so a
# zoo-side update never changes served weights without an app release. Base SAM
# and fine-tuned checkpoints do not load through here.
ZOO_MODELS = {
    "vit_t_lm": ("faithful-chicken", "1.1"),
    "vit_b_lm": ("diplomatic-bug", "1.2"),
    "vit_l_lm": ("idealistic-rat", "1.2"),
    "vit_t_em_organelles": ("greedy-whale", "1"),
    "vit_b_em_organelles": ("noisy-ox", "1.2"),
    "vit_l_em_organelles": ("humorous-crab", "1.2"),
}
_ZOO_BASE_URL = "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io"


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
    pip=_read_pip("requirements-runtime.txt"),
    # Let up to 10 RPCs reach a replica so backlog behind ``_gpu_lock`` is
    # observable to the autoscaler; the lock serialises the actual GPU work.
    max_ongoing_requests=10,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 3,
        # target=1: a single concurrent op (a long train + one infer) is enough
        # to spin the 2nd GPU replica, so training and inference run at the same
        # time on the two GPUs instead of queueing on one lock.
        "target_num_ongoing_requests_per_replica": 1.0,
        "metrics_interval_s": 2.0,
        "look_back_period_s": 10.0,
        "downscale_delay_s": 600,
        "upscale_delay_s": 0.0,
    },
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    # A full training run holds the lock for minutes; give the replica time to
    # drain before it exits.
    graceful_shutdown_timeout_s=600.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class RuntimeApp:
    """GPU compute: resident encoder, shared GPU lock, subprocess fine-tuning."""

    def __init__(self) -> None:
        self.start_time = time.time()
        # Serialises ALL GPU work (serving + training) within a replica; the
        # queue this creates is what the autoscaler measures to add a 2nd replica.
        self._gpu_lock = asyncio.Lock()
        self._predictor = None
        self._segmenter = None
        self._loaded_model_type: Optional[str] = None
        self._loaded_key: Optional[tuple] = None
        self._pkg_model = None
        self._pkg_key: Optional[str] = None
        self._onnx_cache: Dict[str, bytes] = {}
        self._device_cached: Optional[str] = None

    async def ping(self) -> Dict[str, Any]:
        """Internal liveness for the entry's readiness check — not a Hypha method,
        but reachable via the composition handle. Unlocked so it answers during
        GPU work.
        """
        return {
            "status": "ok",
            "loaded_model_type": self._loaded_model_type,
            "gpu_busy": self._gpu_lock.locked(),
            "uptime": time.time() - self.start_time,
        }

    def _device(self) -> str:
        """Pick the compute device internally (device is not a public API param)."""
        if self._device_cached is None:
            try:
                import torch

                self._device_cached = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self._device_cached = "cpu"
        return self._device_cached

    # === resident model (one encoder + AIS segmenter at a time) ===

    @staticmethod
    def _to_image_format(array: np.ndarray) -> np.ndarray:
        """Coerce input to HxWx3 uint8 RGB, as the SAM image encoder expects."""
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        if array.ndim == 2:
            array = np.concatenate([array[..., None]] * 3, axis=-1)
        elif array.ndim == 3 and array.shape[0] == 3:
            array = np.transpose(array, [1, 2, 0])
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(
                f"Invalid input image of shape {array.shape}. Expected 2D (HxW) "
                "grayscale or 3-channel (HxWx3 / 3xHxW) RGB."
            )
        if array.dtype != np.uint8:
            array = array.astype("float32") - array.min(axis=(0, 1))
            denom = array.max(axis=(0, 1))
            denom[denom == 0] = 1.0
            array = (array / denom * 255).astype("uint8")
        return array

    @staticmethod
    def _nd(arr: np.ndarray) -> Dict[str, Any]:
        """Encode an array as hypha-rpc's ndarray wire-dict (numpy-neutral bytes).

        Survives the Ray hop to the CPU entry and its numpy-1.x proxy; the hypha
        client still decodes it to a real ndarray.
        """
        arr = np.ascontiguousarray(arr)
        return {
            "_rtype": "ndarray",
            "_rvalue": arr.tobytes(),
            "_rshape": list(arr.shape),
            "_rdtype": arr.dtype.name,
        }

    def _zoo_weights(self, model_type: str):
        """Fetch the pinned bioimage.io generalist (encoder + AIS decoder) into the
        local cache and return ``(encoder_path, decoder_path)``. Files are pinned by
        nickname+version so a zoo-side update never changes served weights without a
        release. Blocking — under the GPU lock; cached after first fetch."""
        import json

        import pooch

        nickname, version = ZOO_MODELS[model_type]
        arch = "vit_" + model_type.split("_")[1]
        cache = Path.home() / ".bioengine" / "micro_sam_zoo_cache" / model_type
        base = f"{_ZOO_BASE_URL}/{nickname}/{version}/files"
        enc = pooch.retrieve(f"{base}/{arch}.pt", known_hash=None, fname=f"{arch}.pt", path=cache)
        dec = pooch.retrieve(f"{base}/{arch}_decoder.pt", known_hash=None, fname=f"{arch}_decoder.pt", path=cache)
        src = cache / "source.json"
        if not src.exists():
            src.write_text(json.dumps({"id": nickname, "version": version, "model_type": model_type}))
        return enc, dec

    def _ensure_model(self, model_type: str, checkpoint: Optional[str] = None):
        """Load (predictor, AIS segmenter), reusing the resident pair when the
        (model_type, checkpoint) is unchanged. A generalist (no checkpoint) loads
        its pinned encoder + AIS decoder from the zoo cache; ``checkpoint`` serves a
        fine-tuned ``best.pt``. Blocking — call via ``asyncio.to_thread`` under the
        GPU lock.
        """
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter

        device = self._device()
        key = (model_type, checkpoint)
        if key != self._loaded_key:
            self._release_model()
            label = f"{model_type}" + (f" (finetuned: {checkpoint})" if checkpoint else "")
            logger.info(f"🔄 Loading μSAM model '{label}' on {device}...")
            if checkpoint is None and model_type in ZOO_MODELS:
                # get_predictor_and_segmenter has no decoder_path, so pre-load the
                # two pinned files via get_sam_model and hand it the assembled state.
                from micro_sam.util import get_sam_model

                enc, dec = self._zoo_weights(model_type)
                predictor, state = get_sam_model(
                    model_type=model_type, checkpoint_path=enc, decoder_path=dec,
                    device=device, return_state=True,
                )
                predictor, segmenter = get_predictor_and_segmenter(
                    model_type=model_type, predictor=predictor, state=state,
                    segmentation_mode=None,
                )
            else:
                predictor, segmenter = get_predictor_and_segmenter(
                    model_type=model_type,
                    checkpoint=checkpoint,
                    device=device,
                    # None → μSAM auto-selects AIS when the model ships a decoder.
                    segmentation_mode=None,
                )
            self._predictor = predictor
            self._segmenter = segmenter
            self._loaded_key = key
            self._loaded_model_type = model_type
            logger.info(f"✅ μSAM model '{label}' loaded.")
        return self._predictor, self._segmenter

    def _release_model(self) -> None:
        """Drop the resident model and free its VRAM (blocking; under gpu lock)."""
        self._predictor = None
        self._segmenter = None
        self._loaded_key = None
        self._pkg_model = None
        self._pkg_key = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _auto_segment(self, model_type, image, generate_kwargs, checkpoint):
        """μSAM automatic instance segmentation → int32 [H,W] mask. Blocking."""
        from micro_sam.automatic_segmentation import automatic_instance_segmentation

        predictor, segmenter = self._ensure_model(model_type, checkpoint)
        image = self._to_image_format(image)
        labels = automatic_instance_segmentation(
            predictor=predictor, segmenter=segmenter, input_path=image,
            ndim=2, verbose=False, **generate_kwargs,
        )
        return np.asarray(labels).astype(np.int32)

    def _encode(self, model_type, image, checkpoint):
        """SAM image encoder → embedding payload for onnxruntime-web. Blocking."""
        predictor, _ = self._ensure_model(model_type, checkpoint)
        image = self._to_image_format(image)
        original_image_shape = image.shape[:2]
        sam = predictor.model
        sam_scale = sam.image_encoder.img_size / max(original_image_shape)
        predictor.reset_image()
        predictor.set_image(image)
        features = predictor.get_image_embedding().cpu().numpy()
        input_size = [int(x) for x in predictor.input_size]
        predictor.reset_image()
        return {
            "features": features.astype(np.float32),
            "original_image_shape": [int(x) for x in original_image_shape],
            "input_size": input_size,
            "sam_scale": float(sam_scale),
            "mask_threshold": float(sam.mask_threshold),
        }

    def _auto_segment_from_embedding(self, model_type, features, input_size, original_size,
                                     generate_kwargs, checkpoint):
        """Run the AIS decoder on a **precomputed** SAM embedding (no encoder pass).
        Blocking — call under the GPU lock. For the AIS generalist models; matches
        the image path, where ``generate`` returns the instance-label array directly.
        """
        _, segmenter = self._ensure_model(model_type, checkpoint)
        image_embeddings = {
            "features": np.asarray(features).astype(np.float32),
            "input_size": tuple(int(x) for x in input_size),
            "original_size": tuple(int(x) for x in original_size),
        }
        # image is unused when image_embeddings is provided (initialize only
        # encodes when embeddings are None).
        segmenter.initialize(image=None, image_embeddings=image_embeddings)
        instances = segmenter.generate(**generate_kwargs)
        return np.asarray(instances).astype(np.int32)

    def _load_package_model(self, weights_path: str, model_type: str, use_conv_transpose: bool):
        """Load an exported AIS package's ``MicroSAMAIS`` weights as the resident
        model, reusing it when ``weights_path`` is unchanged. Blocking — under lock."""
        import torch

        sys.path.insert(0, str(Path(__file__).parent))
        from ais_adapter import MicroSAMAIS

        key = f"pkg:{weights_path}"
        if key != self._pkg_key:
            self._release_model()
            mod = MicroSAMAIS(model_type=model_type, use_conv_transpose=use_conv_transpose)
            mod.load_state_dict(torch.load(weights_path, map_location="cpu"))
            mod.eval().to(self._device())
            self._pkg_model = mod
            self._pkg_key = key
            self._loaded_model_type = f"package:{model_type}"
            logger.info(f"✅ Loaded exported AIS package model ({model_type}).")
        return self._pkg_model

    def _watershed_from_maps(self, maps: np.ndarray, generate_kwargs: Dict[str, Any]) -> np.ndarray:
        """micro-SAM native post-processing on the three AIS maps → int32 instance
        labels (channel 0 foreground, 1 center distance, 2 boundary distance).
        Injects the maps into micro_sam's own ``InstanceSegmentationWithDecoder``
        and calls ``generate`` so the smoothing + watershed exactly match the
        built-in AIS path (and track whatever smoothing backend the installed
        micro_sam ships — it no longer hard-depends on vigra)."""
        from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder

        allowed = (
            "center_distance_threshold", "boundary_distance_threshold",
            "foreground_threshold", "foreground_smoothing", "distance_smoothing", "min_size",
        )
        kw = {k: v for k, v in (generate_kwargs or {}).items() if k in allowed}
        seg = InstanceSegmentationWithDecoder.__new__(InstanceSegmentationWithDecoder)
        seg._foreground = maps[0]
        seg._center_distances = maps[1]
        seg._boundary_distances = maps[2]
        seg._is_initialized = True
        instances = seg.generate(output_mode="instance_segmentation", **kw)
        return np.asarray(instances).astype(np.int32)

    def _auto_segment_package(self, weights_path, model_type, use_conv_transpose, image, generate_kwargs):
        """Run an exported AIS package on one image → int32 instance mask. Blocking."""
        import torch

        mod = self._load_package_model(weights_path, model_type, use_conv_transpose)
        img = self._to_image_format(image)
        inp = torch.from_numpy(img.astype("float32")).permute(2, 0, 1)[None].contiguous().to(self._device())
        with torch.no_grad():
            maps = mod(inp)[0].cpu().numpy()
        return self._watershed_from_maps(maps, generate_kwargs)

    def _export_onnx(self, model_type: str, quantize: bool) -> bytes:
        """Export the lightweight SAM prompt decoder to ONNX bytes. Blocking."""
        import tempfile
        import warnings

        import torch
        from onnxruntime import InferenceSession
        from onnxruntime.quantization import QuantType
        from onnxruntime.quantization.quantize import quantize_dynamic
        from segment_anything.utils.onnx import SamOnnxModel

        device = self._device()
        predictor, _ = self._ensure_model(model_type)
        sam = predictor.model
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / f"{model_type}.onnx"
            sam.to("cpu")
            try:
                onnx_model = SamOnnxModel(
                    model=sam, return_single_mask=True,
                    use_stability_score=False, return_extra_metrics=False,
                )
                embed_dim = sam.prompt_encoder.embed_dim
                embed_size = sam.prompt_encoder.image_embedding_size
                mask_input_size = [4 * x for x in embed_size]
                dummy_inputs = {
                    "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
                    "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
                    "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
                    "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
                    "has_mask_input": torch.tensor([1], dtype=torch.float),
                    "orig_im_size": torch.tensor([1024, 1024], dtype=torch.float),
                }
                dynamic_axes = {
                    "point_coords": {1: "num_points"},
                    "point_labels": {1: "num_points"},
                }
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    torch.onnx.export(
                        onnx_model, tuple(dummy_inputs.values()), str(onnx_path),
                        export_params=True, verbose=False, opset_version=17,
                        do_constant_folding=True, input_names=list(dummy_inputs.keys()),
                        output_names=["masks", "iou_predictions", "low_res_masks"],
                        dynamic_axes=dynamic_axes,
                    )
                ort_inputs = {k: v.cpu().numpy() for k, v in dummy_inputs.items()}
                InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(None, ort_inputs)
                if quantize:
                    q_path = onnx_path.with_stem(onnx_path.stem + "_quantized")
                    quantize_dynamic(
                        model_input=str(onnx_path), model_output=str(q_path),
                        per_channel=False, reduce_range=False, weight_type=QuantType.QUInt8,
                    )
                    onnx_path = q_path
                return onnx_path.read_bytes()
            finally:
                sam.to(device)

    # === composition endpoints (called by EntryApp via the runtime handle) ===

    async def auto_segment(
        self, images: List[np.ndarray], model_type: str = "vit_l_lm",
        generate_kwargs: Optional[Dict[str, Any]] = None,
        checkpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """AIS masks for a batch of (already-resolved) images → wire-dict list."""
        generate_kwargs = generate_kwargs or {}
        results: List[Dict[str, Any]] = []
        async with self._gpu_lock:
            for image in images:
                labels = await asyncio.to_thread(
                    self._auto_segment, model_type, image, generate_kwargs, checkpoint
                )
                results.append({"output": self._nd(labels)})
        return results

    async def auto_segment_from_embedding(
        self, embeddings: List[Dict[str, Any]], model_type: str = "vit_l_lm",
        generate_kwargs: Optional[Dict[str, Any]] = None, checkpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """AIS masks for a batch of precomputed embeddings (each a dict with
        ``features``/``input_size``/``original_size``) → wire-dict list.
        """
        generate_kwargs = generate_kwargs or {}
        results: List[Dict[str, Any]] = []
        async with self._gpu_lock:
            for emb in embeddings:
                labels = await asyncio.to_thread(
                    self._auto_segment_from_embedding, model_type,
                    emb["features"], emb["input_size"], emb["original_size"],
                    generate_kwargs, checkpoint,
                )
                results.append({"output": self._nd(labels)})
        return results

    async def auto_segment_package(
        self, images: List[np.ndarray], weights_path: str, model_type: str,
        use_conv_transpose: bool, generate_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """AIS masks from an exported BioImage.IO package (``MicroSAMAIS`` weights +
        micro-SAM native watershed) for a batch of images → wire-dict list."""
        generate_kwargs = generate_kwargs or {}
        results: List[Dict[str, Any]] = []
        async with self._gpu_lock:
            for image in images:
                labels = await asyncio.to_thread(
                    self._auto_segment_package, weights_path, model_type,
                    use_conv_transpose, image, generate_kwargs,
                )
                results.append({"output": self._nd(labels)})
        return results

    async def encode(
        self, image: np.ndarray, model_type: str = "vit_l_lm",
        checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Encoder embedding. ``features`` comes back as a wire-dict (the entry
        reconstructs/forwards it); ``input_size``/``original_image_shape``/
        ``sam_scale``/``mask_threshold`` describe it for the ONNX decoder and for
        running AIS on it later via ``infer(embeddings=...)``.
        """
        async with self._gpu_lock:
            payload = await asyncio.to_thread(self._encode, model_type, image, checkpoint)
        payload["features"] = self._nd(payload["features"])
        return payload

    async def export_onnx(
        self, model_type: str = "vit_l_lm", quantize: bool = True,
    ) -> bytes:
        cache_key = f"{model_type}:{quantize}"
        if cache_key in self._onnx_cache:
            return self._onnx_cache[cache_key]
        logger.info(f"📦 Exporting ONNX prompt decoder for '{model_type}'...")
        async with self._gpu_lock:
            onnx_bytes = await asyncio.to_thread(self._export_onnx, model_type, quantize)
        self._onnx_cache[cache_key] = onnx_bytes
        return onnx_bytes

    def _subprocess_env(self) -> Dict[str, str]:
        """Inherit the replica env (incl. Ray's CUDA_VISIBLE_DEVICES) minus secrets."""
        return {
            k: v for k, v in os.environ.items()
            if "TOKEN" not in k.upper() and "SECRET" not in k.upper()
        }

    def _run_train_subprocess(self, session_id: str):
        worker = str(Path(__file__).parent / "train_worker.py")
        return subprocess.run(
            [sys.executable, worker, session_id],
            cwd=str(Path(__file__).parent),
            capture_output=True, text=True, env=self._subprocess_env(),
        )

    def _run_export_subprocess(self, session_id: str, export_dir: str):
        worker = str(Path(__file__).parent / "export_worker.py")
        env = self._subprocess_env()
        env["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only export → no GPU contention
        return subprocess.run(
            [sys.executable, worker, session_id, export_dir],
            cwd=str(Path(__file__).parent),
            capture_output=True, text=True, env=env,
        )

    async def export_bioimageio(self, session_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Build a prompt-free AIS BioImage.IO package from a trained session in a
        CPU subprocess (no GPU lock). ``request`` carries the RDF metadata
        (name/description/authors/license/provenance). Returns the export result
        (package dir, zip path, file listing) for the entry to serve/upload.
        """
        import json

        import training

        export_dir = training.session_dir(session_id) / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "request.json").write_text(json.dumps(request))

        proc = await asyncio.to_thread(self._run_export_subprocess, session_id, str(export_dir))
        res_path = export_dir / "export_result.json"
        if proc.returncode != 0 or not res_path.exists():
            raise RuntimeError(
                f"BioImage.IO export failed (rc={proc.returncode}): {(proc.stderr or '')[-3000:]}"
            )
        return json.loads(res_path.read_text())

    async def train(self, session_id: str, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fine-tune in a subprocess under the shared GPU lock (long-running).

        Evicts the resident inference model first so the subprocess owns the GPU;
        the subprocess (``train_worker.py``) writes COMPLETED/FAILED to the
        session status. Held for the run's duration → the ongoing-request count
        drives autoscaling so concurrent inference lands on a second replica.
        """
        import training

        async with self._gpu_lock:
            await asyncio.to_thread(self._release_model)
            training.write_status(
                session_id, status="TRAINING", start_time=time.time(),
                n_epochs=params.get("n_epochs"), device=self._device(),
            )
            proc = await asyncio.to_thread(self._run_train_subprocess, session_id)

        # Fallback: if the child died without writing a terminal status, record it.
        st = training.read_status(session_id)
        if st.get("status") not in ("COMPLETED", "FAILED", "STOPPED"):
            tail = (proc.stderr or "")[-800:]
            training.write_status(
                session_id, status="FAILED",
                message=f"training subprocess exited rc={proc.returncode}: {tail}",
                end_time=time.time(),
            )
        return {"session_id": session_id, "returncode": proc.returncode}
