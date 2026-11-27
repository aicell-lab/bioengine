"""micro-sam CPU entry — transport, routing, and fine-tuning orchestration.

The always-on CPU half of the two-deployment micro-sam app. It owns all Hypha/S3
transport and session orchestration, and composes the GPU ``RuntimeApp`` (in
``runtime.py``) by type hint — forwarding serving calls to it and firing a
long-running training call that drives the runtime's autoscaling so training and
inference can run on separate GPU replicas at the same time.

Only this entry is named in ``manifest.yaml``'s ``entry:``. Serving methods
return the runtime's ndarray wire-dicts unchanged (numpy-neutral across the Ray
hop and this entry's numpy-1.x proxy); the hypha client decodes them to real
ndarrays.
"""

import asyncio
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

import bioengine
import busy
import httpx

import ngff_region
import numpy as np
import yaml
from hypha_rpc import connect_to_server
from pydantic import Field


def _parse_backends(raw: Optional[str]) -> set:
    """Backends this worker deploys, from ``MODEL_FINETUNE_BACKENDS`` (comma list
    over ``microsam``/``cellpose``); unset or empty → both. A disabled backend's
    runtime class is left ``None`` so its ``EntryApp.__init__`` annotation resolves
    to ``None`` at introspection — the composition graph drops it, so it reserves
    no GPU and never builds its pip env."""
    if not raw or not raw.strip():
        return {"microsam", "cellpose"}
    picked = {b.strip().lower() for b in raw.split(",") if b.strip()}
    unknown = picked - {"microsam", "cellpose"}
    if unknown:
        raise ValueError(
            f"MODEL_FINETUNE_BACKENDS: unknown backend(s) {sorted(unknown)}; "
            "valid values are 'microsam' and 'cellpose'."
        )
    return picked


_ENABLED_BACKENDS = _parse_backends(os.getenv("MODEL_FINETUNE_BACKENDS"))
# How many fine-tunes may run at once. Default 1 keeps the historical behaviour
# (serialize training so autoscaled replicas stay free for inference). Set >1
# only when there is a distinct GPU per concurrent run (one runtime replica per
# GPU); Ray Serve then spreads the concurrent train() calls across replicas.
_MAX_CONCURRENT_TRAINING = max(
    1, int(os.getenv("MODEL_FINETUNE_MAX_CONCURRENT_TRAINING", "1") or "1")
)
RuntimeApp = None
CellposeRuntime = None
if "microsam" in _ENABLED_BACKENDS:
    from runtime import RuntimeApp
if "cellpose" in _ENABLED_BACKENDS:
    from runtime_cellpose import CellposeRuntime

logger = bioengine.logger

SERVER_URL = "https://hypha.aicell.io"
# Ceiling on the liveness call in `_health`. Must stay well under the
# deployment's `health_check_timeout_s=30.0`, so a hung Hypha connection is reported
# as UNHEALTHY by this check rather than timing the check itself out. Matches
# annotation-broker's `_HEALTH_CALL_TIMEOUT_S`, which guards the same shape.
_HEALTH_CALL_TIMEOUT_S = 15.0

# How long `get_training_capabilities` waits for a backend's store verdict, and
# how long it then stops asking a backend that did not answer.
#
# BOTH NUMBERS EXIST BECAUSE THIS METHOD IS POLLED BY THE ANNOTATE PAGE, and a
# review measured what the first version cost: with VRAM declared, both backends
# enabled and one runtime whose `ping` hangs, the call took 10.01 s and returned
# no verdict anyway. Two faults, fixed separately. The views are now gathered
# concurrently, so the worst case is ONE timeout rather than one per backend;
# and the timeout is 2 s, matching the readiness probe, because `ping` reports a
# cached verdict and does no I/O, so a runtime that is up answers in
# milliseconds and 5 s only lengthened the wait for one that is not.
#
# The negative window is the other half. A cluster with fewer GPUs than enabled
# backends leaves a backend UPSCALING indefinitely -- that is the permanent
# state this method's own docstring describes, not a blip -- so retrying it on
# every poll would pay the timeout forever. 30 s is short enough that a runtime
# coming up is reflected within one or two polls.
_STORE_VIEW_PING_TIMEOUT_S = 2.0
_STORE_VIEW_UNREACHABLE_FOR_S = 30.0
SUPPORTED_FILE_TYPES = Literal[".npy", ".png", ".tiff", ".tif", ".jpeg", ".jpg"]
UPLOAD_FILE_TYPES = Literal[".npy", ".png", ".tiff", ".tif", ".jpeg", ".jpg", ".npz"]
ModelType = Literal[
    "vit_l_lm",
    "vit_b_lm",
    "vit_t_lm",
    "vit_l_em_organelles",
    "vit_b_em_organelles",
    "vit_t_em_organelles",
    "vit_b",
    "vit_l",
    "vit_h",
    "cpsam",
    "cpdino",
    "cpdino-vitb",
]
# Model types served by the isolated Cellpose runtime rather than micro-sam.
CELLPOSE_MODEL_TYPES = ("cpsam", "cpdino", "cpdino-vitb")

# Backends that cost a GPU. Each enabled one composes a runtime deployment
# declaring gpu_memory_mb=-1, which the worker resolves to a WHOLE device
# (bioengine/apps/builder.py), so the app's GPU floor is simply how many of
# these are on. Named here because _gpu_placement_report has to count them.
GPU_BACKENDS = ("microsam", "cellpose")

# How long the whole base-checkpoint preflight may take, across all of a model
# type's parts. Bounded because it sits on the admission path: a preflight that
# could hang would be a worse version of the late failure it prevents.
BASE_CHECKPOINT_PREFLIGHT_TIMEOUT_S = 12.0

# The fallback's own budget, and it is a DOWNLOAD budget rather than a probe
# one. The largest mirrored encoder is ~375 MB; the probe's 12 s is sized for a
# HEAD, so reusing it would turn every fallback into a timeout and the feature
# would look broken rather than slow. Reached only when the probe already
# refused, so a site that can reach upstream never waits on this.
BASE_CHECKPOINT_MIRROR_TIMEOUT_S = 900.0


def _gpu_placement_report(enabled: Iterable[str]) -> Dict[str, Any]:
    """Whether this cluster can place the GPU runtimes the enabled backends need.

    Exists because getting this wrong is otherwise INVISIBLE from the running
    system. Two enabled backends book two whole GPUs even to serve one
    request; on a cluster whose second GPU is never free the second replica simply
    never schedules. Measured 2026-11-13 on a one-GPU ``external-cluster`` worker: the
    unplaceable runtime stays ``UPDATING`` with its replica ``STARTING``, the
    application stays ``DEPLOYING``, and neither ever times out — so
    ``deploy_app`` returns successfully, ``get_app_status`` reads like a slow
    first start forever, and nothing names the mismatch or mentions
    ``MODEL_FINETUNE_BACKENDS``. The worker's own resource check does say it, but
    only in ``external-cluster`` mode does the deploy get that far, and it says it
    in the WORKER's log — not in anything the app's caller can read.

    Answered from the CPU entry on purpose: the entry deployment asks for no GPU,
    so it is the one part of the app that is still alive to say why the rest is not.

    TWO different numbers, because the two questions they answer have different
    remedies and only one of them is a configuration mistake:

    ``gpus_in_cluster``  ``ray.cluster_resources()`` — the TOTAL the cluster owns,
                         free or not. Whether this app can EVER place its runtimes
                         here is a fact about the total, so ``sufficient`` is keyed
                         to it and to nothing else.
    ``gpus_free``        ``ray.available_resources()`` — what is unbooked at the
                         instant of the call. Reported in ``detail`` and
                         deliberately NOT escalated. A momentary zero is normal:
                         the app's own placed replicas hold whole devices, so once
                         this app is running its free count is legitimately short
                         by exactly what it booked, and on an elastic cluster a
                         neighbour's job finishing changes the answer a second
                         later. Keying ``sufficient`` to it would make this
                         diagnostic cry wolf at the app itself.

    So a cluster that owns enough but has none free right now reads
    ``sufficient: True`` with a ``detail`` that says the runtimes pend until a
    device is released — which is the whole of the answer an operator needs, with
    no alarm attached.

    Never raises and never blocks the deploy. ``sufficient`` is deliberately
    three-valued, because "not enough" and "cannot tell" are different answers:

    ``True``   the cluster advertises at least ``gpus_required`` in TOTAL;
    ``False``  it advertises fewer — the deployment will pend forever, and
               ``advice`` says what to set;
    ``None``   unknown. Either no GPU node has reported in yet, which is the
               NORMAL state on an elastic SLURM cluster before the autoscaler
               provides one, or Ray's resource view could not be read at all.
               Not an error either way: reporting a shortfall that is merely a
               cold autoscaler would be a false alarm, which is the failure mode
               this whole report exists to avoid repeating in the other direction.
    """
    enabled = set(enabled)
    required = len([b for b in GPU_BACKENDS if b in enabled])
    report: Dict[str, Any] = {
        "backends_enabled": sorted(enabled),
        "gpus_required": required,
        "gpus_in_cluster": None,
        "gpus_free": None,
        "sufficient": None,
        "detail": "",
        "advice": "",
    }
    try:
        import ray

        total = float(ray.cluster_resources().get("GPU", 0.0))
        free = float(ray.available_resources().get("GPU", 0.0))
    except Exception as e:  # pragma: no cover - needs a Ray-less replica
        report["detail"] = (
            f"could not read the cluster's GPU capacity ({type(e).__name__}: {e})"
        )
        return report

    report["gpus_in_cluster"] = total
    report["gpus_free"] = free
    if required == 0:
        report["sufficient"] = True
        report["detail"] = "no GPU backend is enabled, so no GPU is required"
        return report
    if total >= required:
        report["sufficient"] = True
        if free < required:
            # Reported, never escalated: `sufficient` stays True and `advice` stays
            # empty. See the two-numbers block above — the alternative warns at the
            # app's own replicas and at every elastic cluster between jobs.
            report["detail"] = (
                f"{total:g} GPU(s) visible but {free:g} free right now; {required} "
                f"required by {', '.join(sorted(enabled))} — the runtime(s) that "
                "cannot be placed pend until a device is released. Not a "
                "configuration problem: this cluster owns enough. If this app is "
                "already running, its own GPU replicas are part of what is booked."
            )
        else:
            report["detail"] = (
                f"{total:g} GPU(s) visible, {free:g} free, {required} required by "
                f"{', '.join(sorted(enabled))}"
            )
        return report
    if total == 0:
        report["detail"] = (
            "no GPU has reported into the cluster yet, so whether the "
            f"{required} required by {', '.join(sorted(enabled))} will be "
            "placeable cannot be told from here. On an elastic (SLURM) cluster "
            "this is the normal state until the autoscaler provides a GPU node."
        )
        return report

    report["sufficient"] = False
    report["detail"] = (
        f"this cluster advertises {total:g} GPU(s) but the enabled backends "
        f"({', '.join(sorted(enabled))}) need {required} — each GPU runtime books "
        "a WHOLE device. The runtime(s) that cannot be placed stay UPDATING with a "
        "replica STARTING and the application stays DEPLOYING, indefinitely: none "
        "of that is a failure state and none of it times out, so the app looks "
        "like it is still starting rather than stuck. Its Hypha service is never "
        "registered while a sibling pends, so get_app_status is the only surface "
        "that can answer — which is where you are reading this."
    )
    report["advice"] = (
        "Deploy one backend: pass "
        'application_env_vars={"*": {"MODEL_FINETUNE_BACKENDS": "cellpose"}} '
        '(or "microsam") to deploy_app, or launch the worker from an image that '
        "bakes it. Each single backend has a 1-GPU floor. See the app README, "
        "'Selecting backends'."
    )
    return report


# Which backends each start_training parameter actually affects. The training UI
# reads this (via get_training_capabilities) to show a control only for the
# selected model's backend. Parameters common to both are listed under both.
TRAINING_PARAM_BACKENDS = {
    "n_epochs": ["microsam", "cellpose"],
    "batch_size": ["microsam", "cellpose"],
    "learning_rate": ["microsam", "cellpose"],
    "val_fraction": ["microsam", "cellpose"],
    "n_samples": ["microsam", "cellpose"],
    "n_objects_per_batch": ["microsam"],
    "patch_size": ["microsam"],
    "diam_mean": ["cellpose"],
    "train_tile_size": ["cellpose"],
}

# cellpose locks the training tile to its backbone: cpsam raises
# ValueError for any bsize != 256 (cellpose/train.py), while cpdino is free and
# defaults to 384. Keyed on the model_type the caller passes, not the backbone.
CELLPOSE_FIXED_TILE_SIZE = {"cpsam": 256}


def _read_pip(name: str) -> List[str]:
    text = (Path(__file__).parent / name).read_text()
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _apply_clahe(img_array: np.ndarray) -> np.ndarray:
    """Grayscale uint8 + CLAHE contrast enhancement. Accepts (H,W), (H,W,C) or
    (C,H,W); returns a (H,W) uint8 array. Rescues low-contrast fluorescence that
    Cellpose-SAM otherwise cannot segment (opt-in via preflight_training_dataset).
    Pure numpy — NO cv2 — so the CPU entry's numpy ABI is never perturbed: adding
    opencv dragged in an incompatible numpy and crashed EntryApp init at deploy
    (numpy.dtype size mismatch vs the pandas imported at init; deNBI verify of 0.21.0).
    """
    arr = np.asarray(img_array)
    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[0] < arr.shape[1]:
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[2] >= 3:
            arr = (
                0.299 * arr[:, :, 0].astype(np.float32)
                + 0.587 * arr[:, :, 1].astype(np.float32)
                + 0.114 * arr[:, :, 2].astype(np.float32)
            ).astype(np.uint8)
    if arr.dtype != np.uint8:
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr = ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            arr = np.zeros(arr.shape, dtype=np.uint8)
    return _clahe_numpy(arr, clip_limit=3.0, grid=(16, 16))


def _clahe_numpy(
    gray: np.ndarray, clip_limit: float = 3.0, grid=(16, 16)
) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation, bilinearly interpolated
    between per-tile clipped-CDF LUTs — a numpy reimplementation of
    ``cv2.createCLAHE(clipLimit, tileGridSize).apply()`` so no compiled dep is pulled
    into the CPU entry. gray: (H,W) uint8 -> (H,W) uint8."""
    g = np.asarray(gray, dtype=np.uint8)
    h, w = g.shape
    gy, gx = grid
    ys = np.linspace(0, h, gy + 1).astype(int)
    xs = np.linspace(0, w, gx + 1).astype(int)
    luts = np.empty((gy, gx, 256), dtype=np.float32)
    for i in range(gy):
        for j in range(gx):
            tile = g[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            n = tile.size
            if n == 0:
                luts[i, j] = np.arange(256, dtype=np.float32)
                continue
            hist = np.bincount(tile.ravel(), minlength=256).astype(np.float32)
            limit = max(1.0, clip_limit * n / 256.0)
            excess = np.maximum(hist - limit, 0.0).sum()
            hist = np.minimum(hist, limit) + excess / 256.0
            cdf = np.cumsum(hist)
            lo, hi = cdf.min(), cdf.max()
            luts[i, j] = (cdf - lo) / (hi - lo + 1e-9) * 255.0
    cy = (ys[:-1] + ys[1:]) / 2.0
    cx = (xs[:-1] + xs[1:]) / 2.0

    def _iw(coords, centers):
        idx = np.searchsorted(centers, coords)
        i1 = np.clip(idx, 0, len(centers) - 1)
        i0 = np.clip(idx - 1, 0, len(centers) - 1)
        c0, c1 = centers[i0], centers[i1]
        wt = np.where(
            i0 == i1,
            0.0,
            np.clip((coords - c0) / np.where(c1 > c0, c1 - c0, 1.0), 0, 1),
        )
        return i0, i1, wt

    ri0, ri1, rw = _iw(np.arange(h), cy)
    ci0, ci1, cw = _iw(np.arange(w), cx)
    R0 = np.repeat(ri0[:, None], w, 1)
    R1 = np.repeat(ri1[:, None], w, 1)
    C0 = np.repeat(ci0[None, :], h, 0)
    C1 = np.repeat(ci1[None, :], h, 0)
    gi = g.astype(np.int64)
    v00 = luts[R0, C0, gi]
    v01 = luts[R0, C1, gi]
    v10 = luts[R1, C0, gi]
    v11 = luts[R1, C1, gi]
    RW = rw[:, None]
    CW = cw[None, :]
    out = (v00 * (1 - CW) + v01 * CW) * (1 - RW) + (v10 * (1 - CW) + v11 * CW) * RW
    return np.clip(out, 0, 255).astype(np.uint8)


def _is_missing_file(exc: BaseException) -> bool:
    """True only when an artifact-manager call failed because the file is absent.

    hypha-rpc flattens a server-side exception into a RemoteException carrying the
    remote traceback as text, so the original class survives only on its last line.
    Hypha raises FileNotFoundError for an absent file and KeyError for an artifact
    that is missing or unreachable — and both messages contain "does not exist", so
    the class name is the only thing that separates absence from a transport,
    permission or token failure that must never be read as absence.
    """
    if isinstance(exc, FileNotFoundError):
        return True
    lines = [line for line in str(exc).splitlines() if line.strip()]
    return bool(lines) and lines[-1].startswith("FileNotFoundError:")


def _write_file_atomic(path: Path, content: bytes) -> None:
    """Write ``content`` to ``path`` so no reader can observe a partial file.

    The HOME caches are keyed on the request (pool, version, model id) and HOME is
    shared by every call the worker serves, so two callers asking for the same thing
    write the same path. ``write_bytes`` opens ``'wb'``, which truncates first — a
    reader of a 1.2 GB checkpoint then sees a prefix, and identical content is no
    defence because the problem is the length, not the bytes. The temp name carries
    a uuid so concurrent writers cannot share it either.

    Both fsyncs are for the cross-node case rather than the same-process one: callers
    hand ``path`` to a runtime replica on a different node, and ``os.replace`` publishes
    the name immediately while the data may still sit in this client's page cache — so
    the reader opens the new name and legitimately sees fewer bytes than were written.
    Syncing the file before the rename and the directory after closes that window.
    """
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        tmp.unlink(missing_ok=True)


class _RecognisedRedeposit(Exception):
    """Internal control flow: the weights just uploaded are already in the pool.

    Raised from *inside* ``contribute``'s ``async with self._staged(...)`` block
    and caught immediately outside it, so the duplicate's already-minted
    ``put_file`` manifest row unwinds down ``_staged``'s discard path — the one
    that is already proven and already tested. Returning normally from inside
    that block instead would leave the stage open holding a row whose bytes
    never landed, and ``_staged``'s own docstring says what follows: every
    later ``commit`` raises ``File '<path>' does not exist in the artifact``,
    which ``_commit_with_retry`` misreads as the read-after-write 404 and
    retries for ~62 s before re-raising. One retried deposit would then take
    the whole pool down rather than just its own merge — strictly worse than
    the double-weighting this is fixing. Never crosses the RPC boundary: the
    caller sees a normal return with ``deduplicated: true``.
    """

    def __init__(self, existing: Dict[str, Any]) -> None:
        super().__init__(
            f"weights already deposited as '{existing.get('contribution_id')}'"
        )
        self.existing = existing


@bioengine.app(
    num_cpus=1,
    memory_mb=4 * 1024,
    pip=_read_pip("requirements-entry.txt"),
    max_ongoing_requests=10,
    # Scales to zero so an idle deployment holds nothing on an HPC login node;
    # the proxy stays up and serves the service either way. Held at one replica
    # because the training/export task maps and the GPU semaphore are
    # replica-local; the records a caller polls are on disk, so a teardown
    # between calls is invisible to the frontend.
    #
    # initial_replicas is 1, not 0, so the entry can SPEAK at deploy time.
    # Measured 2026-11-13 on a one-GPU external cluster with both backends on: a
    # GPU runtime that cannot be placed leaves the app in DEPLOYING forever, the
    # proxy registers its Hypha service only once every sibling is serviceable, so
    # nothing can call the app — and with initial_replicas 0 nothing wakes the entry
    # either, so the placement warning below is never written. One CPU-only replica
    # at deploy time puts it in deployments.<entry>.logs, which get_app_status
    # returns while the app is still DEPLOYING. min_replicas stays 0, so the scale-to-zero
    # requirement ("should also scale down if not used") is untouched: this replica
    # goes away after downscale_delay_s like any other.
    #
    # metrics_interval_s and look_back_period_s are declared because this deployment
    # scales to ZERO, which makes the autoscaler's own sampling cadence the dominant
    # term in what a caller waits for.
    #
    # Serve's defaults are metrics_interval_s=10.0 and look_back_period_s=30.0. With
    # min_replicas 0 that means a request arriving at an idle app waits for the next
    # handle->controller metrics push before the controller can even see that anything
    # is queued, and then for a 30 s window to carry enough signal to decide. The cost
    # was measured on this app's own cold-start path, varying ONLY this field:
    #
    #     metrics_interval_s = 10 (default)   phase A median 14.698 s
    #     metrics_interval_s = 1              phase A median  2.142 s     6.9x
    #
    # where phase A is autoscale + placement + runtime_env, i.e. everything before the
    # replica's actor exists. In production that cost is plain: on the deNBI
    # watcher's own series the app answers in ~190-490 ms when a replica is already up
    # and 9.7-12.6 s when it is not, with nothing in between -- a bimodal distribution
    # whose slow mode IS this wait.
    #
    # 1.0 is used rather than the sibling's 2.0 because 1.0 is the value actually
    # measured; 2.0 would be an interpolation. The cost is this ONE deployment's
    # handle->controller push rate, max_replicas being 1, which is why this is the
    # per-deployment field and not the worker-wide
    # RAY_SERVE_HANDLE_AUTOSCALING_METRIC_PUSH_INTERVAL_S env var -- that one would
    # change every deployment the worker hosts and is set elsewhere.
    #
    # Both keys are real AutoscalingConfig fields in Ray 2.55.1, asserted rather than
    # assumed: AutoscalingConfig drops unknown keys instead of raising, so a
    # renamed field would reach Serve and be lost with no warning. Verified in the
    # worker image with a control -- `zzz_not_a_field` is absent from model_fields
    # while these two are present, and both round-trip.
    autoscaling_config={
        "min_replicas": 0,
        "initial_replicas": 1,
        "max_replicas": 1,
        "metrics_interval_s": 1.0,
        "look_back_period_s": 10.0,
        "downscale_delay_s": 600.0,
        "upscale_delay_s": 0.0,
    },
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    graceful_shutdown_timeout_s=300.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class EntryApp:
    """CPU entry: transport + routing to the GPU runtime + session orchestration."""

    def __init__(
        self,
        runtime: RuntimeApp = None,
        cellpose_runtime: CellposeRuntime = None,
        declared_gpu_memory_mb: Optional[int] = None,
        declared_gpu_device: Optional[str] = None,
    ) -> None:
        # A backend disabled via MODEL_FINETUNE_BACKENDS is not composed, so its
        # handle arrives as None; routing/capability methods reject it by name.
        self._enabled_backends = set(_ENABLED_BACKENDS)
        self.runtime = runtime
        self.cellpose_runtime = cellpose_runtime
        # Deploy-time declared GPU VRAM (via deploy_app application_kwargs). When
        # set, get_training_capabilities answers entirely CPU-side from it and
        # never composes to a GPU runtime — required under SLURM autoscale, where
        # a live probe would wake a cold GPU allocation and time out the RPC. The
        # detected value still enforces in start_training, so a wrong declaration
        # (declare 40 GB, land on a T4) fails fast instead of OOMing mid-train.
        self.declared_gpu_memory_mb = declared_gpu_memory_mb
        self.declared_gpu_device = declared_gpu_device
        self.start_time = time.time()
        self._hypha_token = os.getenv("HYPHA_TOKEN")
        if not self._hypha_token:
            raise RuntimeError("HYPHA_TOKEN environment variable is not set")
        self._hypha = None
        self._s3 = None
        self._http: Optional[httpx.AsyncClient] = None
        self._training_tasks: Dict[str, asyncio.Task] = {}
        self._export_tasks: Dict[str, asyncio.Task] = {}
        # Bound concurrent GPU fine-tuning. Default 1 serializes training so
        # autoscaled replicas stay free to serve inference during a long run
        # (mirrors model-runner's _env_build_lock); MODEL_FINETUNE_MAX_CONCURRENT_TRAINING
        # raises it when each concurrent run has its own GPU replica.
        self._training_sem = asyncio.Semaphore(_MAX_CONCURRENT_TRAINING)
        # Community-pool weight-transport accounting. bytes_up/down count the
        # WEIGHT bytes this worker moved to/from the pool — the flagship's
        # "weights travel, data stays put" evidence. Persisted under the mounted
        # ~/.bioengine so a worker restart does not silently reset the cumulative
        # total (the number would otherwise under-count in the unsafe direction);
        # the final accumulation location is being settled with the campaign page.
        self._transport = self._load_transport()
        # A merge is minutes to an hour on the entry; the runtimes report their
        # own bounds, but neither sees that an aggregate is what is holding them.
        self._merges_in_flight = 0
        self._busy = busy.BusyLog({"aggregate": lambda: self._merges_in_flight > 0})
        # One merge at a time per pool. aggregate() picks its community version by
        # probing which ones exist and then spends minutes building the soup before
        # publishing, so two overlapping calls both read the same free vN. Hypha does
        # not arbitrate it: two callers can hold stage=True on one artifact at once,
        # and the first commit clears the single artifact-level flag, so a caller's
        # commit can report success over bytes another caller PUT.
        # This is process-local and that is sufficient ONLY because EntryApp is pinned
        # to max_replicas=1 (see the deployment decorator) — every aggregate lands in this
        # process. Raising max_replicas voids it and needs a pool-side lease instead.
        self._pool_locks: Dict[str, asyncio.Lock] = {}
        # When each backend's `ping` last failed to answer, so the POLLED
        # capabilities method stops asking a runtime that is not there. An
        # under-GPU'd cluster leaves a backend UPSCALING permanently, and that
        # is not a transient: without this, every poll pays the full timeout
        # forever. Per-instance rather than class-level, so two replicas cannot
        # share one another's negative verdict.
        self._store_view_unreachable_at: Dict[str, float] = {}

        # session_id -> the JSON this instance last pushed to the shared run
        # record. Only used to skip an unchanged write; it is NOT a claim about
        # what the artifact holds, since another instance may have written the
        # same path since. Starting empty after a restart therefore costs one
        # re-push per run and cannot lose anything.
        self._runs_pushed: Dict[str, str] = {}
        self._runs_sync_task: Optional[asyncio.Task] = None
        # The shared history as last read. None means "never read": distinct
        # from {} so the first listing refreshes rather than reporting an empty
        # shared history it never asked for.
        self._runs_cache: Optional[Dict[str, Dict[str, Any]]] = None
        # file name -> (size, last_modified) as the artifact last reported, so
        # a refresh downloads only what changed.
        self._runs_meta: Dict[str, Any] = {}
        self._runs_cache_at: float = 0.0

    @bioengine.async_init
    async def _connect(self) -> None:
        self._hypha = await connect_to_server(
            {"server_url": SERVER_URL, "token": self._hypha_token}
        )
        # Resolved once per replica and cached for its whole life, so a plain
        # proxy pinned to a client id the server later forgets would hang every
        # call to the full method timeout. `put_file`/`get_file` only mint a
        # presigned URL, so both are safe to re-send after a call that may
        # already have landed.
        #
        # `self._s3` IS wrapped and `self._hypha` above is deliberately NOT wrapped.
        # That asymmetry is a decision, not an oversight, and this is where a reader
        # meets it. `reconnecting_service` is a drop-in for `get_service(id)`: it
        # re-resolves a service proxy, which is a different object from the connection
        # it was resolved through, so it cannot be applied to `self._hypha` at all —
        # and re-resolving anything through a dead connection fails the same way.
        # What covers the connection instead is the bounded round trip in `_health`:
        # it fails, Serve replaces the replica, and `_connect` runs again on a fresh
        # connection. Replacing the connection is the only repair that works when the
        # connection itself is what died.
        self._s3 = await bioengine.reconnecting_service(
            self._hypha,
            "public/s3-storage",
            repeatable=("put_file", "get_file"),
            logger=logger,
        )
        self._http = httpx.AsyncClient(timeout=60.0)
        self._busy.start()
        # Started here rather than in __init__ because it needs the connection
        # and the HTTP client this method builds. A restart re-runs _connect, so
        # the loop comes back with the replica.
        self._runs_sync_task = asyncio.create_task(self._runs_sync_loop())
        logger.info(f"Connected to Hypha server at {SERVER_URL}")
        self._log_gpu_placement()

    def _log_gpu_placement(self) -> None:
        """Say in the entry's own log whether the GPU runtimes can be placed.

        This is the signal that was missing. The entry replica reaches HEALTHY on
        any node, so these lines are in ``deployments.<entry>.logs`` — which
        ``get_app_status`` already returns — while the GPU deployments the operator
        is actually staring at are still reporting ``UPSCALING`` with no reason
        attached. A warning only: an over-subscribed cluster is a deployment
        mistake to be told about, not a reason to refuse to serve the CPU methods
        that work fine.
        """
        report = _gpu_placement_report(self._enabled_backends)
        if report["sufficient"] is False:
            logger.warning(f"⚠ GPU placement: {report['detail']} {report['advice']}")
        else:
            logger.info(f"GPU placement: {report['detail']}")

    @bioengine.health_check
    async def _health(self) -> None:
        # Entry's own dependency only; the GPU runtime is checked per-call so
        # CPU-only methods stay served when the runtime is down/scaling.
        if self._s3 is None:
            raise RuntimeError("S3 storage service not connected")

        # The run store has to be WRITABLE, and this replica is the only thing
        # that can find out -- so this probe writes, rather than asking whether
        # the directory exists. Every run-touching method resolves it, so a replica
        # that cannot reach it serves none of them -- but it answers everything
        # else perfectly well, so without this check Serve sees a healthy actor
        # and leaves it in the routing table.
        #
        # Measured 2026-11-25: a replica whose `Path.home()` was `/home/ray` on
        # a read-only filesystem reported RUNNING with every deployment HEALTHY
        # while `start_training`, `list_training_sessions` and
        # `get_training_status` all failed identically. `get_training_capabilities`
        # still worked, which is why nothing looked wrong from outside.
        #
        # Raising makes Serve replace the replica, which is the remedy this
        # method already relies on for a dead Hypha connection below. Do NOT
        # read that as "rescheduling fixes this": the broken replica and a
        # working one had the SAME read-only /home/ray, and what differed was
        # whether the per-application HOME override took effect inside the
        # process (`bioengine/_app/replica_init.py` sets HOME to the app's own
        # directory on the shared mount). So replacement helps only if that
        # failure was transient; if it is not, the replica keeps failing this
        # check and the app reports UNHEALTHY, which is the honest outcome and
        # the one this issue asks for. Cheap either way: the records a caller
        # polls are on disk, so a teardown between calls is invisible to the
        # frontend.
        #
        # The probe WRITES. `sessions_root()` alone proves only that the
        # directory could be created, and `mkdir(exist_ok=True)` succeeds on a
        # directory that already exists however read-only it is -- Linux reports
        # EEXIST before EROFS. A store that exists and refuses writes is exactly
        # this issue's symptom, so checking creatability would have passed the
        # replica that provoked the report.
        #
        # Off the event loop and time-limited, like the echo probe below: this
        # is a synchronous round trip to an NFS mount, and a hung mount would
        # otherwise block the whole replica's loop and fail only via Serve's own
        # 30s timeout, with the less specific failure that probe exists to avoid.
        import training

        loop = asyncio.get_event_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, training.assert_sessions_root_writable),
                timeout=_HEALTH_CALL_TIMEOUT_S,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"training-run store did not answer a write within "
                f"{_HEALTH_CALL_TIMEOUT_S:.0f}s; treating the mount as hung"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"training-run store unusable: {exc}") from exc
        # The `is None` test above cannot fail on a DEAD handle. hypha_rpc pins a proxy
        # to one client instance of the server; when that instance is gone every call
        # fails at send time forever while the websocket underneath happily reconnects.
        # A replica in that state keeps passing this check, stays in the Serve routing
        # table, and turns every call into a full-method-timeout hang instead of a fast
        # failure.
        #
        # So exercise the connection for real. `echo` is the cheapest call that proves
        # it — it reaches the Hypha server and nothing else, touching neither S3 nor a
        # GPU runtime — and it is the same probe `apps/model-runner/entry.py` uses in
        # its own health path. Raising here makes Serve replace the replica, which
        # rebuilds the connection; that is the remedy annotation-broker settled on
        # after the 2026-08-24 outage (`broker.py` `_health`), and it is cheap for this
        # app for the reason the autoscaling block above already relies on: the records
        # a caller polls are on disk, so a teardown between calls is invisible to the
        # frontend. Why the connection is covered this way rather than by wrapping it
        # is recorded in `_connect`, at the call that wraps `self._s3`.
        #
        # Bounded well under `health_check_timeout_s=30.0` so a hung connection is
        # reported as unhealthy rather than timing the check itself out, which Serve
        # would otherwise surface as a less specific failure.
        try:
            await asyncio.wait_for(
                self._hypha.echo("ping"), timeout=_HEALTH_CALL_TIMEOUT_S
            )
        except Exception as exc:
            raise RuntimeError(f"Hypha server connection unusable: {exc}") from exc

    def _runtime_for(self, model_type: Optional[str]):
        """Route to the backend that owns ``model_type`` (cpsam → CellposeRuntime,
        vit_* / None → micro-sam RuntimeApp). Raises if that backend is not
        deployed on this worker — the single choke point every serving/training
        path funnels through, so one guard here rejects a disabled backend
        everywhere (all direct ``self.runtime`` uses are gated by a prior
        ``_check_runtime_available``, which routes through here)."""
        backend = "cellpose" if model_type in CELLPOSE_MODEL_TYPES else "microsam"
        if backend not in self._enabled_backends:
            raise ValueError(
                f"The '{backend}' backend is not deployed on this worker "
                f"(MODEL_FINETUNE_BACKENDS={','.join(sorted(self._enabled_backends))}); "
                f"model_type '{model_type}' is unavailable here."
            )
        return self.cellpose_runtime if backend == "cellpose" else self.runtime

    async def _refuse_unavailable_base_checkpoint(self, model_type: str) -> None:
        """Raise if this site cannot produce *model_type*'s base checkpoint.

        The probe runs on the RUNTIME replica, because that is where micro-sam
        and its cache live -- `micro-sam` is in requirements-runtime.txt and is
        not importable in the worker image, so an entry-side check would be
        measuring the wrong filesystem.

        AN OLDER REPLICA IS DETECTED THROUGH `ping`, NOT THROUGH `getattr`.
        `BioEngineRuntimeHandle.__getattr__` returns a bound method for ANY
        public name, so `getattr(runtime, "base_checkpoint_status", None)` is
        never None and a capability test written that way is dead code: the
        call would fail at the far end, land in the except below, and refuse
        EVERY micro-sam run during a rollout. `ping` exists on every version of
        this runtime, so a payload whose `capabilities` lack the probe
        identifies an old replica positively, and that skips -- restoring the
        previous behaviour rather than stopping all training.

        Once a replica HAS advertised the probe, the decision fails closed: a
        probe error or an unanswerable part refuses, because admitting the run
        when the check could not answer restores exactly what this removes.
        """
        import training

        runtime = self._runtime_for(model_type)
        try:
            pong = await asyncio.wait_for(runtime.ping(), timeout=2.0)
        except Exception as exc:
            # The runtime was reachable a moment ago via _check_runtime_available;
            # if its ping now fails, that is a runtime problem and not this
            # guard's to convert into a training refusal.
            self.logger.warning(
                "could not read runtime capabilities (%s); skipping the "
                "base-checkpoint preflight for %s",
                exc,
                model_type,
            )
            return
        caps = (pong or {}).get("capabilities") or []
        if "base_checkpoint_status" not in caps:
            self.logger.warning(
                "this runtime does not advertise base_checkpoint_status "
                "(capabilities=%s); skipping the base-checkpoint preflight for "
                "%s, so a run with unfetchable weights fails late as before",
                list(caps),
                model_type,
            )
            return

        try:
            status = await asyncio.wait_for(
                runtime.base_checkpoint_status(model_type),
                timeout=BASE_CHECKPOINT_PREFLIGHT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            # A TIMEOUT IS AN ANSWER, and it is the one that matters here.
            # The case this guards against is a host that silently drops
            # packets, and "could not be checked" is wrong for that -- the
            # weights are not obtainable here within any budget a caller would
            # wait for. Reported as unreachable parts so the message says so.
            status = {
                "model_type": model_type,
                "parts": {
                    "the weights": {
                        "cached": False,
                        "reachable": False,
                        "url": None,
                        # SAID EXPLICITLY, because the derived sentence would be
                        # a falsehood: a part with no URL renders as "micro-sam
                        # publishes no URL for it", which points a reader at the
                        # registry when what actually happened is that this site
                        # could not get an answer in time.
                        "reason": (
                            "the availability check did not answer within "
                            f"{BASE_CHECKPOINT_PREFLIGHT_TIMEOUT_S:g}s, so the "
                            "weights are treated as unobtainable here"
                        ),
                    },
                },
                "probe_error": None,
            }
        except Exception as exc:
            status = {
                "model_type": model_type,
                "parts": {},
                "probe_error": "%s: %s" % (type(exc).__name__, exc),
            }
        refusal = training.base_checkpoint_refusal(model_type, status)

        # THE MIRROR FALLBACK, AND IT BELONGS HERE RATHER THAN IN THE PROBE.
        # The probe is read-only and runs on the admission path for every run;
        # fetching inside it would make the common case pay for the rare one.
        # This runs only once the probe's answer is a REFUSAL, so a site that
        # can reach upstream never reaches this branch at all.
        #
        # Gated on the advertised capability, NOT on `getattr`: the runtime
        # handle returns a bound method for any public name, so a getattr test
        # is dead code and would call a method an older replica does not have.
        if refusal and "ensure_base_checkpoint" in caps:
            try:
                seeded = await asyncio.wait_for(
                    runtime.ensure_base_checkpoint(model_type),
                    # ITS OWN BUDGET, because this one DOWNLOADS. The probe's
                    # 12 s is sized for a HEAD; the largest mirrored encoder is
                    # ~375 MB, which no probe timeout would fit, and reusing
                    # that budget would turn every fallback into a timeout.
                    timeout=BASE_CHECKPOINT_MIRROR_TIMEOUT_S,
                )
            except Exception as exc:
                # FAILS CLOSED: the refusal computed above still stands, so a
                # mirror that could not help leaves the run refused with the
                # original reason rather than admitted. Logged because the
                # reason the fallback did not help is otherwise invisible --
                # and a fence rejection arrives here as a plain RPC error, so
                # this is also where "the mirror served bytes that did not
                # verify" becomes readable.
                self.logger.warning(
                    "the base-checkpoint mirror fallback did not help for %s "
                    "(%s: %s); the run stays refused",
                    model_type,
                    type(exc).__name__,
                    exc,
                )
            else:
                if seeded.get("installed"):
                    self.logger.warning(
                        "seeded %d base-checkpoint part(s) for %s from the "
                        "mirror: %s",
                        len(seeded["installed"]),
                        model_type,
                        ", ".join(
                            "%s sha256=%s" % (r["name"], r["sha256"])
                            for r in seeded["installed"]
                        ),
                    )
                # RE-PROBE WHENEVER THE FALLBACK RETURNED CLEANLY, not only
                # when it installed something.
                #
                # A review caught this: `ensure_base_checkpoint` legitimately
                # returns `installed: []` with no error when the state changed
                # under the probe -- the file arrived between the probe and the
                # fetch (the window the disk re-check NARROWS -- it does not
                # close it, as `ensure_base_checkpoint` says in terms), or its own
                # re-probe found the part cached or upstream reachable after all.
                # Keying the re-probe on `installed` kept the STALE refusal in
                # exactly those cases, so a run whose weights ARE cached was
                # refused with the message "not cached here and not reachable
                # from here". The fallback having nothing to do is the GOOD
                # outcome, and it was the one treated as failure.
                #
                # RE-PROBE rather than trusting the seeding result either way.
                # The question is what the CACHE now holds, and the only
                # authority on that is the same probe that refused -- this is
                # the artifact-versus-our-own-account distinction, on a
                # filesystem instead of an artifact.
                if seeded.get("error") is None:
                    try:
                        status = await asyncio.wait_for(
                            runtime.base_checkpoint_status(model_type),
                            timeout=BASE_CHECKPOINT_PREFLIGHT_TIMEOUT_S,
                        )
                        refusal = training.base_checkpoint_refusal(model_type, status)
                    except Exception as exc:
                        self.logger.warning(
                            "could not re-check the base checkpoint for %s "
                            "after seeding (%s: %s); the run stays refused",
                            model_type,
                            type(exc).__name__,
                            exc,
                        )

        if refusal:
            raise ValueError(refusal)

    async def _check_runtime_available(self, model_type: Optional[str] = None) -> None:
        runtime = self._runtime_for(model_type)
        try:
            await asyncio.wait_for(runtime.ping(), timeout=2.0)
        except Exception as e:
            raise RuntimeError(
                "GPU runtime is not available yet — inference, embedding, ONNX "
                "export, and training are unavailable until it starts."
            ) from e

    async def _runtime_store(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """The store verdict of the RUNTIME that would serve *model_type*.

        Asked of the runtime rather than computed here, and that distinction is
        the whole point: a runtime replica has its own HOME, so the entry's own
        store says nothing about the one training writes. #0931 measured a
        replica writing the share while another in the same image could not
        write anything, because the redirect pointing HOME at the share had not
        run in that process.

        An unreachable runtime yields `{}` rather than a verdict: "I could not
        ask" must not read as "the store is fine", and it must not read as
        "the store is broken" either -- the caller distinguishes them.
        """
        try:
            # Inside the try, deliberately: `_runtime_for` itself raises when a
            # backend is not deployed, and a half-constructed instance has no
            # runtime handle at all. Neither is a store verdict, and neither
            # may turn this diagnostic into the thing that breaks the call.
            runtime = self._runtime_for(model_type)
            return await asyncio.wait_for(runtime.store_status(), timeout=30.0)
        except Exception as exc:
            logger.warning(
                f"could not read the runtime's store status: "
                f"{type(exc).__name__}: {str(exc)[:160]}"
            )
            return {}

    def _store_view_failures(self) -> Dict[str, float]:
        """When each backend's store `ping` last failed, created on first use.

        Reached through an accessor rather than read off the attribute directly
        because this is a CACHE, "no failures recorded yet" is the right answer
        for an instance that has no history, and several call sites build this
        class without running `__init__`. Crashing one of them with an
        AttributeError would turn a missing optimisation into a broken status
        method -- which it did, once, before this accessor existed.
        """
        failures = getattr(self, "_store_view_unreachable_at", None)
        if failures is None:
            failures = {}
            self._store_view_unreachable_at = failures
        return failures

    async def _backend_store_view(self, backend: str) -> Dict[str, Any]:
        """A backend's store verdict for the STATUS surface, from the cache.

        Read through `ping`, which reports the runtime's cached verdict and
        never probes -- so this cannot write a file and cannot hang on a wedged
        mount, which matters because `get_training_capabilities` is what the
        annotate page polls. `store_status` would be the fresher answer and is
        deliberately NOT used here for that reason; the refusal path uses it,
        where one probe per training request is affordable.

        `{}` means "could not ask", which is a third answer and is reported as
        `writable: None` rather than collapsed into either verdict.

        BOUNDED, AND IT STOPS ASKING A BACKEND THAT IS NOT THERE. See
        `_STORE_VIEW_PING_TIMEOUT_S` for the measurement that forced both: the
        caller is polled, and a backend left UPSCALING by an under-GPU'd cluster
        never answers, so an unconditional ping per poll is an unbounded cost
        for an answer that will not arrive. A backend that failed is skipped for
        `_STORE_VIEW_UNREACHABLE_FOR_S` and reports the same `{}` it would have
        reported after waiting -- the ANSWER is unchanged, only the waiting is.
        """
        failures = self._store_view_failures()
        last_failed = failures.get(backend)
        if (
            last_failed is not None
            and (time.time() - last_failed) < _STORE_VIEW_UNREACHABLE_FOR_S
        ):
            return {}
        probe = "cpsam" if backend == "cellpose" else None
        try:
            runtime = self._runtime_for(probe)
            out = await asyncio.wait_for(
                runtime.ping(), timeout=_STORE_VIEW_PING_TIMEOUT_S
            )
        except Exception as exc:
            failures[backend] = time.time()
            logger.warning(
                f"could not read the {backend} runtime's store verdict: "
                f"{type(exc).__name__}: {str(exc)[:120]}. Not asking it again "
                f"for {_STORE_VIEW_UNREACHABLE_FOR_S:.0f}s."
            )
            return {}
        # Cleared on success, so a runtime that comes back is asked again
        # without waiting out the window.
        failures.pop(backend, None)
        store = (out or {}).get("store")
        return store if isinstance(store, dict) else {}

    async def _require_runtime_store_writable(
        self, model_type: Optional[str] = None
    ) -> None:
        """Refuse a training request up front if the runtime cannot write.

        Before this, such a run was ACCEPTED, queued, handed to the runtime,
        and died at the runtime's first status write -- leaving the user a Ray
        traceback as the reason and the run FAILED. Refusing at the door costs
        one cached probe and says which store and why.

        An unreadable verdict does NOT refuse: `_check_runtime_available` has
        already established that the runtime answers, so a missing verdict here
        is this check's own failure, and taking the app's training offline
        because a status call timed out would be a worse trade than letting the
        run proceed and fail at its write, which is the behaviour today.
        """
        report = await self._runtime_store(model_type)
        if report.get("writable") is False:
            raise RuntimeError(
                "Training is unavailable on this worker: "
                + str(report.get("detail") or "the runtime cannot write its run store")
                + " No run was started. This is a worker-side fault rather "
                "than a problem with the request."
            )

    async def _runtime_gpu_memory(
        self, model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """VRAM of the backend runtime that owns ``model_type`` (empty dict if the
        runtime doesn't answer)."""
        runtime = self._runtime_for(model_type)
        try:
            return await asyncio.wait_for(runtime.gpu_memory_info(), timeout=5.0)
        except Exception:
            return {}

    async def _gpu_view(self, backend: str) -> Dict[str, Any]:
        """VRAM view for a backend used by get_training_capabilities. A deploy-time
        declared value wins and is answered CPU-side (never wakes a GPU); otherwise
        fall back to a live runtime probe, which is only cheap on a warm cluster."""
        if self.declared_gpu_memory_mb is not None:
            return {
                "available": True,
                "total_mb": self.declared_gpu_memory_mb,
                "device_name": self.declared_gpu_device or "declared",
                "source": "declared",
            }
        model_type = "cpsam" if backend == "cellpose" else None
        gpu = await self._runtime_gpu_memory(model_type)
        return {**gpu, "source": "detected"} if gpu else {}

    # === I/O transport (S3 + URL) ===

    @bioengine.method
    async def get_upload_url(
        self,
        file_type: UPLOAD_FILE_TYPES = Field(
            ...,
            description='File type for the upload: ".npy" (NumPy array), ".png", '
            '".tiff"/".tif", ".jpeg"/".jpg", or ".npz" (a compute_embedding bundle '
            "to store yourself and feed to infer).",
        ),
    ) -> Dict[str, str]:
        """Presigned S3 PUT URL (1-hour TTL) for staging an input image or embedding."""
        file_path = f"temp/{uuid.uuid4()}{file_type}"
        upload_url = await self._s3.put_file(file_path=file_path, ttl=3600)
        return {"upload_url": upload_url, "file_path": file_path}

    async def _get_download_url(self, file_path: str) -> str:
        return await self._s3.get_file(file_path=file_path, use_proxy=True)

    # Exponential backoff (~30s total) covers a full broker commit + re-stage
    # cycle, during which a presigned URL can transiently 403/404/5xx. micro-sam
    # stays artifact-agnostic: it retries the raw HTTP transfer, never inspects
    # stage state, and never retries indefinitely.
    _HTTP_RETRY_BACKOFF = (1.0, 2.0, 4.0, 8.0, 15.0)

    async def _http_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Issue a request on a presigned URL, retrying transient failures.

        Retries on 403/404/408/429/5xx and httpx transport errors with the
        backoff above, then returns the final response WITHOUT raising for
        status — callers keep their own handling (e.g. 404 -> FileNotFoundError,
        raise_for_status on a PUT). An exhausted retry loop must surface loudly.

        If ``content`` is callable it is invoked fresh on every attempt to yield
        a new body — required for streamed uploads, whose async iterators are
        single-use and would be spent after a retry.
        """
        retry_status = {403, 404, 408, 429}
        body_factory = (
            kwargs.pop("content", None) if callable(kwargs.get("content")) else None
        )
        for delay in (*self._HTTP_RETRY_BACKOFF, None):
            if body_factory is not None:
                kwargs["content"] = body_factory()
            try:
                resp = await self._http.request(method, url, **kwargs)
            except httpx.TransportError:
                if delay is None:
                    raise
                await asyncio.sleep(delay)
                continue
            if delay is not None and (
                resp.status_code in retry_status or resp.status_code >= 500
            ):
                await asyncio.sleep(delay)
                continue
            return resp

    async def _get_verified(self, url: str, **kwargs) -> bytes:
        """GET a body whole, rejecting a short read.

        httpx raises on a body cut short of its declared length, so this check is a
        second line of defence — but the one that matters: a prefix reaching
        ``_write_file_atomic`` is indistinguishable from a prefix a reader sees for
        lack of an fsync, both surfacing as ``failed finding central directory`` from
        ``torch.load`` far from the cause. Same contract as
        ``runtime_cellpose._download_member``.
        """
        from http.client import IncompleteRead

        resp = await self._http_retry("GET", url, **kwargs)
        # The length check alone cannot catch an error response: 403/404 are in
        # _http_retry's retry set and it returns the last one unraised, and an S3
        # error body declares its own length correctly — so a few hundred bytes of
        # XML passes as "whole" and gets written as a .pt.
        resp.raise_for_status()
        content = resp.content
        declared = resp.headers.get("Content-Length")
        if declared is not None and len(content) != int(declared):
            raise IncompleteRead(content, int(declared) - len(content))
        return content

    @staticmethod
    async def _file_chunks(fpath: Path, size: int = 1 << 20):
        with open(fpath, "rb") as fh:
            while True:
                chunk = await asyncio.to_thread(fh.read, size)
                if not chunk:
                    break
                yield chunk

    async def _put_file(
        self, url: str, fpath: Path, timeout: float = 600.0
    ) -> httpx.Response:
        """PUT a file to a presigned URL, streaming from disk so large weights
        (cpsam packages ship a ~1.2GB model_weights.pth) never load whole into
        RAM and OOM the memory-constrained EntryApp actor. Content-Length is set
        explicitly because S3 presigned PUTs reject chunked transfer encoding."""
        size = await asyncio.to_thread(lambda: fpath.stat().st_size)
        return await self._http_retry(
            "PUT",
            url,
            content=lambda: self._file_chunks(fpath),
            headers={"Content-Length": str(size)},
            timeout=timeout,
        )

    async def _load_image_from_source(self, source: str) -> np.ndarray:
        ext = Path(source.split("?")[0]).suffix.lower()
        if ext not in SUPPORTED_FILE_TYPES.__args__:
            raise ValueError(
                f"Unsupported file extension '{ext}' in source '{source}'. "
                f"Supported: {SUPPORTED_FILE_TYPES.__args__}"
            )
        url = (
            source
            if source.startswith(("http://", "https://"))
            else (await self._get_download_url(source))
        )
        resp = await self._http_retry("GET", url)
        if resp.status_code == 404:
            raise FileNotFoundError(f"Source '{source}' does not exist or has expired.")
        resp.raise_for_status()
        buffer = BytesIO(resp.content)
        if ext == ".npy":
            return await asyncio.to_thread(np.load, buffer)
        import imageio.v3 as iio

        return await asyncio.to_thread(iio.imread, buffer)

    async def _resolve_image(self, source: Union[np.ndarray, str]) -> np.ndarray:
        # A region reference is read BEFORE the string path, and only its own
        # shape routes it here: `is_region_source` requires a mapping carrying
        # kind="ngff" and a store_root, so every existing caller — array, URL or
        # artifact path — falls through untouched.
        #
        # WHY THE CALLER SUPPLIES THE LAYOUT rather than this app asking the
        # broker for it. The entry talks to `public/artifact-manager` and to
        # nothing else today; calling annotation-broker from here would make
        # training depend on that app being deployed and reachable, for metadata
        # the caller already holds — it got `dataset_paths`, `axis_roles` and
        # `level0_shape` from `probe_ngff_source` when it imported the source.
        # So the reference carries the layout beside the region, and a caller
        # that omits it is refused BY NAME in `read_region` rather than having a
        # 2D no-channel layout guessed for it.
        if ngff_region.is_region_source(source):
            return await asyncio.to_thread(
                ngff_region.read_region, source, dict(source)
            )
        if isinstance(source, str):
            return await self._load_image_from_source(source)
        if isinstance(source, np.ndarray):
            return source
        return np.asarray(source)

    async def _resolve_label_array(
        self, source: Union[np.ndarray, str], ref_shape: tuple
    ) -> np.ndarray:
        import training

        if isinstance(source, np.ndarray):
            return source
        if not isinstance(source, str):
            return np.asarray(source)
        ext = Path(source.split("?")[0]).suffix.lower()
        if ext in (".geojson", ".json"):
            url = (
                source
                if source.startswith(("http://", "https://"))
                else (await self._get_download_url(source))
            )
            resp = await self._http.get(url)
            resp.raise_for_status()
            h, w = int(ref_shape[0]), int(ref_shape[1])
            payload = resp.json()
            mask = training.rasterize_geojson(payload, width=w, height=h)
            # A GeoJSON that carries polygons and rasterises to NOTHING is a
            # coordinate-frame mismatch, and it is silent: training proceeds on
            # an all-background mask and reports success, having learnt nothing.
            #
            # The case that motivates it: a crop read from a whole-slide source
            # is a WxH canvas, while the broker describes a region "in level-0
            # pixels". If the annotator writes polygons in slide coordinates,
            # every one of them lands outside the crop. Which frame it writes is
            # not settled in this repository, so rather than assume one, this
            # refuses the outcome that is wrong under either.
            #
            # Deliberately not region-specific: features that draw nothing on a
            # full image are the same defect with the same cause.
            if mask.max() == 0:
                bounds = training.polygon_coordinate_bounds(payload)
                if bounds:
                    raise ValueError(
                        f"The annotation for this image has "
                        f"{int(bounds['features'])} polygon(s) but rasterises to "
                        f"an empty mask on the {w}x{h} canvas: their coordinates "
                        f"span x {bounds['x_min']:.0f}..{bounds['x_max']:.0f}, "
                        f"y {bounds['y_min']:.0f}..{bounds['y_max']:.0f}. Training "
                        f"on it would learn only background. If this image is a "
                        f"region of a larger source, the polygons are most likely "
                        f"in the source's coordinates and need offsetting by the "
                        f"region origin."
                    )
            return mask
        return await self._load_image_from_source(source)

    async def _save_npy_to_temp(self, array: np.ndarray) -> str:
        info = await self.get_upload_url(file_type=".npy")
        buffer = BytesIO()
        np.save(buffer, array)
        await self._http.put(info["upload_url"], content=buffer.getvalue())
        return await self._get_download_url(info["file_path"])

    @staticmethod
    def _embedding_npz_bytes(features, input_size, original_size, model_type) -> bytes:
        buffer = BytesIO()
        np.savez(
            buffer,
            features=np.asarray(features),
            input_size=np.asarray(input_size),
            original_size=np.asarray(original_size),
            model_type=str(model_type),
        )
        return buffer.getvalue()

    async def _put_temp_embedding(self, content: bytes) -> str:
        file_path = f"temp/{uuid.uuid4()}.npz"
        upload_url = await self._s3.put_file(file_path=file_path, ttl=3600)
        resp = await self._http_retry("PUT", upload_url, content=content)
        resp.raise_for_status()
        return await self._get_download_url(file_path)

    async def _resolve_embedding(self, e: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve an infer ``embeddings`` item to ``{features, input_size,
        original_size, model_type}``. Accepts an embedding_url (.npz from
        compute_embedding) or an inline compute_embedding result dict.
        """
        if isinstance(e, str):
            url = (
                e
                if e.startswith(("http://", "https://"))
                else await self._get_download_url(e)
            )
            resp = await self._http_retry("GET", url)
            resp.raise_for_status()
            data = await asyncio.to_thread(
                np.load, BytesIO(resp.content), allow_pickle=False
            )
            return {
                "features": np.asarray(data["features"]),
                "input_size": [int(x) for x in np.asarray(data["input_size"]).tolist()],
                "original_size": [
                    int(x) for x in np.asarray(data["original_size"]).tolist()
                ],
                "model_type": str(data["model_type"]),
            }
        if isinstance(e, dict):
            original_size = e.get("original_image_shape") or e.get("original_size")
            input_size = e.get("input_size")
            if input_size is None:
                scale = float(e["sam_scale"])
                input_size = [
                    round(original_size[0] * scale),
                    round(original_size[1] * scale),
                ]
            return {
                "features": np.asarray(e["features"]),
                "input_size": [int(x) for x in input_size],
                "original_size": [int(x) for x in original_size],
                "model_type": e.get("model_type"),
            }
        raise ValueError(
            "embeddings items must be an embedding_url (str) or a compute_embedding dict."
        )

    def _session_checkpoint(self, session_id: str) -> str:
        import training

        ckpt = training.checkpoint_path(session_id)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"No trained checkpoint for session '{session_id}' "
                "(training may be unfinished or failed)."
            )
        return str(ckpt)

    async def _fetch_model_package(
        self, model_id: str, model_token: Optional[str]
    ) -> Dict[str, Any]:
        """Download an exported bioimage.io micro-sam package (published or the
        caller's draft/staged) into a shared HOME cache and return
        ``{weights_path, model_type}``. The package is a combined
        ``{model_state, decoder_state}`` checkpoint served through the same AIS
        path as a fine-tuned session. ``model_token`` (the caller's own token)
        grants read access to a draft in their workspace; without it the app token
        is used, which only reaches published models.
        """
        cache_dir = (
            Path.home()
            / ".bioengine"
            / "micro_sam_zoo_cache"
            / re.sub(r"[^\w.-]", "_", model_id)
        )
        weights_path = cache_dir / "weights.pt"
        meta_path = cache_dir / "kwargs.json"
        if weights_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            # The recorded length, not the pair existing, is what says the weights are
            # whole. Both files existing only proves they were written by a version that
            # had ``_write_file_atomic``; versions before it wrote ``weights.pt`` in
            # place, and this cache lives in an app HOME that outlives any one version.
            # So an entry carrying no length is a MISS and re-downloads once.
            if meta.pop("bytes", None) == weights_path.stat().st_size:
                return {"weights_path": str(weights_path), **meta}

        caller_client = (
            await connect_to_server({"server_url": SERVER_URL, "token": model_token})
            if model_token
            else None
        )
        try:
            am = await (caller_client or self._hypha).get_service(
                "public/artifact-manager"
            )

            # Prefer the new ``bioimageio.yaml`` name; fall back to legacy ``rdf.yaml``.
            rdf_url = None
            for rdf_name in ("bioimageio.yaml", "rdf.yaml"):
                try:
                    rdf_url = await am.get_file(model_id, file_path=rdf_name)
                    break
                except Exception:
                    continue
            if rdf_url is None:
                raise FileNotFoundError(
                    f"No RDF (bioimageio.yaml or rdf.yaml) found in artifact '{model_id}'."
                )
            rdf_resp = await self._http_retry("GET", rdf_url)
            rdf_resp.raise_for_status()
            rdf = yaml.safe_load(rdf_resp.text)
            ps = rdf["weights"]["pytorch_state_dict"]
            model_type = ps["architecture"]["kwargs"]["model_type"]

            w_url = await am.get_file(model_id, file_path=ps["source"])
            content = await self._get_verified(w_url, timeout=600.0)
            cache_dir.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(_write_file_atomic, weights_path, content)
            meta = {"model_type": model_type}
            # kwargs.json is the commit marker for the pair above: the cache-hit branch
            # gates on both files existing, so it must appear whole or not at all — a
            # plain write_text makes it exist (empty) before it has content, and the
            # reader then json.loads("").
            _write_file_atomic(
                meta_path, json.dumps({**meta, "bytes": len(content)}).encode()
            )
            return {"weights_path": str(weights_path), **meta}
        finally:
            if caller_client is not None:
                # A per-request caller client stays registered server-side until
                # its socket drops; disconnect frees it now. Bounded so a wedged
                # transport can't stall the request.
                try:
                    await asyncio.wait_for(caller_client.disconnect(), timeout=2.0)
                except Exception:
                    pass

    # === serving (forwarded to the GPU runtime) ===

    @bioengine.method
    async def infer(
        self,
        input_arrays: Optional[List[Union[np.ndarray, str]]] = Field(
            None,
            description="List of input images (numpy HxW/HxWx3/3xHxW, an http(s) URL, "
            "or a get_upload_url file path). Provide this OR embeddings.",
        ),
        embeddings: Optional[List[Union[str, Dict[str, Any]]]] = Field(
            None,
            description="Run AIS on precomputed embeddings instead of images (skips "
            "the encoder). Each item is an embedding_url from "
            "compute_embedding(return_url=True), or a compute_embedding result dict. "
            "Provide this OR input_arrays.",
        ),
        model_type: ModelType = Field(
            "vit_l_lm",
            description="μSAM generalist. All six — LM (vit_t/b/l_lm) and EM organelles "
            "(vit_t/b/l_em_organelles) — carry the AIS decoder. For embeddings, the model "
            "must match the one that produced them (inferred from the embedding when available).",
        ),
        min_size: Optional[int] = Field(
            None, description="Minimum object size in pixels (smaller objects dropped)."
        ),
        session_id: Optional[str] = Field(
            None,
            description="Serve the fine-tuned model from this training session's checkpoint.",
        ),
        model_id: Optional[str] = Field(
            None,
            description="Run an exported BioImage.IO micro-sam model by artifact id "
            "(e.g. 'bioimage-io/<alias>') — published, or your own draft/staged. "
            "Downloads the combined SAM+decoder package and runs its AIS decoder → "
            "instances. Requires input_arrays (not embeddings).",
        ),
        model_token: Optional[str] = Field(
            None,
            description="Hypha token with read access to model_id — needed for a "
            "draft/staged model in your workspace; omit for published models.",
        ),
        postprocessing: Optional[Dict[str, Any]] = Field(
            None,
            description="Cellpose-SAM mask-generation tuning (cpsam / cpdino only): a "
            "dict of flow_threshold (default 0.4), cellprob_threshold (default 0.0) and "
            "niter. Omit a key to keep its default. Lower cellprob_threshold to pick up "
            "faint cells; raise flow_threshold to admit noisier masks. Ignored by the "
            "micro-sam (vit_*) path.",
        ),
    ) -> List[Dict[str, Any]]:
        """Automatic instance segmentation (μSAM AIS, or Cellpose when
        ``model_type`` is a cellpose type — cpsam / cpdino / cpdino-vitb — or a
        cellpose session/model). Returns a bare list, one
        item per input, each ``{"output": <int32 [H,W] instance label mask>}``.
        Pass ``input_arrays`` (images) OR ``embeddings`` (micro-sam only —
        precomputed, runs the AIS decoder without re-encoding). Pass ``model_id``
        to run an exported BioImage.IO package instead of a built-in/finetuned model.
        """
        import training

        generate_kwargs: Dict[str, Any] = {}
        if min_size is not None:
            generate_kwargs["min_size"] = min_size
        if postprocessing:
            generate_kwargs["postprocessing"] = postprocessing
        if model_id is not None:
            if input_arrays is None:
                raise ValueError(
                    "model_id requires input_arrays (images), not embeddings."
                )
            pkg = await self._fetch_model_package(model_id, model_token)
            await self._check_runtime_available(pkg["model_type"])
            images = [await self._resolve_image(src) for src in input_arrays]
            return await self._runtime_for(pkg["model_type"]).auto_segment(
                images=images,
                model_type=pkg["model_type"],
                checkpoint=pkg["weights_path"],
                generate_kwargs=generate_kwargs,
            )
        if (input_arrays is None) == (embeddings is None):
            raise ValueError("Provide exactly one of 'input_arrays' or 'embeddings'.")
        checkpoint = self._session_checkpoint(session_id) if session_id else None
        if embeddings is not None:
            # Embeddings are a micro-sam-only path (cpsam has no encoder embedding).
            await self._check_runtime_available()
            embs = [await self._resolve_embedding(e) for e in embeddings]
            emb_model = next(
                (e["model_type"] for e in embs if e.get("model_type")), None
            )
            return await self.runtime.auto_segment_from_embedding(
                embeddings=embs,
                model_type=emb_model or model_type,
                generate_kwargs=generate_kwargs,
                checkpoint=checkpoint,
            )
        # A fine-tuned session's backend fixes the runtime; otherwise route by model_type.
        served_type = model_type
        if session_id and training.session_backend(session_id) == "cellpose":
            cparams = training.read_training_params(session_id)
            served_type = cparams.get("model_type", "cpsam")
            # Serve at the diameter the session trained/exports at, so live GPU
            # masks match the exported package's CPU self-test.
            diam = cparams.get("diam_mean")
            if diam is not None:
                generate_kwargs["diameter"] = diam
        await self._check_runtime_available(served_type)
        images = [await self._resolve_image(src) for src in input_arrays]
        return await self._runtime_for(served_type).auto_segment(
            images=images,
            model_type=served_type,
            generate_kwargs=generate_kwargs,
            checkpoint=checkpoint,
        )

    @bioengine.method
    async def compute_embedding(
        self,
        inputs: Union[np.ndarray, str] = Field(
            ..., description="A single input image."
        ),
        model_type: ModelType = Field("vit_l_lm", description="μSAM model."),
        return_url: bool = Field(
            False,
            description="If True, save the embedding as a self-contained .npz in a "
            "temporary S3 file and return its download URL as 'embedding_url' (feed it "
            "to infer(embeddings=[...])) instead of the inline 'features' ndarray.",
        ),
        embedding_upload_url: Optional[str] = Field(
            None,
            description="Presigned PUT URL (from get_upload_url('.npz')) to store the "
            "embedding .npz at, instead of a temporary S3 file. You keep the matching "
            "download URL to pass to infer(embeddings=[...]).",
        ),
        session_id: Optional[str] = Field(
            None, description="Serve a fine-tuned session checkpoint."
        ),
    ) -> Dict[str, Any]:
        """Run the encoder once and return its embedding: ``{features |
        embedding_url, original_image_shape, input_size, sam_scale, mask_threshold,
        model_type}``. The embedding feeds the in-browser ONNX prompt decoder and can
        be passed to ``infer(embeddings=[...])`` to run AIS without re-encoding. With
        ``embedding_upload_url`` the .npz is PUT to your URL and ``features`` is
        dropped from the return.
        """
        if model_type in CELLPOSE_MODEL_TYPES:
            raise ValueError(
                "compute_embedding is micro-sam only; cpsam has no embedding path."
            )
        await self._check_runtime_available()
        checkpoint = self._session_checkpoint(session_id) if session_id else None
        image = await self._resolve_image(inputs)
        payload = await self.runtime.encode(
            image=image,
            model_type=model_type,
            checkpoint=checkpoint,
        )
        payload["model_type"] = model_type
        if embedding_upload_url or return_url:
            feat = payload.pop("features")
            arr = np.frombuffer(feat["_rvalue"], dtype=feat["_rdtype"]).reshape(
                feat["_rshape"]
            )
            bundle = self._embedding_npz_bytes(
                arr, payload["input_size"], payload["original_image_shape"], model_type
            )
            if embedding_upload_url:
                resp = await self._http_retry(
                    "PUT", embedding_upload_url, content=bundle
                )
                resp.raise_for_status()
            else:
                payload["embedding_url"] = await self._put_temp_embedding(bundle)
        return payload

    @bioengine.method
    async def get_onnx_model(
        self,
        model_type: ModelType = Field("vit_l_lm", description="μSAM model."),
        quantize: bool = Field(
            True, description="Quantize for faster browser runtime."
        ),
    ) -> bytes:
        """The interactive prompt decoder as ONNX bytes for onnxruntime-web."""
        if model_type in CELLPOSE_MODEL_TYPES:
            raise ValueError(
                "get_onnx_model is micro-sam only; cpsam has no ONNX prompt decoder."
            )
        await self._check_runtime_available()
        return await self.runtime.export_onnx(model_type=model_type, quantize=quantize)

    # === fine-tuning orchestration (async-job model) ===

    async def _prepare_training_data(
        self, session_id, train_images, train_labels, val_images, val_labels, params
    ) -> Dict[str, Any]:
        import training

        if len(train_images) != len(train_labels):
            raise ValueError("train_images and train_labels must have equal length.")

        async def pairs(images, labels):
            out = []
            for img_src, lbl_src in zip(images, labels):
                img = await self._resolve_image(img_src)
                lbl = await self._resolve_label_array(
                    lbl_src, np.asarray(img).shape[:2]
                )
                out.append((img, lbl))
            return out

        train = await pairs(train_images, train_labels)
        val = await pairs(val_images, val_labels) if val_images and val_labels else None
        return await asyncio.to_thread(
            training.materialize_pairs,
            session_id,
            train,
            val,
            params["val_fraction"],
            params["patch_shape"],
        )

    async def _preparing_heartbeat(
        self, session_id: str, interval: float = 60.0
    ) -> None:
        """Refresh status.json's ``updated_at`` while a run is PREPARING or QUEUED.

        Nothing between the PREPARING write and the runtime's TRAINING write
        reports progress — input resolution, a cold runtime replica coming up,
        the GPU lock — so on a cold instance that span can exceed
        STATUS_STALE_SECONDS and get_status sweeps a healthy run to STOPPED.

        **QUEUED is covered too, and that is not cosmetic.** The semaphore
        caps concurrent trainings at one by default, so a second run waits here
        for as long as the first one trains — routinely far longer than the
        stale window. Nothing else refreshes the record in that phase, so
        without this the operator listing marks a perfectly healthy queued run
        as stale, i.e. "probably dead", and the obvious response to that is to
        stop somebody's valid job. Covering QUEUED is what makes a stale QUEUED
        record mean what it claims: the replica that was waiting is gone.

        Exits as soon as the status moves beyond those two, so a wedged
        TRAINING run is still swept; the worker's own heartbeat covers that
        phase.

        **BEST-EFFORT, like every other heartbeat**, and the raising
        `write_status` here was a defect rather than a choice. The record lock
        makes that function raise when the mutex cannot be taken or the record
        will not parse, which is right for a terminal write and wrong for a
        heartbeat -- so `write_status_best_effort` exists for exactly these call
        sites. This one runs as a detached task, so an exception ends it
        silently and nothing restarts it: `updated_at` stops advancing and the
        run it was protecting is reported stale, which is the failure this
        function exists to prevent, arrived at from the other side. Measured
        with the mutex held and the wait shortened: the task ended with
        `RunRecordLockTimeout` and the healthy QUEUED run then read
        `stale: True`. The QUEUED span can be hours, so the window is wide.
        """
        import training

        while True:
            await asyncio.sleep(interval)
            if training.read_status(session_id).get("status") not in (
                "PREPARING",
                "QUEUED",
            ):
                return
            training.write_status_best_effort(session_id)

    async def _run_training(
        self,
        session_id,
        model_type,
        train_images,
        train_labels,
        val_images,
        val_labels,
        params,
    ) -> None:
        import training

        preparing_hb = asyncio.create_task(self._preparing_heartbeat(session_id))
        try:
            data = await self._prepare_training_data(
                session_id, train_images, train_labels, val_images, val_labels, params
            )
            if data.get("val_reused_train"):
                training.write_status(session_id, val_reused_train=True)
            resolved = {}
            resume_id = params.get("resume_session_id")
            init_checkpoint = params.get("init_checkpoint")
            if resume_id:
                prior_model = training.read_training_params(resume_id).get("model_type")
                if prior_model and prior_model != model_type:
                    raise ValueError(
                        f"resume_session_id '{resume_id}' was trained as {prior_model}, "
                        f"not {model_type}; model_type must match to resume."
                    )
                resolved["checkpoint_path"] = self._session_checkpoint(resume_id)
            elif init_checkpoint:
                url = (
                    init_checkpoint
                    if init_checkpoint.startswith(("http://", "https://"))
                    else await self._get_download_url(init_checkpoint)
                )
                resp = await self._http_retry("GET", url, timeout=600.0)
                if resp.status_code != 200:
                    raise FileNotFoundError(
                        f"init_checkpoint download failed ({resp.status_code}) for {init_checkpoint}"
                    )
                dest = training.session_dir(session_id) / "init_checkpoint.pt"
                dest.parent.mkdir(parents=True, exist_ok=True)
                # dest reaches a train worker as the params' checkpoint_path, so a
                # plain write_bytes would publish the name at length zero and let
                # that worker torch.load a prefix from another node.
                await asyncio.to_thread(_write_file_atomic, dest, resp.content)
                resolved["checkpoint_path"] = str(dest)
                if params.get("pool_provenance"):
                    self._bump_transport(bytes_down=len(resp.content))
            training.write_training_params(
                session_id,
                {
                    **params,
                    "model_type": model_type,
                    "patch_shape": list(data["patch_shape"]),
                    "train_images": data["train_images"],
                    "train_labels": data["train_labels"],
                    "val_images": data["val_images"],
                    "val_labels": data["val_labels"],
                    **resolved,
                },
            )
            # Long-running: this await holds one GPU runtime replica for the whole
            # training run, and a concurrent infer queues behind it on that same
            # replica — at the effective target of 2, two calls in flight ask for
            # one replica, not two.
            # The semaphore caps concurrent trainings (default 1); extra runs wait
            # here. QUEUED is exempt from the PREPARING/TRAINING stale-window sweep,
            # so a session waiting here is not false-flagged STOPPED -- but the
            # record still has to be refreshed while it waits, or the operator
            # listing reports a healthy queued run as stale. That is what
            # `_preparing_heartbeat` covers; it runs through this phase.
            if self._training_sem.locked():
                training.write_status(
                    session_id,
                    status="QUEUED",
                    message="waiting for GPU — another fine-tuning is in progress",
                )
            async with self._training_sem:
                await self._runtime_for(model_type).train(
                    session_id=session_id, model_type=model_type, params=params
                )
        except asyncio.CancelledError:
            # Only a user stop may claim rank-4 user_stop, which no later writer
            # can correct. This task is also cancelled when the entry replica is
            # torn down (it scales to zero), while the runtime subprocess keeps
            # running and owns the real outcome — recording that as a user stop
            # would discard the child's COMPLETED for good.
            if training.stop_requested(session_id):
                training.write_status(
                    session_id,
                    status="STOPPED",
                    message="training task cancelled",
                    terminated_by="user_stop",
                )
            raise
        except Exception as e:
            # REDACTED HERE AS WELL AS AT THE RAISE SITE, and deliberately not
            # only there. Everything this message can contain comes from some
            # library's idea of a good error, and a store URL carries its
            # credential as a path segment; the session status document is
            # served by an endpoint that takes no caller context, so one of
            # those strings reaching it publishes the credential to anyone who
            # can list sessions. The region reader already refuses without
            # quoting the URL. This catches the paths nobody has thought of,
            # which is the only kind that matters for a credential.
            safe = ngff_region.redact_store_credential(e)
            logger.error(f"Training session {session_id} failed: {safe}")
            training.write_status(
                session_id, status="FAILED", message=safe[:800], terminated_by="entry"
            )
        finally:
            preparing_hb.cancel()
            self._training_tasks.pop(session_id, None)
            # NOTHING is shared from here, deliberately. The terminal state is
            # what a run history is most often asked about, so pushing it
            # immediately is tempting -- but this `finally` also runs while a
            # replica is being torn down, with a CancelledError in flight
            # (`_prepare_training_data` raising CancelledError is a tested
            # case). Awaiting a network call there either delays the
            # cancellation or swallows it, and an export that reported a user
            # stop it never received is a bug this app has already had. The
            # record is on disk; the sync loop shares it within one poll, and
            # after a restart it is re-read from disk and shared then.

    # ------------------------------------------------------------------
    # The shared run record
    #
    # Run history is a property of the PROJECT, not of one worker's filesystem.
    # While each instance answers from its own disk, two instances of the app
    # answer differently: measured on one worker as one administrator, the
    # production instance listed 261 runs, a second instance listed 19, and the
    # two had NOTHING in common. The records below are the
    # shared copy that makes the history a property of the project rather than
    # of whichever instance answered.
    #
    # Written with the APP's own credential, to ONE artifact named in
    # `training.RUNS_ARTIFACT_ID`. Deliberately not a parameter anywhere: a
    # Hypha token cannot be narrowed below its workspace, so a caller who could
    # name the artifact could make this app write anything its credential
    # reaches -- and `commit` on an artifact somebody else is staging publishes
    # their pending work. The dataset paths in this app keep the opposite rule
    # (`_pool_service`: a caller token, because the write acts as the caller);
    # the two rules sit side by side, so which is which is stated rather than
    # inferred.
    #
    # Measured on the live artifact before this was written: `edit(stage=True)`
    # + `put_file` + `commit()` does NOT add a version -- the version list
    # stayed at one entry across two commits -- and a second write to the same
    # path replaces the first. A commit per record is therefore affordable; a
    # design that added an immutable version per status update would have
    # reached thousands.
    # ------------------------------------------------------------------

    #: How often an instance pushes the records whose content has changed.
    #: Clause 3 of the issue is "within one poll", so this interval IS the
    #: promise; it is a class attribute so a test can shorten it.
    RUNS_SYNC_INTERVAL_S = 60.0

    #: How long to wait before the single retry of a share. A class attribute
    #: so a test can set it to zero: patching `asyncio.sleep` instead is how a
    #: no-op lambda ends up calling the function it replaced, which recurses.
    RUNS_RETRY_DELAY_S = 2.0

    #: How stale a cached copy of the shared history may be before a listing
    #: refreshes it. The sync loop refreshes it unconditionally each pass, so
    #: this only bounds the gap for a listing that arrives between passes.
    RUNS_CACHE_TTL_S = 30.0

    #: How many entries to ask the artifact store for per page. A class
    #: attribute so a test can set it to a handful and exercise the loop with
    #: tens of records instead of thousands. The server's own default is 1000
    #: and its ceiling is 10000, so asking for the maximum is not a fix -- it
    #: is a higher cap, with the same silent truncation one threshold up.
    RUNS_LIST_PAGE = 1000

    #: How many record uploads (and read-backs) the backfill has in flight at
    #: once. A class attribute so a test can set it to 1 and get the old
    #: serial order back, or to a large number to check the bound is what
    #: limits concurrency rather than luck.
    #:
    #: Bounded rather than unbounded on purpose: `asyncio.gather` over 313
    #: uploads would open 313 connections at once against a shared artifact
    #: store, which trades a timeout for a different kind of outage.
    #:
    #: MEASURED at 8 on a fabricated 320-record store: the apply finished in
    #: 248 s, well inside the 600 s ceiling, but 30 PUTs reported failure -- and
    #: 23 of those records WERE in the artifact afterwards while 7 were not. The
    #: store's verdict is therefore unreliable in both directions under that
    #: load, so the bound is 4 and the read-back is the only judge.
    RUNS_MIGRATE_CONCURRENCY = 4

    #: Seconds to wait after the commit before the first read-back. The ruling
    #: that shaped this asked for "a short settle"; the repair pass covers a
    #: record that is late rather than absent, but a settle costs one second and
    #: removes a class of repair work that was never needed. A class attribute so
    #: a test can set it to 0.
    RUNS_READBACK_SETTLE_S = 1.0

    # Class-level fallbacks for the cache, so a published method never raises
    # AttributeError on an instance whose __init__ did not run. That is not a
    # hypothetical: a half-constructed instance is how the suite exercises a
    # single method, and it is also what a future refactor of the constructor
    # would produce.
    #
    # Every write to these REPLACES the object and assigns the result to the
    # INSTANCE; none mutates the shared default. That is a rule to keep rather
    # than a property of the types: a review found this comment claiming it
    # while `_persist_run_record` was doing `self._runs_pushed[sid] = ...`,
    # which writes into the class dict on exactly the instances these
    # fallbacks exist for, and would share one replica's bookkeeping with
    # every other.
    _runs_cache: Optional[Dict[str, Dict[str, Any]]] = None
    _runs_meta: Dict[str, Any] = {}
    _runs_cache_at: float = 0.0
    _runs_pushed: Dict[str, str] = {}

    async def _runs_artifact_manager(self):
        """The artifact manager on the APP's own connection.

        No caller client, by construction: there is no argument to pass one.
        An earlier draft of this took `context["client"]` so the write could
        carry the caller's identity, and measured on that branch the framework
        never injected a context at all -- the method was declared without
        `context=True`, so the fence read as present and was absent, while
        `context` became a *published* parameter a caller could supply.
        """
        return await self._hypha.get_service("public/artifact-manager")

    async def _persist_run_record(
        self,
        session_id: str,
        status: Optional[Dict[str, Any]] = None,
        *,
        log_failure: bool = False,
        record: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Share one run's record. True if it landed.

        Returns rather than raises, because every caller is doing something
        else that must not fail with it: a finished run, or a poll that will
        come round again.

        `record` SUPPLIES THE DOCUMENT INSTEAD OF BUILDING ONE, and exists so a
        caller can hash the exact bytes that will be written. Rebuilding the
        record here would re-read the run, and a document built twice is not
        necessarily the same document twice: a run that moved on, or merely an
        `elapsed_s` derived from the clock, makes the second build differ from
        the first. A caller that hashed the first and then wrote the second
        would be attesting to bytes that never existed.

        A commit can legitimately lose a race -- two instances can hold
        `stage=True` on one artifact, and the first commit clears the single
        artifact-level flag, so the second can find nothing staged. The bytes
        are already uploaded at that point and the other commit may well have
        published them, so one retry is attempted and then it is left to the
        next poll.
        """
        import json as _json
        import training

        store = training.record_store_id()
        if store is None:
            # Sharing is OFF rather than guessed. See `record_store_id`: a
            # replica that cannot tell which deployment it is must not write
            # into another deployment's history.
            return False
        for attempt in (1, 2):
            try:
                if record is None:
                    record = training.run_record(session_id, status)
                path = training.run_record_path(session_id, store)
                am = await self._runs_artifact_manager()
                await am.edit(artifact_id=training.RUNS_ARTIFACT_ID, stage=True)
                url = await am.put_file(training.RUNS_ARTIFACT_ID, file_path=path)
                resp = await self._http.put(
                    url, content=_json.dumps(record).encode("utf-8")
                )
                resp.raise_for_status()
                await am.commit(artifact_id=training.RUNS_ARTIFACT_ID)
                # REPLACED, not mutated: `_runs_pushed` has a class-level
                # fallback for an instance whose __init__ did not run, and
                # writing into it there would share one replica's bookkeeping
                # with every other. A review caught the comment claiming these
                # are never mutated in place while this line did exactly that.
                self._runs_pushed = {
                    **self._runs_pushed,
                    session_id: _json.dumps(record, sort_keys=True),
                }
                return True
            except training.InvalidSessionId:
                # Not retryable and not this method's business to repair: the
                # id never came from this server.
                logger.warning(
                    f"refusing to share a record for an id that is not a "
                    f"session (length {len(str(session_id))})"
                )
                return False
            except Exception as exc:
                if attempt == 1:
                    await asyncio.sleep(self.RUNS_RETRY_DELAY_S)
                    continue
                if log_failure:
                    logger.warning(
                        f"could not share the run record for {session_id}: "
                        f"{type(exc).__name__}: {str(exc)[:200]}. The run is "
                        "unaffected; the next sync retries."
                    )
                return False
        return False

    async def _list_run_record_files(self, am, store) -> tuple:
        """``(entries, complete)`` for a deployment's record directory.

        **A TUPLE, AND THAT IS THE POINT.** `list_files` is paginated: the
        server's own parameters are ``limit`` defaulting to **1000** with a
        ceiling of **10000**, plus an ``offset``, and it returns a bare list of
        ``{name, type, size, last_modified}``. There is nowhere in that answer
        for a completeness flag to live, so a single call is a PAGE and not a
        directory, and a caller handed a bare list cannot tell "all of them"
        from "as many as I could get". Returning a tuple makes a caller that
        ignores the second element fail loudly instead of quietly reading a
        subset.

        **The loop stops on an EMPTY page, never a SHORT one.** A page shorter
        than the limit asked for is an ordinary answer -- the server may cap the
        limit below the ask -- so treating "shorter than requested" as "that is
        all of them" reintroduces the incomplete read one layer down.

        **Progress is measured by NEW NAMES, not by page length.** A server that
        ignored ``offset`` would serve page one forever; that terminates the
        loop with ``complete=False`` and a warning rather than spinning. The
        check counts names not seen before across ALL entries, including
        directories and non-record files, because a page made entirely of those
        is progress even though it contributes no records.

        Deliberately NOT bounded by an arbitrary maximum: a cap here would be
        the very defect this removes. The directory holds one small document per
        run.
        """
        import training

        dir_path = training.run_record_dir(store)
        entries: Dict[str, tuple] = {}
        seen: set = set()
        offset = 0
        while True:
            page = await am.list_files(
                artifact_id=training.RUNS_ARTIFACT_ID,
                dir_path=dir_path,
                limit=self.RUNS_LIST_PAGE,
                offset=offset,
            )
            page = page or []
            if not page:
                return entries, True
            fresh = 0
            for e in page:
                if not isinstance(e, dict):
                    continue
                name = e.get("name")
                if not isinstance(name, str):
                    continue
                if name not in seen:
                    seen.add(name)
                    fresh += 1
                if e.get("type") == "directory" or not name.endswith(".json"):
                    continue
                entries[name] = (e.get("size"), e.get("last_modified"))
            if not fresh:
                logger.warning(
                    f"the run record listing stopped making progress at offset "
                    f"{offset} ({len(seen)} name(s) seen); the store may be "
                    "ignoring `offset`, so this listing is NOT known to be "
                    "complete"
                )
                return entries, False
            offset += len(page)

    async def _read_run_records(
        self, *, max_age_s: Optional[float] = None
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Every shared record for THIS deployment, keyed by session id.

        None and `{}` are different answers and the caller treats them
        differently: `{}` is "the shared history is empty", None is "the shared
        history could not be read", and only the second one is a reason to say
        so in the log while still answering from local disk.

        CACHED, AND THE CACHE IS WHY THIS IS NOT A PAGE-POLL PROBLEM. The first
        version re-fetched every record on every `list_training_sessions`: one
        `get_file` RPC plus one HTTP GET per run, serially, which a review put
        at about 520 round trips per poll at the 261 runs production already
        holds. The listing now serves a cached copy, refreshed at most once per
        `max_age_s`, and a refresh only downloads the records whose `size` or
        `last_modified` changed -- both of which the artifact listing returns
        (measured on the live artifact: `{"type": "file", "name": ...,
        "size": 41, "last_modified": 1795649029.585}`). A steady-state refresh
        is therefore ONE `list_files` call.

        The change test is size-and-mtime, which cannot see a rewrite that
        preserved both. Every write here goes through `put_file`, which sets a
        fresh mtime, so the case needs two writes inside one mtime tick with an
        identical length; the next changed write repairs it, and the
        alternative -- re-downloading everything every time -- is the cost this
        exists to avoid.

        A record that will not parse is COUNTED and skipped rather than
        dropping the whole read: one bad file must not empty a run history.
        """
        import json as _json
        import training

        if max_age_s is None:
            max_age_s = self.RUNS_CACHE_TTL_S
        now = time.time()
        if self._runs_cache is not None and (now - self._runs_cache_at) <= max_age_s:
            return self._runs_cache

        store = training.record_store_id()
        if store is None:
            logger.warning(
                f"{training.APPLICATION_ID_ENV} is not set on this replica, so it "
                "cannot tell which deployment's run history is its own; the "
                "listing answers from this instance's own disk only"
            )
            return None

        try:
            am = await self._runs_artifact_manager()
            entries, listing_complete = await self._list_run_record_files(am, store)
        except Exception as exc:
            logger.warning(
                f"the shared run history could not be listed "
                f"({type(exc).__name__}: {str(exc)[:160]}); answering from this "
                "instance's own disk only, which is incomplete"
            )
            return None

        cached = dict(self._runs_cache or {})
        meta = dict(self._runs_meta)
        # A record removed from the artifact leaves the cache with it, so a
        # deleted run does not live on in this replica's answer.
        #
        # ONLY WHEN THE LISTING IS KNOWN WHOLE. Absence from an INCOMPLETE
        # listing is not evidence of deletion, and this is the sharper half of
        # the paging defect: a truncated page would not merely shorten the
        # answer, it would evict every already-known run that fell past the
        # page and drop it from the history this replica serves. Keeping the
        # cache is the safe direction -- a run that really was deleted lingers
        # until a listing that can be trusted says so.
        if listing_complete:
            for gone in set(meta) - set(entries):
                meta.pop(gone, None)
                cached.pop(gone[: -len(".json")], None)
        else:
            logger.warning(
                "the shared run history listing could not be proven complete, "
                "so no record is being evicted from the cache; the answer may "
                "omit runs this replica has never seen"
            )

        unreadable = fetched = 0
        for name, stamp in entries.items():
            session_id = name[: -len(".json")]
            if meta.get(name) == stamp and session_id in cached:
                continue
            try:
                url = await am.get_file(
                    artifact_id=training.RUNS_ARTIFACT_ID,
                    file_path=f"{training.run_record_dir(store)}/{name}",
                )
                resp = await self._http.get(url)
                resp.raise_for_status()
                record = _json.loads(resp.text)
                key = (record or {}).get("session_id") or session_id
                cached[key] = record
                meta[name] = stamp
                fetched += 1
            except Exception:
                unreadable += 1
        if unreadable:
            logger.warning(
                f"{unreadable} shared run record(s) could not be read; the rest "
                f"({len(cached)}) are listed"
            )
        self._runs_cache = cached
        self._runs_meta = meta
        self._runs_cache_at = now
        if fetched:
            logger.info(
                f"shared run history: {len(cached)} record(s), {fetched} refreshed"
            )
        return cached

    async def _sync_run_records(self) -> Dict[str, int]:
        """Push the records whose content has changed since the last push.

        Content-addressed rather than time-addressed: a record is pushed when
        its JSON differs from what this instance last pushed. A poll that
        re-pushed everything would rewrite 261 files a minute, and one that
        pushed only on a status TRANSITION would miss a per-epoch metric, which
        is most of what a run history is for.
        """
        import json as _json
        import training

        pushed = skipped = failed = 0
        root = training.sessions_root()
        for d in sorted(root.iterdir()) if root.exists() else []:
            if not (d.is_dir() and (d / "status.json").exists()):
                continue
            if not training.SESSION_ID_PATTERN.match(d.name):
                continue
            try:
                record = training.run_record(d.name)
            except Exception:
                failed += 1
                continue
            digest = _json.dumps(record, sort_keys=True)
            if self._runs_pushed.get(d.name) == digest:
                skipped += 1
                continue
            if await self._persist_run_record(d.name):
                pushed += 1
            else:
                failed += 1
        return {"pushed": pushed, "unchanged": skipped, "failed": failed}

    async def _runs_sync_loop(self) -> None:
        """The poll clause 3 is measured against.

        Deliberately never raises out: a loop that dies takes the shared
        history with it silently, and the instance would keep serving its own
        disk as if that were the whole story.
        """
        while True:
            try:
                await asyncio.sleep(self.RUNS_SYNC_INTERVAL_S)
                counts = await self._sync_run_records()
                # Pull as well as push, so a run started on another replica is
                # in this one's cache within a poll of being shared -- the
                # clause this interval is measured against.
                await self._read_run_records(max_age_s=0)
                if counts["pushed"] or counts["failed"]:
                    logger.info(f"shared run records: {counts}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    f"a run-record sync pass failed ({type(exc).__name__}: "
                    f"{str(exc)[:160]}); the next pass retries"
                )

    @bioengine.method
    async def get_training_capabilities(self) -> Dict[str, Any]:
        """Which models this worker's GPU(s) can fine-tune. The annotate-page
        finetuning selector calls this to offer every model and grey out the ones
        the hardware can't train. Returns VRAM per backend and, for all model
        types, ``{model_type, backend, min_gpu_memory_mb, family, size, trainable,
        reason, source}``. ``source`` is ``declared`` (deploy-time config, answered
        CPU-side without waking a GPU), ``detected`` (live runtime probe),
        ``unavailable`` (backend runtime down → ``trainable`` is None), or
        ``disabled`` (backend not deployed on this worker → ``trainable`` is
        False).

        Also returns ``placement``, which says whether this cluster has enough
        GPUs for the backends that are enabled — each GPU runtime books a whole
        device, so two backends need two GPUs. When it does not,
        ``placement.sufficient`` is False and ``placement.advice`` names
        ``MODEL_FINETUNE_BACKENDS``; without it the only symptom is a backend
        reading ``unavailable`` forever while the deployment reports
        ``UPSCALING``, which never times out."""
        import training

        # THE RUN STORE, per backend. A runtime replica has its own HOME, so a
        # backend can have a perfectly good GPU and still be unable to write a
        # single run record -- which is what happened in production while this
        # method answered `trainable: True, reason: "fits"`. A review called
        # that the status half of this row's clause 1, and it was right: the
        # refusal at `start_training` tells a user who has already committed to
        # a run, while the page reads THIS.
        #
        # EVERY VIEW IS GATHERED CONCURRENTLY, and the first version of this
        # awaited them one after another. A review measured what that cost: with
        # VRAM declared, both backends enabled and one runtime whose `ping`
        # hangs, the call took 10.01 s and still returned no store verdict. The
        # declared path exists precisely to answer CPU-side without waking a
        # GPU, and serialising four awaits turned the page's poll into the sum
        # of every timeout. Concurrently the worst case is the LONGEST one, not
        # the sum -- and with VRAM declared the GPU views cost nothing at all,
        # so what is left is a single bounded store probe.
        enabled = [b for b in ("microsam", "cellpose") if b in self._enabled_backends]

        async def _nothing() -> Dict[str, Any]:
            return {}

        gpu_order = ("microsam", "cellpose")
        views = await asyncio.gather(
            *(
                self._gpu_view(b) if b in self._enabled_backends else _nothing()
                for b in gpu_order
            ),
            *(self._backend_store_view(b) for b in enabled),
        )
        microsam, cellpose = views[0], views[1]
        stores = dict(zip(enabled, views[len(gpu_order) :]))

        models = []
        for m in ModelType.__args__:
            backend = "cellpose" if m in CELLPOSE_MODEL_TYPES else "microsam"
            enabled = backend in self._enabled_backends
            gpu = cellpose if backend == "cellpose" else microsam
            store = stores.get(backend) or {}
            required = training.min_training_vram_mb(m)
            rec = {
                "model_type": m,
                "backend": backend,
                "min_gpu_memory_mb": required,
                "family": training.model_family(m),
                "size": training.model_size(m),
                "source": (
                    "disabled"
                    if not enabled
                    else (gpu.get("source", "unavailable") if gpu else "unavailable")
                ),
            }
            if not enabled:
                rec["trainable"], rec["reason"] = (
                    False,
                    "backend not deployed on this worker",
                )
            elif not gpu:
                rec["trainable"], rec["reason"] = None, "runtime unavailable"
            elif not gpu.get("available"):
                rec["trainable"], rec["reason"] = (
                    False,
                    "no GPU detected on the runtime",
                )
            elif store.get("writable") is False:
                # Before the VRAM comparison on purpose: a replica that cannot
                # write its run store cannot train whatever its GPU has, and
                # reporting the VRAM reason would send the user to buy a bigger
                # card for a read-only filesystem.
                rec["trainable"] = False
                rec["reason"] = str(
                    store.get("detail") or "the runtime cannot write its run store"
                )
            elif gpu["total_mb"] < required:
                rec["trainable"] = False
                rec["reason"] = f"needs ~{required} MB, GPU has {gpu['total_mb']} MB"
            else:
                rec["trainable"], rec["reason"] = True, "fits"
            models.append(rec)
        return {
            "gpus": {
                "microsam": (
                    microsam
                    if "microsam" in self._enabled_backends
                    else {"enabled": False}
                ),
                "cellpose": (
                    cellpose
                    if "cellpose" in self._enabled_backends
                    else {"enabled": False}
                ),
            },
            "models": models,
            # The per-backend store verdict as DATA, not only folded into a
            # model's reason string: an operator wants to see which backend
            # cannot write, and a UI should not have to parse prose for it.
            # `writable: None` means the runtime could not be asked.
            "stores": {
                b: (
                    stores.get(b) or {"writable": None, "detail": "not reported"}
                    if b in self._enabled_backends
                    else {"enabled": False}
                )
                for b in ("microsam", "cellpose")
            },
            "parameters": self._training_parameter_schema(),
            # Why a backend reads 'runtime unavailable' when nothing is broken: on
            # a cluster with fewer GPUs than enabled backends the runtime never
            # schedules, and the only other symptom is a permanent UPSCALING that
            # never times out. Answered CPU-side; never wakes a GPU.
            "placement": _gpu_placement_report(self._enabled_backends),
        }

    def _training_parameter_schema(self) -> List[Dict[str, Any]]:
        """Per-parameter schema for the start_training controls, derived from
        ``start_training.__schema__`` so it can't drift from the actual signature.
        Emits ``{name, applies_to, type, default, min, max, description}`` for
        each tunable in ``TRAINING_PARAM_BACKENDS``; ``min``/``max`` are the
        inclusive/exclusive Field bounds, ``applies_to`` the backends it affects."""
        props = self.start_training.__schema__["parameters"]["properties"]
        out = []
        for name, backends in TRAINING_PARAM_BACKENDS.items():
            spec = props.get(name, {})
            # Optional params surface as anyOf([<real>, {"type": "null"}]).
            typed = spec
            if "anyOf" in spec:
                typed = next((s for s in spec["anyOf"] if s.get("type") != "null"), {})
            lo = typed.get("minimum", typed.get("exclusiveMinimum"))
            hi = typed.get("maximum", typed.get("exclusiveMaximum"))
            out.append(
                {
                    "name": name,
                    "applies_to": backends,
                    "type": typed.get("type"),
                    "default": spec.get("default"),
                    "min": lo,
                    "max": hi,
                    "description": spec.get("description", ""),
                }
            )
        return out

    @bioengine.method(context=True)
    async def start_training(
        self,
        train_images: List[Union[np.ndarray, str, Dict[str, Any]]] = Field(
            ...,
            description="Training images: arrays, http(s) URLs, get_upload_url paths, "
            "or an OME-Zarr region reference {kind: 'ngff', store_root, region: "
            "{x,y,width,height}, dataset_paths, axis_roles, level0_shape} — the last "
            "reads only that region's chunks from the source, never the whole image.",
        ),
        train_labels: List[Union[np.ndarray, str]] = Field(
            ...,
            description="Dense instance-label masks paired 1:1 with train_images "
            "(.tif/.png/.npy, or a .geojson polygon FeatureCollection). AIS fine-tuning "
            "needs DENSE labels — annotate all objects per image.",
        ),
        val_images: Optional[List[Union[np.ndarray, str, Dict[str, Any]]]] = Field(
            None, description="Optional validation images; same forms as train_images."
        ),
        val_labels: Optional[List[Union[np.ndarray, str]]] = Field(
            None, description="Validation labels."
        ),
        model_type: ModelType = Field(
            "vit_l_lm",
            description="Base model to fine-tune. vit_* → micro-sam (AIS decoder); "
            "'cpsam' → Cellpose-SAM, 'cpdino'/'cpdino-vitb' → Cellpose-DINO "
            "(all cellpose types share the isolated Cellpose runtime).",
        ),
        n_epochs: int = Field(5, ge=1, description="Number of training epochs."),
        n_objects_per_batch: int = Field(
            8,
            ge=1,
            description="micro-sam only: objects per batch — main GPU-memory knob; "
            "lower it on small GPUs, raise it on large ones (8 fits vit_b on 24GB).",
        ),
        patch_size: int = Field(
            512,
            ge=16,
            description="micro-sam only: square training patch side (clamped to the smallest image).",
        ),
        diam_mean: float = Field(
            30.0,
            gt=0,
            description="cellpose only (cpsam/cpdino): mean object diameter in pixels.",
        ),
        train_tile_size: Optional[int] = Field(
            None,
            ge=16,
            description="cellpose only: square training tile side (cellpose "
            "'bsize'). Omit for the backbone's own default — 256 for cpsam, 384 for "
            "cpdino. A smaller tile trains faster and in less VRAM, at some cost in "
            "accuracy: measured on synthetic narrow-object data, 256 matched the 384 "
            "default to within noise while 128 scored below it, so lower this when "
            "training cost is the constraint rather than to improve a narrow-data fit. "
            "cpsam accepts 256 and nothing else, so this is settable for "
            "cpdino/cpdino-vitb only.",
        ),
        batch_size: int = Field(1, ge=1, description="Training batch size."),
        learning_rate: float = Field(1e-5, gt=0, description="AdamW learning rate."),
        val_fraction: float = Field(
            0.2, ge=0.0, le=0.9, description="Val split fraction when val is omitted."
        ),
        n_samples: Optional[int] = Field(
            None, ge=1, description="Patches sampled per epoch (auto if omitted)."
        ),
        resume_session_id: Optional[str] = Field(
            None,
            description="Continue fine-tuning from a prior session's checkpoint (same model_type).",
        ),
        dataset_artifact_id: Optional[str] = Field(
            None,
            description="The dataset artifact this run is trained on, e.g. "
            "'my-workspace/my-dataset'. Recorded on the run, and it decides who may "
            "COLLABORATE on it: the run's owner always, plus anyone who owns or manages "
            "this dataset, so a colleague can list the run and continue training from its "
            "checkpoint. It does NOT widen stopping, exporting or contributing, which stay "
            "with the run's owner and `bioimage-io` administrators. You must own or manage "
            "the dataset you name — naming one you do not is refused, so a run cannot be "
            "placed into another team's listing. Omit it for a private run with no "
            "collaborators.",
        ),
        init_checkpoint: Optional[str] = Field(
            None,
            description="Start fine-tuning from an existing checkpoint given as an "
            "http(s) URL (or a get_upload_url path) — e.g. the resumable weights file of an "
            "exported BioImage.IO draft (see get_export_status.resume_checkpoint_file) — instead "
            "of the base model. Mutually exclusive with resume_session_id; model_type must match "
            "the checkpoint's architecture.",
        ),
        init_from_pool: Optional[str] = Field(
            None,
            description="Seed this fine-tune from a community model-soup pool's CURRENT "
            "checkpoint (artifact id, e.g. 'bioimage-io/cpsam-community-pool'). Resolves the "
            "pool's latest community weights and trains from them — the at-your-own-pace round "
            "start. Requires pool_token; mutually exclusive with init_checkpoint and "
            "resume_session_id; the pool's model_type must match.",
        ),
        init_from_pool_version: Optional[str] = Field(
            None,
            description="Seed from a NAMED community version ('v1', 'v2', …) instead of the "
            "pool's current pointer. Resolves the version's own manifest.json rather than the "
            "index, so it still reaches a version whose weights were published but whose index "
            "entry was never committed (an aggregate interrupted between its two commits). "
            "Requires init_from_pool.",
        ),
        pool_token: Optional[str] = Field(
            None,
            description="Token with read access to init_from_pool's artifact. Scoped to "
            "the pool, not the app's own token.",
        ),
        metrics: Optional[List[str]] = Field(
            None,
            description="Which quantitative metrics to compute, chosen from the fixed "
            "set ['loss', 'pixel', 'instance'] — an unrecognised name is rejected, not "
            "ignored. 'loss': per-epoch train/validation loss curves, free, the default. "
            "'pixel': foreground/background accuracy, precision, recall, f1 and iou on the "
            "validation split. 'instance': instance AP and F1 at IoU 0.5/0.75/0.9 — a full "
            "instance-segmentation eval, the expensive one, never computed unless asked. "
            "'pixel' and 'instance' score the finished model on held-out data and share one "
            "inference pass. Results surface in get_training_status. Cellpose backend only; "
            "defaults to ['loss'].",
        ),
        label: str = Field(
            "", description="Optional human-readable tag for this session."
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Start a μSAM fine-tuning session (AIS decoder) and return immediately
        with the status incl. ``session_id``. Data prep runs on this CPU entry;
        the training itself runs in a subprocess on a GPU runtime replica. Poll
        ``get_training_status`` and, once ``checkpoint_available``, serve via
        ``infer(session_id=...)``.
        """
        import training

        if (
            sum(
                x is not None and x != ""
                for x in (init_checkpoint, resume_session_id, init_from_pool)
            )
            > 1
        ):
            raise ValueError(
                "Pass at most one of init_checkpoint, resume_session_id, or init_from_pool — "
                "they select different starting checkpoints."
            )

        # A claimed dataset is VERIFIED before it is recorded. Without this a
        # caller could name any dataset id and have their run appear in that
        # team's listing — not a privilege escalation, since the run is their
        # own, but a way to put arbitrary content in front of people who never
        # agreed to see it. Checked before the session directory exists, so a
        # refused claim leaves nothing behind.
        if dataset_artifact_id:
            if not await self._caller_managed_datasets([dataset_artifact_id], context):
                raise training.NotTheOwner(
                    "dataset_artifact_id names a dataset you do not own or manage; "
                    "a run can only be attached to a dataset you manage"
                )

        # Resuming reads ANOTHER RUN'S CHECKPOINT and continues from it, so the
        # resulting model DERIVES from weights the caller did not train. That is
        # why it is gated at all despite changing nothing on the other run.
        #
        # It is gated by ACCESS, not by ownership: the run's owner, a MANAGER OF
        # ITS DATASET, or an administrator. Wider than stop/export/contribute on
        # purpose — continuing a colleague's training is what a shared dataset is
        # for, while disposing of their run is not. Note the consequence for the
        # documented position: "you shared the id so you shared the history" is
        # still true, but a dataset's managers can now also build on each other's
        # weights, which the id alone never permitted.
        #
        # Checked BEFORE the session directory exists and before anything else,
        # so a refused resume leaves no PREPARING session behind and reveals
        # nothing about whether the named run exists.
        if resume_session_id:
            prior_dataset = training.read_status(resume_session_id).get(
                training.DATASET_FIELD
            )
            training.require_caller_may_resume_run(
                resume_session_id,
                context,
                writable_datasets=await self._caller_managed_datasets(
                    [prior_dataset] if prior_dataset else [], context
                ),
            )

        # Before the session dir exists, so a bad name fails the call outright
        # rather than leaving a PREPARING session behind.
        selected_metrics = training.validate_metrics(metrics)

        if init_from_pool_version and not init_from_pool:
            raise ValueError(
                "init_from_pool_version names a version inside a pool, so it requires "
                "init_from_pool."
            )

        # Refuse here rather than let cellpose raise it: train_seg checks this only
        # after data prep and after the run has taken the single GPU training slot.
        fixed_tile = CELLPOSE_FIXED_TILE_SIZE.get(model_type)
        if (
            train_tile_size is not None
            and fixed_tile is not None
            and train_tile_size != fixed_tile
        ):
            raise ValueError(
                f"{model_type} trains at a fixed tile size of {fixed_tile} — cellpose rejects "
                f"any other value. Omit train_tile_size, or use a cpdino model type, which "
                f"accepts it."
            )

        pool_provenance = None
        if init_from_pool:
            init_checkpoint, pool_provenance = await self._resolve_pool_init(
                model_type, init_from_pool, pool_token, init_from_pool_version
            )

        await self._check_runtime_available(model_type)
        # REFUSED AT THE DOOR if the runtime that would serve this cannot write
        # its run store. Before this, the request was accepted, queued, handed
        # to the runtime and died at the runtime's first status write -- the
        # user's reason was a Ray traceback. The verdict is the runtime's own
        # and is cached there, so this costs no write per request.
        await self._require_runtime_store_writable(model_type)

        # REFUSE A BASE CHECKPOINT THIS SITE CANNOT PRODUCE, before the GPU
        # slot. Same reasoning as the fixed-tile check above: without this the
        # first thing that consults the network about the base weights runs
        # INSIDE the training worker, after the run has been admitted and the
        # single GPU slot taken, so the caller waits for a slot and then gets a
        # download error minutes later.
        #
        # Only when the base model would actually be fetched. A caller
        # supplying their own starting weights -- init_checkpoint, a resume, or
        # a pool init, all of which have resolved into `init_checkpoint` by
        # now -- is not waiting on micro-sam's registry, and refusing them
        # would be refusing a run this site CAN do.
        #
        # micro-sam types only: cellpose does not resolve weights through that
        # registry, and asking the micro-sam runtime about `cpsam` would be
        # asking the wrong backend.
        if not init_checkpoint and model_type not in CELLPOSE_MODEL_TYPES:
            await self._refuse_unavailable_base_checkpoint(model_type)

        gpu = await self._runtime_gpu_memory(model_type)
        required = training.min_training_vram_mb(model_type)
        if gpu.get("available") and gpu["total_mb"] < required:
            same_backend = model_type in CELLPOSE_MODEL_TYPES
            fits = [
                m
                for m, req in training.MODEL_TRAIN_VRAM_MB.items()
                if (m in CELLPOSE_MODEL_TYPES) == same_backend
                and req <= gpu["total_mb"]
            ]
            raise ValueError(
                f"{model_type} needs ~{required} MB of GPU memory to fine-tune, but "
                f"this runtime's GPU ({gpu.get('device_name') or 'unknown'}) has only "
                f"{gpu['total_mb']} MB. Models that fit here: "
                f"{', '.join(fits) or 'none'}. Call get_training_capabilities() for "
                f"the full per-model list, or reduce n_objects_per_batch and retry a "
                f"smaller model."
            )

        backend = "cellpose" if model_type in CELLPOSE_MODEL_TYPES else "microsam"
        session_id = training.new_session_id()
        training.write_status(
            session_id,
            status="PREPARING",
            created_at=time.time(),
            # Who this run belongs to, from Hypha's injected context. Recorded
            # at creation because it cannot be recovered later: nothing else on
            # the record says whose work it was.
            **training.caller_identity(context),
            # Which dataset this run belongs to, and so who may collaborate on
            # it. Recorded at creation for the same reason as the owner: nothing
            # else on the record says it, and it is VERIFIED above, so what is
            # stored is a dataset the caller actually manages rather than a
            # claim. Absent when the caller named none, which leaves the run
            # private to its owner.
            **(
                {training.DATASET_FIELD: dataset_artifact_id}
                if dataset_artifact_id
                else {}
            ),
            model_type=model_type,
            backend=backend,
            label=label,
            n_train_inputs=len(train_images),
            resume_session_id=resume_session_id,
            init_checkpoint=init_checkpoint,
            pool_provenance=pool_provenance,
        )
        params = dict(
            n_epochs=n_epochs,
            n_objects_per_batch=n_objects_per_batch,
            batch_size=batch_size,
            learning_rate=learning_rate,
            diam_mean=diam_mean,
            val_fraction=val_fraction,
            n_samples=n_samples,
            resume_session_id=resume_session_id,
            init_checkpoint=init_checkpoint,
            pool_provenance=pool_provenance,
            train_tile_size=train_tile_size,
            patch_shape=(patch_size, patch_size),
            num_workers=0,
            metrics=selected_metrics,
        )
        task = asyncio.create_task(
            self._run_training(
                session_id,
                model_type,
                train_images,
                train_labels,
                val_images,
                val_labels,
                params,
            )
        )
        self._training_tasks[session_id] = task
        return training.get_status(session_id)

    @bioengine.method
    async def preflight_training_dataset(
        self,
        train_images: List[str] = Field(
            ...,
            description="Raw training image sources to ingest from an external "
            "store before a run: http(s) URLs (e.g. BioImage Archive FIRE files) or "
            "get_upload_url paths, one per training pair.",
        ),
        train_labels: List[str] = Field(
            ...,
            description="Instance-label mask sources paired 1:1 with train_images "
            "(.tif/.png/.npy, or a .geojson polygon FeatureCollection).",
        ),
        val_images: Optional[List[str]] = Field(
            None, description="Optional validation image sources."
        ),
        val_labels: Optional[List[str]] = Field(
            None,
            description="Optional validation label sources, paired with val_images.",
        ),
        enable_clahe: bool = Field(
            False,
            description="Apply CLAHE (contrast-limited adaptive histogram "
            "equalisation) to each image while staging — rescues low-contrast "
            "fluorescence Cellpose-SAM otherwise cannot segment. Off by default; the "
            "min-max rescale during training is unaffected. Forces staging on.",
        ),
        stage: bool = Field(
            True,
            description="Download, validate and RE-STAGE each prepared image+label "
            "into the app's temp store, returning URL lists ready to pass straight into "
            "start_training. If False, validate the sources in place and report only.",
        ),
    ) -> Dict[str, Any]:
        """Validate and ingest a training dataset from an external store before a run.

        Resolves each (image, label) source, checks the pair is usable (readable,
        matching H×W, a 2-D integer instance-label map with at least one object),
        optionally applies CLAHE, and — when ``stage`` is set (forced on by
        ``enable_clahe``) — re-stages the prepared arrays into the app's temp store so
        the returned ``train_images``/``train_labels`` URL lists feed straight into
        ``start_training``. Returns a per-pair report plus counts so a dataset can be
        vetted before committing to a long run. Staged URLs are short-lived — call
        start_training promptly.
        """
        if len(train_images) != len(train_labels):
            raise ValueError("train_images and train_labels must have equal length.")
        if (val_images is None) ^ (val_labels is None):
            raise ValueError("val_images and val_labels must be provided together.")
        if val_images is not None and len(val_images) != len(val_labels):
            raise ValueError("val_images and val_labels must have equal length.")

        do_stage = bool(stage or enable_clahe)

        async def _prepare(split, img_sources, lbl_sources):
            report, staged_imgs, staged_lbls = [], [], []
            for i, (img_src, lbl_src) in enumerate(zip(img_sources, lbl_sources)):
                item = {
                    "split": split,
                    "index": i,
                    "image": str(img_src),
                    "label": str(lbl_src),
                }
                try:
                    img = await self._resolve_image(img_src)
                    lbl = np.squeeze(
                        await self._resolve_label_array(lbl_src, img.shape)
                    )
                    img_hw = tuple(int(x) for x in img.shape[:2])
                    if lbl.ndim != 2:
                        raise ValueError(
                            f"label is not 2-D (shape {tuple(lbl.shape)})."
                        )
                    if tuple(int(x) for x in lbl.shape[:2]) != img_hw:
                        raise ValueError(
                            f"image/label shape mismatch: image H×W {img_hw} vs "
                            f"label {tuple(int(x) for x in lbl.shape)}."
                        )
                    if not np.issubdtype(lbl.dtype, np.integer):
                        lbl = lbl.astype(np.int64)
                    n_inst = int(np.unique(lbl[lbl > 0]).size)
                    if n_inst == 0:
                        raise ValueError("label map has no foreground objects.")
                    if enable_clahe:
                        img = _apply_clahe(img)
                    item.update(
                        ok=True,
                        image_shape=list(img.shape),
                        label_shape=list(lbl.shape),
                        n_instances=n_inst,
                    )
                    if do_stage:
                        staged_imgs.append(
                            await self._save_npy_to_temp(np.asarray(img))
                        )
                        staged_lbls.append(await self._save_npy_to_temp(lbl))
                except Exception as exc:
                    item.update(ok=False, error=str(exc))
                report.append(item)
            return report, staged_imgs, staged_lbls

        train_report, train_imgs, train_lbls = await _prepare(
            "train", train_images, train_labels
        )
        val_report, val_imgs, val_lbls = [], [], []
        if val_images is not None:
            val_report, val_imgs, val_lbls = await _prepare(
                "val", val_images, val_labels
            )

        pairs = train_report + val_report
        n_bad = sum(1 for p in pairs if not p["ok"])
        n_train_ok = sum(1 for p in train_report if p["ok"])
        ok = n_bad == 0 and n_train_ok > 0

        result = {
            "ok": ok,
            "train_pair_count": len(train_report),
            "train_pair_ok": n_train_ok,
            "val_pair_count": len(val_report),
            "clahe_applied": bool(enable_clahe),
            "staged": do_stage,
            "pairs": pairs,
        }
        if do_stage and ok:
            result["train_images"] = train_imgs
            result["train_labels"] = train_lbls
            if val_images is not None:
                result["val_images"] = val_imgs
                result["val_labels"] = val_lbls
        if ok:
            msg = f"Validated {n_train_ok} training pair(s)"
            if val_report:
                msg += (
                    f" and {sum(1 for p in val_report if p['ok'])} validation pair(s)"
                )
            result["message"] = (
                (msg + "; staged for start_training.") if do_stage else (msg + ".")
            )
        elif n_bad:
            result["message"] = (
                f"{n_bad} of {len(pairs)} pair(s) failed validation; see 'pairs' for "
                "per-pair errors."
            )
        else:
            result["message"] = "No usable training pairs resolved."
        return result

    @bioengine.method
    async def get_training_status(
        self,
        session_id: str = Field(..., description="Session id from start_training."),
    ) -> Dict[str, Any]:
        """Fine-tuning session status (PREPARING/QUEUED/TRAINING/COMPLETED/FAILED/
        STOPPED), elapsed time, and whether the checkpoint is servable. QUEUED
        means another fine-tuning holds the single training slot."""
        import training

        return training.get_status(session_id)

    @bioengine.method(context=True)
    async def migrate_run_records(
        self,
        apply: bool = Field(
            False,
            description=(
                "Write the records. Omitted or false performs a DRY RUN that "
                "writes nothing and returns the plan."
            ),
        ),
        confirm: str = Field(
            "",
            description=(
                "Must equal the number of records the dry run planned. Proves "
                "the caller read the plan rather than passing apply=true."
            ),
        ),
        verify_only: bool = Field(
            False,
            description=(
                "Check whether the records are ALREADY in the shared record and "
                "write nothing. For a caller whose apply was interrupted and who "
                "needs to know whether it landed. Note it re-plans from the "
                "worker's disk AS IT IS NOW, not from the interrupted run's plan, "
                "so a run that has progressed since will read as mismatched."
            ),
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Backfill this worker's on-disk runs into the shared record. ADMIN ONLY.

        DRY RUN BY DEFAULT. `apply` is the only thing that writes, and it
        refuses unless `confirm` matches the count the dry run reported, so a
        caller cannot write without having read the plan.

        THE STORE IS ENUMERATED BY THE FILESYSTEM, never by a listing. The
        per-caller listing returns what the CALLER may see -- measured at 269
        of 291 on the production store, because a run belonging to somebody
        else, or to an anonymous identity that no longer exists, is invisible
        to it by design. A backfill driven by it would silently skip exactly
        the runs most in need of being preserved, and `before == after` would
        still pass because both sides would be counting the same subset.

        EQUALITY IS BY ID SET AND CONTENT HASH, not by count. Two sets of the
        same size are not the same set, and a record that lands with the wrong
        content is not a record that landed. Both halves are enforced against
        the ARTIFACT after the write, not against this replica's own account of
        what it did: the records are listed and downloaded again, and `complete`
        requires every one of them to hash to what the plan said it would.

        Each record is built ONCE, in the plan, and that same document is what
        gets written -- so the hash in the plan is a hash of the bytes that were
        actually uploaded.

        A run that starts between the dry run and the apply changes the count,
        and the apply then REFUSES because `confirm` no longer matches. That is
        the safe direction: it costs a second dry run and cannot write a plan
        the caller has not seen.

        Nothing on disk is written, moved or deleted. The migration only adds
        documents to the runs artifact.
        """
        import hashlib
        import json as _json

        import training

        if not training.caller_is_workspace_admin(context):
            raise training.NotTheOwner(
                "migrating the run history is an administrator action: it reads "
                "every run on this worker, including runs that belong to other "
                "callers, and writes them into the shared record"
            )

        store = training.record_store_id()
        if not store:
            # Sharing is OFF for this replica, so it cannot tell which
            # deployment's history it would be writing into. Refusing is the
            # only safe answer: a fallback bucket shared by every such replica
            # is precisely the thing `record_store_id` exists to prevent.
            raise RuntimeError(
                "this replica cannot tell which deployment it belongs to, so it "
                "must not write into any deployment's run history"
            )

        names, skipped = training.run_names_on_disk()
        planned, unreadable = [], []
        for name in names:
            try:
                # THE RECORD IS BUILT EXACTLY AS THE WRITER BUILDS IT, and it is
                # kept so the write below uses this very document. An earlier
                # version hashed `run_record(name, get_status(...))` -- the
                # SHAPED status, which adds session_id, checkpoint_available,
                # config and a clock-derived elapsed_s, and has the stale sweep
                # applied -- while the write produced the RAW document. The two
                # disagreed on every run, and on a run that had gone stale they
                # disagreed on the status itself, so the plan attested to bytes
                # nobody ever wrote.
                # STRICT, and this is the one place that must be. The ordinary
                # reader answers `{}` for a status document that will not
                # parse, which is right for a listing -- one damaged run must
                # not take out the poll around it. Here it would copy a corrupt
                # run into the permanent record as an EMPTY one, replacing that
                # run's history with nothing and reporting success, and the
                # `unreadable` refusal below would never fire because nothing
                # ever raised.
                document = training.read_status_strict(name)
                record = training.run_record(name, document)
                path = training.run_record_path(name, store)
            except Exception as exc:
                unreadable.append(
                    {"id": name, "error": "%s: %s" % (type(exc).__name__, exc)}
                )
                continue
            body = _json.dumps(record).encode("utf-8")
            planned.append(
                {
                    "id": name,
                    "path": path,
                    "record": record,
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "owner": (record.get("status") or {}).get("owner"),
                }
            )

        plan = {
            "store": store,
            "on_disk": len(names),
            "skipped_names": skipped,
            "unreadable": unreadable,
            "planned": len(planned),
            "ids": sorted(p["id"] for p in planned),
            "hashes": {p["id"]: p["sha256"] for p in planned},
            "applied": 0,
            "dry_run": not apply,
        }

        plan["verify_only"] = bool(verify_only)
        if verify_only:
            # READ-ONLY third mode, and the reason it exists: after an apply is
            # interrupted -- a proxy timeout, a lost connection -- the caller has
            # no way to learn whether the records landed. The only answer
            # available was to list and hash the artifact from OUTSIDE the app,
            # which is exactly the work this operation exists to spare them, and
            # which is what had to be done once already.
            #
            # It runs the SAME read-back the apply runs -- against the artifact,
            # by id set and content hash -- and writes nothing: no stage, no
            # upload, no commit. `confirm` is not required, because there is
            # nothing to confirm.
            written, failed, put_unknown = [], [], []
            stage_error = commit_error = None
        elif not apply:
            return plan
        else:
            if confirm != str(len(planned)):
                raise ValueError(
                    f"confirm must be {len(planned)!r} — the number of records this "
                    f"run plans to write — and it was {confirm!r}. Run without "
                    f"apply first and read the plan."
                )
            if unreadable:
                raise RuntimeError(
                    f"{len(unreadable)} run(s) could not be read, so a complete "
                    f"backfill cannot be claimed; refusing rather than writing a "
                    f"partial history"
                )

            # ONE STAGE, MANY UPLOADS, ONE COMMIT.
            #
            # This loop used to call `_persist_run_record` per record, and that
            # helper wraps `edit(stage=True)` ... `commit()` around a SINGLE file.
            # That is right for its other caller -- one run finishing on its own --
            # and ruinous here: a 313-record backfill became 313 artifact commits,
            # about 1565 round trips taken serially, which is what exceeded the
            # proxy's 600 s ceiling and left this operation with no success path at
            # production scale. The write itself was never the problem; the number
            # of round trips was.
            #
            # `_persist_run_record` is deliberately NOT changed. The bulk shape
            # belongs here, where the records are known to be independent of one
            # another and a single stage/commit pair covers all of them.
            written, failed, put_unknown = [], [], []
            am = await self._runs_artifact_manager()

            # NEITHER THE STAGE NOR THE COMMIT MAY RAISE OUT OF HERE.
            #
            # The stage flag is ARTIFACT-LEVEL, and another instance's commit clears
            # it -- `_persist_run_record`'s docstring documents that race, and this
            # replica's own `_runs_sync_loop` is a concurrent stage/commit writer
            # every `RUNS_SYNC_INTERVAL_S`. Holding the flag open for the whole
            # upload phase instead of one file makes that window much wider, so the
            # race is likelier here, not rarer.
            #
            # An exception escaping this method is indistinguishable, from the
            # caller's side, from "the write did not happen" -- which is the exact
            # harm this operation exists to avoid, and a regression against the
            # per-record path, which retried and still returned a report. So both
            # are retried like that path, their failure is RECORDED, and the
            # read-back below still runs so the caller gets a report either way.
            async def _retrying(label: str, op) -> Optional[str]:
                err = None
                for attempt in (1, 2):
                    try:
                        await op()
                        return None
                    except Exception as exc:
                        err = "%s: %s" % (type(exc).__name__, str(exc)[:200])
                        if attempt == 1:
                            await asyncio.sleep(self.RUNS_RETRY_DELAY_S)
                logger.warning(
                    f"model-finetune: backfill could not {label} the runs artifact: {err}"
                )
                return err

            stage_error = await _retrying(
                "stage",
                lambda: am.edit(artifact_id=training.RUNS_ARTIFACT_ID, stage=True),
            )
            upload_sem = asyncio.Semaphore(max(1, int(self.RUNS_MIGRATE_CONCURRENCY)))

            async def _upload_one(entry: Dict[str, Any]) -> tuple:
                # The PLANNED record, not a freshly built one, and serialised the
                # same way the plan hashed it: the plan's sha256 is of exactly
                # these bytes, so sending anything else would make the plan a
                # statement about a document that was never uploaded.
                body = _json.dumps(entry["record"]).encode("utf-8")
                async with upload_sem:
                    try:
                        url = await am.put_file(
                            training.RUNS_ARTIFACT_ID, file_path=entry["path"]
                        )
                        resp = await self._http.put(url, content=body)
                        resp.raise_for_status()
                    except Exception as exc:
                        logger.warning(
                            f"model-finetune: backfill could not upload the record "
                            f"for '{entry['id']}': {type(exc).__name__}: {exc}"
                        )
                        return entry["id"], False
                    return entry["id"], True

            if stage_error is None:
                # gather preserves argument order, so `written`/`failed` are built
                # deterministically rather than in completion order.
                # A PUT'S OWN VERDICT IS *UNKNOWN*, NOT AUTHORITATIVE.
                #
                # Measured on a 320-record store at concurrency 8: 30 PUTs
                # raised, and 23 of those records were in the artifact
                # afterwards while 7 were not. So reading a raised PUT as "this
                # record is missing" would condemn records that landed, and
                # reading a returned PUT as "landed" is the very error the
                # read-back exists to catch. Both outcomes are recorded for the
                # report and NEITHER decides.
                for sid, ok in await asyncio.gather(*(_upload_one(i) for i in planned)):
                    (written if ok else put_unknown).append(sid)
            else:
                # The artifact was never staged, so nothing could be uploaded. Every
                # planned id is a failure, which is the truth and is reported rather
                # than raised.
                failed = [p["id"] for p in planned]

            # Committed ONCE, and after every upload has been awaited. A commit
            # taken while uploads were still in flight would publish a version the
            # read-back below then compares against an incomplete set.
            # WHAT THIS RETRY DOES AND DOES NOT COVER, because a review found the
            # comment claiming more than the code delivers.
            #
            # It retries the commit ALONE, so it covers a TRANSIENT commit failure
            # and nothing else. It cannot win the race named in
            # `_persist_run_record`'s docstring: there, another writer's commit
            # clears the ARTIFACT-LEVEL stage flag, and that flag stays cleared
            # until somebody re-stages -- so a second `commit` fails for exactly
            # the reason the first did. The per-record path survives that race
            # because its retry repeats `edit(stage=True)` as well.
            #
            # Re-staging here would mean calling `edit(stage=True)` AFTER the
            # uploads, and whether that preserves or discards a staged upload is
            # NOT measured -- every other caller stages before uploading, so
            # nothing in this file establishes it. Doing it on an assumption could
            # throw away the whole upload phase to save a commit. So the narrow
            # retry stays, and the behaviour under a cleared flag is: fail closed,
            # record `commit_error`, and still report (there is an arm for it).
            commit_error = None
            if stage_error is None:
                commit_error = await _retrying(
                    "commit",
                    lambda: am.commit(artifact_id=training.RUNS_ARTIFACT_ID),
                )
        plan["applied"] = len(written)
        plan["failed"] = sorted(failed)
        plan["stage_error"] = stage_error
        plan["commit_error"] = commit_error

        # THE ARTIFACT IS READ BACK. `written` only means "our commit returned"
        # -- it is this replica's account of its own actions, and a backfill
        # that grades itself on that has not checked anything. What the clause
        # asks is whether the ARTIFACT holds the records, so the artifact is
        # asked, and the bytes it serves are hashed and compared to the plan.
        if not verify_only and stage_error is None:
            # Short settle before asking what landed. Not load-bearing -- the
            # repair pass handles a record that is merely late -- but cheap, and
            # it keeps the repair for records that are actually wrong.
            await asyncio.sleep(max(0.0, float(self.RUNS_READBACK_SETTLE_S)))

        verified, verify_error = {}, None
        try:
            am = await self._runs_artifact_manager()
            # PAGED, AND THE COMPLETENESS IS CARRIED, NOT INFERRED. A read-back
            # that stopped at the first page would compare the plan against a
            # SUBSET of the artifact. That direction happens to fail closed
            # here -- an unseen id has no hash and lands in `mismatched` -- but
            # it fails closed while blaming the records, reporting as corrupt a
            # set of documents that are in fact present and correct. Saying
            # instead that the READ could not be proven whole is the same
            # refusal with the true reason attached.
            listing, listing_complete = await self._list_run_record_files(am, store)
            if not listing_complete:
                raise RuntimeError(
                    "the read-back listing could not be proven complete, so the "
                    "artifact's contents cannot be compared against the plan"
                )
            # Filtered BEFORE fetching, not inside the loop: a record that
            # belongs to another deployment or to an older run is not part of
            # this plan and is not evidence either way about it, so there is no
            # reason to spend a round trip on it.
            targets = [f for f in listing if f[: -len(".json")] in plan["hashes"]]
            verify_sem = asyncio.Semaphore(max(1, int(self.RUNS_MIGRATE_CONCURRENCY)))

            async def _fetch_one(fname: str) -> tuple:
                sid = fname[: -len(".json")]
                async with verify_sem:
                    url = await am.get_file(
                        artifact_id=training.RUNS_ARTIFACT_ID,
                        file_path=f"{training.run_record_dir(store)}/{fname}",
                    )
                    resp = await self._http.get(url)
                    resp.raise_for_status()
                    return sid, hashlib.sha256(resp.text.encode("utf-8")).hexdigest()

            # A raising fetch propagates out of gather and is caught by the
            # handler below, which records it as `verify_error` -- the same
            # behaviour the serial loop had. It must NOT be swallowed per
            # record: a read-back that quietly skipped a record would leave its
            # id absent from `verified`, and the mismatch list would then blame
            # the record instead of the read.
            for sid, digest in await asyncio.gather(*(_fetch_one(f) for f in targets)):
                verified[sid] = digest
        except Exception as exc:
            verify_error = "%s: %s" % (type(exc).__name__, str(exc)[:200])

        plan["verified"] = len(verified)
        plan["verify_error"] = verify_error
        plan["mismatched"] = sorted(
            sid for sid, h in plan["hashes"].items() if verified.get(sid) != h
        )
        plan["put_unknown"] = sorted(put_unknown)
        # Always present, so a caller can read it without knowing whether a
        # repair pass ran. An absent key and an empty one are indistinguishable
        # through `.get()`, and a caller that cannot tell them apart will read
        # "no read failures" out of a report that never looked.
        plan["reread_errors"] = []
        plan["reread_error_detail"] = {}

        # REPAIR PASS: re-upload ONLY what the read-back condemns.
        #
        # The read-back is the only judge of what landed, so it is also the only
        # thing that can say what still needs writing. Retrying everything would
        # re-upload hundreds of records that are already correct -- which is the
        # cost this whole change exists to remove -- and retrying the PUTs that
        # raised would retry the wrong set: measured at concurrency 8, 30 raised
        # and only 7 of those were actually absent, while a PUT that returned
        # cleanly can still have landed nothing.
        #
        # One pass, not a loop: if a record cannot be written twice under a
        # bound of 4, that is a condition to report rather than to grind at.
        plan["repaired"] = []
        if (
            not verify_only
            and plan["mismatched"]
            and stage_error is None
            and commit_error is None
            and verify_error is None
        ):
            bad = list(plan["mismatched"])
            by_id = {e["id"]: e for e in planned}
            retry = [by_id[sid] for sid in bad if sid in by_id]
            logger.info(
                f"model-finetune: backfill repairing {len(retry)} record(s) the "
                f"read-back found missing or mismatched"
            )
            repair_error = await _retrying(
                "re-stage for the repair pass",
                lambda: am.edit(artifact_id=training.RUNS_ARTIFACT_ID, stage=True),
            )
            if repair_error is None:
                await asyncio.gather(*(_upload_one(e) for e in retry))
                repair_error = await _retrying(
                    "commit the repair pass",
                    lambda: am.commit(artifact_id=training.RUNS_ARTIFACT_ID),
                )
            plan["repair_error"] = repair_error

            # RE-READ the repaired ids, because the repair's own PUT verdicts are
            # no more authoritative than the first pass's were.
            #
            # PER ID, and an ABSENT record is "still missing" rather than a read
            # failure. A review measured the difference: a single `gather` over
            # every condemned id raised on the first failure and discarded
            # EVERY digest in the batch -- including those of records the repair
            # had just landed correctly. With 2 transient and 1 permanent
            # failure the report read `repaired: []` and condemned all three,
            # when the truth was two repaired and one unresolved. That is the
            # same misreport this whole row exists to remove, so it cannot be
            # allowed to live in the repair's own reporting.
            #
            # WHICH IDS ARE ABSENT IS DECIDED BY THE LISTING BELOW, not by a
            # 404 on the fetch. An earlier version of this comment said a
            # missing record "404s on its fetch", which stopped being how this
            # works: a record the listing contains and the fetch then fails on
            # -- 404 included, since it can only mean the record went away
            # mid-pass -- is an absence of KNOWLEDGE and goes to
            # `reread_errors`, not to `mismatched`.
            #
            # `repair_error` is reserved for exactly three things, and the code
            # sets it for exactly those: a failed stage, a failed commit, and a
            # listing that could not be proven complete -- a read that could not
            # run AT ALL. Two things it is deliberately NOT used for:
            #   * an individual record that is not there. That is not an error,
            #     it is the answer, and it belongs in `mismatched`.
            #   * an individual record that could not be READ. That is an
            #     absence of knowledge about one record, not a failure of the
            #     pass, and it belongs in `reread_errors`.
            # The earlier wording of this comment claimed the wider "or a read
            # that could not run", which read as covering a single failed fetch
            # and did not match what the code did with one.
            # ABSENCE IS DECIDED BY THE LISTING, NOT BY A FAILING FETCH, which
            # is how the first read-back already does it. Treating every raised
            # fetch as "the record is missing" blames the record for the read: a
            # record the repair DID land, whose re-read then fails on the
            # connection, was reported in `mismatched`/`failed` while the
            # artifact held it correctly, and a caller acting on that re-uploads
            # a record that is already right.
            #
            # Deciding it from the listing also keeps this honest across
            # environments: an absent file surfaces as a different exception type
            # depending on the HTTP client, so anything that sniffed the
            # exception would behave differently in a test than in production.
            # The listing answers the same way everywhere.
            reread_errors: Dict[str, str] = {}
            if repair_error is None:
                # WRAPPED, because this function REPORTS rather than raises.
                # The per-id gather uses `return_exceptions=True`, so before
                # this re-listing existed no read failure in the repair could
                # propagate. Adding an unwrapped await put the caller back to
                # receiving a traceback and NO report -- the one outcome this
                # whole operation exists to prevent, and a direct contradiction
                # of the rule stated further down ("Reported as False rather
                # than raised: the caller needs a verdict").
                relist_unusable = None
                relisting, relisting_complete = [], False
                try:
                    relisting, relisting_complete = await self._list_run_record_files(
                        am, store
                    )
                except Exception as exc:
                    relist_unusable = (
                        "the repair's re-read listing could not be read: %s: %s"
                        % (type(exc).__name__, str(exc)[:200])
                    )
                if relist_unusable is None and not relisting_complete:
                    relist_unusable = (
                        "the repair's re-read listing could not be proven "
                        "complete, so which records are absent cannot be said"
                    )
                if relist_unusable is not None:
                    # A listing that cannot be proven whole cannot say which
                    # records are absent, and GUESSING WOULD BLAME THEM -- so
                    # not one of the condemned ids is reported as mismatched
                    # here. Every one is unestablished, which is the true
                    # statement: the repair may well have landed them, and an
                    # earlier version of this code said exactly that in a
                    # comment and then blamed them anyway.
                    repair_error = relist_unusable
                    plan["repair_error"] = repair_error
                    for sid in bad:
                        reread_errors[sid] = relist_unusable
                else:
                    present = {f[: -len(".json")] for f in relisting}
                    missing = [sid for sid in bad if sid not in present]
                    for sid in missing:
                        # Left out of `verified`, so the comparison below reports
                        # it as still mismatched -- which is exactly true.
                        logger.info(
                            f"model-finetune: record '{sid}' is still absent "
                            f"from the artifact after the repair"
                        )
                    to_fetch = [sid for sid in bad if sid in present]
                    results = await asyncio.gather(
                        *(_fetch_one(f"{sid}.json") for sid in to_fetch),
                        return_exceptions=True,
                    )
                    for sid, outcome in zip(to_fetch, results):
                        if isinstance(outcome, BaseException):
                            # The listing said this record IS there, so a failed
                            # read is an absence of KNOWLEDGE about it, not
                            # knowledge that it is absent. A 404 here means it
                            # went away between the listing and the fetch, which
                            # also cannot be called established.
                            reread_errors[sid] = "%s: %s" % (
                                type(outcome).__name__,
                                str(outcome)[:200],
                            )
                            logger.warning(
                                f"model-finetune: record '{sid}' was listed but "
                                f"could not be read back, so its state is not "
                                f"established: {type(outcome).__name__}: "
                                f"{outcome}"
                            )
                            continue
                        got_sid, digest = outcome
                        verified[got_sid] = digest
            plan["reread_errors"] = sorted(reread_errors)
            plan["reread_error_detail"] = dict(sorted(reread_errors.items()))
            plan["verified"] = len(verified)
            # An id whose state could not be established is NOT mismatched and
            # NOT repaired. It is neither claim, and reporting it as either one
            # is the misreport this row exists to remove.
            unknown = set(reread_errors)
            plan["mismatched"] = sorted(
                sid
                for sid, h in plan["hashes"].items()
                if verified.get(sid) != h and sid not in unknown
            )
            plan["repaired"] = sorted(set(bad) - set(plan["mismatched"]) - unknown)

        # THE PUSHED BOOKKEEPING FOLLOWS THE READ-BACK, so it belongs here --
        # after the repair pass -- not next to the uploads. Keying it on what the
        # ARTIFACT holds is the whole point: recording only the ids whose PUT
        # returned leaves out the put_unknown ids that did land and the ids the
        # repair fixed, and the next sync then re-pushes them one commit each.
        #
        # Placing it before the read-back is also simply wrong: `verified` does not
        # exist yet there, which is how this was found -- 22 tests raised
        # UnboundLocalError rather than a wrong answer, which is the good direction.
        if not verify_only and stage_error is None and commit_error is None:
            pushed = dict(self._runs_pushed)
            by_id = {e["id"]: e["record"] for e in planned}
            for sid, digest in verified.items():
                if sid in by_id and digest == plan["hashes"].get(sid):
                    pushed[sid] = _json.dumps(by_id[sid], sort_keys=True)
            self._runs_pushed = pushed
        # COMPLETE IS THE CONJUNCTION, and every term is load-bearing: nothing
        # failed to write, the id set that landed is the id set that was
        # planned, the read-back itself succeeded, and every record the artifact
        # serves hashes to what the plan said it would. Dropping the last term
        # is what let a record land with the wrong content and still be called
        # complete.
        # "NOT LANDED" AND "PRESENT, BUT NOT PUT THERE BY THIS RUN" ARE
        # DIFFERENT, and a report that cannot distinguish them invites the wrong
        # repair: a caller seeing only `complete: False` re-runs, and re-uploading
        # 313 records to fix nothing is the cost of not saying so.
        #
        # Named for a WRITE error, not a commit error, because a review showed it
        # fires on a failed STAGE too -- and truthfully so. With the stage
        # refused nothing is uploaded, yet the artifact can still hold every
        # planned record from an EARLIER run, which is exactly what an
        # interrupted caller wants to know. The earlier name said "commit" and
        # the comment claimed another writer's commit had published our uploads;
        # neither is true in the stage case.
        #
        # `complete` stays False either way: this run did not establish that
        # state and must not claim it.
        plan["records_present_despite_write_error"] = bool(
            (stage_error or commit_error)
            and verify_error is None
            and not plan["mismatched"]
            and set(plan["ids"]) <= set(verified)
        )
        # `complete` MEANS SOMETHING DIFFERENT IN A READ than in a write, and
        # saying so explicitly is the point. In verify_only nothing was written,
        # so `set(written) == set(ids)` can never hold and the answer would have
        # been False however complete the artifact was -- which is a verify that
        # always says "no". For a read the question is only whether the artifact
        # holds every planned record, by the same id set and content hash.
        if verify_only:
            plan["complete"] = (
                verify_error is None
                and not plan["mismatched"]
                and set(plan["ids"]) <= set(verified)
            )
            return plan

        # THE READ-BACK IS THE ONLY JUDGE, so `complete` no longer consults
        # `written`. That term asserted "every PUT returned", and a PUT's verdict
        # has been measured unreliable in both directions -- it condemned 23
        # records that had landed and cleared none of the 7 that had not. What
        # remains is what the artifact itself says, plus the write-side errors
        # that mean this replica cannot vouch for the artifact's state at all.
        plan["failed"] = sorted(plan["mismatched"])
        plan["complete"] = (
            not plan["mismatched"]
            # A record whose state could not be ESTABLISHED is not a pass. It is
            # excluded from `mismatched` on purpose -- it is not known to be
            # wrong -- so without this term narrowing the blame would have
            # quietly turned an unknown into a success.
            and not plan.get("reread_errors")
            and verify_error is None
            and plan.get("repair_error") is None
            # A stage or commit that failed after its retry means the artifact
            # may not hold what was uploaded, so the run is not complete even if
            # every PUT returned. Reported as False rather than raised: the
            # caller needs a verdict, and an exception here is indistinguishable
            # from "the write never happened".
            and stage_error is None
            and commit_error is None
        )
        return plan

    @bioengine.method(context=True)
    async def list_training_sessions(self, context=None) -> Dict[str, Any]:
        """The fine-tuning runs YOU may see, keyed by session_id.

        Not every run on the worker: a run is returned to the caller whose
        identity it was recorded under, and a run carrying no owner only to a
        caller holding rw or admin on the `bioimage-io` workspace. An
        anonymous caller is a real identity here — Hypha mints a fresh one per
        connection — so it sees the runs it started on that connection and no
        others.

        A run also comes back to a MANAGER of the dataset it was trained on: a
        run's collaborators are its dataset's managers, so a colleague can pick
        up where another left off. That is listing and resuming only — stopping,
        exporting and contributing stay with the owner.

        `context=True` is load-bearing and not decoration: without it the proxy
        forwards no context, `training.list_sessions` reads no identity, and the
        honest consequence of that is an EMPTY listing for everybody. That is
        the safe direction, and it is also indistinguishable from "this worker
        has no runs", so the decorator is covered by a test rather than trusted.

        The runs come from the SHARED record, with this instance's own disk
        layered on top of it -- so a run started on another instance is listed
        here, and a run this instance started a moment ago is listed before the
        next sync has shared it. If the shared record cannot be read the
        listing still answers from local disk, which is incomplete and says so
        in the log: a run history that is short is worse than one that is
        slow, and both are better than one that raises.
        """
        import training

        # ONE walk of the store: the records and the dataset ids together.
        # Asking for them separately read every record twice over on top of the
        # read inside `get_status` -- 3 per run, so 783 for the 261 runs on the
        # production store.
        records, dataset_ids, skipped = training.scan_runs_for_listing()

        # The shared half. None and {} are DIFFERENT answers here: {} is "the
        # shared history is empty", None is "it could not be read", and only
        # the second is a reason to say so while still answering from local
        # disk. Collapsing them with `or {}` computes the distinction and then
        # throws it away, which leaves a silently short history -- the failure
        # this log exists to make visible.
        shared = await self._read_run_records()
        if shared is None:
            print(
                "[model-finetune] the shared run history could not be read; "
                "answering from this instance's disk only, so the listing is "
                "INCOMPLETE",
                flush=True,
            )
            shared = {}

        # The collision rule and the dataset union live in `merge_run_sources`,
        # not here, so that what a caller gets and what a test composes are the
        # same code.
        records, shared_ids, dataset_ids = training.merge_run_sources(
            records, dataset_ids, shared
        )

        writable = await self._caller_managed_datasets(dataset_ids, context)
        # `skipped` is passed on, not dropped: `list_sessions` logs the stray
        # directory names it was handed, and those names are caller-controlled,
        # so the log is their only trace.
        return training.list_sessions(
            context,
            writable_datasets=writable,
            records=records,
            shared_ids=shared_ids,
            skipped=skipped,
        )

    async def _caller_managed_datasets(self, dataset_ids, context) -> frozenset:
        """Which of *dataset_ids* this caller owns or manages.

        Resolved ONCE per listing and passed down, rather than per run: the
        permission check is an artifact-manager round trip and the listing is a
        synchronous loop over every run on disk. One RPC per run would be 261 of
        them per call on the production store, and could not be expressed inside
        a `def` anyway.

        **Fails closed per artifact, not per call.** A dataset that cannot be
        read — deleted, renamed, permissions withheld, the artifact-manager
        down — is simply not in the returned set, so its runs stay invisible to
        anyone but their owner and an administrator. One unreadable dataset must
        not decide the answer for the others, and an unreadable one must never
        read as permitted.

        Short-circuits to empty when there is nothing to ask about, which is the
        common case today: no run recorded a dataset before this feature, so a
        worker full of older runs makes no RPCs at all.
        """
        import training

        if not dataset_ids or not isinstance(context, dict):
            return frozenset()
        try:
            am = await self._hypha.get_service("public/artifact-manager")
        except Exception as exc:
            print(
                "[model-finetune] could not reach the artifact manager to resolve "
                f"dataset managers; no co-manager access this call ({exc})",
                flush=True,
            )
            return frozenset()
        managed = set()
        for artifact_id in dataset_ids:
            try:
                artifact = await am.read(artifact_id=artifact_id)
            except Exception:
                # Deliberately not logged per artifact: a caller can name a
                # dataset, so this is attacker-reachable and would be a log
                # amplifier. The closed outcome is the whole response.
                continue
            permissions = (artifact.get("config") or {}).get("permissions") or {}
            if training.caller_manages_dataset(permissions, context):
                managed.add(artifact_id)
        return frozenset(managed)

    @bioengine.method(context=True)
    async def list_active_training_runs(self, context=None) -> Dict[str, Any]:
        """Every UNFINISHED run on this worker, for an operator. Administrators only.

        Answers the question the per-caller run listing deliberately cannot:
        what is this worker currently working on. `list_training_sessions`
        returns a run to its owner, to a manager of its dataset, or -- when it
        has no owner -- to a `bioimage-io` administrator. A run started by a
        caller who was not signed in therefore becomes invisible to everybody
        once their connection closes, because Hypha mints that identity per
        connection: the owner no longer exists to look, and the run is owned so
        no administrator may list it. It keeps occupying the worker regardless.

        **Each record is projected to an allow-list of operator fields** --
        status, timings, backend and epoch counts -- rather than having its
        owner fields stripped. An earlier version stripped `owner` and
        `owner_anonymous` and claimed it said nothing about whose the run was;
        that was false, because the record also carries `dataset_artifact_id`,
        and a dataset its manager owns lives in that manager's own workspace,
        so the id names them. The free-text label was a second such channel.
        An allow-list has to be right only about what an operator needs; a
        strip-list has to be right about every field anybody ever adds.

        That is also why this is a separate call rather than a loosened
        `list_training_sessions`: widening the administrator leg of that one
        would hand every owned run to every administrator and undo a privacy
        decision made on purpose, which a test pins.

        **`stale` marks a record that is non-terminal but probably dead**, with
        `stale_for_s` giving how long it has been quiet. A non-terminal status
        is not a running process: the stale sweep in `get_status` only moves
        PREPARING and TRAINING on, so a run that was QUEUED when its replica
        restarted stays QUEUED for ever. Without the mark an operator would
        read a busy worker that is in fact idle. Stale records are marked, not
        hidden, because one of them may be a genuinely stuck run.

        Returns `{}` for anyone who is not a `bioimage-io` administrator --
        refusing rather than raising, because an operator surface should not
        become a way to probe whether anything is running at all.
        """
        import training

        if not training.caller_is_workspace_admin(context):
            return {}
        return training.non_terminal_runs()

    @bioengine.method(context=True)
    async def stop_training(
        self,
        session_id: str = Field(..., description="Session id to stop."),
        context=None,
    ) -> Dict[str, Any]:
        """Request cancellation (an in-flight training epoch may finish first).

        Only the run's own caller, or a `bioimage-io` administrator, may stop it.
        A run with no recorded owner is administrator-only: it cannot be
        attributed to anybody, and guessing is worse than admitting that.

        Refuses a session that does not exist, and creates nothing when it does
        so. `write_status` below calls `mkdir(parents=True)`, which is how a real
        session's directory comes into being on its first write -- so without
        that check, stopping a run that never existed MANUFACTURES it, complete
        with a status record that then appears in the run listing.

        THE ORDER OF THE TWO IS LOAD-BEARING. The ownership gate runs FIRST, so a
        caller who does not own the run is refused without learning whether it
        exists; only a caller already entitled to act on it reaches the
        existence check and can tell the two apart.

        `context=True` is load-bearing too. Without it the proxy forwards no
        context, the check sees no identity, and it then refuses EVERYBODY -- the
        safe direction, but a silent outage, so a test asserts the decorator.
        """
        import training

        training.require_caller_may_mutate(session_id, context)
        if not training.session_dir(session_id).exists():
            raise training.SessionNotFound(
                f"no training session {session_id!r} on this worker"
            )
        training.request_stop(session_id)
        task = self._training_tasks.get(session_id)
        if task is not None and not task.done():
            task.cancel()
        training.write_status(
            session_id,
            status="STOPPED",
            message="stop requested by user",
            terminated_by="user_stop",
        )
        return training.get_status(session_id)

    @bioengine.method(context=True)
    async def export_model(
        self,
        session_id: str = Field(
            ..., description="COMPLETED fine-tuning session to export."
        ),
        name: str = Field(..., description="Model name for the BioImage.IO package."),
        description: str = Field("", description="Model description for the RDF."),
        authors: Optional[List[Dict[str, Any]]] = Field(
            None,
            description="Authors, each {name, affiliation?, email?, github_user?, orcid?}.",
        ),
        license: str = Field("CC-BY-4.0", description="SPDX license id for the model."),
        provenance: Optional[Dict[str, Any]] = Field(
            None,
            description="Baked server-side into RDF config.microsam_provenance, e.g. "
            "{dataset_artifact_id, label, split_name, session_lineage}.",
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Start building a BioImage.IO model package from a COMPLETED fine-tuning
        session — a **combined SAM+decoder** package (interactive prompt head + AIS
        decoder, via ``micro_sam.bioimageio.export_sam_model``) for micro-sam
        sessions, or a **Cellpose pytorch_state_dict** package for cellpose
        (cpsam / cpdino) sessions. Async, like start_training: returns immediately with an
        ``export_id`` — poll ``get_export_status``. The package is self-tested on
        CPU (``bioimageio.core.test_model``), built server-side, and staged on
        temporary storage; this method **publishes nothing**. The frontend creates
        the draft artifact with the user's own token and either downloads the zip
        from ``download_url`` or calls ``push_export`` to stream the package files
        straight into the draft. ``license`` applies to micro-sam packages only
        (cellpose packages carry Cellpose's BSD-3-Clause).
        """
        import training

        # BEFORE any work, and before the checkpoint lookup, which would tell an
        # unauthorised caller whether the session exists and whether it trained.
        training.require_caller_may_mutate(session_id, context)
        self._session_checkpoint(session_id)  # raises if no trained checkpoint
        served = "cpsam" if training.session_backend(session_id) == "cellpose" else None
        await self._check_runtime_available(served)
        export_id = uuid.uuid4().hex
        # WHO asked, recorded from the injected context at creation time. The
        # export id is a handle to this run's weights, so the handle has to be
        # bound to an identity -- `export_model` being owner-or-admin says
        # nothing about who may later USE the id it returns. Stored under
        # underscore keys so the single projection in `get_export_status`,
        # which drops everything starting with "_", keeps the creator out of
        # every response without a second rule to keep in step.
        creator = training.caller_identity(context)
        request = {
            "name": name,
            "description": description,
            "authors": authors,
            "license": license,
            "provenance": provenance,
        }
        training.write_export(
            export_id,
            status="PENDING",
            progress=0.0,
            message="queued",
            download_url=None,
            size_bytes=None,
            error=None,
            _owner=creator.get("owner"),
            _owner_anonymous=creator.get("owner_anonymous"),
        )
        self._export_tasks[export_id] = asyncio.create_task(
            self._run_export(export_id, session_id, request)
        )
        return {"export_id": export_id, "status": "PENDING"}

    async def _export_heartbeat(self, export_id: str, interval: float = 60.0) -> None:
        """Refresh the export record's ``updated_at`` while its build is alive.

        Building the package and uploading it each run for minutes with no
        progress write, which would exceed STATUS_STALE_SECONDS and have
        get_export sweep a healthy export to FAILED. This dies with the replica,
        which is what lets the sweep fire for a build nobody is driving.
        """
        import training

        while True:
            await asyncio.sleep(interval)
            if training.read_export(export_id).get("status") not in (
                "PENDING",
                "BUILDING",
            ):
                return
            training.write_export(export_id)

    async def _run_export(
        self, export_id: str, session_id: str, request: Dict[str, Any]
    ) -> None:
        import training

        export_hb = asyncio.create_task(self._export_heartbeat(export_id))
        try:
            training.write_export(
                export_id, status="BUILDING", progress=0.1, message="building package"
            )
            served = (
                "cpsam" if training.session_backend(session_id) == "cellpose" else None
            )
            res = await self._runtime_for(served).export_bioimageio(
                session_id=session_id, request=request
            )
            training.write_export(
                export_id,
                _package_dir=res["package_dir"],
                progress=0.8,
                message="staging package",
            )
            zip_path = Path(res["zip"])
            file_path = f"temp/{export_id}/{zip_path.name}"
            upload_url = await self._s3.put_file(file_path=file_path, ttl=6 * 3600)
            resp = await self._put_file(upload_url, zip_path, timeout=600.0)
            resp.raise_for_status()
            download_url = await self._get_download_url(file_path)
            training.write_export(
                export_id,
                status="READY",
                progress=1.0,
                message="ready",
                download_url=download_url,
                size_bytes=res["zip_size"],
                files=res["files"],
                resume_checkpoint_file=res.get("resume_checkpoint_file"),
            )
        except Exception as e:
            logger.exception(f"Export {export_id} failed")
            training.write_export(
                export_id, status="FAILED", progress=1.0, message="failed", error=str(e)
            )
        finally:
            export_hb.cancel()

    @bioengine.method(context=True)
    async def get_export_status(
        self,
        export_id: str = Field(..., description="export_id from export_model."),
        context=None,
    ) -> Dict[str, Any]:
        """Export progress. ``status`` ∈ {PENDING, BUILDING, READY, FAILED}.
        ``download_url`` (6h TTL) + ``size_bytes`` are set on READY; ``error`` on
        FAILED. ``files`` lists the package members (name + size) so the frontend
        can mirror them when creating the draft artifact and calling push_export.
        ``resume_checkpoint_file`` names the package's resumable weights file, to
        feed back into ``start_training(init_checkpoint=...)``."""
        import training

        # BEFORE the record is read, so the refusal cannot depend on what the
        # record says. An unauthorised caller must not be able to tell "not
        # yours" from "no such export": both come out of this call. (`get_export`
        # does NOT write the swept record -- an earlier version of this comment
        # said it did -- so the ordering is about the refusal, not a side
        # effect.)
        training.require_caller_may_use_export(export_id, context)
        rec = training.get_export(export_id)
        if not rec:
            raise KeyError(f"Unknown export_id '{export_id}'.")
        return {k: v for k, v in rec.items() if not k.startswith("_")}

    @bioengine.method(context=True)
    async def push_export(
        self,
        export_id: str = Field(..., description="A READY export_id from export_model."),
        files: Dict[str, str] = Field(
            ...,
            description="Map of package file name -> presigned PUT URL. Keys must "
            "match the 'files' from get_export_status; mint the URLs via the draft "
            "artifact's put_file with the user's token.",
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Stream the built package files straight into a draft artifact so the
        browser never moves the ~350 MB twice. The frontend creates the staged
        draft with the user's token, mints a presigned put_file URL per package
        file, then calls this. Draft-only: this app never creates, commits, or
        reads the artifact."""
        import training

        # The sharper half of the same gap: this streams the whole package --
        # weights included -- to any presigned URL the caller hands over. Gated
        # before the record is read, for the same reason as the status read.
        training.require_caller_may_use_export(export_id, context)
        rec = training.get_export(export_id)
        if not rec:
            raise KeyError(f"Unknown export_id '{export_id}'.")
        if rec.get("status") != "READY":
            raise RuntimeError(
                f"Export '{export_id}' is not READY (status={rec.get('status')})."
            )
        pkg_dir = Path(rec["_package_dir"])
        # EVERY name is bounded BEFORE the first byte moves, not as each file
        # comes up. `files` keys come from the caller: before this bound
        # existed, a key was joined to the package directory unchecked and a
        # caller holding any READY export of their own could push
        # "../../<another run>/best.pt" -- another run's checkpoint -- to a URL
        # of their choosing. `package_member_path` requires the name to be one
        # this export ANNOUNCED and to resolve inside the package.
        #
        # Validating the whole mapping first is the review's point and it is
        # right: checking inside the loop meant a push whose last key was bad
        # had already streamed the earlier files. Nothing leaked -- they were
        # the caller's own announced files -- but a refused push that half
        # happened is a worse thing to explain than one that did nothing.
        resolved = []
        for rel in files:
            fpath = training.package_member_path(
                pkg_dir, rel, announced=rec.get("files")
            )
            if not fpath.is_file():
                raise FileNotFoundError(
                    f"Package file '{rel}' not found for export '{export_id}'."
                )
            resolved.append((rel, files[rel], fpath))

        pushed = []
        for rel, url, fpath in resolved:
            resp = await self._put_file(url, fpath, timeout=600.0)
            resp.raise_for_status()
            pushed.append(rel)
        return {"pushed": pushed, "n_files": len(pushed)}

    # === community model-soup pool (async, at-your-own-pace) ===
    #
    # Contributors fine-tune cpsam on their own local data and upload only the
    # resulting weights into a shared Hypha artifact (the pool); a coordinator
    # later averages them into a growing community checkpoint. The data never
    # moves. All pool I/O runs through a SEPARATE artifact-manager client authed
    # with the caller's pool_token — never this app's own HYPHA_TOKEN — so the
    # app cannot be a confused deputy for a pool it was not granted. The
    # weight-averaging body (aggregate) and the contribute acceptance gate
    # (drift-bound / sample-weighting) are deliberately absent here: their family
    # is a pending decision, and this backend only measures, it does not gate.

    def _pool_transport_path(self) -> Path:
        return Path.home() / ".bioengine" / "pool_transport.json"

    def _load_transport(self) -> Dict[str, int]:
        """Cumulative weight-bytes moved through the pool, persisted under the
        mounted ~/.bioengine so the flagship's headline figure survives a worker
        restart. bytes_up/bytes_down are the two numbers reported to the page —
        never combined into a ratio."""
        base = {"bytes_up": 0, "bytes_down": 0, "contributions": 0, "pool_reads": 0}
        try:
            saved = json.loads(self._pool_transport_path().read_text())
            for k in base:
                if isinstance(saved.get(k), int):
                    base[k] = saved[k]
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(
                f"pool transport counter unreadable, starting from zero: {e}"
            )
        return base

    def _persist_transport(self) -> None:
        p = self._pool_transport_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._transport))
            os.replace(tmp, p)
        except Exception as e:
            logger.warning(f"could not persist pool transport counter: {e}")

    def _bump_transport(self, **deltas: int) -> None:
        for k, v in deltas.items():
            self._transport[k] = self._transport.get(k, 0) + int(v)
        self._persist_transport()

    async def _pool_service(self, pool_token: Optional[str]):
        """Open a caller-scoped artifact-manager client for pool I/O. Requires a
        pool_token — the app never touches a pool with its own credential."""
        if not pool_token:
            raise ValueError(
                "pool_token is required for pool access — pass a token scoped to "
                "the pool artifact, not the app's own token."
            )
        client = await connect_to_server(
            {"server_url": SERVER_URL, "token": pool_token}
        )
        am = await client.get_service("public/artifact-manager")
        return am, client

    async def _read_pool_index(self, am, pool_artifact_id: str) -> Dict[str, Any]:
        import pool

        try:
            url = await am.get_file(pool_artifact_id, file_path=pool.POOL_INDEX)
        except Exception as e:
            if not _is_missing_file(e):
                raise
            raise FileNotFoundError(
                f"pool '{pool_artifact_id}' has no {pool.POOL_INDEX} — it is not an "
                f"initialized model-soup pool ({e})."
            )
        resp = await self._http_retry("GET", url, timeout=120.0)
        if resp.status_code == 404:
            raise FileNotFoundError(f"pool index missing for '{pool_artifact_id}'.")
        resp.raise_for_status()
        return resp.json()

    async def _read_merge_journal(
        self, am, pool_artifact_id: str
    ) -> Optional[Dict[str, Any]]:
        """Read aggregate()'s resume journal, or None when there is nothing to resume.

        Absent is the normal first-merge case, so unlike the pool index this never
        raises on a miss: a merge that cannot read its journal must still run, just
        without the saved passes.
        """
        import pool

        try:
            url = await am.get_file(pool_artifact_id, file_path=pool.MERGE_JOURNAL)
            resp = await self._http_retry("GET", url, timeout=120.0)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(
                f"no usable merge journal for '{pool_artifact_id}' ({e}); "
                f"this merge starts from no saved passes."
            )
            return None

    async def _write_merge_journal(
        self, am, pool_artifact_id: str, journal: Dict[str, Any]
    ) -> None:
        """Persist the resume journal, after every score so none is paid for twice.

        Never raises. The journal buys resumability; it is not an input to the
        merge, so a failed write must cost the saved passes and not the GPU hours
        still running. It is logged at warning on every miss rather than once,
        because the degraded mode — a merge that completes but cannot be resumed —
        is otherwise indistinguishable from a healthy one.

        Safe to call only while no staged edit is open: committing with a stage
        pending promotes that stage to a new version instead of overwriting in
        place. Every call site is in the scoring phase, before aggregate() opens
        its own stage to publish the head.
        """
        import pool

        try:
            async with self._staged(am, pool_artifact_id):
                url = await am.put_file(pool_artifact_id, file_path=pool.MERGE_JOURNAL)
                (
                    await self._http_retry(
                        "PUT",
                        url,
                        content=json.dumps(journal).encode("utf-8"),
                        timeout=120.0,
                    )
                ).raise_for_status()
                await self._commit_with_retry(am, pool_artifact_id)
        except Exception as e:
            logger.warning(
                f"could not persist the merge journal for "
                f"'{pool_artifact_id}' ({e}); a kill now would lose the "
                f"{len(journal.get('scores', {}))} scored pass(es) since "
                f"the last successful write."
            )

    @asynccontextmanager
    async def _staged(self, am, artifact_id: str):
        """Open a staged edit, discarding the stage if the body does not reach commit.

        ``put_file`` records a manifest row when it mints the URL, not when the
        bytes land, so an exception between staging and commit leaves a row with no
        object behind it. ``list_files`` lists objects and cannot show it, while
        every later ``commit`` raises ``File '<path>' does not exist in the
        artifact`` — which ``_commit_with_retry`` below then misreads as the
        read-after-write 404 and retries for ~62s before re-raising. Without the
        discard the artifact is permanently uncommittable, so one failed upload
        takes the whole pool down rather than just its own merge.
        """
        await am.edit(artifact_id, stage=True)
        try:
            yield
        except BaseException:
            try:
                await am.discard(artifact_id)
            except Exception as e:
                logger.warning(
                    f"⚠️ could not discard the staged edit on '{artifact_id}' ({e}); "
                    f"if a put_file row was minted without its bytes, every later "
                    f"commit on this artifact fails until the stage is discarded."
                )
            raise

    async def _commit_with_retry(
        self, am, artifact_id: str, tries: int = 6, base: float = 2.0
    ) -> None:
        """Commit a staged artifact, retrying the transient read-after-write 404 the
        deNBI/isilon S3 raises when a just-PUT object is not yet visible to commit's
        HeadObject check. The uploads themselves already succeeded (their PUT raises
        on failure), so a bounded backoff clears the visibility lag; anything else,
        or exhausting the retries, re-raises."""
        for i in range(tries):
            try:
                await am.commit(artifact_id)
                return
            except Exception as e:
                msg = str(e)
                transient = any(
                    s in msg
                    for s in ("HeadObject", "does not exist", "Not Found", "404")
                )
                if not transient or i == tries - 1:
                    raise
                await asyncio.sleep(base * (2**i))

    async def _read_community_manifest(
        self, am, pool_artifact_id: str, community_version: str
    ) -> Dict[str, Any]:
        """Read ``community/<version>/manifest.json`` and shape it like an index
        ``current_community`` entry, so a named version and the pointer resolve
        through one code path downstream."""
        import pool

        paths = pool.community_paths(community_version)
        try:
            url = await am.get_file(pool_artifact_id, file_path=paths["manifest"])
        except Exception as e:
            if not _is_missing_file(e):
                raise
            raise FileNotFoundError(
                f"pool '{pool_artifact_id}' has no {paths['manifest']} — version "
                f"'{community_version}' was never published ({e}). Versions count from "
                f"'v1'; note get_pool_state reads the index, so it cannot list a version "
                f"whose index entry was never committed."
            )
        resp = await self._http_retry("GET", url, timeout=120.0)
        if resp.status_code == 404:
            raise FileNotFoundError(
                f"manifest missing for '{pool_artifact_id}' version '{community_version}'."
            )
        resp.raise_for_status()
        manifest = resp.json()
        return {
            "version": manifest.get("community_version", community_version),
            "path": paths["weights"],
            "sha256": manifest.get("weights_sha256"),
            # A published manifest is the SECOND stored surface reached only from here —
            # it does not pass through pool_wire_state — so both of the corrections that
            # the pool index gets have to be repeated at this chokepoint or they are
            # absent from every named community version. Both callers come through here.
            #
            # The manifest carries this digest too, so say which of the two it is.
            # It has its own created_at, so no join is needed.
            # The manifest also carries the basis wording 0.27.15 retracted, so
            # correcting only the pool index would still leave a contributor reading the
            # withdrawn claim off a named version.
            "sha256_provenance": pool.digest_provenance(
                manifest, digest_key="weights_sha256"
            ),
            "selection_metric": pool.corrected_metric(manifest.get("selection_metric")),
            "witness_metric": pool.corrected_metric(manifest.get("witness_metric")),
        }

    async def _next_free_community_version(
        self, am, pool_artifact_id: str, index: Dict[str, Any]
    ) -> str:
        """The next ``vN`` that is not already published.

        pool.next_community_version derives it from the index pointer alone, so a pool
        whose pointer was never committed (an aggregate interrupted between its two
        commits) gets back a version whose weights are already on disk — and publishing
        onto it overwrites an immutable community checkpoint in place. Probe the manifest
        and advance past anything already there."""
        import pool

        v = pool.next_community_version(index)
        for _ in range(100):
            try:
                await self._read_community_manifest(am, pool_artifact_id, v)
            except FileNotFoundError:
                return v
            logger.warning(
                f"community {v} is already published in '{pool_artifact_id}' but the index "
                f"does not name it (an interrupted publish); advancing rather than "
                f"overwriting it."
            )
            v = f"v{int(v.lstrip('v')) + 1}"
        raise RuntimeError(
            f"no free community version below v100 in '{pool_artifact_id}'."
        )

    async def _resolve_pool_init(
        self, model_type, init_from_pool, pool_token, community_version=None
    ):
        """Resolve a pool community checkpoint to a download URL for
        start_training(init_from_pool=...). Returns (url, provenance); provenance
        is persisted with the session so _run_training can count the download.

        ``community_version`` ('v1', …) resolves that version's own manifest.json
        instead of the index pointer. aggregate() publishes a version in two commits
        — weights + manifest first, then the index pointer — so an interruption
        between them leaves complete, scoreable weights the pointer never names. At
        v1 there is no prior pointer to fall back to, so the index reads
        ``current_community: null`` and the pointer path cannot reach the weights at
        all. The manifest is written in the FIRST commit, so requiring it here also
        proves the version was fully published rather than half-written."""
        import pool

        am, client = await self._pool_service(pool_token)
        try:
            index = await self._read_pool_index(am, init_from_pool)
            idx_model = index.get("model_type")
            if idx_model != model_type:
                raise ValueError(
                    f"pool '{init_from_pool}' holds {idx_model} checkpoints, but "
                    f"model_type is {model_type}; they must match to seed from it."
                )
            if community_version:
                cc = await self._read_community_manifest(
                    am, init_from_pool, community_version
                )
            else:
                cc = pool.current_community(index)
                if not cc:
                    raise ValueError(
                        f"pool '{init_from_pool}' has no community checkpoint yet — "
                        f"nothing to seed from (contribute + aggregate first). If an "
                        f"aggregate was interrupted after publishing weights but before "
                        f"the index pointer, name the version directly with "
                        f"init_from_pool_version='v1'."
                    )
            url = await am.get_file(init_from_pool, file_path=cc["path"])
            provenance = {
                "pool_artifact_id": init_from_pool,
                "community_version": cc.get("version"),
                "community_sha256": cc.get("sha256"),
            }
            return url, provenance
        finally:
            try:
                await asyncio.wait_for(client.disconnect(), timeout=2.0)
            except Exception:
                pass

    @bioengine.method(context=True)
    async def contribute(
        self,
        session_id: str = Field(
            ..., description="A COMPLETED cpsam fine-tuning session to contribute."
        ),
        pool_artifact_id: str = Field(
            ...,
            description="Community pool artifact id, e.g. 'bioimage-io/cpsam-community-pool'.",
        ),
        pool_token: str = Field(
            ...,
            description="Token with write access to pool_artifact_id. Scoped to the pool, not the app's own token.",
        ),
        dataset_fingerprint: Optional[str] = Field(
            None,
            description="Optional caller-supplied fingerprint of the local training set "
            "(the data never moves; this only lets the pool tell contributions apart).",
        ),
        solo_val_metric: Optional[float] = Field(
            None,
            description="Optional solo validation score for this contribution, recorded for audit.",
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Upload a finished cpsam session's weights into a community model-soup
        pool. Records weights + an auditable metadata.json (architecture
        signature, samples_seen, L2 drift from the base it started from,
        provenance) and appends a stub to the pool index — one commit per
        contribution, so lineage is fully traceable. Averaging happens later, in
        the coordinator's aggregate step; this only deposits. Idempotent by
        weights digest: re-depositing a checkpoint the pool already holds
        appends nothing and returns the first deposit's record with
        deduplicated=true, so a client that lost its response to a gateway
        timeout can safely retry. cpsam(cellpose) sessions only for now."""
        import time as _time

        import pool
        import training

        # BEFORE get_status, whose "session ... is <status>, not COMPLETED" error
        # would otherwise tell an unauthorised caller that a run exists and what
        # state it is in -- the same oracle deliberately avoided in export_model.
        #
        # Publishing someone else's checkpoint into the shared community pool is
        # the same harm as exporting it, and worse in reach: the pool index is
        # append-only and shared, so a contribution cannot simply be withdrawn.
        training.require_caller_may_mutate(session_id, context)
        st = training.get_status(session_id)
        if st.get("status") != "COMPLETED":
            raise ValueError(
                f"session '{session_id}' is {st.get('status')}, not COMPLETED — only a "
                f"finished fine-tune can be contributed."
            )
        if training.session_backend(session_id) != "cellpose":
            raise ValueError(
                "contribute currently accepts cpsam (cellpose) sessions only."
            )
        params = training.read_training_params(session_id)
        model_type = params.get("model_type") or st.get("model_type")
        if model_type != "cpsam":
            raise ValueError(f"contribute is cpsam-only for now, not '{model_type}'.")

        provenance = params.get("pool_provenance")
        if provenance:
            base_checkpoint_id = f"{provenance.get('pool_artifact_id')}@{provenance.get('community_version')}"
        elif params.get("init_checkpoint"):
            base_checkpoint_id = str(params["init_checkpoint"])
        else:
            base_checkpoint_id = "stock-cpsam"

        await self._check_runtime_available(model_type)
        runtime = self._runtime_for(model_type)

        epochs = st.get("n_epochs_completed") or params.get("n_epochs") or 0
        n_inputs = st.get("n_train_inputs") or 0
        samples_seen = int(n_inputs) * int(epochs)

        cid = uuid.uuid4().hex
        uploaded_at = _time.time()
        paths = pool.contribution_paths(cid)

        # Resolved once and used by both the stored metadata and the value
        # returned to the caller, so the two records cannot date the same
        # contribution differently. Empty is the same as unset here: an
        # exported-but-empty var would otherwise stamp "" and read as a known
        # version. Read out here rather than mid-block because the recognised
        # re-deposit below returns without reaching the metadata build.
        app_version = os.environ.get("BIOENGINE_ARTIFACT_VERSION") or None

        # Set by the except clause below, and the only thing that distinguishes
        # the two return paths.
        redeposited: Optional[Dict[str, Any]] = None

        am, client = await self._pool_service(pool_token)
        try:
            try:
                index = await self._read_pool_index(am, pool_artifact_id)
            except FileNotFoundError:
                index = None  # uninitialized pool: the first contribution bootstraps it
            async with self._staged(am, pool_artifact_id):
                w_url = await am.put_file(pool_artifact_id, file_path=paths["weights"])
                # The checkpoint is torch + a runtime-local file, neither reachable from
                # the torch-free CPU entry; the GPU runtime inspects it, PUTs the weights
                # to the presigned URL, and returns the signature + hashes, so nothing
                # large or torch-dependent crosses the entry.
                info = await runtime.prepare_contribution(
                    session_id, w_url, model_type=model_type
                )
                signature = info["architecture_signature"]
                if index is None:
                    # A fine-tune preserves the parameter layout, so the first
                    # contribution's signature is the pool's reference signature
                    # (equals stock cpsam). The pool artifact itself must already
                    # exist (created out-of-band with the pool_token's scope).
                    #
                    # The base commitment is written HERE, once, at pool
                    # creation. It is built on the runtime because a commitment needs
                    # the base's state_dict and this entry is torch-free; passing it
                    # through is what makes the pool mergeable at all, since
                    # assert_base_commitment refuses a pool that records none and has
                    # deliberately no fallback. A bootstrap that omitted it produced a
                    # pool that could never merge.
                    index = pool.empty_index(
                        model_type, signature, info.get("base_commitment")
                    )
                else:
                    pool.assert_compatible(
                        index, model_type=model_type, signature=signature
                    )

                # Earliest point a re-deposit is knowable: the digest is computed
                # from the bytes by prepare_contribution, which has already PUT
                # them. Identity is the digest alone — see
                # pool.find_contribution_by_digest for why metadata is not part of
                # it, and why decided_contribution_ids is not consulted. Leaving by
                # raise, never by return: see _RecognisedRedeposit.
                existing = pool.find_contribution_by_digest(
                    index, info["weights_sha256"]
                )
                if existing is not None:
                    raise _RecognisedRedeposit(existing)

                metadata = pool.build_contribution_metadata(
                    contribution_id=cid,
                    model_type=model_type,
                    app_version=app_version,
                    cellpose_version=info["cellpose_version"],
                    architecture_signature=signature,
                    base_checkpoint_id=base_checkpoint_id,
                    base_checkpoint_sha256=info["base_checkpoint_sha256"],
                    samples_seen=samples_seen,
                    training_params=params,
                    l2_from_base=info["l2_from_base"],
                    # .get, not ["…"]: an older runtime replica that predates
                    # the label still answers this call, and it should write a null
                    # label — which reads as undetermined — rather than raise.
                    l2_from_base_basis=info.get("l2_from_base_basis"),
                    dataset_fingerprint=dataset_fingerprint,
                    solo_val_metric=solo_val_metric,
                    weights_sha256=info["weights_sha256"],
                    uploaded_at=uploaded_at,
                )
                m_url = await am.put_file(pool_artifact_id, file_path=paths["metadata"])
                (
                    await self._http_retry(
                        "PUT",
                        m_url,
                        content=json.dumps(metadata).encode("utf-8"),
                        timeout=120.0,
                    )
                ).raise_for_status()
                pool.append_contribution(
                    index,
                    {
                        "contribution_id": cid,
                        "weights_path": paths["weights"],
                        "metadata_path": paths["metadata"],
                        "weights_sha256": info["weights_sha256"],
                        "architecture_signature": signature,
                        "samples_seen": samples_seen,
                        "n_train_images": int(n_inputs),
                        "n_bytes": int(info["n_bytes"]),
                        "uploaded_at": uploaded_at,
                    },
                )
                i_url = await am.put_file(pool_artifact_id, file_path=pool.POOL_INDEX)
                (
                    await self._http_retry(
                        "PUT",
                        i_url,
                        content=json.dumps(index).encode("utf-8"),
                        timeout=120.0,
                    )
                ).raise_for_status()
                await self._commit_with_retry(am, pool_artifact_id)
        except _RecognisedRedeposit as recognised:
            # _staged has already discarded the stage on the way out, so the pool
            # is left exactly as it was found: same members, no open edit, and no
            # index PUT ever issued.
            redeposited = recognised.existing
        finally:
            try:
                await asyncio.wait_for(client.disconnect(), timeout=2.0)
            except Exception:
                pass

        if redeposited is not None:
            # The bytes did cross the wire — prepare_contribution PUT them before
            # anything could know they were a duplicate — so bytes_up counts them;
            # that counter is "weight-bytes moved", and pretending they did not
            # move would understate the cost of exactly the retries this handles.
            # contributions does NOT move: the pool gained no member, and that is
            # the number quoted as "deposits".
            self._bump_transport(bytes_up=int(info["n_bytes"]))
            return {
                "schema_version": pool.SCHEMA_VERSION,
                "app_version": app_version,
                # The FIRST deposit's id, and the pool's own record of what it
                # holds, so a client that lost its response to a gateway timeout
                # ends up with the record it would have had. samples_seen is read
                # back off the index for the same reason: if this retry reported a
                # different count, the stored member is the one the soup will
                # weight. l2_from_base comes from this call because the index does
                # not carry it — it is recomputed from the same bytes against the
                # same stock base, not read back from the first deposit.
                "contribution_id": redeposited.get("contribution_id"),
                "pool_artifact_id": pool_artifact_id,
                "architecture_signature": redeposited.get(
                    "architecture_signature", signature
                ),
                "weights_sha256": info["weights_sha256"],
                "samples_seen": int(redeposited.get("samples_seen", samples_seen)),
                "l2_from_base": info["l2_from_base"],
                # The caller keeps THIS record, so it carries the label for
                # the same reason the stored one does: a distance whose origin is
                # not named is not comparable with anyone else's.
                "l2_from_base_basis": info.get("l2_from_base_basis"),
                "deduplicated": True,
            }

        self._bump_transport(bytes_up=int(info["n_bytes"]), contributions=1)
        return {
            "schema_version": pool.SCHEMA_VERSION,
            # The caller keeps this record, not the stored metadata.json, so it
            # carries app_version too: schema_version does not cover a release
            # that changes a value without changing the shape (0.27.6 moved
            # every l2_from_base), which leaves a saved response undatable.
            "app_version": app_version,
            "contribution_id": cid,
            "pool_artifact_id": pool_artifact_id,
            "architecture_signature": signature,
            "weights_sha256": info["weights_sha256"],
            "samples_seen": samples_seen,
            "l2_from_base": info["l2_from_base"],
            "l2_from_base_basis": info.get("l2_from_base_basis"),
            # False, not absent, on the fresh path. A key present only when true
            # cannot be dated: a client holding a response without it cannot tell
            # "fresh deposit" from "a release that predates de-duplication" —
            # the defect app_version exists to avoid.
            "deduplicated": False,
        }

    @bioengine.method
    async def get_transport_stats(self) -> Dict[str, Any]:
        """Cumulative weight-bytes moved through the pool since this counter was
        first created, persisted across worker restarts. bytes_up (contributions)
        and bytes_down (pool seeds / eval downloads) are reported as two separate
        numbers — the flagship's 'data too big to move' headline contrasts
        bytes-moved against data-held; it is never expressed as a ratio here."""
        import pool

        return {
            **self._transport,
            "schema_version": pool.SCHEMA_VERSION,
            "since": "worker-persistent",
        }

    @bioengine.method
    async def get_pool_state(
        self,
        pool_artifact_id: str = Field(..., description="Community pool artifact id."),
        pool_token: str = Field(
            ...,
            description="Token with read access to the pool. Scoped to the pool, not the app's own token.",
        ),
    ) -> Dict[str, Any]:
        """The pool's state projected onto the campaign-page wire contract
        (schema_version 0.15.0-draft): merge_trigger, baseline_metric (role=witness),
        soups[] (published merges), empty_merges[] (merges that admitted nothing),
        and enriched contribution stubs — each carrying that contribution's
        weights_sha256, so a contributor whose deposit response was lost can tell a
        landed upload from one it must retry. Role-tagged metrics come straight from
        storage — the role is a property of the artifact, not assigned here."""
        import pool

        am, client = await self._pool_service(pool_token)
        try:
            index = await self._read_pool_index(am, pool_artifact_id)
        finally:
            try:
                await asyncio.wait_for(client.disconnect(), timeout=2.0)
            except Exception:
                pass
        self._bump_transport(pool_reads=1)
        return pool.pool_wire_state(index, pool_artifact_id=pool_artifact_id)

    @bioengine.method
    async def evaluate_community(
        self,
        pool_artifact_id: str = Field(..., description="Community pool artifact id."),
        pool_token: str = Field(
            ...,
            description="Token with read access to the pool. Scoped to the pool, not the app's own token.",
        ),
        val_images: List[Union[np.ndarray, str]] = Field(
            ...,
            description="Validation images (arrays, URLs, or get_upload_url paths).",
        ),
        community_version: Optional[str] = Field(
            None,
            description="Which community checkpoint to evaluate; defaults to the pool's current one.",
        ),
        diam_mean: float = Field(
            30.0, gt=0, description="Mean object diameter in pixels."
        ),
    ) -> List[Dict[str, Any]]:
        """Run a community checkpoint over a validation set and return its
        instance masks — the hook a scoring pass builds on. It is parameterised on
        the val set the caller supplies; the CHOICE of the canonical, non-
        overlapping split and the metric applied to these masks are decided with
        the fork, so this returns masks only and does not itself score."""
        import pool

        am, client = await self._pool_service(pool_token)
        try:
            index = await self._read_pool_index(am, pool_artifact_id)
            model_type = index.get("model_type", "cpsam")
            cc = pool.current_community(index)
            if community_version:
                weights_rel = pool.community_paths(community_version)["weights"]
            else:
                if not cc:
                    raise ValueError(
                        f"pool '{pool_artifact_id}' has no community checkpoint to evaluate."
                    )
                weights_rel = cc["path"]
            w_url = await am.get_file(pool_artifact_id, file_path=weights_rel)
            content = await self._get_verified(w_url, timeout=600.0)
        finally:
            try:
                await asyncio.wait_for(client.disconnect(), timeout=2.0)
            except Exception:
                pass

        # Per-version dest path: the GPU runtime caches its resident model by
        # checkpoint PATH (_ensure_model keys on it), so a fixed "community.pt"
        # made every version reuse the first-loaded (current) model. Tagging the
        # path with the version gives each community checkpoint a distinct cache
        # key, so a requested historical version actually loads.
        version_tag = re.sub(
            r"[^\w.-]",
            "_",
            str(community_version or (cc or {}).get("version") or "current"),
        )
        dest = (
            Path.home()
            / ".bioengine"
            / "pool_eval"
            / re.sub(r"[^\w.-]", "_", pool_artifact_id)
            / f"community_{version_tag}.pt"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        # dest is handed to the runtime as the checkpoint path below, so writing
        # it in place would let a concurrent evaluation of the same community
        # truncate ('wb') the file a torch.load is reading. Unique temp name:
        # the path is keyed on (pool, version) only, so writers do collide.
        await asyncio.to_thread(_write_file_atomic, dest, content)
        self._bump_transport(bytes_down=len(content))

        await self._check_runtime_available(model_type)
        images = [await self._resolve_image(src) for src in val_images]
        return await self._runtime_for(model_type).auto_segment(
            images=images,
            model_type=model_type,
            checkpoint=str(dest),
            generate_kwargs={"diameter": diam_mean},
        )

    @staticmethod
    def _decode_nd(d: Dict[str, Any]) -> np.ndarray:
        """Decode a runtime ndarray wire-dict (auto_segment returns these unchanged
        across the Ray hop; hypha only decodes at the RPC boundary, so an in-process
        caller must decode itself)."""
        return (
            np.frombuffer(d["_rvalue"], dtype=d["_rdtype"]).reshape(d["_rshape"]).copy()
        )

    async def _resolve_val_split(self, images, labels):
        """Resolve a (images, label-maps) split once: images to arrays, then each
        label to an instance map sized to its image."""
        imgs = [await self._resolve_image(src) for src in images]
        gts = [
            await self._resolve_label_array(lb, im.shape)
            for lb, im in zip(labels, imgs)
        ]
        return imgs, gts

    @bioengine.method
    async def aggregate(
        self,
        pool_artifact_id: str = Field(..., description="Community pool artifact id."),
        pool_token: str = Field(
            ...,
            description="COORDINATOR token with WRITE access to the pool. Scoped to the pool, never the app's own HYPHA_TOKEN.",
        ),
        gate_val_images: List[Union[np.ndarray, str]] = Field(
            ...,
            description="Gate split A images — disjoint from ALL contributor training data; a precondition the app cannot check.",
        ),
        gate_val_labels: List[Union[np.ndarray, str]] = Field(
            ...,
            description="Gate split A ground-truth instance label maps (0=background), one per image.",
        ),
        witness_val_images: List[Union[np.ndarray, str]] = Field(
            ...,
            description="Witness split B images — disjoint from A AND from all training data; never influences selection.",
        ),
        witness_val_labels: List[Union[np.ndarray, str]] = Field(
            ...,
            description="Witness split B ground-truth instance label maps, one per image.",
        ),
        gate_split: Dict[str, Any] = Field(
            ...,
            description="{artifact_id, sha256} provenance of split A, recorded in every community manifest.",
        ),
        witness_split: Dict[str, Any] = Field(
            ..., description="{artifact_id, sha256} provenance of split B."
        ),
        eps: float = Field(
            0.0,
            ge=0.0,
            description="Pre-registered non-degradation tolerance on split A (0 = strict). Do NOT tune post-hoc to hit an inclusion rate.",
        ),
        max_batch: int = Field(
            32,
            ge=1,
            description="Max newly-arrived candidates weighed at this merge; overflow is deferred (and re-scored) to the next.",
        ),
        min_members: Optional[int] = Field(
            None,
            ge=1,
            description="Refuse to publish a head with fewer members than this. Default: the published head's own member count (1 if there is no head), so quarantining can never silently thin the soup.",
        ),
        diam_mean: float = Field(
            30.0,
            gt=0,
            description="Mean object diameter in pixels for auto-segmentation.",
        ),
    ) -> Dict[str, Any]:
        """Coordinator-only greedy merge — grow the community soup by full-weight
        uniform mean, gated on split A, reported on witness split B.

        A candidate joins iff the mean of ``{current members} ∪ {candidate}`` does
        not degrade mean instance F1 (IoU>=0.5, the paper's matcher) on split A
        versus the current head, within ``eps``. Candidates within a merge are added
        greedily best-first; across merges the order is chronological (only arrived,
        undecided contributions are weighed; overflow defers and is re-scored later).
        A rejected candidate is recorded "evaluated, not included" — never a failure.

        Selection reads split A only; the reported improvement curve is scored on the
        disjoint witness split B (selecting on A then reporting on A would be
        circular). The first witness point is stock cpsam on B — the level to beat.

        Crash-safe: the new ``community/<version>/`` weights + manifest are committed
        BEFORE the index pointer is repointed, so a reader never sees the head aimed
        at a half-written version.

        Resumable: a merge is ``2 * len(batch) + 3`` scoring passes and each is
        minutes of exclusive GPU, so every score is written to the pool as it is
        earned. Calling aggregate() again on the same pool, splits and ``diam_mean``
        re-reads them, and a run killed mid-merge pays only for the passes it lost
        rather than starting over. ``resumed_passes`` in the result reports how many
        were reused; anything else — different splits, a different ``diam_mean``, a
        schema bump — discards the journal and scores from scratch."""
        import pool

        model_type = "cpsam"
        await self._check_runtime_available(model_type)
        runtime = self._runtime_for(model_type)
        gate_imgs, gate_gts = await self._resolve_val_split(
            gate_val_images, gate_val_labels
        )
        wit_imgs, wit_gts = await self._resolve_val_split(
            witness_val_images, witness_val_labels
        )

        am, client = await self._pool_service(pool_token)
        # Held from the index read to the index commit, not just around the publish:
        # a merge that only locked the publish would still have built its soup from
        # the head it read before waiting, and committing that would drop the merge
        # that ran while it waited. Serialised, the second caller re-reads the index
        # and weighs what is genuinely still undecided.
        lock = self._pool_locks.setdefault(pool_artifact_id, asyncio.Lock())
        if lock.locked():
            logger.info(
                f"aggregate() is waiting: another merge holds '{pool_artifact_id}'. "
                f"Scores already earned are in the pool journal, so the wait is not lost work."
            )
        await lock.acquire()
        self._merges_in_flight += 1
        try:
            index = await self._read_pool_index(am, pool_artifact_id)
            if index.get("model_type") != model_type:
                raise ValueError(
                    f"aggregate() is cpsam-only; pool model_type is '{index.get('model_type')}'."
                )

            async def _download(path: str) -> bytes:
                url = await am.get_file(pool_artifact_id, file_path=path)
                resp = await self._http_retry("GET", url, timeout=600.0)
                resp.raise_for_status()
                self._bump_transport(bytes_down=len(resp.content))
                return resp.content

            # The soup math + cpsam scoring live in the GPU runtime (the only
            # deployment with torch + cellpose). A member is addressed to it by its
            # pool weights_path plus a presigned GET, resolved once here; the runtime
            # downloads + caches the tensors by path, so nothing large crosses the
            # entry and no cross-actor local checkpoint is ever shared.
            url_cache: Dict[str, str] = {}
            # Bytes this merge downloaded, per serving replica. The member cache is
            # per-replica instance state and each gate pass is routed independently,
            # so a merge served by more than one replica downloads on each of them —
            # draining a single replica's counter at the end would drop the rest.
            replica_dl: Dict[str, int] = {}

            def _fold(result: Dict[str, Any]) -> Dict[str, Any]:
                replica_dl[result["replica"]] = replica_dl.get(
                    result["replica"], 0
                ) + int(result["soup_dl_bytes"])
                return result

            # What the pool recorded for each member's bytes, by path. Carried on
            # every spec so the runtime can reject a bad FRESH download at the point
            # it arrives — the preflight below hashes every member once per merge,
            # but the member cache is evicted under memory pressure, so a member can
            # be re-fetched mid-merge after it was verified.
            sha_by_path: Dict[str, Any] = {
                c.get("weights_path"): c.get("weights_sha256")
                for c in index.get("contributions", [])
            }

            async def _specs(paths: List[str]) -> List[Dict[str, str]]:
                out = []
                for p in paths:
                    if p not in url_cache:
                        url_cache[p] = await am.get_file(pool_artifact_id, file_path=p)
                    spec = {"key": p, "url": url_cache[p]}
                    recorded = sha_by_path.get(p)
                    if pool.is_usable_digest(recorded):
                        spec["sha256"] = recorded
                    out.append(spec)
                return out

            # Resume journal. A merge is 2*len(batch)+3 scoring passes and each pass
            # is minutes of exclusive GPU; the run that prompted this was killed at
            # pass 11 of 35 and returned nothing, losing 62 minutes. Every score is
            # written to the pool as it is earned, so a replica kill costs the pass
            # in flight and not the merge. The pool is the right home for it rather
            # than replica-local disk: the process that dies mid-merge is exactly the
            # one whose disk cannot be trusted afterwards.
            # Validate the splits before anything derives an identity from them:
            # the journal stamps those identities, so a bad one is baked in here.
            bad_split = pool.split_provenance_error(gate_split, witness_split)
            if bad_split:
                raise ValueError(f"aggregate() refused this merge: {bad_split}")

            journal = pool.usable_merge_journal(
                await self._read_merge_journal(am, pool_artifact_id),
                gate_split,
                witness_split,
                diam_mean,
            )
            resumed_from = len(journal["scores"])
            gate_id = pool.split_identity(gate_split)
            wit_id = pool.split_identity(witness_split)
            splits = {gate_id: (gate_imgs, gate_gts), wit_id: (wit_imgs, wit_gts)}

            async def _score(member_paths: List[str], split_id: str) -> Dict[str, Any]:
                """One scoring pass as ``{"f1": scalar, "per_image": vector}``.

                Both halves go through the journal together. Returning only the
                scalar and re-deriving the vector later is not available: a resumed
                merge never runs the pass again, so a vector that is not banked here
                is gone, and the published record would be missing its breakdown on
                exactly the merges that were interrupted.
                """
                key = pool.score_key(split_id, member_paths)
                if key in journal["scores"]:
                    # usable_merge_journal() prunes any entry missing its vector, so a
                    # hit on `scores` is a hit on both maps.
                    return {
                        "f1": float(journal["scores"][key]),
                        "per_image": list(journal["per_image_scores"][key]),
                    }
                imgs, gts = splits[split_id]
                # A journal hit returns above without folding, which is correct: that
                # pass transported nothing on this run.
                scored = _fold(
                    await runtime.score_soup(
                        await _specs(member_paths),
                        imgs,
                        gts,
                        diam_mean,
                        base_commitment=index.get("base_commitment"),
                    )
                )
                value, per_image = float(scored["f1"]), list(scored["per_image"])
                pool.record_score(journal, key, value, per_image=per_image)
                await self._write_merge_journal(am, pool_artifact_id, journal)
                return {"f1": value, "per_image": per_image}

            def _as_pass(metric: Dict[str, Any]) -> Dict[str, Any]:
                """A previously published metric record read back as a scoring pass,
                so a merge that reuses the baseline's number carries its breakdown too
                rather than dropping to a bare scalar. A record written before this app
                banked vectors has none, and ``.get`` is deliberate: an absent key and
                a null value must read alike here, because they mean the same thing."""
                return {
                    "f1": float(metric["aggregate"]),
                    "per_image": list(metric.get("per_image") or []),
                }

            def _metric(
                role: str, scored: Dict[str, Any], basis: str
            ) -> Dict[str, Any]:
                """One metric record from one scoring pass, breakdown attached.

                The vector and its basis are passed together or not at all, from one
                place, so no call site can publish half the pair. It is empty only for
                a number read back out of a pool written before this app measured
                breakdowns; that publishes null rather than inventing a breakdown for
                a number that was never measured with one.
                """
                vector = scored.get("per_image") or None
                return pool.round_metric(
                    role=role,
                    aggregate=round(float(scored["f1"]), 4),
                    aggregate_basis=basis,
                    per_image=vector,
                    per_image_basis=pool.PER_IMAGE_BASIS if vector else None,
                )

            cc = pool.current_community(index)
            cur_member_ids: List[str] = []
            by_id = {c["contribution_id"]: c for c in index.get("contributions", [])}
            if cc and cc.get("path"):
                cur_manifest = json.loads(
                    (
                        await _download(pool.community_paths(cc["version"])["manifest"])
                    ).decode()
                )
                cur_member_ids = list(cur_manifest.get("selected_contribution_ids", []))
            cur_member_paths = [by_id[cid]["weights_path"] for cid in cur_member_ids]

            pending = pool.undecided_contributions(index)
            batch = pending[:max_batch]
            deferred_ids = [c["contribution_id"] for c in pending[max_batch:]]
            cand_path = {c["contribution_id"]: c["weights_path"] for c in batch}

            # Refuse a merge that cannot fit BEFORE any scoring pass, rather than
            # discovering it as an OOM kill 11 passes in. This is as early as the
            # check can go: it needs `batch`, which is the first point at which the
            # members that will actually be loaded are known.
            budget = await runtime.soup_memory_budget()
            refusal = pool.soup_admission(
                [c.get("n_bytes") for c in batch]
                + [by_id[cid].get("n_bytes") for cid in cur_member_ids],
                budget_bytes=budget["declared_bytes"],
                base_bytes=budget["rss_bytes"] or 0,
            )
            if refusal:
                raise ValueError(f"aggregate() refused this merge: {refusal}")

            # ---- Provenance, before a single byte is souped ----------
            #
            # Until now nothing checked that a member's bytes were the bytes its
            # contribution attested to. The digest was written at upload and then
            # never read, so a member with no digest and a member whose digest was
            # simply never checked produced the same published checkpoint, and
            # neither was visible in anything the merge returned.
            #
            # A member that fails here is QUARANTINED, not fatal: it is excluded
            # from the soup, left in the pool, and recorded with a machine-readable
            # reason. The merge of the rest proceeds. Two deliberate consequences:
            #
            # - This runs AFTER soup_admission. Admission is a memory bound computed
            #   from recorded sizes and nothing is downloaded to evaluate it;
            #   verifying first would pull tens of gigabytes to the replica before
            #   the check that says the merge cannot fit in RAM. Admission over the
            #   unquarantined set is conservative — it refuses a superset — so the
            #   order is safe in the direction that matters.
            # - A quarantined CANDIDATE is still marked decided below, with the rest
            #   of the batch. Leaving it undecided would re-download and re-flag it
            #   on every future merge, and because undecided_contributions is
            #   chronological and max_batch caps the batch, a growing head of
            #   permanently-quarantined rows would starve real candidates out of
            #   every batch forever. "Kept in the pool" means its record and its
            #   weights stay in the artifact, not that it is re-weighed each time.
            verify_records = [by_id[cid] for cid in cur_member_ids] + list(batch)
            to_hash = pool.members_to_hash(verify_records)
            actual_by_path: Dict[str, str] = {}
            if to_hash:
                actual_by_path = _fold(
                    await runtime.verify_members(await _specs(to_hash))
                )["digests"]
            _, quarantined = pool.digest_quarantine(verify_records, actual_by_path)
            quarantined_ids = {q["contribution_id"] for q in quarantined}
            q_counts = pool.quarantine_counts(quarantined)
            floor = (
                int(min_members)
                if min_members is not None
                else max(1, len(cur_member_ids))
            )

            for q in quarantined:
                if q["reason"] == pool.QUARANTINE_DIGEST_MISMATCH:
                    # Not a skip. The digest IS recorded and the bytes in the pool do
                    # not hash to it, which means what is stored is not what was
                    # deposited. Logged at error with everything a human needs to
                    # open it as its own incident: which pool, which contribution,
                    # and both hashes.
                    logger.error(
                        f"🚨 INTEGRITY: pool '{pool_artifact_id}' contribution "
                        f"'{q['contribution_id']}' ({q['weights_path']}) hashes to "
                        f"{q['actual_sha256']} but the pool recorded "
                        f"{q['expected_sha256']}. Quarantined from this merge. The "
                        f"stored bytes are not the bytes that were contributed; this "
                        f"needs a human, not a retry."
                    )
                else:
                    logger.warning(
                        f"⚠️ Quarantined contribution '{q['contribution_id']}' "
                        f"({q['weights_path']}): {q['reason']} — the pool records "
                        f"{q['expected_sha256']!r} as its digest, which is not a "
                        f"sha256, so its bytes cannot be checked against anything."
                    )

            # The head's own members are checked too, and a head that cannot be
            # reconstituted stops the merge rather than quietly becoming a different
            # soup under the same name. See head_integrity_refusal for why this one
            # is fatal where a candidate's failure is not.
            head_refusal = pool.head_integrity_refusal(
                head_version=(cc or {}).get("version"),
                head_member_ids=cur_member_ids,
                quarantined_ids=quarantined_ids,
                floor=floor,
            )
            if head_refusal:
                raise ValueError(f"aggregate() refused this merge: {head_refusal}")

            batch_all_ids = [c["contribution_id"] for c in batch]
            cur_member_ids = [
                cid for cid in cur_member_ids if cid not in quarantined_ids
            ]
            cur_member_paths = [by_id[cid]["weights_path"] for cid in cur_member_ids]
            batch = [c for c in batch if c["contribution_id"] not in quarantined_ids]
            cand_path = {c["contribution_id"]: c["weights_path"] for c in batch}
            if quarantined:
                logger.info(
                    f"🔒 {len(quarantined)} member(s) quarantined from this merge "
                    f"({q_counts}); {len(cur_member_ids)} head member(s) and "
                    f"{len(batch)} candidate(s) carry a verified digest."
                )

            # Stock baseline on A and B (anchor of the witness curve — the level to beat).
            baseline = index.get("baseline")
            if baseline is None:
                stock_a = await _score([], gate_id)
                stock_b = await _score([], wit_id)
                baseline = {
                    "model": "stock-cpsam",
                    "selection_metric": _metric(
                        "selection", stock_a, pool.SELECTION_BASIS
                    ),
                    "witness_metric": _metric("witness", stock_b, pool.BASELINE_BASIS),
                    "created_at": time.time(),
                }

            cur = (
                await _score(cur_member_paths, gate_id)
                if cur_member_paths
                else _as_pass(baseline["selection_metric"])
            )
            cur_a = cur["f1"]

            # Best-first ordering: individual marginal of adding each candidate to the
            # current soup, on A. Re-scored against the CURRENT head (a stale solo
            # score misorders once the soup has grown).
            marg = {}
            for c in batch:
                cid = c["contribution_id"]
                marg[cid] = (
                    await _score(cur_member_paths + [cand_path[cid]], gate_id)
                )["f1"]
            ordered = sorted(
                batch, key=lambda c: marg[c["contribution_id"]], reverse=True
            )

            s_paths = list(cur_member_paths)
            s_ids = list(cur_member_ids)
            s_a = cur_a
            # The admitted head's own scoring pass, carried alongside its scalar: the
            # record published below must breakdown the number the gate actually
            # decided on, and re-scoring the head afterwards to recover a vector would
            # be a second pass that is not guaranteed to agree with the first.
            s_pass = cur
            members_record: List[Dict[str, Any]] = []
            admitted: List[str] = []
            for c in ordered:
                cid = c["contribution_id"]
                trial = await _score(s_paths + [cand_path[cid]], gate_id)
                trial_a = trial["f1"]
                included = trial_a >= (s_a - eps)
                members_record.append(
                    {
                        "contribution_id": cid,
                        "marginal_score": round(marg[cid], 4),
                        "gate_score": round(trial_a, 4),
                        "included": bool(included),
                    }
                )
                if included:
                    s_paths.append(cand_path[cid])
                    s_ids.append(cid)
                    s_a = trial_a
                    s_pass = trial
                    admitted.append(cid)

            changed = bool(admitted)
            baseline_version = (
                cc["version"] if (cc and cc.get("path")) else "stock-cpsam"
            )

            if changed:
                # Last gate before anything is written: a merge does not publish a
                # thinner soup than it was told to, and it says so rather than
                # shipping one quietly. With the default floor this cannot fire —
                # the head's survivors already cleared it above and admitting only
                # adds — so it is the explicit min_members that this enforces.
                thin = pool.publish_floor_refusal(
                    member_ids=s_ids, floor=floor, quarantined=quarantined
                )
                if thin:
                    raise ValueError(f"aggregate() refused this merge: {thin}")
                new_version = await self._next_free_community_version(
                    am, pool_artifact_id, index
                )
                head_a = s_a
                wit_pass = await _score(s_paths, wit_id)
                head_b = wit_pass["f1"]
                sel_metric = _metric("selection", s_pass, pool.SELECTION_BASIS)
                wit_metric = _metric("witness", wit_pass, pool.WITNESS_BASIS)
                paths = pool.community_paths(new_version)
                # Crash-safe: commit weights + manifest FIRST (index still points at
                # the old head), then repoint the index in a SECOND commit. The runtime
                # builds the soup and PUTs it straight to the pool, so the head weights
                # never traverse the CPU entry.
                async with self._staged(am, pool_artifact_id):
                    w_url = await am.put_file(
                        pool_artifact_id, file_path=paths["weights"]
                    )
                    mat = _fold(
                        await runtime.materialize_soup(
                            await _specs(s_paths),
                            w_url,
                            base_commitment=index.get("base_commitment"),
                        )
                    )
                    head_sha = mat["sha256"]
                    head_bytes_out = int(mat["n_bytes"])
                    # The contributors' own measured diameters go in
                    # the manifest. The runtime is given weights PATHS and never learns a
                    # contribution id, so the mapping is made here, where s_paths and
                    # s_ids are the parallel lists the gate built. An unmapped key RAISES
                    # rather than recording a null id: a diameter attributed to the wrong
                    # contributor, or to nobody, is worse than one not recorded, and this
                    # is the only place the correspondence is checkable.
                    # A runtime that predates the base commitment returns no such keys. entry and the
                    # Cellpose runtime are SEPARATE Ray deployments and can sit at
                    # different app versions during a rolling update, so this must not
                    # KeyError — and must not quietly write null either, because a reader
                    # has to tell "this runtime did not report them" from "this soup had
                    # no contributor diameters". The published weights are correct either
                    # way: the carry-through happens inside uniform_soup.
                    if "non_learned_inherited" in mat:
                        _cid_of = dict(zip(s_paths, s_ids))
                        member_nl = []
                        for _nl in mat.get("member_non_learned") or []:
                            _cid = _cid_of.get(_nl.get("key"))
                            if _cid is None:
                                raise RuntimeError(
                                    "materialize_soup returned a member key that is not "
                                    f"among the admitted paths: {_nl.get('key')!r}"
                                )
                            member_nl.append(
                                {
                                    "contribution_id": _cid,
                                    **{k: v for k, v in _nl.items() if k != "key"},
                                }
                            )
                        non_learned_block = pool.inherited_non_learned(
                            mat["non_learned_inherited"],
                            source=mat["non_learned_source"],
                        )
                    else:
                        member_nl = None
                        non_learned_block = pool.unreported_non_learned(
                            reason="the Cellpose runtime serving this merge returned no "
                            "non_learned_inherited field, so it predates the base commitment"
                        )
                    manifest = pool.build_community_manifest(
                        community_version=new_version,
                        combine="uniform",
                        selected_contribution_ids=s_ids,
                        gate_split=gate_split,
                        witness_split=witness_split,
                        selection_metric=sel_metric,
                        witness_metric=wit_metric,
                        members=members_record,
                        quarantined=quarantined,
                        architecture_signature=index.get("architecture_signature"),
                        weights_sha256=head_sha,
                        created_at=time.time(),
                        non_learned=non_learned_block,
                        member_non_learned=member_nl,
                    )
                    m_url = await am.put_file(
                        pool_artifact_id, file_path=paths["manifest"]
                    )
                    (
                        await self._http_retry(
                            "PUT",
                            m_url,
                            content=json.dumps(manifest).encode("utf-8"),
                            timeout=120.0,
                        )
                    ).raise_for_status()
                    await self._commit_with_retry(am, pool_artifact_id)
                self._bump_transport(bytes_up=head_bytes_out)
                pool.set_current_community(
                    index,
                    community_version=new_version,
                    weights_path=paths["weights"],
                    sha256=head_sha,
                    selection_metric=sel_metric,
                    witness_metric=wit_metric,
                )
            else:
                new_version = None
                head_a = cur_a
                head_b = (
                    cc["witness_metric"]["aggregate"]
                    if (cc and cc.get("path") and cc.get("witness_metric"))
                    else baseline["witness_metric"]["aggregate"]
                )
                head_sha = None
                head_bytes_out = 0
                # Nothing admitted: the selection metric records the bar the assessed
                # set failed to beat; there is no witness point (nothing re-measured).
                sel_metric = _metric("selection", cur, pool.EMPTY_SELECTION_BASIS)
                wit_metric = None

            soup_bytes_down = sum(replica_dl.values())
            if len(replica_dl) > 1:
                logger.warning(
                    f"⚠️ This merge was served by {len(replica_dl)} CellposeRuntime "
                    f"replicas, so at least one re-downloaded the member set: "
                    f"{replica_dl}"
                )
            self._bump_transport(bytes_down=soup_bytes_down)

            index["baseline"] = baseline
            # batch_all_ids, not batch: a quarantined candidate is decided too. See
            # the quarantine block above for why it is not left to be re-weighed.
            index["decided_contribution_ids"] = (
                list(index.get("decided_contribution_ids", [])) + batch_all_ids
            )
            # bytes_in counts the assessed set (the batch weighed at this merge);
            # bytes_out is the published head, 0 when nothing was admitted. n_bytes is
            # written per contribution in a different function, so if any assessed member
            # is missing it the sum is short — sources_complete goes null (fails closed,
            # page shows no per-record bytes) rather than presenting a partial as whole.
            member_bytes = [c.get("n_bytes") for c in batch]
            assessed_bytes = sum(int(nb) for nb in member_bytes if nb is not None)
            merge_transport = {
                "bytes_in": assessed_bytes,
                "bytes_out": head_bytes_out,
                "n_transfers": len(batch) + (1 if changed else 0),
                "sources_complete": (
                    True if all(nb is not None for nb in member_bytes) else None
                ),
                "soup_bytes_down": soup_bytes_down,
                "soup_replicas": len(replica_dl),
            }
            pool.append_merge(
                index,
                {
                    "published_version": new_version,
                    "baseline_version": baseline_version,
                    "members": members_record,
                    # Kept out of "members": a members row reads "evaluated, not
                    # included" and carries the gate scores that decision rests on. A
                    # quarantined member was never evaluated and has no such scores, so
                    # filing it there would report an integrity failure as a rejection.
                    "quarantined": quarantined,
                    "quarantine_counts": q_counts,
                    "selection_metric": sel_metric,
                    "witness_metric": wit_metric,
                    "global_sha256": head_sha,
                    # Written unconditionally, null included: a basis that is absent when
                    # there is no digest reads the same as one absent because the app
                    # predates the stamp, which is the ambiguity it exists to remove.
                    "global_sha256_basis": (
                        pool.CURRENT_DIGEST_BASIS if head_sha else None
                    ),
                    "transport": merge_transport,
                    "deferred": deferred_ids,
                    "created_at": time.time(),
                },
            )
            async with self._staged(am, pool_artifact_id):
                i_url = await am.put_file(pool_artifact_id, file_path=pool.POOL_INDEX)
                (
                    await self._http_retry(
                        "PUT",
                        i_url,
                        content=json.dumps(index).encode("utf-8"),
                        timeout=120.0,
                    )
                ).raise_for_status()
                await self._commit_with_retry(am, pool_artifact_id)
        finally:
            self._merges_in_flight -= 1
            lock.release()
            try:
                await asyncio.wait_for(client.disconnect(), timeout=2.0)
            except Exception:
                pass

        return {
            "schema_version": pool.SCHEMA_VERSION,
            "published_version": new_version,
            "admitted": admitted,
            "rejected": [
                m["contribution_id"] for m in members_record if not m["included"]
            ],
            # Reported separately from "rejected" and from "admitted", and always
            # present even when empty: a caller that wants to know whether this
            # merge excluded anything on provenance grounds must not have to infer
            # it from a count that only appears when it is non-zero.
            "quarantined": quarantined,
            "quarantine_counts": q_counts,
            "n_admitted_members": len(s_ids),
            "n_quarantined": len(quarantined),
            "deferred": deferred_ids,
            "gate_metric": round(head_a, 4),
            # The digest of the head this call just published, and which of the two
            # things it digests. This is the moment the false alarm happens —
            # someone re-merges an identical member set, gets a number, compares it
            # against a stored one and concludes corruption — so the attribution is
            # returned here, where both numbers are in the caller's hands, and not
            # only on a page they would have to go and read.
            "global_sha256": head_sha,
            "global_sha256_provenance": pool.digest_provenance(
                {
                    "global_sha256": head_sha,
                    "global_sha256_basis": pool.CURRENT_DIGEST_BASIS,
                }
            ),
            "witness_metric": round(head_b, 4) if head_b is not None else None,
            "baseline": baseline,
            "n_members": len(s_ids),
            # Passes this merge did not have to pay for because a previous attempt
            # had already scored them. Non-zero means a killed merge resumed rather
            # than restarted, which is the only caller-visible proof it did.
            "resumed_passes": resumed_from,
            "scored_passes": len(journal["scores"]) - resumed_from,
        }
