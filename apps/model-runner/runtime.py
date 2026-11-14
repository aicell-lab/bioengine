"""GPU runtime for bioimage.io model inference.

The runtime is the GPU half of the model-runner app. ``EntryDeployment`` keeps a
type-hint reference to ``RuntimeDeployment`` so the v0.6 composition graph wires
them together; ``EntryDeployment`` then calls ``await self.runtime.ping()`` /
``await self.runtime.predict_from_disk(...)`` /
``await self.runtime.test(...)`` to delegate the heavy work. Inputs and
outputs for ``predict_from_disk`` stream through the shared PVC-backed
inference dir instead of over the RPC, so large arrays don't sit in RAM
while the request queues on ``_gpu_lock``.

Module-level imports stay deliberately lightweight (just stdlib +
bioengine) so the introspection task can load this file with only the
BioEngine baseline runtime_env. Heavy deps (``bioimageio.core``,
``careamics``, ``cellpose``, ``torch``, ``tensorflow``, …) are installed
by the ``@bioengine.app(pip=REQUIREMENTS)`` declaration and imported
inside method bodies (or, for prediction, inside the child process).
"""


import asyncio
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional

import bioengine

logger = logging.getLogger("ray.serve")
logger.setLevel("INFO")

# Filename key under which EntryDeployment.infer stages a bare (single, unnamed)
# input array. On read-back the runtime unwraps it to a bare array so
# bioimageio.core maps it to the model's sole input member id itself,
# rather than us guessing a member id that differs across spec versions.
SINGLE_INPUT_KEY = "__single_input__"


# Resident inference child: loads the model once from its argv identity,
# then loops handling predict requests (input_dir/output_dir/sample_id on
# stdin) until stdin closes. Responses (ready / ok / error) go on a private
# ``pass_fds`` pipe so chatty framework stdout can never be mis-parsed as a
# protocol message. Large arrays stay on disk — never over the pipe.
_WARM_CHILD_SCRIPT = r"""
import ctypes, json, os, signal, sys
from pathlib import Path
import numpy as np

# Kernel SIGKILLs this child the instant the parent replica dies — even
# mid-predict — so a VRAM-holding process can never be orphaned.
try:
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)
except Exception:
    pass

from bioimageio.core import create_prediction_pipeline, load_model_description
from bioimageio.core.digest_spec import create_sample_for_model
from bioimageio.spec.model.v0_5 import ParameterizedSize

rdf_path, load_params_path, resp_fd = sys.argv[1:4]
resp = os.fdopen(int(resp_fd), "w")


def _respond(obj):
    resp.write(json.dumps(obj) + "\n")
    resp.flush()


with open(load_params_path) as f:
    lp = json.load(f)
sys.path.insert(0, lp["app_dir"])
single_input_key = lp["single_input_key"]
default_blocksize_parameter = lp["default_blocksize_parameter"]
overrides = lp["overrides"]

# Tile size to aim for on loosely-sized Cellpose models, matching upstream
# cellpose's default for the DINO backbone (``models.py``: 256 for sam_vitl,
# else 384).
DINO_BLOCK_SIZE = 384


def _proc_id(op):
    return str(getattr(op, "id", None) or getattr(op, "name", ""))


def _apply_overrides(descr, overrides):
    # Per-request pre/postprocessing overrides. A dict patches the op's kwargs
    # on a deep copy of the description; ``None`` drops the op, which core can
    # only express as "skip this whole processing stage" — the declared tensor
    # shapes assume the ops ran, so a partial drop fails block reassembly.
    # Returns (description, skip_preprocessing, skip_postprocessing). The
    # artifact on disk is never touched; the copy lives only as long as this
    # child, which the parent evicts when the override set changes.
    pre = dict(overrides.get("preprocessing") or {})
    post = dict(overrides.get("postprocessing") or {})
    if not pre and not post:
        return descr, False, False

    declared = {"preprocessing": set(), "postprocessing": set()}
    for tensors, attr in (
        (descr.inputs, "preprocessing"),
        (descr.outputs, "postprocessing"),
    ):
        for tensor in tensors:
            for op in getattr(tensor, attr, None) or []:
                declared[attr].add(_proc_id(op))

    skip = {}
    for attr, updates in (("preprocessing", pre), ("postprocessing", post)):
        unknown = set(updates) - declared[attr]
        if unknown:
            raise ValueError(
                "model declares no " + attr + " op named: "
                + ", ".join(sorted(unknown))
            )
        dropped = {k for k, v in updates.items() if v is None}
        if dropped and dropped != set(updates):
            raise ValueError(
                "a request either patches or drops " + attr
                + " ops, not both: dropping one drops them all"
            )
        skip[attr] = bool(dropped)

    patched = descr.model_copy(deep=True)
    for tensors, updates, attr in (
        (patched.inputs, pre, "preprocessing"),
        (patched.outputs, post, "postprocessing"),
    ):
        if not updates or skip[attr]:
            continue
        for tensor in tensors:
            ops = getattr(tensor, attr, None) or []
            new_ops = list(ops)
            changed = False
            for i, op in enumerate(new_ops):
                op_id = _proc_id(op)
                if op_id not in updates:
                    continue
                kwargs = getattr(op, "kwargs", None)
                if kwargs is None:
                    raise ValueError(op_id + " takes no kwargs to override")
                new_ops[i] = op.model_copy(
                    update={"kwargs": kwargs.model_copy(update=updates[op_id])}
                )
                changed = True
            if changed:
                object.__setattr__(tensor, attr, type(ops)(new_ops))

    return patched, skip["preprocessing"], skip["postprocessing"]


def _is_cellpose(descr):
    for output in descr.outputs:
        for op in getattr(output, "postprocessing", None) or []:
            if "cellpose" in _proc_id(op).lower():
                return True
    return False


def _block_size_ns(descr, sample):
    # Cellpose-DINO declares y/x as ``min + n*step`` rather than a fixed size,
    # and core's default of n=10 would tile it at 144 px — well under the 384
    # the backbone expects. Capped at the sample's own extent, because core
    # rejects a block larger than the image. Empty for fixed-size Cellpose-SAM.
    ns = {}
    for ipt in descr.inputs:
        member = sample.members.get(ipt.id)
        for axis in getattr(ipt, "axes", None) or []:
            if getattr(axis, "type", None) != "space":
                continue
            if not isinstance(axis.size, ParameterizedSize):
                continue
            extent = None if member is None else member.sizes.get(axis.id)
            target = min(DINO_BLOCK_SIZE, extent or DINO_BLOCK_SIZE)
            n = (target - axis.size.min) // axis.size.step
            ns[(ipt.id, axis.id)] = max(0, n)
    return ns


def _fits_whole(descr, sample):
    # True when every space extent is already a size the model accepts and is
    # no larger than the block ``_block_size_ns`` targets, so there is nothing
    # to tile. A size kind we cannot reason about counts as not fitting, which
    # keeps the blocked path.
    for ipt in descr.inputs:
        member = sample.members.get(ipt.id)
        if member is None:
            return False
        for axis in getattr(ipt, "axes", None) or []:
            if getattr(axis, "type", None) != "space":
                continue
            extent = member.sizes.get(axis.id)
            if extent is None:
                return False
            if isinstance(axis.size, int):
                if extent != axis.size:
                    return False
            elif isinstance(axis.size, ParameterizedSize):
                if extent > DINO_BLOCK_SIZE:
                    return False
                if extent < axis.size.min:
                    return False
                if (extent - axis.size.min) % axis.size.step:
                    return False
            else:
                return False
    return True


loaded_description = load_model_description(rdf_path)
# Cellpose models are tile-based with a halo, so a sample that needs tiling
# must run blocked: a plain unblocked call pads the whole sample by the halo
# and breaks the ViT positional embedding. A sample that already fits whole
# takes the unblocked path with padding switched off instead — see the
# dispatch below. Everything else keeps the historical behaviour — blocked
# only when the caller asked for a block size. How a model has to run is a
# property of the model, so the flag is read before the overrides get a chance
# to drop the op it is recognised by.
cellpose = _is_cellpose(loaded_description)
model_description, skip_pre, skip_post = _apply_overrides(
    loaded_description, overrides
)

# Passing ``default_blocksize_parameter=None`` explicitly would override
# core's own default with None, and a blocked predict whose per-axis ``ns``
# is empty (every axis fixed) then falls back to it and crashes.
blocksize_kwargs = (
    {"default_blocksize_parameter": default_blocksize_parameter}
    if default_blocksize_parameter
    else {}
)
pipeline = create_prediction_pipeline(
    model_description,
    weights_format=lp["weights_format"],
    **blocksize_kwargs,
)
pipeline.load()
_respond({"status": "ready"})

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    try:
        inputs = {}
        for entry in sorted(Path(req["input_dir"]).iterdir()):
            if entry.is_file() and entry.suffix == ".npy":
                inputs[entry.stem] = np.load(str(entry))
        if not inputs:
            raise ValueError("No .npy input files found under " + req["input_dir"])
        # A bare single input arrives under the sentinel key; hand it to core
        # as a bare array so it maps to the model's sole input member id.
        # Explicit multi/named inputs pass through as-is.
        sample_inputs = (
            inputs[single_input_key]
            if list(inputs) == [single_input_key]
            else inputs
        )
        sample = create_sample_for_model(
            pipeline.model_description,
            inputs=sample_inputs,
            sample_id=req["sample_id"],
        )
        skips = {
            "skip_preprocessing": skip_pre,
            "skip_postprocessing": skip_post,
        }
        if default_blocksize_parameter:
            result = pipeline.predict_sample_with_blocking(sample, **skips)
        elif cellpose and _fits_whole(pipeline.model_description, sample):
            # Nothing to tile, and blocking would still pad the outer boundary
            # by the declared halo, which the model segments into extra cells
            # along the border. The skip flags are a matched pair: cropping
            # fires unconditionally and would remove a halo never added.
            result = pipeline.predict_sample_without_blocking(
                sample,
                skip_input_padding=True,
                skip_output_cropping=True,
                **skips,
            )
        elif cellpose:
            result = pipeline.predict_sample_with_blocking(
                sample,
                ns=_block_size_ns(pipeline.model_description, sample),
                **skips,
            )
        else:
            result = pipeline.predict_sample_without_blocking(sample, **skips)
        members = {
            str(key): member.data.data for key, member in result.members.items()
        }
        out = Path(req["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        for key, array in members.items():
            np.save(str(out / (key + ".npy")), array)
        _respond({"status": "ok"})
    except Exception as e:
        _respond({"status": "error", "message": repr(e)})
"""


_ARRAY_SHAPE_RE = re.compile(r"Array shape \((?P<shape>[^)]*)\) does not map to axes")
_AXIS_INFO_RE = re.compile(
    r"AxisInfo\(id='(?P<id>[^']*)', type='(?P<type>[^']*)', "
    r"size=AxisSize\(min=(?P<min>\d+|None), max=(?P<max>\d+|None), "
    r"step=(?P<step>\d+|None)\)\)"
)


def _axis_mismatch_hint(message: str) -> Optional[str]:
    """Turn bioimageio's axis ValueError into a sentence naming the remedy.

    The raw error is accurate and unactionable — it reprs the model's
    ``AxisInfo`` objects and leaves the caller to work out which axis is
    wrong and what to do. A participant hit exactly this in the 2026-11-11
    workshop with a grayscale image against Cellpose-SAM, whose channel
    axis is ``min=3, max=3`` (myco #0308).

    Returns ``None`` whenever anything is unrecognised, so an unparsed
    message falls through to the original text and is never made worse.

    The shape and the axis list are paired BY POSITION, so no axis is
    named when the two sequences have different lengths: bioimageio also
    tries permutations and singleton insertions, and pairing a 2-entry
    shape against 4 axes would name an axis at random. But "which axis"
    is only half the deliverable — the required sizes and the shape that
    arrived are still true on a rank mismatch, and that half is the
    actionable one. A grayscale file uploaded the ordinary way is exactly
    this case: ``imageio`` yields ``(H, W)`` and nothing between the
    upload and ``create_sample_for_model`` adds an axis, so the caller
    who prompted #0308 lands here rather than in the per-axis branch.
    """
    shape_m = _ARRAY_SHAPE_RE.search(message)
    if not shape_m:
        return None
    try:
        sizes = [
            int(part.strip())
            for part in shape_m.group("shape").split(",")
            if part.strip()
        ]
    except ValueError:
        return None

    axes = [
        (m.group("id"), m.group("type"), m.group("min"), m.group("max"))
        for m in _AXIS_INFO_RE.finditer(message)
    ]
    if not axes:
        return None

    wanted = ", ".join(
        f"{a}={lo_ if lo_ != 'None' else 1}"
        if lo_ == hi_
        else f"{a}>={lo_ if lo_ != 'None' else 1}"
        for a, _t, lo_, hi_ in axes
    )
    tail = (
        f" Required axis sizes: {wanted}; array shape given: "
        f"({', '.join(str(n) for n in sizes)}). Original error: {message}"
    )

    if len(axes) != len(sizes):
        missing = len(axes) - len(sizes)
        chan = [
            (i, int(lo_))
            for i, (a, _t, lo_, hi_) in enumerate(axes)
            if a == "channel" and lo_ == hi_ and lo_ != "None" and int(lo_) > 1
        ]
        remedy = ""
        # Only prescribe a concrete reshape when the axes the array is
        # missing are the LEADING ones — i.e. every axis it does supply is
        # a space axis. Otherwise the missing axes could be trailing and
        # ``arr[np.newaxis, ...]`` would put the data in the wrong place.
        if missing > 0 and chan and all(t == "space" for _a, t, _lo, _hi in axes[missing:]):
            c_idx, c_n = chan[0]
            if c_idx < missing:
                news = ", ".join(["np.newaxis"] * missing)
                remedy = (
                    f" A single-channel (grayscale) image sent as a plain "
                    f"{len(sizes)}-D array carries no channel axis at all, and this "
                    f"model wants {c_n}. Add the missing leading axes and replicate "
                    f"the channel before calling infer, e.g. "
                    f"np.repeat(arr[{news}], {c_n}, axis={c_idx}), or supply a "
                    f"{c_n}-channel image."
                )
        return (
            f"the array has {len(sizes)} dimensions but this model declares "
            f"{len(axes)} axes, so no single axis can be named.{remedy}{tail}"
        )

    for idx, ((axis_id, _type, lo, hi), got) in enumerate(zip(axes, sizes)):
        low = int(lo) if lo != "None" else None
        high = int(hi) if hi != "None" else None
        if (low is not None and got < low) or (high is not None and got > high):
            break
    else:
        return None

    if low is not None and low == high:
        need = f"exactly {low}"
    elif high is None:
        need = f"at least {low}"
    else:
        need = f"between {low} and {high}"

    remedy = ""
    if axis_id == "channel" and got == 1 and low == high:
        remedy = (
            f" The image is single-channel (grayscale); this model wants "
            f"{low}. Replicate the channel before calling infer, e.g. "
            f"np.repeat(arr, {low}, axis={idx}) "
            f"for an array in this axis order, or supply a {low}-channel image."
        )
    return (
        f"the {axis_id!r} axis must be {need} but the array supplies {got}."
        f"{remedy}{tail}"
    )


def _read_pip(name: str) -> List[str]:
    """Load a ``requirements-*.txt`` file next to this module.

    Keeps the heavy pin list out of the ``@bioengine.app`` decorator so
    the deps look like a real requirements file — Dependabot / pip-audit
    can point at the file directly and PR diffs isolate dep bumps.
    """
    text = (Path(__file__).parent / name).read_text()
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


#: Conda implementations a ``custom_environment=True`` test can be driven
#: through, in preference order. ``mamba`` first for the libmamba solver;
#: ``conda`` accepts the same ``env create`` / ``env remove`` / ``run -n`` /
#: ``list -n`` / ``clean`` arguments this app passes, so either satisfies us.
#: ``micromamba`` is deliberately absent — it takes ``create`` rather than
#: ``env create`` and would need its own argv shapes, so claiming support for
#: it here would be untested.
CONDA_IMPLEMENTATIONS = ("mamba", "conda")


def resolve_conda_binary() -> Optional[str]:
    """Name of the first conda implementation on PATH, or ``None`` if neither is.

    ``custom_environment=True`` is the only feature that needs one: it shells
    out to build the model's declared env and to run ``bioimageio test``
    inside it. **No image this repo builds installs one** —
    ``docker/worker.Dockerfile`` and ``docker/model-runner.Dockerfile`` are
    both ``FROM python:3.11-slim``, and measured in
    ``ghcr.io/aicell-lab/bioengine-worker:0.16.112`` all three of ``conda``,
    ``mamba`` and ``micromamba`` are absent (myco #0527). The code predates
    that: it was written against a ``rayproject/ray`` base, which ships
    anaconda at ``/home/ray/anaconda3`` — still visible in the recorded
    output of ``tutorial_model_runner.ipynb``, where ``conda list`` reports
    ``mamba 2.5.0`` in that prefix.

    So resolve rather than assume. Returning ``None`` lets the caller refuse
    the call naming what is missing, instead of spawning a binary that is not
    there and rendering the resulting ``FileNotFoundError`` as a failed
    validation report about the model.
    """
    for name in CONDA_IMPLEMENTATIONS:
        if shutil.which(name):
            return name
    return None


def pin_bioimageio_conda_env(conda_env, core_version: str, spec_version: str):
    """Return ``conda_env`` with ``bioimageio.core``/``.spec`` pinned to the
    given versions in its pip section; the model's own deps are untouched.

    An ``as-described`` custom env otherwise solves ``bioimageio.core>=0.9.4``
    fresh from conda-forge, so the harness the test actually runs under drifts
    from the RuntimeApp's own core/spec. Pinning makes a custom-env test run
    against the same core/spec a standard-env run uses. Applied identically by
    the RuntimeApp (wrapping ``get_conda_env`` while ``test_description`` builds
    each weight format's env) and the EntryApp pre-build
    (``_compute_conda_env_name``) so both derive the SAME env-name hash and the
    pre-built env is reused instead of cold-rebuilt. ``==`` on core also
    suppresses the upstream ``>=0.9.4`` floor; ``.spec`` is never touched by
    normalization.

    The env ``name`` is dropped so the hash depends only on channels +
    dependencies: two models declaring the same deps (even under different
    ``environment.yaml`` names) resolve to the same env and share it.
    """
    from bioimageio.spec.conda_env import BioimageioCondaEnv

    data = conda_env.model_dump(mode="python")
    data["name"] = None
    kept: List = []
    pip: List[str] = []
    for dep in data.get("dependencies", []):
        if isinstance(dep, dict) and "pip" in dep:
            pip = list(dep["pip"])
            continue
        bare = dep.split("::", 1)[-1] if isinstance(dep, str) else dep
        if isinstance(bare, str) and (
            bare.startswith("bioimageio.core") or bare.startswith("bioimageio.spec")
        ):
            continue
        kept.append(dep)

    pip = [
        p
        for p in pip
        if not (p.startswith("bioimageio.core") or p.startswith("bioimageio.spec"))
    ]
    pip += [f"bioimageio.core=={core_version}", f"bioimageio.spec=={spec_version}"]
    kept.append({"pip": pip})
    data["dependencies"] = kept
    return BioimageioCondaEnv.model_validate(data)


@bioengine.app(
    num_cpus=1,
    gpu_memory_mb=-1,
    memory_mb=12 * 1024,
    pip=_read_pip("requirements-runtime.txt"),
    # ``max_ongoing_requests=10`` intentionally lets requests queue at
    # the replica's asyncio queue behind ``_gpu_lock``. Ray Serve's
    # autoscaler only counts ongoing requests that reached a replica
    # (not those waiting at the router), so keeping this at 1 would
    # hide backlog and never trigger scale-up. With 10 and the internal
    # lock, the router forwards up to 10 concurrent RPCs per replica
    # and the lock serialises GPU work — autoscaler observes queue
    # depth. Precedent: apps/cell-image-search/main.py uses the same
    # pattern with max_ongoing_requests=10 and target=3.
    max_ongoing_requests=10,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 2,
        # Scale up when >3 ongoing per replica — with 1 replica and
        # 4 ongoing (1 running + 3 queued behind the lock), metric
        # 4/1 > 3 triggers a scale-up to 2. Then 4/2 = 2 ≤ 3 → stay.
        "target_ongoing_requests": 3.0,
        "metrics_interval_s": 2.0,
        "look_back_period_s": 10.0,
        # 10 min after the last scale-up condition clears before
        # scaling down — avoids thrashing on bursty traffic.
        "downscale_delay_s": 600,
        "upscale_delay_s": 0.0,
    },
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    # Bumped from 120s: with ``max_ongoing_requests=10`` a full backlog
    # may need several minutes to drain through the lock-serialised
    # GPU work before the replica exits cleanly.
    graceful_shutdown_timeout_s=300.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class RuntimeDeployment:
    """GPU-resident bioimage.io model executor."""

    # Per-request scratch on the app's shared PVC-backed HOME. EntryDeployment
    # writes ``input/<key>.npy`` here on ``infer()`` receipt; this app
    # reads inputs from disk, deletes the input dir, writes outputs to
    # ``output/<key>.npy``, and records step timestamps in
    # ``state.json``. Kept in sync with ``EntryDeployment._INFERENCE_DIR_NAME``.
    _INFERENCE_DIR_NAME = ".model-runner-inference"

    def __init__(self) -> None:
        # Serialises GPU work — predict + test both acquire this lock
        # so only one request touches the GPU at a time even when
        # ``max_ongoing_requests=10`` lets multiple RPCs reach the
        # replica. That extra queue is intentional: Ray Serve's
        # autoscaler needs to see it to scale up.
        self._gpu_lock = asyncio.Lock()
        # One resident inference child, reused for consecutive predicts of
        # the SAME model and evicted (killed → VRAM reclaimed) only on a
        # model switch / test / error / teardown. ``None`` when no child is
        # resident. Guarded entirely by ``_gpu_lock``. See ``_spawn_warm``.
        self._warm = None
        # Route bioimageio.spec + bioimageio.core log messages through
        # loguru. Their loggers are ``logger.disable()``-d by default
        # (standard convention for libraries), so per-weight-format
        # progress, conda subprocess spawns, and the like never surface
        # in the replica's stderr otherwise. Enabling once at replica
        # init is idempotent.
        try:
            from loguru import logger as _loguru_logger

            _loguru_logger.enable("bioimageio")
        except Exception as e:
            logger.warning(f"Could not enable bioimageio loguru sink: {e}")

    def _inference_dir(self) -> Path:
        """Base directory for per-request input/output/state files."""
        return Path(os.environ["HOME"]) / self._INFERENCE_DIR_NAME

    @staticmethod
    def _write_state_file(state_path: Path, state: Dict[str, float]) -> None:
        """Write ``state.json`` atomically via rename so the entry
        never observes a half-written file when polling.
        """
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(state_path)

    # === Liveness ===

    @bioengine.method
    async def ping(self) -> str:
        """Fast liveness probe used by EntryDeployment before every GPU-bound method."""
        return "pong"

    # === Memory accounting (used by test/predict log lines) ===

    def _get_memory_usage(self) -> tuple:
        """Return current ``(cpu_bytes, gpu_bytes)`` for this process."""
        import psutil

        cpu_mem = psutil.Process().memory_info().rss
        gpu_mem = 0
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_mem += info.used
            finally:
                pynvml.nvmlShutdown()
        except (ImportError, Exception):
            pass
        return cpu_mem, gpu_mem

    def _conda_list(self, env_names, env_vars) -> str:
        """Packages in the environment(s) the test actually ran in.

        Upstream leaves ``saved_conda_list`` to a lazy property that shells
        ``conda list`` with no target, so it reports whichever env is active in
        the serializing process — for us the replica's base anaconda, which
        contains none of the model's dependencies.

        ``test_description`` builds one env per weight format, so a multi-format
        model legitimately has several; all are listed rather than picking one.
        """
        import subprocess
        import sys

        def _run(cmd):
            proc = subprocess.run(
                cmd, capture_output=True, text=True, env=env_vars, check=False
            )
            if proc.returncode != 0:
                return f"`{' '.join(cmd)}` exited with {proc.returncode}"
            return proc.stdout

        try:
            if env_names:
                # Reached only from the custom-env path, which has already
                # refused if nothing resolves — but this writes a *provenance*
                # field, so say so plainly rather than spawning a missing
                # binary and reporting its errno as the model's environment.
                conda_bin = resolve_conda_binary()
                if conda_bin is None:
                    return (
                        "could not list the model's environment: none of "
                        f"{list(CONDA_IMPLEMENTATIONS)} is installed in this "
                        "replica's image"
                    )
                # Each ``conda list`` already opens with its own
                # "# packages in environment at <path>:" line, so that doubles as
                # the section delimiter — nothing needs inventing.
                return "\n\n".join(
                    _run([conda_bin, "list", "-n", name])
                    for name in sorted(env_names)
                )
            return _run([sys.executable, "-m", "pip", "list", "--format=freeze"])
        except Exception as e:
            return f"Failed to list packages: {e}"

    def _get_accelerator_name(self) -> str:
        """Name of the accelerator visible to THIS replica, or ``cpu``.

        Reads CUDA_VISIBLE_DEVICES before touching NVML: NVML enumerates every
        device injected into the container, so a CPU-only replica would
        otherwise name a GPU it cannot use. Deliberately not torch —
        ``get_device_name`` calls ``_lazy_init`` and leaves a primary CUDA
        context in a process whose whole design is to keep CUDA in child
        processes.
        """
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible:
            return "cpu"
        # Resolve the FIRST VISIBLE device, not physical index 0: Ray writes
        # this var from its own NVML enumeration, so reading it back through
        # NVML recovers the device Ray actually assigned.
        first = visible.split(",")[0].strip()
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                if first.startswith(("GPU-", "MIG-")):
                    handle = pynvml.nvmlDeviceGetHandleByUUID(first.encode())
                else:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(int(first))
                raw = pynvml.nvmlDeviceGetName(handle)
                return raw.decode() if isinstance(raw, bytes) else str(raw)
            finally:
                pynvml.nvmlShutdown()
        except Exception as e:
            logger.warning(f"⚠️ Could not resolve accelerator name: {e}")
            return "unknown"

    # === Subprocess env hardening ===

    _SENSITIVE_ENV_NEEDLES = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY")

    def _safe_subprocess_env(self) -> Dict[str, str]:
        """Return ``os.environ`` minus obviously-sensitive entries.

        Test subprocesses (bioimageio CLI, mamba) don't need the
        RuntimeDeployment's Hypha credentials. Denylist any env-var name
        containing ``TOKEN`` / ``SECRET`` / ``PASSWORD`` /
        ``CREDENTIAL`` / ``API_KEY`` — covers ``HYPHA_TOKEN``,
        ``BIOENGINE_ARTIFACT_TOKEN``, ``BIOIMAGE_IO_TOKEN``, and any
        generic cloud credentials leaking through. Everything else
        (``PATH``, ``HOME``, ``TMPDIR``, ``PYTHONPATH``, ``LANG``,
        ``CUDA_VISIBLE_DEVICES``, ``BIOENGINE_APP_DIR``,
        ``HYPHA_ARTIFACT_VERSION``) is preserved.
        """
        return {
            k: v
            for k, v in os.environ.items()
            if not any(needle in k.upper() for needle in self._SENSITIVE_ENV_NEEDLES)
        }

    def _mamba_env_vars(self) -> Dict[str, str]:
        """Scrubbed env with mamba's env-prefix + package cache
        redirected to the replica's writable HOME.

        A conda implementation's default env prefix and package cache
        (``$CONDA_PREFIX/envs/``, ``~/.mamba/pkgs``) are not writable
        cross-replica inside the Ray worker pod. HOME is
        set by ``bioengine._app.replica_init`` to
        ``<app_dir>/home/`` on the app's PVC, which is
        cross-replica RWX under the bioengine layout, so envs built
        by EntryDeployment are visible to this RuntimeDeployment on the same
        path.

        This used to cite ``/home/ray/anaconda3/envs/`` as the prefix being
        redirected away from. That is the ``rayproject/ray`` layout and no
        image this repo builds has it (myco #0527) — the redirect is still
        needed, just not for that reason.
        """
        env_vars = self._safe_subprocess_env()
        mamba_root = Path(os.environ["HOME"]) / ".bioengine-conda"
        (mamba_root / "envs").mkdir(parents=True, exist_ok=True)
        (mamba_root / "pkgs").mkdir(parents=True, exist_ok=True)
        env_vars["CONDA_ENVS_PATH"] = str(mamba_root / "envs")
        env_vars["CONDA_PKGS_DIRS"] = str(mamba_root / "pkgs")
        return env_vars

    # === Testing ===

    def _test(self, rdf_path: str, custom_environment: bool = False) -> dict:
        """Run ``bioimageio.core.test_description`` in a subprocess.

        **Both paths spawn a subprocess.** The point is GPU / CUDA
        context isolation: PyTorch caches allocator arenas inside
        its Python process's CUDA context and does not release them
        across calls even after ``torch.cuda.empty_cache()``. Running
        the model in the RuntimeDeployment's own process therefore leaves
        multi-GB of VRAM pinned across tests. Testing in a child
        process side-steps this — the OS reclaims the entire context
        (and all its VRAM) when the subprocess exits, no matter what
        state the model left behind.

        ``custom_environment=False`` (default):
            Spawn ``sys.executable -c <script>`` where the script
            imports ``test_description`` and runs it with
            ``runtime_env="currently-active"``. The child inherits
            the replica's venv via ``sys.executable`` + Ray's
            virtualenv layout, so the same package pins are in
            effect — this is the environment ``infer()`` will use in
            production. See ``_run_bioimageio_test_subprocess``.

        ``custom_environment=True``:
            Delegates to
            ``test_description(runtime_env="as-described")`` which
            already shells out through ``run_command`` (env create,
            env exec, ``bioimageio test`` inside the env). We rewrite
            ``conda`` on the first arg to whatever
            ``resolve_conda_binary()`` found — ``mamba`` when present,
            for the faster libmamba solver, else ``conda`` itself.
            **Neither is guaranteed:** this used to claim "both binaries
            ship in the replica image at ``/home/ray/anaconda3/bin/``",
            which was true of the ``rayproject/ray`` base and is false of
            every image this repo builds (myco #0527). So the branch
            raises up front when neither resolves, rather than spawning a
            missing binary. Envs are pre-built and
            cached on the shared PVC by the EntryDeployment (LRU eviction under
            a size ceiling), so this side normally finds them present
            and does not create or remove them.

        Enabling ``loguru`` output for the ``bioimageio`` namespace
        (spec + core) at replica init makes conda subprocess spawns
        and per-weight-format progress visible in the replica's
        stderr.
        """
        import subprocess

        if not Path(rdf_path).exists():
            raise FileNotFoundError(f"RDF not found: {rdf_path}")

        try:
            if not custom_environment:
                report = self._run_bioimageio_test_subprocess(rdf_path)
                if isinstance(report, dict):
                    # Overwritten post-hoc rather than assigned pre-dump: the
                    # summary is serialized in the child process, so the lazy
                    # property has already fired by the time we see the dict.
                    report["saved_conda_list"] = self._conda_list(
                        None, self._safe_subprocess_env()
                    )
                return report

            # custom_environment=True: run inside the model's declared
            # conda env via mamba. Env creation is now owned by the
            # EntryDeployment (``_prebuild_conda_envs``) so this replica's
            # GPU isn't held during a multi-minute mamba solve.
            # ``bioimageio.core.test_description(runtime_env="as-described")``
            # still does its own existence probe
            # (``mamba run -n <hash> python --version``) — that lands
            # on the PVC-backed HOME that Entry populated → env
            # found → mamba env create is skipped → straight to the
            # actual ``bioimageio test`` subprocess.
            #
            # Envs are cached across calls in
            # ``$HOME/.bioengine-conda/envs/`` so a second test of
            # the same model reuses the same env in seconds. No
            # auto-cleanup here — Entry decides lifecycle.
            #
            # Refuse before spawning anything. ``test_description`` shells out
            # to the conda implementation for its own env probe, so with none
            # installed the first thing that happens is
            # ``FileNotFoundError: 'mamba'`` from deep inside the call — which
            # the EntryDeployment then renders as a ``status="failed"``
            # *validation report about the model*, carrying the traceback as its
            # single unnamed detail. The model is not what failed. The
            # EntryDeployment refuses this call earlier (``_execute_test``);
            # this guard covers a direct ``runtime.test`` caller and keeps the
            # precondition stated where it is actually needed (myco #0527).
            conda_bin = resolve_conda_binary()
            if conda_bin is None:
                raise RuntimeError(
                    "custom_environment=True needs a conda implementation on "
                    f"PATH to build the model's declared environment, and none "
                    f"of {list(CONDA_IMPLEMENTATIONS)} is installed in this "
                    "replica's image. This is a limitation of the image, not a "
                    "fault in the model: re-run with custom_environment=False "
                    "to test in the serving runtime's own environment, or "
                    "deploy on a worker image that ships one."
                )

            from bioimageio.core import test_description

            mamba_env_vars = self._mamba_env_vars()

            solved_env_names = set()

            def mamba_run_command(args):
                args = list(args)
                if args and args[0] == "conda":
                    args[0] = conda_bin
                # core passes its own "conda run -n <hash> ..." through here, so
                # this is the only place the solved env's name is visible to us.
                if "-n" in args and args.index("-n") + 1 < len(args):
                    solved_env_names.add(args[args.index("-n") + 1])
                logger.info(f"🐍 [conda] running: {' '.join(args)}")
                proc = subprocess.run(
                    args, capture_output=True, text=True, env=mamba_env_vars
                )
                # Route child stdio through the replica logger — a
                # compact tail is enough to diagnose mamba failures
                # without flooding the log ring with libmamba trace
                # lines.
                if proc.stdout:
                    for line in proc.stdout.rstrip().splitlines()[-20:]:
                        logger.info(f"[mamba:stdout] {line}")
                if proc.stderr:
                    for line in proc.stderr.rstrip().splitlines()[-20:]:
                        logger.info(f"[mamba:stderr] {line}")
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(
                        proc.returncode,
                        args,
                        output=proc.stdout,
                        stderr=proc.stderr,
                    )

            # Deliberately omit ``expected_type`` here — the ``bioimageio
            # test`` CLI in some model envs does not recognise
            # ``--expected-type=<type>``, and ``test_description`` would
            # otherwise pass it into the subprocess and fail with
            # ``unrecognized arguments: --expected-type=model``. We know
            # ``model_id`` resolves to a model artifact (the
            # ``bioimage-io/model-runner`` service is scoped to models),
            # so losing the type assertion is not a real gap.
            #
            # Pin ``bioimageio.core``/``.spec`` in every weight-format env to
            # this replica's installed versions so the custom-env test runs
            # against the same harness a standard-env test does — rather than
            # whatever conda-forge resolves ``>=0.9.4`` to. ``test_description``
            # loops weight formats internally and calls the module-level
            # ``get_conda_env`` per format; wrap that name so each env is
            # pinned while the model's own deps stay as declared. The
            # EntryApp pre-build applies the identical pin, so the env-name
            # hash still matches and the pre-built env is reused. Safe to
            # patch the module global here: ``_test`` is the process's only
            # ``get_conda_env`` caller and runs serialized under ``_gpu_lock``.
            from importlib.metadata import version
            import bioimageio.core._resource_tests as _rt

            core_version = version("bioimageio.core")
            spec_version = version("bioimageio.spec")
            _orig_get_conda_env = _rt.get_conda_env

            def _pinned_get_conda_env(*, entry):
                return pin_bioimageio_conda_env(
                    _orig_get_conda_env(entry=entry), core_version, spec_version
                )

            _rt.get_conda_env = _pinned_get_conda_env
            try:
                validation_summary = test_description(
                    rdf_path,
                    runtime_env="as-described",
                    run_command=mamba_run_command,
                )
            finally:
                _rt.get_conda_env = _orig_get_conda_env

            # Assign BEFORE model_dump so the lazy ``conda_list`` property never
            # fires; overwriting afterwards would let it run first.
            validation_summary.saved_conda_list = (
                self._conda_list(solved_env_names, mamba_env_vars)
                if solved_env_names
                else "could not determine the model's environment"
            )
            return validation_summary.model_dump(mode="json")
        except Exception as e:
            logger.error(f"❌ Model test failed: {str(e)}")
            raise

    def _run_bioimageio_test_subprocess(self, rdf_path: str) -> dict:
        """Run ``test_description(runtime_env="currently-active")`` in a
        child Python process and return the summary dict.

        We spawn via ``sys.executable`` so the child lands in the same
        Ray-managed virtualenv as this replica (same package pins),
        then drive ``test_description`` via a small inline script. The
        subprocess writes the ``ValidationSummary`` model-dump to a
        JSON temp file and we read that back — round-tripping through
        JSON keeps the data flat and avoids any cross-process
        cloudpickle. RDF path and output path are passed as
        ``argv[1:]`` so the script has no interpolated string content
        (safe against filenames with quotes / backslashes).
        """
        import json
        import subprocess
        import sys
        import tempfile

        script = (
            "import json, sys\n"
            f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
            "from bioimageio.core import test_description\n"
            "rdf_path, out_path = sys.argv[1], sys.argv[2]\n"
            "summary = test_description(\n"
            "    rdf_path, expected_type='model', runtime_env='currently-active'\n"
            ")\n"
            "with open(out_path, 'w') as f:\n"
            "    json.dump(summary.model_dump(mode='json'), f)\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = str(Path(tmpdir) / "summary.json")
            logger.info(
                f"🐍 [test] Spawning bioimageio subprocess for CUDA "
                f"context isolation: {sys.executable} -c <inline> "
                f"{rdf_path} {summary_path}"
            )
            result = subprocess.run(
                [sys.executable, "-c", script, str(rdf_path), summary_path],
                capture_output=True,
                text=True,
                env=self._safe_subprocess_env(),
            )
            # Route child stdio through the replica's logger so
            # bioimageio's own log lines (progress, per-weight-format
            # status) surface here — without this, replica stderr is
            # silent about what the child actually did.
            if result.stdout:
                for line in result.stdout.rstrip().splitlines()[-40:]:
                    logger.info(f"[test:stdout] {line}")
            if result.stderr:
                for line in result.stderr.rstrip().splitlines()[-40:]:
                    logger.info(f"[test:stderr] {line}")

            summary_file = Path(summary_path)
            if not summary_file.exists():
                raise RuntimeError(
                    f"bioimageio test subprocess exited with code "
                    f"{result.returncode} without producing a summary "
                    f"(stderr tail: {(result.stderr or '')[-500:]!r})"
                )
            with summary_file.open() as f:
                return json.load(f)

    async def test(
        self,
        rdf_path: str,
        custom_environment: bool = False,
    ) -> dict:
        """Test model inference.

        When ``custom_environment=False`` (default), the test runs in
        the currently-active RuntimeDeployment venv — same interpreter that
        serves inference. Fast, and validates what the caller will
        actually use.

        When ``custom_environment=True``, the test runs inside the
        conda environment declared by the model's weights description
        (``bioimageio.core`` ``runtime_env="as-described"``). Env
        creation uses ``mamba`` (via a swapping ``run_command``); envs
        are pre-built and cached on the shared PVC by the EntryDeployment
        (LRU-evicted under a size ceiling), not removed per call.
        """
        # GPU serialisation: ``max_ongoing_requests=10`` allows requests
        # to queue at the replica so the autoscaler sees them, but only
        # one may hold the GPU at a time.
        async with self._gpu_lock:
            cpu_before, gpu_before = self._get_memory_usage()
            logger.info(
                f"📊 [test] Memory before: CPU: {cpu_before / (1024 * 1024):.2f} MB, "
                f"GPU: {gpu_before / (1024 * 1024):.2f} MB"
            )

            # The test child assumes a clean CUDA context. Evict any
            # resident warm inference child first so its VRAM is reclaimed
            # before the test subprocess loads the model — otherwise a
            # warm model would contend for VRAM with the tested one on a
            # time-sliced GPU (no per-replica VRAM isolation).
            if self._warm is not None:
                logger.info("🔄 [test] Evicting warm inference child before test")
            await asyncio.to_thread(self._evict_warm)

            # Run the sync ``_test`` in a thread so it doesn't block
            # this replica's asyncio loop. ``_test`` blocks on
            # ``subprocess.run`` / ``check_call`` (bioimageio subprocess
            # for the standard path, mamba env-create + ``bioimageio
            # test`` inside the env for the custom path). A multi-minute
            # mamba env build would otherwise starve Ray Serve's health
            # probes on this replica. ``asyncio.to_thread`` keeps
            # ``ping()`` responsive for the duration of the subprocess
            # wait (ping isn't guarded by the lock).
            test_report = await asyncio.to_thread(
                self._test, rdf_path, custom_environment
            )

            # Extend the report with a default-environment inference
            # smoke test: run the model on its own example input in THIS
            # replica's active venv (not the model's declared conda env)
            # to confirm it actually runs where infer() will serve it.
            # Fully guarded — any failure is recorded as a report field,
            # never propagated, so it cannot fail the test itself.
            # A model that declares its own environment does so because the
            # runner venv lacks its dependencies, so this check would fail on an
            # import every time and report nothing about the model (#0079).
            if custom_environment:
                # ``error`` rather than a new key because that is the field the
                # website renders in its failure dialog; leaving it null blanks
                # the message the user sees.
                inference_check = {
                    "status": "skipped",
                    "error": (
                        "model declares its own environment; the "
                        "default-environment check does not apply"
                    ),
                }
            else:
                try:
                    inference_check = await asyncio.to_thread(
                        self._run_inference_smoke_test, rdf_path
                    )
                except Exception as e:  # pragma: no cover - defensive backstop
                    inference_check = {
                        "status": "failed",
                        "error": f"inference smoke-test harness error: {e}",
                    }
            if isinstance(test_report, dict):
                test_report["inference_check"] = inference_check
                test_report["replica_accelerator"] = self._get_accelerator_name()

            cpu_after, gpu_after = self._get_memory_usage()
            logger.info(
                f"📊 [test] Memory after: CPU: {cpu_after / (1024 * 1024):.2f} MB, "
                f"GPU: {gpu_after / (1024 * 1024):.2f} MB"
            )
            return test_report

    def _run_inference_smoke_test(self, rdf_path: str) -> dict:
        """Run the model on its own declared test input in THIS replica's
        active venv (the default environment ``infer()`` serves in) and
        report whether it produced output.

        A *runs-at-all* check in the default env — distinct from the
        reproducibility test, which may run in the model's declared conda
        env and also checks output correctness. Runs in an isolated child
        process (CUDA-context cleanup) with a hard timeout so a hung model
        cannot pin the GPU lock. Returns ``{"status": "passed", "error":
        None}`` or ``{"status": "failed", "error": str}``; never raises.
        """
        import subprocess
        import sys

        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
            "from bioimageio.core import load_model_description, create_prediction_pipeline\n"
            "from bioimageio.core.digest_spec import get_test_inputs\n"
            "descr = load_model_description(sys.argv[1])\n"
            "pipeline = create_prediction_pipeline(descr)\n"
            "pipeline.load()\n"
            "sample = get_test_inputs(descr)\n"
            # Padding by the output halo grows the input, which a model declaring a
            # fixed input size cannot accept. The two flags are a matched pair:
            # cropping fires unconditionally and would remove a halo never added.
            "pipeline.predict_sample_without_blocking(\n"
            "    sample, skip_input_padding=True, skip_output_cropping=True\n"
            ")\n"
            "print('SMOKE_OK')\n"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", script, rdf_path],
                capture_output=True,
                text=True,
                env=self._safe_subprocess_env(),
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            return {"status": "failed", "error": "inference smoke test timed out"}
        except Exception as e:  # pragma: no cover - defensive
            return {
                "status": "failed",
                "error": f"could not launch inference smoke test: {e}",
            }

        if result.stdout:
            for line in result.stdout.rstrip().splitlines()[-20:]:
                logger.info(f"[smoke:stdout] {line}")
        if result.stderr:
            for line in result.stderr.rstrip().splitlines()[-20:]:
                logger.info(f"[smoke:stderr] {line}")
        if result.returncode == 0 and "SMOKE_OK" in (result.stdout or ""):
            return {"status": "passed", "error": None}
        stderr_tail = (result.stderr or "")[-800:]
        return {
            "status": "failed",
            "error": f"inference failed (returncode={result.returncode}): {stderr_tail}",
        }

    # === Prediction ===

    def _spawn_warm(
        self,
        key: tuple,
        rdf_path: str,
        load_params: Dict[str, object],
        python_env: Optional[str] = None,
    ) -> None:
        """Spawn a resident inference child that loads the model once and
        then serves predict requests until stdin closes.

        A child process is the only framework-agnostic way to guarantee
        the model's VRAM is fully reclaimed on eviction: the OS tears
        down the entire CUDA context on exit, releasing torch's caching
        allocator, TensorFlow's greedy whole-GPU grab, and onnxruntime's
        CUDA arena no matter what state the model left behind. Reusing
        the child across consecutive predicts of the *same* model keeps
        the loaded weights resident (no per-call ``pipeline.load()``);
        eviction on a model switch reclaims that VRAM before the next
        model loads.

        Blocks until the child signals the model finished loading, so the
        first predict never races the load and the cold-start cost is
        paid here (under ``_gpu_lock``). Stores the live child on
        ``self._warm``.
        """
        import collections
        import subprocess
        import tempfile
        import threading

        r_fd, w_fd = os.pipe()
        tail = collections.deque(maxlen=40)
        lp_dir = tempfile.mkdtemp(prefix="mr-warm-")
        lp_path = os.path.join(lp_dir, "load_params.json")
        with open(lp_path, "w") as f:
            json.dump(load_params, f)

        logger.info(
            f"🐍 [predict] Spawning resident inference child for {rdf_path}"
        )
        proc = subprocess.Popen(
            [
                self._interpreter(python_env),
                "-c",
                _WARM_CHILD_SCRIPT,
                rdf_path,
                lp_path,
                str(w_fd),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            pass_fds=(w_fd,),
            env=self._safe_subprocess_env(),
            text=True,
        )
        os.close(w_fd)
        resp = os.fdopen(r_fd, "r")

        def _drain() -> None:
            for line in proc.stdout:
                line = line.rstrip()
                tail.append(line)
                logger.info(f"[predict:child] {line}")

        reader = threading.Thread(target=_drain, daemon=True)
        reader.start()

        self._warm = {
            "proc": proc,
            "key": key,
            "stdin": proc.stdin,
            "resp": resp,
            "reader": reader,
            "tail": tail,
        }

        ready = resp.readline()
        if not ready or json.loads(ready).get("status") != "ready":
            tail_str = "\n".join(tail)[-800:]
            self._evict_warm()
            shutil.rmtree(lp_dir, ignore_errors=True)
            if "out of memory" in tail_str.lower():
                raise RuntimeError(
                    f"CUDA out of memory during inference: {tail_str}"
                )
            raise RuntimeError(
                f"Inference child failed to load model (stderr tail: {tail_str!r})"
            )
        shutil.rmtree(lp_dir, ignore_errors=True)

    def _send_predict(
        self,
        input_dir: str,
        output_dir: str,
        sample_id: str,
    ) -> None:
        """Send one predict request to the resident child and wait for its
        result. On child crash or error the child is evicted (a poisoned
        CUDA context is never reused) and the same ``RuntimeError`` the
        one-shot path used to raise is surfaced for the entry to handle.
        """
        warm = self._warm
        req = {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "sample_id": sample_id,
        }
        try:
            warm["stdin"].write(json.dumps(req) + "\n")
            warm["stdin"].flush()
        except (BrokenPipeError, OSError):
            line = ""
        else:
            line = warm["resp"].readline()

        if not line:
            tail_str = "\n".join(warm["tail"])[-800:]
            self._evict_warm()
            if "out of memory" in tail_str.lower():
                raise RuntimeError(
                    f"CUDA out of memory during inference: {tail_str}"
                )
            raise RuntimeError(
                f"Inference child exited during prediction (stderr tail: {tail_str!r})"
            )

        resp = json.loads(line)
        if resp.get("status") != "ok":
            message = resp.get("message", "")
            self._evict_warm()
            if "out of memory" in message.lower():
                raise RuntimeError(
                    f"CUDA out of memory during inference: {message}"
                )
            hint = _axis_mismatch_hint(message)
            if hint:
                raise RuntimeError(f"Inference failed: {hint}")
            raise RuntimeError(f"Inference failed: {message}")

    def _evict_warm(self) -> None:
        """Kill the resident inference child and reap it — reclaiming its
        VRAM. Safe to call when no child is resident.
        """
        warm = self._warm
        if warm is None:
            return
        self._warm = None
        proc = warm["proc"]
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=30)
        except Exception:
            pass
        for stream in ("stdin", "resp"):
            try:
                warm[stream].close()
            except Exception:
                pass
        reader = warm.get("reader")
        if reader is not None:
            reader.join(timeout=5)

    def _interpreter(self, python_env: Optional[str]) -> str:
        """Python that runs the inference child.

        ``None`` means the replica's own venv. A named env is a separate venv
        built on first use from ``requirements-<name>.txt``: Cellpose 3 and 4
        are one distribution name at two versions and cannot share a
        site-packages, so the only way to serve both from one GPU deployment
        is to run one of them under a different interpreter. Swapping the
        interpreter keeps ``pass_fds`` and the whole warm-child protocol
        intact, which wrapping the spawn in ``mamba run`` would not.
        """
        import sys

        if not python_env:
            return sys.executable
        return self._ensure_alt_venv(python_env)

    def _ensure_alt_venv(self, name: str) -> str:
        """Path to ``name``'s venv python on the shared HOME, building it once.

        Built in place rather than staged-then-renamed: a venv bakes its own
        absolute path into ``pyvenv.cfg`` and its console scripts, so moving
        it breaks it. A ``.ready`` marker written only after the install
        succeeds is what makes a half-built directory detectable, and an
        ``flock`` keeps two replicas — which share this HOME — from building
        into the same directory at once.
        """
        import fcntl
        import shutil
        import subprocess
        import sys

        req = Path(__file__).parent / f"requirements-{name}.txt"
        if not req.exists():
            raise RuntimeError(f"no requirements file for python_env {name!r}")

        root = Path(os.environ.get("HOME", "/tmp")) / ".model-runner-envs"
        root.mkdir(parents=True, exist_ok=True)
        target = root / name
        python = target / "bin" / "python"
        if (target / ".ready").exists():
            return str(python)

        with open(root / f".{name}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (target / ".ready").exists():
                return str(python)
            if target.exists():
                logger.warning(f"🧹 [{name}] Removing a half-built venv at {target}")
                shutil.rmtree(target, ignore_errors=True)
            logger.info(f"🐍 [{name}] Building inference venv at {target} (first use)")
            # Timeouts, because an install that never returns would otherwise
            # leave every later request waiting on this lock forever. A torch
            # wheel is gigabytes, so the install budget is generous.
            try:
                # No --system-site-packages: inheriting the replica's venv is
                # what would put the wrong Cellpose on the path.
                subprocess.run(
                    [sys.executable, "-m", "venv", str(target)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                subprocess.run(
                    [str(target / "bin" / "pip"), "install", "-r", str(req)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2700,
                )
            except subprocess.TimeoutExpired as e:
                shutil.rmtree(target, ignore_errors=True)
                raise RuntimeError(
                    f"building the {name} inference venv timed out after "
                    f"{e.timeout:.0f}s; no model needing it can run until a "
                    f"later request rebuilds it"
                ) from e
            except subprocess.CalledProcessError as e:
                shutil.rmtree(target, ignore_errors=True)
                raise RuntimeError(
                    f"building the {name} inference venv failed: "
                    f"{(e.stderr or '')[-800:]}"
                ) from e
            (target / ".ready").write_text("")
            logger.info(f"✅ [{name}] Inference venv ready at {target}")
        return str(python)

    async def predict_from_disk(
        self,
        request_id: str,
        rdf_path: str,
        weights_format: Optional[str] = None,
        default_blocksize_parameter: Optional[int] = None,
        sample_id: str = "sample",
        remote_modified: Optional[str] = None,
        force_reload: bool = False,
        overrides: Optional[Dict[str, Dict[str, Optional[dict]]]] = None,
        python_env: Optional[str] = None,
    ) -> None:
        """Read inputs from disk, run inference in a subprocess, write
        outputs to disk.

        Called by ``EntryDeployment._execute_infer`` after it has staged
        ``<inference_dir>/<request_id>/input/<key>.npy`` on the shared
        PVC. This method:

        1. Acquires ``self._gpu_lock`` so only one request runs on the
           GPU at a time — the extra RPCs sitting on the lock are what
           the Ray Serve autoscaler measures.
        2. Writes ``state.json`` with ``runtime_started_at`` so the
           entry can distinguish "still queued at runtime" from
           "actively running" when computing ``queue_position``.
        3. Runs the model in a resident child process (``_spawn_warm`` /
           ``_send_predict``) which reads the inputs, predicts, and
           writes ``output/<key>.npy``. The child is reused for
           consecutive predicts of the same model (identity ``key``
           below) and evicted — reclaiming all of the model's VRAM
           regardless of framework — on a model switch.
        4. Deletes the input dir and records ``runtime_completed_at``.

        The entry's poll (``get_infer_status``) reads the outputs off
        disk and deletes the request dir after the caller collects them.
        """
        request_dir = self._inference_dir() / request_id
        input_dir = request_dir / "input"
        output_dir = request_dir / "output"
        state_file = request_dir / "state.json"

        # Build the alternate venv BEFORE taking the lock. It touches no GPU,
        # and a multi-gigabyte pip install held under ``_gpu_lock`` would
        # stall every other request on this replica for its whole duration.
        if python_env:
            await asyncio.to_thread(self._ensure_alt_venv, python_env)

        async with self._gpu_lock:
            cpu_before, gpu_before = self._get_memory_usage()
            logger.info(
                f"📊 [predict] Memory before: CPU: {cpu_before / (1024 * 1024):.2f} MB, "
                f"GPU: {gpu_before / (1024 * 1024):.2f} MB"
            )

            # Signal to the entry that this request has left the
            # (Ray Serve replica-side) queue and is now on the GPU.
            # Written before any expensive work so the ``running``
            # timestamp reflects the true start.
            state: Dict[str, float] = {"runtime_started_at": time.time()}
            await asyncio.to_thread(self._write_state_file, state_file, state)

            if not await asyncio.to_thread(input_dir.is_dir):
                raise FileNotFoundError(
                    f"Input directory missing for request {request_id!r}: "
                    f"{input_dir}"
                )
            if not await asyncio.to_thread(Path(rdf_path).exists):
                raise FileNotFoundError(f"RDF not found: {rdf_path}")

            logger.info(
                f"🚀 Starting prediction for model at {rdf_path} with "
                f"weights_format={weights_format}"
            )
            # Identity of the resident child. ``remote_modified`` is the
            # cache's own staleness token (entry passes
            # ``package.latest_remote_modified``): a genuine model update
            # changes it → key mismatch → evict + reload, so the warm
            # pipeline can never outlive a cache refresh.
            # ``overrides`` patch the model description the pipeline is built
            # from, so they belong in the identity: a changed override set must
            # reload rather than silently reuse a differently-configured child.
            key = (
                rdf_path,
                remote_modified,
                weights_format,
                default_blocksize_parameter,
                json.dumps(overrides or {}, sort_keys=True),
                python_env,
            )
            # ``force_reload`` (entry passes it for cache="skip") bypasses reuse
            # so a byte-identical re-download — which leaves ``remote_modified``
            # unchanged, hence the key unchanged — still reloads the model.
            warm = self._warm
            reuse = (
                warm is not None
                and warm["proc"].poll() is None
                and warm["key"] == key
                and not force_reload
            )
            if reuse:
                logger.info(f"♻️ [predict] Reusing warm inference child for {rdf_path}")
            else:
                if warm is not None:
                    reason = (
                        "forced reload (cache=skip)"
                        if force_reload
                        else "model switch"
                        if warm["key"] != key
                        else "warm child exited"
                    )
                    logger.info(
                        f"🔄 [predict] Evicting warm child ({reason}) before "
                        f"loading {rdf_path}"
                    )
                await asyncio.to_thread(self._evict_warm)
                load_params: Dict[str, object] = {
                    "weights_format": weights_format,
                    "default_blocksize_parameter": default_blocksize_parameter,
                    "single_input_key": SINGLE_INPUT_KEY,
                    "overrides": overrides or {},
                    "app_dir": str(Path(__file__).parent),
                }
                await asyncio.to_thread(
                    self._spawn_warm, key, rdf_path, load_params, python_env
                )
            await asyncio.to_thread(
                self._send_predict,
                str(input_dir),
                str(output_dir),
                sample_id,
            )

            # Free the input images now the child has consumed them.
            await asyncio.to_thread(
                shutil.rmtree, str(input_dir), ignore_errors=True
            )

            cpu_after, gpu_after = self._get_memory_usage()
            logger.info(
                f"📊 [predict] Memory after: CPU: {cpu_after / (1024 * 1024):.2f} MB, "
                f"GPU: {gpu_after / (1024 * 1024):.2f} MB"
            )

            state["runtime_completed_at"] = time.time()
            await asyncio.to_thread(self._write_state_file, state_file, state)
