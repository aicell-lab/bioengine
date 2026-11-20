"""Tests for the cellpose training tile size: ``train_tile_size`` -> ``bsize``.

cellpose's ``train_seg`` takes a ``bsize`` (the square training tile side) that this
app never passed, so every cellpose fine-tune ran at whatever its backbone defaults
to.

The motivation this parameter was filed under -- that a smaller tile concentrates the
signal for narrow or small-object data and so improves the fit -- was MEASURED and did
not hold: on a synthetic trench corpus, AP rose monotonically with tile size (128 ->
0.804, 256 -> 0.887, 384 stock -> 0.889 at IoU 0.5). What the measurement does support
is a cost knob: 256 matched the 384 default to within noise at 27% less VRAM, and it ran
faster -- but the wall-clock side of that is INDICATIVE only, not a measured multiplier.
Those runs shared a GPU, and 890.9 s for 2.25x the pixels is superlinear, which points at
contention rather than at the tile size, so no speed-up figure is quoted -- here or in
the parameter's own description, which says only "trains faster and in less VRAM". The
parameter is documented that way, and these tests pin the wiring and the boundary, which
hold either way.

The parameter is **not** free on every model, and that asymmetry is the whole reason a
guard exists. cellpose/train.py (4.2.1.1, lines 360-363) reads:

    if bsize is not None and net.backbone == "sam_vitl" and bsize != 256:
        raise ValueError("bsize != 256 is not supported for cpsam, please set bsize to 256")
    bsize = 256 if bsize is None and net.backbone == "sam_vitl" else 384 if bsize is None else bsize

So cpsam (backbone ``sam_vitl``) accepts 256 and nothing else, cpdino is free, and the
no-bsize default is 256 only for cpsam -- it is **384** for a dino backbone. Upstream
raises that ValueError inside ``train_seg``, which runs after data prep and after the
run has already taken the single GPU training slot, so the refusal is made here at the
``start_training`` boundary instead.

Two things these tests pin that a reading of the diff does not:

*   The refusal fires **before** the runtime is touched. That is asserted structurally
    rather than by matching the message: the stub replica raises ``_PastTheGuard`` from
    ``_check_runtime_available``, the first thing after the guard, so "refused" and
    "accepted" are two different exception types and cannot be confused.
*   The value actually **reaches the params** the training subprocess reads. A silently
    dropped parameter completes exactly like the default, so "the run finished" is not
    evidence; ``test_the_tile_size_reaches_the_subprocess_params`` captures the dict.

Plain pytest -- no GPU, no Ray, no live Hypha, no cellpose. The guard and the wiring
are all CPU-side, so they are provable without any of them. What is NOT provable here
is that a smaller tile *improves* anything on real narrow data; that needs a GPU run on
a dino backbone against the 384 stock, and it is deliberately not simulated.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

import entry
import training

CELLPOSE_TYPES = ("cpsam", "cpdino", "cpdino-vitb")

# Every parameter is passed explicitly on purpose. Calling the underlying function
# directly bypasses the RPC layer that resolves pydantic ``Field`` defaults, so an
# omitted argument arrives as a ``FieldInfo`` object rather than its default value --
# which fails in a way that looks like a bug in the code under test.
BASE_KWARGS = dict(
    train_images=["i"], train_labels=["l"], val_images=None, val_labels=None,
    model_type="cpsam", n_epochs=1, n_objects_per_batch=8, patch_size=512,
    diam_mean=30.0, train_tile_size=None, batch_size=1, learning_rate=1e-5,
    val_fraction=0.2, n_samples=None, resume_session_id=None, init_checkpoint=None,
    init_from_pool=None, init_from_pool_version=None, pool_token=None, metrics=None,
    label="",
)


class _PastTheGuard(Exception):
    """Raised by the stub runtime check: the tile size was accepted."""


class _Replica:
    """Just enough of ``EntryApp`` to reach the tile-size guard, and no further."""

    def __init__(self) -> None:
        self._training_tasks: dict = {}

    async def _check_runtime_available(self, model_type=None) -> None:
        raise _PastTheGuard(model_type)


def _start_training(instance):
    """The real coroutine function off the Serve-wrapped class.

    ``entry.EntryApp`` is a ``ray.serve.deployment.Deployment`` once ``@bioengine.app``
    has run, so its methods are not reachable through it; ``func_or_class`` is the class
    the decorator wrapped.
    """
    return entry.EntryApp.func_or_class.start_training.__get__(instance)


async def _call(replica=None, **overrides):
    return await _start_training(replica or _Replica())(**{**BASE_KWARGS, **overrides})


# --- the boundary refusal ------------------------------------------------


@pytest.mark.asyncio
async def test_cpsam_refuses_a_tile_other_than_256() -> None:
    """The case upstream forbids, refused before the GPU slot is taken."""
    with pytest.raises(ValueError) as excinfo:
        await _call(model_type="cpsam", train_tile_size=128)

    message = str(excinfo.value)
    assert "train_tile_size" in message
    assert "256" in message
    # Our refusal, not cellpose's: the upstream wording never reaches the caller,
    # because the upstream check is never got to.
    assert "not supported for cpsam" not in message


@pytest.mark.asyncio
async def test_the_refusal_happens_before_the_runtime_is_touched() -> None:
    """Separately pinned: a guard placed after ``_check_runtime_available`` would
    still raise ValueError, and the test above would not notice."""
    with pytest.raises(ValueError):
        await _call(model_type="cpsam", train_tile_size=128)
    # _PastTheGuard is not a ValueError, so reaching the runtime check cannot satisfy
    # the assertion above -- but assert the accept path really does raise it, or the
    # discriminator is vacuous.
    with pytest.raises(_PastTheGuard):
        await _call(model_type="cpsam", train_tile_size=256)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", CELLPOSE_TYPES)
async def test_omitting_the_tile_is_accepted_for_every_cellpose_model(model_type) -> None:
    """``None`` keeps cellpose's own per-backbone default and must never be refused."""
    with pytest.raises(_PastTheGuard):
        await _call(model_type=model_type, train_tile_size=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", ("cpdino", "cpdino-vitb"))
@pytest.mark.parametrize("tile", (16, 128, 512))
async def test_a_dino_backbone_accepts_any_tile(model_type, tile) -> None:
    """The backbone where the parameter is free -- the only one the efficiency claim
    can ever be demonstrated on."""
    with pytest.raises(_PastTheGuard):
        await _call(model_type=model_type, train_tile_size=tile)


@pytest.mark.asyncio
async def test_the_guard_does_not_fire_for_a_microsam_model() -> None:
    """Scope check. ``train_tile_size`` is a cellpose-only parameter and micro-sam
    ignores it, exactly as it already ignores ``diam_mean``; the guard must not turn
    that into a refusal."""
    with pytest.raises(_PastTheGuard):
        await _call(model_type="vit_l_lm", train_tile_size=128)


def test_only_cpsam_has_a_fixed_tile() -> None:
    """The table the guard reads. If a future model type is added here, the refusal
    above extends to it automatically, so the table is worth pinning."""
    assert entry.CELLPOSE_FIXED_TILE_SIZE == {"cpsam": 256}


def test_the_parameter_is_advertised_as_cellpose_only() -> None:
    assert entry.TRAINING_PARAM_BACKENDS["train_tile_size"] == ["cellpose"]


# --- the value reaches the training subprocess ---------------------------


@pytest.fixture
def _sandbox_home(monkeypatch, tmp_path):
    """``training.sessions_root()`` hangs off ``Path.home()`` -- keep it in tmp_path."""
    monkeypatch.setenv("HOME", str(tmp_path))


class _CapturingReplica:
    """Runs ``start_training`` to the point where it hands params to the subprocess."""

    def __init__(self) -> None:
        self._training_tasks: dict = {}
        self.params: dict = {}

    async def _check_runtime_available(self, model_type=None) -> None:
        return None

    async def _runtime_gpu_memory(self, model_type=None) -> dict:
        return {"available": False}

    async def _run_training(self, session_id, model_type, *args, **kwargs) -> None:
        self.params = args[-1] if args else kwargs["params"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tile", (None, 128))
async def test_the_tile_size_reaches_the_subprocess_params(_sandbox_home, tile) -> None:
    """The assertion that a dropped parameter cannot pass.

    ``None`` is carried through as ``None`` rather than omitted, so the subprocess's
    ``p.get("train_tile_size")`` yields ``None`` and cellpose keeps its own default.
    """
    replica = _CapturingReplica()
    await _call(replica=replica, model_type="cpdino", train_tile_size=tile)
    await asyncio.sleep(0)  # let the created task run far enough to record params

    assert "train_tile_size" in replica.params
    assert replica.params["train_tile_size"] == tile


def test_the_worker_passes_it_to_train_seg_as_bsize() -> None:
    """Clause 2's citation, as a runnable check rather than a line reference.

    Read from the AST, not by grepping for ``bsize``: a grep matches the comment above
    the keyword, and a line number stops being true on the next edit.
    """
    source = (Path(__file__).parent / "train_worker_cellpose.py").read_text()
    tree = ast.parse(source)

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", getattr(node.func, "id", None)) == "train_seg"
    ]
    assert len(calls) == 1, f"expected exactly one train_seg call, found {len(calls)}"

    bsize = [kw for kw in calls[0].keywords if kw.arg == "bsize"]
    assert bsize, "train_seg is called without a bsize keyword, so the tile is ignored"
    assert "train_tile_size" in ast.get_source_segment(source, bsize[0].value)
