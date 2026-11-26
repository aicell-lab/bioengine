"""A base checkpoint this site cannot produce is refused BEFORE a GPU is taken.

Nothing in admission used to look at the base checkpoint. `start_training`
checked metrics, the tile size, the pool init and runtime availability;
`_runtime_for` raised only when the BACKEND was absent, which is a statement
about `MODEL_FINETUNE_BACKENDS` and says nothing about weights. The first thing
to consult the network about the base model ran inside the training worker --
`train_sam` with no `checkpoint_path` resolves it through micro-sam's own
registry at download time -- so the caller queued for the single GPU slot, got
it, and only then saw a download error.

TWO DIFFERENT WEIGHT SOURCES, and this is about the second. `RuntimeApp.
_zoo_weights` fetches pinned bioimage.io artifacts into a zoo cache for
INFERENCE, and `vit_t_em_organelles` is in that table -- so a probe aimed there
would look like evidence and pass while the training fetch still failed. The
training base model comes from micro-sam's own pooch registry, whose files land
under `<microsam_cachedir()>/models/<key>`, keyed by the model type for the
encoder and `<model_type>_decoder` for the AIS decoder.

THE CONTROL IS THE POINT OF HALF THESE CASES. A check that refused everything
would satisfy "an unavailable type is refused" perfectly, so every refusal case
here is paired with one that must still start.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

import training


def status(parts, error=None, model_type="vit_t_em_organelles"):
    return {"model_type": model_type, "parts": parts, "probe_error": error}


CACHED = {"cached": True, "reachable": None, "url": "https://example.invalid/x"}
FETCHABLE = {"cached": False, "reachable": True, "url": "https://example.invalid/x"}
UNREACHABLE = {"cached": False, "reachable": False, "url": "https://example.invalid/x"}
NO_URL = {"cached": False, "reachable": None, "url": None}


# ── refused, and the model type is named ────────────────────────────────────


@pytest.mark.parametrize(
    "parts",
    [
        {"encoder": UNREACHABLE, "decoder": UNREACHABLE},
        {"encoder": CACHED, "decoder": UNREACHABLE},
        {"encoder": UNREACHABLE, "decoder": CACHED},
        {"encoder": NO_URL, "decoder": CACHED},
    ],
    ids=["neither", "decoder-only", "encoder-only", "no-url"],
)
def test_a_part_this_site_cannot_produce_is_REFUSED(parts):
    """Every part must be obtainable, not just one.

    The decoder cases are not padding: the row was filed about a model type
    whose ENCODER is reachable and whose AIS decoder is not, and a check that
    looked only at the encoder would admit exactly that run.
    """
    msg = training.base_checkpoint_refusal("vit_t_em_organelles", status(parts))
    assert msg, f"{parts} was admitted"
    assert "vit_t_em_organelles" in msg, (
        "the refusal must NAME the model type — the caller's fix is to choose a "
        f"different one, and they cannot if it is not said: {msg!r}"
    )


def test_the_refusal_SAYS_NOTHING_WAS_STARTED():
    """The message has to distinguish itself from a mid-run failure, which is
    the thing this guard exists to stop being the user's experience."""
    msg = training.base_checkpoint_refusal(
        "vit_t_em_organelles", status({"encoder": UNREACHABLE})
    )
    assert "no GPU was claimed" in msg and "Nothing was started" in msg, msg


# ── the CONTROL: a type this site CAN produce still starts ──────────────────


@pytest.mark.parametrize(
    "parts",
    [
        {"encoder": CACHED, "decoder": CACHED},
        {"encoder": FETCHABLE, "decoder": FETCHABLE},
        {"encoder": CACHED, "decoder": FETCHABLE},
        {"encoder": CACHED},
    ],
    ids=["both-cached", "both-fetchable", "mixed", "encoder-only-published"],
)
def test_a_type_this_site_CAN_produce_is_ADMITTED(parts):
    """Without this, a check that refused everything would pass the refusal
    cases above and be indistinguishable from a working guard."""
    assert training.base_checkpoint_refusal("vit_b_lm", status(parts)) is None, parts


# ── it fails CLOSED on its own ignorance ────────────────────────────────────


@pytest.mark.parametrize(
    "bad",
    [
        status({}, error="ImportError: no module named micro_sam"),
        status({}),
        None,
        "not a mapping",
        status({"encoder": "unreadable"}),
    ],
    ids=["probe-error", "no-parts", "none", "not-a-dict", "part-not-a-dict"],
)
def test_an_UNANSWERABLE_probe_REFUSES_rather_than_admitting(bad):
    """"I could not tell" has to behave like "not available here".

    Admitting the run when the check could not answer restores exactly the
    behaviour being removed: the caller waits for a slot and finds out later.
    """
    msg = training.base_checkpoint_refusal("vit_t_em_organelles", bad)
    assert msg, f"{bad!r} was admitted"
    assert "vit_t_em_organelles" in msg, msg


# ── and it runs BEFORE the GPU gate, pinned structurally ────────────────────


def _start_training_source():
    import entry

    cls = getattr(entry.EntryApp, "func_or_class", entry.EntryApp)
    return inspect.getsource(cls.start_training)


def test_the_PREFLIGHT_RUNS_BEFORE_THE_GPU_GATE():
    """Order is the whole claim of this row, so it is asserted rather than
    assumed. A call placed after the GPU gate would still refuse the run and
    every test above would still pass, while the caller kept paying the wait
    this change exists to remove.
    """
    src = inspect.cleandoc(_start_training_source())
    tree = ast.parse(src)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                names.append((node.lineno, f.attr))
    order = [n for _, n in sorted(names)]
    assert "_refuse_unavailable_base_checkpoint" in order, (
        "start_training does not call the preflight at all; the guard is dead "
        f"code. Calls seen: {order}"
    )
    assert "_runtime_gpu_memory" in order, (
        "the GPU gate this is ordered against is gone — re-aim this test rather "
        f"than deleting it. Calls seen: {order}"
    )
    assert order.index("_refuse_unavailable_base_checkpoint") < order.index(
        "_runtime_gpu_memory"
    ), f"the preflight runs AFTER the GPU gate: {order}"


def test_the_preflight_is_SKIPPED_when_the_caller_brings_their_own_weights():
    """A caller supplying a checkpoint is not waiting on micro-sam's registry,
    so refusing them would refuse a run this site can actually do."""
    src = _start_training_source()
    assert "if not init_checkpoint and model_type not in CELLPOSE_MODEL_TYPES:" in src, (
        "the preflight is no longer conditioned on the caller NOT supplying "
        "weights and on the model type being a micro-sam one"
    )


def test_the_runtime_probe_NEVER_DOWNLOADS():
    """The cost this guard removes is a long wait, so a probe that downloaded
    the weights to find out would reproduce it."""
    src = pathlib.Path(
        pathlib.Path(__file__).parent / "runtime.py"
    ).read_text()
    i = src.index("async def base_checkpoint_status")
    body = src[i : src.index("\n    def _device", i)]
    for forbidden in ("pooch.retrieve", "fetch(", "urlretrieve", "get_sam_model"):
        assert forbidden not in body, (
            f"base_checkpoint_status calls {forbidden!r}, which downloads; the "
            "probe must only stat the cache and ask whether a URL answers"
        )
    assert "HEAD" in src, "the reachability probe no longer uses a HEAD request"
