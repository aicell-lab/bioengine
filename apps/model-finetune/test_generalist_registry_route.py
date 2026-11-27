"""A MIRRORED generalist loads through micro-sam's registry.

WHAT THIS IS ABOUT. Generalist annotation -- no fine-tuned checkpoint --
fetched its weights from the bioimage.io zoo over a host this deployment cannot
reach, with `known_hash=None` on both files. The training path already resolves
base weights through micro-sam's registry, which is seeded from a verified
mirror and fences with a pinned sha256. Branch A puts the generalist on that
same path.

SCOPED, AND THE SCOPE IS ASKED OF THE MIRROR. Only the model types the mirror
holds move; the rest keep the zoo path untouched. The predicate asks
`base_checkpoint_mirror.parts_for`, so a second list cannot go stale as the
mirror grows.

THE OBSERVABLE THAT MATTERS IS *WHICH SEGMENTATION MODE COMES OUT*, not which
function got called. Today's code pre-loads the two zoo files and hands
micro-sam an assembled state, and its comment says that is necessary because
`get_predictor_and_segmenter` takes no `decoder_path`. If the registry path
produced no decoder, branch A would silently downgrade annotation from AIS to
AMG. Measured in the PINNED micro-sam 1.8.11 instead of assumed:

    _download_sam_model:  decoder_name = f"{model_type}_decoder"
                          fetched when the registry holds that key
    get_sam_model:        state["decoder_state"] = torch.load(decoder_path)
    get_predictor_...:    AIS when "decoder_state" in state, else "amg"

so the registry path keeps AIS. The stubs below reproduce exactly that rule,
and one test asserts the mode rather than the call, because the mode is what a
user would notice.

No torch, no micro-sam, no Ray: micro-sam is a RUNTIME dependency and is not
importable here, so it is injected through `sys.modules`.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

MIRRORED = "vit_b_lm"
UNMIRRORED = "vit_l_em_organelles"


def _runtime_cls():
    import runtime

    return getattr(runtime.RuntimeApp, "func_or_class", runtime.RuntimeApp)


class _Calls:
    def __init__(self):
        self.get_sam_model = []
        self.get_predictor_and_segmenter = []
        self.zoo_weights = []


@pytest.fixture()
def stubbed(monkeypatch):
    """micro-sam stubbed to the pinned version's DECODER RULE, not to a guess.

    `get_sam_model` puts `decoder_state` in the state when it has a decoder
    path -- supplied by the caller, or resolved from the registry when the
    caller passed no checkpoint. `get_predictor_and_segmenter` then picks AIS
    on that key. Both halves are the pinned 1.8.11 behaviour, read from the
    wheel; a stub that always produced a decoder could not tell branch A's
    path from a silent downgrade.
    """
    calls = _Calls()
    registry = {"vit_b_lm", "vit_b_lm_decoder", "vit_t_lm", "vit_t_lm_decoder"}

    def get_sam_model(
        model_type=None,
        checkpoint_path=None,
        decoder_path=None,
        device=None,
        return_state=False,
        **kw,
    ):
        calls.get_sam_model.append(
            {
                "model_type": model_type,
                "checkpoint_path": checkpoint_path,
                "decoder_path": decoder_path,
            }
        )
        if checkpoint_path is None and decoder_path is None:
            # _download_sam_model's rule, verbatim in effect.
            if f"{model_type}_decoder" in registry:
                decoder_path = f"<registry:{model_type}_decoder>"
        state = {"model_state": "..."}
        if decoder_path is not None and return_state:
            state["decoder_state"] = "..."
        predictor = types.SimpleNamespace(name=f"predictor:{model_type}")
        return (predictor, state) if return_state else predictor

    def get_predictor_and_segmenter(
        model_type=None,
        checkpoint=None,
        device=None,
        segmentation_mode=None,
        predictor=None,
        state=None,
        **kw,
    ):
        if predictor is None:
            predictor, state = get_sam_model(
                model_type=model_type,
                checkpoint_path=checkpoint,
                device=device,
                return_state=True,
            )
        mode = segmentation_mode
        if mode in (None, "auto"):
            mode = "ais" if "decoder_state" in (state or {}) else "amg"
        calls.get_predictor_and_segmenter.append(
            {"model_type": model_type, "checkpoint": checkpoint, "mode": mode}
        )
        return predictor, types.SimpleNamespace(mode=mode)

    util = types.ModuleType("micro_sam.util")
    util.get_sam_model = get_sam_model
    auto = types.ModuleType("micro_sam.automatic_segmentation")
    auto.get_predictor_and_segmenter = get_predictor_and_segmenter
    pkg = types.ModuleType("micro_sam")
    pkg.util = util
    pkg.automatic_segmentation = auto
    monkeypatch.setitem(sys.modules, "micro_sam", pkg)
    monkeypatch.setitem(sys.modules, "micro_sam.util", util)
    monkeypatch.setitem(sys.modules, "micro_sam.automatic_segmentation", auto)
    return calls


def _app(calls, monkeypatch):
    cls = _runtime_cls()
    inst = object.__new__(cls)
    inst._loaded_key = None
    inst._predictor = None
    inst._segmenter = None
    inst._loaded_model_type = None
    monkeypatch.setattr(cls, "_device", lambda self: "cpu", raising=False)
    monkeypatch.setattr(cls, "_release_model", lambda self: None, raising=False)

    def _zoo(self, model_type):
        calls.zoo_weights.append(model_type)
        return f"<zoo:{model_type}>", f"<zoo:{model_type}_decoder>"

    monkeypatch.setattr(cls, "_zoo_weights", _zoo, raising=False)
    return cls, inst


# --------------------------------------------------------------------------
# The routing decision
# --------------------------------------------------------------------------


def test_a_MIRRORED_generalist_does_NOT_touch_the_zoo(stubbed, monkeypatch):
    cls, inst = _app(stubbed, monkeypatch)

    cls._ensure_model(inst, MIRRORED)

    assert stubbed.zoo_weights == [], (
        f"the zoo path was used for {MIRRORED}, which the mirror holds: "
        f"{stubbed.zoo_weights}"
    )
    asked = stubbed.get_predictor_and_segmenter[-1]
    assert asked["model_type"] == MIRRORED and asked["checkpoint"] is None, asked


def test_an_UNMIRRORED_generalist_STILL_uses_the_zoo(stubbed, monkeypatch):
    """The control. A change that moved everything would pass the test above."""
    cls, inst = _app(stubbed, monkeypatch)

    cls._ensure_model(inst, UNMIRRORED)

    assert stubbed.zoo_weights == [UNMIRRORED], (
        f"{UNMIRRORED} has no mirrored copy, so moving it swaps one unreachable "
        f"host for another: {stubbed.zoo_weights}"
    )


def test_a_FINETUNED_checkpoint_is_untouched_by_this(stubbed, monkeypatch):
    cls, inst = _app(stubbed, monkeypatch)

    cls._ensure_model(inst, MIRRORED, checkpoint="/runs/x/best.pt")

    assert stubbed.zoo_weights == []
    asked = stubbed.get_predictor_and_segmenter[-1]
    assert asked["checkpoint"] == "/runs/x/best.pt", asked


# --------------------------------------------------------------------------
# The consequence a user would notice
# --------------------------------------------------------------------------


def test_the_MIRRORED_route_STILL_SELECTS_AIS(stubbed, monkeypatch):
    """The whole risk of branch A in one assertion.

    If the registry path produced no decoder, the mode would come out "amg"
    and annotation quality would change with nothing reporting it.
    """
    cls, inst = _app(stubbed, monkeypatch)

    _, segmenter = cls._ensure_model(inst, MIRRORED)

    assert segmenter.mode == "ais", (
        "the mirrored generalist came back in AMG mode, so the registry resolve "
        "produced no decoder and branch A is a silent downgrade"
    )


def test_a_model_the_REGISTRY_has_NO_decoder_for_falls_back_to_AMG(
    stubbed, monkeypatch
):
    """The discriminating control for the test above.

    A stub that always returned a decoder would make that assertion vacuous.
    This asks for a type whose decoder the registry does not hold and requires
    the mode to differ — so the AIS result above is the decoder's doing.
    """
    cls, inst = _app(stubbed, monkeypatch)

    _, segmenter = cls._ensure_model(inst, "vit_h", checkpoint=None)

    assert segmenter.mode == "amg", (
        "a model with no registry decoder still reported AIS, so the stub is not "
        "discriminating and the AIS assertion proves nothing"
    )


# --------------------------------------------------------------------------
# The predicate, and the seam that seeds the mirror
# --------------------------------------------------------------------------


def test_the_PREDICATE_FOLLOWS_the_MIRROR_when_the_mirror_MOVES(monkeypatch):
    """The predicate must be DERIVED from the mirror, not a copy of it.

    Comparing the predicate against `parts_for` with the mirror as it is cannot
    establish that: a hard-coded `("vit_b_lm", "vit_t_lm")` satisfies it exactly,
    because that pair EQUALS the mirror today. A review made this point and it is
    right -- the old version of this arm passed against both implementations, so
    it measured nothing about where the set comes from.

    So the mirror is MOVED and the predicate is required to move with it. Both
    directions, because one is not enough: a predicate returning True for
    everything would follow the mirror gaining a type and fail only on the
    removal, and one returning False for everything the reverse. Together they
    leave no constant-valued implementation passing.

    Measured against both implementations before being written: the derived one
    passes, the hard-coded one fails on the first move.
    """
    import base_checkpoint_mirror as bcm
    import runtime

    ADDED = "vit_l_lm"  # a generalist the mirror does NOT hold today
    REMOVED = "vit_b_lm"  # one it does
    assert ADDED in runtime.ZOO_MODELS and REMOVED in runtime.ZOO_MODELS
    assert not bcm.parts_for(ADDED), f"{ADDED} is mirrored now; pick another"
    assert bcm.parts_for(REMOVED), f"{REMOVED} is not mirrored; pick another"

    patched = dict(bcm.MIRROR_CHECKPOINTS)
    patched[ADDED] = ("x", "y", 1)
    patched[f"{ADDED}_decoder"] = ("x", "y", 1)
    patched.pop(REMOVED, None)
    patched.pop(f"{REMOVED}_decoder", None)
    # monkeypatch so the mirror is restored even if an assertion below fails --
    # a leaked mutation would silently corrupt every later arm in this session.
    monkeypatch.setattr(bcm, "MIRROR_CHECKPOINTS", patched)

    assert runtime._generalist_uses_registry(ADDED) is True, (
        "the predicate did not follow the mirror GAINING a type, so it is not "
        "derived from it"
    )
    assert runtime._generalist_uses_registry(REMOVED) is False, (
        "the predicate did not follow the mirror LOSING a type, so it is not "
        "derived from it"
    )


def test_the_PREDICATE_agrees_with_the_mirror_AS_IT_IS():
    """The unmoved check, kept as well -- it is still worth asserting that the
    predicate is correct today, it just cannot establish WHERE the set comes
    from. That is the arm above."""
    import base_checkpoint_mirror as bcm
    import runtime

    for model_type in sorted(runtime.ZOO_MODELS):
        expected = bool(bcm.parts_for(model_type))
        assert runtime._generalist_uses_registry(model_type) is expected, model_type
    # And a type that is not a generalist at all is not routed by this.
    assert runtime._generalist_uses_registry("vit_b") is False


def test_EVERY_composition_ENDPOINT_seeds_the_base_FIRST(monkeypatch):
    """A seam missed on one endpoint is an endpoint that cannot work offline.

    Asserted on the SOURCE of each endpoint rather than by calling them,
    because calling them needs a GPU lock, numpy arrays and a thread per case;
    what matters here is that no endpoint reaches `to_thread` without the seed.
    """
    import ast
    import inspect

    import runtime

    cls = _runtime_cls()
    src = inspect.getsource(cls)
    tree = ast.parse(ast.unparse(ast.parse(src)))
    body = tree.body[0].body
    endpoints = {
        "auto_segment",
        "auto_segment_from_embedding",
        "encode",
        "export_onnx",
    }
    seen = {}
    for node in body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name in endpoints:
            text = ast.unparse(node)
            seen[node.name] = "_seed_generalist_base" in text
    assert set(seen) == endpoints, f"endpoints not found: {endpoints - set(seen)}"
    missing = sorted(n for n, ok in seen.items() if not ok)
    assert not missing, f"these endpoints never seed the base checkpoint: {missing}"


def test_the_SEED_is_a_NO_OP_for_an_unmirrored_type_and_for_a_checkpoint(monkeypatch):
    cls = _runtime_cls()
    inst = object.__new__(cls)
    called = []

    async def _ensure(model_type, **kw):
        called.append(model_type)
        return {"unavailable": []}

    inst.ensure_base_checkpoint = _ensure

    asyncio.run(cls._seed_generalist_base(inst, UNMIRRORED))
    asyncio.run(cls._seed_generalist_base(inst, MIRRORED, checkpoint="/x/best.pt"))
    assert called == [], f"the seed ran where it has nothing to seed: {called}"

    asyncio.run(cls._seed_generalist_base(inst, MIRRORED))
    assert called == [MIRRORED], called


def _seed_inst(exc):
    """A runtime whose `ensure_base_checkpoint` raises *exc*, and nothing else."""
    cls = _runtime_cls()
    inst = object.__new__(cls)

    async def _raise(model_type, **kw):
        raise exc

    inst.ensure_base_checkpoint = _raise
    return cls, inst


def test_a_TRANSPORT_FAILURE_does_not_fail_the_request():
    """`ensure_base_checkpoint` reports an unobtainable part; the refusal a
    caller deserves comes from the admission check, with the part named. A
    raise here would fail annotation on a site whose cache already holds the
    weights.

    The exception is a `ConnectionError` ON PURPOSE. An earlier version of this
    arm raised a plain `RuntimeError`, which is `MirrorVerificationError`'s BASE
    class -- so it did not merely miss the swallowed-fence defect below, it
    CODIFIED it: a correct implementation that re-raised the fence error passed
    this arm unchanged. Using a transport-shaped error means this case and the
    fence case cannot be satisfied by one exception.
    """
    cls, inst = _seed_inst(ConnectionError("uk1s3 refused the connection"))
    asyncio.run(cls._seed_generalist_base(inst, MIRRORED))  # must not raise


def test_a_FENCE_FAILURE_PROPAGATES_rather_than_becoming_a_warning():
    """The one seeding failure that is NOT best-effort.

    `MirrorVerificationError` means the mirror served bytes that did not match a
    digest pinned in this repository. `ensure_base_checkpoint` raises for it --
    rather than reporting it like an unobtainable part -- precisely because it
    must stop everything. Swallowing it here turned that refusal into a warning
    reading "the load will use whatever is on disk", on the single path whose
    whole purpose is to stop.

    WHAT WAS LOST WAS THE REPORT, NOT THE FENCE, and an earlier version of this
    docstring overstated it as "the load would then proceed on unverified
    weights". A review corrected that and the correction matters: the bytes are
    never installed -- `install_verified` hashes into a temp file and only
    `os.replace`s it after `check_digest` passes -- and pooch re-checks
    upstream's xxh128 on every use of a cached file. So the load does not run on
    bad weights. What is lost is that nobody is TOLD the mirror served bytes
    that failed a pinned digest: the run continues as though the seed had merely
    been unnecessary, and a substituted or corrupted mirror stays invisible.

    It subclasses `RuntimeError`, which is why the broad `except Exception`
    caught it and why the re-raise has to come FIRST.
    """
    import base_checkpoint_mirror as bcm

    cls, inst = _seed_inst(bcm.MirrorVerificationError("sha256 mismatch"))
    with pytest.raises(bcm.MirrorVerificationError, match="sha256"):
        asyncio.run(cls._seed_generalist_base(inst, MIRRORED))


def test_a_PLAIN_RuntimeError_is_STILL_swallowed_so_the_fence_case_discriminates():
    """The control that makes the arm above mean something.

    `MirrorVerificationError`'s base class must still be best-effort. If a plain
    `RuntimeError` propagated too, the fence arm would pass for the wrong reason
    -- the repair would simply be catching less, not catching the right thing --
    and the pair would no longer distinguish them.
    """
    cls, inst = _seed_inst(RuntimeError("something else went wrong"))
    asyncio.run(cls._seed_generalist_base(inst, MIRRORED))  # must not raise
