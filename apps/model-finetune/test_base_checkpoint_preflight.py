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


# ── the probe ITSELF, executed. Everything above this line runs on hand-built
#    dicts, and a review pointed out what that leaves open: a probe that
#    reported every part uncached and unreachable -- wrong path, wrong key, an
#    exception swallowed -- would pass every case above, because the ADMITTED
#    cases never went through it. Two real bugs were exactly that shape and
#    both survived. These execute the probe. ─────────────────────────────────


import asyncio
import http.server
import socket
import threading
import time
import types


def _runtime_cls():
    import runtime

    return getattr(runtime.RuntimeApp, "func_or_class", runtime.RuntimeApp)


def _stub_micro_sam(monkeypatch, cache_root, urls, keys):
    """A stand-in for `micro_sam.util` shaped like the pinned 1.8.11.

    `models()` there is `pooch.create(path=<cachedir>/models, registry, urls)`,
    so the stub exposes exactly `path`, `urls` and `registry` — the three the
    probe reads. Injected through sys.modules because micro-sam is a RUNTIME
    dependency and is not importable here at all.
    """
    reg = types.SimpleNamespace(path=str(cache_root), urls=dict(urls),
                                registry={k: "xxh128:0" for k in keys})
    mod = types.ModuleType("micro_sam.util")
    mod.models = lambda: reg
    pkg = types.ModuleType("micro_sam")
    pkg.util = mod
    monkeypatch.setitem(__import__("sys").modules, "micro_sam", pkg)
    monkeypatch.setitem(__import__("sys").modules, "micro_sam.util", mod)
    return reg


def test_a_CACHED_type_is_admitted_and_the_probe_TOUCHES_NO_NETWORK(tmp_path, monkeypatch):
    """(a) Both parts on disk must admit, and must not ask the network at all —
    a cached model is the common case and paying a round trip for it would be
    a cost this guard is supposed to remove, not add."""
    import runtime

    models = tmp_path / "models"
    models.mkdir()
    for key in ("vit_b_lm", "vit_b_lm_decoder"):
        (models / key).write_bytes(b"weights")
    _stub_micro_sam(monkeypatch, models, {}, ("vit_b_lm", "vit_b_lm_decoder"))

    def _never(*a, **k):
        raise AssertionError("the probe went to the network for a CACHED model")

    monkeypatch.setattr(runtime, "_url_reachable", _never)
    st = asyncio.run(_runtime_cls().base_checkpoint_status(object(), "vit_b_lm"))
    assert st["probe_error"] is None, st
    assert set(st["parts"]) == {"encoder", "decoder"}, st
    assert all(p["cached"] for p in st["parts"].values()), st
    assert training.base_checkpoint_refusal("vit_b_lm", st) is None, st


def test_a_store_that_answers_HEAD_with_405_is_still_ADMITTED(tmp_path, monkeypatch):
    """(b) Some stores refuse HEAD and serve GET perfectly well. Treating that
    as unreachable would refuse a model this site can actually train, so the
    probe falls back to a 1-byte ranged GET — exercised here against a real
    server rather than asserted from the source."""

    class H(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(405)
            self.end_headers()

        def do_GET(self):
            self.send_response(206)
            self.send_header("Content-Range", "bytes 0-0/1")
            self.end_headers()
            self.wfile.write(b"x")

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        models = tmp_path / "models"
        models.mkdir()
        _stub_micro_sam(
            monkeypatch, models,
            {"vit_b_lm": base + "/enc", "vit_b_lm_decoder": base + "/dec"},
            ("vit_b_lm", "vit_b_lm_decoder"),
        )
        st = asyncio.run(_runtime_cls().base_checkpoint_status(object(), "vit_b_lm"))
    finally:
        srv.shutdown()
    assert st["probe_error"] is None, st
    assert all(p["reachable"] is True for p in st["parts"].values()), st
    assert training.base_checkpoint_refusal("vit_b_lm", st) is None, st


def test_an_UNREACHABLE_store_is_REFUSED_with_the_unavailable_text(tmp_path, monkeypatch):
    """(c) The case the row was filed for. A refused connection is used rather
    than a black hole so the test is fast and deterministic; the BUDGET under a
    black hole is the next test."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dead = s.getsockname()[1]
    s.close()  # nothing listens here now -> connection refused, immediately

    models = tmp_path / "models"
    models.mkdir()
    base = f"http://127.0.0.1:{dead}"
    _stub_micro_sam(
        monkeypatch, models,
        {"vit_t_em_organelles": base + "/enc",
         "vit_t_em_organelles_decoder": base + "/dec"},
        ("vit_t_em_organelles", "vit_t_em_organelles_decoder"),
    )
    st = asyncio.run(
        _runtime_cls().base_checkpoint_status(object(), "vit_t_em_organelles")
    )
    assert st["probe_error"] is None, st
    assert all(p["reachable"] is False for p in st["parts"].values()), st
    msg = training.base_checkpoint_refusal("vit_t_em_organelles", st)
    assert msg and "cannot be trained at this site" in msg, msg
    assert "vit_t_em_organelles" in msg, msg


def test_the_probe_FITS_the_entry_budget_against_a_BLACK_HOLE(tmp_path, monkeypatch):
    """The arithmetic a review caught. Two parts, each up to two attempts: run
    sequentially that exceeded the entry's own timeout, so the caller's
    deadline fired first and the user was told the check "could not be
    checked" rather than that the weights are unavailable here — the wrong
    sentence for a host that silently drops packets.
    """
    import entry
    import runtime

    models = tmp_path / "models"
    models.mkdir()
    # 10.255.255.1 is unroutable: packets are dropped, not refused, so each
    # attempt costs its full timeout rather than failing fast.
    _stub_micro_sam(
        monkeypatch, models,
        {"vit_t_em_organelles": "http://10.255.255.1/enc",
         "vit_t_em_organelles_decoder": "http://10.255.255.1/dec"},
        ("vit_t_em_organelles", "vit_t_em_organelles_decoder"),
    )
    t0 = time.monotonic()
    st = asyncio.run(
        _runtime_cls().base_checkpoint_status(object(), "vit_t_em_organelles")
    )
    elapsed = time.monotonic() - t0
    budget = entry.BASE_CHECKPOINT_PREFLIGHT_TIMEOUT_S
    assert elapsed < budget, (
        f"the probe took {elapsed:.1f}s against a black hole, which does not fit "
        f"the entry's {budget}s budget — the caller's timeout fires first and the "
        f"refusal says 'could not be checked' instead of naming the weights"
    )
    assert all(p["reachable"] is False for p in st["parts"].values()), st
    msg = training.base_checkpoint_refusal("vit_t_em_organelles", st)
    assert msg and "cannot be trained at this site" in msg, msg


# ── the WIRING, which a review found was dead code ──────────────────────────


class _FakeRuntime:
    """Shaped like the composition handle, which is the point.

    `BioEngineRuntimeHandle.__getattr__` returns a bound method for ANY public
    name, so `getattr(runtime, "base_checkpoint_status", None)` is never None.
    A capability test written that way looks like it skips on an older replica
    and actually calls, fails, and refuses. This fake reproduces that: every
    attribute exists.
    """

    def __init__(self, caps, probe=None):
        self._caps = caps
        self._probe = probe

    async def ping(self):
        return {"status": "ok", "capabilities": list(self._caps)}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        async def _call(*a, **k):
            if self._probe is None:
                raise AttributeError(
                    "'RuntimeApp' object has no attribute %r" % name
                )
            return self._probe(*a, **k)

        return _call


def _entry_instance(runtime_obj):
    import entry

    cls = getattr(entry.EntryApp, "func_or_class", entry.EntryApp)
    inst = object.__new__(cls)
    inst._enabled_backends = {"microsam", "cellpose"}
    inst.runtime = runtime_obj
    import logging

    inst.logger = logging.getLogger("test-entry")
    return cls, inst


def test_an_OLDER_RUNTIME_SKIPS_rather_than_refusing_every_run():
    """A replica that does not advertise the probe must not stop training.

    During a rollout the entry can be newer than the runtime. This guard
    removes a late failure; it is not a safety boundary, so its absence has to
    restore the previous behaviour. A review measured that the first version
    did the opposite — `getattr` answered for a method that was not there, the
    call failed, and EVERY micro-sam run was refused.
    """
    cls, inst = _entry_instance(_FakeRuntime(caps=[], probe=None))
    asyncio.run(cls._refuse_unavailable_base_checkpoint(inst, "vit_b_lm"))


def test_a_runtime_that_ADVERTISES_the_probe_but_FAILS_it_REFUSES():
    """The other direction. Once a replica says it has the probe, a failure is
    a real failure and must not be mistaken for an old replica."""
    cls, inst = _entry_instance(
        _FakeRuntime(caps=["base_checkpoint_status"], probe=None)
    )
    with pytest.raises(ValueError) as exc:
        asyncio.run(cls._refuse_unavailable_base_checkpoint(inst, "vit_b_lm"))
    assert "vit_b_lm" in str(exc.value), exc.value


def test_an_unavailable_type_RAISES_naming_it():
    cls, inst = _entry_instance(
        _FakeRuntime(
            caps=["base_checkpoint_status"],
            probe=lambda mt: status({"encoder": UNREACHABLE}, model_type=mt),
        )
    )
    with pytest.raises(ValueError) as exc:
        asyncio.run(
            cls._refuse_unavailable_base_checkpoint(inst, "vit_t_em_organelles")
        )
    assert "vit_t_em_organelles" in str(exc.value), exc.value
    assert "no GPU was claimed" in str(exc.value), exc.value


def test_an_AVAILABLE_type_does_NOT_raise():
    """The control for the two above: a wiring that refused everything would
    pass both of them."""
    cls, inst = _entry_instance(
        _FakeRuntime(
            caps=["base_checkpoint_status"],
            probe=lambda mt: status(
                {"encoder": CACHED, "decoder": CACHED}, model_type=mt
            ),
        )
    )
    asyncio.run(cls._refuse_unavailable_base_checkpoint(inst, "vit_b_lm"))
