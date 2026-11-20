"""Regression tests for #0370 — a GPU shortfall must be visible from the running app.

Both GPU runtimes declare ``gpu_memory_mb=-1`` (``runtime.py:65``,
``runtime_cellpose.py:79``), which ``bioengine/apps/builder.py:514-516`` resolves to
``num_gpus=1.0`` — a whole device each, deliberately skipping the ``VRAM_MB`` packing
path below it. Both pin ``min_replicas: 1``, so the application's GPU floor is the
number of enabled backends.

That reservation is the recorded decision and is NOT changed here: the app's own
``MODEL_TRAIN_VRAM_MB`` spans 6 000–40 000 MB across model types one runtime serves, so
no fixed positive ``gpu_memory_mb`` is right — 24 000 makes the app undeployable on
deNBI's T4s, and 6 000 lets two replicas pack onto one GPU and OOM mid-training. The
defect fixed here is that asking for more GPUs than exist is INVISIBLE: the unplaceable
replica reports ``UPSCALING``, which is not a failure state and never times out.

Measured 2026-11-13 and the reason these tests exist rather than a cluster assertion:
``git show`` of app 0.16.1 (commit 7d982097) declares the same ``-1`` with the same
``min_replicas: 1`` in both runtimes, so this is a design property since the Cellpose
backend arrived (0.12.0), not a regression.

Plain pytest, no Ray and no cluster: ``ray.cluster_resources`` is stubbed, which is the
only way to exercise a 1-GPU cluster, a 2-GPU cluster and a Ray-less replica in one run.
"""

from __future__ import annotations

import sys
import types

import entry

_SAME = object()
"""Sentinel for "free == total", kept distinct from ``None``, which means the
resource view omits the GPU key entirely — a different case with a different answer."""


def _with_cluster_gpus(monkeypatch, total, free=_SAME):
    """Make ``import ray`` inside the report see a cluster with ``total`` GPUs, of
    which ``free`` are unbooked.

    The two views are SEPARATE arguments on purpose: ``cluster_resources()`` is the
    total and ``available_resources()`` is what is free, and the report keys its
    verdict to the first while reporting the second. A harness that set only a total
    could not express the shape the row was filed about — deNBI, 4 GPUs, 0 free —
    which is the gap the re-review found (#0370). ``free`` defaults to ``total``,
    i.e. an idle cluster, so a case that is not about the distinction stays short.

    ``entry`` does not import ray at module scope — the report imports it lazily so a
    diagnostic can never be what stops the app loading — so the stub goes in
    ``sys.modules`` and monkeypatch removes it again.
    """
    if free is _SAME:
        free = total
    fake = types.ModuleType("ray")
    fake.cluster_resources = lambda: ({"CPU": 8.0} if total is None else {"CPU": 8.0, "GPU": total})
    fake.available_resources = lambda: ({"CPU": 8.0} if free is None else {"CPU": 8.0, "GPU": free})
    monkeypatch.setitem(sys.modules, "ray", fake)


def _entry_class():
    """``entry.EntryApp`` is a ``ray.serve.deployment.Deployment`` once
    ``@bioengine.app`` has run, so its methods are not reachable through it;
    ``func_or_class`` is the class the decorator wrapped."""
    return entry.EntryApp.func_or_class


class _RecordingLogger:
    """Stands in for ``entry.logger``, which is a lazy proxy with no ``warning``
    attribute until the framework resolves it — so it is replaced, not patched."""

    def __init__(self):
        self.warning_calls = []
        self.info_calls = []

    def warning(self, msg, *a):
        self.warning_calls.append(msg)

    def info(self, msg, *a):
        self.info_calls.append(msg)


def _ray_unimportable(monkeypatch):
    """A replica where ``import ray`` raises, to pin the never-raises promise."""
    broken = types.ModuleType("ray")

    class _Boom:
        def __getattr__(self, name):
            raise RuntimeError("ray is not initialised in this process")

    broken.cluster_resources = _Boom().__getattr__
    # BOTH views, so the promise holds whichever of the two the report reads first.
    broken.available_resources = _Boom().__getattr__
    monkeypatch.setitem(sys.modules, "ray", broken)


# ── the count itself ─────────────────────────────────────────────────────────


def test_each_enabled_gpu_backend_costs_one_whole_gpu(monkeypatch) -> None:
    """The arithmetic the whole row is about: N backends on → N whole GPUs needed."""
    _with_cluster_gpus(monkeypatch, 8.0)
    assert entry._gpu_placement_report({"microsam", "cellpose"})["gpus_required"] == 2
    assert entry._gpu_placement_report({"cellpose"})["gpus_required"] == 1
    assert entry._gpu_placement_report({"microsam"})["gpus_required"] == 1
    assert entry._gpu_placement_report(set())["gpus_required"] == 0


def test_the_gpu_backend_list_matches_what_entry_actually_composes() -> None:
    """CONTROL with teeth: the count is only right if GPU_BACKENDS is the set of
    backends ``_parse_backends`` accepts. If someone adds a third backend to the
    parser and not here, every report under-counts and silently reads sufficient."""
    assert set(entry.GPU_BACKENDS) == entry._parse_backends(None)


# ── the three answers ────────────────────────────────────────────────────────


def test_enough_gpus_reads_sufficient(monkeypatch) -> None:
    """DISCRIMINATOR: a report that always warned would be useless. KTH's real shape
    measured 2026-11-13 — 3 GPUs, 1 in use, on three one-GPU nodes — places both
    backends, because nothing colocates the replicas (no placement group anywhere in
    bioengine/apps/ or bioengine/_app/)."""
    _with_cluster_gpus(monkeypatch, 3.0)
    report = entry._gpu_placement_report({"microsam", "cellpose"})
    assert report["sufficient"] is True
    assert report["gpus_in_cluster"] == 3.0
    assert report["advice"] == ""


def test_too_few_gpus_reads_insufficient_and_names_the_remedy(monkeypatch) -> None:
    """The reported condition: one GPU, two backends. This is the case that used to
    produce nothing at all."""
    _with_cluster_gpus(monkeypatch, 1.0)
    report = entry._gpu_placement_report({"microsam", "cellpose"})
    assert report["sufficient"] is False
    assert report["gpus_in_cluster"] == 1.0
    assert report["gpus_required"] == 2
    # An operator reading this must be able to act without reading the source.
    assert "MODEL_FINETUNE_BACKENDS" in report["advice"]
    # The states measured live on 2026-11-13, not the ones one would guess: Serve
    # reports the unplaceable runtime as UPDATING (replica STARTING) and the app as
    # DEPLOYING. A detail naming a state the operator will not see is a wrong answer
    # dressed as a diagnosis.
    assert "UPDATING" in report["detail"]
    assert "DEPLOYING" in report["detail"]
    for name in ("cellpose", "microsam"):
        assert name in report["detail"]


def test_one_backend_on_one_gpu_is_sufficient(monkeypatch) -> None:
    """The supported single-GPU configuration, and the thing the live demo deploys."""
    _with_cluster_gpus(monkeypatch, 1.0)
    report = entry._gpu_placement_report({"cellpose"})
    assert report["sufficient"] is True
    assert report["gpus_required"] == 1


def test_no_gpu_reported_yet_is_unknown_not_a_shortfall(monkeypatch) -> None:
    """An elastic SLURM cluster has no GPU node until the autoscaler provides one.

    Reporting that as insufficient would make this diagnostic cry wolf on every
    Berzelius deploy — which is the same class of mistake as the silence it replaces,
    pointed the other way. ``sufficient`` is three-valued for exactly this.
    """
    _with_cluster_gpus(monkeypatch, 0.0)
    report = entry._gpu_placement_report({"microsam", "cellpose"})
    assert report["sufficient"] is None
    assert report["gpus_in_cluster"] == 0.0
    assert "SLURM" in report["detail"]


def test_a_cluster_advertising_no_gpu_key_is_unknown_too(monkeypatch) -> None:
    """``cluster_resources()`` omits the key entirely on a CPU-only cluster rather
    than reporting 0.0, and an absent key must not read as a finding."""
    _with_cluster_gpus(monkeypatch, None)
    report = entry._gpu_placement_report({"cellpose"})
    assert report["sufficient"] is None
    assert report["gpus_in_cluster"] == 0.0


def test_a_cluster_that_owns_enough_but_has_none_free_names_the_free_count(monkeypatch) -> None:
    """deNBI's measured shape, and the case the first version of this report got wrong.

    4 GPUs owned, 0 free (read from that worker's own ``get_status()`` → the
    ``ray_cluster.nodes`` block, 2026-11-13), both backends enabled. The report read
    only ``cluster_resources()``, so it printed a reassuring *"4 GPU(s) visible, 2
    required"* on exactly the cluster whose runtimes were pending — a true sentence
    that answers the wrong question.

    The two views are set DIFFERENTLY here, which is what makes this arm able to see
    the defect at all: with ``free`` defaulted to ``total`` every other case in this
    file passes either implementation.
    """
    _with_cluster_gpus(monkeypatch, 4.0, free=0.0)
    report = entry._gpu_placement_report({"microsam", "cellpose"})

    assert report["gpus_in_cluster"] == 4.0
    assert report["gpus_free"] == 0.0
    # The free figure must be IN the operator-facing sentence, not only in a key
    # nobody reads: the log line is `detail` and nothing else.
    assert "0 free" in report["detail"], (
        f"detail does not name the free count, so the log line still reads as "
        f"reassurance on a cluster whose runtimes pend: {report['detail']!r}"
    )
    assert "pend" in report["detail"]

    # DELIBERATELY unchanged, and this is the half that is easy to get wrong in the
    # other direction (#0370 re-review): a free shortfall is NOT a shortfall. The
    # cluster owns enough, this app's own replicas are part of what books the
    # devices, and an elastic cluster is momentarily 0-free between jobs. Escalating
    # it would make the diagnostic fire at itself.
    assert report["sufficient"] is True
    assert report["advice"] == ""


def test_a_real_shortfall_and_a_merely_busy_cluster_do_not_read_alike(monkeypatch) -> None:
    """DISCRIMINATOR for the test above. Reporting the free count is only worth
    anything if the two situations still come apart: owning too few is a configuration
    mistake with a remedy, owning enough but being busy is neither."""
    _with_cluster_gpus(monkeypatch, 1.0, free=0.0)
    short = entry._gpu_placement_report({"microsam", "cellpose"})
    _with_cluster_gpus(monkeypatch, 4.0, free=0.0)
    busy = entry._gpu_placement_report({"microsam", "cellpose"})

    assert short["sufficient"] is False and busy["sufficient"] is True
    assert "MODEL_FINETUNE_BACKENDS" in short["advice"]
    assert busy["advice"] == ""
    assert short["detail"] != busy["detail"]


def test_the_free_count_is_reported_on_a_cluster_with_room_to_spare(monkeypatch) -> None:
    """The other arm of the same key: ``gpus_free`` is populated whenever the view was
    readable, not only when it is short. A field present only on the bad path cannot be
    read by anything that polls."""
    _with_cluster_gpus(monkeypatch, 4.0, free=3.0)
    report = entry._gpu_placement_report({"cellpose"})
    assert report["sufficient"] is True
    assert report["gpus_free"] == 3.0
    assert "3 free" in report["detail"]


def test_a_fractional_shortfall_still_reads_insufficient(monkeypatch) -> None:
    """A GPU already fractionally booked by another app leaves <1.0 advertised; the
    runtime needs a whole device, so this is a shortfall and not a near miss."""
    _with_cluster_gpus(monkeypatch, 0.5)
    report = entry._gpu_placement_report({"cellpose"})
    assert report["sufficient"] is False


# ── it must never be what breaks the app ────────────────────────────────────


def test_an_unreadable_ray_view_degrades_instead_of_raising(monkeypatch) -> None:
    """A diagnostic that can take the entry down is worse than no diagnostic.

    The entry calls this from its ``@bioengine.async_init``, so a raise here would
    fail the replica — turning an over-subscribed cluster into a dead app.
    """
    _ray_unimportable(monkeypatch)
    report = entry._gpu_placement_report({"microsam", "cellpose"})
    assert report["sufficient"] is None
    assert report["gpus_in_cluster"] is None
    # Both keys present and null rather than absent — an absent key reads as a
    # finding to anything that does `.get()` and compares.
    assert report["gpus_free"] is None
    assert "could not read" in report["detail"]
    # The part that needs no cluster must still be answered.
    assert report["gpus_required"] == 2


def test_no_gpu_backend_enabled_needs_no_gpu(monkeypatch) -> None:
    """Degenerate but reachable: a worker with neither backend on. Must not warn."""
    _with_cluster_gpus(monkeypatch, 0.0)
    report = entry._gpu_placement_report(set())
    assert report["sufficient"] is True
    assert report["advice"] == ""


# ── the two places it is surfaced ───────────────────────────────────────────


def _async_init_source() -> str:
    """The body of the entry's ``@bioengine.async_init`` coroutine, read off disk.

    Text rather than ``inspect``, for the same reason
    ``test_the_declaration_this_row_is_about_is_unchanged`` reads off disk: this must
    hold in an environment where neither ML stack is installed, and
    ``bioengine.async_init`` wraps the function so what is reachable through the class
    is not guaranteed to be the source.
    """
    from pathlib import Path

    src = (Path(__file__).parent / "entry.py").read_text()
    marker = "@bioengine.async_init"
    assert src.count(marker) == 1, (
        f"CONTROL: expected exactly one {marker} in entry.py, found {src.count(marker)}"
        " — this slice no longer identifies the startup coroutine"
    )
    rest = src[src.index(marker) + len(marker) :]
    end = rest.index("\n    @")  # the next method decorator ends the block
    return rest[:end]


def test_the_placement_log_is_actually_called_at_startup() -> None:
    """The warning is worth writing only if something writes it (#0370, re-review).

    Replacing ``self._log_gpu_placement()`` in ``_connect`` with ``pass`` left the
    whole suite green at 139 passed, while a comment-only control arm was also green
    and six other mutation arms went red — so the harness discriminated and this was a
    real hole, not a quiet arm.
    ``test_the_entry_logs_a_warning_only_on_a_real_shortfall`` cannot see it: it binds
    the method with ``__get__(inst)`` and so exercises the helper, never the wiring.
    A grep for the call site existed, but outside the repo, where CI never runs it — and
    a report nothing calls is indistinguishable from no report at all, which is this
    row's whole subject.

    Asserted against the startup coroutine specifically, not the file: the call
    surviving somewhere else in ``entry.py`` is not the thing that makes it run.
    """
    body = _async_init_source()

    # CONTROL first: the slice is the startup coroutine and is bounded. Without these
    # a slice that silently became the empty string, or the whole file, would make the
    # assertion below either vacuously red or vacuously green.
    assert "connect_to_server" in body, "CONTROL: the slice is not _connect's body"
    assert "def get_training_capabilities" not in body, (
        "CONTROL: the slice ran past the end of _connect and covers the whole class"
    )

    assert "self._log_gpu_placement()" in body, (
        "nothing calls _log_gpu_placement from the entry's startup path, so the "
        "⚠ GPU placement line the README calls the ONLY readable surface in the hung "
        "state is never written"
    )
    assert "def _log_gpu_placement" in entry_source(), (
        "CONTROL: the method the call site names is gone, so the call above would "
        "raise rather than log"
    )


def entry_source() -> str:
    """``entry.py`` as text. Separate from the slice above so a control can check the
    whole file without re-deriving the slice."""
    from pathlib import Path

    return (Path(__file__).parent / "entry.py").read_text()


def test_the_entry_logs_a_warning_only_on_a_real_shortfall(monkeypatch) -> None:
    """The log line is the signal an operator gets without calling anything: the
    entry replica is HEALTHY on any node, so it lands in deployments.<entry>.logs,
    which get_app_status already returns.

    Both arms asserted — a logger that warned unconditionally would pass a
    one-sided check.
    """
    log = _RecordingLogger()
    monkeypatch.setattr(entry, "logger", log)
    inst = object.__new__(_entry_class())
    inst._enabled_backends = {"microsam", "cellpose"}
    say = _entry_class()._log_gpu_placement.__get__(inst)

    _with_cluster_gpus(monkeypatch, 1.0)
    say()
    assert len(log.warning_calls) == 1
    assert "MODEL_FINETUNE_BACKENDS" in log.warning_calls[0]
    assert log.info_calls == []

    _with_cluster_gpus(monkeypatch, 2.0)
    say()
    assert len(log.warning_calls) == 1, "warned on a cluster that has enough GPUs"
    assert len(log.info_calls) == 1


def test_get_training_capabilities_explains_the_unavailable_it_reports(monkeypatch) -> None:
    """The queryable half, reproducing the whole reported symptom in one call.

    This is the exact state of an over-subscribed cluster: the GPU runtimes never
    schedule, so ``_runtime_gpu_memory`` comes back empty and every model record reads
    ``runtime unavailable`` — with, before this change, nothing anywhere saying why.
    Asserting both halves together is the point: the diagnosis has to arrive in the
    same response as the symptom, because that response is what the annotate page and
    an operator already call.
    """
    import asyncio

    _with_cluster_gpus(monkeypatch, 1.0)
    cls = _entry_class()
    inst = object.__new__(cls)
    inst._enabled_backends = {"microsam", "cellpose"}
    inst.declared_gpu_memory_mb = None  # no deploy-time figure -> live probe path
    inst.declared_gpu_device = None

    async def _no_runtime(self, model_type=None):
        return {}  # what an unplaceable runtime looks like from the entry

    monkeypatch.setattr(cls, "_runtime_gpu_memory", _no_runtime)
    monkeypatch.setattr(cls, "_training_parameter_schema", lambda self: [])

    caps = asyncio.run(cls.get_training_capabilities.__get__(inst)())

    # The symptom.
    assert caps["models"], "no model records at all — the call did not do its work"
    assert all(m["reason"] == "runtime unavailable" for m in caps["models"])
    # The explanation, in the same payload.
    assert caps["placement"]["sufficient"] is False
    assert caps["placement"]["gpus_required"] == 2
    assert caps["placement"]["gpus_in_cluster"] == 1.0
    assert caps["placement"]["gpus_free"] == 1.0  # both views reach the RPC surface
    assert "MODEL_FINETUNE_BACKENDS" in caps["placement"]["advice"]


def test_the_entry_starts_one_replica_so_it_can_speak_while_the_app_hangs() -> None:
    """The warning above is only worth writing if something writes it.

    Measured 2026-11-13 on a one-GPU external-cluster worker with both backends
    enabled: the unplaceable GPU runtime leaves the app in ``DEPLOYING``, and the
    ProxyDeployment registers the app's Hypha service only once every sibling is
    serviceable (``bioengine/apps/proxy_deployment.py``, ``_is_serviceable``) — so no
    caller can reach ``get_training_capabilities`` in that state. With
    ``initial_replicas: 0`` nothing wakes the entry either and the warning is never
    written; the line only appeared after a replica was forced up by hand over a Serve
    handle, which no operator can do. One CPU-only replica at deploy time is what puts
    it into ``deployments.<entry>.logs``, which ``get_app_status`` returns throughout
    ``DEPLOYING``.

    ``min_replicas`` must stay 0: #0045's brief is that an idle entry holds nothing on
    an HPC login node. Both halves are asserted, because satisfying one by breaking the
    other is the easy mistake.
    """
    cfg = entry.EntryApp._deployment_config.autoscaling_config
    assert cfg.min_replicas == 0, "#0045: an idle entry must still scale to zero"
    assert cfg.initial_replicas == 1, (
        "#0370: with 0 the placement warning is never written on the cluster it is for"
    )
    # CONTROL: this is the live resolved config, not a default-shaped object — these
    # two values are set in the decorator and are not Ray Serve defaults.
    assert cfg.max_replicas == 1
    assert cfg.downscale_delay_s == 600.0


def test_the_declaration_this_row_is_about_is_unchanged() -> None:
    """The recorded decision, pinned. Changing either runtime to a positive
    ``gpu_memory_mb`` opts it into VRAM packing (builder.py:518-536) with a figure
    fixed at author time while the real need is per-model and per-caller — so it
    must be re-argued on the row, not slipped in.

    Read off disk rather than by importing, so this holds in an environment where
    neither ML stack is installed (which is every environment this suite runs in).
    """
    from pathlib import Path

    here = Path(__file__).parent
    for name in ("runtime.py", "runtime_cellpose.py"):
        src = (here / name).read_text()
        assert "gpu_memory_mb=-1" in src, f"{name}: the -1 declaration is gone"
        assert "min_replicas" in src, f"{name}: control — the autoscaling block moved"
