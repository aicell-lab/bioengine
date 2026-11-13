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


def _with_cluster_gpus(monkeypatch, total):
    """Make ``import ray`` inside the report see a cluster with ``total`` GPUs.

    ``entry`` does not import ray at module scope — the report imports it lazily so a
    diagnostic can never be what stops the app loading — so the stub goes in
    ``sys.modules`` and monkeypatch removes it again.
    """
    fake = types.ModuleType("ray")
    fake.cluster_resources = lambda: ({"CPU": 8.0} if total is None else {"CPU": 8.0, "GPU": total})
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
    assert "UPSCALING" in report["detail"]
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
    assert "MODEL_FINETUNE_BACKENDS" in caps["placement"]["advice"]


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
