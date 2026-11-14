"""A ``custom_environment=True`` test on an image with no conda must REFUSE,
not come back as a failed verdict on the model (myco #0527).

The code used to spawn a bare ``mamba``, documented as "both binaries ship in
the replica image at ``/home/ray/anaconda3/bin/``" — true of the
``rayproject/ray`` base it was written against, false of every image this repo
builds (both Dockerfiles are ``FROM python:3.11-slim``, and
``bioengine-worker:latest`` installs none of mamba/conda/micromamba). So the
first thing that happened was ``FileNotFoundError: 'mamba'`` *inside*
``_execute_test``'s ``except Exception``, which turns any exception into a
**fallback test report**: ``status="failed"``, ``name="bioimageio format
validation"``, the traceback as its single unnamed detail. That object is shaped
like a verdict on the model and is not one.

The cases below pin all three halves of the fix, which is why the module is not
just one assertion:

* the refusal fires, and fires *before* anything is spent on an env build;
* it fires on the **effective** mode, so a model that declares no
  ``environment.yaml`` still downgrades to standard and runs on an image with no
  conda at all (that path works today and must keep working);
* it does **not** fire where a conda implementation exists — without that
  control, "no model report was returned" is a pass condition that a change
  breaking every custom-env run would also satisfy.

``test_a_failure_inside_the_try_is_rendered_as_a_report_about_the_model`` is the
pre-fix behaviour, kept deliberately: it is the shape the refusal exists to
avoid, and it pins the ``except`` clause so a future reader can see why raising
earlier is the honest path rather than catching later.

Plain pytest: no Ray, no network, no conda, no GPU. The conda implementation is
decided by ``resolve_conda_binary()``, which is stubbed in both directions, so
every case reads the same on an image that has mamba and on one that does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml

import entry
import runtime

# --- reaching the real code off the Serve-wrapped classes -----------------
#
# ``@bioengine.app`` has already turned both classes into
# ``ray.serve.deployment.Deployment`` objects, so their methods are not
# reachable through the module attribute. Same accessor the model-finetune
# bundle's tests use.

_ENTRY_CLS = entry.EntryDeployment.func_or_class
_RUNTIME_CLS = runtime.RuntimeDeployment.func_or_class


def _method_of(cls, name: str, instance):
    return getattr(cls, name).__get__(instance)


# --- fixtures: an RDF that declares a custom env, and one that does not ---


def _write_rdf(tmp_path: Path, *, declares_env: bool) -> Path:
    """A minimal RDF, enough for ``_model_declares_custom_env``.

    It reads the file as raw YAML and looks for an authored dependency file on
    any weight format, so nothing else about the model matters here.
    """
    rdf: Dict[str, Any] = {
        "type": "model",
        "name": "harness-model",
        "weights": {"pytorch_state_dict": {"source": "weights.pt"}},
    }
    if declares_env:
        rdf["weights"]["pytorch_state_dict"]["dependencies"] = {
            "source": "environment.yaml"
        }
    path = tmp_path / "rdf.yaml"
    path.write_text(yaml.safe_dump(rdf))
    return path


class _Package:
    """The slice of a model package ``_execute_test`` touches."""

    def __init__(self, source: Path) -> None:
        self.source = str(source)
        self.package_path = source.parent
        self.latest_remote_modified = 1_700_000_000.0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _ModelCache:
    def __init__(self, package: _Package) -> None:
        self._package = package
        self.calls: List[dict] = []

    async def get_model_package(self, **kwargs):
        self.calls.append(kwargs)
        return self._package


class _Runtime:
    """Stands in for the RuntimeDeployment handle. Records whether the test was
    ever actually run — the refusal's whole point is that it is not."""

    def __init__(self, report: Optional[dict]) -> None:
        self._report = report
        self.calls: List[dict] = []

    async def test(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self._report or {})


class _Harness:
    """A stand-in EntryDeployment carrying only what ``_execute_test`` reads.

    Every decision under test — the no-``environment.yaml`` downgrade, the
    refusal, the fallback-report builder — runs as the real method bound to this
    object. Only I/O is stubbed: the package fetch, the RuntimeDeployment
    handle, the HF localisation, the publish authorization.
    """

    def __init__(self, rdf: Path, *, report: Optional[dict] = None) -> None:
        self.package = _Package(rdf)
        self.model_cache = _ModelCache(self.package)
        self.runtime = _Runtime(report)
        self._env_inuse: Dict[str, int] = {}
        self._test_jobs: Dict[str, dict] = {}
        self._test_jobs_ttl_sec = 3600
        self.prebuild_calls: List[dict] = []
        self.prebuild_error: Optional[BaseException] = None

        # Class constants the real methods read, taken off the real class so a
        # rename cannot leave the harness attesting to a stale name.
        for const in (
            "_MODELS_COLLECTION",
            "_TEST_REPORTS_COLLECTION",
            "_TEST_REPORTS_WORKSPACE",
            "_UNTRUSTWORTHY_RETRY_FLOOR_S",
        ):
            setattr(self, const, getattr(_ENTRY_CLS, const))

        # Real methods, bound here so the test exercises them rather than a
        # paraphrase of them.
        for name in (
            "_execute_test",
            "_new_test_job",
            "_sweep_expired_test_jobs",
            "_update_test_job",
            "_get_bioimageio_versions",
            "_stamp_runtime_versions_in_test_env",
            "_release_inuse_envs",
            "_compute_report_score",
        ):
            setattr(self, name, _method_of(_ENTRY_CLS, name, self))
        self._model_declares_custom_env = _ENTRY_CLS._model_declares_custom_env

    # --- stubbed I/O ---

    async def _localize_hf_weights(self, package, model_id: str) -> None:
        return None

    async def _prebuild_conda_envs(self, **kwargs) -> List[str]:
        self.prebuild_calls.append(kwargs)
        if self.prebuild_error is not None:
            raise self.prebuild_error
        return []

    async def _read_published_report(self, model_alias: str, stage: bool):
        return None

    async def _caller_may_publish(self, model_id, caller) -> bool:
        return False

    def _caller_label(self, caller) -> str:
        return "test-harness"


_PASSING_REPORT = {
    "name": "bioimageio validation",
    "status": "passed",
    "details": [{"name": "reproduce test outputs", "status": "passed"}],
    "env": [],
}


def _run(harness: _Harness, *, custom_environment: bool, model_id: str):
    job = harness._new_test_job(model_id, custom_environment)
    coro = harness._execute_test(
        job=job,
        model_id=model_id,
        stage=True,
        custom_environment=custom_environment,
        cache="skip",
        caller={},
    )
    return job, coro


# --- the resolver -----------------------------------------------------------


def _which_only(*present: str):
    """A ``shutil.which`` that knows exactly ``present``."""
    return lambda name: f"/usr/bin/{name}" if name in present else None


def test_resolver_returns_none_when_no_implementation_is_installed(monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", _which_only())
    assert runtime.resolve_conda_binary() is None


def test_resolver_finds_conda_on_an_image_that_ships_only_conda(monkeypatch):
    """The positive control the refusal needs: the predicate is "no conda
    implementation", not "no mamba". An image with conda alone is serviceable —
    it takes the same ``env create`` / ``run -n`` / ``list -n`` arguments this
    app passes — so it must not be refused."""
    monkeypatch.setattr(runtime.shutil, "which", _which_only("conda"))
    assert runtime.resolve_conda_binary() == "conda"


def test_resolver_prefers_mamba_when_both_are_present(monkeypatch):
    """Preference order is load-bearing: mamba is chosen for the libmamba
    solver, and on an image with both the resolved binary must be byte-identical
    to what the code used before this change."""
    monkeypatch.setattr(runtime.shutil, "which", _which_only("mamba", "conda"))
    assert runtime.resolve_conda_binary() == "mamba"
    assert runtime.CONDA_IMPLEMENTATIONS[0] == "mamba"


def test_micromamba_alone_is_not_accepted(monkeypatch):
    """``micromamba`` takes ``create``, not ``env create``, so the argv shapes
    this app passes would not work against it. Claiming support would be
    untested code; refusing is the honest answer."""
    monkeypatch.setattr(runtime.shutil, "which", _which_only("micromamba"))
    assert runtime.resolve_conda_binary() is None


# --- the RuntimeDeployment guard (a direct ``runtime.test`` caller) ---------


def test_runtime_test_refuses_a_custom_env_run_with_no_conda(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "resolve_conda_binary", lambda: None)
    rdf = _write_rdf(tmp_path, declares_env=True)

    with pytest.raises(RuntimeError) as exc:
        _method_of(_RUNTIME_CLS, "_test", object())(
            rdf_path=str(rdf), custom_environment=True
        )

    msg = str(exc.value)
    assert "custom_environment=True" in msg
    assert "mamba" in msg and "conda" in msg
    # It must say whose fault this is not. The whole defect was an image
    # limitation reading as a model defect.
    assert "not a" in msg and "fault in the model" in msg


# --- the EntryDeployment refusal -------------------------------------------


@pytest.mark.asyncio
async def test_execute_test_refuses_custom_env_when_no_conda_is_installed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(entry, "resolve_conda_binary", lambda: None)
    harness = _Harness(_write_rdf(tmp_path, declares_env=True))
    job, coro = _run(harness, custom_environment=True, model_id="bioimage-io/affable-shark")

    with pytest.raises(RuntimeError) as exc:
        await coro

    msg = str(exc.value)
    assert "affable-shark" in msg
    assert "custom_environment=True" in msg
    assert "no test was run and no report was produced" in msg.replace("\n", " ")
    assert "custom_environment=False" in msg

    # Nothing was spent on an env build, and the model was never tested: the
    # refusal is before both.
    assert harness.prebuild_calls == []
    assert harness.runtime.calls == []
    assert job["env_setup_ts"] is None
    assert job["running_ts"] is None


@pytest.mark.asyncio
async def test_the_refusal_names_every_implementation_it_looked_for(
    tmp_path, monkeypatch
):
    """A refusal that does not say what would fix it is a dead end for whoever
    reads it, and the fix is an image that ships one of these."""
    monkeypatch.setattr(entry, "resolve_conda_binary", lambda: None)
    harness = _Harness(_write_rdf(tmp_path, declares_env=True))
    _, coro = _run(harness, custom_environment=True, model_id="bioimage-io/x")

    with pytest.raises(RuntimeError) as exc:
        await coro

    for impl in runtime.CONDA_IMPLEMENTATIONS:
        assert impl in str(exc.value)


@pytest.mark.asyncio
async def test_a_model_declaring_no_environment_yaml_still_runs_without_conda(
    tmp_path, monkeypatch
):
    """The downgrade must win over the refusal.

    A model with no authored ``environment.yaml`` on any weight format has
    ``custom_environment`` forced to False and tests in the serving runtime's
    own environment — no conda involved. That works today on an image with no
    conda at all, so a refusal placed *before* the downgrade would break calls
    that currently succeed. This is why the guard sits after it and reads the
    EFFECTIVE mode.
    """
    monkeypatch.setattr(entry, "resolve_conda_binary", lambda: None)
    harness = _Harness(
        _write_rdf(tmp_path, declares_env=False), report=dict(_PASSING_REPORT)
    )
    job, coro = _run(harness, custom_environment=True, model_id="bioimage-io/y")

    report = await coro

    assert report["status"] == "passed"
    assert report["test_environment"] == "standard"
    assert report["report_is_trustworthy"] is True
    assert job["custom_environment"] is False
    assert harness.prebuild_calls == []
    assert len(harness.runtime.calls) == 1
    assert harness.runtime.calls[0]["custom_environment"] is False


@pytest.mark.asyncio
async def test_a_custom_env_run_still_returns_a_normal_report_where_conda_exists(
    tmp_path, monkeypatch
):
    """The control: the refusal must not fire where a conda implementation is
    installed.

    Without this case, "no report came back" would be a pass condition, and a
    guard that refused every custom-env run unconditionally would satisfy the
    other tests. Here the env build and the test run are stubbed — what is being
    measured is that the guard lets the call through and the normal report flows
    out, not that a real mamba solve works.
    """
    monkeypatch.setattr(entry, "resolve_conda_binary", lambda: "mamba")
    harness = _Harness(
        _write_rdf(tmp_path, declares_env=True), report=dict(_PASSING_REPORT)
    )
    job, coro = _run(harness, custom_environment=True, model_id="bioimage-io/z")

    report = await coro

    assert report["status"] == "passed"
    assert report["test_environment"] == "custom"
    assert report["report_is_trustworthy"] is True
    assert job["custom_environment"] is True
    assert len(harness.prebuild_calls) == 1
    assert len(harness.runtime.calls) == 1
    assert harness.runtime.calls[0]["custom_environment"] is True


@pytest.mark.asyncio
async def test_a_failure_inside_the_try_is_rendered_as_a_report_about_the_model(
    tmp_path, monkeypatch
):
    """The pre-fix shape, pinned on purpose — this is what #0527 was.

    With the guard bypassed (a conda implementation resolves) a spawn failure
    from the env build lands in ``_execute_test``'s ``except Exception``, which
    builds a report named ``bioimageio format validation`` with
    ``status="failed"`` and the traceback as its one unnamed detail. The
    ``report_is_trustworthy=False`` flag keeps it out of the collection and out
    of the cache's answer slot, but it is still what the caller is handed and
    what the UI renders — so for the missing-conda case the honest answer is to
    raise before reaching here, which the test above checks.

    The injected error is the exact one a bare ``mamba`` spawn raises.
    """
    monkeypatch.setattr(entry, "resolve_conda_binary", lambda: "mamba")
    harness = _Harness(
        _write_rdf(tmp_path, declares_env=True), report=dict(_PASSING_REPORT)
    )
    harness.prebuild_error = FileNotFoundError(2, "No such file or directory", "mamba")
    _, coro = _run(harness, custom_environment=True, model_id="bioimage-io/w")

    report = await coro

    assert report["status"] == "failed"
    assert report["name"] == "bioimageio format validation"
    assert report["report_is_trustworthy"] is False
    assert len(report["details"]) == 1
    assert "mamba" in report["details"][0]["errors"][0]["msg"]
    # Never ran the model, yet the report is about the model.
    assert harness.runtime.calls == []
    # An untrustworthy report is never published, whatever the caller's grant.
    assert report["published"] is False
