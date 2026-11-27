"""`runtime_env_pip_cache_size_gb` is accepted in every mode and applied in only two.

MEASURED on a live external-cluster site. With `RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB` unset on every
Ray pod, two GPU-backed apps whose pip environments are 5.7-7.9 GB evicted each other continuously
— 21-22 deletes and 29-31 adds per pod in `runtime_env_setup-*.log` — and a deploy sat in
`DEPLOYING` with its replicas `STARTING` and no error at all.

The worker exports that variable only inside `if mode in ["single-machine", "slurm"]`, which is
correct as far as it goes: the variable is read by the process running Ray's runtime_env agent, and
in those two modes that is a Ray the worker launches itself. On an external cluster the Ray pods
belong to the cluster operator and are already running when the worker attaches, so the export can
never reach them and the pods keep Ray's built-in default of 10 GB.

The defect is therefore not the missing export — it is that a caller can pass 30 and be told
nothing. These arms pin the *telling*.

WHY A WARNING AND NOT A REFUSAL: every external-cluster site running today has been running on the
default, so refusing would break working deployments in order to report a performance fault. Arm B
pins that a value equal to Ray's own default stays silent, so the warning means something when it
does appear.
"""

from __future__ import annotations

import logging

import pytest

import bioengine.cluster.ray_cluster as rc
from bioengine.cluster.ray_cluster import RAY_DEFAULT_PIP_CACHE_SIZE_GB, RayCluster


class _Recorder(logging.Logger):
    def __init__(self):
        super().__init__("rec")
        self.warnings: list[str] = []
        self.debugs: list[str] = []

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg) % args if args else str(msg))

    def debug(self, msg, *args, **kwargs):
        self.debugs.append(str(msg) % args if args else str(msg))

    def info(self, msg, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        pass


@pytest.fixture
def recorder(monkeypatch):
    """Substitute the logger factory, because the warning is emitted inside the constructor and
    there is no instance to attach a handler to until it returns."""
    log = _Recorder()
    monkeypatch.setattr(rc, "create_logger", lambda *a, **k: log)
    return log


def _external(monkeypatch, size):
    # The worker must not be the one setting it in this mode; start from unset so arm A can assert
    # that it stays unset.
    monkeypatch.delenv("RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB", raising=False)
    return RayCluster(mode="external-cluster", runtime_env_pip_cache_size_gb=size)


def test_a_a_non_default_size_warns_that_it_has_no_effect(recorder, monkeypatch):
    _external(monkeypatch, 30)
    joined = " ".join(recorder.warnings)
    assert "NO EFFECT" in joined, f"a silently ignored parameter must be reported: {joined!r}"
    # The operator needs three things to act on it: the value, the variable, and where to put it.
    assert "30" in joined, "the message must name the value that is being ignored"
    assert "RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB" in joined, "it must name the variable to set"
    assert "RayCluster" in joined, "it must name where the variable has to go"
    assert str(RAY_DEFAULT_PIP_CACHE_SIZE_GB) in joined, "it must name what is in effect instead"
    # And the worker must not pretend to apply it.
    import os

    assert "RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB" not in os.environ, (
        "the worker must not export it in a mode where it cannot reach the Ray pods"
    )


def test_b_a_size_equal_to_rays_default_stays_silent(recorder, monkeypatch):
    """Nothing is lost at the default, so a warning there would be noise — and noise is what makes
    a real warning get ignored."""
    _external(monkeypatch, RAY_DEFAULT_PIP_CACHE_SIZE_GB)
    joined = " ".join(recorder.warnings)
    assert "NO EFFECT" not in joined, f"warned at Ray's own default: {joined!r}"


@pytest.mark.parametrize("bad", [0, -1])
def test_c_a_non_positive_size_is_rejected_in_this_mode_too(recorder, monkeypatch, bad):
    """It used to be validated only in the modes that apply it, so an external-cluster caller could
    pass 0 and hear nothing at all."""
    with pytest.raises(ValueError):
        _external(monkeypatch, bad)


def test_d_the_rays_default_constant_matches_what_ray_actually_does():
    """A hard-coded 10 would rot silently if Ray changed its fallback. This asserts the constant
    against the installed Ray rather than against the comment beside it.

    Skips rather than passing vacuously when the private module moves: a check that cannot run is
    not a check.
    """
    try:
        from ray._private.runtime_env.uri_cache import DEFAULT_MAX_URI_CACHE_SIZE_BYTES
    except Exception:  # pragma: no cover - depends on Ray internals
        pytest.skip("ray._private.runtime_env.uri_cache moved; cannot cross-check the default")
    assert DEFAULT_MAX_URI_CACHE_SIZE_BYTES == RAY_DEFAULT_PIP_CACHE_SIZE_GB * (1024**3), (
        f"Ray's default is {DEFAULT_MAX_URI_CACHE_SIZE_BYTES} bytes, which is not "
        f"{RAY_DEFAULT_PIP_CACHE_SIZE_GB} GB — the constant has rotted"
    )
