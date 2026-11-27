"""A parameter that cannot take effect must say so, not be accepted in silence.

``runtime_env_pip_cache_size_gb`` is an ``__init__`` argument in every mode and is documented as
"Size of pip cache for runtime environments in GB". It is only exported as
``RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB`` where this process starts Ray — ``single-machine`` and
``slurm``. In ``external-cluster`` mode the Ray pods already exist and were created by the cluster
operator, so the export cannot reach them and the pods run on Ray's own default.

WHY IT MATTERS, measured on a live external-cluster site: the Ray workers held app environments of
5.7-7.9 GB each and evicted one another continuously (21-22 cache deletes and 29-31 adds per pod in
one session), so a freshly deployed app's replicas sat in ``STARTING`` indefinitely while gigabytes
were installed and discarded. The operator had set this parameter and had no way to learn it did
nothing — the silence is the defect, not the eviction.

The logger is captured by replacing ``create_logger`` BEFORE construction, because the message is
emitted inside ``__init__``; attaching a handler afterwards would miss it, and relying on
propagation would make the arm depend on how the logger happens to be configured.
"""

from __future__ import annotations

import logging

import pytest

import bioengine.cluster.ray_cluster as rc_mod
from bioengine.cluster.ray_cluster import RayCluster

NO_EFFECT = "NO EFFECT in external-cluster mode"
ENV_VAR = "RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB"


@pytest.fixture
def captured(monkeypatch):
    """Capture every record the cluster logs during construction."""
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    def fake_create_logger(*args, **kwargs):
        logger = logging.getLogger(f"test-pip-cache-{len(records)}-{id(records)}")
        logger.handlers = [_Collect()]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        return logger

    monkeypatch.setattr(rc_mod, "create_logger", fake_create_logger)
    return records


def _messages(records):
    out = []
    for r in records:
        try:
            out.append(r.getMessage())
        except Exception:  # noqa: BLE001
            out.append(str(r.msg))
    return out


def test_external_cluster_says_the_pip_cache_size_has_no_effect(captured):
    RayCluster(mode="external-cluster", runtime_env_pip_cache_size_gb=30)
    msgs = _messages(captured)
    hits = [m for m in msgs if NO_EFFECT in m]
    assert hits, (
        "external-cluster mode accepted runtime_env_pip_cache_size_gb in silence; "
        f"logged messages were: {msgs!r}"
    )
    said = hits[0]
    assert ENV_VAR in said, f"the message must name the variable to set: {said!r}"
    assert (
        "30" in said
    ), f"the message must quote the value that is being ignored: {said!r}"


def test_the_message_names_where_the_value_must_actually_be_set(captured):
    """A warning that does not say what to do instead is only half a message."""
    RayCluster(mode="external-cluster", runtime_env_pip_cache_size_gb=30)
    said = next(m for m in _messages(captured) if NO_EFFECT in m)
    assert "pods" in said, f"it must point at the Ray pods: {said!r}"
    assert (
        "STARTING" in said
    ), f"it must name the symptom an operator will actually see: {said!r}"


def test_a_bad_value_is_refused_in_external_cluster_mode_too(captured):
    """The validation used to live inside the branch that consumes the value, so an
    external-cluster caller could pass nonsense and hear nothing at all."""
    with pytest.raises(ValueError, match="runtime_env_pip_cache_size_gb"):
        RayCluster(mode="external-cluster", runtime_env_pip_cache_size_gb=0)
    with pytest.raises(ValueError, match="runtime_env_pip_cache_size_gb"):
        RayCluster(mode="external-cluster", runtime_env_pip_cache_size_gb=-5)


def test_the_warning_is_specific_to_external_cluster(captured, monkeypatch, tmp_path):
    """single-machine DOES export the value, so it must stay quiet about having no effect.

    Construction for that mode touches the filesystem and looks for executables; only the absence
    of the message is asserted, so a construction failure here would surface as an error rather
    than as a vacuous pass.
    """
    monkeypatch.setattr(
        RayCluster, "_find_ray_executable", lambda self: "/usr/bin/true"
    )
    monkeypatch.setattr(
        RayCluster, "_find_serve_executable", lambda self: "/usr/bin/true"
    )
    try:
        # `head_num_cpus` must be positive in this mode, and the class says so by raising. Passing
        # it keeps this arm about the message rather than about an unrelated refusal.
        RayCluster(
            mode="single-machine", runtime_env_pip_cache_size_gb=30, head_num_cpus=1
        )
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            "single-machine construction raised, so this arm could not check the absence of the "
            f"external-cluster message (that would be a vacuous pass): {type(exc).__name__}: {exc}"
        )
    msgs = _messages(captured)
    assert not [
        m for m in msgs if NO_EFFECT in m
    ], f"single-machine must not claim the parameter has no effect; it does: {msgs!r}"
