"""A parameter that cannot take effect must say so loudly, and say so in the status too.

``runtime_env_pip_cache_size_gb`` is an ``__init__`` argument in every mode. It is applied by
exporting ``RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB``, which only reaches the Ray processes THIS worker
starts — ``single-machine`` and ``slurm``. In ``external-cluster`` mode the Ray pods already exist
and were created by the cluster operator, so the export cannot reach them and the pods run on
Ray's own default of 10 GB.

WHY IT MATTERS, measured on a live external-cluster site: the Ray workers held app environments of
5.7-7.9 GB each and evicted one another continuously (21-22 cache deletes and 29-31 adds per pod in
one session), so a freshly deployed app's replicas sat in ``STARTING`` indefinitely while gigabytes
were installed and discarded. The operator had set this parameter and had no way to learn it did
nothing — the silence is the defect, not the eviction.

THREE THINGS THESE ARMS PIN, each of which a plausible implementation gets wrong:

1. The startup message is at ERROR level. A message emitted at a level a handler may drop is the
   same silence with extra steps, so the LEVEL is asserted and not merely the text.
2. It fires only for a value the caller PASSED. ``--runtime-env-pip-cache-size-gb`` has no
   argparse default and the worker drops ``None`` arguments, so a defaulted value would otherwise
   raise this alarm on every external-cluster pod start, including the ones whose pods are
   configured correctly. An always-on alarm stops being read.
3. It does NOT fire in the modes that DO apply the value — ``single-machine`` AND ``slurm``. Both
   are parametrized, because a guard written as ``mode != "single-machine"`` passes a
   single-machine-only arm while telling every SLURM operator a falsehood.

The logger is captured by replacing ``create_logger`` BEFORE construction, because the message is
emitted inside ``__init__``; attaching a handler afterwards would miss it, and relying on
propagation would make the arm depend on how the logger happens to be configured. The handler
keeps the RECORDS rather than formatted strings, because a formatted string has thrown the level
away.
"""

from __future__ import annotations

import logging

import pytest

import bioengine.cluster.ray_cluster as rc_mod
from bioengine.cluster.ray_cluster import (
    DEFAULT_RUNTIME_ENV_PIP_CACHE_SIZE_GB,
    SELF_STARTED_MODES,
    RayCluster,
)

IGNORED = "is IGNORED in"
ENV_VAR = "RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB"
PASSED_SIZE = 42


@pytest.fixture
def captured(monkeypatch):
    """Collect every record the cluster logs during construction, with its level."""
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    def fake_create_logger(*args, **kwargs):
        logger = logging.getLogger(f"test-pip-cache-{id(records)}")
        logger.handlers = [_Collect()]
        logger.setLevel(logging.DEBUG)  # collect everything; the arms assert the level
        logger.propagate = False
        return logger

    monkeypatch.setattr(rc_mod, "create_logger", fake_create_logger)
    return records


@pytest.fixture
def no_cache_env_leak(monkeypatch):
    """Stop a construction that EXPORTS the variable from leaking it into later tests.

    ``single-machine`` and ``slurm`` construction really does
    ``os.environ["RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB"] = ...`` in this process. Recording the
    variable with monkeypatch first means its pre-test state — usually absent — is restored at
    teardown, so no later test inherits a value this file set.
    """
    monkeypatch.delenv(ENV_VAR, raising=False)


@pytest.fixture
def self_started(monkeypatch, tmp_path, no_cache_env_leak):
    """Build a cluster in a mode where the worker starts Ray itself.

    Construction for those modes looks for executables and, for slurm, for SLURM itself. Those
    are replaced so an arm is about the pip-cache message rather than about an unrelated refusal.
    """

    def build(mode, **kwargs):
        monkeypatch.setattr(
            RayCluster, "_find_ray_executable", lambda self: "/usr/bin/true"
        )
        monkeypatch.setattr(
            RayCluster, "_find_serve_executable", lambda self: "/usr/bin/true"
        )
        monkeypatch.setattr(RayCluster, "_check_slurm_available", lambda self: None)
        if mode == "single-machine":
            # Positive in this mode by the class's own rule, which it enforces by raising.
            kwargs.setdefault("head_num_cpus", 1)
        if mode == "slurm":
            kwargs.setdefault("worker_workspace_dir", str(tmp_path / "ws"))
        return RayCluster(mode=mode, **kwargs)

    return build


def _ignored_records(records):
    return [r for r in records if IGNORED in str(r.msg)]


# --- 1. the message, its level, and what it has to name -----------------------------------------


def test_a_passed_size_is_an_error_in_external_cluster_mode(captured):
    """Level included: a message nobody is required to surface is the original silence."""
    RayCluster(mode="external-cluster", runtime_env_pip_cache_size_gb=PASSED_SIZE)

    hits = _ignored_records(captured)
    assert hits, (
        "external-cluster mode accepted a passed runtime_env_pip_cache_size_gb in silence; "
        f"logged messages were: {[r.getMessage() for r in captured]!r}"
    )
    worst = max(r.levelno for r in hits)
    assert worst >= logging.ERROR, (
        "the message must be an ERROR, not something a handler may drop: highest level was "
        f"{logging.getLevelName(worst)} ({worst})"
    )
    said = next(r.getMessage() for r in hits if r.levelno >= logging.ERROR)
    assert str(PASSED_SIZE) in said, f"it must quote the ignored value: {said!r}"
    assert ENV_VAR in said, f"it must name the variable to set: {said!r}"


def test_the_error_names_where_the_value_must_actually_be_set(captured):
    """A message that does not say what to do instead is only half a message."""
    RayCluster(mode="external-cluster", runtime_env_pip_cache_size_gb=PASSED_SIZE)

    said = next(
        r.getMessage() for r in _ignored_records(captured) if r.levelno >= logging.ERROR
    )
    assert "pods" in said, f"it must point at the Ray pods: {said!r}"
    assert (
        "STARTING" in said
    ), f"it must name the symptom an operator will actually see: {said!r}"


def test_the_error_does_not_assert_what_the_pods_cache_size_actually_is(captured):
    """It must not tell an operator a falsehood about their own cluster.

    This worker can neither read nor set the Ray pods' environment, so it cannot know whether
    RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB is set there. An operator who HAS set it correctly would
    otherwise be told flatly that their pods are running on Ray's default — by a message whose
    entire purpose is to be believed. Ray's default is still worth naming; it has to be named as
    the consequence of the variable being unset, not as a reading of the pods.
    """
    RayCluster(mode="external-cluster", runtime_env_pip_cache_size_gb=PASSED_SIZE)

    said = next(
        r.getMessage() for r in _ignored_records(captured) if r.levelno >= logging.ERROR
    )
    assert "cannot be applied from here" in said, (
        "it must say the value cannot be applied here rather than describing the pods' state: "
        f"{said!r}"
    )
    assert (
        "cannot see their environment" in said
    ), f"it must say the worker cannot see the pods' environment: {said!r}"
    assert "where that variable is unset" in said, (
        "Ray's 10 GB default must be conditional on the variable being unset there, not asserted "
        f"as the size in force: {said!r}"
    )


# --- 2. passed vs defaulted ---------------------------------------------------------------------


def test_a_value_nobody_passed_does_not_raise_an_alarm(captured):
    """The flag has no argparse default and the worker drops None arguments, so a defaulted value
    would otherwise make every external-cluster worker shout on every pod start — including the
    ones whose pods are configured correctly."""
    cluster = RayCluster(mode="external-cluster")

    loud = [r for r in _ignored_records(captured) if r.levelno >= logging.WARNING]
    assert not loud, (
        "a value nobody passed must not raise an alarm, or the alarm stops being read: "
        f"{[(logging.getLevelName(r.levelno), r.getMessage()) for r in loud]!r}"
    )
    # ...and the fact is still reported, so staying quiet hides nothing.
    cache = cluster.status["runtime_env_pip_cache"]
    assert cache["state"] == "not applied", cache
    assert cache["requested"] is None, cache
    assert cache["size_gb"] == DEFAULT_RUNTIME_ENV_PIP_CACHE_SIZE_GB, cache


# --- 3. the modes that DO apply it stay quiet ---------------------------------------------------


@pytest.mark.parametrize("mode", SELF_STARTED_MODES)
def test_no_ignored_message_in_a_mode_that_applies_the_value(
    mode, captured, self_started
):
    """Both self-started modes, because a guard written as `mode != "single-machine"` passes a
    single-machine-only arm while telling every SLURM operator a falsehood."""
    self_started(mode, runtime_env_pip_cache_size_gb=PASSED_SIZE)

    hits = _ignored_records(captured)
    assert not hits, (
        f"{mode} mode DOES export the value, so it must not claim the value is ignored: "
        f"{[(logging.getLevelName(r.levelno), r.getMessage()) for r in hits]!r}"
    )


@pytest.mark.parametrize("mode", SELF_STARTED_MODES)
def test_status_reports_applied_in_a_mode_that_applies_the_value(
    mode, captured, self_started
):
    cluster = self_started(mode, runtime_env_pip_cache_size_gb=PASSED_SIZE)

    cache = cluster.status["runtime_env_pip_cache"]
    assert cache.get("applied", "<ABSENT>") is True, cache
    assert cache["state"] == "applied", cache
    assert cache["size_gb"] == PASSED_SIZE, cache


# --- 4. the status surface ----------------------------------------------------------------------


def test_status_reports_not_applied_on_an_external_cluster(captured):
    """The startup line scrolls away; the operator asking "did my setting take?" reads status."""
    cluster = RayCluster(
        mode="external-cluster", runtime_env_pip_cache_size_gb=PASSED_SIZE
    )

    status = cluster.status
    assert "runtime_env_pip_cache" in status, sorted(status)
    cache = status["runtime_env_pip_cache"]
    # An absent key and a False value are indistinguishable through `.get()`, so compare against
    # a sentinel rather than letting a missing field read as a passing assertion.
    assert cache.get("applied", "<ABSENT>") is False, cache
    assert cache["state"] == "not applied", cache
    assert cache["requested"] == PASSED_SIZE, cache
    assert cache["size_gb"] == PASSED_SIZE, cache
    assert ENV_VAR in (cache["detail"] or ""), cache


# --- 5. a bad value is refused everywhere -------------------------------------------------------


@pytest.mark.parametrize("mode", ["external-cluster", *SELF_STARTED_MODES])
@pytest.mark.parametrize("bad", [0, -5])
def test_a_bad_value_is_refused_in_every_mode(mode, bad, captured, self_started):
    """The validation used to live inside the branch that consumes the value, so an
    external-cluster caller could pass nonsense and hear nothing at all."""
    with pytest.raises(ValueError, match="runtime_env_pip_cache_size_gb"):
        if mode == "external-cluster":
            RayCluster(mode=mode, runtime_env_pip_cache_size_gb=bad)
        else:
            self_started(mode, runtime_env_pip_cache_size_gb=bad)
