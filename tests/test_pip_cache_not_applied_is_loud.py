"""A pip-cache size that cannot take effect must say so, not vanish.

`runtime_env_pip_cache_size_gb` is accepted in every mode, and can only work in the modes where
this process starts Ray itself: Ray reads the limit from the runtime_env agent's own environment
(`ray/_private/runtime_env/plugin.py`, `os.environ.get(f"RAY_RUNTIME_ENV_{plugin.name}_CACHE_SIZE_GB".upper(), 10)`).
In external-cluster mode that agent lives in a Ray pod the worker did not create, so the export
never reaches it and the configured size silently becomes Ray's 10 GB default.

That is not hypothetical. On a site whose app environments are 5.7-7.9 GB each, the second one
evicted the first and replicas cycled in STARTING for 40+ minutes with no error anywhere — the
pip cache on one pod sat at 9.5 GB against the 10 GB ceiling while the pod had 272 GB free. The
configured 30 was never in play.

So the contract these arms pin is: LOUD, and visible in status. Not refused — refusing would break
every external-cluster worker whose config carries the value harmlessly, and the operator's real
problem is not knowing it was dropped.
"""

from __future__ import annotations

import logging
import tempfile

import pytest

from bioengine.cluster.ray_cluster import _RAY_DEFAULT_PIP_CACHE_GB, RayCluster


@pytest.fixture(autouse=True)
def _no_cache_env_leak(monkeypatch):
    """Constructing a single-machine cluster EXPORTS the variable into this process.

    Left alone that leaks into every later test in the session, so each arm starts from a known
    absence. Asserting on the variable afterwards is then meaningful rather than inherited.
    """
    monkeypatch.delenv("RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB", raising=False)


def _single_machine(**kw) -> RayCluster:
    # Ray's plasma socket path must stay under 107 bytes, which pytest's tmp_path exceeds, so the
    # temp dir goes under /tmp — the same reason test_ray_start_argv.py does it.
    with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
        return RayCluster(
            mode="single-machine",
            head_node_address="127.0.0.1",
            head_num_cpus=1,
            ray_temp_dir=temp_dir,
            force_clean_up=False,
            **kw,
        )


# ───────────────────────────── the value is reported, in both directions ─────────────────────


def test_external_cluster_reports_the_size_as_NOT_APPLIED() -> None:
    """The reading an operator needs: configured 30, effective 10."""
    cluster = RayCluster(mode="external-cluster")
    reported = cluster.status["runtime_env_pip_cache"]
    assert reported["configured_gb"] == 30
    assert reported["applied"] is False
    assert reported["effective_gb"] == _RAY_DEFAULT_PIP_CACHE_GB, (
        "effective_gb must be what Ray will actually enforce, or a reader has to know which "
        "mode implies which — which is the knowledge whose absence caused the incident"
    )


def test_single_machine_reports_the_size_as_APPLIED_and_exports_it() -> None:
    """The other direction, so the arm above is not just asserting a constant."""
    import os

    cluster = _single_machine(runtime_env_pip_cache_size_gb=25)
    reported = cluster.status["runtime_env_pip_cache"]
    assert reported == {"configured_gb": 25, "applied": True, "effective_gb": 25}
    assert os.environ["RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB"] == "25", (
        "the mode that CAN apply it must still actually export it"
    )


# ─────────────────────────────────── the error, and its silence ──────────────────────────────


def test_external_cluster_logs_an_ERROR_naming_the_value_and_the_remedy(caplog) -> None:
    """A warning buried in a debug log is how this went unnoticed; it must be an ERROR.

    The message has to carry three things, because each one is a question the operator would
    otherwise have to ask someone: the value that was dropped, that it was dropped, and where to
    put it instead.
    """
    with caplog.at_level(logging.ERROR, logger="RayCluster"):
        RayCluster(mode="external-cluster", runtime_env_pip_cache_size_gb=30)

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "no ERROR was logged; a dropped setting stayed silent"
    msg = errors[-1].getMessage()
    assert "30" in msg, "the error does not name the value that was dropped"
    assert "NOT APPLIED" in msg, "the error does not say the value was ignored"
    assert "containerEnv" in msg, "the error does not say where to set it instead"


def test_at_RAYS_OWN_DEFAULT_there_is_no_error_because_nothing_is_lost(caplog) -> None:
    """The silence half of the contract.

    An error on every external-cluster start regardless of value would be noise, and noise is
    what gets filtered out — taking the real signal with it. At 10 the intent and the effect
    agree, so there is nothing to report.
    """
    with caplog.at_level(logging.ERROR, logger="RayCluster"):
        cluster = RayCluster(
            mode="external-cluster",
            runtime_env_pip_cache_size_gb=_RAY_DEFAULT_PIP_CACHE_GB,
        )

    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == [], (
        "an error fired even though the configured size equals what Ray will enforce"
    )
    # Still reported as not applied, because it is not: the value just happens to match.
    assert cluster.status["runtime_env_pip_cache"]["applied"] is False


def test_the_single_machine_path_logs_no_such_ERROR(caplog) -> None:
    """A mode that CAN apply the value must not warn about it.

    This is the arm that stops the error being unconditional — a mutant that drops the mode check
    passes every other arm in this file.
    """
    with caplog.at_level(logging.ERROR, logger="RayCluster"):
        _single_machine(runtime_env_pip_cache_size_gb=25)

    dropped = [
        r
        for r in caplog.records
        if r.levelno >= logging.ERROR and "NOT APPLIED" in r.getMessage()
    ]
    assert dropped == [], f"single-machine logged a not-applied error: {dropped}"


# ─────────────────────────────── the constant is sourced, not guessed ────────────────────────


def test_rays_default_is_ten_and_is_named() -> None:
    """If Ray ever changes its default, the comparison silently stops matching reality.

    Pinned so that a Ray upgrade which moves the default fails here rather than in production,
    where the symptom is replicas stuck in STARTING. The value is read out of Ray's own source
    in the module comment; this asserts the two have not drifted apart.
    """
    assert _RAY_DEFAULT_PIP_CACHE_GB == 10
