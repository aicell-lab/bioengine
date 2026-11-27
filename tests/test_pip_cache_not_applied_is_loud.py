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


class _Collector(logging.Handler):
    """Records every record emitted, at any level."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def ray_log():
    """Capture the RayCluster logger's own records.

    `caplog` CANNOT see them, and that is the dangerous part: `create_logger` sets
    `propagate = False` and attaches its own stderr handler, so caplog's root handler never
    receives anything from this logger. A `caplog`-based absence assertion on this surface
    therefore passes unconditionally -- it reads "no error was logged" from a channel that could
    never have carried one. MEASURED here: the not-applied ERROR appears in captured stderr while
    `caplog.records` is empty.

    So the handler is attached to the logger itself. `create_logger` only removes the handlers it
    attached (it marks its own), which `test_a_handler_the_caller_attached_is_not_removed` pins,
    so this one survives a RayCluster construction that reconfigures the logger.
    """
    logger = logging.getLogger("RayCluster")
    collector = _Collector()
    previous_level = logger.level
    logger.addHandler(collector)
    try:
        yield collector
    finally:
        logger.removeHandler(collector)
        logger.setLevel(previous_level)


def _not_applied_errors(collector: _Collector) -> list[logging.LogRecord]:
    return [
        r
        for r in collector.records
        if r.levelno >= logging.ERROR and "NOT APPLIED" in r.getMessage()
    ]


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
    assert reported["requested_gb"] is None, (
        "nobody passed a size here, and status has to say so: an operator reading 'not applied' "
        "needs to know whether it was their own value that got dropped"
    )
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
    assert reported == {
        "configured_gb": 25,
        "requested_gb": 25,
        "applied": True,
        "effective_gb": 25,
    }
    assert (
        os.environ["RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB"] == "25"
    ), "the mode that CAN apply it must still actually export it"


# ─────────────────────────────────── the error, and its silence ──────────────────────────────


def test_external_cluster_logs_an_ERROR_naming_the_value_and_the_remedy(
    ray_log,
) -> None:
    """A warning buried in a debug log is how this went unnoticed; it must be an ERROR.

    The message has to carry three things, because each one is a question the operator would
    otherwise have to ask someone: the value that was dropped, that it was dropped, and where to
    put it instead.
    """
    RayCluster(mode="external-cluster", runtime_env_pip_cache_size_gb=30)

    assert (
        ray_log.records
    ), "the logger emitted nothing at all; the capture channel is wrong"
    errors = [r for r in ray_log.records if r.levelno >= logging.ERROR]
    assert errors, (
        "a dropped setting was not reported at ERROR. Emitted instead: "
        f"{[(logging.getLevelName(r.levelno), r.getMessage()[:60]) for r in ray_log.records]}"
    )
    msg = errors[-1].getMessage()
    assert "30" in msg, "the error does not name the value that was dropped"
    assert "NOT APPLIED" in msg, "the error does not say the value was ignored"
    assert "containerEnv" in msg, "the error does not say where to set it instead"


def test_at_RAYS_OWN_DEFAULT_there_is_no_error_because_nothing_is_lost(ray_log) -> None:
    """The silence half of the contract.

    An error on every external-cluster start regardless of value would be noise, and noise is
    what gets filtered out — taking the real signal with it. At 10 the intent and the effect
    agree, so there is nothing to report.
    """
    cluster = RayCluster(
        mode="external-cluster",
        runtime_env_pip_cache_size_gb=_RAY_DEFAULT_PIP_CACHE_GB,
    )

    assert (
        _not_applied_errors(ray_log) == []
    ), "an error fired even though the configured size equals what Ray will enforce"
    # Still reported as not applied, because it is not: the value just happens to match.
    assert cluster.status["runtime_env_pip_cache"]["applied"] is False


def test_an_operator_who_never_ASKED_for_a_size_gets_no_error(ray_log) -> None:
    """The arm that keeps this from becoming an alarm nobody reads.

    `--runtime-env-pip-cache-size-gb` carries no argparse default and the worker drops None
    arguments, so an operator who never passes the flag lands on this constructor's own default.
    On a KubeRay site that is EVERY pod. If the error fired on the default too it would be in
    every healthy worker's log forever, including the pods whose operators set
    RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB correctly on the RayCluster, and the one occurrence that
    means something would be indistinguishable from the noise.

    Nothing is lost in this case: no human asked for anything.
    """
    cluster = RayCluster(mode="external-cluster")

    dropped = _not_applied_errors(ray_log)
    assert (
        dropped == []
    ), f"an error fired for a size nobody asked for: {[r.getMessage() for r in dropped]}"
    # Still honestly reported as not applied -- silent is not the same as applied.
    assert cluster.status["runtime_env_pip_cache"]["applied"] is False


def test_a_size_that_cannot_work_is_still_REFUSED_in_every_mode() -> None:
    """The validation used to live beside the export, so only self-started modes ran it.

    An external-cluster caller could pass 0 and have it accepted in silence. The value is
    nonsense wherever it is written, so it is rejected in every mode now.
    """
    with pytest.raises(ValueError, match="greater than 0"):
        RayCluster(mode="external-cluster", runtime_env_pip_cache_size_gb=0)
    with pytest.raises(ValueError, match="greater than 0"):
        _single_machine(runtime_env_pip_cache_size_gb=-1)


def test_the_slurm_path_logs_no_such_ERROR(ray_log) -> None:
    """The second mode that CAN apply the value, and the one a mode check is easy to get wrong.

    A guard written as `mode != "single-machine"` passes every other arm in this file while
    telling a SLURM operator their size was dropped when it was in fact exported.

    This host has no SLURM, so the constructor raises once it reaches the SLURM branch. That
    raise is the point: the branch sits AFTER the not-applied report, so reaching it proves the
    report site was executed and skipped, rather than this arm passing because construction died
    before getting there.
    """
    with pytest.raises(RuntimeError, match="SLURM is not available"):
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            RayCluster(
                mode="slurm",
                runtime_env_pip_cache_size_gb=25,
                ray_temp_dir=temp_dir,
                force_clean_up=False,
                worker_workspace_dir=temp_dir,
            )

    dropped = _not_applied_errors(ray_log)
    assert dropped == [], f"slurm logged a not-applied error: {dropped}"


def test_the_single_machine_path_logs_no_such_ERROR(ray_log) -> None:
    """A mode that CAN apply the value must not warn about it.

    This is the arm that stops the error being unconditional — a mutant that drops the mode check
    passes every other arm in this file.
    """
    _single_machine(runtime_env_pip_cache_size_gb=25)

    dropped = _not_applied_errors(ray_log)
    assert dropped == [], f"single-machine logged a not-applied error: {dropped}"


# ─────────────────────────────── the constant is sourced, not guessed ────────────────────────


def test_rays_default_is_ten_and_is_named() -> None:
    """If Ray ever changes its default, the comparison silently stops matching reality.

    Pinned so that a Ray upgrade which moves the default fails here rather than in production,
    where the symptom is replicas stuck in STARTING. The value is read out of Ray's own source
    in the module comment; this asserts the two have not drifted apart.
    """
    assert _RAY_DEFAULT_PIP_CACHE_GB == 10
