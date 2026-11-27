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
import os

import pytest

import bioengine.cluster.ray_cluster as rc_mod
from bioengine.cluster.ray_cluster import (
    DEFAULT_RUNTIME_ENV_PIP_CACHE_SIZE_GB,
    SELF_STARTED_MODES,
    RayCluster,
)

# Spelled out HERE, deliberately, rather than parametrizing over the module's own
# ``SELF_STARTED_MODES``. Importing the tuple and parametrizing over it makes the arms
# agree with the code even when both are wrong: shrinking the module tuple to
# ``("single-machine",)`` would silently COLLECT ZERO slurm cases, so the arm that guards
# slurm is deleted by the very bug it guards against, and the suite still reports green.
# The modes are a fact about the worker, so the test states the fact and then checks that
# the module still agrees with it.
MODES_THAT_APPLY_THE_VALUE = ("single-machine", "slurm")

IGNORED = "is IGNORED in"
ENV_VAR = "RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB"
PASSED_SIZE = 42

# What the variable looked like BEFORE any test in this file ran. pytest imports test modules at
# collection, so this is the pre-run state. The leak check at the end of the file compares against
# it rather than against absence: a session that legitimately has the variable set in its own
# environment must not make this file fail, and "absent" is only the usual case, not the invariant.
ENV_VAR_BEFORE_ANY_TEST = os.environ.get(ENV_VAR)


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
def no_cache_env_leak():
    """Stop a construction that EXPORTS the variable from leaking it into later tests.

    ``single-machine`` and ``slurm`` construction really does
    ``os.environ["RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB"] = ...`` in this process, so without this the
    value survives the test and every later test in the session inherits it.

    This does NOT use ``monkeypatch.delenv``. Deleting a variable that is ABSENT records nothing
    to undo, so monkeypatch has no teardown to run and whatever the construction then writes
    simply stays — which is precisely what the earlier version of this fixture did while claiming
    the opposite. Restoring the pre-test state explicitly is the only version that holds when the
    variable starts out absent, which is the normal case.
    """
    before = os.environ.pop(ENV_VAR, None)
    try:
        yield
    finally:
        os.environ.pop(ENV_VAR, None)  # drop whatever the construction exported
        if before is not None:
            os.environ[ENV_VAR] = before  # and put back what was there first


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


@pytest.mark.parametrize("mode", MODES_THAT_APPLY_THE_VALUE)
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


@pytest.mark.parametrize("mode", MODES_THAT_APPLY_THE_VALUE)
def test_the_value_is_actually_exported_in_a_mode_that_applies_it(
    mode, captured, self_started, monkeypatch
):
    """The one thing the other arms never check: that the EXPORT happens.

    Every other arm here reads the log message or the status field, and both of those are computed
    from the mode. So narrowing the export guard alone — leaving `applied` reporting True while
    nothing is written to the environment — passed the whole suite: a worker would claim the cap
    was applied while Ray never saw it. `os.environ` is the only witness that can tell the claim
    from the act.

    The real export this triggers is undone by `no_cache_env_leak`, which `self_started` pulls
    in. That fixture restores the pre-test state EXPLICITLY rather than through
    `monkeypatch.delenv`, because deleting an absent variable records nothing to undo and so
    undoes nothing -- see its docstring, and the arm at the end of this file that proves it.
    """
    # NOT a leak detector: `no_cache_env_leak` has just removed the variable, so this cannot
    # fail while that fixture is in place. It is a guard against the fixture being REMOVED or
    # reordered — then a value left behind by an earlier test would make the assertion below pass
    # without this construction having exported anything. Leak detection is
    # `test_no_test_in_this_file_leaves_the_cache_variable_behind`, at the end of the file.
    assert (
        ENV_VAR not in os.environ
    ), f"{ENV_VAR} was set before construction, so finding it afterwards would prove nothing"

    self_started(mode, runtime_env_pip_cache_size_gb=PASSED_SIZE)

    assert os.environ.get(ENV_VAR, "<ABSENT>") == str(PASSED_SIZE), (
        f"{mode} mode reports the cap as applied but did not export {ENV_VAR}; Ray reads the cap "
        f"from the environment, so an unexported value is not applied. os.environ has "
        f"{os.environ.get(ENV_VAR, '<ABSENT>')!r}"
    )


@pytest.mark.parametrize("mode", MODES_THAT_APPLY_THE_VALUE)
def test_status_reports_the_size_in_every_mode_that_exports_it(
    mode, captured, self_started
):
    """Common to both self-started modes: the size itself is reported back.

    What DIFFERS between them is where the cap reaches, which the two arms below pin separately.
    This arm deliberately asserts nothing about `applied` or `state`, because a single expectation
    for both modes is what previously let slurm claim the cap was cluster-wide.
    """
    cluster = self_started(mode, runtime_env_pip_cache_size_gb=PASSED_SIZE)

    cache = cluster.status["runtime_env_pip_cache"]
    assert cache["size_gb"] == PASSED_SIZE, cache
    assert cache["requested"] == PASSED_SIZE, cache


def test_single_machine_reports_the_cap_as_cluster_wide(captured, self_started):
    """One host, so exporting it here really does govern every Ray process in the cluster."""
    cluster = self_started("single-machine", runtime_env_pip_cache_size_gb=PASSED_SIZE)

    cache = cluster.status["runtime_env_pip_cache"]
    assert cache.get("applied", "<ABSENT>") is True, cache
    assert cache["scope"] == "cluster", cache
    assert cache["state"] == "applied", cache
    assert cache["detail"] is None, cache


def test_slurm_reports_the_cap_as_reaching_the_head_node_only(captured, self_started):
    """The defect this row exists for: slurm exports the value and governs the HEAD ONLY.

    `slurm_workers.py` launches the compute-node raylets under `apptainer exec --cleanenv`, which
    starts from an empty environment and is handed back only SLURM_JOB_ID and PYTHONNOUSERSITE.
    So every compute node keeps Ray's own 10 GB however large a value was passed, and a status
    field saying a bare `applied: true` tells an operator something false about the nodes their
    replicas actually run on.
    """
    cluster = self_started("slurm", runtime_env_pip_cache_size_gb=PASSED_SIZE)

    cache = cluster.status["runtime_env_pip_cache"]
    assert cache.get("applied", "<ABSENT>") is False, (
        "`applied` means in force for the WHOLE cluster, and on slurm the compute nodes are on a "
        f"different cap: {cache}"
    )
    assert cache["scope"] == "head-only", cache
    assert cache["state"] == "applied on the head node only", cache
    detail = cache["detail"] or ""
    assert "--cleanenv" in detail, f"the detail must name the mechanism: {detail!r}"
    assert "10 GB" in detail, (
        f"it must name what the compute nodes actually use, which is the figure to size scratch "
        f"from: {detail!r}"
    )


def test_the_slurm_and_single_machine_readings_are_not_the_same(captured, self_started):
    """Guards against the two collapsing back into one answer.

    They were one answer before this change, and the wrong one for slurm. An arm per mode would
    both pass if someone unified them on the single-machine wording and updated each arm to
    match; this one fails unless they genuinely differ.
    """
    sm = self_started(
        "single-machine", runtime_env_pip_cache_size_gb=PASSED_SIZE
    ).status["runtime_env_pip_cache"]
    sl = self_started("slurm", runtime_env_pip_cache_size_gb=PASSED_SIZE).status[
        "runtime_env_pip_cache"
    ]

    assert (sm["scope"], sm["applied"], sm["state"]) != (
        sl["scope"],
        sl["applied"],
        sl["state"],
    ), (
        "single-machine and slurm report the cap identically, but the cap does not reach the same "
        f"nodes in the two modes: single-machine={sm}, slurm={sl}"
    )


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
    assert cache["scope"] == "none", cache
    assert cache["state"] == "not applied", cache
    assert cache["requested"] == PASSED_SIZE, cache
    assert cache["size_gb"] == PASSED_SIZE, cache
    assert ENV_VAR in (cache["detail"] or ""), cache


# --- 5. a bad value is refused everywhere -------------------------------------------------------


@pytest.mark.parametrize("mode", ["external-cluster", *MODES_THAT_APPLY_THE_VALUE])
@pytest.mark.parametrize("bad", [0, -5])
def test_a_bad_value_is_refused_in_every_mode(mode, bad, captured, self_started):
    """The validation used to live inside the branch that consumes the value, so an
    external-cluster caller could pass nonsense and hear nothing at all."""
    with pytest.raises(ValueError, match="runtime_env_pip_cache_size_gb"):
        if mode == "external-cluster":
            RayCluster(mode=mode, runtime_env_pip_cache_size_gb=bad)
        else:
            self_started(mode, runtime_env_pip_cache_size_gb=bad)


def test_the_module_still_agrees_about_which_modes_apply_the_value():
    """Guards the guard. These arms spell the modes themselves so that shrinking
    ``SELF_STARTED_MODES`` cannot delete the slurm case by emptying a parametrize list. That
    protection is only worth anything if something still notices when the module's own tuple
    stops matching — otherwise the code could quietly drop slurm while every arm above kept
    testing a mode the worker no longer treats that way.
    """
    assert tuple(SELF_STARTED_MODES) == MODES_THAT_APPLY_THE_VALUE, (
        "the worker's SELF_STARTED_MODES no longer matches the modes these arms cover: "
        f"module says {tuple(SELF_STARTED_MODES)!r}, arms cover "
        f"{MODES_THAT_APPLY_THE_VALUE!r}. If that change is intended, update both."
    )


@pytest.fixture(scope="module", autouse=True)
def _cache_variable_is_left_as_it_was_found():
    """The leak detector, as a module-scoped teardown rather than as a test.

    The arms here construct self-started clusters, which really export
    ``RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB`` into this process. If their cleanup does not work the
    value is still set afterwards, and every later test in the session inherits it -- which can
    turn an unrelated test green or red for a reason nobody will find.

    This was a test declared last in the file, which the review rightly flagged: ordering holds
    under this repo's ``--dist=no -n1`` addopts, but under ``-n>1`` load distribution the detector
    could be handed to a worker that ran none of the exporting arms, and then it checks nothing
    while reporting green. A module-scoped teardown cannot be separated from the arms that way:
    whichever worker runs any of them runs this, for exactly the subset it ran.

    It compares against the value captured at setup, not against absence, so a session that
    legitimately has the variable set in its own environment does not fail spuriously.
    """
    before = os.environ.get(ENV_VAR)
    yield
    after = os.environ.get(ENV_VAR)
    assert after == before, (
        f"{ENV_VAR} is {after!r} but was {before!r} when this module's tests began, so the tests "
        "in it -- or, under parallel distribution, the subset of them that ran on this worker -- "
        "leak it into every later test here. The no_cache_env_leak fixture is meant to restore "
        "the pre-test state; note that monkeypatch.delenv on an ABSENT variable records nothing "
        "to undo, which is how this leak existed while a fixture claimed to prevent it."
    )
