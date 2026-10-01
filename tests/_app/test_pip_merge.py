"""Unit tests for the runtime_env pip-list merge in
:func:`bioengine._app.bootstrap._merge_pip_lists`.

The merge runs at bind time inside ``build_and_run_application`` and is
how the framework's required deps make it onto every user replica's venv
alongside whatever the user declared via ``@bioengine.app(pip=…)``. In
production that list is ``hypha-rpc`` and ``httpx`` (see
``bioengine.apps.builder._USER_REPLICA_FRAMEWORK_PACKAGES``); the lists below
are test data, chosen so ordering and collision are exercised separately.
Two invariants matter:

* A framework dep is added if it's not already present.
* A framework dep *replaces* a user entry on the same package name. The
  worker dictates those versions; an app pinning its own is how two
  hypha-rpc versions ended up running on one node.
"""
from __future__ import annotations

import logging

import pytest

from bioengine._app.bootstrap import _merge_pip_lists
from bioengine.utils.requirements import requirement_name


class _WarningCollector(logging.Handler):
    """Collect ``ray.serve`` warnings on the ``ray.serve`` logger itself.

    ``import ray.serve`` runs ``configure_default_serve_logger``, which sets
    ``propagate = False`` and installs its own stderr handler — and
    ``tests/conftest.py`` reaches it via ``bioengine.cluster.ray_cluster``, so
    the logger is always in that state here. Whether pytest's ``caplog``
    handler, which is attached to the *root* logger, still sees a
    non-propagating record is a pytest-version detail (9.0.0 does not, 9.1.1
    does), so reading ``caplog.records`` asserts on the runner rather than on
    the code under test.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def serve_warnings():
    """Every ``ray.serve`` warning emitted inside the test, in order."""
    logger = logging.getLogger("ray.serve")
    handler = _WarningCollector()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield handler.messages
    finally:
        logger.setLevel(previous_level)
        logger.removeHandler(handler)


@pytest.mark.parametrize(
    "req,expected",
    [
        ("hypha-rpc", "hypha-rpc"),
        ("hypha-rpc==0.21.40", "hypha-rpc"),
        ("hypha-rpc>=0.21.40", "hypha-rpc"),
        ("hypha-rpc<=1.0.0", "hypha-rpc"),
        ("hypha-rpc~=0.21.0", "hypha-rpc"),
        ("hypha-rpc>0.21", "hypha-rpc"),
        ("hypha-rpc<2", "hypha-rpc"),
        ("httpx[http2]==0.28.1", "httpx"),
        ("Pandas==2.2.0", "pandas"),
        ("  spaces==1.0  ", "spaces"),
        ("hypha_rpc==0.21.40", "hypha-rpc"),
        ("hypha.rpc==0.21.40", "hypha-rpc"),
        ("HYPHA-RPC==0.21.40", "hypha-rpc"),
    ],
)
def test_requirement_name_extracts_package(req: str, expected: str) -> None:
    """The comparison the merge below is built on. It is
    :func:`bioengine.utils.requirements.requirement_name` — the repo's single
    normaliser — not a private copy in ``bootstrap``; the copy did not fold
    ``_``/``.`` to ``-``, so every underscore spelling escaped the override."""
    assert requirement_name(req) == expected


def test_appends_missing_framework_deps() -> None:
    merged = _merge_pip_lists(
        ["pandas==2.2.0"],
        ["hypha-rpc==0.21.40", "pydantic==2.12.0"],
    )
    assert merged == [
        "pandas==2.2.0",
        "hypha-rpc==0.21.40",
        "pydantic==2.12.0",
    ]


def test_framework_pin_overrides_user_pin() -> None:
    """The drift the override exists to stop: an app pinning an older
    hypha-rpc than the worker it unpickles deployments from."""
    merged = _merge_pip_lists(
        ["hypha-rpc==0.20.0", "pandas==2.2.0"],
        ["hypha-rpc==0.21.40", "pydantic==2.12.0"],
    )
    assert merged == [
        "hypha-rpc==0.21.40",
        "pandas==2.2.0",
        "pydantic==2.12.0",
    ]


def test_framework_pin_overrides_user_floor() -> None:
    """A floor is the worse case: it has no fixed version at all, so Ray
    resolves it against PyPI afresh on every runtime_env build."""
    merged = _merge_pip_lists(
        ["hypha-rpc>=0.21.40"],
        ["hypha-rpc==0.21.40"],
    )
    assert merged == ["hypha-rpc==0.21.40"]


def test_framework_pin_overrides_unpinned_user_entry() -> None:
    merged = _merge_pip_lists(
        ["pydantic"],
        ["pydantic==2.12.0"],
    )
    assert merged == ["pydantic==2.12.0"]


def test_user_entries_the_framework_does_not_own_are_untouched() -> None:
    merged = _merge_pip_lists(
        ["torch>=2.0", "pandas", "scikit-image==0.24.0"],
        ["hypha-rpc==0.21.40"],
    )
    assert merged == [
        "torch>=2.0",
        "pandas",
        "scikit-image==0.24.0",
        "hypha-rpc==0.21.40",
    ]


def test_override_matches_across_extras() -> None:
    """``httpx[http2]`` and ``httpx`` are the same distribution — an extras
    spec must not smuggle a second version past the override."""
    merged = _merge_pip_lists(
        ["httpx[http2]==0.27.0"],
        ["httpx[http2]==0.28.1"],
    )
    assert merged == ["httpx[http2]==0.28.1"]


def test_empty_base_returns_framework_list() -> None:
    merged = _merge_pip_lists(
        [],
        ["hypha-rpc==0.21.40", "pydantic==2.12.0"],
    )
    assert merged == ["hypha-rpc==0.21.40", "pydantic==2.12.0"]


def test_empty_framework_returns_base() -> None:
    merged = _merge_pip_lists(["pandas==2.2.0"], [])
    assert merged == ["pandas==2.2.0"]


def test_idempotent_when_framework_already_in_base() -> None:
    merged = _merge_pip_lists(
        ["hypha-rpc==0.21.40", "pydantic==2.12.0"],
        ["hypha-rpc==0.21.40", "pydantic==2.12.0"],
    )
    assert merged == ["hypha-rpc==0.21.40", "pydantic==2.12.0"]


def test_override_is_logged_with_both_versions(serve_warnings) -> None:
    """A silent override trades one invisible version for another. The app
    author has to be able to see that their pin did not take, and what it
    was replaced with."""
    _merge_pip_lists(["hypha-rpc==0.20.0"], ["hypha-rpc==0.21.40"])
    logged = "\n".join(serve_warnings)
    assert "hypha-rpc==0.20.0" in logged
    assert "hypha-rpc==0.21.40" in logged


def test_no_log_when_the_app_already_asked_for_the_enforced_version(
    serve_warnings,
) -> None:
    """An app that agrees with the worker is not doing anything wrong, and a
    warning on every build would train people to ignore the one above.

    The test above is this one's positive control: it proves the same fixture
    does see a warning when there is one, so an empty list here is silence
    rather than a capture that never worked."""
    _merge_pip_lists(["hypha-rpc==0.21.40"], ["hypha-rpc==0.21.40"])
    _merge_pip_lists(["pandas==2.2.0"], ["hypha-rpc==0.21.40"])
    assert serve_warnings == []


def test_case_insensitive_name_match() -> None:
    """PEP 503 folds case, so ``Pydantic`` and ``pydantic`` are one package
    and must not both reach the replica."""
    merged = _merge_pip_lists(
        ["Pydantic==2.12.0"],
        ["pydantic==2.10.0"],
    )
    assert merged == ["pydantic==2.10.0"]


@pytest.mark.parametrize(
    "spelling",
    ["hypha_rpc", "hypha.rpc", "HYPHA-RPC", "hypha-rpc"],
)
def test_every_spelling_of_hypha_rpc_is_overridden_by_the_workers_pin(
    spelling: str, serve_warnings
) -> None:
    """PEP 503 says these four strings name one distribution, and pip agrees.
    The override is a name match, so a spelling it failed to recognise did not
    merely go un-normalised — it let the app's pin through *and* appended the
    worker's, so the replica venv received two entries for one package and
    whichever pip resolved was outside the worker's control. ``hypha_rpc`` is
    the import name, so it is the spelling an author types from memory.

    The framework version is a literal the test owns. It used to be read from
    ``get_pip_requirements``, which resolves it off the *installed bioengine
    distribution*'s metadata — so in a checkout with no bioengine installed
    this test failed on the environment rather than on the merge. That the
    worker's real list carries the installed hypha-rpc is asserted in
    ``tests/test_replica_framework_pins.py``; here only the name match matters.
    """
    framework_pip = ["hypha-rpc==9.9.9-probe"]

    merged = _merge_pip_lists([f"{spelling}==0.0.1", "pandas==2.2.0"], framework_pip)

    # Asserted first, and counted without the normaliser under test, so a
    # broken one cannot hide a second entry and the failure output names the
    # defect — both spellings in one list — rather than a list mismatch.
    assert [req for req in merged if "rpc" in req.lower()] == ["hypha-rpc==9.9.9-probe"]
    assert merged == ["hypha-rpc==9.9.9-probe", "pandas==2.2.0"]

    logged = "\n".join(serve_warnings)
    assert f"{spelling}==0.0.1" in logged
    assert "hypha-rpc==9.9.9-probe" in logged
