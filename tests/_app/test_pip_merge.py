"""Unit tests for the runtime_env pip-list merge in
:func:`bioengine._app.bootstrap._merge_pip_lists`.

The merge runs at bind time inside ``build_and_run_application`` and is
how the framework's required deps make it onto every user replica's venv
alongside whatever the user declared via ``@bioengine.app(pip=…)``. In
production that list is ``hypha-rpc`` alone; the two-element lists below
are test data, chosen so ordering and collision are exercised separately.
Two invariants matter:

* A framework dep is added if it's not already present.
* A framework dep *replaces* a user entry on the same package name. Those
  packages cross the cloudpickle boundary, so the worker's version is the
  only one that works; an app pinning its own is how two hypha-rpc
  versions ended up running on one node.
"""
from __future__ import annotations

import logging

import pytest

from bioengine._app.bootstrap import _merge_pip_lists, _requirement_name


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
    ],
)
def test_requirement_name_extracts_package(req: str, expected: str) -> None:
    assert _requirement_name(req) == expected


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


def test_override_is_logged_with_both_versions(caplog) -> None:
    """A silent override trades one invisible version for another. The app
    author has to be able to see that their pin did not take, and what it
    was replaced with."""
    with caplog.at_level(logging.WARNING, logger="ray.serve"):
        _merge_pip_lists(["hypha-rpc==0.20.0"], ["hypha-rpc==0.21.40"])
    logged = "\n".join(r.message for r in caplog.records)
    assert "hypha-rpc==0.20.0" in logged
    assert "hypha-rpc==0.21.40" in logged


def test_no_log_when_the_app_already_asked_for_the_enforced_version(caplog) -> None:
    """An app that agrees with the worker is not doing anything wrong, and a
    warning on every build would train people to ignore the one above."""
    with caplog.at_level(logging.WARNING, logger="ray.serve"):
        _merge_pip_lists(["hypha-rpc==0.21.40"], ["hypha-rpc==0.21.40"])
        _merge_pip_lists(["pandas==2.2.0"], ["hypha-rpc==0.21.40"])
    assert caplog.records == []


def test_case_insensitive_name_match() -> None:
    """PEP 503-style name normalization is conservative; we lowercase to
    avoid duplicate-but-different-cased entries on the replica."""
    merged = _merge_pip_lists(
        ["Pydantic==2.12.0"],
        ["pydantic==2.10.0"],
    )
    assert merged == ["pydantic==2.10.0"]
