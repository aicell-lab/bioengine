"""Unit tests for the runtime_env pip-list merge in
:func:`bioengine._app.bootstrap._merge_pip_lists`.

The merge runs at bind time inside ``build_and_run_application`` and is
how the framework's required deps (``hypha-rpc``, ``pydantic``) make it
onto every user replica's venv alongside whatever the user declared via
``@bioengine.app(pip=…)``. Two invariants matter:

* A framework dep is added if it's not already present.
* A user-declared entry on the same package name is preserved verbatim,
  including any version pin or extra spec — the framework never
  silently overrides what the user set.
"""
from __future__ import annotations

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


def test_user_pin_wins_over_framework_pin() -> None:
    merged = _merge_pip_lists(
        ["hypha-rpc==0.20.0", "pandas==2.2.0"],
        ["hypha-rpc==0.21.40", "pydantic==2.12.0"],
    )
    assert merged == [
        "hypha-rpc==0.20.0",
        "pandas==2.2.0",
        "pydantic==2.12.0",
    ]


def test_user_unpinned_wins_over_framework_pin() -> None:
    merged = _merge_pip_lists(
        ["pydantic"],
        ["pydantic==2.12.0"],
    )
    assert merged == ["pydantic"]


def test_user_extras_preserved() -> None:
    merged = _merge_pip_lists(
        ["httpx[http2]==0.28.1"],
        ["httpx==0.28.1"],
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


def test_case_insensitive_name_match() -> None:
    """PEP 503-style name normalization is conservative; we lowercase to
    avoid duplicate-but-different-cased entries on the replica."""
    merged = _merge_pip_lists(
        ["Pydantic==2.12.0"],
        ["pydantic==2.10.0"],
    )
    assert merged == ["Pydantic==2.12.0"]
