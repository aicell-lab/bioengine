"""Selecting a package by name must not depend on how bioengine happens to
declare it, and asking for a name that does not exist must not be silent.

``bioengine.utils.requirements.get_pip_requirements(select=[...])`` builds the
pip list for the Ray Serve proxy, for the build task, and for every user
replica. It used to compare the requested name against the raw metadata
string, so ``httpx``, declared as ``httpx[http2]>=0.28.1``, never matched and
was dropped from the proxy list — for years, with nothing logged. A replica
venv that is missing a package the worker believes it put there is the same
failure shape either way; the difference is whether anybody finds out.
"""

from __future__ import annotations

import importlib.metadata as md

import pytest

from bioengine.utils.requirements import (
    get_pip_requirements,
    requirement_name,
    update_requirements,
)

INSTALLED_HTTPX = md.version("httpx")
INSTALLED_PYDANTIC = md.version("pydantic")


@pytest.mark.parametrize(
    "requirement,expected",
    [
        ("httpx[http2]>=0.28.1", "httpx"),
        ("httpx[http2]==0.28.1", "httpx"),
        ("httpx", "httpx"),
        ("hypha-rpc>=0.21.40", "hypha-rpc"),
        ("pydantic~=2.12.0", "pydantic"),
        ("ray[client,serve]>=2.53.0,<3.0.0", "ray"),
        ("  Pillow >= 10.0 ", "pillow"),
        ("typing_extensions", "typing-extensions"),
    ],
)
def test_requirement_name_drops_specifier_and_extras(
    requirement: str, expected: str
) -> None:
    assert requirement_name(requirement) == expected


def test_a_plain_selector_matches_a_metadata_name_carrying_extras() -> None:
    """The drop this file exists for: ``httpx`` is declared as
    ``httpx[http2]``, and the proxy asks for it as ``httpx``."""
    assert get_pip_requirements(select=["httpx"], extras=[]) == [
        f"httpx[http2]=={INSTALLED_HTTPX}"
    ]


def test_the_proxy_list_carries_httpx_with_its_http2_extra() -> None:
    """Asserted on the proxy list specifically — the list the drop was
    observed in. The extra has to survive: without it pip resolves no ``h2``
    into the venv and the proxy falls back to whatever the Ray node image
    happens to ship."""
    proxy_pip = get_pip_requirements(
        select=["aiortc", "httpx", "hypha-rpc", "pydantic"], extras=["worker"]
    )
    assert f"httpx[http2]=={INSTALLED_HTTPX}" in proxy_pip


def test_a_selector_that_matches_nothing_raises() -> None:
    """Asserted on an unmatched name. A test on a matched one passes with or
    without the diagnostic and proves nothing about the drop."""
    with pytest.raises(ValueError) as excinfo:
        get_pip_requirements(select=["httpx", "not-a-bioengine-dependency"], extras=[])
    assert "not-a-bioengine-dependency" in str(excinfo.value)


def test_a_package_only_declared_in_an_unrequested_extra_is_unmatched() -> None:
    """The commonest way a selector misses: ``pydantic`` exists, but only in
    the ``worker`` extra, so a caller passing ``extras=[]`` gets nothing."""
    with pytest.raises(ValueError) as excinfo:
        get_pip_requirements(select=["pydantic"], extras=[])
    assert "pydantic" in str(excinfo.value)

    assert get_pip_requirements(select=["pydantic"], extras=["worker"]) == [
        f"pydantic=={INSTALLED_PYDANTIC}"
    ]


def test_selecting_everything_needs_no_selector_and_still_succeeds() -> None:
    """``select=None`` is a different branch and must not acquire the check."""
    assert get_pip_requirements()


def test_update_requirements_does_not_duplicate_a_name_declared_with_extras() -> None:
    """Sibling caller of the same comparison: without the normalization
    ``httpx[http2]==…`` and a caller's ``httpx==…`` look like two packages
    and both land in one pip list."""
    assert update_requirements(["httpx==0.27.0"], select=["httpx"], extras=[]) == [
        "httpx==0.27.0"
    ]
