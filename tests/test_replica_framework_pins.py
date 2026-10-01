"""What the worker dictates for a *user replica's* venv, and what it leaves alone.

``bioengine.apps.builder`` builds three pip lists. Two of them (the proxy's and
the build task's) are covered elsewhere; this file covers the third, the list
merged into every ``@bioengine.app`` deployment's ``runtime_env.pip`` at bind
time. Membership is decided by one measured property of the Ray node image,
because Ray builds runtime_env venvs with ``--system-site-packages``:

* ``pydantic`` IS in ``rayproject/ray:2.55.1-py311``, so a replica inherits it.
  A pin here would install a venv-local copy shadowing the inherited one.
* ``httpx`` is NOT (nor is ``httpcore``), while ``hypha-rpc`` declares a *bare*
  ``httpx``. So the hypha-rpc pin alone already pip-installs httpx into every
  replica venv at whatever version PyPI serves that minute — #0020's drift
  mechanism, one package over. Naming httpx here changes which version lands,
  not whether it lands.

The tests below hold both halves: the pin exists, it is read off the installed
distribution rather than written down, and pydantic is still absent.

These tests read the real installed ``bioengine`` distribution on purpose — that
table is their subject, so faking it would reduce them to comparing a fixture
against itself. With no installed distribution (``pip uninstall bioengine`` over
a checkout, the usual way to stop an image's own copy shadowing the source) they
fail with ``PackageNotFoundError``: a broken measurement setup, not a regression.
"""

from __future__ import annotations

import importlib.metadata as md

import pytest

from bioengine.apps.builder import _USER_REPLICA_FRAMEWORK_PACKAGES
from bioengine.utils.requirements import get_pip_requirements


def _replica_pip() -> list[str]:
    """Exactly the call ``AppBuilder.build`` makes for the replica list."""
    return get_pip_requirements(select=_USER_REPLICA_FRAMEWORK_PACKAGES, extras=[])


def test_the_replica_list_asks_for_httpx() -> None:
    """Dropping ``httpx`` from the selector is the state this file was written
    against: the replica still gets httpx, but via hypha-rpc's bare
    requirement, so nothing records which version it got."""
    assert "httpx" in _USER_REPLICA_FRAMEWORK_PACKAGES


def test_the_replica_list_carries_an_exact_httpx_pin() -> None:
    entries = _replica_pip()
    assert f"httpx[http2]=={md.version('httpx')}" in entries, entries


def test_no_replica_entry_is_left_unpinned() -> None:
    """A floor anywhere in this list is re-resolved against PyPI on every
    runtime_env build, which is the whole defect — so assert over the list
    rather than over the one package that prompted it."""
    entries = _replica_pip()
    assert entries, "list must not be empty or the assertion is vacuous"
    for requirement in entries:
        assert "==" in requirement, requirement
        for floating in (">=", "<=", "~=", ">", "<", ","):
            assert floating not in requirement, requirement


@pytest.mark.parametrize("package", ["httpx", "hypha-rpc"])
def test_each_pin_is_read_from_the_installed_distribution(
    monkeypatch, package: str
) -> None:
    """#0020's mutation, applied to this list. A hardcoded ``0.28.1`` passes
    every assertion above for as long as it happens to match what the image
    resolved. Move the installed version out from under the worker and the
    emitted pin has to move with it."""
    real_version = md.version

    def fake_version(name: str) -> str:
        return "9.9.9-probe" if name == package else real_version(name)

    monkeypatch.setattr(
        "bioengine.utils.requirements.md.version", fake_version, raising=True
    )
    assert any(
        entry.endswith("==9.9.9-probe") and entry.startswith(package)
        for entry in _replica_pip()
    ), _replica_pip()


def test_pydantic_stays_out_of_the_replica_list() -> None:
    """Deliberate, not an omission. The Ray node image ships pydantic and Ray
    builds the venv with ``--system-site-packages``, so the replica already has
    one; pinning here shadows the inherited copy rather than complementing it.
    If you are adding pydantic, read
    ``bioengine.apps.builder._USER_REPLICA_FRAMEWORK_PACKAGES`` first — the
    open residual there is an image-band check, not a pip entry."""
    assert "pydantic" not in _USER_REPLICA_FRAMEWORK_PACKAGES
    assert not any(entry.startswith("pydantic") for entry in _replica_pip())
