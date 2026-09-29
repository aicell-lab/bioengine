"""The version an app's ``runtime_env`` installs must be the version the
worker has loaded, not the lower bound written in a requirements file.

``bioengine.utils.requirements`` builds the pip list that Ray installs into
every app's per-deployment virtualenv. It used to derive the pin from the
*specifier* — rewriting ``hypha-rpc>=0.21.40`` to ``hypha-rpc==0.21.40`` —
which is only the same thing as long as the image's own resolution of that
floor never moves. It did move: a KTH Ray node was found carrying four
hypha-rpc installs at two versions, none of them chosen by anybody, and no
commit anywhere recorded the change.

These tests hold the fix to the property that matters: the emitted pin
tracks the *installed distribution*, so a version bump inside the worker
image propagates to apps without a repo edit, and reading the repo can no
longer disagree with what is loaded.
"""

from __future__ import annotations

import asyncio
import importlib.metadata as md
from types import SimpleNamespace

import pytest

from bioengine.utils.requirements import get_pip_requirements, normalize_requirement
from bioengine.worker.worker import BioEngineWorker

# hypha-rpc is a hard bioengine dependency, so it is importable wherever the
# suite runs. The tests that assert on what ``get_pip_requirements`` emits do
# not use it — they control their own environment via ``pinned_environment``.
INSTALLED_HYPHA_RPC = md.version("hypha-rpc")


def test_floor_resolves_to_the_installed_version_not_the_floor() -> None:
    """The exact rewrite that produced the drift."""
    assert (
        normalize_requirement("hypha-rpc>=0.1.0") == f"hypha-rpc=={INSTALLED_HYPHA_RPC}"
    )


@pytest.mark.parametrize(
    "requirement",
    [
        "hypha-rpc",
        "hypha-rpc>=0.1.0",
        "hypha-rpc<=99.0.0",
        "hypha-rpc~=0.1.0",
        "hypha-rpc>0.1",
        "hypha-rpc==0.1.0",
    ],
)
def test_every_operator_lands_on_the_same_installed_version(requirement: str) -> None:
    """Which comparison operator a dependency happens to be declared with
    decided whether Ray built the app a private venv at all. The resolved
    pin must not depend on it."""
    assert normalize_requirement(requirement) == f"hypha-rpc=={INSTALLED_HYPHA_RPC}"


def test_extras_are_kept_while_the_version_is_resolved() -> None:
    """``httpx[http2]`` is the requirement bioengine declares; dropping the
    extra would silently uninstall HTTP/2 support on every replica."""
    resolved = normalize_requirement("httpx[http2]>=0.1.0")
    assert resolved.startswith("httpx[http2]==")
    assert resolved.endswith(md.version("httpx"))


def test_unknown_distribution_falls_back_to_the_specifier() -> None:
    """Only bioengine's own ``Requires-Dist`` entries reach this function —
    an app's ``@bioengine.app(pip=…)`` list goes straight into
    ``runtime_env["pip"]`` without passing through it. So on the worker
    path every input is installed and this branch does not fire.

    It fires when an extra is asked for that the environment does not have:
    ``get_pip_requirements(extras=["cli"])`` in the worker image collapses
    ``rich``, ``tifffile`` and ``Pillow`` to their floors, while ``click``
    (installed) resolves past its. Without the fallback a missing extra
    would raise ``PackageNotFoundError`` out of a requirement-rewriting
    helper instead of returning a usable pin."""
    assert (
        normalize_requirement("definitely-not-installed-xyz>=1.2.3")
        == "definitely-not-installed-xyz==1.2.3"
    )


def test_empty_requirement_passes_through() -> None:
    assert normalize_requirement("") == ""


# bioengine's own dependency table, shaped exactly as ``Requires-Dist`` spells
# it, but owned by the test. The ``ray[client,serve]`` entry is here because
# ``get_pip_requirements`` drops it by name, and dropping it is what keeps the
# only comma-bearing specifier out of a runtime_env.
_REQUIRES_DIST = [
    "httpx[http2]>=0.28.1",
    "hypha-rpc>=0.21.40",
    "numpy==1.26.4",
    "packaging>=21.0",
    "aiortc==1.14.0; extra == 'worker'",
    "pydantic~=2.12.0; extra == 'worker'",
    "ray[client,serve]>=2.53.0,<3.0.0; extra == 'worker'",
    "zarr>=3.0.8; extra == 'datasets'",
]

# Deliberately unlike any version an image resolves, and deliberately not
# covering every name above: ``aiortc`` and ``pydantic`` are absent so the
# specifier-fallback branch is exercised alongside the installed-distribution
# one.
_INSTALLED = {
    "httpx": "9.9.1-probe",
    "hypha-rpc": "9.9.2-probe",
    "numpy": "9.9.3-probe",
}


@pytest.fixture
def pinned_environment(monkeypatch) -> None:
    """Hand ``get_pip_requirements`` a dependency table and an installed-version
    table this file owns.

    Both halves are needed. ``md.version`` is where the pin is read from, and
    faking only that leaves the test asserting against whichever ``bioengine``
    distribution the runner happens to have — with none at all (``pip uninstall
    bioengine`` over a checkout, the usual way to stop an image's own copy
    shadowing the source) ``md.metadata`` raises ``PackageNotFoundError`` and
    the test reports on the environment instead of on the code.
    """

    class _Metadata:
        def get_all(self, key, failobj=None):
            return _REQUIRES_DIST if key == "Requires-Dist" else failobj

    def fake_version(name: str) -> str:
        try:
            return _INSTALLED[name]
        except KeyError:
            raise md.PackageNotFoundError(name) from None

    monkeypatch.setattr(md, "metadata", lambda name: _Metadata())
    monkeypatch.setattr(md, "version", fake_version)


def test_injected_baseline_pins_the_loaded_hypha_rpc(pinned_environment) -> None:
    """End of the path the worker actually walks: what
    ``bioengine.apps.builder`` injects into a replica's runtime_env.

    Reading the versions off the fixture rather than off the image is also
    what gives the assertion content. In a real worker image the declared
    ``hypha-rpc>=0.21.40`` floor and the resolved install are the same string,
    so the old form held whether the pin came from the specifier or from the
    distribution — the exact ambiguity #204 existed to remove. ``numpy`` is
    declared with an ``==`` here and still has to move to the installed
    version."""
    baseline = get_pip_requirements(select=["hypha-rpc", "numpy"], extras=[])
    assert baseline == ["hypha-rpc==9.9.2-probe", "numpy==9.9.3-probe"]


def test_no_injected_baseline_entry_is_left_unpinned(pinned_environment) -> None:
    """A floor anywhere in this list is resolved by Ray against PyPI on
    every runtime_env build — that is the drift vector, so assert against
    the whole list rather than the one package the incident named."""
    baseline = get_pip_requirements(
        select=["aiortc", "httpx", "hypha-rpc", "pydantic"], extras=["worker"]
    )
    assert baseline, "baseline must not be empty or the assertion is vacuous"
    # Both resolution branches are represented, so the loop below is not just
    # walking four entries that took the same path.
    assert "httpx[http2]==9.9.1-probe" in baseline, baseline
    assert "aiortc==1.14.0" in baseline, baseline
    for requirement in baseline:
        assert "==" in requirement, requirement
        for floating in (">=", "<=", "~=", ">", "<", ","):
            assert floating not in requirement, requirement


def _bare_worker() -> BioEngineWorker:
    """Only the state ``get_status`` reads, with no Ray or Hypha behind it."""
    worker = BioEngineWorker.__new__(BioEngineWorker)
    worker.start_time = None  # also silences __del__ on a half-built instance
    worker.workspace = "ws"
    worker.client_id = "client"
    worker.admin_users = []
    worker.geo_location = None
    worker.is_ready = asyncio.Event()
    worker._monitor_consecutive_errors = 0
    worker._monitor_degraded_threshold = 5
    worker.ray_cluster = SimpleNamespace(
        mode="single-machine", status={}, slurm_workers=None
    )
    return worker


async def test_worker_status_reports_the_loaded_hypha_rpc() -> None:
    """With the repo no longer able to answer "what is running", the worker
    has to. ``get_status`` is what the dashboard and every operator already
    call."""
    status = await _bare_worker().get_status(context={})
    assert status["hypha_rpc_version"] == INSTALLED_HYPHA_RPC


async def test_worker_status_hypha_rpc_is_read_not_hardcoded(monkeypatch) -> None:
    """A literal would satisfy the test above for as long as it happened to
    match — the whole failure this fixes. Move the installed version out from
    under the worker and the report has to move with it."""
    monkeypatch.setattr(md, "version", lambda name: "9.9.9-probe")
    status = await _bare_worker().get_status(context={})
    assert status["hypha_rpc_version"] == "9.9.9-probe"
