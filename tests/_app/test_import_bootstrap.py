"""Unit tests for the ``sys.meta_path`` finder installed by
:func:`bioengine.__init__._install_replica_bootstrap_finder`.

The finder is the v0.11.4 fix that lets Ray Serve replicas import the
user's entry module even when cloudpickle.loads issues the import before
any ``import bioengine`` could fire a bootstrap side effect (the
cellpose-finetuning case where ``main:CellposeFinetune`` is the entry).

We exercise the finder by installing one instance against a freshly
populated tmp_path, patching ``setup_replica_environment`` to put a tiny
user package on ``sys.path``, and asserting that an import of a name
that didn't exist before the finder fired now resolves.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Tuple

import pytest


@pytest.fixture
def fake_user_package(tmp_path) -> Tuple[Path, str]:
    """Write a one-file user package onto disk and return its dir + name."""
    pkg_dir = tmp_path / "source"
    pkg_dir.mkdir()
    module_name = "fake_bioengine_user_entry"
    (pkg_dir / f"{module_name}.py").write_text(
        "def hello() -> str: return 'from-user-package'\n"
    )
    return pkg_dir, module_name


@pytest.fixture
def install_finder(monkeypatch, fake_user_package):
    """Install the meta_path finder + patch setup_replica_environment to
    sys.path.insert the fake user package on first fire.
    """
    pkg_dir, _ = fake_user_package

    fired = {"count": 0}

    def fake_setup() -> None:
        fired["count"] += 1
        sys.path.insert(0, str(pkg_dir))

    from bioengine._app import replica_init as _replica_init

    monkeypatch.setattr(
        _replica_init, "setup_replica_environment", fake_setup, raising=True
    )
    monkeypatch.setenv("BIOENGINE_APP_DIR", str(pkg_dir.parent))

    import bioengine

    bioengine._install_replica_bootstrap_finder()

    # Pull the finder we just installed back so the test can inspect it.
    finder = sys.meta_path[-1]
    yield finder, fired

    sys.meta_path.remove(finder)
    if str(pkg_dir) in sys.path:
        sys.path.remove(str(pkg_dir))
    sys.modules.pop("fake_bioengine_user_entry", None)


def test_finder_fires_setup_then_resolves_user_module(
    install_finder, fake_user_package
) -> None:
    finder, fired = install_finder
    _, module_name = fake_user_package

    # Before the finder fires, the module is not importable.
    sys.modules.pop(module_name, None)
    if str(fake_user_package[0]) in sys.path:
        sys.path.remove(str(fake_user_package[0]))

    mod = importlib.import_module(module_name)
    assert mod.hello() == "from-user-package"
    assert fired["count"] == 1


def test_finder_only_fires_once(install_finder, fake_user_package) -> None:
    finder, fired = install_finder
    _, module_name = fake_user_package

    importlib.import_module(module_name)
    sys.modules.pop(module_name, None)
    # A second import should NOT re-run setup_replica_environment.
    importlib.import_module(module_name)
    assert fired["count"] == 1


def test_finder_does_not_intercept_bioengine_imports(install_finder) -> None:
    finder, fired = install_finder
    # bioengine.utils is already on sys.path via the installed wheel; the
    # finder must skip names under "bioengine" so it can't recurse onto
    # its own ``from bioengine._app.replica_init import …`` call.
    spec = finder.find_spec("bioengine.utils", None)
    assert spec is None
    assert fired["count"] == 0


def test_finder_no_op_when_app_dir_unset(monkeypatch) -> None:
    monkeypatch.delenv("BIOENGINE_APP_DIR", raising=False)
    before = list(sys.meta_path)
    import bioengine

    bioengine._install_replica_bootstrap_finder()
    # No new finder is appended on worker processes.
    assert sys.meta_path == before
