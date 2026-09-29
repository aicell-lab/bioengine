"""The zarr floor in ``http_zarr_store`` has to order versions, not strings.

A string comparison accepts ``3.0.8`` and ``3.1.0`` but rejects ``3.0.10`` and
``3.10.0``, and the rejection surfaces as an ``ImportError`` at collection time
— so the two dataset test modules silently drop out of the suite instead of
failing. Asserting only that the installed zarr is accepted would not catch
that: the installed version is usually one a string comparison also gets right.
"""

import importlib.util

import pytest
import zarr

# Imported by full path on purpose: ``from bioengine.datasets import http_zarr_store``
# goes through the package's __getattr__ delegation and raises
# MissingDataServerError instead of importing the submodule — which would mask
# the gate's own ImportError in exactly the case this file exists to check.
import bioengine.datasets.http_zarr_store as http_zarr_store

SOURCE = http_zarr_store.__file__


def _import_under_zarr_version(version: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-execute the module as a throwaway, pretending zarr is ``version``.

    A fresh module object keeps the already-imported
    ``bioengine.datasets.http_zarr_store`` (and the ``HttpZarrStore`` identity
    other tests hold) untouched.
    """
    monkeypatch.setattr(zarr, "__version__", version)
    spec = importlib.util.spec_from_file_location(
        "_http_zarr_store_version_gate_probe", SOURCE
    )
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


@pytest.mark.parametrize("version", ["3.0.8", "3.0.9", "3.0.10", "3.1.0", "3.10.0"])
def test_gate_accepts_versions_at_or_above_the_floor(
    version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _import_under_zarr_version(version, monkeypatch)


@pytest.mark.parametrize("version", ["3.0.0", "3.0.7", "2.18.3", "2.99.99"])
def test_gate_rejects_versions_below_the_floor(
    version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ImportError, match=r"zarr>=3\.0\.8 is required but found"):
        _import_under_zarr_version(version, monkeypatch)
