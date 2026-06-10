"""Conftest for ``bioengine._app`` unit tests.

These tests exercise the decorator-and-mixin layer in isolation — no Ray
cluster, no Hypha, no data server. Override the heavy session-scoped
fixtures inherited from ``tests/conftest.py`` so the suite stays fast.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def validate_environment():
    """No-op override of the session-wide environment check."""
    yield


@pytest.fixture(scope="session", params=["single-machine"])
def worker_mode(request):
    """Skip multi-mode parametrisation — these tests are mode-agnostic."""
    return request.param
