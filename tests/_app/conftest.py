"""Conftest for ``bioengine._app`` unit tests.

These tests exercise the decorator-and-mixin layer in isolation — no Ray
cluster, no Hypha, no data server. Override the heavy session-scoped
fixtures inherited from ``tests/conftest.py`` so the suite stays fast.
"""

import os
from typing import Generator

import pytest


@pytest.fixture(autouse=True)
def restore_working_directory() -> Generator[None, None, None]:
    """Confine the replica-setup ``os.chdir`` to the test that triggered it.

    ``bioengine._app.mixin._ensure_working_directory`` moves the process to
    the app directory, and constructing a decorated class here runs that in
    the pytest process the rest of the suite shares. Runtime behaviour is
    deliberately left unchanged; the containment is on this side.
    """
    entry = os.getcwd()
    yield
    os.chdir(entry)


@pytest.fixture(scope="session", autouse=True)
def validate_environment():
    """No-op override of the session-wide environment check."""
    yield


@pytest.fixture(scope="session", params=["single-machine"])
def worker_mode(request):
    """Skip multi-mode parametrisation — these tests are mode-agnostic."""
    return request.param
