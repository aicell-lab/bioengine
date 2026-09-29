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

    ``bioengine._app.mixin._ensure_working_directory`` anchors a replica
    process at its app directory and the change has to outlive the call —
    user code resolves relative paths against it for the replica's whole
    life — so the production code cannot restore it. Constructing a
    decorated class here runs that in the pytest process, which the rest of
    the suite shares.
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
