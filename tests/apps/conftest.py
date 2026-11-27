"""Override root conftest fixtures for app-level tests.

App tests (tests/apps/) connect directly to the live Hypha service and do
not require the full worker environment (Ray, haikunator, etc.).
"""

import pytest

import bioengine.apps.manager as manager_mod


@pytest.fixture(scope="session", autouse=True)
def validate_environment():
    """No-op override: app tests don't need worker package validation."""
    pass


@pytest.fixture(autouse=True)
def fast_undeploy_confirm_window(monkeypatch):
    """The undeploy confirm window sleeps between polls; no arm should pay the real 3s x 5.

    Shared from here rather than from one test module, because the cost is paid by any arm
    that drives ``_undeploy_application`` at all — it was measured at 12.02s per arm, twice,
    in ``test_undeploy_delete_failure_keeps_tracking.py``, whose own arms are not about the
    window and had no such fixture.

    Only the INTERVAL is zeroed. The poll COUNT is left alone: the number of consecutive
    absences required is the contract under test, and shortening it would make these arms
    agree with a weaker fix.
    """
    if hasattr(manager_mod, "_POST_DELETE_CONFIRM_INTERVAL_SECONDS"):
        monkeypatch.setattr(
            manager_mod, "_POST_DELETE_CONFIRM_INTERVAL_SECONDS", 0, raising=False
        )


@pytest.fixture(scope="session")
def workspace_folder(tmp_path_factory):
    return tmp_path_factory.mktemp("app_tests")
