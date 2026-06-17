"""Unit tests for the cross-workspace recovery filter.

When several bioengine workers from different Hypha workspaces share a
Ray cluster, ``serve.status()`` returns every workspace's applications.
``AppsManager.recover_deployed_applications`` only adopts the apps that
belong to *this* worker's workspace; the filter is implemented as
:func:`bioengine.apps.manager._belongs_to_worker_workspace`.
"""

import logging

import pytest

from bioengine.apps.manager import _belongs_to_worker_workspace


@pytest.fixture
def caplog_at_debug(caplog):
    caplog.set_level(logging.DEBUG, logger="test")
    return caplog


def _make_logger() -> logging.Logger:
    return logging.getLogger("test")


def test_match_workspace_returns_true(caplog_at_debug) -> None:
    assert _belongs_to_worker_workspace(
        application_id="model-runner",
        app_data={"worker_workspace": "bioimage-io"},
        current_workspace="bioimage-io",
        logger=_make_logger(),
    )
    assert caplog_at_debug.records == []  # silent success


def test_mismatched_workspace_returns_false_and_logs_debug(
    caplog_at_debug,
) -> None:
    assert not _belongs_to_worker_workspace(
        application_id="model-runner",
        app_data={"worker_workspace": "bioimage-io"},
        current_workspace="ws-user-github|49943582",
        logger=_make_logger(),
    )
    debug_records = [r for r in caplog_at_debug.records if r.levelno == logging.DEBUG]
    assert any(
        "belongs to workspace" in r.message and "model-runner" in r.message
        for r in debug_records
    ), "expected a DEBUG line naming the application and the foreign workspace"
    # Mismatch is expected when multiple workers share a cluster — no
    # WARNING or higher should fire.
    assert all(
        r.levelno < logging.WARNING for r in caplog_at_debug.records
    ), "workspace mismatch must stay at DEBUG, not warn"


def test_missing_marker_returns_false_and_warns(caplog_at_debug) -> None:
    assert not _belongs_to_worker_workspace(
        application_id="legacy-app",
        app_data={
            # no worker_workspace key
            "artifact_id": "ws/legacy-app",
            "version": "1.0.0",
        },
        current_workspace="bioimage-io",
        logger=_make_logger(),
    )
    warning_records = [
        r for r in caplog_at_debug.records if r.levelno == logging.WARNING
    ]
    assert any(
        "legacy-app" in r.message and "worker_workspace" in r.message
        for r in warning_records
    ), "expected a WARNING that names the app and the missing marker"


def test_empty_string_marker_treated_as_present_and_mismatched(
    caplog_at_debug,
) -> None:
    # Defensive: an empty-string marker is technically present but doesn't
    # match any real workspace, so it must NOT be adopted as ours.
    assert not _belongs_to_worker_workspace(
        application_id="malformed",
        app_data={"worker_workspace": ""},
        current_workspace="bioimage-io",
        logger=_make_logger(),
    )
