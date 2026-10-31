"""Pin that deleting a version refuses when later versions would be corrupted.

Hypha maps an artifact version to its stored files by POSITION in the version
list, so deleting version i re-points every later version's payload down one
slot: version i+1 starts serving version i's bytes under its own label, with a
200 and a version list that still reads correctly (myco #0019, re-measured on
live Hypha 2026-10-24).

``delete_app_version`` used to call ``artifact_manager.delete`` unconditionally
once its three guards passed, and none of them is about the shift: the "tag must
contain 'dev'" guard constrains only the version being *deleted*, while the
version that gets corrupted is a released one *after* it. That is what poisoned
model-runner 2.11.1 on 2026-10-02 — deleting a leftover ``2.11.0-dev1`` left
``2.11.1`` serving 2.11.0's ``runtime.py`` under a 2.11.1 label.

The guard is positional on purpose. ``_enforce_version_increases`` ranks by
PEP 440 and ``latest_committed_version`` ranks by ``created_at``; both can
disagree with list order once a version has been deleted and a lower one
re-added, and it is list order that decides which payloads move.

Live coverage against a real artifact is in ``tests/test_artifact_version.py``.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps.manager import AppsManager

WORKSPACE = "bioimage-io"
ARTIFACT_ID = f"{WORKSPACE}/model-runner"
ADMIN = "admin@example.com"
CONTEXT = {"user": {"id": "admin", "email": ADMIN}}


def _make_manager(*, versions) -> AppsManager:
    """An AppsManager wired with only what delete_app_version touches."""
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.delete_app_version")
    manager.admin_users = [ADMIN]
    manager._deployed_applications = {}

    server = MagicMock()
    server.config.workspace = WORKSPACE
    manager.server = server

    artifact_manager = MagicMock()
    artifact_manager.read = AsyncMock(
        return_value={"versions": [{"version": v} for v in versions]}
    )
    artifact_manager.delete = AsyncMock(return_value=None)
    manager.artifact_manager = artifact_manager
    return manager


@pytest.mark.asyncio
async def test_deleting_a_version_with_a_release_after_it_is_refused() -> None:
    # The production incident verbatim: drop 2.11.0-dev1 and 2.11.1 is corrupted.
    manager = _make_manager(versions=["2.11.0", "2.11.0-dev1", "2.11.1"])

    with pytest.raises(ValueError) as excinfo:
        await manager.delete_app_version(
            artifact_id=ARTIFACT_ID, version="2.11.0-dev1", context=CONTEXT
        )

    message = str(excinfo.value)
    assert "2.11.1" in message, "The versions at risk have to be named."
    assert "newest-first" in message, (
        "The error has to name the way out, or it just restates the symptom."
    )
    manager.artifact_manager.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_deleting_the_newest_version_still_works() -> None:
    # Nothing follows it, so nothing shifts — the dev-iteration workflow's case.
    manager = _make_manager(versions=["1.0.0", "1.1.0-dev1"])

    await manager.delete_app_version(
        artifact_id=ARTIFACT_ID, version="1.1.0-dev1", context=CONTEXT
    )

    manager.artifact_manager.delete.assert_awaited_once_with(
        ARTIFACT_ID, version="1.1.0-dev1"
    )


@pytest.mark.asyncio
async def test_the_guard_reads_list_position_not_version_ordering() -> None:
    # Re-adding a lower version leaves the highest PEP 440 version sitting
    # mid-list. "Is any version greater than the target?" answers no here and
    # would allow the delete; 1.2.0 follows it in the list and is what shifts.
    manager = _make_manager(versions=["1.0.0", "1.5.0-dev1", "1.2.0"])

    with pytest.raises(ValueError, match="1.2.0"):
        await manager.delete_app_version(
            artifact_id=ARTIFACT_ID, version="1.5.0-dev1", context=CONTEXT
        )

    manager.artifact_manager.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unreadable_artifact_blocks_the_delete() -> None:
    # Failing open here would delete exactly the version we could not clear.
    manager = _make_manager(versions=["1.0.0", "1.1.0-dev1"])
    manager.artifact_manager.read = AsyncMock(side_effect=RuntimeError("artifact gone"))

    with pytest.raises(RuntimeError, match="artifact gone"):
        await manager.delete_app_version(
            artifact_id=ARTIFACT_ID, version="1.1.0-dev1", context=CONTEXT
        )

    manager.artifact_manager.delete.assert_not_awaited()
