"""A strayed application is MOVED into the collection, not deleted and recreated.

THE DEFECT. ``stage_artifact`` resolves the manifest ``id`` to ``<workspace>/<alias>``
across the whole workspace, while applications live in ``<workspace>/applications``. An
application that ended up outside that collection was handled by DELETING it and creating
it again in the right place. That destroyed its committed version history, and it also
skipped the version-increase check, because that check only runs in the
``if existing_artifact:`` branch which a recreated artifact never reaches.

WHY THESE ARMS MODEL THE SERVER RATHER THAN SPYING ON CALLS. Asserting "``set_parent`` was
called and ``delete`` was not" would pass against an implementation that called
``set_parent`` and then dropped the versions anyway, and it would pin the method name
rather than the property that matters. So ``_FakeArtifactManager`` gives the two operations
their real semantics -- ``delete`` removes the record and its versions, ``set_parent``
re-parents and keeps them -- and the arms assert on the surviving versions and payloads.
That is what makes the first arm RED on main: main's ``delete`` genuinely empties the
version list in this fake, exactly as it does on the server.

Measured on a live throwaway Hypha workspace before this was written: ``set_parent`` is a
true move -- two committed versions and their payloads were byte-identical afterwards --
and the parent is POSITIONAL, ``parent_id=`` raises TypeError. The fake mirrors both.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from bioengine.utils.artifact_utils import stage_artifact

WS = "bioimage-io"
COLLECTION = f"{WS}/applications"
ALIAS = "strayed-app"
AID = f"{WS}/{ALIAS}"

#: What a committed lineage looks like. The payload bytes are carried so the arms can
#: assert content survival, not merely that a version string is still listed.
LINEAGE = {"1.0.0": b"payload-v1-61-bytes", "2.0.0": b"payload-v2-61-bytes"}


def _manifest(version: str = "3.0.0") -> Dict[str, Any]:
    return {
        "id": ALIAS,
        "name": "Strayed App",
        "type": "ray-serve",
        "format_version": "0.6.0",
        "description": "an application that ended up outside the collection",
        "version": version,
        "documentation": "docs",
        "entry": "entry:App",
    }


class _Record:
    """One artifact, with the fields ``stage_artifact`` reads off a proxy."""

    def __init__(
        self, *, atype: str, parent_id: Optional[str], versions: Dict[str, bytes]
    ):
        self.id = AID
        self.alias = ALIAS
        self.type = atype
        self.parent_id = parent_id
        self.config: Dict[str, Any] = {}
        self.manifest: Dict[str, Any] = {}
        self.payloads: Dict[str, bytes] = dict(versions)

    @property
    def versions(self) -> List[Dict[str, Any]]:
        return [{"version": v, "created_at": i} for i, v in enumerate(self.payloads)]


class _FakeArtifactManager:
    """An artifact manager with the SEMANTICS of the two operations under test.

    Deliberately not a mock: ``delete`` destroys the lineage and ``set_parent`` preserves
    it, so an arm that asserts on surviving versions is testing behaviour rather than a
    call list.
    """

    def __init__(self, record: Optional[_Record]):
        self.store: Dict[str, _Record] = {record.id: record} if record else {}
        self.calls: List[str] = []

    async def read(self, artifact_id, **_kw):
        self.calls.append("read")
        aid = (
            artifact_id
            if isinstance(artifact_id, str)
            else artifact_id.get("artifact_id")
        )
        if aid not in self.store:
            raise KeyError(f"Artifact with ID '{aid}' does not exist")
        return self.store[aid]

    async def delete(self, artifact_id=None, **_kw):
        # The real thing is final: the record and every committed version go with it.
        self.calls.append("delete")
        self.store.pop(artifact_id, None)

    async def set_parent(self, *args, **kwargs):
        # Positional only, as measured against the live server.
        self.calls.append("set_parent")
        if kwargs:
            raise TypeError(
                f"set_parent() takes positional arguments, got {sorted(kwargs)}"
            )
        artifact_id, parent_id = args[0], args[1]
        rec = self.store[artifact_id]
        rec.parent_id = parent_id  # versions and payloads untouched: a true move

    async def edit(self, **kwargs):
        self.calls.append("edit")
        rec = self.store[kwargs.get("artifact_id", AID)]
        if kwargs.get("manifest") is not None:
            rec.manifest = kwargs["manifest"]
        if kwargs.get("config") is not None:
            rec.config = kwargs["config"]
        return rec

    async def create(self, **kwargs):
        self.calls.append("create")
        rec = _Record(
            atype="application", parent_id=kwargs.get("parent_id"), versions={}
        )
        self.store[rec.id] = rec
        return rec


def _logger() -> logging.Logger:
    return logging.getLogger("test.strayed_application")


@pytest.mark.asyncio
async def test_a_strayed_application_keeps_its_versions_and_payloads() -> None:
    """RED on main: main deletes the record, so the lineage is gone.

    The assertion is on the surviving versions and their bytes, not on which method ran.
    """
    rec = _Record(atype="application", parent_id=None, versions=LINEAGE)
    am = _FakeArtifactManager(rec)

    await stage_artifact(
        artifact_manager=am, workspace=WS, manifest=_manifest(), logger=_logger()
    )

    assert AID in am.store, (
        "the strayed application was removed from the store, so its committed versions "
        "went with it -- this is the delete-and-recreate path"
    )
    survivor = am.store[AID]
    assert sorted(survivor.payloads) == [
        "1.0.0",
        "2.0.0",
    ], f"committed versions did not survive the migration: {sorted(survivor.payloads)}"
    assert (
        survivor.payloads == LINEAGE
    ), "a version survived by name but its bytes changed"
    assert (
        survivor.parent_id == COLLECTION
    ), f"the application was not moved into the collection: parent_id={survivor.parent_id!r}"


@pytest.mark.asyncio
async def test_the_migration_never_deletes() -> None:
    """The narrower statement, kept separate so a failure says which property broke.

    The first arm can fail because the lineage was lost; this one fails only if a delete
    was issued at all. Two different repairs, so two arms.
    """
    am = _FakeArtifactManager(
        _Record(atype="application", parent_id=None, versions=LINEAGE)
    )
    await stage_artifact(
        artifact_manager=am, workspace=WS, manifest=_manifest(), logger=_logger()
    )
    assert (
        "delete" not in am.calls
    ), f"a delete was issued during the migration: {am.calls}"
    assert "set_parent" in am.calls, f"no set_parent in the call sequence: {am.calls}"


@pytest.mark.asyncio
async def test_a_correctly_placed_application_is_untouched_CONTROL() -> None:
    """CONTROL -- green before and after. An app already in the collection is not moved.

    Without this, a 'fix' that moved every artifact unconditionally would satisfy the
    arms above.
    """
    rec = _Record(atype="application", parent_id=COLLECTION, versions=LINEAGE)
    am = _FakeArtifactManager(rec)

    await stage_artifact(
        artifact_manager=am, workspace=WS, manifest=_manifest(), logger=_logger()
    )

    assert (
        "set_parent" not in am.calls
    ), f"an application already in the collection was re-parented anyway: {am.calls}"
    assert (
        "delete" not in am.calls
    ), f"a correctly placed application was deleted: {am.calls}"
    assert am.store[AID].payloads == LINEAGE
    assert am.store[AID].parent_id == COLLECTION


@pytest.mark.asyncio
async def test_a_non_application_at_the_alias_is_still_REFUSED() -> None:
    """The the refusal rule guard must survive this change -- and must refuse BEFORE any move.

    The point of the earlier fix was that someone else's dataset is not ours to touch.
    Replacing the delete with a move does not make it ours, so the refusal still has to
    come first: assert that NEITHER set_parent nor delete ran.
    """
    rec = _Record(
        atype="dataset", parent_id=None, versions={"1.0.0": b"someone-elses-data"}
    )
    am = _FakeArtifactManager(rec)

    with pytest.raises(ValueError) as err:
        await stage_artifact(
            artifact_manager=am, workspace=WS, manifest=_manifest(), logger=_logger()
        )

    msg = str(err.value)
    assert "not a BioEngine application" in msg
    assert "'dataset'" in msg, f"the refusal does not name the type it found: {msg}"
    assert (
        "set_parent" not in am.calls and "delete" not in am.calls
    ), f"the dataset was touched before being refused: {am.calls}"
    assert am.store[AID].payloads == {"1.0.0": b"someone-elses-data"}


def test_the_refusal_message_does_not_print_a_bare_None() -> None:
    """The cosmetic half, and it is the common case rather than an edge one.

    A strayed artifact sits at root, where ``parent_id`` is None, so "in collection None"
    is what a reader of this error would usually have seen.
    """
    from bioengine.utils.artifact_utils import _describe_collection

    assert _describe_collection(None) == "no collection (root level)"
    assert _describe_collection(COLLECTION) == f"'{COLLECTION}'"
    assert "None" not in _describe_collection(None)


@pytest.mark.asyncio
async def test_a_refused_version_does_NOT_move_the_artifact_first() -> None:
    """The ordering arm. RED before the repair, and the repair is the point of it.

    A strayed application at 1.0.0 and 2.0.0, uploaded with manifest 1.5.0, must be
    REFUSED for the non-increasing version -- and must still be where it was. Before the
    repair the move happened first, so the caller got a ValueError while the artifact had
    already silently changed collection: exactly the harm this function refuses to do to a
    dataset, done to an application instead.

    The `set_parent` assertion is what makes this arm bite. Asserting only the ValueError
    passes on the broken ordering, because the broken ordering also raises.
    """
    rec = _Record(atype="application", parent_id=None, versions=LINEAGE)
    am = _FakeArtifactManager(rec)

    with pytest.raises(ValueError) as err:
        await stage_artifact(
            artifact_manager=am,
            workspace=WS,
            manifest=_manifest(version="1.5.0"),
            logger=_logger(),
        )

    assert "greater" in str(err.value).lower() or "version" in str(err.value).lower()
    assert "set_parent" not in am.calls, (
        f"the artifact was MOVED before the version check refused the upload, so a "
        f"refused call still relocated it: {am.calls}"
    )
    assert am.store[AID].parent_id is None, (
        f"the refused upload left the artifact in a different collection: "
        f"{am.store[AID].parent_id!r}"
    )
    assert am.store[AID].payloads == LINEAGE, "the refused upload altered the lineage"


@pytest.mark.asyncio
async def test_a_migrated_application_IS_version_gated() -> None:
    """Finding 2's arm: the gating claim was untested, so a mutation disabling it passed.

    A strayed application at 2.0.0 uploaded with manifest 2.0.0 must be refused. Without
    this, turning the gate off for exactly the migrated case left all arms green -- every
    other arm feeds 3.0.0, which is above the lineage and so never exercises the gate.
    """
    rec = _Record(
        atype="application", parent_id=None, versions={"2.0.0": b"payload-v2"}
    )
    am = _FakeArtifactManager(rec)

    with pytest.raises(ValueError):
        await stage_artifact(
            artifact_manager=am,
            workspace=WS,
            manifest=_manifest(version="2.0.0"),
            logger=_logger(),
        )
    assert "set_parent" not in am.calls, f"refused, but moved anyway: {am.calls}"
