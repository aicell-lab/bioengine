"""A strayed application must be MOVED into the applications collection, not rebuilt.

``stage_artifact`` resolves the manifest's ``id`` across the whole workspace, so it can find
an application artifact that sits somewhere other than ``<workspace>/applications``. Before
this change, that case was handled by deleting the artifact and creating a fresh one at the same
alias, which destroyed its committed version history and its file content while leaving
something that still resolved at that id -- so the loss was invisible to anyone who checked by
reading the id back.

``set_parent`` relocates an artifact with its versions and payloads intact (measured against
the deployed artifact manager: two committed versions and both payloads survived a move
byte-identically). So the migration is a move.

The arms that matter go through ``stage_artifact`` rather than testing a helper in isolation:
on the unfixed code the call site is what deletes, and a helper-only arm cannot see it.
"""

import pytest

from bioengine.utils.artifact_utils import (
    _APPLICATION_ARTIFACT_TYPE,
    _enforce_replaceable_artifact,
    stage_artifact,
)

WORKSPACE = "ws"
COLLECTION = f"{WORKSPACE}/applications"
ALIAS = "strayed-app"
ARTIFACT_ID = f"{WORKSPACE}/{ALIAS}"
STRAY_PARENT = f"{WORKSPACE}/somewhere-else"


def _manifest(version):
    return {
        "id": ALIAS,
        "name": "Strayed App",
        "description": "d",
        "type": "ray-serve",
        "entry": "pkg.mod:Cls",
        "version": version,
    }


class _FakeArtifact:
    """Attribute access, because the call site uses ``getattr`` -- a dict would make every
    lookup fall to its default and the arms would pass for the wrong reason."""

    def __init__(self, artifact_type, parent_id, versions=None):
        self.id = ARTIFACT_ID
        self.alias = ALIAS
        self.type = artifact_type
        self.parent_id = parent_id
        self.config = {}
        self.versions = versions if versions is not None else []
        self.manifest = {"name": "Strayed App"}


class _FakeArtifactManager:
    def __init__(self, existing):
        self._existing = existing
        self.deleted = []
        self.created = []
        self.edited = []
        self.set_parent_calls = []  # list of (args, kwargs)

    async def read(self, artifact_id):
        if self._existing is None:
            raise KeyError(f"Artifact with ID '{artifact_id}' does not exist.")
        return self._existing

    async def delete(self, artifact_id=None, **kw):
        self.deleted.append(artifact_id)
        self._existing = None

    async def create(self, **kw):
        self.created.append(kw)
        return _FakeArtifact(kw.get("type"), kw.get("parent_id"))

    async def edit(self, **kw):
        self.edited.append(kw)
        return _FakeArtifact(
            _APPLICATION_ARTIFACT_TYPE,
            COLLECTION,
            versions=(self._existing.versions if self._existing else []),
        )

    async def set_parent(self, *args, **kwargs):
        self.set_parent_calls.append((args, kwargs))
        # Mirror the real thing: the artifact survives, keeps its versions, and is now in
        # the new collection. A fake that left parent_id stale would misreport the state the
        # code reads back immediately afterwards.
        if self._existing is not None and args:
            self._existing.parent_id = args[-1]


def _versions(*tags):
    return [{"version": t} for t in tags]


def _strayed_app(*tags):
    return _FakeArtifact(
        _APPLICATION_ARTIFACT_TYPE, STRAY_PARENT, versions=_versions(*tags)
    )


# --------------------------------------------------------------------------- #
# the migration itself
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_strayed_application_is_MOVED_and_its_VERSIONS_SURVIVE():
    """The arm this change exists for. RED on the delete-and-recreate code.

    ``deleted`` is asserted FIRST so that a regression's failure message names the data loss
    rather than a missing call: "the strayed application was DELETED" is the finding, and
    "set_parent was not called" is only how it happened.
    """
    existing = _strayed_app("1.0.0", "2.0.0")
    am = _FakeArtifactManager(existing)

    await stage_artifact(
        artifact_manager=am,
        workspace=WORKSPACE,
        manifest=_manifest("3.0.0"),
    )

    assert am.deleted == [], (
        "a strayed application was DELETED; its committed version history and file "
        f"content are gone (delete calls: {am.deleted})"
    )
    assert am.set_parent_calls, "the strayed application was not moved with set_parent"
    assert am.created == [], (
        "the application was recreated rather than moved; a fresh artifact at the same "
        "alias has no version history"
    )
    # The versions are still on the artifact the code went on to edit.
    assert [v["version"] for v in existing.versions] == ["1.0.0", "2.0.0"]
    assert existing.parent_id == COLLECTION


@pytest.mark.asyncio
async def test_set_parent_is_called_POSITIONALLY():
    """`set_parent(artifact_id, parent_id=...)` raises TypeError on the deployed manager.

    Measured, not assumed. A keyword call would fail only at run time against a real Hypha,
    which is exactly the kind of break a fake cannot catch unless something pins the shape.
    """
    am = _FakeArtifactManager(_strayed_app("1.0.0"))
    await stage_artifact(
        artifact_manager=am, workspace=WORKSPACE, manifest=_manifest("2.0.0")
    )
    assert am.set_parent_calls, "set_parent was never called"
    args, kwargs = am.set_parent_calls[0]
    assert "parent_id" not in kwargs, (
        "set_parent was called with parent_id= as a keyword, which raises TypeError "
        f"against the deployed artifact manager (kwargs: {kwargs})"
    )
    assert args[0] == ARTIFACT_ID
    assert args[-1] == COLLECTION


@pytest.mark.asyncio
async def test_a_correctly_placed_application_is_NOT_moved():
    """The control. Without it, an implementation that moved every artifact every time
    would pass the arm above, and nothing would notice the pointless RPC."""
    existing = _FakeArtifact(
        _APPLICATION_ARTIFACT_TYPE, COLLECTION, versions=_versions("1.0.0")
    )
    am = _FakeArtifactManager(existing)

    await stage_artifact(
        artifact_manager=am, workspace=WORKSPACE, manifest=_manifest("2.0.0")
    )

    assert (
        am.set_parent_calls == []
    ), "an application already in the applications collection was moved anyway"
    assert am.deleted == []


@pytest.mark.asyncio
async def test_a_migrated_application_is_STILL_version_gated():
    """The safety consequence of keeping the artifact: the version guard applies again.

    On the delete-and-recreate code this upload SUCCEEDED -- the delete dropped the handle
    to None, so the monotonicity check never ran and a lower version silently replaced a
    published release that happened to have strayed. RED on main for that reason.
    """
    am = _FakeArtifactManager(_strayed_app("1.0.0", "2.0.0"))

    with pytest.raises(ValueError) as excinfo:
        await stage_artifact(
            artifact_manager=am,
            workspace=WORKSPACE,
            manifest=_manifest("1.5.0"),  # lower than the existing 2.0.0
        )

    assert "1.5.0" in str(excinfo.value)
    assert (
        am.deleted == []
    ), "the artifact was deleted before the version check could run"


@pytest.mark.asyncio
async def test_a_strayed_NON_application_is_still_REFUSED_and_NOT_moved():
    """The migration must not become a way to relocate someone else's artifact.

    Guards the non-application refusal against this change: a dataset at the target alias
    must still be
    refused, and must not be quietly moved into the applications collection either -- a move
    is less destructive than a delete but it is still not ours to make.
    """
    am = _FakeArtifactManager(
        _FakeArtifact("dataset", None, versions=_versions("1.0.0"))
    )

    with pytest.raises(ValueError) as excinfo:
        await stage_artifact(
            artifact_manager=am, workspace=WORKSPACE, manifest=_manifest("2.0.0")
        )

    assert am.deleted == [], "a non-application artifact was DELETED"
    assert am.set_parent_calls == [], "a non-application artifact was MOVED"
    assert "not a BioEngine application" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# the refusal message
# --------------------------------------------------------------------------- #


def test_the_refusal_says_WORKSPACE_ROOT_rather_than_collection_None():
    """`parent_id` is None for an artifact at the workspace root -- the common case for the
    dataset or model this guard protects. The message used to render "in collection None",
    which reads as a bug in the error rather than a fact about the artifact."""
    with pytest.raises(ValueError) as excinfo:
        _enforce_replaceable_artifact(
            artifact_id=ARTIFACT_ID,
            artifact_type="dataset",
            current_parent_id=None,
            collection_id=COLLECTION,
        )
    msg = str(excinfo.value)
    assert "at the workspace root" in msg
    assert "collection None" not in msg
    assert "None" not in msg.replace(
        "'dataset'", ""
    ), f"a bare None still leaks into the message: {msg}"


def test_the_refusal_NAMES_the_collection_when_the_artifact_is_in_one():
    """The other branch. Dropping the collection name entirely would also remove
    'collection None' and would pass an absence-only assertion."""
    with pytest.raises(ValueError) as excinfo:
        _enforce_replaceable_artifact(
            artifact_id=ARTIFACT_ID,
            artifact_type="model",
            current_parent_id=f"{WORKSPACE}/datasets",
            collection_id=COLLECTION,
        )
    msg = str(excinfo.value)
    assert f"in collection '{WORKSPACE}/datasets'" in msg
    assert "at the workspace root" not in msg
