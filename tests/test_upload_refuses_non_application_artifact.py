"""An upload must not delete an artifact that is not a BioEngine application.

``stage_artifact`` resolves the manifest's ``id`` to ``<workspace>/<alias>`` across the
whole workspace, while applications live in ``<workspace>/applications``. The placement
check that follows therefore sees non-application artifacts too -- a dataset, a model, a
collection -- and used to delete every one of them to "recreate" it in the right place.

Two layers of arms, deliberately:

* The ``_enforce_replaceable_artifact`` arms pin the decision itself. They are cheap and
  need no Hypha server, matching ``test_artifact_version_guard.py``.
* The ``stage_artifact`` arms drive the real call site through a fake artifact manager and
  assert on whether ``delete`` was called. Those are the ones that go RED against the
  unfixed code: a pure-helper arm cannot, because on unfixed code the helper does not
  exist and the module fails to import, which is a collection ERROR and not a failure.
"""

import pytest

from bioengine.utils.artifact_utils import (
    _APPLICATION_ARTIFACT_TYPE,
    _enforce_replaceable_artifact,
    stage_artifact,
)

WORKSPACE = "ws"
COLLECTION = f"{WORKSPACE}/applications"
ALIAS = "my-thing"
ARTIFACT_ID = f"{WORKSPACE}/{ALIAS}"


def _manifest(version="1.0.0"):
    return {
        "id": ALIAS,
        "name": "My Thing",
        "description": "d",
        "type": "ray-serve",
        "entry": "pkg.mod:Cls",
        "version": version,
    }


class _FakeArtifact:
    """Stands in for what ``artifact_manager.read`` returns.

    Attribute access, not a dict: the call site uses ``getattr(existing_artifact, ...)``,
    so a dict fixture would make every ``getattr`` fall to its default and the arm would
    pass for the wrong reason.
    """

    def __init__(self, artifact_type, parent_id, versions=None):
        self.id = ARTIFACT_ID
        self.alias = ALIAS
        self.type = artifact_type
        self.parent_id = parent_id
        self.config = {}
        self.versions = versions if versions is not None else []
        self.manifest = {}


class _FakeArtifactManager:
    """Records the calls that matter and fails loudly on the ones that should not happen."""

    def __init__(self, existing):
        self._existing = existing
        self.deleted = []
        self.created = []
        self.edited = []
        self.set_parent_calls = []

    async def set_parent(self, *args, **kwargs):
        # The migration path moves a strayed application instead of rebuilding it. The
        # parent is positional: `parent_id=` raises TypeError on the deployed manager.
        self.set_parent_calls.append((args, kwargs))
        if self._existing is not None and args:
            self._existing.parent_id = args[-1]

    async def read(self, artifact_id):
        if self._existing is None:
            raise KeyError(f"Artifact with ID '{artifact_id}' does not exist.")
        return self._existing

    async def delete(self, artifact_id=None, **kw):
        self.deleted.append(artifact_id)

    async def create(self, **kw):
        self.created.append(kw)
        return _FakeArtifact(kw.get("type"), kw.get("parent_id"))

    async def edit(self, **kw):
        self.edited.append(kw)
        return _FakeArtifact(
            _APPLICATION_ARTIFACT_TYPE, COLLECTION, versions=self._existing.versions
        )


# --------------------------------------------------------------------------- #
# the decision, in isolation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "artifact_type",
    ["dataset", "model", "collection", "generic", "", None],
)
def test_a_non_application_artifact_is_REFUSED(artifact_type):
    """Anything that is not an application artifact must raise, including no type at all.

    Parametrised over the type because that is the input the guard reads; the constant it
    compares against is asserted separately below, so this does not police its own value.
    """
    with pytest.raises(ValueError) as excinfo:
        _enforce_replaceable_artifact(
            artifact_id=ARTIFACT_ID,
            artifact_type=artifact_type,
            current_parent_id=f"{WORKSPACE}/datasets",
            collection_id=COLLECTION,
        )
    msg = str(excinfo.value)
    # The message has to be actionable: name the id, say it would be destroyed, and say
    # what to do. A bare "refused" would satisfy pytest.raises and help nobody.
    assert ARTIFACT_ID in msg
    assert "DELETE" in msg
    assert "manifest.yaml" in msg
    assert repr(artifact_type) in msg


def test_an_application_artifact_in_the_wrong_collection_is_ALLOWED():
    """The migration path this guard must not break."""
    _enforce_replaceable_artifact(
        artifact_id=ARTIFACT_ID,
        artifact_type=_APPLICATION_ARTIFACT_TYPE,
        current_parent_id=f"{WORKSPACE}/something-else",
        collection_id=COLLECTION,
    )


def test_the_guard_compares_against_the_type_the_code_actually_WRITES():
    """Pins the constant to the literal ``create`` uses, in the same module.

    Without this, the guard and the create path could drift apart and every arm above
    would still pass -- they all speak through the same constant.
    """
    import inspect

    from bioengine.utils import artifact_utils

    src = inspect.getsource(artifact_utils.stage_artifact)
    assert f'"type": "{_APPLICATION_ARTIFACT_TYPE}"' in src, (
        "stage_artifact no longer creates artifacts with type "
        f"{_APPLICATION_ARTIFACT_TYPE!r}; the guard is comparing against a dead value."
    )
    # And it is the ARTIFACT type, not the manifest's. If these ever coincide the
    # distinction that made this bug subtle has been lost.
    assert _APPLICATION_ARTIFACT_TYPE != "ray-serve"


def test_the_validate_note_describes_the_REFUSAL_not_a_deletion():
    """The user-facing note must not survive this change unamended.

    Before the fix the honest note said upload DELETES such an artifact. That sentence is
    now false, and a note that confidently describes the opposite of the behaviour is worse
    than a vague one -- so assert the old wording is gone, not merely that new wording was
    added. Rendered through the real command, because a substring search over the source
    proves the bytes and not that the f-string renders.
    """
    from click.testing import CliRunner

    from bioengine.cli.apps import apps_group

    result = CliRunner().invoke(apps_group, ["validate", "apps/model-runner"])
    assert result.exit_code == 0, result.output
    out = result.output

    assert "refuses outright" in out, out
    assert "WHOLE workspace" in out, out
    assert "has to be free in" not in out, out

    # POLICY REFINED after this arm was written: the note may now say an OLD WORKER deletes,
    # because stating the refusal unconditionally misled a reader whose worker still deletes.
    # So a blanket `"DELETES that artifact" not in out` is no longer the right assertion -- it
    # was this arm's original form and it failed the moment the caveat gained the worker
    # boundary. What must hold is that any deletion claim is VERSION-QUALIFIED: it is a
    # statement about a specific worker range, never about the command's own behaviour.
    if "DELETES that artifact" in out:
        assert "0.16.141" in out, (
            "the note says an upload DELETES the artifact without naming the worker range "
            "that does it, which is the unconditional claim this arm exists to prevent"
        )
        assert "bioengine_version" in out, (
            "the note describes a destructive worker without telling the reader how to check "
            "which worker they have"
        )


# --------------------------------------------------------------------------- #
# the real call site -- these go RED on the unfixed code
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stage_artifact_does_NOT_DELETE_a_dataset_at_the_same_alias():
    """The regression. On unfixed code the dataset is deleted and this arm fails."""
    dataset = _FakeArtifact("dataset", f"{WORKSPACE}/datasets")
    am = _FakeArtifactManager(dataset)

    # Catch rather than pytest.raises, and assert on the DELETE first, so that the RED on
    # unfixed code names the DATA LOSS. Wrapped in pytest.raises the unfixed code fails
    # with "DID NOT RAISE" -- the missing refusal -- and the am.deleted assertion below is
    # never reached, so the arm would report the wrong defect for the right reason.
    error = None
    try:
        await stage_artifact(
            artifact_manager=am,
            workspace=WORKSPACE,
            manifest=_manifest(),
        )
    except ValueError as e:
        error = e

    assert (
        am.deleted == []
    ), f"stage_artifact DELETED a non-application artifact: {am.deleted}"
    assert am.created == [], f"it also recreated it as an application: {am.created}"
    assert (
        error is not None
    ), "stage_artifact accepted the upload instead of refusing it"
    assert "not a BioEngine application" in str(error), str(error)


@pytest.mark.asyncio
async def test_a_manifest_id_of_applications_cannot_delete_the_COLLECTION_itself():
    """The worst case: `id: applications` targets the collection that indexes every app.

    ``artifact_id`` is ``f"{ws}/{alias}"`` and ``collection_id`` is ``f"{ws}/applications"``,
    so this alias makes the two equal. The collection's own ``parent_id`` is None, which is
    != the collection id, so the placement check fires on the collection itself. "applications"
    is a legal manifest id -- lowercase, no underscore, a valid identifier -- so nothing
    upstream rejects it.

    Measured live on an unfixed tree in a personal workspace: the collection (42 children)
    was deleted, the apps were ORPHANED rather than destroyed, and the upload then failed
    with a RuntimeError about creating the artifact, so the operator sees an error that says
    nothing about what was lost. ``list_apps`` enumerates this exact collection, so
    ``bioengine apps list`` goes empty while the app artifacts still exist, detached.
    """
    collection = _FakeArtifact("collection", None)
    am = _FakeArtifactManager(collection)

    error = None
    try:
        await stage_artifact(
            artifact_manager=am,
            workspace=WORKSPACE,
            manifest={**_manifest(), "id": "applications"},
        )
    except ValueError as e:
        error = e

    assert (
        am.deleted == []
    ), f"stage_artifact DELETED the applications collection: {am.deleted}"
    assert error is not None, "the upload was accepted against the collection itself"
    assert "'collection'" in str(
        error
    ), f"the refusal should name the artifact type it found: {error}"


@pytest.mark.asyncio
async def test_a_wrong_collection_APPLICATION_is_still_migrated():
    """The behaviour the guard must not swallow: a strayed APPLICATION is still migrated.

    Here so that "refuse everything" is not an accepted answer to this issue. What it must
    NOT pin is HOW the migration happens -- this arm used to require
    ``deleted == [ARTIFACT_ID]``, i.e. it asserted the delete-and-recreate that destroyed
    the artifact's version history, so the arm itself defended the data loss. The migration
    is now a move; the version-preservation arms live in
    ``test_strayed_application_is_migrated_not_recreated.py``.
    """
    strayed_app = _FakeArtifact(_APPLICATION_ARTIFACT_TYPE, f"{WORKSPACE}/elsewhere")
    am = _FakeArtifactManager(strayed_app)

    await stage_artifact(
        artifact_manager=am,
        workspace=WORKSPACE,
        manifest=_manifest(),
    )

    assert am.set_parent_calls, (
        "the strayed application was not migrated at all; the migration path regressed "
        "into refusing it"
    )
    assert (
        am.deleted == []
    ), f"the strayed application was DELETED rather than moved (deleted={am.deleted})"
    assert am.created == [], "it was recreated, which loses its version history"


@pytest.mark.asyncio
async def test_an_application_ALREADY_in_the_collection_is_never_deleted():
    """The common path: edited in place, so the version guard still gets to run."""
    app = _FakeArtifact(
        _APPLICATION_ARTIFACT_TYPE, COLLECTION, versions=[{"version": "0.9.0"}]
    )
    am = _FakeArtifactManager(app)

    await stage_artifact(
        artifact_manager=am,
        workspace=WORKSPACE,
        manifest=_manifest("1.0.0"),
    )

    assert am.deleted == []
    assert am.edited, "the existing artifact was not edited in place"


@pytest.mark.asyncio
async def test_the_version_guard_still_fires_for_an_artifact_in_the_collection():
    """Shows the delete is not silently bypassing the monotonicity check.

    This is the protection the old code skipped in exactly the destructive case: the
    version guard only runs for an artifact the caller KEEPS.
    """
    app = _FakeArtifact(
        _APPLICATION_ARTIFACT_TYPE, COLLECTION, versions=[{"version": "2.0.0"}]
    )
    am = _FakeArtifactManager(app)

    with pytest.raises(ValueError, match="strictly greater"):
        await stage_artifact(
            artifact_manager=am,
            workspace=WORKSPACE,
            manifest=_manifest("1.0.0"),
        )
    assert am.deleted == []
