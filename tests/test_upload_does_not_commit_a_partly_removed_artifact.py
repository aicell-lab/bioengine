"""Two separate things an upload must not do, and only one of them is the field report.

THE FIELD REPORT (the `em-mito-quant-app` skill report, section 1) is an ID COLLISION, not a
removal failure. The reporter's id was being uploaded CONCURRENTLY by another client; that client
committed it mid-upload; this client's `put_file` calls then failed per file with
`AssertionError: Artifact must be in staging mode`, and `am.read` afterwards showed the artifact
**committed at 1.0.0** holding a mix of the other client's files and this client's manifest.yaml.
The report's own words: "the server-side error named staging rather than the collision".

So that error means the artifact is NOT staged — it means somebody else committed the id. An
earlier version of this file said the opposite, and an earlier version of the failure message
asserted "left STAGED and UNCOMMITTED, no published version changed, re-run the upload", which is
false in all three parts for exactly the case that was reported. That is why the message now
RE-READS the artifact instead of describing a state it never checked.

THE SECOND THING is a latent defect found while reading this path, and it is not what the field
hit: the removal loop swallowed its failures with `logger.warning(... "Continuing anyway...")` and
committed regardless, which would publish an immutable version mixing this upload's files with
stale ones. Real, worth fixing, and separate.

WHAT IS PINNED HERE:
  1. a removal failure does not commit, and the message reports the re-read state;
  2. an id collision is reported as a COLLISION, with the artifact's real (committed) state —
     not as a staged artifact;
  3. the happy path still commits, so "never commit" cannot pass as the fix;
  4. a failed upload followed by a successful one works, which is what the "re-stage" advice
     claims and what nothing previously tested.

The fake artifact manager is the Hypha surface these helpers call — `read`, `put_file`,
`remove_file`, `commit` — plus the httpx PUT the uploader makes to the URL `put_file` returns.
"""

from __future__ import annotations

import logging

import pytest

from bioengine.utils.artifact_utils import create_application_from_files

MANIFEST = """\
name: My App
id: my-app
id_emoji: "🧪"
description: a test app
type: ray-serve
entry: my_app.deployment:MyApp
format_version: 0.6.0
version: 1.0.0
"""


class _Artifact:
    """Stands in for the ObjectProxy the artifact manager returns.

    `.config` matters: the collection check reads `collection.config or {}`, so a plain dict
    here fails with `'dict' object has no attribute 'config'` — which is what my first fake did,
    and it made all three arms fail for a reason that had nothing to do with the contract. A
    fake that does not match the surface tests the fake.
    """

    def __init__(self, artifact_id, config=None, versions=None, parent_id=None):
        self.id = artifact_id
        self.config = config if config is not None else {"permissions": {"*": "r"}}
        self.versions = versions if versions is not None else []
        self.parent_id = parent_id


class _File:
    def __init__(self, name):
        self.name = name


class _FakeArtifactManager:
    """Enough of the Hypha artifact manager for this path, and it RECORDS the calls.

    `commits` is the assertion that matters: the question is not what the function logged but
    whether it published anything.
    """

    def __init__(self, existing=(), remove_raises=()):
        self.existing = list(existing)
        self.remove_raises = set(remove_raises)
        self.put_files = []
        self.removed = []
        self.commits = []
        self.edits = []

    async def read(self, artifact_id=None, **kw):
        # An EXISTING artifact, already in the right collection — which is the only situation
        # this contract can arise in: stale files to remove exist only on a RE-upload. Its
        # `versions` is empty so the monotonicity check passes, and `parent_id` matches the
        # collection so stage_artifact takes the edit path rather than delete-and-recreate.
        return _Artifact(artifact_id, parent_id="ws/applications", versions=[])

    async def edit(self, *a, **kw):
        self.edits.append((a, kw))
        return _Artifact(kw.get("artifact_id") or "ws/my-app")

    async def create(self, *a, **kw):
        return _Artifact("ws/my-app")

    async def list_files(self, artifact_id=None, **kw):
        return [_File(n) for n in self.existing]

    async def put_file(self, artifact_id=None, file_path=None, **kw):
        self.put_files.append(file_path)
        return f"https://upload.example/{file_path}"

    async def remove_file(self, artifact_id=None, file_path=None, **kw):
        if file_path in self.remove_raises:
            raise RuntimeError(f"permission denied removing {file_path}")
        self.removed.append(file_path)

    async def commit(self, artifact_id=None, version=None, **kw):
        self.commits.append(version)
        return _Artifact(artifact_id or "ws/my-app")


@pytest.fixture(autouse=True)
def _no_real_http(monkeypatch):
    """`upload_file_to_artifact` PUTs to the URL `put_file` returns, with a real httpx client.

    Left alone, the arms failed with `[Errno -2] Name or service not known` — a DNS lookup for
    the fake upload host. That is the fixture being less than the surface it stands in for: the
    upload path is artifact-manager call PLUS an HTTP hop, and a fake that covers only the first
    tests half the path and reports the other half as the defect.

    Patched at the httpx boundary so the arms still drive the real uploader.
    """
    import httpx

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def put(self, url, *a, **kw):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)


def _files():
    return [
        {"name": "manifest.yaml", "content": MANIFEST, "type": "text"},
        {"name": "deployment.py", "content": "class MyApp: ...\n", "type": "text"},
    ]


async def _upload(manager):
    return await create_application_from_files(
        artifact_manager=manager,
        files=_files(),
        workspace="ws",
        user_id="someone",
        logger=logging.getLogger("test.upload"),
    )


@pytest.mark.asyncio
async def test_a_removal_failure_is_NOT_committed() -> None:
    """The defect: a stale file that could not be removed was published anyway.

    `stale.py` exists on the artifact and is absent from the new file set, so it must go. The
    removal fails. Committing here publishes an immutable version containing both the new files
    and `stale.py`, which is the "committed, mixed artifact" measured in the field — and which
    cannot be corrected afterwards, only superseded.
    """
    mgr = _FakeArtifactManager(existing=["stale.py"], remove_raises=["stale.py"])

    # Caught by hand rather than with `pytest.raises`, so that the FIRST thing asserted is the
    # harm and not the symptom. Wrapped in `pytest.raises`, a tree without the fix reports only
    # "DID NOT RAISE <class 'RuntimeError'>" and the commit assertion below is never reached —
    # measured on origin/main. The reader of a red arm should be told that a mixed version was
    # published, not that an exception was missing.
    raised = None
    try:
        await _upload(mgr)
    except RuntimeError as exc:
        raised = exc

    assert mgr.commits == [], (
        "a version was COMMITTED although a stale file could not be removed, so a mixture of "
        f"this upload and the previous one is now published and immutable: commits={mgr.commits}"
    )
    assert raised is not None, (
        "nothing was committed, but the caller was not told the upload failed either — a silent "
        "no-op is its own defect"
    )
    text = str(raised)
    assert (
        "stale.py" in text
    ), f"the error does not name the file that survived: {text!r}"
    # Asserted on the RE-READ, not on a fixed phrase. The contract is that the message reports a
    # state it actually looked at: describing it from memory is how the old message came to be
    # false in all three of its parts for the case the field actually hit.
    assert "Re-read of" in text, (
        "the message describes the artifact's state without re-reading it, which is the defect "
        f"review refused this branch for: {text!r}"
    )
    assert "STAGED" in text and "no committed version" in text, (
        "the error must say what state the artifact was LEFT in — staged and recoverable is a "
        f"different situation from published and not, and wants a different action: {text!r}"
    )


@pytest.mark.asyncio
async def test_a_clean_upload_still_commits() -> None:
    """The control, and the one that stops "never commit" passing as the fix.

    Without it, deleting the commit call entirely would satisfy the arm above.
    """
    mgr = _FakeArtifactManager(existing=["stale.py"])

    artifact_id = await _upload(mgr)

    assert mgr.commits == [
        "1.0.0"
    ], f"a clean upload must publish exactly its manifest version, got {mgr.commits}"
    assert mgr.removed == [
        "stale.py"
    ], f"the stale file should have been removed on the happy path, got {mgr.removed}"
    assert artifact_id


@pytest.mark.asyncio
async def test_an_upload_failure_also_names_the_state_it_left() -> None:
    """The other half of the same contract: a failed PUT leaves it staged, and says so.

    This path already raised before committing, so the artifact was always left recoverable —
    but the message said only "failed to upload N files", which does not distinguish the
    recoverable outcome from the unrecoverable one. The staged version it leaves behind is also
    what the next attempt meets, which is where a later "must be in staging mode" comes from.
    """

    class _PutFails(_FakeArtifactManager):
        async def put_file(self, artifact_id=None, file_path=None, **kw):
            if file_path == "deployment.py":
                raise RuntimeError("upload stream reset")
            return await super().put_file(
                artifact_id=artifact_id, file_path=file_path, **kw
            )

    mgr = _PutFails(existing=[])

    raised = None
    try:
        await _upload(mgr)
    except RuntimeError as exc:
        raised = exc

    assert mgr.commits == [], f"a failed upload was committed: {mgr.commits}"
    assert raised is not None, "a failed upload reported nothing to the caller"
    text = str(raised)
    assert (
        "deployment.py" in text
    ), f"the error does not name the file that failed: {text!r}"
    assert (
        "Re-read of" in text and "STAGED" in text
    ), f"the error does not report a RE-READ state for the artifact: {text!r}"


@pytest.mark.asyncio
async def test_an_ID_COLLISION_is_reported_as_one_with_the_artifacts_REAL_state() -> (
    None
):
    """The case the field report actually describes, which no arm covered before.

    Another client commits the id mid-upload. Our `put_file` then fails with the server's
    staging assertion, and the artifact is COMMITTED — by them, holding their files plus
    whatever of ours landed first.

    The old message claimed the artifact was "left STAGED and UNCOMMITTED, so no published
    version was changed" and advised re-running the upload. Measured against this case, all
    three were false: it is committed, a published version did change, and a re-run under the
    same id meets the strictly-greater-version refusal the reporter met.
    """

    class _CommittedByOther(_FakeArtifactManager):
        """The other client commits MID-upload, so the state CHANGES under this one.

        Modelling the timing correctly matters, and getting it wrong is what told me so. With
        1.0.0 already present at STAGE time, `stage_artifact` refuses up front with its own very
        clear message and there is no defect to fix at all — measured:

            ValueError: Cannot upload 'ws/my-app' at version '1.0.0': must be strictly greater
            than the highest existing version '1.0.0' … the id collided with someone else's
            artifact, so choose a different id instead …

        So the version list is EMPTY when we stage and holds 1.0.0 by the time the failure is
        re-read. That gap is the entire argument for re-reading rather than describing the state
        from memory: the state at stage time is not the state at failure time.
        """

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.staged = False

        async def read(self, artifact_id=None, **kw):
            versions = [{"version": "1.0.0"}] if self.staged else []
            return _Artifact(
                artifact_id, parent_id="ws/applications", versions=versions
            )

        async def edit(self, *a, **kw):
            self.staged = True  # from here on, the other client's commit is visible
            return await super().edit(*a, **kw)

        async def put_file(self, artifact_id=None, file_path=None, **kw):
            raise RuntimeError(
                "Failed to get upload URL for "
                f"'{file_path}': AssertionError: Artifact must be in staging mode."
            )

    mgr = _CommittedByOther()

    raised = None
    try:
        await _upload(mgr)
    except RuntimeError as exc:
        raised = exc

    assert (
        mgr.commits == []
    ), f"this client must not add a commit of its own: {mgr.commits}"
    assert raised is not None, "a wholly failed upload reported nothing"
    text = str(raised)

    assert "STAGED" not in text, (
        "the message claims the artifact is STAGED while the server's own error says it is not "
        f"and the re-read says it is committed: {text!r}"
    )
    assert (
        "COMMITTED" in text and "1.0.0" in text
    ), f"the message must report the artifact's real state, re-read: {text!r}"
    assert "concurrent upload of the same id" in text, (
        "the staging assertion must be explained as an id collision — the report's whole point "
        f"was that the server error named staging rather than the collision: {text!r}"
    )
    assert (
        "different id" in text
    ), f"the message must say what to actually do about a collision: {text!r}"


@pytest.mark.asyncio
async def test_a_failed_upload_can_be_FOLLOWED_by_a_successful_one() -> None:
    """Backs the "re-running re-stages it" advice, which was an inference until now.

    Review's F3: nothing modelled a second upload after a failure, so the message's own remedy
    was untested. The fake keeps its staged state between calls, as a real artifact would.
    """
    calls = {"n": 0}

    class _FailsFirstTime(_FakeArtifactManager):
        async def put_file(self, artifact_id=None, file_path=None, **kw):
            if calls["n"] == 0 and file_path == "deployment.py":
                raise RuntimeError("upload stream reset")
            return await super().put_file(
                artifact_id=artifact_id, file_path=file_path, **kw
            )

    mgr = _FailsFirstTime(existing=[])

    with pytest.raises(RuntimeError):
        await _upload(mgr)
    assert mgr.commits == [], "the failed attempt must not have committed"

    calls["n"] = 1
    artifact_id = await _upload(mgr)

    assert mgr.commits == [
        "1.0.0"
    ], f"the retry after a failure did not publish its version: {mgr.commits}"
    assert artifact_id
    assert (
        "deployment.py" in mgr.put_files
    ), f"the retry did not re-upload the file that failed: {mgr.put_files}"
