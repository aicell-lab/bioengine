"""An upload that could not remove its stale files must not publish the mixture.

MEASURED IN THE FIELD. An `upload_app` that failed left "a committed, mixed artifact", and the
next attempt met `must be in staging mode` — the staged-mode error being the first thing the user
actually saw, two steps from the cause.

THE MECHANISM. `create_application_from_files` stages the artifact, uploads the new files, removes
the files the new set no longer contains, and commits. The removal loop used to swallow its
failures::

    except Exception as e:
        logger.warning(f"Failed to remove old file '{file_name}': {e}. Continuing anyway...")

and then committed anyway. A committed artifact version is IMMUTABLE and is exactly what a deploy
runs, so that published a version holding this upload's files PLUS leftovers from the previous
one, permanently, under a version string that can never mean anything else. The only thing
standing between a partial removal and a permanently wrong published version was a log line at
WARNING.

WHAT IS PINNED HERE, as the two outcomes an upload is allowed to have:

  1. a removal failure does NOT commit — the artifact is left staged, which is recoverable,
     and the error says so;
  2. the happy path still commits, so "never commit" cannot pass as the fix.

The fake artifact manager is the Hypha surface these helpers actually call — `read`, `put_file`,
`remove_file`, `commit` — so the arms drive the shipped function rather than a restatement of it.
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
    assert "STAGED" in text and "UNCOMMITTED" in text, (
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
        "STAGED" in text and "UNCOMMITTED" in text
    ), f"the error does not say what state the artifact was left in: {text!r}"
