"""Saving an annotation on a dataset whose artifact has been COMMITTED.

This is the state every dataset is in after somebody adds an image: the upload
finish commits, and committing is routine besides. `put_file` asserts the
artifact is staged, so minting the annotation's URLs ahead of the restage meant
the first mint raised and the restage-and-retry that exists to repair exactly
that never ran. The user saw "Artifact must be in staging mode." and the
annotation they had just drawn was lost.

The fake artifact manager here MODELS STAGING rather than ignoring it -- it
refuses `put_file` while committed, with Hypha's own wording -- because a
double that accepts every `put_file` cannot fail this test no matter where the
mint sits, which is how the gap survived in the first place.
"""

from __future__ import annotations

import pytest

import broker_core as core
from test_broker_ngff import bk


ARTIFACT = "ws/ds"
LABEL = "l1"
STAGE_ERROR = "Artifact must be in staging mode."


class StagingAm:
    """Just enough artifact manager to have a staged/committed state."""

    def __init__(self, staged: bool):
        self.staged = staged
        self.calls: list[str] = []
        self.minted: list[str] = []

    async def put_file(self, _artifact_id, file_path=None, **_kw):
        self.calls.append(f"put_file:{file_path}")
        if not self.staged:
            # Hypha's actual message, which is what `is_stage_mode_error`
            # matches on. Paraphrasing it here would make the test pass against
            # a matcher that cannot recognise the real thing.
            raise AssertionError(STAGE_ERROR)
        self.minted.append(file_path)
        return f"https://signed.invalid/{file_path}"

    async def edit(self, artifact_id=None, stage=None, **_kw):
        self.calls.append(f"edit:stage={stage}")
        if stage:
            self.staged = True

    async def commit(self, artifact_id=None, **_kw):
        self.calls.append("commit")
        self.staged = False

    async def get_file(self, artifact_id=None, file_path=None, stage=True, **_kw):
        # Reads are NOT gated on staging -- only `put_file` is. Modelling them
        # as refused too would hide which call the staging rule actually bites.
        self.calls.append("get_file:%s" % file_path)
        return "https://signed.invalid/read/%s" % file_path


def _broker(monkeypatch, *, staged: bool):
    b = bk.AnnotationBroker()
    am = StagingAm(staged)
    docs: dict = {}

    monkeypatch.setattr(b, "_canonical_id", lambda a: a)
    monkeypatch.setattr(
        b,
        "_require_role",
        lambda context, artifact_id, minimum: (
            {"labels": [{"name": LABEL}]},
            {"id": "u1", "email": "u1@example.org"},
            "manager",
        ),
    )

    async def _read_json_file(_a, path, default=None):
        return docs.get(path, default if default is not None else {})

    async def _write_json_file(_a, path, data):
        if not am.staged:
            raise AssertionError(STAGE_ERROR)
        docs[path] = data

    monkeypatch.setattr(b, "_read_json_file", _read_json_file)
    monkeypatch.setattr(b, "_write_json_file", _write_json_file)
    monkeypatch.setattr(b, "_am", am)

    # No real backoff. The previous line here set `core.ensure_staged` to
    # ITSELF -- a no-op under a comment claiming it skipped the wait -- so each
    # test really slept the full retry delay. `ensure_staged` takes an
    # injectable `sleep` for exactly this, so the broker's wrapper is replaced
    # with one that passes a sleep doing nothing.
    async def _no_sleep(_seconds):
        return None

    async def _ensure_staged(artifact_id, fn):
        async def restage():
            await am.edit(artifact_id=artifact_id, stage=True)

        return await core.ensure_staged(fn, restage, sleep=_no_sleep)

    monkeypatch.setattr(b, "_ensure_staged", _ensure_staged)
    return b, am, docs


async def _save(b):
    return await bk.AnnotationBroker.get_save_urls(
        b, artifact_id=ARTIFACT, label=LABEL, image_stem="probe", context={}
    )


@pytest.mark.asyncio
async def test_a_save_on_a_COMMITTED_artifact_SUCCEEDS(monkeypatch):
    """The reported break, as the user meets it. RED before the mint moved."""
    b, am, _docs = _broker(monkeypatch, staged=False)

    res = await _save(b)

    assert res["png_put_url"].startswith("https://signed.invalid/")
    assert res["geojson_put_url"].startswith("https://signed.invalid/")
    assert res["png_put_url"] != res["geojson_put_url"]
    assert am.staged is True, "the artifact was left committed"


@pytest.mark.asyncio
async def test_the_RESTAGE_happens_BEFORE_a_successful_mint(monkeypatch):
    """Ordering, not just outcome: the first mint fails, THEN the restage runs.

    Asserting only that the call succeeds would also pass if the artifact had
    simply never been committed, which is the case that was already green.
    """
    b, am, _docs = _broker(monkeypatch, staged=False)

    await _save(b)

    first_put = next(i for i, c in enumerate(am.calls) if c.startswith("put_file"))
    restage = next(i for i, c in enumerate(am.calls) if c == "edit:stage=True")
    assert first_put < restage, (
        "no mint was attempted before the restage, so this test is not "
        f"exercising the committed path at all: {am.calls}"
    )
    assert any(
        c.startswith("put_file") for c in am.calls[restage + 1 :]
    ), f"nothing was minted after the restage: {am.calls}"


@pytest.mark.asyncio
async def test_the_URLS_RETURNED_are_the_ones_MINTED_AFTER_the_restage(monkeypatch):
    """A retry mints again; the caller must get the URLs that actually exist.

    Returning a value captured on the failed first attempt would hand back
    something minted against an artifact state that no longer applies -- or,
    if the code kept the first attempt's variables, nothing at all.
    """
    b, am, _docs = _broker(monkeypatch, staged=False)

    res = await _save(b)

    assert am.minted, "nothing was ever successfully minted"
    returned = {res["png_put_url"], res["geojson_put_url"]}
    available = {f"https://signed.invalid/{p}" for p in am.minted}
    assert returned <= available, (returned, available)


@pytest.mark.asyncio
async def test_an_ALREADY_STAGED_artifact_needs_NO_restage(monkeypatch):
    """The paired control: the fix must not make every save restage.

    Without this, 'the save works' is equally satisfied by code that commits
    and restages on every call, which would be a different defect.
    """
    b, am, _docs = _broker(monkeypatch, staged=True)

    res = await _save(b)

    assert res["png_put_url"]
    assert "edit:stage=True" not in am.calls, (
        f"a staged artifact was restaged anyway: {am.calls}"
    )


@pytest.mark.asyncio
async def test_the_CALLER_is_recorded_in_the_label_users_document(monkeypatch):
    """The upsert must still happen -- it moved into the staged block with the mint."""
    b, _am, docs = _broker(monkeypatch, staged=False)

    await _save(b)

    users_path = core.label_users_path(LABEL)
    assert users_path in docs, f"no users document was written: {sorted(docs)}"
    assert "u1" in str(docs[users_path])


@pytest.mark.asyncio
async def test_a_NON_STAGE_error_is_NOT_retried(monkeypatch):
    """Erring wide on restage must not turn every failure into a retry loop."""
    b, am, _docs = _broker(monkeypatch, staged=True)

    async def _boom(_artifact_id, file_path=None, **_kw):
        am.calls.append("put_file:boom")
        raise RuntimeError("the bucket is on fire")

    monkeypatch.setattr(am, "put_file", _boom)

    with pytest.raises(RuntimeError):
        await _save(b)
    assert "edit:stage=True" not in am.calls, (
        f"a non-staging failure triggered a restage: {am.calls}"
    )


# ── the sibling: embeddings mint an upload URL the same way ─────────────────


def _embedding_broker(monkeypatch, *, staged: bool):
    """Like `_broker`, plus the reads `get_embedding_urls` makes before minting."""
    b, am, docs = _broker(monkeypatch, staged=staged)

    async def _file_exists(_a, _path):
        return False          # no embedding yet, so it takes the minting branch

    async def _image_file_path(_a, stem):
        return "images/%s.png" % stem

    monkeypatch.setattr(b, "_file_exists", _file_exists)
    monkeypatch.setattr(b, "_image_file_path", _image_file_path)
    return b, am, docs


async def _embeddings(b):
    return await bk.AnnotationBroker.get_embedding_urls(
        b, artifact_id=ARTIFACT, image_stem="probe", model_type="vit_b_lm", context={}
    )


@pytest.mark.asyncio
async def test_an_EMBEDDING_url_on_a_COMMITTED_artifact_SUCCEEDS(monkeypatch):
    """The sibling of the save bug, on the path a user walks right after upload.

    Adding an image commits the artifact; the very next thing the annotate page
    needs is an embedding. Minting that URL with a bare `put_file` fails for
    exactly the same reason, and this is the arm that would have caught it
    before a release rather than after.
    """
    b, am, _docs = _embedding_broker(monkeypatch, staged=False)

    res = await _embeddings(b)

    assert res["exists"] is False
    assert res["embedding_put_url"].startswith("https://signed.invalid/")
    assert am.staged is True, "the artifact was left committed"


@pytest.mark.asyncio
async def test_the_EMBEDDING_mint_RESTAGES_before_it_succeeds(monkeypatch):
    """Ordering again: a mint is attempted, fails, THEN the restage runs."""
    b, am, _docs = _embedding_broker(monkeypatch, staged=False)

    await _embeddings(b)

    first_put = next(i for i, c in enumerate(am.calls) if c.startswith("put_file"))
    restage = next(i for i, c in enumerate(am.calls) if c == "edit:stage=True")
    assert first_put < restage, am.calls
    assert any(c.startswith("put_file") for c in am.calls[restage + 1:]), am.calls


@pytest.mark.asyncio
async def test_an_ALREADY_STAGED_artifact_needs_no_restage_for_embeddings(monkeypatch):
    """The paired control, so the fix cannot be 'restage on every call'."""
    b, am, _docs = _embedding_broker(monkeypatch, staged=True)

    res = await _embeddings(b)

    assert res["embedding_put_url"]
    assert "edit:stage=True" not in am.calls, am.calls
