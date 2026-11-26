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

import json
import time

import pytest

import broker_core as core

# Nothing is in flight, said explicitly. `merge_image_manifest` REQUIRES
# `withheld_paths` and has no default, because an omission there is silent and
# changes correctness -- a review measured both manifest writers omitting it
# and recording somebody's unfinished upload as a permanent dataset image.
NOTHING_WITHHELD = frozenset()
from test_broker_ngff import bk
from test_local_image_upload_route import ARTIFACT, _broker, _open_mint

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
    """A tiny object store: PUT records bytes, the listing reports lengths.

    `published` models an object a MOUNTED-folder upload already committed --
    the case the discard must never touch. It is deliberately ALSO visible in
    the staged view, and the staged and committed listings here are deliberately
    the same, because that is what the real store does: measured against the
    live artifact manager, an object just PUT into an open stage is returned by
    `stage=True`, by `stage` unset, by `stage=False`, by `version="v0"` and by
    `version="latest"` alike. An earlier version of this double kept the two
    views apart, which modelled a distinction the store does not make -- and a
    guard resting on that distinction passed here while being certain to refuse
    every real upload in production.

    Markers are ordinary objects in `objects`, under `.uploads/`, so removing
    one goes through the same `remove_file` as an image and shows up in
    `removed`. Nothing about the marker is special-cased in the double.
    """

    def __init__(self):
        self.objects: dict = {}  # everything staged, markers included
        self.published: dict = {}  # committed by some other route
        self.removed: list = []
        self.committed = 0
        self.put_file_calls: list = []  # (file_path, expires_in) per mint
        self.listing_calls: list = []  # (dir_path, offset, asked, served) per read

    # -- the artifact-manager surface the route uses -------------------
    async def list_files(
        self, _artifact_id, dir_path=None, stage=True, limit=None, offset=0
    ):
        """The artifact-manager surface the paging loop calls, WITH limit and
        offset. Separate from `listing` so a test can see which pages were
        asked for."""
        return self.listing(dir_path, stage=stage, limit=limit, offset=offset)

    async def put_file(self, _artifact_id, file_path=None, expires_in=None):
        # `expires_in` is RECORDED, not ignored. The real `put_file` defaults it
        # to an hour and the route now passes a short value deliberately, so a
        # double that swallowed the argument would let that regress silently --
        # and one that did not accept it at all would turn the route's call into
        # a TypeError, which is how this double started out.
        self.put_file_calls.append((file_path, expires_in))
        return f"https://store.invalid/{file_path}"

    async def remove_file(self, artifact_id=None, file_path=None):
        self.removed.append(file_path)
        self.objects.pop(file_path, None)
        self.published.pop(file_path, None)

    # -- what a client does with the minted URL ------------------------
    def put(self, url: str, body: bytes):
        self.objects[url.split("://", 1)[1].split("/", 1)[1]] = body

    # -- what the broker reads back ------------------------------------
    def publish(self, path: str, body: bytes):
        """Pre-existing committed object, as a mounted-folder upload leaves."""
        self.published[path] = body
        self.objects[path] = body

    #: Entries one `list_files` call returns. The real one is PAGINATED -- its
    #: default page is 1000 and the documented maximum is 10000 -- so a double
    #: that answered the whole directory in one call modelled a server that does
    #: not exist, and a review measured the consequence: everything past the
    #: first page was invisible to a decision that had to see all of it. Small
    #: here so the paging path is exercised by tens of entries rather than
    #: thousands.
    page_size = 50

    def listing(self, dir_path: str, stage: bool = True, limit=None, offset=0):
        # `stage` is accepted and IGNORED, on purpose: see the class docstring.
        # The real store answers both views the same for a staged object, and a
        # double that disagreed with it is what let a previous round ship.
        source = {**self.published, **self.objects}
        out = []
        for path, body in sorted(source.items()):
            head, _, nm = path.rpartition("/")
            if head != dir_path:
                continue
            out.append({"name": nm, "type": "file", "size": len(body)})
        start = int(offset or 0)
        asked = int(limit) if limit else self.page_size
        size = min(asked, self.page_size)  # the server's own page caps the ask
        # BOTH numbers are recorded: what was ASKED for and what was SERVED. The
        # whole round-7 finding is that those differ, so a record that kept only
        # the served size could not show whether the caller asked for more --
        # which is what a first version of this test tried to assert and could
        # not.
        self.listing_calls.append((dir_path, start, asked, size))
        return out[start : start + size]


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

    async def _read_json_file(_a, path, _default=None):
        if path == core.IMAGE_MANIFEST_PATH:
            # THE STORED MANIFEST IS A REAL DOCUMENT, read back out of the same
            # store a write puts it in. It used to be hard-coded empty, which
            # made every manifest WRITE invisible to the next read -- so a test
            # could not see `sync_image_manifest` recording an unfinished
            # upload, which is the defect a review measured on this very path.
            # A double that cannot represent the write cannot test the writer.
            raw = store.objects.get(path)
            if raw is None:
                return {"manifest_version": core.IMAGE_MANIFEST_VERSION, "images": {}}
            return json.loads(raw.decode("utf-8"))
        raw = store.objects.get(path)
        if raw is None:
            return _default
        return json.loads(raw.decode("utf-8"))

    async def _write_json_file(_a, path, data):
        store.objects[path] = json.dumps(data).encode("utf-8")

    async def _list_files_safe(_a, dir_path, stage=True):
        return store.listing(dir_path, stage=stage)

    # `_list_files_all` is NOT replaced: it is the paging loop under test, and
    # it must drive the store's own pagination. It calls `self._am.list_files`,
    # which is the store, so only the AM surface below needs to accept the
    # limit/offset it passes.

    async def _ensure_staged(_a, fn):
        return await fn()

    async def _commit(_a):
        store.committed += 1

    async def _object_head(_a, path, n):
        return store.objects.get(path, b"")[:n]

    monkeypatch.setattr(b, "_read_json_file", _read_json_file)
    monkeypatch.setattr(b, "_write_json_file", _write_json_file)
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


def _marker_for(name: str) -> str:
    """The marker file for an uploaded image *name*, via the SHIPPED keying.

    Through `broker_core` rather than by formatting a digest here: a test that
    recomputed the filename would still pass if the app and the test drifted
    apart in the same direction, and the marker's name is the whole mechanism
    now that the finish finds its record by computing it from the object path.
    Takes the NAME, not the mint id -- a re-mint of one name overwrites one
    file, which is why there is no longer a duplicate for the finish to find.
    """
    return core.upload_marker_path_for_object(f"images/{name}")


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
        await b.upload_local_image(
            ARTIFACT, "scan01.png", done=True, mint_id=mint["mint_id"]
        )
    assert "65" in str(exc.value) and "64" in str(exc.value), str(exc.value)

    assert store.committed == 0, "the oversized object was committed"
    # THE FINDING round 1 made: refusing is not enough. The dataset's image
    # list is built from the STAGED listing, so an object left in place
    # becomes an image regardless of this route's commit.
    assert "images/scan01.png" in store.removed, "the refused object was left staged"
    manifest = await b._image_manifest(ARTIFACT)
    assert "scan01" not in manifest, f"the refused object is still an image: {manifest}"
    # The marker goes with the object: one mint authorises exactly one finish.
    marker_path = _marker_for("scan01.png")
    assert marker_path in store.removed, store.removed
    assert marker_path not in store.objects


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
        await b.upload_local_image(
            ARTIFACT, "scan01.png", done=True, mint_id=mint["mint_id"]
        )

    again = await b.upload_local_image(ARTIFACT, "scan01.png")
    assert again["already"] is False, f"the name is still taken: {again}"
    assert "put_url" in again
    # A FRESH mint, not the discarded one. The discard consumed the first
    # marker, so the first mint_id is spent and the retry gets its own.
    assert again["mint_id"] != mint["mint_id"], again

    # And the retry with a body that fits completes.
    store.put(again["put_url"], _png(32))
    out = await b.upload_local_image(
        ARTIFACT, "scan01.png", done=True, mint_id=again["mint_id"]
    )
    assert out["committed"] is True
    assert store.committed == 1

    # The spent first mint_id cannot be replayed against the committed image.
    store.removed.clear()
    with pytest.raises(ValueError) as exc:
        await b.upload_local_image(
            ARTIFACT, "scan01.png", done=True, mint_id=mint["mint_id"]
        )
    assert "NOTHING was changed" in str(exc.value), str(exc.value)
    assert store.removed == [], f"a replayed mint id deleted something: {store.removed}"


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
        await b.upload_local_image(
            ARTIFACT, "scan01.png", done=True, mint_id=mint["mint_id"]
        )
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

    out = await b.upload_local_image(
        ARTIFACT, "scan01.png", done=True, mint_id=mint["mint_id"]
    )
    assert out["committed"] is True
    # The MARKER is removed on success -- it is consumed, not discarded -- but
    # no image is. Asserting on `removed` wholesale would conflate the two, so
    # this names the object and the marker separately.
    marker_path = _marker_for("scan01.png")
    assert store.removed == [
        marker_path
    ], f"a good upload was discarded: {store.removed}"
    assert "images/scan01.png" in store.objects, "the accepted image was removed"
    # And `.uploads/` is not left behind in the committed dataset.
    assert marker_path not in store.objects

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
        await b.upload_local_image(
            ARTIFACT, "scan01.png", done=True, mint_id=_open_mint(b)
        )

    msg = str(exc.value)
    assert str(CEILING) in msg and str(CEILING + 1) in msg, msg
    assert b._tstate["committed"] == 0, "an oversized object was committed"


@pytest.mark.asyncio
async def test_an_object_AT_the_ceiling_still_completes(monkeypatch):
    """Clause 2, and the control. Without it, a change that refused everything
    would satisfy the test above -- which is how a check that breaks the
    feature passes for the wrong reason."""
    b = _broker(monkeypatch, manifest={}, listing=_listing(CEILING))

    out = await b.upload_local_image(
        ARTIFACT, "scan01.png", done=True, mint_id=_open_mint(b)
    )

    assert out["committed"] is True, out
    assert b._tstate["committed"] == 1


@pytest.mark.asyncio
async def test_an_ordinary_small_object_still_completes(monkeypatch):
    b = _broker(monkeypatch, manifest={}, listing=_listing(2048))
    out = await b.upload_local_image(
        ARTIFACT, "scan01.png", done=True, mint_id=_open_mint(b)
    )
    assert out["committed"] is True, out
    assert b._tstate["committed"] == 1


@pytest.mark.asyncio
async def test_an_object_whose_size_is_UNREADABLE_is_refused(monkeypatch):
    b = _broker(monkeypatch, manifest={}, listing=_listing(None))
    with pytest.raises(ValueError) as exc:
        await b.upload_local_image(
            ARTIFACT, "scan01.png", done=True, mint_id=_open_mint(b)
        )
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
        await b.upload_local_image(
            ARTIFACT, "scan01.png", done=True, mint_id=_open_mint(b)
        )
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
        await b.upload_local_image(
            ARTIFACT, "scan01.png", done=True, mint_id=_open_mint(b)
        )
    assert read == [], f"the oversized object was read anyway: {read}"


@pytest.mark.asyncio
async def test_the_MINTING_call_still_advertises_the_same_number(monkeypatch):
    """The advertised value was never wrong -- the comparison was missing. This
    pins that the two now agree, so a future change to the ceiling cannot move
    one without the other."""
    b = _broker(monkeypatch, manifest={}, listing=[])
    out = await b.upload_local_image(ARTIFACT, "scan01.png")
    assert out["max_bytes"] == CEILING


# ── an EXISTING image is never touched ──────────────────────────────────────
#
# The regression the discard introduced, found in review round 2. An image
# already in a dataset can be over THIS route's ceiling, because the
# mounted-folder path has no cap and the ceiling binds this route only. The
# discard removed whatever sat at `images/<name>` whenever the size check
# refused, so one finishing call for an existing image's filename deleted that
# image and orphaned its annotations. Before the discard, that call was a
# harmless commit.


@pytest.mark.asyncio
async def test_finishing_for_an_EXISTING_IMAGE_destroys_nothing(monkeypatch):
    """A review's own probe, as a test."""
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    store.publish("images/big.png", _png(100))  # committed, over this ceiling

    # The mint refuses the name, as it always did -- and THEREFORE mints no
    # marker. That is the chain the safety rests on: no marker is ever written
    # for a path that is already an image, so nothing can ever authorise
    # touching one. The two halves are one mechanism.
    mint = await b.upload_local_image(ARTIFACT, "big.png")
    assert mint["already"] is True, mint
    assert "mint_id" not in mint, mint
    assert not [
        p for p in store.objects if p.startswith(core.UPLOAD_MARKER_DIR)
    ], f"a marker was minted for an existing image: {sorted(store.objects)}"

    # A client that ignores that and finishes anyway must change NOTHING --
    # with no mint id, with a made-up one, and with a well-formed one that
    # names no marker.
    for bad in (None, "not-a-mint-id", core.new_mint_id()):
        with pytest.raises(ValueError) as exc:
            if bad is None:
                await b.upload_local_image(ARTIFACT, "big.png", done=True)
            else:
                await b.upload_local_image(ARTIFACT, "big.png", done=True, mint_id=bad)
        assert "NOTHING was changed" in str(exc.value), str(exc.value)
        assert store.removed == [], f"an existing image was deleted: {store.removed}"

    assert store.removed == [], f"an existing image was deleted: {store.removed}"
    assert store.committed == 0
    manifest = await b._image_manifest(ARTIFACT)
    assert (
        "big" in manifest
    ), f"the existing image is gone from the manifest: {manifest}"


@pytest.mark.asyncio
async def test_an_existing_image_WITH_NO_REPORTED_SIZE_is_also_safe(monkeypatch):
    """The unreadable-size branch had the same shape, and would have deleted
    an existing image whose listing entry carried no size."""
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    store.publish("images/odd.png", _png(32))

    # strip the size from what the store reports, as a store that does not
    # report sizes would
    real_listing = store.listing

    def no_sizes(dir_path, stage=True):
        return [
            {k: v for k, v in e.items() if k != "size"}
            for e in real_listing(dir_path, stage=stage)
        ]

    monkeypatch.setattr(store, "listing", no_sizes)

    # The marker check runs BEFORE the size is read, so an existing image is
    # safe whether or not its size can be read at all. That ordering is the
    # point: the unreadable-size branch refuses AND discards, so reaching it
    # without a marker is what used to delete the image.
    with pytest.raises(ValueError) as exc:
        await b.upload_local_image(ARTIFACT, "odd.png", done=True)
    assert "NOTHING was changed" in str(exc.value), str(exc.value)
    assert store.removed == [], store.removed
    assert "images/odd.png" in store.objects, "the existing image was deleted"


@pytest.mark.asyncio
async def test_a_GENUINE_upload_still_discards_on_refusal(monkeypatch):
    """The control for the guard above: a name that is NOT already committed
    still gets the round-1 behaviour, so the guard did not simply disable the
    discard."""
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    mint = await b.upload_local_image(ARTIFACT, "fresh.png")
    store.put(mint["put_url"], _png(65))

    with pytest.raises(ValueError) as exc:
        await b.upload_local_image(
            ARTIFACT, "fresh.png", done=True, mint_id=mint["mint_id"]
        )
    assert "65" in str(exc.value), str(exc.value)
    assert store.removed == [
        "images/fresh.png",
        _marker_for("fresh.png"),
    ], store.removed


# ── clause 1, the hole a round-4 review measured: never calling finish ───────
#
# The dataset's image list is built by unioning the stored manifest with a
# listing of `images/`, so an object that has merely been PUT is an image
# whether or not the finishing call is ever made. A caller that simply does not
# finish therefore bypassed the ceiling completely -- the finishing check cannot
# refuse a call nobody makes. Measured in review at 10,000 bytes against a
# 64-byte ceiling.


@pytest.mark.asyncio
async def test_an_upload_that_NEVER_FINISHES_is_not_an_image(monkeypatch):
    """The review's probe, as a test: mint, PUT over the ceiling, never finish."""
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    mint = await b.upload_local_image(ARTIFACT, "scan01.png")
    store.put(mint["put_url"], _png(10000))  # 156x the ceiling

    manifest = await b._image_manifest(ARTIFACT)

    assert "scan01" not in manifest, (
        "an upload that was never finished is listed as a dataset image, so the "
        f"ceiling can be bypassed by not calling finish: {manifest}"
    )


@pytest.mark.asyncio
async def test_a_FINISHED_upload_IS_an_image(monkeypatch):
    """THE CONTROL. Withholding every PUT object would satisfy the test above
    and break the feature outright, so the paired case has to be here: an
    upload that completes within the ceiling must appear."""
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    mint = await b.upload_local_image(ARTIFACT, "scan01.png")
    store.put(mint["put_url"], _png(32))
    out = await b.upload_local_image(
        ARTIFACT, "scan01.png", done=True, mint_id=mint["mint_id"]
    )
    assert out["committed"] is True

    manifest = await b._image_manifest(ARTIFACT)

    assert "scan01" in manifest, (
        f"a completed upload inside the ceiling is missing from the image "
        f"list: {manifest}"
    )


@pytest.mark.asyncio
async def test_an_upload_whose_marker_EXPIRED_unfinished_loses_its_OBJECT(
    monkeypatch,
):
    """The second half of the fix: at expiry the object goes, not just the marker.

    Withholding alone would leave an oversized object sitting at the image's
    path for the whole marker TTL, publishable by any other route's commit. The
    expired marker is the warrant for deleting it.

    The marker is aged by rewriting it through the shipped builder with an
    expiry in the past -- not by patching the clock, which would test the patch.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    mint = await b.upload_local_image(ARTIFACT, "scan01.png")
    store.put(mint["put_url"], _png(10000))
    marker_path = _marker_for("scan01.png")
    stale = core.upload_marker_document(
        path="images/scan01.png",
        mint_id=mint["mint_id"],
        max_bytes=64,
        minted_at=time.time() - 10 * core.UPLOAD_MARKER_TTL_S,
    )
    store.objects[marker_path] = json.dumps(stale).encode("utf-8")

    swept = await b._sweep_expired_markers(ARTIFACT)

    assert swept == 1, swept
    assert (
        "images/scan01.png" in store.removed
    ), f"the abandoned object survived its marker's expiry: {store.removed}"
    assert "images/scan01.png" not in store.objects
    manifest = await b._image_manifest(ARTIFACT)
    assert "scan01" not in manifest, manifest


@pytest.mark.asyncio
async def test_the_sweep_does_NOT_delete_an_object_whose_marker_is_OPEN(monkeypatch):
    """THE CONTROL for the sweep. A sweep that deleted on every pass would
    satisfy the test above and destroy uploads in flight."""
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    mint = await b.upload_local_image(ARTIFACT, "scan01.png")
    store.put(mint["put_url"], _png(32))

    swept = await b._sweep_expired_markers(ARTIFACT)

    assert swept == 0, swept
    assert "images/scan01.png" in store.objects, "an in-flight upload was deleted"
    assert "images/scan01.png" not in store.removed


@pytest.mark.asyncio
async def test_the_sweep_NEVER_deletes_an_image_the_dataset_ALREADY_HAD(monkeypatch):
    """The sweep's hard limit, and the reason an unparseable marker authorises
    no object removal: it names no path, and a path guessed from a filename is
    the one way this could delete somebody's image."""
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    store.publish("images/already.png", _png(16))
    # An expired marker that cannot be parsed, plus one for a DIFFERENT path.
    store.objects[f"{core.UPLOAD_MARKER_DIR}/{'a' * 32}.json"] = b"{ not json"

    await b._sweep_expired_markers(ARTIFACT)

    assert "images/already.png" in store.objects, store.removed
    assert "images/already.png" not in store.removed


@pytest.mark.asyncio
async def test_the_minted_URL_is_given_a_SHORT_life_not_the_hour_default(monkeypatch):
    """Clause 3's point-in-time limitation, bounded rather than documented away.

    The size check reads the object once. The presigned URL stays writable for
    its whole life and writes the same key the image lives at, so after a
    successful finish a client can PUT an oversized body to the same URL. This
    app cannot close that without the storage service; what it must do is pass a
    short expiry instead of accepting the one-hour default.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    await b.upload_local_image(ARTIFACT, "scan01.png")

    assert store.put_file_calls, "no put_file call was recorded"
    path, expires_in = store.put_file_calls[-1]
    assert path == "images/scan01.png", path
    assert expires_in == core.UPLOAD_URL_TTL_S, expires_in
    # An absolute bound, not one derived from the constant under test: a
    # threshold read out of the thing it is checking cannot fail.
    assert isinstance(expires_in, int) and 0 < expires_in <= 600, expires_in
    # And it must be shorter than the marker's life, or a PUT landing just
    # before the URL dies could not be finished.
    assert expires_in < core.UPLOAD_MARKER_TTL_S


# ── round 5: the gap after expiry, and the sweep deleting FINISHED images ────


def _age_marker(store, mint_id, *, path, ceiling=64):
    """Rewrite a marker through the SHIPPED builder with an expiry in the past.

    Not by patching the clock: a test that moves `time.time` measures the patch.
    """
    marker_path = core.upload_marker_path_for_object(path)
    stale = core.upload_marker_document(
        path=path,
        mint_id=mint_id,
        max_bytes=ceiling,
        minted_at=time.time() - 10 * core.UPLOAD_MARKER_TTL_S,
    )
    store.objects[marker_path] = json.dumps(stale).encode("utf-8")
    return marker_path


@pytest.mark.asyncio
async def test_an_EXPIRED_marker_STILL_withholds_until_the_object_is_SWEPT(
    monkeypatch,
):
    """The round-5 probe. Expiry must not re-admit the object.

    An earlier version lifted the withholding at expiry, reasoning the sweep was
    about to delete the file. A review measured the gap: the sweep ran only at
    the END of a successful mint, and minting the SAME name returns
    `already: True` before reaching it -- so on a dataset where nobody uploads
    anything else, "mint, PUT oversized, never finish, wait for expiry" left the
    object as a dataset image with nothing bounding how long.

    No sweep is run here, which is the whole point: the stem must stay absent.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    mint = await b.upload_local_image(ARTIFACT, "scan01.png")
    store.put(mint["put_url"], _png(10000))
    _age_marker(store, mint["mint_id"], path="images/scan01.png")

    manifest = await b._image_manifest(ARTIFACT)

    assert "scan01" not in manifest, (
        "after its marker expired and before any sweep, an abandoned oversized "
        f"upload is a dataset image again: {manifest}"
    )


@pytest.mark.asyncio
async def test_the_MINT_SWEEPS_FIRST_so_the_same_name_can_be_retried(monkeypatch):
    """Why the sweep moved to the START of the mint.

    At the end of the mint it was unreachable for the one name that mattered:
    the no-overwrite check answers `already` for the abandoned object and
    returns before any sweep. Sweeping first deletes the abandoned file, so the
    check sees a free name -- and the withholding above can therefore last until
    the object is really gone without ever contradicting that answer.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    first = await b.upload_local_image(ARTIFACT, "scan01.png")
    store.put(first["put_url"], _png(10000))
    _age_marker(store, first["mint_id"], path="images/scan01.png")

    second = await b.upload_local_image(ARTIFACT, "scan01.png")

    assert (
        second["already"] is False
    ), f"the same name is still blocked by the abandoned object: {second}"
    assert "images/scan01.png" in store.removed, store.removed
    assert second.get("mint_id") and second["mint_id"] != first["mint_id"]


@pytest.mark.asyncio
async def test_a_RETRIED_MINT_does_not_cost_the_finished_image(monkeypatch):
    """Round 5's data-loss case (a), and round 10's fix for it.

    A client whose first mint answer was lost asks again; two browser tabs on
    one filename do the same. When the marker was keyed by MINT ID that left
    two files, and consuming only the finishing call's own left the other open
    -- so the committed image was withheld at once and deleted by the sweep an
    hour later, which in Hypha a later commit makes permanent. Finding the other
    one meant listing a paginated directory, and four rounds went on bounding
    that read safely.

    Keyed by the OBJECT PATH there is no other one: the re-mint overwrites the
    single file. This asserts that as a COUNT -- one marker for the name, not
    two -- so it fails if the keying ever goes back to the mint id, and then
    asserts the consequence that mattered, which is that the finished image
    survives a sweep.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    first = await b.upload_local_image(ARTIFACT, "scan01.png")
    second = await b.upload_local_image(ARTIFACT, "scan01.png")
    assert second["already"] is False, second  # nothing PUT yet, so it is free
    assert second["mint_id"] != first["mint_id"]
    # TWO MINTS, ONE MARKER FILE. The count is the fix; everything below is
    # what the count buys.
    markers = [k for k in store.objects if k.startswith(f"{core.UPLOAD_MARKER_DIR}/")]
    assert markers == [_marker_for("scan01.png")], markers

    store.put(second["put_url"], _png(32))
    out = await b.upload_local_image(
        ARTIFACT, "scan01.png", done=True, mint_id=second["mint_id"]
    )
    assert out["committed"] is True

    # Right after the successful finish: listed, and no marker left behind.
    manifest = await b._image_manifest(ARTIFACT)
    assert "scan01" in manifest, f"the committed image is withheld: {manifest}"
    assert (
        _marker_for("scan01.png") not in store.objects
    ), "a marker for the finished image outlived the finish"

    # And it survives a sweep after any expiry window.
    await b._sweep_expired_markers(ARTIFACT)
    assert "images/scan01.png" not in store.removed, store.removed
    manifest = await b._image_manifest(ARTIFACT)
    assert "scan01" in manifest, f"the sweep deleted a finished image: {manifest}"


@pytest.mark.asyncio
async def test_a_SUPERSEDED_mint_id_cannot_finish_AND_IS_NOT_A_DEAD_END(monkeypatch):
    """The cost of the path keying, stated and bounded rather than discovered.

    Under the mint-id keying either of two markers for one path could be
    finished. Under the path keying a re-mint REPLACES the record, so a finish
    presenting the earlier mint id is refused. That is the right answer -- both
    URLs write the same key, so the bytes a stale finish would publish are
    whichever PUT landed last and nothing in the request says which -- but it
    would be a bad answer if it stranded the client, which is what the
    round-9 dead end was. So: the refusal touches NOTHING, the object stays
    withheld because the marker is still there, and the retry the error names
    actually works.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    first = await b.upload_local_image(ARTIFACT, "scan01.png")
    second = await b.upload_local_image(ARTIFACT, "scan01.png")
    store.put(first["put_url"], _png(32))  # the first tab PUT its bytes

    with pytest.raises(ValueError, match="cannot be completed"):
        await b.upload_local_image(
            ARTIFACT, "scan01.png", done=True, mint_id=first["mint_id"]
        )

    assert store.committed == 0, "a superseded mint committed"
    assert store.removed == [], f"a refused finish removed something: {store.removed}"
    assert _marker_for("scan01.png") in store.objects, "the refusal ate the marker"
    manifest = await b._image_manifest(ARTIFACT)
    assert (
        "scan01" not in manifest
    ), f"the uncommitted object is listed as an image: {manifest}"

    # NOT a dead end: the live mint id still finishes, and so does a fresh one.
    out = await b.upload_local_image(
        ARTIFACT, "scan01.png", done=True, mint_id=second["mint_id"]
    )
    assert out["committed"] is True, out


@pytest.mark.asyncio
async def test_a_finish_that_CANNOT_clear_the_record_REFUSES_to_commit(monkeypatch):
    """Round 5's data-loss case (b): the marker removal used to be best-effort.

    The commit went ahead regardless, and the leftover marker had the sweep
    delete the published image later. Refusing leaves the object staged and
    withheld -- the next mint's sweep clears it and the client can retry -- so
    the cost of the safe direction is an error on an upload that was otherwise
    fine, which is the right way round.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    mint = await b.upload_local_image(ARTIFACT, "scan01.png")
    store.put(mint["put_url"], _png(32))

    real_remove = store.remove_file

    async def refuse_markers(artifact_id=None, file_path=None):
        if isinstance(file_path, str) and file_path.startswith(
            f"{core.UPLOAD_MARKER_DIR}/"
        ):
            raise RuntimeError("the store refused to remove the marker")
        return await real_remove(artifact_id=artifact_id, file_path=file_path)

    monkeypatch.setattr(store, "remove_file", refuse_markers)

    with pytest.raises(RuntimeError, match="NOT published"):
        await b.upload_local_image(
            ARTIFACT, "scan01.png", done=True, mint_id=mint["mint_id"]
        )

    assert store.committed == 0, "it committed despite the record surviving"
    manifest = await b._image_manifest(ARTIFACT)
    assert (
        "scan01" not in manifest
    ), f"the unpublished object is listed as an image anyway: {manifest}"


@pytest.mark.asyncio
async def test_a_NORMAL_finish_still_commits_and_appears(monkeypatch):
    """THE CONTROL for the two refusals above. A finish that refused whenever a
    marker existed, or committed nothing at all, would satisfy them both."""
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    mint = await b.upload_local_image(ARTIFACT, "scan01.png")
    store.put(mint["put_url"], _png(32))
    out = await b.upload_local_image(
        ARTIFACT, "scan01.png", done=True, mint_id=mint["mint_id"]
    )

    assert out["committed"] is True
    assert store.committed == 1, store.committed
    manifest = await b._image_manifest(ARTIFACT)
    assert "scan01" in manifest, manifest
    assert "images/scan01.png" not in store.removed


@pytest.mark.asyncio
async def test_the_marker_remover_REFUSES_a_path_outside_the_marker_directory(
    monkeypatch,
):
    """The one deletion that takes a path it was HANDED rather than one it built
    from a validated mint id, so it checks where that path points.

    Both directions, because a remover that refused everything would satisfy the
    first assertion and break the sweep and the finish together.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    store.publish("images/already.png", _png(16))

    assert await b._remove_marker_file(ARTIFACT, "images/already.png") is False
    assert "images/already.png" in store.objects, "it removed a dataset image"
    assert "images/already.png" not in store.removed

    marker_path = f"{core.UPLOAD_MARKER_DIR}/{'a' * 32}.json"
    store.objects[marker_path] = b"{}"
    assert await b._remove_marker_file(ARTIFACT, marker_path) is True
    assert marker_path in store.removed


@pytest.mark.asyncio
async def test_finishing_one_upload_leaves_ANOTHER_name_s_marker_alone(monkeypatch):
    """The finish consumes every marker for ITS path and no others.

    The paired control for the retried-mint fix: widening the match from "this
    path" to "any marker" would satisfy that test and silently cancel an upload
    in flight for a different image, whose client would then be refused at its
    own finish with nothing saying why.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    other = await b.upload_local_image(ARTIFACT, "other99.png")
    mine = await b.upload_local_image(ARTIFACT, "scan01.png")
    store.put(mine["put_url"], _png(32))

    out = await b.upload_local_image(
        ARTIFACT, "scan01.png", done=True, mint_id=mine["mint_id"]
    )
    assert out["committed"] is True

    other_marker = _marker_for("other99.png")
    assert other_marker in store.objects, (
        "finishing one image consumed the marker of an upload still in flight "
        f"for another name: {store.removed}"
    )
    assert other_marker not in store.removed
    # And the other upload can still be finished on its own marker.
    store.put(other["put_url"], _png(16))
    done = await b.upload_local_image(
        ARTIFACT, "other99.png", done=True, mint_id=other["mint_id"]
    )
    assert done["committed"] is True


# ── round 6: the cap on the marker READ, and what it hid ─────────────────────
#
# The listing read used to stop at UPLOAD_MARKER_SWEEP_LIMIT entries, and two
# correctness decisions were computed from it. A folder upload that mints
# everything before PUTting anything puts markers past that cap.


@pytest.mark.asyncio
async def test_a_FOLDER_UPLOAD_past_the_cap_keeps_its_finished_image(monkeypatch):
    """Round 6's probe. 60 markers, finish the one whose marker sorts LAST.

    Measured at the refused head: the finish consumed only the markers in the
    first 50 listing entries, so this upload's OWN marker was never seen, the
    fail-closed check passed on a list that did not contain it, and the sweep
    deleted the finished image afterwards.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    mints = {}
    for i in range(60):
        name = f"img{i:02d}.png"
        mints[name] = await b.upload_local_image(ARTIFACT, name)

    # The one whose MARKER FILENAME sorts last, which is where the cap bites.
    last = max(mints, key=_marker_for)
    store.put(mints[last]["put_url"], _png(32))

    out = await b.upload_local_image(
        ARTIFACT, last, done=True, mint_id=mints[last]["mint_id"]
    )
    assert out["committed"] is True

    own = _marker_for(last)
    assert own not in store.objects, (
        f"the finishing upload's own marker survived the finish ({own}), so the "
        "sweep will delete the image it belongs to"
    )

    stem = last.rsplit(".", 1)[0]
    manifest = await b._image_manifest(ARTIFACT)
    assert stem in manifest, f"the committed image is not listed: {stem}"

    # And it survives sweeping. Several passes, because the sweep caps how much
    # it REMOVES per call and 59 markers remain.
    for _ in range(4):
        await b._sweep_expired_markers(ARTIFACT)
    assert (
        f"images/{last}" not in store.removed
    ), f"the sweep deleted a finished image: {store.removed[:5]}"
    manifest = await b._image_manifest(ARTIFACT)
    assert stem in manifest, f"the finished image is gone from the list: {stem}"


@pytest.mark.asyncio
async def test_an_unfinished_upload_PAST_THE_CAP_is_still_withheld(monkeypatch):
    """The other half: the withholding was computed from the same capped list.

    With more than the cap in flight, an unfinished object past it was a dataset
    image again -- the size bypass, reachable by minting 51 names.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    mints = {}
    for i in range(60):
        name = f"img{i:02d}.png"
        mints[name] = await b.upload_local_image(ARTIFACT, name)

    last = max(mints, key=_marker_for)
    store.put(mints[last]["put_url"], _png(10000))  # 156x the ceiling, no finish

    manifest = await b._image_manifest(ARTIFACT)

    stem = last.rsplit(".", 1)[0]
    assert stem not in manifest, (
        "an unfinished oversized upload whose marker lies past the old read cap "
        f"is listed as a dataset image: {stem}"
    )


@pytest.mark.asyncio
async def test_the_sweep_still_CAPS_WHAT_IT_REMOVES_per_call(monkeypatch):
    """The cap is kept where it belongs: on the work a caller can inflate.

    THE CONTROL for the two tests above -- they are satisfied by removing every
    bound, and an unbounded sweep is a mint that can be made to do arbitrarily
    many deletions.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    # Mint everything FIRST, then age it. Ageing inside the loop would have the
    # NEXT mint's own sweep remove each marker as it goes -- which it did on the
    # first run of this test, leaving 1 marker for the explicit sweep to find
    # and making the cap look broken when it was the fixture that was wrong.
    mints = [await b.upload_local_image(ARTIFACT, f"img{i:02d}.png") for i in range(60)]
    for i, mint in enumerate(mints):
        _age_marker(store, mint["mint_id"], path=f"images/img{i:02d}.png")

    swept = await b._sweep_expired_markers(ARTIFACT)

    assert swept == core.UPLOAD_MARKER_SWEEP_LIMIT, (
        f"one sweep removed {swept} markers; the per-call cap is "
        f"{core.UPLOAD_MARKER_SWEEP_LIMIT}"
    )
    assert swept < 60, "the sweep is unbounded"


# ── round 7: the server's own page is a cap too ──────────────────────────────


@pytest.mark.asyncio
async def test_the_marker_read_FOLLOWS_the_servers_pagination(monkeypatch):
    """Round 7's probe. `list_files` is paginated; one call is a PAGE.

    With the double's page at 50 and 60 markers in flight, everything past the
    first page used to be invisible to the withholding -- the size bypass
    again, now reachable by minting one more than the server's page rather than
    one more than the app's own former cap.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    assert store.page_size == 50, "the double must page for this test to mean anything"

    mints = {}
    for i in range(60):
        name = f"img{i:02d}.png"
        mints[name] = await b.upload_local_image(ARTIFACT, name)

    last = max(mints, key=_marker_for)
    store.put(mints[last]["put_url"], _png(10000))  # 156x the ceiling, no finish

    pairs, complete = await b._marker_pairs(ARTIFACT)
    assert complete is True, "the marker listing was not read in full"
    manifest = await b._image_manifest(ARTIFACT)

    assert (
        len(pairs) == 60
    ), f"the marker read stopped at a page boundary: {len(pairs)} of 60 seen"
    stem = last.rsplit(".", 1)[0]
    assert stem not in manifest, (
        "an unfinished oversized upload whose marker lies past the server's "
        f"first page is listed as a dataset image: {stem}"
    )
    # And it really did page, rather than the double handing over everything.
    marker_reads = [c for c in store.listing_calls if c[0] == core.UPLOAD_MARKER_DIR]
    assert (
        len(marker_reads) >= 2
    ), f"only one page was requested, so nothing was paged: {marker_reads}"


@pytest.mark.asyncio
async def test_a_page_SHORTER_than_asked_for_does_not_end_the_read(monkeypatch):
    """The termination rule, asserted directly, because getting it wrong is how
    the first attempt at this fix failed.

    The loop asks for the largest page the server documents and the double caps
    it at 50. If "shorter than I asked for" ended the read, every directory
    would be truncated to one page -- which is what happened, and what the test
    above caught.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    for i in range(60):
        await b.upload_local_image(ARTIFACT, f"img{i:02d}.png")

    entries, complete = await b._list_files_all(ARTIFACT, core.UPLOAD_MARKER_DIR)

    assert complete is True, "a short page was reported as incomplete"
    assert len(entries) == 60, len(entries)
    asks = [c for c in store.listing_calls if c[0] == core.UPLOAD_MARKER_DIR]
    # c = (dir_path, offset, ASKED, SERVED). The ask must exceed the server's
    # page, and the server must have served less -- that is the shape of the
    # answer the loop has to keep reading through.
    assert any(
        c[2] > store.page_size for c in asks
    ), f"the loop never asked for more than the server's page: {asks[:3]}"
    assert any(
        c[2] > c[3] for c in asks
    ), f"no call was served less than it asked for: {asks[:3]}"


@pytest.mark.asyncio
async def test_a_server_that_IGNORES_OFFSET_terminates_the_read(monkeypatch):
    """The pathological case, because an unbounded loop is worse than a short
    read. A page that repeats what has already been seen ends the loop, and the
    broker logs that the answer may be incomplete rather than claiming it is
    whole."""
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    for i in range(60):
        await b.upload_local_image(ARTIFACT, f"img{i:02d}.png")

    async def ignores_offset(_a, dir_path=None, stage=True, limit=None, offset=0):
        return store.listing(dir_path, stage=stage, limit=limit, offset=0)

    monkeypatch.setattr(store, "list_files", ignores_offset)

    entries, complete = await b._list_files_all(ARTIFACT, core.UPLOAD_MARKER_DIR)

    assert (
        len(entries) == store.page_size
    ), f"expected to stop after the repeated page: {len(entries)}"
    # And it REPORTS the shortfall. A repeated page means the server is ignoring
    # `offset`, so the answer may be short -- claiming it complete is the silent
    # partial the finish now refuses on.
    assert complete is False, "a repeated page was reported as a complete read"


@pytest.mark.asyncio
async def test_a_finish_DOES_NOT_READ_the_marker_listing_AT_ALL(monkeypatch):
    """Round 10's fix, asserted at its strongest: the scan is GONE, not bounded.

    Four rounds tried to make the finish's `.uploads/` scan safe. Every bound on
    a paginated read was wrong in one direction or the other -- short enough to
    terminate against a server that ignores `offset` meant refusing valid
    uploads once staged removals sat in the directory; loose enough not to
    refuse meant it might not terminate. A cap on a read a correctness decision
    consumes is a silent wrong answer, and no constant fixes that.

    So this does not test a better bound. It makes listing `.uploads/`
    IMPOSSIBLE -- every call raises -- and asserts the finish still commits.
    Nothing about page sizes, orderings, flags or counts can affect a read that
    does not happen. The marker is still consumed: it is found by computing
    `.uploads/<sha256(path)>.json` from the path being finished.

    It fails if the finish reads that listing for any reason, which is exactly
    the regression worth a test.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    # 60 markers and a tiny page size, so the OLD code paged here several times.
    for i in range(60):
        await b.upload_local_image(ARTIFACT, f"other{i:02d}.png")
    mint = await b.upload_local_image(ARTIFACT, "scan01.png")
    store.put(mint["put_url"], _png(32))

    real = store.list_files
    marker_listings = []

    async def refuses_to_list_the_markers(
        _a, dir_path=None, stage=True, limit=None, offset=0
    ):
        if dir_path == core.UPLOAD_MARKER_DIR:
            marker_listings.append((dir_path, offset))
            raise RuntimeError("the store will not list the marker directory")
        return await real(
            _a, dir_path=dir_path, stage=stage, limit=limit, offset=offset
        )

    monkeypatch.setattr(store, "list_files", refuses_to_list_the_markers)

    out = await b.upload_local_image(
        ARTIFACT, "scan01.png", done=True, mint_id=mint["mint_id"]
    )

    assert out["committed"] is True, out
    assert store.committed == 1, store.committed
    assert marker_listings == [], (
        "the finish listed the marker directory, so its correctness still "
        f"depends on a paginated read: {marker_listings}"
    )
    # The marker was still consumed -- found by name, not by scanning.
    assert _marker_for("scan01.png") in store.removed, store.removed
    assert _marker_for("scan01.png") not in store.objects


@pytest.mark.asyncio
async def test_a_finish_COMMITS_with_MANY_flagged_markers_before_the_last_live_one(
    monkeypatch,
):
    """Round 10's probe, as the test that made the previous head RED.

    At F >= the no-progress bound, with the flagged markers placed just before
    the last live one, the finish must commit. Under Hypha's ordering each page
    that serves only already-seen names advances the offset by the number of
    UNFLAGGED entries it served, so a run of flagged markers in the tail
    produced one stall per flagged marker -- more stalls than any fixed bound
    allows. Measured at 14 of 20 trials refused with 30 live and 50 flagged.

    The names are chosen so the flagged markers sort immediately before the live
    one being finished, which is the placement that made it worst. The finish
    reads no listing now, so this passes structurally rather than by being under
    a threshold -- and it is kept because it is the measurement that refused.
    """
    store = _RawOffsetStore()
    store.page_size = 2  # tiny, so every flagged marker is its own stall
    b = _store_broker(monkeypatch, store, ceiling=64)

    # Flag more markers than the no-progress bound tolerates, by a wide margin.
    flagged_count = bk._NO_PROGRESS_PAGES * 3
    assert flagged_count >= bk._NO_PROGRESS_PAGES, flagged_count

    live = await b.upload_local_image(ARTIFACT, "zz-live.png")
    store.put(live["put_url"], _png(32))
    live_marker_name = _marker_for("zz-live.png").split("/")[-1]

    # Mint names until enough markers sort BEFORE the live one, then flag them:
    # a marker is `_remove`-flagged whenever another route commits while an
    # upload is open, and annotation saves, set_role and materialise all commit.
    flagged = []
    i = 0
    while len(flagged) < flagged_count:
        nm = f"f{i:03d}.png"
        await b.upload_local_image(ARTIFACT, nm)
        marker_name = _marker_for(nm).split("/")[-1]
        if marker_name < live_marker_name:
            flagged.append(marker_name)
        i += 1
        assert i < 2000, "could not place enough markers before the live one"
    for marker_name in flagged:
        store.flagged.add(marker_name)

    out = await b.upload_local_image(
        ARTIFACT, "zz-live.png", done=True, mint_id=live["mint_id"]
    )

    assert out["committed"] is True, (
        f"{len(flagged)} flagged markers before the live one refused a valid "
        f"upload: {out}"
    )
    assert store.committed == 1, store.committed


@pytest.mark.asyncio
async def test_the_marker_read_REPORTS_a_partial_page_rather_than_hiding_it(
    monkeypatch,
):
    """The flag itself, both ways. A read that always said `False` would satisfy
    the refusal test above and make every finish refuse."""
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    for i in range(60):
        await b.upload_local_image(ARTIFACT, f"img{i:02d}.png")

    pairs, complete = await b._marker_pairs(ARTIFACT)
    assert complete is True and len(pairs) == 60, (complete, len(pairs))

    real = store.list_files

    async def breaks_after_the_first_page(
        _a, dir_path=None, stage=True, limit=None, offset=0
    ):
        if int(offset or 0) > 0:
            raise RuntimeError("the store refused a later page")
        return await real(
            _a, dir_path=dir_path, stage=stage, limit=limit, offset=offset
        )

    monkeypatch.setattr(store, "list_files", breaks_after_the_first_page)

    pairs, complete = await b._marker_pairs(ARTIFACT)
    assert complete is False, "a partial read was reported as complete"
    assert len(pairs) == store.page_size, len(pairs)


@pytest.mark.asyncio
async def test_the_WITHHOLDING_still_answers_on_a_partial_read(monkeypatch):
    """The fail-open half, kept deliberately and asserted so it is not quietly
    turned into a refusal: an unreadable `.uploads/` must not present the
    dataset as having no images at all."""
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    store.publish("images/already.png", _png(16))
    for i in range(60):
        await b.upload_local_image(ARTIFACT, f"img{i:02d}.png")

    real = store.list_files

    async def breaks_after_the_first_page(
        _a, dir_path=None, stage=True, limit=None, offset=0
    ):
        if int(offset or 0) > 0:
            raise RuntimeError("the store refused a later page")
        return await real(
            _a, dir_path=dir_path, stage=stage, limit=limit, offset=offset
        )

    monkeypatch.setattr(store, "list_files", breaks_after_the_first_page)

    withheld = await b._withheld_upload_paths(ARTIFACT)
    manifest = await b._image_manifest(ARTIFACT)

    assert len(withheld) == store.page_size, len(withheld)
    assert (
        "already" in manifest
    ), f"a partial marker read hid an image that was already committed: {manifest}"


# ── round 9: Hypha applies offset to the RAW listing, then filters ───────────
#
# `offset`/`max_length` cut the raw S3 listing; the stage filter that hides
# `_remove`-flagged entries runs AFTERWARDS. So advancing by the SERVED count
# makes the next page start that many raw entries early and re-serve names
# already seen. That OVERLAP is ordinary -- a marker is flagged whenever any
# other route commits while an upload is open, and annotation saves commit --
# and reading it as "incomplete" refused valid uploads.


class _RawOffsetStore(_Store):
    """A store with Hypha's own ordering: slice the RAW list, then filter.

    `flagged` holds marker names that are staged for removal: still present in
    the raw listing, hidden from the served page. Modelled from the Hypha source
    the review cited rather than from my own guess at the behaviour.
    """

    def __init__(self):
        super().__init__()
        self.flagged: set = set()

    def listing(self, dir_path: str, stage: bool = True, limit=None, offset=0):
        source = {**self.published, **self.objects}
        raw = []
        for path, body in sorted(source.items()):
            head, _, nm = path.rpartition("/")
            if head != dir_path:
                continue
            raw.append({"name": nm, "type": "file", "size": len(body)})
        start = int(offset or 0)
        asked = int(limit) if limit else self.page_size
        size = min(asked, self.page_size)
        self.listing_calls.append((dir_path, start, asked, size))
        page = raw[start : start + size]  # the RAW slice comes first...
        return [e for e in page if e["name"] not in self.flagged]  # ...then the filter


@pytest.mark.asyncio
async def test_a_finish_COMMITS_through_an_overlapping_page(monkeypatch):
    """Round 9's probe: a staged-removed marker must not refuse a good upload.

    Measured at the refused head: with `a.png`'s mint id sorting before another
    open marker's, every trial refused with "could not be read in full" and
    committed nothing -- and because the own marker had already been removed,
    the uncommitted object was then LISTED as an image while the error told the
    caller to retry, which answered `already: True`.
    """
    store = _RawOffsetStore()
    store.page_size = 2  # tiny, so two markers already span pages
    b = _store_broker(monkeypatch, store, ceiling=64)

    a_mint = await b.upload_local_image(ARTIFACT, "a.png")
    await b.upload_local_image(ARTIFACT, "b.png")
    store.put(a_mint["put_url"], _png(32))
    # a.png's marker is COMMITTED and then flagged for removal, which is what
    # any other route's commit during an open upload produces.
    store.flagged.add(_marker_for("a.png").split("/")[-1])

    out = await b.upload_local_image(
        ARTIFACT, "a.png", done=True, mint_id=a_mint["mint_id"]
    )

    assert out["committed"] is True, f"an overlapping page refused a good upload: {out}"
    assert store.committed == 1, store.committed


@pytest.mark.asyncio
async def test_an_overlapping_page_is_NOT_reported_incomplete(monkeypatch):
    """The read itself, so the distinction is pinned where it is made rather
    than only through the route."""
    store = _RawOffsetStore()
    store.page_size = 2
    b = _store_broker(monkeypatch, store, ceiling=64)
    mints = [await b.upload_local_image(ARTIFACT, f"m{i}.png") for i in range(6)]
    store.flagged.add(_marker_for("m0.png").split("/")[-1])

    entries, complete = await b._list_files_all(ARTIFACT, core.UPLOAD_MARKER_DIR)

    assert complete is True, "an ordinary overlap was reported as incomplete"
    # Five of the six are served: the flagged one is hidden by the filter, and
    # every unflagged marker is reached despite the overlap.
    assert len(entries) == 5, [e["name"] for e in entries]


@pytest.mark.asyncio
async def test_a_refusal_now_leaves_the_upload_UNCHANGED(monkeypatch):
    """The dead end, closed, re-aimed at the ONE fail-closed path that is left.

    Round 9 measured it on the incomplete-listing refusal: the own marker had
    already been removed when the refusal fired, so the uncommitted object was
    no longer withheld and appeared as an image, while the error told the caller
    to retry with `done=false` -- which answered `already: True`. A dead end,
    and a false message.

    That refusal is gone with the listing. The remaining one is the marker
    removal failing, and it has to keep the same property: nothing committed,
    the marker still there so the object stays withheld, and the retry the
    message prescribes actually available. The store here refuses to remove the
    marker and nothing else.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    for i in range(60):
        await b.upload_local_image(ARTIFACT, f"img{i:02d}.png")
    mint = await b.upload_local_image(ARTIFACT, "scan01.png")
    store.put(mint["put_url"], _png(32))

    real_remove = store.remove_file

    async def refuses_to_remove_the_marker(artifact_id=None, file_path=None):
        if isinstance(file_path, str) and file_path.startswith(
            f"{core.UPLOAD_MARKER_DIR}/"
        ):
            raise RuntimeError("the store refused to remove the marker")
        return await real_remove(artifact_id=artifact_id, file_path=file_path)

    monkeypatch.setattr(store, "remove_file", refuses_to_remove_the_marker)

    with pytest.raises(RuntimeError, match="NOT published"):
        await b.upload_local_image(
            ARTIFACT, "scan01.png", done=True, mint_id=mint["mint_id"]
        )

    assert store.committed == 0, "it committed without clearing the record"
    own = _marker_for("scan01.png")
    assert own in store.objects, "the refusal removed the marker anyway"
    manifest = await b._image_manifest(ARTIFACT)
    assert (
        "scan01" not in manifest
    ), f"the uncommitted object is listed as an image: {manifest}"
    # And the retry the message prescribes is really available: the marker is
    # still there, so the path is still withheld, so a re-mint is not told the
    # name already exists.
    monkeypatch.setattr(store, "remove_file", real_remove)
    again = await b.upload_local_image(ARTIFACT, "scan01.png")
    assert again["already"] is False, f"the retry the error names is blocked: {again}"


@pytest.mark.asyncio
async def test_an_UNPAGINATED_server_does_not_over_claim_completeness(monkeypatch):
    """The fallback branch, for a store that rejects limit/offset outright.

    It falls back to one unpaginated call, and that call's result may itself be
    a page. A review caught the previous version returning `complete=True`
    there while the log beside it said "may be incomplete" -- the log and the
    return value disagreed, and the return value is the one callers act on.

    `_SERVER_DEFAULT_PAGE` is lowered for the test rather than creating a
    thousand markers: the rule under test is "a result at or above the server's
    default page may have been truncated", and that rule is what the constant
    expresses. Both directions are asserted, so a branch that always reported
    incompleteness would fail here.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    async def rejects_pagination(_a, dir_path=None, stage=True, limit=None, offset=0):
        if limit is not None or offset:
            raise TypeError("list_files() got an unexpected keyword argument")
        return store.listing(dir_path, stage=stage)

    monkeypatch.setattr(store, "list_files", rejects_pagination)
    monkeypatch.setattr(bk, "_SERVER_DEFAULT_PAGE", 4, raising=False)

    for i in range(3):
        await b.upload_local_image(ARTIFACT, f"few{i}.png")
    entries, complete = await b._list_files_all(ARTIFACT, core.UPLOAD_MARKER_DIR)
    assert len(entries) == 3 and complete is True, (len(entries), complete)

    for i in range(3, 9):
        await b.upload_local_image(ARTIFACT, f"few{i}.png")
    entries, complete = await b._list_files_all(ARTIFACT, core.UPLOAD_MARKER_DIR)
    assert len(entries) == 9, len(entries)
    assert (
        complete is False
    ), "a single page at or above the server's default page was called complete"


# ── round 11: the manifest WRITERS must not record an unfinished upload ──────
#
# The read reconciliation withheld an open upload; the two writers did not, and
# they COMMIT. So one manager action during somebody's upload turned that
# upload into a permanent dataset image -- size unchecked, signature unchecked
# -- and the expiry sweep then deleted the object and left the entry dangling.
# The rule now lives in one place (`merge_image_manifest`, with a REQUIRED
# `withheld_paths`) and governs both directions: never ADD a withheld name,
# never DROP a stored entry on a withheld path.


def _stored_manifest(store):
    """The manifest as it is stored, not as a read reconstructs it."""
    raw = store.objects.get(core.IMAGE_MANIFEST_PATH)
    return core.parse_image_manifest(json.loads(raw.decode("utf-8"))) if raw else {}


@pytest.mark.asyncio
async def test_SYNC_does_not_record_an_UNFINISHED_upload_as_an_image(monkeypatch):
    """The measured hole, as the test that makes it RED.

    Mint a name, PUT an oversized body, never finish. The object is in
    `images/` and the listing reports it, but nothing has checked it. A manager
    then runs `sync_image_manifest`, whose own docstring tells callers to run it
    "after adding or removing image files" -- so this is an ordinary action, not
    a contrived one. Before the fix that wrote the stem into the stored manifest
    and committed it.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    mint = await b.upload_local_image(ARTIFACT, "big.png")
    store.put(mint["put_url"], _png(10000))  # 156x the ceiling, never finished

    before = await b._image_manifest(ARTIFACT)
    assert "big" not in before, f"the unfinished upload was listed: {before}"

    await b.sync_image_manifest(ARTIFACT, context=None)

    assert "big" not in _stored_manifest(store), (
        "sync_image_manifest wrote an unfinished upload into the STORED "
        f"manifest: {_stored_manifest(store)}"
    )
    after = await b._image_manifest(ARTIFACT)
    assert "big" not in after, f"the unfinished upload became an image: {after}"


@pytest.mark.asyncio
async def test_SYNC_still_records_a_FINISHED_upload(monkeypatch):
    """The control, and it is not optional.

    Without it, a `sync_image_manifest` that records NOTHING -- or that was
    never called -- satisfies the test above. This asserts the same call writes
    the stem once the upload is legitimately complete.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    mint = await b.upload_local_image(ARTIFACT, "ok.png")
    store.put(mint["put_url"], _png(32))
    out = await b.upload_local_image(
        ARTIFACT, "ok.png", done=True, mint_id=mint["mint_id"]
    )
    assert out["committed"] is True, out

    await b.sync_image_manifest(ARTIFACT, context=None)

    assert "ok" in _stored_manifest(store), (
        "sync_image_manifest recorded nothing at all, so the refusal above "
        f"proves nothing: {_stored_manifest(store)}"
    )


def test_the_PRUNE_treats_a_withheld_path_as_PRESENT():
    """The opposite direction, and the reason the withheld set goes INTO the
    merge instead of being filtered out of the listing beforehand.

    A pre-filtered list makes a withheld path look ABSENT to the pruning loop,
    so the fix for the writer would have become a different data loss: a stored
    entry deleted because an upload to the same path was in flight. The
    docstring on the old filter warned about exactly this.

    **Asserted on the function, not through a route, and that is deliberate.**
    I tried to reach it through `sync_image_manifest` and measured that I
    cannot: a stored entry on a withheld path is unreachable today, because the
    mint refuses a stem already in the merged manifest, and both writers now
    refuse to add a withheld name. So this is a contract on
    `merge_image_manifest` rather than a repair of a live path -- and saying
    that plainly is better than a route-level test that passes for a reason
    unrelated to the clause. The guard stays because the contract is what the
    next caller will rely on; an unreachable state is not the same as an
    impossible one.
    """
    stored = {"keep": core.local_image_entry("keep.png")}

    kept, _ = core.merge_image_manifest(
        stored,
        [],  # the listing does NOT report it
        prune=True,
        withheld_paths={"images/keep.png"},
    )
    assert "keep" in kept, f"a withheld path was pruned: {kept}"

    # The control: with nothing withheld, the same call DOES prune it. Without
    # this, a merge that never prunes anything would satisfy the line above.
    dropped, _ = core.merge_image_manifest(
        stored, [], prune=True, withheld_paths=NOTHING_WITHHELD
    )
    assert "keep" not in dropped, f"the prune stopped working altogether: {dropped}"


@pytest.mark.asyncio
async def test_the_REMOTE_IMPORT_does_not_record_an_unfinished_upload(monkeypatch):
    """The second writer, reached from a different route.

    `import_ngff_image` merges the listing and writes the manifest too, so an
    unfinished upload in flight was recorded as an image by somebody else's
    remote import. The probe is stubbed because this test is about the manifest
    write, not about OME-Zarr.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    async def _probe(root, multiscale_path, origin):
        return {
            "width": 64,
            "height": 48,
            "dtype": "uint8",
            "channels": ["C0"],
            "levels": 1,
            "content_hint": None,
            "dataset_paths": ["0"],
            "axis_roles": {"x": 1, "y": 0},
            "level0_shape": [48, 64],
            "axes": ["y", "x"],
        }

    monkeypatch.setattr(b, "_probe_ngff_source", _probe)

    mint = await b.upload_local_image(ARTIFACT, "big.png")
    store.put(mint["put_url"], _png(10000))  # never finished

    await b.import_ngff_image(
        ARTIFACT, "https://example.invalid/s.zarr", stem="remote1", context=None
    )

    stored = _stored_manifest(store)
    assert "remote1" in stored, f"the import recorded nothing: {stored}"
    assert (
        "big" not in stored
    ), f"a remote import swept an unfinished upload into the manifest: {stored}"


def test_EVERY_manifest_merge_in_the_broker_passes_the_withheld_set():
    """A structural pin, because the defect was an OMISSION at a call site.

    Four call sites merge the manifest -- two reads and two writers -- and the
    two writers were the ones that forgot. The argument is required at the
    function, so an omission is a `TypeError` rather than silence; this asserts
    the stronger thing, that no call site passes a literal empty set to make
    that `TypeError` go away. Over the AST, so a call spanning several lines is
    read as one call rather than as whatever a text search happened to match.
    """
    import ast
    import pathlib

    src = (pathlib.Path(__file__).parent / "broker.py").read_text()
    calls = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", None) == "merge_image_manifest"
    ]
    assert len(calls) == 4, f"expected 4 merge call sites, found {len(calls)}"
    for call in calls:
        kw = {k.arg: k for k in call.keywords}
        assert "withheld_paths" in kw, (
            f"broker.py:{call.lineno} merges the manifest without passing "
            "withheld_paths"
        )
        value = kw["withheld_paths"].value
        assert not (
            isinstance(value, ast.Call)
            and getattr(value.func, "id", None) == "frozenset"
            and not value.args
        ), (
            f"broker.py:{call.lineno} passes an empty frozenset, which "
            "satisfies the signature while withholding nothing"
        )
