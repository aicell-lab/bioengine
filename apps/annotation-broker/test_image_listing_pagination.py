"""The dataset's image list is a DIRECTORY, not the first page of one.

``list_files`` is paginated. Its ``limit`` defaults to **1000** and is capped at
**10000** — measured from the live artifact manager's own published schema,
``{"default": 1000, "minimum": 1, "maximum": 10000, "description": "Maximum
number of files to return. Range: 1-10000."}``, read off ``public/artifact-manager``
rather than out of a copy of Hypha's source, because the installed copy need not
be the version the app talks to.

So a single unpaginated call answers at most 1000 entries, and every image past
that was missing from the dataset's image list with no error anywhere: the page
looked complete and the extra images looked as though they had never been
uploaded. Four other call sites read the same listing to decide something, and
their consequences differ enough to be worth naming here, because three of them
are not "a short list":

* the mint's no-overwrite check got LOOSER — a name missing from the answer
  looks free, so above 1000 images it handed out an upload URL for a stem that
  already had an image, the PUT overwrote that object, and the marker recorded
  "not a pre-existing image", which is the whole warrant for the finish's
  discard. That loses an image AND the justification for deleting it.
* ``sync_image_manifest`` PRUNES: an image whose file sits on a later page looks
  like one whose file has gone, so its stored entry was dropped and committed.
* the NGFF import's check got looser the same way as the mint's, but its write
  is the stored manifest plus one entry, so the local file is shadowed rather
  than deleted.
* the finish's size check already failed CLOSED, and that is worth saying
  plainly because the obvious guess is the opposite. ``file_size_from_listing``
  answers ``None`` for a name it cannot find and ``oversized_image_refusal``
  refuses on ``None``, so finishing an upload into a dataset above 1000 images
  refused every time — wrongly, blaming the store for not reporting sizes, but
  never a way past the ceiling.

THE DOUBLE HERE MODELS THE DOCUMENTED SERVER, not the fix. ``limit=None`` caps
at 1000 and an explicit limit caps at ``min(limit, 10000)``, which is what the
schema says. That is what lets these tests carry a CONTROL showing the
unpaginated read losing an image against the same double — without it, a double
that answered everything in one call would pass both before and after the fix
and this file would be decoration.
"""

from __future__ import annotations

import time

import pytest

import broker_core as core
from test_broker_ngff import ROOT, bk
from test_broker_ngff import importer as importer  # a fixture, re-exported here

ARTIFACT = "ws/ds"

#: What the artifact manager documents. Named rather than inlined so a test
#: that means "the server's default page" cannot be read as an arbitrary 1000.
SERVER_DEFAULT_PAGE = 1000
SERVER_MAX_PAGE = 10000


class _PagingAm:
    """``list_files`` as the artifact manager documents it, and nothing else.

    ``fail_from_offset`` makes a later page fail, which is how a test asks for
    an INCOMPLETE read: that is the shape ``_list_files_all`` reports as
    ``complete=False``, and it is reachable in production from a transport
    error mid-read.
    """

    def __init__(self, rows, *, fail_from_offset=None):
        self.rows = list(rows)
        self.fail_from_offset = fail_from_offset
        self.calls = []

    async def list_files(self, _a, dir_path=None, stage=True, limit=None, offset=0):
        self.calls.append({"dir_path": dir_path, "limit": limit, "offset": offset})
        if dir_path != "images":
            return []
        start = int(offset or 0)
        if self.fail_from_offset is not None and start >= self.fail_from_offset:
            raise RuntimeError("the store stopped answering mid-read")
        size = (
            SERVER_DEFAULT_PAGE if limit is None else min(int(limit), SERVER_MAX_PAGE)
        )
        return self.rows[start : start + size]

    async def put_file(self, _a, file_path=None, expires_in=None):
        return f"https://signed.invalid/{file_path}"

    async def remove_file(self, artifact_id=None, file_path=None):
        self.removed = getattr(self, "removed", [])
        self.removed.append(file_path)


def _rows(n, *, first="scan"):
    return [
        {"name": f"{first}{i:05d}.png", "type": "file", "size": 16} for i in range(n)
    ]


def _broker(monkeypatch, *, rows, manifest=None, fail_from_offset=None, markers=None):
    """A real broker whose ONLY faked storage surface is `list_files`.

    `_list_files_safe` and `_list_files_all` are both the shipped methods here,
    deliberately: a fixture that stubbed them would be testing the stub's idea
    of pagination rather than the app's.
    """
    b = bk.AnnotationBroker()
    am = _PagingAm(rows, fail_from_offset=fail_from_offset)
    store = dict(markers or {})

    async def _read_json_file(_a, path, default=None):
        if path in store:
            return store[path]
        if path.startswith(f"{core.UPLOAD_MARKER_DIR}/"):
            return default
        if path == core.IMAGE_MANIFEST_PATH:
            return {
                "manifest_version": core.IMAGE_MANIFEST_VERSION,
                "images": manifest or {},
            }
        return default

    written = []

    async def _write_json_file(_a, path, data):
        written.append((path, data))
        store[path] = data

    async def _ensure_staged(_a, fn):
        return await fn()

    async def _commit(_a):
        pass

    async def _marker_listing(_a, dir_path, stage=True):
        if dir_path == core.UPLOAD_MARKER_DIR:
            return [
                {"name": p.split("/")[-1], "type": "file", "size": 1}
                for p in sorted(store)
                if p.startswith(f"{core.UPLOAD_MARKER_DIR}/")
            ]
        return []

    monkeypatch.setattr(b, "_canonical_id", lambda a: a)
    monkeypatch.setattr(
        b,
        "_require_role",
        lambda context, artifact_id, minimum: ({}, {"id": "u1"}, "manager"),
    )
    monkeypatch.setattr(b, "_am", am)
    monkeypatch.setattr(b, "_read_json_file", _read_json_file)
    monkeypatch.setattr(b, "_write_json_file", _write_json_file)
    monkeypatch.setattr(b, "_ensure_staged", _ensure_staged)
    monkeypatch.setattr(b, "_commit_tolerating_pending_uploads", _commit)
    # The MARKER directory is not what this file is about, and #0938 already
    # pages it. Served through `_list_files_safe`, which `_marker_pairs` uses,
    # so these tests say nothing about marker pagination either way.
    monkeypatch.setattr(b, "_list_files_safe", _marker_listing)
    monkeypatch.setattr(b, "_object_head", lambda *_a, **_k: _png())
    b._tam = am
    b._twritten = written
    b._tstore = store
    return b


async def _png():
    return core.PNG_SIGNATURE


# ── clause 1: the reconciled list is complete ─────────────────────────────


@pytest.mark.parametrize("n", [1, 2, SERVER_DEFAULT_PAGE, SERVER_DEFAULT_PAGE + 1])
async def test_the_image_list_holds_EVERY_image(monkeypatch, n):
    """1 is the control the row asked to keep, 1001 is the smallest case that
    shows the defect, and 1000 is the boundary where the old read was still
    accidentally right."""
    b = _broker(monkeypatch, rows=_rows(n))

    entries = await b._image_manifest(ARTIFACT)

    assert len(entries) == n, f"{len(entries)} of {n} images in the list"
    assert f"scan{n - 1:05d}" in entries, "the LAST image is the one that went missing"


async def test_an_UNPAGINATED_READ_LOSES_AN_IMAGE_against_this_very_double(
    monkeypatch,
):
    """THE CONTROL FOR THE WHOLE FILE. Without it, a double that answered
    everything in one call would pass the test above before and after the fix,
    and nothing here would be evidence. This asks the same double the way the
    old code asked it — one call, no limit — and loses exactly one image."""
    b = _broker(monkeypatch, rows=_rows(SERVER_DEFAULT_PAGE + 1))

    one_page = await b._am.list_files(ARTIFACT, dir_path="images")

    assert len(one_page) == SERVER_DEFAULT_PAGE
    assert len(core.image_names_from_listing(one_page)) == SERVER_DEFAULT_PAGE
    reconciled = await b._image_manifest(ARTIFACT)
    assert (
        len(reconciled) == SERVER_DEFAULT_PAGE + 1
    ), "the paged read must see the image the single call did not"


async def test_the_list_is_complete_ABOVE_the_servers_MAXIMUM_PAGE(monkeypatch):
    """A dataset larger than the biggest page the server will serve, so the
    PAGING LOOP itself is what makes the answer complete — not one generous
    limit. At 10001 entries a single call cannot be right however it is asked.
    """
    n = SERVER_MAX_PAGE + 1
    b = _broker(monkeypatch, rows=_rows(n))

    entries = await b._image_manifest(ARTIFACT)

    assert len(entries) == n
    # The offset advances by the number of entries RECEIVED, not by the limit
    # asked for — which is the only form that is right when the server caps the
    # page below the ask. 0, then 10000 after a full page, then 10001 after the
    # single remaining entry, then the empty page that ends the loop.
    offsets = [c["offset"] for c in b._tam.calls if c["dir_path"] == "images"]
    assert offsets == [0, SERVER_MAX_PAGE, n], offsets


async def test_the_list_is_complete_across_MANY_MORE_PAGES_than_the_stall_bound(
    monkeypatch,
):
    """The case a review had to find by hand, and the shape of test the rest of
    this file was missing.

    The paging loop tolerates up to ``_NO_PROGRESS_PAGES`` consecutive pages
    that add nothing new, because a filtered overlap is ordinary and an ignored
    ``offset`` is not. Every test above goes through at most two non-empty
    pages, so a counter that climbed on EVERY page -- which is what a duplicated
    merge loop made it do -- stayed under the bound and nothing noticed. Above
    the bound the read came back short again and called itself incomplete,
    which after this change makes every mint, import and finish into a dataset
    that size refuse permanently.

    The page is forced down to 1000 so the case is 20 pages rather than a
    dataset of 80000 fixtures: the quantity that matters is the number of
    non-empty PAGES, not the number of entries.
    """
    n = 20_000
    monkeypatch.setattr(bk, "_LIST_PAGE", SERVER_DEFAULT_PAGE)
    b = _broker(monkeypatch, rows=_rows(n))

    entries, complete = await b._list_files_all(ARTIFACT, "images")

    assert len(entries) == n, f"{len(entries)} of {n} entries"
    assert (
        complete is True
    ), "a server that paged correctly for 20 pages must not be called incomplete"
    pages = [c for c in b._tam.calls if c["dir_path"] == "images"]
    assert len(pages) == n // SERVER_DEFAULT_PAGE + 1, len(pages)


async def test_a_SERVER_THAT_IGNORES_OFFSET_is_still_called_INCOMPLETE(monkeypatch):
    """The control for the test above. The stall bound exists to catch a server
    that never advances, and a fix that simply stopped counting would satisfy
    the previous test while removing the only protection against an unreadable
    directory."""

    class _StuckAm(_PagingAm):
        async def list_files(self, _a, dir_path=None, stage=True, limit=None, offset=0):
            self.calls.append({"dir_path": dir_path, "limit": limit, "offset": offset})
            if dir_path != "images":
                return []
            size = (
                SERVER_DEFAULT_PAGE
                if limit is None
                else min(int(limit), SERVER_MAX_PAGE)
            )
            return self.rows[:size]  # offset IGNORED: always the first page

    # The page is forced below the row count so that "one page" and "all the
    # rows" are different numbers: with the default ask of 10000 against 5000
    # rows, a stuck server and a working one both return 5000 and the length
    # assertion below would prove nothing.
    monkeypatch.setattr(bk, "_LIST_PAGE", SERVER_DEFAULT_PAGE)
    b = _broker(monkeypatch, rows=_rows(5000))
    monkeypatch.setattr(b, "_am", _StuckAm(_rows(5000)))

    entries, complete = await b._list_files_all(ARTIFACT, "images")

    assert complete is False, "an offset that is ignored cannot be read past"
    assert (
        len(entries) == SERVER_DEFAULT_PAGE
    ), f"{len(entries)} entries: the loop must stop, not accumulate"


async def test_the_read_costs_TWO_list_files_calls_and_the_CACHE_costs_NONE(
    monkeypatch,
):
    """The documented cost, pinned. `_image_manifest` was explicitly "the
    annotate page's featherweight first call [which] must not grow two
    round-trips", and paging makes it two: one page of entries, then the empty
    page that ends the loop. That is the price of a correct list and it is paid
    on a cache MISS only. A change that doubles it again fails here."""
    b = _broker(monkeypatch, rows=_rows(5))

    await b._image_manifest(ARTIFACT)
    first = [c for c in b._tam.calls if c["dir_path"] == "images"]
    assert len(first) == 2, first

    await b._image_manifest(ARTIFACT)
    assert (
        len([c for c in b._tam.calls if c["dir_path"] == "images"]) == 2
    ), "the second read was served from the cache and must cost nothing"


# ── clause 2: no writer acts on a partial listing ─────────────────────────


async def test_the_PRUNING_WRITER_SKIPS_on_a_short_listing(monkeypatch):
    """Clause 2. `sync_image_manifest` drops a stored entry whose file it cannot
    see, so a partial listing makes it delete metadata for images that are
    simply on a later page — and it COMMITS. It must refuse to write at all."""
    manifest = {
        f"scan{i:05d}": {"path": f"images/scan{i:05d}.png", "state": "local"}
        for i in range(3)
    }
    b = _broker(
        monkeypatch,
        rows=_rows(SERVER_MAX_PAGE + 1),
        manifest=manifest,
        fail_from_offset=SERVER_MAX_PAGE,
    )

    out = await b.sync_image_manifest(ARTIFACT, context=None)

    assert out["changed"] is False
    assert out.get("skipped") == "the images directory could not be read in full"
    assert b._twritten == [], "nothing may be written on a partial listing"
    assert out["images"] == len(
        manifest
    ), "the stored manifest must be reported unchanged, not recomputed"


async def test_the_SKIP_NAMES_THE_RIGHT_STORE(monkeypatch):
    """Two different short reads skip this writer — the marker set and the image
    listing — and they send a reader to two different stores. A single combined
    condition would report the marker store for an image-listing failure."""
    b = _broker(
        monkeypatch,
        rows=_rows(SERVER_MAX_PAGE + 1),
        fail_from_offset=SERVER_MAX_PAGE,
    )

    out = await b.sync_image_manifest(ARTIFACT, context=None)

    assert out["skipped"] == "the images directory could not be read in full"
    assert "marker" not in out["skipped"]


async def test_the_PRUNING_WRITER_DOES_WRITE_when_the_listing_is_COMPLETE(
    monkeypatch,
):
    """The control for the two tests above. A writer that skipped
    unconditionally would satisfy them both and do nothing useful ever."""
    manifest = {"gone": {"path": "images/gone.png", "state": "local"}}
    b = _broker(monkeypatch, rows=_rows(3), manifest=manifest)

    out = await b.sync_image_manifest(ARTIFACT, context=None)

    assert "skipped" not in out
    assert out["changed"] is True
    assert [p for p, _ in b._twritten] == [core.IMAGE_MANIFEST_PATH]
    written = b._twritten[0][1]["images"]
    assert "gone" not in written, "the entry whose file is really absent is pruned"
    assert len(written) == 3


# ── the other three readers, each with the case its own consequence names ──


async def test_the_MINT_REFUSES_rather_than_handing_out_a_clobbering_URL(
    monkeypatch,
):
    """A short listing makes the no-overwrite check LOOSER, so the mint would
    hand out an upload URL for a stem that already has an image on a later
    page. Refusing is the only safe answer: `already: True` would be a claim
    nothing here can support, and it cannot be given at all for a stem the
    short answer never mentioned."""
    b = _broker(
        monkeypatch,
        rows=_rows(SERVER_MAX_PAGE + 1),
        fail_from_offset=SERVER_MAX_PAGE,
    )

    with pytest.raises(RuntimeError) as exc:
        await b.upload_local_image(ARTIFACT, "newname.png", context=None)

    assert "could not be listed in full" in str(exc.value)
    assert "no upload URL was handed out" in str(exc.value)


async def test_the_MINT_STILL_WORKS_when_the_listing_is_COMPLETE(monkeypatch):
    """The control. A mint that refused unconditionally would pass the test
    above and break every upload."""
    b = _broker(monkeypatch, rows=_rows(3))

    out = await b.upload_local_image(ARTIFACT, "newname.png", context=None)

    assert out["already"] is False
    assert out["path"] == "images/newname.png"
    assert out["mint_id"]


async def test_the_MINT_still_answers_ALREADY_for_an_image_on_a_LATER_PAGE(
    monkeypatch,
):
    """The case the defect was about, now that the read is complete: a stem that
    exists only beyond the server's default page must still be refused as
    taken. Before, it looked free."""
    rows = _rows(SERVER_DEFAULT_PAGE + 1)
    b = _broker(monkeypatch, rows=rows)

    last = rows[-1]["name"]
    out = await b.upload_local_image(ARTIFACT, last, context=None)

    assert out["already"] is True, out
    assert out["path"] == f"images/{last}"


async def test_the_IMPORT_REFUSES_on_a_short_listing(monkeypatch, importer):
    """The NGFF import's no-overwrite check gets looser the same way the mint's
    does. Its write is the stored manifest plus this one entry, so a local file
    on a later page is shadowed rather than deleted — the stem resolves to a
    remote source and the object at `images/<stem><ext>` becomes unreachable
    through the manifest while still occupying the path.

    Uses the ngff fixture rather than this file's, because `import_ngff_image`
    probes a remote store before it ever reads the listing, and a fixture that
    faked the probe here would be a second, divergent copy of that one.
    """

    async def _short(_a, _dir_path, stage=True):
        return [], False

    monkeypatch.setattr(importer, "_list_files_all", _short)

    with pytest.raises(RuntimeError) as exc:
        await importer.import_ngff_image("ws/ds", ROOT, context=None)

    assert "could not be listed in full" in str(exc.value)
    assert "NOTHING was imported" in str(exc.value)


async def test_the_IMPORT_STILL_WORKS_when_the_listing_is_COMPLETE(importer):
    """The control for the test above."""
    out = await importer.import_ngff_image("ws/ds", ROOT, context=None)
    assert out["stem"] == "sample"
    assert out["state"] == "remote"


async def test_the_FINISH_REFUSES_AND_DISCARDS_on_a_short_listing(monkeypatch):
    """The finish already failed closed, but through the size check, with a
    message blaming the store's size reporting. It now names the real condition
    — and it must still DISCARD, because an object left at `images/<stem><ext>`
    becomes a dataset image anyway when any route next commits the stage."""
    path = "images/scan00000.png"
    b = _broker(
        monkeypatch,
        rows=_rows(SERVER_MAX_PAGE + 1),
        fail_from_offset=SERVER_MAX_PAGE,
    )
    mint_id = core.new_mint_id()
    b._tstore[core.upload_marker_path_for_object(path)] = core.upload_marker_document(
        path=path,
        mint_id=mint_id,
        max_bytes=core.MAX_LOCAL_IMAGE_BYTES,
        minted_at=time.time(),
        ttl_s=core.UPLOAD_MARKER_TTL_S,
    )
    discarded = []
    monkeypatch.setattr(
        b,
        "_discard_staged_upload",
        lambda _a, p: discarded.append(p) or _noop(),
    )
    monkeypatch.setattr(b, "_remove_upload_marker", lambda _a, _p: _noop())

    with pytest.raises(ValueError) as exc:
        await b.upload_local_image(
            ARTIFACT, "scan00000.png", done=True, mint_id=mint_id, context=None
        )

    assert "could not be listed in full" in str(exc.value)
    assert "has been removed" in str(exc.value)
    assert discarded == [path], "the staged object must be discarded, not left"


async def _noop():
    return None
