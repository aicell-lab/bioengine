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
