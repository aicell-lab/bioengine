"""A staged file that has aged out must say so, and say what to do.

`get_upload_url` hands back a `temp/<uuid>.<ext>` path. The staged OBJECT lasts
about an hour -- which is not the same statement as the presigned PUT URL lasting
an hour, and is the one that catches people out, because a fresh download URL is
minted on every read so an expired signature is never what bites.

Measured by a caller driving the public app: a `start_training` submitted 74
minutes after staging, and an `infer` on a path staged 80 minutes earlier, both
failed with `Source 'temp/<uuid>.png' does not exist or has expired.` That
message is true and useless -- "does not exist OR has expired" leaves the caller
unable to tell a typo from an aged-out file, and those have different repairs.

WHAT REMOVES THE OBJECT, AND WHOSE HOUR IT IS. `get_upload_url` passes
`ttl=3600` to `put_file`, and that argument IS the deletion timer: the storage
service schedules the object's removal when the URL is minted. So the hour is
this app's own number, raising it is a one-line change here, and the clock starts
at the MINT rather than at the upload — which is why staging a dataset from URLs
minted up front spends part of it before the first byte moves. These tests cover
only what the app SAYS about that; what the number should be is a separate
decision and is not taken here.
"""

from __future__ import annotations

import asyncio

import pytest


class _Resp:
    def __init__(self, status):
        self.status_code = status
        self.content = b""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Stub:
    """Drives the real `_load_image_from_source` with a 404 on the fetch."""

    def __init__(self, status=404):
        self._status = status

    async def _get_download_url(self, file_path):
        # Minted fresh on every read, which is why a stale signature is not the
        # failure being reported here.
        return f"https://signed.invalid/{file_path}?fresh=1"

    async def _http_retry(self, method, url, **kwargs):
        return _Resp(self._status)


def _load(stub, source):
    import entry

    return asyncio.run(
        entry.EntryApp.func_or_class._load_image_from_source(stub, source)
    )


def test_an_aged_out_STAGED_path_names_the_lifetime_and_the_repair():
    """THE case. The caller needs three things the old message withheld: that a
    staged file has a lifetime, roughly what it is, and that the repair is to
    stage again rather than to retry the same path."""
    with pytest.raises(FileNotFoundError) as err:
        _load(_Stub(), "temp/4f0e2a1c-0000-0000-0000-000000000000.png")
    msg = str(err.value)
    assert "about an hour" in msg, f"the lifetime is not given: {msg}"
    assert (
        "re-staged" in msg or "Stage it again" in msg
    ), f"the repair is not stated: {msg}"
    assert (
        "get_upload_url" in msg
    ), f"the message does not say which call produced this path: {msg}"
    # The waiting cases are the ones that hit it, so they are named.
    assert "queued" in msg.lower(), f"the common cause is not named: {msg}"


def test_a_NON_staged_source_keeps_the_original_message():
    """The control. A caller's own URL or a mistyped artifact path has nothing to
    do with staging, and telling them to re-stage would be wrong advice."""
    with pytest.raises(FileNotFoundError) as err:
        _load(_Stub(), "images/does-not-exist.png")
    msg = str(err.value)
    assert msg == "Source 'images/does-not-exist.png' does not exist or has expired."


def test_a_non_404_failure_is_not_turned_into_an_expiry_claim():
    """A 500 from the store is not an expired file. Reporting it as one would
    send the caller to re-stage a file that is still there."""
    with pytest.raises(RuntimeError) as err:
        _load(_Stub(status=500), "temp/4f0e2a1c-0000-0000-0000-000000000000.png")
    assert "500" in str(err.value)
    assert "hour" not in str(err.value)


# ── the DOCS half of "stated in the error and docs" ─────────────────────────
#
# The error text was the easy half. A caller who has not hit the failure yet
# reads the README, and it said only "1-hour TTL", which is the sentence the
# reporter read as a limit on how long they could upload.


def _readme() -> str:
    from pathlib import Path

    return (Path(__file__).with_name("README.md")).read_text(encoding="utf-8")


def test_the_README_says_the_staged_OBJECT_is_deleted_not_just_the_url():
    text = _readme()
    # CONTROL: the file has to be the README, or every assertion below is about
    # an empty string.
    assert len(text) > 5000, f"README read implausibly short ({len(text)})"
    assert "get_upload_url" in text, "this is not the README that documents staging"

    lowered = text.lower()
    assert "object is deleted" in lowered or "object is gone" in lowered, (
        "the README does not say the staged OBJECT is removed — only a TTL, "
        "which reads as a limit on how long you may upload"
    )


def test_the_README_names_where_the_hour_is_measured_from():
    """The clock starts at the mint, not at the upload. The reporter staged 76
    files from URLs minted up front, which is exactly the case where that
    difference is spent before the first byte moves."""
    lowered = _readme().lower()
    assert "ttl=3600" in lowered or "ttl` is passed" in lowered, (
        "the README does not name the ttl that schedules the deletion, so a "
        "reader cannot tell whose hour it is or that it can be changed here"
    )
    assert (
        "minted" in lowered
    ), "the README does not say the clock starts when the URL is minted"
