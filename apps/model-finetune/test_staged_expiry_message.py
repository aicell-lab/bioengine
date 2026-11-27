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

NOT CLAIMED HERE: which mechanism removes the object. There is no lifecycle or
retention configuration for `temp/` anywhere in this repository (searched, with a
control), so the expiry is enforced by the storage service and extending it is
not a change that can be made here. Stating it is.
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
    assert "re-staged" in msg or "Stage it again" in msg, (
        f"the repair is not stated: {msg}"
    )
    assert "get_upload_url" in msg, (
        f"the message does not say which call produced this path: {msg}"
    )
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
