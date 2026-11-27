"""A 404 on a staged training input must say what window closed, not just "expired".

`get_upload_url` hands back a presigned PUT valid for one hour. A caller who holds
that handout longer than the window never uploads at all, and S3 then answers the
read with the same 404 it gives for a path that was never written and for one that
has been removed. The old message asserted one of those three ("does not exist or
has expired") and named no window, so a caller could not tell a typo from a timeout.

Offline by construction: the subject's two collaborators on this path are replaced,
so no network and no credentials. Nothing else is attached to the instance --
`test_base_checkpoint_preflight.py` records what that costs, where a logger the
class cannot have hid a defect from 21 arms.
"""

from __future__ import annotations

import io
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STAGED = "temp/6f1c0f7a-not-a-real-upload.npy"


def _entry_instance(status_code: int, payload: bytes = b""):
    """An `EntryApp` with only the two attributes this code path reaches.

    `_get_download_url` is what a non-http source goes through, and `_http_retry`
    is the boundary the HTTP status arrives over. Anything else is deliberately
    absent so an arm cannot pass by being handed a shape production lacks.
    """
    import entry

    cls = getattr(entry.EntryApp, "func_or_class", entry.EntryApp)
    inst = object.__new__(cls)

    async def _dl(file_path):
        return "https://example.invalid/presigned/" + file_path

    async def _retry(method, url, **kwargs):
        return types.SimpleNamespace(
            status_code=status_code,
            content=payload,
            raise_for_status=lambda: None,
        )

    inst._get_download_url = _dl
    inst._http_retry = _retry
    return inst


async def _raise_message(inst) -> str:
    with pytest.raises(FileNotFoundError) as excinfo:
        await inst._load_image_from_source(STAGED)
    return str(excinfo.value)


@pytest.mark.asyncio
async def test_the_404_names_the_one_hour_staging_window() -> None:
    """The window is the actionable part: it tells the caller to re-request a URL."""
    msg = await _raise_message(_entry_instance(404))
    assert "1 hour" in msg, (
        "the message must name the staging window; without it a caller cannot tell "
        "a path typo from an upload URL that timed out: %r" % msg
    )


@pytest.mark.asyncio
async def test_the_404_does_not_assert_one_of_three_indistinguishable_causes() -> None:
    """S3 answers 404 for never-written, removed and never-uploaded alike.

    So the message has to offer all three. The old text picked two and stated them
    as fact ("does not exist or has expired"), which reads as a diagnosis the HTTP
    status cannot support.
    """
    msg = (await _raise_message(_entry_instance(404))).lower()
    for cause in ("never have been uploaded", "removed", "expired"):
        assert cause in msg, "the message must offer %r as a possibility: %r" % (
            cause,
            msg,
        )


@pytest.mark.asyncio
async def test_the_404_tells_the_caller_what_to_do() -> None:
    msg = await _raise_message(_entry_instance(404))
    assert "re-request" in msg.lower() and "re-upload" in msg.lower(), (
        "a message that names three causes and no remedy leaves the caller nowhere: %r"
        % msg
    )


# --- controls -----------------------------------------------------------------
# Each of these would stay green if the message changed, and goes red if the
# surrounding code path broke -- so they bound what the arms above are measuring.


@pytest.mark.asyncio
async def test_a_200_still_loads_and_does_not_raise() -> None:
    """CONTROL: the arms above must not be passing because every input raises.

    A real `.npy` body goes through the same function and comes back as an array.
    """
    buf = io.BytesIO()
    np.save(buf, np.arange(6, dtype=np.uint8).reshape(2, 3))
    arr = await _entry_instance(200, buf.getvalue())._load_image_from_source(STAGED)
    assert arr.shape == (2, 3) and arr.dtype == np.uint8


@pytest.mark.asyncio
async def test_an_unsupported_extension_still_fails_before_any_request() -> None:
    """CONTROL: the extension guard ahead of the request is untouched.

    It raises `ValueError`, not `FileNotFoundError`, so a change that collapsed the
    two would show up here rather than silently widening the 404 path.
    """
    with pytest.raises(ValueError, match="Unsupported file extension"):
        await _entry_instance(404)._load_image_from_source("temp/x.bmp")
