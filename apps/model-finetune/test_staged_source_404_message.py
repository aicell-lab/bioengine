"""A 404 on a staged training input must name the lifetime and when its clock starts.

A file staged through `get_upload_url` is DELETED about an hour later, and the
storage service schedules that deletion when the URL is minted -- so the clock runs
from the mint, not from the moment the upload finished. The old message ("does not
exist or has expired") named no window and no remedy, and a caller whose queued run
had waited could not tell a deleted input from a mistyped path.

The mint-not-upload distinction is the half that costs time, and it has its own arm:
staging a dataset by minting many URLs up front and uploading them over the next
twenty minutes spends the window on the loop.

An earlier version of this file asserted the OPPOSITE -- that the object persists and
only the PUT URL expires -- on a probe that read one object per age band and found a
225-day-old one still readable. That inference was wrong: those old objects predate
scheduled deletion. The evidence is the DISTRIBUTION, not a sample. Of 585 objects
under `temp/`, 453 were under an hour old, 17 between one and 1.25 hours, and zero
between 1.25 hours and 30 days; an app staging hundreds an hour cannot leave that
range empty unless they are removed.

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
async def test_the_404_names_the_one_hour_lifetime() -> None:
    """The window is the actionable part: it tells the caller to re-stage."""
    msg = await _raise_message(_entry_instance(404))
    assert "1 hour" in msg, (
        "the message must name the staged file's lifetime; without it a caller "
        "cannot tell a path typo from a file that has been deleted: %r" % msg
    )


@pytest.mark.asyncio
async def test_the_404_says_the_clock_starts_AT_THE_MINT_not_at_the_upload() -> None:
    """The half that costs time, and the half a reader gets wrong by default.

    "one-hour TTL" reads as a limit on how long you may UPLOAD. It is not: the
    deletion is scheduled when `get_upload_url` returns, so a caller who minted 76
    URLs and then uploaded them over twenty minutes has spent a third of the
    window before the last PUT lands. A message that says only "lasts about an
    hour" leaves that caller measuring from the wrong event.
    """
    msg = (await _raise_message(_entry_instance(404))).lower()
    assert "mint" in msg, (
        "the message must say the hour runs from when the URL was MINTED: %r" % msg
    )
    assert "not" in msg and "upload finished" in msg, (
        "the message must say explicitly that the clock is NOT from the upload, "
        "which is the reading a caller arrives with: %r" % msg
    )


@pytest.mark.asyncio
async def test_the_404_still_admits_the_causes_it_cannot_rule_out() -> None:
    """Deletion is the likely cause, not the only one, and the 404 cannot tell.

    A message that asserted deletion alone would send someone who simply mistyped
    a path off to re-upload a file that was never there. So "most likely" plus the
    alternatives, rather than a diagnosis the HTTP status cannot support.
    """
    msg = (await _raise_message(_entry_instance(404))).lower()
    assert "most likely" in msg, (
        "deletion must be offered as likely, not certain: %r" % msg
    )
    assert "never uploaded" in msg and "typo" in msg, (
        "the message must still admit a never-uploaded path and a typo: %r" % msg
    )


@pytest.mark.asyncio
async def test_the_404_tells_the_caller_what_to_do() -> None:
    msg = (await _raise_message(_entry_instance(404))).lower()
    assert "re-stage" in msg, (
        "a message naming a lifetime and no remedy leaves the caller nowhere: %r" % msg
    )
    assert "re-request" in msg and "re-upload" in msg, msg


@pytest.mark.asyncio
async def test_a_NON_staged_404_does_not_claim_a_staging_lifetime() -> None:
    """The staged branch is keyed on the `temp/` prefix, and that needs its own arm.

    An http(s) source or an artifact path reaching this 404 has nothing to do with
    `get_upload_url`, and telling its caller to re-stage would be a wrong
    instruction confidently given. Without this arm the prefix test could be
    inverted or dropped and every message would still read plausibly.
    """
    inst = _entry_instance(404)
    with pytest.raises(FileNotFoundError) as excinfo:
        await inst._load_image_from_source("datasets/ds-1/image.npy")
    msg = str(excinfo.value).lower()
    assert "re-stage" not in msg and "1 hour" not in msg, (
        "a non-staged source must not be told about the staging lifetime: %r" % msg
    )
    assert "never have been uploaded" in msg or "never been uploaded" in msg, msg


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
