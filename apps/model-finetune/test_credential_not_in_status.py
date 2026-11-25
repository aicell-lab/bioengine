"""A store credential must not end up in the session status document.

The credential for a remote OME-Zarr store travels as a path segment,
``/t/<credential>/``. It is deliberately never written into the dataset artifact,
because that artifact can be made public. It reaches the trainer only because the
broker splices it into ``store_root`` on the way out, for one call.

WHAT PUT IT WITHIN REACH OF A READER is that the error path writes it down. zarr's
not-found error quotes the store it tried, URL and all; that string becomes the
exception, the exception's text is written into ``status.json`` as the FAILED
message and logged, and the endpoint that lists session status takes no caller
context. So one ordinary failure -- a rotated credential, a moved store, a wrong
multiscale name -- publishes a credential to anyone who can list sessions.

A presigned URL going the same way is pre-existing and expires. A store credential
does not, which is why this is guarded in two places rather than one: the region
reader refuses without quoting the URL, and the status write redacts whatever it
is handed. The second is not redundant. Everything in that message comes from some
library's idea of a good error, and the paths nobody has thought of are the only
ones that matter for a credential.

These tests drive the REAL ``_run_training`` failure branch, which is the line that
publishes. A test of the redaction helper alone would pass while that branch wrote
the raw string.
"""

from __future__ import annotations

import pytest

import ngff_region
import training

# A session id must match the shape the server itself generates
# (YYYY-MM-DD-HHMMSS-xxxxxxxx). `training.session_dir` refuses anything else --
# a caller-supplied id was previously used as a directory name unchecked, which
# let `../x` and absolute paths write outside the sessions root. So a descriptive
# id like "s-stale" can no longer be a directory name; these constants keep the
# tests readable while using ids the app accepts.
S_PLAIN = "2026-11-25-120003-00000003"
S_PRESIGNED = "2026-11-25-120004-00000004"
S_RAW = "2026-11-25-120002-00000002"
S_REGION = "2026-11-25-120001-00000001"

SECRET = "SECRET-TOKEN-123"
CRED_ROOT = f"https://ov.example/t/{SECRET}/zarr/ds1/"


class _Stub:
    """Enough of the app to reach the failure branch, and nothing else."""

    _training_tasks: dict = {}

    def __init__(self, exc):
        self._exc = exc

    async def _preparing_heartbeat(self, sid, interval=60.0):
        return None

    async def _prepare_training_data(self, *_a, **_kw):
        raise self._exc


async def _run_and_read(session_id, exc, monkeypatch, tmp_path):
    import entry

    monkeypatch.setattr(training, "sessions_root", lambda: tmp_path / "sessions")
    await entry.EntryApp.func_or_class._run_training(
        _Stub(exc), session_id, "cpsam", [], [], [], [], {}
    )
    return training.get_status(session_id)


@pytest.mark.asyncio
async def test_a_region_read_failure_leaves_no_credential_in_the_status(
    monkeypatch, tmp_path
):
    """THE case, with the exception the region reader actually produces.

    The reader has already redacted it, so this asserts the whole chain holds
    rather than only the helper.
    """
    exc = ngff_region.RegionSourceError(
        f"could not open '0' in "
        f"{ngff_region.redact_store_credential(CRED_ROOT)}: "
        f"FileNotFoundError: No array found in store "
        f"{ngff_region.redact_store_credential(CRED_ROOT)}0"
    )
    st = await _run_and_read(S_REGION, exc, monkeypatch, tmp_path)
    assert st["status"] == "FAILED"
    assert SECRET not in st["message"], f"the credential was published: {st['message']}"
    # The message still has to be useful, or the redaction has traded one
    # unusable outcome for another.
    assert "ov.example" in st["message"]
    assert "No array found" in st["message"]


@pytest.mark.asyncio
async def test_a_RAW_library_error_is_redacted_by_the_status_write_itself(
    monkeypatch, tmp_path
):
    """THE case the second layer exists for.

    This exception has NOT been through the region reader: it is what some other
    library hands over, quoting the credentialed URL verbatim. If redaction lived
    only at the reader's raise site, this would write the credential straight into
    the status document.
    """
    exc = FileNotFoundError(f"No array found in store <FsspecStore({CRED_ROOT}0)>")
    st = await _run_and_read(S_RAW, exc, monkeypatch, tmp_path)
    assert st["status"] == "FAILED"
    assert SECRET not in st["message"], f"the credential was published: {st['message']}"
    assert ngff_region.CREDENTIAL_PLACEHOLDER in st["message"]


@pytest.mark.asyncio
async def test_an_ordinary_failure_message_is_recorded_unchanged(monkeypatch, tmp_path):
    """The control. A status write that mangled or dropped every message would
    pass both tests above while making every other failure undiagnosable."""
    exc = RuntimeError("cellpose ran out of memory on a 4096x4096 tile")
    st = await _run_and_read(S_PLAIN, exc, monkeypatch, tmp_path)
    assert st["status"] == "FAILED"
    assert st["message"] == "cellpose ran out of memory on a 4096x4096 tile"


@pytest.mark.asyncio
async def test_a_presigned_signature_is_NOT_claimed_to_be_redacted(
    monkeypatch, tmp_path
):
    """Stating the limit, so nobody reads this guard as broader than it is.

    A presigned query-string signature in a failure message is pre-existing and
    is NOT scrubbed here. It is a different shape, it expires, and quietly
    half-fixing it would leave the next reader believing failure messages are
    clean of credentials in general. If that is wanted it is its own change.
    """
    exc = RuntimeError("GET https://s3.example/i.tif?X-Amz-Signature=abc123 failed")
    st = await _run_and_read(S_PRESIGNED, exc, monkeypatch, tmp_path)
    assert "abc123" in st["message"], (
        "a presigned signature is now scrubbed too; that is an improvement, but "
        "this test and the module docstring describe the narrower guarantee and "
        "both need updating"
    )
