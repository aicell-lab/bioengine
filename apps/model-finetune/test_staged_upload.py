"""Tests for the two halves of a weights upload: the retry, and the stage it leaves behind.

A transient 504 on ``materialize_soup``'s ~1.2 GiB PUT used to cost two separate
things. The upload got one attempt, so a completed 8-member merge (31 min wall) was
discarded; and because ``put_file`` records a manifest row when it mints the URL
rather than when the bytes land, the failure left a row with no object behind it —
invisible to ``list_files``, and fatal to every later ``commit``, which
``_commit_with_retry`` then misread as its own read-after-write 404 and retried for
~62s before re-raising. The pool stayed uncommittable until the stage was discarded,
so one failed upload took the whole pool down rather than just its own merge.

Measured against live Hypha while fixing this: a minted-but-unuploaded row makes
``commit`` raise ``FileNotFoundError: File '<path>' does not exist in the artifact``
while ``list_files`` returns ``[]``, and ``discard`` clears it (a staged row *with*
bytes then commits, as the positive control).

The async bodies are driven with ``asyncio.run`` inside sync tests on purpose: an
``async def`` test silently does not run when pytest-asyncio is absent, which is the
state of the app image unless it is installed by hand.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError

import pytest


def _upload_weights():
    """The real ``_upload_weights``.

    ``CellposeRuntime`` is a ``ray.serve.deployment.Deployment`` once
    ``@bioengine.app`` has run, so its methods are not reachable through it;
    ``func_or_class`` is the class the decorator wrapped.
    """
    import runtime_cellpose

    return runtime_cellpose.CellposeRuntime.func_or_class._upload_weights


def _staged_of(instance):
    import entry

    return entry.EntryApp.func_or_class._staged.__get__(instance)


class _Resp:
    def read(self):
        return b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code: int) -> HTTPError:
    return HTTPError("https://pool.invalid/put", code, "boom", {}, None)


@pytest.fixture
def weights(tmp_path):
    p = tmp_path / "head-deadbeef.pt"
    p.write_bytes(b"weights-payload")
    return p


@pytest.fixture
def no_sleep(monkeypatch):
    """Collapse the backoff. Recorded so a test can assert the ladder was walked."""
    import runtime_cellpose

    slept: List[float] = []
    monkeypatch.setattr(runtime_cellpose.time, "sleep", lambda s: slept.append(s))
    return slept


def _patch_urlopen(monkeypatch, outcomes: List[Any]) -> List[bytes]:
    """Replace urlopen with one that yields ``outcomes`` in order.

    ``_upload_weights`` imports urlopen inside its own body, so patching the module
    attribute is what reaches it. Bodies seen per attempt are returned, which is how
    the "a retry sends a fresh copy" claim is checked rather than assumed.
    """
    import urllib.request

    seen: List[bytes] = []
    it = iter(outcomes)

    def fake_urlopen(req, timeout=None):
        seen.append(req.data)
        out = next(it)
        if isinstance(out, BaseException):
            raise out
        return out

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


def test_a_504_is_retried_and_a_later_attempt_succeeds(monkeypatch, weights, no_sleep):
    seen = _patch_urlopen(monkeypatch, [_http_error(504), _http_error(504), _Resp()])

    _upload_weights()("https://pool.invalid/put", weights)

    assert len(seen) == 3
    # Every attempt sent the whole body — a retry that re-sent nothing would pass a
    # call count while uploading an empty object.
    assert seen == [b"weights-payload"] * 3
    assert no_sleep == [2.0, 4.0]


def test_a_502_and_a_503_are_retried(monkeypatch, weights, no_sleep):
    seen = _patch_urlopen(monkeypatch, [_http_error(502), _http_error(503), _Resp()])
    _upload_weights()("https://pool.invalid/put", weights)
    assert len(seen) == 3


def test_a_transport_error_is_retried(monkeypatch, weights, no_sleep):
    seen = _patch_urlopen(monkeypatch, [URLError("connection reset"), _Resp()])
    _upload_weights()("https://pool.invalid/put", weights)
    assert len(seen) == 2


def test_a_403_is_a_verdict_and_is_not_retried(monkeypatch, weights, no_sleep):
    """A 403/404 on a URL minted seconds earlier will not heal, and a retry would
    re-send 1.2 GiB against it. Same stance as ``_download_member``."""
    seen = _patch_urlopen(monkeypatch, [_http_error(403), _Resp()])

    with pytest.raises(HTTPError) as got:
        _upload_weights()("https://pool.invalid/put", weights)

    assert got.value.code == 403
    assert len(seen) == 1
    assert no_sleep == []


def test_a_404_is_a_verdict_and_is_not_retried(monkeypatch, weights, no_sleep):
    seen = _patch_urlopen(monkeypatch, [_http_error(404), _Resp()])
    with pytest.raises(HTTPError):
        _upload_weights()("https://pool.invalid/put", weights)
    assert len(seen) == 1


def test_exhausted_attempts_reraise_rather_than_returning_quietly(monkeypatch, weights, no_sleep):
    """The caller must see the failure. A retry loop that falls off its end returns
    None, which reads to ``materialize_soup`` as a completed upload."""
    import runtime_cellpose

    attempts = runtime_cellpose._WEIGHTS_PUT_ATTEMPTS
    seen = _patch_urlopen(monkeypatch, [_http_error(504)] * attempts)

    with pytest.raises(HTTPError) as got:
        _upload_weights()("https://pool.invalid/put", weights)

    assert got.value.code == 504
    assert len(seen) == attempts


class _AM:
    """Records the artifact-manager calls ``_staged`` makes."""

    def __init__(self, discard_raises: Optional[BaseException] = None) -> None:
        self.calls: List[str] = []
        self._discard_raises = discard_raises

    async def edit(self, artifact_id: str, stage: bool = False) -> None:
        self.calls.append(f"edit(stage={stage})")

    async def discard(self, artifact_id: str) -> None:
        self.calls.append("discard")
        if self._discard_raises is not None:
            raise self._discard_raises


class _Entry:
    pass


def test_a_failed_body_discards_the_stage():
    am = _AM()

    async def body():
        async with _staged_of(_Entry())(am, "ws/pool"):
            raise RuntimeError("the weights PUT 504'd")

    with pytest.raises(RuntimeError):
        asyncio.run(body())

    assert am.calls == ["edit(stage=True)", "discard"]


def test_a_successful_body_leaves_the_stage_alone():
    """The discriminating control: if ``discard`` ran unconditionally this would
    also pass a 'discard was called' assertion, so the committed path is checked
    too — the commit inside the block consumes the stage."""
    am = _AM()

    async def body():
        async with _staged_of(_Entry())(am, "ws/pool"):
            pass

    asyncio.run(body())
    assert am.calls == ["edit(stage=True)"]


def test_a_cancelled_merge_discards_the_stage():
    """Cancellation is the live case, not a hypothetical: a caller that gives up
    mid-merge raises ``CancelledError``, which is a ``BaseException`` and would slip
    past ``except Exception`` — leaving exactly the orphan row this guards."""
    am = _AM()

    async def body():
        async with _staged_of(_Entry())(am, "ws/pool"):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(body())

    assert am.calls == ["edit(stage=True)", "discard"]


def test_a_failing_discard_does_not_mask_the_original_error():
    """The upload failure is the diagnosis; a secondary failure while cleaning up
    must not replace it."""
    am = _AM(discard_raises=RuntimeError("artifact manager unreachable"))

    async def body():
        async with _staged_of(_Entry())(am, "ws/pool"):
            raise HTTPError("https://pool.invalid/put", 504, "Gateway Timeout", {}, None)

    with pytest.raises(HTTPError) as got:
        asyncio.run(body())

    assert got.value.code == 504
    assert am.calls == ["edit(stage=True)", "discard"]
