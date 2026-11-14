"""#0139 — a member re-downloaded mid-merge is checked where it lands.

``aggregate()`` hashes every member once per merge before any soup is built. That
alone is not enough, and the gap is easy to miss: the member cache is evicted under
memory pressure, so a member verified in the preflight can be re-downloaded later in
the same merge, and those bytes would enter the soup having been checked only in an
earlier incarnation of the file. The guard therefore also lives at the point of
download, in ``CellposeRuntime._ensure_member``.

Plain pytest — no torch, no GPU, no network. ``_ensure_member`` is exercised on an
instance built without ``__init__`` with the three collaborators it uses replaced,
because the defect is byte bookkeeping and standing up a real runtime would test
cellpose instead.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GOOD = b"the bytes this contribution attested to"
GOOD_SHA = hashlib.sha256(GOOD).hexdigest()
OTHER = b"different bytes altogether"


def _runtime(tmp_path: Path, payload: bytes):
    """A ``CellposeRuntime`` with no ``__init__`` and three collaborators replaced.

    ``_download_member`` is the boundary the bytes arrive over; ``_soup_dir`` is
    where the cache lives; ``_evict_members`` is bookkeeping this test does not
    exercise. Everything else — the cache key, the ``.part`` rename, the digest
    comparison — is the real code.
    """
    import runtime_cellpose

    rt = object.__new__(runtime_cellpose.CellposeRuntime.func_or_class)
    rt._soup_dl_bytes = 0
    rt._soup_dir = lambda: tmp_path
    rt._download_member = lambda url: payload
    rt._evict_members = lambda incoming_bytes: None
    return rt


def test_matching_bytes_are_cached_and_returned(tmp_path):
    rt = _runtime(tmp_path, GOOD)
    path = rt._ensure_member("contrib/c/w.pt", "https://example.invalid/w", GOOD_SHA)
    assert path.exists()
    assert path.read_bytes() == GOOD
    assert rt._soup_dl_bytes == len(GOOD)


def test_bytes_that_disagree_with_the_recorded_digest_raise(tmp_path):
    rt = _runtime(tmp_path, OTHER)
    with pytest.raises(ValueError, match="not the bytes the contribution attested"):
        rt._ensure_member("contrib/c/w.pt", "https://example.invalid/w", GOOD_SHA)


def test_the_bad_download_is_not_left_in_the_cache(tmp_path):
    """The part that makes the raise worth anything. A guard that rejected the
    bytes and cached them anyway would be defeated by the next call, which finds a
    cache hit and trusts it — and the failure would then look intermittent."""
    rt = _runtime(tmp_path, OTHER)
    with pytest.raises(ValueError):
        rt._ensure_member("contrib/c/w.pt", "https://example.invalid/w", GOOD_SHA)
    assert list(tmp_path.iterdir()) == []


def test_a_rejected_download_is_not_counted_as_transport(tmp_path):
    """Bytes that never became a member are not member bytes. Counting them would
    inflate the merge's reported download volume with traffic that bought nothing."""
    rt = _runtime(tmp_path, OTHER)
    with pytest.raises(ValueError):
        rt._ensure_member("contrib/c/w.pt", "https://example.invalid/w", GOOD_SHA)
    assert rt._soup_dl_bytes == 0


def test_a_cache_hit_is_not_re_downloaded(tmp_path):
    """The reason the check sits on this path and not on the cache hit: a warm
    member must not pay for the network, and it has already been hashed by the
    preflight this merge ran."""
    rt = _runtime(tmp_path, GOOD)
    first = rt._ensure_member("contrib/c/w.pt", "https://example.invalid/w", GOOD_SHA)
    rt._download_member = lambda url: pytest.fail("re-downloaded a cached member")
    second = rt._ensure_member("contrib/c/w.pt", "https://example.invalid/w", GOOD_SHA)
    assert first == second


def test_no_recorded_digest_means_no_check_rather_than_a_crash(tmp_path):
    """``score_soup`` is also called directly, by tests and by the stock-base path,
    with no digest to compare against. Inventing one would be worse than recording
    that there was nothing to check."""
    rt = _runtime(tmp_path, OTHER)
    path = rt._ensure_member("contrib/c/w.pt", "https://example.invalid/w", None)
    assert path.read_bytes() == OTHER


def test_case_difference_alone_does_not_reject_a_download(tmp_path):
    rt = _runtime(tmp_path, GOOD)
    path = rt._ensure_member("contrib/c/w.pt", "https://example.invalid/w",
                             GOOD_SHA.upper())
    assert path.read_bytes() == GOOD


def test_two_members_do_not_share_a_cache_entry(tmp_path):
    """A discrimination control for the cache key. If the key collapsed, the second
    member would read back as the first and every digest check after the first
    would be comparing the wrong file — silently, and in the passing direction."""
    rt = _runtime(tmp_path, GOOD)
    a = rt._ensure_member("contrib/a/w.pt", "https://example.invalid/a", GOOD_SHA)
    rt._download_member = lambda url: OTHER
    b = rt._ensure_member("contrib/b/w.pt", "https://example.invalid/b", None)
    assert a != b
    assert a.read_bytes() == GOOD and b.read_bytes() == OTHER
