"""Tests for ``pool.uniform_soup`` — the AXIS-1 combine rule.

Separate from ``test_pool.py`` on purpose: that file's contract is stdlib-only
(``pool``'s heavy imports are function-local), and these need real tensors.

The memory shape is the point. ``uniform_soup`` takes loaders rather than
materialised state dicts so that peak RSS is flat in member count; a version
that holds every member at once measured ~27 GB on 16 cpsam members against a
12 GiB reservation and was OOM-killed. ``test_no_member_is_retained`` is the
regression guard for that — it fails if the members outlive the call.
"""

from __future__ import annotations

import gc
import weakref

import pytest

import pool

torch = pytest.importorskip("torch")


def _sd(value, dtype=torch.float32):
    return {"a.weight": torch.full((4,), value, dtype=dtype),
            "b.bias": torch.full((2,), -value, dtype=dtype)}


def test_mean_of_members():
    out = pool.uniform_soup([lambda: _sd(1.0), lambda: _sd(3.0), lambda: _sd(5.0)])
    assert torch.allclose(out["a.weight"], torch.full((4,), 3.0))
    assert torch.allclose(out["b.bias"], torch.full((2,), -3.0))


def test_single_member_round_trips_unchanged():
    out = pool.uniform_soup([lambda: _sd(2.5)])
    assert torch.allclose(out["a.weight"], torch.full((4,), 2.5))


def test_dtype_is_restored_after_float32_averaging():
    """Averaging happens in float32 for stability, but the community checkpoint
    must round-trip into cellpose with each key's original dtype."""
    out = pool.uniform_soup([lambda: _sd(1.0, torch.float16),
                             lambda: _sd(2.0, torch.float16)])
    assert out["a.weight"].dtype is torch.float16
    assert torch.allclose(out["a.weight"], torch.full((4,), 1.5, dtype=torch.float16))


def test_each_loader_is_called_once():
    calls = []

    def loader(i):
        calls.append(i)
        return _sd(float(i))

    pool.uniform_soup([lambda i=i: loader(i) for i in range(4)])
    assert calls == [0, 1, 2, 3]


def test_no_member_is_resident_once_merged():
    """Peak memory must not grow with member count: by the time the next member is
    loaded, every earlier one must already be unreachable.

    Sampled *during* the merge, not after — once ``uniform_soup`` returns, its
    locals are gone and even an implementation that pinned every member would
    look clean.
    """
    refs = []
    still_alive = []

    def loader():
        gc.collect()
        still_alive.append(sum(1 for ref in refs if ref() is not None))
        sd = _sd(1.0)
        refs.append(weakref.ref(sd["a.weight"]))
        return sd

    pool.uniform_soup([loader] * 8)
    assert still_alive == [0] * 8


def test_empty_is_rejected():
    with pytest.raises(ValueError):
        pool.uniform_soup([])
