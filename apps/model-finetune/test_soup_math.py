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


def test_an_empty_first_member_is_rejected_not_averaged_away():
    """The guard used to be ``if not acc``, so an empty FIRST member left the
    accumulator falsy, the branch re-entered on member 2, and the sum ran over
    n-1 while the divisor stayed n — a soup scaled by (n-1)/n with nothing
    raised. Later positions failed loudly on a KeyError, so only this one was
    silent."""
    with pytest.raises(ValueError, match="member 0 loaded an empty state_dict"):
        pool.uniform_soup([dict] + [lambda: _sd(10.0)] * 3)


def test_an_empty_later_member_is_rejected_by_the_same_guard():
    with pytest.raises(ValueError, match="member 2 loaded an empty state_dict"):
        pool.uniform_soup([lambda: _sd(10.0), lambda: _sd(10.0), dict])


# ── l2_from_base is weight-only (#0070 requirement 2) ────────────────────────
# diam_labels and diam_mean are registered as non-learned nn.Parameter
# (cellpose_model.py:78-79), so they sit in the state dict as float32 scalars
# with matching shapes and pass every dtype/shape filter the drift metric had.
# A site's median object diameter differs from the base's by tens of pixels
# while real weight motion is single digits, which made the contributor-facing
# l2_from_base ~99.5% diameter metadata and inverted the per-site ordering.

BASE_DIAM = 30.0
SITE_DIAM = 117.353


def _ckpt(weight_value, diam, prefix=""):
    sd = _sd(weight_value)
    sd[f"{prefix}diam_labels"] = torch.tensor([diam])
    sd[f"{prefix}diam_mean"] = torch.tensor([diam])
    return sd


def test_l2_from_base_ignores_a_moved_diam_scalar() -> None:
    """A contribution whose *only* difference from its base is the diameter has
    not moved its weights at all, so its reported drift must be exactly zero."""
    base = _ckpt(1.0, BASE_DIAM)
    current = _ckpt(1.0, SITE_DIAM)

    assert pool.state_dict_l2_from_base(current, base) == 0.0


def test_l2_from_base_reports_weight_motion_not_diameter_spread() -> None:
    base = _ckpt(1.0, BASE_DIAM)
    current = _ckpt(1.1, SITE_DIAM)

    measured = pool.state_dict_l2_from_base(current, base)

    # 6 weight elements each moved 0.1; the two diam scalars each moved 87.353.
    weight_only = (6 * 0.1**2) ** 0.5
    with_diameter = (6 * 0.1**2 + 2 * (SITE_DIAM - BASE_DIAM) ** 2) ** 0.5
    assert measured == pytest.approx(weight_only, rel=1e-5)
    # The discrimination control: the two answers differ by ~500x, so this test
    # cannot pass by accident on a metric that still counts the diameter.
    assert with_diameter / weight_only > 100
    assert measured < with_diameter / 100


def test_a_module_prefixed_diam_scalar_is_excluded_too() -> None:
    """The same checkpoint reaches this code bare or module-prefixed depending on
    how it was wrapped when saved, so the exclusion keys off the last component."""
    base = _ckpt(1.0, BASE_DIAM, prefix="net.")
    current = _ckpt(1.0, SITE_DIAM, prefix="net.")

    assert pool.state_dict_l2_from_base(current, base) == 0.0


def test_a_nonweight_name_does_not_swallow_a_real_weight() -> None:
    """A substring match would drop any learned tensor whose name merely contains
    one of the excluded names, silently under-reporting drift instead of over."""
    base = {"encoder.diam_labels_proj.weight": torch.full((4,), 1.0)}
    current = {"encoder.diam_labels_proj.weight": torch.full((4,), 1.1)}

    assert not pool.is_non_weight_param("encoder.diam_labels_proj.weight")
    assert pool.state_dict_l2_from_base(current, base) == pytest.approx(
        (4 * 0.1**2) ** 0.5, rel=1e-5
    )


def test_the_merge_still_averages_diam_so_requirement_1_stays_open() -> None:
    """NOT evidence the diameter defect is fixed — the opposite.

    uniform_soup still takes a uniform mean over the diameter scalars, which is
    #0070 requirement 1, and that has no defined output until requirement 3
    (what a community checkpoint should carry instead) is decided. This pins the
    open half so that implementing it announces itself here rather than passing
    unnoticed.
    """
    out = pool.uniform_soup([lambda: _ckpt(1.0, 29.711), lambda: _ckpt(1.0, 54.986),
                             lambda: _ckpt(1.0, 28.912), lambda: _ckpt(1.0, 117.353)])

    assert float(out["diam_labels"][0]) == pytest.approx(57.7405, rel=1e-4)
