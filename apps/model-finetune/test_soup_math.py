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
import pathlib
import weakref

import pytest

import pool

torch = pytest.importorskip("torch")


def _sd(value, dtype=torch.float32):
    return {"a.weight": torch.full((4,), value, dtype=dtype),
            "b.bias": torch.full((2,), -value, dtype=dtype)}


def _base(dtype=torch.float32):
    """The base checkpoint these members were fine-tuned from.

    ``uniform_soup`` takes it as a REQUIRED loader, because its key set is the
    inventory every member key is classified against (#0070 requirement 2) and its
    non-learned values are what the soup carries through (requirement 3). Passing it
    explicitly in every test is the point: a default would be the "neither
    classifiable nor classified, so average it" hole the ruling closes.
    """
    return _sd(0.0, dtype=dtype)


def test_mean_of_members():
    out = pool.uniform_soup([lambda: _sd(1.0), lambda: _sd(3.0), lambda: _sd(5.0)],
                            base=_base)
    assert torch.allclose(out["a.weight"], torch.full((4,), 3.0))
    assert torch.allclose(out["b.bias"], torch.full((2,), -3.0))


def test_single_member_round_trips_unchanged():
    out = pool.uniform_soup([lambda: _sd(2.5)], base=_base)
    assert torch.allclose(out["a.weight"], torch.full((4,), 2.5))


def test_dtype_is_restored_after_float32_averaging():
    """Averaging happens in float32 for stability, but the community checkpoint
    must round-trip into cellpose with each key's original dtype."""
    out = pool.uniform_soup([lambda: _sd(1.0, torch.float16),
                             lambda: _sd(2.0, torch.float16)],
                            base=_base)
    assert out["a.weight"].dtype is torch.float16
    assert torch.allclose(out["a.weight"], torch.full((4,), 1.5, dtype=torch.float16))


def test_each_loader_is_called_once():
    calls = []

    def loader(i):
        calls.append(i)
        return _sd(float(i))

    pool.uniform_soup([lambda i=i: loader(i) for i in range(4)], base=_base)
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

    pool.uniform_soup([loader] * 8, base=_base)
    assert still_alive == [0] * 8


def test_empty_is_rejected():
    with pytest.raises(ValueError):
        pool.uniform_soup([], base=_base)


def test_an_empty_first_member_is_rejected_not_averaged_away():
    """The guard used to be ``if not acc``, so an empty FIRST member left the
    accumulator falsy, the branch re-entered on member 2, and the sum ran over
    n-1 while the divisor stayed n — a soup scaled by (n-1)/n with nothing
    raised. Later positions failed loudly on a KeyError, so only this one was
    silent."""
    with pytest.raises(ValueError, match="member 0 loaded an empty state_dict"):
        pool.uniform_soup([dict] + [lambda: _sd(10.0)] * 3, base=_base)


def test_an_empty_first_member_is_rejected_at_two_members_too():
    """n=2 is the maximal-error case: the old guard summed one member and divided
    by two, so every published weight was exactly half. Worth its own arm because
    a (n-1)/n scale factor is easy to dismiss as noise at n=16 and impossible to
    at n=2."""
    with pytest.raises(ValueError, match="member 0 loaded an empty state_dict"):
        pool.uniform_soup([dict, lambda: _sd(10.0)], base=_base)


def test_an_empty_later_member_is_rejected_by_the_same_guard():
    with pytest.raises(ValueError, match="member 2 loaded an empty state_dict"):
        pool.uniform_soup([lambda: _sd(10.0), lambda: _sd(10.0), dict], base=_base)


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


# ── the merge carries the BASE diameter, never an average (#0070 req 1+3) ────
# RULED 2026-11-14: the operative value in the tensor is the base checkpoint's,
# unchanged; the contributors' own diameters are recorded in the community
# manifest instead of being averaged into a number nobody measured.
#
# The four diameters below are the real per-site values from the production pool
# (29.711 / 54.986 / 28.912 / 117.353). Their mean is 57.7405, which is what the
# merge used to publish — wrong by about a factor of two for three of the four.

POOL_DIAMS = (29.711, 54.986, 28.912, 117.353)
AVERAGED_DIAM = 57.7405  # what the defect produced; must never be published again


def _pool_loaders():
    return [lambda d=d: _ckpt(1.0, d) for d in POOL_DIAMS]


def test_the_merge_carries_the_base_diameter_unchanged() -> None:
    """Requirement 3: one operative value in the tensor, and it is traceable.

    The base model's diameter is the prior the soup inherited, and the soup did
    not learn a diameter, so it is the only number here that describes something
    real.
    """
    out = pool.uniform_soup(_pool_loaders(), base=lambda: _ckpt(0.0, BASE_DIAM))

    assert float(out["diam_labels"][0]) == pytest.approx(BASE_DIAM, rel=1e-9)
    assert float(out["diam_mean"][0]) == pytest.approx(BASE_DIAM, rel=1e-9)


def test_the_averaged_diameter_is_not_published_and_the_two_differ() -> None:
    """Discrimination control for the test above.

    BASE_DIAM and the average must be far apart, or "carries the base" and "still
    averages" would be the same assertion and the test above could pass on the
    unfixed code. Stated as a ratio so it cannot be satisfied by both numbers
    drifting together.
    """
    out = pool.uniform_soup(_pool_loaders(), base=lambda: _ckpt(0.0, BASE_DIAM))
    published = float(out["diam_labels"][0])

    assert published != pytest.approx(AVERAGED_DIAM, rel=1e-3)
    assert AVERAGED_DIAM / BASE_DIAM > 1.9  # the factor-of-two the row reported
    # And the weights really were averaged, so this is not passing because the
    # merge did nothing at all.
    assert torch.allclose(out["a.weight"], torch.full((4,), 1.0))


def test_the_learned_weights_are_still_averaged_while_the_diameter_is_not() -> None:
    """The exclusion must be surgical: a soup that stopped averaging everything
    would satisfy the diameter assertions and be useless."""
    out = pool.uniform_soup(
        [lambda: _ckpt(1.0, 29.711), lambda: _ckpt(3.0, 117.353)],
        base=lambda: _ckpt(0.0, BASE_DIAM),
    )

    assert torch.allclose(out["a.weight"], torch.full((4,), 2.0))
    assert torch.allclose(out["b.bias"], torch.full((2,), -2.0))
    assert float(out["diam_labels"][0]) == pytest.approx(BASE_DIAM, rel=1e-9)


def test_a_module_prefixed_diameter_is_carried_through_too() -> None:
    """The same checkpoint reaches this code bare or module-prefixed depending on
    how it was wrapped when saved, so the carry-through keys off the last
    component exactly as the l2 exclusion does."""
    out = pool.uniform_soup(
        [lambda: _ckpt(1.0, 29.711, prefix="net."),
         lambda: _ckpt(1.0, 117.353, prefix="net.")],
        base=lambda: _ckpt(0.0, BASE_DIAM, prefix="net."),
    )

    assert float(out["net.diam_labels"][0]) == pytest.approx(BASE_DIAM, rel=1e-9)


def test_the_soup_does_not_alias_the_base_tensors() -> None:
    """The carried value is a clone: a caller that mutates the published soup must
    not reach back into the base checkpoint it was built from."""
    base_sd = _ckpt(0.0, BASE_DIAM)
    out = pool.uniform_soup(_pool_loaders(), base=lambda: base_sd)

    out["diam_labels"] += 1.0
    assert float(base_sd["diam_labels"][0]) == pytest.approx(BASE_DIAM, rel=1e-9)


# ── requirement 2: unclassifiable keys FAIL the merge, never get averaged ────


def test_a_key_absent_from_the_base_fails_the_merge() -> None:
    """Requirement 2, and the property that stops this recurring.

    A future non-learned nn.Parameter added to the architecture is a key an
    existing pool's base does not have. It must break the merge rather than being
    quietly souped into every community checkpoint — which is exactly how
    diam_labels survived review.
    """
    def member():
        sd = _ckpt(1.0, 29.711)
        sd["net.some_future_statistic"] = torch.tensor([7.0])
        return sd

    with pytest.raises(ValueError, match="absent from the base checkpoint"):
        pool.uniform_soup([member], base=lambda: _ckpt(0.0, BASE_DIAM))


def test_the_failure_names_the_offending_key() -> None:
    """An operator has to be able to act on this without re-deriving it."""
    def member():
        sd = _ckpt(1.0, 29.711)
        sd["net.some_future_statistic"] = torch.tensor([7.0])
        return sd

    with pytest.raises(ValueError) as exc:
        pool.uniform_soup([member], base=lambda: _ckpt(0.0, BASE_DIAM))

    assert "net.some_future_statistic" in str(exc.value)
    assert "MUST NOT be averaged" in str(exc.value)


def test_a_member_missing_a_base_key_fails_the_merge() -> None:
    """The other direction: a member that does not carry everything the base
    declares is a layout mismatch, and averaging over the intersection would
    publish a checkpoint missing parameters."""
    def member():
        sd = _ckpt(1.0, 29.711)
        del sd["b.bias"]
        return sd

    with pytest.raises(ValueError, match="missing 1 key"):
        pool.uniform_soup([member], base=lambda: _ckpt(0.0, BASE_DIAM))


def test_an_empty_base_is_rejected() -> None:
    with pytest.raises(ValueError, match="base loaded an empty state_dict"):
        pool.uniform_soup([lambda: _sd(1.0)], base=dict)


def test_base_is_required_so_a_caller_cannot_default_into_averaging() -> None:
    """Requirement 2 at the API boundary: there is no base-less call. A default
    would be the "neither classifiable nor classified, so average it" hole."""
    with pytest.raises(TypeError):
        pool.uniform_soup([lambda: _sd(1.0)])  # type: ignore[call-arg]


def test_the_declared_non_learned_set_is_what_the_model_registers() -> None:
    """#0070 requirement 1: one declaration, not two.

    cellpose_model.py registers these by iterating pool.NON_WEIGHT_PARAMS, so a
    non-learned Parameter cannot be added to the architecture without also being
    excluded from the soup. This asserts the registration site still reads the
    declaration rather than naming fields literally — the drift this guards is a
    future edit that hardcodes a third name in one place only.
    """
    src = pathlib.Path(__file__).with_name("cellpose_model.py").read_text()

    assert "for _name in sorted(pool.NON_WEIGHT_PARAMS):" in src
    # Control: the literal per-field registration must be gone, or the loop could
    # have been added beside it and the two would drift exactly as before.
    assert "self.net.diam_labels = nn.Parameter" not in src
    assert "self.net.diam_mean = nn.Parameter" not in src
    assert pool.NON_WEIGHT_PARAMS == frozenset({"diam_labels", "diam_mean"})


def test_the_published_digest_is_the_same_for_two_merges_of_the_same_members() -> None:
    """#0104: a digest that changes for identical input is worse than no digest.

    Two merges of the same members must publish the same weights_sha256, or the
    obvious diagnostic — re-merge and compare — reports corruption where there
    is none.
    """
    loaders = [lambda: _sd(1.0), lambda: _sd(3.0), lambda: _sd(5.0)]

    first = pool.state_dict_digest(pool.uniform_soup(loaders, base=_base))
    second = pool.state_dict_digest(pool.uniform_soup(loaders, base=_base))

    assert first == second


def test_the_digest_still_moves_when_a_tensor_moves() -> None:
    """The discrimination control for the test above: a digest could be stable
    because it stopped reading the tensors at all."""
    soup = pool.uniform_soup([lambda: _sd(1.0), lambda: _sd(3.0)], base=_base)
    nudged = dict(soup)
    nudged["a.weight"] = soup["a.weight"] + 1e-6

    assert pool.state_dict_digest(nudged) != pool.state_dict_digest(soup)
    assert pool.state_dict_digest({"renamed": soup["a.weight"],
                                   "b.bias": soup["b.bias"]}) != pool.state_dict_digest(soup)


def test_the_digest_does_not_follow_the_file_the_soup_is_saved_under(tmp_path) -> None:
    """Why the digest cannot be ``sha256_file`` of the saved head.

    ``materialize_soup`` writes to ``head-<uuid4>.pt`` — the stem has to stay
    unique so two concurrent merges in one replica's soup dir cannot collide —
    and torch's zip container embeds that stem in every record name. So the file
    digest is a function of the filename. The first two asserts are a positive
    control: they fail if torch ever stops doing that, which would mean this
    test has stopped testing anything.
    """
    soup = pool.uniform_soup([lambda: _sd(1.0), lambda: _sd(3.0)], base=_base)
    a, b = tmp_path / "head-aaaaaaaa.pt", tmp_path / "head-bbbbbbbb.pt"
    torch.save(soup, str(a))
    once = pool.sha256_file(str(a))
    torch.save(soup, str(a))
    torch.save(soup, str(b))

    assert pool.sha256_file(str(a)) == once, "torch.save is not deterministic"
    assert a.stat().st_size == b.stat().st_size
    assert pool.sha256_file(str(a)) != pool.sha256_file(str(b))
    assert pool.state_dict_digest(soup) == pool.state_dict_digest(
        torch.load(str(b), map_location="cpu", weights_only=False)
    )
