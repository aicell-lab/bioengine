"""A region read touches only the chunks the region covers.

The claim this file exists to make checkable is "crops are read from the source and
never downloaded in full, and work for a whole-slide image". That is a statement
about how much is fetched, so the test that matters compares the chunks a region
touches against the chunks the whole image has — and a control asserts that asking
for the whole image really does touch all of them, because a function that always
returned two chunks would satisfy every "few chunks" assertion on its own.

The reassembly is checked against ground truth: build a synthetic array, slice it
directly, and compare with the region assembled chunk by chunk. A plan that is
internally consistent but geometrically wrong passes a shape check and fails this.
"""

import numpy as np
import pytest

import ngff_region as nr

AXES_5D = ["t", "c", "z", "y", "x"]
AXES_2D = ["y", "x"]


def _plan(shape, chunks, axes, region, index=None):
    return nr.plan_region_read(shape, chunks, axes, region, index)


def test_a_region_of_a_whole_slide_image_touches_a_handful_of_chunks():
    """THE CLAUSE, as a number. A 100000x100000 slide at 1024 chunks is ~9500
    chunks; a 512x512 crop must touch at most 4 of them whatever the offset."""
    side, chunk = 100_000, 1024
    total_chunks = ((side + chunk - 1) // chunk) ** 2
    worst = 0
    for x, y in ((0, 0), (1023, 1023), (512, 512), (99_000, 99_000), (4097, 65_535)):
        plan = _plan(
            [1, 1, 1, side, side],
            [1, 1, 1, chunk, chunk],
            AXES_5D,
            {"x": x, "y": y, "width": 512, "height": 512},
            {"t": 0, "c": 0, "z": 0},
        )
        worst = max(worst, plan["chunks_touched"])
        assert plan["chunks_touched"] <= 4, (x, y, plan["chunks_touched"])
    assert total_chunks > 9000, "the control image is not actually whole-slide sized"
    # The point as the fraction it actually is: 4 chunks of 9604 is 0.04% of the
    # store, and the bytes follow the chunks, which is what "never downloaded in
    # full" means in practice. Stated as bytes too, because that is the resource
    # the clause is about.
    assert worst / total_chunks < 0.001
    full_bytes = side * side  # uint8
    read_bytes = worst * chunk * chunk
    assert read_bytes / full_bytes < 0.001
    assert full_bytes > 9e9, "a 10 GB store is the case that cannot be fetched whole"


def test_control_asking_for_the_whole_image_touches_every_chunk():
    """Without this, "touches few chunks" would also pass for a function that can
    only ever return a few."""
    side, chunk = 4096, 1024
    per_axis = side // chunk
    plan = _plan(
        [side, side], [chunk, chunk], AXES_2D, {"x": 0, "y": 0, "width": side, "height": side}
    )
    assert plan["chunks_touched"] == per_axis * per_axis == 16


def test_a_chunk_aligned_region_touches_exactly_one_chunk():
    plan = _plan(
        [1024, 1024], [256, 256], AXES_2D, {"x": 256, "y": 512, "width": 256, "height": 256}
    )
    assert plan["chunks_touched"] == 1
    part = plan["chunks"][0]
    assert part["key"] == (2, 1)
    assert part["src"] == ((0, 256), (0, 256))
    assert part["dst"] == ((0, 256), (0, 256))


def test_a_region_straddling_a_boundary_touches_four_and_splits_correctly():
    plan = _plan(
        [1024, 1024], [256, 256], AXES_2D, {"x": 200, "y": 200, "width": 112, "height": 112}
    )
    assert plan["chunks_touched"] == 4
    # Every destination piece must tile the output exactly once: no gap, no overlap.
    covered = np.zeros((112, 112), dtype=int)
    for part in plan["chunks"]:
        (dy0, dy1), (dx0, dx1) = part["dst"]
        covered[dy0:dy1, dx0:dx1] += 1
    assert covered.min() == 1 and covered.max() == 1, "the plan gaps or overlaps"


def test_a_region_at_the_far_corner_uses_the_partial_edge_chunk():
    """A store's last chunk is usually partial; a plan that assumed full chunks
    would ask for pixels past the end."""
    plan = _plan([1000, 1000], [256, 256], AXES_2D, {"x": 900, "y": 900, "width": 100, "height": 100})
    assert plan["chunks_touched"] == 1
    part = plan["chunks"][0]
    assert part["key"] == (3, 3)
    assert part["src"] == ((900 - 768, 1000 - 768), (900 - 768, 1000 - 768))


def test_reassembly_matches_a_direct_slice_of_the_same_array():
    """Ground truth. Builds the array, slices it, and compares against the region
    assembled chunk by chunk through the plan."""
    rng = np.random.default_rng(0)
    full = rng.integers(0, 255, size=(300, 400), dtype=np.uint8)
    chunks = (64, 128)
    region = {"x": 77, "y": 33, "width": 211, "height": 150}
    plan = _plan(full.shape, chunks, AXES_2D, region)

    def fetch(key):
        cy, cx = key
        return full[
            cy * chunks[0] : (cy + 1) * chunks[0],
            cx * chunks[1] : (cx + 1) * chunks[1],
        ]

    out = np.zeros((region["height"], region["width"]), dtype=np.uint8)
    nr.read_region(plan, fetch, out)
    expect = full[
        region["y"] : region["y"] + region["height"],
        region["x"] : region["x"] + region["width"],
    ]
    assert np.array_equal(out, expect)


def test_reassembly_works_through_pinned_leading_axes():
    rng = np.random.default_rng(1)
    full = rng.integers(0, 255, size=(2, 3, 1, 200, 200), dtype=np.uint8)
    chunks = (1, 1, 1, 64, 64)
    region = {"x": 10, "y": 20, "width": 100, "height": 90}
    plan = _plan(full.shape, chunks, AXES_5D, region, {"t": 1, "c": 2, "z": 0})

    def fetch(key):
        t, c, z, cy, cx = key
        return full[
            t : t + 1,
            c : c + 1,
            z : z + 1,
            cy * chunks[3] : (cy + 1) * chunks[3],
            cx * chunks[4] : (cx + 1) * chunks[4],
        ]

    out = np.zeros((region["height"], region["width"]), dtype=np.uint8)
    nr.read_region(plan, fetch, out)
    expect = full[1, 2, 0, 20:110, 10:110]
    assert np.array_equal(out, expect)


def test_a_missing_chunk_leaves_the_fill_value_rather_than_raising():
    """Zarr treats an absent chunk as the fill value, and a sparse store is normal."""
    plan = _plan([512, 512], [256, 256], AXES_2D, {"x": 0, "y": 0, "width": 512, "height": 512})
    out = np.full((512, 512), 7, dtype=np.uint8)
    nr.read_region(plan, lambda key: None, out)
    assert (out == 7).all()


def test_a_region_outside_the_image_is_refused_not_clamped():
    """Clamping would train on a different area than was annotated."""
    with pytest.raises(nr.RegionOutOfBounds):
        _plan([100, 100], [32, 32], AXES_2D, {"x": 90, "y": 0, "width": 20, "height": 10})
    with pytest.raises(nr.RegionOutOfBounds):
        _plan([100, 100], [32, 32], AXES_2D, {"x": -1, "y": 0, "width": 10, "height": 10})


def test_an_unpinned_non_spatial_axis_is_refused_rather_than_defaulted():
    """Defaulting a channel to 0 would train on one channel of a multi-channel
    store without ever saying so."""
    with pytest.raises(ValueError) as e:
        _plan([3, 100, 100], [1, 32, 32], ["c", "y", "x"], {"x": 0, "y": 0, "width": 10, "height": 10})
    assert "pinned" in str(e.value)


def test_a_transposed_store_is_refused_rather_than_read_wrongly():
    with pytest.raises(ValueError) as e:
        _plan([100, 100], [32, 32], ["x", "y"], {"x": 0, "y": 0, "width": 10, "height": 10})
    assert "y then x" in str(e.value)


def test_a_degenerate_region_is_refused():
    for bad in ({"x": 0, "y": 0, "width": 0, "height": 10}, {"x": 0, "y": 0, "width": 10, "height": -5}):
        with pytest.raises(ValueError):
            _plan([100, 100], [32, 32], AXES_2D, bad)
    with pytest.raises(ValueError) as e:
        _plan([100, 100], [32, 32], AXES_2D, {"x": 0, "y": 0, "width": 10})
    assert "height" in str(e.value)


def test_the_chunk_key_uses_the_separator_the_store_declares():
    """Guessing this yields a 404 on every chunk, which reads as an empty region."""
    assert nr.chunk_key((0, 0, 0, 3, 4)) == "0.0.0.3.4"
    assert nr.chunk_key((0, 0, 0, 3, 4), "/") == "0/0/0/3/4"
    with pytest.raises(ValueError):
        nr.chunk_key((1, 2), "-")


def test_mismatched_dimension_counts_are_refused():
    with pytest.raises(ValueError):
        _plan([100, 100], [32], AXES_2D, {"x": 0, "y": 0, "width": 10, "height": 10})
