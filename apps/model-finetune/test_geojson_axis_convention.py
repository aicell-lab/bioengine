"""The GeoJSON axis convention, pinned where it is now documented.

`rasterize_geojson` reads polygon coordinates as image pixels `(x, y)` with the
origin at the TOP-LEFT and `y` increasing DOWNWARD. That is not a choice made in
this file: the coordinates go to PIL's `ImageDraw.polygon` on a `(width, height)`
canvas unchanged, and that is what PIL does.

A docstring stating a convention is a claim. These arms are what make it a
measurement, and what makes a future flip go red instead of quietly changing
what callers are told. The third arm is the reason the convention has to be
written down at all: a y-UP annotation set rasterises vertically mirrored with
every polygon still inside the image, so nothing raises and training proceeds on
wrong masks.

Offline: `rasterize_geojson` is a pure function over numpy and PIL. No network,
no credentials, no cluster.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

H = W = 64


def _fc(coords):
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        ],
    }


def _halves(mask):
    return int((mask[: H // 2] != 0).sum()), int((mask[H // 2 :] != 0).sum())


def test_geojson_y_runs_downward_from_the_top_left_not_upward() -> None:
    """A polygon at y=4..16 must rasterise into the TOP rows.

    The two halves are compared against each other rather than just asserting
    the top is non-empty: "it drew something somewhere" would pass under either
    convention, which is the whole thing in question.
    """
    import training

    mask = training.rasterize_geojson(
        _fc([[8, 4], [56, 4], [56, 16], [8, 16], [8, 4]]), width=W, height=H
    )
    assert mask.shape == (H, W), mask.shape
    top, bottom = _halves(mask)
    assert top > 0, "the polygon rasterised to nothing at all"
    assert bottom == 0, (
        f"y is being read UPWARD: a polygon at y=4..16 landed in the bottom half "
        f"({bottom} px) instead of the top ({top} px). The documented convention "
        f"is origin top-left, y increasing downward."
    )


def test_geojson_instance_ids_start_at_one_in_document_order() -> None:
    """The other half of what the docstring promises, so the two cannot drift."""
    import training

    fc = _fc([[1, 1], [10, 1], [10, 10], [1, 10], [1, 1]])
    fc["features"].append(
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[20, 20], [30, 20], [30, 30], [20, 30], [20, 20]]],
            },
        }
    )
    mask = training.rasterize_geojson(fc, width=W, height=H)
    ids = sorted(int(v) for v in np.unique(mask) if v != 0)
    assert ids == [1, 2], ids
    # The FIRST feature must be id 1 -- document order, not position order.
    assert int(mask[5, 5]) == 1, int(mask[5, 5])
    assert int(mask[25, 25]) == 2, int(mask[25, 25])


def test_each_multipolygon_part_gets_its_own_instance_id() -> None:
    """One id per POLYGON RING, not one per feature -- the docstring said feature.

    `rasterize_geojson` increments `instance_id` inside the loop over a
    MultiPolygon's parts, so a single two-part feature yields two instances. The
    inherited docstring said "one instance per Polygon/MultiPolygon feature",
    which reads as one id for the whole feature, and a caller counting instances
    to sanity-check an upload would get a plausible-but-wrong number.
    """
    import training

    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [[[1, 1], [10, 1], [10, 10], [1, 10], [1, 1]]],
                        [[[20, 20], [30, 20], [30, 30], [20, 30], [20, 20]]],
                    ],
                },
            }
        ],
    }
    mask = training.rasterize_geojson(fc, width=W, height=H)
    ids = sorted(int(v) for v in np.unique(mask) if v != 0)
    assert ids == [1, 2], (
        f"one feature with two MultiPolygon parts produced ids {ids}; the parts "
        f"are numbered separately, so the count a caller sees is per ring"
    )
    assert int(mask[5, 5]) == 1 and int(mask[25, 25]) == 2


def test_interior_rings_are_ignored_so_a_hole_is_filled() -> None:
    """Only ring 0 is drawn, so a polygon with a hole trains as if it were solid.

    Pinned because it is now written down. A caller who annotates a nucleus as a
    ring inside a cell gets the ring filled with the cell's id instead of a hole,
    and nothing says so at runtime.
    """
    import training

    outer = [[8, 8], [56, 8], [56, 56], [8, 56], [8, 8]]
    hole = [[24, 24], [40, 24], [40, 40], [24, 40], [24, 24]]
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [outer, hole]},
            }
        ],
    }
    mask = training.rasterize_geojson(fc, width=W, height=H)
    assert int(mask[32, 32]) == 1, (
        f"the hole centre is {int(mask[32, 32])}; interior rings are ignored, so "
        f"it must carry the outer ring's id"
    )
    ids = sorted(int(v) for v in np.unique(mask) if v != 0)
    assert ids == [1], ids


def test_a_y_up_annotation_set_rasterises_mirrored_and_raises_nothing() -> None:
    """CONTROL, and the justification for documenting the convention.

    This does not test a defect -- reading `(x, y)` top-left down is correct and
    the arm above pins it. It measures what a caller who assumed the OTHER
    convention gets: the same object, expressed y-up on a 64-row canvas, lands
    in the opposite half with the same pixel count. No exception, no empty mask,
    no out-of-bounds coordinate -- so there is no runtime signal a caller could
    have noticed, and the only possible warning is the written convention.
    """
    import training

    down = training.rasterize_geojson(
        _fc([[8, 4], [56, 4], [56, 16], [8, 16], [8, 4]]), width=W, height=H
    )
    # Same polygon as a y-up annotator would write it: y' = (height - 1) - y.
    up = training.rasterize_geojson(
        _fc([[8, 59], [56, 59], [56, 47], [8, 47], [8, 59]]), width=W, height=H
    )
    d_top, d_bottom = _halves(down)
    u_top, u_bottom = _halves(up)
    assert (d_top, d_bottom) == (u_bottom, u_top), (
        f"expected a clean vertical mirror, got down=({d_top},{d_bottom}) "
        f"up=({u_top},{u_bottom})"
    )
    assert u_top == 0 and u_bottom > 0, (u_top, u_bottom)
    assert int((up != 0).sum()) == int((down != 0).sum()), "mirrored area differs"
