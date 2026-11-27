"""Pin the GeoJSON label convention, because documenting it is not enforcing it.

``start_training`` and ``preflight_training_dataset`` both accept ``.geojson``
polygon FeatureCollections as labels, and until now nothing said which axis
convention they assume. A y-up file — the GIS/mathematical convention, which is
what most desktop annotation tools export — rasterises VERTICALLY MIRRORED and
trains silently: the polygons are still valid, still inside the canvas, and the
instance count is unchanged, so there is no symptom until the model is bad.

The docstrings now state the convention. These tests are what stop the stated
convention and the implemented one drifting apart, and two of them pin
behaviours the old docstring got WRONG:

* a MultiPolygon consumes one id PER PART, not one per feature (the docstring
  claimed per feature; the code has never done that);
* interior rings are IGNORED, so an annulus becomes a filled disc.

Each is asserted on the rasterised array rather than on the docstring, so a
future change that alters the behaviour fails here instead of quietly making the
documentation false.

Plain pytest — no GPU, no Ray, no live Hypha. Needs pillow + numpy only.
"""

from __future__ import annotations

import numpy as np

from training import rasterize_geojson


def _fc(*geometries):
    return {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": g} for g in geometries],
    }


def _square(x0, y0, x1, y1):
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


def test_coordinates_are_COLUMN_ROW_from_a_TOP_LEFT_origin_with_y_DOWNWARD():
    """`(x, y)` is `(column, row)`, origin top-left, y increasing downward.

    The discriminating part is WHICH ROWS get filled. A polygon whose y values
    are 0..4 must land on rows 0..4 — under a y-up reading the same file would
    fill rows 5..9 instead, which is precisely the silent mirroring this pins.
    """
    mask = rasterize_geojson(_fc(_square(0, 0, 4, 4)), width=10, height=10)
    rows, cols = np.nonzero(mask)
    assert rows.min() == 0 and rows.max() == 4, (
        f"y=0..4 filled rows {rows.min()}..{rows.max()}; the convention is y-DOWN "
        f"from a TOP-LEFT origin, so it must be rows 0..4"
    )
    assert cols.min() == 0 and cols.max() == 4, (
        f"x=0..4 filled cols {cols.min()}..{cols.max()}; x is the COLUMN"
    )


def test_a_y_UP_file_is_MIRRORED_and_nothing_raises():
    """The failure mode itself, pinned so the docs' warning stays true.

    This is not asserting desirable behaviour — it is recording that the mirrored
    read is silent, which is the whole reason the convention has to be written
    down. If a future change starts rejecting or auto-flipping such a file, this
    test should fail and be rewritten deliberately.
    """
    y_down = rasterize_geojson(_fc(_square(0, 0, 4, 4)), width=10, height=10)
    y_up = rasterize_geojson(_fc(_square(0, 9, 4, 5)), width=10, height=10)
    rows_up, _ = np.nonzero(y_up)
    assert rows_up.min() == 5 and rows_up.max() == 9, (
        "the same object written y-up should land on rows 5..9 under the y-down "
        f"reading, got {rows_up.min()}..{rows_up.max()}"
    )
    # Same instance count, different pixels: that is what makes it undetectable.
    assert len(np.unique(y_up)) == len(np.unique(y_down))
    assert not np.array_equal(y_up, y_down)


def test_a_MULTIPOLYGON_consumes_ONE_ID_PER_PART_not_one_per_feature():
    """The docstring used to claim one instance per feature. It never was."""
    multi = {
        "type": "MultiPolygon",
        "coordinates": [
            _square(0, 0, 2, 2)["coordinates"],
            _square(5, 5, 7, 7)["coordinates"],
            _square(0, 5, 2, 7)["coordinates"],
        ],
    }
    mask = rasterize_geojson(_fc(multi), width=10, height=10)
    ids = sorted(int(v) for v in np.unique(mask) if v)
    assert ids == [1, 2, 3], (
        f"one MultiPolygon feature with 3 parts produced ids {ids}; the behaviour "
        f"is one id PER PART, and the documentation must keep saying so"
    )


def test_INTERIOR_RINGS_are_ignored_so_an_annulus_becomes_a_filled_disc():
    """Only the outer ring is drawn. Undocumented until now; pinned here."""
    holed = {
        "type": "Polygon",
        "coordinates": [
            _square(0, 0, 9, 9)["coordinates"][0],  # outer
            _square(3, 3, 6, 6)["coordinates"][0],  # hole
        ],
    }
    mask = rasterize_geojson(_fc(holed), width=10, height=10)
    assert mask[5, 5] == 1, (
        "the centre of the interior ring is not filled, so holes are now being "
        "honoured — a real improvement, but it changes documented behaviour and "
        "this test must be updated deliberately rather than deleted"
    )
