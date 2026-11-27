"""The GeoJSON axis convention, measured rather than described.

`start_training` and `preflight_training_dataset` both advertise `.geojson`
polygon FeatureCollections as labels, and until now neither said which axis
convention it assumed. A caller reported that their annotations were y-up and
"would have silently trained on mirrored masks"; they rasterised locally instead
of trusting it.

SILENTLY IS THE WHOLE PROBLEM. A vertically mirrored annotation is a perfectly
plausible annotation: the polygons are inside the canvas, the instance count is
right, the mask is non-empty, so the caller's empty-mask guard -- which exists
for coordinate-frame mismatches -- does not fire. Nothing distinguishes it from
a correct file, which is why the convention is documented and NOT validated.

So these tests pin the convention itself. A docstring that says "y increases
downward" is worth nothing if the code does the opposite, and a reader has no
way to check without a test that draws an asymmetric shape and looks at where it
landed.
"""

from __future__ import annotations

import inspect

import numpy as np

import training


def _fc(*polys):
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [list(p)]},
            }
            for p in polys
        ],
    }


def _box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def test_y_increases_DOWNWARD_from_the_top_left():
    """THE case. A box in the first 20 rows must land in the FIRST rows of the
    array, not the last.

    Deliberately asymmetric in y and offset in x, so a mirrored, transposed or
    flipped result all read differently from a correct one.
    """
    mask = training.rasterize_geojson(_fc(_box(5, 2, 15, 20)), width=100, height=80)
    assert mask.shape == (80, 100), "the mask is not (height, width)"
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    assert rows.min() < 5 and rows.max() < 25, (
        f"a polygon at y 2..20 painted rows {rows.min()}..{rows.max()}; y is not "
        f"measured downward from the top"
    )
    assert (
        cols.min() < 8 and cols.max() < 20
    ), f"a polygon at x 5..15 painted columns {cols.min()}..{cols.max()}"


def test_the_pair_is_x_then_y_so_it_indexes_mask_y_x():
    """Order, not just direction. A swapped pair is invisible on a square canvas
    and on a symmetric shape, so the canvas here is rectangular and the box is
    far from the diagonal."""
    mask = training.rasterize_geojson(_fc(_box(60, 4, 70, 10)), width=100, height=40)
    assert mask[6, 65] != 0, "the point (x=65, y=6) inside the polygon is not set"
    assert mask[30, 8] == 0, "a point outside the polygon is set"


def test_a_Y_UP_file_rasterises_MIRRORED_and_nothing_flags_it():
    """The hazard itself, asserted so the documentation is not the only record.

    The same cell annotated y-up lands at the other end of the image. Both masks
    are non-empty with one instance, which is exactly why this cannot be caught:
    the two are indistinguishable without knowing the convention.
    """
    h = 80
    y_down = training.rasterize_geojson(_fc(_box(5, 2, 15, 20)), width=100, height=h)
    # The same box with y measured up from the bottom.
    y_up = training.rasterize_geojson(
        _fc(_box(5, h - 20, 15, h - 2)), width=100, height=h
    )
    assert y_down.max() == 1 and y_up.max() == 1, "both should hold one instance"

    down_rows = np.flatnonzero(y_down.any(axis=1))
    up_rows = np.flatnonzero(y_up.any(axis=1))
    # The substantive claim: the same cell lands at the OTHER END of the image.
    # Not asserted as an exact pixel mirror -- reflecting a row INDEX is
    # h - 1 - y while reflecting a y COORDINATE is h - y, so the two differ by a
    # row and an equality here would be testing that off-by-one rather than the
    # hazard.
    assert down_rows.max() < h / 2 < up_rows.min(), (
        f"the two conventions do not land at opposite ends: y-down painted rows "
        f"{down_rows.min()}..{down_rows.max()}, y-up {up_rows.min()}..{up_rows.max()}"
    )
    assert set(down_rows).isdisjoint(set(up_rows)), "the two overlap"
    # And the thing that makes it silent: the caller's guard keys on an EMPTY
    # mask, and neither of these is empty, so nothing fires.
    assert y_down.any() and y_up.any()


def test_the_convention_is_stated_where_the_API_is_described():
    """The documentation is the fix for clause 3, so its absence is a failure.

    Checked on the Field descriptions a caller actually reads over RPC, not on
    the module docstring -- the reporter read the RPC schema.
    """
    import entry

    cls = entry.EntryApp.func_or_class
    for name in ("start_training", "preflight_training_dataset"):
        fn = getattr(cls, name)
        sig = inspect.signature(fn)
        default = sig.parameters["train_labels"].default
        text = str(getattr(default, "description", "") or "")
        assert "geojson" in text.lower(), f"{name} no longer advertises geojson: {text}"
        assert "top-left" in text.lower(), (
            f"{name}'s train_labels does not state the axis origin, which is the "
            f"one thing a caller cannot work out: {text}"
        )
        assert (
            "downward" in text.lower()
        ), f"{name}'s train_labels does not state the y direction: {text}"


# ── what the ids actually count, and where the convention is written ────────


def test_a_two_part_MultiPolygon_yields_TWO_instances():
    """The docstring said "one instance per Polygon/MultiPolygon feature". It is
    one per PART: the counter advances inside the part loop. A caller comparing
    their feature count against `n_instances` would read the difference as data
    loss, and a caller relying on one-id-per-feature gets two."""
    import numpy as np

    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [[[1, 1], [4, 1], [4, 4], [1, 4], [1, 1]]],
                        [[[6, 6], [9, 6], [9, 9], [6, 9], [6, 6]]],
                    ],
                },
            }
        ],
    }
    mask = training.rasterize_geojson(payload, width=12, height=12)
    ids = sorted(int(v) for v in np.unique(mask) if v)
    assert ids == [1, 2], f"a two-part MultiPolygon produced ids {ids}"


def test_the_rasteriser_docstring_agrees_with_what_the_ids_count():
    """Prose, so nothing else would catch it drifting back."""
    import inspect

    doc = inspect.getdoc(training.rasterize_geojson) or ""
    assert (
        "per POLYGON PART" in doc or "per polygon part" in doc.lower()
    ), f"the docstring does not say ids are per part: {doc[:200]!r}"
    assert "feature" in doc.lower(), "the docstring does not contrast part with feature"


def test_the_convention_is_also_in_the_README_api_section():
    """A caller reading the README's `start_training` entry — not the RPC schema
    — is the one who was bitten."""
    from pathlib import Path

    text = (Path(__file__).with_name("README.md")).read_text(encoding="utf-8")
    assert len(text) > 5000, f"README read implausibly short ({len(text)})"
    lowered = text.lower()
    assert "top-left" in lowered, "the README does not state the origin"
    assert "downward" in lowered, "the README does not state the y direction"
