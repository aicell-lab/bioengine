"""An annotation that draws nothing must fail loudly, not train on background.

The defect this closes was raised in review as unverified, and it stays unverified
on purpose: nobody has settled whether the annotator writes polygon coordinates in
a crop's frame or the whole source's. What IS certain is the outcome when they
disagree. A crop read from a whole-slide source is a WxH canvas, the broker
describes a region "in level-0 pixels", and if the polygons are in source
coordinates every one of them lands outside the canvas. The mask comes back all
zeros, training runs, and it reports success having learnt only background.

So rather than pick a frame and call it a contract -- which would move the silence
rather than remove it -- the empty outcome is refused. An all-background mask is
never a legitimate training target, so this costs nothing real whichever frame
turns out to be in use, and the refusal names the coordinate span so the mismatch
is readable at a glance instead of being inferred from a useless model.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import entry
import training


def _poly(x0, y0, x1, y1):
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
        },
    }


def _fc(*features):
    return {"type": "FeatureCollection", "features": list(features)}


# ── the bounds helper, which makes the refusal diagnostic ────────────────────


def test_bounds_span_every_polygon():
    b = training.polygon_coordinate_bounds(
        _fc(_poly(10, 20, 30, 40), _poly(5, 6, 7, 8))
    )
    assert (b["x_min"], b["x_max"], b["y_min"], b["y_max"]) == (5, 30, 6, 40)
    assert b["features"] == 2


def test_bounds_walk_into_a_multipolygon():
    """MultiPolygon nests one level deeper than Polygon. The walk descends until
    it finds a coordinate pair rather than trusting the declared type, so a
    writer that nests differently still reports real numbers."""
    mp = {
        "type": "Feature",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [[[[1, 2], [3, 2], [3, 4], [1, 4], [1, 2]]]],
        },
    }
    b = training.polygon_coordinate_bounds(_fc(mp))
    assert (b["x_min"], b["x_max"], b["y_min"], b["y_max"]) == (1, 3, 2, 4)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"features": "not a list"},
        _fc(),
        _fc({"type": "Feature", "geometry": {"type": "Point", "coordinates": [1, 2]}}),
        _fc({"type": "Feature"}),
    ],
)
def test_no_polygons_means_no_bounds(payload):
    """``None``, not a zero-size box. The refusal keys on bounds existing, so a
    collection with nothing to draw must NOT be reported as a frame mismatch —
    it is a different problem with a different answer."""
    assert training.polygon_coordinate_bounds(payload) is None


# ── the refusal, through the real method ────────────────────────────────────


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Http:
    def __init__(self, payload):
        self._payload = payload
        self.gets = []

    async def get(self, url):
        self.gets.append(url)
        return _Resp(self._payload)


def _entry_with(payload):
    inst = object.__new__(entry.EntryApp.func_or_class)
    inst._http = _Http(payload)
    return inst


def _resolve(inst, shape):
    method = entry.EntryApp.func_or_class._resolve_label_array
    return asyncio.run(method(inst, "https://example.invalid/ann.geojson", shape))


def test_polygons_outside_the_crop_are_REFUSED_with_their_span():
    """THE case. A 512x512 crop of a slide, annotated in slide coordinates."""
    inst = _entry_with(_fc(_poly(40000, 30000, 40100, 30100)))
    with pytest.raises(ValueError) as err:
        _resolve(inst, (512, 512))
    msg = str(err.value)
    assert "empty mask" in msg
    assert "512x512" in msg
    assert "40000" in msg and "40100" in msg, f"the span is not in the message: {msg}"
    assert "region origin" in msg, "the likely cause is not named"


def test_polygons_inside_the_crop_rasterise_and_are_returned():
    """The control. Without it, a method that refused everything would pass the
    test above for the wrong reason."""
    inst = _entry_with(_fc(_poly(10, 10, 40, 40)))
    mask = _resolve(inst, (512, 512))
    assert mask.shape == (512, 512)
    assert mask.max() > 0, "a polygon inside the canvas drew nothing"
    assert int(mask.max()) == 1, "one polygon should be one instance id"


def test_an_annotation_with_no_polygons_is_not_treated_as_a_mismatch():
    """An empty FeatureCollection draws nothing for a different reason, and
    telling the user to offset by a region origin would be wrong advice."""
    inst = _entry_with(_fc())
    mask = _resolve(inst, (64, 64))
    assert mask.shape == (64, 64)
    assert mask.max() == 0


def test_a_partly_visible_polygon_is_accepted():
    """A cell clipped by the crop edge is ordinary, not a mismatch: it draws
    something, so the guard must stay out of the way."""
    inst = _entry_with(_fc(_poly(-20, -20, 30, 30)))
    mask = _resolve(inst, (64, 64))
    assert mask.max() > 0
