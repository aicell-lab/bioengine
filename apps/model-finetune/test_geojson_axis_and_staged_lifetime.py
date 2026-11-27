"""The two documented facts from the cpsam/HPA report, pinned as arms.

A docstring that states a convention is a claim. These arms are what make it a
measurement, and what makes a future flip of either one go red instead of
silently changing what users are told.
"""

import json

import numpy as np
import pytest


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


def test_geojson_y_runs_DOWNWARD_from_the_TOP_LEFT_not_upward():
    """The convention the docstrings now state, measured rather than asserted.

    A polygon in the TOP of the image must rasterise into the TOP ROWS. Under a
    y-UP reading (origin bottom-left, which several GIS and plotting tools emit)
    the same coordinates land in the BOTTOM rows instead -- vertically mirrored,
    with every polygon still inside the image, so no bounds check fires and
    training proceeds on wrong masks. That silence is why the direction has to be
    written down and pinned.

    The two halves are checked against each other rather than just asserting the
    top is non-empty: "it drew something somewhere" would pass under either
    convention.
    """
    import training

    H = W = 64
    # y from 4..16 -> the TOP under the raster convention.
    mask = training.rasterize_geojson(
        _fc([[8, 4], [56, 4], [56, 16], [8, 16], [8, 4]]), width=W, height=H
    )
    assert mask.shape == (H, W), mask.shape
    top = int((mask[: H // 2] != 0).sum())
    bottom = int((mask[H // 2 :] != 0).sum())
    assert top > 0, "the polygon rasterised to nothing at all"
    assert bottom == 0, (
        f"y is being read UPWARD: a polygon at y=4..16 landed in the bottom half "
        f"({bottom} px) instead of the top ({top} px). The documented convention "
        f"is origin top-left, y increasing downward."
    )


def test_geojson_instance_ids_start_at_one_in_DOCUMENT_order():
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
    mask = training.rasterize_geojson(fc, width=64, height=64)
    ids = sorted(int(v) for v in np.unique(mask) if v != 0)
    assert ids == [1, 2], ids
    # The FIRST feature must be id 1 -- document order, not position order.
    assert int(mask[5, 5]) == 1, int(mask[5, 5])
    assert int(mask[25, 25]) == 2, int(mask[25, 25])


def test_the_EXPIRY_error_tells_the_caller_to_RE_STAGE():
    """The expiry message must say what to DO, not only that something is gone.

    The reported failure was `Source 'temp/<uuid>.png' does not exist or has
    expired.` after 74 minutes. That is accurate and unactionable: it does not
    say the staged OBJECT expires (the docstring's "1-hour TTL" reads as a limit
    on uploading), nor that re-staging is the fix, nor that a queued run is the
    case most likely to hit it.

    Asserted on the SOURCE of the raiser rather than by driving a live S3 miss,
    which would need a cluster and an hour.
    """
    import ast
    import pathlib

    src = (pathlib.Path(__file__).with_name("entry.py")).read_text()

    # LOCATED BY AST, NOT BY A REGEX OVER THE FILE. A first version searched the
    # whole source for the message text and matched the copy I had just put in
    # `get_upload_url`'s DOCSTRING while documenting this very behaviour -- so
    # the arm inspected the prose instead of the raiser and failed on a word
    # that was only missing from the docstring. My own documentation shadowed my
    # scan; the AST walks to the `raise` itself.
    found = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        call = node.exc
        if not (
            isinstance(call, ast.Call)
            and getattr(call.func, "id", "") == "FileNotFoundError"
        ):
            continue
        text = " ".join(
            v.value
            for a in call.args
            for v in ast.walk(a)
            if isinstance(v, ast.Constant) and isinstance(v.value, str)
        )
        if "does not exist or has expired" in text:
            found.append(text)

    assert len(found) == 1, (
        f"expected exactly one expiry raiser, found {len(found)}; a second one "
        f"would mean two messages can drift apart"
    )
    msg = found[0].lower()
    for phrase in ("re-stage", "about an hour", "queue"):
        assert phrase in msg, (
            f"the expiry error no longer mentions {phrase!r}, so a caller cannot "
            f"act on it. The message is: {found[0]!r}"
        )
