"""Region reading: the arithmetic, and the refusals that keep a crop honest.

These tests never touch a store. ``read_region`` takes its opener as an argument
precisely so the index arithmetic can be exercised here, and the one test that
reads an array hands it a numpy array instead of a Zarr group.

That is not a convenience. CI runs ``pytest apps/model-finetune`` inside
``bioengine-worker`` after installing ``requirements-test.txt`` and the datasets
requirements — NOT ``requirements-entry.txt``, where zarr lives. A module-level
``import zarr`` in the module under test would therefore turn this whole file
into collection ERRORS in CI, which reads as "no failures" to anything counting
failures. ``test_the_module_does_not_import_zarr_at_module_scope`` pins that.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import ngff_region as nr

HERE = Path(__file__).resolve().parent

# The broker's axis roles for the two layouts that matter: a plain 2D image, and
# a 5D tczyx whole-slide store, which is what an OME-Zarr pyramid from a scanner
# actually looks like.
ROLES_2D = {"y": 0, "x": 1, "c": None}
ROLES_TCZYX = {"y": 3, "x": 4, "c": 1}


# ── what is and is not a region reference ────────────────────────────────────


@pytest.mark.parametrize(
    "source",
    [
        "https://example.invalid/image.tif",
        "images/foo.png",
        {"store_root": "https://x/y/"},  # no kind
        {"kind": "ngff"},  # no store_root
        {"kind": "ngff", "store_root": "   "},  # blank store_root
        {"kind": "local", "store_root": "https://x/y/"},
        None,
    ],
)
def test_a_non_region_source_falls_through(source):
    """The existing file path must be untouched for everything it used to take.

    This is the discrimination that keeps the change additive: if
    ``is_region_source`` said yes to a URL, every current caller would be routed
    into a reader that cannot serve them.
    """
    assert nr.is_region_source(source) is False


def test_a_region_reference_is_recognised_and_normalised():
    ref = nr.parse_region_source(
        {
            "kind": "ngff",
            "store_root": " https://example.invalid/zarr/slide/ ",
            "multiscale_path": None,
            "region": {"x": "10", "y": 20, "width": 64, "height": 32, "extra": 1},
        }
    )
    assert ref["store_root"] == "https://example.invalid/zarr/slide/"
    assert ref["region"] == {"x": 10, "y": 20, "width": 64, "height": 32}


def test_a_whole_image_reference_keeps_region_none():
    """``None`` means the whole image, which is how the broker records it: its
    ``normalise_region`` returns None when the box covers the store. Inventing a
    full-extent box here would lose that distinction."""
    ref = nr.parse_region_source({"kind": "ngff", "store_root": "https://x/y/"})
    assert ref["region"] is None


@pytest.mark.parametrize(
    "region, expected",
    [
        ({"x": 0, "y": 0, "width": 8}, "missing height"),
        ({"x": 0, "y": 0, "width": 0, "height": 8}, "non-positive"),
        ({"x": -1, "y": 0, "width": 8, "height": 8}, "negative"),
        ("not a mapping", "must be a mapping"),
    ],
)
def test_a_malformed_region_is_refused_with_the_reason(region, expected):
    with pytest.raises(nr.RegionSourceError) as err:
        nr.parse_region_source(
            {"kind": "ngff", "store_root": "https://x/y/", "region": region}
        )
    assert expected.split()[-1] in str(err.value)


# ── the index arithmetic ─────────────────────────────────────────────────────


def test_a_2d_store_selects_exactly_the_box():
    idx = nr.region_index(
        ROLES_2D, 2, [100, 200], {"x": 10, "y": 20, "width": 30, "height": 40}
    )
    assert idx == (slice(20, 60), slice(10, 40))


def test_a_5d_store_drops_t_and_z_and_keeps_channels():
    """THE case a whole-slide source needs. t and z are reduced to their first
    element; c is kept whole, because dropping it would turn an RGB crop into one
    band without saying so."""
    idx = nr.region_index(
        ROLES_TCZYX,
        5,
        [1, 3, 1, 1000, 2000],
        {"x": 5, "y": 7, "width": 11, "height": 13},
    )
    assert idx == (0, slice(None), 0, slice(7, 20), slice(5, 16))


def test_no_region_reads_the_whole_level():
    idx = nr.region_index(ROLES_2D, 2, [10, 20], None)
    assert idx == (slice(None), slice(None))


def test_a_region_past_the_edge_is_REFUSED_not_clamped():
    """A clamped crop would be a different area from the one annotated, and the
    label mask is rasterised to the region's declared size, so a shrunk image
    would no longer match its labels. Refusing is the only honest answer."""
    with pytest.raises(nr.RegionSourceError) as err:
        nr.region_index(
            ROLES_2D, 2, [100, 100], {"x": 90, "y": 0, "width": 20, "height": 10}
        )
    msg = str(err.value)
    assert "runs past" in msg and "clamped" in msg


def test_a_probe_that_disagrees_with_itself_is_refused():
    """rank and shape come from the same probe; if they disagree, one of them is
    wrong and there is no way to tell which."""
    with pytest.raises(nr.RegionSourceError, match="do not match|does not match"):
        nr.region_index(ROLES_2D, 3, [10, 20], None)


def test_axis_roles_pointing_outside_the_array_are_refused():
    with pytest.raises(nr.RegionSourceError, match="outside an array"):
        nr.region_index({"y": 0, "x": 7, "c": None}, 2, [10, 20], None)


# ── which plane, for an axis that has more than one ──────────────────────────
#
# This block exists because of a measured defect, not a hypothetical one. The
# annotator displays the MIDDLE slice of a z-stack, this module used to take
# index 0 for every non-spatial axis, and nothing downstream could tell: the
# mask rasterises non-empty, the shapes agree, and the run reports success
# having trained polygons from one plane against pixels from another.

BOX = {"x": 5, "y": 7, "width": 11, "height": 13}
AXES_TCZYX = ["t", "c", "z", "y", "x"]


def test_a_multi_slice_z_axis_with_no_plane_is_REFUSED():
    """THE case. A 9-slice stack, annotated on whichever slice the viewer showed.

    The refusal has to name the axis, its size, and that 0 is not assumed —
    a message saying only "cannot read" would send the reader looking at the
    store rather than at the missing plane.
    """
    with pytest.raises(nr.RegionSourceError) as err:
        nr.region_index(ROLES_TCZYX, 5, [1, 3, 9, 1000, 2000], BOX, axes=AXES_TCZYX)
    msg = str(err.value)
    assert "'z'" in msg, f"the axis is not named: {msg}"
    assert "9 elements" in msg, f"the size is not given: {msg}"
    assert "Index 0 is NOT" in msg, f"the wrong default is not ruled out: {msg}"
    assert "middle slice" in msg, f"the annotator's convention is not named: {msg}"
    assert "planes=" in msg, f"the repair is not stated: {msg}"


def test_a_named_plane_is_taken():
    """The control for the refusal above: with the plane given, the SAME call
    reads the slice that was asked for."""
    idx = nr.region_index(
        ROLES_TCZYX, 5, [1, 3, 9, 1000, 2000], BOX, axes=AXES_TCZYX, planes={"z": 4}
    )
    assert idx == (0, slice(None), 4, slice(7, 20), slice(5, 16))


def test_a_singleton_axis_needs_no_plane():
    """Not a choice, so not a refusal. Most OME-Zarr stores from a slide scanner
    are 5D with t and z of size 1, and demanding a plane for them would make the
    common case unreadable for nothing."""
    idx = nr.region_index(ROLES_TCZYX, 5, [1, 3, 1, 1000, 2000], BOX, axes=AXES_TCZYX)
    assert idx == (0, slice(None), 0, slice(7, 20), slice(5, 16))


def test_a_plane_outside_its_axis_is_refused_with_the_range():
    with pytest.raises(nr.RegionSourceError) as err:
        nr.region_index(
            ROLES_TCZYX,
            5,
            [1, 3, 9, 1000, 2000],
            BOX,
            axes=AXES_TCZYX,
            planes={"z": 9},
        )
    assert "0..8" in str(err.value)


def test_a_multi_slice_axis_with_no_recorded_NAME_is_refused_saying_so():
    """A stem imported before axis names were recorded reaches here with no
    names. It must still refuse — the ambiguity is identical — but the repair is
    different, so the message has to be too: no plane can be named for an axis
    whose name is unknown."""
    with pytest.raises(nr.RegionSourceError) as err:
        nr.region_index(ROLES_TCZYX, 5, [1, 3, 9, 1000, 2000], BOX)
    msg = str(err.value)
    assert "at position 2" in msg, f"the axis is not located: {msg}"
    assert "re-import" in msg, f"the repair is not stated: {msg}"


def test_axes_that_disagree_with_the_rank_are_refused():
    """``axes`` and ``shape`` come from one probe. If they describe arrays of
    different rank, neither can be trusted to say which axis is z."""
    with pytest.raises(nr.RegionSourceError, match="different array"):
        nr.region_index(ROLES_2D, 2, [10, 20], None, axes=["t", "y", "x"])


def test_a_plane_survives_parse_and_a_bad_one_does_not():
    ref = nr.parse_region_source(
        {"kind": "ngff", "store_root": "https://x/y/", "planes": {"z": 4}}
    )
    assert ref["planes"] == {"z": 4}
    assert (
        nr.parse_region_source({"kind": "ngff", "store_root": "https://x/y/"})["planes"]
        is None
    ), "an absent planes mapping must stay None rather than becoming {}"


@pytest.mark.parametrize(
    "planes, expected",
    [
        ("z=4", "mapping"),
        ({"z": "4"}, "not an integer"),
        ({"z": 1.5}, "not an integer"),
        ({"z": True}, "not an integer"),  # bool is an int in Python; it is not an index
        ({"z": -1}, "negative"),
    ],
)
def test_a_malformed_planes_mapping_is_refused(planes, expected):
    with pytest.raises(nr.RegionSourceError, match=expected):
        nr.parse_region_source(
            {"kind": "ngff", "store_root": "https://x/y/", "planes": planes}
        )


def test_the_plane_is_refused_BEFORE_the_store_is_opened():
    """A wrong-plane reference must cost no requests. The opener here raises if
    it is called at all, so a refusal that happened after the open would fail
    with that error instead of the plane one."""

    def _never(_root, _path):
        raise AssertionError("the store was opened despite an unresolvable plane")

    with pytest.raises(nr.RegionSourceError, match="no plane was given"):
        nr.read_region(
            {"kind": "ngff", "store_root": "https://x/y/", "region": BOX},
            {
                "dataset_paths": ["0"],
                "axis_roles": ROLES_TCZYX,
                "level0_shape": [1, 3, 9, 1000, 2000],
                "axes": AXES_TCZYX,
            },
            opener=_never,
        )


# ── layout normalisation ─────────────────────────────────────────────────────


def test_a_channel_first_crop_comes_back_channel_last():
    """Both trainers read HxW or HxWxC. A tczyx store yields (c, y, x)."""
    arr = np.zeros((3, 13, 11), dtype=np.uint8)
    arr[2, 0, 0] = 9  # a marker that must survive the transpose
    out = nr.to_hw_or_hwc(arr, ROLES_TCZYX, 5)
    assert out.shape == (13, 11, 3)
    assert out[0, 0, 2] == 9


def test_a_2d_crop_is_passed_through_unchanged():
    arr = np.arange(6, dtype=np.uint8).reshape(2, 3)
    out = nr.to_hw_or_hwc(arr, ROLES_2D, 2)
    assert out.shape == (2, 3)
    assert np.array_equal(out, arr)


def test_level0_is_taken_from_the_probe_not_guessed():
    """A store may name its levels anything; several in the wild use 's0', so
    hardcoding '0' would fail on them."""
    assert nr.level0_path(["s0", "s1", "s2"]) == "s0"
    assert nr.level0_path(["/0/", "1"]) == "0"
    with pytest.raises(nr.RegionSourceError, match="no dataset paths"):
        nr.level0_path([])


# ── end to end, with the store faked ────────────────────────────────────────


def test_read_region_reads_only_the_box_and_records_what_it_asked_for():
    """The whole point, measured: the opener is handed the level-0 path, and the
    values returned are the box's, not the array's."""
    full = np.arange(100 * 200, dtype=np.uint16).reshape(100, 200)
    asked = {}

    def fake_opener(store_root, array_path):
        asked["store_root"] = store_root
        asked["array_path"] = array_path
        return full

    out = nr.read_region(
        {
            "kind": "ngff",
            "store_root": "https://example.invalid/zarr/slide",
            "region": {"x": 10, "y": 20, "width": 5, "height": 4},
        },
        probe={
            "dataset_paths": ["s0", "s1"],
            "axis_roles": ROLES_2D,
            "level0_shape": [100, 200],
        },
        opener=fake_opener,
    )
    assert asked["array_path"] == "s0"
    assert asked["store_root"] == "https://example.invalid/zarr/slide"
    assert out.shape == (4, 5)
    assert np.array_equal(out, full[20:24, 10:15])


def test_read_region_refuses_a_probe_without_the_layout():
    """A broker that has not been updated returns width/height/dtype/levels/
    channels/content_hint and no layout. Reading that as "no channels, 2D" would
    invent an answer; it is refused by name instead."""
    with pytest.raises(nr.RegionSourceError, match="dataset_paths"):
        nr.read_region(
            {"kind": "ngff", "store_root": "https://x/y/"},
            probe={"width": 10, "height": 10, "dtype": "uint8", "levels": 1},
            opener=lambda *a: np.zeros((10, 10)),
        )


# ── the CI guard ─────────────────────────────────────────────────────────────


def test_the_module_does_not_import_zarr_at_module_scope():
    """If zarr is imported at module scope, this file ERRORS at collection in CI
    rather than failing, because CI does not install requirements-entry.txt.

    Parsed over the AST, not grepped: the module's own docstring names zarr
    several times, and a text search would match that prose — the mistake a
    guard in this app made once before. The control below proves the walk sees
    real imports, so a finder that matched nothing cannot pass this vacuously.
    """
    tree = ast.parse((HERE / "ngff_region.py").read_text(encoding="utf-8"))
    top_level = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.append(node.module.split(".")[0])
    assert "zarr" not in top_level, (
        "zarr is imported at module scope; CI cannot install it, so this file "
        "would error at collection instead of running"
    )
    assert "numpy" not in top_level, (
        "numpy at module scope is unnecessary here and couples the pure "
        "functions to an import the arithmetic does not need"
    )
    # CONTROL: the walk must see the imports that ARE there.
    assert "typing" in top_level, "the AST walk found no imports at all"
