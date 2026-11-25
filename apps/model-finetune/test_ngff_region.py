"""Training on a region must read the region, not the slide.

The whole point of importing a patch of a whole-slide source is that nothing ever
holds the slide: it has no file in the artifact to fetch, and materialising one
means reading level 0 whole in a browser, which a gigapixel source cannot
survive. So the arm that matters here is
``test_a_whole_slide_source_is_read_as_one_patch``: a store shaped like a real
slide, and an assertion that what came back is the patch and that nothing larger
was ever asked for. Without it every other arm is satisfied by an implementation
that downloads everything and crops afterwards.

The rest divides into the selection arithmetic (which axis is which, what a
channel axis does, what happens at the far edge) and the refusals, which exist
because this read runs inside an entry actor declared with 4 GiB and pinned to
one replica: a caller asking for a 20000x20000 patch would take the app down
rather than fail its own call.

No zarr, no network: ``read_region``'s two collaborators are injected as modules
for the one arm that needs them, with the save/restore done unconditionally so
the arm cannot leave a fake behind for another test to import.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import ngff_region as nr

_ABSENT = object()


# --- which axis is which ----------------------------------------------------


@pytest.mark.parametrize(
    "axes,rank,expected",
    [
        (["y", "x"], 2, {"y": 0, "x": 1, "c": None}),
        (["c", "y", "x"], 3, {"y": 1, "x": 2, "c": 0}),
        (["t", "c", "z", "y", "x"], 5, {"y": 3, "x": 4, "c": 1}),
        (["y", "x", "c"], 3, {"y": 0, "x": 1, "c": 2}),
    ],
)
def test_the_declared_axes_are_used_when_they_fit(axes, rank, expected):
    assert nr.axis_roles(axes, rank) == expected


def test_axes_that_do_not_fit_the_array_fall_back_to_the_last_two_dimensions():
    """Mirrors the broker's own fallback. A store whose declared axes are wrong
    is common enough that refusing would make real data unreadable, and Y/X being
    the two fastest-varying dimensions holds for every image layout in practice."""
    assert nr.axis_roles(["y", "x"], 4) == {"y": 2, "x": 3, "c": None}
    assert nr.axis_roles(["q", "w", "e"], 3) == {"y": 1, "x": 2, "c": None}


def test_an_array_with_no_plane_is_refused():
    with pytest.raises(nr.NgffRegionError, match="at least 2"):
        nr.axis_roles(["x"], 1)


# --- the selection ----------------------------------------------------------


def test_the_region_becomes_slices_on_y_and_x_only():
    sel = nr.region_selection(
        ["t", "c", "z", "y", "x"],
        (10, 3, 8, 2048, 2048),
        {"x": 100, "y": 200, "width": 512, "height": 256},
    )
    # t and z at one plane: a 2D annotation was drawn on one. c kept whole.
    assert sel == (0, slice(None), 0, slice(200, 456), slice(100, 612))


def test_a_source_with_no_region_reads_the_whole_plane():
    """A whole-image import records ``region: null`` — the broker normalises a
    region covering the store entirely to null, so one manifest entry means one
    thing. That must read the image, not refuse for lack of a region."""
    sel = nr.region_selection(["y", "x"], (64, 48), None)
    assert sel == (slice(0, 64), slice(0, 48))


def test_the_far_edge_is_clamped_and_the_origin_is_not():
    """The same asymmetry the broker applies when it normalises a region, for the
    same reason: a picker drawn on a scaled-down preview rounds the far edge a
    pixel past the end routinely, so refusing that makes the last patch of a
    slide unselectable — while an origin outside the store means the caller has
    the wrong coordinate system, and repairing it would train on a patch
    somewhere else entirely."""
    sel = nr.region_selection(
        ["y", "x"], (100, 100), {"x": 90, "y": 90, "width": 50, "height": 50}
    )
    assert sel == (slice(90, 100), slice(90, 100))

    with pytest.raises(nr.NgffRegionError, match="outside the 100x100"):
        nr.region_selection(
            ["y", "x"], (100, 100), {"x": 100, "y": 0, "width": 10, "height": 10}
        )
    with pytest.raises(nr.NgffRegionError, match="outside the image"):
        nr.region_selection(
            ["y", "x"], (100, 100), {"x": -1, "y": 0, "width": 10, "height": 10}
        )


def test_an_empty_or_malformed_region_is_refused():
    with pytest.raises(nr.NgffRegionError, match="empty"):
        nr.region_selection(
            ["y", "x"], (100, 100), {"x": 0, "y": 0, "width": 0, "height": 10}
        )
    with pytest.raises(nr.NgffRegionError, match="integer x, y, width and height"):
        nr.region_selection(["y", "x"], (100, 100), {"x": 0, "y": 0})


# --- the size bound ---------------------------------------------------------


def test_the_size_counts_channels_and_the_dtype():
    sel = nr.region_selection(
        ["c", "y", "x"], (3, 4096, 4096), {"x": 0, "y": 0, "width": 512, "height": 512}
    )
    assert nr.selection_bytes(sel, (3, 4096, 4096), "<u2") == 512 * 512 * 3 * 2
    assert nr.selection_bytes(sel, (3, 4096, 4096), "uint8") == 512 * 512 * 3
    assert nr.selection_bytes(sel, (3, 4096, 4096), np.dtype("float32")) == (
        512 * 512 * 3 * 4
    )


def test_an_unknown_dtype_counts_as_the_widest_supported():
    """A guess that UNDER-counts turns the cap into no cap, which is the one
    direction that matters: the bound exists to stop a read taking the actor
    down, so an unrecognised dtype must make the estimate bigger, not smaller."""
    sel = nr.region_selection(
        ["y", "x"], (4096, 4096), {"x": 0, "y": 0, "width": 512, "height": 512}
    )
    assert nr.selection_bytes(sel, (4096, 4096), "something-new") == 512 * 512 * 8


def test_the_bound_is_at_least_what_the_annotator_can_materialise():
    """The broker lets the browser materialise up to 64 MiB. A region small
    enough to materialise must always be trainable, or the two limits disagree
    about the same patch and a user is told to shrink something they were just
    allowed to import."""
    assert nr.MAX_REGION_BYTES >= 64 * 1024 * 1024


# --- the read plan ----------------------------------------------------------


def test_a_source_missing_its_read_plan_is_refused_by_name():
    """Refusing beats inferring: selecting the multiscale and the pyramid level
    from the store's own metadata already exists in the broker, and a copy here
    would be a third thing to keep in step."""
    with pytest.raises(nr.NgffRegionError, match="store_root"):
        nr.read_plan({"array_path": "0", "axes": ["y", "x"]})
    with pytest.raises(nr.NgffRegionError, match="array_path"):
        nr.read_plan({"store_root": "https://s.invalid/a.zarr", "axes": ["y", "x"]})
    with pytest.raises(nr.NgffRegionError, match="axes"):
        nr.read_plan({"store_root": "https://s.invalid/a.zarr", "array_path": "0"})


def test_the_plan_is_accepted_wrapped_or_bare():
    bare = {
        "store_root": "https://s.invalid/a.zarr/",
        "array_path": "0",
        "axes": ["y", "x"],
        "region": {"x": 1, "y": 2, "width": 3, "height": 4},
    }
    assert nr.read_plan(bare)["store_root"] == "https://s.invalid/a.zarr"
    assert nr.read_plan({"ngff": bare})["region"] == bare["region"]
    assert nr.read_plan(bare)["token"] is None


# --- the read itself --------------------------------------------------------


class _FakeArray:
    """An array that records what it was asked for and refuses to be copied."""

    def __init__(self, shape, dtype="<u2"):
        self.shape = shape
        self.dtype = dtype
        self.selections = []

    def __getitem__(self, selection):
        self.selections.append(selection)
        out = []
        for sel, size in zip(selection, self.shape):
            if isinstance(sel, slice):
                start, stop, _ = sel.indices(int(size))
                out.append(stop - start)
        return np.zeros(tuple(out), dtype="uint16")


def _with_fake_zarr(array, monkeypatch):
    """Inject the two modules ``read_region`` imports, and take them back out.

    Saved and restored UNCONDITIONALLY, including the absent case: a fake left in
    ``sys.modules`` is invisible here and breaks whichever test imports the real
    thing next, which is the order-dependence that cost this repo a day once
    already.
    """
    opened = {}

    zarr_mod = types.ModuleType("zarr")

    def open_group(store=None, mode="r", **kw):
        opened["store"] = store
        return {"0": array}

    zarr_mod.open_group = open_group

    store_mod = types.ModuleType("bioengine.datasets.http_zarr_store")

    class HttpZarrStore:
        def __init__(self, base_url, token=None, **kw):
            self.base_url = base_url
            self.token = token
            self.closed = False

        def close(self):
            self.closed = True

    store_mod.HttpZarrStore = HttpZarrStore

    for name, module in (
        ("zarr", zarr_mod),
        ("bioengine.datasets.http_zarr_store", store_mod),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return opened


def test_a_whole_slide_source_is_read_as_one_patch(monkeypatch):
    """THE ARM. A 100000x80000 slide, and a 512x512 patch out of it.

    The assertion is on what was ASKED for, not only on what came back: an
    implementation that fetched the array and sliced afterwards would return the
    same 512x512 and would be hopeless on the real thing.
    """
    array = _FakeArray((3, 80000, 100000))
    _with_fake_zarr(array, monkeypatch)

    out = nr.read_region(
        {
            "store_root": "https://bia.invalid/slide.zarr",
            "array_path": "0",
            "axes": ["c", "y", "x"],
            "region": {"x": 50000, "y": 40000, "width": 512, "height": 512},
        }
    )

    assert out.shape == (3, 512, 512)
    assert len(array.selections) == 1, "the array was indexed more than once"
    sel = array.selections[0]
    assert sel == (
        slice(None),
        slice(40000, 40512),
        slice(50000, 50512),
    ), "the read was not confined to the region"
    # Independent of the shape above: no axis was asked for in full except the
    # channel axis, which is 3 planes.
    assert all(
        not (isinstance(s, slice) and s.start == 0 and (s.stop or 0) >= 80000)
        for s in sel
    ), "an axis was read whole"


def test_a_region_over_the_bound_is_refused_before_it_is_read(monkeypatch):
    array = _FakeArray((1, 40000, 40000))
    _with_fake_zarr(array, monkeypatch)

    with pytest.raises(nr.NgffRegionError, match="over the"):
        nr.read_region(
            {
                "store_root": "https://bia.invalid/slide.zarr",
                "array_path": "0",
                "axes": ["c", "y", "x"],
                "region": {"x": 0, "y": 0, "width": 20000, "height": 20000},
            }
        )
    assert array.selections == [], "the oversized region was fetched before refusing"


def test_the_level_the_source_names_is_the_level_read(monkeypatch):
    """A source naming a level the store does not have must say so, not fall back
    to a different level: the region's coordinates are in the pixels of the level
    it was recorded against, so reading another one trains on the wrong crop."""
    array = _FakeArray((64, 64))
    _with_fake_zarr(array, monkeypatch)

    with pytest.raises(nr.NgffRegionError, match="has no array '3'"):
        nr.read_region(
            {
                "store_root": "https://bia.invalid/a.zarr",
                "array_path": "3",
                "axes": ["y", "x"],
                "region": None,
            }
        )


def test_the_credential_reaches_the_store_and_not_the_url(monkeypatch):
    """A protected source carries a token. It belongs in the Authorization
    header the store sends, never appended to a URL, where it would be logged by
    anything that logs a request."""
    array = _FakeArray((64, 64))
    opened = _with_fake_zarr(array, monkeypatch)

    nr.read_region(
        {
            "store_root": "https://s.invalid/a.zarr",
            "array_path": "0",
            "axes": ["y", "x"],
            "region": None,
            "token": "a-token",
        }
    )
    assert opened["store"].token == "a-token"
    assert "a-token" not in opened["store"].base_url


def test_the_store_is_closed_after_a_read(monkeypatch):
    """One store, and one HTTP client, is opened per read."""
    array = _FakeArray((64, 64))
    opened = _with_fake_zarr(array, monkeypatch)
    nr.read_region(
        {
            "store_root": "https://s.invalid/a.zarr",
            "array_path": "0",
            "axes": ["y", "x"],
            "region": None,
        }
    )
    assert opened["store"].closed is True


# --- the entry's dispatch ---------------------------------------------------


async def test_a_dict_image_source_is_read_as_a_region(monkeypatch):
    """``start_training`` takes arrays, URLs, upload paths — and now a region
    source. The dispatch is what connects the two, and nothing else covers it:
    every arm above calls ``read_region`` directly."""
    import entry

    seen = {}

    def _fake_read_region(source, max_bytes=nr.MAX_REGION_BYTES):
        seen["source"] = source
        return np.zeros((3, 32, 48), dtype="uint8")

    monkeypatch.setattr(nr, "read_region", _fake_read_region)
    inst = object.__new__(entry.EntryApp.func_or_class)
    source = {
        "store_root": "https://bia.invalid/slide.zarr",
        "array_path": "0",
        "axes": ["c", "y", "x"],
        "region": {"x": 10, "y": 20, "width": 48, "height": 32},
    }

    out = await entry.EntryApp.func_or_class._resolve_image(inst, source)
    assert seen["source"] == source, "the dict was not routed to the region reader"
    # HWC, because the caller immediately takes `shape[:2]` as (height, width) to
    # rasterize the geojson annotation against. Left in the store's CYX order
    # that would be (3, 32) and every region annotation would be rasterized at
    # the wrong size — silently, since both numbers are plausible.
    assert out.shape == (32, 48, 3)
    assert np.asarray(out).shape[:2] == (32, 48)


async def test_a_url_image_source_still_takes_the_url_path(monkeypatch):
    """CONTROL for the dispatch: adding the dict branch must not divert the
    string one, which is how every non-region image still arrives."""
    import entry

    calls = []

    async def _load(self, src):
        calls.append(src)
        return np.zeros((4, 4), dtype="uint8")

    monkeypatch.setattr(
        entry.EntryApp.func_or_class, "_load_image_from_source", _load, raising=True
    )
    inst = object.__new__(entry.EntryApp.func_or_class)
    out = await entry.EntryApp.func_or_class._resolve_image(
        inst, "https://x.invalid/a.png"
    )
    assert calls == ["https://x.invalid/a.png"]
    assert out.shape == (4, 4)
