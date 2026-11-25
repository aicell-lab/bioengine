"""A region annotation records which slice it was drawn on, and a bad one is refused here.

Why this is checked at import and not only at read time: the reader already
refuses an axis of size greater than one with no plane recorded, and that
refusal protects the PIXELS. It cannot protect the RECORD. A plane naming an
axis the store does not have, or an index past its end, would otherwise be
written into the image manifest and surface only when somebody trained on it —
reported as a training failure, long after the annotation was saved, with
nothing pointing back at the import that was wrong.

The ruling this implements: the annotator records the plane; a frontend
convention is not promoted to a contract. An explicit record is checkable and
survives a change to the viewer's default slice, and annotations saved without
one stay refused rather than being guessed at.
"""

from __future__ import annotations

import pathlib

import pytest

import broker_core as core


# ── the shape of a real multi-plane store ─────────────────────────────────────
TCZYX = ["t", "c", "z", "y", "x"]
SHAPE = [1, 3, 9, 1000, 2000]        # z has NINE slices: the case that must be recorded


def test_no_planes_is_None_not_an_empty_mapping():
    """An absent plane and an empty one must not mean two different things to a
    consumer, so both normalise to None."""
    for empty in (None, {}, 0, ""):
        assert core.normalise_planes(empty, TCZYX, SHAPE) is None


def test_a_named_plane_is_recorded_as_given():
    assert core.normalise_planes({"z": 4}, TCZYX, SHAPE) == {"z": 4}
    assert core.normalise_planes({"z": 0, "t": 0}, TCZYX, SHAPE) == {"z": 0, "t": 0}


def test_the_last_valid_index_is_accepted_and_the_next_one_is_not():
    """The boundary, both sides. An off-by-one here is the whole bug class."""
    assert core.normalise_planes({"z": 8}, TCZYX, SHAPE) == {"z": 8}
    with pytest.raises(ValueError) as e:
        core.normalise_planes({"z": 9}, TCZYX, SHAPE)
    assert "0..8" in str(e.value), str(e.value)


@pytest.mark.parametrize("bad", [-1, 9, 100])
def test_an_index_outside_the_axis_is_REFUSED_naming_the_range(bad):
    with pytest.raises(ValueError) as e:
        core.normalise_planes({"z": bad}, TCZYX, SHAPE)
    msg = str(e.value)
    assert "'z'" in msg and "9 element" in msg, msg


def test_an_axis_the_store_does_not_have_is_REFUSED_naming_the_axes():
    """A caller sending 'slice' or 'Z' has the wrong vocabulary, and silently
    dropping the key would record an annotation with no plane at all — which the
    reader would then refuse far away from here."""
    for name in ("slice", "Z", "depth", "zz"):
        with pytest.raises(ValueError) as e:
            core.normalise_planes({name: 1}, TCZYX, SHAPE)
        assert name in str(e.value) and "does not have" in str(e.value)


@pytest.mark.parametrize("bad", ["two", "4", None, [1], {"a": 1}, 1.5, True, False])
def test_a_non_integer_index_is_REFUSED(bad):
    """Strictly, and `1.5` is the one that matters: `int(1.5)` is 1, so a
    permissive validator turns a fractional index into slice 1 with no
    complaint — the wrong-slice bug this field exists to prevent, arriving by a
    different route. `True` is an int in Python and would become slice 1 too.
    `"4"` is refused rather than coerced: a caller sending a string has a bug,
    and silently parsing it hides the bug instead of the index."""
    with pytest.raises(ValueError):
        core.normalise_planes({"z": bad}, TCZYX, SHAPE)


def test_an_integral_float_is_accepted_because_JSON_has_one_number_type():
    """4.0 over the wire is 4. Refusing it would break a caller that is right."""
    assert core.normalise_planes({"z": 4.0}, TCZYX, SHAPE) == {"z": 4}


def test_a_plane_on_a_source_with_no_recorded_layout_is_REFUSED():
    """Not silently accepted. Without axis names and a shape there is nothing to
    check an index against, and a record that cannot be checked is exactly what
    this validator exists to keep out of the manifest."""
    for axes, shape in ((None, SHAPE), (TCZYX, None), ([], []), (None, None)):
        with pytest.raises(ValueError) as e:
            core.normalise_planes({"z": 1}, axes, shape)
        assert "re-import" in str(e.value) or "does not describe itself" in str(e.value)


def test_a_layout_that_contradicts_itself_is_REFUSED():
    """Axis names and shape of different lengths cannot be zipped, and guessing
    the alignment is how an index lands on the wrong axis."""
    with pytest.raises(ValueError) as e:
        core.normalise_planes({"z": 1}, TCZYX, [1, 3, 9])
    assert "does not describe itself" in str(e.value)


def test_planes_are_not_required_for_a_singleton_axis():
    """The common case: a slide scanner's pyramid is 5-D with t and z of size 1,
    and must not be made to carry a plane it has no choice about."""
    singleton = [1, 3, 1, 1000, 2000]
    assert core.normalise_planes(None, TCZYX, singleton) is None
    assert core.normalise_planes({"z": 0}, TCZYX, singleton) == {"z": 0}
    with pytest.raises(ValueError):
        core.normalise_planes({"z": 1}, TCZYX, singleton)


# ── the entry actually written into the manifest ─────────────────────────────


def _entry(planes):
    return core.remote_image_entry(
        "https://host/img.zarr/",
        None,
        200,
        150,
        dtype="|u1",
        planes=planes,
        layout={"dataset_paths": ["0"], "axis_roles": {"y": 3, "x": 4, "c": 1},
                "level0_shape": SHAPE, "axes": TCZYX},
    )


def test_the_plane_reaches_the_manifest_entry_under_source():
    """Where the reader looks. If this key moves, training stops resolving the
    slice and starts refusing every multi-plane source again."""
    e = _entry({"z": 4})
    assert e["source"]["planes"] == {"z": 4}


def test_an_entry_without_a_plane_carries_no_planes_key_at_all():
    """Rather than a null or an empty object: the reader distinguishes 'nothing
    recorded' from 'recorded as nothing', and only the first is refusable."""
    e = _entry(None)
    assert "planes" not in e["source"]


def _import_ngff_image_body() -> str:
    """The text of `import_ngff_image` from the FILE, not from `inspect`.

    `inspect.getsource` on this method is unreliable: the class is wrapped by
    Ray Serve's `@deployment` and the methods by the app framework, so the
    lookup can land on a wrapper and return a few hundred characters of
    decorator instead of the method. That failure is invisible — the
    assertions below would pass against text that is not the function — which
    is why this reads the source file and slices the function out by name,
    then proves the slice is the right one.
    """
    src = (pathlib.Path(__file__).parent / "broker.py").read_text()
    start = src.find("    async def import_ngff_image(")
    assert start > 0, "import_ngff_image not found in broker.py"
    nxt = src.find("\n    async def ", start + 1)
    body = src[start : nxt if nxt > 0 else len(src)]
    assert len(body) > 2000, f"slice is implausibly short ({len(body)} chars)"
    assert body.startswith("    async def import_ngff_image("), body[:60]
    # One METHOD, checked at the method indent. The body legitimately contains
    # nested `async def`s, so counting the bare keyword would be wrong -- it is
    # 2 here, and asserting 1 made both tests in this file fail against correct
    # code until the slice was printed.
    assert body.count("\n    async def ") == 0, "the slice spans more than one method"
    return body


def test_the_import_path_validates_before_writing():
    """THE WIRING. The validator and the parameter are each correct alone; this
    pins that `import_ngff_image` actually calls one and passes the other, which
    is the only reason either matters."""
    body = _import_ngff_image_body()
    assert "planes: Optional[Dict[str, int]] = Field(" in body, (
        "import_ngff_image does not expose a `planes` parameter at all"
    )
    assert "core.normalise_planes(planes" in body, (
        "import_ngff_image does not validate planes; a plane naming a missing "
        "axis or an out-of-range index would reach the manifest and fail only "
        "when somebody trained on it"
    )
    assert "planes=planes" in body, (
        "import_ngff_image does not pass planes to remote_image_entry, so the "
        "parameter would be accepted over RPC and silently dropped"
    )


def test_the_wiring_check_can_actually_fail():
    """THE CONTROL on the test above. A text assertion over a slice is only
    worth having if a missing call makes it fail, so this removes each call from
    a COPY of the body and asserts the check would reject it."""
    body = _import_ngff_image_body()
    for needle in ("core.normalise_planes(planes", "planes=planes"):
        assert needle in body
        assert needle not in body.replace(needle, "", 1), (
            f"removing {needle!r} from the body did not remove it, so the "
            f"assertion keyed on it cannot fail"
        )
