"""A plane index is range-checked even when the axis has only one element.

`_plane_index` returned 0 for a size-1 axis BEFORE looking at what the caller
supplied, so `planes={"z": 7}` on a z of size 1 was accepted and silently read
z=0. The pixels were right — there is only one plane — but the error was lost: a
caller whose numbering is wrong got no complaint, and a stored record could claim
a plane that does not exist.

The same index on a 4-element axis was refused cleanly, which is how the
inconsistency was found and why it is a defect rather than a design note: the
check existed and was simply unreachable.

Measured on the code before this fix, roles {"y": 2, "x": 3, "c": 0},
axes ["c", "z", "y", "x"]:

    shape [3, 1, 64, 64]   planes {"z": 7}  -> ACCEPTED, z index 0
    shape [3, 4, 64, 64]   planes {"z": 7}  -> refused, "outside axis 'z'"

This file also pins the clause-3 decision: a plane named for a spatial or
channel axis, or for an axis the layout does not have, is REFUSED rather than
ignored. See `region_index` for the reasoning.
"""

import pytest

import ngff_region as nr

ROLES = {"y": 2, "x": 3, "c": 0}
AXES = ["c", "z", "y", "x"]
REGION = {"y": 0, "x": 0, "height": 8, "width": 8}

SINGLETON = [3, 1, 64, 64]  # z of size 1
STACK = [3, 4, 64, 64]  # z of size 4


def idx(shape, planes):
    return nr.region_index(ROLES, len(shape), shape, REGION, axes=AXES, planes=planes)


def z_of(index):
    """The z component of the returned index tuple (axes are c, z, y, x)."""
    return index[1]


# ── the control: the pair that revealed the inconsistency ───────────────────


def test_the_SIZE_4_axis_still_behaves_exactly_as_before():
    """THE PAIRED CONTROL the clause names. The fix must not change the
    behaviour that was already right, or "refused" below proves nothing about
    the singleton case specifically."""
    assert z_of(idx(STACK, {"z": 1})) == 1
    assert z_of(idx(STACK, {"z": 0})) == 0
    assert z_of(idx(STACK, {"z": 3})) == 3
    with pytest.raises(nr.RegionSourceError) as exc:
        idx(STACK, {"z": 7})
    assert "'z'" in str(exc.value) and "4 elements" in str(exc.value)


def test_a_SIZE_4_axis_with_NO_plane_is_still_refused():
    """The pre-existing rule, unchanged: an axis of size > 1 needs a plane."""
    with pytest.raises(nr.RegionSourceError) as exc:
        idx(STACK, None)
    assert "no plane was given" in str(exc.value)


# ── the defect: a supplied index on a size-1 axis ───────────────────────────


@pytest.mark.parametrize("bad", [1, 7, 2, 100, -1, -7])
def test_an_OUT_OF_RANGE_index_on_a_SINGLETON_axis_is_refused(bad):
    with pytest.raises(nr.RegionSourceError) as exc:
        idx(SINGLETON, {"z": bad})
    msg = str(exc.value)
    assert "'z'" in msg, msg
    assert str(bad) in msg, msg
    assert (
        "only 1 element" in msg and "only valid index is 0" in msg
    ), f"the message should state the valid range for a single-element axis: {msg}"


def test_the_ONLY_VALID_index_on_a_singleton_axis_is_still_accepted():
    """0 is in range, so it is accepted and used."""
    assert z_of(idx(SINGLETON, {"z": 0})) == 0


def test_a_singleton_axis_with_NO_plane_still_needs_no_choice():
    """Unchanged: size 1 is not a choice, so omitting the plane is fine."""
    assert z_of(idx(SINGLETON, None)) == 0
    assert z_of(idx(SINGLETON, {})) == 0


def test_the_singleton_and_the_stack_now_AGREE_on_an_impossible_index():
    """The point of the fix, stated as one assertion: the same bad index is
    refused on both, where before it was refused on one."""
    for shape in (SINGLETON, STACK):
        with pytest.raises(nr.RegionSourceError):
            idx(shape, {"z": 7})


# ── clause 3: a plane for a spatial or channel axis is REFUSED ──────────────


@pytest.mark.parametrize("axis,role", [("y", "y"), ("x", "x"), ("c", "c")])
def test_a_plane_for_a_SPATIAL_or_CHANNEL_axis_is_refused(axis, role):
    """DECIDED, not inherited. Those axes never reach the plane lookup — y and x
    are the crop, a channel axis is kept whole — so such a plane used to be
    accepted and discarded. Refusing matches the broker's import validator, so
    the two ends agree."""
    with pytest.raises(nr.RegionSourceError) as exc:
        idx(SINGLETON, {axis: 1})
    msg = str(exc.value)
    assert repr(axis) in msg, msg
    assert "not a plane axis" in msg, msg


def test_a_plane_for_an_UNKNOWN_axis_is_refused():
    """A plane that selects nothing looks like a plane that was honoured."""
    with pytest.raises(nr.RegionSourceError) as exc:
        idx(SINGLETON, {"bogus": 1})
    assert "names no axis" in str(exc.value)


def test_a_VALID_plane_alongside_nothing_else_still_works():
    """THE CONTROL for the two refusals above — they would read the same if the
    check refused every plane."""
    assert z_of(idx(SINGLETON, {"z": 0})) == 0
    assert z_of(idx(STACK, {"z": 2})) == 2


def test_a_valid_plane_is_refused_if_an_INVALID_one_rides_along():
    """All-or-nothing: a partly-honoured mapping is the silent case again."""
    with pytest.raises(nr.RegionSourceError):
        idx(STACK, {"z": 2, "c": 1})


# ── names absent: a non-empty `planes` is REFUSED, not quietly dropped ──────
#
# WHY THIS IS PARAMETRISED OVER THREE SHAPES rather than one. `_plane_index`
# reports unresolvable names only when some non-spatial axis has size > 1, so a
# four-element stack is the single shape that refuses on its own. On a
# single-element axis or a plain cyx store nothing fires, and without the
# explicit refusal a plane naming nothing is accepted and dropped:
#
#     axes=[], shape [3, 1, 64, 64]
#       {"z": 7}      -> index 0
#       {"c": 2}      -> index 0
#       {"bogus": 1}  -> index 0
#
# A test covering only the stack therefore passes whether or not the rule is
# implemented. The axes-absent and axes-empty cases are both covered because a
# stem imported before axis names were recorded produces the second, and that is
# the population the rule exists for.


@pytest.mark.parametrize(
    "shape", [SINGLETON, STACK, [3, 64, 64]], ids=["singleton", "stack", "cyx"]
)
@pytest.mark.parametrize("axes", [None, []], ids=["axes-none", "axes-empty"])
@pytest.mark.parametrize("planes", [{"z": 1}, {"z": 7}, {"c": 2}, {"bogus": 1}])
def test_a_plane_WITHOUT_axis_names_is_refused(shape, axes, planes):
    """Every combination, because the gap was shape-specific and a single shape
    is what hid it."""
    roles = ROLES if len(shape) == 4 else {"y": 1, "x": 2, "c": 0}
    with pytest.raises(nr.RegionSourceError) as exc:
        nr.region_index(roles, len(shape), shape, REGION, axes=axes, planes=planes)
    msg = str(exc.value)
    assert "axis names are not recorded" in msg, msg
    assert "refused rather than ignored" in msg, msg


@pytest.mark.parametrize(
    "shape", [SINGLETON, STACK, [3, 64, 64]], ids=["singleton", "stack", "cyx"]
)
def test_NO_planes_without_axis_names_is_still_handled_as_before(shape):
    """THE CONTROL. The refusal above must be about the planes, not about the
    names being absent — a legacy record that asks for nothing still behaves
    exactly as it did: a singleton or cyx store resolves, and a size-4 axis is
    refused for the pre-existing reason."""
    roles = ROLES if len(shape) == 4 else {"y": 1, "x": 2, "c": 0}
    if shape is STACK:
        with pytest.raises(nr.RegionSourceError) as exc:
            nr.region_index(roles, len(shape), shape, REGION, axes=[], planes=None)
        assert "no plane was given" in str(exc.value)
    else:
        idx = nr.region_index(roles, len(shape), shape, REGION, axes=[], planes=None)
        assert len(idx) == len(shape)


def test_the_refusal_no_longer_claims_NO_PLANE_WAS_GIVEN_when_one_was():
    """The minor the review also named: with names absent and a plane supplied,
    the old path reported "no plane was given", which is the opposite of true."""
    with pytest.raises(nr.RegionSourceError) as exc:
        nr.region_index(ROLES, 4, STACK, REGION, axes=[], planes={"z": 1})
    assert "no plane was given" not in str(exc.value), str(exc.value)
