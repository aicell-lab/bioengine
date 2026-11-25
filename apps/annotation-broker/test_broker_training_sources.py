"""Which images can feed a training run.

Training used to need a materialised file, so every remote image was refused — which
is what made a whole-slide source untrainable, since materialising one means the
browser reading its full level 0. A consumer that can read a region out of the store
makes most of those refusals wrong, and these tests pin the narrowing: the control
compares the new set against the old one and asserts a remote-with-a-source stem has
moved out of it. Without that comparison, "the refusal is narrower" would be a claim
about code rather than about behaviour.
"""

import broker_core as core


def _local(path="images/x.ome.tiff"):
    return {"state": core.IMAGE_STATE_LOCAL, "path": path}


def _remote(store_root="https://s/x.zarr", region=None, multiscale_path=None):
    source = {"kind": core.NGFF_SOURCE_KIND, "store_root": store_root}
    if region is not None:
        source["region"] = region
    if multiscale_path is not None:
        source["multiscale_path"] = multiscale_path
    return {"state": "remote", "source": source}


def test_a_remote_ngff_entry_yields_its_source():
    region = {"x": 10, "y": 20, "width": 512, "height": 512}
    src = core.ngff_source_of(_remote(region=region, multiscale_path="0"))
    assert src is not None
    assert src["store_root"] == "https://s/x.zarr"
    assert src["region"] == region
    assert src["multiscale_path"] == "0"


def test_everything_without_a_store_to_read_yields_none():
    """Three different situations collapse to None on purpose: every caller treats
    them the same way, because none of them has a store to read chunks from."""
    assert core.ngff_source_of(_local()) is None, "a materialised image has no source"
    assert core.ngff_source_of({"state": "remote"}) is None, "no source at all"
    assert core.ngff_source_of({"state": "remote", "source": None}) is None
    assert core.ngff_source_of(
        {"state": "remote", "source": {"kind": "something-else",
                                       "store_root": "https://s/x"}}
    ) is None, "a source of another kind is not readable here"
    assert core.ngff_source_of(
        {"state": "remote", "source": {"kind": core.NGFF_SOURCE_KIND}}
    ) is None, "an ngff source with no store_root is a broken entry"
    assert core.ngff_source_of(
        {"state": "remote", "source": {"kind": core.NGFF_SOURCE_KIND,
                                       "store_root": ""}}
    ) is None, "an empty store_root would be fetched as an empty URL"
    assert core.ngff_source_of({"state": "remote", "source": ["not", "a", "dict"]}) is None
    assert core.ngff_source_of(None) is None
    assert core.ngff_source_of("not an entry") is None


def test_only_stems_with_neither_a_file_nor_a_source_are_unreadable():
    entries = {
        "has_file": _local(),
        "has_source": _remote(region={"x": 0, "y": 0, "width": 8, "height": 8}),
        "remote_but_broken": {"state": "remote", "source": {"kind": "ngff"}},
    }
    stems = ["has_file", "has_source", "remote_but_broken", "never_heard_of_it"]
    assert core.stems_without_a_readable_source(entries, stems) == [
        "never_heard_of_it",
        "remote_but_broken",
    ]


def test_control_the_refusal_really_narrowed_against_the_old_predicate():
    """THE CONTROL. A remote image with a readable source is still 'not local', so
    the old predicate names it and the new one must not. If both agreed, nothing
    about trainability would have changed."""
    entries = {"slide": _remote(region={"x": 0, "y": 0, "width": 512, "height": 512})}
    stems = ["slide"]
    old = core.non_local_stems(entries, stems)
    new = core.stems_without_a_readable_source(entries, stems)
    assert old == ["slide"], "the old predicate no longer names a remote image"
    assert new == [], "the new predicate still refuses a readable remote source"


def test_the_two_predicates_still_agree_where_they_should():
    """The narrowing must not have become 'refuse nothing': an entry that really has
    no way to be read has to appear in both."""
    entries = {"broken": {"state": "remote"}}
    stems = ["broken"]
    assert core.non_local_stems(entries, stems) == ["broken"]
    assert core.stems_without_a_readable_source(entries, stems) == ["broken"]


def test_an_empty_request_is_not_an_error():
    assert core.stems_without_a_readable_source({}, []) == []
    assert core.ngff_source_of({}) is None
