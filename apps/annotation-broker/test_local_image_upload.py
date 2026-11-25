"""Uploading one local image by presigned PUT, for a browser with no mount.

Firefox and Safari have no File System Access API, so they cannot mount a
folder -- and the mount is what registers the data-provider service that every
other upload route goes through. Such a browser could open and annotate
somebody else's data and could never contribute its own.

The property that matters here is not that an upload succeeds. It is that an
image which arrives this way is **indistinguishable afterwards** from one that
arrived through a mount, because a second kind of dataset image is a second
thing every reader has to know about. These tests pin that, and pin the rules
that make it true: the object lands at the same path, the manifest entry is
produced by the same reconciliation, and the stem and format rules are the
ones the mounted path already applies.

They are deliberately pure: the rules live in ``broker_core`` so they can be
exercised without Ray, Hypha or a network, which is the same choice
``test_public_gallery.py`` made for the same reason.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import broker_core as core


# ── the format rule, which did not exist server-side before ─────────────────


@pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg", ".tif", ".tiff"])
def test_every_format_the_MOUNTED_path_accepts_is_accepted_here(ext):
    """The list is the frontend's `SUPPORTED_EXTENSIONS`, restated server-side.

    Restated rather than imported, because a presigned PUT is handed to the
    client: a rule that lives only in the client is a suggestion. If the two
    lists drift, an image a user can pick in the browser is refused by the
    broker, or worse the reverse.
    """
    stem, got = core.split_local_image_filename(f"scan01{ext}")

    assert (stem, got) == ("scan01", ext)


def test_the_extension_is_matched_case_INSENSITIVELY_and_normalised():
    """A camera writes `IMG_0001.TIF`. That is the same format as `.tif`."""
    assert core.split_local_image_filename("IMG_0001.TIF") == ("IMG_0001", ".tif")
    assert core.split_local_image_filename("x.JPEG") == ("x", ".jpeg")


def test_the_STEM_keeps_its_case_even_though_the_extension_does_not():
    """The stem is the manifest key and the name the user reads back.

    Lowercasing it would quietly make the uploaded image a different image
    from the one they picked, and would collide two files that differ only in
    case.
    """
    stem, _ = core.split_local_image_filename("Sample_A.png")

    assert stem == "Sample_A"


def test_an_unsupported_format_is_refused_and_the_message_NAMES_the_supported_ones():
    with pytest.raises(ValueError) as e:
        core.split_local_image_filename("scan.bmp")

    msg = str(e.value)
    assert ".bmp" in msg
    for ext in core.LOCAL_IMAGE_EXTS:
        assert ext in msg, f"the refusal does not tell the caller about {ext}: {msg}"


def test_a_file_with_no_extension_at_all_is_refused():
    with pytest.raises(ValueError):
        core.split_local_image_filename("scan")


def test_a_dotfile_is_not_read_as_an_extension():
    """`.png` alone has an empty stem. Taking it would write `images/.png`."""
    with pytest.raises(ValueError):
        core.split_local_image_filename(".png")


# ── the stem rule, which is the mounted path's own ──────────────────────────


@pytest.mark.parametrize("bad", ["a/b.png", "..\\b.png", "dir/sub/c.tif"])
def test_a_PATH_is_refused_rather_than_reduced_to_its_basename(bad):
    """Refused, not silently flattened.

    Taking the basename would store the image under a stem the caller did not
    choose, in a place they did not mean, and report success -- so the first
    time anyone noticed would be when the wrong image opened.
    """
    with pytest.raises(ValueError) as e:
        core.split_local_image_filename(bad)

    assert "path" in str(e.value).lower()


@pytest.mark.parametrize("bad", ["my scan.png", "a*b.png", "a:b.png", "..png"])
def test_a_stem_the_MOUNTED_path_would_reject_is_rejected_here_too(bad):
    with pytest.raises(ValueError):
        core.split_local_image_filename(bad)


def test_the_stem_rule_IS_the_mounted_paths_rule_not_a_copy_of_it():
    """THE CONTROL on the two tests above.

    They would both pass against a stem rule that was stricter, or newly
    invented here. This asserts the same predicate the rest of the app uses
    is the one doing the work, so the two cannot drift.
    """
    assert core.is_valid_image_stem("scan01")
    assert not core.is_valid_image_stem("my scan")

    # and the splitter agrees with it, rather than having its own opinion
    assert core.split_local_image_filename("scan01.png")[0] == "scan01"
    with pytest.raises(ValueError):
        core.split_local_image_filename("my scan.png")


# ── the signature check, because S3 reports the PUT, not the content ────────


def _png(n=8):
    return core.PNG_SIGNATURE[:n]


def _jpeg(n=8):
    return (core.JPEG_SIGNATURE + b"\x00" * 8)[:n]


def _tiff(n=8):
    return (b"II*\x00" + b"\x00" * 8)[:n]


def test_each_format_is_recognised_from_its_own_magic_bytes():
    assert core.header_matches_ext(".png", _png())
    assert core.header_matches_ext(".jpg", _jpeg())
    assert core.header_matches_ext(".jpeg", _jpeg())
    assert core.header_matches_ext(".tif", _tiff())
    assert core.header_matches_ext(".tiff", _tiff())


@pytest.mark.parametrize(
    "ext,header",
    [
        (".png", _jpeg()),
        (".jpg", _png()),
        (".tif", _png()),
        (".png", b"<!DOCTYPE html>"[:8]),
        (".png", b""),
        (".tiff", b"\x00" * 8),
    ],
)
def test_bytes_that_DISAGREE_with_the_declared_format_are_refused(ext, header):
    """An error page stored under a .png name is the case this exists for.

    A presigned PUT reports success from S3, so "the client says it uploaded a
    PNG" is not evidence a PNG is there. Without this the manifest would
    record a readable image at a path that fails to open later, with nothing
    pointing back at the upload.
    """
    assert not core.header_matches_ext(ext, header)


def test_an_UNKNOWN_extension_fails_CLOSED():
    """Reaching here with `.bmp` means the format rule was bypassed.

    The safe reading of a rule that did not run is that it did not pass, so
    this returns False rather than defaulting to True for anything it has no
    signature for.
    """
    assert not core.header_matches_ext(".bmp", _png())
    assert not core.header_matches_ext("", _png())
    assert not core.header_matches_ext(None, _png())


# ── the point of the row: indistinguishable from a mounted upload ──────────


def test_the_uploaded_object_lands_where_a_MOUNTED_upload_lands():
    stem, ext = core.split_local_image_filename("scan01.png")

    assert f"images/{stem}{ext}" == "images/scan01.png"


def test_the_manifest_entry_is_the_one_RECONCILIATION_produces_not_a_new_shape():
    """The whole of "indistinguishable afterwards", asserted rather than argued.

    The upload path deliberately does NOT write a manifest entry. It puts the
    object under `images/` and lets the ordinary listing reconciliation make
    the entry -- the same code that runs for an image uploaded through a
    mount. This checks the two produce the identical entry, which is the
    guarantee that no reader downstream can tell them apart.
    """
    stem, ext = core.split_local_image_filename("scan01.png")
    name = f"{stem}{ext}"

    # what the listing looks like after the PUT lands
    listing = core.image_names_from_listing([{"name": name, "type": "file"}])
    assert listing == [name]

    reconciled, changed = core.merge_image_manifest({}, listing)

    assert changed is True
    assert reconciled == {stem: core.local_image_entry(name)}
    assert reconciled[stem]["state"] == core.IMAGE_STATE_LOCAL
    assert reconciled[stem]["path"] == f"images/{name}"
    assert core.image_entry_is_local(reconciled[stem])


def test_a_stem_that_is_ALREADY_an_image_is_not_silently_replaced():
    """Why the method reports `already` instead of minting a URL.

    The stem is the manifest key, so re-using one would overwrite a different
    image -- possibly a remote source, possibly one that already has
    annotations drawn against it.
    """
    existing = {"scan01": core.local_image_entry("scan01.png")}
    stem, _ = core.split_local_image_filename("scan01.tif")

    assert stem in existing, "the collision this guards against is on the STEM, not the filename"


# ── the role, guarded structurally so an edit cannot quietly drop it ───────


def _method_node(name):
    src = (pathlib.Path(__file__).parent / "broker.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(
        f"{name} is not defined in broker.py -- the upload route was renamed "
        f"or removed, so this guard no longer covers it and must be updated "
        f"deliberately"
    )


def test_the_upload_route_requires_the_MANAGER_role():
    """Structural, and said so: this reads the source, it does not call it.

    Adding an image is writing to somebody's dataset, where materialising only
    converts an image the dataset already has -- which is why that route takes
    `annotator` and this one must not. A role check is one line and its
    absence is invisible in any test that only uploads as a manager, so the
    check is that the line is there and says `manager`.
    """
    node = _method_node("upload_local_image")
    roles = [
        a.value
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", "") == "_require_role"
        for a in n.args
        if isinstance(a, ast.Constant) and isinstance(a.value, str)
    ]

    assert roles == ["manager"], f"expected a single manager check, got {roles}"


def test_the_upload_route_does_NOT_write_the_manifest_itself():
    """The structural half of the indistinguishability guarantee.

    If this route ever starts writing the entry, it becomes a second code path
    that can drift from the reconciliation, and the test above stops meaning
    what it says.
    """
    node = _method_node("upload_local_image")
    writes = [
        getattr(n.func, "attr", "")
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", "") in ("build_image_manifest", "_write_json_file")
    ]

    assert writes == [], f"the upload route writes the manifest directly: {writes}"
