"""Read one region of a remote OME-Zarr image, without fetching the rest of it.

Training used to receive whole images: ``_load_image_from_source`` GETs a URL and
hands the entire body to ``imread``. For a whole-slide source that is the only
thing standing between a 1 GB read and a 4096x4096 crop the annotator actually
drew on, so this module exists to read the crop and nothing else.

WHAT IT DOES NOT DO: parse OME-NGFF. The dataset paths and the axis roles come
from the broker's ``probe_ngff_source``, which already parses ``multiscales`` for
v0.4, v0.5 and the pre-v0.4 implied order. Two implementations of that parsing
already have to agree -- ``broker_core.parse_ngff_multiscales`` and
``ngffLayout.ts``, which it says it mirrors -- and a third here is how three
copies drift. This module consumes the interpretation; it never forms one.

THE SPLIT IN HERE IS DELIBERATE. Everything above ``read_region`` is pure: given
the probe's numbers it computes an index expression, and it is tested directly.
``read_region`` is the only part that touches a store, and it takes its opener as
an argument so the arithmetic can be tested without zarr installed. That matters
concretely: CI installs ``requirements-test.txt`` and the datasets requirements,
NOT ``requirements-entry.txt``, so a module-level ``import zarr`` here would turn
this file's tests into collection ERRORS in CI rather than failures -- the shape
that once let three skipped cases pass for a guard.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

NGFF_SOURCE_KIND = "ngff"

# A store credential travels as a PATH SEGMENT, ``/t/<credential>/`` -- the shape
# the broker's split_store_credential matches. It is kept out of the artifact on
# purpose, so the only way it reaches here is spliced into store_root for one
# call, and the only way it leaves is in text somebody writes down.
_CREDENTIAL_SEGMENT = re.compile(r"/t/[^/]+/")
CREDENTIAL_PLACEHOLDER = "/t/<redacted>/"


def redact_store_credential(text: Any) -> str:
    """*text* with any ``/t/<credential>/`` segment replaced.

    USED ON EVERY MESSAGE THIS MODULE RAISES, and worth applying anywhere a
    store URL could be stringified. The reason is not hypothetical: zarr's
    not-found error quotes the store it tried, URL and all, and that string
    travels into the session status document, which is served by an endpoint
    taking no caller context. A presigned URL leaking there expires; a store
    credential does not.

    The path form is kept so the message still identifies the store.
    """
    return _CREDENTIAL_SEGMENT.sub(CREDENTIAL_PLACEHOLDER, str(text))


class RegionSourceError(ValueError):
    """A region reference that cannot be read as written."""


def is_region_source(source: Any) -> bool:
    """True for a reference this module can read.

    Checked by SHAPE, not by trying to read it: a caller passing a plain URL or
    an array must fall through to the existing file path untouched.
    """
    return (
        isinstance(source, dict)
        and str(source.get("kind") or "") == NGFF_SOURCE_KIND
        and bool(str(source.get("store_root") or "").strip())
    )


def parse_region_source(source: Any) -> Dict[str, Any]:
    """Normalise a region reference, or raise with the reason.

    Accepts the shape the broker already persists for a remote image -- the
    ``source`` sub-document of ``remote_image_entry``: ``kind``, ``store_root``,
    ``multiscale_path`` and ``region``. Reading the same record the viewer reads
    is the point; a second description of a crop is a second thing to keep true.
    """
    if not is_region_source(source):
        raise RegionSourceError(
            "Not an OME-Zarr region reference: expected a mapping with "
            f"kind={NGFF_SOURCE_KIND!r} and a non-empty store_root."
        )
    region = source.get("region")
    if region is not None:
        if not isinstance(region, dict):
            raise RegionSourceError("region must be a mapping or omitted.")
        missing = [k for k in ("x", "y", "width", "height") if k not in region]
        if missing:
            raise RegionSourceError(
                f"region is missing {', '.join(missing)}; it needs x, y, width "
                f"and height in full-resolution pixels."
            )
        region = {k: int(region[k]) for k in ("x", "y", "width", "height")}
        if region["width"] <= 0 or region["height"] <= 0:
            raise RegionSourceError(
                f"region has non-positive extent "
                f"({region['width']}x{region['height']})."
            )
        if region["x"] < 0 or region["y"] < 0:
            raise RegionSourceError("region origin is negative.")
    planes = source.get("planes")
    if planes is not None:
        if not isinstance(planes, dict):
            raise RegionSourceError("planes must be a mapping of axis name to index.")
        cleaned: Dict[str, int] = {}
        for key, value in planes.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise RegionSourceError(
                    f"planes[{str(key)!r}]={value!r} is not an integer index."
                )
            if value < 0:
                raise RegionSourceError(f"planes[{str(key)!r}]={value} is negative.")
            cleaned[str(key)] = int(value)
        planes = cleaned
    return {
        "kind": NGFF_SOURCE_KIND,
        "store_root": str(source["store_root"]).strip(),
        "multiscale_path": source.get("multiscale_path") or None,
        # None means the whole image, which is how the broker records it: its
        # normalise_region returns None when the box covers the store.
        "region": region,
        # Which slice of each non-spatial axis the annotation belongs to. None
        # is NOT "the first one" -- see region_index, which refuses rather than
        # choose.
        "planes": planes,
    }


def region_index(
    axis_roles: Dict[str, Any],
    rank: int,
    shape: Sequence[int],
    region: Optional[Dict[str, int]],
    axes: Optional[Sequence[str]] = None,
    planes: Optional[Dict[str, int]] = None,
) -> Tuple[Any, ...]:
    """The index expression that selects *region* from a level-0 array.

    ``axis_roles`` is the broker's ``{"y", "x", "c"}`` axis indices. Every axis
    that is not y, x or c is reduced to ONE element rather than kept: a 5D
    tczyx store would otherwise come back with singleton t and z dimensions
    that neither trainer accepts. The channel axis is kept whole, because
    dropping it would silently turn an RGB crop into one band.

    WHICH element, for an axis of size > 1, is not this module's to guess, and
    guessing it was a real defect rather than a hypothetical one. The annotator
    displays the MIDDLE slice of a z-stack (``Math.floor(size / 2)`` in the
    frontend's layout code), so an index of 0 pairs the polygons drawn on one
    plane with the pixels of another. Nothing downstream catches that: the mask
    is non-empty, so the empty-mask guard does not fire, the shapes match, and
    training reports success having learnt a plane the annotator never saw.

    So an axis of size > 1 with no entry in *planes* is REFUSED. Size 1 needs no
    choice and is taken. *planes* maps an axis NAME to its index, which needs
    *axes* to resolve; both come from the probe, through the layout recorded at
    import. A store with only y, x and c reaches none of this.
    """
    y, x = axis_roles.get("y"), axis_roles.get("x")
    c = axis_roles.get("c")
    if not isinstance(y, int) or not isinstance(x, int):
        raise RegionSourceError(f"axis_roles has no usable y/x: {axis_roles!r}")
    if rank != len(shape):
        raise RegionSourceError(
            f"rank {rank} does not match shape {list(shape)!r}; the probe and "
            f"the array disagree, so neither is trusted."
        )
    for name, idx in (("y", y), ("x", x)):
        if not (0 <= idx < rank):
            raise RegionSourceError(
                f"axis_roles[{name!r}]={idx} is outside an array of rank {rank}."
            )

    if region is None:
        y_sel: Any = slice(None)
        x_sel: Any = slice(None)
    else:
        h, w = int(shape[y]), int(shape[x])
        y0, x0 = region["y"], region["x"]
        y1, x1 = y0 + region["height"], x0 + region["width"]
        # REFUSED rather than clamped. A crop silently shrunk at the edge
        # trains on a different area than the annotator drew on, and the label
        # mask is rasterised to the region's declared width/height, so a
        # clamped image would no longer be the same size as its labels.
        if y1 > h or x1 > w:
            raise RegionSourceError(
                f"region x={x0},y={y0},{region['width']}x{region['height']} "
                f"runs past the level-0 array ({w}x{h}). It is refused rather "
                f"than clamped: a shrunk crop would not match its rasterised "
                f"labels."
            )
        y_sel, x_sel = slice(y0, y1), slice(x0, x1)

    names = [str(n) for n in axes] if axes is not None else []
    if names and len(names) != rank:
        raise RegionSourceError(
            f"axes {names!r} has {len(names)} entries for an array of rank "
            f"{rank}; the probe describes a different array."
        )
    want = dict(planes or {})

    # DECISION (#0908 clause 3): a plane named for a SPATIAL or CHANNEL axis, or
    # for an axis this layout does not have, is REFUSED -- not ignored.
    #
    # Those axes never reach `_plane_index`: y and x get the crop's slices and a
    # channel axis is kept whole, so until now `planes={"c": 2}` was accepted and
    # discarded, and `planes={"bogus": 1}` likewise. The caller asked for one
    # thing and got another with no complaint, which is the same class of silence
    # this module refuses elsewhere.
    #
    # Refusing is also what makes the two ends agree: the annotation broker's
    # import validator already refuses a plane for a spatial or channel axis and
    # an unknown axis name, so nothing written through the supported path can
    # carry these shapes. A record written before that validator, or by another
    # path, now fails loudly here instead of training on something the caller did
    # not ask for.
    #
    # WITHOUT AXIS NAMES, A NON-EMPTY `planes` IS REFUSED OUTRIGHT.
    #
    # An earlier version of this comment claimed the refusals below applied
    # generally and that `_plane_index` reported the names-absent case itself.
    # Both were wrong, and wrong in the worst place: a stem imported before axis
    # names were recorded arrives with `axes=[]` -- the exact legacy population
    # this decision is motivated by -- and the role check below is gated on
    # `names`, so it was skipped. `_plane_index` then took the size-1 shortcut
    # without looking at `planes` at all. Measured on that version, axes=[],
    # shape [3, 1, 64, 64]: `{"z": 7}`, `{"c": 2}` and `{"bogus": 1}` were ALL
    # accepted and discarded -- this row's own defect, in the same form.
    #
    # `_plane_index` only reports the missing names when some non-spatial axis
    # has size > 1. For a singleton, or a plain cyx store, it never fires, and
    # those are precisely where the silence lived.
    #
    # So refuse: with no names the keys cannot be resolved to any axis, which
    # makes honouring them and range-checking them equally impossible, and
    # ignoring them is the silence this decision exists to rule out. Nothing
    # written through the supported path has this shape, because the import
    # validator records planes only alongside the names they refer to.
    if want and not names:
        raise RegionSourceError(
            f"planes {sorted(want)!r} was given, but this array's axis names are "
            f"not recorded, so no plane can be resolved to an axis. It is refused "
            f"rather than ignored. Re-import the source so the layout records its "
            f"axis names, or drop planes from the region reference."
        )

    if want and names:
        roles: Dict[str, str] = {}
        for role, idx in (("y", y), ("x", x), ("c", c)):
            if isinstance(idx, int) and 0 <= idx < len(names):
                roles[names[idx]] = role
        for key in sorted(want):
            if key in roles:
                raise RegionSourceError(
                    f"planes[{key!r}] names the {roles[key]!r} axis, which is not a "
                    f"plane axis: y and x are the crop and a channel axis is kept "
                    f"whole. It is refused rather than ignored, because ignoring it "
                    f"would read something other than what was asked for. Drop "
                    f"{key!r} from planes."
                )
            if key not in names:
                raise RegionSourceError(
                    f"planes[{key!r}] names no axis of this array, whose axes are "
                    f"{list(names)!r}. It is refused rather than ignored: a plane "
                    f"that selects nothing looks like a plane that was honoured."
                )

    out: List[Any] = []
    for axis in range(rank):
        if axis == y:
            out.append(y_sel)
        elif axis == x:
            out.append(x_sel)
        elif isinstance(c, int) and axis == c:
            out.append(slice(None))
        else:
            out.append(_plane_index(axis, int(shape[axis]), names, want))
    return tuple(out)


def _plane_index(
    axis: int, size: int, names: Sequence[str], planes: Dict[str, int]
) -> int:
    """The one index to take along a non-spatial axis, or raise.

    Size 1 is not a choice. Any larger size needs the caller to say which plane
    the annotation came from; see ``region_index``'s docstring for why a default
    of 0 is wrong and silent.
    """
    name = str(names[axis]) if axis < len(names) else ""
    # A SUPPLIED plane is range-checked FIRST, before the size-1 shortcut.
    #
    # The shortcut used to come first, so for a single-element axis the caller's
    # index was never looked at: planes={"z": 7} on a z of size 1 was accepted
    # and silently read z=0. The pixels were right -- there is only one plane --
    # but the error was lost, so a caller whose numbering is wrong got no
    # complaint, and a stored record could claim a plane that does not exist.
    # The same index on a 4-element axis was refused cleanly, which is how the
    # inconsistency was found: the check existed and was simply unreachable.
    if name and name in planes:
        index = int(planes[name])
        if not (0 <= index < size):
            where = (
                f"which has only 1 element, so the only valid index is 0"
                if size == 1
                else f"which has {size} elements (0..{size - 1})"
            )
            raise RegionSourceError(
                f"planes[{name!r}]={index} is outside axis {name!r}, {where}."
            )
        return index
    if size == 1:
        return 0
    label = f"{name!r}" if name else f"at position {axis}"
    known = (
        f"Its name is not in the recorded axes ({list(names)!r}), so a plane "
        f"cannot be named for it; re-import the source so the layout records "
        f"its axis names."
        if not name
        else f"Pass planes={{{name!r}: <index>}} on the region reference."
    )
    raise RegionSourceError(
        f"axis {label} has {size} elements and no plane was given, so there is "
        f"no way to tell which one the annotation was drawn on. Index 0 is NOT "
        f"assumed: the annotator shows the middle slice, so index 0 would pair "
        f"the polygons with a different plane's pixels, the mask would still be "
        f"non-empty, and the run would report success having trained on the "
        f"wrong image. {known}"
    )


def to_hw_or_hwc(array: Any, axis_roles: Dict[str, Any], rank: int) -> Any:
    """Move a kept channel axis last, so the result is HxW or HxWxC.

    Both trainers read HxW or HxWxC. ``region_index`` drops every axis except
    y, x and c, so what comes back is already 2D or 3D -- but in the store's
    axis order, which for a tczyx store puts channels FIRST. Transposing here
    rather than in the caller keeps the store's layout from leaking outward.
    """
    import numpy as np

    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr
    c = axis_roles.get("c")
    if arr.ndim != 3 or not isinstance(c, int):
        return arr
    y, x = int(axis_roles["y"]), int(axis_roles["x"])
    # Among the three surviving axes, their relative order is preserved.
    order = sorted([("c", c), ("y", y), ("x", x)], key=lambda t: t[1])
    names = [n for n, _ in order]
    want = ["y", "x", "c"]
    if names == want:
        return arr
    return np.transpose(arr, [names.index(n) for n in want])


def level0_path(dataset_paths: Sequence[str]) -> str:
    """The full-resolution level.

    OME-NGFF orders ``datasets`` coarsest-last, so level 0 is the first entry.
    Taken from the probe rather than guessed as ``"0"``: a store is free to name
    its levels anything, and several in the wild use ``"s0"``.
    """
    if not dataset_paths:
        raise RegionSourceError("the probe returned no dataset paths.")
    return str(dataset_paths[0]).strip("/")


# A whole-level read that nobody asked for is the exact thing this module exists
# to avoid, and ``region`` is optional, so omitting it reaches a full download
# through the region-aware path. 64 megapixels is ~8000x8000, which is a large
# microscope field and a small fraction of a slide; the broker caps its own
# materialisation path at 64 MiB for a comparable reason. A caller that really
# wants a whole big level says so with ``allow_whole_level=True``.
MAX_WHOLE_LEVEL_PIXELS = 64 * 1024 * 1024


def read_region(
    source: Any,
    probe: Dict[str, Any],
    opener: Any = None,
    allow_whole_level: bool = False,
) -> Any:
    """Read *source*'s region out of its store and return HxW or HxWxC.

    *probe* is the broker's ``probe_ngff_source`` result; this reads
    ``dataset_paths``, ``axis_roles`` and ``level0_shape`` from it and parses
    nothing itself.

    *opener* takes ``(store_root, array_path)`` and returns something
    numpy-sliceable. It defaults to a zarr-backed opener, imported HERE rather
    than at module scope so that the pure functions above can be tested in an
    environment without zarr.
    """
    ref = parse_region_source(source)
    for key in ("dataset_paths", "axis_roles", "level0_shape"):
        if key not in probe:
            raise RegionSourceError(
                f"the probe is missing {key!r}; this needs a probe_ngff_source "
                f"result from a broker that returns the store layout."
            )
    shape = list(probe["level0_shape"])
    roles = probe["axis_roles"]
    if ref["region"] is None and not allow_whole_level:
        # Refused BEFORE the open, so nothing is fetched to discover the size.
        px = int(shape[int(roles["y"])]) * int(shape[int(roles["x"])])
        if px > MAX_WHOLE_LEVEL_PIXELS:
            raise RegionSourceError(
                f"no region given and level 0 is "
                f"{shape[int(roles['x'])]}x{shape[int(roles['y'])]} "
                f"({px} pixels), above the {MAX_WHOLE_LEVEL_PIXELS}-pixel cap for "
                f"a whole-level read. Pass a region, or allow_whole_level=True if "
                f"the whole level really is wanted."
            )
    # axes and planes are passed so a multi-plane store refuses rather than
    # reading plane 0 -- and refuses BEFORE the open below, so a wrong-plane
    # reference costs no requests.
    index = region_index(
        probe["axis_roles"],
        len(shape),
        shape,
        ref["region"],
        axes=probe.get("axes"),
        planes=ref["planes"],
    )

    if opener is None:
        opener = _zarr_opener
    path = level0_path(probe["dataset_paths"])
    # EVERY FAILURE FROM HERE DOWN IS RE-RAISED WITHOUT THE URL IN IT.
    #
    # zarr's own errors quote the store they tried, which for a credentialed
    # store is the credential. From here the exception text reaches the session
    # status document and the log, and the endpoint that serves session status
    # takes no caller context -- so one ordinary failure (a rotated credential,
    # a moved store, a wrong multiscale name) publishes it. A presigned URL
    # going the same way expires; this does not.
    #
    # The redaction keeps the path, so the message still says WHICH store.
    # RAISED OUTSIDE THE HANDLER, which is the part that is easy to get wrong.
    # ``raise ... from None`` clears __cause__ and suppresses the chain in the
    # default traceback format, but Python still attaches the original as
    # __context__, and anything that walks that attribute -- an error reporter,
    # a structured logger -- reads the unredacted text straight off it. Leaving
    # the except block first clears the handled exception, so the replacement
    # carries no reference to the original at all.
    array = None
    failure = None
    try:
        array = opener(ref["store_root"], path)
    except Exception as e:
        failure = f"{type(e).__name__}: {redact_store_credential(e)}"
    if failure is not None:
        raise RegionSourceError(
            f"could not open {path!r} in "
            f"{redact_store_credential(ref['store_root'])}: {failure}"
        )

    # THE OPENED ARRAY MUST BE THE ONE THE INDEX WAS COMPUTED FOR.
    #
    # Every number above comes from the layout recorded at import; the array
    # comes from the store now. Two ways they diverge, and both end in wrong
    # pixels rather than an error:
    #
    #   The level is wrong. ``level0_path`` takes the first dataset path, which
    #   OME-NGFF requires to be the finest level -- but the frontend's pyramid
    #   code orders levels by size and states that the order is only a
    #   convention, so a non-conforming store has two consumers disagreeing
    #   about which array is level 0. Read at a downsampled level, a region in
    #   level-0 coordinates is a crop of somewhere else entirely.
    #
    #   The store changed since import. A re-exported pyramid keeps its paths.
    #
    # Either way a numpy slice does NOT complain: an out-of-range slice is
    # silently CLAMPED, so the crop comes back smaller than its rasterised
    # labels -- the exact outcome region_index refuses to produce by clamping.
    # ``.shape`` is metadata, so this costs no chunk fetches.
    actual = getattr(array, "shape", None)
    if actual is not None and tuple(int(n) for n in actual) != tuple(
        int(n) for n in shape
    ):
        raise RegionSourceError(
            f"the array at {level0_path(probe['dataset_paths'])!r} has shape "
            f"{list(int(n) for n in actual)!r}, but the layout recorded at "
            f"import says level 0 is {list(int(n) for n in shape)!r}. The region "
            f"was computed against the recorded shape, so reading this array "
            f"would crop somewhere other than where the annotation was drawn. "
            f"Either the store's first dataset is not its finest level (the "
            f"OME-NGFF order is finest-first, and a store that violates it "
            f"cannot be read by position), or the store changed since it was "
            f"imported: re-import the source."
        )
    # The chunk GETs happen HERE, not at the open, so a credentialed URL reaches
    # a reader's exception on this line too. Raised outside the handler for the
    # same reason as above.
    data = None
    failure = None
    try:
        data = array[index]
    except Exception as e:
        failure = f"{type(e).__name__}: {redact_store_credential(e)}"
    if failure is not None:
        raise RegionSourceError(
            f"could not read the region out of {path!r} in "
            f"{redact_store_credential(ref['store_root'])}: {failure}"
        )
    return to_hw_or_hwc(data, probe["axis_roles"], len(shape))


class ShortStoreRead(RegionSourceError):
    """A store object arrived shorter than the server said it would be.

    Its own class because the response differs from a malformed reference: this is
    a transfer that failed silently, and retrying it may well succeed.
    """


# The zarr-3 ``Store`` subclass is built on FIRST USE, not at import time.
#
# This module must stay importable without zarr (there is a test pinning that),
# and a subclass needs its base class at class-creation time, so the class body
# cannot live at module scope. ``LengthCheckedStore`` below is therefore a
# factory function with the same call shape as the class it returns.
_LENGTH_CHECKED_STORE_CLASS: Optional[Any] = None


def _length_checked_store_class() -> Any:
    """The read-only zarr-3 store class, created once and cached."""
    global _LENGTH_CHECKED_STORE_CLASS
    if _LENGTH_CHECKED_STORE_CLASS is not None:
        return _LENGTH_CHECKED_STORE_CLASS

    import asyncio

    from zarr.abc.store import Store

    class _LengthCheckedStore(Store):  # type: ignore[misc]
        """A read-only Zarr store that supplies bytes and refuses a bad length.

        zarr keeps ALL of its indexing and decoding. This hands it the bytes for
        one key and checks a single thing: that as many bytes arrived on the wire
        as the server said it would send.

        **The measured reason it exists.** A chunk truncated mid-transfer decodes
        WITHOUT raising, to the full chunk shape, with wrong content — and the
        first pixels still match the source, so looking at the crop passes it.
        Both layers underneath were measured and both are blind: the decompressor
        does not notice a truncated stream, and zarr's own HTTP store does not
        notice either, even when the response declares the full length.
        ``Content-Length`` is the last point at which the truncation is still
        visible, which is why the check is here, on the BYTES, and not on the
        image.

        **The comparison is against the bytes on the WIRE, which is not the same
        as the bytes zarr gets.** ``Content-Length`` counts the ENCODED body. An
        object served with ``Content-Encoding: gzip`` is decoded by httpx before
        we see it, so comparing the decoded length against the declared one calls
        a perfectly intact object "cut short" — measured, and it is why the fetch
        reports the wire count separately. A long body is reported as long rather
        than as short, for the same reason: the message has to describe what
        happened.

        A mismatch raises with the KEY and both lengths and never the URL: the
        store root carries the credential as a path segment.
        """

        def __init__(self, root: str, fetch: Optional[Any] = None) -> None:
            super().__init__(read_only=True)
            self._root = root if root.endswith("/") else root + "/"
            self._fetch = fetch or _http_fetch

        # ── identity ──────────────────────────────────────────────────────
        def __eq__(self, other: object) -> bool:
            return isinstance(other, _LengthCheckedStore) and other._root == self._root

        def __hash__(self) -> int:
            return hash(("LengthCheckedStore", self._root))

        def __repr__(self) -> str:
            return f"LengthCheckedStore({redact_store_credential(self._root)!r})"

        # ── the one method that does the work ─────────────────────────────
        def _get_bytes(
            self, key: str, range_header: Optional[str] = None
        ) -> Optional[bytes]:
            """The object's bytes, or the requested RANGE of them, or ``None``.

            ``None`` rather than an error because zarr omits a chunk that is
            entirely the fill value, so absence is DATA. A store that raised here
            would turn every sparse source into a failed read.

            **Why the range goes to the server instead of being applied here.**
            Sharding is the normal layout for OME-NGFF v0.5 whole-slide data:
            many small inner chunks packed into one large shard object with an
            index at its end. zarr reads such an array by asking for a suffix
            range (the index) and then a range per inner chunk. Fetching the
            whole object and slicing locally returns the RIGHT pixels, so nothing
            fails — while pulling the entire shard down once per inner chunk.
            Measured on exactly that mistake: a 1 MB+ single-shard store answered
            a 128x128 crop with the whole shard fetched TWICE and not one Range
            header sent. A multi-gigabyte shard makes that ruinous, and it is
            precisely what this module exists not to do.
            """
            try:
                status, body, declared, wire, content_range = self._fetch(
                    self._root + str(key), range_header
                )
            except ShortStoreRead as exc:
                # The transport caught it first — the common case and the
                # stronger one, since a connection that went away mid-body is
                # unambiguous. Re-raise with the KEY, so the message names what
                # failed and not only how.
                #
                # "the transfer was cut short" is stated HERE as well as on the
                # length-mismatch path below, deliberately: the two paths detect
                # the same fault and an operator should not have to know which
                # layer noticed in order to recognise it. The transport's own
                # wording ("peer closed connection without sending complete
                # message body (received N bytes, expected M)") is kept after it,
                # because it carries both counts.
                raise ShortStoreRead(
                    redact_store_credential(
                        f"store key {key!r}: the transfer was cut short. {exc}"
                    )
                ) from None
            if int(status) == 404:
                return None
            if not 200 <= int(status) < 300:
                raise RegionSourceError(
                    redact_store_credential(f"store key {key!r} returned HTTP {status}")
                )
            if range_header is not None and int(status) != 206:
                # The server ignored the Range and sent the WHOLE object. Slicing
                # it here would be CORRECT and would cost a full download per
                # inner chunk, silently — the sharding cost described above, just
                # moved. Refuse instead and say so, rather than succeed
                # expensively: a store this reader cannot read economically is a
                # fact the caller needs, and a multi-gigabyte shard fetched per
                # chunk is not a smaller problem than a failed read.
                raise RegionSourceError(
                    redact_store_credential(
                        f"store key {key!r}: asked for range {range_header!r} and "
                        f"the server answered HTTP {status} with the whole "
                        f"object instead of 206. This store cannot be read one "
                        f"region at a time, because every inner chunk of a "
                        f"sharded array would cost a full download of its shard."
                    )
                )
            if range_header is not None:
                # 206 established above. Now: are these the bytes we asked for?
                _check_content_range(key, range_header, content_range, len(body))
            want: Optional[int]
            try:
                want = int(declared) if declared is not None else None
            except (TypeError, ValueError):
                want = None
            got: Optional[int]
            try:
                got = int(wire) if wire is not None else None
            except (TypeError, ValueError):
                got = None
            if want is not None and got is not None and got != want:
                how = (
                    "the transfer was cut short"
                    if got < want
                    else "the body was LONGER than declared"
                )
                raise ShortStoreRead(
                    redact_store_credential(
                        f"store key {key!r} arrived with {got} bytes on the wire "
                        f"but the server declared {want}: {how}. A truncated "
                        f"chunk decodes to the right shape with the wrong pixels, "
                        f"so it is refused here rather than trained on."
                    )
                )
            return body

        async def get(self, key: str, prototype: Any, byte_range: Any = None) -> Any:
            # The range is asked of the SERVER, never applied locally. See
            # _range_header and the sharding note in _get_bytes.
            body = await asyncio.to_thread(
                self._get_bytes, key, _range_header(byte_range)
            )
            if body is None:
                return None
            return prototype.buffer.from_bytes(body)

        async def get_partial_values(self, prototype: Any, key_ranges: Any) -> Any:
            out = []
            for key, rng in key_ranges:
                out.append(await self.get(key, prototype, rng))
            return out

        async def exists(self, key: str) -> bool:
            return (await asyncio.to_thread(self._get_bytes, key, None)) is not None

        # ── what this store deliberately does NOT do ──────────────────────
        @property
        def supports_writes(self) -> bool:
            return False

        @property
        def supports_deletes(self) -> bool:
            return False

        @property
        def supports_partial_writes(self) -> bool:
            return False

        @property
        def supports_listing(self) -> bool:
            return False

        async def set(self, key: str, value: Any) -> None:
            raise RegionSourceError("this store is read-only")

        async def delete(self, key: str) -> None:
            raise RegionSourceError("this store is read-only")

        def _no_listing(self) -> Any:
            # Listing is NOT supported, and saying so is deliberate: zarr
            # addresses the metadata key and each chunk key directly when opening
            # an array read-only, and HTTP has no listing to offer. A store that
            # answered an EMPTY listing instead would turn "I cannot list" into
            # "there is nothing here", which is the silent-wrong-answer shape
            # this whole class exists to prevent. `supports_listing` is False, so
            # a correct caller asks first.
            raise NotImplementedError("this store does not support listing")

        def list(self) -> Any:
            return self._no_listing()

        def list_prefix(self, prefix: str) -> Any:
            return self._no_listing()

        def list_dir(self, prefix: str) -> Any:
            return self._no_listing()

    _LENGTH_CHECKED_STORE_CLASS = _LengthCheckedStore
    return _LENGTH_CHECKED_STORE_CLASS


def LengthCheckedStore(root: str, fetch: Optional[Any] = None) -> Any:
    """Build a read-only, length-checking zarr store rooted at ``root``.

    A function rather than a class so that zarr is imported on first use; see
    :func:`_length_checked_store_class`.
    """
    return _length_checked_store_class()(root, fetch=fetch)


class WrongRangeRead(RegionSourceError):
    """The server answered a range request with a DIFFERENT range.

    Its own class because it is not a short read and not a malformed reference:
    the response is complete and well-formed and describes bytes other than the
    ones asked for. Measured consequence, on an uncompressed sharded store: the
    crop comes back with no error and the wrong pixels.
    """


def _check_content_range(
    key: str, requested: str, content_range: Optional[str], body_len: int
) -> None:
    """Refuse unless the 206 describes the range that was actually requested.

    **Why this is not covered by the length check.** The wire-byte check compares
    what arrived against what the server SAID it was sending, so a server that
    returns the wrong bytes with self-consistent headers passes it. Measured on
    an uncompressed sharded store: a 206 carrying the NEIGHBOURING bytes, with an
    honest Content-Length and an honest Content-Range for those bytes, produced a
    crop with no error and the wrong pixels. Compression and the shard index's
    own CRC catch most variants of this; neither is guaranteed to be present.

    So the one remaining question — "are these the bytes I asked for?" — is
    answered by comparing the request against ``Content-Range``, which is the
    only place the server states which bytes it sent.
    """
    if content_range is None:
        raise WrongRangeRead(
            redact_store_credential(
                f"store key {key!r}: the server answered 206 to range "
                f"{requested!r} without a Content-Range header, so there is no "
                f"way to tell which bytes it sent."
            )
        )
    m = re.fullmatch(
        r"\s*bytes\s+(\d+)-(\d+)/(\d+|\*)\s*", content_range, re.IGNORECASE
    )
    if not m:
        raise WrongRangeRead(
            redact_store_credential(
                f"store key {key!r}: unparsable Content-Range "
                f"{content_range!r} for range {requested!r}."
            )
        )
    start, end = int(m.group(1)), int(m.group(2))
    total = None if m.group(3) == "*" else int(m.group(3))
    served = end - start + 1

    def refuse(why: str) -> None:
        raise WrongRangeRead(
            redact_store_credential(
                f"store key {key!r}: asked for {requested!r} and the server "
                f"answered Content-Range {content_range!r} — {why}. These are "
                f"not the bytes that were requested, and they decode without "
                f"complaint into the wrong pixels."
            )
        )

    if served != body_len:
        refuse(f"it describes {served} bytes but {body_len} arrived")

    rm = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
    if not rm:  # pragma: no cover -- _range_header only emits these three forms
        refuse("the request itself could not be re-parsed")
        return
    a, b = rm.group(1), rm.group(2)
    if a == "":  # suffix: bytes=-N, the LAST n bytes
        n = int(b)
        if total is not None and end != total - 1:
            refuse(f"a suffix range must end at the last byte ({total - 1})")
        if body_len > n:
            refuse(f"a suffix range of {n} bytes returned {body_len}")
        if body_len < n and start != 0:
            # FEWER bytes than a suffix asked for is legitimate in exactly one
            # case: the object is smaller than the suffix, so the whole of it is
            # the answer and it must therefore start at byte 0. Any other short
            # suffix is the server choosing a different window — `bytes=-10`
            # answered `bytes 4107-4111/4112` is 5 bytes from the middle of a
            # 4112-byte object, which ends at the last byte and so satisfies
            # every other rule here.
            refuse(
                f"only {body_len} of the {n} bytes asked for came back, and they "
                f"start at {start} rather than 0, so the object is larger than "
                f"the response and this is a different window"
            )
    else:
        if start != int(a):
            refuse(f"it starts at {start}, not at {a}")
        if b != "" and end != int(b):
            refuse(f"it ends at {end}, not at {b}")


def _range_header(byte_range: Any) -> Optional[str]:
    """zarr's byte-range request as an HTTP ``Range`` header value, or ``None``.

    Every variant is handled explicitly and an unknown one REFUSES. Both of the
    tempting fallbacks are wrong in their own way, which is why neither is taken:
    sending no Range fetches the whole object, and for a sharded array that is a
    full shard download per inner chunk; guessing a range reads the WRONG bytes,
    which decode without complaint.

    ``RangeByteRequest`` is end-EXCLUSIVE and HTTP ranges are end-INCLUSIVE, so
    the ``- 1`` is load-bearing: without it every ranged read is one byte long and
    lands one byte past where it should end.
    """
    if byte_range is None:
        return None

    from zarr.abc.store import (
        OffsetByteRequest,
        RangeByteRequest,
        SuffixByteRequest,
    )

    if isinstance(byte_range, RangeByteRequest):
        return f"bytes={byte_range.start}-{byte_range.end - 1}"
    if isinstance(byte_range, OffsetByteRequest):
        return f"bytes={byte_range.offset}-"
    if isinstance(byte_range, SuffixByteRequest):
        return f"bytes=-{byte_range.suffix}"
    raise RegionSourceError(
        f"unsupported byte range request {type(byte_range).__name__}: refusing. "
        f"Sending no range would fetch the whole object, and for a sharded array "
        f"that is a full shard per inner chunk; guessing one would read the wrong "
        f"bytes."
    )


def _http_fetch(url: str, range_header: Optional[str] = None):
    """GET a store key: ``(status, body, declared_length, wire_length, content_range)``.

    httpx is lazy here for the same reason zarr is: this module is imported by a
    path that must stay importable without the heavier stack.

    **httpx is chosen over the client zarr would use by default, and that choice is
    the protection.** Measured: with zarr's own HTTP store a body cut short of its
    declared length is accepted silently and the crop comes back wrong, while httpx
    raises on exactly that. Its error is converted here so the failure belongs to
    this module and goes through the redactor — an httpx error can quote the URL it
    tried, and the store root carries the credential.

    **``body`` and ``wire_length`` are deliberately two different measurements,
    and conflating them was a bug.** ``Content-Length`` counts the bytes the
    server put ON THE WIRE, which for a response carrying
    ``Content-Encoding: gzip`` is the COMPRESSED size; ``resp.content`` is what
    httpx hands back after decoding it. httpx sends ``Accept-Encoding: gzip`` by
    default, so comparing ``len(resp.content)`` against the declared length
    reports an intact, correctly-served object as "cut short" — measured, with
    the decoded body coming back LONGER than declared. Two things are done about
    it: ``identity`` is requested, so there is normally nothing to decode; and
    the count actually compared is ``num_bytes_downloaded``, the wire bytes,
    which stays correct even if a server ignores that and encodes anyway.
    """
    import httpx

    # Ask for no transfer encoding, so the body we measure and the body zarr
    # decodes are the same bytes. See the docstring.
    headers = {"Accept-Encoding": "identity"}
    if range_header is not None:
        headers["Range"] = range_header
    try:
        resp = httpx.get(
            url,
            timeout=60.0,
            follow_redirects=True,
            headers=headers,
        )
    except httpx.RemoteProtocolError as exc:
        # "peer closed connection without sending complete message body
        # (received N bytes, expected M)" — the truncation, named by the transport.
        raise ShortStoreRead(redact_store_credential(str(exc))) from None
    except httpx.HTTPError as exc:
        raise RegionSourceError(
            redact_store_credential(f"{type(exc).__name__}: {exc}")
        ) from None
    return (
        resp.status_code,
        resp.content,
        resp.headers.get("Content-Length"),
        resp.num_bytes_downloaded,
        resp.headers.get("Content-Range"),
    )


def _zarr_opener(store_root: str, array_path: str) -> Any:
    """Open one array inside a remote Zarr group, read-only.

    Imported lazily: see this module's docstring on why zarr must not be a
    module-level import here.

    The array is opened through :class:`LengthCheckedStore` rather than by URL, so
    zarr does its own indexing over bytes that have been checked against the
    length the server declared.
    """
    import zarr

    root = store_root if store_root.endswith("/") else store_root + "/"
    return zarr.open_array(store=LengthCheckedStore(root + array_path), mode="r")
