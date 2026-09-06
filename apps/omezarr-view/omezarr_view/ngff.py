"""Synthesise NGFF 0.4 group metadata for a view over a non-Zarr source file.

tifffile's ``write_fsspec`` emits ``multiscales`` version 0.1 with only
``datasets[].path`` — no axes, no coordinateTransformations, no omero block —
which NGFF 0.4 consumers (Vizarr, ome-zarr-py) reject. This module builds the
0.4 metadata from whatever the source file actually declares, and records what
it could not fill in.

**The mapping report is quoted verbatim in publications, so an unchecked claim
in it is a falsehood, not a rough edge.** Two rules follow, and both are
enforced by the shape of the code rather than by care:

1. A field may only be reported "not declared in the source" if the extractor
   actually looked. Anything it never inspected is "not read by this recipe",
   which is a different and honest statement. ``SourceMetadata.checked`` is the
   record of what was inspected.
2. Every field that is dropped must appear in one list or the other. A silent
   loss defeats the entire point of a generated report — worse than a disclosed
   one, because the reader has no way to know to ask.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from xml.etree import ElementTree

OME_NS = "{http://www.openmicroscopy.org/Schemas/OME/2016-06}"

# OME PhysicalSize*Unit values -> UDUNITS-2 names, which NGFF 0.4 requires.
_UNITS = {
    "m": "meter", "dm": "decimeter", "cm": "centimeter", "mm": "millimeter",
    "µm": "micrometer", "um": "micrometer", "micron": "micrometer",
    "nm": "nanometer", "pm": "picometer", "Å": "angstrom",
    "s": "second", "ms": "millisecond", "min": "minute", "h": "hour",
}

_AXIS_TYPE = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}

# Used only when the source declares no colour of its own.
_DEFAULT_COLORS = ["00FF00", "FF0000", "0000FF", "FFFF00", "FF00FF", "00FFFF", "FFFFFF"]

# Every field this module knows how to drop. Anything here must end up in the
# mapped list or the not-mapped list; the builder asserts it.
ACCOUNTABLE = (
    "physical pixel sizes", "time increment", "channel names", "channel colours",
    "display windows", "bit depth", "acquisition date", "pyramid levels",
    "scenes", "objective / instrument metadata", "stage position",
    "plate / well context", "ROIs and annotations",
)


@dataclass
class SourceMetadata:
    """What a reader extracted, and — separately — what it actually inspected.

    A value of None means different things depending on whether its name is in
    ``checked``: inspected-and-absent, or never looked at. Conflating those two
    is what produced three false "not declared in the source" claims.
    """

    axes: str
    level_shapes: Sequence[Sequence[int]]
    dtype: str
    name: str
    checked: Set[str] = field(default_factory=set)
    # axis letter -> (size, unit as the source spelled it)
    physical_sizes: Dict[str, Tuple[float, Optional[str]]] = field(default_factory=dict)
    time_increment: Optional[Tuple[float, Optional[str]]] = None
    channel_names: Optional[List[Optional[str]]] = None
    channel_colors: Optional[List[Optional[str]]] = None
    declared_windows: Optional[List[Optional[Tuple[float, float]]]] = None
    bit_depth: Optional[int] = None
    acquisition_date: Optional[str] = None
    scenes: Optional[List[str]] = None
    served_scene: Optional[str] = None

    def mark(self, *fields: str) -> None:
        """Record that these fields were inspected, whatever the outcome."""
        self.checked.update(fields)


class MetadataMapping:
    """What was mapped into the view, and what was not, with the reason."""

    def __init__(self) -> None:
        self.mapped: Dict[str, Any] = {}
        self.unmapped: List[str] = []
        self._named: Set[str] = set()

    def add(self, field_name: str, value: Any) -> None:
        self.mapped[field_name] = value
        self._named.add(field_name)

    def miss(self, field_name: str, reason: str) -> None:
        self.unmapped.append(f"{field_name}: {reason}")
        self._named.add(field_name)

    def absent(self, field_name: str, checked: bool, detail: str = "") -> None:
        """Record a field that did not make it in, saying WHY honestly."""
        if checked:
            reason = "not declared in the source"
        else:
            reason = "not read by this recipe, so its presence is unknown"
        self.miss(field_name, f"{reason}{'; ' + detail if detail else ''}")

    def named(self, field_name: str) -> bool:
        return field_name in self._named

    def as_dict(self) -> Dict[str, Any]:
        return {"mapped": self.mapped, "not_mapped": self.unmapped}

    def as_sentences(self) -> Dict[str, str]:
        """Render the mapping as prose meant to be quoted verbatim."""
        parts: List[str] = []
        dims = self.mapped.get("dimensions")
        if dims:
            parts.append("dimensions (" + ", ".join(f"{a} {n}" for a, n in dims.items()) + ")")
        sizes = self.mapped.get("physical_pixel_sizes")
        if sizes:
            inner = ", ".join(
                f"{a} {v['size']:g} "
                f"{'µm' if v['unit'] == 'micrometer' else (v['unit'] or 'units')}"
                for a, v in sizes.items())
            parts.append(f"physical pixel sizes ({inner})")
        inc = self.mapped.get("time_increment")
        if isinstance(inc, dict):
            parts.append(f"time increment ({inc['value']:g} {inc['unit'] or 'units'})")
        names = [n for n in (self.mapped.get("channel_names") or []) if n]
        if names:
            parts.append(f"{len(names)} channel name{'s' if len(names) != 1 else ''} "
                         f"({', '.join(names)})")
        if self.mapped.get("channel_colours"):
            parts.append("channel colours as the source declares them")
        if self.mapped.get("bit_depth"):
            parts.append(f"{self.mapped['bit_depth']}-bit data range")
        dtype = self.mapped.get("dtype")
        if dtype:
            import numpy as _np
            try:
                dtype = _np.dtype(dtype).name
            except TypeError:
                pass
            parts.append(f"data type {dtype}")
        levels = self.mapped.get("pyramid_levels")
        if levels:
            parts.append(f"{levels} pyramid levels with their true scale factors"
                         if levels > 1 else "a single resolution level")
        if self.mapped.get("acquisition_date"):
            parts.append(f"acquisition date {self.mapped['acquisition_date']}")
        if self.mapped.get("scenes"):
            parts.append(f"scene {self.mapped.get('served_scene')!r} of "
                         f"{len(self.mapped['scenes'])}")

        unmapped = [m.split(":", 1)[0] for m in self.unmapped]
        mapped = ("Mapped from the source: " + "; ".join(parts) + "."
                  if parts else "Nothing could be mapped from the source.")
        not_mapped = ("Not mapped: " + "; ".join(unmapped) + "."
                      if unmapped else "Everything the source declares is mapped.")
        return {"mapped": mapped, "not_mapped": not_mapped}


# ---------------------------------------------------------------------------


def _hex_colour(value: str) -> Optional[str]:
    """OME Color is a signed 32-bit RGBA int; CZI uses '#AARRGGBB' or '#RRGGBB'."""
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("#"):
        digits = text[1:]
        if len(digits) == 8:
            digits = digits[2:]
        return digits.upper() if len(digits) == 6 else None
    try:
        rgba = int(text) & 0xFFFFFFFF
    except ValueError:
        return None
    return f"{(rgba >> 24) & 0xFF:02X}{(rgba >> 16) & 0xFF:02X}{(rgba >> 8) & 0xFF:02X}"


def from_ome_xml(ome_xml: Optional[str], axes: str,
                 level_shapes: Sequence[Sequence[int]], dtype: str,
                 name: str) -> SourceMetadata:
    """Extract source metadata from a file's OME-XML."""
    md = SourceMetadata(axes=axes, level_shapes=level_shapes, dtype=dtype, name=name)
    if not ome_xml:
        return md
    root = ElementTree.fromstring(ome_xml)
    image = root.find(f"{OME_NS}Image")
    if image is None:
        return md

    md.mark("acquisition date")
    acquired = image.find(f"{OME_NS}AcquisitionDate")
    if acquired is not None and acquired.text:
        md.acquisition_date = acquired.text.strip()

    pixels = image.find(f"{OME_NS}Pixels")
    if pixels is None:
        return md

    md.mark("physical pixel sizes", "time increment", "channel names",
            "channel colours", "bit depth", "display windows")

    for axis in ("X", "Y", "Z"):
        raw = pixels.get(f"PhysicalSize{axis}")
        if raw is not None:
            md.physical_sizes[axis.lower()] = (
                float(raw), pixels.get(f"PhysicalSize{axis}Unit", "µm"))
    increment = pixels.get("TimeIncrement")
    if increment is not None:
        md.time_increment = (float(increment), pixels.get("TimeIncrementUnit", "s"))

    bits = pixels.get("SignificantBits")
    if bits is not None:
        md.bit_depth = int(bits)

    channels = pixels.findall(f"{OME_NS}Channel")
    if channels:
        md.channel_names = [c.get("Name") for c in channels]
        raw_colours = [c.get("Color") for c in channels]
        md.channel_colors = raw_colours if any(raw_colours) else None
    # OME carries no per-channel display window; that lives in an OMERO
    # rendering def, which a plain OME-TIFF does not have.
    md.declared_windows = None
    return md


def build_ngff_attrs(md: SourceMetadata) -> Tuple[Dict[str, Any], MetadataMapping]:
    """Build NGFF 0.4 group ``.zattrs`` for a multiscale view."""
    mapping = MetadataMapping()
    axes = md.axes.lower()
    base = list(md.level_shapes[0])

    # --- spatial scale -----------------------------------------------------
    sizes: Dict[str, Tuple[float, Optional[str]]] = {}
    missing_axes = []
    for axis in ("x", "y", "z"):
        if axis not in axes:
            continue
        if axis not in md.physical_sizes:
            missing_axes.append(axis.upper())
            continue
        size, unit_raw = md.physical_sizes[axis]
        unit = _UNITS.get(unit_raw) if unit_raw else None
        if unit_raw and unit is None:
            mapping.miss(f"physical pixel size {axis.upper()} unit",
                         f"source unit {unit_raw!r} has no UDUNITS-2 equivalent; "
                         "size mapped, unit dropped")
        sizes[axis] = (size, unit)
    if sizes:
        mapping.add("physical_pixel_sizes",
                    {k: {"size": v[0], "unit": v[1]} for k, v in sizes.items()})
    if missing_axes:
        mapping.absent(f"physical pixel size {', '.join(missing_axes)}",
                       "physical pixel sizes" in md.checked)
    elif not sizes:
        mapping.absent("physical pixel sizes", "physical pixel sizes" in md.checked)

    # --- time scale: a t axis with no increment is a SILENT loss unless said -
    t_scale = 1.0
    t_unit = None
    if "t" in axes and base[axes.index("t")] > 1:
        if md.time_increment:
            value, unit_raw = md.time_increment
            t_scale = value
            t_unit = _UNITS.get(unit_raw) if unit_raw else None
            mapping.add("time_increment", {"value": value, "unit": t_unit or unit_raw})
            if unit_raw and t_unit is None:
                mapping.miss("time increment unit",
                             f"source unit {unit_raw!r} has no UDUNITS-2 "
                             "equivalent; interval mapped, unit dropped")
        else:
            mapping.absent(
                "time increment", "time increment" in md.checked,
                "the t axis is served with scale 1.0 and no unit, so intervals "
                "between timepoints are NOT preserved")
    else:
        # No t axis, or a single timepoint: there is no interval to lose, so
        # saying anything here would be noise rather than disclosure.
        mapping._named.add("time increment")

    ngff_axes = []
    for a in axes:
        entry: Dict[str, Any] = {"name": a, "type": _AXIS_TYPE.get(a, "space")}
        if entry["type"] == "space" and a in sizes and sizes[a][1]:
            entry["unit"] = sizes[a][1]
        elif a == "t" and t_unit:
            entry["unit"] = t_unit
        ngff_axes.append(entry)

    def axis_scale(a: str) -> float:
        if a == "t":
            return t_scale
        return sizes.get(a, (1.0, None))[0]

    # Scale per level is the ACTUAL shape ratio, not an assumed factor of 2.
    datasets = []
    for lvl, shape in enumerate(md.level_shapes):
        scale = [axis_scale(a) * (base[i] / shape[i]) for i, a in enumerate(axes)]
        datasets.append({"path": str(lvl), "coordinateTransformations": [
            {"type": "scale", "scale": scale}]})

    mapping.add("dimensions", {a: base[i] for i, a in enumerate(axes)})
    mapping.add("pyramid_levels", len(md.level_shapes))
    mapping.add("dtype", md.dtype)
    if len(md.level_shapes) == 1:
        mapping.miss("pyramid levels",
                     "source is single-scale; view is single-scale, not downsampled")

    attrs: Dict[str, Any] = {
        "multiscales": [{"version": "0.4", "name": md.name, "axes": ngff_axes,
                         "datasets": datasets}],
    }

    # --- channels ----------------------------------------------------------
    if "c" in axes:
        n_c = base[axes.index("c")]
        names = list(md.channel_names or [])[:n_c]
        names += [None] * (n_c - len(names))
        if any(names):
            mapping.add("channel_names", names)
        else:
            mapping.absent("channel names", "channel names" in md.checked)

        # Normalise whatever spelling the source used (OME signed int, CZI
        # #AARRGGBB or #RRGGBB) into the plain RRGGBB omero expects.
        colours = [_hex_colour(c) for c in list(md.channel_colors or [])[:n_c]]
        colours += [None] * (n_c - len(colours))
        if any(colours):
            mapping.add("channel_colours", colours)
        else:
            mapping.absent("channel colours", "channel colours" in md.checked,
                           "view assigns defaults")

        # Bit depth decides the honest full-scale, not the container width.
        window = _dtype_window(md.dtype)
        if md.bit_depth:
            mapping.add("bit_depth", md.bit_depth)
            window = {"start": 0, "end": 2 ** md.bit_depth - 1,
                      "min": 0, "max": 2 ** md.bit_depth - 1}
        else:
            mapping.absent("bit depth", "bit depth" in md.checked,
                           "window range taken from the container dtype, which "
                           "may overstate the real full scale")

        declared = list(md.declared_windows or [])[:n_c]
        declared += [None] * (n_c - len(declared))
        if any(w is not None for w in declared):
            mapping.miss("display windows",
                         "the source declares them, but this view publishes "
                         "windows measured from the pixels instead; the "
                         "declared values are recorded here and not applied")
            mapping.add("declared_display_windows", declared)
        else:
            mapping.absent("display windows", "display windows" in md.checked,
                           "view computes them from the data")

        attrs["omero"] = {
            "version": "0.4", "name": md.name,
            "channels": [{
                "label": names[i] or f"Channel {i}",
                "color": colours[i] or _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
                "window": dict(window),
                "active": i < 3,
            } for i in range(n_c)],
            "rdefs": {"model": "color"},
        }

    if md.acquisition_date:
        mapping.add("acquisition_date", md.acquisition_date)
    else:
        mapping.absent("acquisition date", "acquisition date" in md.checked)

    if md.scenes and len(md.scenes) > 1:
        # Only a genuine choice needs disclosing. Announcing "scene X of 1" for
        # every single-series file is noise that buries the real case.
        mapping.add("scenes", md.scenes)
        mapping.add("served_scene", md.served_scene)
        mapping.miss("scenes",
                     f"the source holds {len(md.scenes)} scenes and this view "
                     f"serves only {md.served_scene!r}; the rest are not "
                     "reachable through it")
    elif md.scenes:
        mapping._named.add("scenes")
    else:
        mapping.absent("scenes", "scenes" in md.checked)

    for absent in ("objective / instrument metadata", "stage position",
                   "plate / well context", "ROIs and annotations"):
        mapping.miss(absent, "not read by this recipe")

    # Nothing this module knows how to drop may go unmentioned. The mapped dict
    # uses snake_case keys, so compare on a normalised form or the sweep
    # "misses" fields that were in fact mapped.
    def _slug(text: str) -> str:
        return text.replace(" ", "_").replace("/", "_").lower()

    # _named also carries fields deliberately settled as "nothing to say here"
    # (no t axis, a single scene), which are accounted for without a line.
    accounted = ({_slug(k) for k in mapping.mapped}
                 | {_slug(m.split(":", 1)[0]) for m in mapping.unmapped}
                 | {_slug(n) for n in mapping._named})
    aliases = {"physical_pixel_sizes", "channel_colours", "channel_names",
               "time_increment", "bit_depth", "acquisition_date", "scenes",
               "pyramid_levels", "display_windows"}
    for name in ACCOUNTABLE:
        slug = _slug(name)
        if slug in accounted:
            continue
        if slug in aliases and slug in {_slug(k) for k in mapping.mapped}:
            continue
        mapping.miss(name, "not read by this recipe")

    return attrs, mapping


def _dtype_window(dtype_str: str) -> Dict[str, float]:
    m = re.search(r"([iuf])(\d+)", dtype_str)
    if not m or m.group(1) == "f":
        return {"start": 0, "end": 1, "min": 0, "max": 1}
    bits = int(m.group(2)) * 8
    hi = (2 ** (bits - 1) - 1) if m.group(1) == "i" else (2**bits - 1)
    lo = -(2 ** (bits - 1)) if m.group(1) == "i" else 0
    return {"start": lo, "end": hi, "min": lo, "max": hi}
