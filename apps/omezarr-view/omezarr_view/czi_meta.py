"""Read the fields BioIO's own accessors do not expose, from the raw metadata.

BioIO gives dimensions, channel names and physical pixel sizes. It does not give
channel colours, declared display windows, bit depth, the time increment, or the
scene list — and a view that never looks at them cannot honestly say the source
does not declare them. Three false "not declared in the source" claims came from
exactly that gap.

Everything here is best-effort and reports what it INSPECTED separately from
what it found, so an unrecognised layout produces "not read by this recipe"
rather than a confident falsehood.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree


def _text(node, path: str) -> Optional[str]:
    found = node.find(path)
    return found.text.strip() if found is not None and found.text else None


def _as_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def read_czi_extras(metadata: Any) -> Tuple[Dict[str, Any], List[str]]:
    """Return (fields, inspected) from a CZI metadata tree.

    ``inspected`` names what was actually looked for, so the caller can tell
    "absent" apart from "never examined".
    """
    fields: Dict[str, Any] = {}
    inspected: List[str] = []
    if metadata is None or not hasattr(metadata, "find"):
        return fields, inspected

    root: ElementTree.Element = metadata
    image = root.find(".//Information/Image")
    if image is None:
        return fields, inspected

    # --- bit depth --------------------------------------------------------
    inspected.append("bit depth")
    bits = _text(image, "ComponentBitCount")
    if bits and bits.isdigit():
        fields["bit_depth"] = int(bits)

    # --- time increment ---------------------------------------------------
    inspected.append("time increment")
    increment = _as_float(_text(image, "Dimensions/T/Positions/Interval/Increment"))
    if increment is not None:
        # CZI expresses the T interval in seconds.
        fields["time_increment"] = (increment, "s")

    # --- channel colours and declared display windows ---------------------
    inspected.extend(["channel colours", "display windows"])
    channels = root.findall(".//DisplaySetting/Channels/Channel")
    if not channels:
        channels = root.findall(".//Dimensions/Channels/Channel")
    colours: List[Optional[str]] = []
    windows: List[Optional[Tuple[float, float]]] = []
    for channel in channels:
        colours.append(_text(channel, "Color"))
        low = _as_float(_text(channel, "Low"))
        high = _as_float(_text(channel, "High"))
        # Low/High are normalised fractions of the declared bit range. Files
        # commonly carry High alone, so requiring both silently discarded the
        # declaration and produced a false "not declared" line.
        if high is not None and "bit_depth" in fields:
            full = 2 ** fields["bit_depth"] - 1
            windows.append(((low or 0.0) * full, high * full))
        else:
            windows.append(None)
    if any(colours):
        fields["channel_colors"] = colours
    if any(w is not None for w in windows):
        fields["declared_windows"] = windows

    # --- scenes -----------------------------------------------------------
    inspected.append("scenes")
    scenes = [s.get("Name") or s.get("Id") or f"scene {i}"
              for i, s in enumerate(root.findall(".//Dimensions/S/Scenes/Scene"))]
    if scenes:
        fields["scenes"] = scenes

    # --- acquisition date -------------------------------------------------
    inspected.append("acquisition date")
    acquired = _text(image, "AcquisitionDateAndTime")
    if acquired:
        fields["acquisition_date"] = acquired

    return fields, inspected
