#!/usr/bin/env python3
"""Fetch the local half of the demo: public, licence-verified image files.

The point of the demo is local-vs-remote, so the local side needs real files
on disk. Everything here is openly licensed and the licence was checked against
the repository's own API, not a landing page — see LICENCES.md for the record.

    python fetch_demo_data.py [--dest data] [--skip-large]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import httpx

# (filename, url, licence, attribution, bytes, md5 or None, is_large)
FILES = [
    (
        "MouseBrain_41Slices_1Tile_3Channel_2Illuminations_2Angles.czi",
        "https://zenodo.org/records/8305531/files/"
        "MouseBrain_41Slices_1Tile_3Channel_2Illuminations_2Angles.czi?download=1",
        "CC-BY-4.0",
        "Chiaruttini N., EPFL BIOP — 'CZI file examples', Zenodo, "
        "doi:10.5281/zenodo.8305531. Cleared mouse brain, Zeiss LightSheet Z1; "
        "sample Yanqi Liu (Carl Petersen lab, LSENS), imaging Olivier Burri.",
        1476382048,
        "2dc5f8469df878779cc582ed837e0c5f",
        True,
    ),
    (
        "S=1_3x3_T=3_Z=4_CH=2.czi",
        "https://zenodo.org/records/7015307/files/S%3D1_3x3_T%3D3_Z%3D4_CH%3D2.czi"
        "?download=1",
        "CC-BY-4.0",
        "Rhode S. — 'CZI dataset with artificial test camera images', Zenodo, "
        "doi:10.5281/zenodo.7015307.",
        108700000,
        None,
        False,
    ),
    (
        "xz-scan-lsm980.czi",
        "https://zenodo.org/records/8305531/files/xz-scan-lsm980.czi?download=1",
        "CC-BY-4.0",
        "Chiaruttini N., EPFL BIOP — 'CZI file examples', Zenodo, "
        "doi:10.5281/zenodo.8305531. Zeiss LSM980 xz scan.",
        1900000,
        None,
        False,
    ),
]


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {dest.name}: {pct:5.1f}%  "
                          f"{done/1e6:.0f}/{total/1e6:.0f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest",
                    default="../../.dev/omezarr-view-data",
                    help="where to put demo files; kept OUTSIDE the app directory "
                         "because upload_app ships the whole directory")
    ap.add_argument("--skip-large", action="store_true",
                    help="skip files over ~1 GB")
    args = ap.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Local demo data — licences verified from the source repository API",
             "",
             "| file | licence | attribution |", "|---|---|---|"]

    for name, url, licence, attribution, size, expect_md5, large in FILES:
        if large and args.skip_large:
            print(f"skip (large): {name}")
            continue
        path = dest_dir / name
        if path.exists() and path.stat().st_size > 0:
            print(f"have: {name} ({path.stat().st_size/1e6:.0f} MB)")
        else:
            print(f"get:  {name} (~{size/1e6:.0f} MB)")
            download(url, path)
        if expect_md5:
            actual = md5(path)
            status = "ok" if actual == expect_md5 else f"MISMATCH (got {actual})"
            print(f"      md5 {status}")
            if actual != expect_md5:
                return 1
        lines.append(f"| `{name}` | {licence} | {attribution} |")

    (dest_dir / "LICENCES.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {dest_dir/'LICENCES.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
