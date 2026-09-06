"""Tier A arms: the client holds the index and range-reads the original itself.

Two arms differing ONLY in where the original lives — remote object storage
versus a local copy of the very same file. That pairing is what separates
storage locality from route, which a view-vs-converted-copy comparison would
otherwise confound.
"""
from __future__ import annotations

import io
import json
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


def _index_for(source: str) -> Dict[str, Any]:
    """Build a reference index over a TIFF, local or remote."""
    import fsspec
    import tifffile

    if "://" in source:
        handle = fsspec.filesystem(source.split("://")[0],
                                   block_size=2 << 20).open(source, "rb")
    else:
        handle = source
    tf = tifffile.TiffFile(handle)
    buf = io.StringIO()
    tf.series[0].aszarr().write_fsspec(buf, url=str(source).rsplit("/", 1)[0])
    refs = json.loads(buf.getvalue())
    tf.close()
    return refs


def _reader(refs: Dict[str, Any], local_path: str | None):
    """Fetch a chunk by its reference entry, decoding the source codec."""
    import httpx
    import imagecodecs.numcodecs
    import numcodecs

    imagecodecs.numcodecs.register_codecs(verbose=False)
    meta = json.loads(refs["0/.zarray"])
    codec = numcodecs.get_codec(meta["compressor"]) if meta.get("compressor") else None
    client = httpx.Client(timeout=900, follow_redirects=True)

    def read(idx: Tuple[int, ...]):
        key = "0/" + ".".join(str(i) for i in idx)
        entry = refs.get(key)
        if not isinstance(entry, list):
            return b""
        url, offset, length = entry
        if local_path is not None:
            with open(local_path, "rb") as fh:
                fh.seek(offset)
                raw = fh.read(length)
        else:
            r = client.get(url, headers={
                "Range": f"bytes={offset}-{offset + length - 1}"})
            raw = r.content
        return codec.decode(raw) if codec else raw

    return read


def tier_a_arms(remote_url: str, local_tif, indices: List[Tuple[int, ...]],
                stats: Callable, time_reads: Callable) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    refs_remote = _index_for(remote_url)
    read_remote = _reader(refs_remote, local_path=None)
    cold, warm = time_reads(read_remote, indices, True)
    out["B_view_tierA_remote"] = {
        "description": "client holds the byte-offset index and range-reads the "
                       "original in the bucket; the platform is not in the data path",
        "storage": "remote (GCS)", "route": "view Tier A (zero-copy)",
        "cold": stats(cold), "warm_client_side": stats(warm)}

    refs_local = _index_for(str(local_tif))
    read_local = _reader(refs_local, local_path=str(local_tif))
    cold, warm = time_reads(read_local, indices, True)
    out["C_view_tierA_local"] = {
        "description": "same index and same route, but against a local copy of "
                       "the UNCONVERTED original",
        "storage": "local disk", "route": "view Tier A (zero-copy)",
        "cold": stats(cold), "warm_os_page_cache": stats(warm)}
    return out
