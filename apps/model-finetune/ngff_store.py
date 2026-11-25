"""Read a region out of a remote OME-Zarr store over HTTP.

The last of three pieces. :mod:`ngff_region` says which chunks a region needs,
:mod:`ngff_chunk` says what a chunk's bytes mean, and this one does the talking: it
resolves a multiscale group to one array, reads that array's metadata, and fetches
exactly the chunks the plan asked for.

**The transport is injected, as an async ``get(url) -> (status, body)``.** The entry
already owns a retrying ``httpx`` client that knows about presigned URLs and
transient 403/404/5xx during a credential cycle, so re-implementing retries here
would be a second policy to keep in step with the first. Injecting it also means
every path below — including a 404 standing for an absent chunk — is testable against
an in-memory store with no network.

**On duplicating the broker's multiscale parsing.** The annotation broker parses the
same `.zattrs` and does it well, but the two apps are separate Ray deployments with
separate pip environments and no importable package between them, so a small amount
of duplication is forced. It is kept to the one thing needed — resolving a level to
its array path — rather than copying the broker's whole NGFF layer, and the broker
remains the place where that vocabulary is defined.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import ngff_chunk as _chunk
import ngff_region as _region

# An async callable returning (status_code, body_bytes). Non-2xx must NOT raise:
# a 404 on a chunk is data (Zarr omits chunks that are entirely the fill value), so
# the status has to reach this module rather than being turned into an exception by
# the transport.
Getter = Callable[[str], Awaitable[Tuple[int, bytes]]]


class NgffStoreError(RuntimeError):
    """The store could not be read as an OME-Zarr multiscale image."""


async def _read_json(get: Getter, url: str) -> Dict[str, Any]:
    status, body = await get(url)
    if status == 404:
        raise NgffStoreError(f"{url} does not exist (404)")
    if not 200 <= status < 300:
        raise NgffStoreError(f"{url} returned HTTP {status}")
    try:
        doc = json.loads(bytes(body).decode("utf-8"))
    except Exception as exc:
        raise NgffStoreError(f"{url} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise NgffStoreError(f"{url} should hold an object, got {type(doc).__name__}")
    return doc


def multiscale_levels(zattrs: Dict[str, Any]) -> Dict[str, Any]:
    """``{"paths", "axes"}`` for the first multiscale in a group's attributes.

    Level order in NGFF is finest first, so ``paths[0]`` is full resolution — which
    is the level a region annotation refers to. Chosen explicitly rather than by
    taking whatever comes first in iteration order.
    """
    multiscales = zattrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise NgffStoreError(
            "group attributes carry no 'multiscales' list, so this is not an "
            "OME-Zarr multiscale image"
        )
    first = multiscales[0]
    if not isinstance(first, dict):
        raise NgffStoreError("the first multiscale entry is not an object")
    datasets = first.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise NgffStoreError("the multiscale declares no datasets, so it has no levels")
    paths: List[str] = []
    for d in datasets:
        if isinstance(d, dict) and d.get("path") is not None:
            paths.append(str(d["path"]))
        elif isinstance(d, str):
            paths.append(d)
        else:
            raise NgffStoreError(f"a dataset entry has no usable path: {d!r}")
    axes_raw = first.get("axes") or []
    axes: List[str] = []
    for a in axes_raw:
        if isinstance(a, dict) and a.get("name") is not None:
            axes.append(str(a["name"]).lower())
        elif isinstance(a, str):
            axes.append(a.lower())
    return {"paths": paths, "axes": axes}


async def open_level(
    get: Getter,
    root_url: str,
    level: int = 0,
    multiscale_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve a multiscale group to one array, and read that array's metadata.

    Returns ``{"array_url", "spec", "axes", "level", "levels"}``. ``spec`` is a
    parsed ``.zarray``, so an unsupported codec or a Zarr v3 array is refused here —
    before any pixel is requested — rather than after a region's worth of fetches.
    """
    group_url = root_url.rstrip("/")
    if multiscale_path:
        group_url = f"{group_url}/{str(multiscale_path).strip('/')}"
    ms = multiscale_levels(await _read_json(get, f"{group_url}/.zattrs"))
    paths = ms["paths"]
    if not 0 <= level < len(paths):
        raise NgffStoreError(
            f"level {level} does not exist; this image has {len(paths)} "
            f"({', '.join(paths)})"
        )
    array_url = f"{group_url}/{paths[level].strip('/')}"
    spec = _chunk.parse_zarray(await _read_json(get, f"{array_url}/.zarray"))
    axes = ms["axes"]
    if not axes:
        # NGFF before 0.4 omitted axes. Infer from rank rather than refusing, but
        # only for the shapes where the convention is unambiguous.
        rank = len(spec["shape"])
        inferred = {2: ["y", "x"], 3: ["c", "y", "x"], 4: ["c", "z", "y", "x"],
                    5: ["t", "c", "z", "y", "x"]}.get(rank)
        if inferred is None:
            raise NgffStoreError(
                f"the group declares no axes and a rank-{rank} array has no "
                "unambiguous default; this store needs explicit axes"
            )
        axes = inferred
    if len(axes) != len(spec["shape"]):
        raise NgffStoreError(
            f"the group declares {len(axes)} axes {axes} but the array has rank "
            f"{len(spec['shape'])}; one of the two is wrong and guessing would read "
            "the wrong plane"
        )
    return {
        "array_url": array_url,
        "spec": spec,
        "axes": axes,
        "level": level,
        "levels": len(paths),
    }


async def read_region(
    get: Getter,
    opened: Dict[str, Any],
    region: Dict[str, int],
    index: Optional[Dict[str, int]] = None,
    out: Any = None,
    max_chunks: Optional[int] = None,
) -> Tuple[Any, Dict[str, int]]:
    """Fetch and assemble one region, returning ``(array, stats)``.

    The stats are returned rather than attached to the array: a numpy array rejects
    arbitrary attributes, so attaching them would have dropped them silently on the
    one type this is always called with. They are how a caller proves the read was
    proportional to the region — ``chunks_fetched`` against the store's total.

    ``max_chunks`` is a refusal, not a clamp: a caller that asks for more of the
    store than it meant to should be told, not quietly served a slow read. It is
    checked against the PLAN, so nothing is fetched before refusing.
    """
    import numpy as np

    spec = opened["spec"]
    plan = _region.plan_region_read(
        spec["shape"], spec["chunks"], opened["axes"], region, index
    )
    if max_chunks is not None and plan["chunks_touched"] > max_chunks:
        raise NgffStoreError(
            f"this region needs {plan['chunks_touched']} chunks, over the limit of "
            f"{max_chunks}; narrow the region or raise the limit deliberately"
        )
    if out is None:
        out = np.zeros((plan["height"], plan["width"]), dtype=np.dtype(spec["dtype"]))

    fetched = missing = 0
    for part in plan["chunks"]:
        url = _chunk.chunk_url(
            opened["array_url"], part["key"], spec["dimension_separator"]
        )
        status, body = await get(url)
        if status == 404:
            # Zarr omits a chunk that is entirely the fill value, so this is data.
            missing += 1
            block = _chunk.fill_chunk(spec)
        elif 200 <= status < 300:
            fetched += 1
            block = _chunk.decode_chunk(bytes(body), spec)
        else:
            raise NgffStoreError(f"chunk {url} returned HTTP {status}")
        (sy0, sy1), (sx0, sx1) = part["src"]
        (dy0, dy1), (dx0, dx1) = part["dst"]
        out[dy0:dy1, dx0:dx1] = block[..., sy0:sy1, sx0:sx1]

    return out, {
        "chunks_planned": plan["chunks_touched"],
        "chunks_fetched": fetched,
        "chunks_absent": missing,
    }
