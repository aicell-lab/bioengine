"""Reading a region over HTTP, against an in-memory OME-Zarr store.

The transport is injected, so the whole path is exercised here with no network: the
fake serves a store built exactly as a writer would lay one out, and returns 404 for
the chunks a sparse store omits. The decisive assertion is the one the clause is
about — the read fetches the chunks the region covers and not the store.
"""

import asyncio
import json

import numpy as np
import pytest

import ngff_store as ns


def _build_store(shape=(200, 300), chunks=(64, 64), omit=(), dtype="|u1", sep="."):
    """An in-memory NGFF store: group .zattrs, array .zarray, and the chunks."""
    rng = np.random.default_rng(3)
    full = rng.integers(1, 255, size=shape, dtype=np.uint8)
    files = {
        "/.zattrs": json.dumps(
            {
                "multiscales": [
                    {
                        "name": "image",
                        "axes": [{"name": "y", "type": "space"},
                                 {"name": "x", "type": "space"}],
                        "datasets": [{"path": "0"}, {"path": "1"}],
                    }
                ]
            }
        ).encode(),
        "/0/.zarray": json.dumps(
            {
                "zarr_format": 2, "shape": list(shape), "chunks": list(chunks),
                "dtype": dtype, "order": "C", "fill_value": 0,
                "compressor": None, "filters": None, "dimension_separator": sep,
            }
        ).encode(),
    }
    ny = -(-shape[0] // chunks[0])
    nx = -(-shape[1] // chunks[1])
    for cy in range(ny):
        for cx in range(nx):
            if (cy, cx) in omit:
                continue
            block = np.zeros(chunks, dtype=np.uint8)
            src = full[cy * chunks[0]:(cy + 1) * chunks[0],
                       cx * chunks[1]:(cx + 1) * chunks[1]]
            block[: src.shape[0], : src.shape[1]] = src
            name = sep.join((str(cy), str(cx)))
            files[f"/0/{name}"] = block.tobytes()
    expect = full.copy()
    for cy, cx in omit:
        expect[cy * chunks[0]:(cy + 1) * chunks[0],
               cx * chunks[1]:(cx + 1) * chunks[1]] = 0
    return files, expect, ny * nx


ROOT = "https://example.invalid/img"


def _getter(files, log=None, base=ROOT):
    """A fake transport. Store keys are paths RELATIVE to the root url, so the base
    is stripped here; an absolute-path lookup would miss every key and show up as a
    404 on the group metadata, which is a harness bug wearing a store bug's clothes."""

    async def get(url):
        assert url.startswith(base), f"url {url!r} is outside the store root {base!r}"
        path = url[len(base):] or "/"
        if log is not None:
            log.append(path)
        body = files.get(path)
        if body is None:
            return 404, b""
        return 200, body

    return get


def test_a_region_read_over_http_matches_the_source_and_fetches_only_its_chunks():
    files, expect, total = _build_store()
    log = []
    get = _getter(files, log)

    async def main():
        opened = await ns.open_level(get, ROOT)
        return await ns.read_region(
            get, opened, {"x": 70, "y": 50, "width": 150, "height": 120}
        )

    out, stats = asyncio.run(main())
    assert np.array_equal(out, expect[50:170, 70:220])
    # THE CLAUSE: the read is proportional to the region, not the store.
    assert stats["chunks_fetched"] == stats["chunks_planned"]
    assert stats["chunks_fetched"] < total
    chunk_gets = [p for p in log if p.startswith("/0/") and not p.endswith(".zarray")]
    assert len(chunk_gets) == stats["chunks_planned"]


def test_an_absent_chunk_is_data_rather_than_an_error():
    """A sparse store is normal: most of a slide was never written."""
    files, expect, _ = _build_store(omit={(1, 1)})
    get = _getter(files)

    async def main():
        opened = await ns.open_level(get, ROOT)
        return await ns.read_region(
            get, opened, {"x": 0, "y": 0, "width": 200, "height": 200}
        )

    out, stats = asyncio.run(main())
    assert stats["chunks_absent"] == 1
    assert np.array_equal(out, expect[0:200, 0:200])


def test_the_finest_level_is_level_zero_and_a_missing_level_is_named():
    files, _, _ = _build_store()
    get = _getter(files)

    async def main():
        opened = await ns.open_level(get, ROOT)
        assert opened["array_url"].endswith("/0")
        assert opened["levels"] == 2
        with pytest.raises(ns.NgffStoreError) as e:
            await ns.open_level(get, ROOT, level=5)
        assert "level 5" in str(e.value)

    asyncio.run(main())


def test_a_region_bigger_than_the_limit_is_refused_before_anything_is_fetched():
    """The refusal is checked against the plan, so an over-large request costs no
    transfer at all. A clamp would quietly serve a different region."""
    files, _, _ = _build_store()
    log = []
    get = _getter(files, log)

    async def main():
        opened = await ns.open_level(get, ROOT)
        log.clear()
        with pytest.raises(ns.NgffStoreError) as e:
            await ns.read_region(
                get, opened, {"x": 0, "y": 0, "width": 200, "height": 200}, max_chunks=2
            )
        assert "over the limit" in str(e.value)

    asyncio.run(main())
    assert log == [], "a refused read still fetched something"


def test_a_store_that_is_not_a_multiscale_image_is_refused_clearly():
    get = _getter({"/.zattrs": json.dumps({"something": "else"}).encode()})

    async def main():
        with pytest.raises(ns.NgffStoreError) as e:
            await ns.open_level(get, ROOT)
        assert "multiscales" in str(e.value)

    asyncio.run(main())


def test_a_missing_or_unparseable_metadata_document_is_named():
    async def main():
        with pytest.raises(ns.NgffStoreError) as e:
            await ns.open_level(_getter({}), ROOT)
        assert "404" in str(e.value)
        with pytest.raises(ns.NgffStoreError) as e:
            await ns.open_level(
                _getter({"/.zattrs": b"{not json"}), ROOT
            )
        assert "not valid JSON" in str(e.value)

    asyncio.run(main())


def test_axes_disagreeing_with_the_array_rank_is_refused_not_guessed():
    files, _, _ = _build_store()
    files["/.zattrs"] = json.dumps(
        {"multiscales": [{"datasets": [{"path": "0"}],
                          "axes": [{"name": "c"}, {"name": "y"}, {"name": "x"}]}]}
    ).encode()

    async def main():
        with pytest.raises(ns.NgffStoreError) as e:
            await ns.open_level(_getter(files), ROOT)
        assert "rank" in str(e.value)

    asyncio.run(main())


def test_axes_absent_are_inferred_only_where_the_convention_is_unambiguous():
    """Pre-0.4 stores omit axes. A rank-2 array is unambiguous; a rank-6 one is not,
    and inventing an order there would read the wrong plane."""
    files, expect, _ = _build_store()
    files["/.zattrs"] = json.dumps(
        {"multiscales": [{"datasets": [{"path": "0"}]}]}
    ).encode()

    async def main():
        opened = await ns.open_level(_getter(files), ROOT)
        assert opened["axes"] == ["y", "x"]

        odd = dict(files)
        odd["/0/.zarray"] = json.dumps(
            {"zarr_format": 2, "shape": [2, 2, 2, 2, 2, 2], "chunks": [1, 1, 1, 1, 1, 1],
             "dtype": "|u1", "order": "C", "fill_value": 0,
             "compressor": None, "filters": None}
        ).encode()
        with pytest.raises(ns.NgffStoreError) as e:
            await ns.open_level(_getter(odd), ROOT)
        assert "unambiguous" in str(e.value)

    asyncio.run(main())


def test_the_dimension_separator_from_the_metadata_is_the_one_used():
    """A store written with '/' keys must be read with '/' keys. Guessing 404s every
    chunk, which would look like a region that is entirely fill value."""
    files, expect, _ = _build_store(sep="/")
    get = _getter(files)

    async def main():
        opened = await ns.open_level(get, ROOT)
        assert opened["spec"]["dimension_separator"] == "/"
        return await ns.read_region(
            get, opened, {"x": 10, "y": 10, "width": 100, "height": 100}
        )

    out, stats = asyncio.run(main())
    assert stats["chunks_absent"] == 0, "every chunk 404'd: the separator was wrong"
    assert np.array_equal(out, expect[10:110, 10:110])


def test_a_byte_budget_becomes_a_chunk_count_that_respects_the_chunk_size():
    """A limit in chunks is meaningless without the chunk size: four chunks of a
    4096-square store is a 64 MB read, four hundred of a 64-square one is under 2 MB.
    So the caller states bytes and this converts."""
    budget = 512 * 1024 * 1024
    big = {"chunks": [4096, 4096], "dtype": "|u1"}      # 16 MiB per chunk
    small = {"chunks": [64, 64], "dtype": "|u1"}        # 4 KiB per chunk
    assert ns.max_chunks_for_budget(big, budget) == 32
    assert ns.max_chunks_for_budget(small, budget) == budget // 4096
    # A wider dtype costs proportionally more chunks' worth of budget.
    assert ns.max_chunks_for_budget({"chunks": [4096, 4096], "dtype": "<u2"}, budget) == 16


def test_a_budget_below_one_chunk_still_allows_one():
    """One chunk is the least a region can cost, so a bound below it would turn the
    limit into a hard failure for small budgets instead of a cap on large reads."""
    assert ns.max_chunks_for_budget({"chunks": [4096, 4096], "dtype": "|u1"}, 1) == 1


def test_a_nonsense_budget_or_chunk_is_refused():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            ns.max_chunks_for_budget({"chunks": [8, 8], "dtype": "|u1"}, bad)
