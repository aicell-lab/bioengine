"""The gallery's cost in ARTIFACT READS, counted rather than timed.

The row asks for reads at N = 1, 10 and 100, and for the single-dataset case to
be kept as a control. Reads and not time, because the row also says a timing
measurement cannot tell "the reads are the dominant cost" from "the round-trip
latency is", and because a count is reproducible on a laptop while a duration is
not.

WHAT IS COUNTED, and why it is two numbers and not one. `list_public_datasets`
makes two kinds of call per public dataset:

* ``_am.read(artifact_id)`` for the name and description. Uncached.
* ``_image_manifest(artifact_id)`` for the image count, which is itself a JSON
  read plus a PAGED directory listing, behind a short-TTL cache.

So the honest statement of today's cost is per-dataset calls of BOTH kinds, and
a fix that removed one and not the other would look like a win on a single
total. The double counts them separately.

THE CONTROL THAT MATTERS HERE is not a second measurement of the same thing: it
is that the count at N=1 does not get WORSE. A cache or an index added for the
hundred-dataset case can easily cost an extra read on the one-dataset case, and
one dataset is what the gallery actually has today.

WHAT WAS MEASURED BEFORE THE FIX, so the numbers below are compared against a
number and not an impression: at N datasets, N `read` calls per visit, repeating
in FULL on the next visit, plus N `read_json_file` and at least N `list_files`
for the image counts -- those already behind a cache, so they did not repeat.
Half the cost already repeated and half did not, which is why the two legs are
counted separately rather than summed.
"""

from __future__ import annotations

import pytest

import broker_core as core
from test_broker_ngff import bk

OWNER = {"id": "github|49943582", "email": "someone@example.org"}


class _CountingAm:
    """Counts every artifact-manager call by name, and answers plausibly."""

    def __init__(self):
        self.calls: dict[str, int] = {}

    def _tick(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1

    async def read(self, artifact_id=None, **_kw):
        self._tick("read")
        return {"manifest": {"name": f"name of {artifact_id}", "description": "d"}}

    async def list_files(
        self, _a=None, dir_path=None, stage=True, limit=None, offset=0
    ):
        self._tick("list_files")
        return []

    async def get_file(self, artifact_id=None, file_path=None, **_kw):
        self._tick("get_file")
        raise FileNotFoundError(file_path)


def _broker(monkeypatch, tmp_path, n):
    b = bk.AnnotationBroker()
    am = _CountingAm()
    for i in range(n):
        meta = core.new_metadata(f"bioimage-io/pub-{i:03d}", OWNER)
        meta["public"] = True
        core.write_metadata(meta, root=tmp_path)

    monkeypatch.setattr(bk, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(b, "_am", am)
    monkeypatch.setattr(b, "_canonical_id", lambda a: a)

    async def _read_json_file(_a, _path, default=None):
        am._tick("read_json_file")
        return default

    monkeypatch.setattr(b, "_read_json_file", _read_json_file)
    b._tam = am
    return b


async def _counts(b, call_count=1):
    for _ in range(call_count):
        await b.list_public_datasets()
    return dict(b._tam.calls)


# ── clause 1: a warm visit's artifact reads do not grow with N ────────────


@pytest.mark.parametrize("n", [1, 10, 100])
async def test_a_WARM_visit_costs_ZERO_artifact_reads_at_any_N(
    monkeypatch, tmp_path, n
):
    """Clause 1. The cold fill is unavoidably N reads -- the data lives in N
    artifacts -- so what must not grow with N is the cost of a visit to a warm
    cache, which under any real traffic is every visit but one per TTL."""
    b = _broker(monkeypatch, tmp_path, n)

    await b.list_public_datasets()
    cold = dict(b._tam.calls)
    assert cold.get("read") == n, cold

    before = dict(b._tam.calls)
    entries = await b.list_public_datasets()
    after = dict(b._tam.calls)

    assert len(entries) == n
    assert after.get("read") == before.get("read"), (
        f"a warm visit made {after.get('read', 0) - before.get('read', 0)} "
        f"artifact read(s) at N={n}; it must make none"
    )
    assert after.get("read_json_file") == before.get("read_json_file")


async def test_the_SINGLE_DATASET_case_does_not_get_WORSE(monkeypatch, tmp_path):
    """The control the row asks to keep. One dataset is what the gallery
    actually has today, and a cache added for the hundred-dataset case can
    easily cost the one-dataset case an extra read."""
    b = _broker(monkeypatch, tmp_path, 1)

    entries = await b.list_public_datasets()
    counts = dict(b._tam.calls)

    assert len(entries) == 1
    assert counts.get("read") == 1, counts
    assert entries[0]["name"] == "name of bioimage-io/pub-000"


async def test_the_CARD_IS_STILL_SERVED_from_the_cache_not_dropped(
    monkeypatch, tmp_path
):
    """A cache that answered with nothing would satisfy the zero-reads test
    perfectly. The warm visit must return the SAME cards."""
    b = _broker(monkeypatch, tmp_path, 3)

    first = await b.list_public_datasets()
    second = await b.list_public_datasets()

    assert [e["name"] for e in second] == [e["name"] for e in first]
    assert all(e["name"] for e in second), second
    assert [e["artifact_id"] for e in second] == [
        e["artifact_id"] for e in first
    ], "the stable order must survive the concurrent fill"


async def test_a_FAILED_read_is_NOT_cached(monkeypatch, tmp_path):
    """A transient failure must not hold a blank card for the whole TTL. The
    sparse card is the response for that visit; the next visit retries."""
    b = _broker(monkeypatch, tmp_path, 1)
    calls = {"n": 0}

    async def _flaky(artifact_id=None, **_kw):
        b._tam._tick("read")
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("the artifact manager blinked")
        return {"manifest": {"name": "recovered", "description": "d"}}

    monkeypatch.setattr(b._tam, "read", _flaky)

    first = await b.list_public_datasets()
    assert first[0].get("name") is None, first

    second = await b.list_public_datasets()
    assert second[0]["name"] == "recovered", second
    assert calls["n"] == 2, "the failed read must be retried, not served from cache"


# ── clause 2: the cache can never serve what the read path refuses ────────


async def test_UNPUBLISHING_removes_the_card_even_while_it_is_CACHED(
    monkeypatch, tmp_path
):
    """Clause 2, and the one that matters for safety rather than speed. The
    gallery filters on the same `public` flag `resolve_role` uses to decide
    whether an anonymous caller may read a dataset at all, so a cache that
    could serve an unpublished dataset would advertise one the read path then
    refuses.

    The cache is keyed by artifact id and holds only a name and a description.
    It is consulted ONLY for ids `iter_public_metadata` has already returned,
    so a warm entry for an unpublished dataset is unreachable by construction
    rather than by remembering to clear it.
    """
    b = _broker(monkeypatch, tmp_path, 2)

    await b.list_public_datasets()
    assert "bioimage-io/pub-000" in b._public_card_cache

    meta = core.read_metadata("bioimage-io/pub-000", root=tmp_path)
    meta["public"] = False
    core.write_metadata(meta, root=tmp_path)

    entries = await b.list_public_datasets()

    assert [e["artifact_id"] for e in entries] == ["bioimage-io/pub-001"], entries
    # ...and the stale entry being still IN the cache is the point: it is
    # harmless, because the cache does not decide who is public.
    assert "bioimage-io/pub-000" in b._public_card_cache


async def test_SET_PUBLIC_drops_that_card_and_only_that_card(monkeypatch, tmp_path):
    """So a rename or a republish shows up at once rather than after the TTL,
    and a publish does not cost every other card its cache."""
    b = _broker(monkeypatch, tmp_path, 3)
    monkeypatch.setattr(
        b,
        "_require_role",
        lambda context, artifact_id, minimum: (
            core.read_metadata(artifact_id, root=tmp_path),
            {"id": "u1"},
            "owner",
        ),
    )

    async def _noop_permissions(_a, _m):
        return None

    monkeypatch.setattr(b, "_apply_permissions", _noop_permissions)
    await b.list_public_datasets()
    assert set(b._public_card_cache) == {
        "bioimage-io/pub-000",
        "bioimage-io/pub-001",
        "bioimage-io/pub-002",
    }

    await b.set_public("bioimage-io/pub-001", True, context=None)

    assert set(b._public_card_cache) == {
        "bioimage-io/pub-000",
        "bioimage-io/pub-002",
    }, b._public_card_cache


async def test_the_TTL_IS_WHAT_EXPIRES_a_card(monkeypatch, tmp_path):
    """The cache must not be permanent. A dataset renamed by its owner, with no
    `set_public` call to invalidate anything, still shows the new name once the
    window passes."""
    b = _broker(monkeypatch, tmp_path, 1)

    await b.list_public_datasets()
    stamped, card = b._public_card_cache["bioimage-io/pub-000"]
    b._public_card_cache["bioimage-io/pub-000"] = (
        stamped - bk.PUBLIC_CARD_CACHE_TTL_S - 1,
        card,
    )
    before = dict(b._tam.calls)

    await b.list_public_datasets()

    assert (
        dict(b._tam.calls)["read"] == before["read"] + 1
    ), "an expired card must be re-read"


# ── the fill is concurrent, not one dataset after another ─────────────────


async def test_the_COLD_FILL_IS_CONCURRENT_and_BOUNDED(monkeypatch, tmp_path):
    """The row names the serialisation as part of the cost: the reads were
    issued "one dataset after another, with no upper limit". Concurrency does
    not reduce the READ count and is not the fix for clause 1 -- it is the fix
    for the wall clock, and it is bounded so one anonymous visit cannot open an
    unbounded number of connections.

    Measured by watching how many reads are in flight at once, not by timing.
    """
    import asyncio

    n = 40
    b = _broker(monkeypatch, tmp_path, n)
    state = {"now": 0, "peak": 0}

    async def _slow_read(artifact_id=None, **_kw):
        b._tam._tick("read")
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        state["now"] -= 1
        return {"manifest": {"name": str(artifact_id), "description": "d"}}

    monkeypatch.setattr(b._tam, "read", _slow_read)

    await b.list_public_datasets()

    assert state["peak"] > 1, "the reads were still issued one after another"
    assert state["peak"] <= bk.PUBLIC_CARD_FILL_CONCURRENCY, (
        f"{state['peak']} reads were in flight; the bound is "
        f"{bk.PUBLIC_CARD_FILL_CONCURRENCY}"
    )
