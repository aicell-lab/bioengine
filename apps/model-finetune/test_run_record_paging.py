"""The shared run history is read a PAGE at a time, so it must be read to the end.

WHAT THE SERVER ACTUALLY DOES, measured rather than assumed. `list_files` takes
``limit`` (default **1000**, ``ge=1 le=10000``) and ``offset`` (default 0), and
returns a bare ``List[Dict]`` whose entries carry only ``name, type, size,
last_modified``. Three things follow and the third is the one that bites: one
call is a page, not a directory; following the pagination is possible because
``offset`` exists; and the ceiling of 10000 means asking for the maximum is not
a fix but a higher cap. There is nowhere in a bare list for a completeness flag
to live, so the caller has to carry one.

THE SHARPER CONSEQUENCE IS NOT A SHORT ANSWER, IT IS AN EVICTION. The listing
read also expires cached records that have vanished from the artifact, so a
truncated page does not merely omit the runs past it -- it DELETES them from the
history this replica serves, including runs it had already fetched. That is why
an incomplete listing must suppress the eviction rather than simply be logged.

EVERY FAKE HERE PAGES. A double that returns the whole directory in one call
models a server that does not exist, and every paging defect passes against it.
Each one records what it was ASKED for as well as what it served, because the
entire defect is that those two differ.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import training
from test_run_record_migration import ADMIN, FakeAM, FakeHTTP, _app, _entry_cls, _run
from test_run_record_migration import store  # noqa: F401  -- a fixture, used by name

APP = "an-app"


@pytest.fixture()
def shared(monkeypatch):
    """A record store this replica reads but did not write."""
    monkeypatch.setattr(training, "record_store_id", lambda: APP)

    def build(n, *, page_size=None, ignores_offset=False):
        am = FakeAM(page_size=page_size, ignores_offset=ignores_offset)
        ids = []
        for i in range(n):
            sid = f"2026-01-01-{i:06d}-aaaaaaaa"
            ids.append(sid)
            am.store(
                f"{training.run_record_dir(APP)}/{sid}.json",
                json.dumps({"session_id": sid, "status": "COMPLETED"}),
            )
        cls = _entry_cls()
        inst = object.__new__(cls)
        inst._am = am
        inst._http = FakeHTTP(am)
        inst._runs_cache = None
        inst._runs_meta = {}
        inst._runs_cache_at = 0.0
        inst.RUNS_RETRY_DELAY_S = 0
        inst.RUNS_CACHE_TTL_S = 0

        async def _get_am():
            return am

        inst._runs_artifact_manager = _get_am
        return cls, inst, am, ids

    return build


def _read(cls, inst):
    return asyncio.run(cls._read_run_records(inst))


# --------------------------------------------------------------------------
# Clause 1 + 2: the reader follows the pagination to the end.
# --------------------------------------------------------------------------


def test_MORE_ENTRIES_THAN_ONE_PAGE_are_ALL_returned(shared):
    """The core property: 10 records behind a 4-entry page, all 10 read."""
    cls, inst, am, ids = shared(10, page_size=4)
    inst.RUNS_LIST_PAGE = 4

    got = _read(cls, inst)

    assert got is not None, "the listing failed outright"
    assert set(got) == set(ids), (
        f"{len(got)} of {len(ids)} records came back; missing "
        f"{sorted(set(ids) - set(got))}"
    )


def test_a_RUN_PAST_THE_FIRST_PAGE_is_in_the_answer(shared):
    """Named rather than counted: the LAST id is the one a single page loses."""
    cls, inst, am, ids = shared(10, page_size=4)
    inst.RUNS_LIST_PAGE = 4

    got = _read(cls, inst)

    assert ids[-1] in got, (
        f"the run at position {len(ids)} behind a 4-entry page is missing, so "
        "the reader stopped at a page boundary"
    )


def test_the_LISTING_ASKS_with_a_LIMIT_and_a_RISING_OFFSET(shared):
    """What was ASKED, not what was served -- the two differ, and that is the bug.

    A reader could return every record by accident, if the store happened to
    serve everything, while still asking only once. This pins the asking.
    """
    cls, inst, am, ids = shared(10, page_size=4)
    inst.RUNS_LIST_PAGE = 4

    _read(cls, inst)

    offsets = [ask[2] for ask in am.list_asks]
    limits = {ask[1] for ask in am.list_asks}
    assert limits == {4}, f"the reader did not ask with its own page size: {limits}"
    # 4 + 4 + 2 served, then the empty page: the offset advances by what was
    # SERVED, not by what was asked for. Advancing by the limit would skip the
    # entries a short page did not return -- the same defect as stopping on a
    # short page, one step along.
    assert offsets == [0, 4, 8, 10], (
        f"the offsets asked for were {offsets}; a reader that stopped at the "
        "first page asks only for offset 0"
    )


def test_a_page_SHORTER_than_the_ASK_is_NOT_the_end(shared):
    """The server may cap the limit below the ask; that is an ordinary answer.

    Stopping on a short page is the repair that looks right and reintroduces
    the defect one layer down: ask for 1000, be served 3, conclude there are 3.
    """
    cls, inst, am, ids = shared(9, page_size=3)
    inst.RUNS_LIST_PAGE = 1000  # ask big, get 3 at a time

    got = _read(cls, inst)

    assert set(got) == set(ids), (
        f"{len(got)} of {len(ids)} came back: the reader treated a page shorter "
        "than its ask as the end of the directory"
    )
    assert [a[1] for a in am.list_asks] == [1000, 1000, 1000, 1000], am.list_asks


def test_an_EMPTY_page_ENDS_the_loop_rather_than_a_short_one(shared):
    """Termination is on empty, and it does terminate: one extra call, no more."""
    cls, inst, am, ids = shared(8, page_size=4)
    inst.RUNS_LIST_PAGE = 4

    _read(cls, inst)

    # 0..3, 4..7, then the empty page at 8. Not more.
    assert [a[2] for a in am.list_asks] == [0, 4, 8], am.list_asks


# --------------------------------------------------------------------------
# A store that cannot be paged must say so, not pretend.
# --------------------------------------------------------------------------


def test_a_store_that_IGNORES_OFFSET_is_reported_INCOMPLETE(shared):
    """It must terminate, and it must NOT claim completeness it cannot have."""
    cls, inst, am, ids = shared(10, page_size=4, ignores_offset=True)
    inst.RUNS_LIST_PAGE = 4

    entries, complete = asyncio.run(cls._list_run_record_files(inst, am, APP))

    assert complete is False, "a store serving page one forever was called complete"
    assert len(entries) == 4, entries
    assert len(am.list_asks) == 2, (
        f"the loop made {len(am.list_asks)} calls against a store that ignores "
        "offset; it must stop, not spin"
    )


def test_a_PAGEABLE_store_IS_reported_complete(shared):
    """The control: a reader that always said False would pass the test above."""
    cls, inst, am, ids = shared(10, page_size=4)
    inst.RUNS_LIST_PAGE = 4

    entries, complete = asyncio.run(cls._list_run_record_files(inst, am, APP))

    assert complete is True, "a store that pages correctly was called incomplete"
    assert len(entries) == 10, entries


# --------------------------------------------------------------------------
# The eviction, which is the consequence that loses data rather than hiding it.
# --------------------------------------------------------------------------


def test_an_INCOMPLETE_listing_does_NOT_EVICT_an_ALREADY_KNOWN_run(shared):
    """Absence from a listing that could not be proven whole is not deletion."""
    cls, inst, am, ids = shared(10, page_size=4, ignores_offset=True)
    inst.RUNS_LIST_PAGE = 4
    # This replica already knows about a run that the truncated listing cannot
    # reach -- exactly the state a second poll finds itself in.
    far = ids[-1]
    inst._runs_cache = {far: {"session_id": far, "status": "COMPLETED"}}
    inst._runs_meta = {f"{far}.json": (10, 1.0)}

    got = _read(cls, inst)

    assert far in got, (
        "a run already in the cache was evicted because a listing that could "
        "not be proven complete did not mention it: the truncation does not "
        "merely hide runs, it deletes them from this replica's answer"
    )


def test_a_COMPLETE_listing_DOES_evict_a_run_that_is_really_gone(shared):
    """The control. A reader that never evicted would pass the test above too."""
    cls, inst, am, ids = shared(4, page_size=4)
    inst.RUNS_LIST_PAGE = 4
    ghost = "2026-01-01-999999-dddddddd"
    inst._runs_cache = {ghost: {"session_id": ghost, "status": "COMPLETED"}}
    inst._runs_meta = {f"{ghost}.json": (10, 1.0)}

    got = _read(cls, inst)

    assert ghost not in got, (
        "a record absent from a listing that WAS proven complete stayed in the "
        "cache, so deletion never takes effect"
    )
    assert set(got) == set(ids), got


# --------------------------------------------------------------------------
# Clause 3: the backfill cannot report complete on a read-back it cannot trust.
# --------------------------------------------------------------------------


def _migrate(store, n, **am_kw):
    for i in range(n):
        store(f"2026-02-02-{i:06d}-aaaaaaaa", owner="me")
    am = FakeAM(**am_kw)
    cls, inst = _app(am=am)
    inst.RUNS_LIST_PAGE = 2
    plan = _run(cls, inst)
    applied = _run(cls, inst, apply=True, confirm=str(plan["planned"]))
    return applied, am


def test_the_BACKFILL_is_NOT_complete_when_the_READ_BACK_cannot_be_proven_whole(
    store,
):
    """A migration graded against a subset of the artifact is not a migration."""
    plan, am = _migrate(store, 6, page_size=2, ignores_offset=True)

    assert plan["complete"] is False, plan
    assert plan["verify_error"], "nothing recorded WHY the read-back was refused"


def test_the_REFUSAL_BLAMES_THE_READ_and_not_the_records(store):
    """A truncated read-back fails closed either way; only one way says why.

    With the read-back truncated, every id past the page has no hash and lands
    in `mismatched` -- so the migration would already report incomplete, while
    describing a set of records that are present and correct as if they were
    corrupt. The reason has to name the listing.
    """
    plan, am = _migrate(store, 6, page_size=2, ignores_offset=True)

    assert "complete" in plan["verify_error"], plan["verify_error"]
    assert "listing" in plan["verify_error"], plan["verify_error"]


def test_the_BACKFILL_IS_complete_when_the_read_back_PAGES_TO_THE_END(store):
    """The control: a check that refused every migration would pass the above."""
    plan, am = _migrate(store, 6, page_size=2)

    assert plan["complete"] is True, plan
    assert plan["verify_error"] is None, plan
    assert plan["verified"] == 6, plan
    assert [a[2] for a in am.list_asks] == [0, 2, 4, 6], am.list_asks


def test_the_READ_BACK_paging_is_what_makes_SIX_RECORDS_verifiable(store):
    """Named rather than counted: without paging, verified would stop at 2."""
    plan, am = _migrate(store, 6, page_size=2)

    assert plan["verified"] == 6 and not plan["mismatched"], plan
