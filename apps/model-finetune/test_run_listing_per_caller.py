"""The run listing returns the caller's own runs, and nobody else's.

The listing used to return every run on the worker to whoever asked. These tests
pin the three visibility rules and, more importantly, the two NARROWINGS that
make the ownerless rule correct — both of which were measured against the live
platform before being written down, and both of which the obvious implementation
gets wrong:

* read permission on the hosting workspace is NOT authority over other people's
  runs, even though the app's existing caller classifier counts it as operator;
* every token-less caller is minted a fresh identity holding ADMIN on its own
  personal workspace, so "holds an operator grant somewhere" is true for every
  caller alive and would publish the whole unowned history to anonymous callers.

**Every test that asserts somebody sees NOTHING is paired with a demonstration
that the runs are really there.** Without that pairing, a fixture that silently
wrote no sessions would make the whole file pass while measuring nothing — the
failure this suite is least able to notice on its own.
"""

import json
import time

import pytest

import training

ME = "github|49943582"
OTHER = "github|000111"
ANON_A = "anonymouz-periwinkle-honeysuckle-17983078"
ANON_B = "anonymouz-dot-basement-35411643"


def ctx(uid, *, anonymous=True, grants=None):
    """A caller context shaped like the one Hypha injects.

    Field names and the nesting are taken from the measured live contexts, not
    invented: ``user.id``, ``user.is_anonymous``, ``user.scope.workspaces``.
    """
    user = {"id": uid, "is_anonymous": anonymous}
    if grants is not None:
        user["scope"] = {"workspaces": grants}
    return {"user": user}


ADMIN = ctx(ME, anonymous=False, grants={"bioimage-io": "a"})


@pytest.fixture()
def runs(tmp_path, monkeypatch):
    """Plant runs on disk with given owners and return a writer.

    ``sessions_root`` and ``session_dir`` are patched together: the listing walks
    the root and ``get_status`` resolves each session through ``session_dir``, so
    patching only one of them would read two different directories.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(training, "sessions_root", lambda: root)
    monkeypatch.setattr(training, "session_dir", lambda sid: root / sid)

    def add(sid, owner, *, owner_anonymous=False, omit_key=False):
        """Plant one run.

        ``owner=None`` writes the key with a NULL value, which is the shape a
        real unidentified run has: `caller_identity` returns
        ``{"owner": None, "owner_anonymous": None}`` and the PREPARING write
        spreads it, so the key is present and null. ``omit_key=True`` writes no
        key at all, which is the shape of the 261 runs recorded before the field
        existed. Both are "ownerless" and the filter must treat them alike --
        and a fixture that only produced one of them could not tell.
        """
        d = root / sid
        d.mkdir()
        status = {
            "status": "COMPLETED",
            "updated_at": time.time(),
            "start_time": time.time() - 1,
            "end_time": time.time(),
        }
        if not omit_key:
            status["owner"] = owner
            status["owner_anonymous"] = owner_anonymous if owner is not None else None
        (d / "status.json").write_text(json.dumps(status))
        return sid

    return add


# ── the fixture's own control ────────────────────────────────────────────────


def test_the_fixture_really_writes_runs_that_the_listing_can_see(runs):
    """THE CONTROL FOR THE WHOLE FILE. Every "sees nothing" assertion below is
    only meaningful if a caller who SHOULD see runs does. If this fails, the
    empty results elsewhere mean the fixture is broken, not that the filter
    works."""
    runs("r1", ME)
    runs("r2", None)
    runs("r3", OTHER)
    # The admin is the one identity that can see all three classes at once.
    assert sorted(training.list_sessions(ADMIN)) == ["r1", "r2"], (
        "the admin should see its own run and the ownerless one, and NOT the "
        "run owned by somebody else"
    )
    assert sorted(training.list_sessions(ctx(OTHER, anonymous=False))) == ["r3"]


# ── rule 1: a caller sees exactly its own runs ───────────────────────────────


def test_a_caller_sees_its_own_runs_and_not_another_callers(runs):
    runs("mine-1", ME)
    runs("mine-2", ME)
    runs("theirs", OTHER)
    got = training.list_sessions(ctx(ME, anonymous=False))
    assert sorted(got) == ["mine-1", "mine-2"], got


def test_an_anonymous_identity_is_a_REAL_identity_here(runs):
    """Hypha mints a fresh anonymous id per connection, so an anonymous caller
    owns what it started. Filtering on ``is_anonymous`` instead of on the id
    would give this caller nothing and silently break its own history."""
    runs("anon-a-run", ANON_A, owner_anonymous=True)
    runs("anon-b-run", ANON_B, owner_anonymous=True)
    assert sorted(training.list_sessions(ctx(ANON_A))) == ["anon-a-run"]
    assert sorted(training.list_sessions(ctx(ANON_B))) == ["anon-b-run"]


def test_two_fresh_anonymous_callers_cannot_see_each_other(runs):
    """The pair the row measures as (b) and (c)."""
    runs("b-run", ANON_B, owner_anonymous=True)
    assert training.list_sessions(ctx(ANON_A)) == {}
    # paired demonstration: the run exists and IS visible to its owner
    assert sorted(training.list_sessions(ctx(ANON_B))) == ["b-run"]


# ── rule 2: an unreadable caller gets nothing, never everything ──────────────


@pytest.mark.parametrize(
    "bad",
    [None, {}, {"user": None}, {"user": {}}, {"user": {"id": ""}}, "not-a-dict", 7],
    ids=["none", "empty", "user-none", "user-empty", "blank-id", "string", "int"],
)
def test_a_context_that_cannot_be_read_lists_NOTHING(runs, bad):
    """Fails closed. The dangerous direction is an unreadable context inheriting
    the old open behaviour."""
    runs("owned", ME)
    runs("ownerless", None)
    assert training.list_sessions(bad) == {}, bad


def test_a_recorded_owner_is_never_matched_by_an_unidentifiable_caller(runs):
    """`caller is None` must not equal `owner is None` anywhere in the filter."""
    runs("ownerless", None)
    assert training.list_sessions(None) == {}
    # paired demonstration: that same run is visible to an authorised caller
    assert sorted(training.list_sessions(ADMIN)) == ["ownerless"]


@pytest.mark.parametrize("omit_key", [False, True], ids=["null-owner", "no-owner-key"])
def test_BOTH_SHAPES_of_ownerless_are_treated_alike(runs, omit_key):
    """A null `owner` and an absent `owner` are the same case, and the two arrive
    by different routes: a run by an unidentified caller writes the key as null,
    while the 261 runs written before the field existed have no key.

    This is the case that distinguishes the `caller is not None` guard from
    nothing at all. Dropping the `isinstance(owner, str)` branch condition --
    say to `if "owner" in status` -- sends a null owner into the identity
    comparison, where `None == None` is TRUE, and every run by an unidentified
    caller becomes visible to every unidentifiable caller. A fixture that only
    ever omitted the key could not see that.
    """
    runs("old", None, omit_key=omit_key)
    assert training.list_sessions(None) == {}, "an unreadable caller saw a run"
    assert training.list_sessions(ctx(ANON_A)) == {}
    # paired demonstration, so the two empties above are not a broken fixture
    assert sorted(training.list_sessions(ADMIN)) == ["old"]


def test_an_owner_recorded_as_a_non_string_is_ownerless_not_a_wildcard(runs):
    """Defensive, and cheap: a corrupted record must fall into the closed case
    rather than match something. `owner: 0` equalling a falsy caller would be the
    bad outcome."""
    for sid, bad in (("n1", 0), ("n2", False), ("n3", []), ("n4", "")):
        runs(sid, bad)
    assert training.list_sessions(None) == {}
    assert training.list_sessions(ctx(ANON_A)) == {}
    assert sorted(training.list_sessions(ADMIN)) == ["n1", "n2", "n3", "n4"]


# ── rule 3: ownerless runs, and the two narrowings ───────────────────────────


def test_ownerless_runs_are_hidden_from_a_caller_without_authority(runs):
    runs("old-1", None)
    runs("old-2", None)
    assert training.list_sessions(ctx(ANON_A)) == {}
    assert training.list_sessions(ctx(OTHER, anonymous=False)) == {}
    # paired demonstration
    assert sorted(training.list_sessions(ADMIN)) == ["old-1", "old-2"]


@pytest.mark.parametrize("grant", ["a", "rw"])
def test_rw_and_admin_on_the_hosting_workspace_BOTH_grant_the_ownerless_runs(
    runs, grant
):
    runs("old", None)
    who = ctx(OTHER, anonymous=False, grants={"bioimage-io": grant})
    assert sorted(training.list_sessions(who)) == ["old"], grant


def test_READ_permission_on_the_hosting_workspace_is_NOT_authority(runs):
    """The first narrowing. ``usage_ledger.classify_caller`` counts "r" as an
    operator grant, which is right for its question and wrong for this one;
    reusing it here would hand every reader the unowned history."""
    runs("old", None)
    reader = ctx(OTHER, anonymous=False, grants={"bioimage-io": "r"})
    assert training.list_sessions(reader) == {}
    assert training.caller_can_see_ownerless(reader) is False
    # paired demonstration: the SAME caller with rw does see it
    assert sorted(
        training.list_sessions(
            ctx(OTHER, anonymous=False, grants={"bioimage-io": "rw"})
        )
    ) == ["old"]


def test_ADMIN_ON_THE_CALLERS_OWN_WORKSPACE_is_NOT_authority(runs):
    """The second narrowing, and the one that would have leaked everything.

    Measured on the live platform: a token-less caller is minted
    ``anonymouz-...`` holding ``"a"`` on ``ws-user-anonymouz-...`` — its own
    workspace. So "holds an operator grant on some workspace" is true for every
    caller in existence, anonymous ones included.
    """
    runs("old", None)
    anon = ctx(ANON_A, grants={"ws-user-" + ANON_A: "a"})
    assert training.caller_can_see_ownerless(anon) is False
    assert training.list_sessions(anon) == {}
    # paired demonstration: the runs are there for someone with real authority
    assert sorted(training.list_sessions(ADMIN)) == ["old"]


@pytest.mark.parametrize(
    "grant,expected", [("a", True), ("rw", True), ("r", False), ("x", False)]
)
def test_a_wildcard_grant_counts_only_when_it_carries_rw_or_admin(grant, expected):
    """A genuine global operator does hold authority over this workspace, so a
    wildcard is honoured — but it is narrowed by the same grant set, so a
    wildcard carrying only read does not qualify either."""
    who = ctx(OTHER, anonymous=False, grants={"*": grant})
    assert training.caller_can_see_ownerless(who) is expected, grant


def test_authority_is_decided_only_by_the_named_workspace(runs):
    """A grant on an unrelated workspace is not authority here."""
    runs("old", None)
    elsewhere = ctx(OTHER, anonymous=False, grants={"some-other-ws": "a"})
    assert training.caller_can_see_ownerless(elsewhere) is False
    assert training.list_sessions(elsewhere) == {}


def test_the_workspace_being_checked_is_not_hardcoded_past_its_constant():
    """The constant is the single point of truth, so a deployment elsewhere
    changes one name. Asserted by checking a DIFFERENT workspace via the
    keyword, which must then be the one that counts."""
    who = ctx(OTHER, anonymous=False, grants={"other-home": "a"})
    assert training.caller_can_see_ownerless(who) is False
    assert training.caller_can_see_ownerless(who, workspace="other-home") is True


# ── the owner decides visibility and is never published ──────────────────────


def test_the_owner_fields_are_absent_from_every_returned_record(runs):
    runs("mine", ME)
    runs("old", None)
    got = training.list_sessions(ADMIN)
    assert sorted(got) == ["mine", "old"]
    for sid, record in got.items():
        for field in training.PRIVATE_STATUS_FIELDS:
            assert field not in record, f"{field} published on {sid}: {record}"


def test_redaction_is_one_definition_shared_by_the_listing_and_the_single_read(runs):
    """Both paths must go through ``redact_private``, so a field added to the
    tuple is stripped on both. Measured by adding one at runtime."""
    runs("mine", ME)
    original = training.PRIVATE_STATUS_FIELDS
    # Plant a field that is private only for the duration of this test.
    d = training.session_dir("mine")
    status = json.loads((d / "status.json").read_text())
    status["a_new_private_field"] = "secret"
    (d / "status.json").write_text(json.dumps(status))
    try:
        training.PRIVATE_STATUS_FIELDS = original + ("a_new_private_field",)
        listed = training.list_sessions(ctx(ME, anonymous=False))["mine"]
        single = training.get_status("mine")
        assert "a_new_private_field" not in listed, listed
        assert "a_new_private_field" not in single, single
    finally:
        training.PRIVATE_STATUS_FIELDS = original
    # paired demonstration: with the tuple restored the field IS returned, so
    # the assertions above measured the tuple and not some unrelated filter.
    assert (
        training.get_status("mine")["a_new_private_field"] == "secret"
    ), "the field was dropped for a reason other than the private-field tuple"


# ── there is no unfiltered listing to reach by accident ──────────────────────


def test_list_sessions_REQUIRES_a_context(runs):
    """The old signature returned every run to whoever asked. Keeping it
    available beside a filtered sibling would put the open behaviour one
    forgotten argument away, so it has no default."""
    runs("mine", ME)
    with pytest.raises(TypeError):
        training.list_sessions()


def test_no_module_level_function_returns_an_unfiltered_listing(runs):
    """A weaker but broader guard: nothing else in the module hands back every
    run. If a convenience wrapper is added later, it has to be considered."""
    runs("a", ME)
    runs("b", OTHER)
    runs("c", None)
    suspects = [
        name
        for name in dir(training)
        if name.startswith("list_") and callable(getattr(training, name))
    ]
    assert suspects == ["list_sessions"], (
        f"new list_* function(s) in training.py: {suspects}. Each one is a "
        "potential unfiltered listing and needs its own visibility test."
    )


# ── the decorator, same guard as the ownership half ──────────────────────────


def _entry_class():
    """The CLASS behind the Serve deployment.

    ``entry.EntryApp`` is a ``Deployment``, not a type — `object.__new__` on it
    raises `TypeError: X is not a type object`. The class is under
    ``func_or_class``, and reaching for it is not optional: a test that reads
    attributes off the Deployment is reading the wrapper.
    """
    import entry

    return getattr(entry.EntryApp, "func_or_class", entry.EntryApp)


def _entry_method(name):
    return getattr(_entry_class(), name)


def test_list_training_sessions_is_declared_with_the_injected_context():
    """Without ``context=True`` the proxy forwards nothing, the filter reads no
    identity, and every caller gets an EMPTY listing — safe, but silent and
    indistinguishable from a worker with no runs."""
    fn = _entry_method("list_training_sessions")
    schema = getattr(fn, "__schema__", None)
    assert schema, "list_training_sessions carries no __schema__"
    props = (schema.get("parameters") or {}).get("properties") or {}
    assert "context" not in props, (
        "`context` is in the published schema, so list_training_sessions was not "
        "declared @bioengine.method(context=True) and will never receive a "
        "caller — the listing comes back empty for everyone."
    )
    flags = [
        getattr(fn, n)
        for n in dir(fn)
        if "context" in n.lower() and isinstance(getattr(fn, n, None), bool)
    ]
    assert flags and any(flags), (
        f"list_training_sessions is not flagged as wanting the injected context; "
        f"bool attrs mentioning context: {flags}"
    )


@pytest.mark.asyncio
async def test_the_entry_method_PASSES_the_context_through(runs, monkeypatch):
    """Declaring the decorator is not the same as using what it delivers. This
    fails if the method drops the context on the way to the filter."""
    import entry

    seen = {}

    def spy(context):
        seen["context"] = context
        return {"sentinel": {}}

    monkeypatch.setattr(training, "list_sessions", spy)
    cls = _entry_class()
    fn = getattr(cls, "list_training_sessions")
    inst = cls.__new__(cls)
    out = await fn(inst, context=ADMIN)
    assert out == {"sentinel": {}}
    assert seen["context"] is ADMIN, seen
    # The spy takes `context` POSITIONALLY-or-by-keyword and would accept a
    # dropped context only if the method passed None explicitly; assert the
    # object identity so a method that rebuilt a context from somewhere else
    # fails here too.
    assert seen["context"]["user"]["id"] == ME
