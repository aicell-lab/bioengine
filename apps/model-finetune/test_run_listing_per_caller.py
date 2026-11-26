"""The run listing returns the caller's own runs, and nobody else's.

The listing used to return every run on the worker to whoever asked. These tests
pin the three visibility rules and, more importantly, the three NARROWINGS that
make the ownerless rule correct — each measured against the live platform before
being written down, and each one something the obvious implementation gets wrong:

* read permission on the NAMED workspace is NOT authority over other people's
  runs. The app's existing classifier, `usage_ledger.classify_caller`, does count
  it as operator — measured: it buckets a caller holding only "r" here as
  `internal`, where this check refuses it.
* the workspace is NAMED, and the caller's OWN workspace does not count. Every
  token-less caller is minted a fresh identity holding ADMIN on its own personal
  workspace, so an implementation asking "does this caller hold an operator grant
  anywhere" is true for every caller alive and would publish the whole unowned
  history to anonymous callers.
* a WILDCARD grant does not count either, not even carrying admin. The rule names
  one workspace, and widening a privacy filter past its rule is not a call to
  make in passing.

**The second bullet is a hazard of THIS function, not of reusing that one.**
`classify_caller` returns `anonymous` before it ever reads the grants — measured,
including for an anonymous caller holding `*: a` — so reusing it would not have
leaked to anonymous callers. This check deliberately has no such short-circuit,
because here an anonymous caller is a real identity that owns its own runs. An
earlier version of this note pinned the leak on that function, which was wrong.

**Every test that asserts somebody sees NOTHING is paired with a demonstration
that the runs are really there.** Without that pairing, a fixture that silently
wrote no sessions would make the whole file pass while measuring nothing — the
failure this suite is least able to notice on its own.
"""

import json
import re
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


# A DISTINCT identity from every owner used below. Using ME here made
# "the admin can" indistinguishable from "the owner can", which is exactly how
# a bug where the admin could not touch an OWNED run passed the suite.
ADMIN_ID = "github|777000"
ADMIN = ctx(ADMIN_ID, anonymous=False, grants={"bioimage-io": "a"})


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

    # These tests use readable labels ("mine", "theirs", "old") as session ids.
    # The id ALLOW-LIST added for path containment refuses anything that is not
    # the shape the server generates, and the listing skips what it refuses --
    # so without this every planted run would be skipped and each assertion
    # below would pass while measuring nothing.
    #
    # Relaxed deliberately and only here: this file is about WHO MAY SEE WHICH
    # RUN. The shape rule, and the listing's tolerance of a name that breaks it,
    # are covered by test_session_id_containment.py against the real pattern.
    monkeypatch.setattr(training, "SESSION_ID_PATTERN", re.compile(r"\A[\w.-]+\Z"))

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
    runs("r1", ADMIN_ID)
    runs("r2", None)
    runs("r3", OTHER)
    # The admin sees ITS OWN run plus the ownerless one, and NOT the run owned by
    # somebody else. That last part is the listing rule and it does NOT change
    # for an administrator: admin rights grant the ownerless rows, not everyone's
    # history. Stopping somebody else's run is a different rule -- see clause 5
    # below -- and the two are deliberately not the same.
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
def test_rw_and_admin_on_the_NAMED_workspace_BOTH_grant_the_ownerless_runs(runs, grant):
    runs("old", None)
    who = ctx(OTHER, anonymous=False, grants={"bioimage-io": grant})
    assert sorted(training.list_sessions(who)) == ["old"], grant


def test_READ_permission_on_the_NAMED_workspace_is_NOT_authority(runs):
    """The first narrowing. ``usage_ledger.classify_caller`` counts "r" as an
    operator grant, which is right for its question and wrong for this one.
    Measured: that function buckets a caller holding only "r" on this workspace
    as `internal`, where this one refuses it."""
    runs("old", None)
    reader = ctx(OTHER, anonymous=False, grants={"bioimage-io": "r"})
    assert training.list_sessions(reader) == {}
    assert training.caller_is_workspace_admin(reader) is False
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
    assert training.caller_is_workspace_admin(anon) is False
    assert training.list_sessions(anon) == {}
    # paired demonstration: the runs are there for someone with real authority
    assert sorted(training.list_sessions(ADMIN)) == ["old"]


@pytest.mark.parametrize("grant", ["a", "rw", "r", "x"])
def test_a_WILDCARD_grant_NEVER_qualifies(runs, grant):
    """Not even carrying admin.

    A global operator arguably does hold authority here, so accepting `"*"`
    would be defensible — but the rule this implements names ONE workspace, and
    widening a privacy filter past its rule is not a call to make in passing.
    Refusing a wildcard costs a genuine global operator nothing but these rows;
    accepting it would show 261 other people's runs to an identity nobody
    authorised.
    """
    runs("old", None)
    who = ctx(OTHER, anonymous=False, grants={"*": grant})
    assert training.caller_is_workspace_admin(who) is False, grant
    assert training.list_sessions(who) == {}
    # paired demonstration: the run IS there for the named workspace's admin
    assert sorted(training.list_sessions(ADMIN)) == ["old"]


def test_a_wildcard_does_not_help_even_ALONGSIDE_a_read_grant(runs):
    """The two refused shapes combined are still refused — neither is a partial
    credit toward the other."""
    runs("old", None)
    who = ctx(OTHER, anonymous=False, grants={"*": "a", "bioimage-io": "r"})
    assert training.caller_is_workspace_admin(who) is False
    assert training.list_sessions(who) == {}


def test_authority_is_decided_only_by_the_named_workspace(runs):
    """A grant on an unrelated workspace is not authority here."""
    runs("old", None)
    elsewhere = ctx(OTHER, anonymous=False, grants={"some-other-ws": "a"})
    assert training.caller_is_workspace_admin(elsewhere) is False
    assert training.list_sessions(elsewhere) == {}


def test_the_workspace_being_checked_is_not_hardcoded_past_its_constant():
    """The constant is the single point of truth, so a deployment elsewhere
    changes one name. Asserted by checking a DIFFERENT workspace via the
    keyword, which must then be the one that counts."""
    who = ctx(OTHER, anonymous=False, grants={"other-home": "a"})
    assert training.caller_is_workspace_admin(who) is False
    assert training.caller_is_workspace_admin(who, workspace="other-home") is True


# ── the owner decides visibility and is never published ──────────────────────


def test_the_owner_fields_are_absent_from_every_returned_record(runs):
    runs("mine", ADMIN_ID)
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

    def spy(context, *, writable_datasets=frozenset(), records=None, skipped=None):
        seen["context"] = context
        seen["writable_datasets"] = writable_datasets
        seen["records"] = records
        return {"sentinel": {}}

    class _AM:
        async def read(self, artifact_id):
            return {"config": {"permissions": {}}}

    class _Hypha:
        async def get_service(self, name):
            return _AM()

    monkeypatch.setattr(training, "list_sessions", spy)
    monkeypatch.setattr(training, "scan_runs_for_listing", lambda: ({}, [], []))
    cls = _entry_class()
    fn = getattr(cls, "list_training_sessions")
    inst = cls.__new__(cls)
    inst._hypha = _Hypha()
    out = await fn(inst, context=ADMIN)
    # This instance has no connection, so the shared run record cannot be read.
    # The listing must still answer -- from local disk alone -- rather than
    # raise: a short history beats no history. That it is `{}` and not None is
    # the filter's contract, which has no None branch.
    assert seen["records"] == {}, seen["records"]
    assert out == {"sentinel": {}}
    assert seen["context"] is ADMIN, seen
    # The spy takes `context` POSITIONALLY-or-by-keyword and would accept a
    # dropped context only if the method passed None explicitly; assert the
    # object identity so a method that rebuilt a context from somewhere else
    # fails here too.
    assert seen["context"]["user"]["id"] == ADMIN_ID


# ── clause 5: only the owner, or a bioimage-io admin, may CHANGE a run ───────
#
# The listing stopped showing other people's runs. Changing them was still open:
# `stop_training` took a session_id with no caller check at all, so anyone who
# knew or could read an id could stop anyone's training. Export is the same
# shape and worse in kind -- it builds a publishable package out of somebody
# else's checkpoint.
#
# An ownerless run is ADMIN-ONLY, the same rule the listing applies to the 261
# runs recorded before the owner field existed, and for the same reason: a run
# that cannot be attributed to anybody must not be handed to the first caller
# who asks.


def _entry():
    import entry

    cls = getattr(entry.EntryApp, "func_or_class", entry.EntryApp)
    inst = cls.__new__(cls)
    inst._training_tasks = {}
    return cls, inst


def _owned(runs, sid, owner, **kw):
    runs(sid, owner, **kw)
    return sid


@pytest.mark.asyncio
async def test_a_DIFFERENT_caller_cannot_stop_my_run(runs):
    """The reviewer's first named RED-able case."""
    sid = _owned(runs, "2026-11-25-150730-abcdef12", ME)
    cls, inst = _entry()
    with pytest.raises(training.NotTheOwner):
        await cls.stop_training(
            inst, session_id=sid, context=ctx(OTHER, anonymous=False)
        )
    assert (
        training.read_status(sid).get("status") == "COMPLETED"
    ), "the refused call still changed the run"


@pytest.mark.asyncio
async def test_an_ANONYMOUS_caller_cannot_stop_an_owned_run(runs):
    """The reviewer's second named RED-able case."""
    sid = _owned(runs, "2026-11-25-150730-abcdef12", ME)
    cls, inst = _entry()
    for who in (ctx(ANON_A), ctx(ANON_B), None, {}):
        with pytest.raises(training.NotTheOwner):
            await cls.stop_training(inst, session_id=sid, context=who)
    assert training.read_status(sid).get("status") == "COMPLETED"


@pytest.mark.asyncio
async def test_THE_OWNER_CAN_stop_their_own_run(runs):
    """THE CONTROL. Every refusal above would read the same if the gate refused
    everybody, which is an outage rather than a fix."""
    sid = _owned(runs, "2026-11-25-150730-abcdef12", ME)
    cls, inst = _entry()
    out = await cls.stop_training(
        inst, session_id=sid, context=ctx(ME, anonymous=False)
    )
    assert out["status"] == "STOPPED", out
    assert training.read_status(sid)["terminated_by"] == "user_stop"


@pytest.mark.asyncio
async def test_an_ANONYMOUS_caller_CAN_stop_the_run_IT_started(runs):
    """An anonymous identity is a real identity here, so it owns what it started
    and must be able to stop it. A gate keyed on `is_anonymous` instead of on the
    id would refuse this and look correct."""
    sid = _owned(runs, "2026-11-25-150730-abcdef12", ANON_A, owner_anonymous=True)
    cls, inst = _entry()
    out = await cls.stop_training(inst, session_id=sid, context=ctx(ANON_A))
    assert out["status"] == "STOPPED", out


@pytest.mark.asyncio
@pytest.mark.parametrize("omit_key", [False, True], ids=["null-owner", "no-owner-key"])
async def test_an_OWNERLESS_run_is_ADMIN_ONLY(runs, omit_key):
    sid = "2026-11-25-150730-abcdef12"
    runs(sid, None, omit_key=omit_key)
    cls, inst = _entry()
    for who in (ctx(ANON_A), ctx(OTHER, anonymous=False), None):
        with pytest.raises(training.NotTheOwner):
            await cls.stop_training(inst, session_id=sid, context=who)
    assert training.read_status(sid).get("status") == "COMPLETED"
    # paired demonstration: the admin CAN
    out = await cls.stop_training(inst, session_id=sid, context=ADMIN)
    assert out["status"] == "STOPPED", out


@pytest.mark.asyncio
async def test_READ_permission_does_not_let_you_stop_an_ownerless_run(runs):
    """The same narrowing as the listing: "r" is not authority."""
    sid = "2026-11-25-150730-abcdef12"
    runs(sid, None)
    cls, inst = _entry()
    reader = ctx(OTHER, anonymous=False, grants={"bioimage-io": "r"})
    with pytest.raises(training.NotTheOwner):
        await cls.stop_training(inst, session_id=sid, context=reader)
    # paired demonstration: rw is enough
    out = await cls.stop_training(
        inst,
        session_id=sid,
        context=ctx(OTHER, anonymous=False, grants={"bioimage-io": "rw"}),
    )
    assert out["status"] == "STOPPED"


@pytest.mark.asyncio
async def test_EXPORT_is_gated_too_and_refuses_BEFORE_touching_the_session(runs):
    """Export builds a publishable package from a run's checkpoint, so it is a
    mutating call in the sense that matters. The gate runs before the checkpoint
    lookup, so the refusal does not reveal whether the session exists or trained.
    """
    sid = _owned(runs, "2026-11-25-150730-abcdef12", ME)
    cls, inst = _entry()
    with pytest.raises(training.NotTheOwner):
        await cls.export_model(
            inst, session_id=sid, name="x", context=ctx(OTHER, anonymous=False)
        )


@pytest.mark.asyncio
async def test_the_gate_CREATES_NOTHING_for_a_session_that_does_not_exist(
    runs, root=None
):
    """Asking the question must not manufacture the run. A non-admin is refused
    before any write, so the directory must still be absent afterwards."""
    cls, inst = _entry()
    missing = "2026-01-01-000000-deadbeef"
    with pytest.raises(training.NotTheOwner):
        await cls.stop_training(inst, session_id=missing, context=ctx(ANON_A))
    assert not training.session_dir(missing).exists(), "the refused call created it"


@pytest.mark.parametrize("method", ["stop_training", "export_model"])
def test_every_gated_method_is_declared_with_the_injected_context(method):
    """Without the decorator the proxy forwards nothing, the gate sees no
    identity and refuses EVERYBODY — safe, silent, and indistinguishable from a
    broken worker."""
    fn = _entry_method(method)
    schema = getattr(fn, "__schema__", None)
    assert schema, f"{method} carries no __schema__"
    props = (schema.get("parameters") or {}).get("properties") or {}
    assert "context" not in props, (
        f"`context` is published on {method}, so it was not declared "
        "@bioengine.method(context=True) and will never receive a caller"
    )
    flags = [
        getattr(fn, n)
        for n in dir(fn)
        if "context" in n.lower() and isinstance(getattr(fn, n, None), bool)
    ]
    assert flags and any(flags), f"{method} is not flagged as wanting the context"


def test_caller_may_mutate_fails_closed_on_an_unreadable_context(runs):
    sid = _owned(runs, "2026-11-25-150730-abcdef12", ME)
    for bad in (None, {}, {"user": None}, {"user": {}}, "nope", 7):
        assert training.caller_may_mutate(sid, bad) is False, bad
    # paired control
    assert training.caller_may_mutate(sid, ctx(ME, anonymous=False)) is True


# ── the admin leg, with an id DISTINCT from the owner ───────────────────────
#
# THE BUG THESE EXIST FOR. The rule is "the owner OR a bioimage-io admin", and
# an earlier version reached the admin test only for OWNERLESS runs — so an
# administrator could not touch an owned run, while the function's own docstring
# said they could. The suite missed it because its admin fixture used the SAME id
# as the owner, so "the admin can" was passing as "the owner can".
#
# Every test below therefore uses ADMIN_ID, which owns nothing.


@pytest.mark.asyncio
async def test_an_ADMIN_WITH_A_DISTINCT_ID_can_stop_someone_elses_run(runs):
    sid = _owned(runs, "2026-11-25-150730-abcdef12", ANON_A, owner_anonymous=True)
    assert training.read_status(sid)["owner"] != ADMIN_ID, "fixture: ids must differ"
    cls, inst = _entry()
    out = await cls.stop_training(inst, session_id=sid, context=ADMIN)
    assert out["status"] == "STOPPED", out


@pytest.mark.asyncio
async def test_an_ADMIN_WITH_A_DISTINCT_ID_can_export_someone_elses_run(runs):
    sid = _owned(runs, "2026-11-25-150730-abcdef12", ANON_A, owner_anonymous=True)
    cls, inst = _entry()
    # The gate must PASS, so the call proceeds past it and fails later on the
    # missing checkpoint instead. NotTheOwner here would mean the gate refused.
    with pytest.raises(Exception) as exc:
        await cls.export_model(inst, session_id=sid, name="x", context=ADMIN)
    assert not isinstance(
        exc.value, training.NotTheOwner
    ), "the admin was refused by the owner gate on an owned run"


@pytest.mark.parametrize("grant", ["a", "rw"])
def test_caller_may_mutate_grants_an_admin_on_an_OWNED_run(runs, grant):
    """The unit-level statement of the same thing, across both grants."""
    sid = _owned(runs, "2026-11-25-150730-abcdef12", ANON_A, owner_anonymous=True)
    admin = ctx("github|999", anonymous=False, grants={"bioimage-io": grant})
    assert training.caller_may_mutate(sid, admin) is True, grant
    # paired control: the same identity WITHOUT the grant is refused
    assert training.caller_may_mutate(sid, ctx("github|999", anonymous=False)) is False


def test_READ_permission_does_not_grant_mutation_of_an_OWNED_run(runs):
    """The "r" narrowing applies to the admin override too, not just to the
    ownerless rule."""
    sid = _owned(runs, "2026-11-25-150730-abcdef12", ANON_A, owner_anonymous=True)
    reader = ctx("github|999", anonymous=False, grants={"bioimage-io": "r"})
    assert training.caller_may_mutate(sid, reader) is False


def test_a_NON_admin_third_party_still_cannot_mutate_an_owned_run(runs):
    """THE CONTROL for the admin tests: the override must come from the grant,
    not from being a third party."""
    sid = _owned(runs, "2026-11-25-150730-abcdef12", ANON_A, owner_anonymous=True)
    assert training.caller_may_mutate(sid, ctx(OTHER, anonymous=False)) is False
    assert training.caller_may_mutate(sid, ctx(ANON_B)) is False


# ── contribute is a mutating call keyed by session_id, so it is gated ───────
#
# It uploads a finished run's weights into the SHARED community model-soup pool
# and appends to the pool index. Publishing somebody else's checkpoint is the
# same harm as exporting it and worse in reach: the index is append-only and
# shared, so a contribution cannot simply be withdrawn. Gating export_model did
# NOT cover this — contribute does not go through export.


async def _contribute(cls, inst, sid, who):
    """Call contribute with every REQUIRED argument supplied.

    `pool_artifact_id` and `pool_token` are required, and the framework's own
    argument check raises TypeError before the method body runs if they are
    missing -- so omitting them never reaches the gate at all. The values are
    placeholders: a refused caller never gets near the pool, and an allowed one
    fails later on them, which is what the control above asserts.
    """
    return await cls.contribute(
        inst,
        session_id=sid,
        pool_artifact_id="bioimage-io/does-not-exist",
        pool_token="placeholder-not-a-real-token",
        context=who,
    )


@pytest.mark.asyncio
async def test_CONTRIBUTE_refuses_a_run_the_caller_does_not_own(runs):
    sid = _owned(runs, "2026-11-25-150730-abcdef12", ME)
    cls, inst = _entry()
    with pytest.raises(training.NotTheOwner):
        await _contribute(cls, inst, sid, ctx(OTHER, anonymous=False))
    with pytest.raises(training.NotTheOwner):
        await _contribute(cls, inst, sid, ctx(ANON_A))


@pytest.mark.asyncio
async def test_CONTRIBUTE_refuses_BEFORE_revealing_the_runs_state(runs):
    """Its "is <status>, not COMPLETED" error is an existence-and-state oracle,
    so the gate has to run first. A refused caller must not learn the status."""
    sid = _owned(runs, "2026-11-25-150730-abcdef12", ME)
    cls, inst = _entry()
    with pytest.raises(training.NotTheOwner) as exc:
        await _contribute(cls, inst, sid, ctx(OTHER, anonymous=False))
    msg = str(exc.value)
    for leak in ("COMPLETED", "not COMPLETED", sid):
        assert leak not in msg, f"the refusal leaked {leak!r}: {msg}"


@pytest.mark.asyncio
async def test_CONTRIBUTE_lets_the_OWNER_past_the_gate(runs):
    """THE CONTROL. Both refusals above would read the same if the gate refused
    everybody. The owner gets past it and fails later, on the real work."""
    sid = _owned(runs, "2026-11-25-150730-abcdef12", ME)
    cls, inst = _entry()
    seen = []
    real = training.require_caller_may_mutate

    def spy(session_id, context):
        seen.append(session_id)
        return real(session_id, context)

    training.require_caller_may_mutate = spy
    try:
        with pytest.raises(Exception) as exc:
            await _contribute(cls, inst, sid, ctx(ME, anonymous=False))
    finally:
        training.require_caller_may_mutate = real
    # The gate was CONSULTED. Without this the test passes whenever the call
    # blows up BEFORE reaching the gate -- which is exactly what the first
    # version of it did, on a TypeError about a missing required argument. A
    # control that passes for the wrong reason is worse than no control.
    assert seen == [sid], f"the gate was never consulted: {seen}"
    assert not isinstance(
        exc.value, training.NotTheOwner
    ), "the owner was refused by its own run's gate"


@pytest.mark.parametrize("method", ["stop_training", "export_model", "contribute"])
def test_EVERY_mutating_method_keyed_by_session_id_is_gated(method):
    """A structural guard over the three, so a fourth is a deliberate decision.
    The decorator alone is not enough — without it the gate sees no identity and
    refuses everybody, which is safe but silent."""
    fn = _entry_method(method)
    schema = getattr(fn, "__schema__", None)
    assert schema, f"{method} carries no __schema__"
    props = (schema.get("parameters") or {}).get("properties") or {}
    assert "context" not in props, f"`context` is published on {method}"
    flags = [
        getattr(fn, n)
        for n in dir(fn)
        if "context" in n.lower() and isinstance(getattr(fn, n, None), bool)
    ]
    assert flags and any(flags), f"{method} is not flagged as wanting the context"


# Every published method whose signature takes a run id, and which rule applies.
# A method absent from BOTH sets fails the structural test below, so adding one
# forces a decision rather than inheriting whichever behaviour is default.
GATED_BY_RUN_ID = {
    "stop_training",  # cancels the run
    "export_model",  # builds a publishable package from its checkpoint
    "contribute",  # publishes its checkpoint into the shared pool
    "start_training",  # only when resume_session_id is supplied; see the tests above
}
# WHICH RULE APPLIES IS DECIDED BY WHETHER THE CALL HANDS OUT OR CREATES
# WEIGHTS -- not by whether it touches them.
#
# OPEN, gated by possession of the run id alone: status, progress, logs, and
# inference. `infer` and `compute_embedding` return OUTPUTS computed with the
# weights; the caller never receives the checkpoint and no new model derives
# from it. Shared notebook sessions exist for several people to try one run's
# model, and gating inference would break that.
#
# OWNER-OR-ADMIN: export (hands out the weights), contribute (publishes them
# into the shared pool), resume (creates a new model derived from them), and
# stop (acts on someone else's run).
#
# That line is the ruling's, not this lane's. An earlier version of this file
# classified inference as open on the reasoning that it "locates a model rather
# than acting on a run" -- which reached the same answer by a route that would
# also have let export through, and which was mine to begin with.
READ_ONLY_BY_RUN_ID = {"get_training_status", "infer", "compute_embedding"}


def _published_methods_taking_a_run_id(path):
    """Names of `@bioengine.method`s whose signature takes a run id.

    PARSED, not read through a line window. The previous version of this helper
    looked at the first 30 lines after `async def` -- and `start_training`'s
    signature runs 114 lines, with `resume_session_id` far past the window. So it
    was missed entirely: absent from both sets and the test still green. A guard
    that claims to cover every method must not have a shorter reach than the
    methods it inspects.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(path).read_text())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        decorated = any(
            "bioengine" in ast.dump(d) and "method" in ast.dump(d)
            for d in node.decorator_list
        )
        if not decorated:
            continue
        a = node.args
        names = [x.arg for x in (a.posonlyargs + a.args + a.kwonlyargs)]
        if a.vararg:
            names.append(a.vararg.arg)
        if a.kwarg:
            names.append(a.kwarg.arg)
        # Matches the names actually in use for a run id. KNOWN LIMIT: a run id
        # passed under some other name would escape this, so the match list has
        # to grow with the code. Stated rather than papered over -- a structural
        # guard whose reach is narrower than its docstring claims is the defect
        # this function already had once, in another form.
        if any(("session_id" in n) or ("run_id" in n) for n in names):
            out.append(node.name)
    return sorted(out)


def test_no_OTHER_entry_method_takes_a_run_id_without_a_decision():
    """Every published method taking a run id is gated or a listed exception."""
    import entry

    taking = _published_methods_taking_a_run_id(entry.__file__)
    assert taking, "the parser found no methods at all -- it is reading the wrong file"
    assert "start_training" in taking, (
        "start_training takes resume_session_id and must be found; if it is not, "
        "the parser has regressed to a line-window scan"
    )
    unaccounted = sorted(set(taking) - GATED_BY_RUN_ID - READ_ONLY_BY_RUN_ID)
    assert not unaccounted, (
        f"method(s) taking a run id that are in neither set: "
        f"{unaccounted}. Decide which, then add it deliberately. Found: {taking}"
    )


def test_each_set_is_exactly_what_the_ruling_says():
    """Pins both sets, so moving a method between them is a decision rather than
    a refactor. The ruling's line is whether the call hands out or creates
    weights; a new surface has to be placed against that, not appended here."""
    assert GATED_BY_RUN_ID == {
        "stop_training",
        "export_model",
        "contribute",
        "start_training",
    }, "the owner-or-admin set changed; that needs a ruling, not an edit here"
    assert READ_ONLY_BY_RUN_ID == {
        "get_training_status",
        "infer",
        "compute_embedding",
    }, "the open set changed; that needs a ruling, not an edit here"
    assert not (GATED_BY_RUN_ID & READ_ONLY_BY_RUN_ID), "a method cannot be both"


def test_that_structural_guard_CAN_fail(tmp_path):
    """THE CONTROL. It reads a file, so it can silently measure the wrong thing.
    Point it at a module with a long-signature method that is in neither set and
    it must be found -- which is exactly the case the old 30-line window missed.
    """
    fake = tmp_path / "fake_entry.py"
    padding = "\n".join(f"        arg{i}: int = 0," for i in range(60))
    fake.write_text(
        "class EntryApp:\n"
        "    @bioengine.method\n"
        "    async def a_new_method(\n"
        "        self,\n" + padding + "\n"
        "        session_id: str = '',\n"
        "    ) -> dict:\n"
        "        pass\n"
    )
    found = _published_methods_taking_a_run_id(str(fake))
    assert found == [
        "a_new_method"
    ], f"the parser missed a method whose run id sits past a 60-line signature: {found}"
    assert not (
        set(found) & (GATED_BY_RUN_ID | READ_ONLY_BY_RUN_ID)
    ), "the control's method must be in neither set, or it proves nothing"


# ── resuming reads another run's CHECKPOINT, so it is gated too ─────────────
#
# Not a mutating call -- it does not change the run it resumes from -- which is
# why the review set it aside from the mutating clause. It still takes that run's
# trained weights and continues from them, so the resulting model derives from a
# checkpoint the caller was never granted. The ruling is that reads of a run stay
# open and documented, but a RESUME is gated owner-or-admin.


async def _start(cls, inst, resume_from, who):
    # `init_checkpoint` and `init_from_pool` are passed EXPLICITLY as None, and that
    # is load-bearing rather than tidiness. These tests call the undecorated method
    # directly, so a parameter left out keeps its `Field(...)` object as its value --
    # and `start_training` opens by refusing more than one of init_checkpoint,
    # resume_session_id and init_from_pool, counting anything that is `not None and
    # != ""`. Two FieldInfo objects count as two supplied arguments, so every call
    # here died on that ValueError before reaching the gate under test.
    #
    # `dataset_artifact_id` is explicit for the same reason and a different route: a
    # FieldInfo is truthy, so this row's new dataset-claim check ran on it and failed
    # reaching the artifact manager -- which short-circuited the resume gate the
    # control below asserts was CONSULTED. Four tests in this file were being decided
    # by an argument-shape accident rather than by ownership; measured, the three
    # lines take them from 4 failed to 4 passed.
    return await cls.start_training(
        inst,
        train_images=["a.tif"],
        train_labels=["b.tif"],
        resume_session_id=resume_from,
        init_checkpoint=None,
        init_from_pool=None,
        dataset_artifact_id=None,
        context=who,
    )


@pytest.mark.asyncio
async def test_RESUMING_someone_elses_run_is_refused(runs):
    sid = _owned(runs, "2026-11-25-150730-abcdef12", ME)
    cls, inst = _entry()
    for who in (ctx(OTHER, anonymous=False), ctx(ANON_A), None):
        with pytest.raises(training.NotTheOwner):
            await _start(cls, inst, sid, who)


@pytest.mark.asyncio
async def test_resuming_an_OWNERLESS_run_is_admin_only(runs):
    sid = "2026-11-25-150730-abcdef12"
    runs(sid, None)
    cls, inst = _entry()
    with pytest.raises(training.NotTheOwner):
        await _start(cls, inst, sid, ctx(ANON_A))


@pytest.mark.asyncio
async def test_resuming_leaves_NO_session_behind_when_refused(runs, tmp_path):
    """A refused resume must not create the new run either — otherwise the
    refusal still costs a PREPARING record, which is the manufacture-by-calling
    shape this app has already been bitten by."""
    sid = _owned(runs, "2026-11-25-150730-abcdef12", ME)
    before = sorted(p.name for p in training.sessions_root().iterdir())
    cls, inst = _entry()
    with pytest.raises(training.NotTheOwner):
        await _start(cls, inst, sid, ctx(OTHER, anonymous=False))
    assert sorted(p.name for p in training.sessions_root().iterdir()) == before


@pytest.mark.asyncio
async def test_resuming_YOUR_OWN_run_gets_past_the_gate(runs):
    """THE CONTROL, and it asserts the gate was CONSULTED rather than merely that
    some other error came back — the same mistake that made an earlier control
    pass on a missing-argument TypeError."""
    sid = _owned(runs, "2026-11-25-150730-abcdef12", ME)
    cls, inst = _entry()
    seen = []
    # Resume is gated by require_caller_may_RESUME_run, not by the mutate gate:
    # a dataset's managers may continue a colleague's training, while stopping,
    # exporting and contributing stay with the owner. Spying on the mutate gate
    # here would pass vacuously -- it is never called on this path any more.
    real = training.require_caller_may_resume_run

    def spy(session_id, context, **kw):
        seen.append(session_id)
        return real(session_id, context, **kw)

    training.require_caller_may_resume_run = spy
    try:
        with pytest.raises(Exception) as exc:
            await _start(cls, inst, sid, ctx(ME, anonymous=False))
    finally:
        training.require_caller_may_resume_run = real
    assert seen == [sid], f"the gate was not consulted for the resume: {seen}"
    assert not isinstance(exc.value, training.NotTheOwner)


@pytest.mark.asyncio
async def test_NOT_resuming_does_not_consult_EITHER_gate(runs):
    """The paired control the other way: a normal start must not be gated on
    anything, or every caller would need an owned run to train at all.

    It spies BOTH gates deliberately. This row moves the resume check from
    `require_caller_may_mutate` to `require_caller_may_resume_run`, so a control
    that watched only the mutate gate would keep passing while the resume gate
    was consulted on every plain start -- the exact regression it exists to
    catch, made invisible by the very change under review.
    """
    cls, inst = _entry()
    seen = {"mutate": [], "resume": []}
    real_mutate = training.require_caller_may_mutate
    real_resume = training.require_caller_may_resume_run

    def spy_mutate(session_id, context, **kw):
        seen["mutate"].append(session_id)
        return real_mutate(session_id, context, **kw)

    def spy_resume(session_id, context, **kw):
        seen["resume"].append(session_id)
        return real_resume(session_id, context, **kw)

    training.require_caller_may_mutate = spy_mutate
    training.require_caller_may_resume_run = spy_resume
    try:
        with pytest.raises(Exception):
            await _start(cls, inst, None, ctx(ANON_A))
    finally:
        training.require_caller_may_mutate = real_mutate
        training.require_caller_may_resume_run = real_resume
    assert seen == {
        "mutate": [],
        "resume": [],
    }, f"a plain start consulted a run-ownership gate: {seen}"
