"""A run records whose it is, at creation, from the caller context Hypha injects.

Why at creation and nowhere else: nothing on a run record identifies its owner.
Measured on production, the union of keys across all 261 stored runs is 29
fields and the only one with an owner-ish name is `terminated_by`, which holds
`"child"` or `"user"` — a cause, not an identity. So "the listing returns the
caller's own runs" cannot be built as a filter over what exists; the attribution
has to be written when the run starts, and it cannot be recovered afterwards.

The identity is trustworthy because the caller does not supply it: Hypha injects
`context` at the service boundary and the ProxyDeployment forwards it, which is
why `@bioengine.method(context=True)` is the mechanism rather than a parameter.

**Three states, deliberately, and the third is not the second.** A known user, a
known anonymous caller, and a call whose context says nothing. That last one is
also the shape of every run written before this field existed. Collapsing
"unknown" into "anonymous" would either hide a user's own runs or expose an
anonymous one, so the tests below pin all three apart.
"""

from __future__ import annotations

import pytest

import training

# A session id must match the shape the server generates
# (YYYY-MM-DD-HHMMSS-xxxxxxxx). `training.session_dir` refuses anything else --
# a caller-supplied id was once used as a directory name unchecked, which let
# `../x` and absolute paths write outside the sessions root. So the short ids
# these tests used ("s1".."s5") can no longer name a directory; the constants
# keep the tests readable while using ids the app accepts.
S_EXTRA = "2026-11-25-130005-00000005"
S_OWNED = "2026-11-25-130001-00000001"
S_PRIVATE_READ = "2026-11-25-130004-00000004"
S_PUBLIC_READ = "2026-11-25-130003-00000003"
S_UNOWNED = "2026-11-25-130002-00000002"


def test_a_known_user_is_recorded_with_anonymous_False():
    got = training.caller_identity(
        {"user": {"id": "github|49943582", "is_anonymous": False}}
    )
    assert got == {"owner": "github|49943582", "owner_anonymous": False}


def test_a_known_anonymous_caller_is_recorded_AS_anonymous():
    got = training.caller_identity(
        {"user": {"id": "anonymouz-abc123", "is_anonymous": True}}
    )
    assert got == {"owner": "anonymouz-abc123", "owner_anonymous": True}


def test_a_missing_flag_is_UNKNOWN_not_False():
    """An id with no anonymity flag is a caller we can name but cannot classify.
    Recording False would assert something nobody said."""
    got = training.caller_identity({"user": {"id": "someone"}})
    assert got == {"owner": "someone", "owner_anonymous": None}


@pytest.mark.parametrize(
    "ctx",
    [
        None,
        {},
        {"user": None},
        {"user": {}},
        {"user": {"id": None}},
        {"user": {"id": ""}},
        {"user": {"id": 123}},
        {"user": "github|1"},
        "not a dict",
        [],
    ],
)
def test_a_context_that_names_nobody_records_NOTHING(ctx):
    """Not a placeholder, and not an exception. `owner: None` is the same shape
    as every run that predates this field, which is what lets a reader treat
    "not recorded" as its own case instead of guessing."""
    assert training.caller_identity(ctx) == {"owner": None, "owner_anonymous": None}


def test_anonymity_is_NOT_inferred_from_the_shape_of_the_id():
    """An anonymous workspace is recognisably named today and that is a
    convention, not a contract. Keying on it would mislabel runs silently the
    day it changes, so only the flag counts."""
    got = training.caller_identity({"user": {"id": "ws-user-anonymouz-9f3a"}})
    assert got["owner"] == "ws-user-anonymouz-9f3a"
    assert (
        got["owner_anonymous"] is None
    ), "anonymity was inferred from the id rather than read from the flag"
    # and the converse: a real-looking id with the flag set IS anonymous
    got2 = training.caller_identity(
        {"user": {"id": "github|49943582", "is_anonymous": True}}
    )
    assert got2["owner_anonymous"] is True


def test_identity_never_raises_whatever_it_is_handed():
    """A malformed context must not fail a training run that is otherwise valid:
    the run is the user's work and the attribution is bookkeeping."""

    class Hostile(dict):
        def get(self, *a, **k):
            raise RuntimeError("boom")

    for ctx in (Hostile(), object(), 3.14, b"bytes"):
        try:
            got = training.caller_identity(ctx)
        except Exception as exc:  # pragma: no cover - the assertion is the point
            pytest.fail(f"caller_identity raised on {ctx!r}: {exc!r}")
        assert set(got) == {"owner", "owner_anonymous"}


# ── the wiring: the entry must ASK for the context and record what it returns ──


# ── behaviourally: the owner is WRITTEN, and the public read path REDACTS it ──
#
# The previous tests here were substring greps over entry.py. The reviewer broke
# them twice while keeping the substring alive -- once by calling the helper and
# discarding its result, once by passing `None` instead of `context` with the
# original kept in a comment -- and both mutations left the suite green. A grep
# cannot see what a write actually stores, so these drive the write and read the
# file back.


@pytest.fixture()
def sessions(tmp_path, monkeypatch):
    """Point the session store at a temp dir, so the real write/read runs."""
    monkeypatch.setattr(training, "SESSIONS_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(training, "sessions_root", lambda: tmp_path)
    return tmp_path


def _write_owned(sid, ctx):
    training.write_status(
        sid, status="PREPARING", created_at=1.0, **training.caller_identity(ctx)
    )


def test_the_owner_REACHES_THE_STORED_RECORD(sessions):
    """What a grep cannot check: that the write stores the injected identity."""
    _write_owned(S_OWNED, {"user": {"id": "github|49943582", "is_anonymous": False}})
    on_disk = training.read_status(S_OWNED)
    assert on_disk["owner"] == "github|49943582", on_disk
    assert on_disk["owner_anonymous"] is False


def test_an_unidentified_caller_stores_no_owner(sessions):
    _write_owned(S_UNOWNED, None)
    on_disk = training.read_status(S_UNOWNED)
    assert on_disk.get("owner") is None
    assert on_disk.get("owner_anonymous") is None


def test_THE_PUBLIC_READ_PATH_DOES_NOT_RETURN_THE_OWNER(sessions):
    """THE FINDING THIS COMMIT EXISTS FOR. `list_training_sessions` answers
    anonymous callers and returns whatever get_status returns, so recording an
    owner without redacting it publishes the creator's user id to anybody."""
    _write_owned(
        S_PUBLIC_READ, {"user": {"id": "github|49943582", "is_anonymous": False}}
    )
    public = training.get_status(S_PUBLIC_READ)
    assert "owner" not in public, f"the owner is in the public response: {public}"
    assert "owner_anonymous" not in public
    # Listed AS ITS OWNER. The listing is per-caller, so any other context would
    # return {} and make the redaction assertion below vacuous -- the
    # `S_PUBLIC_READ in listed` guard on the next line is what keeps that honest.
    listed = training.list_sessions(
        {"user": {"id": "github|49943582", "is_anonymous": False}}
    )
    assert S_PUBLIC_READ in listed, "the run vanished from the listing entirely"
    assert (
        "owner" not in listed[S_PUBLIC_READ]
    ), f"list_sessions publishes the owner: {listed['s3']}"


def test_a_caller_THAT_ASKS_can_still_read_the_owner(sessions):
    """THE CONTROL for the redaction. Without it, the assertions above would also
    pass if the owner were never stored at all -- which is the bug the reviewer
    produced twice. This proves the field is present and merely withheld, and it
    is the hook the per-caller filter will use."""
    _write_owned(S_PRIVATE_READ, {"user": {"id": "github|7", "is_anonymous": False}})
    private = training.get_status(S_PRIVATE_READ, include_private=True)
    assert private["owner"] == "github|7"
    assert private["owner_anonymous"] is False


def test_redaction_does_not_disturb_the_rest_of_the_record(sessions):
    """A blanket strip would be easy to write and would take the status with it."""
    _write_owned(S_EXTRA, {"user": {"id": "github|7"}})
    public = training.get_status(S_EXTRA)
    assert public["status"] == "PREPARING"
    assert public["session_id"] == S_EXTRA
    assert public["created_at"] == 1.0


# ── the REAL call site: what entry.py passes to the write ─────────────────────
#
# The tests above compose `caller_identity` and `write_status` themselves, so
# they prove each piece works and NOTHING about whether entry.py wires them.
# Two mutations of that call site -- calling the helper and discarding its
# result, and passing `None` with the original kept alive in a comment -- both
# left them green. This drives `start_training` itself and captures what it
# hands the writer, which is exactly what those mutations change.


class _Stop(Exception):
    """Raised from the captured write, to stop start_training at that point."""


def _capture_preparing_write(monkeypatch):
    import entry

    seen = {}

    def _write(session_id, **fields):
        seen["session_id"] = session_id
        seen.update(fields)
        raise _Stop()

    # On the `training` MODULE, not on `entry`: start_training does a local
    # `import training` inside the function body, so there is no `entry.training`
    # attribute to patch -- the name is resolved from sys.modules at call time.
    monkeypatch.setattr(training, "write_status", _write)
    # `object.__new__` skips __init__ on purpose -- constructing the real app
    # needs Hypha. Only the attributes this path touches before the write are
    # supplied, and they were read off the source rather than guessed one
    # exception at a time: _check_runtime_available, _resolve_pool_init and
    # _runtime_gpu_memory are the three.
    cls = getattr(entry.EntryApp, "func_or_class", entry.EntryApp)
    inst = object.__new__(cls)
    inst._enabled_backends = {"microsam", "cellpose"}

    async def _async_none(*a, **k):
        return None

    async def _async_pool(*a, **k):
        return None, None

    # Two of these are awaited on this path, so a plain lambda gives
    # "object NoneType can't be used in 'await' expression" -- which is a
    # harness fault that looks nothing like the thing under test.
    inst._check_runtime_available = _async_none
    inst._resolve_pool_init = _async_pool

    async def _async_gpu(*a, **k):
        # `available: False` so the VRAM floor check is skipped -- this harness
        # is about attribution, not about placement.
        return {"available": False}

    inst._runtime_gpu_memory = _async_gpu
    fn = getattr(cls, "start_training")
    return fn, inst, seen


@pytest.mark.asyncio
async def test_entry_passes_the_INJECTED_caller_to_the_write(monkeypatch):
    fn, inst, seen = _capture_preparing_write(monkeypatch)
    ctx = {"user": {"id": "github|49943582", "is_anonymous": False}}
    with pytest.raises(_Stop):
        await fn(inst, train_images=["a.tif"], train_labels=["b.tif"], context=ctx)
    assert seen.get("status") == "PREPARING", seen
    assert (
        seen.get("owner") == "github|49943582"
    ), f"start_training did not pass the injected caller to the write: {seen}"
    assert seen.get("owner_anonymous") is False


@pytest.mark.asyncio
async def test_entry_passes_NOTHING_when_there_is_no_caller(monkeypatch):
    """THE CONTROL: the owner must track the context, not be a constant. A call
    site hard-coding `None` passes the test above's negation but fails here."""
    fn, inst, seen = _capture_preparing_write(monkeypatch)
    with pytest.raises(_Stop):
        await fn(inst, train_images=["a.tif"], train_labels=["b.tif"], context=None)
    assert seen.get("owner") is None, seen


# ── the decorator: without it, every run records NO owner, silently ───────────
#
# This replaces a test that was deleted and not replaced. The old one grepped
# entry.py for `@bioengine.method(context=True)`; it was removed as "another
# substring grep", which left nothing asserting the decorator at all, so
# reverting it passed the whole file. The behavioural tests above cannot cover
# it either: they pass `context=` straight to the function, so they cannot tell
# an injected context from a client-supplied one.
#
# What reverting the decorator actually does, traced through the transport
# rather than inferred from the schema: the proxy forwards a context to the
# method only `if wants_context` (bioengine/apps/proxy_deployment.py), which the
# decorator sets. Without it the method's `context` parameter keeps its default
# of None, and EVERY run is recorded with no owner -- so a per-caller listing
# degrades to showing nothing rather than to showing the wrong thing. It is
# silent: nothing raises, and the runs still train.
#
# It is NOT an impersonation vector, though the published schema makes it look
# like one. Reverting the decorator does leave `context` in the schema Hypha
# publishes, so a client can send one; but the proxy service is registered with
# `require_context: True`, and hypha_rpc assigns `kwargs["context"] = data["ctx"]`
# after parsing the client's kwargs and before dispatch, overwriting whatever the
# client sent. A client cannot forge an owner by this route. Stated here because
# the schema is one hop and the slot the value lands in is another, and reading
# only the first gives the wrong answer with full confidence.
#
# So these assert the effect, not the decorator's text.


def _start_training_attr():
    import entry

    cls = getattr(entry.EntryApp, "func_or_class", entry.EntryApp)
    fn = getattr(cls, "start_training")
    assert fn is not None
    return fn


def test_context_is_NOT_in_the_schema_hypha_publishes_to_clients():
    """`context` absent from the published schema is the visible signature of the
    decorator being in place. It is not itself the protection -- the transport
    overwrites a client-sent context either way -- but it is the cheapest thing
    to assert that cannot be true unless the method was declared correctly."""
    fn = _start_training_attr()
    schema = getattr(fn, "__schema__", None)
    assert schema, "start_training carries no __schema__ to check"
    props = (schema.get("parameters") or {}).get("properties") or {}
    assert props, f"the schema exposes no properties at all: {schema!r}"
    assert "train_images" in props, (
        f"this test is reading the wrong schema -- it has no train_images: "
        f"{sorted(props)[:8]}"
    )
    assert "context" not in props, (
        "`context` is in the published schema, which means start_training was "
        "NOT declared @bioengine.method(context=True). The proxy forwards a "
        "context only when that flag is set, so every run would be recorded "
        "with owner=None and a per-caller listing would show nobody anything. "
        "Nothing raises when this is wrong."
    )


def test_the_method_is_FLAGGED_as_wanting_the_injected_context():
    """A second, independent arm. The schema check above would also pass if the
    parameter were renamed or removed entirely; this asserts the framework was
    actually asked to inject."""
    fn = _start_training_attr()
    flags = [
        getattr(fn, name)
        for name in dir(fn)
        if "context" in name.lower() and isinstance(getattr(fn, name, None), bool)
    ]
    assert flags, (
        f"no wants-context flag on start_training; attrs: "
        f"{[n for n in dir(fn) if not n.startswith('__')][:12]}"
    )
    assert any(flags), (
        "start_training is not flagged as wanting the injected context, so "
        "Hypha will never pass one and every run records no owner"
    )
