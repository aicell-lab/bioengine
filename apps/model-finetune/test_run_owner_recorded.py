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
    assert got["owner_anonymous"] is None, (
        "anonymity was inferred from the id rather than read from the flag"
    )
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


def _start_training_body() -> str:
    """`start_training`'s source, sliced out of the file by name.

    Read from disk rather than through `inspect`: the class is wrapped by the
    app framework and Ray Serve, so an attribute lookup can land on a wrapper
    and return a few hundred characters of decorator — which would make the
    assertions below pass against text that is not the function.
    """
    import pathlib

    src = (pathlib.Path(__file__).parent / "entry.py").read_text()
    start = src.find("    async def start_training(")
    assert start > 0, "start_training not found in entry.py"
    nxt = src.find("\n    async def ", start + 1)
    body = src[start : nxt if nxt > 0 else len(src)]
    assert len(body) > 2000, f"slice implausibly short ({len(body)} chars)"
    assert body.startswith("    async def start_training("), body[:60]
    assert body.count("\n    async def ") == 0, "the slice spans more than one method"
    return body


def test_start_training_asks_hypha_for_the_caller():
    """`@bioengine.method(context=True)` is what makes Hypha inject the context.
    Without it the parameter is simply never filled and every run records no
    owner — green tests, no attribution."""
    import pathlib

    src = (pathlib.Path(__file__).parent / "entry.py").read_text()
    i = src.find("    async def start_training(")
    assert i > 0
    decorator_region = src[max(0, i - 200) : i]
    assert "@bioengine.method(context=True)" in decorator_region, (
        "start_training is not declared with context=True, so Hypha will not "
        f"inject the caller and the owner can never be recorded. Saw: "
        f"{decorator_region[-80:]!r}"
    )


def test_start_training_takes_a_context_parameter_and_records_it():
    body = _start_training_body()
    assert "context=None," in body, "no context parameter to receive the injection"
    assert "training.caller_identity(context)" in body, (
        "start_training does not record the caller, so the context would be "
        "injected and then dropped"
    )


def test_the_owner_is_written_on_the_record_that_is_CREATED():
    """Not on a later update. A run that fails during preparation must still be
    attributable, and `created_at` is the write that always happens."""
    body = _start_training_body()
    i = body.find('status="PREPARING"')
    assert i > 0, "the PREPARING write moved; this test is keyed on it"
    window = body[i : i + 900]
    assert "caller_identity(context)" in window, (
        "the owner is not part of the PREPARING write, so a run that never "
        "reaches TRAINING would have no owner at all"
    )
