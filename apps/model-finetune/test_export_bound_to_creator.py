"""An export handle is a bearer capability for a run's WEIGHTS, so it is bound.

`export_model` was already owner-or-admin. The id it returns was not: the READY
record from `get_export_status` carries a presigned `download_url` with six
hours of life in it plus the name of the resumable checkpoint inside the
package, and `push_export` streams the whole package to any presigned URL the
caller supplies. "Who may start an export" and "who may use its handle" are
different questions, and only the first was asked.

The handle is now bound to the identity recorded at creation time. These tests
pin the rule and the three ways an ownership check of this shape goes wrong:

* the refusal must not distinguish "not yours" from "no such export", or the
  check becomes an oracle for guessing ids;
* the admin leg has to be reached for an OWNED export too, not only an
  ownerless one -- the sibling check for training runs shipped with exactly
  that bug, and its tests missed it because the admin fixture used the same id
  as the owner;
* the gate has to run BEFORE the record is read, because reading it sweeps a
  stale record and WRITES -- a refused caller must not be able to change
  anything.

The creator is stored under an underscore key, so the single projection in
`get_export_status` keeps it out of every response. A test asserts that against
what the method RETURNS rather than re-filtering the keys itself, which is how
the sibling test for the package directory once passed no matter what the
method did.
"""

from __future__ import annotations

import json

import pytest

import training


def ctx(uid, *, anonymous=True, grants=None):
    user = {"id": uid, "is_anonymous": anonymous}
    if grants is not None:
        user["scope"] = {"workspaces": grants}
    return {"user": user}


CREATOR = "github|49943582"
STRANGER = "github|000111"
ADMIN_ID = "github|777000"

MINE = ctx(CREATOR, anonymous=False)
THEIRS = ctx(STRANGER, anonymous=False)
ADMIN = ctx(ADMIN_ID, anonymous=False, grants={"bioimage-io": "a"})
ANON = ctx("anonymouz-periwinkle-honeysuckle-17983078")


@pytest.fixture(autouse=True)
def _sandbox_home(monkeypatch, tmp_path):
    """`training.exports_root()` hangs off `Path.home()`."""
    monkeypatch.setenv("HOME", str(tmp_path))


def _method(name, instance=None):
    import entry

    class _Replica:
        _export_tasks: dict = {}

        async def _put_file(self, url, path, timeout=None):
            class _R:
                status_code = 200

                def raise_for_status(self):
                    return None

            return _R()

    return getattr(entry.EntryApp.func_or_class, name).__get__(instance or _Replica())


def plant(export_id, *, owner, status="READY", package_dir=None, files=None):
    """An export record as `export_model` writes one."""
    training.write_export(
        export_id,
        status=status,
        progress=1.0,
        message="ready",
        download_url="https://s3.example/pkg.zip?X-Amz-Signature=deadbeef",
        resume_checkpoint_file="weights.pt",
        size_bytes=123,
        error=None,
        files=files or [{"name": "weights.pt", "size": 123}],
        _owner=owner,
        _owner_anonymous=False if owner else None,
        _package_dir=str(package_dir) if package_dir else "/nonexistent",
    )
    return export_id


# ── the predicate ───────────────────────────────────────────────────────────


def test_the_creator_may_use_their_own_export():
    plant("e-mine", owner=CREATOR)
    assert training.caller_may_use_export("e-mine", MINE) is True


def test_a_STRANGER_may_not():
    plant("e-mine", owner=CREATOR)
    assert training.caller_may_use_export("e-mine", THEIRS) is False


def test_an_ADMIN_may_use_an_OWNED_export():
    """The leg the sibling check for training runs shipped broken: an earlier
    version reached the admin test only for ownerless subjects, so an
    administrator could not touch an owned one while the docstring said they
    could. The admin id here is distinct from the creator's, so "the admin can"
    cannot pass as "the creator can"."""
    plant("e-mine", owner=CREATOR)
    assert ADMIN_ID != CREATOR
    assert training.caller_may_use_export("e-mine", ADMIN) is True


def test_an_export_with_NO_recorded_creator_is_admin_only():
    """The shape of an export minted before the field existed. It cannot be
    attributed to anybody, and guessing is worse than admitting there is
    none. Those expire within hours, so the window is bounded."""
    plant("e-old", owner=None)
    assert training.caller_may_use_export("e-old", ADMIN) is True
    assert training.caller_may_use_export("e-old", MINE) is False
    assert training.caller_may_use_export("e-old", ANON) is False


def test_an_UNKNOWN_export_answers_the_same_as_an_unowned_one():
    """Which is what lets the refusal be identical for both, and what makes
    asking about a non-existent id safe."""
    assert training.caller_may_use_export("e-never-existed", MINE) is False
    assert training.caller_may_use_export("e-never-existed", THEIRS) is False


def test_an_UNUSABLE_context_is_refused_rather_than_matched():
    """Fails closed. A `None` caller must never match a `None` owner."""
    plant("e-old", owner=None)
    plant("e-mine", owner=CREATOR)
    for bad in [None, {}, {"user": None}, {"user": {}}, "nonsense", 7]:
        assert training.caller_may_use_export("e-old", bad) is False
        assert training.caller_may_use_export("e-mine", bad) is False


def test_asking_the_question_CREATES_NOTHING():
    """The predicate reads `read_export`, not `get_export`, which sweeps a
    stale record and writes it back. A refused caller must not be able to
    change state by being refused."""
    root = training.exports_root()
    before = sorted(p.name for p in root.iterdir())
    assert training.caller_may_use_export("e-never-existed", THEIRS) is False
    assert sorted(p.name for p in root.iterdir()) == before


# ── the two published methods ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_export_status_REFUSES_a_stranger_and_leaks_no_url():
    plant("e-mine", owner=CREATOR)
    with pytest.raises(training.NotTheOwner) as exc:
        await _method("get_export_status")("e-mine", context=THEIRS)
    text = str(exc.value)
    assert "deadbeef" not in text and "s3.example" not in text
    assert "weights.pt" not in text
    # The refusal says nothing about the export: not that it exists, not who
    # made it, not that the caller merely lacks admin.
    assert "e-mine" not in text


@pytest.mark.asyncio
async def test_the_REFUSAL_IS_THE_SAME_for_a_stranger_and_for_an_unknown_id():
    """If these differed, the check would be an oracle: a caller could learn
    which ids are real by comparing refusals."""
    plant("e-mine", owner=CREATOR)
    errs = []
    for eid in ("e-mine", "e-never-existed"):
        with pytest.raises(training.NotTheOwner) as exc:
            await _method("get_export_status")(eid, context=THEIRS)
        errs.append((type(exc.value).__name__, str(exc.value)))
    assert errs[0] == errs[1], errs


@pytest.mark.asyncio
async def test_the_CREATOR_still_gets_the_download_url():
    """Paired with the refusals above: without this, a method that refused
    everybody would pass every one of them."""
    plant("e-mine", owner=CREATOR)
    out = await _method("get_export_status")("e-mine", context=MINE)
    assert out["status"] == "READY"
    assert out["download_url"].startswith("https://s3.example/pkg.zip")
    assert out["resume_checkpoint_file"] == "weights.pt"


@pytest.mark.asyncio
async def test_an_ADMIN_still_gets_it_too():
    plant("e-mine", owner=CREATOR)
    out = await _method("get_export_status")("e-mine", context=ADMIN)
    assert out["status"] == "READY"


@pytest.mark.asyncio
async def test_the_CREATOR_is_not_in_what_the_method_RETURNS():
    """Asserted against the method's own output, not against a set this test
    filtered itself -- the latter is true by construction and stays green
    however the projection changes."""
    plant("e-mine", owner=CREATOR)
    out = await _method("get_export_status")("e-mine", context=MINE)
    assert [k for k in out if k.startswith("_")] == [], sorted(out)
    assert CREATOR not in json.dumps(out)
    assert "_package_dir" not in out


@pytest.mark.asyncio
async def test_push_export_REFUSES_a_stranger_before_pushing_anything(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "weights.pt").write_bytes(b"x" * 10)
    plant("e-mine", owner=CREATOR, package_dir=pkg)

    pushed = []

    class _Watch:
        _export_tasks: dict = {}

        async def _put_file(self, url, path, timeout=None):
            pushed.append(url)

            class _R:
                status_code = 200

                def raise_for_status(self):
                    return None

            return _R()

    with pytest.raises(training.NotTheOwner):
        await _method("push_export", _Watch())(
            "e-mine", {"weights.pt": "https://evil.example/put"}, context=THEIRS
        )
    assert pushed == [], "the package was streamed to a refused caller's URL"

    # And the creator's own push still works, so the refusal above is the check
    # and not a broken method.
    out = await _method("push_export", _Watch())(
        "e-mine", {"weights.pt": "https://mine.example/put"}, context=MINE
    )
    assert out == {"pushed": ["weights.pt"], "n_files": 1}
    assert pushed == ["https://mine.example/put"]


@pytest.mark.asyncio
async def test_an_ANONYMOUS_creator_keeps_its_own_export():
    """Every token-less caller is minted a real identity per connection, so an
    anonymous export has an owner like any other and is not ownerless."""
    plant("e-anon", owner=ANON["user"]["id"])
    out = await _method("get_export_status")("e-anon", context=ANON)
    assert out["status"] == "READY"
    assert training.caller_may_use_export("e-anon", ctx("anonymouz-other-77")) is False


@pytest.mark.asyncio
async def test_export_model_RECORDS_the_calling_identity(monkeypatch):
    """The arm that found this test missing.

    Every other test here plants a record, so dropping the `_owner` write in
    `export_model` left the whole file green: a mutant that recorded NO creator
    was caught by nothing, which would have shipped a gate that refuses
    everyone except administrators -- including the creator the page polls as.
    So this calls the method and reads what it wrote.
    """
    import entry

    monkeypatch.setattr(training, "require_caller_may_mutate", lambda sid, ctx: None)
    monkeypatch.setattr(training, "session_backend", lambda sid: "micro_sam")

    class _R:
        def __init__(self):
            self._export_tasks = {}

        def _session_checkpoint(self, session_id):
            return "/nonexistent/checkpoint.pt"

        async def _check_runtime_available(self, served=None):
            return None

        async def _run_export(self, *a, **k):
            return None

    fn = entry.EntryApp.func_or_class.export_model.__get__(_R())
    out = await fn(
        session_id="2026-11-25-120000-abcdef12", name="my model", context=MINE
    )
    rec = training.read_export(out["export_id"])
    assert rec["_owner"] == CREATOR, rec
    assert rec["_owner_anonymous"] is False
    # And the identity it recorded is usable by the creator and nobody else.
    assert training.caller_may_use_export(out["export_id"], MINE) is True
    assert training.caller_may_use_export(out["export_id"], THEIRS) is False


# ── the schema ──────────────────────────────────────────────────────────────


def test_NEITHER_METHOD_PUBLISHES_ITS_CONTEXT():
    """`@bioengine.method` without `context=True` publishes `context` as a
    caller-supplied parameter AND never injects one -- so every caller would
    look unidentified and this whole check would refuse everybody except
    administrators, with nothing raising."""
    import entry

    cls = entry.EntryApp.func_or_class
    checked = []
    for name in ("get_export_status", "push_export", "export_model"):
        schema = getattr(getattr(cls, name), "__schema__", None)
        assert schema, f"{name} has no published schema"
        props = (schema.get("parameters") or {}).get("properties") or {}
        checked.append(name)
        assert "context" not in props, (
            f"{name} publishes `context`, so it is declared without "
            "context=True and is never handed a caller"
        )
    assert checked == ["get_export_status", "push_export", "export_model"]
