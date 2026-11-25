"""The entry.py half of dataset co-manager authority, which nothing tested.

Review applied five mutations to `entry.py`, one at a time, and the whole suite
stayed green through all five at 868 passed:

  a) the check that refuses an unmanaged `dataset_artifact_id`, disabled
  b) `_caller_managed_datasets` made fail-OPEN, so an unreadable artifact counts
     as managed
  c) the listing passing every dataset on disk as managed
  d) resume passing the prior run's dataset as managed, without the ACL check
  e) the ACL check replaced by `if True:`

The live harness would have caught c, d and e -- once -- and could never catch a
or b, because it never named a dataset its caller did not manage and never made
an artifact unreadable. It is also untracked, so it is not on the branch and
nobody else can re-run it. Two of the headline guarantees of this change, "a
refused claim" and "fails closed per artifact", were therefore pinned by
nothing.

These tests fake the artifact manager rather than the authority function, so the
thing under test is the wiring in `entry.py`: which permissions document reaches
`caller_manages_dataset`, and what happens when one cannot be read.
"""

import asyncio
import json

import pytest

import training

MANAGER = {"user": {"id": "u-manager", "is_anonymous": False}}
STRANGER = {"user": {"id": "u-stranger", "is_anonymous": False}}

# `build_acl_permissions` maps owner and managers to "*"; the wildcard KEY "*"
# is what makes a dataset public. The manager VALUE and the public KEY are the
# same string, which is the trap this file must not fall into.
MANAGED_BY_MANAGER = {"config": {"permissions": {"u-manager": "*"}}}
READ_ONLY_FOR_MANAGER = {"config": {"permissions": {"u-manager": "r+"}}}
PUBLIC_BUT_UNMANAGED = {"config": {"permissions": {"*": "r+"}}}


class FakeArtifactManager:
    """Answers `read` from a dict; anything absent raises, as the real one does."""

    def __init__(self, artifacts, fail_with=None):
        self._artifacts = artifacts
        self._fail_with = fail_with
        self.reads = []

    async def read(self, artifact_id):
        self.reads.append(artifact_id)
        if self._fail_with is not None:
            raise self._fail_with
        try:
            return self._artifacts[artifact_id]
        except KeyError:
            raise RuntimeError(f"ArtifactNotFound: {artifact_id}")


class FakeHypha:
    def __init__(self, am):
        self._am = am

    async def get_service(self, name):
        assert name == "public/artifact-manager", name
        return self._am


def _entry_class():
    import entry

    for obj in vars(entry).values():
        if isinstance(obj, type) and obj.__name__ == "EntryApp":
            return obj
    for obj in vars(entry).values():
        cls = getattr(obj, "__wrapped_class__", None) or getattr(
            obj, "func_or_class", None
        )
        if isinstance(cls, type) and cls.__name__ == "EntryApp":
            return cls
    raise AssertionError("EntryApp not found")


def _app(am):
    cls = _entry_class()
    inst = cls.__new__(cls)
    inst._hypha = FakeHypha(am)
    return cls, inst


async def _resolve(am, ids, context):
    """Awaited directly. An earlier version drove this with
    `get_event_loop().run_until_complete`, which passed when this file ran
    alone and failed in the full suite, because by then another test had
    closed the loop it fetched."""
    cls, inst = _app(am)
    return await cls._caller_managed_datasets(inst, ids, context)


# ── the resolver itself ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_dataset_the_caller_MANAGES_comes_back():
    """THE CONTROL. Every test below asserts something is absent from the
    result; a resolver that always returned the empty set would satisfy all of
    them and grant nobody anything."""
    am = FakeArtifactManager({"ws/ds": MANAGED_BY_MANAGER})
    assert await _resolve(am, ["ws/ds"], MANAGER) == frozenset({"ws/ds"})


@pytest.mark.asyncio
async def test_a_dataset_the_caller_only_READS_does_not():
    am = FakeArtifactManager({"ws/ds": READ_ONLY_FOR_MANAGER})
    assert await _resolve(am, ["ws/ds"], MANAGER) == frozenset()


@pytest.mark.asyncio
async def test_a_PUBLIC_dataset_does_not_make_everyone_a_manager():
    """The wildcard KEY grants reading to all; it is not a manager grant.

    The manager permission and the public key are both the string "*", so a
    check that looks for "*" in the wrong position hands every public dataset's
    runs to every caller.
    """
    am = FakeArtifactManager({"ws/ds": PUBLIC_BUT_UNMANAGED})
    assert await _resolve(am, ["ws/ds"], STRANGER) == frozenset()


@pytest.mark.asyncio
async def test_an_UNREADABLE_artifact_is_not_managed__fails_CLOSED(caplog):
    """Mutant (b): fail-open here hands a deleted or withheld dataset to anyone."""
    am = FakeArtifactManager({}, fail_with=RuntimeError("permission denied"))
    assert await _resolve(am, ["ws/ds"], MANAGER) == frozenset()


@pytest.mark.asyncio
async def test_one_unreadable_dataset_does_not_decide_for_the_OTHERS():
    """ "Fails closed per artifact, not per call" -- asserted, not asserted-of.

    A resolver that abandoned the whole call on the first failure would also
    pass the test above, and would silently drop co-manager access for every
    other dataset whenever one was deleted.
    """

    class Mixed(FakeArtifactManager):
        async def read(self, artifact_id):
            self.reads.append(artifact_id)
            if artifact_id == "ws/gone":
                raise RuntimeError("ArtifactNotFound")
            return MANAGED_BY_MANAGER

    am = Mixed({})
    got = await _resolve(am, ["ws/gone", "ws/ok"], MANAGER)
    assert got == frozenset({"ws/ok"}), got
    assert am.reads == ["ws/gone", "ws/ok"], am.reads


@pytest.mark.asyncio
async def test_an_unreachable_artifact_MANAGER_grants_nothing():
    class Broken:
        async def get_service(self, name):
            raise RuntimeError("artifact manager is down")

    cls, inst = _app(None)
    inst._hypha = Broken()
    got = await cls._caller_managed_datasets(inst, ["ws/ds"], MANAGER)
    assert got == frozenset()


@pytest.mark.asyncio
async def test_NO_DATASETS_makes_no_round_trips_at_all():
    """The short-circuit, which is the common case on a store of older runs."""
    am = FakeArtifactManager({"ws/ds": MANAGED_BY_MANAGER})
    assert await _resolve(am, [], MANAGER) == frozenset()
    assert am.reads == []


@pytest.mark.asyncio
async def test_an_unusable_context_grants_nothing():
    am = FakeArtifactManager({"ws/ds": MANAGED_BY_MANAGER})
    for bad in (None, "not-a-dict", 42):
        assert await _resolve(am, ["ws/ds"], bad) == frozenset()


# ── what the listing hands the authority function ───────────────────────────


@pytest.mark.asyncio
async def test_the_LISTING_passes_only_the_RESOLVED_datasets_not_all_of_them(
    monkeypatch, tmp_path
):
    """Mutant (c): passing every dataset on disk as managed.

    The resolver can be perfect and the listing still wrong, if what it passes
    down is not what the resolver returned. So capture the argument.
    """
    seen = {}

    def fake_list_sessions(context, *, writable_datasets=frozenset(), records=None):
        seen["writable"] = writable_datasets
        return {}

    monkeypatch.setattr(
        training, "scan_runs_for_listing", lambda: ({}, ["ws/mine", "ws/theirs"], [])
    )
    monkeypatch.setattr(training, "list_sessions", fake_list_sessions)

    am = FakeArtifactManager(
        {"ws/mine": MANAGED_BY_MANAGER, "ws/theirs": PUBLIC_BUT_UNMANAGED}
    )
    cls, inst = _app(am)
    await cls.list_training_sessions(inst, context=MANAGER)
    assert seen["writable"] == frozenset({"ws/mine"}), seen["writable"]


# ── start_training's two gates: the claim, and the resume ────────────────────
#
# Review's mutants (a) and (d). Both live in `start_training`, which is why
# they survived the resolver tests above: nothing drove the call.


async def _start(inst, cls, **kwargs):
    """Call start_training far enough to exercise its gates and no further.

    The gates run before the session directory exists, by design, so a refused
    call must raise before any filesystem work. That is also what lets this be
    driven without a Ray runtime: anything that gets PAST the gates fails later
    for an unrelated reason, which the tests below distinguish by exception
    type rather than by "it raised".
    """
    return await cls.start_training(inst, **kwargs)


@pytest.mark.asyncio
async def test_a_dataset_the_caller_does_NOT_manage_is_REFUSED_on_start(monkeypatch):
    """Mutant (a): disabling this check lets anyone attach a run to any dataset.

    That is not privilege escalation -- the run is the caller's own -- but it
    puts arbitrary content into a team's listing of a dataset they manage, in
    front of people who never agreed to see it.
    """
    am = FakeArtifactManager({"ws/theirs": PUBLIC_BUT_UNMANAGED})
    cls, inst = _app(am)
    with pytest.raises(training.NotTheOwner, match="do not own or manage"):
        await _start(
            inst,
            cls,
            train_images=[],
            train_labels=[],
            dataset_artifact_id="ws/theirs",
            context=STRANGER,
        )


@pytest.mark.asyncio
async def test_a_dataset_the_caller_DOES_manage_gets_past_that_gate(monkeypatch):
    """THE CONTROL for the test above.

    A `start_training` that refused every dataset would satisfy it and break
    the feature. So assert this call does NOT raise NotTheOwner -- it fails
    later, on the empty training data, which is a different exception and
    proves the gate let it through.
    """
    am = FakeArtifactManager({"ws/mine": MANAGED_BY_MANAGER})
    cls, inst = _app(am)
    try:
        await _start(
            inst,
            cls,
            train_images=[],
            train_labels=[],
            dataset_artifact_id="ws/mine",
            context=MANAGER,
        )
    except training.NotTheOwner as exc:  # pragma: no cover - the failure we test for
        raise AssertionError(f"a managed dataset was refused: {exc}")
    except Exception:
        pass  # anything else means the gate passed, which is the claim here


@pytest.mark.asyncio
async def test_RESUMING_a_run_on_a_dataset_the_caller_does_not_manage_is_REFUSED(
    monkeypatch, tmp_path
):
    """Mutant (d): passing the prior run's dataset as managed, unchecked.

    Resuming continues from another run's CHECKPOINT, so the resulting model
    derives from weights the caller never trained. The gate is by access --
    owner, dataset manager, or admin -- and the mutant hands the second of
    those to everybody by asserting the prior run's dataset is managed without
    asking the artifact manager.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(training, "sessions_root", lambda: root)
    monkeypatch.setattr(training, "session_dir", lambda sid: root / sid)
    import re as _re

    monkeypatch.setattr(training, "SESSION_ID_PATTERN", _re.compile(r"\A[\w.-]+\Z"))
    prior = root / "someone-elses-run"
    prior.mkdir()
    (prior / "status.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "owner": "u-manager",
                training.DATASET_FIELD: "ws/theirs",
            }
        )
    )

    # The stranger manages nothing: the artifact reads back with no grant.
    am = FakeArtifactManager({"ws/theirs": PUBLIC_BUT_UNMANAGED})
    cls, inst = _app(am)
    with pytest.raises(training.NotTheOwner):
        await _start(
            inst,
            cls,
            train_images=[],
            train_labels=[],
            resume_session_id="someone-elses-run",
            context=STRANGER,
        )


@pytest.mark.asyncio
async def test_a_CO_MANAGER_may_resume_the_same_run(monkeypatch, tmp_path):
    """THE CONTROL for the resume gate, and the feature this row exists for."""
    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(training, "sessions_root", lambda: root)
    monkeypatch.setattr(training, "session_dir", lambda sid: root / sid)
    import re as _re

    monkeypatch.setattr(training, "SESSION_ID_PATTERN", _re.compile(r"\A[\w.-]+\Z"))
    prior = root / "a-colleagues-run"
    prior.mkdir()
    (prior / "status.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "owner": "u-somebody-else",
                training.DATASET_FIELD: "ws/shared",
            }
        )
    )

    am = FakeArtifactManager({"ws/shared": MANAGED_BY_MANAGER})
    cls, inst = _app(am)
    try:
        await _start(
            inst,
            cls,
            train_images=[],
            train_labels=[],
            resume_session_id="a-colleagues-run",
            context=MANAGER,
        )
    except training.NotTheOwner as exc:  # pragma: no cover
        raise AssertionError(f"a dataset co-manager was refused a resume: {exc}")
    except Exception:
        pass
