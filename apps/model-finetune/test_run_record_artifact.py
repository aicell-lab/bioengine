"""The artifact-backed run record: what is persisted, and where.

Run history lives on the worker's own disk, so two replicas of one app list
different runs. The fix is one app-owned artifact, written with the app's own
credential. Measured on creation: `bioimage-io/model-finetune-runs` is
standalone (no parent, which is what makes "private" mean anything -- a parent
decides a child's visibility), carries no `"*"` or `"@"` grant, and refuses an
anonymous caller on RPC `read`, RPC `list_files` and HTTP (403).

These tests cover the parts that need no network: which fields are persisted,
where a record goes, and that the artifact id cannot be chosen by a caller.
"""

import ast
import json
import pathlib

import pytest

import training


@pytest.fixture
def absent_sessions_root(tmp_path, monkeypatch):
    """A sessions root with nothing in it.

    ``run_record`` reads the params from disk, so without this the record
    tests would read the real sessions root of whatever machine the suite runs
    on -- passing or failing on data they did not create.
    """
    monkeypatch.setattr(
        training, "sessions_root", lambda: tmp_path / "sessions", raising=True
    )
    (tmp_path / "sessions").mkdir()
    return tmp_path / "sessions"


# ── the artifact id is not an input ──────────────────────────────────────────


def test_the_runs_artifact_id_is_FIXED_in_the_app():
    assert training.RUNS_ARTIFACT_ID == "bioimage-io/model-finetune-runs"


def test_NO_PUBLISHED_METHOD_TAKES_AN_ARTIFACT_ID_FOR_THE_RUNS_RECORD():
    """The finding this design closes, closed by removal rather than validation.

    The app writes the runs artifact with its OWN credential. A caller who
    could name that artifact could make the app write anywhere the credential
    reaches, and an allow-list of acceptable ids is a thing to get wrong. So
    the id is not a parameter anywhere, and this reads the source to say so --
    including for methods nobody has written yet.
    """
    src = pathlib.Path(__file__).with_name("entry.py").read_text()
    tree = ast.parse(src)
    offenders = []
    touching = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
        body = ast.get_source_segment(src, node) or ""
        if "RUNS_ARTIFACT_ID" not in body:
            continue
        touching.append(node.name)
        if "artifact_id" in params:
            offenders.append(node.name)
    # Without this the test is VACUOUS: before the write path existed, no
    # function mentioned the artifact at all, so `offenders == []` held for a
    # tree that could not have offended. It has to fail when the thing it
    # guards is absent, not pass.
    assert touching, (
        "no function in entry.py touches RUNS_ARTIFACT_ID, so this test "
        "examined nothing. Either the write path is gone or it was renamed."
    )
    assert offenders == [], (
        f"these methods touch the runs artifact AND take an artifact_id: "
        f"{offenders}. The id must not be a caller's choice."
    )


# ── what a record carries ────────────────────────────────────────────────────


def test_the_record_KEEPS_the_owner_because_the_FILTER_needs_it(absent_sessions_root):
    """The correction to my own plan, pinned so it cannot be "tidied" back.

    I planned to strip the private fields here. The landed per-caller listing
    decides visibility by comparing the caller to the run's recorded owner, so
    a record arriving from another replica without one is unattributable: the
    listing would show it to everybody or to nobody, and two replicas
    answering the same caller differently is the failure this record exists to
    fix. So the owner travels with the record, and the fence is the artifact's
    permissions plus `redact_private` on the way out -- both checked below.
    """
    status = {
        "status": "COMPLETED",
        "owner": "github|1",
        "owner_anonymous": False,
        "n_epochs": 3,
    }
    rec = training.run_record("2026-11-25-120000-abcdef12", status)
    assert rec["status"]["owner"] == "github|1", (
        "the owner was stripped from the stored record, so a run from another "
        "replica can no longer be attributed and the per-caller listing breaks"
    )
    assert rec["status"]["status"] == "COMPLETED"
    assert rec["status"]["n_epochs"] == 3


def test_WHAT_IS_RETURNED_to_a_caller_still_carries_no_private_field():
    """The other half of the same statement: the strip happens on the way out.

    `redact_private` is the single place it happens, which is what makes the
    record able to carry an owner safely. If this ever passes while the record
    is also stripped, the strip has been duplicated and the two copies can
    drift.
    """
    out = training.redact_private(
        {"status": "COMPLETED", "owner": "github|1", "owner_anonymous": False}
    )
    for field in training.PRIVATE_STATUS_FIELDS:
        assert field not in out, f"{field} reached a caller"
    assert out["status"] == "COMPLETED"


def test_a_field_ADDED_to_PRIVATE_STATUS_FIELDS_is_withheld_too(monkeypatch):
    """The test the plan asks for, at the place the plan was wrong about.

    This app has already had a field withheld on one path and published on
    another. The strip iterates the tuple rather than naming fields, so adding
    one to the tuple must be enough -- and this fails if somebody reintroduces
    an explicit list.
    """
    monkeypatch.setattr(
        training,
        "PRIVATE_STATUS_FIELDS",
        training.PRIVATE_STATUS_FIELDS + ("a_new_private_field",),
    )
    out = training.redact_private(
        {"status": "TRAINING", "a_new_private_field": "secret"}
    )
    assert "a_new_private_field" not in out, (
        "a field added to PRIVATE_STATUS_FIELDS still reached a caller, so the "
        "stripping is naming fields instead of iterating the tuple"
    )
    assert "secret" not in json.dumps(out)


def test_the_record_is_SELF_DESCRIBING(absent_sessions_root):
    """A record read on its own must say which run it is, not rely on its key."""
    rec = training.run_record("2026-11-25-120000-abcdef12", {"status": "FAILED"})
    assert rec["session_id"] == "2026-11-25-120000-abcdef12"
    assert rec["record_version"] == training.RUN_RECORD_VERSION


def test_an_empty_status_does_not_raise(absent_sessions_root):
    rec = training.run_record("2026-11-25-120000-abcdef12", {})
    assert rec["session_id"] and rec["status"] == {} and rec["config"] is None


# ── where a record goes ──────────────────────────────────────────────────────


def test_the_path_is_derived_from_a_VALIDATED_session_id():
    p = training.run_record_path("2026-11-25-120000-abcdef12", "model-finetune")
    assert p == "runs/model-finetune/2026-11-25-120000-abcdef12.json", p


@pytest.mark.parametrize(
    "bad",
    ["../escape", "a/b", "..", ".", "", "with space", "sub/dir/x", None, 7],
)
def test_a_session_id_THAT_IS_NOT_ONE_cannot_become_a_path(bad):
    """The id becomes a path component inside the artifact, so traversal and
    separators have to be impossible rather than unlikely."""
    with pytest.raises((training.InvalidSessionId, TypeError, AttributeError)):
        training.run_record_path(bad, "model-finetune")


# ── the deployment the record belongs to ────────────────────────────────────


@pytest.mark.parametrize(
    "bad",
    ["../escape", "a/b", "", ".", "..", "with space", "x" * 65, None, 7],
)
def test_an_APPLICATION_ID_THAT_IS_NOT_ONE_cannot_become_a_path(bad):
    """The deployment id is a path component too. It comes from the
    environment rather than a caller, which is an argument about today's
    deployment path and not about the next one."""
    with pytest.raises((training.InvalidSessionId, TypeError, AttributeError)):
        training.run_record_path("2026-11-25-120000-abcdef12", bad)


def test_record_store_id_is_OFF_rather_than_guessed(monkeypatch):
    """A replica that cannot tell which deployment it is must not write into
    another deployment's history, and a fallback bucket shared by every such
    replica is exactly that."""
    monkeypatch.delenv(training.APPLICATION_ID_ENV, raising=False)
    assert training.record_store_id() is None
    for bad in ("", "../escape", "a/b", "x" * 65):
        monkeypatch.setenv(training.APPLICATION_ID_ENV, bad)
        assert training.record_store_id() is None, bad
    monkeypatch.setenv(training.APPLICATION_ID_ENV, "model-finetune-dev")
    assert training.record_store_id() == "model-finetune-dev"


def test_TWO_DEPLOYMENTS_DO_NOT_SHARE_A_DIRECTORY():
    """The finding this closes. One artifact is shared by every deployment of
    this app, and a worker runs several: measured on one worker, production
    held 261 runs, the soup 37, and four throwaways 19, 6, 2 and 1. A flat
    `runs/` would have merged every throwaway's test runs into the production
    history -- and the issue's own "the count before equals the count after"
    could then only pass by being counted on the wrong surface."""
    a = training.run_record_path("2026-11-25-120000-abcdef12", "model-finetune")
    b = training.run_record_path("2026-11-25-120000-abcdef12", "model-finetune-dev")
    assert a != b
    assert training.run_record_dir("model-finetune") == "runs/model-finetune"
    assert training.run_record_dir("mf-verify") == "runs/mf-verify"


# ── the context guard, generalised ───────────────────────────────────────────


def _entry_class():
    import entry

    for obj in vars(entry).values():
        if isinstance(obj, type) and obj.__name__ == "EntryApp":
            return obj
        inner = getattr(obj, "__wrapped_class__", None) or getattr(
            obj, "func_or_class", None
        )
        if isinstance(inner, type) and inner.__name__ == "EntryApp":
            return inner
    raise AssertionError("EntryApp not found")


def test_NO_METHOD_WITH_A_CONTEXT_PARAMETER_PUBLISHES_IT():
    """Generalised from the single-method guard, because the risk is general.

    A method declared `@bioengine.method` (without `context=True`) that still
    takes a `context` parameter publishes `context` in its schema and is never
    handed one -- so every caller looks anonymous and nothing raises. The
    existing guard asserted this for one method by name. This asserts it for
    EVERY method that takes a context, so a method written next year is
    covered without anybody remembering to add it here.
    """
    cls = _entry_class()
    checked = []
    offenders = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        fn = getattr(cls, name, None)
        schema = getattr(fn, "__schema__", None)
        if not schema:
            continue
        props = (schema.get("parameters") or {}).get("properties") or {}
        checked.append(name)
        if "context" in props:
            offenders.append(name)
    assert checked, "no method schemas were found at all; this test read nothing"
    assert offenders == [], (
        f"these methods publish `context` in their schema: {offenders}. That "
        "means @bioengine.method was used without context=True, so Hypha never "
        "injects a caller and every request looks anonymous. Nothing raises."
    )
