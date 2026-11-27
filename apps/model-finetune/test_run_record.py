"""A run record that outlives the worker, and does not archive a signature.

Run history lives in one worker's filesystem, under the mounted sessions root,
whose own comment says the mount is what makes it survive a RESTART. It does,
and for a restart that is enough. What it does not survive is the worker being
replaced, rolled, or the app running somewhere else -- and a run history can
only be read by asking the worker that happens to hold it, so "previous runs"
is a property of who answered rather than of the project. Measured on one
worker as the same administrator: the production instance listed 261 runs, a
second instance of the same app listed 19, with nothing in common.

``run_record`` is the document that fixes that. Three properties are pinned
here:

VERBATIM. ``config`` and ``status`` are copied, not enumerated. Per-epoch
metrics are written into the status document by the training workers, so a
record naming known fields would need editing every time a worker learns to
report something new -- in a second place, by whoever added it. Copying means a
new metric appears here the moment it appears there. One test pins that with a
field no version of this code has ever heard of.

REDACTED. The params carry the image and label references the run was given,
and those can be presigned, where the query string IS the credential. A record
meant to be read by whoever opens the run history must not be where a signature
is archived.

PRIVATE FIELDS REMOVED. The status document gained the caller identity when the
listing became per-caller. A record read by whoever opens the run history must
not carry it, and the removal iterates the private-field tuple so that adding a
field there is enough -- see ``test_run_record_artifact.py``, which adds one.
"""

from __future__ import annotations

import json

import pytest

import training

# A well-formed id, since `validate_session_id` enforces the generator's shape
# (YYYY-MM-DD-HHMMSS-xxxxxxxx) and every path in this app is built through it.
SESSION = "2026-11-25-120000-abcdef12"
ABSENT = "2020-01-01-000000-deadbeef"


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A session directory with a status and a params file, and nothing else."""
    monkeypatch.setattr(
        training, "sessions_root", lambda: tmp_path / "sessions", raising=True
    )
    (tmp_path / "sessions" / SESSION).mkdir(parents=True)
    return SESSION


def _write(session_id, status=None, params=None):
    d = training.session_dir(session_id)
    if status is not None:
        (d / "status.json").write_text(json.dumps(status))
    if params is not None:
        (d / "training_params.json").write_text(json.dumps(params))


def test_the_IMAGE_REFERENCES_ARE_NOT_IN_THE_RECORD_AT_ALL(session):
    """STRONGER than the redaction this test used to assert, and the change is
    deliberate.

    It used to plant a presigned image URL in the params and check that the
    query string was gone. The params are now projected through the display
    allow-list before they are stored, so `train_images`, `train_labels`,
    `val_images`, `val_labels` and `init_checkpoint` are not in the record in
    any form -- there is no signature to redact because there is no URL. That
    is what the local listing has always done, and a review found the shared
    path bypassing it.
    """
    _write(
        session,
        status={"status": "COMPLETED"},
        params={
            "train_images": [
                "https://s3.example/bucket/img-1.tif?X-Amz-Signature=deadbeef&e=1",
                "https://s3.example/bucket/img-2.tif",
            ],
            "init_checkpoint": "/models/u1/ckpt.pth",
            "n_epochs": 5,
            "model_type": "cpsam",
        },
    )
    rec = training.run_record(session)
    assert set(rec["config"]) == {"n_epochs", "model_type"}, rec["config"]
    blob = json.dumps(rec)
    assert "deadbeef" not in blob, "a signature survived into the record"
    assert "img-1.tif" not in blob, "an image reference survived into the record"
    assert "/models/u1/ckpt.pth" not in blob, "a worker-local path survived"
    assert rec["config"]["n_epochs"] == 5


def test_a_presigned_url_in_an_ALLOW_LISTED_field_still_loses_its_signature(session):
    """The redaction is not dead code after the allow-list: an allow-listed
    field can carry a URL. `pool_provenance` is a free-form mapping recorded
    with the run, so anything a caller put in it travels."""
    _write(
        session,
        status={"status": "COMPLETED"},
        params={
            "model_type": "cpsam",
            "pool_provenance": {
                "dataset_artifact_id": "bioimage-io/x",
                "source": "https://ov.example/zarr/ds1/?token=secret-123",
            },
        },
    )
    rec = training.run_record(session)
    assert "secret-123" not in json.dumps(rec)
    assert (
        rec["config"]["pool_provenance"]["source"]
        == "https://ov.example/zarr/ds1/?<redacted>"
    )
    # The path is kept, because it identifies the object.
    assert "ds1" in rec["config"]["pool_provenance"]["source"]


def test_a_presigned_url_in_the_STATUS_is_redacted_too(session):
    """The status is the half that carries an export handle.

    ``get_export_status`` hands back a presigned download URL for a run's
    weights with hours of life left in it. The config is not the only place a
    signature can reach the record, and redacting one half would read as
    protection while publishing the other.
    """
    _write(
        session,
        status={
            "status": "COMPLETED",
            "download_url": "https://s3.example/w/final.pt?X-Amz-Signature=cafe99",
        },
        params={"n_epochs": 1},
    )
    rec = training.run_record(session)
    assert "cafe99" not in json.dumps(rec), "a signature survived in the status"
    assert rec["status"]["download_url"] == "https://s3.example/w/final.pt?<redacted>"


def test_redaction_reaches_nested_structures(session):
    """A signature hiding one level down, inside a list inside an allow-listed
    mapping, is still a signature."""
    _write(
        session,
        status={"status": "COMPLETED"},
        params={
            "model_type": "cpsam",
            "pool_provenance": {
                "members": [
                    {
                        "kind": "ngff",
                        "store_root": "https://ov.example/zarr/ds1/?token=secret-123",
                        "region": {"x": 1, "y": 2, "width": 3, "height": 4},
                    }
                ]
            },
        },
    )
    rec = training.run_record(session)
    assert "secret-123" not in json.dumps(rec)
    src = rec["config"]["pool_provenance"]["members"][0]
    assert src["store_root"].startswith("https://ov.example/zarr/ds1/")
    # Non-string leaves are untouched, so the region survives as numbers.
    assert src["region"] == {"x": 1, "y": 2, "width": 3, "height": 4}


def test_the_status_is_copied_verbatim_including_fields_this_code_never_heard_of(
    session,
):
    """THE property the design turns on.

    A worker that starts reporting a new metric must not need this module
    edited. The key below is deliberately invented: if the record enumerated
    known fields it would be dropped here, silently, and a run history would
    show a blank curve while the worker had the numbers all along.
    """
    _write(
        session,
        status={
            "status": "COMPLETED",
            "epochs": [{"epoch": 0, "train_loss": 0.5}],
            "some_metric_invented_after_this_test_was_written": {"deep": [1, 2, 3]},
        },
        params={"n_epochs": 1},
    )
    rec = training.run_record(session)
    assert rec["status"]["epochs"] == [{"epoch": 0, "train_loss": 0.5}]
    assert rec["status"]["some_metric_invented_after_this_test_was_written"] == {
        "deep": [1, 2, 3]
    }


def test_a_run_that_died_before_writing_params_still_gets_a_record(session):
    """An absent config is reported as absent, not as ``{}``.

    An empty mapping would read as "trained with no settings", which is a
    different and wrong claim.
    """
    _write(session, status={"status": "FAILED", "message": "boom"})
    rec = training.run_record(session)
    assert rec["config"] is None
    assert rec["status"]["status"] == "FAILED"
    assert rec["session_id"] == session
    assert rec["record_version"] == training.RUN_RECORD_VERSION


def test_an_unknown_session_records_the_unknown_status_rather_than_raising(session):
    """``read_status`` returns {} for a session that does not exist. A record is
    still produced, because a run history asking about a vanished id wants an
    answer rather than a traceback.

    The id has to be WELL-FORMED to get this far: an id the server could not
    have issued is refused before any path is built, which is a different
    failure and is pinned in ``test_run_record_artifact.py``.
    """
    rec = training.run_record(ABSENT)
    assert rec["session_id"] == ABSENT
    assert rec["config"] is None
    assert rec["status"] == {}


def test_a_caller_may_supply_the_status_it_just_wrote(session):
    """The record builder is called right after a status write, so it accepts
    the document rather than re-reading it. Passing one must not quietly fall
    back to the file: the two differ, and this asserts which one wins."""
    _write(session, status={"status": "TRAINING"}, params={"n_epochs": 2})
    rec = training.run_record(session, {"status": "COMPLETED", "fresh": True})
    assert rec["status"] == {"status": "COMPLETED", "fresh": True}
    # The config still comes from disk, so supplying a status is not a way to
    # produce a record with no settings.
    assert rec["config"] == {"n_epochs": 2}


@pytest.mark.parametrize(
    "value, expected",
    [
        ("plain string", "plain string"),
        ("ftp://host/x?y=1", "ftp://host/x?y=1"),  # only http(s) is touched
        (7, 7),
        (None, None),
        (True, True),
        ([{"a": "http://h/p?q=1"}], [{"a": "http://h/p?<redacted>"}]),
    ],
)
def test_redact_urls_leaves_everything_else_alone(value, expected):
    """The redaction must not become a general mangler: a non-http string, a
    number and a bool all pass through untouched."""
    assert training.redact_urls(value) == expected
