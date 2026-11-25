"""A run's configuration is readable from its session record, through an allow-list.

The config was always persisted per run and reachable only from inside the app, so a
history listing could say that a run finished but not what it trained. Exposing it is a
read-surface change, and the risk it carries is publishing something that should not be
published — ``training_params.json`` is assembled by spreading ``**params`` and a
``**resolved`` block, so its contents are whatever those happen to carry.

These tests pin the allow-list behaviour rather than the current contents of that file:
the planted keys below are the point. A change that exposed the whole dict would pass
any test that only checked the wanted fields are present, so each case here also asserts
that an unwanted one is absent.
"""

import json

import pytest

import training


@pytest.fixture()
def session(tmp_path, monkeypatch):
    """A session whose directory is a temp dir, so nothing touches a real worker."""
    sid = "test-session"
    d = tmp_path / sid
    d.mkdir()
    monkeypatch.setattr(training, "session_dir", lambda _sid, _d=d: _d)
    return sid, d


def _write_params(d, params):
    (d / "training_params.json").write_text(json.dumps(params))


def test_the_allow_listed_config_fields_are_returned(session):
    sid, d = session
    _write_params(
        d,
        {
            "model_type": "cpsam",
            "n_epochs": 5,
            "batch_size": 1,
            "learning_rate": 1e-4,
            "patch_shape": [256, 256],
            "metrics": ["loss"],
        },
    )
    cfg = training.run_config(sid)
    assert cfg["model_type"] == "cpsam"
    assert cfg["n_epochs"] == 5
    assert cfg["batch_size"] == 1
    assert cfg["learning_rate"] == 1e-4
    assert cfg["patch_shape"] == [256, 256]
    assert cfg["metrics"] == ["loss"]


def test_local_file_paths_are_not_published(session):
    """The materialised pair paths describe the worker's filesystem and are of no use
    to a client, so they must not travel with the config."""
    sid, d = session
    _write_params(
        d,
        {
            "model_type": "cpsam",
            "train_images": ["/home/bioengine/.bioengine/x/train_0.tif"],
            "train_labels": ["/home/bioengine/.bioengine/x/label_0.tif"],
            "val_images": ["/home/bioengine/.bioengine/x/val_0.tif"],
            "val_labels": ["/home/bioengine/.bioengine/x/vlabel_0.tif"],
            "init_checkpoint": "/home/bioengine/.bioengine/sessions/old/model",
        },
    )
    cfg = training.run_config(sid)
    assert cfg == {"model_type": "cpsam"}, f"a path or checkpoint leaked: {cfg}"


def test_an_unknown_future_field_is_withheld_rather_than_published(session):
    """THE CONTROL THAT MATTERS. The params file is built by spreading other dicts, so
    a field can arrive there without anyone revisiting this surface. An allow-list
    withholds it; a redaction blocklist would have to have guessed the name."""
    sid, d = session
    _write_params(
        d,
        {
            "model_type": "cpsam",
            # Three shapes of thing that must never be published, none of which a
            # name-based blocklist would necessarily catch.
            "pool_token": "eyJhbGciOi.PLANTED.not-a-real-token",
            "caller_email": "someone@example.org",
            "some_field_added_next_year": {"nested": "value"},
        },
    )
    cfg = training.run_config(sid)
    assert cfg == {"model_type": "cpsam"}
    for leaked in ("pool_token", "caller_email", "some_field_added_next_year"):
        assert leaked not in cfg
    # And the values must not appear anywhere in the serialised result either, which
    # catches a nesting mistake that an absent-key check would miss.
    blob = json.dumps(cfg)
    assert "PLANTED" not in blob
    assert "example.org" not in blob


def test_a_run_that_died_before_its_params_were_written_still_lists(session):
    """A preparation failure can terminate before training_params.json exists, and
    those are exactly the runs a history page still has to show. ``get_status`` is on
    every read path, so it must not gain a new way to raise."""
    sid, d = session
    assert not (d / "training_params.json").exists()
    assert training.run_config(sid) == {}


def test_an_unreadable_params_file_is_empty_config_not_an_exception(session):
    sid, d = session
    (d / "training_params.json").write_text("{not json at all")
    assert training.run_config(sid) == {}


def test_a_params_file_holding_a_non_mapping_is_tolerated(session):
    sid, d = session
    (d / "training_params.json").write_text(json.dumps(["not", "a", "mapping"]))
    assert training.run_config(sid) == {}


def test_get_status_carries_the_config_through(session):
    """The allow-list is only useful if it reaches the method a client calls, and
    ``list_sessions`` is built on ``get_status``, so covering this covers both."""
    sid, d = session
    (d / "status.json").write_text(
        json.dumps({"status": "COMPLETED", "created_at": 1.0, "message": "ok"})
    )
    _write_params(d, {"model_type": "cpdino", "n_epochs": 40, "pool_token": "PLANTED"})
    st = training.get_status(sid)
    assert st["status"] == "COMPLETED"
    assert st["config"]["model_type"] == "cpdino"
    assert st["config"]["n_epochs"] == 40
    assert "pool_token" not in st["config"]
    assert "PLANTED" not in json.dumps(st["config"])


def test_the_allow_list_is_not_empty_and_does_filter(session):
    """Guards the two ways this feature could become vacuous: an empty allow-list would
    make every assertion about withholding pass for the wrong reason, and a list
    covering every key would stop it being a filter at all."""
    assert len(training.RUN_CONFIG_FIELDS) >= 6
    sid, d = session
    _write_params(d, {k: 1 for k in training.RUN_CONFIG_FIELDS})
    cfg = training.run_config(sid)
    assert len(cfg) == len(
        training.RUN_CONFIG_FIELDS
    ), "an allow-listed field was dropped"
    _write_params(d, {"definitely_not_allow_listed": 1})
    assert training.run_config(sid) == {}, "the allow-list is not filtering"
