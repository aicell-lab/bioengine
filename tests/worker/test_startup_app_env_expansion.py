"""Env-var expansion in worker startup-application config.

Startup apps may reference the worker's own environment (e.g. a secret
mounted via ``valueFrom.secretKeyRef``) as ``${VAR}``; the worker expands
these before deploying, so the secret value never has to live in
git-tracked helm values.
"""

from bioengine.worker.__main__ import (
    _expand_env_in_config,
    read_startup_applications,
)


def test_expand_env_recurses_and_preserves_types(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "s3kret")
    config = {
        "artifact_id": "bioimage-io/model-runner",
        "application_env_vars": {"*": {"_HF_READ_TOKEN": "${SECRET_TOKEN}"}},
        "nested": [{"k": "$SECRET_TOKEN"}],
        "disable_gpu": False,
        "num_gpus": 1,
    }

    expanded = _expand_env_in_config(config)

    assert expanded["application_env_vars"]["*"]["_HF_READ_TOKEN"] == "s3kret"
    assert expanded["nested"][0]["k"] == "s3kret"
    # Non-string values pass through untouched.
    assert expanded["disable_gpu"] is False
    assert expanded["num_gpus"] == 1


def test_expand_env_leaves_unset_vars_verbatim(monkeypatch):
    monkeypatch.delenv("DEFINITELY_UNSET_VAR", raising=False)
    assert _expand_env_in_config("${DEFINITELY_UNSET_VAR}") == "${DEFINITELY_UNSET_VAR}"


def test_read_startup_applications_expands_env(monkeypatch):
    monkeypatch.setenv("HF_READ_TOKEN", "hf_fromsecret")
    group_configs = {
        "Core Options": {
            "startup_applications": [
                '{"artifact_id": "bioimage-io/model-runner", '
                '"application_env_vars": {"*": {"_HF_READ_TOKEN": "${HF_READ_TOKEN}"}}}'
            ]
        }
    }

    result = read_startup_applications(group_configs)

    apps = result["Core Options"]["startup_applications"]
    assert isinstance(apps, list) and len(apps) == 1
    assert apps[0]["application_env_vars"]["*"]["_HF_READ_TOKEN"] == "hf_fromsecret"


def test_read_startup_applications_noop_without_apps():
    group_configs = {"Core Options": {}}
    assert read_startup_applications(group_configs) == group_configs
