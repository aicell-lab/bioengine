"""Secrets are masked by the SHAPE OF THE VALUE, not only by the name.

The old rule hid a value if and only if its key began with an underscore. That
fails open for every spelling nobody anticipated: the same credential passed as
``HF_READ_TOKEN`` rather than ``_HF_READ_TOKEN`` was written to a shared log in
cleartext, and it outlived the pod that wrote it. The masked line and the
leaked line look equally deliberate, so nothing drew attention to the one that
leaked — it took six weeks and a credential rotation to notice.

Two directions have to hold at once, and the second is the one that decides
whether anybody keeps the feature:

* a credential-shaped value is masked whatever it is called — fail closed;
* an ordinary value is NOT masked, however long or random it looks. Image
  digests, commit SHAs and artifact ids are read constantly, and an operator
  who cannot read their own logs routes around the redaction entirely.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from bioengine.apps.manager import AppsManager
from bioengine.utils.redaction import (
    SECRET_MASK,
    is_secret_env_var,
    looks_like_secret,
)

# Shapes that are credentials and essentially nothing else. The bodies are
# long enough to be realistic; the matcher requires a substantial body so a
# short lookalike does not trip it.
SECRET_SHAPED = {
    "huggingface": "hf_QRSTUVWXabcdefghijklmnopqrstuvwxyz12",
    "github pat": "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "github oauth": "gho_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "openai style": "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "jwt": (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkEifQ"
        ".dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXkw"
    ),
    "pem rsa": "-----BEGIN RSA PRIVATE KEY-----\nMIIEow...",
    "pem openssh": "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNz...",
    "token inside a url": "https://user:hf_QRSTUVWXabcdefghijklmnopqrstuvwxyz12@example.org/x",
}

# Things an operator needs to be able to read. None may be masked.
ORDINARY = {
    "image tag": "ghcr.io/aicell-lab/bioengine-worker:0.16.112",
    "commit sha": "277108ae1f3c9b2d4e5a6071829304abcdef1234",
    "artifact id": "bioimage-io/model-runner",
    "a long digest": "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "a plain word": "true",
    "a path": "/home/bioengine/apps/model-runner/source",
    "a short lookalike": "sk-1",
    "another short one": "hf_abc",
    "a url with no token": "https://hypha.aicell.io/ws/public/services",
}


def _manager() -> AppsManager:
    """An AppsManager with nothing built: the filter reads no instance state."""
    return object.__new__(AppsManager)


# --------------------------------------------------------------------------
# the matcher itself
# --------------------------------------------------------------------------


def test_credential_shapes_are_recognised_under_any_name():
    for label, value in SECRET_SHAPED.items():
        assert looks_like_secret(value), f"{label} not recognised as a secret"
        assert is_secret_env_var("HF_READ_TOKEN", value), label
        assert is_secret_env_var("ANYTHING_AT_ALL", value), label


def test_ordinary_values_are_NOT_masked():
    """CONTROL. Over-masking is the failure mode that gets redaction removed."""
    for label, value in ORDINARY.items():
        assert not looks_like_secret(value), f"{label} would be masked: {value!r}"
        assert not is_secret_env_var("SOME_NAME", value), label


def test_the_underscore_opt_in_still_wins_on_its_own():
    """CONTROL. The name rule is kept as an addition, not replaced."""
    assert is_secret_env_var("_ANY_NAME", "a value nothing would ever match")
    assert is_secret_env_var("_X", "")


def test_non_strings_are_never_secret_shaped():
    """Coercing them to text to scan would only invent matches."""
    for value in (None, 0, 1, True, False, 3.14, [], {}, object()):
        assert not looks_like_secret(value)
    assert not is_secret_env_var("COUNT", 42)
    # ...but the explicit opt-in still applies to them.
    assert is_secret_env_var("_COUNT", 42)


# --------------------------------------------------------------------------
# the status payload
# --------------------------------------------------------------------------


def test_a_secret_under_an_ORDINARY_name_is_masked_in_status():
    """THE DEFECT: this value used to be returned verbatim."""
    out = _manager()._filter_secret_env_vars(
        {"Entry": {"HF_READ_TOKEN": SECRET_SHAPED["huggingface"]}}
    )
    assert out["Entry"]["HF_READ_TOKEN"] == SECRET_MASK


def test_the_underscore_form_is_still_masked_AND_still_renamed_in_status():
    """CONTROL: the pre-existing contract is unchanged."""
    out = _manager()._filter_secret_env_vars(
        {"Entry": {"_HF_READ_TOKEN": SECRET_SHAPED["huggingface"]}}
    )
    assert out["Entry"] == {"HF_READ_TOKEN": SECRET_MASK}


def test_an_unprefixed_secret_keeps_its_KEY_in_status():
    """There is no prefix to strip, and renaming it would make the status
    report disagree with what the caller actually passed."""
    out = _manager()._filter_secret_env_vars(
        {"Entry": {"TOKEN": SECRET_SHAPED["jwt"]}}
    )
    assert list(out["Entry"]) == ["TOKEN"]
    assert out["Entry"]["TOKEN"] == SECRET_MASK


def test_ordinary_status_values_survive_untouched():
    env = {"Entry": dict(ORDINARY)}
    out = _manager()._filter_secret_env_vars(env)
    assert out["Entry"] == dict(sorted(ORDINARY.items()))


# --------------------------------------------------------------------------
# the deploy log line
# --------------------------------------------------------------------------


def test_the_deploy_log_decides_by_VALUE_and_not_by_name_alone():
    """TRIPWIRE on the call site — now that the real guard is behavioural.

    This used to assert that `deploy_app` MENTIONED `is_secret_env_var` somewhere in its
    source, because the log line was built inline inside a long method and no behavioural
    test could reach it. That was not a guard: deleting the matcher call from the CLEARTEXT
    half of the line prints every secret in clear and leaves this assertion perfectly happy
    (reviewer, round 1 — measured, 9 passed on that mutant).

    The construction now lives in `describe_deploy_arguments`, which the tests below render
    and inspect, so the real property is checked on the real output. What remains worth
    tripwiring is only the WIRING: that `deploy_app` still delegates rather than growing a
    second, unmasked copy of the line inline, and that it has not gone back to gating on
    `key.startswith("_")`, the expression that failed open.
    """
    source = textwrap.dedent(inspect.getsource(AppsManager.deploy_app))
    tree = ast.parse(source)

    delegates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "describe_deploy_arguments"
    ]
    assert delegates, (
        "the deploy log no longer calls describe_deploy_arguments; if the line is being "
        "built inline again it is not covered by the rendered-output tests below"
    )

    startswith_underscore = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "startswith"
        and any(
            isinstance(a, ast.Constant) and a.value == "_" for a in node.args
        )
    ]
    assert not startswith_underscore, (
        "the deploy log gates on key.startswith('_') again — that is the rule "
        "that failed open, and it must live behind is_secret_env_var now"
    )


# ── The deploy log line, rendered ─────────────────────────────────────────────
#
# The test that stood in for this before asserted only that `deploy_app` MENTIONED the
# matcher somewhere in its source and contained no `startswith('_')`. The reviewer deleted
# the guard from the CLEARTEXT half of the line, which makes every secret print in clear,
# and the suite stayed green. These assert on the rendered STRING instead.

from bioengine.utils.redaction import SECRET_MASK, describe_deploy_arguments

HF_SECRET = "hf_" + "A" * 30
JWT_SECRET = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r"
PAT_SECRET = "github_pat_" + "B" * 30


def test_an_unprefixed_secret_value_never_reaches_the_rendered_env_vars():
    """The defect in one line: the same credential under a name nobody anticipated."""
    _, env_vars_str = describe_deploy_arguments(
        {}, {"App": {"HF_READ_TOKEN": HF_SECRET, "LOG_LEVEL": "INFO"}}
    )
    assert HF_SECRET not in env_vars_str, env_vars_str
    assert SECRET_MASK in env_vars_str, env_vars_str
    # the non-secret must still be readable, or operators route around the redaction
    assert "LOG_LEVEL" in env_vars_str and "INFO" in env_vars_str, env_vars_str


def test_the_underscore_control_still_masks_whatever_the_value_looks_like():
    """The name rule is an ADDITION, not a replacement: it must still work alone."""
    _, env_vars_str = describe_deploy_arguments(
        {}, {"App": {"_OPAQUE": "not-credential-shaped-at-all"}}
    )
    assert "not-credential-shaped-at-all" not in env_vars_str, env_vars_str
    assert SECRET_MASK in env_vars_str, env_vars_str


def test_a_secret_passed_as_a_KWARG_is_masked_too():
    """kwargs and env_vars share the same log line.

    Masking one and printing the other in clear is the identical failure, one argument
    along -- and this issue's own earlier sweep named it the worst remaining gap.
    """
    kwargs_str, _ = describe_deploy_arguments(
        {"App": {"hf_token": HF_SECRET, "batch_size": 4}}, {}
    )
    assert HF_SECRET not in kwargs_str, kwargs_str
    assert SECRET_MASK in kwargs_str, kwargs_str
    assert "batch_size=4" in kwargs_str, kwargs_str


def test_a_jwt_in_a_kwarg_or_env_var_is_masked_on_both_sides():
    kwargs_str, env_vars_str = describe_deploy_arguments(
        {"App": {"token": JWT_SECRET}}, {"App": {"HYPHA_TOKEN": JWT_SECRET}}
    )
    assert JWT_SECRET not in kwargs_str, kwargs_str
    assert JWT_SECRET not in env_vars_str, env_vars_str


def test_a_fine_grained_github_pat_is_recognised():
    """`github_pat_` does not match the two-letter `gh[pousr]_` family, so it was missed."""
    _, env_vars_str = describe_deploy_arguments({}, {"App": {"GH": PAT_SECRET}})
    assert PAT_SECRET not in env_vars_str, env_vars_str
    assert SECRET_MASK in env_vars_str, env_vars_str


def test_empty_arguments_render_as_None_not_as_an_empty_string():
    """Unchanged behaviour, pinned so the extraction cannot quietly alter the line."""
    assert describe_deploy_arguments({}, {}) == ("None", "None")
