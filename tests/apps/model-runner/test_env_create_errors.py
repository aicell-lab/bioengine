"""Unit tests for the conda-env-create error translation.

Loaded by AST extraction rather than import: ``entry.py`` pulls in bioengine
and ray at module scope, which are not present in a plain test environment.
Extracting the function from the shipped source keeps the test honest — it
exercises the code that ships, not a copy of it.
"""

import ast
import textwrap
from pathlib import Path
from typing import Optional

import pytest

ENTRY = Path(__file__).resolve().parents[3] / "apps" / "model-runner" / "entry.py"

REAL_STDERR = (
    "Pip subprocess error:\n"
    "  Running command git clone --filter=blob:none --quiet "
    "https://github.com/facebookresearch/sam3.git /tmp/pip-req-build-x\n"
    "  fatal: could not read Username for 'https://github.com': "
    "terminal prompts disabled\n"
    "  error: subprocess-exited-with-error\n"
)


@pytest.fixture(scope="module")
def explain():
    src = ENTRY.read_text()
    tree = ast.parse(src)
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "EntryDeployment"
    )
    fn = next(
        (
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_explain_env_create_failure"
        ),
        None,
    )
    if fn is None:
        pytest.fail("_explain_env_create_failure missing from entry.py")
    seg = textwrap.dedent(ast.get_source_segment(src, fn)).replace(
        "@staticmethod\n", "", 1
    )
    ns = {"Optional": Optional}
    exec(seg, ns)
    return ns["_explain_env_create_failure"]


def test_translates_git_https_failure(explain):
    out = explain(REAL_STDERR)
    assert out is not None
    assert "github.com" in out
    # States only what the signature proves: the remote answered and asked
    # for credentials. It must NOT claim the host was unreachable, which this
    # message rules out, nor that authentication is irrelevant, which is the
    # private-or-misspelled-repo case.
    assert "demanded credentials" in out
    assert "unreachable" not in out
    assert "private" in out and "intercepts" in out
    assert "archive/" in out and "sha256=" in out
    # Not an exhaustiveness claim: a public parent with a private submodule
    # produces this signature while the named repo is neither.
    assert "Two causes" not in out
    assert "submodules" in out


def test_offers_the_archive_form_only_where_that_url_shape_exists(explain):
    """GitLab and Bitbucket use different archive paths, so a github-shaped
    example interpolated with their host would send the user to a 404."""
    gitlab = REAL_STDERR.replace("github.com", "gitlab.com")
    out = explain(gitlab)
    assert out is not None and "gitlab.com" in out
    assert "/archive/" not in out
    # The example is conditional, so the punctuation must be too — otherwise
    # the sentence promises an example and delivers an empty line. Asserting
    # only the URL's absence passes on that broken output.
    assert out.rstrip().endswith("."), repr(out[-60:])


def test_fires_when_git_line_is_outside_the_last_kilobyte(explain):
    noisy = REAL_STDERR + ("libmamba trace padding ...........\n" * 80)
    assert "could not read Username" not in noisy[-1000:]
    assert explain(noisy) is not None


def test_call_site_passes_the_whole_stderr_not_the_display_tail():
    """Asserted against the SOURCE, because the function cannot see this.

    Exercising the pure function proves nothing about what the caller hands
    it: swapping the call to ``text[-1000:]`` leaves every behavioural test
    green while the translation silently stops firing on noisy builds, which
    is the only case that matters.
    """
    tree = ast.parse(ENTRY.read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_explain_env_create_failure"
    ]
    assert calls, "nothing calls _explain_env_create_failure"
    for call in calls:
        arg = call.args[0]
        assert isinstance(arg, ast.Name), (
            "argument must be the undecorated stderr, got "
            f"{ast.dump(arg)[:80]}"
        )


def test_silent_on_unrelated_solver_failure(explain):
    assert explain("libmamba Could not solve for environment specs\n") is None


def test_requires_a_git_clone_not_just_an_auth_prompt(explain):
    assert (
        explain("fatal: could not read Username for 'https://example.com'\n")
        is None
    )
