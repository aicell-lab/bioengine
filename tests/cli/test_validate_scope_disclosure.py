"""A green ``bioengine apps validate`` has to say which half it checked (#0447).

Two app authors independently read ``OK: …/manifest.yaml is a valid v0.6.0
BioEngine app manifest.`` as clearance to upload, and the upload was then
refused over the manifest's ``id``. The success line is literally true — it
says *manifest*, not *app* — but ``id`` is a REQUIRED manifest field that the
validator inspects and approves, so "your manifest is valid" reasonably reads
as covering the one field the upload refuses on. The halves differ: structure
and required fields are local, the id's availability in the target workspace is
not, and this command opens no connection at all.

So the assertions below are on the output naming the id VALUE and naming the
limit. A test that only checked "exit 0" or "printed OK" passes against the
pre-fix single line, which is the whole defect.

What is deliberately NOT done here: making ``validate`` consult the server. That
is a behaviour change under ``bioengine/**`` and a held feature request, not
this fix — ``test_validate_opens_no_connection`` is the fence, and it is written
to fail if a later change reaches for the network from this command.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from click.testing import CliRunner

# Imported by full path: the ``bioengine`` package resolves its authoring
# symbols through a PEP 562 ``__getattr__``, so a submodule has to be named.
import bioengine.cli.apps as apps_cli

MANIFEST = """\
name: {name}
id: {app_id}
id_emoji: '🔬'
description: A test app.
type: ray-serve
format_version: 0.6.0
version: 1.0.0
entry: {package}.deployment:TestApp
"""


def _app_dir(tmp_path: Path, app_id: str, package: str = "test_app") -> Path:
    app = tmp_path / app_id
    (app / package).mkdir(parents=True)
    (app / "manifest.yaml").write_text(
        MANIFEST.format(name=f"App {app_id}", app_id=app_id, package=package),
        encoding="utf-8",
    )
    (app / package / "__init__.py").write_text("", encoding="utf-8")
    (app / package / "deployment.py").write_text("", encoding="utf-8")
    return app


def _validate(app_dir: Path, *extra):
    return CliRunner().invoke(
        apps_cli.apps_group,
        ["validate", str(app_dir), *extra],
        env={"HYPHA_TOKEN": "", "BIOENGINE_TOKEN": "", "BIOENGINE_SERVER_URL": ""},
        catch_exceptions=False,
    )


# ── The green verdict says which half it checked ──────────────────────────────


def test_a_green_validate_names_the_id_it_did_not_check(tmp_path):
    """THE DISCRIMINATION CONTROL FOR THIS FILE.

    The pre-fix command printed the ``OK:`` line and nothing else, so it passes
    the first assertion and fails every one after it. Restore that single line
    and this test goes red.
    """
    result = _validate(_app_dir(tmp_path, "my-app"))

    assert result.exit_code == 0, result.output
    assert "is a valid v0.6.0 BioEngine app manifest" in result.output

    # The id, by value — not a generic "some fields are not checked".
    assert "'my-app'" in result.output
    assert "NOT checked" in result.output


def test_the_id_in_the_output_is_the_id_in_the_manifest(tmp_path):
    """A hard-coded id, or a message that names the directory instead, passes
    the test above. Two apps with different ids do not agree by accident."""
    first = _validate(_app_dir(tmp_path, "alpha-app")).output
    second = _validate(_app_dir(tmp_path, "beta-app")).output

    assert "'alpha-app'" in first and "'beta-app'" not in first
    assert "'beta-app'" in second and "'alpha-app'" not in second


def test_the_output_names_the_readers_next_step(tmp_path):
    """Clause 2: the remedy belongs in the output, not only in the skill."""
    output = _validate(_app_dir(tmp_path, "my-app")).output

    assert "bioengine apps list" in output


def test_the_output_explains_what_an_id_already_in_use_costs(tmp_path):
    """Naming the field without naming the consequence sends the reader back to
    the docs. The refusal they hit is version monotonicity against the lineage
    that id already has, so the output has to say that much."""
    output = _validate(_app_dir(tmp_path, "my-app")).output.lower()

    assert "upload" in output
    assert "version" in output


def test_an_invalid_manifest_does_not_advertise_the_scope(tmp_path):
    """The disclosure is part of a GREEN verdict. Printing it on a red one
    would bury the actual validation error under advice about a later step."""
    app = _app_dir(tmp_path, "my-app")
    (app / "manifest.yaml").write_text("format_version: 0.6.0\n", encoding="utf-8")

    result = _validate(app)

    assert result.exit_code != 0
    assert "NOT checked" not in result.output
    assert "bioengine apps list" not in result.output


# ── Clause 5: it stays offline ────────────────────────────────────────────────


def test_validate_opens_no_connection(tmp_path, monkeypatch):
    """The fence around the held feature request.

    Making ``validate`` ask the server whether the id is free is a real
    behaviour change and is not this row's work. Both connection helpers are
    replaced with doubles that fail the test if touched, so a later change that
    reaches for the network from here is caught by this file rather than in
    review.
    """
    monkeypatch.setattr(
        apps_cli,
        "connect_worker",
        AsyncMock(side_effect=AssertionError("validate connected to a worker")),
    )
    monkeypatch.setattr(
        apps_cli,
        "require_worker",
        MagicMock(side_effect=AssertionError("validate required worker credentials")),
    )

    result = _validate(_app_dir(tmp_path, "my-app"))

    assert result.exit_code == 0, result.output


def test_validate_takes_no_worker_options(tmp_path):
    """The other half of staying offline: it must not grow ``--worker`` /
    ``--token``. A command that accepts them is one edit from using them."""
    result = _validate(_app_dir(tmp_path, "my-app"), "--worker", "ws/bioengine-worker")

    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


def test_the_source_still_carries_the_disclosure():
    """What stops the output assertions above being satisfiable by accident.

    Reads the shipping command rather than its output: the disclosure has to be
    a second write that interpolates the manifest's own ``id``, so a future
    tidy-up that folds it back into one static ``OK:`` line fails here too.
    """
    tree = ast.parse(Path(apps_cli.__file__).read_text(encoding="utf-8"))
    fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "validate"
        ),
        None,
    )
    assert fn is not None, "no validate command found — update this test"

    echoes = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "echo"
    ]
    assert len(echoes) >= 2, ast.dump(fn)

    # The manifest returned by the validator is read, not discarded — that is
    # what makes the id in the message the manifest's own.
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_validate_manifest_or_exit"
        for n in ast.walk(fn)
    )
    assigned = [
        t.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    ]
    assert "manifest" in assigned, assigned


# ── The premise, pinned ───────────────────────────────────────────────────────


def test_id_is_a_required_manifest_field_the_validator_approves(tmp_path):
    """Why the green verdict reads as covering the id at all.

    If ``id`` were a field the validator ignored, the fix would be to say
    "validate does not look at ids". It looks: a manifest without one is
    refused, so a green verdict is a statement about the id that was there.
    """
    app = _app_dir(tmp_path, "my-app")
    text = (app / "manifest.yaml").read_text(encoding="utf-8")
    (app / "manifest.yaml").write_text(
        "\n".join(l for l in text.splitlines() if not l.startswith("id:")) + "\n",
        encoding="utf-8",
    )

    result = _validate(app)

    assert result.exit_code != 0
    assert "id" in result.output
