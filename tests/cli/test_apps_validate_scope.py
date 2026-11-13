"""``apps validate`` must say which half of the ``id`` it checked (myco #0447)."""
import ast
import inspect
import textwrap

import yaml
from click.testing import CliRunner

from bioengine.cli.apps import apps_group, validate

MANIFEST = {
    "name": "Scope probe",
    "id": "scope-probe",
    "id_emoji": "🧪",
    "description": "Fixture for the validate-scope wording test.",
    "format_version": "0.6.0",
    "type": "ray-serve",
    "entry": "deployment:Probe",
}


def _run(tmp_path, manifest=None):
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(manifest or MANIFEST))
    return CliRunner().invoke(apps_group, ["validate", str(tmp_path)])


def test_the_green_verdict_names_the_id_and_what_was_not_checked(tmp_path):
    result = _run(tmp_path)
    assert result.exit_code == 0, result.output
    assert "is a valid v0.6.0 BioEngine app manifest" in result.output
    assert "NOT checked" in result.output
    assert "'scope-probe'" in result.output
    assert "free in the target workspace" in result.output


def test_the_green_verdict_names_the_next_step(tmp_path):
    assert "bioengine apps upload" in _run(tmp_path).output


def test_an_invalid_manifest_still_fails_and_claims_nothing(tmp_path):
    broken = {k: v for k, v in MANIFEST.items() if k != "entry"}
    result = _run(tmp_path, broken)
    assert result.exit_code == 1
    assert "NOT checked" not in result.output
    assert "is a valid v0.6.0" not in result.output


def test_validate_opens_no_connection():
    """Clause 5: reaching the server is the feature request held on #0334, not this.

    Structural rather than mocked — a mock only proves the paths it patched were
    not taken, while an AST walk fails on a call added anywhere in the body.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(validate.callback)))
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Await, ast.AsyncWith))]
    names = {
        getattr(n.func, "id", None) or getattr(n.func, "attr", None)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    assert not names & {"connect", "connect_to_server", "require_worker", "run_async"}
