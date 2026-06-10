"""Unit tests for ``bioengine._app.bootstrap``.

Exercises ``introspect_app`` and ``validate_kwargs_against_spec`` against
the synthetic packages under ``tests/_app/fixtures/``. These don't require
a Ray cluster — the bootstrap module is pure Python except for the actual
``cls.bind(...)`` step in ``build_application``, which is covered by the
integration test in ``tests/end_to_end/`` once a cluster is available.
"""

import pytest

from bioengine._app.bootstrap import (
    SPEC_FORMAT_VERSION,
    introspect_app,
    validate_kwargs_against_spec,
)
from bioengine._app.errors import BioEngineUserError, CompositionCycleError


# ─────────────────────────── single-deployment ───────────────────────────


def test_introspect_single_app():
    spec = introspect_app("tests._app.fixtures.single_app.deployment:FixtureApp")
    assert spec["format_version"] == SPEC_FORMAT_VERSION
    assert spec["entry_id"].endswith(":FixtureApp")

    classes = spec["classes"]
    assert len(classes) == 1

    fixture_id = next(iter(classes))
    meta = classes[fixture_id]
    assert meta["qualname"] == "FixtureApp"
    assert meta["ray_actor_options"]["num_cpus"] == 1
    assert meta["ray_actor_options"]["memory"] == 128 * 1024 * 1024
    method_names = {s["name"] for s in meta["method_schemas"]}
    assert method_names == {"hello", "echo"}
    assert meta["lifecycle_methods"]["async_init"] == "setup"
    assert meta["lifecycle_methods"]["smoke_test"] == "_smoke"
    assert meta["lifecycle_methods"]["health_check"] is None
    assert meta["init_params"] == [
        {
            "name": "greeting",
            "kind": "value",
            "annotation": "str",
            "default": "hi",
            "required": False,
        }
    ]


def test_introspect_missing_module_raises_user_error():
    with pytest.raises(BioEngineUserError, match="Cannot import"):
        introspect_app("not_a_real_module.deployment:App")


def test_introspect_missing_class_raises_user_error():
    with pytest.raises(BioEngineUserError, match="has no attribute"):
        introspect_app(
            "tests._app.fixtures.single_app.deployment:NotARealClass"
        )


def test_introspect_undecorated_class_raises():
    # Use a plain stdlib class as the target.
    with pytest.raises(BioEngineUserError, match="not decorated with @bioengine.app"):
        introspect_app("collections:OrderedDict")


# ─────────────────────────── composition graph ───────────────────────────


def test_introspect_composition_walks_graph():
    spec = introspect_app("tests._app.fixtures.composition_app.entry:Entry")
    classes = spec["classes"]
    qualnames = {meta["qualname"] for meta in classes.values()}
    assert qualnames == {"Entry", "RuntimeA", "RuntimeB"}

    entry_meta = classes[spec["entry_id"]]
    # init_params split into deployment-handle and value
    by_name = {p["name"]: p for p in entry_meta["init_params"]}
    assert by_name["runtime_a"]["kind"] == "deployment_handle"
    assert by_name["runtime_a"]["target"].endswith(":RuntimeA")
    assert by_name["runtime_b"]["kind"] == "deployment_handle"
    assert by_name["runtime_b"]["target"].endswith(":RuntimeB")
    assert by_name["label"]["kind"] == "value"
    assert by_name["label"]["default"] == "demo"


def test_introspect_method_schemas_per_class():
    spec = introspect_app("tests._app.fixtures.composition_app.entry:Entry")
    by_cid = {
        meta["qualname"]: {s["name"] for s in meta["method_schemas"]}
        for meta in spec["classes"].values()
    }
    assert by_cid["RuntimeA"] == {"shout"}
    assert by_cid["RuntimeB"] == {"add"}
    assert by_cid["Entry"] == {"shout", "add"}


# ──────────────────────────────── cycles ─────────────────────────────────


def test_cycle_detection(tmp_path, monkeypatch):
    """Build an on-the-fly two-class cycle and verify it's rejected."""
    import sys
    import textwrap

    pkg_root = tmp_path / "cycle_pkg"
    pkg_root.mkdir()
    (pkg_root / "__init__.py").write_text("")
    (pkg_root / "a.py").write_text(
        textwrap.dedent(
            """
            import bioengine
            from .b import B
            @bioengine.app(num_cpus=0)
            class A:
                def __init__(self, b: B):
                    self.b = b
            """
        )
    )
    (pkg_root / "b.py").write_text(
        textwrap.dedent(
            """
            import bioengine
            @bioengine.app(num_cpus=0)
            class B:
                def __init__(self, a: "A"):  # forward ref — kept as string
                    self.a = a
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    # B references A via a forward reference — typing.get_type_hints
    # would need A in scope. Make A available globally via a small shim:
    sys.modules.pop("cycle_pkg", None)
    sys.modules.pop("cycle_pkg.a", None)
    sys.modules.pop("cycle_pkg.b", None)

    import cycle_pkg.a as a_mod  # noqa: F401  # registers A

    # Now patch the forward ref into globals so get_type_hints resolves.
    import cycle_pkg.b as b_mod

    b_mod.A = a_mod.A  # type: ignore[attr-defined]

    with pytest.raises(CompositionCycleError):
        introspect_app("cycle_pkg.a:A")


# ────────────────────── validate_kwargs_against_spec ─────────────────────


def test_validate_kwargs_accepts_known_params():
    spec = introspect_app("tests._app.fixtures.single_app.deployment:FixtureApp")
    fixture_id = next(iter(spec["classes"]))
    validate_kwargs_against_spec(spec, {fixture_id: {"greeting": "yo"}})


def test_validate_kwargs_rejects_unknown_param():
    spec = introspect_app("tests._app.fixtures.single_app.deployment:FixtureApp")
    fixture_id = next(iter(spec["classes"]))
    with pytest.raises(BioEngineUserError, match="Unexpected init kwarg"):
        validate_kwargs_against_spec(spec, {fixture_id: {"typo": "x"}})


def test_validate_kwargs_no_required_for_default_only_init():
    """Single-app fixture has only optional params — empty kwargs is fine."""
    spec = introspect_app("tests._app.fixtures.single_app.deployment:FixtureApp")
    validate_kwargs_against_spec(spec, {})


def test_validate_kwargs_skips_deployment_handle_params():
    """User cannot pass DeploymentHandle kwargs — the bind graph fills them."""
    spec = introspect_app("tests._app.fixtures.composition_app.entry:Entry")
    # No kwargs at all — handles are filled by the bind graph, label has default.
    validate_kwargs_against_spec(spec, {})
