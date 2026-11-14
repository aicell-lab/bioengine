"""A custom env must pin pydantic, or its pinned core/spec is not a reproducible harness.

``bioimageio.spec`` 0.5.12.0 — the version ``bioimageio.core==0.11.0`` requires exactly —
calls the pydantic wrap handler as ``handler(value, info)`` in
``_internal/io_packaging.py:package_file_descr_serializer``. The handler's second
positional parameter is ``index_key``, not ``info``; the call site even carries a
``pyright: ignore[reportArgumentType]``. pydantic tolerated that through 2.13.x and
validates it from 2.14.0, so format validation of any description carrying a packaged
``FileDescr`` dies with::

    Error calling function `package_file_descr_serializer`:
    TypeError: 'index_key' is expected to be an integer or a string, got
    'SerializationInfo(include=None, ..., mode='json', ..., exclude_unset=True, ...)'

MEASURED, one variable at a time (myco #0497). On the live KTH model-runner a
``custom_environment=True`` test of ``faithful-chicken`` returned
``status='valid-format'`` with no "Reproduce test outputs" detail at all — the conda env
had solved python 3.14.10 / pydantic 2.14.3. The SAME ``bioimageio.spec 0.5.12.0`` and
``bioimageio.core 0.11.0`` pass on europa under python 3.11.17 / pydantic 2.12.0, so the
variable is pydantic and not python: holding python at 3.11 in
``ghcr.io/aicell-lab/model-runner:2.11.2`` and moving only pydantic gives
2.12.5 OK, 2.13.0 OK, 2.13.5 OK, **2.14.0 raises**, 2.14.3 raises — the 2.14.3 arm
reproducing the live error text verbatim.

The standard runtime never hit this by luck, not by design: ``careamics==0.0.16``
requires ``pydantic<=2.12,>=2.11`` and caps it. A custom env has no careamics and
declares only what the model asked for, so nothing held pydantic down.

Upstream fixed the call site in ``bioimageio.spec`` 0.5.14.3 (``handler(value)``; 0.5.12.0,
0.5.13.0, 0.5.14.0, 0.5.14.1 and 0.5.14.2 are all still affected — checked by reading
``io_packaging.py`` out of each wheel). That fix is unreachable from here:
``bioimageio.core==0.11.0`` pins ``bioimageio.spec==0.5.12.0`` exactly and 0.11.0 is the
newest core release, so the pin below is the available fix, not the preferred one.

The functions under test are pulled out of ``runtime.py`` with the AST loader this
directory already uses, so none of bioengine, ray or bioimageio is imported here. The one
thing that cannot be loaded that way is the ``BioimageioCondaEnv`` round-trip at the end
of ``pin_bioimageio_conda_env``, so a stub stands in for it and the assertions are made
against the dependency list the function builds.
"""

import ast
import pathlib
import sys
import types

import pytest

APP = pathlib.Path(__file__).resolve().parents[3] / "apps" / "model-runner"
RUNTIME = APP / "runtime.py"
ENTRY = APP / "entry.py"


def _load(path, *names):
    """Exec just the named top-level defs/assignments out of ``path``."""
    tree = ast.parse(path.read_text())
    wanted = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        )
    ]
    found = {n.name if hasattr(n, "name") else n.targets[0].id for n in wanted}
    if found != set(names):
        pytest.fail(f"missing {sorted(set(names) - found)} in {path.name}")
    ns = {"List": __import__("typing").List}
    exec(  # noqa: S102 — the house pattern in this directory
        compile(ast.Module(body=wanted, type_ignores=[]), str(path), "exec"), ns
    )
    return ns


@pytest.fixture(scope="module")
def ns():
    return _load(
        RUNTIME,
        "CUSTOM_ENV_PYDANTIC_PIN",
        "_is_pydantic_req",
        "pin_bioimageio_conda_env",
    )


@pytest.fixture(autouse=True)
def _stub_bioimageio_conda_env():
    """Stand in for ``bioimageio.spec.conda_env.BioimageioCondaEnv``.

    ``model_validate`` returns the dict unchanged so the test can read the dependency
    list the function actually assembled. Only the modules this import needs are
    inserted, and they are removed afterwards so nothing leaks into another test.
    """

    class _Stub:
        @staticmethod
        def model_validate(data):
            return data

    added = []
    for name, attrs in (
        ("bioimageio", {}),
        ("bioimageio.spec", {}),
        ("bioimageio.spec.conda_env", {"BioimageioCondaEnv": _Stub}),
    ):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[name] = mod
            added.append(name)
    yield
    for name in reversed(added):
        del sys.modules[name]


class _FakeEnv:
    """The smallest thing ``pin_bioimageio_conda_env`` accepts."""

    def __init__(self, dependencies, name="sam"):
        self._data = {"name": name, "channels": ["conda-forge"], "dependencies": dependencies}

    def model_dump(self, mode="python"):
        import copy

        return copy.deepcopy(self._data)


def _pip_of(result):
    """Return the pip list out of the dependency list the function built."""
    for dep in result["dependencies"]:
        if isinstance(dep, dict) and "pip" in dep:
            return dep["pip"]
    pytest.fail(f"no pip section in {result['dependencies']!r}")


def _pin(ns, dependencies):
    return ns["pin_bioimageio_conda_env"](_FakeEnv(dependencies), "0.11.0", "0.5.12.0")


# --------------------------------------------------------------------------- the fix


def test_a_pydantic_bound_is_added_even_when_the_model_never_mentions_it(ns):
    """``faithful-chicken``'s own environment.yaml is literally ``- micro_sam``.

    Nothing in it constrains pydantic, which is exactly how the env reached 2.14.3.
    """
    result = _pin(ns, ["micro_sam"])
    pip = _pip_of(result)
    assert ns["CUSTOM_ENV_PYDANTIC_PIN"] in pip
    # and the existing pins are still there — this must not displace them
    assert "bioimageio.core==0.11.0" in pip
    assert "bioimageio.spec==0.5.12.0" in pip
    # the model's own dependency survives
    assert "micro_sam" in result["dependencies"]


def test_the_bound_excludes_the_first_version_measured_to_break(ns):
    """2.13.5 measured OK, 2.14.0 measured raising — the bound must sit between them."""
    spec = ns["CUSTOM_ENV_PYDANTIC_PIN"]
    packaging = pytest.importorskip("packaging.specifiers")
    name, _, constraint = spec.partition("<")
    assert name.strip() == "pydantic", spec
    s = packaging.SpecifierSet(spec[len("pydantic") :])
    assert s.contains("2.12.0"), f"{spec} must allow the version the standard runtime uses"
    assert s.contains("2.13.5"), f"{spec} must allow 2.13.5, measured OK"
    assert not s.contains("2.14.0"), f"{spec} must exclude 2.14.0, measured to raise"
    assert not s.contains("2.14.3"), f"{spec} must exclude 2.14.3, the version the env solved"


def test_a_conda_level_pydantic_is_replaced_not_left_to_be_solved(ns):
    """A conda ``pydantic`` would be solved by conda and would defeat a pip bound."""
    result = _pin(ns, ["micro_sam", "pydantic", "conda-forge::pydantic=2.14.3"])
    deps = [d for d in result["dependencies"] if isinstance(d, str)]
    assert deps == ["micro_sam"], deps
    assert ns["CUSTOM_ENV_PYDANTIC_PIN"] in _pip_of(result)


def test_a_pip_level_pydantic_from_the_model_is_replaced(ns):
    result = _pin(ns, ["micro_sam", {"pip": ["pydantic==2.14.3", "timm"]}])
    pip = _pip_of(result)
    assert "pydantic==2.14.3" not in pip
    assert ns["CUSTOM_ENV_PYDANTIC_PIN"] in pip
    assert "timm" in pip, "the model's other pip deps must survive"


def test_exactly_one_pydantic_requirement_ends_up_in_the_pip_section(ns):
    """Two pydantic lines would let whichever pip reads last decide."""
    result = _pin(ns, [{"pip": ["pydantic>=2.11", "pydantic==2.14.3"]}])
    pip = _pip_of(result)
    assert sum(1 for p in pip if ns["_is_pydantic_req"](p)) == 1, pip


# ------------------------------------------------------- precision of the name match


@pytest.mark.parametrize(
    "dep",
    [
        "pydantic",
        "pydantic==2.14.3",
        "pydantic<2.14",
        "pydantic>=2.11,<2.14",
        "pydantic 2.14.3",
        "conda-forge::pydantic",
        "  pydantic  ",
        "pydantic[email]",
    ],
)
def test_is_pydantic_req_matches_the_distribution_itself(ns, dep):
    assert ns["_is_pydantic_req"](dep) is True, dep


@pytest.mark.parametrize(
    "dep",
    [
        "pydantic-core",
        "pydantic_core==2.50.4",
        "pydantic-settings<3,>=2.5",
        "pydantic-extra-types",
        "micro_sam",
        "timm",
        "",
        None,
        {"pip": ["pydantic"]},
    ],
)
def test_is_pydantic_req_leaves_everything_else_alone(ns, dep):
    """``pydantic-core`` follows whatever pydantic is chosen; do not fight the model for it."""
    assert ns["_is_pydantic_req"](dep) is False, dep


def test_pydantic_core_survives_the_pin(ns):
    """The precision guard, end to end rather than on the predicate alone."""
    result = _pin(ns, [{"pip": ["pydantic-core==2.50.4", "pydantic==2.14.3"]}])
    pip = _pip_of(result)
    assert "pydantic-core==2.50.4" in pip
    assert "pydantic==2.14.3" not in pip


# ------------------------------------------- the two call sites must still agree on the hash


def test_the_pydantic_bound_is_a_literal_not_a_derived_version(ns):
    """``version("pydantic")`` here would break pre-built env reuse.

    The EntryApp and the RuntimeApp run in different deployments with different
    requirement files, and both compute the env-name hash. Measured 2026-11-14:
    ``requirements-runtime.txt`` resolves pydantic 2.12.0 (careamics caps it) while
    ``requirements-entry.txt`` resolves 2.14.3. A pin derived from the running
    interpreter would therefore differ between the two, the hashes would not match, and
    every custom-env test would pay the cold build the pre-build exists to avoid.

    **Named for what it asserts, which is one of three inputs to that hash** (myco
    #0529). ``pin_bioimageio_conda_env`` appends ``bioimageio.core==``,
    ``bioimageio.spec==`` and this bound; only the third is a literal, and the other
    two are parameters each call site fills from its OWN interpreter
    (``runtime.py`` via ``version(...)``, ``entry.py`` via
    ``_get_bioimageio_versions``). The test below covers those two. This one used to
    be called ``test_the_bound_is_a_literal_so_both_deployments_hash_the_same``,
    which promised the whole conjunction while checking a third of it.
    """
    assert isinstance(ns["CUSTOM_ENV_PYDANTIC_PIN"], str)
    tree = ast.parse(RUNTIME.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CUSTOM_ENV_PYDANTIC_PIN"
            for t in node.targets
        ):
            assert isinstance(node.value, ast.Constant), (
                "the bound must be a plain literal, not a computed value"
            )
            break
    else:  # pragma: no cover - the ns fixture would have failed first
        pytest.fail("CUSTOM_ENV_PYDANTIC_PIN is not a module-level assignment")


def _version_pydantic_calls(path):
    """Every real ``version("pydantic")`` CALL in ``path``, ignoring prose.

    Matched on the AST rather than on the file text: this module's own commentary
    explains why the bound is not derived and so contains that call as a string,
    which a substring search cannot tell from the thing it forbids. (It did not tell
    them apart — this test failed on runtime.py's own docstring first.)
    """
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if name != "version":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and arg.value == "pydantic":
                found.append(getattr(node, "lineno", -1))
    return found


def test_neither_call_site_builds_its_own_pydantic_pin():
    """Both sides must get the bound through the one function, or the hashes diverge."""
    for path in (RUNTIME, ENTRY):
        calls = _version_pydantic_calls(path)
        assert calls == [], f"{path.name} derives a pydantic version at line(s) {calls}"

    entry_src = ENTRY.read_text()
    runtime_src = RUNTIME.read_text()
    assert "pin_bioimageio_conda_env(" in entry_src
    assert "pin_bioimageio_conda_env(" in runtime_src
    # the bound is assigned exactly once, in runtime.py, and entry.py never rebinds it
    assert _assignments_of(RUNTIME, "CUSTOM_ENV_PYDANTIC_PIN") == 1
    assert _assignments_of(ENTRY, "CUSTOM_ENV_PYDANTIC_PIN") == 0, (
        "entry.py must not restate the bound; it inherits it through the function"
    )


def _assignments_of(path, name):
    """Count real top-level-or-nested assignments to ``name``, ignoring prose."""
    n = 0
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign):
            n += sum(1 for t in node.targets if isinstance(t, ast.Name) and t.id == name)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                n += 1
    return n


def test_the_prose_guard_can_actually_fail():
    """Control for the AST matcher: it must find a call that IS there.

    Without this, ``_version_pydantic_calls`` returning [] would be indistinguishable
    from a matcher that never matches anything — the empty-grep failure mode.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(
            '"""A docstring mentioning version("pydantic") must NOT count."""\n'
            "from importlib.metadata import version\n"
            "# a comment with version(\"pydantic\") must NOT count either\n"
            'PIN = "pydantic<" + version("pydantic")\n'
        )
        tmp = pathlib.Path(fh.name)
    try:
        hits = _version_pydantic_calls(tmp)
        assert len(hits) == 1, f"expected exactly the line-4 call, got {hits}"
        assert hits == [4], hits
    finally:
        tmp.unlink()


# ---------------------------------------------------------------- discriminating control


def test_the_pin_is_what_puts_pydantic_in_the_pip_section(ns):
    """Control: with the bound removed from the assembled list, nothing else supplies it.

    Without this, every assertion above would still pass if some other line happened to
    mention pydantic, and the test would be attesting to the wrong thing.
    """
    pip = _pip_of(_pin(ns, ["micro_sam"]))
    others = [p for p in pip if ns["_is_pydantic_req"](p) and p != ns["CUSTOM_ENV_PYDANTIC_PIN"]]
    assert others == [], others
    assert [p for p in pip if ns["_is_pydantic_req"](p)] == [ns["CUSTOM_ENV_PYDANTIC_PIN"]]


# ------------------------------- the OTHER two inputs to the hash, and the pin's expiry
#
# myco #0529 and #0530. Everything here is derived from files in this repo and nothing
# is imported from bioimageio: CI installs only requirements-test.txt and
# requirements-datasets.txt into the worker image, so ``importlib.metadata`` cannot see
# bioimageio.core at all and a metadata-based guard would skip rather than check.

#: The requirements files of the two deployments that both call
#: ``pin_bioimageio_conda_env`` and both compute the env-name hash.
REQ_OF_DEPLOYMENT = {
    "entry": APP / "requirements-entry.txt",
    "runtime": APP / "requirements-runtime.txt",
}

#: The ``bioimageio.core`` release ``CUSTOM_ENV_PYDANTIC_PIN`` was measured against.
#: It pins ``bioimageio.spec==0.5.12.0`` exactly, and that spec is the one carrying the
#: serializer crash (myco #0497). Upstream fixed the call site in spec 0.5.14.3.
PIN_MEASURED_AGAINST_CORE = "0.11.0"


def _exact_pin(requirements_text, distribution):
    """The version in a ``<distribution>==<version>`` line, or None.

    Text in, value out, so the negative controls below can feed it a divergent pair
    without writing to the repo. Comments and blank lines are ignored; anything that
    is not an exact ``==`` pin returns None, which the callers treat as a failure
    rather than as agreement.
    """
    for raw in requirements_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, rest = line.partition("==")
        if sep and name.strip() == distribution:
            return rest.strip() or None
    return None


def _core_pins(texts):
    """``{deployment: pinned bioimageio.core}`` for a mapping of deployment -> text."""
    return {
        name: _exact_pin(text, "bioimageio.core") for name, text in texts.items()
    }


def test_both_deployments_pin_the_same_bioimageio_core():
    """The second input to the env-name hash, which no test used to cover (#0529).

    ``pin_bioimageio_conda_env(conda_env, core_version, spec_version)`` takes the two
    bioimageio versions as PARAMETERS, and each call site fills them from its own
    interpreter: ``runtime.py`` with ``version("bioimageio.core")``, ``entry.py`` via
    ``_get_bioimageio_versions``. Those are different deployments with different
    requirement files, so the hashes agree only while both files install the same
    core. Nothing asserted that, and a divergence would not fail anything -- it would
    silently stop the pre-built env from ever being reused, which cost 909.9 s cold
    against 85.2 s warm on #0497.

    ``bioimageio.spec`` is not pinned directly in either file: it arrives transitively,
    pinned exactly by core. So equal core implies equal spec, and core is the thing to
    assert.
    """
    texts = {name: path.read_text() for name, path in REQ_OF_DEPLOYMENT.items()}
    for name, path in REQ_OF_DEPLOYMENT.items():
        assert path.is_file(), f"{path} is missing, so this guard measured nothing"

    pins = _core_pins(texts)
    missing = sorted(name for name, pin in pins.items() if pin is None)
    assert not missing, (
        f"{missing} do not pin bioimageio.core to an exact == version, so the two "
        "deployments can resolve different cores and the env-name hashes will differ"
    )
    assert len(set(pins.values())) == 1, (
        f"the deployments pin different bioimageio.core versions: {pins}. Both call "
        "pin_bioimageio_conda_env and both hash the result, so the pre-built env can "
        "never be reused and every custom-env test pays a cold build."
    )


def test_the_same_core_guard_fails_when_the_two_files_diverge():
    """Negative control for the test above (#0529 clause 2).

    A guard that cannot be made to go red is not a guard. Feeding the same helper a
    divergent pair must produce a different answer, and an un-pinned pair must produce
    None rather than quietly agreeing.
    """
    agree = _core_pins(
        {"entry": "bioimageio.core==0.11.0", "runtime": "bioimageio.core==0.11.0"}
    )
    assert len(set(agree.values())) == 1, agree

    diverge = _core_pins(
        {"entry": "bioimageio.core==0.11.0", "runtime": "bioimageio.core==0.12.0"}
    )
    assert len(set(diverge.values())) == 2, (
        f"the helper cannot see a divergence, so the guard above proves nothing: "
        f"{diverge}"
    )

    # A range, not an exact pin, must NOT read as agreement with an exact one.
    loose = _core_pins(
        {"entry": "bioimageio.core==0.11.0", "runtime": "bioimageio.core>=0.11.0"}
    )
    assert loose["runtime"] is None, loose

    # A comment mentioning the distribution must not be mistaken for a pin.
    commented = _core_pins({"entry": "# bioimageio.core==9.9.9\nnumpy==2.0.0"})
    assert commented["entry"] is None, commented


def test_the_pydantic_pin_is_revisited_when_bioimageio_core_moves():
    """The pin's retirement condition, made mechanical (#0530).

    ``CUSTOM_ENV_PYDANTIC_PIN`` is a workaround for a serializer crash in
    ``bioimageio.spec`` 0.5.12.0 under pydantic 2.14+. Upstream fixed the call site in
    spec **0.5.14.3**, but that fix is unreachable while ``bioimageio.core==0.11.0``
    requires spec ``==0.5.12.0`` exactly. So the pin retires when, and only when, this
    app moves to a core release that pairs with spec >= 0.5.14.3.

    That event is a change to these requirements files, which is exactly what this
    asserts. It is deliberately NOT a dated marker: a date fires on a day nobody chose
    and says nothing about upstream, whereas this cannot fire spuriously and cannot be
    missed, because you cannot bump core without tripping it.

    When it fails, do not just update the constant below. Check which
    ``bioimageio.spec`` the new core pins; if that is >= 0.5.14.3, delete
    ``CUSTOM_ENV_PYDANTIC_PIN`` and the tests above with it.
    """
    pins = _core_pins(
        {name: path.read_text() for name, path in REQ_OF_DEPLOYMENT.items()}
    )
    for name, pin in sorted(pins.items()):
        assert pin == PIN_MEASURED_AGAINST_CORE, (
            f"{name} now pins bioimageio.core=={pin}, but CUSTOM_ENV_PYDANTIC_PIN was "
            f"measured against {PIN_MEASURED_AGAINST_CORE} (which pins "
            f"bioimageio.spec==0.5.12.0, the version carrying the crash). Check the "
            f"spec pairing of {pin}: if it is >= 0.5.14.3 the serializer fix is in and "
            f"the pydantic bound should be DELETED, not carried forward. See myco "
            f"#0530."
        )


def test_the_retirement_guard_fails_on_a_moved_core():
    """Negative control for the retirement guard (#0530).

    The guard above is an equality against a constant, so it is only meaningful if a
    moved core really does compare unequal -- otherwise it would keep passing forever,
    which is the failure mode the row was filed about.
    """
    moved = _exact_pin("bioimageio.core==0.12.0", "bioimageio.core")
    assert moved is not None and moved != PIN_MEASURED_AGAINST_CORE, moved

    unchanged = _exact_pin(
        f"bioimageio.core=={PIN_MEASURED_AGAINST_CORE}", "bioimageio.core"
    )
    assert unchanged == PIN_MEASURED_AGAINST_CORE, unchanged


def test_the_measured_core_constant_matches_what_the_pin_comment_claims():
    """Tie the constant above to the prose it is standing in for.

    If someone edits runtime.py's comment block to name a different core without
    touching this file, the retirement guard would be keyed to a version the pin no
    longer claims -- so assert the two still agree.
    """
    runtime_text = RUNTIME.read_text()
    needle = f"bioimageio.core=={PIN_MEASURED_AGAINST_CORE}"
    assert needle in runtime_text, (
        f"runtime.py no longer mentions {needle}, so PIN_MEASURED_AGAINST_CORE is "
        "keyed to a version the pin's own documentation has moved off"
    )
