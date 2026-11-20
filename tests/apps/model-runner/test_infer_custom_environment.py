"""``infer(custom_environment=True)`` runs the model in the environment it
declares, instead of the serving runtime.

A model may name dependencies this runner does not carry. A micro-SAM model
needs ``micro_sam``, and without it the model fails at LOAD, before any
inference, with ``ModuleNotFoundError`` — so no amount of retrying or input
fixing helps. ``test`` has always been able to run in the model's own declared
conda environment; this route gives ``infer`` the same ability rather than
adding the dependency to the serving runtime, which would commit every request
to carrying it.

The mechanism is an INTERPRETER SWAP, not a ``conda run`` wrapper: the resident
inference child signals "model loaded" over an inherited file descriptor, and a
wrapper process does not pass one through. That is the same reason the
Cellpose-3 split swaps interpreters, and it is why the environment's own
``bin/python`` is what reaches the runtime.

Everything here is derived from the two app modules as text or AST. Nothing
imports bioengine, ray or bioimageio: the app modules pull all three at import
time and none is installed in the environment this suite runs in.
"""

import ast
import pathlib

import pytest

APP = pathlib.Path(__file__).resolve().parents[3] / "apps" / "model-runner"
ENTRY = APP / "entry.py"
RUNTIME = APP / "runtime.py"


def _method_source(path, class_name, method_name):
    """Source of ONE method, so a claim about it cannot be satisfied elsewhere.

    Scoped deliberately. A whole-file grep for "the refusal is present" passes
    when the refusal sits in an unrelated method, or in a comment, so it cannot
    tell "guarded at this site" from "the words appear in this file".
    """
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                ):
                    return ast.get_source_segment(path.read_text(), item) or ""
    pytest.fail(f"{class_name}.{method_name} not found in {path.name}")


# --------------------------------------------------------- the public parameter


def test_infer_accepts_custom_environment():
    """The parameter exists on ``infer``, not only on ``test``."""
    src = _method_source(ENTRY, "EntryDeployment", "infer")
    assert "custom_environment" in src, "infer has no custom_environment parameter"


def test_infer_forwards_custom_environment_to_the_driver():
    """A parameter the background driver never receives would do nothing."""
    src = _method_source(ENTRY, "EntryDeployment", "infer")
    assert "custom_environment=bool(custom_environment)" in src, src[-400:]


def test_the_parameter_test_is_scoped_and_can_fail():
    """Control for ``_method_source``: it must return ONE method, not the file.

    Without this, the two tests above would pass on a file that mentions
    ``custom_environment`` anywhere at all — which ``entry.py`` does, in
    ``test``'s own parameter and in several comments.
    """
    infer_src = _method_source(ENTRY, "EntryDeployment", "infer")
    assert "async def infer(" in infer_src
    # A different method's distinctive text must NOT be in this slice.
    assert "_compute_conda_env_name" not in infer_src
    # And the helper must fail loudly rather than return "" for a bad name.
    # ``pytest.fail`` raises ``Failed``, which derives from BaseException and
    # NOT from Exception, so ``pytest.raises(Exception)`` would not catch it
    # and this control would pass for the wrong reason.
    with pytest.raises(BaseException):
        _method_source(ENTRY, "EntryDeployment", "no_such_method_exists")


# ------------------------------------------------- refusals, at the right site


def test_the_driver_refuses_when_no_conda_implementation_is_installed():
    """Refused before any env solve, in the driver itself.

    No image this project publishes ships a conda implementation, so this is
    the common case on a default deployment, not an edge one. Refusing early
    matters because the alternative is a ``FileNotFoundError`` from deep inside
    the call, which a caller reads as the model being broken.
    """
    src = _method_source(ENTRY, "EntryDeployment", "_execute_infer")
    assert "resolve_conda_binary()" in src, src[:600]
    assert "CONDA_IMPLEMENTATIONS" in src
    assert "no inference was run" in src, "the refusal must say nothing ran"
    assert "custom_environment=False" in src, "it must name the way out"


def test_the_driver_refuses_a_model_that_declares_no_environment():
    """Not a silent downgrade, unlike the test path — and that asymmetry is
    deliberate. A test that quietly ran in the standard environment still
    returns a report about the model; an inference that quietly ran there
    returns the ModuleNotFoundError this route exists to avoid."""
    src = _method_source(ENTRY, "EntryDeployment", "_execute_infer")
    assert "_model_declares_custom_env" in src, src[:600]
    assert "declares no environment of its own" in src


def test_the_refusals_are_in_the_driver_and_not_merely_in_the_file():
    """Control for the two tests above.

    A reviewer's mutant that moved the guard out of ``_execute_infer`` while
    leaving the words in ``entry.py`` must turn these red. The discriminator is
    that the scoped slice is much smaller than the file.
    """
    src = _method_source(ENTRY, "EntryDeployment", "_execute_infer")
    whole = ENTRY.read_text()
    assert len(src) < len(whole) / 4, "the slice is not scoped to one method"
    assert "declares no environment of its own" in src
    # The same phrase is in the file by construction; being in the FILE is
    # what this control shows to be insufficient.
    assert whole.count("declares no environment of its own") >= 1


# ------------------------------------------------ one environment, not several


def test_several_weight_formats_without_a_choice_is_refused_not_guessed():
    """Inference runs ONE weight format, and each declares its own environment.

    Guessing would mean reproducing ``bioimageio.core``'s preference order in
    this repository. ``_compute_conda_env_name`` is already one hand-maintained
    reproduction of an upstream algorithm, and its own docstring says a drift
    there is silent — a second one is a second thing nothing can police.
    """
    src = _method_source(ENTRY, "EntryDeployment", "_conda_env_for_weights")
    assert "Pass weights_format to say which one" in src, src[-500:]
    assert "duplicating bioimageio.core's preference order" in src


def test_a_single_weight_format_needs_no_choice():
    """The row's own models carry exactly one format, so the common path must
    not demand a parameter the caller has no way to pick."""
    src = _method_source(ENTRY, "EntryDeployment", "_conda_env_for_weights")
    assert "len(present) == 1" in src, src[-500:]


def test_an_unknown_weights_format_is_rejected():
    src = _method_source(ENTRY, "EntryDeployment", "_conda_env_for_weights")
    assert "is not one this model" in src, src[-500:]


# ------------------------------------- the interpreter seam, and its precedence


def test_the_runtime_takes_an_absolute_interpreter_verbatim():
    src = _method_source(RUNTIME, "RuntimeDeployment", "_interpreter")
    assert "os.path.isabs(python_env)" in src, src
    assert "return python_env" in src


def test_the_runtime_refuses_an_interpreter_that_does_not_exist():
    """A custom-env request must not reach the spawn with no env built."""
    src = _method_source(RUNTIME, "RuntimeDeployment", "_interpreter")
    assert "does not exist" in src, src


def test_the_declared_environment_outranks_the_id_based_mapping():
    """Both can apply to one request; the explicit ask is the stronger one."""
    src = _method_source(ENTRY, "EntryDeployment", "_execute_infer")
    assert "interpreter or _python_env_for(model_id)" in src, src[-600:]


def test_the_environment_is_released_even_when_the_inference_fails():
    """``_prebuild_conda_envs`` marks envs in use and says the caller MUST
    release them. A failed run holds the same marks, so the release has to be
    in a ``finally`` or those envs are pinned on the shared disk for good."""
    src = _method_source(ENTRY, "EntryDeployment", "_execute_infer")
    assert "finally:" in src, src[-600:]
    tail = src[src.index("finally:") :]
    assert "_release_inuse_envs" in tail, tail


# ------------------------------------------------------- no second format list


def test_the_prebuild_and_the_resolver_share_one_weight_format_list():
    """A format in only one of the two would mean an environment built and
    never used, or asked for and never built."""
    src = ENTRY.read_text()
    assert "weight_formats = list(self._ENV_WEIGHT_FORMATS)" in src
    # Exactly one literal list of the formats exists in the file.
    assert src.count('"tensorflow_saved_model_bundle"') == 1, (
        "a second weight-format list has reappeared; the two can now drift"
    )


def test_tensorflow_js_is_not_offered_an_environment():
    """``bioimageio.core`` refuses to test it, so no env is ever built for it
    and inference must not offer one."""
    tree = ast.parse(ENTRY.read_text())
    formats = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_ENV_WEIGHT_FORMATS"
            for t in node.targets
        ):
            formats = [e.value for e in node.value.elts]
    assert formats, "_ENV_WEIGHT_FORMATS not found as a literal"
    assert "tensorflow_js" not in formats, formats
    # Positive control: the list is not simply empty.
    assert "pytorch_state_dict" in formats, formats
