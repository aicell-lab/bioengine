"""No file in this bundle may call a NumPy array method that 2.0 removed.

This app ships two Ray deployments on two different NumPy majors, on purpose:
``requirements-runtime.txt`` holds NumPy at 2.x because micro-sam's
``python-elf`` requires it, and ``requirements-runtime-cellpose.txt`` pins
1.26.4 and says in terms that the two cannot share a major. So "it works here"
is not a statement about the bundle — a call that is fine in the cellpose
runtime raises ``AttributeError`` in the micro-sam one, and the bundle's own
tests cannot even be *collected* in the environment where it does.

The specific trap pinned here is the removal of array METHODS in NumPy 2.0
whose free-function form survived. ``arr.ptp()`` reads as the natural spelling,
the module-level ``np.ptp(arr)`` works on both majors, and the two are
otherwise interchangeable — so the wrong one is easy to write, easy to review
past, and raises only in the runtime nobody ran the suite in. Three CLAHE tests
in ``test_preflight.py`` sat in exactly that state: green in CI, uncollectable
under 2.x, and the ``AttributeError`` they raised there named an array method,
which reads like a defect in the code under test rather than a version
mismatch in the harness.

Why a source scan rather than a run under both majors: a run can only fail on
a line it reaches. These are single call sites inside plotting and assertion
code, much of which no unit test executes, so a two-environment suite run is
exactly the check that would stay green while the defect was present. The scan
also costs nothing to keep — nothing here installs a second NumPy.

Why the AST and not a regex: the first version of this scan was a regex over
lines and its first finding was this module's own prose, because a docstring
that NAMES the banned spelling is indistinguishable from code that uses it
once you are matching text. ``ast`` sees calls and nothing else, so comments,
docstrings and string literals are out of scope by construction rather than by
a list of exceptions that has to be maintained.
"""

from __future__ import annotations

import ast
import pathlib

# Methods NumPy 2.0 removed from ndarray whose module-level function remains.
# Add to this as more are found; each name is a spelling that works on 1.26.4
# and raises on 2.x.
REMOVED_IN_NUMPY_2 = frozenset({"ptp"})

# The receivers for which `<recv>.<name>(...)` is the REQUIRED form rather than
# the banned one: `np.ptp(arr)` is a module-level function call that happens to
# be written as an attribute access.
MODULE_ALIASES = frozenset({"np", "numpy"})

APP_DIR = pathlib.Path(__file__).resolve().parent


def _offenders_in(source: str, label: str) -> list[str]:
    """Every `<expr>.<removed-method>(...)` call in ``source``.

    A syntax error is an offence rather than a silent zero: a file this scan
    cannot parse is a file it has not checked, and those two must not report
    the same result.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"{label}: could not be parsed, so it was NOT scanned: {e}"]

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in REMOVED_IN_NUMPY_2:
            continue
        recv = func.value
        if isinstance(recv, ast.Name) and recv.id in MODULE_ALIASES:
            continue  # np.ptp(arr) — the required form
        found.append(
            f"{label}:{node.lineno}: a .{func.attr}() call on "
            f"{type(recv).__name__}  (use np.{func.attr}(...))"
        )
    return found


def _python_files() -> list[pathlib.Path]:
    me = pathlib.Path(__file__).name
    return sorted(p for p in APP_DIR.glob("*.py") if p.name != me)


def test_no_file_calls_an_ndarray_method_numpy_2_removed() -> None:
    offenders = []
    for path in _python_files():
        offenders += _offenders_in(path.read_text(), path.name)

    assert not offenders, (
        "NumPy 2.0 removed these ndarray methods; this bundle has a runtime on "
        "2.x, so the module-level function is the only form that works in every "
        "environment the app ships into:\n  " + "\n  ".join(offenders)
    )


def test_THE_SCAN_CAN_SEE_THE_THING_IT_BANS() -> None:
    """The control. A scan whose detector has stopped matching reports the same
    clean result as a clean tree, and nothing distinguishes the two.
    """
    must_flag = {
        "method on a local": "x = out.ptp()",
        "method in an expression": "disp = (disp - disp.min()) / (disp.ptp() or 1.0)",
        "method on an attribute": "x = self._buffer.ptp()",
        "method on a subscript": "x = frames[0].ptp()",
        "method on a call result": "x = np.asarray(a).ptp()",
    }
    must_not_flag = {
        "the required free function": "x = np.ptp(out)",
        "the required form, qualified": "x = numpy.ptp(frame)",
        "two required calls on one line": "w = np.ptp(a) - np.ptp(b)",
        "the banned spelling inside a STRING": 'DOC = "write np.ptp(a), never a.ptp()"',
        "the banned spelling in a COMMENT": "x = np.ptp(a)  # not a.ptp()",
        "the banned spelling in a DOCSTRING": '"""Use np.ptp(a) rather than a.ptp()."""',
        "an unrelated method of the same shape": "x = out.ptp_estimate()",
    }

    missed = [k for k, src in must_flag.items() if not _offenders_in(src, "<t>")]
    assert not missed, f"the scan would NOT have caught: {missed}"

    wrong = [k for k, src in must_not_flag.items() if _offenders_in(src, "<t>")]
    assert not wrong, f"the scan wrongly flags: {wrong}"


def test_AN_UNPARSEABLE_FILE_IS_AN_OFFENCE_NOT_A_PASS() -> None:
    """A file the scan cannot parse has not been checked, and must not report
    the same empty result as a clean one."""
    out = _offenders_in("def broken(:\n    pass\n", "<t>")
    assert out and "NOT scanned" in out[0]


def test_the_scan_actually_reads_files() -> None:
    """A glob that matched nothing would make the first test vacuous, and it
    would pass. Assert the file set is real and holds the two modules the
    original defect lived in."""
    names = {p.name for p in _python_files()}
    assert len(names) > 20, f"only {len(names)} python file(s) scanned — the glob is wrong"
    assert {"test_preflight.py", "export_worker_cellpose.py"} <= names
