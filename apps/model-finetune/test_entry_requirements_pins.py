"""The entry env must not install a numpy of its own.

This exists because the failure it guards against is close to undiagnosable from
the outside. zarr **3.1.6** declares ``numpy>=2.0`` — it is the only release that
does, which is why the bound is on 3.1.6 and not on 3 — and a bare ``zarr`` in
``requirements-entry.txt`` resolves to it, pulling numpy 2.x into the entry
runtime_env overlay where it shadows the worker image's numpy 1.26.4. Every compiled module
in that image — pandas 1.5.3, scipy, cupy — was built against numpy 1.x headers
and raises ``numpy.dtype size changed ... Expected 96 from C header, got 88 from
PyObject``. ``ray.serve._private.utils`` imports pandas unconditionally, so the
replica dies in its constructor, before it can log anything: the deployment
reports DEPLOY_FAILED with an EMPTY log list, which is the same thing an absent
log looks like.

Nothing in the local suite or in CI can see that, because neither runs on the
worker image. A text assertion over the requirements file is the only check that
lives where the mistake is made.
"""

import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ENTRY_REQS = HERE / "requirements-entry.txt"


def _requirement_lines():
    """The requirement lines, with comments and blanks dropped."""
    text = ENTRY_REQS.read_text()
    out = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def test_the_requirements_file_is_where_this_test_thinks_it_is():
    """THE CONTROL. Without it, a moved or renamed file would make every
    assertion below vacuous rather than red."""
    assert ENTRY_REQS.is_file(), f"not found: {ENTRY_REQS}"
    assert _requirement_lines(), "parsed zero requirements — the parser is broken"


def test_zarr_excludes_3_1_6_which_is_the_release_that_requires_numpy_2():
    """A bare ``zarr`` is the specific mistake, and it resolves to 3.1.6.

    Asserted on the parsed specifier rather than the literal text, so any correct
    bound passes. The excluded set and the required set are BOTH asserted: a pin
    that is merely tight enough to keep numpy (``zarr<3``) would also silently
    drop OME-NGFF v0.5, so "no numpy 2" is not on its own the property wanted.
    """
    packaging_requirements = pytest.importorskip("packaging.requirements")
    packaging_version = pytest.importorskip("packaging.version")

    zarr_reqs = [
        packaging_requirements.Requirement(line)
        for line in _requirement_lines()
        if packaging_requirements.Requirement(line).name.lower() == "zarr"
    ]
    assert len(zarr_reqs) == 1, f"expected exactly one zarr requirement, got {zarr_reqs}"

    spec = zarr_reqs[0].specifier
    assert str(spec), (
        "zarr is declared with NO version specifier. A bare `zarr` resolves to "
        "3.1.6, the one release that declares numpy>=2.0, which makes the entry "
        "deployment fail to start with no replica logs at all."
    )

    # Must be EXCLUDED: 3.1.6 and up declare numpy>=2.0.
    for bad in ("3.1.6", "3.2.0", "4.0.0"):
        assert not spec.contains(packaging_version.Version(bad), prereleases=True), (
            f"the zarr specifier {spec} still allows {bad}. From 3.1.6 zarr "
            "declares numpy>=2.0, which replaces the worker image's numpy 1.26.4 "
            "and breaks every numpy-1.x-compiled module in it, including the "
            "pandas that ray.serve imports on every replica."
        )

    # Must be EXCLUDED the other way too: zarr 2 cannot open a zarr.json array,
    # and the store wrapper is written against the zarr-3 Store interface.
    for bad in ("2.18.7", "2.16.1"):
        assert not spec.contains(packaging_version.Version(bad), prereleases=True), (
            f"the zarr specifier {spec} allows {bad}. zarr 2 cannot read "
            "zarr_format=3 (OME-NGFF v0.5) stores, and it rejects this module's "
            "zarr-3 Store with TypeError: Unsupported type for store_like."
        )

    # Must be ALLOWED: what production resolves to, and what CI installs.
    for good in ("3.1.5", "3.1.3"):
        assert spec.contains(packaging_version.Version(good)), (
            f"the zarr specifier {spec} excludes {good}. 3.1.5 is what this env "
            "resolves to and 3.1.3 is what CI installs via "
            "requirements-datasets.txt; the suite has to run on the major that "
            "ships."
        )


def test_numpy_is_declared_without_an_upper_bound():
    """The entry env must accept whatever numpy the image ships. A cap here would
    force pip to install a different numpy, which is the same failure in reverse."""
    packaging_requirements = pytest.importorskip("packaging.requirements")

    numpy_reqs = [
        packaging_requirements.Requirement(line)
        for line in _requirement_lines()
        if packaging_requirements.Requirement(line).name.lower() == "numpy"
    ]
    assert len(numpy_reqs) == 1, f"expected exactly one numpy requirement, got {numpy_reqs}"
    assert not str(numpy_reqs[0].specifier), (
        f"numpy is pinned to {numpy_reqs[0].specifier}. Leave it unpinned so pip "
        "keeps the worker image's own numpy and installs none."
    )


def test_the_reason_for_the_pin_is_recorded_next_to_it():
    """A why-guard. The pin is one character wide and reads like a nicety, so the
    diagnosis has to survive in the file or the next editor will undo it."""
    text = ENTRY_REQS.read_text()
    for needle in ("numpy.dtype size changed", "numpy>=2.0"):
        assert needle in text, (
            f"{needle!r} is no longer in requirements-entry.txt. Removing the "
            "explanation is how this pin gets reverted; keep it with the pin."
        )
