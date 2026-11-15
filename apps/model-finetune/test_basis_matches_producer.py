"""The basis stamped on a contribution must name the function that computed its digest.

Every test in ``TestContributionsStampTheBasis`` takes the stamped label as given and
checks how it is RESOLVED. None of them checks whether the label is TRUE of the number
it sits beside, so a writer stamping the wrong constant is green there. This file pins
the one correspondence that decides it: ``runtime_cellpose.prepare_contribution``
computes a contribution's ``weights_sha256`` with exactly one named function, and the
basis stored alongside must be the one that names that function's output.

The producer is read out of the runtime source rather than asserted, so this stays
correct through a deliberate change: swap ``sha256_file`` for ``state_dict_digest`` on
that line and the expected basis swaps with it. It is torch-free on purpose - the
assertion is about which function is NAMED, not about running it, and importing
``runtime_cellpose`` would need torch and cellpose.

This is the only check in the suite that pins ``CONTRIBUTION_DIGEST_BASIS`` to the code
it describes. ``test_digest_basis.py`` compares the stamp against that constant, so the
two agree by construction and a change to the runtime's digest call leaves the whole of
it green while the constant silently becomes false - which is this row's own defect,
reintroduced from the other side.

Authored by the sibling instance of this lane (cwd ``.../quiet-mountain``) as the missing
assertion in its refutation on #0509, measured there at 3 failed / 2 passed against the
wrong constant. Adopted into the branch rather than rewritten, because a test whose point
is to be independent of its author's reasoning should not be re-derived by that author.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pool  # noqa: E402

# Which basis each digest function's output IS. Both entries are load-bearing: a
# one-entry map would make the detector below unable to return the other answer.
_BASIS_OF = {
    "state_dict_digest": pool.DIGEST_BASIS_TENSORS,
    "sha256_file": pool.DIGEST_BASIS_FILE,
}

_PRODUCER_LINE = re.compile(r"^\s*weights_sha256\s*=")


def _basis_for_line(line: str) -> str:
    """The basis named by a single assignment line, or fail saying why."""
    hits = sorted(fn for fn in _BASIS_OF if fn in line)
    if len(hits) != 1:
        pytest.fail(f"expected exactly one digest function on {line!r}, found {hits}")
    return _BASIS_OF[hits[0]]


def _producer_line() -> str:
    """The one line in the runtime that computes a contribution's weights_sha256."""
    src = (HERE / "runtime_cellpose.py").read_text(encoding="utf-8")
    lines = [ln for ln in src.splitlines() if _PRODUCER_LINE.match(ln)]
    if len(lines) != 1:
        pytest.fail(
            f"expected exactly one 'weights_sha256 =' assignment in "
            f"runtime_cellpose.py, found {len(lines)}: {lines}"
        )
    return lines[0]


def _entry(cid: str, sha: str, uploaded_at: float) -> Dict[str, Any]:
    """A contribution stub shaped like the one entry.py appends."""
    return {
        "contribution_id": cid,
        "weights_path": f"contributions/{cid}/weights.pt",
        "metadata_path": f"contributions/{cid}/metadata.json",
        "weights_sha256": sha,
        "architecture_signature": "sig-cpsam",
        "samples_seen": 40,
        "n_train_images": 5,
        "n_bytes": 1234,
        "uploaded_at": uploaded_at,
    }


def _append_one(sha: str, uploaded_at: float) -> Dict[str, Any]:
    index: Dict[str, Any] = {}
    pool.append_contribution(index, _entry("c-new", sha, uploaded_at))
    return index["contributions"][-1]


def test_the_producer_detector_can_return_either_answer() -> None:
    """Control. If this map only ever yielded one basis, the test below could not
    fail for the right reason, and a wrong stamp would pass as a tautology."""
    tensors = _basis_for_line(
        "        weights_sha256 = await asyncio.to_thread(pool.state_dict_digest, ckpt)"
    )
    file_ = _basis_for_line(
        "        weights_sha256 = await asyncio.to_thread(pool.sha256_file, ckpt)"
    )
    assert tensors == pool.DIGEST_BASIS_TENSORS
    assert file_ == pool.DIGEST_BASIS_FILE
    assert tensors != file_, "the detector cannot discriminate"


def test_the_runtime_names_exactly_one_producer() -> None:
    """Control on the source read: an empty or multiple match must fail loudly rather
    than resolve to whichever branch happens to be first."""
    line = _producer_line()
    assert "weights_sha256" in line
    assert _basis_for_line(line) in (pool.DIGEST_BASIS_TENSORS, pool.DIGEST_BASIS_FILE)


def test_the_CONSTANT_itself_names_the_producer() -> None:
    """The lock-step rule, stated against the constant rather than against a record.

    ``CONTRIBUTION_DIGEST_BASIS`` carries a comment binding it to
    ``prepare_contribution``'s digest call. That comment is a convention; this is the
    check. Both writers and the resolver's fallback read the constant, so if it stops
    describing the runtime every one of them is wrong at once - and nothing else in the
    suite would say so, because they all compare the stamp against this same constant.
    """
    expected = _basis_for_line(_producer_line())
    assert pool.CONTRIBUTION_DIGEST_BASIS == expected, (
        f"the runtime computes a contribution's digest with "
        f"{_producer_line().strip()!r}, so CONTRIBUTION_DIGEST_BASIS must be "
        f"{expected!r}, not {pool.CONTRIBUTION_DIGEST_BASIS!r}"
    )
    # The community head is the OTHER basis and must not be confused with it. Asserted
    # here so this case also fails if the two constants are ever collapsed into one.
    assert pool.CURRENT_DIGEST_BASIS != pool.CONTRIBUTION_DIGEST_BASIS


def test_an_unstamped_contribution_resolves_to_the_producers_basis() -> None:
    """The fallback path, which is every record written before this change.

    ``contribution_digest_basis`` answers an unstamped record from
    ``CONTRIBUTION_DIGEST_BASIS``; that answer has to name the producer too. Post-cutoff
    date on purpose, so a date-reasoning regression cannot supply the right answer by
    accident.
    """
    expected = _basis_for_line(_producer_line())
    legacy = _entry("c-legacy", "e" * 64, pool.TENSOR_DIGEST_SINCE + 86_400.0)
    legacy.pop("weights_sha256_basis", None)
    assert "weights_sha256_basis" not in legacy
    assert pool.contribution_digest_basis(legacy) == expected


def test_the_stamped_basis_names_the_function_that_computed_the_digest() -> None:
    """The assertion this file exists for.

    Post-cutoff ``uploaded_at`` deliberately: the stamp must be right on its own and
    not be rescued by a date that would have proved the file meaning anyway.
    """
    line = _producer_line()
    expected = _basis_for_line(line)
    stored = _append_one("a" * 64, pool.TENSOR_DIGEST_SINCE + 86_400.0)
    actual = stored.get("weights_sha256_basis")
    assert actual == expected, (
        f"the runtime computes a contribution's digest with "
        f"{line.strip()!r}, so the stored basis must be {expected!r}; the record "
        f"carries {actual!r}, which claims the digest is something it is not"
    )


def test_the_resolver_agrees_with_the_producer_too() -> None:
    """The same claim through the public reader, so a correct stamp read back through
    a resolver that overrides it would still be caught."""
    expected = _basis_for_line(_producer_line())
    stored = _append_one("b" * 64, pool.TENSOR_DIGEST_SINCE + 86_400.0)
    assert pool.contribution_digest_basis(stored) == expected


def test_a_contribution_is_not_published_as_reproducible_by_re_merge() -> None:
    """The consequence, which is the harm rather than the label.

    ``_DIGEST_REPRODUCIBLE`` maps the tensor basis to True and the file basis to
    False, and ``digest_provenance`` publishes it next to the number. A contribution
    digest is sha256 over the .pt file, so re-merging cannot reproduce it. Claiming
    otherwise tells a contributor a mismatch is evidence about the weights, which is
    the exact conclusion #0104 and #0417 exist to prevent.
    """
    stored = _append_one("c" * 64, pool.TENSOR_DIGEST_SINCE + 86_400.0)
    prov = pool.digest_provenance(
        stored, digest_key="weights_sha256", created_at_key="uploaded_at"
    )
    assert prov["basis"] == _basis_for_line(_producer_line())
    assert prov["reproducible_by_re_merge"] is False, (
        "a contribution's digest is a file digest, so a re-merge cannot reproduce "
        f"it; provenance published {prov['reproducible_by_re_merge']!r}"
    )


# A value no code path in this repo produces. Not a basis constant, not a digest, and
# not a timestamp — so an assertion that the stored record does NOT carry it cannot be
# satisfied by some other mechanism happening to write the right answer.
_FORGED_BASIS = "caller_supplied_basis_0524_no_code_path_produces_this"


def test_the_forged_basis_is_not_a_real_one() -> None:
    """Control for the test below. If ``_FORGED_BASIS`` ever collided with a real
    constant, "the stored record does not carry the forged value" would be satisfied
    by the writer stamping normally, and the override would stop being tested."""
    real = {
        pool.CONTRIBUTION_DIGEST_BASIS,
        pool.CURRENT_DIGEST_BASIS,
        pool.DIGEST_BASIS_ABSENT,
        pool.DIGEST_BASIS_FILE,
        pool.DIGEST_BASIS_TENSORS,
    }
    assert _FORGED_BASIS not in real, (
        f"{_FORGED_BASIS!r} collides with a real basis constant {real!r}, so the "
        "override test below would pass for the wrong reason"
    )


def test_a_caller_supplied_basis_is_overridden_not_preserved() -> None:
    """#0524. ``append_contribution``'s docstring says the basis is "Stamped here
    rather than accepted as a field on ``entry``", and gives the reason: the writer is
    the only thing that knows which digest it just computed, so a caller-supplied
    basis could disagree with the number beside it.

    That claim was NOT enforced. Replacing the stamping assignment with a
    ``setdefault`` — leaving a caller's value in place — was measured GREEN on the
    ``apps/model-finetune`` suite, the ``tests/`` suite and #0370's own gate during the
    #0509 review ("Arm E"). A control arm that stamps the WRONG constant at the same
    site is caught by 10 tests, so the suite discriminated the VALUE and not its
    PROVENANCE. This test pins the provenance.

    Latent today — no current caller supplies the field, because
    ``prepare_contribution`` computes ``weights_sha256`` in this process and the client
    never sends it. The point is that the invariant is stated at the write site and so
    must hold for a caller who has not read the docstring.
    """
    entry = _entry("c-forged", "d" * 64, pool.TENSOR_DIGEST_SINCE + 86_400.0)
    entry["weights_sha256_basis"] = _FORGED_BASIS

    index: Dict[str, Any] = {}
    pool.append_contribution(index, entry)
    stored = index["contributions"][-1]

    assert stored["weights_sha256_basis"] == pool.CONTRIBUTION_DIGEST_BASIS, (
        "append_contribution must IMPOSE the basis it computed, not accept the "
        f"caller's: stored {stored['weights_sha256_basis']!r}, expected "
        f"{pool.CONTRIBUTION_DIGEST_BASIS!r}. A setdefault at the stamping site "
        "produces exactly this failure."
    )
    # Stated separately: the first assertion alone would also pass if the forged value
    # happened to equal the constant, which the control above forbids.
    assert stored["weights_sha256_basis"] != _FORGED_BASIS

    # The override must happen on the stored COPY. If append_contribution mutated the
    # caller's dict instead, a caller reusing that dict would silently carry the
    # writer's basis into a second, differently-computed record.
    assert entry["weights_sha256_basis"] == _FORGED_BASIS, (
        "append_contribution overwrote the caller's own entry dict; it is supposed to "
        "copy with dict(entry) and stamp the copy"
    )
