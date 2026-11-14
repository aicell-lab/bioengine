"""#0523 — every ``weights_sha256`` the wire publishes carries a basis beside it.

`#0509` stamped ``weights_sha256_basis`` on every new stored contribution. But a
bioimage.io client does not read the stored record; it reads what ``pool_wire_state``
publishes, and that carried the number with no basis next to it — so from outside the
pool, two digests computed under different bases stayed indistinguishable. That is the
harm `#0509`'s own lede described, surviving `#0509`'s fix because one row is about
what is STORED and this one is about what is PUBLISHED.

Read the payload BACK from the call rather than inspecting the writer: a test that
asserts on the projection source would pass even if the key never reached the wire.

Fixtures are distinct PER RECORD — three contributions resolving to three different
bases — so a constant, or a value leaking from a sibling record, cannot pass:

    c-file         stamped CONTRIBUTION_DIGEST_BASIS   -> DIGEST_BASIS_FILE
    c-unstamped    a digest, no stamp, after cutoff    -> DIGEST_BASIS_UNDETERMINED
    c-nodigest     no digest at all                    -> DIGEST_BASIS_ABSENT

The three expected values are read off ``pool`` rather than typed, so a rename moves
the test with the code instead of silently passing against a stale literal.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pool  # noqa: E402

POOL_ARTIFACT_ID = "bioimage-io/cellpose-finetuning-pool"

# Distinct digests, so a map that returned one record's answer for another would have
# to collide on these too.
_SHA_FILE = "a" * 64
_SHA_UNSTAMPED = "b" * 64


def _index() -> Dict[str, Any]:
    """A pool holding one stamped contribution, one unstamped, and one with no digest."""
    after_cutoff = pool.TENSOR_DIGEST_SINCE + 86_400.0
    return {
        "model_type": "cpsam",
        "architecture_signature": "sig-cpsam",
        "contributions": [
            {
                "contribution_id": "c-file",
                "weights_sha256": _SHA_FILE,
                "weights_sha256_basis": pool.CONTRIBUTION_DIGEST_BASIS,
                "uploaded_at": after_cutoff,
                "n_bytes": 11,
                "n_train_images": 1,
                "samples_seen": 10,
            },
            {
                "contribution_id": "c-unstamped",
                "weights_sha256": _SHA_UNSTAMPED,
                "uploaded_at": after_cutoff,
                "n_bytes": 22,
                "n_train_images": 2,
                "samples_seen": 20,
            },
            {
                "contribution_id": "c-nodigest",
                "uploaded_at": after_cutoff,
                "n_bytes": 33,
                "n_train_images": 3,
                "samples_seen": 30,
            },
        ],
        "merges": [],
        "decided_contribution_ids": [],
    }


def _wire() -> Dict[str, Any]:
    return pool.pool_wire_state(_index(), pool_artifact_id=POOL_ARTIFACT_ID)


def _by_contribution() -> Dict[str, Any]:
    return _wire()["digest_provenance"]["contributions"]


def test_the_three_expected_bases_are_distinct() -> None:
    """Control. If any two of these constants were equal, the per-record assertions
    below could not tell "the map answered per record" from "the map answered once"."""
    trio = [
        pool.DIGEST_BASIS_FILE,
        pool.DIGEST_BASIS_UNDETERMINED,
        pool.DIGEST_BASIS_ABSENT,
    ]
    assert len(set(trio)) == 3, f"expected three distinct bases, got {trio}"


def test_every_published_digest_has_a_basis_beside_it() -> None:
    """Clause 1. Read back from the call: one entry per published contribution."""
    wire = _wire()
    published = {c["contribution_id"] for c in wire["contributions"]}
    attributed = set(wire["digest_provenance"]["contributions"])
    assert published == attributed, (
        "every published contribution must be attributed and nothing else: "
        f"published-but-unattributed {sorted(published - attributed)}, "
        f"attributed-but-unpublished {sorted(attributed - published)}"
    )
    assert published, "the fixture published no contributions, so this proves nothing"


def test_each_record_gets_its_own_basis_not_a_constant() -> None:
    """Clause 3. Three records, three different bases, matched per id."""
    prov = _by_contribution()
    assert prov["c-file"]["basis"] == pool.DIGEST_BASIS_FILE
    assert prov["c-unstamped"]["basis"] == pool.DIGEST_BASIS_UNDETERMINED
    assert prov["c-nodigest"]["basis"] == pool.DIGEST_BASIS_ABSENT
    bases = {cid: entry["basis"] for cid, entry in prov.items()}
    assert len(set(bases.values())) == 3, (
        f"the map returned {len(set(bases.values()))} distinct basis value(s) for three "
        f"records with three different provenances: {bases}. A constant, or a leak from "
        "a sibling record, looks exactly like this."
    )


def test_the_absent_case_is_distinguishable_from_the_known_case() -> None:
    """Clause 2. 'no digest stored' and 'a digest whose basis is known' must not read
    the same, and the absent answer must be DIGEST_BASIS_ABSENT rather than a guess at
    the file basis — a silent default is the defect #0509 was filed for."""
    prov = _by_contribution()
    known = prov["c-file"]["basis"]
    absent = prov["c-nodigest"]["basis"]
    assert absent != known
    assert absent == pool.DIGEST_BASIS_ABSENT
    assert absent != pool.CONTRIBUTION_DIGEST_BASIS, (
        "a record with no digest was published as if it carried the file basis; that "
        "is the silent default this row exists to prevent"
    )
    # And the undetermined case is its own third answer, not folded into either.
    assert prov["c-unstamped"]["basis"] == pool.DIGEST_BASIS_UNDETERMINED
    assert prov["c-unstamped"]["basis"] not in {known, absent}


def test_the_payload_still_publishes_the_number_itself() -> None:
    """The basis is published BESIDE the digest, not instead of it. Pairing these
    keeps 'the basis is present' from being satisfied by the digest disappearing."""
    wire = _wire()
    by_id = {c["contribution_id"]: c for c in wire["contributions"]}
    assert by_id["c-file"]["weights_sha256"] == _SHA_FILE
    assert by_id["c-unstamped"]["weights_sha256"] == _SHA_UNSTAMPED
    assert by_id["c-nodigest"].get("weights_sha256") is None


def test_the_pinned_contribution_field_set_is_not_widened() -> None:
    """The map is additive and top-level for a reason: contributions[]'s field set is
    pinned key-for-key by test_pool.py::test_contribution_key_set, and widening it is
    the mismatch SCHEMA_VERSION exists to make visible. This asserts the record shape
    did NOT change, which is the half that test cannot see from this file."""
    wire = _wire()
    for c in wire["contributions"]:
        assert "weights_sha256_basis" not in c, (
            "the per-record projection was widened; publish the basis in the "
            "additive top-level digest_provenance map instead"
        )
        assert "digest_provenance" not in c


def test_the_sibling_maps_still_stand() -> None:
    """The new key is added alongside, not in place of, what #0417 and #0509 put here."""
    dp = _wire()["digest_provenance"]
    assert set(dp) == {"soups", "contributions", "current_community"}, (
        f"digest_provenance should carry exactly three joins, found {sorted(dp)}"
    )


def test_an_unheld_contribution_is_absent_from_the_map() -> None:
    """Negative control, paired with the presence assertion above: the map must not
    answer for an id the pool does not hold, or 'attributed' would be meaningless."""
    assert "c-not-in-this-pool" not in _by_contribution()
