"""#0523 — every ``weights_sha256`` the wire publishes carries a basis beside it, and
the basis it carries is the one the CONTRIBUTION resolver gives.

`#0509` stamped ``weights_sha256_basis`` on every new stored contribution. But a
bioimage.io client does not read the stored record; it reads what ``pool_wire_state``
publishes, and that carried the number with no basis next to it — so from outside the
pool, two digests computed under different bases stayed indistinguishable. That is the
harm `#0509`'s own lede described, surviving `#0509`'s fix because one row is about
what is STORED and this one is about what is PUBLISHED.

Read the payload BACK from the call rather than inspecting the writer: a test that
asserts on the projection source would pass even if the key never reached the wire.

Fixtures are distinct PER RECORD — four contributions resolving to three different
bases — so a constant, or a value leaking from a sibling record, cannot pass:

    c-file         stamped CONTRIBUTION_DIGEST_BASIS    -> DIGEST_BASIS_FILE
    c-unstamped    a digest, no stamp, AFTER the cutoff -> DIGEST_BASIS_FILE
    c-badstamp     a digest, an unrecognised stamp      -> DIGEST_BASIS_UNDETERMINED
    c-nodigest     no digest at all                     -> DIGEST_BASIS_ABSENT

``c-unstamped`` is the record this file got wrong on its first pass, and the reason the
fixture set grew a fourth member. It pinned ``DIGEST_BASIS_UNDETERMINED`` there, which
made the correct one-keyword call go RED on two tests — the same two names a deliberate
constant-answer mutant reds — so the fix was blocked by this file and would have read
as the fix being wrong. The undetermined answer now comes from a record that is
genuinely undetermined: ``c-badstamp`` states a basis no build recognises, which
``test_digest_basis.py::test_an_unrecognised_stamp_is_undetermined_not_passed_through``
independently establishes resolves that way. Three distinct bases, all three correct.

A second, smaller fixture covers a pool whose contribution ids are NOT unique. That is a
caller bug rather than anything a deposit can do, but the projection cannot enforce the
invariant and the map has one slot per id, so a violation silently published one record's
basis for the other — whichever rendered last. The tests there assert the one property
that is still achievable: no record is handed an answer that is confidently wrong for it.

The expected values are read off ``pool`` rather than typed, so a rename moves the test
with the code instead of silently passing against a stale literal.
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
_SHA_BADSTAMP = "c" * 64

# A basis string no build of this app writes. Not a typo of a real constant: it has to
# be recognisably foreign so a reader cannot mistake this fixture for a stale copy of
# one of the real values.
_UNRECOGNISED_BASIS = "sha256_over_something_a_newer_app_knows_about"


def _index() -> Dict[str, Any]:
    """A pool holding one stamped contribution, one unstamped, one stamped with a
    basis this build does not recognise, and one with no digest at all."""
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
                "contribution_id": "c-badstamp",
                "weights_sha256": _SHA_BADSTAMP,
                "weights_sha256_basis": _UNRECOGNISED_BASIS,
                "uploaded_at": after_cutoff,
                "n_bytes": 44,
                "n_train_images": 4,
                "samples_seen": 40,
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


def test_the_unrecognised_fixture_basis_really_is_unrecognised() -> None:
    """Control for ``c-badstamp``. If ``_UNRECOGNISED_BASIS`` ever became a value this
    build knows, that record would resolve to it and the undetermined arm below would
    be testing nothing — passing for the wrong reason."""
    assert _UNRECOGNISED_BASIS not in {
        pool.DIGEST_BASIS_TENSORS,
        pool.DIGEST_BASIS_FILE,
        pool.DIGEST_BASIS_UNDETERMINED,
        pool.DIGEST_BASIS_ABSENT,
    }
    assert (
        pool.contribution_digest_basis(
            {"weights_sha256": _SHA_BADSTAMP, "weights_sha256_basis": _UNRECOGNISED_BASIS}
        )
        == pool.DIGEST_BASIS_UNDETERMINED
    )


def test_every_published_digest_has_a_basis_beside_it() -> None:
    """Clause 1. Read back from the call: one entry per published contribution.

    The length assertion is not redundant with the set comparison: sets collapse
    duplicate ids, so a map that dropped one of two entries sharing an id would still
    satisfy ``published == attributed``. On THIS fixture, whose four ids are distinct,
    the two assertions do catch different failures — a map that dropped a whole record
    fails both, one that merged two into one fails only the length — but neither can be
    made to fail by an id collision, because this pool has none. That case has its own
    fixture below, and it is not simply "the length assertion with duplicates": when two
    records share an id the counts CANNOT match, since the map has one slot per id, so
    the property that has to hold there is a different one."""
    wire = _wire()
    published = {c["contribution_id"] for c in wire["contributions"]}
    attributed = set(wire["digest_provenance"]["contributions"])
    assert published == attributed, (
        "every published contribution must be attributed and nothing else: "
        f"published-but-unattributed {sorted(published - attributed)}, "
        f"attributed-but-unpublished {sorted(attributed - published)}"
    )
    assert published, "the fixture published no contributions, so this proves nothing"
    assert len(wire["digest_provenance"]["contributions"]) == len(wire["contributions"]), (
        f"{len(wire['contributions'])} contributions published but "
        f"{len(wire['digest_provenance']['contributions'])} attributed; ids that "
        "collide or are None collapse in the map and leave a published digest with no "
        "basis beside it"
    )


def _colliding_index(second_basis: str) -> Dict[str, Any]:
    """A pool whose ids are NOT unique — two records claiming ``c-twice``.

    Unreachable through the deposit path: ``prepare_contribution`` requires a
    ``contribution_id`` and ``append_contribution`` writes it, so this is a caller bug
    upstream of the wire. It is fixtured anyway because the projection cannot enforce
    that invariant and silently published one record's basis for the other when it was
    violated — the entry that survived was whichever rendered last.
    """
    after_cutoff = pool.TENSOR_DIGEST_SINCE + 86_400.0
    return {
        "model_type": "cpsam",
        "architecture_signature": "sig-cpsam",
        "contributions": [
            {
                "contribution_id": "c-twice",
                "weights_sha256": _SHA_FILE,
                "weights_sha256_basis": pool.CONTRIBUTION_DIGEST_BASIS,
                "uploaded_at": after_cutoff,
                "n_bytes": 11,
            },
            {
                "contribution_id": "c-twice",
                "weights_sha256": _SHA_BADSTAMP,
                "weights_sha256_basis": second_basis,
                "uploaded_at": after_cutoff + 1.0,
                "n_bytes": 22,
            },
        ],
        "merges": [],
        "decided_contribution_ids": [],
    }


def test_the_colliding_fixture_really_does_defeat_the_set_comparison() -> None:
    """Control for the pair below. If this pool did NOT collapse in the map, the two
    tests after it would be asserting about a case that cannot arise and would pass for
    the wrong reason. Two records, one id: the sets match and the counts cannot."""
    wire = pool.pool_wire_state(
        _colliding_index(_UNRECOGNISED_BASIS), pool_artifact_id=POOL_ARTIFACT_ID
    )
    published = wire["contributions"]
    prov = wire["digest_provenance"]["contributions"]
    assert len(published) == 2, "both records must still be published"
    assert {c["contribution_id"] for c in published} == set(prov), (
        "the set comparison is satisfied here, which is the blind spot: "
        f"{sorted(str(c['contribution_id']) for c in published)} vs {sorted(map(str, prov))}"
    )
    assert len(prov) == 1, (
        f"the map has one slot per id, so two records sharing one id must yield one "
        f"entry; got {len(prov)}"
    )


def test_a_colliding_id_does_not_publish_one_records_basis_for_another() -> None:
    """The map must not hand a client an answer that is confidently wrong for one of the
    two records. Neither record's own basis may be published at a shared id when the two
    disagree, because the map cannot say which record the value beside it belongs to.

    Before the guard this published ``undetermined`` — the SECOND record's answer, right
    for it and wrong for the correctly stamped first one, chosen by nothing but render
    order. The value published now is the same string, so the assertion that carries this
    test is the NOTE: it has to name the collision rather than describe an unrecognised
    stamp, which is the ordinary meaning of that basis and is false of both records here.
    """
    wire = pool.pool_wire_state(
        _colliding_index(_UNRECOGNISED_BASIS), pool_artifact_id=POOL_ARTIFACT_ID
    )
    entry = wire["digest_provenance"]["contributions"]["c-twice"]
    assert entry["basis"] == pool.DIGEST_BASIS_UNDETERMINED
    assert entry["note"] == pool.CONTRIBUTION_ID_COLLISION_NOTE, (
        "a colliding id must be attributed as a collision, not as whichever record "
        f"rendered last; got note {entry['note'][:80]!r}"
    )
    assert entry["note"] not in pool.CONTRIBUTION_DIGEST_BASIS_NOTES.values(), (
        "the collision note must be its own sentence: reusing the undetermined note "
        "tells a contributor their stamp was unrecognised, which is false of both records"
    )
    assert "more than one contribution under this id" in entry["note"]


def test_colliding_ids_that_agree_are_not_downgraded() -> None:
    """The guard fires on DISAGREEMENT only. Two records under one id that resolve to the
    same basis are attributed with it: the map's answer is then true of both, and
    refusing to give it would lose a correct statement to a rule about ids."""
    wire = pool.pool_wire_state(
        _colliding_index(pool.CONTRIBUTION_DIGEST_BASIS), pool_artifact_id=POOL_ARTIFACT_ID
    )
    entry = wire["digest_provenance"]["contributions"]["c-twice"]
    assert entry["basis"] == pool.DIGEST_BASIS_FILE
    assert entry["note"] == pool.CONTRIBUTION_DIGEST_BASIS_NOTES[pool.DIGEST_BASIS_FILE]
    assert entry["note"] != pool.CONTRIBUTION_ID_COLLISION_NOTE


def test_each_record_gets_its_own_basis_not_a_constant() -> None:
    """Clause 3. Four records, three different bases, matched per id."""
    prov = _by_contribution()
    assert prov["c-file"]["basis"] == pool.DIGEST_BASIS_FILE
    assert prov["c-unstamped"]["basis"] == pool.DIGEST_BASIS_FILE
    assert prov["c-badstamp"]["basis"] == pool.DIGEST_BASIS_UNDETERMINED
    assert prov["c-nodigest"]["basis"] == pool.DIGEST_BASIS_ABSENT
    bases = {cid: entry["basis"] for cid, entry in prov.items()}
    assert len(set(bases.values())) == 3, (
        f"the map returned {len(set(bases.values()))} distinct basis value(s) for four "
        f"records with three different provenances: {bases}. A constant, or a leak from "
        "a sibling record, looks exactly like this."
    )


def test_the_published_basis_is_the_contribution_resolvers_answer() -> None:
    """The standing arm, and the one that would have caught the defect this file
    shipped: for every record the wire publishes, the basis beside the number must be
    what ``contribution_digest_basis`` says about that same record.

    Reverting ``pool_wire_state`` to the community-style call
    (``digest_provenance(c, digest_key="weights_sha256", created_at_key="uploaded_at")``)
    makes this RED on ``c-unstamped``, which is the whole point of it existing. It
    compares against the resolver rather than against literals so it cannot drift from
    the module's own answer, and the per-record literals above are what keep it from
    passing when both sides are wrong together."""
    index = _index()
    prov = pool.pool_wire_state(index, pool_artifact_id=POOL_ARTIFACT_ID)[
        "digest_provenance"
    ]["contributions"]
    disagreements = {
        c["contribution_id"]: (
            prov[c["contribution_id"]]["basis"],
            pool.contribution_digest_basis(c),
        )
        for c in index["contributions"]
        if prov[c["contribution_id"]]["basis"] != pool.contribution_digest_basis(c)
    }
    assert not disagreements, (
        "the wire published a basis the contribution resolver does not give, as "
        f"(published, resolver) per id: {disagreements}. The community-head resolver "
        "answers 'undetermined' for an unstamped contribution whose code path proves "
        "the file basis — see CONTRIBUTION_DIGEST_BASIS."
    )


def test_an_unstamped_contribution_is_the_file_basis_not_undetermined() -> None:
    """Named separately from the arm above because it is the specific regression, and a
    reader who sees only this name should know what went wrong.

    An unstamped contribution is every deposit made before `#0509`'s stamping landed —
    the real backlog this attribution exists for. ``prepare_contribution`` has computed
    ``weights_sha256`` with ``sha256_file`` on every build that ever wrote one, so an
    absent stamp proves the file basis rather than leaving it open. Publishing
    "undetermined" there tells a contributor their digest cannot be attributed when the
    code path attributes it with certainty, and ships it with
    ``reproducible_by_re_merge`` and a "treat a mismatch as UNEXPLAINED" note."""
    prov = _by_contribution()
    assert prov["c-unstamped"]["basis"] == pool.CONTRIBUTION_DIGEST_BASIS
    assert prov["c-unstamped"]["basis"] != pool.DIGEST_BASIS_UNDETERMINED


def test_the_deposit_date_does_not_change_the_answer() -> None:
    """The same unchanged code path must not answer two ways on nothing but when a
    deposit happened. Two records identical except for ``uploaded_at``, one either side
    of the community head's cut-over instant: same basis, or this surface is
    date-reasoning on a history where no cut-over ever applied."""
    index = _index()
    early = dict(index["contributions"][1])
    early["contribution_id"] = "c-unstamped-early"
    early["uploaded_at"] = pool.TENSOR_DIGEST_SINCE - 86_400.0
    index["contributions"].append(early)
    prov = pool.pool_wire_state(index, pool_artifact_id=POOL_ARTIFACT_ID)[
        "digest_provenance"
    ]["contributions"]
    late_basis = prov["c-unstamped"]["basis"]
    early_basis = prov["c-unstamped-early"]["basis"]
    assert early_basis == late_basis == pool.CONTRIBUTION_DIGEST_BASIS, (
        f"before the cut-over -> {early_basis!r}, after it -> {late_basis!r}. Both "
        "records were written by the same unchanged function; a difference here is the "
        "date being treated as evidence on a surface that has only ever had one basis."
    )


def test_the_absent_case_is_distinguishable_from_the_known_case() -> None:
    """Clause 2. 'no digest stored' and 'a digest whose basis is known' must not read
    the same, and the absent answer must be DIGEST_BASIS_ABSENT rather than a guess at
    the file basis — a silent default is the defect #0509 was filed for.

    Note the two absences this separates: no DIGEST at all (this record, and it is not
    attributable) versus no STAMP on a digest that exists (``c-unstamped``, and it is
    attributable with certainty). Collapsing them in either direction is a defect."""
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
    assert prov["c-badstamp"]["basis"] == pool.DIGEST_BASIS_UNDETERMINED
    assert prov["c-badstamp"]["basis"] not in {known, absent}


def test_the_note_is_written_about_a_contribution_not_the_community_head() -> None:
    """The explanation travels with the number, so a wrong explanation is worse than
    the bare number this row set out to fix. ``DIGEST_BASIS_NOTES`` is written about
    the community head and three of its sentences are false of a contribution: the app
    did not name the contributor's checkpoint, "re-merge and compare" is not an
    operation that produces a contribution digest, and a contribution is not a merge
    that "admitted nothing"."""
    prov = _by_contribution()
    for cid, entry in prov.items():
        note = entry["note"]
        assert note, f"{cid} published an empty note"
        assert "this merge admitted nothing" not in note, (
            f"{cid} is a contribution, not a merge, and publishes no community "
            f"checkpoint: {note!r}"
        )
        assert "re-merge and compare against" not in note, (
            f"{cid}'s note offers an unperformable remedy — re-merging does not "
            f"produce a contribution digest: {note!r}"
        )
        assert "head-<random>.pt" not in note, (
            f"{cid}'s note claims this app chose the filename, which is true of the "
            f"community head and false of an uploaded contribution: {note!r}"
        )
        assert note not in pool.DIGEST_BASIS_NOTES.values(), (
            f"{cid} published the community-head note verbatim; the contribution "
            "surface has its own in CONTRIBUTION_DIGEST_BASIS_NOTES"
        )
    assert "re-hashing that same file" in prov["c-unstamped"]["note"], (
        "the file-basis note must name the one operation a contributor can actually "
        f"perform: {prov['c-unstamped']['note']!r}"
    )


def test_re_merge_reproducibility_is_not_asserted_on_this_surface() -> None:
    """``False`` here would answer a question that does not apply: a contribution
    digest covers one uploaded file, so re-merging does not bear on it in either
    direction. ``_DIGEST_REPRODUCIBLE``'s own comment reserves a third state for "we
    do not know" precisely so it does not render as "no"; this is a fourth case, and
    the note is where it is said."""
    for cid, entry in _by_contribution().items():
        assert entry["reproducible_by_re_merge"] is None, (
            f"{cid} published reproducible_by_re_merge="
            f"{entry['reproducible_by_re_merge']!r}; re-merging an identical member set "
            "is not an operation that reproduces a contribution's file digest, so "
            "False asserts an answer to a question that does not apply"
        )


def test_the_map_entries_carry_the_same_keys_as_the_soup_map() -> None:
    """A client joins ``contributions`` and ``soups`` from one payload. Diverging key
    sets would make it branch on which map it is reading, which is why
    ``reproducible_by_re_merge`` is kept and set to None rather than dropped."""
    dp = _wire()["digest_provenance"]
    expected = {"basis", "reproducible_by_re_merge", "note"}
    for cid, entry in dp["contributions"].items():
        assert set(entry) == expected, f"{cid} published {sorted(entry)}"
    # The sibling map is built by digest_provenance, whose shape this must match.
    assert set(pool.digest_provenance({})) == expected


def test_the_payload_still_publishes_the_number_itself() -> None:
    """The basis is published BESIDE the digest, not instead of it. Pairing these
    keeps 'the basis is present' from being satisfied by the digest disappearing."""
    wire = _wire()
    by_id = {c["contribution_id"]: c for c in wire["contributions"]}
    assert by_id["c-file"]["weights_sha256"] == _SHA_FILE
    assert by_id["c-unstamped"]["weights_sha256"] == _SHA_UNSTAMPED
    assert by_id["c-badstamp"]["weights_sha256"] == _SHA_BADSTAMP
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
