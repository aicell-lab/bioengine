"""The mirror's decisions, pinned so a later change cannot loosen them quietly.

WHAT THESE COVER, and the reason each matters rather than being shape-checking:

* a generalist model needs its DECODER too, and a plain SAM model has none. A
  mirror that got this wrong would pass its own checks and then fail the run;
* an unparseable digest REFUSES. A digest that cannot be read is not a weaker
  check, it is no check;
* a wrong local file is reported as `replace`, NOT as `place`. The distinction
  is the finding: something put bad bytes in the cache, and collapsing it into
  "absent" loses that;
* a member the mirror does not hold is `unavailable`, not silently handled, so
  a partial mirror cannot look complete;
* the four buckets are DISJOINT and their union is exactly what was asked for,
  which is the assertion that catches a member dropped by a future edit.

WHAT THEY DO NOT COVER: any network, artifact or real checkpoint. This module
takes no part in downloading, and the tests use small files with real digests so
they run anywhere in milliseconds.
"""

from __future__ import annotations

import pathlib

import pytest

import checkpoint_mirror as cm

# A registry shaped like micro-sam's: generalist types carry a decoder, the
# plain SAM types do not. Digests are computed from the bytes each test writes,
# so nothing here is a hand-copied constant that could drift from the content.
GENERALIST = "vit_t_lm"
PLAIN = "vit_b"


# SHA256 IS THE DEFAULT HERE, and that is about the harness rather than the
# design. The registry's own algorithm is xxh128, but `xxhash` is a dependency
# of the RUNTIME (pooch needs it to read an xxh128 registry) and is not
# installed in every test environment -- it is absent from this one. Making the
# structural tests depend on it turned twelve of them into ModuleNotFoundError,
# which is an ERROR dressed as a failure and says nothing about the rules under
# test. So the buckets, the layout and the refusals are exercised with sha256,
# which the module supports for exactly the integrity role described in its
# docstring, and the xxh128-specific cases below guard themselves.
HAVE_XXHASH = True
try:  # noqa: SIM105
    import xxhash  # noqa: F401
except ImportError:
    HAVE_XXHASH = False


def _d(data: bytes, alg: str = "sha256") -> str:
    """A real digest of *data*, so no constant here can drift from the bytes."""
    if alg == "sha256":
        import hashlib

        return f"sha256:{hashlib.sha256(data).hexdigest()}"
    import xxhash as _x

    return f"xxh128:{_x.xxh128(data).hexdigest()}"


def _write(root: pathlib.Path, key: str, data: bytes) -> pathlib.Path:
    p = cm.models_dir(root) / key
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


# ── which members a model type needs ───────────────────────────────────────


def test_a_generalist_model_needs_its_DECODER_too():
    reg = {GENERALIST: "x", f"{GENERALIST}_decoder": "y", PLAIN: "z"}
    assert cm.required_members(GENERALIST, reg) == [GENERALIST, f"{GENERALIST}_decoder"]


def test_a_plain_SAM_model_has_NO_decoder_and_that_is_not_an_omission():
    """The plain SAM types genuinely have no decoder in the registry. Returning
    one would make every mirror of them look permanently incomplete."""
    reg = {GENERALIST: "x", f"{GENERALIST}_decoder": "y", PLAIN: "z"}
    assert cm.required_members(PLAIN, reg) == [PLAIN]


def test_an_UNKNOWN_model_type_RAISES_rather_than_returning_nothing():
    """An empty list reads as 'nothing needed' and would mirror nothing at all
    while reporting success."""
    with pytest.raises(cm.MirrorError, match="not in the checkpoint registry"):
        cm.required_members("vit_does_not_exist", {GENERALIST: "x"})


@pytest.mark.parametrize("bad", ["", None, 0, [], {}])
def test_a_missing_model_type_RAISES(bad):
    with pytest.raises(cm.MirrorError):
        cm.required_members(bad, {GENERALIST: "x"})


# ── digests: fail closed on anything unreadable ─────────────────────────────


def test_a_registry_digest_parses_into_algorithm_and_hex():
    assert cm.parse_digest("xxh128:" + "a" * 32) == ("xxh128", "a" * 32)
    assert cm.parse_digest("sha256:" + "b" * 64) == ("sha256", "b" * 64)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "deadbeef",  # no algorithm
        "xxh128:",  # no hex
        "xxh128:" + "a" * 31,  # one short
        "xxh128:" + "a" * 33,  # one long
        "sha256:" + "a" * 32,  # xxh128 length under sha256's name
        "md5:" + "a" * 32,  # unsupported algorithm
        "XXH128:" + "a" * 32,  # wrong case
        "xxh128:" + "A" * 32,  # uppercase hex
        "xxh128:zzzz" + "a" * 28,  # not hex
        " xxh128:" + "a" * 32 + " x",  # trailing junk
        None,
        1234,
    ],
)
def test_an_UNREADABLE_digest_REFUSES(bad):
    """Fail closed. The caller must not be able to carry on as though a check
    had happened."""
    with pytest.raises(cm.MirrorError):
        cm.parse_digest(bad)


def test_a_file_is_checked_against_its_digest(tmp_path):
    data = b"checkpoint-bytes" * 500
    _write(tmp_path, GENERALIST, data)
    assert cm.member_is_intact(cm.models_dir(tmp_path) / GENERALIST, _d(data))


def test_a_CORRUPT_file_fails_its_digest(tmp_path):
    good = b"checkpoint-bytes" * 500
    _write(tmp_path, GENERALIST, b"substituted" * 500)
    assert not cm.member_is_intact(cm.models_dir(tmp_path) / GENERALIST, _d(good))


def test_an_ABSENT_file_reads_as_not_intact_rather_than_raising(tmp_path):
    assert not cm.member_is_intact(cm.models_dir(tmp_path) / "nope", _d(b"x"))


@pytest.mark.skipif(not HAVE_XXHASH, reason="xxhash absent; see the note by _d")
def test_BOTH_algorithms_work_so_provenance_and_integrity_can_be_recorded(tmp_path):
    """Provenance is the registry's xxh128; a sha256 of our own is integrity.
    Both are supported so recording both costs no special case."""
    data = b"weights" * 1000
    _write(tmp_path, GENERALIST, data)
    p = cm.models_dir(tmp_path) / GENERALIST
    assert cm.member_is_intact(p, _d(data, "sha256"))
    assert cm.member_is_intact(p, _d(data, "xxh128"))


@pytest.mark.skipif(HAVE_XXHASH, reason="xxhash present, so the refusal cannot occur")
def test_WITHOUT_xxhash_an_xxh128_ALONE_REFUSES_rather_than_falling_back(tmp_path):
    """THE INVERSE CASE, and it is only testable where xxhash is missing.

    Upstream publishes no sha256, so hashing with something else would turn a
    provenance check into a self-consistency check. With xxh128 as the ONLY
    recorded digest there is nothing left to check, and the module raises
    instead of answering. A silent fallback is the one substitution that must
    not happen, and this is the environment that can prove it does not.

    Note what is NOT asserted: False. False would read as corruption and send a
    caller re-fetching good bytes from the origin that is down.
    """
    _write(tmp_path, GENERALIST, b"bytes" * 10)
    with pytest.raises(cm.MirrorError, match="NOT verified"):
        cm.member_is_intact(cm.models_dir(tmp_path) / GENERALIST, "xxh128:" + "a" * 32)


@pytest.mark.skipif(
    HAVE_XXHASH, reason="only reachable where xxh128 cannot be computed"
)
def test_WITHOUT_xxhash_an_UNCHECKABLE_PROVENANCE_is_REFUSED_not_waived(tmp_path):
    """A review measured this returning True in silence, which is the defect.

    With xxhash absent, `{xxh128: WRONG, sha256: right}` answered True and
    nothing in the result said the provenance leg had been skipped -- so "both
    digests verified" and "the xxh128 was silently dropped" were the same
    answer. The ruling that the FALLBACK verifies the sha256 is about the
    runtime download path; it does not say a missing hasher may waive the
    provenance claim, so the default is now a refusal.
    """
    data = b"weights" * 512
    _write(tmp_path, GENERALIST, data)
    p = cm.models_dir(tmp_path) / GENERALIST
    good_sha = _d(data, "sha256").split(":", 1)[1]

    with pytest.raises(cm.MirrorError, match="PROVENANCE is unverified"):
        cm.member_is_intact(p, {"xxh128": "a" * 32, "sha256": good_sha})


@pytest.mark.skipif(
    HAVE_XXHASH, reason="only reachable where xxh128 cannot be computed"
)
def test_WITHOUT_xxhash_the_WAIVER_IS_EXPLICIT_and_NAMES_what_it_skipped(tmp_path):
    """The ruling's fallback, taken deliberately at a call site that asks for it.

    Both directions, because a waiver that only ever said True would verify
    nothing -- and the record must NAME the algorithm it could not compute, or
    the caller is back to being unable to tell the two cases apart.
    """
    data = b"weights" * 512
    _write(tmp_path, GENERALIST, data)
    p = cm.models_dir(tmp_path) / GENERALIST
    good_sha = _d(data, "sha256").split(":", 1)[1]

    rec = cm.verify_member(
        p,
        {"xxh128": "a" * 32, "sha256": good_sha},
        allow_unverified_provenance=True,
    )
    assert rec["intact"] is True
    assert rec["checked"] == ["sha256"], rec
    assert rec["skipped"] == ["xxh128"], rec

    bad = cm.verify_member(
        p,
        {"xxh128": "a" * 32, "sha256": "b" * 64},
        allow_unverified_provenance=True,
    )
    assert bad["intact"] is False, bad
    assert bad["skipped"] == ["xxh128"], bad


@pytest.mark.skipif(
    HAVE_XXHASH, reason="only reachable where xxh128 cannot be computed"
)
def test_the_REPORT_distinguishes_fully_checked_from_provenance_skipped(tmp_path):
    """What a seeding log has to be able to say. A list of bad members cannot
    carry "this one matched, but only on integrity", and a report that cannot
    carry it is how an unverified provenance claim gets published as a verified
    one."""
    data = b"weights" * 512
    _write(tmp_path, GENERALIST, data)
    wanted = {
        GENERALIST: {
            "xxh128": "a" * 32,
            "sha256": _d(data, "sha256").split(":", 1)[1],
        }
    }

    report = cm.verify_seeded_report(tmp_path, wanted, allow_unverified_provenance=True)

    assert report[GENERALIST]["intact"] is True
    assert report[GENERALIST]["skipped"] == ["xxh128"]
    assert report[GENERALIST]["checked"] == ["sha256"]


@pytest.mark.skipif(
    not HAVE_XXHASH, reason="a wrong xxh128 can only be detected where it is computable"
)
def test_a_WRONG_PROVENANCE_hash_is_NOT_EXCUSED_by_a_right_integrity_hash(tmp_path):
    """The two digests answer different questions, so both must hold.

    A right sha256 says "unchanged since we hashed it". It says nothing about
    whether the bytes are upstream's, which is the claim a wrong xxh128 refutes.
    Accepting the file because one of the two matched would publish a mirror of
    bytes upstream never served.
    """
    data = b"weights" * 512
    _write(tmp_path, GENERALIST, data)
    p = cm.models_dir(tmp_path) / GENERALIST
    good_sha = _d(data, "sha256").split(":", 1)[1]

    assert not cm.member_is_intact(p, {"xxh128": "a" * 32, "sha256": good_sha})
    # ... and the control: with the real xxh128 it passes, so the refusal above
    # is the hash and not the two-digest form itself.
    assert cm.member_is_intact(
        p, {"xxh128": _d(data, "xxh128").split(":", 1)[1], "sha256": good_sha}
    )


def test_an_EMPTY_expectation_REFUSES_rather_than_verifying_nothing(tmp_path):
    """No recorded digest is not a weaker check, it is no check. An empty
    mapping must not be able to read as "everything matched"."""
    _write(tmp_path, GENERALIST, b"x" * 10)
    with pytest.raises(cm.MirrorError, match="at least one digest"):
        cm.member_is_intact(cm.models_dir(tmp_path) / GENERALIST, {})


def test_a_digest_LABELLED_as_another_algorithm_REFUSES(tmp_path):
    """`{"sha256": "xxh128:..."}` is a record that disagrees with itself. Picking
    either reading would be a guess about which half is the typo."""
    with pytest.raises(cm.MirrorError, match="labelled"):
        cm.parse_expectation({"sha256": "xxh128:" + "a" * 32})


def test_an_expectation_accepts_BARE_HEX_as_well_as_a_prefixed_digest():
    """Both forms, because the registry hands over `xxh128:<hex>` while a record
    keyed by algorithm naturally holds the hex alone."""
    assert cm.parse_expectation({"sha256": "a" * 64}) == {"sha256": "a" * 64}
    assert cm.parse_expectation({"sha256": "sha256:" + "a" * 64}) == {
        "sha256": "a" * 64
    }
    assert cm.parse_expectation("sha256:" + "a" * 64) == {"sha256": "a" * 64}


def test_an_expectation_that_is_NEITHER_a_string_nor_a_mapping_REFUSES():
    for bad in (None, 42, ["sha256:" + "a" * 64], b"sha256:" + b"a" * 64):
        with pytest.raises(cm.MirrorError):
            cm.parse_expectation(bad)


# ── the plan: four disjoint buckets ─────────────────────────────────────────


def test_an_INTACT_member_is_left_alone(tmp_path):
    """Re-placing it would mean moving up to 2.4 GB to learn what the digest
    already said."""
    data = b"a" * 4096
    _write(tmp_path, GENERALIST, data)
    plan = cm.seed_plan(tmp_path, {GENERALIST: _d(data)})
    assert plan["ok"] == [GENERALIST]
    assert plan["place"] == [] and plan["replace"] == [] and plan["unavailable"] == []


def test_an_ABSENT_member_is_PLACED(tmp_path):
    plan = cm.seed_plan(tmp_path, {GENERALIST: _d(b"a" * 10)})
    assert plan["place"] == [GENERALIST] and plan["ok"] == []


def test_a_WRONG_local_file_is_REPLACE_and_NOT_place(tmp_path):
    """The distinction is the finding. `place` means a cold cache; `replace`
    means something wrote bad bytes into it, which is worth seeing rather than
    silently overwriting."""
    _write(tmp_path, GENERALIST, b"wrong" * 100)
    plan = cm.seed_plan(tmp_path, {GENERALIST: _d(b"right" * 100)})
    assert plan["replace"] == [GENERALIST]
    assert plan["place"] == []


def test_an_ABSENT_member_the_MIRROR_LACKS_is_UNAVAILABLE_not_quietly_handled(
    tmp_path,
):
    """A partial mirror must not look complete: the caller has to let the normal
    upstream fetch try.

    ABSENT is in the name because that is what `unavailable` now means. A
    present-but-wrong file the mirror lacks has its own bucket, and conflating
    the two was the defect a review measured.
    """
    plan = cm.seed_plan(
        tmp_path, {GENERALIST: _d(b"x" * 10)}, available=["something-else"]
    )
    assert plan["unavailable"] == [GENERALIST]
    assert plan["place"] == []
    assert plan["corrupt_unavailable"] == []


def test_an_intact_member_is_OK_even_when_the_mirror_LACKS_it(tmp_path):
    """Order matters: already-correct beats not-in-the-mirror, because nothing
    needs fetching. Reversing the two would report a working cache as a gap."""
    data = b"a" * 64
    _write(tmp_path, GENERALIST, data)
    plan = cm.seed_plan(tmp_path, {GENERALIST: _d(data)}, available=[])
    assert plan["ok"] == [GENERALIST] and plan["unavailable"] == []


def test_the_FIVE_BUCKETS_ARE_DISJOINT_and_cover_everything_asked_for(tmp_path):
    """THE STRUCTURAL ASSERTION. A future edit that drops a member from the loop
    would otherwise shrink the plan silently; here the union catches it.

    Five, not four. A review measured that a corrupt file the mirror LACKS came
    back as plain `unavailable`, indistinguishable from one that was never
    downloaded, so it gets its own bucket -- and this case is in the fixture.
    """
    ok_d = b"o" * 32
    bad_d = b"b" * 32
    _write(tmp_path, "m_ok", ok_d)
    _write(tmp_path, "m_bad", b"corrupt" * 10)
    _write(tmp_path, "m_bad_unmirrored", b"corrupt" * 10)
    wanted = {
        "m_ok": _d(ok_d),
        "m_bad": _d(bad_d),
        "m_bad_unmirrored": _d(b"w" * 8),
        "m_absent": _d(b"z" * 8),
        "m_missing_upstream": _d(b"q" * 8),
    }
    plan = cm.seed_plan(tmp_path, wanted, available=["m_ok", "m_bad", "m_absent"])
    buckets = [set(v) for v in plan.values()]
    union = set().union(*buckets)
    assert union == set(wanted), (plan, sorted(wanted))
    total = sum(len(v) for v in plan.values())
    assert total == len(wanted), f"a member is in two buckets: {plan}"
    assert plan["ok"] == ["m_ok"]
    assert plan["replace"] == ["m_bad"]
    assert plan["place"] == ["m_absent"]
    assert plan["unavailable"] == ["m_missing_upstream"]
    assert plan["corrupt_unavailable"] == ["m_bad_unmirrored"]


def test_a_CORRUPT_file_the_mirror_LACKS_is_DISTINGUISHABLE_from_an_absent_one(
    tmp_path,
):
    """The review's probe, as a test, and it is RED without the fix.

    Measured at the refused head: `models/vit_b_lm` holding the wrong bytes with
    `available=[]` returned `{'unavailable': ['vit_b_lm'], 'replace': []}` -- the
    same answer as a member that was never fetched. The two need different
    actions: both fall back upstream, but one of them also means something put a
    wrong file in the cache, and that is a finding whatever the mirror holds.
    """
    _write(tmp_path, GENERALIST, b"corrupt" * 10)
    wanted = {GENERALIST: _d(b"the right bytes")}

    corrupt_plan = cm.seed_plan(tmp_path, wanted, available=[])
    absent_plan = cm.seed_plan(tmp_path / "cold", wanted, available=[])

    assert corrupt_plan["corrupt_unavailable"] == [GENERALIST], corrupt_plan
    assert corrupt_plan["unavailable"] == [], corrupt_plan
    assert absent_plan["unavailable"] == [GENERALIST], absent_plan
    assert absent_plan["corrupt_unavailable"] == [], absent_plan
    assert corrupt_plan != absent_plan, "a corrupt cache still reads as a cold one"


def test_the_PLAN_REFUSES_an_unreadable_digest_BEFORE_touching_the_filesystem(
    tmp_path,
):
    with pytest.raises(cm.MirrorError):
        cm.seed_plan(tmp_path, {GENERALIST: "not-a-digest"})


# ── verification after seeding ──────────────────────────────────────────────


def test_verify_NAMES_the_members_that_are_wrong(tmp_path):
    """A bool cannot be reported. "The mirror failed" with no member named is
    not an actionable sentence."""
    good = b"g" * 100
    _write(tmp_path, "a", good)
    _write(tmp_path, "b", b"bad" * 10)
    bad = cm.verify_seeded(tmp_path, {"a": _d(good), "b": _d(b"expected" * 10)})
    assert bad == ["b"]


def test_verify_is_EMPTY_when_everything_matches(tmp_path):
    """The control: a verifier that always named something would satisfy the
    test above."""
    d1, d2 = b"1" * 50, b"2" * 50
    _write(tmp_path, "a", d1)
    _write(tmp_path, "b", d2)
    assert cm.verify_seeded(tmp_path, {"a": _d(d1), "b": _d(d2)}) == []


# ── the layout, read from micro-sam's source rather than assumed ────────────


def test_checkpoints_live_in_the_models_subfolder_named_by_REGISTRY_KEY():
    """micro-sam's `get_model_cache_directory` returns `<cache>/models`, and
    `model_registry.fetch("vit_t_lm")` caches the file as `vit_t_lm` with no
    extension. Both read from its source; a wrong layout would place files
    somewhere pooch never looks and the mirror would appear to do nothing."""
    assert cm.models_dir("/x") == pathlib.Path("/x/models")
    assert cm.MODELS_SUBDIR == "models"
    assert cm.DECODER_SUFFIX == "_decoder"
    assert cm.CACHE_DIR_ENV == "MICROSAM_CACHEDIR"


# ── round 2: a mismatch is decidable without the missing hasher ──────────────


@pytest.mark.skipif(
    HAVE_XXHASH, reason="only reachable where xxh128 cannot be computed"
)
def test_a_MISMATCH_is_reported_even_when_the_provenance_leg_is_SKIPPED(tmp_path):
    """The round-2 probe. A wrong file must read as wrong, not as unverifiable.

    Measured at the refused head: a file whose sha256 did not match raised the
    provenance refusal instead of reporting the mismatch, and the message said
    the file "matched [sha256]" -- which was false. What a missing hasher costs
    is the ability to call a file GOOD, never the ability to call it WRONG.
    """
    _write(tmp_path, GENERALIST, b"the wrong bytes")
    p = cm.models_dir(tmp_path) / GENERALIST
    other_sha = _d(b"some other bytes", "sha256").split(":", 1)[1]
    expectation = {"xxh128": "a" * 32, "sha256": other_sha}

    rec = cm.verify_member(p, expectation)

    assert rec["intact"] is False, rec
    assert rec["checked"] == ["sha256"], rec
    assert rec["skipped"] == ["xxh128"], rec
    assert cm.member_is_intact(p, expectation) is False


@pytest.mark.skipif(
    HAVE_XXHASH, reason="only reachable where xxh128 cannot be computed"
)
def test_the_refusal_message_does_not_claim_a_MATCH_that_did_not_happen(tmp_path):
    """The message itself, asserted, because this one was actively misleading.

    A refusal that says a file matched when it did not sends a reader looking
    for a missing dependency instead of at the corrupt file in front of them.
    """
    _write(tmp_path, GENERALIST, b"the wrong bytes")
    p = cm.models_dir(tmp_path) / GENERALIST
    other_sha = _d(b"some other bytes", "sha256").split(":", 1)[1]

    # No raise at all now, so there is no message to be wrong. The paired
    # assertion is that the GOOD file still produces the provenance refusal,
    # otherwise this test would pass with the refusal removed entirely.
    assert (
        cm.verify_member(p, {"xxh128": "a" * 32, "sha256": other_sha})["intact"]
        is False
    )

    good = b"exactly these bytes"
    _write(tmp_path, PLAIN, good)
    with pytest.raises(cm.MirrorError) as exc:
        cm.verify_member(
            cm.models_dir(tmp_path) / PLAIN,
            {"xxh128": "a" * 32, "sha256": _d(good, "sha256").split(":", 1)[1]},
        )
    assert "PROVENANCE is unverified" in str(exc.value)
    assert "nothing it was checked against disagreed" in str(exc.value)


@pytest.mark.skipif(
    HAVE_XXHASH, reason="only reachable where xxh128 cannot be computed"
)
def test_a_corrupt_member_still_LANDS_IN_A_BUCKET_rather_than_raising(tmp_path):
    """The consequence the review traced: `seed_plan` raised instead of planning.

    A provably corrupt file with two digests recorded used to raise "PROVENANCE
    is unverified", so the one case the buckets exist to separate became an
    exception. It must land in `replace` when the mirror can supply a copy and
    in `corrupt_unavailable` when it cannot.
    """
    _write(tmp_path, GENERALIST, b"the wrong bytes")
    wanted = {
        GENERALIST: {
            "xxh128": "a" * 32,
            "sha256": _d(b"some other bytes", "sha256").split(":", 1)[1],
        }
    }

    assert cm.seed_plan(tmp_path, wanted, available=[GENERALIST])["replace"] == [
        GENERALIST
    ]
    assert cm.seed_plan(tmp_path, wanted, available=[])["corrupt_unavailable"] == [
        GENERALIST
    ]


@pytest.mark.skipif(
    HAVE_XXHASH, reason="only reachable where xxh128 cannot be computed"
)
def test_seed_plan_can_PASS_THE_WAIVER_THROUGH_and_refuses_by_default(tmp_path):
    """Planning happens in the runtime, where `xxhash` may be absent. Without a
    way to pass the waiver, a two-digest manifest made every planning call raise
    rather than produce a plan -- and the default must still be the refusing
    one, or the waiver is not a decision anybody takes."""
    good = b"exactly these bytes"
    _write(tmp_path, GENERALIST, good)
    wanted = {
        GENERALIST: {
            "xxh128": "a" * 32,
            "sha256": _d(good, "sha256").split(":", 1)[1],
        }
    }

    with pytest.raises(cm.MirrorError, match="PROVENANCE is unverified"):
        cm.seed_plan(tmp_path, wanted, available=[GENERALIST])

    plan = cm.seed_plan(
        tmp_path, wanted, available=[GENERALIST], allow_unverified_provenance=True
    )
    assert plan["ok"] == [GENERALIST], plan
