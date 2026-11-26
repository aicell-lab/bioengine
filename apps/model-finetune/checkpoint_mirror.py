"""Deciding which base checkpoints to seed, where, and whether they are intact.

WHY THIS EXISTS. micro-sam fetches a model's base checkpoint over the network at
the start of a fine-tuning run. Measured 2026-11-26: the host serving the LM and
EM generalist weights, `uk1s3.embassy.ebi.ac.uk`, refuses connections — from a
worker's runtime and from an unrelated network alike, immediately and on both 80
and 443, while its sibling hosts at the same institution answer 200. So the fetch
fails before the first training step, and no amount of GPU or training data
helps. A deployment whose weights happen to be cached keeps working, which is why
the failure looks intermittent from outside.

A mirror fixes that, and this module is the part of it that holds no matter how
the mirror is wired in: given what a model type needs and what a cache already
holds, decide what to place and prove what is there.

TWO THINGS IT DELIBERATELY DOES NOT DO.

It does not download, upload, or touch a network. Everything here is a decision
over paths and digests, so it is testable without a cluster, an artifact or a
checkpoint -- which is the only reason the rules below can be pinned at all.

It does not replace anyone's verification. micro-sam fetches through pooch, and
pooch compares a cached file against its own registry digest before using it --
measured, not assumed: with a correct cached file it uses it and never reaches
the network; with a WRONG cached file and the origin up it re-fetches; with a
wrong cached file and the origin DOWN it RAISES rather than handing the bytes
over. So a seeded cache leaves that check standing, and this module's own
verification is a second line, not a substitute.

THE DIGEST IS UPSTREAM'S, AND IT IS xxh128. micro-sam's registry carries
`xxh128:<hex>` for every checkpoint as a literal in its own source, so a file
can be proved to be upstream's bytes without reaching upstream -- which matters
precisely because upstream is what is down. A sha256 computed here would prove
only that nothing changed SINCE we hashed it; any corruption that arrived before
that would be sealed in. Where both are recorded, the xxh128 is the provenance
claim and a sha256 is integrity.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping as _ABCMapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

# micro-sam's own layout, read from its source rather than guessed:
# `get_cache_directory()` honours MICROSAM_CACHEDIR, and
# `get_model_cache_directory()` returns `<cache>/models`, with SAM and micro-sam
# checkpoints as files directly in it. The FILENAME is the registry key --
# `model_registry.fetch("vit_t_lm")` caches as `vit_t_lm`, with no extension.
MODELS_SUBDIR = "models"

# The decoder of a micro-sam generalist model is a separate file, named by
# suffixing the model type. `_download_sam_model` fetches it only when that key
# is in the registry, which is why a mirror has to ask rather than assume: 14
# encoders carry 10 decoders between them.
DECODER_SUFFIX = "_decoder"

CACHE_DIR_ENV = "MICROSAM_CACHEDIR"

# `xxh128:<32 hex>` is what micro-sam's registry holds. Anchored, and the length
# is fixed, so a truncated or differently-prefixed value is rejected rather than
# silently treated as a digest.
_DIGEST_RE = re.compile(r"\A(?P<alg>[a-z0-9_]+):(?P<hex>[0-9a-f]+)\Z")

_SUPPORTED_ALGS = ("xxh128", "sha256")


class MirrorError(RuntimeError):
    """A mirrored checkpoint could not be trusted, so it was not used."""


def models_dir(cache_root: Any) -> Path:
    """The directory micro-sam reads checkpoints from, for a given cache root."""
    return Path(cache_root) / MODELS_SUBDIR


def required_members(model_type: str, registry: Mapping[str, str]) -> List[str]:
    """The registry keys a fine-tuning run of *model_type* needs.

    The encoder always; the decoder only when the registry has one. Returning
    the decoder unconditionally would make a mirror look incomplete for the
    plain SAM types, which genuinely have none; omitting it for the generalist
    types would produce a mirror that passes its own checks and then fails the
    run, which is worse.

    Raises for an unknown model type rather than returning an empty list: an
    empty answer reads as "nothing needed" and would quietly mirror nothing.
    """
    if not isinstance(model_type, str) or not model_type:
        raise MirrorError("a model type is required")
    if model_type not in registry:
        raise MirrorError(
            f"'{model_type}' is not in the checkpoint registry; "
            f"known types: {sorted(registry)[:6]}…"
        )
    out = [model_type]
    decoder = f"{model_type}{DECODER_SUFFIX}"
    if decoder in registry:
        out.append(decoder)
    return out


def parse_digest(value: Any) -> Tuple[str, str]:
    """``(algorithm, hex)`` for a registry digest, or raise.

    Fail-closed on anything unparseable. A digest that cannot be read is not a
    weaker check, it is no check, and the caller must not be able to proceed as
    though one happened.
    """
    if not isinstance(value, str):
        raise MirrorError(f"digest must be a string, got {type(value).__name__}")
    m = _DIGEST_RE.match(value.strip())
    if not m:
        raise MirrorError(
            f"unrecognised digest {value!r}; expected '<alg>:<hex>', "
            f"with alg one of {_SUPPORTED_ALGS}"
        )
    alg, hexd = m.group("alg"), m.group("hex")
    if alg not in _SUPPORTED_ALGS:
        raise MirrorError(f"unsupported digest algorithm {alg!r}")
    expected_len = {"xxh128": 32, "sha256": 64}[alg]
    if len(hexd) != expected_len:
        raise MirrorError(
            f"{alg} digest must be {expected_len} hex characters, got {len(hexd)}"
        )
    return alg, hexd


def parse_expectation(value: Any) -> Dict[str, str]:
    """``{algorithm: hex}`` for what a member is expected to hash to, or raise.

    A member carries TWO claims, and they are different claims:

    * upstream's **xxh128** is PROVENANCE -- these bytes are the ones upstream
      serves. It is the only digest upstream authors, and it can be checked while
      upstream is unreachable because it is a literal in micro-sam's source.
    * a **sha256** of our own is INTEGRITY -- the mirror has not changed since we
      put it there. It cannot speak to provenance: any corruption that arrived
      before we hashed it would be sealed in.

    Accepts either form, so a caller with one digest is not forced to invent the
    other:

    * ``"xxh128:<hex>"`` -- a single algorithm, the registry's own literal;
    * ``{"xxh128": "<hex>", "sha256": "<hex>"}`` -- both, each value bare hex or
      prefixed. The runtime check is against :class:`collections.abc.Mapping`
      rather than ``dict`` so that any mapping an RPC layer hands back is
      accepted, not only a real ``dict``.

      An earlier version of this comment asserted that hypha's ``dotdict`` fails
      ``isinstance(x, dict)``. **Measured here and it does not**: the installed
      ``hypha_rpc.utils.dotdict`` has MRO ``dotdict -> dict -> object``, so that
      check is True for it. The ``Mapping`` test is therefore the wider of the
      two by choice rather than by necessity, which is still the right way round
      for a value that crosses a transport -- but it is not fixing a known
      failure, and the claim that it was has been removed rather than left to be
      repeated.

    Fail-closed throughout, including on an EMPTY mapping: no recorded digest is
    not a weaker check, it is no check, and it must not be able to read as one.
    """
    if isinstance(value, str):
        alg, hexd = parse_digest(value)
        return {alg: hexd}
    if isinstance(value, _ABCMapping):
        if not value:
            raise MirrorError(
                "a member must record at least one digest; an empty expectation "
                "would verify nothing and report success"
            )
        out: Dict[str, str] = {}
        for alg, raw in value.items():
            if not isinstance(alg, str) or alg not in _SUPPORTED_ALGS:
                raise MirrorError(
                    f"unsupported digest algorithm {alg!r}; expected one of "
                    f"{_SUPPORTED_ALGS}"
                )
            if not isinstance(raw, str):
                raise MirrorError(
                    f"{alg} digest must be a string, got {type(raw).__name__}"
                )
            text = raw.strip()
            got_alg, hexd = parse_digest(text if ":" in text else f"{alg}:{text}")
            if got_alg != alg:
                raise MirrorError(
                    f"digest recorded under {alg!r} is labelled {got_alg!r}; "
                    "refusing rather than trusting one of the two"
                )
            out[alg] = hexd
        return out
    raise MirrorError(
        f"expectation must be '<alg>:<hex>' or a mapping of algorithm to hex, "
        f"got {type(value).__name__}"
    )


def digest_file(path: Any, alg: str, *, chunk: int = 1 << 20) -> str:
    """The hex digest of a file under *alg*, read in chunks.

    Chunked because these are checkpoints: the largest in the registry is about
    2.4 GB, and hashing it by reading it whole would put that in memory inside a
    Serve replica.

    `xxhash` is an optional dependency of this path. If it is absent this raises
    rather than falling back to another algorithm: silently hashing with
    something upstream did not publish would turn a provenance check into a
    self-consistency check, which is the one substitution that must not happen
    quietly.
    """
    if alg == "sha256":
        h = hashlib.sha256()
    elif alg == "xxh128":
        try:
            import xxhash  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MirrorError(
                "xxhash is not installed, so a registry xxh128 digest cannot be "
                "checked. Refusing rather than hashing with a different "
                "algorithm: upstream publishes no sha256, so that would prove "
                "only self-consistency."
            ) from exc
        h = xxhash.xxh128()
    else:
        raise MirrorError(f"unsupported digest algorithm {alg!r}")
    p = Path(path)
    with p.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


#: What :func:`verify_member` reports when a file is absent, so the caller sees
#: the same record shape either way and cannot mistake "absent" for "checked".
_ABSENT = "absent"


def verify_member(
    path: Any, digest: Any, *, allow_unverified_provenance: bool = False
) -> Dict[str, Any]:
    """The full result of checking one member, as a record rather than a bool.

    ``{"intact": bool, "checked": [alg…], "skipped": [alg…], "absent": bool}``.

    This exists because a bool cannot answer the question a caller actually has.
    A review measured that with `xxhash` absent, ``{"xxh128": WRONG, "sha256":
    right}`` returned True and nothing in the result said the provenance check
    had been skipped -- so "both digests verified" and "the xxh128 was silently
    dropped" were the same answer. ``skipped`` is how they differ now.

    *allow_unverified_provenance* has to be passed EXPLICITLY to accept a result
    whose provenance leg could not be computed. The default refuses, which keeps
    this consistent with :func:`digest_file`, which declines to hash with an
    algorithm upstream never published rather than substituting one. The ruling
    that the fallback verifies the sha256 is about the runtime download path;
    it does not say a missing hasher may waive the provenance claim, so the
    waiver is a decision a call site makes in writing.
    """
    expected = parse_expectation(digest)
    p = Path(path)
    record: Dict[str, Any] = {
        "intact": False,
        "checked": [],
        "skipped": [],
        "absent": False,
    }
    if not p.is_file():
        record["absent"] = True
        record["skipped"] = sorted(expected)
        return record

    mismatched = False
    for alg, want in sorted(expected.items()):
        try:
            got = digest_file(p, alg)
        except MirrorError:
            # Only an uncomputable algorithm lands here (no `xxhash`). Whether
            # that is survivable depends on what ELSE was recorded, which is
            # decided after the loop -- never inside it, where one missing
            # hasher would short-circuit a check that could still be made.
            record["skipped"].append(alg)
            continue
        record["checked"].append(alg)
        if got != want:
            mismatched = True

    if not record["checked"]:
        raise MirrorError(
            f"none of the recorded digests ({', '.join(sorted(expected))}) could "
            f"be computed here, so {p.name} was NOT verified. Refusing rather "
            "than reporting a result: install `xxhash`, or record a sha256 "
            "alongside the registry's xxh128 so there is something to fall back "
            "to."
        )

    # A MISMATCH IS REPORTED BEFORE A SKIPPED LEG IS REFUSED, and the order is a
    # fix rather than a style. The other way round, a file whose sha256 did not
    # match raised the provenance refusal instead -- and that message said the
    # file "matched [sha256]", which was false. A review measured it, and its
    # consequence in `seed_plan`: a provably corrupt file with two digests
    # recorded raised "PROVENANCE is unverified" rather than landing in
    # `replace` or `corrupt_unavailable`, so the one case the buckets exist to
    # separate became an exception instead.
    #
    # The principle: a mismatch is DECIDABLE without the missing hasher. What
    # the skipped leg costs is the ability to say a file is GOOD, never the
    # ability to say it is WRONG.
    if mismatched:
        record["intact"] = False
        return record

    if record["skipped"] and not allow_unverified_provenance:
        raise MirrorError(
            f"{p.name} matched {record['checked']}, and nothing it was checked "
            f"against disagreed, but {record['skipped']} could not be computed "
            "here, so its PROVENANCE is unverified. Refusing: a matching sha256 "
            "proves the mirror has not changed since we hashed it, not that the "
            "bytes are upstream's. Install `xxhash`, or pass "
            "allow_unverified_provenance=True at a call site where integrity "
            "alone is the documented standard."
        )
    record["intact"] = True
    return record


def member_is_intact(
    path: Any, digest: Any, *, allow_unverified_provenance: bool = False
) -> bool:
    """Whether the file at *path* matches what is recorded for it.

    The boolean face of :func:`verify_member`, for the callers that only branch
    on it. Anything that has to REPORT should use the record instead: this
    cannot express "it matched, but one of the two checks did not run".

    *digest* is anything :func:`parse_expectation` accepts: one algorithm, or
    both. **Every recorded algorithm that can be computed here must match** --
    a wrong xxh128 is not excused by a right sha256, because the two answer
    different questions and only one of them is about provenance.

    A missing file and a corrupt one both answer False, because the caller's
    action is identical -- place it again -- but they are NOT collapsed in the
    report: :func:`seed_plan` distinguishes them, so "the mirror is wrong" never
    looks like "the cache is cold".

    **What RAISES rather than answering.** A member none of whose digests can be
    computed: there is nothing left to check, and True would be a verification
    that never ran, while False would read as corruption and send a caller
    re-downloading good bytes from an origin that is down. And, unless
    *allow_unverified_provenance* says otherwise, a member whose xxh128 could
    not be computed even though its sha256 matched -- see :func:`verify_member`
    for why that is a refusal and not a pass.
    """
    return bool(
        verify_member(
            path, digest, allow_unverified_provenance=allow_unverified_provenance
        )["intact"]
    )


def seed_plan(
    cache_root: Any,
    wanted: Mapping[str, str],
    *,
    available: Optional[Iterable[str]] = None,
    allow_unverified_provenance: bool = False,
) -> Dict[str, List[str]]:
    """What to do for each wanted member, as four disjoint lists.

    *wanted* maps registry key -> expectation, where an expectation is anything
    :func:`parse_expectation` takes: the registry's ``"xxh128:<hex>"`` literal, or
    ``{"xxh128": ..., "sha256": ...}`` to carry provenance and integrity together.
    *available* is what the mirror holds;
    ``None`` means "assume everything is available", which is what a caller that
    has not listed the mirror yet should say.

    Returns ``{"ok", "place", "replace", "unavailable", "corrupt_unavailable"}``:

    * ``ok``          already present and matching -- do nothing, and in
                      particular do not re-download 2.4 GB to prove it again;
    * ``place``       absent locally, present in the mirror;
    * ``replace``     present locally but the digest does NOT match, and the
                      mirror has a copy. Kept separate from ``place`` because a
                      mismatch is a FINDING: it means something put a wrong file
                      in the cache, and a run that quietly overwrites it loses
                      the only evidence that happened;
    * ``unavailable`` needed, ABSENT locally, and the mirror does not have it.
                      The honest end state of a partial mirror: the caller must
                      let the normal upstream fetch try, not pretend the member
                      is handled;
    * ``corrupt_unavailable``
                      present locally, the digest does NOT match, and the mirror
                      cannot supply a replacement. Its own bucket because the
                      two facts need different actions: the caller still has to
                      fall back upstream, as for ``unavailable``, AND something
                      has put a wrong file in the cache, which is a finding
                      whatever the mirror holds.

                      This existed as a defect before it existed as a bucket. A
                      review measured that a corrupt file the mirror lacked came
                      back as plain ``unavailable`` -- indistinguishable from
                      "never downloaded" -- because the availability test ran
                      before the presence test. The docstring above already said
                      a mismatch must not be collapsed, and the code collapsed
                      it; the test covered only the absent case, so it stayed
                      green.

    The five are disjoint and their union is exactly *wanted*, so a caller can
    assert that and catch a member silently dropped by a future change.

    *allow_unverified_provenance* is passed straight through to
    :func:`verify_member`. It has to exist here because planning happens in the
    runtime, where `xxhash` may be absent: without it, a two-digest manifest
    would make every planning call raise rather than produce a plan. The default
    stays the refusing one, so a caller that wants a plan computed on integrity
    alone says so. **Nothing in this module asserts that `xxhash` is present
    where seeding runs** -- it is in pooch's dependency path in the runtime
    image, which is a reason to expect it, not a check that it is there.
    """
    have = None if available is None else set(available)
    plan: Dict[str, List[str]] = {
        "ok": [],
        "place": [],
        "replace": [],
        "unavailable": [],
        "corrupt_unavailable": [],
    }
    target = models_dir(cache_root)
    for key, digest in wanted.items():
        parse_expectation(digest)  # fail closed before any filesystem work
        path = target / key
        present = path.is_file()
        intact = (
            member_is_intact(
                path,
                digest,
                allow_unverified_provenance=allow_unverified_provenance,
            )
            if present
            else False
        )
        mirrored = have is None or key in have
        # PRESENCE IS TESTED BEFORE AVAILABILITY, and the order is the fix: the
        # other way round, a corrupt file the mirror lacks never reaches a
        # branch that can see it is present.
        if intact:
            plan["ok"].append(key)
        elif present:
            plan["replace" if mirrored else "corrupt_unavailable"].append(key)
        elif mirrored:
            plan["place"].append(key)
        else:
            plan["unavailable"].append(key)
    return plan


def verify_seeded(cache_root: Any, wanted: Mapping[str, str]) -> List[str]:
    """Which wanted members are NOT intact after seeding. Empty means all good.

    Returned as a list rather than a bool so a failure names the members. A
    caller that only needs the boolean can test emptiness; one that has to
    report cannot recover the names from a bool, and "the mirror failed" with no
    member named is not an actionable sentence.

    Raises if any member's provenance could not be checked, rather than counting
    it as good -- see :func:`verify_member`. Use :func:`verify_seeded_report` to
    get that as data instead of as an exception.
    """
    return [
        k
        for k, v in verify_seeded_report(cache_root, wanted).items()
        if not v["intact"]
    ]


def verify_seeded_report(
    cache_root: Any,
    wanted: Mapping[str, str],
    *,
    allow_unverified_provenance: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """:func:`verify_member` for every wanted member, keyed by registry key.

    The reporting face of the verification. A caller that has to say what it
    checked -- a seeding log, a mirror's own manifest, the evidence on an issue
    -- needs to distinguish "both digests matched" from "the integrity digest
    matched and the provenance one could not be computed here". The list
    :func:`verify_seeded` returns cannot carry that, and a report that cannot
    carry it is how an unverified provenance claim gets published as a verified
    one.
    """
    return {
        key: verify_member(
            models_dir(cache_root) / key,
            digest,
            allow_unverified_provenance=allow_unverified_provenance,
        )
        for key, digest in wanted.items()
    }
