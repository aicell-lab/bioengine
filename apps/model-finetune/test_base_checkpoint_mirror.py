"""The mirror fallback's hash fence.

The row's rule is "do NOT relax checksum verification", so these arms are about
REFUSAL: what the fence rejects, and that a rejected file never becomes visible
where the training subprocess would read it.

Offline by construction. The fence takes a file-like object, so a substituted
download is a `BytesIO` with the wrong bytes -- no network, no mirror, and the
arm that matters most (a substituted file) is the one that would otherwise need
somebody to tamper with a production artifact to test.
"""

from __future__ import annotations

import hashlib
import io
import os

import pytest

import base_checkpoint_mirror as bcm

# A name that is really in the pin table, so these arms exercise the real
# constants rather than a fixture's idea of them.
NAME = "vit_t_lm_decoder"
PINNED_XXH, PINNED_SHA, PINNED_SIZE = bcm.MIRROR_CHECKPOINTS[NAME]


def _bytes_hashing_to_pin():
    """Bytes of the pinned LENGTH whose sha256 is NOT the pinned digest.

    The size check must not be what rejects these, or the digest arm would
    never run -- so the length is correct on purpose.
    """
    body = b"\x00" * PINNED_SIZE
    assert hashlib.sha256(body).hexdigest() != PINNED_SHA
    return body


def test_the_PIN_TABLE_is_internally_consistent():
    """Every entry has a 32-hex xxh128, a 64-hex sha256 and a positive size.

    A pin table is the one place where a typo is invisible: a 63-character
    sha256 refuses every file, and the failure reads as a corrupt mirror.
    """
    assert bcm.MIRROR_CHECKPOINTS, "the pin table is empty"
    for name, (xxh, sha, size) in bcm.MIRROR_CHECKPOINTS.items():
        assert len(xxh) == 32, (name, xxh)
        assert len(sha) == 64, (name, sha)
        assert int(sha, 16) >= 0, (name, sha)  # it is hex
        assert size > 0, (name, size)
    # The decoder of every mirrored encoder is mirrored too: a site that holds
    # one without the other cannot train, which is what the probe's own
    # docstring warns about.
    for name in list(bcm.MIRROR_CHECKPOINTS):
        if name.endswith("_decoder"):
            assert name[: -len("_decoder")] in bcm.MIRROR_CHECKPOINTS, name


def test_a_SUBSTITUTED_file_is_REFUSED_and_never_appears_in_the_cache(tmp_path):
    """The arm the ruling asks for by name.

    Right length, wrong bytes -- a substituted file rather than a truncated
    one, so the digest is the only thing that can catch it.
    """
    root = str(tmp_path / "models")
    with pytest.raises(bcm.MirrorVerificationError) as exc:
        bcm.install_verified(root, NAME, io.BytesIO(_bytes_hashing_to_pin()))
    assert "does not match the pinned" in str(exc.value)
    # NOT present at the real name...
    assert not os.path.exists(os.path.join(root, NAME))
    # ...and no temp debris left behind for a later run to trip over.
    leftovers = os.listdir(root) if os.path.isdir(root) else []
    assert leftovers == [], leftovers


def test_a_TRUNCATED_file_is_REFUSED_by_the_size_check(tmp_path):
    """Short bytes, and the message must say so rather than blaming the digest.

    A truncated transfer is the likeliest real failure and the one whose digest
    mismatch tells the reader least; naming the length is what makes the log
    actionable.
    """
    root = str(tmp_path / "models")
    with pytest.raises(bcm.MirrorVerificationError) as exc:
        bcm.install_verified(root, NAME, io.BytesIO(b"\x01\x02\x03"))
    msg = str(exc.value)
    assert "bytes, pinned size is" in msg, msg
    assert not os.path.exists(os.path.join(root, NAME))


def test_a_file_we_do_not_MIRROR_is_REFUSED_rather_than_installed_unchecked(tmp_path):
    """No pin, no install. The dangerous reading of "not in the table" is
    "nothing to check against, so let it through"."""
    root = str(tmp_path / "models")
    with pytest.raises(bcm.MirrorVerificationError) as exc:
        bcm.install_verified(root, "vit_h_lm", io.BytesIO(b"whatever"))
    assert "not a mirrored checkpoint" in str(exc.value)
    assert not os.path.exists(os.path.join(root, "vit_h_lm"))


def test_a_file_that_VERIFIES_is_installed_and_reported(tmp_path):
    """THE CONTROL. Every arm above asserts a refusal, and a fence that refused
    EVERYTHING would pass all of them while making training impossible -- which
    is the failure mode a refusal-only suite cannot see.

    Uses a one-off pin entry rather than fabricating 44 MB of real weights.
    """
    root = str(tmp_path / "models")
    body = b"the quick brown fox" * 7
    name = "vit_fixture_lm"
    bcm.MIRROR_CHECKPOINTS[name] = (
        "0" * 32,
        hashlib.sha256(body).hexdigest(),
        len(body),
    )
    try:
        out = bcm.install_verified(root, name, io.BytesIO(body))
    finally:
        del bcm.MIRROR_CHECKPOINTS[name]
    dest = os.path.join(root, name)
    assert os.path.exists(dest)
    with open(dest, "rb") as fh:
        assert fh.read() == body
    assert out["sha256"] == hashlib.sha256(body).hexdigest()
    assert out["bytes"] == len(body)
    assert out["xxh128_provenance"] == "0" * 32
    assert os.listdir(root) == [name], "a temp file was left next to the result"


def test_check_digest_is_the_ONLY_fence_and_it_refuses_both_ways():
    """Called directly, because `install_verified` is not the only future caller
    and a second copy of this comparison is what this consolidation removed."""
    bcm.check_digest(NAME, PINNED_SHA, PINNED_SIZE)  # must NOT raise
    with pytest.raises(bcm.MirrorVerificationError):
        bcm.check_digest(NAME, "f" * 64, PINNED_SIZE)
    with pytest.raises(bcm.MirrorVerificationError):
        bcm.check_digest(NAME, PINNED_SHA, PINNED_SIZE - 1)


def test_missing_parts_reports_BOTH_parts_and_respects_what_is_cached(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    assert bcm.missing_parts(str(root), "vit_t_lm") == [
        "vit_t_lm",
        "vit_t_lm_decoder",
    ]
    (root / "vit_t_lm").write_bytes(b"x")
    assert bcm.missing_parts(str(root), "vit_t_lm") == ["vit_t_lm_decoder"]
    (root / "vit_t_lm_decoder").write_bytes(b"x")
    assert bcm.missing_parts(str(root), "vit_t_lm") == []


def test_an_UNMIRRORED_model_type_yields_no_parts_rather_than_raising(tmp_path):
    """ "We do not mirror this" is an answer. A KeyError here would turn a
    model type we simply have no fallback for into a crash on the admission
    path."""
    assert bcm.parts_for("vit_l_em_organelles") == []
    assert bcm.missing_parts(str(tmp_path), "vit_l_em_organelles") == []
