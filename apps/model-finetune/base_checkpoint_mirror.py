"""Seed micro-sam's base-checkpoint cache from our mirror, under a hash fence.

WHY THIS EXISTS. `train_sam` with no `checkpoint_path` resolves its base model
through micro-sam's own pooch registry, which downloads from
`uk1s3.embassy.ebi.ac.uk`. That host has been refusing connections on 443 from
every site we run on, so training cannot start at all where the file is not
already cached -- measured on KTH, and later from the de.NBI jump host too. The
mirror is a Hypha artifact both clusters can reach.

THE FENCE IS NOT OPTIONAL, and it is the reason this is a module rather than
three lines. A mirror is a second place for weights to come from, so it is also
a second place for the wrong weights to come from. Every file is verified
against a digest **pinned in this repository** before it is put where micro-sam
will read it, and a mismatch REFUSES rather than falling back to "well, it
downloaded". Nothing here relaxes a checksum.

TWO DIGESTS, TWO DIFFERENT JOBS, and conflating them is the trap:

* ``xxh128`` is UPSTREAM's, copied from `micro_sam/v1/util.py`'s registry
  literals. It is *provenance*: evidence that the bytes we mirrored are the
  bytes micro-sam publishes. It is recorded here, and enforced by POOCH
  rather than by this module. pooch
  hash-checks a file ALREADY IN THE CACHE, so upstream's digest gates every USE
  of a seeded file and not merely the fetches pooch performs itself -- measured
  four ways, including that a wrong cached file with the origin DOWN makes pooch
  RAISE rather than hand the bytes over. An earlier version of this comment said
  pooch checks it "whenever it is the one fetching", which implied a seeded file
  sails past until pooch happens to download. The opposite is true, and it is
  what makes the sha256 below a genuine SECOND line rather than the only one.
* ``sha256`` is OURS, measured over the mirrored bytes. It is *integrity*: it
  catches a truncated transfer, a corrupted object, or a substituted file. This
  is what the fence enforces.

The size is pinned and checked too. A truncated transfer whose prefix happens
to be shared is the case a digest comparison is least likely to be read
carefully for, and the length is free to check.

ANONYMOUS ON PURPOSE. The mirror is world-readable, verified with a ranged GET
from an anonymous connection. So the fallback needs no credential -- which
matters because an app the worker *adopted* rather than deployed stores no
token, and that is exactly the state a worker is in after the restart where
recovery is most needed. A fallback that required a credential would fail in
the one case it exists for.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Dict, Optional, Tuple

#: The artifact and the committed version the fallback reads. PINNED: a
#: floating version would let the bytes under a fixed digest change, which is
#: the whole thing the digest is for.
MIRROR_ARTIFACT_ID = "bioimage-io/micro-sam-checkpoint-mirror"
MIRROR_VERSION = "v0"

#: The micro-sam release whose registry literals the xxh128 values below were
#: copied from. Recorded because "upstream's digest" is only a provenance claim
#: if you can say WHICH upstream: micro-sam may republish a model under the same
#: registry key, and then a future reader comparing our xxh128 against a newer
#: installed micro-sam would get a mismatch with no way to tell a corrupted
#: mirror from a moved upstream. The mirror artifact records the same version,
#: and the cross-check asserts the two agree.
MIRROR_MICRO_SAM_VERSION = "1.8.11"

#: name -> (upstream xxh128 provenance, our sha256 integrity, exact size)
#:
#: The sha256 and the size were measured over the MIRROR's own bytes, streamed
#: and hashed, and they agree with the digests measured independently in the
#: de.NBI replica cache. Both ends, so this is not a constant copied from a
#: comment: a wrong constant here refuses every fallback and the failure reads
#: as a corrupt mirror rather than as a typo.
MIRROR_CHECKPOINTS: Dict[str, Tuple[str, str, int]] = {
    "vit_t_lm": (
        "72ec5074774761a6e5c05a08942f981e",
        "32b92ab64959088878a1cc3e52b9b64aaa0ba4b24c98e9959746d7390bec7b7a",
        40677198,
    ),
    "vit_t_lm_decoder": (
        "3e914a5f397b0312cdd36813031f8823",
        "95279735430ceb97026d5fdd127cf889205f784b0dd9927fe337fb342817ede6",
        44143988,
    ),
    "vit_b_lm": (
        "fe9252a29f3f4ea53c15a06de471e186",
        "ffe862fa5b58d78375506041faf0eb3927ced5b5391e8be383e6d7ac07aed924",
        375023499,
    ),
    "vit_b_lm_decoder": (
        "708b15ac620e235f90bb38612c4929ba",
        "3a1566aada52e24299286527ec60c966d213db86ee656c67439c71c1b4182c26",
        38393613,
    ),
}

#: Read in 1 MiB chunks and hashed as it arrives, so a 375 MB encoder is never
#: held in memory and never lands anywhere but its own temp file.
_CHUNK = 1 << 20


class MirrorVerificationError(RuntimeError):
    """A mirrored file did not match its pinned digest or size.

    Deliberately NOT a subclass of anything the fetch path catches as
    "download failed". A failed download is a reason to try elsewhere; a file
    that arrived and hashed WRONG is a reason to stop, because the next thing
    that happens otherwise is training on unknown weights.
    """


def parts_for(model_type: str) -> list:
    """The mirrored parts a run of *model_type* needs, encoder first.

    BOTH parts matter and a site can hold one without the other, so a caller
    that checked only the encoder would admit a run whose decoder is
    unobtainable -- the same mistake `base_checkpoint_status` documents.
    Returns only parts the mirror actually holds; a model type we do not
    mirror yields an empty list, which is a truthful "no fallback available"
    rather than a KeyError.
    """
    return [
        name
        for name in (model_type, f"{model_type}_decoder")
        if name in MIRROR_CHECKPOINTS
    ]


def missing_parts(cache_root: str, model_type: str) -> list:
    """Mirrored parts for *model_type* not already present in *cache_root*.

    PRESENT, NOT VALID -- this is `os.path.exists`, matching what micro-sam
    itself treats as cached. A file already on disk is left alone: re-hashing
    hundreds of megabytes on every run would be a cost this path cannot carry,
    and `pooch` re-downloads a file whose own hash disagrees anyway.
    """
    return [
        name
        for name in parts_for(model_type)
        if not os.path.exists(os.path.join(cache_root, name))
    ]


def check_digest(name: str, got_sha: str, got_size: int) -> None:
    """THE fence. Raises `MirrorVerificationError` unless *name* matches its pin.

    One function, called from every path that installs a mirrored file. An
    earlier draft of this module had the comparison written out twice -- once
    for a verify helper and once in the installer -- which is two fences that
    can drift apart, and a fence that is right in one copy and wrong in the
    other is worse than one fence, because the passing copy is the one you
    will read.
    """
    if name not in MIRROR_CHECKPOINTS:
        raise MirrorVerificationError(
            f"{name!r} is not a mirrored checkpoint, so there is no pinned "
            f"digest to verify it against; refusing to install it"
        )
    _, want_sha, want_size = MIRROR_CHECKPOINTS[name]
    if got_size != want_size:
        raise MirrorVerificationError(
            f"{name}: mirrored file is {got_size} bytes, pinned size is "
            f"{want_size} -- refusing a truncated or substituted file"
        )
    if got_sha != want_sha:
        raise MirrorVerificationError(
            f"{name}: sha256 {got_sha} does not match the pinned {want_sha} "
            f"-- refusing to install it"
        )


def hash_stream(fh) -> Tuple[str, int]:
    """``(sha256_hex, bytes_read)`` over *fh*, in chunks. No verification."""
    h = hashlib.sha256()
    size = 0
    while True:
        chunk = fh.read(_CHUNK)
        if not chunk:
            break
        h.update(chunk)
        size += len(chunk)
    return h.hexdigest(), size


def install_verified(cache_root: str, name: str, body) -> Dict[str, object]:
    """Write *body* into *cache_root* as *name*, but ONLY if it verifies.

    *body* is any object with ``.read(n)``.

    THE ORDER IS THE POINT. The bytes go to a temp file in the SAME directory,
    are hashed there, and are moved into place with `os.replace` only after the
    digest and the size both match. So a file that fails the fence is never
    visible at the path micro-sam reads, not even briefly -- which matters
    because the reader here is a separate training subprocess that could pick
    up a half-written file at any moment. Writing first and checking afterwards
    would be a fence with a hole in it exactly the width of the download.

    The temp file is removed on every failure path, so a refused download does
    not accumulate.
    """
    os.makedirs(cache_root, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{name}.", suffix=".part", dir=cache_root)
    try:
        with os.fdopen(fd, "wb") as out:
            h = hashlib.sha256()
            size = 0
            while True:
                chunk = body.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                size += len(chunk)
        got = h.hexdigest()
        # The one fence, before the file becomes visible at its real name.
        check_digest(name, got, size)
        dest = os.path.join(cache_root, name)
        os.replace(tmp, dest)
        tmp = None
        return {
            "name": name,
            "sha256": got,
            "bytes": size,
            "xxh128_provenance": MIRROR_CHECKPOINTS[name][0],
            "path": dest,
        }
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)
