"""The upload route itself, called for real — not just the rules it applies.

`test_local_image_upload.py` covers the pure stem/format/signature rules. A
review found that nothing covered the route's own no-overwrite check: removing
that check entirely left all 31 of those tests passing, because they never call
the method. These do.

The check was also wrong, and in a way only a call could show. It read the
saved `manifest.json` alone, but a dataset's image list is that manifest
reconciled against the `images/` listing — which is precisely why this route
can leave the manifest to the reconciliation, and therefore precisely why its
own earlier uploads are absent from the manifest it was reading. A name already
on disk read as free, and the mint handed out a URL that would overwrite it.

The runtime stubs come from `test_broker_ngff`, deliberately: that module
installs `bioengine`/`hypha_rpc` stubs unconditionally and then restores
`sys.modules` exactly as it found it, with a long comment explaining why both
halves matter. Re-implementing that here would be a second copy of subtle
logic that is already right, and getting the restore wrong poisons every other
app's suite in the same pytest process.
"""

from __future__ import annotations

import time

import pytest

import broker_core as core
from test_broker_ngff import bk


ARTIFACT = "ws/ds"


def _broker(monkeypatch, *, manifest=None, listing=None):
    """A real `AnnotationBroker` with storage and role checks faked out.

    Only the collaborators are replaced. `upload_local_image` itself is the
    shipped method, so a change to its logic changes what these tests see.
    """
    b = bk.AnnotationBroker()
    state = {
        "minted": [],
        "committed": 0,
        "manifest": manifest or {},
        "listing": listing or [],
    }

    monkeypatch.setattr(b, "_canonical_id", lambda a: a)
    monkeypatch.setattr(
        b,
        "_require_role",
        lambda context, artifact_id, minimum: ({}, {"id": "u1"}, "manager"),
    )

    async def _read_json_file(_a, _path, _default=None):
        return {
            "manifest_version": core.IMAGE_MANIFEST_VERSION,
            "images": state["manifest"],
        }

    async def _list_files_safe(_a, _dir):
        return state["listing"]

    class _Am:
        async def put_file(self, _a, file_path=None):
            state["minted"].append(file_path)
            return f"https://signed.invalid/{file_path}"

    async def _ensure_staged(_a, fn):
        return await fn()

    async def _commit(_a):
        state["committed"] += 1

    async def _object_head(_a, _path, _n):
        return core.PNG_SIGNATURE

    monkeypatch.setattr(b, "_read_json_file", _read_json_file)
    monkeypatch.setattr(b, "_list_files_safe", _list_files_safe)
    monkeypatch.setattr(b, "_am", _Am())
    monkeypatch.setattr(b, "_ensure_staged", _ensure_staged)
    monkeypatch.setattr(b, "_commit_tolerating_pending_uploads", _commit)
    monkeypatch.setattr(b, "_object_head", _object_head)
    b._tstate = state
    return b


# ── the defect the review found ─────────────────────────────────────────────


async def test_a_file_ALREADY_IN_THE_LISTING_is_not_handed_an_overwriting_URL(
    monkeypatch,
):
    """THE case. `images/scan01.png` on disk, manifest empty.

    This route does not write the manifest, so its own earlier uploads are
    never in it. Reading the manifest alone reported the name free and minted a
    URL for that very path, so the second upload replaced the first image and
    orphaned any annotations drawn on it.
    """
    b = _broker(
        monkeypatch, manifest={}, listing=[{"name": "scan01.png", "type": "file"}]
    )

    out = await b.upload_local_image(ARTIFACT, "scan01.png")

    assert out["already"] is True, out
    assert "put_url" not in out, "a URL was minted for a name already on disk"
    assert b._tstate["minted"] == [], b._tstate["minted"]


async def test_the_SAME_STEM_under_a_different_extension_is_refused(monkeypatch):
    """`scan01.tif` on disk, `scan01.png` uploaded.

    The stem is the manifest key, so the two collapse into ONE entry. Depending
    on which extension the reconciliation keeps, either the new image never
    appears or it silently replaces the one the annotations belong to.
    """
    b = _broker(
        monkeypatch, manifest={}, listing=[{"name": "scan01.tif", "type": "file"}]
    )

    out = await b.upload_local_image(ARTIFACT, "scan01.png")

    assert out["already"] is True, out
    assert b._tstate["minted"] == []


async def test_a_name_in_the_MANIFEST_is_still_refused(monkeypatch):
    """The half that already worked. Kept so a fix cannot trade one for the other."""
    b = _broker(
        monkeypatch,
        manifest={"scan01": core.local_image_entry("scan01.png")},
        listing=[],
    )

    out = await b.upload_local_image(ARTIFACT, "scan01.png")

    assert out["already"] is True, out
    assert b._tstate["minted"] == []


async def test_A_FREE_NAME_STILL_GETS_A_URL(monkeypatch):
    """THE CONTROL on all three refusals above.

    A check that refused everything would satisfy them and make the route
    useless. This is the case that must still work.
    """
    b = _broker(
        monkeypatch,
        manifest={"other": core.local_image_entry("other.png")},
        listing=[{"name": "other.png", "type": "file"}],
    )

    out = await b.upload_local_image(ARTIFACT, "scan01.png")

    assert out["already"] is False, out
    assert out["path"] == "images/scan01.png"
    assert out["put_url"].endswith("images/scan01.png")
    assert out["max_bytes"] == core.MAX_LOCAL_IMAGE_BYTES
    assert b._tstate["minted"] == ["images/scan01.png"]


async def test_the_manifest_is_ignored_when_the_LISTING_disagrees_with_it(monkeypatch):
    """Why the fix reconciles rather than reading either source alone.

    A stem in the manifest whose file is gone is still 'taken' here: pruning is
    the manifest writer's job, and treating it as free would hand out a URL
    under a name readers may still be resolving.
    """
    b = _broker(
        monkeypatch,
        manifest={"scan01": core.local_image_entry("scan01.png")},
        listing=[],  # the file is not there
    )

    out = await b.upload_local_image(ARTIFACT, "scan01.png")

    assert out["already"] is True, out


# ── the stale cache the review also found ──────────────────────────────────


async def test_FINISHING_an_upload_drops_the_cached_image_list(monkeypatch):
    """Otherwise the caller is told the upload succeeded and cannot see it.

    Every other route that adds or removes an image refreshes this cache. This
    one did not, so `get_dataset_index` could serve a list computed up to
    `MANIFEST_CACHE_TTL_S` ago — without the image just uploaded.
    """
    # The listing carries the uploaded object WITH its size, which is what a
    # staged listing really looks like after a PUT lands (measured: a staged
    # listing reports `{"type": "file", "name": ..., "size": N, ...}`). It used
    # to be `[]` here, which says the opposite -- that nothing was uploaded --
    # and the finishing call now refuses that, correctly: see
    # test_finishing_with_NOTHING_IN_THE_LISTING_is_refused below.
    b = _broker(
        monkeypatch,
        manifest={},
        listing=[{"name": "scan01.png", "type": "file", "size": 2048}],
    )
    b._image_manifest_cache[ARTIFACT] = (time.time(), {"stale": {}})

    out = await b.upload_local_image(ARTIFACT, "scan01.png", done=True)

    assert out["committed"] is True
    assert ARTIFACT not in b._image_manifest_cache, "the stale image list survived"
    assert b._tstate["committed"] == 1


async def test_finishing_still_REFUSES_bytes_that_are_not_the_declared_format(
    monkeypatch,
):
    """The signature check must survive the cache fix, not be traded for it."""
    # A listing with a plausible size, so this test measures the SIGNATURE
    # check rather than tripping the size check that now runs before it.
    b = _broker(
        monkeypatch,
        manifest={},
        listing=[{"name": "scan01.png", "type": "file", "size": 2048}],
    )

    async def _bad_head(_a, _path, _n):
        return b"<!DOCTYPE html>"[:8]

    monkeypatch.setattr(b, "_object_head", _bad_head)

    with pytest.raises(ValueError) as e:
        await b.upload_local_image(ARTIFACT, "scan01.png", done=True)

    assert "not a PNG" in str(e.value)
    assert b._tstate["committed"] == 0, "it committed despite refusing"
