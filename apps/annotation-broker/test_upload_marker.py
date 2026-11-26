"""The mint marker: the only thing that authorises touching a staged upload.

WHY A MARKER AT ALL, since the previous round believed a listing could answer
this. Enforcing a size ceiling after the bytes have landed means the finishing
call may have to DELETE the object, and the one thing it must never delete is an
image the dataset already had. Those two cases have to be distinguishable.

They are not distinguishable from the store. Measured against the live artifact
manager: an object PUT into an open stage is returned by `list_files(stage=True)`,
by `list_files` with `stage` unset, by `list_files(stage=False)`, by
`version="v0"` and by `version="latest"` alike. So the committed view does not
exclude the current request's upload, and the previous round's guard -- "refuse
if the name is in the committed listing" -- matches EVERY upload. It would have
refused every legitimate finish in production while still not seeing the case it
was written for. The test double of the day disagreed with the real store, which
is the only reason it passed.

So the fact is RECORDED when it is created rather than inferred afterwards. The
minting call writes a marker outside `images/`; the finishing call requires one.
The guarantee is a conjunction of two rules, and neither may be removed alone:

  * the mint refuses a filename that is already an image, so a marker is never
    written for an existing image's path;
  * the finish touches nothing without a marker for that exact path and mint id.

Together: an object the finish may delete is provably one this route minted, and
provably not a pre-existing image.

WHAT THESE TESTS DO NOT DO. They never assert on the marker document by reading
it back out of a fixture that also wrote it with hand-built fields -- every
marker here is built by the shipped `upload_marker_document` or written by the
shipped route. And the two arms that matter most are negative: a finish with no
marker, and a finish for an image that was already there.
"""

from __future__ import annotations

import ast
import json
import pathlib
import time

import pytest

import broker_core as core
from test_local_image_ceiling import _Store, _png, _store_broker
from test_local_image_upload_route import ARTIFACT


# ── the pure parts ─────────────────────────────────────────────────────────


def test_a_marker_path_is_built_only_from_a_VALID_mint_id():
    mint_id = core.new_mint_id()
    assert core.upload_marker_path(mint_id) == f".uploads/{mint_id}.json"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        None,
        12345,
        "../images/scan01.png",
        "..%2Fimages",
        "a/b",
        "/abs",
        "ABCDEF01234567890123456789012345",  # uppercase hex
        "0123456789abcdef",  # too short
        core.new_mint_id() + "0",  # too long
        core.new_mint_id() + "/../x",
        "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",  # right length, not hex
    ],
)
def test_a_mint_id_THAT_COULD_NAME_A_PATH_yields_NO_marker_path(bad):
    """The mint id becomes a filename, so an id that could express a separator
    or a parent reference would aim the marker read -- and therefore the discard
    decision -- at a path of the caller's choosing. None here means the route
    refuses, which is the same answer it gives for an unknown id."""
    assert core.upload_marker_path(bad) is None


def test_a_marker_records_the_path_the_ceiling_and_an_ABSOLUTE_expiry():
    mint_id = core.new_mint_id()
    doc = core.upload_marker_document(
        path="images/scan01.png", mint_id=mint_id, max_bytes=64, minted_at=1000.0
    )
    assert doc["path"] == "images/scan01.png"
    assert doc["mint_id"] == mint_id
    assert doc["max_bytes"] == 64
    assert doc["minted_at"] == 1000.0
    assert doc["expires_at"] == 1000.0 + core.UPLOAD_MARKER_TTL_S


def test_the_EXPIRY_is_judged_against_the_markers_OWN_recorded_instant():
    """Requirement 4, and the reason it is worded that way.

    On the production NFS filer the server's clock runs ~79 s ahead of the pods
    (measured). So `time.time()` compared against a stored `st_mtime` or a
    listing's `last_modified` compares two different clocks, and a marker
    written a moment ago can read as stale -- or as dated in the future -- with
    nothing in either value saying which clock set it. The marker carries an
    absolute `expires_at` written by the same clock that later reads it, so both
    sides of the comparison come from one source.
    """
    doc = core.upload_marker_document(
        path="images/x.png", mint_id=core.new_mint_id(), max_bytes=64, minted_at=0.0
    )
    ttl = core.UPLOAD_MARKER_TTL_S
    assert core.upload_marker_is_expired(doc, now=ttl - 1) is False
    assert core.upload_marker_is_expired(doc, now=ttl) is False  # boundary: not yet
    assert core.upload_marker_is_expired(doc, now=ttl + 1) is True
    # A clock skewed 79 s the WRONG way does not expire a fresh marker, because
    # the skew is not in this comparison at all.
    fresh = core.upload_marker_document(
        path="images/x.png",
        mint_id=core.new_mint_id(),
        max_bytes=64,
        minted_at=time.time(),
    )
    assert core.upload_marker_is_expired(fresh, now=time.time() + 78.6) is False


MARKER_FUNCTIONS = frozenset(
    {
        "new_mint_id",
        "upload_marker_path",
        "upload_marker_document",
        "parse_upload_marker",
        "upload_marker_is_expired",
        "marker_authorises_discard",
        "expired_marker_removals",
    }
)

FORBIDDEN_TIME_SOURCES = ("st_mtime", "last_modified", "getmtime", "stat")


def _file_time_reads(src: str):
    """``{function: {forbidden tokens}}`` for the marker functions in *src*.

    OVER THE AST, AND WITH THE DOCSTRING DROPPED, both for the same reason: the
    first version of this guard was a substring search over the source text, and
    it went red on the sentence in `upload_marker_document` that FORBIDS reading
    a file time. A guard that reads prose cannot tell an implementation from a
    comment about one -- and the failure direction is the dangerous one, because
    deleting the explanation would have made it pass.

    String constants are collected as well as attribute and name references.
    A file time can arrive as `os.stat(p).st_mtime`, as `entry["last_modified"]`
    or as `entry.get("last_modified")`, and only the first of those is an
    attribute; a check that looked at attributes alone would miss the subscript
    and the `.get`.
    """
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in MARKER_FUNCTIONS:
            continue
        body = list(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]  # the docstring is prose, not an implementation
        tokens = set()
        for stmt in body:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Attribute):
                    tokens.add(sub.attr)
                elif isinstance(sub, ast.Name):
                    tokens.add(sub.id)
                elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    tokens.add(sub.value)
        out[node.name] = {t for t in tokens if t in FORBIDDEN_TIME_SOURCES}
    return out


def test_nothing_in_the_marker_code_reads_a_FILE_TIME():
    """Requirement 4 as a mechanism rather than a docstring.

    An expiry judged by comparing this app's `time.time()` against a stored
    `st_mtime` or a listing's `last_modified` compares the pod clock with the
    filer clock, which were measured ~79 s apart on the production NFS mount.
    """
    found = _file_time_reads(pathlib.Path(core.__file__).read_text())
    missing = MARKER_FUNCTIONS - set(found)
    assert not missing, (
        f"the guard covered none of {sorted(missing)} -- a guard that inspects "
        "nothing reports clean, which looks identical to a pass"
    )
    assert {k: sorted(v) for k, v in found.items() if v} == {}, found


def test_THE_FILE_TIME_GUARD_CAN_GO_RED():
    """The control. A guard whose red branch is never exercised is a comment.

    Two mutants, because the two ways a file time can be read are different AST
    shapes and an attribute-only check would pass the second.
    """
    src = pathlib.Path(core.__file__).read_text()

    attribute_mutant = src.replace(
        '    return float(now) > float(marker["expires_at"])',
        "    return float(now) > os.stat(marker).st_mtime",
    )
    assert attribute_mutant != src, "the attribute mutant did not apply"
    assert _file_time_reads(attribute_mutant)["upload_marker_is_expired"] == {
        "stat",
        "st_mtime",
    }

    subscript_mutant = src.replace(
        '    return float(now) > float(marker["expires_at"])',
        '    return float(now) > float(marker["last_modified"])',
    )
    assert subscript_mutant != src, "the subscript mutant did not apply"
    assert _file_time_reads(subscript_mutant)["upload_marker_is_expired"] == {
        "last_modified"
    }


def test_THE_FILE_TIME_GUARD_IGNORES_PROSE():
    """And the other control, which is the bug this guard was born with.

    The shipped `upload_marker_document` docstring explains why a file time must
    not be used, naming `st_mtime` and `last_modified`. If the guard counted
    that, it would be red on correct code -- and green once somebody deleted the
    explanation.
    """
    src = pathlib.Path(core.__file__).read_text()
    assert "st_mtime" in src, "the explanation this control is about is gone"
    assert _file_time_reads(src)["upload_marker_document"] == set()


@pytest.mark.parametrize(
    "doc",
    [
        None,
        "not a document",
        {},
        {
            "marker_version": 999,
            "mint_id": "0" * 32,
            "path": "p",
            "max_bytes": 1,
            "expires_at": 1.0,
        },
        {"marker_version": 1, "path": "p", "max_bytes": 1, "expires_at": 1.0},
        {
            "marker_version": 1,
            "mint_id": "nope",
            "path": "p",
            "max_bytes": 1,
            "expires_at": 1.0,
        },
        {"marker_version": 1, "mint_id": "0" * 32, "max_bytes": 1, "expires_at": 1.0},
        {
            "marker_version": 1,
            "mint_id": "0" * 32,
            "path": "",
            "max_bytes": 1,
            "expires_at": 1.0,
        },
        {"marker_version": 1, "mint_id": "0" * 32, "path": "p", "max_bytes": 1},
        {
            "marker_version": 1,
            "mint_id": "0" * 32,
            "path": "p",
            "max_bytes": 1,
            "expires_at": "soon",
        },
        {
            "marker_version": 1,
            "mint_id": "0" * 32,
            "path": "p",
            "max_bytes": 1,
            "expires_at": True,
        },
        {
            "marker_version": 1,
            "mint_id": "0" * 32,
            "path": "p",
            "max_bytes": 0,
            "expires_at": 1.0,
        },
        {
            "marker_version": 1,
            "mint_id": "0" * 32,
            "path": "p",
            "max_bytes": True,
            "expires_at": 1.0,
        },
    ],
)
def test_an_UNUSABLE_marker_reads_as_NO_MARKER(doc):
    """Fail closed on every doubt: an unusable marker must authorise nothing, so
    the route refuses and touches nothing rather than deleting on the strength
    of a document it does not understand."""
    assert core.parse_upload_marker(doc) is None
    assert (
        core.marker_authorises_discard(doc, path="p", mint_id="0" * 32, now=0.0)
        is False
    )


def test_a_marker_authorises_ONLY_its_own_path_and_its_own_mint_id():
    mint_id = core.new_mint_id()
    other = core.new_mint_id()
    doc = core.upload_marker_document(
        path="images/scan01.png", mint_id=mint_id, max_bytes=64, minted_at=0.0
    )
    now = 1.0
    assert core.marker_authorises_discard(
        doc, path="images/scan01.png", mint_id=mint_id, now=now
    )
    assert not core.marker_authorises_discard(
        doc, path="images/other.png", mint_id=mint_id, now=now
    )
    assert not core.marker_authorises_discard(
        doc, path="images/scan01.png", mint_id=other, now=now
    )
    assert not core.marker_authorises_discard(
        doc,
        path="images/scan01.png",
        mint_id=mint_id,
        now=core.UPLOAD_MARKER_TTL_S + 1,
    )


def test_the_SWEEP_removes_expired_and_unusable_markers_and_SPARES_live_ones():
    live = core.upload_marker_document(
        path="images/a.png", mint_id=core.new_mint_id(), max_bytes=64, minted_at=0.0
    )
    dead = core.upload_marker_document(
        path="images/b.png", mint_id=core.new_mint_id(), max_bytes=64, minted_at=-9e9
    )
    pairs = [
        (".uploads/live.json", live),
        (".uploads/dead.json", dead),
        (".uploads/junk.json", {"nonsense": True}),
        (".uploads/none.json", None),
    ]
    out = core.expired_marker_removals(pairs, now=1.0)
    assert ".uploads/live.json" not in out, out
    assert sorted(out) == [
        ".uploads/dead.json",
        ".uploads/junk.json",
        ".uploads/none.json",
    ], out


def test_the_SWEEP_is_BOUNDED_so_minting_cannot_become_unbounded_work():
    dead = core.upload_marker_document(
        path="images/b.png", mint_id=core.new_mint_id(), max_bytes=64, minted_at=-9e9
    )
    pairs = [(f".uploads/{i}.json", dead) for i in range(200)]
    assert len(core.expired_marker_removals(pairs, now=1.0, limit=7)) == 7
    assert len(core.expired_marker_removals(pairs, now=1.0)) == (
        core.UPLOAD_MARKER_SWEEP_LIMIT
    )


# ── the route: the five arms the design asked for ──────────────────────────


@pytest.mark.asyncio
async def test_ARM1_an_oversized_PUT_through_the_REAL_minted_URL_is_discarded(
    monkeypatch,
):
    """Arm 1: and the stem never appears in the image listing afterwards.

    The listing is the assertion that matters. The dataset's image list is built
    by reconciling the manifest against the `images/` listing, so an object left
    in place becomes an image whatever this route's own commit does -- which is
    why "it raised" is not the test.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    mint = await b.upload_local_image(ARTIFACT, "scan01.png")
    store.put(mint["put_url"], _png(65))  # one byte over, real bytes
    assert "images/scan01.png" in store.objects, "the fixture did not store the PUT"

    with pytest.raises(ValueError) as exc:
        await b.upload_local_image(
            ARTIFACT, "scan01.png", done=True, mint_id=mint["mint_id"]
        )
    assert "65" in str(exc.value) and "64" in str(exc.value), str(exc.value)

    names = [e["name"] for e in store.listing("images")]
    assert "scan01.png" not in names, f"the oversized object is still listed: {names}"
    manifest = await b._image_manifest(ARTIFACT)
    assert "scan01" not in manifest, manifest
    assert store.committed == 0


@pytest.mark.asyncio
async def test_ARM2_an_EXISTING_image_over_the_ceiling_survives_EVERY_finish(
    monkeypatch,
):
    """Arm 2. The mounted-folder route has no cap, so an image already in the
    dataset can be over this route's ceiling -- and a finishing call naming it
    used to delete it, because the discard removed whatever sat at that path.

    Every way a client can get the finish wrong is tried, including a marker
    that is real but was minted for a DIFFERENT path, which is the one a
    path-blind marker check would wave through.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    store.publish("images/big.png", _png(100))  # committed, over this ceiling

    # A real, live, correctly-signed marker -- for another path entirely.
    elsewhere = await b.upload_local_image(ARTIFACT, "fresh.png")
    attempts = [
        {},
        {"mint_id": ""},
        {"mint_id": "not-a-mint-id"},
        {"mint_id": "../images/big"},
        {"mint_id": core.new_mint_id()},  # well-formed, no such marker
        {"mint_id": elsewhere["mint_id"]},  # live marker, WRONG path
    ]
    for kwargs in attempts:
        store.removed.clear()
        with pytest.raises(ValueError) as exc:
            await b.upload_local_image(ARTIFACT, "big.png", done=True, **kwargs)
        assert "NOTHING was changed" in str(exc.value), (kwargs, str(exc.value))
        assert store.removed == [], (kwargs, store.removed)
        assert "images/big.png" in store.objects, kwargs
        assert store.published.get("images/big.png"), kwargs

    assert store.committed == 0
    manifest = await b._image_manifest(ARTIFACT)
    assert "big" in manifest, f"the existing image is gone: {manifest}"


@pytest.mark.asyncio
async def test_ARM3_a_finish_WITHOUT_a_marker_touches_nothing(monkeypatch):
    """Arm 3, on a FRESH path rather than an existing image.

    Distinct from arm 2 on purpose: here the object at the path really is an
    unauthorised upload, and it is STILL left alone. "Touches nothing" means
    nothing -- not the object, not the stage, not the manifest -- because
    without a marker the route cannot know which case it is in.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    # An object at the upload path that this route never minted: a direct PUT,
    # or an upload whose marker has already been consumed or swept.
    store.objects["images/orphan.png"] = _png(65)

    with pytest.raises(ValueError) as exc:
        await b.upload_local_image(ARTIFACT, "orphan.png", done=True)
    msg = str(exc.value)
    assert "NOTHING was changed" in msg, msg
    assert "mint_id" in msg, "the refusal does not say what is missing"

    assert store.removed == [], store.removed
    assert store.objects["images/orphan.png"] == _png(65)
    assert store.committed == 0


@pytest.mark.asyncio
async def test_ARM3b_an_EXPIRED_marker_is_no_marker(monkeypatch):
    """The crashed-client case, which is what the expiry is for.

    The marker is aged by rewriting it with an older `minted_at` through the
    shipped builder -- NOT by moving a clock and not by touching a file time.
    Patching `time.time` would patch it for the manifest cache too, and a file
    time is the comparison this design exists to avoid.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    mint = await b.upload_local_image(ARTIFACT, "scan01.png")
    store.put(mint["put_url"], _png(65))

    marker_path = core.upload_marker_path(mint["mint_id"])
    aged = core.upload_marker_document(
        path="images/scan01.png",
        mint_id=mint["mint_id"],
        max_bytes=64,
        minted_at=time.time() - core.UPLOAD_MARKER_TTL_S - 10,
    )
    store.objects[marker_path] = json.dumps(aged).encode("utf-8")
    # A control on the ageing itself: the SAME document, un-aged, authorises.
    assert core.marker_authorises_discard(
        aged, path="images/scan01.png", mint_id=mint["mint_id"], now=aged["minted_at"]
    )
    assert not core.marker_authorises_discard(
        aged, path="images/scan01.png", mint_id=mint["mint_id"], now=time.time()
    )

    store.removed.clear()
    with pytest.raises(ValueError) as exc:
        await b.upload_local_image(
            ARTIFACT, "scan01.png", done=True, mint_id=mint["mint_id"]
        )
    assert "NOTHING was changed" in str(exc.value), str(exc.value)
    assert store.removed == [], store.removed
    assert store.objects["images/scan01.png"] == _png(65), "the object was touched"
    assert store.committed == 0


@pytest.mark.asyncio
async def test_ARM4_a_REMINT_after_an_oversized_discard_works(monkeypatch):
    """Arm 4: no `already: True` dead end.

    The discard is what makes this possible -- with the object left in place the
    re-mint reconciles it as an existing image and the user can never retry that
    name with a correct file. Carried through to a completed upload, because a
    re-mint that yields a URL the finish then refuses is not a working retry.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    first = await b.upload_local_image(ARTIFACT, "scan01.png")
    store.put(first["put_url"], _png(65))
    with pytest.raises(ValueError):
        await b.upload_local_image(
            ARTIFACT, "scan01.png", done=True, mint_id=first["mint_id"]
        )

    second = await b.upload_local_image(ARTIFACT, "scan01.png")
    assert second["already"] is False, f"the name is still taken: {second}"
    assert second["mint_id"] != first["mint_id"]

    store.put(second["put_url"], _png(32))
    out = await b.upload_local_image(
        ARTIFACT, "scan01.png", done=True, mint_id=second["mint_id"]
    )
    assert out["committed"] is True
    assert store.committed == 1
    manifest = await b._image_manifest(ARTIFACT)
    assert manifest.get("scan01", {}).get("state") == "local", manifest


# ── requirement 1: outside `images/`, and never an image ───────────────────


@pytest.mark.asyncio
async def test_the_marker_lives_OUTSIDE_images_and_is_NEVER_listed_as_one(
    monkeypatch,
):
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    mint = await b.upload_local_image(ARTIFACT, "scan01.png")
    marker_path = core.upload_marker_path(mint["mint_id"])
    assert marker_path in store.objects, sorted(store.objects)
    assert not marker_path.startswith("images/"), marker_path

    # It is in the artifact, and it is not an image.
    names = [e["name"] for e in store.listing("images")]
    assert all(core.UPLOAD_MARKER_DIR not in n for n in names), names
    manifest = await b._image_manifest(ARTIFACT)
    assert all(core.UPLOAD_MARKER_DIR not in k for k in manifest), manifest
    assert all(
        core.UPLOAD_MARKER_DIR not in (v or {}).get("path", "")
        for v in manifest.values()
    ), manifest


@pytest.mark.asyncio
async def test_a_mint_for_an_EXISTING_image_writes_NO_marker(monkeypatch):
    """Requirement 2, as the thing that makes requirement 3 safe.

    This is the load-bearing link: because no marker is ever minted for a path
    that is already an image, a marker is proof the object is not one. Pinned
    here rather than left as an argument in a comment.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    store.publish("images/big.png", _png(100))

    out = await b.upload_local_image(ARTIFACT, "big.png")

    assert out["already"] is True, out
    assert "mint_id" not in out and "put_url" not in out, out
    assert [
        p for p in store.objects if p.startswith(core.UPLOAD_MARKER_DIR)
    ] == [], f"a marker was minted for an existing image: {sorted(store.objects)}"


@pytest.mark.asyncio
async def test_a_mint_whose_MARKER_WRITE_FAILS_hands_out_NO_url(monkeypatch):
    """Otherwise the client holds a URL it can never discharge, and whatever it
    uploads becomes litter that nothing is allowed to clean up."""
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    async def _boom(_a, _path, _data):
        raise RuntimeError("the store refused the marker write")

    monkeypatch.setattr(b, "_write_json_file", _boom)

    with pytest.raises(RuntimeError):
        await b.upload_local_image(ARTIFACT, "scan01.png")
    assert store.objects == {}, store.objects


@pytest.mark.asyncio
async def test_a_SWEEP_FAILURE_does_not_cost_a_good_mint(monkeypatch):
    """The sweep is housekeeping and must not be load-bearing.

    The authorisation check treats an expired marker as no marker whether or not
    a sweep ever ran, so a sweep that cannot read the marker directory costs
    disk and nothing else. Driven by breaking the COLLABORATOR the sweep uses,
    not by replacing the sweep -- stubbing the sweep itself would measure the
    stub.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)
    # Broken at the STORE, not at `_list_files_safe`. The sweep reads the marker
    # directory through the paging loop, which calls `self._am.list_files`
    # directly -- so patching `_list_files_safe` stopped observing the sweep
    # when paging was introduced, and this test's own guard caught that by
    # reporting that the sweep never ran rather than passing quietly.
    real_list = store.list_files
    calls = []

    async def _breaks_on_the_marker_dir(
        artifact_id, dir_path=None, stage=True, limit=None, offset=0
    ):
        calls.append(dir_path)
        if dir_path == core.UPLOAD_MARKER_DIR:
            raise RuntimeError("the marker listing failed")
        return await real_list(
            artifact_id, dir_path=dir_path, stage=stage, limit=limit, offset=offset
        )

    monkeypatch.setattr(store, "list_files", _breaks_on_the_marker_dir)

    out = await b.upload_local_image(ARTIFACT, "scan01.png")

    assert (
        core.UPLOAD_MARKER_DIR in calls
    ), f"the sweep never ran, so this proves nothing: {calls}"
    assert out["already"] is False and out["mint_id"], out
    assert out["put_url"].endswith("images/scan01.png"), out
    assert core.upload_marker_path(out["mint_id"]) in store.objects


@pytest.mark.asyncio
async def test_the_SWEEP_really_removes_an_expired_marker_through_the_route(
    monkeypatch,
):
    """The other direction: the sweep is wired up and does something.

    Without this the test above is satisfied by a sweep that is never called at
    all, which is the same failure as a scan that never executed reporting zero
    findings.
    """
    store = _Store()
    b = _store_broker(monkeypatch, store, ceiling=64)

    stale_id = core.new_mint_id()
    stale_path = core.upload_marker_path(stale_id)
    store.objects[stale_path] = json.dumps(
        core.upload_marker_document(
            path="images/abandoned.png",
            mint_id=stale_id,
            max_bytes=64,
            minted_at=time.time() - core.UPLOAD_MARKER_TTL_S - 10,
        )
    ).encode("utf-8")

    fresh = await b.upload_local_image(ARTIFACT, "scan01.png")

    assert stale_path in store.removed, store.removed
    assert stale_path not in store.objects
    # The live marker this very call minted is NOT swept.
    assert core.upload_marker_path(fresh["mint_id"]) in store.objects
