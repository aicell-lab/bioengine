"""Unit tests for the remote-NGFF import path in ``broker.py`` (0.10.0).

``broker.py`` imports ``bioengine`` and ``hypha_rpc``, which only exist inside
a Ray runtime, so it has never had unit tests. The §7 preflight is the piece
of this feature most likely to be quietly wrong -- it is the difference
between "this source is rejected at import with a reason" and "every image
open fails" -- so the two runtime imports are stubbed here and the real
methods are exercised against an in-memory Zarr store.

The stubs replace decorators with identity functions and nothing else. No
Ray, no Hypha, no network.
"""

from __future__ import annotations

import json
import logging
import sys
import types

import pytest
from pydantic.fields import FieldInfo

import broker_core as bc


def _resolving_method(**_kw):
    """Stand-in for ``bioengine.method``.

    The one thing it has to reproduce is default resolution: the parameters
    are declared as pydantic ``Field(...)``, and calling the raw function
    without them would pass a ``FieldInfo`` object straight into the body.
    The real runtime validates against the model first, so a test that did
    not do this would be testing a case production never sees.
    """
    import inspect

    def decorate(fn):
        params = inspect.signature(fn).parameters
        defaults = {
            name: p.default.default
            for name, p in params.items()
            if isinstance(p.default, FieldInfo)
        }

        async def call(self, *args, **kwargs):
            bound = inspect.signature(fn).bind_partial(self, *args, **kwargs)
            for name, value in defaults.items():
                if name not in bound.arguments:
                    kwargs[name] = value
            return await fn(self, *args, **kwargs)

        call.__name__ = fn.__name__
        return call

    return decorate


def _install_runtime_stubs() -> list:
    inserted = []
    if "bioengine" not in sys.modules:
        stub = types.ModuleType("bioengine")
        stub.logger = logging.getLogger("test-broker")
        stub.app = lambda **_kw: (lambda cls: cls)
        stub.method = _resolving_method
        stub.async_init = lambda fn: fn
        stub.health_check = lambda fn: fn
        sys.modules["bioengine"] = stub
        inserted.append("bioengine")
    if "hypha_rpc" not in sys.modules:
        stub = types.ModuleType("hypha_rpc")

        async def _connect(*_a, **_kw):  # pragma: no cover - never called
            raise AssertionError("tests must not connect to Hypha")

        stub.connect_to_server = _connect
        sys.modules["hypha_rpc"] = stub
        inserted.append("hypha_rpc")
    return inserted


_stubbed = _install_runtime_stubs()
import os

os.environ.setdefault("HYPHA_TOKEN", "test-token")

import broker as bk  # noqa: E402  (must follow the stubs)

# ``broker`` holds its own references now, so take the stubs back out. Leaving a
# fake ``bioengine`` in sys.modules breaks every other app suite collected into
# the same pytest process -- it imports the stub instead of the real package.
for _name in _stubbed:
    del sys.modules[_name]


# ---------------------------------------------------------------------------
# An in-memory Zarr store behind an httpx-shaped client
# ---------------------------------------------------------------------------

ROOT = "https://store.example/zarr/sample/"
ORIGIN = "https://bioimage.io"


class _Response:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}

    def json(self):
        if isinstance(self._body, (bytes, str)):
            return json.loads(self._body)
        if self._body is None:
            raise ValueError("no body")
        return self._body


class FakeHttp:
    """Serves a dict of ``{url: (status, body, headers)}``; anything else 404s.

    Records every URL it was asked for, so a test can assert *which* chunk the
    preflight chose rather than only that it succeeded.
    """

    def __init__(self, objects):
        self.objects = objects
        self.requested = []

    async def get(self, url, headers=None):
        self.requested.append(url)
        if url not in self.objects:
            return _Response(404)
        status, body, resp_headers = self.objects[url]
        return _Response(status, body, resp_headers)


CORS_OK = {"Access-Control-Allow-Origin": "*"}


def _zarray(shape, chunks, dtype="|u1", compressor=None, filters=None):
    return {
        "zarr_format": 2,
        "shape": list(shape),
        "chunks": list(chunks),
        "dtype": dtype,
        "order": "C",
        "fill_value": 0,
        "compressor": compressor if compressor is not None else {"id": "blosc"},
        "filters": filters,
    }


def _store(
    axes=("c", "y", "x"),
    shapes=((3, 600, 800), (3, 300, 400)),
    dtype="|u1",
    compressor=None,
    chunk_headers=CORS_OK,
    chunk_status=200,
    omero=None,
):
    """A conforming two-level v2 OME-Zarr store, with knobs for each way it
    can fail."""
    zattrs = {
        "multiscales": [
            {
                "version": "0.4",
                "axes": [{"name": a} for a in axes],
                "datasets": [{"path": str(i)} for i in range(len(shapes))],
            }
        ]
    }
    if omero:
        zattrs["omero"] = omero
    objects = {
        ROOT + ".zattrs": (200, zattrs, {}),
        ROOT + ".zgroup": (200, {"zarr_format": 2}, {}),
    }
    for i, shape in enumerate(shapes):
        chunks = [1] * (len(shape) - 2) + [256, 256]
        objects[f"{ROOT}{i}/.zarray"] = (
            200, _zarray(shape, chunks, dtype, compressor), {},
        )
        key = ".".join(["0"] * len(shape))
        objects[f"{ROOT}{i}/{key}"] = (chunk_status, b"\x00\x00", chunk_headers)
    return objects


def _broker(objects):
    broker = bk.AnnotationBroker()
    broker._http = FakeHttp(objects)
    return broker


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


class TestProbeNgffSource:
    async def test_accepts_a_conforming_v2_store(self):
        broker = _broker(_store())
        probe = await broker._probe_ngff_source(ROOT, None, ORIGIN)
        assert (probe["width"], probe["height"]) == (800, 600)
        assert probe["dtype"] == "|u1"
        assert probe["levels"] == 2
        assert probe["channels"] is None
        assert len(probe["content_hint"]) == 16

    async def test_dimensions_come_from_the_axes_not_the_shape_tail(self):
        # yxc, so width/height are NOT shape[-1]/shape[-2].
        broker = _broker(_store(axes=("y", "x", "c"), shapes=((600, 800, 3),)))
        probe = await broker._probe_ngff_source(ROOT, None, ORIGIN)
        assert (probe["width"], probe["height"]) == (800, 600)

    async def test_preflight_fetches_the_coarsest_level(self):
        # The point of the preflight is to be cheap; fetching level 0's chunk
        # of a whole-slide image would not be.
        broker = _broker(_store(shapes=((3, 4096, 4096), (3, 512, 512), (3, 64, 64))))
        await broker._probe_ngff_source(ROOT, None, ORIGIN)
        chunks = [
            u for u in broker._http.requested
            if "/.z" not in u and not u.endswith("zarr.json")
        ]
        assert chunks == [f"{ROOT}2/0.0.0"]

    async def test_coarsest_is_by_size_not_by_position(self):
        # A store that lists its levels finest-last.
        broker = _broker(_store(shapes=((3, 64, 64), (3, 4096, 4096))))
        await broker._probe_ngff_source(ROOT, None, ORIGIN)
        chunks = [
            u for u in broker._http.requested
            if "/.z" not in u and not u.endswith("zarr.json")
        ]
        assert chunks == [f"{ROOT}0/0.0.0"]
        # ...and level 0 by *pyramid* order still sets the dimensions.
        probe = await broker._probe_ngff_source(ROOT, None, ORIGIN)
        assert probe["width"] == 64

    async def test_reads_omero_channel_names(self):
        broker = _broker(_store(omero={"channels": [{"label": "DAPI"}, {"label": "GFP"}]}))
        probe = await broker._probe_ngff_source(ROOT, None, ORIGIN)
        assert probe["channels"] == ["DAPI", "GFP"]

    async def test_content_hint_is_stable_and_source_dependent(self):
        a = await _broker(_store())._probe_ngff_source(ROOT, None, ORIGIN)
        b = await _broker(_store())._probe_ngff_source(ROOT, None, ORIGIN)
        c = await _broker(
            _store(omero={"channels": [{"label": "DAPI"}]})
        )._probe_ngff_source(ROOT, None, ORIGIN)
        assert a["content_hint"] == b["content_hint"]
        assert a["content_hint"] != c["content_hint"]

    async def test_rejects_a_group_that_is_not_ome_zarr(self):
        objects = {ROOT + ".zattrs": (200, {"something": "else"}, {})}
        with pytest.raises(bc.NgffSourceError, match="multiscales"):
            await _broker(objects)._probe_ngff_source(ROOT, None, ORIGIN)

    async def test_rejects_a_root_with_no_zarr_metadata_at_all(self):
        with pytest.raises(bc.NgffSourceError, match="No Zarr metadata"):
            await _broker({})._probe_ngff_source(ROOT, None, ORIGIN)

    async def test_rejects_a_level_that_is_a_group_rather_than_an_array(self):
        # `.zattrs` present, `.zarray` absent: the path resolves to something,
        # but not to pixels.
        objects = _store()
        del objects[f"{ROOT}1/.zarray"]
        objects[f"{ROOT}1/.zattrs"] = (200, {}, {})
        with pytest.raises(bc.NgffSourceError, match="does not contain"):
            await _broker(objects)._probe_ngff_source(ROOT, None, ORIGIN)

    async def test_rejects_a_level_that_is_not_there_at_all(self):
        objects = _store()
        del objects[f"{ROOT}1/.zarray"]
        with pytest.raises(bc.NgffSourceError, match="No Zarr metadata"):
            await _broker(objects)._probe_ngff_source(ROOT, None, ORIGIN)

    async def test_rejects_an_undecodable_codec_and_names_it(self):
        broker = _broker(_store(compressor={"id": "jpeg2k"}))
        with pytest.raises(bc.NgffSourceError, match="jpeg2k"):
            await broker._probe_ngff_source(ROOT, None, ORIGIN)

    async def test_rejects_an_undecodable_filter(self):
        objects = _store()
        objects[f"{ROOT}0/.zarray"] = (
            200, _zarray((3, 600, 800), (1, 256, 256), filters=[{"id": "delta"}]), {},
        )
        with pytest.raises(bc.NgffSourceError, match="delta"):
            await _broker(objects)._probe_ngff_source(ROOT, None, ORIGIN)

    async def test_accepts_an_uncompressed_store(self):
        broker = _broker(_store(compressor=False))
        probe = await broker._probe_ngff_source(ROOT, None, ORIGIN)
        assert probe["width"] == 800

    async def test_rejects_a_bigint_dtype(self):
        broker = _broker(_store(dtype="<i8"))
        with pytest.raises(bc.NgffSourceError, match="64-bit"):
            await broker._probe_ngff_source(ROOT, None, ORIGIN)

    async def test_rejects_a_chunk_that_does_not_allow_our_origin(self):
        broker = _broker(_store(chunk_headers={}))
        with pytest.raises(bc.NgffSourceError, match="Access-Control-Allow-Origin: absent"):
            await broker._probe_ngff_source(ROOT, None, ORIGIN)

    async def test_rejects_a_chunk_that_allows_only_another_origin(self):
        broker = _broker(_store(chunk_headers={"Access-Control-Allow-Origin": "https://evil.example"}))
        with pytest.raises(bc.NgffSourceError, match="cross-origin"):
            await broker._probe_ngff_source(ROOT, None, ORIGIN)

    async def test_accepts_an_exact_origin_echo(self):
        broker = _broker(_store(chunk_headers={"Access-Control-Allow-Origin": ORIGIN}))
        assert (await broker._probe_ngff_source(ROOT, None, ORIGIN))["width"] == 800

    async def test_metadata_readable_but_data_not(self):
        broker = _broker(_store(chunk_status=403))
        with pytest.raises(bc.NgffSourceError, match="HTTP 403"):
            await broker._probe_ngff_source(ROOT, None, ORIGIN)

    async def test_a_chunk_that_is_simply_absent_is_reported_as_such(self):
        objects = _store()
        for key in [k for k in objects if k.endswith("0.0.0")]:
            del objects[key]
        with pytest.raises(bc.NgffSourceError, match="HTTP 404"):
            await _broker(objects)._probe_ngff_source(ROOT, None, ORIGIN)

    async def test_a_transport_failure_names_the_url(self):
        class Boom(FakeHttp):
            async def get(self, url, headers=None):
                raise ConnectionError("dns")

        broker = bk.AnnotationBroker()
        broker._http = Boom({})
        with pytest.raises(bc.NgffSourceError, match="Could not reach"):
            await broker._probe_ngff_source(ROOT, None, ORIGIN)


class TestProbeV3Store:
    def _objects(self, separator="/"):
        group = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "ome": {
                    "multiscales": [
                        {
                            "axes": [{"name": "y"}, {"name": "x"}],
                            "datasets": [{"path": "0"}],
                        }
                    ]
                }
            },
        }
        array = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [600, 800],
            "data_type": "uint16",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [256, 256]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": separator}},
            "codecs": [{"name": "bytes"}, {"name": "blosc"}],
            "fill_value": 0,
        }
        key = "c" + separator + separator.join(["0", "0"])
        return {
            ROOT + "zarr.json": (200, group, {}),
            f"{ROOT}0/zarr.json": (200, array, {}),
            f"{ROOT}0/{key}": (200, b"\x00", CORS_OK),
        }

    async def test_accepts_a_v3_store(self):
        probe = await _broker(self._objects())._probe_ngff_source(ROOT, None, ORIGIN)
        assert (probe["width"], probe["height"], probe["dtype"]) == (800, 600, "uint16")

    async def test_v3_chunk_key_uses_the_declared_separator(self):
        broker = _broker(self._objects(separator="."))
        await broker._probe_ngff_source(ROOT, None, ORIGIN)
        assert f"{ROOT}0/c.0.0" in broker._http.requested

    async def test_zarr_json_wins_over_a_stale_zattrs(self):
        objects = self._objects()
        objects[ROOT + ".zattrs"] = (200, {"multiscales": [{"datasets": [{"path": "old"}]}]}, {})
        probe = await _broker(objects)._probe_ngff_source(ROOT, None, ORIGIN)
        assert probe["levels"] == 1


# ---------------------------------------------------------------------------
# import_ngff_image
# ---------------------------------------------------------------------------


@pytest.fixture
def importer(tmp_path, monkeypatch):
    """A broker whose artifact I/O is in memory, so the test exercises the
    import logic rather than Hypha."""
    monkeypatch.setattr(bk, "STATE_ROOT", tmp_path)
    broker = _broker(_store())
    state = {
        "meta": {
            "artifact_id": "ws/ds",
            "owner": {"id": "u1"},
            "labels": [],
            "image_dims": {},
        },
        "manifest": None,
        "listing": [],
        "written": [],
        "committed": 0,
    }
    broker._state = state

    def _require_role(_context, _artifact_id, _minimum):
        return state["meta"], {"id": "u1"}, "owner"

    async def _read_json_file(_artifact_id, path, default=None):
        return state["manifest"] if path == bc.IMAGE_MANIFEST_PATH else default

    async def _list_files_safe(_artifact_id, _dir_path):
        return state["listing"]

    async def _write_json_file(_artifact_id, path, data):
        state["written"].append((path, data))
        if path == bc.IMAGE_MANIFEST_PATH:
            state["manifest"] = json.loads(json.dumps(data))

    async def _ensure_staged(_artifact_id, fn):
        return await fn()

    async def _commit(_artifact_id):
        state["committed"] += 1

    monkeypatch.setattr(broker, "_require_role", _require_role)
    monkeypatch.setattr(broker, "_read_json_file", _read_json_file)
    monkeypatch.setattr(broker, "_list_files_safe", _list_files_safe)
    monkeypatch.setattr(broker, "_write_json_file", _write_json_file)
    monkeypatch.setattr(broker, "_ensure_staged", _ensure_staged)
    monkeypatch.setattr(broker, "_commit_tolerating_pending_uploads", _commit)
    # write_metadata would otherwise touch the real state dir; tmp_path is
    # already wired above, but the record has to exist for a re-read.
    bc.write_metadata(state["meta"], root=tmp_path)
    return broker


class TestImportNgffImage:
    async def test_writes_a_remote_entry_and_commits(self, importer):
        out = await importer.import_ngff_image("ws/ds", ROOT, context=None)
        assert out["stem"] == "sample"
        assert out["state"] == "remote"
        assert (out["width"], out["height"]) == (800, 600)
        assert out["levels"] == 2
        assert "path" not in out

        stored = bc.parse_image_manifest(importer._state["manifest"])
        assert set(stored) == {"sample"}
        assert stored["sample"]["source"]["store_root"] == ROOT
        # A staged-only write disappears after a restage, so it must commit.
        assert importer._state["committed"] == 1

    async def test_derives_the_stem_from_the_store_root(self, importer):
        importer._http.objects = {
            k.replace(ROOT, "https://s3.example/bucket/my-sample.zarr/"): v
            for k, v in importer._http.objects.items()
        }
        out = await importer.import_ngff_image(
            "ws/ds", "https://s3.example/bucket/my-sample.zarr", context=None
        )
        assert out["stem"] == "my-sample"

    async def test_an_explicit_stem_wins(self, importer):
        out = await importer.import_ngff_image("ws/ds", ROOT, stem="chosen", context=None)
        assert out["stem"] == "chosen"

    async def test_rejects_a_stem_that_could_be_a_path(self, importer):
        for stem in ("../escape", "a/b", "with space", ".."):
            with pytest.raises(ValueError, match="Invalid image name"):
                await importer.import_ngff_image("ws/ds", ROOT, stem=stem, context=None)

    async def test_rejects_a_collision_with_an_existing_image(self, importer):
        importer._state["listing"] = [{"name": "sample.png", "type": "file"}]
        with pytest.raises(ValueError, match="already an image"):
            await importer.import_ngff_image("ws/ds", ROOT, context=None)

    async def test_a_second_import_keeps_the_first(self, importer):
        await importer.import_ngff_image("ws/ds", ROOT, stem="one", context=None)
        await importer.import_ngff_image("ws/ds", ROOT, stem="two", context=None)
        assert set(bc.parse_image_manifest(importer._state["manifest"])) == {"one", "two"}

    async def test_a_query_string_credential_is_refused_before_any_fetch(self, importer):
        with pytest.raises(bc.NgffSourceError, match="query string"):
            await importer.import_ngff_image("ws/ds", ROOT + "?token=abc", context=None)
        assert importer._http.requested == []
        assert importer._state["manifest"] is None

    async def test_a_failed_preflight_writes_nothing(self, importer):
        importer._http.objects = _store(compressor={"id": "jpeg2k"})
        with pytest.raises(bc.NgffSourceError):
            await importer.import_ngff_image("ws/ds", ROOT, context=None)
        assert importer._state["manifest"] is None
        assert importer._state["committed"] == 0


class TestImportARegion:
    """§13 step 6. The entry *is* the patch."""

    REGION = {"x": 100, "y": 50, "width": 256, "height": 128}

    async def test_the_entry_is_sized_to_the_patch(self, importer):
        out = await importer.import_ngff_image(
            "ws/ds", ROOT, region=self.REGION, context=None
        )
        # 800x600 store, 256x128 patch.
        assert (out["width"], out["height"]) == (256, 128)
        assert out["source"]["region"] == self.REGION

    async def test_no_region_is_still_the_whole_store(self, importer):
        out = await importer.import_ngff_image("ws/ds", ROOT, context=None)
        assert (out["width"], out["height"]) == (800, 600)
        assert out["source"]["region"] is None

    async def test_a_whole_store_region_is_indistinguishable_from_none(self, importer):
        whole = await importer.import_ngff_image(
            "ws/ds", ROOT, stem="a",
            region={"x": 0, "y": 0, "width": 800, "height": 600}, context=None,
        )
        assert whole["source"]["region"] is None

    async def test_the_far_edge_is_clamped_to_the_store(self, importer):
        out = await importer.import_ngff_image(
            "ws/ds", ROOT, region={"x": 700, "y": 500, "width": 400, "height": 400},
            context=None,
        )
        assert (out["width"], out["height"]) == (100, 100)

    async def test_a_region_outside_the_store_writes_nothing(self, importer):
        with pytest.raises(ValueError, match="outside"):
            await importer.import_ngff_image(
                "ws/ds", ROOT, region={"x": 900, "y": 0, "width": 8, "height": 8},
                context=None,
            )
        assert importer._state["manifest"] is None

    async def test_the_region_survives_a_manifest_round_trip(self, importer):
        await importer.import_ngff_image(
            "ws/ds", ROOT, stem="patch", region=self.REGION, context=None
        )
        stored = bc.parse_image_manifest(importer._state["manifest"])
        assert stored["patch"]["source"]["region"] == self.REGION


class TestProbeNgffSourceRpc:
    """The RPC around ``_probe_ngff_source`` (whose own tests are above).

    The dialog needs the slide's extent before it can offer a patch of it,
    and nothing is written to get it."""

    @pytest.fixture
    def prober(self, importer, monkeypatch):
        monkeypatch.setattr(
            importer, "_ctx_user", staticmethod(lambda _c: {"id": "u1", "email": None})
        )
        return importer

    async def test_reports_the_extent_and_writes_nothing(self, prober):
        out = await prober.probe_ngff_source(ROOT, context=None)
        assert (out["width"], out["height"]) == (800, 600)
        assert out["levels"] == 2
        assert out["store_root"] == ROOT
        assert out["stem"] == "sample"
        assert out["max_materialise_bytes"] == bc.MAX_MATERIALISE_BYTES
        assert prober._state["manifest"] is None
        assert prober._state["committed"] == 0

    async def test_the_credential_is_used_but_not_returned(self, prober):
        tokened = "https://ov.example/t/secret-123/zarr/ds1/"
        prober._http.objects = {k.replace(ROOT, tokened): v for k, v in _store().items()}
        out = await prober.probe_ngff_source(tokened, context=None)
        assert out["store_root"] == "https://ov.example/zarr/ds1/"
        assert all(u.startswith(tokened) for u in prober._http.requested)

    async def test_an_anonymous_caller_cannot_make_the_broker_fetch(self, importer):
        with pytest.raises(PermissionError):
            await importer.probe_ngff_source(
                ROOT, context={"user": {"id": "http-anonymous", "is_anonymous": True}}
            )
        assert importer._http.requested == []


class TestImportKeepsTheCredentialOutOfTheArtifact:
    """§12, decided 2026-09-10. The manifest lives in something ``set_public``
    can expose, so a token in it would leak to anyone with the dataset URL."""

    TOKENED = "https://ov.example/t/secret-123/zarr/ds1/"
    UNTOKENED = "https://ov.example/zarr/ds1/"

    @pytest.fixture
    def tokened(self, importer):
        importer._http.objects = {
            k.replace(ROOT, self.TOKENED): v for k, v in _store().items()
        }
        return importer

    async def test_the_token_is_used_to_fetch_but_not_stored(self, tokened):
        out = await tokened.import_ngff_image("ws/ds", self.TOKENED, stem="s", context=None)

        assert all(u.startswith(self.TOKENED) for u in tokened._http.requested)
        blob = json.dumps(tokened._state["manifest"])
        assert "secret-123" not in blob
        assert out["source"]["store_root"] == self.UNTOKENED

    async def test_the_token_lands_in_broker_metadata(self, tokened):
        await tokened.import_ngff_image("ws/ds", self.TOKENED, stem="s", context=None)
        creds = tokened._state["meta"]["ngff_credentials"]
        assert creds == {self.UNTOKENED: "secret-123"}

    async def test_an_untokened_source_records_no_credential(self, importer):
        await importer.import_ngff_image("ws/ds", ROOT, stem="s", context=None)
        assert "ngff_credentials" not in importer._state["meta"]


class TestGetImageUrl:
    @pytest.fixture
    def served(self, importer, monkeypatch):
        async def _get_file(artifact_id=None, file_path=None, stage=None):
            return f"https://signed.example/{file_path}?sig=1"

        importer._am = types.SimpleNamespace(get_file=_get_file)

        async def _image_entry(_artifact_id, stem):
            return importer._state["entries"][stem]

        monkeypatch.setattr(importer, "_image_entry", _image_entry)
        importer._state["entries"] = {}
        return importer

    async def test_a_local_image_keeps_the_pre_0_10_0_shape(self, served):
        served._state["entries"] = {"a": bc.local_image_entry("a.tif")}
        out = await served.get_image_url("ws/ds", "a", context=None)
        assert out["kind"] == "file"
        assert out["read_url"].startswith("https://signed.example/images/a.tif")
        assert out["stem"] == "a"

    async def test_a_remote_image_returns_the_store_root(self, served):
        served._state["entries"] = {
            "r": bc.remote_image_entry("https://x/zarr/ds/", "0", 800, 600)
        }
        out = await served.get_image_url("ws/ds", "r", context=None)
        assert out == {
            "stem": "r",
            "kind": "ngff",
            "store_root": "https://x/zarr/ds/",
            "multiscale_path": "0",
            "region": None,
        }

    async def test_a_patch_carries_its_region_to_the_reader(self, served):
        region = {"x": 10, "y": 20, "width": 512, "height": 384}
        served._state["entries"] = {
            "r": bc.remote_image_entry(
                "https://x/zarr/slide/", None, 512, 384, region=region
            )
        }
        out = await served.get_image_url("ws/ds", "r", context=None)
        # Without this the browser would open the whole slide and the viewer
        # would show the wrong pixels at the wrong size, which is exactly the
        # failure a region is supposed to prevent.
        assert out["region"] == region

    async def test_the_credential_is_spliced_in_after_the_role_check(self, served):
        served._state["entries"] = {
            "r": bc.remote_image_entry("https://ov.example/zarr/ds1/", None, 8, 8)
        }
        served._state["meta"]["ngff_credentials"] = {
            "https://ov.example/zarr/ds1/": "secret-123"
        }
        out = await served.get_image_url("ws/ds", "r", context=None)
        assert out["store_root"] == "https://ov.example/t/secret-123/zarr/ds1/"

    async def test_a_remote_image_whose_credential_is_gone_still_resolves(self, served):
        # Moving a dataset to another deployment loses the token; the source
        # is then unreadable, but that must surface as a failed fetch, not as
        # a broker error that hides the whole image.
        served._state["entries"] = {
            "r": bc.remote_image_entry("https://ov.example/zarr/ds1/", None, 8, 8)
        }
        out = await served.get_image_url("ws/ds", "r", context=None)
        assert out["store_root"] == "https://ov.example/zarr/ds1/"

    async def test_an_entry_with_neither_a_file_nor_a_source_raises(self, served):
        served._state["entries"] = {"broken": {"state": "local"}}
        with pytest.raises(FileNotFoundError, match="no file and no readable source"):
            await served.get_image_url("ws/ds", "broken", context=None)


class TestSplitsExcludeRemoteImages:
    @pytest.fixture
    def guard(self, importer, monkeypatch):
        async def _image_manifest(_artifact_id, **_kw):
            return importer._state["entries"]

        monkeypatch.setattr(importer, "_image_manifest", _image_manifest)
        importer._state["entries"] = {
            "local-a": bc.local_image_entry("local-a.tif"),
            "local-b": bc.local_image_entry("local-b.tif"),
            "remote-a": bc.remote_image_entry("https://x/", None, 8, 8),
        }
        return importer

    async def test_local_stems_pass(self, guard):
        await guard._validate_split_stems_are_images("ws/ds", ["local-a", "local-b"])

    async def test_a_remote_stem_is_refused_by_name(self, guard):
        with pytest.raises(ValueError, match="Not materialised.*remote-a"):
            await guard._validate_split_stems_are_images("ws/ds", ["local-a", "remote-a"])

    async def test_an_unknown_stem_still_reports_unknown_not_unmaterialised(self, guard):
        with pytest.raises(ValueError, match="Not images of this dataset: ghost"):
            await guard._validate_split_stems_are_images("ws/ds", ["ghost"])


class TestForgetRemoteImage:
    """An import with a typo in the URL has to be undoable. ``remote``
    entries are never pruned by ``sync_image_manifest``, so without this the
    row would be permanent."""

    async def test_drops_the_entry_and_commits(self, importer):
        await importer.import_ngff_image("ws/ds", ROOT, stem="gone", context=None)
        committed_before = importer._state["committed"]

        out = await importer.forget_remote_image("ws/ds", "gone", context=None)

        assert out == {"stem": "gone", "forgotten": True, "images": 0}
        assert bc.parse_image_manifest(importer._state["manifest"]) == {}
        assert importer._state["committed"] == committed_before + 1

    async def test_keeps_every_other_entry(self, importer):
        await importer.import_ngff_image("ws/ds", ROOT, stem="one", context=None)
        await importer.import_ngff_image("ws/ds", ROOT, stem="two", context=None)

        await importer.forget_remote_image("ws/ds", "one", context=None)

        assert set(bc.parse_image_manifest(importer._state["manifest"])) == {"two"}

    async def test_refuses_a_local_image(self, importer):
        importer._state["manifest"] = bc.build_image_manifest(
            {"a": bc.local_image_entry("a.tif")}
        )
        with pytest.raises(ValueError, match="local image"):
            await importer.forget_remote_image("ws/ds", "a", context=None)
        assert set(bc.parse_image_manifest(importer._state["manifest"])) == {"a"}

    async def test_an_unknown_stem_raises(self, importer):
        with pytest.raises(FileNotFoundError, match="not an image"):
            await importer.forget_remote_image("ws/ds", "ghost", context=None)

    async def test_a_local_file_absent_from_the_manifest_is_not_swept_in(self, importer):
        # The write-back is stored-minus-one, never a merge, so a file the
        # manifest has not caught up with must not gain an entry here.
        await importer.import_ngff_image("ws/ds", ROOT, stem="remote-one", context=None)
        importer._state["listing"] = [{"name": "late.png", "type": "file"}]

        await importer.forget_remote_image("ws/ds", "remote-one", context=None)

        assert bc.parse_image_manifest(importer._state["manifest"]) == {}
