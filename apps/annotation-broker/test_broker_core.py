"""Unit tests for ``broker_core.py``.

Plain pytest, no Ray, no live Hypha connection. Metadata read/write tests
point ``root`` at ``tmp_path`` so nothing touches the real
``~/annotation_broker`` state directory.
"""

from __future__ import annotations

import json
import time

import pytest

import broker_core as bc


# ---------------------------------------------------------------------------
# Role resolution
# ---------------------------------------------------------------------------


def _meta(**overrides):
    base = {
        "artifact_id": "bioimage-io/annotation-abc123",
        "owner": {"id": "owner-1", "email": "owner@example.com"},
        "managers": [{"id": "mgr-1"}],
        "annotators": [{"email": "annotator@example.com"}],
        "public": False,
        "labels": [],
    }
    base.update(overrides)
    return base


class TestResolveRole:
    def test_owner_by_id(self):
        assert bc.resolve_role(_meta(), "owner-1", None) == "owner"

    def test_owner_by_email(self):
        assert bc.resolve_role(_meta(), None, "owner@example.com") == "owner"

    def test_owner_email_case_insensitive(self):
        assert bc.resolve_role(_meta(), None, "Owner@Example.com") == "owner"

    def test_manager_by_id(self):
        assert bc.resolve_role(_meta(), "mgr-1", None) == "manager"

    def test_annotator_by_email(self):
        assert bc.resolve_role(_meta(), None, "annotator@example.com") == "annotator"

    def test_unknown_user_private_dataset_is_none(self):
        assert bc.resolve_role(_meta(), "stranger", "stranger@example.com") == "none"

    def test_public_dataset_logged_in_user_is_annotator(self):
        meta = _meta(public=True)
        assert bc.resolve_role(meta, "some-user-id", None) == "annotator"

    def test_public_dataset_anonymous_user_is_public(self):
        meta = _meta(public=True)
        assert bc.resolve_role(meta, None, None) == "public"
        assert bc.resolve_role(meta, "anonymous", None) == "public"

    def test_private_dataset_anonymous_user_is_none(self):
        assert bc.resolve_role(_meta(), None, None) == "none"

    def test_missing_metadata_is_none(self):
        assert bc.resolve_role(None, "owner-1", "owner@example.com") == "none"

    def test_owner_outranks_also_listed_manager(self):
        # Owner id is never expected to also appear in managers/annotators,
        # but if it did, ownership must still win.
        meta = _meta(managers=[{"id": "owner-1"}])
        assert bc.resolve_role(meta, "owner-1", None) == "owner"


class TestRoleAtLeast:
    @pytest.mark.parametrize(
        "role,minimum,expected",
        [
            ("owner", "owner", True),
            ("owner", "manager", True),
            ("owner", "annotator", True),
            ("manager", "owner", False),
            ("manager", "manager", True),
            ("annotator", "manager", False),
            ("public", "annotator", False),
            ("public", "public", True),
            ("none", "public", False),
            ("none", "none", True),
        ],
    )
    def test_ordering(self, role, minimum, expected):
        assert bc.role_at_least(role, minimum) is expected


class TestSetRemoveUserRole:
    def test_set_manager_appends(self):
        meta = _meta(managers=[], annotators=[])
        bc.set_user_role(meta, {"id": "u1", "email": "u1@example.com"}, "manager")
        assert meta["managers"] == [{"id": "u1", "email": "u1@example.com"}]
        assert meta["annotators"] == []

    def test_set_role_moves_between_lists(self):
        meta = _meta(managers=[], annotators=[{"id": "u1"}])
        bc.set_user_role(meta, {"id": "u1"}, "manager")
        assert meta["managers"] == [{"id": "u1"}]
        assert meta["annotators"] == []

    def test_set_role_is_idempotent(self):
        meta = _meta(managers=[], annotators=[])
        bc.set_user_role(meta, {"id": "u1"}, "annotator")
        bc.set_user_role(meta, {"id": "u1"}, "annotator")
        assert meta["annotators"] == [{"id": "u1"}]

    def test_set_role_matches_by_email_when_id_absent(self):
        meta = _meta(managers=[], annotators=[{"email": "u1@example.com"}])
        bc.set_user_role(meta, {"email": "u1@example.com"}, "manager")
        assert meta["managers"] == [{"email": "u1@example.com"}]
        assert meta["annotators"] == []

    def test_invalid_role_raises(self):
        meta = _meta()
        with pytest.raises(ValueError):
            bc.set_user_role(meta, {"id": "u1"}, "owner")

    def test_remove_user_role(self):
        meta = _meta(managers=[{"id": "u1"}], annotators=[{"id": "u2"}])
        bc.remove_user_role(meta, {"id": "u1"})
        assert meta["managers"] == []
        assert meta["annotators"] == [{"id": "u2"}]

    def test_remove_user_role_no_match_is_noop(self):
        meta = _meta(managers=[{"id": "u1"}], annotators=[])
        bc.remove_user_role(meta, {"id": "nobody"})
        assert meta["managers"] == [{"id": "u1"}]


# ---------------------------------------------------------------------------
# Path building
# ---------------------------------------------------------------------------


class TestSanitizeUserId:
    def test_basic(self):
        assert bc.sanitize_user_id("alice") == "user-alice"

    def test_pipe_and_digits(self):
        assert bc.sanitize_user_id("github|49943582") == "user-github-49943582"

    def test_none_falls_back_to_anonymous(self):
        assert bc.sanitize_user_id(None) == "user-anonymous"

    def test_empty_string_falls_back_to_anonymous(self):
        assert bc.sanitize_user_id("   ") == "user-anonymous"

    def test_allows_dot_underscore_hyphen(self):
        assert bc.sanitize_user_id("a.b_c-d") == "user-a.b_c-d"

    def test_whitespace_and_slash_replaced(self):
        assert bc.sanitize_user_id("some user/name") == "user-some-user-name"


class TestLabelName:
    @pytest.mark.parametrize("name", ["cells", "cell-nuclei", "cell_nuclei", "v1.0"])
    def test_valid_names(self, name):
        assert bc.is_valid_label_name(name)

    @pytest.mark.parametrize("name", ["Cells", "cells!", "cell nuclei", "", None, "label:cells"])
    def test_invalid_names(self, name):
        assert not bc.is_valid_label_name(name)

    def test_label_folder_uses_underscore_not_colon(self):
        assert bc.label_folder("cells") == "label_cells"
        assert ":" not in bc.label_folder("cells")

    def test_label_users_path(self):
        assert bc.label_users_path("cells") == "label_cells/users.json"

    def test_label_metadata_path(self):
        assert bc.label_metadata_path("cells") == "label_cells/metadata.json"

    def test_user_label_dir(self):
        assert bc.user_label_dir("cells", "alice") == "label_cells/user-alice"


class TestMiscPaths:
    def test_image_path(self):
        assert bc.image_path("img001") == "images/img001.png"
        assert bc.image_path("img001", ext=".tif") == "images/img001.tif"

    def test_embedding_path(self):
        assert bc.embedding_path("img001", "vit_l_lm") == "embeddings/img001_vit_l_lm.npz"

    def test_annotation_save_paths(self):
        paths = bc.annotation_save_paths("cells", "alice", "img001", "20260814-120000")
        assert paths == {
            "png": "label_cells/user-alice/img001-20260814-120000.png",
            "geojson": "label_cells/user-alice/img001-20260814-120000.geojson",
        }

    def test_annotation_save_paths_sanitizes_user(self):
        paths = bc.annotation_save_paths("cells", "github|123", "img001", "20260814-120000")
        assert paths["png"].startswith("label_cells/user-github-123/")


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


class TestTimestamps:
    def test_format_timestamp_is_utc(self):
        import datetime as dt

        d = dt.datetime(2026, 8, 14, 12, 34, 56, tzinfo=dt.timezone.utc)
        assert bc.format_timestamp(d) == "20260814-123456"

    def test_format_timestamp_converts_non_utc(self):
        import datetime as dt

        tz = dt.timezone(dt.timedelta(hours=2))
        d = dt.datetime(2026, 8, 14, 14, 34, 56, tzinfo=tz)  # 12:34:56 UTC
        assert bc.format_timestamp(d) == "20260814-123456"

    def test_new_timestamp_matches_shape(self):
        ts = bc.new_timestamp()
        assert bc.is_valid_timestamp(ts)

    def test_is_valid_timestamp(self):
        assert bc.is_valid_timestamp("20260814-123456")
        assert not bc.is_valid_timestamp("2026-08-14")
        assert not bc.is_valid_timestamp("")

    def test_lexicographic_sort_equals_chronological(self):
        import datetime as dt

        earlier = bc.format_timestamp(dt.datetime(2026, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc))
        later = bc.format_timestamp(dt.datetime(2026, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc))
        assert sorted([later, earlier]) == [earlier, later]


# ---------------------------------------------------------------------------
# Annotation filename parsing + latest-pair lookup
# ---------------------------------------------------------------------------


class TestParseAnnotationFilename:
    def test_simple_stem(self):
        assert bc.parse_annotation_filename("img001-20260814-120000.png") == {
            "stem": "img001",
            "timestamp": "20260814-120000",
            "ext": "png",
        }

    def test_stem_with_hyphens(self):
        parsed = bc.parse_annotation_filename("my-cool-image-01-20260814-120000.geojson")
        assert parsed == {
            "stem": "my-cool-image-01",
            "timestamp": "20260814-120000",
            "ext": "geojson",
        }

    def test_non_matching_returns_none(self):
        assert bc.parse_annotation_filename("users.json") is None
        assert bc.parse_annotation_filename(".keep") is None
        assert bc.parse_annotation_filename("img001.png") is None  # no timestamp


class TestLatestPairsByStem:
    def test_single_complete_pair(self):
        files = ["img001-20260814-120000.png", "img001-20260814-120000.geojson"]
        assert bc.latest_pairs_by_stem(files) == {
            "img001": {
                "timestamp": "20260814-120000",
                "png": "img001-20260814-120000.png",
                "geojson": "img001-20260814-120000.geojson",
            }
        }

    def test_picks_latest_of_multiple_saves(self):
        files = [
            "img001-20260814-090000.png",
            "img001-20260814-090000.geojson",
            "img001-20260814-150000.png",
            "img001-20260814-150000.geojson",
        ]
        result = bc.latest_pairs_by_stem(files)
        assert result["img001"]["timestamp"] == "20260814-150000"

    def test_incomplete_pair_is_excluded(self):
        # png-only save (e.g. upload interrupted) never counts as annotated.
        files = ["img001-20260814-120000.png"]
        assert bc.latest_pairs_by_stem(files) == {}

    def test_mismatched_timestamps_not_paired(self):
        files = ["img001-20260814-090000.png", "img001-20260814-100000.geojson"]
        assert bc.latest_pairs_by_stem(files) == {}

    def test_multiple_stems(self):
        files = [
            "img001-20260814-090000.png",
            "img001-20260814-090000.geojson",
            "img002-20260814-090000.png",
            "img002-20260814-090000.geojson",
        ]
        result = bc.latest_pairs_by_stem(files)
        assert set(result.keys()) == {"img001", "img002"}

    def test_ignores_non_annotation_files(self):
        files = ["users.json", ".keep", "img001-20260814-090000.png", "img001-20260814-090000.geojson"]
        result = bc.latest_pairs_by_stem(files)
        assert list(result.keys()) == ["img001"]

    def test_latest_pick_is_lexicographic(self):
        # Cross-year-boundary sanity check for the "string sort == time sort" claim.
        files = [
            "img001-20251231-235959.png",
            "img001-20251231-235959.geojson",
            "img001-20260101-000000.png",
            "img001-20260101-000000.geojson",
        ]
        result = bc.latest_pairs_by_stem(files)
        assert result["img001"]["timestamp"] == "20260101-000000"


class TestIsAnnotated:
    def test_true_with_complete_pair(self):
        files = ["img001-20260814-120000.png", "img001-20260814-120000.geojson"]
        assert bc.is_annotated(files)

    def test_false_when_empty(self):
        assert not bc.is_annotated([])

    def test_false_with_only_png(self):
        assert not bc.is_annotated(["img001-20260814-120000.png"])


# ---------------------------------------------------------------------------
# Embedding filename parsing
# ---------------------------------------------------------------------------


class TestParseEmbeddingFilename:
    def test_vit_l_lm(self):
        assert bc.parse_embedding_filename("img001_vit_l_lm.npz") == {
            "stem": "img001",
            "model_type": "vit_l_lm",
        }

    def test_vit_b(self):
        # Longest-suffix-first matching must not mistake "vit_b" for "vit_b_lm".
        assert bc.parse_embedding_filename("img001_vit_b.npz") == {
            "stem": "img001",
            "model_type": "vit_b",
        }

    def test_stem_with_underscores(self):
        assert bc.parse_embedding_filename("my_image_01_vit_l_lm.npz") == {
            "stem": "my_image_01",
            "model_type": "vit_l_lm",
        }

    def test_non_npz_returns_none(self):
        assert bc.parse_embedding_filename("img001_vit_l_lm.png") is None

    def test_unknown_model_type_returns_none(self):
        assert bc.parse_embedding_filename("img001_unknown_model.npz") is None


# ---------------------------------------------------------------------------
# Dataset metadata: paths, new/read/write/delete
# ---------------------------------------------------------------------------


class TestMetadataPaths:
    def test_artifact_id_to_filename(self):
        assert bc.artifact_id_to_filename("bioimage-io/annotation-abc123") == (
            "bioimage-io__annotation-abc123.json"
        )

    def test_filename_roundtrip(self):
        artifact_id = "bioimage-io/annotation-abc123"
        filename = bc.artifact_id_to_filename(artifact_id)
        assert bc.filename_to_artifact_id(filename) == artifact_id

    def test_metadata_path_under_root(self, tmp_path):
        p = bc.metadata_path("bioimage-io/annotation-abc123", root=tmp_path)
        assert p == tmp_path / "datasets" / "bioimage-io__annotation-abc123.json"


class TestMetadataReadWrite:
    def test_read_missing_returns_none(self, tmp_path):
        assert bc.read_metadata("bioimage-io/annotation-xyz", root=tmp_path) is None

    def test_write_then_read_roundtrip(self, tmp_path):
        meta = bc.new_metadata(
            "bioimage-io/annotation-xyz", owner={"id": "u1", "email": "u1@example.com"}
        )
        bc.write_metadata(meta, root=tmp_path)
        loaded = bc.read_metadata("bioimage-io/annotation-xyz", root=tmp_path)
        assert loaded["artifact_id"] == "bioimage-io/annotation-xyz"
        assert loaded["owner"] == {"id": "u1", "email": "u1@example.com"}
        assert loaded["labels"] == []
        assert loaded["managers"] == []
        assert loaded["annotators"] == []
        assert loaded["public"] is False

    def test_write_is_atomic_no_tmp_file_left_behind(self, tmp_path):
        meta = bc.new_metadata("bioimage-io/annotation-xyz", owner={"id": "u1"})
        bc.write_metadata(meta, root=tmp_path)
        d = tmp_path / "datasets"
        assert sorted(p.name for p in d.iterdir()) == ["bioimage-io__annotation-xyz.json"]

    def test_write_updates_updated_at(self, tmp_path):
        meta = bc.new_metadata("bioimage-io/annotation-xyz", owner={"id": "u1"})
        written = bc.write_metadata(meta, root=tmp_path)
        assert written["updated_at"] >= written["created_at"]

    def test_write_produces_valid_json_on_disk(self, tmp_path):
        meta = bc.new_metadata("bioimage-io/annotation-xyz", owner={"id": "u1"})
        bc.write_metadata(meta, root=tmp_path)
        p = bc.metadata_path("bioimage-io/annotation-xyz", root=tmp_path)
        json.loads(p.read_text())  # does not raise

    def test_delete_metadata(self, tmp_path):
        meta = bc.new_metadata("bioimage-io/annotation-xyz", owner={"id": "u1"})
        bc.write_metadata(meta, root=tmp_path)
        assert bc.delete_metadata("bioimage-io/annotation-xyz", root=tmp_path) is True
        assert bc.read_metadata("bioimage-io/annotation-xyz", root=tmp_path) is None

    def test_delete_missing_returns_false(self, tmp_path):
        assert bc.delete_metadata("bioimage-io/annotation-nope", root=tmp_path) is False

    def test_list_dataset_ids(self, tmp_path):
        for suffix in ("a", "b", "c"):
            meta = bc.new_metadata(f"bioimage-io/annotation-{suffix}", owner={"id": "u1"})
            bc.write_metadata(meta, root=tmp_path)
        assert bc.list_dataset_ids(root=tmp_path) == [
            "bioimage-io/annotation-a",
            "bioimage-io/annotation-b",
            "bioimage-io/annotation-c",
        ]

    def test_list_dataset_ids_empty(self, tmp_path):
        assert bc.list_dataset_ids(root=tmp_path) == []

    def test_list_dataset_ids_ignores_tmp_files(self, tmp_path):
        meta = bc.new_metadata("bioimage-io/annotation-a", owner={"id": "u1"})
        bc.write_metadata(meta, root=tmp_path)
        (tmp_path / "datasets" / "stray.json.tmp").write_text("{}")
        assert bc.list_dataset_ids(root=tmp_path) == ["bioimage-io/annotation-a"]


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


class TestAddLabel:
    def test_appends_label(self):
        meta = _meta(labels=[])
        bc.add_label(meta, "cells", "cell bodies")
        assert meta["labels"] == [{"name": "cells", "description": "cell bodies"}]

    def test_idempotent(self):
        meta = _meta(labels=[])
        bc.add_label(meta, "cells", "cell bodies")
        bc.add_label(meta, "cells", "different description ignored")
        assert meta["labels"] == [{"name": "cells", "description": "cell bodies"}]

    def test_invalid_name_raises(self):
        meta = _meta(labels=[])
        with pytest.raises(ValueError):
            bc.add_label(meta, "Not Valid")


# ---------------------------------------------------------------------------
# Per-label users.json upsert
# ---------------------------------------------------------------------------


class TestUpsertLabelUser:
    def test_adds_new_entry(self):
        users = bc.upsert_label_user({}, "user-alice", "alice", "alice@example.com")
        assert users == {"user-alice": {"id": "alice", "email": "alice@example.com"}}

    def test_preserves_other_entries(self):
        existing = {"user-bob": {"id": "bob", "email": None}}
        users = bc.upsert_label_user(existing, "user-alice", "alice", "alice@example.com")
        assert users["user-bob"] == {"id": "bob", "email": None}
        assert users["user-alice"] == {"id": "alice", "email": "alice@example.com"}

    def test_idempotent_refresh(self):
        users = bc.upsert_label_user({}, "user-alice", "alice", "old@example.com")
        users = bc.upsert_label_user(users, "user-alice", "alice", "new@example.com")
        assert users == {"user-alice": {"id": "alice", "email": "new@example.com"}}

    def test_none_map_treated_as_empty(self):
        users = bc.upsert_label_user(None, "user-alice", "alice", None)
        assert users == {"user-alice": {"id": "alice", "email": None}}


# ---------------------------------------------------------------------------
# Hypha ACL permissions mirror
# ---------------------------------------------------------------------------


class TestBuildAclPermissions:
    def test_owner_and_managers_get_star(self):
        meta = _meta(managers=[{"id": "mgr-1"}], annotators=[], public=False)
        perms = bc.build_acl_permissions(meta)
        assert perms["owner-1"] == "*"
        assert perms["mgr-1"] == "*"

    def test_annotators_get_read_plus(self):
        meta = _meta(managers=[], annotators=[{"id": "ann-1"}], public=False)
        perms = bc.build_acl_permissions(meta)
        assert perms["ann-1"] == "r+"

    def test_public_adds_wildcard_read(self):
        meta = _meta(public=True)
        perms = bc.build_acl_permissions(meta)
        assert perms["*"] == "r+"

    def test_not_public_no_wildcard(self):
        meta = _meta(public=False)
        assert "*" not in bc.build_acl_permissions(meta)

    def test_manager_also_listed_as_annotator_keeps_star(self):
        # set_user_role prevents this in practice, but build_acl_permissions
        # should still be defensive: a manager entry must not be downgraded.
        meta = _meta(managers=[{"id": "dup"}], annotators=[{"id": "dup"}])
        perms = bc.build_acl_permissions(meta)
        assert perms["dup"] == "*"

    def test_prefers_id_over_email_as_key(self):
        meta = _meta(managers=[{"id": "mgr-1", "email": "mgr@example.com"}], annotators=[])
        perms = bc.build_acl_permissions(meta)
        assert "mgr-1" in perms
        assert "mgr@example.com" not in perms

    def test_falls_back_to_email_when_no_id(self):
        meta = _meta(owner={"email": "owner@example.com"}, managers=[], annotators=[])
        perms = bc.build_acl_permissions(meta)
        assert perms["owner@example.com"] == "*"


# ---------------------------------------------------------------------------
# ensure_staged retry wrapper
# ---------------------------------------------------------------------------


class TestEnsureStaged:
    @pytest.mark.asyncio
    async def test_success_first_try_no_restage(self):
        calls = []

        async def call():
            calls.append("call")
            return "ok"

        async def restage():
            calls.append("restage")

        result = await bc.ensure_staged(call, restage, sleep=_no_sleep)
        assert result == "ok"
        assert calls == ["call"]

    @pytest.mark.asyncio
    async def test_retries_after_stage_error(self):
        attempts = {"n": 0}
        calls = []

        async def call():
            attempts["n"] += 1
            calls.append("call")
            if attempts["n"] < 2:
                raise RuntimeError("Artifact is not in stage mode")
            return "ok"

        async def restage():
            calls.append("restage")

        result = await bc.ensure_staged(call, restage, sleep=_no_sleep)
        assert result == "ok"
        assert calls == ["call", "restage", "call"]

    @pytest.mark.asyncio
    async def test_gives_up_after_max_attempts(self):
        async def call():
            raise RuntimeError("not in stage mode")

        async def restage():
            pass

        with pytest.raises(RuntimeError):
            await bc.ensure_staged(call, restage, max_attempts=3, sleep=_no_sleep)

    @pytest.mark.asyncio
    async def test_non_stage_error_propagates_immediately_without_restage(self):
        calls = []

        async def call():
            calls.append("call")
            raise ValueError("something else entirely")

        async def restage():
            calls.append("restage")

        with pytest.raises(ValueError):
            await bc.ensure_staged(call, restage, sleep=_no_sleep)
        assert calls == ["call"]

    @pytest.mark.asyncio
    async def test_default_sleep_is_real_asyncio_sleep(self):
        # Exercise the default sleep= path (fast: max_attempts=1 means no sleep needed).
        async def call():
            return "ok"

        async def restage():
            pass

        result = await bc.ensure_staged(call, restage, max_attempts=1)
        assert result == "ok"


async def _no_sleep(_seconds):
    return None


# ---------------------------------------------------------------------------
# Cache freshness
# ---------------------------------------------------------------------------


class TestIsCacheFresh:
    def test_fresh(self):
        now = time.time()
        assert bc.is_cache_fresh(now - 10, ttl_s=60, now=now)

    def test_stale(self):
        now = time.time()
        assert not bc.is_cache_fresh(now - 61, ttl_s=60, now=now)

    def test_boundary_is_stale(self):
        now = time.time()
        assert not bc.is_cache_fresh(now - 60, ttl_s=60, now=now)


# ---------------------------------------------------------------------------
# register_dataset ownership check
# ---------------------------------------------------------------------------


class TestCallerMatchesArtifactOwner:
    def test_matches_manifest_owner_id(self):
        manifest = {"owner": {"id": "u1", "email": "u1@example.com"}}
        assert bc.caller_matches_artifact_owner(manifest, None, "u1", None)

    def test_matches_manifest_owner_email(self):
        manifest = {"owner": {"id": "u1", "email": "u1@example.com"}}
        assert bc.caller_matches_artifact_owner(manifest, None, None, "u1@example.com")

    def test_matches_created_by(self):
        assert bc.caller_matches_artifact_owner({}, "u1", "u1", None)

    def test_no_match(self):
        manifest = {"owner": {"id": "u1"}}
        assert not bc.caller_matches_artifact_owner(manifest, "u1", "someone-else", None)

    def test_empty_manifest_and_no_created_by(self):
        assert not bc.caller_matches_artifact_owner({}, None, "u1", None)


def test_resolve_role_http_anonymous_is_public_on_public_dataset():
    import broker_core as core
    meta = core.new_metadata("bioimage-io/x", owner={"id": "u1"})
    meta["public"] = True
    assert core.resolve_role(meta, "http-anonymous", None) == "public"
    assert core.resolve_role(meta, "anonymous", None) == "public"
    assert core.resolve_role(meta, None, None) == "public"
    assert core.resolve_role(meta, "real-user", None) == "annotator"
