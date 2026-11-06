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

    def test_em_organelles_types(self):
        # 0.9.0: EM organelle generalists must parse (they were missing from
        # KNOWN_MODEL_TYPES, silently dropping their embeddings from the index).
        assert bc.parse_embedding_filename("img001_vit_t_em_organelles.npz") == {
            "stem": "img001",
            "model_type": "vit_t_em_organelles",
        }


class TestCollectEmbeddings:
    def test_multiple_models_per_stem(self):
        out = bc.collect_embeddings(
            [
                "img001_vit_l_lm.npz",
                "img001_vit_t_em_organelles.npz",
                "img002_vit_b_lm.npz",
                "junk.txt",
            ]
        )
        assert out["img001"] == {
            "model_type": "vit_l_lm",
            "model_types": ["vit_l_lm", "vit_t_em_organelles"],
        }
        assert out["img002"] == {
            "model_type": "vit_b_lm",
            "model_types": ["vit_b_lm"],
        }

    def test_preferred_falls_back_to_first_sorted(self):
        out = bc.collect_embeddings(
            ["img_vit_t_lm.npz", "img_vit_b_em_organelles.npz"]
        )
        assert out["img"]["model_type"] == "vit_b_em_organelles"
        assert out["img"]["model_types"] == ["vit_b_em_organelles", "vit_t_lm"]

    def test_duplicates_collapse(self):
        out = bc.collect_embeddings(["a_vit_l_lm.npz", "a_vit_l_lm.npz"])
        assert out["a"]["model_types"] == ["vit_l_lm"]


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
    async def test_retries_on_hyphas_verbatim_staging_message(self):
        # Regression: every other test here uses the invented phrase "stage
        # mode". Hypha actually raises "Artifact must be in staging mode.",
        # and "staging" does not contain the substring "stage", so a
        # predicate keyed on the longer form matched none of the real
        # failures. Captured verbatim from a live KTH broker traceback.
        attempts = {"n": 0}
        calls = []

        async def call():
            attempts["n"] += 1
            calls.append("call")
            if attempts["n"] < 2:
                raise AssertionError("Artifact must be in staging mode.")
            return "ok"

        async def restage():
            calls.append("restage")

        result = await bc.ensure_staged(call, restage, sleep=_no_sleep)
        assert result == "ok"
        assert calls == ["call", "restage", "call"]

    def test_is_stage_mode_error_accepts_both_wordings(self):
        assert bc.is_stage_mode_error(AssertionError("Artifact must be in staging mode."))
        assert bc.is_stage_mode_error(RuntimeError("Artifact is not in stage mode"))
        assert not bc.is_stage_mode_error(ValueError("something else entirely"))

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


class TestAccessRequests:
    def _meta(self):
        import broker_core as core
        return core.new_metadata("bioimage-io/x", owner={"id": "owner1"})

    def test_add_and_remove_request(self):
        import broker_core as core
        m = self._meta()
        core.add_access_request(m, {"id": "u2", "email": "u2@x.org"}, "annotator", "t1")
        assert len(m["access_requests"]) == 1
        assert m["access_requests"][0]["requested_role"] == "annotator"
        core.remove_access_request(m, {"id": "u2"})
        assert m["access_requests"] == []

    def test_repeat_request_updates_role(self):
        import broker_core as core
        m = self._meta()
        core.add_access_request(m, {"id": "u2"}, "annotator", "t1")
        core.add_access_request(m, {"id": "u2"}, "manager", "t2")
        assert len(m["access_requests"]) == 1
        assert m["access_requests"][0]["requested_role"] == "manager"
        assert m["access_requests"][0]["requested_at"] == "t2"

    def test_invalid_role_rejected(self):
        import broker_core as core, pytest
        m = self._meta()
        with pytest.raises(ValueError):
            core.add_access_request(m, {"id": "u2"}, "owner", "t1")

    def test_legacy_metadata_without_key(self):
        import broker_core as core
        m = self._meta(); m.pop("access_requests")
        core.add_access_request(m, {"email": "a@b.c"}, "annotator", "t1")
        assert len(m["access_requests"]) == 1
        core.remove_access_request(m, {"email": "A@B.C"})
        assert m["access_requests"] == []


# ---------------------------------------------------------------------------
# count_pairs_by_stem
# ---------------------------------------------------------------------------


def test_count_pairs_counts_complete_saves_only():
    files = [
        "a-20260101-000001.png", "a-20260101-000001.geojson",
        "a-20260102-000002.png", "a-20260102-000002.geojson",
        "a-20260103-000003.png",  # incomplete save
        "b-20260101-000001.geojson",  # incomplete
        "junk.txt",
    ]
    assert bc.count_pairs_by_stem(files) == {"a": 2}


def test_count_pairs_empty():
    assert bc.count_pairs_by_stem([]) == {}
    assert bc.count_pairs_by_stem(["x.png", "y.geojson"]) == {}


# ---------------------------------------------------------------------------
# Split paths + names
# ---------------------------------------------------------------------------


def test_split_paths():
    assert bc.splits_dir_path("cells") == "label_cells/splits"
    assert bc.split_path("cells", "fold-1") == "label_cells/splits/fold-1.json"


def test_split_name_validation():
    assert bc.is_valid_split_name("default")
    assert bc.is_valid_split_name("fold-1.a_b")
    assert not bc.is_valid_split_name("")
    assert not bc.is_valid_split_name("With Space")
    assert not bc.is_valid_split_name("UPPER")


# ---------------------------------------------------------------------------
# new_split
# ---------------------------------------------------------------------------

COUNTS = {"a": 3, "b": 1, "c": 2, "d": 1}


def test_new_split_happy_path():
    doc = bc.new_split("cells", "default", ["a", "b"], ["c"], COUNTS, "user-1")
    assert doc["train"] == ["a", "b"]
    assert doc["test"] == ["c"]
    assert doc["ratio"] == pytest.approx(2 / 3)
    assert doc["annotation_counts"] == {"a": 3, "b": 1, "c": 2}
    assert doc["checkpoint"] is None
    assert doc["history"] == []
    assert doc["label"] == "cells"


def test_new_split_explicit_ratio_and_dedup():
    doc = bc.new_split("cells", "default", ["a", "a", "b"], [], COUNTS, None, ratio=0.8)
    assert doc["train"] == ["a", "b"]
    assert doc["ratio"] == 0.8


def test_new_split_rejects_unannotated():
    with pytest.raises(ValueError, match="unannotated"):
        bc.new_split("cells", "default", ["a", "zz"], [], COUNTS, None)
    with pytest.raises(ValueError, match="unannotated"):
        bc.new_split("cells", "default", ["a"], ["zz"], COUNTS, None)


def test_new_split_rejects_overlap_empty_train_bad_name_bad_ratio():
    with pytest.raises(ValueError, match="both train and test"):
        bc.new_split("cells", "default", ["a"], ["a"], COUNTS, None)
    with pytest.raises(ValueError, match="at least one training image"):
        bc.new_split("cells", "default", [], ["a"], COUNTS, None)
    with pytest.raises(ValueError, match="Invalid split name"):
        bc.new_split("cells", "Bad Name", ["a"], [], COUNTS, None)
    with pytest.raises(ValueError, match="ratio"):
        bc.new_split("cells", "default", ["a"], [], COUNTS, None, ratio=1.5)


# ---------------------------------------------------------------------------
# extend_split
# ---------------------------------------------------------------------------


def _base_split():
    return bc.new_split("cells", "default", ["a"], ["c"], COUNTS, "user-1")


def test_extend_split_adds_and_refreshes_counts():
    doc = _base_split()
    counts_now = {"a": 5, "b": 1, "c": 2, "d": 1}
    doc = bc.extend_split(doc, ["b"], ["d"], counts_now, "user-2")
    assert doc["train"] == ["a", "b"]
    assert doc["test"] == ["c", "d"]
    assert doc["annotation_counts"] == {"a": 5, "b": 1, "c": 2, "d": 1}
    assert len(doc["history"]) == 1
    assert doc["history"][0]["added_train"] == ["b"]
    assert doc["history"][0]["added_test"] == ["d"]
    assert doc["updated_by"] == "user-2"


def test_extend_split_train_to_test_guard():
    doc = _base_split()
    with pytest.raises(ValueError, match="ever-trained"):
        bc.extend_split(doc, [], ["a"], COUNTS, None)


def test_extend_split_rejects_existing_members_and_unannotated():
    doc = _base_split()
    with pytest.raises(ValueError, match="never move"):
        bc.extend_split(doc, ["c"], [], COUNTS, None)
    with pytest.raises(ValueError, match="unannotated"):
        bc.extend_split(doc, ["zz"], [], COUNTS, None)
    with pytest.raises(ValueError, match="Nothing to add"):
        bc.extend_split(doc, [], [], COUNTS, None)
    with pytest.raises(ValueError, match="both train and test"):
        bc.extend_split(doc, ["b"], ["b"], COUNTS, None)


def test_extend_split_member_losing_files_keeps_zero_count():
    doc = _base_split()
    doc = bc.extend_split(doc, ["b"], [], {"b": 1}, None)
    assert doc["annotation_counts"]["a"] == 0
    assert doc["annotation_counts"]["b"] == 1


def test_split_summary():
    doc = _base_split()
    s = bc.split_summary(doc)
    assert s["n_train"] == 1 and s["n_test"] == 1
    assert s["name"] == "default" and s["label"] == "cells"
    assert s["checkpoint"] is None


# --- parse_png_dims (0.8.0) ---


def test_parse_png_dims_valid_header():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (200, 150)).save(buf, format="PNG")
    assert bc.parse_png_dims(buf.getvalue()[:24]) == [200, 150]


def test_parse_png_dims_rejects_non_png_and_short_input():
    assert bc.parse_png_dims(b"notapng" * 4) is None
    assert bc.parse_png_dims(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4) is None
    assert bc.parse_png_dims(b"") is None


def test_parse_png_dims_rejects_zero_dims():
    header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0d" + b"IHDR" + (0).to_bytes(4, "big") + (150).to_bytes(4, "big")
    assert bc.parse_png_dims(header) is None


class TestEmbeddingRemovalPaths:
    def test_single_model(self):
        assert bc.embedding_removal_paths("img001", "vit_l_lm") == [
            "embeddings/img001_vit_l_lm.npz"
        ]

    def test_all_models_plus_legacy(self):
        paths = bc.embedding_removal_paths("img001")
        assert len(paths) == len(bc.KNOWN_MODEL_TYPES) + 1
        for mt in bc.KNOWN_MODEL_TYPES:
            assert f"embeddings/img001_{mt}.npz" in paths
        assert paths[-1] == "embeddings/img001.npz"

    def test_stem_with_underscores(self):
        # Real stems contain underscores; the model suffix must append cleanly.
        assert bc.embedding_removal_paths("19681_1836_A2_1", "vit_l_lm") == [
            "embeddings/19681_1836_A2_1_vit_l_lm.npz"
        ]


# ---------------------------------------------------------------------------
# Image manifest (0.10.0)
# ---------------------------------------------------------------------------


def _listing(*names, dirs=()):
    """A list_files("images") result: files plus optional directory entries."""
    return (
        [{"name": n, "type": "file"} for n in names]
        + [{"name": d, "type": "directory"} for d in dirs]
    )


class TestImageNamesFromListing:
    def test_keeps_files_drops_directories(self):
        entries = _listing("a.png", "b.tif", dirs=("nested",))
        assert bc.image_names_from_listing(entries) == ["a.png", "b.tif"]

    def test_excludes_the_manifest_itself(self):
        # manifest.json lives inside the directory it describes, so a naive
        # listing would import it as an image called "manifest".
        entries = _listing("a.png", "manifest.json")
        assert bc.image_names_from_listing(entries) == ["a.png"]

    def test_accepts_bare_strings(self):
        assert bc.image_names_from_listing(["a.png", ""]) == ["a.png"]

    def test_is_dir_flag_is_honoured(self):
        assert bc.image_names_from_listing([{"name": "sub", "is_dir": True}]) == []


class TestParseImageManifest:
    def test_round_trip(self):
        entries = {"a": bc.local_image_entry("a.png", [10, 20])}
        assert bc.parse_image_manifest(bc.build_image_manifest(entries)) == entries

    def test_rejects_unknown_version(self):
        doc = bc.build_image_manifest({"a": bc.local_image_entry("a.png")})
        doc["manifest_version"] = 999
        assert bc.parse_image_manifest(doc) == {}

    def test_rejects_garbage_without_raising(self):
        for doc in (None, [], "nope", {}, {"manifest_version": 1, "images": []}):
            assert bc.parse_image_manifest(doc) == {}

    def test_drops_entries_with_an_unknown_state(self):
        doc = {
            "manifest_version": 1,
            "images": {
                "a": {"state": "local", "path": "images/a.png"},
                "b": {"state": "banana", "path": "images/b.png"},
                "c": "not-a-dict",
            },
        }
        assert set(bc.parse_image_manifest(doc)) == {"a"}

    def test_build_sorts_stems(self):
        entries = {s: bc.local_image_entry(f"{s}.png") for s in ("c", "a", "b")}
        assert list(bc.build_image_manifest(entries)["images"]) == ["a", "b", "c"]


class TestLocalImageEntry:
    def test_path_and_state(self):
        assert bc.local_image_entry("img001.png") == {
            "state": "local",
            "path": "images/img001.png",
        }

    def test_dims_included_when_plausible(self):
        entry = bc.local_image_entry("a.png", [640, 480])
        assert (entry["width"], entry["height"]) == (640, 480)

    def test_implausible_dims_omitted(self):
        for dims in (None, [], [0, 100], [100], [0, 0]):
            assert "width" not in bc.local_image_entry("a.png", dims)


class TestMergeImageManifest:
    def test_listing_only_builds_the_whole_manifest(self):
        entries, changed = bc.merge_image_manifest({}, ["a.png", "b.tif"])
        assert changed is True
        assert set(entries) == {"a", "b"}
        assert entries["b"]["path"] == "images/b.tif"

    def test_no_op_when_already_in_sync(self):
        stored = {"a": bc.local_image_entry("a.png")}
        entries, changed = bc.merge_image_manifest(stored, ["a.png"])
        assert changed is False
        assert entries == stored

    def test_read_never_drops_a_missing_local_file(self):
        # A transient listing failure must not delete images from the dataset.
        stored = {"a": bc.local_image_entry("a.png")}
        entries, changed = bc.merge_image_manifest(stored, [])
        assert set(entries) == {"a"}
        assert changed is False

    def test_prune_drops_a_deleted_local_file(self):
        stored = {"a": bc.local_image_entry("a.png"), "b": bc.local_image_entry("b.png")}
        entries, changed = bc.merge_image_manifest(stored, ["a.png"], prune=True)
        assert set(entries) == {"a"}
        assert changed is True

    def test_prune_never_drops_a_remote_entry(self):
        # A remote image has no file under images/ by definition.
        stored = {"r": {"state": "remote", "source": {"kind": "ngff"}}}
        entries, changed = bc.merge_image_manifest(stored, [], prune=True)
        assert set(entries) == {"r"}
        assert changed is False

    def test_dims_backfilled_into_an_existing_entry(self):
        stored = {"a": bc.local_image_entry("a.png")}
        entries, changed = bc.merge_image_manifest(stored, ["a.png"], dims={"a": [8, 9]})
        assert changed is True
        assert (entries["a"]["width"], entries["a"]["height"]) == (8, 9)

    def test_existing_dims_are_not_overwritten(self):
        stored = {"a": bc.local_image_entry("a.png", [1, 2])}
        entries, changed = bc.merge_image_manifest(stored, ["a.png"], dims={"a": [8, 9]})
        assert changed is False
        assert (entries["a"]["width"], entries["a"]["height"]) == (1, 2)

    def test_a_remote_entry_survives_a_read_merge(self):
        stored = {"r": {"state": "remote", "source": {"kind": "ngff"}}}
        entries, _ = bc.merge_image_manifest(stored, ["a.png"])
        assert set(entries) == {"r", "a"}

    def test_same_stem_two_extensions_collapses_deterministically(self):
        entries, _ = bc.merge_image_manifest({}, ["a.tif", "a.png"])
        assert set(entries) == {"a"}
        assert entries["a"]["path"] == "images/a.tif"

    def test_does_not_mutate_the_input(self):
        stored = {"a": bc.local_image_entry("a.png")}
        bc.merge_image_manifest(stored, ["b.png"], dims={"a": [1, 2]}, prune=True)
        assert stored == {"a": {"state": "local", "path": "images/a.png"}}


# ---------------------------------------------------------------------------
# Remote NGFF sources (0.10.0)
# ---------------------------------------------------------------------------


class TestNormaliseStoreRoot:
    def test_adds_exactly_one_trailing_slash(self):
        assert bc.normalise_store_root("https://x/y") == "https://x/y/"
        assert bc.normalise_store_root("https://x/y/") == "https://x/y/"
        assert bc.normalise_store_root("https://x/y///") == "https://x/y/"

    def test_strips_surrounding_whitespace(self):
        assert bc.normalise_store_root("  https://x/y  ") == "https://x/y/"

    def test_rejects_a_query_string(self):
        # The whole point: '?token=' would land before the chunk key.
        with pytest.raises(bc.NgffSourceError, match="query string"):
            bc.normalise_store_root("https://x/y?token=abc")

    def test_rejects_a_fragment(self):
        with pytest.raises(bc.NgffSourceError, match="query string"):
            bc.normalise_store_root("https://x/y#frag")

    def test_rejects_a_non_http_scheme(self):
        with pytest.raises(bc.NgffSourceError, match="http"):
            bc.normalise_store_root("s3://bucket/key")

    def test_rejects_empty(self):
        for value in ("", "   ", None, 7):
            with pytest.raises(bc.NgffSourceError):
                bc.normalise_store_root(value)


class TestStoreCredential:
    def test_splits_the_omezarr_view_path_token(self):
        root, token = bc.split_store_credential(
            "https://ov.example/t/secret-123/zarr/ds1/"
        )
        assert root == "https://ov.example/zarr/ds1/"
        assert token == "secret-123"

    def test_a_plain_bucket_root_is_returned_unchanged(self):
        root, token = bc.split_store_credential("https://s3.example/bucket/img.zarr/")
        assert root == "https://s3.example/bucket/img.zarr/"
        assert token is None

    def test_round_trips(self):
        original = "https://ov.example/t/secret-123/zarr/ds1/"
        root, token = bc.split_store_credential(original)
        assert bc.apply_store_credential(root, token) == original

    def test_apply_is_a_no_op_without_a_credential(self):
        assert bc.apply_store_credential("https://x/y/", None) == "https://x/y/"
        assert bc.apply_store_credential("https://x/y/", "") == "https://x/y/"

    def test_apply_onto_a_bare_host(self):
        assert bc.apply_store_credential("https://x", "tok") == "https://x/t/tok/"

    def test_a_path_segment_merely_named_t_is_not_a_token(self):
        # '/t/' must be a *credential* segment; the regex is anchored on
        # slashes so a directory called 't' at the end is left alone.
        root, token = bc.split_store_credential("https://x/data/t/")
        assert (root, token) == ("https://x/data/t/", None)


def _zattrs(axes=None, datasets=("0", "1"), name=None):
    entry = {"datasets": [{"path": p} for p in datasets]}
    if axes is not None:
        entry["axes"] = axes
    if name is not None:
        entry["name"] = name
    return {"multiscales": [entry]}


class TestParseNgffMultiscales:
    def test_v04_axes_and_datasets(self):
        parsed = bc.parse_ngff_multiscales(
            _zattrs(axes=[{"name": "c"}, {"name": "y"}, {"name": "x"}])
        )
        assert parsed["dataset_paths"] == ["0", "1"]
        assert parsed["axes"] == ["c", "y", "x"]

    def test_absent_axes_implies_tczyx(self):
        assert bc.parse_ngff_multiscales(_zattrs())["axes"] == ["t", "c", "z", "y", "x"]

    def test_string_axes_are_accepted(self):
        assert bc.parse_ngff_multiscales(_zattrs(axes=["Y", "X"]))["axes"] == ["y", "x"]

    def test_string_datasets_are_accepted(self):
        doc = {"multiscales": [{"datasets": ["0", "1", "2"]}]}
        assert bc.parse_ngff_multiscales(doc)["dataset_paths"] == ["0", "1", "2"]

    def test_v3_nests_under_ome(self):
        doc = {"ome": _zattrs(axes=["y", "x"])}
        assert bc.parse_ngff_multiscales(doc)["dataset_paths"] == ["0", "1"]

    def test_named_multiscale_selection(self):
        doc = {
            "multiscales": [
                {"name": "a", "datasets": [{"path": "a0"}]},
                {"name": "b", "datasets": [{"path": "b0"}]},
            ]
        }
        assert bc.parse_ngff_multiscales(doc, "b")["dataset_paths"] == ["b0"]

    def test_missing_named_multiscale_lists_what_exists(self):
        doc = {"multiscales": [{"name": "a", "datasets": [{"path": "a0"}]}]}
        with pytest.raises(bc.NgffSourceError, match="'nope'.*a"):
            bc.parse_ngff_multiscales(doc, "nope")

    def test_rejects_a_group_that_is_not_ome_zarr(self):
        for doc in ({}, {"multiscales": []}, None, {"multiscales": "0"}):
            with pytest.raises(bc.NgffSourceError, match="multiscales"):
                bc.parse_ngff_multiscales(doc)

    def test_rejects_a_multiscale_with_no_datasets(self):
        with pytest.raises(bc.NgffSourceError, match="no datasets"):
            bc.parse_ngff_multiscales({"multiscales": [{"datasets": []}]})


class TestNgffAxisRoles:
    def test_tczyx(self):
        roles = bc.ngff_axis_roles(["t", "c", "z", "y", "x"], 5)
        assert (roles["y"], roles["x"], roles["c"]) == (3, 4, 1)

    def test_two_dimensional(self):
        roles = bc.ngff_axis_roles(["y", "x"], 2)
        assert (roles["y"], roles["x"], roles["c"]) == (0, 1, None)

    def test_channel_last(self):
        roles = bc.ngff_axis_roles(["y", "x", "c"], 3)
        assert (roles["y"], roles["x"], roles["c"]) == (0, 1, 2)

    def test_rank_disagreement_falls_back_to_the_last_two_dims(self):
        # Declared 5 axes, actual rank 3: trust the array, not the metadata.
        roles = bc.ngff_axis_roles(["t", "c", "z", "y", "x"], 3)
        assert (roles["y"], roles["x"], roles["c"]) == (1, 2, None)

    def test_no_y_or_x_falls_back(self):
        roles = bc.ngff_axis_roles(["a", "b", "c"], 3)
        assert (roles["y"], roles["x"]) == (1, 2)

    def test_rank_below_two_raises(self):
        with pytest.raises(bc.NgffSourceError, match="at least 2"):
            bc.ngff_axis_roles(["x"], 1)

    def test_level_dims_reads_the_role_axes_not_the_tail(self):
        roles = bc.ngff_axis_roles(["c", "y", "x"], 3)
        assert bc.ngff_level_dims([3, 600, 800], roles) == {"width": 800, "height": 600}


class TestParseZarrArrayMeta:
    def test_v2_blosc(self):
        meta = bc.parse_zarr_array_meta({
            "shape": [3, 600, 800],
            "chunks": [1, 256, 256],
            "dtype": "<u2",
            "compressor": {"id": "blosc", "cname": "zstd"},
            "zarr_format": 2,
        })
        assert meta["shape"] == [3, 600, 800]
        assert meta["chunks"] == [1, 256, 256]
        assert meta["dtype"] == "<u2"
        assert meta["codecs"] == ["blosc"]
        assert meta["separator"] == "."

    def test_v2_filters_come_before_the_compressor(self):
        meta = bc.parse_zarr_array_meta({
            "shape": [4, 4], "chunks": [2, 2], "dtype": "|u1",
            "filters": [{"id": "delta", "dtype": "|u1"}],
            "compressor": {"id": "zstd"},
        })
        assert meta["codecs"] == ["delta", "zstd"]

    def test_v2_uncompressed(self):
        meta = bc.parse_zarr_array_meta({
            "shape": [4, 4], "chunks": [2, 2], "dtype": "|u1", "compressor": None,
        })
        assert meta["codecs"] == []

    def test_v2_honours_dimension_separator(self):
        meta = bc.parse_zarr_array_meta({
            "shape": [4, 4], "chunks": [2, 2], "dtype": "|u1",
            "dimension_separator": "/",
        })
        assert meta["separator"] == "/"

    def test_v3(self):
        meta = bc.parse_zarr_array_meta({
            "zarr_format": 3,
            "node_type": "array",
            "shape": [600, 800],
            "data_type": "uint16",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [256, 256]}},
            "chunk_key_encoding": {"name": "default"},
            "codecs": [{"name": "bytes"}, {"name": "blosc"}],
        })
        assert meta["dtype"] == "uint16"
        assert meta["codecs"] == ["bytes", "blosc"]
        assert meta["separator"] == "/"

    def test_v3_with_v2_key_encoding_defaults_to_a_dot(self):
        meta = bc.parse_zarr_array_meta({
            "zarr_format": 3, "shape": [4, 4], "data_type": "uint8",
            "chunk_grid": {"configuration": {"chunk_shape": [2, 2]}},
            "chunk_key_encoding": {"name": "v2"},
            "codecs": [],
        })
        assert meta["separator"] == "."

    def test_rejects_junk(self):
        with pytest.raises(bc.NgffSourceError):
            bc.parse_zarr_array_meta(None)
        with pytest.raises(bc.NgffSourceError, match="shape"):
            bc.parse_zarr_array_meta({"dtype": "|u1"})


class TestNgffCodecAndDtypeGates:
    def test_accepts_what_zarrita_ships(self):
        for codec in ("blosc", "lz4", "zstd", "gzip", "zlib", "transpose", "bytes", "crc32c"):
            assert bc.undecodable_ngff_codecs({"codecs": [codec]}) == []

    def test_names_only_the_offenders(self):
        assert bc.undecodable_ngff_codecs(
            {"codecs": ["bytes", "jpeg2k", "blosc", "delta"]}
        ) == ["delta", "jpeg2k"]

    def test_no_codecs_is_fine(self):
        assert bc.undecodable_ngff_codecs({}) == []

    def test_dtype_gate_accepts_both_spellings(self):
        for dtype in ("<u2", "|u1", ">f4", "uint16", "float32", "bool"):
            assert bc.ngff_dtype_is_readable(dtype) is True

    def test_dtype_gate_rejects_bigint_and_strings(self):
        # 64-bit ints decode to BigInt64Array, which cannot be scaled to RGBA.
        for dtype in ("<i8", "<u8", "int64", "uint64", "|S10", "<U5", ""):
            assert bc.ngff_dtype_is_readable(dtype) is False


class TestFirstChunkKey:
    def test_v2_dot_separated(self):
        assert bc.first_chunk_key(
            {"zarr_format": 2, "chunks": [1, 256, 256], "separator": ".", "key_encoding": "v2"}
        ) == "0.0.0"

    def test_v2_slash_separated(self):
        assert bc.first_chunk_key(
            {"zarr_format": 2, "chunks": [256, 256], "separator": "/", "key_encoding": "v2"}
        ) == "0/0"

    def test_v3_default_encoding(self):
        assert bc.first_chunk_key(
            {"zarr_format": 3, "chunks": [1, 256, 256], "separator": "/", "key_encoding": "default"}
        ) == "c/0/0/0"

    def test_v3_with_v2_keys(self):
        assert bc.first_chunk_key(
            {"zarr_format": 3, "chunks": [4, 4], "separator": ".", "key_encoding": "v2"}
        ) == "0.0"


class TestNgffChannelNames:
    def test_reads_omero_labels(self):
        names = bc.ngff_channel_names(
            {"omero": {"channels": [{"label": "DAPI"}, {"label": "GFP"}]}}
        )
        assert names == ["DAPI", "GFP"]

    def test_falls_back_to_name(self):
        assert bc.ngff_channel_names({"omero": {"channels": [{"name": "Cy5"}]}}) == ["Cy5"]

    def test_absent_omero_is_none_not_an_error(self):
        for doc in ({}, None, {"omero": {}}, {"omero": {"channels": []}}, {"omero": 3}):
            assert bc.ngff_channel_names(doc) is None

    def test_v3_nested_under_ome(self):
        assert bc.ngff_channel_names(
            {"ome": {"omero": {"channels": [{"label": "BF"}]}}}
        ) == ["BF"]


class TestRemoteImageEntry:
    def test_shape(self):
        entry = bc.remote_image_entry(
            "https://x/zarr/ds1/", "0", 800, 600,
            dtype="uint16", channels=["DAPI"], levels=3, content_hint="etag-1",
            fetched_at="2026-09-10T00:00:00Z",
        )
        assert entry["state"] == "remote"
        assert (entry["width"], entry["height"]) == (800, 600)
        assert entry["dtype"] == "uint16"
        assert entry["channels"] == ["DAPI"]
        assert entry["levels"] == 3
        assert entry["source"] == {
            "kind": "ngff",
            "store_root": "https://x/zarr/ds1/",
            "multiscale_path": "0",
            "region": None,
            "fetched_at": "2026-09-10T00:00:00Z",
            "content_hint": "etag-1",
        }

    def test_has_no_path_so_presigning_callers_fail_loudly(self):
        entry = bc.remote_image_entry("https://x/", None, 4, 4)
        assert "path" not in entry

    def test_optional_fields_are_omitted_rather_than_null(self):
        entry = bc.remote_image_entry("https://x/", None, 4, 4)
        assert "dtype" not in entry and "channels" not in entry and "levels" not in entry

    def test_survives_a_manifest_round_trip(self):
        entry = bc.remote_image_entry("https://x/", "0", 4, 4)
        doc = bc.build_image_manifest({"r": entry})
        assert bc.parse_image_manifest(json.loads(json.dumps(doc)))["r"] == entry


class TestNormaliseRegion:
    def test_absent_region_is_the_whole_store(self):
        assert bc.normalise_region(None, 1000, 800) is None

    def test_a_patch_is_kept_verbatim(self):
        assert bc.normalise_region(
            {"x": 100, "y": 50, "width": 256, "height": 256}, 1000, 800
        ) == {"x": 100, "y": 50, "width": 256, "height": 256}

    def test_string_numbers_from_json_are_coerced(self):
        assert bc.normalise_region(
            {"x": "0", "y": "0", "width": "10", "height": "20"}, 1000, 800
        ) == {"x": 0, "y": 0, "width": 10, "height": 20}

    def test_a_whole_store_region_collapses_to_none(self):
        # Identical to a whole-store import, so it must produce an identical
        # entry rather than a second spelling of the same thing.
        assert bc.normalise_region(
            {"x": 0, "y": 0, "width": 1000, "height": 800}, 1000, 800
        ) is None

    def test_far_edge_is_clamped_not_refused(self):
        assert bc.normalise_region(
            {"x": 900, "y": 700, "width": 256, "height": 256}, 1000, 800
        ) == {"x": 900, "y": 700, "width": 100, "height": 100}

    def test_origin_outside_the_store_is_refused(self):
        with pytest.raises(ValueError, match="outside"):
            bc.normalise_region({"x": 1200, "y": 0, "width": 8, "height": 8}, 1000, 800)
        with pytest.raises(ValueError, match="outside"):
            bc.normalise_region({"x": 0, "y": -4, "width": 8, "height": 8}, 1000, 800)

    def test_empty_region_is_refused(self):
        with pytest.raises(ValueError, match="empty"):
            bc.normalise_region({"x": 0, "y": 0, "width": 0, "height": 8}, 1000, 800)

    def test_malformed_region_is_refused(self):
        with pytest.raises(ValueError):
            bc.normalise_region({"x": 0, "y": 0}, 1000, 800)
        with pytest.raises(ValueError):
            bc.normalise_region([0, 0, 8, 8], 1000, 800)
        with pytest.raises(ValueError):
            bc.normalise_region({"x": 0, "y": 0, "width": "wide", "height": 8}, 1000, 800)

    def test_entry_dimensions_are_the_patchs(self):
        region = bc.normalise_region(
            {"x": 10, "y": 20, "width": 512, "height": 384}, 100000, 80000
        )
        entry = bc.remote_image_entry(
            "https://x/slide.zarr/", None, region["width"], region["height"],
            dtype="uint8", channels=["r", "g", "b"], region=region,
        )
        assert (entry["width"], entry["height"]) == (512, 384)
        assert entry["source"]["region"] == region
        # ...which is the whole point: the size gate sees the patch, so a
        # slide that could never be materialised whole becomes materialisable
        # a patch at a time with no new code in the gate.
        assert bc.materialisation_bytes(entry) == 512 * 384 * 3
        bc.check_materialisable("slide", entry)
        whole = bc.remote_image_entry(
            "https://x/slide.zarr/", None, 100000, 80000,
            dtype="uint8", channels=["r", "g", "b"],
        )
        with pytest.raises(ValueError):
            bc.check_materialisable("slide", whole)


class TestImageEntryIsLocal:
    def test_local_with_a_path(self):
        assert bc.image_entry_is_local(bc.local_image_entry("a.png")) is True

    def test_remote_is_not_local(self):
        assert bc.image_entry_is_local(bc.remote_image_entry("https://x/", None, 4, 4)) is False

    def test_local_without_a_path_is_not_local(self):
        # A bug if it happens; treating it as local would push the failure
        # down to whoever tried to presign it.
        assert bc.image_entry_is_local({"state": "local"}) is False

    def test_junk(self):
        for entry in (None, {}, "local", []):
            assert bc.image_entry_is_local(entry) is False

    def test_non_local_stems_names_only_known_unmaterialised_ones(self):
        entries = {
            "a": bc.local_image_entry("a.png"),
            "r": bc.remote_image_entry("https://x/", None, 4, 4),
            "q": bc.remote_image_entry("https://x/", None, 4, 4),
        }
        # 'ghost' is unknown, which is the *other* validator's error to raise.
        assert bc.non_local_stems(entries, ["a", "r", "q", "ghost"]) == ["q", "r"]

    def test_non_local_stems_empty_when_all_materialised(self):
        entries = {"a": bc.local_image_entry("a.png")}
        assert bc.non_local_stems(entries, ["a"]) == []


class TestDtypeItemsize:
    @pytest.mark.parametrize(
        "dtype,size",
        [
            ("<u2", 2), ("uint16", 2), ("|u1", 1), ("uint8", 1),
            (">f4", 4), ("float32", 4), ("float64", 8), ("b1", 1),
        ],
    )
    def test_both_spellings(self, dtype, size):
        assert bc.dtype_itemsize(dtype) == size

    def test_unknown_is_zero_not_a_guess(self):
        # 0 has to mean "no opinion" downstream, so an unrecognised dtype must
        # not fall back to some plausible itemsize.
        for dtype in ("", None, "str", "<U8", 7, "complex128"):
            assert bc.dtype_itemsize(dtype) == 0


class TestMaterialisationBytes:
    def test_counts_channels(self):
        entry = bc.remote_image_entry(
            "https://x/", "0", 1000, 1000, dtype="uint16", channels=["a", "b"]
        )
        assert bc.materialisation_bytes(entry) == 1000 * 1000 * 2 * 2

    def test_no_channel_metadata_counts_one(self):
        entry = bc.remote_image_entry("https://x/", "0", 1000, 1000, dtype="uint16")
        assert bc.materialisation_bytes(entry) == 1000 * 1000 * 2

    @pytest.mark.parametrize(
        "entry",
        [
            None,
            {},
            {"state": "remote", "width": 10, "height": 10},          # no dtype
            {"state": "remote", "width": 0, "height": 10, "dtype": "uint8"},
            {"state": "remote", "width": 10, "height": 10, "dtype": "str"},
        ],
    )
    def test_unknown_is_zero(self, entry):
        assert bc.materialisation_bytes(entry) == 0


class TestCheckMaterialisable:
    def _remote(self, **kw):
        return bc.remote_image_entry("https://x/", "0", 512, 512, **kw)

    def test_accepts_a_small_remote_image(self):
        bc.check_materialisable("r", self._remote(dtype="uint16"))

    def test_missing_entry(self):
        with pytest.raises(FileNotFoundError):
            bc.check_materialisable("ghost", None)

    def test_local_entry_is_a_value_error_not_a_silent_pass(self):
        with pytest.raises(ValueError, match="not a remote image"):
            bc.check_materialisable("a", bc.local_image_entry("a.tif"))

    def test_refuses_a_slide_sized_source(self):
        huge = bc.remote_image_entry(
            "https://x/", "0", 100_000, 80_000, dtype="uint16"
        )
        with pytest.raises(ValueError, match="over the 64 MB"):
            bc.check_materialisable("wsi", huge)

    def test_unknown_size_is_allowed_through(self):
        # 0 bytes means "no opinion", not "zero pixels". Refusing on it would
        # block every store whose dtype we failed to record.
        bc.check_materialisable("r", {"state": "remote", "width": 4, "height": 4})


class TestMaterialiseImageEntry:
    def _remote(self):
        return bc.remote_image_entry(
            "https://x/zarr/ds1/", "0", 800, 600,
            dtype="uint16", channels=["DAPI"], levels=3, content_hint="etag-1",
        )

    def test_flips_state_and_sets_the_path(self):
        out = bc.materialise_image_entry(self._remote(), "r.tif", levels=2)
        assert out["state"] == "local"
        assert out["path"] == "images/r.tif"
        assert out["levels"] == 2
        assert bc.image_entry_is_local(out) is True

    def test_keeps_provenance(self):
        remote = self._remote()
        out = bc.materialise_image_entry(remote, "r.tif", levels=2)
        assert out["source"] == remote["source"]
        assert (out["width"], out["height"], out["dtype"]) == (800, 600, "uint16")
        assert out["channels"] == ["DAPI"]

    def test_does_not_mutate_the_input(self):
        remote = self._remote()
        bc.materialise_image_entry(remote, "r.tif", levels=2)
        assert remote["state"] == "remote" and "path" not in remote

    def test_drops_the_sources_level_count_when_none_is_given(self):
        # levels describes the file we wrote, not the store we read. Carrying
        # the remote's 3 over to a file with a different pyramid would make
        # the manifest lie about the artifact's own content.
        out = bc.materialise_image_entry(self._remote(), "r.tif")
        assert "levels" not in out

    def test_survives_a_manifest_round_trip(self):
        out = bc.materialise_image_entry(self._remote(), "r.tif", levels=2)
        doc = bc.build_image_manifest({"r": out})
        assert bc.parse_image_manifest(json.loads(json.dumps(doc)))["r"] == out

    def test_materialised_name(self):
        assert bc.materialised_image_name("a.b-c_1") == "a.b-c_1.tif"


class TestMergeImageManifestPromotesRemote:
    """A file appearing under a stem that is still marked remote.

    This is the recoverable half of a crashed materialisation: the OME-TIFF
    upload landed, the manifest write did not. The union must promote the
    entry, not overwrite it with a bare local one, or the crash silently
    destroys the provenance the design promises to keep (§2).
    """

    def _remote(self):
        return bc.remote_image_entry(
            "https://x/zarr/ds1/", "0", 800, 600,
            dtype="uint16", channels=["DAPI", "GFP"], levels=3,
            content_hint="etag-1",
        )

    def test_promotes_instead_of_replacing(self):
        entries, changed = bc.merge_image_manifest(
            {"r": self._remote()}, ["r.tif"]
        )
        assert changed is True
        assert entries["r"]["state"] == "local"
        assert entries["r"]["path"] == "images/r.tif"

    def test_keeps_provenance_through_the_promotion(self):
        remote = self._remote()
        entries, _ = bc.merge_image_manifest({"r": remote}, ["r.tif"])
        assert entries["r"]["source"] == remote["source"]
        assert entries["r"]["dtype"] == "uint16"
        assert entries["r"]["channels"] == ["DAPI", "GFP"]
        assert (entries["r"]["width"], entries["r"]["height"]) == (800, 600)

    def test_drops_the_stores_level_count(self):
        entries, _ = bc.merge_image_manifest({"r": self._remote()}, ["r.tif"])
        assert "levels" not in entries["r"]

    def test_a_remote_with_no_file_is_left_alone(self):
        remote = self._remote()
        entries, changed = bc.merge_image_manifest({"r": remote}, [])
        assert changed is False
        assert entries["r"] == remote

    def test_promotion_survives_a_prune(self):
        # prune only removes local entries whose file is gone. The promoted
        # entry's file is right there in the listing, so it stays.
        entries, _ = bc.merge_image_manifest(
            {"r": self._remote()}, ["r.tif"], prune=True
        )
        assert entries["r"]["path"] == "images/r.tif"

    def test_an_unrelated_local_file_still_gets_a_bare_entry(self):
        entries, _ = bc.merge_image_manifest(
            {"r": self._remote()}, ["r.tif", "plain.png"], dims={"plain": [10, 20]}
        )
        assert "source" not in entries["plain"]
        assert (entries["plain"]["width"], entries["plain"]["height"]) == (10, 20)


class TestLooksLikeTiff:
    @pytest.mark.parametrize(
        "head",
        [
            b"II*\x00\x08\x00\x00\x00",   # classic, little endian
            b"MM\x00*\x00\x00\x00\x08",   # classic, big endian
            b"II+\x00\x08\x00",           # BigTIFF, little endian
            b"MM\x00+\x00\x08",           # BigTIFF, big endian
        ],
    )
    def test_accepts_every_tiff_flavour(self, head):
        assert bc.looks_like_tiff(head) is True

    @pytest.mark.parametrize(
        "head",
        [
            b"",
            b"II",                        # truncated signature
            b"\x89PNG\r\n\x1a\n",         # a PNG
            b"<?xml version",             # an S3 error document
            b"II*\x01",                   # wrong magic
            None,
            "II*\x00",                    # str, not bytes
        ],
    )
    def test_rejects_everything_else(self, head):
        assert bc.looks_like_tiff(head) is False

    def test_accepts_a_bytearray(self):
        assert bc.looks_like_tiff(bytearray(b"II*\x00")) is True


class TestCheckMaterialisableChannels:
    def _remote(self, channels):
        return bc.remote_image_entry(
            "https://x/zarr/ds1/", "0", 64, 64,
            dtype="uint16", channels=channels, levels=1,
        )

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
    def test_accepts_up_to_four(self, n):
        bc.check_materialisable("r", self._remote([f"c{i}" for i in range(n)]))

    @pytest.mark.parametrize("n", [5, 40])
    def test_refuses_wider(self, n):
        with pytest.raises(ValueError, match="over the 4 the OME-TIFF profile"):
            bc.check_materialisable("r", self._remote([f"c{i}" for i in range(n)]))

    def test_unknown_channel_count_is_allowed_through(self):
        # A store with no omero metadata reports nothing. Refusing on "we do
        # not know" would block the common single-channel case.
        entry = self._remote(None)
        entry.pop("channels", None)
        bc.check_materialisable("r", entry)

    def test_the_channel_gate_runs_before_the_size_gate(self):
        # Both would fire here. The channel message is the actionable one:
        # a smaller region (design §13 step 6) fixes size, never width.
        wide = bc.remote_image_entry(
            "https://x/zarr/ds1/", "0", 100000, 80000,
            dtype="uint16", channels=[f"c{i}" for i in range(9)], levels=1,
        )
        with pytest.raises(ValueError, match="9 channels"):
            bc.check_materialisable("r", wide)
