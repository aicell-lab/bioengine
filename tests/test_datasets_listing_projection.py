"""The public dataset catalog must not publish anyone's collaborator roster.

`GET /datasets` is deliberately unauthenticated — a user has to see that a
dataset exists before asking for access to it. What was never a deliberate
decision is that it returned each manifest *whole*, so `authorized_users` —
the email addresses of third parties who never interacted with this server —
came out with it.

Every address in this file is a fake at example.invalid, which by RFC 2606 can
never be a real domain. Deliberately imports only `proxy_server`, not
`HttpZarrStore`, so these run in the standard suite (the worker image has no
`zarr`).
"""

import pytest
import yaml
from fastapi.testclient import TestClient

# Imported by full path on purpose: ``from bioengine.datasets import proxy_server``
# goes through the package's __getattr__ delegation and raises MissingDependency
# instead of importing the submodule.
import bioengine.datasets.proxy_server as proxy_server
from bioengine.datasets.proxy_server import (
    PUBLIC_MANIFEST_FIELDS,
    _build_app,
    load_datasets,
    public_manifest,
)

MEMBER = "member@example.invalid"
OUTSIDER = "outsider@example.invalid"
OTHER_MEMBER = "other-member@example.invalid"

USERS = {
    "tok-member": {"id": "user:member", "email": MEMBER},
    "tok-outsider": {"id": "user:outsider", "email": OUTSIDER},
    "tok-other": {"id": "user:other", "email": OTHER_MEMBER},
}

# A manifest carrying every documented field plus two the data owner invented.
# manifest.yaml is unschema'd, so invented keys are the normal case, not an edge.
FULL_MANIFEST = {
    "id": "blood-atlas",
    "name": "Blood Cell Atlas",
    "description": "Single-cell RNA-seq.",
    "version": "1.2",
    "license": "CC-BY-4.0",
    "authors": [{"name": "Jane Smith", "affiliation": "KTH"}],
    "tags": ["single-cell"],
    "documentation": "https://example.invalid/docs",
    "git_repo": "https://example.invalid/repo",
    "authorized_users": [MEMBER],
    "internal_contact": "pi@example.invalid",
    "cohort_note": "recruited at site B",
}


@pytest.fixture(autouse=True)
def stub_token_parsing(monkeypatch):
    """Resolve fake bearer tokens locally instead of calling Hypha."""

    async def fake_parse_token(token, cached_user_info):
        if token is None:
            return {"id": "anonymous-user", "email": "no-email"}
        if token not in USERS:
            from fastapi import HTTPException

            raise HTTPException(status_code=401, detail="Invalid token")
        return dict(USERS[token])

    monkeypatch.setattr(proxy_server, "parse_token", fake_parse_token)


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def build(tmp_path, manifests=None):
    if manifests is None:
        manifests = [FULL_MANIFEST]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for manifest in manifests:
        dataset_dir = data_dir / manifest["id"]
        dataset_dir.mkdir()
        (dataset_dir / "manifest.yaml").write_text(yaml.dump(manifest))
    return TestClient(
        _build_app(
            data_dir=data_dir,
            datasets=load_datasets(data_dir),
            cached_user_info={},
        )
    )


# ---------------------------------------------------------------------------
# The unauthenticated response, asserted on its SHAPE
# ---------------------------------------------------------------------------


def test_anonymous_listing_carries_no_field_outside_the_allowlist(tmp_path):
    """Asserted on the key set, not on the one field we know about today: a
    manifest key invented tomorrow must not reach the public catalog either."""
    listing = build(tmp_path).get("/datasets").json()
    for dataset_id, manifest in listing.items():
        unexpected = set(manifest) - set(PUBLIC_MANIFEST_FIELDS)
        assert not unexpected, f"'{dataset_id}' published {sorted(unexpected)}"


def test_anonymous_listing_drops_the_roster(tmp_path):
    manifest = build(tmp_path).get("/datasets").json()["blood-atlas"]
    assert "authorized_users" not in manifest
    assert MEMBER not in str(manifest)


def test_anonymous_listing_drops_invented_fields(tmp_path):
    """The fault was returning the object whole; an allowlist is what fixes it."""
    manifest = build(tmp_path).get("/datasets").json()["blood-atlas"]
    assert "internal_contact" not in manifest
    assert "cohort_note" not in manifest


def test_an_invalid_token_is_still_refused(tmp_path):
    assert build(tmp_path).get("/datasets", headers=auth("tok-bogus")).status_code == 401


# ---------------------------------------------------------------------------
# Discovery is preserved — this is what PR #205's access requests rely on
# ---------------------------------------------------------------------------


def test_the_dataset_is_still_discoverable_anonymously(tmp_path):
    listing = build(tmp_path).get("/datasets").json()
    assert list(listing) == ["blood-atlas"]


def test_the_public_view_still_describes_the_dataset(tmp_path):
    manifest = build(tmp_path).get("/datasets").json()["blood-atlas"]
    assert manifest["id"] == "blood-atlas"
    assert manifest["name"] == "Blood Cell Atlas"
    assert manifest["description"] == "Single-cell RNA-seq."


def test_the_public_view_keeps_the_documented_citation_metadata(tmp_path):
    manifest = build(tmp_path).get("/datasets").json()["blood-atlas"]
    for field in ("version", "license", "authors", "tags", "documentation", "git_repo"):
        assert field in manifest, field


def test_a_dataset_with_no_optional_fields_lists_cleanly(tmp_path):
    client = build(tmp_path, [{"id": "bare", "authorized_users": [MEMBER]}])
    assert client.get("/datasets").json() == {"bare": {"id": "bare"}}


# ---------------------------------------------------------------------------
# The roster is returned to whoever is named in it, and nobody else
# ---------------------------------------------------------------------------


def test_a_named_member_sees_the_whole_manifest(tmp_path):
    manifest = build(tmp_path).get("/datasets", headers=auth("tok-member")).json()[
        "blood-atlas"
    ]
    assert manifest["authorized_users"] == [MEMBER]
    assert manifest["internal_contact"] == "pi@example.invalid"


def test_an_authenticated_outsider_sees_only_the_public_view(tmp_path):
    manifest = build(tmp_path).get("/datasets", headers=auth("tok-outsider")).json()[
        "blood-atlas"
    ]
    assert "authorized_users" not in manifest


def test_membership_is_per_dataset(tmp_path):
    """Being on one roster must not reveal another dataset's."""
    client = build(
        tmp_path,
        [
            {"id": "mine", "authorized_users": [MEMBER]},
            {"id": "theirs", "authorized_users": [OTHER_MEMBER]},
        ],
    )
    listing = client.get("/datasets", headers=auth("tok-member")).json()
    assert listing["mine"]["authorized_users"] == [MEMBER]
    assert "authorized_users" not in listing["theirs"]


def test_a_user_id_entry_also_earns_the_roster(tmp_path):
    client = build(tmp_path, [{"id": "by-id", "authorized_users": ["user:member"]}])
    listing = client.get("/datasets", headers=auth("tok-member")).json()
    assert listing["by-id"]["authorized_users"] == ["user:member"]


# ---------------------------------------------------------------------------
# A wildcard grants the data, not the roster
# ---------------------------------------------------------------------------


def test_a_wildcard_does_not_hand_the_roster_to_anonymous(tmp_path):
    """A dataset listing both '*' and a named address would otherwise publish
    that address to everyone — the wildcard authorizes reading the data, not
    reading who else may read it."""
    client = build(tmp_path, [{"id": "open", "authorized_users": ["*", MEMBER]}])
    manifest = client.get("/datasets").json()["open"]
    assert "authorized_users" not in manifest
    assert MEMBER not in str(manifest)


def test_a_wildcard_does_not_hand_the_roster_to_a_logged_in_outsider(tmp_path):
    client = build(tmp_path, [{"id": "open", "authorized_users": ["*", MEMBER]}])
    manifest = client.get("/datasets", headers=auth("tok-outsider")).json()["open"]
    assert "authorized_users" not in manifest


def test_a_named_member_of_a_wildcard_dataset_still_sees_the_roster(tmp_path):
    client = build(tmp_path, [{"id": "open", "authorized_users": ["*", MEMBER]}])
    manifest = client.get("/datasets", headers=auth("tok-member")).json()["open"]
    assert manifest["authorized_users"] == ["*", MEMBER]


def test_a_wildcard_still_grants_the_data_itself(tmp_path):
    """Narrowing the catalog must not narrow access to the files."""
    data_dir = tmp_path / "data"
    (data_dir / "open").mkdir(parents=True)
    (data_dir / "open" / "manifest.yaml").write_text(
        yaml.dump({"id": "open", "authorized_users": ["*"]})
    )
    (data_dir / "open" / "notes.txt").write_text("payload")
    client = TestClient(
        _build_app(
            data_dir=data_dir,
            datasets=load_datasets(data_dir),
            cached_user_info={},
        )
    )
    assert client.get("/datasets/open/files").status_code == 200
    assert client.get("/data/open/notes.txt").text == "payload"


# ---------------------------------------------------------------------------
# The projection helper on its own
# ---------------------------------------------------------------------------


def test_public_manifest_never_emits_the_roster():
    assert "authorized_users" not in public_manifest(FULL_MANIFEST)


def test_public_manifest_omits_absent_fields_rather_than_nulling_them():
    assert public_manifest({"id": "x"}) == {"id": "x"}


def test_the_allowlist_does_not_contain_the_roster():
    assert "authorized_users" not in PUBLIC_MANIFEST_FIELDS
