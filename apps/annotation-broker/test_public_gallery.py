"""The public gallery: only published datasets, and never an email address.

Measured anonymously against the live collection on 2026-11-25: all four
datasets were enumerable with no token, and three carried the owner's personal
and work email addresses. The collection's wildcard grant has since been
removed, which closed that -- and also made the one deliberately-public dataset
unreadable anonymously, because a child read needs read on the parent. The
gallery therefore cannot be "a list of ids the page fetches itself": it has to
be served BY the broker, which reads with the app's own credential.

These tests pin the two properties that follow from that:

* the gallery filters on the BROKER's own ``public`` flag -- the same one
  ``resolve_role`` uses to decide whether an anonymous caller may read a
  dataset at all, so the gallery and the read path cannot disagree;
* nothing in the response is an email address, asserted over the whole
  serialised payload rather than field by field.
"""

import json
import re

import broker_core as core

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

OWNER = {"id": "github|49943582", "email": "someone@example.org"}
OWNER_WITH_EMAIL_ID = {"id": "person@example.org", "email": "person@example.org"}


def _write(root, artifact_id, *, public, owner=None, **extra):
    meta = core.new_metadata(artifact_id, owner or OWNER)
    meta["public"] = public
    meta.update(extra)
    core.write_metadata(meta, root=root)
    return meta


# ── which datasets appear ────────────────────────────────────────────────────


def test_only_datasets_whose_owner_PUBLISHED_them_appear(tmp_path):
    _write(tmp_path, "bioimage-io/pub-one", public=True)
    _write(tmp_path, "bioimage-io/pub-two", public=True)
    _write(tmp_path, "bioimage-io/private-one", public=False)

    got = [m["artifact_id"] for m in core.iter_public_metadata(tmp_path)]

    assert got == ["bioimage-io/pub-one", "bioimage-io/pub-two"], got


def test_unpublishing_REMOVES_a_dataset_from_the_gallery(tmp_path):
    """The control the row asks for: making one public adds it, unpublishing
    removes it. A filter that only ever added would pass the test above."""
    _write(tmp_path, "bioimage-io/toggle", public=True)
    assert [m["artifact_id"] for m in core.iter_public_metadata(tmp_path)] == [
        "bioimage-io/toggle"
    ]

    _write(tmp_path, "bioimage-io/toggle", public=False)
    assert core.iter_public_metadata(tmp_path) == []


def test_a_record_that_will_not_parse_does_not_EMPTY_the_gallery(tmp_path):
    _write(tmp_path, "bioimage-io/good", public=True)
    bad = core.datasets_dir(tmp_path) / "broken.json"
    bad.write_text("{ not json")

    got = [m["artifact_id"] for m in core.iter_public_metadata(tmp_path)]

    assert got == ["bioimage-io/good"], got


def test_the_gallery_reads_the_SAME_FLAG_the_read_path_gates_on(tmp_path):
    """Why this filters on broker metadata rather than on the artifact.

    `resolve_role` grants an anonymous caller the `public` role from exactly
    this flag. If the gallery used any other source it could advertise a
    dataset the read path then refuses, or hide one it would serve.
    """
    meta = _write(tmp_path, "bioimage-io/pub", public=True)
    assert core.resolve_role(meta, None, None) == "public"

    meta = _write(tmp_path, "bioimage-io/priv", public=False)
    assert core.resolve_role(meta, None, None) == "none"


# ── what the cards carry ─────────────────────────────────────────────────────


def test_NO_EMAIL_ADDRESS_appears_anywhere_in_a_card(tmp_path):
    """Asserted over the whole payload, not field by field.

    The exposure this row exists for was not in a field anybody would have
    thought to strip: the addresses were in `created_by` and in the permission
    keys. A test that checked the owner field alone would have passed through
    the original bug.
    """
    meta = _write(tmp_path, "bioimage-io/pub", public=True)

    entry = core.public_dataset_entry(
        meta,
        name="Dataset from alice@lab.example.com",
        description="Questions to bob.smith+x@sub.example.co.uk please",
        image_count=3,
    )

    blob = json.dumps(entry)
    assert EMAIL.findall(blob) == [], f"an email survived into the card: {blob}"
    assert "[email removed]" in entry["name"]
    assert "[email removed]" in entry["description"]


def test_a_card_carries_WHAT_A_GALLERY_NEEDS(tmp_path):
    """THE CONTROL. A projection that returned nothing, or null for
    everything, would satisfy the email test and be useless."""
    meta = _write(tmp_path, "bioimage-io/pub", public=True)

    entry = core.public_dataset_entry(
        meta, name="HPA Demo", description="A demo set", image_count=86
    )

    assert entry["artifact_id"] == "bioimage-io/pub"
    assert entry["name"] == "HPA Demo"
    assert entry["description"] == "A demo set"
    assert entry["image_count"] == 86
    assert entry["owner_display"] == "github|49943582"


def test_a_card_is_an_ALLOW_LIST_so_a_new_field_cannot_leak(tmp_path):
    meta = _write(
        tmp_path,
        "bioimage-io/pub",
        public=True,
        access_requests=[{"email": "hopeful@example.org"}],
        some_future_field="whatever somebody adds next",
    )

    entry = core.public_dataset_entry(meta, name="n", description="d", image_count=1)

    assert set(entry) == set(core.PUBLIC_DATASET_FIELDS), sorted(entry)
    assert EMAIL.findall(json.dumps(entry)) == []


def test_an_owner_whose_ID_IS_an_email_shows_NO_owner_rather_than_the_address(
    tmp_path,
):
    """Some logins record an address as the id. There is then nothing
    publishable, and inventing a name would be worse than showing none."""
    meta = _write(tmp_path, "bioimage-io/pub", public=True, owner=OWNER_WITH_EMAIL_ID)

    assert core.owner_display(meta) is None
    entry = core.public_dataset_entry(meta, name="n", description="d", image_count=0)
    assert entry["owner_display"] is None
    assert EMAIL.findall(json.dumps(entry)) == []
