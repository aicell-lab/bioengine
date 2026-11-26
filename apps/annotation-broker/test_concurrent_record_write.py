"""Two concurrent changes to one dataset record must BOTH survive.

The broker stores a dataset as one JSON document and every mutation is a
read-modify-write: the caller reads the whole record, changes a field in memory, and writes the
whole thing back. Two callers overlapping on one dataset therefore lose one change entirely --
and, worse, the loser is handed back the record it *believed* it wrote, so nothing reports a
failure. That is how a label created right after an upload disappears while `create_label`
returns a record containing it.

HOW THE RACE IS MADE DETERMINISTIC rather than hoped for. A test that just starts two threads
and hopes they overlap is flaky, and a flaky test here would be worse than none: it would go
green on the broken code often enough to look fixed. Instead a `threading.Barrier` holds both
writers until each has finished READING. With a plain read-modify-write that guarantees the loss
(both read the same old record, both write their own version, the last one wins). With a
serialised mutation it guarantees nothing but a wait: the second writer cannot read until the
first has written, so it sees the first's change and keeps it.

So these tests fail on the unfixed code for the reason under test, not by timing luck.
"""

from __future__ import annotations

import json
import threading
import time

import broker_core as core


def _new(root, artifact_id="ws/ds-1"):
    meta = core.new_metadata(artifact_id, {"id": "u1", "email": "u1@example.org"})
    return core.write_metadata(meta, root=root)


def _read(root, artifact_id="ws/ds-1"):
    return core.read_metadata(artifact_id, root=root)


def test_two_concurrent_mutations_both_survive(tmp_path):
    """The headline guarantee: an upload recording image dimensions and a label being created
    at the same time must not erase one another.

    WHY NOT A BARRIER INSIDE THE MUTATION. My first version of this test put a
    `threading.Barrier` inside both mutate functions so neither could write until both had read.
    Under a correct, serialised implementation that DEADLOCKS by construction: the first writer
    holds the lock while waiting for a second writer that cannot acquire it. Both timed out and
    nothing was written, and the test reported `the label was LOST` against working code. A
    synchronisation primitive placed inside the critical section tests the lock, not the data.

    The shape below instead makes writer A occupy the lock for a known interval while writer B
    arrives during it. That is deterministic in BOTH directions: with the lock, B waits, re-reads
    and keeps A's change; without it, B read before A wrote and A's change is erased.
    """
    _new(tmp_path)
    a_inside = threading.Event()
    errors = []

    def slow_label():
        try:
            def mutate(meta):
                a_inside.set()                  # B may now start
                time.sleep(0.6)                 # ... and must wait this long for the lock
                core.add_label(meta, "cells", "")
                return meta
            core.mutate_metadata("ws/ds-1", mutate, root=tmp_path)
        except Exception as exc:                # noqa: BLE001
            errors.append(("label", exc))

    t = threading.Thread(target=slow_label)
    t.start()
    assert a_inside.wait(timeout=10), "writer A never entered its mutation"
    # B starts while A is demonstrably inside its own mutation.
    def add_dims(meta):
        meta.setdefault("image_dims", {})["scan-2"] = [384, 384]
        return meta
    core.mutate_metadata("ws/ds-1", add_dims, root=tmp_path)
    t.join(timeout=30)
    assert not t.is_alive(), "writer A did not finish -- deadlock?"
    assert not errors, f"a writer raised: {errors!r}"

    stored = _read(tmp_path)
    names = [lab["name"] for lab in stored.get("labels", [])]
    assert "cells" in names, f"the label was LOST: labels={stored.get('labels')!r}"
    assert "scan-2" in (stored.get("image_dims") or {}), (
        f"the image dimensions were LOST: image_dims={stored.get('image_dims')!r}"
    )


def test_locked_metadata_serialises_the_same_way(tmp_path):
    """The context-manager form carries the same guarantee -- it is what the call sites use, so
    testing only `mutate_metadata` would leave the converted sites unmeasured."""
    _new(tmp_path)
    a_inside = threading.Event()
    errors = []

    def slow_label():
        try:
            with core.locked_metadata("ws/ds-1", root=tmp_path) as meta:
                a_inside.set()
                time.sleep(0.6)
                core.add_label(meta, "cells", "")
        except Exception as exc:                # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=slow_label)
    t.start()
    assert a_inside.wait(timeout=10), "writer A never entered the block"
    with core.locked_metadata("ws/ds-1", root=tmp_path) as meta:
        meta.setdefault("image_dims", {})["scan-2"] = [384, 384]
    t.join(timeout=30)
    assert not t.is_alive() and not errors, f"writer A failed: {errors!r}"
    stored = _read(tmp_path)
    assert "cells" in [lab["name"] for lab in stored.get("labels", [])], "label LOST"
    assert "scan-2" in (stored.get("image_dims") or {}), "dims LOST"


def test_locked_metadata_writes_nothing_when_the_block_raises(tmp_path):
    _new(tmp_path)
    before = _read(tmp_path)
    try:
        with core.locked_metadata("ws/ds-1", root=tmp_path) as meta:
            meta["public"] = True
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    else:
        raise AssertionError("the exception must reach the caller")
    assert _read(tmp_path) == before, "a failed block must leave the record untouched"


def test_mutate_metadata_returns_what_was_actually_stored(tmp_path):
    """The second half of the defect: the caller must never be handed a record describing a
    write that did not land."""
    _new(tmp_path)

    def mutate(meta):
        core.add_label(meta, "cells", "")
        return meta

    returned = core.mutate_metadata("ws/ds-1", mutate, root=tmp_path)
    stored = _read(tmp_path)
    assert returned == stored, "the returned record differs from what is on disk"


def test_mutate_metadata_raises_rather_than_reporting_a_phantom_write(tmp_path):
    """If the mutation raises, nothing is written and the error reaches the caller -- the
    'either persists or raises' half of the contract."""
    _new(tmp_path)
    before = _read(tmp_path)

    def mutate(meta):
        core.add_label(meta, "Bad Name!", "")   # add_label rejects this
        return meta

    try:
        core.mutate_metadata("ws/ds-1", mutate, root=tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("a rejected label name must raise, not be silently dropped")
    assert _read(tmp_path) == before, "a failed mutation must leave the record untouched"


def test_mutate_metadata_refuses_a_missing_record(tmp_path):
    """A mutation against a dataset that does not exist is an error, never a silent create:
    creating an empty record here would hand a caller a dataset with no owner and no ACL."""
    try:
        core.mutate_metadata("ws/nope", lambda m: m, root=tmp_path)
    except (KeyError, FileNotFoundError, ValueError):
        pass
    else:
        raise AssertionError("mutating a missing record must raise")


def test_write_metadata_tmp_path_is_unique_per_writer(tmp_path):
    """Two writers must not share one scratch file.

    `write_metadata` stages through a temp file and then `os.replace`s it, which is atomic for
    ONE writer. With a temp path derived only from the artifact id, two concurrent writers write
    the SAME scratch file, so one can replace a half-written document -- a torn record rather
    than merely a lost one. This asserts the staged path varies between writers, which is what
    makes the replace safe.
    """
    seen = set()
    for _ in range(5):
        seen.add(core.metadata_tmp_path("ws/ds-1", root=tmp_path).name)
    assert len(seen) == 5, f"the staging path repeats across writers: {seen!r}"


def test_record_on_disk_stays_valid_json_under_concurrent_writers(tmp_path):
    """The torn-write consequence, exercised: hammer one record from several threads and assert
    it always parses. A truncated document here takes the whole dataset out, not one field."""
    _new(tmp_path)
    errors = []

    def writer(n):
        try:
            for i in range(20):
                def mutate(meta, n=n, i=i):
                    meta.setdefault("image_dims", {})[f"img-{n}-{i}"] = [8, 8]
                    return meta
                core.mutate_metadata("ws/ds-1", mutate, root=tmp_path)
        except Exception as exc:                # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, f"writers raised: {errors!r}"
    raw = core.metadata_path("ws/ds-1", root=tmp_path).read_text()
    doc = json.loads(raw)                        # raises if the document was torn
    # 4 writers x 20 writes, every key distinct: none may be missing.
    dims = doc.get("image_dims") or {}
    assert len(dims) == 80, f"writes were lost: {len(dims)} of 80 keys present"
