"""Regression tests for the HOME checkpoint caches' write discipline.

``evaluate_community`` cached its 1.2 GB community checkpoint with a plain
``Path.write_bytes`` **at the path it then handed the GPU runtime as the checkpoint
to load**. The path is keyed on ``(pool, version)`` only and ``Path.home()`` is shared
by every call the worker serves, so a second evaluation of the same community opened
that file ``'wb'`` — truncating it — while the first call's ``torch.load`` was reading
it. The symptom is a deserialisation error far from the cause
(``PytorchStreamReader failed finding central directory``; the zip central directory
lives at the end of the file, so a prefix loses it).

Identical content is no defence here, which is why this is tested as a property
rather than as a diff: both writers write the same bytes, and the reader's problem is
the file's *length* mid-write. The sibling cache in ``runtime_cellpose._fetch_member``
was already safe for the opposite reason — it writes ``member-<sha>.part`` and renames
onto the name it reads — so this only brings ``entry.py`` up to the idiom the app
already uses.

Plain pytest, no torch, no numpy, no Ray, no live Hypha: ``_write_file_atomic`` is
module-level and the defect is file bookkeeping.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest


OLD = b"the previously cached checkpoint, whole" * 8
NEW = b"the newly downloaded checkpoint, whole" * 16


def _atomic_write():
    """``entry._write_file_atomic``, imported lazily.

    ``entry`` pulls bioengine/httpx/hypha-rpc at import, so importing it at module
    scope would turn a missing optional dependency into a collection error for the
    whole file instead of a failure of these tests.
    """
    import entry

    return entry._write_file_atomic


def test_the_destination_is_never_opened_for_writing(tmp_path: Path) -> None:
    """The invariant: at the moment of the rename, dest still holds the OLD bytes.

    This is the discriminating assertion. The pre-fix implementation wrote straight to
    ``dest``, so by the time any content existed the old file was already gone — and it
    never called ``os.replace`` at all, which ``saw`` catches.
    """
    dest = tmp_path / "community_v1.pt"
    dest.write_bytes(OLD)

    saw = {}
    real_replace = os.replace

    def observing_replace(src, dst):
        # Read through the destination NAME, not the handle, so this sees exactly what
        # a concurrent torch.load would see.
        saw["dest_at_rename"] = Path(dst).read_bytes()
        saw["src_at_rename"] = Path(src).read_bytes()
        saw["src_name"] = str(src)
        return real_replace(src, dst)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "replace", observing_replace)
        _atomic_write()(dest, NEW)

    assert "dest_at_rename" in saw, "the write never renamed; it wrote in place"
    assert saw["dest_at_rename"] == OLD, "dest was modified before the rename"
    assert saw["src_at_rename"] == NEW
    assert str(dest) != saw["src_name"], "the temp name must differ from the destination"
    assert dest.read_bytes() == NEW
    assert list(tmp_path.iterdir()) == [dest], "a temp file was left behind"


def test_two_writers_do_not_share_a_temp_name(tmp_path: Path) -> None:
    """The path is keyed on (pool, version) only, so two writers DO collide on it.

    A temp name derived from the destination alone (``dest.with_suffix('.part')``)
    would be shared, which is safe in ``_fetch_member`` only because its content is
    content-addressed. Not a property worth depending on twice.
    """
    dest = tmp_path / "community_v1.pt"
    names = []
    real_replace = os.replace

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "replace", lambda src, dst: (names.append(str(src)), real_replace(src, dst))[1])
        _atomic_write()(dest, NEW)
        _atomic_write()(dest, NEW)

    assert len(names) == 2
    assert names[0] != names[1]


def test_a_concurrent_reader_never_observes_a_partial_file(tmp_path: Path) -> None:
    """The property the incident is about, exercised with a real second thread.

    The payload is sized so the write spans many reader samples. Pre-fix, the sampler
    sees intermediate lengths (including 0, from the ``'wb'`` truncate); the fix admits
    only the two whole states.
    """
    dest = tmp_path / "community_v1.pt"
    old = b"\x01" * (4 * 1024 * 1024)
    new = b"\x02" * (48 * 1024 * 1024)
    dest.write_bytes(old)

    observed: set = set()
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            try:
                observed.add(dest.stat().st_size)
            except FileNotFoundError:
                observed.add(-1)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        _atomic_write()(dest, new)
    finally:
        stop.set()
        sampler.join(timeout=5)

    assert observed, "the sampler never ran, so this test asserted nothing"
    unexpected = observed - {len(old), len(new)}
    assert not unexpected, f"a reader could see dest at {sorted(unexpected)} bytes"
