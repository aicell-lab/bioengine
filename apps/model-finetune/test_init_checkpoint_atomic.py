"""The init_checkpoint write must not be able to damage its own destination.

``_run_training`` downloads ``init_checkpoint`` and hands the resulting path to a
train worker as ``checkpoint_path``. That worker may be on another node, so the
name must never be observable at a length other than the full content's — a
``write_bytes`` there opens ``'wb'``, truncating the destination before it has the
bytes to replace it with.

WHY THIS TEST LOOKS LIKE THIS, because the obvious shapes do not discriminate
(myco #0268). Asserting that ``_write_file_atomic`` was *called* passes on a revert
that keeps the call and changes the target, so this asserts on the destination's
CONTENT instead. And a successful write leaves the destination identical either
way, so there is nothing to see unless the write fails partway — hence the fault
injection. The fault is injected into ``builtins.open``, below the function under
test and identically for both implementations, so neither arm is handed an
advantage; what differs is only whether the half-written bytes land on the
destination or on a temp name.

Plain pytest against a temporary HOME — no GPU, no Ray, no live Hypha.
"""

from __future__ import annotations

import builtins
import io

import pytest

import training

OLD = b"OLD-CHECKPOINT-" + b"o" * 4096
NEW = b"NEW-CHECKPOINT-" + b"n" * 65536
SESSION = "sess-init-ckpt-atomic"


@pytest.fixture(autouse=True)
def _sandbox_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))


def _method_of(name: str, instance):
    """The real coroutine function off the Serve-wrapped class."""
    import entry

    return getattr(entry.EntryApp.func_or_class, name).__get__(instance)


class _Resp:
    status_code = 200
    content = NEW


class _Stub:
    """Enough of EntryApp to reach the init_checkpoint write and no further.

    ``_run_training`` catches Exception, records FAILED and then pops
    ``_training_tasks`` in its ``finally`` — so the injected fault does not
    propagate out, and the recorded status is how the test knows it fired.
    """

    def __init__(self):
        self._training_tasks = {}

    async def _preparing_heartbeat(self, *a, **k):
        # _run_training creates this as a task and cancels it in its own cleanup;
        # a no-op coroutine is enough to get past that line.
        return None

    async def _prepare_training_data(self, *a, **k):
        # Every key _run_training reads after the write is present, so if the write
        # ever SUCCEEDS the run proceeds instead of dying on a missing key. Without
        # that, a defect arm fails with KeyError and looks like a discriminating
        # result while saying nothing about the destination (caught 2026-11-06).
        return {"patch_shape": (512, 512), "train_images": [], "train_labels": [],
                "val_images": None, "val_labels": None}

    async def _http_retry(self, *a, **k):
        return _Resp()

    def _bump_transport(self, **k):
        return None


def _fail_half_way_through_writing(monkeypatch, target_name: str):
    """Make any write of a file called ``target_name*`` die after half the bytes.

    Patched below the function under test, so both implementations meet the same
    fault in the same place and neither arm is handed an advantage.

    BOTH namespaces must be patched. ``_write_file_atomic`` calls the bare builtin
    ``open``; ``Path.write_bytes`` goes through ``io.open``. They are the same
    function object, but each name is bound separately, so patching only
    ``builtins.open`` leaves the truncating path untouched — and the defect arm then
    writes successfully and fails later for an unrelated reason, which reads as a
    discriminating result and is not one.
    """
    real_open = builtins.open

    class _Torn:
        def __init__(self, fh):
            self._fh = fh

        def write(self, data):
            self._fh.write(data[: len(data) // 2])
            self._fh.flush()
            raise OSError("injected: the write died half way through")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._fh.close()
            return False

        def __getattr__(self, item):
            return getattr(self._fh, item)

    def fake_open(file, mode="r", *a, **k):
        fh = real_open(file, mode, *a, **k)
        if "w" in mode and target_name in str(file):
            return _Torn(fh)
        return fh

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(io, "open", fake_open)


@pytest.mark.asyncio
async def test_a_torn_init_checkpoint_write_does_not_damage_the_destination(monkeypatch) -> None:
    dest = training.session_dir(SESSION) / "init_checkpoint.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(OLD)
    # Positive control: the destination really does hold the old checkpoint before
    # the write, so "unchanged" below means preserved rather than never-written.
    assert dest.read_bytes() == OLD

    _fail_half_way_through_writing(monkeypatch, "init_checkpoint.pt")

    await _method_of("_run_training", _Stub())(
        SESSION, "vit_b_lm", [], [], None, None,
        {"init_checkpoint": "https://example.invalid/init_checkpoint.pt"},
    )

    # Positive control that the fault FIRED and the write was really attempted.
    # Without it, a run in which the branch was never reached would satisfy the
    # assertion below for the wrong reason.
    status = training.read_status(SESSION)
    assert status.get("status") == "FAILED", status
    assert "injected" in str(status.get("message", "")), status

    # The whole claim: a write that failed must not have published a prefix under
    # the name a train worker is given. The atomic path tears a temp file; a
    # truncating write tears this one.
    survived = dest.read_bytes()
    assert survived == OLD, (
        "init_checkpoint.pt was damaged by a failed write: expected the previous "
        f"{len(OLD)} bytes to survive, found {len(survived)} bytes"
        + (" (a prefix of the new content)" if NEW.startswith(survived) else "")
    )


@pytest.mark.asyncio
async def test_the_destination_is_never_left_as_a_prefix_when_absent_beforehand(monkeypatch) -> None:
    """Same fault with no pre-existing file: the name must not appear at all.

    Separate from the arm above because "the old bytes survived" and "the name was
    never published" are different guarantees, and a reader hitting a zero-length
    file is the case the fix's own docstring names.
    """
    dest = training.session_dir(SESSION + "-fresh") / "init_checkpoint.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    assert not dest.exists()

    _fail_half_way_through_writing(monkeypatch, "init_checkpoint.pt")

    await _method_of("_run_training", _Stub())(
        SESSION + "-fresh", "vit_b_lm", [], [], None, None,
        {"init_checkpoint": "https://example.invalid/init_checkpoint.pt"},
    )

    status = training.read_status(SESSION + "-fresh")
    assert status.get("status") == "FAILED", status
    assert "injected" in str(status.get("message", "")), status

    assert not dest.exists(), (
        "init_checkpoint.pt exists after a failed write, holding "
        f"{dest.stat().st_size} of {len(NEW)} bytes — a train worker handed this "
        "path would torch.load a prefix"
    )
