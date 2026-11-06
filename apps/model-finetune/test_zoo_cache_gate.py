"""Regression tests for the zoo cache's HIT gate — the read side of the same cache.

``_fetch_model_package`` returned a cached checkpoint whenever ``weights.pt`` and
``kwargs.json`` both existed, treating ``kwargs.json`` as the commit marker for the
pair. That is sound only for entries written by a version that has
``_write_file_atomic``: before it, both files were written in place, so an interrupted
or truncated write leaves exactly the state the gate reads as committed. The cache
lives in an app HOME that is per-``application_id`` on shared storage and survives
redeploys, so entries outlive the version that wrote them and a prefix left behind is
permanent — the write-side fix in ``test_atomic_cache_write.py`` prevents creating one
and says nothing about reading one.

The gate now compares ``kwargs.json``'s recorded ``bytes`` against the file on disk,
and an entry with no recorded length is a miss that re-downloads once.

Plain pytest, no torch, no numpy, no Ray, no live Hypha: a miss is detected by making
the download path raise, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import textwrap
from pathlib import Path

import pytest


MODEL_ID = "bioimage-io/affable-shark"
WEIGHTS = b"a whole micro-sam checkpoint" * 32


class _FellThrough(Exception):
    """Raised from the download path, so reaching it is observable as a miss."""


def _entry_class():
    """``EntryApp`` as written, behind the Ray Serve deployment the decorator returns.

    Imported lazily: ``entry`` pulls bioengine/httpx/hypha-rpc at import, so a module
    scope import would turn a missing optional dependency into a collection error for
    the whole file instead of a failure of these tests.
    """
    import entry

    return entry.EntryApp.func_or_class


class _Caller:
    """Just enough of ``EntryApp`` to reach the gate and then stop."""

    def __init__(self, *, rdf=None, content=None) -> None:
        self._hypha = self
        self._rdf, self._content = rdf, content
        self.downloaded = False

    async def get_service(self, name):
        if self._rdf is None:
            raise _FellThrough(name)
        return self

    async def get_file(self, artifact_id, file_path):
        return f"https://example.invalid/{artifact_id}/{file_path}"

    async def _http_retry(self, *args, **kwargs):
        return type(
            "R",
            (),
            {
                "text": json.dumps(self._rdf),
                "status_code": 200,
                "raise_for_status": lambda self_: self_,
            },
        )()

    async def _get_verified(self, url, timeout=None):
        self.downloaded = True
        return self._content


def _cache_dir(home: Path) -> Path:
    import entry
    import re as _re

    return (
        home / ".bioengine" / "micro_sam_zoo_cache" / _re.sub(r"[^\w.-]", "_", MODEL_ID)
    )


def _write_entry(home: Path, weights: bytes, meta: dict) -> Path:
    d = _cache_dir(home)
    d.mkdir(parents=True)
    (d / "weights.pt").write_bytes(weights)
    (d / "kwargs.json").write_text(json.dumps(meta))
    return d


def _fetch(caller, home: Path, monkeypatch, fetch=None):
    """Drive the gate with ``Path.home()`` pointed at ``home``.

    ``Path.home()`` resolves ``~`` through the environment on POSIX, so setting HOME is
    enough and no class attribute is patched.
    """
    monkeypatch.setenv("HOME", str(home))
    fn = fetch or _entry_class()._fetch_model_package
    return asyncio.run(fn(caller, MODEL_ID, None))


def test_a_complete_entry_still_hits(tmp_path: Path, monkeypatch) -> None:
    """The discrimination control: the fix must not degrade into "always miss".

    Without this, a gate that rejected everything would satisfy both miss tests below.
    """
    d = _write_entry(
        tmp_path, WEIGHTS, {"model_type": "vit_b_lm", "bytes": len(WEIGHTS)}
    )
    caller = _Caller()

    pkg = _fetch(caller, tmp_path, monkeypatch)

    assert pkg == {"weights_path": str(d / "weights.pt"), "model_type": "vit_b_lm"}, (
        "a hit must return the documented contract and must not leak the bookkeeping "
        f"key into it; got {pkg}"
    )
    assert not caller.downloaded


def test_a_truncated_entry_with_well_formed_meta_is_a_miss(
    tmp_path: Path, monkeypatch
) -> None:
    """(a) ``kwargs.json`` is valid and records the true length; the file is short.

    Both files exist and the metadata parses, so every existence- or parse-based gate
    calls this committed. Only the recorded length says otherwise.
    """
    _write_entry(
        tmp_path, WEIGHTS[:17], {"model_type": "vit_b_lm", "bytes": len(WEIGHTS)}
    )

    with pytest.raises(_FellThrough):
        _fetch(_Caller(), tmp_path, monkeypatch)


def test_a_legacy_entry_with_no_length_field_is_a_miss(
    tmp_path: Path, monkeypatch
) -> None:
    """(b) exactly what 0.16.1-era code left behind: no length recorded at all.

    0.16.1's ``_fetch_model_package`` wrote ``{"model_type": …}`` and nothing else
    (measured on the published artifact: no ``_write_file_atomic``, zero ``os.fsync``,
    zero ``os.replace``), so its entries cannot be shown to be whole. They must
    re-download once rather than be trusted — including the case where the bytes happen
    to be fine, which is why this asserts a miss on a full-length ``weights.pt``.
    """
    _write_entry(tmp_path, WEIGHTS, {"model_type": "vit_b_lm"})

    with pytest.raises(_FellThrough):
        _fetch(_Caller(), tmp_path, monkeypatch)


def test_the_writer_records_the_length_the_gate_checks(
    tmp_path: Path, monkeypatch
) -> None:
    """Writer and reader have to agree, or every entry this version writes is a miss.

    Asserted end to end — download once, then call again and require a hit with no
    second download — because a writer recording the wrong number would still pass both
    miss tests above.
    """
    rdf = {
        "weights": {
            "pytorch_state_dict": {
                "source": "weights.pt",
                "architecture": {"kwargs": {"model_type": "vit_b_lm"}},
            }
        }
    }
    writer = _Caller(rdf=rdf, content=WEIGHTS)

    first = _fetch(writer, tmp_path, monkeypatch)

    assert writer.downloaded
    assert first == {
        "weights_path": str(_cache_dir(tmp_path) / "weights.pt"),
        "model_type": "vit_b_lm",
    }
    on_disk = json.loads((_cache_dir(tmp_path) / "kwargs.json").read_text())
    assert on_disk == {"model_type": "vit_b_lm", "bytes": len(WEIGHTS)}

    reader = _Caller()
    assert _fetch(reader, tmp_path, monkeypatch) == first
    assert not reader.downloaded


def test_the_pre_fix_gate_accepts_both_bad_entries(tmp_path: Path, monkeypatch) -> None:
    """Mutation control: with the length check disabled, both miss tests stop failing.

    The mutation is applied to the shipped function's own source rather than to a
    hand-written copy of the old gate, so this cannot drift into asserting something
    about a reimplementation. If the gate is rewritten such that the marker line is
    gone, this fails loudly instead of silently vacuously passing.
    """
    src = textwrap.dedent(inspect.getsource(_entry_class()._fetch_model_package))
    marker = 'if meta.pop("bytes", None) == weights_path.stat().st_size:'
    assert src.count(marker) == 1, (
        "the length check this control mutates away is no longer in "
        "_fetch_model_package; the two miss tests above may be passing for another "
        "reason"
    )

    import entry

    ns = dict(vars(entry))
    exec(compile(src.replace(marker, "if True or " + marker[3:]), "<pre-fix>", "exec"), ns)
    pre_fix = ns["_fetch_model_package"]

    short = tmp_path / "short"
    short.mkdir()
    _write_entry(short, WEIGHTS[:17], {"model_type": "vit_b_lm", "bytes": len(WEIGHTS)})
    caller = _Caller()
    assert _fetch(caller, short, monkeypatch, fetch=pre_fix)["model_type"] == "vit_b_lm"
    assert not caller.downloaded, "the pre-fix gate was expected to hit, not download"

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    _write_entry(legacy, WEIGHTS, {"model_type": "vit_b_lm"})
    caller = _Caller()
    assert _fetch(caller, legacy, monkeypatch, fetch=pre_fix)["model_type"] == "vit_b_lm"
    assert not caller.downloaded, "the pre-fix gate was expected to hit, not download"
