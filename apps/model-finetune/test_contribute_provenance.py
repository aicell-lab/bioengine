"""The record ``contribute`` returns must date the contribution the same way the
record it stores does.

``app_version`` exists because a field that is absent when unknown cannot be told
apart from one written before versions were stamped at all (``pool.py``,
``build_contribution_metadata``). ``schema_version`` does not substitute for it:
0.27.6 moved every ``l2_from_base`` value without changing the record's shape, so
two responses of the same shape can hold values computed by different releases.

The stored ``metadata.json`` carried ``app_version`` from 0.27.14 (#0116). The dict
returned to the caller did not — and that dict, not the stored file, is what a
depositing site keeps. A campaign auditing 92 saved responses found five shapes it
could not date and had to reconstruct provenance by joining across arms (#0140).

Asserted on the real ``contribute`` coroutine with its collaborators stubbed, and by
comparing the returned value against the metadata bytes actually PUT — not against a
second read of the environment, which would pass even if the two writers diverged.
No GPU, no torch, no live Hypha.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

import pytest

_METADATA_URL = "https://example.invalid/put/metadata"
_INDEX_URL = "https://example.invalid/put/index"
_WEIGHTS_URL = "https://example.invalid/put/weights"

_SESSION = "2026-10-25-020700"
_DIGEST = "a" * 64


def _contribute_of(instance) -> Any:
    """The real ``contribute`` coroutine, bound to a stub instance.

    ``entry.EntryApp`` is a ``ray.serve.deployment.Deployment`` once ``@bioengine.app``
    has run, so its methods are not reachable through it; ``func_or_class`` is the class
    the decorator wrapped. Imported lazily so a missing optional dependency surfaces as
    a test failure rather than a collection error.
    """
    import entry

    return entry.EntryApp.func_or_class.contribute.__get__(instance)


@pytest.fixture
def fake_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """A COMPLETED cpsam session. Stubbed rather than created on disk: this test is
    about what the two records agree on, not about reading a real session."""
    mod = types.ModuleType("training")
    mod.get_status = lambda sid: {
        "status": "COMPLETED", "model_type": "cpsam",
        "n_epochs_completed": 2, "n_train_inputs": 5,
    }
    mod.session_backend = lambda sid: "cellpose"
    mod.read_training_params = lambda sid: {"model_type": "cpsam", "init_checkpoint": "stock"}
    monkeypatch.setitem(sys.modules, "training", mod)


class _Runtime:
    async def prepare_contribution(self, session_id: str, weights_put_url: str) -> Dict[str, Any]:
        return {
            "architecture_signature": "sig-1",
            "base_checkpoint_sha256": "b" * 64,
            "cellpose_version": "4.0.1",
            "l2_from_base": 0.5,
            "weights_sha256": _DIGEST,
            "n_bytes": 1218617275,
        }


class _Stub:
    """Just enough of ``EntryApp`` for ``contribute`` to reach its return."""

    def __init__(self, puts: List[Tuple[str, bytes]]) -> None:
        self._puts = puts
        self._transport: Dict[str, Any] = {}

    async def _check_runtime_available(self, model_type: str) -> None:
        return None

    def _runtime_for(self, model_type: str) -> object:
        return _Runtime()

    async def _pool_service(self, pool_token: str):
        class _Client:
            async def disconnect(self_inner) -> None:
                return None

        class _Am:
            async def edit(self_inner, *a: Any, **k: Any) -> None:
                return None

            async def put_file(self_inner, artifact_id: str, file_path: str) -> str:
                if file_path.endswith("metadata.json"):
                    return _METADATA_URL
                if "weights" in file_path:
                    return _WEIGHTS_URL
                return _INDEX_URL

        return _Am(), _Client()

    async def _read_pool_index(self, am: Any, pool_artifact_id: str) -> Dict[str, Any]:
        raise FileNotFoundError("fresh pool")

    async def _http_retry(self, method: str, url: str, content: bytes = b"", **kw: Any):
        self._puts.append((url, content))

        class _Resp:
            def raise_for_status(self_inner) -> None:
                return None

        return _Resp()

    async def _commit_with_retry(self, am: Any, pool_artifact_id: str) -> None:
        return None

    def _bump_transport(self, **kw: Any) -> None:
        return None


async def _deposit(monkeypatch: pytest.MonkeyPatch, app_version: Optional[str]):
    """One contribute call. Returns (returned_dict, stored_metadata_dict)."""
    if app_version is None:
        monkeypatch.delenv("BIOENGINE_ARTIFACT_VERSION", raising=False)
    else:
        monkeypatch.setenv("BIOENGINE_ARTIFACT_VERSION", app_version)

    puts: List[Tuple[str, bytes]] = []
    returned = await _contribute_of(_Stub(puts))(
        session_id=_SESSION,
        pool_artifact_id="bioimage-io/cpsam-community-pool",
        pool_token="t",
    )
    stored = [json.loads(c) for url, c in puts if url == _METADATA_URL]
    assert len(stored) == 1, f"expected exactly one metadata PUT, got {len(stored)}"
    return returned, stored[0]


@pytest.mark.asyncio
async def test_returned_record_carries_app_version(monkeypatch: pytest.MonkeyPatch, fake_training: None) -> None:
    """The defect: the caller's copy could not be dated at all."""
    returned, _ = await _deposit(monkeypatch, "0.27.16")
    assert "app_version" in returned, (
        "contribute's return value is the only record the depositing site keeps; "
        "without app_version a saved response cannot be dated"
    )
    assert returned["app_version"] == "0.27.16"


@pytest.mark.asyncio
async def test_returned_and_stored_app_version_cannot_disagree(
    monkeypatch: pytest.MonkeyPatch, fake_training: None
) -> None:
    """Compared against the bytes actually PUT, so a second independent read of the
    environment in one of the two writers would fail this rather than pass it."""
    for version in ("0.27.16", "0.99.0"):
        returned, stored = await _deposit(monkeypatch, version)
        assert returned["app_version"] == stored["app_version"] == version


@pytest.mark.asyncio
async def test_app_version_key_is_present_in_the_return_even_when_unknown(
    monkeypatch: pytest.MonkeyPatch, fake_training: None
) -> None:
    """Null is written, not omitted — an absent key is the ambiguity the field removes."""
    returned, stored = await _deposit(monkeypatch, None)
    assert "app_version" in returned
    assert returned["app_version"] is None
    assert stored["app_version"] is None


@pytest.mark.asyncio
async def test_empty_env_var_is_unset_in_both_records(
    monkeypatch: pytest.MonkeyPatch, fake_training: None
) -> None:
    """An exported-but-empty var would otherwise stamp "" and read as a known version."""
    returned, stored = await _deposit(monkeypatch, "")
    assert returned["app_version"] is None
    assert stored["app_version"] is None


@pytest.mark.asyncio
async def test_the_digest_the_campaign_found_null_is_still_recorded_in_both(
    monkeypatch: pytest.MonkeyPatch, fake_training: None
) -> None:
    """#0140's report claimed the digest could be written null. It cannot on this path:
    it comes from ``sha256_file``, which returns a hexdigest or raises. Pinned here so a
    future refactor cannot quietly make null reachable."""
    returned, stored = await _deposit(monkeypatch, "0.27.16")
    assert returned["weights_sha256"] == _DIGEST
    assert stored["weights_sha256"] == _DIGEST
