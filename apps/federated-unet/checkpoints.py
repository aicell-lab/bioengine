"""Checkpoint exchange over a Hypha artifact, with an append-only audit log.

The federated claim is "no image ever left the site". That is only worth
something if it can be checked, so every byte this module moves in either
direction is recorded: direction, path, size, sha256 and a content kind. The
site app exposes the log as a service method, which makes the audit a call
anyone can make rather than a sentence in a paper.

The run artifact is created staged and never committed. Round checkpoints are
scratch, and a permanently staged artifact keeps them out of any release
history while still being readable by both sites.
"""

import asyncio
import hashlib
import time
from typing import Any, Callable, Dict, List, Optional

import httpx


class TransportLog:
    """Append-only record of everything that crossed the site boundary."""

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []

    def record(self, direction: str, kind: str, path: str, payload: bytes, note: str = "") -> Dict[str, Any]:
        entry = {
            "seq": len(self._entries),
            "timestamp": time.time(),
            "direction": direction,
            "kind": kind,
            "path": path,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "note": note,
        }
        self._entries.append(entry)
        return entry

    def dump(self) -> Dict[str, Any]:
        outbound = [e for e in self._entries if e["direction"] == "out"]
        kinds = sorted({e["kind"] for e in self._entries})
        return {
            "entries": list(self._entries),
            "n_transfers": len(self._entries),
            "bytes_out": sum(e["bytes"] for e in outbound),
            "bytes_in": sum(e["bytes"] for e in self._entries if e["direction"] == "in"),
            "kinds_transferred": kinds,
            # The assertion the experiment rests on, stated as a check over the
            # log rather than as a claim: nothing but model weights left here.
            "only_weights_left_site": all(e["kind"] == "model_weights" for e in outbound),
        }


class CheckpointStore:
    """Read and write checkpoint files on a staged Hypha artifact."""

    def __init__(
        self,
        artifact_manager: Any,
        artifact_id: str,
        log: TransportLog,
        timeout: float = 600.0,
        attempts: int = 5,
    ) -> None:
        self.artifact_manager = artifact_manager
        self.artifact_id = artifact_id
        self.log = log
        self.timeout = timeout
        self.attempts = attempts

    async def _retry(self, describe: str, transfer: Callable):
        """Re-presign and retry with exponential backoff.

        A multi-hour run moves thousands of checkpoints, so a transient S3 or
        gateway hiccup is a matter of when, not if — one truncated chunked read
        once killed a run 42 rounds in. The URL is re-requested on every attempt
        because a presigned URL can expire while the transfer is being retried.
        """
        for attempt in range(self.attempts):
            try:
                return await transfer()
            except (httpx.HTTPError, httpx.StreamError) as error:
                if attempt == self.attempts - 1:
                    raise
                delay = 2.0**attempt
                print(
                    f"transfer {describe} failed ({type(error).__name__}: {error}), "
                    f"retrying in {delay:.0f}s [{attempt + 1}/{self.attempts - 1}]",
                    flush=True,
                )
                await asyncio.sleep(delay)

    async def put(self, path: str, payload: bytes, kind: str = "model_weights", note: str = "") -> Dict[str, Any]:
        async def _transfer() -> None:
            url = await self.artifact_manager.put_file(self.artifact_id, file_path=path)
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.put(url, content=payload)
                response.raise_for_status()

        await self._retry(f"put {path}", _transfer)
        return self.log.record("out", kind, f"{self.artifact_id}/{path}", payload, note)

    async def get(self, path: str, kind: str = "model_weights", note: str = "") -> bytes:
        async def _transfer() -> bytes:
            url = await self.artifact_manager.get_file(self.artifact_id, path, version="stage")
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
            return response.content

        payload = await self._retry(f"get {path}", _transfer)
        self.log.record("in", kind, f"{self.artifact_id}/{path}", payload, note)
        return payload


async def ensure_run_artifact(
    artifact_manager: Any,
    alias: str,
    manifest: Optional[Dict[str, Any]] = None,
) -> str:
    """Create the staged run artifact if it does not already exist."""
    try:
        existing = await artifact_manager.read(alias)
        return existing["id"]
    except Exception:
        created = await artifact_manager.create(
            type="generic",
            alias=alias,
            manifest=manifest or {"name": alias, "description": "federated-unet run checkpoints"},
            stage=True,
        )
        return created["id"]
