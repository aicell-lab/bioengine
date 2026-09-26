"""Durable per-app call counters, written from the Hypha proxy.

Every Hypha-exposed call to every BioEngine app passes through
``ProxyDeployment``. This module is what makes that fact countable once the
replica that served it is gone: Ray Serve replica logs die with the Ray
session, so a number derived from them can only ever describe the current
session.

Three properties the storage layout buys:

* **Monotonic.** Shards live next to — not inside — the app cache directory,
  under ``<apps_workdir>/.bioengine-usage/<worker-ws>-<app-id>/``, so they
  survive replica restarts, pod rolls, version bumps and
  ``clear_app_directory``.
* **Concurrency-safe.** Each writer owns exactly one shard file and never
  touches another's, so there is no shared read-modify-write to lose an
  increment and no lock to take. A read sums every shard in the directory.
* **Attributable.** Counts are bucketed by method and by caller *class*
  (:func:`classify_caller`) — never by caller identity. An unattributed total
  is dominated by whoever drives the app hardest, which is usually its own
  operators, and is not a number anyone can publish.

A shard holds running totals rather than one record per call, so it stays
around a kilobyte however long the app runs. Nothing user-identifying is ever
written: no ids, no emails, no tokens, no per-call timestamps, no arguments.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

FORMAT_VERSION = 1

#: Sibling of the per-app cache directories rather than a child of one, so the
#: dashboard's "clear app directory" action frees disk without resetting a
#: counter that is supposed to be cumulative forever.
LEDGER_DIRNAME = ".bioengine-usage"

EVENTS = ("attempted", "answered", "failed")

ANONYMOUS = "anonymous"
EXTERNAL = "external"
INTERNAL = "internal"
UNKNOWN = "unknown"
CALLER_CLASSES = (ANONYMOUS, EXTERNAL, INTERNAL, UNKNOWN)

#: Hypha workspace permission values that imply write access. A caller holding
#: one of these on the workspace hosting the app can operate the deployment, so
#: their traffic is the project's own rather than a user's.
_OPERATOR_GRANTS = frozenset({"rw", "rw+", "a", "*"})

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def ledger_dir_for_app(app_dir: os.PathLike | str) -> Path:
    """Where the shards for the app rooted at ``app_dir`` live."""
    app_dir = Path(app_dir)
    return app_dir.parent / LEDGER_DIRNAME / app_dir.name


def classify_caller(
    context: Optional[Dict[str, Any]],
    app_workspace: Optional[str],
    authorized_users: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Bucket a Hypha caller into one of :data:`CALLER_CLASSES`.

    ``internal`` means the caller can operate this deployment — a write grant
    on the workspace hosting the app (or globally), or a named entry in the
    app's own ``authorized_users``, which carries the deploying user and the
    worker's admin users. ``external`` is any other logged-in caller and
    ``anonymous`` any caller Hypha did not authenticate.

    The split is deliberately biased: a maintainer of the hosting workspace who
    is also a genuine user counts as internal, so the external figure
    under-reports rather than over-reports.
    """
    user = (context or {}).get("user")
    if not isinstance(user, dict):
        return UNKNOWN
    if user.get("is_anonymous"):
        return ANONYMOUS

    identities = {
        value
        for value in (user.get("id"), user.get("email"), user.get("parent"))
        if isinstance(value, str) and value
    }
    if not identities:
        return UNKNOWN

    grants = (user.get("scope") or {}).get("workspaces") or {}
    if isinstance(grants, dict):
        for key in (app_workspace, "*"):
            if key and grants.get(key) in _OPERATOR_GRANTS:
                return INTERNAL

    for entry in (authorized_users or {}).values():
        for name in entry or []:
            if name != "*" and name in identities:
                return INTERNAL

    return EXTERNAL


def _zero() -> Dict[str, int]:
    return {event: 0 for event in EVENTS}


def _add(target: Dict[str, int], source: Dict[str, Any]) -> None:
    for event in EVENTS:
        value = source.get(event, 0)
        if isinstance(value, int) and value > 0:
            target[event] += value


class UsageLedger:
    """Single-writer shard of an app's cumulative call counts.

    Counting is in memory; :meth:`serialize` / :meth:`write` persist the running
    totals. They are split so a caller on an event loop can build the payload
    inline and push only the file write to an executor — the shards sit on
    cluster storage that can stall, and the proxy's event loop must not.
    """

    def __init__(
        self,
        directory: os.PathLike | str,
        writer_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.directory = Path(directory)
        self.writer_id = writer_id
        self.path = self.directory / f"{_SAFE_NAME.sub('-', writer_id)}.json"
        self.metadata = {k: v for k, v in (metadata or {}).items() if v is not None}
        self._counts: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._opened_at = time.time()
        self._revision = 0
        self._written_revision = 0
        self._adopt_existing_shard()

    def _adopt_existing_shard(self) -> None:
        """Resume from our own shard if this writer id has been used before.

        Ray can hand a restarted replica the same id; without this the new
        writer would overwrite its predecessor's totals instead of continuing
        them.
        """
        try:
            shard = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(shard, dict):
            return
        for method, per_class in (shard.get("counts") or {}).items():
            if not isinstance(per_class, dict):
                continue
            for caller_class, counts in per_class.items():
                if isinstance(counts, dict):
                    _add(self._bucket(str(method), str(caller_class)), counts)
        opened_at = shard.get("opened_at")
        if isinstance(opened_at, (int, float)):
            self._opened_at = float(opened_at)

    def _bucket(self, method: str, caller_class: str) -> Dict[str, int]:
        per_class = self._counts.setdefault(method, {})
        return per_class.setdefault(caller_class, _zero())

    def record(self, method: str, caller_class: str, event: str) -> None:
        if event not in EVENTS:
            raise ValueError(f"Unknown usage event {event!r}; expected one of {EVENTS}")
        self._bucket(method, caller_class)[event] += 1
        self._revision += 1

    @property
    def dirty(self) -> bool:
        return self._revision != self._written_revision

    def serialize(self) -> tuple[int, str]:
        payload = {
            "format": FORMAT_VERSION,
            "writer_id": self.writer_id,
            "opened_at": self._opened_at,
            "updated_at": time.time(),
            **self.metadata,
            "counts": self._counts,
        }
        return self._revision, json.dumps(payload)

    def write(self, revision: int, payload: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp.{os.getpid()}")
        try:
            tmp.write_text(payload)
            os.replace(tmp, self.path)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        self._written_revision = revision

    def flush(self) -> None:
        if not self.dirty:
            return
        self.write(*self.serialize())

    def read_all(self) -> Dict[str, Any]:
        return read_usage(self.directory)


def read_usage(directory: os.PathLike | str) -> Dict[str, Any]:
    """Sum every shard in ``directory`` into one cumulative report.

    Unreadable or half-written shards are counted and skipped rather than
    failing the read — a usage figure that is short one shard is still worth
    more than an exception.
    """
    directory = Path(directory)
    totals = _zero()
    by_caller_class: Dict[str, Dict[str, int]] = {}
    by_method: Dict[str, Dict[str, Dict[str, int]]] = {}
    app_versions: set[str] = set()
    shards = 0
    unreadable = 0
    first_seen: Optional[float] = None
    last_updated: Optional[float] = None

    for path in sorted(_shard_paths(directory)):
        try:
            shard = json.loads(path.read_text())
        except (OSError, ValueError):
            unreadable += 1
            continue
        if not isinstance(shard, dict) or not isinstance(shard.get("counts"), dict):
            unreadable += 1
            continue
        shards += 1

        version = shard.get("app_version")
        if isinstance(version, str):
            app_versions.add(version)
        opened_at = shard.get("opened_at")
        if isinstance(opened_at, (int, float)):
            first_seen = opened_at if first_seen is None else min(first_seen, opened_at)
        updated_at = shard.get("updated_at")
        if isinstance(updated_at, (int, float)):
            last_updated = (
                updated_at if last_updated is None else max(last_updated, updated_at)
            )

        for method, per_class in shard["counts"].items():
            if not isinstance(per_class, dict):
                continue
            method_bucket = by_method.setdefault(str(method), {})
            for caller_class, counts in per_class.items():
                if not isinstance(counts, dict):
                    continue
                caller_class = str(caller_class)
                _add(totals, counts)
                _add(by_caller_class.setdefault(caller_class, _zero()), counts)
                _add(method_bucket.setdefault(caller_class, _zero()), counts)

    return {
        "format": FORMAT_VERSION,
        "totals": totals,
        "by_caller_class": by_caller_class,
        "by_method": by_method,
        "app_versions": sorted(app_versions),
        "shards": shards,
        "unreadable_shards": unreadable,
        "first_seen": first_seen,
        "last_updated": last_updated,
    }


def _shard_paths(directory: Path) -> Iterable[Path]:
    try:
        entries = list(directory.iterdir())
    except OSError:
        return []
    return [
        path
        for path in entries
        if path.is_file() and path.suffix == ".json" and not path.name.startswith(".")
    ]


__all__ = [
    "ANONYMOUS",
    "CALLER_CLASSES",
    "EVENTS",
    "EXTERNAL",
    "FORMAT_VERSION",
    "INTERNAL",
    "LEDGER_DIRNAME",
    "UNKNOWN",
    "UsageLedger",
    "classify_caller",
    "ledger_dir_for_app",
    "read_usage",
]
