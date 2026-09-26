"""Liveness heartbeat file written by the BioEngine worker's monitoring loop.

Lives at the top level rather than in ``bioengine.worker`` because importing
that package pulls in Ray and hypha_rpc (~2 s) — far too slow and too heavy for
a container liveness probe, which runs this module instead::

    python -m bioengine.heartbeat /path/to/worker_heartbeat.json

Exit code 0 means the monitoring loop completed a full pass recently enough,
1 means it did not (or the file is missing/unreadable). The freshness deadline
travels inside the file, so the probe needs no knowledge of the worker's
monitoring interval.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

MISSED_PASSES_BEFORE_STALE = 6

# Node-local by default: a heartbeat on a network filesystem makes the write
# (on the event loop) and the probe's read hostage to that filesystem, and two
# pods sharing one RWX volume would write the same file.
DEFAULT_HEARTBEAT_PATH = Path(tempfile.gettempdir()) / "bioengine_worker_heartbeat.json"

# Deadline carried by the heartbeat written before the monitoring loop exists.
# Starting a Ray cluster and deploying startup applications has no useful upper
# bound, so this only rules out a worker that never got anywhere at all.
STARTUP_STALE_AFTER_SECONDS = 1800.0


def heartbeat_stale_after_seconds(
    monitoring_interval_seconds: float,
    backoff_max_seconds: float,
) -> float:
    """Longest gap between two monitoring passes that still counts as alive.

    A loop whose steps keep failing is alive but slow: between two beats it
    sleeps up to ``backoff_max_seconds``, waits out its own one-second tick and
    then runs a whole pass. A pass has no bounded duration of its own — the
    Hypha check alone allows ten seconds, and reconnecting is untimed — so the
    backoff is cleared exactly and the missed-pass budget on top of it is what
    a slow pass is spent out of.
    """
    return (
        backoff_max_seconds + MISSED_PASSES_BEFORE_STALE * monitoring_interval_seconds
    )


def write_heartbeat(
    path: Union[str, Path],
    stale_after_seconds: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"timestamp": time.time(), "stale_after_seconds": stale_after_seconds}
    )
    # Replace atomically so a probe can never read a half-written file.
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(payload)
    os.replace(tmp_path, path)


def check_heartbeat(path: Union[str, Path]) -> Tuple[bool, str]:
    """Return whether the heartbeat at ``path`` is fresh, plus a reason."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
        timestamp = float(payload["timestamp"])
        stale_after = float(payload["stale_after_seconds"])
    except FileNotFoundError:
        return False, f"No heartbeat file at {path}"
    except Exception as e:
        return False, f"Unreadable heartbeat file at {path}: {e}"

    age = time.time() - timestamp
    if age > stale_after:
        return False, (
            f"Monitoring loop last completed a pass {age:.0f}s ago, "
            f"more than the {stale_after:.0f}s deadline"
        )
    return True, (
        f"Monitoring loop last completed a pass {age:.0f}s ago, "
        f"within the {stale_after:.0f}s deadline"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bioengine.heartbeat",
        description=(
            "Check that a BioEngine worker's monitoring loop is still running. "
            "Exits 0 if the heartbeat file is fresh, 1 if it is stale, missing "
            "or unreadable. Purely local: it never contacts the Hypha server."
        ),
    )
    parser.add_argument(
        "path",
        metavar="PATH",
        help="Path to the heartbeat file written by the worker "
        "(see the worker's --heartbeat-file option).",
    )
    args = parser.parse_args(argv)

    is_fresh, reason = check_heartbeat(args.path)
    print(reason)
    return 0 if is_fresh else 1


if __name__ == "__main__":
    sys.exit(main())
