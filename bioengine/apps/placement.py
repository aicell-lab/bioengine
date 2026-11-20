"""Read Ray Serve's resource pair out of a deployment status message.

When replicas stay ``PENDING_ALLOCATION`` past Serve's slow-start threshold, Serve
writes the deployment's requirement and the cluster's free resources into the
deployment status message. Nothing else in the status carries that pair:
``update_message`` keeps ``status_trigger`` as it was, and a deployment waiting for
a placement and one waiting for a ``runtime_env`` venv are both ``UPSCALING`` at
zero ``RUNNING`` replicas.

**The phrase is not the signal.** Ray builds a replica's venv before the actor
process starts, so a scale-from-zero wake-up is ``PENDING_ALLOCATION`` for the whole
build and gets this same message — Serve's own wording says so ("waiting for the
cluster to auto-scale or for a runtime environment to be installed"). Matching the
sentence therefore flags every cold wake-up. What separates the two cases is the
*pair*: a wake-up whose replicas fit reports a requirement the cluster can meet.

**What ``available`` actually is, which the message gets wrong.** Serve labels it
"total resources available" and fills it from ``ray.available_resources()``
(``deployment_state.py``, ``DeploymentReplica.resource_requirements``), which is
what is *free right now*. A shortfall reported here therefore means "does not fit at
this moment", not "can never fit": a cluster still adding a node, or a sibling
holding the GPUs for another minute, both produce one. Treat it as a reason to look,
not as a verdict.

The field is called ``placement_shortfall`` and not ``resource_shortfall`` because
an application already carries a ``resource_shortfall`` — the deploy-time report
``_check_resources`` leaves behind when an external cluster had no node with room.
That one is written once, per application, before anything runs; this one is read
off the live status, per deployment, while it runs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

# Serve's message, verbatim around the two values, from
# ``_check_and_update_transitioning_replicas``. Matched as literals rather than by
# regex because both values are JSON and may contain commas, braces and decimal
# points; ``raw_decode`` finds where each one ends.
_REQUIRED_PREFIX = "Resources required for each replica: "
_AVAILABLE_PREFIX = ", total resources available: "

_DECODER = json.JSONDecoder()

# Float slack, so a requirement that equals availability after a round trip through
# JSON is not read as exceeding it.
_EPSILON = 1e-9


def parse_resource_pair(
    message: Optional[str],
) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
    """``(required, available)`` from a Serve status message, or ``None``.

    ``None`` means this message carries no pair this can read, which covers a
    message about something else, a deployment whose actor resources Serve reports
    as ``UNKNOWN``, and placement-group bundles — Serve sends those as a JSON *list*
    of per-bundle requirements, and a bundle has to fit on one node while
    ``available`` is cluster-wide, so the two are not comparable and are not
    guessed at here.
    """
    if not message:
        return None

    start = message.find(_REQUIRED_PREFIX)
    if start < 0:
        return None

    try:
        required, end = _DECODER.raw_decode(message, start + len(_REQUIRED_PREFIX))
    except ValueError:
        return None

    if not message.startswith(_AVAILABLE_PREFIX, end):
        return None

    try:
        available, _ = _DECODER.raw_decode(message, end + len(_AVAILABLE_PREFIX))
    except ValueError:
        return None

    if not isinstance(required, dict) or not isinstance(available, dict):
        return None

    return required, available


def placement_shortfall(message: Optional[str]) -> Dict[str, Dict[str, float]]:
    """Which required resources the cluster could not supply, keyed by name.

    Empty when the deployment's requirement is met, and empty when no pair could be
    read — absence of evidence is reported as no shortfall on purpose, because the
    consumers are health checks and an unreadable message is not a fault.

    Serve filters ``available`` down to the keys in ``required``, so a required name
    missing from it has none free.
    """
    pair = parse_resource_pair(message)
    if pair is None:
        return {}

    required, available = pair
    shortfall: Dict[str, Dict[str, float]] = {}
    for name, needed in required.items():
        try:
            needed_f = float(needed)
            have_f = float(available.get(name, 0.0))
        except (TypeError, ValueError):
            # A value this cannot compare is not a shortfall. One unreadable entry
            # does not say anything about the others, so the rest are still checked.
            continue
        if needed_f > have_f + _EPSILON:
            shortfall[name] = {"required": needed_f, "available": have_f}
    return shortfall


def describe_placement(message: Optional[str]) -> Dict[str, Any]:
    """The projection ``get_app_status`` adds per deployment.

    ``unschedulable`` is true only when a pair was read AND it falls short, so a
    caller never has to tell an absent key from a false one, and never has to parse
    ``replica_states`` or the message itself.
    """
    shortfall = placement_shortfall(message)
    return {"unschedulable": bool(shortfall), "placement_shortfall": shortfall}
