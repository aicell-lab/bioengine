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

**Only the requirement is read out of the message. Its "available" figure is
unusable, and quietly so.** Serve labels it "total resources available" and fills it
from ``ray.available_resources()`` (``deployment_state.py``,
``DeploymentReplica.resource_requirements``) — what is *free right now*. By the time
Serve writes this message the replica's own lease has already been granted, so its
own requirement has already been subtracted from that figure. Comparing the two
therefore compares a replica against a cluster that has already set its share aside,
and **any replica needing more than half of what was free reports a shortfall
against itself** for the whole of a healthy venv build. Measured: a 3-CPU deployment
waking on an idle 4-CPU cluster reports ``required 3.0, available 1.0`` and then
serves. A one-GPU deployment on a one-GPU node does it on every cold start.

So the comparison here is against **node capacity**, which does not move: a replica
has to fit on ONE node, so a requirement no single node could ever satisfy — however
idle the cluster — is the thing that cannot be waited out. That is immune both to
the self-reservation above and to a sibling transiently holding the resource.

What it still cannot see is a cluster that could grow a bigger node than it
currently has. On a KubeRay cluster with a larger instance type available, this
reports a shortfall for a replica the autoscaler would eventually place. That false
positive is left in, and said out loud, because the alternative is reporting nothing
at all.

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


# Serve states a requirement in Ray's own resource names; the cluster status reports
# per-node capacity under its own. Only these four can be compared. A requirement in
# any other name -- another custom resource, an accelerator label -- has no capacity
# to compare against here and is reported as no shortfall rather than guessed at,
# which is why `unschedulable: false` is never a guarantee of placeability.
#
# VRAM_MB is here because on this platform it is how a GPU app states its
# requirement: ``@bioengine.app`` rejects ``num_gpus`` outright and sizes GPUs by
# ``gpu_memory_mb``, which ``builder.py`` turns into a ``VRAM_MB`` custom resource.
#
# It is NOT here to catch an over-large request at deploy time -- measured on a live
# worker, ``deploy_app`` already refuses that up front: *"gpu_memory_mb=N does not fit
# the smallest GPU on the cluster (M MB)"*. What this catches is the case that gate
# cannot see, which is the one this whole module is about: a deployment that WAS
# placeable and stops being so, because the cluster lost the node that could hold it.
# Without the mapping that case reports nothing at all, since every GPU requirement
# here arrives under this name.
#
# A node that advertises no ``VRAM_MB`` reports ``total_vram_mb: 0`` -- the key is
# always written, never absent, so there is no "unknown VRAM" shape to fall through.
# On the ordinary path that zero cannot meet a VRAM requirement, because
# ``builder.py`` only requests ``VRAM_MB`` for clusters that advertise it: measured on
# a single-machine worker with one GPU, Ray advertised ``accelerator_type:P400`` and no
# ``VRAM_MB``, the node reported ``total_vram_mb: 0``, and the requirement arrived as a
# GPU fraction.
#
# **They CAN co-occur, and then ``True`` is the right answer.** Elastic SLURM requests
# ``VRAM_MB`` before any GPU node exists, so a worker that joins without advertising it
# -- ``num_gpus != 1``, or nvidia-smi detection failed -- leaves a VRAM requirement
# against zero capacity. That replica genuinely cannot be placed, which is exactly what
# this reports. An earlier version of this comment claimed the two could never co-occur;
# that was checked on the single-machine path only and is false.
_CAPACITY_KEY_BY_RESOURCE = {
    "CPU": "total_cpu",
    "GPU": "total_gpu",
    "memory": "total_memory",
    "VRAM_MB": "total_vram_mb",
}


def largest_node_capacity(
    nodes: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, float]:
    """The biggest single-node capacity per comparable resource.

    Per node and not summed, because a replica is one actor and has to fit on one
    node. Summing would call a 2-GPU requirement placeable on two one-GPU nodes,
    which it is not.

    Each resource is maximised independently. That is deliberately generous -- no
    single node need hold all three maxima -- so this errs towards reporting nothing,
    which is the right direction for a signal consumed by health checks.
    """
    capacity: Dict[str, float] = {}
    for node in (nodes or {}).values():
        if not isinstance(node, dict):
            continue
        for resource, key in _CAPACITY_KEY_BY_RESOURCE.items():
            try:
                total = float(node[key])
            except (KeyError, TypeError, ValueError):
                continue
            if total > capacity.get(resource, -1.0):
                capacity[resource] = total
    return capacity


def placement_shortfall(
    message: Optional[str],
    nodes: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Which required resources no single node in the cluster could ever supply.

    Empty whenever the question cannot be answered: no pair in the message, no node
    capacities, or a requirement named in units this cannot map. Absence of evidence
    is reported as no shortfall on purpose -- the consumers are health checks, and an
    unreadable message is not a fault.

    The ``available`` half of Serve's message is deliberately ignored; see the module
    docstring for why it reports a replica as short against itself.
    """
    pair = parse_resource_pair(message)
    if pair is None:
        return {}

    capacity = largest_node_capacity(nodes)
    required, _available = pair
    shortfall: Dict[str, Dict[str, float]] = {}
    for name, needed in required.items():
        if name not in capacity:
            # No capacity figure for this resource, because the cluster status could
            # not be read at all or because the name has no counterpart there. Either
            # way there is nothing to compare, and "we do not know" is reported as no
            # shortfall -- NOT as a capacity of zero, which would turn an unreadable
            # cluster into an unschedulable verdict for every deployment at once.
            continue
        try:
            needed_f = float(needed)
        except (TypeError, ValueError):
            # One unreadable entry says nothing about the others, so the rest are
            # still checked.
            continue
        largest = capacity[name]
        if needed_f > largest + _EPSILON:
            shortfall[name] = {"required": needed_f, "largest_node": largest}
    return shortfall


def describe_placement(
    message: Optional[str],
    nodes: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """The projection ``get_app_status`` adds per deployment.

    ``unschedulable`` is true only when a requirement was read AND no single node
    could ever hold it, so a caller never has to tell an absent key from a false one,
    and never has to parse ``replica_states`` or the message itself.
    """
    shortfall = placement_shortfall(message, nodes)
    return {"unschedulable": bool(shortfall), "placement_shortfall": shortfall}
