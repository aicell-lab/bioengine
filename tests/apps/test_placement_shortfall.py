"""A deployment that cannot be placed must be distinguishable from one mid-wake-up.

Both are ``UPSCALING`` at zero ``RUNNING`` replicas, and Serve writes the same
slow-start message for both.

Two wrong readings have already been built and refuted here, and the arms below are
shaped by them:

1. **Matching the message's wording.** Ray builds a replica's ``runtime_env`` venv
   *before* the actor process starts, so an ordinary scale-from-zero wake-up is
   ``PENDING_ALLOCATION`` for the whole build and carries the same phrase. Every cold
   start read as unplaceable.
2. **Comparing the requirement against the message's own "available" figure.** By the
   time Serve writes the message the replica's lease is already granted, so its own
   requirement has been subtracted from that figure. Measured live: a 3-CPU
   deployment waking on an idle 4-CPU cluster reports ``required 3.0, available 1.0``
   and then serves. Any replica needing more than half of what was free reports a
   shortfall against itself.

So the comparison is against **node capacity**, and ``SELF_RESERVING_WAKE_UP`` below
is case 2 with the live numbers. The arm asserting it is placeable is red against
both earlier versions.

Messages are Serve's own, copied from ``_check_and_update_transitioning_replicas`` in
``ray/serve/_private/deployment_state.py``.
``test_the_literals_this_parses_are_the_ones_serve_writes`` pins them against that
source, so a Serve rewording fails here instead of quietly turning every reading into
"no pair".
"""

from __future__ import annotations

import inspect
import json
import re

import pytest

from bioengine.apps import placement
from bioengine.apps.placement import (
    describe_placement,
    largest_node_capacity,
    parse_resource_pair,
    placement_shortfall,
)

GIB = 1024**3


def _slow_allocation(required, available, *, replicas: int = 1, threshold: int = 30):
    """Serve's PENDING_ALLOCATION message, assembled exactly as Serve assembles it."""
    return (
        f"Deployment 'EntryApp' in application 'demo' has {replicas} replicas that "
        f"have taken more than {threshold}s to be scheduled. "
        "This may be due to waiting for the cluster to auto-scale or for a "
        "runtime environment to be installed. "
        f"Resources required for each replica: {json.dumps(required)}, "
        f"total resources available: {json.dumps(available)}. "
        "Use `ray status` for more details."
    )


SLOW_INITIALIZATION = (
    "Deployment 'EntryApp' in application 'demo' has 1 replicas that have taken "
    "more than 30s to initialize.\n"
    "This may be caused by a slow __init__ or reconfigure method."
)


def _node(cpu=4.0, gpu=0.0, memory=16 * GIB, vram=0.0):
    return {
        "total_cpu": cpu,
        "total_gpu": gpu,
        "total_memory": memory,
        "total_vram_mb": vram,
    }


ONE_SMALL_NODE = {"n1": _node(cpu=4.0, gpu=1.0, vram=24000.0)}
TWO_SMALL_NODES = {
    "n1": _node(cpu=4.0, gpu=1.0, vram=24000.0),
    "n2": _node(cpu=4.0, gpu=1.0, vram=24000.0),
}

# The live repro, and the reason this file exists in its current shape: a 3-CPU
# replica waking on an idle 4-CPU node. Serve has already granted its lease, so the
# message says only 1.0 is free -- against the replica's own 3.0.
SELF_RESERVING_WAKE_UP = _slow_allocation({"CPU": 3.0}, {"CPU": 1.0})

# The same cluster, a requirement no node could ever hold.
TOO_BIG_FOR_ANY_NODE = _slow_allocation({"CPU": 5.0}, {"CPU": 4.0})

# A trivially small wake-up: placeable under either reading, so it cannot by itself
# tell the two apart.
TINY_WAKE_UP = _slow_allocation({"CPU": 0.1}, {"CPU": 3.9})


class TestTheDistinction:
    """The two cases every previous version got wrong, and their opposites."""

    def test_a_replica_that_fits_a_node_is_placeable_while_it_reserves_itself(self):
        # RED against comparing with the message's "available" (3.0 > 1.0), and RED
        # against matching the message's wording. This is a healthy cold start.
        assert placement_shortfall(SELF_RESERVING_WAKE_UP, ONE_SMALL_NODE) == {}
        assert (
            describe_placement(SELF_RESERVING_WAKE_UP, ONE_SMALL_NODE)["unschedulable"]
            is False
        )

    def test_a_requirement_no_node_could_hold_is_reported(self):
        assert placement_shortfall(TOO_BIG_FOR_ANY_NODE, ONE_SMALL_NODE) == {
            "CPU": {"required": 5.0, "largest_node": 4.0}
        }
        assert (
            describe_placement(TOO_BIG_FOR_ANY_NODE, ONE_SMALL_NODE)["unschedulable"]
            is True
        )

    def test_the_two_cases_are_indistinguishable_in_the_message_alone(self):
        # Without this, the pair above could pass for the wrong reason -- one message
        # simply not looking like the other. Both carry the phrase the first attempt
        # matched, and in both the requirement exceeds the stated availability.
        phrase = "Resources required for each replica"
        for message in (SELF_RESERVING_WAKE_UP, TOO_BIG_FOR_ANY_NODE):
            required, available = parse_resource_pair(message)
            assert phrase in message
            assert required["CPU"] > available["CPU"]

    def test_capacity_is_per_node_not_summed(self):
        # Two one-GPU nodes cannot run a two-GPU replica. Summing would say they can.
        message = _slow_allocation({"GPU": 2.0}, {"GPU": 0.0})
        assert placement_shortfall(message, TWO_SMALL_NODES) == {
            "GPU": {"required": 2.0, "largest_node": 1.0}
        }


class TestVramIsTheGpuRequirementOnThisPlatform:
    """``@bioengine.app`` rejects ``num_gpus``; GPUs are sized by ``gpu_memory_mb``.

    Measured on a live worker 2026-11-20: deploying an app with ``num_gpus`` raises
    *"@bioengine.app no longer accepts 'num_gpus' -- GPUs are now sized by VRAM"*, and
    ``builder.py`` turns ``gpu_memory_mb`` into a ``VRAM_MB`` custom resource. So a
    VRAM requirement is not an exotic case here -- it is how every GPU app asks, and
    asking for more than any device has is the one failure that queues forever
    instead of failing.
    """

    def test_more_vram_than_any_device_has_is_reported(self):
        message = _slow_allocation({"CPU": 1, "VRAM_MB": 80000}, {"CPU": 3})
        assert placement_shortfall(message, ONE_SMALL_NODE) == {
            "VRAM_MB": {"required": 80000.0, "largest_node": 24000.0}
        }

    def test_vram_that_fits_a_device_is_not(self):
        message = _slow_allocation({"CPU": 1, "VRAM_MB": 8000}, {"CPU": 3})
        assert placement_shortfall(message, ONE_SMALL_NODE) == {}

    def test_vram_capacity_is_per_node_too(self):
        # Two 24 GB cards on two nodes do not serve one 40 GB replica.
        message = _slow_allocation({"VRAM_MB": 40000}, {})
        assert placement_shortfall(message, TWO_SMALL_NODES) == {
            "VRAM_MB": {"required": 40000.0, "largest_node": 24000.0}
        }

    def test_zero_vram_capacity_against_a_vram_requirement_IS_a_shortfall(self):
        """The shape `proxy_actor` actually produces, which is not the obvious one.

        It always writes `total_vram_mb`, as 0 when the node advertises no VRAM -- the
        key is never absent. On the ordinary path a VRAM requirement cannot arrive on
        such a cluster, because `builder.py` only requests `VRAM_MB` where it is
        advertised. But elastic SLURM requests it before any GPU node exists, so a
        worker that joins without advertising it leaves exactly this pair -- and then
        the replica really cannot be placed, so True is correct.

        An earlier version of this arm used a node with NO `total_vram_mb` key and
        asserted no shortfall. `proxy_actor` never produces that shape, so it pinned a
        branch real data cannot reach.
        """
        nodes = {
            "n1": {
                "total_cpu": 4.0,
                "total_gpu": 1.0,
                "total_memory": GIB,
                "total_vram_mb": 0,
            }
        }
        message = _slow_allocation({"CPU": 1, "GPU": 0.01, "VRAM_MB": 8000}, {})
        assert placement_shortfall(message, nodes) == {
            "VRAM_MB": {"required": 8000.0, "largest_node": 0.0}
        }

    def test_a_node_status_missing_the_key_entirely_still_reports_nothing(self):
        # Not a shape proxy_actor produces, but the fail-safe that keeps an
        # unreadable status from becoming a verdict. Kept as the defensive case it
        # is, and labelled so nobody mistakes it for the real-data path above.
        nodes = {"n1": {"total_cpu": 4.0, "total_gpu": 1.0, "total_memory": GIB}}
        assert (
            placement_shortfall(_slow_allocation({"VRAM_MB": 80000}, {}), nodes) == {}
        )


class TestCapacity:
    def test_the_largest_node_wins_per_resource(self):
        nodes = {
            "small": _node(cpu=4.0, gpu=0.0),
            "big": _node(cpu=64.0, gpu=8.0),
        }
        capacity = largest_node_capacity(nodes)
        assert capacity["CPU"] == 64.0
        assert capacity["GPU"] == 8.0

    def test_a_node_missing_a_key_does_not_erase_the_others(self):
        nodes = {"ok": _node(cpu=8.0), "broken": {"total_cpu": "NA"}}
        assert largest_node_capacity(nodes)["CPU"] == 8.0

    @pytest.mark.parametrize("nodes", [None, {}, {"n": "not a dict"}])
    def test_no_usable_capacity_yields_nothing(self, nodes):
        assert largest_node_capacity(nodes) == {}


class TestParsing:
    def test_a_decimal_is_not_truncated(self):
        # Pins that each value's end is found by decoding JSON, not by scanning to
        # the next ".". A regex ending at a full stop reads 1.5 as 1.
        message = _slow_allocation({"CPU": 1.5}, {"CPU": 0.0})
        required, available = parse_resource_pair(message)
        assert required == {"CPU": 1.5}
        assert available == {"CPU": 0.0}

    def test_a_requirement_exactly_filling_a_node_is_not_a_shortfall(self):
        assert (
            placement_shortfall(_slow_allocation({"CPU": 4.0}, {}), ONE_SMALL_NODE)
            == {}
        )

    def test_every_short_resource_is_named_not_just_the_first(self):
        message = _slow_allocation({"CPU": 16, "GPU": 4, "memory": 8}, {})
        assert sorted(placement_shortfall(message, ONE_SMALL_NODE)) == ["CPU", "GPU"]


class TestNoVerdictWithoutEvidence:
    """Everything this cannot answer is reported as no shortfall, deliberately."""

    @pytest.mark.parametrize(
        "message",
        [
            pytest.param(None, id="none"),
            pytest.param("", id="empty"),
            pytest.param(SLOW_INITIALIZATION, id="slow-initialization-carries-no-pair"),
            pytest.param(
                "Deployment 'EntryApp' in application 'demo' has 1 replicas that have "
                "taken more than 30s to be scheduled. "
                "Resources required for each replica: UNKNOWN, "
                "total resources available: UNKNOWN. Use `ray status` for more details.",
                id="serve-reports-unknown",
            ),
            pytest.param(
                _slow_allocation([{"CPU": 1}, {"GPU": 1}], {}),
                id="placement-group-bundles-arrive-as-a-list",
            ),
            pytest.param(
                'Resources required for each replica: {"CPU": 1}, and then nothing',
                id="pair-truncated-after-the-first-value",
            ),
        ],
    )
    def test_an_unreadable_requirement_reports_nothing(self, message):
        assert parse_resource_pair(message) is None
        assert placement_shortfall(message, ONE_SMALL_NODE) == {}
        assert describe_placement(message, ONE_SMALL_NODE)["unschedulable"] is False

    @pytest.mark.parametrize("nodes", [None, {}])
    def test_an_impossible_requirement_with_no_capacity_to_compare_reports_nothing(
        self, nodes
    ):
        # Not knowing the cluster is not evidence that a replica fits, and it is not
        # evidence that it does not. A health check gets silence either way.
        assert placement_shortfall(TOO_BIG_FOR_ANY_NODE, nodes) == {}

    def test_a_resource_with_no_comparable_capacity_reports_nothing(self):
        # An accelerator label is a real requirement with no counterpart in the node
        # status, so `unschedulable: false` is never a guarantee of placeability.
        # Asserting the gap means closing it turns this arm red.
        message = _slow_allocation({"accelerator_type:A100": 1, "CPU": 1}, {})
        assert placement_shortfall(message, ONE_SMALL_NODE) == {}

    def test_a_bundle_list_would_otherwise_have_looked_short(self):
        # A bundle must fit on one node and arrives as a JSON list; deliberately not
        # interpreted rather than guessed at.
        assert (
            placement_shortfall(_slow_allocation([{"GPU": 4}], {}), ONE_SMALL_NODE)
            == {}
        )


class TestProjection:
    def test_both_keys_are_always_present(self):
        # An absent key and a false one read the same through .get(), so a consumer
        # must never have to tell them apart.
        for message in (None, TINY_WAKE_UP, TOO_BIG_FOR_ANY_NODE):
            projection = describe_placement(message, ONE_SMALL_NODE)
            assert set(projection) == {"unschedulable", "placement_shortfall"}

    def test_it_does_not_reuse_the_application_level_field_name(self):
        # `resource_shortfall` already means the deploy-time check's report, at the
        # application level. Two different computations under one name is a trap.
        assert "resource_shortfall" not in describe_placement(
            TOO_BIG_FOR_ANY_NODE, ONE_SMALL_NODE
        )


ray_serve_source = pytest.importorskip(
    "ray.serve._private.deployment_state",
    reason="Serve's source is what the message literals are pinned against",
)


def test_the_literals_this_parses_are_the_ones_serve_writes():
    """Pin the module's own constants against Serve's source.

    Keyed on Serve's source rather than on a message this file wrote, because the
    module is a coupling to text Serve owns. If Serve rewords either label -- or if
    someone edits a constant here -- every reading silently becomes "no pair", and a
    fully green suite then reports that nothing is ever unschedulable.

    Serve builds the message from adjacent f-string literals, so the source is joined
    at those seams first; otherwise a label straddling two literals could never be
    found however correct it is.
    """
    source = inspect.getsource(ray_serve_source)
    joined = re.sub(r'"\s*\n\s*f?"', "", source)

    assert (
        placement._REQUIRED_PREFIX + "{required}" in joined
    ), "Serve no longer writes the requirement under this label"
    assert (
        "{required}" + placement._AVAILABLE_PREFIX + "{available}" in joined
    ), "Serve no longer writes availability under this label, or not right after it"
