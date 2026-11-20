"""A deployment that cannot be placed must be distinguishable from one mid-wake-up.

Both are ``UPSCALING`` at zero ``RUNNING`` replicas, and Serve writes the same
slow-start message for both, so the only thing that separates them is the
required-vs-available pair inside that message.

**The arm that matters is the negative one.** A previous attempt keyed on the
message's wording — the phrase "Resources required for each replica" — and was
refuted live: Ray builds a replica's ``runtime_env`` venv *before* the actor process
starts, so a scale-from-zero wake-up is ``PENDING_ALLOCATION`` for the whole build
and gets that same phrase. Every cold wake-up read as unplaceable. The
``placeable_wake_up`` message below is that live case, with the numbers the repro
actually produced (``num_cpus=0.1`` on an otherwise idle 4-CPU cluster), and the
arm asserting it reports NO shortfall is RED against the phrase-matching version.

Messages are Serve's own, copied from ``_check_and_update_transitioning_replicas``
in ``ray/serve/_private/deployment_state.py``. ``test_serve_message_shape_still_matches``
pins that they are still Serve's by reading its source, so a Serve rewording fails
here instead of quietly turning every reading into "no pair".
"""
from __future__ import annotations

import inspect
import json
import re

import pytest

from bioengine.apps import placement
from bioengine.apps.placement import (
    describe_placement,
    parse_resource_pair,
    placement_shortfall,
)


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

# The live repro: a trivially placeable replica whose wake-up is building a venv.
PLACEABLE_WAKE_UP = _slow_allocation({"CPU": 0.1}, {"CPU": 3.9})

# A GPU the cluster has none of free.
UNPLACEABLE = _slow_allocation({"CPU": 1, "GPU": 1}, {"CPU": 8})


class TestTheDistinction:
    """The one case the previous attempt got backwards, and its opposite."""

    def test_a_placeable_wake_up_reports_no_shortfall(self):
        # RED against a predicate that matches the message's wording: this message
        # carries the phrase and is a perfectly healthy cold start.
        assert placement_shortfall(PLACEABLE_WAKE_UP) == {}
        assert describe_placement(PLACEABLE_WAKE_UP)["unschedulable"] is False

    def test_an_unmeetable_requirement_reports_the_resource_that_falls_short(self):
        assert placement_shortfall(UNPLACEABLE) == {
            "GPU": {"required": 1.0, "available": 0.0}
        }
        assert describe_placement(UNPLACEABLE)["unschedulable"] is True

    def test_both_messages_carry_the_phrase_the_old_predicate_keyed_on(self):
        # Without this, the arm above could pass for the wrong reason -- a message
        # that simply does not look like the one the old predicate matched.
        phrase = "Resources required for each replica"
        assert phrase in PLACEABLE_WAKE_UP
        assert phrase in UNPLACEABLE


class TestParsing:
    def test_a_decimal_in_available_is_not_truncated(self):
        # Pins that the end of each value is found by decoding JSON, not by
        # scanning to the next ".". A regex ending at a full stop reads 1.5 as 1
        # and then reports a shortfall that is not there.
        msg = _slow_allocation({"CPU": 2.0}, {"CPU": 1.5})
        required, available = parse_resource_pair(msg)
        assert available == {"CPU": 1.5}
        assert placement_shortfall(msg) == {
            "CPU": {"required": 2.0, "available": 1.5}
        }

    def test_a_requirement_exactly_met_is_not_a_shortfall(self):
        assert placement_shortfall(_slow_allocation({"CPU": 0.1}, {"CPU": 0.1})) == {}

    def test_every_short_resource_is_named_not_just_the_first(self):
        msg = _slow_allocation({"CPU": 16, "GPU": 2, "memory": 8}, {"CPU": 4, "memory": 8})
        assert sorted(placement_shortfall(msg)) == ["CPU", "GPU"]


class TestNoPairMeansNoVerdict:
    """Absence of evidence is reported as no shortfall, deliberately."""

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
                _slow_allocation([{"CPU": 1}, {"GPU": 1}], {"CPU": 0}),
                id="placement-group-bundles-arrive-as-a-list",
            ),
            pytest.param(
                "Resources required for each replica: {\"CPU\": 1}, and then nothing",
                id="pair-truncated-after-the-first-value",
            ),
        ],
    )
    def test_unreadable_or_absent_pairs_report_nothing(self, message):
        assert parse_resource_pair(message) is None
        assert placement_shortfall(message) == {}
        assert describe_placement(message)["unschedulable"] is False

    def test_a_bundle_list_would_otherwise_have_looked_short(self):
        # The list case is a deliberate gap, not an oversight: a bundle must fit on
        # one node while `available` is cluster-wide. Asserting it here means
        # closing the gap turns this arm RED rather than passing unnoticed.
        msg = _slow_allocation([{"GPU": 1}], {})
        assert placement_shortfall(msg) == {}


class TestProjection:
    def test_both_keys_are_always_present(self):
        # An absent key and a false one read the same through .get(), so a consumer
        # must never have to tell them apart.
        for message in (None, PLACEABLE_WAKE_UP, UNPLACEABLE):
            projection = describe_placement(message)
            assert set(projection) == {"unschedulable", "placement_shortfall"}

    def test_it_does_not_reuse_the_application_level_field_name(self):
        # `resource_shortfall` already means the deploy-time check's report, at the
        # application level. Two different computations under one name is a trap.
        assert "resource_shortfall" not in describe_placement(UNPLACEABLE)


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

    Serve builds the message from adjacent f-string literals, so the source is
    joined at those seams first; otherwise a label that straddles two literals could
    never be found however correct it is.
    """
    source = inspect.getsource(ray_serve_source)
    joined = re.sub(r'"\s*\n\s*f?"', "", source)

    assert placement._REQUIRED_PREFIX + "{required}" in joined, (
        "Serve no longer writes the requirement under this label"
    )
    assert "{required}" + placement._AVAILABLE_PREFIX + "{available}" in joined, (
        "Serve no longer writes availability under this label, or not right after "
        "the requirement"
    )
