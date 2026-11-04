"""``deploy_app(scaling={"X": {"num_replicas": N}})`` must actually pin ``X``.

Two regressions are pinned here, both measured against ray 2.55.1:

1. ``num_replicas=1`` was skipped outright. The bind path guarded the
   ``.options()`` call with ``per_class_replicas is not None and
   per_class_replicas != 1``, so the one value the ``scaling`` field's own
   docstring uses as its example made *no* call at all — the class kept
   whatever ``@bioengine.app`` declared.
2. ``.options(num_replicas=N)`` does not clear an ``autoscaling_config`` the
   decorator already set, and Ray's controller reads ``num_replicas`` only in
   the ``else`` of ``if deployment_config.autoscaling_config``
   (``ray/serve/_private/deployment_state.py``). So *every* fixed count —
   including the ``2`` that reached ``.options()`` before — was silently
   ignored for any autoscaling class. ``apps/model-finetune``'s three classes
   all declare one, two of them with ``max_replicas: 3``.

The fake-class tests assert which kwargs the production helper passes; the
ray-gated test asserts that those kwargs resolve to a pinned deployment
config, which is what makes the fake's premise true. Run with ``-rs``: ray is
not installed on every host, and a silently skipped gate reads as a pass.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from bioengine._app.bootstrap import _apply_scaling


class _FakeDeployment:
    """Records every ``.options()`` call the bind path makes."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def options(self, **kwargs: Any) -> "_FakeDeployment":
        self.calls.append(kwargs)
        return self


def test_num_replicas_one_reaches_ray() -> None:
    """The regression: ``1`` used to make no ``.options()`` call at all."""
    cls = _FakeDeployment()
    assert _apply_scaling(cls, {"num_replicas": 1}) is cls
    assert cls.calls == [{"num_replicas": 1, "autoscaling_config": None}]


def test_fixed_count_clears_the_decorators_autoscaling_config() -> None:
    """A fixed count is only honoured if ``autoscaling_config`` is cleared."""
    for requested in (1, 2, 3):
        cls = _FakeDeployment()
        _apply_scaling(cls, {"num_replicas": requested})
        assert cls.calls == [
            {"num_replicas": requested, "autoscaling_config": None}
        ], f"num_replicas={requested}"


def test_autoscaling_config_entry_is_passed_through_alone() -> None:
    """The sibling branch must not grow a ``num_replicas`` kwarg: Ray rejects
    the two together, and this branch is how #0088 expresses a pin today."""
    cls = _FakeDeployment()
    config = {"min_replicas": 1, "max_replicas": 1}
    _apply_scaling(cls, {"autoscaling_config": config})
    assert cls.calls == [{"autoscaling_config": config}]


def test_class_absent_from_the_scaling_map_is_left_untouched() -> None:
    """No entry means the decorator's own config stands — no ``.options()``."""
    cls = _FakeDeployment()
    assert _apply_scaling(cls, {}) is cls
    assert cls.calls == []


def test_num_replicas_none_with_no_autoscaling_config_is_a_no_op() -> None:
    """``{"num_replicas": None}`` is rejected by the deploy_app validator, so
    the bind path must not turn it into ``.options(num_replicas=None)``."""
    cls = _FakeDeployment()
    assert _apply_scaling(cls, {"num_replicas": None}) is cls
    assert cls.calls == []


def test_resolved_deployment_config_is_pinned() -> None:
    """The kwargs above, applied to a real deployment that autoscales to 3.

    Without ``autoscaling_config=None`` the resolved config keeps
    ``max_replicas=3`` and the controller's autoscaler owns the replica
    count — ``num_replicas`` is then dead config.
    """
    pytest.importorskip("ray")
    from ray import serve

    decorator_autoscaling = {
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 3,
        "upscale_delay_s": 0.0,
    }

    @serve.deployment(
        autoscaling_config=decorator_autoscaling, max_ongoing_requests=10
    )
    class _Runtime:
        def __call__(self) -> int:
            return 1

    # Premise: the decorator really does autoscale, so a pin has work to do.
    assert _Runtime._deployment_config.autoscaling_config is not None

    pinned = _apply_scaling(_Runtime, {"num_replicas": 1})
    assert pinned._deployment_config.num_replicas == 1
    assert pinned._deployment_config.autoscaling_config is None

    two = _apply_scaling(_Runtime, {"num_replicas": 2})
    assert two._deployment_config.num_replicas == 2
    assert two._deployment_config.autoscaling_config is None

    # The autoscaling branch still reaches the resolved config.
    autoscaled = _apply_scaling(
        _Runtime, {"autoscaling_config": {"min_replicas": 2, "max_replicas": 2}}
    )
    resolved = autoscaled._deployment_config.autoscaling_config
    assert (resolved.min_replicas, resolved.max_replicas) == (2, 2)
