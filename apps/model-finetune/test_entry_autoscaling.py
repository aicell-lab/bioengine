"""The EntryApp deployment must declare its autoscaler sampling cadence.

`EntryApp` scales to **zero**. With Serve's defaults (`metrics_interval_s=10.0`,
`look_back_period_s=30.0`) a request arriving at an idle app waits for the next
handle->controller metrics push before the controller can see anything is queued. That was
measured on this app's own cold-start path, varying only that field: phase A median
**14.698 s at 10** against **2.142 s at 1**, a 6.9x collapse. In production the
consequence is visible: the deNBI watcher series shows ~190-490 ms when a replica is up and
9.7-12.6 s when it is not, with nothing in between.

So the cadence is load-bearing, and a later refactor dropping it would silently restore a
ten-second floor on every cold call. These tests exist to make that loud.

**They assert over the AST, not by importing `entry`.** Importing it would pull `bioengine`
and the whole entry module for a question that is answered by the decorator's own literal,
and an import-time failure would read as this check failing rather than as the module being
unloadable. There is deliberately **no `pytest.importorskip`** anywhere here: CI runs
`pytest apps/model-finetune` inside `bioengine-worker`, and a skip would turn this guard into
nothing exactly when it matters -- the shape a previous review spent a refusal on.
"""
import ast
import pathlib

# Imported directly, NOT via importorskip: ray is present in the image CI uses, so an
# absent ray should fail this file loudly rather than skip it.
from ray.serve.config import AutoscalingConfig

HERE = pathlib.Path(__file__).parent

# The ceiling, not the exact value. 1.0 is the measured value; the sibling deployment in
# this same app uses 2.0. Anything at or below 2.0 keeps the cold-start wait off the
# ten-second default, and pinning the exact number would turn a tuning change into a test
# failure.
MAX_METRICS_INTERVAL_S = 2.0


def _entryapp_autoscaling_config():
    """The `autoscaling_config` literal from the decorator on `class EntryApp`.

    Returns a dict of the keys it declares, with constant values where they are literals.
    """
    tree = ast.parse((HERE / "entry.py").read_text())
    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "EntryApp"), None)
    assert cls is not None, "class EntryApp not found in entry.py — this test has stopped looking"
    for dec in cls.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        for kw in dec.keywords:
            if kw.arg != "autoscaling_config" or not isinstance(kw.value, ast.Dict):
                continue
            out = {}
            for k, v in zip(kw.value.keys, kw.value.values):
                if isinstance(k, ast.Constant):
                    try:
                        out[k.value] = ast.literal_eval(v)
                    except Exception:
                        out[k.value] = "<non-literal>"
            return out
    raise AssertionError("no autoscaling_config found on EntryApp's decorator")


def test_the_finder_actually_reads_the_decorator():
    """CONTROL. Without this, every assertion below could pass on an empty dict.

    Checks the finder returns the keys that have been on this deployment since long before
    this change, so a finder that silently matched nothing would fail HERE rather than make
    real assertions vacuous.
    """
    cfg = _entryapp_autoscaling_config()
    assert cfg, "the finder returned nothing; the assertions below would be vacuous"
    for key in ("min_replicas", "max_replicas", "downscale_delay_s"):
        assert key in cfg, f"{key} missing — the finder is not reading the real config"
    assert "zzz_not_a_real_key" not in cfg


def test_entryapp_scales_to_zero_which_is_why_the_cadence_matters():
    """The precondition. If this ever changes, the reason for the next test changes too."""
    cfg = _entryapp_autoscaling_config()
    assert cfg.get("min_replicas") == 0, (
        "EntryApp no longer scales to zero; re-derive whether metrics_interval_s is still "
        "load-bearing before relaxing the next assertion"
    )


def test_entryapp_declares_a_metrics_interval_below_the_serve_default():
    """THE arm this file exists for. Serve's default is 10.0; that is the slow mode."""
    cfg = _entryapp_autoscaling_config()
    assert "metrics_interval_s" in cfg, (
        "EntryApp declares no metrics_interval_s, so it inherits Serve's 10.0 s default — "
        "which was measured as phase A median 14.698 s against 2.142 s at 1.0, and appears "
        "as a 9.7-12.6 s cold call in production"
    )
    val = cfg["metrics_interval_s"]
    assert isinstance(val, (int, float)), f"metrics_interval_s is {val!r}, not a number"
    assert 0 < val <= MAX_METRICS_INTERVAL_S, (
        f"metrics_interval_s is {val}, above the {MAX_METRICS_INTERVAL_S} s ceiling this "
        f"measurement here supports"
    )


def test_the_look_back_window_is_not_left_at_the_serve_default():
    """A short cadence is diluted by a long look-back, so the pair has to move together."""
    cfg = _entryapp_autoscaling_config()
    assert "look_back_period_s" in cfg, (
        "look_back_period_s is unset, so Serve averages the queue signal over its 30 s "
        "default and a 1 s metrics interval buys less than it appears to"
    )
    assert cfg["look_back_period_s"] <= 10.0


def test_both_keys_are_real_autoscaling_config_fields():
    """AutoscalingConfig DROPS unknown keys instead of raising.

    So a renamed or misspelled field reaches Serve and is lost with no warning, and the
    deployment would quietly run on the defaults this file is about.

    This asserts a fact about **Ray**, not about this deployment, so it passes on both
    trees by design — it is the thing that would catch the key being *renamed upstream*,
    which no amount of reading `entry.py` can tell you. The control is the whole point:
    a membership test that returned True for everything would prove nothing.
    """
    fields = AutoscalingConfig.model_fields
    for key in ("metrics_interval_s", "look_back_period_s"):
        assert key in fields, f"{key} is not an AutoscalingConfig field in this Ray"
    assert "zzz_not_a_field" not in fields, (
        "the membership test matched a key that cannot exist, so it discriminates nothing"
    )


def test_the_declared_values_survive_autoscaling_config_construction():
    """The pair of the two above: the keys are real AND these values round-trip.

    Separate from the membership test because they fail for different reasons — that one
    goes red if Ray renames a field, this one if the declared value is a type Serve will
    not take.
    """
    cfg = _entryapp_autoscaling_config()
    built = AutoscalingConfig(
        min_replicas=cfg["min_replicas"],
        max_replicas=cfg["max_replicas"],
        metrics_interval_s=cfg["metrics_interval_s"],
        look_back_period_s=cfg["look_back_period_s"],
    )
    assert built.metrics_interval_s == cfg["metrics_interval_s"]
    assert built.look_back_period_s == cfg["look_back_period_s"]
