"""#0070: a bootstrapped pool must record a base commitment, or it can never merge.

`d0679f8f` added `pool.base_commitment()` / `assert_base_commitment()` and wired the READ
path, but no production code ever CALLED the builder: the only pool-bootstrap site,
`entry.py`, passed two positional arguments to `empty_index(model_type,
architecture_signature, base_commitment=None)`. So a freshly created pool recorded
`base_commitment: None` and `assert_base_commitment` raised out of `uniform_soup` — the
branch could not merge any pool at all, existing or new.

A unit test of `base_commitment()` cannot catch that, because the function was correct;
what was missing was a caller. Hence the two structural tests here, which assert on the
production source. They are deliberately the kind of test that fails when a writer
disappears rather than when it misbehaves, and each carries a control so it cannot pass by
failing to look.

Torch-free on purpose: this must run in any image, including one without cellpose.
"""
import ast
import pathlib

import pytest

import pool

HERE = pathlib.Path(__file__).parent
SIG = "a" * 64


def _complete_commitment():
    """A commitment with every field `assert_base_commitment` requires.

    Built as a literal rather than through `pool.base_commitment`, so these tests still
    fail if the BUILDER and the VALIDATOR drift apart — going through the builder would
    make the pair self-consistent by construction and assert nothing about the contract.
    """
    return {
        "non_learned": {"diam_labels": 30.0, "diam_mean": 30.0},
        "key_set_sha256": "c" * 64,
        "key_count": 350,
        "source": "session_init_checkpoint:" + "b" * 64,
    }


# === the no-fallback rule, which is the ruling's and not negotiable ==================

def test_bootstrapped_index_with_a_commitment_is_mergeable():
    idx = pool.empty_index("cpsam", SIG, _complete_commitment())
    assert idx["base_commitment"] is not None
    # assert_base_commitment RETURNS the commitment on success, so this also pins that it
    # does not quietly normalise it away.
    got = pool.assert_base_commitment(idx["base_commitment"])
    assert got["key_count"] == 350
    assert got["source"].startswith("session_init_checkpoint:")


def test_bootstrapped_index_without_a_commitment_still_raises():
    """The defect's signature: this is what a pool created by the old call site looked like.

    It must RAISE rather than fall back to loading a stock base, which the ruling forbids
    because a fallback makes the field permanently optional.
    """
    idx = pool.empty_index("cpsam", SIG)
    assert idx["base_commitment"] is None
    with pytest.raises(ValueError, match="records no base commitment"):
        pool.assert_base_commitment(idx["base_commitment"])


@pytest.mark.parametrize("missing", ["non_learned", "key_set_sha256", "key_count", "source"])
def test_an_incomplete_commitment_raises_naming_the_field(missing):
    c = _complete_commitment()
    c[missing] = None
    with pytest.raises(ValueError, match="incomplete"):
        pool.assert_base_commitment(c)


# === the structural guards: a writer that does not exist ============================

def _calls_to(tree, func_name):
    """Every Call node whose callee attribute or name is `func_name`."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name == func_name:
            out.append(node)
    return out


def test_every_production_empty_index_call_passes_a_base_commitment():
    """The exact defect: `pool.empty_index(model_type, signature)` with no third argument.

    Asserted over entry.py's AST rather than by running the contribute path, because the
    bug was a missing argument at one call site and that is visible in the source without
    a GPU, a pool or a live runtime.
    """
    tree = ast.parse((HERE / "entry.py").read_text())
    calls = _calls_to(tree, "empty_index")
    # CONTROL on the finder itself: if the call site is ever renamed or removed, an empty
    # list would make every assertion below vacuously true and this test would pass while
    # checking nothing.
    assert calls, "no empty_index call found in entry.py - this test has stopped looking"
    for call in calls:
        n = len(call.args) + len(call.keywords)
        assert n >= 3, (
            "entry.py line %d calls empty_index with %d argument(s): a pool bootstrapped "
            "without a base commitment can never merge (#0070)" % (call.lineno, n)
        )


def test_the_runtime_builds_a_base_commitment():
    """`pool.base_commitment` must have a PRODUCTION caller, not only tests and a bench.

    At `d0679f8f` its only callers were test_soup_math.py and bench_soup_memory.py, which
    is exactly how a correct builder ships with nothing writing its output.
    """
    tree = ast.parse((HERE / "runtime_cellpose.py").read_text())
    calls = _calls_to(tree, "base_commitment")
    assert calls, (
        "runtime_cellpose.py never calls pool.base_commitment, so nothing in production "
        "produces a commitment for the entry to write (#0070)"
    )
    # CONTROL: the same finder must report nothing for a name that cannot be there, so a
    # bug that made it match everything would be visible.
    assert not _calls_to(tree, "zzz_not_a_function_name")


def test_prepare_contribution_advertises_the_commitment_and_the_model_type():
    """The seam the entry depends on: the key it reads, and the argument it passes.

    `model_type` is load-bearing rather than cosmetic — it selects the stock base for a
    session that kept no init_checkpoint.pt, which is the branch where the commitment
    would otherwise be unbuildable.
    """
    tree = ast.parse((HERE / "runtime_cellpose.py").read_text())
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.AsyncFunctionDef) and n.name == "prepare_contribution"),
              None)
    assert fn is not None, "prepare_contribution not found - this test has stopped looking"
    arg_names = [a.arg for a in fn.args.args]
    assert "model_type" in arg_names, (
        "prepare_contribution takes no model_type, so the entry cannot say which stock "
        "base to commit to for a stock-initialised session (#0070)"
    )
    # The returned dict must mention the key the entry reads. A string search over the
    # function's own source only, so an unrelated mention elsewhere in the file cannot
    # satisfy it.
    src = ast.get_source_segment((HERE / "runtime_cellpose.py").read_text(), fn) or ""
    assert "base_commitment" in src, (
        "prepare_contribution does not mention base_commitment, so the entry's "
        "info.get('base_commitment') would always be None (#0070)"
    )
