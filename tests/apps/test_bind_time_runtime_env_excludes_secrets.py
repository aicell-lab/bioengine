"""#0534: the one line that keeps secrets out of the per-replica runtime_env is guarded here.

Reverting `builder.py`'s ``split_secret_env_vars`` call left the whole suite green, so the invariant
this PR exists for was protected by nothing. The existing tests
(``test_hypha_token_explicit_clear.py``, ``tests/_app/test_env_vars_propagation.py``) call
``split_secret_env_vars`` **directly** and assert on its return value — they establish that the
function works, not that ``build()`` uses it. Remove the call site and every one of them still
passes.

WHAT THIS GUARDS, and it is the bind-time surface specifically: ``build()`` must derive
``replica_env_vars`` from ``split_secret_env_vars(env_vars)`` and hand *that* to ``BuiltApp``.
``BuiltApp.replica_env_vars`` is what bootstrap merges into every deployment's ``runtime_env`` at
bind time, so if it is the unsplit ``env_vars`` then every ``_BIOENGINE_SECRET_*`` value is written
into Ray's ``runtime_env_agent.log`` on every node — which is #0517.

WHY THIS IS STRUCTURAL AND NOT BEHAVIOURAL, stated so nobody mistakes its reach: ``build()`` is a
291-line async method taking 12 arguments, and **no test in this suite drives it** — doing so needs
the introspect Ray task, Hypha artifact access and a Serve binding stubbed. A harness that large is
itself a place for a vacuous pass to hide. These assertions read the wiring out of the AST instead,
which is enough to fail on the revert that prompted the row and is the same approach #0321's stored
verify takes.

WHAT IT DOES NOT CATCH, and this is the honest boundary:
  * ``split_secret_env_vars`` being changed to split nothing. That function has its own direct tests
    and they would go red.
  * a secret arriving under a name the prefix does not match. ``BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN``
    is exactly that case and is tracked on #0522, not here.
  * anything about the proxy-actor delivery path, which has its own tests.
"""

import ast
import pathlib

BUILDER = (
    pathlib.Path(__file__).resolve().parents[2] / "bioengine" / "apps" / "builder.py"
)
SPLIT = "split_secret_env_vars"
TARGET = "replica_env_vars"


def _build_fn() -> ast.AsyncFunctionDef:
    tree = ast.parse(BUILDER.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "build":
            return node
    raise AssertionError("builder.py has no `async def build` -- the anchor moved, not the fix")


def _assigns_from_split(fn: ast.AST) -> bool:
    """True if some assignment in `fn` binds TARGET from a call to SPLIT."""
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        names = {
            n.id
            for t in node.targets
            for n in ast.walk(t)
            if isinstance(n, ast.Name)
        }
        if TARGET not in names:
            continue
        for call in ast.walk(node.value):
            if isinstance(call, ast.Call):
                fname = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
                if fname == SPLIT:
                    return True
    return False


def _builtapp_kwarg_source(fn: ast.AST) -> str | None:
    """The NAME passed as BuiltApp(replica_env_vars=...), or None if not a bare name."""
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        fname = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if fname != "BuiltApp":
            continue
        for kw in node.keywords:
            if kw.arg == TARGET:
                return getattr(kw.value, "id", None)
    return None


def test_build_derives_replica_env_vars_from_the_split() -> None:
    """The bind-time env_vars must come OUT of split_secret_env_vars, not straight from env_vars."""
    fn = _build_fn()
    assert _assigns_from_split(fn), (
        f"`{TARGET}` in builder.build() is not assigned from `{SPLIT}(...)`. That is #0517: "
        f"BuiltApp.{TARGET} is merged into every replica's runtime_env at bind time, so unsplit "
        "env_vars put every _BIOENGINE_SECRET_* value into Ray's runtime_env_agent.log on every node."
    )


def test_builtapp_receives_the_split_result_and_not_raw_env_vars() -> None:
    """Splitting and then passing the wrong name through would be the same leak."""
    fn = _build_fn()
    passed = _builtapp_kwarg_source(fn)
    assert passed is not None, (
        f"BuiltApp(...) in builder.build() does not pass `{TARGET}=` as a bare name, so this test "
        "cannot tell which value reaches the bind-time runtime_env. Re-read the call site."
    )
    assert passed == TARGET, (
        f"BuiltApp received `{TARGET}={passed}`. It must receive the split result bound to "
        f"`{TARGET}`; anything else risks handing the unsplit env_vars to the replica runtime_env."
    )
    assert passed != "env_vars", (
        "BuiltApp receives the raw `env_vars` as the bind-time replica env — this is #0517 exactly."
    )


def test_the_prefix_the_split_keys_on_is_still_the_one_the_fix_documents() -> None:
    """A renamed prefix would leave both assertions above green while leaking everything."""
    src = BUILDER.read_text()
    assert '_SECRET_ENV_VAR_PREFIX = "_BIOENGINE_SECRET_"' in src, (
        "the secret prefix constant in builder.py is no longer _BIOENGINE_SECRET_. The split would "
        "then key on something the worker never sets, and every secret would travel in the clear "
        "while the wiring assertions above still pass."
    )
