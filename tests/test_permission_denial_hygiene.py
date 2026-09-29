"""A permission denial must not hand the refused caller the allowlist.

``check_permissions`` is the single gate behind every admin-only worker method,
every per-dataset HTTP route and every app manifest's ``authorized_users``, so
these assertions are made against the helper rather than against one gated
method. Asserting only that ``list_admin_users`` is clean would leave every
sibling caller open to a one-method patch.

The worker service is registered with ``"visibility": "public"``, and the
datasets proxy puts ``str(exc)`` straight into an HTTP 403 body, so this string
reaches unauthenticated callers on the open internet.
"""

import ast
from pathlib import Path

import pytest

from bioengine.utils import check_permissions, create_context

# Stand-ins for the real maintainer addresses. The local parts are checked
# separately so a partial leak ("authorized: alice, bob") still fails.
ALLOWLIST = [
    "alice.admin@lab.example.invalid",
    "bob.admin@other.example.invalid",
    "user-id-42",
]

ANONYMOUS = create_context("anonymouz-http", None)
NAMED_OUTSIDER = create_context("carol-id", "carol@elsewhere.example.invalid")


def _fragments(allowlist):
    for entry in allowlist:
        yield entry
        if "@" in entry:
            yield entry.split("@", 1)[0]
            yield entry.split("@", 1)[1]


def _denial_message(context, authorized_users, **kwargs) -> str:
    with pytest.raises(PermissionError) as excinfo:
        check_permissions(
            context=context,
            authorized_users=authorized_users,
            resource_name="listing BioEngine Worker admin users",
            **kwargs,
        )
    return str(excinfo.value)


@pytest.mark.parametrize(
    "context", [ANONYMOUS, NAMED_OUTSIDER], ids=["anonymous", "named"]
)
@pytest.mark.parametrize(
    "authorized_users",
    [ALLOWLIST, ALLOWLIST[0], [*ALLOWLIST, "*"]],
    ids=["list", "single-string", "wildcard-plus-named"],
)
def test_denial_never_names_the_allowlist(context, authorized_users):
    kwargs = {"allow_wildcard": False} if "*" in authorized_users else {}
    message = _denial_message(context, authorized_users, **kwargs)

    for fragment in _fragments(
        authorized_users if isinstance(authorized_users, list) else [authorized_users]
    ):
        assert fragment not in message, f"denial message disclosed '{fragment}'"


def test_denial_still_says_who_was_refused_and_what_was_refused():
    """Dropping the allowlist must not cost the caller the diagnostic half."""
    message = _denial_message(NAMED_OUTSIDER, ALLOWLIST)

    assert "carol-id" in message
    assert "carol@elsewhere.example.invalid" in message
    assert "listing BioEngine Worker admin users" in message


def test_denial_tells_the_caller_what_to_do_instead():
    message = _denial_message(ANONYMOUS, ALLOWLIST)

    assert "administrator" in message.lower()


def test_an_empty_allowlist_denial_names_nobody():
    for authorized_users in (None, []):
        message = _denial_message(ANONYMOUS, authorized_users)
        assert "@" not in message


def test_no_permission_denial_in_the_package_interpolates_an_identity_list():
    """A tripwire for the obvious regression, not a proof that it cannot happen.

    A behavioural test on the helper only protects the helper. Any module that
    grows its own denial path — ``proxy_deployment._check_permissions`` already
    has one — can reintroduce the disclosure without touching ``permissions.py``.
    This catches the two forms a reviewer would plausibly type at such a raise:
    a bare local and ``self.<attr>``.

    It does not close the class, and widening it to try is a losing game against
    the AST. Known blind spots: a constant subscript (``self.app_data['authorized_users']``,
    a live expression at ``proxy_deployment.py:496``), building the message into
    a local first, a qualified ``raise builtins.PermissionError``, and any
    disclosure raised as a different exception type — including the
    ``HTTPException`` the datasets proxy raises. The behavioural tests above are
    the real guarantee.
    """
    package = Path(__file__).resolve().parent.parent / "bioengine"
    leaky = []

    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            call = node.exc
            if (
                not isinstance(call, ast.Call)
                or getattr(call.func, "id", None) != "PermissionError"
            ):
                continue
            # Attributes too: inside a class the idiomatic reintroduction is
            # `self.authorized_users`, which is not an ast.Name.
            names = [n.id for n in ast.walk(call) if isinstance(n, ast.Name)]
            names += [n.attr for n in ast.walk(call) if isinstance(n, ast.Attribute)]
            for name in names:
                if any(
                    word in name.lower()
                    for word in ("users", "admins", "allowed", "allowlist")
                ):
                    leaky.append(
                        f"{path.relative_to(package.parent)}:{node.lineno} -> {name}"
                    )

    assert (
        not leaky
    ), "PermissionError messages interpolating an identity list: " + ", ".join(leaky)
