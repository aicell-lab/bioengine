"""Mask credential-shaped VALUES, not just credential-shaped NAMES.

Worker logs and status payloads used to decide what to hide purely from the
caller's key: a leading underscore meant "secret". That rule fails **open** for
every name nobody anticipated. The same credential passed as ``HF_READ_TOKEN``
rather than ``_HF_READ_TOKEN`` went into a log verbatim — and because the
masked line and the leaked line look equally deliberate, nothing drew attention
to the one that leaked. It took six weeks and a credential rotation to notice.

Matching on the value fails **closed** instead: an unfamiliar name no longer
decides whether a credential is printed. The name rule stays on top as an
explicit opt-in, so a caller can still mask something these patterns cannot
recognise.

**The patterns are deliberately narrow**, and that is a judgement worth stating
rather than burying. They cover issuer-prefixed tokens, JWTs and PEM private
keys — shapes that are credentials and essentially nothing else. A generic
"long random-looking string" rule would also mask image digests, commit SHAs,
artifact ids and URLs, which operators read constantly; an operator who cannot
read their own logs routes around the redaction, and that costs more than it
saves.

So this is **not** a claim to catch every secret. It is a floor under the name
rule: the known-bad shapes can no longer be printed because somebody spelled
the variable differently.
"""

from __future__ import annotations

import re
from typing import Any

#: What a masked value renders as. Unchanged from the name-prefix rule it
#: extends, so existing log readers and UI see the same token.
SECRET_MASK = "*****"

#: The explicit opt-in. A caller who knows a value is sensitive can still say
#: so by name, including for shapes below that nothing here recognises.
SECRET_ENV_VAR_PREFIX = "_"

#: Credential shapes. Each is anchored and requires a substantial body, so a
#: short lookalike ("sk-1", "hf_abc") does not trip it.
_SECRET_VALUE_PATTERNS = (
    # Hugging Face user/read tokens.
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    # GitHub: personal access, OAuth, user-to-server, server-to-server, refresh.
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    # GitHub FINE-GRAINED PATs do not use the two-letter family above -- they are
    # `github_pat_` followed by an underscore-separated body, so the `\b…{20,}\b`
    # rule misses them entirely. These are the tokens GitHub now issues by default.
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    # OpenAI-style and the many services that copied the prefix.
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    # A JWT: three base64url segments, and the header all but always starts
    # `eyJ` because it opens with `{"`. Hypha tokens are JWTs.
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    # Any PEM private key, however it is labelled (RSA, EC, OPENSSH, plain).
    re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----"),
)


def looks_like_secret(value: Any) -> bool:
    """True if ``value`` contains something shaped like a credential.

    Substring rather than full match on purpose: a secret is frequently
    embedded in something larger — a URL with a token in it, a JSON blob, a
    connection string — and a full-match rule would print all of those.

    Non-strings are never secret-shaped: an int, a bool or ``None`` carries no
    credential, and coercing them to text to scan would only invent matches.
    """
    if not isinstance(value, str):
        return False
    return any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS)


def is_secret_env_var(key: str, value: Any) -> bool:
    """True if this key/value pair must not be rendered in cleartext.

    Both rules, in the order that makes the name rule an *addition*: an
    underscore-prefixed key is secret whatever its value, and any other key is
    secret if its value is credential-shaped.
    """
    if isinstance(key, str) and key.startswith(SECRET_ENV_VAR_PREFIX):
        return True
    return looks_like_secret(value)


def _render_pairs(pairs: Any) -> str:
    """One deployment's ``key=value`` pairs, masking anything secret-shaped."""
    return " ".join(
        f'{key}="{SECRET_MASK}"' if is_secret_env_var(key, value) else f"{key}={value!r}"
        for key, value in pairs.items()
    )


def describe_deploy_arguments(application_kwargs: Any, application_env_vars: Any) -> tuple:
    """The ``kwargs:``/``env_vars:`` text of the deploy log line, already masked.

    Pulled out of ``AppsManager.deploy_app`` so it can be tested on its OUTPUT. The test
    that stood in for this before only asserted that ``deploy_app`` *mentioned* the matcher
    somewhere in its source; deleting the guard from the cleartext half left that test green
    while the log printed every secret in clear. A guard you cannot render is not a guard.

    **``application_kwargs`` is masked by the same rule**, which it was not before. The
    kwargs and the env vars sit on the SAME log line, so a credential passed as a kwarg under
    an unremarkable name was written in cleartext beside the carefully masked env vars -- the
    identical failure, one argument along, and the one this row's own earlier sweep called the
    worst remaining gap.

    Values are rendered with ``repr`` exactly as before for everything not masked, so an
    operator reading these lines sees what they always saw.
    """
    kwargs_str = (
        ", ".join(
            f"{deployment_class}(" + ", ".join(
                f'{key}="{SECRET_MASK}"' if is_secret_env_var(key, value) else f"{key}={value!r}"
                for key, value in deployment_kwargs.items()
            ) + ")"
            for deployment_class, deployment_kwargs in application_kwargs.items()
        )
        if application_kwargs
        else "None"
    )
    env_vars_str = (
        ", ".join(
            f"{deployment_class}: " + _render_pairs(env_vars)
            for deployment_class, env_vars in application_env_vars.items()
        )
        if application_env_vars
        else "None"
    )
    return kwargs_str, env_vars_str
