"""Per-dataset access requests for the BioEngine datasets server.

A user who is not in a dataset's ``authorized_users`` can ask for access; a
named approver grants, denies or clears the request. Grants are held here
rather than written back into the dataset's ``manifest.yaml`` because a dataset
directory is served in place and may sit on a read-only mount — the server can
always write its own state directory, but it cannot assume it may edit a data
owner's manifest.

That makes this store an additive overlay on the manifest, in the same shape as
the worker's ``admin_users.json`` overlay on ``--admin-users``: the manifest is
the seed, granted records widen it, and nothing here can take away what the
manifest grants.

Who may decide a request is *not* recorded here. This server keeps no admin
list; it asks the worker, on a connection carrying the approver's own token, so
the worker's live admin list stays the single source of truth. See
``is_worker_admin``.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# parse_token maps an absent token onto these placeholders, and Hypha reports no
# email at all for an anonymous websocket. Either way there is no account a
# grant could name.
UNAUTHENTICATED_EMAILS = frozenset({"anonymous@example.com", "no-email", "anonymous-user"})

DECISIONS = ("grant", "deny", "clear")


async def is_worker_admin(
    token: Optional[str],
    worker_service_id: str,
    server_url: str,
    logger: logging.Logger = logging.getLogger("AccessRequestStore"),
) -> bool:
    """Ask the worker whether the holder of ``token`` is one of its admins.

    The question is asked *as the caller*: the connection carries the caller's
    own token, the worker's service is registered with ``require_context``, and
    ``check_access`` takes no arguments — it reports on whoever called it. So
    nothing here asserts an identity on someone else's behalf, and there is no
    way to ask about a third party, which would be an oracle for enumerating
    the worker's admin list.

    It is deliberately not cached. The worker's admin list is editable at
    runtime through ``add_admin_user`` / ``remove_admin_user``, and a cached
    answer would mean a revoked admin kept deciding requests. Approving is a
    low-frequency action, so the round trip costs nothing that matters.

    Any failure — worker down, wrong service id, network — returns False, so an
    unreachable worker refuses decisions rather than waving them through.
    """
    if not token:
        # Refuse before opening an outbound connection: an unauthenticated
        # caller is never an admin, and each attempt would otherwise cost a
        # websocket to Hypha.
        return False

    from hypha_rpc import connect_to_server

    try:
        async with connect_to_server(
            {"server_url": server_url, "token": token}
        ) as client:
            worker = await client.get_service(worker_service_id)
            return bool(await worker.check_access())
    except Exception as e:
        logger.warning(
            f"Could not confirm worker admin status against "
            f"'{worker_service_id}': {type(e).__name__}: {e}. Refusing the decision."
        )
        return False


def requester_identity(user_info: Dict[str, Any]) -> Tuple[str, str]:
    """``(key, email)`` for the caller: the dedup key and what a grant records.

    Email, not user id: Hypha's ``generate_token`` mints a fresh client id per
    token while inheriting the email, so an id-keyed request would let one
    person file unlimited requests by refreshing their token.

    The two differ by case on purpose. The key is lowercased so one account
    cannot hold several requests by varying capitalisation, but the stored email
    has to be the string Hypha will report on the caller's next request, because
    ``check_permissions`` compares it case-sensitively.

    Raises:
        PermissionError: If the caller carries no usable account identity.
    """
    if not isinstance(user_info, dict):
        raise PermissionError("Invalid user information for a dataset access request.")

    email = (user_info.get("email") or "").strip()
    # Anonymous identities are free and unlimited — Hypha mints a fresh random id
    # per connection and reports no email — so one-request-per-user would bound
    # nothing for them.
    if (
        user_info.get("is_anonymous")
        or not email
        or email.lower() in UNAUTHENTICATED_EMAILS
        or "@" not in email
    ):
        raise PermissionError(
            "Log in before requesting access to a dataset. An unauthenticated "
            "caller carries no email address, so the request would name no "
            "account that could be granted."
        )
    return email.lower(), email


class AccessRequestStore:
    """Persisted per-dataset access requests, keyed ``dataset_id`` → email.

    Constructed even when the feature is off, so ``enabled`` is the second of
    two independent gates: the routes are not registered at all on a server
    started without ``--enable-access-requests``, and every method here refuses
    as well, in case some later wiring forgets the registration gate.
    """

    def __init__(
        self,
        store_file: Path,
        worker_service_id: Optional[str] = None,
        enabled: bool = False,
        logger: logging.Logger = logging.getLogger("AccessRequestStore"),
    ):
        self.store_file = Path(store_file)
        # Who may decide a request is not a list this server keeps. It asks the
        # worker, whose admin list is the single source of truth and changes at
        # runtime via add_admin_user / remove_admin_user.
        self.worker_service_id = (worker_service_id or "").strip() or None
        self.logger = logger
        if enabled and not self.worker_service_id:
            # Requesting is a public write to this server's disk; with nobody
            # able to drain the queue it is an unbounded one, and a request that
            # can never be decided is worse than no request surface at all.
            self.logger.warning(
                "Access requests were enabled but no worker service id was given, "
                "so no decision could ever be authorized. The request endpoints "
                "stay off — set --worker-service-id to turn them on."
            )
            enabled = False
        self.enabled = enabled
        self._requests = self._load()
        if not self.enabled:
            self._warn_about_dropped_grants()

    def _warn_about_dropped_grants(self) -> None:
        """Say out loud how much access the disabled overlay is withholding.

        Switching the feature off revokes every granted address, which is the
        safe direction but is otherwise invisible: the grants live only here,
        so nothing in any manifest explains why those users stopped having
        access. Counting them turns a silent revoke into a visible one.
        """
        dropped = [
            record
            for records in self._requests.values()
            for record in records.values()
            if record.get("status") == "granted"
        ]
        if not dropped:
            return
        datasets = {record.get("dataset_id") for record in dropped}
        self.logger.warning(
            f"Access requests are disabled, so {len(dropped)} existing grant(s) "
            f"across {len(datasets)} dataset(s) in '{self.store_file}' are NOT "
            "being honoured; those users lose the access they were granted. "
            "Re-enable access requests to restore it, or add them to the "
            "dataset's authorized_users to make it permanent."
        )

    def _load(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        if not self.store_file.exists():
            return {}
        try:
            stored = json.loads(self.store_file.read_text())
            if not isinstance(stored, dict) or not all(
                isinstance(dataset_id, str)
                and isinstance(records, dict)
                and all(
                    isinstance(key, str) and isinstance(record, dict)
                    for key, record in records.items()
                )
                for dataset_id, records in stored.items()
            ):
                raise ValueError("expected an object of dataset_id -> email -> request")
        except Exception as e:
            self.logger.error(
                f"Ignoring unreadable dataset access requests file '{self.store_file}': "
                f"{e}. Starting with no requests on record; pending requesters will "
                "have to ask again."
            )
            return {}
        return stored

    def _persist(self, requests: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """Write before the in-memory copy changes.

        A decision that only exists in memory reverts on the next restart
        without ever failing.
        """
        try:
            self.store_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = self.store_file.with_suffix(".json.tmp")
            tmp_file.write_text(json.dumps(requests, indent=2))
            os.replace(tmp_file, self.store_file)
        except Exception as e:
            raise RuntimeError(
                f"Failed to persist dataset access requests to '{self.store_file}': "
                f"{e}. Access requests are unchanged."
            )

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError(
                "Dataset access requests are disabled on this server. Start it with "
                "--enable-access-requests to let users ask for access."
            )

    def granted_users(self, dataset_id: str) -> List[str]:
        """Emails granted access to this dataset, as the manifest would spell them.

        Empty while the feature is off: an overlay grant is invisible in the
        dataset's manifest, so a server whose request surface is switched off
        must not keep honouring one.
        """
        if not self.enabled:
            return []
        return [
            record["email"]
            for record in self._requests.get(dataset_id, {}).values()
            if record.get("status") == "granted" and record.get("email")
        ]

    def get(self, dataset_id: str, key: str) -> Optional[Dict[str, Any]]:
        """One caller's own request on one dataset, or None."""
        self._require_enabled()
        record = self._requests.get(dataset_id, {}).get(key)
        return dict(record) if record else None

    def list_all(self) -> List[Dict[str, Any]]:
        """Every request across every dataset, oldest first."""
        self._require_enabled()
        return sorted(
            (
                dict(record)
                for records in self._requests.values()
                for record in records.values()
            ),
            key=lambda record: record.get("requested_at") or 0,
        )

    def submit(
        self, dataset_id: str, key: str, email: str, user_id: Optional[str], reason: str
    ) -> Dict[str, Any]:
        """Record a pending request.

        One per account per dataset. A second is refused rather than queued or
        silently replacing the first, so a decision already taken cannot be
        reset by asking again — including a denial, which only an approver clears.
        """
        self._require_enabled()
        existing = self._requests.get(dataset_id, {}).get(key)
        if existing:
            raise ValueError(
                f"'{email}' already has an access request for dataset '{dataset_id}' "
                f"with status '{existing.get('status')}'. Only an approver can clear it."
            )

        record = {
            "dataset_id": dataset_id,
            # The email as Hypha reports it, which is what a grant has to put in
            # front of check_permissions; the dict key is its lowercased form.
            "email": email,
            "user_id": user_id,
            "status": "pending",
            "reason": str(reason or "").strip()[:500],
            "requested_at": time.time(),
            "resolved_at": None,
            "resolved_by": None,
        }
        self._persist(
            {
                **self._requests,
                dataset_id: {**self._requests.get(dataset_id, {}), key: record},
            }
        )
        self._requests.setdefault(dataset_id, {})[key] = record
        self.logger.info(
            f"Access request from '{email}' for dataset '{dataset_id}' is pending."
        )
        return dict(record)

    def resolve(
        self, dataset_id: str, user: str, decision: str, resolved_by: str
    ) -> Optional[Dict[str, Any]]:
        """Grant, deny or clear one request.

        'grant' is the record itself — there is no second list to write, because
        the manifest the grant widens may not be writable. 'deny' stays in place
        so the requester cannot re-file. 'clear' removes the record, which both
        lifts a denial and revokes a grant.
        """
        self._require_enabled()
        if decision not in DECISIONS:
            raise ValueError(
                f"Unknown decision '{decision}'; expected one of {', '.join(DECISIONS)}."
            )

        # Requests are keyed on the lowercased email, so an approver may name the
        # requester in any case and still reach the record.
        key = (user or "").strip().lower()
        record = self._requests.get(dataset_id, {}).get(key)
        if not record:
            raise ValueError(
                f"No access request for '{key}' on dataset '{dataset_id}'."
            )

        if decision == "clear":
            remaining = {
                stored_key: value
                for stored_key, value in self._requests[dataset_id].items()
                if stored_key != key
            }
            self._persist({**self._requests, dataset_id: remaining})
            self._requests[dataset_id] = remaining
            self.logger.info(
                f"Cleared the access request for '{key}' on dataset '{dataset_id}' "
                f"(by '{resolved_by}'). They may request access again."
            )
            return None

        updated = {
            **record,
            "status": "granted" if decision == "grant" else "denied",
            "resolved_at": time.time(),
            "resolved_by": resolved_by,
        }
        self._persist(
            {
                **self._requests,
                dataset_id: {**self._requests[dataset_id], key: updated},
            }
        )
        self._requests[dataset_id][key] = updated
        self.logger.info(
            f"Access request from '{record['email']}' for dataset '{dataset_id}' "
            f"was {updated['status']} by '{resolved_by}'."
        )
        return dict(updated)
