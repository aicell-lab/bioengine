"""annotation-broker — standing authority for shared annotation datasets.

CPU-only BioEngine app. Holds the ``bioimage-io`` workspace token and owns
role/permission metadata, presigned URL handout, label folder creation, and
Hypha ACL mirroring for every dataset artifact under
``bioimage-io/colab-annotations``. Historically this logic lived in the host
browser's Pyodide kernel (``colab_service.py``) and died with the host's
tab; this app makes annotation sessions survive independent of any single
browser tab.

This module is the thin transport layer: it owns the Hypha connection and
artifact-manager calls. All decision logic (role resolution, path building,
metadata read/write, the ACL-permissions mirror, the stage-mode retry
wrapper) lives in ``broker_core.py``, which is plain Python and unit-tested
without Ray or a live Hypha connection.
"""

import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import bioengine
import httpx
from hypha_rpc import connect_to_server
from pydantic import Field

import broker_core as core

logger = bioengine.logger

SERVER_URL = "https://hypha.aicell.io"
COLLECTION_ID = "bioimage-io/colab-annotations"
# PVC-persistent in the deployed environment, so registered datasets survive
# app restarts.
STATE_ROOT = Path.home() / "annotation_broker"
MANIFEST_CACHE_TTL_S = 60.0


# A send-side RPC failure: the request never reached the server, so re-resolving
# the service and retrying is safe even for writes.
_HEALTH_CALL_TIMEOUT_S = 15.0

_SEND_FAILURE_MARKERS = (
    "Failed to send the request",
    "WebSocket reconnection timed out",
)


#: The page size used when a directory has to be read in FULL. `list_files` is
#: paginated with a default page of 1000 and a documented range of 1..10000, so
#: a single call answers a PAGE, not a directory. This asks for the largest
#: page the server documents, which keeps the number of round trips down -- the
#: paging loop is what makes the answer complete, not this number.
_LIST_PAGE = 10000

#: How many consecutive pages may add NOTHING before the read is called
#: incomplete. A filtered overlap stalls for at most as many pages as there are
#: staged-removed entries in the window; a server ignoring `offset` stalls for
#: ever. Generous, because the cost of being wrong in one direction is a refused
#: upload and in the other is one extra page read.
_NO_PROGRESS_PAGES = 8

#: `list_files`'s own default page. Used only to judge whether a SINGLE
#: unpaginated call can be called complete: a result shorter than this cannot
#: have been truncated by the default, while one at or above it may have been.
_SERVER_DEFAULT_PAGE = 1000


def _is_send_failure(exc: Exception) -> bool:
    return any(marker in str(exc) for marker in _SEND_FAILURE_MARKERS)


class _ReconnectingService:
    """Auto-re-resolving wrapper around a Hypha service proxy.

    hypha_rpc pins a proxy to one *client instance* of the target service. When
    the Hypha server restarts, that instance is gone and every call fails at
    send time forever — the websocket underneath reconnects, but the proxy still
    addresses the dead client id. Re-resolve and retry once instead.
    """

    def __init__(self, resolve) -> None:
        self._resolve = resolve
        self._svc = None

    async def _service(self):
        if self._svc is None:
            self._svc = await self._resolve()
        return self._svc

    def __getattr__(self, name: str):
        async def call(*args, **kwargs):
            svc = await self._service()
            try:
                return await getattr(svc, name)(*args, **kwargs)
            except Exception as exc:
                if not _is_send_failure(exc):
                    raise
                logger.warning(
                    f"annotation-broker: artifact-manager.{name} failed at send time "
                    f"({exc}); re-resolving the service and retrying once"
                )
                self._svc = None
                svc = await self._service()
                return await getattr(svc, name)(*args, **kwargs)

        return call


def _read_pip(name: str) -> List[str]:
    text = (Path(__file__).parent / name).read_text()
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@bioengine.app(
    num_cpus=1,
    memory_mb=512,
    pip=_read_pip("requirements-broker.txt"),
    max_ongoing_requests=20,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 1,
    },
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    graceful_shutdown_timeout_s=60.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class AnnotationBroker:
    """Role-gated authority for ``bioimage-io/colab-annotations`` datasets."""

    def __init__(self) -> None:
        self._hypha_token = os.getenv("HYPHA_TOKEN")
        if not self._hypha_token:
            raise RuntimeError("HYPHA_TOKEN environment variable is not set")
        self._hypha = None
        self._am = None
        self._http: Optional[httpx.AsyncClient] = None
        # artifact_id -> (cached_at, name)
        self._manifest_name_cache: Dict[str, tuple] = {}
        # artifact_id -> (cached_at, {stem: entry}); see _image_manifest
        self._image_manifest_cache: Dict[str, tuple] = {}
        self._image_manifest_locks: Dict[str, asyncio.Lock] = {}

    @bioengine.async_init
    async def _connect(self) -> None:
        self._hypha = await connect_to_server(
            {"server_url": SERVER_URL, "token": self._hypha_token}
        )
        self._am = _ReconnectingService(self._resolve_artifact_manager)
        self._http = httpx.AsyncClient(timeout=60.0)
        logger.info(f"annotation-broker connected to Hypha server at {SERVER_URL}")

    async def _resolve_artifact_manager(self):
        """Rebuild the Hypha connection too if the socket itself is gone."""
        try:
            return await self._hypha.get_service("public/artifact-manager")
        except Exception as exc:
            if not _is_send_failure(exc):
                raise
            logger.warning(f"annotation-broker: reconnecting to {SERVER_URL} ({exc})")
            self._hypha = await connect_to_server(
                {"server_url": SERVER_URL, "token": self._hypha_token}
            )
            return await self._hypha.get_service("public/artifact-manager")

    @bioengine.health_check
    async def _health(self) -> None:
        if self._am is None:
            raise RuntimeError("artifact-manager service not connected")
        # A non-None handle proves nothing: during the 2026-08-24 outage the proxy
        # was pinned to a dead client instance and every call failed at send time
        # while this check kept reporting healthy. Exercise the dependency for real
        # so an unreachable artifact-manager restarts the replica instead of
        # serving confidently wrong data. STATE_ROOT is PVC-backed, so a restart
        # costs nothing.
        try:
            await asyncio.wait_for(
                self._am.read(artifact_id=COLLECTION_ID), timeout=_HEALTH_CALL_TIMEOUT_S
            )
        except Exception as exc:
            raise RuntimeError(f"artifact-manager unreachable: {exc}") from exc

    # === context / role helpers ===

    @staticmethod
    def _ctx_user(context: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        user = (context or {}).get("user") or {}
        # Hypha marks anonymous connections with is_anonymous and gives them
        # ids like "http-anonymous"; treat them as identity-less so they can
        # never hold a role or a user folder.
        if user.get("is_anonymous"):
            return {"id": None, "email": None}
        return {"id": user.get("id"), "email": user.get("email")}

    @staticmethod
    def _canonical_id(artifact_id: str) -> str:
        """Frontend URLs carry the bare alias (no 'bioimage-io/' prefix);
        canonicalize to the full artifact id for all broker state and
        artifact-manager calls."""
        return artifact_id if "/" in artifact_id else f"bioimage-io/{artifact_id}"

    def _metadata_or_raise(self, artifact_id: str) -> Dict[str, Any]:
        meta = core.read_metadata(artifact_id, root=STATE_ROOT)
        if meta is None:
            raise ValueError(
                f"Dataset '{artifact_id}' is not registered with the broker."
            )
        return meta

    def _require_role(
        self, context: Optional[Dict[str, Any]], artifact_id: str, minimum: str
    ) -> tuple:
        """Load metadata, resolve the caller's role, and enforce *minimum*.

        Returns ``(metadata, caller, role)``. Raises ``PermissionError`` if
        the caller's role is below *minimum*.
        """
        meta = self._metadata_or_raise(artifact_id)
        caller = self._ctx_user(context)
        role = core.resolve_role(meta, caller["id"], caller["email"])
        if not core.role_at_least(role, minimum):
            who = caller["id"] or caller["email"] or "anonymous"
            raise PermissionError(
                f"User '{who}' has role '{role}' on '{artifact_id}'; "
                f"'{minimum}' or higher is required."
            )
        return meta, caller, role

    def _require_logged_in(
        self, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Optional[str]]:
        """Caller identity for an RPC that hangs off no dataset.

        ``_require_role`` cannot be used for those: it starts by loading a
        dataset's metadata, and there is no dataset. This is the smaller
        guarantee such a method still needs, which is simply that somebody is
        on the other end.
        """
        caller = self._ctx_user(context)
        if not caller["id"]:
            raise PermissionError("This requires a logged-in user.")
        return caller

    # === artifact-manager helpers ===
    # Call-shape notes (see colab_service.py / micro-sam's entry.py):
    #   - get_file needs KEYWORD args (artifact_id=..., file_path=...) — this
    #     bit a previous implementer.
    #   - put_file/list_files/commit take artifact_id positionally.
    #   - read/edit read+write the STAGED version via stage=True.

    async def _ensure_staged(self, artifact_id: str, fn) -> Any:
        async def call():
            return await fn()

        async def restage():
            await self._am.edit(artifact_id=artifact_id, stage=True)

        return await core.ensure_staged(call, restage)

    async def _list_files_safe(
        self, artifact_id: str, dir_path: str, stage: bool = True
    ) -> List[Dict[str, Any]]:
        """Entries under *dir_path*, staged view by default.

        **`stage=False` does not answer "was this already here?", and nothing
        in this app may use it as though it did.** The intuition is that the
        committed view excludes an object the current request has just PUT into
        the open stage, so comparing the two views would separate "the caller
        uploaded this" from "this was already somebody's image". Measured
        against the live artifact manager, it does not: a just-PUT staged object
        was returned by `stage=True`, by `stage` unset, by `stage=False`, by
        `version="v0"` and by `version="latest"` alike. A guard built on the
        difference matches every upload, so it refuses the legitimate ones and
        still cannot see the case it was written for.

        The route that needs that distinction -- the finishing half of
        `upload_local_image`, which may delete the object -- therefore gets it
        from a marker it wrote itself at mint time, not from any listing.
        """
        try:
            files = await self._am.list_files(
                artifact_id, dir_path=dir_path, stage=stage
            )
            return files or []
        except Exception as exc:
            # [] means "no such directory yet". Reporting a transport failure as
            # an empty directory once turned a Hypha restart into a dataset that
            # rendered as zero images with the call still logged OK.
            if _is_send_failure(exc):
                raise
            return []

    async def _list_files_all(
        self, artifact_id: str, dir_path: str, stage: bool = True
    ) -> tuple:
        """``(entries, complete)`` for *dir_path*, following the server's pagination.

        **A TUPLE, not a list, and that is the point.** A review measured the
        previous version returning a PARTIAL list silently when a page after the
        first failed -- 7 of 30 markers, with the past-page unfinished upload
        listed as an image. For the withholding that is the documented
        fail-open; for the finish's by-path extras scan it is the data-loss
        direction, because a missed duplicate marker survives the commit and the
        sweep later deletes that image. Returning a bare list let every caller
        treat "all of them" and "as many as I could get" as the same value. A
        caller that ignores the second element now gets a tuple where it
        expected a list, which fails loudly.

        `list_files` is paginated and its default page is 1000 entries, so a
        single call is a PAGE and not a directory. A review measured what that
        costs where a correctness decision reads the result: with more than 1000
        markers in ``.uploads/``, everything past the first page was invisible,
        which re-opened both round-6 consequences one threshold up -- an
        unfinished upload past the page was not withheld (the size bypass,
        reachable by minting 1001 names or by one large folder upload), and a
        duplicate marker past the page survived a successful finish and had the
        sweep delete the image.

        That is the same defect as the app's own 50-entry cap, with the cap
        imposed by the server instead, and it deserves the same sentence: a cap
        on a read that a correctness decision depends on is not a bound, it is a
        silent wrong answer.

        **The loop stops on an EMPTY page, not on a short one.** A page shorter
        than the limit asked for is an ordinary answer -- the server may cap the
        limit below the ask -- so treating "shorter than requested" as "that is
        all of them" reintroduces the incomplete read. Measured against a paging
        double whose page is 50: an ask of 10000 came back with 50, the loop
        ended, and the read was short again. Entries are also de-duplicated by
        name, so a server that ignores `offset` terminates the loop with a
        warning instead of spinning -- and the warning says the result may be
        incomplete, because the one thing this must not do is claim a
        completeness it cannot have.

        Deliberately NOT bounded by an arbitrary maximum. A cap here would be
        the very defect being fixed; the directory is finite, and the sweep at
        the start of every mint keeps it near the number of uploads in flight.
        The cost of a pathological case -- a caller that mints hundreds of
        thousands of names -- is read volume on an authenticated, manager-only
        route, and it is logged from the second page onwards so it is visible
        rather than inferred.
        """
        out: List[Dict[str, Any]] = []
        seen: set = set()
        stalled = 0
        offset = 0
        page = 0
        while True:
            try:
                files = await self._am.list_files(
                    artifact_id,
                    dir_path=dir_path,
                    stage=stage,
                    limit=_LIST_PAGE,
                    offset=offset,
                )
            except TypeError:
                # An artifact manager (or a test double) that does not accept
                # limit/offset. Fall back to the single call rather than
                # failing the route -- and say so, because this is the case
                # where the answer may be a page rather than a directory.
                logger.warning(
                    f"list_files does not accept limit/offset; {dir_path} was "
                    "read as ONE page and may be incomplete"
                )
                # Complete ONLY if the single page is shorter than the server's
                # documented default -- a full-looking page may be a page. The
                # previous version returned True unconditionally while the log
                # above said "may be incomplete", which is the contradiction a
                # review caught: the log and the return value disagreed, and the
                # return value is the one callers act on.
                one = await self._list_files_safe(artifact_id, dir_path, stage)
                return one, len(one) < _SERVER_DEFAULT_PAGE
            except Exception as exc:
                if _is_send_failure(exc):
                    raise
                # INCOMPLETE, and said so. Page 0 failing means the directory
                # could not be read at all; a later page failing means a partial
                # answer. Both are reported as incomplete, and both are logged
                # with the offset, because "I read everything" and "I read what
                # I could" must never be the same return value.
                logger.warning(
                    f"listing {dir_path} failed at offset {offset} (page "
                    f"{page + 1}) after {len(out)} entries: "
                    f"{type(exc).__name__}: {str(exc)[:160]}. The answer is "
                    "INCOMPLETE."
                )
                return out, False
            batch = files or []
            # STOP ONLY ON AN EMPTY PAGE, never on a page shorter than the limit
            # that was asked for. The server is free to cap the limit below the
            # ask -- Hypha documents a range, and a page smaller than requested
            # is a perfectly ordinary answer -- so "shorter than I asked for"
            # does not mean "that is all of them". Measured: with a paging
            # double whose page is 50 and an ask of 10000, that test ended the
            # loop after one page and the read was incomplete again, which is
            # the very defect this method exists to fix.
            if not batch:
                return out, True
            before = len(seen)
            for entry in batch:
                name = entry.get("name") if isinstance(entry, dict) else str(entry)
                if isinstance(name, str) and name not in seen:
                    seen.add(name)
                    out.append(entry)
            # TWO DIFFERENT THINGS LOOK ALIKE HERE, and neither the naive
            # version nor my first fix told them apart.
            #
            #   * the server IGNORES `offset` and hands back the same page --
            #     which cannot be read past, so the answer is INCOMPLETE;
            #   * an ordinary OVERLAP. Hypha applies offset to the RAW S3
            #     listing and filters staged-removed entries AFTERWARDS, so
            #     advancing by the SERVED count makes the next window start
            #     early and re-serve names already seen.
            #
            # Comparing the page's CONTENT to the previous page's does not
            # separate them: measured, with a page size of 2 and one flagged
            # marker, a genuine overlap served a page IDENTICAL to the previous
            # one, so a content test called the offset ignored and the finish
            # refused a valid upload. The distinguishing fact is not what a page
            # contains but whether progress ever RESUMES: under a filtered
            # overlap, pages that add nothing new are bounded by the number of
            # filtered entries, while an ignored offset never progresses again.
            #
            # So: tolerate up to _NO_PROGRESS_PAGES consecutive pages that add
            # nothing, then declare the read incomplete. That is a bound rather
            # than a proof, and it is deliberately generous; the alternative
            # designs are recorded on the row.
            #
            # `before` is taken ABOVE, before the page is merged, and there is
            # exactly ONE merge loop. There used to be a second copy of both
            # here: the page was added, `before` was then taken from the
            # already-grown set, and the second loop added nothing -- so
            # `len(seen) == before` was true for EVERY non-empty page and
            # `stalled` climbed on all of them. A correctly paging server was
            # therefore declared INCOMPLETE after _NO_PROGRESS_PAGES pages.
            # Measured by a review, through the shipped method against a double
            # that serves every page in full: at a 1000-entry page, 7000
            # entries came back complete and 8000 came back as 8000 entries
            # with complete=False; at 10000, the cliff was at 80000. Harmless
            # before the image reads depended on the flag, and not harmless
            # after: every mint, import and finish into a dataset that size
            # refused permanently and the pruning sync always skipped.
            if len(seen) == before:
                stalled += 1
                if stalled >= _NO_PROGRESS_PAGES:
                    logger.warning(
                        f"listing {dir_path} made no progress for {stalled} "
                        f"consecutive pages at offset {offset}; treating the "
                        "result as INCOMPLETE"
                    )
                    return out, False
                logger.info(
                    f"listing {dir_path} overlapped at offset {offset} "
                    f"({len(batch)} entries, none new); reading on"
                )
            else:
                stalled = 0
            offset += len(batch)
            page += 1
            logger.info(
                f"{dir_path} has more than {offset} entries; reading page "
                f"{page + 1}"
            )

    async def _file_exists(self, artifact_id: str, file_path: str) -> bool:
        # Presigning does not check object existence, so a get_file probe can
        # succeed for a key that was never written. List the parent dir instead.
        dir_path, _, name = file_path.rpartition("/")
        entries = await self._list_files_safe(artifact_id, dir_path)
        return any(
            (e.get("name") if isinstance(e, dict) else str(e)) == name for e in entries
        )

    async def _read_json_file(
        self, artifact_id: str, file_path: str, default: Any
    ) -> Any:
        try:
            url = await self._am.get_file(
                artifact_id=artifact_id, file_path=file_path, stage=True
            )
        except Exception as exc:
            if _is_send_failure(exc):
                raise
            return default
        resp = await self._http.get(url)
        if resp.status_code == 404:
            return default
        resp.raise_for_status()
        return resp.json()

    async def _write_json_file(
        self, artifact_id: str, file_path: str, data: Any
    ) -> None:
        url = await self._am.put_file(artifact_id, file_path=file_path)
        resp = await self._http.put(url, content=json.dumps(data).encode("utf-8"))
        resp.raise_for_status()

    async def _put_bytes(
        self, artifact_id: str, file_path: str, content: bytes
    ) -> None:
        url = await self._am.put_file(artifact_id, file_path=file_path)
        resp = await self._http.put(url, content=content)
        resp.raise_for_status()

    # === image manifest (0.10.0) ===
    # The authoritative image list. See the module comment in broker_core for
    # the read-is-a-union / write-prunes asymmetry.

    def _image_manifest_lock(self, artifact_id: str) -> asyncio.Lock:
        lock = self._image_manifest_locks.get(artifact_id)
        if lock is None:
            lock = asyncio.Lock()
            self._image_manifest_locks[artifact_id] = lock
        return lock

    async def _image_manifest(
        self,
        artifact_id: str,
        dims: Optional[Dict[str, Any]] = None,
        require_stem: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Every image in the dataset, keyed by stem.

        Reads ``images/manifest.json`` and unions it with the ``images/``
        listing, so a dataset whose manifest was never written (every dataset
        before 0.10.0) still resolves through this one path.

        Cached for ``MANIFEST_CACHE_TTL_S``, because ``get_image_url`` is the
        annotate page's featherweight first call. *require_stem* forces a
        refresh when the cache does not contain it, so a just-uploaded image is
        never reported missing -- the cache can be stale, but only in the
        harmless direction.

        **THE LISTING IS PAGED, AND THAT COSTS A SECOND ROUND TRIP ON A CACHE
        MISS.** This comment used to say the call "must not grow two
        round-trips", and the reason it now does is that one call was not
        reading the directory. `list_files`'s `limit` defaults to 1000 --
        measured from the live artifact manager's own published schema,
        ``{"default": 1000, "minimum": 1, "maximum": 10000}`` -- so a single
        call answers a PAGE, and every image past the first page was simply
        missing from the dataset's image list, with no error and nothing saying
        so. `_list_files_all` pages until an EMPTY page, which for any
        non-empty directory under 10000 entries is two calls rather than one.
        That is the whole cost, it is paid only on a cache MISS, and the
        alternative to it is an image list that is silently wrong above 1000
        images. A test pins the call count so a change that doubles it again is
        visible rather than inferred.

        ``complete`` is discarded here, DELIBERATELY and named so. This is a
        read: a short answer under-reports for one cache period and the next
        read recomputes it. The callers that must NOT be handed a partial
        listing are the ones that WRITE, and each of those checks it.
        """
        cached = self._image_manifest_cache.get(artifact_id)
        if (
            cached is not None
            and core.is_cache_fresh(cached[0], ttl_s=MANIFEST_CACHE_TTL_S)
            and (require_stem is None or require_stem in cached[1])
        ):
            return cached[1]

        doc = await self._read_json_file(artifact_id, core.IMAGE_MANIFEST_PATH, None)
        listing, _listing_complete = await self._list_files_all(artifact_id, "images")
        # Uploads still in flight are withheld from the LISTING side. An object
        # that has only been PUT would otherwise be a dataset image whether or
        # not the finishing call was ever made, which is how a caller that
        # simply never finishes gets past the size ceiling entirely -- measured
        # in review: 10,000 bytes against a 64-byte ceiling, no finish, listed
        # as a `local` image. The withholding is what makes the finish a gate.
        #
        # The stored manifest is NOT filtered: a successful finish removes its
        # marker before committing, so a completed image is never withheld, and
        # filtering a stored entry could hide an image that is genuinely there.
        # Fail-open by design: a short set here costs a transient extra
        # name in the list, and the next read recomputes it. `complete` is
        # discarded DELIBERATELY and named so, rather than silently dropped.
        withheld, _complete = await self._withheld_upload_paths(artifact_id)
        entries, _ = core.merge_image_manifest(
            core.parse_image_manifest(doc),
            core.image_names_from_listing(listing),
            withheld_paths=withheld,
            dims=dims,
        )
        self._image_manifest_cache[artifact_id] = (time.time(), entries)
        return entries

    async def _image_entry(self, artifact_id: str, stem: str) -> Dict[str, Any]:
        entries = await self._image_manifest(artifact_id, require_stem=stem)
        entry = entries.get(stem)
        if entry is None:
            raise FileNotFoundError(
                f"Image '{stem}' is not in dataset '{artifact_id}'."
            )
        return entry

    async def _image_file_path(self, artifact_id: str, stem: str) -> str:
        """Path under the artifact for a *materialised* image.

        Raises for a ``remote`` entry rather than inventing a path: until the
        image is materialised there is no object to presign, and every caller
        of this (image url, embedding source, training input) needs a real
        file.
        """
        entry = await self._image_entry(artifact_id, stem)
        path = entry.get("path")
        if not path:
            raise FileNotFoundError(
                f"Image '{stem}' is not materialised in '{artifact_id}' "
                f"(state={entry.get('state')!r}); no file to read."
            )
        return path

    async def _read_artifact_manifest(self, artifact_id: str) -> Dict[str, Any]:
        artifact = await self._am.read(artifact_id=artifact_id, stage=True)
        if isinstance(artifact, dict):
            return dict(artifact.get("manifest") or {})
        return dict(getattr(artifact, "manifest", {}) or {})

    async def _cached_manifest_info(self, artifact_id: str) -> Dict[str, str]:
        cached = self._manifest_name_cache.get(artifact_id)
        now = time.time()
        if cached is not None and core.is_cache_fresh(
            cached[0], ttl_s=MANIFEST_CACHE_TTL_S, now=now
        ):
            return cached[1]
        try:
            manifest = await self._read_artifact_manifest(artifact_id)
            info = {
                "name": manifest.get("name") or artifact_id,
                "description": str(manifest.get("description") or ""),
            }
        except Exception as exc:
            logger.warning(
                f"annotation-broker: failed to read manifest for '{artifact_id}': {exc}"
            )
            info = {"name": artifact_id, "description": ""}
        self._manifest_name_cache[artifact_id] = (now, info)
        return info

    async def _cached_manifest_name(self, artifact_id: str) -> str:
        return (await self._cached_manifest_info(artifact_id))["name"]

    _MISSING_FILE_RE = re.compile(r"File '([^']+)' does not exist")

    async def _commit_tolerating_pending_uploads(self, artifact_id: str) -> None:
        """Commit, tolerating staged files that were minted via put_file but
        not (yet) uploaded to S3.

        put_file registers the file in the staged manifest immediately, and
        Hypha's commit validates every staged file against S3 — so a single
        in-flight or abandoned annotation/embedding upload would make every
        set_role/set_public fail. Strategy: retry with short waits so
        in-flight uploads can land, then prune entries that never arrived
        (an annotator whose upload lands after the prune sees the pair
        missing on the next index refresh and can simply re-save).
        """
        delays = [5.0, 10.0]
        prunes = 0
        while True:
            try:
                await self._am.commit(artifact_id)
                return
            except Exception as exc:
                match = self._MISSING_FILE_RE.search(str(exc))
                if not match:
                    raise
                if delays:
                    await asyncio.sleep(delays.pop(0))
                    continue
                if prunes >= 20:
                    raise
                missing = match.group(1)
                logger.warning(
                    f"annotation-broker: pruning never-uploaded staged file "
                    f"'{missing}' from '{artifact_id}' to unblock commit"
                )
                await self._am.remove_file(artifact_id=artifact_id, file_path=missing)
                prunes += 1

    async def _apply_permissions(self, artifact_id: str, meta: Dict[str, Any]) -> None:
        """Mirror broker-metadata roles into the Hypha ACL.

        Cycle (owned here, per the architecture plan): read current
        permissions -> edit(config=...) writes the new ACL -> commit ->
        edit(stage=True) re-stages, all wrapped in the stage-mode retry.
        """
        perms = core.build_acl_permissions(meta)

        async def _do_edit():
            current = await self._am.read(artifact_id=artifact_id, stage=True)
            current_config = (
                dict(current.get("config") or {})
                if isinstance(current, dict)
                else dict(getattr(current, "config", {}) or {})
            )
            current_config["permissions"] = perms
            await self._am.edit(
                artifact_id=artifact_id, config=current_config, stage=True
            )
            await self._commit_tolerating_pending_uploads(artifact_id)
            await self._am.edit(artifact_id=artifact_id, stage=True)

        await self._ensure_staged(artifact_id, _do_edit)

    # === public service API ===

    @bioengine.method(context=True)
    async def register_dataset(
        self,
        artifact_id: str = Field(
            ..., description="Full artifact id, e.g. 'bioimage-io/annotation-abc123'."
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Register a freshly-created dataset artifact with the broker.

        Only the artifact's own owner (``manifest.owner`` or ``created_by``)
        may register it. Idempotent: calling it again on an already
        registered dataset just returns the existing record.
        """
        artifact_id = self._canonical_id(artifact_id)
        caller = self._ctx_user(context)
        existing = core.read_metadata(artifact_id, root=STATE_ROOT)
        if existing is not None:
            return existing

        artifact = await self._am.read(artifact_id=artifact_id, stage=True)
        if isinstance(artifact, dict):
            manifest = dict(artifact.get("manifest") or {})
            created_by = artifact.get("created_by")
        else:
            manifest = dict(getattr(artifact, "manifest", {}) or {})
            created_by = getattr(artifact, "created_by", None)
        # colab_service.py historically wrote created_by into the manifest,
        # not the artifact top level.
        created_by = created_by or manifest.get("created_by")

        if not core.caller_matches_artifact_owner(
            manifest, created_by, caller["id"], caller["email"]
        ):
            who = caller["id"] or caller["email"] or "anonymous"
            raise PermissionError(
                f"User '{who}' does not own artifact '{artifact_id}'; cannot register it."
            )

        owner = manifest.get("owner") or {"id": caller["id"], "email": caller["email"]}
        meta = core.new_metadata(artifact_id, owner=owner)
        return core.write_metadata(meta, root=STATE_ROOT)

    @bioengine.method(context=True)
    async def list_my_datasets(self, context=None) -> Dict[str, Any]:
        """Datasets where the caller is a manager or annotator (NOT owner —
        the frontend gets the caller's own datasets from the collection)."""
        caller = self._ctx_user(context)
        shared: List[Dict[str, Any]] = []
        for artifact_id in core.list_dataset_ids(root=STATE_ROOT):
            meta = core.read_metadata(artifact_id, root=STATE_ROOT)
            if meta is None:
                continue
            role = core.resolve_role(meta, caller["id"], caller["email"])
            if role not in ("manager", "annotator"):
                continue
            info = await self._cached_manifest_info(artifact_id)
            shared.append(
                {
                    "artifact_id": artifact_id,
                    "name": info["name"],
                    "description": info["description"],
                    "role": role,
                    "labels": meta.get("labels", []),
                }
            )
        return {"shared": shared}

    @bioengine.method(context=True)
    async def get_dataset(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        context=None,
    ) -> Dict[str, Any]:
        """Broker metadata + the caller's resolved role, for gating the
        dataset-overview route. Annotators get labels and their role but not
        the member lists (those are for the manager-side sharing panel)."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, role = self._require_role(context, artifact_id, "annotator")
        result = dict(meta)
        result["role"] = role
        if not core.role_at_least(role, "manager"):
            result.pop("managers", None)
            result.pop("annotators", None)
            result.pop("access_requests", None)
        return result

    # ------------------------------------------------------------------
    # Per-label train/test splits (add-only snapshots, see broker_core)
    # ------------------------------------------------------------------

    async def _image_stems(self, artifact_id: str) -> set:
        """Stems of every image in the dataset, materialised or not."""
        return set(await self._image_manifest(artifact_id))

    async def _annotation_counts(self, artifact_id: str, label: str) -> Dict[str, int]:
        """Complete (png+geojson) annotation saves per stem, summed across
        every user folder of *label*."""
        label_dir = core.label_folder(label)
        entries = await self._list_files_safe(artifact_id, label_dir)
        user_dirs = [
            e.get("name")
            for e in entries
            if isinstance(e, dict)
            and (e.get("type") == "directory" or e.get("is_dir"))
            and str(e.get("name", "")).startswith("user-")
        ]
        totals: Dict[str, int] = {}
        for user_dir in user_dirs:
            files = await self._list_files_safe(artifact_id, f"{label_dir}/{user_dir}")
            filenames = [
                f.get("name", "") if isinstance(f, dict) else str(f) for f in files
            ]
            for stem, n in core.count_pairs_by_stem(filenames).items():
                totals[stem] = totals.get(stem, 0) + n
        return totals

    def _require_label(self, meta: Dict[str, Any], label: str) -> None:
        if not any(l.get("name") == label for l in meta.get("labels", []) or []):
            raise ValueError(f"Unknown label '{label}' on this dataset.")

    async def _read_split_or_raise(
        self, artifact_id: str, label: str, name: str
    ) -> Dict[str, Any]:
        doc = await self._read_json_file(
            artifact_id, core.split_path(label, name), default=None
        )
        if not isinstance(doc, dict):
            raise ValueError(f"No split named '{name}' for label '{label}'.")
        return doc

    async def _validate_split_stems_are_images(
        self, artifact_id: str, stems: List[str]
    ) -> None:
        """Every stem must be an image of this dataset *and* materialised.

        The second half is a guard against an ordering violation, not a
        routine rejection: annotating is what materialises an image and only
        annotated images can enter a split, so a remote stem here means
        something upstream went wrong (design §9). Catching it at split time
        keeps the failure next to the decision instead of surfacing it much
        later as an unusable training URL.
        """
        entries = await self._image_manifest(artifact_id)
        unknown = [s for s in stems if s not in entries]
        if unknown:
            raise ValueError(
                "Not images of this dataset: " + ", ".join(sorted(unknown))
            )
        unmaterialised = core.non_local_stems(entries, stems)
        if unmaterialised:
            raise ValueError(
                "Not materialised, so they cannot be trained on: "
                + ", ".join(unmaterialised)
                + ". Open each one in the annotator to materialise it first."
            )

    @bioengine.method(context=True)
    async def create_split(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: str = Field(..., description="Label the split belongs to."),
        name: str = Field(..., description="Split name (^[a-z0-9._-]+$)."),
        train: List[str] = Field(
            ..., description="Annotated image stems for training."
        ),
        test: Optional[List[str]] = Field(
            None, description="Annotated image stems for testing."
        ),
        ratio: Optional[float] = Field(
            None,
            description="Target train fraction (0..1] used for later auto-distribution; defaults to the actual fraction at creation.",
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Create a per-label split: an add-only snapshot of annotation
        progress. Only images with at least one complete annotation save for
        *label* may enter. Fails if the split name already exists."""
        artifact_id = self._canonical_id(artifact_id)
        meta, caller, _role = self._require_role(context, artifact_id, "manager")
        self._require_label(meta, label)
        train = train if isinstance(train, list) else []
        test = test if isinstance(test, list) else []
        ratio = ratio if isinstance(ratio, (int, float)) else None
        if not core.is_valid_split_name(name if isinstance(name, str) else ""):
            raise ValueError(f"Invalid split name {name!r}: must match ^[a-z0-9._-]+$")
        existing = await self._read_json_file(
            artifact_id, core.split_path(label, name), default=None
        )
        if isinstance(existing, dict):
            raise ValueError(
                f"Split '{name}' already exists for label '{label}'. "
                "Extend it with update_split or pick another name."
            )
        stems = [str(s) for s in train] + [str(s) for s in test]
        await self._validate_split_stems_are_images(artifact_id, stems)
        counts = await self._annotation_counts(artifact_id, label)
        doc = core.new_split(
            label,
            name,
            train,
            test,
            counts,
            caller.get("id") or caller.get("email"),
            ratio,
        )

        async def _write():
            await self._write_json_file(artifact_id, core.split_path(label, name), doc)

        await self._ensure_staged(artifact_id, _write)
        return doc

    @bioengine.method(context=True)
    async def update_split(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: str = Field(..., description="Label the split belongs to."),
        name: str = Field(..., description="Split name."),
        add_train: Optional[List[str]] = Field(
            None, description="Annotated stems to add to train."
        ),
        add_test: Optional[List[str]] = Field(
            None, description="Annotated stems to add to test."
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Add-only extension of an existing split. Added stems must be
        annotated and new to both sets; an image that was ever in train can
        never enter test. Refreshes the annotation-count snapshot for every
        member."""
        artifact_id = self._canonical_id(artifact_id)
        meta, caller, _role = self._require_role(context, artifact_id, "manager")
        self._require_label(meta, label)
        add_train = add_train if isinstance(add_train, list) else []
        add_test = add_test if isinstance(add_test, list) else []
        doc = await self._read_split_or_raise(artifact_id, label, name)
        added = [str(s) for s in add_train] + [str(s) for s in add_test]
        await self._validate_split_stems_are_images(artifact_id, added)
        counts = await self._annotation_counts(artifact_id, label)
        doc = core.extend_split(
            doc,
            add_train,
            add_test,
            counts,
            caller.get("id") or caller.get("email"),
        )

        async def _write():
            await self._write_json_file(artifact_id, core.split_path(label, name), doc)

        await self._ensure_staged(artifact_id, _write)
        return doc

    @bioengine.method(context=True)
    async def set_split_checkpoint(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: str = Field(..., description="Label the split belongs to."),
        name: str = Field(..., description="Split name."),
        checkpoint: Dict[str, Any] = Field(
            ...,
            description="Training lineage record, e.g. {'session_id': ..., 'model_type': ...}.",
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Record the fine-tuning checkpoint a split's training produced.
        A non-null checkpoint locks the split's lineage: the frontend shows
        continued fine-tuning from this checkpoint instead of a base-model
        selector, and the split can no longer be deleted."""
        artifact_id = self._canonical_id(artifact_id)
        _meta, caller, _role = self._require_role(context, artifact_id, "manager")
        if not isinstance(checkpoint, dict) or not checkpoint:
            raise ValueError("checkpoint must be a non-empty object.")
        doc = await self._read_split_or_raise(artifact_id, label, name)
        doc["checkpoint"] = dict(checkpoint)
        doc["checkpoint"]["recorded_at"] = core.now_iso()
        doc["checkpoint"]["recorded_by"] = caller.get("id") or caller.get("email")
        doc["updated_at"] = core.now_iso()
        doc["updated_by"] = caller.get("id") or caller.get("email")

        async def _write():
            await self._write_json_file(artifact_id, core.split_path(label, name), doc)

        await self._ensure_staged(artifact_id, _write)
        return doc

    @bioengine.method(context=True)
    async def get_split(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: str = Field(..., description="Label the split belongs to."),
        name: str = Field(..., description="Split name."),
        context=None,
    ) -> Dict[str, Any]:
        """Read one split document (full: members, counts, history,
        checkpoint)."""
        artifact_id = self._canonical_id(artifact_id)
        self._require_role(context, artifact_id, "annotator")
        return await self._read_split_or_raise(artifact_id, label, name)

    @bioengine.method(context=True)
    async def list_splits(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: Optional[str] = Field(
            None, description="Label to list splits for; omit for ALL labels."
        ),
        context=None,
    ) -> List[Dict[str, Any]]:
        """Compact summaries of the splits of one label, or of every label
        when *label* is omitted (the image-delete guard wants all of them)."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, _role = self._require_role(context, artifact_id, "annotator")
        label = label if isinstance(label, str) else None
        labels = (
            [label]
            if label
            else [l.get("name") for l in meta.get("labels", []) or [] if l.get("name")]
        )
        summaries: List[Dict[str, Any]] = []
        for lbl in labels:
            entries = await self._list_files_safe(
                artifact_id, core.splits_dir_path(lbl)
            )
            for e in entries:
                fname = e.get("name", "") if isinstance(e, dict) else str(e)
                if not fname.endswith(".json"):
                    continue
                doc = await self._read_json_file(
                    artifact_id, f"{core.splits_dir_path(lbl)}/{fname}", default=None
                )
                if isinstance(doc, dict):
                    summaries.append(core.split_summary(doc))
        return summaries

    @bioengine.method(context=True)
    async def delete_split(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: str = Field(..., description="Label the split belongs to."),
        name: str = Field(..., description="Split name."),
        context=None,
    ) -> Dict[str, Any]:
        """Delete a split that never produced a checkpoint. A trained
        split's lineage is permanent and cannot be deleted."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, _role = self._require_role(context, artifact_id, "manager")
        doc = await self._read_split_or_raise(artifact_id, label, name)
        if doc.get("checkpoint"):
            raise ValueError(
                f"Split '{name}' has a trained checkpoint and cannot be deleted."
            )

        async def _rm():
            await self._am.remove_file(
                artifact_id=artifact_id, file_path=core.split_path(label, name)
            )

        await self._ensure_staged(artifact_id, _rm)
        # Persist the deletion into the committed version (same staleness
        # trap as delete_label: a staged-only removal reappears on restage).
        await self._apply_permissions(artifact_id, meta)
        return {"deleted": True, "label": label, "name": name}

    @bioengine.method(context=True)
    async def get_training_urls(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: str = Field(
            ..., description="Label whose annotations feed the training."
        ),
        split_name: str = Field(
            ..., description="Split that defines the train/test membership."
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Training inputs for fine-tuning: for the LATEST pair per (user, stem)
        under the label, a presigned geojson URL plus EITHER a presigned
        ``image_url`` for a materialised image OR an ``image_source`` region
        reference for a remote OME-Zarr one, which the trainer reads crop-wise
        without downloading the image. ONLY stems that are members of the split
        are included, partitioned by its train/test lists."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, _role = self._require_role(context, artifact_id, "manager")
        split = await self._read_split_or_raise(artifact_id, label, split_name)
        train_stems = set(split.get("train") or [])
        test_stems = set(split.get("test") or [])

        label_dir = core.label_folder(label)
        entries = await self._list_files_safe(artifact_id, label_dir)
        user_dirs = [
            e.get("name")
            for e in entries
            if isinstance(e, dict)
            and (e.get("type") == "directory" or e.get("is_dir"))
            and str(e.get("name", "")).startswith("user-")
        ]

        pairs: List[Dict[str, str]] = []
        for user_dir in user_dirs:
            dir_path = f"{label_dir}/{user_dir}"
            files = await self._list_files_safe(artifact_id, dir_path)
            filenames = [
                f.get("name", "") if isinstance(f, dict) else str(f) for f in files
            ]
            for stem, pair in core.latest_pairs_by_stem(filenames).items():
                if stem not in train_stems and stem not in test_stems:
                    continue
                pairs.append(
                    {
                        "stem": stem,
                        "user": user_dir,
                        "geojson_path": f"{dir_path}/{pair['geojson']}",
                    }
                )

        # A stem with no file in the artifact used to be refused outright, because
        # the only thing that could be handed over was a URL to fetch whole. A
        # remote OME-Zarr stem is now served as a REGION REFERENCE instead: the
        # trainer reads the crop's chunks from the source, so there is nothing to
        # materialise and nothing to download whole.
        #
        # Still refused: a stem with no manifest entry at all (deleted outright),
        # a non-NGFF remote source, and an NGFF source imported before the layout
        # was recorded. `ngff_region_source` returns None for each, which is why
        # the refusal is driven by its answer rather than by a second rule here
        # that could disagree with it.
        manifest = await self._image_manifest(artifact_id)
        stems = [p["stem"] for p in pairs]
        region_sources = {
            s: core.ngff_region_source(manifest.get(s))
            for s in stems
            if s in manifest and not core.image_entry_is_local(manifest.get(s))
        }
        unreadable = sorted(
            {s for s, src in region_sources.items() if src is None}
            | {s for s in stems if s not in manifest}
        )
        if unreadable:
            raise ValueError(
                f"Split '{split_name}' contains {len(unreadable)} image(s) that can "
                f"neither be read as a region nor fetched as a file: "
                f"{', '.join(unreadable)}. A remote OME-Zarr image is trainable "
                f"without materialising only if its store layout was recorded when "
                f"it was imported, which an image imported by an older version of "
                f"this service has not got: re-import those sources, or open each "
                f"one in the annotator once to materialise it."
            )

        # The credential is spliced in HERE, on the way out, exactly as the
        # browser path does it in get_image_url -- and for the same reason: it
        # is never stored in the artifact (design §12), so a reference built
        # from the manifest alone cannot open a credentialed store. Without
        # this the training path fails at its first chunk GET on precisely the
        # stores the import flow went to the trouble of accepting a credential
        # for.
        credentials = meta.get("ngff_credentials") or {}
        if credentials:
            for src in region_sources.values():
                if src is None:
                    continue
                root = src.get("store_root") or ""
                src["store_root"] = core.apply_store_credential(
                    root, credentials.get(root)
                )

        sem = asyncio.Semaphore(16)

        async def _mint(file_path: str) -> str:
            async with sem:
                return await self._am.get_file(
                    artifact_id=artifact_id, file_path=file_path, stage=True
                )

        # Only the materialised stems get a presigned image URL; a region stem has
        # no file to mint one for. The geojson is minted for every pair either way.
        file_pairs = [p for p in pairs if p["stem"] not in region_sources]
        image_urls = await asyncio.gather(
            *[_mint(manifest[p["stem"]]["path"]) for p in file_pairs]
        )
        by_stem_url = {p["stem"]: u for p, u in zip(file_pairs, image_urls)}
        geojson_urls = await asyncio.gather(*[_mint(p["geojson_path"]) for p in pairs])

        train_out: List[Dict[str, Any]] = []
        test_out: List[Dict[str, Any]] = []
        for p, geo_url in zip(pairs, geojson_urls):
            row: Dict[str, Any] = {
                "stem": p["stem"],
                "user": p["user"],
                "geojson_url": geo_url,
            }
            src = region_sources.get(p["stem"])
            if src is not None:
                # The trainer branches on this mapping's shape, so the key is
                # distinct from image_url rather than overloading it: a caller
                # that only knows the old field fails loudly instead of trying
                # to GET a mapping.
                row["image_source"] = src
            else:
                row["image_url"] = by_stem_url[p["stem"]]
            (test_out if p["stem"] in test_stems else train_out).append(row)
        return {
            "train": train_out,
            "test": test_out,
            "split": core.split_summary(split),
        }

    @bioengine.method(context=True)
    async def delete_annotation(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: str = Field(..., description="Label the annotation belongs to."),
        user_folder: str = Field(
            ..., description="Sanitized annotator folder, e.g. 'user-github-12345'."
        ),
        stem: str = Field(..., description="Image stem the annotation belongs to."),
        timestamp: str = Field(
            ..., description="Timestamp of the save pair (YYYYMMDD-HHMMSS)."
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Delete one annotation save (its png + geojson pair). Allowed for
        managers/owners, and for the annotation's own author. The removal is
        persisted through a commit cycle (a staged-only removal would
        resurface on the next re-stage)."""
        artifact_id = self._canonical_id(artifact_id)
        meta, caller, role = self._require_role(context, artifact_id, "annotator")
        if not core.is_valid_timestamp(timestamp):
            raise ValueError(
                f"Invalid timestamp {timestamp!r}; expected YYYYMMDD-HHMMSS."
            )
        is_own = core.sanitize_user_id(caller.get("id")) == user_folder
        if not core.role_at_least(role, "manager") and not is_own:
            raise PermissionError(
                "Only managers or the annotation's author may delete an annotation."
            )
        base = f"{core.label_folder(label)}/{user_folder}/{stem}-{timestamp}"
        removed = []
        for ext in ("png", "geojson"):
            path = f"{base}.{ext}"

            async def _rm(p=path):
                await self._am.remove_file(artifact_id=artifact_id, file_path=p)

            try:
                await self._ensure_staged(artifact_id, _rm)
                removed.append(path)
            except Exception as exc:
                logger.warning(
                    f"annotation-broker: could not remove '{path}' from '{artifact_id}': {exc}"
                )
        if not removed:
            raise ValueError(f"No annotation files found for {base}.*")
        await self._apply_permissions(artifact_id, meta)
        return {"deleted": removed}

    @bioengine.method(context=True)
    async def delete_label(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        name: str = Field(..., description="Label name to delete."),
        context=None,
    ) -> Dict[str, Any]:
        """Delete a label: remove it from the broker metadata AND delete the
        whole ``label_<name>/`` folder (metadata.json, users.json, every
        user's annotation files) via the workspace token. Frontend deletion
        with the user token was unreliable and left the broker's label list
        stale, which made deleted labels reappear."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, _role = self._require_role(context, artifact_id, "manager")
        folder = core.label_folder(name)

        failed: List[str] = []

        async def _rm_dir(dir_path: str) -> None:
            entries = await self._list_files_safe(artifact_id, dir_path)
            for entry in entries:
                n = entry.get("name") if isinstance(entry, dict) else str(entry)
                if not n:
                    continue
                is_dir = isinstance(entry, dict) and (
                    entry.get("type") == "directory" or entry.get("is_dir")
                )
                path = f"{dir_path}/{n}"
                if is_dir:
                    await _rm_dir(path)
                else:
                    try:
                        await self._am.remove_file(
                            artifact_id=artifact_id, file_path=path
                        )
                    except Exception as exc:
                        failed.append(path)
                        logger.warning(
                            f"annotation-broker: failed to remove '{path}' from "
                            f"'{artifact_id}': {exc}"
                        )

        async def _do():
            await _rm_dir(folder)

        await self._ensure_staged(artifact_id, _do)
        meta["labels"] = [
            l for l in meta.get("labels", []) or [] if l.get("name") != name
        ]
        meta = core.write_metadata(meta, root=STATE_ROOT)
        # Persist the deletion into the committed version as well: staged
        # removals alone leave the last committed snapshot holding the label
        # files. Reuses the ACL mirror cycle (commit + re-stage, tolerant of
        # pending uploads); permissions are rebuilt unchanged.
        await self._apply_permissions(artifact_id, meta)
        result = dict(meta)
        result["failed_files"] = failed
        return result

    @bioengine.method(context=True)
    async def update_sharing(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        add: Optional[List[Dict[str, Any]]] = Field(
            None,
            description="Users to add/move: [{'user': {'id'|'email': ...}, 'role': 'annotator'|'manager'}].",
        ),
        remove: Optional[List[Dict[str, Any]]] = Field(
            None, description="Users to remove: [{'id'|'email': ...}]."
        ),
        set_public: Optional[bool] = Field(
            None, description="New public flag, or omit to leave unchanged."
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Apply a batch of sharing changes with a SINGLE ACL commit +
        re-stage cycle (the Share dialog's Apply button). Same rules as
        set_role/remove_user: only the owner may add or remove managers.
        Granted users' pending access requests are cleared. A call with no
        changes returns the metadata without touching the artifact."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, caller_role = self._require_role(context, artifact_id, "manager")
        # Omitted optional params arrive as pydantic FieldInfo (truthy!), not
        # None — the classic bioengine decorator trap. Normalize by type.
        add = add if isinstance(add, list) else []
        remove = remove if isinstance(remove, list) else []
        if not isinstance(set_public, bool):
            set_public = None

        # Validate everything before mutating anything.
        for entry in add:
            role = entry.get("role")
            user = entry.get("user") or {}
            if role not in core.ROLE_LIST_KEYS:
                raise ValueError(
                    f"Invalid role '{role}'; expected 'manager' or 'annotator'."
                )
            if not user.get("id") and not user.get("email"):
                raise ValueError("Each add entry needs a user with an id or email.")
            target_role = core.resolve_role(meta, user.get("id"), user.get("email"))
            if caller_role != "owner" and (
                role == "manager" or target_role == "manager"
            ):
                raise PermissionError(
                    "Only the dataset owner may add or remove managers."
                )
        for user in remove:
            target_role = core.resolve_role(meta, user.get("id"), user.get("email"))
            if caller_role != "owner" and target_role == "manager":
                raise PermissionError("Only the dataset owner may remove a manager.")

        changed = bool(add or remove) or (
            set_public is not None and bool(set_public) != bool(meta.get("public"))
        )
        for entry in add:
            core.set_user_role(meta, entry["user"], entry["role"])
            core.remove_access_request(meta, entry["user"])
        for user in remove:
            core.remove_user_role(meta, user)
        if set_public is not None:
            meta["public"] = bool(set_public)

        if not changed:
            return meta
        meta = core.write_metadata(meta, root=STATE_ROOT)
        await self._apply_permissions(artifact_id, meta)
        return meta

    @bioengine.method(context=True)
    async def request_access(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        role: str = Field(
            "annotator", description="Requested role: 'annotator' or 'manager'."
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Register an access request from a logged-in user without a role
        (the annotate page offers this when the broker denies access). The
        dataset owner sees pending requests in the Share dialog and grants
        them via set_role, which clears the request."""
        artifact_id = self._canonical_id(artifact_id)
        meta = self._metadata_or_raise(artifact_id)
        caller = self._ctx_user(context)
        if not caller["id"] and not caller["email"]:
            raise PermissionError("Log in to request access to this dataset.")
        current = core.resolve_role(meta, caller["id"], caller["email"])
        if core.role_at_least(current, "annotator"):
            return {"status": "already_has_access", "role": current}
        if not isinstance(role, str):
            role = "annotator"
        core.add_access_request(meta, caller, role, core.now_iso())
        core.write_metadata(meta, root=STATE_ROOT)
        return {"status": "requested", "requested_role": role}

    @bioengine.method(context=True)
    async def dismiss_access_request(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        user: Dict[str, Any] = Field(..., description="{'id': ...} or {'email': ...}."),
        context=None,
    ) -> Dict[str, Any]:
        """Reject a pending access request without granting a role."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, _role = self._require_role(context, artifact_id, "manager")
        core.remove_access_request(meta, user)
        meta = core.write_metadata(meta, root=STATE_ROOT)
        return meta

    @bioengine.method(context=True)
    async def set_role(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        user: Dict[str, Any] = Field(..., description="{'id': ...} or {'email': ...}."),
        role: str = Field(..., description="'manager' or 'annotator'."),
        context=None,
    ) -> Dict[str, Any]:
        """Grant *user* the given role. Only the dataset owner may add or
        remove the manager role (either direction)."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, caller_role = self._require_role(context, artifact_id, "manager")
        if role not in core.ROLE_LIST_KEYS:
            raise ValueError(
                f"Invalid role '{role}'; expected 'manager' or 'annotator'."
            )

        target_role = core.resolve_role(meta, user.get("id"), user.get("email"))
        if caller_role != "owner" and (role == "manager" or target_role == "manager"):
            raise PermissionError("Only the dataset owner may add or remove managers.")

        core.set_user_role(meta, user, role)
        core.remove_access_request(meta, user)
        meta = core.write_metadata(meta, root=STATE_ROOT)
        await self._apply_permissions(artifact_id, meta)
        return meta

    @bioengine.method(context=True)
    async def remove_user(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        user: Dict[str, Any] = Field(..., description="{'id': ...} or {'email': ...}."),
        context=None,
    ) -> Dict[str, Any]:
        """Remove *user* from the dataset's managers/annotators. Only the
        owner may remove a manager."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, caller_role = self._require_role(context, artifact_id, "manager")
        target_role = core.resolve_role(meta, user.get("id"), user.get("email"))
        if caller_role != "owner" and target_role == "manager":
            raise PermissionError("Only the dataset owner may remove a manager.")

        core.remove_user_role(meta, user)
        meta = core.write_metadata(meta, root=STATE_ROOT)
        await self._apply_permissions(artifact_id, meta)
        return meta

    @bioengine.method()
    async def list_public_datasets(self) -> List[Dict[str, Any]]:
        """The public gallery: datasets whose owner called ``set_public(True)``.

        Callable by anyone, including an anonymous visitor, because that is
        who the gallery is for. It takes no ``context`` at all -- there is no
        identity-dependent behaviour here, and a method that cannot see who is
        calling cannot accidentally vary by caller.

        **Served from the broker's OWN flag**, the same one ``resolve_role``
        consults when it decides whether an anonymous caller may read a
        dataset. That is deliberate: a gallery built on any other source could
        advertise a dataset the read path then refuses, or hide one it would
        happily serve. It also means the gallery works regardless of the
        parent collection's permissions -- the broker reads the artifacts with
        the app's own credential, and an anonymous visitor never touches the
        artifact manager directly.

        **No email addresses, and NO OWNER AT ALL.** The response is an
        allow-list of fields and every string in it is scrubbed on the way out,
        because a name or a description is free text somebody typed.

        The card names nobody. An earlier version published the owner's account
        id on the grounds that it is not an email address, which was true and
        still wrong: the requirement is no emails AND no user ids, and the id
        measured on the live gallery was a named individual's real account --
        the same one holding the collection's `"*"` grant. No display name
        exists to offer instead, so the choice was "an account id" or "no
        owner". See the block comment above `public_dataset_entry` in
        broker_core for the measurement behind that.

        One artifact read per public dataset, for the name, description and
        image count. That is proportional to the number of PUBLIC datasets
        rather than to all of them, and a dataset whose artifact cannot be
        read is listed with whatever is known rather than dropped -- a
        gallery that silently shrinks is worse than one with a sparse card.
        """
        entries: List[Dict[str, Any]] = []
        for meta in core.iter_public_metadata(STATE_ROOT):
            artifact_id = meta["artifact_id"]
            name = description = None
            image_count = None
            try:
                art = await self._am.read(artifact_id=artifact_id)
                manifest = (art or {}).get("manifest") or {}
                name = manifest.get("name")
                description = manifest.get("description")
            except Exception:
                # Deliberately not logged per dataset: the id comes from the
                # broker's own state, but the read is a network call and a
                # transient failure must not turn the gallery into a log
                # amplifier. The sparse card is the whole response.
                pass
            try:
                manifest_images = await self._image_manifest(artifact_id)
                image_count = len(manifest_images)
            except Exception:
                pass
            entries.append(
                core.public_dataset_entry(
                    meta,
                    name=name,
                    description=description,
                    image_count=image_count,
                )
            )
        return entries

    @bioengine.method(context=True)
    async def set_public(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        is_public: bool = Field(
            ...,
            description="Whether anyone can read (and, if logged in, annotate) this dataset.",
        ),
        context=None,
    ) -> Dict[str, Any]:
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, _role = self._require_role(context, artifact_id, "manager")
        meta["public"] = bool(is_public)
        meta = core.write_metadata(meta, root=STATE_ROOT)
        await self._apply_permissions(artifact_id, meta)
        return meta

    @bioengine.method(context=True)
    async def sync_image_manifest(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        context=None,
    ) -> Dict[str, Any]:
        """Reconcile ``images/manifest.json`` against the ``images/`` listing.

        Call this after adding or removing image files. Files present but not
        in the manifest are added as ``local`` entries; ``local`` entries
        whose file has gone are dropped. ``remote`` entries are never touched
        -- having no file is their normal state.

        Idempotent, and safe to skip: reads union the manifest with the
        listing anyway, so a missed call costs the manifest's extra metadata
        for those stems, never their visibility.
        """
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, _role = self._require_role(context, artifact_id, "manager")

        async with self._image_manifest_lock(artifact_id):
            doc = await self._read_json_file(
                artifact_id, core.IMAGE_MANIFEST_PATH, None
            )
            # THE LISTING IS PAGED AND A SHORT ONE SKIPS THE WRITE, for the
            # same reason as the withheld set below and with a worse
            # consequence. This writer PRUNES: it drops a stored entry whose
            # file it cannot see in the listing. `list_files` answers at most
            # its default page of 1000 entries, so on a dataset above that,
            # every image from page two onwards looked like an entry whose file
            # had gone -- and this call would have deleted their manifest
            # entries and committed. The images themselves survive (reads union
            # the manifest with the listing), but their stored metadata does
            # not, and a commit cannot be taken back.
            listing, listing_complete = await self._list_files_all(
                artifact_id, "images"
            )
            # THE WITHHELD SET IS PASSED HERE TOO, and omitting it was a
            # measured data-loss hole rather than an oversight about style.
            # This writes the stored manifest and COMMITS, so an unfinished
            # upload reconciled in here became a permanent dataset image --
            # size unchecked, signature unchecked -- just because a manager ran
            # this call while somebody's upload was open. The expiry sweep then
            # deleted the object and left the entry behind. It governs the
            # pruning half as well: a withheld path counts as present, so this
            # call cannot drop a stored entry on account of an open upload.
            #
            # AND IT FAILS CLOSED ON A SHORT READ. The withheld set is computed
            # fail-open, which is right for a reader and wrong here: this write
            # is COMMITTED and the stored manifest is never filtered again, so a
            # transient `.uploads/` error would become a permanent record of an
            # unchecked upload as a dataset image. Measured: with the marker
            # listing raising during the sync, the stem was stored, listed, and
            # committed. Skipping is the honest response rather than refusing,
            # because this call is documented as idempotent and safe to skip --
            # a missed sync costs the manifest's extra metadata for those stems,
            # never their visibility, since reads union the manifest with the
            # listing anyway.
            withheld, complete = await self._withheld_upload_paths(artifact_id)
            # ONE SKIP, TWO REASONS, and each names itself. A single `if not
            # (complete and listing_complete)` would report "the markers could
            # not be read" for a short image listing, which sends whoever reads
            # the result to the wrong store.
            skipped = None
            if not complete:
                skipped = "the upload markers could not be read in full"
                detail = (
                    "writing on a short marker set would record an unfinished "
                    "upload as an image permanently"
                )
            elif not listing_complete:
                skipped = "the images directory could not be read in full"
                detail = (
                    "this call PRUNES, and on a short listing an image whose "
                    "file is simply on a later page looks like one whose file "
                    "has gone, so its stored entry would be dropped and "
                    "committed"
                )
            if skipped:
                logger.warning(
                    f"skipping the manifest write for {artifact_id}: {skipped}, "
                    f"and {detail}. Nothing was changed; run this again once "
                    "the store is healthy."
                )
                entries = core.parse_image_manifest(doc)
                return {
                    "changed": False,
                    "images": len(entries),
                    "remote": sum(
                        1
                        for e in entries.values()
                        if e.get("state") == core.IMAGE_STATE_REMOTE
                    ),
                    "skipped": skipped,
                }
            entries, changed = core.merge_image_manifest(
                core.parse_image_manifest(doc),
                core.image_names_from_listing(listing),
                dims=dict(meta.get("image_dims") or {}),
                prune=True,
                withheld_paths=withheld,
            )
            if changed:

                async def _write():
                    await self._write_json_file(
                        artifact_id,
                        core.IMAGE_MANIFEST_PATH,
                        core.build_image_manifest(entries),
                    )

                await self._ensure_staged(artifact_id, _write)
                # A staged-only write reappears as absent after a restage,
                # the same trap delete_label and delete_annotation hit.
                await self._commit_tolerating_pending_uploads(artifact_id)
            self._image_manifest_cache[artifact_id] = (time.time(), entries)

        return {
            "changed": changed,
            "images": len(entries),
            "remote": sum(
                1 for e in entries.values() if e.get("state") == core.IMAGE_STATE_REMOTE
            ),
        }

    # --- Remote NGFF import (0.10.0) --------------------------------------

    async def _fetch_ngff_json(self, root: str, key: str) -> Optional[Any]:
        """One JSON object out of a Zarr store, or ``None`` if it is absent.

        404 is a normal answer here, not an error: probing ``.zattrs`` and
        then ``zarr.json`` is how the store's Zarr version is discovered.
        """
        try:
            resp = await self._http.get(root + key)
        except Exception as exc:
            raise core.NgffSourceError(f"Could not reach {root + key}: {exc}") from exc
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            raise core.NgffSourceError(
                f"{root + key} returned HTTP {resp.status_code}. Check the "
                f"store root and, for a protected source, the token."
            )
        try:
            return resp.json()
        except Exception as exc:
            raise core.NgffSourceError(f"{root + key} is not JSON.") from exc

    async def _read_zarr_node(self, root: str, path: str = "") -> Dict[str, Any]:
        """``{"attrs", "meta_doc"}`` for a group or array, either Zarr version.

        v2 splits attributes and array metadata across ``.zattrs`` and
        ``.zarray``; v3 puts both in one ``zarr.json``. Callers should not
        have to care which, so this normalises the two.
        """
        prefix = f"{path.strip('/')}/" if path.strip("/") else ""
        v3 = await self._fetch_ngff_json(root, prefix + "zarr.json")
        if isinstance(v3, dict):
            return {"attrs": v3.get("attributes") or {}, "meta_doc": v3}
        attrs = await self._fetch_ngff_json(root, prefix + ".zattrs")
        array_meta = await self._fetch_ngff_json(root, prefix + ".zarray")
        if attrs is None and array_meta is None:
            raise core.NgffSourceError(
                f"No Zarr metadata at {root + prefix}. Expected .zattrs / "
                f".zarray (v2) or zarr.json (v3)."
            )
        return {"attrs": attrs or {}, "meta_doc": array_meta}

    async def _probe_ngff_source(
        self, root: str, multiscale_path: Optional[str], origin: str
    ) -> Dict[str, Any]:
        """Read a remote store's metadata and prove the browser can read it.

        This is the whole of §7's preflight. It runs **once**, at import, so
        that a source the annotator's browser cannot actually decode is
        rejected here with a reason, rather than importing cleanly and then
        failing on every single image open.

        Four things are checked, in the order that gives the most useful
        message first: the group is OME-Zarr at all, its dtype maps to a JS
        typed array, its codecs are ones zarrita ships, and one real chunk
        comes back with a CORS header that lets *origin* read it.
        """
        group = await self._read_zarr_node(root)
        parsed = core.parse_ngff_multiscales(group["attrs"], multiscale_path)

        levels = await asyncio.gather(
            *(self._read_zarr_node(root, p) for p in parsed["dataset_paths"])
        )
        metas = []
        for path, node in zip(parsed["dataset_paths"], levels):
            if node["meta_doc"] is None:
                raise core.NgffSourceError(
                    f"Multiscale level {path!r} has no array metadata; the "
                    f"store advertises a level it does not contain."
                )
            metas.append(core.parse_zarr_array_meta(node["meta_doc"]))

        full = metas[0]
        roles = core.ngff_axis_roles(parsed["axes"], len(full["shape"]))
        dims = core.ngff_level_dims(full["shape"], roles)

        if not core.ngff_dtype_is_readable(full["dtype"]):
            raise core.NgffSourceError(
                f"Data type {full['dtype']!r} cannot be displayed in the "
                f"browser. 8/16/32-bit integer and float stores are "
                f"supported; 64-bit integers and string dtypes are not."
            )
        undecodable = core.undecodable_ngff_codecs(full)
        if undecodable:
            raise core.NgffSourceError(
                f"This store uses codec(s) the browser cannot decode: "
                f"{', '.join(undecodable)}. Supported: "
                f"{', '.join(sorted(core.NGFF_DECODABLE_CODECS))}."
            )

        # The coarsest level's first chunk is the cheapest object in the
        # store that exercises the real read path.
        coarsest_index = min(
            range(len(metas)),
            key=lambda i: core.ngff_level_dims(
                metas[i]["shape"],
                core.ngff_axis_roles(parsed["axes"], len(metas[i]["shape"])),
            )["width"],
        )
        chunk_url = (
            root
            + parsed["dataset_paths"][coarsest_index].strip("/")
            + "/"
            + core.first_chunk_key(metas[coarsest_index])
        )
        try:
            chunk = await self._http.get(chunk_url, headers={"Origin": origin})
        except Exception as exc:
            raise core.NgffSourceError(
                f"Could not fetch a chunk from {chunk_url}: {exc}"
            ) from exc
        if chunk.status_code >= 400:
            raise core.NgffSourceError(
                f"Chunk {chunk_url} returned HTTP {chunk.status_code}. The "
                f"metadata is readable but the data is not."
            )
        allow = chunk.headers.get("access-control-allow-origin", "")
        if allow not in ("*", origin):
            raise core.NgffSourceError(
                f"{chunk_url} does not allow cross-origin reads from "
                f"{origin} (Access-Control-Allow-Origin: {allow or 'absent'}). "
                f"The browser reads chunks directly, so the store's CORS "
                f"policy has to permit it."
            )

        return {
            "width": dims["width"],
            "height": dims["height"],
            "dtype": full["dtype"],
            "levels": len(metas),
            "channels": core.ngff_channel_names(group["attrs"]),
            "content_hint": hashlib.sha256(
                json.dumps(group["attrs"], sort_keys=True).encode("utf-8")
            ).hexdigest()[:16],
            # The layout, for a caller that reads pixels itself.
            #
            # All four are already computed above, so returning them costs no extra
            # request: ``dataset_paths`` and ``axes`` come from the one
            # ``parse_ngff_multiscales`` call, ``axis_roles`` from the
            # ``ngff_axis_roles`` call that ``dims`` needed, and ``level0_shape`` is
            # the level-0 array metadata already parsed for the dtype and codec
            # checks.
            #
            # WHY EXPOSE THEM. Slicing a crop out of a remote store needs the level-0
            # dataset path and which axis is x and which is y. Without these a caller
            # has to parse ``multiscales`` for itself, and this parsing is already
            # duplicated once — ``parse_ngff_multiscales`` and ``ngff_axis_roles``
            # both say they mirror ``ngffLayout.ts``. A third copy is how the three
            # drift, and the one that drifts is the one nobody renders with, so it
            # drifts silently. Returning the interpretation keeps one of them
            # authoritative.
            #
            # This is metadata, not pixels. The no-pixel rule this app keeps (see
            # ``materialise_image``: "The broker never touches a pixel, which is what
            # keeps a 200 MB read off the worker") is about reads of image data; it is
            # not weakened by handing back a path and two axis indices.
            "dataset_paths": list(parsed["dataset_paths"]),
            "axes": list(parsed["axes"]),
            "axis_roles": dict(roles),
            "level0_shape": [int(n) for n in full["shape"]],
        }

    @bioengine.method(context=True)
    async def probe_ngff_source(
        self,
        store_root: str = Field(
            ...,
            description=(
                "Root of the OME-Zarr group holding 'multiscales'. Same form "
                "import_ngff_image accepts, credential included."
            ),
        ),
        multiscale_path: Optional[str] = Field(
            None,
            description="Name of the multiscale to use when the group holds several.",
        ),
        origin: str = Field(
            "https://bioimage.io",
            description="Browser origin the CORS preflight is checked against.",
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Read a store's metadata and run §7's preflight, writing nothing.

        Split out of ``import_ngff_image`` in 0.12.0 so the import dialog can
        show the slide, and its dimensions, before the user chooses the patch
        to take out of it. Nothing about a whole-slide source can be presented
        for selection until its extent is known, and the extent is exactly
        what this returns.

        Not attached to a dataset, so there is no artifact to authorise
        against. It is still login-gated, because it makes the broker fetch a
        URL of the caller's choosing and the anonymous role must not have
        that.

        The preflight stays here rather than moving into the browser for the
        reason §7 gives: a CORS rejection is opaque to JS, so only the broker
        can turn one into a message worth reading.
        """
        self._require_logged_in(context)
        root = core.normalise_store_root(store_root)
        untokened, _credential = core.split_store_credential(root)
        multiscale_path = multiscale_path or None
        probe = await self._probe_ngff_source(root, multiscale_path, origin)
        return {
            "store_root": untokened,
            "multiscale_path": multiscale_path,
            "stem": core.default_stem_from_store_root(untokened),
            "max_materialise_bytes": core.MAX_MATERIALISE_BYTES,
            **probe,
        }

    @bioengine.method(context=True)
    async def import_ngff_image(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        store_root: str = Field(
            ...,
            description=(
                "Root of the OME-Zarr group holding 'multiscales'. For a "
                "protected omezarr-view source use the path form, "
                "https://host/t/<token>/zarr/<id>/, never a '?token=' query."
            ),
        ),
        stem: Optional[str] = Field(
            None,
            description="Name for the image; derived from the store root when omitted.",
        ),
        multiscale_path: Optional[str] = Field(
            None,
            description="Name of the multiscale to use when the group holds several.",
        ),
        region: Optional[Dict[str, int]] = Field(
            None,
            description=(
                "{x, y, width, height} in full-resolution pixels, to import a "
                "patch of a whole-slide source rather than the whole of it. "
                "Omit for the whole image."
            ),
        ),
        planes: Optional[Dict[str, int]] = Field(
            None,
            description=(
                "Which slice of each non-spatial axis the annotation was drawn "
                "on, by axis NAME, e.g. {'z': 4}. REQUIRED for a store whose t "
                "or z is larger than one: training refuses such a source with "
                "no plane recorded rather than reading index 0, because index 0 "
                "would pair the polygons with a different slice's pixels and "
                "still look like a successful run. Omit for a 2-D store or one "
                "whose non-spatial axes are all singleton."
            ),
        ),
        origin: str = Field(
            "https://bioimage.io",
            description="Browser origin the CORS preflight is checked against.",
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Add a remote OME-Zarr source to the dataset without copying it.

        No pixels are copied and none are read beyond a single preflight
        chunk, so this is effectively instant regardless of how large the
        source is. The image is immediately browsable and annotatable; it
        becomes trainable only once it is materialised (design §8).

        A path-form credential in *store_root* is **split off and kept in the
        broker's own state**, never written to the artifact: the manifest
        lives in something ``set_public`` can expose (design §12).

        With a *region*, the image is a patch of the source rather than the
        whole of it (design §13 step 6). The manifest entry then describes the
        patch: its width and height are the crop's, and the store root is only
        how those pixels are fetched. That is what makes a 100000x80000 slide
        importable at all, since a whole one could never be materialised and
        so could never be trained on.
        """
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, _role = self._require_role(context, artifact_id, "manager")

        root = core.normalise_store_root(store_root)
        untokened, credential = core.split_store_credential(root)
        multiscale_path = multiscale_path or None

        stem = (stem or "").strip() or core.default_stem_from_store_root(untokened)
        if not stem:
            raise ValueError(
                "Could not derive an image name from the store root; pass one."
            )
        if not core.is_valid_image_stem(stem):
            raise ValueError(
                f"Invalid image name {stem!r}: must match ^[A-Za-z0-9._-]+$."
            )

        probe = await self._probe_ngff_source(root, multiscale_path, origin)
        region = core.normalise_region(region, probe["width"], probe["height"])
        # Checked against THIS store's axes and shape, so a plane naming an axis
        # the source does not have, or an index past its end, is refused at
        # import instead of being written into the manifest and failing later as
        # a training error.
        planes = core.normalise_planes(
            planes, probe["axes"], probe["level0_shape"], probe["axis_roles"]
        )
        # The entry's dimensions are the *patch's*, not the slide's. That one
        # choice is what makes a region cost nothing elsewhere: the 64 MiB
        # materialisation gate, the dataset index dims, the OME-TIFF that gets
        # written and the viewer extent all read width/height and are correct
        # for the crop without knowing a crop happened.
        entry = core.remote_image_entry(
            untokened,
            multiscale_path,
            region["width"] if region else probe["width"],
            region["height"] if region else probe["height"],
            dtype=probe["dtype"],
            channels=probe["channels"],
            levels=probe["levels"],
            content_hint=probe["content_hint"],
            region=region,
            planes=planes,
            layout={
                "dataset_paths": probe["dataset_paths"],
                "axis_roles": probe["axis_roles"],
                "level0_shape": probe["level0_shape"],
                "axes": probe["axes"],
            },
        )

        async with self._image_manifest_lock(artifact_id):
            doc = await self._read_json_file(
                artifact_id, core.IMAGE_MANIFEST_PATH, None
            )
            # THE LISTING IS PAGED, AND A SHORT ONE MAKES THE CHECK BELOW
            # LOOSER -- which is the opposite of the withheld set's direction
            # and is why this needs its own refusal. Fewer listed names means
            # fewer names look taken, so on a dataset above `list_files`'s
            # default page of 1000 a remote import could take a stem whose local
            # FILE sits on a later page. The write is the stored manifest plus
            # this one entry, so that file is not deleted -- it is shadowed: the
            # stem now resolves to a remote source, and the local object at
            # `images/<stem><ext>` is unreachable through the manifest while
            # still occupying the path.
            listing, listing_complete = await self._list_files_all(
                artifact_id, "images"
            )
            if not listing_complete:
                raise RuntimeError(
                    f"The images in '{artifact_id}' could not be listed in "
                    f"full, so whether '{stem}' is already an image in this "
                    "dataset cannot be decided. NOTHING was imported. Try "
                    "again in a moment; if this repeats, the dataset's storage "
                    "is not answering and a manager should look at it."
                )
            # THE CHECK AND THE WRITE USE DIFFERENT SETS, deliberately, and
            # separating them is what makes this writer safe without a
            # fail-closed branch of its own.
            #
            # The CHECK wants the widest view -- the stored manifest UNION the
            # listing, minus withheld names -- because its job is to refuse a
            # stem that is already an image, and a stem present only as a file
            # would otherwise be overwritten. A short withheld set makes that
            # check STRICTER (more names look taken), which is the refusing
            # direction, so it is safe on a partial read.
            #
            # The WRITE takes only the stored manifest plus the remote entry
            # this call came to add. It no longer folds listing-derived local
            # names into the committed manifest at all -- so it cannot record
            # an unfinished upload as an image however the marker read went,
            # which is the hole a review measured here and at the sync. What
            # that gives up is incidental manifest healing: a local file absent
            # from the stored manifest stays absent from it. That costs the
            # extra metadata for those stems and never their visibility, since
            # reads union the manifest with the listing anyway, and healing is
            # exactly what `sync_image_manifest` exists to do.
            withheld, _complete = await self._withheld_upload_paths(artifact_id)
            stored = core.parse_image_manifest(doc)
            visible, _ = core.merge_image_manifest(
                stored,
                core.image_names_from_listing(listing),
                dims=dict(meta.get("image_dims") or {}),
                withheld_paths=withheld,
            )
            if stem in visible:
                raise ValueError(
                    f"'{stem}' is already an image of this dataset. Pass a "
                    f"different name."
                )
            entries = dict(stored)
            entries[stem] = entry

            async def _write():
                await self._write_json_file(
                    artifact_id,
                    core.IMAGE_MANIFEST_PATH,
                    core.build_image_manifest(entries),
                )

            await self._ensure_staged(artifact_id, _write)
            # Staged-only writes reappear as absent after a restage, the same
            # trap sync_image_manifest, delete_label and delete_annotation hit.
            await self._commit_tolerating_pending_uploads(artifact_id)
            # POPPED, not written. `entries` is now the STORED manifest plus
            # this import -- deliberately narrower than what a reader sees,
            # since the write no longer folds in listing-derived names. Seeding
            # the cache with it would drop every local file from the image list
            # until the entry expired. The next read rebuilds the union, which
            # is the only thing that can produce the reader's view.
            self._image_manifest_cache.pop(artifact_id, None)

        if credential:
            credentials = dict(meta.get("ngff_credentials") or {})
            credentials[untokened] = credential
            meta["ngff_credentials"] = credentials
            core.write_metadata(meta, root=STATE_ROOT)

        return {"stem": stem, **entry}

    @bioengine.method(context=True)
    async def forget_remote_image(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        stem: str = Field(..., description="Name of the remote image to drop."),
        context=None,
    ) -> Dict[str, Any]:
        """Drop a ``remote`` manifest entry, undoing an ``import_ngff_image``.

        There is no file to delete: the entry *is* the image. This is the
        only way a remote entry ever leaves the manifest, because
        ``sync_image_manifest`` deliberately never prunes one (a remote entry
        is not expected to have a file, so absence proves nothing).

        Refuses on a ``local`` entry: those are deleted by removing their
        file and re-syncing, and silently unlisting one here would leave an
        orphaned image file behind.
        """
        artifact_id = self._canonical_id(artifact_id)
        _meta, _caller, _role = self._require_role(context, artifact_id, "manager")

        async with self._image_manifest_lock(artifact_id):
            doc = await self._read_json_file(
                artifact_id, core.IMAGE_MANIFEST_PATH, None
            )
            entries = core.parse_image_manifest(doc)
            entry = entries.get(stem)
            if entry is None:
                raise FileNotFoundError(f"'{stem}' is not an image of this dataset.")
            if entry.get("state") != core.IMAGE_STATE_REMOTE:
                raise ValueError(
                    f"'{stem}' is a local image, so it cannot be forgotten. "
                    f"Delete its file instead."
                )
            del entries[stem]

            async def _write():
                await self._write_json_file(
                    artifact_id,
                    core.IMAGE_MANIFEST_PATH,
                    core.build_image_manifest(entries),
                )

            await self._ensure_staged(artifact_id, _write)
            await self._commit_tolerating_pending_uploads(artifact_id)
            self._image_manifest_cache[artifact_id] = (time.time(), entries)

        return {"stem": stem, "forgotten": True, "images": len(entries)}

    # --- Materialisation (0.11.0) -----------------------------------------

    async def _object_head(
        self, artifact_id: str, file_path: str, nbytes: int
    ) -> bytes:
        """First *nbytes* of an artifact file, or ``b""`` if it cannot be read.

        A presigned GET is minted per call rather than cached: the URL is
        short-lived and this runs once per materialisation.
        """
        try:
            url = await self._am.get_file(
                artifact_id=artifact_id, file_path=file_path, stage=True
            )
            resp = await self._http.get(url, headers={"Range": f"bytes=0-{nbytes - 1}"})
            resp.raise_for_status()
            return resp.content[:nbytes]
        except Exception:
            return b""

    @bioengine.method(context=True)
    async def materialise_image(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        stem: str = Field(..., description="Name of the remote image to materialise."),
        done: bool = Field(
            False,
            description=(
                "False to start (returns a presigned PUT URL), true to finish "
                "(verifies the upload and flips the manifest entry)."
            ),
        ),
        levels: Optional[int] = Field(
            None,
            description="Pyramid levels the encoder wrote. Only read when done=true.",
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Turn a ``remote`` image into a ``local`` one, in two calls.

        The frontend does the work in between: it reads level 0 through the
        same Zarr reader that renders the image, encodes an OME-TIFF pyramid
        in the Pyodide kernel, and PUTs it to the URL this returns. The broker
        never touches a pixel, which is what keeps a 200 MB read off the
        worker (design §8).

        Call once with ``done=False``:

            {"already": false, "path", "put_url", "max_bytes"}
            {"already": true,  "path"}   # someone else already did it

        then, after the PUT lands, once with ``done=True`` and the level count
        the encoder produced.

        The PUT must send the body **bare, with no Content-Type header**: the
        presigned signature covers that header, so setting it is a 403.

        Both phases are safe to repeat. The object path is deterministic and
        its bytes are a pure function of the source, so a retry overwrites
        rather than duplicating, and a run that dies between the PUT and the
        ``done=True`` call self-heals: the file shows up in the ``images/``
        listing and the next manifest read promotes the entry, provenance
        intact.

        Requires the annotator role, not manager. Materialisation is triggered
        by opening an image, and every annotator can open one.
        """
        artifact_id = self._canonical_id(artifact_id)
        _meta, _caller, _role = self._require_role(context, artifact_id, "annotator")
        name = core.materialised_image_name(stem)
        path = f"images/{name}"

        if not done:
            # Read past the TTL cache: "has someone else already done this?"
            # is exactly the question a cached answer gets wrong.
            entry = core.parse_image_manifest(
                await self._read_json_file(artifact_id, core.IMAGE_MANIFEST_PATH, None)
            ).get(stem)
            if isinstance(entry, dict) and core.image_entry_is_local(entry):
                return {"already": True, "stem": stem, "path": entry.get("path")}
            core.check_materialisable(stem, entry)

            put_url_holder: Dict[str, str] = {}

            async def _mint():
                put_url_holder["url"] = await self._am.put_file(
                    artifact_id, file_path=path
                )

            await self._ensure_staged(artifact_id, _mint)
            return {
                "already": False,
                "stem": stem,
                "path": path,
                "put_url": put_url_holder["url"],
                "max_bytes": core.MAX_MATERIALISE_BYTES,
            }

        # A presigned PUT reports success from S3, not from the artifact
        # manager, so "the client says it uploaded" is not evidence the object
        # is there. Read the first bytes back before the manifest starts
        # claiming a readable image lives at this path.
        head = await self._object_head(artifact_id, path, 8)
        if not core.looks_like_tiff(head):
            raise ValueError(
                f"No OME-TIFF found at '{path}'. The upload did not land, or "
                f"stored something else. '{stem}' stays remote."
            )

        async with self._image_manifest_lock(artifact_id):
            doc = await self._read_json_file(
                artifact_id, core.IMAGE_MANIFEST_PATH, None
            )
            entries = core.parse_image_manifest(doc)
            entry = entries.get(stem)
            if entry is None:
                raise FileNotFoundError(f"'{stem}' is not an image of this dataset.")
            if core.image_entry_is_local(entry):
                # Lost the race, and the file is verified present either way.
                return {"already": True, "stem": stem, "path": entry.get("path")}
            entries[stem] = core.materialise_image_entry(entry, name, levels)

            async def _write():
                await self._write_json_file(
                    artifact_id,
                    core.IMAGE_MANIFEST_PATH,
                    core.build_image_manifest(entries),
                )

            await self._ensure_staged(artifact_id, _write)
            # Commits the OME-TIFF too: it was PUT into the same open stage.
            await self._commit_tolerating_pending_uploads(artifact_id)
            self._image_manifest_cache[artifact_id] = (time.time(), entries)

        return {"already": False, "stem": stem, **entries[stem]}

    async def _discard_staged_upload(self, artifact_id: str, path: str) -> bool:
        """Remove a staged object this app has just refused. True if it went.

        Never raises: every caller is already refusing a request for a reason
        the caller needs to hear, and a failure to clean up must not replace
        that reason with a different error. It is logged instead, because an
        object that could not be removed is still staged and the next commit
        from any route would publish it.
        """
        try:
            await self._am.remove_file(artifact_id=artifact_id, file_path=path)
            return True
        except Exception as exc:
            logger.warning(
                f"could not remove the refused upload at {path}: "
                f"{type(exc).__name__}: {str(exc)[:160]}. It is still staged, "
                "so the next commit on this artifact would publish it."
            )
            return False

    # === the upload marker ===
    # Provenance for the discard. See the block comment above
    # `UPLOAD_MARKER_DIR` in broker_core for why no listing can supply it.

    async def _write_upload_marker(self, artifact_id: str, marker: Dict[str, Any]):
        """Record a minted upload. Raises, so a failure refuses the mint.

        Deliberately not tolerant: a mint whose marker did not land hands the
        client a URL it can never discharge, and the object it uploads would
        then be unauthorised litter that nothing is allowed to clean up. Better
        to refuse the mint and let the client ask again.
        """
        marker_path = core.upload_marker_path_for_object(marker["path"])
        if marker_path is None:  # unreachable: the route built `path` itself
            raise ValueError("refusing to write a marker for an invalid path")
        # OVERWRITES any marker already recorded for this path, and that is the
        # mechanism rather than a side effect: a re-mint of the same object --
        # a client whose first answer was lost, a second browser tab -- replaces
        # the one record instead of adding a second one that the finish would
        # then have to find by listing. The new mint id is what the finish
        # checks, so the superseded URL stops being dischargeable at the same
        # moment it stops being the live one.
        await self._write_json_file(artifact_id, marker_path, marker)

    async def _read_upload_marker(self, artifact_id: str, object_path: Any) -> Any:
        """The stored marker document for the object at *object_path*, or None.

        Keyed by the OBJECT PATH, not by the mint id: the mint id is what the
        document is checked AGAINST (`marker_authorises_discard`), not how the
        document is found. That is what lets the finish read one file by name
        instead of listing `.uploads/` to discover how many records exist.

        None covers every way the marker can fail to be usable -- an unkeyable
        path, no such marker, a transport hiccup reading it -- because the
        route's response to all of them is identical: refuse, touch nothing.
        """
        marker_path = core.upload_marker_path_for_object(object_path)
        if marker_path is None:
            return None
        return await self._read_json_file(artifact_id, marker_path, None)

    async def _remove_upload_marker(self, artifact_id: str, object_path: Any) -> bool:
        """Consume the marker for *object_path*. Never raises; see the discard.

        The local is `marker_path`, not `path`, deliberately: `path` in this
        module means the IMAGE path, and the two removals must not be confusable
        -- by a reader or by a textual mutation aimed at one of them.
        """
        marker_path = core.upload_marker_path_for_object(object_path)
        if marker_path is None:
            return False
        return await self._remove_marker_file(artifact_id, marker_path)

    async def _remove_marker_file(self, artifact_id: str, marker_path: str) -> bool:
        """Remove ONE marker file by its path. True if it is gone.

        Never raises: every caller decides for itself what a failure means, and
        they differ. The housekeeping sweep logs and carries on, because an
        unswept marker costs disk and the authorisation check treats an expired
        marker as no marker anyway. The finishing call refuses to commit,
        because a marker left behind there has the sweep delete a published
        image. A helper that raised would force one of those two to be wrong.

        The parameter is `marker_path`, not `path`, deliberately: `path` in this
        module means the IMAGE path, and the two removals must not be confusable
        -- by a reader or by a textual mutation aimed at one of them.
        """
        if not isinstance(marker_path, str) or not marker_path.startswith(
            f"{core.UPLOAD_MARKER_DIR}/"
        ):
            # Refuses rather than removing: this is the one place that deletes a
            # path it was handed rather than one it built from a validated mint
            # id, so it checks that the path is inside the marker directory.
            logger.warning(f"refusing to remove a non-marker path: {marker_path!r}")
            return False
        try:
            await self._am.remove_file(artifact_id=artifact_id, file_path=marker_path)
            return True
        except Exception as exc:
            logger.warning(
                f"could not remove the upload marker at {marker_path}: "
                f"{type(exc).__name__}: {str(exc)[:160]}. It will be swept once "
                "it expires."
            )
            return False

    async def _marker_pairs(self, artifact_id: str) -> tuple:
        """``(pairs, complete)`` where pairs is ``(marker_path, document)``.

        **NO CORRECTNESS DECISION READS THIS ANY MORE, and that is the point of
        the path-keyed marker.** Both remaining callers are housekeeping:

        * the WITHHOLDING, which is fail-open by design -- an
          unreadable ``.uploads/`` must not present the dataset as having no
          images -- so a short read leaves "an abandoned upload is listed until
          the next mint sweeps it", and it logs that;
        * the SWEEP, which is best-effort, because less housekeeping this pass
          costs a later pass rather than a wrong answer.

        The FINISH used to be the third caller, and it was the one that could
        not tolerate a short read: a duplicate marker it did not see stayed
        open, withheld the image it had just committed, and then had the sweep
        delete it. Four review rounds tried to bound this read safely and the
        bound was wrong in both directions -- a cap short enough to terminate
        against a server that ignores ``offset`` refused valid uploads once
        staged removals sat in the directory, and one loose enough not to refuse
        could not terminate. The finish no longer reads a listing at all: a
        marker is named from the digest of its object path, so a re-mint
        overwrites the one record and the finish computes its name. See
        :func:`broker_core.upload_marker_path_for_object`.

        `complete` is still returned, and still a TUPLE, for the reason the
        review gave: a caller that ignores completeness then gets a tuple where
        it expected a list and fails loudly, rather than treating "all of them"
        and "as many as I could get" as one value. Both callers above log the
        partial read; neither refuses on it.

        **Uncapped, and it used to be capped at the sweep's limit of 50.** The
        cap now applies only to how much the sweep DELETES per call, which is
        the part a caller can inflate. That cap is sound exactly because the
        sweep is the place where doing less per call is merely slower.

        **The cost, stated rather than hidden**: one listing plus one document
        read per marker present, so a mint is O(markers present). The documents
        are read because the path is in the DOCUMENT; the filename is a digest
        and cannot be inverted. A cheaper withholding is now available and is
        deliberately not taken here -- for each image name, test whether
        ``.uploads/<sha256("images/<name>")>.json`` is in the listing, which
        needs no document reads at all -- because it changes a second mechanism
        in a round whose scope is the finish. It is a follow-up, not a residual:
        the current read is correct, only O(markers) rather than O(1).
        """
        try:
            entries, complete = await self._list_files_all(
                artifact_id, core.UPLOAD_MARKER_DIR
            )
        except Exception:
            return [], False
        pairs = []
        for entry in entries:
            name = entry.get("name") if isinstance(entry, dict) else None
            if not isinstance(name, str) or not name.endswith(".json"):
                continue
            marker_path = f"{core.UPLOAD_MARKER_DIR}/{name}"
            try:
                doc = await self._read_json_file(artifact_id, marker_path, None)
            except Exception:
                doc = None
            pairs.append((marker_path, doc))
        return pairs, complete

    async def _withheld_upload_paths(self, artifact_id: str) -> tuple:
        """``(paths, complete)`` -- object paths not yet treatable as images.

        Every marker that parses withholds its path, expired or not. An object
        that has merely been PUT would otherwise be a dataset image with no size
        check ever applied: the caller that never finishes is not refused by the
        finishing check, because it never makes that call.

        Expiry used to lift the withholding, on the reasoning that the sweep was
        about to delete the object anyway. A review measured the gap that left --
        the sweep was only reachable by minting a DIFFERENT name -- so the
        withholding now lasts until the object is actually gone, and the sweep
        runs at the start of every mint to make sure that happens.

        **`complete` is carried rather than resolved here, and it is a TUPLE**,
        for the reason the paging read is a tuple: a caller that ignores
        completeness gets a tuple where it expected a set and fails loudly. It
        used to return a bare set that was computed fail-open, which meant an
        unreadable ``.uploads/`` read as "nothing is withheld" at every caller
        alike -- and a review measured that the callers are NOT alike:

        * a READ may answer on a short set. The effect is transient, the next
          read recomputes it, and the alternative -- withholding everything on
          an error -- makes an unreadable ``.uploads/`` present the dataset as
          having no images at all, a total outage of the image list triggered by
          housekeeping state, on the annotate page's first call;
        * a WRITE may not. The manifest write is COMMITTED and the stored
          manifest is never filtered afterwards, so a short set there turns a
          transient storage error into a permanent record of an unchecked
          upload as a dataset image. That is the shape ruled on for the finish
          in an earlier round -- a partial read must not feed a decision with a
          lasting effect -- and the writers are on the finish's side of it;
        * the MINT may answer on a short set because short is the REFUSING
          direction there: fewer withheld names means more names look taken, so
          it answers ``already`` rather than minting.

        A short read is logged here once, so every caller's decision is visible
        in one place rather than repeated at four.
        """
        pairs, complete = await self._marker_pairs(artifact_id)
        if not complete:
            logger.warning(
                f"the marker listing for {artifact_id} was INCOMPLETE "
                f"({len(pairs)} marker(s) read); the withheld set may be short. "
                "Readers answer anyway; the manifest WRITERS must not."
            )
        if not pairs:
            return frozenset(), complete
        return core.withheld_upload_paths(pairs), complete

    async def _sweep_expired_markers(self, artifact_id: str) -> int:
        """Housekeeping: drop expired markers AND the objects they vouch for.

        Dropping the marker is tidiness. Dropping the OBJECT is enforcement, and
        it is the half that matters: a caller that mints, PUTs an oversized body
        and never finishes is never refused by the finishing check, so what the
        abandonment leaves behind has to be removed rather than waited on.

        An expired marker is a sound warrant for that delete for exactly the
        reason the finish-time discard is sound -- it says this app minted that
        path for an upload that did not complete, so the object at it was never
        a dataset image. An unparseable marker authorises no object removal at
        all; it names no path, and guessing one from the filename is the single
        way this sweep could delete an image the dataset already had.

        Still best-effort for the MARKER half: the authorisation check treats an
        expired marker as no marker whether or not this ever ran. Every failure
        is logged rather than raised into a caller that was in the middle of
        minting a perfectly good upload.
        """
        pairs, complete = await self._marker_pairs(artifact_id)
        if not complete:
            # Housekeeping only, so a partial read costs a later sweep rather
            # than correctness: the authorisation check treats an expired marker
            # as no marker whether or not it was ever swept, and the withholding
            # holds a path until its object is gone.
            logger.info(
                f"sweeping {artifact_id} from an INCOMPLETE marker listing "
                f"({len(pairs)} read); the rest waits for the next mint"
            )
        if not pairs:
            return 0
        # `now` is THIS app's clock, the one that wrote `expires_at`. Never a
        # listing's `last_modified`: on the production filer the server clock
        # runs ~79 s ahead of the pods, so that comparison crosses two clocks.
        now = time.time()
        abandoned = core.abandoned_upload_removals(pairs, now=now)
        for marker_path, object_path in abandoned:
            try:
                await self._discard_staged_upload(artifact_id, object_path)
            except Exception as exc:
                logger.warning(
                    f"could not discard the abandoned upload {object_path} "
                    f"(marker {marker_path}): {type(exc).__name__}: "
                    f"{str(exc)[:160]}"
                )
        removals = core.expired_marker_removals(pairs, now=now)
        gone = 0
        for marker_path in removals:
            try:
                await self._am.remove_file(
                    artifact_id=artifact_id, file_path=marker_path
                )
                gone += 1
            except Exception as exc:
                logger.warning(
                    f"could not sweep the expired marker {marker_path}: "
                    f"{type(exc).__name__}: {str(exc)[:160]}"
                )
        return gone

    @bioengine.method(context=True)
    async def upload_local_image(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        filename: str = Field(
            ...,
            description=(
                "Final filename component of the image, with extension: "
                "'scan01.png'. Not a path."
            ),
        ),
        done: bool = Field(
            False,
            description=(
                "False to start (returns a presigned PUT URL), true to finish "
                "(verifies the bytes landed and commits them)."
            ),
        ),
        mint_id: str = Field(
            "",
            description=(
                "The 'mint_id' returned by the starting call. Required when "
                "done=true; a finish without it is refused and changes nothing."
            ),
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Presigned PUT for ONE local image, in two calls.

        This is the only route by which an image the user holds locally can
        enter a dataset without mounting a folder. A browser without the File
        System Access API cannot mount one, and the mount is what registers
        the data-provider service every other upload route goes through, so
        without this such a browser could open and annotate somebody else's
        data but could never contribute its own.

        Call once with ``done=False``:

            {"already": false, "stem", "path", "put_url", "max_bytes", "mint_id"}
            {"already": true,  "stem", "path"}   # that stem is already an image

        then, after the PUT lands, once with ``done=True`` and that ``mint_id``.

        **The ``mint_id`` is what makes the finish safe to act on.** Finishing
        may have to DELETE the object -- that is what enforcing a size ceiling
        after the bytes have landed means -- and the one thing it must never
        delete is an image the dataset already had. No listing can tell those
        apart (measured: an object just PUT into the stage is returned by every
        listing view, committed included), so the fact has to be recorded when
        it is created. The starting call writes a marker outside ``images/``;
        this id names it. A finish whose marker is missing, expired, or minted
        for a different path is refused and changes nothing at all.

        As with ``materialise_image`` the PUT must send the body **bare, with
        no Content-Type header**: the presigned signature covers that header,
        so setting it is a 403.

        **The manifest entry is not written here.** The object lands at
        ``images/<stem><ext>`` and the ordinary listing reconciliation turns
        it into a ``local`` entry, which is exactly what happens to an image
        uploaded through a mount. Writing the entry here instead would be a
        second code path that could drift from that one; leaving it to the
        reconciliation is what makes the two indistinguishable afterwards,
        rather than merely similar.

        Requires the manager role: this adds data to somebody's dataset, where
        materialising only converts an image the dataset already has.
        """
        artifact_id = self._canonical_id(artifact_id)
        self._require_role(context, artifact_id, "manager")
        stem, ext = core.split_local_image_filename(filename)
        name = f"{stem}{ext}"
        path = f"images/{name}"

        if not done:
            # Past the TTL cache deliberately: "is this stem taken?" is
            # exactly the question a cached answer gets wrong, and the cost of
            # getting it wrong is overwriting an image somebody else uploaded.
            #
            # The manifest ALONE is not the dataset's image list, and reading
            # only it was a real defect rather than a hypothetical one. The
            # list is the manifest reconciled against the `images/` listing,
            # which is exactly why this route can leave the entry to that
            # reconciliation -- and therefore exactly why its own earlier
            # uploads are NOT in the manifest it would have read. Two ways that
            # bit:
            #
            #   * `images/scan01.png` present but not yet in the manifest read
            #     as free, so the mint handed out a URL for that same path and
            #     the upload replaced the image, orphaning its annotations;
            #   * `scan01.tif` present and `scan01.png` uploaded collapsed into
            #     ONE stem entry, so depending on the extensions either the new
            #     image never appeared or it silently replaced the one the
            #     annotations were drawn on.
            #
            # Reconciling here asks the same question the readers ask, so the
            # answer cannot disagree with them.
            # THE SWEEP RUNS FIRST, before the listing is read. It used to run
            # at the END of a successful mint, and a review measured what that
            # cost: the only way to trigger it was to mint a DIFFERENT name,
            # because minting the same name returns `already: True` from the
            # check below and never reaches the sweep. So "mint, PUT oversized,
            # never finish, wait for expiry" left the object as an image on any
            # dataset where nobody uploaded anything else.
            #
            # Running it here closes that and removes a disagreement at the same
            # time: an abandoned object is deleted BEFORE the no-overwrite check
            # reads the listing, so the check cannot answer `already` about a
            # file that is on its way out, and withholding an expired marker's
            # path until it is swept can no longer contradict that answer.
            #
            # Still best-effort inside: a sweep failure is logged, not raised,
            # so housekeeping cannot refuse a perfectly good upload.
            await self._sweep_expired_markers(artifact_id)

            doc = await self._read_json_file(
                artifact_id, core.IMAGE_MANIFEST_PATH, None
            )
            # AN INCOMPLETE LISTING REFUSES THE MINT, and this is the direction
            # the paged read matters most. `list_files` answers at most its
            # default page of 1000, and a name missing from the answer looks
            # free: on a dataset above 1000 images this check handed out an
            # upload URL for a stem that ALREADY HAD AN IMAGE on a later page,
            # the PUT overwrote that object at `images/<stem><ext>`, and the
            # marker then recorded "not a pre-existing image" -- which is the
            # whole warrant for the finish's discard. So a short listing here
            # does not merely under-report, it loses an image and takes the
            # justification for the deletion with it.
            #
            # Refusing is safe in a way answering is not. `already: True` on a
            # short read would be the cautious answer for the stem it names, but
            # it is the wrong statement -- it tells the caller an image exists
            # when nothing here knows that -- and it cannot be given for a stem
            # the short answer did not mention at all, which is every stem on a
            # later page. A refusal names the condition and the caller retries.
            listing, listing_complete = await self._list_files_all(
                artifact_id, "images"
            )
            if not listing_complete:
                raise RuntimeError(
                    f"The images in '{artifact_id}' could not be listed in "
                    f"full, so whether '{name}' would overwrite an existing "
                    "image cannot be decided. NOTHING was changed and no "
                    "upload URL was handed out. Try again in a moment; if this "
                    "repeats, the dataset's storage is not answering and a "
                    "manager should look at it."
                )
            # A WITHHELD PATH IS NOT TAKEN, and omitting this made a refusal's
            # own instructions false. The refusals that leave the object staged
            # -- the marker removal failing, and before it the incomplete
            # listing -- all tell the caller to "retry with done=false and PUT
            # again". Without the withheld set here that retry answered
            # `already: True` for up to the marker's whole hour, because the
            # object IS in the listing: a dead end, and the round-9 review
            # blocked on exactly that shape on the sibling path.
            #
            # It is sound, not a convenience, and the induction is short: a
            # marker exists for a path only because this check passed when that
            # marker was minted, so an object at a withheld path is provably not
            # a pre-existing image -- which is the same fact that licenses the
            # discard. It also removes the last disagreement between the readers
            # and this check: the image list hides a withheld path, and a
            # re-mint no longer says that same path already exists.
            #
            # Fail-open, like the withholding itself: an unreadable `.uploads/`
            # yields an empty set, so the answer degrades to "already", which is
            # the refusing direction for a mint and merely unhelpful.
            # Short is the REFUSING direction here: fewer withheld names
            # means more names look taken, so an incomplete read answers
            # `already` rather than minting a URL it should not.
            withheld, _complete = await self._withheld_upload_paths(artifact_id)
            # prune=False: this is a read, and pruning is for the writer that
            # owns the manifest. A stem whose file has gone is still "taken"
            # for the purposes of not clobbering it here.
            entries, _changed = core.merge_image_manifest(
                core.parse_image_manifest(doc),
                core.image_names_from_listing(listing),
                withheld_paths=withheld,
            )
            existing = entries.get(stem)
            if isinstance(existing, dict):
                # Refused rather than overwritten, and reported as "already"
                # rather than as an error, because the caller uploading a
                # whole folder should be able to skip this one and carry on.
                # A stem is the manifest key, so re-using one would replace a
                # different image -- possibly a remote source, possibly one
                # with annotations already drawn against it.
                return {
                    "already": True,
                    "stem": stem,
                    "path": existing.get("path"),
                }

            # Reaching here means `path` is NOT an image in this dataset, by
            # the reconciled check above. That is the fact the marker records,
            # and it is the entire warrant for the discard on the finishing
            # call: an object with a marker cannot be a pre-existing image,
            # because a marker is only ever minted after this check passes.
            # Removing either half breaks the guarantee, so the no-overwrite
            # check above and the marker requirement below are one mechanism.
            new_id = core.new_mint_id()
            marker = core.upload_marker_document(
                path=path,
                mint_id=new_id,
                max_bytes=core.MAX_LOCAL_IMAGE_BYTES,
                minted_at=time.time(),
            )
            put_url_holder: Dict[str, str] = {}

            async def _mint():
                # The marker FIRST. A URL handed out before the marker exists
                # is a URL the client could discharge against no marker if the
                # marker write then failed, and the finish would refuse an
                # upload that did everything right.
                await self._write_upload_marker(artifact_id, marker)
                # `expires_in` is passed explicitly and SHORT. The default is an
                # hour, and a presigned PUT stays writable for its whole life:
                # under edit-staging the URL writes the same key the image lives
                # at, so after a successful finish a client can PUT an oversized
                # body to the same URL and nothing re-checks it. Measured in
                # review -- finish 32 bytes, then PUT 10,000 to the same URL, and
                # the listing size became 10,000. This app cannot close that
                # without help from the storage service; what it can do is make
                # the window minutes instead of an hour.
                put_url_holder["url"] = await self._am.put_file(
                    artifact_id,
                    file_path=path,
                    expires_in=core.UPLOAD_URL_TTL_S,
                )

            await self._ensure_staged(artifact_id, _mint)
            # No sweep here any more: it runs at the START of this call, above,
            # where it is reached whatever the no-overwrite check answers.
            return {
                "already": False,
                "stem": stem,
                "path": path,
                "put_url": put_url_holder["url"],
                "max_bytes": core.MAX_LOCAL_IMAGE_BYTES,
                "mint_id": new_id,
            }

        # THE CEILING, enforced here rather than advertised and forgotten. The
        # minting call above hands the client `max_bytes`, and until this check
        # existed nothing compared anything to it: an object of any size whose
        # first bytes matched the declared format became a dataset image.
        #
        # Read from the staged LISTING, not by fetching the object: a staged
        # listing already reports the size of what was just PUT (measured --
        # `{"type": "file", "name": ..., "size": 12345, ...}` for a
        # 12345-byte body), so enforcing the limit costs one listing and never
        # a download of the very object that is too big to want.
        #
        # Checked BEFORE the signature, deliberately: the signature check
        # fetches the first bytes of the object, and there is no reason to
        # reach into something already known to be over the limit.
        # AN EXISTING IMAGE IS NEVER TOUCHED, and the marker is the reason the
        # discard below is safe. A review measured the hole: the discard removed
        # whatever sat at `images/<name>` whenever the size check refused, and
        # an image ALREADY in the dataset can be over this route's ceiling,
        # because the mounted-folder path has no cap and MAX_LOCAL_IMAGE_BYTES
        # binds this route only. So one finishing call for an existing image's
        # filename -- a client that ignores `already: True` and finishes anyway,
        # a stray retry, or exactly the client bug this ceiling exists for --
        # deleted that image from the stage, and the next commit from any route
        # published the removal and orphaned its annotations. Before the discard
        # existed the same call was a harmless commit.
        #
        # THE COMMITTED LISTING CANNOT ANSWER THIS, and an earlier round of this
        # fix assumed it could. Measured against the live artifact manager: an
        # object PUT into an open stage is returned by `stage=True`, by `stage`
        # unset, by `stage=False`, by `version="v0"` and by `version="latest"`
        # alike. So "is `<name>` in the committed listing?" is true for every
        # upload that has landed, and a guard asking it would have refused every
        # legitimate finish while still not distinguishing the case it was
        # written for. That is why provenance is RECORDED at mint time instead.
        #
        # The marker is therefore not a convenience: it is the only fact
        # available that separates "this request's object" from "somebody's
        # image". No marker means no permission to touch anything -- not the
        # object, not the manifest, not the stage.
        marker = await self._read_upload_marker(artifact_id, path)
        if not core.marker_authorises_discard(
            marker, path=path, mint_id=mint_id, now=time.time()
        ):
            raise ValueError(
                f"This upload of '{name}' cannot be completed and NOTHING was "
                "changed. Finishing needs the 'mint_id' returned by the call "
                "that handed out the upload URL, and that upload must still be "
                "open: the record of it is missing, already used, older than "
                f"{int(core.UPLOAD_MARKER_TTL_S)} seconds, or belongs to a "
                "different file. Start again with done=false, PUT the bytes, "
                "and finish with the 'mint_id' from that answer."
            )

        # THE LISTING IS PAGED, AND HERE A SHORT ONE ALREADY FAILED CLOSED --
        # which is worth saying plainly, because the obvious guess is that this
        # was a size-ceiling bypass and it was not. `file_size_from_listing`
        # returns None for a name it does not find, and
        # `oversized_image_refusal` refuses on None by design. So before this
        # change, finishing an upload into a dataset of more than 1000 images
        # refused every time the just-PUT object landed past the first page,
        # with a message blaming the store for not reporting sizes. Wrong, and
        # a dead end for the user, but never a way past the ceiling.
        #
        # Paging fixes the false refusal. The completeness flag is still
        # checked, so that the one refusal left names the real condition
        # instead of the store's size reporting.
        listing, listing_complete = await self._list_files_all(artifact_id, "images")
        if not listing_complete:
            # A REFUSAL STRING, NOT A `raise` HERE, and the difference is the
            # staged object. Raising at this point would skip the discard below,
            # which the comment there explains is the whole reason the refusal
            # is safe: an object left at `images/<stem><ext>` becomes a dataset
            # image anyway, because the image list is built from the staged
            # listing and the next commit from any route commits the stage. So
            # this joins the existing path instead, and the ceiling's own
            # unreadable-size branch is the precedent -- it also removes.
            refusal = (
                f"The images in '{artifact_id}' could not be listed in full, "
                f"so the size of '{name}' could not be checked against the "
                f"{core.MAX_LOCAL_IMAGE_BYTES}-byte limit. The upload was NOT "
                "accepted and the file has been removed. Start again with "
                "done=false once the dataset's storage is answering; if this "
                "repeats, a manager should look at it."
            )
        else:
            refusal = core.oversized_image_refusal(
                name,
                core.file_size_from_listing(listing, name),
                core.MAX_LOCAL_IMAGE_BYTES,
            )
        if refusal:
            # THE STAGED OBJECT IS REMOVED, and raising without removing it was
            # a defect rather than an omission: a review measured that the
            # dataset's image list is built from the STAGED listing, so an
            # object left at `images/<stem><ext>` becomes an image anyway --
            # the refusal would have stopped only THIS route's commit. Three
            # consequences, all measured: the stem appeared in the image
            # manifest after the refusal; a re-mint answered `already: True`,
            # so the user could not retry that name with a correct file; and
            # the next commit from any route, which commits the whole stage,
            # would have made the oversized object permanent.
            #
            # Removed on the unreadable-size refusal too, deliberately. That
            # message also covers "the object is not there", and removing
            # something absent is a no-op -- while leaving something present
            # whose size cannot be read is exactly the case the ceiling exists
            # for. A removal that fails is logged and does not replace the
            # refusal: the caller must still hear why the upload was rejected.
            #
            # THE MARKER GOES WITH THE OBJECT. Leaving it would let the same
            # mint id be finished again against whatever lands at that path
            # next, and the whole point of consuming it here is that one mint
            # authorises exactly one finish.
            await self._discard_staged_upload(artifact_id, path)
            await self._remove_upload_marker(artifact_id, path)
            raise ValueError(refusal)

        # A presigned PUT reports success from S3, not from the artifact
        # manager, so the client saying it uploaded an image is not evidence
        # that an image is there. Check the signature against the extension
        # the caller declared before anything records a readable image at this
        # path: an error page or a truncated body stored under a .png name
        # would otherwise become a dataset image that fails to open later,
        # with nothing pointing back here.
        head = await self._object_head(artifact_id, path, 8)
        if not core.header_matches_ext(ext, head):
            # DISCARDED on this refusal too, for the same measured reason as
            # the size refusal above: the dataset's image list is built from the
            # staged listing, so an object left at `images/<stem><ext>` becomes
            # an image anyway and the refusal would have stopped only THIS
            # route's commit. The object the marker vouches for is not a readable
            # image, and publishing it is the outcome the check exists to stop.
            # Safe for exactly the same reason the size discard is safe -- the
            # marker -- and unsafe without it, which is why this moved only now.
            await self._discard_staged_upload(artifact_id, path)
            await self._remove_upload_marker(artifact_id, path)
            raise ValueError(
                f"The bytes at '{path}' are not a {ext.lstrip('.').upper()} "
                f"image. The upload did not land, or something other than the "
                f"file was stored, so it was not accepted. Re-run with "
                f"done=False and PUT again, sending the body with no "
                f"Content-Type header."
            )
        # THE MARKER FOR THIS PATH IS CONSUMED, and the consumption FAILS
        # CLOSED. Both of those were measured as defects before they were rules.
        #
        # A removal that fails must stop the commit. It used to be best-effort:
        # `_remove_upload_marker` never raises, so the commit went ahead and the
        # leftover marker had the sweep delete the committed image later. Hypha
        # carries out a staged removal of a committed file on the next commit,
        # so that loss becomes permanent. Refusing here leaves the object staged
        # and still withheld, which the next mint's sweep clears and the client
        # can retry -- the cost of the safe direction is an error on an upload
        # that was otherwise fine, and that is the right way round.
        #
        # Consumed BEFORE the commit, so the commit publishes the image without
        # it. Removing it afterwards would leave the removal staged and waiting
        # on somebody else's commit, which is how `.uploads/` would end up in a
        # published dataset.
        #
        # ONE MARKER, FOUND BY NAME, WITH NO LISTING. This is where four review
        # rounds went, so the reasoning is worth keeping. More than one marker
        # for one path used to be ordinary -- a client whose mint answer was lost
        # asks again, two browser tabs upload the same filename -- and each one
        # left behind withheld the committed image and then had the expiry sweep
        # delete it. Consuming them meant FINDING them, which meant listing
        # `.uploads/`, which is paginated: a short read left a duplicate alive
        # (data loss through housekeeping), and every bound tight enough to
        # terminate against a server that ignores `offset` refused valid uploads
        # once a few staged removals sat in the directory (a clause-2
        # regression). A cap cannot decide this correctly in either direction,
        # because the question itself was the mistake.
        #
        # So the marker is keyed by the OBJECT PATH -- `.uploads/<sha256(path)>`
        # -- and a re-mint overwrites the one file instead of adding a second.
        # The duplicate cannot exist, so the finish does not look for one: it
        # computes the name from the path it is finishing and removes that file.
        # Authorisation is unchanged and still comes from the DOCUMENT (this
        # path, this mint id, unexpired), checked above, so the filename is a
        # lookup key and confers nothing. Nothing here paginates, so nothing
        # here can be short.
        # ONE KNOWN WRINKLE OF REUSING A FILENAME, read out of Hypha rather
        # than measured here, and recorded so the next reader does not have to
        # rediscover it. Removing a COMMITTED path appends a `_remove` entry to
        # the staging list, and a later put_file to the SAME path is not added,
        # because the staging check matches on path alone and finds that entry.
        # So after an abandoned upload's object and marker are published by
        # somebody's commit and then swept, a re-upload of the same name --
        # before any further commit -- is still hidden from staged listings, and
        # its finish reads the size as unreadable and refuses ONCE. That
        # refusal's own discard and marker removal clear the flags, so the
        # retry works. One spurious refusal, no loss and no bypass. It applies
        # to object paths too, which were already reused; the path-keyed marker
        # only adds a second filename with the same property.
        own = core.upload_marker_path_for_object(path)
        if own is None or not await self._remove_marker_file(artifact_id, own):
            raise RuntimeError(
                f"'{name}' was uploaded and checked, but the record of the "
                "upload could not be cleared, so it was NOT published. "
                "Publishing with that record still in place would have let "
                "housekeeping delete the image later. Nothing was committed; "
                "retry with done=false and PUT again."
            )
        await self._commit_tolerating_pending_uploads(artifact_id)
        # Drop the cached image list. Every other route that adds or removes an
        # image refreshes it; without this the next `get_dataset_index` can be
        # served a list computed up to `MANIFEST_CACHE_TTL_S` ago and leave out
        # the image that was just uploaded -- so the caller is told the upload
        # succeeded and then cannot see it. Popped rather than rewritten
        # because this route does not build `entries`: the reconciliation does,
        # and the next read will do it with the new file present.
        self._image_manifest_cache.pop(artifact_id, None)
        return {"already": False, "stem": stem, "path": path, "committed": True}

    @bioengine.method(context=True)
    async def create_label(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        name: str = Field(..., description="Label name, must match ^[a-z0-9._-]+$."),
        description: str = Field("", description="Human-readable label description."),
        context=None,
    ) -> Dict[str, Any]:
        """Create a new annotation label: broker metadata + a
        ``label_<name>/metadata.json`` file carrying the description (and any
        future per-label info). The artifact manifest is NOT touched, and no
        empty placeholder files are written."""
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, _role = self._require_role(context, artifact_id, "manager")
        description = description if isinstance(description, str) else ""
        core.add_label(meta, name, description)  # raises ValueError on a bad name
        meta = core.write_metadata(meta, root=STATE_ROOT)

        async def _put_label_metadata():
            await self._write_json_file(
                artifact_id,
                core.label_metadata_path(name),
                {
                    "name": name,
                    "description": description,
                    "created_at": core.now_iso(),
                },
            )

        await self._ensure_staged(artifact_id, _put_label_metadata)
        return meta

    @bioengine.method(context=True)
    async def get_dataset_index(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        context=None,
    ) -> Dict[str, Any]:
        """One-shot payload for the annotate page: images, embeddings, labels,
        and the caller's own latest annotation per (label, stem). Callers
        should never have to reason about stage/permission edge cases
        themselves.

        Role ``public`` (anonymous visitor on a public dataset) gets the
        read-only payload with ``my_annotations`` omitted — annotating always
        requires a login.

        Contract since 0.5.0: NO presigned URLs in the index. Minting one URL
        per file made an 86-image index take ~30 s; the index now returns
        stems and presence only (a few list_files calls), and callers fetch
        URLs on demand via get_image_url / get_embedding_urls /
        get_my_annotation_url.
        """
        artifact_id = self._canonical_id(artifact_id)
        meta, caller, role = self._require_role(context, artifact_id, "public")

        dims_cache: Dict[str, Any] = dict(meta.get("image_dims") or {})
        manifest = await self._image_manifest(artifact_id, dims=dims_cache)
        images = [
            {"stem": stem, "state": entry.get("state"), "path": entry.get("path")}
            for stem, entry in sorted(manifest.items())
        ]

        # Per-image width/height (0.8.0) so the frontend can flag images
        # below the segmentation models' input minimum without loading
        # pixels. Since 0.10.0 the manifest carries the dimensions for any
        # image imported with them known, and this ranged PNG-header read is
        # the fallback for the ones that predate it. It costs 24 bytes per
        # stem, is cached in the dataset's broker metadata, and failed reads
        # stay uncached so a later index call retries them.
        missing_dims = [
            stem
            for stem, entry in manifest.items()
            if "width" not in entry and stem not in dims_cache and entry.get("path")
        ]
        if missing_dims:
            dims_sem = asyncio.Semaphore(16)

            async def _fetch_dims(stem: str):
                async with dims_sem:
                    try:
                        url = await self._am.get_file(
                            artifact_id=artifact_id,
                            file_path=manifest[stem]["path"],
                            stage=True,
                        )
                        resp = await self._http.get(
                            url, headers={"Range": "bytes=0-23"}
                        )
                        resp.raise_for_status()
                        return stem, core.parse_png_dims(resp.content[:24])
                    except Exception:
                        return stem, None

            dims_results = await asyncio.gather(*(_fetch_dims(s) for s in missing_dims))
            cache_grew = False
            for stem, dims in dims_results:
                if dims is not None:
                    dims_cache[stem] = dims
                    cache_grew = True
            if cache_grew:
                meta["image_dims"] = dims_cache
                core.write_metadata(meta, root=STATE_ROOT)
        for img in images:
            entry = manifest[img["stem"]]
            dims = (
                [entry["width"], entry["height"]]
                if "width" in entry
                else dims_cache.get(img["stem"])
            )
            if dims:
                img["width"], img["height"] = int(dims[0]), int(dims[1])

        embedding_entries = await self._list_files_safe(artifact_id, "embeddings")
        embedding_names = [
            entry.get("name", "") if isinstance(entry, dict) else str(entry)
            for entry in embedding_entries
        ]
        # Per-stem entries carry every stored model in ``model_types`` (0.9.0,
        # for the per-model embedding badges) plus the single backward-
        # compatible ``model_type`` older clients read.
        embeddings: Dict[str, Any] = core.collect_embeddings(embedding_names)

        my_annotations: Dict[str, Any] = {}
        if role != "public":
            for label in meta.get("labels", []):
                label_name = label.get("name")
                if not label_name:
                    continue
                dir_path = core.user_label_dir(label_name, caller["id"])
                entries = await self._list_files_safe(artifact_id, dir_path)
                filenames = [
                    e.get("name", "") if isinstance(e, dict) else str(e)
                    for e in entries
                ]
                pairs = core.latest_pairs_by_stem(filenames)
                if pairs:
                    my_annotations[label_name] = {
                        stem: {"latest_ts": pairs[stem]["timestamp"]} for stem in pairs
                    }

        return {
            "images": images,
            "embeddings": embeddings,
            "labels": meta.get("labels", []),
            "my_annotations": my_annotations,
            "role": role,
        }

    @bioengine.method(context=True)
    async def get_my_annotation_url(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: str = Field(..., description="Label name."),
        image_stem: str = Field(..., description="Image stem."),
        context=None,
    ) -> Dict[str, Any]:
        """Presigned GET URL for the caller's LATEST geojson annotation of
        one image under one label (the refine flow), minted on demand."""
        artifact_id = self._canonical_id(artifact_id)
        _meta, caller, _role = self._require_role(context, artifact_id, "annotator")
        dir_path = core.user_label_dir(label, caller["id"])
        entries = await self._list_files_safe(artifact_id, dir_path)
        filenames = [
            e.get("name", "") if isinstance(e, dict) else str(e) for e in entries
        ]
        pairs = core.latest_pairs_by_stem(filenames)
        pair = pairs.get(image_stem)
        if not pair:
            return {"exists": False}
        url = await self._am.get_file(
            artifact_id=artifact_id,
            file_path=f"{dir_path}/{pair['geojson']}",
            stage=True,
        )
        return {"exists": True, "latest_ts": pair["timestamp"], "geojson_read_url": url}

    @bioengine.method(context=True)
    async def get_image_url(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        image_stem: str = Field(
            ..., description="Image stem (filename without extension)."
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Where to read one image from. Featherweight first call for the
        annotate page so the image renders before the full dataset index
        (which mints URLs for every file) has been assembled.

        Two shapes, discriminated by ``kind``:

        * ``{"stem", "kind": "file", "read_url"}`` -- a materialised image,
          presigned as before. The pre-0.10.0 shape is a subset of this one,
          so a stale frontend still renders local images.
        * ``{"stem", "kind": "ngff", "store_root", "multiscale_path",
          "region", "planes", "level0_shape"}`` -- a remote OME-Zarr source the
          browser reads directly. ``region`` is null unless the image was
          imported as a patch of a whole-slide source, in which case it is the
          crop, in level-0 pixels, that the reader must confine itself to.
          ``planes`` is which slice of each non-spatial axis the annotation
          belongs to, by axis NAME, or null when none was recorded; and
          ``level0_shape`` is the level-0 dataset shape the per-level plane
          index is derived from, or null. Note it is POSITIONAL and the axis
          names are NOT returned, so a reader must take axis order from the
          store's own ``multiscales`` metadata. Both are needed together: a pyramid
          that resamples z has a different index per level for the same
          physical slice, and neither value alone lets a reader work out which.

        For a remote source any credential is spliced in **here**, after the
        role check, and never stored in the artifact (design §12).
        """
        artifact_id = self._canonical_id(artifact_id)
        meta, _caller, _role = self._require_role(context, artifact_id, "public")
        entry = await self._image_entry(artifact_id, image_stem)

        if not core.image_entry_is_local(entry):
            source = entry.get("source") or {}
            if source.get("kind") != core.NGFF_SOURCE_KIND:
                raise FileNotFoundError(
                    f"Image '{image_stem}' has no file and no readable source "
                    f"(state={entry.get('state')!r})."
                )
            store_root = source.get("store_root") or ""
            credential = (meta.get("ngff_credentials") or {}).get(store_root)
            layout = source.get("layout") or {}
            return {
                "stem": image_stem,
                "kind": "ngff",
                "store_root": core.apply_store_credential(store_root, credential),
                "multiscale_path": source.get("multiscale_path"),
                "region": source.get("region"),
                # Both recorded at import and both needed by the viewer to pick
                # the right slice at every pyramid level. `planes` says which
                # slice of each non-spatial axis the annotation belongs to;
                # `level0_shape` is what a level's own plane index is derived
                # FROM, since a pyramid that resamples z has a different index
                # per level for the same physical slice.
                #
                # Dropping them here is not cosmetic. On a z-downsampled store
                # each level otherwise pins its own middle slice, so a region
                # recorded at z 4 of 9 is drawn at level 1 index 2 -- which
                # covers level-0 slices 4.5 to 6.75 and does not include 4. The
                # annotation would be made on one slice and the record would
                # name another, which is the mispairing the plane exists to
                # prevent, reappearing one level down.
                #
                # NULL rather than absent when unrecorded: a 2-D source has no
                # plane to report, and the viewer distinguishes "no plane, draw
                # the only slice" from "plane unknown, refuse". An absent key
                # and an explicit null are the same thing over this transport,
                # but the explicit form is what the shape above documents.
                "planes": source.get("planes") or None,
                "level0_shape": layout.get("level0_shape") or None,
            }

        read_url = await self._am.get_file(
            artifact_id=artifact_id, file_path=entry["path"], stage=True
        )
        return {"stem": image_stem, "kind": "file", "read_url": read_url}

    @bioengine.method(context=True)
    async def get_embedding_urls(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        image_stem: str = Field(
            ..., description="Image stem (filename without extension)."
        ),
        model_type: str = Field(..., description="μSAM model type, e.g. 'vit_l_lm'."),
        context=None,
    ) -> Dict[str, Any]:
        artifact_id = self._canonical_id(artifact_id)
        """Existing embedding read URL, or URLs for the micro-sam app to
        compute and upload a new one."""
        self._require_role(context, artifact_id, "annotator")
        emb_path = core.embedding_path(image_stem, model_type)
        if await self._file_exists(artifact_id, emb_path):
            read_url = await self._am.get_file(
                artifact_id=artifact_id, file_path=emb_path, stage=True
            )
            return {"exists": True, "read_url": read_url}

        image_file_path = await self._image_file_path(artifact_id, image_stem)
        image_read_url = await self._am.get_file(
            artifact_id=artifact_id, file_path=image_file_path, stage=True
        )
        embedding_put_url = await self._am.put_file(artifact_id, file_path=emb_path)
        return {
            "exists": False,
            "image_read_url": image_read_url,
            "embedding_put_url": embedding_put_url,
        }

    @bioengine.method(context=True)
    async def remove_embedding(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        image_stem: str = Field(
            ..., description="Image stem (filename without extension)."
        ),
        model_type: Optional[str] = Field(
            None,
            description="μSAM model type to clear, e.g. 'vit_l_lm'. Omit to clear "
            "this image's cached embeddings for every model type.",
        ),
        context=None,
    ) -> Dict[str, Any]:
        """Delete cached embedding ``.npz`` files for one image so the next
        segmentation recomputes them (recovery path for a stale or mismatched
        embedding). Embeddings are pure cache, so any annotator may clear
        them."""
        artifact_id = self._canonical_id(artifact_id)
        self._require_role(context, artifact_id, "annotator")
        model_type = model_type if isinstance(model_type, str) else None
        removed = []

        async def _rm_all():
            for path in core.embedding_removal_paths(image_stem, model_type):
                if await self._file_exists(artifact_id, path):
                    await self._am.remove_file(artifact_id=artifact_id, file_path=path)
                    removed.append(path)

        await self._ensure_staged(artifact_id, _rm_all)
        if removed:
            # Removing a file that exists in the COMMITTED version only stages
            # a deletion marker; a subsequent put_file + upload for the same
            # path then stays invisible to list_files (and thus _file_exists)
            # until the deletion is committed — which made every recompute of
            # a committed embedding fail with "unavailable after upload".
            # Commit the removal and re-stage so the path can be re-uploaded
            # and observed immediately (same pattern as delete_label).
            await self._commit_tolerating_pending_uploads(artifact_id)
            await self._am.edit(artifact_id=artifact_id, stage=True)
        return {"removed": removed}

    @bioengine.method(context=True)
    async def get_save_urls(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        label: str = Field(..., description="Label name."),
        image_stem: str = Field(
            ..., description="Image stem (filename without extension)."
        ),
        context=None,
    ) -> Dict[str, Any]:
        artifact_id = self._canonical_id(artifact_id)
        """Presigned PUT URLs for a new (png, geojson) annotation pair,
        sharing one server-generated UTC timestamp. Also upserts
        ``label_<label>/users.json`` with the caller's identity."""
        meta, caller, _role = self._require_role(context, artifact_id, "annotator")
        if not any(l.get("name") == label for l in meta.get("labels", [])):
            raise ValueError(f"Unknown label '{label}' for dataset '{artifact_id}'.")

        timestamp = core.new_timestamp()
        paths = core.annotation_save_paths(label, caller["id"], image_stem, timestamp)
        png_put_url = await self._am.put_file(artifact_id, file_path=paths["png"])
        geojson_put_url = await self._am.put_file(
            artifact_id, file_path=paths["geojson"]
        )

        safe_user = core.sanitize_user_id(caller["id"])
        users_path = core.label_users_path(label)

        async def _upsert_users():
            current = await self._read_json_file(artifact_id, users_path, default={})
            updated = core.upsert_label_user(
                current, safe_user, caller["id"], caller["email"]
            )
            await self._write_json_file(artifact_id, users_path, updated)

        await self._ensure_staged(artifact_id, _upsert_users)

        return {
            "timestamp": timestamp,
            "png_put_url": png_put_url,
            "geojson_put_url": geojson_put_url,
        }

    @bioengine.method(context=True)
    async def delete_dataset_record(
        self,
        artifact_id: str = Field(..., description="Dataset artifact id."),
        context=None,
    ) -> Dict[str, Any]:
        artifact_id = self._canonical_id(artifact_id)
        """Broker-side cleanup after the frontend deletes the underlying
        artifact."""
        self._require_role(context, artifact_id, "owner")
        deleted = core.delete_metadata(artifact_id, root=STATE_ROOT)
        self._manifest_name_cache.pop(artifact_id, None)
        self._image_manifest_cache.pop(artifact_id, None)
        self._image_manifest_locks.pop(artifact_id, None)
        return {"deleted": deleted}
