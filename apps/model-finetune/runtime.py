"""micro-sam GPU runtime — resident SAM encoder + shared GPU lock + training.

The GPU half of the two-deployment micro-sam app (the CPU ``EntryApp`` in
``entry.py`` owns transport/orchestration and composes this runtime). One
resident SAM encoder backs three serving consumers (Leg B) — embedding, AIS
masks, and the ONNX prompt decoder — and a single ``asyncio.Lock`` serialises
**every** GPU operation (serving *and* training), exactly like model-runner's
predict+test. Fine-tuning runs in a **subprocess** so the OS reclaims all
training VRAM on exit; the resident inference model is evicted first so the
subprocess gets the whole GPU.

Autoscaling declares ``max_replicas=3``. The per-replica target it also declared
was spelled with a field name Ray Serve removed, so the effective target has
always been Serve's default of 2 — at which a long training plus one
concurrent inference asks for one replica, not two. This runtime has therefore
never run training and inference on separate GPU replicas; whether to pin it to
one replica or restore the eager target is a separate open question.

Import rule: heavy deps (``torch``, ``micro_sam``, ``segment_anything``) are
imported inside method bodies so the ``@bioengine.app`` decorator module stays
introspectable with only ``bioengine[worker]`` + the stdlib.
"""

import asyncio
import gc
import importlib
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import bioengine
import busy
import numpy as np

logger = bioengine.logger

# The six micro-SAM generalists, each pinned to an exact bioimage.io record so a
# zoo-side update never changes served weights without an app release. Base SAM
# and fine-tuned checkpoints do not load through here.
ZOO_MODELS = {
    "vit_t_lm": ("faithful-chicken", "1.1"),
    "vit_b_lm": ("diplomatic-bug", "1.2"),
    "vit_l_lm": ("idealistic-rat", "1.2"),
    "vit_t_em_organelles": ("greedy-whale", "1"),
    "vit_b_em_organelles": ("noisy-ox", "1.2"),
    "vit_l_em_organelles": ("humorous-crab", "1.2"),
}
_ZOO_BASE_URL = "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io"

# How long the base-checkpoint reachability probe may take PER FILE. Short on
# purpose: this runs on the admission path, and the thing it protects against
# is a run that claims the GPU and then discovers the weights minutes later.
# A probe that could itself take minutes would reproduce the problem.
BASE_CHECKPOINT_PROBE_TIMEOUT_S = 3.0

# Methods a caller may rely on being present. Advertised through `ping`.
RUNTIME_CAPABILITIES = ("base_checkpoint_status",)


def _url_reachable(url: str, timeout_s: float) -> bool:
    """Is *url* fetchable from HERE, without downloading it?

    A HEAD, falling back to a 1-byte ranged GET, because some stores answer
    HEAD with 405 while serving GET perfectly well -- treating that as
    unreachable would refuse a model the site can actually train.

    Any failure is False rather than an exception: the caller is deciding
    whether to admit a run, and "I could not tell" has to behave like "not
    available here" or the refusal it feeds is not a refusal at all.
    """
    import urllib.error
    import urllib.request

    for make in (
        lambda: urllib.request.Request(url, method="HEAD"),
        lambda: urllib.request.Request(url, headers={"Range": "bytes=0-0"}),
    ):
        try:
            with urllib.request.urlopen(make(), timeout=timeout_s) as resp:
                if 200 <= getattr(resp, "status", 0) < 400:
                    return True
        except urllib.error.HTTPError as exc:
            if 200 <= exc.code < 400:
                return True
            continue
        except Exception:
            continue
    return False


#: Where the mirror artifact lives. Overridable by environment for a test
#: deployment; the default is the production server, same as everything else
#: this app talks to.
_HYPHA_SERVER_URL = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")


def _open_url_stream(url: str, timeout_s: float = 300.0):
    """A readable stream over *url*. Separate so a test can pass its own.

    `urlopen`'s object already has `.read(n)`, which is all the fence needs, so
    the bytes are never collected into memory: a 375 MB encoder would otherwise
    be held whole inside a Serve replica.
    """
    import urllib.request

    return urllib.request.urlopen(url, timeout=timeout_s)


def _install_from_url(bcm, root: str, name: str, url: str, opener) -> dict:
    """Stream *url* into *root* as *name*, through the fence. Blocking.

    Runs in a thread because it is a long synchronous read; the fence is
    `bcm.install_verified`, and a `MirrorVerificationError` propagates on
    purpose.
    """
    stream = opener(url)
    try:
        return bcm.install_verified(root, name, stream)
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass


# The third-party top-level modules this runtime's own method bodies import.
# Imported once at replica startup by the smoke test below so a replica whose
# runtime_env install did not take fails instead of being advertised as ready.
# Keep this list equal to what the bodies actually import: it is a
# readiness gate, not a restatement of requirements-runtime.txt, and a module
# nothing imports would fail a replica that can serve every call. ``numpy`` is
# absent on purpose — it is imported at module scope above, so a replica missing
# it never constructs.
_RUNTIME_IMPORTS = ("torch", "micro_sam", "segment_anything", "onnxruntime", "pooch")


def _read_pip(name: str) -> List[str]:
    text = (Path(__file__).parent / name).read_text()
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@bioengine.app(
    num_cpus=2,
    gpu_memory_mb=-1,
    memory_mb=12 * 1024,
    pip=_read_pip("requirements-runtime.txt"),
    # Let up to 10 RPCs reach a replica so backlog behind ``_gpu_lock`` is
    # observable to the autoscaler; the lock serialises the actual GPU work.
    max_ongoing_requests=10,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 3,
        # No target declared on purpose: the name this dict used is one Serve
        # removed, so the effective target is Serve's default of 2 either way,
        # and declaring 1.0 now is a behaviour change, not a fix.
        "metrics_interval_s": 2.0,
        "look_back_period_s": 10.0,
        "downscale_delay_s": 600,
        "upscale_delay_s": 0.0,
    },
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    # A full training run holds the lock for minutes; give the replica time to
    # drain before it exits.
    graceful_shutdown_timeout_s=600.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class RuntimeApp:
    """GPU compute: resident encoder, shared GPU lock, subprocess fine-tuning."""

    def __init__(self) -> None:
        self.start_time = time.time()
        # Serialises ALL GPU work (serving + training) within a replica. The queue
        # it creates is what the autoscaler measures, but at the effective target
        # of 2 a single concurrent call waits here rather than adding a
        # replica; it takes three in flight to ask for a second one.
        self._gpu_lock = asyncio.Lock()
        self._predictor = None
        self._segmenter = None
        self._loaded_model_type: Optional[str] = None
        self._loaded_key: Optional[tuple] = None
        self._onnx_cache: Dict[str, bytes] = {}
        self._device_cached: Optional[str] = None
        self._gpu_mem_cached: Optional[Dict[str, Any]] = None
        # No soup path here, but starvation is not soup-specific: any long call
        # holding this lock queues every other caller on the replica.
        self._busy = busy.BusyLog(
            {"micro-sam GPU work (_gpu_lock)": self._gpu_lock.locked}
        )
        # Whether THIS replica's run store accepts a write, and when that was
        # last established. A runtime replica has its own HOME and therefore
        # its own store: #0931 measured one replica writing the share while
        # another, in the same pod image, could not write anything, because
        # the redirect that points HOME at the share had not run. So this is a
        # fact about the process, not about the cluster, and it cannot be
        # answered by asking the entry.
        #
        # None means "not established yet" and is reported as such rather than
        # as either answer.
        self._store_ok: Optional[bool] = None
        self._store_detail: str = "not probed yet"
        self._store_checked_at: float = 0.0

    @bioengine.async_init
    async def _start_busy_log(self) -> None:
        self._busy.start()
        # The store verdict is established ONCE here, so the first caller gets
        # an answer rather than "not probed yet". It shares this hook because
        # an app may declare only one `@bioengine.async_init` -- adding a
        # second raises ReservedMethodNameError at import, which is how I found
        # out.
        #
        # Deliberately NOT fatal. A replica that cannot write the run store can
        # still serve inference, and failing init would take the GPU runtime
        # down for a fault only training cares about -- the restart loop this
        # design exists to avoid, reached from the other side.
        try:
            await self._probe_store()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"the startup store probe itself failed: {exc}")

    @bioengine.smoke_test
    async def _import_runtime_stack(self) -> None:
        """Import this backend's ML stack once, at startup, and fail the replica if
        it is not there.

        A replica whose ``runtime_env`` install did not take answered the health
        probe correctly and was advertised as ready while **every** call raised
        ``ModuleNotFoundError``: nothing on the health path imports the ML stack.
        ``check_health`` runs ``async_init`` (the busy log) and, because this
        deployment declares a GPU, retains a CUDA context via ctypes on
        ``libcuda.so.1`` — which the NVIDIA container runtime injects independently
        of the pip layer, so it succeeds on a replica with no torch. This runtime
        declares no ``@bioengine.health_check``, so nothing else ran.

        This is deliberately a smoke test and not a deeper ``ping()``: the entry
        calls ``ping()`` with a 2 s timeout before every GPU call and importing
        torch exceeds that, while a per-poll import of a heavy stack is its own
        defect. The framework runs a smoke test once after ``async_init`` and
        fails the replica when it raises, which makes the failure *visible*.

        It is **not** a traffic gate, and not a bounded one either. Two layers:
        ``check_health`` creates the task and tests ``.done()`` in the same call,
        so poll 1 cannot fail, and Serve condemns a replica only after 3
        consecutive failed checks (``REPLICA_HEALTH_CHECK_UNHEALTHY_THRESHOLD``)
        — ≈90 s at this deployment's ``health_check_period_s=30.0``. Serve then
        **replaces** the condemned replica and the cycle repeats indefinitely,
        because its circuit breaker ``_terminally_failed()`` is gated on ``not
        _replica_has_started``, which any successful start clears. Measured on
        a probe deployment over 100 health-check periods — that probe ran at
        ``health_check_period_s=1.0``, so ≈100 s in all, **not** 100 × the 30 s
        above: 105 calls, **0 failed**, 84 answered by a replica that already
        knew its own smoke test had failed, across 22 distinct replica pids.

        So what this buys is that the missing modules are **named in the replica
        log on every cycle**, instead of nothing being said at all — not that
        broken traffic stops. The import is not wasted work — the first real
        call would do it anyway.

        Reports every missing module rather than the first, because a half-installed
        env usually lost more than one and the whole point is diagnosability.
        """
        missing: List[str] = []
        for name in _RUNTIME_IMPORTS:
            try:
                importlib.import_module(name)
            except Exception as e:
                missing.append(f"{name} ({type(e).__name__}: {e})")
        if missing:
            raise RuntimeError(
                "micro-sam runtime dependencies are not importable in this "
                "replica's environment — the runtime_env install did not take: "
                + "; ".join(missing)
                + ". pip's own output is not in this app's logs; read "
                "runtime_env_setup-*.log on the node."
            )

    #: How long a store verdict is reused before `store_status` re-probes. The
    #: probe writes a file, so this is not free; the fault it detects is a
    #: property of the process and effectively static, so a minute is ample.
    # Class-level defaults, so `ping` cannot raise AttributeError on an
    # instance whose __init__ did not run -- which is how the suite exercises
    # `ping` alone, and it is also what any future constructor refactor would
    # produce. Immutable values only; every write assigns to the INSTANCE.
    _store_ok: Optional[bool] = None
    _store_detail: str = "not probed yet"
    _store_checked_at: float = 0.0

    STORE_PROBE_TTL_S = 60.0

    async def _probe_store(self) -> Dict[str, Any]:
        """Establish whether this replica can write its run store.

        Runs the write probe off the event loop: it touches a filesystem that
        may be a network mount, and a hung mount must not block this replica's
        other callers.

        NO `@bioengine.health_check` CALLS THIS, and that is the design rather
        than an omission. A failing Serve health check RESTARTS the replica, so
        reporting an unwritable store through the health check would crash-loop
        the GPU runtimes under a cluster-side fault the app cannot repair --
        which is exactly the condition that produced this row, and it lasted
        hours. Any "fail after N attempts" threshold is a bet on how long such
        a fault lasts; a probe that never fails the check cannot be tuned
        wrong. The fault is surfaced instead through `ping`/`store_status` and
        refused at the door by the entry, which is what an operator and a user
        need. The cost, stated: Serve's own health view stays green.
        """
        import training

        try:
            root = await asyncio.to_thread(training.assert_sessions_root_writable)
            self._store_ok = True
            self._store_detail = f"the run store at {root} accepts writes"
        except Exception as exc:
            self._store_ok = False
            self._store_detail = str(exc)
        self._store_checked_at = time.time()
        return self._store_report()

    def _store_report(self) -> Dict[str, Any]:
        return {
            "writable": self._store_ok,
            "detail": self._store_detail,
            "checked_at": self._store_checked_at,
            "age_s": (
                None
                if not self._store_checked_at
                else round(time.time() - self._store_checked_at, 1)
            ),
        }

    async def store_status(self, max_age_s: Optional[float] = None) -> Dict[str, Any]:
        """This replica's store verdict, re-probed if the cached one is stale.

        Composition endpoint, like `ping` and `gpu_memory_info` -- not a Hypha
        method, so a caller cannot reach it directly and cannot ask this
        replica to write on demand.
        """
        ttl = self.STORE_PROBE_TTL_S if max_age_s is None else max_age_s
        if self._store_ok is None or (time.time() - self._store_checked_at) > ttl:
            return await self._probe_store()
        return self._store_report()

    async def ping(self) -> Dict[str, Any]:
        """Internal liveness for the entry's readiness check — not a Hypha method,
        but reachable via the composition handle. Unlocked so it answers during
        GPU work.
        """
        return {
            "status": "ok",
            "loaded_model_type": self._loaded_model_type,
            "gpu_busy": self._gpu_lock.locked(),
            "uptime": time.time() - self.start_time,
            # The CACHED store verdict. `ping` never probes: the entry calls it
            # inside a 2 s readiness timeout, and a replica wedged on a hung
            # mount must not make that readiness check hang -- which is the
            # failure the timeout exists for in the first place.
            "store": self._store_report(),
            # WHAT THIS REPLICA CAN BE ASKED, so a newer entry can tell an older
            # replica from a broken one. `getattr` on the composition handle
            # cannot: BioEngineRuntimeHandle.__getattr__ returns a bound method
            # for ANY public name, so a missing method is indistinguishable from
            # a present one until the call fails -- and treating that failure as
            # "unavailable" would refuse every run during a rollout.
            "capabilities": sorted(RUNTIME_CAPABILITIES),
        }

    async def gpu_memory_info(self) -> Dict[str, Any]:
        """Total/free VRAM of this runtime's GPU, for the entry's training gate.
        Composition endpoint (not a Hypha method); total VRAM is static so it's
        cached after the first sample."""
        if self._gpu_mem_cached is None:
            import training

            self._gpu_mem_cached = training.detect_gpu_memory()
        return self._gpu_mem_cached

    async def base_checkpoint_status(self, model_type: str) -> Dict[str, Any]:
        """Can THIS SITE produce the base weights training would start from?

        Composition endpoint, not a Hypha method. Answers for the TRAINING base
        model, which is a different thing from the zoo weights `_zoo_weights`
        fetches for inference: `train_sam` with no `checkpoint_path` resolves
        the base model through micro-sam's OWN pooch registry, so probing the
        zoo cache would answer a question nobody asked and pass while the
        training fetch still failed.

        A STATEMENT ABOUT THE SITE, not a list of model types. A hard-coded
        list goes stale the moment a mirror gains a file, and the whole point
        of this probe is that the same model_type is runnable where a mirror or
        upstream is reachable and not runnable where neither is.

        Read-only and bounded: it stats the cache and, only for what is not
        cached, issues a HEAD with a short timeout. It never downloads -- the
        cost it exists to avoid is exactly the one a download would pay.

        CACHED MEANS PRESENT, NOT VALID. The check is `os.path.exists`, so a
        file whose hash no longer matches micro-sam's registry counts as
        cached here while `pooch.fetch` would re-download it. That is the
        right trade for an admission check -- verifying hashes would mean
        reading every byte of multi-hundred-megabyte weights on a path that
        must stay quick -- but it means this answers "is there a file here",
        and a corrupt one still fails later.

        Returns the parts rather than a verdict, so the caller decides and the
        decision is testable without a network: ``{"model_type", "parts":
        {name: {"cached": bool, "reachable": bool|None, "url": str|None}},
        "probe_error": str|None}``. ``reachable`` is None when it was not
        asked, because the file is already cached.
        """
        import os

        out: Dict[str, Any] = {
            "model_type": model_type,
            "parts": {},
            "probe_error": None,
        }
        try:
            from micro_sam.util import models as _models

            registry = _models()
            root = str(registry.path)
            urls = dict(getattr(registry, "urls", {}) or {})
            known = set(getattr(registry, "registry", {}) or {})
        except Exception as exc:  # the probe itself failed; say so, do not guess
            out["probe_error"] = "%s: %s" % (type(exc).__name__, exc)
            return out

        # The encoder is keyed by the model type; the AIS decoder by
        # "<model_type>_decoder". BOTH are needed, and a site can hold one and
        # not the other -- so checking only the encoder would admit a run whose
        # decoder is unobtainable here.
        # THE PARTS ARE PROBED CONCURRENTLY, and the arithmetic is the reason.
        # `_url_reachable` makes up to TWO attempts (HEAD, then a ranged GET),
        # so a part costs up to 2 x BASE_CHECKPOINT_PROBE_TIMEOUT_S. Run
        # sequentially, two parts cost up to 4x that -- which overran the
        # entry's own budget, so the caller's timeout fired first and the user
        # was told the check "could not be checked" instead of that the weights
        # are unavailable here. That is the wrong sentence for the one case
        # this probe exists for: a host that silently drops packets.
        pending = []
        for part, key in (
            ("encoder", model_type),
            ("decoder", f"{model_type}_decoder"),
        ):
            if key not in known:
                continue  # micro-sam does not publish this part for this type
            cached = os.path.exists(os.path.join(root, key))
            info: Dict[str, Any] = {
                "cached": cached,
                "reachable": None,
                "url": urls.get(key),
            }
            out["parts"][part] = info
            if not cached and info["url"]:
                pending.append((info, info["url"]))

        if pending:
            results = await asyncio.gather(
                *(
                    asyncio.to_thread(
                        _url_reachable, url, BASE_CHECKPOINT_PROBE_TIMEOUT_S
                    )
                    for _, url in pending
                ),
                return_exceptions=True,
            )
            for (info, _), res in zip(pending, results):
                # An exception here is not "reachable unknown": the caller is
                # deciding whether to admit a run, so it has to read as "not
                # obtainable from here".
                info["reachable"] = res is True
        return out

    async def ensure_base_checkpoint(
        self,
        model_type: str,
        resolve_mirror_urls=None,
        open_url=None,
    ) -> Dict[str, Any]:
        """Make this site able to produce *model_type*'s base weights.

        UPSTREAM FIRST, AND THAT IS A REQUIREMENT RATHER THAN A PREFERENCE.
        The mirror is only consulted for a part that is neither already cached
        nor reachable upstream, so on a site that can reach EBI this method
        performs no download and installs nothing -- it reduces to the probe it
        already ran. A fallback that fetched from the mirror whenever the
        mirror was closer would quietly make our copy the source of truth for
        every site, and the next zoo-side correction would never arrive.

        Returns ``{"model_type", "source", "parts", "installed",
        "unavailable", "error"}``. ``source`` is "cache", "upstream",
        "mirror" or "unavailable". ``unavailable`` names the parts no route
        could supply, so the ADMISSION CHECK refuses with a reason instead of
        a run discovering it minutes later under the GPU lock.

        It REPORTS rather than raises for a missing route, and RAISES for a
        file that failed the fence. Those are different: "I cannot get this"
        is a verdict the caller turns into a refusal, while "I got this and it
        is not what it claims to be" must stop everything, because the next
        thing that happens otherwise is training on unknown weights.

        `resolve_mirror_urls` and `open_url` are injectable so the
        upstream-first rule and the fence can be exercised with no network.
        Production passes neither.
        """
        import base_checkpoint_mirror as bcm

        out: Dict[str, Any] = {
            "model_type": model_type,
            "source": "unavailable",
            "parts": {},
            "installed": [],
            "unavailable": [],
            "error": None,
        }
        status = await self.base_checkpoint_status(model_type)
        out["parts"] = status.get("parts") or {}
        if status.get("probe_error"):
            # A probe that could not run is not evidence that a fallback is
            # needed, and downloading on that basis would make a transient
            # local failure rewrite the cache.
            out["error"] = f"probe failed: {status['probe_error']}"
            return out

        try:
            from micro_sam.util import models as _models

            root = str(_models().path)
        except Exception as exc:
            out["error"] = "%s: %s" % (type(exc).__name__, exc)
            return out

        need = []
        for part, info in out["parts"].items():
            name = model_type if part == "encoder" else f"{model_type}_decoder"
            if info.get("cached"):
                continue
            if info.get("reachable"):
                continue  # upstream will serve it; micro-sam fetches as before
            need.append(name)

        if not need:
            any_fetch = any(
                not i.get("cached") and i.get("reachable")
                for i in out["parts"].values()
            )
            out["source"] = "upstream" if any_fetch else "cache"
            return out

        # Only now is the mirror involved, and only for these names.
        mirrored = [n for n in need if n in bcm.MIRROR_CHECKPOINTS]
        out["unavailable"] = sorted(set(need) - set(mirrored))
        if not mirrored:
            return out

        resolve = resolve_mirror_urls or self._mirror_urls
        opener = open_url or _open_url_stream
        try:
            urls = await resolve(mirrored)
        except Exception as exc:
            out["error"] = "could not resolve mirror urls: %s: %s" % (
                type(exc).__name__,
                exc,
            )
            return out

        for name in mirrored:
            url = urls.get(name)
            if not url:
                out["unavailable"].append(name)
                continue
            # The fence runs inside `install_verified`; a MirrorVerificationError
            # is deliberately NOT caught here.
            record = await asyncio.to_thread(
                _install_from_url, bcm, root, name, url, opener
            )
            out["installed"].append(record)
            logger.warning(
                "model-finetune: base checkpoint '%s' taken from the MIRROR "
                "(upstream unreachable) and verified: sha256=%s bytes=%d",
                name,
                record["sha256"],
                record["bytes"],
            )

        out["unavailable"] = sorted(set(out["unavailable"]))
        out["source"] = (
            "mirror" if out["installed"] and not out["unavailable"] else out["source"]
        )
        if out["installed"] and out["unavailable"]:
            out["source"] = "unavailable"
        return out

    async def _mirror_urls(self, names) -> Dict[str, str]:
        """Signed URLs for *names* from the mirror, over an ANONYMOUS connection.

        No credential on purpose: the mirror is world-readable (verified with a
        ranged GET), and an app the worker ADOPTED rather than deployed holds
        no token -- which is exactly the state after the restart where a
        fallback matters most.
        """
        from hypha_rpc import connect_to_server

        import base_checkpoint_mirror as bcm

        server = await connect_to_server({"server_url": _HYPHA_SERVER_URL})
        try:
            am = await server.get_service("public/artifact-manager")
            out = {}
            for name in names:
                out[name] = await am.get_file(
                    artifact_id=bcm.MIRROR_ARTIFACT_ID,
                    file_path=name,
                    version=bcm.MIRROR_VERSION,
                )
            return out
        finally:
            try:
                await server.disconnect()
            except Exception:
                pass

    def _device(self) -> str:
        """Pick the compute device internally (device is not a public API param)."""
        if self._device_cached is None:
            try:
                import torch

                self._device_cached = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self._device_cached = "cpu"
        return self._device_cached

    # === resident model (one encoder + AIS segmenter at a time) ===

    @staticmethod
    def _to_image_format(array: np.ndarray) -> np.ndarray:
        """Coerce input to HxWx3 uint8 RGB, as the SAM image encoder expects."""
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        if array.ndim == 2:
            array = np.concatenate([array[..., None]] * 3, axis=-1)
        elif array.ndim == 3 and array.shape[0] == 3:
            array = np.transpose(array, [1, 2, 0])
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(
                f"Invalid input image of shape {array.shape}. Expected 2D (HxW) "
                "grayscale or 3-channel (HxWx3 / 3xHxW) RGB."
            )
        if array.dtype != np.uint8:
            array = array.astype("float32") - array.min(axis=(0, 1))
            denom = array.max(axis=(0, 1))
            denom[denom == 0] = 1.0
            array = (array / denom * 255).astype("uint8")
        return array

    @staticmethod
    def _nd(arr: np.ndarray) -> Dict[str, Any]:
        """Encode an array as hypha-rpc's ndarray wire-dict (numpy-neutral bytes).

        Survives the Ray hop to the CPU entry and its numpy-1.x proxy; the hypha
        client still decodes it to a real ndarray.
        """
        arr = np.ascontiguousarray(arr)
        return {
            "_rtype": "ndarray",
            "_rvalue": arr.tobytes(),
            "_rshape": list(arr.shape),
            "_rdtype": arr.dtype.name,
        }

    def _zoo_weights(self, model_type: str):
        """Fetch the pinned bioimage.io generalist (encoder + AIS decoder) into the
        local cache and return ``(encoder_path, decoder_path)``. Files are pinned by
        nickname+version so a zoo-side update never changes served weights without a
        release. Blocking — under the GPU lock; cached after first fetch."""
        import json

        import pooch

        nickname, version = ZOO_MODELS[model_type]
        arch = "vit_" + model_type.split("_")[1]
        cache = Path.home() / ".bioengine" / "micro_sam_zoo_cache" / model_type
        base = f"{_ZOO_BASE_URL}/{nickname}/{version}/files"
        enc = pooch.retrieve(
            f"{base}/{arch}.pt", known_hash=None, fname=f"{arch}.pt", path=cache
        )
        dec = pooch.retrieve(
            f"{base}/{arch}_decoder.pt",
            known_hash=None,
            fname=f"{arch}_decoder.pt",
            path=cache,
        )
        src = cache / "source.json"
        if not src.exists():
            src.write_text(
                json.dumps(
                    {"id": nickname, "version": version, "model_type": model_type}
                )
            )
        return enc, dec

    def _ensure_model(self, model_type: str, checkpoint: Optional[str] = None):
        """Load (predictor, AIS segmenter), reusing the resident pair when the
        (model_type, checkpoint) is unchanged. A generalist (no checkpoint) loads
        its pinned encoder + AIS decoder from the zoo cache; ``checkpoint`` serves a
        fine-tuned ``best.pt``. Blocking — call via ``asyncio.to_thread`` under the
        GPU lock.
        """
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter

        device = self._device()
        key = (model_type, checkpoint)
        if key != self._loaded_key:
            self._release_model()
            label = f"{model_type}" + (
                f" (finetuned: {checkpoint})" if checkpoint else ""
            )
            logger.info(f"🔄 Loading μSAM model '{label}' on {device}...")
            if checkpoint is None and model_type in ZOO_MODELS:
                # get_predictor_and_segmenter has no decoder_path, so pre-load the
                # two pinned files via get_sam_model and hand it the assembled state.
                from micro_sam.util import get_sam_model

                enc, dec = self._zoo_weights(model_type)
                predictor, state = get_sam_model(
                    model_type=model_type,
                    checkpoint_path=enc,
                    decoder_path=dec,
                    device=device,
                    return_state=True,
                )
                predictor, segmenter = get_predictor_and_segmenter(
                    model_type=model_type,
                    predictor=predictor,
                    state=state,
                    segmentation_mode=None,
                )
            else:
                predictor, segmenter = get_predictor_and_segmenter(
                    model_type=model_type,
                    checkpoint=checkpoint,
                    device=device,
                    # None → μSAM auto-selects AIS when the model ships a decoder.
                    segmentation_mode=None,
                )
            self._predictor = predictor
            self._segmenter = segmenter
            self._loaded_key = key
            self._loaded_model_type = model_type
            logger.info(f"✅ μSAM model '{label}' loaded.")
        return self._predictor, self._segmenter

    def _release_model(self) -> None:
        """Drop the resident model and free its VRAM (blocking; under gpu lock)."""
        self._predictor = None
        self._segmenter = None
        self._loaded_key = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _auto_segment(self, model_type, image, generate_kwargs, checkpoint):
        """μSAM automatic instance segmentation → int32 [H,W] mask. Blocking."""
        from micro_sam.automatic_segmentation import automatic_instance_segmentation

        predictor, segmenter = self._ensure_model(model_type, checkpoint)
        image = self._to_image_format(image)
        labels = automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=image,
            ndim=2,
            verbose=False,
            **generate_kwargs,
        )
        return np.asarray(labels).astype(np.int32)

    def _encode(self, model_type, image, checkpoint):
        """SAM image encoder → embedding payload for onnxruntime-web. Blocking."""
        predictor, _ = self._ensure_model(model_type, checkpoint)
        image = self._to_image_format(image)
        original_image_shape = image.shape[:2]
        sam = predictor.model
        sam_scale = sam.image_encoder.img_size / max(original_image_shape)
        predictor.reset_image()
        predictor.set_image(image)
        features = predictor.get_image_embedding().cpu().numpy()
        input_size = [int(x) for x in predictor.input_size]
        predictor.reset_image()
        return {
            "features": features.astype(np.float32),
            "original_image_shape": [int(x) for x in original_image_shape],
            "input_size": input_size,
            "sam_scale": float(sam_scale),
            "mask_threshold": float(sam.mask_threshold),
        }

    def _auto_segment_from_embedding(
        self,
        model_type,
        features,
        input_size,
        original_size,
        generate_kwargs,
        checkpoint,
    ):
        """Run the AIS decoder on a **precomputed** SAM embedding (no encoder pass).
        Blocking — call under the GPU lock. For the AIS generalist models; matches
        the image path, where ``generate`` returns the instance-label array directly.
        """
        _, segmenter = self._ensure_model(model_type, checkpoint)
        image_embeddings = {
            "features": np.asarray(features).astype(np.float32),
            "input_size": tuple(int(x) for x in input_size),
            "original_size": tuple(int(x) for x in original_size),
        }
        # image is unused when image_embeddings is provided (initialize only
        # encodes when embeddings are None).
        segmenter.initialize(image=None, image_embeddings=image_embeddings)
        instances = segmenter.generate(**generate_kwargs)
        return np.asarray(instances).astype(np.int32)

    def _export_onnx(self, model_type: str, quantize: bool) -> bytes:
        """Export the lightweight SAM prompt decoder to ONNX bytes. Blocking."""
        import tempfile
        import warnings

        import torch
        from onnxruntime import InferenceSession
        from onnxruntime.quantization import QuantType
        from onnxruntime.quantization.quantize import quantize_dynamic
        from segment_anything.utils.onnx import SamOnnxModel

        device = self._device()
        predictor, _ = self._ensure_model(model_type)
        sam = predictor.model
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / f"{model_type}.onnx"
            sam.to("cpu")
            try:
                onnx_model = SamOnnxModel(
                    model=sam,
                    return_single_mask=True,
                    use_stability_score=False,
                    return_extra_metrics=False,
                )
                embed_dim = sam.prompt_encoder.embed_dim
                embed_size = sam.prompt_encoder.image_embedding_size
                mask_input_size = [4 * x for x in embed_size]
                dummy_inputs = {
                    "image_embeddings": torch.randn(
                        1, embed_dim, *embed_size, dtype=torch.float
                    ),
                    "point_coords": torch.randint(
                        low=0, high=1024, size=(1, 5, 2), dtype=torch.float
                    ),
                    "point_labels": torch.randint(
                        low=0, high=4, size=(1, 5), dtype=torch.float
                    ),
                    "mask_input": torch.randn(
                        1, 1, *mask_input_size, dtype=torch.float
                    ),
                    "has_mask_input": torch.tensor([1], dtype=torch.float),
                    "orig_im_size": torch.tensor([1024, 1024], dtype=torch.float),
                }
                dynamic_axes = {
                    "point_coords": {1: "num_points"},
                    "point_labels": {1: "num_points"},
                }
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        str(onnx_path),
                        export_params=True,
                        verbose=False,
                        opset_version=17,
                        do_constant_folding=True,
                        input_names=list(dummy_inputs.keys()),
                        output_names=["masks", "iou_predictions", "low_res_masks"],
                        dynamic_axes=dynamic_axes,
                    )
                ort_inputs = {k: v.cpu().numpy() for k, v in dummy_inputs.items()}
                InferenceSession(
                    str(onnx_path), providers=["CPUExecutionProvider"]
                ).run(None, ort_inputs)
                if quantize:
                    q_path = onnx_path.with_stem(onnx_path.stem + "_quantized")
                    quantize_dynamic(
                        model_input=str(onnx_path),
                        model_output=str(q_path),
                        per_channel=False,
                        reduce_range=False,
                        weight_type=QuantType.QUInt8,
                    )
                    onnx_path = q_path
                return onnx_path.read_bytes()
            finally:
                sam.to(device)

    # === composition endpoints (called by EntryApp via the runtime handle) ===

    async def auto_segment(
        self,
        images: List[np.ndarray],
        model_type: str = "vit_l_lm",
        generate_kwargs: Optional[Dict[str, Any]] = None,
        checkpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """AIS masks for a batch of (already-resolved) images → wire-dict list."""
        generate_kwargs = generate_kwargs or {}
        results: List[Dict[str, Any]] = []
        async with self._gpu_lock:
            for image in images:
                labels = await asyncio.to_thread(
                    self._auto_segment, model_type, image, generate_kwargs, checkpoint
                )
                results.append({"output": self._nd(labels)})
        return results

    async def auto_segment_from_embedding(
        self,
        embeddings: List[Dict[str, Any]],
        model_type: str = "vit_l_lm",
        generate_kwargs: Optional[Dict[str, Any]] = None,
        checkpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """AIS masks for a batch of precomputed embeddings (each a dict with
        ``features``/``input_size``/``original_size``) → wire-dict list.
        """
        generate_kwargs = generate_kwargs or {}
        results: List[Dict[str, Any]] = []
        async with self._gpu_lock:
            for emb in embeddings:
                labels = await asyncio.to_thread(
                    self._auto_segment_from_embedding,
                    model_type,
                    emb["features"],
                    emb["input_size"],
                    emb["original_size"],
                    generate_kwargs,
                    checkpoint,
                )
                results.append({"output": self._nd(labels)})
        return results

    async def encode(
        self,
        image: np.ndarray,
        model_type: str = "vit_l_lm",
        checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Encoder embedding. ``features`` comes back as a wire-dict (the entry
        reconstructs/forwards it); ``input_size``/``original_image_shape``/
        ``sam_scale``/``mask_threshold`` describe it for the ONNX decoder and for
        running AIS on it later via ``infer(embeddings=...)``.
        """
        async with self._gpu_lock:
            payload = await asyncio.to_thread(
                self._encode, model_type, image, checkpoint
            )
        payload["features"] = self._nd(payload["features"])
        return payload

    async def export_onnx(
        self,
        model_type: str = "vit_l_lm",
        quantize: bool = True,
    ) -> bytes:
        cache_key = f"{model_type}:{quantize}"
        if cache_key in self._onnx_cache:
            return self._onnx_cache[cache_key]
        logger.info(f"📦 Exporting ONNX prompt decoder for '{model_type}'...")
        async with self._gpu_lock:
            onnx_bytes = await asyncio.to_thread(
                self._export_onnx, model_type, quantize
            )
        self._onnx_cache[cache_key] = onnx_bytes
        return onnx_bytes

    def _subprocess_env(self) -> Dict[str, str]:
        """Inherit the replica env (incl. Ray's CUDA_VISIBLE_DEVICES) minus secrets."""
        return {
            k: v
            for k, v in os.environ.items()
            if "TOKEN" not in k.upper() and "SECRET" not in k.upper()
        }

    def _run_train_subprocess(self, session_id: str):
        import training

        worker = str(Path(__file__).parent / "train_worker.py")
        return training.run_cancellable_subprocess(
            [sys.executable, worker, session_id],
            session_id,
            cwd=str(Path(__file__).parent),
            env=self._subprocess_env(),
        )

    def _run_export_subprocess(self, session_id: str, export_dir: str):
        worker = str(Path(__file__).parent / "export_worker.py")
        env = self._subprocess_env()
        env["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only export → no GPU contention
        return subprocess.run(
            [sys.executable, worker, session_id, export_dir],
            cwd=str(Path(__file__).parent),
            capture_output=True,
            text=True,
            env=env,
        )

    async def export_bioimageio(
        self, session_id: str, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build a standard combined SAM+decoder BioImage.IO package (interactive
        prompt head + AIS decoder in one checkpoint) from a trained session in a
        CPU subprocess (no GPU lock), via ``micro_sam.bioimageio.export_sam_model``.
        ``request`` carries the RDF metadata (name/description/authors/license/
        provenance). Returns the export result (package dir, zip path, file
        listing) for the entry to serve/upload.
        """
        import json

        import training

        export_dir = training.session_dir(session_id) / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "request.json").write_text(json.dumps(request))

        proc = await asyncio.to_thread(
            self._run_export_subprocess, session_id, str(export_dir)
        )
        res_path = export_dir / "export_result.json"
        if proc.returncode != 0 or not res_path.exists():
            raise RuntimeError(
                f"BioImage.IO export failed (rc={proc.returncode}): {(proc.stderr or '')[-3000:]}"
            )
        return json.loads(res_path.read_text())

    async def train(
        self, session_id: str, model_type: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fine-tune in a subprocess under the shared GPU lock (long-running).

        Evicts the resident inference model first so the subprocess owns the GPU;
        the subprocess (``train_worker.py``) writes COMPLETED/FAILED to the
        session status. Held for the run's duration, so concurrent inference
        queues behind it on this replica — at the effective target of 2 a
        training plus one inference asks for one replica, not two.
        """
        import training

        # THE REPLICA THAT WILL WRITE, asked here. The entry refuses up front
        # using a verdict from whichever replica its handle routed to, which
        # need not be this one if a backend ever has more than one replica --
        # a review named that, and my own design note had admitted the window
        # between a cached verdict and the write. This closes it: the check
        # runs in the process that is about to write, BEFORE the GPU lock, so
        # a refused run neither holds the GPU nor dies mid-epoch. One probe per
        # training run is nothing next to training.
        report = await self.store_status(max_age_s=0)
        if report.get("writable") is False:
            raise RuntimeError(
                "This GPU runtime cannot write the training-run store, so the "
                "run was not started: "
                + str(report.get("detail") or "the store refuses writes")
            )

        async with self._gpu_lock:
            await asyncio.to_thread(self._release_model)
            training.write_status(
                session_id,
                status="TRAINING",
                start_time=time.time(),
                n_epochs=params.get("n_epochs"),
                device=self._device(),
            )
            rc, tail, stopped = await asyncio.to_thread(
                self._run_train_subprocess, session_id
            )

        if stopped:
            training.write_status(
                session_id,
                status="STOPPED",
                message="stopped by user",
                end_time=time.time(),
                terminated_by="user_stop",
            )
        else:
            # Fallback: if the child died without writing a terminal status, record it.
            st = training.read_status(session_id)
            if st.get("status") not in ("COMPLETED", "FAILED", "STOPPED"):
                training.write_status(
                    session_id,
                    status="FAILED",
                    message=f"training subprocess exited rc={rc}: {tail}",
                    end_time=time.time(),
                    terminated_by="supervisor",
                )
        return {"session_id": session_id, "returncode": rc}
