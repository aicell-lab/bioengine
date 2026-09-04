# Smart Microscopy Assistant

VLM-backed microscopy analyst. Define re-usable visual tests with a few positive and negative reference images, then run them against new images to get a **PASSED / FAILED / UNSURE** verdict. Or describe an image with a free-text instruction.

## Method

| | |
|---|---|
| Model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| License | Apache 2.0 |
| Engine | HuggingFace `transformers` 4.51.3 + `torch==2.5.1` |
| Precision | 4-bit NF4 (`bitsandbytes`, double-quantised, FP16 compute); vision tower and `lm_head` left in FP16 |
| Hardware | 1× NVIDIA A40-16C vGPU slice per replica (Ampere sm_86, 16 GB framebuffer, time-shared with co-tenants on the host A40) |
| Image budget | server-side downscale to ≤ `1280 × 28 × 28` pixels and ≤ 2048 longest side, with a hard reject above 200 MP |

### Why Qwen2.5-VL-7B-Instruct (NF4)

- Apache 2.0 license — usable in any deployment.
- 7B FP16 weights are ~15.4 GiB against a 16 GiB framebuffer, which leaves nothing for the CUDA context and activations. NF4 lands the same model at **9.02 GiB measured under load** — 2 GiB *less* than the 3B at FP16 used, with 7 GiB of headroom.
- Dynamic input resolution via Qwen's processor — works with arbitrary microscopy frame sizes once the server-side downscale step has bounded them.
- Returns coherent multi-bullet QC reports (focus, illumination uniformity, object count, contamination, etc.) on real fluorescence-microscopy frames; see Operating characteristics below for measured behaviour.

The vision tower is excluded from quantisation. It is a small share of the weights, and 4-bit quantising the encoder is what costs fine visual detail — the one thing this app exists to read.

### Why not AWQ (the earlier 7B attempt)

`bitsandbytes` replaced AWQ as the route to a 7B here. AWQ was tried first and no kernel stack served Qwen2.5-VL on this cluster:

- **vLLM 0.10.x** — V0 multimodal input-prep raises `InputProcessingError: list index out of range` on every prompt shape; V1 engine refuses to initialise from a Ray Serve actor thread.
- **vLLM 0.9.x** — model-registry subprocess fails to inspect `Qwen2_5_VLForConditionalGeneration` and swallows the underlying error.
- **vLLM 0.7.x** — transitively pins an older Ray than the host pod, which Ray refuses to load.
- **autoawq Triton kernel** — bundled `awq_gemm_triton` doesn't compile against the Triton shipped with current torch.
- **autoawq-kernels CUDA path** — `awq_ext.gemm_forward_cuda` raises `expected scalar type Int but found Half` on the `lm_head` linear for `Qwen2.5-VL-7B-Instruct-AWQ` (known upstream issue, fix sits in autoawq 0.2.8 which itself caps `transformers <= 4.47.1` — older than the 4.49 minimum Qwen2.5-VL needs).

The app shipped on 3B FP16 from 0.11.x to 0.13.2 for that reason. 0.14.0 moves to the 7B on NF4, which needs no kernel beyond `bitsandbytes` and the stock `transformers` integration.

## Image and instruction limits

| Limit | Value | Where enforced |
|---|---|---|
| Image file size | 25 MB | Streamed download; raises if exceeded mid-stream |
| Image pixel count | ≤ 1280 × 28 × 28 (≈ 1.0 MP) AND longest side ≤ 2048 | Server-side downscale in `_download_image` (PIL `Image.resize` with LANCZOS) |
| Image hard reject | 200 MP (≈ 14000 × 14000) | Raises `ValueError` before downscale |
| Instruction length | 4000 characters | Server-side `ValueError` (also mirrored in the browser UI which clips the textarea) |
| Generation timeout | 180 s per `_generate_with_images` call | `asyncio.wait_for` |

If the server downscales, the response carries `downscaled_from: [W, H]` and a `downscale_note` so callers can see whether the verdict was rendered on the original or a resized version.

## Accepted image references

An `image_ref` is either a Hypha artifact reference `<workspace>/<alias>:<file_path>`, or an `https://` URL whose host is the app's own Hypha server (`hypha.aicell.io`). Anything else — another host, plain `http://`, a URL carrying userinfo — is refused before any request is made, and a same-origin URL that redirects off-origin is refused at the hop.

The reason is that the app fetches with its own network position, so a caller who can name an arbitrary host gets to probe what the replica can route to. No credential is involved; reachability is. Both URL shapes the app actually consumes are same-origin anyway — `artifact-manager.get_file` and the browser UI's `s3-storage.get_file` both return `https://hypha.aicell.io/s3/...` — so the restriction costs nothing in practice. To use an image held elsewhere, upload it to a Hypha artifact or to `s3-storage` first and pass that reference.

## Whose credential reads an artifact ref

**The app holds none.** Since 0.13.0 the replica has no Hypha token of its own: its standing connection is anonymous, so an artifact ref reaches public artifacts and nothing else.

To read a **private** artifact, pass your own Hypha token as the `token` parameter of `inspect`, `submit_inspect` or `create_visual_test`. It is used for that one call, on a connection opened and closed inside the call, and is never cached, logged, or written onto the job record that `get_inspect_status` returns.

A ref that fails to resolve reports why: a missing artifact and a missing file inside one raise `FileNotFoundError`, and only a genuine access denial raises `PermissionError` asking for a `token`. Before 0.13.2 all three read as "pass a token", which sent callers hunting for credentials they already had.

This replaces the arrangement up to 0.12.1, where a ref naming a workspace the caller's token scope showed as readable was resolved with the *app's* token. That was safe as written — the scope came from Hypha server-side and a client could not widen it — but it made the app a standing deputy: a bug anywhere in that check, or a later widening of the app token's own access, would have turned into a read of files the caller never had. Holding no credential removes the class of failure rather than guarding it. The cost is explicit: a private ref that used to resolve silently now needs a `token`.

## Who can call this app

`authorized_users` in `manifest.yaml` is a named allowlist, not `"*"`. BioEngine matches the caller's Hypha `id` or `email` exactly (`bioengine/utils/permissions.py`), and the app builder additionally admits the deploying identity and the worker's admin users. Anonymous callers are rejected. To widen access, add the address to `authorized_users` and redeploy — note that a Hypha API token carries no email in its JWT payload, but the server enriches the identity from the parent account, so an email entry matches API tokens and browser logins alike.

## Two modes

The app has two operating modes that share the same `inspect()` entry point:

1. **Describe mode** (`instruction` only) — single image + free-text question, get a natural-language description.
2. **Few-shot verdict mode** (`visual_test_name` set) — a previously-defined "visual test" supplies positive/negative reference images and a criterion. The model is prompted with the references first, then the new image, and asked to return one of `passed`, `failed`, or `unsure` with a short reason.

Define a visual test once with `create_visual_test(...)`, then call `inspect(image_ref, visual_test_name=...)` as many times as you like — references stay on the replica's disk.

## API

### `inspect(image_ref, instruction=None, visual_test_name=None, max_new_tokens=512, token=None) -> dict`

| Parameter | Type | Description |
|---|---|---|
| `image_ref` | `str` | Either an `https://hypha.aicell.io/...` URL issued by that server (an artifact or `s3-storage` presigned link) **or** a Hypha artifact reference `<workspace>/<alias>:<file_path>` (e.g. `ws-user-github\|49943582/qc-samples:images/frame_001.tif`). URLs on any other host are refused — see [Accepted image references](#accepted-image-references). |
| `instruction` | `str?` | Free-text instruction. Required when `visual_test_name` is not given. Optional when it is — then it overrides the visual test's stored description. Max 4000 chars. |
| `visual_test_name` | `str?` | Name of a visual test created via `create_visual_test(...)`. Switches into few-shot verdict mode. |
| `max_new_tokens` | `int` | Response token budget. Default 512, range 1–1024. |
| `token` | `str?` | Your own Hypha token, used only to read a **private** artifact ref. Omit for public artifacts and for `https://` URLs. See [Whose credential reads an artifact ref](#whose-credential-reads-an-artifact-ref). |

**Returns (describe mode):**

```json
{
  "mode": "describe",
  "description": "- Focus: in focus, clear outlines …",
  "image_size": [1024, 1024],
  "source_url": "https://hypha.aicell.io/s3/…",
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "tokens_generated": 66,
  "generation_time_s": 2.55,
  "tokens_per_second": 25.9,
  "processing_time_s": 3.0
}
```

**Returns (few-shot verdict mode):**

```json
{
  "mode": "few-shot",
  "visual_test_name": "focus-quality",
  "visual_test_description": "Sharp cell outlines, no motion blur, distinct staining patterns.",
  "verdict": "passed",
  "reason": "Cell outlines are crisp and the staining is well-resolved.",
  "description": "VERDICT: passed\nREASON: Cell outlines are crisp …",
  "n_positive_examples": 3,
  "n_negative_examples": 3,
  "image_size": [1024, 1024],
  "source_url": "https://hypha.aicell.io/s3/…",
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "tokens_generated": 24,
  "generation_time_s": 0.92,
  "tokens_per_second": 26.1,
  "processing_time_s": 1.3
}
```

`verdict` is one of `"passed"`, `"failed"`, or `"unsure"`. The model returns `unsure` when the visible evidence is genuinely ambiguous or insufficient; the parser also defaults to `unsure` when the output doesn't follow the `VERDICT: …` schema (the raw text is always in `description`).

`downscaled_from` and `downscale_note` may be present in either mode when the server resized the inspected image.

### Few-shot quality notes

The model handles **specific, visually-grounded criteria** ("at least 5 distinct cells", "any saturated pixels", "vertical motion blur") considerably better than **coarse class differences** ("good vs. bad image"). Two patterns observed on the live deployment:

- A criterion phrased as a measurable property (cell count, focus sharpness on a defined region, presence of a specific artefact) generally returns a verdict aligned with the actual content.
- A criterion phrased as broad quality vs. anti-quality, with references that span very different visual styles, can produce verdicts that echo the positive-class reason regardless of the new image.

If a visual test isn't discriminating well: tighten the criteria (they go into the prompt verbatim) to spell out *what to look for*; consider asking a more specific question via the `instruction` override at inspect time.

**Do not treat a verdict as a measurement.** Both models were run against three labelled microscopy defect axes (defocus, histology air bubbles, tissue folds) with pre-registered criterion wordings, 3B and 7B on the same images through the same instrument:

| | 3B FP16 (0.13.2) | 7B NF4 (0.14.0) |
|---|---|---|
| Defocus, balanced accuracy, text-only | 0.609 | 0.797 |
| Defocus, verdict flips when PASS/FAIL are swapped in the criterion | 0.109 | 0.016 |
| Bubbles, balanced accuracy, text-only | 0.500 (120/120 `passed`) | 0.508 (119/120 `passed`) |
| Bubbles, balanced accuracy, 5+5 references | 0.500 | 0.542 |
| Bubbles, verdict flips when PASS/FAIL are swapped | 0.000 | 0.908 |
| Median serial latency | 1.5 s | 1.9 s |

Neither model beat a trivial per-image pixel statistic on any axis (0.98 defocus, 0.84 bubbles). The 7B is the better model on the axis it can see and it is the only one of the two that changes its answer when the criterion's PASS/FAIL assignment is swapped — but on bubbles it flips the verdict while its stated reason describes the same image state 118 times out of 120, so the movement is in mapping the instruction, not in seeing the defect. The app is an assistant for a human reading the reason text, not an unattended gate.

### Visual-test management

| Method | Description |
|---|---|
| `create_visual_test(name, pass_criterion, fail_criterion, positive_image_refs, negative_image_refs, is_public=False, token=None)` | Define or replace one of *your* visual tests. References follow the same rule as `inspect` — a `hypha.aicell.io` URL or a Hypha artifact ref, with `token` needed only for a private one. Images are downloaded, downscaled (capped at ~512×512), and persisted under your own directory. |
| `list_visual_tests()` | Your own tests plus every public test. Each record carries `created_by`, `is_public`, and `owned_by_you`. |
| `get_visual_test(name)` | One record you can see — your own test of that name wins over a public one. |
| `delete_visual_test(name)` | Remove one of your own tests and its cached images. Owner-only, even for public tests. |

Limits enforced by `create_visual_test`:

- `0 ≤ N_positive ≤ 5`, `0 ≤ N_negative ≤ 5`. Omit both for a text-only test; more than five examples eat the model's context budget without improving few-shot quality.
- `name` must match `^[a-z0-9][a-z0-9-]{0,49}$`. Two users can hold the same name without colliding; re-using your own overwrites.
- `pass_criterion` and `fail_criterion` ≤ 800 characters each.
- Each reference image is fetched once and stored at ≤ 512×512 to keep the prompt's image-token cost bounded.

#### Ownership and visibility

Every test is owned by the identity Hypha reports for the caller — email where the token carries one, otherwise the user id. Tests are private by default; `is_public=True` makes a test listable and usable by everyone, but never deletable or overwritable by anyone but its owner. Source image refs (`positive_refs`/`negative_refs`) are returned **only to the owner**: a caller-supplied `https://` ref may be a presigned URL, which is a bearer capability for a file the reader has no permission on, so handing it to every reader of a public test would route around the permission check that accepted it.

**Changed in 0.11.0 — anonymous callers no longer share one library.** Before 0.11.0, every unauthenticated caller was keyed to the single owner string `"anonymous"`, so any two anonymous callers were the same principal: each could read, overwrite, and delete the others' private tests, and every record reported `owned_by_you: True` to a non-creator. Anonymous callers are now keyed on the throwaway workspace Hypha allocates per connection, so each is a distinct principal — and an anonymous caller's tests are no longer reachable after that connection closes. Since 0.12.0 the allowlist rejects anonymous callers outright, so this keying no longer has a live path; it is kept because records written before 0.12.0 still carry those owner keys.

That last clause is a real capability removal, so it is worth being explicit about whether anyone was relying on it. Our reading is that no one was — but the evidence is bounded, and the bound matters more than the conclusion.

The load-bearing leg is the browser UI, which has never had an anonymous path: `connectToServer` is called from exactly one place, inside a function that requires a token, and a visitor without one gets a login gate. No UI user could have created an anonymous test. That is *not* the same as the pool being unreachable. A scripted RPC caller is an ordinary user of a served app rather than an exotic bypass, and is precisely the class that could have populated the pool; private anonymous records are not enumerable from outside the replica, so we cannot rule that out. Corroborating but weaker: the pool held no public tests at the time of the change — though a listing that returns a single record is not a survey of anything.

On that evidence we read the shared pool as speculative rather than load-bearing. If you were relying on it, the symptom will be anonymous tests appearing to vanish between connections, and this paragraph is the explanation.

**Fixed in 0.11.1 — records written before 0.7.0 are reachable by their creators again.** Until 0.7.0 the owner key was a `caller_user_id` the *client* passed in, and clients passed their Hypha workspace name (`ws-user-<id>`). 0.7.0 replaced that with the caller's email or user id taken from the server-side context. Both are better keys, but nothing migrated the records already on disk, so a test written under the old scheme was keyed under a string the new scheme can never compute — its own creator was told *"owned by another user; only its creator can delete it"*, and on a public test the record's source fields were redacted from the person who wrote them. Ownership checks now accept the caller's personal workspace as a legacy key in addition to the current one, and re-creating a test under the same name clears the old copy so the library converges on the current scheme.

The legacy key is deliberately narrow, in three ways.

**It is derived, not read.** The alias is computed as `ws-user-<account>` from the authenticated user, *not* taken from `context["ws"]`. That distinction is the whole security property: `context["ws"]` is the workspace the caller's **token** was minted for, not a fixed property of the caller — one token of this maintainer's carries `bioimage-io`. A token minted against someone else's personal workspace would carry that workspace, so keying on it would make legacy ownership grantable by adding a member. Deriving it means a caller can only ever name their own personal workspace. The account is `user.parent` where present, falling back to `user.id`, because for API-token callers `user.id` is a per-token synthetic account — two tokens of the same human report different ids and the same `parent`.

**It expires.** A legacy key is honoured only for records created before the 0.7.0 cutover; anything newer must match the current key. Read-time compatibility is a migration you have chosen never to finish, so it needs a stated end — otherwise every future key change appends an entry, nothing is ever removed, and in a few years nobody can tell which entries still matter.

**Anonymous callers get no legacy key at all.** Every pre-0.11.0 anonymous record was written under the bare string `"anonymous"`, which identifies nobody, so honouring it would hand each anonymous caller the entire old pool and undo the fix above. Those records stay orphaned, which is the correct outcome for data whose owner was never established.

One residual is scoped rather than removed: pre-0.7.0 owner keys were self-asserted by the client, so a record written then under someone else's workspace name is inherited by that person rather than by its author. The cutover does not prevent this, because such a record is by definition older than the cutover. It yields data rather than a capability, and it requires having done so deliberately in June 2026.

That residual stays bounded only while one invariant holds, so it is worth stating for anyone extending the app: `created_by` and `created_at` are written in exactly one place — the record literal in `create_visual_test` — and both are derived server-side, the first from `_owner_from_context(context)` and the second from the clock. No caller can influence either. A future write path that reconstructs a record from a caller-supplied blob (an import endpoint being the obvious candidate) must re-derive both rather than carry them across; carrying them restores the pre-0.7.0 ability to assert your own ownership key, and the cutover cannot catch it, because a forged record can simply claim a pre-cutover timestamp. `_owner_from_context` is the write-side counterpart to `_owns` on the read side: one function decides who you are, one decides what you own, and every path should go through them.

Persistence: visual tests live under `$HOME/visual_tests/` on the replica's filesystem. On the KTH BioEngine worker (and any worker whose `apps_workdir` resolves to PVC-backed storage), that directory is mounted from a persistent volume — the per-app working directory is the same path across actor restarts, pod rolls, and full stop+deploy cycles. Empirically verified by creating a visual test, performing `stop_app → deploy_app` (fresh deploy, `recovered_app=False`), and seeing the test still present on the new actor.

A visual test is therefore **persistent within a worker** but **not portable across workers** (each worker pod has its own PVC). To share a library across deployments, re-call `create_visual_test()` against the source image refs.

### `ping() -> dict`

Liveness probe returning `{status, model, uptime_s}`.

### `get_model_info() -> dict`

Describes the served model and the input/output contract:

```json
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "task": "vision-language",
  "engine": "huggingface-transformers",
  "dtype": "float16",
  "device": "cuda:0",
  "max_image_bytes": 26214400,
  "max_instruction_chars": 4000,
  "max_pixels": 1003520,
  "max_long_side": 2048,
  "hard_reject_pixels": 209715200,
  "min_examples_per_class": 1,
  "max_examples_per_class": 5,
  "max_visual_test_name_chars": 50,
  "max_visual_test_desc_chars": 800,
  "verdicts": ["passed", "failed", "unsure"],
  "license": "Qwen2.5-VL Apache 2.0 weights"
}
```

## Operating characteristics (measured on KTH A40-16C vGPU)

10 back-to-back `inspect()` calls, same 512×512 HPA RGB image, identical 200-char instruction, `max_new_tokens=192`, 66 generated tokens each:

| Metric | min | median | mean | max | std | spread |
|---|---:|---:|---:|---:|---:|---:|
| tok/s | 20.4 | 24.5 | 24.1 | 26.3 | 1.8 | 5.9 |
| e2e seconds | 2.79 | 3.13 | 3.30 | 4.26 | — | 1.47 |

The vGPU profile time-shares the underlying A40 with other tenants; the tight ~2 tok/s standard deviation indicates contention is currently mild.

VRAM is not exposed directly via the Hypha service. The model load reports ~6 GB for weights at FP16; with KV cache, activations, and the vision encoder, the steady-state working set is well within the 16 GB framebuffer.

## Browser UI

`frontend/index.html` ships as the artifact's `frontend_entry`, so once the artifact is uploaded it is reachable at:

```
https://hypha.aicell.io/{workspace}/view/{artifact-id}/
```

The page has two modes:

- **Analyze** — drag in one or more images, pick a saved visual test (or type a free-text instruction), and hit *Run analysis*. Each image is uploaded and inspected sequentially; the result row shows a colored verdict chip (Passed / Failed / Unsure / Described / Error), the reason, and an expandable details panel with tok/s, timing, and the raw model output.
- **Define visual test** — name, criterion, plus positive + negative example galleries (1–5 each). On save, examples are uploaded to a scratch artifact, presigned with the caller's session, and handed to `create_visual_test()`; the worker downloads them once, downscales, and caches under `$HOME/visual_tests/<name>/`.

An info button in the top bar opens a popover with a short description of the app (prose held in the `APP_INFO` object) followed by the served model details (pulled from `get_model_info`). The activity log is hidden behind an expandable "Activity log" panel at the bottom.

Failures reach the user one of two ways. Something the user asked for opens a modal with the full trace; a background failure — the boot auto-connect, a list refresh — is parked instead behind a *Details* button, which appears on the status line and, because the status lines sit inside the signed-in views, also on the login gate. The activity log escapes everything it renders: it carries filenames, server error text, and other users' public test names, none of which are trusted.

The page expects the service ID via `?ws_service_id=<full-id>&server=<hypha-url>` URL params; without them it falls back to the short artifact form `bioimage-io/smart-microscopy-assistant`.

## Usage example (Python)

```python
from hypha_rpc import connect_to_server

server = await connect_to_server({
    "server_url": "https://hypha.aicell.io",
    "token": HYPHA_TOKEN,
})
worker  = await server.get_service("bioimage-io/bioengine-worker-kth-...:bioengine-worker")
status  = await worker.get_app_status(["smart-microscopy-assistant"])
ws_sid  = status["smart-microscopy-assistant"]["service_ids"]["websocket_service_id"]
qc      = await server.get_service(ws_sid)

# Define a visual test once
await qc.create_visual_test(
    name="has-cells",
    pass_criterion="Visible cellular structures with nuclei.",
    fail_criterion="Flat, empty, or uniform regions.",
    positive_image_refs=["my-workspace/qc-samples:cells_1.tif", "my-workspace/qc-samples:cells_2.tif"],
    negative_image_refs=["my-workspace/qc-samples:flat_1.png",  "my-workspace/qc-samples:flat_2.png"],
)

# Run it against any number of new images
result = await qc.inspect(
    image_ref="my-workspace/qc-samples:scan.tif",
    visual_test_name="has-cells",
)
print(result["verdict"], "—", result["reason"])
```

Both calls above assume `my-workspace/qc-samples` is publicly readable. If it is not, add `token=HYPHA_TOKEN` to each call — the app holds no credential of its own and will otherwise refuse the ref.
