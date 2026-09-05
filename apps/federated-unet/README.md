# Federated U-Net

A development app for testing whether BioEngine can carry federated training on
image data. Several sites train the same small U-Net on their own images, only
weights cross the site boundary, and the merged model is expected to segment
both domains — which neither single-site model can do.

The point is not the model. It is the transport: can a BioEngine deployment on
one cluster and a BioEngine deployment on another cluster train one model
together, with nothing but a `state_dict` ever moving between them, and can that
be *checked* rather than asserted.

## Design

One app, deployed once per participant plus once for the control. `deploy.py
--layout` picks a named layout; `LAYOUTS` in `deploy.py` is the definition, and
the default is `acquisition-4site`:

| Instance | Where | Data | Role |
|---|---|---|---|
| `fedunet-fluo-0` | Europa worker | `bbbc038-fluo@0/3` | participant |
| `fedunet-fluo-1` | Europa worker | `bbbc038-fluo@1/3` | participant |
| `fedunet-fluo-2` | de.NBI worker, GPU node | `bbbc038-fluo@2/3` | participant |
| `fedunet-histo` | de.NBI worker, GPU node | `bbbc038-histo` | participant |
| `fedunet-pooled` | Europa worker | `bbbc038-fluo@*/3` + `bbbc038-histo` | pooled-oracle control |

The two earlier two-site pairings are kept as the `caricature` and `acquisition`
layouts so either previous run can be rebuilt unchanged.

Participants physically hold disjoint images and never load another's data. The
pooled instance deliberately violates the premise — it holds the union — and
exists only to give the federated arm an upper bound to be compared against.
Keeping it as a separate deployment rather than loading everything into one
participant means the participants' transport logs stay clean.

### Dataset specs and sharding

A site's data is named by a spec, not just a dataset name:

| Spec | Meaning |
|---|---|
| `name` | the whole training pool of that dataset |
| `name@k/n` | slice *k* of *n* disjoint slices of the training pool |
| `name@*/n` | the union of all *n* slices — what the pooled oracle needs |

Sharding exists because there are only three non-overlapping domains available
but a consortium demonstration wants more participants than that. Without it,
two sites asking for the same dataset receive the *same images*, and the
federation would be duplicating one site rather than joining several.

The test block is cut where an unsharded site would put it and is identical for
every *n*, so all arms are still scored on the same held-out images and
**shard 0 of 1 reproduces the unsharded split exactly, indices included** — the
`split_fingerprint`s from the two-site runs are unchanged, and those numbers stay
comparable. Each site additionally reports a `shard_fingerprint` over its *train*
indices, and the driver refuses to start if two clients on one domain report the
same one.

### Where the instances fit

Placement is a capacity decision, not a scientific one, and it is measured rather
than assumed (`PLACEMENT` in `deploy.py`):

| Worker | Admission mechanism | Binding resource | Replicas of this app |
|---|---|---|---|
| Europa | `VRAM_MB` packing (single-machine head) | host RAM, 30 GiB | ⌊30 / 8⌋ = 3 |
| de.NBI | GPU **fraction** (Kubernetes, no `VRAM_MB`) | GPU fraction, 0.40 each | ⌊1 / 0.40⌋ = 2 |

On Europa the GPU is *not* what runs out: `VRAM_MB` packing would allow
⌊24576 / 6144⌋ = 4, but the Ray cluster's 30 GiB of memory is shared with other
apps and `memory_mb` is what caps it. Only one of de.NBI's three T4 nodes has
free GPU fraction; the other two are fully booked by whole-GPU apps.

Both ceilings are set by the declarations, not by what the app uses: a replica
measures 0.65 GiB of RAM and about 2.5 GB of device-wide VRAM. Neither
`gpu_memory_mb` nor `memory_mb` is an enforced limit — nothing stops a replica
from over-running and OOM-ing a co-tenant — which is why both are left
comfortably above the measurement rather than trimmed to it.

**Clients that share a worker share a physical GPU.** "One client per site" is
the framing, so this is disclosed rather than left to be inferred:
`provenance.json` records `co_located_clients` per run.

No Flower, no gRPC mesh, no persistent peer connections. A round is:

1. each site runs `train(steps=S)` on its own images;
2. each site calls `push_weights(...)` — a `torch.save` of its `state_dict`
   into a shared Hypha artifact;
3. the driver downloads every participant's checkpoint, computes a
   sample-count-weighted average, uploads it as `round_rr/global.pt`;
4. each site calls `pull_weights(...)` and continues from the aggregate.

The driver (`run_federated.py`) runs outside both clusters and never receives an
image. Aggregation is FedAvg (McMahan et al., 2017) over the full `state_dict`.

### Why GroupNorm and not BatchNorm

The U-Net uses GroupNorm throughout. BatchNorm carries running mean and variance
in the `state_dict`, so FedAvg would be averaging activation statistics
collected on two visually different domains, and any federated failure would be
uninterpretable — an artefact of normalisation statistics rather than of the
federation. GroupNorm has no running buffers, so the merge is a weighted mean of
learned parameters and nothing else.

### Step-matched compute

Every arm gets the same number of optimiser steps per model (`rounds × steps`).
The pooled arm sees every participant's images per epoch, so matching epochs
would have handed it a multiple of the gradient updates and the comparison would
have measured compute rather than data access.

**To lengthen a run, raise `--rounds`, not `--steps`.** `steps` is the number of
local optimiser steps between two merges, so it *is* the FedAvg synchronisation
frequency: doubling it to buy a longer run would silently double the local drift
the aggregate has to reconcile, changing the federated algorithm under the label
"train longer". Raising `rounds` leaves the merge cadence untouched and only
extends the run, which keeps a longer run comparable to a shorter one.

### Convergence criterion

Validation Dice is scored after every round on each arm's own val split — for
the federated arm after the merge and pull, so the curve belongs to the
aggregate rather than to either local model.

The rule is **block plateau, per dataset**: split the rounds into equal thirds;
the arm has converged on a dataset if the mean validation Dice over the final
third exceeds the mean over the middle third by less than 0.005. It is reported
per dataset and **never averaged across datasets**.

It replaces a first-stall window rule that was pre-registered for the
`20260905-115455` run and **failed there**, declaring convergence at round 8 of
60 for arms whose best value arrived at round 50+. It fires on the first
transient stall in a noisy curve, and averaging the two datasets diluted the
harder domain by half. The superseded rule is still computed and reported so the
two runs stay comparable, but its verdict is retracted.

Both live in `CONVERGENCE` in `run_federated.py` and are copied verbatim into
each run's `provenance.json`, so they are fixed before the numbers exist.

`summary.json` also carries **deficit stability** — pooled − fedavg by block of
rounds — which is what makes a comparison budget-robust even when the absolute
values have not converged.

### Pre-registering predictions

`--pre-registration <file>` records that file's commit hash, commit timestamp
and sha256 in `provenance.json`, and **hard-fails if the file is uncommitted or
has unstaged edits**. A prediction stated in a message is a promise; a committed
hash is evidence, and the guard is what stops the recorded hash from describing
a different document than the one the run was designed against.

Omitting the flag is allowed but not silent: the driver warns, and
`provenance.json` records `"pre_registration": {"declared": false}` so an
exploratory run cannot later be mistaken for a pre-registered one.

### Federated evaluation

Checkpoints travel to the test data, never the reverse. Each arm's final
checkpoint is pushed to one scoring client per domain and scored locally against
that client's held-out test split. `metrics.json` records per-image Dice and IoU,
not only means, so any number in it can be independently recomputed.

One client per domain does the scoring, not all of them: clients sharing a domain
hold the identical test split, so scoring on each would only repeat the same
number under the same key. `provenance.json` records which client scored which
domain in `scored_on`.

Each dataset's split carries a `split_fingerprint` — a sha256 over the dataset
name, split seed, dataset length and the exact test indices. Every arm and every
client holding a dataset must report the same fingerprint for it, otherwise they
were not scored on the same images and the comparison is void. The driver checks
this — and the `shard_fingerprint` disjointness — before training starts.

## Data

All CC0. `deploy.py --layout` picks which one the instances hold:

| Layout | Participants | What differs |
|---|---|---|
| `caricature` | `dsb2018-fluo`, `bbbc010-worms` | the object **class** |
| `acquisition` | `bbbc038-fluo`, `bbbc038-histo` | the imaging **modality** |
| `acquisition-4site` | `bbbc038-fluo` in three disjoint shards, `bbbc038-histo` | modality, at 3:1 representation |

`dsb2018-fluo` and `bbbc038-fluo` are the **same primary source** — the BBBC038v1
fluorescence subset, one via StarDist's repackaging — so they can never be two
independent sites of one federation.

| Dataset | Objects | Source | Licence |
|---|---|---|---|
| `dsb2018-fluo` | fluorescence cell nuclei — many small round objects | BBBC038v1 (2018 Data Science Bowl), fluorescence subset as repackaged by StarDist | CC0 |
| `bbbc010-worms` | *C. elegans* — few long thin objects | BBBC010 v1/v2, Broad Bioimage Benchmark Collection | CC0 |
| `bbbc038-fluo` | cell nuclei by fluorescence (546 images) | BBBC038v1 `stage1_train`, direct | CC0 |
| `bbbc038-histo` | cell nuclei in H&E histology (108 images) | BBBC038v1 `stage1_train`, direct | CC0 |

### The caricature pairing

Both are bright objects on a dark background, and every image is percentile
(1 / 99.8) normalised per image, so the domain difference the model has to
bridge is **object shape**, not intensity or contrast. A generalisation failure
therefore cannot be waved away as a contrast artefact.

It is, deliberately, a caricature of non-IID: the classes are far enough apart
that a single-site model collapses out of domain, which makes federation look
very good. Useful as an existence proof, not as a utility bound.

### The acquisition pairing

One object class — nuclei — split by imaging modality inside a single archive,
so nothing but the modality differs. This is the hospital-consortium setting:
every site segments the same thing, but their scanners differ.

`metadata.xlsx` has a `stain_type` column, but only per project group and with no
mapping to an ImageId, so it cannot label images. The split is read off the
pixels instead, where for this collection it is a physical consequence of the
stain rather than a proxy for it: saturation is exactly zero up to the 80th
percentile, all 108 coloured images have a bright background, and only 2 of 670
images fall anywhere in the ambiguous band. The 16 greyscale bright-background
**brightfield** images are excluded — a third modality, the only 1024×1024 images
in the set, and a foreground fraction 7× below the other two.

Each modality is reduced to its own physically correct signal: fluorescence is
emission, so intensity is the signal; histology is absorbance, so **optical
density** is, being linear in stain concentration. This is load-bearing rather
than cosmetic. Raw greyscale polarity is *inverted* between the two — objects are
brighter than background in 100% of fluorescence images and 0% of histology ones
— so putting both through one intensity pipeline would turn an acquisition shift
into a sign flip, and rebuild the caricature by accident.

### Why worms and not leaves, and not the Cellpose training set

The original suggestion was "cells vs leaves" on the Cellpose data. Two changes:

- **The Cellpose training set was not used.** Its licence terms need checking
  before anything derived from it can go into a figure, and there was no reason
  to take that risk when a CC0 alternative gives the same effect. Both datasets
  here are CC0.
- **Leaves became worms.** There is no freely licensed instance-segmented leaf
  dataset that fits — but the property that mattered was "an obviously different
  object class so a small model shows the effect fast". Long thin worms against
  small round nuclei delivers that at CC0.

`dsb2018-fluo` is fetched via the StarDist repackaging of BBBC038, one hop from
the primary source. The `bbbc038-*` datasets read `stage1_train.zip` from the
Broad directly, so the acquisition split has no intermediary.

## Verifying that no image left a site

`TransportLog` in `checkpoints.py` records every byte moved in either direction:
direction, path, size, sha256 and a content `kind`. `get_transport_log()`
exposes it as a service method, so the audit is a call anyone can make. The dump
includes `only_weights_left_site`, which is computed as
`all(kind == "model_weights" for outbound entries)` — a check over the log
rather than a claim in prose.

Structurally there is also no code path out: `push_weights` is the only method
that writes anything to the artifact, and all it can serialise is a
`state_dict`. Images enter each replica by direct download from the public
source and stay there.

A full audit dump is written to `transport_audit.json` alongside the run
metrics.

## Files

| File | What |
|---|---|
| `entry.py` | the deployed service — one site |
| `unet.py` | shared architecture, signature check, FedAvg |
| `datasets.py` | dataset download, splits, fingerprints, licence metadata |
| `training.py` | step-based training loop and evaluation |
| `checkpoints.py` | artifact transport plus the audit log |
| `deploy.py` | the layouts, the measured placement, and the deploy calls |
| `run_federated.py` | the driver; runs outside the clusters, reads the layout |

## Running it

Deploy the instances of a layout (`hypha_token` is required — `entry.py` reads
`HYPHA_TOKEN` at `__init__`; `deploy.py` passes it):

```bash
python deploy.py --version 0.3.1 --layout acquisition-4site
python deploy.py --version 0.3.1 --layout acquisition-4site --only fluo-2 pooled
```

Then run the driver, which reads the same layout to learn who the clients are:

```bash
python run_federated.py --layout acquisition-4site \
    --seeds 0 1 2 3 4 --rounds 60 --steps 50 --previews \
    --pre-registration ../../../bioengine-paper/analysis/results/<design>.md
```

which writes `metrics.json`, `provenance.json` and `transport_audit.json` into
`../bioengine-paper/analysis/results/federated-unet-<run-id>/`.

This is a development app. It is deployed private
(`authorized_users: [nils.mech@gmail.com]`) and is not intended for general use.
