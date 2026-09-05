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

One app, deployed three times:

| Instance | Where | Data | Role |
|---|---|---|---|
| `fedunet-site-a` | Europa worker | site A of the chosen split | participant |
| `fedunet-site-b` | de.NBI worker, GPU node | site B of the chosen split | participant |
| `fedunet-pooled` | Europa worker | both | pooled-oracle control |

`deploy.py --split` picks which pair of datasets the three instances hold; see
**Data** below.

Sites A and B physically hold disjoint domains and never load the other's data.
The pooled instance deliberately violates the premise — it holds both — and
exists only to give the federated arm an upper bound to be compared against.
Keeping it as a separate deployment rather than loading both datasets into site
A means the two participants' transport logs stay clean.

No Flower, no gRPC mesh, no persistent peer connections. A round is:

1. each site runs `train(steps=S)` on its own images;
2. each site calls `push_weights(...)` — a `torch.save` of its `state_dict`
   into a shared Hypha artifact;
3. the driver downloads both checkpoints, computes a sample-count-weighted
   average, uploads it as `round_rr/global.pt`;
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
The pooled arm sees twice as many distinct images per epoch, so matching epochs
would have handed it 2× the gradient updates and the comparison would have
measured compute rather than data access.

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
checkpoint is pushed to site A and site B in turn and scored locally against
that site's held-out test split. `metrics.json` records per-image Dice and IoU,
not only means, so any number in it can be independently recomputed.

Each dataset's split carries a `split_fingerprint` — a sha256 over the dataset
name, split seed, dataset length and the exact test indices. All four arms and
both sites must report the same fingerprint for a dataset, otherwise they were
not scored on the same images and the comparison is void. The driver checks this
before training starts.

## Data

All CC0. Two pairings are available, and `deploy.py --split` picks one:

| Split | Site A | Site B | What differs |
|---|---|---|---|
| `caricature` | `dsb2018-fluo` | `bbbc010-worms` | the object **class** |
| `acquisition` | `bbbc038-fluo` | `bbbc038-histo` | the imaging **modality** |

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
| `run_federated.py` | the four-arm driver; runs outside the clusters |

## Running it

Deploy the three instances (`hypha_token` is required — `entry.py` reads
`HYPHA_TOKEN` at `__init__`):

```python
await worker.deploy_app(
    artifact_id="bioimage-io/federated-unet",
    version="0.1.0",
    application_id="fedunet-site-a",
    application_kwargs={"FederatedUNetSite": {
        "site_name": "site-a-europa",
        "datasets": ["dsb2018-fluo"],
    }},
    hypha_token=BIOIMAGE_IO_TOKEN,
)
```

Then:

```bash
python deploy.py --version 0.2.0 --split acquisition
python run_federated.py --seeds 0 1 2 3 4 --rounds 60 --steps 50 --previews \
    --pre-registration ../../../bioengine-paper/analysis/results/<design>.md
```

which writes `metrics.json`, `provenance.json` and `transport_audit.json` into
`../bioengine-paper/analysis/results/federated-unet-<run-id>/`.

This is a development app. It is deployed private
(`authorized_users: [nils.mech@gmail.com]`) and is not intended for general use.
