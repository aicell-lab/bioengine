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
| `fedunet-site-a` | Europa worker | `dsb2018-fluo` | participant |
| `fedunet-site-b` | de.NBI worker, GPU node | `bbbc010-worms` | participant |
| `fedunet-pooled` | Europa worker | both | pooled-oracle control |

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
aggregate rather than to either local model. An arm is converged at round `r` if
no round in `(r, r+5]` beats the best value seen up to and including `r` by more
than 0.005 absolute; the reported round is the smallest such `r`, and an arm
only counts as converged within the run if `r+5` fits inside it. `None` means
the arm was still improving at the end — that is, the step budget was too short.
The rule lives in `CONVERGENCE` in `run_federated.py` and is copied verbatim
into each run's `provenance.json`, so it is fixed before the numbers exist.

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

Two public, permissively licensed datasets with obviously different object
shapes:

| Dataset | Objects | Source | Licence |
|---|---|---|---|
| `dsb2018-fluo` | fluorescence cell nuclei — many small round objects | BBBC038v1 (2018 Data Science Bowl), fluorescence subset as repackaged by StarDist | CC0 |
| `bbbc010-worms` | *C. elegans* — few long thin objects | BBBC010 v1/v2, Broad Bioimage Benchmark Collection | CC0 |

Both are bright objects on a dark background, and every image is percentile
(1 / 99.8) normalised per image, so the domain difference the model has to
bridge is **object shape**, not intensity or contrast. A generalisation failure
therefore cannot be waved away as a contrast artefact.

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

`dsb2018-fluo` is currently fetched via the StarDist repackaging of BBBC038,
which is one hop from the primary source. Swapping to BBBC038 directly is
available and is a change to one URL and one loader in `datasets.py`; it should
be done if this becomes a paper figure.

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
python run_federated.py --seeds 0 1 2 --rounds 10 --steps 50
```

which writes `metrics.json`, `provenance.json` and `transport_audit.json` into
`../bioengine-paper/analysis/results/federated-unet-<run-id>/`.

This is a development app. It is deployed private
(`authorized_users: [nils.mech@gmail.com]`) and is not intended for general use.
