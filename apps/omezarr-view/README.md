# omezarr-view — serve existing images as OME-Zarr, without converting them

Point this app at a folder of images or a list of remote URLs. It serves each
one as an OME-Zarr endpoint that Vizarr, `ome-zarr-py`, `zarr`, or anything
else speaking NGFF can read. Nothing is converted, nothing is copied, and the
original files are never modified.

The claim it demonstrates: the **storage** format and the **access** format are
separate problems. Your archive can stay CZI and OME-TIFF while the access
layer speaks OME-Zarr.

## Quick start

```bash
pip install -r requirements.txt
python fetch_demo_data.py            # public CC-BY files for the local half
./run_server.sh                      # http://localhost:8842
```

Open the catalog, press **Build view** on a dataset, then **Open in Vizarr** or
**Copy URL**.

Point it at your own data by editing `datasets.yaml`:

```yaml
sources:
  - type: local_dir            # every readable image under a folder
    path: /mnt/archive/imaging

  - type: remote_files         # explicit URLs (buckets often deny listing)
    files:
      - url: https://example.org/scan.ome.tif

  - type: remote_prefix        # listed, where the bucket allows it
    url: s3://my-bucket/imaging/
    anonymous: true

  - type: files                # access-controlled entries
    token_env: MY_TOKEN        # fails closed if the variable is unset
    files:
      - path: ./private/study.czi
```

Views are built on first request, not at startup — indexing a deep pyramid
means walking its whole IFD chain, which a catalog of many files must not do
eagerly.

## The two recipes

| | `ReferenceView` | `BioIOView` |
|---|---|---|
| source | tiled TIFF / OME-TIFF, local or remote | anything BioIO reads (CZI, LIF, ND2, …), **local only** |
| how | maps each Zarr chunk to a byte range in the original file | calls the reader for each chunk |
| chunk size | the file's own tiles (usually 512²) | one YX plane — what the reader reads at once |
| pyramid | the file's own levels | single-scale unless the file has levels |
| server in the data path | optional — see below | always |

`open_view(..., recipe="auto")` tries the reference index first and falls back
to BioIO.

### Two ways to consume a reference view

The reference index is a genuine zero-copy artefact: `GET
/api/datasets/{id}/reference.json` hands the client a map of byte ranges, and
the client then reads the original file directly with `fsspec`'s reference
filesystem — this server is not involved at all. That path needs `imagecodecs`
client-side, because the chunks are still in the source's codec (LZW, Deflate).

The `/zarr/{id}/…` endpoint instead transcodes each chunk to zlib on the way
out, because **no browser Zarr implementation has an LZW codec**. That path
works everywhere but puts the server in the data path. Both avoid conversion;
only the first is literally zero-copy.

## Endpoints

| route | purpose |
|---|---|
| `GET /` | catalog page |
| `GET /api/datasets` | catalog as JSON (protected entries redacted) |
| `GET /api/datasets/{id}` | one dataset; builds the view if needed |
| `GET /api/datasets/{id}/reference.json` | byte-offset index, where the recipe has one |
| `GET /api/datasets/{id}/thumbnail.png` | RGB composite of a coarse level |
| `GET /zarr/{id}/{key}` | the OME-Zarr store |
| `GET /t/{token}/zarr/{id}/{key}` | same, with the token in the path |
| `GET /health` | status and chunk-cache stats |

## Access control

Give a dataset a `token` (or `token_env`) and every route refuses without it:
the metadata, the chunks, the thumbnail, and the catalog entry, which is
reduced to a stub — image dimensions, pixel sizes and channel names are
themselves sensitive.

Tokens travel three ways: `Authorization: Bearer …`, `?token=…`, or a
`/t/{token}/zarr/…` path prefix. **Use the path form for viewers.** A Zarr
client builds chunk URLs by appending the key to the store root, so a
`?token=` query ends up before the key and every chunk 404s.

## Metadata: what gets mapped

Each view reports its own mapping — `metadata_mapping.mapped` and
`.not_mapped` on the dataset endpoint, and the "What this view maps, and what
it does not" panel on each card. This is generated from what the source
actually declared, so it is the honest answer rather than a promise.

Typically mapped: dimensions and axis order, physical pixel sizes with units,
channel names, dtype, pyramid levels and their true scale factors.

Typically **not** mapped: channel colours and display windows (absent from most
acquisition files — the view synthesises them, and says so), objective and
instrument metadata, stage position, plate/well context, ROIs.

## Known limits

- **`bioio-czi` cannot read remote URLs.** It rejects any non-local filesystem
  outright. Remote CZI needs a FUSE mount or a download.
- **Bucket listing is frequently denied** even where reads are allowed; use
  `remote_files` with explicit URLs there.
- **A lazy view is not free.** Cold chunk latency is dominated by the round
  trip to object storage — on the measured setup, decode was ~1.6% of it. See
  `bioengine-paper/analysis/results/omezarr_view/RESULTS.md` for numbers with
  provenance.
- Zarr v2 only (`zarr>=2.18,<3`), which is what the tifffile reference format
  and current NGFF 0.4 consumers use.

## Tests

```bash
python tests/measure_served.py --base <url> <dataset> …     # latency, cold vs warm
xvfb-run -a python tests/vizarr_screenshot.py --base <url> <dataset> …
```

`vizarr_screenshot.py` needs `xvfb-run`: Playwright's headless Chromium has no
WebGL, so deck.gl fails at init and Vizarr renders nothing even when the data
arrives correctly. It records every network request Vizarr made alongside the
screenshot, because a black canvas and a working render are not distinguishable
from the image alone.

## Derived views — computed, not re-addressed

A raw view re-addresses the original's own pixels. A **derived view** computes
new ones and serves them the same lazy way: rescaled to a target physical
spacing, channel-selected, dtype-normalised — a "training view" shaped to what
a model wants to be fed.

```yaml
  - type: derived
    views:
      - id: my-training-view
        from: my-raw-dataset       # any dataset id above
        target_spacing_um: 10.0    # or target_scale: 0.5
        channels: [DAPI, GFP]      # names or indices; omit to keep all
        dtype: uint8               # uint8 | uint16 | float32
        normalise: percentile      # percentile | none
```

**The honesty line.** The original file is still never migrated or rewritten,
so "no conversion" still holds. But these pixels are *computed*, so a derived
view is **never zero-copy**, and it says so itself: every one reports its
`transform_chain`, sets `computed_product: true`, and carries a caption stating
that its output is a computed product.

What it does under the hood, and what that costs:

- **It reads the coarsest source level that still holds the detail requested**,
  per output level. A 10 µm/px view of a 3.22 µm/px source reads pyramid level
  1, not level 0 — four times fewer bytes.
- **Level 0 is exactly the target you asked for.** Coarser levels are added
  only when the target is big enough that a viewer could not otherwise draw it
  (with nothing coarser available, a viewer has to fetch every tile of the full
  grid before it can show anything — measured: 353 requests and a blank canvas
  before, 50 requests and a correct render after). Those extra levels are
  navigation aids, not part of the request.
- **A derived chunk costs several source chunks.** One output tile overlaps
  ~4 source tiles, each a separate read. On the remote demo a derived tile is
  roughly 2–4× the cold latency of a raw tile. A chunk-keyed LRU sits in front
  of the source (hit rates around 50% in normal viewing) — note it must be keyed
  on the *source chunk*, not the requested region, since no two output tiles ask
  for the same rectangle.
- **Only y/x are resampled.** z spacing carries over from the source unchanged,
  and the mapping report says so.
- **Normalisation bounds are measured, not declared** — the 1–99.8 percentiles
  the base view sampled. Absolute intensities are therefore not preserved, which
  the mapping report also states.
