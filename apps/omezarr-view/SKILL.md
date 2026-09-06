---
name: serve-images-as-omezarr
description: Serve existing image files (OME-TIFF, CZI, and other BioIO formats) as lazy OME-Zarr views without converting them, and wire them to real consumers — Vizarr, Neuroglancer, an annotation UI, Python, and training loaders. Use when someone wants their existing images to be AI-ready or web-viewable without migrating the archive.
---

# Serve your existing images as OME-Zarr, without converting them

Storage format and access format are separate problems. An archive can stay CZI
and OME-TIFF while the access layer speaks OME-Zarr, because a **view** answers
Zarr requests by reading the original file on demand.

The reference implementation is `bioengine/apps/omezarr-view`. Read it before
writing a new one — everything below is already implemented and measured there,
including the failure modes, and most tasks are a config change rather than new
code.

## Decide the recipe first

| source | recipe | why |
|---|---|---|
| tiled TIFF / OME-TIFF, local **or remote** | **reference index** | each Zarr chunk maps to a byte range; nothing is decoded to build it, and a client can read the original directly |
| CZI, LIF, ND2, LSM, … | **BioIO reader** | no addressable chunk layout exists, so every read goes through the reader |
| anything, but reshaped for a consumer | **derived view** on top of either | computed per request: rescale, channel-select, normalise |

`open_view(id, source, recipe="auto")` tries the reference index first and falls
back to BioIO.

## The claim boundary — get this right or the work is worthless

- **"No conversion" holds for everything here.** The original file is never
  migrated, rewritten, or copied.
- **"Zero-copy" holds only for the reference index consumed directly** — the
  client takes `/api/datasets/{id}/reference.json` and range-reads the original
  itself, with no server in the data path.
- **The served HTTP endpoint is not zero-copy.** It transcodes each chunk,
  because no browser Zarr implementation has an LZW codec. Bytes pass through.
- **A derived view is never zero-copy.** Its pixels are computed. Label it
  `computed_product` and state its transform chain.

## Metadata: generate the report, never write it by hand

Each view emits `metadata_mapping.mapped` / `.not_mapped` and a prose `caption`.
Quote those verbatim; do not paraphrase into a sentence, because a paraphrase
drifts from what the code did. When this was first written as prose it
immediately exposed a false claim (see gotcha 6).

Usually mapped: dimensions and axis order, physical pixel sizes with units,
channel names, dtype, pyramid levels with true scale factors, acquisition date.
Usually **not** mapped: channel colours, display windows, objective/instrument,
stage position, plate/well context, ROIs.

## Consumers — all verified against this implementation

### Vizarr (viewing)

`https://hms-dbmi.github.io/vizarr/?source=<zarr_url>`. Works with no
special-casing. Reads the `omero` block, so channel names and contrast come out
right.

### Vizarr fork with annotation (oeway/vizarr)

<https://github.com/oeway/vizarr>, hosted at <https://oeway.github.io/vizarr/>,
API documented in that repo's `api.md`. Use it when you want annotation inside
Vizarr itself rather than a separate UI. Two access modes:

- **Hypha Core in an iframe** — `const vizarr = await api.getService("vizarr")`,
  then `vizarr.addImage({source})`, `vizarr.add_shapes(shapes, opts)`.
- **Standalone** — `window.annotationController` exposes the layers directly.

Annotation surface: `add_shapes(shapes, {name, shape_type, label, edge_color,
face_color, edge_width})` where `shape_type` is `polygon | path | rectangle`;
`get_layers()`, `get_layer(id)`, `remove_layer(id)`; and per layer
`get_features()` / `set_features(features)` in **GeoJSON**. Persist those
features to your own store — see "Annotation storage" below.

### Neuroglancer (viewing, second independent client)

`https://neuroglancer-demo.appspot.com/#!<url-encoded state JSON>` with
`{"layers":[{"type":"image","source":"zarr://<zarr_url>"}],"layout":"xy"}`.
Supports zarr v2, NGFF 0.4 multiscale, and the `blosc | gzip | zlib | zstd |
raw` compressors. It reads the NGFF axes and scales — the physical pixel sizes
show up in its UI, which makes it good evidence the mapping is standard.

**It does not read the OME `omero` block**, so 16-bit data opens near black.
Pass the measured window explicitly:
`layer.shaderControls = {normalized: {range: [start, end]}}`. Do not set
`crossSectionScale` unless you know the right value — the default framing fits,
and a wrong value renders uniform background grey.

### OpenLayers / Leaflet and custom annotation UIs (tiles)

Map libraries want RGB image tiles, not Zarr chunks. `omezarr_view/tiles.py`
renders them on demand from the same lazy view:

- `GET /api/datasets/{id}/tilegrid.json` — width, height, tile size,
  `resolutions[]` (coarsest first), the non-spatial axes and their sizes, and
  the channel list with colours and windows.
- `GET /tiles/{id}/{zoom}/{x}/{y}.png?z=&t=&channels=0,2`

Build an `ol.tilegrid.TileGrid` from `resolutions` and use a pixel projection
with `extent = [0, -height, width, 0]`; then a click at image `(x, y)` is map
`(x, -y)`, so annotations are stored in image pixel coordinates.

### Customising the viewer app (controls, annotation)

`frontend/annotate.html` is the worked example and the thing to copy. It shows
how to add:

- **draw tools** — `ol.interaction.Draw` for polygon, box
  (`Draw.createBox()`), and point, with a Pan mode that removes the interaction.
- **plane controls** — one slider per non-spatial axis, built from
  `tilegrid.index_axes` / `index_sizes`; changing one calls `source.refresh()`
  so tiles re-render for the new z or t.
- **channel controls** — pass `channels=` to the tile route; the tile grid hands
  you labels, colours and windows to build the UI from.
- **a feature list** with per-feature removal, wired to the vector source's
  `addfeature` / `removefeature` / `changefeature` events.

**Stamp the plane onto every shape before saving.** A polygon without its `z` is
meaningless in a volume, and nothing else will record it for you.

### Annotation storage

Annotations never go into the image — the untouched original is the whole
claim. Store them beside the catalog:

- `GET /api/datasets/{id}/annotations` → GeoJSON `FeatureCollection`
- `PUT /api/datasets/{id}/annotations` → validates `type == FeatureCollection`,
  writes `<annotations_dir>/<id>.geojson`

### Python

`zarr.open_group(zarr.storage.FSStore(zarr_url))` or
`ome_zarr.io.parse_url(zarr_url)` + `ome_zarr.reader.Reader`. Both work
unmodified. For the true zero-copy path, fetch `reference.json` and open it with
fsspec's reference filesystem — that needs `imagecodecs` client-side, because
the chunks are still in the source codec.

### Training loaders

Read crops straight from the view with zarr. Expect it to be slow per sample
when cold and put a prefetcher in front (a bounded queue plus a small thread
pool). Measure it; see "Honest measurement" below.

## Gotchas, all of which cost real debugging time here

1. **NGFF 0.4 wants `/` between chunk indices; tifffile's index uses `.`.**
   Serve only one spelling and the other client silently reads `fill_value`
   instead of pixels and reports success. Accept both, and declare
   `dimension_separator` in `.zarray`.
2. **A `?token=` query cannot serve a viewer.** Zarr clients append the chunk
   key to the store root, so the query lands before the key and every chunk
   404s. Use a path-scoped token: `/t/{token}/zarr/{id}/{key}`.
3. **No browser Zarr implementation has an LZW codec.** Transcode on the way
   out (zlib is the safe universal choice) and say the served tier is not
   zero-copy.
4. **Gating chunks is not gating the dataset.** Dimensions, pixel sizes and
   channel names are themselves sensitive; redact the catalog entry too.
5. **A capability-URL/auth proxy may eat the `Authorization` header**, so bearer
   auth 401s through the public URL while working against the app. Support all
   three credential forms and record which endpoint a result came from.
6. **OME-XML `AcquisitionDate` is a child element, not an attribute.** Reading
   it with `.get()` silently yields `None` and the view then claims the source
   declared no date.
7. **Display windows must be measured or a viewer opens on black.** 16-bit
   microscopy uses a few percent of the dtype range. Sample percentiles from a
   coarse level and report that the window is computed, not declared.
8. **Cache the source *chunk*, not the requested region.** No two output tiles
   ask for the same rectangle, so a region-keyed cache measures 0 hits. Keying
   on the chunk took cold latency from 5627 ms to 2473 ms at ~50% hit rate.
9. **A large single-scale derived view is unusable in a viewer** — with nothing
   coarser to draw it must fetch the entire grid first (353 requests, blank
   canvas). Give derived views their own pyramid; level 0 stays the requested
   target.
10. **Do not close a shared `httpx.Client` to recover from a dropped
    connection.** A sibling thread mid-request gets "Cannot send a request, as
    the client has been closed". Swap the reference and let it be collected.
11. **Prefer pooled HTTP/1.1 over HTTP/2 to object storage under concurrency.**
    HTTP/2 multiplexes every range read onto one connection, so one reset takes
    out all of them.
12. **Benchmark conditions leak through shared working sets.** Disjoint samples
    are not enough if they share an underlying unit — crops on different (y, x)
    but the same z plane hit the same source chunks, so whichever condition ran
    second measured the first one's cache. This turned a real 2.7x prefetch
    speedup into a reported 19x. Make the working sets disjoint at the level the
    cache is keyed on, and write that rule into the record so the wrong number
    cannot resurface from the artefact.
13. **Playwright's headless Chromium has no WebGL at all.** Vizarr and
    Neuroglancer render nothing. Run under `xvfb-run` with `headless=False` and
    swiftshader.

## Honest measurement

- Split point-to-first-tile into parts. The IFD-chain read dominates a deep
  pyramid and must not stand for the whole; it is also a one-time cost, since
  reopening from a saved index is fast.
- No warm number without a real cache, and say which cache. Identify hits from
  the server (`X-View-Cache`) rather than assuming from request order.
- Cold latency is mostly the round trip to object storage, not the view. Decode
  measured ~1.6% of it here. Decompose before attributing.
- Give each measured condition **disjoint working sets**, including disjoint z
  planes. Sharing a plane shares source chunks, and the second condition then
  measures the first one's cache — this inflated a prefetch speedup from 2.7x
  to a bogus 19x until it was fixed.
- A screenshot is not evidence on its own. Record the viewer's requests and
  their statuses, and count lit pixels **and** pixel spread — a uniform grey
  canvas passes a lit-pixel check.
- Report what a local copy costs to materialise beside its throughput. It is
  fast per sample only after paying the same read up front.

## Licences

Verify from the source repository's API, not a landing page, and grade the
evidence. A Zenodo record exposes `metadata.license.id`; IDR and HuBMAP have
only site-wide policies with no per-dataset field, which is weaker. Say which
kind you have, and exclude anything unverifiable from publication.
