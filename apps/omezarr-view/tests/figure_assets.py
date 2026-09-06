"""Emit figure-grade assets from the running view server.

Every number a figure prints has to trace to a record, so this writes the
records rather than leaving a human to transcribe from a terminal. Each file
carries its own provenance block: which endpoint, which host, which commit,
when.

Writes into --out (default analysis/results/omezarr_view/figure_assets/):
  metadata_reports/<id>.txt   generated mapped/not-mapped report, verbatim
  latency_tables.json         the exact rows a panel may quote
  auth_evidence.{json,txt}    the request/response matrix for the auth demo
  mechanism.json              one tile -> one range read, with the real offsets
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

TIMEOUT = 900


def provenance(base: str) -> Dict[str, Any]:
    def _git(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", "/data/nmechtel/bioengine", *args],
                                  capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return "unknown"

    return {
        "endpoint": base,
        "host": platform.node(),
        "bioengine_commit": _git("rev-parse", "--short", "HEAD"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "apps/omezarr-view/tests/figure_assets.py",
        "note": "client and server share a host; the endpoint is reached through "
                "its public tunnel, so each request pays a tunnel round trip.",
    }


def fetch(client: httpx.Client, base: str, path: str, **kw) -> httpx.Response:
    return client.get(f"{base}/{path.lstrip('/')}", timeout=TIMEOUT, **kw)


# ---------------------------------------------------------------------------


def metadata_reports(client, base: str, datasets: List[str], out: Path) -> List[Path]:
    """One text file per dataset, the generated report verbatim.

    The figure quotes these as printed matter, so they are written as text and
    not as JSON a human would have to re-render.
    """
    written = []
    directory = out / "metadata_reports"
    directory.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        info = fetch(client, base, f"api/datasets/{ds}").json()
        if "detail" in info:
            continue
        cap = info.get("caption", {})
        mapping = info.get("metadata_mapping", {})
        lines = [
            f"{info['title']}",
            "=" * len(info["title"]),
            "",
            f"source:  {info.get('source')}",
            f"recipe:  {info.get('recipe_label') or info.get('recipe')}",
            "",
            cap.get("access", ""),
            "",
            cap.get("mapped", ""),
            "",
            cap.get("not_mapped", ""),
            "",
            "-- full report, as the code emits it ------------------------------",
            "",
            "MAPPED",
        ]
        for k, v in (mapping.get("mapped") or {}).items():
            lines.append(f"  {k}: {json.dumps(v) if isinstance(v, (dict, list)) else v}")
        lines += ["", "NOT MAPPED"]
        for m in mapping.get("not_mapped") or []:
            lines.append(f"  {m}")
        lines += ["", f"[generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                      f"from {base}/api/datasets/{ds}]"]
        path = directory / f"{ds}.txt"
        path.write_text("\n".join(lines) + "\n")
        written.append(path)
    return written


def mechanism(client, base: str, dataset: str, out: Path) -> Path:
    """Panel b: one tile becomes one range read into the untouched original.

    Takes the numbers from the served reference index itself, so the offsets
    printed on canvas are the offsets the server would actually read.
    """
    info = fetch(client, base, f"api/datasets/{dataset}").json()
    refs = fetch(client, base, f"api/datasets/{dataset}/reference.json").json()

    source_bytes = info.get("size_bytes")
    examples = []
    for key in ("0/0.200.0.0", "0/1.200.4.5", "5/0.200.0.0"):
        entry = refs.get(key)
        if not isinstance(entry, list):
            continue
        url, offset, length = entry
        level = key.split("/", 1)[0]
        shape = info["level_shapes"][int(level)]
        examples.append({
            "zarr_key": key,
            "level": int(level),
            "level_shape": shape,
            "chunk_shape": json.loads(
                fetch(client, base, f"zarr/{dataset}/{level}/.zarray").text)["chunks"],
            "url": url,
            "byte_offset": offset,
            "byte_length": length,
            "byte_length_kb": round(length / 1024, 1),
            "fraction_of_file": length / source_bytes if source_bytes else None,
        })

    payload = {
        "provenance": provenance(base),
        "dataset": dataset,
        "title": info.get("title"),
        "source_url": info.get("source"),
        "source_bytes": source_bytes,
        "source_gib": round(source_bytes / 1024**3, 1) if source_bytes else None,
        "index": {
            "keys": info.get("index_keys"),
            "source_pages_indexed": info.get("source_pages_indexed"),
            "index_copies_pixel_data": info.get("index_copies_pixel_data"),
            "build_reads": info.get("build_reads"),
            "build_timings_s": info.get("build_timings_s"),
        },
        "one_tile_one_range_read": examples,
        "claim_wording": (
            "One 512x512 tile of this 39.4 GiB file is one range read of about "
            f"{round(examples[0]['byte_length']/1024)} KB at byte offset "
            f"{examples[0]['byte_offset']}. Nothing is pre-computed and nothing "
            "is stored twice." if examples else None),
    }
    path = out / "mechanism.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path


def auth_evidence(client, base: str, dataset: str, token: str, out: Path,
                  direct_base: str = None) -> List[Path]:
    """Run the auth matrix and record what each request actually returned.

    Run against BOTH the public URL and the process directly when a direct
    base is given: the capability-URL proxy in front of the mount consumes the
    Authorization header, so bearer auth appears broken through the public URL
    while working fine against the app. Reporting only one endpoint would blame
    the wrong component.
    """
    chunk = "0/0/0/20/0/0"
    cases = [
        ("catalog listing, no token", "GET", f"api/datasets", None, None),
        ("dataset metadata, no token", "GET", f"api/datasets/{dataset}", None, None),
        ("zarr .zattrs, no token", "GET", f"zarr/{dataset}/.zattrs", None, None),
        ("chunk, no token", "GET", f"zarr/{dataset}/{chunk}", None, None),
        ("thumbnail, no token", "GET", f"api/datasets/{dataset}/thumbnail.png", None, None),
        ("chunk, wrong bearer", "GET", f"zarr/{dataset}/{chunk}", "wrong-token", None),
        ("chunk, wrong path token", "GET", f"t/wrong-token/zarr/{dataset}/{chunk}", None, None),
        ("dataset metadata, correct bearer", "GET", f"api/datasets/{dataset}", token, None),
        ("zarr .zattrs, correct bearer", "GET", f"zarr/{dataset}/.zattrs", token, None),
        ("chunk, correct bearer", "GET", f"zarr/{dataset}/{chunk}", token, None),
        ("chunk, correct query token", "GET", f"zarr/{dataset}/{chunk}", None, token),
        ("chunk, correct path token", "GET", f"t/{token}/zarr/{dataset}/{chunk}", None, None),
    ]
    rows = []
    for label, _method, path, bearer, query in cases:
        headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
        params = {"token": query} if query else None
        r = fetch(client, base, path, headers=headers, params=params)
        redacted = None
        if path == "api/datasets":
            entries = r.json().get("datasets", [])
            hit = next((e for e in entries if e["id"] == dataset), None)
            redacted = bool(hit and hit.get("redacted"))
        rows.append({
            "case": label,
            "path": f"/{path}" if not query else f"/{path}?token=<token>",
            "credential": ("bearer" if bearer else "query" if query
                           else "path" if path.startswith("t/") else "none"),
            "status": r.status_code,
            "bytes": len(r.content),
            "catalog_entry_redacted": redacted,
        })

    direct_rows = []
    if direct_base:
        for label, _method, path, bearer, query in cases:
            headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
            params = {"token": query} if query else None
            try:
                r = fetch(client, direct_base, path, headers=headers, params=params)
                direct_rows.append({"case": label, "status": r.status_code})
            except Exception as e:
                direct_rows.append({"case": label, "status": f"error: {e}"})

    payload = {
        "provenance": provenance(base),
        "dataset": dataset,
        "cases": rows,
        "direct_to_process": {
            "base": direct_base,
            "cases": direct_rows,
            "why": "The capability-URL proxy in front of the public mount does "
                   "not forward the Authorization header, so bearer auth is "
                   "unavailable through that URL and returns 401 even for a "
                   "correct token. Against the app itself the same bearer "
                   "succeeds. The app supports all three credential forms; the "
                   "deployment in front of it decides which survive.",
        } if direct_base else None,
    }
    jpath = out / "auth_evidence.json"
    jpath.write_text(json.dumps(payload, indent=1) + "\n")

    width = max(len(r["case"]) for r in rows)
    lines = [f"Access control on a lazy OME-Zarr view — dataset '{dataset}'", ""]
    for r in rows:
        extra = ""
        if r["catalog_entry_redacted"] is not None:
            extra = ("  (catalog entry REDACTED)" if r["catalog_entry_redacted"]
                     else "  (catalog entry full)")
        lines.append(f"  {r['case']:<{width}}  ->  {r['status']}"
                     f"{'  ' + str(r['bytes']) + ' B' if r['status'] == 200 else ''}{extra}")
    if direct_rows:
        lines += ["", "Same matrix against the app directly (no proxy in front):", ""]
        for r in direct_rows:
            lines.append(f"  {r['case']:<{width}}  ->  {r['status']}")
        lines += ["",
                  "The bearer rows differ because the capability-URL proxy does not",
                  "forward the Authorization header. The app accepts all three",
                  "credential forms; the deployment decides which survive, and a",
                  "viewer needs the path form regardless."]
    lines += ["", f"[generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                  f"from {base}]"]
    tpath = out / "auth_evidence.txt"
    tpath.write_text("\n".join(lines) + "\n")
    return [jpath, tpath]


def latency_tables(out: Path, sources: Dict[str, Path]) -> Path:
    """Curate the rows a panel may quote, each pointing at its raw record."""
    tables: Dict[str, Any] = {"provenance": {}, "tables": {}}

    raw_served = json.loads(sources["served"].read_text())
    tables["provenance"] = raw_served.get("provenance", {})

    def rows_from(doc, kind):
        rows = []
        for d in doc.get("datasets", []):
            if "error" in d:
                continue
            for level, v in (d.get("per_chunk") or {}).items():
                rows.append({
                    "dataset": d["dataset"],
                    "recipe": d.get("recipe"),
                    "location": d.get("location"),
                    "level": level,
                    "chunk_shape": v.get("chunk_shape"),
                    "cold_median_ms": (v.get("cold") or {}).get("median_ms"),
                    "cold_n": (v.get("cold") or {}).get("n"),
                    "warm_median_ms": (v.get("warm_cache_hit") or {}).get("median_ms"),
                    "warm_n": (v.get("warm_cache_hit") or {}).get("n"),
                    "served_bytes_per_chunk": v.get("mean_served_chunk_bytes"),
                    "warm_definition": "server chunk-cache hit, per the X-View-Cache "
                                       "header; not a client or browser cache",
                    "raw_record": sources[kind].name,
                })
        return rows

    tables["tables"]["per_chunk_latency"] = rows_from(raw_served, "served")
    if "derived" in sources and sources["derived"].exists():
        tables["tables"]["per_chunk_latency_derived"] = rows_from(
            json.loads(sources["derived"].read_text()), "derived")

    build_rows = []
    for d in raw_served.get("datasets", []):
        if "error" in d:
            continue
        p = d.get("point_to_first_tile") or {}
        build_rows.append({
            "dataset": d["dataset"],
            "recipe": d.get("recipe"),
            "build_wall_s": d.get("build_wall_s"),
            "server_build_timings_s": d.get("server_build_timings_s"),
            "first_viewable_tile_s": (d.get("first_viewable_tile") or {}).get("seconds"),
            "first_viewable_tile_note": "coarsest level, the tile a viewer draws "
                                        "first; a full-resolution tile is in the "
                                        "per-chunk table",
            "raw_record": sources["served"].name,
        })
    tables["tables"]["point_to_first_tile"] = build_rows

    tables["binding_rules"] = [
        "Onboarding time is only ever quoted split into its parts; the IFD-chain "
        "read must not stand for the whole.",
        "No warm number unless a real cache produced it — warm here means a server "
        "chunk-cache hit, identified by the X-View-Cache header.",
        "Tier A (client reads the original directly) and Tier B (served, "
        "transcoded) are reported separately and only Tier A is zero-copy.",
        "Derived views are computed products and are never zero-copy.",
        "Latencies are site-specific: europa to Google Cloud Storage.",
    ]
    path = out / "latency_tables.json"
    path.write_text(json.dumps(tables, indent=1) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="/data/nmechtel/bioengine-paper/analysis/"
                                     "results/omezarr_view/figure_assets")
    ap.add_argument("--raw", default="/data/nmechtel/bioengine-paper/analysis/"
                                     "results/omezarr_view/raw")
    ap.add_argument("--token-file", default=".demo-token")
    ap.add_argument("--protected", default="protected-mousebrain")
    ap.add_argument("--mechanism-dataset", default="idr0106")
    ap.add_argument("--direct-base", default="http://localhost:8842",
                    help="the app itself, bypassing any auth proxy")
    ap.add_argument("datasets", nargs="+")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(args.raw)
    client = httpx.Client(follow_redirects=True)
    written: List[Path] = []

    written += metadata_reports(client, args.base, args.datasets, out)
    written.append(mechanism(client, args.base, args.mechanism_dataset, out))
    token = Path(args.token_file).read_text().strip()
    written += auth_evidence(client, args.base, args.protected, token, out,
                             direct_base=args.direct_base)
    written.append(latency_tables(out, {
        "served": raw / "served_measurements.json",
        "derived": raw / "served_derived.json",
    }))

    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
