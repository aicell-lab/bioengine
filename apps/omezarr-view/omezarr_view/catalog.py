"""Discover the files a view server should expose, locally and remotely.

Three source kinds, matching the two halves of the claim:

  ``local_dir``      every readable image under a directory — the "point it at
                     your archive" case.
  ``remote_prefix``  an S3/GCS bucket prefix, listed if the bucket allows it.
  ``remote_files``   an explicit list of URLs, for buckets that permit reads
                     but not listing (the Viv demo bucket is one of these).

Views are built lazily. Indexing a deep pyramid costs a walk of its whole IFD
chain, so a catalog of many files must not do that at startup.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .derived import DerivedView
from .recipes import BaseView, ViewError, open_view

logger = logging.getLogger("omezarr-view")

# Extensions a recipe has a chance with. Not a support claim — open_view still
# has to succeed, and the catalog reports the ones that did not.
IMAGE_SUFFIXES = {
    ".tif", ".tiff", ".ome.tif", ".ome.tiff", ".btf",
    ".czi", ".lif", ".nd2", ".lsm", ".oib", ".oif", ".ims", ".dv",
}


@dataclass
class DatasetEntry:
    """A catalog entry. ``view`` stays None until something asks for pixels."""

    id: str
    source: str
    title: str
    location: str
    recipe: str = "auto"
    licence: Optional[str] = None
    attribution: Optional[str] = None
    source_page: Optional[str] = None
    size_bytes: Optional[int] = None
    token: Optional[str] = None
    view: Optional[BaseView] = None
    error: Optional[str] = None
    # Set for a derived entry: the base dataset id, this view's transform
    # config, and a resolver the Catalog injects so it can find the base.
    derived_from: Optional[str] = None
    derived_config: Optional[Dict[str, Any]] = None
    resolve_base: Optional[Any] = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def is_protected(self) -> bool:
        return self.token is not None

    def ensure_view(self) -> BaseView:
        """Build the view on first use; subsequent calls reuse it."""
        with self._lock:
            if self.view is not None:
                return self.view
            if self.error is not None:
                raise ViewError(self.error)
            try:
                if self.derived_from:
                    base = self.resolve_base(self.derived_from)
                    if base is None:
                        raise ViewError(
                            f"derived view '{self.id}' needs dataset "
                            f"'{self.derived_from}', which is not in the catalog")
                    self.view = DerivedView(self.id, base.ensure_view(), self.title,
                                            **(self.derived_config or {}))
                else:
                    self.view = open_view(self.id, self.source, self.title, self.recipe)
            except Exception as e:
                self.error = str(e)
                raise
            return self.view

    def summary(self) -> Dict[str, Any]:
        """Catalog-level description, cheap enough to call before indexing."""
        out: Dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "location": self.location,
            "derived_from": self.derived_from,
            "licence": self.licence,
            "attribution": self.attribution,
            "source_page": self.source_page,
            "size_bytes": self.size_bytes,
            "protected": self.is_protected,
            "indexed": self.view is not None,
            "error": self.error,
        }
        if self.view is not None:
            out.update(self.view.report())
            out["protected"] = self.is_protected
        return out


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "dataset"


def _is_image(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(s) for s in IMAGE_SUFFIXES)


def _from_local_dir(spec: Dict[str, Any]) -> List[DatasetEntry]:
    root = Path(spec["path"]).expanduser().resolve()
    if not root.is_dir():
        logger.warning("local_dir %s does not exist; skipping", root)
        return []
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not _is_image(path.name):
            continue
        entries.append(DatasetEntry(
            id=_slug(path.stem),
            source=str(path),
            title=path.stem,
            location="local",
            recipe=spec.get("recipe", "auto"),
            licence=spec.get("licence"),
            attribution=spec.get("attribution"),
            source_page=spec.get("source_page"),
            size_bytes=path.stat().st_size,
            token=spec.get("token"),
        ))
    logger.info("local_dir %s -> %d file(s)", root, len(entries))
    return entries


def _from_remote_prefix(spec: Dict[str, Any]) -> List[DatasetEntry]:
    """List an S3/GCS prefix. Many public buckets allow reads but not listing."""
    import fsspec

    url = spec["url"].rstrip("/")
    protocol = url.split("://")[0] if "://" in url else "s3"
    try:
        fs = fsspec.filesystem(protocol, anon=spec.get("anonymous", True))
        names = fs.find(url.split("://", 1)[-1] if protocol in ("s3", "gs") else url)
    except Exception as e:
        logger.warning("remote_prefix %s not listable (%s); "
                       "use remote_files with explicit URLs instead", url, e)
        return []
    entries = []
    for name in names:
        if not _is_image(name):
            continue
        full = f"{protocol}://{name}" if protocol in ("s3", "gs") else name
        entries.append(DatasetEntry(
            id=_slug(Path(name).stem),
            source=full,
            title=Path(name).stem,
            location="remote",
            recipe=spec.get("recipe", "auto"),
            licence=spec.get("licence"),
            attribution=spec.get("attribution"),
            source_page=spec.get("source_page"),
            token=spec.get("token"),
        ))
    logger.info("remote_prefix %s -> %d file(s)", url, len(entries))
    return entries


def _remote_size(url: str) -> Optional[int]:
    import httpx

    try:
        r = httpx.head(url, follow_redirects=True, timeout=20)
        length = r.headers.get("content-length")
        return int(length) if length else None
    except Exception:
        return None


def _resolve_token(item: Dict[str, Any], spec: Dict[str, Any]) -> tuple[Optional[str], bool]:
    """Return (token, ok). A named-but-unset token env var fails closed."""
    env_name = item.get("token_env", spec.get("token_env"))
    if env_name:
        value = os.environ.get(env_name)
        if not value:
            return None, False
        return value, True
    return item.get("token", spec.get("token")), True


def _from_files(spec: Dict[str, Any]) -> List[DatasetEntry]:
    """Explicit entries — local paths or URLs, one per configured file."""
    base = spec.get("base", "").rstrip("/")
    entries = []
    for item in spec["files"]:
        item = {"url": item} if isinstance(item, str) else dict(item)
        url = item.get("url") or item["path"]
        if base and "://" not in url:
            url = f"{base}/{url}"
        remote = "://" in url
        if not remote:
            path = Path(url).expanduser().resolve()
            if not path.is_file():
                logger.warning("file %s does not exist; skipping", path)
                continue
            url = str(path)
        token, ok = _resolve_token(item, spec)
        if not ok:
            logger.warning(
                "%s names token_env %r but it is unset; skipping rather than "
                "serving it unprotected", url,
                item.get("token_env", spec.get("token_env")))
            continue
        entries.append(DatasetEntry(
            id=item.get("id") or _slug(url.rsplit("/", 1)[-1].split(".")[0]),
            source=url,
            title=item.get("title") or url.rsplit("/", 1)[-1],
            location="remote" if remote else "local",
            recipe=item.get("recipe", spec.get("recipe", "auto")),
            licence=item.get("licence", spec.get("licence")),
            attribution=item.get("attribution", spec.get("attribution")),
            source_page=item.get("source_page", spec.get("source_page")),
            size_bytes=(item.get("size_bytes")
                        or (_remote_size(url) if remote
                            else Path(url).stat().st_size)),
            token=token,
        ))
    logger.info("files -> %d entr(ies)", len(entries))
    return entries


def _from_derived(spec: Dict[str, Any]) -> List[DatasetEntry]:
    """Computed views over other catalog entries."""
    entries = []
    for item in spec["views"]:
        item = dict(item)
        base_id = item.pop("from")
        config = {k: item.pop(k) for k in
                  ("target_spacing_um", "target_scale", "channels", "dtype",
                   "normalise") if k in item}
        token, ok = _resolve_token(item, spec)
        if not ok:
            logger.warning("derived view %s names an unset token_env; skipping",
                           item.get("id"))
            continue
        entries.append(DatasetEntry(
            id=item.get("id") or f"{base_id}-derived",
            # Left blank until the view is built, when the report fills in the
            # underlying file; the card already names the base dataset.
            source="",
            title=item.get("title") or f"{base_id} (derived)",
            location="computed",
            licence=item.get("licence", spec.get("licence")),
            attribution=item.get("attribution", spec.get("attribution")),
            source_page=item.get("source_page", spec.get("source_page")),
            token=token,
            derived_from=base_id,
            derived_config=config,
        ))
    logger.info("derived -> %d view(s)", len(entries))
    return entries


_LOADERS = {
    "local_dir": _from_local_dir,
    "remote_prefix": _from_remote_prefix,
    "remote_files": _from_files,
    "files": _from_files,
    "derived": _from_derived,
}


class Catalog:
    """The set of datasets a server exposes, keyed by id."""

    def __init__(self, entries: List[DatasetEntry], title: str = "OME-Zarr views"):
        self.title = title
        self.entries: Dict[str, DatasetEntry] = {}
        for entry in entries:
            eid = entry.id
            n = 2
            while eid in self.entries:
                eid, n = f"{entry.id}-{n}", n + 1
            entry.id = eid
            entry.resolve_base = self.get
            self.entries[eid] = entry

    @classmethod
    def from_config(cls, path: str | Path) -> "Catalog":
        config = yaml.safe_load(Path(path).read_text()) or {}
        entries: List[DatasetEntry] = []
        for spec in config.get("sources", []):
            kind = spec.get("type")
            loader = _LOADERS.get(kind)
            if loader is None:
                logger.warning("unknown source type %r; skipping", kind)
                continue
            entries.extend(loader(spec))
        return cls(entries, title=config.get("title", "OME-Zarr views"))

    def get(self, dataset_id: str) -> Optional[DatasetEntry]:
        return self.entries.get(dataset_id)

    def summaries(self) -> List[Dict[str, Any]]:
        return [e.summary() for e in self.entries.values()]
