"""
BioEngine Datasets Proxy Server - Privacy-preserved dataset management service.

Serves zarr datasets in-place from a local directory. A FastAPI app handles
all client-facing endpoints (dataset listing, file listing, and zarr chunk
serving) with no Hypha service registration needed.

Token authentication is delegated to the remote Hypha server on demand and
cached locally — the server itself requires no credentials at startup. Clients
authenticate with an ``Authorization: Bearer`` header; the legacy ``?token=``
query parameter is still accepted but deprecated.

Architecture:
- Dataset directories are scanned for manifest.yaml at startup
- A background asyncio task polls data_dir every 30 s and hot-reloads the
  in-memory registry when datasets are added, removed, or their manifest
  changes — no server restart needed
- Access control is defined per-dataset in manifest.yaml (authorized_users field)
- Token authentication is delegated to the remote Hypha server (cached locally)
- FastAPI exposes all endpoints under /public/services/bioengine-datasets/
- File bytes are served directly at /data/{dataset_id}/{path} with Range support
- Files can be uploaded via POST /save (public or private per-user)
"""

import asyncio
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
import yaml
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from hypha_rpc import connect_to_server
from uvicorn.logging import AccessFormatter

from bioengine import __version__
from bioengine.utils import (
    acquire_free_port,
    create_logger,
    get_internal_ip,
)
from bioengine.utils.permissions import check_permissions

AUTHENTICATION_SERVER_URL: str = "https://hypha.aicell.io"

logger = logging.getLogger("ProxyServer")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


_TOKEN_QUERY_RE = re.compile(r"([?&]token=)[^&\s]*")


class RedactingAccessFormatter(AccessFormatter):
    """Access formatter that scrubs ``?token=`` values from the request line.

    The query-param form is deprecated but still accepted, and the value is a
    full-power Hypha credential rather than a scoped, expiring URL signature —
    a zarr read would otherwise write it to the log once per chunk.
    """

    def formatMessage(self, record: logging.LogRecord) -> str:
        return _TOKEN_QUERY_RE.sub(r"\1<redacted>", super().formatMessage(record))


def get_log_config(log_file: Optional[Path] = None) -> Dict[str, Any]:
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s %(levelprefix)s %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "bioengine.datasets.proxy_server.RedactingAccessFormatter",
                "fmt": '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }
    if log_file:
        for handler in log_config["handlers"].values():
            handler["class"] = "logging.FileHandler"
            handler["filename"] = str(log_file)
            handler.pop("stream", None)
    return log_config


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _scan_dir_for_datasets(scan_dir: Path, datasets: Dict[str, dict]) -> None:
    """Scan one directory level for dataset subdirectories (must have manifest.yaml)."""
    for subdir in sorted(scan_dir.iterdir()):
        if not subdir.is_dir():
            continue
        manifest_file = subdir / "manifest.yaml"
        if not manifest_file.exists():
            continue
        try:
            manifest = yaml.safe_load(manifest_file.read_text())
        except Exception as e:
            logger.warning(f"Failed to parse {manifest_file}: {e}")
            continue

        dataset_id = manifest.get("id") or subdir.name
        authorized_users = manifest.get("authorized_users", [])
        if isinstance(authorized_users, str):
            authorized_users = [authorized_users]

        datasets[dataset_id] = {
            "manifest": manifest,
            "path": subdir,
            "authorized_users": authorized_users,
        }
        logger.debug(f"Loaded dataset '{dataset_id}' from {subdir}")


def load_datasets(data_dir: Path) -> Dict[str, dict]:
    """Scan data_dir for datasets (excludes data_dir/saved/ — user-saved files).

    Returns a dict keyed by dataset_id. Each value has: manifest (dict),
    path (Path), authorized_users (list).
    """
    if not data_dir.is_dir():
        raise ValueError(f"data_dir does not exist or is not a directory: {data_dir}")

    datasets = {}
    _scan_dir_for_datasets(data_dir, datasets)
    # Remove saved/ entries — they are managed via POST /save and GET /saved/,
    # not through the datasets catalog.
    datasets.pop("saved", None)
    return datasets


async def _watch_data_dir(
    data_dir: Path,
    datasets: Dict[str, dict],
    poll_interval: int = 30,
) -> None:
    """Poll data_dir every poll_interval seconds and hot-reload the datasets registry.

    Updates the shared ``datasets`` dict in-place so all running route handlers
    see changes immediately without a server restart.
    """
    while True:
        await asyncio.sleep(poll_interval)
        try:
            fresh = load_datasets(data_dir)

            added = set(fresh) - set(datasets)
            removed = set(datasets) - set(fresh)
            changed = {
                ds_id
                for ds_id in set(fresh) & set(datasets)
                if fresh[ds_id]["manifest"] != datasets[ds_id]["manifest"]
            }

            for ds_id in added:
                datasets[ds_id] = fresh[ds_id]
                logger.info(f"Dataset added: '{ds_id}' ({fresh[ds_id]['path']})")

            for ds_id in removed:
                del datasets[ds_id]
                logger.info(f"Dataset removed: '{ds_id}'")

            for ds_id in changed:
                datasets[ds_id] = fresh[ds_id]
                logger.info(f"Dataset reloaded (manifest changed): '{ds_id}'")

        except Exception as e:
            logger.warning(f"Data directory watch error: {e}")


# ---------------------------------------------------------------------------
# Token authentication
# ---------------------------------------------------------------------------


def resolve_token(authorization: Optional[str], token: Optional[str]) -> Optional[str]:
    """Take the token from the Authorization header, else the ?token= query param.

    The query param is deprecated — it leaks the credential into access logs and
    breaks zarr store URLs, which append the key path after the query string.
    """
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
    return token


async def parse_token(
    token: Optional[str],
    cached_user_info: Dict[str, dict],
) -> Dict[str, str]:
    """Validate a user token against the remote Hypha server (cached)."""
    global AUTHENTICATION_SERVER_URL

    if token is not None:
        cached = cached_user_info.get(token)
        if cached is not None and cached.get("expires_at", 0) > time.time():
            return cached

        try:
            async with connect_to_server(
                {"server_url": AUTHENTICATION_SERVER_URL, "token": token}
            ) as user_client:
                user_info = user_client.config.user
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

        cached_user_info[token] = user_info
        # Evict oldest entry when cache exceeds 1000 tokens
        if len(cached_user_info) > 1000:
            cached_user_info.pop(next(iter(cached_user_info)))
    else:
        user_info = {"id": "anonymous-user", "email": "no-email"}

    return user_info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _authenticate_user(
    token: Optional[str],
    cached_user_info: Dict[str, dict],
) -> tuple[str, str]:
    """Validate token and return (user_id, safe_uid).

    user_id  — raw Hypha user ID (e.g. "user:github|abc123")
    safe_uid — filesystem-safe version (special chars replaced with "_")

    Raises HTTPException 401 if token is missing or identity cannot be resolved.
    """
    if token is None:
        raise HTTPException(status_code=401, detail="Authentication token required")
    user_info = await parse_token(token, cached_user_info)
    user_id = user_info.get("id", "")
    if not user_id:
        raise HTTPException(status_code=401, detail="Could not determine user identity from token")
    safe_uid = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
    return user_id, safe_uid


def safe_join(path: str, base_dir: Path) -> Path:
    """Join a client-supplied relative path onto base_dir, rejecting traversal.

    Unlike ``_validate_path`` this deliberately does not resolve symlinks:
    dataset directories are curated by the data owner and legitimately symlink
    out to other mounts, which is the whole point of serving data in place.
    """
    if not path:
        raise HTTPException(status_code=400, detail="path must not be empty")
    parts = Path(path).parts
    if Path(path).is_absolute() or any(part == ".." for part in parts):
        raise HTTPException(status_code=400, detail="path traversal is not allowed")
    return base_dir / path


async def _serve_file_response(full_path: Path, request: Optional[Request]) -> Response:
    """Serve a file with HTTP Range request support."""
    file_size = full_path.stat().st_size
    range_header = request.headers.get("Range") if request else None

    if range_header:
        m = re.match(r"bytes=(\d*)-(\d*)", range_header)
        if m:
            start = int(m.group(1)) if m.group(1) else 0
            end = int(m.group(2)) if m.group(2) else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1

            def read_range():
                with open(full_path, "rb") as f:
                    f.seek(start)
                    return f.read(length)

            data = await asyncio.to_thread(read_range)
            return Response(
                content=data,
                status_code=206,
                media_type="application/octet-stream",
                headers={
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(length),
                },
            )

    return FileResponse(full_path, headers={"Accept-Ranges": "bytes"})


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def _build_app(
    data_dir: Path,
    datasets: Dict[str, dict],
    cached_user_info: Dict[str, dict],
    watch_interval: int = 30,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(
            _watch_data_dir(data_dir, datasets, poll_interval=watch_interval)
        )
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    app = FastAPI(
        title="BioEngine Datasets",
        description="Zarr dataset server with manifest-based access control",
        version=__version__,
        lifespan=lifespan,
    )

    # Browser-based Zarr viewers fetch chunks cross-origin. Access control is
    # enforced per request from the token, so the origin itself grants nothing.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["Authorization", "Range"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
    )

    @app.get("/health/liveness")
    async def liveness():
        return {"status": "ok"}

    @app.get("/ping")
    async def ping():
        return "pong"

    @app.get("/datasets")
    async def list_datasets_route():
        return {
            dataset_id: info["manifest"]
            for dataset_id, info in datasets.items()
        }

    @app.get("/datasets/{dataset_id}/files")
    async def list_files_route(
        dataset_id: str,
        dir_path: Optional[str] = None,
        token: Optional[str] = None,
        authorization: Optional[str] = Header(None),
    ):
        if dataset_id not in datasets:
            raise HTTPException(
                status_code=400,
                detail=f"ValueError: Dataset '{dataset_id}' does not exist",
            )

        try:
            user_info = await parse_token(
                resolve_token(authorization, token), cached_user_info
            )
            check_permissions(
                context={"user": user_info},
                authorized_users=datasets[dataset_id]["authorized_users"],
                resource_name=f"list files in dataset '{dataset_id}'",
            )
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))

        base_path = datasets[dataset_id]["path"]
        scan_path = safe_join(dir_path, base_path) if dir_path else base_path

        if not scan_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"ValueError: Path '{dir_path}' does not exist in dataset '{dataset_id}'",
            )

        files = []
        for root, dirs, filenames in os.walk(scan_path):
            dirs.sort()
            for fname in sorted(filenames):
                full = Path(root) / fname
                files.append(str(full.relative_to(base_path)))
        return files

    @app.get("/data/{dataset_id}/{path:path}")
    async def serve_file(
        dataset_id: str,
        path: str,
        token: Optional[str] = None,
        request: Request = None,
        authorization: Optional[str] = Header(None),
    ):
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

        try:
            user_info = await parse_token(
                resolve_token(authorization, token), cached_user_info
            )
            check_permissions(
                context={"user": user_info},
                authorized_users=datasets[dataset_id]["authorized_users"],
                resource_name=f"access '{path}' in dataset '{dataset_id}'",
            )
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))

        full_path = safe_join(path, datasets[dataset_id]["path"])
        if not full_path.exists() or not full_path.is_file():
            raise HTTPException(status_code=404, detail=f"'{path}' not found")

        return await _serve_file_response(full_path, request)

    def _validate_filename(filename: str) -> None:
        """Validate a flat filename (no path separators — used by POST /save)."""
        if not filename:
            raise HTTPException(status_code=400, detail="filename must not be empty")
        if "/" in filename or "\\" in filename or filename in (".", ".."):
            raise HTTPException(status_code=400, detail="filename must not contain path separators")

    def _validate_path(path: str, base_dir: Path) -> Path:
        """Validate a relative path (slashes allowed) and return the resolved absolute path.

        Two-layer traversal protection:
        1. Explicit rejection of any '..' component in the raw path string.
        2. resolve() + relative_to() check so that symlinks, absolute path
           components, and any other trickery that slips past layer 1 are
           caught by the filesystem itself.
        """
        if not path:
            raise HTTPException(status_code=400, detail="path must not be empty")
        if any(part == ".." for part in Path(path).parts):
            raise HTTPException(status_code=400, detail="path traversal is not allowed")
        full_path = (base_dir / path).resolve()
        try:
            full_path.relative_to(base_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="path traversal is not allowed")
        return full_path

    def _ensure_save_dir(
        save_dir: Path,
        dataset_id: str,
        authorized_users: List[str],
        display_name: str,
    ) -> None:
        """Create the save directory and manifest.yaml on first use."""
        save_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = save_dir / "manifest.yaml"
        if not manifest_path.exists():
            manifest = {
                "id": dataset_id,
                "name": display_name,
                "description": "Files saved via the BioEngine Datasets save API.",
                "authorized_users": authorized_users,
            }
            manifest_path.write_text(yaml.dump(manifest, default_flow_style=False))
            logger.info(f"Created save directory '{dataset_id}' at {save_dir}")

    @app.post("/save")
    async def save_file_route(
        request: Request,
        filename: str,
        public: bool = False,
        token: Optional[str] = None,
        authorization: Optional[str] = Header(None),
    ):
        """Save a file to the server.

        Public files go to data_dir/saved/public/ and cannot overwrite existing
        files. Private files go to data_dir/saved/{user_id}/ and allow overwrite.
        The user identity is derived from the Hypha token (GitHub-backed OAuth).

        Args:
            filename: Name of the file (no path separators).
            public: True → world-readable, no overwrite. False (default) → owner-only, overwrite allowed.
            token: Hypha authentication token (required).
        """
        _validate_filename(filename)
        user_id, safe_uid = await _authenticate_user(
            resolve_token(authorization, token), cached_user_info
        )

        if public:
            dataset_id = "saved-public"
            save_dir = data_dir / "saved" / "public"
            authorized_users: List[str] = ["*"]
            display_name = "Public Saved Files"
        else:
            dataset_id = f"saved-{safe_uid}"
            save_dir = data_dir / "saved" / safe_uid
            authorized_users = [user_id]
            display_name = f"Private files for {user_id}"

        _ensure_save_dir(save_dir, dataset_id, authorized_users, display_name)

        file_path = save_dir / filename
        if public and file_path.exists():
            raise HTTPException(
                status_code=409,
                detail=f"'{filename}' already exists in the public directory and cannot be overwritten",
            )

        content = await request.body()
        await asyncio.to_thread(file_path.write_bytes, content)
        logger.info(f"Saved '{filename}' to '{dataset_id}' ({len(content)} bytes) by '{user_id}'")

        return {
            "dataset_id": dataset_id,
            "filename": filename,
            "size": len(content),
            "public": public,
        }

    @app.get("/saved")
    async def list_saved_files(
        public: bool = False,
        dir_path: Optional[str] = None,
        token: Optional[str] = None,
        authorization: Optional[str] = Header(None),
    ):
        """List files in the public or private saved directory.

        Args:
            public: True to list the public directory. Default: False.
            dir_path: Optional subdirectory to list within the save directory.
            token: Hypha authentication token (required when public=False).
        """
        if public:
            save_dir = data_dir / "saved" / "public"
        else:
            _, safe_uid = await _authenticate_user(
                resolve_token(authorization, token), cached_user_info
            )
            save_dir = data_dir / "saved" / safe_uid

        if not save_dir.exists():
            return []

        scan_dir = _validate_path(dir_path, save_dir) if dir_path else save_dir
        if not scan_dir.exists() or not scan_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Directory '{dir_path}' not found")

        files = []
        for root, dirs, filenames in os.walk(scan_dir):
            dirs.sort()
            for fname in sorted(filenames):
                if fname == "manifest.yaml":
                    continue
                full = Path(root) / fname
                files.append(str(full.relative_to(save_dir)))
        return files

    @app.get("/saved/{filename:path}")
    async def get_saved_file(
        filename: str,
        public: bool = False,
        token: Optional[str] = None,
        request: Request = None,
        authorization: Optional[str] = Header(None),
    ):
        """Retrieve a previously saved file or file inside a subdirectory.

        Routes to data_dir/saved/public/{filename} when public=True (no token needed),
        or data_dir/saved/{user_id}/{filename} when public=False (token required).
        filename may contain slashes to address files in subdirectories.
        The user identity is derived from the Hypha token (GitHub-backed OAuth).

        Args:
            filename: Relative path to the file (slashes allowed, traversal blocked).
            public: True to fetch from the public directory. Default: False.
            token: Hypha authentication token (required when public=False).
        """
        if public:
            save_dir = data_dir / "saved" / "public"
        else:
            _, safe_uid = await _authenticate_user(
                resolve_token(authorization, token), cached_user_info
            )
            save_dir = data_dir / "saved" / safe_uid

        full_path = _validate_path(filename, save_dir)
        if not full_path.exists() or not full_path.is_file():
            raise HTTPException(status_code=404, detail=f"'{filename}' not found")

        return await _serve_file_response(full_path, request)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def start_proxy_server(
    data_dir: Union[str, Path],
    server_ip: Optional[str] = None,
    server_port: Optional[int] = None,
    authentication_server_url: str = "https://hypha.aicell.io",
    log_file: Optional[Union[str, Path]] = None,
) -> None:
    """
    Start the BioEngine Datasets proxy server.

    Scans data_dir for dataset subdirectories (each must contain a manifest.yaml),
    then starts a FastAPI server that:
    - Lists datasets and their manifests
    - Lists files within a dataset (access-controlled)
    - Streams zarr chunk files with HTTP Range request support

    No data is copied — datasets are served in-place from data_dir.
    No Hypha service registration is performed — clients connect directly
    to this server. The local server URL is written to
    ~/.bioengine/datasets/bioengine_current_server for client auto-discovery.

    Args:
        data_dir: Directory containing dataset subdirectories. Each must have a
                  manifest.yaml with at minimum an "id" and "authorized_users" field.
        server_ip: IP address for the HTTP server. Defaults to the machine's
                   internal IP.
        server_port: Port for the HTTP server. If None (default), scans for a
                     free port starting from 39527.
        authentication_server_url: URL of the central Hypha server used for
                                   per-request token validation
                                   (default: https://hypha.aicell.io).
        log_file: Path to log file. Pass "off" for console-only logging.
                  Defaults to a timestamped file in ~/.bioengine/logs/.
    """
    global AUTHENTICATION_SERVER_URL

    data_dir = Path(data_dir).resolve()
    AUTHENTICATION_SERVER_URL = authentication_server_url

    # Logging setup
    if log_file != "off":
        if log_file is None:
            log_dir = Path.home() / ".bioengine" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"bioengine_datasets_{time.strftime('%Y%m%d_%H%M%S')}.log"
        else:
            log_file = Path(log_file).resolve()
            log_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_file = None

    global logger
    logger = create_logger("ProxyServer", log_file=log_file)

    # Client auto-discovery file — fixed location read by BioEngineDatasets("auto")
    current_server_file = Path.home() / ".bioengine" / "datasets" / "bioengine_current_server"
    try:
        current_server_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(f"Could not create auto-discovery directory: {e}")
        current_server_file = None

    try:
        logger.info(f"Starting BioEngine Datasets proxy server v{__version__}")
        logger.info(f"Serving datasets from: {data_dir}")

        datasets = load_datasets(data_dir)
        logger.info(
            f"Found {len(datasets)} dataset(s): {list(datasets.keys())} "
            f"(watching for changes every 30 s)"
        )

        server_ip = server_ip or get_internal_ip()
        # Default port range: 39527–39999 (avoids Hypha's 9527 and Ray's ports)
        start_port = server_port if server_port is not None else 39527
        free_server_port, server_s = acquire_free_port(
            port=start_port, step=1, ip=server_ip, keep_open=True
        )
        server_s.close()
        if server_port is not None and free_server_port != server_port:
            logger.warning(
                f"Port {server_port} unavailable, using {free_server_port} instead."
            )

        local_http_url = f"http://{server_ip}:{free_server_port}"

        # Write local URL for BioEngineDatasets("auto") discovery
        if current_server_file is not None:
            current_server_file.write_text(local_http_url)
        logger.info(f"Server URL: {local_http_url}")

        cached_user_info: Dict[str, dict] = {}

        app = _build_app(
            data_dir=data_dir,
            datasets=datasets,
            cached_user_info=cached_user_info,
        )

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=free_server_port,
            log_config=get_log_config(log_file=log_file),
            log_level="info",
        )

    except Exception as e:
        logger.error(f"Failed to start BioEngine Datasets proxy server: {e}")
        raise

    finally:
        if current_server_file is not None and current_server_file.exists():
            try:
                current_server_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up current server file: {e}")
