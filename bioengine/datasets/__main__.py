#!/usr/bin/env python3
"""
BioEngine Datasets - Privacy-preserved scientific data management system.

Serves zarr datasets in-place from a local directory with per-user access
control. Runs a local FastAPI server for dataset discovery, access control,
and efficient zarr chunk streaming directly to clients.

No data copy is performed — datasets are served from their original location.
Token authentication is validated against the remote Hypha server on demand.

Usage:
    python -m bioengine.datasets --data-dir /path/to/datasets
    python -m bioengine.datasets --data-dir /path/to/datasets --server-port 39527
"""

import argparse
import sys

from bioengine.datasets.proxy_server import start_proxy_server


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BioEngine Datasets - Privacy-Preserved Data Streaming Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start datasets service (scans for a free port starting from 39527)
  %(prog)s --data-dir /shared/data

  # Explicit port
  %(prog)s --data-dir /shared/data --server-port 39527

  # Custom authentication server
  %(prog)s --data-dir /shared/data --authentication-server-url https://hypha.aicell.io

For detailed documentation, visit: https://github.com/aicell-lab/bioengine
""",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        metavar="PATH",
        required=True,
        help="Directory containing dataset subdirectories. Each subdirectory must "
        "have a manifest.yaml file defining metadata and access permissions.",
    )
    parser.add_argument(
        "--server-ip",
        type=str,
        metavar="IP_ADDRESS",
        help="IP address for the local file-serving HTTP server. Defaults to the "
        "machine's internal IP.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        metavar="PORT",
        help="Port for the local HTTP server. If not provided, scans for a free "
        "port starting from 39527.",
    )
    parser.add_argument(
        "--authentication-server-url",
        type=str,
        metavar="URL",
        default="https://hypha.aicell.io",
        help="URL of the central Hypha server used for token validation "
        "(default: https://hypha.aicell.io).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        metavar="PATH",
        help="Path to the log file. Pass 'off' to log to console only. "
        "Defaults to a timestamped file in ~/.bioengine/logs/.",
    )
    parser.add_argument(
        "--enable-access-requests",
        action="store_true",
        help="Expose the per-dataset access-request endpoints so a user who is "
        "not in a dataset's authorized_users can ask for access. Requires "
        "--worker-service-id. This is separate from the worker's flag of "
        "the same name — the two processes carry their own request surfaces.",
    )
    parser.add_argument(
        "--worker-service-id",
        type=str,
        metavar="ID",
        help="Full Hypha service id of the BioEngine worker whose admin users "
        "may decide dataset access requests, e.g. "
        "'my-workspace/my-worker:bioengine-worker'. This server keeps no admin "
        "list of its own — it asks that worker, so runtime changes to its admin "
        "users take effect immediately. Required by --enable-access-requests.",
    )

    return parser


if __name__ == "__main__":
    try:
        parser = create_parser()
        args = parser.parse_args()
        kwargs = {k: v for k, v in vars(args).items() if v is not None}
        start_proxy_server(**kwargs)
    except Exception as e:
        print(f"Failed to start BioEngine Datasets proxy server: {e}")
        sys.exit(1)
