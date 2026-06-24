"""Module-level Ray-task helpers for the app cache API.

Kept in a separate module so the Ray Client server can re-import them by
reference without pulling in the heavy ``bioengine.apps.manager`` namespace
(which depends on ``haikunator``, ``hypha_rpc``, ``ray.serve``, etc., not
guaranteed to be installed in the Ray Client server's base venv).

Imports here must stay limited to Python stdlib.
"""

from __future__ import annotations


def fs_probe_write(apps_workdir: str, marker_name: str, content: str) -> bool:
    """Write a marker file in ``apps_workdir`` so peer nodes can probe it."""
    from pathlib import Path

    p = Path(apps_workdir)
    p.mkdir(parents=True, exist_ok=True)
    (p / marker_name).write_text(content)
    return True


def fs_probe_read(apps_workdir: str, marker_name: str, expected: str) -> bool:
    """True iff this node sees the marker file with the expected bytes."""
    from pathlib import Path

    try:
        return (Path(apps_workdir) / marker_name).read_text() == expected
    except (FileNotFoundError, OSError):
        return False


def fs_probe_delete(apps_workdir: str, marker_name: str) -> bool:
    """Remove the marker file if present (idempotent)."""
    from pathlib import Path

    try:
        (Path(apps_workdir) / marker_name).unlink()
        return True
    except FileNotFoundError:
        return False


def list_dirs_on_node(
    apps_workdir_str: str,
    prefix: str,
    running_ids: list,
    node_id: str,
) -> list:
    """Walk ``apps_workdir`` on this Ray node, returning per-directory stats."""
    from pathlib import Path

    apps_workdir = Path(apps_workdir_str)
    if not apps_workdir.exists():
        return []
    running_set = set(running_ids)
    result = []
    for entry in sorted(apps_workdir.iterdir()):
        if not entry.is_dir():
            continue
        size = 0
        latest_mtime = None
        for f in entry.rglob("*"):
            if not f.is_file():
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            size += st.st_size
            if latest_mtime is None or st.st_mtime > latest_mtime:
                latest_mtime = st.st_mtime
        application_id = (
            entry.name[len(prefix):] if entry.name.startswith(prefix) else None
        )
        result.append(
            {
                "name": entry.name,
                "application_id": application_id,
                "path": str(entry),
                "is_running": application_id in running_set if application_id else False,
                "size_bytes": size,
                "last_used_unix": latest_mtime,
                "node_id": node_id,
            }
        )
    return result


def clear_dir_on_node(
    apps_workdir_str: str,
    worker_workspace: str,
    application_id: str,
    node_id: str,
) -> dict:
    """Remove the app cache directory on this Ray node if present."""
    import shutil
    from pathlib import Path

    apps_workdir = Path(apps_workdir_str)
    prefixed = apps_workdir / f"{worker_workspace}-{application_id}"
    bare = apps_workdir / application_id
    target = prefixed if prefixed.exists() else bare
    if not target.exists():
        return {
            "ok": False,
            "error": "not_found",
            "path": str(target),
            "node_id": node_id,
        }
    if not target.is_dir():
        return {
            "ok": False,
            "error": "not_a_dir",
            "path": str(target),
            "node_id": node_id,
        }
    try:
        shutil.rmtree(target)
    except Exception as e:
        return {
            "ok": False,
            "error": f"rmtree_failed: {e}",
            "path": str(target),
            "node_id": node_id,
        }
    return {"ok": True, "error": None, "path": str(target), "node_id": node_id}
