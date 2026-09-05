"""Locating a BioEngine worker without hardcoding its pod name."""


async def resolve_worker(server, prefix: str):
    """Find the worker service whose id starts with ``prefix``.

    Kubernetes bakes the pod name into the service id, so a hardcoded id goes
    stale the moment the worker pod is replaced -- which has already happened
    three times here, on de.NBI. During a rolling replacement both pods are
    registered at once and they front the *same* Ray cluster, so the match is
    ambiguous but harmless; two pods on different clusters would silently split
    the run and must not be. The head address is what tells the two cases apart.
    """
    workspace = prefix.split("/", 1)[0]
    matches = [
        service["id"]
        for service in await server.list_services(workspace)
        if service["id"].startswith(prefix) and service["id"].endswith(":bioengine-worker")
    ]
    if not matches:
        raise RuntimeError(f"no worker matching {prefix!r}")
    workers = [await server.get_service(match) for match in matches]
    heads = {(await worker.get_status())["ray_cluster"]["head_address"] for worker in workers}
    if len(heads) != 1:
        raise RuntimeError(
            f"{prefix!r} matches workers on different Ray clusters {heads}: {matches}"
        )
    return workers[0]
