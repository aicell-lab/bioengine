"""Locating a BioEngine worker without hardcoding its pod name."""


async def resolve_worker(server, prefix: str):
    """Find the worker service whose id starts with ``prefix``.

    Kubernetes bakes the pod name into the service id, so a hardcoded id goes
    stale the moment the worker pod is replaced -- which has already happened
    twice here, on de.NBI. Fail loudly rather than pick one if the prefix is
    ambiguous: two workers in a workspace would silently split the run across
    two clusters.
    """
    workspace = prefix.split("/", 1)[0]
    matches = [
        service["id"]
        for service in await server.list_services(workspace)
        if service["id"].startswith(prefix) and service["id"].endswith(":bioengine-worker")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one worker matching {prefix!r}, got {matches}")
    return await server.get_service(matches[0])
