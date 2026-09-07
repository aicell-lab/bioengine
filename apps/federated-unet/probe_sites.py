"""Three consecutive all-RUNNING probes 30 s apart before a relaunch.

Status flaps while the Serve controller is recovering, so one clean reading is a
false all-clear. Note this gate is necessary and NOT sufficient: a restarted
replica reports RUNNING while holding no prepared data (crash #8).
"""

import asyncio
import os

from hypha_rpc import connect_to_server

from workers import resolve_worker

SITES = {
    "ws-user-github|49943582/bioengine-worker-europa": (
        "HYPHA_TOKEN",
        ["fedunet-bbbc038-fluo", "fedunet-bbbc039", "fedunet-nuinsseg", "fedunet-pooled"],
    ),
    "bioimage-io/bioengine-worker-denbi": (
        "BIOIMAGE_IO_TOKEN",
        ["fedunet-bbbc038-histo", "fedunet-cellbindb", "fedunet-kromp"],
    ),
}


async def probe():
    results = {}
    for prefix, (token_var, apps) in SITES.items():
        server = await connect_to_server(
            {"server_url": "https://hypha.aicell.io", "token": os.environ[token_var]}
        )
        worker = await resolve_worker(server, prefix)
        status = await worker.get_app_status(application_ids=apps)
        for app in apps:
            results[app] = status.get(app, {}).get("status", "MISSING")
        await server.disconnect()
    return results


async def main():
    for attempt in range(3):
        if attempt:
            await asyncio.sleep(30)
        results = await probe()
        ok = all(state == "RUNNING" for state in results.values())
        print(f"probe {attempt + 1}: {'ALL RUNNING' if ok else 'NOT READY'}")
        for app, state in sorted(results.items()):
            print(f"  {app}: {state}")
        if not ok:
            raise SystemExit(1)
    print("gate passed: three consecutive all-RUNNING probes")


asyncio.run(main())
