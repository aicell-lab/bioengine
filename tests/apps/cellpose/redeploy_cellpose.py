"""Redeploy the cellpose finetuning application with the latest code."""

import argparse
import asyncio
import os

from hypha_rpc import connect_to_server


async def redeploy(artifact_id: str, application_id: str):
    """Redeploy ``application_id`` in place, then wait for it to serve again.

    Deploying an application_id that is already running *is* the update — there
    is no stop_app step, and believing there was one is what made the
    ``version=None`` below look safe: it reads as "start fresh from latest" but
    actually inherits the running version.
    """
    server_url = "https://hypha.aicell.io"
    token = os.environ.get("HYPHA_TOKEN")
    if not token:
        raise ValueError("HYPHA_TOKEN environment variable is not set")

    async with connect_to_server(
        {"server_url": server_url, "token": token, "workspace": "ri-scale"}
    ) as hypha:
        workspace = hypha.config.workspace
        print(f"Using workspace: {workspace}")

        worker = await hypha.get_service("bioimage-io/bioengine-worker")

        # Start the new deployment
        print("Starting new deployment with latest artifact...")
        deployed = await worker.deploy_app(
            artifact_id=artifact_id,
            application_id=application_id,
            hypha_token=token,
            version=None,  # latest version
            disable_gpu=False,  # set True to force CPU-only
            max_ongoing_requests=1,  # keep at 1 for GPU
        )
        app_id = deployed["application_id"]
        print(f"App ID: {app_id} (version {deployed['version']})")
        # version=None on a running application_id inherits that app's version
        # instead of resolving latest, so this script would otherwise redeploy
        # the code it is trying to replace and report success.
        if deployed["version_source"] == "inherited":
            raise SystemExit(
                f"Redeploy inherited version {deployed['version']} from the "
                f"running '{app_id}' instead of picking up the latest. Pass an "
                f"explicit version=. Note the redeploy of {deployed['version']} "
                f"is already in flight — deploy_app starts it before returning, "
                f"so exiting here does not call it back."
            )

        # Wait for services to become available
        print("Waiting for services to start...")
        import pprint

        for i in range(10):
            app_status = await worker.get_app_status(
                application_ids=[application_id]
            )
            print(f"DEBUG: Status check {i}:")
            pprint.pprint(app_status, indent=2)

            # Try to find service_ids in nested structure if possible
            if isinstance(app_status, dict):
                # Check if it's a list of statuses or a single status dict keyed by app_id
                if application_id in app_status:
                    status = app_status[application_id]
                    if status.get("status") == "RUNNING":
                        print(f"Application is running! Details: {status}")
                        return

            service_ids = app_status.get("service_ids", [])
            if service_ids:
                print(f"Services: {service_ids}")
                return service_ids
            await asyncio.sleep(2)
        print(
            "Warning: Services not yet available. The deployment may still be starting up."
        )
        return []


if __name__ == "__main__":
    # add argparse argument artifact Id and application id
    parser = argparse.ArgumentParser(
        description="Redeploy the cellpose finetuning application with the latest code.",
    )
    parser.add_argument(
        "--artifact-id",
        type=str,
        default="bioimage-io/cellpose-finetuning",
        help="Artifact ID to deploy (default: bioimage-io/cellpose-finetuning)",
    )
    parser.add_argument(
        "--application-id",
        type=str,
        default="cellpose-finetuning",
        help="Application ID to deploy (default: cellpose-finetuning)",
    )
    args = parser.parse_args()
    asyncio.run(
        redeploy(
            artifact_id=args.artifact_id,
            application_id=args.application_id,
        )
    )
