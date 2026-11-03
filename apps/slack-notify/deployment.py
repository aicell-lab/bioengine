"""Slack DM notifier — lets BioEngine apps and agent sessions page a person.

The bot token is passed at deploy time as a MASKED secret env var:

    application_env_vars={"SlackNotifier": {"_SLACK_BOT_TOKEN": "xoxb-..."}}

The leading underscore is what makes it a secret — the worker stores it as
``_BIOENGINE_SECRET_SLACK_BOT_TOKEN``, ``get_app_status`` reports it as
``*****``, and the replica sees the real value under ``SLACK_BOT_TOKEN``.
Passing it without the underscore would publish the token in app status.

No pip dependencies: ``urllib`` on a worker thread keeps the runtime_env
build to the base image, which matters for an app whose whole job is to be
available when something else has broken.
"""

import asyncio
import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from pydantic import Field

import bioengine

logger = bioengine.logger

SLACK_API = "https://slack.com/api"


def _call(method: str, token: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to one Slack Web API method and return the parsed body."""
    req = urllib.request.Request(
        f"{SLACK_API}/{method}",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


@bioengine.app(num_cpus=0.1, memory_mb=256)
class SlackNotifier:
    """Send a Slack direct message to a configured user."""

    def __init__(self, default_user_id: Optional[str] = None) -> None:
        self.token = os.environ.get("SLACK_BOT_TOKEN")
        if not self.token:
            raise RuntimeError(
                "SLACK_BOT_TOKEN is not set. Deploy with "
                'application_env_vars={"SlackNotifier": {"_SLACK_BOT_TOKEN": "xoxb-..."}} '
                "— the leading underscore keeps it masked in app status."
            )
        # Not defaulted in the manifest: the recipient is a person, and a Slack
        # user id baked into the repo would page whoever that id belongs to.
        self.default_user_id = default_user_id or os.environ.get("SLACK_DM_USER_ID")
        self._dm_channels: Dict[str, str] = {}

    @bioengine.smoke_test
    async def smoke(self) -> None:
        """Confirm the token authenticates — deliberately sends no message."""
        identity = await self.ping()
        assert identity["ok"], identity

    async def _dm_channel(self, user_id: str) -> str:
        """Resolve (and cache) the DM channel id for a user."""
        if user_id not in self._dm_channels:
            body = await asyncio.to_thread(
                _call, "conversations.open", self.token, {"users": user_id}
            )
            if not body.get("ok"):
                raise RuntimeError(
                    f"conversations.open failed for {user_id}: {body.get('error')}"
                )
            self._dm_channels[user_id] = body["channel"]["id"]
        return self._dm_channels[user_id]

    @bioengine.method
    async def ping(self) -> Dict[str, Any]:
        """Report the bot identity Slack sees for this token.

        Returns:
            ok, bot user name, user id, team, and the configured default recipient.
        """
        body = await asyncio.to_thread(_call, "auth.test", self.token, {})
        return {
            "ok": bool(body.get("ok")),
            "error": body.get("error", ""),
            "bot_user": body.get("user"),
            "bot_user_id": body.get("user_id"),
            "team": body.get("team"),
            "default_user_id": self.default_user_id,
        }

    @bioengine.method
    async def send_dm(
        self,
        text: str = Field(..., description="Message text (Slack mrkdwn)."),
        user_id: Optional[str] = Field(
            None,
            description="Slack user id to DM. Defaults to the recipient configured at deploy time.",
        ),
        blocks: Optional[List[Dict[str, Any]]] = Field(
            None,
            description="Optional Slack Block Kit blocks. When set, 'text' is used as the notification fallback.",
        ),
    ) -> Dict[str, Any]:
        """Send a direct message and return Slack's acknowledgement.

        Returns:
            ok, channel, ts (the message timestamp), and error when ok is false.
        """
        target = user_id or self.default_user_id
        if not target:
            raise ValueError(
                "No recipient: pass user_id, or set default_user_id / SLACK_DM_USER_ID at deploy time."
            )

        channel = await self._dm_channel(target)
        payload: Dict[str, Any] = {"channel": channel, "text": text}
        if blocks:
            payload["blocks"] = blocks

        body = await asyncio.to_thread(_call, "chat.postMessage", self.token, payload)
        if not body.get("ok"):
            # Surface the Slack error rather than returning a success-shaped dict.
            raise RuntimeError(f"chat.postMessage failed: {body.get('error')}")

        logger.info(f"Sent Slack DM to {target} (channel {channel}, ts {body.get('ts')})")
        return {
            "ok": True,
            "channel": channel,
            "ts": body.get("ts"),
            "user_id": target,
        }
