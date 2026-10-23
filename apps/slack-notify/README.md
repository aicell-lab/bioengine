# Slack Notifier

Sends a Slack **direct message**, so a BioEngine app or an agent session can page a
person instead of only writing to a log.

## Methods

| method | what it does |
|---|---|
| `send_dm(text, user_id=None, blocks=None)` | DMs the user; `user_id` defaults to the recipient configured at deploy time |
| `ping()` | returns the bot identity Slack reports for the token — sends no message |

`send_dm` raises on a Slack error rather than returning a success-shaped dict, so a
failed page is never mistaken for a delivered one.

## The Slack side

Slack app **BioEngine Notifier** (`A0CASRKNYDT`) in the `aicell-lab` workspace, bot
scopes `chat:write`, `im:write`, `users:read`. Get the bot token from
<https://api.slack.com/apps/A0CASRKNYDT/oauth> after installing it to the workspace;
it starts with `xoxb-`.

## Deploying

The bot token goes in as a **masked secret** — the leading underscore is what makes it
one. The worker stores it as `_BIOENGINE_SECRET_SLACK_BOT_TOKEN`, `get_app_status`
reports `*****`, and the replica reads the real value from `SLACK_BOT_TOKEN`. Without
the underscore the token is published in app status.

```python
app_id = await worker.deploy_app(
    artifact_id="bioimage-io/slack-notify",
    version="0.1.0",
    application_id="slack-notify",
    application_kwargs={"SlackNotifier": {"default_user_id": "U06DE2TPG1W"}},
    application_env_vars={"SlackNotifier": {"_SLACK_BOT_TOKEN": "xoxb-..."}},
)
svc = await client.get_service(f"bioimage-io/{app_id}")
await svc.send_dm(text="worker is back up")
```

The recipient is **not** defaulted in the code: a Slack user id committed to the repo
would page whoever that id belongs to. Pass `default_user_id` at deploy time, or
`SLACK_DM_USER_ID` as a (non-secret) env var.

## Two things to know before relying on it

- **Anyone who can invoke the service can send Slack messages as the bot.** The token
  lives in the replica, not in the caller — app auth gates invocation, so the blast
  radius is whoever can call this app in the workspace.
- **It cannot report that BioEngine itself is down.** Paging through a BioEngine app
  means the page needs the worker to be healthy. For worker/cluster alerts use a path
  with no BioEngine dependency (e.g. `~/.claude/skills/send-email/`); use this app for
  everything that assumes a working worker.

## No pip dependencies

`urllib` on a worker thread, deliberately: the runtime_env stays the base image, so an
app whose job is to be available when something else broke has nothing to build first.
