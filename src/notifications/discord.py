"""Discord notification helper for CI/training events."""

from __future__ import annotations

import os
import requests


def notify_discord(message: str) -> None:
    """Send a Discord webhook message if configured."""
    webhook = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook:
        return
    try:
        requests.post(webhook, json={"content": message}, timeout=10)
    except Exception:
        # Swallow errors to keep callers resilient
        pass
