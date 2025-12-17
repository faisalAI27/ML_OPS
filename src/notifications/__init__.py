"""Notification utilities (Discord + generic webhook)."""

from __future__ import annotations

import os
import requests

from .discord import notify_discord  # re-export


def send_webhook(message: str, status: str = "info") -> None:
    """Send a generic webhook if NOTIFY_WEBHOOK_URL is set; best-effort."""
    url = os.getenv("NOTIFY_WEBHOOK_URL")
    if not url:
        return
    payload = {"status": status, "message": message}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception:
        # Do not crash the caller
        pass
