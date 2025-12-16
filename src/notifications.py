"""Simple webhook notification helper."""

from __future__ import annotations

import os
import requests


def send_webhook(message: str, status: str = "info") -> None:
    url = os.getenv("NOTIFY_WEBHOOK_URL")
    if not url:
        return
    payload = {"status": status, "message": message}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception:
        # Do not crash the caller
        pass
