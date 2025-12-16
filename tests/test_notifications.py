import os
from src.notifications import send_webhook


def test_send_webhook_no_url(monkeypatch):
    monkeypatch.delenv("NOTIFY_WEBHOOK_URL", raising=False)
    # should not raise
    send_webhook("noop")


def test_send_webhook_with_url(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json

    monkeypatch.setenv("NOTIFY_WEBHOOK_URL", "http://example.com")
    monkeypatch.setattr("src.notifications.requests.post", fake_post)
    send_webhook("hello", status="success")
    assert captured["url"] == "http://example.com"
    assert captured["json"]["status"] == "success"
    assert captured["json"]["message"] == "hello"
