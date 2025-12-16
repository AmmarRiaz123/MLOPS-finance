import os
import logging
from datetime import datetime, timezone
from typing import Optional

import requests

LOG = logging.getLogger("alerting")


def send_discord_alert(message: str, *, webhook_url: Optional[str] = None, include_timestamp: bool = True) -> bool:
    """
    Send a message to Discord via webhook.
    - Reads webhook URL from env DISCORD_WEBHOOK_URL unless explicitly provided.
    - Returns True if sent successfully, False otherwise (never raises).
    """
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        LOG.info("DISCORD_WEBHOOK_URL not set; skipping alert")
        return False

    msg = str(message).strip()
    if include_timestamp:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        msg = f"[{ts}] {msg}"

    payload = {"content": msg}

    try:
        resp = requests.post(url, json=payload, timeout=5)
        if 200 <= resp.status_code < 300:
            return True
        LOG.warning("Discord alert failed: status=%s body=%s", resp.status_code, resp.text[:500])
        return False
    except requests.RequestException as e:
        LOG.warning("Discord alert request error: %s", e)
        return False
