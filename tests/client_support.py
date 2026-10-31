"""Shared response fixtures for client test modules."""

from __future__ import annotations

from typing import Any

BASE_URL = "https://api.test.layerv.ai"

_ERR_429 = {
    "error": {
        "status": 429,
        "code": "rate_limited",
        "title": "Rate Limited",
        "detail": "Slow down",
    },
}
_ERR_503 = {
    "error": {
        "status": 503,
        "code": "unavailable",
        "title": "Unavailable",
        "detail": "Down",
    },
}
_QUOTA_OK = {
    "data": {
        "plan": "growth",
        "period_start": "2026-03-01T00:00:00Z",
        "period_end": "2026-04-01T00:00:00Z",
    },
}


def _qurl_item(rid: str, url: str) -> dict[str, Any]:
    return {
        "resource_id": rid,
        "target_url": url,
        "status": "active",
        "created_at": "2026-03-10T10:00:00Z",
        "tags": [],
    }
