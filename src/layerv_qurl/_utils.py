"""Shared utilities for sync and async clients."""

from __future__ import annotations

import dataclasses
import functools
import logging
import random
import re
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING, Any

from layerv_qurl.errors import (
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    QURLError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from layerv_qurl.types import (
    QURL,
    AccessGrant,
    AccessPolicy,
    AccessToken,
    AIAgentPolicy,
    BatchCreateOutput,
    BatchItemError,
    BatchItemResult,
    CreateOutput,
    ListOutput,
    MintOutput,
    Quota,
    RateLimits,
    ResolveOutput,
    Usage,
    _parse_dt,
)

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger("layerv_qurl")

DEFAULT_BASE_URL = "https://api.layerv.ai"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
RETRYABLE_STATUS = {429, 502, 503, 504}
RETRYABLE_STATUS_POST = {429}  # POST is not idempotent — only retry rate limits

_RESOURCE_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")

_ERROR_CLASS_MAP: dict[int, type[QURLError]] = {
    400: ValidationError,
    401: AuthenticationError,
    403: AuthorizationError,
    404: NotFoundError,
    422: ValidationError,
    429: RateLimitError,
}


@functools.lru_cache(maxsize=1)
def default_user_agent() -> str:
    """Return the default User-Agent string, caching the version lookup."""
    try:
        v = _pkg_version("layerv-qurl")
    except PackageNotFoundError:
        v = "dev"
    return f"qurl-python-sdk/{v}"


def validate_id(value: str, name: str = "resource_id") -> str:
    """Validate that an ID is non-empty and contains no path traversal characters."""
    if not value or not _RESOURCE_ID_RE.match(value):
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def _serialize_value(v: Any) -> Any:
    """Serialize a single value for JSON, handling dataclasses and datetimes recursively."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {
            f.name: _serialize_value(getattr(v, f.name))
            for f in dataclasses.fields(v)
            if getattr(v, f.name) is not None
        }
    if isinstance(v, list):
        return [_serialize_value(item) for item in v]
    return v


def build_body(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build a request body dict from kwargs, dropping None values.

    Always returns a dict (at least ``{}``) so POST/PATCH endpoints
    receive a valid JSON body.  Nested dataclasses are recursively
    serialized to dicts.
    """
    body: dict[str, Any] = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        body[k] = _serialize_value(v)
    return body


def _parse_access_policy(data: dict[str, Any]) -> AccessPolicy:
    """Parse an AccessPolicy from API response data."""
    ai_policy = None
    if data.get("ai_agent_policy") is not None:
        ap = data["ai_agent_policy"]
        ai_policy = AIAgentPolicy(
            block_all=ap.get("block_all"),
            deny_categories=ap.get("deny_categories"),
            allow_categories=ap.get("allow_categories"),
        )
    return AccessPolicy(
        ip_allowlist=data.get("ip_allowlist"),
        ip_denylist=data.get("ip_denylist"),
        geo_allowlist=data.get("geo_allowlist"),
        geo_denylist=data.get("geo_denylist"),
        user_agent_allow_regex=data.get("user_agent_allow_regex"),
        user_agent_deny_regex=data.get("user_agent_deny_regex"),
        ai_agent_policy=ai_policy,
    )


def _parse_access_token(data: dict[str, Any]) -> AccessToken:
    """Parse an AccessToken from API response data."""
    policy = None
    if data.get("access_policy") is not None:
        policy = _parse_access_policy(data["access_policy"])
    return AccessToken(
        qurl_id=data["qurl_id"],
        status=data["status"],
        one_time_use=data.get("one_time_use", False),
        max_sessions=data.get("max_sessions", 0),
        session_duration=data.get("session_duration", 0),
        use_count=data.get("use_count", 0),
        label=data.get("label"),
        qurl_site=data.get("qurl_site"),
        access_policy=policy,
        created_at=_parse_dt(data.get("created_at")),
        expires_at=_parse_dt(data.get("expires_at")),
    )


def parse_qurl(data: dict[str, Any]) -> QURL:
    """Parse a QURL resource from API response data."""
    tokens = None
    # API returns "qurls" array; SDK exposes as "access_tokens" for clarity.
    raw_tokens = data.get("qurls") if "qurls" in data else data.get("access_tokens")
    if raw_tokens is not None:
        tokens = [_parse_access_token(t) for t in raw_tokens]
    return QURL(
        resource_id=data["resource_id"],
        target_url=data["target_url"],
        status=data["status"],
        created_at=_parse_dt(data.get("created_at")),
        expires_at=_parse_dt(data.get("expires_at")),
        description=data.get("description"),
        tags=data.get("tags", []),
        qurl_site=data.get("qurl_site"),
        custom_domain=data.get("custom_domain"),
        qurl_count=data.get("qurl_count"),
        access_tokens=tokens,
    )


def parse_create_output(data: dict[str, Any]) -> CreateOutput:
    """Parse a CreateOutput from API response data."""
    return CreateOutput(
        resource_id=data["resource_id"],
        qurl_link=data["qurl_link"],
        qurl_site=data["qurl_site"],
        expires_at=_parse_dt(data.get("expires_at")),
        qurl_id=data.get("qurl_id"),
        label=data.get("label"),
    )


def parse_mint_output(data: dict[str, Any]) -> MintOutput:
    """Parse a MintOutput from API response data."""
    return MintOutput(
        qurl_link=data["qurl_link"],
        expires_at=_parse_dt(data.get("expires_at")),
    )


def parse_resolve_output(data: dict[str, Any]) -> ResolveOutput:
    """Parse a ResolveOutput from API response data."""
    grant = None
    if data.get("access_grant"):
        g = data["access_grant"]
        grant = AccessGrant(
            expires_in=g["expires_in"],
            granted_at=_parse_dt(g.get("granted_at")),
            src_ip=g.get("src_ip", ""),
        )
    return ResolveOutput(
        target_url=data["target_url"],
        resource_id=data["resource_id"],
        access_grant=grant,
    )


def parse_quota(data: dict[str, Any]) -> Quota:
    """Parse a Quota from API response data."""
    rl = None
    if data.get("rate_limits"):
        r = data["rate_limits"]
        rl = RateLimits(
            create_per_minute=r.get("create_per_minute", 0),
            create_per_hour=r.get("create_per_hour", 0),
            list_per_minute=r.get("list_per_minute", 0),
            resolve_per_minute=r.get("resolve_per_minute", 0),
            max_active_qurls=r.get("max_active_qurls", 0),
            max_tokens_per_qurl=r.get("max_tokens_per_qurl", 0),
            max_expiry_seconds=r.get("max_expiry_seconds", 0),
        )
    usage = None
    if data.get("usage"):
        u = data["usage"]
        usage = Usage(
            qurls_created=u.get("qurls_created", 0),
            active_qurls=u.get("active_qurls", 0),
            active_qurls_percent=u.get("active_qurls_percent"),
            total_accesses=u.get("total_accesses", 0),
        )
    return Quota(
        plan=data.get("plan", ""),
        period_start=_parse_dt(data.get("period_start")),
        period_end=_parse_dt(data.get("period_end")),
        rate_limits=rl,
        usage=usage,
    )


def parse_list_output(data: Any, meta: dict[str, Any] | None) -> ListOutput:
    """Parse a ListOutput from API response data."""
    qurls = [parse_qurl(q) for q in data] if isinstance(data, list) else []
    return ListOutput(
        qurls=qurls,
        next_cursor=meta.get("next_cursor") if meta else None,
        has_more=meta.get("has_more", False) if meta else False,
    )


def parse_batch_create_output(data: dict[str, Any]) -> BatchCreateOutput:
    """Parse a BatchCreateOutput from API response data."""
    results: list[BatchItemResult] = []
    for item in data.get("results", []):
        err = None
        if item.get("error"):
            e = item["error"]
            err = BatchItemError(
                code=e.get("code", ""),
                message=e.get("message", ""),
            )
        results.append(
            BatchItemResult(
                index=item.get("index", 0),
                success=item.get("success", False),
                resource_id=item.get("resource_id"),
                qurl_link=item.get("qurl_link"),
                qurl_site=item.get("qurl_site"),
                expires_at=_parse_dt(item.get("expires_at")),
                error=err,
            )
        )
    return BatchCreateOutput(
        succeeded=data.get("succeeded", 0),
        failed=data.get("failed", 0),
        results=results,
    )


def parse_error(response: httpx.Response) -> QURLError:
    """Parse an API error response into the appropriate QURLError subclass."""
    retry_after = None
    if response.status_code == 429:
        ra = response.headers.get("Retry-After")
        if ra and ra.isdigit():
            retry_after = int(ra)

    # Pick the right subclass, defaulting to ServerError for 5xx or QURLError
    cls: type[QURLError]
    if response.status_code >= 500:
        cls = ServerError
    else:
        cls = _ERROR_CLASS_MAP.get(response.status_code, QURLError)

    try:
        envelope = response.json()
        err = envelope.get("error", {})
        return cls(
            status=err.get("status", response.status_code),
            code=err.get("code", "unknown"),
            title=err.get("title", response.reason_phrase or ""),
            detail=err.get("detail", ""),
            invalid_fields=err.get("invalid_fields"),
            request_id=envelope.get("meta", {}).get("request_id"),
            retry_after=retry_after,
        )
    except (ValueError, KeyError, TypeError):
        return cls(
            status=response.status_code,
            code="unknown",
            title=response.reason_phrase or "",
            detail=response.text,
            retry_after=retry_after,
        )


def retry_delay(attempt: int, last_error: Exception | None) -> float:
    """Compute retry delay with exponential backoff, jitter, and Retry-After cap."""
    if isinstance(last_error, QURLError) and last_error.retry_after:
        return min(float(last_error.retry_after), 30.0)
    base: float = 0.5 * (2 ** (attempt - 1))
    jitter = random.random() * base * 0.5  # noqa: S311
    return min(base + jitter, 30.0)


def build_list_params(
    limit: int | None,
    cursor: str | None,
    status: str | None,
    q: str | None,
    sort: str | None,
    created_after: datetime | str | None = None,
    created_before: datetime | str | None = None,
    expires_before: datetime | str | None = None,
    expires_after: datetime | str | None = None,
) -> dict[str, str]:
    """Build query params for list endpoints, dropping None values."""
    pairs: dict[str, str | int | datetime | None] = {
        "limit": limit,
        "cursor": cursor,
        "status": status,
        "q": q,
        "sort": sort,
        "created_after": created_after,
        "created_before": created_before,
        "expires_before": expires_before,
        "expires_after": expires_after,
    }
    return {
        k: v.isoformat() if isinstance(v, datetime) else str(v)
        for k, v in pairs.items()
        if v is not None
    }


def mask_key(api_key: str) -> str:
    """Mask an API key for display, showing first 4 + last 4 chars."""
    if len(api_key) > 8:
        return api_key[:4] + "***" + api_key[-4:]
    return "***"
