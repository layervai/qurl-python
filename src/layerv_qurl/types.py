"""Type definitions for the QURL API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

#: Valid QURL status values. Accepts known values for IDE autocomplete,
#: plus ``str`` for forward compatibility with new API statuses.
QURLStatus = Literal["active", "revoked"] | str


def _parse_dt(s: str | None) -> datetime | None:
    """Parse an ISO 8601 datetime string, handling Z suffix for Python 3.10 compat."""
    if s is None:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


@dataclass
class AIAgentPolicy:
    """Structured policy for controlling AI agent access."""

    block_all: bool | None = None
    deny_categories: list[str] | None = None
    allow_categories: list[str] | None = None


@dataclass
class AccessPolicy:
    """Access control policy for a QURL."""

    ip_allowlist: list[str] | None = None
    ip_denylist: list[str] | None = None
    geo_allowlist: list[str] | None = None
    geo_denylist: list[str] | None = None
    user_agent_allow_regex: str | None = None
    user_agent_deny_regex: str | None = None
    ai_agent_policy: AIAgentPolicy | None = None


@dataclass
class AccessToken:
    """An individual access token within a QURL."""

    qurl_id: str
    status: QURLStatus
    one_time_use: bool = False
    max_sessions: int = 0
    session_duration: int = 0
    use_count: int = 0
    label: str | None = None
    qurl_site: str | None = None
    access_policy: AccessPolicy | None = None
    created_at: datetime | None = None
    expires_at: datetime | None = None


@dataclass
class QURL:
    """A QURL resource as returned by the API."""

    resource_id: str
    target_url: str
    status: QURLStatus
    created_at: datetime | None = None
    expires_at: datetime | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    qurl_site: str | None = None
    custom_domain: str | None = None
    qurl_count: int | None = None
    access_tokens: list[AccessToken] | None = None


@dataclass
class CreateOutput:
    """Response from creating a QURL."""

    resource_id: str
    qurl_link: str
    qurl_site: str
    expires_at: datetime | None = None
    qurl_id: str = ""
    label: str | None = None


@dataclass
class MintOutput:
    """Response from minting an access link."""

    qurl_link: str
    expires_at: datetime | None = None


@dataclass
class AccessGrant:
    """Details of the firewall access that was granted."""

    expires_in: int
    granted_at: datetime | None = None
    src_ip: str = ""


@dataclass
class ResolveOutput:
    """Response from headless resolution."""

    target_url: str
    resource_id: str
    access_grant: AccessGrant | None = None


@dataclass
class ListOutput:
    """Response from listing QURLs."""

    qurls: list[QURL] = field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False


@dataclass
class RateLimits:
    """Rate limit configuration."""

    create_per_minute: int = 0
    create_per_hour: int = 0
    list_per_minute: int = 0
    resolve_per_minute: int = 0
    max_active_qurls: int = 0
    max_tokens_per_qurl: int = 0
    max_expiry_seconds: int = 0


@dataclass
class Usage:
    """Usage statistics."""

    qurls_created: int = 0
    active_qurls: int = 0
    active_qurls_percent: float | None = None
    total_accesses: int = 0


@dataclass
class Quota:
    """Quota and usage information."""

    plan: str = ""
    period_start: datetime | None = None
    period_end: datetime | None = None
    rate_limits: RateLimits | None = None
    usage: Usage | None = None


@dataclass
class BatchItemError:
    """Error details for a failed batch item."""

    code: str = ""
    message: str = ""


@dataclass
class BatchItemResult:
    """Result for a single item in a batch create."""

    index: int = 0
    success: bool = False
    resource_id: str | None = None
    qurl_link: str | None = None
    qurl_site: str | None = None
    expires_at: datetime | None = None
    error: BatchItemError | None = None


@dataclass
class BatchCreateOutput:
    """Response from batch creating QURLs."""

    succeeded: int = 0
    failed: int = 0
    results: list[BatchItemResult] = field(default_factory=list)
