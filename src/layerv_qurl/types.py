"""Type definitions for the QURL API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TypedDict

#: Valid resource-level status values. Resources only have two states per
#: the OpenAPI spec (``QurlData.status``) and the server code. Accepts known
#: values for IDE autocomplete, plus ``str`` for forward compatibility.
QURLStatus = Literal["active", "revoked"] | str

#: Valid per-token status values. Wider than :data:`QURLStatus` because
#: individual access tokens can additionally be ``consumed`` (one-time use)
#: or ``expired``, per the OpenAPI spec (``QurlSummary.status``).
TokenStatus = Literal["active", "consumed", "expired", "revoked"] | str

#: Valid subscription plan values. Matches the ``QuotaData.plan`` enum in
#: the OpenAPI spec (``[free, growth, enterprise]``). Accepts arbitrary
#: strings so the API can add new plans without a breaking SDK change.
QuotaPlan = Literal["free", "growth", "enterprise"] | str


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
    """An individual access token within a QURL.

    ``status`` uses the wider :data:`TokenStatus` alias — tokens can be
    ``active``/``consumed``/``expired``/``revoked`` (per ``QurlSummary.status``
    in the spec), while resources are only ``active``/``revoked``.
    """

    qurl_id: str
    status: TokenStatus
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
    """Response from creating a QURL.

    ``resource_id`` identifies the resource container (grouped by target URL).
    ``qurl_id`` identifies the specific access token created (``q_`` prefix).
    Multiple QURLs for the same target URL share one ``resource_id``.
    """

    resource_id: str
    qurl_link: str
    qurl_site: str
    expires_at: datetime | None = None
    qurl_id: str | None = None
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
    # Changed from float=0.0 — callers must None-check before arithmetic.
    active_qurls_percent: float | None = None
    total_accesses: int = 0


@dataclass
class Quota:
    """Quota and usage information.

    ``plan`` is typed via :data:`QuotaPlan` (Literal enum + ``str``
    escape hatch for forward compat). Defaults to the sentinel
    ``"unknown"`` — only hit by tests/bootstrap paths, since the
    ``/v1/quota`` endpoint always returns a populated plan string.
    """

    plan: QuotaPlan = "unknown"
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


# ---- batch_create input shape -------------------------------------------
# TypedDicts for :meth:`QURLClient.batch_create` items. Split into a
# required-fields base class and an `total=False` subclass so callers on
# Python 3.10 don't need ``typing.Required`` (added in 3.11) or a
# ``typing_extensions`` dependency. Fields mirror the corresponding
# keyword arguments on :meth:`QURLClient.create` one-for-one; the
# single-create endpoint and the batch endpoint share the same
# ``CreateQurlRequest`` schema in the OpenAPI spec.
#
# The runtime behavior is unchanged — ``batch_create`` still iterates
# items as plain dicts and passes them through ``build_body`` /
# ``_serialize_value``. The TypedDict is purely for IDE autocomplete and
# static type checking.


class _BatchCreateItemRequired(TypedDict):
    target_url: str


class BatchCreateItem(_BatchCreateItemRequired, total=False):
    """Input shape for a single item in :meth:`QURLClient.batch_create`.

    ``target_url`` is required; every other field is optional and mirrors
    the corresponding keyword argument on :meth:`QURLClient.create`.

    ``access_policy`` accepts either an :class:`AccessPolicy` dataclass
    (recommended for type safety and IDE autocomplete) or a plain
    ``dict[str, Any]`` (for callers who prefer dicts or are working from
    dynamic config). Both forms are converted to the same JSON body at
    request time via ``_serialize_value``.
    """

    expires_in: str
    label: str
    one_time_use: bool
    max_sessions: int
    session_duration: str
    access_policy: AccessPolicy | dict[str, Any]
    custom_domain: str
