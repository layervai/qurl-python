"""Type definitions for the qURL API."""

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

#: Valid resource type values. Public callers primarily use ``url`` and
#: ``tunnel``; the API may also surface integration-owned ``transit`` rows.
ResourceType = Literal["url", "tunnel", "transit"] | str

WebhookEventType = (
    Literal[
        "qurl.created",
        "qurl.expired",
        "qurl.revoked",
        "qurl.updated",
        "resource.closed",
        "qurl.accessed",
        "qurl.access_denied",
        "qurl.token_exhausted",
        "quota.warning",
        "quota.exceeded",
        "token.minted",
        "token.expired",
        "domain.verified",
        "domain.failed",
        "domain.deleted",
    ]
    | str
)


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
    """Access control policy for a qURL."""

    ip_allowlist: list[str] | None = None
    ip_denylist: list[str] | None = None
    geo_allowlist: list[str] | None = None
    geo_denylist: list[str] | None = None
    user_agent_allow_regex: str | None = None
    user_agent_deny_regex: str | None = None
    ai_agent_policy: AIAgentPolicy | None = None


@dataclass
class AccessToken:
    """An individual access token within a qURL.

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
    """A qURL resource as returned by the API."""

    resource_id: str
    target_url: str | None
    status: QURLStatus
    created_at: datetime | None = None
    expires_at: datetime | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    qurl_site: str | None = None
    custom_domain: str | None = None
    slug: str | None = None
    qurl_count: int | None = None
    access_tokens: list[AccessToken] | None = None


@dataclass
class CreateOutput:
    """Response from creating a qURL.

    ``resource_id`` identifies the resource container (grouped by target URL).
    ``qurl_id`` identifies the specific access token created (``q_`` prefix).
    Multiple qURLs for the same target URL share one ``resource_id``.
    """

    resource_id: str
    qurl_link: str
    qurl_site: str
    expires_at: datetime | None = None
    qurl_id: str | None = None
    label: str | None = None
    branded_domain: str | None = None
    resource_type: ResourceType | None = None


@dataclass
class MintOutput:
    """Response from minting an access link."""

    qurl_link: str
    qurl_id: str | None = None
    expires_at: datetime | None = None
    branded_domain: str | None = None
    resource_type: ResourceType | None = None


@dataclass
class AccessGrant:
    """Details of the firewall access that was granted."""

    expires_in: int
    granted_at: datetime | None = None
    src_ip: str = ""


@dataclass
class ResolveOutput:
    """Response from headless resolution."""

    target_url: str | None
    resource_id: str
    access_grant: AccessGrant | None = None


@dataclass
class ListOutput:
    """Response from listing qURLs."""

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
    branded_domain: str | None = None
    expires_at: datetime | None = None
    error: BatchItemError | None = None


@dataclass
class BatchCreateOutput:
    """Response from batch creating qURLs."""

    succeeded: int = 0
    failed: int = 0
    results: list[BatchItemResult] = field(default_factory=list)


@dataclass
class Resource:
    """Resource container returned by the resource-management API."""

    resource_id: str
    status: QURLStatus
    resource_type: ResourceType | None = None
    target_url: str | None = None
    knock_resource_id: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    custom_domain: str | None = None
    alias: str | None = None
    slug: str | None = None
    preserve_host: bool = False
    session_duration_cap: int | None = None
    qurl_count: int | None = None
    created_at: datetime | None = None
    expires_at: datetime | None = None
    tombstoned_at: datetime | None = None


@dataclass
class ResourceDetail:
    """Detailed resource response with a bounded qURL token preview."""

    resource: Resource
    qurls: list[AccessToken] = field(default_factory=list)


@dataclass
class ResourceListOutput:
    """Response from listing resources."""

    resources: list[Resource] = field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False


@dataclass
class Session:
    """Active access session for a resource."""

    session_id: str
    qurl_id: str | None = None
    src_ip: str | None = None
    user_agent: str | None = None
    created_at: datetime | None = None
    last_seen_at: datetime | None = None


@dataclass
class SessionListOutput:
    """Response from listing active sessions."""

    sessions: list[Session] = field(default_factory=list)


@dataclass
class SessionTerminateOutput:
    """Response from terminating all active sessions on a resource."""

    terminated: int = 0


@dataclass
class DNSRecord:
    """DNS record required for custom-domain setup."""

    type: str = ""
    name: str = ""
    value: str = ""
    verified: bool = False


@dataclass
class Domain:
    """Custom-domain registration state."""

    domain: str
    status: str
    # Verification tokens are public DNS values, so repr keeps them visible.
    verification_token: str | None = None
    token_expires_at: datetime | None = None
    acme_cname_target: str | None = None
    created_at: datetime | None = None
    verified_at: datetime | None = None
    activated_at: datetime | None = None
    ready_for_qurls: bool = False
    dns_records: list[DNSRecord] = field(default_factory=list)


@dataclass
class DomainListOutput:
    """Response from listing custom domains."""

    domains: list[Domain] = field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False


@dataclass
class CheckDetail:
    """DNS verification check detail."""

    verified: bool
    error: str | None = None
    found: str | None = None


@dataclass
class DomainVerifyOutput:
    """Response from triggering custom-domain verification."""

    domain: str
    status: str
    checks: dict[str, CheckDetail] = field(default_factory=dict)


@dataclass
class Webhook:
    """Webhook subscription."""

    webhook_id: str
    url: str
    events: list[WebhookEventType]
    owner_id: str | None = None
    status: str | None = None
    description: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    failure_count: int = 0
    last_delivery_success: bool | None = None
    last_delivery_time: int | None = None
    secret: str | None = field(default=None, repr=False)


@dataclass
class WebhookListOutput:
    """Response from listing webhooks."""

    webhooks: list[Webhook] = field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False


@dataclass
class WebhookDelivery:
    """Webhook delivery attempt."""

    delivery_id: str
    webhook_id: str | None = None
    event_type: WebhookEventType | None = None
    status: str | None = None
    response_code: int | None = None
    response_body: str | None = None
    error_message: str | None = None
    duration_ms: int | None = None
    retry_count: int = 0
    created_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class WebhookDeliveryListOutput:
    """Response from listing webhook deliveries."""

    deliveries: list[WebhookDelivery] = field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False


@dataclass
class WebhookEventTypeInfo:
    """Webhook event catalog entry."""

    type: WebhookEventType
    category: str | None = None
    description: str | None = None


@dataclass
class WebhookEventTypesOutput:
    """Response from listing supported webhook event types."""

    events: list[WebhookEventTypeInfo] = field(default_factory=list)


@dataclass
class APIKey:
    """API key metadata.

    ``api_key`` is populated only on create responses.
    """

    key_id: str
    key_prefix: str
    name: str
    scopes: list[str] = field(default_factory=list)
    status: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    purpose: str | None = None
    tunnel_slug: str | None = None
    api_key: str | None = field(default=None, repr=False)


@dataclass
class APIKeyListOutput:
    """Response from listing API keys."""

    api_keys: list[APIKey] = field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False


@dataclass
class RedeemAccessCodeOutput:
    """Response from redeeming a public access code."""

    redirect_url: str


@dataclass
class AccessCode:
    """Access-code metadata.

    ``code`` is populated only on create responses.
    """

    access_code_id: str
    resource_id: str
    name: str | None = None
    status: str | None = None
    max_uses: int = 0
    use_count: int = 0
    created_at: datetime | None = None
    expires_at: datetime | None = None
    code: str | None = field(default=None, repr=False)


@dataclass
class AccessCodeListOutput:
    """Response from listing access codes."""

    access_codes: list[AccessCode] = field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False


@dataclass
class UsageCostEstimate:
    """Estimated usage cost for the current billing period."""

    currency: str
    amount_cents: int
    description: str


@dataclass
class CurrentPeriodUsage:
    """Current billing-period usage summary."""

    tier: QuotaPlan
    period_start: datetime | None
    period_end: datetime | None
    qurls_created: int
    active_qurls: int
    cost_estimate: UsageCostEstimate | None = None


@dataclass
class UsageDailyEntry:
    """Daily qURL creation count."""

    date: str
    qurls_created: int


@dataclass
class DailyUsage:
    """Daily usage breakdown for the current billing period."""

    tier: QuotaPlan
    period_start: datetime | None
    period_end: datetime | None
    daily: list[UsageDailyEntry] = field(default_factory=list)


@dataclass
class Customer:
    """Customer profile and billing settings."""

    tier: QuotaPlan
    spending_cap_cents: int
    current_period_usage_count: int
    frozen: bool
    frozen_reason: str | None = None


@dataclass
class CheckoutSession:
    """Stripe checkout session URL."""

    url: str


@dataclass
class PortalSession:
    """Stripe billing portal session URL."""

    url: str


@dataclass
class Invoice:
    """Billing invoice summary."""

    id: str
    amount_cents: int
    status: str
    created_at: datetime | None
    pdf_url: str | None = None


@dataclass
class InvoiceListOutput:
    """Response from listing billing invoices."""

    invoices: list[Invoice] = field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False


@dataclass
class ConnectorInstallationStats:
    """Connector installation activity counters."""

    resources: int = 0
    qurls: int = 0
    accesses_24h: int = 0
    accesses_7d: int = 0
    errors_24h: int = 0


@dataclass
class ConnectorInstallationCapabilities:
    """Connector installation capability flags."""

    configure: bool = False
    disconnect: bool = False
    reauth: bool = False
    view_activity: bool = False


@dataclass
class ConnectorInstallation:
    """Normalized connector installation summary."""

    installation_id: str
    plugin_id: str
    label: str
    subject_kind: str
    subject_display_name: str
    status: str
    installed_at: datetime | None
    last_activity_at: datetime | None = None
    stats: ConnectorInstallationStats | None = None
    capabilities: ConnectorInstallationCapabilities | None = None


@dataclass
class ConnectorInstallationListOutput:
    """Response from listing connector installations."""

    installations: list[ConnectorInstallation] = field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False


@dataclass
class NHPServerPeerInfo:
    """NHP server peer info returned to connector agents."""

    public_key_b64: str
    host: str
    port: int
    expire_time: int


@dataclass
class AgentBootstrapOutput:
    """Response from connector agent bootstrap."""

    agent_id: str
    registered_at: datetime | None
    nhp_server_peer: NHPServerPeerInfo


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

    type: str
    expires_in: str
    label: str
    one_time_use: bool
    max_sessions: int
    session_duration: str
    access_policy: AccessPolicy | dict[str, Any]
    custom_domain: str
