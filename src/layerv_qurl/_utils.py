"""Shared utilities for sync and async clients.

Underscore-prefixed helpers here (e.g. :func:`_validate_batch_create_shape`)
are **package-internal**, not strict module-private: they're imported
by both ``client.py`` and ``async_client.py`` to keep sync/async logic
in lockstep, but are excluded from the public ``from layerv_qurl import``
surface and carry no stability guarantees. Downstream consumers should
not import them directly.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import random
import re
from collections.abc import Iterable, Mapping, Set
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import quote
from uuid import uuid4

from layerv_qurl.errors import (
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    GoneError,
    NotFoundError,
    QURLError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from layerv_qurl.types import (
    QURL,
    AccessCode,
    AccessCodeListOutput,
    AccessGrant,
    AccessPolicy,
    AccessToken,
    AgentBootstrapOutput,
    AIAgentPolicy,
    APIKey,
    APIKeyListOutput,
    BatchCreateOutput,
    BatchItemError,
    BatchItemResult,
    CheckDetail,
    CheckoutSession,
    ConnectorInstallation,
    ConnectorInstallationCapabilities,
    ConnectorInstallationListOutput,
    ConnectorInstallationStats,
    CreateOutput,
    CurrentPeriodUsage,
    Customer,
    DailyUsage,
    DNSRecord,
    Domain,
    DomainListOutput,
    DomainVerifyOutput,
    Invoice,
    InvoiceListOutput,
    ListOutput,
    MintOutput,
    NHPServerPeerInfo,
    PortalSession,
    Quota,
    RateLimits,
    RedeemAccessCodeOutput,
    ResolveOutput,
    Resource,
    ResourceDetail,
    ResourceListOutput,
    Session,
    SessionListOutput,
    SessionTerminateOutput,
    Usage,
    UsageCostEstimate,
    UsageDailyEntry,
    Webhook,
    WebhookDelivery,
    WebhookDeliveryListOutput,
    WebhookEventTypeInfo,
    WebhookEventTypesOutput,
    WebhookListOutput,
    _parse_dt,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx

logger = logging.getLogger("layerv_qurl")
_T = TypeVar("_T")

DEFAULT_BASE_URL = "https://api.layerv.ai"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
RETRYABLE_STATUS = {429, 502, 503, 504}
# POST requests still only retry rate limits: resolve can consume one-time
# tokens after an NHP knock failure, and service errors are not cached.
RETRYABLE_STATUS_POST = {429}
# DELETE keeps the wider HTTP retry set without an idempotency key because
# repeated deletes are safe by HTTP semantics; create/update mutations need
# service-side replay protection.
IDEMPOTENCY_METHODS = {"POST", "PATCH"}

_RESOURCE_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")

_ERROR_CLASS_MAP: dict[int, type[QURLError]] = {
    400: ValidationError,
    401: AuthenticationError,
    403: AuthorizationError,
    404: NotFoundError,
    409: ConflictError,
    410: GoneError,
    422: ValidationError,
    429: RateLimitError,
}


class UnsetType:
    """Sentinel for fields where omitted and explicit null differ."""


UNSET = UnsetType()


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
    """Serialize a single value for JSON, handling dataclasses/datetimes/lists/dicts.

    Unlike :func:`build_body` (which strips top-level ``None`` values so the
    request body only carries fields the caller explicitly set), this
    recursive helper preserves ``None`` values inside lists and nested
    dicts. This matters because some API fields use explicit ``null`` as a
    signalling value (e.g. ``"access_policy": {"ai_agent_policy": null}``
    to clear a policy). Dataclass fields still skip ``None`` because the
    dataclass itself distinguishes "unset" from "explicitly null".
    """
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
    if isinstance(v, dict):
        # Preserve explicit None inside nested dicts — callers who want
        # "drop this field" should omit it from the dict, not set it to None.
        return {k: _serialize_value(val) for k, val in v.items()}
    return v


def build_body(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build a request body dict from kwargs, dropping top-level None values.

    Only strips ``None`` at the top level — nested values are preserved
    as-is by :func:`_serialize_value`. Always returns a dict (at least
    ``{}``) so POST/PATCH endpoints receive a valid JSON body. Nested
    dataclasses are recursively serialized to dicts.
    """
    body: dict[str, Any] = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        body[k] = _serialize_value(v)
    return body


def build_string_list(value: Any, field: str) -> list[str]:
    """Build a non-empty list of strings without accepting strings or mappings."""
    if isinstance(value, str):
        raise ValueError(f"{field}: must be an iterable of strings, not a string")
    if isinstance(value, Mapping):
        raise ValueError(f"{field}: must be an iterable of strings, not a mapping")
    if isinstance(value, Set):
        raise ValueError(f"{field}: must be an ordered iterable of strings, not a set")
    if not isinstance(value, Iterable):
        raise ValueError(f"{field}: must be an iterable of strings")
    items = list(value)
    if not items:
        raise ValueError(f"{field}: cannot be empty; pass at least one value")
    if any(not isinstance(item, str) for item in items):
        raise ValueError(f"{field}: all values must be strings")
    return items


def build_query_params(pairs: dict[str, Any]) -> dict[str, str]:
    """Build query params from optional values, dropping ``None`` values."""
    return {k: _query_value(v) for k, v in pairs.items() if v is not None}


def _query_value(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def idempotency_headers(
    idempotency_key: str | None, *, min_length: int = 1
) -> dict[str, str] | None:
    """Build the optional Idempotency-Key header."""
    if idempotency_key is None:
        return None
    if len(idempotency_key) < min_length:
        raise ValueError(
            f"idempotency_key: must be at least {min_length} characters "
            f"(got {len(idempotency_key)})"
        )
    if len(idempotency_key) > 256:
        raise ValueError(
            f"idempotency_key: must be 256 characters or fewer (got {len(idempotency_key)})"
        )
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in idempotency_key):
        raise ValueError("idempotency_key: must not contain control characters")
    return {"Idempotency-Key": idempotency_key}


def ensure_mutation_idempotency(method: str, headers: dict[str, str]) -> None:
    """Generate a per-call idempotency key for supported mutating requests."""
    if method.upper() not in IDEMPOTENCY_METHODS:
        return
    if any(key.lower() == "idempotency-key" for key in headers):
        return
    headers["Idempotency-Key"] = str(uuid4())


def _meta_page(meta: dict[str, Any] | None) -> tuple[str | None, bool]:
    if not meta:
        return None, False
    return meta.get("next_cursor"), meta.get("has_more", False)


def _parse_list_items(data: Any, parser: Callable[[dict[str, Any]], _T]) -> list[_T]:
    if not isinstance(data, list):
        return []
    items: list[_T] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            items.append(parser(item))
        except KeyError as exc:
            missing = exc.args[0] if exc.args else "<unknown>"
            parser_name = getattr(parser, "__name__", "parser")
            logger.debug(
                "Skipping malformed API list item in %s: missing %r",
                parser_name,
                missing,
            )
            continue
    return items


def domain_path_segment(domain: str) -> str:
    """Encode a custom domain for use as one URL path segment."""
    return quote(domain, safe="")


def require_nonempty_update(body: dict[str, Any], method: str, fields: str) -> None:
    """Raise a consistent error when an update method has no fields."""
    if not body:
        raise ValueError(f"{method}: at least one field ({fields}) must be provided")


# ---- Spec-derived input validation --------------------------------------
# These mirror the constraints documented on each request schema in
# qurl/api/openapi.yaml so obvious mistakes fail fast with a ValueError
# instead of round-tripping to the API and coming back as a generic 400.

MAX_TARGET_URL = 2048
MAX_LABEL = 500
MAX_DESCRIPTION = 500
MAX_CUSTOM_DOMAIN = 253
MAX_MAX_SESSIONS = 1000
MAX_TAGS = 10
MAX_TAG_LENGTH = 50
# Tag item pattern mirrored from the OpenAPI spec schema
# `UpdateQurlRequest.tags.items.pattern` in qurl/api/openapi.yaml.
# Keep in lockstep with the spec — if the server pattern ever relaxes
# (e.g. allowing colons) the SDK must widen this regex or it will
# reject strings the API would otherwise accept.
_TAG_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9 _-]*$")
_ALIAS_PATTERN = re.compile(r"^[a-z][a-z0-9-]{1,62}[a-z0-9]$")
_JWT_LIKE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")
_RESERVED_ALIASES = frozenset(
    {
        "setalias",
        "unsetalias",
        "get",
        "aliases",
        "admin",
        "claim",
        "register",
        "agent",
        "knock",
        "token",
        "bootstrap",
        "status",
        "info",
        "whoami",
        "revoke",
        "delete",
        "enable",
        "disable",
        "audit",
        "version",
        "health",
        "create",
        "new",
        "list",
        "help",
        "all",
        "frpc",
        "frps",
        "tunnel",
        "qurl",
        "me",
        "*",
    }
)
RESOURCE_ID_PREFIX = "r_"
# target_url must use an http(s) scheme per the API's SSRF protection.
# This is a cheap client-side sanity check — the server is still the
# authoritative validator (e.g. it rejects localhost, cloud metadata,
# and private-range hosts; the SDK doesn't need to duplicate that).
_ALLOWED_URL_SCHEMES = ("http://", "https://")


def _require_max_length(value: str | None, field_name: str, maximum: int) -> None:
    if value is not None and len(value) > maximum:
        raise ValueError(
            f"{field_name}: must be {maximum} characters or fewer (got {len(value)})"
        )


def validate_required_string(value: str, field_name: str) -> None:
    """Validate required string request fields that have no richer local schema."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name}: must be a non-empty string")


def validate_alias(value: str | None, field_name: str = "alias") -> None:
    """Validate resource alias strings against the qurl-service contract."""
    if value is None:
        return
    if not isinstance(value, str) or not _ALIAS_PATTERN.match(value):
        raise ValueError(
            f"{field_name}: must be 3-64 lowercase alphanumeric characters or "
            "hyphens, start with a letter, and end alphanumeric"
        )
    if value in _RESERVED_ALIASES:
        raise ValueError(f"{field_name}: reserved alias")


def validate_nonnegative_int(value: int, field_name: str) -> None:
    """Validate required non-negative integer request fields."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name}: must be a non-negative integer")


def _require_http_url(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.startswith(_ALLOWED_URL_SCHEMES):
        raise ValueError(
            f"{field_name}: must start with http:// or https:// (got {repr(value)[:40]})"
        )
    _require_max_length(value, field_name, MAX_TARGET_URL)


def _require_max_sessions_in_range(value: int | None) -> None:
    if value is None:
        return
    # `bool` is a subclass of `int` in Python (True == 1, False == 0), so
    # a caller passing `max_sessions=True` would sneak through an
    # `isinstance(value, int)` check alone. Reject bool explicitly so
    # obvious type confusion fails loudly instead of silently meaning 1.
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"max_sessions: must be an integer (got {type(value).__name__})")
    if value < 0 or value > MAX_MAX_SESSIONS:
        raise ValueError(
            f"max_sessions: must be an integer between 0 and {MAX_MAX_SESSIONS} (got {value})"
        )


def _require_valid_tags(tags: list[str] | None) -> None:
    if tags is None:
        return
    if len(tags) > MAX_TAGS:
        raise ValueError(f"tags: max {MAX_TAGS} items allowed (got {len(tags)})")
    for tag in tags:
        if not isinstance(tag, str) or len(tag) < 1 or len(tag) > MAX_TAG_LENGTH:
            raise ValueError(
                f"tags: each tag must be 1-{MAX_TAG_LENGTH} characters "
                f"(got {len(tag) if isinstance(tag, str) else type(tag).__name__})"
            )
        if not _TAG_PATTERN.match(tag):
            raise ValueError(
                "tags: each tag must start with an alphanumeric and contain only "
                "letters, numbers, spaces, underscores, or hyphens"
            )


def validate_create_input(
    *,
    target_url: str,
    label: str | None = None,
    max_sessions: int | None = None,
    custom_domain: str | None = None,
) -> None:
    """Validate a single create_qurl input against spec-documented constraints.

    Used by both ``create()`` and ``batch_create()`` (for each item).
    Raises ``ValueError`` on constraint violations so obvious mistakes
    fail fast instead of round-tripping to the API.
    """
    # `repr(...)[:40]` inside `_require_http_url` avoids the old
    # `target_url[:32]!r` trap for non-subscriptable inputs (None, int,
    # bool, ...), surfacing a clean ValueError instead of TypeError.
    _require_http_url(target_url, "target_url")
    _require_max_length(label, "label", MAX_LABEL)
    _require_max_length(custom_domain, "custom_domain", MAX_CUSTOM_DOMAIN)
    _require_max_sessions_in_range(max_sessions)


def validate_update_input(
    *,
    description: str | None = None,
    tags: list[str] | None = None,
    custom_domain: str | None = None,
) -> None:
    """Validate update_qurl input against spec-documented constraints."""
    _require_max_length(description, "description", MAX_DESCRIPTION)
    _require_max_length(custom_domain, "custom_domain", MAX_CUSTOM_DOMAIN)
    _require_valid_tags(tags)


def validate_domain_input(domain: str) -> None:
    """Validate custom-domain strings before using them in URL path segments."""
    validate_required_string(domain, "domain")
    _require_max_length(domain, "domain", MAX_CUSTOM_DOMAIN)


def validate_webhook_url(url: str) -> None:
    """Validate webhook callback URLs with the same basic HTTP(S) guard."""
    _require_http_url(url, "url")


def validate_mint_input(
    *,
    label: str | None = None,
    max_sessions: int | None = None,
) -> None:
    """Validate mint_link input against spec-documented constraints."""
    _require_max_length(label, "label", MAX_LABEL)
    _require_max_sessions_in_range(max_sessions)


def require_resource_id_prefix(resource_id: str, operation: str = "delete") -> None:
    """Enforce the ``r_`` prefix on endpoints that only accept resource IDs.

    Per the OpenAPI spec, ``DELETE /v1/qurls/:id`` explicitly requires a
    resource ID (r_ prefix) — the token-scoped endpoint is
    ``DELETE /v1/resources/:id/qurls/:qurl_id``. Catch the common mistake
    of passing a ``q_`` display ID here with a clear client-side error.
    """
    if not resource_id.startswith(RESOURCE_ID_PREFIX):
        # Don't echo the raw ID — even truncated to 16 chars it may
        # contain caller-sensitive data that ends up in error logs.
        # Echo only the 2-char prefix so the caller sees which kind of
        # ID they passed without leaking the value.
        observed_prefix = resource_id[:2]
        # TODO: update the "not yet available in this SDK version"
        # wording once a token-scoped `revoke_token()` / equivalent
        # method lands on the client. Until then, the error points
        # users at the API-level endpoint without a concrete SDK call.
        raise ValueError(
            f"{operation}: only resource IDs ({RESOURCE_ID_PREFIX} prefix) are accepted — "
            f"got an ID starting with {observed_prefix!r}. To revoke a single "
            "access token, use the token-scoped revoke endpoint (not yet "
            "available in this SDK version)."
        )


def _parse_access_policy(data: dict[str, Any] | None) -> AccessPolicy | None:
    """Parse an AccessPolicy from API response data."""
    if data is None:
        return None
    ai_policy = None
    ap = data.get("ai_agent_policy")
    # Guard against non-dict values (e.g. API returning a bare string
    # or boolean for ai_agent_policy). Without this, `.get("block_all")`
    # would raise AttributeError. Consistent with the defensive posture
    # in `_validate_batch_create_shape`.
    if isinstance(ap, dict):
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


def parse_access_token(data: dict[str, Any]) -> AccessToken:
    """Parse a qURL token summary."""
    return AccessToken(
        qurl_id=data["qurl_id"],
        status=data["status"],
        one_time_use=data.get("one_time_use", False),
        max_sessions=data.get("max_sessions", 0),
        session_duration=data.get("session_duration", 0),
        use_count=data.get("use_count", 0),
        label=data.get("label"),
        qurl_site=data.get("qurl_site"),
        access_policy=_parse_access_policy(data.get("access_policy")),
        created_at=_parse_dt(data.get("created_at")),
        expires_at=_parse_dt(data.get("expires_at")),
    )


def _parse_access_token_preview_items(data: Any) -> list[AccessToken]:
    if not isinstance(data, list):
        return []
    return [
        parse_access_token(item)
        for item in data
        if isinstance(item, dict) and "qurl_id" in item and "status" in item
    ]


def parse_qurl(data: dict[str, Any]) -> QURL:
    """Parse a qURL resource from API response data."""
    tokens = None
    # API returns "qurls" array; SDK exposes as "access_tokens" for clarity.
    raw_tokens = data.get("qurls") if "qurls" in data else data.get("access_tokens")
    if raw_tokens is not None:
        tokens = _parse_list_items(raw_tokens, parse_access_token)
    return QURL(
        resource_id=data["resource_id"],
        target_url=data.get("target_url"),
        status=data["status"],
        created_at=_parse_dt(data.get("created_at")),
        expires_at=_parse_dt(data.get("expires_at")),
        description=data.get("description"),
        tags=data.get("tags") or [],
        qurl_site=data.get("qurl_site"),
        custom_domain=data.get("custom_domain"),
        slug=data.get("slug"),
        qurl_count=data.get("qurl_count"),
        access_tokens=tokens,
    )


def parse_create_output(data: dict[str, Any]) -> CreateOutput:
    """Parse a CreateOutput from API response data."""
    # Normalize empty-string `qurl_id` → None for idiomatic truthiness
    # checks. Intentionally asymmetric with `label` (preserved as-is):
    # `""` is never a meaningful identifier but IS a meaningful "cleared"
    # value for user-facing metadata.
    qurl_id = data.get("qurl_id") or None
    return CreateOutput(
        resource_id=data["resource_id"],
        qurl_link=data["qurl_link"],
        qurl_site=data["qurl_site"],
        expires_at=_parse_dt(data.get("expires_at")),
        qurl_id=qurl_id,
        label=data.get("label"),
        branded_domain=data.get("branded_domain"),
        resource_type=data.get("type"),
    )


def parse_mint_output(data: dict[str, Any]) -> MintOutput:
    """Parse a MintOutput from API response data."""
    return MintOutput(
        qurl_link=data["qurl_link"],
        qurl_id=data.get("qurl_id") or None,
        expires_at=_parse_dt(data.get("expires_at")),
        branded_domain=data.get("branded_domain"),
        resource_type=data.get("type"),
    )


def _parse_access_grant(data: dict[str, Any] | None) -> AccessGrant | None:
    if data is None:
        return None
    return AccessGrant(
        expires_in=data["expires_in"],
        granted_at=_parse_dt(data.get("granted_at")),
        src_ip=data.get("src_ip", ""),
    )


def parse_resolve_output(data: dict[str, Any]) -> ResolveOutput:
    """Parse a ResolveOutput from API response data."""
    return ResolveOutput(
        target_url=data.get("target_url"),
        resource_id=data["resource_id"],
        access_grant=_parse_access_grant(data.get("access_grant")),
    )


def parse_quota(data: dict[str, Any]) -> Quota:
    """Parse a Quota from API response data."""
    return Quota(
        # Fall back to the same sentinel the dataclass default uses
        # (see ``Quota.plan`` in types.py) so a malformed API response
        # that omits the field produces a consistent "not-yet-populated"
        # value regardless of whether the Quota was constructed via
        # parse_quota or directly. In practice the /v1/quota endpoint
        # always returns a populated plan string, so this fallback is
        # only hit for malformed responses or internal bootstrap paths.
        plan=data.get("plan", "unknown"),
        period_start=_parse_dt(data.get("period_start")),
        period_end=_parse_dt(data.get("period_end")),
        rate_limits=_parse_rate_limits(data.get("rate_limits")),
        usage=_parse_usage_block(data.get("usage")),
    )


def _parse_rate_limits(data: dict[str, Any] | None) -> RateLimits | None:
    if data is None:
        return None
    return RateLimits(
        create_per_minute=data.get("create_per_minute", 0),
        create_per_hour=data.get("create_per_hour", 0),
        list_per_minute=data.get("list_per_minute", 0),
        resolve_per_minute=data.get("resolve_per_minute", 0),
        max_active_qurls=data.get("max_active_qurls", 0),
        max_tokens_per_qurl=data.get("max_tokens_per_qurl", 0),
        max_expiry_seconds=data.get("max_expiry_seconds", 0),
    )


def _parse_usage_block(data: dict[str, Any] | None) -> Usage | None:
    if data is None:
        return None
    return Usage(
        qurls_created=data.get("qurls_created", 0),
        active_qurls=data.get("active_qurls", 0),
        # Nullable per the API spec — the field is null when
        # max_active_qurls is unlimited.
        active_qurls_percent=data.get("active_qurls_percent"),
        total_accesses=data.get("total_accesses", 0),
    )


def parse_list_output(data: Any, meta: dict[str, Any] | None) -> ListOutput:
    """Parse a ListOutput from API response data."""
    next_cursor, has_more = _meta_page(meta)
    qurls = _parse_list_items(data, parse_qurl)
    return ListOutput(qurls=qurls, next_cursor=next_cursor, has_more=has_more)


def _parse_resource(data: dict[str, Any], *, strict_identity: bool) -> Resource:
    resource_id = data["resource_id"] if strict_identity else data.get("resource_id", "")
    status = data["status"] if strict_identity else data.get("status", "unknown")
    return Resource(
        resource_id=resource_id,
        status=status,
        resource_type=data.get("type"),
        target_url=data.get("target_url"),
        knock_resource_id=data.get("knock_resource_id"),
        description=data.get("description"),
        tags=data.get("tags") or [],
        custom_domain=data.get("custom_domain"),
        alias=data.get("alias"),
        slug=data.get("slug"),
        preserve_host=data.get("preserve_host", False),
        session_duration_cap=data.get("session_duration_cap"),
        qurl_count=data.get("qurl_count"),
        created_at=_parse_dt(data.get("created_at")),
        expires_at=_parse_dt(data.get("expires_at")),
        tombstoned_at=_parse_dt(data.get("tombstoned_at")),
    )


def parse_resource(data: dict[str, Any]) -> Resource:
    """Parse a resource-management API resource."""
    return _parse_resource(data, strict_identity=True)


def parse_resource_detail(data: dict[str, Any]) -> ResourceDetail:
    """Parse a resource detail response."""
    raw_resource = data.get("resource")
    resource = _parse_resource(
        raw_resource if isinstance(raw_resource, dict) else {},
        strict_identity=False,
    )
    qurls = _parse_access_token_preview_items(data.get("qurls"))
    return ResourceDetail(resource=resource, qurls=qurls)


def parse_resource_list_output(
    data: Any, meta: dict[str, Any] | None
) -> ResourceListOutput:
    """Parse a resource list response."""
    next_cursor, has_more = _meta_page(meta)
    resources = _parse_list_items(data, parse_resource)
    return ResourceListOutput(resources=resources, next_cursor=next_cursor, has_more=has_more)


def parse_session(data: dict[str, Any]) -> Session:
    """Parse an active resource session."""
    return Session(
        session_id=data["session_id"],
        qurl_id=data.get("qurl_id"),
        src_ip=data.get("src_ip"),
        user_agent=data.get("user_agent"),
        created_at=_parse_dt(data.get("created_at")),
        last_seen_at=_parse_dt(data.get("last_seen_at")),
    )


def parse_session_list_output(data: Any) -> SessionListOutput:
    """Parse an active session list response."""
    sessions = _parse_list_items(data, parse_session)
    return SessionListOutput(sessions=sessions)


def parse_session_terminate_output(data: dict[str, Any] | None) -> SessionTerminateOutput:
    """Parse a session termination response."""
    return SessionTerminateOutput(terminated=(data or {}).get("terminated", 0))


def _parse_dns_record(data: dict[str, Any]) -> DNSRecord:
    return DNSRecord(
        type=data.get("type", ""),
        name=data.get("name", ""),
        value=data.get("value", ""),
        verified=data.get("verified", False),
    )


# Object identity fields stay strict so malformed rows fail at the parser,
# while ancillary fields use defaults for partial-payload tolerance.
def parse_domain(data: dict[str, Any]) -> Domain:
    """Parse a custom domain response."""
    return Domain(
        domain=data["domain"],
        status=data["status"],
        verification_token=data.get("verification_token"),
        token_expires_at=_parse_dt(data.get("token_expires_at")),
        acme_cname_target=data.get("acme_cname_target"),
        created_at=_parse_dt(data.get("created_at")),
        verified_at=_parse_dt(data.get("verified_at")),
        activated_at=_parse_dt(data.get("activated_at")),
        ready_for_qurls=data.get("ready_for_qurls", False),
        dns_records=[
            _parse_dns_record(r)
            for r in data.get("dns_records") or []
            if isinstance(r, dict)
        ],
    )


def parse_domain_list_output(data: Any, meta: dict[str, Any] | None) -> DomainListOutput:
    """Parse a custom-domain list response."""
    next_cursor, has_more = _meta_page(meta)
    domains = _parse_list_items(data, parse_domain)
    return DomainListOutput(domains=domains, next_cursor=next_cursor, has_more=has_more)


def _parse_check_detail(data: dict[str, Any]) -> CheckDetail:
    return CheckDetail(
        verified=data.get("verified", False),
        error=data.get("error"),
        found=data.get("found"),
    )


def parse_domain_verify_output(data: dict[str, Any]) -> DomainVerifyOutput:
    """Parse custom-domain verification output."""
    checks = {
        name: _parse_check_detail(check)
        for name, check in data.get("checks", {}).items()
        if isinstance(check, dict)
    }
    return DomainVerifyOutput(
        domain=data["domain"],
        status=data["status"],
        checks=checks,
    )


def parse_webhook(data: dict[str, Any]) -> Webhook:
    """Parse webhook subscription data."""
    return Webhook(
        webhook_id=data["webhook_id"],
        owner_id=data.get("owner_id"),
        url=data.get("url", ""),
        events=data.get("events") or [],
        status=data.get("status"),
        description=data.get("description"),
        created_at=_parse_dt(data.get("created_at")),
        updated_at=_parse_dt(data.get("updated_at")),
        failure_count=data.get("failure_count", 0),
        last_delivery_success=data.get("last_delivery_success"),
        last_delivery_time=data.get("last_delivery_time"),
        secret=data.get("secret"),
    )


def parse_webhook_list_output(data: Any, meta: dict[str, Any] | None) -> WebhookListOutput:
    """Parse a webhook list response."""
    next_cursor, has_more = _meta_page(meta)
    webhooks = _parse_list_items(data, parse_webhook)
    return WebhookListOutput(webhooks=webhooks, next_cursor=next_cursor, has_more=has_more)


def parse_webhook_delivery(data: dict[str, Any]) -> WebhookDelivery:
    """Parse webhook delivery data."""
    return WebhookDelivery(
        delivery_id=data["delivery_id"],
        webhook_id=data.get("webhook_id"),
        event_type=data.get("event_type"),
        status=data.get("status"),
        response_code=data.get("response_code"),
        response_body=data.get("response_body"),
        error_message=data.get("error_message"),
        duration_ms=data.get("duration_ms"),
        retry_count=data.get("retry_count", 0),
        created_at=_parse_dt(data.get("created_at")),
        completed_at=_parse_dt(data.get("completed_at")),
    )


def parse_webhook_delivery_list_output(
    data: Any, meta: dict[str, Any] | None
) -> WebhookDeliveryListOutput:
    """Parse a webhook delivery list response."""
    next_cursor, has_more = _meta_page(meta)
    deliveries = _parse_list_items(data, parse_webhook_delivery)
    return WebhookDeliveryListOutput(
        deliveries=deliveries, next_cursor=next_cursor, has_more=has_more
    )


def _parse_event_type_info(data: dict[str, Any]) -> WebhookEventTypeInfo:
    return WebhookEventTypeInfo(
        type=data.get("type", ""),
        category=data.get("category"),
        description=data.get("description"),
    )


def parse_webhook_event_types_output(data: Any) -> WebhookEventTypesOutput:
    """Parse supported webhook event types."""
    events = [
        event
        for event in _parse_list_items(data, _parse_event_type_info)
        if event.type
    ]
    return WebhookEventTypesOutput(events=events)


def parse_api_key(data: dict[str, Any]) -> APIKey:
    """Parse API key metadata."""
    return APIKey(
        key_id=data["key_id"],
        key_prefix=data["key_prefix"],
        name=data.get("name", ""),
        scopes=data.get("scopes") or [],
        status=data.get("status"),
        created_at=_parse_dt(data.get("created_at")),
        updated_at=_parse_dt(data.get("updated_at")),
        last_used_at=_parse_dt(data.get("last_used_at")),
        expires_at=_parse_dt(data.get("expires_at")),
        purpose=data.get("purpose"),
        tunnel_slug=data.get("tunnel_slug"),
        api_key=data.get("api_key"),
    )


def parse_api_key_list_output(data: Any, meta: dict[str, Any] | None) -> APIKeyListOutput:
    """Parse an API key list response."""
    next_cursor, has_more = _meta_page(meta)
    api_keys = _parse_list_items(data, parse_api_key)
    return APIKeyListOutput(api_keys=api_keys, next_cursor=next_cursor, has_more=has_more)


def parse_redeem_access_code_output(data: dict[str, Any]) -> RedeemAccessCodeOutput:
    """Parse public access-code redemption output."""
    return RedeemAccessCodeOutput(redirect_url=data["redirect_url"])


def parse_access_code(data: dict[str, Any]) -> AccessCode:
    """Parse access-code metadata."""
    return AccessCode(
        access_code_id=data["access_code_id"],
        resource_id=data["resource_id"],
        name=data.get("name"),
        status=data.get("status"),
        max_uses=data.get("max_uses", 0),
        use_count=data.get("use_count", 0),
        created_at=_parse_dt(data.get("created_at")),
        expires_at=_parse_dt(data.get("expires_at")),
        code=data.get("code"),
    )


def parse_access_code_list_output(
    data: Any, meta: dict[str, Any] | None
) -> AccessCodeListOutput:
    """Parse an access-code list response."""
    next_cursor, has_more = _meta_page(meta)
    access_codes = _parse_list_items(data, parse_access_code)
    return AccessCodeListOutput(
        access_codes=access_codes, next_cursor=next_cursor, has_more=has_more
    )


def _parse_usage_cost(data: dict[str, Any] | None) -> UsageCostEstimate | None:
    if data is None:
        return None
    return UsageCostEstimate(
        currency=data.get("currency", ""),
        amount_cents=data.get("amount_cents", 0),
        description=data.get("description", ""),
    )


def parse_current_period_usage(data: dict[str, Any]) -> CurrentPeriodUsage:
    """Parse current-period usage."""
    return CurrentPeriodUsage(
        tier=data.get("tier", "unknown"),
        period_start=_parse_dt(data.get("period_start")),
        period_end=_parse_dt(data.get("period_end")),
        qurls_created=data.get("qurls_created", 0),
        active_qurls=data.get("active_qurls", 0),
        cost_estimate=_parse_usage_cost(data.get("cost_estimate")),
    )


def parse_daily_usage(data: dict[str, Any]) -> DailyUsage:
    """Parse daily usage breakdown."""
    return DailyUsage(
        tier=data.get("tier", "unknown"),
        period_start=_parse_dt(data.get("period_start")),
        period_end=_parse_dt(data.get("period_end")),
        daily=[
            UsageDailyEntry(
                date=item.get("date", ""),
                qurls_created=item.get("qurls_created", 0),
            )
            for item in data.get("daily") or []
            if isinstance(item, dict)
        ],
    )


def parse_customer(data: dict[str, Any]) -> Customer:
    """Parse customer profile data."""
    # The current OpenAPI schema is an integer; tolerate older/future object
    # wrappers without letting bool sneak through Python's int hierarchy.
    current_period_usage = data.get("current_period_usage", 0)
    if isinstance(current_period_usage, bool):
        current_period_usage_count = 0
    elif isinstance(current_period_usage, int):
        current_period_usage_count = current_period_usage
    elif isinstance(current_period_usage, dict):
        count = current_period_usage.get("count", 0)
        current_period_usage_count = (
            count if isinstance(count, int) and not isinstance(count, bool) else 0
        )
    else:
        current_period_usage_count = 0
    return Customer(
        tier=data.get("tier", "unknown"),
        spending_cap_cents=data.get("spending_cap_cents", 0),
        # Wire name is `current_period_usage`; SDK suffixes `_count` to
        # avoid confusion with the richer CurrentPeriodUsage response.
        current_period_usage_count=current_period_usage_count,
        frozen=data.get("frozen", False),
        frozen_reason=data.get("frozen_reason"),
    )


def parse_checkout_session(data: dict[str, Any]) -> CheckoutSession:
    """Parse a Stripe checkout session response."""
    return CheckoutSession(url=data["url"])


def parse_portal_session(data: dict[str, Any]) -> PortalSession:
    """Parse a Stripe billing portal session response."""
    return PortalSession(url=data["url"])


def parse_invoice(data: dict[str, Any]) -> Invoice:
    """Parse a billing invoice summary."""
    return Invoice(
        id=data.get("id", ""),
        amount_cents=data.get("amount_cents", 0),
        status=data.get("status", ""),
        created_at=_parse_dt(data.get("created_at")),
        pdf_url=data.get("pdf_url"),
    )


def parse_invoice_list_output(
    data: Any, meta: dict[str, Any] | None
) -> InvoiceListOutput:
    """Parse billing invoice list output."""
    next_cursor, has_more = _meta_page(meta)
    # qurl-service returns billing invoices as {"invoices": [...]}, unlike
    # the top-level array shape used by most list endpoints.
    if isinstance(data, dict):
        raw_invoices = data.get("invoices")
        if raw_invoices is None and data:
            logger.debug(
                "parse_invoice_list_output: expected data.invoices, got keys=%s",
                list(data.keys()),
            )
    else:
        raw_invoices = None
        if data:
            logger.debug(
                "parse_invoice_list_output: expected object with invoices, got %s",
                type(data).__name__,
            )
    invoices = _parse_list_items(raw_invoices, parse_invoice)
    return InvoiceListOutput(invoices=invoices, next_cursor=next_cursor, has_more=has_more)


def _parse_connector_stats(data: dict[str, Any] | None) -> ConnectorInstallationStats | None:
    if data is None:
        return None
    return ConnectorInstallationStats(
        resources=data.get("resources", 0),
        qurls=data.get("qurls", 0),
        accesses_24h=data.get("accesses_24h", 0),
        accesses_7d=data.get("accesses_7d", 0),
        errors_24h=data.get("errors_24h", 0),
    )


def _parse_connector_capabilities(
    data: dict[str, Any] | None,
) -> ConnectorInstallationCapabilities | None:
    if data is None:
        return None
    return ConnectorInstallationCapabilities(
        configure=data.get("configure", False),
        disconnect=data.get("disconnect", False),
        reauth=data.get("reauth", False),
        view_activity=data.get("view_activity", False),
    )


def parse_connector_installation(data: dict[str, Any]) -> ConnectorInstallation:
    """Parse a connector installation summary."""
    return ConnectorInstallation(
        installation_id=data["installation_id"],
        plugin_id=data["plugin_id"],
        label=data["label"],
        subject_kind=data["subject_kind"],
        subject_display_name=data["subject_display_name"],
        status=data["status"],
        installed_at=_parse_dt(data.get("installed_at")),
        last_activity_at=_parse_dt(data.get("last_activity_at")),
        stats=_parse_connector_stats(data.get("stats")),
        capabilities=_parse_connector_capabilities(data.get("capabilities")),
    )


def parse_connector_installation_list_output(
    data: Any, meta: dict[str, Any] | None
) -> ConnectorInstallationListOutput:
    """Parse connector installation list output."""
    next_cursor, has_more = _meta_page(meta)
    installations = _parse_list_items(data, parse_connector_installation)
    return ConnectorInstallationListOutput(
        installations=installations, next_cursor=next_cursor, has_more=has_more
    )


def parse_agent_bootstrap_output(data: dict[str, Any]) -> AgentBootstrapOutput:
    """Parse connector agent bootstrap output."""
    raw_peer = data.get("nhp_server_peer")
    peer = raw_peer if isinstance(raw_peer, dict) else {}
    return AgentBootstrapOutput(
        agent_id=data.get("agent_id", ""),
        registered_at=_parse_dt(data.get("registered_at")),
        nhp_server_peer=NHPServerPeerInfo(
            public_key_b64=peer.get("public_key_b64", ""),
            host=peer.get("host", ""),
            port=peer.get("port", 0),
            expire_time=peer.get("expire_time", 0),
        ),
    )


def _validate_batch_create_shape(data: Any) -> None:
    """Defense-in-depth structural check for batch_create responses.

    The ``batch_create`` endpoint whitelists HTTP 400 into the success
    path so the populated ``BatchCreateOutput`` body is surfaced instead
    of being swallowed by the generic error path. If the API ever returns
    400 with a *different* body (e.g., a plain validation error envelope,
    a proxy error, or malformed JSON), ``parse_batch_create_output`` would
    silently produce ``(succeeded=0, failed=0, results=[])`` via its
    ``.get()`` defaults — the caller would get no indication anything went
    wrong.

    Raise a clear error instead when the shape doesn't match. The error
    message intentionally does not embed the raw body — an unexpected
    body could contain sensitive data (auth details, request echoes)
    and error strings may end up in client-side logs. Structural hints
    (type name, top-level keys) are emitted at DEBUG level for
    production triage, which is safe because JSON key names come from
    the API's published schema — not user-supplied data.
    """

    def _fail(reason: str, *, top_level_keys: list[str] | None = None) -> ValidationError:
        # DEBUG log carries structural hints (type + top-level key names
        # only — JSON keys come from the published schema, not user
        # data) so operators can triage shape-guard trips without
        # leaking raw body content into logs.
        logger.debug(
            "batch_create shape guard tripped: %s (type=%s, top_level_keys=%s)",
            reason,
            type(data).__name__,
            top_level_keys,
        )
        # Uses `ValidationError` (subclass) not bare `QURLError` so
        # `except ValidationError` catches shape-guard trips; `code=
        # "unexpected_response"` distinguishes from client-side
        # preflight (`client_validation`). `status=0` is the SDK
        # convention for all client-detected failures (not real HTTP
        # status). See qurl-typescript's `unexpectedResponseError`.
        return ValidationError(
            status=0,
            code="unexpected_response",
            title="Unexpected Response",
            detail="Unexpected response shape from POST /v1/qurls/batch",
        )

    if not isinstance(data, dict):
        raise _fail("not a dict")
    top_keys = sorted(data.keys())
    # `bool` is a subclass of `int` in Python, so a response with
    # `"succeeded": True` would silently pass an `isinstance(..., int)`
    # check and then slip a truthy bool into the counts. Reject
    # explicitly — matches the same guard in `_require_max_sessions_in_range`.
    succeeded = data.get("succeeded")
    failed = data.get("failed")
    if (
        not isinstance(succeeded, int)
        or isinstance(succeeded, bool)
        or not isinstance(failed, int)
        or isinstance(failed, bool)
    ):
        raise _fail("succeeded/failed missing or wrong type", top_level_keys=top_keys)
    if not isinstance(data.get("results"), list):
        raise _fail("results missing or not a list", top_level_keys=top_keys)
    results = data["results"]
    # Arithmetic invariant: `succeeded + failed` must equal the number of
    # result entries. A mismatch indicates either a proxy/middleware
    # mangled the response or the API returned inconsistent counts —
    # both cases warrant raising rather than trusting the data.
    if succeeded + failed != len(results):
        raise _fail(
            f"counts/results length mismatch (succeeded={succeeded}, "
            f"failed={failed}, len(results)={len(results)})",
            top_level_keys=top_keys,
        )
    # Each entry must carry a boolean `success` discriminant so consumers
    # can reliably branch on it — anything else would break the
    # BatchItemResult contract. Deeper per-field validation is
    # intentionally left to the API.
    for i, entry in enumerate(results):
        if not isinstance(entry, dict) or not isinstance(entry.get("success"), bool):
            raise _fail(
                f"results[{i}] missing boolean 'success' discriminant",
                top_level_keys=top_keys,
            )


def parse_batch_create_output(data: dict[str, Any]) -> BatchCreateOutput:
    """Parse a BatchCreateOutput from API response data.

    Runs :func:`_validate_batch_create_shape` internally before
    parsing, so a malformed envelope raises :class:`ValidationError`
    here rather than silently producing an empty result. This enforces
    the shape contract at the parser boundary rather than relying on
    every call site to remember to validate first — previously the
    validation was called explicitly by ``batch_create`` and a future
    refactor that forgot the step would silently get ``(succeeded=0,
    failed=0, results=[])`` from the ``.get()`` defaults below.
    """
    _validate_batch_create_shape(data)
    results: list[BatchItemResult] = []
    for item in data.get("results") or []:
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
                branded_domain=item.get("branded_domain"),
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
    """Parse an API error response into the appropriate QURLError subclass.

    Handles the full RFC 7807 Problem Details shape (``type``, ``title``,
    ``status``, ``detail``, ``instance``, ``code``) plus the pre-RFC-7807
    legacy ``{error: {code, message}}`` envelope for backward compatibility.

    The ``detail`` fallback chain is:
        1. ``err.detail``  — RFC 7807 primary
        2. ``err.message`` — legacy pre-RFC-7807 shape
        3. ``err.title``   — RFC 7807 required field
        4. ``HTTP {status}`` — final safety net

    This prevents ``"Title (403): "`` when the API omits ``detail``.
    """
    retry_after = None
    if response.status_code == 429:
        retry_after_header = response.headers.get("Retry-After")
        # Per RFC 7231 §7.1.3, `Retry-After` can be either a
        # delay-seconds integer OR an HTTP-date. The `.isdigit()` check
        # accepts only the integer form — HTTP-date strings contain
        # letters/spaces/commas and deliberately fall through to `None`,
        # which causes the retry path to use exponential backoff
        # instead. This is a safe fallback: we don't honor the server's
        # exact hint, but we also don't hang waiting for a parsed date
        # value or crash on an unexpected header format. If full
        # HTTP-date support becomes a requirement, replace `.isdigit()`
        # with a `parsedate_to_datetime`-based parse.
        if retry_after_header and retry_after_header.isdigit():
            retry_after = int(retry_after_header)

    # Pick the right subclass, defaulting to ServerError for 5xx or QURLError
    cls: type[QURLError]
    if response.status_code >= 500:
        cls = ServerError
    else:
        cls = _ERROR_CLASS_MAP.get(response.status_code, QURLError)

    try:
        envelope = response.json()
        # `.get("error") or {}` — not the more-common `.get("error", {})` —
        # because the API may return `"error": null` explicitly, not just
        # omit the key. Both cases must collapse to the empty-dict default
        # so the subsequent `err.get(...)` chains don't raise
        # `AttributeError` on `None`. Same pattern is applied to `meta`
        # below and to `envelope.get("data")` callers elsewhere.
        err = envelope.get("error") or {}
        title = err.get("title") or response.reason_phrase or ""
        detail = (
            err.get("detail")
            or err.get("message")  # legacy envelope
            or title
            or f"HTTP {response.status_code}"
        )
        return cls(
            status=err.get("status", response.status_code),
            code=err.get("code", "unknown"),
            title=title,
            detail=detail,
            type=err.get("type"),
            instance=err.get("instance"),
            invalid_fields=err.get("invalid_fields"),
            request_id=(envelope.get("meta") or {}).get("request_id"),
            meta=envelope.get("meta"),
            retry_after=retry_after,
        )
    except (ValueError, KeyError, TypeError):
        return cls(
            status=response.status_code,
            code="unknown",
            title=response.reason_phrase or "",
            detail=response.text or f"HTTP {response.status_code}",
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
    *,
    status: str | None = None,
    q: str | None = None,
    sort: str | None = None,
    created_after: datetime | str | None = None,
    created_before: datetime | str | None = None,
    expires_before: datetime | str | None = None,
    expires_after: datetime | str | None = None,
) -> dict[str, str]:
    """Build query params for list endpoints, dropping None values."""
    # Per the OpenAPI spec (GET /v1/qurls → limit: integer, minimum: 1,
    # maximum: 100, default: 20). Client-side validation catches obvious
    # mistakes before a round-trip, matching the existing style for
    # max_sessions, tag count, URL length. Omitting `limit` (None) lets
    # the server apply its default page size.
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError(
                f"limit: must be an integer between 1 and 100 (got {type(limit).__name__})"
            )
        if limit < 1 or limit > 100:
            raise ValueError(
                f"limit: must be an integer between 1 and 100 (got {limit})"
            )
    # ``status`` is a QURLStatus (Literal | str) — covered by the ``str`` arm.
    # Ordered most-specific to least-specific: datetime/int are concrete
    # types, str is the widening arm, None is the "drop" sentinel.
    pairs: dict[str, datetime | int | str | None] = {
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
    return build_query_params(pairs)


def mask_key(api_key: str) -> str:
    """Mask an API key for display, hiding JWT suffix fragments."""
    if api_key.startswith("eyJ") and _JWT_LIKE_PATTERN.match(api_key):
        return api_key[:4] + "***"
    if len(api_key) > 8:
        return api_key[:4] + "***" + api_key[-4:]
    return "***"
