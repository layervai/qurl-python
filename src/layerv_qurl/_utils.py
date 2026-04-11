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
    if not isinstance(target_url, str) or not target_url.startswith(_ALLOWED_URL_SCHEMES):
        # `repr(...)[:40]` instead of `target_url[:32]!r` — the original
        # subscript would raise `TypeError` on any non-subscriptable
        # input (None, int, bool, …) *before* the ValueError could
        # surface, masking the real validation failure with a cryptic
        # slicing error. `repr()` works on any object.
        raise ValueError(
            f"target_url: must start with http:// or https:// (got {repr(target_url)[:40]})"
        )
    _require_max_length(target_url, "target_url", MAX_TARGET_URL)
    _require_max_length(label, "label", MAX_LABEL)
    _require_max_length(custom_domain, "custom_domain", MAX_CUSTOM_DOMAIN)
    _require_max_sessions_in_range(max_sessions)


def validate_update_input(
    *,
    description: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Validate update_qurl input against spec-documented constraints."""
    _require_max_length(description, "description", MAX_DESCRIPTION)
    _require_valid_tags(tags)


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
    # Normalize empty-string `qurl_id` → None for idiomatic truthiness
    # checks. Intentionally asymmetric with `label` (preserved as-is):
    # `""` is never a meaningful identifier but IS a meaningful "cleared"
    # value for user-facing metadata.
    qurl_id_raw = data.get("qurl_id")
    qurl_id = qurl_id_raw if qurl_id_raw else None
    return CreateOutput(
        resource_id=data["resource_id"],
        qurl_link=data["qurl_link"],
        qurl_site=data["qurl_site"],
        expires_at=_parse_dt(data.get("expires_at")),
        qurl_id=qurl_id,
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
    if data.get("access_grant") is not None:
        grant_data = data["access_grant"]
        grant = AccessGrant(
            expires_in=grant_data["expires_in"],
            granted_at=_parse_dt(grant_data.get("granted_at")),
            src_ip=grant_data.get("src_ip", ""),
        )
    return ResolveOutput(
        target_url=data["target_url"],
        resource_id=data["resource_id"],
        access_grant=grant,
    )


def parse_quota(data: dict[str, Any]) -> Quota:
    """Parse a Quota from API response data."""
    rate_limits = None
    if data.get("rate_limits") is not None:
        limits_data = data["rate_limits"]
        rate_limits = RateLimits(
            create_per_minute=limits_data.get("create_per_minute", 0),
            create_per_hour=limits_data.get("create_per_hour", 0),
            list_per_minute=limits_data.get("list_per_minute", 0),
            resolve_per_minute=limits_data.get("resolve_per_minute", 0),
            max_active_qurls=limits_data.get("max_active_qurls", 0),
            max_tokens_per_qurl=limits_data.get("max_tokens_per_qurl", 0),
            max_expiry_seconds=limits_data.get("max_expiry_seconds", 0),
        )
    usage = None
    if data.get("usage") is not None:
        usage_data = data["usage"]
        usage = Usage(
            qurls_created=usage_data.get("qurls_created", 0),
            active_qurls=usage_data.get("active_qurls", 0),
            # Nullable per the API spec — the field is null when
            # max_active_qurls is unlimited.
            active_qurls_percent=usage_data.get("active_qurls_percent"),
            total_accesses=usage_data.get("total_accesses", 0),
        )
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
        rate_limits=rate_limits,
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
    status: str | None,
    q: str | None,
    sort: str | None,
    created_after: datetime | str | None = None,
    created_before: datetime | str | None = None,
    expires_before: datetime | str | None = None,
    expires_after: datetime | str | None = None,
) -> dict[str, str]:
    """Build query params for list endpoints, dropping None values."""
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
