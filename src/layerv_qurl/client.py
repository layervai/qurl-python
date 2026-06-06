"""Synchronous qURL API client.

NOTE: Business logic mirrors async_client.py — keep both in sync. Input
validation, body construction, and error handling must match exactly.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote

import httpx

from layerv_qurl._utils import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    RETRYABLE_STATUS,
    RETRYABLE_STATUS_POST,
    UNSET,
    _UnsetType,
    build_body,
    build_list_params,
    build_query_params,
    default_user_agent,
    idempotency_headers,
    logger,
    mask_key,
    parse_access_code,
    parse_access_code_list_output,
    parse_access_token,
    parse_agent_bootstrap_output,
    parse_api_key,
    parse_api_key_list_output,
    parse_batch_create_output,
    parse_checkout_session,
    parse_connector_installation_list_output,
    parse_create_output,
    parse_current_period_usage,
    parse_customer,
    parse_daily_usage,
    parse_domain,
    parse_domain_list_output,
    parse_domain_verify_output,
    parse_error,
    parse_invoice_list_output,
    parse_list_output,
    parse_mint_output,
    parse_portal_session,
    parse_quota,
    parse_qurl,
    parse_redeem_access_code_output,
    parse_resolve_output,
    parse_resource,
    parse_resource_detail,
    parse_resource_list_output,
    parse_session_list_output,
    parse_session_terminate_output,
    parse_webhook,
    parse_webhook_delivery_list_output,
    parse_webhook_event_types_output,
    parse_webhook_list_output,
    require_resource_id_prefix,
    retry_delay,
    validate_create_input,
    validate_id,
    validate_mint_input,
    validate_update_input,
)
from layerv_qurl.errors import QURLError, QURLNetworkError, QURLTimeoutError

if TYPE_CHECKING:
    # The `list()` method on QURLClient shadows the `list` type inside the
    # class body, so parameter/return annotations that need the builtin
    # must reference `builtins.list[...]` explicitly. The import lives in
    # a TYPE_CHECKING block because it's only needed for type annotations.
    import builtins
    from collections.abc import Iterator, Sequence
    from datetime import datetime

    from layerv_qurl.types import (
        QURL,
        AccessCode,
        AccessCodeListOutput,
        AccessPolicy,
        AccessToken,
        AgentBootstrapOutput,
        APIKey,
        APIKeyListOutput,
        BatchCreateItem,
        BatchCreateOutput,
        CheckoutSession,
        ConnectorInstallationListOutput,
        CreateOutput,
        CurrentPeriodUsage,
        Customer,
        DailyUsage,
        Domain,
        DomainListOutput,
        DomainVerifyOutput,
        InvoiceListOutput,
        ListOutput,
        MintOutput,
        PortalSession,
        Quota,
        QURLStatus,
        RedeemAccessCodeOutput,
        ResolveOutput,
        Resource,
        ResourceDetail,
        ResourceListOutput,
        SessionListOutput,
        SessionTerminateOutput,
        Webhook,
        WebhookDeliveryListOutput,
        WebhookEventTypesOutput,
        WebhookListOutput,
    )


class QURLClient:
    """Synchronous qURL API client.

    Usage::

        from layerv_qurl import QURLClient

        client = QURLClient(api_key="lv_live_xxx")

        # Create a protected link
        result = client.create(target_url="https://example.com", expires_in="24h")

        # Resolve an access token (opens firewall for your IP)
        access = client.resolve("at_k8xqp9h2sj9lx7r4a")

        # Extend a qURL's expiration
        qurl = client.extend("r_xxx", "7d")

        # Update metadata
        qurl = client.update("r_xxx", description="updated")

        # Iterate all active qURLs
        for qurl in client.list_all(status="active"):
            print(qurl.resource_id)

    ``api_key`` may be omitted for public endpoints such as
    :meth:`redeem_access_code`; authenticated endpoints will return 401
    without credentials.

    Enable debug logging to see requests::

        import logging
        logging.getLogger("layerv_qurl").setLevel(logging.DEBUG)
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        user_agent: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        if api_key is not None and not api_key.strip():
            raise ValueError("api_key must not be empty")

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_retries = max_retries
        self._user_agent = user_agent or default_user_agent()
        self._client = http_client or httpx.Client(timeout=timeout)
        self._owns_client = http_client is None
        self._base_headers: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": self._user_agent,
        }
        if api_key is not None:
            self._base_headers["Authorization"] = f"Bearer {api_key}"

    def __repr__(self) -> str:
        api_key = mask_key(self._api_key) if self._api_key is not None else None
        return f"QURLClient(api_key={api_key!r}, base_url='{self._base_url}')"

    def close(self) -> None:
        """Close the underlying HTTP client (only if owned by this instance)."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> QURLClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # --- Public API ---

    def create(
        self,
        target_url: str,
        *,
        resource_type: str | None = None,
        expires_in: str | None = None,
        label: str | None = None,
        one_time_use: bool | None = None,
        max_sessions: int | None = None,
        session_duration: str | None = None,
        access_policy: AccessPolicy | None = None,
        custom_domain: str | None = None,
        idempotency_key: str | None = None,
    ) -> CreateOutput:
        """Create a new qURL.

        Returns a :class:`CreateOutput` with the ``resource_id``, ``qurl_link``,
        ``qurl_site``, and ``expires_at``. Use :meth:`get` to fetch the full
        :class:`QURL` object with status, timestamps, and policy details.

        Note: ``tags`` and ``description`` are not accepted on create — they
        live on the resource and must be set via :meth:`update` after
        creation. The API uses different field names for the create-time
        token label (``label``) and the resource-level description on
        update/get responses.

        Args:
            target_url: The URL to protect. Max length 2048.
            resource_type: Resource type to create. Defaults to ``"url"``
                server-side. Public callers usually omit this.
            expires_in: Duration string (e.g. ``"24h"``, ``"7d"``). The API
                uses ``expires_in`` on create; use :meth:`update` with
                ``expires_at`` if you need an absolute expiry afterwards.
            label: Human-readable label for the qURL. Max length 500.
            one_time_use: If True, the qURL is consumed on first access.
            max_sessions: Maximum concurrent sessions (0 = unlimited).
                Must be between 0 and 1000 inclusive.
            session_duration: Duration string for sessions (e.g. ``"1h"``).
            access_policy: IP/geo/user-agent access restrictions.
            custom_domain: Custom domain for the qURL link. Max length 253.
            idempotency_key: Optional idempotency key for safe retries.

        Raises:
            ValueError: If any field violates the documented API constraints.
        """
        validate_create_input(
            target_url=target_url,
            label=label,
            max_sessions=max_sessions,
            custom_domain=custom_domain,
        )
        body = build_body(
            {
                "type": resource_type,
                "target_url": target_url,
                "expires_in": expires_in,
                "label": label,
                "one_time_use": one_time_use,
                "max_sessions": max_sessions,
                "session_duration": session_duration,
                "access_policy": access_policy,
                "custom_domain": custom_domain,
            }
        )
        resp = self._request(
            "POST",
            "/v1/qurls",
            body=body,
            headers=idempotency_headers(idempotency_key),
        )
        return parse_create_output(resp)

    def get(self, resource_id: str) -> QURL:
        """Get a qURL resource and its access tokens.

        Accepts either a resource ID (``r_`` prefix) or a qURL display ID
        (``q_`` prefix); the API resolves ``q_`` IDs to the parent resource
        automatically.

        Args:
            resource_id: The resource or qURL display ID.
        """
        validate_id(resource_id)
        resp = self._request("GET", f"/v1/qurls/{resource_id}")
        return parse_qurl(resp)

    def list(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
        status: QURLStatus | None = None,
        q: str | None = None,
        sort: str | None = None,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        expires_before: datetime | str | None = None,
        expires_after: datetime | str | None = None,
    ) -> ListOutput:
        """List qURLs with optional filters.

        Args:
            limit: Maximum number of results per page.
            cursor: Pagination cursor from a previous response.
            status: Filter by qURL status (``"active"``, ``"revoked"``).
            q: Search query string.
            sort: Sort order (e.g. ``"created_at"``, ``"-created_at"``).
            created_after: Filter qURLs created after this timestamp.
                Accepts a :class:`datetime` (serialized via ``.isoformat()``)
                or a string. String values must be ISO 8601 / RFC 3339
                format (e.g. ``"2026-04-01T00:00:00Z"``) and are passed
                through to the API unvalidated — the server rejects
                malformed timestamps with a 400.
            created_before: Filter qURLs created before this timestamp.
                Same format rules as ``created_after``.
            expires_before: Filter qURLs expiring before this timestamp.
                Same format rules as ``created_after``.
            expires_after: Filter qURLs expiring after this timestamp.
                Same format rules as ``created_after``.
        """
        params = build_list_params(
            limit,
            cursor,
            status,
            q,
            sort,
            created_after=created_after,
            created_before=created_before,
            expires_before=expires_before,
            expires_after=expires_after,
        )
        data, meta = self._raw_request("GET", "/v1/qurls", params=params)
        return parse_list_output(data, meta)

    def list_all(
        self,
        *,
        status: QURLStatus | None = None,
        q: str | None = None,
        sort: str | None = None,
        page_size: int = 50,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        expires_before: datetime | str | None = None,
        expires_after: datetime | str | None = None,
    ) -> Iterator[QURL]:
        """Iterate over all qURLs, automatically paginating.

        Yields individual :class:`QURL` objects, fetching pages transparently.

        Args:
            status: Filter by status (``"active"``, ``"revoked"``).
            q: Search query string.
            sort: Sort order.
            page_size: Number of items per page (default 50).
            created_after: Filter qURLs created after this timestamp.
                Accepts a :class:`datetime` (serialized via ``.isoformat()``)
                or a string. String values must be ISO 8601 / RFC 3339
                format (e.g. ``"2026-04-01T00:00:00Z"``) and are passed
                through to the API unvalidated — the server rejects
                malformed timestamps with a 400.
            created_before: Filter qURLs created before this timestamp.
                Same format rules as ``created_after``.
            expires_before: Filter qURLs expiring before this timestamp.
                Same format rules as ``created_after``.
            expires_after: Filter qURLs expiring after this timestamp.
                Same format rules as ``created_after``.
        """
        cursor: str | None = None
        while True:
            page = self.list(
                limit=page_size,
                cursor=cursor,
                status=status,
                q=q,
                sort=sort,
                created_after=created_after,
                created_before=created_before,
                expires_before=expires_before,
                expires_after=expires_after,
            )
            yield from page.qurls
            if not page.has_more or not page.next_cursor:
                break
            cursor = page.next_cursor

    def delete(self, resource_id: str) -> None:
        """Delete (revoke) a qURL resource and all its access tokens.

        Only accepts a resource ID (``r_`` prefix), not a qURL display ID
        (``q_`` prefix). Per the OpenAPI spec:
        *"Requires a resource ID (r_ prefix). To revoke a single token,
        use DELETE /v1/resources/:id/qurls/:qurl_id"*.

        A client-side prefix check catches the mistake before the API
        round-trip.

        Args:
            resource_id: The resource ID (must start with ``r_``).

        Raises:
            ValueError: If ``resource_id`` is malformed or does not start
                with ``r_``.
        """
        validate_id(resource_id)
        require_resource_id_prefix(resource_id, "delete")
        self._request("DELETE", f"/v1/qurls/{resource_id}")

    def extend(self, resource_id: str, duration: str) -> QURL:
        """Extend a qURL's expiration.

        Convenience method — equivalent to ``update(resource_id, extend_by=duration)``.
        Accepts either a resource ID (``r_`` prefix) or a qURL display ID
        (``q_`` prefix); the API resolves ``q_`` IDs to the parent resource
        automatically.

        Args:
            resource_id: Resource or qURL display ID.
            duration: Duration to add (e.g. ``"7d"``, ``"24h"``).
        """
        return self.update(resource_id, extend_by=duration)

    def update(
        self,
        resource_id: str,
        *,
        extend_by: str | None = None,
        expires_at: datetime | str | None = None,
        description: str | None = None,
        tags: builtins.list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> QURL:
        """Update a qURL — extend expiration, change description, set tags.

        Accepts either a resource ID (``r_`` prefix) or a qURL display ID
        (``q_`` prefix). All fields are optional, but at least one must be
        provided. ``extend_by`` and ``expires_at`` are mutually exclusive.

        **Cannot change `access_policy`** — the policy is immutable after
        create, per the OpenAPI spec's ``UpdateQurlRequest``. To change
        access policy, either create a new resource via :meth:`create`
        with the new policy, or mint a new access token on the existing
        resource via :meth:`mint_link` with a policy override (per-token
        scope, base resource policy unchanged).

        Args:
            resource_id: Resource or qURL display ID.
            extend_by: Duration to add (e.g. ``"7d"``). Mutually exclusive
                with ``expires_at``.
            expires_at: New absolute expiry. Mutually exclusive with
                ``extend_by``.
            description: New resource description. Pass an empty string to
                clear. Max length 500.
            tags: Replacement tag list — this is always a REPLACE, never
                a merge. Pass ``tags=[]`` to clear all tags (always a
                real clear operation, even if no tags were set); pass
                ``None`` (the default) to leave the existing tags
                unchanged. Max 10 items, each 1-50 chars matching
                ``^[a-zA-Z0-9][a-zA-Z0-9 _-]*$``.
            idempotency_key: Optional idempotency key for safe retries.

        Raises:
            ValueError: If ``extend_by`` and ``expires_at`` are both set, if
                no update fields are provided, or if any field violates the
                documented API constraints.
        """
        validate_id(resource_id)
        if extend_by is not None and expires_at is not None:
            raise ValueError(
                "update: `extend_by` and `expires_at` are mutually exclusive "
                "— provide at most one"
            )
        if (
            extend_by is None
            and expires_at is None
            and description is None
            and tags is None
        ):
            raise ValueError(
                "update: at least one field (extend_by, expires_at, description, "
                "tags) must be provided"
            )
        validate_update_input(description=description, tags=tags)
        # `build_body` strips top-level ``None`` only — falsy values like
        # ``tags=[]`` and ``description=""`` are preserved. This is
        # load-bearing: ``tags=[]`` is an intentional "clear all tags"
        # API operation and ``description=""`` clears the description.
        # A future refactor that adds a truthiness check here would
        # silently drop both.
        body = build_body(
            {
                "extend_by": extend_by,
                "expires_at": expires_at,
                "description": description,
                "tags": tags,
            }
        )
        resp = self._request(
            "PATCH",
            f"/v1/qurls/{resource_id}",
            body=body,
            headers=idempotency_headers(idempotency_key),
        )
        return parse_qurl(resp)

    def mint_link(
        self,
        resource_id: str,
        *,
        expires_at: datetime | str | None = None,
        expires_in: str | None = None,
        label: str | None = None,
        one_time_use: bool | None = None,
        max_sessions: int | None = None,
        session_duration: str | None = None,
        access_policy: AccessPolicy | None = None,
        idempotency_key: str | None = None,
    ) -> MintOutput:
        """Mint a new access link for a qURL.

        Accepts either a resource ID (``r_`` prefix) or a qURL display ID
        (``q_`` prefix). ``expires_in`` and ``expires_at`` are mutually
        exclusive — if neither is set, the link defaults to 24 hours.

        Args:
            resource_id: Resource or qURL display ID.
            expires_at: Absolute expiry for the minted link. Mutually
                exclusive with ``expires_in``.
            expires_in: Duration string for the link (e.g. ``"24h"``).
                Mutually exclusive with ``expires_at``.
            label: Human-readable label for the link. Max length 500.
            one_time_use: If True, the link can only be used once.
            max_sessions: Maximum concurrent sessions allowed.
                Must be between 0 and 1000 inclusive.
            session_duration: Duration string for sessions (e.g. ``"1h"``).
            access_policy: IP/geo/user-agent access restrictions.
            idempotency_key: Optional idempotency key for safe retries.

        Raises:
            ValueError: If ``expires_in`` and ``expires_at`` are both set
                or if any field violates the documented API constraints.
        """
        validate_id(resource_id)
        if expires_in is not None and expires_at is not None:
            raise ValueError(
                "mint_link: `expires_in` and `expires_at` are mutually exclusive "
                "— provide at most one"
            )
        validate_mint_input(label=label, max_sessions=max_sessions)
        body = build_body(
            {
                "expires_at": expires_at,
                "expires_in": expires_in,
                "label": label,
                "one_time_use": one_time_use,
                "max_sessions": max_sessions,
                "session_duration": session_duration,
                "access_policy": access_policy,
            }
        )
        resp = self._request(
            "POST",
            f"/v1/qurls/{resource_id}/mint_link",
            body=body,
            headers=idempotency_headers(idempotency_key),
        )
        return parse_mint_output(resp)

    def batch_create(
        self,
        items: Sequence[BatchCreateItem],
        *,
        idempotency_key: str | None = None,
    ) -> BatchCreateOutput:
        """Create multiple qURLs at once (1-100 items).

        Each item is validated against the same spec constraints as
        :meth:`create` before the request is sent, with per-item errors
        attributed by index (``items[N]: ...``).

        **Partial failures do not raise.** Two API status codes resolve
        normally with structured per-item results:

        - **HTTP 207 Multi-Status** (some succeeded, some failed).
        - **HTTP 400** (every item failed validation) — the API returns a
          populated ``BatchCreateOutput`` body on this path, so the SDK
          whitelists 400 and surfaces the per-item errors instead of
          raising a generic :class:`ValidationError`.

        Other error statuses (401, 403, 429, 5xx) still raise the
        appropriate :class:`QURLError` subclass. Inspect
        ``result.failed > 0`` and iterate ``result.results`` to see
        which items succeeded and which errored.

        Args:
            items: List of dicts, each with at least ``target_url``.

        Raises:
            ValueError: If ``items`` is empty, exceeds 100 items, or any
                item violates the documented API constraints.
        """
        if not items:
            raise ValueError("batch_create requires at least 1 item")
        if len(items) > 100:
            raise ValueError(
                f"batch_create accepts at most 100 items (got {len(items)})"
            )
        # Validate each item against the same spec constraints as single
        # create() so obvious mistakes fail fast with the offending index.
        for i, item in enumerate(items):
            try:
                validate_create_input(
                    target_url=item["target_url"],
                    label=item.get("label"),
                    max_sessions=item.get("max_sessions"),
                    custom_domain=item.get("custom_domain"),
                )
            except KeyError as exc:
                raise ValueError(
                    f"batch_create items[{i}]: missing required field 'target_url'"
                ) from exc
            except ValueError as exc:
                raise ValueError(f"batch_create items[{i}]: {exc}") from exc
        # `BatchCreateItem` is structurally a `dict[str, Any]` at runtime —
        # TypedDicts compile to plain dicts and carry no runtime overhead.
        # The `cast` narrows the type for `build_body` without any runtime
        # conversion.
        serialized = [build_body(cast("dict[str, Any]", item)) for item in items]
        # HTTP 400 carries structured per-item errors on this endpoint —
        # whitelist it so the generic error path doesn't swallow the body.
        # `allow_statuses=(400,)` only — HTTP 207 Multi-Status (partial
        # success) flows through the normal `status_code < 400` success
        # path automatically, so it doesn't need to be whitelisted. Only
        # the total-failure 400 needs the opt-in, because the API
        # populates a `BatchCreateOutput` body there that the generic
        # error path would otherwise swallow.
        resp = self._request(
            "POST",
            "/v1/qurls/batch",
            body={"items": serialized},
            allow_statuses=(400,),
            headers=idempotency_headers(idempotency_key),
        )
        # `parse_batch_create_output` runs the shape guard internally
        # (see its docstring) — so if the API returns 400 with an
        # unexpected body shape, the parser raises ValidationError
        # rather than silently producing `(succeeded=0, failed=0,
        # results=[])`. The guard is enforced at the parser boundary,
        # not documented by convention at every call site.
        return parse_batch_create_output(resp)

    def resolve(
        self, access_token: str, *, idempotency_key: str | None = None
    ) -> ResolveOutput:
        """Resolve a qURL access token (headless).

        Triggers an NHP knock to open firewall access for the caller's IP.
        Requires ``qurl:resolve`` scope on the API key.

        Args:
            access_token: The access token string (e.g. ``"at_k8xqp9h2sj9lx7r4a"``).
        """
        validate_id(access_token, "access_token")
        resp = self._request(
            "POST",
            "/v1/resolve",
            body={"access_token": access_token},
            headers=idempotency_headers(idempotency_key),
        )
        return parse_resolve_output(resp)

    def get_quota(self) -> Quota:
        """Get quota and usage information."""
        resp = self._request("GET", "/v1/quota")
        return parse_quota(resp)

    # --- Resources ---

    def list_resources(
        self,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        alias: str | None = None,
        slug: str | None = None,
        status: str | None = None,
        resource_type: str | None = None,
    ) -> ResourceListOutput:
        """List resources.

        ``resource_type`` serializes to the API's ``type`` query parameter.
        Supported public filter values are ``"url"`` and ``"tunnel"``.
        """
        params = build_query_params(
            {
                "cursor": cursor,
                "limit": limit,
                "alias": alias,
                "slug": slug,
                "status": status,
                "type": resource_type,
            }
        )
        data, meta = self._raw_request("GET", "/v1/resources", params=params)
        return parse_resource_list_output(data, meta)

    def create_resource(
        self,
        *,
        resource_type: str | None = None,
        target_url: str | None = None,
        description: str | None = None,
        tags: builtins.list[str] | None = None,
        custom_domain: str | None = None,
        alias: str | None = None,
        slug: str | None = None,
        find_or_create: bool | None = None,
        idempotency_key: str | None = None,
    ) -> Resource:
        """Create or find a resource.

        ``resource_type`` serializes to the API's ``type`` request field.
        Tunnel resources use ``slug`` and may set ``find_or_create=True``.
        """
        if target_url is not None:
            validate_create_input(target_url=target_url, custom_domain=custom_domain)
        validate_update_input(
            description=description, tags=tags, custom_domain=custom_domain
        )
        body = build_body(
            {
                "type": resource_type,
                "target_url": target_url,
                "description": description,
                "tags": tags,
                "custom_domain": custom_domain,
                "alias": alias,
                "slug": slug,
                "find_or_create": find_or_create,
            }
        )
        resp = self._request(
            "POST",
            "/v1/resources",
            body=body,
            headers=idempotency_headers(idempotency_key),
        )
        return parse_resource(resp)

    def get_resource(self, resource_id: str) -> ResourceDetail:
        """Get resource details and a bounded qURL token preview."""
        validate_id(resource_id)
        resp = self._request("GET", f"/v1/resources/{resource_id}")
        return parse_resource_detail(resp)

    def update_resource(
        self,
        resource_id: str,
        *,
        description: str | None = None,
        tags: builtins.list[str] | None = None,
        custom_domain: str | None = None,
        preserve_host: bool | None = None,
        alias: str | None | _UnsetType = UNSET,
        idempotency_key: str | None = None,
    ) -> Resource:
        """Update resource metadata.

        ``alias`` is tri-state: omit for no change, pass a string to set
        or rebind, and pass ``None`` to clear.
        """
        validate_id(resource_id)
        validate_update_input(
            description=description, tags=tags, custom_domain=custom_domain
        )
        body = build_body(
            {
                "description": description,
                "tags": tags,
                "custom_domain": custom_domain,
                "preserve_host": preserve_host,
            }
        )
        if alias is not UNSET:
            body["alias"] = alias
        if not body:
            raise ValueError(
                "update_resource: at least one field (description, tags, "
                "custom_domain, preserve_host, alias) must be provided"
            )
        resp = self._request(
            "PATCH",
            f"/v1/resources/{resource_id}",
            body=body,
            headers=idempotency_headers(idempotency_key),
        )
        return parse_resource(resp)

    def delete_resource(self, resource_id: str) -> None:
        """Revoke a resource and all qURLs associated with it."""
        validate_id(resource_id)
        self._request("DELETE", f"/v1/resources/{resource_id}")

    def create_qurl_for_resource(
        self,
        resource_id: str,
        *,
        expires_in: str | None = None,
        label: str | None = None,
        one_time_use: bool | None = None,
        max_sessions: int | None = None,
        session_duration: str | None = None,
        access_policy: AccessPolicy | None = None,
        idempotency_key: str | None = None,
    ) -> CreateOutput:
        """Mint a qURL against an existing resource."""
        validate_id(resource_id)
        validate_mint_input(label=label, max_sessions=max_sessions)
        body = build_body(
            {
                "expires_in": expires_in,
                "label": label,
                "one_time_use": one_time_use,
                "max_sessions": max_sessions,
                "session_duration": session_duration,
                "access_policy": access_policy,
            }
        )
        resp = self._request(
            "POST",
            f"/v1/resources/{resource_id}/qurls",
            body=body,
            headers=idempotency_headers(idempotency_key),
        )
        return parse_create_output(resp)

    def mint_resource_qurl(
        self,
        resource_id: str,
        *,
        expires_in: str | None = None,
        label: str | None = None,
        one_time_use: bool | None = None,
        max_sessions: int | None = None,
        session_duration: str | None = None,
        access_policy: AccessPolicy | None = None,
        idempotency_key: str | None = None,
    ) -> CreateOutput:
        """Alias for :meth:`create_qurl_for_resource`."""
        return self.create_qurl_for_resource(
            resource_id,
            expires_in=expires_in,
            label=label,
            one_time_use=one_time_use,
            max_sessions=max_sessions,
            session_duration=session_duration,
            access_policy=access_policy,
            idempotency_key=idempotency_key,
        )

    def update_resource_qurl(
        self,
        resource_id: str,
        qurl_id: str,
        *,
        extend_by: str | None = None,
        expires_at: datetime | str | None = None,
        label: str | None = None,
        access_policy: AccessPolicy | None = None,
        max_sessions: int | None = None,
        session_duration: str | None = None,
        idempotency_key: str | None = None,
    ) -> AccessToken:
        """Update a specific qURL token on a resource."""
        validate_id(resource_id)
        validate_id(qurl_id, "qurl_id")
        if extend_by is not None and expires_at is not None:
            raise ValueError(
                "update_resource_qurl: `extend_by` and `expires_at` are mutually "
                "exclusive — provide at most one"
            )
        validate_mint_input(label=label, max_sessions=max_sessions)
        body = build_body(
            {
                "extend_by": extend_by,
                "expires_at": expires_at,
                "label": label,
                "access_policy": access_policy,
                "max_sessions": max_sessions,
                "session_duration": session_duration,
            }
        )
        if not body:
            raise ValueError(
                "update_resource_qurl: at least one field (extend_by, expires_at, "
                "label, access_policy, max_sessions, session_duration) must be provided"
            )
        resp = self._request(
            "PATCH",
            f"/v1/resources/{resource_id}/qurls/{qurl_id}",
            body=body,
            headers=idempotency_headers(idempotency_key),
        )
        return parse_access_token(resp)

    def revoke_resource_qurl(self, resource_id: str, qurl_id: str) -> None:
        """Revoke one qURL token without revoking the parent resource."""
        validate_id(resource_id)
        validate_id(qurl_id, "qurl_id")
        self._request("DELETE", f"/v1/resources/{resource_id}/qurls/{qurl_id}")

    def list_resource_sessions(self, resource_id: str) -> SessionListOutput:
        """List active sessions for a resource."""
        validate_id(resource_id)
        resp = self._request("GET", f"/v1/resources/{resource_id}/sessions")
        return parse_session_list_output(resp)

    def terminate_all_resource_sessions(
        self, resource_id: str
    ) -> SessionTerminateOutput:
        """Terminate all active sessions for a resource."""
        validate_id(resource_id)
        resp = self._request("DELETE", f"/v1/resources/{resource_id}/sessions")
        return parse_session_terminate_output(resp)

    def terminate_resource_session(self, resource_id: str, session_id: str) -> None:
        """Terminate a specific active session."""
        validate_id(resource_id)
        validate_id(session_id, "session_id")
        self._request(
            "DELETE",
            f"/v1/resources/{resource_id}/sessions/{session_id}",
        )

    # --- Custom Domains ---

    def register_domain(
        self, domain: str, *, idempotency_key: str | None = None
    ) -> Domain:
        """Register a custom domain and return DNS setup records."""
        resp = self._request(
            "POST",
            "/v1/domains",
            body={"domain": domain},
            headers=idempotency_headers(idempotency_key),
        )
        return parse_domain(resp)

    def list_domains(
        self, *, limit: int | None = None, cursor: str | None = None
    ) -> DomainListOutput:
        """List custom domains."""
        data, meta = self._raw_request(
            "GET",
            "/v1/domains",
            params=build_query_params({"limit": limit, "cursor": cursor}),
        )
        return parse_domain_list_output(data, meta)

    def get_domain(self, domain: str) -> Domain:
        """Get custom-domain status and DNS configuration."""
        resp = self._request("GET", f"/v1/domains/{quote(domain, safe='')}")
        return parse_domain(resp)

    def delete_domain(self, domain: str) -> None:
        """Remove a custom-domain registration."""
        self._request("DELETE", f"/v1/domains/{quote(domain, safe='')}")

    def verify_domain(
        self, domain: str, *, idempotency_key: str | None = None
    ) -> DomainVerifyOutput:
        """Trigger DNS verification for a custom domain."""
        resp = self._request(
            "POST",
            f"/v1/domains/{quote(domain, safe='')}/verify",
            headers=idempotency_headers(idempotency_key),
        )
        return parse_domain_verify_output(resp)

    def regenerate_domain_token(
        self, domain: str, *, idempotency_key: str | None = None
    ) -> Domain:
        """Regenerate a custom-domain verification token."""
        resp = self._request(
            "POST",
            f"/v1/domains/{quote(domain, safe='')}/regenerate-token",
            headers=idempotency_headers(idempotency_key),
        )
        return parse_domain(resp)

    # --- Webhooks ---

    def list_webhooks(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
        event: str | None = None,
    ) -> WebhookListOutput:
        """List webhook subscriptions."""
        data, meta = self._raw_request(
            "GET",
            "/v1/webhooks",
            params=build_query_params({"limit": limit, "cursor": cursor, "event": event}),
        )
        return parse_webhook_list_output(data, meta)

    def create_webhook(
        self,
        *,
        url: str,
        events: Sequence[str],
        description: str | None = None,
        idempotency_key: str | None = None,
    ) -> Webhook:
        """Create a webhook subscription.

        The returned ``Webhook.secret`` is only available on create and
        regenerate-secret responses.
        """
        body = build_body(
            {
                "url": url,
                "events": list(events),
                "description": description,
            }
        )
        resp = self._request(
            "POST",
            "/v1/webhooks",
            body=body,
            headers=idempotency_headers(idempotency_key),
        )
        return parse_webhook(resp)

    def get_webhook(self, webhook_id: str) -> Webhook:
        """Get webhook subscription details."""
        validate_id(webhook_id, "webhook_id")
        resp = self._request("GET", f"/v1/webhooks/{webhook_id}")
        return parse_webhook(resp)

    def update_webhook(
        self,
        webhook_id: str,
        *,
        url: str | None = None,
        events: Sequence[str] | None = None,
        description: str | None = None,
        status: str | None = None,
        idempotency_key: str | None = None,
    ) -> Webhook:
        """Update a webhook subscription."""
        validate_id(webhook_id, "webhook_id")
        body = build_body(
            {
                "url": url,
                "events": list(events) if events is not None else None,
                "description": description,
                "status": status,
            }
        )
        if not body:
            raise ValueError(
                "update_webhook: at least one field (url, events, description, "
                "status) must be provided"
            )
        resp = self._request(
            "PATCH",
            f"/v1/webhooks/{webhook_id}",
            body=body,
            headers=idempotency_headers(idempotency_key),
        )
        return parse_webhook(resp)

    def delete_webhook(self, webhook_id: str) -> None:
        """Delete a webhook subscription."""
        validate_id(webhook_id, "webhook_id")
        self._request("DELETE", f"/v1/webhooks/{webhook_id}")

    def regenerate_webhook_secret(
        self, webhook_id: str, *, idempotency_key: str | None = None
    ) -> Webhook:
        """Regenerate a webhook signing secret."""
        validate_id(webhook_id, "webhook_id")
        resp = self._request(
            "POST",
            f"/v1/webhooks/{webhook_id}/secret",
            headers=idempotency_headers(idempotency_key),
        )
        return parse_webhook(resp)

    def list_webhook_deliveries(
        self,
        webhook_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> WebhookDeliveryListOutput:
        """List delivery attempts for a webhook."""
        validate_id(webhook_id, "webhook_id")
        data, meta = self._raw_request(
            "GET",
            f"/v1/webhooks/{webhook_id}/deliveries",
            params=build_query_params({"limit": limit, "cursor": cursor}),
        )
        return parse_webhook_delivery_list_output(data, meta)

    def list_webhook_event_types(self) -> WebhookEventTypesOutput:
        """List supported webhook event types."""
        resp = self._request("GET", "/v1/webhooks/events")
        return parse_webhook_event_types_output(resp)

    # --- API Keys ---

    def create_api_key(
        self,
        *,
        name: str,
        scopes: Sequence[str],
        expires_in: str | None = None,
        purpose: str | None = None,
        tunnel_slug: str | None = None,
        idempotency_key: str | None = None,
    ) -> APIKey:
        """Create an API key.

        JWT auth is required for normal key management; API-key auth may
        only create restricted tunnel-bootstrap keys.

        ``idempotency_key`` must be 32-256 characters for this
        security-sensitive endpoint.
        """
        body = build_body(
            {
                "name": name,
                "scopes": list(scopes),
                "expires_in": expires_in,
                "purpose": purpose,
                "tunnel_slug": tunnel_slug,
            }
        )
        resp = self._request(
            "POST",
            "/v1/api-keys",
            body=body,
            headers=idempotency_headers(idempotency_key, min_length=32),
        )
        return parse_api_key(resp)

    def list_api_keys(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
        status: str | None = None,
    ) -> APIKeyListOutput:
        """List API keys. JWT auth is required by the API."""
        data, meta = self._raw_request(
            "GET",
            "/v1/api-keys",
            params=build_query_params({"limit": limit, "cursor": cursor, "status": status}),
        )
        return parse_api_key_list_output(data, meta)

    def update_api_key(
        self,
        key_id: str,
        *,
        name: str | None = None,
        scopes: Sequence[str] | None = None,
        idempotency_key: str | None = None,
    ) -> APIKey:
        """Update API key name or scopes. JWT auth is required by the API."""
        validate_id(key_id, "key_id")
        body = build_body({"name": name, "scopes": list(scopes) if scopes is not None else None})
        resp = self._request(
            "PATCH",
            f"/v1/api-keys/{key_id}",
            body=body,
            headers=idempotency_headers(idempotency_key),
        )
        return parse_api_key(resp)

    def revoke_api_key(self, key_id: str) -> None:
        """Revoke an API key. JWT auth is required by the API."""
        validate_id(key_id, "key_id")
        self._request("DELETE", f"/v1/api-keys/{key_id}")

    # --- Access Codes ---

    def redeem_access_code(
        self,
        code: str,
        *,
        honeypot: str = "",
        elapsed_ms: int | None = None,
        idempotency_key: str | None = None,
    ) -> RedeemAccessCodeOutput:
        """Redeem a public access code and return its redirect URL."""
        body = build_body({"code": code, "honeypot": honeypot, "elapsed_ms": elapsed_ms})
        resp = self._request(
            "POST",
            "/v1/access-codes/redeem",
            body=body,
            headers=idempotency_headers(idempotency_key),
        )
        return parse_redeem_access_code_output(resp)

    def create_access_code(
        self,
        *,
        resource_id: str,
        name: str | None = None,
        max_uses: int | None = None,
        expires_at: datetime | str | None = None,
        idempotency_key: str | None = None,
    ) -> AccessCode:
        """Create an access code for a resource."""
        validate_id(resource_id)
        body = build_body(
            {
                "resource_id": resource_id,
                "name": name,
                "max_uses": max_uses,
                "expires_at": expires_at,
            }
        )
        resp = self._request(
            "POST",
            "/v1/access-codes",
            body=body,
            headers=idempotency_headers(idempotency_key),
        )
        return parse_access_code(resp)

    def list_access_codes(self) -> AccessCodeListOutput:
        """List access codes."""
        resp = self._request("GET", "/v1/access-codes")
        return parse_access_code_list_output(resp)

    def revoke_access_code(self, access_code_id: str) -> None:
        """Revoke an access code."""
        validate_id(access_code_id, "access_code_id")
        self._request("DELETE", f"/v1/access-codes/{access_code_id}")

    # --- Usage, Customer, Billing, Connectors, Agent ---

    def get_usage_current_period(self) -> CurrentPeriodUsage:
        """Get current billing-period usage. JWT auth is required by the API."""
        resp = self._request("GET", "/v1/usage/current-period")
        return parse_current_period_usage(resp)

    def get_usage_daily(self) -> DailyUsage:
        """Get daily qURL creation counts for the current billing period."""
        resp = self._request("GET", "/v1/usage/daily")
        return parse_daily_usage(resp)

    def get_customer(self) -> Customer:
        """Get the authenticated customer profile. JWT auth is required by the API."""
        resp = self._request("GET", "/v1/customer")
        return parse_customer(resp)

    def update_customer(
        self, *, spending_cap_cents: int, idempotency_key: str | None = None
    ) -> Customer:
        """Update customer billing settings. JWT auth is required by the API."""
        resp = self._request(
            "PATCH",
            "/v1/customer",
            body={"spending_cap_cents": spending_cap_cents},
            headers=idempotency_headers(idempotency_key),
        )
        return parse_customer(resp)

    def create_billing_checkout(
        self, *, plan: str, idempotency_key: str | None = None
    ) -> CheckoutSession:
        """Create a Stripe checkout session. JWT auth is required by the API."""
        resp = self._request(
            "POST",
            "/v1/billing/checkout",
            body={"plan": plan},
            headers=idempotency_headers(idempotency_key),
        )
        return parse_checkout_session(resp)

    def create_billing_portal(
        self, *, idempotency_key: str | None = None
    ) -> PortalSession:
        """Create a Stripe billing portal session. JWT auth is required by the API."""
        resp = self._request(
            "POST",
            "/v1/billing/portal",
            headers=idempotency_headers(idempotency_key),
        )
        return parse_portal_session(resp)

    def list_billing_invoices(
        self, *, limit: int | None = None, cursor: str | None = None
    ) -> InvoiceListOutput:
        """List billing invoices. JWT auth is required by the API."""
        data, meta = self._raw_request(
            "GET",
            "/v1/billing/invoices",
            params=build_query_params({"limit": limit, "cursor": cursor}),
        )
        return parse_invoice_list_output(data, meta)

    def list_connector_installations(
        self, *, limit: int | None = None, cursor: str | None = None
    ) -> ConnectorInstallationListOutput:
        """List normalized connector installations."""
        data, meta = self._raw_request(
            "GET",
            "/v1/connectors/installations",
            params=build_query_params({"limit": limit, "cursor": cursor}),
        )
        return parse_connector_installation_list_output(data, meta)

    def bootstrap_agent(
        self,
        *,
        public_key: str,
        agent_id: str | None = None,
        hostname: str | None = None,
        version: str | None = None,
        idempotency_key: str | None = None,
    ) -> AgentBootstrapOutput:
        """Bootstrap a LayerV qURL Connector agent."""
        body = build_body(
            {
                "public_key": public_key,
                "agent_id": agent_id,
                "hostname": hostname,
                "version": version,
            }
        )
        resp = self._request(
            "POST",
            "/v1/agent/bootstrap",
            body=body,
            headers=idempotency_headers(idempotency_key),
        )
        return parse_agent_bootstrap_output(resp)

    # --- Internal HTTP plumbing ---

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        allow_statuses: tuple[int, ...] = (),
        headers: dict[str, str] | None = None,
    ) -> Any:
        return self._raw_request(
            method,
            path,
            body=body,
            params=params,
            allow_statuses=allow_statuses,
            headers=headers,
        )[0]

    def _raw_request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        allow_statuses: tuple[int, ...] = (),
        headers: dict[str, str] | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Issue an HTTP request and parse the JSON envelope.

        ``allow_statuses`` lets a caller opt specific non-2xx codes out of
        the default raise-on-error path and receive the parsed body
        instead. This is used by :meth:`batch_create`, where the API
        returns a structured ``BatchCreateOutput`` on HTTP 400 (all items
        rejected) — raising would drop the per-item errors.

        **`allow_statuses` takes precedence over retries.** The check
        order in the response-handling loop is:

        1. ``response.status_code < 400 or in allow_statuses`` →
           return the parsed body immediately as a success.
        2. Otherwise, build an error and check the retry filter
           (``RETRYABLE_STATUS_POST`` for POST, ``RETRYABLE_STATUS``
           for everything else).

        This means a status listed in ``allow_statuses`` is returned
        to the caller **without ever running through the retry
        filter**, even if that status would normally be retried. For
        the only current use case (``batch_create`` with
        ``allow_statuses=(400,)``) the interaction is harmless because
        400 isn't in any retry set — a 400 carries the authoritative
        per-item errors and retrying would just reproduce them.

        Callers adding a *retryable* status (e.g. 429 or 5xx) to
        ``allow_statuses`` should be aware this bypasses the SDK's
        retry path entirely: the status is surfaced on the first
        attempt with no transparent backoff. If that's not what you
        want, leave the status out of ``allow_statuses`` and let the
        normal retry logic handle it.
        """
        url = f"{self._base_url}{path}"
        last_error: Exception | None = None
        request_headers = dict(self._base_headers)
        if headers:
            request_headers.update(headers)

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                delay = retry_delay(attempt, last_error)
                logger.debug("Retry %d/%d after %.1fs", attempt, self._max_retries, delay)
                time.sleep(delay)

            logger.debug("%s %s", method, url)

            try:
                response = self._client.request(
                    method,
                    url,
                    json=body,
                    params=params,
                    headers=request_headers,
                )
            except httpx.TimeoutException as exc:
                logger.debug("%s %s timed out", method, url)
                if attempt < self._max_retries:
                    last_error = exc
                    continue
                raise QURLTimeoutError(str(exc), cause=exc) from exc
            except httpx.TransportError as exc:
                logger.debug("%s %s transport error: %s", method, url, exc)
                if attempt < self._max_retries:
                    last_error = exc
                    continue
                raise QURLNetworkError(str(exc), cause=exc) from exc

            logger.debug("%s %s → %d", method, url, response.status_code)

            if response.status_code < 400 or response.status_code in allow_statuses:
                if response.status_code == 204 or not response.content:
                    return None, None
                try:
                    envelope = response.json()
                except (json.JSONDecodeError, ValueError):
                    # If a whitelisted status (e.g. 400 on batch_create)
                    # comes back with a non-JSON body — a proxy HTML
                    # error page, a truncated response, a gateway's own
                    # plaintext error — we CAN'T surface it as success.
                    # Fall through to `parse_error`, which handles
                    # non-JSON error bodies gracefully and returns a
                    # well-formed QURLError using the response status
                    # and reason phrase. `raise ... from None` hides
                    # the JSONDecodeError chain since it's noise: the
                    # QURLError already captures "the body wasn't a
                    # parseable envelope" as its detail.
                    raise parse_error(response) from None
                return envelope.get("data"), envelope.get("meta")

            err = parse_error(response)
            retryable = RETRYABLE_STATUS_POST if method == "POST" else RETRYABLE_STATUS
            if response.status_code in retryable and attempt < self._max_retries:
                last_error = err
                continue
            raise err

        if isinstance(last_error, httpx.TimeoutException):
            raise QURLTimeoutError(str(last_error), cause=last_error) from last_error
        if isinstance(last_error, httpx.TransportError):
            raise QURLNetworkError(str(last_error), cause=last_error) from last_error
        raise last_error or QURLError(
            status=0, code="unknown", title="Request failed", detail="Exhausted retries"
        )
