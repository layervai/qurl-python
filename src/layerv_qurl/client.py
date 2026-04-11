"""Synchronous QURL API client.

NOTE: Business logic mirrors async_client.py — keep both in sync. Input
validation, body construction, and error handling must match exactly.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import httpx

from layerv_qurl._utils import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    RETRYABLE_STATUS,
    RETRYABLE_STATUS_POST,
    _validate_batch_create_shape,
    build_body,
    build_list_params,
    default_user_agent,
    logger,
    mask_key,
    parse_batch_create_output,
    parse_create_output,
    parse_error,
    parse_list_output,
    parse_mint_output,
    parse_quota,
    parse_qurl,
    parse_resolve_output,
    require_resource_id_prefix,
    retry_delay,
    validate_create_input,
    validate_id,
    validate_mint_input,
    validate_update_input,
)
from layerv_qurl.errors import QURLError, QURLNetworkError, QURLTimeoutError

if TYPE_CHECKING:
    import builtins
    from collections.abc import Iterator, Sequence
    from datetime import datetime

    from layerv_qurl.types import (
        QURL,
        AccessPolicy,
        BatchCreateItem,
        BatchCreateOutput,
        CreateOutput,
        ListOutput,
        MintOutput,
        Quota,
        QURLStatus,
        ResolveOutput,
    )


class QURLClient:
    """Synchronous QURL API client.

    Usage::

        from layerv_qurl import QURLClient

        client = QURLClient(api_key="lv_live_xxx")

        # Create a protected link
        result = client.create(target_url="https://example.com", expires_in="24h")

        # Resolve an access token (opens firewall for your IP)
        access = client.resolve("at_k8xqp9h2sj9lx7r4a")

        # Extend a QURL's expiration
        qurl = client.extend("r_xxx", "7d")

        # Update metadata
        qurl = client.update("r_xxx", description="updated")

        # Iterate all active QURLs
        for qurl in client.list_all(status="active"):
            print(qurl.resource_id)

    Enable debug logging to see requests::

        import logging
        logging.getLogger("layerv_qurl").setLevel(logging.DEBUG)
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        user_agent: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("api_key must not be empty")

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_retries = max_retries
        self._user_agent = user_agent or default_user_agent()
        self._client = http_client or httpx.Client(timeout=timeout)
        self._owns_client = http_client is None
        self._base_headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": self._user_agent,
        }

    def __repr__(self) -> str:
        return f"QURLClient(api_key='{mask_key(self._api_key)}', base_url='{self._base_url}')"

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
        expires_in: str | None = None,
        label: str | None = None,
        one_time_use: bool | None = None,
        max_sessions: int | None = None,
        session_duration: str | None = None,
        access_policy: AccessPolicy | None = None,
        custom_domain: str | None = None,
    ) -> CreateOutput:
        """Create a new QURL.

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
            expires_in: Duration string (e.g. ``"24h"``, ``"7d"``). The API
                uses ``expires_in`` on create; use :meth:`update` with
                ``expires_at`` if you need an absolute expiry afterwards.
            label: Human-readable label for the QURL. Max length 500.
            one_time_use: If True, the QURL is consumed on first access.
            max_sessions: Maximum concurrent sessions (0 = unlimited).
                Must be between 0 and 1000 inclusive.
            session_duration: Duration string for sessions (e.g. ``"1h"``).
            access_policy: IP/geo/user-agent access restrictions.
            custom_domain: Custom domain for the QURL link. Max length 253.

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
        resp = self._request("POST", "/v1/qurls", body=body)
        return parse_create_output(resp)

    def get(self, resource_id: str) -> QURL:
        """Get a QURL resource and its access tokens.

        Accepts either a resource ID (``r_`` prefix) or a QURL display ID
        (``q_`` prefix); the API resolves ``q_`` IDs to the parent resource
        automatically.

        Args:
            resource_id: The resource or QURL display ID.
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
        """List QURLs with optional filters.

        Args:
            limit: Maximum number of results per page.
            cursor: Pagination cursor from a previous response.
            status: Filter by QURL status (``"active"``, ``"revoked"``).
            q: Search query string.
            sort: Sort order (e.g. ``"created_at"``, ``"-created_at"``).
            created_after: Filter QURLs created after this timestamp.
                Accepts a :class:`datetime` (serialized via ``.isoformat()``)
                or a string. String values must be ISO 8601 / RFC 3339
                format (e.g. ``"2026-04-01T00:00:00Z"``) and are passed
                through to the API unvalidated — the server rejects
                malformed timestamps with a 400.
            created_before: Filter QURLs created before this timestamp.
                Same format rules as ``created_after``.
            expires_before: Filter QURLs expiring before this timestamp.
                Same format rules as ``created_after``.
            expires_after: Filter QURLs expiring after this timestamp.
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
        """Iterate over all QURLs, automatically paginating.

        Yields individual :class:`QURL` objects, fetching pages transparently.

        Args:
            status: Filter by status (``"active"``, ``"revoked"``).
            q: Search query string.
            sort: Sort order.
            page_size: Number of items per page (default 50).
            created_after: Filter QURLs created after this timestamp.
                Accepts a :class:`datetime` (serialized via ``.isoformat()``)
                or a string. String values must be ISO 8601 / RFC 3339
                format (e.g. ``"2026-04-01T00:00:00Z"``) and are passed
                through to the API unvalidated — the server rejects
                malformed timestamps with a 400.
            created_before: Filter QURLs created before this timestamp.
                Same format rules as ``created_after``.
            expires_before: Filter QURLs expiring before this timestamp.
                Same format rules as ``created_after``.
            expires_after: Filter QURLs expiring after this timestamp.
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
        """Delete (revoke) a QURL resource and all its access tokens.

        Only accepts a resource ID (``r_`` prefix), not a QURL display ID
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
        """Extend a QURL's expiration.

        Convenience method — equivalent to ``update(resource_id, extend_by=duration)``.
        Accepts either a resource ID (``r_`` prefix) or a QURL display ID
        (``q_`` prefix); the API resolves ``q_`` IDs to the parent resource
        automatically.

        Args:
            resource_id: Resource or QURL display ID.
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
    ) -> QURL:
        """Update a QURL — extend expiration, change description, set tags.

        Accepts either a resource ID (``r_`` prefix) or a QURL display ID
        (``q_`` prefix). All fields are optional, but at least one must be
        provided. ``extend_by`` and ``expires_at`` are mutually exclusive.

        Args:
            resource_id: Resource or QURL display ID.
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
        resp = self._request("PATCH", f"/v1/qurls/{resource_id}", body=body)
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
    ) -> MintOutput:
        """Mint a new access link for a QURL.

        Accepts either a resource ID (``r_`` prefix) or a QURL display ID
        (``q_`` prefix). ``expires_in`` and ``expires_at`` are mutually
        exclusive — if neither is set, the link defaults to 24 hours.

        Args:
            resource_id: Resource or QURL display ID.
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
        resp = self._request("POST", f"/v1/qurls/{resource_id}/mint_link", body=body)
        return parse_mint_output(resp)

    def batch_create(
        self,
        items: Sequence[BatchCreateItem],
    ) -> BatchCreateOutput:
        """Create multiple QURLs at once (1-100 items).

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
        resp = self._request(
            "POST",
            "/v1/qurls/batch",
            body={"items": serialized},
            allow_statuses=(400,),
        )
        # Defense-in-depth: the 400 passthrough trusts the response shape,
        # but if the API ever returns 400 with a non-BatchCreateOutput body
        # (e.g., a plain error envelope or malformed JSON) we'd silently
        # get an empty result. Verify the shape before parsing and raise
        # a clear error otherwise.
        _validate_batch_create_shape(resp)
        return parse_batch_create_output(resp)

    def resolve(self, access_token: str) -> ResolveOutput:
        """Resolve a QURL access token (headless).

        Triggers an NHP knock to open firewall access for the caller's IP.
        Requires ``qurl:resolve`` scope on the API key.

        Args:
            access_token: The access token string (e.g. ``"at_k8xqp9h2sj9lx7r4a"``).
        """
        validate_id(access_token, "access_token")
        resp = self._request("POST", "/v1/resolve", body={"access_token": access_token})
        return parse_resolve_output(resp)

    def get_quota(self) -> Quota:
        """Get quota and usage information."""
        resp = self._request("GET", "/v1/quota")
        return parse_quota(resp)

    # --- Internal HTTP plumbing ---

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        allow_statuses: tuple[int, ...] = (),
    ) -> Any:
        data, _ = self._raw_request(
            method, path, body=body, params=params, allow_statuses=allow_statuses
        )
        return data

    def _raw_request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        allow_statuses: tuple[int, ...] = (),
    ) -> tuple[Any, dict[str, Any] | None]:
        """Issue an HTTP request and parse the JSON envelope.

        ``allow_statuses`` lets a caller opt specific non-2xx codes out of
        the default raise-on-error path and receive the parsed body
        instead. This is used by :meth:`batch_create`, where the API
        returns a structured ``BatchCreateOutput`` on HTTP 400 (all items
        rejected) — raising would drop the per-item errors.
        """
        url = f"{self._base_url}{path}"
        last_error: Exception | None = None

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
                    headers=self._base_headers,
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
                envelope = response.json()
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
