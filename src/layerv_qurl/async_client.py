"""Asynchronous QURL API client.

NOTE: Business logic mirrors client.py — keep both in sync.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx

from layerv_qurl._utils import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    RETRYABLE_STATUS,
    RETRYABLE_STATUS_POST,
    build_body,
    build_list_params,
    default_user_agent,
    logger,
    mask_key,
    parse_create_output,
    parse_error,
    parse_list_output,
    parse_mint_output,
    parse_quota,
    parse_qurl,
    parse_resolve_output,
    retry_delay,
    validate_id,
)
from layerv_qurl.errors import QURLError, QURLNetworkError, QURLTimeoutError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime

    from layerv_qurl.types import (
        QURL,
        AccessPolicy,
        CreateOutput,
        ListOutput,
        MintOutput,
        Quota,
        QURLStatus,
        ResolveOutput,
    )


class AsyncQURLClient:
    """Asynchronous QURL API client.

    Usage::

        import asyncio
        from layerv_qurl import AsyncQURLClient

        async def main():
            async with AsyncQURLClient(api_key="lv_live_xxx") as client:
                result = await client.create(target_url="https://example.com")
                access = await client.resolve("at_k8xqp9h2sj9lx7r4a")

        asyncio.run(main())
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        user_agent: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("api_key must not be empty")

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_retries = max_retries
        self._user_agent = user_agent or default_user_agent()
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = http_client is None
        self._base_headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": self._user_agent,
        }

    def __repr__(self) -> str:
        return f"AsyncQURLClient(api_key='{mask_key(self._api_key)}', base_url='{self._base_url}')"

    async def close(self) -> None:
        """Close the underlying HTTP client (only if owned by this instance)."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncQURLClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # --- Public API ---

    async def create(
        self,
        target_url: str,
        *,
        expires_in: str | None = None,
        expires_at: datetime | str | None = None,
        description: str | None = None,
        one_time_use: bool | None = None,
        max_sessions: int | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> CreateOutput:
        """Create a new QURL.

        Returns a :class:`CreateOutput` with the ``resource_id``, ``qurl_link``,
        ``qurl_site``, and ``expires_at``. Use :meth:`get` to fetch the full
        :class:`QURL` object with status, timestamps, and policy details.

        Args:
            target_url: The URL to protect.
            expires_in: Duration string (e.g. ``"24h"``, ``"7d"``).
            expires_at: Absolute expiry as datetime or ISO string.
            description: Human-readable description.
            one_time_use: If True, the QURL can only be used once.
            max_sessions: Maximum concurrent sessions allowed.
            access_policy: IP/geo/user-agent access restrictions.
        """
        body = build_body({
            "target_url": target_url,
            "expires_in": expires_in,
            "expires_at": expires_at,
            "description": description,
            "one_time_use": one_time_use,
            "max_sessions": max_sessions,
            "access_policy": access_policy,
        })
        resp = await self._request("POST", "/v1/qurl", body=body)
        return parse_create_output(resp)

    async def get(self, resource_id: str) -> QURL:
        """Get a QURL by ID."""
        validate_id(resource_id)
        resp = await self._request("GET", f"/v1/qurls/{resource_id}")
        return parse_qurl(resp)

    async def list(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
        status: QURLStatus | None = None,
        q: str | None = None,
        sort: str | None = None,
    ) -> ListOutput:
        """List QURLs with optional filters.

        Args:
            limit: Maximum number of results per page.
            cursor: Pagination cursor from a previous response.
            status: Filter by QURL status
                (``"active"``, ``"expired"``, ``"revoked"``, ``"consumed"``, ``"frozen"``).
            q: Search query string.
            sort: Sort order (e.g. ``"created_at"``, ``"-created_at"``).
        """
        params = build_list_params(limit, cursor, status, q, sort)
        data, meta = await self._raw_request("GET", "/v1/qurls", params=params)
        return parse_list_output(data, meta)

    async def list_all(
        self,
        *,
        status: QURLStatus | None = None,
        q: str | None = None,
        sort: str | None = None,
        page_size: int = 50,
    ) -> AsyncIterator[QURL]:
        """Iterate over all QURLs, automatically paginating.

        Yields individual :class:`QURL` objects, fetching pages transparently.
        """
        cursor: str | None = None
        while True:
            page = await self.list(
                limit=page_size, cursor=cursor, status=status, q=q, sort=sort,
            )
            for qurl in page.qurls:
                yield qurl
            if not page.has_more or not page.next_cursor:
                break
            cursor = page.next_cursor

    async def delete(self, resource_id: str) -> None:
        """Delete (revoke) a QURL."""
        validate_id(resource_id)
        await self._request("DELETE", f"/v1/qurls/{resource_id}")

    async def extend(self, resource_id: str, duration: str) -> QURL:
        """Extend a QURL's expiration.

        Convenience method — equivalent to ``await update(resource_id, extend_by=duration)``.

        Args:
            resource_id: QURL resource ID.
            duration: Duration to add (e.g. ``"7d"``, ``"24h"``).
        """
        return await self.update(resource_id, extend_by=duration)

    async def update(
        self,
        resource_id: str,
        *,
        extend_by: str | None = None,
        expires_at: datetime | str | None = None,
        description: str | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> QURL:
        """Update a QURL — extend expiration, change description, etc.

        All fields are optional; only provided fields are sent.

        Args:
            resource_id: QURL resource ID.
            extend_by: Duration to add (e.g. ``"7d"``).
            expires_at: New absolute expiry.
            description: New description.
            access_policy: New access restrictions.
        """
        validate_id(resource_id)
        body = build_body({
            "extend_by": extend_by,
            "expires_at": expires_at,
            "description": description,
            "access_policy": access_policy,
        })
        resp = await self._request("PATCH", f"/v1/qurls/{resource_id}", body=body)
        return parse_qurl(resp)

    async def mint_link(
        self,
        resource_id: str,
        *,
        expires_at: datetime | str | None = None,
    ) -> MintOutput:
        """Mint a new access link for a QURL.

        Args:
            resource_id: QURL resource ID.
            expires_at: Optional expiry override for the minted link.
        """
        validate_id(resource_id)
        body = build_body({"expires_at": expires_at})
        resp = await self._request("POST", f"/v1/qurls/{resource_id}/mint_link", body=body)
        return parse_mint_output(resp)

    async def resolve(self, access_token: str) -> ResolveOutput:
        """Resolve a QURL access token (headless).

        Triggers an NHP knock to open firewall access for the caller's IP.
        Requires ``qurl:resolve`` scope on the API key.

        Args:
            access_token: The access token string (e.g. ``"at_k8xqp9h2sj9lx7r4a"``).
        """
        validate_id(access_token, "access_token")
        resp = await self._request("POST", "/v1/resolve", body={"access_token": access_token})
        return parse_resolve_output(resp)

    async def get_quota(self) -> Quota:
        """Get quota and usage information."""
        resp = await self._request("GET", "/v1/quota")
        return parse_quota(resp)

    # --- Internal HTTP plumbing ---

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> Any:
        data, _ = await self._raw_request(method, path, body=body, params=params)
        return data

    async def _raw_request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:
        url = f"{self._base_url}{path}"
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                delay = retry_delay(attempt, last_error)
                logger.debug("Retry %d/%d after %.1fs", attempt, self._max_retries, delay)
                await asyncio.sleep(delay)

            logger.debug("%s %s", method, url)

            try:
                response = await self._client.request(
                    method,
                    url,
                    json=body if body is not None else None,
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

            if response.status_code < 400:
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
