"""Client construction, HTTP transport, retry, and error tests."""

from __future__ import annotations

import inspect
import json
import logging
import uuid
from datetime import datetime
from unittest.mock import patch

import httpx
import pytest
import respx

import layerv_qurl
import layerv_qurl.types as qurl_types
from layerv_qurl import (
    AsyncQURLClient,
    QURLClient,
    QURLError,
    QURLNetworkError,
    QURLTimeoutError,
)
from layerv_qurl._utils import (
    build_query_params,
    build_string_list,
    ensure_mutation_idempotency,
)
from layerv_qurl.errors import (
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from layerv_qurl.types import (
    AccessCode,
    APIKey,
    Domain,
    Webhook,
)
from tests.client_support import _ERR_429, _ERR_503, _QUOTA_OK, BASE_URL

# --- Constructor tests ---


def test_path_traversal_rejected(client: QURLClient) -> None:
    """resource_id with path traversal characters is rejected."""
    with pytest.raises(ValueError, match="Invalid resource_id"):
        client.get("../../admin/secrets")


def test_empty_resource_id_rejected(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="Invalid resource_id"):
        client.delete("")


def test_empty_api_key_raises() -> None:
    with pytest.raises(ValueError, match="api_key must not be empty"):
        QURLClient(api_key="")


def test_whitespace_api_key_raises() -> None:
    with pytest.raises(ValueError, match="api_key must not be empty"):
        QURLClient(api_key="   ")


def test_native_agent_http_lifecycle_is_not_exposed() -> None:
    for client_type in (QURLClient, AsyncQURLClient):
        assert not hasattr(client_type, "bootstrap_agent")
        assert "/v1/agent/bootstrap" not in inspect.getsource(client_type)

    for type_name in ("AgentBootstrapOutput", "NHPServerPeerInfo"):
        assert type_name not in layerv_qurl.__all__
        assert not hasattr(layerv_qurl, type_name)
        assert not hasattr(qurl_types, type_name)


def test_retired_credential_request_fields_are_not_exposed() -> None:
    for client_type in (QURLClient, AsyncQURLClient):
        source = inspect.getsource(client_type)
        for retired in ("key_type", "tunnel_slug", "purpose"):
            assert retired not in source

    for retired in ("purpose", "tunnel_slug"):
        assert not hasattr(qurl_types.APIKey(key_id="k", key_prefix="p", name="n"), retired)


def test_repr_masks_api_key() -> None:
    c = QURLClient(api_key="lv_live_abcdefghij", base_url=BASE_URL)
    r = repr(c)
    assert "lv_l" in r
    assert "ghij" in r
    assert "abcdefghij" not in r
    assert "QURLClient(" in r
    c.close()


def test_repr_short_api_key() -> None:
    c = QURLClient(api_key="short123", base_url=BASE_URL)
    r = repr(c)
    assert "***" in r
    assert "short123" not in r
    c.close()


def test_repr_masks_jwt_without_suffix_fragment() -> None:
    token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturetail"
    c = QURLClient(api_key=token, base_url=BASE_URL)
    r = repr(c)
    assert "eyJh***" in r
    assert "signaturetail" not in r
    assert token not in r
    c.close()


def test_repr_masks_non_jwt_two_dot_secret_normally() -> None:
    c = QURLClient(api_key="lv_live.segment.tail1234", base_url=BASE_URL)
    r = repr(c)
    assert "lv_l" in r
    assert "1234" in r
    assert "segment" not in r
    c.close()


def test_repr_no_auth_api_key_is_unquoted_none() -> None:
    c = QURLClient(base_url=BASE_URL)
    assert "api_key=None" in repr(c)
    c.close()


def test_secret_dataclass_repr_omits_one_time_values() -> None:
    webhook = Webhook(
        webhook_id="wh_secret",
        url="https://example.com/hook",
        events=["qurl.created"],
        secret="whsec_secret",
    )
    api_key = APIKey(
        key_id="key_secret",
        key_prefix="lv_live_abcd",
        name="Production",
        api_key="lv_live_secret",
    )
    access_code = AccessCode(
        access_code_id="ac_secret",
        resource_id="r_secret",
        code="ac_code_secret",
    )
    domain = Domain(
        domain="example.com",
        status="pending",
        verification_token="qurl_verify_secret",
    )

    assert "whsec_secret" not in repr(webhook)
    assert "lv_live_secret" not in repr(api_key)
    assert "ac_code_secret" not in repr(access_code)
    assert "qurl_verify_secret" in repr(domain)


@respx.mock
def test_error_handling(client: QURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/qurls/r_notfound0000").mock(
        return_value=httpx.Response(
            404,
            json={
                "error": {
                    "type": "https://api.qurl.link/problems/not_found",
                    "title": "Not Found",
                    "status": 404,
                    "detail": "QURL not found",
                    "code": "not_found",
                },
                "meta": {"request_id": "req_err"},
            },
        )
    )

    with pytest.raises(QURLError) as exc_info:
        client.get("r_notfound0000")

    err = exc_info.value
    assert err.status == 404
    assert err.code == "not_found"
    assert err.request_id == "req_err"


# --- Injected http_client ---


@respx.mock
def test_injected_http_client_gets_auth_headers() -> None:
    custom_client = httpx.Client(timeout=10)
    qurl = QURLClient(api_key="lv_live_custom", base_url=BASE_URL, http_client=custom_client)

    route = respx.get(f"{BASE_URL}/v1/quota").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "plan": "free",
                    "period_start": "2026-03-01T00:00:00Z",
                    "period_end": "2026-04-01T00:00:00Z",
                },
            },
        )
    )

    qurl.get_quota()
    assert route.called
    req = route.calls[0].request
    assert req.headers["authorization"] == "Bearer lv_live_custom"
    # Content-Type should NOT be set for GET requests
    assert "content-type" not in req.headers
    custom_client.close()


# --- Retry logic ---


@respx.mock
def test_retry_success_after_429(retry_client: QURLClient) -> None:
    route = respx.get(f"{BASE_URL}/v1/quota")
    route.side_effect = [
        httpx.Response(429, json=_ERR_429),
        httpx.Response(200, json=_QUOTA_OK),
    ]

    with patch("layerv_qurl.client.time.sleep"):
        result = retry_client.get_quota()

    assert result.plan == "growth"
    assert route.call_count == 2


@respx.mock
def test_retry_exhausted_raises_last_error(retry_client: QURLClient) -> None:
    route = respx.get(f"{BASE_URL}/v1/quota")
    route.side_effect = [
        httpx.Response(503, json=_ERR_503),
        httpx.Response(503, json=_ERR_503),
        httpx.Response(503, json=_ERR_503),
    ]

    with patch("layerv_qurl.client.time.sleep"), pytest.raises(QURLError) as exc_info:
        retry_client.get_quota()

    assert exc_info.value.status == 503
    assert route.call_count == 3


@respx.mock
def test_retry_after_header_respected(retry_client: QURLClient) -> None:
    route = respx.get(f"{BASE_URL}/v1/quota")
    route.side_effect = [
        httpx.Response(
            429,
            headers={"Retry-After": "5"},
            json=_ERR_429,
        ),
        httpx.Response(200, json=_QUOTA_OK),
    ]

    with patch("layerv_qurl.client.time.sleep") as mock_sleep:
        result = retry_client.get_quota()

    assert result.plan == "growth"
    mock_sleep.assert_called_once_with(5.0)


@respx.mock
def test_retry_after_capped_at_30s(retry_client: QURLClient) -> None:
    route = respx.get(f"{BASE_URL}/v1/quota")
    route.side_effect = [
        httpx.Response(
            429,
            headers={"Retry-After": "120"},
            json=_ERR_429,
        ),
        httpx.Response(200, json=_QUOTA_OK),
    ]

    with patch("layerv_qurl.client.time.sleep") as mock_sleep:
        retry_client.get_quota()

    mock_sleep.assert_called_once_with(30.0)


@respx.mock
def test_retry_after_http_date_falls_back_to_exponential_backoff(
    retry_client: QURLClient,
) -> None:
    """Per RFC 7231 §7.1.3, ``Retry-After`` can be either a delay-seconds
    integer OR an HTTP-date. The SDK's ``.isdigit()`` check accepts only
    the integer form — HTTP-date strings (which contain letters/spaces/
    commas) fall through and the retry uses exponential backoff instead.

    This is a safe fallback: we don't honor the server's exact hint,
    but we also don't hang waiting for a parsed date value or crash on
    the unexpected header format. Locks in the intentional behavior
    against a future refactor that might try to parse HTTP-dates
    eagerly and introduce a new bug class.

    Mirrors the qurl-typescript SDK's same-named test for cross-SDK
    parity.
    """
    route = respx.get(f"{BASE_URL}/v1/quota")
    route.side_effect = [
        httpx.Response(
            429,
            # Valid RFC 7231 HTTP-date format — the SDK should NOT parse this.
            headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"},
            json=_ERR_429,
        ),
        httpx.Response(200, json=_QUOTA_OK),
    ]

    with patch("layerv_qurl.client.time.sleep") as mock_sleep:
        result = retry_client.get_quota()

    # The retry still fires and succeeds — the HTTP-date header
    # doesn't crash anything. We don't assert the exact sleep value
    # because that's exponential-backoff territory (with jitter),
    # but we do assert that sleep was called (meaning the retry
    # path ran) with a positive float.
    assert result.plan == "growth"
    mock_sleep.assert_called_once()
    (call_arg,) = mock_sleep.call_args.args
    assert isinstance(call_arg, float)
    assert call_arg > 0
    # Must NOT be the literal 0 or any absurdly large value — just
    # bounded by the retry_delay() cap at 30s.
    assert 0 < call_arg <= 30.0


@respx.mock
@pytest.mark.asyncio
async def test_async_retry_after_http_date_falls_back_to_exponential_backoff() -> None:
    """Async mirror of the sync HTTP-date fallback test. Locks in
    sync/async parity for the RFC 7231 ``Retry-After: <HTTP-date>``
    fallback behavior.

    Constructs a fresh AsyncQURLClient instead of using the
    async_client fixture because the fixture is configured with
    max_retries=0, and this test specifically needs retries enabled
    to exercise the retry-after fallback path.
    """
    client = AsyncQURLClient(api_key="lv_live_test", base_url=BASE_URL, max_retries=2)
    try:
        route = respx.get(f"{BASE_URL}/v1/quota")
        route.side_effect = [
            httpx.Response(
                429,
                headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"},
                json=_ERR_429,
            ),
            httpx.Response(200, json=_QUOTA_OK),
        ]

        with patch("layerv_qurl.async_client.asyncio.sleep") as mock_sleep:
            result = await client.get_quota()

        assert result.plan == "growth"
        mock_sleep.assert_called_once()
        (call_arg,) = mock_sleep.call_args.args
        assert isinstance(call_arg, float)
        assert 0 < call_arg <= 30.0
    finally:
        await client.close()


# --- Mutation retry safety ---
# Mutating requests carry an idempotency key across internal retries, but the
# client still deliberately restricts POST status-code retries to {429}. These
# tests lock that split in so a future refactor that naively unifies retry sets
# across methods will trip the guard.


@respx.mock
def test_post_does_not_retry_on_503(retry_client: QURLClient) -> None:
    """POST /v1/qurls must not retry on 5xx even when retries are configured."""
    route = respx.post(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(503, json=_ERR_503),
    )

    with patch("layerv_qurl.client.time.sleep"), pytest.raises(QURLError) as exc_info:
        retry_client.create(target_url="https://example.com", expires_in="24h")

    assert exc_info.value.status == 503
    # retry_client has max_retries=2, so a retry-on-503 regression would
    # produce 3 attempts. Assert exactly one — no retries for POST 5xx.
    assert route.call_count == 1


@respx.mock
def test_post_still_retries_on_429(retry_client: QURLClient) -> None:
    """POST retries on 429 specifically (rate limits are safe to retry)."""
    route = respx.post(f"{BASE_URL}/v1/qurls")
    route.side_effect = [
        httpx.Response(429, json=_ERR_429),
        httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_abc123def45",
                    "qurl_link": "https://qurl.link/#at_test",
                    "qurl_site": "https://r_abc123def45.qurl.site",
                    "qurl_id": "q_abc",
                },
            },
        ),
    ]

    with patch("layerv_qurl.client.time.sleep"):
        result = retry_client.create(target_url="https://example.com", expires_in="24h")

    assert result.resource_id == "r_abc123def45"
    assert route.call_count == 2
    first_key = route.calls[0].request.headers["idempotency-key"]
    assert first_key == route.calls[1].request.headers["idempotency-key"]
    assert len(first_key) == 36


@respx.mock
def test_post_network_retry_reuses_auto_idempotency_key(
    retry_client: QURLClient,
) -> None:
    route = respx.post(f"{BASE_URL}/v1/qurls")
    route.side_effect = [
        httpx.ConnectError("connection reset"),
        httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_abc123def45",
                    "qurl_link": "https://qurl.link/#at_test",
                    "qurl_site": "https://r_abc123def45.qurl.site",
                    "qurl_id": "q_abc",
                },
            },
        ),
    ]

    with patch("layerv_qurl.client.time.sleep"):
        result = retry_client.create(target_url="https://example.com", expires_in="24h")

    assert result.resource_id == "r_abc123def45"
    assert route.call_count == 2
    first_key = route.calls[0].request.headers["idempotency-key"]
    assert first_key == route.calls[1].request.headers["idempotency-key"]
    assert len(first_key) == 36


@respx.mock
def test_auto_idempotency_applies_to_supported_mutations(client: QURLClient) -> None:
    quota_route = respx.get(f"{BASE_URL}/v1/quota").mock(
        return_value=httpx.Response(200, json=_QUOTA_OK)
    )
    customer_route = respx.patch(f"{BASE_URL}/v1/customer").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "tier": "growth",
                    "spending_cap_cents": 100,
                    "current_period_usage": 0,
                },
            },
        )
    )
    webhook_route = respx.delete(f"{BASE_URL}/v1/webhooks/wh_abcdefghijklmnop").mock(
        return_value=httpx.Response(204)
    )

    client.get_quota()
    client.update_customer(spending_cap_cents=100)
    client.delete_webhook("wh_abcdefghijklmnop")

    assert "idempotency-key" not in quota_route.calls[0].request.headers
    assert len(customer_route.calls[0].request.headers["idempotency-key"]) == 36
    assert "idempotency-key" not in webhook_route.calls[0].request.headers


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH"])
def test_ensure_mutation_idempotency_generates_uuid7_for_supported_methods(
    method: str,
) -> None:
    headers: dict[str, str] = {}

    ensure_mutation_idempotency(method, headers)

    parsed = uuid.UUID(headers["Idempotency-Key"])
    assert parsed.version == 7
    assert parsed.variant == uuid.RFC_4122


def test_ensure_mutation_idempotency_preserves_explicit_header() -> None:
    headers = {"idempotency-key": "caller-provided-key"}

    ensure_mutation_idempotency("POST", headers)

    assert headers == {"idempotency-key": "caller-provided-key"}


@respx.mock
def test_patch_retry_reuses_auto_idempotency_key(retry_client: QURLClient) -> None:
    route = respx.patch(f"{BASE_URL}/v1/qurls/r_abc123def45")
    route.side_effect = [
        httpx.Response(503, json=_ERR_503),
        httpx.Response(
            200,
            json={
                "data": {
                    "resource_id": "r_abc123def45",
                    "target_url": "https://example.com",
                    "status": "active",
                    "created_at": "2026-03-10T10:00:00Z",
                    "expires_at": "2026-03-20T10:00:00Z",
                    "tags": [],
                },
            },
        ),
    ]

    with patch("layerv_qurl.client.time.sleep"):
        result = retry_client.update("r_abc123def45", extend_by="7d")

    assert isinstance(result.expires_at, datetime)
    assert route.call_count == 2
    first_key = route.calls[0].request.headers["idempotency-key"]
    assert first_key == route.calls[1].request.headers["idempotency-key"]
    assert len(first_key) == 36
    assert json.loads(route.calls[0].request.content) == {"extend_by": "7d"}
    assert json.loads(route.calls[1].request.content) == {"extend_by": "7d"}


@respx.mock
@pytest.mark.asyncio
async def test_async_post_does_not_retry_on_503() -> None:
    """Async POST must also not retry on 5xx — sync/async parity guard."""
    client = AsyncQURLClient(api_key="lv_live_test", base_url=BASE_URL, max_retries=2)
    try:
        route = respx.post(f"{BASE_URL}/v1/qurls").mock(
            return_value=httpx.Response(503, json=_ERR_503),
        )

        with (
            patch("layerv_qurl.async_client.asyncio.sleep"),
            pytest.raises(QURLError) as exc_info,
        ):
            await client.create(target_url="https://example.com", expires_in="24h")

        assert exc_info.value.status == 503
        assert route.call_count == 1
        assert len(route.calls[0].request.headers["idempotency-key"]) == 36
    finally:
        await client.close()


@respx.mock
@pytest.mark.asyncio
async def test_async_patch_retry_reuses_auto_idempotency_key() -> None:
    client = AsyncQURLClient(api_key="lv_live_test", base_url=BASE_URL, max_retries=2)
    try:
        route = respx.patch(f"{BASE_URL}/v1/qurls/r_abc")
        route.side_effect = [
            httpx.Response(503, json=_ERR_503),
            httpx.Response(
                200,
                json={
                    "data": {
                        "resource_id": "r_abc",
                        "target_url": "https://example.com",
                        "status": "active",
                        "created_at": "2026-03-10T10:00:00Z",
                        "expires_at": "2026-03-20T10:00:00Z",
                        "tags": [],
                    },
                },
            ),
        ]

        with patch("layerv_qurl.async_client.asyncio.sleep"):
            result = await client.update("r_abc", extend_by="7d")

        assert isinstance(result.expires_at, datetime)
        assert route.call_count == 2
        first_key = route.calls[0].request.headers["idempotency-key"]
        assert first_key == route.calls[1].request.headers["idempotency-key"]
        assert len(first_key) == 36
        assert json.loads(route.calls[0].request.content) == {"extend_by": "7d"}
        assert json.loads(route.calls[1].request.content) == {"extend_by": "7d"}
    finally:
        await client.close()


@respx.mock
@pytest.mark.asyncio
async def test_async_post_network_retry_reuses_auto_idempotency_key() -> None:
    client = AsyncQURLClient(api_key="lv_live_test", base_url=BASE_URL, max_retries=2)
    try:
        route = respx.post(f"{BASE_URL}/v1/qurls")
        route.side_effect = [
            httpx.ConnectError("connection reset"),
            httpx.Response(
                201,
                json={
                    "data": {
                        "resource_id": "r_async",
                        "qurl_link": "https://qurl.link/#at_async",
                        "qurl_site": "https://r_async.qurl.site",
                        "qurl_id": "q_async",
                    },
                },
            ),
        ]

        with patch("layerv_qurl.async_client.asyncio.sleep"):
            result = await client.create(
                target_url="https://example.com",
                expires_in="24h",
            )

        assert result.resource_id == "r_async"
        assert route.call_count == 2
        first_key = route.calls[0].request.headers["idempotency-key"]
        assert first_key == route.calls[1].request.headers["idempotency-key"]
        assert len(first_key) == 36
    finally:
        await client.close()


# --- Non-JSON error ---


@respx.mock
def test_non_json_error_response(client: QURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/qurls/r_bad").mock(
        return_value=httpx.Response(
            500,
            text="Internal Server Error",
            headers={"content-type": "text/plain"},
        )
    )

    with pytest.raises(QURLError) as exc_info:
        client.get("r_bad")

    err = exc_info.value
    assert err.status == 500
    assert err.code == "unknown"
    assert "Internal Server Error" in err.detail


# --- Network error wrapping ---


@respx.mock
def test_network_error_wrapped(client: QURLClient) -> None:
    """httpx errors are wrapped in QURLNetworkError."""
    respx.get(f"{BASE_URL}/v1/quota").mock(side_effect=httpx.ConnectError("Connection refused"))

    with pytest.raises(QURLNetworkError, match="Connection refused"):
        client.get_quota()


@respx.mock
def test_network_error_preserves_cause(client: QURLClient) -> None:
    """QURLNetworkError preserves the original httpx exception as __cause__."""
    respx.get(f"{BASE_URL}/v1/quota").mock(side_effect=httpx.ConnectError("DNS lookup failed"))

    with pytest.raises(QURLNetworkError) as exc_info:
        client.get_quota()

    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)


@respx.mock
def test_timeout_error_wrapped(client: QURLClient) -> None:
    """httpx.TimeoutException is wrapped in QURLTimeoutError."""
    respx.get(f"{BASE_URL}/v1/quota").mock(side_effect=httpx.ReadTimeout("Read timed out"))

    with pytest.raises(QURLTimeoutError, match="Read timed out"):
        client.get_quota()


@respx.mock
def test_timeout_error_is_network_error(client: QURLClient) -> None:
    """QURLTimeoutError is a subclass of QURLNetworkError."""
    respx.get(f"{BASE_URL}/v1/quota").mock(side_effect=httpx.ReadTimeout("Read timed out"))

    with pytest.raises(QURLNetworkError):
        client.get_quota()


@respx.mock
def test_timeout_retried_then_wrapped() -> None:
    """Timeout errors are retried, then wrapped as QURLTimeoutError."""
    c = QURLClient(api_key="lv_live_test", base_url=BASE_URL, max_retries=1)
    route = respx.get(f"{BASE_URL}/v1/quota")
    route.side_effect = [
        httpx.ReadTimeout("timeout 1"),
        httpx.ReadTimeout("timeout 2"),
    ]

    with patch("layerv_qurl.client.time.sleep"), pytest.raises(QURLTimeoutError):
        c.get_quota()

    assert route.call_count == 2


@respx.mock
def test_network_error_retried_then_wrapped() -> None:
    """Network errors are retried, then wrapped if all retries fail."""
    c = QURLClient(api_key="lv_live_test", base_url=BASE_URL, max_retries=1)
    route = respx.get(f"{BASE_URL}/v1/quota")
    route.side_effect = [
        httpx.ConnectError("fail 1"),
        httpx.ConnectError("fail 2"),
    ]

    with patch("layerv_qurl.client.time.sleep"), pytest.raises(QURLNetworkError):
        c.get_quota()

    assert route.call_count == 2


# --- Context manager / close() tests ---


def test_close_closes_owned_client() -> None:
    c = QURLClient(api_key="lv_live_test", base_url=BASE_URL)
    assert c._owns_client is True
    c.close()
    assert c._client.is_closed


def test_close_does_not_close_injected_client() -> None:
    custom = httpx.Client(timeout=10)
    c = QURLClient(api_key="lv_live_test", base_url=BASE_URL, http_client=custom)
    assert c._owns_client is False
    c.close()
    assert not custom.is_closed
    custom.close()


def test_context_manager_closes_owned_client() -> None:
    with QURLClient(api_key="lv_live_test", base_url=BASE_URL) as c:
        assert c._owns_client is True
    assert c._client.is_closed


def test_context_manager_does_not_close_injected_client() -> None:
    custom = httpx.Client(timeout=10)
    with QURLClient(api_key="lv_live_test", base_url=BASE_URL, http_client=custom) as c:
        assert c._owns_client is False
    assert not custom.is_closed
    custom.close()


@respx.mock
def test_post_request_has_content_type(client: QURLClient) -> None:
    """POST requests with body should send Content-Type: application/json."""
    route = respx.post(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_abc",
                    "qurl_link": "https://qurl.link/#at_test",
                    "qurl_site": "https://r_abc.qurl.site",
                    "qurl_id": "",
                },
            },
        )
    )

    client.create(target_url="https://example.com")
    req = route.calls[0].request
    assert req.headers["content-type"] == "application/json"


@respx.mock
@pytest.mark.asyncio
async def test_async_network_error_wrapped(async_client: AsyncQURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/quota").mock(side_effect=httpx.ConnectError("Connection refused"))

    with pytest.raises(QURLNetworkError, match="Connection refused"):
        await async_client.get_quota()


@respx.mock
@pytest.mark.asyncio
async def test_async_timeout_error_wrapped(async_client: AsyncQURLClient) -> None:
    """Async: httpx.TimeoutException is wrapped in QURLTimeoutError."""
    respx.get(f"{BASE_URL}/v1/quota").mock(side_effect=httpx.ReadTimeout("Read timed out"))

    with pytest.raises(QURLTimeoutError, match="Read timed out"):
        await async_client.get_quota()


@respx.mock
@pytest.mark.asyncio
async def test_async_timeout_is_network_error(async_client: AsyncQURLClient) -> None:
    """Async: QURLTimeoutError is caught by except QURLNetworkError."""
    respx.get(f"{BASE_URL}/v1/quota").mock(side_effect=httpx.ReadTimeout("Read timed out"))

    with pytest.raises(QURLNetworkError):
        await async_client.get_quota()


def test_async_repr() -> None:
    c = AsyncQURLClient(api_key="lv_live_abcdefghij", base_url=BASE_URL)
    r = repr(c)
    assert "AsyncQURLClient(" in r
    assert "lv_l" in r
    assert "ghij" in r
    assert "abcdefghij" not in r


# --- Error subclass tests ---


@respx.mock
def test_401_raises_authentication_error(client: QURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/quota").mock(
        return_value=httpx.Response(
            401,
            json={
                "error": {
                    "status": 401,
                    "code": "unauthorized",
                    "title": "Unauthorized",
                    "detail": "Invalid API key",
                },
            },
        )
    )

    with pytest.raises(AuthenticationError) as exc_info:
        client.get_quota()
    assert exc_info.value.status == 401
    assert isinstance(exc_info.value, QURLError)


@respx.mock
def test_403_raises_authorization_error(client: QURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/quota").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": {
                    "status": 403,
                    "code": "forbidden",
                    "title": "Forbidden",
                    "detail": "Insufficient scope",
                },
            },
        )
    )

    with pytest.raises(AuthorizationError) as exc_info:
        client.get_quota()
    assert exc_info.value.status == 403


@respx.mock
def test_404_raises_not_found_error(client: QURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/qurls/r_notfound0000").mock(
        return_value=httpx.Response(
            404,
            json={
                "error": {
                    "status": 404,
                    "code": "not_found",
                    "title": "Not Found",
                    "detail": "QURL not found",
                },
                "meta": {"request_id": "req_err"},
            },
        )
    )

    with pytest.raises(NotFoundError) as exc_info:
        client.get("r_notfound0000")
    assert exc_info.value.status == 404
    assert exc_info.value.request_id == "req_err"


@respx.mock
def test_422_raises_validation_error(client: QURLClient) -> None:
    """The API's 422 path is exercised by sending a syntactically valid URL
    that passes client-side checks but is rejected by the API (e.g. a
    host that fails SSRF protection). The mock returns 422 regardless of
    the request body, so any valid-scheme URL works here."""
    respx.post(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            422,
            json={
                "error": {
                    "status": 422,
                    "code": "validation_error",
                    "title": "Validation Error",
                    "detail": "Invalid target_url",
                    "invalid_fields": {"target_url": "must be a valid URL"},
                },
            },
        )
    )

    with pytest.raises(ValidationError) as exc_info:
        client.create(target_url="https://localhost/rejected-by-ssrf-protection")
    assert exc_info.value.invalid_fields == {"target_url": "must be a valid URL"}


@respx.mock
def test_429_raises_rate_limit_error(client: QURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/quota").mock(
        return_value=httpx.Response(
            429,
            headers={"Retry-After": "10"},
            json={
                "error": {
                    "status": 429,
                    "code": "rate_limited",
                    "title": "Rate Limited",
                    "detail": "Slow down",
                },
            },
        )
    )

    with pytest.raises(RateLimitError) as exc_info:
        client.get_quota()
    assert exc_info.value.retry_after == 10


@respx.mock
def test_500_raises_server_error(client: QURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/quota").mock(
        return_value=httpx.Response(
            500,
            json={
                "error": {
                    "status": 500,
                    "code": "internal",
                    "title": "Internal Server Error",
                    "detail": "Something broke",
                },
            },
        )
    )

    with pytest.raises(ServerError) as exc_info:
        client.get_quota()
    assert exc_info.value.status == 500


@respx.mock
def test_400_raises_validation_error(client: QURLClient) -> None:
    """Exercise the API's 400 path with a URL that passes client-side
    validation — the mock returns 400 regardless of the request body."""
    respx.post(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "status": 400,
                    "code": "bad_request",
                    "title": "Bad Request",
                    "detail": "Missing target_url",
                },
            },
        )
    )

    with pytest.raises(ValidationError):
        client.create(target_url="https://example.com/triggers-mocked-400")


def test_require_max_sessions_in_range_rejects_bool() -> None:
    """Direct unit test for the ``bool`` rejection in
    :func:`_require_max_sessions_in_range`.

    ``bool`` is a subclass of ``int`` in Python (``True == 1``,
    ``False == 0``), so a naive ``isinstance(value, int)`` check
    would silently accept ``max_sessions=True``. The validator has
    an explicit ``isinstance(value, bool)`` rejection to catch this.
    Reviewer's argument for this test: a future simplification that
    drops the bool guard would trip this regression immediately.

    The rejection is already exercised indirectly via batch bool-
    counts tests, but a direct unit test makes the intent explicit.
    """
    import layerv_qurl._utils as utils

    # `True` and `False` are type-compatible with `int` in Python (bool
    # is a subclass of int), so mypy doesn't need a type ignore — the
    # test exercises a RUNTIME check that catches what the type system
    # can't.
    with pytest.raises(ValueError, match="max_sessions"):
        utils._require_max_sessions_in_range(True)
    with pytest.raises(ValueError, match="max_sessions"):
        utils._require_max_sessions_in_range(False)
    # Sanity: plain ints still pass through.
    utils._require_max_sessions_in_range(0)
    utils._require_max_sessions_in_range(500)
    utils._require_max_sessions_in_range(1000)
    # And None (the "not provided" sentinel) is still a no-op.
    utils._require_max_sessions_in_range(None)


# ---- Error type/instance exposure, detail fallback, legacy message ---------


@respx.mock
def test_error_surfaces_rfc7807_type_and_instance(client: QURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/qurls/r_nf0000000000").mock(
        return_value=httpx.Response(
            404,
            json={
                "error": {
                    "type": "https://api.qurl.link/problems/not_found",
                    "title": "Not Found",
                    "status": 404,
                    "detail": "QURL not found",
                    "instance": "/v1/qurls/r_nf0000000000",
                    "code": "not_found",
                },
                "meta": {"request_id": "req_nf"},
            },
        )
    )
    with pytest.raises(NotFoundError) as excinfo:
        client.get("r_nf0000000000")
    err = excinfo.value
    assert err.type == "https://api.qurl.link/problems/not_found"
    assert err.instance == "/v1/qurls/r_nf0000000000"
    assert err.request_id == "req_nf"


@respx.mock
def test_error_handles_explicit_null_error_envelope(client: QURLClient) -> None:
    """Regression guard for the `envelope.get("error") or {}` pattern.

    Some APIs return ``{"error": null, ...}`` explicitly instead of
    omitting the ``error`` key entirely. The standard
    ``.get("error", {})`` would only handle the missing-key case —
    an explicit ``null`` would pass through and then crash on the
    subsequent ``err.get(...)`` chain with ``AttributeError: 'NoneType'
    object has no attribute 'get'``. The inline comment on ``parse_error``
    documents the intentional ``or {}`` form; this test locks it in
    against a future refactor that might "simplify" it back to the
    broken ``, {}`` form.
    """
    respx.get(f"{BASE_URL}/v1/quota").mock(
        return_value=httpx.Response(
            500,
            json={
                # Explicit null, not a missing key.
                "error": None,
                "meta": {"request_id": "req_null_err"},
            },
        )
    )
    with pytest.raises(ServerError) as excinfo:
        client.get_quota()
    err = excinfo.value
    # The `or {}` fallback collapses `err` to an empty dict, which
    # means none of the RFC 7807 fields are populated — we get the
    # pure fallback behavior: status from the HTTP response,
    # reason_phrase as the title, and "HTTP 500" as the detail.
    assert err.status == 500
    assert err.code == "unknown"
    # Detail falls back to title → reason_phrase → "HTTP {status}".
    # httpx's reason phrase for 500 is "Internal Server Error" or
    # empty depending on the transport; either way the detail is
    # not None and doesn't contain "None"/"undefined" placeholders.
    assert err.detail is not None
    assert "None" not in str(err)
    assert "undefined" not in str(err)


@respx.mock
def test_error_falls_back_to_title_when_detail_missing(client: QURLClient) -> None:
    """RFC 7807 `detail` is optional — fall back to title."""
    respx.get(f"{BASE_URL}/v1/quota").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": {
                    "type": "https://api.qurl.link/problems/forbidden",
                    "title": "Forbidden",
                    "status": 403,
                    "code": "forbidden",
                    # no detail
                },
            },
        )
    )
    with pytest.raises(AuthorizationError) as excinfo:
        client.get_quota()
    err = excinfo.value
    assert err.detail == "Forbidden"
    assert "None" not in str(err)
    assert "undefined" not in str(err)


@respx.mock
def test_error_falls_back_to_legacy_message_field(client: QURLClient) -> None:
    """Pre-RFC-7807 `{error: {code, message}}` envelope still works."""
    respx.get(f"{BASE_URL}/v1/quota").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "code": "invalid_request",
                    "message": "legacy-format detail string",
                },
            },
        )
    )
    with pytest.raises(ValidationError) as excinfo:
        client.get_quota()
    err = excinfo.value
    assert err.detail == "legacy-format detail string"


def test_build_query_params_serializes_booleans_lowercase() -> None:
    params = build_query_params({"enabled": True, "archived": False, "skip": None})
    assert params == {"enabled": "true", "archived": "false"}


@respx.mock
def test_latest_create_contract_fields_and_idempotency_header(client: QURLClient) -> None:
    route = respx.post(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "qurl_id": "q_abc123def45",
                    "resource_id": "r_abc123def45",
                    "qurl_link": "https://qurl.link/#at_test",
                    "branded_domain": "secure.example.com",
                    "qurl_site": "https://q_abc123def45.qurl.site",
                    "expires_at": "2026-03-15T10:00:00Z",
                    "label": "contract",
                    "type": "url",
                },
            },
        )
    )

    result = client.create(
        target_url="https://example.com",
        resource_type="url",
        idempotency_key="idem-create-1",
    )

    body = json.loads(route.calls[0].request.content)
    assert body["type"] == "url"
    assert route.calls[0].request.headers["idempotency-key"] == "idem-create-1"
    assert result.qurl_id == "q_abc123def45"
    assert result.branded_domain == "secure.example.com"
    assert result.resource_type == "url"


def test_string_sequence_fields_reject_bare_string_and_empty_lists(
    client: QURLClient,
) -> None:
    assert build_string_list((item for item in ["qurl.created"]), "events") == ["qurl.created"]

    with pytest.raises(ValueError, match="events"):
        client.create_webhook(
            url="https://example.com/webhook",
            events="qurl.created",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="cannot be empty"):
        client.update_webhook("wh_abcdefghijklmnop", events=[])

    with pytest.raises(ValueError, match="mapping"):
        build_string_list({"event": "qurl.created"}, "events")

    with pytest.raises(ValueError, match="set"):
        build_string_list({"qurl.created"}, "events")

    with pytest.raises(ValueError, match="scopes"):
        client.create_api_key(
            name="bad scopes",
            scopes="qurl:read",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="scopes"):
        client.update_api_key("key_abc123def456", scopes=[])


@respx.mock
def test_connector_contracts(client: QURLClient, caplog: pytest.LogCaptureFixture) -> None:
    respx.get(f"{BASE_URL}/v1/connectors/installations").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"installation_id": "inst_missing_identity"},
                    {
                        "installation_id": "inst_1",
                        "plugin_id": "slack",
                        "label": "Engineering Slack",
                        "subject_kind": "slack_workspace",
                        "subject_display_name": "LayerV Engineering",
                        "status": "active",
                        "installed_at": "2026-03-01T00:00:00Z",
                        "stats": {
                            "resources": 2,
                            "qurls": 4,
                            "accesses_24h": 1,
                            "accesses_7d": 5,
                            "errors_24h": 0,
                        },
                        "capabilities": {
                            "configure": True,
                            "disconnect": True,
                            "reauth": False,
                            "view_activity": True,
                        },
                    },
                ],
            },
        )
    )
    with caplog.at_level(logging.DEBUG, logger="layerv_qurl"):
        connector = client.list_connector_installations().installations[0]

    assert connector.installation_id == "inst_1"
    assert connector.stats is not None
    assert connector.stats.qurls == 4
    assert any(
        "parse_connector_installation" in record.message and "plugin_id" in record.message
        for record in caplog.records
    )


def test_sync_contract_lists_validate_limit_bounds(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        client.list_domains(limit=0)

    with pytest.raises(ValueError, match="between 1 and 100"):
        client.list_api_keys(limit=101)


def test_idempotency_key_validation(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="control characters"):
        client.create(target_url="https://example.com", idempotency_key="bad\nkey")

    with pytest.raises(ValueError, match="control characters"):
        client.create(target_url="https://example.com", idempotency_key="bad\tkey")

    with pytest.raises(ValueError, match="256 characters"):
        client.create(
            target_url="https://example.com",
            idempotency_key="x" * 257,
        )

    with pytest.raises(ValueError, match="at least 32 characters"):
        client.create_api_key(
            name="too short",
            scopes=["qurl:read"],
            idempotency_key="short",
        )


@respx.mock
@pytest.mark.asyncio
async def test_async_public_redeem_omits_auth_and_sends_idempotency() -> None:
    client = AsyncQURLClient(base_url=BASE_URL, max_retries=0)
    route = respx.post(f"{BASE_URL}/v1/access-codes/redeem").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"redirect_url": "https://qurl.link/#at_async_code"}},
        )
    )

    try:
        redeem = await client.redeem_access_code(
            code="ac_async",
            idempotency_key="idem-async-redeem",
        )
    finally:
        await client.close()

    assert redeem.redirect_url == "https://qurl.link/#at_async_code"
    assert "authorization" not in route.calls[0].request.headers
    assert route.calls[0].request.headers["idempotency-key"] == "idem-async-redeem"
    assert json.loads(route.calls[0].request.content) == {"code": "ac_async"}
