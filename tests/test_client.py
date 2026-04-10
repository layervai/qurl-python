"""Tests for the QURL Python client."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import httpx
import pytest
import respx

from layerv_qurl import (
    AsyncQURLClient,
    QURLClient,
    QURLError,
    QURLNetworkError,
    QURLTimeoutError,
)
from layerv_qurl.errors import (
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from layerv_qurl.types import AccessPolicy, AccessToken, AIAgentPolicy

BASE_URL = "https://api.test.layerv.ai"

_ERR_429 = {
    "error": {
        "status": 429,
        "code": "rate_limited",
        "title": "Rate Limited",
        "detail": "Slow down",
    },
}
_ERR_503 = {
    "error": {
        "status": 503,
        "code": "unavailable",
        "title": "Unavailable",
        "detail": "Down",
    },
}
_QUOTA_OK = {
    "data": {
        "plan": "growth",
        "period_start": "2026-03-01T00:00:00Z",
        "period_end": "2026-04-01T00:00:00Z",
    },
}


def _qurl_item(rid: str, url: str) -> dict:
    return {
        "resource_id": rid,
        "target_url": url,
        "status": "active",
        "created_at": "2026-03-10T10:00:00Z",
        "tags": [],
    }


@pytest.fixture
def client() -> QURLClient:
    return QURLClient(api_key="lv_live_test", base_url=BASE_URL, max_retries=0)


@pytest.fixture
async def async_client() -> AsyncQURLClient:
    client = AsyncQURLClient(api_key="lv_live_test", base_url=BASE_URL, max_retries=0)
    yield client  # type: ignore[misc]
    await client.close()


@pytest.fixture
def retry_client() -> QURLClient:
    return QURLClient(api_key="lv_live_test", base_url=BASE_URL, max_retries=2)


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


# --- CRUD tests with kwargs API ---


@respx.mock
def test_create(client: QURLClient) -> None:
    respx.post(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_abc123def45",
                    "qurl_link": "https://qurl.link/#at_test",
                    "qurl_site": "https://r_abc123def45.qurl.site",
                    "expires_at": "2026-03-15T10:00:00Z",
                    "qurl_id": "q_abc",
                },
                "meta": {"request_id": "req_1"},
            },
        )
    )

    result = client.create(target_url="https://example.com", expires_in="24h")
    assert result.resource_id == "r_abc123def45"
    assert result.qurl_link == "https://qurl.link/#at_test"
    assert result.qurl_site == "https://r_abc123def45.qurl.site"
    assert isinstance(result.expires_at, datetime)
    assert result.qurl_id == "q_abc"


@respx.mock
def test_create_sends_correct_body(client: QURLClient) -> None:
    route = respx.post(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_abc",
                    "qurl_link": "https://qurl.link/#at_test",
                    "qurl_site": "https://r_abc.qurl.site",
                    "qurl_id": "q_abc",
                },
            },
        )
    )

    client.create(
        target_url="https://example.com",
        expires_in="24h",
        label="test",
    )
    body = json.loads(route.calls[0].request.content)
    assert body == {
        "target_url": "https://example.com",
        "expires_in": "24h",
        "label": "test",
    }


@respx.mock
def test_create_omits_none_values(client: QURLClient) -> None:
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
    body = json.loads(route.calls[0].request.content)
    assert body == {"target_url": "https://example.com"}
    assert "expires_in" not in body
    assert "label" not in body


@respx.mock
def test_get(client: QURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/qurls/r_abc123def45").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "resource_id": "r_abc123def45",
                    "target_url": "https://example.com",
                    "status": "active",
                    "created_at": "2026-03-10T10:00:00Z",
                    "expires_at": "2026-03-15T10:00:00Z",
                    "tags": [],
                    "qurl_count": 2,
                    # API wire format uses "qurls"; parse_qurl maps to access_tokens
                    "qurls": [
                        {
                            "qurl_id": "at_abc",
                            "status": "active",
                            "one_time_use": True,
                            "max_sessions": 5,
                            "session_duration": 300,
                            "use_count": 1,
                            "label": "test token",
                            "qurl_site": "https://r_abc123def45.qurl.site",
                            "created_at": "2026-03-10T10:00:00Z",
                            "expires_at": "2026-03-15T10:00:00Z",
                        },
                    ],
                },
                "meta": {"request_id": "req_2"},
            },
        )
    )

    result = client.get("r_abc123def45")
    assert result.resource_id == "r_abc123def45"
    assert result.status == "active"
    assert isinstance(result.created_at, datetime)
    assert result.created_at == datetime(2026, 3, 10, 10, 0, 0, tzinfo=timezone.utc)
    assert result.qurl_count == 2
    assert result.access_tokens is not None
    assert len(result.access_tokens) == 1
    token = result.access_tokens[0]
    assert isinstance(token, AccessToken)
    assert token.qurl_id == "at_abc"
    assert token.one_time_use is True
    assert token.max_sessions == 5
    assert token.session_duration == 300
    assert token.use_count == 1
    assert token.label == "test token"


@respx.mock
def test_get_token_with_access_policy(client: QURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/qurls/r_abc123def45").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "resource_id": "r_abc123def45",
                    "target_url": "https://example.com",
                    "status": "active",
                    "created_at": "2026-03-10T10:00:00Z",
                    "qurls": [
                        {
                            "qurl_id": "q_abc12345678",
                            "status": "active",
                            "access_policy": {
                                "ip_allowlist": ["10.0.0.0/8"],
                                "geo_denylist": ["CN"],
                            },
                        },
                    ],
                },
                "meta": {"request_id": "req_p"},
            },
        )
    )

    result = client.get("r_abc123def45")
    assert result.access_tokens is not None
    token = result.access_tokens[0]
    assert token.access_policy is not None
    assert token.access_policy.ip_allowlist == ["10.0.0.0/8"]
    assert token.access_policy.geo_denylist == ["CN"]
    # Verify defaults on sparse token
    assert token.one_time_use is False
    assert token.max_sessions == 0
    assert token.session_duration == 0
    assert token.use_count == 0
    assert token.label is None
    assert token.qurl_site is None
    assert token.created_at is None
    assert token.expires_at is None


@respx.mock
def test_get_without_tokens(client: QURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/qurls/r_abc123def45").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "resource_id": "r_abc123def45",
                    "target_url": "https://example.com",
                    "status": "active",
                    "created_at": "2026-03-10T10:00:00Z",
                    "qurl_count": 0,
                },
                "meta": {"request_id": "req_x"},
            },
        )
    )

    result = client.get("r_abc123def45")
    assert result.resource_id == "r_abc123def45"
    assert result.qurl_count == 0
    assert result.access_tokens is None


@respx.mock
def test_list(client: QURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "resource_id": "r_abc123def45",
                        "target_url": "https://example.com",
                        "status": "active",
                        "created_at": "2026-03-10T10:00:00Z",
                        "tags": [],
                    }
                ],
                "meta": {"has_more": False, "page_size": 20},
            },
        )
    )

    result = client.list(status="active", limit=10)
    assert len(result.qurls) == 1
    assert result.qurls[0].resource_id == "r_abc123def45"
    assert result.has_more is False


@respx.mock
def test_list_all_paginates(client: QURLClient) -> None:
    route = respx.get(f"{BASE_URL}/v1/qurls")
    route.side_effect = [
        httpx.Response(
            200,
            json={
                "data": [_qurl_item("r_1", "https://1.com"), _qurl_item("r_2", "https://2.com")],
                "meta": {"has_more": True, "next_cursor": "cur_abc"},
            },
        ),
        httpx.Response(
            200,
            json={
                "data": [_qurl_item("r_3", "https://3.com")],
                "meta": {"has_more": False},
            },
        ),
    ]

    all_qurls = list(client.list_all(status="active", page_size=2))
    assert len(all_qurls) == 3
    assert [q.resource_id for q in all_qurls] == ["r_1", "r_2", "r_3"]
    assert route.call_count == 2


@respx.mock
def test_delete(client: QURLClient) -> None:
    respx.delete(f"{BASE_URL}/v1/qurls/r_abc123def45").mock(return_value=httpx.Response(204))
    client.delete("r_abc123def45")  # Should not raise


@respx.mock
def test_update_with_extend(client: QURLClient) -> None:
    route = respx.patch(f"{BASE_URL}/v1/qurls/r_abc123def45").mock(
        return_value=httpx.Response(
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
        )
    )

    result = client.update("r_abc123def45", extend_by="7d")
    assert isinstance(result.expires_at, datetime)
    body = json.loads(route.calls[0].request.content)
    assert body == {"extend_by": "7d"}


@respx.mock
def test_update_with_description(client: QURLClient) -> None:
    route = respx.patch(f"{BASE_URL}/v1/qurls/r_abc123def45").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "resource_id": "r_abc123def45",
                    "target_url": "https://example.com",
                    "status": "active",
                    "created_at": "2026-03-10T10:00:00Z",
                    "description": "new desc",
                    "tags": [],
                },
            },
        )
    )

    result = client.update("r_abc123def45", description="new desc")
    assert result.description == "new desc"
    body = json.loads(route.calls[0].request.content)
    assert body == {"description": "new desc"}


@respx.mock
def test_update_combined(client: QURLClient) -> None:
    """update() can extend and change description in one call."""
    route = respx.patch(f"{BASE_URL}/v1/qurls/r_abc").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "resource_id": "r_abc",
                    "target_url": "https://example.com",
                    "status": "active",
                    "created_at": "2026-03-10T10:00:00Z",
                    "expires_at": "2026-03-20T10:00:00Z",
                    "description": "updated",
                    "tags": [],
                },
            },
        )
    )

    client.update("r_abc", extend_by="7d", description="updated")
    body = json.loads(route.calls[0].request.content)
    assert body == {"extend_by": "7d", "description": "updated"}


@respx.mock
def test_mint_link(client: QURLClient) -> None:
    respx.post(f"{BASE_URL}/v1/qurls/r_abc123def45/mint_link").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "qurl_link": "https://qurl.link/#at_newtoken",
                    "expires_at": "2026-03-20T10:00:00Z",
                },
            },
        )
    )

    result = client.mint_link("r_abc123def45", expires_at="2026-03-20T10:00:00Z")
    assert result.qurl_link == "https://qurl.link/#at_newtoken"
    assert isinstance(result.expires_at, datetime)


@respx.mock
def test_mint_link_no_input(client: QURLClient) -> None:
    respx.post(f"{BASE_URL}/v1/qurls/r_abc123def45/mint_link").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "qurl_link": "https://qurl.link/#at_default",
                },
            },
        )
    )

    result = client.mint_link("r_abc123def45")
    assert result.qurl_link == "https://qurl.link/#at_default"
    assert result.expires_at is None


@respx.mock
def test_resolve_plain_string(client: QURLClient) -> None:
    """resolve() accepts a plain string token."""
    respx.post(f"{BASE_URL}/v1/resolve").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "target_url": "https://api.example.com/data",
                    "resource_id": "r_abc123def45",
                    "access_grant": {
                        "expires_in": 305,
                        "granted_at": "2026-03-10T15:30:00Z",
                        "src_ip": "203.0.113.42",
                    },
                },
            },
        )
    )

    result = client.resolve("at_k8xqp9h2sj9lx7r4a")
    assert result.target_url == "https://api.example.com/data"
    assert result.access_grant is not None
    assert result.access_grant.expires_in == 305
    assert result.access_grant.src_ip == "203.0.113.42"
    assert isinstance(result.access_grant.granted_at, datetime)


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


@respx.mock
def test_quota_typed(client: QURLClient) -> None:
    """get_quota() returns typed RateLimits and Usage objects."""
    respx.get(f"{BASE_URL}/v1/quota").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "plan": "growth",
                    "period_start": "2026-03-01T00:00:00Z",
                    "period_end": "2026-04-01T00:00:00Z",
                    "rate_limits": {
                        "create_per_minute": 60,
                        "create_per_hour": 1000,
                        "list_per_minute": 120,
                        "resolve_per_minute": 300,
                        "max_active_qurls": 5000,
                        "max_tokens_per_qurl": 10,
                    },
                    "usage": {
                        "qurls_created": 10,
                        "active_qurls": 5,
                        "active_qurls_percent": 0.1,
                        "total_accesses": 42,
                    },
                },
            },
        )
    )

    result = client.get_quota()
    assert result.plan == "growth"
    assert isinstance(result.period_start, datetime)

    # Typed RateLimits
    assert result.rate_limits is not None
    assert result.rate_limits.create_per_minute == 60
    assert result.rate_limits.max_active_qurls == 5000

    # Typed Usage
    assert result.usage is not None
    assert result.usage.active_qurls == 5
    assert result.usage.qurls_created == 10
    assert result.usage.total_accesses == 42


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


# --- Content-Type header tests ---


@respx.mock
def test_get_request_has_no_content_type(client: QURLClient) -> None:
    """GET requests should not send Content-Type header."""
    route = respx.get(f"{BASE_URL}/v1/qurls/r_abc").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "resource_id": "r_abc",
                    "target_url": "https://example.com",
                    "status": "active",
                    "created_at": "2026-03-10T10:00:00Z",
                    "tags": [],
                },
            },
        )
    )

    client.get("r_abc")
    req = route.calls[0].request
    assert "content-type" not in req.headers


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


# --- Async client ---


@respx.mock
@pytest.mark.asyncio
async def test_async_create(async_client: AsyncQURLClient) -> None:
    respx.post(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_async",
                    "qurl_link": "https://qurl.link/#at_async",
                    "qurl_site": "https://r_async.qurl.site",
                    "expires_at": "2026-03-15T10:00:00Z",
                    "qurl_id": "q_async",
                },
            },
        )
    )

    result = await async_client.create(target_url="https://example.com", expires_in="24h")
    assert result.resource_id == "r_async"
    assert isinstance(result.expires_at, datetime)


@respx.mock
@pytest.mark.asyncio
async def test_async_resolve(async_client: AsyncQURLClient) -> None:
    respx.post(f"{BASE_URL}/v1/resolve").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "target_url": "https://api.example.com/data",
                    "resource_id": "r_async",
                    "access_grant": {
                        "expires_in": 305,
                        "granted_at": "2026-03-10T15:30:00Z",
                        "src_ip": "203.0.113.42",
                    },
                },
            },
        )
    )

    result = await async_client.resolve("at_test_token")
    assert result.target_url == "https://api.example.com/data"
    assert result.access_grant is not None
    assert result.access_grant.expires_in == 305


@respx.mock
@pytest.mark.asyncio
async def test_async_list_all(async_client: AsyncQURLClient) -> None:
    route = respx.get(f"{BASE_URL}/v1/qurls")
    route.side_effect = [
        httpx.Response(
            200,
            json={
                "data": [_qurl_item("r_1", "https://1.com")],
                "meta": {"has_more": True, "next_cursor": "cur_abc"},
            },
        ),
        httpx.Response(
            200,
            json={
                "data": [_qurl_item("r_2", "https://2.com")],
                "meta": {"has_more": False},
            },
        ),
    ]

    all_qurls = [q async for q in async_client.list_all(status="active", page_size=1)]
    assert len(all_qurls) == 2
    assert [q.resource_id for q in all_qurls] == ["r_1", "r_2"]


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
        client.create(target_url="not-a-url")
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
        client.create(target_url="")


# --- extend() convenience method ---


@respx.mock
def test_extend(client: QURLClient) -> None:
    """extend() delegates to update(extend_by=...)."""
    route = respx.patch(f"{BASE_URL}/v1/qurls/r_abc123def45").mock(
        return_value=httpx.Response(
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
        )
    )

    result = client.extend("r_abc123def45", "7d")
    assert isinstance(result.expires_at, datetime)
    body = json.loads(route.calls[0].request.content)
    assert body == {"extend_by": "7d"}


@respx.mock
@pytest.mark.asyncio
async def test_async_extend(async_client: AsyncQURLClient) -> None:
    """Async extend() delegates to update(extend_by=...)."""
    route = respx.patch(f"{BASE_URL}/v1/qurls/r_abc").mock(
        return_value=httpx.Response(
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
        )
    )

    result = await async_client.extend("r_abc", "24h")
    assert result.resource_id == "r_abc"
    body = json.loads(route.calls[0].request.content)
    assert body == {"extend_by": "24h"}


# --- Batch create ---


@respx.mock
def test_batch_create_all_succeed(client: QURLClient) -> None:
    """batch_create() parses a fully successful batch response."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "succeeded": 2,
                    "failed": 0,
                    "results": [
                        {
                            "index": 0,
                            "success": True,
                            "resource_id": "r_batch1",
                            "qurl_link": "https://qurl.link/#at_b1",
                            "qurl_site": "https://r_batch1.qurl.site",
                            "expires_at": "2026-04-01T00:00:00Z",
                        },
                        {
                            "index": 1,
                            "success": True,
                            "resource_id": "r_batch2",
                            "qurl_link": "https://qurl.link/#at_b2",
                            "qurl_site": "https://r_batch2.qurl.site",
                        },
                    ],
                },
            },
        )
    )

    result = client.batch_create(
        [
            {"target_url": "https://a.com", "expires_in": "24h"},
            {"target_url": "https://b.com"},
        ]
    )
    assert result.succeeded == 2
    assert result.failed == 0
    assert len(result.results) == 2
    assert result.results[0].resource_id == "r_batch1"
    assert result.results[0].success is True
    assert isinstance(result.results[0].expires_at, datetime)
    assert result.results[1].resource_id == "r_batch2"
    assert result.results[1].expires_at is None


@respx.mock
def test_batch_create_partial_failure(client: QURLClient) -> None:
    """batch_create() correctly parses partial failures."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "succeeded": 1,
                    "failed": 1,
                    "results": [
                        {
                            "index": 0,
                            "success": True,
                            "resource_id": "r_ok",
                            "qurl_link": "https://qurl.link/#at_ok",
                            "qurl_site": "https://r_ok.qurl.site",
                        },
                        {
                            "index": 1,
                            "success": False,
                            "error": {
                                "code": "validation_error",
                                "message": "Invalid target_url",
                            },
                        },
                    ],
                },
            },
        )
    )

    result = client.batch_create(
        [
            {"target_url": "https://good.com"},
            {"target_url": "bad"},
        ]
    )
    assert result.succeeded == 1
    assert result.failed == 1
    assert result.results[0].success is True
    assert result.results[1].success is False
    assert result.results[1].error is not None
    assert result.results[1].error.code == "validation_error"
    assert result.results[1].error.message == "Invalid target_url"


# --- Date filter tests ---


@respx.mock
def test_list_with_date_filters(client: QURLClient) -> None:
    """list() passes date filter params to the request."""
    route = respx.get(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [_qurl_item("r_filtered", "https://filtered.com")],
                "meta": {"has_more": False},
            },
        )
    )

    result = client.list(
        status="active",
        created_after="2026-03-01T00:00:00Z",
        expires_before="2026-04-01T00:00:00Z",
    )
    assert len(result.qurls) == 1
    req = route.calls[0].request
    assert "created_after=2026-03-01T00%3A00%3A00Z" in str(req.url)
    assert "expires_before=2026-04-01T00%3A00%3A00Z" in str(req.url)


# --- Mint link full input ---


@respx.mock
def test_mint_link_full_input(client: QURLClient) -> None:
    """mint_link() sends all expanded params."""
    route = respx.post(f"{BASE_URL}/v1/qurls/r_abc/mint_link").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "qurl_link": "https://qurl.link/#at_full",
                    "expires_at": "2026-04-01T00:00:00Z",
                },
            },
        )
    )

    policy = AccessPolicy(ip_allowlist=["10.0.0.0/8"])
    result = client.mint_link(
        "r_abc",
        expires_in="12h",
        label="my-link",
        one_time_use=True,
        max_sessions=5,
        session_duration="30m",
        access_policy=policy,
    )
    assert result.qurl_link == "https://qurl.link/#at_full"
    body = json.loads(route.calls[0].request.content)
    assert body["expires_in"] == "12h"
    assert body["label"] == "my-link"
    assert body["one_time_use"] is True
    assert body["max_sessions"] == 5
    assert body["session_duration"] == "30m"
    assert body["access_policy"] == {"ip_allowlist": ["10.0.0.0/8"]}


@respx.mock
def test_mint_link_nested_ai_agent_policy(client: QURLClient) -> None:
    """mint_link() correctly serializes nested AIAgentPolicy inside AccessPolicy."""
    route = respx.post(f"{BASE_URL}/v1/qurls/r_abc/mint_link").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "qurl_link": "https://qurl.link/#at_ai",
                    "expires_at": "2026-04-01T00:00:00Z",
                },
            },
        )
    )

    policy = AccessPolicy(
        ip_allowlist=["10.0.0.0/8"],
        ai_agent_policy=AIAgentPolicy(
            block_all=True,
            deny_categories=["scraping"],
        ),
    )
    result = client.mint_link("r_abc", access_policy=policy)
    assert result.qurl_link == "https://qurl.link/#at_ai"
    body = json.loads(route.calls[0].request.content)
    assert body["access_policy"] == {
        "ip_allowlist": ["10.0.0.0/8"],
        "ai_agent_policy": {
            "block_all": True,
            "deny_categories": ["scraping"],
        },
    }


# --- Update with tags ---


@respx.mock
def test_update_with_tags(client: QURLClient) -> None:
    """update() sends tags in the request body."""
    route = respx.patch(f"{BASE_URL}/v1/qurls/r_abc").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "resource_id": "r_abc",
                    "target_url": "https://example.com",
                    "status": "active",
                    "created_at": "2026-03-10T10:00:00Z",
                    "tags": ["team:engineering", "env:prod"],
                },
            },
        )
    )

    result = client.update("r_abc", tags=["team:engineering", "env:prod"])
    assert result.tags == ["team:engineering", "env:prod"]
    body = json.loads(route.calls[0].request.content)
    assert body == {"tags": ["team:engineering", "env:prod"]}


# --- Create with label and session_duration ---


@respx.mock
def test_create_with_label_and_session_duration(client: QURLClient) -> None:
    """create() sends label and session_duration in the body."""
    route = respx.post(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_new",
                    "qurl_link": "https://qurl.link/#at_new",
                    "qurl_site": "https://r_new.qurl.site",
                    "qurl_id": "q_new",
                    "label": "my label",
                    "expires_at": "2026-04-01T00:00:00Z",
                },
            },
        )
    )

    result = client.create(
        target_url="https://example.com",
        expires_in="7d",
        label="my label",
        session_duration="1h",
    )
    assert result.resource_id == "r_new"
    assert result.label == "my label"
    assert result.qurl_id == "q_new"
    body = json.loads(route.calls[0].request.content)
    assert body["label"] == "my label"
    assert body["session_duration"] == "1h"


# --- Create with custom_domain ---


@respx.mock
def test_create_with_custom_domain(client: QURLClient) -> None:
    """create() sends custom_domain in the body."""
    route = respx.post(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_cd",
                    "qurl_link": "https://links.example.com/#at_cd",
                    "qurl_site": "https://r_cd.qurl.site",
                    "qurl_id": "q_cd",
                    "expires_at": "2026-04-01T00:00:00Z",
                },
            },
        )
    )

    result = client.create(
        target_url="https://example.com",
        expires_in="7d",
        custom_domain="links.example.com",
    )
    assert result.resource_id == "r_cd"
    body = json.loads(route.calls[0].request.content)
    assert body["custom_domain"] == "links.example.com"


# --- Async batch create ---


@respx.mock
@pytest.mark.asyncio
async def test_async_batch_create(async_client: AsyncQURLClient) -> None:
    """Async batch_create() works correctly."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "succeeded": 1,
                    "failed": 0,
                    "results": [
                        {
                            "index": 0,
                            "success": True,
                            "resource_id": "r_async_batch",
                            "qurl_link": "https://qurl.link/#at_ab",
                            "qurl_site": "https://r_async_batch.qurl.site",
                        },
                    ],
                },
            },
        )
    )

    result = await async_client.batch_create(
        [
            {"target_url": "https://async.com"},
        ]
    )
    assert result.succeeded == 1
    assert result.results[0].resource_id == "r_async_batch"


# --- Async date filters ---


@respx.mock
@pytest.mark.asyncio
async def test_async_list_with_date_filters(async_client: AsyncQURLClient) -> None:
    """Async list() passes date filter params."""
    route = respx.get(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [_qurl_item("r_af", "https://af.com")],
                "meta": {"has_more": False},
            },
        )
    )

    result = await async_client.list(
        created_before="2026-03-15T00:00:00Z",
        expires_after="2026-03-01T00:00:00Z",
    )
    assert len(result.qurls) == 1
    req = route.calls[0].request
    assert "created_before=2026-03-15T00%3A00%3A00Z" in str(req.url)
    assert "expires_after=2026-03-01T00%3A00%3A00Z" in str(req.url)


# --- Async mint link full input ---


@respx.mock
@pytest.mark.asyncio
async def test_async_mint_link_full_input(async_client: AsyncQURLClient) -> None:
    """Async mint_link() sends all expanded params."""
    route = respx.post(f"{BASE_URL}/v1/qurls/r_abc/mint_link").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "qurl_link": "https://qurl.link/#at_afull",
                    "expires_at": "2026-04-01T00:00:00Z",
                },
            },
        )
    )

    result = await async_client.mint_link(
        "r_abc",
        expires_in="6h",
        label="async-link",
        one_time_use=False,
        max_sessions=3,
        session_duration="15m",
    )
    assert result.qurl_link == "https://qurl.link/#at_afull"
    body = json.loads(route.calls[0].request.content)
    assert body["expires_in"] == "6h"
    assert body["label"] == "async-link"
    assert body["one_time_use"] is False
    assert body["max_sessions"] == 3
    assert body["session_duration"] == "15m"


# --- Async update with tags ---


@respx.mock
@pytest.mark.asyncio
async def test_async_update_with_tags(async_client: AsyncQURLClient) -> None:
    """Async update() sends tags."""
    route = respx.patch(f"{BASE_URL}/v1/qurls/r_abc").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "resource_id": "r_abc",
                    "target_url": "https://example.com",
                    "status": "active",
                    "created_at": "2026-03-10T10:00:00Z",
                    "tags": ["internal"],
                },
            },
        )
    )

    result = await async_client.update("r_abc", tags=["internal"])
    assert result.tags == ["internal"]
    body = json.loads(route.calls[0].request.content)
    assert body == {"tags": ["internal"]}
