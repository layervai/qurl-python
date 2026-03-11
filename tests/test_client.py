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
from layerv_qurl.types import AccessPolicy

BASE_URL = "https://api.test.layerv.ai"

_ERR_429 = {
    "error": {
        "status": 429, "code": "rate_limited",
        "title": "Rate Limited", "detail": "Slow down",
    },
}
_ERR_503 = {
    "error": {
        "status": 503, "code": "unavailable",
        "title": "Unavailable", "detail": "Down",
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
    respx.post(f"{BASE_URL}/v1/qurl").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_abc123def45",
                    "qurl_link": "https://qurl.link/#at_test",
                    "qurl_site": "https://r_abc123def45.qurl.site",
                    "expires_at": "2026-03-15T10:00:00Z",
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


@respx.mock
def test_create_sends_correct_body(client: QURLClient) -> None:
    route = respx.post(f"{BASE_URL}/v1/qurl").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_abc",
                    "qurl_link": "https://qurl.link/#at_test",
                    "qurl_site": "https://r_abc.qurl.site",
                },
            },
        )
    )

    client.create(
        target_url="https://example.com",
        expires_in="24h",
        description="test",
        one_time_use=True,
    )
    body = json.loads(route.calls[0].request.content)
    assert body == {
        "target_url": "https://example.com",
        "expires_in": "24h",
        "description": "test",
        "one_time_use": True,
    }


@respx.mock
def test_create_omits_none_values(client: QURLClient) -> None:
    route = respx.post(f"{BASE_URL}/v1/qurl").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_abc",
                    "qurl_link": "https://qurl.link/#at_test",
                    "qurl_site": "https://r_abc.qurl.site",
                },
            },
        )
    )

    client.create(target_url="https://example.com")
    body = json.loads(route.calls[0].request.content)
    assert body == {"target_url": "https://example.com"}
    assert "expires_in" not in body
    assert "description" not in body


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
                    "one_time_use": False,
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
        httpx.Response(200, json={
            "data": [_qurl_item("r_1", "https://1.com"), _qurl_item("r_2", "https://2.com")],
            "meta": {"has_more": True, "next_cursor": "cur_abc"},
        }),
        httpx.Response(200, json={
            "data": [_qurl_item("r_3", "https://3.com")],
            "meta": {"has_more": False},
        }),
    ]

    all_qurls = list(client.list_all(status="active", page_size=2))
    assert len(all_qurls) == 3
    assert [q.resource_id for q in all_qurls] == ["r_1", "r_2", "r_3"]
    assert route.call_count == 2


@respx.mock
def test_delete(client: QURLClient) -> None:
    respx.delete(f"{BASE_URL}/v1/qurls/r_abc123def45").mock(
        return_value=httpx.Response(204)
    )
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
            429, headers={"Retry-After": "5"}, json=_ERR_429,
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
            429, headers={"Retry-After": "120"}, json=_ERR_429,
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
    respx.get(f"{BASE_URL}/v1/quota").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    with pytest.raises(QURLNetworkError, match="Connection refused"):
        client.get_quota()


@respx.mock
def test_network_error_preserves_cause(client: QURLClient) -> None:
    """QURLNetworkError preserves the original httpx exception as __cause__."""
    respx.get(f"{BASE_URL}/v1/quota").mock(
        side_effect=httpx.ConnectError("DNS lookup failed")
    )

    with pytest.raises(QURLNetworkError) as exc_info:
        client.get_quota()

    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)


@respx.mock
def test_timeout_error_wrapped(client: QURLClient) -> None:
    """httpx.TimeoutException is wrapped in QURLTimeoutError."""
    respx.get(f"{BASE_URL}/v1/quota").mock(
        side_effect=httpx.ReadTimeout("Read timed out")
    )

    with pytest.raises(QURLTimeoutError, match="Read timed out"):
        client.get_quota()


@respx.mock
def test_timeout_error_is_network_error(client: QURLClient) -> None:
    """QURLTimeoutError is a subclass of QURLNetworkError."""
    respx.get(f"{BASE_URL}/v1/quota").mock(
        side_effect=httpx.ReadTimeout("Read timed out")
    )

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
    route = respx.post(f"{BASE_URL}/v1/qurl").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_abc",
                    "qurl_link": "https://qurl.link/#at_test",
                    "qurl_site": "https://r_abc.qurl.site",
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
    respx.post(f"{BASE_URL}/v1/qurl").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_async",
                    "qurl_link": "https://qurl.link/#at_async",
                    "qurl_site": "https://r_async.qurl.site",
                    "expires_at": "2026-03-15T10:00:00Z",
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
        httpx.Response(200, json={
            "data": [_qurl_item("r_1", "https://1.com")],
            "meta": {"has_more": True, "next_cursor": "cur_abc"},
        }),
        httpx.Response(200, json={
            "data": [_qurl_item("r_2", "https://2.com")],
            "meta": {"has_more": False},
        }),
    ]

    all_qurls = [q async for q in async_client.list_all(status="active", page_size=1)]
    assert len(all_qurls) == 2
    assert [q.resource_id for q in all_qurls] == ["r_1", "r_2"]


@respx.mock
@pytest.mark.asyncio
async def test_async_network_error_wrapped(async_client: AsyncQURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/quota").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    with pytest.raises(QURLNetworkError, match="Connection refused"):
        await async_client.get_quota()


@respx.mock
@pytest.mark.asyncio
async def test_async_timeout_error_wrapped(async_client: AsyncQURLClient) -> None:
    """Async: httpx.TimeoutException is wrapped in QURLTimeoutError."""
    respx.get(f"{BASE_URL}/v1/quota").mock(
        side_effect=httpx.ReadTimeout("Read timed out")
    )

    with pytest.raises(QURLTimeoutError, match="Read timed out"):
        await async_client.get_quota()


@respx.mock
@pytest.mark.asyncio
async def test_async_timeout_is_network_error(async_client: AsyncQURLClient) -> None:
    """Async: QURLTimeoutError is caught by except QURLNetworkError."""
    respx.get(f"{BASE_URL}/v1/quota").mock(
        side_effect=httpx.ReadTimeout("Read timed out")
    )

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
                    "status": 401, "code": "unauthorized",
                    "title": "Unauthorized", "detail": "Invalid API key",
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
                    "status": 403, "code": "forbidden",
                    "title": "Forbidden", "detail": "Insufficient scope",
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
                    "status": 404, "code": "not_found",
                    "title": "Not Found", "detail": "QURL not found",
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
    respx.post(f"{BASE_URL}/v1/qurl").mock(
        return_value=httpx.Response(
            422,
            json={
                "error": {
                    "status": 422, "code": "validation_error",
                    "title": "Validation Error", "detail": "Invalid target_url",
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
                    "status": 429, "code": "rate_limited",
                    "title": "Rate Limited", "detail": "Slow down",
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
                    "status": 500, "code": "internal",
                    "title": "Internal Server Error", "detail": "Something broke",
                },
            },
        )
    )

    with pytest.raises(ServerError) as exc_info:
        client.get_quota()
    assert exc_info.value.status == 500


@respx.mock
def test_400_raises_validation_error(client: QURLClient) -> None:
    respx.post(f"{BASE_URL}/v1/qurl").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "status": 400, "code": "bad_request",
                    "title": "Bad Request", "detail": "Missing target_url",
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
                },
            },
        )
    )

    result = await async_client.extend("r_abc", "24h")
    assert result.resource_id == "r_abc"
    body = json.loads(route.calls[0].request.content)
    assert body == {"extend_by": "24h"}


# --- AccessPolicy serialization ---


@respx.mock
def test_access_policy_serialized(client: QURLClient) -> None:
    """AccessPolicy dataclass is serialized correctly in create()."""
    route = respx.post(f"{BASE_URL}/v1/qurl").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_policy",
                    "qurl_link": "https://qurl.link/#at_test",
                    "qurl_site": "https://r_policy.qurl.site",
                },
            },
        )
    )

    policy = AccessPolicy(
        ip_allowlist=["10.0.0.0/8"],
        geo_denylist=["CN", "RU"],
    )
    client.create(target_url="https://example.com", access_policy=policy)
    body = json.loads(route.calls[0].request.content)
    assert body["access_policy"] == {
        "ip_allowlist": ["10.0.0.0/8"],
        "geo_denylist": ["CN", "RU"],
    }
    # None fields should be omitted from the serialized policy
    assert "ip_denylist" not in body["access_policy"]
    assert "geo_allowlist" not in body["access_policy"]


@respx.mock
def test_access_policy_in_update(client: QURLClient) -> None:
    """AccessPolicy can also be passed to update()."""
    route = respx.patch(f"{BASE_URL}/v1/qurls/r_abc").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "resource_id": "r_abc",
                    "target_url": "https://example.com",
                    "status": "active",
                    "created_at": "2026-03-10T10:00:00Z",
                    "access_policy": {
                        "user_agent_deny_regex": "curl.*",
                    },
                },
            },
        )
    )

    policy = AccessPolicy(user_agent_deny_regex="curl.*")
    result = client.update("r_abc", access_policy=policy)
    assert result.access_policy is not None
    assert result.access_policy.user_agent_deny_regex == "curl.*"
    body = json.loads(route.calls[0].request.content)
    assert body["access_policy"] == {"user_agent_deny_regex": "curl.*"}
