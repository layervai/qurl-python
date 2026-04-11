"""Tests for the QURL Python client."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import httpx
import pytest
import respx

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

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
from layerv_qurl.types import AccessPolicy, AccessToken, AIAgentPolicy, BatchCreateItem

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


def _qurl_item(rid: str, url: str) -> dict[str, Any]:
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
async def async_client() -> AsyncGenerator[AsyncQURLClient, None]:
    client = AsyncQURLClient(api_key="lv_live_test", base_url=BASE_URL, max_retries=0)
    yield client
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
    """create() serializes the full new-spec input shape into the request body.

    Covers the fields added in the v2 API alignment (label, one_time_use,
    max_sessions, session_duration) so each has at least one regression
    guard beyond the access_policy test.
    """
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
        label="Alice from Acme",
        one_time_use=True,
        max_sessions=5,
        session_duration="1h",
    )
    body = json.loads(route.calls[0].request.content)
    assert body == {
        "target_url": "https://example.com",
        "expires_in": "24h",
        "label": "Alice from Acme",
        "one_time_use": True,
        "max_sessions": 5,
        "session_duration": "1h",
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
def test_list_all_propagates_date_filters_to_every_page(client: QURLClient) -> None:
    """Regression guard: ``list_all`` delegates to ``list`` on each
    page fetch and must pass the date filter params through to the
    underlying HTTP request on EVERY iteration, not just the first.
    If a future refactor hoisted the filter params out of the loop
    body, pagination would silently drop the filter on page 2+.
    """
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

    created_after = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    expires_before = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
    all_qurls = list(
        client.list_all(
            page_size=1,
            created_after=created_after,
            expires_before=expires_before,
        )
    )
    assert len(all_qurls) == 2
    assert route.call_count == 2

    # Every HTTP call must carry the date filters in the query string.
    for call in route.calls:
        url = str(call.request.url)
        assert "created_after=2026-03-01T00%3A00%3A00" in url
        assert "expires_before=2026-04-01T00%3A00%3A00" in url


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
                        # Populated to exercise the parse path — earlier
                        # revisions of this test let `max_expiry_seconds`
                        # fall through the `.get(..., 0)` default, which
                        # meant the field wasn't actually tested.
                        "max_expiry_seconds": 604800,  # 7 days
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
    assert result.rate_limits.max_expiry_seconds == 604800

    # Typed Usage
    assert result.usage is not None
    assert result.usage.active_qurls == 5
    assert result.usage.qurls_created == 10
    assert result.usage.total_accesses == 42
    assert result.usage.active_qurls_percent == 0.1


@respx.mock
def test_quota_active_qurls_percent_null(client: QURLClient) -> None:
    """`active_qurls_percent` is nullable per the API spec — when the
    plan's `max_active_qurls` is unlimited, the field comes back as
    `null`. Lock in that the parser preserves `None` (rather than
    defaulting to `0.0` like the pre-alignment behavior did) so
    callers doing arithmetic on the field get a proper TypeError
    instead of silently treating unlimited as 0%.
    """
    respx.get(f"{BASE_URL}/v1/quota").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "plan": "enterprise",
                    "period_start": "2026-03-01T00:00:00Z",
                    "period_end": "2026-04-01T00:00:00Z",
                    "rate_limits": {
                        "create_per_minute": 1000,
                        "max_active_qurls": 0,  # unlimited on enterprise
                    },
                    "usage": {
                        "qurls_created": 500,
                        "active_qurls": 200,
                        # The load-bearing field under test: null when
                        # max_active_qurls is unlimited.
                        "active_qurls_percent": None,
                        "total_accesses": 12345,
                    },
                },
            },
        )
    )

    result = client.get_quota()
    assert result.usage is not None
    assert result.usage.active_qurls_percent is None
    # Sanity: the other nullable-adjacent fields still parse normally.
    assert result.usage.active_qurls == 200
    assert result.usage.total_accesses == 12345


@respx.mock
def test_quota_plan_missing_falls_back_to_unknown(client: QURLClient) -> None:
    """Regression guard for the `parse_quota` plan fallback being
    aligned with the `Quota.plan` dataclass default. A malformed API
    response that omits `plan` should produce `quota.plan == "unknown"`,
    not `""` — consistent with the dataclass default and the docstring.
    In practice the /v1/quota endpoint always returns a populated plan,
    so this exercises the defensive fallback path.
    """
    respx.get(f"{BASE_URL}/v1/quota").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    # Deliberately missing `plan`
                    "period_start": "2026-03-01T00:00:00Z",
                    "period_end": "2026-04-01T00:00:00Z",
                },
            },
        )
    )

    result = client.get_quota()
    assert result.plan == "unknown"


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


# --- POST retry safety ---
# POST is non-idempotent (create() might actually hit the DB even if the
# response is 5xx), so the client deliberately restricts POST retries to
# {429} only — retrying a 5xx on a create request risks duplicate records.
# These tests lock that decision in so a future refactor that naively
# unifies retry sets across methods will trip the guard.


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


@respx.mock
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
async def test_async_list_all_propagates_date_filters_to_every_page(
    async_client: AsyncQURLClient,
) -> None:
    """Async mirror of test_list_all_propagates_date_filters_to_every_page.
    Locks in sync/async parity for the list_all date-filter passthrough
    behavior — both clients must carry the filter params on every
    paginated HTTP call, not just the first.
    """
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

    created_after = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    expires_before = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
    all_qurls = [
        q
        async for q in async_client.list_all(
            page_size=1,
            created_after=created_after,
            expires_before=expires_before,
        )
    ]
    assert len(all_qurls) == 2
    assert route.call_count == 2

    for call in route.calls:
        url = str(call.request.url)
        assert "created_after=2026-03-01T00%3A00%3A00" in url
        assert "expires_before=2026-04-01T00%3A00%3A00" in url


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

    # Both URLs must pass client-side validation (syntactically valid
    # https://) — the mock returns a partial-failure payload regardless
    # of what's in the request body, so we're exercising the parser.
    result = client.batch_create(
        [
            {"target_url": "https://good.example.com"},
            {"target_url": "https://bad.example.com"},
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
                    "tags": ["team engineering", "env-prod"],
                },
            },
        )
    )

    # Tags must match the API regex ^[a-zA-Z0-9][a-zA-Z0-9 _-]*$ —
    # alphanumerics, spaces, underscores, and hyphens only.
    result = client.update("r_abc", tags=["team engineering", "env-prod"])
    assert result.tags == ["team engineering", "env-prod"]
    body = json.loads(route.calls[0].request.content)
    assert body == {"tags": ["team engineering", "env-prod"]}


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


# --- batch_create validation ---


def test_batch_create_empty_raises(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="requires at least 1 item"):
        client.batch_create([])


def test_batch_create_over_100_raises(client: QURLClient) -> None:
    items: list[BatchCreateItem] = [{"target_url": f"https://{i}.com"} for i in range(101)]
    with pytest.raises(ValueError, match="at most 100"):
        client.batch_create(items)


# --- create() with access_policy serialization ---


@respx.mock
def test_create_with_access_policy(client: QURLClient) -> None:
    """create() correctly serializes AccessPolicy including nested AIAgentPolicy."""
    route = respx.post(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_pol",
                    "qurl_link": "https://qurl.link/#at_pol",
                    "qurl_site": "https://r_pol.qurl.site",
                    "qurl_id": "q_pol",
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
    result = client.create(
        target_url="https://example.com",
        one_time_use=True,
        max_sessions=3,
        access_policy=policy,
    )
    assert result.resource_id == "r_pol"
    body = json.loads(route.calls[0].request.content)
    assert body["target_url"] == "https://example.com"
    assert body["one_time_use"] is True
    assert body["max_sessions"] == 3
    assert body["access_policy"] == {
        "ip_allowlist": ["10.0.0.0/8"],
        "ai_agent_policy": {
            "block_all": True,
            "deny_categories": ["scraping"],
        },
    }


# --- _serialize_value end-to-end with nested dataclasses ---


@respx.mock
def test_mint_link_nested_serialization_e2e(client: QURLClient) -> None:
    """Verifies _serialize_value recursively handles AccessPolicy > AIAgentPolicy."""
    route = respx.post(f"{BASE_URL}/v1/qurls/r_abc/mint_link").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "qurl_link": "https://qurl.link/#at_e2e",
                    "expires_at": "2026-04-01T00:00:00Z",
                },
            },
        )
    )

    policy = AccessPolicy(
        geo_denylist=["CN", "RU"],
        ai_agent_policy=AIAgentPolicy(
            allow_categories=["claude", "chatgpt"],
        ),
    )
    client.mint_link("r_abc", access_policy=policy, label="e2e-test")
    body = json.loads(route.calls[0].request.content)
    assert body["access_policy"]["geo_denylist"] == ["CN", "RU"]
    assert body["access_policy"]["ai_agent_policy"]["allow_categories"] == ["claude", "chatgpt"]
    assert "block_all" not in body["access_policy"]["ai_agent_policy"]  # None fields omitted


def test_serialize_value_none_asymmetry_dataclass_vs_dict() -> None:
    """Regression guard for the deliberate None-handling asymmetry in
    :func:`_serialize_value`:

    * Dataclass fields with ``None`` values are **dropped** — the
      dataclass distinguishes "unset" from "explicitly null."
    * ``None`` values inside nested dicts/lists are **preserved** —
      some API fields use explicit ``null`` as a signalling value
      (e.g. ``"ai_agent_policy": null`` to clear a policy).

    This test exercises both rules in a single call so a future
    refactor that unifies the behavior would trip immediately.
    """
    import layerv_qurl._utils as utils

    # AIAgentPolicy is a real dataclass from the types module. block_all
    # stays None — dataclass rule should drop it.
    policy = AIAgentPolicy(allow_categories=["claude"])

    value = {
        "explicit_null": None,  # dict None: must survive
        "dataclass": policy,  # dataclass with a None field: field must be dropped
        "nested": {
            "also_null": None,  # nested dict None: must survive
            "real": "value",
        },
        "list_with_nulls": [None, "x", None],  # list Nones: must survive
    }

    serialized = utils._serialize_value(value)

    # Dict-level None is preserved
    assert serialized["explicit_null"] is None
    assert serialized["nested"]["also_null"] is None
    # List-level None is preserved
    assert serialized["list_with_nulls"] == [None, "x", None]
    # Dataclass None field is dropped (block_all was None on the policy)
    assert "block_all" not in serialized["dataclass"]
    assert "deny_categories" not in serialized["dataclass"]  # also None
    # Non-None dataclass field survives
    assert serialized["dataclass"]["allow_categories"] == ["claude"]


# ---- Spec-derived input validation (create) --------------------------------


def test_create_rejects_target_url_longer_than_2048(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="target_url"):
        client.create(target_url="https://a.com/" + "x" * 2048)


def test_create_rejects_label_longer_than_500(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="label"):
        client.create(target_url="https://example.com", label="x" * 501)


def test_create_rejects_custom_domain_longer_than_253(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="custom_domain"):
        client.create(
            target_url="https://example.com",
            custom_domain="a" * 254,
        )


def test_create_rejects_max_sessions_above_1000(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="max_sessions"):
        client.create(target_url="https://example.com", max_sessions=1001)


def test_create_rejects_negative_max_sessions(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="max_sessions"):
        client.create(target_url="https://example.com", max_sessions=-1)


@respx.mock
def test_create_accepts_max_sessions_boundaries(client: QURLClient) -> None:
    """max_sessions=0 (unlimited) and max_sessions=1000 (hard limit) are both valid."""
    respx.post(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_x",
                    "qurl_link": "https://qurl.link/#x",
                    "qurl_site": "https://x.qurl.site",
                    "qurl_id": "q_x",
                },
            },
        )
    )
    client.create(target_url="https://example.com", max_sessions=0)
    client.create(target_url="https://example.com", max_sessions=1000)


# ---- Spec-derived input validation (update) --------------------------------


def test_update_rejects_description_longer_than_500(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="description"):
        client.update("r_abc", description="x" * 501)


def test_update_rejects_more_than_10_tags(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="tags"):
        client.update("r_abc", tags=[f"tag{i}" for i in range(11)])


def test_update_rejects_tags_longer_than_50_chars(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="1-50 characters"):
        client.update("r_abc", tags=["x" * 51])


def test_update_rejects_tags_that_dont_match_pattern(client: QURLClient) -> None:
    # Tags must start with an alphanumeric character.
    with pytest.raises(ValueError, match="alphanumeric"):
        client.update("r_abc", tags=["-leading-dash"])


@respx.mock
def test_update_accepts_empty_tags_to_clear(client: QURLClient) -> None:
    """Empty tag list clears all tags per the API spec."""
    respx.patch(f"{BASE_URL}/v1/qurls/r_abc").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "resource_id": "r_abc",
                    "target_url": "https://example.com",
                    "status": "active",
                    "tags": [],
                    "created_at": "2026-03-10T10:00:00Z",
                },
            },
        )
    )
    result = client.update("r_abc", tags=[])
    assert result.tags == []


@respx.mock
def test_update_wire_body_preserves_tags_empty_list(client: QURLClient) -> None:
    """Regression guard for the load-bearing `build_body` contract:
    `tags=[]` is a "clear all tags" API operation, not a "no change"
    signal. The empty list must survive into the serialized request
    body — `build_body` only strips top-level ``None``, never falsy
    values. A future refactor that adds a truthiness check would
    silently break tag-clearing, so this test asserts the wire body
    explicitly rather than just the parsed response.
    """
    route = respx.patch(f"{BASE_URL}/v1/qurls/r_abc").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "resource_id": "r_abc",
                    "target_url": "https://example.com",
                    "status": "active",
                    "tags": [],
                    "created_at": "2026-03-10T10:00:00Z",
                },
            },
        )
    )
    client.update("r_abc", tags=[])
    body = json.loads(route.calls[0].request.content)
    assert "tags" in body
    assert body["tags"] == []


@respx.mock
def test_update_wire_body_preserves_description_empty_string(
    client: QURLClient,
) -> None:
    """Same contract as tags=[] but for `description=""` — an empty
    string means "clear the description" per the API spec, and must
    survive `build_body`'s top-level-None-only strip. No previous
    regression test covered this path; this fills that gap.
    """
    route = respx.patch(f"{BASE_URL}/v1/qurls/r_abc").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "resource_id": "r_abc",
                    "target_url": "https://example.com",
                    "status": "active",
                    "description": "",
                    "created_at": "2026-03-10T10:00:00Z",
                },
            },
        )
    )
    client.update("r_abc", description="")
    body = json.loads(route.calls[0].request.content)
    assert "description" in body
    assert body["description"] == ""


def test_update_rejects_empty_input(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="at least one field"):
        client.update("r_abc")


def test_update_rejects_both_extend_by_and_expires_at(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.update("r_abc", extend_by="24h", expires_at="2026-04-01T00:00:00Z")


# ---- Spec-derived input validation (mint_link) -----------------------------


def test_mint_link_rejects_label_longer_than_500(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="label"):
        client.mint_link("r_abc", label="x" * 501)


def test_mint_link_rejects_max_sessions_above_1000(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="max_sessions"):
        client.mint_link("r_abc", max_sessions=5000)


def test_mint_link_rejects_both_expires_in_and_expires_at(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.mint_link(
            "r_abc",
            expires_in="7d",
            expires_at="2026-04-01T00:00:00Z",
        )


# ---- delete() r_ prefix enforcement ----------------------------------------


def test_delete_rejects_q_prefix_client_side(client: QURLClient) -> None:
    """DELETE /v1/qurls/:id only accepts r_ prefix per the API spec."""
    with pytest.raises(ValueError, match="r_ prefix"):
        client.delete("q_3a7f2c8e91b")


def test_delete_error_does_not_leak_full_resource_id(client: QURLClient) -> None:
    """Info-leak regression: the error message must echo only the 2-char
    prefix, not the raw caller-supplied ID. Error strings may end up in
    observability pipelines and the ID suffix could contain
    caller-sensitive data.
    """
    with pytest.raises(ValueError) as exc_info:
        client.delete("q_3a7f2c8e91b_sensitive_suffix")
    msg = str(exc_info.value)
    assert "'q_'" in msg
    # Must not echo any part of the caller-supplied ID beyond the prefix.
    assert "3a7f2c8e91b" not in msg
    assert "sensitive_suffix" not in msg


# ---- batch_create per-item validation & async validators -------------------


def test_batch_create_rejects_per_item_violation_with_index(client: QURLClient) -> None:
    """Per-item validation failures include the offending index."""
    with pytest.raises(ValueError, match=r"items\[1\].*max_sessions"):
        client.batch_create(
            [
                {"target_url": "https://a.example.com"},
                {"target_url": "https://b.example.com", "max_sessions": 9999},
            ]
        )


def test_batch_create_rejects_items_missing_target_url(client: QURLClient) -> None:
    with pytest.raises(ValueError, match=r"items\[0\].*target_url"):
        # Deliberately passing an item missing the required `target_url`
        # field to exercise the runtime validation. The type: ignore is
        # expected because BatchCreateItem enforces `target_url` at the
        # type level — this test guards against untyped callers
        # (e.g. Python scripts without strict type-checking) accidentally
        # sending incomplete items.
        client.batch_create([{"label": "no url"}])  # type: ignore[typeddict-item]


@pytest.mark.asyncio
async def test_async_batch_create_empty_raises(async_client: AsyncQURLClient) -> None:
    with pytest.raises(ValueError, match="requires at least 1 item"):
        await async_client.batch_create([])


@pytest.mark.asyncio
async def test_async_batch_create_over_100_raises(async_client: AsyncQURLClient) -> None:
    items: list[BatchCreateItem] = [{"target_url": f"https://{i}.com"} for i in range(101)]
    with pytest.raises(ValueError, match="at most 100"):
        await async_client.batch_create(items)


# ---- batch_create HTTP 400 passthrough -------------------------------------


@respx.mock
def test_batch_create_passes_through_400_with_per_item_errors(
    client: QURLClient,
) -> None:
    """HTTP 400 on batch/ carries a BatchCreateOutput body with per-item errors.

    The SDK whitelists 400 and surfaces the structured body instead of
    raising a generic ValidationError — matching the qurl-mcp and
    qurl-typescript behavior on this endpoint.
    """
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            400,
            json={
                "data": {
                    "succeeded": 0,
                    "failed": 2,
                    "results": [
                        {
                            "index": 0,
                            "success": False,
                            "error": {
                                "code": "validation_error",
                                "message": "items[0]: target_url must be HTTPS",
                            },
                        },
                        {
                            "index": 1,
                            "success": False,
                            "error": {
                                "code": "validation_error",
                                "message": "items[1]: target_url must be HTTPS",
                            },
                        },
                    ],
                },
                "meta": {"request_id": "req_allfail"},
            },
        )
    )
    result = client.batch_create(
        [
            {"target_url": "https://ok1.example.com"},
            {"target_url": "https://ok2.example.com"},
        ]
    )
    assert result.failed == 2
    assert result.succeeded == 0
    assert len(result.results) == 2
    assert result.results[0].success is False
    assert result.results[0].error is not None
    assert result.results[0].error.code == "validation_error"


@respx.mock
def test_batch_create_still_raises_on_401(client: QURLClient) -> None:
    """Non-400 error statuses still raise — the 400 passthrough is surgical."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            401,
            json={
                "error": {
                    "type": "https://api.qurl.link/problems/unauthorized",
                    "title": "Unauthorized",
                    "status": 401,
                    "code": "unauthorized",
                    "detail": "Invalid API key",
                },
            },
        )
    )
    with pytest.raises(AuthenticationError):
        client.batch_create([{"target_url": "https://example.com"}])


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


# ---- parse_create_output normalizes empty qurl_id --------------------------


@respx.mock
def test_create_normalizes_empty_qurl_id_to_none(client: QURLClient) -> None:
    """Empty-string qurl_id is normalized to None so `if result.qurl_id:` works."""
    respx.post(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_abc",
                    "qurl_link": "https://qurl.link/#at_x",
                    "qurl_site": "https://r_abc.qurl.site",
                    "qurl_id": "",
                },
            },
        )
    )
    result = client.create(target_url="https://example.com")
    assert result.qurl_id is None


# ---- Target URL scheme validation (create) --------------------------------


def test_create_rejects_target_url_without_scheme(client: QURLClient) -> None:
    """Client-side URL scheme check catches the common 'forgot http://' mistake."""
    with pytest.raises(ValueError, match="http:// or https://"):
        client.create(target_url="example.com")


def test_create_rejects_target_url_with_unsupported_scheme(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="http:// or https://"):
        client.create(target_url="ftp://files.example.com")


def test_create_accepts_http_and_https_schemes(client: QURLClient) -> None:
    """Both http:// and https:// pass the client-side check."""
    import layerv_qurl._utils as utils

    # Direct check — doesn't need a mocked HTTP response.
    utils.validate_create_input(target_url="http://example.com")
    utils.validate_create_input(target_url="https://example.com")


def test_create_rejects_non_string_target_url_with_value_error() -> None:
    """Non-string target_url inputs (None, int, bool, bytes) must raise
    ``ValueError`` — not a cryptic ``TypeError`` from slicing inside the
    error message. Regression guard: the previous ``target_url[:32]!r``
    form would raise ``TypeError`` on any non-subscriptable input before
    the ValueError could surface.
    """
    import layerv_qurl._utils as utils

    for bad in (None, 42, True, b"https://example.com"):
        with pytest.raises(ValueError, match="http:// or https://"):
            utils.validate_create_input(target_url=bad)  # type: ignore[arg-type]


# ---- _parse_access_policy deserialization (reviewer gap #8) ----------------


@respx.mock
def test_get_response_parses_nested_ai_agent_policy(client: QURLClient) -> None:
    """GET responses with a populated ai_agent_policy inside an access_policy
    on a token should deserialize cleanly. Serialization of this shape is
    covered elsewhere; this test closes the deserialization loop for the
    _parse_access_policy changes in this PR.
    """
    respx.get(f"{BASE_URL}/v1/qurls/r_abc123def45").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "resource_id": "r_abc123def45",
                    "target_url": "https://example.com",
                    "status": "active",
                    "created_at": "2026-03-10T10:00:00Z",
                    "qurl_count": 1,
                    "qurls": [
                        {
                            "qurl_id": "q_3a7f2c8e91b",
                            "status": "active",
                            "one_time_use": False,
                            "max_sessions": 5,
                            "session_duration": 3600,
                            "use_count": 0,
                            "created_at": "2026-03-10T10:00:00Z",
                            "expires_at": "2026-03-17T10:00:00Z",
                            "access_policy": {
                                "ip_allowlist": ["10.0.0.0/8"],
                                "geo_allowlist": ["US", "CA"],
                                "ai_agent_policy": {
                                    "block_all": False,
                                    "deny_categories": ["gptbot", "commoncrawl"],
                                    "allow_categories": ["claude", "perplexity"],
                                },
                            },
                        },
                    ],
                },
            },
        )
    )
    qurl = client.get("r_abc123def45")
    assert qurl.access_tokens is not None
    assert len(qurl.access_tokens) == 1
    token = qurl.access_tokens[0]
    assert token.access_policy is not None
    assert token.access_policy.ip_allowlist == ["10.0.0.0/8"]
    assert token.access_policy.geo_allowlist == ["US", "CA"]
    # The big one: ai_agent_policy nested inside access_policy must parse.
    assert token.access_policy.ai_agent_policy is not None
    ai_policy = token.access_policy.ai_agent_policy
    assert ai_policy.block_all is False
    assert ai_policy.deny_categories == ["gptbot", "commoncrawl"]
    assert ai_policy.allow_categories == ["claude", "perplexity"]


# ---- datetime list filter params (reviewer gap #9) ------------------------


@respx.mock
def test_list_serializes_datetime_filter_params_as_isoformat(
    client: QURLClient,
) -> None:
    """build_list_params handles datetime via .isoformat() — exercise that
    branch explicitly (existing tests only pass string timestamps)."""
    route = respx.get(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            200,
            json={"data": [], "meta": {"has_more": False}},
        )
    )
    cutoff = datetime(2026, 3, 1, 0, 0, 0)
    client.list(created_after=cutoff)

    called_url = str(route.calls[0].request.url)
    # urlencode-safe ISO 8601 — check the raw URL for the encoded value.
    assert "created_after=2026-03-01T00%3A00%3A00" in called_url


# ---- Async delete q_ prefix rejection (reviewer gap #10) ------------------


@pytest.mark.asyncio
async def test_async_delete_rejects_q_prefix_client_side(
    async_client: AsyncQURLClient,
) -> None:
    """Sync test exists; async symmetry gap closed."""
    with pytest.raises(ValueError, match="r_ prefix"):
        await async_client.delete("q_3a7f2c8e91b")


# ---- batch_create response shape guard (defense-in-depth) -----------------


@respx.mock
def test_batch_create_rejects_unexpected_400_body_shape(client: QURLClient) -> None:
    """The 400 passthrough trusts the BatchCreateOutput shape. If the API
    ever returns 400 with a different body (e.g. a plain error envelope
    or a proxy error), the SDK must raise a clear error instead of
    silently producing `(succeeded=0, failed=0, results=[])`. Defense
    in depth — matches qurl-typescript and qurl-mcp.
    """
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            400,
            json={"data": {"unexpected": "not a batch envelope"}},
        )
    )
    with pytest.raises(QURLError, match="Unexpected response shape"):
        client.batch_create([{"target_url": "https://example.com"}])


@respx.mock
def test_batch_create_rejects_400_body_with_non_boolean_success(
    client: QURLClient,
) -> None:
    """Per-entry shape guard: results[i].success must be a bool for the
    BatchItemResult discriminated-union contract to hold."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            400,
            json={
                "data": {
                    "succeeded": 0,
                    "failed": 1,
                    "results": [
                        {"index": 0, "success": "oops"},  # should be bool
                    ],
                },
            },
        )
    )
    with pytest.raises(QURLError, match="Unexpected response shape"):
        client.batch_create([{"target_url": "https://example.com"}])


@respx.mock
def test_batch_create_rejects_400_body_with_bool_counts(
    client: QURLClient,
) -> None:
    """`bool` is a subclass of `int` in Python, so a naive
    ``isinstance(..., int)`` check would silently accept
    ``"succeeded": True``. The shape guard explicitly rejects bool in
    the counts — this test locks that in against a future simplification
    that might drop the explicit bool check."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            400,
            json={
                "data": {
                    "succeeded": True,  # bool should be rejected
                    "failed": False,
                    "results": [],
                },
            },
        )
    )
    with pytest.raises(QURLError, match="Unexpected response shape"):
        client.batch_create([{"target_url": "https://example.com"}])


@respx.mock
def test_batch_create_rejects_counts_arithmetic_mismatch(
    client: QURLClient,
) -> None:
    """The shape guard asserts `succeeded + failed == len(results)`.
    A mismatch suggests a proxy or middleware mangled the response,
    or the API had a counting bug — either case warrants raising
    rather than trusting the data."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            400,
            json={
                "data": {
                    "succeeded": 5,  # claims 5 succeeded
                    "failed": 0,
                    "results": [
                        # …but only 1 entry
                        {"index": 0, "success": True, "resource_id": "r_only1"},
                    ],
                },
            },
        )
    )
    with pytest.raises(QURLError, match="Unexpected response shape"):
        client.batch_create([{"target_url": "https://example.com"}])


@respx.mock
def test_batch_create_accepts_access_policy_dataclass(
    client: QURLClient,
) -> None:
    """End-to-end coverage for passing an ``AccessPolicy`` dataclass
    (rather than a plain dict) in a batch item. The bridge between
    typed-caller convenience and the serialized request body is
    ``_serialize_value`` — this test locks in that the dataclass path
    produces the same nested JSON as a plain-dict caller would."""
    route = respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "succeeded": 1,
                    "failed": 0,
                    "results": [
                        {
                            "index": 0,
                            "success": True,
                            "resource_id": "r_dc_policy",
                            "qurl_link": "https://qurl.link/#at_dc",
                            "qurl_site": "https://r_dc_policy.qurl.site",
                        },
                    ],
                },
            },
        )
    )

    policy = AccessPolicy(
        geo_denylist=["CN", "RU"],
        ai_agent_policy=AIAgentPolicy(allow_categories=["claude", "chatgpt"]),
    )
    item: BatchCreateItem = {
        "target_url": "https://example.com",
        "label": "dc-test",
        "access_policy": policy,
    }
    client.batch_create([item])

    body = json.loads(route.calls[0].request.content)
    assert body["items"][0]["target_url"] == "https://example.com"
    assert body["items"][0]["access_policy"]["geo_denylist"] == ["CN", "RU"]
    assert body["items"][0]["access_policy"]["ai_agent_policy"]["allow_categories"] == [
        "claude",
        "chatgpt",
    ]
    # None fields on AIAgentPolicy (block_all, deny_categories) must be
    # dropped by _serialize_value's dataclass rule, not preserved.
    assert "block_all" not in body["items"][0]["access_policy"]["ai_agent_policy"]
    assert "deny_categories" not in body["items"][0]["access_policy"]["ai_agent_policy"]
