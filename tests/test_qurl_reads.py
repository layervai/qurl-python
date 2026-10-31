"""qURL reads, resolution, and cursor listing tests."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import httpx
import pytest
import respx

from layerv_qurl.errors import (
    ServerError,
)
from layerv_qurl.types import (
    AccessToken,
)
from tests.client_support import _ERR_503, BASE_URL, _qurl_item

if TYPE_CHECKING:
    from layerv_qurl import AsyncQURLClient, QURLClient


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
def test_resolve_plain_string(client: QURLClient) -> None:
    """resolve() accepts a plain string token."""
    route = respx.post(f"{BASE_URL}/v1/resolve").mock(
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

    result = client.resolve("at_k8xqp9h2sj9lx7r4a", idempotency_key="idem-resolve")
    assert result.target_url == "https://api.example.com/data"
    assert route.calls[0].request.headers["idempotency-key"] == "idem-resolve"
    assert result.access_grant is not None
    assert result.access_grant.expires_in == 305
    assert result.access_grant.src_ip == "203.0.113.42"
    assert isinstance(result.access_grant.granted_at, datetime)


@respx.mock
def test_resolve_stable_idempotency_key_can_cross_caller_retries(
    client: QURLClient,
) -> None:
    route = respx.post(f"{BASE_URL}/v1/resolve")
    route.side_effect = [
        httpx.Response(503, json=_ERR_503),
        httpx.Response(
            200,
            json={
                "data": {
                    "target_url": "https://api.example.com/data",
                    "resource_id": "r_abc123def45",
                }
            },
        ),
    ]

    with pytest.raises(ServerError):
        client.resolve("at_k8xqp9h2sj9lx7r4a", idempotency_key="idem-resolve-stable")
    result = client.resolve(
        "at_k8xqp9h2sj9lx7r4a",
        idempotency_key="idem-resolve-stable",
    )

    assert result.target_url == "https://api.example.com/data"
    assert route.call_count == 2
    assert route.calls[0].request.headers["idempotency-key"] == "idem-resolve-stable"
    assert route.calls[1].request.headers["idempotency-key"] == "idem-resolve-stable"


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
@pytest.mark.asyncio
async def test_async_resolve(async_client: AsyncQURLClient) -> None:
    route = respx.post(f"{BASE_URL}/v1/resolve").mock(
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
    assert len(route.calls[0].request.headers["idempotency-key"]) == 36
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


# ---- list() limit validation (OpenAPI spec: integer, 1-100) ------------------


def test_list_rejects_limit_zero(client: QURLClient) -> None:
    """Per OpenAPI (GET /v1/qurls → limit: minimum: 1, maximum: 100)."""
    with pytest.raises(ValueError, match="1 and 100"):
        client.list(limit=0)


def test_list_rejects_limit_above_100(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="1 and 100"):
        client.list(limit=101)


def test_list_rejects_negative_limit(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="1 and 100"):
        client.list(limit=-5)


def test_list_rejects_non_integer_limit(client: QURLClient) -> None:
    """Floats pass Python's ``int | None`` type annotation at runtime
    but violate the spec's ``type: integer``."""
    with pytest.raises(ValueError, match="integer"):
        client.list(limit=2.5)  # type: ignore[arg-type]


def test_list_rejects_bool_limit(client: QURLClient) -> None:
    """``bool`` is a subclass of ``int`` — must be explicitly rejected
    like ``_require_max_sessions_in_range``."""
    with pytest.raises(ValueError):
        client.list(limit=True)  # type: ignore[arg-type]


@respx.mock
def test_list_accepts_limit_at_boundaries(client: QURLClient) -> None:
    """Boundary regression: 1 and 100 are both valid."""
    respx.get(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            200,
            json={"data": [], "meta": {"has_more": False}},
        )
    )
    result_1 = client.list(limit=1)
    assert result_1.has_more is False
    result_100 = client.list(limit=100)
    assert result_100.has_more is False


@respx.mock
def test_list_omitted_limit_uses_server_default(client: QURLClient) -> None:
    """Omitting ``limit`` entirely must not trip the validator and must
    not produce a ``limit=`` query param."""
    route = respx.get(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            200,
            json={"data": [], "meta": {"has_more": False}},
        )
    )
    client.list()
    assert "limit" not in str(route.calls[0].request.url)


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


@respx.mock
@pytest.mark.parametrize("token", ["", "at_secret!", "at_secret/../token", "at_secret\n"])
def test_resolve_does_not_echo_invalid_token(client: QURLClient, token: str) -> None:
    with pytest.raises(ValueError, match="Invalid access_token") as exc:
        client.resolve(token)
    assert "at_secret" not in str(exc.value)


@respx.mock
@pytest.mark.parametrize("token", ["", "at_secret!", "at_secret/../token", "at_secret\n"])
async def test_async_resolve_does_not_echo_invalid_token(
    async_client: AsyncQURLClient, token: str
) -> None:
    with pytest.raises(ValueError, match="Invalid access_token") as exc:
        await async_client.resolve(token)
    assert "at_secret" not in str(exc.value)
