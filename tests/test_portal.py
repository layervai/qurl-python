"""Tests for the portal-verb surface (mirrors qurl-go's portal API)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

import httpx
import pytest
import respx

from layerv_qurl import (
    AsyncProtectedResource,
    AsyncQURLClient,
    Portal,
    ProtectedResource,
    QURLClient,
    QURLError,
    ResourceHandle,
)
from layerv_qurl.errors import AuthorizationError, NotFoundError, ValidationError

BASE_URL = "https://api.test.layerv.ai"

_RESOURCE_DATA: dict[str, Any] = {
    "resource_id": "r_abc123def45",
    "target_url": "https://internal.example.com/dashboard",
    "status": "active",
    "alias": "prod-dashboard",
    "tags": [],
    "created_at": "2026-03-10T10:00:00Z",
}

_PORTAL_DATA: dict[str, Any] = {
    "resource_id": "r_abc123def45",
    "qurl_link": "https://qurl.link/#at_portal1",
    "qurl_site": "https://r_abc123def45.qurl.site",
    "expires_at": "2026-03-10T10:05:00Z",
    "qurl_id": "q_portal1",
    "label": "Alice from Acme",
}

_RESOLVE_DATA: dict[str, Any] = {
    "target_url": "https://internal.example.com/dashboard",
    "resource_id": "r_abc123def45",
    "access_grant": {
        "expires_in": 305,
        "granted_at": "2026-03-10T15:30:00Z",
        "src_ip": "203.0.113.42",
    },
}


@pytest.fixture
def client() -> QURLClient:
    return QURLClient(api_key="lv_live_test", base_url=BASE_URL, max_retries=0)


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncQURLClient, None]:
    client = AsyncQURLClient(api_key="lv_live_test", base_url=BASE_URL, max_retries=0)
    yield client
    await client.close()


# --- protect_url ---


@respx.mock
def test_protect_url(client: QURLClient) -> None:
    """protect_url posts the resource fields and returns a bound handle."""
    route = respx.post(f"{BASE_URL}/v1/resources").mock(
        return_value=httpx.Response(201, json={"data": _RESOURCE_DATA})
    )

    resource = client.protect_url(
        "https://internal.example.com/dashboard",
        alias="prod-dashboard",
        description="Admin dashboard",
    )

    body = json.loads(route.calls[0].request.content)
    assert body == {
        "target_url": "https://internal.example.com/dashboard",
        "alias": "prod-dashboard",
        "description": "Admin dashboard",
    }
    # Mutating portal verbs carry an auto-generated idempotency key.
    assert route.calls[0].request.headers["idempotency-key"]
    assert isinstance(resource, ProtectedResource)
    assert resource.id == "r_abc123def45"
    assert resource.target_url == "https://internal.example.com/dashboard"
    assert resource.details is not None
    assert resource.details.status == "active"
    assert resource.details.alias == "prod-dashboard"
    assert repr(resource) == (
        "ProtectedResource(id='r_abc123def45', "
        "target_url='https://internal.example.com/dashboard')"
    )


def test_protect_url_rejects_non_http_target(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="target_url"):
        client.protect_url("ftp://internal.example.com")


def test_protect_url_rejects_embedded_credentials(client: QURLClient) -> None:
    """qurl-go parity: userinfo URLs fail client-side, without echoing them."""
    with pytest.raises(ValueError, match="embedded credentials") as exc_info:
        client.protect_url("https://alice:hunter2@internal.example.com/dashboard")
    assert "hunter2" not in str(exc_info.value)

    # Bare userinfo without a password is rejected too, like Go's u.User check.
    with pytest.raises(ValueError, match="embedded credentials"):
        client.protect_url("https://alice@internal.example.com/dashboard")


def test_protect_url_rejects_hostless_url(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="must include a host"):
        client.protect_url("https:///dashboard")


def test_create_portal_for_url_rejects_embedded_credentials(
    client: QURLClient,
) -> None:
    with pytest.raises(ValueError, match="embedded credentials"):
        client.create_portal_for_url("https://bob:secret@internal.example.com")


def test_protect_url_rejects_bad_alias(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="alias"):
        client.protect_url("https://internal.example.com", alias="Bad_Alias")


@respx.mock
def test_protect_url_keeps_caller_target_when_redacted(client: QURLClient) -> None:
    """A redacted response target_url falls back to the caller-supplied URL."""
    redacted = {**_RESOURCE_DATA}
    del redacted["target_url"]
    respx.post(f"{BASE_URL}/v1/resources").mock(
        return_value=httpx.Response(201, json={"data": redacted})
    )

    resource = client.protect_url("https://internal.example.com/dashboard")
    assert resource.target_url == "https://internal.example.com/dashboard"


# --- create_portal ---


@respx.mock
def test_create_portal_via_handle(client: QURLClient) -> None:
    """resource.create_portal mints against /v1/resources/{id}/qurls."""
    respx.post(f"{BASE_URL}/v1/resources").mock(
        return_value=httpx.Response(201, json={"data": _RESOURCE_DATA})
    )
    mint = respx.post(f"{BASE_URL}/v1/resources/r_abc123def45/qurls").mock(
        return_value=httpx.Response(201, json={"data": _PORTAL_DATA})
    )

    resource = client.protect_url("https://internal.example.com/dashboard")
    portal = resource.create_portal(
        valid_for=timedelta(minutes=5),
        label="Alice from Acme",
        one_time_use=True,
        max_sessions=0,
        session_duration=timedelta(minutes=30),
    )

    body = json.loads(mint.calls[0].request.content)
    assert body == {
        "expires_in": "5m",
        "label": "Alice from Acme",
        "one_time_use": True,
        # Explicit 0 means unlimited and must survive body construction.
        "max_sessions": 0,
        "session_duration": "30m",
    }
    assert isinstance(portal, Portal)
    assert portal.resource_id == "r_abc123def45"
    assert portal.link == "https://qurl.link/#at_portal1"
    assert portal.site == "https://r_abc123def45.qurl.site"
    assert portal.qurl_id == "q_portal1"
    assert portal.label == "Alice from Acme"
    assert isinstance(portal.expires_at, datetime)


@respx.mock
def test_create_portal_accepts_resource_id_string(client: QURLClient) -> None:
    mint = respx.post(f"{BASE_URL}/v1/resources/r_abc123def45/qurls").mock(
        return_value=httpx.Response(201, json={"data": _PORTAL_DATA})
    )

    portal = client.create_portal(
        "r_abc123def45", valid_for="45m", idempotency_key="mint-alice-1"
    )

    assert json.loads(mint.calls[0].request.content) == {"expires_in": "45m"}
    assert mint.calls[0].request.headers["idempotency-key"] == "mint-alice-1"
    assert portal.link == "https://qurl.link/#at_portal1"


@respx.mock
def test_create_portal_without_options_sends_empty_body(client: QURLClient) -> None:
    """No options → empty body, so the API default lifetime applies."""
    mint = respx.post(f"{BASE_URL}/v1/resources/r_abc123def45/qurls").mock(
        return_value=httpx.Response(201, json={"data": _PORTAL_DATA})
    )

    client.resource_by_id("r_abc123def45").create_portal()
    assert json.loads(mint.calls[0].request.content) == {}


@respx.mock
def test_create_portal_fails_closed_on_malformed_response(client: QURLClient) -> None:
    """A mint response without identity fields raises a typed error, not KeyError."""
    respx.post(f"{BASE_URL}/v1/resources/r_abc123def45/qurls").mock(
        return_value=httpx.Response(
            201, json={"data": {"resource_id": "r_abc123def45"}}
        )
    )

    with pytest.raises(ValidationError) as exc_info:
        client.create_portal("r_abc123def45")
    assert exc_info.value.code == "unexpected_response"


def test_create_portal_rejects_foreign_handle(client: QURLClient) -> None:
    other = QURLClient(api_key="lv_live_other", base_url=BASE_URL, max_retries=0)
    try:
        resource = other.resource_by_id("r_abc123def45")
        with pytest.raises(ValueError, match="bound to a different client"):
            client.create_portal(resource)
    finally:
        other.close()


def test_create_portal_rejects_wrong_handle_class(client: QURLClient) -> None:
    """A cross-class handle fails with a clean ValueError, not TypeError."""
    # Construct the async handle directly (its client binding is never
    # dereferenced — rejection happens on the isinstance check first), so
    # the test needs no live AsyncQURLClient to close.
    async_handle = AsyncProtectedResource(client, "r_abc123def45")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a ProtectedResource or a resource id"):
        client.create_portal(async_handle)  # type: ignore[arg-type]


def test_create_portal_rejects_non_handle_non_string(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="must be a ProtectedResource or a resource id"):
        client.create_portal(12345)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"valid_for": timedelta(seconds=30)}, "valid_for: must be at least 60s"),
        # Below-minimum wins over whole-seconds, matching qurl-go's order.
        (
            {"valid_for": timedelta(seconds=30, microseconds=500_000)},
            "valid_for: must be at least 60s",
        ),
        (
            {"valid_for": timedelta(seconds=61, microseconds=500_000)},
            "valid_for: must be whole seconds",
        ),
        ({"valid_for": ""}, "valid_for: must be a non-empty duration string"),
        ({"valid_for": 300}, "valid_for: must be a duration string"),
        (
            {"session_duration": timedelta(seconds=0)},
            "session_duration: must be at least 1s",
        ),
        ({"max_sessions": 1001}, "max_sessions"),
        ({"label": "x" * 501}, "label"),
        ({"label": ""}, "label: must not be empty"),
        ({"label": "   "}, "label: must not be empty"),
    ],
)
def test_create_portal_option_guardrails(
    client: QURLClient, options: dict[str, Any], match: str
) -> None:
    """Client-side guardrails reject bad options before any request."""
    resource = client.resource_by_id("r_abc123def45")
    with pytest.raises(ValueError, match=match):
        resource.create_portal(**options)


@respx.mock
@pytest.mark.parametrize(
    ("valid_for", "expected"),
    [
        (timedelta(hours=2), "2h"),
        (timedelta(minutes=5), "5m"),
        (timedelta(seconds=90), "90s"),
        (timedelta(days=1), "24h"),
        ("36h", "36h"),
    ],
)
def test_valid_for_duration_grammar(
    client: QURLClient, valid_for: str | timedelta, expected: str
) -> None:
    """timedeltas serialize with hours as the largest unit, like qurl-go."""
    mint = respx.post(f"{BASE_URL}/v1/resources/r_abc123def45/qurls").mock(
        return_value=httpx.Response(201, json={"data": _PORTAL_DATA})
    )

    client.create_portal("r_abc123def45", valid_for=valid_for)
    assert json.loads(mint.calls[0].request.content) == {"expires_in": expected}


# --- connector_resource ---


@respx.mock
def test_connector_resource(client: QURLClient) -> None:
    route = respx.get(f"{BASE_URL}/v1/resources", params={"slug": "prod-dashboard"}).mock(
        return_value=httpx.Response(200, json={"data": [_RESOURCE_DATA]})
    )

    resource = client.connector_resource("prod-dashboard")

    assert route.called
    assert resource.id == "r_abc123def45"
    assert resource.target_url == "https://internal.example.com/dashboard"
    assert resource.details is not None
    assert resource.details.alias == "prod-dashboard"


@respx.mock
def test_connector_resource_not_found(client: QURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/resources", params={"slug": "missing-conn"}).mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    with pytest.raises(NotFoundError) as exc_info:
        client.connector_resource("missing-conn")
    assert exc_info.value.status == 0
    assert exc_info.value.code == "resource_not_found"


@respx.mock
def test_connector_resource_ambiguous(client: QURLClient) -> None:
    second = {**_RESOURCE_DATA, "resource_id": "r_other9999999"}
    respx.get(f"{BASE_URL}/v1/resources", params={"slug": "prod-dashboard"}).mock(
        return_value=httpx.Response(200, json={"data": [_RESOURCE_DATA, second]})
    )

    with pytest.raises(QURLError) as exc_info:
        client.connector_resource("prod-dashboard")
    assert exc_info.value.code == "ambiguous_resource"


@respx.mock
def test_connector_resource_alias_mismatch(client: QURLClient) -> None:
    mismatched = {**_RESOURCE_DATA, "alias": "some-other-alias"}
    respx.get(f"{BASE_URL}/v1/resources", params={"slug": "prod-dashboard"}).mock(
        return_value=httpx.Response(200, json={"data": [mismatched]})
    )

    with pytest.raises(ValidationError) as exc_info:
        client.connector_resource("prod-dashboard")
    assert exc_info.value.code == "unexpected_response"


def test_connector_resource_requires_id(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="connector_id"):
        client.connector_resource("   ")


# --- create_portal_for_url ---


@respx.mock
def test_create_portal_for_url(client: QURLClient) -> None:
    """One-call flow posts to /v1/qurls and returns a reusable handle."""
    create = respx.post(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(201, json={"data": _PORTAL_DATA})
    )
    mint = respx.post(f"{BASE_URL}/v1/resources/r_abc123def45/qurls").mock(
        return_value=httpx.Response(201, json={"data": _PORTAL_DATA})
    )

    portal, resource = client.create_portal_for_url(
        "https://internal.example.com/dashboard", valid_for=timedelta(minutes=5)
    )

    assert json.loads(create.calls[0].request.content) == {
        "target_url": "https://internal.example.com/dashboard",
        "expires_in": "5m",
    }
    assert portal.link == "https://qurl.link/#at_portal1"
    assert resource.id == "r_abc123def45"
    assert resource.target_url == "https://internal.example.com/dashboard"
    # Only id + caller-supplied target URL are populated on this path.
    assert resource.details is None

    # The returned handle mints more portals without re-protecting.
    resource.create_portal(valid_for="1h")
    assert json.loads(mint.calls[0].request.content) == {"expires_in": "1h"}


# --- enter_portal ---


@respx.mock
def test_enter_portal_with_link(client: QURLClient) -> None:
    route = respx.post(f"{BASE_URL}/v1/resolve").mock(
        return_value=httpx.Response(200, json={"data": _RESOLVE_DATA})
    )

    handle = client.enter_portal(
        "https://qurl.link/#at_k8xqp9h2sj9lx7r4a", idempotency_key="open-once-1"
    )

    assert json.loads(route.calls[0].request.content) == {
        "access_token": "at_k8xqp9h2sj9lx7r4a"
    }
    assert route.calls[0].request.headers["idempotency-key"] == "open-once-1"
    assert isinstance(handle, ResourceHandle)
    assert handle.resource_url == "https://internal.example.com/dashboard"
    assert handle.open_seconds == 305
    assert handle.resource_id == "r_abc123def45"


@respx.mock
def test_enter_portal_with_bare_token(client: QURLClient) -> None:
    route = respx.post(f"{BASE_URL}/v1/resolve").mock(
        return_value=httpx.Response(200, json={"data": _RESOLVE_DATA})
    )

    handle = client.enter_portal("at_k8xqp9h2sj9lx7r4a")
    assert json.loads(route.calls[0].request.content) == {
        "access_token": "at_k8xqp9h2sj9lx7r4a"
    }
    assert handle.open_seconds == 305


def test_enter_portal_rejects_tokenless_link(client: QURLClient) -> None:
    """Unusable links raise without echoing the input (links are credentials)."""
    with pytest.raises(ValueError, match="no access token found"):
        client.enter_portal("https://qurl.link/")

    with pytest.raises(ValueError, match="no access token found") as exc_info:
        client.enter_portal("https://qurl.link/#at bad token")
    assert "bad token" not in str(exc_info.value)


def test_enter_portal_rejects_token_without_at_prefix(client: QURLClient) -> None:
    """A bare string that isn't an at_ token fails locally, not at the server."""
    with pytest.raises(ValueError, match="no access token found") as exc_info:
        client.enter_portal("hello")
    # Never echo the rejected input — it could be a mistyped credential.
    assert "hello" not in str(exc_info.value)


def test_enter_portal_rejects_signed_fragment_link(client: QURLClient) -> None:
    """qurl-go's offline signed links get a precise error, with no echo."""
    with pytest.raises(ValueError, match="signed qURL link") as exc_info:
        client.enter_portal("https://qurl.link/#v2.claimspart.secretpart.sigpart")
    assert "secretpart" not in str(exc_info.value)

    # A bare signed fragment (no link wrapper) is caught the same way.
    with pytest.raises(ValueError, match="signed qURL link"):
        client.enter_portal("v2.claimspart.secretpart.sigpart")


@respx.mock
def test_enter_portal_propagates_api_errors(client: QURLClient) -> None:
    """Scope/permission failures surface as the usual typed API errors."""
    respx.post(f"{BASE_URL}/v1/resolve").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": {
                    "status": 403,
                    "code": "insufficient_scope",
                    "title": "Forbidden",
                    "detail": "API key lacks qurl:resolve scope",
                },
            },
        )
    )

    with pytest.raises(AuthorizationError) as exc_info:
        client.enter_portal("at_k8xqp9h2sj9lx7r4a")
    assert exc_info.value.code == "insufficient_scope"


@respx.mock
def test_enter_portal_fails_closed_without_resource_url(client: QURLClient) -> None:
    respx.post(f"{BASE_URL}/v1/resolve").mock(
        return_value=httpx.Response(
            200, json={"data": {"resource_id": "r_abc123def45"}}
        )
    )

    with pytest.raises(ValidationError) as exc_info:
        client.enter_portal("at_k8xqp9h2sj9lx7r4a")
    assert exc_info.value.code == "unexpected_response"


@respx.mock
def test_enter_portal_without_grant_reports_zero_open_seconds(
    client: QURLClient,
) -> None:
    respx.post(f"{BASE_URL}/v1/resolve").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "target_url": "https://internal.example.com/dashboard",
                    "resource_id": "r_abc123def45",
                },
            },
        )
    )

    handle = client.enter_portal("at_k8xqp9h2sj9lx7r4a")
    assert handle.open_seconds == 0


# --- Async portal flow ---


@respx.mock
async def test_async_protect_url_and_create_portal(
    async_client: AsyncQURLClient,
) -> None:
    respx.post(f"{BASE_URL}/v1/resources").mock(
        return_value=httpx.Response(201, json={"data": _RESOURCE_DATA})
    )
    mint = respx.post(f"{BASE_URL}/v1/resources/r_abc123def45/qurls").mock(
        return_value=httpx.Response(201, json={"data": _PORTAL_DATA})
    )

    resource = await async_client.protect_url("https://internal.example.com/dashboard")
    assert isinstance(resource, AsyncProtectedResource)
    assert resource.id == "r_abc123def45"

    portal = await resource.create_portal(valid_for=timedelta(minutes=5))
    assert json.loads(mint.calls[0].request.content) == {"expires_in": "5m"}
    assert portal.link == "https://qurl.link/#at_portal1"


@respx.mock
async def test_async_enter_portal(async_client: AsyncQURLClient) -> None:
    respx.post(f"{BASE_URL}/v1/resolve").mock(
        return_value=httpx.Response(200, json={"data": _RESOLVE_DATA})
    )

    handle = await async_client.enter_portal("https://qurl.link/#at_k8xqp9h2sj9lx7r4a")
    assert handle.resource_url == "https://internal.example.com/dashboard"
    assert handle.open_seconds == 305


@respx.mock
async def test_async_create_portal_for_url(async_client: AsyncQURLClient) -> None:
    create = respx.post(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(201, json={"data": _PORTAL_DATA})
    )

    portal, resource = await async_client.create_portal_for_url(
        "https://internal.example.com/dashboard", valid_for="5m"
    )
    assert json.loads(create.calls[0].request.content) == {
        "target_url": "https://internal.example.com/dashboard",
        "expires_in": "5m",
    }
    assert portal.resource_id == "r_abc123def45"
    assert resource.id == "r_abc123def45"


@respx.mock
async def test_async_connector_resource_not_found(
    async_client: AsyncQURLClient,
) -> None:
    respx.get(f"{BASE_URL}/v1/resources", params={"slug": "missing-conn"}).mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    with pytest.raises(NotFoundError) as exc_info:
        await async_client.connector_resource("missing-conn")
    assert exc_info.value.code == "resource_not_found"


@respx.mock
async def test_async_connector_resource_alias_mismatch(
    async_client: AsyncQURLClient,
) -> None:
    mismatched = {**_RESOURCE_DATA, "alias": "some-other-alias"}
    respx.get(f"{BASE_URL}/v1/resources", params={"slug": "prod-dashboard"}).mock(
        return_value=httpx.Response(200, json={"data": [mismatched]})
    )

    with pytest.raises(ValidationError) as exc_info:
        await async_client.connector_resource("prod-dashboard")
    assert exc_info.value.code == "unexpected_response"


async def test_async_protect_url_rejects_embedded_credentials(
    async_client: AsyncQURLClient,
) -> None:
    with pytest.raises(ValueError, match="embedded credentials") as exc_info:
        await async_client.protect_url("https://alice:hunter2@internal.example.com")
    assert "hunter2" not in str(exc_info.value)


async def test_async_create_portal_rejects_foreign_handle(
    async_client: AsyncQURLClient,
) -> None:
    other = AsyncQURLClient(api_key="lv_live_other", base_url=BASE_URL, max_retries=0)
    try:
        resource = other.resource_by_id("r_abc123def45")
        with pytest.raises(ValueError, match="bound to a different client"):
            await async_client.create_portal(resource)
    finally:
        await other.close()


async def test_async_create_portal_rejects_wrong_handle_class(
    async_client: AsyncQURLClient,
) -> None:
    """A sync handle passed to the async client fails with a clean ValueError."""
    sync_handle = ProtectedResource(async_client, "r_abc123def45")  # type: ignore[arg-type]
    with pytest.raises(
        ValueError, match="must be an AsyncProtectedResource or a resource id"
    ):
        await async_client.create_portal(sync_handle)  # type: ignore[arg-type]
