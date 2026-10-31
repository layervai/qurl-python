"""Resource, account, credential, domain, webhook, and billing API tests."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import httpx
import pytest
import respx

import layerv_qurl.types as qurl_types
from layerv_qurl import (
    AsyncQURLClient,
    QURLClient,
)
from layerv_qurl._utils import (
    domain_path_segment,
    parse_resource,
)
from layerv_qurl.errors import (
    ConflictError,
    GoneError,
)
from layerv_qurl.types import (
    APIKey,
)
from tests.client_support import BASE_URL


@respx.mock
def test_create_api_key_sends_kind_first_durable_body(client: QURLClient) -> None:
    route = respx.post(f"{BASE_URL}/v1/api-keys").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "key_id": "key_abc123def456",
                    "key_prefix": "lv_live_a3x9",
                    "api_key": "lv_live_secret",
                    "kind": "api_key",
                    "name": "Production",
                    "scopes": ["qurl:read", "qurl:write"],
                    "status": "active",
                },
            },
        )
    )
    key = client.create_api_key(name="Production", scopes=["qurl:read", "qurl:write"])

    body = json.loads(route.calls[0].request.content)
    assert body == {
        "kind": "api_key",
        "name": "Production",
        "scopes": ["qurl:read", "qurl:write"],
    }
    assert key.kind == "api_key"


@respx.mock
def test_create_api_key_mints_connector_enrollment_token(client: QURLClient) -> None:
    route = respx.post(f"{BASE_URL}/v1/api-keys").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "key_id": "key_enroll123456",
                    "key_prefix": "lv_live_enrl",
                    "api_key": "lv_live_enrollment_secret",
                    "kind": "enrollment_token",
                    "name": "prod-dashboard enrollment",
                    "target": "connector",
                    "claims": [{"type": "connector", "id": "prod-dashboard"}],
                    "scopes": ["qurl:agent", "qurl:write"],
                    "status": "active",
                },
            },
        )
    )
    key = client.create_api_key(
        name="prod-dashboard enrollment",
        kind="enrollment_token",
        target="connector",
        claims=[qurl_types.CredentialClaim(type="connector", id="prod-dashboard")],
        expires_in="2h",
    )

    body = json.loads(route.calls[0].request.content)
    assert body == {
        "kind": "enrollment_token",
        "name": "prod-dashboard enrollment",
        "target": "connector",
        "claims": [{"type": "connector", "id": "prod-dashboard"}],
        "expires_in": "2h",
    }
    # The server assigns enrollment-token scopes; sending them is a 400.
    assert "scopes" not in body
    assert key.target == "connector"
    assert key.claims == [qurl_types.CredentialClaim(type="connector", id="prod-dashboard")]


def test_create_api_key_rejects_mismatched_scopes(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="not accepted for kind='enrollment_token'"):
        client.create_api_key(
            name="bound enrollment",
            kind="enrollment_token",
            target="agent",
            scopes=["qurl:agent"],
        )

    with pytest.raises(ValueError, match="required for kind='api_key'"):
        client.create_api_key(name="durable")

    with pytest.raises(ValueError, match="target: only accepted for kind='enrollment_token'"):
        client.create_api_key(name="durable", scopes=["qurl:read"], target="connector")

    with pytest.raises(ValueError, match="claims: only accepted for kind='enrollment_token'"):
        client.create_api_key(
            name="durable",
            scopes=["qurl:read"],
            claims=[qurl_types.CredentialClaim(type="connector", id="prod-dashboard")],
        )


@respx.mock
@pytest.mark.asyncio
async def test_async_create_api_key_kind_first_parity(async_client: AsyncQURLClient) -> None:
    route = respx.post(f"{BASE_URL}/v1/api-keys").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "key_id": "key_async12345",
                    "key_prefix": "lv_live_asyn",
                    "api_key": "lv_live_async_enrollment",
                    "kind": "enrollment_token",
                    "name": "async enrollment",
                    "target": "agent",
                    "status": "active",
                },
            },
        )
    )
    key = await async_client.create_api_key(name="async enrollment", kind="enrollment_token")

    assert json.loads(route.calls[0].request.content) == {
        "kind": "enrollment_token",
        "name": "async enrollment",
    }
    assert key.kind == "enrollment_token"

    with pytest.raises(ValueError, match="not accepted for kind='enrollment_token'"):
        await async_client.create_api_key(
            name="async bad",
            kind="enrollment_token",
            scopes=["qurl:agent"],
        )


def test_api_key_preserves_positional_scopes_and_status() -> None:
    key = APIKey("key_example", "lv_live_test", "Example", ["qurl:read"], "active")

    assert key.scopes == ["qurl:read"]
    assert key.status == "active"
    assert key.kind is None


@respx.mock
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("legacy_limit", [False, True], ids=["absent-limit", "legacy-limit"])
async def test_quota_typed(
    client: QURLClient, async_client: AsyncQURLClient, use_async: bool, legacy_limit: bool
) -> None:
    """get_quota() preserves the quota contract without inventing a token cap."""
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
                        **({"max_tokens_per_qurl": 10} if legacy_limit else {}),
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

    result = await async_client.get_quota() if use_async else client.get_quota()
    assert result.plan == "growth"
    assert isinstance(result.period_start, datetime)

    # Typed RateLimits
    assert result.rate_limits is not None
    assert not hasattr(result.rate_limits, "max_tokens_per_qurl")
    assert result.rate_limits.create_per_minute == 60
    assert result.rate_limits.max_active_qurls == 5000
    assert result.rate_limits.max_expiry_seconds == 604800
    assert result.rate_limits.max_active_resources is None
    assert result.rate_limits.max_data_transfer_bytes is None

    # Typed Usage
    assert result.usage is not None
    assert result.usage.active_qurls == 5
    assert result.usage.qurls_created == 10
    assert result.usage.total_accesses == 42
    assert result.usage.active_qurls_percent == 0.1
    assert result.usage.active_resources is None
    assert result.usage.active_resources_percent is None
    assert result.usage.data_transfer_bytes is None


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("limits", "usage", "expected"),
    [
        pytest.param(
            {"max_active_resources": 10, "max_data_transfer_bytes": 10_737_418_240},
            {
                "active_resources": 8,
                "active_resources_percent": 80.0,
                "data_transfer_bytes": 23_456_789,
            },
            (10, 10_737_418_240, 8, 80.0, 23_456_789),
            id="canonical-only",
        ),
        pytest.param(
            {"max_active_resources": 0, "max_data_transfer_bytes": 0},
            {"active_resources": 0, "active_resources_percent": 0.0, "data_transfer_bytes": 0},
            (0, 0, 0, 0.0, 0),
            id="zero",
        ),
        pytest.param(
            {"max_active_resources": -1, "max_data_transfer_bytes": -1},
            {
                "active_resources": 87,
                "active_resources_percent": None,
                "data_transfer_bytes": 9_007_199_254_740_993,
            },
            (-1, -1, 87, None, 9_007_199_254_740_993),
            id="unlimited-and-large-bytes",
        ),
        pytest.param({}, {}, (None, None, None, None, None), id="absent-fields"),
        pytest.param(
            {"max_active_resources": None, "max_data_transfer_bytes": None},
            {
                "active_resources": None,
                "active_resources_percent": None,
                "data_transfer_bytes": None,
            },
            (None, None, None, None, None),
            id="null-fields",
        ),
    ],
)
@respx.mock
async def test_quota_preserves_canonical_resources_and_monthly_bytes(
    client: QURLClient,
    async_client: AsyncQURLClient,
    use_async: bool,
    limits: dict[str, Any],
    usage: dict[str, Any],
    expected: tuple[int | None, int | None, int | None, float | None, int | None],
) -> None:
    respx.get(f"{BASE_URL}/v1/quota").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "plan": "free",
                    "rate_limits": limits,
                    "usage": usage,
                }
            },
        )
    )

    result = await async_client.get_quota() if use_async else client.get_quota()
    assert result.rate_limits is not None
    assert result.usage is not None
    assert (
        result.rate_limits.max_active_resources,
        result.rate_limits.max_data_transfer_bytes,
        result.usage.active_resources,
        result.usage.active_resources_percent,
        result.usage.data_transfer_bytes,
    ) == expected
    # An omitted legacy alias must not invent a zero-resource account.
    assert result.rate_limits.max_active_qurls is None
    assert result.usage.active_qurls is None


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@respx.mock
async def test_quota_preserves_divergent_canonical_and_legacy_values(
    client: QURLClient, async_client: AsyncQURLClient, use_async: bool
) -> None:
    respx.get(f"{BASE_URL}/v1/quota").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "rate_limits": {"max_active_resources": 10, "max_active_qurls": 5000},
                    "usage": {
                        "active_resources": 8,
                        "active_qurls": 3,
                        "active_resources_percent": 80.0,
                        "active_qurls_percent": 0.06,
                    },
                }
            },
        )
    )

    result = await async_client.get_quota() if use_async else client.get_quota()
    assert result.rate_limits is not None
    assert result.rate_limits.max_active_resources == 10
    assert result.rate_limits.max_active_qurls == 5000
    assert result.usage is not None
    assert result.usage.active_resources == 8
    assert result.usage.active_qurls == 3
    assert result.usage.active_resources_percent == 80.0
    assert result.usage.active_qurls_percent == 0.06


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("wire_name", ["active_resources", "active_qurls"])
@pytest.mark.parametrize(
    ("fields", "expected_count", "expected_bytes"),
    [
        pytest.param({"count": 8, "data_transfer_bytes": 23_456_789}, 8, 23_456_789, id="observed"),
        pytest.param({"count": 0, "data_transfer_bytes": 0}, 0, 0, id="zero"),
        pytest.param(
            {"count": 87, "data_transfer_bytes": 9_007_199_254_740_993},
            87,
            9_007_199_254_740_993,
            id="large-bytes",
        ),
        pytest.param({}, None, None, id="absent"),
        pytest.param({"count": None, "data_transfer_bytes": None}, None, None, id="null"),
    ],
)
@respx.mock
async def test_current_period_preserves_resource_counts_and_monthly_bytes(
    client: QURLClient,
    async_client: AsyncQURLClient,
    use_async: bool,
    wire_name: str,
    fields: dict[str, int | None],
    expected_count: int | None,
    expected_bytes: int | None,
) -> None:
    payload = {k: v for k, v in fields.items() if k != "count"}
    if "count" in fields:
        payload[wire_name] = fields["count"]
    respx.get(f"{BASE_URL}/v1/usage/current-period").mock(
        return_value=httpx.Response(200, json={"data": payload})
    )

    result = (
        await async_client.get_usage_current_period()
        if use_async
        else client.get_usage_current_period()
    )
    assert result.active_resources == (expected_count if wire_name == "active_resources" else None)
    assert result.active_qurls == (expected_count if wire_name == "active_qurls" else None)
    assert result.data_transfer_bytes == expected_bytes


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
                        "max_active_qurls": -1,  # unlimited on enterprise
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


@respx.mock
def test_resource_create_update_and_detail_contract_methods(client: QURLClient) -> None:
    create_resource = respx.post(f"{BASE_URL}/v1/resources").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_tunnel12345",
                    "type": "tunnel",
                    "status": "active",
                    "slug": "prod-dashboard",
                    "knock_resource_id": "qurl-tunnel-server",
                    "preserve_host": False,
                    "created_at": "2026-03-10T10:00:00Z",
                },
            },
        )
    )
    resource = client.create_resource(
        resource_type="tunnel",
        slug="prod-dashboard",
        find_or_create=True,
        idempotency_key="idem-resource-create",
    )
    create_body = json.loads(create_resource.calls[0].request.content)
    assert create_body == {"type": "tunnel", "slug": "prod-dashboard", "find_or_create": True}
    assert create_resource.calls[0].request.headers["idempotency-key"] == "idem-resource-create"
    assert resource.resource_type == "tunnel"
    assert resource.knock_resource_id == "qurl-tunnel-server"

    update_resource = respx.patch(f"{BASE_URL}/v1/resources/r_tunnel12345").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "resource_id": "r_tunnel12345",
                    "type": "tunnel",
                    "status": "active",
                    "alias": None,
                    "tags": [],
                },
            },
        )
    )
    client.update_resource(
        "r_tunnel12345",
        alias=None,
        tags=[],
        preserve_host=False,
        idempotency_key="idem-resource-update",
    )
    update_body = json.loads(update_resource.calls[0].request.content)
    assert update_body == {"tags": [], "preserve_host": False, "alias": None}
    assert update_resource.calls[0].request.headers["idempotency-key"] == "idem-resource-update"

    client.update_resource("r_tunnel12345", alias="prod-dashboard")
    set_alias_body = json.loads(update_resource.calls[1].request.content)
    assert set_alias_body == {"alias": "prod-dashboard"}

    respx.get(f"{BASE_URL}/v1/resources/r_tunnel12345").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "resource": {
                        "resource_id": "r_tunnel12345",
                        "type": "tunnel",
                        "status": "active",
                        "qurl_count": 1,
                    },
                    "qurls": [
                        {
                            "qurl_id": "q_token12345",
                            "status": "active",
                            "session_duration": 3600,
                        }
                    ],
                },
            },
        )
    )
    detail = client.get_resource("r_tunnel12345")
    assert detail.resource.qurl_count == 1
    assert detail.qurls[0].session_duration == 3600


@respx.mock
def test_resource_qurl_token_contract_methods(client: QURLClient) -> None:
    mint_route = respx.post(f"{BASE_URL}/v1/resources/r_tunnel12345/qurls").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "qurl_id": "q_0123456789a",
                    "resource_id": "r_tunnel12345",
                    "qurl_link": "https://qurl.link/#at_new",
                    "qurl_site": "https://q_0123456789a.qurl.site",
                    "type": "tunnel",
                },
            },
        )
    )
    minted = client.create_qurl_for_resource(
        "r_tunnel12345",
        session_duration="1h",
        idempotency_key="idem-resource-mint",
    )
    assert minted.qurl_id == "q_0123456789a"
    assert mint_route.calls[0].request.headers["idempotency-key"] == "idem-resource-mint"

    update_token_route = respx.patch(
        f"{BASE_URL}/v1/resources/r_tunnel12345/qurls/q_0123456789a"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "qurl_id": "q_0123456789a",
                    "status": "active",
                    "label": "updated",
                    "max_sessions": 2,
                },
            },
        )
    )
    token = client.update_resource_qurl(
        "r_tunnel12345",
        "q_0123456789a",
        label="updated",
        idempotency_key="idem-token-update",
    )
    assert token.label == "updated"
    assert token.max_sessions == 2
    assert update_token_route.calls[0].request.headers["idempotency-key"] == "idem-token-update"


@respx.mock
def test_resource_session_contract_methods(client: QURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/resources/r_tunnel12345/sessions").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "session_id": "s_active123",
                        "qurl_id": "q_0123456789a",
                        "src_ip": "203.0.113.42",
                    }
                ],
            },
        )
    )
    sessions = client.list_resource_sessions("r_tunnel12345")
    assert sessions.sessions[0].src_ip == "203.0.113.42"

    respx.delete(f"{BASE_URL}/v1/resources/r_tunnel12345/sessions").mock(
        return_value=httpx.Response(200, json={"data": {"terminated": 3}})
    )
    assert client.terminate_all_resource_sessions("r_tunnel12345").terminated == 3

    session_route = respx.delete(
        f"{BASE_URL}/v1/resources/r_tunnel12345/sessions/s_active123"
    ).mock(return_value=httpx.Response(204))
    client.terminate_resource_session("r_tunnel12345", "s_active123")
    assert session_route.called


def test_resource_methods_validate_shared_metadata(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="create_resource"):
        client.create_resource()

    with pytest.raises(ValueError, match="target_url"):
        client.create_resource(target_url="ftp://example.com")

    with pytest.raises(ValueError, match="tags"):
        client.create_resource(target_url="https://example.com", tags=["bad/tag"])

    with pytest.raises(ValueError, match="custom_domain"):
        client.update_resource("r_tunnel12345", custom_domain="x" * 254)

    with pytest.raises(ValueError, match="domain"):
        client.get_domain("")

    with pytest.raises(ValueError, match="url"):
        client.create_webhook(url="ftp://example.com/webhook", events=["qurl.created"])

    with pytest.raises(ValueError, match="code"):
        client.redeem_access_code(code="")

    with pytest.raises(ValueError, match="elapsed_ms"):
        client.redeem_access_code(code="ac_k8xqp9h2sj9lx7r4abcdef", elapsed_ms=-1)

    with pytest.raises(ValueError, match="name"):
        client.create_api_key(name="", scopes=["qurl:read"])

    with pytest.raises(ValueError, match="plan"):
        client.create_billing_checkout(plan="")

    with pytest.raises(ValueError, match="spending_cap_cents"):
        client.update_customer(spending_cap_cents=-1)

    with pytest.raises(ValueError, match="alias"):
        client.create_resource(alias="ab")

    with pytest.raises(ValueError, match="create_resource"):
        client.create_resource(find_or_create=False)

    with pytest.raises(ValueError, match="alias"):
        client.update_resource("r_tunnel12345", alias="Bad_Alias")

    with pytest.raises(ValueError, match="reserved"):
        client.list_resources(alias="qurl")


@respx.mock
def test_resource_detail_tolerates_missing_resource_wrapper(client: QURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/resources/r_partial").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "qurls": [
                        {"qurl_id": "q_valid", "status": "active"},
                        {"qurl_id": "q_missing_status"},
                        "ignored",
                    ]
                }
            },
        )
    )
    detail = client.get_resource("r_partial")
    assert detail.resource.resource_id == ""
    assert detail.resource.status == "unknown"
    assert [qurl.qurl_id for qurl in detail.qurls] == ["q_valid"]


def test_parse_resource_requires_identity_fields() -> None:
    with pytest.raises(KeyError):
        parse_resource({"status": "active"})

    with pytest.raises(KeyError):
        parse_resource({"resource_id": "r_tunnel12345"})


def test_update_resource_methods_reject_empty_updates(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="at least one field"):
        client.update_resource("r_tunnel12345")

    with pytest.raises(ValueError, match="at least one field"):
        client.update_resource_qurl("r_tunnel12345", "q_0123456789a")

    with pytest.raises(ValueError, match="mutually exclusive"):
        client.update_resource_qurl(
            "r_tunnel12345",
            "q_0123456789a",
            extend_by="1h",
            expires_at="2026-04-01T00:00:00Z",
        )


def test_update_api_key_rejects_empty_update(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="at least one field"):
        client.update_api_key("key_abc123def456")


@respx.mock
def test_domain_path_segments_are_url_encoded(client: QURLClient) -> None:
    domain = "../evil%2Fhost.example.com"
    encoded_domain = "..%2Fevil%252Fhost.example.com"
    assert domain_path_segment(domain) == encoded_domain
    route = respx.delete(f"{BASE_URL}/v1/domains/{encoded_domain}").mock(
        return_value=httpx.Response(204)
    )

    client.delete_domain(domain)

    assert route.called


@respx.mock
def test_domain_webhook_and_error_contracts(client: QURLClient) -> None:
    respx.post(f"{BASE_URL}/v1/domains").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "domain": "secure.example.com",
                    "status": "pending_verification",
                    "verification_token": "tok_123",
                    "ready_for_qurls": False,
                    "dns_records": [
                        {
                            "type": "TXT",
                            "name": "_qurl.secure.example.com",
                            "value": "tok_123",
                            "verified": False,
                        }
                    ],
                },
            },
        )
    )
    domain = client.register_domain("secure.example.com")
    assert domain.dns_records[0].type == "TXT"

    verify_domain_route = respx.post(f"{BASE_URL}/v1/domains/secure.example.com/verify").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "domain": "secure.example.com",
                    "status": "verified",
                    "checks": {
                        "txt": {"verified": True},
                        "cname": {},
                    },
                },
            },
        )
    )
    verify = client.verify_domain("secure.example.com", idempotency_key="idem-domain-verify")
    assert verify.checks["txt"].verified is True
    assert verify.checks["cname"].verified is False
    assert verify_domain_route.calls[0].request.headers["idempotency-key"] == "idem-domain-verify"

    regenerate_domain_route = respx.post(
        f"{BASE_URL}/v1/domains/secure.example.com/regenerate-token"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "domain": "secure.example.com",
                    "status": "pending_verification",
                    "verification_token": "tok_regenerated",
                },
            },
        )
    )
    regenerated = client.regenerate_domain_token(
        "secure.example.com",
        idempotency_key="idem-domain-regenerate",
    )
    assert regenerated.verification_token == "tok_regenerated"
    assert (
        regenerate_domain_route.calls[0].request.headers["idempotency-key"]
        == "idem-domain-regenerate"
    )

    respx.post(f"{BASE_URL}/v1/webhooks").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "webhook_id": "wh_abcdefghijklmnop",
                    "url": "https://example.com/webhook",
                    "events": ["qurl.accessed"],
                    "status": "active",
                    "secret": "whsec_test",
                },
            },
        )
    )
    webhook = client.create_webhook(
        url="https://example.com/webhook",
        events=["qurl.accessed"],
    )
    assert webhook.secret == "whsec_test"

    respx.get(f"{BASE_URL}/v1/webhooks/wh_abcdefghijklmnop").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "webhook_id": "wh_abcdefghijklmnop",
                    "url": "https://example.com/webhook",
                    "events": ["qurl.accessed"],
                    "status": "active",
                },
            },
        )
    )
    assert client.get_webhook("wh_abcdefghijklmnop").status == "active"

    update_webhook_route = respx.patch(f"{BASE_URL}/v1/webhooks/wh_abcdefghijklmnop").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "webhook_id": "wh_abcdefghijklmnop",
                    "url": "https://example.com/webhook",
                    "events": ["qurl.created"],
                    "status": "active",
                },
            },
        )
    )
    client.update_webhook(
        "wh_abcdefghijklmnop",
        events=["qurl.created"],
        idempotency_key="idem-webhook-update",
    )
    assert update_webhook_route.calls[0].request.headers["idempotency-key"] == "idem-webhook-update"

    with pytest.raises(ValueError, match="at least one field"):
        client.update_webhook("wh_abcdefghijklmnop")

    secret_route = respx.post(f"{BASE_URL}/v1/webhooks/wh_abcdefghijklmnop/secret").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "webhook_id": "wh_abcdefghijklmnop",
                    "url": "https://example.com/webhook",
                    "events": ["qurl.created"],
                    "status": "active",
                    "secret": "whsec_rotated",
                },
            },
        )
    )
    rotated = client.regenerate_webhook_secret(
        "wh_abcdefghijklmnop",
        idempotency_key="idem-webhook-secret",
    )
    assert rotated.secret == "whsec_rotated"
    assert secret_route.calls[0].request.headers["idempotency-key"] == ("idem-webhook-secret")

    deliveries_route = respx.get(f"{BASE_URL}/v1/webhooks/wh_abcdefghijklmnop/deliveries").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "delivery_id": "wd_123",
                        "webhook_id": "wh_abcdefghijklmnop",
                        "event_type": "qurl.created",
                        "status": "delivered",
                    }
                ],
                "meta": {"next_cursor": "cur_delivery", "has_more": True},
            },
        )
    )
    deliveries = client.list_webhook_deliveries(
        "wh_abcdefghijklmnop",
        limit=5,
        cursor="cur_previous_delivery",
    )
    assert deliveries.deliveries[0].delivery_id == "wd_123"
    assert deliveries.next_cursor == "cur_delivery"
    assert deliveries_route.calls[0].request.url.params["limit"] == "5"
    assert deliveries_route.calls[0].request.url.params["cursor"] == "cur_previous_delivery"

    respx.get(f"{BASE_URL}/v1/webhooks/events").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "type": "domain.verified",
                        "category": "resource",
                        "description": "Custom domain verified",
                    },
                    {},
                    "ignored",
                ],
            },
        )
    )
    event_types = client.list_webhook_event_types().events
    assert [event.type for event in event_types] == ["domain.verified"]

    respx.get(f"{BASE_URL}/v1/qurls/r_conflict").mock(
        return_value=httpx.Response(
            409,
            json={"error": {"status": 409, "code": "conflict", "title": "Conflict"}},
        )
    )
    with pytest.raises(ConflictError):
        client.get("r_conflict")

    respx.get(f"{BASE_URL}/v1/qurls/r_gone").mock(
        return_value=httpx.Response(
            410,
            json={
                "error": {"status": 410, "code": "gone", "title": "Gone"},
                "meta": {"tombstone": {"tombstoned_at": "2026-01-01T00:00:00Z"}},
            },
        )
    )
    with pytest.raises(GoneError) as exc_info:
        client.get("r_gone")
    assert exc_info.value.meta is not None
    assert "tombstone" in exc_info.value.meta


@respx.mock
def test_api_key_and_access_code_contracts(client: QURLClient) -> None:
    key_route = respx.post(f"{BASE_URL}/v1/api-keys").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "key_id": "key_abc123def456",
                    "key_prefix": "lv_live_a3x9",
                    "api_key": "lv_live_secret",
                    "name": "Production",
                    "scopes": ["qurl:read", "qurl:write"],
                    "status": "active",
                },
            },
        )
    )
    key = client.create_api_key(
        name="Production",
        scopes=["qurl:read", "qurl:write"],
        idempotency_key="0192f7c4-3b8a-7e2f-9d01-4cf8a1b6e3d2",
    )
    assert key.api_key == "lv_live_secret"
    assert (
        key_route.calls[0].request.headers["idempotency-key"]
        == "0192f7c4-3b8a-7e2f-9d01-4cf8a1b6e3d2"
    )

    create_access_code_route = respx.post(f"{BASE_URL}/v1/access-codes").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "access_code_id": "acd_create123",
                    "resource_id": "r_code12345",
                    "name": "Partner access",
                    "status": "active",
                    "max_uses": 5,
                    "code": "ac_k8xqp9h2sj9lx7r4abcdef",
                },
            },
        )
    )
    access_code = client.create_access_code(
        resource_id="r_code12345",
        name="Partner access",
        max_uses=5,
        idempotency_key="idem-access-code-create",
    )
    create_access_code_body = json.loads(create_access_code_route.calls[0].request.content)
    assert access_code.code == "ac_k8xqp9h2sj9lx7r4abcdef"
    assert create_access_code_body == {
        "resource_id": "r_code12345",
        "name": "Partner access",
        "max_uses": 5,
    }
    assert (
        create_access_code_route.calls[0].request.headers["idempotency-key"]
        == "idem-access-code-create"
    )

    access_codes_route = respx.get(f"{BASE_URL}/v1/access-codes").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "access_code_id": "acd_list123",
                        "resource_id": "r_code12345",
                        "status": "active",
                    }
                ],
                "meta": {"next_cursor": "cur_codes", "has_more": True},
            },
        )
    )
    access_codes = client.list_access_codes(limit=10, cursor="cur_prev_codes")
    assert access_codes.access_codes[0].access_code_id == "acd_list123"
    assert access_codes.next_cursor == "cur_codes"
    assert access_codes.has_more is True
    assert access_codes_route.calls[0].request.url.params["limit"] == "10"
    assert access_codes_route.calls[0].request.url.params["cursor"] == "cur_prev_codes"


@respx.mock
def test_public_access_code_redeem_contracts() -> None:
    client = QURLClient(api_key="lv_live_test", base_url=BASE_URL, max_retries=0)
    no_auth_client = QURLClient(base_url=BASE_URL, max_retries=0)

    redeem_route = respx.post(f"{BASE_URL}/v1/access-codes/redeem").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"redirect_url": "https://qurl.link/#at_code"}},
        )
    )
    redeem = no_auth_client.redeem_access_code(
        code="ac_k8xqp9h2sj9lx7r4abcdef",
        elapsed_ms=5200,
        idempotency_key="idem-public-redeem",
    )
    assert redeem.redirect_url == "https://qurl.link/#at_code"
    assert "authorization" not in redeem_route.calls[0].request.headers
    assert redeem_route.calls[0].request.headers["idempotency-key"] == "idem-public-redeem"
    redeem_body = json.loads(redeem_route.calls[0].request.content)
    assert redeem_body == {
        "code": "ac_k8xqp9h2sj9lx7r4abcdef",
        "elapsed_ms": 5200,
    }

    no_auth_client.redeem_access_code(
        code="ac_k8xqp9h2sj9lx7r4abcdef",
        honeypot="bot",
    )
    assert len(redeem_route.calls[1].request.headers["idempotency-key"]) == 36
    explicit_honeypot_body = json.loads(redeem_route.calls[1].request.content)
    assert explicit_honeypot_body["honeypot"] == "bot"

    client.redeem_access_code(code="ac_k8xqp9h2sj9lx7r4abcdef")
    assert "authorization" not in redeem_route.calls[2].request.headers
    assert len(redeem_route.calls[2].request.headers["idempotency-key"]) == 36


@respx.mock
def test_usage_and_billing_contracts(client: QURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/usage/current-period").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "tier": "growth",
                    "period_start": "2026-03-01T00:00:00Z",
                    "period_end": "2026-03-31T23:59:59Z",
                    "qurls_created": 12,
                    "active_qurls": 3,
                    "cost_estimate": {
                        "currency": "usd",
                        "amount_cents": 120,
                        "description": "12 qURLs",
                    },
                },
            },
        )
    )
    assert client.get_usage_current_period().cost_estimate is not None

    respx.get(f"{BASE_URL}/v1/billing/invoices").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "invoices": [
                        {
                            "id": "in_123",
                            "amount_cents": 1500,
                            "status": "paid",
                            "created_at": "2026-02-01T00:00:00Z",
                            "pdf_url": None,
                        }
                    ]
                },
                "meta": {"has_more": False},
            },
        )
    )
    assert client.list_billing_invoices().invoices[0].id == "in_123"


@respx.mock
def test_account_parsers_tolerate_partial_usage_payloads(client: QURLClient) -> None:
    respx.get(f"{BASE_URL}/v1/usage/current-period").mock(
        return_value=httpx.Response(200, json={"data": {}})
    )
    usage = client.get_usage_current_period()
    assert usage.tier == "unknown"
    assert usage.qurls_created == 0
    assert usage.active_qurls is None
    assert usage.cost_estimate is None

    respx.get(f"{BASE_URL}/v1/usage/daily").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"daily": [{}, {"date": "2026-03-02"}]}},
        )
    )
    daily = client.get_usage_daily()
    assert daily.tier == "unknown"
    assert daily.daily[0].date == ""
    assert daily.daily[0].qurls_created == 0
    assert daily.daily[1].date == "2026-03-02"
    assert daily.daily[1].qurls_created == 0

    respx.get(f"{BASE_URL}/v1/customer").mock(
        return_value=httpx.Response(200, json={"data": {"frozen_reason": "manual"}})
    )
    customer = client.get_customer()
    assert customer.tier == "unknown"
    assert customer.current_period_usage_count == 0
    assert customer.frozen is False
    assert customer.frozen_reason == "manual"


@respx.mock
def test_customer_parser_tolerates_future_usage_object(client: QURLClient) -> None:
    route = respx.get(f"{BASE_URL}/v1/customer")
    route.side_effect = [
        httpx.Response(
            200,
            json={
                "data": {
                    "tier": "growth",
                    "current_period_usage": {"count": 9, "period": "current"},
                }
            },
        ),
        httpx.Response(
            200,
            json={"data": {"tier": "growth", "current_period_usage": {"count": True}}},
        ),
    ]
    customer = client.get_customer()
    assert customer.current_period_usage_count == 9
    customer_with_bool_count = client.get_customer()
    assert customer_with_bool_count.current_period_usage_count == 0


@respx.mock
def test_invoice_list_tolerates_non_dict_payload(
    client: QURLClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    respx.get(f"{BASE_URL}/v1/billing/invoices").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "in_unexpected_shape",
                        "amount_cents": 1,
                        "status": "paid",
                    }
                ]
            },
        )
    )
    with caplog.at_level(logging.DEBUG, logger="layerv_qurl"):
        invoices = client.list_billing_invoices()
    assert invoices.invoices == []
    assert any("parse_invoice_list_output" in record.message for record in caplog.records)


@respx.mock
def test_billing_session_methods_send_idempotency(client: QURLClient) -> None:
    checkout_route = respx.post(f"{BASE_URL}/v1/billing/checkout").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"url": "https://checkout.stripe.com/c/pay/cs_test"}},
        )
    )
    portal_route = respx.post(f"{BASE_URL}/v1/billing/portal").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"url": "https://billing.stripe.com/p/session/test"}},
        )
    )

    assert (
        client.create_billing_checkout(
            plan="growth",
            idempotency_key="idem-checkout",
        ).url
        == "https://checkout.stripe.com/c/pay/cs_test"
    )
    assert client.create_billing_portal(idempotency_key="idem-portal").url == (
        "https://billing.stripe.com/p/session/test"
    )
    assert checkout_route.calls[0].request.headers["idempotency-key"] == "idem-checkout"
    assert portal_route.calls[0].request.headers["idempotency-key"] == "idem-portal"


@respx.mock
@pytest.mark.asyncio
async def test_async_resource_scoped_mint_parity(async_client: AsyncQURLClient) -> None:
    route = respx.post(f"{BASE_URL}/v1/resources/r_async/qurls").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "qurl_id": "q_abcdef01234",
                    "resource_id": "r_async",
                    "qurl_link": "https://qurl.link/#at_async",
                    "qurl_site": "https://q_abcdef01234.qurl.site",
                    "branded_domain": "async.example.com",
                    "type": "url",
                },
            },
        )
    )

    result = await async_client.create_qurl_for_resource(
        "r_async",
        label="async",
        idempotency_key="idem-async-resource",
    )

    assert route.calls[0].request.headers["idempotency-key"] == "idem-async-resource"
    assert result.branded_domain == "async.example.com"


@respx.mock
@pytest.mark.asyncio
async def test_async_domain_and_list_limit_contracts(
    async_client: AsyncQURLClient,
) -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        await async_client.list_webhooks(limit=101)

    route = respx.get(f"{BASE_URL}/v1/domains").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "domain": "async.example.com",
                        "status": "verified",
                        "ready_for_qurls": True,
                    }
                ],
                "meta": {"next_cursor": "cur_next", "has_more": True},
            },
        )
    )

    domains = await async_client.list_domains(limit=10, cursor="cur_prev")

    assert domains.domains[0].domain == "async.example.com"
    assert domains.next_cursor == "cur_next"
    assert route.calls[0].request.url.params["limit"] == "10"
    assert route.calls[0].request.url.params["cursor"] == "cur_prev"


@respx.mock
@pytest.mark.asyncio
async def test_async_resource_connector_domain_contracts(
    async_client: AsyncQURLClient,
) -> None:
    resources_route = respx.get(f"{BASE_URL}/v1/resources").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "resource_id": "r_asyncresource",
                        "type": "url",
                        "status": "active",
                    }
                ],
                "meta": {"next_cursor": "cur_async_resources", "has_more": True},
            },
        )
    )
    connector_route = respx.get(f"{BASE_URL}/v1/connectors/installations").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"installation_id": "inst_async_missing_identity"},
                    {
                        "installation_id": "inst_async",
                        "plugin_id": "slack",
                        "label": "Async Slack",
                        "subject_kind": "slack_workspace",
                        "subject_display_name": "Async Engineering",
                        "status": "active",
                    },
                ],
                "meta": {"has_more": False},
            },
        )
    )
    respx.get(f"{BASE_URL}/v1/domains/async.example.com").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"domain": "async.example.com", "status": "verified"}},
        )
    )
    verify_route = respx.post(f"{BASE_URL}/v1/domains/async.example.com/verify").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "domain": "async.example.com",
                    "status": "verified",
                    "checks": {"txt": {"verified": True}},
                },
            },
        )
    )
    regen_route = respx.post(f"{BASE_URL}/v1/domains/async.example.com/regenerate-token").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "domain": "async.example.com",
                    "status": "pending_verification",
                    "verification_token": "tok_async",
                },
            },
        )
    )
    resources = await async_client.list_resources(limit=2, status="active")
    connectors = await async_client.list_connector_installations(limit=1)
    domain = await async_client.get_domain("async.example.com")
    verify = await async_client.verify_domain(
        "async.example.com",
        idempotency_key="idem-async-domain-verify",
    )
    regenerated = await async_client.regenerate_domain_token(
        "async.example.com",
        idempotency_key="idem-async-domain-token",
    )
    assert resources.resources[0].resource_id == "r_asyncresource"
    assert resources.next_cursor == "cur_async_resources"
    assert resources_route.calls[0].request.url.params["limit"] == "2"
    assert resources_route.calls[0].request.url.params["status"] == "active"
    assert connectors.installations[0].installation_id == "inst_async"
    assert len(connectors.installations) == 1
    assert connector_route.calls[0].request.url.params["limit"] == "1"
    assert domain.status == "verified"
    assert verify.checks["txt"].verified is True
    assert regenerated.verification_token == "tok_async"
    assert verify_route.calls[0].request.headers["idempotency-key"] == ("idem-async-domain-verify")
    assert regen_route.calls[0].request.headers["idempotency-key"] == ("idem-async-domain-token")


@respx.mock
@pytest.mark.asyncio
async def test_async_webhook_crud_and_delivery_contracts(
    async_client: AsyncQURLClient,
) -> None:
    list_route = respx.get(f"{BASE_URL}/v1/webhooks").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "webhook_id": "wh_asynccrud12345",
                        "url": "https://example.com/async-hook",
                        "events": ["qurl.created"],
                        "status": "active",
                    }
                ],
                "meta": {"next_cursor": "cur_async_hooks", "has_more": True},
            },
        )
    )
    create_route = respx.post(f"{BASE_URL}/v1/webhooks").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "webhook_id": "wh_asynccrud12345",
                    "url": "https://example.com/async-hook",
                    "events": ["qurl.created"],
                    "status": "active",
                    "secret": "whsec_async_create",
                },
            },
        )
    )
    get_route = respx.get(f"{BASE_URL}/v1/webhooks/wh_asynccrud12345").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "webhook_id": "wh_asynccrud12345",
                    "url": "https://example.com/async-hook",
                    "events": ["qurl.created"],
                    "status": "active",
                },
            },
        )
    )
    update_route = respx.patch(f"{BASE_URL}/v1/webhooks/wh_asynccrud12345").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "webhook_id": "wh_asynccrud12345",
                    "url": "https://example.com/async-hook",
                    "events": ["qurl.created", "qurl.accessed"],
                    "status": "paused",
                },
            },
        )
    )
    deliveries_route = respx.get(f"{BASE_URL}/v1/webhooks/wh_asynccrud12345/deliveries").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "delivery_id": "wd_async123",
                        "webhook_id": "wh_asynccrud12345",
                        "event_type": "qurl.created",
                        "status": "delivered",
                    }
                ],
                "meta": {"next_cursor": "cur_async_delivery", "has_more": False},
            },
        )
    )

    webhooks = await async_client.list_webhooks(
        limit=2,
        cursor="cur_async_hooks_prev",
        event="qurl.created",
    )
    created = await async_client.create_webhook(
        url="https://example.com/async-hook",
        events=["qurl.created"],
        idempotency_key="idem-async-webhook-create",
    )
    fetched = await async_client.get_webhook("wh_asynccrud12345")
    updated = await async_client.update_webhook(
        "wh_asynccrud12345",
        events=["qurl.created", "qurl.accessed"],
        status="paused",
        idempotency_key="idem-async-webhook-update",
    )
    deliveries = await async_client.list_webhook_deliveries(
        "wh_asynccrud12345",
        limit=1,
        cursor="cur_async_delivery_prev",
    )

    assert webhooks.webhooks[0].webhook_id == "wh_asynccrud12345"
    assert webhooks.next_cursor == "cur_async_hooks"
    assert list_route.calls[0].request.url.params["limit"] == "2"
    assert list_route.calls[0].request.url.params["cursor"] == "cur_async_hooks_prev"
    assert list_route.calls[0].request.url.params["event"] == "qurl.created"
    assert created.secret == "whsec_async_create"
    assert create_route.calls[0].request.headers["idempotency-key"] == ("idem-async-webhook-create")
    assert json.loads(create_route.calls[0].request.content)["events"] == ["qurl.created"]
    assert fetched.status == "active"
    assert get_route.called
    assert updated.status == "paused"
    assert update_route.calls[0].request.headers["idempotency-key"] == ("idem-async-webhook-update")
    assert json.loads(update_route.calls[0].request.content) == {
        "events": ["qurl.created", "qurl.accessed"],
        "status": "paused",
    }
    assert deliveries.deliveries[0].delivery_id == "wd_async123"
    assert deliveries.next_cursor == "cur_async_delivery"
    assert deliveries_route.calls[0].request.url.params["limit"] == "1"
    assert deliveries_route.calls[0].request.url.params["cursor"] == "cur_async_delivery_prev"


@respx.mock
@pytest.mark.asyncio
async def test_async_api_key_and_access_code_create_contracts(
    async_client: AsyncQURLClient,
) -> None:
    key_route = respx.post(f"{BASE_URL}/v1/api-keys").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "key_id": "key_asynccreate",
                    "key_prefix": "lv_live_async",
                    "api_key": "lv_live_async_secret",
                    "name": "Async Production",
                    "scopes": ["qurl:read", "qurl:write"],
                    "status": "active",
                },
            },
        )
    )
    list_keys_route = respx.get(f"{BASE_URL}/v1/api-keys").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "key_id": "key_asynccreate",
                        "key_prefix": "lv_live_async",
                        "name": "Async Production",
                        "scopes": ["qurl:read"],
                        "status": "active",
                    }
                ],
                "meta": {"next_cursor": "cur_async_keys", "has_more": True},
            },
        )
    )
    update_key_route = respx.patch(f"{BASE_URL}/v1/api-keys/key_asynccreate").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "key_id": "key_asynccreate",
                    "key_prefix": "lv_live_async",
                    "name": "Async CI",
                    "scopes": ["qurl:read"],
                    "status": "active",
                },
            },
        )
    )
    code_route = respx.post(f"{BASE_URL}/v1/access-codes").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "access_code_id": "acd_asynccreate",
                    "resource_id": "r_asyncaccess",
                    "name": "Async access",
                    "status": "active",
                    "code": "ac_async_code",
                },
            },
        )
    )

    key = await async_client.create_api_key(
        name="Async Production",
        scopes=["qurl:read", "qurl:write"],
        idempotency_key="0192f7c4-3b8a-7e2f-9d01-4cf8a1b6e3d2",
    )
    listed = await async_client.list_api_keys(
        limit=3,
        cursor="cur_async_keys_prev",
        status="active",
    )
    updated = await async_client.update_api_key(
        "key_asynccreate",
        name="Async CI",
        scopes=["qurl:read"],
        idempotency_key="idem-async-api-key-update",
    )
    code = await async_client.create_access_code(
        resource_id="r_asyncaccess",
        name="Async access",
        idempotency_key="idem-async-access-code-create",
    )

    assert key.api_key == "lv_live_async_secret"
    assert key_route.calls[0].request.headers["idempotency-key"] == (
        "0192f7c4-3b8a-7e2f-9d01-4cf8a1b6e3d2"
    )
    assert json.loads(key_route.calls[0].request.content)["scopes"] == [
        "qurl:read",
        "qurl:write",
    ]
    assert listed.api_keys[0].key_id == "key_asynccreate"
    assert listed.next_cursor == "cur_async_keys"
    assert list_keys_route.calls[0].request.url.params["limit"] == "3"
    assert list_keys_route.calls[0].request.url.params["cursor"] == ("cur_async_keys_prev")
    assert list_keys_route.calls[0].request.url.params["status"] == "active"
    assert updated.name == "Async CI"
    assert update_key_route.calls[0].request.headers["idempotency-key"] == (
        "idem-async-api-key-update"
    )
    assert code.code == "ac_async_code"
    assert code_route.calls[0].request.headers["idempotency-key"] == (
        "idem-async-access-code-create"
    )
    assert json.loads(code_route.calls[0].request.content) == {
        "resource_id": "r_asyncaccess",
        "name": "Async access",
    }


@respx.mock
@pytest.mark.asyncio
async def test_async_usage_billing_and_resource_qurl_validation_contracts(
    async_client: AsyncQURLClient,
) -> None:
    respx.get(f"{BASE_URL}/v1/usage/current-period").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "tier": "growth",
                    "period_start": "2026-03-01T00:00:00Z",
                    "period_end": "2026-03-31T23:59:59Z",
                    "qurls_created": 12,
                    "active_qurls": 4,
                },
            },
        )
    )
    respx.get(f"{BASE_URL}/v1/usage/daily").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "tier": "growth",
                    "period_start": "2026-03-01T00:00:00Z",
                    "period_end": "2026-03-31T23:59:59Z",
                    "daily": [{"date": "2026-03-01", "qurls_created": 2}],
                },
            },
        )
    )
    respx.get(f"{BASE_URL}/v1/customer").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "tier": "growth",
                    "spending_cap_cents": 10000,
                    "current_period_usage": {"count": 9},
                    "frozen": False,
                },
            },
        )
    )
    checkout_route = respx.post(f"{BASE_URL}/v1/billing/checkout").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"url": "https://billing.example.com/checkout"}},
        )
    )
    portal_route = respx.post(f"{BASE_URL}/v1/billing/portal").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"url": "https://billing.example.com/portal"}},
        )
    )
    invoices_route = respx.get(f"{BASE_URL}/v1/billing/invoices").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "invoices": [
                        {
                            "id": "in_async",
                            "amount_cents": 1500,
                            "status": "paid",
                            "created_at": "2026-02-01T00:00:00Z",
                        }
                    ]
                },
                "meta": {"next_cursor": "cur_async_invoice", "has_more": True},
            },
        )
    )

    usage = await async_client.get_usage_current_period()
    daily = await async_client.get_usage_daily()
    customer = await async_client.get_customer()
    checkout = await async_client.create_billing_checkout(
        plan="growth",
        idempotency_key="idem-async-checkout",
    )
    portal = await async_client.create_billing_portal(idempotency_key="idem-async-portal")
    invoices = await async_client.list_billing_invoices(
        limit=1,
        cursor="cur_async_invoice_prev",
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        await async_client.update_resource_qurl(
            "r_asyncresource",
            "q_abcdef01234",
            extend_by="7d",
            expires_at="2026-04-01T00:00:00Z",
        )

    assert usage.qurls_created == 12
    assert daily.daily[0].qurls_created == 2
    assert customer.current_period_usage_count == 9
    assert checkout.url == "https://billing.example.com/checkout"
    assert checkout_route.calls[0].request.headers["idempotency-key"] == ("idem-async-checkout")
    assert portal.url == "https://billing.example.com/portal"
    assert portal_route.calls[0].request.headers["idempotency-key"] == ("idem-async-portal")
    assert invoices.invoices[0].id == "in_async"
    assert invoices.next_cursor == "cur_async_invoice"
    assert invoices_route.calls[0].request.url.params["limit"] == "1"
    assert invoices_route.calls[0].request.url.params["cursor"] == ("cur_async_invoice_prev")
