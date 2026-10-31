"""qURL creation, updates, minting, expiration, and revocation tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import httpx
import pytest
import respx

from layerv_qurl.types import (
    AccessPolicy,
    AIAgentPolicy,
)
from tests.client_support import BASE_URL

if TYPE_CHECKING:
    from layerv_qurl import AsyncQURLClient, QURLClient

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
def test_update_serializes_datetime_expires_at(client: QURLClient) -> None:
    """Integration test: a ``datetime`` object passed as ``expires_at``
    is serialized to an ISO 8601 string in the wire body via the
    ``build_body → _serialize_value`` pipeline. The unit test
    ``test_serialize_value_datetime`` covers the function directly;
    this closes the loop through the full client method path.
    """
    route = respx.patch(f"{BASE_URL}/v1/qurls/r_abc").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "resource_id": "r_abc",
                    "target_url": "https://example.com",
                    "status": "active",
                    "created_at": "2026-03-10T10:00:00Z",
                    "expires_at": "2026-04-01T00:00:00+00:00",
                },
            },
        )
    )

    client.update(
        "r_abc",
        expires_at=datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc),
    )
    body = json.loads(route.calls[0].request.content)
    assert body["expires_at"] == "2026-04-01T00:00:00+00:00"


@respx.mock
def test_mint_link(client: QURLClient) -> None:
    respx.post(f"{BASE_URL}/v1/qurls/r_abc123def45/mint_link").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "qurl_link": "https://qurl.link/#at_newtoken",
                    "qurl_id": "",
                    "expires_at": "2026-03-20T10:00:00Z",
                },
            },
        )
    )

    result = client.mint_link("r_abc123def45", expires_at="2026-03-20T10:00:00Z")
    assert result.qurl_link == "https://qurl.link/#at_newtoken"
    assert result.qurl_id is None
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


async def test_async_update_rejects_access_policy_kwarg(
    async_client: AsyncQURLClient,
) -> None:
    """Async mirror of `test_update_rejects_access_policy_kwarg`.

    Locks in the immutability invariant on the async surface. The
    reasoning is identical: a future refactor adding `**kwargs` to
    `async_client.update()` would silently accept `access_policy`,
    breaking the spec-enforced contract. This guards against that
    independently of the function signature.
    """
    with pytest.raises(TypeError, match="access_policy"):
        await async_client.update(
            "r_abc",
            access_policy={"ai_agent_policy": "allow"},  # type: ignore[call-arg]
        )


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


@respx.mock
def test_create_with_minimal_ai_agent_policy_only(client: QURLClient) -> None:
    """Regression guard for the reviewer-noted gap: an AccessPolicy
    that contains ONLY ``ai_agent_policy`` (with every other policy
    field left None) must still serialize correctly. The existing
    test pairs ai_agent_policy with ip_allowlist, so the None-drop
    rule for the other policy fields wasn't exercised in isolation.

    This test verifies two things:
      1. `_serialize_value`'s dataclass None-drop works across ALL
         other `AccessPolicy` fields (ip_allowlist, ip_denylist,
         geo_allowlist, geo_denylist, user_agent_allow_regex,
         user_agent_deny_regex) — the serialized body contains ONLY
         `ai_agent_policy` under the `access_policy` key.
      2. `ai_agent_policy`'s own None fields (allow_categories,
         deny_categories) are also dropped, leaving ONLY `block_all`.
    """
    route = respx.post(f"{BASE_URL}/v1/qurls").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "resource_id": "r_minpol",
                    "qurl_link": "https://qurl.link/#at_minpol",
                    "qurl_site": "https://r_minpol.qurl.site",
                    "qurl_id": "q_minpol",
                },
            },
        )
    )

    # Only ai_agent_policy set; every other AccessPolicy field left None.
    # And within ai_agent_policy, only block_all is set.
    policy = AccessPolicy(ai_agent_policy=AIAgentPolicy(block_all=True))
    client.create(target_url="https://example.com", access_policy=policy)

    body = json.loads(route.calls[0].request.content)
    # The entire access_policy payload should be just the nested
    # ai_agent_policy, nothing else.
    assert body["access_policy"] == {"ai_agent_policy": {"block_all": True}}
    # Explicit checks that the other policy fields do NOT appear at all.
    assert "ip_allowlist" not in body["access_policy"]
    assert "ip_denylist" not in body["access_policy"]
    assert "geo_allowlist" not in body["access_policy"]
    assert "geo_denylist" not in body["access_policy"]
    assert "user_agent_allow_regex" not in body["access_policy"]
    assert "user_agent_deny_regex" not in body["access_policy"]
    # And None fields on AIAgentPolicy itself are dropped too.
    assert "allow_categories" not in body["access_policy"]["ai_agent_policy"]
    assert "deny_categories" not in body["access_policy"]["ai_agent_policy"]


# --- _parse_access_policy deserialization edge cases ---


def test_parse_access_policy_null_ai_agent_policy() -> None:
    """``ai_agent_policy: null`` / missing yields ``ai_agent_policy=None``.

    The happy-path test (test_create_with_access_policy) covers a
    populated AIAgentPolicy, but the None branch wasn't directly
    asserted. This locks in that null and missing are both normalised
    to ``None``, not to an empty ``AIAgentPolicy()``.
    """
    import layerv_qurl._utils as utils

    # Explicit null
    policy = utils._parse_access_policy({"ip_allowlist": ["10.0.0.0/8"], "ai_agent_policy": None})
    assert policy.ai_agent_policy is None
    assert policy.ip_allowlist == ["10.0.0.0/8"]

    # Key missing entirely
    policy2 = utils._parse_access_policy({"ip_denylist": ["192.168.0.0/16"]})
    assert policy2.ai_agent_policy is None
    assert policy2.ip_denylist == ["192.168.0.0/16"]


def test_parse_access_policy_non_dict_ai_agent_policy_is_ignored() -> None:
    """Non-dict ``ai_agent_policy`` (e.g. bare string or bool) is ignored.

    Without the ``isinstance(ap, dict)`` guard in ``_parse_access_policy``,
    ``.get("block_all")`` would raise ``AttributeError`` on a non-dict
    value. This locks in the defensive posture against unexpected API
    shapes — a string or boolean should be treated as "no policy"
    rather than crashing the entire response parser.
    """
    import layerv_qurl._utils as utils

    # Bare string
    policy = utils._parse_access_policy({"ai_agent_policy": "unexpected_string"})
    assert policy.ai_agent_policy is None

    # Boolean
    policy2 = utils._parse_access_policy({"ai_agent_policy": True})
    assert policy2.ai_agent_policy is None

    # Integer
    policy3 = utils._parse_access_policy({"ai_agent_policy": 42})
    assert policy3.ai_agent_policy is None


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


def test_update_rejects_access_policy_kwarg(client: QURLClient) -> None:
    """Lock in that `access_policy` is immutable on update().

    Per the OpenAPI spec's `UpdateQurlRequest` schema, access policy is
    set only at create time and cannot be modified on an existing
    resource. The signature itself currently enforces this (Python
    raises `TypeError` on unknown kwargs), but this test is an explicit
    invariant guard: a future refactor adding `**kwargs` to `update()`
    for forward-compat pass-through would silently accept
    `access_policy` again, breaking the spec-enforced immutability
    contract. Callers needing policy changes should create a new
    resource or mint a per-token override via `mint_link()`.
    """
    with pytest.raises(TypeError, match="access_policy"):
        client.update("r_abc", access_policy={"ai_agent_policy": "allow"})  # type: ignore[call-arg]


def test_create_rejects_expires_at_kwarg(client: QURLClient) -> None:
    """Lock in that ``expires_at`` was removed from ``create()``.

    Per the OpenAPI spec's ``CreateQurlRequest`` schema, creation accepts
    only ``expires_in`` (relative duration), not ``expires_at`` (absolute
    timestamp). The old ``expires_at`` parameter was removed in this PR.
    Like the ``access_policy`` invariant guard on ``update()``, this test
    is an explicit safety net: a future refactor adding ``**kwargs``
    would silently re-accept the removed parameter. Callers needing an
    absolute expiry should use ``create(expires_in=...) + update(expires_at=...)``.
    """
    with pytest.raises(TypeError, match="expires_at"):
        client.create(target_url="https://example.com", expires_at="2026-04-01T00:00:00Z")  # type: ignore[call-arg]


async def test_async_create_rejects_expires_at_kwarg(
    async_client: AsyncQURLClient,
) -> None:
    """Async mirror of ``test_create_rejects_expires_at_kwarg``."""
    with pytest.raises(TypeError, match="expires_at"):
        await async_client.create(  # type: ignore[call-arg]
            target_url="https://example.com",
            expires_at="2026-04-01T00:00:00Z",
        )


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


# ---- delete() token ID rejection ----------------------------------------


def test_delete_rejects_q_prefix_client_side(client: QURLClient) -> None:
    """Resource-wide deletion rejects token IDs."""
    with pytest.raises(ValueError, match="resource locator"):
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


# ---- Async delete q_ prefix rejection (reviewer gap #10) ------------------


@pytest.mark.asyncio
async def test_async_delete_rejects_q_prefix_client_side(
    async_client: AsyncQURLClient,
) -> None:
    """Sync test exists; async symmetry gap closed."""
    with pytest.raises(ValueError, match="resource locator"):
        await async_client.delete("q_3a7f2c8e91b")


@respx.mock
def test_delete_contract_wrappers(client: QURLClient) -> None:
    resource_route = respx.delete(f"{BASE_URL}/v1/resources/r_delete123").mock(
        return_value=httpx.Response(204)
    )
    webhook_route = respx.delete(f"{BASE_URL}/v1/webhooks/wh_abcdefghijklmnop").mock(
        return_value=httpx.Response(204)
    )
    api_key_route = respx.delete(f"{BASE_URL}/v1/api-keys/key_abc123def456").mock(
        return_value=httpx.Response(204)
    )

    client.delete_resource("r_delete123")
    client.delete_webhook("wh_abcdefghijklmnop")
    client.revoke_api_key("key_abc123def456")

    assert resource_route.called
    assert webhook_route.called
    assert api_key_route.called


@respx.mock
@pytest.mark.asyncio
async def test_async_update_customer_sends_idempotency(
    async_client: AsyncQURLClient,
) -> None:
    route = respx.patch(f"{BASE_URL}/v1/customer").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "tier": "growth",
                    "spending_cap_cents": 5000,
                    "current_period_usage": 7,
                    "frozen": False,
                },
            },
        )
    )

    customer = await async_client.update_customer(
        spending_cap_cents=5000,
        idempotency_key="idem-async-customer",
    )

    assert customer.current_period_usage_count == 7
    assert route.calls[0].request.headers["idempotency-key"] == "idem-async-customer"


@respx.mock
@pytest.mark.asyncio
async def test_async_delete_list_and_secret_contracts(
    async_client: AsyncQURLClient,
) -> None:
    access_codes_route = respx.get(f"{BASE_URL}/v1/access-codes").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "access_code_id": "acd_async123",
                        "resource_id": "r_asynccode",
                        "status": "active",
                    }
                ],
                "meta": {"has_more": False},
            },
        )
    )
    resource_route = respx.delete(f"{BASE_URL}/v1/resources/r_asyncdelete").mock(
        return_value=httpx.Response(204)
    )
    session_route = respx.delete(f"{BASE_URL}/v1/resources/r_asyncdelete/sessions/s_async123").mock(
        return_value=httpx.Response(204)
    )
    webhook_delete_route = respx.delete(f"{BASE_URL}/v1/webhooks/wh_asyncabcdefghijkl").mock(
        return_value=httpx.Response(204)
    )
    api_key_route = respx.delete(f"{BASE_URL}/v1/api-keys/key_async123456").mock(
        return_value=httpx.Response(204)
    )
    secret_route = respx.post(f"{BASE_URL}/v1/webhooks/wh_asyncabcdefghijkl/secret").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "webhook_id": "wh_asyncabcdefghijkl",
                    "url": "https://example.com/async-hook",
                    "events": ["qurl.created"],
                    "status": "active",
                    "secret": "whsec_async",
                },
            },
        )
    )

    access_codes = await async_client.list_access_codes()
    await async_client.delete_resource("r_asyncdelete")
    await async_client.terminate_resource_session("r_asyncdelete", "s_async123")
    await async_client.delete_webhook("wh_asyncabcdefghijkl")
    await async_client.revoke_api_key("key_async123456")
    webhook = await async_client.regenerate_webhook_secret(
        "wh_asyncabcdefghijkl",
        idempotency_key="idem-async-secret",
    )

    assert access_codes.access_codes[0].access_code_id == "acd_async123"
    assert access_codes.next_cursor is None
    assert access_codes.has_more is False
    assert not access_codes_route.calls[0].request.url.query
    assert webhook.secret == "whsec_async"
    assert secret_route.calls[0].request.headers["idempotency-key"] == "idem-async-secret"
    assert access_codes_route.called
    assert resource_route.called
    assert session_route.called
    assert webhook_delete_route.called
    assert api_key_route.called
