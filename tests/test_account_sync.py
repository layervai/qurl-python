"""Account-sync documents: saved access rules, inbox state, received history, link defaults."""

import inspect
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from layerv_qurl import (
    AccessRule,
    AccessRuleLinkLimits,
    AccessRulePayment,
    AsyncQURLClient,
    ConflictError,
    LinkDefaults,
    QURLClient,
    QURLError,
    ReceivedLink,
)
from tests.client_support import BASE_URL

pytestmark = pytest.mark.asyncio

CRID = "a" * 60
RULE_ID = "0192f0c4-7d1a-7c3e-8b2a-1f2e3d4c5b6a"
PUBLIC_RULE = {
    "id": RULE_ID,
    "name": "Quick look",
    "mode": "public",
    "link_limits": {"expires_in": "1h", "one_time_use": False},
}
PAID_RULE = {
    "id": "0192f0c4-7d1a-7c3e-8b2a-1f2e3d4c5b6b",
    "name": "People who pay",
    "mode": "contract",
    "contract_id": "ref-1",
    "payment": {"price_id": "price_1PqA2b3C4d5E6f7G", "access_seconds": 0},
}
LIBRARY = {
    "account_id": "acct_1",
    "version": 4,
    "items": [PUBLIC_RULE, PAID_RULE],
    "built_ins": [{"id": "review", "name": "Approve each person", "mode": "review"}],
    "quick_share_rule_id": RULE_ID,
}


@pytest.fixture(params=[QURLClient, AsyncQURLClient], ids=["sync", "async"])
async def client(request):
    with (
        patch("layerv_qurl.client.time.sleep"),
        patch("layerv_qurl.async_client.asyncio.sleep", new=AsyncMock()),
    ):
        instance = request.param(api_key="lv_test", base_url=BASE_URL, max_retries=2)
        yield instance
        result = instance.close()
        if inspect.isawaitable(result):
            await result


async def invoke(method, *args, **kwargs):
    result = method(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


def sent(route):
    return json.loads(route.calls.last.request.content)


@respx.mock
async def test_access_rules_round_trip_without_quick_share(client):
    respx.get(f"{BASE_URL}/v1/me/access-rules").respond(200, json={"data": LIBRARY})
    rules = await invoke(client.get_access_rules)
    assert rules.version == 4
    assert rules.quick_share_rule_id == RULE_ID
    assert rules.items[0].link_limits == AccessRuleLinkLimits(expires_in="1h", one_time_use=False)
    assert rules.items[1].payment == AccessRulePayment("price_1PqA2b3C4d5E6f7G", 0)
    assert rules.built_ins[0].mode == "review"

    put = respx.put(f"{BASE_URL}/v1/me/access-rules").respond(
        200, json={"data": {**LIBRARY, "version": 5}}
    )
    saved = await invoke(client.save_access_rules, version=rules.version, items=rules.items)
    # Desktop owns quick_share_rule_id; the service keeps it when a save omits it.
    assert sent(put) == {"version": 4, "items": [PUBLIC_RULE, PAID_RULE]}
    assert saved.version == 5
    assert "idempotency-key" in put.calls.last.request.headers


@respx.mock
async def test_save_access_rules_conflict_is_not_retried(client):
    route = respx.put(f"{BASE_URL}/v1/me/access-rules").respond(
        409, json={"error": {"status": 409, "code": "conflict", "title": "Conflict"}}
    )
    with pytest.raises(ConflictError):
        await invoke(client.save_access_rules, version=1, items=[])
    assert route.call_count == 1
    assert sent(route) == {"version": 1, "items": []}


@pytest.mark.parametrize(
    "data",
    [
        {k: v for k, v in LIBRARY.items() if k != "items"},
        {k: v for k, v in LIBRARY.items() if k != "version"},
        {**LIBRARY, "account_id": ""},
        {**LIBRARY, "items": [{"id": RULE_ID, "mode": "public"}]},
        {**LIBRARY, "items": ["not-a-rule"]},
    ],
)
@respx.mock
async def test_incomplete_library_fails_closed(client, data):
    respx.get(f"{BASE_URL}/v1/me/access-rules").respond(200, json={"data": data})
    with pytest.raises(QURLError) as raised:
        await invoke(client.get_access_rules)
    assert raised.value.code == "unexpected_response"


@pytest.mark.parametrize(
    "items",
    [
        [{"id": RULE_ID, "name": "x", "mode": "public"}],
        [AccessRule(RULE_ID, "bad\ud800", "public")],
    ],
)
async def test_save_access_rules_rejects_bad_items_locally(client, items):
    with pytest.raises(ValueError):
        await invoke(client.save_access_rules, version=0, items=items)


@respx.mock
async def test_inbox_state_round_trip(client):
    respx.get(f"{BASE_URL}/v1/me/inbox-state").respond(
        200,
        json={"data": {"version": 2, "seen_through": "2026-11-01T10:00:00Z", "dismissed": ["k1"]}},
    )
    state = await invoke(client.get_inbox_state)
    assert state.version == 2
    assert state.seen_through == datetime(2026, 11, 1, 10, tzinfo=timezone.utc)
    assert state.dismissed == ["k1"]

    put = respx.put(f"{BASE_URL}/v1/me/inbox-state").respond(
        200, json={"data": {"version": 3, "dismissed": ["k1", "k2"]}}
    )
    saved = await invoke(client.save_inbox_state, version=2, dismissed=["k1", "k2"])
    assert sent(put) == {"version": 2, "dismissed": ["k1", "k2"]}
    assert saved.seen_through is None

    await invoke(
        client.save_inbox_state, version=3, dismissed=(),
        seen_through=datetime(2026, 11, 2, tzinfo=timezone.utc),
    )
    assert sent(put) == {"version": 3, "dismissed": [], "seen_through": "2026-11-02T00:00:00+00:00"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"version": -1, "dismissed": []},
        {"version": 0, "dismissed": "k1"},
        {"version": 0, "dismissed": [], "seen_through": datetime(2026, 11, 2)},
    ],
)
async def test_save_inbox_state_rejects_bad_input_locally(client, kwargs):
    with pytest.raises(ValueError):
        await invoke(client.save_inbox_state, **kwargs)


@respx.mock
async def test_received_history_round_trip(client):
    entry = {"crid": CRID, "opened_at": 1_700_000_000_000, "name": "Report", "added_at": 1}
    respx.get(f"{BASE_URL}/v1/me/received-history").respond(
        200, json={"data": {"version": 7, "cleared_at": 5, "items": [entry]}}
    )
    history = await invoke(client.get_received_history)
    assert history.cleared_at == 5
    assert history.items == [ReceivedLink(CRID, 1_700_000_000_000, "Report", 1)]

    put = respx.put(f"{BASE_URL}/v1/me/received-history").respond(
        200, json={"data": {"version": 7, "cleared_at": 5, "items": [entry]}}
    )
    await invoke(
        client.save_received_history,
        version=history.version,
        items=[*history.items, ReceivedLink(CRID, 1_700_000_000_001)],
    )
    # cleared_at is read-only; unset optional fields are omitted.
    assert sent(put) == {
        "version": 7,
        "items": [entry, {"crid": CRID, "opened_at": 1_700_000_000_001}],
    }


@respx.mock
async def test_clear_received_history_retries_conflict(client):
    conflict = httpx.Response(
        409, json={"error": {"status": 409, "code": "conflict", "title": "Conflict"}}
    )
    cleared = httpx.Response(200, json={"data": {"version": 9, "cleared_at": 8, "items": []}})
    route = respx.delete(f"{BASE_URL}/v1/me/received-history").mock(
        side_effect=[conflict, cleared]
    )
    history = await invoke(client.clear_received_history)
    assert route.call_count == 2
    assert (history.version, history.cleared_at, history.items) == (9, 8, [])


@respx.mock
async def test_link_defaults_set_clear_and_read(client):
    resource = {
        "resource_id": "r_1",
        "status": "active",
        "link_defaults": {"expires_in": "1h", "one_time_use": False, "rule_name": "Quick look"},
    }
    route = respx.patch(f"{BASE_URL}/v1/resources/r_1").respond(200, json={"data": resource})
    updated = await invoke(
        client.update_resource, "r_1",
        link_defaults=LinkDefaults(expires_in="1h", rule_name="Quick look"),
    )
    assert sent(route) == {
        "link_defaults": {"expires_in": "1h", "one_time_use": False, "rule_name": "Quick look"}
    }
    assert updated.link_defaults == LinkDefaults("1h", False, None, "Quick look")

    await invoke(client.update_resource, "r_1", link_defaults=None)
    assert sent(route) == {"link_defaults": None}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"link_defaults": {}},
        {"link_defaults": LinkDefaults("1h"), "custom_domain": "secure.example.com"},
    ],
)
async def test_link_defaults_rejects_bad_input_locally(client, kwargs):
    with pytest.raises(ValueError):
        await invoke(client.update_resource, "r_1", **kwargs)
