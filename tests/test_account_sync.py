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
    QURLNetworkError,
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


@respx.mock
async def test_save_access_rules_conflict_is_not_retried(client):
    route = respx.put(f"{BASE_URL}/v1/me/access-rules").respond(
        409, json={"error": {"status": 409, "code": "conflict", "title": "Conflict"}}
    )
    with pytest.raises(ConflictError):
        await invoke(client.save_access_rules, version=1, items=[])
    assert route.call_count == 1
    assert sent(route) == {"version": 1, "items": []}


@respx.mock
async def test_unmodeled_rule_fields_round_trip(client):
    # A field a newer service or client adds must survive an older SDK's save.
    future = {
        **PUBLIC_RULE,
        "audience": {"kind": "team", "ids": ["t1"]},
        "extra": None,
        "link_limits": {**PUBLIC_RULE["link_limits"], "max_sessions": 3},
    }
    paid = {**PAID_RULE, "payment": {**PAID_RULE["payment"], "currency": "usd"}}
    respx.get(f"{BASE_URL}/v1/me/access-rules").respond(
        200, json={"data": {**LIBRARY, "items": [future, paid]}}
    )
    put = respx.put(f"{BASE_URL}/v1/me/access-rules").respond(200, json={"data": LIBRARY})
    rules = await invoke(client.get_access_rules)
    assert rules.items[0].extra == {"audience": {"kind": "team", "ids": ["t1"]}, "extra": None}
    assert "audience" not in repr(rules.items[0])
    await invoke(client.save_access_rules, version=rules.version, items=rules.items)
    assert sent(put)["items"] == [future, paid]


@pytest.mark.parametrize(
    "data",
    [
        {k: v for k, v in LIBRARY.items() if k != "items"},
        {k: v for k, v in LIBRARY.items() if k != "version"},
        {**LIBRARY, "account_id": ""},
        {**LIBRARY, "items": [{"id": RULE_ID, "mode": "public"}]},
        {**LIBRARY, "items": ["not-a-rule"]},
        {**LIBRARY, "items": [{**PUBLIC_RULE, "id": 1}]},
        {**LIBRARY, "items": [{**PAID_RULE, "payment": {"price_id": "p", "access_seconds": "0"}}]},
        {**LIBRARY, "items": [{**PUBLIC_RULE, "link_limits": {"expires_in": None}}]},
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
        {"version": 0, "dismissed": [], "seen_through": "2026-11-02T00:00:00Z"},
        {"version": 0, "dismissed": [], "seen_through": datetime(2026, 11, 2).date()},
    ],
)
async def test_save_inbox_state_rejects_bad_input_locally(client, kwargs):
    with pytest.raises(ValueError):
        await invoke(client.save_inbox_state, **kwargs)


@respx.mock
async def test_null_optional_fields_read_as_absent(client):
    respx.get(f"{BASE_URL}/v1/me/inbox-state").respond(
        200, json={"data": {"version": 1, "dismissed": [], "seen_through": None}}
    )
    respx.get(f"{BASE_URL}/v1/me/received-history").respond(200, json={"data": {
        "version": 1, "cleared_at": None,
        "items": [{"crid": CRID, "opened_at": 1, "name": None, "added_at": None}],
    }})
    respx.get(f"{BASE_URL}/v1/me/access-rules").respond(200, json={"data": {
        **LIBRARY, "quick_share_rule_id": None,
        "items": [{**PUBLIC_RULE, "email_domain": None,
                   "link_limits": {"expires_in": "1h", "one_time_use": None}}],
    }})
    assert (await invoke(client.get_inbox_state)).seen_through is None
    history = await invoke(client.get_received_history)
    assert history.cleared_at is None and history.items == [ReceivedLink(CRID, 1)]
    rules = await invoke(client.get_access_rules)
    assert rules.quick_share_rule_id is None
    assert rules.items[0].link_limits == AccessRuleLinkLimits(expires_in="1h")


@respx.mock
async def test_unmodeled_received_link_fields_round_trip(client):
    entry = {"crid": CRID, "opened_at": 1, "source": {"app": "desktop"}, "extra": None}
    respx.get(f"{BASE_URL}/v1/me/received-history").respond(
        200, json={"data": {"version": 2, "items": [entry]}}
    )
    put = respx.put(f"{BASE_URL}/v1/me/received-history").respond(
        200, json={"data": {"version": 3, "items": [entry]}}
    )
    history = await invoke(client.get_received_history)
    assert history.items[0].extra == {"source": {"app": "desktop"}, "extra": None}
    await invoke(client.save_received_history, version=history.version, items=history.items)
    assert sent(put)["items"] == [entry]


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
        {"link_defaults": None, "custom_domain": "secure.example.com"},
    ],
)
async def test_link_defaults_rejects_bad_input_locally(client, kwargs):
    with pytest.raises(ValueError):
        await invoke(client.update_resource, "r_1", **kwargs)


@respx.mock
async def test_missing_quick_share_and_empty_exceptions(client):
    rule = AccessRule(RULE_ID, "Domain", "email_domain", email_domain="example.com",
                      allowed_emails=[])
    respx.get(f"{BASE_URL}/v1/me/access-rules").respond(
        200, json={"data": {k: v for k, v in LIBRARY.items() if k != "quick_share_rule_id"}}
    )
    assert (await invoke(client.get_access_rules)).quick_share_rule_id is None
    put = respx.put(f"{BASE_URL}/v1/me/access-rules").respond(200, json={"data": LIBRARY})
    await invoke(client.save_access_rules, version=0, items=[rule])
    assert sent(put)["items"] == [
        {"id": RULE_ID, "name": "Domain", "mode": "email_domain", "email_domain": "example.com"}
    ]


@pytest.mark.parametrize(
    "path,method,data",
    [
        ("inbox-state", "get_inbox_state", {"dismissed": []}),
        ("inbox-state", "get_inbox_state", {"version": 1}),
        ("inbox-state", "get_inbox_state", {"version": 1, "dismissed": [3]}),
        ("received-history", "get_received_history", {"items": []}),
        ("received-history", "get_received_history", {"version": True, "items": []}),
        ("received-history", "get_received_history", {"version": 1, "items": [{"crid": CRID}]}),
        (
            "received-history",
            "get_received_history",
            {"version": 1, "items": [{"crid": CRID, "opened_at": "x"}]},
        ),
        # A malformed or mistyped optional field fails closed too.
        ("inbox-state", "get_inbox_state", {"version": 1, "dismissed": [], "seen_through": "x"}),
        ("inbox-state", "get_inbox_state", {"version": 1, "dismissed": [], "seen_through": 5}),
        ("inbox-state", "get_inbox_state",
         {"version": 1, "dismissed": [], "seen_through": "2026-11-02T00:00:00"}),
        ("received-history", "get_received_history",
         {"version": 1, "items": [], "cleared_at": "5"}),
        (
            "received-history",
            "get_received_history",
            {"version": 1, "items": [{"crid": CRID, "opened_at": 1, "name": 5}]},
        ),
        (
            "received-history",
            "get_received_history",
            {"version": 1, "items": [{"crid": CRID, "opened_at": 1, "added_at": True}]},
        ),
        ("access-rules", "get_access_rules", {**LIBRARY, "quick_share_rule_id": 5}),
        ("access-rules", "get_access_rules",
         {**LIBRARY, "items": [{**PAID_RULE, "contract_id": 1}]}),
        (
            "access-rules",
            "get_access_rules",
            {**LIBRARY, "items": [{**PUBLIC_RULE, "approval_expires_in": ["1h"]}]},
        ),
        (
            "access-rules",
            "get_access_rules",
            {**LIBRARY, "items": [{**PUBLIC_RULE, "email_domain": False}]},
        ),
        (
            "access-rules",
            "get_access_rules",
            {**LIBRARY, "items": [{**PUBLIC_RULE, "link_limits": {"expires_in": "1h",
                                                                   "one_time_use": "no"}}]},
        ),
        (
            "access-rules",
            "get_access_rules",
            {**LIBRARY, "items": [{**PUBLIC_RULE, "link_limits": {"expires_in": "1h",
                                                                   "session_duration": 60}}]},
        ),
    ],
)
@respx.mock
async def test_incomplete_documents_fail_closed(client, path, method, data):
    respx.get(f"{BASE_URL}/v1/me/{path}").respond(200, json={"data": data})
    with pytest.raises(QURLError) as raised:
        await invoke(getattr(client, method))
    assert raised.value.code == "unexpected_response"


@respx.mock
async def test_clear_received_history_raises_after_retries(client):
    route = respx.delete(f"{BASE_URL}/v1/me/received-history").respond(
        409, json={"error": {"status": 409, "code": "conflict", "title": "Conflict"}}
    )
    with pytest.raises(ConflictError):
        await invoke(client.clear_received_history)
    assert route.call_count == 3  # max_retries=2


@pytest.mark.parametrize(
    "link_defaults", [{"one_time_use": True}, {"expires_in": "1h", "one_time_use": "true"}]
)
@respx.mock
async def test_malformed_link_defaults_fail_closed(client, link_defaults):
    respx.get(f"{BASE_URL}/v1/resources/r_1").respond(
        200, json={"data": {"resource": {"resource_id": "r_1", "status": "active",
                                         "link_defaults": link_defaults}}}
    )
    with pytest.raises(QURLError) as raised:
        await invoke(client.get_resource, "r_1")
    assert raised.value.code == "unexpected_response"


@pytest.mark.parametrize(
    "path,method,kwargs",
    [
        ("access-rules", "save_access_rules", {"version": 1, "items": []}),
        ("inbox-state", "save_inbox_state", {"version": 1, "dismissed": []}),
        ("received-history", "save_received_history", {"version": 1, "items": []}),
    ],
)
@pytest.mark.parametrize("failure", [503, 429, httpx.ConnectError("reset")])
@respx.mock
async def test_saves_are_never_retried(client, path, method, kwargs, failure):
    # The client has max_retries=2; a replayed save that landed would raise ConflictError.
    route = respx.put(f"{BASE_URL}/v1/me/{path}")
    if isinstance(failure, int):
        route.respond(failure, json={"error": {"status": failure, "code": "x", "title": "x"}})
    else:
        route.mock(side_effect=failure)
    with pytest.raises((QURLError, QURLNetworkError)):
        await invoke(getattr(client, method), **kwargs)
    assert route.call_count == 1


@respx.mock
async def test_email_domain_exceptions_round_trip(client):
    rule = {**PUBLIC_RULE, "mode": "email_domain", "email_domain": "example.com",
            "allowed_emails": ["Ann@example.org"]}
    del rule["link_limits"]
    respx.get(f"{BASE_URL}/v1/me/access-rules").respond(
        200, json={"data": {**LIBRARY, "items": [rule]}}
    )
    put = respx.put(f"{BASE_URL}/v1/me/access-rules").respond(200, json={"data": LIBRARY})
    rules = await invoke(client.get_access_rules)
    await invoke(client.save_access_rules, version=rules.version, items=rules.items)
    assert sent(put)["items"] == [rule]


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("save_received_history", {"version": 0, "items": [{"crid": CRID, "opened_at": 1}]}),
        ("save_received_history", {"version": 0, "items": [ReceivedLink(CRID, 1, "bad\ud800")]}),
        ("update_resource", {"link_defaults": LinkDefaults("1h", rule_name="bad\ud800")}),
        ("save_access_rules", {"version": 0, "items": [AccessRule(RULE_ID, 5, "public")]}),
    ],
)
async def test_bad_write_input_rejected_locally(client, method, kwargs):
    args = ("r_1",) if method == "update_resource" else ()
    with pytest.raises(ValueError):
        await invoke(getattr(client, method), *args, **kwargs)
