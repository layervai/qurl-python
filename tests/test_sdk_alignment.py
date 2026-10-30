"""Current HTTP resource/sharing contract, with qurl-go as the reference."""
from __future__ import annotations

import base64
import inspect
import json
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
import respx

from layerv_qurl import AsyncQURLClient, QURLClient, ValidationError

BASE = "https://api.test.layerv.ai"
VECTORS = json.loads((Path(__file__).parent / "fixtures/crid_v1_vectors.json").read_text())
CRID = VECTORS["producer_cases"][0]["expected_crid"]
KEY = VECTORS["producer_cases"][0]["der_spki_b64url"]
OTHER_CRID = VECTORS["producer_cases"][2]["expected_crid"]
RESOURCE = {"resource_id": KEY, "crid": CRID, "status": "active"}
SHARE = {"qurl": "https://qurl.link/#qv2.claims.secret.signature", "crid": CRID,
         "qurl_id": "q_shared", "type": "qv2", "expires_in_seconds": 60,
         "single_use": False}


@pytest.fixture(params=[QURLClient, AsyncQURLClient], ids=["sync", "async"])
async def client(request):
    instance = request.param(api_key="lv_test", base_url=BASE, max_retries=0)
    yield instance
    if isinstance(instance, AsyncQURLClient):
        await instance.close()
    else:
        instance.close()


async def invoke(method, *args, **kwargs):
    result = method(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


@pytest.mark.parametrize("identifier", [CRID, KEY])
@respx.mock
async def test_delete_accepts_public_identifiers(client, identifier):
    route = respx.delete(f"{BASE}/v1/qurls/{identifier}").respond(204)
    await invoke(client.delete, identifier)
    assert route.called


@pytest.mark.parametrize("private", [None, False, True])
@respx.mock
async def test_protect_privacy_and_crid_handle(client, private):
    data = dict(RESOURCE)
    if private is not None:
        data["private"] = private
    route = respx.post(f"{BASE}/v1/resources").respond(201, json={"data": data})
    resource = await invoke(client.protect_url, "https://example.com", private=private)
    assert resource.id == CRID
    assert resource.details.crid == CRID
    assert resource.details.resource_id == KEY
    assert resource.details.private is private
    body = json.loads(route.calls[0].request.content)
    assert ("private" in body) == (private is not None)
    if private is not None:
        assert body["private"] is private


@respx.mock
async def test_resource_grants_round_trip(client):
    keys = [base64.b64encode(bytes(range(32))).decode()]
    route = respx.post(f"{BASE}/v1/resources").respond(
        201, json={"data": {**RESOURCE, "private": True, "allowed_device_keys": keys}})
    result = await invoke(client.create_resource, target_url="https://example.com",
                          private=True, allowed_device_keys=keys)
    assert result.allowed_device_keys == keys
    assert json.loads(route.calls[0].request.content)["allowed_device_keys"] == keys
    patch = respx.patch(f"{BASE}/v1/resources/{CRID}").respond(
        200, json={"data": {**RESOURCE, "allowed_device_keys": []}})
    result = await invoke(client.update_resource, CRID, allowed_device_keys=[])
    assert result.allowed_device_keys == []
    assert json.loads(patch.calls[0].request.content) == {"allowed_device_keys": []}


@respx.mock
async def test_share_validates_response_and_duration_contract(client):
    route = respx.post(f"{BASE}/v1/resources/{CRID}/share").respond(200, json={
        "data": {**SHARE, "publisher": {"name": "Example", "verified": True},
                 "resource_created_at": "2026-10-01T00:00:00Z"}})
    link = await invoke(client.share_resource, CRID, ttl=timedelta(seconds=60),
                        session_duration=timedelta(seconds=5))
    assert json.loads(route.calls[0].request.content) == {
        "ttl_seconds": 60, "session_duration": "5s"}
    assert link.crid == CRID
    assert link.link == SHARE["qurl"]
    assert link.qurl_id == "q_shared"
    assert link.publisher.name == "Example"
    assert link.publisher.verified is True
    assert link.resource_created_at.year == 2026
    assert link.verify_crid(base64.urlsafe_b64decode(KEY + "==")) is None
    with pytest.raises(ValueError, match="match"):
        link.verify_crid(b"substituted key")


@pytest.mark.parametrize("data", [
    {**SHARE, "crid": OTHER_CRID}, {**SHARE, "crid": ""},
    {**SHARE, "qurl": " "}, {**SHARE, "publisher": {"verified": "true"}},
])
@respx.mock
async def test_share_rejects_invalid_response(client, data):
    respx.post(f"{BASE}/v1/resources/{CRID}/share").respond(200, json={"data": data})
    with pytest.raises(ValidationError):
        await invoke(client.share_resource, CRID)


@respx.mock
async def test_share_defaults_and_no_replay_promise(client):
    route = respx.post(f"{BASE}/v1/resources/{CRID}/share").respond(200, json={"data": SHARE})
    link = await invoke(client.share_resource, CRID, ttl=timedelta(0))
    assert json.loads(route.calls[0].request.content) == {}
    assert link.publisher.verified is False
    assert link.resource_created_at is None


@pytest.mark.parametrize("ttl", [timedelta(seconds=-1), timedelta(microseconds=1)])
@respx.mock
async def test_share_rejects_invalid_ttl_without_request(client, ttl):
    with pytest.raises(ValueError):
        await invoke(client.share_resource, CRID, ttl=ttl)
    assert not respx.calls


@pytest.mark.parametrize("vector", VECTORS["consumer_value_cases"], ids=lambda v: v["name"])
def test_crid_gate_matches_shared_vectors(vector):
    from layerv_qurl.crid import validate
    if vector["outcome"] == "accept":
        assert validate(vector["value"]) == vector["value"]
    else:
        with pytest.raises(ValueError, match=vector["reject_class"]):
            validate(vector["value"])


@pytest.mark.parametrize("vector", VECTORS["key_match_cases"], ids=lambda v: v["name"])
def test_crid_key_binding_matches_shared_vectors(vector):
    from layerv_qurl.crid import key_matches
    assert key_matches(vector["crid"], base64.urlsafe_b64decode(
        vector["der_spki_b64url"] + "==")) is (vector["outcome"] == "match")


@respx.mock
async def test_crid_handle_rejects_mismatched_mint_response(client):
    resource = client.resource_by_crid(CRID)
    respx.post(f"{BASE}/v1/resources/{CRID}/qurls").respond(201, json={"data": {
        "resource_id": KEY, "crid": OTHER_CRID, "qurl_link": SHARE["qurl"]}})
    with pytest.raises(ValidationError):
        await invoke(resource.create_portal)


@pytest.mark.parametrize("kwargs", [
    {"allowed_device_keys": [], "allowed_device_keys_add": []},
    {"allowed_device_keys_remove": [], "custom_domain": "secure.example.com"},
    {"allowed_device_keys": ["not a key"]},
    {"allowed_device_keys": "not a list"},
    {"allowed_device_keys": {"key": "value"}},
    {"allowed_device_keys": {"a", "b"}},
    {"allowed_device_keys_add": [base64.b64encode(bytes(32)).decode()],
     "allowed_device_keys_remove": [base64.b64encode(bytes(32)).decode()]},
])
@respx.mock
async def test_invalid_grant_updates_fail_before_request(client, kwargs):
    with pytest.raises(ValueError):
        await invoke(client.update_resource, CRID, **kwargs)
    assert not respx.calls


@respx.mock
async def test_atomic_grant_updates_preserve_other_keys(client):
    key = base64.b64encode(bytes(range(32))).decode()
    route = respx.patch(f"{BASE}/v1/resources/{CRID}").respond(200, json={"data": RESOURCE})
    await invoke(client.update_resource, CRID, allowed_device_keys_add=(k for k in [key]),
                 allowed_device_keys_remove=[])
    assert json.loads(route.calls[0].request.content) == {
        "allowed_device_keys_add": [key], "allowed_device_keys_remove": []}


@pytest.mark.parametrize("private", ["false", 0, 1])
@respx.mock
async def test_private_only_accepts_boolean(client, private):
    with pytest.raises(ValueError):
        await invoke(client.protect_url, "https://example.com", private=private)
    assert not respx.calls


@pytest.mark.parametrize("kwargs", [
    {"ttl": "5s"}, {"session_duration": "5s"},
    {"session_duration": timedelta(microseconds=1)},
    {"session_duration": timedelta(seconds=-1)},
])
@respx.mock
async def test_share_rejects_invalid_duration_types(client, kwargs):
    with pytest.raises(ValueError):
        await invoke(client.share_resource, CRID, **kwargs)
    assert not respx.calls


@pytest.mark.parametrize("data", [
    {**RESOURCE, "crid": OTHER_CRID}, {**RESOURCE, "private": "false"},
    {**RESOURCE, "crid": CRID.upper()}, {**RESOURCE, "resource_id": "not-a-key"},
])
@respx.mock
async def test_protect_rejects_malformed_current_identity(client, data):
    respx.post(f"{BASE}/v1/resources").respond(201, json={"data": data})
    with pytest.raises(ValidationError):
        await invoke(client.protect_url, "https://example.com")


@pytest.mark.parametrize("publisher", [[], "verified", {"verified": 1}, {"name": 123}])
@respx.mock
async def test_share_rejects_malformed_publisher(client, publisher):
    respx.post(f"{BASE}/v1/resources/{CRID}/share").respond(
        200, json={"data": {**SHARE, "publisher": publisher}})
    with pytest.raises(ValidationError):
        await invoke(client.share_resource, CRID)


@respx.mock
async def test_share_failure_does_not_retry_mint(client):
    from layerv_qurl import ServerError
    route = respx.post(f"{BASE}/v1/resources/{CRID}/share").respond(
        503, json={"code": "temporary_access_links_disabled", "title": "Unavailable"})
    client._max_retries = 2
    with pytest.raises(ServerError):
        await invoke(client.share_resource, CRID)
    assert route.call_count == 1


@respx.mock
async def test_create_portal_for_url_carries_privacy_and_crid(client):
    route = respx.post(f"{BASE}/v1/qurls").respond(201, json={"data": {
        **RESOURCE, "qurl_link": SHARE["qurl"]}})
    portal, resource = await invoke(
        client.create_portal_for_url, "https://example.com", private=False)
    assert portal.crid == resource.id == CRID
    assert json.loads(route.calls[0].request.content)["private"] is False


@respx.mock
async def test_share_does_not_remint_after_ambiguous_timeout(client):
    import httpx

    from layerv_qurl import QURLTimeoutError
    route = respx.post(f"{BASE}/v1/resources/{CRID}/share").mock(
        side_effect=[httpx.ReadTimeout("response lost"), httpx.Response(200, json={"data": SHARE})])
    client._max_retries = 1
    with pytest.raises(QURLTimeoutError):
        await invoke(client.share_resource, CRID)
    assert route.call_count == 1


@pytest.mark.parametrize("fields", [
    {"single_use": "false"}, {"expires_in_seconds": "60"},
    {"expires_in_seconds": True}, {"resource_created_at": 42},
    {"expires_at": "not a date"}, {"qurl_id": 123}, {"type": []},
    {"resource_created_at": "2026-10-01"},
])
@respx.mock
async def test_share_rejects_malformed_typed_fields(client, fields):
    respx.post(f"{BASE}/v1/resources/{CRID}/share").respond(
        200, json={"data": {**SHARE, **fields}})
    with pytest.raises(ValidationError):
        await invoke(client.share_resource, CRID)


def test_share_durations_do_not_lose_subseconds_for_large_values():
    from layerv_qurl._utils import build_share_body
    with pytest.raises(ValueError):
        build_share_body(timedelta(days=999999999, microseconds=1), None)


@pytest.mark.parametrize("fields", [
    {"access_requests": "false"}, {"allowed_device_keys": "key"},
    {"allowed_device_keys": [1]},
])
@respx.mock
async def test_resource_rejects_malformed_grant_metadata(client, fields):
    respx.get(f"{BASE}/v1/resources/{CRID}").respond(
        200, json={"data": {"resource": {**RESOURCE, **fields}}})
    with pytest.raises(ValidationError):
        await invoke(client.get_resource, CRID)


@respx.mock
async def test_share_429_is_returned_without_retry(client):
    from layerv_qurl import RateLimitError
    route = respx.post(f"{BASE}/v1/resources/{CRID}/share").respond(
        429, headers={"Retry-After": "1"}, json={"title": "Too many requests"})
    client._max_retries = 2
    with pytest.raises(RateLimitError) as exc:
        await invoke(client.share_resource, CRID)
    assert exc.value.retry_after == 1
    assert route.call_count == 1


def test_crid_gate_rejects_non_string_consistently():
    from layerv_qurl.crid import validate
    with pytest.raises(ValueError):
        validate(None)


def test_crid_shape_dispatch_matches_go():
    from layerv_qurl.crid import matches_shape
    assert matches_shape(CRID)
    assert not matches_shape("X" * 60)
    assert not matches_shape(KEY)
    assert matches_shape("a" * 60)  # Shape is not validation.


@respx.mock
async def test_share_rejects_negative_lifetime(client):
    respx.post(f"{BASE}/v1/resources/{CRID}/share").respond(
        200, json={"data": {**SHARE, "expires_in_seconds": -1}})
    with pytest.raises(ValidationError):
        await invoke(client.share_resource, CRID)


@pytest.mark.parametrize("fraction, microsecond", [
    ("1", 100000), ("12", 120000), ("123", 123000), ("1234", 123400),
    ("12345", 123450), ("123456", 123456), ("1234567", 123456),
    ("12345678", 123456), ("123456789", 123456),
])
@respx.mock
async def test_share_accepts_go_rfc3339nano_on_supported_pythons(client, fraction, microsecond):
    respx.post(f"{BASE}/v1/resources/{CRID}/share").respond(200, json={"data": {
        **SHARE, "expires_at": f"2026-10-01T00:00:00.{fraction}Z",
        "resource_created_at": f"2026-09-01T00:00:00.{fraction}+00:00"}})
    link = await invoke(client.share_resource, CRID)
    assert link.expires_at.microsecond == microsecond
    assert link.resource_created_at.microsecond == microsecond


@respx.mock
async def test_revoke_one_token_keeps_parent_resource(client):
    token_id = "q_0123456789a"
    route = respx.delete(f"{BASE}/v1/resources/{CRID}/qurls/{token_id}").respond(204)
    assert await invoke(client.revoke_resource_qurl, CRID, token_id) is None
    assert route.call_count == 1
    assert len(respx.calls) == 1


@pytest.mark.parametrize("method", ["revoke_resource_qurl", "update_resource_qurl"])
@pytest.mark.parametrize("resource_id,token_id", [
    (CRID, "at_synthetic_secret"), (CRID, "q_0123456789a\n"),
    (CRID, "q_0123456789A"), (CRID, "q_0123456789"),
    (CRID, "q_0123456789ab"), (CRID, "r_parent"), (CRID, ".."), (CRID, "q_0123456789a/path"),
    ("at_synthetic_secret", "q_0123456789a"),
    ("at_synthetic_secret\n", "q_0123456789a"),
])
@respx.mock
async def test_token_item_methods_refuse_wrong_ids_without_echo_or_http(
    client, method, resource_id, token_id
):
    with pytest.raises(ValueError) as error:
        await invoke(getattr(client, method), resource_id, token_id)
    assert "synthetic_secret" not in str(error.value)
    assert not respx.calls


@pytest.mark.parametrize("client_type", [QURLClient, AsyncQURLClient])
@respx.mock
async def test_token_revoke_retries_pending_enforcement(client_type):
    instance = client_type(api_key="lv_test", base_url=BASE, max_retries=1)
    token_id = "q_0123456789a"
    route = respx.delete(f"{BASE}/v1/resources/{CRID}/qurls/{token_id}").mock(side_effect=[
        httpx.Response(503, headers={"Retry-After": "0"}, json={"error": {
            "status": 503, "code": "protection_update_failed", "title": "Enforcement pending"}}),
        httpx.Response(204),
    ])
    try:
        assert await invoke(instance.revoke_resource_qurl, CRID, token_id) is None
        assert route.call_count == 2
    finally:
        await invoke(instance.close)


@pytest.mark.parametrize("method", ["delete", "delete_resource"])
@pytest.mark.parametrize("credential", ["at_synthetic_secret", "at_synthetic_secret\n"])
@respx.mock
async def test_resource_deletes_refuse_credentials_without_echo(client, method, credential):
    with pytest.raises(ValueError) as error:
        await invoke(getattr(client, method), credential)
    assert "synthetic_secret" not in str(error.value)
    assert not respx.calls
