"""POST transport retries must not replay a request that may have committed."""

import inspect
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from layerv_qurl import AsyncQURLClient, QURLClient, QURLError, QURLNetworkError
from tests.client_support import _QUOTA_OK, BASE_URL

pytestmark = pytest.mark.asyncio
CREATE_OK = {
    "data": {
        "resource_id": "r_retry_test",
        "qurl_link": "https://qurl.link/#at_test",
        "qurl_site": "https://example.qurl.site",
    }
}


@pytest.fixture(params=[QURLClient, AsyncQURLClient], ids=["sync", "async"])
async def retrying_client(request):
    with (
        patch("layerv_qurl.client.time.sleep"),
        patch("layerv_qurl.async_client.asyncio.sleep", new=AsyncMock()),
    ):
        client = request.param(api_key="lv_test", base_url=BASE_URL, max_retries=1)
        yield client
        result = client.close()
        if inspect.isawaitable(result):
            await result


async def invoke(method, *args, **kwargs):
    result = method(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


@pytest.mark.parametrize(
    "error_type",
    [httpx.ReadTimeout, httpx.WriteError, httpx.RemoteProtocolError, httpx.TransportError],
)
@pytest.mark.parametrize(
    "method,path,args,kwargs",
    [
        ("create", "/v1/qurls", (), {"target_url": "https://example.com"}),
        ("resolve", "/v1/resolve", ("at_fixture",), {}),
        ("redeem_access_code", "/v1/access-codes/redeem", (), {"code": "fixture-code"}),
    ],
)
@respx.mock
async def test_ambiguous_post_failure_is_not_replayed(
    retrying_client, error_type, method, path, args, kwargs
):
    accepted = []

    def accepted_then_response_lost(request):
        accepted.append(request)
        raise error_type("response path failed", request=request)

    route = respx.post(BASE_URL + path).mock(side_effect=accepted_then_response_lost)
    with pytest.raises(QURLNetworkError) as raised:
        await invoke(getattr(retrying_client, method), *args, **kwargs)
    assert isinstance(raised.value.__cause__, error_type)
    assert route.call_count == 1
    assert len(accepted) == 1


@pytest.mark.parametrize(
    "error_type", [httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout]
)
@respx.mock
async def test_unsent_post_can_retry_with_the_same_body_and_key(retrying_client, error_type):
    route = respx.post(BASE_URL + "/v1/qurls")
    route.side_effect = [error_type("request not sent"), httpx.Response(201, json=CREATE_OK)]
    result = await invoke(retrying_client.create, target_url="https://example.com")
    assert result.resource_id == "r_retry_test"
    assert route.call_count == 2
    first, second = [call.request for call in route.calls]
    assert first.headers["Idempotency-Key"]
    assert first.headers["Idempotency-Key"] == second.headers["Idempotency-Key"]
    assert first.content == second.content


@pytest.mark.parametrize("status", [502, 503, 504])
@respx.mock
async def test_post_server_errors_still_do_not_retry(retrying_client, status):
    route = respx.post(BASE_URL + "/v1/qurls").mock(
        return_value=httpx.Response(
            status,
            json={"error": {"status": status, "code": "unavailable", "title": "Unavailable"}},
        )
    )
    with pytest.raises(QURLError) as raised:
        await invoke(retrying_client.create, target_url="https://example.com")
    assert raised.value.status == status
    assert route.call_count == 1


@respx.mock
async def test_post_rate_limit_can_still_retry(retrying_client):
    route = respx.post(BASE_URL + "/v1/qurls")
    route.side_effect = [
        httpx.Response(
            429, json={"error": {"status": 429, "code": "rate_limited", "title": "Rate limited"}}
        ),
        httpx.Response(201, json=CREATE_OK),
    ]
    await invoke(retrying_client.create, target_url="https://example.com")
    assert route.call_count == 2


@respx.mock
async def test_get_can_still_retry_a_read_timeout(retrying_client):
    route = respx.get(BASE_URL + "/v1/quota")
    route.side_effect = [httpx.ReadTimeout("response lost"), httpx.Response(200, json=_QUOTA_OK)]
    result = await invoke(retrying_client.get_quota)
    assert result.plan == "growth"
    assert route.call_count == 2


@pytest.mark.parametrize(
    "error_type", [httpx.ReadTimeout, httpx.WriteError, httpx.RemoteProtocolError]
)
@respx.mock
async def test_ambiguous_patch_extension_is_not_replayed(retrying_client, error_type):
    route = respx.patch(BASE_URL + "/v1/qurls/r_patch").mock(
        side_effect=error_type("extension may have committed")
    )
    with pytest.raises(QURLNetworkError):
        await invoke(retrying_client.update, "r_patch", extend_by="7d")
    assert route.call_count == 1


@pytest.mark.parametrize("status", [502, 503, 504])
@respx.mock
async def test_patch_server_error_is_not_replayed(retrying_client, status):
    route = respx.patch(BASE_URL + "/v1/qurls/r_patch").mock(
        return_value=httpx.Response(
            status,
            json={"error": {"status": status, "code": "unavailable", "title": "Unavailable"}},
        )
    )
    with pytest.raises(QURLError):
        await invoke(retrying_client.update, "r_patch", extend_by="7d")
    assert route.call_count == 1


@respx.mock
async def test_unsent_patch_can_retry(retrying_client):
    route = respx.patch(BASE_URL + "/v1/qurls/r_patch")
    route.side_effect = [
        httpx.ConnectError("not sent"),
        httpx.Response(
            200,
            json={
                "data": {
                    "resource_id": "r_patch",
                    "target_url": "https://example.com",
                    "status": "active",
                    "created_at": "2026-01-01T00:00:00Z",
                    "tags": [],
                }
            },
        ),
    ]
    await invoke(retrying_client.update, "r_patch", extend_by="7d")
    assert route.call_count == 2
    first, second = [call.request for call in route.calls]
    assert first.content == second.content
    assert first.headers["Idempotency-Key"] == second.headers["Idempotency-Key"]


@respx.mock
async def test_delete_can_still_retry_a_read_timeout(retrying_client):
    route = respx.delete(BASE_URL + "/v1/qurls/r_delete")
    route.side_effect = [httpx.ReadTimeout("response lost"), httpx.Response(204)]
    await invoke(retrying_client.delete, "r_delete")
    assert route.call_count == 2
