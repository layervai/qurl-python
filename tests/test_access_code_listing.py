"""Access-code listing is a complete, unpaginated API response."""

import inspect

import httpx
import pytest
import respx

from layerv_qurl import AsyncQURLClient, QURLClient, QURLError
from tests.client_support import BASE_URL

pytestmark = pytest.mark.asyncio


@pytest.fixture(params=[QURLClient, AsyncQURLClient], ids=["sync", "async"])
async def client(request):
    instance = request.param(api_key="lv_test", base_url=BASE_URL, max_retries=0)
    yield instance
    result = instance.close()
    if inspect.isawaitable(result):
        await result


async def invoke(method, **kwargs):
    result = method(**kwargs)
    return await result if inspect.isawaitable(result) else result


async def collect(iterator):
    if hasattr(iterator, "__aiter__"):
        return [item async for item in iterator]
    return list(iterator)


@respx.mock
async def test_all_access_codes_fetch_once_without_pagination(client):
    def complete_response(request):
        assert not request.url.query
        return httpx.Response(
            200,
            json={
                "data": [
                    {"access_code_id": "acd_first", "resource_id": "r_test"},
                    {"access_code_id": "acd_last", "resource_id": "r_test"},
                ]
            },
        )

    route = respx.get(BASE_URL + "/v1/access-codes").mock(side_effect=complete_response)
    iterator = client.list_access_codes_all()
    assert route.call_count == 0
    result = await collect(iterator)
    assert [item.access_code_id for item in result] == ["acd_first", "acd_last"]
    assert route.call_count == 1


@pytest.mark.parametrize(
    "meta",
    [{"has_more": True}, {"next_cursor": "unexpected"}, {"has_more": True, "next_cursor": "next"}],
)
@respx.mock
async def test_unexpected_pagination_fails_before_yielding_partial_codes(client, meta):
    route = respx.get(BASE_URL + "/v1/access-codes").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [{"access_code_id": "acd_partial", "resource_id": "r_test"}],
                "meta": meta,
            },
        )
    )
    with pytest.raises(QURLError) as raised:
        await collect(client.list_access_codes_all())
    assert raised.value.code == "unexpected_response"
    assert route.call_count == 1


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("list_access_codes", {"limit": 1}),
        ("list_access_codes", {"cursor": "unsupported"}),
        ("list_access_codes_all", {"page_size": 1}),
    ],
)
@respx.mock
async def test_unsupported_pagination_arguments_fail_before_a_request(client, method, kwargs):
    route = respx.get(BASE_URL + "/v1/access-codes").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    with pytest.raises(TypeError):
        await invoke(getattr(client, method), **kwargs)
    assert route.call_count == 0
