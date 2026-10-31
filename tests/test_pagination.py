"""Cursor iteration contracts, independent of the HTTP parsers."""

from unittest.mock import AsyncMock, Mock, call, patch

import httpx
import pytest
import respx

from layerv_qurl import AsyncQURLClient, QURLClient, QURLError, types
from layerv_qurl._pagination import async_pages, pages

CASES = [
    (
        "resources",
        types.ResourceListOutput,
        "resources",
        (),
        {"alias": "docs", "slug": "app", "status": "active", "resource_type": "url"},
    ),
    ("domains", types.DomainListOutput, "domains", (), {}),
    ("webhooks", types.WebhookListOutput, "webhooks", (), {"event": "resource.revoked"}),
    ("webhook_deliveries", types.WebhookDeliveryListOutput, "deliveries", ("wh_test",), {}),
    ("api_keys", types.APIKeyListOutput, "api_keys", (), {"status": "active"}),
    ("access_codes", types.AccessCodeListOutput, "access_codes", (), {}),
    ("billing_invoices", types.InvoiceListOutput, "invoices", (), {}),
    ("connector_installations", types.ConnectorInstallationListOutput, "installations", (), {}),
]


@pytest.mark.parametrize("name,page_type,field,args,filters", CASES)
def test_contract_iterators_are_lazy_and_preserve_filters(name, page_type, field, args, filters):
    first, last = object(), object()
    fetch = Mock(
        side_effect=[
            page_type(**{field: [first]}, has_more=True, next_cursor="next"),
            page_type(has_more=True, next_cursor="after-empty"),
            page_type(**{field: [last]}, has_more=False, next_cursor="ignored"),
        ]
    )
    with QURLClient(api_key="test-key") as client, patch.object(client, "list_" + name, fetch):
        iterator = getattr(client, "list_" + name + "_all")(*args, page_size=2, **filters)
        fetch.assert_not_called()
        assert next(iterator) is first
        assert fetch.call_count == 1
        assert list(iterator) == [last]
        assert fetch.call_args_list == [
            call(*args, limit=2, cursor=cursor, **filters)
            for cursor in [None, "next", "after-empty"]
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("name,page_type,field,args,filters", CASES)
async def test_async_contract_iterators_match_sync(name, page_type, field, args, filters):
    first, last = object(), object()
    fetch = AsyncMock(
        side_effect=[
            page_type(**{field: [first]}, has_more=True, next_cursor="next"),
            page_type(has_more=True, next_cursor="after-empty"),
            page_type(**{field: [last]}, has_more=False, next_cursor="ignored"),
        ]
    )
    async with AsyncQURLClient(api_key="test-key") as client:
        with patch.object(client, "list_" + name, fetch):
            iterator = getattr(client, "list_" + name + "_all")(*args, page_size=2, **filters)
            fetch.assert_not_called()
            assert await anext(iterator) is first
            assert fetch.await_count == 1
            assert [item async for item in iterator] == [last]
            assert fetch.await_args_list == [
                call(*args, limit=2, cursor=cursor, **filters)
                for cursor in [None, "next", "after-empty"]
            ]


@pytest.mark.parametrize(
    "cursors", [[None], [""], [123], [{}], ["same", "same"], ["one", "two", "one"]]
)
def test_pagination_rejects_missing_and_repeated_cursors(cursors):
    fetch = Mock(
        side_effect=[
            types.DomainListOutput(has_more=True, next_cursor=cursor) for cursor in cursors
        ]
    )
    with pytest.raises(QURLError, match="Pagination requires") as raised:
        list(pages(fetch))
    assert raised.value.code == "unexpected_response"
    assert fetch.call_count == len(cursors)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cursors", [[None], [""], [123], [{}], ["same", "same"], ["one", "two", "one"]]
)
async def test_async_pagination_rejects_missing_and_repeated_cursors(cursors):
    fetch = AsyncMock(
        side_effect=[
            types.DomainListOutput(has_more=True, next_cursor=cursor) for cursor in cursors
        ]
    )
    with pytest.raises(QURLError, match="Pagination requires"):
        [page async for page in async_pages(fetch)]
    assert fetch.await_count == len(cursors)


def test_iterator_can_stop_early_and_propagates_page_failure():
    failure = RuntimeError("page unavailable")
    fetch = Mock(side_effect=[types.DomainListOutput(has_more=True, next_cursor="next"), failure])
    iterator = pages(fetch)
    next(iterator)
    assert fetch.call_count == 1
    iterator.close()
    assert fetch.call_count == 1
    fetch.reset_mock(side_effect=True)
    fetch.side_effect = [types.DomainListOutput(has_more=True, next_cursor="next"), failure]
    iterator = pages(fetch)
    next(iterator)
    with pytest.raises(RuntimeError, match="page unavailable"):
        next(iterator)


@pytest.mark.asyncio
async def test_async_iterator_can_stop_early_and_propagates_page_failure():
    failure = RuntimeError("page unavailable")
    fetch = AsyncMock(
        side_effect=[types.DomainListOutput(has_more=True, next_cursor="next"), failure]
    )
    iterator = async_pages(fetch)
    await anext(iterator)
    assert fetch.await_count == 1
    await iterator.aclose()
    assert fetch.await_count == 1
    fetch.reset_mock(side_effect=True)
    fetch.side_effect = [types.DomainListOutput(has_more=True, next_cursor="next"), failure]
    iterator = async_pages(fetch)
    await anext(iterator)
    with pytest.raises(RuntimeError, match="page unavailable"):
        await anext(iterator)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@respx.mock
async def test_domain_iteration_crosses_the_http_boundary(async_mode):
    route = respx.get("https://api.example.com/v1/domains")
    route.side_effect = [
        httpx.Response(
            200,
            json={
                "data": [{"domain": "first.example", "status": "active"}],
                "meta": {"has_more": True, "next_cursor": "next"},
            },
        ),
        httpx.Response(
            200,
            json={
                "data": [{"domain": "last.example", "status": "active"}],
                "meta": {"has_more": False},
            },
        ),
    ]
    if async_mode:
        async with AsyncQURLClient(
            api_key="test-key", base_url="https://api.example.com"
        ) as client:
            result = [domain async for domain in client.list_domains_all(page_size=2)]
    else:
        with QURLClient(api_key="test-key", base_url="https://api.example.com") as client:
            result = list(client.list_domains_all(page_size=2))
    assert [domain.domain for domain in result] == ["first.example", "last.example"]
    assert dict(route.calls[0].request.url.params) == {"limit": "2"}
    assert dict(route.calls[1].request.url.params) == {"limit": "2", "cursor": "next"}
