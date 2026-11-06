"""The public resource name is sent by every call that can create a resource."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx

from layerv_qurl import AsyncQURLClient, QURLClient
from layerv_qurl.errors import ValidationError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

BASE_URL = "https://api.test.layerv.ai"
TARGET = "https://internal.example.com/dashboard"
_RESOURCE: dict[str, Any] = {"resource_id": "r_abc123def45", "status": "active"}
_PORTAL: dict[str, Any] = {
    "resource_id": "r_abc123def45",
    "qurl_link": "https://qurl.link/#at_portal1",
    "qurl_site": "https://r_abc123def45.qurl.site",
    "qurl_id": "q_portal1",
}
_BATCH: dict[str, Any] = {
    "succeeded": 1,
    "failed": 0,
    "results": [{"index": 0, "success": True, **_PORTAL}],
}


@pytest.fixture
def client() -> QURLClient:
    return QURLClient(api_key="lv_live_test", base_url=BASE_URL, max_retries=0)


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncQURLClient, None]:
    client = AsyncQURLClient(api_key="lv_live_test", base_url=BASE_URL, max_retries=0)
    yield client
    await client.close()


_CALLS: list[tuple[str, dict[str, Any], Callable[[Any], Any]]] = [
    ("/v1/resources", _RESOURCE, lambda c: c.protect_url(TARGET, name="Team dashboard")),
    (
        "/v1/resources",
        _RESOURCE,
        lambda c: c.create_resource(target_url=TARGET, name="Team dashboard"),
    ),
    (
        "/v1/resources",
        _RESOURCE,
        lambda c: c.create_resource(
            resource_type="tunnel", slug="prod", find_or_create=True, name="Team dashboard"
        ),
    ),
    ("/v1/qurls", _PORTAL, lambda c: c.create_portal_for_url(TARGET, name="Team dashboard")),
    ("/v1/qurls", _PORTAL, lambda c: c.create(TARGET, name="Team dashboard")),
]


def _sent_name(route: respx.Route) -> Any:
    body = json.loads(route.calls[0].request.content)
    if "items" in body:
        body = body["items"][0]
    return body.get("name")


@respx.mock
@pytest.mark.parametrize(("path", "data", "call"), _CALLS)
def test_create_calls_send_name(
    client: QURLClient, path: str, data: dict[str, Any], call: Callable[[Any], Any]
) -> None:
    route = respx.post(f"{BASE_URL}{path}").mock(
        return_value=httpx.Response(201, json={"data": data})
    )
    call(client)
    assert _sent_name(route) == "Team dashboard"


@respx.mock
@pytest.mark.parametrize(("path", "data", "call"), _CALLS)
async def test_async_create_calls_send_name(
    async_client: AsyncQURLClient,
    path: str,
    data: dict[str, Any],
    call: Callable[[Any], Any],
) -> None:
    route = respx.post(f"{BASE_URL}{path}").mock(
        return_value=httpx.Response(201, json={"data": data})
    )
    await call(async_client)
    assert _sent_name(route) == "Team dashboard"


@respx.mock
def test_batch_create_sends_item_name(client: QURLClient) -> None:
    route = respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(201, json={"data": _BATCH})
    )
    client.batch_create([{"target_url": TARGET, "name": "Team dashboard"}])
    assert _sent_name(route) == "Team dashboard"


@respx.mock
def test_empty_name_rejected_before_sending(client: QURLClient) -> None:
    route = respx.post(url__startswith=BASE_URL)
    attempts: list[Callable[[], Any]] = [
        lambda: client.protect_url(TARGET, name=" "),
        lambda: client.protect_url(TARGET, name=" Team"),
        lambda: client.protect_url(TARGET, name="Team\ndashboard"),
        lambda: client.protect_url(TARGET, name="Team\u2028dashboard"),
        lambda: client.protect_url(TARGET, name="Team\u202edashboard"),
        lambda: client.protect_url(TARGET, name=123),  # type: ignore[arg-type]
        lambda: client.create_resource(
            resource_type="tunnel", slug="prod", find_or_create=True, name=""
        ),
        lambda: client.create_resource(target_url=TARGET, name=""),
        lambda: client.create_portal_for_url(TARGET, name=""),
        lambda: client.create(TARGET, name=" "),
        lambda: client.batch_create([{"target_url": TARGET, "name": ""}]),
    ]
    for attempt in attempts:
        with pytest.raises(ValueError, match="name"):
            attempt()
    assert not route.called


@respx.mock
def test_name_required_refusal_is_a_validation_error(client: QURLClient) -> None:
    respx.post(f"{BASE_URL}/v1/resources").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "status": 400,
                    "code": "name_required",
                    "title": "Bad Request",
                    "detail": "name is required: the public name recipients see "
                    "for this resource",
                }
            },
        )
    )
    with pytest.raises(ValidationError) as excinfo:
        client.protect_url(TARGET)
    assert excinfo.value.status == 400
    assert excinfo.value.code == "name_required"


@respx.mock
async def test_async_batch_create_and_empty_name(async_client: AsyncQURLClient) -> None:
    route = respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(201, json={"data": _BATCH})
    )
    await async_client.batch_create([{"target_url": TARGET, "name": "Team dashboard"}])
    assert _sent_name(route) == "Team dashboard"
    with pytest.raises(ValueError, match="name"):
        await async_client.protect_url(TARGET, name="")
    assert route.call_count == 1


@respx.mock
@pytest.mark.parametrize(("path", "data", "call"), _CALLS)
def test_name_left_out_when_not_given(
    client: QURLClient, path: str, data: dict[str, Any], call: Callable[[Any], Any]
) -> None:
    route = respx.post(f"{BASE_URL}{path}").mock(
        return_value=httpx.Response(201, json={"data": data})
    )

    class Unnamed:
        """Proxy that drops the name keyword from the shared call table."""

        def __getattr__(self, attr: str) -> Any:
            method = getattr(client, attr)
            return lambda *a, **kw: method(*a, **{k: v for k, v in kw.items() if k != "name"})

    call(Unnamed())
    assert "name" not in json.loads(route.calls[0].request.content)


@respx.mock
async def test_async_empty_name_rejected_before_sending(
    async_client: AsyncQURLClient,
) -> None:
    route = respx.post(url__startswith=BASE_URL)
    for name in ("", " Team", "Team\ndashboard"):
        for attempt in (
            lambda n=name: async_client.protect_url(TARGET, name=n),
            lambda n=name: async_client.create_resource(target_url=TARGET, name=n),
            lambda n=name: async_client.create_portal_for_url(TARGET, name=n),
            lambda n=name: async_client.create(TARGET, name=n),
            lambda n=name: async_client.batch_create([{"target_url": TARGET, "name": n}]),
        ):
            with pytest.raises(ValueError, match="name"):
                await attempt()
    assert not route.called


def test_find_or_create_false_with_only_name_is_refused(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="find_or_create"):
        client.create_resource(find_or_create=False, name="Team dashboard")


@respx.mock
@pytest.mark.parametrize("name", ["Café ダッシュボード", "Ünïcode — dashboard ✓"])
def test_non_ascii_name_is_accepted(client: QURLClient, name: str) -> None:
    route = respx.post(f"{BASE_URL}/v1/resources").mock(
        return_value=httpx.Response(201, json={"data": _RESOURCE})
    )
    client.protect_url(TARGET, name=name)
    assert _sent_name(route) == name


def test_zwj_emoji_name_is_refused_like_the_service(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="name"):
        client.protect_url(TARGET, name="\U0001f469\u200d\U0001f4bb Dev portal")


@respx.mock
@pytest.mark.parametrize(("path", "data", "call"), _CALLS)
async def test_async_name_left_out_when_not_given(
    async_client: AsyncQURLClient,
    path: str,
    data: dict[str, Any],
    call: Callable[[Any], Any],
) -> None:
    route = respx.post(f"{BASE_URL}{path}").mock(
        return_value=httpx.Response(201, json={"data": data})
    )

    class Unnamed:
        def __getattr__(self, attr: str) -> Any:
            method = getattr(async_client, attr)
            return lambda *a, **kw: method(*a, **{k: v for k, v in kw.items() if k != "name"})

    await call(Unnamed())
    assert "name" not in json.loads(route.calls[0].request.content)


@respx.mock
async def test_async_name_required_refusal_is_a_validation_error(
    async_client: AsyncQURLClient,
) -> None:
    respx.post(f"{BASE_URL}/v1/resources").mock(
        return_value=httpx.Response(
            400, json={"error": {"status": 400, "code": "name_required", "title": "Bad Request"}}
        )
    )
    with pytest.raises(ValidationError) as excinfo:
        await async_client.protect_url(TARGET)
    assert excinfo.value.code == "name_required"
