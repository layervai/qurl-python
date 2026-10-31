"""Batch creation, validation, partial results, and response parsing tests."""

from __future__ import annotations

import json
from datetime import datetime

import httpx
import pytest
import respx

from layerv_qurl import (
    AsyncQURLClient,
    QURLClient,
    QURLError,
)
from layerv_qurl.errors import (
    AuthenticationError,
)
from layerv_qurl.types import (
    AccessPolicy,
    AIAgentPolicy,
    BatchCreateItem,
)
from tests.client_support import BASE_URL

# --- Batch create ---


@respx.mock
def test_batch_create_all_succeed(client: QURLClient) -> None:
    """batch_create() parses a fully successful batch response."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "succeeded": 2,
                    "failed": 0,
                    "results": [
                        {
                            "index": 0,
                            "success": True,
                            "resource_id": "r_batch1",
                            "qurl_link": "https://qurl.link/#at_b1",
                            "qurl_site": "https://r_batch1.qurl.site",
                            "expires_at": "2026-04-01T00:00:00Z",
                        },
                        {
                            "index": 1,
                            "success": True,
                            "resource_id": "r_batch2",
                            "qurl_link": "https://qurl.link/#at_b2",
                            "qurl_site": "https://r_batch2.qurl.site",
                        },
                    ],
                },
            },
        )
    )

    result = client.batch_create(
        [
            {"target_url": "https://a.com", "expires_in": "24h"},
            {"target_url": "https://b.com"},
        ]
    )
    assert result.succeeded == 2
    assert result.failed == 0
    assert len(result.results) == 2
    assert result.results[0].resource_id == "r_batch1"
    assert result.results[0].success is True
    assert isinstance(result.results[0].expires_at, datetime)
    assert result.results[1].resource_id == "r_batch2"
    assert result.results[1].expires_at is None


@respx.mock
def test_batch_create_207_multi_status(client: QURLClient) -> None:
    """HTTP 207 Multi-Status routes through the success path (< 400).

    The success path in ``_raw_request`` is gated by
    ``response.status_code < 400``, so 207 flows through naturally like
    200/201. This test locks in the status routing against a future
    refactor that might accidentally narrow the range (e.g. ``== 200``
    or ``< 300``). Without this, a partial-success 207 response
    could be misclassified as an error.
    """
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            207,
            json={
                "data": {
                    "succeeded": 1,
                    "failed": 1,
                    "results": [
                        {
                            "index": 0,
                            "success": True,
                            "resource_id": "r_multi",
                            "qurl_link": "https://qurl.link/#at_multi",
                            "qurl_site": "https://r_multi.qurl.site",
                        },
                        {
                            "index": 1,
                            "success": False,
                            "error": {
                                "code": "validation_error",
                                "message": "items[1]: target_url must be HTTPS",
                            },
                        },
                    ],
                },
            },
        )
    )

    result = client.batch_create(
        [
            {"target_url": "https://a.com"},
            {"target_url": "https://b.com"},
        ]
    )
    assert result.succeeded == 1
    assert result.failed == 1
    assert result.results[0].success is True
    assert result.results[0].resource_id == "r_multi"
    assert result.results[1].success is False


@respx.mock
async def test_async_batch_create_207_multi_status(
    async_client: AsyncQURLClient,
) -> None:
    """Async mirror: HTTP 207 Multi-Status routes through success path."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            207,
            json={
                "data": {
                    "succeeded": 1,
                    "failed": 0,
                    "results": [
                        {
                            "index": 0,
                            "success": True,
                            "resource_id": "r_async207",
                            "qurl_link": "https://qurl.link/#at_a207",
                            "qurl_site": "https://r_async207.qurl.site",
                        },
                    ],
                },
            },
        )
    )

    result = await async_client.batch_create([{"target_url": "https://a.com"}])
    assert result.succeeded == 1
    assert result.results[0].resource_id == "r_async207"


@respx.mock
def test_batch_create_partial_failure(client: QURLClient) -> None:
    """batch_create() correctly parses partial failures."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "succeeded": 1,
                    "failed": 1,
                    "results": [
                        {
                            "index": 0,
                            "success": True,
                            "resource_id": "r_ok",
                            "qurl_link": "https://qurl.link/#at_ok",
                            "qurl_site": "https://r_ok.qurl.site",
                        },
                        {
                            "index": 1,
                            "success": False,
                            "error": {
                                "code": "validation_error",
                                "message": "Invalid target_url",
                            },
                        },
                    ],
                },
            },
        )
    )

    # Both URLs must pass client-side validation (syntactically valid
    # https://) — the mock returns a partial-failure payload regardless
    # of what's in the request body, so we're exercising the parser.
    result = client.batch_create(
        [
            {"target_url": "https://good.example.com"},
            {"target_url": "https://bad.example.com"},
        ]
    )
    assert result.succeeded == 1
    assert result.failed == 1
    assert result.results[0].success is True
    assert result.results[1].success is False
    assert result.results[1].error is not None
    assert result.results[1].error.code == "validation_error"
    assert result.results[1].error.message == "Invalid target_url"


# --- Async batch create ---


@respx.mock
@pytest.mark.asyncio
async def test_async_batch_create(async_client: AsyncQURLClient) -> None:
    """Async batch_create() works correctly."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "succeeded": 1,
                    "failed": 0,
                    "results": [
                        {
                            "index": 0,
                            "success": True,
                            "resource_id": "r_async_batch",
                            "qurl_link": "https://qurl.link/#at_ab",
                            "qurl_site": "https://r_async_batch.qurl.site",
                        },
                    ],
                },
            },
        )
    )

    result = await async_client.batch_create(
        [
            {"target_url": "https://async.com"},
        ]
    )
    assert result.succeeded == 1
    assert result.results[0].resource_id == "r_async_batch"


# --- batch_create validation ---


def test_batch_create_empty_raises(client: QURLClient) -> None:
    with pytest.raises(ValueError, match="requires at least 1 item"):
        client.batch_create([])


def test_batch_create_over_100_raises(client: QURLClient) -> None:
    items: list[BatchCreateItem] = [{"target_url": f"https://{i}.com"} for i in range(101)]
    with pytest.raises(ValueError, match="at most 100"):
        client.batch_create(items)


# ---- batch_create per-item validation & async validators -------------------


def test_batch_create_rejects_per_item_violation_with_index(client: QURLClient) -> None:
    """Per-item validation failures include the offending index."""
    with pytest.raises(ValueError, match=r"items\[1\].*max_sessions"):
        client.batch_create(
            [
                {"target_url": "https://a.example.com"},
                {"target_url": "https://b.example.com", "max_sessions": 9999},
            ]
        )


def test_batch_create_rejects_items_missing_target_url(client: QURLClient) -> None:
    with pytest.raises(ValueError, match=r"items\[0\].*target_url"):
        # Deliberately passing an item missing the required `target_url`
        # field to exercise the runtime validation. The type: ignore is
        # expected because BatchCreateItem enforces `target_url` at the
        # type level — this test guards against untyped callers
        # (e.g. Python scripts without strict type-checking) accidentally
        # sending incomplete items.
        client.batch_create([{"label": "no url"}])  # type: ignore[typeddict-item]


@pytest.mark.asyncio
async def test_async_batch_create_empty_raises(async_client: AsyncQURLClient) -> None:
    with pytest.raises(ValueError, match="requires at least 1 item"):
        await async_client.batch_create([])


@pytest.mark.asyncio
async def test_async_batch_create_over_100_raises(async_client: AsyncQURLClient) -> None:
    items: list[BatchCreateItem] = [{"target_url": f"https://{i}.com"} for i in range(101)]
    with pytest.raises(ValueError, match="at most 100"):
        await async_client.batch_create(items)


# ---- batch_create HTTP 400 passthrough -------------------------------------


@respx.mock
def test_batch_create_passes_through_400_with_per_item_errors(
    client: QURLClient,
) -> None:
    """HTTP 400 on batch/ carries a BatchCreateOutput body with per-item errors.

    The SDK whitelists 400 and surfaces the structured body instead of
    raising a generic ValidationError — matching the qurl-mcp and
    qurl-typescript behavior on this endpoint.
    """
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            400,
            json={
                "data": {
                    "succeeded": 0,
                    "failed": 2,
                    "results": [
                        {
                            "index": 0,
                            "success": False,
                            "error": {
                                "code": "validation_error",
                                "message": "items[0]: target_url must be HTTPS",
                            },
                        },
                        {
                            "index": 1,
                            "success": False,
                            "error": {
                                "code": "validation_error",
                                "message": "items[1]: target_url must be HTTPS",
                            },
                        },
                    ],
                },
                "meta": {"request_id": "req_allfail"},
            },
        )
    )
    result = client.batch_create(
        [
            {"target_url": "https://ok1.example.com"},
            {"target_url": "https://ok2.example.com"},
        ]
    )
    assert result.failed == 2
    assert result.succeeded == 0
    assert len(result.results) == 2
    assert result.results[0].success is False
    assert result.results[0].error is not None
    assert result.results[0].error.code == "validation_error"


@respx.mock
@pytest.mark.asyncio
async def test_async_batch_create_passes_through_400_with_per_item_errors(
    async_client: AsyncQURLClient,
) -> None:
    """Async mirror of `test_batch_create_passes_through_400_with_per_item_errors`.
    The sync/async parity goal requires the async client to also
    surface the structured 400 body as a normal return value rather
    than raising — reviewer flagged this as a coverage gap.
    """
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            400,
            json={
                "data": {
                    "succeeded": 0,
                    "failed": 2,
                    "results": [
                        {
                            "index": 0,
                            "success": False,
                            "error": {
                                "code": "validation_error",
                                "message": "items[0]: target_url must be HTTPS",
                            },
                        },
                        {
                            "index": 1,
                            "success": False,
                            "error": {
                                "code": "validation_error",
                                "message": "items[1]: target_url must be HTTPS",
                            },
                        },
                    ],
                },
                "meta": {"request_id": "req_allfail_async"},
            },
        )
    )
    result = await async_client.batch_create(
        [
            {"target_url": "https://ok1.example.com"},
            {"target_url": "https://ok2.example.com"},
        ]
    )
    assert result.failed == 2
    assert result.succeeded == 0
    assert len(result.results) == 2
    assert result.results[0].success is False
    assert result.results[0].error is not None
    assert result.results[0].error.code == "validation_error"
    assert result.results[1].success is False
    assert result.results[1].error is not None
    assert result.results[1].error.code == "validation_error"


@respx.mock
def test_batch_create_still_raises_on_401(client: QURLClient) -> None:
    """Non-400 error statuses still raise — the 400 passthrough is surgical."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            401,
            json={
                "error": {
                    "type": "https://api.qurl.link/problems/unauthorized",
                    "title": "Unauthorized",
                    "status": 401,
                    "code": "unauthorized",
                    "detail": "Invalid API key",
                },
            },
        )
    )
    with pytest.raises(AuthenticationError):
        client.batch_create([{"target_url": "https://example.com"}])


# ---- batch_create response shape guard (defense-in-depth) -----------------


@respx.mock
def test_batch_create_rejects_unexpected_400_body_shape(client: QURLClient) -> None:
    """The 400 passthrough trusts the BatchCreateOutput shape. If the API
    ever returns 400 with a different body (e.g. a plain error envelope
    or a proxy error), the SDK must raise a clear error instead of
    silently producing `(succeeded=0, failed=0, results=[])`. Defense
    in depth — matches qurl-typescript and qurl-mcp.
    """
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            400,
            json={"data": {"unexpected": "not a batch envelope"}},
        )
    )
    with pytest.raises(QURLError, match="Unexpected response shape"):
        client.batch_create([{"target_url": "https://example.com"}])


@respx.mock
def test_batch_create_rejects_400_body_with_non_boolean_success(
    client: QURLClient,
) -> None:
    """Per-entry shape guard: results[i].success must be a bool for the
    BatchItemResult discriminated-union contract to hold."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            400,
            json={
                "data": {
                    "succeeded": 0,
                    "failed": 1,
                    "results": [
                        {"index": 0, "success": "oops"},  # should be bool
                    ],
                },
            },
        )
    )
    with pytest.raises(QURLError, match="Unexpected response shape"):
        client.batch_create([{"target_url": "https://example.com"}])


@respx.mock
def test_batch_create_rejects_400_body_with_bool_counts(
    client: QURLClient,
) -> None:
    """`bool` is a subclass of `int` in Python, so a naive
    ``isinstance(..., int)`` check would silently accept
    ``"succeeded": True``. The shape guard explicitly rejects bool in
    the counts — this test locks that in against a future simplification
    that might drop the explicit bool check."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            400,
            json={
                "data": {
                    "succeeded": True,  # bool should be rejected
                    "failed": False,
                    "results": [],
                },
            },
        )
    )
    with pytest.raises(QURLError, match="Unexpected response shape"):
        client.batch_create([{"target_url": "https://example.com"}])


@respx.mock
def test_batch_create_rejects_counts_arithmetic_mismatch(
    client: QURLClient,
) -> None:
    """The shape guard asserts `succeeded + failed == len(results)`.
    A mismatch suggests a proxy or middleware mangled the response,
    or the API had a counting bug — either case warrants raising
    rather than trusting the data."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            400,
            json={
                "data": {
                    "succeeded": 5,  # claims 5 succeeded
                    "failed": 0,
                    "results": [
                        # …but only 1 entry
                        {"index": 0, "success": True, "resource_id": "r_only1"},
                    ],
                },
            },
        )
    )
    with pytest.raises(QURLError, match="Unexpected response shape"):
        client.batch_create([{"target_url": "https://example.com"}])


# ---- Async mirrors of the batch shape-guard tests -------------------------
# Sync/async parity: every sync shape-guard test above has an async twin.
# Without these, an async-specific regression (e.g. a refactor that diverged
# the two code paths) could slip past CI.


@respx.mock
@pytest.mark.asyncio
async def test_async_batch_create_rejects_unexpected_400_body_shape(
    async_client: AsyncQURLClient,
) -> None:
    """Async mirror of test_batch_create_rejects_unexpected_400_body_shape."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            400,
            json={"data": {"unexpected": "not a batch envelope"}},
        )
    )
    with pytest.raises(QURLError, match="Unexpected response shape"):
        await async_client.batch_create([{"target_url": "https://example.com"}])


@respx.mock
@pytest.mark.asyncio
async def test_async_batch_create_rejects_400_body_with_non_boolean_success(
    async_client: AsyncQURLClient,
) -> None:
    """Async mirror of test_batch_create_rejects_400_body_with_non_boolean_success."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            400,
            json={
                "data": {
                    "succeeded": 0,
                    "failed": 1,
                    "results": [
                        {"index": 0, "success": "oops"},  # should be bool
                    ],
                },
            },
        )
    )
    with pytest.raises(QURLError, match="Unexpected response shape"):
        await async_client.batch_create([{"target_url": "https://example.com"}])


@respx.mock
@pytest.mark.asyncio
async def test_async_batch_create_rejects_400_body_with_bool_counts(
    async_client: AsyncQURLClient,
) -> None:
    """Async mirror of test_batch_create_rejects_400_body_with_bool_counts."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            400,
            json={
                "data": {
                    "succeeded": True,  # bool should be rejected
                    "failed": False,
                    "results": [],
                },
            },
        )
    )
    with pytest.raises(QURLError, match="Unexpected response shape"):
        await async_client.batch_create([{"target_url": "https://example.com"}])


@respx.mock
@pytest.mark.asyncio
async def test_async_batch_create_rejects_counts_arithmetic_mismatch(
    async_client: AsyncQURLClient,
) -> None:
    """Async mirror of test_batch_create_rejects_counts_arithmetic_mismatch."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            400,
            json={
                "data": {
                    "succeeded": 5,  # claims 5 succeeded
                    "failed": 0,
                    "results": [
                        # …but only 1 entry
                        {"index": 0, "success": True, "resource_id": "r_only1"},
                    ],
                },
            },
        )
    )
    with pytest.raises(QURLError, match="Unexpected response shape"):
        await async_client.batch_create([{"target_url": "https://example.com"}])


@respx.mock
def test_batch_create_falls_through_to_parse_error_on_non_json_body(
    client: QURLClient,
) -> None:
    """Defense-in-depth: if a whitelisted 400 (or other status in
    ``allow_statuses``) comes back with non-JSON content — e.g. a proxy
    HTML error page, a CDN captive portal, a gateway plaintext error —
    the SDK must NOT raise a raw ``JSONDecodeError`` from inside
    ``response.json()``. Instead it should fall through to
    ``parse_error``, which handles non-JSON error bodies gracefully
    and returns a well-formed ``QURLError`` with the response status.

    Without this guard, a JSONDecodeError would propagate raw to the
    caller, bypassing both the batch shape guard and the standard
    error path — confusing and hard to handle. Note: this test does
    NOT assert on detail content, because `parse_error` intentionally
    echoes non-JSON response body text into the detail field (so
    callers can see plaintext gateway errors); that behavior is
    scoped to this bug fix only via the early-return path.
    """
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            400,
            text="<html><body>Bad Gateway</body></html>",
        )
    )
    # Load-bearing assertion: raises a QURLError (not raw
    # JSONDecodeError/ValueError from response.json()) with the
    # correct HTTP status. The class hierarchy dispatch in parse_error
    # maps 400 → ValidationError, and we accept any QURLError subclass
    # here since the specific class isn't the contract under test.
    with pytest.raises(QURLError) as exc_info:
        client.batch_create([{"target_url": "https://example.com"}])
    assert exc_info.value.status == 400


@respx.mock
@pytest.mark.asyncio
async def test_async_batch_create_falls_through_to_parse_error_on_non_json_body(
    async_client: AsyncQURLClient,
) -> None:
    """Async mirror of the non-JSON body fall-through test."""
    respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            400,
            text="<html><body>Bad Gateway</body></html>",
        )
    )
    with pytest.raises(QURLError) as exc_info:
        await async_client.batch_create([{"target_url": "https://example.com"}])
    assert exc_info.value.status == 400


@respx.mock
def test_batch_create_accepts_access_policy_dataclass(
    client: QURLClient,
) -> None:
    """End-to-end coverage for passing an ``AccessPolicy`` dataclass
    (rather than a plain dict) in a batch item. The bridge between
    typed-caller convenience and the serialized request body is
    ``_serialize_value`` — this test locks in that the dataclass path
    produces the same nested JSON as a plain-dict caller would."""
    route = respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "succeeded": 1,
                    "failed": 0,
                    "results": [
                        {
                            "index": 0,
                            "success": True,
                            "resource_id": "r_dc_policy",
                            "qurl_link": "https://qurl.link/#at_dc",
                            "qurl_site": "https://r_dc_policy.qurl.site",
                        },
                    ],
                },
            },
        )
    )

    policy = AccessPolicy(
        geo_denylist=["CN", "RU"],
        ai_agent_policy=AIAgentPolicy(allow_categories=["claude", "chatgpt"]),
    )
    item: BatchCreateItem = {
        "target_url": "https://example.com",
        "label": "dc-test",
        "access_policy": policy,
    }
    client.batch_create([item])

    body = json.loads(route.calls[0].request.content)
    assert body["items"][0]["target_url"] == "https://example.com"
    assert body["items"][0]["access_policy"]["geo_denylist"] == ["CN", "RU"]
    assert body["items"][0]["access_policy"]["ai_agent_policy"]["allow_categories"] == [
        "claude",
        "chatgpt",
    ]
    # None fields on AIAgentPolicy (block_all, deny_categories) must be
    # dropped by _serialize_value's dataclass rule, not preserved.
    assert "block_all" not in body["items"][0]["access_policy"]["ai_agent_policy"]
    assert "deny_categories" not in body["items"][0]["access_policy"]["ai_agent_policy"]


@respx.mock
def test_batch_create_serializes_type_and_parses_branded_domain(client: QURLClient) -> None:
    route = respx.post(f"{BASE_URL}/v1/qurls/batch").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "succeeded": 1,
                    "failed": 0,
                    "results": [
                        {
                            "index": 0,
                            "success": True,
                            "resource_id": "r_batch",
                            "qurl_link": "https://qurl.link/#at_batch",
                            "branded_domain": "files.example.com",
                            "qurl_site": "https://r_batch.qurl.site",
                        }
                    ],
                },
            },
        )
    )

    item: BatchCreateItem = {
        "type": "url",
        "target_url": "https://example.com",
        "custom_domain": "files.example.com",
    }
    result = client.batch_create([item], idempotency_key="idem-batch-1")

    body = json.loads(route.calls[0].request.content)
    assert body["items"][0]["type"] == "url"
    assert route.calls[0].request.headers["idempotency-key"] == "idem-batch-1"
    assert result.results[0].branded_domain == "files.example.com"
