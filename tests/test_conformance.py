"""qURL cross-language conformance package integration tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import qurl_conformance

from layerv_qurl import AsyncQURLClient, QURLClient
from layerv_qurl._utils import _ACCESS_TOKEN_RE as ACCESS_TOKEN_SHAPE_RE
from layerv_qurl._utils import _SIGNED_FRAGMENT_RE as SIGNED_FRAGMENT_SHAPE_RE

BASE_URL = "https://api.test.layerv.ai"
# Keep these static messages duplicated here deliberately: an accidental
# production wording change should fail this non-echoing credential test.
SIGNED_FRAGMENT_ERROR = (
    "qurl_link: this is a signed qURL link (offline fragment format), "
    "which cannot be opened through the API resolver \u2014 enter_portal "
    "only accepts platform links (https://qurl.link/#at_...) or bare "
    "access tokens"
)
NO_ACCESS_TOKEN_ERROR = (
    "qurl_link: no access token found \u2014 pass a platform qURL link "
    "(https://qurl.link/#at_...) or a bare access token (at_...)"
)
RESOLVER_REACHED_MESSAGE = "qurl-conformance fragment reached API resolver"


def _qurl_conformance_fragment_cases() -> list[Any]:
    """Return the SDK-consumed qv2 fragment inputs from qurl-conformance."""
    conformance = qurl_conformance.qv2_vectors()
    assert conformance["artifact"] == "qurl-v2-conformance-vectors"
    assert conformance["schema_version"] == 1
    vectors = conformance["classes"]["fragment"]["vectors"]
    assert vectors, "fragment class must not be empty"
    assert any(SIGNED_FRAGMENT_SHAPE_RE.match(vector["fragment"]) for vector in vectors), (
        "expected at least one signed-shaped qv2 fragment vector"
    )
    cases = []
    for vector in vectors:
        name = vector["name"]
        fragment = vector["fragment"]
        if ACCESS_TOKEN_SHAPE_RE.match(fragment):
            raise AssertionError(
                f"{name}: fragment vectors must be offline qv2 fragments, not platform tokens"
            )
        cases.append(pytest.param(fragment, id=name))
    return cases


FRAGMENT_CASES = _qurl_conformance_fragment_cases()


def _assert_qv2_fragment_rejected(error: ValueError, fragment: str) -> None:
    """Assert local fail-closed behavior, not full offline qv2 verification."""
    expected_message = (
        SIGNED_FRAGMENT_ERROR
        if SIGNED_FRAGMENT_SHAPE_RE.match(fragment)
        else NO_ACCESS_TOKEN_ERROR
    )
    assert str(error) == expected_message


@pytest.mark.parametrize("fragment", FRAGMENT_CASES)
def test_enter_portal_rejects_qurl_conformance_fragments_before_api_call(
    fragment: str,
) -> None:
    """Shared qv2 fragments are offline links, so Python must not resolve them."""
    with (
        QURLClient(api_key="lv_live_test", base_url=BASE_URL) as client,
        patch.object(
            client,
            "resolve",
            side_effect=AssertionError(RESOLVER_REACHED_MESSAGE),
        ),
        pytest.raises(ValueError) as exc_info,
    ):
        client.enter_portal(f"https://qurl.link/#{fragment}")

    _assert_qv2_fragment_rejected(exc_info.value, fragment)


@pytest.mark.parametrize("fragment", FRAGMENT_CASES)
async def test_async_enter_portal_rejects_qurl_conformance_fragments_before_api_call(
    fragment: str,
) -> None:
    """Async portal entry shares the same fail-closed qv2 fragment guard."""
    async with AsyncQURLClient(
        api_key="lv_live_test", base_url=BASE_URL
    ) as client:
        with (
            patch.object(
                client,
                "resolve",
                new_callable=AsyncMock,
                side_effect=AssertionError(RESOLVER_REACHED_MESSAGE),
            ),
            pytest.raises(ValueError) as exc_info,
        ):
            await client.enter_portal(f"https://qurl.link/#{fragment}")

    _assert_qv2_fragment_rejected(exc_info.value, fragment)
