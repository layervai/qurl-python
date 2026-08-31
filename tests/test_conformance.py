"""qURL cross-language conformance package integration tests."""

from __future__ import annotations

from importlib.metadata import version as package_version
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
# qv2 artifact schema revisions whose `fragment` class this SDK understands.
# An allowlist rather than `== N` because the qurl-conformance range in
# pyproject.toml spans both revisions, so either can legitimately resolve:
# v2 is purely additive over v1 (it adds a `transport_contract` key and a
# `transport` class) and leaves the `fragment` class byte-identical.
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})


def _assert_supported_schema_version(conformance: dict[str, Any]) -> None:
    """Reject an artifact revision whose shape nobody has re-verified yet."""
    schema_version = conformance["schema_version"]
    assert schema_version in SUPPORTED_SCHEMA_VERSIONS, (
        f"unrecognized qv2 artifact schema_version {schema_version} "
        f"(qurl-conformance {package_version('qurl-conformance')}, supported: "
        f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}): re-verify the `fragment` class "
        f"shape, then add it to SUPPORTED_SCHEMA_VERSIONS"
    )


def _qurl_conformance_fragment_cases() -> list[Any]:
    """Return the SDK-consumed qv2 fragment inputs from qurl-conformance."""
    conformance = qurl_conformance.qv2_vectors()
    assert conformance["artifact"] == "qurl-v2-conformance-vectors"
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


# The version gate is a test, not a collection-time assert: an unrecognized
# revision must fail one red test, not abort collection and take all 275
# unrelated tests down with it (which is exactly what schema 2 did on main).
def test_qv2_artifact_schema_version_is_supported() -> None:
    """The resolved artifact revision is one whose shape we have checked."""
    _assert_supported_schema_version(qurl_conformance.qv2_vectors())


def test_unsupported_qv2_schema_version_is_rejected() -> None:
    """The tripwire must actually fire — a silent pass would hide a reshape."""
    with pytest.raises(AssertionError, match=r"unrecognized qv2 artifact schema_version 3"):
        _assert_supported_schema_version({"schema_version": 3})


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
