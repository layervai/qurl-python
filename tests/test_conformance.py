"""qURL cross-language conformance package integration tests."""

from __future__ import annotations

import sys
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
    schema_version = conformance.get("schema_version")
    assert schema_version in SUPPORTED_SCHEMA_VERSIONS, (
        f"unrecognized qv2 artifact schema_version {schema_version} "
        f"(qurl-conformance {package_version('qurl-conformance')}, supported: "
        f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}): re-verify the `fragment` class "
        f"shape, then add it to SUPPORTED_SCHEMA_VERSIONS"
    )


def _build_fragment_cases() -> list[Any]:
    """Extract the SDK-consumed qv2 fragment inputs; raises on a reshaped artifact."""
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


def _qurl_conformance_fragment_cases() -> list[Any]:
    """Build the parametrize cases without failing collection on a reshaped artifact.

    The version gate above only covers a *version-number* change. A reshaped
    artifact — a renamed class, a dropped key, an unexpected vector shape —
    raises while extracting cases, and this runs at import time, so pytest
    would interrupt the session and report nothing for every other test in
    the suite (the schema-2 outage on main, in a different disguise). Carry
    the error into the parametrization and surface it inside the test instead.

    Note the bound: the module-level `import qurl_conformance` is still
    outside this guard, so a package that renames the top-level module or
    raises at import reproduces the original outage. `importorskip` would
    only trade it for a silent skip, which the paragraph below rejects.

    Deliberately not `return []` on failure: an empty parametrize list makes
    these tests *skip*, and a silently skipped fail-closed test is worse than
    the outage it replaces.
    """
    try:
        return _build_fragment_cases()
    # Intentionally broad — anything that stops us building cases must become
    # a red test rather than a collection abort. (BLE isn't in ruff's select
    # list, so this comment is documentation, not suppression.)
    except Exception as exc:
        return [pytest.param(exc, id="qurl_conformance_artifact_unusable")]


FRAGMENT_CASES = _qurl_conformance_fragment_cases()


def _fragment_or_raise(fragment: str | BaseException) -> str:
    """Surface a collection-time artifact failure as this test's failure."""
    if isinstance(fragment, BaseException):
        # A fresh exception rather than `raise fragment`: FRAGMENT_CASES is
        # module-level and holds one instance, so re-raising it would mutate
        # `__traceback__` in place and bleed the sync test's frames into the
        # async test's report.
        raise AssertionError(f"{type(fragment).__name__}: {fragment}") from fragment
    return fragment


# The version gate is a test, not a collection-time assert: an unrecognized
# revision must fail one red test, not abort collection and take every other
# test in the suite down with it (exactly what schema 2 did on main).
def test_qv2_artifact_schema_version_is_supported() -> None:
    """The resolved artifact revision is one whose shape we have checked."""
    _assert_supported_schema_version(qurl_conformance.qv2_vectors())


def test_supported_schema_versions_covers_the_declared_dependency_range() -> None:
    """Dropping a revision from the allowlist must be a deliberate edit.

    This is a change-detector on purpose. The parametrized test below walks
    the set itself, so it narrows along with it: cutting `1` leaves CI fully
    green (verified — 282 passed) while breaking anyone who resolves the
    lower end of `qurl-conformance>=0.1.2,<0.13`, which is an old lockfile,
    a `--resolution lowest` job, or a constrained downstream install. Only a
    pinned expectation catches that, and it forces the range in
    pyproject.toml to be re-read at the same time.
    """
    assert sorted(SUPPORTED_SCHEMA_VERSIONS) == [1, 2], (
        "SUPPORTED_SCHEMA_VERSIONS changed: qurl-conformance <=0.11.0 ships "
        "schema 1 and 0.12.x ships schema 2, so both must stay allowlisted "
        "while the range in pyproject.toml still admits both"
    )


@pytest.mark.parametrize("schema_version", sorted(SUPPORTED_SCHEMA_VERSIONS))
def test_every_supported_schema_version_is_accepted(schema_version: int) -> None:
    """Each allowlisted revision must pass — including ones CI never resolves.

    Only whatever `<0.13` resolves (schema 2) gets live coverage, so without
    this the `1` branch is never exercised at all.
    """
    _assert_supported_schema_version({"schema_version": schema_version})


def test_unsupported_qv2_schema_version_is_rejected() -> None:
    """The tripwire must actually fire — a silent pass would hide a reshape."""
    with pytest.raises(AssertionError, match=r"unrecognized qv2 artifact schema_version 3"):
        _assert_supported_schema_version({"schema_version": 3})


def test_missing_qv2_schema_version_is_rejected_with_the_actionable_message() -> None:
    """A dropped key must reach the same guidance, not a bare KeyError."""
    with pytest.raises(AssertionError, match=r"schema_version None"):
        _assert_supported_schema_version({})


def test_fragment_or_raise_surfaces_a_carried_artifact_failure() -> None:
    """A carried failure must fail the test, not flow through as fragment data."""
    assert _fragment_or_raise("qv2.abc.def") == "qv2.abc.def"
    with pytest.raises(AssertionError, match=r"ValueError: boom") as exc_info:
        _fragment_or_raise(ValueError("boom"))
    assert isinstance(exc_info.value.__cause__, ValueError), "original cause must be chained"


def test_unusable_artifact_yields_a_red_case_not_an_empty_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback must never degrade to `[]`, which would silently skip."""

    def _boom() -> list[Any]:
        raise AssertionError("reshaped")

    monkeypatch.setattr(sys.modules[__name__], "_build_fragment_cases", _boom)
    cases = _qurl_conformance_fragment_cases()
    assert len(cases) == 1, "an unusable artifact must still produce one red test"
    with pytest.raises(AssertionError, match="reshaped"):
        _fragment_or_raise(cases[0].values[0])


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
    fragment: str | BaseException,
) -> None:
    """Shared qv2 fragments are offline links, so Python must not resolve them."""
    fragment = _fragment_or_raise(fragment)
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
    fragment: str | BaseException,
) -> None:
    """Async portal entry shares the same fail-closed qv2 fragment guard."""
    fragment = _fragment_or_raise(fragment)
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
