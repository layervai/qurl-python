"""Shared HTTP client fixtures for SDK contract tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

import pytest

from layerv_qurl import (
    AsyncQURLClient,
    QURLClient,
)
from tests.client_support import BASE_URL


@pytest.fixture
def client() -> QURLClient:
    return QURLClient(api_key="lv_live_test", base_url=BASE_URL, max_retries=0)


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncQURLClient, None]:
    client = AsyncQURLClient(api_key="lv_live_test", base_url=BASE_URL, max_retries=0)
    yield client
    await client.close()


@pytest.fixture
def retry_client() -> QURLClient:
    return QURLClient(api_key="lv_live_test", base_url=BASE_URL, max_retries=2)
