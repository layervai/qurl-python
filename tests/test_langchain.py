"""Tests for the LangChain tool integration.

Requires the ``langchain`` extra: ``pip install layerv-qurl[langchain]``.
All tests are skipped when ``langchain-core`` is not installed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from layerv_qurl.langchain import (
    _HAS_LANGCHAIN,
    CreateQURLTool,
    DeleteQURLTool,
    ListQURLsTool,
    QURLToolkit,
    ResolveQURLTool,
)
from layerv_qurl.types import (
    QURL,
    AccessGrant,
    CreateOutput,
    ListOutput,
    ResolveOutput,
)

pytestmark = pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain-core not installed")


def _mock_client() -> MagicMock:
    return MagicMock()


def test_create_qurl_tool() -> None:
    client = _mock_client()
    expires_at = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
    client.create.return_value = CreateOutput(
        resource_id="r_abc123def45",
        qurl_link="https://qurl.link/#at_test",
        qurl_site="https://r_abc123def45.qurl.site",
        expires_at=expires_at,
    )

    tool = CreateQURLTool(client=client)
    result = tool._run(target_url="https://example.com", expires_in="24h")

    assert result == (
        "Created qURL r_abc123def45\n"
        "Link: https://qurl.link/#at_test\n"
        "Site: https://r_abc123def45.qurl.site\n"
        f"Expires: {expires_at}"
    )
    client.create.assert_called_once_with(
        target_url="https://example.com",
        expires_in="24h",
        label=None,
    )


def test_create_qurl_tool_no_expiration() -> None:
    client = _mock_client()
    client.create.return_value = CreateOutput(
        resource_id="r_abc123def45",
        qurl_link="https://qurl.link/#at_test",
        qurl_site="https://r_abc123def45.qurl.site",
        expires_at=None,
    )

    tool = CreateQURLTool(client=client)
    result = tool._run(target_url="https://example.com", expires_in="24h")

    assert result == (
        "Created qURL r_abc123def45\n"
        "Link: https://qurl.link/#at_test\n"
        "Site: https://r_abc123def45.qurl.site\n"
        "Expires: N/A"
    )
    client.create.assert_called_once_with(
        target_url="https://example.com",
        expires_in="24h",
        label=None,
    )


def test_resolve_qurl_tool() -> None:
    client = _mock_client()
    client.resolve.return_value = ResolveOutput(
        target_url="https://api.example.com/data",
        resource_id="r_abc123def45",
        access_grant=AccessGrant(
            expires_in=305,
            granted_at=datetime(2026, 3, 10, 15, 30, 0, tzinfo=timezone.utc),
            src_ip="203.0.113.42",
        ),
    )

    tool = ResolveQURLTool(client=client)
    result = tool._run(access_token="at_k8xqp9h2sj9lx7r4a")

    assert result == (
        "Resolved: https://api.example.com/data\n"
        "Resource: r_abc123def45\n"
        "Access expires in: 305s\n"
        "Granted to IP: 203.0.113.42"
    )
    # resolve() now takes a plain string
    client.resolve.assert_called_once_with("at_k8xqp9h2sj9lx7r4a")


def test_resolve_qurl_tool_no_grant() -> None:
    client = _mock_client()
    client.resolve.return_value = ResolveOutput(
        target_url=None,
        resource_id="r_abc123def45",
        access_grant=None,
    )

    tool = ResolveQURLTool(client=client)
    result = tool._run(access_token="at_k8xqp9h2sj9lx7r4a")

    assert result == "Resolved: <redacted>\nResource: r_abc123def45"


def test_list_qurls_tool() -> None:
    client = _mock_client()
    expires_at = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
    client.list.return_value = ListOutput(
        qurls=[
            QURL(
                resource_id="r_abc123def45",
                target_url="https://example.com",
                status="active",
                created_at=datetime(2026, 3, 10, 10, 0, 0, tzinfo=timezone.utc),
                expires_at=expires_at,
            ),
            QURL(
                resource_id="r_tunnel12345",
                target_url=None,
                status="active",
            )
        ],
        has_more=False,
    )

    tool = ListQURLsTool(client=client)
    result = tool._run(status="active", limit=10)

    assert result == (
        "- r_abc123def45: https://example.com [active] "
        f"expires={expires_at}\n"
        "- r_tunnel12345: <redacted> [active] expires=N/A"
    )
    client.list.assert_called_once_with(status="active", limit=10)


def test_list_qurls_tool_empty() -> None:
    client = _mock_client()
    client.list.return_value = ListOutput(qurls=[], has_more=False)

    tool = ListQURLsTool(client=client)
    result = tool._run()

    assert result == "No qURLs found."


def test_delete_qurl_tool() -> None:
    client = _mock_client()
    client.delete.return_value = None

    tool = DeleteQURLTool(client=client)
    result = tool._run(resource_id="r_abc123def45")

    assert result == "qURL r_abc123def45 has been revoked."
    client.delete.assert_called_once_with("r_abc123def45")


def test_toolkit_returns_all_tools() -> None:
    client = _mock_client()
    toolkit = QURLToolkit(client=client)
    tools = toolkit.get_tools()

    assert len(tools) == 4
    names = {t.name for t in tools}
    assert names == {"create_qurl", "resolve_qurl", "list_qurls", "delete_qurl"}
