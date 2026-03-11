"""Tests for the LangChain tool integration."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from layerv_qurl.langchain import (
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


def _mock_client() -> MagicMock:
    return MagicMock()


def test_create_qurl_tool() -> None:
    client = _mock_client()
    client.create.return_value = CreateOutput(
        resource_id="r_abc123def45",
        qurl_link="https://qurl.link/#at_test",
        qurl_site="https://r_abc123def45.qurl.site",
        expires_at=datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc),
    )

    tool = CreateQURLTool(client=client)
    result = tool._run(target_url="https://example.com", expires_in="24h")

    assert "r_abc123def45" in result
    assert "https://qurl.link/#at_test" in result
    client.create.assert_called_once_with(
        target_url="https://example.com",
        expires_in="24h",
        description=None,
        one_time_use=None,
        max_sessions=None,
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

    assert "https://api.example.com/data" in result
    assert "305" in result
    assert "203.0.113.42" in result
    # resolve() now takes a plain string
    client.resolve.assert_called_once_with("at_k8xqp9h2sj9lx7r4a")


def test_resolve_qurl_tool_no_grant() -> None:
    client = _mock_client()
    client.resolve.return_value = ResolveOutput(
        target_url="https://api.example.com/data",
        resource_id="r_abc123def45",
        access_grant=None,
    )

    tool = ResolveQURLTool(client=client)
    result = tool._run(access_token="at_k8xqp9h2sj9lx7r4a")

    assert "https://api.example.com/data" in result
    assert "r_abc123def45" in result


def test_list_qurls_tool() -> None:
    client = _mock_client()
    client.list.return_value = ListOutput(
        qurls=[
            QURL(
                resource_id="r_abc123def45",
                target_url="https://example.com",
                status="active",
                created_at=datetime(2026, 3, 10, 10, 0, 0, tzinfo=timezone.utc),
                expires_at=datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc),
            )
        ],
        has_more=False,
    )

    tool = ListQURLsTool(client=client)
    result = tool._run(status="active", limit=10)

    assert "r_abc123def45" in result
    assert "https://example.com" in result
    client.list.assert_called_once_with(status="active", limit=10)


def test_list_qurls_tool_empty() -> None:
    client = _mock_client()
    client.list.return_value = ListOutput(qurls=[], has_more=False)

    tool = ListQURLsTool(client=client)
    result = tool._run()

    assert result == "No QURLs found."


def test_delete_qurl_tool() -> None:
    client = _mock_client()
    client.delete.return_value = None

    tool = DeleteQURLTool(client=client)
    result = tool._run(resource_id="r_abc123def45")

    assert "r_abc123def45" in result
    assert "revoked" in result
    client.delete.assert_called_once_with("r_abc123def45")


def test_toolkit_returns_all_tools() -> None:
    client = _mock_client()
    toolkit = QURLToolkit(client=client)
    tools = toolkit.get_tools()

    assert len(tools) == 4
    names = {t.name for t in tools}
    assert names == {"create_qurl", "resolve_qurl", "list_qurls", "delete_qurl"}
