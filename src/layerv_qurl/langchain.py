"""LangChain tool integration for QURL.

Install with: pip install layerv-qurl[langchain]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from langchain_core.tools import BaseTool
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    BaseTool = object  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForToolRun

    from layerv_qurl.client import QURLClient

__all__ = [
    "CreateQURLTool",
    "DeleteQURLTool",
    "ListQURLsTool",
    "QURLToolkit",
    "ResolveQURLTool",
]


class CreateQURLTool(BaseTool):
    """Create a secure, time-limited access link to a protected URL."""

    name: str = "create_qurl"
    description: str = (
        "Create a QURL — a secure, time-limited access link. "
        "Input should be a JSON string with 'target_url' (required), "
        "and optionally 'expires_in' (e.g. '24h', '7d'), 'description', "
        "'one_time_use' (bool), 'max_sessions' (int)."
    )
    client: Any = None  # QURLClient, typed as Any for Pydantic compatibility

    def _run(
        self,
        target_url: str,
        expires_in: str = "24h",
        description: str | None = None,
        one_time_use: bool | None = None,
        max_sessions: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        result = self.client.create(
            target_url=target_url,
            expires_in=expires_in,
            description=description,
            one_time_use=one_time_use,
            max_sessions=max_sessions,
        )
        return (
            f"Created QURL {result.resource_id}\n"
            f"Link: {result.qurl_link}\n"
            f"Site: {result.qurl_site}\n"
            f"Expires: {result.expires_at or 'N/A'}"
        )


class ResolveQURLTool(BaseTool):
    """Resolve a QURL access token to open firewall access."""

    name: str = "resolve_qurl"
    description: str = (
        "Resolve a QURL access token to gain firewall access to the protected resource. "
        "Input should be the access token string (e.g. 'at_k8xqp9h2sj9lx7r4a')."
    )
    client: Any = None

    def _run(
        self,
        access_token: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        result = self.client.resolve(access_token)
        grant = result.access_grant
        lines = [
            f"Resolved: {result.target_url}",
            f"Resource: {result.resource_id}",
        ]
        if grant:
            lines.append(f"Access expires in: {grant.expires_in}s")
            lines.append(f"Granted to IP: {grant.src_ip}")
        return "\n".join(lines)


class ListQURLsTool(BaseTool):
    """List active QURL links."""

    name: str = "list_qurls"
    description: str = (
        "List active QURL links. Optionally filter by status (active, expired, revoked, consumed)."
    )
    client: Any = None

    def _run(
        self,
        status: str = "active",
        limit: int = 10,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        result = self.client.list(status=status, limit=limit)
        if not result.qurls:
            return "No QURLs found."
        lines = []
        for q in result.qurls:
            lines.append(f"- {q.resource_id}: {q.target_url} [{q.status}] expires={q.expires_at}")
        return "\n".join(lines)


class DeleteQURLTool(BaseTool):
    """Revoke a QURL, immediately ending all access."""

    name: str = "delete_qurl"
    description: str = "Revoke (delete) a QURL by resource ID (e.g. 'r_k8xqp9h2sj9')."
    client: Any = None

    def _run(
        self,
        resource_id: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        self.client.delete(resource_id)
        return f"QURL {resource_id} has been revoked."


class QURLToolkit:
    """LangChain toolkit providing all QURL tools.

    Usage::

        from layerv_qurl import QURLClient
        from layerv_qurl.langchain import QURLToolkit

        client = QURLClient(api_key="lv_live_xxx")
        toolkit = QURLToolkit(client=client)
        tools = toolkit.get_tools()
    """

    def __init__(self, client: QURLClient) -> None:
        if not _HAS_LANGCHAIN:
            raise ImportError(
                "langchain-core is required for LangChain integration. "
                "Install with: pip install layerv-qurl[langchain]"
            )
        self.client = client

    def get_tools(self) -> list[BaseTool]:
        """Return all QURL tools configured with the client."""
        return [
            CreateQURLTool(client=self.client),
            ResolveQURLTool(client=self.client),
            ListQURLsTool(client=self.client),
            DeleteQURLTool(client=self.client),
        ]
