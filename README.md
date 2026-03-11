# layerv-qurl

Python SDK for the [QURL API](https://docs.layerv.ai) — secure, time-limited access links for AI agents.

## Installation

```bash
pip install layerv-qurl
```

For LangChain integration:

```bash
pip install layerv-qurl[langchain]
```

## Quick Start

```python
from layerv_qurl import QURLClient

client = QURLClient(api_key="lv_live_xxx")

# Create a protected link
result = client.create(
    target_url="https://api.example.com/data",
    expires_in="24h",
    description="API access for agent",
)
print(result.qurl_link)

# Resolve a token (opens firewall for your IP)
access = client.resolve("at_k8xqp9h2sj9lx7r4a")
print(f"Access granted to {access.target_url} for {access.access_grant.expires_in}s")
# access.access_grant is None if no firewall grant was needed

# Update a QURL (extend, change description, etc.)
qurl = client.update("r_xxx", extend_by="7d", description="extended")
```

## Async Usage

```python
import asyncio
from layerv_qurl import AsyncQURLClient

async def main():
    async with AsyncQURLClient(api_key="lv_live_xxx") as client:
        result = await client.create(target_url="https://example.com", expires_in="1h")
        access = await client.resolve("at_...")

asyncio.run(main())
```

## Pagination

```python
# Iterate all active QURLs (auto-paginates)
for qurl in client.list_all(status="active"):
    print(f"{qurl.resource_id}: {qurl.target_url}")

# Or fetch a single page
page = client.list(status="active", limit=10)
for qurl in page.qurls:
    print(qurl.resource_id)
```

## Error Handling

```python
from layerv_qurl import QURLClient, QURLError, QURLNetworkError, QURLTimeoutError

client = QURLClient(api_key="lv_live_xxx")

try:
    client.resolve("at_k8xqp9h2sj9lx7r4a")
except QURLTimeoutError:
    # Request timed out — server may be slow
    print("Timed out, retrying...")
except QURLNetworkError as e:
    # Transport errors (DNS, connection refused)
    print(f"Network error: {e}")
except QURLError as e:
    # API errors (4xx/5xx) with structured detail
    print(f"API error: {e.status} {e.code} — {e.detail}")
    if e.invalid_fields:
        print(f"Invalid fields: {e.invalid_fields}")
    if e.request_id:
        print(f"Request ID: {e.request_id}")
```

## Typed Quota

```python
quota = client.get_quota()
print(f"Plan: {quota.plan}")
print(f"Active QURLs: {quota.usage.active_qurls}")
print(f"Rate limit: {quota.rate_limits.create_per_minute}/min")
```

## LangChain Integration

```python
from layerv_qurl import QURLClient
from layerv_qurl.langchain import QURLToolkit

client = QURLClient(api_key="lv_live_xxx")
toolkit = QURLToolkit(client=client)
tools = toolkit.get_tools()  # [CreateQURLTool, ResolveQURLTool, ListQURLsTool, DeleteQURLTool]
```

## Configuration

| Parameter | Required | Default |
|-----------|----------|---------|
| `api_key` | Yes | — |
| `base_url` | No | `https://api.layerv.ai` |
| `timeout` | No | `30.0` |
| `max_retries` | No | `3` |
| `user_agent` | No | `qurl-python-sdk/<version>` |
| `http_client` | No | Auto-created `httpx.Client` |

## License

MIT
