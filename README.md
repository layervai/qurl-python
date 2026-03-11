# layerv-qurl

[![PyPI](https://img.shields.io/pypi/v/layerv-qurl)](https://pypi.org/project/layerv-qurl/)
[![CI](https://github.com/layervai/qurl-python/actions/workflows/ci.yml/badge.svg)](https://github.com/layervai/qurl-python/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/layerv-qurl)](https://pypi.org/project/layerv-qurl/)
[![License](https://img.shields.io/github/license/layervai/qurl-python)](LICENSE)

Python SDK for the [QURL API](https://docs.layerv.ai) — secure, time-limited access links for AI agents.

## Why QURL?

AI agents need to access APIs, databases, and internal tools — but permanent credentials are a security risk. QURL creates **time-limited, auditable access links** that automatically expire:

- **Time-limited** — links expire after minutes, hours, or days
- **IP-scoped** — firewall opens only for the requesting IP via NHP
- **Auditable** — every access is logged with who, when, and from where
- **Revocable** — kill access instantly if something goes wrong

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
print(result.qurl_link)  # Share this link

# Resolve a token (opens firewall for your IP)
access = client.resolve("at_k8xqp9h2sj9lx7r4a")
print(f"Access granted to {access.target_url} for {access.access_grant.expires_in}s")

# Extend a QURL's expiration
qurl = client.extend("r_xxx", "7d")

# Update metadata and policy
qurl = client.update("r_xxx", description="extended", extend_by="7d")
```

## Async Usage

```python
import asyncio
from layerv_qurl import AsyncQURLClient

async def main():
    async with AsyncQURLClient(api_key="lv_live_xxx") as client:
        result = await client.create(target_url="https://example.com", expires_in="1h")
        access = await client.resolve("at_...")

        # Extend expiration
        qurl = await client.extend("r_xxx", "7d")

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

Every API error maps to a specific exception class, so you can catch exactly what you need:

```python
from layerv_qurl import (
    QURLClient,
    QURLError,
    QURLNetworkError,
    QURLTimeoutError,
)
from layerv_qurl.errors import (
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)

client = QURLClient(api_key="lv_live_xxx")

try:
    client.resolve("at_k8xqp9h2sj9lx7r4a")
except AuthenticationError:
    print("Bad API key")
except AuthorizationError:
    print("Valid key but missing qurl:resolve scope")
except NotFoundError:
    print("Token doesn't exist or already expired")
except RateLimitError as e:
    print(f"Rate limited — retry in {e.retry_after}s")
except ValidationError as e:
    print(f"Bad request: {e.detail}")
    if e.invalid_fields:
        for field, reason in e.invalid_fields.items():
            print(f"  {field}: {reason}")
except QURLTimeoutError:
    print("Request timed out")
except QURLNetworkError as e:
    print(f"Network error: {e}")
except QURLError as e:
    # Catch-all for any other API error
    print(f"API error {e.status}: {e.detail}")
```

All error classes inherit from `QURLError`, so `except QURLError` catches everything.

## Typed Quota

```python
quota = client.get_quota()
print(f"Plan: {quota.plan}")
print(f"Active QURLs: {quota.usage.active_qurls}")
print(f"Rate limit: {quota.rate_limits.create_per_minute}/min")
```

## Debug Logging

Enable debug logs to see every request and retry:

```python
import logging
logging.getLogger("layerv_qurl").setLevel(logging.DEBUG)

# Output:
# DEBUG:layerv_qurl:POST https://api.layerv.ai/v1/qurl
# DEBUG:layerv_qurl:POST https://api.layerv.ai/v1/qurl → 201
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
