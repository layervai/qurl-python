# layerv-qurl

[![PyPI](https://img.shields.io/pypi/v/layerv-qurl)](https://pypi.org/project/layerv-qurl/)
[![CI](https://github.com/layervai/qurl-python/actions/workflows/ci.yml/badge.svg)](https://github.com/layervai/qurl-python/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/layerv-qurl)](https://pypi.org/project/layerv-qurl/)
[![License](https://img.shields.io/github/license/layervai/qurl-python)](LICENSE)

Python SDK for the [qURL™ API](https://docs.layerv.ai) — secure, time-limited access links for AI agents.

> **Quantum URL (qURL)** · The internet has a hidden layer. This is how you enter.

## Why qURL?

AI agents need to access APIs, databases, and internal tools — but permanent credentials are a security risk. qURL creates **time-limited, auditable access links** that automatically expire:

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
    label="API access for agent",
)
print(result.qurl_link)  # Share this link

# Resolve a token (opens firewall for your IP)
access = client.resolve("at_k8xqp9h2sj9lx7r4a")
print(f"Access granted to {access.target_url} for {access.access_grant.expires_in}s")

# Extend a qURL's expiration
qurl = client.extend("r_xxx", "7d")

# Update resource metadata
qurl = client.update("r_xxx", description="extended", extend_by="7d")
```

## Authentication Notes

`QURLClient(api_key=...)` accepts either a qURL API key or a JWT bearer token.
Dashboard/account endpoints such as billing, customer, connector, webhook, and
API-key management require JWT authentication. You may omit `api_key` only for
public endpoints such as access-code redemption; authenticated endpoints return
401 without credentials.

Some resource-list responses intentionally omit `target_url` for redacted
resource types. Treat `QURL.target_url` as `str | None` before formatting or
parsing it.

Mutating SDK methods generate a per-call `Idempotency-Key` when you do not
provide one and reuse it across the client's internal retries; qurl-service
supports that header on mutating endpoints, including `POST /v1/resolve`.
Automatic POST status-code retries remain limited to rate limits because
one-time resolve tokens can be consumed by server-side knock failures. Pass a
stable `idempotency_key` when you need retry-safe behavior across your own retry
loop, process restart, or job replay. Caller-supplied keys should be globally
unique for each logical operation; UUID or ULID values are recommended.

Fields such as webhook `events` and API-key `scopes` accept ordered non-string
iterables of strings. Lists, tuples, and generators preserve the caller's
iteration order; sets are rejected because their iteration order is not stable.

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
# Iterate all active qURLs (auto-paginates)
for qurl in client.list_all(status="active"):
    target = qurl.target_url or "<redacted>"
    print(f"{qurl.resource_id}: {target}")

# Or fetch a single page
page = client.list(status="active", limit=10)
for qurl in page.qurls:
    print(qurl.resource_id)
```

## Resources

```python
# Create a resource explicitly, then mint scoped qURLs against it
resource = client.create_resource(
    resource_type="url",
    target_url="https://api.example.com/data",
    alias="reports-api",
)

link = client.create_qurl_for_resource(
    resource.resource_id,
    expires_in="1h",
    label="Alice from Acme",
    idempotency_key="invite-alice-2026-03-10",
)

# Revoke one token without closing the whole resource
assert link.qurl_id is not None
client.revoke_resource_qurl(resource.resource_id, link.qurl_id)
```

## Custom Domains And Webhooks

```python
domain = client.register_domain("secure.example.com")
for record in domain.dns_records:
    print(record.type, record.name, record.value)

webhook = client.create_webhook(
    url="https://example.com/qurl-webhooks",
    events=["qurl.accessed", "domain.verified"],
)
print(webhook.secret)  # Returned only on create/regenerate
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
print(f"Active qURLs: {quota.usage.active_qurls}")
print(f"Rate limit: {quota.rate_limits.create_per_minute}/min")
```

JWT-authenticated dashboard endpoints are also available for usage, customer
settings, billing sessions, invoices, and API-key management. API-key auth
continues to work for normal qURL, resource, domain, webhook, connector, and
access-code operations according to the API scopes on the key.

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
