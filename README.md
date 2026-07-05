# qurl-python

[![PyPI](https://img.shields.io/pypi/v/qurl-python)](https://pypi.org/project/qurl-python/)
[![CI](https://github.com/layervai/qurl-python/actions/workflows/ci.yml/badge.svg)](https://github.com/layervai/qurl-python/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/qurl-python)](https://pypi.org/project/qurl-python/)
[![License](https://img.shields.io/github/license/layervai/qurl-python)](LICENSE)

**Use the LayerV [qURL™ Platform](https://docs.layerv.ai) from Python: protect a
private URL once, then mint short-lived portal links for it.**

> **Quantum URL (qURL)** · The internet has a hidden layer. This is how you enter.

Portal recipients do not need LayerV credentials, API keys, or SDK state. They
open the qURL link. Credentials are only for software that protects URLs or
creates portals.

## Why qURL?

Agents and services increasingly need to reach private MCP servers, APIs, and
internal tools. The issue is visibility: every standing public endpoint becomes
inventory for scanners, fingerprinting, credential attacks, and AI-assisted
probing before a legitimate user or agent ever arrives.

qURL flips that model. The private resource is not public inventory. A portal
is **cryptographic, just-in-time permission for one actor to reach one private
resource** — not another externally visible endpoint in front of the same
service:

- **Time-limited** — portals expire after minutes, hours, or days
- **IP-scoped** — access is granted only to the requesting IP via NHP
- **Auditable** — every access is logged with who, when, and from where
- **Revocable** — kill access instantly if something goes wrong

## Installation

```bash
pip install qurl-python
```

For LangChain integration:

```bash
pip install qurl-python[langchain]
```

## Quickstart

```python
from layerv_qurl import QURLClient

client = QURLClient(api_key="lv_live_xxx")

resource = client.protect_url("https://internal.example.com/dashboard")
portal = resource.create_portal(valid_for="5m")

print(portal.link)  # Share this link — recipients need no credentials
```

That is the core flow:

| Step | Call | What you provide |
| --- | --- | --- |
| Protect a private URL | `client.protect_url` | The target URL you already know |
| Mint a short-lived access link | `resource.create_portal` | The returned resource handle |

`protect_url` is idempotent for the same account and target URL: protecting
the same URL again returns the existing resource. `valid_for` accepts a
duration string (`"5m"`, `"24h"`) or a `datetime.timedelta`; prefer short
portal lifetimes.

If qURL Connector already protects the service, use the connector id instead
of calling `protect_url`:

```python
resource = client.connector_resource("prod-dashboard")
portal = resource.create_portal(valid_for="5m")
```

If you persist the resource id, future calls do not need to recreate the
handle (no API call is made until you mint):

```python
resource = client.resource_by_id("r_demo1234567")
portal = resource.create_portal(valid_for="1h")
```

For one-off scripts, `client.create_portal_for_url` combines the two API calls
and returns both the portal and a reusable resource handle:

```python
portal, resource = client.create_portal_for_url(
    "https://internal.example.com/dashboard", valid_for="5m"
)
```

Portal options mirror qurl-go:

```python
from datetime import timedelta

portal = resource.create_portal(
    valid_for=timedelta(minutes=5),
    label="Alice from Acme",
    one_time_use=True,
    max_sessions=1,
)
```

## Opening Portals

Most recipients open qURL links directly and do not use this SDK at all. If
you are building a service or agent that opens received qURL links
programmatically, `enter_portal` accepts a full link or a bare access token,
grants network access for the caller's IP, and returns the reachable resource:

```python
handle = client.enter_portal(link)
print(handle.resource_url)   # The reachable resource location
print(handle.open_seconds)   # How long access stays open
```

Unlike qurl-go's offline `EnterPortal`, this SDK opens links through the
LayerV API: the client needs an API key with the `qurl:resolve` scope.
`enter_portal` fails closed — if access is granted but no resource URL comes
back, it raises instead of returning an empty handle.

## Async Usage

```python
import asyncio
from layerv_qurl import AsyncQURLClient

async def main():
    async with AsyncQURLClient(api_key="lv_live_xxx") as client:
        resource = await client.protect_url("https://internal.example.com/dashboard")
        portal = await resource.create_portal(valid_for="5m")
        print(portal.link)

asyncio.run(main())
```

## REST-Shaped API (Compatibility)

The original REST-shaped methods remain fully supported and share the same
client. Use them for the qURL/resource/token management surface that has no
portal-verb equivalent (listing, updating, revoking, quotas, webhooks, ...) or
if you already build on them:

```python
# Create a protected link (portal equivalent: create_portal_for_url)
result = client.create(
    target_url="https://api.example.com/data",
    expires_in="24h",
    label="API access for agent",
)
print(result.qurl_link)

# Resolve a token headlessly (portal equivalent: enter_portal)
access = client.resolve("at_k8xqp9h2sj9lx7r4a")
print(f"Access granted to {access.target_url} for {access.access_grant.expires_in}s")

# Extend a qURL's expiration and update metadata
qurl = client.extend("r_xxx", "7d")
qurl = client.update("r_xxx", description="extended")
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
    client.enter_portal("https://qurl.link/#at_k8xqp9h2sj9lx7r4a")
except AuthenticationError:
    print("Bad API key")
except AuthorizationError:
    print("Valid key but missing qurl:resolve scope")
except NotFoundError:
    print("Portal doesn't exist or already expired")
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

## Security Notes

- Treat API keys and qURL links like credentials. Do not log them.
- Prefer short portal lifetimes such as `valid_for="5m"`.
- Do not ask portal recipients to handle credentials. Recipients only need
  the link.
- `protect_url` and `create_portal_for_url` reject malformed target URLs and
  URLs with embedded credentials (`https://user:pass@...`) before any request,
  matching qurl-go.

## License

MIT
