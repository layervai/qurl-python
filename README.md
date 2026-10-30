# qURL Python SDK for expiring access links

[![PyPI](https://img.shields.io/pypi/v/qurl-python)](https://pypi.org/project/qurl-python/)
[![CI](https://github.com/layervai/qurl-python/actions/workflows/ci.yml/badge.svg)](https://github.com/layervai/qurl-python/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/qurl-python)](https://pypi.org/project/qurl-python/)
[![License](https://img.shields.io/github/license/layervai/qurl-python)](LICENSE)

Use [qURL™ by LayerV](https://layerv.ai/qurl/) to share localhost web apps,
private APIs, and files through expiring access links. This
[Python SDK](https://layerv.ai/sdk/) creates and manages resource access from
your code. Use `qurl-python` in Python
services, automation scripts, or LangChain tools to register resources and
issue links with a chosen lifetime.

For a local app, [publish it with the CLI](https://github.com/layervai/qurl-integrations/tree/main/apps/cli)
or configure a connector before creating access links. SDK calls register
resources and issue links; they do not start the local service or remove
any separate public route to it. Your app's own sign-in still applies.

Portal recipients do not need LayerV credentials, API keys, or SDK state. They
open the qURL link. Credentials are only for software that protects URLs or
creates portals.

[Installation](#installation) · [Quickstart](#quickstart) ·
[Async usage](#async-usage) · [LangChain integration](#langchain-integration) ·
[Resource and token management](#rest-shaped-api-compatibility)

[Local preview workflow](#share-a-localhost-preview-from-python) ·
[File-sharing workflows](#share-files-from-a-python-workflow) ·
[Sharing questions](#python-sharing-questions) · [Private links](https://qurl.link/)

## Share a localhost preview from Python

Use this SDK when a Python service, automation job, or LangChain tool should
issue a short-lived link to an existing app. Examples include a dashboard
review, an internal API handoff, or a preview produced by an automation job.

There are two parts to the workflow:

| Part | What to do |
| --- | --- |
| Run and publish the local service | Keep the app running and follow the [CLI localhost quickstart](https://github.com/layervai/qurl-integrations#share-a-localhost-web-app), or deploy a compatible Connector beside the app. |
| Issue access from Python | Use a resource managed by this SDK's configured API and accessible to its credential. Ask for an appropriate lifetime and send the returned link. |

For a Connector resource already configured for your API credential, the
following script requests a five-minute link. Set `QURL_CONNECTOR_SLUG` to the
resource's immutable slug, and `QURL_API_KEY` to an issuer credential with the
required permissions. The optional display alias is not a lookup identifier:

```python
import os

from layerv_qurl import QURLClient

with QURLClient(api_key=os.environ["QURL_API_KEY"]) as client:
    resource = client.connector_resource(os.environ["QURL_CONNECTOR_SLUG"])
    portal = resource.create_portal(valid_for="5m")
    print(portal.link)
```

The slug and credential must refer to the same authorized resource.
Do not substitute a CLI CRID for a resource slug or assume that an
account key manages a resource published by another device. If the CLI alone
published the preview, its `qurl share` command is the direct sharing path;
use SDK management only after the compatible resource is configured.

Keep the full link private. A browser recipient opens it without installing
this SDK; the app's own login still applies. See [async usage](#async-usage)
for an async Python service and [LangChain integration](#langchain-integration)
for agent tools.

## Share files from a Python workflow

For reports or generated documents, choose the actual file-serving path before
creating an access link:

- Serve a dedicated directory with a localhost HTTP server and publish it with
  the [CLI file-sharing workflow](https://github.com/layervai/qurl-integrations#share-files-and-documents).
  That host and server must remain available for the download.
- Use the [MCP upload tools](https://github.com/layervai/qurl-mcp#let-an-ai-agent-share-a-pdf-or-image)
  or [Gmail integration](https://github.com/layervai/qurl-integrations/tree/main/apps/chrome-extension)
  for supported uploaded documents and images.
- If a compatible private file service is already registered for your SDK
  credential, issue access through its resource handle using the portal APIs.

This SDK's link-issuing calls do not upload a local Python file or start a
file server. File-upload link minting can have different identifiers and
revocation rules; follow the service or integration that created the upload.

## Link issuance and Python connectivity

Use `qurl-python` when an automation job or LangChain tool should issue access to a resource its credential can manage. In the localhost workflow above, a CLI or compatible Connector first supplies connectivity, and Python requests the portal. The [ngrok Python Agent SDK](https://github.com/ngrok/ngrok-python) can establish forwarding to a local address from the Python process itself. These are different operations: resource/link management and establishing an endpoint. Choose the lifecycle your application needs, including its authentication and the runtime that must stay alive.

For a generated report already stored in S3, a [presigned download URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html) may fit the existing storage workflow. For a dashboard or files served by a live local app, publish that service and issue access to the compatible resource. This SDK's portal calls do not upload a local report. The [app and file-sharing comparison](https://github.com/layervai/qurl-integrations#choose-how-to-share-an-app-or-file) covers hosting, recipient access, and expiry; [async usage](#async-usage) covers issuing from an async Python service.

## Python sharing questions

### Can I pass localhost to `protect_url` and share it immediately?

A URL registration does not create a connection to your laptop. Publish or
configure the running service first. Use the appropriate registered resource
and permissions; another machine's `localhost` is not your development app.

### Does a five-minute link close every session after five minutes?

Link expiry controls new redemption. Session duration is a separate setting;
an already admitted session can outlive the link's redemption deadline. See
[portal options](#quickstart) and the service's enforced policy.

### Do browser recipients and Python openers need the same credentials?

Browser recipients open the link and retain any app login requirement.
This Python SDK's `enter_portal` uses the LayerV API and requires a credential
with `qurl:resolve`; see [opening portals](#opening-portals). Do not pass an
issuer API key to someone merely to let them open a browser link.

## Why qURL?

qURL lets your code issue temporary access to a resource while the local
origin can remain on localhost or a private network. The Connector supplies
that access path; the SDK controls resource and link operations. Restrict any
separate public route if qURL should be the only way to reach the app.

A portal link is an access credential, not proof that its holder is the person
named in a label. Keep application authentication when identity matters. Use
the resource, token, and session APIs for the access rules your workflow needs.

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

If qURL Connector already protects the service, use its immutable resource
slug instead of calling `protect_url`. The optional display alias is not a
lookup identifier:

```python
resource = client.connector_resource("prod-dashboard")
portal = resource.create_portal(valid_for="5m")
```

If you persist the resource id, future calls do not need to recreate the
handle (no API call is made until you mint):

```python
resource = client.resource_by_crid(saved_crid)
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

## Resource Identity, Privacy, and Sharing

`qurl-go` is the behavior reference for this SDK. Resource handles prefer the
public **CRID**; `resource.details.resource_id` remains the resource public key
for verification and compatibility. Persist the CRID, then call
`client.resource_by_crid(saved_crid)` to reuse it. The old `resource_by_id`
method still accepts a public key. Legacy `r_` IDs are no longer accepted by
the current service. **Migration:** when a response includes a CRID,
`ProtectedResource.id` now holds that CRID rather than `Resource.resource_id`.
Use `resource.details.resource_id` if your code needs the public key.

Sharing requires a CRID from the current service. If an older response omits
it, the compatibility handle can still mint with `create_portal`, but its
public-key ID cannot be passed to `share_resource`.

Privacy is fixed at creation. Omit `private` to use the service default
(private for new URL and Connector resources), or explicitly pass `False`
for a public resource. Reusing an existing resource preserves its privacy;
asking for a different value returns an API error. An omitted response
`private` remains `None`, not an assertion that the resource is public.

```python
resource = client.protect_url("https://internal.example.com/dashboard", private=True)
link = client.share_resource(resource.id, ttl=timedelta(minutes=5))
print(link.link)
# Revoke only this link, leaving the resource and other links intact:
if link.qurl_id:
    client.revoke_resource_qurl(link.crid, link.qurl_id)
```

`share_resource` mirrors Go's `ShareResource`: it requires a locally valid
CRID, rejects a response for another CRID, and accepts `timedelta` values for
`ttl` and `session_duration`. Durations must be nonnegative whole seconds;
zero or `None` uses the service default. Each call mints a fresh link, and the
share endpoint does **not** support idempotency replay, so the SDK does not
automatically retry this call after errors. Retain `qurl_id` with
the link if individual revocation is needed.

`link.verify_crid(der_spki_bytes)` checks a delivered key against the CRID you
requested. This does not verify the signed link or the publisher metadata.
`link.publisher.name` is untrusted display text; escape it before rendering,
and use `link.publisher.verified` for the service's verification status.

Owners can pass `allowed_device_keys` to `create_resource` or
`update_resource` (an empty list clears grants). To preserve other grants
while changing individual devices, use `allowed_device_keys_add` and
`allowed_device_keys_remove` on `update_resource`. Keys are canonical padded
standard-base64 X25519 public keys. Replacement and incremental updates cannot
be combined, and grants must be updated separately from `custom_domain`.
Removing a grant blocks new links; already minted links retain their lifetime.

Both sync and async clients expose these methods. Native UDP enrollment,
credential recovery, and opening/verifying signed `qv2` links remain outside
this Python SDK's current scope. Open signed links in a browser or with Go's
`EnterPortal`; Python's `enter_portal` still accepts only platform access tokens.

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
qurl = client.extend(saved_crid, "7d")
qurl = client.update(saved_crid, description="extended")
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
supports that header on mutating endpoints, including `POST /v1/resolve`,
except that `share_resource` always mints a new link.
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
usage = quota.usage
active = usage.active_resources if usage is not None else None
if active is None and usage is not None:
    active = usage.active_qurls
print("Active protected resources:", active if active is not None else "unavailable")
if quota.rate_limits is not None:
    print(f"Rate limit: {quota.rate_limits.create_per_minute}/min")
```

Use `max_active_resources`, `active_resources`, and `active_resources_percent`
for resource quotas. `max_data_transfer_bytes` is the monthly byte allowance;
`data_transfer_bytes` is measured usage, available on both quota usage and
current-period responses. This usage always covers the current UTC calendar
month, independently of a paid account's billing period. Missing or null values
remain `None`, distinct from a reported zero. The [service OpenAPI spec](https://github.com/layervai/qurl-service/blob/01d7c47361b5f72f890428f1a53bd04e8d7f67cc/api/openapi.yaml#L5796)
defines `-1` as unlimited for `max_active_resources` and its legacy alias
`max_active_qurls`, as well as `max_data_transfer_bytes`.

**Migration:** the deprecated `max_active_qurls` and `active_qurls` fields still
represent protected resources. Missing values now return `None` instead of
`0`; check `is not None` before arithmetic or displaying a count. Canonical
and legacy fields preserve their own wire values, so an older response that
omits canonical fields leaves those fields `None`. Prefer the canonical fields
when available.

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
