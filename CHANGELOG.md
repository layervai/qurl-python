# Changelog

## Unreleased

- Synced the Python client with the latest qURL API contract, including
  resource, domain, webhook, billing, connector, agent bootstrap, API key,
  and access-code endpoints.
- `QURL.target_url` is now typed as `str | None` because resource-list
  responses may omit the protected target URL for redacted resource types.
