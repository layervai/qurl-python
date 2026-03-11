# CLAUDE.md — qurl-python

## Critical Rules

- **NEVER push directly to `main`.** Always create a branch and PR.
- All commits must be signed.

## Project

Python SDK for the QURL API (`pip install layerv-qurl`). Extracted from `layervai/qurl-integrations`.

## Commands

```bash
pip install -e ".[dev]"    # Install with dev dependencies
ruff check                 # Lint
mypy src/                  # Type check
pytest tests/ -v           # Test
```

## Commit Format

```
<type>: <description>

type: feat | fix | chore | docs | test | refactor | ci
```

Conventional commits drive Release Please versioning.

## Release

Merging to `main` triggers Release Please. Merging the release PR publishes to PyPI via OIDC.
