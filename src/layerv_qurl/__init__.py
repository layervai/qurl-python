"""QURL Python SDK — secure, time-limited access links for AI agents."""

from importlib.metadata import version as _pkg_version

from layerv_qurl.async_client import AsyncQURLClient
from layerv_qurl.client import QURLClient
from layerv_qurl.errors import QURLError, QURLNetworkError, QURLTimeoutError
from layerv_qurl.types import (
    QURL,
    AccessGrant,
    AccessPolicy,
    CreateOutput,
    ListOutput,
    MintOutput,
    Quota,
    QURLStatus,
    RateLimits,
    ResolveOutput,
    Usage,
)

__all__ = [
    "AsyncQURLClient",
    "QURLClient",
    "QURLError",
    "QURLNetworkError",
    "QURLStatus",
    "QURLTimeoutError",
    "AccessGrant",
    "AccessPolicy",
    "CreateOutput",
    "ListOutput",
    "MintOutput",
    "QURL",
    "Quota",
    "RateLimits",
    "ResolveOutput",
    "Usage",
]

try:
    __version__ = _pkg_version("layerv-qurl")
except Exception:
    __version__ = "dev"
