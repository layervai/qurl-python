"""qURL Python SDK — secure, time-limited access links for AI agents."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from layerv_qurl.async_client import AsyncQURLClient
from layerv_qurl.client import QURLClient
from layerv_qurl.errors import (
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    QURLError,
    QURLNetworkError,
    QURLTimeoutError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from layerv_qurl.types import (
    QURL,
    AccessGrant,
    AccessPolicy,
    AccessToken,
    AIAgentPolicy,
    BatchCreateItem,
    BatchCreateOutput,
    BatchItemError,
    BatchItemResult,
    CreateOutput,
    ListOutput,
    MintOutput,
    Quota,
    QuotaPlan,
    QURLStatus,
    RateLimits,
    ResolveOutput,
    TokenStatus,
    Usage,
)

__all__ = [
    "AsyncQURLClient",
    "QURLClient",
    # Errors
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "QURLError",
    "QURLNetworkError",
    "QURLTimeoutError",
    "RateLimitError",
    "ServerError",
    "ValidationError",
    # Types
    "AIAgentPolicy",
    "BatchCreateItem",
    "BatchCreateOutput",
    "BatchItemError",
    "BatchItemResult",
    "QuotaPlan",
    "QURLStatus",
    "TokenStatus",
    "AccessGrant",
    "AccessPolicy",
    "AccessToken",
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
except PackageNotFoundError:
    __version__ = "dev"
