"""Error types for the qURL API client."""

from __future__ import annotations


class QURLError(Exception):
    """Error raised for API-level errors (4xx/5xx responses).

    Carries the full RFC 7807 Problem Details shape when the API provides
    it: ``status``, ``code``, ``title``, ``detail``, plus the optional
    ``type`` (problem-type URI) and ``instance`` (occurrence URI).

    .. note::

       ``detail`` is **always non-empty** on the instance — the
       constructor falls back to ``title`` when the API omits detail
       (RFC 7807 allows this). Use ``code`` / ``status`` / ``type`` to
       distinguish between error cases rather than inspecting ``detail``
       for the "was it absent?" signal.

    .. note::

       ``type`` shadows Python's built-in ``type()`` inside method
       bodies. This is intentional — the name mirrors the RFC 7807 field
       name and matches the other SDKs (``qurl-typescript``,
       ``qurl-mcp``). The shadowing only matters inside ``QURLError``
       method definitions; external code can still use ``type(err)``
       safely since attribute access doesn't shadow the builtin in that
       scope.

    Catch specific subclasses for fine-grained handling::

        try:
            client.resolve("at_xxx")
        except AuthenticationError:
            print("Bad API key")
        except NotFoundError:
            print("qURL doesn't exist")
        except RateLimitError as e:
            print(f"Rate limited — retry in {e.retry_after}s")
        except QURLError as e:
            print(f"API error: {e.status} {e.code}")
            if e.type:
                print(f"  problem type: {e.type}")
    """

    def __init__(
        self,
        *,
        status: int,
        code: str,
        title: str,
        detail: str | None = None,
        type: str | None = None,
        instance: str | None = None,
        invalid_fields: dict[str, str] | None = None,
        request_id: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        # RFC 7807 leaves `detail` optional, and `title` is always present.
        # When `detail` is `None` (omitted), fall back to `title` so the
        # Exception message stays meaningful instead of producing
        # "Title (403): None". An explicit empty string is stored as-is —
        # the caller opted in. Uses `is not None` rather than truthiness
        # so `detail=""` is distinguishable from "not provided".
        message_detail = detail if detail is not None else title
        super().__init__(f"{title} ({status}): {message_detail}")
        self.status = status
        self.code = code
        self.title = title
        self.detail = message_detail
        self.type = type
        self.instance = instance
        self.invalid_fields = invalid_fields
        self.request_id = request_id
        self.retry_after = retry_after


class AuthenticationError(QURLError):
    """401 Unauthorized — invalid or missing API key."""


class AuthorizationError(QURLError):
    """403 Forbidden — valid key but insufficient permissions/scope."""


class NotFoundError(QURLError):
    """404 Not Found — resource does not exist."""


class ValidationError(QURLError):
    """400/422 — invalid request parameters.

    Check :attr:`invalid_fields` for per-field details::

        except ValidationError as e:
            if e.invalid_fields:
                for field, reason in e.invalid_fields.items():
                    print(f"  {field}: {reason}")
    """


class RateLimitError(QURLError):
    """429 Too Many Requests.

    Check :attr:`retry_after` for the server-suggested wait time::

        except RateLimitError as e:
            if e.retry_after:
                time.sleep(e.retry_after)
    """


class ServerError(QURLError):
    """5xx server-side error."""


class QURLNetworkError(Exception):
    """Error raised for transport-level failures (DNS, connection refused)."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


class QURLTimeoutError(QURLNetworkError):
    """Error raised when a request times out.

    Subclass of :class:`QURLNetworkError` — caught by
    ``except QURLNetworkError`` but can also be caught specifically::

        try:
            client.resolve("at_xxx")
        except QURLTimeoutError:
            print("Request timed out — server may be slow")
        except QURLNetworkError:
            print("Network issue — DNS, connection, etc.")
        except QURLError:
            print("API error — 4xx/5xx")
    """
