"""Error types for the QURL API client."""

from __future__ import annotations


class QURLError(Exception):
    """Error raised for API-level errors (4xx/5xx responses).

    Catch specific subclasses for fine-grained handling::

        try:
            client.resolve("at_xxx")
        except AuthenticationError:
            print("Bad API key")
        except NotFoundError:
            print("QURL doesn't exist")
        except RateLimitError as e:
            print(f"Rate limited — retry in {e.retry_after}s")
        except QURLError as e:
            print(f"API error: {e.status} {e.code}")
    """

    def __init__(
        self,
        *,
        status: int,
        code: str,
        title: str,
        detail: str,
        invalid_fields: dict[str, str] | None = None,
        request_id: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(f"{title} ({status}): {detail}")
        self.status = status
        self.code = code
        self.title = title
        self.detail = detail
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
