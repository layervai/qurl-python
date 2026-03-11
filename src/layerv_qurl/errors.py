"""Error types for the QURL API client."""

from __future__ import annotations


class QURLError(Exception):
    """Error raised for API-level errors (4xx/5xx responses)."""

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
