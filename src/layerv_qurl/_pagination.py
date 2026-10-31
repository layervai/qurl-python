"""Cursor traversal shared by the synchronous and asynchronous clients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

from layerv_qurl.errors import QURLError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator


class CursorPage(Protocol):
    next_cursor: str | None
    has_more: bool


Page = TypeVar("Page", bound=CursorPage)


def _next_cursor(page: CursorPage, seen: set[str]) -> str | None:
    if not page.has_more:
        return None
    cursor = page.next_cursor
    if not isinstance(cursor, str) or not cursor or cursor in seen:
        raise QURLError(
            status=0,
            code="unexpected_response",
            title="Pagination requires a new non-empty cursor when has_more is true",
        )
    seen.add(cursor)
    return cursor


def pages(fetch: Callable[[str | None], Page]) -> Iterator[Page]:
    cursor = None
    seen: set[str] = set()
    while True:
        page = fetch(cursor)
        yield page
        cursor = _next_cursor(page, seen)
        if cursor is None:
            return


async def async_pages(fetch: Callable[[str | None], Awaitable[Page]]) -> AsyncIterator[Page]:
    cursor = None
    seen: set[str] = set()
    while True:
        page = await fetch(cursor)
        yield page
        cursor = _next_cursor(page, seen)
        if cursor is None:
            return
