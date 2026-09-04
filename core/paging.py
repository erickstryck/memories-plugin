"""Paging an ORDERED listing, which Qdrant makes the client's job.

Measured against Qdrant 1.18.2 on 2026-09-03, and each of these is why this module
exists rather than a couple of lines in `memory`:

  - `order_by` together with `offset` is refused outright: "Cannot use an `offset` when
    using `order_by`. The alternative for paging is to use `order_by.start_from` and a
    filter to exclude the IDs that you've already seen".
  - with `order_by`, `next_page_offset` comes back NULL even when more pages exist. The
    server keeps no cursor for an ordered scroll.
  - `start_from` is INCLUSIVE of the boundary value, so the records already returned at
    that value come back again unless they are excluded by id.

The last point is the whole reason a cursor here is a PAIR (value, ids-seen-at-it) and
not just a timestamp: `store_many` writes N records with a single timestamp, so a tie at
the page boundary is ordinary traffic.

Nothing here knows about Qdrant, HTTP or memory: it takes data and returns data, so the
one delicate rule in this feature is provable without a server.
"""
import base64
import binascii
import json

from .errors import CoreError

#: The payload key the listing orders by, and the index schema it needs. Named here
#: because three modules would otherwise each spell the string themselves.
ORDER_KEY = "updated_at"
INDEX_SCHEMA = "datetime"

#: The two values `list_page` reports in `order`. A caller has to be able to tell an
#: ordered page from a degraded one WITHOUT parsing prose.
ORDER_DESC = "updated_at_desc"
ORDER_NONE = "unordered"

_CURSOR_VERSION = 1


class PagingError(CoreError):
    """A cursor that cannot be honoured, reported instead of restarting silently."""


def encode_cursor(value: str, seen_ids: list) -> str:
    raw = json.dumps({"v": _CURSOR_VERSION, "value": value, "seen": list(seen_ids)},
                     separators=(",", ":"), default=str).encode()

    return base64.urlsafe_b64encode(raw).decode()


def decode_cursor(cursor: str) -> tuple[str, list]:
    """The pair the cursor carries, or `PagingError`.

    It refuses rather than falling back to the first page: a caller handed page 1 while
    believing it advanced re-reads what it already saw and never terminates.
    """
    if not isinstance(cursor, str) or not cursor.strip():
        raise PagingError("empty pagination cursor: omit it to start from the newest")
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode()))
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        raise PagingError(
            f"unreadable pagination cursor: {exc}. Use the 'next_offset' a previous "
            f"listing returned, or omit it to start from the newest."
        ) from exc
    if not isinstance(payload, dict) or payload.get("v") != _CURSOR_VERSION:
        raise PagingError("pagination cursor is not one of ours, or is from an older "
                          "format: omit it to start from the newest")
    value, seen = payload.get("value"), payload.get("seen")
    if not isinstance(value, str) or not isinstance(seen, list):
        raise PagingError("pagination cursor is malformed: omit it to start from the "
                          "newest")

    return value, seen


def page_request(cursor: str | None) -> dict:
    """The `order_by` and `filter` for the page `cursor` points at.

    `filter` is None rather than an empty `must_not`: an empty clause is one the server
    still has to evaluate, and it reads as "something is excluded" to anyone debugging.
    """
    order_by = {"key": ORDER_KEY, "direction": "desc"}
    if cursor is None:
        return {"order_by": order_by, "filter": None}
    value, seen = decode_cursor(cursor)
    order_by["start_from"] = value

    return {"order_by": order_by,
            "filter": {"must_not": [{"has_id": list(seen)}]} if seen else None}


def next_cursor(points: list[dict], limit: int, previous: str | None) -> str | None:
    """The cursor for the page AFTER `points`, or None when this was the last one.

    A page shorter than `limit` is the last page. That is the only termination signal
    available: the server's own `next_page_offset` is null on every ordered scroll, so
    trusting it would end every listing after page one.
    """
    if not points or len(points) < limit:
        return None
    boundary = (points[-1].get("payload") or {}).get(ORDER_KEY)
    if not isinstance(boundary, str) or not boundary:
        # Nothing to anchor `start_from` on. Such a record is invisible to an ordered
        # scroll anyway, so claiming "no more pages" is the honest end of the walk.
        return None
    seen = {p.get("id") for p in points
            if (p.get("payload") or {}).get(ORDER_KEY) == boundary}
    if previous is not None:
        prior_value, prior_seen = decode_cursor(previous)
        # Only while the boundary has not moved. Carrying ids past their value would
        # grow the exclusion list without bound across a long listing.
        if prior_value == boundary:
            seen |= set(prior_seen)

    return encode_cursor(boundary, sorted(i for i in seen if i is not None))
