"""Paging a listing, ordered when the store can order and honest when it cannot.

Measured against Qdrant 1.18.2 on 2026-09-03, and each of these is why this module
exists rather than a couple of lines in `memory`:

  - `order_by` together with `offset` is refused outright: "Cannot use an `offset` when
    using `order_by`. The alternative for paging is to use `order_by.start_from` and a
    filter to exclude the IDs that you've already seen".
  - with `order_by`, `next_page_offset` comes back NULL even when more pages exist. The
    server keeps no cursor for an ordered scroll.
  - `start_from` is INCLUSIVE of the boundary value, so the records already returned at
    that value come back again unless they are excluded by id.

The third point is why an ordered cursor is a PAIR (value, ids-seen-at-it) and not just a
timestamp: `store_many` writes N records with a single timestamp, so a tie at the page
boundary is ordinary traffic.

TWO MODES, and the second one is not decoration. When the store cannot order at all, the
listing degrades to an unordered scroll, and an unordered scroll pages the way it always
did: by the server's own `next_page_offset`. A first version of this module had one mode
and forced `next_offset` to null whenever ordering failed, which silently reduced the
whole archive to a single page and was a REGRESSION against the unordered listing it
replaced. So a cursor names the mode it belongs to, and a cursor from one mode is never
fed to the other.

Nothing here knows about Qdrant, HTTP or memory: it takes data and returns data, so the
delicate rules in this feature are provable without a server.
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

_CURSOR_VERSION = 2


class PagingError(CoreError):
    """A cursor that cannot be honoured, reported instead of restarting silently."""


def encode_cursor(value, seen_ids: list, *, ordered: bool = True) -> str:
    """An opaque cursor. `ordered=False` carries the server's own scroll offset in
    `value`, which is a point id and not a timestamp."""
    raw = json.dumps({"v": _CURSOR_VERSION, "ordered": bool(ordered),
                      "value": value, "seen": list(seen_ids)},
                     separators=(",", ":"), default=str).encode()

    return base64.urlsafe_b64encode(raw).decode()


def decode_cursor(cursor: str) -> tuple:
    """`(value, seen_ids, ordered)`, or `PagingError`.

    It refuses rather than falling back to the first page: a caller handed page 1 while
    believing it advanced re-reads what it already saw and never terminates.

    The ORDERED form validates that `value` is a timestamp, and that is not pedantry. A
    cursor whose value is any other string was accepted here, went out as
    `order_by.start_from`, and the server answered 400 ("Format error in JSON body:
    Expected a string, or an object with a key, direction and/or start_from"). That 400
    is indistinguishable from "there is no index", so the listing degraded and returned
    PAGE ONE labelled as a later page, blaming a missing index that existed. Measured
    against the real server on 2026-09-04, and the exact failure the spec forbids by
    name.
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
    ordered = payload.get("ordered")
    if not isinstance(seen, list) or not isinstance(ordered, bool) or value is None:
        raise PagingError("pagination cursor is malformed: omit it to start from the "
                          "newest")
    if ordered and _as_instant(value) is None:
        raise PagingError(
            f"pagination cursor does not carry a {ORDER_KEY} this archive could be "
            f"positioned at ({value!r}). It is a cursor for another archive or another "
            f"field: omit it to start from the newest."
        )

    return value, seen, ordered


def _as_instant(value) -> str | None:
    """`value` as a comparable instant, or None when it is not a timestamp.

    It lives HERE, and only here, because comparing boundary values is this module's job.
    An identical helper briefly existed in `memory` too, from a version where the
    comparison was expected to live there; it ended up never called, and a second copy of
    a rule this delicate is a rule that will drift.

    It converts to a COMMON ZONE, and that is the whole point. `fromisoformat(...)
    .isoformat()` round-trips the offset it was given, so `...T00:00:00+00:00` and
    `...T21:00:00-03:00` came back as two different strings for one moment and the tie
    at a page boundary went undetected: measured, a 4-record archive walked with mixed
    offsets returned 60 rows, repeated one record 16 times and never terminated. Handling
    only `Z` was handling only the pair the textual replace below already collapses.

    A NAIVE value is read as UTC, matching how the server reads one: Qdrant's `datetime`
    index stores an instant, so a stamp with no zone has to be assigned one somewhere,
    and assuming the local zone of whichever host happens to run the listing would make
    the comparison depend on the reader.

    The import is local so this module keeps depending on nothing but its error base,
    which is what lets the cursor rule be tested with no store, no config and no network.
    """
    from datetime import datetime, timezone

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        # `Z` is not accepted by `fromisoformat` before 3.11 and this package supports
        # older hosts, so it is normalized rather than relied upon.
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc).isoformat()


def page_request(cursor: str | None) -> dict:
    """What to ask the store for the page `cursor` points at.

    Returns `{"order_by", "filter", "offset", "ordered", "continuing"}`. `ordered` False
    means the caller must NOT send `order_by` (the store already said it cannot order) and
    must pass `offset` through, which is how an unordered scroll pages. `continuing` says
    whether this is a later page of a walk already in progress: falling back to an
    unordered scroll is only safe on the FIRST page, because an unordered scroll cannot be
    positioned at an ordered cursor and would re-deliver what earlier pages already gave.

    `filter` is None rather than an empty `must_not`: an empty clause is one the server
    still has to evaluate, and it reads as "something is excluded" to anyone debugging.
    """
    order_by = {"key": ORDER_KEY, "direction": "desc"}
    if cursor is None:
        return {"order_by": order_by, "filter": None, "offset": None,
                "ordered": True, "continuing": False}
    value, seen, ordered = decode_cursor(cursor)
    if not ordered:
        # An unordered continuation: the server's own scroll offset, nothing else.
        return {"order_by": None, "filter": None, "offset": value,
                "ordered": False, "continuing": True}
    order_by["start_from"] = value

    return {"order_by": order_by,
            "filter": {"must_not": [{"has_id": list(seen)}]} if seen else None,
            "offset": None, "ordered": True, "continuing": True}


def next_cursor(points: list[dict], limit: int, previous: str | None) -> str | None:
    """The cursor for the ORDERED page after `points`, or None when it was the last one.

    A page shorter than `limit` is the last page. That is the only termination signal
    available: the server's own `next_page_offset` is null on every ordered scroll, so
    trusting it would end every listing after page one.

    Ties are compared as INSTANTS, not as strings. Two spellings of one moment
    (`...T00:00:00Z` and `...T00:00:00+00:00`) are equal to the server and unequal to
    `==`, and the measured consequence of comparing text was a walk that returned 50 rows
    for 6 records, repeated one of them 18 times, never reached a seventh and never
    terminated.
    """
    if not points or len(points) < limit:
        return None
    boundary_raw = (points[-1].get("payload") or {}).get(ORDER_KEY)
    boundary = _as_instant(boundary_raw)
    if boundary is None:
        # Nothing to anchor `start_from` on. Such a record is invisible to an ordered
        # scroll anyway, so claiming "no more pages" is the honest end of the walk.
        return None
    seen = {p.get("id") for p in points
            if _as_instant((p.get("payload") or {}).get(ORDER_KEY)) == boundary}
    if previous is not None:
        prior_value, prior_seen, prior_ordered = decode_cursor(previous)
        # Only while the boundary has not moved. Carrying ids past their value would
        # grow the exclusion list without bound across a long listing.
        if prior_ordered and _as_instant(prior_value) == boundary:
            seen |= set(prior_seen)

    # `boundary_raw` and not `boundary`: the server has to be positioned with a value it
    # wrote, not with this module's normalization of it.
    return encode_cursor(boundary_raw, sorted(i for i in seen if i is not None),
                         ordered=True)


def unordered_cursor(server_offset) -> str | None:
    """The cursor for the next page of an UNORDERED walk, or None when it ended.

    An unordered scroll keeps the server's cursor, so degrading the ORDER must not also
    destroy the PAGINATION: forcing null here reduced a 749-record archive to whatever
    fitted in one page, while the tool description told the reader that a null
    `next_offset` means there is nothing more.
    """
    if server_offset is None:
        return None

    return encode_cursor(server_offset, [], ordered=False)
