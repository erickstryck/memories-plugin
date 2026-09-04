# Memory List Ordering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `memory_list` and `qctx memory list` actually return the newest records first, with honest reporting when the ordered path is unavailable, so the tool description stops promising something the code does not do.

**Architecture:** A new pure module `core/paging.py` owns the value-cursor rule (Qdrant refuses `order_by` together with `offset`, and returns `next_page_offset: null` when ordering, so the client must page by value plus seen-ids). `core/qdrant.py:scroll` gains a pass-through `order_by`; `core/ports.py` mirrors it; `MemoryStore.ensure` creates the `datetime` payload index and `MemoryStore.list_page` orchestrates, degrading to unordered WITH a stated warning when ordering cannot happen. The fake store learns to refuse ordering without an index so the fallback is exercised offline.

**Tech Stack:** Python 3 stdlib only (`unittest`, `json`, `base64`, `urllib` via the existing `core/http.py`). No new dependency. Qdrant 1.18.2 HTTP API.

**Spec:** `docs/superpowers/specs/2026-09-03-memory-list-ordering-design.md`

## Global Constraints

- **Stdlib only.** This package is imported by hooks that run on every user prompt; a missing dependency turns into silent loss of functionality (`core/qdrant.py` docstring). No `pip install` anywhere in this plan.
- **The adapter holds no business rule.** `core/qdrant.py` translates the contract into HTTP and nothing else. Cursor logic goes in `core/paging.py`, never in the adapter.
- **A read never creates.** `require_existing` exists so a typo cannot produce an empty collection and a false "there is no precedent" (`core/memory.py:145-158`). `list_page` must keep calling `require_existing`, and must never call `ensure`.
- **Decide by HTTP status, never by message substring.** `QdrantError` carries `.status` (`core/qdrant.py:43-53`); a proxy echoes upstream statuses into its body, so substring matching is wrong for a measured reason (`core/qdrant.py:19-34`).
- **Exact ordering key:** `updated_at`. Exact index schema string: `datetime`. Exact `order` values in the return: `"updated_at_desc"` and `"unordered"`.
- **No new tool.** `tests/test_readme_fidelity.py:105-107` pins every "N tools" phrase in the docs to `len(tools.SCHEMAS)`. This plan adds a PARAMETER to an existing tool, so the count must not change.
- **No spaced em-dash (`—`) in `README.md`, `docs/usage.md`, `docs/install.md`, `docs/architecture.md`.** Pinned by `tests/test_readme_fidelity.py:251`. (This plan only touches `docs/usage.md` among those, in Task 6.)
- **Commit style:** lowercase, imperative, descriptive, one commit per task. `feat:` for new behaviour, `test:` for test-only, `docs:` for prose.
- **Suite command:** `python3 -m unittest discover -s tests -t . -q` from the repo root. It must stay offline, and no task may add a failure.
- **Baseline, MEASURED and not assumed:** at `190e682` the suite runs **1297 tests, 19 skipped, with ONE failure** in 47s: `tests.test_hermes_provider.TestTheContract.test_it_answers_every_method_the_REAL_installed_abc_declares`. The installed hermes declares `pre_compress_checkpoint_api_version` (`~/.hermes/hermes-agent/agent/memory_manager.py:1103`) and `MemoriesProvider` does not answer it. That failure is **preexisting, unrelated to this work, and STRICTLY OUT OF SCOPE**: do not fix it, do not touch `hosts/hermes/__init__.py` for it, do not silence the test. The success criterion for every task is therefore **"no failure other than that one"**, not "OK". If a second failure appears, it is yours.
- **Do not run the suite from a `git archive` export.** Measured: the same commit exported to `/tmp` fails 9 tests instead of 1, because this working copy is the tree that `~/.hermes/plugins/memories` symlinks to and several tests read the real install. Run it in the repo.

---

### Task 1: The value-cursor rule, as a pure module

**Files:**
- Create: `core/paging.py`
- Test: `tests/test_paging.py`

**Interfaces:**
- Consumes: nothing. This module imports only `base64`, `json` and `core.errors.CoreError`.
- Produces, and every later task uses exactly these names:
  - `PagingError(CoreError)`
  - `ORDER_KEY = "updated_at"`, `ORDER_DESC = "updated_at_desc"`, `ORDER_NONE = "unordered"`, `INDEX_SCHEMA = "datetime"`
  - `encode_cursor(value: str, seen_ids: list) -> str`
  - `decode_cursor(cursor: str) -> tuple[str, list]`
  - `page_request(cursor: str | None) -> dict` returning `{"order_by": {...}, "filter": {...} | None}`
  - `next_cursor(points: list[dict], limit: int, previous: str | None) -> str | None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_paging.py`:

```python
"""The value-cursor rule, with no Qdrant and no fake store.

Qdrant refuses `offset` together with `order_by` and returns `next_page_offset: null`
whenever it orders (both measured 2026-09-03 against 1.18.2), so paging an ordered
listing is the CLIENT's job. That rule lives here, alone, because it is the only
delicate part and it must be provable without a server.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import paging  # noqa: E402


def point(pid, value):
    return {"id": pid, "payload": {"updated_at": value, "document": f"doc {pid}"}}


class TestTheFirstPage(unittest.TestCase):
    def test_no_cursor_asks_for_descending_order_and_no_filter(self):
        req = paging.page_request(None)
        self.assertEqual(req["order_by"], {"key": "updated_at", "direction": "desc"})
        self.assertIsNone(req["filter"], "the first page excludes nothing")


class TestTheCursorRoundTrip(unittest.TestCase):
    def test_a_cursor_carries_the_value_and_the_ids_seen_at_it(self):
        cursor = paging.encode_cursor("2026-05-05T10:00:00+00:00", ["a", "b"])
        self.assertEqual(paging.decode_cursor(cursor),
                         ("2026-05-05T10:00:00+00:00", ["a", "b"]))

    def test_the_cursor_is_opaque_text_that_survives_json(self):
        import json
        cursor = paging.encode_cursor("2026-05-05T10:00:00+00:00", ["a"])
        self.assertIsInstance(cursor, str)
        self.assertEqual(json.loads(json.dumps({"c": cursor}))["c"], cursor)

    def test_a_corrupt_cursor_is_refused_by_name_instead_of_silently_restarting(self):
        """Silently returning page 1 for a bad cursor is the worst option: the caller
        believes it advanced and re-reads what it already saw."""
        for bad in ("not-base64!!", "", "eyJub3RfbWluZSI6IDF9"):
            with self.subTest(cursor=bad):
                with self.assertRaises(paging.PagingError):
                    paging.decode_cursor(bad)


class TestTheNextPageRequest(unittest.TestCase):
    def test_a_cursor_becomes_start_from_plus_an_id_exclusion(self):
        cursor = paging.encode_cursor("2026-05-05T10:00:00+00:00", ["a", "b"])
        req = paging.page_request(cursor)
        self.assertEqual(req["order_by"], {"key": "updated_at", "direction": "desc",
                                           "start_from": "2026-05-05T10:00:00+00:00"})
        self.assertEqual(req["filter"], {"must_not": [{"has_id": ["a", "b"]}]})

    def test_no_ids_to_exclude_means_no_filter_at_all(self):
        """An empty `must_not` is a filter Qdrant has to evaluate for nothing."""
        req = paging.page_request(paging.encode_cursor("2026-05-05T10:00:00+00:00", []))
        self.assertIsNone(req["filter"])


class TestDerivingTheNextCursor(unittest.TestCase):
    def test_a_short_page_is_the_last_page(self):
        points = [point("a", "2026-09-01T00:00:00+00:00")]
        self.assertIsNone(paging.next_cursor(points, limit=5, previous=None))

    def test_an_empty_page_is_the_last_page(self):
        self.assertIsNone(paging.next_cursor([], limit=5, previous=None))

    def test_a_full_page_hands_back_the_last_value_and_the_ids_at_it(self):
        points = [point("a", "2026-09-01T00:00:00+00:00"),
                  point("b", "2026-05-05T10:00:00+00:00")]
        value, seen = paging.decode_cursor(
            paging.next_cursor(points, limit=2, previous=None))
        self.assertEqual(value, "2026-05-05T10:00:00+00:00")
        self.assertEqual(seen, ["b"], "only the ids AT the boundary value matter")

    def test_every_id_sharing_the_boundary_value_is_carried(self):
        """`store_many` writes N records with ONE timestamp, so ties are the normal
        case, not the exotic one. Measured against 1.18.2: `start_from` re-includes the
        boundary value, so without excluding the ids seen at it the next page repeats
        them."""
        points = [point("a", "2026-09-01T00:00:00+00:00"),
                  point("b", "2026-05-05T10:00:00+00:00"),
                  point("c", "2026-05-05T10:00:00+00:00")]
        value, seen = paging.decode_cursor(
            paging.next_cursor(points, limit=3, previous=None))
        self.assertEqual(value, "2026-05-05T10:00:00+00:00")
        self.assertEqual(sorted(seen), ["b", "c"])

    def test_ids_already_seen_at_the_same_value_accumulate(self):
        """Page 2 of a long tie must not forget what page 1 excluded, or the pages
        alternate between the same two records forever."""
        previous = paging.encode_cursor("2026-05-05T10:00:00+00:00", ["a"])
        points = [point("b", "2026-05-05T10:00:00+00:00"),
                  point("c", "2026-05-05T10:00:00+00:00")]
        value, seen = paging.decode_cursor(
            paging.next_cursor(points, limit=2, previous=previous))
        self.assertEqual(value, "2026-05-05T10:00:00+00:00")
        self.assertEqual(sorted(seen), ["a", "b", "c"])

    def test_a_new_boundary_value_drops_the_old_exclusions(self):
        """Carrying stale ids past their value would grow the filter without bound."""
        previous = paging.encode_cursor("2026-09-01T00:00:00+00:00", ["a"])
        points = [point("b", "2026-07-01T00:00:00+00:00"),
                  point("c", "2026-05-05T10:00:00+00:00")]
        value, seen = paging.decode_cursor(
            paging.next_cursor(points, limit=2, previous=previous))
        self.assertEqual(value, "2026-05-05T10:00:00+00:00")
        self.assertEqual(seen, ["c"])

    def test_a_page_whose_last_point_has_no_value_stops_paging(self):
        """A record with no `updated_at` cannot anchor `start_from`. It is invisible to
        an ordered scroll anyway (measured), so reaching one means the cursor cannot be
        built and saying "no more pages" beats inventing a boundary."""
        self.assertIsNone(paging.next_cursor([{"id": "a", "payload": {}}],
                                             limit=1, previous=None))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python3 -m unittest tests.test_paging -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'core.paging'`.

- [ ] **Step 3: Write the implementation**

Create `core/paging.py`:

```python
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
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `python3 -m unittest tests.test_paging -v`
Expected: PASS, 13 tests.

- [ ] **Step 5: Run the whole suite and confirm the baseline**

Run: `python3 -m unittest discover -s tests -t . -q`
Expected: 1310 tests, the SAME single preexisting failure named in Global Constraints, and nothing else red. If a second failure shows up, it came from this task.

- [ ] **Step 6: Commit**

```bash
git add core/paging.py tests/test_paging.py
git commit -m "feat: the value-cursor rule for an ordered listing, as a pure module"
```

---

### Task 2: `scroll` carries `order_by` to the server

**Files:**
- Modify: `core/qdrant.py:168-186` (`Qdrant.scroll`)
- Modify: `core/ports.py:111-113` (`VectorStore.scroll`)
- Modify: `tests/fakes.py:112-118` (`FakeVectorStore.scroll`), and `tests/fakes.py:48-49` (`ensure_payload_index`) to record the index
- Test: `tests/test_paging.py` (new class appended)

**Interfaces:**
- Consumes: `paging.ORDER_KEY`, `paging.INDEX_SCHEMA` from Task 1.
- Produces: `scroll(name, limit=..., offset=None, with_vector=False, filter_=None, payload_fields=None, order_by=None)` on both the real adapter and the fake. `FakeVectorStore` gains `self.indexes: dict[str, set]` mapping collection name to indexed field names, and raises `FakeQdrantRefusal` (a `ValueError` subclass carrying `.status = 400`) when asked to order by an unindexed field.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_paging.py`, before `if __name__ == "__main__":`:

```python
class TestTheFakeStoreIsAsPoorAsTheRealOne(unittest.TestCase):
    """The fake must REFUSE what Qdrant refuses.

    Measured: ordering by a payload key with no range index answers HTTP 400 ("No range
    index for `order_by` key"). A fake that happily orders anyway makes the degradation
    path unreachable offline, and the degradation path is the entire honesty of this
    feature. `tests/fakes.py` already argues this for `payload_fields`.
    """

    def setUp(self):
        from tests.fakes import FakeVectorStore
        self.q = FakeVectorStore()
        self.q.ensure_collection("mem", 8)
        for pid, value in (("a", "2026-01-01T00:00:00+00:00"),
                           ("b", "2026-09-01T00:00:00+00:00"),
                           ("c", "2026-05-05T10:00:00+00:00")):
            self.q.upsert("mem", [{"id": pid, "vector": [0.0] * 8,
                                   "payload": {"updated_at": value,
                                               "document": f"doc {pid}"}}])

    def test_ordering_without_an_index_is_refused_the_way_the_server_refuses_it(self):
        from tests.fakes import FakeQdrantRefusal
        with self.assertRaises(FakeQdrantRefusal) as caught:
            self.q.scroll("mem", limit=10,
                          order_by={"key": "updated_at", "direction": "desc"})
        self.assertEqual(caught.exception.status, 400,
                         "the caller decides by status, so the fake has to carry one")

    def test_with_the_index_it_orders_newest_first(self):
        self.q.ensure_payload_index("mem", "updated_at", "datetime")
        points, _ = self.q.scroll("mem", limit=10,
                                  order_by={"key": "updated_at", "direction": "desc"})
        self.assertEqual([p["id"] for p in points], ["b", "c", "a"])

    def test_start_from_is_inclusive_like_the_server(self):
        self.q.ensure_payload_index("mem", "updated_at", "datetime")
        points, _ = self.q.scroll("mem", limit=10,
                                  order_by={"key": "updated_at", "direction": "desc",
                                            "start_from": "2026-05-05T10:00:00+00:00"})
        self.assertEqual([p["id"] for p in points], ["c", "a"],
                         "the boundary record comes back, which is why ids are excluded")

    def test_an_ordered_scroll_returns_no_server_offset(self):
        self.q.ensure_payload_index("mem", "updated_at", "datetime")
        _, offset = self.q.scroll("mem", limit=1,
                                  order_by={"key": "updated_at", "direction": "desc"})
        self.assertIsNone(offset, "measured: next_page_offset is null when ordering")

    def test_a_record_without_the_key_is_invisible_to_an_ordered_scroll(self):
        self.q.ensure_payload_index("mem", "updated_at", "datetime")
        self.q.upsert("mem", [{"id": "d", "vector": [0.0] * 8,
                               "payload": {"document": "undated"}}])
        points, _ = self.q.scroll("mem", limit=10,
                                  order_by={"key": "updated_at", "direction": "desc"})
        self.assertNotIn("d", [p["id"] for p in points])
        plain, _ = self.q.scroll("mem", limit=10)
        self.assertIn("d", [p["id"] for p in plain], "and it is still THERE")

    def test_an_unordered_scroll_behaves_exactly_as_before(self):
        points, offset = self.q.scroll("mem", limit=2)
        self.assertEqual(len(points), 2)
        self.assertIsNone(offset)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python3 -m unittest tests.test_paging.TestTheFakeStoreIsAsPoorAsTheRealOne -v`
Expected: FAIL, `ImportError: cannot import name 'FakeQdrantRefusal'`.

- [ ] **Step 3: Teach the fake to refuse**

In `tests/fakes.py`, add the exception right before `class FakeVectorStore`:

```python
class FakeQdrantRefusal(ValueError):
    """What the real server answers, in the shape callers branch on.

    A `ValueError` and not a `QdrantError` on purpose: the fakes import nothing from
    `core`, and the property under test is that the CALLER decides by `.status`.
    """

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status
```

In `FakeVectorStore.__init__`, add the index registry next to `self.calls`:

```python
        self.indexes: dict[str, set] = {}        # collection -> indexed payload fields
```

Replace `ensure_payload_index` (`tests/fakes.py:48-49`) with:

```python
    def ensure_payload_index(self, name: str, field: str, schema: str) -> None:
        self.calls.append(("ensure_payload_index", name, field))
        self.indexes.setdefault(name, set()).add(field)
```

Replace `scroll` (`tests/fakes.py:112-118`) with:

```python
    def scroll(self, name: str, limit: int = 256, offset=None,
               with_vector: bool = False, filter_: dict | None = None,
               payload_fields: list[str] | None = None,
               order_by: dict | None = None):
        """REFUSES to order by an unindexed field, because the server does.

        Measured on 1.18.2: `order_by` without a range index answers HTTP 400, ordering
        excludes records that lack the key entirely, `start_from` INCLUDES the boundary
        value, and `next_page_offset` comes back null on any ordered scroll. A fake that
        is kinder than that hides the degradation path this feature is built around.
        """
        items = [{"id": pid, "payload": p.get("payload", {})}
                 for pid, p in self.collections.get(name, {}).get("points", {}).items()
                 if not filter_ or _matches_filter(p.get("payload", {}), filter_)]
        if order_by is None:
            return items[:limit], None
        key = order_by.get("key")
        if key not in self.indexes.get(name, set()):
            raise FakeQdrantRefusal(
                f"Wrong input: No range index for `order_by` key: `{key}`. Please "
                f"create one to use `order_by`.")
        items = [i for i in items if i["payload"].get(key) is not None]
        items.sort(key=lambda i: i["payload"][key],
                   reverse=order_by.get("direction", "asc") == "desc")
        start = order_by.get("start_from")
        if start is not None:
            items = [i for i in items if i["payload"][key] <= start]

        return items[:limit], None
```

Note for the implementer: `_matches_filter` (`tests/fakes.py:144`) supports `must` with
`match`/`range` only. The ordered path needs `must_not` with `has_id`, so extend it:

```python
def _matches_filter(payload: dict, filter_: dict) -> bool:
    """Supports the shapes the package uses: `must` with `match.value` or `range`, and
    `must_not` with `has_id` (the ordered listing's tie exclusion)."""
    for cond in filter_.get("must", []):
        key = cond.get("key")
        value = payload.get(key)
        if "match" in cond:
            if value != cond["match"].get("value"):
                return False
        elif "range" in cond:
            range_ = cond["range"]
            if value is None:
                return False
            if "gte" in range_ and not value >= range_["gte"]:
                return False
            if "lte" in range_ and not value <= range_["lte"]:
                return False
            if "gt" in range_ and not value > range_["gt"]:
                return False
            if "lt" in range_ and not value < range_["lt"]:
                return False

    return True
```

**Read the current body before editing** and keep every existing `range` branch exactly
as it is; the block above shows the shape, not a licence to drop a comparison. The
`must_not`/`has_id` handling belongs in `FakeVectorStore.scroll`, which knows the point
ids, so add it there instead of in this helper:

```python
        excluded = {i for cond in (filter_ or {}).get("must_not", [])
                    for i in cond.get("has_id", [])}
        if excluded:
            items = [i for i in items if i["id"] not in excluded]
```

Place that immediately after `items` is first built in `scroll`, so it applies to the
ordered and unordered paths alike.

- [ ] **Step 4: Run the fake's tests**

Run: `python3 -m unittest tests.test_paging -v`
Expected: PASS, 19 tests.

- [ ] **Step 5: Pass `order_by` through the real adapter and the contract**

In `core/qdrant.py`, change the signature and body of `scroll` (`core/qdrant.py:168-186`)
so the new parameter reaches the request body, leaving the existing docstring paragraph
about `payload_fields` untouched and adding one about ordering:

```python
    def scroll(self, name: str, limit: int = 256, offset=None,
               with_vector: bool = False, filter_: dict | None = None,
               payload_fields: list[str] | None = None,
               order_by: dict | None = None) -> tuple[list[dict], object]:
        """`payload_fields` names the payload keys to fetch; None means the whole payload.

        Naming them matters for anything that runs on a loop. The watcher polls this every few
        seconds only to read a handful of `metadata` fields, and the full payload carries the
        CHUNK TEXT — so the cheap half of the change check was pulling the repository's entire
        indexed content over the network on every cycle.

        `order_by` is handed to the server as it comes: it needs a range index on the key,
        it cannot be combined with `offset` (the server refuses both together), and it makes
        `next_page_offset` null even when more pages exist. Those are the caller's problem
        to reason about, in `core/paging.py`; no rule about them lives here.
        """
        body = {"limit": limit, "with_vector": with_vector,
                "with_payload": payload_fields if payload_fields else True}
        if offset is not None:
            body["offset"] = offset
        if filter_:
            body["filter"] = filter_
        if order_by:
            body["order_by"] = order_by
        res = self.request("POST", f"/collections/{name}/points/scroll", body).get("result", {})

        return res.get("points", []), res.get("next_page_offset")
```

In `core/ports.py`, extend the contract (`core/ports.py:111-113`):

```python
    def scroll(self, name: str, limit: int = ..., offset=...,
               with_vector: bool = ..., filter_: dict | None = ...,
               payload_fields: list[str] | None = ...,
               order_by: dict | None = ...) -> tuple[list[dict], object]:
        """`order_by` orders the walk by a payload key; it requires a range index on that
        key and is mutually exclusive with `offset`. An implementation that cannot order
        must RAISE rather than return an unordered page as if it had ordered."""
        ...
```

- [ ] **Step 6: Run the whole suite**

Run: `python3 -m unittest discover -s tests -t . -q`
Expected: 1316 tests (1297 baseline plus the 19 from Tasks 1 and 2), and only the one preexisting failure.

- [ ] **Step 7: Commit**

```bash
git add core/qdrant.py core/ports.py tests/fakes.py tests/test_paging.py
git commit -m "feat: scroll can order by a payload key, and the fake refuses like the server"
```

---

### Task 3: `ensure` creates the index; `list_page` orders and says so

**Files:**
- Modify: `core/memory.py:141-143` (`ensure`) and `core/memory.py:374-384` (`list_page`)
- Test: `tests/test_memory_offline.py` (new class appended before `if __name__ == "__main__":`)

**Interfaces:**
- Consumes: everything Task 1 produced, plus `order_by=` on `scroll` from Task 2.
- Produces: `MemoryStore.list_page(limit: int = 20, offset: str | None = None) -> dict`
  returning `{"count", "memories", "next_offset", "order"}` and, only when degraded,
  `"warning"`. `order` is `paging.ORDER_DESC` or `paging.ORDER_NONE`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_memory_offline.py`:

```python
class TestTheListingIsOrderedAndHonest(unittest.TestCase):
    """The defect this fixes: the tool promised "newest page first" and returned id order.

    So the property is not only "it sorts" but "it never claims to have sorted when it
    did not". A listing that quietly falls back is the same defect wearing a new coat.
    """

    def _archive(self):
        s, q, _ = store()
        # Written oldest-first so insertion order cannot be mistaken for date order.
        for text, when in (("oldest fact", "2026-01-01T00:00:00+00:00"),
                           ("middle fact", "2026-05-05T10:00:00+00:00"),
                           ("newest fact", "2026-09-01T00:00:00+00:00")):
            mid = s.store(text)["id"]
            point = q.get_point("mem", mid)
            payload = dict(point["payload"], created_at=when, updated_at=when)
            q.set_payload("mem", mid, payload)

        return s, q

    def test_the_first_page_is_the_newest_records(self):
        s, _ = self._archive()
        page = s.list_page(limit=2)
        self.assertEqual([m["document"] for m in page["memories"]],
                         ["newest fact", "middle fact"])

    def test_it_reports_the_order_it_actually_used(self):
        s, _ = self._archive()
        self.assertEqual(s.list_page(limit=2)["order"], "updated_at_desc")

    def test_the_cursor_walks_the_whole_archive_once(self):
        s, _ = self._archive()
        seen, cursor, pages = [], None, 0
        while pages < 10:
            page = s.list_page(limit=2, offset=cursor)
            seen.extend(m["document"] for m in page["memories"])
            cursor = page["next_offset"]
            pages += 1
            if cursor is None:
                break
        self.assertIsNone(cursor, "the walk has to terminate")
        self.assertEqual(seen, ["newest fact", "middle fact", "oldest fact"],
                         "every record exactly once, newest first")

    def test_a_write_creates_the_index_the_ordering_needs(self):
        s, q, _ = store()
        s.store("a fact")
        self.assertIn("updated_at", q.indexes.get("mem", set()))
        self.assertIn(("ensure_payload_index", "mem", "updated_at"), q.calls)

    def test_a_listing_on_an_unindexed_archive_still_comes_back_ordered(self):
        """The archive predates this feature: 744 records and no payload index at all,
        measured on 2026-09-03. The first listing has to fix that itself."""
        s, q = self._archive()
        q.indexes.pop("mem", None)
        page = s.list_page(limit=2)
        self.assertEqual(page["order"], "updated_at_desc")
        self.assertEqual([m["document"] for m in page["memories"]],
                         ["newest fact", "middle fact"])

    def test_when_ordering_cannot_be_had_it_degrades_AND_SAYS_SO(self):
        s, q = self._archive()

        def refuse(*a, **kw):
            from tests.fakes import FakeQdrantRefusal
            raise FakeQdrantRefusal("No range index for `order_by` key: `updated_at`.")

        q.ensure_payload_index = lambda *a, **kw: None      # creating it does nothing
        original = q.scroll
        q.scroll = lambda *a, **kw: (refuse() if kw.get("order_by") else original(*a, **kw))
        page = s.list_page(limit=2)
        self.assertEqual(page["order"], "unordered")
        self.assertIn("warning", page)
        self.assertEqual(len(page["memories"]), 2, "it still answers")

    def test_the_degraded_warning_names_the_reason_not_just_the_word_warning(self):
        s, q = self._archive()

        def refuse(*a, **kw):
            from tests.fakes import FakeQdrantRefusal
            raise FakeQdrantRefusal("No range index for `order_by` key: `updated_at`.")

        q.ensure_payload_index = lambda *a, **kw: None
        original = q.scroll
        q.scroll = lambda *a, **kw: (refuse() if kw.get("order_by") else original(*a, **kw))
        warning = s.list_page(limit=2)["warning"].lower()
        self.assertIn("order", warning)
        self.assertRegex(warning, r"(index|recency|newest)")

    def test_a_read_never_creates_the_collection(self):
        """`require_existing` is the rule; creating an index must not have smuggled an
        `ensure_collection` into the read path."""
        s, q, _ = store(collection="mem")
        q.collections.clear()
        with self.assertRaises(MemoryStoreError):
            s.list_page()
        self.assertEqual(q.list_collections(), [])

    def test_a_corrupt_cursor_is_refused_rather_than_restarting(self):
        from core.paging import PagingError
        s, _ = self._archive()
        with self.assertRaises(PagingError):
            s.list_page(limit=2, offset="not-a-cursor!!")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python3 -m unittest tests.test_memory_offline.TestTheListingIsOrderedAndHonest -v`
Expected: FAIL. `test_it_reports_the_order_it_actually_used` fails with `KeyError: 'order'`,
and `test_the_first_page_is_the_newest_records` fails on the id-ordered list.

- [ ] **Step 3: Write the implementation**

In `core/memory.py`, add `paging` to the existing package import at line 25:

```python
from . import paging, ports, retrieval
```

Replace `ensure` (`core/memory.py:141-143`):

```python
    def ensure(self) -> None:
        """Ensures the collection exists AND that the listing can order by date.

        The index is created here, on the WRITE path, and not where the listing needs it:
        `require_existing` exists precisely so a read creates nothing (see below), and a
        payload index is a write against the collection. `ensure_payload_index` swallows
        its own failure by design, so this can never break a store.

        Measured on 2026-09-03: the index is RETROACTIVE. The user's archive of 744
        records, written long before this existed, orders correctly the moment the index
        is created, with no reindexing and no payload migration.
        """
        self.q.ensure_collection(self.collection, self.vector_size)
        self.q.ensure_payload_index(self.collection, paging.ORDER_KEY,
                                    paging.INDEX_SCHEMA)
```

Replace `list_page` (`core/memory.py:374-384`):

```python
    def list_page(self, limit: int = 20, offset: str | None = None) -> dict:
        """A page of the archive, NEWEST FIRST, saying which order it actually used.

        The promise and the behaviour used to disagree: the tool description said "newest
        page first" while the scroll ran with no `order_by` at all, so pages came back in
        uuid4 order. Someone checking whether a fresh write landed concluded it had not.

        `offset` is the opaque cursor a previous call returned, NOT a point id: ordering
        and Qdrant's own `offset` are mutually exclusive, so paging is by value plus the
        ids already seen at that value. `core/paging.py` owns that rule.

        `order` is part of the answer, always. When ordering is impossible the listing
        still answers, unordered, and says so with a `warning`, because degrading in silence
        would be the original defect in a new shape.
        """
        self.require_existing()
        request = paging.page_request(offset)
        points, warning = self._scroll_ordered(limit, request)
        memories = [{
            "id": pt.get("id"),
            "document": pt.get("payload", {}).get("document"),
            "metadata": pt.get("payload", {}).get("metadata", {}),
            "updated_at": pt.get("payload", {}).get("updated_at"),
        } for pt in points]
        page = {"count": len(memories), "memories": memories,
                "order": paging.ORDER_NONE if warning else paging.ORDER_DESC,
                "next_offset": (None if warning
                                else paging.next_cursor(points, limit, offset))}
        if warning:
            page["warning"] = warning

        return page

    def _scroll_ordered(self, limit: int, request: dict) -> tuple[list, str | None]:
        """The ordered page, or the unordered one plus the sentence that admits it.

        Three attempts at most, and the middle one is the whole point: an archive written
        before this feature has no payload index, so the FIRST ordered listing on it is
        expected to fail. Creating the index and retrying once turns that into a
        self-healing read rather than a permanent degradation.

        The retry is decided by nothing but the failure itself, and the fallback message
        never quotes the server's text: `_is_absent` in `core/qdrant.py` records what
        reading error bodies costs: a proxy echoes upstream statuses into its own body,
        so behaviour keyed to message text is wrong for a measured reason.
        """
        try:
            return self.q.scroll(self.collection, limit=limit,
                                 filter_=request["filter"],
                                 order_by=request["order_by"])[0], None
        except CoreError:
            raise
        except Exception:
            pass
        # The index is missing on an archive nobody has written to since this shipped.
        # Creating it is a write, but a write of SCHEMA, not of anyone's data.
        self.q.ensure_payload_index(self.collection, paging.ORDER_KEY,
                                    paging.INDEX_SCHEMA)
        try:
            return self.q.scroll(self.collection, limit=limit,
                                 filter_=request["filter"],
                                 order_by=request["order_by"])[0], None
        except CoreError:
            raise
        except Exception:
            points, _ = self.q.scroll(self.collection, limit=limit)

            return points, (
                "this page is NOT ordered by recency: the archive could not be ordered "
                f"by {paging.ORDER_KEY} (the payload index for it is missing and could "
                "not be created), so the records come back in arbitrary id order. Do "
                "not read the first page as the newest records."
            )
```

Note for the implementer: `CoreError` is already imported at `core/memory.py:26`. The
`except CoreError: raise` before the broad `except` is deliberate and is the one thing in
this method not to simplify: a collection that vanished, an auth failure or an
unreachable server are `CoreError`s that must reach the caller as failures, and only the
store's refusal to order may be absorbed into degradation. If you find yourself wanting
to catch `QdrantError` specifically, do not: `core/memory.py` talks to `ports.VectorStore`
and must not import the Qdrant adapter (see the module docstring in `core/ports.py`).

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `python3 -m unittest tests.test_memory_offline -v`
Expected: PASS, including the 9 new tests.

- [ ] **Step 5: Run the whole suite**

Run: `python3 -m unittest discover -s tests -t . -q`
Expected: only the one preexisting failure. `tests/test_hermes_tools.py:658` (`test_list_pages_the_archive`) asserts
only `count` and `len(memories)`, so it keeps passing; if anything else fails, read it
before changing it.

- [ ] **Step 6: Commit**

```bash
git add core/memory.py tests/test_memory_offline.py
git commit -m "feat: the memory listing is ordered by recency, and reports when it is not"
```

---

### Task 4: the hermes tool accepts and returns the cursor

**Files:**
- Modify: `hosts/hermes/tools.py:312-313` (`_memory_list`) and `hosts/hermes/tools.py:554-567` (the `memory_list` schema)
- Test: `tests/test_hermes_tools.py` (two tests appended to `TestMemoryTools`, one to `TestSchemas`)

**Interfaces:**
- Consumes: `MemoryStore.list_page(limit, offset)` from Task 3.
- Produces: `memory_list` accepting `{limit?: integer, offset?: string}`.

- [ ] **Step 1: Write the failing test**

Append to `class TestMemoryTools` in `tests/test_hermes_tools.py`:

```python
    def test_list_returns_the_newest_first_and_names_the_order(self):
        for i in range(3):
            self.call("memory_store", information=f"fact {i}")
        res = self.call("memory_list", limit=2)
        self.assertEqual(res["order"], "updated_at_desc")
        self.assertEqual(res["count"], 2)

    def test_list_accepts_the_cursor_it_handed_out(self):
        """`next_offset` was decorative: nothing could send it back, so page 2 did not
        exist on any surface. A cursor no caller can use is not pagination."""
        for i in range(5):
            self.call("memory_store", information=f"fact {i}")
        first = self.call("memory_list", limit=2)
        self.assertIsNotNone(first["next_offset"])
        second = self.call("memory_list", limit=2, offset=first["next_offset"])
        self.assertEqual(second["count"], 2)
        first_ids = {m["id"] for m in first["memories"]}
        second_ids = {m["id"] for m in second["memories"]}
        self.assertEqual(first_ids & second_ids, set(), "page 2 repeated page 1")

    def test_a_bad_cursor_is_reported_as_an_error_the_model_can_act_on(self):
        res = self.call("memory_list", offset="garbage!!")
        self.assertIn("error", res)
        self.assertIn("cursor", res["error"].lower())
```

Append to `class TestSchemas`:

```python
    def test_the_list_description_matches_what_the_listing_does(self):
        """It claimed "newest page first" while the scroll had no `order_by` at all, so
        pages came back in uuid4 order and a fresh write looked absent. The description
        and the ordering are asserted together so neither can drift alone."""
        by_name = {s["name"]: s for s in tools.SCHEMAS}
        listing = by_name["memory_list"]
        self.assertIn("offset", listing["parameters"]["properties"],
                      "a cursor the caller cannot send back is not pagination")
        text = listing["description"].lower()
        self.assertIn("newest", text)
        self.assertIn("order", text, "it has to name the field it reports the order in")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python3 -m unittest tests.test_hermes_tools.TestSchemas.test_the_list_description_matches_what_the_listing_does -v`
Expected: FAIL, `AssertionError: 'offset' not found in {'limit': ...}`.

- [ ] **Step 3: Write the implementation**

Replace `_memory_list` (`hosts/hermes/tools.py:312-313`):

```python
def _memory_list(args: dict, cfg) -> str:
    # `_text` and not `_require`: the first page has no cursor, and a blank string from a
    # model that filled the field with nothing means "no cursor", not "cursor of empty".
    return _ok(_memory(cfg).list_page(_int(args, "limit", 20),
                                      _text(args, "offset")))
```

Replace the `memory_list` schema (`hosts/hermes/tools.py:554-567`):

```python
    {
        "name": "memory_list",
        "description": ("Page through the archive without a query, newest first (by "
                        "`updated_at`). Use it to inspect or audit what is stored, "
                        "never to answer a question, which is what memory_recall is "
                        "for. Every page states the order it actually used in `order`: "
                        "`updated_at_desc` is ordered by recency, `unordered` means the "
                        "archive could not be ordered and the page carries a `warning` "
                        "saying so, in which case the first page is NOT the newest "
                        "records. To read the next page, send back the `next_offset` "
                        "this call returns; a null `next_offset` means there is no "
                        "next page."),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer",
                          "description": "Records in this page (default 20)."},
                "offset": {"type": "string",
                           "description": ("The `next_offset` from a previous "
                                           "memory_list call, to continue where it "
                                           "stopped. Omit it for the newest page. It is "
                                           "an opaque cursor, not a record id.")},
            },
            "required": [],
        },
    },
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `python3 -m unittest tests.test_hermes_tools -v`
Expected: PASS. `test_every_description_says_when_to_use_it` and
`test_every_property_carries_a_description` cover the new text automatically.

- [ ] **Step 5: Run the whole suite**

Run: `python3 -m unittest discover -s tests -t . -q`
Expected: only the one preexisting failure.

- [ ] **Step 6: Commit**

```bash
git add hosts/hermes/tools.py tests/test_hermes_tools.py
git commit -m "feat: memory_list takes the cursor back and describes the order it used"
```

---

### Task 5: the CLI can ask for page 2

**Files:**
- Modify: `cli/qctx.py:1089-1090` (`cmd_memory_list`) and `cli/qctx.py:1625-1627` (the `list` subparser)
- Test: `tests/test_cli_render.py` (new class appended before `if __name__ == "__main__":`)

**Interfaces:**
- Consumes: `MemoryStore.list_page(limit, offset)` from Task 3.
- Produces: `qctx memory list [--limit N] [--offset CURSOR]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_render.py`:

```python
class TestTheMemoryListingPages(unittest.TestCase):
    """The CLI is the claude-code surface for this command (`skills/memory/SKILL.md`),
    so a cursor only the hermes tool can send would leave one host without page 2."""

    def test_the_parser_accepts_an_offset(self):
        cli = load_cli()
        args = cli.build_parser().parse_args(
            ["memory", "list", "--limit", "2", "--offset", "abc"])
        self.assertEqual(args.limit, 2)
        self.assertEqual(args.offset, "abc")

    def test_omitting_it_asks_for_the_newest_page(self):
        cli = load_cli()
        args = cli.build_parser().parse_args(["memory", "list"])
        self.assertIsNone(args.offset)

    def test_the_handler_forwards_both_arguments(self):
        import unittest.mock
        cli = load_cli()
        seen = {}

        class Store:
            def list_page(self, limit, offset=None):
                seen.update(limit=limit, offset=offset)

                return {"count": 0, "memories": [], "next_offset": None,
                        "order": "updated_at_desc"}

        with unittest.mock.patch.object(cli.core, "build_memory",
                                        lambda cfg, **kw: Store()):
            rendered(cli.cmd_memory_list,
                     type("A", (), {"limit": 3, "offset": "cur", "json": True})(), None)
        self.assertEqual(seen, {"limit": 3, "offset": "cur"})
```

Note for the implementer: `load_cli` and `rendered` are module-level helpers already in
`tests/test_cli_render.py`. Read their definitions at the top of that file before use;
`rendered(fn, args, cfg)` captures stdout and returns it.

- [ ] **Step 2: Run it to make sure it fails**

Run: `python3 -m unittest tests.test_cli_render.TestTheMemoryListingPages -v`
Expected: FAIL, `AttributeError: 'Namespace' object has no attribute 'offset'`.

- [ ] **Step 3: Write the implementation**

Replace `cmd_memory_list` (`cli/qctx.py:1089-1090`):

```python
def cmd_memory_list(args, cfg):
    output(core.build_memory(cfg).list_page(args.limit, args.offset), True)
```

Add the flag to the `list` subparser (`cli/qctx.py:1625-1627`):

```python
    p = memsub.add_parser("list")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--offset", default=None,
                   help="the next_offset from a previous listing, to continue it")
    p.set_defaults(fn=cmd_memory_list)
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `python3 -m unittest tests.test_cli_render -v`
Expected: PASS.

- [ ] **Step 5: Run the whole suite**

Run: `python3 -m unittest discover -s tests -t . -q`
Expected: only the one preexisting failure.

- [ ] **Step 6: Commit**

```bash
git add cli/qctx.py tests/test_cli_render.py
git commit -m "feat: qctx memory list continues from a cursor"
```

---

### Task 6: prove it against the real Qdrant, and correct the docs

**Files:**
- Modify: `tests/test_integration.py` (new class appended before `if __name__ == "__main__":`)
- Modify: `docs/usage.md:23` (the `memory list` row)
- Modify: `skills/memory/SKILL.md:162` (the `list` row)

**Interfaces:**
- Consumes: everything above. Produces nothing for later tasks; this is the last one.

- [ ] **Step 1: Write the integration test**

Append to `tests/test_integration.py`:

```python
@unittest.skipUnless(ENABLED, "set QCTX_INTEGRATION=1")
class TestTheListingOrdersAgainstTheRealServer(unittest.TestCase):
    """Ordering is a SERVER capability, so an offline test cannot prove this one.

    Measured 2026-09-03 against 1.18.2: `order_by` needs a range index (400 without
    one), and the index is retroactive. Both are properties of the deployment, not of
    this code, which is exactly what an integration test is for. It writes only to the
    throwaway collection; the real archive is never touched.
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg = write_config()
        cls.store = core.build_memory(cls.cfg)
        cls.q = core.build_qdrant(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.q.delete_collection(THROWAWAY_COLLECTION)

    def test_the_first_page_is_the_most_recent_record(self):
        newest = self.store.store("the most recent fact")["id"]
        page = self.store.list_page(limit=1)
        self.assertEqual(page["order"], "updated_at_desc")
        self.assertEqual(page["memories"][0]["id"], newest,
                         "the record written last has to come back first")

    def test_the_index_the_ordering_needs_exists_after_a_write(self):
        self.store.store("a fact that creates the schema")
        info = self.q.request("GET", f"/collections/{THROWAWAY_COLLECTION}")
        schema = info["result"].get("payload_schema", {})
        self.assertIn("updated_at", schema,
                      "the write path has to leave the collection orderable")

    def test_the_cursor_walks_without_repeating(self):
        ids = [self.store.store(f"walkable fact {i}")["id"] for i in range(5)]
        seen, cursor = [], None
        for _ in range(10):
            page = self.store.list_page(limit=2, offset=cursor)
            seen.extend(m["id"] for m in page["memories"])
            cursor = page["next_offset"]
            if cursor is None:
                break
        self.assertIsNone(cursor, "the walk has to terminate")
        self.assertEqual(len(seen), len(set(seen)), "a record came back twice")
        for mid in ids:
            self.assertIn(mid, seen, "the walk skipped a record")

    def test_a_batch_written_with_one_timestamp_still_pages(self):
        """`store_many` stamps every record in the batch with the SAME `updated_at`, so
        the tie handling is exercised by ordinary use, not by a contrived fixture."""
        res = self.store.store_many([{"information": f"tied fact {i}"} for i in range(4)])
        seen, cursor = [], None
        for _ in range(10):
            page = self.store.list_page(limit=2, offset=cursor)
            seen.extend(m["id"] for m in page["memories"])
            cursor = page["next_offset"]
            if cursor is None:
                break
        self.assertEqual(len(seen), len(set(seen)), "a tied record came back twice")
        for mid in res["ids"]:
            self.assertIn(mid, seen, "a tied record was skipped")
```

Note for the implementer: `write_config()`, `THROWAWAY_COLLECTION` and `ENABLED` are
already defined at `tests/test_integration.py:33-51`. `core.build_qdrant(cfg)` is the
adapter factory; `Qdrant.request` is public and used the same way elsewhere in the repo.

- [ ] **Step 2: Run the integration suite against the real server**

Run: `QCTX_INTEGRATION=1 python3 -m unittest tests.test_integration -v`
Expected: OK. This is the step that proves the index can be created with the user's own
API key against their own Qdrant, which the spec lists as the one thing offline tests
cannot cover. If it fails on permissions, STOP and report: the degradation path is
correct behaviour, but the user needs to know their key cannot create an index.

- [ ] **Step 3: Correct the two documentation rows**

In `docs/usage.md`, replace the `memory list` row (line 23):

```markdown
| `memory list` | `memory_list` | list what is stored, newest first |
```

In `skills/memory/SKILL.md`, replace the `list` row (line 162):

```markdown
| list | `qctx memory list --limit N [--offset <cursor>]` |
```

Do not restate the ordering rule anywhere else in either file. `docs/usage.md` is a
reference table, and a second copy of a behavioural claim is a second thing to keep true.

- [ ] **Step 4: Run the documentation guards**

Run: `python3 -m unittest tests.test_readme_fidelity tests.test_repo_skill tests.test_host_equivalence -q`
Expected: OK. These check that every cited `qctx` command exists, that no `—` entered the
four guarded docs, and that the skill's command table matches the real `--help`.

- [ ] **Step 5: Run the whole suite and compare with the baseline**

Run: `python3 -m unittest discover -s tests -t . -q`
Expected: the 1297 baseline plus every test this plan added, and STILL only the one
preexisting `test_hermes_provider` failure. State the count, the delta, and that the
single remaining failure is the preexisting one, in the final report.

- [ ] **Step 6: Commit**

```bash
git add tests/test_integration.py docs/usage.md skills/memory/SKILL.md
git commit -m "test: prove the ordered listing against a real Qdrant, and fix the docs"
```

---

## Self-review

Ran against the spec after writing the plan.

**1. Spec coverage.** Every section maps to a task:

| spec section | task |
|---|---|
| `core/paging.py`, pure | 1 |
| `scroll` gains `order_by` | 2 |
| `core/ports.py` mirrors it | 2 |
| the fake refuses like the server | 2 |
| `ensure` creates the index | 3 |
| `list_page` orders and reports `order` | 3 |
| honest degradation with `warning` | 3 |
| `memory_list` takes `offset` | 4 |
| `qctx memory list --offset` | 5 |
| tool description matches behaviour | 4 |
| integration proof against the real server | 6 |
| the two doc rows | 6 |

**2. Placeholder scan.** No "TBD", no "add error handling", no "similar to Task N". Every
code step carries the code. The two places that say "read the current body first"
(`_matches_filter` in Task 2, `rendered`/`load_cli` in Task 5) name the exact file and
line to read and why, rather than leaving the content out.

**3. Type consistency.** Checked across tasks: `paging.ORDER_KEY`, `paging.INDEX_SCHEMA`,
`paging.ORDER_DESC` (`"updated_at_desc"`), `paging.ORDER_NONE` (`"unordered"`),
`page_request(cursor) -> {"order_by", "filter"}`, `next_cursor(points, limit, previous)`,
`FakeQdrantRefusal.status == 400`, `list_page(limit, offset)` returning
`{count, memories, next_offset, order}` plus `warning` only when degraded. The names used
in Tasks 3 through 6 are exactly the ones Tasks 1 and 2 define.

**4. Value consistency** (spec against plan, the check a past plan of mine failed):
`updated_at` (key), `datetime` (index schema), `desc` (direction), 400 (status), 744
(records measured), 1.18.2 (server) all match the spec verbatim.
