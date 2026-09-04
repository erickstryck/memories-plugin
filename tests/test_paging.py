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


if __name__ == "__main__":
    unittest.main()
