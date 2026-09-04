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
from tests.fakes import FakeQdrantRefusal, FakeVectorStore  # noqa: E402


def point(pid, value):
    return {"id": pid, "payload": {"updated_at": value, "document": f"doc {pid}"}}


class TestTheFirstPage(unittest.TestCase):
    def test_no_cursor_asks_for_descending_order_and_no_filter(self):
        req = paging.page_request(None)
        self.assertEqual(req["order_by"], {"key": "updated_at", "direction": "desc"})
        self.assertIsNone(req["filter"], "the first page excludes nothing")


class TestTheCursorRoundTrip(unittest.TestCase):
    def test_a_cursor_carries_the_value_the_ids_seen_at_it_and_its_MODE(self):
        cursor = paging.encode_cursor("2026-05-05T10:00:00+00:00", ["a", "b"])
        self.assertEqual(paging.decode_cursor(cursor),
                         ("2026-05-05T10:00:00+00:00", ["a", "b"], True))

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
        value, seen, _ = paging.decode_cursor(
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
        value, seen, _ = paging.decode_cursor(
            paging.next_cursor(points, limit=3, previous=None))
        self.assertEqual(value, "2026-05-05T10:00:00+00:00")
        self.assertEqual(sorted(seen), ["b", "c"])

    def test_ids_already_seen_at_the_same_value_accumulate(self):
        """Page 2 of a long tie must not forget what page 1 excluded, or the pages
        alternate between the same two records forever."""
        previous = paging.encode_cursor("2026-05-05T10:00:00+00:00", ["a"])
        points = [point("b", "2026-05-05T10:00:00+00:00"),
                  point("c", "2026-05-05T10:00:00+00:00")]
        value, seen, _ = paging.decode_cursor(
            paging.next_cursor(points, limit=2, previous=previous))
        self.assertEqual(value, "2026-05-05T10:00:00+00:00")
        self.assertEqual(sorted(seen), ["a", "b", "c"])

    def test_a_new_boundary_value_drops_the_old_exclusions(self):
        """Carrying stale ids past their value would grow the filter without bound."""
        previous = paging.encode_cursor("2026-09-01T00:00:00+00:00", ["a"])
        points = [point("b", "2026-07-01T00:00:00+00:00"),
                  point("c", "2026-05-05T10:00:00+00:00")]
        value, seen, _ = paging.decode_cursor(
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
        self.q = FakeVectorStore()
        self.q.ensure_collection("mem", 8)
        for pid, value in (("a", "2026-01-01T00:00:00+00:00"),
                           ("b", "2026-09-01T00:00:00+00:00"),
                           ("c", "2026-05-05T10:00:00+00:00")):
            self.q.upsert("mem", [{"id": pid, "vector": [0.0] * 8,
                                   "payload": {"updated_at": value,
                                               "document": f"doc {pid}"}}])

    def test_ordering_without_an_index_is_refused_the_way_the_server_refuses_it(self):
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

    def test_an_unordered_scroll_PAGES_like_the_server(self):
        """It used to assert `next_page_offset is None` here, which was the fake being
        kinder than production: the real server pages an unordered scroll and hands back
        a cursor. That generosity hid a real defect, where the degraded listing dropped
        the cursor and reported a 749-record archive as finished after one page."""
        points, offset = self.q.scroll("mem", limit=2)
        self.assertEqual(len(points), 2)
        self.assertIsNotNone(offset, "an unordered scroll with more to give has a cursor")
        rest, final = self.q.scroll("mem", limit=2, offset=offset)
        self.assertEqual(len(rest), 1, "the third record")
        self.assertIsNone(final, "and now the walk is over")
        self.assertEqual({p["id"] for p in points} | {p["id"] for p in rest},
                         {"a", "b", "c"}, "the walk covers the archive exactly once")

    def test_ordering_together_with_an_offset_is_refused_like_the_server(self):
        """Measured fact 4: the server answers 400 for the combination. A fake that
        accepts it lets a caller ship a request production rejects."""
        self.q.ensure_payload_index("mem", "updated_at", "datetime")
        with self.assertRaises(FakeQdrantRefusal):
            self.q.scroll("mem", limit=2, offset="a",
                          order_by={"key": "updated_at", "direction": "desc"})

    def test_a_non_string_value_is_excluded_not_fatal(self):
        """The real server drops such a record from an ordered scroll. Raising TypeError
        here trained the tests on a failure production does not have."""
        self.q.ensure_payload_index("mem", "updated_at", "datetime")
        self.q.upsert("mem", [{"id": "n", "vector": [0.0] * 8,
                               "payload": {"updated_at": 1234, "document": "numeric"}}])
        points, _ = self.q.scroll("mem", limit=10,
                                  order_by={"key": "updated_at", "direction": "desc"})
        self.assertNotIn("n", [p["id"] for p in points])


class TestTiesAcrossTIMESTAMPFORMATS(unittest.TestCase):
    """The defect a string comparison hid, found by an independent reviewer.

    A `datetime` index makes the server compare INSTANTS. `2026-09-01T00:00:00Z` and
    `2026-09-01T00:00:00+00:00` are one instant and two strings. While the tie was
    detected with `==` on the raw text, the boundary record's twin was never excluded, so
    it came back on every page: measured against the real server, a 6-record archive
    returned 50 rows, repeated one record 18 times, never reached a seventh and never
    terminated.
    """

    Z = "2026-09-01T00:00:00Z"
    OFFSET = "2026-09-01T00:00:00+00:00"

    def test_the_two_spellings_of_one_instant_are_the_same_boundary(self):
        points = [point("a", self.Z), point("b", self.OFFSET)]
        value, seen, _ = paging.decode_cursor(
            paging.next_cursor(points, limit=2, previous=None))
        self.assertEqual(sorted(seen), ["a", "b"],
                         "both ids share the instant, so both must be excluded next page")

    def test_a_cursor_written_in_one_spelling_matches_a_page_in_the_other(self):
        previous = paging.encode_cursor(self.Z, ["a"])
        points = [point("b", self.OFFSET), point("c", self.OFFSET)]
        value, seen, _ = paging.decode_cursor(
            paging.next_cursor(points, limit=2, previous=previous))
        self.assertEqual(sorted(seen), ["a", "b", "c"],
                         "the accumulated id must survive a change of spelling")

    def test_the_value_sent_back_is_the_one_the_server_wrote(self):
        """Normalizing for COMPARISON is right; positioning the server with a
        normalization it never wrote is not."""
        points = [point("a", self.Z), point("b", self.Z)]
        value, _, _ = paging.decode_cursor(
            paging.next_cursor(points, limit=2, previous=None))
        self.assertEqual(value, self.Z)


class TestTheDegradedCursor(unittest.TestCase):
    """An unordered walk pages by the server's own offset, and its cursor must never be
    mistaken for a timestamp cursor."""

    def test_it_carries_the_server_offset_and_declares_itself_unordered(self):
        cursor = paging.unordered_cursor("a-point-id")
        value, seen, ordered = paging.decode_cursor(cursor)
        self.assertEqual(value, "a-point-id")
        self.assertEqual(seen, [])
        self.assertFalse(ordered)

    def test_no_server_offset_means_the_walk_ended(self):
        self.assertIsNone(paging.unordered_cursor(None))

    def test_continuing_it_asks_for_no_ordering_and_passes_the_offset_through(self):
        req = paging.page_request(paging.unordered_cursor("a-point-id"))
        self.assertFalse(req["ordered"])
        self.assertIsNone(req["order_by"], "the store already said it cannot order")
        self.assertEqual(req["offset"], "a-point-id")
        self.assertIsNone(req["filter"])

    def test_an_ordered_cursor_whose_value_is_not_a_timestamp_is_REFUSED(self):
        """The failure the spec forbids by name, measured end to end: such a cursor was
        accepted, went out as `order_by.start_from`, took a 400 from the server ("Format
        error in JSON body"), was absorbed as "cannot order", and the listing answered
        PAGE ONE labelled as a later page while blaming an index that existed."""
        for value in ("i am not a timestamp", "2026-13-45", "a-point-id"):
            with self.subTest(value=value):
                with self.assertRaises(paging.PagingError):
                    paging.decode_cursor(paging.encode_cursor(value, []))

    def test_the_two_modes_do_not_accept_each_others_cursors(self):
        ordered = paging.next_cursor(
            [point("a", "2026-09-01T00:00:00+00:00")], limit=1, previous=None)
        self.assertTrue(paging.page_request(ordered)["ordered"])
        self.assertFalse(paging.page_request(paging.unordered_cursor("x"))["ordered"])


if __name__ == "__main__":
    unittest.main()
