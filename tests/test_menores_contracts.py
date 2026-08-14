"""Contracts the code states about itself, checked adversarially.

Each of these is a promise written in a docstring or a comment that nothing enforced.
They were found by asking, for every such claim, "what input makes this false?" — which
is a different exercise from reading the code and agreeing with it.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
from core.memory import MemoryStore
from core.qdrant import QdrantError, _is_absent
from core.reranking import Reranker
from tests.fakes import FakeEmbedder, FakeVectorStore


class TestAbsenceIsDecidedByStatus(unittest.TestCase):
    """`core/qdrant.py` writes the rule and then broke it twice."""

    def _err(self, message: str, status=None) -> QdrantError:
        e = QdrantError(message)
        e.status = status

        return e

    def test_a_real_404_means_absent(self):
        self.assertTrue(_is_absent(self._err("HTTP 404 on GET /collections/x", status=404)))

    def test_a_proxy_502_quoting_404_in_its_body_does_NOT_mean_absent(self):
        """The measured failure. This Qdrant is behind a reverse proxy, and proxies echo
        upstream statuses into their own error bodies. Read as absence, it made
        `ensure_collection` believe a live collection was missing — which is how a
        dimension mismatch walks past the guard written to catch it."""
        exc = self._err('HTTP 502 on GET /collections/x: upstream error: backend '
                        'returned HTTP 404 while reloading', status=502)
        self.assertFalse(_is_absent(exc))

    def test_a_transport_failure_with_no_status_is_not_absence(self):
        self.assertFalse(_is_absent(self._err("could not reach host: timed out")))

    def test_other_error_statuses_are_not_absence(self):
        for status in (401, 403, 500, 503):
            self.assertFalse(_is_absent(self._err(f"HTTP {status}", status=status)))


class TestRankNeverRaises(unittest.TestCase):
    """The promise is only worth having if it holds when the caller is wrong too."""

    def _rr(self) -> Reranker:
        # Port 1 on loopback refuses instantly: no server, no DNS, no waiting.
        return Reranker("http://127.0.0.1:1/rerank", "m", timeout=2.0)

    def test_a_non_string_document_does_not_raise(self):
        for documents in ([None], [3], [{"not": "a string"}], ["ok", None]):
            pairs, info = self._rr().rank("q", documents)
            self.assertEqual(pairs, [])
            self.assertFalse(info["ok"])
            self.assertTrue(info["error"], "a failure has to be reported, not swallowed")

    def test_a_non_string_query_does_not_raise(self):
        pairs, info = self._rr().rank(None, ["a"])
        self.assertFalse(info["ok"])

    def test_an_unreachable_endpoint_does_not_raise(self):
        pairs, info = self._rr().rank("q", ["a"])
        self.assertEqual(pairs, [])
        self.assertFalse(info["ok"])


class TestUpdateNeverCorruptsTheVector(unittest.TestCase):
    def _store(self):
        q, emb = FakeVectorStore(), FakeEmbedder()
        q.ensure_collection("mem", emb.dim)

        return MemoryStore(q, emb, None, "mem", emb.dim), q, emb

    def test_metadata_only_update_on_a_payload_WITHOUT_a_document_key(self):
        """The corrupting case: `previous.get("document", "")` on one side and
        `previous.get("document")` on the other. For a payload with no `document` key
        those are "" and None, which compare unequal — so a metadata-only update took the
        re-embed branch, embedded the empty string, and overwrote the point's vector with
        it. A record that cannot be found again is worse than an update that fails."""
        s, q, emb = self._store()
        q.upsert("mem", [{"id": "odd", "vector": [1.0] + [0.0] * (emb.dim - 1),
                          "payload": {"metadata": {"type": "reference"}}}])
        before = list(q.get_point("mem", "odd")["vector"])
        calls = len(emb.calls)

        res = s.update("odd", metadata={"type": "corrected"})

        self.assertEqual(res["status"], "updated")
        self.assertFalse(res["reembedded"], "there is no new text; nothing to embed")
        self.assertEqual(len(emb.calls), calls, "the endpoint must not have been called")
        self.assertEqual(q.get_point("mem", "odd")["vector"], before,
                         "the vector was replaced by the embedding of an empty string")
        self.assertEqual(q.get_point("mem", "odd")["payload"]["metadata"],
                         {"type": "corrected"})


class TestSearchCollectionsIsHonestAboutWhatItSkipped(unittest.TestCase):
    """READ-ONLY search across other systems' archives — it had no test of any kind.

    The reason it needs one is the guard, not the happy path: reading from an archive
    built by a DIFFERENT embedding model returns random neighbours with plausible scores,
    which is worse than returning nothing, so a dimension mismatch has to be skipped and
    REPORTED rather than silently searched.
    """

    def _world(self):
        q, emb = FakeVectorStore(), FakeEmbedder()
        q.ensure_collection("ours", emb.dim)
        q.upsert("ours", [{"id": "1", "vector": emb.embed_one("pagination poll cursor"),
                           "payload": {"document": "poll pagination truncates at 100"}}])
        q.ensure_collection("other_model", emb.dim + 4)   # a different embedding model

        return q, emb

    def test_a_collection_of_another_dimension_is_skipped_and_named(self):
        q, emb = self._world()
        res = core.search_collections(q, emb, "pagination", None, emb.dim)
        self.assertIn("ours", res["searched"])
        self.assertNotIn("other_model", res["searched"])
        skipped = {s["collection"]: s["reason"] for s in res["skipped"]}
        self.assertIn("other_model", skipped)
        self.assertIn("dimension", skipped["other_model"],
                      "the reason has to say WHY, or the caller cannot act on it")

    def test_a_missing_collection_is_reported_rather_than_raising(self):
        q, emb = self._world()
        res = core.search_collections(q, emb, "pagination", ["ours", "not_there"], emb.dim)
        self.assertEqual(res["searched"], ["ours"])
        self.assertEqual([s["collection"] for s in res["skipped"]], ["not_there"])

    def test_it_finds_the_document_through_the_field_heuristic(self):
        q, emb = self._world()
        hit = core.search_collections(q, emb, "pagination poll", None, emb.dim)["results"][0]
        self.assertEqual(hit["collection"], "ours")
        self.assertIn("pagination", hit["document"])
        self.assertIsNone(hit["payload"], "the raw payload is only for when the guess fails")

    def test_an_unrecognised_payload_shape_returns_the_payload_instead_of_nothing(self):
        """Degrading honestly: when no known field holds the text, hand back everything
        and let the caller look, rather than reporting a hit with no content."""
        q, emb = self._world()
        q.upsert("ours", [{"id": "2", "vector": emb.embed_one("pagination poll cursor"),
                           "payload": {"weird_field": "the text lives here"}}])
        res = core.search_collections(q, emb, "pagination poll", ["ours"], emb.dim)
        odd = [h for h in res["results"] if h["id"] == "2"][0]
        self.assertIsNone(odd["document"])
        self.assertEqual(odd["payload"], {"weird_field": "the text lives here"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
