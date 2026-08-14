"""Memory and the document index, with fakes — no network, no credentials.

It covers offline what previously existed only under an integration test. The most
important case is the PAYLOAD SHAPE: it is the property that allows replacing the
previous server without migrating data, and it was verified only against the user's live
collection. A safety property tested only in production is not tested.
"""
import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
from core import retrieval
from core.docs import DocIndex
from core.memory import MemoryError_, MemoryStore
from tests.fakes import (FakeEmbedder, FailingFakeEmbedder, FakeReranker,
                         FakeVectorStore)

#: The four keys the previous MCP server wrote. Changing this makes the existing archive
#: inconsistent, with no error appearing anywhere.
KEYS = {"document", "metadata", "created_at", "updated_at"}

POL = retrieval.Policy(dense_floor=0.45, strict_floor=0.58, min_score=0.10,
                       max_results=6, veto=True)


def store(reranker=None, collection="mem"):
    q, emb = FakeVectorStore(), FakeEmbedder()
    q.ensure_collection(collection, emb.dim)

    return MemoryStore(q, emb, reranker, collection, emb.dim), q, emb


class TestPayloadShape(unittest.TestCase):
    """Compatibility with the archive already stored — the property the swap rests on."""

    def test_store_writes_exactly_the_four_keys(self):
        s, q, _ = store()
        r = s.store("a fact", {"type": "reference"})
        point = q.get_point("mem", r["id"])
        self.assertEqual(set(point["payload"]), KEYS)

    def test_store_many_writes_the_same_keys(self):
        s, q, _ = store()
        r = s.store_many([{"information": "a"}, {"information": "b", "metadata": {"x": 1}}])
        for mid in r["ids"]:
            self.assertEqual(set(q.get_point("mem", mid)["payload"]), KEYS)

    def test_update_preserves_created_at(self):
        s, q, _ = store()
        mid = s.store("original")["id"]
        before = q.get_point("mem", mid)["payload"]["created_at"]
        time.sleep(0.01)
        s.update(mid, information="corrected")
        after = q.get_point("mem", mid)["payload"]
        self.assertEqual(after["created_at"], before, "created_at must not be rewritten")
        self.assertNotEqual(after["updated_at"], before)
        self.assertEqual(set(after), KEYS)

    def test_update_without_metadata_keeps_the_previous_one(self):
        s, q, _ = store()
        mid = s.store("x", {"type": "reference", "project": "p"})["id"]
        s.update(mid, information="y")
        self.assertEqual(q.get_point("mem", mid)["payload"]["metadata"],
                         {"type": "reference", "project": "p"})

    def test_update_without_text_keeps_the_document(self):
        s, q, _ = store()
        mid = s.store("original text")["id"]
        s.update(mid, metadata={"type": "user"})
        self.assertEqual(q.get_point("mem", mid)["payload"]["document"], "original text")

    def test_update_of_a_missing_id_does_not_create(self):
        s, q, _ = store()
        self.assertEqual(s.update("does-not-exist", information="x")["status"], "not_found")
        self.assertEqual(len(q.collections["mem"]["points"]), 0)


class TestRefusedWrites(unittest.TestCase):
    def test_empty_memory(self):
        s, _, _ = store()
        for bad in ("", "   ", "\n"):
            with self.assertRaises(MemoryError_):
                s.store(bad)

    def test_batch_validates_before_writing_anything(self):
        s, q, _ = store()
        with self.assertRaises(MemoryError_):
            s.store_many([{"information": "valid"}, {"information": "  "}])
        self.assertEqual(len(q.collections["mem"]["points"]), 0,
                         "validation has to happen BEFORE writing, not during")

    def test_batch_makes_ONE_embedding_call(self):
        s, _, emb = store()
        s.store_many([{"information": f"fact {i}"} for i in range(5)])
        self.assertEqual(len(emb.calls), 1, "the point of a batch is one network trip, not N")


class TestReadsNeverCreate(unittest.TestCase):
    """A read must not CREATE a collection: with a typo in the name, the search would
    return zero and the consumer would conclude 'there is no precedent' — the most
    dangerous statement possible."""

    def _missing_collection(self):
        q, emb = FakeVectorStore(), FakeEmbedder()

        return MemoryStore(q, emb, None, "does_not_exist", emb.dim), q

    def test_find_fails_loudly(self):
        s, q = self._missing_collection()
        with self.assertRaises(MemoryError_):
            s.find("x")
        self.assertEqual(q.list_collections(), [], "nothing may have been created")

    def test_recall_fails_loudly(self):
        s, _ = self._missing_collection()
        with self.assertRaises(MemoryError_):
            s.recall(["x"], POL, top_k=5)

    def test_list_page_fails_loudly(self):
        s, _ = self._missing_collection()
        with self.assertRaises(MemoryError_):
            s.list_page()

    def test_store_MAY_create(self):
        s, q = self._missing_collection()
        s.store("first fact")
        self.assertIn("does_not_exist", q.list_collections(), "a write creates, a read does not")


class TestRecall(unittest.TestCase):
    def _populate(self, reranker=None):
        s, q, _ = store(reranker)
        s.store("poll pagination truncates at 100 items", {"type": "reference"})
        s.store("the cursor regresses when the clock steps back", {"type": "reference"})
        s.store("carrot cake recipe with frosting", {"type": "user"})

        return s, q

    def test_retrieves_by_similarity_and_returns_Recalled(self):
        s, _ = self._populate()
        hits, outcome = s.recall(["poll pagination truncates"], POL, top_k=10)
        self.assertTrue(hits)
        self.assertIsInstance(hits[0], core.Recalled)
        self.assertIn("pagination", hits[0].document,
                      "the query shares three words with this record")
        self.assertEqual(hits[0].origin, "dense")

    def test_floor_relaxes_ONLY_when_a_reranker_exists(self):
        without_ce, _ = self._populate(None)
        with_ce, _ = self._populate(FakeReranker(scores=[0.9, 0.8, 0.7]))
        _, out_without_ce = without_ce.recall(["poll"], POL, top_k=10)
        _, out_with_ce = with_ce.recall(["poll"], POL, top_k=10)
        self.assertGreaterEqual(out_with_ce.candidates, out_without_ce.candidates,
                                "with a second stage the first one can be more generous")

    def test_fusing_angles_does_not_duplicate(self):
        s, _ = self._populate()
        hits, _ = s.recall(["poll pagination", "pagination poll", "truncates 100 items"],
                           POL, top_k=10)
        ids = [h.id for h in hits]
        self.assertEqual(len(ids), len(set(ids)))

    def test_best_dense_reports_the_best_of_ALL_hits(self):
        """To say 'nothing cleared the cut, the best was X', the useful number is the best
        of them all — not the best of those that cleared the floor."""
        s, _ = self._populate()
        strict = retrieval.Policy(0.99, 0.99, 0.10, 6, veto=True)
        hits, outcome = s.recall(["completely absent subject xyz"], strict, top_k=10)
        self.assertEqual(hits, [])
        self.assertGreater(outcome.best_dense, 0.0)

    def test_record_without_a_document_is_discarded(self):
        s, q, _ = store()
        q.upsert("mem", [{"id": "empty", "vector": [1.0] * 8,
                          "payload": {"document": "   ", "metadata": {}}}])
        hits, _ = s.recall(["anything"], retrieval.Policy(0.0, 0.0, 0.10, 6), top_k=10)
        self.assertEqual([h.id for h in hits], [],
                         "a vector with no text can be neither judged nor presented")

    def test_embedding_failure_propagates_as_a_domain_error(self):
        q = FakeVectorStore()
        q.ensure_collection("mem", 8)
        s = MemoryStore(q, FailingFakeEmbedder(core.EmbeddingError("down")), None, "mem", 8)
        with self.assertRaises(core.CoreError):
            s.recall(["x"], POL, top_k=5)


class TestDocsOffline(unittest.TestCase):
    def _index(self, reranker=None):
        q, emb = FakeVectorStore(), FakeEmbedder()

        return DocIndex(q, emb, reranker, "tmp", "lib", emb.dim), q

    def _write_file(self, content="# Title\n\nbody of the document here\n"):
        import tempfile
        d = tempfile.mkdtemp()
        path = os.path.join(d, "doc.md")
        with open(path, "w") as fh:
            fh.write(content)

        return path

    def test_temporary_carries_an_expiry_and_library_does_not(self):
        idx, q = self._index()
        path = self._write_file()
        idx.index_file(path, ttl_seconds=60)
        idx.keep_file(path)
        tmp = list(q.collections["tmp"]["points"].values())[0]["payload"]
        lib = list(q.collections["lib"]["points"].values())[0]["payload"]
        self.assertIn("expires_at_ts", tmp)
        self.assertNotIn("expires_at_ts", lib, "the library does not expire, by construction")

    def test_reindexing_replaces_instead_of_accumulating(self):
        idx, q = self._index()
        path = self._write_file()
        idx.keep_file(path)
        before = len(q.collections["lib"]["points"])
        idx.keep_file(path)
        self.assertEqual(len(q.collections["lib"]["points"]), before)

    def test_reindexing_a_SHRUNK_file_leaves_no_orphan_chunks(self):
        """The version above cannot fail, and that is why this one exists.

        Reindexing the same content overwrites in place, because the point ids are
        derived deterministically from (doc_id, index) — so deleting the `drop` call in
        `_write` left the whole suite green, offline AND integration. The guarantee only
        has teeth when the new version has FEWER chunks than the old one: the surplus
        chunks of the previous version survive with stale metadata and compete as equals
        in every later search, which is the "two states of the same file" the code exists
        to prevent.
        """
        idx, q = self._index()
        # Each section has to clear TARGET_CHARS on its own, otherwise the packer merges
        # them into one chunk and the test has nothing to observe.
        long_body = "\n\n".join(f"# Section {i}\n\nbody of section {i} " + ("word " * 700)
                                 for i in range(6))
        path = self._write_file(long_body)
        idx.keep_file(path)
        many = len(q.collections["lib"]["points"])
        self.assertGreater(many, 2, "the fixture has to produce several chunks to be a test")

        Path(path).write_text("# Section 0\n\nall that is left\n")
        idx.keep_file(path)
        few = len(q.collections["lib"]["points"])
        self.assertLess(few, many, "chunks from the previous, longer version survived")
        texts = " ".join(p["payload"]["document"]
                         for p in q.collections["lib"]["points"].values())
        self.assertNotIn("body of section 5", texts, "an orphan chunk is still searchable")

    def test_sweep_removes_only_what_expired(self):
        idx, q = self._index()
        idx.index_file(self._write_file(), ttl_seconds=-1)
        idx.index_file(self._write_file("other content here\n"), ttl_seconds=600)
        idx.sweep()
        remaining_docs = [p["payload"]["document"]
                     for p in q.collections["tmp"]["points"].values()]
        self.assertTrue(remaining_docs, "the 600s one has to survive")
        self.assertTrue(all("other" in d for d in remaining_docs),
                        "only the expired one is removed")

    def test_purging_temporary_does_not_touch_the_library(self):
        idx, q = self._index()
        path = self._write_file()
        idx.index_file(path, ttl_seconds=600)
        idx.keep_file(path)
        idx.drop_all_tmp()
        self.assertNotIn("tmp", q.list_collections())
        self.assertIn("lib", q.list_collections())

    def test_binary_is_refused(self):
        idx, _ = self._index()
        with self.assertRaises(core.DocsError):
            idx.index_file(self._write_file("text\x00binary"))

    def test_line_range_reproduces_the_chunk(self):
        """Read back with the READER's primitive, never with `str.splitlines()`.

        This test used `.splitlines()` — the same call the code under test used to make —
        so it agreed with the code even when both were wrong about what a line is. It
        passed happily while a single form feed shifted every line number in the file.
        """
        content = "\n".join(f"line {i}" for i in range(1, 21)) + "\n"
        path = self._write_file(content)
        idx, q = self._index()
        idx.keep_file(path)
        with open(path, newline="") as fh:
            lines = [ln.rstrip("\n") for ln in fh.readlines()]
        for p in q.collections["lib"]["points"].values():
            md, doc = p["payload"]["metadata"], p["payload"]["document"]
            slice_text = "\n".join(lines[md["start_line"] - 1:md["end_line"]])
            self.assertEqual(slice_text.strip("\n"), doc,
                             "this is the contract of locator mode")


if __name__ == "__main__":
    unittest.main(verbosity=2)
