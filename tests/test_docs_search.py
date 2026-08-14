"""The document search path, with fakes — the policy it hands the pipeline.

`DocIndex.search`, `_search_scope` and `_to_hit` executed ZERO lines in the offline
suite. The consequence is not that they were untested in the abstract: it is that the
policy this whole half of the design rests on was undefended. Both of these mutations
left all 210 offline tests AND all 17 integration tests green:

    DENSE_FLOOR = 0.30  ->  0.99
    Policy(veto=False, detect_collapse=True, order_matters=True)
        ->  Policy(veto=True, detect_collapse=False, order_matters=False)

`tests/test_retrieval.py` covers `two_stage` thoroughly, but with a policy the TEST
constructs. Nobody asserted which policy `DocIndex` actually passes it — so the module
docstring, the README and the skill could all describe "no veto, order matters" while the
code did the opposite, and nothing offline would notice.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import retrieval
from core.docs import DENSE_FLOOR, DocIndex
from tests.fakes import FakeEmbedder, FakeReranker, FakeVectorStore


class DocsBase(unittest.TestCase):
    def _index(self, reranker=None):
        q, emb = FakeVectorStore(), FakeEmbedder()

        return DocIndex(q, emb, reranker, "tmp", "lib", emb.dim), q

    def _file(self, body: str) -> str:
        path = os.path.join(tempfile.mkdtemp(), "doc.md")
        Path(path).write_text(body)

        return path

    #: Tokens chosen to land in DISTINCT buckets of FakeEmbedder's crc32 % dim.
    #:
    #: The fake is bag-of-words into `dim` buckets, so a section padded with 700
    #: repetitions of one shared filler word produces a vector pinned to that filler's
    #: bucket — and every section then looks identical and nothing matches a query.
    #: Measured with the obvious fixture: all dense scores came back 0.001 or 0.000,
    #: below the 0.30 floor, so `search` returned nothing and the tests failed for a
    #: reason that had nothing to do with the code. Padding each section with its OWN
    #: token instead makes the section vector point at that token's bucket, which is what
    #: gives a single-word query something to be similar to.
    TOPICS = ("pagination", "billing", "ratelimit", "authentication", "indexing")

    def _doc(self, n_sections: int = 4) -> str:
        """A document whose sections are individually big enough to become chunks, and
        distinguishable to the fake embedder."""
        return self._file("\n\n".join(
            f"# {t.title()}\n\n" + (f"{t} " * 700) for t in self.TOPICS[:n_sections]))


class TestTheDocsPolicy(DocsBase):
    """What `DocIndex.search` hands to `two_stage`, asserted directly."""

    def _captured_policy(self, **kw) -> retrieval.Policy:
        seen = {}
        real = retrieval.two_stage

        def spy(candidates, query, reranker, policy, **rest):
            seen["policy"] = policy

            return real(candidates, query, reranker, policy, **rest)

        retrieval.two_stage = spy
        try:
            idx, _ = self._index(FakeReranker(scores=[0.9, 0.5, 0.02]))
            idx.keep_file(self._doc())
            idx.search("pagination", scope="library", **kw)
        finally:
            retrieval.two_stage = real

        return seen["policy"]

    def test_the_second_stage_may_not_veto(self):
        """Whoever asks has already chosen the document; silence is worse than imperfect
        order. A veto here turns a slightly-off question into no answer at all."""
        self.assertFalse(self._captured_policy().veto)

    def test_the_order_is_the_product(self):
        """A document result is a list read top to bottom, so reranking is worth paying
        for even when everything fits and everything passes."""
        self.assertTrue(self._captured_policy().order_matters)

    def test_the_two_floors_are_equal(self):
        """With no veto, "fall back to the strict cut" has to be a no-op. If the floors
        differed, a discarded judgement would return silence — exactly what this policy
        exists to prevent."""
        p = self._captured_policy()
        self.assertEqual(p.dense_floor, p.strict_floor)
        self.assertAlmostEqual(p.dense_floor, DENSE_FLOOR)

    def test_the_cutoff_is_the_caller_s_to_set(self):
        self.assertAlmostEqual(self._captured_policy(min_score=0.42).min_score, 0.42)

    def test_the_dense_floor_is_generous_enough_to_admit_a_partial_match(self):
        """Pins the VALUE of the floor, not just that a floor exists.

        Raising DENSE_FLOOR from 0.30 to 0.99 left the whole suite green, because every
        other fixture here matches its query almost exactly and would clear any floor
        below 1.0. What the 0.30 is FOR is the chunk that is partly on topic — documents
        have no veto, so the first stage is deliberately permissive and lets the
        cross-encoder sort it out.

        A chunk padded with two topics in equal measure sits at cosine 0.707 to a
        single-topic query under the fake embedder: comfortably above 0.30 and just as
        comfortably below 0.99.
        """
        idx, _ = self._index()
        mixed = self._file("# Mixed\n\n" + ("pagination billing " * 700))
        idx.keep_file(mixed)
        hits, outcome = idx.search("pagination", scope="library")
        self.assertTrue(hits, "a half-on-topic chunk has to survive the first stage")
        self.assertLess(outcome.best_dense, 0.99, "the fixture is not exercising the floor")
        self.assertGreater(outcome.best_dense, DENSE_FLOOR)


class TestSearchBehaviour(DocsBase):
    def test_a_weak_chunk_is_delivered_marked_rather_than_dropped(self):
        """The observable consequence of no-veto, without reaching into the policy."""
        idx, _ = self._index(FakeReranker(scores=[0.9, 0.005, 0.004, 0.003]))
        idx.keep_file(self._doc())
        hits, outcome = idx.search("pagination", scope="library", limit=5)
        self.assertTrue(hits, "with a veto this would be silence")
        self.assertIn(retrieval.CE_WEAK, {h.origin for h in hits},
                      "below the cutoff means marked, not removed")

    def test_a_failed_rerank_still_answers(self):
        idx, _ = self._index(FakeReranker(ok=False, error="timeout"))
        idx.keep_file(self._doc())
        hits, outcome = idx.search("billing", scope="library")
        self.assertTrue(hits)
        self.assertEqual(outcome.rerank_error, "timeout")
        self.assertTrue(all(h.origin == retrieval.DENSE for h in hits))

    def test_a_collapse_falls_back_to_dense_order_without_silence(self):
        idx, _ = self._index(FakeReranker(scores=[0.0004, 0.0002, 0.0009, 0.0001]))
        idx.keep_file(self._doc())
        hits, outcome = idx.search("ratelimit", scope="library")
        self.assertTrue(outcome.collapsed)
        self.assertTrue(hits, "silence here would be worse than imperfect order")

    def test_scope_selects_the_archive(self):
        idx, _ = self._index()
        idx.keep_file(self._doc())
        idx.index_file(self._doc(), ttl_seconds=600)
        self.assertTrue(all(h.scope == "library"
                            for h in idx.search("pagination", scope="library")[0]))
        self.assertTrue(all(h.scope == "tmp" for h in idx.search("pagination", scope="tmp")[0]))
        self.assertEqual({h.scope for h in idx.search("pagination", scope="all")[0]},
                         {"library", "tmp"})

    def test_an_expired_chunk_is_invisible_to_search(self):
        idx, _ = self._index()
        idx.index_file(self._doc(), ttl_seconds=-1)
        self.assertEqual(idx.search("pagination", scope="tmp")[0], [])

    def test_doc_id_narrows_to_one_document(self):
        idx, _ = self._index()
        first = idx.keep_file(self._doc())
        idx.keep_file(self._file("# Other\n\n" + ("pagination " * 700)))
        hits, _ = idx.search("pagination", scope="library", doc_id=first["doc_id"])
        self.assertTrue(hits)
        self.assertTrue(all(h.path == first["path"] for h in hits))

    def test_a_hit_carries_the_location_and_the_mode(self):
        idx, _ = self._index()
        idx.keep_file(self._doc())
        hit = idx.search("pagination", scope="library")[0][0]
        self.assertEqual(hit.mode, "locator")
        self.assertGreater(hit.start_line, 0)
        self.assertGreaterEqual(hit.end_line, hit.start_line)
        self.assertIsNone(hit.stale, "the file has not been touched since indexing")

    def test_a_changed_file_marks_every_hit_from_it(self):
        idx, _ = self._index()
        path = self._doc()
        idx.keep_file(path)
        Path(path).write_text("# Pagination\n\n" + ("pagination " * 700))
        hits, _ = idx.search("pagination", scope="library")
        self.assertTrue(hits)
        self.assertTrue(all(h.stale for h in hits),
                        "a chunk from a version that no longer exists must say so")

    def test_an_invalid_scope_is_refused_rather_than_guessed(self):
        idx, _ = self._index()
        with self.assertRaises(Exception):
            idx.search("x", scope="libary")


if __name__ == "__main__":
    unittest.main(verbosity=2)
