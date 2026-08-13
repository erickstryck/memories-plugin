"""Tests for the text the recall hook INJECTS.

This block is the hook's entire product, and it had no coverage. The consequence showed
up in a live log: with 19 candidates left unjudged by the cross-encoder's pair ceiling,
the injected block said both "there may be relevant memory outside" AND "There is no
recorded precedent on this subject". Two sentences, same block, contradicting each other
— and the flat claim is precisely the failure the hook exists to prevent, because a
model that reads it goes on to assert something is unprecedented when nothing was
exhaustively searched.

What these tests pin is therefore not formatting but the CLAIM the block makes: whenever
the pipeline was partial for any reason, the block must not assert absence of precedent.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks"))

from core.retrieval import CE, DENSE, Outcome, Scored


def load_hook():
    """Imports the hook with STATE_DIR pointing at a temporary directory."""
    import importlib
    os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()
    import recall
    importlib.reload(recall)

    return recall


#: The sentence that must never appear alongside a partial judgement.
FLAT_CLAIM = "There is no recorded precedent"
HEDGE = "not evidence that no precedent exists"


class TestEmptyBlockClaim(unittest.TestCase):
    def setUp(self):
        self.recall = load_hook()

    def test_a_clean_empty_search_may_state_there_is_no_precedent(self):
        """The whole point of the hard claim: it stops the model re-running a search that
        genuinely came back empty. It has to survive when the pipeline WAS complete."""
        out = self.recall.empty_block(Outcome(candidates=4, best_dense=0.31, reranked=True), 3)
        self.assertIn(FLAT_CLAIM, out)
        self.assertNotIn(HEDGE, out)
        self.assertIn("0.310", out, "the best dense score is what makes the claim checkable")

    def test_unjudged_candidates_withdraw_the_claim(self):
        """The live defect. 26 candidates, 12 judged, none passed: the other 14 were never
        looked at, so "no precedent" is not something the data supports."""
        out = self.recall.empty_block(
            Outcome(candidates=26, best_dense=0.503, reranked=True, dropped=14), 2)
        self.assertNotIn(FLAT_CLAIM, out,
                         "asserting absence while warning of unjudged candidates is self-contradictory")
        self.assertIn(HEDGE, out)
        self.assertIn("14 candidate(s) went unjudged", out)

    def test_a_failed_rerank_withdraws_the_claim(self):
        out = self.recall.empty_block(
            Outcome(candidates=5, best_dense=0.5, rerank_error="timeout"), 2)
        self.assertNotIn(FLAT_CLAIM, out)
        self.assertIn(HEDGE, out)

    def test_a_collapse_withdraws_the_claim(self):
        out = self.recall.empty_block(
            Outcome(candidates=5, best_dense=0.46, reranked=True, collapsed=True,
                    best_rerank=0.0004), 2)
        self.assertNotIn(FLAT_CLAIM, out)
        self.assertIn(HEDGE, out)
        self.assertIn("0.0004", out)

    def test_the_note_and_the_conclusion_cannot_disagree(self):
        """The structural invariant, stated once so it holds for degradations added later.

        A new kind of partial judgement only has to be taught to `_degradation_note`; the
        conclusion follows from whether that note exists. This test fails if someone
        reintroduces a separate condition for choosing the conclusion.
        """
        cases = [
            Outcome(candidates=3, best_dense=0.2, reranked=True),
            Outcome(candidates=26, best_dense=0.5, reranked=True, dropped=14),
            Outcome(candidates=5, best_dense=0.5, rerank_error="boom"),
            Outcome(candidates=5, best_dense=0.46, reranked=True, collapsed=True),
        ]
        for outcome in cases:
            note = self.recall._degradation_note(outcome)
            out = self.recall.empty_block(outcome, 2)
            if note:
                self.assertNotIn(FLAT_CLAIM, out, f"note present but claim flat: {note!r}")
            else:
                self.assertIn(FLAT_CLAIM, out, "no degradation, so the hard claim is earned")


class TestDegradationNote(unittest.TestCase):
    def setUp(self):
        self.recall = load_hook()

    def test_a_complete_pipeline_gets_no_note(self):
        self.assertEqual(self.recall._degradation_note(Outcome(candidates=3, reranked=True)), "")

    def test_dropped_candidates_are_silent_when_the_slots_were_filled(self):
        """Warning on every prompt is crying wolf: what got dropped is the lowest-scoring
        tail, cut by design, and if the slots filled up the list was never going to
        include it anyway."""
        full = Outcome(candidates=40, reranked=True, dropped=28,
                       scored=[Scored({}, 0.9, CE)] * self.recall.MAX_MEMORIES)
        self.assertEqual(self.recall._degradation_note(full), "")

    def test_a_failed_rerank_says_the_strict_cut_came_back(self):
        note = self.recall._degradation_note(
            Outcome(candidates=5, rerank_error="HTTP 503 on POST /rerank"))
        self.assertIn("did NOT run", note)
        self.assertIn("strict cut was reapplied", note,
                      "the reader has to know the floor moved, not just that something failed")

    def test_a_collapse_is_reported_as_a_language_problem_not_irrelevance(self):
        note = self.recall._degradation_note(
            Outcome(candidates=5, reranked=True, collapsed=True, best_rerank=0.0004))
        self.assertIn("collapsed", note)
        self.assertIn("different languages", note)


class TestBuildBlock(unittest.TestCase):
    """The populated block: what the model is told to do with what it received."""

    def setUp(self):
        self.recall = load_hook()

    def _hit(self, mid="abc123", doc="a durable fact", origin=CE, score=0.9, meta=None):
        class Hit:
            id = mid
            document = doc
            metadata = meta or {"type": "reference", "date": "2026-08-13"}
        h = Hit()
        h.origin, h.score = origin, score

        return h

    def test_the_rules_of_use_travel_with_the_memories(self):
        out = self.recall.build_block([self._hit()], [], 3, Outcome(candidates=1, reranked=True))
        self.assertIn(self.recall.INSTRUCTIONS, out,
                      "a memory without its rules of use gets applied out of context")
        self.assertIn("a durable fact", out)
        self.assertIn("abc123", out, "the id is how the model retrieves the rest")

    def test_dense_origin_is_marked_so_it_is_not_read_as_a_verdict(self):
        out = self.recall.build_block([self._hit(origin=DENSE, score=0.61)], [], 2,
                                      Outcome(candidates=1))
        self.assertIn(DENSE, out)

    def test_pointers_say_why_they_are_not_included_in_full(self):
        out = self.recall.build_block([self._hit()], [self._hit(mid="ptr9", doc="x " * 200)],
                                      2, Outcome(candidates=2, reranked=True))
        self.assertIn("ptr9", out)
        self.assertIn("retrieve them by", out)

    def test_an_oversized_memory_is_truncated_and_says_so_with_its_id(self):
        big = self._hit(mid="big1", doc="y" * (self.recall.MAX_PER_MEM + 500))
        out = self.recall.build_block([big], [], 2, Outcome(candidates=1, reranked=True))
        self.assertIn("truncated at", out)
        self.assertIn("big1", out, "truncation is only recoverable if the id is given")
        self.assertLess(len(out), self.recall.MAX_PER_MEM + 3000)

    def test_a_partial_judgement_is_carried_into_the_populated_block_too(self):
        out = self.recall.build_block([self._hit()], [], 2,
                                      Outcome(candidates=30, reranked=True, dropped=18))
        self.assertIn("partial judgement", out)


class TestUnavailableBlock(unittest.TestCase):
    """The contract that matters most: silence for the user, never for the model."""

    def setUp(self):
        self.recall = load_hook()

    def test_it_forbids_concluding_absence_of_precedent(self):
        out = self.recall.unavailable_block("embeddings", "HttpError")
        self.assertIn("UNAVAILABLE", out)
        self.assertIn("was not consulted", out)
        self.assertNotIn(FLAT_CLAIM, out)
        self.assertIn("embeddings", out, "the reader has to know WHICH stage failed")
        self.assertIn("HttpError", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
