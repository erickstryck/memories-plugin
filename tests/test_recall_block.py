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

`empty_block` and `unavailable_block` are still called through the hook, because it still
delegates to them directly and that delegation is itself the thing worth pinning. The
degradation-note and populated-block logic moved into `core.blocks` (see tests/test_blocks.py
for its own coverage) and lost its `hooks.recall` names in the process, so those calls go
straight to `core.blocks` / `core.prompts` here too — same assertions, new address.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks"))

from core import blocks, prompts
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

    def test_unjudged_candidates_ABOVE_THE_FLOOR_withdraw_the_claim(self):
        """The live defect. 26 candidates, 12 judged, none passed, and 3 of the unjudged
        ones would have survived on dense score alone — so "no precedent" is not something
        the data supports."""
        out = self.recall.empty_block(
            Outcome(candidates=26, best_dense=0.503, reranked=True,
                    dropped=14, dropped_above_floor=3), 2)
        self.assertNotIn(FLAT_CLAIM, out,
                         "asserting absence while warning of unjudged candidates is self-contradictory")
        self.assertIn(HEDGE, out)
        self.assertIn("3 candidate(s) that clear the dense floor", out)

    def test_an_unjudged_tail_BELOW_the_floor_is_not_worth_a_word(self):
        """The other half, and the reason the previous version was noise.

        Candidates arrive dense-sorted, so the pair ceiling always cuts the tail. A tail
        entirely below the strict floor could not have held anything the single-stage mode
        would have returned. Measured before this discrimination existed: a real prompt with
        27 candidates and a best dense score of 0.510 — every one below the 0.58 floor —
        still announced "21 candidate(s) went unjudged … there may be relevant memory
        outside". Production runs 25-40 candidates against a ceiling of 12, so it fired on
        essentially every empty result.
        """
        out = self.recall.empty_block(
            Outcome(candidates=27, best_dense=0.510, reranked=True,
                    dropped=15, dropped_above_floor=0), 2)
        self.assertNotIn("went unjudged", out)
        self.assertIn(FLAT_CLAIM, out, "a complete judgement earns the plain claim")

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
            Outcome(candidates=26, best_dense=0.5, reranked=True,
                    dropped=14, dropped_above_floor=2),
            Outcome(candidates=5, best_dense=0.5, rerank_error="boom"),
            Outcome(candidates=5, best_dense=0.46, reranked=True, collapsed=True),
            Outcome(candidates=27, best_dense=0.54, suppressed="circuit breaker: 12s ago"),
        ]
        for outcome in cases:
            note = blocks.degradation_note(outcome, self.recall.MAX_MEMORIES)
            out = self.recall.empty_block(outcome, 2)
            if note:
                self.assertNotIn(FLAT_CLAIM, out, f"note present but claim flat: {note!r}")
            else:
                self.assertIn(FLAT_CLAIM, out, "no degradation, so the hard claim is earned")


class TestDegradationNote(unittest.TestCase):
    def setUp(self):
        self.recall = load_hook()

    def test_a_complete_pipeline_gets_no_note(self):
        self.assertEqual(
            blocks.degradation_note(Outcome(candidates=3, reranked=True), self.recall.MAX_MEMORIES), "")

    def test_dropped_candidates_are_silent_when_the_slots_were_filled(self):
        """Warning on every prompt is crying wolf: what got dropped is the lowest-scoring
        tail, cut by design, and if the slots filled up the list was never going to
        include it anyway."""
        full = Outcome(candidates=40, reranked=True, dropped=28,
                       scored=[Scored({}, 0.9, CE)] * self.recall.MAX_MEMORIES)
        self.assertEqual(blocks.degradation_note(full, self.recall.MAX_MEMORIES), "")

    def test_a_failed_rerank_says_the_strict_cut_came_back(self):
        note = blocks.degradation_note(
            Outcome(candidates=5, rerank_error="HTTP 503 on POST /rerank"), self.recall.MAX_MEMORIES)
        self.assertIn("did NOT run", note)
        self.assertIn("strict cut was reapplied", note,
                      "the reader has to know the floor moved, not just that something failed")

    def test_a_collapse_names_both_causes_because_the_scores_cannot_tell_them_apart(self):
        """This used to assert the note blamed a language mismatch. It cannot know that.

        The same server scores a plainly irrelevant document at 1.6e-05, well inside the
        collapse band, so a crushed result is equally consistent with "different languages"
        and with "nothing is relevant". Naming only the first sends the reader looking for a
        translation problem that may not exist.
        """
        note = blocks.degradation_note(
            Outcome(candidates=5, reranked=True, collapsed=True, best_rerank=0.0004),
            self.recall.MAX_MEMORIES)
        self.assertIn("at or near zero", note)
        self.assertIn("different languages", note)
        self.assertIn("nothing is relevant", note,
                      "the note must not assert a cause the scores cannot establish")

    def test_a_suppressed_rerank_is_reported_even_though_nothing_errored(self):
        """The breaker case, which had no signal at all before `Outcome.suppressed`.

        `_run` sets `store.reranker = None` when the breaker is open, so the pipeline sees
        an absent reranker — indistinguishable, from inside, from a deployment that has
        none. No error, no collapse, no note, and the block then claimed no precedent
        exists. For 300 seconds after every rerank failure.
        """
        note = blocks.degradation_note(
            Outcome(candidates=27, best_dense=0.536,
                    suppressed="circuit breaker: the re-rank failed 12s ago"),
            self.recall.MAX_MEMORIES)
        self.assertIn("circuit breaker", note)
        self.assertIn("strict cut was reapplied", note)

    def test_the_breaker_withdraws_the_flat_claim(self):
        out = self.recall.empty_block(
            Outcome(candidates=27, best_dense=0.536,
                    suppressed="circuit breaker: the re-rank failed 12s ago"), 2)
        self.assertNotIn(FLAT_CLAIM, out)
        self.assertIn(HEDGE, out)


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
        out = blocks.recall_block([self._hit()], [], 3, Outcome(candidates=1, reranked=True),
                                  self.recall.BUDGET)
        self.assertIn(prompts.INSTRUCTIONS, out,
                      "a memory without its rules of use gets applied out of context")
        self.assertIn("a durable fact", out)
        self.assertIn("abc123", out, "the id is how the model retrieves the rest")

    def test_dense_origin_is_marked_so_it_is_not_read_as_a_verdict(self):
        out = blocks.recall_block([self._hit(origin=DENSE, score=0.61)], [], 2,
                                  Outcome(candidates=1), self.recall.BUDGET)
        self.assertIn(DENSE, out)

    def test_pointers_say_why_they_are_not_included_in_full(self):
        out = blocks.recall_block([self._hit()], [self._hit(mid="ptr9", doc="x " * 200)],
                                  2, Outcome(candidates=2, reranked=True), self.recall.BUDGET)
        self.assertIn("ptr9", out)
        self.assertIn("retrieve them by", out)

    def test_an_oversized_memory_is_truncated_and_says_so_with_its_id(self):
        big = self._hit(mid="big1", doc="y" * (self.recall.MAX_PER_MEM + 500))
        out = blocks.recall_block([big], [], 2, Outcome(candidates=1, reranked=True), self.recall.BUDGET)
        self.assertIn("truncated at", out)
        self.assertIn("big1", out, "truncation is only recoverable if the id is given")
        self.assertLess(len(out), self.recall.MAX_PER_MEM + 3000)

    def test_a_partial_judgement_is_carried_into_the_populated_block_too(self):
        out = blocks.recall_block([self._hit()], [], 2,
                                  Outcome(candidates=30, reranked=True,
                                          dropped=18, dropped_above_floor=4),
                                  self.recall.BUDGET)
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
