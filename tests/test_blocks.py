"""Prose and block assembly live in core, so both hosts render the same text.

They used to live in hooks/, which is the claude-code adapter. A second host would have
had to copy them, and copies drift on the first fix — this repo already paid that bill
with three copies of the two-stage pipeline.
"""
import os
import sys
import unittest
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import blocks, prompts
from core.retrieval import CE, DENSE, Outcome


class TestPrompts(unittest.TestCase):
    def test_instructions_carry_the_four_rules(self):
        for fragment in ("PREVAILS", "VERIFY it against the current tree",
                         "the measurement wins", "another angle"):
            self.assertIn(fragment, prompts.INSTRUCTIONS)

    def test_checkpoint_procedure_formats_without_leftover_braces(self):
        rendered = prompts.CHECKPOINT_PROCEDURE.format(count=5, interval=5)
        self.assertIn("Interaction 5 of this conversation (every 5)", rendered)
        head = rendered.split("Mandatory metadata")[0]
        self.assertNotIn("{", head, "an unfilled placeholder means the format keys drifted")

    def test_the_metadata_example_survives_formatting(self):
        rendered = prompts.CHECKPOINT_PROCEDURE.format(count=1, interval=1)
        self.assertIn('{"type": "user|feedback|project|reference"', rendered)

    def test_the_five_steps_are_all_there(self):
        for step in ("1. SWEEP", "2. DEDUPE", "3. FIX", "4. WRITE", "5. CONFIRM"):
            self.assertIn(step, prompts.CHECKPOINT_PROCEDURE)


FLAT_CLAIM = "There is no recorded precedent"
HEDGE = "not evidence that no precedent exists"
BUDGET = blocks.Budget(max_memories=6, max_chars=14000, max_per_mem=4500, reinject_after=8)


@dataclass
class FakeHit:
    """Structural stand-in for core.memory.Recalled — blocks only reads these fields."""
    id: str = "abc123"
    document: str = "a durable fact"
    origin: str = CE
    score: float = 0.9
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {"type": "reference", "date": "2026-08-14"}


class TestTheFourStates(unittest.TestCase):
    def test_populated_block_carries_the_rules_and_the_id(self):
        out = blocks.recall_block([FakeHit()], [], 2, Outcome(candidates=1, reranked=True), BUDGET)
        self.assertIn(prompts.INSTRUCTIONS, out)
        self.assertIn("abc123", out)
        self.assertIn("a durable fact", out)

    def test_a_complete_empty_search_may_claim_no_precedent(self):
        out = blocks.empty_block(Outcome(candidates=4, best_dense=0.31, reranked=True), 3)
        self.assertIn(FLAT_CLAIM, out)
        self.assertIn("0.310", out)

    def test_a_partial_judgement_withdraws_the_claim(self):
        for outcome in (
            Outcome(candidates=5, best_dense=0.5, rerank_error="timeout"),
            Outcome(candidates=5, best_dense=0.46, reranked=True, collapsed=True),
            Outcome(candidates=27, best_dense=0.54, suppressed="circuit breaker: 12s ago"),
            Outcome(candidates=26, best_dense=0.5, reranked=True, dropped=14, dropped_above_floor=3),
        ):
            out = blocks.empty_block(outcome, 2)
            self.assertNotIn(FLAT_CLAIM, out, repr(outcome))
            self.assertIn(HEDGE, out)

    def test_the_note_and_the_conclusion_cannot_disagree(self):
        cases = [
            Outcome(candidates=3, best_dense=0.2, reranked=True),
            Outcome(candidates=26, best_dense=0.5, reranked=True, dropped=14),
            Outcome(candidates=5, best_dense=0.5, rerank_error="boom"),
            Outcome(candidates=27, best_dense=0.54, suppressed="breaker"),
        ]
        for outcome in cases:
            note = blocks.degradation_note(outcome, BUDGET.max_memories)
            out = blocks.empty_block(outcome, 2)
            if note:
                self.assertNotIn(FLAT_CLAIM, out, f"note present but claim flat: {note!r}")
            else:
                self.assertIn(FLAT_CLAIM, out)

    def test_unavailable_forbids_concluding_absence(self):
        out = blocks.unavailable_block("embeddings", "HttpError")
        self.assertIn("UNAVAILABLE", out)
        self.assertIn("was not consulted", out)
        self.assertNotIn(FLAT_CLAIM, out)
        self.assertIn("embeddings", out)


class TestBudget(unittest.TestCase):
    def test_a_recently_seen_memory_becomes_a_pointer(self):
        hits = [FakeHit(id="old"), FakeHit(id="new")]
        seen = {"old": 10}
        full, pointers = blocks.split_by_budget(hits, seen, round_no=12, budget=BUDGET)
        self.assertEqual([h.id for h in full], ["new"])
        self.assertEqual([h.id for h in pointers], ["old"])

    def test_an_old_enough_memory_comes_back_in_full(self):
        seen = {"old": 1}
        full, _ = blocks.split_by_budget([FakeHit(id="old")], seen, round_no=12, budget=BUDGET)
        self.assertEqual([h.id for h in full], ["old"], "12-1 >= reinject_after")

    def test_what_goes_in_full_is_marked_seen_at_this_round(self):
        seen = {}
        blocks.split_by_budget([FakeHit(id="a")], seen, round_no=7, budget=BUDGET)
        self.assertEqual(seen, {"a": 7})

    def test_the_char_budget_stops_before_the_slot_ceiling(self):
        tight = blocks.Budget(max_memories=6, max_chars=100, max_per_mem=4500, reinject_after=8)
        hits = [FakeHit(id=str(i), document="x" * 80) for i in range(3)]
        full, pointers = blocks.split_by_budget(hits, {}, round_no=1, budget=tight)
        self.assertEqual(len(full), 1)
        self.assertEqual(len(pointers), 2)

    def test_the_slot_ceiling_stops_before_the_char_budget(self):
        two = blocks.Budget(max_memories=2, max_chars=99999, max_per_mem=4500, reinject_after=8)
        hits = [FakeHit(id=str(i)) for i in range(5)]
        full, pointers = blocks.split_by_budget(hits, {}, round_no=1, budget=two)
        self.assertEqual(len(full), 2)
        self.assertEqual(len(pointers), 3)

    def test_an_oversized_memory_is_truncated_and_names_its_id(self):
        big = FakeHit(id="big1", document="y" * 9000)
        out = blocks.recall_block([big], [], 2, Outcome(candidates=1, reranked=True), BUDGET)
        self.assertIn("truncated at", out)
        self.assertIn("big1", out)



class TestBudgetWithACorruptedSeen(unittest.TestCase):
    """`seen` is loaded from a state file the caller does not fully control (a hand-edited
    file, a future format change). A non-dict `seen` must degrade to "nothing has been
    seen" — everything goes to `full` subject to the budget, nothing raises, nothing marks
    a memory as seen (there is nothing sane to mark it ON). Losing the dedup memory costs
    one memory reinjected once; it must never cost the search that already ran.
    """

    def test_a_string_seen_treats_everything_as_unseen(self):
        hits = [FakeHit(id="a"), FakeHit(id="b")]
        full, pointers = blocks.split_by_budget(hits, "corrupted", round_no=5, budget=BUDGET)
        self.assertEqual([h.id for h in full], ["a", "b"])
        self.assertEqual(pointers, [])

    def test_a_list_seen_treats_everything_as_unseen(self):
        full, _ = blocks.split_by_budget([FakeHit(id="a")], ["a"], round_no=5, budget=BUDGET)
        self.assertEqual([h.id for h in full], ["a"])

    def test_a_None_seen_treats_everything_as_unseen(self):
        full, _ = blocks.split_by_budget([FakeHit(id="a")], None, round_no=5, budget=BUDGET)
        self.assertEqual([h.id for h in full], ["a"])
