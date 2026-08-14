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


class TestTheInstructionsTheModelActuallyRECEIVES(unittest.TestCase):
    """The four states exist to make three distinctions. These hold the distinctions.

    Everything else in this suite checks which BLOCK is emitted. That is not the same as
    checking that the block still tells the model what to DO — measured, five of the
    load-bearing instructions could be rewritten into their opposites with the whole suite
    green, including turning "the archive was not consulted" into "the archive is empty".
    A block that names itself UNAVAILABLE and then invites the reader to assume absence is
    worse than no block, because it carries the authority of having searched.

    Deliberately NOT word-for-word. This prose was reworded several times while the plugin
    was being built, and a test that fails on a comma makes it unmaintainable. Each assertion
    below names the CONCLUSION the model must be prevented from reaching, or the action it
    must be told to take, and matches only the fragment carrying it.
    """

    def test_the_unavailable_block_forbids_concluding_absence(self):
        """The whole point of this state. The search did not run, so silence carries no
        information — and the model must be told so in the imperative, not merely hinted at."""
        out = blocks.unavailable_block("embedding", "connection refused")
        self.assertIn("Do not claim", out, "the prohibition has to be an instruction")
        self.assertIn("unprecedented", out, "and it has to name what must not be claimed")
        self.assertNotIn(FLAT_CLAIM, out, "an unavailable search cannot assert absence")

    def test_the_unavailable_block_says_NOT_CONSULTED_and_never_EMPTY(self):
        """"Not consulted" and "empty" are opposite facts that read almost the same. The
        first is an outage; the second is evidence. Only one of them is true here."""
        out = blocks.unavailable_block("qdrant", "timeout")
        self.assertIn("was not consulted", out)
        self.assertNotIn("archive is empty", out)

    def test_the_unavailable_block_tells_the_model_to_say_it_is_without_memory(self):
        """Without this the model answers normally and the user never learns that the turn
        had no archive behind it. The failure becomes invisible exactly when it matters."""
        out = blocks.unavailable_block("rerank", "boom")
        self.assertIn("without memory", out)
        self.assertIn("tell the user", out.lower())

    def test_a_partial_judgement_demands_a_targeted_search(self):
        """Partial means candidates went unjudged. Saying so and stopping would leave the
        model with a shrug; the instruction is what converts the caution into an action."""
        outcome = Outcome(candidates=40, best_dense=0.5, reranked=True,
                          scored=[0.2] * 3, dropped_above_floor=14)
        out = blocks.empty_block(outcome, n_angles=3)
        self.assertIn(HEDGE, out)
        self.assertIn("targeted search", out)
        self.assertNotIn(FLAT_CLAIM, out)
        # The CONDITION, not just the instruction. Measured: asserting only "targeted search"
        # let the trigger be rewritten from "if the subject might have history" to "if you
        # feel like it" with this test still green — which turns an obligation the model owes
        # the archive into a preference it can decline. The instruction and the circumstance
        # that fires it are one requirement, so they get asserted together.
        #
        # A regex, not a literal, and that distinction was itself a review finding: pinning
        # the exact words "might have" also failed "could have history" and "if there may be
        # history on the subject" — synonyms that keep the obligation exactly where it
        # belongs. Fixing prose to the letter is the defect this class's docstring warns
        # about, and the first version of this line committed it.
        self.assertRegex(out, r"if (the subject|there) (might|may|could)",
                         "the search has to be owed to the SUBJECT, not left to inclination")

    def test_a_clean_empty_search_is_the_ONLY_state_that_asserts_absence(self):
        """The counterpart. If this one hedged too, the four states would collapse into one
        undifferentiated shrug and the model could never conclude anything."""
        out = blocks.empty_block(Outcome(candidates=0, best_dense=0.1), n_angles=3)
        self.assertIn(FLAT_CLAIM, out)
        self.assertNotIn(HEDGE, out)
