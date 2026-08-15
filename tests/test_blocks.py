"""Prose and block assembly live in core, so both hosts render the same text.

They used to live in hooks/, which is the claude-code adapter. A second host would have
had to copy them, and copies drift on the first fix — this repo already paid that bill
with three copies of the two-stage pipeline.
"""
import os
import re
import sys
import unittest
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import blocks, prompts
from core.retrieval import CE, DENSE, Outcome


class TestPrompts(unittest.TestCase):
    def test_instructions_carry_the_four_rules(self):
        for fragment in ("PREVAILS", "the measurement wins", "another angle"):
            self.assertIn(fragment, prompts.INSTRUCTIONS)
        # A regex and not the literal sentence, for the reason the class below states at
        # length: "VERIFY it against the current SOURCE tree" is the same rule with one word
        # added, and it failed the literal pin. A pin that an honest synonym trips is a pin
        # that gets deleted the third time it cries wolf, taking the real coverage with it.
        self.assertRegex(prompts.INSTRUCTIONS, r"VERIFY it against the current[\w ]*tree",
                         "the rule that a cited file, line or flag must be re-checked")

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
        information — and the model must be told so in the imperative, not merely hinted at.

        Both halves are regexes now. Measured: "Never claim anything is without precedent"
        failed the literal pins on "Do not claim" and "unprecedented" while forbidding exactly
        the same conclusion in the same imperative. What is required is a PROHIBITION that
        NAMES the claim, not two particular strings.
        """
        out = blocks.unavailable_block("embedding", "connection refused")
        self.assertRegex(out, r"(?i)\b(do not|don't|never)\s+(claim|assert|state|say)\b",
                         "the prohibition has to be an instruction")
        self.assertRegex(out, r"(?i)unprecedented|without (a |any )?(precedent|history)",
                         "and it has to name what must not be claimed")
        self.assertNotIn(FLAT_CLAIM, out, "an unavailable search cannot assert absence")

    def test_the_unavailable_block_says_NOT_CONSULTED_and_never_EMPTY(self):
        """"Not consulted" and "empty" are opposite facts that read almost the same. The
        first is an outage; the second is evidence. Only one of them is true here."""
        out = blocks.unavailable_block("qdrant", "timeout")
        self.assertIn("was not consulted", out)
        self.assertNotIn("archive is empty", out)

    def test_the_unavailable_block_denies_the_absence_READING_and_not_only_the_claim(self):
        """The prohibition and the DENIAL are two requirements, and only the first was pinned.

        Measured: "This does NOT mean there is no precedent" could be rewritten into "This
        probably means there is no precedent, but strictly it means the archive was not
        consulted" with the whole suite green — every pinned fragment still present ("was not
        consulted", "Do not claim", "unprecedented") and the block nonetheless inviting the
        reader to assume absence. A prohibition the same paragraph then concedes is not a
        prohibition; the reader keeps the concession.
        """
        out = blocks.unavailable_block("qdrant", "timeout")
        self.assertRegex(out, r"(?i)(does NOT mean|is not evidence|says nothing about|"
                              r"carries no information|proves nothing)",
                         "the block has to deny the absence reading outright, not merely "
                         "forbid stating it")

    def test_the_unavailable_block_tells_the_model_to_say_it_is_without_memory(self):
        """Without this the model answers normally and the user never learns that the turn
        had no archive behind it. The failure becomes invisible exactly when it matters.

        Regexes for the same reason as above: "say to the user that you have no long-term
        memory" is the identical instruction and failed the literal pins.
        """
        out = blocks.unavailable_block("rerank", "boom")
        self.assertRegex(out, r"(?i)(tell|say to|inform) the user")
        self.assertRegex(out, r"(?i)you (are|have) (no |without )(long-term |durable )?memory",
                         "and what it has to tell them is that THIS TURN had no archive")

    def test_a_partial_judgement_demands_a_targeted_search(self):
        """Partial means candidates went unjudged. Saying so and stopping would leave the
        model with a shrug; the instruction is what converts the caution into an action."""
        outcome = Outcome(candidates=40, best_dense=0.5, reranked=True,
                          scored=[0.2] * 3, dropped_above_floor=14)
        out = blocks.empty_block(outcome, n_angles=3)
        self.assertIn(HEDGE, out)
        # A regex: "run a specific search" is the same instruction and failed the literal
        # "targeted search". What is load-bearing is that the search be NARROWED — the
        # generic one already ran — not the adjective chosen for it.
        self.assertRegex(out, r"(?i)(targeted|specific|explicit|focused|narrower?) search")
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

    #: Words that hand the forbidden inference straight back, or turn a fact about what the
    #: harness DID into a guess. A block whose job is "do not conclude absence from this
    #: silence" cannot also say the silence probably means absence — the reader keeps the
    #: second sentence — and a block delivering memories cannot say they are probably from
    #: earlier sessions, which is not a thing anyone is uncertain about. Two measured
    #: mutations survived the whole suite by ADDING one of these, with every pinned fragment
    #: still in place; hence a pin on the block's stance rather than on its fragments.
    CONCESSIONS = re.compile(r"(?i)\b(probably|likely|usually|typically|generally|"
                             r"in practice|almost always|most of the time|"
                             r"nine times out of ten|for all practical purposes)\b")

    #: The adverb list above is necessary and NOT sufficient, and a review proved it by
    #: walking through: twelve sentences that hand the reader the forbidden inference with no
    #: hedging word at all — "treat the subject as one with no recorded precedent", "assume
    #: nothing is stored on the subject and move on", "this list is exhaustive: whatever is
    #: not below is not in the archive" — all passed. A closed vocabulary is one imperative
    #: away from useless, and the docstring below it CLAIMED to catch added contradictions,
    #: which made the gap read as covered.
    #:
    #: So the second pattern matches the ACT rather than the wording: telling the reader to
    #: treat, assume, proceed or answer AS IF the subject had none. That is what every escape
    #: had in common, and it is the thing these blocks exist to forbid.
    LICENCES = re.compile(
        r"(?i)("
        r"(treat|regard|consider)\b[^.]{0,60}\bas (one with no|having no|without)|"
        r"assume\b[^.]{0,40}\b(nothing|no precedent|no history|not (stored|recorded))|"
        r"(proceed|answer|reply|respond|move on)\b[^.]{0,60}"
        r"\bas (if|though)\b[^.]{0,40}\b(no|nothing|none)\b|"
        r"\b(is|are) exhaustive\b|"
        r"\bwhat(ever)? is (not|missing)\b[^.]{0,40}\b(not in|never (recorded|stored))"
        r")")

    # -- the POPULATED state, which had no pin on its framing at all ----------------------
    #
    # It is the state that fires most often, and until this review nothing held the two
    # sentences that make the injection worth its context: the order to read the memories,
    # and the claim that the search really happened.

    def _framing(self):
        """A populated block's own framing — everything before the shared INSTRUCTIONS.

        Split rather than asserted over the whole block on purpose: INSTRUCTIONS is pinned by
        TestPrompts and by test_host_equivalence, and it legitimately contains conditionals
        ("if you think it should change…") that the negative pins below must not read as the
        framing offering the memories as optional.
        """
        out = blocks.recall_block([FakeHit()], [], 2, Outcome(candidates=1, reranked=True),
                                  BUDGET)
        head, _, _ = out.partition(prompts.INSTRUCTIONS)
        self.assertTrue(head.strip(), "the framing disappeared, or INSTRUCTIONS moved")

        return head

    def test_the_populated_block_ORDERS_the_reading_and_does_not_offer_it(self):
        """Measured: "read it BEFORE answering, investigating or proposing a design" could be
        rewritten to "read it only if it looks relevant to you" with the whole suite green.

        That single edit undoes the reason automatic recall exists. Leaving the reading to the
        model's discretion is the very situation the hook was built to replace — reading is the
        direction that gets skipped, because nothing fails visibly when it does. An unread
        memory costs exactly as much context as a read one and buys nothing.
        """
        head = self._framing()
        self.assertRegex(head, r"(?i)read (it|them|these)[^.]{0,40}before (answering|acting)",
                         "the reading has to be ordered, and ordered BEFORE the answer")
        self.assertNotRegex(head, r"(?i)(only if|if it (looks|seems)|if you (feel|want|"
                                  r"prefer|like)|at your discretion|optional)",
                            "the framing turned the order into an invitation")

    def test_the_populated_block_says_the_search_REALLY_RAN(self):
        """"This search was EXECUTED by the harness" → "was attempted by the harness" also
        survived the whole suite. It is a small word and it changes what the model may
        conclude from the block: an attempt that may not have run cannot ground anything, so
        the memories below it become hearsay and their absence becomes uninformative — which
        is the unavailable state's message smuggled into the populated one.
        """
        head = self._framing()
        self.assertRegex(head, r"(?i)this search was (EXECUTED|run|performed|carried out)",
                         "the block has to state that the search actually happened")
        self.assertNotRegex(head, r"(?i)(attempted|tried|may have (run|been run)|"
                                  r"if it ran)",
                            "a search reported as merely attempted grounds nothing")
        # The same stance check the unavailable and partial blocks get below, and found the
        # same way: "What follows PROBABLY is knowledge from earlier sessions" passed every
        # other assertion here. The provenance of an injected memory is not a guess.
        self.assertEqual(self.CONCESSIONS.findall(head), [],
                         "the framing hedges what the harness actually did")

    # -- no block may take back the prohibition it just issued ----------------------------

    def NO_HEDGING_CASES(self):
        return {
            "unavailable": blocks.unavailable_block("embeddings", "HttpError"),
            "partial/rerank-error": blocks.empty_block(
                Outcome(candidates=5, best_dense=0.5, rerank_error="timeout"), 2),
            "partial/breaker": blocks.empty_block(
                Outcome(candidates=27, best_dense=0.54,
                        suppressed="circuit breaker: the re-rank failed 12s ago"), 2),
            "partial/collapsed": blocks.empty_block(
                Outcome(candidates=5, best_dense=0.46, reranked=True, collapsed=True), 2),
            "partial/dropped": blocks.empty_block(
                Outcome(candidates=26, best_dense=0.5, reranked=True, dropped=14,
                        dropped_above_floor=3), 2),
        }

    def test_no_block_that_forbids_ABSENCE_may_then_concede_it(self):
        """The pin the review asked for by name: one that catches a CONTRADICTING SENTENCE
        added elsewhere in the same block, not only the removal of a fragment.

        Measured survivors: appending "In practice, though, an empty result almost always
        means nothing is stored on it." to the hedged conclusion, and rewriting "This does NOT
        mean there is no precedent" into "This probably means there is no precedent, but
        strictly…". Both passed 544 tests. Every fragment-based pin is blind to them by
        construction, because nothing was removed.

        TWO patterns, because the first version of this test used only the adverb list and a
        later review walked twelve sentences straight through it — every one of them
        conceding absence outright, none of them hedging. Worse than the gap was this
        docstring, which already claimed the added-contradiction case was covered. A test
        that overstates its own reach converts a hole into documented assurance, which is the
        defect this suite has now caught six times.
        """
        for label, out in self.NO_HEDGING_CASES().items():
            with self.subTest(state=label):
                hedges = self.CONCESSIONS.findall(out)
                self.assertEqual(hedges, [],
                                 f"the {label} block forbids concluding absence and then "
                                 f"concedes it with {hedges}: the reader keeps the concession")
                licences = self.LICENCES.findall(out)
                self.assertEqual(licences, [],
                                 f"the {label} block forbids concluding absence and then "
                                 f"LICENSES it outright ({licences}) — no hedging word "
                                 f"needed, which is how twelve such sentences got through")

    def test_the_degradation_note_says_the_SECOND_STAGE_DID_NOT_HAPPEN(self):
        """"the re-rank was not used" → "was not needed" survived the whole suite, and it
        inverts the note's meaning: not needed says the pipeline was fine, which is the exact
        opposite of the CAUTION the note exists to raise. A judgement is partial because
        something did not happen, never because it was superfluous.
        """
        for outcome in (Outcome(candidates=5, best_dense=0.5, rerank_error="timeout"),
                        Outcome(candidates=27, best_dense=0.54,
                                suppressed="circuit breaker: the re-rank failed 12s ago")):
            with self.subTest(outcome=repr(outcome)):
                note = blocks.degradation_note(outcome, BUDGET.max_memories)
                self.assertIn("partial judgement", note)
                self.assertRegex(note, r"(?i)re-rank (was not (used|applied|run)|did not run|"
                                       r"was skipped|was held back|never ran)",
                                 "the note has to say the stage did not happen")
                self.assertNotRegex(note, r"(?i)not (needed|necessary|required)|"
                                          r"unnecessary|superfluous|redundant",
                                    "a stage described as unneeded is not a caution")


if __name__ == "__main__":
    # Without this, `python3 tests/test_blocks.py` printed nothing and exited 0 — a file
    # that reports "no tests" as success is worse than one that fails, and this file holds
    # the pins on what the model is told about an archive that could not be searched.
    unittest.main()
