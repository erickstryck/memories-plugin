"""Prose and block assembly live in core, so both hosts render the same text.

They used to live in hooks/, which is the claude-code adapter. A second host would have
had to copy them, and copies drift on the first fix — this repo already paid that bill
with three copies of the two-stage pipeline.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import prompts


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
