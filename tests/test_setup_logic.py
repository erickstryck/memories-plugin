"""Tests for the wizard's logic.

Only `choose_by_index` has any logic; the rest of `setup` is network I/O and terminal
reading. Testing the pure part and keeping the shell trivial is what makes the wizard
dependable without needing a TTY in the test.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.setup import choose_by_index


class TestChoice(unittest.TestCase):
    def setUp(self):
        self.options = ["claude_memory", "hermes_memory", "other"]

    def test_a_number_selects_by_position(self):
        self.assertEqual(choose_by_index(self.options, "2"), "hermes_memory")

    def test_numbering_starts_at_1(self):
        self.assertEqual(choose_by_index(self.options, "1"), "claude_memory")

    def test_empty_means_keep_the_current_value(self):
        for entry in ("", "   ", None):
            self.assertIsNone(choose_by_index(self.options, entry))

    def test_a_number_outside_the_list_selects_nothing(self):
        for entry in ("0", "4", "99"):
            self.assertIsNone(choose_by_index(self.options, entry),
                              "an invalid index must not become a collection name")

    def test_a_typed_name_is_accepted_even_if_not_listed(self):
        self.assertEqual(choose_by_index(self.options, "new_collection"), "new_collection")

    def test_surrounding_whitespace_is_ignored(self):
        self.assertEqual(choose_by_index(self.options, "  2  "), "hermes_memory")
        self.assertEqual(choose_by_index(self.options, " name "), "name")

    def test_empty_list_with_a_number(self):
        self.assertIsNone(choose_by_index([], "1"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
