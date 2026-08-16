import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import bigfile
from core.bigfile import Budget


def a_file(size_bytes: int, suffix: str = ".txt") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as fh:
        fh.write("x" * size_bytes)

    return path


class TestCostOf(unittest.TestCase):
    def test_cost_is_derived_from_size_without_reading_the_file(self):
        """The whole point is not paying for the file. Reading it to measure it would
        defeat the guard on the very path it protects."""
        path = a_file(4000)
        self.assertEqual(bigfile.cost_of(path), 1000)   # 4000 / CHARS_PER_TOKEN

    def test_a_missing_file_costs_zero(self):
        """Fail open: the read will fail on its own, with a better message than ours."""
        self.assertEqual(bigfile.cost_of("/nonexistent/nope.txt"), 0)


class TestTheTwoCriteria(unittest.TestCase):
    def test_a_small_file_in_a_fresh_window_is_allowed(self):
        path = a_file(4000)                                   # 1k tokens
        v = bigfile.decide(path, Budget(window=1_000_000, used=10_000, exact=True))
        self.assertFalse(v.block)

    def test_the_final_remainder_floor_blocks(self):
        """After reading, less than 20% of the window would remain."""
        path = a_file(4000 * 100)                             # 100k tokens
        v = bigfile.decide(path, Budget(window=200_000, used=90_000, exact=True))
        self.assertTrue(v.block)          # 90k + 100k = 190k of 200k -> 5% left

    def test_the_share_of_free_blocks(self):
        """The measured case that motivated the guard: 604,023 used of 1M, a file worth
        ~171k. The floor does NOT fire (775k of 1M leaves 22%), the share does
        (171/396 = 43% > 40%)."""
        path = a_file(4 * 171_000)
        v = bigfile.decide(path, Budget(window=1_000_000, used=604_023, exact=True))
        self.assertTrue(v.block)
        self.assertIn("43%", v.reason)

    def test_neither_criterion_fires_just_below_both(self):
        path = a_file(4 * 100_000)                            # 100k tokens
        v = bigfile.decide(path, Budget(window=1_000_000, used=100_000, exact=True))
        self.assertFalse(v.block)         # 200k of 1M left 80%; 100/900 = 11%

    def test_an_unknown_window_allows(self):
        """window=0 means we could not learn it. Blocking on a guessed window is the one
        failure this guard must not produce."""
        path = a_file(4 * 900_000)
        v = bigfile.decide(path, Budget(window=0, used=0, exact=False))
        self.assertFalse(v.block)

    def test_used_above_window_does_not_divide_by_a_negative(self):
        """Defensive: the hermes estimate can drift above a mis-declared window."""
        path = a_file(4000)
        v = bigfile.decide(path, Budget(window=100, used=500, exact=False))
        self.assertTrue(v.block)
        self.assertGreaterEqual(v.free, 0)


class TestTheNumbersInTheMessage(unittest.TestCase):
    def test_an_estimated_budget_marks_the_number_as_approximate(self):
        """hermes cannot measure the context; it sums message bodies. A number that looks
        exact and is a guess is worse than an admitted guess."""
        path = a_file(4 * 171_000)
        v = bigfile.decide(path, Budget(window=1_000_000, used=604_023, exact=False))
        self.assertIn("≈", v.reason)

    def test_an_exact_budget_does_not(self):
        path = a_file(4 * 171_000)
        v = bigfile.decide(path, Budget(window=1_000_000, used=604_023, exact=True))
        self.assertNotIn("≈", v.reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
