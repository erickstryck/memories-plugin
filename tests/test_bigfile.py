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

    def test_used_at_or_above_the_window_means_the_window_was_wrong(self):
        """A session cannot use more of the window than the window holds, so this does not
        describe a full context — it REFUTES the number. Measured before this rule existed:
        `window_for('claude-opus-5')` returned 200k in a real 1M session whose `used` was
        989,479, `free` collapsed to 0, and the guard denied a 4 KB file. Erring large
        costs a sleeping guard; erring small cost a cage.
        """
        path = a_file(4000)
        for used in (100, 500):          # exactly at the window, and past it
            with self.subTest(used=used):
                v = bigfile.decide(path, Budget(window=100, used=used, exact=False))
                self.assertFalse(v.block, "a refuted window must allow, never block")
                self.assertGreaterEqual(v.free, 0)

    def test_the_floor_blocks_alone_with_the_window_intact(self):
        """THE ISOLATOR for the final-remainder floor, and it has to be its own fixture.

        `test_the_final_remainder_floor_blocks` trips BOTH criteria at once (F4), and the
        test that used to isolate the floor did it by driving `free` to 0 — which now
        returns early as a refuted window. So the isolation is rebuilt with the window
        intact: 790k used of 1M, a file worth 20k tokens. After the read that is 810k of
        1M, past the 800k the floor allows; and 20k is well under 40% of the 210k free, so
        the share cannot be what fired.
        """
        path = a_file(4 * 20_000)                             # 20k tokens
        b = Budget(window=1_000_000, used=790_000, exact=True)
        self.assertLess(b.used, b.window, "the window must NOT be refuted here")
        self.assertGreater(790_000 + 20_000, 1_000_000 * 0.8, "the floor must fire")
        self.assertLessEqual(20_000, 210_000 * 0.4, "the share must NOT fire")
        self.assertTrue(bigfile.decide(path, b).block)


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


from core import docs as core_docs


class TestSpecialCases(unittest.TestCase):
    def test_a_binary_file_is_allowed_because_indexing_it_is_not_an_option(self):
        """`docs_index` slices TEXT. Telling the model to index a binary is wrong advice,
        and blocking without an alternative is just a wall."""
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as fh:
            fh.write(b"\x00\x01\x02" * 400_000)
        v = bigfile.decide(path, Budget(window=200_000, used=190_000, exact=True))
        self.assertFalse(v.block)

    def test_a_text_file_is_indexable(self):
        self.assertTrue(bigfile.is_indexable(a_file(100)))

    def test_a_high_entropy_file_with_no_null_byte_is_not_indexable(self):
        """The incident `is_probably_binary` was written for: 20 KB of high random bytes
        with no NUL produced chunks of U+FFFD, spending embedding calls on noise. A NUL
        scan alone waves this through; the shared policy in `core.chunk` does not."""
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as fh:
            fh.write(bytes((b or 1) for b in os.urandom(20_000)))   # no zero bytes
        self.assertFalse(bigfile.is_indexable(path))

    def test_decide_allows_a_no_null_high_entropy_file_instead_of_telling_it_to_index_noise(self):
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as fh:
            fh.write(bytes((b or 1) for b in os.urandom(1_200_000)))   # no zero bytes
        v = bigfile.decide(path, Budget(window=200_000, used=190_000, exact=True))
        self.assertFalse(v.block)

    def test_an_already_indexed_file_is_told_to_SEARCH_not_to_reindex(self):
        """Reindexing 258 chunks the archive already holds is waste, and the model would
        do it because the message told it to."""
        path = a_file(4 * 171_000)
        known = {core_docs.doc_id_for(path)}
        v = bigfile.decide(path, Budget(window=1_000_000, used=604_023, exact=True),
                           indexed_ids=known)
        self.assertTrue(v.block)
        self.assertIn("already indexed", v.reason)
        self.assertIn(core_docs.doc_id_for(path), v.reason)

    def test_a_file_not_yet_indexed_is_told_to_INDEX(self):
        path = a_file(4 * 171_000)
        v = bigfile.decide(path, Budget(window=1_000_000, used=604_023, exact=True),
                           indexed_ids=set())
        self.assertTrue(v.block)
        self.assertIn("docs_index", v.reason)

    def test_the_message_names_the_escape(self):
        """A block with no way out is a cage. The message has to carry its own key."""
        path = a_file(4 * 171_000)
        v = bigfile.decide(path, Budget(window=1_000_000, used=604_023, exact=True))
        self.assertIn("--full", v.reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
