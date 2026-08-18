import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import windowcache  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


class TestReadingAndWriting(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_an_unknown_pair_reads_as_nothing(self):
        self.assertEqual(windowcache.get("http://x/v1", "m"), (0, False))

    def test_what_was_written_comes_back_fresh(self):
        windowcache.put("http://x/v1", "m", 524288)
        self.assertEqual(windowcache.get("http://x/v1", "m"), (524288, True))

    def test_the_same_model_on_a_DIFFERENT_endpoint_is_a_different_entry(self):
        """Two servers can serve one model name with different windows. Keying by the name
        alone lets one deployment answer for another."""
        windowcache.put("http://a/v1", "m", 100_000)
        windowcache.put("http://b/v1", "m", 200_000)
        self.assertEqual(windowcache.get("http://a/v1", "m")[0], 100_000)
        self.assertEqual(windowcache.get("http://b/v1", "m")[0], 200_000)

    def test_a_stale_entry_is_STILL_RETURNED_but_marked_stale(self):
        """A stale window beats no window: without it the guard falls to the ceiling table
        and sleeps. The caller needs the flag to know a refresh is worth attempting."""
        windowcache.put("http://x/v1", "m", 524288, ttl=-1)
        window, fresh = windowcache.get("http://x/v1", "m")
        self.assertEqual(window, 524288)
        self.assertFalse(fresh)


class TestItNeverRaises(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_a_corrupt_file_reads_as_empty(self):
        with open(os.path.join(windowcache.state_dir(), windowcache.FILENAME), "w") as fh:
            fh.write("{not json")
        self.assertEqual(windowcache.get("http://x/v1", "m"), (0, False))

    def test_a_corrupt_file_is_REPLACED_by_the_next_write(self):
        path = os.path.join(windowcache.state_dir(), windowcache.FILENAME)
        with open(path, "w") as fh:
            fh.write("{not json")
        windowcache.put("http://x/v1", "m", 1000)
        self.assertEqual(windowcache.get("http://x/v1", "m"), (1000, True))

    def test_a_window_of_zero_or_less_is_NOT_stored(self):
        """Storing 0 would make the cache assert 'this endpoint says unknown' — and the next
        step of the cascade would never be consulted. Verify by reading the cache file."""
        windowcache.put("http://x/v1", "m", 0)
        windowcache.put("http://x/v1", "n", -5)
        # Verify they were NOT stored by checking the file directly
        cache_path = os.path.join(windowcache.state_dir(), windowcache.FILENAME)
        if os.path.exists(cache_path):
            with open(cache_path) as fh:
                data = json.load(fh)
            self.assertNotIn("http://x/v1|m", data)
            self.assertNotIn("http://x/v1|n", data)
        # Also verify get returns (0, False) for both
        self.assertEqual(windowcache.get("http://x/v1", "m"), (0, False))
        self.assertEqual(windowcache.get("http://x/v1", "n"), (0, False))

    def test_a_zero_does_not_DESTROY_a_window_we_already_knew(self):
        """The guard's real job. A probe that learns nothing calls nobody's put — but if it
        ever did, overwriting a measured window with zero would send the cascade back to a
        ceiling it had already improved on."""
        windowcache.put("http://x/v1", "m", 524288)
        windowcache.put("http://x/v1", "m", 0)
        self.assertEqual(windowcache.get("http://x/v1", "m"), (524288, True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
