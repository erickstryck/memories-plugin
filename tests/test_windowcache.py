import json
import os
import sys
import tempfile
import time
import unittest
import unittest.mock

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

    def test_put_does_not_RAISE_when_the_state_dir_cannot_be_created(self):
        """`get` already degrades to a miss here because its path call sits inside its own
        try. `put` must degrade the same way: the module promises a cache that cannot be
        written is a miss, and two entry points failing differently under one condition is
        how that promise gets trusted for behaviour it does not have."""
        blocked = os.path.join(tempfile.mkdtemp(), "state")
        with open(blocked, "w") as fh:
            fh.write("a file, not a directory")
        os.environ["QCTX_STATE_DIR"] = blocked
        self.assertEqual(windowcache.get("http://x/v1", "m"), (0, False))
        windowcache.put("http://x/v1", "m", 524288)     # must not raise


class TestTheTempFileIsNamedPerProcess(unittest.TestCase):
    """Two hermes sessions refreshing at once both used to open a FIXED `.tmp` name; after
    the first `os.replace`, the loser's still-open descriptor wrote into the inode the first
    one just published. Readers never saw a torn file — `os.replace` is atomic and `_load`
    catches `ValueError` — so it already degraded to a cache miss and self-healed. Naming the
    temp file per-process removes the race outright instead of merely tolerating it."""

    def setUp(self):
        a_state_dir()

    def test_the_temp_file_used_during_put_carries_this_process_id(self):
        captured = {}
        real_replace = os.replace

        def spy_replace(src, dst):
            captured["src"] = src

            return real_replace(src, dst)

        with unittest.mock.patch("os.replace", spy_replace):
            windowcache.put("http://x/v1", "m", 12345)
        self.assertIn(f".{os.getpid()}.tmp", captured.get("src", ""),
                     "the temp file name does not carry this process's pid")


class TestAge(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_nothing_cached_has_no_age(self):
        self.assertIsNone(windowcache.age_seconds("http://x/v1", "m"))

    def test_a_fresh_write_is_close_to_zero_seconds_old(self):
        windowcache.put("http://x/v1", "m", 524288)
        age = windowcache.age_seconds("http://x/v1", "m")
        self.assertIsNotNone(age)
        self.assertLess(age, 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
