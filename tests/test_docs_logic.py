"""Tests for the pure logic in `core.docs` — TTL, document id and staleness detection.
No network.

The mtime tolerance test is a REGRESSION test for a real bug: the comparison was exact
float equality and the JSON round trip lost the last bits, so the "file changed" warning
fired on every search of every document and `refresh` reindexed the whole archive on
each run.
"""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.docs import GONE, DocsError, doc_id_for, parse_ttl, source_changed


class TestTTL(unittest.TestCase):
    def test_units(self):
        self.assertEqual(parse_ttl("30s"), 30)
        self.assertEqual(parse_ttl("30m"), 1800)
        self.assertEqual(parse_ttl("2h"), 7200)
        self.assertEqual(parse_ttl("7d"), 604800)

    def test_bare_seconds(self):
        self.assertEqual(parse_ttl("90"), 90)

    def test_accepts_spaces_and_uppercase(self):
        self.assertEqual(parse_ttl(" 24H "), 86400)

    def test_invalid_raises(self):
        for bad in ("", "abc", "24x", "-5h"):
            with self.assertRaises(DocsError):
                parse_ttl(bad)


class TestDocId(unittest.TestCase):
    def test_stable_for_the_same_path(self):
        self.assertEqual(doc_id_for("/tmp/a.md"), doc_id_for("/tmp/a.md"))

    def test_relative_and_absolute_paths_give_the_same_id(self):
        cwd = os.getcwd()
        try:
            os.chdir("/tmp")
            self.assertEqual(doc_id_for("a.md"), doc_id_for("/tmp/a.md"),
                             "the id comes from the ABSOLUTE path, so reindexing replaces")
        finally:
            os.chdir(cwd)

    def test_different_paths_give_different_ids(self):
        self.assertNotEqual(doc_id_for("/tmp/a.md"), doc_id_for("/tmp/b.md"))


class TestStaleness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.file_path = Path(self.tmp.name) / "doc.md"
        self.file_path.write_text("original content\n")
        st = os.stat(self.file_path)
        self.mtime = st.st_mtime
        self.size = st.st_size

    def tearDown(self):
        self.tmp.cleanup()

    def test_untouched_file_is_not_stale(self):
        self.assertIsNone(source_changed(str(self.file_path), self.mtime, self.size))

    def test_float_round_trip_does_not_produce_a_false_positive(self):
        """The original bug: JSON returned ...9956775 for a stored ...9956777."""
        nearly = self.mtime + 2.4e-07
        self.assertIsNone(source_changed(str(self.file_path), nearly, self.size),
                          "a 1e-7 difference is serialization noise, not an edit")

    def test_mtime_rounded_to_the_millisecond_produces_no_false_positive(self):
        self.assertIsNone(source_changed(str(self.file_path), round(self.mtime, 3), self.size))

    def test_size_change_is_detected(self):
        self.file_path.write_text("original content with more text\n")
        reason = source_changed(str(self.file_path), self.mtime, self.size)
        self.assertIsNotNone(reason)
        self.assertIn("size", reason)

    def test_mtime_change_at_the_same_size_is_detected(self):
        future = self.mtime + 60
        os.utime(self.file_path, (future, future))
        reason = source_changed(str(self.file_path), self.mtime, self.size)
        self.assertIsNotNone(reason, "an edit that preserves the size still moves the mtime")

    def test_removed_file_is_reported(self):
        path = str(self.file_path)
        os.unlink(path)
        self.assertEqual(source_changed(path, self.mtime, self.size),
                         GONE)

    def test_an_mtime_preserving_same_size_edit_is_still_detected(self):
        """What the digest buys, and what metadata alone could never see.

        `cp -p`, `rsync --times`, `touch -r` and any tar or backup restore preserve the
        mtime; an edit that swaps one character preserves the size. Together they make a
        genuinely changed file report itself unchanged — and a stale chunk that says it is
        current is worse than one marked stale, because nothing prompts a re-read.
        """
        from core.docs import content_digest
        digest = content_digest(self.file_path.read_text())
        st = os.stat(self.file_path)

        self.file_path.write_text("original contXnt\n")   # same length, one character
        os.utime(self.file_path, (st.st_mtime, st.st_mtime))
        fresh = os.stat(self.file_path)
        self.assertEqual(fresh.st_size, self.size, "the probe must preserve the size")
        self.assertEqual(fresh.st_mtime, self.mtime, "the probe must preserve the mtime")

        self.assertIsNone(source_changed(str(self.file_path), self.mtime, self.size),
                          "metadata alone cannot see this, which is the point")
        self.assertIsNotNone(
            source_changed(str(self.file_path), self.mtime, self.size, digest),
            "the digest has to catch what the metadata cannot")

    def test_the_digest_agrees_when_nothing_changed(self):
        from core.docs import content_digest
        digest = content_digest(self.file_path.read_text())
        self.assertIsNone(source_changed(str(self.file_path), self.mtime, self.size, digest))

    def test_documents_indexed_before_the_digest_existed_still_work(self):
        """Backward compatibility: an archive written by the previous version carries no
        `src_digest`, and must keep falling back to the metadata comparison rather than
        being reported as changed on every search."""
        self.assertIsNone(source_changed(str(self.file_path), self.mtime, self.size, None))

    def test_missing_metadata_does_not_break(self):
        self.assertIsNone(source_changed(str(self.file_path), None, None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
