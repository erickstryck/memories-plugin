"""The context window a host reported, recorded per session for the guard to read.

Neither guard can ask its host for the window: the claude-code hook payload does not carry
it, and the hermes guard is a subprocess that does not import hermes. So the side of each
host that DOES know it publishes it here, one file per session, and the guard only reads.
One module owns the record's location and shape, so the writer and the reader cannot drift.
"""
import json
import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import hostwindow, session_state  # noqa: E402


class StateDirCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self._old = os.environ.get("QCTX_STATE_DIR")
        os.environ["QCTX_STATE_DIR"] = self.dir
        self.addCleanup(self._restore)

    def _restore(self):
        if self._old is None:
            os.environ.pop("QCTX_STATE_DIR", None)
        else:
            os.environ["QCTX_STATE_DIR"] = self._old

    def files(self) -> list:
        return sorted(p.name for p in Path(self.dir).iterdir())


class TestARecordRoundTrips(StateDirCase):
    def test_what_was_published_is_what_is_read(self):
        self.assertTrue(hostwindow.publish("s1", "claude-opus-5-5[1m]", 1_000_000,
                                           "claude-code"))
        got = hostwindow.read("s1")
        self.assertEqual((got.model, got.window, got.source, got.guess),
                         ("claude-opus-5-5[1m]", 1_000_000, "claude-code", False))
        self.assertAlmostEqual(got.at, time.time(), delta=5)

    def test_a_guess_is_recorded_as_one(self):
        hostwindow.publish("s1", "mystery", 256_000, "hermes", guess=True)
        self.assertTrue(hostwindow.read("s1").guess)

    def test_the_latest_publication_wins(self):
        hostwindow.publish("s1", "claude-opus-5-5[1m]", 1_000_000, "claude-code")
        hostwindow.publish("s1", "claude-haiku-4-5-20251001", 200_000, "claude-code")
        self.assertEqual(hostwindow.read("s1").window, 200_000)

    def test_another_session_has_its_own_record(self):
        hostwindow.publish("s1", "m", 1_000_000, "claude-code")
        self.assertIsNone(hostwindow.read("s2"))

    def test_the_file_is_private(self):
        hostwindow.publish("s1", "m", 1_000_000, "claude-code")
        (name,) = self.files()
        mode = stat.S_IMODE(os.stat(os.path.join(self.dir, name)).st_mode)
        self.assertEqual(mode, 0o600, oct(mode))

    def test_the_file_matches_the_pattern_the_sweep_uses(self):
        hostwindow.publish("s1", "m", 1_000_000, "claude-code")
        self.assertEqual([p.name for p in Path(self.dir).glob(hostwindow.PATTERN)],
                         self.files())


class TestNothingInvalidIsRecorded(StateDirCase):
    def test_no_session_no_record(self):
        for session in ("", None):
            with self.subTest(session=session):
                self.assertFalse(hostwindow.publish(session, "m", 1_000_000, "claude-code"))
        self.assertEqual(self.files(), [])

    def test_a_window_that_is_not_a_positive_whole_number_is_not_recorded(self):
        for window in (0, -1, "1M", None, 2.5, True):
            with self.subTest(window=window):
                self.assertFalse(hostwindow.publish("s1", "m", window, "claude-code"))
        self.assertEqual(self.files(), [])

    def test_a_session_id_cannot_escape_the_state_dir(self):
        hostwindow.publish("../../outside/x", "m", 1_000_000, "claude-code")
        (name,) = self.files()
        self.assertTrue(name.startswith("window-"), name)
        self.assertEqual(hostwindow.read("../../outside/x").window, 1_000_000)

    def test_an_unwritable_state_dir_is_a_false_not_a_raise(self):
        os.environ["QCTX_STATE_DIR"] = os.path.join(self.dir, "file-in-the-way")
        Path(os.environ["QCTX_STATE_DIR"]).write_text("not a directory")
        self.assertFalse(hostwindow.publish("s1", "m", 1_000_000, "claude-code"))


class TestAnUnreadableRecordIsNoRecord(StateDirCase):
    def _write(self, text: str) -> None:
        hostwindow.publish("s1", "m", 1_000_000, "claude-code")
        (name,) = self.files()
        Path(self.dir, name).write_text(text)

    def test_corrupt_json(self):
        self._write("{not json")
        self.assertIsNone(hostwindow.read("s1"))

    def test_wrong_types_inside(self):
        for record in ({"model": "m", "window": "x", "source": "s", "guess": False, "at": 1},
                       {"model": "m", "window": 0, "source": "s", "guess": False, "at": 1},
                       {"model": "m", "source": "s"},
                       [1, 2, 3]):
            with self.subTest(record=record):
                self._write(json.dumps(record))
                self.assertIsNone(hostwindow.read("s1"))

    def test_no_session_reads_nothing(self):
        self.assertIsNone(hostwindow.read(""))


class TestTheNewestRecordOfASource(StateDirCase):
    def test_it_picks_the_most_recent_of_that_source(self):
        hostwindow.publish("a", "m-old", 200_000, "claude-code")
        hostwindow.publish("h", "m-hermes", 1_000_000, "hermes")
        time.sleep(0.01)
        hostwindow.publish("b", "m-new", 1_000_000, "claude-code")
        session, record = hostwindow.newest("claude-code")
        self.assertEqual((session, record.model), ("b", "m-new"))
        self.assertEqual(hostwindow.newest("hermes")[1].model, "m-hermes")

    def test_none_when_that_source_never_published(self):
        hostwindow.publish("a", "m", 200_000, "claude-code")
        self.assertIsNone(hostwindow.newest("hermes"))


class TestTheSweepRemovesDeadRecords(StateDirCase):
    def test_an_old_record_is_swept_with_the_other_session_files(self):
        hostwindow.publish("old", "m", 1_000_000, "claude-code")
        hostwindow.publish("new", "m", 1_000_000, "claude-code")
        old = next(Path(self.dir).glob("window-old*"))
        week_ago = time.time() - 8 * 86400
        os.utime(old, (week_ago, week_ago))
        removed = session_state.purge_dead(self.dir, days=7)
        self.assertEqual(removed, 1)
        self.assertIsNone(hostwindow.read("old"))
        self.assertIsNotNone(hostwindow.read("new"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
