"""Session state and cadence, in core because both hosts keep the same kind of state.

The claude-code hook counts turns in a file because the host does not tell it; hermes
passes turn_number to on_turn_start. Same decision, two sources — so the DECISION lives
here and each adapter supplies the number.
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import session_state as st


class TestLoadAndSave(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "recall-s1.json"

    def test_a_missing_file_is_a_fresh_session_not_an_error(self):
        self.assertEqual(st.load(self.path), {"round": 0, "seen": {}})

    def test_corrupted_content_is_a_fresh_session_not_an_error(self):
        self.path.write_text("{not json")
        self.assertEqual(st.load(self.path), {"round": 0, "seen": {}})

    def test_a_round_trip_preserves_the_state(self):
        st.save(self.path, {"round": 3, "seen": {"a": 2}})
        self.assertEqual(st.load(self.path), {"round": 3, "seen": {"a": 2}})

    def test_saving_to_None_is_a_no_op_not_a_crash(self):
        st.save(None, {"round": 1, "seen": {}})

    def test_an_unwritable_path_does_not_raise(self):
        st.save(Path("/proc/impossible/state.json"), {"round": 1, "seen": {}})


class TestPrune(unittest.TestCase):
    def test_it_drops_what_can_no_longer_change_a_decision(self):
        state = {"round": 20, "seen": {"recent": 19, "at_the_edge": 12, "old": 5}}
        self.assertEqual(st.prune(state), 2)
        self.assertEqual(set(state["seen"]), {"recent"})

    def test_it_keeps_what_still_prevents_a_reinjection(self):
        state = {"round": 10, "seen": {"a": 9, "b": 4}}
        st.prune(state)
        self.assertEqual(set(state["seen"]), {"a", "b"})

    def test_a_corrupted_value_is_discarded(self):
        state = {"round": 5, "seen": {"ok": 4, "junk": "not a number", "null": None}}
        st.prune(state)
        self.assertEqual(set(state["seen"]), {"ok"})

    def test_empty_state_does_not_break(self):
        self.assertEqual(st.prune({}), 0)


class TestPurgeDead(unittest.TestCase):
    def test_it_removes_the_old_and_keeps_the_recent(self):
        d = Path(tempfile.mkdtemp())
        old, new = d / "recall-dead.json", d / "recall-alive.json"
        for f in (old, new):
            f.write_text(json.dumps({"round": 1, "seen": {}}))
        stamp = time.time() - 10 * 86400
        os.utime(old, (stamp, stamp))
        self.assertEqual(st.purge_dead(d, days=7.0), 1)
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())

    def test_it_does_not_touch_a_log(self):
        d = Path(tempfile.mkdtemp())
        log = d / "recall.log"
        log.write_text("a line")
        stamp = time.time() - 30 * 86400
        os.utime(log, (stamp, stamp))
        st.purge_dead(d, days=1.0)
        self.assertTrue(log.exists(), "the log is not session state")

    def test_a_missing_directory_does_not_raise(self):
        self.assertEqual(st.purge_dead(Path("/proc/impossible"), days=1.0), 0)


class TestCadence(unittest.TestCase):
    def test_it_is_due_on_the_multiple_and_only_there(self):
        self.assertEqual([t for t in range(1, 13) if st.due(t, 5)], [5, 10])

    def test_an_interval_of_zero_or_less_never_fires(self):
        for interval in (0, -1):
            self.assertEqual([t for t in range(1, 20) if st.due(t, interval)], [])

    def test_an_interval_of_one_fires_every_turn(self):
        self.assertEqual([t for t in range(1, 5) if st.due(t, 1)], [1, 2, 3, 4])


class TestNextRound(unittest.TestCase):
    def test_it_increments_and_returns(self):
        state = {"round": 4, "seen": {}}
        self.assertEqual(st.next_round(state), 5)
        self.assertEqual(state["round"], 5)

    def test_a_corrupted_round_restarts_from_one(self):
        state = {"round": "wat", "seen": {}}
        self.assertEqual(st.next_round(state), 1)
