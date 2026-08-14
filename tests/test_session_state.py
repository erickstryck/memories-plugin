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

    def test_a_non_dict_seen_is_replaced_rather_than_carried_forward(self):
        """The HEALING half of the pair `core.blocks.split_by_budget` keeps the other half of.

        `split_by_budget` refuses to crash on a non-dict `seen`, but it substitutes a LOCAL
        dict on purpose — "the caller owns persistence" — so nothing there fixes the file. If
        `load` handed the corruption back, the next `save` would write it out again and every
        round from then on would lose dedup: every recalled memory reinjected in full every
        turn. It lives here because it is the one function both hosts read state through; it
        used to live in `hooks/recall.py`, which is why the hermes adapter never had it.
        """
        for corrupt in ("corrupted-not-a-dict", ["hit-1"], 7, None):
            self.path.write_text(json.dumps({"round": 3, "seen": corrupt}))
            state = st.load(self.path)
            self.assertEqual(state, {"round": 3, "seen": {}},
                             f"{corrupt!r} survived the load")

    def test_healing_seen_does_not_discard_the_round_or_anything_else(self):
        """The corruption is worth exactly its own key: a session that has run 40 rounds must
        not be restarted, and a key some later version added must not be dropped."""
        self.path.write_text(json.dumps({"round": 40, "seen": "corrupted", "extra": "keep"}))
        self.assertEqual(st.load(self.path),
                         {"round": 40, "seen": {}, "extra": "keep"})

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

    def test_a_corrupted_round_does_not_raise(self):
        state = {"round": "abc", "seen": {"a": 1}}
        self.assertEqual(st.prune(state), 0)

    def test_a_non_dict_seen_does_not_raise(self):
        for bad_seen in ("not a dict", ["a", "b"]):
            state = {"round": 5, "seen": bad_seen}
            self.assertEqual(st.prune(state), 0)

    def test_a_corrupted_round_container_does_not_raise(self):
        state = {"round": [1, 2], "seen": {"a": 1}}
        self.assertEqual(st.prune(state), 0)

    def test_the_corrupted_state_from_a_bad_write_does_not_stop_pruning(self):
        """Pins the end-to-end consequence: a state file saved with a wrong-typed `seen`
        must not make `prune` raise and take the whole hook down with it."""
        state = {"round": 3, "seen": "corrupted"}
        self.assertEqual(st.prune(state), 0)


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


class TestSweepIfDue(unittest.TestCase):
    """The sweep CADENCE, in core because both hosts have to run the same one.

    While `round_no % 20` was inline in `hooks/recall.py`, `purge_dead` had "moved into core
    so both hosts share it" and the hermes adapter called it from nowhere — 60 measured
    prefetch rounds left a 30-day-old abandoned file exactly where it was.
    """

    def _dir(self):
        d = Path(tempfile.mkdtemp())
        dead = d / "recall-dead.json"
        dead.write_text(json.dumps({"round": 1, "seen": {}}))
        stamp = time.time() - 30 * 86400
        os.utime(dead, (stamp, stamp))

        return d, dead

    def test_it_sweeps_on_the_cadence(self):
        d, dead = self._dir()
        self.assertEqual(st.sweep_if_due(d, st.PURGE_EVERY_ROUNDS), 1)
        self.assertFalse(dead.exists())

    def test_it_does_nothing_off_the_cadence(self):
        d, dead = self._dir()
        for round_no in range(1, st.PURGE_EVERY_ROUNDS):
            self.assertEqual(st.sweep_if_due(d, round_no), 0)
        self.assertTrue(dead.exists(), "a glob on every prompt is what the cadence avoids")

    def test_round_zero_is_never_due(self):
        """Round 0 means no round has happened; `0 % anything == 0` would make it due."""
        d, dead = self._dir()
        self.assertEqual(st.sweep_if_due(d, 0), 0)
        self.assertTrue(dead.exists())

    def test_a_corrupted_round_or_cadence_is_not_due_rather_than_an_error(self):
        d, _ = self._dir()
        for bad in ("abc", None, [1, 2]):
            self.assertEqual(st.sweep_if_due(d, bad), 0)
        self.assertEqual(st.sweep_if_due(d, st.PURGE_EVERY_ROUNDS, every=0), 0)

    def test_an_impossible_directory_does_not_raise(self):
        self.assertEqual(st.sweep_if_due(Path("/proc/impossible"),
                                        st.PURGE_EVERY_ROUNDS), 0)

    def test_a_corrupted_RETENTION_is_not_due_rather_than_an_error_either(self):
        """The tolerance stopped one argument short of the module's own contract.

        `sweep_if_due` coerced `round_no` and `every` and forwarded `days` untouched, and
        `purge_dead` computed `cutoff = time.time() - days * 86400` OUTSIDE its `try` — so
        `days=None` raised TypeError from the one module whose docstring says nothing here
        raises, one line above a guard that would have caught it. No host passes a bad value
        today; the contract is what every caller in this package relies on instead of checking
        who calls it, so it has to hold without them.
        """
        d, dead = self._dir()
        for bad in (None, "x", [7], {}):
            with self.subTest(days=bad):
                self.assertEqual(st.sweep_if_due(d, st.PURGE_EVERY_ROUNDS, days=bad), 0)
                self.assertEqual(st.purge_dead(d, days=bad), 0)
        self.assertTrue(dead.exists(), "a housekeeping sweep it could not size ran anyway")
        # And a NUMERIC STRING still works, the same way `next_round` and `due` accept one.
        self.assertEqual(st.sweep_if_due(d, st.PURGE_EVERY_ROUNDS, days="7"), 1)
        self.assertFalse(dead.exists())


class TestCadence(unittest.TestCase):
    def test_it_is_due_on_the_multiple_and_only_there(self):
        self.assertEqual([t for t in range(1, 13) if st.due(t, 5)], [5, 10])

    def test_an_interval_of_zero_or_less_never_fires(self):
        for interval in (0, -1):
            self.assertEqual([t for t in range(1, 20) if st.due(t, interval)], [])

    def test_an_interval_of_one_fires_every_turn(self):
        self.assertEqual([t for t in range(1, 5) if st.due(t, 1)], [1, 2, 3, 4])

    def test_a_numeric_string_is_coerced_not_rejected(self):
        self.assertTrue(st.due("5", 5))
        self.assertTrue(st.due(5, "5"))

    def test_a_non_numeric_turn_or_interval_is_never_due_not_an_error(self):
        self.assertFalse(st.due("abc", 5))
        self.assertFalse(st.due(5, "abc"))
        self.assertFalse(st.due(None, 5))
        self.assertFalse(st.due(5, None))
        self.assertFalse(st.due([1, 2], 5))


class TestNextRound(unittest.TestCase):
    def test_it_increments_and_returns(self):
        state = {"round": 4, "seen": {}}
        self.assertEqual(st.next_round(state), 5)
        self.assertEqual(state["round"], 5)

    def test_a_corrupted_round_restarts_from_one(self):
        state = {"round": "wat", "seen": {}}
        self.assertEqual(st.next_round(state), 1)


if __name__ == "__main__":
    # See the note in tests/test_blocks.py: run directly, this file used to report nothing
    # and exit 0.
    unittest.main()
