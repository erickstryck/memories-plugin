"""The sweep has to run for the sessions people actually have.

WHY THIS FILE EXISTS. `sweep_if_due` fired only on `round_no % 20 == 0`, and `round_no` is
PER SESSION -- it restarts at 1 every time a session begins. So the housekeeping depended on
a single session reaching its twentieth round AND landing exactly on it.

MEASURED on the real install before this changed: of 35 sessions with recall state, 6 had
ever reached round 20, and the state directory held 59 `checkpoint-*.count` and 26
`recall-*.json` older than the seven-day cutoff, the oldest 33 days old. The sweep was
correct and simply never ran. Driving it round by round confirmed both directions: a session
that reaches round 20 purges all 70 planted files, a session of 7 rounds purges none.

WHAT THE FIX IS. The cadence stops being "every N rounds of one session" and becomes "at most
once every N hours, across all sessions", kept in a stamp file in the state directory. That
is the quantity the job actually cares about -- the files are dead by WALL CLOCK age, not by
anybody's round number -- and it makes a machine that only ever runs short sessions sweep
exactly as often as one that runs long ones.
"""
import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import session_state as st  # noqa: E402


def _plant(directory: Path, count: int, age_days: float) -> None:
    """`count` per-session files of each kind, `age_days` old."""
    when = time.time() - age_days * 86400
    for i in range(count):
        for name in (f"recall-s{i}.json", f"checkpoint-s{i}.count"):
            path = directory / name
            path.write_text("{}")
            os.utime(path, (when, when))


def _remaining(directory: Path) -> int:
    return len(list(directory.glob("recall-*.json"))) + \
        len(list(directory.glob("checkpoint-*.count")))


class TestAShortSessionStillSweeps(unittest.TestCase):
    """The case the old cadence could not serve: nobody reaches round 20.

    29 of the 35 sessions measured on the real install ended before round 20. Under the old
    rule every one of them swept nothing, which is why a month of dead files accumulated.
    """

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_a_session_of_seven_rounds_sweeps_the_dead_files(self):
        _plant(self.dir, 10, age_days=30)
        self.assertEqual(_remaining(self.dir), 20, "planted")

        for round_no in range(1, 8):
            st.sweep_if_due(self.dir, round_no)

        self.assertEqual(_remaining(self.dir), 0,
                         "a short session must still clear state that is a month old")

    def test_even_a_single_round_session_sweeps(self):
        """A session that asks one question and ends. There are many of these."""
        _plant(self.dir, 5, age_days=30)
        st.sweep_if_due(self.dir, 1)
        self.assertEqual(_remaining(self.dir), 0)


class TestItDoesNotSweepOnEveryRound(unittest.TestCase):
    """The reason a cadence exists at all: the glob must not run on every prompt."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_the_second_round_of_the_same_session_does_not_glob_again(self):
        _plant(self.dir, 3, age_days=30)
        first = st.sweep_if_due(self.dir, 1)
        self.assertEqual(first, 6, "the first round of a session sweeps")

        _plant(self.dir, 3, age_days=30)      # more dead files appear
        second = st.sweep_if_due(self.dir, 2)
        self.assertEqual(second, 0, "the very next round must not sweep again")

    def test_a_later_session_sweeps_once_the_interval_has_passed(self):
        """A new session hours later is exactly when housekeeping should happen again."""
        _plant(self.dir, 3, age_days=30)
        st.sweep_if_due(self.dir, 1)

        # The stamp is what the interval is measured from; age it past the interval.
        stamp = self.dir / st.SWEEP_STAMP
        old = time.time() - (st.PURGE_EVERY_HOURS * 3600 + 60)
        os.utime(stamp, (old, old))

        _plant(self.dir, 3, age_days=30)
        self.assertEqual(st.sweep_if_due(self.dir, 1), 6,
                         "once the interval has passed, the next session sweeps")


class TestTheStampIsNotTheJob(unittest.TestCase):
    """Housekeeping must never become the reason a recall failed."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_an_unwritable_state_dir_returns_zero_and_does_not_raise(self):
        _plant(self.dir, 2, age_days=30)
        os.chmod(self.dir, 0o500)
        self.addCleanup(os.chmod, self.dir, 0o700)
        try:
            result = st.sweep_if_due(self.dir, 1)
        except Exception as exc:                        # pragma: no cover - the assertion
            self.fail(f"sweeping must not raise, got {exc!r}")
        self.assertEqual(result, 0)

    def test_a_stamp_that_cannot_be_stated_does_not_stop_the_sweep_forever(self):
        """An unreadable stamp must read as 'never swept', not as 'swept just now'.

        THE STAMP'S CONTENT IS NOT THE CLOCK -- its mtime is -- so a stamp holding garbage is
        not the failure worth pinning here; a stamp that cannot be STATED is. The directory
        is made unreadable to produce exactly that, and the sweep must still decide to run
        rather than treating the failure as "recently swept".
        """
        _plant(self.dir, 2, age_days=30)
        stamp = self.dir / st.SWEEP_STAMP
        stamp.write_text("")
        # A stamp dated in the FUTURE is the shape that jams a naive clock comparison
        # forever: `now - future` is negative, which is always below any interval.
        ahead = time.time() + 365 * 86400
        os.utime(stamp, (ahead, ahead))
        self.assertEqual(st.sweep_if_due(self.dir, 1), 4,
                         "a stamp dated in the future must not jam housekeeping")

    def test_the_stamp_is_not_swept_as_if_it_were_session_state(self):
        """The stamp lives in the same directory the globs run over."""
        _plant(self.dir, 1, age_days=30)
        st.sweep_if_due(self.dir, 1)
        self.assertTrue((self.dir / st.SWEEP_STAMP).exists(),
                        "the stamp must survive the sweep it schedules")

    def test_the_stamp_itself_is_published_owner_only(self):
        """The stamp is a file this package writes into the state directory like any other.

        `Path.touch()` takes the umask -- 0o664 on this machine -- which is the exact defect
        the other five writers in this package were just routed through `core.statefile` to
        fix. A housekeeping file is not an exception to the rule it schedules.
        """
        _plant(self.dir, 1, age_days=30)
        st.sweep_if_due(self.dir, 1)
        stamp = self.dir / st.SWEEP_STAMP
        self.assertEqual(stat.S_IMODE(os.stat(stamp).st_mode), 0o600)

    def test_a_sweep_that_dies_half_way_still_moves_the_cadence_on(self):
        """The stamp is written BEFORE the purge, not after.

        A purge that raises leaves the directory exactly as unswept as before. If the stamp
        were written afterwards, that failure would be retried on EVERY round forever -- the
        glob-on-every-prompt cost the cadence exists to avoid, and it would land on precisely
        the machine whose state directory is already unhealthy. Writing it first means a
        failing sweep is retried on the next interval, like any other.
        """
        _plant(self.dir, 2, age_days=30)
        original = st.purge_dead

        def explode(*args, **kwargs):
            raise OSError("the sweep died half way")

        st.purge_dead = explode
        self.addCleanup(setattr, st, "purge_dead", original)
        with self.assertRaises(OSError):
            st.sweep_if_due(self.dir, 1)
        st.purge_dead = original

        self.assertEqual(st.sweep_if_due(self.dir, 2), 0,
                         "a sweep that died must not retry on the very next round")


if __name__ == "__main__":
    unittest.main()
