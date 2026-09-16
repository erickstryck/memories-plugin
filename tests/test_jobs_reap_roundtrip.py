"""`reap` compares a start time that has been through JSON. The types must survive it.

`lease.process_start` answers a STRING (field 22 of `/proc/<pid>/stat`, unparsed). A job
records that value and a later cycle compares the two with `==`. JSON preserves the
difference between `"5306546"` and `5306546`, and `==` does not bridge it — so a comparison
that looks obviously correct reads FALSE for a daemon that is alive and working, and the
reaper then marks its job interrupted underneath it.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import jobs, lease  # noqa: E402


class TestTheStartTimeSurvivesTheRoundTrip(unittest.TestCase):
    """A live daemon's job must not be reaped because a number was stored as a number.

    WHY THIS IS NOT HYPOTHETICAL ARITHMETIC. `reap`'s own docstring says it compares the pair
    `(pid, starttime)` "which is the test `lease.alive` already applies". `lease.alive`
    compares `process_start(pid) == recorded`, and every writer today happens to store the
    str — but the value is read back out of JSON, where an int stays an int, and nothing in
    `update()` or `_write` coerces it. A single caller storing `int(start)`, or a host whose
    `process_start` returns a number, silently turns every reap into a false positive.

    The cost is the exact state `core/jobs.py`'s module docstring calls worse than absence:
    `status` reporting a running job as "interrupted: the daemon running this job is gone"
    while that daemon is mid-file.
    """

    def setUp(self):
        os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()
        self.pid = os.getpid()
        self.start = lease.process_start(self.pid)
        self.assertIsNotNone(self.start, "this platform cannot report a start time")

    def _a_running_job(self, recorded) -> None:
        jobs.enqueue("r1", "index", ["/f"])
        jobs.update("r1", state=jobs.RUNNING, daemon_pid=self.pid, daemon_start=recorded)

    def _reap(self) -> None:
        jobs.reap(lambda pid: lease.process_start(pid) is not None, lease.process_start)

    def test_a_live_daemon_keeps_its_job_when_the_start_time_was_stored_as_TEXT(self):
        self._a_running_job(self.start)
        self._reap()

        self.assertEqual(jobs.all_jobs()[0]["state"], jobs.RUNNING)

    def test_a_live_daemon_keeps_its_job_when_the_start_time_was_stored_as_a_NUMBER(self):
        """The same fact, written the other way. `"5306546"` and `5306546` name one instant."""
        self._a_running_job(int(self.start))
        self._reap()

        self.assertEqual(jobs.all_jobs()[0]["state"], jobs.RUNNING,
                         "a live daemon's job was reaped because its start time was an int")

    def test_a_job_whose_daemon_really_is_gone_is_STILL_reaped(self):
        """The guard must not become 'never reap'.

        A start time that does not match is a different process wearing a recycled pid, and
        that job IS stale — this is the case `reap` exists for.
        """
        self._a_running_job("1")                # not this process's start time
        self._reap()

        self.assertEqual(jobs.all_jobs()[0]["state"], jobs.FAILED)


if __name__ == "__main__":
    unittest.main()
