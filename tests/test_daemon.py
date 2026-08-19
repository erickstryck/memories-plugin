"""The loop: runs what is queued, watches, and ends when nobody is using it anymore.

NO TEST HERE STARTS A REAL DAEMON, nor touches Qdrant. `run` takes the work executor and a
cycle count, so the whole loop is exercised in-process — the same choice `refresh_window
(probe=...)` already makes.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import daemon, jobs, lease  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


def a_live_lease() -> dict:
    return lease.write("s1", "claude", pid=os.getpid())


class TestItEndsWithTheLastHost(unittest.TestCase):
    """The user's requirement, verbatim: "the daemon must be killed when claude or hermes
    exits/dies"."""

    def setUp(self):
        a_state_dir()

    def test_with_NO_live_lease_it_exits_on_the_first_cycle(self):
        self.assertEqual(daemon.run(lambda job: None, cycles=10, sleep=lambda s: None),
                         "no live lease")

    def test_with_a_live_lease_it_keeps_going(self):
        a_live_lease()
        self.assertEqual(daemon.run(lambda job: None, cycles=3, sleep=lambda s: None),
                         "cycles exhausted")

    def test_it_exits_when_the_lease_DIES_mid_run(self):
        """The real case: the host closes while the daemon is running."""
        a_live_lease()
        seen = []

        def kill_the_lease_after_one(seconds):
            seen.append(1)
            if len(seen) == 1:
                for path in lease.dir().glob("*.json"):
                    path.unlink()

        self.assertEqual(daemon.run(lambda job: None, cycles=10,
                                    sleep=kill_the_lease_after_one), "no live lease")
        self.assertEqual(len(seen), 1, "did not stop on the cycle right after the lease died")


class TestItRunsWhatIsQueued(unittest.TestCase):
    def setUp(self):
        a_state_dir()
        a_live_lease()

    def test_a_pending_job_is_handed_to_the_worker(self):
        jobs.enqueue("alpha", "index", ["/a.py"])
        seen = []
        daemon.run(lambda job: seen.append(job["repo"]), cycles=1, sleep=lambda s: None)
        self.assertEqual(seen, ["alpha"])

    def test_the_job_is_marked_RUNNING_with_the_daemon_pid_while_it_runs(self):
        """Without the pid, `reap` would have no way to tell whether the job's owner still
        exists."""
        jobs.enqueue("alpha", "index", ["/a.py"])
        during = {}

        def worker(job):
            during.update(jobs.load("alpha"))

        daemon.run(worker, cycles=1, sleep=lambda s: None)
        self.assertEqual(during["state"], jobs.RUNNING)
        self.assertEqual(during["daemon_pid"], os.getpid())

    def test_a_finished_job_is_marked_done(self):
        jobs.enqueue("alpha", "index", ["/a.py"])
        daemon.run(lambda job: None, cycles=1, sleep=lambda s: None)
        self.assertEqual(jobs.load("alpha")["state"], jobs.DONE)

    def test_a_worker_that_RAISES_marks_the_job_failed_and_the_daemon_survives(self):
        """A job that blows up must not take the daemon down with it: the other repos keep
        going."""
        jobs.enqueue("alpha", "index", ["/a.py"])

        def broken(job):
            raise RuntimeError("qdrant is down")

        out = daemon.run(broken, cycles=2, sleep=lambda s: None)
        self.assertEqual(out, "cycles exhausted")
        job = jobs.load("alpha")
        self.assertEqual(job["state"], jobs.FAILED)
        self.assertIn("qdrant is down", job["error"])

    def test_a_cancelled_job_is_marked_cancelled_and_not_run_again(self):
        jobs.enqueue("alpha", "index", ["/a.py"])
        jobs.request_cancel("alpha")
        seen = []
        daemon.run(lambda job: seen.append(1), cycles=3, sleep=lambda s: None)
        self.assertEqual(seen, [], "a cancelled job was run")
        self.assertEqual(jobs.load("alpha")["state"], jobs.CANCELLED)


class TestOnlyONEDaemon(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_the_record_of_a_dead_daemon_reads_as_none(self):
        daemon._write_record({"pid": 4_000_000, "starttime": "1"})
        self.assertIsNone(daemon.record())

    def test_the_record_of_a_live_daemon_reads_back(self):
        daemon._write_record({"pid": os.getpid(), "starttime": lease.process_start(os.getpid())})
        self.assertIsNotNone(daemon.record())

    def test_starting_when_one_is_already_running_does_not_start_a_second(self):
        daemon._write_record({"pid": os.getpid(), "starttime": lease.process_start(os.getpid())})
        spawned = []
        out = daemon.start(spawn=lambda argv: spawned.append(argv) or 999)
        self.assertEqual(spawned, [], "a second daemon was started")
        self.assertEqual(out["action"], "already")

    def test_starting_with_no_daemon_spawns_one(self):
        spawned = []
        out = daemon.start(spawn=lambda argv: spawned.append(argv) or 4_000_000)
        self.assertEqual(len(spawned), 1)
        self.assertEqual(out["action"], "started")


if __name__ == "__main__":
    unittest.main(verbosity=2)
