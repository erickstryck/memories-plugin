"""The state of jobs on disk, which is the only contract between commands and the daemon.

There is no socket or pipe: `add-all` writes a file, the daemon reads it next cycle, and
`status` reads the same file. This is what makes `status` work even with the daemon dead — and
this is why a `reap` must exist, because a file that says `running` under a daemon that has
died is lying.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import jobs  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


class TestTheQueue(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_an_enqueued_job_comes_back_pending(self):
        jobs.enqueue("alpha", "index", ["/a.py", "/b.py"])
        job = jobs.load("alpha")
        self.assertEqual(job["state"], jobs.PENDING)
        self.assertEqual(job["total"], 2)
        self.assertEqual(job["done"], 0)

    def test_next_pending_returns_it_and_None_when_there_is_nothing(self):
        self.assertIsNone(jobs.next_pending())
        jobs.enqueue("alpha", "index", ["/a.py"])
        self.assertEqual(jobs.next_pending()["repo"], "alpha")

    def test_next_pending_SKIPS_what_is_already_running(self):
        """One job at a time: two workers on one repository duplicate the work without
        finishing sooner.

        TWO jobs, deliberately. With only a running one the pending list is empty and
        `next_pending` returns None without ever consulting the guard — the test would pass
        with the guard deleted, which is exactly what the mutation probe measured.
        """
        jobs.enqueue("alpha", "index", ["/a.py"])
        jobs.update("alpha", state=jobs.RUNNING)
        jobs.enqueue("beta", "index", ["/b.py"])
        self.assertIsNone(jobs.next_pending(),
                          "a second job was handed out while one was already running")

    def test_next_pending_RESUMES_once_nothing_is_running(self):
        """The other half. A guard that never hands out work would satisfy the test above,
        so something has to prove the queue still moves."""
        jobs.enqueue("alpha", "index", ["/a.py"])
        jobs.update("alpha", state=jobs.RUNNING)
        jobs.enqueue("beta", "index", ["/b.py"])
        jobs.update("alpha", state=jobs.DONE)
        self.assertEqual(jobs.next_pending()["repo"], "beta")

    def test_enqueuing_the_same_repo_again_REPLACES_the_job(self):
        """Asking again means asking about the current state of the disk, not summing onto the
        old queue."""
        jobs.enqueue("alpha", "index", ["/a.py"])
        jobs.enqueue("alpha", "index", ["/a.py", "/b.py"])
        self.assertEqual(jobs.load("alpha")["total"], 2)
        self.assertEqual(len(jobs.all_jobs()), 1)

    def test_a_repo_with_no_job_loads_as_None(self):
        self.assertIsNone(jobs.load("never-queued"))


class TestProgressAndCancellation(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_progress_is_readable_while_it_runs(self):
        jobs.enqueue("alpha", "index", ["/a.py", "/b.py", "/c.py"])
        jobs.update("alpha", state=jobs.RUNNING, done=2, current="/b.py")
        job = jobs.load("alpha")
        self.assertEqual((job["done"], job["total"]), (2, 3))
        self.assertEqual(job["current"], "/b.py")

    def test_a_cancel_request_is_visible_to_whoever_is_running_it(self):
        """Cancellation is a request on disk, not a signal: the daemon reads it between files,
        which lets it stop at a point where what is already indexed is consistent."""
        jobs.enqueue("alpha", "index", ["/a.py"])
        self.assertFalse(jobs.cancel_requested("alpha"))
        self.assertTrue(jobs.request_cancel("alpha"))
        self.assertTrue(jobs.cancel_requested("alpha"))

    def test_cancelling_a_repo_with_no_job_answers_False_instead_of_raising(self):
        self.assertFalse(jobs.request_cancel("never-queued"))

    def test_update_on_a_missing_job_answers_None_instead_of_creating_one(self):
        """An `update` that creates a job from nothing would let a late cancellation
        resurrect a job that already finished."""
        self.assertIsNone(jobs.update("never-queued", done=1))


class TestAJobThatOUTLIVEDItsDaemon(unittest.TestCase):
    """A file saying `running` under a daemon that has died is lying, and a state that lies is
    worse than a state that is absent: `status` would show stalled progress as if it were
    activity."""

    def setUp(self):
        a_state_dir()

    def test_reap_marks_a_running_job_of_a_dead_daemon_as_interrupted(self):
        jobs.enqueue("alpha", "index", ["/a.py"])
        jobs.update("alpha", state=jobs.RUNNING, daemon_pid=4_000_000, done=1)
        reaped = jobs.reap(lambda pid: False)
        self.assertEqual(reaped, ["alpha"])
        job = jobs.load("alpha")
        self.assertEqual(job["state"], jobs.FAILED)
        self.assertIn("interrupted", job["error"])
        self.assertEqual(job["done"], 1, "progress was not lost when marking as interrupted")

    def test_reap_leaves_a_job_of_a_LIVING_daemon_alone(self):
        jobs.enqueue("alpha", "index", ["/a.py"])
        jobs.update("alpha", state=jobs.RUNNING, daemon_pid=os.getpid())
        self.assertEqual(jobs.reap(lambda pid: True), [])
        self.assertEqual(jobs.load("alpha")["state"], jobs.RUNNING)

    def test_reap_leaves_a_PENDING_job_alone(self):
        """Pending is not the responsibility of any daemon yet — it is the normal state of a
        job queued before the daemon comes up."""
        jobs.enqueue("alpha", "index", ["/a.py"])
        self.assertEqual(jobs.reap(lambda pid: False), [])
        self.assertEqual(jobs.load("alpha")["state"], jobs.PENDING)


if __name__ == "__main__":
    unittest.main(verbosity=2)
