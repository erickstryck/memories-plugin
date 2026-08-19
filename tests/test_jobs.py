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

    def test_enqueue_raises_JobError_when_state_directory_cannot_be_written(self):
        """Enqueue must raise rather than returning a phantom job. A failed write means the
        work will never happen, which is worse than a missing job: status could report it
        queued when it was not."""
        # Create a file where the state directory should be
        state_dir_path = jobs.dir()
        state_dir_path.parent.mkdir(parents=True, exist_ok=True)
        state_dir_path.touch()
        
        # Now enqueue should fail because it cannot create the directory
        with self.assertRaises(jobs.JobError):
            jobs.enqueue("alpha", "index", ["/a.py"])
        
        # Clean up by removing the blocking file
        state_dir_path.unlink()
        # Verify that the job was NOT created
        self.assertIsNone(jobs.load("alpha"),
                          "enqueue created a phantom job despite failing to write")


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

    def test_progress_update_does_NOT_clear_a_cancel_request(self):
        """A daemon updates progress every batch while a user can request cancel at any moment.
        The cancel is a separate file so the daemon's read-modify-write of the job dict does not
        overwrite it. This test reproduces the race: daemon loads stale job, user cancels, daemon
        writes progress, and the cancel must still be visible."""
        jobs.enqueue("alpha", "index", ["/a.py"])
        jobs.update("alpha", state=jobs.RUNNING, done=0)
        jobs.request_cancel("alpha")
        # Daemon's progress update does not see the cancel file, writes its view of the job
        jobs.update("alpha", done=1)
        # Cancel must still be visible
        self.assertTrue(jobs.cancel_requested("alpha"),
                        "progress update overwrote the cancel request")

    def test_enqueue_clears_a_stale_cancel(self):
        """A fresh job must not inherit a cancellation from the previous one."""
        jobs.enqueue("alpha", "index", ["/a.py"])
        jobs.request_cancel("alpha")
        self.assertTrue(jobs.cancel_requested("alpha"))
        # Re-enqueue should clear the old cancel
        jobs.enqueue("alpha", "index", ["/a.py", "/b.py"])
        self.assertFalse(jobs.cancel_requested("alpha"),
                         "enqueue did not clear the stale cancel from previous job")

    def test_cancel_requested_is_False_for_missing_job_and_file(self):
        """No job and no cancel file means nothing to cancel."""
        self.assertFalse(jobs.cancel_requested("never-existed"))

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
        self.assertEqual(job["done"], 1, "progress was lost when marking the job interrupted")

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
