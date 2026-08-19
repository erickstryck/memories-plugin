"""The loop: runs what is queued, watches, and ends when nobody is using it anymore.

NO TEST HERE STARTS A REAL DAEMON, nor touches Qdrant. `run` takes the work executor and a
cycle count, so the whole loop is exercised in-process — the same choice `refresh_window
(probe=...)` already makes.
"""
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

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


class TestItWatches(unittest.TestCase):
    """`watch()` runs on every cycle where no job is pending — same loop, same survival rule
    the worker already gets."""

    def setUp(self):
        a_state_dir()

    def test_a_watcher_that_RAISES_does_not_end_the_daemon(self):
        """`run`'s contract is that one broken repository must not stop the others, and the
        worker already honours it. The watcher reaches `jobs.enqueue`, which raises when the
        state directory cannot be written — so without the same guard, one unwritable repo
        would take down indexing for every repository being watched."""
        a_live_lease()

        def exploding_watch():
            raise jobs.JobError("the state dir vanished")

        out = daemon.run(lambda job: None, cycles=3, sleep=lambda s: None,
                         watch=exploding_watch)
        self.assertEqual(out, "cycles exhausted",
                         "an exception from watch() ended the daemon")


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

    def test_a_second_start_LOSES_the_race_and_does_not_spawn(self):
        """The real race window is between `_claim()` succeeding and the real record being
        written by `spawn` — two SEQUENTIAL calls (one fully finished before the next begins)
        cannot tell an atomic claim from a plain check-then-write, since by the time the second
        call runs, the first has already written its final record either way. So the second
        `start()` is nested INSIDE the first one's `spawn`, landing exactly in that window: with
        `_claim()`, the placeholder written before spawn already blocks it; with check-then-write,
        nothing has been written yet at that point and the nested call sails through and spawns
        a second daemon."""
        spawned = []
        nested = {}

        def first_spawn(argv):
            spawned.append(argv)
            nested["result"] = daemon.start(spawn=lambda a: spawned.append(a) or 4_000_000)

            return os.getpid()

        first = daemon.start(spawn=first_spawn)
        self.assertEqual(first["action"], "started")
        self.assertEqual(nested["result"]["action"], "already")
        self.assertEqual(len(spawned), 1, "a second daemon was spawned inside the race window")

    def test_a_STALE_record_does_not_block_a_start_forever(self):
        """A daemon killed without cleaning up leaves a record whose pid is dead. Without the
        retry, that one crash would make every future start impossible."""
        daemon._write_record({"pid": 4_000_000, "starttime": "1"})
        spawned = []
        out = daemon.start(spawn=lambda argv: spawned.append(argv) or os.getpid())
        self.assertEqual(out["action"], "started")
        self.assertEqual(len(spawned), 1)


class TestStartCleansUpItsOwnClaim(unittest.TestCase):
    """Whole-branch review finding 4's hazard: `start()` is now called from INSIDE a
    long-lived host process (hermes' `initialize()`), so it cannot lean on "the caller exits
    soon and the stale claim self-heals" the way an ephemeral CLI invocation can. Both ways a
    start can fail after the claim is taken must release it — and the harder of the two must
    not leave a live, untracked process behind either."""

    def setUp(self):
        a_state_dir()

    def test_when_spawn_RAISES_the_claim_is_released_and_the_reason_is_reported(self):
        def boom(argv):
            raise OSError("no such file or directory")

        with self.assertRaises(daemon.DaemonError) as ctx:
            daemon.start(spawn=boom)
        self.assertIn("no such file or directory", str(ctx.exception),
                      "the OSError's reason did not reach the caller")
        self.assertFalse(daemon.path().exists(),
                         "the claim was left behind after a spawn that never ran")

    def test_a_start_after_a_failed_spawn_is_free_to_try_again(self):
        """Proves the claim release is not merely present but EFFECTIVE: a second start with a
        working spawn must succeed, not find the slot still held."""
        with self.assertRaises(daemon.DaemonError):
            daemon.start(spawn=lambda argv: (_ for _ in ()).throw(OSError("boom")))
        spawned = []
        out = daemon.start(spawn=lambda argv: spawned.append(argv) or os.getpid())
        self.assertEqual(out["action"], "started")
        self.assertEqual(len(spawned), 1)

    def test_when_WRITE_RECORD_FAILS_after_a_successful_spawn_the_process_is_killed_and_the_claim_released(self):
        """The harder case, and the one the review's accepted disagreement with an earlier
        ruling was actually about: `spawn` succeeds — a real process now exists — but
        `_write_record` cannot persist it. Leaving the claim as-is would misrepresent the
        CALLER as the daemon; releasing it without stopping the spawned process would let it
        keep running untracked, and a LATER start() would then be free to spawn a second one
        — the one outcome the design calls impossible. This proves both halves: the spawned
        process is actually killed (not merely "should be"), and the claim is gone afterward."""
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            with patch.object(daemon, "_write_record", return_value=False):
                with self.assertRaises(daemon.DaemonError):
                    daemon.start(spawn=lambda argv: proc.pid)
            self.assertFalse(daemon.path().exists(),
                             "the claim was left behind after an unrecordable start")
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.fail("the spawned process was not actually killed")
            self.assertIsNotNone(proc.poll(), "the spawned process is still alive")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


class TestStopWaitsForTheProcessToActuallyDie(unittest.TestCase):
    """Whole-branch review finding 11: `stop()` used to unlink the claim right after sending
    the signal, with no confirmation the process had actually gone. `stop()` immediately
    followed by `start()` — exactly what `add-all` does — could then spawn a SECOND live
    daemon while the first was still mid-shutdown. Entirely untested before this.

    A REAPER THREAD RUNS ALONGSIDE `stop()` IN EVERY TEST HERE THAT USES A REAL SUBPROCESS.
    In production the process `stop()` signals is never a direct child of the caller — it was
    spawned (possibly minutes earlier, by a different invocation) and reparented to init,
    which reaps it the instant it exits. In THIS test the child IS a direct child of the test
    process, so without something calling `wait()` on it, a process that has actually exited
    stays a ZOMBIE — and a zombie's `/proc/<pid>/stat` still reports the SAME starttime, so
    `lease.alive` would read it as alive forever and `stop()` would time out for a reason that
    has nothing to do with the code under test. The reaper thread stands in for init.
    """

    def setUp(self):
        a_state_dir()

    @staticmethod
    def _reap_in_background(proc: subprocess.Popen) -> None:
        threading.Thread(target=proc.wait, daemon=True).start()

    def test_stop_returns_False_when_no_daemon_is_recorded(self):
        self.assertFalse(daemon.stop())

    def test_stop_WAITS_for_a_slow_exit_before_releasing_the_claim(self):
        """The process traps SIGTERM and delays its own exit — proving `stop()` actually
        blocks for that delay (elapsed time), not merely calls `os.kill` and returns.

        WAITS FOR THE CHILD'S HANDLER TO BE INSTALLED before signalling it: without this, the
        signal can arrive before `signal.signal(...)` has run, so the process dies under the
        DEFAULT disposition (immediate) instead of the trapped one — a race that would make
        this test flaky rather than wrong. The child writes `ready` to a marker file the
        instant its handler is registered."""
        ready = Path(tempfile.mkdtemp()) / "ready"
        proc = subprocess.Popen([sys.executable, "-c",
                                 "import signal, sys, time\n"
                                 "signal.signal(signal.SIGTERM, "
                                 "lambda *a: (time.sleep(0.3), sys.exit(0)))\n"
                                 f"open({str(ready)!r}, 'w').write('ready')\n"
                                 "time.sleep(30)\n"])
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(ready.exists(), "the child never installed its SIGTERM handler")
            daemon._write_record({"pid": proc.pid,
                                  "starttime": lease.process_start(proc.pid)})
            self._reap_in_background(proc)
            started_at = time.monotonic()
            ok = daemon.stop(poll_s=0.01)
            elapsed = time.monotonic() - started_at
            self.assertTrue(ok)
            self.assertGreaterEqual(elapsed, 0.25,
                                    "stop() returned before the process actually exited")
            self.assertFalse(daemon.path().exists())
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)

    def test_stop_GIVES_UP_and_keeps_the_claim_if_the_process_will_not_die(self):
        proc = subprocess.Popen([sys.executable, "-c",
                                 "import signal, time\n"
                                 "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                                 "time.sleep(30)\n"])
        try:
            daemon._write_record({"pid": proc.pid,
                                  "starttime": lease.process_start(proc.pid)})
            ok = daemon.stop(timeout_s=0.2, poll_s=0.02)
            self.assertFalse(ok, "stop() claimed success for a process still alive")
            self.assertIsNotNone(daemon.record(),
                                 "the claim was released while the daemon was still alive")
        finally:
            proc.kill()
            proc.wait(timeout=5)

    def test_stop_then_start_does_not_produce_two_daemons(self):
        """The scenario the fix exists for, end to end: a slow-dying daemon must not still be
        claimable by `start()` the instant `stop()` returns True."""
        proc = subprocess.Popen([sys.executable, "-c",
                                 "import signal, sys, time\n"
                                 "signal.signal(signal.SIGTERM, "
                                 "lambda *a: (time.sleep(0.2), sys.exit(0)))\n"
                                 "time.sleep(30)\n"])
        try:
            daemon._write_record({"pid": proc.pid,
                                  "starttime": lease.process_start(proc.pid)})
            self._reap_in_background(proc)
            self.assertTrue(daemon.stop(poll_s=0.01))
            spawned = []
            out = daemon.start(spawn=lambda argv: spawned.append(argv) or os.getpid())
            self.assertEqual(out["action"], "started")
            self.assertEqual(len(spawned), 1)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
