"""The loop: runs what is queued, watches, and ends when nobody is using it anymore.

NO TEST HERE STARTS A REAL DAEMON, nor touches Qdrant. `run` takes the work executor and a
cycle count, so the whole loop is exercised in-process — the same choice `refresh_window
(probe=...)` already makes.
"""
import errno
import json
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


class TestSTOPConfirmsBeforeReleasing(unittest.TestCase):
    """The PUBLIC `stop()`, against a record whose liveness cannot be judged.

    THE HOLE THIS CLOSES. `_stop_and_confirm` was taught that a record with no `starttime`
    cannot be re-read, so it falls back to the pid alone. `stop()` carries a SECOND COPY of
    the same signal-then-confirm rule and was not taught anything — it still loops on
    `lease.alive(entry)`, which answers False on its first guard for such a record. So it
    confirmed, in 0.000s, the death of a process that was ignoring SIGTERM and still running,
    then released the claim. That is the precise sequence its own docstring says it exists to
    prevent, and `repos add-all` runs `stop()` then `start()` back to back.

    Measured before the fix: stop() -> True in 0.000s, child still alive, claim unlinked.
    """

    def setUp(self):
        a_state_dir()
        self.child = subprocess.Popen(
            [sys.executable, "-c",
             "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
             "time.sleep(60)"])
        time.sleep(0.4)                       # let the handler install before we signal
        self.addCleanup(self._end_child)

    def _end_child(self):
        self.child.kill()
        self.child.wait()

    def test_it_does_not_confirm_a_death_it_cannot_see(self):
        daemon.path().write_text(json.dumps({"pid": self.child.pid, "starttime": ""}))

        started = time.monotonic()
        got = daemon.stop(timeout_s=0.5)

        self.assertIsNone(self.child.poll(), "the child died; this test proves nothing")
        self.assertFalse(got, "a live process that ignored SIGTERM was reported as stopped")
        self.assertGreaterEqual(time.monotonic() - started, 0.5,
                                "it answered without waiting, so it never confirmed anything")

    def test_it_KEEPS_the_claim_when_it_could_not_confirm(self):
        """The claim is what stops the next `start()` from spawning a second daemon."""
        daemon.path().write_text(json.dumps({"pid": self.child.pid, "starttime": ""}))

        daemon.stop(timeout_s=0.5)

        self.assertTrue(daemon.path().exists(),
                        "the claim was released over a process that is still running")

    def test_it_does_not_signal_a_stranger_that_inherited_the_pid(self):
        """A recycled pid with no starttime: the (pid, starttime) guard cannot help here.

        The record names a pid that now belongs to someone else. The pid is all the evidence
        there is, and it reads as alive — so `stop()` waits out its timeout and answers False
        rather than releasing the claim. Being wrong in this direction costs a delayed daemon;
        the other direction spawns a second one onto a live first.

        THE STRANGER HAS TO SURVIVE THE SIGNAL, and this fixture did not — which made the test
        pass for the opposite of its own reason. `stop()` signals before it confirms, and a
        plain `time.sleep(60)` child has no SIGTERM handler, so it DIED. What then held the
        assertion up was the corpse: nobody reaps a `Popen` this test never waits on, so the
        pid stayed in the table and `os.kill(pid, 0)` reported the dead child as alive. The
        test was green because liveness was being read wrong — the very defect the module was
        changed to fix — and it went red the moment a zombie stopped counting as alive.

        So the stranger now ignores SIGTERM and SAYS SO OVER A PIPE before we signal it, the
        same handshake `test_stop_does_not_confirm_a_death_it_cannot_see` already uses and for
        the same reason: the process exists from the moment it is forked, well before it
        reaches `signal.signal`, so signalling on a timer races the handler.
        """
        other = subprocess.Popen([sys.executable, "-c",
                                  "import signal, sys, time\n"
                                  "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                                  "sys.stdout.write('ready\\n')\n"
                                  "sys.stdout.flush()\n"
                                  "time.sleep(60)\n"],
                                 stdout=subprocess.PIPE, text=True)
        self.addCleanup(lambda: (other.kill(), other.wait()))
        self.assertEqual(other.stdout.readline().strip(), "ready",
                         "the stranger never installed its handler")
        daemon.path().write_text(json.dumps({"pid": other.pid, "starttime": ""}))

        got = daemon.stop(timeout_s=0.3)

        self.assertIsNone(other.poll(), "the stranger died; this test proves nothing")
        self.assertFalse(got, "it reported a stop it could not confirm")
        self.assertTrue(daemon.path().exists(), "it released the claim over a live process")

    def test_it_confirms_nothing_on_a_platform_WITHOUT_proc(self):
        """macOS and Windows, both listed as supported, have no `/proc`.

        THE HOLE THIS CLOSES. The unjudgeable fallback first asked
        `lease.process_start(pid) is not None`, which fixed the Linux symptom and nothing else:
        with no `/proc` that answers None for EVERY pid, so the confirmation loop exits on its
        first turn. Measured with `process_start` stubbed to None, exactly as those platforms
        behave: stop() returned True in 0.000s against a live child ignoring SIGTERM, and
        unlinked the claim. Signal 0 answers on every platform."""
        daemon.path().write_text(json.dumps({"pid": self.child.pid, "starttime": ""}))

        with patch.object(lease, "process_start", lambda pid: None):
            got = daemon.stop(timeout_s=0.3)

        self.assertIsNone(self.child.poll(), "the child died; this test proves nothing")
        self.assertFalse(got, "a live process was confirmed dead where /proc does not exist")
        self.assertTrue(daemon.path().exists(), "the claim was released over a live process")


class TestAnUnREADABLERecordDoesNotJamStartForever(unittest.TestCase):
    """A record with no `starttime` must still be reclaimable, or indexing stops for good.

    THE HOLE THIS CLOSES. `record()` answers such an entry as truthy, so `_claim()`'s
    `if record() is not None: return False` can never take the corpse path and nothing else
    removes the file. Measured: three consecutive `start()` calls each answered
    `{"action": "already"}` with ZERO spawns, where the pre-change tree spawned every time.

    This is not exotic. `start()` writes `starttime = lease.process_start(pid) or ""`, so a
    child that exits immediately persists an empty one on Linux; and on macOS and Windows,
    both of which README.md lists as supported, `/proc` never exists, so `process_start`
    always answers None and EVERY record is written this way. Indexing would stop forever
    with nothing surfaced to the user."""

    def setUp(self):
        a_state_dir()

    def test_a_record_naming_a_DEAD_pid_is_reclaimed_even_without_a_starttime(self):
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        daemon.path().write_text(json.dumps({"pid": dead.pid, "starttime": ""}))
        spawned = []

        def spawn(argv):
            spawned.append(argv)

            return 424242

        first = daemon.start(spawn=spawn, argv=["x"])

        self.assertEqual(first["action"], "started",
                         "a dead daemon's record blocked a new one forever")
        self.assertEqual(len(spawned), 1)

    def test_a_record_naming_a_LIVE_pid_without_a_starttime_still_blocks(self):
        """The other direction, which is the one that must not regress: unknown means alive."""
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(lambda: (child.kill(), child.wait()))
        time.sleep(0.2)
        daemon.path().write_text(json.dumps({"pid": child.pid, "starttime": ""}))
        spawned = []

        got = daemon.start(spawn=lambda argv: spawned.append(argv) or 1, argv=["x"])

        self.assertEqual(got["action"], "already",
                         "it spawned a second daemon on top of a live one")
        self.assertEqual(spawned, [], "a second daemon was spawned")


class TestTheClaimWorksWithoutHardLinks(unittest.TestCase):
    """`os.link` is not available everywhere, and its absence must not mean "never start".

    THE HOLE THIS CLOSES. `_try_create` swallowed every `OSError` from `os.link` as "we did
    not get the claim". `FileExistsError` genuinely means that; `ENOSYS`/`EPERM` from a
    filesystem without hard links means the opposite — nobody holds it and nobody ever will.
    Measured with `os.link` raising ENOSYS: `start()` answered `{"action": "already",
    "pid": 0}` with zero spawns and no record on disk, forever."""

    def setUp(self):
        a_state_dir()

    def test_a_filesystem_without_hard_links_can_still_claim(self):
        spawned = []

        def spawn(argv):
            spawned.append(argv)

            return 4242

        def no_links(src, dst):
            raise OSError(errno.ENOSYS, "Function not implemented")

        with patch("os.link", no_links):
            got = daemon.start(spawn=spawn, argv=["x"])

        self.assertEqual(got["action"], "started",
                         "the daemon is unstartable where hard links are unavailable")
        self.assertEqual(len(spawned), 1)
        self.assertTrue(daemon.path().exists(), "no record was published")

    def test_it_is_still_exclusive_without_hard_links(self):
        """The fallback must keep the guarantee the link was there to give."""
        def no_links(src, dst):
            raise OSError(errno.ENOSYS, "Function not implemented")

        with patch("os.link", no_links):
            first = daemon._try_create()
            second = daemon._try_create()

        self.assertTrue(first, "the first caller did not get the claim")
        self.assertFalse(second, "two callers both won the same claim")

    def test_the_claim_is_not_world_readable(self):
        """It names a pid this user controls; the old `os.open(..., 0o600)` said so."""
        daemon._try_create()

        mode = os.stat(daemon.path()).st_mode & 0o777

        self.assertEqual(mode, 0o600, f"the claim is mode {oct(mode)}")

    def test_it_is_STILL_private_after_the_daemon_saves(self):
        """The mode has to survive the first `record()`, and it did not.

        MEASURED ON THE RUNNING DAEMON, which is how this was found and not by reading code:
        `~/.memories-plugin/state/daemon.json` was mode 0o664 on a machine whose umask is
        0o002, while the test above passed. `_claim` opens the file 0o600, then the daemon
        publishes through `statefile.write_json`, whose temporary was created by
        `Path.write_text` -- umask, so 0o664 -- and `os.replace` carries the TEMPORARY's mode
        onto the target. The careful mode on the claim lasted until the first save.

        Pinning only the creation moment is what let that through, so this pins the file
        AFTER the write the daemon actually performs."""
        daemon._try_create()
        daemon._write_record({"pid": os.getpid(), "started_at": 1.0, "starttime": "42"})

        mode = os.stat(daemon.path()).st_mode & 0o777

        self.assertEqual(
            mode, 0o600,
            f"the claim is mode {oct(mode)} after a save; the umask was applied by the "
            f"publisher and overwrote the mode the claim was created with")


class TestAClaimIsNotAValidCorpse(unittest.TestCase):
    """The window between creating the claim file and filling it must not read as a corpse.

    `_try_create` takes the claim with `os.open(O_CREAT|O_EXCL)` and writes the placeholder
    in a SECOND call, so between the two the file exists and is EMPTY. `record()` parses it,
    `json.loads("")` raises, and an empty file therefore reads exactly like the record of a
    daemon that died — which sends the concurrent caller down `_claim`'s corpse path, where it
    unlinks the live claim and creates its own on top. The docstring of `_try_create` argues
    at length that a colliding `_claim()` "reads back an alive entry (this process) and
    correctly backs off, instead of mistaking our in-progress claim for a stale one and
    tearing it out from under us". That is the behaviour these tests demand; it was not the
    behaviour the code had.

    Measured before the fix, deterministically (no race needed): a 0-byte `daemon.json` gave
    `record() -> None`, and `_claim()` on top of it returned True with the file rewritten to
    the caller's own pid. Under real concurrency that is a SECOND daemon: 6 of 12 races of 8
    processes ended with two spawns."""

    def setUp(self):
        self.state = a_state_dir()

    def test_an_EMPTY_record_is_not_read_as_a_dead_daemon(self):
        daemon.path().parent.mkdir(parents=True, exist_ok=True)
        daemon.path().write_bytes(b"")
        self.assertIsNotNone(daemon.record(),
                             "a half-written claim read as a corpse")

    def test_a_claim_in_progress_is_not_STOLEN_by_a_concurrent_claim(self):
        daemon.path().parent.mkdir(parents=True, exist_ok=True)
        daemon.path().write_bytes(b"")          # another process, mid-claim
        self.assertFalse(daemon._claim(),
                         "we took a claim another process was still writing")
        self.assertEqual(daemon.path().read_bytes(), b"",
                         "we overwrote a live claim")


class TestUnknownIsNotDead(unittest.TestCase):
    """"I could not tell" must never be read as "it is dead" where the answer frees a claim.

    `lease.process_start` returns None on a platform without `/proc`, and `core/lease.py:18`
    calls that the safe direction — for a LEASE it is: a host whose liveness cannot be read
    should not keep a daemon alive forever. For the CLAIM the safe direction is the opposite,
    and reusing the same predicate inverted the guarantee in two places:

      `_claim()` — a record with an empty starttime reads as a corpse, so the caller unlinks
      a claim whose owner is alive and creates its own on top. Measured with `process_start`
      returning None: three consecutive `_claim()` calls all returned True.

      `_stop_and_confirm()` — its docstring promises it "returns True only once it is
      CONFIRMED gone", by re-reading `(pid, starttime)`. With an empty starttime `lease.alive`
      returns False on its first guard, so it confirmed the death of a process that was still
      running, in 0.000s, measured against a child ignoring SIGTERM.

    Both end the same way: a second daemon on top of a live first one, which the spec calls
    impossible."""

    def setUp(self):
        self.state = a_state_dir()

    def test_a_claim_whose_owner_CANNOT_BE_JUDGED_is_not_stolen(self):
        with patch.object(lease, "process_start", lambda pid: None):
            self.assertTrue(daemon._claim(), "the first claim should win")
            self.assertFalse(daemon._claim(),
                             "a claim we cannot judge was taken from its owner")

    def test_stop_does_not_confirm_a_death_it_cannot_see(self):
        # The child TELLS US when its handler is installed, over a pipe. Waiting for
        # `process_start` to answer instead is not the same event: the process exists from the
        # moment it is forked, well before it reaches `signal.signal`, so the SIGTERM landed in
        # the window before the handler and the child really did die — making the assertion
        # pass for the wrong reason, which is worse than failing.
        child = subprocess.Popen([sys.executable, "-c",
                                  "import signal, sys, time\n"
                                  "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                                  "sys.stdout.write('ready\\n')\n"
                                  "sys.stdout.flush()\n"
                                  "time.sleep(30)\n"],
                                 stdout=subprocess.PIPE, text=True)
        self.addCleanup(child.wait)
        self.addCleanup(child.kill)
        self.assertEqual(child.stdout.readline().strip(), "ready",
                         "the child never installed its handler")
        confirmed = daemon._stop_and_confirm({"pid": child.pid, "starttime": ""},
                                             timeout_s=0.3)
        self.assertFalse(confirmed,
                         "it confirmed the death of a process that is still running")
        self.assertIsNotNone(lease.process_start(child.pid), "the child died on its own")


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
        """WAITS FOR THE HANDLER, for the same reason its sibling above does — and this test
        proves the reason is not hypothetical. Without the handshake the SIGTERM lands before
        `signal.signal(...)` runs, the child dies under the default disposition, and nothing
        reaps it; the surviving `/proc/<pid>/stat` of the ZOMBIE then read as "still alive" and
        this test passed while exercising a process that had already exited — the opposite of
        what its name claims. Treating a zombie as gone removed that cover, which is how the
        sibling path was found."""
        ready = Path(tempfile.mkdtemp()) / "ready"
        proc = subprocess.Popen([sys.executable, "-c",
                                 "import signal, time\n"
                                 "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                                 f"open({str(ready)!r}, 'w').write('ready')\n"
                                 "time.sleep(30)\n"])
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(ready.exists(), "the child never installed its SIGTERM handler")
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
        ALIVE when `start()` is free to spawn its replacement.

        ASSERTS THE FIRST PROCESS IS DEAD, not merely that the second one spawned. An earlier
        version checked only `action == "started"` and `len(spawned) == 1`, which an immediate
        unlink satisfies just as well as a confirmed one — the claim is free either way, so
        exactly one spawn happens either way and the test could not fail for the defect it was
        named after. What separates the two designs is WHEN the claim is released relative to
        the first process exiting, so that is what this asserts."""
        ready = Path(tempfile.mkdtemp()) / "ready"
        proc = subprocess.Popen([sys.executable, "-c",
                                 "import signal, sys, time\n"
                                 "signal.signal(signal.SIGTERM, "
                                 "lambda *a: (time.sleep(0.2), sys.exit(0)))\n"
                                 f"open({str(ready)!r}, 'w').write('ready')\n"
                                 "time.sleep(30)\n"])
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(ready.exists(), "the child never installed its SIGTERM handler")
            recorded = lease.process_start(proc.pid)
            daemon._write_record({"pid": proc.pid, "starttime": recorded})
            self._reap_in_background(proc)
            self.assertTrue(daemon.stop(poll_s=0.01))
            self.assertFalse(lease.alive({"pid": proc.pid, "starttime": recorded}),
                             "the claim was released while the first daemon was still alive — "
                             "a start() here would put a second one on top of it")
            spawned = []
            out = daemon.start(spawn=lambda argv: spawned.append(argv) or os.getpid())
            self.assertEqual(out["action"], "started")
            self.assertEqual(len(spawned), 1)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)


class TestCoreDoesNotDependOnTheCliLayer(unittest.TestCase):
    """Whole-project review, the one real dependency inversion. `core.daemon.start` spawned
    `cli/qctx.py repos daemon run`, so the host-neutral layer named a file in the CLI layer:
    shipping `core` alone, or adding a third host, meant editing core to point elsewhere. The
    loop being started lives in core, so core is now its own entry point."""

    def setUp(self):
        # ITS OWN STATE DIR. Without this the class inherited whatever directory the previous
        # test left in the environment, `start()` found that test's claim still on disk, and
        # returned `already` without spawning. It passed only because a claim left behind read
        # as a corpse worth removing — the very confusion `record()` no longer makes.
        a_state_dir()

    def test_the_spawned_command_does_not_name_the_cli(self):
        seen = []
        daemon.start(spawn=lambda argv: seen.append(argv) or os.getpid())
        self.assertTrue(seen, "nothing was spawned, so this asserts nothing")
        joined = " ".join(seen[0])
        self.assertNotIn("qctx", joined,
                         f"core spawns the CLI again: {joined}")
        self.assertIn("core.daemon", joined,
                      f"core does not spawn its own entry point: {joined}")

    def test_the_entry_point_actually_runs_and_exits_without_a_lease(self):
        """Not just that the module is named -- that `python -m core.daemon` REALLY starts and
        reaches the lease check. A command that only looks right in an assertion is how a
        detached process that never runs goes unnoticed: nothing reports its failure."""
        root = Path(__file__).resolve().parent.parent
        env = dict(os.environ, QCTX_STATE_DIR=tempfile.mkdtemp(), PYTHONPATH=str(root))
        out = subprocess.run([sys.executable, "-m", "core.daemon"],
                             capture_output=True, text=True, timeout=60,
                             cwd=tempfile.mkdtemp(), env=env)
        self.assertEqual(out.returncode, 0, f"the entry point failed: {out.stderr}")
        self.assertIn("no live lease", out.stdout,
                      f"it did not reach the lease check: {out.stdout!r} {out.stderr!r}")

    def test_the_child_can_import_core_regardless_of_the_working_directory(self):
        """`-m` needs `core` importable, and this is spawned from a hook, from the hermes
        provider and from the CLI -- each with a different cwd. The spawn puts the package root
        on PYTHONPATH rather than trusting whatever directory it inherited."""
        captured = {}
        real_popen = subprocess.Popen

        def recording(argv, **kwargs):
            captured.update(kwargs)

            return real_popen([sys.executable, "-c", ""], **kwargs)

        with patch.object(daemon.subprocess, "Popen", recording):
            daemon._spawn([sys.executable, "-m", "core.daemon"])
        root = str(Path(__file__).resolve().parent.parent)
        self.assertIn(root, (captured.get("env") or {}).get("PYTHONPATH", ""),
                      "the package root is not on the child's PYTHONPATH")


if __name__ == "__main__":
    unittest.main(verbosity=2)
