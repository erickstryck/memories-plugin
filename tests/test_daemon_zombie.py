"""The daemon's claim, against a child that died and was never reaped.

WHY A SEPARATE FILE. `tests/test_daemon.py` covers the claim under a LIVE process and under
a genuinely absent one. The state between those two — a process that has exited but whose
parent has not collected it — is reachable by construction here (`core/daemon.py::_spawn`
never waits, and says so: "Measured: one Z per spawn"), and it is the state where the two
liveness tests in this module disagree with each other.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import daemon, lease  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


#: Holds every `Popen` this module creates, for the LIFETIME OF THE MODULE.
#:
#: NOT tidiness — without it these tests pass for the wrong reason. `Popen.__del__` reaps the
#: child, so letting the object fall out of scope makes the pid vanish entirely and
#: `os.kill(pid, 0)` then raises `ProcessLookupError`: the assertions below go green while no
#: zombie was ever observed. Measured: with the reference dropped, `/proc/<pid>/stat` was gone
#: 0.2s later; with it held, the child sits in `Z` and signal 0 still reports it alive.
#:
#: Production has exactly this reference — `core/daemon.py::_spawned` keeps every `Popen` and
#: only polls them opportunistically on the NEXT spawn — so holding it here is what makes the
#: fixture match the code under test.
_UNREAPED: list = []


def an_unreaped_child() -> int:
    """A pid in state `Z`: exited, never collected. Returns the pid.

    `subprocess.Popen` with nobody calling `wait()` is exactly what `daemon._spawn` does, so
    this is the production shape and not a contrivance.
    """
    child = subprocess.Popen([sys.executable, "-c", ""])
    _UNREAPED.append(child)                     # see `_UNREAPED`: dropping this reaps it
    for _ in range(200):                        # wait for it to actually reach Z
        time.sleep(0.01)
        try:
            with open(f"/proc/{child.pid}/stat", encoding="utf-8") as fh:
                if fh.read().rsplit(")", 1)[1].split()[0] == "Z":
                    return child.pid
        except OSError:
            break

    raise unittest.SkipTest("could not produce a zombie on this platform")


@unittest.skipUnless(os.path.isdir("/proc"), "needs /proc to observe a zombie")
class TestAZombieIsNotAliveForTheCLAIM(unittest.TestCase):
    """A daemon that died and was never reaped must not hold its claim forever.

    THE DISAGREEMENT. `lease.process_start` reports state `Z` as gone, deliberately and with
    its own comment saying why. `os.kill(pid, 0)` reports it as ALIVE, because the pid still
    occupies a slot in the process table. `_pid_alive_fn` asks the second question, and
    `_reclaimable` and `_stop_and_confirm` both route through it for a record with no
    `starttime`.

    WHY THAT RECORD IS THE COMMON ONE, not an edge case. `start()` persists
    `lease.process_start(pid) or ""`, so a daemon that dies before that read gets an empty
    `starttime` on Linux — and on macOS and Windows, both listed as supported, `/proc` never
    exists, so EVERY record is written that way.

    MEASURED before the fix, against a child that exits immediately and is never waited on:
    `start()` answered `{"action": "already"}` three times in a row, `stop()` returned False
    after its full timeout, and the claim stayed on disk. Indexing stopped for good with no
    documented way out.
    """

    def setUp(self):
        a_state_dir()

    def test_a_zombie_does_not_read_as_alive(self):
        pid = an_unreaped_child()

        self.assertFalse(daemon._pid_alive_fn({"pid": pid})(),
                         "signal 0 reported a zombie as a running process")

    def test_the_claim_of_a_zombie_daemon_can_be_RECLAIMED(self):
        pid = an_unreaped_child()
        entry = {"pid": pid, "starttime": ""}

        self.assertTrue(daemon._reclaimable(entry),
                        "a claim held by a dead-but-unreaped daemon could not be reclaimed")

    def test_START_is_not_jammed_forever_by_a_zombie_daemon(self):
        """The whole point: a new daemon can still be started.

        This drives the PUBLIC `start()`, through the real `_claim`, because that is the
        surface whose failure the user experiences as "indexing silently stopped".
        """
        pid = an_unreaped_child()
        daemon.path().parent.mkdir(parents=True, exist_ok=True)
        daemon.path().write_text(json.dumps({"pid": pid, "starttime": ""}),
                                 encoding="utf-8")

        started = daemon.start(spawn=lambda argv: 4242)

        self.assertEqual(started["action"], "started")
        self.assertEqual(started["pid"], 4242)

    def test_STOP_reports_the_zombie_daemon_as_stopped_and_frees_the_claim(self):
        """`stop()` must not spin for its whole timeout over a process already gone.

        Before the fix `_stop_and_confirm` polled `_pid_alive_fn`, which never went false, so
        `stop()` ran out its timeout and returned False — and `_release_claim()` after it was
        never reached.
        """
        pid = an_unreaped_child()
        daemon.path().parent.mkdir(parents=True, exist_ok=True)
        daemon.path().write_text(json.dumps({"pid": pid, "starttime": ""}),
                                 encoding="utf-8")

        self.assertTrue(daemon.stop(timeout_s=1.0),
                        "stop() could not confirm the death of an already-dead daemon")
        self.assertFalse(daemon.path().exists(), "the claim was not released")

@unittest.skipUnless(os.path.isdir("/proc"), "needs /proc")
class TestALiveProcessStillHoldsItsClaim(unittest.TestCase):
    """The guard must not swing the other way.

    Teaching the claim to see a zombie must not teach it to steal a claim from a process that
    is merely unfamiliar — that is the failure `_reclaimable` was written to prevent, and it
    costs two daemons rather than one delay.
    """

    def setUp(self):
        a_state_dir()

    def test_a_LIVE_process_without_a_starttime_is_not_reclaimable(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(self._end, child)
        time.sleep(0.2)

        self.assertFalse(daemon._reclaimable({"pid": child.pid, "starttime": ""}),
                         "a live daemon's claim was declared reclaimable")

    def test_a_record_with_no_readable_pid_is_not_reclaimable(self):
        """A claim being written RIGHT NOW looks like this from here."""
        self.assertFalse(daemon._reclaimable({"starttime": ""}))
        self.assertFalse(daemon._reclaimable({"pid": "not-a-number", "starttime": ""}))

    def test_a_pid_this_user_may_not_signal_is_treated_as_alive(self):
        """`os.kill` raises PermissionError for a process owned by someone else.

        That is evidence the process EXISTS, so it must keep reading as alive; only
        `ProcessLookupError` and a zombie mean gone.
        """
        self.assertTrue(daemon._pid_alive_fn({"pid": 1})(),
                        "pid 1 (init, not ours to signal) read as dead")

    def _end(self, child):
        child.kill()
        child.wait()


if __name__ == "__main__":
    unittest.main()
