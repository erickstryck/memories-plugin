"""Who is still using the daemon.

A lease is a note saying "I am alive": the HOST process's pid and the moment it started. The
daemon checks the notes each cycle and exits when none is left.

WHY `(pid, starttime)` AND NOT JUST THE PID: the system reuses process numbers. With the pid
alone any process that inherited that number would hold the daemon open forever, and the
failure would be invisible — the daemon merely never exits.
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import lease  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


class TestReadingAProcess(unittest.TestCase):
    def test_our_own_start_time_is_readable(self):
        self.assertTrue(lease.process_start(os.getpid()))

    def test_a_pid_that_does_not_exist_reads_as_None(self):
        """Without raising: the daemon calls this each cycle for pids it expects to see die."""
        self.assertIsNone(lease.process_start(4_000_000))

    def test_the_start_time_is_STABLE_across_reads(self):
        """If it varied, every lease would look recycled and the daemon would exit with the host alive."""
        self.assertEqual(lease.process_start(os.getpid()), lease.process_start(os.getpid()))


class TestWritingAndCheckingALease(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_a_lease_for_a_live_process_is_alive(self):
        entry = lease.write("s1", "claude", pid=os.getpid())
        self.assertTrue(lease.alive(entry))

    def test_a_lease_for_a_DEAD_process_is_not_alive(self):
        done = subprocess.Popen([sys.executable, "-c", "pass"])
        done.wait(timeout=60)
        entry = lease.write("s1", "claude", pid=done.pid)
        self.assertFalse(lease.alive(entry))

    def test_a_RECYCLED_pid_is_not_alive(self):
        """The case the pid alone cannot catch: the number exists but it is a different process."""
        entry = lease.write("s1", "claude", pid=os.getpid())
        entry["starttime"] = "1"          # como se o processo original tivesse começado antes
        self.assertFalse(lease.alive(entry))

    def test_a_lease_missing_its_fields_is_not_alive(self):
        """Truncated file or from an older version: does not hold the daemon."""
        for broken in ({}, {"pid": os.getpid()}, {"starttime": "1"}, {"pid": "x"}):
            with self.subTest(broken=broken):
                self.assertFalse(lease.alive(broken))


class TestSweeping(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_live_returns_the_living_and_REMOVES_the_dead(self):
        done = subprocess.Popen([sys.executable, "-c", "pass"])
        done.wait(timeout=60)
        lease.write("alive", "claude", pid=os.getpid())
        lease.write("dead", "hermes", pid=done.pid)
        live = lease.live()
        self.assertEqual([e["session_id"] for e in live], ["alive"])
        self.assertEqual(sorted(p.name for p in lease.dir().glob("*.json")), ["alive.json"])

    def test_no_leases_at_all_is_an_empty_list_and_not_an_error(self):
        """The normal state before the first session, and the state that makes the daemon exit."""
        self.assertEqual(lease.live(), [])

    def test_a_corrupt_lease_file_is_removed_rather_than_crashing_the_sweep(self):
        """A corrupt lease file is removed rather than crashing the sweep."""
        lease.dir().mkdir(parents=True, exist_ok=True)
        (lease.dir() / "junk.json").write_text("{not json")
        self.assertEqual(lease.live(), [])
        self.assertFalse((lease.dir() / "junk.json").exists())

    def test_a_lease_that_cannot_be_READ_is_kept_rather_than_deleted(self):
        """A transient read error does not mean the process is dead. Corrupt JSON is unusable
        and removing it is right; a permission blip or I/O error says nothing about the process
        the lease names, and deleting it would end the daemon under a live host — the failure
        this module exists to prevent."""
        entry = lease.write("s1", "claude", pid=os.getpid())
        path = lease.dir() / "s1.json"
        os.chmod(path, 0)
        try:
            lease.live()
            self.assertTrue(path.exists(), "a lease that could not be read was deleted")
        finally:
            os.chmod(path, 0o600)


class TestAnAPPROXIMATEResolutionSaysSo(unittest.TestCase):
    """The spec's rule for when the walk up the process tree does not find the host: "o lease
    grava o pid que achou e REGISTRA QUE A RESOLUÇÃO FOI APROXIMADA; o pior caso é o daemon
    sobreviver ao host e ser encerrado no `status` seguinte, NUNCA O CONTRÁRIO."

    Both halves were missing. Nothing recorded the approximation, and the fallback
    (`os.getppid()`) points at the per-hook bash on claude-code — the spec's own measurement
    is `python3 → bash → claude`, and that bash exits the moment the hook returns. So the
    lease was born naming a process already dead, `live()` swept it on the next cycle, and the
    daemon stopped with the host still running: exactly the direction the spec forbids.
    """

    def setUp(self):
        a_state_dir()

    def test_an_approximate_lease_is_MARKED(self):
        entry = lease.write("s1", "claude", pid=os.getpid(), approximate=True)

        self.assertTrue(entry.get("approximate"),
                        "nothing on disk says the host was never actually found")

    def test_an_exact_lease_is_not_marked(self):
        """The other direction, or every lease could carry the flag and mean nothing."""
        self.assertFalse(lease.write("s2", "hermes").get("approximate"))

    def test_an_approximate_lease_of_a_DEAD_process_stays_alive(self):
        """The rule the spec states as "never the other way round". An approximate resolution
        is a guess about WHICH process is the host; a guess that turns out to name a dead
        process must not be read as "the host is gone", because the cost of being wrong that
        way is the background indexing stopping silently while the user is still working.

        A daemon that outlives its host is the cheap error: the next `status` reaps it."""
        dead = {"session_id": "s3", "host": "claude", "pid": 4_000_000,
                "starttime": "never", "approximate": True, "written_at": time.time()}

        self.assertTrue(lease.alive(dead),
                        "an approximate lease was swept, stopping the daemon under a live host")

    def test_an_EXACT_lease_of_a_dead_process_is_still_dead(self):
        """And the tolerance must not leak into the normal case: an exact lease whose process
        is gone is the ordinary end of a session, and the daemon should stop."""
        dead = {"session_id": "s4", "host": "hermes", "pid": 4_000_000, "starttime": "never"}

        self.assertFalse(lease.alive(dead))

    def test_an_approximate_lease_EXPIRES(self):
        """The tolerance is bounded, or it becomes the leak it was meant to avoid: a reboot
        that leaves state behind must not keep a daemon alive on a lease nobody can vouch
        for."""
        stale = {"session_id": "s5", "host": "claude", "pid": 4_000_000, "starttime": "never",
                 "approximate": True,
                 "written_at": time.time() - lease.APPROXIMATE_TTL_S - 1}

        self.assertFalse(lease.alive(stale), "an approximate lease was trusted forever")

    def test_a_FRESH_approximate_lease_is_still_trusted(self):
        fresh = {"session_id": "s6", "host": "claude", "pid": 4_000_000, "starttime": "never",
                 "approximate": True, "written_at": time.time()}

        self.assertTrue(lease.alive(fresh))


class TestFindingTheHOST(unittest.TestCase):
    def _own_exec_name(self) -> str:
        """The name `/proc` reports for THIS process, which is what `find_host_pid` matches.

        NOT `basename(sys.executable)`. The two disagree whenever the interpreter is reached
        through a symlink: measured on this machine, `/home/linuxbrew/.linuxbrew/bin/python3`
        resolves `sys.executable` to `.../bin/python3.14` while `/proc/<pid>/stat` reports
        `python3`, the name it was EXECUTED under. A test asserting on the first form fails
        under that interpreter and passes under the other two, at every commit — it states a
        fact about how the runner was invoked, not about the code.
        """
        with open(f"/proc/{os.getpid()}/stat", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        return raw.split("(", 1)[1].rsplit(")", 1)[0]

    def test_it_walks_up_and_finds_a_named_ancestor(self):
        """On claude-code the hook is a subprocess: python3 -> bash -> claude. Measured on this
        machine in 2026-08-18."""
        found = lease.find_host_pid(names=(self._own_exec_name(),))
        self.assertIsNotNone(found, "did not find the interpreter in the process tree")
        pid, start = found
        self.assertEqual(lease.process_start(pid), start)

    def test_an_ancestor_that_is_not_there_yields_None(self):
        """An ancestor that is not there yields None."""
        self.assertIsNone(lease.find_host_pid(names=("nao-existe-este-processo",)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
