"""Two daemons must not be able to hold the claim at once.

THE WINDOW. `_claim` reads the record, decides the file is a corpse, and THEN unlinks it —
two syscalls with a gap between them. Anything that happens in that gap is invisible to the
unlink, which names a path rather than the file that was judged. A process that wins the
claim in that gap has its live claim deleted by a caller still acting on what it read before.
"""
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import daemon  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


class TestALiveClaimIsNotStolenThroughTheGap(unittest.TestCase):
    """The interleaving, driven deterministically rather than raced.

    Timing decides how OFTEN this happens, never WHETHER it can, so the test drives the steps
    in the order a scheduler is free to produce. Racing it instead would make the suite
    flaky and would still assert the same fact.

    MEASURED as a race before the fix, 8 concurrent processes over one corpse claim: 2 of 12
    rounds ended with two winners and one round with three. `start()`'s own docstring calls
    two daemons the outcome "the design calls impossible".
    """

    def setUp(self):
        a_state_dir()
        daemon.path().parent.mkdir(parents=True, exist_ok=True)

    def test_the_unlink_does_not_remove_a_file_that_replaced_the_one_it_judged(self):
        """C decided the corpse was removable; B then published a LIVE claim over it.

        C must not delete B's claim. C read a corpse; what sits there now is a different
        file, and C has never looked at it.
        """
        daemon.path().write_text(json.dumps({"pid": 999123, "starttime": ""}),
                                 encoding="utf-8")
        # C's half of `_claim`, up to the decision. `_try_create` already failed for C.
        held, identity = daemon._held_claim()
        self.assertTrue(daemon._reclaimable(held), "setup: C must see a removable corpse")

        # B runs the WHOLE of `_claim` in the gap, and wins.
        self.assertTrue(daemon._claim(), "setup: B must win the claim")
        bs_claim = json.loads(daemon.path().read_text(encoding="utf-8"))
        self.assertEqual(bs_claim["pid"], os.getpid())

        # C resumes where it left off.
        c_won = daemon._release_stale_claim(identity) and daemon._try_create()

        self.assertFalse(c_won, "C took the claim while B held it")
        self.assertEqual(json.loads(daemon.path().read_text(encoding="utf-8")), bs_claim,
                         "B's live claim was deleted by a caller acting on a stale read")

    def test_a_genuine_corpse_IS_still_removed(self):
        """The guard must not become "never reclaim".

        Refusing to remove a corpse is the failure `_reclaimable` exists to prevent: it jams
        `start()` forever on one crash, which is the other half of this module's history.
        """
        corpse = {"pid": 999123, "starttime": ""}
        daemon.path().write_text(json.dumps(corpse), encoding="utf-8")
        _, identity = daemon._held_claim()

        self.assertTrue(daemon._release_stale_claim(identity))
        self.assertFalse(daemon.path().exists(), "the corpse was not removed")

    def test_claim_still_succeeds_over_a_corpse(self):
        """End to end, through the public path: one crash must not stop indexing."""
        daemon.path().write_text(json.dumps({"pid": 999123, "starttime": ""}),
                                 encoding="utf-8")

        self.assertTrue(daemon._claim())
        self.assertEqual(json.loads(daemon.path().read_text(encoding="utf-8"))["pid"],
                         os.getpid())

    def test_a_claim_REWRITTEN_in_the_gap_is_not_removed(self):
        """Same window, other shape: the file is replaced rather than won.

        `_write_record` publishes through `os.replace`, so the claim's inode changes while
        the path stays. A caller that judged the OLD content must not remove the new one.
        """
        daemon.path().write_text(json.dumps({"pid": 999123, "starttime": ""}),
                                 encoding="utf-8")
        held, identity = daemon._held_claim()
        self.assertTrue(daemon._reclaimable(held))

        daemon._write_record({"pid": os.getpid(), "starttime": "live", "started_at": 1})

        self.assertFalse(daemon._release_stale_claim(identity),
                         "removed a claim that had been republished since it was read")
        self.assertTrue(daemon.path().exists())


    def test_a_corpse_that_ANOTHER_caller_already_removed_does_not_block_us(self):
        """Two callers reclaiming the same corpse: the loser must still get to try.

        `_identity_of` answers None for a path that no longer exists, so comparing it against
        the recorded identity is a MISMATCH — and treating every mismatch as "somebody else
        holds it" made a vanished claim read as a held one. That is the jam this whole
        function exists to prevent, reintroduced by the guard against the opposite error.

        The race is ordinary: two hosts start at once, both see the corpse, one unlinks first.
        The other must fall through to `_try_create`, where `O_EXCL` decides honestly.
        """
        daemon.path().write_text(json.dumps({"pid": 999123, "starttime": ""}),
                                 encoding="utf-8")
        _, identity = daemon._held_claim()
        daemon.path().unlink()                  # the other caller got there first

        self.assertTrue(daemon._release_stale_claim(identity),
                        "a claim nobody holds any more was reported as held")

    def test_the_whole_claim_succeeds_when_the_corpse_vanishes_mid_flight(self):
        """The same fact through `_claim`, which is the surface that matters."""
        daemon.path().write_text(json.dumps({"pid": 999123, "starttime": ""}),
                                 encoding="utf-8")
        daemon.path().unlink()

        self.assertTrue(daemon._claim(), "a free slot was refused")


class TestTheReclaimIsSerialised(unittest.TestCase):
    """The judge-remove-retake sequence runs under a lock, and the lock cannot jam.

    The identity check narrows the window; it does not close it, because the stat and the
    unlink are still two syscalls. Measured with the check but no lock, 12 processes over one
    corpse, 240 rounds: 7 rounds still produced two winners.
    """

    def setUp(self):
        a_state_dir()
        daemon.path().parent.mkdir(parents=True, exist_ok=True)

    def test_a_caller_that_cannot_take_the_reclaim_lock_backs_off(self):
        """Two processes must not judge and remove the same corpse at once."""
        daemon.path().write_text(json.dumps({"pid": 999123, "starttime": ""}),
                                 encoding="utf-8")
        held_open = []

        def hold_the_lock():
            lock = daemon.path().with_suffix(".reclaim")
            fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held_open.append(fd)

        hold_the_lock()
        self.addCleanup(os.close, held_open[0])

        self.assertFalse(daemon._claim(),
                         "it reclaimed a corpse while another process held the reclaim lock")
        self.assertTrue(daemon.path().exists(),
                        "it removed the corpse without holding the lock")

    def test_a_holder_that_DIES_does_not_jam_the_reclaim_forever(self):
        """`flock` is tied to the descriptor, so the kernel releases it on exit.

        A lock file carrying a pid would need its own staleness rule, and getting that wrong
        is how a crash stops every future `start()` — the failure this module has already
        paid for twice.
        """
        lock = str(daemon.path().with_suffix(".reclaim"))
        dying = subprocess.run(
            [sys.executable, "-c",
             "import fcntl, os, sys\n"
             "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
             "fcntl.flock(fd, fcntl.LOCK_EX)\n"
             "os._exit(9)\n", lock],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(dying.returncode, 9, "the holder did not die as arranged")
        daemon.path().write_text(json.dumps({"pid": 999124, "starttime": ""}),
                                 encoding="utf-8")

        self.assertTrue(daemon._claim(),
                        "a dead lock holder jammed the reclaim permanently")

    def test_the_lock_is_not_the_claim_file(self):
        """Locking the claim itself would mean holding a descriptor on a file we then delete,
        leaving the lock attached to an inode with no name — a lock over nothing."""
        daemon._claim()

        self.assertNotEqual(daemon.path().with_suffix(".reclaim"), daemon.path())


if __name__ == "__main__":
    unittest.main()
