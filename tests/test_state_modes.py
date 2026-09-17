"""Every file this package publishes into the state directory carries the same mode.

WHY THIS FILE EXISTS. `core/statefile.py` owns the atomic publish AND the file mode, and its
own comment explains why the mode is not cosmetic: the daemon's claim is opened 0o600 and
"the careful mode on the claim lasted exactly until the first save" until that module took
the write over. That reasoning applies to every state file, not only the JSON ones -- but
five writers never went through it, each hand-rolling `write_text` and taking whatever the
umask gave them.

MEASURED on the real machine before this existed: 105 of 107 files in
`~/.memories-plugin/state` were published 0o664 under the common umask 0o002, and only the
two that go through `statefile.write_json` were 0o600. Among the 0o664 files are
`recall-<session>.json`, which hold the round number and THE IDS OF THE MEMORIES INJECTED
INTO EACH SESSION -- not the memory text, but a per-session map of what was recalled and
when, readable by every account on the machine.

WHAT THIS PINS, AND WHAT IT DELIBERATELY DOES NOT. It pins the MODE of what each writer
publishes, by driving the real writer against a real temporary directory and reading the
mode back off the disk. It does not pin atomicity for the text writers: a counter and a log
have no torn-read consequence worth a rename, and claiming otherwise in a test name would be
the more dangerous kind of wrong.
"""
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import breaker, session_state, statefile  # noqa: E402

#: What every state file this package writes must be published as: owner read/write only.
#: Named once so a change of policy is one edit and every writer below moves together.
STATE_MODE = 0o600


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


class TestEveryStateWriterPublishesTheSameMode(unittest.TestCase):
    """One mode, whatever the umask, for every writer that lands a file in the state dir.

    Each test drives the REAL writer -- no monkeypatching of the thing under test -- with a
    umask that would produce 0o664 if the writer let the umask decide. That umask is the
    whole point: a writer that sets its mode explicitly is unaffected by it, and a writer
    that does not is caught by it.
    """

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        # THE UMASK IS THE INSTRUMENT. 0o002 is the common default on this distribution and
        # is what produced the 105 group-readable files measured on the real machine; a
        # writer that opens its file without an explicit mode inherits it and fails here.
        self.old_umask = os.umask(0o002)
        self.addCleanup(os.umask, self.old_umask)

    def test_the_json_publisher_that_already_owns_the_mode_still_owns_it(self):
        """The baseline: `statefile.write_json` is the writer the others must match."""
        target = self.dir / "owned.json"
        self.assertTrue(statefile.write_json(target, {"a": 1}))
        self.assertEqual(_mode(target), STATE_MODE)

    def test_the_session_state_save_publishes_600(self):
        """`recall-<session>.json` carries the ids of the memories injected into a session."""
        target = self.dir / "recall-abc.json"
        session_state.save(target, {"round": 3, "seen": {"id-1": 1}})
        self.assertEqual(_mode(target), STATE_MODE,
                         "the per-session recall map must not be group-readable")

    def test_the_breaker_arm_publishes_600(self):
        target = self.dir / "rerank-breaker"
        breaker.Breaker(target, 60).arm()
        self.assertEqual(_mode(target), STATE_MODE)

    def test_the_checkpoint_counter_publishes_600(self):
        """`hooks/checkpoint.py` writes one counter per session, by hand."""
        from hooks import checkpoint

        target = self.dir / "checkpoint-abc.count"
        checkpoint.bump(target)
        self.assertEqual(_mode(target), STATE_MODE)

    def test_the_recall_log_rotation_publishes_600(self):
        """Rotating the log REWRITES it, which is where a careful mode gets thrown away.

        The log is created 0o664 on purpose here: rotation has to PUBLISH 0o600 over a file
        that was group-readable, which is the real sequence on a machine where earlier
        versions of this package already left one behind.
        """
        from hooks import recall

        target = self.dir / "recall.log"
        target.write_text("x" * 100)
        os.chmod(target, 0o664)
        self.assertTrue(recall.rotate(target, max_bytes=10), "it must actually rotate")
        self.assertEqual(_mode(target), STATE_MODE)

    def test_the_config_save_publishes_600(self):
        """The config file holds the qdrant url and the collection names."""
        from core import config

        target = self.dir / "config.json"
        config.save({"qdrant_url": "http://x"}, path=target)
        self.assertEqual(_mode(target), STATE_MODE)


class TestTheModeSurvivesRewriting(unittest.TestCase):
    """A file rewritten in place must not silently regain the umask's mode.

    This is the shape that produced the original defect in `statefile`: the careful mode was
    applied once at creation and lost on the next save. A writer that only chmods on create
    passes the tests above and fails here.
    """

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.old_umask = os.umask(0o002)
        self.addCleanup(os.umask, self.old_umask)

    def test_a_second_session_state_save_keeps_600(self):
        target = self.dir / "recall-abc.json"
        session_state.save(target, {"round": 1, "seen": {}})
        session_state.save(target, {"round": 2, "seen": {"x": 1}})
        self.assertEqual(_mode(target), STATE_MODE)
        self.assertEqual(json.loads(target.read_text())["round"], 2,
                         "the content still lands; the mode fix must not cost the write")

    def test_a_second_checkpoint_bump_keeps_600(self):
        from hooks import checkpoint

        target = self.dir / "checkpoint-abc.count"
        checkpoint.bump(target)
        checkpoint.bump(target)
        self.assertEqual(_mode(target), STATE_MODE)
        self.assertEqual(target.read_text().strip(), "2",
                         "the counter still counts; the mode fix must not cost the write")


if __name__ == "__main__":
    unittest.main()
