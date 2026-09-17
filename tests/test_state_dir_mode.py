"""The state DIRECTORY, not just the files in it.

WHY THIS FILE EXISTS. `core/statefile.py` was made the owner of the file mode, and every
writer routed through it: the files this package publishes are 0o600. The directory holding
them was still created by twelve scattered `mkdir(parents=True, exist_ok=True)` calls, none
of which names a mode, so it takes the umask -- MEASURED on a fresh install with the common
0o002 umask: `0o775`.

WHY THAT MATTERS EVEN THOUGH THE FILES ARE 0o600. A directory's read bit grants the right to
LIST it, which is a different question from reading what is inside. The names in this one are
`recall-<session-id>.json` and `checkpoint-<session-id>.count`, so a listable state directory
publishes how many sessions this user has had, when each was last active, and the id of every
one of them -- to every account on the machine, without opening a single file.

WHAT IT PINS. That the directory this package CREATES is owner-only (itself and the parents it
makes), and that an existing directory is left as its owner set it: a mode somebody chose is
not this package's to overrule, including a deliberate 0o500 read-only state directory. The
first version corrected every directory it touched and, measured, re-opened that 0o500
directory -- the plugin undoing its user's decision, silently. So the correction is deliberately
narrower than the files next door: a new directory is ours to create correctly, an existing
one's mode belongs to whoever set it.
"""
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import statefile  # noqa: E402

#: What the state directory must be: owner may read, write and traverse; nobody else.
STATE_DIR_MODE = 0o700


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


class TestTheStateDirectoryIsOwnerOnly(unittest.TestCase):
    """The umask decides this today, and 0o002 is the common default on this distribution."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.old_umask = os.umask(0o002)
        self.addCleanup(os.umask, self.old_umask)

    def test_a_directory_this_package_creates_is_700(self):
        target = self.root / "fresh" / "state"
        statefile.ensure_dir(target)
        self.assertEqual(_mode(target), STATE_DIR_MODE)

    def test_the_parents_it_creates_are_700_too(self):
        """`parents=True` creates intermediate directories, which the umask also decides."""
        target = self.root / "a" / "b" / "state"
        statefile.ensure_dir(target)
        self.assertEqual(_mode(self.root / "a"), STATE_DIR_MODE)
        self.assertEqual(_mode(self.root / "a" / "b"), STATE_DIR_MODE)

    def test_an_existing_directory_is_left_as_its_owner_set_it(self):
        """A mode somebody chose is not this package's to overrule.

        The tempting version corrects every directory it touches, to fix the 0o775 that old
        installs left behind. Measured, that also re-opens a state directory deliberately set
        to 0o500 -- the plugin undoing its user's decision, and silently. Creating a NEW
        directory correctly is ours; an existing one's mode is not.
        """
        target = self.root / "legacy"
        target.mkdir()
        os.chmod(target, 0o775)
        statefile.ensure_dir(target)
        self.assertEqual(_mode(target), 0o775,
                         "an existing directory's mode belongs to whoever set it")

    def test_a_read_only_directory_is_not_made_writable(self):
        """The failure mode that matters: 0o500 is a decision, not damage to repair."""
        target = self.root / "read-only"
        target.mkdir()
        os.chmod(target, 0o500)
        self.addCleanup(os.chmod, target, 0o700)
        statefile.ensure_dir(target)
        self.assertEqual(_mode(target), 0o500,
                         "the plugin must not hand itself write access its user removed")

    def test_it_raises_rather_than_hiding_a_directory_it_could_not_make(self):
        """The callers this replaced each built their own answer on top of that raise.

        `config.save` and `install.write_env_file` propagate it, `jobs._create_cancel_file` turns
        it into its False so a cancel that did not land is not reported as landed, and the hooks
        catch it to stay silent. The write path never sees the raise: `_publish` catches it and
        reports its False, which is the signal `bindings._save` already knows how to turn into its
        own `OSError`. Swallowing the error here would hide the failure from all of them.
        """
        self.assertTrue(statefile.ensure_dir(self.root / "ok"))
        blocker = self.root / "a-file"
        blocker.write_text("")
        with self.assertRaises(OSError):
            statefile.ensure_dir(blocker / "under-a-file")

    def test_the_write_path_still_reports_rather_than_raising(self):
        """`write_json`/`write_text` keep their own contract: a bool, never an exception."""
        blocker = self.root / "another-file"
        blocker.write_text("")
        self.assertFalse(statefile.write_json(blocker / "under" / "x.json", {"a": 1}))
        self.assertFalse(statefile.write_text(blocker / "under" / "x.txt", "1"))


class TestPublishingCreatesTheDirectoryOwnerOnly(unittest.TestCase):
    """The write path creates the directory when it is missing; it must use the same owner."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.old_umask = os.umask(0o002)
        self.addCleanup(os.umask, self.old_umask)

    def test_write_json_into_a_missing_directory_creates_it_700(self):
        target = self.root / "made-by-write" / "daemon.json"
        self.assertTrue(statefile.write_json(target, {"a": 1}))
        self.assertEqual(_mode(target.parent), STATE_DIR_MODE)
        self.assertEqual(_mode(target), 0o600, "the file's own mode is unchanged")

    def test_write_text_into_a_missing_directory_creates_it_700(self):
        target = self.root / "made-by-text" / "counter.count"
        self.assertTrue(statefile.write_text(target, "1"))
        self.assertEqual(_mode(target.parent), STATE_DIR_MODE)


if __name__ == "__main__":
    unittest.main()
