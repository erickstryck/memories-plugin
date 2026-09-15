"""Files that can never be indexed, remembered so they are not tried forever.

WHAT THIS PREVENTS, measured on 2026-09-14: the watcher re-queued the same 22 files every ~40 s
indefinitely, because a file that fails to index is a file the archive has no chunk for, and the
next poll reports it as missing all over again. The resulting load on the shared embedding
endpoint pushed automatic recall from 0.04 s to 1.98 s against a 2.00 s ceiling, so the user saw
"[automatic recall — UNAVAILABLE]" with no visible connection to indexing.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import quarantine  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


def a_file(text: str = "content\n") -> str:
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, "w") as fh:
        fh.write(text)

    return path


class TestHoldingAndReleasing(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_a_recorded_path_is_held(self):
        path = a_file()
        quarantine.record("alpha", path, "nothing indexable")
        self.assertIn(path, quarantine.held("alpha"))

    def test_an_unknown_repo_holds_nothing(self):
        self.assertEqual(quarantine.held("never-seen"), set())

    def test_the_reason_is_kept_verbatim_for_the_user_to_read(self):
        path = a_file()
        quarantine.record("alpha", path, "HTTP 500: input (83086 tokens) is too large")
        self.assertIn("83086", quarantine.load("alpha")[path]["reason"])

    def test_a_repo_does_not_see_another_repos_quarantine(self):
        path = a_file()
        quarantine.record("alpha", path, "nothing indexable")
        self.assertEqual(quarantine.held("beta"), set())

    def test_clear_releases_a_path(self):
        path = a_file()
        quarantine.record("alpha", path, "nothing indexable")
        quarantine.clear("alpha", [path])
        self.assertEqual(quarantine.held("alpha"), set())

    def test_clearing_a_path_that_was_never_held_is_not_an_error(self):
        quarantine.clear("alpha", ["/nonexistent/never-recorded.py"])
        self.assertEqual(quarantine.held("alpha"), set())


class TestChangedContentIsRetried(unittest.TestCase):
    """The quarantine describes a CONTENT, not a path. This is what makes a 0-byte file that
    later gains content come back on its own, with no manual command — which is the state 20 of
    the 22 looping files were in."""

    def setUp(self):
        a_state_dir()

    def test_a_file_whose_size_changed_is_no_longer_held(self):
        path = a_file("")
        quarantine.record("alpha", path, "nothing indexable")
        with open(path, "w") as fh:
            fh.write("it has content now\n")
        self.assertNotIn(path, quarantine.held("alpha"))

    def test_a_file_that_did_not_change_stays_held(self):
        path = a_file("same content\n")
        quarantine.record("alpha", path, "some failure")
        self.assertIn(path, quarantine.held("alpha"))

    def test_a_file_that_vanished_is_not_held(self):
        path = a_file()
        quarantine.record("alpha", path, "some failure")
        os.unlink(path)
        self.assertNotIn(path, quarantine.held("alpha"),
                         "a path that is gone must not be held by a stale entry")

    def test_a_released_entry_is_dropped_from_disk(self):
        """A stale entry that no longer describes anything is noise in what the user reads."""
        path = a_file("")
        quarantine.record("alpha", path, "nothing indexable")
        with open(path, "w") as fh:
            fh.write("content now\n")
        quarantine.held("alpha")
        self.assertNotIn(path, quarantine.load("alpha"))

    def test_a_path_that_cannot_be_stated_is_not_recorded(self):
        """With no mtime/size there is nothing to compare later, so the entry could never be
        released: a transient error would become a permanent exclusion."""
        quarantine.record("alpha", "/nonexistent/vanished.py", "some failure")
        self.assertEqual(quarantine.load("alpha"), {})


class TestFailureDegradesToHoldingNothing(unittest.TestCase):
    """The rule this project already follows: a tolerated failure must never become a LIE.
    Holding nothing means re-indexing something we might have skipped — today's behaviour.
    Claiming everything is indexed would be the lie."""

    def setUp(self):
        a_state_dir()

    def test_corrupt_json_reads_as_nothing_held(self):
        quarantine.dir().mkdir(parents=True, exist_ok=True)
        (quarantine.dir() / "alpha.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(quarantine.held("alpha"), set())

    def test_a_json_document_of_the_wrong_SHAPE_reads_as_nothing_held(self):
        quarantine.dir().mkdir(parents=True, exist_ok=True)
        (quarantine.dir() / "alpha.json").write_text("[1, 2, 3]", encoding="utf-8")
        self.assertEqual(quarantine.load("alpha"), {})

    def test_a_malformed_ENTRY_does_not_bring_the_watcher_down(self):
        """`held` runs inside the watcher, and `daemon.run` swallows a raising watcher — so
        one bad value used to stop indexing for EVERY repository on the machine, silently and
        permanently, while `repos status` raised on the same entry. Validating only the top
        level was not enough."""
        path = a_file()
        for bad in (None, "a string", [1, 2], 42):
            quarantine.dir().mkdir(parents=True, exist_ok=True)
            (quarantine.dir() / "alpha.json").write_text(json.dumps({path: bad}),
                                                         encoding="utf-8")
            self.assertEqual(quarantine.held("alpha"), set(), f"value {bad!r} was not dropped")

    def test_a_good_entry_survives_beside_a_malformed_one(self):
        good, bad = a_file(), a_file()
        quarantine.record("alpha", good, "nothing indexable")
        entry = quarantine.load("alpha")
        entry[bad] = "not a dict at all"
        quarantine._write("alpha", entry)
        self.assertEqual(quarantine.held("alpha"), {good},
                         "one bad entry must not discard the good ones")

    def test_an_unwritable_state_dir_does_not_raise(self):
        with mock.patch("core.quarantine._write", return_value=False):
            quarantine.record("alpha", a_file(), "some failure")   # must not raise

    def test_an_oserror_while_reading_does_not_raise(self):
        with mock.patch("pathlib.Path.read_text", side_effect=OSError("boom")):
            self.assertEqual(quarantine.held("alpha"), set())


class TestItDependsOnNothingHeavy(unittest.TestCase):
    """Dependency direction: `indexer` -> `quarantine`, never the other way. Importing `repos`
    or `core` here would put a Qdrant client behind a module the watcher calls every cycle."""

    def test_it_does_not_import_the_archive_layer(self):
        import core.quarantine as module

        source = open(module.__file__, encoding="utf-8").read()
        self.assertNotIn("import repos", source)
        self.assertNotIn("from .repos", source)
        self.assertNotIn("import core", source)


class TestForgettingOnPurpose(unittest.TestCase):
    """The automatic release covers a file whose CONTENT changed. It cannot cover a file the
    user knows is fine but the archive refused for a reason that has since gone away — a
    server limit that was raised, a model that was swapped. `clear` already served the
    indexer; `forget` is the same operation offered deliberately, to a person."""

    def setUp(self):
        a_state_dir()

    def test_forgetting_one_path_releases_only_that_one(self):
        kept, released = a_file(), a_file()
        quarantine.record("alpha", kept, "nothing indexable")
        quarantine.record("alpha", released, "HTTP 500: too large")
        self.assertEqual(quarantine.forget("alpha", [released]), 1)
        self.assertEqual(quarantine.held("alpha"), {kept})

    def test_forgetting_the_whole_repo_releases_everything(self):
        for _ in range(3):
            quarantine.record("alpha", a_file(), "nothing indexable")
        self.assertEqual(quarantine.forget("alpha"), 3)
        self.assertEqual(quarantine.load("alpha"), {})

    def test_it_reports_how_many_it_released_so_the_caller_can_say_so(self):
        """The count is the answer to "did that do anything?" — a command that prints
        "released" after releasing nothing is the kind of lie this project refuses."""
        self.assertEqual(quarantine.forget("never-seen"), 0)
        self.assertEqual(quarantine.forget("alpha", ["/not/held.py"]), 0)

    def test_forgetting_a_repo_leaves_another_alone(self):
        mine, theirs = a_file(), a_file()
        quarantine.record("alpha", mine, "r")
        quarantine.record("beta", theirs, "r")
        quarantine.forget("alpha")
        self.assertEqual(quarantine.held("beta"), {theirs})

    def test_a_forgotten_file_can_be_held_again_if_it_fails_again(self):
        """Forgetting is not an exemption: it drops the record, and the next attempt decides
        afresh. A permanent allow-list would be a second policy nobody asked for."""
        path = a_file()
        quarantine.record("alpha", path, "nothing indexable")
        quarantine.forget("alpha", [path])
        quarantine.record("alpha", path, "nothing indexable")
        self.assertIn(path, quarantine.held("alpha"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
