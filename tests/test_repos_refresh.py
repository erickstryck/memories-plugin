"""Reindexing what changed on disk, for a repository archive.

`docs` has had `refresh` since the library existed, and for the same reason: an archive that
never expires holds chunks of a file as it was, so a file edited later returns text that no
longer exists there. Search already SAYS so — every hit carries a `stale` reason — and that
warning was the whole repair story for repositories. This is the other half.

WHAT IT IS NOT: a watcher. Nothing here runs by itself. The trigger is a git `post-commit` hook
the user installs deliberately (see `TestTheHookScript`), and this command is what that hook — or
the user — calls.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.repos import RepoIndex  # noqa: E402
from tests.fakes import FakeEmbedder, FakeVectorStore  # noqa: E402

CHUNKS, REG = "repos_c", "repos_r"


def an_index(*declared: str) -> RepoIndex:
    ix = RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), CHUNKS, REG, 8)
    for name in declared:
        ix.register_request(name)

    return ix


def a_file(text: str, suffix: str = ".py") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)

    return path


def rewrite(path: str, text: str) -> None:
    """Changes the file's CONTENT. The digest is what `source_changed` trusts, so a rewrite is
    detected even when mtime and size happen to match."""
    with open(path, "w") as fh:
        fh.write(text)


class TestItReindexesOnlyWhatCHANGED(unittest.TestCase):
    def test_an_untouched_file_is_reported_ok_and_NOT_reindexed(self):
        """Re-embedding an unchanged file is pure network cost for an identical result, and the
        hook that calls this runs on every commit."""
        ix = an_index("alpha")
        path = a_file("x = 1\n")
        ix.add_files("alpha", [path])
        before = len(ix.q.collections[CHUNKS]["points"])
        report = ix.refresh("alpha")
        self.assertEqual([r["action"] for r in report], ["ok"])
        self.assertEqual(len(ix.q.collections[CHUNKS]["points"]), before)

    def test_a_changed_file_is_reindexed(self):
        ix = an_index("alpha")
        path = a_file("x = 1\n")
        ix.add_files("alpha", [path])
        rewrite(path, "x = 1\ny = 2\nz = 3\n")
        report = ix.refresh("alpha")
        self.assertEqual([r["action"] for r in report], ["reindexed"])
        self.assertEqual(report[0]["path"], path)
        self.assertGreater(report[0]["chunks"], 0)

    def test_the_reindex_REPLACES_the_old_chunks_instead_of_adding_to_them(self):
        """The point of the repair. Two copies of one file would double its weight in every
        search and answer with text from both versions."""
        ix = an_index("alpha")
        path = a_file("x = 1\n")
        ix.add_files("alpha", [path])
        rewrite(path, "y = 2\n")
        ix.refresh("alpha")
        # `.strip()` because the chunker drops the trailing newline — an assertion that
        # depends on it is testing the chunker, not the refresh.
        texts = [p["payload"]["document"].strip()
                 for p in ix.q.collections[CHUNKS]["points"].values()]
        self.assertIn("y = 2", texts)
        self.assertNotIn("x = 1", texts, "the old version survived the refresh")

    def test_a_file_that_is_GONE_is_reported_and_NOT_deleted(self):
        """Deletion in this plugin is explicit and permanent — `repos drop` demands --yes — so a
        refresh does not quietly remove an archive. The hits stay, and they stay MARKED, which is
        the visible state a user can act on; silently dropping them would be a deletion nobody
        asked for."""
        ix = an_index("alpha")
        path = a_file("x = 1\n")
        ix.add_files("alpha", [path])
        before = len(ix.q.collections[CHUNKS]["points"])
        os.unlink(path)
        report = ix.refresh("alpha")
        self.assertEqual([r["action"] for r in report], ["missing"])
        self.assertEqual(len(ix.q.collections[CHUNKS]["points"]), before,
                         "refresh deleted chunks for a file that vanished")

    def test_one_unreadable_file_does_not_abort_the_others(self):
        """The same rule `add_files` follows: a batch stops being usable the moment one bad
        member can end it."""
        ix = an_index("alpha")
        good, bad = a_file("x = 1\n"), a_file("y = 2\n")
        ix.add_files("alpha", [good, bad])
        rewrite(good, "x = 11\n")
        os.unlink(bad)
        actions = {r["path"]: r["action"] for r in ix.refresh("alpha")}
        self.assertEqual(actions[good], "reindexed")
        self.assertEqual(actions[bad], "missing")


class TestItStaysInsideITSOwnRepository(unittest.TestCase):
    def test_a_changed_file_of_ANOTHER_repo_is_left_alone(self):
        """The archive is one collection keyed by `repo`, so a refresh that forgot the filter
        would reindex every repository on the machine — and charge the user for it."""
        ix = an_index("alpha", "beta")
        mine, theirs = a_file("x = 1\n"), a_file("y = 2\n")
        ix.add_files("alpha", [mine])
        ix.add_files("beta", [theirs])
        rewrite(mine, "x = 11\n")
        rewrite(theirs, "y = 22\n")
        report = ix.refresh("alpha")
        self.assertEqual([r["path"] for r in report], [mine])
        texts = [p["payload"]["document"].strip()
                 for p in ix.q.collections[CHUNKS]["points"].values()]
        self.assertIn("y = 2", texts, "beta was reindexed by alpha's refresh")

    def test_refreshing_an_unregistered_repository_is_refused(self):
        from core.repos import RepoError
        with self.assertRaises(RepoError):
            an_index().refresh("never-declared")

    def test_a_repository_with_nothing_indexed_reports_nothing(self):
        self.assertEqual(an_index("alpha").refresh("alpha"), [])


class TestEachPathIsJudgedONCE(unittest.TestCase):
    def test_a_file_of_many_chunks_is_reindexed_once_not_once_per_chunk(self):
        """A file is stored as N chunks that all carry the same source metadata. Judging each
        chunk would re-embed the file N times and report it N times."""
        ix = an_index("alpha")
        # 900 lines is ~13 KB, which the chunker splits into six; 400 lines fits in ONE and
        # the first version of this fixture measured nothing. Verified below rather than
        # assumed, because the threshold is the chunker's business and can move.
        path = a_file("\n".join(f"line_{i} = {i}" for i in range(900)) + "\n")
        ix.add_files("alpha", [path])
        self.assertGreater(len(ix.q.collections[CHUNKS]["points"]), 1, "fixture is not multi-chunk")
        rewrite(path, "small = 1\n")
        report = ix.refresh("alpha")
        self.assertEqual(len(report), 1, f"the file was judged {len(report)} times")


if __name__ == "__main__":
    unittest.main(verbosity=2)
