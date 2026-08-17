import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.repos import RepoError, RepoIndex  # noqa: E402
from tests.fakes import FakeEmbedder, FakeVectorStore  # noqa: E402


def a_file(text: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)

    return path


def an_index_with_two_repos() -> RepoIndex:
    ix = RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), "c", "r", 8)
    ix.register("loud", "Loud", [], "/tmp/loud")
    ix.register("quiet", "Quiet", [], "/tmp/quiet")
    ix.add_files("loud", [a_file("billing invoice charge total\n" * 40)])
    ix.add_files("quiet", [a_file("invoice\n")])

    return ix


class TestScopedSearch(unittest.TestCase):
    def test_a_scoped_search_returns_only_that_repo(self):
        out = an_index_with_two_repos().search("billing invoice", repo="quiet")
        self.assertEqual(out["scope"], "repo")
        self.assertEqual({g["repo"] for g in out["groups"]}, {"quiet"})

    def test_an_unknown_repo_is_an_error_and_not_an_empty_result(self):
        """Empty would read as "this repo has nothing about it", which is a different and
        false statement."""
        with self.assertRaises(RepoError):
            an_index_with_two_repos().search("billing", repo="no-such-repo")

    def test_naming_no_repo_says_what_to_do_instead_of_reporting_a_missing_repo(self):
        """It raises either way — `None` is in no registry — so what this guard is worth is
        the message. "repository None is not indexed" sends the caller looking for a repo
        that was never named; naming `across` tells them the one thing they can act on."""
        with self.assertRaisesRegex(RepoError, "across"):
            an_index_with_two_repos().search("invoice")

    def test_a_hit_carries_the_location_and_not_only_the_text(self):
        out = an_index_with_two_repos().search("invoice", repo="quiet")
        hit = out["groups"][0]["hits"][0]
        self.assertTrue(hit.path)
        self.assertGreaterEqual(hit.start_line, 1)

    def test_a_changed_file_is_reported_stale_instead_of_answered_as_current(self):
        ix = RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), "c", "r", 8)
        ix.register("alpha", "Alpha", [], "/tmp/alpha")
        path = a_file("original invoice content\n")
        ix.add_files("alpha", [path])
        with open(path, "w") as fh:
            fh.write("something else entirely\n")
        hit = ix.search("invoice", repo="alpha")["groups"][0]["hits"][0]
        self.assertTrue(hit.stale)


class TestTheTraversal(unittest.TestCase):
    def test_across_returns_the_quiet_repo_too(self):
        """The whole point of the feature. A repo with one mention must not be shadowed by a
        repo with forty."""
        out = an_index_with_two_repos().search("billing invoice charge total", across=True)
        self.assertEqual(out["scope"], "across")
        self.assertEqual({g["repo"] for g in out["groups"]}, {"loud", "quiet"})

    def test_it_asks_for_as_many_groups_as_the_registry_knows(self):
        """The registry is what makes this number a fact instead of a guess."""
        ix = an_index_with_two_repos()
        seen = {}
        real = ix.q.search_groups

        def spy(name, vector, group_by, limit, group_size, **kw):
            seen["limit"] = limit

            return real(name, vector, group_by, limit, group_size, **kw)

        ix.q.search_groups = spy
        ix.search("invoice", across=True)
        self.assertEqual(seen["limit"], len(ix.list_repos()))

    def test_across_with_no_repos_registered_is_an_error(self):
        ix = RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), "c", "r", 8)
        with self.assertRaises(RepoError):
            ix.search("anything", across=True)


class TestFailureDoesNotOpen(unittest.TestCase):
    def test_an_unreachable_store_raises_instead_of_returning_nothing(self):
        """The inverse of the big-file guard, on purpose: there, blocking on a doubtful
        number kept the user from reading, so failure had to allow. Here an empty result is
        indistinguishable from "there is nothing", so failure must SAY so."""
        ix = an_index_with_two_repos()

        def boom(*a, **kw):
            raise OSError("connection refused")

        ix.q.search_groups = boom
        with self.assertRaises(RepoError):
            ix.search("invoice", repo="quiet")


if __name__ == "__main__":
    unittest.main(verbosity=2)
