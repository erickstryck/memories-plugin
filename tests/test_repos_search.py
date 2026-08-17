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


class TestAnEmptyAnswerNEVERAffirmsAbsence(unittest.TestCase):
    """An answer with no groups has to SAY "nothing above the cut", in words.

    The spec calls this the cardinal error of the feature: the grouping is best-effort over
    what the search reached, never an exhaustive sweep, so `[]` means "nothing came back
    above the cut" and NOT "no repository mentions this". A model calling
    `repos_search(across=true)` — whose description promises every indexed repository — reads
    a bare empty list as proof of absence, which is exactly the conclusion the recall hook
    exists to refuse to let it draw.

    It lives in the core so both hosts inherit the same sentence: rendering it twice is
    rendering it two ways, and this is the sentence that must not vary.
    """

    def _registered_but_empty(self) -> RepoIndex:
        ix = RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), "c", "r", 8)
        ix.register("alpha", "Alpha", [], "/tmp/alpha")
        ix.register("beta", "Beta", [], "/tmp/beta")

        return ix

    def test_an_across_search_with_no_groups_carries_the_sentence(self):
        out = self._registered_but_empty().search("invoice", across=True)
        self.assertEqual(out["groups"], [])
        self.assertTrue(out["note"], "an empty result answered with silence")

    def test_the_sentence_says_above_the_cut_and_never_that_nothing_exists(self):
        note = self._registered_but_empty().search("invoice", across=True)["note"].lower()
        self.assertIn("above", note, "it has to name the CUT, not the absence")
        self.assertIn("reach", note, "it has to say the search is best-effort over what it "
                                     "reached")
        self.assertIn("not a statement that no repository mentions this", note,
                      "the one conclusion it exists to forbid has to be named and refused, "
                      "not merely left unsaid")

    def test_a_scoped_search_with_no_groups_carries_it_too(self):
        """Same harm, one repository down: "alpha has nothing about x" is a claim the top-K
        cannot support either."""
        out = self._registered_but_empty().search("invoice", repo="alpha")
        self.assertEqual(out["groups"], [])
        self.assertTrue(out["note"])
        self.assertIn("alpha", out["note"], "it has to name the repository it searched")

    def test_an_answer_WITH_hits_carries_no_sentence(self):
        """A caveat printed on every successful search is a caveat nobody reads."""
        out = an_index_with_two_repos().search("billing invoice charge total", across=True)
        self.assertTrue(out["groups"])
        self.assertIsNone(out["note"])

    def test_the_key_is_always_present_so_a_consumer_can_read_it_unconditionally(self):
        for out in (an_index_with_two_repos().search("invoice", across=True),
                    self._registered_but_empty().search("invoice", across=True)):
            self.assertIn("note", out)


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

    def test_a_response_whose_shape_drifted_is_an_error_and_not_a_traceback(self):
        """The parsing is inside the guard for the same reason the call is. A caller can act
        on a RepoError; an AttributeError raised from inside a comprehension reads as a bug
        in the search itself, and hides that the archive answered with something unexpected.
        """
        shapes = {
            "a group that is not a dict": [None],
            "a hit that is not a dict": [{"id": "quiet", "hits": [None]}],
            "a start_line that is not a number": [
                {"id": "quiet", "hits": [{"score": 1.0, "payload": {
                    "repo": "quiet",
                    "metadata": {"path": "/nowhere", "start_line": "seven"}}}]}],
        }
        for shape, raw in shapes.items():
            with self.subTest(shape=shape):
                ix = an_index_with_two_repos()
                ix.q.search_groups = lambda *a, _raw=raw, **kw: _raw
                with self.assertRaises(RepoError):
                    ix.search("invoice", repo="quiet")


class TestTheDocumentArchivesAreUntouched(unittest.TestCase):
    def test_the_repos_collection_is_not_one_of_the_docs_scopes(self):
        """A regression here floods every existing `docs search --scope all` with tens of
        thousands of code chunks — shipped behaviour, silently degraded."""
        from core.docs import SCOPES
        self.assertEqual(set(SCOPES), {"all", "tmp", "library"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
