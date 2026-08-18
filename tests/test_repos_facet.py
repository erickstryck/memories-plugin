"""The listing asks the SERVER for the distinct repo names, and falls back when it must.

`_repos_with_chunks` fed both divergence reads by scrolling the ENTIRE chunk collection — every
point, every payload, over the wire — to end up with a handful of names. Qdrant answers exactly
this question server-side (`/facet` with an indexed keyword key, measured against the live
server on 2026-08-18), and the collection already carries the index the facet needs.

WHY THE FALLBACK IS THE POINT OF THIS FILE. `/facet` takes a `limit` and TRUNCATES AT IT IN
SILENCE — measured: a probe collection holding 7 distinct values answered with 3 and no
indication that 4 were missing. A truncated set here is not a slow answer, it is a WRONG one:
`divergent_repos` reports the repos the listing cannot even name, so a name dropped from this
set is a repo that has chunks, cannot be dropped, and is now invisible — the exact failure the
divergence read exists to prevent. So a result that MIGHT be truncated is discarded in favour of
the scroll, which cannot truncate.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.qdrant import QdrantError  # noqa: E402
from core.repos import FACET_LIMIT, RepoIndex  # noqa: E402
from tests.fakes import FakeEmbedder, FakeVectorStore  # noqa: E402

CHUNKS, REG = "repos_c", "repos_r"


class FacetingStore(FakeVectorStore):
    """A store that can answer a facet, refuse it, or truncate it — and counts both calls."""

    def __init__(self, *, values=None, fails=False, truncate_at=None):
        super().__init__()
        self._values = values or []
        self._fails = fails
        self._truncate_at = truncate_at
        self.facet_calls = 0
        self.scroll_calls = 0

    def facet(self, name, key, limit, exact=True):
        self.facet_calls += 1
        if self._fails:
            raise QdrantError("no appropriate index for faceting: `repo`")
        values = self._values[:self._truncate_at] if self._truncate_at else self._values

        return [{"value": v, "count": 1} for v in values][:limit]

    def scroll_all(self, name, filter_=None, with_vector=False):
        self.scroll_calls += 1

        return super().scroll_all(name, filter_, with_vector)


def an_index(store) -> RepoIndex:
    return RepoIndex(store, FakeEmbedder(dim=8), CHUNKS, REG, 8)


class TestTheServerAnswersInsteadOfTheWire(unittest.TestCase):
    def test_the_names_come_from_the_facet_and_the_archive_is_NOT_scrolled(self):
        store = FacetingStore(values=["alpha", "beta"])
        ix = an_index(store)
        self.assertEqual(ix._repos_with_chunks(), {"alpha", "beta"})
        self.assertEqual(store.facet_calls, 1)
        self.assertEqual(store.scroll_calls, 0, "the whole archive was scrolled anyway")

    def test_an_empty_archive_answers_empty_without_scrolling(self):
        store = FacetingStore(values=[])
        self.assertEqual(an_index(store)._repos_with_chunks(), set())
        self.assertEqual(store.scroll_calls, 0)


class TestWhatItDoesWhenTheFacetCannotBeTrusted(unittest.TestCase):
    def test_a_result_AT_the_limit_is_discarded_for_the_scroll(self):
        """The measured behaviour: at the limit, "these are all the values" and "these are the
        first N of more" are the same response. Trusting it would drop a divergent repo out of
        the only report that can name it, so a result that might be truncated is not used."""
        store = FacetingStore(values=[f"r{i}" for i in range(FACET_LIMIT + 5)],
                              truncate_at=FACET_LIMIT)
        ix = an_index(store)
        ix.register_request("alpha")
        ix.add_files("alpha", [_a_file()])
        got = ix._repos_with_chunks()
        self.assertEqual(store.scroll_calls, 1, "a possibly-truncated facet was trusted")
        self.assertEqual(got, {"alpha"}, "the fallback did not answer from the archive")

    def test_a_server_that_cannot_facet_still_gets_a_correct_listing(self):
        """An older Qdrant, or a collection whose index was never created: the listing must
        degrade to the slower read, never to an error the user sees."""
        store = FacetingStore(fails=True)
        ix = an_index(store)
        ix.register_request("alpha")
        ix.add_files("alpha", [_a_file()])
        self.assertEqual(ix._repos_with_chunks(), {"alpha"})
        self.assertEqual(store.scroll_calls, 1)

    def test_a_facet_row_without_a_value_is_ignored_rather_than_stored_as_None(self):
        store = FacetingStore(values=["alpha", "", None])
        self.assertEqual(an_index(store)._repos_with_chunks(), {"alpha"})


class TestTheListingStillReportsBothDivergences(unittest.TestCase):
    """The faceting is an implementation of one private read; what the listing MEANS is
    unchanged, and these are the two answers a truncation would have corrupted."""

    def test_a_repo_with_chunks_and_no_registry_entry_is_still_named(self):
        store = FacetingStore(values=["ghost"])
        out = an_index(store).list_request()
        self.assertEqual(out["divergent"], ["ghost"])

    def test_an_entry_CLAIMING_chunks_over_an_empty_archive_is_still_named(self):
        """`emptied` is not "registered with no chunks" — a fresh registration claims zero and
        is deliberately silent, because a divergence that fires on every registration is one
        people learn to ignore. The reported state is an entry whose recorded count says N over
        an archive holding none, so the fixture has to INDEX first and then lose the chunks."""
        store = FacetingStore(values=["hollow"])
        ix = an_index(store)
        ix.register_request("hollow")
        ix.add_files("hollow", [_a_file()])          # the count is recorded here
        store._values = []                           # and the archive loses them
        self.assertEqual(ix.list_request()["emptied"], ["hollow"])


def _a_file() -> str:
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as fh:
        fh.write("x = 1\n")

    return path


if __name__ == "__main__":
    unittest.main(verbosity=2)
