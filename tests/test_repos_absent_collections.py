"""A repository archive that does not exist yet is EMPTY, not an error.

Both repo collections are created on first use — `qctx setup` says so, and that is the correct
design: nothing should be created by a machine that has not been asked to index anything. The
consequence is that on a fresh install, before any `repos register`, the collections are absent.

MEASURED on 2026-08-18, on a real installation: `qctx repos list` answered

    error: HTTP 404 ... {"status":{"error":"Not found: Collection `memories_repos` doesn't exist!"}}

and the `repos_list` tool returned the same string as an error object. That is the FIRST thing a
new user runs after installing, and it reads as a broken plugin rather than an empty one.

WHY IT IS DECIDED BY STATUS AND NOT BY MESSAGE: `core.qdrant._is_absent` already carries that rule
and the reason for it — this Qdrant sits behind a proxy that echoes upstream statuses into its own
error bodies, so a 502 whose body mentions 404 must NOT read as absence. Absence is a 404 on OUR
request, nothing else.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.qdrant import QdrantError  # noqa: E402
from core.repos import RepoIndex  # noqa: E402
from tests.fakes import FakeEmbedder, FakeVectorStore  # noqa: E402

CHUNKS, REG = "repos_c", "repos_r"


def absent(name: str) -> QdrantError:
    exc = QdrantError(f"HTTP 404 on POST /collections/{name}/points/scroll: "
                      '{"status":{"error":"Not found: Collection doesn\'t exist!"}}')
    exc.status = 404

    return exc


def unreachable() -> QdrantError:
    """A real failure, which must NOT be read as emptiness."""
    exc = QdrantError("HTTP 502 on POST /collections/x/points/scroll: upstream error")
    exc.status = 502

    return exc


class MissingCollections(FakeVectorStore):
    """A store where the repo collections were never created — a fresh install."""

    def __init__(self, error_factory=None, missing=(CHUNKS, REG)):
        super().__init__()
        self._error = error_factory or (lambda name: absent(name))
        self._missing = set(missing)

    def facet(self, name, key, limit, exact=True):
        if name in self._missing:
            raise self._error(name)

        return []

    def scroll_all(self, name, filter_=None, with_vector=False):
        if name in self._missing:
            raise self._error(name)

        return super().scroll_all(name, filter_, with_vector)


def an_index(store) -> RepoIndex:
    return RepoIndex(store, FakeEmbedder(dim=8), CHUNKS, REG, 8)


class TestAFreshInstallListsNothingInsteadOfFailing(unittest.TestCase):
    def test_list_request_answers_empty_when_NEITHER_collection_exists(self):
        out = an_index(MissingCollections()).list_request()
        self.assertEqual(out, {"repos": [], "divergent": [], "emptied": []})

    def test_the_registry_alone_being_absent_is_also_empty(self):
        out = an_index(MissingCollections(missing={REG})).list_request()
        self.assertEqual(out["repos"], [])

    def test_the_chunk_archive_alone_being_absent_is_also_empty(self):
        """This is the one the user hit: the registry may exist while the archive does not, and
        the divergence reads must answer 'nothing' rather than raise."""
        out = an_index(MissingCollections(missing={CHUNKS})).list_request()
        self.assertEqual(out["divergent"], [])
        self.assertEqual(out["emptied"], [])

    def test_SEARCH_refuses_instead_of_answering_empty_and_that_is_the_OPPOSITE_choice(self):
        """Listing and searching answer different questions, so emptiness means different things.

        `list` asks WHAT EXISTS. Nothing exists, so an empty list is a true answer.

        `search` asks IS THERE ANYTHING ABOUT X. An empty result there would be read as "no,
        nothing about X" — absence concluded from a state that never held an answer. So it
        raises, naming the state, and `search()`'s own docstring says why: a search that cannot
        reach the archive and returns [] is indistinguishable from a real negative.

        This test exists because the first version of it asserted the opposite, and the code was
        right."""
        with self.assertRaises(Exception) as caught:
            an_index(MissingCollections()).search_request("anything", across=True)
        self.assertIn("no repository is indexed", str(caught.exception))


class TestARealFailureIsStillARealFailure(unittest.TestCase):
    """The half that makes the fix safe. If any unreachable archive read as empty, `repos list`
    would print 'no repositories' during an outage — a claim of absence produced by a failure,
    which is the one thing this plugin exists to never do."""

    def test_a_502_is_NOT_read_as_emptiness(self):
        store = MissingCollections(error_factory=lambda name: unreachable())
        with self.assertRaises(Exception) as caught:
            an_index(store).list_request()
        self.assertIn("502", str(caught.exception))

    def test_a_502_on_the_CHUNK_ARCHIVE_ALONE_is_not_read_as_emptiness(self):
        """The one that isolates the archive's own guard.

        With BOTH collections failing, the test above passes through the REGISTRY's guard and
        proves nothing about the archive's — measured: replacing the archive's `_is_absent`
        check with a bare `except QdrantError: return set()` left the suite green. A guard needs
        a test that reaches it without passing through its sibling."""
        store = MissingCollections(error_factory=lambda name: unreachable(), missing={CHUNKS})
        with self.assertRaises(Exception) as caught:
            an_index(store).list_request()
        self.assertIn("502", str(caught.exception))

    def test_a_502_on_SEARCH_is_not_read_as_emptiness(self):
        store = MissingCollections(error_factory=lambda name: unreachable())
        with self.assertRaises(Exception):
            an_index(store).search_request("anything", across=True)


class TestTheEmptyLISTINGSaysSoOutLoud(unittest.TestCase):
    """An empty screen is not an answer, and this is the first command a new user runs.

    The search path already carries this rule — it prints a sentence rather than rendering
    nothing — and the listing did not: after the 404 was fixed it printed absolutely nothing,
    which reads as a broken command rather than an empty archive."""

    def test_the_cli_prints_a_sentence_when_there_is_nothing_to_list(self):
        import subprocess
        out = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "cli", "qctx.py"), "repos", "list"],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "QCTX_REPOS_COLLECTION": "qctx_absent_probe_chunks",
                 "QCTX_REPOS_REGISTRY_COLLECTION": "qctx_absent_probe_registry"})
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        self.assertIn("no repository is indexed yet", out.stdout,
                      f"an empty listing printed nothing: {out.stdout!r}")
        self.assertIn("repos register", out.stdout, "it does not say what to do next")


if __name__ == "__main__":
    unittest.main(verbosity=2)
