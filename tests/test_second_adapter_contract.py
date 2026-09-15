"""A second store adapter, written only against the port, driving the rule files.

WHY THIS EXISTS. `ports.VectorStore` says in its own words that swapping Qdrant "is writing an
adapter — no rule file changes", and `runtime_checkable` cannot check that: it compares method
NAMES, so an adapter gets a yes from `isinstance` and a failure from production. That is worse
than having no port, because it tells the next adapter's author they are finished when they are
not.

The class below is deliberately NOT a subclass of the fake or of `Qdrant`. It raises its own
error type and carries its own status attribute, which is the whole point: a rule file that
decides "the collection is absent" by catching `QdrantError` works for the vendor it was
written against and breaks for every other one, turning a fresh install into "the repository
registry could not be read".
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ports  # noqa: E402
from core.repos import RepoIndex  # noqa: E402
from tests.fakes import FakeEmbedder, FakeVectorStore  # noqa: E402

CHUNKS, REG = "repos_c", "repos_r"


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


class OtherVendorError(Exception):
    """What a different store's adapter would raise. Carries `status`, like every HTTP
    client does, but shares no ancestry with `QdrantError`."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class OtherVendorStore(FakeVectorStore):
    """Conformant to the port, and absent collections reported the way ITS vendor reports
    them. Everything else is inherited, because what is under test is the error contract and
    not the storage."""

    def _absent(self, name: str) -> bool:
        return name not in self.collections

    def scroll_all(self, name, filter_=None, with_vector=False, payload_fields=None):
        if self._absent(name):
            raise OtherVendorError(f"collection {name!r} does not exist", status=404)

        return super().scroll_all(name, filter_, with_vector, payload_fields)

    def facet(self, name, key, limit, exact=True):
        if self._absent(name):
            raise OtherVendorError(f"collection {name!r} does not exist", status=404)

        return super().facet(name, key, limit, exact=exact)


def an_index(store) -> RepoIndex:
    return RepoIndex(store, FakeEmbedder(dim=8), CHUNKS, REG, 8)


class TestASecondAdapterDrivesTheRuleFiles(unittest.TestCase):
    def setUp(self):
        a_state_dir()
        self.store = OtherVendorStore()
        self.ix = an_index(self.store)

    def test_it_satisfies_the_port(self):
        """The precondition, and the reason the rest is a trap: the port says yes."""
        self.assertIsInstance(self.store, ports.VectorStore)

    def test_a_fresh_install_lists_nothing_instead_of_failing(self):
        """Both collections are created on first use, so before any `repos register` they are
        absent — the state of EVERY fresh install. A rule file that decides absence by the
        vendor's exception class reports this as "the repository registry could not be read"."""
        self.assertEqual(self.ix.list_repos(), [])

    def test_a_fresh_install_counts_no_chunks_instead_of_raising(self):
        self.assertEqual(self.ix._chunks_per_repo(), {})

    def test_a_REAL_failure_is_still_a_failure(self):
        """The other direction, and the one that makes the tolerance safe: absence is a 404,
        not "any error from the store". A 502 must not empty the listing.

        It arrives UNWRAPPED, and that is the honest outcome: `RepoError` is this project's
        word for "the registry could not be read", and a rule file cannot claim that about an
        error class it has never heard of. What it must never do is swallow it."""
        class Broken(OtherVendorStore):
            def scroll_all(self, name, filter_=None, with_vector=False, payload_fields=None):
                raise OtherVendorError("bad gateway", status=502)

        with self.assertRaises(OtherVendorError):
            an_index(Broken()).list_repos()

    def test_a_502_that_MENTIONS_404_does_not_empty_the_listing(self):
        """The measured proxy case, now through a second vendor: a gateway echoing the
        upstream status into its body must not read as absence. Deciding by text is what this
        project already had to fix once."""
        class Proxy(OtherVendorStore):
            def scroll_all(self, name, filter_=None, with_vector=False, payload_fields=None):
                raise OtherVendorError("upstream returned HTTP 404 while reloading", status=502)

        with self.assertRaises(Exception):
            an_index(Proxy()).list_repos()


if __name__ == "__main__":
    unittest.main(verbosity=2)
