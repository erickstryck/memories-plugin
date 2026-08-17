import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import bindings  # noqa: E402
from core.repos import RepoError, RepoIndex  # noqa: E402
from tests.fakes import FakeEmbedder, FakeVectorStore  # noqa: E402


def a_file(text: str = "content = 1\n") -> str:
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)

    return path


def a_populated_index() -> RepoIndex:
    os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()
    ix = RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), "c", "r", 8)
    for name in ("alpha", "beta"):
        ix.register(name, name.title(), [], f"/tmp/{name}")
        ix.add_files(name, [a_file()])
        bindings.bind(f"/tmp/{name}", name)

    return ix


class TestDropping(unittest.TestCase):
    def test_it_removes_the_chunks_the_entry_and_the_bindings(self):
        ix = a_populated_index()
        out = ix.drop_repo("alpha")
        self.assertEqual(out["unbound"], ["/tmp/alpha"])
        self.assertIsNone(ix.get_repo("alpha"))
        self.assertEqual({(p["payload"]["repo"]) for p in ix.q.scroll_all("c")}, {"beta"})
        self.assertEqual(bindings.get("/tmp/alpha"), None)

    def test_it_leaves_the_other_repos_alone(self):
        ix = a_populated_index()
        ix.drop_repo("alpha")
        self.assertEqual([r["repo"] for r in ix.list_repos()], ["beta"])
        self.assertEqual(bindings.get("/tmp/beta"), "beta")

    def test_dropping_an_unknown_repo_is_an_error(self):
        with self.assertRaises(RepoError):
            a_populated_index().drop_repo("no-such-repo")

    def test_the_chunks_go_FIRST_so_a_failure_leaves_a_visible_remainder(self):
        """Order is load-bearing. Registry first with the chunks failing leaves chunks that no
        listing can reach and that still compete in an across search — unreachable garbage.
        Chunks first with the registry failing leaves an entry pointing at zero chunks:
        visible, and a second run finishes the job."""
        ix = a_populated_index()
        calls = []
        real_delete = ix.q.delete_by_filter

        def spy_delete(name, filter_):
            calls.append(("delete", name))

            return real_delete(name, filter_)

        real_points = ix.q.delete_points

        def spy_points(name, ids):
            calls.append(("delete_points", name))

            return real_points(name, ids)

        ix.q.delete_by_filter, ix.q.delete_points = spy_delete, spy_points
        ix.drop_repo("alpha")
        touched = [name for _, name in calls]
        self.assertEqual(touched.index("c"), 0, f"chunks were not touched first: {calls}")

    def test_a_failure_deleting_chunks_does_NOT_remove_the_entry(self):
        """Otherwise the second run has nothing to finish from."""
        ix = a_populated_index()

        def boom(*a, **kw):
            raise OSError("connection refused")

        ix.q.delete_by_filter = boom
        with self.assertRaises(RepoError):
            ix.drop_repo("alpha")
        self.assertIsNotNone(ix.get_repo("alpha"))

    def test_a_failure_deleting_chunks_does_NOT_unbind_the_checkout(self):
        """Bindings go LAST, and only once the archive agrees — the third half of the order,
        which nothing else here holds: with the archive intact and the binding already gone,
        the checkout is orphaned from a repo that is still fully indexed, and the next
        detection offers to create a SECOND entry for content that is already there.
        """
        ix = a_populated_index()

        def boom(*a, **kw):
            raise OSError("connection refused")

        ix.q.delete_by_filter = boom
        with self.assertRaises(RepoError):
            ix.drop_repo("alpha")
        self.assertEqual(bindings.get("/tmp/alpha"), "alpha")


class TestDivergence(unittest.TestCase):
    def test_chunks_without_a_registry_entry_are_reported(self):
        """The price of honesty for having two sources of truth about which repos exist. The
        registry is authoritative; this is how the copy is caught diverging."""
        ix = a_populated_index()
        ix.add_files("ghost", [a_file()])
        self.assertEqual(ix.divergent_repos(), ["ghost"])

    def test_a_healthy_archive_reports_no_divergence(self):
        self.assertEqual(a_populated_index().divergent_repos(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
