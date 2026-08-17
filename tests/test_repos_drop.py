import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import bindings  # noqa: E402
from core.repos import RepoError, RepoIndex  # noqa: E402
from tests.fakes import (FakeEmbedder, FakeVectorStore, make_divergent,  # noqa: E402
                         make_emptied)


def a_file(text: str = "content = 1\n") -> str:
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)

    return path


def _raise(error):
    """A stand-in that fails with exactly `error`, whatever it is called with."""
    def raising(*a, **kw):
        raise error

    return raising


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

    def test_the_REAL_store_error_still_says_the_entry_was_kept(self):
        """The message the chunks-first order exists to produce, against the error a real
        store actually raises.

        The sibling test injects `OSError`, which is NOT a `CoreError` — so it cannot see
        the edit the ledger warned about: widening `except RepoError:` to `except CoreError:`
        would re-raise a `QdrantError` bare and this sentence would vanish, while every test
        stayed green. A real Qdrant raises `QdrantError`, and `QdrantError` IS a `CoreError`.
        """
        from core.qdrant import QdrantError

        ix = a_populated_index()

        def boom(*a, **kw):
            raise QdrantError("connection refused")

        ix.q.delete_by_filter = boom
        with self.assertRaises(RepoError) as caught:
            ix.drop_repo("alpha")
        self.assertIn("kept for a second attempt", str(caught.exception),
                      "the message that tells the user a rerun finishes the job was lost")
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

    def test_a_failure_clearing_the_bindings_says_the_archive_WAS_deleted(self):
        """Still a failure — three things were asked for and two were done — but an accurate
        one, and the accuracy is the point. `bindings._save` writes a file, so a read-only
        state dir arrives as a bare OSError that reads as "the deletion failed" when the
        deletion succeeded; the re-run then answers "is not indexed" and the whole thing looks
        permanently broken while nothing is. The message must not be the chunks one either:
        those two failures leave OPPOSITE remainders and call for opposite reactions.
        """
        ix = a_populated_index()
        self.addCleanup(setattr, bindings, "forget_repo", bindings.forget_repo)
        bindings.forget_repo = _raise(OSError("read-only file system"))
        with self.assertRaises(RepoError) as caught:
            ix.drop_repo("alpha")
        said = str(caught.exception).lower()
        self.assertIn("was deleted", said)
        self.assertIn("bindings", said)
        self.assertIn("read-only file system", said)      # the cause survives, never swallowed
        self.assertNotIn("chunks", said)                  # not the other failure's story
        # And it points at the rerun, which the test below proves is a real path and not a hope.
        self.assertIn("again clears the remainder", said)
        # And the message told the truth: the archive really is gone.
        self.assertIsNone(ix.get_repo("alpha"))
        self.assertEqual({p["payload"]["repo"] for p in ix.q.scroll_all("c")}, {"beta"})


class TestFinishingAHalfDoneDrop(unittest.TestCase):
    """The rerun promise, for the ONE step that used to be unable to keep it.

    Reviewed 2026-08-17: after a step-1 or step-2 failure the entry survives, so a rerun finds
    it and finishes. After a step-3 failure the entry is correctly gone, and `drop_repo` used to
    answer "is not indexed" forever while the stray binding stayed unreachable — `forget_repo`
    has no other caller. The ordering's whole justification is "break toward the state a rerun
    can fix", so the step it was written for has to be one a rerun can actually fix.
    """

    def test_a_rerun_after_a_bindings_failure_clears_the_remainder(self):
        ix = a_populated_index()
        real_forget = bindings.forget_repo
        self.addCleanup(setattr, bindings, "forget_repo", real_forget)
        bindings.forget_repo = _raise(OSError("read-only file system"))
        with self.assertRaises(RepoError):
            ix.drop_repo("alpha")
        # The half-done state the ordering deliberately produces: archive gone, binding left.
        self.assertIsNone(ix.get_repo("alpha"))
        self.assertEqual(bindings.get("/tmp/alpha"), "alpha")

        bindings.forget_repo = real_forget          # the disk comes back
        out = ix.drop_repo("alpha")                 # and the SAME request finishes the job

        self.assertEqual(out["unbound"], ["/tmp/alpha"])
        self.assertTrue(out["already_gone"])
        self.assertIsNone(bindings.get("/tmp/alpha"))
        self.assertEqual(bindings.get("/tmp/beta"), "beta")     # and only this repo's

    def test_a_repo_with_neither_an_entry_nor_bindings_is_STILL_an_error(self):
        """The cost of the fix if it is wrong, so it is pinned: `drop_repo` now succeeds in a
        case that used to raise, and the two cases are one `if` apart. A name that was never a
        repository must still be refused — including while OTHER repos are bound, which is the
        state that would let a careless "did anything get unbound?" answer yes for the wrong
        reason.
        """
        ix = a_populated_index()
        with self.assertRaises(RepoError) as caught:
            ix.drop_repo("gamma")
        self.assertIn("not indexed", str(caught.exception))
        # And the refusal cost the bound repos nothing.
        self.assertEqual(bindings.get("/tmp/alpha"), "alpha")
        self.assertEqual(bindings.get("/tmp/beta"), "beta")

    def test_finishing_does_NOT_absorb_a_divergence(self):
        """The finishing path finishes the BINDING step and nothing else, which is the only
        step that could otherwise never be finished. It deliberately leaves chunks-without-an
        -entry alone: `list_repos` refuses to invent an entry for them for the same reason, and
        deleting them here would make the same divergence droppable or not according to whether
        a binding happens to exist — silently absorbing, on one path only, the exact state
        `divergent_repos` exists to put in front of a human.
        """
        ix = a_populated_index()
        make_divergent(ix, "ghost", a_file())      # chunks, and no registry entry
        bindings.bind("/tmp/ghost", "ghost")

        out = ix.drop_repo("ghost")

        self.assertEqual(out["unbound"], ["/tmp/ghost"])
        self.assertIsNone(bindings.get("/tmp/ghost"))
        # Still there, and still named as a divergence rather than quietly swallowed.
        self.assertEqual(ix.divergent_repos(), ["ghost"])


class TestItsOwnErrorIsNotWrappedTwice(unittest.TestCase):
    """`search` puts `except RepoError: raise` ahead of its broad handler (`core/repos.py`), and
    `drop_repo`'s two handlers must not drift from that sibling. Nothing inside either `try`
    raises `RepoError` today, so the guard is unreachable in production — which is exactly why
    it is pinned here instead of left as a comment: the day a call is added inside one of them,
    this module's own message must reach the caller once, not nested in a paraphrase of itself.
    """

    def test_from_the_chunk_delete(self):
        ix = a_populated_index()
        ix.q.delete_by_filter = _raise(RepoError("the archive said no"))
        with self.assertRaises(RepoError) as caught:
            ix.drop_repo("alpha")
        self.assertEqual(str(caught.exception), "the archive said no")

    def test_from_clearing_the_bindings(self):
        ix = a_populated_index()
        self.addCleanup(setattr, bindings, "forget_repo", bindings.forget_repo)
        bindings.forget_repo = _raise(RepoError("the archive said no"))
        with self.assertRaises(RepoError) as caught:
            ix.drop_repo("alpha")
        self.assertEqual(str(caught.exception), "the archive said no")


class TestDivergence(unittest.TestCase):
    def test_chunks_without_a_registry_entry_are_reported(self):
        """The price of honesty for having two sources of truth about which repos exist. The
        registry is authoritative; this is how the copy is caught diverging."""
        ix = a_populated_index()
        make_divergent(ix, "ghost", a_file())
        self.assertEqual(ix.divergent_repos(), ["ghost"])

    def test_a_healthy_archive_reports_no_divergence(self):
        self.assertEqual(a_populated_index().divergent_repos(), [])


class TestTheOtherDivergenceIsAnEntryOverAnEmptyArCHIVE(unittest.TestCase):
    """An entry claiming chunks over an archive that has none — and it IS detectable now.

    It was called undetectable, and the reasoning was sound while it held: an entry with
    zero chunks is byte-for-byte a repository that was just registered, which is the normal
    path in a design where `register` and `add_files` are separate calls. A report firing on
    every fresh registration is one people learn to ignore.

    The registry recording what the last `add_files` wrote is what separates them: a fresh
    registration claims zero, and an archive whose chunks are gone claims N and has none.
    The spec's failure table asked for this — "registro tem repo, acervo não tem chunk:
    listagem marca a divergência" — and nothing marked it.
    """

    def test_an_entry_whose_chunks_are_gone_is_reported(self):
        ix = a_populated_index()
        make_emptied(ix, "gamma", a_file())
        self.assertEqual(ix.emptied_repos(), ["gamma"])

    def test_a_FRESH_REGISTRATION_is_NOT_reported(self):
        """The whole reason this was refused before. Crying wolf on the happy path costs
        more than the case it catches, so the distinction has to be real, not hopeful."""
        ix = a_populated_index()
        ix.register_request("just-declared")
        self.assertEqual(ix.emptied_repos(), [])

    def test_a_healthy_archive_reports_neither_direction(self):
        ix = a_populated_index()
        self.assertEqual((ix.divergent_repos(), ix.emptied_repos()), ([], []))

    def test_the_two_directions_stay_separate(self):
        """Chunks with no entry and an entry with no chunks are different repairs — one is
        `drop`, the other is reindex — so one list would send half the readers to the wrong
        remedy."""
        ix = a_populated_index()
        make_divergent(ix, "ghost", a_file())
        make_emptied(ix, "gamma", a_file())
        self.assertEqual(ix.divergent_repos(), ["ghost"])
        self.assertEqual(ix.emptied_repos(), ["gamma"])

    def test_the_listing_carries_both_so_a_host_cannot_answer_with_half(self):
        ix = a_populated_index()
        make_emptied(ix, "gamma", a_file())
        out = ix.list_request()
        self.assertEqual(out["emptied"], ["gamma"])
        self.assertIn("gamma", [r["repo"] for r in out["repos"]],
                      "it is still a listed repo — that is what makes it fixable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
