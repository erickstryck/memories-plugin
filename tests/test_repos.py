import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.repos import RepoError, RepoIndex  # noqa: E402
from tests.fakes import FakeEmbedder, FakeVectorStore  # noqa: E402

CHUNKS, REG = "repos_c", "repos_r"


def an_index() -> RepoIndex:
    return RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), CHUNKS, REG, 8)


def a_file(text: str, suffix: str = ".py") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)

    return path


class TestWriting(unittest.TestCase):
    def test_every_chunk_carries_the_repo(self):
        ix = an_index()
        ix.add_files("alpha", [a_file("def one():\n    return 1\n")])
        points = list(ix.q.scroll_all(CHUNKS))
        self.assertTrue(points)
        self.assertEqual({p["payload"]["repo"] for p in points}, {"alpha"})

    def test_the_repo_is_top_level_and_not_buried_in_metadata(self):
        """`group_by` and the payload index address a top-level key, exactly as `doc_id`
        already does. Burying it under metadata would work for reading and break both."""
        ix = an_index()
        ix.add_files("alpha", [a_file("x = 1\n")])
        payload = next(iter(ix.q.scroll_all(CHUNKS)))["payload"]
        self.assertIn("repo", payload)
        self.assertNotIn("repo", payload.get("metadata", {}))

    def test_reindexing_the_same_file_replaces_instead_of_accumulating(self):
        """Without this the old version and the new one coexist and one search mixes chunks
        from two states of the same file."""
        ix = an_index()
        path = a_file("first content here\n")
        ix.add_files("alpha", [path])
        before = len(list(ix.q.scroll_all(CHUNKS)))
        with open(path, "w") as fh:
            fh.write("second content, entirely different\n")
        ix.add_files("alpha", [path])
        after = list(ix.q.scroll_all(CHUNKS))
        self.assertEqual(len(after), before)
        self.assertIn("second", " ".join(p["payload"]["document"] for p in after))

    def test_a_file_that_shrinks_leaves_no_orphan_chunks_behind(self):
        """The replacement above is invisible when the chunk count does not change: point ids
        are derived from `(doc_id, chunk index)`, so re-upserting a file that still needs the
        same N chunks overwrites the same N ids and looks clean with no deletion at all.

        Only a file that now needs FEWER chunks exposes the orphans, and they are the real
        damage: chunks of a version that no longer exists on disk, still answering searches
        and still reporting themselves as that file's lines.
        """
        ix = an_index()
        path = a_file("\n\n".join(f"def f{i}():\n    return {i}\n" + "# padding line here\n" * 40
                                 for i in range(6)))
        ix.add_files("alpha", [path])
        self.assertGreater(len(list(ix.q.scroll_all(CHUNKS))), 1)
        with open(path, "w") as fh:
            fh.write("x = 1\n")
        ix.add_files("alpha", [path])
        self.assertEqual(len(list(ix.q.scroll_all(CHUNKS))), 1)

    def test_an_empty_file_is_skipped_and_reported_not_raised(self):
        """One unindexable file in a list of eight hundred must not abort the other 799."""
        ix = an_index()
        out = ix.add_files("alpha", [a_file("   \n\n"), a_file("real = 1\n")])
        self.assertEqual(out["files"], 1)
        self.assertEqual(len(out["skipped"]), 1)

    def test_an_unreadable_file_is_skipped_and_reported(self):
        ix = an_index()
        gone = os.path.join(tempfile.mkdtemp(), "never-existed.py")
        out = ix.add_files("alpha", [gone, a_file("real = 1\n")])
        self.assertEqual(out["files"], 1)
        self.assertEqual([p for p, _ in out["skipped"]], [gone])

    def test_writing_under_an_empty_repo_name_is_refused(self):
        """Chunks whose `repo` is empty can never be filtered to and never be dropped: the
        one operation that would remove them is the one that needs the name. Refusing at the
        door is the only point where the mistake is still cheap."""
        ix = an_index()
        with self.assertRaises(RepoError):
            ix.add_files("", [a_file("x = 1\n")])
        self.assertEqual(list(ix.q.scroll_all(CHUNKS)), [])

    def test_the_digest_is_stored_so_staleness_can_be_judged_later(self):
        """`source_changed` compares by digest because mtime and size both lie: cp -p,
        rsync --times and any restore preserve mtime, and a one-character edit preserves
        size. Storing it is what lets the watcher (sub-project E) exist at all."""
        ix = an_index()
        ix.add_files("alpha", [a_file("content = 1\n")])
        md = next(iter(ix.q.scroll_all(CHUNKS)))["payload"]["metadata"]
        self.assertTrue(md["src_digest"])
        self.assertIn("src_mtime", md)
        self.assertIn("src_size", md)


class TestTheRegistry(unittest.TestCase):
    def test_registering_makes_the_repo_listable(self):
        ix = an_index()
        ix.register("alpha", "Alpha", ["git@host:me/alpha.git"], "/home/me/alpha")
        self.assertEqual([r["repo"] for r in ix.list_repos()], ["alpha"])

    def test_the_registry_is_authoritative_over_WHICH_repos_exist(self):
        """Chunks own content; the registry owns existence. A repo with chunks and no entry
        is a divergence, not a repo — and it must be visible as one."""
        ix = an_index()
        ix.add_files("ghost", [a_file("x = 1\n")])
        self.assertEqual(ix.list_repos(), [])
        self.assertIsNone(ix.get_repo("ghost"))

    def test_registering_the_same_repo_twice_accumulates_checkouts_without_duplicating(self):
        ix = an_index()
        ix.register("alpha", "Alpha", ["git@host:me/alpha.git"], "/home/me/alpha")
        ix.register("alpha", "Alpha", ["git@host:me/alpha.git"], "/home/me/alpha-2")
        ix.register("alpha", "Alpha", ["git@host:me/alpha.git"], "/home/me/alpha")
        entry = ix.get_repo("alpha")
        self.assertEqual(sorted(entry["checkouts"]), ["/home/me/alpha", "/home/me/alpha-2"])
        self.assertEqual(len(ix.list_repos()), 1)

    def test_the_registry_never_lands_in_the_chunk_collection(self):
        ix = an_index()
        ix.register("alpha", "Alpha", [], "/home/me/alpha")
        self.assertEqual(list(ix.q.scroll_all(CHUNKS)), [])

    def test_an_empty_repo_name_is_refused(self):
        ix = an_index()
        with self.assertRaises(RepoError):
            ix.register("", "Alpha", [], "/home/me/alpha")


if __name__ == "__main__":
    unittest.main(verbosity=2)
