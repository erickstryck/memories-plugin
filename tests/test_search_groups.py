import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fakes import FakeEmbedder, FakeVectorStore  # noqa: E402

COLL = "repos_test"


def a_store_with(loud_chunks: int, quiet_chunks: int) -> tuple:
    """A store where one repo is LOUD (many near-identical chunks about the subject) and
    another is QUIET (one weaker mention). This asymmetry is the whole point: a balanced
    fixture passes under client-side grouping and proves nothing."""
    store, embedder = FakeVectorStore(), FakeEmbedder(dim=8)
    store.ensure_collection(COLL, 8)
    points = []
    for i in range(loud_chunks):
        points.append({"id": 1000 + i, "vector": embedder.embed_one("billing invoice charge"),
                       "payload": {"repo": "loud", "document": f"billing {i}"}})
    for i in range(quiet_chunks):
        points.append({"id": 2000 + i, "vector": embedder.embed_one("invoice"),
                       "payload": {"repo": "quiet", "document": f"invoice {i}"}})
    store.upsert(COLL, points)

    return store, embedder


class TestTheQuietRepoIsNotShadowed(unittest.TestCase):
    def test_both_repos_come_back_even_when_one_dominates(self):
        store, embedder = a_store_with(loud_chunks=50, quiet_chunks=1)
        groups = store.search_groups(COLL, embedder.embed_one("billing invoice charge"),
                                     group_by="repo", limit=10, group_size=3)
        self.assertEqual({g["id"] for g in groups}, {"loud", "quiet"},
                         "the single-mention repo was shadowed by the loud one")

    def test_a_plain_search_DOES_shadow_it(self):
        """The control. Without this, the test above could pass for the wrong reason and we
        would never learn that grouping is what fixed it."""
        store, embedder = a_store_with(loud_chunks=50, quiet_chunks=1)
        hits = store.search(COLL, embedder.embed_one("billing invoice charge"), limit=10)
        self.assertEqual({h["payload"]["repo"] for h in hits}, {"loud"},
                         "the premise of this whole method is false: top-K did NOT shadow")

    def test_group_size_caps_hits_per_group(self):
        store, embedder = a_store_with(loud_chunks=50, quiet_chunks=1)
        groups = store.search_groups(COLL, embedder.embed_one("billing invoice charge"),
                                     group_by="repo", limit=10, group_size=3)
        loud = next(g for g in groups if g["id"] == "loud")
        self.assertEqual(len(loud["hits"]), 3)

    def test_limit_caps_the_number_of_groups(self):
        store, embedder = a_store_with(loud_chunks=50, quiet_chunks=1)
        groups = store.search_groups(COLL, embedder.embed_one("billing invoice charge"),
                                     group_by="repo", limit=1, group_size=3)
        self.assertEqual(len(groups), 1)

    def test_a_point_without_the_field_is_not_a_group(self):
        store, embedder = a_store_with(loud_chunks=2, quiet_chunks=1)
        store.upsert(COLL, [{"id": 9999, "vector": embedder.embed_one("billing invoice charge"),
                             "payload": {"document": "no repo key at all"}}])
        groups = store.search_groups(COLL, embedder.embed_one("billing invoice charge"),
                                     group_by="repo", limit=10, group_size=3)
        self.assertNotIn(None, {g["id"] for g in groups})


@unittest.skipUnless(os.environ.get("QCTX_INTEGRATION") == "1",
                     "integration: needs a reachable Qdrant")
class TestTheRealServerGroups(unittest.TestCase):
    """Measured on 1.18.2 while designing this: `search/groups` returned 4 distinct groups in
    one query. This pins the CONTRACT (shape of the response) against the real server, which
    is the half a fake cannot prove."""

    def test_the_response_shape_is_groups_with_hits(self):
        from core import load
        from core.qdrant import build_qdrant
        cfg = load()
        q = build_qdrant(cfg)
        name = cfg.require_docs_collection()
        info = q.collection_info(name)
        if not info or not info.get("points_count"):
            self.skipTest(f"{name} is empty: zero groups from an empty collection would "
                          f"look like a missing capability and prove nothing")
        dim = cfg.vector_size
        groups = q.search_groups(name, [0.01] * dim, group_by="doc_id", limit=4, group_size=2)
        self.assertTrue(groups, "the real server returned no groups for a populated collection")
        for g in groups:
            self.assertIn("id", g)
            self.assertTrue(g["hits"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
