"""INTEGRATION tests — they require a real Qdrant and real model endpoints.

They stay out of the default suite on purpose: `python3 -m unittest discover -s tests`
has to run offline and in milliseconds. To run these:

    QCTX_INTEGRATION=1 python3 -m unittest tests.test_integration -v

SAFETY RULE, and it is not negotiable: the configured memory archive is treated as
PRODUCTION. Every write goes to a throwaway collection created and deleted by the test
itself; against the real archive there is only READING. A test that deletes real memory
is worse than no test.

What these tests prove, and an offline test cannot: that the core reads the payload
written by the old MCP server with no conversion, and that the payload it writes has
exactly the same keys — that is what makes the replacement safe.

The queries against the real archive stay in Portuguese, because the archive is: this is
where the retrieval actually meets the language it will be used in.
"""
import os
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
from core import retrieval

ENABLED = os.environ.get("QCTX_INTEGRATION") == "1"
THROWAWAY_COLLECTION = f"qctx_test_{uuid.uuid4().hex[:8]}"

#: The keys the previous MCP server wrote. The core has to read and write exactly this,
#: otherwise the existing archive becomes unreadable or inconsistent.
PAYLOAD_KEYS = {"document", "metadata", "created_at", "updated_at"}


def read_config():
    return core.load()


def write_config():
    """Config pointing memory at the throwaway collection."""
    base = core.load()
    fields = {f: getattr(base, f) for f in base.__dataclass_fields__}
    fields["memory_collection"] = THROWAWAY_COLLECTION

    return core.Config(**fields)


@unittest.skipUnless(ENABLED, "set QCTX_INTEGRATION=1")
class TestReadingTheRealArchive(unittest.TestCase):
    """READ ONLY. Proves the core understands what is already stored."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = read_config()
        if not cls.cfg.memory_collection:
            raise unittest.SkipTest("memory_collection is not configured")
        cls.store = core.build_memory(cls.cfg)

    def test_archive_has_points(self):
        total = self.store.count()
        self.assertIsNotNone(total)
        self.assertGreater(total, 0, "the real archive should hold memories")

    def test_payload_written_by_the_old_mcp_is_readable(self):
        page = self.store.list_page(limit=5)
        self.assertGreater(page["count"], 0)
        for m in page["memories"]:
            self.assertIsInstance(m["document"], str)
            self.assertTrue(m["document"].strip(), "the document must not come back empty")
            self.assertIsInstance(m["metadata"], dict)

    def test_get_by_id_returns_the_four_keys(self):
        page = self.store.list_page(limit=1)
        mid = page["memories"][0]["id"]
        m = self.store.get(mid)
        self.assertNotEqual(m.get("status"), "not_found")
        for key in ("document", "metadata", "created_at", "updated_at"):
            self.assertIn(key, m, f"{key} has to exist in the legacy payload")

    def test_find_returns_descending_scores(self):
        hits = self.store.find("memória de longo prazo", limit=5)
        self.assertTrue(hits, "dense search against the real archive returned nothing")
        scores = [h["score"] for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_two_gate_recall_against_the_real_archive(self):
        policy = core.Policy(0.45, 0.58, 0.10, 6, veto=True)
        hits, outcome = self.store.recall(["memória de longo prazo e recall automático"],
                                       policy, top_k=20)
        self.assertGreater(outcome.candidates, 0)
        for h in hits:
            self.assertIsInstance(h, core.Recalled)
            self.assertIn(h.origin, ("CE", "dense"))
            self.assertGreaterEqual(h.dense_score, 0.0)
        if outcome.by_rerank:
            # NOT `all(origin == "CE")`: under veto, `two_stage` keeps only `above`, which
            # is CE by construction, so that assertion cannot fail. A tautology dressed as
            # a live-model check is worse than no check — it reads like coverage.
            # What IS falsifiable: the model actually separated the candidates rather than
            # crushing everything into the collapse band.
            self.assertGreater(outcome.best_rerank, retrieval.COLLAPSE_MAX,
                               "every score in the collapse band means the CE told us nothing")
            self.assertTrue(all(h.score >= 0.10 for h in hits),
                            "the veto is supposed to have removed anything below the cutoff")

    def test_recall_with_several_angles_fuses_by_id(self):
        hits, _ = self.store.recall(
            ["como funciona o hook de recall",
             "hook recall funciona como",
             "recall automático a cada prompt"],
            core.Policy(0.45, 0.58, 0.10, 6, veto=True), top_k=10)
        ids = [h.id for h in hits]
        self.assertEqual(len(ids), len(set(ids)), "fusion by id must not duplicate")


@unittest.skipUnless(ENABLED, "set QCTX_INTEGRATION=1")
class TestCrudInAThrowawayCollection(unittest.TestCase):
    """Writes ONLY here. The collection is created and destroyed by the test itself."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = write_config()
        cls.store = core.build_memory(cls.cfg)
        cls.q = core.build_qdrant(cls.cfg)
        assert cls.cfg.memory_collection == THROWAWAY_COLLECTION

    @classmethod
    def tearDownClass(cls):
        try:
            cls.q.delete_collection(THROWAWAY_COLLECTION)
        except Exception:
            pass

    def test_full_cycle(self):
        created_at = self.store.store("The connector poll truncates at 100 items per page.",
                                  {"type": "reference", "date": "2026-08-13"})
        self.assertEqual(created_at["status"], "created")
        mid = created_at["id"]

        read_back = self.store.get(mid)
        self.assertIn("truncates at 100", read_back["document"])
        self.assertEqual(read_back["metadata"]["type"], "reference")

        found_hit = self.store.find("poll pagination", limit=5)
        self.assertIn(mid, [h["id"] for h in found_hit])

        updated_at = self.store.update(mid, information="Corrected: it truncates at 50 items.")
        self.assertEqual(updated_at["status"], "updated")
        reread = self.store.get(mid)
        self.assertIn("50 items", reread["document"])
        self.assertEqual(reread["metadata"]["type"], "reference",
                         "an update without metadata has to PRESERVE the previous metadata")
        self.assertEqual(reread["created_at"], read_back["created_at"],
                         "created_at must not be rewritten by an update")
        self.assertNotEqual(reread["updated_at"], read_back["updated_at"])

        self.store.delete(mid)
        self.assertEqual(self.store.get(mid)["status"], "not_found")

    def test_written_payload_has_the_same_keys_as_the_old_mcp(self):
        created_at = self.store.store("fact for checking the payload shape", {"type": "test"})
        point = self.q.get_point(THROWAWAY_COLLECTION, created_at["id"])
        self.assertEqual(set(point["payload"].keys()), PAYLOAD_KEYS,
                         "the payload has to be identical to the previous server's, "
                         "otherwise the existing archive becomes inconsistent")
        self.store.delete(created_at["id"])

    def test_store_many_is_all_or_nothing(self):
        items = [{"information": f"batch fact number {i}", "metadata": {"type": "test"}}
                 for i in range(5)]
        res = self.store.store_many(items)
        self.assertEqual(res["count"], 5)
        for mid in res["ids"]:
            self.assertNotEqual(self.store.get(mid).get("status"), "not_found")
        for mid in res["ids"]:
            self.store.delete(mid)

    def test_store_many_refuses_an_invalid_item_before_writing(self):
        before = self.store.count() or 0
        with self.assertRaises(core.memory.MemoryStoreError):
            self.store.store_many([{"information": "valid"}, {"information": "  "}])
        time.sleep(0.2)
        self.assertEqual(self.store.count() or 0, before,
                         "validation has to happen BEFORE any write")

    def test_empty_store_is_refused(self):
        with self.assertRaises(core.memory.MemoryStoreError):
            self.store.store("   ")


@unittest.skipUnless(ENABLED, "set QCTX_INTEGRATION=1")
class TestDocsIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = core.load()
        fields = {f: getattr(base, f) for f in base.__dataclass_fields__}
        fields["docs_collection"] = f"{THROWAWAY_COLLECTION}_tmp"
        fields["library_collection"] = f"{THROWAWAY_COLLECTION}_lib"
        cls.cfg = core.Config(**fields)
        cls.idx = core.build_docs(cls.cfg)
        cls.q = core.build_qdrant(cls.cfg)
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.file_path = Path(cls.tmpdir.name) / "manual.md"
        # A realistic document: each section has to be big enough for the slicing to have
        # work to do, otherwise the "finds the right section" test is vacuous — with a
        # single chunk, getting it right is inevitable.
        filler = ("Operational detail relevant to this section, repeated to give the "
                   "document some body without changing what the section is about. ")
        cls.file_path.write_text("\n".join([
            "# Authentication",
            "",
            "To authenticate, send the Authorization header with a Bearer token.",
            "The token expires in one hour and has to be renewed through the refresh endpoint.",
            filler * 12,
            "",
            "# Pagination",
            "",
            "Listings return at most 100 items per page.",
            "Use the cursor returned in next_page to fetch the following page.",
            filler * 12,
            "",
            "# Rate limits",
            "",
            "The limit is 5000 requests per hour, with a 180-second sliding window.",
            filler * 12,
        ]) + "\n")

    @classmethod
    def tearDownClass(cls):
        for name in (f"{THROWAWAY_COLLECTION}_tmp", f"{THROWAWAY_COLLECTION}_lib"):
            try:
                cls.q.delete_collection(name)
            except Exception:
                pass
        cls.tmpdir.cleanup()

    def test_index_then_search_locates_the_right_section(self):
        res = self.idx.index_file(str(self.file_path), ttl_seconds=600)
        self.assertGreater(res["chunks"], 1, "a document with 3 long sections has to become several chunks")
        self.assertEqual(res["mode"], "locator")

        hits, info = self.idx.search("what is the request limit per hour?",
                                     scope="tmp", limit=3)
        self.assertTrue(hits, "it should find the rate-limit section")
        top_text = hits[0].text.lower()
        self.assertIn("5000", top_text)
        self.assertGreater(hits[0].start_line, 0)
        self.assertGreaterEqual(hits[0].end_line, hits[0].start_line)

    def test_line_range_points_at_the_real_content(self):
        self.idx.index_file(str(self.file_path), ttl_seconds=600)
        hits, _ = self.idx.search("how do I paginate?", scope="tmp", limit=1)
        # `readlines`, not `splitlines`: the promise is about what a READER sees, and
        # asserting it with the same primitive the code uses proves only self-consistency.
        with open(self.file_path, newline="") as fh:
            lines = [ln.rstrip("\n") for ln in fh.readlines()]
        slice_text = "\n".join(lines[hits[0].start_line - 1:hits[0].end_line])
        self.assertEqual(slice_text.strip("\n"), hits[0].text,
                         "the contract of locator mode is that these lines reproduce "
                         "exactly the chunk that was indexed")

    def test_library_never_expires_and_temporary_does(self):
        self.idx.keep_file(str(self.file_path))
        self.idx.index_file(str(self.file_path), ttl_seconds=600)
        docs = {d["scope"]: d for d in self.idx.list_docs("all")}
        self.assertIn("library", docs)
        self.assertIn("tmp", docs)
        self.assertIsNone(docs["library"]["expires_at_ts"])
        self.assertIsNotNone(docs["tmp"]["expires_at_ts"])

    def test_expired_ttl_disappears_from_search(self):
        self.idx.index_file(str(self.file_path), ttl_seconds=-1)  # expired at birth
        hits, _ = self.idx.search("authentication", scope="tmp", limit=3)
        self.assertEqual(hits, [], "an expired chunk must not show up")

    def test_purging_temporary_preserves_the_library(self):
        self.idx.keep_file(str(self.file_path))
        self.idx.index_file(str(self.file_path), ttl_seconds=600)
        self.idx.drop_all_tmp()
        docs = self.idx.list_docs("all")
        scopes = {d["scope"] for d in docs}
        self.assertIn("library", scopes, "the library MUST survive the purge")
        self.assertNotIn("tmp", scopes)

    def test_reindexing_replaces_instead_of_duplicating(self):
        first = self.idx.keep_file(str(self.file_path))
        second = self.idx.keep_file(str(self.file_path))
        self.assertEqual(first["doc_id"], second["doc_id"])
        docs = [d for d in self.idx.list_docs("library") if d["doc_id"] == second["doc_id"]]
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["chunks"], second["chunks"],
                         "no chunk from the previous indexing may be left behind")


if __name__ == "__main__":
    unittest.main(verbosity=2)
