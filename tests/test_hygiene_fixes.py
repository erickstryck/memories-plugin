"""Tests for the four hygiene fixes raised in review.

None of them broke functionality, and that is exactly why they need tests: a defect that
breaks nothing is the one that comes back without anyone noticing.
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
from core.docs import DEFAULT_TTL_SECONDS, DocIndex
from core.memory import MemoryStore
from tests.fakes import FakeEmbedder, FailingFakeEmbedder, FakeVectorStore

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks"))


def load_hook():
    """Imports the hook with STATE_DIR pointing at a temporary directory."""
    import importlib
    tmp = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = tmp
    import recall
    importlib.reload(recall)

    return recall, Path(tmp)


class TestStatePruning(unittest.TestCase):
    """An entry in `seen` only matters while it can prevent a reinjection."""

    def test_drops_what_can_no_longer_change_a_decision(self):
        recall, _ = load_hook()
        state = {"round": 20, "seen": {"recent": 19, "at_the_edge": 12, "old": 5}}
        pruned = recall.prune_state(state)
        self.assertEqual(pruned, 2, "12 and 5 are >= 8 rounds away")
        self.assertEqual(set(state["seen"]), {"recent"})

    def test_keeps_what_still_prevents_a_reinjection(self):
        recall, _ = load_hook()
        state = {"round": 10, "seen": {"a": 9, "b": 4}}  # 10-4 = 6 < 8
        recall.prune_state(state)
        self.assertEqual(set(state["seen"]), {"a", "b"})

    def test_corrupted_value_is_discarded(self):
        recall, _ = load_hook()
        state = {"round": 5, "seen": {"ok": 4, "junk": "not a number", "null": None}}
        recall.prune_state(state)
        self.assertEqual(set(state["seen"]), {"ok"})

    def test_empty_state_does_not_break(self):
        recall, _ = load_hook()
        self.assertEqual(recall.prune_state({}), 0)


class TestDeadSessionCleanup(unittest.TestCase):
    def test_removes_the_old_and_keeps_the_recent(self):
        recall, dir_ = load_hook()
        previous = dir_ / "recall-dead.json"
        recent = dir_ / "recall-alive.json"
        for f in (previous, recent):
            f.write_text(json.dumps({"round": 1, "seen": {}}))
        old_ts = time.time() - 10 * 86400
        os.utime(previous, (old_ts, old_ts))
        removed = recall.purge_dead_sessions(days=7.0)
        self.assertEqual(removed, 1)
        self.assertFalse(previous.exists())
        self.assertTrue(recent.exists())

    def test_does_not_touch_the_log(self):
        recall, dir_ = load_hook()
        log = dir_ / "recall.log"
        log.write_text("a line")
        old_ts = time.time() - 30 * 86400
        os.utime(log, (old_ts, old_ts))
        recall.purge_dead_sessions(days=1.0)
        self.assertTrue(log.exists(), "the log is not session state")


class TestTolerantEnvReading(unittest.TestCase):
    """Read at module load, BEFORE the catch-all — it must not blow up."""

    def test_valid_number(self):
        recall, _ = load_hook()
        os.environ["QCTX_TESTE_NUM"] = "42"
        try:
            self.assertEqual(recall.env_num("QCTX_TESTE_NUM", "X", "7", int), 42)
        finally:
            del os.environ["QCTX_TESTE_NUM"]

    def test_invalid_number_falls_back_to_the_default_and_is_recorded(self):
        recall, _ = load_hook()
        recall._pending_notes.clear()
        os.environ["QCTX_TESTE_NUM"] = "14k"
        try:
            self.assertEqual(recall.env_num("QCTX_TESTE_NUM", "X", "14000", int), 14000)
            self.assertTrue(any("14k" in p for p in recall._pending_notes),
                            "the bad value has to be recorded, not vanish")
        finally:
            del os.environ["QCTX_TESTE_NUM"]

    def test_absent_uses_the_default(self):
        recall, _ = load_hook()
        self.assertAlmostEqual(recall.env_num("QCTX_NAO_EXISTE", "NEM_ESTE", "0.58"), 0.58)


class TestTtlPreservedAcrossRefresh(unittest.TestCase):
    def _index(self):
        q, emb = FakeVectorStore(), FakeEmbedder()

        return DocIndex(q, emb, None, "tmp", "lib", emb.dim), q

    def _write_file(self, text="document content\n"):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "doc.md")
        Path(path).write_text(text)

        return path

    def test_lifetime_is_stored_at_index_time(self):
        idx, q = self._index()
        idx.index_file(self._write_file(), ttl_seconds=3600)
        md = list(q.collections["tmp"]["points"].values())[0]["payload"]["metadata"]
        self.assertEqual(md["ttl_seconds"], 3600)

    def test_refresh_reuses_the_lifetime_instead_of_the_default(self):
        idx, q = self._index()
        path = self._write_file()
        idx.index_file(path, ttl_seconds=3600)          # the user asked for 1 hour
        Path(path).write_text("changed, longer content\n")  # forces a reindex
        idx.refresh(scope="tmp")
        p = list(q.collections["tmp"]["points"].values())[0]["payload"]
        remaining = p["expires_at_ts"] - time.time()
        self.assertLess(remaining, 3700, "it must not become the 24h default")
        self.assertGreater(remaining, 3500)
        self.assertNotAlmostEqual(remaining, DEFAULT_TTL_SECONDS, delta=100)

    def test_library_gains_no_expiry_on_refresh(self):
        idx, q = self._index()
        path = self._write_file()
        idx.keep_file(path)
        Path(path).write_text("quite different content\n")
        idx.refresh(scope="library")
        for p in q.collections["lib"]["points"].values():
            self.assertNotIn("expires_at_ts", p["payload"])


class TestUpdateWithoutReembedding(unittest.TestCase):
    def _store(self, embedder=None):
        q = FakeVectorStore()
        emb = embedder or FakeEmbedder()
        q.ensure_collection("mem", 8)

        return MemoryStore(q, emb, None, "mem", 8), q, emb

    def test_metadata_alone_does_not_call_embedding(self):
        s, q, emb = self._store()
        mid = s.store("text that does not change")["id"]
        calls_before = len(emb.calls)
        res = s.update(mid, metadata={"type": "feedback"})
        self.assertFalse(res["reembedded"])
        self.assertEqual(len(emb.calls), calls_before, "an identical vector is not recomputed")
        self.assertEqual(q.get_point("mem", mid)["payload"]["metadata"], {"type": "feedback"})

    def test_text_equal_to_the_previous_one_also_skips_the_call(self):
        s, _, emb = self._store()
        mid = s.store("same text")["id"]
        before = len(emb.calls)
        res = s.update(mid, information="same text")
        self.assertFalse(res["reembedded"])
        self.assertEqual(len(emb.calls), before)

    def test_different_text_calls(self):
        s, _, emb = self._store()
        mid = s.store("original")["id"]
        before = len(emb.calls)
        res = s.update(mid, information="genuinely changed")
        self.assertTrue(res["reembedded"])
        self.assertEqual(len(emb.calls), before + 1)

    def test_fixing_a_label_WORKS_with_embedding_down(self):
        """The real reason for the fix: the operation does not depend on embedding, so it
        must not become impossible when the endpoint is down."""
        q = FakeVectorStore()
        q.ensure_collection("mem", 8)
        live_store = MemoryStore(q, FakeEmbedder(), None, "mem", 8)
        mid = live_store.store("fact stored while the endpoint was working")["id"]

        broken_store = MemoryStore(q, FailingFakeEmbedder(core.EmbeddingError("down")),
                               None, "mem", 8)
        res = broken_store.update(mid, metadata={"type": "corrected"})
        self.assertEqual(res["status"], "updated")
        self.assertEqual(q.get_point("mem", mid)["payload"]["metadata"], {"type": "corrected"})

    def test_vector_is_preserved_untouched(self):
        s, q, _ = self._store()
        mid = s.store("text")["id"]
        vector_before = list(q.get_point("mem", mid)["vector"])
        s.update(mid, metadata={"x": 1})
        self.assertEqual(q.get_point("mem", mid)["vector"], vector_before)

    def test_the_four_keys_survive_the_no_reembedding_path(self):
        s, q, _ = self._store()
        mid = s.store("text", {"type": "a"})["id"]
        s.update(mid, metadata={"type": "b"})
        self.assertEqual(set(q.get_point("mem", mid)["payload"]),
                         {"document", "metadata", "created_at", "updated_at"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
