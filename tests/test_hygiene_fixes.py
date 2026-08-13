"""Testes das quatro correções de higiene apontadas em revisão.

Nenhuma delas quebrava funcionalidade, e é justamente por isso que precisam de
teste: defeito que não quebra nada é o que volta sem ninguém notar.
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
    """Importa o hook com STATE_DIR apontando para um diretório temporário."""
    import importlib
    tmp = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = tmp
    import recall
    importlib.reload(recall)

    return recall, Path(tmp)


class TestStatePruning(unittest.TestCase):
    """Entrada em `seen` só importa enquanto pode evitar uma reinjeção."""

    def test_drops_what_can_no_longer_change_a_decision(self):
        recall, _ = load_hook()
        state = {"round": 20, "seen": {"recente": 19, "no_limite": 12, "velha": 5}}
        pruned = recall.prune_state(state)
        self.assertEqual(pruned, 2, "12 e 5 estão a >= 8 rodadas de distância")
        self.assertEqual(set(state["seen"]), {"recente"})

    def test_keeps_what_still_prevents_a_reinjection(self):
        recall, _ = load_hook()
        state = {"round": 10, "seen": {"a": 9, "b": 4}}  # 10-4 = 6 < 8
        recall.prune_state(state)
        self.assertEqual(set(state["seen"]), {"a", "b"})

    def test_corrupted_value_is_discarded(self):
        recall, _ = load_hook()
        state = {"round": 5, "seen": {"ok": 4, "lixo": "nao é numero", "nulo": None}}
        recall.prune_state(state)
        self.assertEqual(set(state["seen"]), {"ok"})

    def test_empty_state_does_not_break(self):
        recall, _ = load_hook()
        self.assertEqual(recall.prune_state({}), 0)


class TestDeadSessionCleanup(unittest.TestCase):
    def test_removes_the_old_and_keeps_the_recent(self):
        recall, dir_ = load_hook()
        previous = dir_ / "recall-morta.json"
        recent = dir_ / "recall-viva.json"
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
        log.write_text("linha")
        old_ts = time.time() - 30 * 86400
        os.utime(log, (old_ts, old_ts))
        recall.purge_dead_sessions(days=1.0)
        self.assertTrue(log.exists(), "o log não é estado de sessão")


class TestTolerantEnvReading(unittest.TestCase):
    """Lido no carregamento do módulo, ANTES do catch-all — não pode explodir."""

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
                            "o valor ruim tem de ficar registrado, não sumir")
        finally:
            del os.environ["QCTX_TESTE_NUM"]

    def test_absent_uses_the_default(self):
        recall, _ = load_hook()
        self.assertAlmostEqual(recall.env_num("QCTX_NAO_EXISTE", "NEM_ESTE", "0.58"), 0.58)


class TestTtlPreservedAcrossRefresh(unittest.TestCase):
    def _index(self):
        q, emb = FakeVectorStore(), FakeEmbedder()

        return DocIndex(q, emb, None, "tmp", "lib", emb.dim), q

    def _write_file(self, text="conteudo do documento\n"):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "doc.md")
        Path(path).write_text(text)

        return path

    def test_lifetime_is_stored_at_index_time(self):
        idx, q = self._index()
        idx.index_file(self._write_file(), ttl_seconds=3600)
        md = list(q.collections["tmp"]["pontos"].values())[0]["payload"]["metadata"]
        self.assertEqual(md["ttl_seconds"], 3600)

    def test_refresh_reuses_the_lifetime_instead_of_the_default(self):
        idx, q = self._index()
        path = self._write_file()
        idx.index_file(path, ttl_seconds=3600)          # o usuário pediu 1 hora
        Path(path).write_text("conteudo alterado maior\n")  # força reindexação
        idx.refresh(scope="tmp")
        p = list(q.collections["tmp"]["pontos"].values())[0]["payload"]
        remaining = p["expires_at_ts"] - time.time()
        self.assertLess(remaining, 3700, "não pode virar o default de 24h")
        self.assertGreater(remaining, 3500)
        self.assertNotAlmostEqual(remaining, DEFAULT_TTL_SECONDS, delta=100)

    def test_library_gains_no_expiry_on_refresh(self):
        idx, q = self._index()
        path = self._write_file()
        idx.keep_file(path)
        Path(path).write_text("outro conteudo bem diferente\n")
        idx.refresh(scope="library")
        for p in q.collections["lib"]["pontos"].values():
            self.assertNotIn("expires_at_ts", p["payload"])


class TestUpdateWithoutReembedding(unittest.TestCase):
    def _store(self, embedder=None):
        q = FakeVectorStore()
        emb = embedder or FakeEmbedder()
        q.ensure_collection("mem", 8)

        return MemoryStore(q, emb, None, "mem", 8), q, emb

    def test_metadata_alone_does_not_call_embedding(self):
        s, q, emb = self._store()
        mid = s.store("texto que não muda")["id"]
        calls_before = len(emb.calls)
        res = s.update(mid, metadata={"type": "feedback"})
        self.assertFalse(res["reembedded"])
        self.assertEqual(len(emb.calls), calls_before, "vetor idêntico não se recalcula")
        self.assertEqual(q.get_point("mem", mid)["payload"]["metadata"], {"type": "feedback"})

    def test_text_equal_to_the_previous_one_also_skips_the_call(self):
        s, _, emb = self._store()
        mid = s.store("mesmo texto")["id"]
        antes = len(emb.calls)
        res = s.update(mid, information="mesmo texto")
        self.assertFalse(res["reembedded"])
        self.assertEqual(len(emb.calls), antes)

    def test_different_text_calls(self):
        s, _, emb = self._store()
        mid = s.store("original")["id"]
        antes = len(emb.calls)
        res = s.update(mid, information="mudou de verdade")
        self.assertTrue(res["reembedded"])
        self.assertEqual(len(emb.calls), antes + 1)

    def test_fixing_a_label_WORKS_with_embedding_down(self):
        """O motivo real do conserto: a operação não depende de embedding, então não
        pode ficar impossível quando o endpoint está fora."""
        q = FakeVectorStore()
        q.ensure_collection("mem", 8)
        live_store = MemoryStore(q, FakeEmbedder(), None, "mem", 8)
        mid = live_store.store("fato gravado enquanto o endpoint funcionava")["id"]

        broken_store = MemoryStore(q, FailingFakeEmbedder(core.EmbeddingError("fora do ar")),
                               None, "mem", 8)
        res = broken_store.update(mid, metadata={"type": "corrigido"})
        self.assertEqual(res["status"], "updated")
        self.assertEqual(q.get_point("mem", mid)["payload"]["metadata"], {"type": "corrigido"})

    def test_vector_is_preserved_untouched(self):
        s, q, _ = self._store()
        mid = s.store("texto")["id"]
        vetor_antes = list(q.get_point("mem", mid)["vector"])
        s.update(mid, metadata={"x": 1})
        self.assertEqual(q.get_point("mem", mid)["vector"], vetor_antes)

    def test_the_four_keys_survive_the_no_reembedding_path(self):
        s, q, _ = self._store()
        mid = s.store("texto", {"type": "a"})["id"]
        s.update(mid, metadata={"type": "b"})
        self.assertEqual(set(q.get_point("mem", mid)["payload"]),
                         {"document", "metadata", "created_at", "updated_at"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
