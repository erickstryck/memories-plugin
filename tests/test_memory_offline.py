"""Memória e índice de documentos, com dublês — sem rede, sem credencial.

Cobre offline o que antes só existia sob teste de integração. O caso mais
importante é a FORMA DO PAYLOAD: ela é a propriedade que permite substituir o
servidor anterior sem migrar dado, e estava verificada apenas contra a coleção viva
do usuário. Propriedade de segurança que só é testada em produção não é testada.
"""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
from core import retrieval
from core.docs import DocIndex
from core.memory import MemoryError_, MemoryStore
from tests.fakes import (FakeEmbedder, FailingFakeEmbedder, FakeReranker,
                         FakeVectorStore)

#: As quatro chaves que o servidor MCP anterior gravava. Mudar isto torna o acervo
#: existente inconsistente, sem nenhum erro aparecer.
CHAVES = {"document", "metadata", "created_at", "updated_at"}

POL = retrieval.Policy(dense_floor=0.45, strict_floor=0.58, min_score=0.10,
                       max_results=6, veto=True)


def store(reranker=None, colecao="mem"):
    q, emb = FakeVectorStore(), FakeEmbedder()
    q.ensure_collection(colecao, emb.dim)

    return MemoryStore(q, emb, reranker, colecao, emb.dim), q, emb


class TestPayloadShape(unittest.TestCase):
    """Compatibilidade com o acervo já gravado — a propriedade que sustenta a troca."""

    def test_store_writes_exactly_the_four_keys(self):
        s, q, _ = store()
        r = s.store("um fato", {"type": "reference"})
        point = q.get_point("mem", r["id"])
        self.assertEqual(set(point["payload"]), CHAVES)

    def test_store_many_writes_the_same_keys(self):
        s, q, _ = store()
        r = s.store_many([{"information": "a"}, {"information": "b", "metadata": {"x": 1}}])
        for mid in r["ids"]:
            self.assertEqual(set(q.get_point("mem", mid)["payload"]), CHAVES)

    def test_update_preserves_created_at(self):
        s, q, _ = store()
        mid = s.store("original")["id"]
        antes = q.get_point("mem", mid)["payload"]["created_at"]
        time.sleep(0.01)
        s.update(mid, information="corrigido")
        depois = q.get_point("mem", mid)["payload"]
        self.assertEqual(depois["created_at"], antes, "created_at não pode ser reescrito")
        self.assertNotEqual(depois["updated_at"], antes)
        self.assertEqual(set(depois), CHAVES)

    def test_update_without_metadata_keeps_the_previous_one(self):
        s, q, _ = store()
        mid = s.store("x", {"type": "reference", "project": "p"})["id"]
        s.update(mid, information="y")
        self.assertEqual(q.get_point("mem", mid)["payload"]["metadata"],
                         {"type": "reference", "project": "p"})

    def test_update_without_text_keeps_the_document(self):
        s, q, _ = store()
        mid = s.store("texto original")["id"]
        s.update(mid, metadata={"type": "user"})
        self.assertEqual(q.get_point("mem", mid)["payload"]["document"], "texto original")

    def test_update_of_a_missing_id_does_not_create(self):
        s, q, _ = store()
        self.assertEqual(s.update("nao-existe", information="x")["status"], "not_found")
        self.assertEqual(len(q.collections["mem"]["pontos"]), 0)


class TestRefusedWrites(unittest.TestCase):
    def test_empty_memory(self):
        s, _, _ = store()
        for bad in ("", "   ", "\n"):
            with self.assertRaises(MemoryError_):
                s.store(bad)

    def test_batch_validates_before_writing_anything(self):
        s, q, _ = store()
        with self.assertRaises(MemoryError_):
            s.store_many([{"information": "válido"}, {"information": "  "}])
        self.assertEqual(len(q.collections["mem"]["pontos"]), 0,
                         "validação tem de ser ANTES de escrever, não durante")

    def test_batch_makes_ONE_embedding_call(self):
        s, _, emb = store()
        s.store_many([{"information": f"fato {i}"} for i in range(5)])
        self.assertEqual(len(emb.calls), 1, "o ganho do lote é uma ida à rede, não N")


class TestReadsNeverCreate(unittest.TestCase):
    """Ler não pode CRIAR coleção: com um typo no nome, a busca devolveria zero e o
    consumidor concluiria 'não há precedente' — a afirmação mais perigosa possível."""

    def _missing_collection(self):
        q, emb = FakeVectorStore(), FakeEmbedder()

        return MemoryStore(q, emb, None, "nao_existe", emb.dim), q

    def test_find_fails_loudly(self):
        s, q = self._missing_collection()
        with self.assertRaises(MemoryError_):
            s.find("x")
        self.assertEqual(q.list_collections(), [], "nada pode ter sido criado")

    def test_recall_fails_loudly(self):
        s, _ = self._missing_collection()
        with self.assertRaises(MemoryError_):
            s.recall(["x"], POL, top_k=5)

    def test_list_page_fails_loudly(self):
        s, _ = self._missing_collection()
        with self.assertRaises(MemoryError_):
            s.list_page()

    def test_store_MAY_create(self):
        s, q = self._missing_collection()
        s.store("primeiro fato")
        self.assertIn("nao_existe", q.list_collections(), "escrita cria, leitura não")


class TestRecall(unittest.TestCase):
    def _populate(self, reranker=None):
        s, q, _ = store(reranker)
        s.store("paginação do poll trunca em 100 itens", {"type": "reference"})
        s.store("o cursor regride quando o relógio recua", {"type": "reference"})
        s.store("receita de bolo de cenoura com cobertura", {"type": "user"})

        return s, q

    def test_retrieves_by_similarity_and_returns_Recalled(self):
        s, _ = self._populate()
        hits, outcome = s.recall(["paginação do poll trunca"], POL, top_k=10)
        self.assertTrue(hits)
        self.assertIsInstance(hits[0], core.Recalled)
        self.assertIn("paginação", hits[0].document,
                      "a consulta compartilha três palavras com este registro")
        self.assertEqual(hits[0].origin, "denso")

    def test_floor_relaxes_ONLY_when_a_reranker_exists(self):
        sem, _ = self._populate(None)
        com, _ = self._populate(FakeReranker(scores=[0.9, 0.8, 0.7]))
        _, f_sem = sem.recall(["poll"], POL, top_k=10)
        _, f_com = com.recall(["poll"], POL, top_k=10)
        self.assertGreaterEqual(f_com.candidates, f_sem.candidates,
                                "com segundo estágio o primeiro pode ser mais generoso")

    def test_fusing_angles_does_not_duplicate(self):
        s, _ = self._populate()
        hits, _ = s.recall(["paginação do poll", "poll paginação", "trunca 100 itens"],
                           POL, top_k=10)
        ids = [h.id for h in hits]
        self.assertEqual(len(ids), len(set(ids)))

    def test_best_dense_reports_the_best_of_ALL_hits(self):
        """Para dizer 'nada passou do corte, o melhor foi X', o número útil é o
        melhor de todos — não o melhor dos que passaram o piso."""
        s, _ = self._populate()
        estrito = retrieval.Policy(0.99, 0.99, 0.10, 6, veto=True)
        hits, outcome = s.recall(["assunto totalmente ausente xyz"], estrito, top_k=10)
        self.assertEqual(hits, [])
        self.assertGreater(outcome.best_dense, 0.0)

    def test_record_without_a_document_is_discarded(self):
        s, q, _ = store()
        q.upsert("mem", [{"id": "vazio", "vector": [1.0] * 8,
                          "payload": {"document": "   ", "metadata": {}}}])
        hits, _ = s.recall(["qualquer"], retrieval.Policy(0.0, 0.0, 0.10, 6), top_k=10)
        self.assertEqual([h.id for h in hits], [],
                         "vetor sem texto não pode ser julgado nem apresentado")

    def test_embedding_failure_propagates_as_a_domain_error(self):
        q = FakeVectorStore()
        q.ensure_collection("mem", 8)
        s = MemoryStore(q, FailingFakeEmbedder(core.EmbeddingError("fora")), None, "mem", 8)
        with self.assertRaises(core.CoreError):
            s.recall(["x"], POL, top_k=5)


class TestDocsOffline(unittest.TestCase):
    def _index(self, reranker=None):
        q, emb = FakeVectorStore(), FakeEmbedder()

        return DocIndex(q, emb, reranker, "tmp", "lib", emb.dim), q

    def _write_file(self, content="# Titulo\n\ncorpo do documento aqui\n"):
        import tempfile
        d = tempfile.mkdtemp()
        path = os.path.join(d, "doc.md")
        with open(path, "w") as fh:
            fh.write(content)

        return path

    def test_temporary_carries_an_expiry_and_library_does_not(self):
        idx, q = self._index()
        path = self._write_file()
        idx.index_file(path, ttl_seconds=60)
        idx.keep_file(path)
        tmp = list(q.collections["tmp"]["pontos"].values())[0]["payload"]
        lib = list(q.collections["lib"]["pontos"].values())[0]["payload"]
        self.assertIn("expires_at_ts", tmp)
        self.assertNotIn("expires_at_ts", lib, "biblioteca não expira, por construção")

    def test_reindexing_replaces_instead_of_accumulating(self):
        idx, q = self._index()
        path = self._write_file()
        idx.keep_file(path)
        antes = len(q.collections["lib"]["pontos"])
        idx.keep_file(path)
        self.assertEqual(len(q.collections["lib"]["pontos"]), antes)

    def test_sweep_removes_only_what_expired(self):
        idx, q = self._index()
        idx.index_file(self._write_file(), ttl_seconds=-1)
        idx.index_file(self._write_file("outro conteudo aqui\n"), ttl_seconds=600)
        idx.sweep()
        remaining_docs = [p["payload"]["document"]
                     for p in q.collections["tmp"]["pontos"].values()]
        self.assertTrue(remaining_docs, "o de 600s tem de sobreviver")
        self.assertTrue(all("outro" in d for d in remaining_docs),
                        "só o vencido é removido")

    def test_purging_temporary_does_not_touch_the_library(self):
        idx, q = self._index()
        path = self._write_file()
        idx.index_file(path, ttl_seconds=600)
        idx.keep_file(path)
        idx.drop_all_tmp()
        self.assertNotIn("tmp", q.list_collections())
        self.assertIn("lib", q.list_collections())

    def test_binary_is_refused(self):
        idx, _ = self._index()
        with self.assertRaises(core.DocsError):
            idx.index_file(self._write_file("texto\x00binario"))

    def test_line_range_reproduces_the_chunk(self):
        content = "\n".join(f"linha {i}" for i in range(1, 21)) + "\n"
        path = self._write_file(content)
        idx, q = self._index()
        idx.keep_file(path)
        lines = content.splitlines()
        for p in q.collections["lib"]["pontos"].values():
            md, doc = p["payload"]["metadata"], p["payload"]["document"]
            slice_text = "\n".join(lines[md["start_line"] - 1:md["end_line"]])
            self.assertEqual(slice_text.strip("\n"), doc,
                             "é o contrato do modo localizador")


if __name__ == "__main__":
    unittest.main(verbosity=2)
