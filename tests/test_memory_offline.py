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
from tests.fakes import (FakeEmbedder, FakeEmbedderQueQuebra, FakeReranker,
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


class TestFormaDoPayload(unittest.TestCase):
    """Compatibilidade com o acervo já gravado — a propriedade que sustenta a troca."""

    def test_store_grava_exatamente_as_quatro_chaves(self):
        s, q, _ = store()
        r = s.store("um fato", {"type": "reference"})
        ponto = q.get_point("mem", r["id"])
        self.assertEqual(set(ponto["payload"]), CHAVES)

    def test_store_many_grava_as_mesmas_chaves(self):
        s, q, _ = store()
        r = s.store_many([{"information": "a"}, {"information": "b", "metadata": {"x": 1}}])
        for mid in r["ids"]:
            self.assertEqual(set(q.get_point("mem", mid)["payload"]), CHAVES)

    def test_update_preserva_created_at(self):
        s, q, _ = store()
        mid = s.store("original")["id"]
        antes = q.get_point("mem", mid)["payload"]["created_at"]
        time.sleep(0.01)
        s.update(mid, information="corrigido")
        depois = q.get_point("mem", mid)["payload"]
        self.assertEqual(depois["created_at"], antes, "created_at não pode ser reescrito")
        self.assertNotEqual(depois["updated_at"], antes)
        self.assertEqual(set(depois), CHAVES)

    def test_update_sem_metadata_preserva_a_anterior(self):
        s, q, _ = store()
        mid = s.store("x", {"type": "reference", "project": "p"})["id"]
        s.update(mid, information="y")
        self.assertEqual(q.get_point("mem", mid)["payload"]["metadata"],
                         {"type": "reference", "project": "p"})

    def test_update_sem_texto_preserva_o_documento(self):
        s, q, _ = store()
        mid = s.store("texto original")["id"]
        s.update(mid, metadata={"type": "user"})
        self.assertEqual(q.get_point("mem", mid)["payload"]["document"], "texto original")

    def test_update_de_id_inexistente_nao_cria(self):
        s, q, _ = store()
        self.assertEqual(s.update("nao-existe", information="x")["status"], "not_found")
        self.assertEqual(len(q.colecoes["mem"]["pontos"]), 0)


class TestEscritaRecusada(unittest.TestCase):
    def test_memoria_vazia(self):
        s, _, _ = store()
        for ruim in ("", "   ", "\n"):
            with self.assertRaises(MemoryError_):
                s.store(ruim)

    def test_lote_valida_antes_de_gravar_qualquer_coisa(self):
        s, q, _ = store()
        with self.assertRaises(MemoryError_):
            s.store_many([{"information": "válido"}, {"information": "  "}])
        self.assertEqual(len(q.colecoes["mem"]["pontos"]), 0,
                         "validação tem de ser ANTES de escrever, não durante")

    def test_lote_faz_UMA_chamada_de_embedding(self):
        s, _, emb = store()
        s.store_many([{"information": f"fato {i}"} for i in range(5)])
        self.assertEqual(len(emb.chamadas), 1, "o ganho do lote é uma ida à rede, não N")


class TestLeituraNaoCria(unittest.TestCase):
    """Ler não pode CRIAR coleção: com um typo no nome, a busca devolveria zero e o
    consumidor concluiria 'não há precedente' — a afirmação mais perigosa possível."""

    def _sem_colecao(self):
        q, emb = FakeVectorStore(), FakeEmbedder()

        return MemoryStore(q, emb, None, "nao_existe", emb.dim), q

    def test_find_falha_alto(self):
        s, q = self._sem_colecao()
        with self.assertRaises(MemoryError_):
            s.find("x")
        self.assertEqual(q.list_collections(), [], "nada pode ter sido criado")

    def test_recall_falha_alto(self):
        s, _ = self._sem_colecao()
        with self.assertRaises(MemoryError_):
            s.recall(["x"], POL, top_k=5)

    def test_list_page_falha_alto(self):
        s, _ = self._sem_colecao()
        with self.assertRaises(MemoryError_):
            s.list_page()

    def test_store_PODE_criar(self):
        s, q = self._sem_colecao()
        s.store("primeiro fato")
        self.assertIn("nao_existe", q.list_collections(), "escrita cria, leitura não")


class TestRecall(unittest.TestCase):
    def _povoa(self, reranker=None):
        s, q, _ = store(reranker)
        s.store("paginação do poll trunca em 100 itens", {"type": "reference"})
        s.store("o cursor regride quando o relógio recua", {"type": "reference"})
        s.store("receita de bolo de cenoura com cobertura", {"type": "user"})

        return s, q

    def test_recupera_por_similaridade_e_devolve_Recalled(self):
        s, _ = self._povoa()
        hits, fora = s.recall(["paginação do poll trunca"], POL, top_k=10)
        self.assertTrue(hits)
        self.assertIsInstance(hits[0], core.Recalled)
        self.assertIn("paginação", hits[0].document,
                      "a consulta compartilha três palavras com este registro")
        self.assertEqual(hits[0].origem, "denso")

    def test_piso_relaxa_SO_quando_ha_reranker(self):
        sem, _ = self._povoa(None)
        com, _ = self._povoa(FakeReranker(scores=[0.9, 0.8, 0.7]))
        _, f_sem = sem.recall(["poll"], POL, top_k=10)
        _, f_com = com.recall(["poll"], POL, top_k=10)
        self.assertGreaterEqual(f_com.candidates, f_sem.candidates,
                                "com segundo estágio o primeiro pode ser mais generoso")

    def test_fusao_de_angulos_nao_duplica(self):
        s, _ = self._povoa()
        hits, _ = s.recall(["paginação do poll", "poll paginação", "trunca 100 itens"],
                           POL, top_k=10)
        ids = [h.id for h in hits]
        self.assertEqual(len(ids), len(set(ids)))

    def test_best_dense_reporta_o_melhor_de_TODOS_os_hits(self):
        """Para dizer 'nada passou do corte, o melhor foi X', o número útil é o
        melhor de todos — não o melhor dos que passaram o piso."""
        s, _ = self._povoa()
        estrito = retrieval.Policy(0.99, 0.99, 0.10, 6, veto=True)
        hits, fora = s.recall(["assunto totalmente ausente xyz"], estrito, top_k=10)
        self.assertEqual(hits, [])
        self.assertGreater(fora.best_dense, 0.0)

    def test_registro_sem_documento_e_descartado(self):
        s, q, _ = store()
        q.upsert("mem", [{"id": "vazio", "vector": [1.0] * 8,
                          "payload": {"document": "   ", "metadata": {}}}])
        hits, _ = s.recall(["qualquer"], retrieval.Policy(0.0, 0.0, 0.10, 6), top_k=10)
        self.assertEqual([h.id for h in hits], [],
                         "vetor sem texto não pode ser julgado nem apresentado")

    def test_erro_de_embedding_propaga_como_erro_do_dominio(self):
        q = FakeVectorStore()
        q.ensure_collection("mem", 8)
        s = MemoryStore(q, FakeEmbedderQueQuebra(core.EmbeddingError("fora")), None, "mem", 8)
        with self.assertRaises(core.CoreError):
            s.recall(["x"], POL, top_k=5)


class TestDocsOffline(unittest.TestCase):
    def _index(self, reranker=None):
        q, emb = FakeVectorStore(), FakeEmbedder()

        return DocIndex(q, emb, reranker, "tmp", "lib", emb.dim), q

    def _arquivo(self, conteudo="# Titulo\n\ncorpo do documento aqui\n"):
        import tempfile
        d = tempfile.mkdtemp()
        caminho = os.path.join(d, "doc.md")
        with open(caminho, "w") as fh:
            fh.write(conteudo)

        return caminho

    def test_temporario_carrega_validade_e_biblioteca_nao(self):
        idx, q = self._index()
        caminho = self._arquivo()
        idx.index_file(caminho, ttl_seconds=60)
        idx.keep_file(caminho)
        tmp = list(q.colecoes["tmp"]["pontos"].values())[0]["payload"]
        lib = list(q.colecoes["lib"]["pontos"].values())[0]["payload"]
        self.assertIn("expires_at_ts", tmp)
        self.assertNotIn("expires_at_ts", lib, "biblioteca não expira, por construção")

    def test_reindexar_substitui_em_vez_de_acumular(self):
        idx, q = self._index()
        caminho = self._arquivo()
        idx.keep_file(caminho)
        antes = len(q.colecoes["lib"]["pontos"])
        idx.keep_file(caminho)
        self.assertEqual(len(q.colecoes["lib"]["pontos"]), antes)

    def test_sweep_remove_so_o_vencido(self):
        idx, q = self._index()
        idx.index_file(self._arquivo(), ttl_seconds=-1)
        idx.index_file(self._arquivo("outro conteudo aqui\n"), ttl_seconds=600)
        idx.sweep()
        restantes = [p["payload"]["document"]
                     for p in q.colecoes["tmp"]["pontos"].values()]
        self.assertTrue(restantes, "o de 600s tem de sobreviver")
        self.assertTrue(all("outro" in d for d in restantes),
                        "só o vencido é removido")

    def test_purge_do_temporario_nao_toca_a_biblioteca(self):
        idx, q = self._index()
        caminho = self._arquivo()
        idx.index_file(caminho, ttl_seconds=600)
        idx.keep_file(caminho)
        idx.drop_all_tmp()
        self.assertNotIn("tmp", q.list_collections())
        self.assertIn("lib", q.list_collections())

    def test_binario_e_recusado(self):
        idx, _ = self._index()
        with self.assertRaises(core.DocsError):
            idx.index_file(self._arquivo("texto\x00binario"))

    def test_intervalo_de_linhas_reproduz_o_trecho(self):
        conteudo = "\n".join(f"linha {i}" for i in range(1, 21)) + "\n"
        caminho = self._arquivo(conteudo)
        idx, q = self._index()
        idx.keep_file(caminho)
        linhas = conteudo.splitlines()
        for p in q.colecoes["lib"]["pontos"].values():
            md, doc = p["payload"]["metadata"], p["payload"]["document"]
            recorte = "\n".join(linhas[md["start_line"] - 1:md["end_line"]])
            self.assertEqual(recorte.strip("\n"), doc,
                             "é o contrato do modo localizador")


if __name__ == "__main__":
    unittest.main(verbosity=2)
