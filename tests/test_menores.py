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
from tests.fakes import FakeEmbedder, FakeEmbedderQueQuebra, FakeVectorStore

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks"))


def carrega_hook():
    """Importa o hook com STATE_DIR apontando para um diretório temporário."""
    import importlib
    tmp = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = tmp
    import recall
    importlib.reload(recall)

    return recall, Path(tmp)


class TestPodaDoEstado(unittest.TestCase):
    """Entrada em `seen` só importa enquanto pode evitar uma reinjeção."""

    def test_descarta_o_que_nao_muda_mais_decisao(self):
        recall, _ = carrega_hook()
        estado = {"round": 20, "seen": {"recente": 19, "no_limite": 12, "velha": 5}}
        podadas = recall.poda_estado(estado)
        self.assertEqual(podadas, 2, "12 e 5 estão a >= 8 rodadas de distância")
        self.assertEqual(set(estado["seen"]), {"recente"})

    def test_mantem_o_que_ainda_evita_reinjecao(self):
        recall, _ = carrega_hook()
        estado = {"round": 10, "seen": {"a": 9, "b": 4}}  # 10-4 = 6 < 8
        recall.poda_estado(estado)
        self.assertEqual(set(estado["seen"]), {"a", "b"})

    def test_valor_corrompido_e_descartado(self):
        recall, _ = carrega_hook()
        estado = {"round": 5, "seen": {"ok": 4, "lixo": "nao é numero", "nulo": None}}
        recall.poda_estado(estado)
        self.assertEqual(set(estado["seen"]), {"ok"})

    def test_estado_vazio_nao_quebra(self):
        recall, _ = carrega_hook()
        self.assertEqual(recall.poda_estado({}), 0)


class TestLimpezaDeSessoesMortas(unittest.TestCase):
    def test_remove_antigo_e_preserva_recente(self):
        recall, dir_ = carrega_hook()
        antigo = dir_ / "recall-morta.json"
        recente = dir_ / "recall-viva.json"
        for f in (antigo, recente):
            f.write_text(json.dumps({"round": 1, "seen": {}}))
        velho = time.time() - 10 * 86400
        os.utime(antigo, (velho, velho))
        removidos = recall.limpa_sessoes_mortas(dias=7.0)
        self.assertEqual(removidos, 1)
        self.assertFalse(antigo.exists())
        self.assertTrue(recente.exists())

    def test_nao_toca_o_log(self):
        recall, dir_ = carrega_hook()
        log = dir_ / "recall.log"
        log.write_text("linha")
        velho = time.time() - 30 * 86400
        os.utime(log, (velho, velho))
        recall.limpa_sessoes_mortas(dias=1.0)
        self.assertTrue(log.exists(), "o log não é estado de sessão")


class TestEnvTolerante(unittest.TestCase):
    """Lido no carregamento do módulo, ANTES do catch-all — não pode explodir."""

    def test_numero_valido(self):
        recall, _ = carrega_hook()
        os.environ["QCTX_TESTE_NUM"] = "42"
        try:
            self.assertEqual(recall.env_num("QCTX_TESTE_NUM", "X", "7", int), 42)
        finally:
            del os.environ["QCTX_TESTE_NUM"]

    def test_numero_invalido_cai_no_default_e_registra(self):
        recall, _ = carrega_hook()
        recall._pendencias.clear()
        os.environ["QCTX_TESTE_NUM"] = "14k"
        try:
            self.assertEqual(recall.env_num("QCTX_TESTE_NUM", "X", "14000", int), 14000)
            self.assertTrue(any("14k" in p for p in recall._pendencias),
                            "o valor ruim tem de ficar registrado, não sumir")
        finally:
            del os.environ["QCTX_TESTE_NUM"]

    def test_ausente_usa_default(self):
        recall, _ = carrega_hook()
        self.assertAlmostEqual(recall.env_num("QCTX_NAO_EXISTE", "NEM_ESTE", "0.58"), 0.58)


class TestTtlPreservadoNoRefresh(unittest.TestCase):
    def _idx(self):
        q, emb = FakeVectorStore(), FakeEmbedder()

        return DocIndex(q, emb, None, "tmp", "lib", emb.dim), q

    def _arquivo(self, texto="conteudo do documento\n"):
        d = tempfile.mkdtemp()
        caminho = os.path.join(d, "doc.md")
        Path(caminho).write_text(texto)

        return caminho

    def test_duracao_e_guardada_na_indexacao(self):
        idx, q = self._idx()
        idx.index_file(self._arquivo(), ttl_seconds=3600)
        md = list(q.colecoes["tmp"]["pontos"].values())[0]["payload"]["metadata"]
        self.assertEqual(md["ttl_seconds"], 3600)

    def test_refresh_reusa_a_duracao_em_vez_do_default(self):
        idx, q = self._idx()
        caminho = self._arquivo()
        idx.index_file(caminho, ttl_seconds=3600)          # o usuário pediu 1 hora
        Path(caminho).write_text("conteudo alterado maior\n")  # força reindexação
        idx.refresh(scope="tmp")
        p = list(q.colecoes["tmp"]["pontos"].values())[0]["payload"]
        restante = p["expires_at_ts"] - time.time()
        self.assertLess(restante, 3700, "não pode virar o default de 24h")
        self.assertGreater(restante, 3500)
        self.assertNotAlmostEqual(restante, DEFAULT_TTL_SECONDS, delta=100)

    def test_biblioteca_nao_ganha_validade_no_refresh(self):
        idx, q = self._idx()
        caminho = self._arquivo()
        idx.keep_file(caminho)
        Path(caminho).write_text("outro conteudo bem diferente\n")
        idx.refresh(scope="library")
        for p in q.colecoes["lib"]["pontos"].values():
            self.assertNotIn("expires_at_ts", p["payload"])


class TestUpdateSemReembedding(unittest.TestCase):
    def _store(self, embedder=None):
        q = FakeVectorStore()
        emb = embedder or FakeEmbedder()
        q.ensure_collection("mem", 8)

        return MemoryStore(q, emb, None, "mem", 8), q, emb

    def test_metadata_sozinha_nao_chama_embedding(self):
        s, q, emb = self._store()
        mid = s.store("texto que não muda")["id"]
        chamadas_antes = len(emb.chamadas)
        res = s.update(mid, metadata={"type": "feedback"})
        self.assertFalse(res["reembedded"])
        self.assertEqual(len(emb.chamadas), chamadas_antes, "vetor idêntico não se recalcula")
        self.assertEqual(q.get_point("mem", mid)["payload"]["metadata"], {"type": "feedback"})

    def test_texto_igual_ao_anterior_tambem_nao_chama(self):
        s, _, emb = self._store()
        mid = s.store("mesmo texto")["id"]
        antes = len(emb.chamadas)
        res = s.update(mid, information="mesmo texto")
        self.assertFalse(res["reembedded"])
        self.assertEqual(len(emb.chamadas), antes)

    def test_texto_diferente_chama(self):
        s, _, emb = self._store()
        mid = s.store("original")["id"]
        antes = len(emb.chamadas)
        res = s.update(mid, information="mudou de verdade")
        self.assertTrue(res["reembedded"])
        self.assertEqual(len(emb.chamadas), antes + 1)

    def test_corrigir_etiqueta_FUNCIONA_com_embedding_fora(self):
        """O motivo real do conserto: a operação não depende de embedding, então não
        pode ficar impossível quando o endpoint está fora."""
        q = FakeVectorStore()
        q.ensure_collection("mem", 8)
        vivo = MemoryStore(q, FakeEmbedder(), None, "mem", 8)
        mid = vivo.store("fato gravado enquanto o endpoint funcionava")["id"]

        quebrado = MemoryStore(q, FakeEmbedderQueQuebra(core.EmbeddingError("fora do ar")),
                               None, "mem", 8)
        res = quebrado.update(mid, metadata={"type": "corrigido"})
        self.assertEqual(res["status"], "updated")
        self.assertEqual(q.get_point("mem", mid)["payload"]["metadata"], {"type": "corrigido"})

    def test_vetor_e_preservado_intacto(self):
        s, q, _ = self._store()
        mid = s.store("texto")["id"]
        vetor_antes = list(q.get_point("mem", mid)["vector"])
        s.update(mid, metadata={"x": 1})
        self.assertEqual(q.get_point("mem", mid)["vector"], vetor_antes)

    def test_as_quatro_chaves_sobrevivem_ao_caminho_sem_reembedding(self):
        s, q, _ = self._store()
        mid = s.store("texto", {"type": "a"})["id"]
        s.update(mid, metadata={"type": "b"})
        self.assertEqual(set(q.get_point("mem", mid)["payload"]),
                         {"document", "metadata", "created_at", "updated_at"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
