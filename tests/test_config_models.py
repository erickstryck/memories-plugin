"""Testes de configuração e de escala de score.

As duas coisas aqui são armadilhas de falha SILENCIOSA — não dão erro, só entregam
resultado pior. Por isso têm teste: é a única forma de notar.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config as cfgmod
from core.models import normalize_scores, sigmoid


class TestPrecedencia(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.arquivo = Path(self.tmp.name) / "config.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_quando_nao_ha_nada(self):
        cfg = cfgmod.load(self.arquivo, env={})
        self.assertEqual(cfg.embed_model, "bge-m3")
        self.assertEqual(cfg.vector_size, 1024)
        self.assertEqual(cfg.memory_collection, "", "memória nasce vazia de propósito")

    def test_arquivo_vence_default(self):
        self.arquivo.write_text(json.dumps({"embed_model": "outro-modelo"}))
        self.assertEqual(cfgmod.load(self.arquivo, env={}).embed_model, "outro-modelo")

    def test_env_vence_arquivo(self):
        self.arquivo.write_text(json.dumps({"embed_model": "do-arquivo"}))
        cfg = cfgmod.load(self.arquivo, env={"QCTX_EMBED_MODEL": "do-ambiente"})
        self.assertEqual(cfg.embed_model, "do-ambiente")

    def test_alias_legado_e_aceito(self):
        cfg = cfgmod.load(self.arquivo, env={"SERVER_BASE_URL": "http://x/v1",
                                             "QDRANT_SERVICE_API_KEY": "k"})
        self.assertEqual(cfg.api_base_url, "http://x/v1")
        self.assertEqual(cfg.qdrant_api_key, "k")

    def test_nome_canonico_vence_o_legado(self):
        cfg = cfgmod.load(self.arquivo, env={"QCTX_QDRANT_URL": "canonico",
                                             "QDRANT_URL": "legado"})
        self.assertEqual(cfg.qdrant_url, "canonico")

    def test_save_preserva_as_outras_chaves(self):
        cfgmod.save({"embed_model": "a"}, self.arquivo)
        cfgmod.save({"memory_collection": "b"}, self.arquivo)
        dados = json.loads(self.arquivo.read_text())
        self.assertEqual(dados["embed_model"], "a")
        self.assertEqual(dados["memory_collection"], "b")

    def test_save_recusa_chave_desconhecida(self):
        with self.assertRaises(cfgmod.ConfigError):
            cfgmod.save({"chave_inventada": 1}, self.arquivo)

    def test_save_recusa_segredo_e_aponta_o_ambiente(self):
        """Segredo em arquivo de texto entra em backup e em sync de dotfiles."""
        for campo in ("qdrant_api_key", "api_key"):
            with self.assertRaises(cfgmod.ConfigError) as ctx:
                cfgmod.save({campo: "valor-secreto"}, self.arquivo)
            self.assertIn("QCTX_", str(ctx.exception), "a mensagem tem de dizer ONDE colocar")
            self.assertNotIn("valor-secreto", str(ctx.exception), "nem no erro o valor aparece")
        self.assertFalse(self.arquivo.exists(), "nada foi gravado")

    def test_save_de_campo_normal_continua_funcionando(self):
        cfgmod.save({"memory_collection": "x"}, self.arquivo)
        self.assertEqual(cfgmod.read_file(self.arquivo)["memory_collection"], "x")


class TestUrlDerivada(unittest.TestCase):
    def _cfg(self, **kw):
        base = dict(qdrant_url="", qdrant_api_key="", api_base_url="", api_key="",
                    embed_url="", rerank_url="", embed_model="m", rerank_model="r",
                    memory_collection="", docs_collection="d", library_collection="l",
                    vector_size=1024)
        base.update(kw)

        return cfgmod.Config(**base)

    def test_url_completa_tem_prioridade(self):
        cfg = self._cfg(embed_url="http://direto/v1/embeddings", api_base_url="http://base/v1")
        self.assertEqual(cfg.resolved_embed_url(), "http://direto/v1/embeddings")

    def test_deriva_da_base_quando_nao_ha_completa(self):
        cfg = self._cfg(api_base_url="http://base/v1/")
        self.assertEqual(cfg.resolved_embed_url(), "http://base/v1/embeddings")
        self.assertEqual(cfg.resolved_rerank_url(), "http://base/v1/rerank")

    def test_sem_nenhuma_das_duas_levanta(self):
        with self.assertRaises(cfgmod.ConfigError):
            self._cfg().resolved_embed_url()


class TestColisaoDeColecoes(unittest.TestCase):
    """Cada colisão degrada em silêncio, então a guarda tem de ser dura."""

    def _cfg(self, mem, docs, lib):
        return cfgmod.Config(qdrant_url="q", qdrant_api_key="", api_base_url="b", api_key="",
                             embed_url="", rerank_url="", embed_model="m", rerank_model="r",
                             memory_collection=mem, docs_collection=docs,
                             library_collection=lib, vector_size=1024)

    def test_documentos_na_colecao_de_memoria_e_recusado(self):
        with self.assertRaises(cfgmod.ConfigError):
            self._cfg("mesma", "mesma", "lib").require_docs_collection()

    def test_biblioteca_na_colecao_temporaria_e_recusada(self):
        with self.assertRaises(cfgmod.ConfigError):
            self._cfg("mem", "igual", "igual").require_library_collection()

    def test_biblioteca_na_colecao_de_memoria_e_recusada(self):
        with self.assertRaises(cfgmod.ConfigError):
            self._cfg("igual", "docs", "igual").require_library_collection()

    def test_tres_distintas_passa(self):
        cfg = self._cfg("mem", "docs", "lib")
        self.assertEqual(cfg.require_docs_collection(), "docs")
        self.assertEqual(cfg.require_library_collection(), "lib")
        self.assertEqual(cfg.require_memory_collection(), "mem")

    def test_memoria_nao_configurada_levanta_com_instrucao(self):
        with self.assertRaises(cfgmod.ConfigError) as ctx:
            self._cfg("", "docs", "lib").require_memory_collection()
        self.assertIn("collections list", str(ctx.exception))


class TestEscalaDeScore(unittest.TestCase):
    """O mesmo modelo devolve sigmoid num servidor e logit cru noutro. Um corte
    calibrado numa escala é inócuo na outra, sem erro nenhum aparecer."""

    def test_sigmoid_passa_intacto(self):
        pares = [(0, 0.9), (1, 0.05), (2, 0.5)]
        saida, era_logit = normalize_scores(list(pares))
        self.assertFalse(era_logit)
        self.assertEqual(saida, pares)

    def test_logit_e_convertido(self):
        # -11.04 é logit(1.6e-05): o valor medido para o mesmo documento
        # irrelevante nos dois servidores.
        saida, era_logit = normalize_scores([(0, 5.5), (1, -11.04)])
        self.assertTrue(era_logit)
        self.assertAlmostEqual(saida[1][1], 1.6e-05, places=6)
        self.assertGreater(saida[0][1], 0.99)

    def test_equivalencia_do_corte(self):
        """sigmoid 0.10 <=> logit -2.197: o corte calibrado tem de transferir."""
        self.assertAlmostEqual(sigmoid(-2.1972), 0.10, places=4)

    def test_um_negativo_basta_para_detectar_logit(self):
        saida, era_logit = normalize_scores([(0, 0.8), (1, -0.5)])
        self.assertTrue(era_logit, "faixa fora de [0,1] em QUALQUER par indica logit")

    def test_sigmoid_nao_estoura_com_valor_extremo(self):
        self.assertAlmostEqual(sigmoid(-1000.0), 0.0, places=10)
        self.assertAlmostEqual(sigmoid(1000.0), 1.0, places=10)

    def test_lista_vazia(self):
        self.assertEqual(normalize_scores([]), ([], False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
