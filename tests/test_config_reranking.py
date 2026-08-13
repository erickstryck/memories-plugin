"""Testes de configuração e da escala de score do re-rank.

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
from core.reranking import normalize_scores, sigmoid


class TestPrecedencia(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.file_path = Path(self.tmp.name) / "config.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_quando_nao_ha_nada(self):
        cfg = cfgmod.load(self.file_path, env={})
        self.assertEqual(cfg.embed_model, "bge-m3")
        self.assertEqual(cfg.vector_size, 1024)
        self.assertEqual(cfg.memory_collection, "", "memória nasce vazia de propósito")

    def test_arquivo_vence_default(self):
        self.file_path.write_text(json.dumps({"embed_model": "outro-modelo"}))
        self.assertEqual(cfgmod.load(self.file_path, env={}).embed_model, "outro-modelo")

    def test_env_vence_arquivo(self):
        self.file_path.write_text(json.dumps({"embed_model": "do-arquivo"}))
        cfg = cfgmod.load(self.file_path, env={"QCTX_EMBED_MODEL": "do-ambiente"})
        self.assertEqual(cfg.embed_model, "do-ambiente")

    def test_alias_legado_e_aceito(self):
        cfg = cfgmod.load(self.file_path, env={"SERVER_BASE_URL": "http://x/v1",
                                             "QDRANT_SERVICE_API_KEY": "k"})
        self.assertEqual(cfg.api_base_url, "http://x/v1")
        self.assertEqual(cfg.qdrant_api_key, "k")

    def test_nome_canonico_vence_o_legado(self):
        cfg = cfgmod.load(self.file_path, env={"QCTX_QDRANT_URL": "canonico",
                                             "QDRANT_URL": "legado"})
        self.assertEqual(cfg.qdrant_url, "canonico")

    def test_save_preserva_as_outras_chaves(self):
        cfgmod.save({"embed_model": "a"}, self.file_path)
        cfgmod.save({"memory_collection": "b"}, self.file_path)
        dados = json.loads(self.file_path.read_text())
        self.assertEqual(dados["embed_model"], "a")
        self.assertEqual(dados["memory_collection"], "b")

    def test_save_recusa_chave_desconhecida(self):
        with self.assertRaises(cfgmod.ConfigError):
            cfgmod.save({"chave_inventada": 1}, self.file_path)

    def test_save_recusa_segredo_e_aponta_o_ambiente(self):
        """Segredo em arquivo de texto entra em backup e em sync de dotfiles."""
        for field in ("qdrant_api_key", "api_key"):
            with self.assertRaises(cfgmod.ConfigError) as ctx:
                cfgmod.save({field: "valor-secreto"}, self.file_path)
            self.assertIn("QCTX_", str(ctx.exception), "a mensagem tem de dizer ONDE colocar")
            self.assertNotIn("valor-secreto", str(ctx.exception), "nem no erro o valor aparece")
        self.assertFalse(self.file_path.exists(), "nada foi gravado")

    def test_save_de_campo_normal_continua_funcionando(self):
        cfgmod.save({"memory_collection": "x"}, self.file_path)
        self.assertEqual(cfgmod.read_file(self.file_path)["memory_collection"], "x")


class TestUrlDerivada(unittest.TestCase):
    def _config(self, **kw):
        base = dict(qdrant_url="", qdrant_api_key="", api_base_url="", api_key="",
                    embed_url="", rerank_url="", embed_model="m", rerank_model="r",
                    memory_collection="", docs_collection="d", library_collection="l",
                    vector_size=1024)
        base.update(kw)

        return cfgmod.Config(**base)

    def test_url_completa_tem_prioridade(self):
        cfg = self._config(embed_url="http://direto/v1/embeddings", api_base_url="http://base/v1")
        self.assertEqual(cfg.resolved_embed_url(), "http://direto/v1/embeddings")

    def test_deriva_da_base_quando_nao_ha_completa(self):
        cfg = self._config(api_base_url="http://base/v1/")
        self.assertEqual(cfg.resolved_embed_url(), "http://base/v1/embeddings")
        self.assertEqual(cfg.resolved_rerank_url(), "http://base/v1/rerank")

    def test_sem_nenhuma_das_duas_levanta(self):
        with self.assertRaises(cfgmod.ConfigError):
            self._config().resolved_embed_url()


class TestColisaoDeColecoes(unittest.TestCase):
    """Cada colisão degrada em silêncio, então a guarda tem de ser dura."""

    def _config(self, mem, docs, lib):
        return cfgmod.Config(qdrant_url="q", qdrant_api_key="", api_base_url="b", api_key="",
                             embed_url="", rerank_url="", embed_model="m", rerank_model="r",
                             memory_collection=mem, docs_collection=docs,
                             library_collection=lib, vector_size=1024)

    def test_documentos_na_colecao_de_memoria_e_recusado(self):
        with self.assertRaises(cfgmod.ConfigError):
            self._config("mesma", "mesma", "lib").require_docs_collection()

    def test_biblioteca_na_colecao_temporaria_e_recusada(self):
        with self.assertRaises(cfgmod.ConfigError):
            self._config("mem", "igual", "igual").require_library_collection()

    def test_biblioteca_na_colecao_de_memoria_e_recusada(self):
        with self.assertRaises(cfgmod.ConfigError):
            self._config("igual", "docs", "igual").require_library_collection()

    def test_tres_distintas_passa(self):
        cfg = self._config("mem", "docs", "lib")
        self.assertEqual(cfg.require_docs_collection(), "docs")
        self.assertEqual(cfg.require_library_collection(), "lib")
        self.assertEqual(cfg.require_memory_collection(), "mem")

    def test_memoria_nao_configurada_levanta_com_instrucao(self):
        with self.assertRaises(cfgmod.ConfigError) as ctx:
            self._config("", "docs", "lib").require_memory_collection()
        self.assertIn("collections list", str(ctx.exception))


class TestEscalaDeScore(unittest.TestCase):
    """O mesmo modelo devolve sigmoid num servidor e logit cru noutro. Um corte
    calibrado numa escala é inócuo na outra, sem erro nenhum aparecer."""

    def test_sigmoid_passa_intacto(self):
        pairs = [(0, 0.9), (1, 0.05), (2, 0.5)]
        output, was_logit = normalize_scores(list(pairs))
        self.assertFalse(was_logit)
        self.assertEqual(output, pairs)

    def test_logit_e_convertido(self):
        # -11.04 é logit(1.6e-05): o valor medido para o mesmo documento
        # irrelevante nos dois servidores.
        output, was_logit = normalize_scores([(0, 5.5), (1, -11.04)])
        self.assertTrue(was_logit)
        self.assertAlmostEqual(output[1][1], 1.6e-05, places=6)
        self.assertGreater(output[0][1], 0.99)

    def test_equivalencia_do_corte(self):
        """sigmoid 0.10 <=> logit -2.197: o corte calibrado tem de transferir."""
        self.assertAlmostEqual(sigmoid(-2.1972), 0.10, places=4)

    def test_um_negativo_basta_para_detectar_logit(self):
        output, was_logit = normalize_scores([(0, 0.8), (1, -0.5)])
        self.assertTrue(was_logit, "faixa fora de [0,1] em QUALQUER par indica logit")

    def test_sigmoid_nao_estoura_com_valor_extremo(self):
        self.assertAlmostEqual(sigmoid(-1000.0), 0.0, places=10)
        self.assertAlmostEqual(sigmoid(1000.0), 1.0, places=10)

    def test_lista_vazia(self):
        self.assertEqual(normalize_scores([]), ([], False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
