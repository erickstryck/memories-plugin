"""Testes da lógica pura de `core.docs` — TTL, id de documento e detecção de
obsolescência. Nada de rede.

O teste de tolerância de mtime é REGRESSÃO de um bug real: a comparação era por
igualdade exata de float e o round-trip por JSON perdia os últimos bits, então o
aviso de "arquivo mudou" disparava em toda busca de todo documento e o `refresh`
reindexava o acervo inteiro a cada execução.
"""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.docs import DocsError, doc_id_for, parse_ttl, source_changed


class TestTTL(unittest.TestCase):
    def test_unidades(self):
        self.assertEqual(parse_ttl("30s"), 30)
        self.assertEqual(parse_ttl("30m"), 1800)
        self.assertEqual(parse_ttl("2h"), 7200)
        self.assertEqual(parse_ttl("7d"), 604800)

    def test_segundos_puros(self):
        self.assertEqual(parse_ttl("90"), 90)

    def test_aceita_espaco_e_maiuscula(self):
        self.assertEqual(parse_ttl(" 24H "), 86400)

    def test_invalido_levanta(self):
        for ruim in ("", "abc", "24x", "-5h"):
            with self.assertRaises(DocsError):
                parse_ttl(ruim)


class TestDocId(unittest.TestCase):
    def test_estavel_para_o_mesmo_caminho(self):
        self.assertEqual(doc_id_for("/tmp/a.md"), doc_id_for("/tmp/a.md"))

    def test_caminho_relativo_e_absoluto_dao_o_mesmo_id(self):
        cwd = os.getcwd()
        try:
            os.chdir("/tmp")
            self.assertEqual(doc_id_for("a.md"), doc_id_for("/tmp/a.md"),
                             "o id vem do caminho ABSOLUTO, para reindexar substituir")
        finally:
            os.chdir(cwd)

    def test_caminhos_diferentes_dao_ids_diferentes(self):
        self.assertNotEqual(doc_id_for("/tmp/a.md"), doc_id_for("/tmp/b.md"))


class TestObsolescencia(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.arquivo = Path(self.tmp.name) / "doc.md"
        self.arquivo.write_text("conteudo original\n")
        st = os.stat(self.arquivo)
        self.mtime = st.st_mtime
        self.size = st.st_size

    def tearDown(self):
        self.tmp.cleanup()

    def test_arquivo_intocado_nao_e_obsoleto(self):
        self.assertIsNone(source_changed(str(self.arquivo), self.mtime, self.size))

    def test_round_trip_de_float_nao_gera_falso_positivo(self):
        """O bug original: JSON devolvia ...9956775 para ...9956777 gravado."""
        quase = self.mtime + 2.4e-07
        self.assertIsNone(source_changed(str(self.arquivo), quase, self.size),
                          "diferença de 1e-7 é ruído de serialização, não edição")

    def test_mtime_arredondado_a_milissegundo_nao_gera_falso_positivo(self):
        self.assertIsNone(source_changed(str(self.arquivo), round(self.mtime, 3), self.size))

    def test_mudanca_de_tamanho_e_detectada(self):
        self.arquivo.write_text("conteudo original com mais texto\n")
        motivo = source_changed(str(self.arquivo), self.mtime, self.size)
        self.assertIsNotNone(motivo)
        self.assertIn("tamanho", motivo)

    def test_mudanca_de_mtime_com_mesmo_tamanho_e_detectada(self):
        futuro = self.mtime + 60
        os.utime(self.arquivo, (futuro, futuro))
        motivo = source_changed(str(self.arquivo), self.mtime, self.size)
        self.assertIsNotNone(motivo, "edição que preserva o tamanho ainda muda o mtime")

    def test_arquivo_removido_e_reportado(self):
        caminho = str(self.arquivo)
        os.unlink(caminho)
        self.assertEqual(source_changed(caminho, self.mtime, self.size),
                         "arquivo não existe mais")

    def test_metadado_ausente_nao_quebra(self):
        self.assertIsNone(source_changed(str(self.arquivo), None, None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
