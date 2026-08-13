"""Testes do fatiamento. Puro, sem rede — é o módulo que decide a qualidade da
busca, então é o que mais merece teste que MORDE."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.chunk import (Chunk, chunk_text, is_probably_binary, mode_for_suffix,
                        pack_chunks, split_blocks)


class TestFronteiras(unittest.TestCase):
    def test_heading_de_markdown_abre_bloco(self):
        linhas = ["intro\n", "# Titulo\n", "corpo\n"]
        self.assertIn(1, [ini for ini, _ in split_blocks(linhas)])

    def test_paragrafo_depois_de_linha_em_branco_abre_bloco(self):
        linhas = ["um\n", "\n", "dois\n"]
        inicios = [ini for ini, _ in split_blocks(linhas)]
        self.assertIn(2, inicios)

    def test_definicao_de_topo_em_codigo_abre_bloco(self):
        linhas = ["def a():\n", "    return 1\n", "def b():\n", "    return 2\n"]
        inicios = [ini for ini, _ in split_blocks(linhas)]
        self.assertIn(2, inicios, "linha na coluna 0 depois de linha indentada é fronteira")

    def test_arquivo_vazio_nao_produz_bloco(self):
        self.assertEqual(split_blocks([]), [])

    def test_blocos_cobrem_o_arquivo_inteiro_sem_lacuna(self):
        linhas = [f"linha {i}\n" for i in range(20)]
        linhas[5] = "\n"
        linhas[12] = "# secao\n"
        blocos = split_blocks(linhas)
        self.assertEqual(blocos[0][0], 0)
        self.assertEqual(blocos[-1][1], len(linhas))
        for anterior, seguinte in zip(blocos, blocos[1:]):
            self.assertEqual(anterior[1], seguinte[0], "fim de um bloco é início do próximo")


class TestEmpacotamento(unittest.TestCase):
    def test_linhas_sao_1_based_e_inclusivas(self):
        trechos = pack_chunks(["a\n", "b\n", "c\n"], target=10_000)
        self.assertEqual(len(trechos), 1)
        self.assertEqual((trechos[0].start_line, trechos[0].end_line), (1, 3))

    def test_texto_do_trecho_corresponde_as_linhas_declaradas(self):
        linhas = [f"L{i}\n" for i in range(1, 31)]
        linhas[9] = "\n"
        for trecho in pack_chunks(linhas, target=40):
            esperado = "".join(linhas[trecho.start_line - 1:trecho.end_line]).strip("\n")
            self.assertEqual(trecho.text, esperado,
                             "intervalo de linhas tem de reproduzir o texto — é o contrato "
                             "do modo localizador, que manda alguém reler essa região")

    def test_bloco_gigante_cai_na_janela_fixa(self):
        gigante = ["x" * 200 + "\n" for _ in range(100)]  # 20k chars num só bloco
        trechos = pack_chunks(gigante, target=2000, hard_max=4000)
        self.assertGreater(len(trechos), 1, "bloco acima do teto tem de ser dividido")
        for t in trechos:
            self.assertLessEqual(len(t.text), 5000)

    def test_janela_fixa_tem_sobreposicao(self):
        gigante = ["y" * 300 + "\n" for _ in range(40)]
        trechos = pack_chunks(gigante, target=1500, hard_max=2000)
        pares = list(zip(trechos, trechos[1:]))
        self.assertTrue(any(b.start_line <= a.end_line for a, b in pares),
                        "janelas consecutivas têm de se sobrepor para não perder a costura")

    def test_trecho_so_com_espaco_em_branco_e_descartado(self):
        self.assertEqual(pack_chunks(["\n", "   \n", "\t\n"]), [])

    def test_respeita_o_alvo_de_tamanho(self):
        linhas = []
        for i in range(60):
            linhas += [f"paragrafo {i}\n", "\n"]
        for t in pack_chunks(linhas, target=200, hard_max=1000):
            self.assertLessEqual(len(t.text), 1000)

    def test_nenhum_trecho_perde_conteudo_significativo(self):
        linhas = [f"conteudo unico {i}\n" for i in range(50)]
        trechos = pack_chunks(linhas, target=100)
        juntos = "\n".join(t.text for t in trechos)
        for i in range(50):
            self.assertIn(f"conteudo unico {i}", juntos)


class TestModo(unittest.TestCase):
    def test_extensao_de_texto_vira_localizador(self):
        for suf in (".py", ".md", ".ts", ".LOG"):
            self.assertEqual(mode_for_suffix(suf), "locator")

    def test_extensao_desconhecida_vira_foto(self):
        for suf in (".pdf", ".docx", ""):
            self.assertEqual(mode_for_suffix(suf), "snapshot")


class TestBinario(unittest.TestCase):
    def test_detecta_nulo(self):
        self.assertTrue(is_probably_binary("abc\x00def"))
        self.assertFalse(is_probably_binary("texto normal com acento é"))


class TestChunkText(unittest.TestCase):
    def test_documento_realista_de_markdown(self):
        doc = "\n".join(
            f"## Secao {i}\n\nConteudo da secao {i}. " + ("palavra " * 40)
            for i in range(10)
        )
        trechos = chunk_text(doc, target=600, hard_max=2000)
        self.assertGreater(len(trechos), 3)
        self.assertTrue(all(isinstance(t, Chunk) for t in trechos))
        self.assertEqual(trechos[0].start_line, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
