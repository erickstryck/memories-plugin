"""Testes da preparação de consulta e do disjuntor. Lógica pura, sem rede."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.breaker import Breaker
from core.query import angles, longest_sentence, skip_reason, content_words


class TestPular(unittest.TestCase):
    def test_prompt_curto_e_pulado(self):
        for p in ("oi", "ok", "e?", "sim"):
            self.assertIsNotNone(skip_reason(p))

    def test_confirmacao_de_uma_palavra_e_pulada(self):
        for p in ("prossiga", "continue", "obrigado", "perfeito"):
            self.assertIsNotNone(skip_reason(p))

    def test_confirmacao_de_VARIAS_palavras_e_pulada(self):
        """Estas passam do tamanho mínimo — é o caso que a checagem existe para pegar,
        e que ficava inalcançável quando ela comparava o texto inteiro."""
        for p in ("ok, pode continuar", "beleza, segue", "sim, perfeito",
                  "ok obrigado", "isso, pode fazer"):
            self.assertEqual(skip_reason(p), "prompt trivial", p)

    def test_confirmacao_com_UMA_palavra_de_conteudo_nao_e_pulada(self):
        """"pode continuar o poll" tem assunto: `poll`. Não pode ser descartado."""
        self.assertIsNone(skip_reason("ok, pode continuar o poll"))
        self.assertIsNone(skip_reason("sim, e a paginação?"))

    def test_pontuacao_nao_escapa_do_filtro(self):
        self.assertIsNotNone(skip_reason("continue!"))
        self.assertIsNotNone(skip_reason("  ok.  "))

    def test_comando_sem_argumento_e_pulado(self):
        self.assertEqual(skip_reason("/memoria-status"), "comando sem argumento")

    def test_comando_com_argumento_nao_e_pulado(self):
        self.assertIsNone(skip_reason("/buscar paginação do poll"))

    def test_pergunta_de_verdade_passa(self):
        self.assertIsNone(skip_reason("como funciona a paginação do poll?"))

    def test_none_e_vazio_nao_quebram(self):
        self.assertIsNotNone(skip_reason(None))
        self.assertIsNotNone(skip_reason(""))


class TestAngulos(unittest.TestCase):
    def test_texto_cru_e_sempre_o_primeiro(self):
        p = "como funciona a paginação do poll no conector?"
        self.assertEqual(angles(p)[0], p)

    def test_angulo_de_conteudo_remove_estrutura(self):
        content = content_words("como funciona a paginação do poll no conector?")
        for estrutura in ("como", "a", "do", "no"):
            self.assertNotIn(f" {estrutura} ", f" {content} ")
        for termo in ("funciona", "paginação", "poll", "conector"):
            self.assertIn(termo, content)

    def test_palavras_repetidas_aparecem_uma_vez(self):
        self.assertEqual(content_words("poll poll poll cursor"), "poll cursor")

    def test_frase_mais_longa_so_com_multiplas_frases(self):
        self.assertEqual(longest_sentence("uma frase só"), "")
        longa = longest_sentence("curta. esta aqui é bem mais longa que a outra. fim")
        self.assertIn("bem mais longa", longa)

    def test_sem_duplicata_entre_angulos(self):
        for p in ("paginação poll cursor", "ABC DEF GHI", "termo"):
            a = angles(p)
            self.assertEqual(len(a), len(set(a)), "embedar o mesmo texto duas vezes é gasto puro")

    def test_no_maximo_tres_angulos(self):
        p = ("primeira frase com algum conteudo tecnico. segunda frase bem mais longa "
             "que a primeira e com outros termos relevantes. terceira.")
        self.assertLessEqual(len(angles(p)), 3)

    def test_respeita_o_limite_de_chars(self):
        for a in angles("x" * 5000, limite_chars=100):
            self.assertLessEqual(len(a), 100)


class TestBreaker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "breaker"

    def tearDown(self):
        self.tmp.cleanup()

    def test_arquivo_ausente_permite_tentar(self):
        self.assertIsNone(Breaker(self.path, 300).is_open())

    def test_depois_de_armar_fica_aberto(self):
        b = Breaker(self.path, 300)
        b.arm()
        ocioso = b.is_open()
        self.assertIsNotNone(ocioso)
        self.assertLess(ocioso, 5)

    def test_espera_vencida_permite_tentar_de_novo(self):
        b = Breaker(self.path, 0.05)
        b.arm()
        time.sleep(0.1)
        self.assertIsNone(b.is_open())

    def test_limpar_reabre_na_hora(self):
        b = Breaker(self.path, 300)
        b.arm()
        b.clear()
        self.assertIsNone(b.is_open())

    def test_cooldown_zero_desliga_o_disjuntor(self):
        b = Breaker(self.path, 0)
        b.arm()
        self.assertIsNone(b.is_open(), "cooldown 0 significa nunca bloquear")

    def test_conteudo_corrompido_permite_tentar(self):
        self.path.write_text("isto não é um timestamp")
        self.assertIsNone(Breaker(self.path, 300).is_open(),
                          "disjuntor com defeito não pode ser o motivo de a busca parar")

    def test_diretorio_inexistente_e_criado_ao_armar(self):
        fundo = Path(self.tmp.name) / "a" / "b" / "breaker"
        Breaker(fundo, 300).arm()
        self.assertTrue(fundo.exists())

    def test_armar_em_caminho_impossivel_nao_levanta(self):
        Breaker("/proc/impossivel/breaker", 300).arm()  # não pode explodir


if __name__ == "__main__":
    unittest.main(verbosity=2)
