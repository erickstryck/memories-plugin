"""Testes da lógica do wizard.

Só `escolher_por_indice` tem lógica; o resto de `setup` é I/O de rede e leitura de
terminal. Testar a parte pura e manter a casca trivial é o que torna o wizard
confiável sem precisar de um TTY no teste.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.setup import choose_by_index


class TestEscolha(unittest.TestCase):
    def setUp(self):
        self.options = ["claude_memory", "hermes_memory", "outra"]

    def test_numero_seleciona_pela_posicao(self):
        self.assertEqual(choose_by_index(self.options, "2"), "hermes_memory")

    def test_numeracao_comeca_em_1(self):
        self.assertEqual(choose_by_index(self.options, "1"), "claude_memory")

    def test_vazio_significa_manter_o_atual(self):
        for entry in ("", "   ", None):
            self.assertIsNone(choose_by_index(self.options, entry))

    def test_numero_fora_da_lista_nao_seleciona(self):
        for entry in ("0", "4", "99"):
            self.assertIsNone(choose_by_index(self.options, entry),
                              "índice inválido não pode virar nome de coleção")

    def test_nome_digitado_e_aceito_mesmo_fora_da_lista(self):
        self.assertEqual(choose_by_index(self.options, "colecao_nova"), "colecao_nova")

    def test_espaco_em_volta_e_ignorado(self):
        self.assertEqual(choose_by_index(self.options, "  2  "), "hermes_memory")
        self.assertEqual(choose_by_index(self.options, " nome "), "nome")

    def test_lista_vazia_com_numero(self):
        self.assertIsNone(choose_by_index([], "1"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
