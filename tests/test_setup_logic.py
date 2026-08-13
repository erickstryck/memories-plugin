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


class TestChoice(unittest.TestCase):
    def setUp(self):
        self.options = ["claude_memory", "hermes_memory", "outra"]

    def test_a_number_selects_by_position(self):
        self.assertEqual(choose_by_index(self.options, "2"), "hermes_memory")

    def test_numbering_starts_at_1(self):
        self.assertEqual(choose_by_index(self.options, "1"), "claude_memory")

    def test_empty_means_keep_the_current_value(self):
        for entry in ("", "   ", None):
            self.assertIsNone(choose_by_index(self.options, entry))

    def test_a_number_outside_the_list_selects_nothing(self):
        for entry in ("0", "4", "99"):
            self.assertIsNone(choose_by_index(self.options, entry),
                              "índice inválido não pode virar nome de coleção")

    def test_a_typed_name_is_accepted_even_if_not_listed(self):
        self.assertEqual(choose_by_index(self.options, "colecao_nova"), "colecao_nova")

    def test_surrounding_whitespace_is_ignored(self):
        self.assertEqual(choose_by_index(self.options, "  2  "), "hermes_memory")
        self.assertEqual(choose_by_index(self.options, " nome "), "nome")

    def test_empty_list_with_a_number(self):
        self.assertIsNone(choose_by_index([], "1"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
