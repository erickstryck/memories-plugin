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


class TestSkipping(unittest.TestCase):
    def test_short_prompt_is_skipped(self):
        for p in ("oi", "ok", "e?", "sim"):
            self.assertIsNotNone(skip_reason(p))

    def test_one_word_confirmation_is_skipped(self):
        for p in ("prossiga", "continue", "obrigado", "perfeito"):
            self.assertIsNotNone(skip_reason(p))

    def test_MULTI_word_confirmation_is_skipped(self):
        """Estas passam do tamanho mínimo — é o caso que a checagem existe para pegar,
        e que ficava inalcançável quando ela comparava o texto inteiro."""
        for p in ("ok, pode continuar", "beleza, segue", "sim, perfeito",
                  "ok obrigado", "isso, pode fazer"):
            self.assertEqual(skip_reason(p), "prompt trivial", p)

    def test_confirmation_with_ONE_content_word_is_not_skipped(self):
        """"pode continuar o poll" tem assunto: `poll`. Não pode ser descartado."""
        self.assertIsNone(skip_reason("ok, pode continuar o poll"))
        self.assertIsNone(skip_reason("sim, e a paginação?"))

    def test_punctuation_does_not_escape_the_filter(self):
        self.assertIsNotNone(skip_reason("continue!"))
        self.assertIsNotNone(skip_reason("  ok.  "))

    def test_bare_command_is_skipped(self):
        self.assertEqual(skip_reason("/memoria-status"), "comando sem argumento")

    def test_command_with_an_argument_is_not_skipped(self):
        self.assertIsNone(skip_reason("/buscar paginação do poll"))

    def test_a_real_question_passes(self):
        self.assertIsNone(skip_reason("como funciona a paginação do poll?"))

    def test_none_and_empty_do_not_break(self):
        self.assertIsNotNone(skip_reason(None))
        self.assertIsNotNone(skip_reason(""))


class TestAngles(unittest.TestCase):
    def test_raw_text_is_always_first(self):
        p = "como funciona a paginação do poll no conector?"
        self.assertEqual(angles(p)[0], p)

    def test_content_angle_strips_structure(self):
        content = content_words("como funciona a paginação do poll no conector?")
        for structure in ("como", "a", "do", "no"):
            self.assertNotIn(f" {structure} ", f" {content} ")
        for term in ("funciona", "paginação", "poll", "conector"):
            self.assertIn(term, content)

    def test_repeated_words_appear_once(self):
        self.assertEqual(content_words("poll poll poll cursor"), "poll cursor")

    def test_longest_sentence_only_with_several_sentences(self):
        self.assertEqual(longest_sentence("uma frase só"), "")
        longest = longest_sentence("curta. esta aqui é bem mais longa que a outra. fim")
        self.assertIn("bem mais longa", longest)

    def test_no_duplicates_across_angles(self):
        for p in ("paginação poll cursor", "ABC DEF GHI", "termo"):
            a = angles(p)
            self.assertEqual(len(a), len(set(a)), "embedar o mesmo texto duas vezes é gasto puro")

    def test_at_most_three_angles(self):
        p = ("primeira frase com algum conteudo tecnico. segunda frase bem mais longa "
             "que a primeira e com outros termos relevantes. terceira.")
        self.assertLessEqual(len(angles(p)), 3)

    def test_respects_the_char_limit(self):
        for a in angles("x" * 5000, char_limit=100):
            self.assertLessEqual(len(a), 100)


class TestBreaker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "breaker"

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_allows_an_attempt(self):
        self.assertIsNone(Breaker(self.path, 300).is_open())

    def test_after_arming_it_stays_open(self):
        b = Breaker(self.path, 300)
        b.arm()
        idle = b.is_open()
        self.assertIsNotNone(idle)
        self.assertLess(idle, 5)

    def test_expired_wait_allows_another_attempt(self):
        b = Breaker(self.path, 0.05)
        b.arm()
        time.sleep(0.1)
        self.assertIsNone(b.is_open())

    def test_clear_reopens_immediately(self):
        b = Breaker(self.path, 300)
        b.arm()
        b.clear()
        self.assertIsNone(b.is_open())

    def test_zero_cooldown_disables_the_breaker(self):
        b = Breaker(self.path, 0)
        b.arm()
        self.assertIsNone(b.is_open(), "cooldown 0 significa nunca bloquear")

    def test_corrupted_content_allows_an_attempt(self):
        self.path.write_text("isto não é um timestamp")
        self.assertIsNone(Breaker(self.path, 300).is_open(),
                          "disjuntor com defeito não pode ser o motivo de a busca parar")

    def test_missing_directory_is_created_on_arm(self):
        deep_path = Path(self.tmp.name) / "a" / "b" / "breaker"
        Breaker(deep_path, 300).arm()
        self.assertTrue(deep_path.exists())

    def test_arming_on_an_impossible_path_does_not_raise(self):
        Breaker("/proc/impossivel/breaker", 300).arm()  # não pode explodir


if __name__ == "__main__":
    unittest.main(verbosity=2)
