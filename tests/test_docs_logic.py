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
    def test_units(self):
        self.assertEqual(parse_ttl("30s"), 30)
        self.assertEqual(parse_ttl("30m"), 1800)
        self.assertEqual(parse_ttl("2h"), 7200)
        self.assertEqual(parse_ttl("7d"), 604800)

    def test_bare_seconds(self):
        self.assertEqual(parse_ttl("90"), 90)

    def test_accepts_spaces_and_uppercase(self):
        self.assertEqual(parse_ttl(" 24H "), 86400)

    def test_invalid_raises(self):
        for bad in ("", "abc", "24x", "-5h"):
            with self.assertRaises(DocsError):
                parse_ttl(bad)


class TestDocId(unittest.TestCase):
    def test_stable_for_the_same_path(self):
        self.assertEqual(doc_id_for("/tmp/a.md"), doc_id_for("/tmp/a.md"))

    def test_relative_and_absolute_paths_give_the_same_id(self):
        cwd = os.getcwd()
        try:
            os.chdir("/tmp")
            self.assertEqual(doc_id_for("a.md"), doc_id_for("/tmp/a.md"),
                             "o id vem do caminho ABSOLUTO, para reindexar substituir")
        finally:
            os.chdir(cwd)

    def test_different_paths_give_different_ids(self):
        self.assertNotEqual(doc_id_for("/tmp/a.md"), doc_id_for("/tmp/b.md"))


class TestStaleness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.file_path = Path(self.tmp.name) / "doc.md"
        self.file_path.write_text("conteudo original\n")
        st = os.stat(self.file_path)
        self.mtime = st.st_mtime
        self.size = st.st_size

    def tearDown(self):
        self.tmp.cleanup()

    def test_untouched_file_is_not_stale(self):
        self.assertIsNone(source_changed(str(self.file_path), self.mtime, self.size))

    def test_float_round_trip_does_not_produce_a_false_positive(self):
        """O bug original: JSON devolvia ...9956775 para ...9956777 gravado."""
        nearly = self.mtime + 2.4e-07
        self.assertIsNone(source_changed(str(self.file_path), nearly, self.size),
                          "diferença de 1e-7 é ruído de serialização, não edição")

    def test_mtime_rounded_to_the_millisecond_produces_no_false_positive(self):
        self.assertIsNone(source_changed(str(self.file_path), round(self.mtime, 3), self.size))

    def test_size_change_is_detected(self):
        self.file_path.write_text("conteudo original com mais texto\n")
        reason = source_changed(str(self.file_path), self.mtime, self.size)
        self.assertIsNotNone(reason)
        self.assertIn("tamanho", reason)

    def test_mtime_change_at_the_same_size_is_detected(self):
        future = self.mtime + 60
        os.utime(self.file_path, (future, future))
        reason = source_changed(str(self.file_path), self.mtime, self.size)
        self.assertIsNotNone(reason, "edição que preserva o tamanho ainda muda o mtime")

    def test_removed_file_is_reported(self):
        path = str(self.file_path)
        os.unlink(path)
        self.assertEqual(source_changed(path, self.mtime, self.size),
                         "arquivo não existe mais")

    def test_missing_metadata_does_not_break(self):
        self.assertIsNone(source_changed(str(self.file_path), None, None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
