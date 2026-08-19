"""O que entra no acervo quando alguém pede "indexe este projeto".

A fonte é `git ls-files` e não uma varredura do disco. Isso faz o `.gitignore` ser respeitado
POR DEFINIÇÃO em vez de por uma reimplementação nossa, e `node_modules`, build e cache ficam de
fora sem regra especial — eles não estão versionados.
"""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import scan  # noqa: E402


def a_repo(**files) -> str:
    root = tempfile.mkdtemp()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=60)
    for name, content in files.items():
        path = os.path.join(root, name.replace("__", "/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = "wb" if isinstance(content, bytes) else "w"
        with open(path, mode) as fh:
            fh.write(content)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, timeout=60)

    return root


class TestTheSourceIsGIT(unittest.TestCase):
    def test_only_tracked_files_are_eligible(self):
        root = a_repo(**{"kept.py": "x = 1\n"})
        with open(os.path.join(root, "untracked.py"), "w") as fh:
            fh.write("y = 2\n")
        out = scan.eligible(root)
        names = [os.path.basename(p) for p in out["eligible"]]
        self.assertEqual(names, ["kept.py"])

    def test_a_gitignored_file_is_absent_without_us_parsing_gitignore(self):
        """O ponto de usar `git ls-files`: nunca reimplementamos a semântica de `.gitignore`,
        que tem negação, precedência e regras por diretório."""
        root = a_repo(**{".gitignore": "build/\n", "app.py": "x = 1\n"})
        os.makedirs(os.path.join(root, "build"), exist_ok=True)
        with open(os.path.join(root, "build", "out.js"), "w") as fh:
            fh.write("console.log(1)\n")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True, timeout=60)
        names = [os.path.basename(p) for p in scan.eligible(root)["eligible"]]
        self.assertIn("app.py", names)
        self.assertNotIn("out.js", names)

    def test_the_paths_are_ABSOLUTE(self):
        """`add_files` recebe caminhos e o daemon roda com outro cwd — um caminho relativo
        resolveria contra o diretório errado, em silêncio."""
        root = a_repo(**{"a.py": "x = 1\n"})
        for path in scan.eligible(root)["eligible"]:
            self.assertTrue(os.path.isabs(path), path)

    def test_a_directory_that_is_not_a_repo_yields_nothing_and_does_not_raise(self):
        out = scan.eligible(tempfile.mkdtemp())
        self.assertEqual(out["eligible"], [])
        self.assertEqual(out["tracked"], 0)


class TestTheFourDiscards(unittest.TestCase):
    """Cada um isolado, porque um teste com dois motivos de descarte não prova nenhum."""

    def test_a_binary_file_is_skipped(self):
        root = a_repo(**{"logo.png": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"})
        out = scan.eligible(root)
        self.assertEqual(out["eligible"], [])
        self.assertEqual(out["skipped"]["binary"], 1)

    def test_a_lockfile_is_skipped(self):
        root = a_repo(**{"package-lock.json": '{"lockfileVersion": 3}\n'})
        out = scan.eligible(root)
        self.assertEqual(out["eligible"], [])
        self.assertEqual(out["skipped"]["lockfile"], 1)

    def test_a_minified_file_is_skipped(self):
        """Uma linha de 40 KB não responde pergunta nenhuma e domina o acervo do repo."""
        root = a_repo(**{"bundle.js": "var a=1;" * 6000 + "\n"})
        out = scan.eligible(root)
        self.assertEqual(out["eligible"], [])
        self.assertEqual(out["skipped"]["minified"], 1)

    def test_a_file_above_the_ceiling_is_skipped(self):
        root = a_repo(**{"huge.txt": "line\n" * 300})
        out = scan.eligible(root, max_bytes=100)
        self.assertEqual(out["eligible"], [])
        self.assertEqual(out["skipped"]["too_big"], 1)

    def test_an_ordinary_source_file_survives_ALL_FOUR(self):
        """A guarda das guardas: se um filtro ficar largo demais, isto cai."""
        root = a_repo(**{"core__app.py": "def main():\n    return 1\n"})
        out = scan.eligible(root)
        self.assertEqual(len(out["eligible"]), 1)
        self.assertEqual(sum(out["skipped"].values()), 0)


class TestTheFunnelIsREPORTED(unittest.TestCase):
    def test_it_counts_what_was_seen_and_what_was_dropped(self):
        """Uma contagem que só aparece no fim é uma contagem que ninguém usa para decidir."""
        root = a_repo(**{"a.py": "x = 1\n", "package-lock.json": "{}\n",
                         "logo.png": b"\x00\x01\x02binary"})
        out = scan.eligible(root)
        self.assertEqual(out["tracked"], 3)
        self.assertEqual(len(out["eligible"]), 1)
        self.assertEqual(out["skipped"]["lockfile"], 1)
        self.assertEqual(out["skipped"]["binary"], 1)

    def test_every_discard_reason_is_present_even_when_zero(self):
        """Uma chave ausente obriga todo consumidor a usar `.get`, e um deles vai esquecer."""
        out = scan.eligible(a_repo(**{"a.py": "x = 1\n"}))
        self.assertEqual(set(out["skipped"]), {"binary", "minified", "lockfile", "too_big",
                                               "unreadable"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
