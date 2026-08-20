"""The plumbing the wizard installs, checked as data.

Host-neutral by contract: this module is the one the other host imports too, so a check
that named claude or hermes would be a check the other host cannot use. The host sections
live in the CLI.
"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core import install  # noqa: E402


class Plumbing(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        (self.home / ".local" / "bin").mkdir(parents=True)
        self.addCleanup(self.tmp.cleanup)

    def env(self, on_path=True):
        bindir = self.home / ".local" / "bin"
        return {"HOME": str(self.home), "PATH": str(bindir) if on_path else "/usr/bin"}

    def test_launcher_missing_is_a_blocker_and_names_where_it_goes(self):
        check = install.launcher_check(REPO, self.env())
        self.assertFalse(check.ok)
        self.assertFalse(check.warning)
        self.assertIn(str(self.home / ".local" / "bin"), check.fix_hint)

    def test_launcher_identical_copy_is_ok(self):
        copy = self.home / ".local" / "bin" / "qctx"
        copy.write_bytes((REPO / "bin" / "qctx").read_bytes())
        copy.chmod(0o755)
        self.assertTrue(install.launcher_check(REPO, self.env()).ok)

    def test_launcher_stale_copy_is_reported(self):
        copy = self.home / ".local" / "bin" / "qctx"
        copy.write_text("#!/usr/bin/env bash\necho old\n")
        copy.chmod(0o755)
        check = install.launcher_check(REPO, self.env())
        self.assertFalse(check.ok)
        self.assertIn("differs", check.detail)

    def test_a_symlink_to_the_tree_counts_as_current(self):
        link = self.home / ".local" / "bin" / "qctx"
        link.symlink_to(REPO / "bin" / "qctx")
        self.assertTrue(install.launcher_check(REPO, self.env()).ok)

    def test_bin_dir_off_the_path_is_reported(self):
        self.assertFalse(install.path_check(self.env(on_path=False)).ok)
        self.assertTrue(install.path_check(self.env()).ok)

    def test_no_shell_check_reads_the_file_only(self):
        cfg_path = Path(self.tmp.name) / "config.json"
        cfg_path.write_text('{"qdrant_url": "https://q", "api_base_url": "https://e/v1",'
                            ' "memory_collection": "mem"}')
        self.assertTrue(install.no_shell_check(cfg_path).ok)

    def test_no_shell_check_ignores_the_environment(self):
        """The failure this exists to catch: exported URLs, empty file."""
        import os
        cfg_path = Path(self.tmp.name) / "empty.json"
        cfg_path.write_text("{}")
        os.environ["QCTX_QDRANT_URL"] = "https://exported"
        self.addCleanup(os.environ.pop, "QCTX_QDRANT_URL", None)
        check = install.no_shell_check(cfg_path)
        self.assertFalse(check.ok)
        self.assertIn("qdrant_url", check.detail)


if __name__ == "__main__":
    unittest.main()
