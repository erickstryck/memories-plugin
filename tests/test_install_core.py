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


class FieldCoverage(unittest.TestCase):
    """Simple must not cost complete.

    The wizard is allowed to be short, and it is not allowed to leave a field of `Config`
    with no place to be set. `_check_collections` learned this the expensive way: it
    enumerated three of five roles and reported `ready` over a configuration where the
    repository archive sat on top of the memory one.
    """

    def test_every_field_is_reachable_exactly_once(self):
        from core import config
        walked = (install.REQUIRED_FIELDS + install.OPTIONAL_FIELDS
                  + install.DETECTED_FIELDS)
        self.assertEqual(len(walked), len(set(walked)), "a field is listed twice")
        self.assertEqual(set(walked), set(config.DEFAULTS),
                         "the wizard and Config disagree about what configuration is")

    def test_the_two_keys_are_asked_for(self):
        """SECRET_FIELDS governs WHERE a value is written, never whether it is asked."""
        from core import config
        for secret in config.SECRET_FIELDS:
            self.assertIn(secret, install.REQUIRED_FIELDS)


class CredentialFile(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / ".env"
        self.addCleanup(self.tmp.cleanup)

    def test_creates_with_owner_only_permissions(self):
        install.write_env_file(self.path, {"SERVER_API_KEY": "abc"})
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertIn("SERVER_API_KEY=abc", self.path.read_text())

    def test_replaces_instead_of_duplicating(self):
        install.write_env_file(self.path, {"SERVER_API_KEY": "old"})
        install.write_env_file(self.path, {"SERVER_API_KEY": "new"})
        body = self.path.read_text()
        self.assertEqual(body.count("SERVER_API_KEY="), 1)
        self.assertIn("SERVER_API_KEY=new", body)

    def test_leaves_other_lines_alone(self):
        self.path.write_text("OTHER=keep\n")
        install.write_env_file(self.path, {"SERVER_API_KEY": "abc"})
        self.assertIn("OTHER=keep", self.path.read_text())


if __name__ == "__main__":
    unittest.main()
