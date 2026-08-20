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


class Credentials(unittest.TestCase):
    """`--check` has to say whether each key is present, and IN WHICH SPELLING.

    Checking fewer spellings than `core.config` accepts would be worse than not checking
    at all: it would report a correctly configured machine as missing its key. So the
    list is read off `ENV_ALIASES` and never typed out here either.
    """

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def named(self, checks):
        return {c.name: c for c in checks}

    def test_both_keys_are_reported(self):
        from core import config
        checks = self.named(install.credentials_check({}))
        self.assertEqual(set(checks), set(config.SECRET_FIELDS))

    def test_it_names_the_spelling_found_in_the_environment(self):
        check = self.named(install.credentials_check(
            {"QDRANT_SERVICE_API_KEY": "abcdef"}))["qdrant_api_key"]
        self.assertTrue(check.ok)
        self.assertIn("QDRANT_SERVICE_API_KEY", check.detail)
        self.assertIn("environment", check.detail)

    def test_it_finds_a_key_that_lives_only_in_a_credential_file(self):
        """The whole point: a key in a file the process does not export still counts,
        and the report says which file it is in."""
        env_file = self.dir / ".env"
        env_file.write_text("# a comment\nexport SERVER_API_KEY=abcdefgh\n")
        check = self.named(install.credentials_check({}, [env_file]))["api_key"]
        self.assertTrue(check.ok)
        self.assertIn("SERVER_API_KEY", check.detail)
        self.assertIn(str(env_file), check.detail)

    def test_every_spelling_the_core_accepts_is_checked(self):
        """Three for the Qdrant key. A hand-copied list is what this guards against."""
        from core import config
        for field, aliases in config.ENV_ALIASES.items():
            if field not in config.SECRET_FIELDS:
                continue
            for alias in aliases:
                check = self.named(install.credentials_check({alias: "abc"}))[field]
                self.assertTrue(check.ok, f"{alias} was not recognised")
                self.assertIn(alias, check.detail)

    def test_a_missing_key_is_pending_and_names_the_spellings(self):
        """Not a blocker: a local Qdrant with no auth is a legitimate install, and a
        FAIL here would report every one of them as broken."""
        from core import config
        check = self.named(install.credentials_check({}))["qdrant_api_key"]
        self.assertFalse(check.ok)
        self.assertTrue(check.warning)
        for alias in config.ENV_ALIASES["qdrant_api_key"]:
            self.assertIn(alias, check.detail + (check.fix_hint or ""))

    def test_no_value_is_ever_printed(self):
        env_file = self.dir / ".env"
        env_file.write_text("SERVER_API_KEY=file-s3cr3t\n")
        checks = install.credentials_check({"QCTX_QDRANT_API_KEY": "env-s3cr3t"},
                                           [env_file])
        printed = " ".join(f"{c.name} {c.detail} {c.fix_hint or ''}" for c in checks)
        self.assertNotIn("env-s3cr3t", printed)
        self.assertNotIn("file-s3cr3t", printed)
        self.assertIn("10 chars", printed)   # the length is what stands in for it

    def test_the_plumbing_carries_them(self):
        names = [c.name for c in install.plumbing(REPO, {"HOME": str(self.dir),
                                                         "PATH": ""})]
        self.assertIn("qdrant_api_key", names)
        self.assertIn("api_key", names)


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
