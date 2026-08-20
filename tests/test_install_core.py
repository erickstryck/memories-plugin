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
from core.errors import CoreError  # noqa: E402


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

    def test_the_fix_hint_is_not_the_command_that_is_missing(self):
        """"`qctx` is not on PATH" fixed by "run `qctx install`" is circular in exactly
        the state it describes. The bootstrap is the thing that runs with no launcher,
        and it can be named without naming a host — which this module may not do."""
        hint = install.launcher_check(REPO, self.env()).fix_hint
        self.assertNotIn(f"{install.LAUNCHER_NAME} install", hint)
        self.assertIn("install.sh", hint)

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

    HALF the proof, and it says so. What follows compares three tuples against
    `config.DEFAULTS`, which catches a new field with nowhere to go — but a tuple is a
    declaration, and `vector_size` sat in `DETECTED_FIELDS` while nothing in the wizard
    wrote it. The other half is
    `tests.test_cli_install.EveryFieldIsSet.test_the_wizard_sets_all_fifteen`, which
    drives the wizard end to end and reads back where each of the fifteen landed.
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

    def test_an_environment_only_key_says_no_file_holds_it(self):
        """It stays `ok`, and it says WHERE ok came from.

        Downgrading an environment-only key would nag every machine that exports its
        keys from `.bashrc`, a systemd unit or a secrets manager — durable storage this
        check cannot see, so calling it a warning would be a false negative on a
        correctly configured machine. But `ok` alone reads as "this is stored", and for
        a key that lives only in this process's environment that is the one thing not
        yet established.
        """
        other = self.dir / ".env"
        other.write_text("SOMETHING_ELSE=x\n")
        check = self.named(install.credentials_check({"QCTX_API_KEY": "abcdef"},
                                                     [other]))["api_key"]
        self.assertTrue(check.ok)
        self.assertIn("no credential file", check.detail)

    def test_a_key_that_a_file_holds_does_not_get_that_sentence(self):
        env_file = self.dir / ".env"
        env_file.write_text("SERVER_API_KEY=abcdefgh\n")
        check = self.named(install.credentials_check({"QCTX_API_KEY": "abcdef"},
                                                     [env_file]))["api_key"]
        self.assertTrue(check.ok)
        self.assertNotIn("no credential file", check.detail)

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
        """`config_path` is passed, and that is not a detail.

        Without it `no_shell_check(None)` falls back to `DEFAULT_CONFIG_PATH` and reads
        the developer's real `~/.config/memories-plugin/config.json` — an unnoticed
        real-machine read, in the very module whose reason for existing is to state the
        opposite rule. The verdict below is asserted too, so the argument cannot be
        dropped again without something going red: over a temporary directory with no
        config file in it, the no-shell check has to FAIL.
        """
        checks = install.plumbing(REPO, {"HOME": str(self.dir), "PATH": ""},
                                  config_path=self.dir / "config.json")
        named = {c.name: c for c in checks}
        self.assertIn("qdrant_api_key", named)
        self.assertIn("api_key", named)
        self.assertFalse(named["no-shell config"].ok,
                         "a config file was read from somewhere outside this test")


class UnreadableCredentialFiles(unittest.TestCase):
    """A credential file that is not UTF-8 is somebody else's file, and `--check` runs
    over every file it was handed."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / ".secrets"
        self.addCleanup(self.tmp.cleanup)

    def test_a_non_utf8_file_reports_nothing_rather_than_raising(self):
        """`read_env_names` caught `OSError` and not `UnicodeDecodeError`, so a single
        stray byte in `~/.secrets` turned the whole report into a traceback. A file that
        cannot be decoded tells us nothing about which names it sets, and "nothing" is
        the true answer to the question this function asks."""
        self.path.write_bytes(b"SERVER_API_KEY=abc\n\xff\xfe not utf-8\n")
        self.assertEqual(install.read_env_names(self.path), {})

    def test_the_whole_credential_check_survives_one(self):
        self.path.write_bytes(b"\xff\xfe\x00\x00\n")
        checks = install.credentials_check({}, [self.path])
        self.assertEqual(len(checks), 2)

    def test_writing_into_one_REFUSES_instead_of_replacing_its_contents(self):
        """The one place where catching and carrying on would be the worse bug.

        `write_env_file` rewrites the file from the lines it read. If a failed read gave
        it an empty list, the atomic replace would put a two-line file over one that may
        hold every credential the user has — a silent deletion, dressed as a successful
        write. So it refuses, as a `CoreError` the CLI already renders as one line, and
        names the file.
        """
        self.path.write_bytes(b"OTHER=\xff\xfe\n")
        before = self.path.read_bytes()
        with self.assertRaises(CoreError) as caught:
            install.write_env_file(self.path, {"SERVER_API_KEY": "abc"})
        self.assertIn(str(self.path), str(caught.exception))
        self.assertEqual(self.path.read_bytes(), before)


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

    def test_it_replaces_the_export_form_too(self):
        """`export NAME=` is the form this repository supports everywhere else — it is
        what `tests/test_hermes_cutover.py` writes. Matching only `NAME=` appended a
        second line for the same variable, and which one the loader reads last is a coin
        toss the operator cannot see."""
        self.path.write_text("export SERVER_API_KEY=old\n")
        install.write_env_file(self.path, {"SERVER_API_KEY": "new"})
        body = self.path.read_text()
        self.assertEqual(body.count("SERVER_API_KEY="), 1)
        self.assertIn("export SERVER_API_KEY=new", body)

    def test_it_keeps_the_export_prefix_it_found(self):
        """Rewriting `export NAME=` as `NAME=` would leave the variable set but not
        exported, so a sourced file would stop reaching child processes."""
        self.path.write_text("  export SERVER_API_KEY=old\n")
        install.write_env_file(self.path, {"SERVER_API_KEY": "new"})
        self.assertTrue(self.path.read_text().startswith("export SERVER_API_KEY=new"))

    def test_the_mode_comes_from_the_creation_not_from_a_later_chmod(self):
        """A `write_text` followed by `chmod` leaves the credential world-readable for
        as long as the two calls are apart. The window is small and the file is a
        plaintext key, so it is closed at the source: the mode is an argument to
        `os.open`, and there is no chmod afterwards to observe."""
        import os
        from unittest import mock
        old_umask = os.umask(0)
        self.addCleanup(os.umask, old_umask)
        with mock.patch.object(Path, "chmod") as path_chmod, \
                mock.patch("os.chmod") as os_chmod:
            install.write_env_file(self.path, {"SERVER_API_KEY": "abc"})
        path_chmod.assert_not_called()
        os_chmod.assert_not_called()
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_the_original_survives_a_write_that_fails_halfway(self):
        """`~/.secrets` may hold every credential the user has, and the old shape was
        `os.open(O_TRUNC)` then `write`: the truncation landed first, so a disk that
        filled up — or a process killed between the two — left the file EMPTY. Every
        other thing in this repository that rewrites a user's file takes a dated backup
        before touching it; both cutovers do, and refuse to proceed without one.
        """
        import os
        from unittest import mock
        before = "EVERY=other-credential\nOTHER=keep\n"
        self.path.write_text(before)

        real_fdopen = os.fdopen

        class Exploding:
            def __init__(self, real):
                self.real = real

            def write(self, _body):
                raise OSError(28, "No space left on device")

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                self.real.close()

                return False

        with mock.patch("os.fdopen", lambda fd, *a, **kw: Exploding(real_fdopen(fd, *a, **kw))):
            with self.assertRaises(OSError):
                install.write_env_file(self.path, {"SERVER_API_KEY": "abc"})
        self.assertEqual(self.path.read_text(), before,
                         "a failed write truncated the user's credential file")

    def test_the_replace_is_atomic_and_leaves_no_debris_when_it_fails(self):
        """The target is never opened for writing at all: a temporary file in the same
        directory is written in full and then `os.replace`d over it, which is atomic on
        the same filesystem. And a replace that fails takes its own temporary with it —
        a directory of `.env.…tmp` files each holding a plaintext key would be a worse
        outcome than the truncation."""
        import os
        from unittest import mock
        before = "OTHER=keep\n"
        self.path.write_text(before)
        with mock.patch("os.replace", side_effect=OSError(28, "No space left")):
            with self.assertRaises(OSError):
                install.write_env_file(self.path, {"SERVER_API_KEY": "abc"})
        self.assertEqual(self.path.read_text(), before)
        self.assertEqual([p.name for p in self.path.parent.iterdir()], [self.path.name],
                         "a temporary file holding the key was left behind")

    def test_it_does_not_re_mode_a_file_the_user_already_had(self):
        """`~/.secrets` is the user's file, with the user's permissions. Writing a key
        into it is not a licence to change them behind their back."""
        self.path.write_text("OTHER=keep\n")
        self.path.chmod(0o640)
        install.write_env_file(self.path, {"SERVER_API_KEY": "abc"})
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o640)


if __name__ == "__main__":
    unittest.main()
