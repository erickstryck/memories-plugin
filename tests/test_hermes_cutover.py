"""The hermes cutover script: `scripts/hermes_cutover.sh`.

A shell script is testable, and this one has to be: its whole job is to REPORT the state
of a live install and then change it. Both halves fail silently if wrong — a check that
cannot fail prints "ok" for a state it never verified, and an apply step that fails while
the script exits 0 reports success it did not establish. Both happened in the claude-code
cutover (`scripts/cutover.sh`), and its comments name them.

Everything here runs against a FAKE `HERMES_HOME` in a temp directory, with `HOME` pointed
there too, so no test can reach the real `~/.hermes`, `~/.config/memories-plugin` or the
user's running sessions. The apply path is exercised ONLY against those fakes.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SCRIPT = REPO / "scripts" / "hermes_cutover.sh"

#: A value that must never appear in the output. The script names the variables it wants
#: and must never echo what is in them: this output gets pasted into issues and chats.
SECRET = "s3cr3t-value-that-must-not-be-printed"

CONFIG_YAML = (
    "model:\n"
    "  default: MiniMax-M2.7\n"
    "memory:\n"
    "  memory_enabled: true\n"
    "  provider: qdrant\n"
    "  nudge_interval: 10\n"
    "delegation:\n"
    "  max_iterations: 80\n"
)


class CutoverCase(unittest.TestCase):
    """One fake HERMES_HOME per test, plus a config.json the plugin reads as configured."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.home = self.tmp / "fake-home"
        self.hermes = self.home / ".hermes"
        (self.hermes / "plugins").mkdir(parents=True)
        self.config_yaml = self.hermes / "config.yaml"
        self.config_yaml.write_text(CONFIG_YAML)
        self.qctx_config = self.tmp / "config.json"
        self.qctx_config.write_text(
            '{"qdrant_url": "http://127.0.0.1:6333",'
            ' "embed_url": "http://127.0.0.1:8000/v1/embeddings",'
            ' "memory_collection": "fake_memory",'
            ' "docs_collection": "fake_docs",'
            ' "library_collection": "fake_library"}\n'
        )

    def env(self, **over):
        """A minimal environment. HOME points at the fake home as well, so a bug that
        ignored HERMES_HOME still could not reach the real one."""
        env = {
            "PATH": os.environ["PATH"],
            "HOME": str(self.home),
            "HERMES_HOME": str(self.hermes),
            "QCTX_CONFIG": str(self.qctx_config),
            "QCTX_STATE_DIR": str(self.tmp / "state"),
            # The suite this script runs CONTAINS the tests that run this script, so the
            # dry-run tests skip it. The guard deliberately does NOT buy an `--apply` (the
            # script refuses), so the apply tests run the script from a fake ROOT whose
            # `tests/` holds one trivial test instead — see `fake_root`.
            "HERMES_CUTOVER_SKIP_SUITE": "1",
            "QDRANT_SERVICE_API_KEY": SECRET,
            "SERVER_API_KEY": SECRET,
        }
        for key, value in over.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value

        return env

    def run_script(self, *args, script=None, env=None):
        out = subprocess.run(
            ["bash", str(script or SCRIPT), *args],
            capture_output=True, text=True, env=env if env is not None else self.env(),
        )
        self.report = out.stdout + out.stderr

        return out

    PASSING_TEST = ("import unittest\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_ok(self):\n        self.assertTrue(True)\n")

    def fake_root(self, test_body=None, *parts):
        """The script copied into a temp ROOT that looks like the repo: `cli`, `core` and
        `hosts` symlinked to the real ones (the availability probe imports through them) and a
        `tests/` holding ONE trivial test.

        Two jobs. It makes the suite check provable in both directions without recursion, and
        it gives the `--apply` tests a real suite to pass — which they need, because
        HERMES_CUTOVER_SKIP_SUITE deliberately refuses to authorise an apply.

        `parts` chooses the path, because two checks depend on it: the worktree refusal reads
        `$ROOT`, and this repo is itself checked out in a worktree.
        """
        root = self.tmp.joinpath(*(parts or ("fake-root",)))
        (root / "scripts").mkdir(parents=True)
        (root / "tests").mkdir()
        (root / "tests" / "test_fake.py").write_text(test_body or self.PASSING_TEST)
        shutil.copy2(SCRIPT, root / "scripts" / SCRIPT.name)
        for shared in ("cli", "core", "hosts"):
            (root / shared).symlink_to(REPO / shared)

        return root / "scripts" / SCRIPT.name

    def assertLine(self, out, needle):
        self.assertIn(needle, out.stdout + out.stderr,
                      f"expected a line containing {needle!r}:\n{out.stdout}{out.stderr}")

    def assertNoLine(self, out, needle):
        self.assertNotIn(needle, out.stdout + out.stderr,
                         f"did NOT expect {needle!r}:\n{out.stdout}{out.stderr}")


class TestTheDocumentedAliasesMatchTheCode(unittest.TestCase):
    """The README's table and the script's credential check both enumerate `ENV_ALIASES`, and
    both were missing `QDRANT_API_KEY` — the third name the core accepts for the Qdrant key.
    A list of accepted names is worth exactly what it is complete to: the core resolved that
    key while the script FAILED and the README said "both spellings".
    """

    def aliases(self):
        from core.config import ENV_ALIASES

        return ENV_ALIASES

    def test_the_readme_table_lists_every_alias_the_core_accepts(self):
        rows = dict(re.findall(r"^\| `([a-z_]+)` \| (.+?) \|$", (REPO / "README.md").read_text(),
                               re.M))
        for field, aliases in self.aliases().items():
            with self.subTest(field=field):
                self.assertIn(field, rows, "the config field is not in the README table")
                self.assertEqual(tuple(re.findall(r"`([A-Z_]+)`", rows[field])), aliases)

    def test_the_script_checks_every_alias_of_the_two_secrets(self):
        """The script cannot import the core to enumerate them (it checks the shell it is
        running in, name by name), so the list is duplicated there — which is exactly why it
        needs a test tying it back to the source of truth."""
        text = SCRIPT.read_text()
        checked = re.findall(r"^check_key\s+\"[^\"]+\"\s+(.+)$", text, re.M)
        self.assertEqual(len(checked), 2, f"expected two credential checks, found: {checked}")
        found = {names.split()[0]: tuple(names.split()) for names in checked}
        for field in ("qdrant_api_key", "api_key"):
            aliases = self.aliases()[field]
            with self.subTest(field=field):
                self.assertIn(aliases[0], found, "the canonical name is not checked at all")
                self.assertEqual(found[aliases[0]], aliases,
                                 "the script checks a different set of names than the core "
                                 "accepts — a key the core would resolve reads as missing")


class TestUsage(CutoverCase):
    def test_the_script_parses(self):
        out = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)

    def test_it_is_executable(self):
        self.assertTrue(os.access(SCRIPT, os.X_OK), "the script is not executable")

    def test_an_unknown_argument_is_refused_rather_than_treated_as_a_dry_run(self):
        """`--aply` (a typo) must not silently become a dry run, and `-apply` must not
        silently become an apply. Anything that is not exactly --apply is a usage error."""
        for bad in ("--aply", "-apply", "apply", "--force"):
            out = self.run_script(bad)
            self.assertEqual(out.returncode, 2, f"{bad}: {out.stdout}{out.stderr}")
            self.assertLine(out, "usage")
            self.assertNoLine(out, "DRY RUN")


class TestDryRunReportsAndChangesNothing(CutoverCase):
    def test_the_healthy_case_is_a_dry_run_that_exits_zero(self):
        before = self.config_yaml.read_bytes()
        out = self.run_script()
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertLine(out, "DRY RUN")
        self.assertLine(out, "=== what changes ===")
        self.assertEqual(self.config_yaml.read_bytes(), before, "the dry run wrote config.yaml")
        self.assertFalse((self.hermes / "plugins" / "memories").exists(),
                         "the dry run created the symlink")

    def test_it_names_the_provider_it_replaces_and_says_the_data_stays(self):
        """Asserted against the sentences that carry the PROVIDER NAME, not just against the
        `search-collections` string — which is printed whatever provider is being replaced,
        and so would keep this green with the name gone."""
        out = self.run_script()
        self.assertLine(out, "memory.provider: qdrant -> memories")
        self.assertLine(out, "qdrant provider's own directory (disabled by configuration")
        self.assertLine(out, "1423 points in hermes_memory")
        self.assertLine(out, "--collections hermes_memory")

    def test_it_never_echoes_a_credential_value(self):
        out = self.run_script()
        # The exit code and the banner are asserted FIRST on purpose: an absence check
        # passes trivially against a script that failed to start, and this suite has
        # already been bitten by a test that passed on an error message.
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertLine(out, "DRY RUN")
        self.assertNoLine(out, SECRET)

    def test_no_hermes_home_fails(self):
        out = self.run_script(env=self.env(HERMES_HOME=str(self.tmp / "nope")))
        self.assertEqual(out.returncode, 1)
        self.assertLine(out, "FAIL")
        self.assertLine(out, "HERMES_HOME")

    def test_no_config_yaml_fails(self):
        self.config_yaml.unlink()
        out = self.run_script()
        self.assertEqual(out.returncode, 1, out.stdout)
        self.assertLine(out, "FAIL")
        self.assertLine(out, "config.yaml")

    def test_it_reports_when_the_provider_is_already_selected(self):
        self.config_yaml.write_text(CONFIG_YAML.replace("provider: qdrant", "provider: memories"))
        out = self.run_script()
        self.assertEqual(out.returncode, 0, out.stdout)
        self.assertLine(out, "already selected")


class TestTheEnvironmentCheck(CutoverCase):
    """The check that has to exist because the plugin cannot carry these itself.

    `core.save()` REFUSES to write the two API keys to config.json, so they can only come
    from the environment — and `is_available()` does not look at them (it checks the URL,
    the embed endpoint and the collection). So a hermes started from a shell without them
    reports a healthy provider and then fails every single search.
    """

    def test_credentials_missing_from_everywhere_fails_with_both_accepted_names(self):
        out = self.run_script(env=self.env(QDRANT_SERVICE_API_KEY=None, SERVER_API_KEY=None))
        self.assertEqual(out.returncode, 1, out.stdout)
        self.assertLine(out, "FAIL")
        self.assertLine(out, "QCTX_QDRANT_API_KEY")
        self.assertLine(out, "QDRANT_SERVICE_API_KEY")
        self.assertLine(out, "QCTX_API_KEY")
        self.assertLine(out, "SERVER_API_KEY")

    def test_one_missing_credential_is_still_a_failure(self):
        out = self.run_script(env=self.env(SERVER_API_KEY=None))
        self.assertEqual(out.returncode, 1, out.stdout)
        self.assertLine(out, "FAIL")

    def test_the_canonical_names_are_accepted_too(self):
        env = self.env(QDRANT_SERVICE_API_KEY=None, SERVER_API_KEY=None)
        env["QCTX_QDRANT_API_KEY"] = SECRET
        env["QCTX_API_KEY"] = SECRET
        out = self.run_script(env=env)
        self.assertEqual(out.returncode, 0, out.stdout)
        self.assertNoLine(out, "FAIL")

    def test_the_THIRD_qdrant_alias_is_accepted(self):
        """`core/config.py` accepts three names for the Qdrant key, and `QDRANT_API_KEY` — the
        upstream Qdrant spelling — is the one a script checking "both" would miss. The cost is
        not cosmetic: the core resolves that key fine, so the operator gets a FAIL telling
        them to export what they already have, and the cutover is blocked by a false
        negative."""
        env = self.env(QDRANT_SERVICE_API_KEY=None, SERVER_API_KEY=None)
        env["QDRANT_API_KEY"] = SECRET
        env["QCTX_API_KEY"] = SECRET
        out = self.run_script(env=env)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertNoLine(out, "FAIL")
        self.assertLine(out, "QDRANT_API_KEY")

    def test_it_names_every_alias_the_core_accepts_when_the_key_is_missing(self):
        out = self.run_script(env=self.env(QDRANT_SERVICE_API_KEY=None, SERVER_API_KEY=None))
        self.assertEqual(out.returncode, 1, out.stdout)
        for alias in ("QCTX_QDRANT_API_KEY", "QDRANT_SERVICE_API_KEY", "QDRANT_API_KEY",
                      "QCTX_API_KEY", "SERVER_API_KEY"):
            self.assertLine(out, alias)

    def test_it_reports_the_alias_that_is_actually_set(self):
        """Printing the canonical name for a value that came from a legacy alias is
        misleading in exactly the debugging session this report exists for."""
        out = self.run_script()          # env() sets the two LEGACY names
        self.assertLine(out, "in this shell: QDRANT_SERVICE_API_KEY SERVER_API_KEY")
        self.assertNoLine(out, "in this shell: QCTX_QDRANT_API_KEY")

    def test_credentials_only_in_the_shell_warn_about_the_gateway(self):
        """hermes loads `$HERMES_HOME/.env` itself (`hermes_cli/env_loader.py`, called from
        run_agent.py and cli.py), but a systemd/gateway hermes inherits no shell."""
        out = self.run_script()
        self.assertEqual(out.returncode, 0, out.stdout)
        self.assertLine(out, "WARN")
        self.assertLine(out, ".env")

    def test_credentials_in_the_dotenv_silence_that_warning(self):
        """The other direction, so the WARN is not something the script always prints."""
        (self.hermes / ".env").write_text(
            f"QDRANT_SERVICE_API_KEY={SECRET}\nSERVER_API_KEY={SECRET}\n")
        out = self.run_script()
        self.assertEqual(out.returncode, 0, out.stdout)
        self.assertNoLine(out, "will not have it")
        self.assertLine(out, f"in {self.hermes}/.env: QDRANT_SERVICE_API_KEY SERVER_API_KEY")

    def test_the_dotenv_alone_satisfies_the_credential_check(self):
        """hermes loads it with override=True before the provider is asked anything, so
        keys that exist ONLY there are enough — for the gateway too."""
        (self.hermes / ".env").write_text(
            f"QCTX_QDRANT_API_KEY={SECRET}\nQCTX_API_KEY={SECRET}\n")
        out = self.run_script(env=self.env(QDRANT_SERVICE_API_KEY=None, SERVER_API_KEY=None))
        self.assertEqual(out.returncode, 0, out.stdout)
        self.assertNoLine(out, "FAIL")

    def test_the_export_form_in_the_dotenv_counts(self):
        """`export KEY=value` is accepted by python-dotenv, so it has to be accepted here
        too — otherwise the script FAILS on a file hermes reads perfectly well."""
        (self.hermes / ".env").write_text(
            f"export QCTX_QDRANT_API_KEY={SECRET}\nexport SERVER_API_KEY={SECRET}\n")
        out = self.run_script(env=self.env(QDRANT_SERVICE_API_KEY=None, SERVER_API_KEY=None))
        self.assertEqual(out.returncode, 0, out.stdout)
        self.assertNoLine(out, "FAIL")

    def test_a_key_assigned_nothing_does_not_count(self):
        """`KEY=` reads as configured and authenticates nothing."""
        (self.hermes / ".env").write_text("QCTX_QDRANT_API_KEY=\nQCTX_API_KEY=\n")
        out = self.run_script(env=self.env(QDRANT_SERVICE_API_KEY=None, SERVER_API_KEY=None))
        self.assertEqual(out.returncode, 1, out.stdout)
        self.assertLine(out, "FAIL")

    def test_a_commented_out_key_in_the_dotenv_does_not_count(self):
        (self.hermes / ".env").write_text(
            "# QCTX_QDRANT_API_KEY=x\n#QCTX_API_KEY=y\n")
        out = self.run_script(env=self.env(QDRANT_SERVICE_API_KEY=None, SERVER_API_KEY=None))
        self.assertEqual(out.returncode, 1, out.stdout)
        self.assertLine(out, "FAIL")

    def test_settings_that_exist_only_in_the_shell_are_reported_as_a_gateway_gap(self):
        """The other half of the same failure, and the half the credential remedy cannot fix.
        With the URLs exported from ~/.bashrc and absent from config.json — the state on this
        machine — an interactive hermes has memory and a gateway hermes does not, because
        `require_qdrant()` fails on the URL and not on the key. Putting the KEYS in .env, the
        only remedy the first check prints, would leave that operator still memory-less."""
        self.qctx_config.write_text('{"memory_collection": "fake_memory"}\n')
        env = self.env()
        env["QCTX_QDRANT_URL"] = "http://127.0.0.1:6333"
        env["QCTX_EMBED_URL"] = "http://127.0.0.1:8000/v1/embeddings"
        out = self.run_script(env=env)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertLine(out, "the plugin reports itself available")   # in THIS shell, yes
        self.assertLine(out, "with no shell environment the plugin is NOT configured")
        self.assertLine(out, "qctx config set qdrant-url")

    def test_settings_in_the_config_file_are_reported_as_reachable(self):
        """The other direction, so the gateway warning is not something always printed."""
        out = self.run_script()
        self.assertEqual(out.returncode, 0, out.stdout)
        self.assertLine(out, "every non-secret setting is in config.json")
        self.assertNoLine(out, "NOT configured")

    def test_an_unconfigured_plugin_fails_with_the_reason_the_provider_gives(self):
        """The provider's own `unavailable_reason()`, not a message this script invents."""
        out = self.run_script(env=self.env(QCTX_CONFIG=str(self.tmp / "missing.json")))
        self.assertEqual(out.returncode, 1, out.stdout)
        self.assertLine(out, "FAIL")
        self.assertLine(out, "QCTX_")


class TestWhereTheProviderHasToLive(CutoverCase):
    """`$HERMES_HOME/plugins/<name>/`, FLAT — measured against the installed loader
    (`plugins/memory/__init__.py::_iter_provider_dirs`, `find_provider_dir`). The
    third-party provider on this machine sits at `$HERMES_HOME/plugins/memory/qdrant`,
    which that loader does not scan, so the wrong-but-plausible location is not
    hypothetical: it is the layout already on disk."""

    def test_the_wrong_but_plausible_location_is_reported_rather_than_ignored(self):
        wrong = self.hermes / "plugins" / "memory" / "memories"
        wrong.mkdir(parents=True)
        (wrong / "__init__.py").write_text("class X(MemoryProvider): pass\n")
        out = self.run_script()
        self.assertLine(out, "WARN")
        self.assertLine(out, str(wrong))

    def test_an_existing_correct_symlink_is_reported_as_installed(self):
        link = self.hermes / "plugins" / "memories"
        link.symlink_to(REPO / "hosts" / "hermes")
        out = self.run_script()
        self.assertEqual(out.returncode, 0, out.stdout)
        self.assertLine(out, "already installed")

    def test_a_symlink_pointing_somewhere_else_is_reported_as_repointed(self):
        link = self.hermes / "plugins" / "memories"
        link.symlink_to(self.tmp / "some-other-checkout")
        out = self.run_script()
        self.assertLine(out, "repoint")

    def test_a_replaced_provider_one_level_too_deep_is_reported_as_already_inert(self):
        """What is on this machine: the qdrant provider's files are at
        `$HERMES_HOME/plugins/memory/qdrant`, which discovery does not scan."""
        deep = self.hermes / "plugins" / "memory" / "qdrant"
        deep.mkdir(parents=True)
        (deep / "__init__.py").write_text("# MemoryProvider\n")
        out = self.run_script()
        self.assertEqual(out.returncode, 0, out.stdout)
        self.assertLine(out, "already inert")

    def test_a_replaced_provider_where_the_loader_looks_is_not_called_inert(self):
        """The other direction: a provider installed correctly must not be described as
        inert just because this check exists."""
        proper = self.hermes / "plugins" / "qdrant"
        proper.mkdir(parents=True)
        (proper / "__init__.py").write_text("# MemoryProvider\n")
        out = self.run_script()
        self.assertEqual(out.returncode, 0, out.stdout)
        self.assertNoLine(out, "already inert")
        self.assertLine(out, "is where the loader looks")

    def test_the_dry_run_from_a_git_worktree_warns_and_still_reports(self):
        """A worktree is a temporary checkout. The dry run changes nothing, so it warns."""
        script = self.fake_root(None, ".claude", "worktrees", "some-branch")
        out = self.run_script(script=script)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertLine(out, "WORKTREE")
        self.assertLine(out, "DRY RUN")

    def test_apply_from_a_git_worktree_is_REFUSED(self):
        """Not a warning: removing the worktree leaves a dangling `plugins/memories`, which
        discovery skips while `load_memory_provider` returns None — and hermes 0.20.1 warns
        only when the provider is not None (`agent/agent_init.py:1784-1798`). So the operator
        would get a hermes with no memory and no message anywhere. A WARN inside a long
        report is too easy to scroll past for a failure that silent."""
        script = self.fake_root(None, ".claude", "worktrees", "some-branch")
        out = self.run_script("--apply", script=script, env=self.env(
            HERMES_CUTOVER_SKIP_SUITE=None))
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertLine(out, "refusing to --apply")
        self.assertLine(out, "nothing was changed")
        self.assertEqual(self.config_yaml.read_text(), CONFIG_YAML)
        self.assertFalse((self.hermes / "plugins" / "memories").exists())

    def test_the_worktree_refusal_can_be_overridden_deliberately(self):
        script = self.fake_root(None, ".claude", "worktrees", "some-branch")
        out = self.run_script("--apply", "--i-know-its-a-worktree", script=script,
                              env=self.env(HERMES_CUTOVER_SKIP_SUITE=None))
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertLine(out, "WORKTREE")
        self.assertIn("provider: memories", self.config_yaml.read_text())

    def test_the_worktree_warning_is_not_printed_for_a_normal_checkout(self):
        out = self.run_script(script=self.fake_root(None, "dev", "memories-plugin"))
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertNoLine(out, "WORKTREE")

    def test_an_adapter_that_cannot_even_import_says_WHY_twice(self):
        """The one path where the operator has nothing else to go on.

        Every other failure has a later check that narrows it down. Here the adapter did not
        import at all, so the exception text IS the diagnosis — and the reader consumes four
        lines, so BOTH reason lines have to carry it. An earlier version printed the real
        message first and a generic "the adapter did not import" second, throwing the only
        clue away in the half the file-configuration check reports.
        """
        root = self.tmp / "unimportable"
        (root / "scripts").mkdir(parents=True)
        (root / "tests").mkdir()
        (root / "tests" / "test_fake.py").write_text(self.PASSING_TEST)
        shutil.copy2(SCRIPT, root / "scripts" / SCRIPT.name)
        for shared in ("cli", "core"):
            (root / shared).symlink_to(REPO / shared)
        (root / "hosts" / "hermes").mkdir(parents=True)
        # Carries the discovery marker, so THAT check passes and this one fails on its own.
        (root / "hosts" / "hermes" / "__init__.py").write_text(
            "def register_memory_provider():\n"
            "    pass\n"
            "raise RuntimeError('a very specific reason nobody could guess')\n")
        out = self.run_script(script=root / "scripts" / SCRIPT.name)
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertEqual(
            out.stdout.count("a very specific reason nobody could guess"), 2,
            "both reason lines must carry the real exception — the availability line AND "
            f"the file-configuration line:\n{out.stdout}")

    def test_the_discovery_marker_is_actually_checked(self):
        """The check whose failure mode is "the provider does not exist": without
        `register_memory_provider` or `MemoryProvider` in the first 8192 bytes,
        `_is_memory_provider_dir` does not even consider the directory a provider. Proved with
        an adapter copy whose marker is gone — the check has to notice."""
        root = self.tmp / "markerless"
        (root / "scripts").mkdir(parents=True)
        (root / "tests").mkdir()
        (root / "tests" / "test_fake.py").write_text(self.PASSING_TEST)
        shutil.copy2(SCRIPT, root / "scripts" / SCRIPT.name)
        for shared in ("cli", "core"):
            (root / shared).symlink_to(REPO / shared)
        (root / "hosts" / "hermes").mkdir(parents=True)
        # A file the loader's text scan would reject, and that the availability probe can
        # still import: the checks are independent and this one must fail on its own.
        (root / "hosts" / "hermes" / "__init__.py").write_text(
            "class MemoriesProvider:\n"
            "    name = 'memories'\n"
            "    def is_available(self): return True\n"
            "    def unavailable_reason(self): return ''\n"
        )
        out = self.run_script(script=root / "scripts" / SCRIPT.name)
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertLine(out, "neither register_memory_provider nor MemoryProvider")
        self.assertLine(out, "nothing was changed")

    def test_a_plugin_dir_override_that_the_memory_loader_ignores_is_reported(self):
        """`env.sh` exports HERMES_PLUGIN_DIR, and the memory loader does not read it —
        it builds `get_hermes_home() / "plugins"` itself. An operator who moved their
        plugins with that variable would install into a directory nothing scans."""
        out = self.run_script(env=self.env(HERMES_PLUGIN_DIR=str(self.tmp / "elsewhere")))
        self.assertLine(out, "HERMES_PLUGIN_DIR")


class TestTheSuiteCheck(CutoverCase):
    """The check runs `$ROOT/tests`, so it is provable against a fake ROOT: a copy of the
    script in a temp tree whose `tests/` holds one trivial test. Same script, same branch,
    both outcomes, and no recursion into the real suite."""

    def test_a_passing_suite_reports_ok(self):
        script = self.fake_root()
        out = self.run_script(script=script, env=self.env(HERMES_CUTOVER_SKIP_SUITE=None))
        self.assertLine(out, "offline suite passes")
        self.assertEqual(out.returncode, 0, out.stdout)

    def test_a_failing_suite_fails_the_script(self):
        script = self.fake_root(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_boom(self):\n        self.assertTrue(False)\n"
        )
        out = self.run_script(script=script, env=self.env(HERMES_CUTOVER_SKIP_SUITE=None))
        self.assertEqual(out.returncode, 1, out.stdout)
        self.assertLine(out, "FAIL")
        self.assertLine(out, "suite")

    def test_skipping_the_suite_is_announced_and_never_reads_as_a_pass(self):
        out = self.run_script()
        self.assertLine(out, "WARN")
        self.assertLine(out, "HERMES_CUTOVER_SKIP_SUITE")
        self.assertNoLine(out, "offline suite passes")

    def test_the_guard_does_NOT_buy_an_apply(self):
        """It gates the dry run only. A stray export in a shell would otherwise apply the
        cutover with the suite unverified, which is the one thing it must never buy."""
        out = self.run_script("--apply")          # env() sets the guard
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertLine(out, "refusing to --apply with the suite unverified")
        self.assertLine(out, "nothing was changed")
        self.assertEqual(self.config_yaml.read_text(), CONFIG_YAML)
        self.assertEqual(list(self.hermes.glob("config.yaml.bak-*")), [])


class TestApplyAgainstAFakeHome(CutoverCase):
    """`--apply` is NEVER run against the real `~/.hermes` — not by a test and not by a
    probe. Every case here writes only inside the temp tree built in setUp.

    These run the script from a fake ROOT, whose `tests/` holds one trivial passing test.
    Not for speed: `HERMES_CUTOVER_SKIP_SUITE` refuses to authorise an `--apply`, so an apply
    test has to give the script a suite it can actually pass. The fake ROOT is also outside
    any `.claude/worktrees/` path, which the worktree refusal reads.
    """

    def setUp(self):
        super().setUp()
        self.script = self.fake_root()
        self.root = self.script.parent.parent

    def run_script(self, *args, script=None, env=None):
        env = dict(env if env is not None else self.env())
        env.pop("HERMES_CUTOVER_SKIP_SUITE", None)

        return super().run_script(*args, script=script or self.script, env=env)

    def link(self):
        return self.hermes / "plugins" / "memories"

    def test_it_installs_the_symlink_and_selects_the_provider(self):
        out = self.run_script("--apply")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertTrue(self.link().is_symlink())
        self.assertEqual(Path(os.readlink(self.link())), self.root / "hosts" / "hermes")
        self.assertEqual((self.root / "hosts" / "hermes").resolve(),
                         (REPO / "hosts" / "hermes").resolve())
        text = self.config_yaml.read_text()
        self.assertIn("provider: memories", text)
        self.assertNotIn("provider: qdrant", text)

    def test_it_preserves_every_other_key_and_backs_the_file_up(self):
        self.run_script("--apply")
        text = self.config_yaml.read_text()
        for kept in ("memory_enabled: true", "nudge_interval: 10", "max_iterations: 80",
                     "default: MiniMax-M2.7"):
            self.assertIn(kept, text)
        backups = list(self.hermes.glob("config.yaml.bak-*"))
        self.assertEqual(len(backups), 1, f"expected exactly one backup, got {backups}")
        self.assertEqual(backups[0].read_text(), CONFIG_YAML, "the backup is not the original")

    def test_it_is_idempotent(self):
        first = self.run_script("--apply")
        self.assertEqual(first.returncode, 0, first.stdout)
        second = self.run_script("--apply")
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(self.config_yaml.read_text().count("provider: memories"), 1)

    def test_it_inserts_the_key_when_the_memory_block_has_no_provider(self):
        self.config_yaml.write_text("memory:\n  memory_enabled: true\nother:\n  keep: yes\n")
        out = self.run_script("--apply")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        text = self.config_yaml.read_text()
        self.assertIn("  provider: memories", text)
        self.assertIn("keep: yes", text)

    def test_it_creates_the_block_when_there_is_no_memory_section(self):
        self.config_yaml.write_text("other:\n  keep: yes\n")
        out = self.run_script("--apply")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        text = self.config_yaml.read_text()
        self.assertIn("memory:", text)
        self.assertIn("provider: memories", text)
        self.assertIn("keep: yes", text)

    def test_it_does_not_touch_a_provider_under_a_key_that_is_not_memory(self):
        """`auxiliary.memory_query_rewrite.provider` exists in the real config.yaml, and
        `memory_enabled` starts with the same six letters as the block being edited.

        NOTE what this does NOT prove: with `auxiliary:` BEFORE `memory:`, the `inside` gate
        alone protects that key and the indent guard is never reached. The test below is the
        one that exercises the guard."""
        self.config_yaml.write_text(
            "auxiliary:\n"
            "  memory_query_rewrite:\n"
            "    provider: auto\n"
            "memory:\n"
            "  provider: qdrant\n"
        )
        out = self.run_script("--apply")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        text = self.config_yaml.read_text()
        self.assertIn("provider: auto", text)
        self.assertIn("provider: memories", text)

    def test_it_does_not_touch_a_provider_nested_DEEPER_INSIDE_the_memory_block(self):
        """The case the indent guard exists for, which nothing exercised: a `provider:` under a
        sub-key of `memory:`, ahead of the direct child. Without the guard the rewriter edits
        the nested one, `memory.provider` stays qdrant, the sub-key loses its child — and every
        signal the script had before this test looked like success."""
        self.config_yaml.write_text(
            "memory:\n"
            "  backends:\n"
            "    provider: some-backend\n"
            "  provider: qdrant\n"
            "delegation:\n"
            "  max_iterations: 80\n"
        )
        out = self.run_script("--apply")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        text = self.config_yaml.read_text()
        self.assertIn("    provider: some-backend", text, "the nested key was edited")
        self.assertIn("  provider: memories", text)
        self.assertNotIn("provider: qdrant", text)
        self.assertEqual(text.count("provider: memories"), 1)
        self.assertIn("max_iterations: 80", text)

    def test_it_refuses_to_claim_a_success_it_has_not_verified(self):
        """A file that already holds TWO direct `provider:` children — legal input, since
        PyYAML keeps the last one and does not complain. The rewriter replaces the first, so
        after the write `memory.provider` STILL resolves to qdrant. The rewriter exiting 0 and
        `mv` succeeding are not evidence; re-reading the key is. Printing ok for a state it had
        not verified is the defect `scripts/cutover.sh` already paid for once."""
        self.config_yaml.write_text("memory:\n  provider: qdrant\n  provider: qdrant\n")
        out = self.run_script("--apply")
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertLine(out, "the rewrite did not take")
        self.assertLine(out, "one or more steps FAILED")
        self.assertNoLine(out, "=== now ===")

    def test_it_leaves_the_replaced_provider_on_disk(self):
        """Disabled by configuration, never deleted: its points are the user's data and a
        wrong call here is only reversible while the files still exist."""
        old = self.hermes / "plugins" / "memory" / "qdrant"
        old.mkdir(parents=True)
        (old / "__init__.py").write_text("# MemoryProvider\n")
        out = self.run_script("--apply")
        # An apply that actually ran, then the absence check — otherwise this passes
        # against a script that did nothing at all.
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("provider: memories", self.config_yaml.read_text())
        self.assertTrue((old / "__init__.py").exists(), "the old provider was removed")

    def unwritable_plugins(self):
        """An APPLY-TIME failure, which is the only kind that can prove the exit code
        carries the apply block. A non-symlink sitting where the link goes does not work
        for this: the checks catch that one first and refuse to touch anything (which is
        the right behaviour — see the test below). An unwritable plugins/ passes every
        check and then makes `ln` fail."""
        plugins = self.hermes / "plugins"
        plugins.chmod(0o500)
        self.addCleanup(plugins.chmod, 0o700)

    def test_a_failed_step_makes_the_script_exit_non_zero(self):
        """`scripts/cutover.sh` once printed FAIL and then exited 0 from exactly this
        shape: the apply block set its own failure flag and nothing looked at it again."""
        self.unwritable_plugins()
        out = self.run_script("--apply")
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertLine(out, "FAIL")
        self.assertLine(out, "one or more steps FAILED")
        self.assertNoLine(out, "=== now ===")

    def test_a_failed_step_does_not_stop_the_later_ones_from_running(self):
        """`set -e` must not abort mid-cutover: the config rewrite is independent of the
        symlink and still has to happen (or be refused) on its own, with both outcomes
        reported. Aborting here is what leaves a cutover half done and unexplained."""
        self.unwritable_plugins()
        out = self.run_script("--apply")
        self.assertLine(out, "memory.provider = memories")
        self.assertIn("provider: memories", self.config_yaml.read_text())

    def test_a_real_directory_where_the_symlink_goes_is_refused_before_anything_changes(self):
        (self.hermes / "plugins" / "memories").mkdir()
        out = self.run_script("--apply")
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertLine(out, "is NOT a symlink")
        self.assertLine(out, "nothing was changed")
        self.assertEqual(self.config_yaml.read_text(), CONFIG_YAML)
        self.assertEqual(list(self.hermes.glob("config.yaml.bak-*")), [])

    def test_a_plugins_path_that_cannot_be_created_is_reported_like_every_other_step(self):
        """`mkdir -p` failing used to exit through `set -e` with no message, unlike every
        neighbouring step — the operator would see the applying banner and nothing else.
        A FILE where the plugins directory goes is the cheapest way to make it fail."""
        (self.hermes / "plugins").rmdir()
        (self.hermes / "plugins").write_text("not a directory\n")
        out = self.run_script("--apply")
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertLine(out, "could not create")
        self.assertLine(out, "one or more steps FAILED")

    def test_failed_checks_gate_the_apply_and_write_nothing(self):
        out = self.run_script("--apply", env=self.env(SERVER_API_KEY=None))
        self.assertEqual(out.returncode, 1, out.stdout)
        self.assertLine(out, "nothing was changed")
        self.assertEqual(self.config_yaml.read_text(), CONFIG_YAML)
        self.assertFalse(self.link().exists())
        self.assertEqual(list(self.hermes.glob("config.yaml.bak-*")), [])

    def test_it_prints_how_to_go_back(self):
        """Both strings this used to match appear elsewhere in the report, so deleting the
        "To go back" line entirely left it green. Asserted on the line itself now, with the
        two paths a reversal needs."""
        out = self.run_script("--apply")
        backup = next(iter(self.hermes.glob("config.yaml.bak-*")))
        self.assertLine(out, f"To go back: restore {backup} and remove {self.link()}")


if __name__ == "__main__":
    unittest.main()
