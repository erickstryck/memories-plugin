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
import json
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

    # ------------------------------------------------------- the big-file guard, applied

    def test_apply_registers_the_guard_under_the_event_KEY_hermes_parses(self):
        """The shape is the whole test, and it is measured rather than styled.

        `agent/shell_hooks.py::_parse_hooks_block` (v0.20.1, installed) starts with
        `if not isinstance(hooks_cfg, dict): return []` and iterates `hooks_cfg.items()`
        as event name -> LIST of entries. A `hooks:` written as a sequence of
        `{event, matcher, command}` mappings — the shape this plan specified — parses to
        ZERO hooks and logs nothing at all on that path: the guard is installed, reported
        as installed, and never runs.
        """
        out = self.run_script("--apply")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        text = self.config_yaml.read_text()
        self.assertIn("hooks:\n", text)
        after = text.split("hooks:\n", 1)[1].splitlines()
        first = next(line for line in after if line.strip())
        self.assertEqual(first.strip(), "pre_tool_call:",
                         f"`hooks:` must map event names to lists, got {first!r}")
        self.assertRegex(text, r"\n {4}- matcher: read_file\n {6}command: ")
        self.assertIn(f'command: python3 "{self.root}/hosts/hermes/bigfile.py"', text)

    def test_apply_bounds_the_guards_runtime_instead_of_taking_the_host_default(self):
        """`DEFAULT_TIMEOUT_SECONDS = 60` in the installed shell_hooks.py. Sixty seconds in
        front of every file read is not a guard, it is a hang; the sibling host gives the
        same hook 5 (`hooks/hooks.json`)."""
        self.run_script("--apply")
        self.assertIn("      timeout: 5\n", self.config_yaml.read_text())

    def test_apply_is_idempotent_for_the_guard_too(self):
        self.assertEqual(self.run_script("--apply").returncode, 0)
        second = self.run_script("--apply")
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        text = self.config_yaml.read_text()
        self.assertEqual(text.count("bigfile.py"), 1, text)
        self.assertEqual(text.count("pre_tool_call:"), 1, text)

    def test_apply_appends_to_an_existing_event_list_and_keeps_the_other_hooks(self):
        """A `hooks:` block already carrying entries is the normal case for anyone who has
        used the feature, and losing one of them would be a silent removal of the user's
        own tooling."""
        self.config_yaml.write_text(
            CONFIG_YAML
            + "hooks:\n"
              "  post_tool_call:\n"
              "    - command: /usr/bin/true\n"
              "  pre_tool_call:\n"
              "    - matcher: terminal\n"
              "      command: /usr/bin/false\n"
        )
        out = self.run_script("--apply")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        text = self.config_yaml.read_text()
        self.assertIn("    - command: /usr/bin/true\n", text)
        self.assertIn("      command: /usr/bin/false\n", text)
        self.assertIn("bigfile.py", text)
        self.assertEqual(text.count("pre_tool_call:"), 1, text)

    def test_apply_does_not_mistake_a_DEEPER_pre_tool_call_for_the_event_key(self):
        """`outbound:` is a real reserved sub-key of `hooks:` (`_parse_hooks_block` skips it
        by name, alongside `output_spill`), and anything under it is not an event list.

        Matching `pre_tool_call:` at any indent puts our entry inside that sub-section, where
        hermes never looks — and then the re-read finds it with the same wrong rule and
        reports success. A guard installed into a place nothing reads, reported as
        installed, is the exact failure this whole feature is about.
        """
        self.config_yaml.write_text(
            CONFIG_YAML
            + "hooks:\n"
              "  outbound:\n"
              "    pre_tool_call:\n"
              "      - command: /usr/bin/true\n"
        )
        out = self.run_script("--apply")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        text = self.config_yaml.read_text()
        self.assertIn("      - command: /usr/bin/true\n", text)
        self.assertRegex(text, r"\n  pre_tool_call:\n {4}- matcher: read_file\n",
                         "the guard did not land on a direct child of hooks:")

    def test_apply_refuses_a_hooks_block_it_cannot_extend_and_says_so(self):
        """The sequence shape hermes silently ignores. Rewriting it into a mapping would
        disable every hook the user has; leaving it and reporting `ok` would be the
        already-paid-for defect. It refuses, loudly, and changes nothing in that block."""
        broken = (CONFIG_YAML
                  + "hooks:\n"
                    "  - event: pre_tool_call\n"
                    "    command: /usr/bin/true\n")
        self.config_yaml.write_text(broken)
        out = self.run_script("--apply")
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertLine(out, "one or more steps FAILED")
        text = self.config_yaml.read_text()
        self.assertNotIn("bigfile.py", text)
        self.assertIn("  - event: pre_tool_call\n", text)
        # The independent step still ran: a refusal here is not an abort.
        self.assertIn("provider: memories", text)

    def test_apply_refuses_a_pre_tool_call_it_cannot_append_to(self):
        """`pre_tool_call:` mapped to a scalar is a legal file that hermes parses to zero
        hooks (`hooks.%s must be a list`), and appending to it is not possible — so the
        write cannot take, and the script has to say that rather than print the line it
        hoped for.

        WHAT THIS DOES NOT REACH, and its old name claimed it did: the post-write re-read.
        The rewriter exits 3 here, so the failure is reported before anything is written at
        all — measured, by replacing the re-read's condition with `true` and watching the
        whole suite stay green. The test below is the one that reaches it.
        """
        self.config_yaml.write_text(CONFIG_YAML + "hooks:\n  pre_tool_call: disabled\n")
        out = self.run_script("--apply")
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertLine(out, "could not register the big-file guard")
        self.assertNoLine(out, "=== now ===")
        self.assertNotIn("bigfile.py", self.config_yaml.read_text())

    def test_apply_re_reads_the_guard_instead_of_trusting_the_rewrite(self):
        """The branch the re-read OWNS, and nothing had ever reached it: the rewrite
        succeeds, `mv` succeeds, and the file still does not hold what hermes needs.

        An entry that already runs THIS command is left exactly as it is — the rewriter is
        idempotent on purpose, and an entry the user wrote is the user's. So an entry
        carrying the wrong matcher survives the write intact, and every signal short of
        re-reading looks like success: the rewriter exits 0, `mv` returns 0, and the file is
        byte-for-byte what it was. Only asking hermes' own parser what fires can tell that
        `read_file` does not.

        The matcher is not a detail: `write_file` and `patch` take a `path` too, so the
        entry above is a read-cost guard wired to WRITES. Printing "registered: matcher
        read_file" for it is the same defect `scripts/cutover.sh` already paid for once —
        reporting a state it had not verified.
        """
        guard_cmd = f'python3 "{self.root}/hosts/hermes/bigfile.py"'
        self.config_yaml.write_text(
            CONFIG_YAML
            + "hooks:\n"
              "  pre_tool_call:\n"
              "    - matcher: write_file\n"
              f"      command: {guard_cmd}\n")
        out = self.run_script("--apply")
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertLine(out, "the big-file guard did not take")
        self.assertLine(out, "one or more steps FAILED")
        self.assertNoLine(out, "matcher read_file ->")
        self.assertNoLine(out, "=== now ===")
        # The independent step still ran, and the user's entry was not rewritten behind
        # their back: the refusal is a report, not an edit.
        text = self.config_yaml.read_text()
        self.assertIn("provider: memories", text)
        self.assertIn("- matcher: write_file\n", text)
        self.assertEqual(text.count("bigfile.py"), 1, text)

    def test_the_real_hermes_config_is_never_touched_by_an_apply(self):
        """The one file in this suite that a bug here would destroy for real. Asserted on
        content AND mtime: a rewrite that produced identical bytes would still prove the
        script had opened the user's live configuration."""
        real = Path.home() / ".hermes" / "config.yaml"
        if str(real).startswith(str(self.home)):     # pragma: no cover - HOME is faked
            self.skipTest("HOME is already redirected; there is no real file to protect")
        if not real.exists():
            self.skipTest(f"no {real} on this machine")
        before = (real.read_bytes(), real.stat().st_mtime_ns)
        out = self.run_script("--apply")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertEqual((real.read_bytes(), real.stat().st_mtime_ns), before,
                         f"{real} was modified by a test")


class TestTheBigFileGuardCheck(CutoverCase):
    """The guard is useless if hermes never calls it, and a cutover script that stays
    silent about that is the defect this script already paid for once.

    Two questions, and the second one exists because "installed" is not "working": is the
    hook registered at all, and is the context window DECLARED — because without a declared
    window the guard falls back to a table of per-name CEILINGS, and a ceiling that is too
    large only makes the guard sleep. Silently inert is indistinguishable from working, and
    the user finds out on the day it did not protect them.
    """

    def guard_command(self, root=REPO):
        return f'python3 "{root}/hosts/hermes/bigfile.py"'

    def with_hooks(self, command, matcher="read_file", event="pre_tool_call"):
        entry = f"    - command: {command}\n"
        if matcher is not None:
            entry = f"    - matcher: {matcher}\n      command: {command}\n"
        self.config_yaml.write_text(f"{CONFIG_YAML}hooks:\n  {event}:\n{entry}")

    def test_the_report_says_whether_the_big_file_guard_is_registered(self):
        out = self.run_script()
        self.assertLine(out, "big-file guard")

    def test_an_unregistered_guard_names_the_path_DERIVED_from_this_checkout(self):
        """Run from a copy of the script in a temp directory: the path it prints can only
        be right if it was derived from `$ROOT`, the way the provider symlink target
        already is. A path typed into the script would be somebody's home directory."""
        script = self.fake_root()
        root = script.parent.parent
        out = self.run_script(script=script)
        self.assertLine(out, f"{root}/hosts/hermes/bigfile.py")

    def test_a_guard_already_registered_is_reported_as_registered(self):
        self.with_hooks(self.guard_command())
        out = self.run_script()
        self.assertLine(out, "big-file guard registered")

    def test_a_guard_registered_with_no_matcher_is_reported_as_too_broad(self):
        """Measured, and it is why the matcher is not a detail: `write_file` and `patch`
        take a `path` argument too, so a matcher-less `pre_tool_call` hook gets an opinion
        about WRITES, which this guard has no business blocking."""
        self.with_hooks(self.guard_command(), matcher=None)
        out = self.run_script()
        self.assertLine(out, "registered with no matcher")
        self.assertLine(out, "write_file")

    def test_a_guard_pointing_at_another_checkout_is_reported_and_not_left_implicit(self):
        self.with_hooks('python3 "/somewhere/else/hosts/hermes/bigfile.py"')
        out = self.run_script()
        self.assertLine(out, "/somewhere/else/hosts/hermes/bigfile.py")

    def test_a_hooks_block_written_as_a_LIST_is_reported_as_inert(self):
        """`_parse_hooks_block` returns [] for a non-dict without logging anything on that
        path. Nothing else in the system would ever mention it."""
        self.config_yaml.write_text(
            f"{CONFIG_YAML}hooks:\n  - event: pre_tool_call\n    command: /usr/bin/true\n")
        out = self.run_script()
        self.assertLine(out, "WARN")
        self.assertLine(out, "hermes reads no hook at all from it")

    def test_a_guard_that_is_not_allowlisted_yet_is_reported(self):
        """Registration is only half of it: `register_from_config` skips any hook whose
        (event, command) pair is not in `shell-hooks-allowlist.json` unless a TTY approves
        it. A gateway hermes has no TTY, so it skips the hook and logs a warning nobody
        reads."""
        self.with_hooks(self.guard_command())
        out = self.run_script()
        self.assertLine(out, "shell-hooks-allowlist.json")

    def test_an_allowlisted_guard_is_reported_as_approved(self):
        self.with_hooks(self.guard_command())
        (self.hermes / "shell-hooks-allowlist.json").write_text(json.dumps(
            {"approvals": [{"event": "pre_tool_call",
                            "command": self.guard_command()}]}))
        out = self.run_script()
        self.assertLine(out, "approved in shell-hooks-allowlist.json")

    def test_an_undeclared_context_window_says_which_ceiling_and_what_it_costs(self):
        self.config_yaml.write_text(CONFIG_YAML.replace("MiniMax-M2.7", "claude-opus-5"))
        out = self.run_script()
        self.assertLine(out, "context_window is not declared")
        self.assertLine(out, "ceiling")
        self.assertLine(out, "QCTX_CONTEXT_WINDOW")

    def test_the_ceiling_it_prints_is_the_one_the_TABLE_holds(self):
        """Not a number typed into the script. The table is the single owner of these, and
        a copy in a shell script is the kind of divergence this repo has paid for."""
        from core.windows import MODEL_WINDOWS
        self.config_yaml.write_text(CONFIG_YAML.replace("MiniMax-M2.7", "claude-opus-5"))
        out = self.run_script()
        self.assertLine(out, str(MODEL_WINDOWS["claude-opus-5"]))

    def test_a_model_the_table_does_not_know_is_reported_as_no_guard_at_all(self):
        """The measured hermes case, and the reason this line exists: the table holds
        claude names, `model.default` on this machine is `MiniMax-M2.7`, so `window_for`
        returns 0, and 0 ALLOWS. Reported, because a guard that never fires and never says
        why is exactly what this feature promised not to be."""
        out = self.run_script()
        self.assertLine(out, "MiniMax-M2.7")
        self.assertLine(out, "allows every read")

    def test_a_declared_context_window_is_reported_as_declared(self):
        out = self.run_script(env=self.env(QCTX_CONTEXT_WINDOW="123456"))
        self.assertLine(out, "context window declared: 123456")

    def test_a_context_window_only_in_the_shell_is_reported_as_a_gateway_gap(self):
        """The same question the credentials section asks, for the same reason: a
        systemd/gateway hermes inherits no interactive shell, and `context_window` is not a
        secret, so it can and should live in the file both hosts read."""
        out = self.run_script(env=self.env(QCTX_CONTEXT_WINDOW="123456"))
        self.assertLine(out, "qctx config set context-window")

    def test_a_context_window_in_the_config_file_does_not_warn(self):
        self.qctx_config.write_text(self.qctx_config.read_text()
                                    .replace("{", '{"context_window": 654321, ', 1))
        out = self.run_script()
        self.assertLine(out, "context window declared: 654321")
        self.assertNoLine(out, "qctx config set context-window")

    def with_endpoint(self, base_url="https://server.example/api/v1"):
        """`CONFIG_YAML` with an active `model:` block that also names an endpoint — the
        shape `hosts.hermes.endpoint.from_hermes_config` reads — so the cache-based branch
        below has something to resolve the window against."""
        self.config_yaml.write_text(
            CONFIG_YAML.replace("model:\n  default: MiniMax-M2.7\n",
                                f"model:\n  default: MiniMax-M2.7\n  base_url: {base_url}\n"))

        return base_url

    def _seed_window_cache(self, base_url, model, window, ttl=None):
        """Writes into the SAME cache `refresh_window` fills, under THIS test's own
        `QCTX_STATE_DIR` — never a value the script invents, and never the real state
        directory. A subprocess, and not an in-process call: the script itself always runs
        as a subprocess, so this keeps the two on the same footing regarding environment."""
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from core import windowcache\n"
            "windowcache.put(%r, %r, %r%s)\n"
        ) % (str(REPO), base_url, model, window, f", ttl={ttl}" if ttl is not None else "")
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                             env=self.env())
        self.assertEqual(out.returncode, 0, out.stderr)

    def test_a_window_learned_from_the_endpoint_is_reported_and_not_called_inert(self):
        """Re-review round 2 found the diagnostic LYING: after this feature shipped, it kept
        printing 'no ceiling is known … the guard is installed and inert' whether the probe
        had learned a real window or had learned nothing at all — the exact failure this
        feature exists to remove. The cache is the one place that tells the two apart."""
        base = self.with_endpoint()
        self._seed_window_cache(base, "MiniMax-M2.7", 524288)
        out = self.run_script()
        self.assertLine(out, "window learned from the endpoint: 524288 tokens")
        self.assertLine(out, "fresh")
        self.assertNoLine(out, "is installed and inert")

    def test_a_stale_learned_window_is_still_reported_as_learned_not_unknown(self):
        """Stale beats absent — the guard's own cascade never discards a stale cached
        window — so the diagnostic must say STALE, not fall silent about it."""
        base = self.with_endpoint()
        self._seed_window_cache(base, "MiniMax-M2.7", 262144, ttl=-3600)
        out = self.run_script()
        self.assertLine(out, "window learned from the endpoint: 262144 tokens")
        self.assertLine(out, "STALE")
        self.assertNoLine(out, "is installed and inert")

    def test_the_learned_window_outranks_the_ceiling_table(self):
        """A model the ceiling table ALSO knows must still report the LEARNED value and not
        the table's: `core/windows.py`'s own cascade consults the cache before the ceiling,
        and this diagnostic has to agree with the guard it is describing."""
        base = self.with_endpoint()
        self.config_yaml.write_text(
            self.config_yaml.read_text().replace("MiniMax-M2.7", "claude-opus-5"))
        self._seed_window_cache(base, "claude-opus-5", 999000)
        out = self.run_script()
        self.assertLine(out, "window learned from the endpoint: 999000 tokens")
        self.assertNoLine(out, "falls back to the ceiling")

    def test_no_endpoint_configured_the_cache_branch_is_silent(self):
        """Without a `base_url` there is nothing to key the cache on, and the diagnostic
        must fall through to the ceiling/unknown branches exactly as it did before this
        round's fix — the default fixture carries no endpoint."""
        out = self.run_script()
        self.assertNoLine(out, "window learned from the endpoint")

    def test_no_secret_is_ever_printed_by_the_cache_branch(self):
        base = self.with_endpoint()
        self._seed_window_cache(base, "MiniMax-M2.7", 524288)
        out = self.run_script()
        self.assertNoLine(out, SECRET)

    def test_the_dry_run_says_the_hook_is_part_of_what_changes(self):
        out = self.run_script()
        self.assertLine(out, "DRY RUN")
        self.assertLine(out, "pre_tool_call")



class TestNoCheckIsDECIDEDByAPipeline(unittest.TestCase):
    """The bug that was mistaken for a flaky test, pinned so it cannot come back.

    `head -c 8192 … | grep -qE …` decided the discovery check, under `set -o pipefail`. `grep -q`
    exits the instant it matches; when it won the race against `head`'s next write, `head` died
    of SIGPIPE, the pipeline reported 141, and a check that had PASSED was reported as failed.

    Measured on 2026-08-18: 1 failure in 600 runs of that exact pipeline on a busy machine, 0 in
    400 on an idle one. That load-dependence is why it only ever appeared inside a full suite run
    and was written off as a flaky test for weeks. It was not a test problem: on a correct
    installation the script would refuse to apply, at random, and blame the adapter for it.

    THE RULE THIS HOLDS, which is broader than the one bug: under `pipefail` the exit status of a
    pipeline is not a property of the question being asked — any reader that stops early can
    turn a true answer into a false one. So no check in this script may be decided by one.
    """

    #: `| grep -q`, `| grep -qE`, `| head -1`, and friends — a reader that can exit early.
    EARLY_EXIT_READERS = re.compile(r"\|\s*(grep\s+-[a-zA-Z]*q|head\b|sed\s+-n\s*'?1q)")

    def test_the_script_sets_pipefail_so_this_rule_is_load_bearing(self):
        """If it ever stops setting pipefail, this whole class is about a risk that is gone —
        and a test whose reason has expired should say so rather than pass by accident."""
        self.assertRegex(SCRIPT.read_text(), r"(?m)^set -[a-z]*e[a-z]*u[a-z]*o pipefail|^set -o pipefail")

    def test_no_early_exiting_reader_decides_a_condition(self):
        """Only pipelines in a CONDITION matter: `if …`, `while …`, `&&`/`||`. A pipeline whose
        output is printed cannot silently invert a decision."""
        offenders = []
        for number, line in enumerate(SCRIPT.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            in_condition = stripped.startswith(("if ", "elif ", "while ", "until ")) \
                or " && " in stripped or " || " in stripped
            if in_condition and self.EARLY_EXIT_READERS.search(stripped):
                offenders.append(f"{SCRIPT.name}:{number}: {stripped}")
        self.assertEqual(offenders, [], "a check is decided by a pipeline that can lose a race:\n"
                                        + "\n".join(offenders))


class TestTheDiscoveryMarkerCheckStillDecidesBOTHWays(unittest.TestCase):
    """Rewriting the check must not have made it always-true, which is the cheap way to make a
    flake go away and the reason to test both directions rather than just the green one."""

    def test_the_marker_present_is_recognised(self):
        self.assertTrue(_marker_check_says_ok("class MemoryProvider:\n"))

    def test_the_marker_present_LATE_in_a_big_file_is_recognised(self):
        """Within the 8192 bytes the loader itself reads — the check exists to mirror that
        window, so a marker at byte 8000 must still count."""
        filler = "# " + "x" * 60 + "\n"
        text = filler * 100 + "def register_memory_provider():\n"
        self.assertLess(len(text.encode()), 8192)
        self.assertTrue(_marker_check_says_ok(text))

    def test_the_marker_ABSENT_is_still_a_failure(self):
        self.assertFalse(_marker_check_says_ok("# nothing the loader greps for\n"))

    def test_the_marker_BEYOND_the_window_is_a_failure(self):
        """The loader reads 8192 bytes and no more, so a marker at byte 9000 is invisible to it
        and the check has to agree — reporting a provider hermes would skip is the failure this
        check exists to prevent."""
        text = ("# " + "y" * 78 + "\n") * 120 + "class MemoryProvider:\n"
        self.assertGreater(len("".join(text).encode()), 8192)
        self.assertFalse(_marker_check_says_ok(text))


def _marker_check_says_ok(file_text: str) -> bool:
    """Runs the script's own check, extracted verbatim, over a file holding `file_text`.

    Extracted rather than restated: a copy of the logic here would be a second implementation
    that could agree with a broken script. The lines come from the script itself.
    """
    lines = SCRIPT.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("marker_head="))
    end = next(i for i, ln in enumerate(lines[start:], start) if ln.strip() == "esac")
    body = "\n".join(lines[start:end + 1])
    body = body.replace('ok "', 'echo "OK:').replace('fail "', 'echo "FAIL:').replace('say "', ': "')
    with tempfile.TemporaryDirectory() as box:
        target = Path(box) / "hermes"
        target.mkdir()
        (target / "__init__.py").write_text(file_text)
        out = subprocess.run(["bash", "-c", f'set -euo pipefail\nTARGET="{target}"\n{body}'],
                             capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, f"the extracted check itself errored: {out.stderr}"

    return "OK:" in out.stdout


if __name__ == "__main__":
    unittest.main()
