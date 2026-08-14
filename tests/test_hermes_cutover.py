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
            # The suite this script runs CONTAINS the tests that run this script. Without
            # the guard the first `--apply` test would re-enter the whole suite.
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

    def assertLine(self, out, needle):
        self.assertIn(needle, out.stdout + out.stderr,
                      f"expected a line containing {needle!r}:\n{out.stdout}{out.stderr}")

    def assertNoLine(self, out, needle):
        self.assertNotIn(needle, out.stdout + out.stderr,
                         f"did NOT expect {needle!r}:\n{out.stdout}{out.stderr}")


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
        out = self.run_script()
        self.assertLine(out, "qdrant")
        self.assertLine(out, "memory.provider")
        self.assertLine(out, "search-collections")

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

    def copy_at(self, *parts):
        """The script copied to a chosen path, because what it reports here depends on the
        path it runs FROM — and this repo is itself checked out in a worktree, so the
        no-warning case cannot be shown from the real one."""
        root = self.tmp.joinpath(*parts)
        (root / "scripts").mkdir(parents=True)
        (root / "hosts").symlink_to(REPO / "hosts")
        shutil.copy2(SCRIPT, root / "scripts" / SCRIPT.name)

        return root / "scripts" / SCRIPT.name

    def test_installing_from_a_git_worktree_is_reported(self):
        """A worktree is a temporary checkout. Removing it leaves hermes with a dangling
        `plugins/memories`, which discovery skips — no provider and no error."""
        out = self.run_script(script=self.copy_at(".claude", "worktrees", "some-branch"))
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertLine(out, "WORKTREE")

    def test_the_worktree_warning_is_not_printed_for_a_normal_checkout(self):
        out = self.run_script(script=self.copy_at("dev", "memories-plugin"))
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertNoLine(out, "WORKTREE")

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

    def fake_root(self, test_body):
        root = self.tmp / "fake-root"
        (root / "scripts").mkdir(parents=True)
        (root / "tests").mkdir()
        (root / "tests" / "test_fake.py").write_text(test_body)
        shutil.copy2(SCRIPT, root / "scripts" / SCRIPT.name)
        for shared in ("cli", "core", "hosts"):
            (root / shared).symlink_to(REPO / shared)

        return root / "scripts" / SCRIPT.name

    def test_a_passing_suite_reports_ok(self):
        script = self.fake_root(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_ok(self):\n        self.assertTrue(True)\n"
        )
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


class TestApplyAgainstAFakeHome(CutoverCase):
    """`--apply` is NEVER run against the real `~/.hermes` — not by a test and not by a
    probe. Every case here writes only inside the temp tree built in setUp."""

    def link(self):
        return self.hermes / "plugins" / "memories"

    def test_it_installs_the_symlink_and_selects_the_provider(self):
        out = self.run_script("--apply")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertTrue(self.link().is_symlink())
        self.assertEqual(Path(os.readlink(self.link())), REPO / "hosts" / "hermes")
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
        `memory_enabled` starts with the same six letters as the block being edited."""
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

    def test_failed_checks_gate_the_apply_and_write_nothing(self):
        out = self.run_script("--apply", env=self.env(SERVER_API_KEY=None))
        self.assertEqual(out.returncode, 1, out.stdout)
        self.assertLine(out, "nothing was changed")
        self.assertEqual(self.config_yaml.read_text(), CONFIG_YAML)
        self.assertFalse(self.link().exists())
        self.assertEqual(list(self.hermes.glob("config.yaml.bak-*")), [])

    def test_it_prints_how_to_go_back(self):
        out = self.run_script("--apply")
        self.assertLine(out, "config.yaml.bak-")
        self.assertLine(out, "hermes_memory")


if __name__ == "__main__":
    unittest.main()
