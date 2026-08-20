"""`qctx install` — the wizard.

`--check` is the mode everything else is measured against: it must report the same picture
and write NOTHING. A wizard that repairs while you are looking cannot be used to find out
what state a machine is in.
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
CLI = REPO / "cli" / "qctx.py"

from tests.isolation import assert_hermetic, hermetic_env  # noqa: E402


class CheckMode(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.config = Path(self.tmp.name) / "config.json"
        self.config.write_text("{}")
        self.addCleanup(self.tmp.cleanup)

    def env(self):
        return hermetic_env(self.home, QCTX_CONFIG=self.config)

    def run_cli(self, *argv, **kwargs):
        return subprocess.run([sys.executable, str(CLI), "install", *argv],
                              capture_output=True, text=True, env=self.env(),
                              timeout=180, stdin=subprocess.DEVNULL, **kwargs)

    def test_the_environment_it_runs_in_is_assembled_not_inherited(self):
        """This module used to start from `dict(os.environ)`. The legacy aliases
        (`QDRANT_URL`, `SERVER_BASE_URL`, …) survived that scrubbing, so five of these
        check-only tests dialled the developer's real Qdrant and waited out its
        timeouts."""
        assert_hermetic(self, self.env(), allowed=("QCTX_CONFIG",))

    def test_check_writes_nothing(self):
        before = self.config.read_text()
        self.run_cli("--check")
        self.assertEqual(self.config.read_text(), before)
        self.assertFalse((self.home / ".local").exists())

    def test_check_reports_the_plumbing_section(self):
        done = self.run_cli("--check")
        self.assertIn("launcher", done.stdout)
        self.assertIn("no-shell config", done.stdout)

    def test_json_is_parseable_and_carries_both_shapes(self):
        done = self.run_cli("--check", "--json")
        payload = json.loads(done.stdout)
        self.assertIn("checks", payload)
        self.assertIn("hosts", payload)
        for section in payload["hosts"]:
            self.assertEqual({"host", "exit_code", "text"}, set(section))

    def test_absent_host_is_skipped_not_failed(self):
        """A machine with only one of the two hosts is the normal case."""
        done = self.run_cli("--check")
        self.assertNotIn("Traceback", done.stderr)
        self.assertEqual(done.returncode, 0)

    def test_no_tty_never_blocks(self):
        """stdin closed, no --yes: it diagnoses and exits, like `qctx setup`."""
        done = self.run_cli()
        self.assertEqual(done.returncode, 0)
        self.assertIn("no interactive terminal", done.stdout)


class SkipSuiteVariable(unittest.TestCase):
    def test_cutover_refuses_to_apply_with_the_suite_skipped(self):
        """Copied from the hermes script, including the refusal — the flag exists to make
        the PLAN cheap, never to make an apply cheap.

        HOME is pointed at a scratch directory, not the real one: `--apply` rewrites
        ~/.claude/settings.json, .mcp.json and .claude.json, and MOVES ~/.claude/skills.
        The refusal below fires before any of that (it is unconditional on `--apply` with
        the suite skipped), so a real HOME was never touched even without this — but a
        test that depends on that ordering never changing is not a test worth trusting the
        developer's machine to.
        """
        with TemporaryDirectory() as home:
            env = hermetic_env(home, CUTOVER_SKIP_SUITE="1")
            done = subprocess.run(["bash", str(REPO / "scripts" / "cutover.sh"), "--apply"],
                                  capture_output=True, text=True, env=env, timeout=120)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("suite unverified", done.stdout + done.stderr)


class WritingPass(unittest.TestCase):
    """The interactive pass, driven through stdin with --yes off."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.config = Path(self.tmp.name) / "config.json"
        self.config.write_text("{}")
        self.addCleanup(self.tmp.cleanup)

    def env(self):
        return hermetic_env(self.home, QCTX_CONFIG=self.config,
                            QCTX_INSTALL_FORCE_TTY="1")

    def run_with_input(self, keystrokes: str, *argv):
        return subprocess.run([sys.executable, str(CLI), "install", "--config-only",
                               *argv],
                              input=keystrokes, capture_output=True, text=True,
                              env=self.env(), timeout=180)

    def test_the_environment_it_runs_in_is_assembled_not_inherited(self):
        """With `HERMES_HOME` exported, the inherited form rewrote the developer's live
        `~/.hermes/.env` with the `qkey`/`skey` this class types."""
        assert_hermetic(self, self.env(),
                        allowed=("QCTX_CONFIG", "QCTX_INSTALL_FORCE_TTY"))

    def test_writes_the_urls_and_keeps_the_defaults_on_enter(self):
        answers = "\n".join([
            "https://q.example",        # qdrant_url
            "https://e.example/v1",     # api_base_url
            "qkey", "skey",             # the two keys
            "mem",                      # memory_collection
        ] + [""] * 9) + "\n"            # pass 2: Enter all the way
        self.run_with_input(answers)
        written = json.loads(self.config.read_text())
        self.assertEqual(written["qdrant_url"], "https://q.example")
        self.assertEqual(written["memory_collection"], "mem")

    def test_no_key_reaches_the_config_file_in_any_spelling(self):
        answers = "\n".join([
            "https://q.example", "https://e.example/v1", "qkey", "skey", "mem",
        ] + [""] * 9) + "\n"
        self.run_with_input(answers)
        body = self.config.read_text()
        for forbidden in ("qkey", "skey", "qdrant_api_key", "api_key"):
            self.assertNotIn(forbidden, body)

    def test_an_empty_hermes_home_does_not_drop_a_key_in_the_current_directory(self):
        """`Path("")` is `Path(".")`, and `Path(".").is_dir()` is True, so the wizard
        wrote a plaintext `.env` wherever it happened to be run — very likely a git
        repository. The sibling `hermes_install_path` already used the `or` idiom."""
        answers = "\n".join([
            "https://q.example", "https://e.example/v1", "qkey", "skey", "mem",
        ] + [""] * 9) + "\n"
        here = Path(self.tmp.name) / "somebodys-repo"
        here.mkdir()
        subprocess.run([sys.executable, str(CLI), "install", "--config-only"],
                       input=answers, capture_output=True, text=True, timeout=180,
                       cwd=here, env=hermetic_env(self.home, QCTX_CONFIG=self.config,
                                                  QCTX_INSTALL_FORCE_TTY="1",
                                                  HERMES_HOME=""))
        self.assertFalse((here / ".env").exists(),
                         "a credential file was written into the working directory")

    def test_the_key_value_is_never_echoed(self):
        answers = "\n".join([
            "https://q.example", "https://e.example/v1", "qkey", "s3cr3t-value", "mem",
        ] + [""] * 9) + "\n"
        done = self.run_with_input(answers)
        self.assertNotIn("s3cr3t-value", done.stdout + done.stderr)


class YesWithNoTerminal(unittest.TestCase):
    """`--yes` exists FOR the machine with no terminal, and that is where it crashed.

    The guard was `not _interactive and not args.yes`, so `--yes` with a closed stdin
    fell straight into the prompts and the first `input()` raised an uncaught EOFError.
    On the full path the launcher had already been copied into ~/.local/bin by then, so
    the run died half-done.
    """

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.config = Path(self.tmp.name) / "config.json"
        self.config.write_text('{"qdrant_url": "https://kept.example"}')
        self.addCleanup(self.tmp.cleanup)

    def run_cli(self, *argv):
        return subprocess.run([sys.executable, str(CLI), "install", *argv],
                              capture_output=True, text=True, timeout=180,
                              env=hermetic_env(self.home, QCTX_CONFIG=self.config),
                              stdin=subprocess.DEVNULL)

    def test_config_only_yes_with_no_stdin_exits_clean(self):
        done = self.run_cli("--config-only", "--yes")
        self.assertNotIn("Traceback", done.stderr)
        self.assertNotIn("EOFError", done.stderr)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_the_full_path_with_yes_and_no_stdin_exits_clean(self):
        """The worse case: `install_launcher` has already written before the crash."""
        done = self.run_cli("--yes")
        self.assertNotIn("Traceback", done.stderr)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_it_keeps_every_current_value(self):
        """Nothing was answered, so nothing may change. A `--yes` that wrote defaults
        over a working configuration would be worse than the crash."""
        before = self.config.read_text()
        self.run_cli("--config-only", "--yes")
        self.assertEqual(self.config.read_text(), before)


def load_cli():
    """Imports cli/qctx.py as a module. It is a script, not a package member.

    Copied verbatim from `tests/test_cli_render.py` — `import cli.qctx` does NOT work:
    there is no `cli/__init__.py`, deliberately, because the CLI is an entry point and
    not something the core imports.
    """
    import importlib.util
    path = REPO / "cli" / "qctx.py"
    spec = importlib.util.spec_from_file_location("qctx_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


class HostDetection(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.addCleanup(self.tmp.cleanup)
        self.qctx = load_cli()

    def test_claude_absent_reads_as_none(self):
        self.assertIsNone(self.qctx.claude_install_path(self.home))

    def test_claude_present_returns_the_live_path(self):
        registry = self.home / ".claude" / "plugins" / "installed_plugins.json"
        registry.parent.mkdir(parents=True)
        registry.write_text(json.dumps({"plugins": {"memories-plugin@memories-plugin": [
            {"installPath": "/somewhere/b8008f7dac88"}]}}))
        self.assertEqual(self.qctx.claude_install_path(self.home),
                         "/somewhere/b8008f7dac88")

    def test_the_hermes_command_carries_force_and_the_provider_switch(self):
        joined = " ".join(self.qctx.HOST_INSTALL_COMMANDS["hermes"])
        self.assertIn("--force", joined)
        self.assertIn("memory.provider memories", joined)


class LauncherInstall(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.addCleanup(self.tmp.cleanup)
        self.qctx = load_cli()

    def test_copies_and_makes_it_executable(self):
        target = self.qctx.install_launcher(REPO, {"HOME": str(self.home)})
        self.assertTrue(os.access(target, os.X_OK))
        self.assertEqual(target.read_bytes(), (REPO / "bin" / "qctx").read_bytes())

    def bin_dir(self) -> Path:
        d = self.home / ".local" / "bin"
        d.mkdir(parents=True, exist_ok=True)

        return d

    def test_it_replaces_a_symlink_into_the_source_tree(self):
        """The DOCUMENTED development install is exactly this symlink, and copying onto
        it raised `shutil.SameFileError` — uncaught, so the wizard died on the one
        machine shape the README tells people to create."""
        link = self.bin_dir() / "qctx"
        link.symlink_to(REPO / "bin" / "qctx")
        target = self.qctx.install_launcher(REPO, {"HOME": str(self.home)})
        self.assertFalse(target.is_symlink(), "the symlink survived the install")
        self.assertEqual(target.read_bytes(), (REPO / "bin" / "qctx").read_bytes())

    def test_it_never_writes_through_a_symlink_into_another_checkout(self):
        """Worse than the crash: pointing at a DIFFERENT checkout, the copy went
        through the link and silently rewrote that checkout's `bin/qctx`."""
        other = Path(self.tmp.name) / "other-checkout" / "bin"
        other.mkdir(parents=True)
        stand_in = other / "qctx"
        stand_in.write_text("#!/usr/bin/env bash\necho the other checkout\n")
        (self.bin_dir() / "qctx").symlink_to(stand_in)
        self.qctx.install_launcher(REPO, {"HOME": str(self.home)})
        self.assertEqual(stand_in.read_text(),
                         "#!/usr/bin/env bash\necho the other checkout\n")


class ClosingBehaviour(unittest.TestCase):
    """Two things the spec asks for that are easy to leave out, and both are about NOT
    pretending: stop when the thing underneath is still broken, and say out loud what the
    wizard cannot do for you."""

    def setUp(self):
        self.qctx = load_cli()

    def test_blockers_stop_the_run_before_the_host_steps(self):
        """No Qdrant, no point installing into a host. It names what did not answer and
        stops — instead of asking fifteen questions that cannot work."""
        blocked = {"ready": False, "blockers": [{"name": "Qdrant", "detail": "no answer",
                                                 "ok": False, "fix_hint": None,
                                                 "warning": False}]}
        self.assertTrue(self.qctx.should_stop_before_hosts(blocked))
        self.assertFalse(self.qctx.should_stop_before_hosts({"ready": True,
                                                             "blockers": []}))

    def test_the_manual_steps_are_named(self):
        text = "\n".join(self.qctx.MANUAL_STEPS)
        self.assertIn("hermes hooks list", text)
        self.assertIn("restart", text.lower())


if __name__ == "__main__":
    unittest.main()
