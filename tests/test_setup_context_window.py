"""`qctx setup` says where each host's context window comes from.

The big-file guard resolves a session's window in one order (`core/windows.py`): what the
host reported for that session, what an endpoint reported, `context_window` from the
config, otherwise nothing, which allows every read. Since v1.3 there is no table of model
names behind it, so a machine where no host reports and nothing is declared has a guard
that is installed and inert. This check is where a person sees which case they are in.

claude-code reports its window to one external process only, the statusLine command, so
on claude-code the question is whether `qctx statusline` is installed. hermes reports
through the provider on every session, so there the question is what it last said.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core import config, hostwindow, setup, statusline  # noqa: E402


def cfg_with(**over):
    values = dict(config.DEFAULTS)
    values.update(over)
    return config.Config(**values)


class StateDirCase(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.mkdtemp()
        patcher = mock.patch.dict(os.environ, {"QCTX_STATE_DIR": self.state})
        patcher.start()
        self.addCleanup(patcher.stop)

    def check(self, statusline_state=None, **over):
        return setup._check_context_window(cfg_with(**over), statusline_state)


class TheCheckSaysWhereTheWindowComesFrom(StateDirCase):
    def test_an_installed_status_line_is_ok_and_shows_what_claude_code_last_reported(self):
        hostwindow.publish("s1", "claude-opus-5-5[1m]", 1_000_000, "claude-code")
        check = self.check("installed")
        self.assertTrue(check.ok, check.detail)
        self.assertIn("statusLine installed", check.detail)
        self.assertIn("claude-opus-5-5[1m] 1M", check.detail)
        self.assertRegex(check.detail, r"just now|\d+ (min|h|d) ago")

    def test_an_installed_status_line_that_has_not_reported_yet_is_ok(self):
        check = self.check("installed")
        self.assertTrue(check.ok, check.detail)
        self.assertIn("none yet", check.detail)

    def test_without_the_status_line_a_declared_window_is_ok_and_named_as_the_config(self):
        check = self.check("missing", context_window=200_000)
        self.assertTrue(check.ok, check.detail)
        self.assertIn("200000 declared", check.detail)
        self.assertIn("only when no host reports one", check.detail)

    def test_without_the_status_line_or_a_declaration_it_warns_with_both_fixes(self):
        check = self.check("missing", context_window=0)
        self.assertFalse(check.ok)
        self.assertTrue(check.warning, "a guard that allows every read is a warning, not a blocker")
        self.assertIn("claude-code reports no window", check.detail)
        self.assertIn("statusline install --apply", check.fix_hint)
        self.assertIn("config set context-window", check.fix_hint)

    def test_a_stale_status_line_warns_and_names_the_repair(self):
        check = self.check("stale", context_window=0)
        self.assertFalse(check.ok)
        self.assertTrue(check.warning)
        self.assertIn("no longer exists", check.detail)
        self.assertIn("statusline install --apply", check.fix_hint)

    def test_another_status_line_is_named_and_counts_as_none(self):
        check = self.check("foreign", context_window=0)
        self.assertFalse(check.ok)
        self.assertIn("another status line", check.detail)
        # `install --apply` never replaces somebody else's status line: not the repair.
        self.assertNotIn("install --apply", check.fix_hint)
        self.assertIn("qctx statusline", check.fix_hint)

    def test_an_unreadable_settings_file_is_where_the_repair_starts(self):
        check = self.check("unreadable", context_window=0)
        self.assertFalse(check.ok)
        self.assertIn("repair ~/.claude/settings.json", check.fix_hint)

    def test_a_hermes_guess_warns_even_with_the_status_line_installed(self):
        """claude-code being covered says nothing about hermes: its last report was a guess
        and nothing is declared, so the hermes guard allows every read in that session."""
        hostwindow.publish("s1", "claude-opus-5-5[1m]", 1_000_000, "claude-code")
        hostwindow.publish("h1", "Qwen3.8-27B", 131_072, "hermes", guess=True)
        check = self.check("installed", context_window=0)
        self.assertFalse(check.ok, check.detail)
        self.assertTrue(check.warning)
        self.assertIn("on hermes", check.detail)
        self.assertIn("config set context-window", check.fix_hint)

    def test_a_hermes_guess_with_the_window_declared_is_ok(self):
        hostwindow.publish("h1", "Qwen3.8-27B", 131_072, "hermes", guess=True)
        self.assertTrue(self.check("installed", context_window=200_000).ok)

    def test_hermes_last_report_is_shown_and_counts(self):
        hostwindow.publish("h1", "Qwen3.8-27B", 524_288, "hermes")
        check = self.check(None, context_window=0)
        self.assertTrue(check.ok, check.detail)
        self.assertIn("hermes: last report Qwen3.8-27B 524k", check.detail)

    def test_both_hosts_are_shown_together(self):
        hostwindow.publish("s1", "claude-opus-5-5[1m]", 1_000_000, "claude-code")
        hostwindow.publish("h1", "claude-haiku-4-5-20251001", 200_000, "hermes")
        check = self.check("installed")
        self.assertIn("claude-code:", check.detail)
        self.assertIn("hermes: last report claude-haiku-4-5-20251001 200k", check.detail)

    def test_a_guess_is_shown_as_one_and_does_not_count(self):
        hostwindow.publish("h1", "mystery-model", 256_000, "hermes", guess=True)
        check = self.check(None, context_window=0)
        self.assertFalse(check.ok, check.detail)
        self.assertTrue(check.warning)
        self.assertIn("a guess the guard ignores", check.detail)

    def test_no_host_and_no_declaration_warns(self):
        check = self.check(None, context_window=0)
        self.assertFalse(check.ok)
        self.assertTrue(check.warning)
        self.assertIn("config set context-window", check.fix_hint)
        self.assertIn("allows every read", check.detail)

    def test_a_negative_declaration_is_not_a_declaration(self):
        """`core/windows.py` gates on `declared > 0`; the check must agree with it."""
        check = self.check(None, context_window=-1)
        self.assertFalse(check.ok)
        self.assertTrue(check.warning)


class DiagnoseRunsIt(StateDirCase):
    def test_diagnose_includes_it(self):
        """Reachability is irrelevant here: the check must be present even with Qdrant
        down, which is what an offline suite gives us."""
        with mock.patch.object(statusline, "state", return_value=None):
            names = [c["name"] for c in setup.diagnose(cfg_with())["checks"]]
        self.assertIn("Context window", names)

    def test_diagnose_hands_it_the_status_line_state(self):
        with mock.patch.object(statusline, "state", return_value="missing"):
            checks = {c["name"]: c for c in setup.diagnose(cfg_with(context_window=0))["checks"]}
        self.assertIn("statusline install --apply", checks["Context window"]["fix_hint"])


class TheStatusLineState(unittest.TestCase):
    """`core.statusline.state()`: what setup is told about claude-code's settings, read only."""

    def home(self, settings_text=None, claude_dir=True) -> Path:
        home = Path(tempfile.mkdtemp())
        if claude_dir:
            (home / ".claude").mkdir()
        if settings_text is not None:
            (home / ".claude" / "settings.json").write_text(settings_text)
        patcher = mock.patch.dict(os.environ, {"HOME": str(home)})
        patcher.start()
        self.addCleanup(patcher.stop)

        return home

    def test_no_claude_directory_means_claude_code_is_not_set_up_here(self):
        self.home(claude_dir=False)
        self.assertIsNone(statusline.state())

    def test_a_claude_directory_without_settings_is_missing(self):
        self.home()
        self.assertEqual(statusline.state(), "missing")

    def test_settings_without_a_status_line_is_missing_and_unchanged(self):
        home = self.home('{"model": "opus[1m]"}\n')
        self.assertEqual(statusline.state(), "missing")
        self.assertEqual((home / ".claude" / "settings.json").read_text(), '{"model": "opus[1m]"}\n')

    def test_ours_is_installed(self):
        home = self.home()
        launcher = home / "bin" / "qctx"
        launcher.parent.mkdir()
        launcher.write_text("#!/bin/sh\n")
        launcher.chmod(0o755)
        (home / ".claude" / "settings.json").write_text(json.dumps(
            {"statusLine": {"type": "command", "command": f"{launcher} statusline"}}))
        self.assertEqual(statusline.state(), "installed")

    def test_ours_pointing_at_a_launcher_that_is_gone_is_stale(self):
        self.home('{"statusLine": {"type": "command", "command": "/gone/bin/qctx statusline"}}')
        self.assertEqual(statusline.state(), "stale")

    def test_the_settings_path_is_the_one_the_install_command_uses(self):
        home = self.home()
        self.assertEqual(statusline.settings_path(), home / ".claude" / "settings.json")


if __name__ == "__main__":
    unittest.main()
