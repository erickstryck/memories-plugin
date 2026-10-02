"""The statusLine command: how claude-code's own context window reaches the big-file guard.

claude-code computes `context_window.context_window_size` for the model selected now and
hands it to ONE external process: the statusLine command (measured on 2.1.282; the hook
payloads carry no model and no window). It runs when the REPL mounts, before any prompt,
after every assistant message, and right after a `/model`. So `qctx statusline` publishes
what it receives (`core.hostwindow`) and prints a short line; the guard reads the record.

The payloads in `tests/fixtures/statusline-2.1.282.json` are real ones, captured from an
interactive session; only paths and ids were replaced.
"""
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core import hostwindow, statusline  # noqa: E402

FIXTURE = json.loads((REPO / "tests" / "fixtures" / "statusline-2.1.282.json").read_text())
AT_START = FIXTURE["session_start"]
AFTER_SWITCH = FIXTURE["after_model_switch"]
CLI = REPO / "cli" / "qctx.py"


class StateDirCase(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.mkdtemp()
        patcher = mock.patch.dict(os.environ, {"QCTX_STATE_DIR": self.state})
        patcher.start()
        self.addCleanup(patcher.stop)


class TestTheStatusLinePublishesWhatClaudeCodeReported(StateDirCase):
    def run_main(self, text: str) -> tuple[int, str]:
        out = io.StringIO()
        code = statusline.main(io.StringIO(text), out)

        return code, out.getvalue()

    def test_the_real_payload_at_session_start_is_published(self):
        code, _ = self.run_main(json.dumps(AT_START))
        self.assertEqual(code, 0)
        got = hostwindow.read(AT_START["session_id"])
        self.assertEqual((got.model, got.window, got.source, got.guess),
                         ("claude-opus-5-5[1m]", 1_000_000, "claude-code", False))

    def test_a_model_switch_replaces_the_window(self):
        self.run_main(json.dumps(AT_START))
        self.run_main(json.dumps(AFTER_SWITCH))
        got = hostwindow.read(AFTER_SWITCH["session_id"])
        self.assertEqual((got.model, got.window), ("claude-haiku-4-5-20251001", 200_000))

    def test_it_prints_one_short_line(self):
        _, printed = self.run_main(json.dumps(AT_START))
        self.assertEqual(printed, "ctx · 1M\n")

    def test_the_used_share_is_shown_when_claude_code_knows_it(self):
        payload = json.loads(json.dumps(AFTER_SWITCH))
        payload["context_window"]["used_percentage"] = 23.4
        _, printed = self.run_main(json.dumps(payload))
        self.assertEqual(printed, "ctx 23% · 200k\n")

    def test_a_payload_without_a_window_publishes_nothing_and_still_prints(self):
        payload = {k: v for k, v in AT_START.items() if k != "context_window"}
        code, printed = self.run_main(json.dumps(payload))
        self.assertEqual((code, printed), (0, "ctx\n"))
        self.assertIsNone(hostwindow.read(AT_START["session_id"]))

    def test_garbage_on_stdin_still_prints_and_exits_zero(self):
        for text in ("", "{not json", "[1, 2]", json.dumps({"context_window": "big"})):
            with self.subTest(text=text):
                self.assertEqual(self.run_main(text), (0, "ctx\n"))
        self.assertEqual(list(Path(self.state).glob(hostwindow.PATTERN)), [])

    def test_an_unwritable_state_dir_costs_the_record_not_the_line(self):
        blocked = Path(self.state) / "in-the-way"
        blocked.write_text("a file, not a directory")
        with mock.patch.dict(os.environ, {"QCTX_STATE_DIR": str(blocked)}):
            self.assertEqual(self.run_main(json.dumps(AT_START)), (0, "ctx · 1M\n"))


class TestTheWindowIsShownForHumans(unittest.TestCase):
    def test_the_sizes_claude_code_reported_read_naturally(self):
        cases = {1_000_000: "1M", 200_000: "200k", 1_500_000: "1.5M", 131_072: "131k",
                 1_048_575: "1M", 8_192: "8k"}
        for tokens, text in cases.items():
            with self.subTest(tokens=tokens):
                self.assertEqual(statusline.human(tokens), text)


class TestTheRealCommand(unittest.TestCase):
    """`qctx statusline` as claude-code runs it: a process, payload on stdin."""

    def run_cli(self, payload, **env):
        state = tempfile.mkdtemp()
        full = dict(os.environ, QCTX_STATE_DIR=state, **env)
        started = time.monotonic()
        done = subprocess.run([sys.executable, str(CLI), "statusline"], input=payload,
                              capture_output=True, text=True, env=full, timeout=30)

        return done, state, time.monotonic() - started

    def test_it_publishes_and_prints(self):
        done, state, _ = self.run_cli(json.dumps(AT_START))
        self.assertEqual((done.returncode, done.stdout), (0, "ctx · 1M\n"), done.stderr)
        with mock.patch.dict(os.environ, {"QCTX_STATE_DIR": state}):
            self.assertEqual(hostwindow.read(AT_START["session_id"]).window, 1_000_000)

    def test_a_corrupt_config_does_not_cost_the_status_line(self):
        """It is dispatched before the config is loaded: a statusLine that printed a
        traceback would sit in the user's screen on every message."""
        bad = Path(tempfile.mkdtemp()) / "config.json"
        bad.write_text("{not json")
        done, state, _ = self.run_cli(json.dumps(AT_START), QCTX_CONFIG=str(bad))
        self.assertEqual((done.returncode, done.stdout), (0, "ctx · 1M\n"), done.stderr)
        self.assertEqual(done.stderr, "")
        with mock.patch.dict(os.environ, {"QCTX_STATE_DIR": state}):
            self.assertEqual(hostwindow.read(AT_START["session_id"]).window, 1_000_000)

    def test_it_is_fast_enough_to_run_after_every_message(self):
        timings = [self.run_cli(json.dumps(AT_START))[2] for _ in range(3)]
        self.assertLess(min(timings), 0.5, timings)


class TestTheGuardReadsWhatTheStatusLineWrote(unittest.TestCase):
    """The real writer and the real reader, as two processes, agreeing on one file."""

    def decision(self, payload: dict) -> str:
        sys.path.insert(0, str(REPO / "tests"))
        from tests.test_bigfile_claude import a_file_of, a_transcript, a_usage, hook_env, run_hook
        env = hook_env(QCTX_CONTEXT_WINDOW="0")
        statusline_env = dict(os.environ, QCTX_STATE_DIR=env["QCTX_STATE_DIR"])
        payload = dict(payload, session_id="s1")
        subprocess.run([sys.executable, str(CLI), "statusline"], input=json.dumps(payload),
                       capture_output=True, text=True, env=statusline_env, timeout=30, check=True)
        _, out, code = run_hook(a_file_of(4 * 40_000), a_transcript([a_usage(150_000)]), env)
        self.assertEqual(code, 0)

        return "deny" if out.strip() else "allow"

    def test_a_200k_session_refuses_what_a_1m_session_allows(self):
        self.assertEqual(self.decision(AFTER_SWITCH), "deny")
        self.assertEqual(self.decision(AT_START), "allow")


class TestInstallingTheStatusLine(unittest.TestCase):
    COMMAND = "/opt/x/bin/qctx statusline"

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.settings = self.dir / "settings.json"

    def write(self, data) -> bytes:
        text = json.dumps(data, indent=2) + "\n"
        self.settings.write_text(text)
        os.chmod(self.settings, 0o644)

        return self.settings.read_bytes()

    def test_missing_in_a_dry_run_changes_nothing(self):
        before = self.write({"model": "opus[1m]", "hooks": {}})
        self.assertEqual(statusline.install(self.settings, self.COMMAND, apply=False),
                         ("missing", self.COMMAND))
        self.assertEqual(self.settings.read_bytes(), before)

    def test_apply_adds_only_the_status_line(self):
        self.write({"model": "opus[1m]", "env": {"A": "1"}, "hooks": {"X": []}})
        self.assertEqual(statusline.install(self.settings, self.COMMAND, apply=True),
                         ("added", self.COMMAND))
        data = json.loads(self.settings.read_text())
        self.assertEqual(data.pop("statusLine"), {"type": "command", "command": self.COMMAND})
        self.assertEqual(data, {"model": "opus[1m]", "env": {"A": "1"}, "hooks": {"X": []}})
        self.assertEqual(stat.S_IMODE(os.stat(self.settings).st_mode), 0o644)
        self.assertEqual([p.name for p in self.dir.iterdir()], ["settings.json"],
                         "a temporary was left behind")

    def test_ours_already_there_is_installed(self):
        before = self.write({"statusLine": {"type": "command",
                                            "command": "/elsewhere/bin/qctx statusline"}})
        self.assertEqual(statusline.install(self.settings, self.COMMAND, apply=True),
                         ("installed", "/elsewhere/bin/qctx statusline"))
        self.assertEqual(self.settings.read_bytes(), before)

    def test_someone_elses_status_line_is_left_alone(self):
        before = self.write({"statusLine": {"type": "command", "command": "~/my-line.sh"}})
        self.assertEqual(statusline.install(self.settings, self.COMMAND, apply=True),
                         ("foreign", "~/my-line.sh"))
        self.assertEqual(self.settings.read_bytes(), before)

    def test_an_unreadable_settings_file_is_left_alone(self):
        self.settings.write_text("{not json")
        state, _ = statusline.install(self.settings, self.COMMAND, apply=True)
        self.assertEqual(state, "unreadable")
        self.assertEqual(self.settings.read_text(), "{not json")

    def test_no_settings_file_yet_is_created_private(self):
        self.assertEqual(statusline.install(self.settings, self.COMMAND, apply=True),
                         ("added", self.COMMAND))
        self.assertEqual(json.loads(self.settings.read_text()),
                         {"statusLine": {"type": "command", "command": self.COMMAND}})
        self.assertEqual(stat.S_IMODE(os.stat(self.settings).st_mode), 0o600)

    def test_what_counts_as_ours(self):
        for command, ours in (("/x/qctx statusline", True), ("qctx statusline", True),
                              ("/x/qctx statusline  ", True), ("/x/qctx stats", False),
                              ("/x/my-statusline", False), (None, False), (42, False)):
            with self.subTest(command=command):
                self.assertEqual(statusline.is_ours(command), ours)


class TestInstallingThroughTheCLI(unittest.TestCase):
    def test_install_does_not_need_a_readable_config(self):
        """The cutover calls it on machines whose config may be half written."""
        settings = Path(tempfile.mkdtemp()) / "settings.json"
        settings.write_text("{}\n")
        bad = Path(tempfile.mkdtemp()) / "config.json"
        bad.write_text("{not json")
        done = subprocess.run([sys.executable, str(CLI), "statusline", "install",
                               "--settings", str(settings)], capture_output=True, text=True,
                              timeout=30, env=dict(os.environ, QCTX_CONFIG=str(bad)))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("would add", done.stdout)

    def test_install_reports_and_apply_writes(self):
        settings = Path(tempfile.mkdtemp()) / "settings.json"
        settings.write_text("{}\n")

        def qctx(*args):
            return subprocess.run([sys.executable, str(CLI), "statusline", "install",
                                   "--settings", str(settings), *args],
                                  capture_output=True, text=True, timeout=30,
                                  env=dict(os.environ, QCTX_STATE_DIR=tempfile.mkdtemp()))

        dry = qctx()
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertIn("would add", dry.stdout)
        self.assertEqual(settings.read_text(), "{}\n")
        done = qctx("--apply")
        self.assertEqual(done.returncode, 0, done.stderr)
        command = json.loads(settings.read_text())["statusLine"]["command"]
        self.assertTrue(command.endswith("qctx statusline"), command)
        self.assertTrue(os.path.isabs(command.split()[0]), command)


if __name__ == "__main__":
    unittest.main(verbosity=2)
