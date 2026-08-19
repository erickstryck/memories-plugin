"""Detect the repo and OFFER — never index on its own.

`candidates_for` has existed in the core since sub-project A and no host ever called it: it
was dead code. This is the consumer that was missing, on both hosts.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

QCTX = REPO / "cli" / "qctx.py"


def a_repo() -> str:
    root = tempfile.mkdtemp()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=60)

    return root


def run_cli(*args, cwd=None):
    env = {**os.environ, "QCTX_STATE_DIR": tempfile.mkdtemp()}

    return subprocess.run([sys.executable, str(QCTX), *args], capture_output=True, text=True,
                          cwd=cwd, env=env, timeout=180)


class TestInitOffersAndDoesNotWrite(unittest.TestCase):
    def test_outside_a_repo_it_says_so_instead_of_guessing(self):
        out = run_cli("repos", "init", cwd=tempfile.mkdtemp())
        self.assertIn("not inside a git", (out.stdout + out.stderr).lower())

    def test_inside_a_fresh_repo_it_reports_what_it_WOULD_do(self):
        """Without a TTY it does not ask and does not write — the same rule `setup` already
        follows, because a prompt waiting for an answer that never comes hangs the call."""
        out = run_cli("repos", "init", "--json", cwd=a_repo())
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        payload = json.loads(out.stdout)
        self.assertIn("suggest", payload)
        self.assertFalse(payload.get("indexed"), "indexed without consent")


class TestTheLeaseHookWritesALease(unittest.TestCase):
    def test_the_hook_writes_a_lease_that_reads_back_alive(self):
        from core import lease
        box = tempfile.mkdtemp()
        # QCTX_DAEMON_AUTOSTART_DISABLED=1: the hook now also calls `daemon.start()`
        # (finding 4 of the whole-branch review), and the default `spawn` launches a REAL,
        # detached `qctx repos daemon run` — exactly what "no test may start a real daemon"
        # forbids. This is the documented escape hatch for it, not a workaround.
        env = {**os.environ, "QCTX_STATE_DIR": box, "QCTX_DAEMON_AUTOSTART_DISABLED": "1"}
        out = subprocess.run([sys.executable, str(REPO / "hooks" / "lease.py")],
                             input=json.dumps({"session_id": "s-1"}), capture_output=True,
                             text=True, env=env, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        # `lease.live()` reads `QCTX_STATE_DIR` from THIS process's environment, not the
        # subprocess's — so it has to be set here too, but ONLY for this test: an assignment
        # with no restore would leak `box` into every test file that runs afterward in the
        # same `unittest` process, each one then reading (or writing) a state directory it
        # never asked for instead of its own.
        previous = os.environ.get("QCTX_STATE_DIR")
        self.addCleanup(lambda: self._restore_state_dir(previous))
        os.environ["QCTX_STATE_DIR"] = box
        live = lease.live()
        self.assertEqual(len(live), 1, f"no lease was written: {out.stderr[-200:]}")
        self.assertEqual(live[0]["session_id"], "s-1")

    @staticmethod
    def _restore_state_dir(previous) -> None:
        if previous is None:
            os.environ.pop("QCTX_STATE_DIR", None)
        else:
            os.environ["QCTX_STATE_DIR"] = previous

    def test_the_hook_never_fails_the_session_even_with_a_broken_payload(self):
        """A hook that fails SessionStart is a hook the user uninstalls."""
        out = subprocess.run([sys.executable, str(REPO / "hooks" / "lease.py")],
                             input="{not json", capture_output=True, text=True,
                             env={**os.environ, "QCTX_STATE_DIR": tempfile.mkdtemp(),
                                 "QCTX_DAEMON_AUTOSTART_DISABLED": "1"},
                             timeout=120)
        self.assertEqual(out.returncode, 0)

class TestTheHookStartsTheDaemon(unittest.TestCase):
    """IN-PROCESS, unlike the class above: proving `daemon.start()` is actually CALLED (and
    gated by the escape hatch) must not go through a real subprocess, because the default
    `spawn` launches a REAL detached `qctx repos daemon run` — exactly what "no test may start
    a real daemon" forbids. `core.daemon.start` is patched instead; the hook module is
    imported the same way tests/test_recall_block.py imports `recall`, by putting `hooks/` on
    `sys.path` and importing the bare module name (it has no `__init__.py`)."""

    def setUp(self):
        self._previous_state_dir = os.environ.get("QCTX_STATE_DIR")
        self._previous_autostart = os.environ.get("QCTX_DAEMON_AUTOSTART_DISABLED")
        os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()
        os.environ.pop("QCTX_DAEMON_AUTOSTART_DISABLED", None)
        hooks_dir = str(REPO / "hooks")
        if hooks_dir not in sys.path:
            sys.path.insert(0, hooks_dir)
        import importlib

        import lease as lease_hook

        importlib.reload(lease_hook)
        self.lease_hook = lease_hook

    def tearDown(self):
        self._restore("QCTX_STATE_DIR", self._previous_state_dir)
        self._restore("QCTX_DAEMON_AUTOSTART_DISABLED", self._previous_autostart)

    @staticmethod
    def _restore(key, previous) -> None:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous

    def _run_hook(self, session_id="s-inproc"):
        import io
        from unittest import mock

        payload = json.dumps({"session_id": session_id})
        with mock.patch("sys.stdin", io.StringIO(payload)), \
             mock.patch("core.daemon.start") as started:
            self.lease_hook.main()

        return started

    def test_it_calls_daemon_start_by_default(self):
        started = self._run_hook()
        started.assert_called_once()

    def test_QCTX_DAEMON_AUTOSTART_DISABLED_skips_the_call(self):
        os.environ["QCTX_DAEMON_AUTOSTART_DISABLED"] = "1"
        started = self._run_hook()
        started.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
