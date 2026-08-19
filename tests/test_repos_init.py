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
        env = {**os.environ, "QCTX_STATE_DIR": box}
        out = subprocess.run([sys.executable, str(REPO / "hooks" / "lease.py")],
                             input=json.dumps({"session_id": "s-1"}), capture_output=True,
                             text=True, env=env, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        os.environ["QCTX_STATE_DIR"] = box
        live = lease.live()
        self.assertEqual(len(live), 1, f"no lease was written: {out.stderr[-200:]}")
        self.assertEqual(live[0]["session_id"], "s-1")

    def test_the_hook_never_fails_the_session_even_with_a_broken_payload(self):
        """A hook that fails SessionStart is a hook the user uninstalls."""
        out = subprocess.run([sys.executable, str(REPO / "hooks" / "lease.py")],
                             input="{not json", capture_output=True, text=True,
                             env={**os.environ, "QCTX_STATE_DIR": tempfile.mkdtemp()},
                             timeout=120)
        self.assertEqual(out.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
