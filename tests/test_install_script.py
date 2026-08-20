"""The bootstrap.

The only piece that runs before `qctx` exists anywhere, and therefore the only piece in
bash. It decides nothing: everything past the `exec` is Python the offline suite reaches.
"""
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "install.sh"
BASH = shutil.which("bash") or "bash"  # absolute: the test below empties PATH


class Bootstrap(unittest.TestCase):
    def test_it_is_executable(self):
        self.assertTrue(os.access(SCRIPT, os.X_OK))

    def test_it_forwards_to_qctx_install(self):
        env = dict(os.environ, PATH="/usr/bin:/bin")
        done = subprocess.run([BASH, str(SCRIPT), "--check", "--json"],
                              capture_output=True, text=True, env=env, timeout=300,
                              stdin=subprocess.DEVNULL)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('"hosts"', done.stdout)

    def test_it_runs_from_any_directory(self):
        with TemporaryDirectory() as elsewhere:
            done = subprocess.run([BASH, str(SCRIPT), "--check"], cwd=elsewhere,
                                  capture_output=True, text=True, timeout=300,
                                  stdin=subprocess.DEVNULL)
            self.assertEqual(done.returncode, 0, done.stderr)

    def test_it_says_what_is_missing_when_python3_is_absent(self):
        env = dict(os.environ, PATH="/nonexistent")
        done = subprocess.run([BASH, str(SCRIPT)], capture_output=True, text=True,
                              env=env, timeout=60, stdin=subprocess.DEVNULL)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("python3", done.stderr)


if __name__ == "__main__":
    unittest.main()