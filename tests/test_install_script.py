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
sys.path.insert(0, str(REPO))
SCRIPT = REPO / "scripts" / "install.sh"
BASH = shutil.which("bash") or "bash"  # absolute: the test below empties PATH

from tests.isolation import assert_hermetic, hermetic_env  # noqa: E402


class Bootstrap(unittest.TestCase):
    """Every run here is given an ASSEMBLED environment.

    `test_it_runs_from_any_directory` used to pass none at all: real `HOME`, real `PATH`.
    Both hosts resolve on the developer's machine, so a check-only test shelled out to
    both cutovers against the live `~/.claude` and `~/.hermes`, and took over 100
    seconds. `hermetic_env` is what makes "offline" true rather than hoped for.
    """

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.config = Path(self.tmp.name) / "config.json"
        self.config.write_text("{}")
        self.addCleanup(self.tmp.cleanup)

    def env(self, **overrides):
        return hermetic_env(self.home, QCTX_CONFIG=self.config, **overrides)

    def test_it_is_executable(self):
        self.assertTrue(os.access(SCRIPT, os.X_OK))

    def test_the_environment_it_runs_in_is_assembled_not_inherited(self):
        assert_hermetic(self, self.env(), allowed=("QCTX_CONFIG",))

    def test_it_forwards_to_qctx_install(self):
        done = subprocess.run([BASH, str(SCRIPT), "--check", "--json"],
                              capture_output=True, text=True, env=self.env(),
                              timeout=300, stdin=subprocess.DEVNULL)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('"hosts"', done.stdout)

    def test_it_runs_from_any_directory(self):
        with TemporaryDirectory() as elsewhere:
            done = subprocess.run([BASH, str(SCRIPT), "--check"], cwd=elsewhere,
                                  capture_output=True, text=True, env=self.env(),
                                  timeout=300, stdin=subprocess.DEVNULL)
            self.assertEqual(done.returncode, 0, done.stderr)

    def test_it_says_what_is_missing_when_python3_is_absent(self):
        done = subprocess.run([BASH, str(SCRIPT)], capture_output=True, text=True,
                              env=self.env(PATH="/nonexistent"), timeout=60,
                              stdin=subprocess.DEVNULL)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("python3", done.stderr)


if __name__ == "__main__":
    unittest.main()