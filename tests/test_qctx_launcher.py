"""The launcher's resolution order.

`bin/qctx` is COPIED to `~/.local/bin/qctx` by the wizard, so it cannot rely on sitting
inside the tree it runs. It resolves the live install on every call instead — which is what
makes it survive `claude plugin update`, where the install path is a fresh directory named
after the new commit.
"""
import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parent.parent
LAUNCHER = REPO / "bin" / "qctx"


def fake_tree(root: Path) -> Path:
    """The minimum a directory needs to count as an install."""
    (root / "cli").mkdir(parents=True)
    (root / "cli" / "qctx.py").write_text("raise SystemExit(0)\n")
    (root / "bin").mkdir()
    return root


def run_root(launcher: Path, env: dict) -> str:
    done = subprocess.run([str(launcher), "--root"], capture_output=True, text=True,
                          env=env, timeout=30)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


class LauncherResolution(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.env = {"HOME": str(self.home), "PATH": os.environ["PATH"]}
        self.addCleanup(self.tmp.cleanup)

    def test_qctx_home_wins_over_everything(self):
        chosen = fake_tree(Path(self.tmp.name) / "chosen")
        fake_tree(self.home / ".hermes" / "plugins" / "memories")
        self.env["QCTX_HOME"] = str(chosen)
        self.assertEqual(run_root(LAUNCHER, self.env), str(chosen))

    def test_own_tree_when_no_override(self):
        # Invoked from the repository itself: the tree it lives in is the answer.
        self.assertEqual(run_root(LAUNCHER, self.env), str(REPO))

    def test_copy_outside_a_tree_finds_the_hermes_install(self):
        tree = fake_tree(self.home / ".hermes" / "plugins" / "memories")
        copy = self.home / ".local" / "bin" / "qctx"
        copy.parent.mkdir(parents=True)
        copy.write_bytes(LAUNCHER.read_bytes())
        copy.chmod(0o755)
        self.assertEqual(run_root(copy, self.env), str(tree))

    def test_copy_falls_through_to_the_claude_install_path(self):
        tree = fake_tree(Path(self.tmp.name) / "cache" / "b8008f7dac88")
        registry = self.home / ".claude" / "plugins" / "installed_plugins.json"
        registry.parent.mkdir(parents=True)
        registry.write_text(json.dumps({"version": 2, "plugins": {
            "memories-plugin@memories-plugin": [{"installPath": str(tree)}]}}))
        copy = self.home / ".local" / "bin" / "qctx"
        copy.parent.mkdir(parents=True)
        copy.write_bytes(LAUNCHER.read_bytes())
        copy.chmod(0o755)
        self.assertEqual(run_root(copy, self.env), str(tree))

    def test_nothing_to_resolve_fails_loudly(self):
        copy = self.home / ".local" / "bin" / "qctx"
        copy.parent.mkdir(parents=True)
        copy.write_bytes(LAUNCHER.read_bytes())
        copy.chmod(0o755)
        done = subprocess.run([str(copy), "--root"], capture_output=True, text=True,
                              env=self.env, timeout=30)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("could not find", done.stderr)


if __name__ == "__main__":
    unittest.main()
