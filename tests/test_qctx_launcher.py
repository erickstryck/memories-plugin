"""The launcher's resolution order.

`bin/qctx` is COPIED to `~/.local/bin/qctx` by the wizard, so it cannot rely on sitting
inside the tree it runs. It resolves the live install on every call instead — which is what
makes it survive `claude plugin update`, where the install path is a fresh directory named
after the new commit.
"""
import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parent.parent
LAUNCHER = REPO / "bin" / "qctx"


def run_with_own_tree_deleted(work: Path, env: dict, args=("--root",)):
    """Run a copy of the launcher via `bash /proc/self/fd/N` after deleting the
    directory it lives in — the race `own_tree()`'s header comment describes: the tree
    vanishing after the script was opened, exactly what `claude plugin update` does
    when it removes the old SHA directory.

    All fd entries under /proc are symlinks, so `own_tree()`'s `[ -L "$target" ]` loop
    follows `/proc/self/fd/N`, `readlink` yields the now-deleted path, and the final
    `cd` inside `own_tree()` fails. Measured: `readlink` gives
    `…/doomed/bin/qctx (deleted)`, and `cd -P …/doomed/bin/..` then returns non-zero.

    WHAT THIS DOES NOT EXERCISE, said plainly because the docstring used to claim it
    did: the `|| candidate=""` in `resolve_root()`. Removing that guard leaves every
    test in this file green, because `resolve_root` is only ever called as
    `$(resolve_root)` and errexit does not fire on a failing assignment inside a
    command-substitution subshell. What IS pinned below is the behaviour a user sees —
    an empty candidate is not mistaken for a resolved tree, resolution carries on to
    the next one, and `own_tree`'s raw `cd:` error never reaches stderr.
    """
    doomed = work / "doomed" / "bin"
    doomed.mkdir(parents=True)
    script = doomed / "qctx"
    script.write_bytes(LAUNCHER.read_bytes())
    script.chmod(0o755)

    fd = os.open(script, os.O_RDONLY)
    try:
        shutil.rmtree(work / "doomed")
        return subprocess.run(["bash", f"/proc/self/fd/{fd}", *args],
                              capture_output=True, text=True, env=env,
                              pass_fds=(fd,), timeout=30)
    finally:
        os.close(fd)


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

    def test_own_tree_failure_falls_through_to_hermes(self):
        # own_tree() genuinely fails here (see run_with_own_tree_deleted): the script's
        # directory is gone by the time it runs. What is pinned is the OUTCOME — the
        # unusable candidate is skipped and the hermes install answers — and not the
        # `|| candidate=""` guard, which this cannot reach; the helper's docstring says
        # why.
        tree = fake_tree(self.home / ".hermes" / "plugins" / "memories")
        done = run_with_own_tree_deleted(Path(self.tmp.name), self.env)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), str(tree))

    def test_own_tree_failure_with_no_candidates_fails_loudly_not_raw(self):
        # Same broken own_tree(), but nothing else to fall back to. The user must see
        # the controlled "could not find" message, never own_tree()'s raw `cd:` error —
        # which is what the `2>/dev/null` beside that `cd` is for, and this is the test
        # that keeps it there.
        done = run_with_own_tree_deleted(Path(self.tmp.name), self.env)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("could not find", done.stderr)
        self.assertNotIn("cd:", done.stderr)

    def test_lazy_evaluation_skips_claude_tree_when_qctx_home_resolves(self):
        # Verify that claude_tree() is not called when QCTX_HOME resolves.
        # This test uses a python3 shim on PATH that writes to a witness file
        # when called with "-" as the first argument (the signature of claude_tree's call).
        # If the witness file exists after qctx --root, it means claude_tree() was called
        # unnecessarily, and the test fails.

        # Create a witness file path
        witness_file = Path(self.tmp.name) / "python3_called_during_resolution"

        # Create a shim python3 that detects resolution-time calls
        shim_dir = Path(self.tmp.name) / "shim"
        shim_dir.mkdir()
        shim_python = shim_dir / "python3"
        shim_python.write_text(f"""#!/bin/bash
# This shim detects if python3 is called during resolution (with "-" as first arg).
# claude_tree() calls: python3 - "$registry" <<'PY'...
# This shim writes to a witness file if that pattern is detected, allowing us to
# verify that lazy evaluation is working.
if [ "${{1:-}}" = "-" ]; then
  touch {witness_file}
fi
exec /usr/bin/python3 "$@"
""")
        shim_python.chmod(0o755)

        # Prepend shim to PATH so it shadows /usr/bin/python3
        self.env["PATH"] = f"{shim_dir}:{self.env['PATH']}"

        # Create the chosen tree (will be resolved via QCTX_HOME)
        chosen = fake_tree(Path(self.tmp.name) / "chosen")

        # Create a claude registry that would resolve but should NOT be consulted
        # when QCTX_HOME is set
        registry = self.home / ".claude" / "plugins" / "installed_plugins.json"
        registry.parent.mkdir(parents=True)
        registry.write_text(json.dumps({"version": 2, "plugins": {
            "memories-plugin@memories-plugin": [{"installPath": str(Path(self.tmp.name) / "unused_registry_tree")}]}}))

        # Set QCTX_HOME to the chosen tree
        self.env["QCTX_HOME"] = str(chosen)

        # Create a copy of the launcher outside the tree (to test resolution)
        copy = self.home / ".local" / "bin" / "qctx"
        copy.parent.mkdir(parents=True)
        copy.write_bytes(LAUNCHER.read_bytes())
        copy.chmod(0o755)

        # Run qctx --root with our shim on PATH
        result = run_root(copy, self.env)

        # Verify the chosen tree is returned
        self.assertEqual(result, str(chosen))

        # Verify python3 was NOT called during resolution (witness file not created)
        self.assertFalse(
            witness_file.exists(),
            "Python3 was called with '-' during resolution, meaning claude_tree() was evaluated "
            "even though QCTX_HOME resolved. This indicates eager evaluation — the resolver should "
            "short-circuit and not call claude_tree() when earlier candidates succeed."
        )


if __name__ == "__main__":
    unittest.main()
