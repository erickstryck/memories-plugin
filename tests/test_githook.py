"""The hook writes into SOMEBODY ELSE'S repository, so what it must never do matters most.

Three guarantees the user asked for by name on 2026-08-18, when the question was "ele vai commitar
algo???": it does not commit, stage, push or touch the working tree; it can never fail a commit;
and it never writes over a `post-commit` that belongs to another tool.
"""
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import githook  # noqa: E402
from core.githook import HookError  # noqa: E402


def a_working_copy() -> str:
    root = tempfile.mkdtemp()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=60)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True, timeout=60)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True, timeout=60)

    return root


def hook_of(root: str) -> Path:
    return Path(root) / ".git" / "hooks" / "post-commit"


class TestWhatItInstalls(unittest.TestCase):
    def test_it_writes_an_EXECUTABLE_post_commit(self):
        """A hook without the execute bit is a file git ignores in silence."""
        root = a_working_copy()
        out = githook.install("alpha", root)
        self.assertEqual(out["action"], "installed")
        self.assertTrue(hook_of(root).is_file())
        self.assertTrue(os.stat(hook_of(root)).st_mode & stat.S_IXUSR, "not executable")

    def test_it_ends_in_exit_0_so_it_can_never_FAIL_a_commit(self):
        """The guarantee that keeps the hook installed. A hook that rejects a commit gets
        deleted the first time it is wrong, and this one is not worth anybody's commit."""
        root = a_working_copy()
        githook.install("alpha", root)
        self.assertTrue(hook_of(root).read_text().rstrip().endswith("exit 0"))

    def test_it_uses_an_ABSOLUTE_qctx_when_one_is_on_PATH(self):
        """A git hook does not inherit an interactive shell's PATH — it runs with whatever git
        was started with. A bare `qctx` resolves in a terminal and silently does not resolve from
        an editor's git integration, which is the failure nobody would connect to this."""
        if not githook.qctx_command().startswith("/"):
            self.skipTest("qctx is not on PATH here, so there is no absolute form to expect")
        root = a_working_copy()
        githook.install("alpha", root)
        self.assertIn(githook.qctx_command(), hook_of(root).read_text())

    def test_installing_twice_is_not_an_error(self):
        root = a_working_copy()
        githook.install("alpha", root)
        self.assertEqual(githook.install("alpha", root)["action"], "already")


class TestWhatItMustNEVERDo(unittest.TestCase):
    def test_the_hook_body_contains_no_command_that_writes_to_git(self):
        """Read as TEXT rather than executed, because the point is that these can never run —
        an execution-based test would only prove they did not run for one input."""
        root = a_working_copy()
        githook.install("alpha", root)
        body = hook_of(root).read_text()
        for forbidden in ("git commit", "git add", "git push", "git checkout", "git reset",
                          "git stash", "git rm"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, body)

    def test_it_REFUSES_a_post_commit_that_belongs_to_somebody_else(self):
        """husky, pre-commit, or the user's own. Writing over another tool's file to install a
        convenience is worse than not installing at all."""
        root = a_working_copy()
        hook_of(root).parent.mkdir(parents=True, exist_ok=True)
        hook_of(root).write_text("#!/bin/sh\n# husky\nnpx husky-run post-commit\n")
        out = githook.install("alpha", root)
        self.assertEqual(out["action"], "refused")
        self.assertIn("husky", hook_of(root).read_text(), "somebody else's hook was overwritten")
        self.assertIn("repos add alpha", out["line"], "it refused without saying what to add")

    def test_a_path_outside_a_working_copy_is_refused_with_a_reason(self):
        with self.assertRaises(HookError) as caught:
            githook.install("alpha", tempfile.mkdtemp())
        self.assertIn("not inside a git working copy", str(caught.exception))


class TestTheHookACTUALLYRunsOnACommit(unittest.TestCase):
    """The half no text inspection can give: git really fires it, and the commit really survives.

    `qctx` is replaced by a stub on PATH that records its arguments, so nothing is indexed and no
    network is touched — what is measured is that git ran the hook and what it passed."""

    def test_a_commit_fires_it_with_the_files_of_that_commit_and_SUCCEEDS(self):
        root = a_working_copy()
        bin_dir = tempfile.mkdtemp()
        record = os.path.join(bin_dir, "seen.txt")
        stub = os.path.join(bin_dir, "qctx")
        with open(stub, "w") as fh:
            fh.write(f'#!/bin/sh\necho "$@" >> "{record}"\n')
        os.chmod(stub, 0o755)

        githook.install("alpha", root)
        # Point the installed hook at the stub instead of the real qctx.
        hook = hook_of(root)
        hook.write_text(hook.read_text().replace(githook.qctx_command(), stub))

        (Path(root) / "a.py").write_text("x = 1\n")
        (Path(root) / "b.py").write_text("y = 2\n")
        subprocess.run(["git", "add", "."], cwd=root, check=True, timeout=60)
        done = subprocess.run(["git", "commit", "-m", "first"], cwd=root,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, f"the hook broke the commit:\n{done.stderr}")

        for _ in range(100):                      # it is detached; give it a moment to land
            if os.path.exists(record):
                break
            import time
            time.sleep(0.05)
        self.assertTrue(os.path.exists(record), "the hook never invoked qctx")
        seen = open(record).read()
        self.assertIn("repos add alpha", seen)
        self.assertIn("a.py", seen)
        self.assertIn("b.py", seen)

    def test_a_commit_SUCCEEDS_even_when_the_command_is_missing_entirely(self):
        """The failure that must not cost a commit: qctx uninstalled, moved, or broken."""
        root = a_working_copy()
        githook.install("alpha", root)
        hook = hook_of(root)
        hook.write_text(hook.read_text().replace(githook.qctx_command(),
                                                 "/nonexistent/qctx-does-not-exist"))
        (Path(root) / "a.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "."], cwd=root, check=True, timeout=60)
        done = subprocess.run(["git", "commit", "-m", "first"], cwd=root,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0,
                         f"a missing qctx failed the commit:\n{done.stderr}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
