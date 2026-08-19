"""What goes into the archive when someone asks "index this project".

The source is `git ls-files`, not a disk scan. This makes `.gitignore` respected BY DEFINITION
instead of by our own reimplementation, and `node_modules`, build and cache stay out without
special rules — they are not versioned.
"""
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import scan  # noqa: E402


def a_repo(**files) -> str:
    root = tempfile.mkdtemp()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=60)
    for name, content in files.items():
        path = os.path.join(root, name.replace("__", "/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = "wb" if isinstance(content, bytes) else "w"
        with open(path, mode) as fh:
            fh.write(content)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, timeout=60)

    return root


class TestTheSourceIsGIT(unittest.TestCase):
    def test_only_tracked_files_are_eligible(self):
        root = a_repo(**{"kept.py": "x = 1\n"})
        with open(os.path.join(root, "untracked.py"), "w") as fh:
            fh.write("y = 2\n")
        out = scan.eligible(root)
        names = [os.path.basename(p) for p in out["eligible"]]
        self.assertEqual(names, ["kept.py"])

    def test_a_gitignored_file_is_absent_without_us_parsing_gitignore(self):
        """The point of using `git ls-files`: we never reimplement the semantics of `.gitignore`,
        which has negation, precedence, and per-directory rules."""
        root = a_repo(**{".gitignore": "build/\n", "app.py": "x = 1\n"})
        os.makedirs(os.path.join(root, "build"), exist_ok=True)
        with open(os.path.join(root, "build", "out.js"), "w") as fh:
            fh.write("console.log(1)\n")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True, timeout=60)
        names = [os.path.basename(p) for p in scan.eligible(root)["eligible"]]
        self.assertIn("app.py", names)
        self.assertNotIn("out.js", names)

    def test_the_paths_are_ABSOLUTE(self):
        """`add_files` receives paths and the daemon runs with a different cwd — a relative
        path would resolve against the wrong directory, silently."""
        root = a_repo(**{"a.py": "x = 1\n"})
        for path in scan.eligible(root)["eligible"]:
            self.assertTrue(os.path.isabs(path), path)

    def test_a_directory_that_is_not_a_repo_yields_nothing_and_does_not_raise(self):
        out = scan.eligible(tempfile.mkdtemp())
        self.assertEqual(out["eligible"], [])
        self.assertEqual(out["tracked"], 0)


class TestTheFourDiscards(unittest.TestCase):
    """Each one isolated, because a test with two discard reasons proves neither."""

    def test_a_binary_file_is_skipped(self):
        root = a_repo(**{"logo.png": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"})
        out = scan.eligible(root)
        self.assertEqual(out["eligible"], [])
        self.assertEqual(out["skipped"]["binary"], 1)

    def test_a_lockfile_is_skipped(self):
        root = a_repo(**{"package-lock.json": '{"lockfileVersion": 3}\n'})
        out = scan.eligible(root)
        self.assertEqual(out["eligible"], [])
        self.assertEqual(out["skipped"]["lockfile"], 1)

    def test_a_minified_file_is_skipped(self):
        """A 40 KB line answers no questions and dominates the repo's archive."""
        root = a_repo(**{"bundle.js": "var a=1;" * 6000 + "\n"})
        out = scan.eligible(root)
        self.assertEqual(out["eligible"], [])
        self.assertEqual(out["skipped"]["minified"], 1)

    def test_a_minified_file_whose_FIRST_line_is_short_is_still_skipped(self):
        """The most common minified shape: a license comment, then the entire bundle on the
        next line. Judging only lines that ended before the read completed threw the long line
        away and let the bundle through — a guard that hid the case it was meant to catch."""
        root = a_repo(**{"vendor.js": "/*! lib v1.0 */\n" + "!function(e){}(x);" * 900 + "\n"})
        out = scan.eligible(root)
        self.assertEqual(out["eligible"], [])
        self.assertEqual(out["skipped"]["minified"], 1)

    def test_a_file_above_the_ceiling_is_skipped(self):
        root = a_repo(**{"huge.txt": "line\n" * 300})
        out = scan.eligible(root, max_bytes=100)
        self.assertEqual(out["eligible"], [])
        self.assertEqual(out["skipped"]["too_big"], 1)

    def test_an_ordinary_source_file_survives_ALL_FOUR(self):
        """The guard of guards: if a filter gets too broad, this fails."""
        root = a_repo(**{"core__app.py": "def main():\n    return 1\n"})
        out = scan.eligible(root)
        self.assertEqual(len(out["eligible"]), 1)
        self.assertEqual(sum(out["skipped"].values()), 0)


class TestTheFunnelIsREPORTED(unittest.TestCase):
    def test_it_counts_what_was_seen_and_what_was_dropped(self):
        """A count that appears only at the end is a count nobody uses to decide."""
        root = a_repo(**{"a.py": "x = 1\n", "package-lock.json": "{}\n",
                         "logo.png": b"\x00\x01\x02binary"})
        out = scan.eligible(root)
        self.assertEqual(out["tracked"], 3)
        self.assertEqual(len(out["eligible"]), 1)
        self.assertEqual(out["skipped"]["lockfile"], 1)
        self.assertEqual(out["skipped"]["binary"], 1)

    def test_every_discard_reason_is_present_even_when_zero(self):
        """A missing key forces every consumer to use `.get`, and one will forget."""
        out = scan.eligible(a_repo(**{"a.py": "x = 1\n"}))
        self.assertEqual(set(out["skipped"]), {"binary", "minified", "lockfile", "too_big",
                                               "unreadable"})


class TestAGitFailureIsNotAnEmptyRepository(unittest.TestCase):
    """Whole-project review, finding 6. `tracked_files` returned `[]` for EVERY failure -- git
    missing, the 120 s timeout, a permission error -- and `add-all` then printed
    `0 tracked -> nothing to index` and exited 0. That is a failure dressed as a true negative,
    which this project refuses everywhere else. Only "not a git repository" may be empty."""

    def test_a_directory_that_is_not_a_repository_is_honestly_empty(self):
        self.assertEqual(scan.tracked_files(tempfile.mkdtemp()), [],
                         "a plain directory should report no tracked files, not raise")

    def test_git_missing_from_PATH_raises_instead_of_reporting_nothing(self):
        with unittest.mock.patch.object(scan.subprocess, "run",
                                        side_effect=FileNotFoundError("git")):
            with self.assertRaises(scan.CoreError) as caught:
                scan.tracked_files("/anywhere")
        self.assertIn("git", str(caught.exception).lower(),
                      "the message does not say what went wrong")

    def test_a_timeout_raises_instead_of_reporting_nothing(self):
        with unittest.mock.patch.object(
                scan.subprocess, "run",
                side_effect=subprocess.TimeoutExpired(cmd="git", timeout=120)):
            with self.assertRaises(scan.CoreError):
                scan.tracked_files("/anywhere")

    def test_an_unexplained_nonzero_exit_raises_rather_than_reading_as_empty(self):
        """The dangerous middle case: git RAN and failed for a reason that is not
        "not a repository" -- a corrupt index, a locked object store. Empty would be a lie."""
        class Failed:
            returncode = 128
            stdout = b""
            stderr = b"fatal: unable to read index file\n"
        with unittest.mock.patch.object(scan.subprocess, "run", return_value=Failed()):
            with self.assertRaises(scan.CoreError) as caught:
                scan.tracked_files("/anywhere")
        self.assertIn("index file", str(caught.exception),
                      "git's own reason never reached the caller")


if __name__ == "__main__":
    unittest.main(verbosity=2)
