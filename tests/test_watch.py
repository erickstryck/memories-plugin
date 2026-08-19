"""Keeping the index current without anyone asking.

The daemon walks the indexed repos and compares `mtime` against what is in the archive. Measured
on 2026-08-18: 16 ms for 2,000 files — cheap enough to make `inotify` unnecessary, which would
otherwise mean an external dependency or platform-specific code.

THE DEBOUNCE IS WHAT SEPARATES THIS FROM REINDEXING ON EVERY KEYSTROKE: a file only enters the
queue once it has been stable for one cycle.

A SECOND SOURCE OF CHANGE, besides a file already in the archive being edited: a file added to
`git` after the initial index, which `changed_paths` structurally cannot see (it only walks
paths the archive ALREADY has). `FakeIndex.checkouts` and `indexed_paths()` exist so the tests
below can exercise that path (`_new_tracked_paths` in `core/indexer.py`) without a real git
repository or a real Qdrant collection — see `TestWatchingPicksUpNewlyTrackedFiles`.
"""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import indexer, jobs  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


class FakeIndex:
    """An index that can say what changed, with no Qdrant behind it."""

    def __init__(self, changed=(), checkouts=("/tmp/alpha",), indexed=()):
        self._changed = list(changed)
        self._checkouts = list(checkouts)
        self._indexed = set(indexed)
        self.refreshed = []
        self.indexed_calls = []

    def list_repos(self):
        return [{"repo": "alpha", "checkouts": self._checkouts}]

    def changed_paths(self, repo):
        return list(self._changed)

    def indexed_paths(self, repo):
        self.indexed_calls.append(repo)

        return set(self._indexed)

    def refresh(self, repo):
        self.refreshed.append(repo)

        return []


def a_git_repo() -> str:
    root = tempfile.mkdtemp()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=60)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)

    return root


def track(root: str, name: str, text: str = "x = 1\n") -> str:
    path = os.path.join(root, name)
    with open(path, "w") as fh:
        fh.write(text)
    subprocess.run(["git", "-C", root, "add", name], check=True, timeout=60)

    return path


class TestWatchingPicksUpNewlyTrackedFiles(unittest.TestCase):
    """Whole-branch review finding 8: `changed_paths` only ever walks paths the archive
    ALREADY has, so a file added to git after the initial index (`git add newfile.py`, no
    commit needed) was invisible to the watcher forever, silently. `_new_tracked_paths`
    reuses `scan.eligible` — the same selection `add-all` runs — against each checkout the
    repo is registered under, and diffs it against `indexed_paths()`."""

    def setUp(self):
        a_state_dir()

    def test_a_newly_tracked_file_is_enqueued_as_an_index_job_on_the_second_sighting(self):
        root = a_git_repo()
        new_path = track(root, "brand_new.py")
        ix = FakeIndex(changed=[], checkouts=[root], indexed=set())
        watch = indexer.watcher(index=ix)
        watch()
        self.assertIsNone(jobs.load("alpha"), "enqueued on the first sighting, no debounce")
        watch()
        job = jobs.load("alpha")
        self.assertEqual(job["kind"], "index",
                         "a never-indexed file went through `refresh` instead of `add_files`")
        self.assertIn(os.path.abspath(new_path), job["paths"])

    def test_a_file_ALREADY_indexed_is_not_treated_as_new(self):
        root = a_git_repo()
        existing = track(root, "already.py")
        ix = FakeIndex(changed=[], checkouts=[root], indexed={os.path.abspath(existing)})
        watch = indexer.watcher(index=ix)
        watch()
        watch()
        self.assertIsNone(jobs.load("alpha"), "an already-indexed file was re-queued as new")

    def test_nothing_tracked_means_no_job(self):
        root = a_git_repo()
        ix = FakeIndex(changed=[], checkouts=[root], indexed=set())
        watch = indexer.watcher(index=ix)
        watch()
        watch()
        self.assertIsNone(jobs.load("alpha"))

    def test_a_bad_checkout_root_does_not_blind_the_watcher_to_the_rest(self):
        """One repo's checkout raising (moved, deleted, unreadable) must not stop OTHER new
        files in OTHER checkouts of the same repo from being found. `scan.eligible` itself
        already swallows the ordinary failures (a missing directory just yields zero tracked
        files), so the guard is exercised directly here by making it raise, rather than
        relying on a filesystem state that happens to reach the same branch."""
        from unittest import mock

        good_root = a_git_repo()
        new_path = track(good_root, "found_me.py")
        real_eligible = indexer.scan.eligible

        def flaky(root, *a, **kw):
            if root == "poisoned":
                raise RuntimeError("permission denied, or whatever else scan could not catch")

            return real_eligible(root, *a, **kw)

        ix = FakeIndex(changed=[], checkouts=["poisoned", good_root], indexed=set())
        watch = indexer.watcher(index=ix)
        with mock.patch("core.indexer.scan.eligible", side_effect=flaky):
            watch()
            watch()
        job = jobs.load("alpha")
        self.assertIsNotNone(job, "a checkout that raised blinded the watcher to a working one")
        self.assertIn(os.path.abspath(new_path), job["paths"])


class TestWatching(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_a_stable_change_becomes_a_refresh_job_on_the_SECOND_sighting(self):
        """First sighting: noted. Second sighting, still changed: enqueued."""
        ix = FakeIndex(changed=["/tmp/alpha/a.py"])
        watch = indexer.watcher(index=ix)
        watch()
        self.assertIsNone(jobs.load("alpha"), "enqueued on the first sighting, with no debounce")
        watch()
        self.assertEqual(jobs.load("alpha")["kind"], "refresh")

    def test_nothing_changed_means_no_job_at_all(self):
        watch = indexer.watcher(index=FakeIndex(changed=[]))
        watch()
        watch()
        self.assertEqual(jobs.all_jobs(), [])

    def test_it_does_not_queue_a_second_job_while_one_is_running(self):
        """Without this, every cycle would stack a refresh on top of the previous one."""
        ix = FakeIndex(changed=["/tmp/alpha/a.py"])
        watch = indexer.watcher(index=ix)
        watch(); watch()
        jobs.update("alpha", state=jobs.RUNNING)
        watch(); watch()
        self.assertEqual(jobs.load("alpha")["state"], jobs.RUNNING)


if __name__ == "__main__":
    unittest.main(verbosity=2)
