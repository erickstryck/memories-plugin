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
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import indexer, jobs  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


class FakeIndex:
    """An index that can say what changed, with no Qdrant behind it."""

    # THE DEFAULT CHECKOUT MUST NOT NAME A PATH ANYTHING COULD CREATE. `watch` now shells out
    # to `git ls-files` under every checkout, so a real repository sitting at this path would
    # make `new_paths` non-empty and silently flip the enqueued job from `refresh` to `index`,
    # failing tests that have nothing to do with checkouts. `/tmp/alpha` was one `mkdir` away
    # from doing exactly that; a path under a directory that cannot exist is not.
    def __init__(self, changed=(), checkouts=("/nonexistent/alpha",), indexed=()):
        self._changed = list(changed)
        self._checkouts = list(checkouts)
        self._indexed = set(indexed)
        self.refreshed = []
        self.indexed_calls = []
        # `added` records EVERY embed, with repeats, so a test can tell "indexed once" from
        # "indexed again every few cycles" -- which `_indexed` alone (a set) cannot show.
        self.added = []

    def list_repos(self):
        return [{"repo": "alpha", "checkouts": self._checkouts}]

    def changed_paths(self, repo):
        return list(self._changed)

    def indexed_paths(self, repo):
        self.indexed_calls.append(repo)

        return set(self._indexed)

    def poll(self, repo):
        """Mirrors `RepoIndex.poll`: both halves of the watch question from ONE archive read.

        Counts through `indexed_calls` exactly once per call, which is what lets the tests here
        assert how many archive reads a cycle costs.
        """
        return {"changed": self.changed_paths(repo), "indexed": self.indexed_paths(repo)}

    def add_files(self, repo, paths, **kwargs):
        self.added.extend(paths)
        self._indexed.update(paths)

        return {"files": len(paths), "chunks": len(paths)}

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
        ix = FakeIndex(changed=["/nonexistent/alpha/a.py"])
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
        ix = FakeIndex(changed=["/nonexistent/alpha/a.py"])
        watch = indexer.watcher(index=ix)
        watch(); watch()
        jobs.update("alpha", state=jobs.RUNNING)
        watch(); watch()
        self.assertEqual(jobs.load("alpha")["state"], jobs.RUNNING)



class TestTheScanIsNotRepaidEveryCycle(unittest.TestCase):
    """Re-review of the fix wave, finding (d): `_new_tracked_paths` closed a real hole — a
    newly `git add`ed file was invisible — but paid for it on EVERY cycle, forever. It is not
    a cheap call: `scan.eligible` shells out to `git ls-files` and then stats AND opens the
    first 8 KB of every tracked file to sniff binaries and minified bundles, and each call also
    pulled a second full `indexed_paths()` scroll out of the archive. Measured by the reviewer:
    ~5 ms for 114 files against 0.28 ms of stat, extrapolating to ~70-90 ms and up to 16 MB of
    reads per cycle per repository — against the spec's stated 16 ms watch budget.

    The answer is memoised against `mtime` of `.git/index`, because that file is written by
    every operation that can change WHICH files are tracked and by nothing else the watcher
    cares about. These tests hold the memo to both halves: it must actually skip the work, and
    it must never skip it when the index moved.
    """

    def setUp(self):
        a_state_dir()

    def test_a_second_cycle_over_an_untouched_git_index_repeats_neither_the_scan_nor_the_scroll(self):
        root = a_git_repo()
        track(root, "a.py")
        ix = FakeIndex(changed=[], checkouts=[root], indexed=set())
        calls = []
        real = indexer.scan.eligible

        def counting(r, *a, **kw):
            calls.append(r)

            return real(r, *a, **kw)

        with unittest.mock.patch.object(indexer.scan, "eligible", counting):
            watch = indexer.watcher(index=ix)
            watch()
            watch()
        self.assertEqual(len(calls), 1,
                         "the tracked-file scan ran again although `.git/index` never moved")
        self.assertEqual(len(ix.indexed_calls), 2,
                         "a watch cycle read the archive more than once -- `poll` exists so "
                         "both halves of the question share ONE fetch")

    def test_a_cycle_reads_the_archive_exactly_once(self):
        """The archive read cannot be memoised -- indexing changes it, and a memo that outlived
        a job made the watcher re-queue the same files forever (measured: one re-embed every 3
        cycles, see TestTheWatcherDoesNotReindexForever). So it is paid every cycle, and the
        thing to hold is that it is paid ONCE: `changed_paths` and the newly-tracked-file diff
        used to fetch the same source metadata separately."""
        root = a_git_repo()
        track(root, "a.py")
        ix = FakeIndex(changed=[], checkouts=[root], indexed=set())
        watch = indexer.watcher(index=ix)
        watch()
        self.assertEqual(len(ix.indexed_calls), 1,
                         f"one cycle cost {len(ix.indexed_calls)} archive reads, not 1")

    def test_tracking_a_NEW_file_moves_the_index_and_forces_a_rescan(self):
        """The half that matters for correctness: the memo must never answer "nothing new" for
        an index that actually moved, or a file added while the daemon runs stays unindexed
        forever — in silence, which is the failure mode this project refuses."""
        root = a_git_repo()
        track(root, "a.py")
        ix = FakeIndex(changed=[], checkouts=[root], indexed={os.path.abspath(
            os.path.join(root, "a.py"))})
        watch = indexer.watcher(index=ix)
        watch()
        watch()
        self.assertIsNone(jobs.load("alpha"), "nothing new yet, so nothing should be queued")
        fresh = track(root, "brand_new.py")
        watch()
        watch()
        job = jobs.load("alpha")
        self.assertIsNotNone(job, "a file tracked after the memo was warm was never noticed")
        self.assertIn(os.path.abspath(fresh), job["paths"])

    def test_a_WORKTREE_is_memoised_too_and_not_rescanned_every_cycle(self):
        """Whole-project review, finding 5. In a worktree `.git` is a FILE holding a pointer and
        the real index sits under `<main>/.git/worktrees/<name>/index`, so the assumed path
        `<root>/.git/index` never existed: the stamp was permanently None, the memo never hit,
        and the watcher re-ran `git ls-files` plus an 8 KB sniff of every tracked file on every
        cycle -- forever, and only for users on the more advanced setup. Worktrees are
        supported (`core/bindings.py`), so this has to be memoised like any other checkout."""
        main = a_git_repo()
        track(main, "a.py")
        subprocess.run(["git", "-C", main, "commit", "-qm", "first"], check=True, timeout=60)
        wt = os.path.join(tempfile.mkdtemp(), "wt")
        subprocess.run(["git", "-C", main, "worktree", "add", "-q", wt],
                       check=True, timeout=60)
        self.assertTrue(os.path.isfile(os.path.join(wt, ".git")),
                        "this test proves nothing unless `.git` really is a file here")
        self.assertIsNotNone(indexer._index_stamp(wt),
                             "the worktree's index was not found, so the memo can never hit")
        ix = FakeIndex(changed=[], checkouts=[wt], indexed=set())
        calls = []
        real = indexer.scan.eligible
        with unittest.mock.patch.object(indexer.scan, "eligible",
                                        lambda r, *a, **kw: calls.append(r) or real(r)):
            watch = indexer.watcher(index=ix)
            watch()
            watch()
        self.assertEqual(len(calls), 1,
                         "a worktree paid for the full tracked-file scan on every cycle")

    def test_an_unreadable_git_index_recomputes_rather_than_assuming_unchanged(self):
        """A worktree or submodule keeps `.git` as a FILE, so there is no `.git/index` to
        stamp. Guessing "unchanged" there would silently freeze the watcher for that checkout,
        so the unknown stamp must fall back to doing the work."""
        root = tempfile.mkdtemp()                      # no `.git` at all
        ix = FakeIndex(changed=[], checkouts=[root], indexed=set())
        calls = []
        real = indexer.scan.eligible

        with unittest.mock.patch.object(indexer.scan, "eligible",
                                        lambda r, *a, **kw: calls.append(r) or real(r)):
            watch = indexer.watcher(index=ix)
            watch()
            watch()
        self.assertEqual(len(calls), 2,
                         "an unstampable checkout was memoised as if it were known unchanged")

class TestTheWatcherDoesNotReindexForever(unittest.TestCase):
    """Whole-project review, Critical 1. The memo that stops the tracked-file scan from being
    repaid every cycle is keyed on the git index -- which indexing does NOT touch, because a
    job changes the ARCHIVE. So a completed job left the memo still listing the files it had
    just embedded, the watcher queued them again, and the loop never closed: MEASURED at one
    re-embed every 3 cycles, forever, on a repository with a single file.

    WHY THE EARLIER TESTS MISSED IT, and why this class drives the loop itself. `daemon.run`
    runs a pending job INSTEAD of watching (`if job is not None: _run_one(...) elif watch is
    not None: watch()`), so a job is never in flight when `watch()` runs. The invalidation was
    written in the "job in flight" branch and was therefore unreachable in production -- but
    perfectly reachable in a test whose fake index never executes a job, which is exactly what
    the other tests here do. A watcher test that never runs a job is testing a daemon that
    does not exist.
    """

    def setUp(self):
        a_state_dir()

    def _cycle(self, n, ix):
        """The daemon's own precedence: a pending job wins, watching only fills the gaps."""
        from core import daemon
        work = indexer.work(index=ix)
        watch = indexer.watcher(index=ix)
        for _ in range(n):
            job = jobs.next_pending()
            if job is not None:
                daemon._run_one(job, work)
            else:
                watch()

    def test_a_file_indexed_once_is_not_embedded_again_cycle_after_cycle(self):
        root = a_git_repo()
        track(root, "a.py")
        ix = FakeIndex(changed=[], checkouts=[root], indexed=set())
        self._cycle(20, ix)
        self.assertEqual(len(ix.added), 1,
                         f"the same file was embedded {len(ix.added)} times in 20 cycles -- "
                         f"the watcher is paying for embeddings forever")

    def test_a_file_tracked_AFTER_the_first_job_is_still_picked_up(self):
        """The other half: invalidating on job state must not blind the watcher to real work."""
        root = a_git_repo()
        track(root, "a.py")
        ix = FakeIndex(changed=[], checkouts=[root], indexed=set())
        self._cycle(20, ix)
        second = track(root, "b.py")
        self._cycle(20, ix)
        self.assertIn(os.path.abspath(second), ix.added,
                      "a file tracked after the first job was never indexed")
        self.assertEqual(len(ix.added), 2, f"embedded more than once each: {ix.added}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
