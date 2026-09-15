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

from core import indexer, jobs, quarantine  # noqa: E402
from core.embedding import EmbeddingError  # noqa: E402


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
    def __init__(self, changed=(), checkouts=("/nonexistent/alpha",), indexed=(), fails=()):
        self._changed = list(changed)
        self._checkouts = list(checkouts)
        self._indexed = set(indexed)
        # LISKOV: a fake that only knows how to SUCCEED is not a substitute for the real
        # index, and that gap is exactly what let the re-queue loop through review. `fails`
        # maps a path to the reason `add_files` reports for it, the way the real one does.
        self._fails = dict(fails)
        #: Set by a test to make `refresh` raise the way the real one does on an outage.
        self.refresh_raises: Exception | None = None
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
        skipped = [(p, self._fails[p]) for p in paths if p in self._fails]
        stored = [p for p in paths if p not in self._fails]
        self.added.extend(paths)
        self._indexed.update(stored)

        return {"repo": repo, "files": len(stored), "chunks": len(stored), "skipped": skipped}

    def refresh(self, repo, should_stop=None):
        self.refreshed.append(repo)
        # LISKOV: the real one raises on an infrastructure failure rather than reporting it
        # per file — an endpoint that is down is a fact about the minute, not about a file.
        # A fake that could only succeed or skip could not exercise the backoff at all.
        if self.refresh_raises is not None:
            raise self.refresh_raises
        report = []
        for path in self._changed:
            if path in self._fails:
                report.append({"path": path, "action": "skipped",
                               "reason": self._fails[path]})
                continue
            report.append({"path": path, "action": "reindexed", "chunks": 1})

        return report


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


def a_file_on_disk(text: str = "content\n") -> str:
    """A real file OUTSIDE any repository, for the `refresh` path: the quarantine records the
    mtime and size it reads from disk, so a path that cannot be `stat`ed is never recorded."""
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, "w") as fh:
        fh.write(text)

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
        # A REAL file on disk: the watcher drops a changed path it cannot stat, because a
        # deleted file reports as changed forever and queued a refresh every other cycle.
        ix = FakeIndex(changed=[a_file_on_disk("x = 1\n")])
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
        ix = FakeIndex(changed=[a_file_on_disk("x = 1\n")])
        watch = indexer.watcher(index=ix)
        watch()
        watch()
        jobs.update("alpha", state=jobs.RUNNING)
        watch()
        watch()
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


class TestAFileThatCannotBeIndexedIsNotRetriedForever(unittest.TestCase):
    """Measured on 2026-09-14: the same 22 files were re-queued every ~40 s indefinitely,
    because `work()` discarded the `skipped` report `add_files` already returns. The load that
    put on the shared embedding endpoint pushed automatic recall from 0.04 s to 1.98 s against
    its 2.00 s ceiling, and the user saw UNAVAILABLE blocks naming nothing about indexing."""

    def setUp(self):
        a_state_dir()

    def test_a_file_that_fails_to_index_is_not_enqueued_again(self):
        root = a_git_repo()
        doomed = os.path.abspath(track(root, "empty.json", text=""))
        ix = FakeIndex(changed=[], checkouts=[root], indexed=set(),
                       fails={doomed: "nothing indexable (empty file, or whitespace only)"})
        watch = indexer.watcher(index=ix)
        run_job = indexer.work(index=ix)

        watch(); watch()                       # debounce: enqueued on the second sighting
        job = jobs.load("alpha")
        self.assertIsNotNone(job, "the new file was never enqueued in the first place")
        run_job(job)                           # the daemon runs it; add_files reports a skip

        jobs.update("alpha", state=jobs.DONE)
        watch(); watch()
        self.assertEqual(jobs.load("alpha")["state"], jobs.DONE,
                         "a file that can never be indexed was queued all over again")

    def test_a_file_that_succeeds_is_not_quarantined(self):
        root = a_git_repo()
        track(root, "fine.py")
        ix = FakeIndex(changed=[], checkouts=[root], indexed=set())
        watch = indexer.watcher(index=ix)
        run_job = indexer.work(index=ix)
        watch(); watch()
        run_job(jobs.load("alpha"))
        self.assertEqual(quarantine.held("alpha"), set())

    def test_the_reason_is_kept_so_the_user_can_read_it(self):
        root = a_git_repo()
        doomed = os.path.abspath(track(root, "huge.json", text="x"))
        ix = FakeIndex(changed=[], checkouts=[root], indexed=set(),
                       fails={doomed: "HTTP 500: input (83086 tokens) is too large"})
        watch = indexer.watcher(index=ix)
        run_job = indexer.work(index=ix)
        watch(); watch()
        run_job(jobs.load("alpha"))
        self.assertIn("83086", quarantine.load("alpha")[doomed]["reason"])

    def test_a_refresh_that_skips_a_file_also_quarantines_it(self):
        """The `refresh` path needs this as much as `index`: a file that changed and fails to
        re-embed stays in `changed` forever, so it is queued on every cycle."""
        doomed = a_file_on_disk("broken\n")
        ix = FakeIndex(changed=[doomed], fails={doomed: "nothing indexable"})
        run_job = indexer.work(index=ix)
        run_job({"repo": "alpha", "kind": "refresh", "paths": []})
        self.assertIn("nothing indexable", quarantine.load("alpha")[doomed]["reason"])

    def test_a_refresh_does_NOT_quarantine_a_file_that_merely_vanished(self):
        """`missing` is a different state from "cannot be indexed": it costs no embedding, and
        `refresh` reports it on purpose for the user to see. Holding it would erase that."""
        gone = "/nonexistent/alpha/deleted.py"
        ix = FakeIndex(changed=[])
        ix.refresh = lambda repo, should_stop=None: [
            {"path": gone, "action": "missing", "reason": "gone"}]
        run_job = indexer.work(index=ix)
        run_job({"repo": "alpha", "kind": "refresh", "paths": []})
        self.assertEqual(quarantine.load("alpha"), {})


class TestQuarantineReleasesWhenTheContentChanges(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_a_quarantined_file_is_queued_again_once_it_changes(self):
        root = a_git_repo()
        path = os.path.abspath(track(root, "was_empty.py", text=""))
        ix = FakeIndex(changed=[], checkouts=[root], indexed=set(),
                       fails={path: "nothing indexable (empty file, or whitespace only)"})
        watch = indexer.watcher(index=ix)
        run_job = indexer.work(index=ix)
        watch(); watch()
        run_job(jobs.load("alpha"))
        jobs.update("alpha", state=jobs.DONE)

        with open(path, "w") as fh:            # the file gains content
            fh.write("def real_code():\n    return 1\n")
        ix._fails.clear()                      # and now it indexes fine

        watch(); watch()
        job = jobs.load("alpha")
        self.assertEqual(job["state"], jobs.PENDING,
                         "a file that gained content was never retried")
        self.assertIn(path, job["paths"])


class TestTheWatcherDoesNotREADEveryTrackedFile(unittest.TestCase):
    """Deciding whether a file is eligible means opening and reading it — which is why
    `core/scan.py` says that judgement belongs to the eligibility pass and never to the
    watcher's cycle. It landed on the cycle anyway: measured 207 ms for a 10,380-file checkout
    against the 16 ms this module budgets, re-paid on every `git add`, checkout or commit,
    because those invalidate the memo. Nearly all of it was re-judging files already indexed
    only to discard the verdict on the last line."""

    def setUp(self):
        a_state_dir()

    def test_a_file_the_ARCHIVE_LOST_is_found_again_even_with_a_warm_memo(self):
        """The recovery this narrowing must not cancel. `_new_tracked_paths` is the only path
        that re-queues a file the archive no longer holds — its own docstring calls it "the
        one case `changed_paths` structurally cannot cover", because `poll` reports a file
        with no chunks as neither indexed nor changed.

        The memo is keyed on the git index, which does not move when the ARCHIVE changes. So
        caching the already-narrowed set made a lost file invisible until someone happened to
        run `git add`: measured 0 index jobs in 10 cycles where the unmemoized path finds it
        at once."""
        root = tempfile.mkdtemp()
        subprocess.run(["git", "init", "-q", root], check=True)
        paths = []
        for name in ("a.py", "b.py"):
            path = os.path.join(root, name)
            with open(path, "w") as fh:
                fh.write("x = 1\n")
            paths.append(os.path.abspath(path))
        subprocess.run(["git", "-C", root, "add", "-A"], check=True)

        entry = {"repo": "alpha", "checkouts": [root]}
        memo: dict = {}
        self.assertEqual(indexer._new_tracked_paths(entry, set(paths), memo), set(),
                         "precondition: everything is indexed, so nothing is new")

        # The archive loses one file. Nothing touched git, so the memo is still warm.
        found = indexer._new_tracked_paths(entry, {paths[1]}, memo)
        self.assertEqual(found, {paths[0]},
                         "a file the archive lost is invisible until someone runs `git add`")

    def test_a_file_the_archive_already_holds_is_never_opened(self):
        root = tempfile.mkdtemp()
        subprocess.run(["git", "init", "-q", root], check=True)
        paths = []
        for name in ("a.py", "b.py", "c.py"):
            path = os.path.join(root, name)
            with open(path, "w") as fh:
                fh.write("x = 1\n")
            paths.append(os.path.abspath(path))
        subprocess.run(["git", "-C", root, "add", "-A"], check=True)

        sniffed = []
        real_sniff = indexer.scan._sniff
        try:
            indexer.scan._sniff = lambda p: sniffed.append(p) or real_sniff(p)
            # Two of the three are already indexed; only the third is a candidate.
            found = indexer._new_tracked_paths({"repo": "alpha", "checkouts": [root]},
                                               set(paths[:2]), {})
        finally:
            indexer.scan._sniff = real_sniff

        self.assertEqual(found, {paths[2]}, "the new file must still be found")
        self.assertEqual([os.path.abspath(p) for p in sniffed], [paths[2]],
                         f"it read files the archive already has: {sniffed}")


class TestAnOutageBacksOffInsteadOfHammering(unittest.TestCase):
    """An infrastructure error propagates by design — an endpoint that is down is a fact about
    the minute, not about the file, so nothing is quarantined and the job lands FAILED. But
    FAILED is neither PENDING nor RUNNING, so the watcher's guard does not skip the repo and
    the same work is queued again on the next cycle. Measured before the breaker: 10 hits on a
    down endpoint in 30 cycles — on the very endpoint automatic recall shares, during the
    outage when recall needs it most."""

    def setUp(self):
        a_state_dir()

    def test_a_repo_whose_job_failed_on_infrastructure_is_not_re_queued_at_once(self):
        path = a_file_on_disk("x = 1\n")
        ix = FakeIndex(changed=[path], indexed=[path])
        watch = indexer.watcher(index=ix)
        run_job = indexer.work(index=ix)

        attempts = 0
        for _ in range(12):
            watch()
            job = jobs.load("alpha")
            if job and job.get("state") == jobs.PENDING:
                attempts += 1
                ix.refresh_raises = EmbeddingError("connection refused")
                try:
                    run_job(job)
                except EmbeddingError:
                    jobs.update("alpha", state=jobs.FAILED, error="connection refused")

        self.assertLessEqual(attempts, 2,
                             f"a down endpoint was hit {attempts} times in 12 cycles")

    def test_a_SUCCESSFUL_job_clears_the_backoff(self):
        """The other half of the breaker, and the one that costs silence when it is missing:
        an endpoint that came back must not leave every repo idle for the rest of the cooldown.
        Its sibling below expires the backoff by hand, which holds `Breaker.clear()` but not
        the indexer's call to it."""
        path = a_file_on_disk("x = 1\n")
        ix = FakeIndex(changed=[path], indexed=[path])
        run_job = indexer.work(index=ix)
        indexer.index_breaker().arm()
        self.assertIsNotNone(indexer.index_breaker().is_open(),
                             "precondition: the breaker is armed")

        jobs.enqueue("alpha", "refresh", [path])
        run_job(jobs.load("alpha"))

        self.assertIsNone(indexer.index_breaker().is_open(),
                          "the endpoint answered, but every repo stays idle for the cooldown")

    def test_the_backoff_expires_so_the_repo_is_retried(self):
        """A breaker that never reopens is an outage that never ends."""
        path = a_file_on_disk("x = 1\n")
        ix = FakeIndex(changed=[path], indexed=[path])
        indexer.index_breaker().arm()
        self.assertIsNotNone(indexer.index_breaker().is_open(), "precondition: armed")

        indexer.index_breaker().clear()
        watch = indexer.watcher(index=ix)
        watch()
        watch()
        self.assertTrue(jobs.load("alpha"), "a cleared breaker must let work through")


class TestAnUnknownJobKindIsNotASilentSuccess(unittest.TestCase):
    """The `index` path is the fall-through, so a kind nobody implemented iterates an empty
    `paths`, does nothing, and gets stamped `done`. The one axis the spec calls open/closed is
    where a future job kind would fail silently — and tolerated failure must not become a lie.
    """

    def setUp(self):
        a_state_dir()

    def test_a_kind_the_worker_does_not_know_raises_instead_of_reporting_done(self):
        ix = FakeIndex()
        run_job = indexer.work(index=ix)
        with self.assertRaises(Exception) as caught:
            run_job({"repo": "alpha", "kind": "reindex-all", "paths": []})
        self.assertIn("reindex-all", str(caught.exception),
                      "the message must name the kind nobody handled")

    def test_the_two_known_kinds_still_run(self):
        """The guard must key on the kind being UNKNOWN, not on `paths` being empty: a
        refresh job legitimately carries none."""
        path = a_file_on_disk("x = 1\n")
        ix = FakeIndex(changed=[path], indexed=[path])
        run_job = indexer.work(index=ix)
        run_job({"repo": "alpha", "kind": "refresh", "paths": []})
        run_job({"repo": "alpha", "kind": "index", "paths": [path]})
        self.assertTrue(ix.refreshed, "the refresh path did not run")
        self.assertTrue(ix.added, "the index path did not run")


class TestAVanishedFileDoesNotLoopForever(unittest.TestCase):
    """A deleted file stays in `changed_paths` forever — correctly, because `refresh` reports
    it as missing rather than deleting its chunks, which is a deliberate decision. But the
    WATCHER kept acting on that report: a refresh job every other cycle, and a refresh is not
    free (it read-and-SHAs every indexed path). The quarantine deliberately does not hold
    `missing`, so nothing stopped it."""

    def setUp(self):
        a_state_dir()

    def test_a_deleted_file_stops_producing_refresh_jobs(self):
        gone = a_file_on_disk("content\n")
        os.unlink(gone)
        ix = FakeIndex(changed=[gone], indexed=[gone])
        watch = indexer.watcher(index=ix)

        enqueued = 0
        for _ in range(10):
            watch()
            job = jobs.load("alpha")
            if job and job.get("state") == jobs.PENDING:
                enqueued += 1
                jobs.update("alpha", state=jobs.DONE)

        self.assertEqual(enqueued, 0,
                         f"a file that no longer exists queued {enqueued} refresh jobs")

    def test_a_file_that_still_exists_is_untouched_by_that_filter(self):
        """The guard must key on the file being GONE, not on it being in `changed`."""
        here = a_file_on_disk("def a():\n    return 1\n")
        ix = FakeIndex(changed=[here], indexed=[here])
        watch = indexer.watcher(index=ix)
        watch()
        watch()
        job = jobs.load("alpha")
        self.assertTrue(job and job.get("state") == jobs.PENDING,
                        "a real changed file must still be refreshed")


class TestBothDoorsIntoTheQueueAreShut(unittest.TestCase):
    """The watcher has TWO sources of work: paths the archive has never seen, and paths it
    has seen and that changed. Subtracting the quarantine from only the first left the loop
    open through the second."""

    def setUp(self):
        a_state_dir()

    def test_an_INDEXED_file_that_later_breaks_stops_being_re_queued(self):
        """The other door into the queue. A file that was indexed and then emptied keeps its
        old chunks (`_write_one` raises before it deletes), so `changed_paths` reports it
        forever — and the quarantine was only subtracted from NEW paths. Measured before the
        fix: 3 refresh jobs in 6 cycles for one emptied file, with the file already held."""
        doomed = a_file_on_disk("content\n")
        ix = FakeIndex(changed=[doomed], fails={doomed: "nothing indexable"})
        watch = indexer.watcher(index=ix)
        run_job = indexer.work(index=ix)

        enqueued = 0
        for _ in range(6):
            watch()
            job = jobs.load("alpha")
            if job and job.get("state") == jobs.PENDING:
                enqueued += 1
                run_job(job)
                jobs.update("alpha", state=jobs.DONE)
        self.assertEqual(enqueued, 1,
                         f"an indexed-then-broken file was re-queued {enqueued} times")

    def test_a_held_file_does_not_keep_a_HEALTHY_file_out_of_the_queue(self):
        """The leak the other tests cannot see, because each uses a repo whose ONLY candidate
        is the doomed file — so `changed` empties and nothing is enqueued either way.

        With a second, healthy candidate the two halves show up. `new_paths` is read RAW at
        the branch and at the payload, so the held file is re-queued forever AND, worse, it
        keeps `new_paths` non-empty so the `refresh` branch is never reached: the genuinely
        changed file is never repaired. The comment promising it is picked up "on a LATER
        cycle, once new_paths is empty again" is unreachable for a permanently held file."""
        doomed = a_file_on_disk("")
        healthy = a_file_on_disk("def a():\n    return 1\n")
        quarantine.record("alpha", doomed, "nothing indexable")

        ix = FakeIndex(changed=[healthy], indexed=[healthy],
                       fails={doomed: "nothing indexable"})
        with unittest.mock.patch.object(indexer, "_new_tracked_paths",
                               lambda entry, indexed, memo: {doomed}):
            watch = indexer.watcher(index=ix)
            queued = []
            for _ in range(6):
                watch()
                job = jobs.load("alpha")
                if job and job.get("state") == jobs.PENDING:
                    queued.append((job["kind"], list(job.get("paths") or [])))
                    jobs.update("alpha", state=jobs.DONE)

        self.assertFalse([k for k, paths in queued if doomed in paths],
                         f"the held file was queued again: {queued}")
        self.assertTrue([k for k, _ in queued if k == "refresh"],
                        f"the healthy file was never refreshed: {queued}")

    def test_held_is_read_once_per_cycle_not_once_per_candidate(self):
        """`held()` reads the file, stats every entry and may rewrite it. Called inside the
        candidate comprehension it cost 110 ms for 2,000 paths, against the 16 ms this
        module's docstring budgets for a whole cycle."""
        root = a_git_repo()
        for i in range(5):
            track(root, f"f{i}.py")
        ix = FakeIndex(changed=[], checkouts=[root], indexed=set())
        watch = indexer.watcher(index=ix)
        calls = []
        real = indexer.quarantine.held
        try:
            indexer.quarantine.held = lambda repo: (calls.append(repo), real(repo))[1]
            watch()
        finally:
            indexer.quarantine.held = real
        self.assertEqual(len(calls), 1, f"held() ran {len(calls)} times in one cycle")


if __name__ == "__main__":
    unittest.main(verbosity=2)
