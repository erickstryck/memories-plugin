"""Keeping the index current without anyone asking.

The daemon walks the indexed repos and compares `mtime` against what is in the archive. Measured
on 2026-08-18: 16 ms for 2,000 files — cheap enough to make `inotify` unnecessary, which would
otherwise mean an external dependency or platform-specific code.

THE DEBOUNCE IS WHAT SEPARATES THIS FROM REINDEXING ON EVERY KEYSTROKE: a file only enters the
queue once it has been stable for one cycle.
"""
import os
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

    def __init__(self, changed=()):
        self._changed = list(changed)
        self.refreshed = []

    def list_repos(self):
        return [{"repo": "alpha", "checkouts": ["/tmp/alpha"]}]

    def changed_paths(self, repo):
        return list(self._changed)

    def refresh(self, repo):
        self.refreshed.append(repo)

        return []


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
