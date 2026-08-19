"""The surface the user touches, and the worker the daemon runs.

The worker is tested with a fake index: what is measured is that it indexes in batches, updates
progress, and STOPS when a cancellation arrives — not that Qdrant works.
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
    def __init__(self):
        self.added = []

    def add_files(self, repo, paths):
        self.added.append(list(paths))

        return {"repo": repo, "files": len(paths), "chunks": len(paths), "skipped": []}


class TestTheWorker(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_it_indexes_every_path_of_the_job(self):
        ix = FakeIndex()
        jobs.enqueue("alpha", "index", ["/a.py", "/b.py", "/c.py"])
        indexer.work(index=ix)(jobs.load("alpha"))
        self.assertEqual(sorted(p for batch in ix.added for p in batch),
                         ["/a.py", "/b.py", "/c.py"])

    def test_progress_advances_as_it_goes(self):
        ix = FakeIndex()
        jobs.enqueue("alpha", "index", ["/a.py", "/b.py"])
        indexer.work(index=ix, batch=1)(jobs.load("alpha"))
        self.assertEqual(jobs.load("alpha")["done"], 2)

    def test_it_STOPS_when_a_cancel_arrives_mid_job(self):
        """Cancellation is read BETWEEN batches, so it stops at a point where what is already
        indexed is consistent — and what already went in STAYS."""
        ix = FakeIndex()
        jobs.enqueue("alpha", "index", [f"/f{i}.py" for i in range(10)])

        original = ix.add_files

        def cancel_after_first(repo, paths):
            out = original(repo, paths)
            jobs.request_cancel("alpha")

            return out

        ix.add_files = cancel_after_first
        indexer.work(index=ix, batch=1)(jobs.load("alpha"))
        self.assertEqual(len(ix.added), 1, "did not stop at the cancellation")
        self.assertEqual(jobs.load("alpha")["done"], 1, "what already went in was discarded")

    def test_a_refresh_job_calls_refresh_and_not_add_files(self):
        class RefreshIndex(FakeIndex):
            def __init__(self):
                super().__init__()
                self.refreshed = []

            def refresh(self, repo, should_stop=None):
                self.refreshed.append(repo)

                return []

        ix = RefreshIndex()
        jobs.enqueue("alpha", "refresh", [])
        indexer.work(index=ix)(jobs.load("alpha"))
        self.assertEqual(ix.refreshed, ["alpha"])
        self.assertEqual(ix.added, [])

    def test_a_refresh_job_PASSES_should_stop_so_a_cancel_can_interrupt_it(self):
        """Whole-branch review finding 9: `target.refresh(repo)` used to be called with no
        cancel check reachable inside it, so a refresh queued by the watcher and cancelled
        mid-run would re-embed everything changed and still end up marked `cancelled` — a
        claim about work that had fully run. This proves the WIRING: `should_stop` reaches
        `refresh` and, when it fires, `refresh` actually stops (here, after the first file)."""
        class RefreshIndex(FakeIndex):
            def __init__(self):
                super().__init__()
                self.judged = []

            def refresh(self, repo, should_stop=None):
                for path in ("/a.py", "/b.py", "/c.py"):
                    if should_stop is not None and should_stop():
                        break
                    self.judged.append(path)

                return [{"path": p, "action": "reindexed"} for p in self.judged]

        ix = RefreshIndex()
        jobs.enqueue("alpha", "refresh", [])
        jobs.request_cancel("alpha")                    # cancelled BEFORE the job even starts
        indexer.work(index=ix)(jobs.load("alpha"))
        self.assertEqual(ix.judged, [],
                         "should_stop did not reach refresh, or refresh ignored it")


if __name__ == "__main__":
    unittest.main(verbosity=2)
