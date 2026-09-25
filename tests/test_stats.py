"""`qctx stats`: what both hosts and the daemon recorded, summarised.

The numbers in the design of this feature (p50 1.1 s, p95 2.7 s, 13 Qdrant failures over a
month) were computed by hand from `recall.log`, and that is the job this does. It reads only
what `core.recall_log` and `core.daemon` write, so it has no opinion about how recall works.
"""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import stats  # noqa: E402

RECALL = """\
2026-09-20 10:00:00 round 1: 6 injected + 0 pointers (out of 6 relevant / 26 candidates) in 0.6s | 2 angles | CE=True | 'a'
2026-09-20 10:01:00 round 2: 0 above the cut (best 0.407) in 0.2s | 2 angles | CE=False collapsed=False dropped=0 | 'b'
2026-09-20 10:02:00 Qdrant failed (x) \u2014 no recall on this prompt
2026-09-25 11:00:00 [hermes] round 1: 2 injected + 1 pointers (out of 3 relevant / 20 candidates) in 1.0s | 2 angles | CE=True | 'c'
2026-09-25 11:01:00 [hermes] round 2: 1 injected + 0 pointers (out of 1 relevant / 20 candidates) in 3.0s | 2 angles | CE=True | 'd'
2026-09-25 11:02:00 [hermes] skip (trivial prompt): 'ok'
2026-09-25 11:03:00 [hermes] embeddings failed (timeout), no recall on this prompt
2026-09-25 11:04:00 [hermes] re-rank in breaker: failed 12s ago, strict dense cut
this line is garbage
"""

DAEMON = """\
2026-09-25 11:33:27 start pid=1 version=1.1.0
2026-09-25 11:33:31 sources repo=awesome-cv3 files=10759 in 3.9s
2026-09-25 11:33:38 enqueue repo=awesome-cv3 kind=index paths=6
2026-09-25 11:33:46 job repo=awesome-cv3 kind=index result=done in 3.7s
2026-09-25 11:34:46 job repo=core kind=refresh result=failed in 0.2s (EmbeddingError: down)
2026-09-25 11:35:00 watch failed (ConnectionError: down)
2026-09-25 11:36:00 stop reason='no live lease'
"""


def a_state_dir(recall: str = RECALL, daemon: str = DAEMON) -> Path:
    d = Path(tempfile.mkdtemp())
    os.environ["QCTX_STATE_DIR"] = str(d)
    if recall is not None:
        (d / "recall.log").write_text(recall)
    if daemon is not None:
        (d / "daemon.log").write_text(daemon)

    return d


class TestSummarize(unittest.TestCase):
    def test_rounds_are_counted_per_host(self):
        out = stats.summarize(a_state_dir())
        cc, hermes = out["recall"]["claude-code"], out["recall"]["hermes"]
        self.assertEqual((cc["rounds"], cc["with_memories"], cc["empty"]), (2, 1, 1))
        self.assertEqual((hermes["rounds"], hermes["with_memories"], hermes["empty"]), (2, 2, 0))

    def test_latency_percentiles(self):
        hermes = stats.summarize(a_state_dir())["recall"]["hermes"]
        self.assertEqual((hermes["p50"], hermes["max"]), (1.0, 3.0))
        self.assertEqual(hermes["p95"], 3.0)

    def test_failures_by_dependency_skips_and_breaker(self):
        out = stats.summarize(a_state_dir())["recall"]
        self.assertEqual(out["claude-code"]["failures"], {"qdrant": 1})
        self.assertEqual(out["hermes"]["failures"], {"embeddings": 1})
        self.assertEqual((out["hermes"]["skips"], out["hermes"]["breaker"]), (1, 1))

    def test_the_period_each_host_covers(self):
        out = stats.summarize(a_state_dir())["recall"]
        self.assertEqual(out["hermes"]["first"], "2026-09-25 11:00:00")
        self.assertEqual(out["hermes"]["last"], "2026-09-25 11:04:00")

    def test_the_daemon_summary(self):
        d = stats.summarize(a_state_dir())["daemon"]
        self.assertEqual(d["jobs"], {"done": 1, "failed": 1})
        self.assertEqual((d["enqueued"], d["watcher_errors"], d["archive_reads"]), (1, 1, 1))
        self.assertEqual(d["last_start"], "2026-09-25 11:33:27")
        self.assertEqual(d["version"], "1.1.0")
        self.assertIn("ConnectionError: down", d["last_error"], "the MOST RECENT error")

    def test_no_logs_at_all_is_an_empty_summary_not_an_error(self):
        out = stats.summarize(a_state_dir(recall=None, daemon=None))
        self.assertEqual(out["recall"], {})
        self.assertEqual(out["daemon"]["jobs"], {})

    def test_an_unreadable_log_is_an_empty_summary_not_an_error(self):
        d = a_state_dir(recall=None, daemon=None)
        (d / "recall.log").write_bytes(b"\xff\xfe\x00garbage")
        self.assertEqual(stats.summarize(d)["recall"], {})


class TestTheCommand(unittest.TestCase):
    def _run(self, *argv) -> str:
        from tests.test_cli_repos import load_cli
        cli = load_cli()
        args = cli.build_parser().parse_args(["stats", *argv])
        if not hasattr(args, "json"):
            args.json = False
        out = io.StringIO()
        with redirect_stdout(out):
            args.fn(args, None)

        return out.getvalue()

    def test_json(self):
        a_state_dir()
        payload = json.loads(self._run("--json"))
        self.assertEqual(payload["recall"]["hermes"]["rounds"], 2)

    def test_text_names_each_host_and_the_daemon(self):
        a_state_dir()
        text = self._run()
        for fragment in ("hermes", "claude-code", "p50 1.0s", "p95 3.0s", "embeddings 1",
                         "daemon", "failed 1", "watcher errors 1"):
            self.assertIn(fragment, text)

    def test_text_with_no_logs_says_so(self):
        a_state_dir(recall=None, daemon=None)
        self.assertIn("nothing recorded", self._run())

    def test_it_does_not_need_a_configuration(self):
        """Reading two local files needs no Qdrant URL. `main` loads the config before any
        handler runs, and a machine with no config must still be able to see its own logs."""
        import subprocess
        d = a_state_dir()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("QCTX_", "QDRANT"))}
        env.update(QCTX_STATE_DIR=str(d), QCTX_CONFIG=str(d / "absent.json"))
        root = Path(__file__).resolve().parent.parent
        done = subprocess.run([sys.executable, str(root / "cli" / "qctx.py"), "stats", "--json"],
                              capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout)["recall"]["hermes"]["rounds"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
