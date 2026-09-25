"""The two log modules: `eventlog` writes a line, `recall_log` owns what a recall line says.

WHY THEY EXIST. Only the claude-code hook ever wrote `recall.log`, with its own file handling
inline; the hermes provider, which is the host used every day, wrote nothing, and the daemon
sent everything to /dev/null. The writing moved into one owner so all three can use it, and the
line format moved into another so the hosts cannot drift apart and `qctx stats` can read what
both of them wrote.
"""
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import eventlog, recall_log  # noqa: E402
from core.retrieval import CE, Outcome, Scored  # noqa: E402


def judged(**kw) -> Outcome:
    """An Outcome the cross-encoder took part in, so `by_rerank` is true."""
    return Outcome(scored=[Scored(item=None, score=0.9, origin=CE)], **kw)


def a_state_dir() -> Path:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return Path(d)


class TestEventlog(unittest.TestCase):
    def setUp(self):
        self.dir = a_state_dir()

    def test_a_line_lands_with_a_timestamp(self):
        self.assertTrue(eventlog.write(eventlog.DAEMON, "hello"))
        text = (self.dir / "daemon.log").read_text()
        self.assertRegex(text, r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d hello\n$")

    def test_lines_are_appended(self):
        eventlog.write(eventlog.DAEMON, "one")
        eventlog.write(eventlog.DAEMON, "two")
        self.assertEqual(len((self.dir / "daemon.log").read_text().splitlines()), 2)

    def test_a_new_log_is_created_owner_only(self):
        """It names sessions, prompts and pids. `open("a")` would have taken the umask."""
        eventlog.write(eventlog.DAEMON, "x")
        self.assertEqual(stat.S_IMODE((self.dir / "daemon.log").stat().st_mode), 0o600)

    def test_rotation_keeps_the_tail_and_publishes_600(self):
        target = self.dir / "recall.log"
        target.write_text("old\n" * 50 + "newest\n")
        os.chmod(target, 0o664)
        self.assertTrue(eventlog.rotate(target, max_bytes=40))
        self.assertTrue(target.read_text().endswith("newest\n"), "rotation kept the wrong end")
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)

    def test_write_rotates_once_the_log_grows_past_the_ceiling(self):
        target = self.dir / "daemon.log"
        target.write_text("x" * (eventlog.MAX_BYTES + 10))
        eventlog.write(eventlog.DAEMON, "after")
        self.assertLess(target.stat().st_size, eventlog.MAX_BYTES)
        self.assertTrue(target.read_text().endswith(" after\n"))

    def test_an_unwritable_state_dir_answers_false_and_never_raises(self):
        os.chmod(self.dir, 0o500)
        self.addCleanup(os.chmod, self.dir, 0o700)
        self.assertFalse(eventlog.write(eventlog.DAEMON, "lost"))

    def test_a_newline_in_the_message_cannot_forge_a_second_line(self):
        """A prompt is user text. A newline in it would become a line of its own, which
        `qctx stats` would then read as a round that never happened."""
        eventlog.write(eventlog.RECALL, "first\n2026-01-01 00:00:00 [hermes] round 1: forged")
        self.assertEqual(len((self.dir / "recall.log").read_text().splitlines()), 1)


class TestRecallLog(unittest.TestCase):
    def setUp(self):
        self.dir = a_state_dir()

    def _last(self) -> str:
        return (self.dir / "recall.log").read_text().splitlines()[-1]

    def test_a_record_names_its_host(self):
        recall_log.record("hermes", recall_log.skip_line("short prompt", "ok"))
        self.assertIn(" [hermes] skip (short prompt)", self._last())

    def test_a_round_line_parses_back_into_its_numbers(self):
        line = recall_log.round_line(3, 4, 2, 6, judged(candidates=27), elapsed=1.234,
                                     angles=2, prompt="how does the poll paginate?")
        recall_log.record("hermes", line)
        parsed = recall_log.parse(self._last())
        self.assertEqual(parsed["host"], "hermes")
        self.assertEqual(parsed["kind"], "round")
        self.assertEqual((parsed["injected"], parsed["pointers"]), (4, 2))
        self.assertAlmostEqual(parsed["elapsed"], 1.2)

    def test_an_empty_round_parses_as_a_round_with_nothing_injected(self):
        o = Outcome(candidates=4, best_dense=0.31, reranked=True)
        recall_log.record("claude-code", recall_log.empty_line(1, o, elapsed=0.5, angles=3,
                                                                prompt="nothing here"))
        parsed = recall_log.parse(self._last())
        self.assertEqual((parsed["kind"], parsed["injected"]), ("round", 0))
        self.assertAlmostEqual(parsed["elapsed"], 0.5)

    def test_the_empty_line_keeps_what_the_hook_logged_before(self):
        """The hook's empty line carried CE, collapse, dropped, suppression and the error, so
        that "vetoed everything" and "there was no second stage" can be told apart afterwards.
        Moving the format must not lose any of it."""
        o = Outcome(candidates=4, best_dense=0.31, reranked=True, collapsed=False,
                    dropped_above_floor=2, suppressed="circuit breaker: 3s ago",
                    rerank_error="timeout")
        line = recall_log.empty_line(2, o, elapsed=0.5, angles=3, prompt="p")
        for fragment in ("round 2: 0 above the cut (best 0.310)", "CE=True", "collapsed=False",
                         "dropped=2", "suppressed='circuit breaker: 3s ago'",
                         "error='timeout'", "3 angles"):
            self.assertIn(fragment, line)

    def test_the_round_line_keeps_what_the_hook_logged_before(self):
        o = judged(candidates=41, scale_converted=True)
        line = recall_log.round_line(1, 4, 2, 6, o, elapsed=1.2, angles=3,
                                     prompt="o sistema de logs")
        self.assertEqual(line, "round 1: 4 injected + 2 pointers (out of 6 relevant / 41 "
                               "candidates) in 1.2s | 3 angles | CE=True (scale converted) | "
                               "'o sistema de logs'")

    def test_the_prompt_is_cut_to_sixty_characters_as_before(self):
        line = recall_log.round_line(1, 1, 0, 1, Outcome(), elapsed=0.1, angles=1,
                                     prompt="x" * 200)
        self.assertIn("'" + "x" * 60 + "'", line)
        self.assertNotIn("x" * 61, line)

    def test_a_line_written_before_hosts_were_named_reads_as_claude_code(self):
        """Every existing `recall.log` on disk was written by the hook alone. Reading those
        lines as belonging to nobody would make `qctx stats` forget a month of history."""
        old = ("2026-09-16 10:02:05 round 1: 6 injected + 0 pointers (out of 6 relevant / 26 "
               "candidates) in 0.6s | 2 angles | CE=True (scale converted) | 'como funciona'")
        parsed = recall_log.parse(old)
        self.assertEqual((parsed["host"], parsed["kind"], parsed["injected"]),
                         ("claude-code", "round", 6))

    def test_failures_are_classified_by_dependency(self):
        """Both hosts write these through `recall_log`, so the parser and the writer agree."""
        cases = {
            recall_log.failure_line("embeddings", "boom"): "embeddings",
            recall_log.failure_line("qdrant", "boom"): "qdrant",
            recall_log.failure_line("config", "no url"): "config",
            recall_log.failure_line("unexpected", "KeyError: 'x'"): "unexpected",
            recall_log.rerank_failed_line("timeout", 300): "rerank",
        }
        for msg, dep in cases.items():
            parsed = recall_log.parse(f"2026-09-25 10:00:00 [hermes] {msg}")
            self.assertEqual((parsed["kind"], parsed["dependency"]), ("failure", dep), msg)

    def test_failures_written_by_earlier_versions_are_classified_too(self):
        """The lines already on disk, verbatim from the 1.0.1 hook (em dash included)."""
        dash = "\u2014"
        cases = {
            f"embeddings failed (boom) {dash} no recall on this prompt": "embeddings",
            f"Qdrant failed (boom) {dash} no recall on this prompt": "qdrant",
            f"re-rank failed (timeout) {dash} breaker armed for 300s": "rerank",
            f"incomplete config (no url) {dash} no recall on this prompt": "config",
            "unexpected failure (KeyError: 'x')": "unexpected",
        }
        for msg, dep in cases.items():
            parsed = recall_log.parse(f"2026-09-25 10:00:00 {msg}")
            self.assertEqual((parsed["host"], parsed["kind"], parsed["dependency"]),
                             ("claude-code", "failure", dep), msg)

    def test_other_kinds(self):
        for msg, kind in ((recall_log.skip_line("short prompt", "x"), "skip"),
                          (recall_log.breaker_line(12), "breaker"),
                          (recall_log.cleanup_line(3), "cleanup"),
                          ("config: QCTX_RECALL_MAX_CHARS='14k' is not a number", "config")):
            self.assertEqual(recall_log.parse(f"2026-09-25 10:00:00 [hermes] {msg}")["kind"],
                             kind, msg)

    def test_garbage_is_ignored_not_raised(self):
        for line in ("", "not a log line", "\x00\x01", "2026-09-25 [hermes]"):
            self.assertIsNone(recall_log.parse(line))


if __name__ == "__main__":
    unittest.main(verbosity=2)
