# tests/test_bigfile_claude.py
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks"))

import bigfile as adapter


def a_transcript(lines) -> str:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as fh:
        for obj in lines:
            fh.write(json.dumps(obj) + "\n")

    return path


ASSISTANT = {"message": {"role": "assistant", "model": "claude-opus-5",
                         "usage": {"input_tokens": 2,
                                   "cache_creation_input_tokens": 5_906,
                                   "cache_read_input_tokens": 598_115}}}
REAL_TURN = {"userType": "external", "entrypoint": "cli",
             "message": {"role": "user", "content": "analise o arquivo"}}
SKILL_NOISE = {"userType": "external", "entrypoint": "cli",
               "message": {"role": "user", "content": "Base directory for this skill: /x"}}
TOOL_RESULT = {"toolUseResult": {"ok": 1}, "userType": "external", "entrypoint": "cli",
               "message": {"role": "user", "content": "--full"}}


class TestReadingTheBudget(unittest.TestCase):
    def test_used_is_the_sum_of_the_three_usage_fields(self):
        """Measured on a real session: 2 + 5,906 + 598,115 = 604,023."""
        b = adapter.budget_from(a_transcript([ASSISTANT]), lambda m: 1_000_000)
        self.assertEqual(b.used, 604_023)
        self.assertTrue(b.exact, "claude-code MEASURES this; it is not an estimate")

    def test_an_unreadable_transcript_yields_an_unusable_budget(self):
        b = adapter.budget_from("/nonexistent/none.jsonl", lambda m: 1_000_000)
        self.assertEqual(b.window, 0, "window 0 makes decide() allow — fail open")


class TestTheEscapeMarker(unittest.TestCase):
    def test_it_is_found_in_a_real_user_turn(self):
        t = a_transcript([REAL_TURN, ASSISTANT,
                          dict(REAL_TURN, message={"role": "user", "content": "leia --full"})])
        self.assertTrue(adapter.escape_requested(t))

    def test_a_tool_result_is_not_a_user_turn(self):
        """Tool results carry role=user. Counting them would let any tool output that
        happens to contain the marker unlock the guard."""
        t = a_transcript([REAL_TURN, TOOL_RESULT])
        self.assertFalse(adapter.escape_requested(t))

    def test_the_marker_only_counts_in_the_LAST_turn(self):
        """The escape is scoped to one turn by construction: it evaporates on the next
        prompt, so the guard can never be left off by forgetting."""
        t = a_transcript([dict(REAL_TURN, message={"role": "user", "content": "--full"}),
                          ASSISTANT, REAL_TURN])
        self.assertFalse(adapter.escape_requested(t))


class TestFailOpen(unittest.TestCase):
    def test_a_broken_transcript_line_does_not_raise(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as fh:
            fh.write("{not json\n")
        self.assertEqual(adapter.budget_from(path, lambda m: 1_000_000).window, 0)


# --- Beyond the plan's six tests --------------------------------------------------------
# Ruling 2 of this task's brief makes the ORDER a REQUIREMENT: `decide()` runs first with no
# `indexed_ids` and no network, and only a verdict of block may pay for the ids. The six
# tests above never reach `main()`, so that requirement — and the deny contract, and every
# fail-open path through the entry point — would ship with nothing biting on them.
#
# These imports sit HERE, below the block transcribed from the plan, so that block stays
# byte-identical to it.
import contextlib          # noqa: E402
import io                  # noqa: E402
import types               # noqa: E402
import unittest.mock       # noqa: E402

from core.docs import doc_id_for  # noqa: E402

#: 171k tokens at the core's 4 chars/token. With window=1M and used=604,023 this is 43% of
#: the 396k free — over the 40% share — and the same arithmetic the plan's own numbers use.
A_BIG_FILE = 4 * 171_000


def a_file_of(chars: int) -> str:
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as fh:
        fh.write("x" * chars)

    return path


def a_read_payload(path: str, transcript: str) -> str:
    """The shape read out of the claude v2.1.233 binary, keys and all."""
    return json.dumps({"session_id": "s1", "transcript_path": transcript, "cwd": "/x",
                       "permission_mode": "default", "hook_event_name": "PreToolUse",
                       "tool_name": "Read", "tool_input": {"file_path": path},
                       "tool_use_id": "t1"})


class Spy:
    """Stands in for `indexed_ids`, and COUNTS. A stub that only returned a value would
    let the round trip move to the common path without a single test noticing."""

    def __init__(self, result=None):
        self.calls = 0
        self.result = result

    def __call__(self, cfg):
        self.calls += 1

        return self.result


def run_main(payload: str, ids_spy, loader=None, window: int = 1_000_000) -> str:
    """`main()` end to end on a fabricated payload, with NO network anywhere.

    `indexed_ids` is replaced wholesale, so no test here can reach Qdrant even if the
    ordering regresses — the spy's call count is what reports the regression instead.
    """
    out = io.StringIO()
    cfg = types.SimpleNamespace(context_window=window)
    with unittest.mock.patch.object(adapter.core, "load", loader or (lambda: cfg)), \
         unittest.mock.patch.object(adapter, "indexed_ids", ids_spy), \
         unittest.mock.patch.object(sys, "stdin", io.StringIO(payload)), \
         contextlib.redirect_stdout(out):
        adapter.main()

    return out.getvalue()


class TestTheOrderThatProtectsTheCommonPath(unittest.TestCase):
    """Who is indexed costs a round trip, and this hook runs before EVERY read."""

    def test_an_allowed_read_never_asks_who_is_indexed(self):
        """The common case is "small file, allow". Taxing it with a Qdrant call it never
        needed is the whole reason `decide` takes `indexed_ids` from outside."""
        spy = Spy()
        out = run_main(a_read_payload(a_file_of(400), a_transcript([ASSISTANT])), spy)
        self.assertEqual(out, "", "an allowed read must emit nothing at all")
        self.assertEqual(spy.calls, 0, "the common path paid for ids it never used")

    def test_a_block_pays_for_the_ids_exactly_once(self):
        spy = Spy(set())
        out = run_main(a_read_payload(a_file_of(A_BIG_FILE), a_transcript([ASSISTANT])), spy)
        self.assertEqual(spy.calls, 1)
        emitted = json.loads(out)["hookSpecificOutput"]
        self.assertEqual(emitted["hookEventName"], "PreToolUse")
        self.assertEqual(emitted["permissionDecision"], "deny")
        self.assertIn("index it with docs_index", emitted["permissionDecisionReason"])

    def test_the_second_pass_is_what_upgrades_the_message(self):
        """Without the second call the model is told to index what is already indexed."""
        path = a_file_of(A_BIG_FILE)
        spy = Spy({doc_id_for(path)})
        reason = json.loads(run_main(a_read_payload(path, a_transcript([ASSISTANT])),
                                     spy))["hookSpecificOutput"]["permissionDecisionReason"]
        self.assertIn("already indexed", reason)
        self.assertIn("docs_search", reason)

    def test_the_escape_marker_stops_the_block_before_the_round_trip(self):
        """--full is the user overruling the guard; there is nothing left to enrich."""
        t = a_transcript([ASSISTANT,
                          dict(REAL_TURN, message={"role": "user", "content": "leia --full"})])
        spy = Spy(set())
        self.assertEqual(run_main(a_read_payload(a_file_of(A_BIG_FILE), t), spy), "")
        self.assertEqual(spy.calls, 0)


class TestMainFailsOpen(unittest.TestCase):
    """Every error path emits nothing and exits 0: a guard that breaks gets out of the way."""

    def test_a_payload_that_is_not_json_emits_nothing(self):
        self.assertEqual(run_main("{not json", Spy(set())), "")

    def test_no_transcript_path_means_allow_and_not_a_guess(self):
        """Ruling 1: the path is never derived from session_id. The host writes the key
        unconditionally (read from the v2.1.233 binary); a host that does not is a host
        this guard stays out of the way of."""
        spy = Spy(set())
        payload = json.dumps({"session_id": "s1", "tool_name": "Read",
                              "tool_input": {"file_path": a_file_of(A_BIG_FILE)}})
        self.assertEqual(run_main(payload, spy), "")
        self.assertEqual(spy.calls, 0)

    def test_a_config_that_will_not_load_allows_the_read(self):
        """Ledger F3: `core.load()` raises on a malformed numeric field before a Config
        exists, so `window_for`'s tolerance never runs. A typo in a QCTX_* variable must
        not turn into a file nobody can read."""
        def explode():
            raise ValueError("invalid literal for int() with base 10: 'banana'")

        spy = Spy(set())
        payload = a_read_payload(a_file_of(A_BIG_FILE), a_transcript([ASSISTANT]))
        self.assertEqual(run_main(payload, spy, loader=explode), "")
        self.assertEqual(spy.calls, 0)


if __name__ == "__main__":
    unittest.main()
