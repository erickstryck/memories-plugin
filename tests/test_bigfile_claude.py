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
import http.server        # noqa: E402
import subprocess         # noqa: E402
import threading          # noqa: E402
import time               # noqa: E402


from core import inventory  # noqa: E402
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
         unittest.mock.patch.object(inventory, "indexed_ids", ids_spy), \
         unittest.mock.patch.object(sys, "stdin", io.StringIO(payload)), \
         contextlib.redirect_stdout(out):
        adapter.main()

    return out.getvalue()


#: The real hook, run as a real process. Anything read at IMPORT time — both env knobs
#: are — can only be exercised this way; an in-process patch would test a value the module
#: never read.
HOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "hooks", "bigfile.py")


def hook_env(**overrides) -> dict:
    """A hermetic environment for a hook subprocess.

    Nothing here may reach the operator's world: `QCTX_CONFIG` points at a file that will
    never exist (so the environment is the whole config), `QCTX_STATE_DIR` is a fresh temp
    directory (so the breaker never touches ~/.memories-plugin), and Qdrant points at a
    port nobody listens on — refused instantly, which is the degraded path these tests
    WANT unless one of them stands a server up itself.
    """
    state = tempfile.mkdtemp()
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG")}
    env.update({
        "QCTX_CONFIG": os.path.join(state, "config.json"),
        "QCTX_STATE_DIR": state,
        "QCTX_QDRANT_URL": "http://127.0.0.1:1",
        "QCTX_EMBED_URL": "http://127.0.0.1:1/v1/embeddings",
        "QCTX_EMBED_MODEL": "test-embed",
        "QCTX_MEMORY_COLLECTION": "t_mem",
        "QCTX_DOCS_COLLECTION": "t_tmp",
        "QCTX_LIBRARY_COLLECTION": "t_lib",
        "QCTX_CONTEXT_WINDOW": "1000000",
    })
    env.update(overrides)

    return env


def run_hook(path: str, transcript: str, env: dict):
    """(elapsed, stdout, returncode) for the real hook process."""
    started = time.monotonic()
    done = subprocess.run([sys.executable, HOOK], input=a_read_payload(path, transcript),
                          env=env, capture_output=True, text=True, timeout=30)

    return time.monotonic() - started, done.stdout, done.returncode


def a_usage(used: int) -> dict:
    """An assistant turn reporting exactly `used` tokens of context."""
    return {"message": {"role": "assistant", "model": "claude-opus-5",
                        "usage": {"input_tokens": used, "cache_creation_input_tokens": 0,
                                  "cache_read_input_tokens": 0}}}


#: Isolates the FLOOR: 790k used of 1M plus 20k costs 810k, past the 800k the floor allows,
#: while 20k stays under 40% of the 210k free. Same arithmetic as the core's own isolator.
FLOOR_ONLY = (a_usage(790_000), 4 * 20_000)
#: Isolates the SHARE: 604,023 used of 1M leaves 396k free; 171k is 43% of it, over the 40%
#: share, while 775k of 1M keeps 22% and never reaches the floor.
SHARE_ONLY = (a_usage(604_023), 4 * 171_000)


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


class SlowQdrant(http.server.BaseHTTPRequestHandler):
    """A REAL Qdrant, answering every endpoint `list_docs` touches, correctly, but slowly.

    Correctly is the load-bearing half. A server that 500s would also make the hook finish
    fast — by failing — and would prove nothing about the deadline. So the fast variant of
    the same server has to produce the "already indexed" message, which is only reachable
    when every one of the six round trips was understood and answered.
    """

    delay = 0.0
    doc_id = ""

    def log_message(self, *a):
        pass                                   # keep the test output clean

    def _answer(self, payload):
        time.sleep(self.delay)
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        # `ensure` asks for the collection. Reporting no declared vector size is what makes
        # the client accept it as compatible instead of trying to create it.
        self._answer({"result": {"config": {"params": {"vectors": {}}}, "points_count": 1}})

    def do_PUT(self):
        self._answer({"result": True})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if self.path.endswith("/points/scroll"):
            self._answer({"result": {"points": [{"id": 1, "payload": {
                "doc_id": self.doc_id,
                "metadata": {"path": "/x", "mode": "snapshot", "indexed_at": "now"},
            }}], "next_page_offset": None}})
        else:
            self._answer({"result": {"status": "ok"}})       # sweep's delete_by_filter


class TestTheEnrichmentCannotBlowTheHookDeadline(unittest.TestCase):
    """`timeout: 5` in hooks.json is a hard budget, and a per-call timeout cannot defend it.

    `list_docs("all")` is a structural minimum of SIX sequential round trips. Measured
    before the wall clock existed, against this very server at 1.2 s per call: every
    per-call ceiling respected, and the hook took 7.267 s against a 5 s budget. The count
    of round trips is knowledge that rots; a wall clock does not care how many there are.
    """

    def _serve(self, delay, doc_id):
        handler = type("H", (SlowQdrant,), {"delay": delay, "doc_id": doc_id})
        # A server that stays QUIET about the broken pipe it is guaranteed to see: the
        # abandoned worker's socket dies with the hook process, which is the deadline
        # working, not a fault. Left alone it prints a traceback into the suite's output.
        quiet = type("QuietServer", (http.server.ThreadingHTTPServer,),
                     {"handle_error": lambda self, request, addr: None})
        server = quiet(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)

        return f"http://127.0.0.1:{server.server_address[1]}"

    def _run_the_hook(self, delay):
        """Returns (elapsed, stdout, returncode) for the REAL hook process."""
        path = a_file_of(A_BIG_FILE)
        url = self._serve(delay, doc_id_for(path))

        return run_hook(path, a_transcript([ASSISTANT]), hook_env(QCTX_QDRANT_URL=url))

    def test_a_slow_but_healthy_qdrant_cannot_blow_the_deadline(self):
        elapsed, out, code = self._run_the_hook(delay=1.2)
        self.assertEqual(code, 0, "the hook must exit 0 on every path")
        self.assertLess(elapsed, 3.5,
                        f"took {elapsed:.3f}s of the 5s budget; six 1.2s round trips "
                        f"unbounded would be ~7.3s")
        reason = json.loads(out)["hookSpecificOutput"]["permissionDecisionReason"]
        self.assertIn("index it with docs_index", reason,
                      "abandoning the lookup costs the better message, never the block")

    def test_the_same_server_answered_fast_delivers_the_better_message(self):
        """The control, and it is what makes the test above mean anything: the same server
        that was too slow is UNDERSTOOD when it is quick, so the slow run degraded because
        of the clock and not because the fake spoke a language the client rejects."""
        elapsed, out, code = self._run_the_hook(delay=0.0)
        self.assertEqual(code, 0)
        self.assertLess(elapsed, 3.5, f"took {elapsed:.3f}s")
        reason = json.loads(out)["hookSpecificOutput"]["permissionDecisionReason"]
        self.assertIn("already indexed", reason)
        self.assertIn("docs_search", reason)


class TestTheTwoThresholdsAreRealKnobs(unittest.TestCase):
    """Both are read at IMPORT time and handed to `decide` — and until T6 puts this file
    under the knob scan, nothing else would notice if either stopped arriving there.

    Each fixture trips exactly ONE criterion, so loosening one knob can only be observed
    through the criterion it governs. A fixture that tripped both would let a dead knob
    look alive because the other criterion kept blocking.
    """

    def _decision(self, fixture, **env):
        usage, size = fixture
        elapsed, out, code = run_hook(a_file_of(size), a_transcript([usage]),
                                      hook_env(**env))
        self.assertEqual(code, 0, "the hook must exit 0 on every path")

        return "deny" if out.strip() else "allow"

    def test_the_floor_knob_moves_the_decision(self):
        self.assertEqual(self._decision(FLOOR_ONLY), "deny")
        self.assertEqual(self._decision(FLOOR_ONLY, QCTX_BIGFILE_FLOOR_PCT="0.001"), "allow")

    def test_the_share_knob_moves_the_decision(self):
        self.assertEqual(self._decision(SHARE_ONLY), "deny")
        self.assertEqual(self._decision(SHARE_ONLY, QCTX_BIGFILE_SHARE_PCT="0.9"), "allow")

    def test_the_legacy_names_are_accepted_too(self):
        self.assertEqual(self._decision(FLOOR_ONLY, BIGFILE_FLOOR_PCT="0.001"), "allow")
        self.assertEqual(self._decision(SHARE_ONLY, BIGFILE_SHARE_PCT="0.9"), "allow")

    def test_a_malformed_knob_falls_back_instead_of_killing_the_hook(self):
        """These are read ABOVE main()'s catch-all, where a raise is not a fail-open — it
        is a non-zero exit and a traceback on a hook that runs before every file read."""
        garbage = {"QCTX_BIGFILE_FLOOR_PCT": "20%", "QCTX_BIGFILE_SHARE_PCT": "banana"}
        self.assertEqual(self._decision(FLOOR_ONLY, **garbage), "deny")
        self.assertEqual(self._decision(SHARE_ONLY, **garbage), "deny")

    def test_both_passes_of_decide_are_given_the_knobs(self):
        """The silent regression this exists for: dropping `floor_pct=`/`share_pct=` from
        either call leaves every test above green — the module defaults happen to match —
        and the knobs quietly stop working. So assert the ARGUMENTS, not the outcome."""
        seen = []
        real = adapter.bigfile.decide

        def recording(path, budget, **kw):
            seen.append(kw)

            return real(path, budget, **kw)

        spy = Spy(set())
        with unittest.mock.patch.object(adapter.bigfile, "decide", recording):
            run_main(a_read_payload(a_file_of(A_BIG_FILE), a_transcript([ASSISTANT])), spy)
        self.assertEqual(len(seen), 2, "the two-pass order is what this rides on")
        for call in seen:
            self.assertIn("floor_pct", call, "decide() was called without the floor knob")
            self.assertIn("share_pct", call, "decide() was called without the share knob")
            self.assertEqual(call["floor_pct"], adapter.FLOOR_PCT)
            self.assertEqual(call["share_pct"], adapter.SHARE_PCT)


#: Violates ONLY `userType`: an internal turn that is otherwise a perfect user turn.
AGENT_TURN = {"userType": "agent", "entrypoint": "cli",
              "message": {"role": "user", "content": "leia --full"}}
#: Violates ONLY `entrypoint`: a real external user, arriving through something that is
#: not the interactive CLI.
SDK_TURN = {"userType": "external", "entrypoint": "sdk",
            "message": {"role": "user", "content": "leia --full"}}


class TestTheOtherTwoHalvesOfTheRealTurnFilter(unittest.TestCase):
    """`toolUseResult` already had a test that bites; these two did not.

    ONE condition violated per fixture, never two — a fixture that broke both would go
    green with either check deleted and prove neither of them. Each transcript ends with
    the impostor and keeps a genuine, marker-free user turn behind it, so the marker can
    only be found by accepting the impostor.
    """

    def test_an_internal_turn_does_not_carry_the_escape(self):
        t = a_transcript([REAL_TURN, ASSISTANT, AGENT_TURN])
        self.assertFalse(adapter.escape_requested(t),
                         "userType=agent is not the human overruling the guard")

    def test_a_turn_from_another_entrypoint_does_not_carry_the_escape(self):
        t = a_transcript([REAL_TURN, ASSISTANT, SDK_TURN])
        self.assertFalse(adapter.escape_requested(t),
                         "entrypoint=sdk is not the interactive CLI the marker is typed in")


#: Runs `main()` in a fresh interpreter and reports, on stderr, whether the ONE module that
#: reaches the network was ever loaded. stderr because stdout is the hook protocol.
_LAZY_IMPORT_PROBE = """
import json, sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import bigfile

bigfile.main()
print(json.dumps({"network_module_loaded": "core.inventory" in sys.modules}), file=sys.stderr)
"""


class TestTheCommonPathNeverLoadsTheNetworkModule(unittest.TestCase):
    """`core.inventory` is imported INSIDE the rare branch, and this is the proof by
    EXECUTION that it stays there — a grep for the import line would pass just as happily
    with the import at the top of the file, which is the regression that matters.

    The allow case is the claim; the block case is the CONTROL. Without it, "never loaded"
    could equally mean the probe never worked.
    """

    def _probe(self, size):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        done = subprocess.run(
            [sys.executable, "-c", _LAZY_IMPORT_PROBE, repo, os.path.join(repo, "hooks")],
            input=a_read_payload(a_file_of(size), a_transcript([ASSISTANT])),
            env=hook_env(), cwd="/", capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)

        return json.loads(done.stderr.strip().splitlines()[-1])["network_module_loaded"]

    def test_an_allowed_read_does_not_even_import_it(self):
        self.assertFalse(self._probe(size=400),
                         "the common path loaded the module that talks to Qdrant")

    def test_a_blocked_read_does(self):
        self.assertTrue(self._probe(size=A_BIG_FILE),
                        "the control failed: the probe cannot tell loaded from not loaded")


if __name__ == "__main__":
    unittest.main()
