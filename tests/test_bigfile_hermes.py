# tests/test_bigfile_hermes.py
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hosts.hermes import bigfile as adapter


def a_state_db(messages) -> str:
    path = os.path.join(tempfile.mkdtemp(), "state.db")
    c = sqlite3.connect(path)
    c.execute("create table messages (session_id text, role text, content text, "
              "active int, timestamp real)")
    c.execute("create table sessions (session_id text, model text)")
    c.execute("insert into sessions values ('s1', 'MiniMax-M2.7')")
    for i, (role, content, active) in enumerate(messages):
        c.execute("insert into messages values ('s1', ?, ?, ?, ?)", (role, content, active, i))
    c.commit()
    c.close()

    return path


class TestReadingTheBudget(unittest.TestCase):
    def test_used_is_estimated_from_active_message_bodies(self):
        """hermes leaves messages.token_count NULL on every row — measured. The ratio is
        hermes' own `_chars_to_tokens`: (len+3)//4."""
        db = a_state_db([("user", "x" * 400, 1), ("assistant", "y" * 400, 1)])
        b = adapter.budget_from(db, "s1", lambda m: 200_000)
        self.assertEqual(b.used, 200)
        self.assertFalse(b.exact, "hermes cannot measure it; the message must say so")

    def test_compacted_messages_do_not_count(self):
        db = a_state_db([("user", "x" * 400, 1), ("assistant", "y" * 4000, 0)])
        self.assertEqual(adapter.budget_from(db, "s1", lambda m: 200_000).used, 100)

    def test_a_missing_db_fails_open(self):
        b = adapter.budget_from("/nonexistent/state.db", "s1", lambda m: 200_000)
        self.assertEqual(b.window, 0)


class TestTheEscapeMarker(unittest.TestCase):
    def test_the_marker_is_read_from_the_last_user_message(self):
        db = a_state_db([("user", "primeiro", 1), ("assistant", "r", 1),
                         ("user", "leia --full", 1)])
        self.assertTrue(adapter.escape_requested(db, "s1"))

    def test_an_older_marker_does_not_leak_forward(self):
        db = a_state_db([("user", "--full", 1), ("assistant", "r", 1), ("user", "agora nao", 1)])
        self.assertFalse(adapter.escape_requested(db, "s1"))


class TestItNeverWrites(unittest.TestCase):
    def test_the_database_is_opened_read_only(self):
        """hermes writes to state.db live. A write lock from a hook that fires before every
        file read would be a self-inflicted outage."""
        db = a_state_db([("user", "x", 1)])
        before = os.path.getmtime(db)
        adapter.budget_from(db, "s1", lambda m: 200_000)
        self.assertEqual(os.path.getmtime(db), before)


# --- Beyond the plan's six tests --------------------------------------------------------
# The six above never reach `main()`, and they build a fixture whose `sessions` table is
# keyed by `session_id`. The LIVE hermes database keys it by `id` — measured — so a guard
# that only satisfied the fixture would resolve no model, no window, and quietly never fire.
# The tests below run the real schema, the real payload shape, and the real process.
#
# These imports sit HERE, below the block transcribed from the plan, so that block stays
# byte-identical to it.
import contextlib          # noqa: E402
import io                  # noqa: E402
import json                # noqa: E402
import subprocess          # noqa: E402
import time                # noqa: E402
import types               # noqa: E402
import unittest.mock       # noqa: E402

from core.docs import doc_id_for  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: The guard as hermes runs it: a script, by absolute path, from a working directory that
#: has nothing to do with this repo. Every subprocess test below therefore also exercises
#: the sys.path bootstrap — without it `import core` raises before `main()` exists.
GUARD = os.path.join(REPO, "hosts", "hermes", "bigfile.py")


def a_live_shaped_db(messages, model="MiniMax-M2.7", session="s1") -> str:
    """A `state.db` with the schema the LIVE one has, measured on ~/.hermes/state.db:
    `sessions` keyed by `id` (there is no `session_id` column), `messages` with an
    autoincrement `id`, `token_count` NULL on every row, and a WAL journal."""
    path = os.path.join(tempfile.mkdtemp(), "state.db")
    c = sqlite3.connect(path)
    c.execute("pragma journal_mode=wal")
    c.execute("create table messages (id integer primary key autoincrement, "
              "session_id text not null, role text not null, content text, "
              "timestamp real not null, token_count integer, "
              "active integer not null default 1)")
    c.execute("create table sessions (id text primary key, model text)")
    c.execute("insert into sessions (id, model) values (?, ?)", (session, model))
    for i, (role, content, active) in enumerate(messages):
        c.execute("insert into messages (session_id, role, content, timestamp, active) "
                  "values (?, ?, ?, ?, ?)", (session, role, content, float(i), active))
    c.commit()
    c.close()

    return path


def a_file_of(size: int) -> str:
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as fh:
        fh.write("x" * size)

    return path


def a_read_payload(path: str, session: str = "s1") -> str:
    """The shape `agent/shell_hooks.py::_serialize_payload` writes — read out of the
    installed v0.20.1, keys and all. `tool_input` is the invoke-site kwarg `args` RENAMED,
    and `read_file`'s argument is `path`, not `file_path`."""
    return json.dumps({"hook_event_name": "pre_tool_call", "tool_name": "read_file",
                       "tool_input": {"path": path}, "session_id": session, "cwd": "/x",
                       "extra": {"task_id": "", "tool_call_id": "c1", "turn_id": "t1",
                                 "api_request_id": "r1", "middleware_trace": []}})


class Spy:
    """Stands in for `indexed_ids`, and COUNTS. A stub that only returned a value would let
    the round trip move to the common path without a single test noticing."""

    def __init__(self, result=None):
        self.calls = 0
        self.result = result

    def __call__(self, cfg):
        self.calls += 1

        return self.result


def run_main(payload: str, ids_spy, loader=None, window: int = 100_000):
    """(exit code, stdout) for `main()` on a fabricated payload, with NO network anywhere.

    `indexed_ids` is replaced wholesale, so no test here can reach Qdrant even if the
    ordering regresses — the spy's call count is what reports the regression instead.
    `SystemExit` is caught rather than allowed to end the run: exit 2 IS the block signal,
    so it is a result to assert, not a failure.
    """
    out = io.StringIO()
    cfg = types.SimpleNamespace(context_window=window)
    code = 0
    with unittest.mock.patch.object(adapter.core, "load", loader or (lambda: cfg)), \
         unittest.mock.patch.object(adapter, "indexed_ids", ids_spy), \
         unittest.mock.patch.object(sys, "stdin", io.StringIO(payload)), \
         contextlib.redirect_stdout(out):
        try:
            adapter.main()
        except SystemExit as exc:
            code = exc.code

    return code, out.getvalue()


def guard_env(db_path: str, **overrides) -> dict:
    """A hermetic environment for a guard subprocess.

    Nothing here may reach the operator's world: `QCTX_CONFIG` points at a file that will
    never exist (so the environment is the whole config), `QCTX_STATE_DIR` is a fresh temp
    directory (so the breaker never touches ~/.memories-plugin), `QCTX_HERMES_STATE_DB`
    points at the fixture (so ~/.hermes/state.db is never opened), and Qdrant points at a
    port nobody listens on — refused instantly, which is the degraded path these tests want.
    """
    state = tempfile.mkdtemp()
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG")}
    env.update({
        "QCTX_CONFIG": os.path.join(state, "config.json"),
        "QCTX_STATE_DIR": state,
        "QCTX_HERMES_STATE_DB": db_path,
        "QCTX_QDRANT_URL": "http://127.0.0.1:1",
        "QCTX_EMBED_URL": "http://127.0.0.1:1/v1/embeddings",
        "QCTX_EMBED_MODEL": "test-embed",
        "QCTX_MEMORY_COLLECTION": "t_mem",
        "QCTX_DOCS_COLLECTION": "t_tmp",
        "QCTX_LIBRARY_COLLECTION": "t_lib",
        "QCTX_CONTEXT_WINDOW": "100000",
    })
    env.update(overrides)

    return env


def run_guard(file_path: str, db_path: str, env=None, session: str = "s1"):
    """(stdout, returncode) for the REAL guard process, started the way hermes starts it.

    `cwd="/"` on purpose: hermes spawns the hook by absolute path from wherever it happens
    to be running, so a module that needed this repo to be the working directory would work
    in the test suite and nowhere else.
    """
    done = subprocess.run([sys.executable, GUARD], input=a_read_payload(file_path, session),
                          env=env if env is not None else guard_env(db_path), cwd="/",
                          capture_output=True, text=True, timeout=60)

    return done.stdout, done.returncode


#: Isolates the FLOOR: 79,000 used of a 100,000 window, plus 2,000, is 81,000 — past the
#: 80,000 the floor allows — while 2,000 stays under 40% of the 21,000 free.
FLOOR_ONLY = (316_000, 8_000)
#: Isolates the SHARE: 60,402 used leaves 39,598 free, and 17,100 is 43% of it, over the 40%
#: share — while 77,502 of 100,000 never reaches the floor.
SHARE_ONLY = (241_608, 68_400)


def a_session_using(chars: int, marker: str = "") -> str:
    """A database whose active messages estimate to `(chars + 3) // 4` tokens of context."""
    rows = [("user", "x" * chars, 1)]
    if marker:
        rows.append(("user", marker, 1))

    return a_live_shaped_db(rows)


class TestTheSchemaItQueriesIsTheLiveOne(unittest.TestCase):
    """`sessions` has an `id` column and no `session_id` one — measured against the live
    ~/.hermes/state.db, where the plan's query raises `OperationalError: no such column`.
    `_rows` swallows that, so the failure is SILENT: no model, window 0, guard asleep
    forever. Nothing in the plan's six tests could see it, because its fixture built the
    column the query asked for."""

    def test_the_model_is_resolved_from_the_live_sessions_table(self):
        db = a_live_shaped_db([("user", "x" * 400, 1)], model="MiniMax-M2.7")
        b = adapter.budget_from(db, "s1", lambda m: 1_000_000 if m == "MiniMax-M2.7" else 0)
        self.assertEqual(b.window, 1_000_000,
                         "the model never arrived, so the window resolved to 'unknown'")
        self.assertEqual(b.used, 100)

    def test_a_sessions_table_keyed_by_session_id_resolves_too(self):
        """The other half of the tolerance, and it has to bite on its own: this is the shape
        the plan's fixture above builds, and NOTHING in the plan's six tests exercises model
        resolution at all — their `window_of` ignores the model it is handed, so both query
        forms could be broken and all six would stay green."""
        db = a_state_db([("user", "x" * 400, 1)])
        b = adapter.budget_from(db, "s1", lambda m: 1_000_000 if m == "MiniMax-M2.7" else 0)
        self.assertEqual(b.window, 1_000_000)

    def test_a_session_that_is_not_there_fails_open(self):
        db = a_live_shaped_db([("user", "x" * 400, 1)])
        self.assertEqual(adapter.budget_from(db, "other", lambda m: 1_000_000).window, 0)


class TestItNeverWritesTheLiveDatabase(unittest.TestCase):
    """The plan's `mtime` check is weak — a connection that opened read-write and read
    nothing leaves the mtime alone too. These two probe the property itself."""

    def test_a_writer_holding_the_database_costs_a_timeout_and_not_a_hang(self):
        """hermes writes to state.db live. Another PROCESS takes the database exclusively;
        the guard must come back inside its own short timeout and fail OPEN, and the writer
        must be able to commit — a guard that queued behind it, or that made it queue behind
        us, would be an outage this plugin inflicted on itself before every file read."""
        db = a_live_shaped_db([("user", "x" * 400, 1)])
        holder = subprocess.Popen(
            [sys.executable, "-c",
             "import sqlite3, sys, time\n"
             "c = sqlite3.connect(sys.argv[1])\n"
             "c.execute('pragma locking_mode=exclusive')\n"
             "c.execute('begin immediate')\n"
             "c.execute(\"insert into messages (session_id, role, content, timestamp, "
             "active) values ('s1', 'user', 'held', 9.0, 1)\")\n"
             "print('locked', flush=True)\n"
             "time.sleep(4)\n"
             "c.commit()\n"
             "print('committed', flush=True)\n", db],
            stdout=subprocess.PIPE, text=True)
        self.addCleanup(holder.stdout.close)
        self.addCleanup(holder.wait)
        self.addCleanup(holder.kill)
        self.assertEqual(holder.stdout.readline().strip(), "locked")

        started = time.monotonic()
        budget = adapter.budget_from(db, "s1", lambda m: 1_000_000)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 2.0,
                        f"waited {elapsed:.3f}s on a database somebody else holds; the "
                        f"writer holds it for 4s and the guard runs before EVERY read")
        self.assertEqual(budget.window, 0, "a locked database must fail open")
        self.assertEqual(holder.stdout.readline().strip(), "committed",
                         "the writer could not finish — the guard took a lock it must not")

    def test_the_connection_it_opens_has_no_write_authority_at_all(self):
        """`mode=ro` and not a promise to only run SELECTs: the connection itself refuses.
        Probed with `pragma user_version`, which SQLite applies immediately — an UPDATE
        would be rolled back on close and would look identical under `mode=rw`, proving
        nothing."""
        db = a_live_shaped_db([("user", "x" * 400, 1)])
        self.assertEqual(adapter._rows(db, "pragma user_version = 42"), [],
                         "a write through the adapter's own opener must not go through")
        self.assertEqual(adapter._rows(db, "pragma user_version"), [(0,)],
                         "the write landed: the database was opened read-write")


class TestTheOrderThatProtectsTheCommonPath(unittest.TestCase):
    """Who is indexed costs a round trip, and this guard runs before EVERY read."""

    def test_an_allowed_read_never_asks_who_is_indexed(self):
        spy = Spy()
        code, out = run_main(a_read_payload(a_file_of(400)), spy,
                             window=100_000)
        self.assertEqual((code, out), (0, ""), "an allowed read must emit nothing at all")
        self.assertEqual(spy.calls, 0, "the common path paid for ids it never used")

    def test_a_block_pays_for_the_ids_exactly_once(self):
        chars, size = SHARE_ONLY
        db = a_session_using(chars)
        spy = Spy(set())
        with unittest.mock.patch.object(adapter, "state_db_path", lambda: db):
            code, out = run_main(a_read_payload(a_file_of(size)), spy)
        self.assertEqual(spy.calls, 1)
        self.assertEqual(code, adapter.BLOCK_EXIT_CODE)
        emitted = json.loads(out)
        self.assertEqual(emitted["decision"], "block")
        self.assertIn("index it with docs_index", emitted["reason"])

    def test_the_second_pass_is_what_upgrades_the_message(self):
        """Without the second call the model is told to index what is already indexed."""
        chars, size = SHARE_ONLY
        db = a_session_using(chars)
        path = a_file_of(size)
        spy = Spy({doc_id_for(path)})
        with unittest.mock.patch.object(adapter, "state_db_path", lambda: db):
            _, out = run_main(a_read_payload(path), spy)
        reason = json.loads(out)["reason"]
        self.assertIn("already indexed", reason)
        self.assertIn("docs_search", reason)

    def test_the_escape_marker_stops_the_block_before_the_round_trip(self):
        """--full is the user overruling the guard; there is nothing left to enrich."""
        chars, size = SHARE_ONLY
        db = a_session_using(chars, marker="leia --full")
        spy = Spy(set())
        with unittest.mock.patch.object(adapter, "state_db_path", lambda: db):
            code, out = run_main(a_read_payload(a_file_of(size)), spy)
        self.assertEqual((code, out), (0, ""))
        self.assertEqual(spy.calls, 0)

    def test_both_passes_of_decide_are_given_the_knobs(self):
        """The silent regression this exists for: dropping `floor_pct=`/`share_pct=` from
        either call leaves every other test green — the module defaults happen to match —
        and the knobs quietly stop working. So assert the ARGUMENTS, not the outcome."""
        chars, size = SHARE_ONLY
        db = a_session_using(chars)
        seen = []
        real = adapter.bigfile.decide

        def recording(path, budget, **kw):
            seen.append(kw)

            return real(path, budget, **kw)

        with unittest.mock.patch.object(adapter, "state_db_path", lambda: db), \
             unittest.mock.patch.object(adapter.bigfile, "decide", recording):
            run_main(a_read_payload(a_file_of(size)), Spy(set()))
        self.assertEqual(len(seen), 2, "the two-pass order is what this rides on")
        for call in seen:
            self.assertEqual(call.get("floor_pct"), adapter.FLOOR_PCT)
            self.assertEqual(call.get("share_pct"), adapter.SHARE_PCT)


class TestMainFailsOpen(unittest.TestCase):
    """Every error path emits nothing and exits 0: a guard that breaks gets out of the way."""

    def test_a_payload_that_is_not_json_emits_nothing(self):
        self.assertEqual(run_main("{not json", Spy(set())), (0, ""))

    def test_a_payload_with_no_session_id_allows_and_does_not_guess(self):
        chars, size = SHARE_ONLY
        db = a_session_using(chars)
        spy = Spy(set())
        payload = json.dumps({"tool_name": "read_file",
                              "tool_input": {"path": a_file_of(size)}, "session_id": ""})
        with unittest.mock.patch.object(adapter, "state_db_path", lambda: db):
            self.assertEqual(run_main(payload, spy), (0, ""))
        self.assertEqual(spy.calls, 0)

    def test_a_payload_with_no_path_allows(self):
        spy = Spy(set())
        payload = json.dumps({"tool_name": "read_file", "tool_input": {}, "session_id": "s1"})
        self.assertEqual(run_main(payload, spy), (0, ""))
        self.assertEqual(spy.calls, 0)

    def test_a_config_that_will_not_load_allows_the_read(self):
        """Ledger F3: `core.load()` raises on a malformed numeric field before a Config
        exists, so `window_for`'s tolerance never runs. A typo in a QCTX_* variable must not
        turn into a file nobody can read."""
        def explode():
            raise ValueError("invalid literal for int() with base 10: 'banana'")

        chars, size = SHARE_ONLY
        db = a_session_using(chars)
        spy = Spy(set())
        with unittest.mock.patch.object(adapter, "state_db_path", lambda: db):
            self.assertEqual(run_main(a_read_payload(a_file_of(size)), spy, loader=explode),
                             (0, ""))
        self.assertEqual(spy.calls, 0)

    def test_a_database_that_is_not_there_allows_the_read(self):
        spy = Spy(set())
        chars, size = SHARE_ONLY
        with unittest.mock.patch.object(adapter, "state_db_path",
                                        lambda: "/nonexistent/state.db"):
            self.assertEqual(run_main(a_read_payload(a_file_of(size)), spy), (0, ""))
        self.assertEqual(spy.calls, 0)


class TestTheBlockContractIsTheOneHermesHonours(unittest.TestCase):
    """Run as a process, from `/`, exactly as hermes runs it — which is also the only way
    the sys.path bootstrap and the import-time knobs are exercised at all."""

    def test_a_blocked_read_exits_2_and_says_why_on_stdout(self):
        chars, size = SHARE_ONLY
        out, code = run_guard(a_file_of(size), a_session_using(chars))
        self.assertEqual(code, adapter.BLOCK_EXIT_CODE,
                         "exit 2 is what blocks a pre_tool_call even with no stdout")
        emitted = json.loads(out)
        self.assertEqual(emitted["decision"], "block")
        self.assertIn("--full", emitted["reason"], "the way out must be in the message")

    def test_the_number_is_marked_approximate_because_it_is_an_estimate(self):
        """`messages` holds no system prompt, no tool definitions and no skills index, so
        this host UNDERSTATES. The message says so instead of pretending."""
        chars, size = SHARE_ONLY
        out, _ = run_guard(a_file_of(size), a_session_using(chars))
        self.assertIn("≈", json.loads(out)["reason"])

    def test_an_allowed_read_exits_0_and_says_nothing(self):
        chars, _ = SHARE_ONLY
        out, code = run_guard(a_file_of(400), a_session_using(chars))
        self.assertEqual((out, code), ("", 0))


class TestTheTwoThresholdsAreRealKnobs(unittest.TestCase):
    """Both are read at IMPORT time and handed to `decide`, and they must answer to the same
    variable names the claude-code hook answers to.

    Each fixture trips exactly ONE criterion, so loosening one knob can only be observed
    through the criterion it governs. A fixture that tripped both would let a dead knob look
    alive because the other criterion kept blocking.
    """

    def _decision(self, fixture, **env):
        chars, size = fixture
        db = a_session_using(chars)
        out, code = run_guard(a_file_of(size), db, env=guard_env(db, **env))
        self.assertIn(code, (0, adapter.BLOCK_EXIT_CODE), f"unexpected exit {code}")

        return "block" if out.strip() else "allow"

    def test_the_floor_knob_moves_the_decision(self):
        self.assertEqual(self._decision(FLOOR_ONLY), "block")
        self.assertEqual(self._decision(FLOOR_ONLY, QCTX_BIGFILE_FLOOR_PCT="0.001"), "allow")

    def test_the_share_knob_moves_the_decision(self):
        self.assertEqual(self._decision(SHARE_ONLY), "block")
        self.assertEqual(self._decision(SHARE_ONLY, QCTX_BIGFILE_SHARE_PCT="0.9"), "allow")

    def test_the_legacy_names_are_accepted_too(self):
        self.assertEqual(self._decision(FLOOR_ONLY, BIGFILE_FLOOR_PCT="0.001"), "allow")
        self.assertEqual(self._decision(SHARE_ONLY, BIGFILE_SHARE_PCT="0.9"), "allow")

    def test_a_malformed_knob_falls_back_instead_of_killing_the_guard(self):
        """These are read ABOVE main()'s catch-all — and on this host an import-time raise
        is worse than a traceback: the loader pre-execs this file and swallows the failure
        at debug level."""
        garbage = {"QCTX_BIGFILE_FLOOR_PCT": "20%", "QCTX_BIGFILE_SHARE_PCT": "banana"}
        self.assertEqual(self._decision(FLOOR_ONLY, **garbage), "block")
        self.assertEqual(self._decision(SHARE_ONLY, **garbage), "block")


class TestWhereTheDatabaseIs(unittest.TestCase):
    """The guard is a subprocess: it inherits an environment, not hermes' in-process state."""

    def test_hermes_home_decides_and_the_default_is_the_platform_one(self):
        home = tempfile.mkdtemp()
        with unittest.mock.patch.dict(os.environ, {"HERMES_HOME": home}, clear=False):
            os.environ.pop("QCTX_HERMES_STATE_DB", None)
            self.assertEqual(adapter.state_db_path(), os.path.join(home, "state.db"))

    def test_an_explicit_override_wins(self):
        with unittest.mock.patch.dict(os.environ, {"QCTX_HERMES_STATE_DB": "/tmp/x.db"}):
            self.assertEqual(adapter.state_db_path(), "/tmp/x.db")


# The fake Qdrant is IMPORTED from the claude-code adapter's tests rather than copied: it is
# one server, answering the same six round trips, and a second copy would be free to drift
# into agreeing with whichever host it was last edited for.
import http.server     # noqa: E402
import threading       # noqa: E402

from tests.test_bigfile_claude import SlowQdrant  # noqa: E402


class TestTheEnrichmentCannotBlowTheGuardDeadline(unittest.TestCase):
    """A per-call timeout cannot bound a SUM of calls, and `list_docs("all")` is six of them.

    Measured on the sibling host before the wall clock existed, against this very server at
    1.2 s per call: every per-call ceiling respected, 7.267 s spent against a 5 s budget.
    hermes allows a hook 60 s by default, which changes nothing — the deadline is there so a
    guard that runs before EVERY read cannot add latency the user feels, and the user's
    patience does not scale with the host's configuration.
    """

    def _serve(self, delay, doc_id):
        handler = type("H", (SlowQdrant,), {"delay": delay, "doc_id": doc_id})
        # A server that stays QUIET about the broken pipe it is guaranteed to see: the
        # abandoned worker's socket dies with the guard process, which is the deadline
        # working, not a fault.
        quiet = type("QuietServer", (http.server.ThreadingHTTPServer,),
                     {"handle_error": lambda self, request, addr: None})
        server = quiet(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)

        return f"http://127.0.0.1:{server.server_address[1]}"

    def _run_the_guard(self, delay):
        chars, size = SHARE_ONLY
        path = a_file_of(size)
        db = a_session_using(chars)
        url = self._serve(delay, doc_id_for(path))
        started = time.monotonic()
        out, code = run_guard(path, db, env=guard_env(db, QCTX_QDRANT_URL=url))

        return time.monotonic() - started, out, code

    def test_a_slow_but_healthy_qdrant_cannot_blow_the_deadline(self):
        elapsed, out, code = self._run_the_guard(delay=1.2)
        self.assertEqual(code, adapter.BLOCK_EXIT_CODE)
        self.assertLess(elapsed, 4.0,
                        f"took {elapsed:.3f}s; six unbounded 1.2s round trips would be ~7.3s")
        self.assertIn("index it with docs_index", json.loads(out)["reason"],
                      "abandoning the lookup costs the better message, never the block")

    def test_the_same_server_answered_fast_delivers_the_better_message(self):
        """The control, and it is what makes the test above mean anything: the same server
        that was too slow is UNDERSTOOD when it is quick, so the slow run degraded because of
        the clock and not because the fake spoke a language the client rejects."""
        elapsed, out, code = self._run_the_guard(delay=0.0)
        self.assertEqual(code, adapter.BLOCK_EXIT_CODE)
        self.assertLess(elapsed, 4.0, f"took {elapsed:.3f}s")
        reason = json.loads(out)["reason"]
        self.assertIn("already indexed", reason)
        self.assertIn("docs_search", reason)


#: The loader's own mechanism, replicated: `spec_from_file_location`, registered in
#: `sys.modules` BEFORE `exec_module`, with the repo root nowhere on `sys.path`. Printed as
#: JSON so the assertions below read the SAME run that produced the module.
_LOADER_REPLICA = """
import importlib.util, json, sys

# The guard on the guard, and it is what makes this test honest. Run from inside the repo
# root, Python's implicit `sys.path[0] = ""` makes `core` importable anyway and the replica
# goes green with the bootstrap deleted — a measured false negative. So prove `core` is
# UNREACHABLE first, and let the run report it.
visible_before = importlib.util.find_spec("core") is not None

name = "hermes_user_plugins.memories.bigfile"
spec = importlib.util.spec_from_file_location(name, sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module          # the loader registers FIRST, then execs
error = ""
try:
    spec.loader.exec_module(module)
except Exception as exc:            # what `logger.debug` would swallow
    error = f"{type(exc).__name__}: {exc}"

print(json.dumps({"core_visible_before": visible_before, "error": error,
                  "shell_left_behind": name in sys.modules,
                  "usable": hasattr(sys.modules[name], "main")}))
"""


class TestTheBootstrapSurvivesTheLoadersOwnPreExec(unittest.TestCase):
    """hermes pre-execs every sibling `*.py` BEFORE `__init__.py`, registering each in
    `sys.modules` first and swallowing the failure at `logger.debug`
    (`plugins/memory/__init__.py::_load_provider_from_dir`, :486-501 in the installed
    v0.20.1). A sibling that cannot find `core` on its own raises THERE, where nothing is
    listening, and leaves a shell behind that any later `from . import` hands back empty.

    The subprocess tests above already fail without the bootstrap, but they fail as a
    SCRIPT. This one fails the way the incident happened — and the proof the brief called
    the most important of the task cannot be the one the CI never runs.
    """

    def _replicate(self):
        """Always from a temp directory, never from the repo root: Python prepends the
        working directory to `sys.path` for `-c`, which makes `core` importable without the
        bootstrap and turns this into a false negative. Measured by the reviewer, and the
        `core_visible_before` assertion below is what stops it coming back."""
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG")}
        done = subprocess.run([sys.executable, "-c", _LOADER_REPLICA, GUARD],
                              cwd=tempfile.mkdtemp(), env=env, capture_output=True,
                              text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)

        return json.loads(done.stdout)

    def test_the_module_execs_standalone_where_core_is_not_importable(self):
        seen = self._replicate()
        self.assertFalse(seen["core_visible_before"],
                         "`core` was already importable, so this run could not have proven "
                         "anything — the replica must not be started from the repo root")
        self.assertEqual(seen["error"], "",
                         "the pre-exec raised; hermes would swallow this at debug level and "
                         "the provider would go down with it")
        self.assertTrue(seen["usable"], "the module exec'd but has no main() to call")

    def test_the_shell_the_loader_leaves_behind_is_a_working_module(self):
        """`sys.modules` keeps whatever was registered even when the exec failed — the loader
        pops only the package's own entry, never a sibling's. So "registered" proves nothing
        on its own; what matters is that the thing left behind is the real module."""
        seen = self._replicate()
        self.assertTrue(seen["shell_left_behind"])
        self.assertTrue(seen["usable"])


if __name__ == "__main__":
    unittest.main()
