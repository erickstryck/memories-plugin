#!/usr/bin/env python3
"""pre_tool_call guard for hermes: refuse a file read that would cost too much context.

WHY THE sys.path BOOTSTRAP. hermes' loader pre-execs every sibling `*.py` in the provider
directory BEFORE `__init__.py`, registering each in `sys.modules` first and swallowing any
failure at `logger.debug` (`plugins/memory/__init__.py::_load_provider_from_dir`, measured
at :486-501 in the installed v0.20.1). Without finding `core` on its own, this module
raises during that pre-exec, leaves a broken shell in `sys.modules`, and the whole provider
then fails to load — no recall, no checkpoint, no tools, one debug line. That is exactly
how this plugin once shipped broken, and it is why the three `dirname` calls below are the
same three `hosts/hermes/tools.py` carries. `realpath` and not `abspath` for the reason the
package docstring gives: the plugin is installed as a symlink.

WHY AN ESTIMATE. `messages.token_count` is NULL on every row — measured, 59 of 59 rows in
the live `~/.hermes/state.db` — and `session_model_usage` is CUMULATIVE across API calls,
not the current context. So the used figure is summed from message bodies with hermes' own
ratio (`agent/context_breakdown.py::_chars_to_tokens`, `(len+3)//4`). It UNDERSTATES, and
knowingly: `messages.content` holds no system prompt, no tool definitions and no skills
index, and the sum here does not reach the sibling columns (`reasoning`, `tool_calls`,
`api_content`) that also travel to the model. The guard therefore fires LATER here than on
claude-code, and the message marks the number approximate (`Budget.exact=False`) instead of
pretending. That asymmetry is the deliberate one; the DECISION on identical inputs is not
allowed to differ, which is what the host-equivalence test holds.

WHY THE DATABASE IS OPENED READ-ONLY. hermes writes to `state.db` LIVE, and this runs
before every file read. A hook that took a write lock — or that waited minutes for one —
would be a self-inflicted outage caused by the guard itself. `mode=ro` plus a short
`timeout` bounds both: we never ask for write authority, and a database somebody else has
locked exclusively costs us `SQLITE_TIMEOUT_S` and then a fail-open, never a hang. (Honest
footnote: on a WAL database SQLite may still CREATE the `-shm`/`-wal` sidecars from a
read-only connection — measured. What `mode=ro` guarantees is that no statement of ours can
modify the database, and that is the property hermes' data needs.)

WHERE THE READING IS BOUNDED, AND IT IS NOT WHERE THE SIBLING BOUNDS IT. `hooks/bigfile.py`
reads at most `TAIL_BYTES` (256 KB) of a transcript that was measured at 15.8 MB — an explicit
ceiling in the adapter. This one puts NO `LIMIT` on `select content from messages … active=1`;
it leans on hermes flipping `active` off when it compacts, so the rows it gets back are already
the size of the window. Stress-measured at 20,000 active rows of 5,000 chars (100 MB) and still
back in ~40 ms, so this is not a performance note — it is a structural difference resting on an
invariant this file does not verify, and the equivalence work needs to know the two hosts reach
"bounded read" by different means.

WHAT THE PAYLOAD ACTUALLY LOOKS LIKE, MEASURED. `agent/shell_hooks.py::_serialize_payload`
(v0.20.1, installed) writes `{"hook_event_name", "tool_name", "tool_input", "session_id",
"cwd", "extra"}` — `tool_input` is the invoke-site kwarg `args` RENAMED, so a hook that
reads `args` from stdin finds nothing. The file-read tool is `read_file` and its argument is
`path` (`tools/file_tools.py::READ_FILE_SCHEMA`), not `file_path`. Which tools this fires
for is the host's job through the `matcher` field of the `hooks:` block, exactly as
`matcher: "Read"` does in `hooks/hooks.json` on claude-code — so this file does not
second-guess `tool_name`, and installing it without a matcher is an installation defect.

BLOCK CONTRACT. `{"decision": "block", "reason": …}` on stdout AND exit 2: hermes honours
either (`_normalise_pre_tool_call` reads the Claude-Code shape; exit 2 blocks a
`pre_tool_call` even with no stdout at all). Both together, so a truncated stdout still
blocks with the reason and an unreadable exit code still blocks.

FAIL OPEN IS THE WHOLE RULE. Any error: nothing on stdout, exit 0, the read proceeds. A
guard that breaks must get out of the way, never become a cage.
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import core  # noqa: E402
from core import bigfile, windows  # noqa: E402
from core.knobs import env_num  # noqa: E402

#: The exit code `agent/shell_hooks.py::BLOCK_EXIT_CODE` honours on `pre_tool_call`.
BLOCK_EXIT_CODE = 2


#: The two thresholds of `core.bigfile.decide`, read HERE and not there: the core stays pure
#: and environment-free, and the adapter is what knows it is running on a host. The names are
#: byte-for-byte the ones `hooks/bigfile.py` reads — a deployer who tuned the guard on one
#: host expects the same variable to move the same number on the other, and the knob-parity
#: scan in the test suite is what keeps that true.
FLOOR_PCT = env_num("QCTX_BIGFILE_FLOOR_PCT", "BIGFILE_FLOOR_PCT", "0.20", float)
SHARE_PCT = env_num("QCTX_BIGFILE_SHARE_PCT", "BIGFILE_SHARE_PCT", "0.40", float)

#: What ONE `read_file` can put in the context, measured in the installed v0.20.1:
#: `_DEFAULT_MAX_READ_CHARS = 100_000` (`tools/file_tools.py:65`, applied through
#: `_get_max_read_chars`) and a line ceiling of 2000 that is both the schema's default and
#: its maximum. Two ceilings, and the price takes the smaller of what they allow and what
#: the file holds — pricing the whole file made a limited read cost what it never loads,
#: which is a block based on a wrong number, the one failure this guard must not produce.
#:
#: `file_read_max_chars` in config.yaml can RAISE the char budget. An operator who raises it
#: makes this number too small, which under-prices, which lets a read through — the safe
#: direction, and the one this design already accepts everywhere else.
READ_CHAR_CEILING = 100_000
DEFAULT_READ_LINES = 2000


def _read_lines(tool_input: dict) -> int:
    """How many lines this particular request can pull.

    A malformed `limit` falls back to the default rather than raising: `main()` would turn a
    raise into an allow anyway, but an allow decided by a typo is not a decision.
    """
    try:
        limit = int(tool_input.get("limit") or 0)
    except (TypeError, ValueError):
        return DEFAULT_READ_LINES

    return limit if limit > 0 else DEFAULT_READ_LINES


#: Short on purpose: hermes writes to this database live, and this runs before every read.
#: It is the ceiling on how long a locked database may cost us before we fail open.
SQLITE_TIMEOUT_S = 0.5

#: `sessions` keys the session by `id` — measured against the live `~/.hermes/state.db`,
#: where `select … where session_id=?` raises `OperationalError: no such column: session_id`.
#: The second form is the one this task's plan specified and the one its verbatim test
#: fixture builds; both are tried, cheapest-and-real first, because a query that only works
#: against the fixture would have left the guard permanently window-less in production —
#: `_rows` swallows the error, the model comes back empty, `window_for` resolves 0, and 0
#: allows. A guard that never fires and never says why is the failure this ordering removes.
_MODEL_QUERIES = ("select model from sessions where id=? limit 1",
                  "select model from sessions where session_id=? limit 1")


def state_db_path() -> str:
    """Where hermes keeps `state.db`.

    `QCTX_HERMES_STATE_DB` wins when set (the tests point it at a fixture; an operator with a
    relocated database gets the same escape hatch). Otherwise the resolution hermes itself
    uses for subprocesses: `HERMES_HOME` from the environment, else `~/.hermes`
    (`hermes_constants.py::get_hermes_home`, and `mcp_serve.py:490-492` for the same fallback
    written out). A per-profile home installed as a CONTEXT-LOCAL override is invisible to a
    subprocess; the guard then reads the default home, finds no rows for the session id, and
    fails open — the correct degradation, not a silent wrong answer.
    """
    explicit = os.environ.get("QCTX_HERMES_STATE_DB")
    if explicit:
        return explicit
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")

    return str(Path(home) / "state.db")


def _rows(db_path: str, sql: str, args=()) -> list:
    """Read-only query that never raises and never waits long.

    Every failure — file missing, database locked, column absent, database corrupt — lands
    here as an empty list, which every caller above turns into "we could not learn this",
    which `decide` turns into an allow.
    """
    try:
        uri = f"file:{db_path}?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=SQLITE_TIMEOUT_S)
        try:
            return list(con.execute(sql, args))
        finally:
            con.close()
    except Exception:      # noqa: BLE001 — fail open, always
        return []


def budget_from(db_path: str, session_id: str, window_of) -> bigfile.Budget:
    """The context budget, estimated. `window=0` when anything was unavailable — fail open.

    `active=1` is what makes this the CURRENT context rather than the session's history:
    hermes flips the flag off when it compacts, and a compacted message is no longer being
    sent to the model.
    """
    rows = _rows(db_path,
                 "select content from messages where session_id=? and active=1", (session_id,))
    if not rows:
        return bigfile.Budget(window=0, used=0, exact=False)
    used = sum((len(r[0] or "") + 3) // 4 for r in rows)
    model = ""
    for sql in _MODEL_QUERIES:
        got = _rows(db_path, sql, (session_id,))
        if got:
            model = got[0][0] or ""
            break

    return bigfile.Budget(window=int(window_of(model) or 0), used=used, exact=False)


def escape_requested(db_path: str, session_id: str) -> bool:
    """Did the LAST user message carry the marker?

    Scoped to one turn on purpose: nothing to clear, and no way to leave the guard off by
    forgetting — which is what ruled out an environment variable.

    No `userType`/`entrypoint` filter is needed here, and that is measured rather than
    assumed: on this host tool results are stored with `role='tool'` (21 of them in the live
    database) and every one of the 11 `role='user'` rows is a human turn — the recall block
    this plugin injects does not become a message row. The claude-code adapter needs that
    filter because THERE tool results and injected skill text both carry `role=user`.

    `rowid` breaks a timestamp tie deterministically. It exists on both the live table (where
    `id INTEGER PRIMARY KEY AUTOINCREMENT` is its alias) and on any fixture table, so the
    ordering cannot fall back to "whatever the file order happens to be" on either.
    """
    rows = _rows(db_path,
                 "select content from messages where session_id=? and role='user' and "
                 "active=1 order by timestamp desc, rowid desc limit 1", (session_id,))

    return bool(rows) and bigfile.ESCAPE_MARKER in (rows[0][0] or "")


def _run() -> str:
    """The reason to block with, or "" to allow. Prints nothing: emission belongs to main()."""
    data = json.load(sys.stdin)
    tool_input = data.get("tool_input") or {}
    path = tool_input.get("path") or ""
    session_id = data.get("session_id") or ""
    db_path = state_db_path()
    if not path or not session_id:
        return ""

    # `core.load()` is not tolerant of a malformed numeric field — `QCTX_CONTEXT_WINDOW=x`
    # raises from inside it, before a Config exists — so a typo in an env var must not become
    # a file nobody can read. main()'s catch-all turns it into an allow.
    cfg = core.load()
    budget = budget_from(db_path, session_id, lambda model: windows.window_for(model, cfg))

    # PASS ONE: no `indexed_ids`, no network. This is the path every read takes.
    verdict = bigfile.decide(path, budget, floor_pct=FLOOR_PCT, share_pct=SHARE_PCT,
                             read_lines=_read_lines(tool_input),
                             read_bytes=READ_CHAR_CEILING)
    if not verdict.block:
        return ""

    # From here down we are on the rare path, and only here may it cost anything.
    if escape_requested(db_path, session_id):
        return ""

    # PASS TWO: same decision, better message. The ids are fetched only now, and the verdict
    # is re-read rather than assumed — `decide` owns whether it blocks, always.
    #
    # The import is HERE and not at the top, the way `core.bigfile` imports `doc_id_for`:
    # `core.inventory` is the only thing this guard has that reaches the network, and the
    # common path — every allowed read — must never so much as load it.
    from core.inventory import indexed_ids

    verdict = bigfile.decide(path, budget, indexed_ids=indexed_ids(cfg),
                             floor_pct=FLOOR_PCT, share_pct=SHARE_PCT,
                             read_lines=_read_lines(tool_input),
                             read_bytes=READ_CHAR_CEILING)

    return verdict.reason if verdict.block else ""


def main() -> None:
    """The armoured entry point: on ANY failure, emit nothing and exit 0.

    `BaseException` and not a list of types, for the reason `recall.py` learned the hard way:
    a list has to be updated in every consumer when a new error appears, and forgetting does
    not fail loudly — it fails as a traceback where the contract asked for silence.

    The exit code is decided OUTSIDE the try, and that is not style: `sys.exit` raises
    `SystemExit`, which is a `BaseException`, so an exit-2 raised inside would be swallowed by
    the very catch-all that guarantees fail-open — the block would be silently downgraded to
    an allow. And if the reason cannot be written, we do NOT exit 2 either: hermes would then
    block with a generic default, and a block the user cannot act on is worse than the read.
    """
    try:
        reason = _run()
    except BaseException:      # noqa: BLE001 — see docstring
        return
    if not reason:
        return
    try:
        print(json.dumps({"decision": "block", "reason": reason}, ensure_ascii=False))
    except BaseException:      # noqa: BLE001 — a block nobody can read is not a block
        return
    sys.exit(BLOCK_EXIT_CODE)


if __name__ == "__main__":
    main()
