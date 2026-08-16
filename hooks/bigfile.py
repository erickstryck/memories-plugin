#!/usr/bin/env python3
"""PreToolUse guard for claude-code: refuse a file read that would cost too much context.

WHY THE TAIL AND NOT THE FILE. The transcript of the session this was written in was
15.8 MB. This hook runs before EVERY file read, so reading the whole transcript would cost
more than the guard saves. Only the last few KB are needed: the most recent `usage` block
and the most recent real user turn.

WHAT COUNTS AS A REAL USER TURN. `role=user` alone is not it — tool results and injected
skill text carry that role too. Measured on a live transcript, the discriminator is
`userType=external` AND `entrypoint=cli` AND no `toolUseResult`. Without the filter, the
"last user message" is skill boilerplate, and any tool output containing the escape marker
would unlock the guard.

WHERE THE PAYLOAD KEYS COME FROM, HONESTLY. They were READ OUT OF THE `claude` BINARY
v2.1.233 (`~/.local/share/claude/versions/2.1.233`, via `strings`) — they were NOT observed
on a live payload; the headless probe that would have produced one was refused. What the
binary shows is `executePreToolHooks` building its payload as
`{...session fields, hook_event_name: "PreToolUse", tool_name, tool_input, tool_use_id}`,
where the session fields are `{session_id, transcript_path, cwd, prompt_id,
permission_mode, agent_id, agent_type, effort}`. Evidence from the construction site is
stronger than one sighting — it shows the key always being written rather than one
occurrence of it — but it is evidence about ONE version, so every key is read defensively
here and anything missing means ALLOW.

The block contract comes from the same binary (the zod schema of `hookSpecificOutput`):
`{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
"permissionDecisionReason": "<text the model reads>"}}`.

NO DERIVATION OF THE TRANSCRIPT FROM `session_id`. An earlier design derived
`~/.claude/projects/<slug>/<session_id>.jsonl` when `transcript_path` was absent. The
measurement above kills that branch: the key is unconditional, and the derivation is
exactly what the host already does internally when the id is not the current session's.
Code no test can honestly exercise rots and lies about what was verified. If the key is
missing or the file will not open, this hook FAILS OPEN — emits nothing, exits 0 — which
is the same thing `window=0` already produces in the core.

FAIL OPEN IS THE WHOLE RULE. A guard that breaks must get out of the way, never become a
cage: every error path here emits nothing and exits 0, so the read proceeds.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core  # noqa: E402
from core import bigfile, windows  # noqa: E402
from core.breaker import Breaker  # noqa: E402

#: Enough for the last usage block and the last user turn, never the whole file.
TAIL_BYTES = 256 * 1024


def _env(name: str, legacy: str, default: str) -> str:
    return os.environ.get(name) or os.environ.get(legacy) or default


def _env_num(name: str, legacy: str, default: str, kind=float):
    """Reads a number from the environment WITHOUT killing the process if it is malformed.

    Same shape and same reason as `recall.py::env_num`: this runs at IMPORT, above main()'s
    catch-all, so a `QCTX_BIGFILE_FLOOR_PCT=20%` would take the hook down before any guard
    could turn it into a fail-open. No clamp on either of these two: they are FRACTIONS,
    and a 0 means "this criterion never fires", which is a coherent thing to ask for —
    unlike the recall ceilings, where a 0 makes the hook claim an empty archive.
    """
    raw = _env(name, legacy, default)
    try:
        return kind(raw)
    except (TypeError, ValueError):
        return kind(default)


#: The two thresholds of `core.bigfile.decide`, read HERE and not there: the core stays
#: pure and environment-free, and the adapter is what knows it is running on a host. The
#: names are the ones hermes must use too — a deployer who tuned the guard on one host
#: expects the same variable to move the same number on the other.
FLOOR_PCT = _env_num("QCTX_BIGFILE_FLOOR_PCT", "BIGFILE_FLOOR_PCT", "0.20", float)
SHARE_PCT = _env_num("QCTX_BIGFILE_SHARE_PCT", "BIGFILE_SHARE_PCT", "0.40", float)

#: Where the breaker below keeps its one timestamp. Same variable `recall.py` reads, so a
#: deployer who moved the state moved all of it.
STATE_DIR = Path(os.environ.get("QCTX_STATE_DIR") or (Path.home() / ".memories-plugin" / "state"))

#: Seconds allowed per Qdrant call on the RARE enrichment path. Sized against the
#: `timeout: 5` in hooks.json, which is the deadline the whole invocation must fit inside.
QDRANT_TIMEOUT_S = 1.5

#: Above this, the inventory lookup is treated as a failure even though it answered — the
#: enrichment is a nicety and it may not eat the deadline that protects the read.
#: Measured on this deployment: build 0.000s, `list_docs("all")` 0.10-0.12s over 297
#: chunks, four consecutive runs.
SLOW_S = 1.5

#: How long a failed (or too slow) inventory lookup keeps the enrichment switched off.
#: Not an env knob on purpose: a knob here would have to be mirrored name-for-name on
#: hermes, and nothing about the message quality is worth a second vocabulary.
BREAKER_SECONDS = 300.0


def _tail_objects(path: str) -> list:
    """Parsed JSON objects from the tail, oldest first. Never raises."""
    try:
        size = os.path.getsize(path)
        start = max(0, size - TAIL_BYTES)
        with open(path, "rb") as fh:
            fh.seek(start)
            raw = fh.read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = raw.splitlines()
    if start:
        # Only a seek that actually moved can land mid-object; that leading fragment is
        # not a line. Dropping it UNCONDITIONALLY would throw away the first real entry of
        # every transcript smaller than TAIL_BYTES — including a session whose only
        # `usage` block is its first line, which is precisely what the budget is read from.
        lines = lines[1:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue

    return out


def budget_from(path: str, window_of) -> bigfile.Budget:
    """The context budget, measured. window=0 when anything was unavailable — fail open."""
    used, model = 0, ""
    for obj in _tail_objects(path):
        m = obj.get("message") or {}
        u = m.get("usage")
        if u:
            used = (int(u.get("input_tokens") or 0)
                    + int(u.get("cache_creation_input_tokens") or 0)
                    + int(u.get("cache_read_input_tokens") or 0))
            model = m.get("model") or model
    if not used:
        return bigfile.Budget(window=0, used=0, exact=True)

    return bigfile.Budget(window=int(window_of(model) or 0), used=used, exact=True)


def escape_requested(path: str) -> bool:
    """Did the LAST real user turn carry the marker?

    Scoped to one turn on purpose: nothing to clear, and no way to leave the guard off by
    forgetting — which is what ruled out an environment variable.
    """
    for obj in reversed(_tail_objects(path)):
        if obj.get("toolUseResult") is not None:
            continue
        if obj.get("userType") != "external" or obj.get("entrypoint") != "cli":
            continue
        m = obj.get("message") or {}
        if m.get("role") != "user":
            continue
        c = m.get("content")
        text = c if isinstance(c, str) else " ".join(
            b.get("text", "") for b in c if isinstance(b, dict)) if isinstance(c, list) else ""

        return bigfile.ESCAPE_MARKER in text

    return False


def indexed_ids(cfg) -> set | None:
    """Doc ids already in the archive, or None when finding out was not worth it.

    CALLED ONLY AFTER a `decide(..., indexed_ids=None)` has already said it would block.
    Knowing this costs a round trip to Qdrant and this hook runs before EVERY file read;
    the common case is "small file, allow", and taxing it with a round trip it never
    needed is the one thing the ordering here exists to prevent.

    None is a complete answer, not a failure: it only costs the better half of the
    message ("already indexed, search it" instead of "index it"). A worse message beats a
    hook that blew its deadline, so anything that raises, and anything slower than
    `SLOW_S`, degrades to None AND arms the breaker so the next blocked read does not pay
    to rediscover it.
    """
    breaker = Breaker(STATE_DIR / "bigfile-docs-breaker", BREAKER_SECONDS)
    if breaker.is_open() is not None:
        return None
    started = time.monotonic()
    try:
        index = core.DocIndex(core.build_qdrant(cfg, timeout=QDRANT_TIMEOUT_S),
                              core.build_embedder(cfg), core.build_reranker(cfg),
                              cfg.require_docs_collection(), cfg.require_library_collection(),
                              cfg.vector_size)
        # `doc_id` is the key `list_docs` publishes, confirmed against a live listing
        # rather than assumed: {'doc_id', 'scope', 'chunks', 'path', 'mode', 'indexed_at',
        # 'expires_at_ts', 'src_mtime', 'src_size', 'src_digest', 'ttl_seconds'}.
        ids = {d["doc_id"] for d in index.list_docs(scope="all")}
    except BaseException:      # noqa: BLE001 — a lost nicety may never cost the read
        breaker.arm()

        return None
    if time.monotonic() - started > SLOW_S:
        breaker.arm()

        return ids
    breaker.clear()

    return ids


def deny(reason: str) -> None:
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }}))


def _run() -> None:
    data = json.load(sys.stdin)
    path = (data.get("tool_input") or {}).get("file_path") or ""
    transcript = data.get("transcript_path") or ""
    if not path or not transcript:
        return

    # `core.load()` is not tolerant of a malformed numeric field — `QCTX_CONTEXT_WINDOW=x`
    # raises from inside it, before a Config exists — so a typo in an env var must not
    # become a file nobody can read. main()'s catch-all turns it into an allow.
    cfg = core.load()
    budget = budget_from(transcript, lambda model: windows.window_for(model, cfg))

    # PASS ONE: no `indexed_ids`, no network. This is the path every read takes.
    verdict = bigfile.decide(path, budget, floor_pct=FLOOR_PCT, share_pct=SHARE_PCT)
    if not verdict.block:
        return

    # From here down we are on the rare path, and only here may it cost anything.
    if escape_requested(transcript):
        return

    # PASS TWO: same decision, better message. The ids are fetched only now, and the
    # verdict is re-read rather than assumed — `decide` owns whether it blocks, always.
    verdict = bigfile.decide(path, budget, indexed_ids=indexed_ids(cfg),
                             floor_pct=FLOOR_PCT, share_pct=SHARE_PCT)
    if verdict.block:
        deny(verdict.reason)


def main() -> None:
    """The armoured entry point: on ANY failure, emit nothing and exit 0.

    `BaseException` and not a list of types, for the reason `recall.py` learned the hard
    way: a list has to be updated in every consumer when a new error appears, and
    forgetting does not fail loudly — it fails as a traceback where the contract asked for
    silence. Here the stake is higher than a missing memory block: a stray stdout line or
    a non-zero exit on this hook is a file the user cannot read.
    """
    try:
        _run()
    except BaseException:      # noqa: BLE001 — see docstring
        pass


if __name__ == "__main__":
    main()
