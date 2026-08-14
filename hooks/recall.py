#!/usr/bin/env python3
"""AUTOMATIC RECALL hook: searches memory before the prompt reaches the model.

A host adapter. All the retrieval lives in `core`; what stays here are the three
things that belong to the host: reading the hook payload, deciding what to inject
within a context budget, and formatting the block.

The point of the hook is that memory arrives WITHOUT depending on the model deciding
to search. Without it, reading is left to the agent's discipline — and reading is
precisely the direction that gets forgotten, because nothing fails visibly when it
does not happen.

IT FAILS SILENTLY FOR THE USER, NEVER FOR THE MODEL. If the search does not run, the
prompt goes through as usual (the user is not penalized), but the model receives an
EXPLICIT unavailability warning. Without that warning, an absence of results is
indistinguishable from "there is no precedent", and then the model asserts something
is unprecedented when nobody consulted the archive — the exact failure mode this hook
exists to prevent.

Configuration (all optional; QCTX_* are the canonical names, RECALL_* accepted for
compatibility with the previous version):
    QCTX_RECALL_DISABLED       "1" turns it off
    QCTX_RECALL_STRICT_FLOOR   dense cut when alone                (0.58)
    QCTX_RECALL_DENSE_FLOOR    dense floor with a cross-encoder    (0.45)
    QCTX_RECALL_MIN_SCORE      cross-encoder cutoff                (0.10)
    QCTX_RECALL_TOP_K          hits per angle                      (8 / 20 with CE)
    QCTX_RECALL_MAX_MEMORIES   memories injected at a time         (6)
    QCTX_RECALL_MAX_CHARS      total budget                        (14000)
    QCTX_RECALL_MAX_PER_MEM    ceiling per memory                  (4500)
    QCTX_RECALL_BREAKER        breaker cooldown in seconds         (300)
    QCTX_STATE_DIR             where to keep state and the log
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core  # noqa: E402
from core import query  # noqa: E402
from core import session_state as st  # noqa: E402
from core.blocks import Budget, empty_block, recall_block, split_by_budget, unavailable_block  # noqa: E402
from core.breaker import Breaker  # noqa: E402


def env(name: str, legacy: str, default: str) -> str:
    return os.environ.get(name) or os.environ.get(legacy) or default


def env_num(name: str, legacy: str, default: str, kind=float):
    """Reads a number from the environment WITHOUT killing the process if it is malformed.

    This is read at module load, i.e. BEFORE `main`'s catch-all — a
    `QCTX_RECALL_MAX_CHARS=14k` blew up before any of our code ran, and the user lost
    recall while getting a traceback instead of the unavailability warning. An invalid
    value falls back to the default and is recorded in the log.
    """
    raw = env(name, legacy, default)
    try:
        return kind(raw)
    except (TypeError, ValueError):
        _pending_notes.append(f"{name}={raw!r} is not a number — using {default}")

        return kind(default)


#: Warnings collected before the log exists (the log depends on STATE_DIR, which
#: depends on env). Flushed on the first log write.
_pending_notes: list[str] = []


STATE_DIR = Path(os.environ.get("QCTX_STATE_DIR") or (Path.home() / ".memories-plugin" / "state"))
LOG = STATE_DIR / "recall.log"
LOG_MAX_BYTES = 256 * 1024

STRICT_FLOOR = env_num("QCTX_RECALL_STRICT_FLOOR", "RECALL_MIN_SCORE", "0.58")
DENSE_FLOOR = env_num("QCTX_RECALL_DENSE_FLOOR", "RECALL_DENSE_FLOOR", "0.45")
MIN_SCORE = env_num("QCTX_RECALL_MIN_SCORE", "RECALL_RERANK_MIN_SCORE", "0.10")
MAX_MEMORIES = env_num("QCTX_RECALL_MAX_MEMORIES", "RECALL_MAX_MEMORIES", "6", int)
MAX_CHARS = env_num("QCTX_RECALL_MAX_CHARS", "RECALL_MAX_CHARS", "14000", int)
MAX_PER_MEM = env_num("QCTX_RECALL_MAX_PER_MEM", "RECALL_MAX_PER_MEM", "4500", int)
BREAKER_SECONDS = env_num("QCTX_RECALL_BREAKER", "RECALL_RERANK_BREAKER", "300")

#: Total seconds allowed for ALL Qdrant calls in one invocation, divided among them.
#: A per-call timeout multiplies by the number of angles; a total does not, which is
#: what keeps the worst case inside the host deadline in hooks.json.
QDRANT_BUDGET = env_num("QCTX_RECALL_QDRANT_BUDGET", "RECALL_QDRANT_BUDGET", "5.0")

#: What fits in this host's context window — read from our own environment and handed
#: down to core.blocks, which does not know what host it is running in.
BUDGET = Budget(max_memories=MAX_MEMORIES, max_chars=MAX_CHARS,
               max_per_mem=MAX_PER_MEM, reinject_after=st.REINJECT_AFTER)


def log(msg: str) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        while _pending_notes:
            _write_log(f"config: {_pending_notes.pop(0)}")
        if LOG.exists() and LOG.stat().st_size > LOG_MAX_BYTES:
            LOG.write_text(LOG.read_text(errors="replace")[-LOG_MAX_BYTES // 2:])
        _write_log(msg)
    except Exception:
        pass


def _write_log(msg: str) -> None:
    with LOG.open("a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def extract_prompt(data: dict) -> str:
    """The field name varies by host version; accepts the known candidates."""
    for key in ("prompt", "user_prompt", "userPrompt", "message", "current_prompt", "text"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return ""


def emit(context: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }))


def main() -> None:
    """The armoured entry point.

    The unavailability warning to the model is a property of the OUTERMOST frame, not of
    a list of types to catch. A list of types is fragile by construction: it has to be
    updated in every consumer when a new error appears, and forgetting does not produce
    a compile error — it produces a traceback for the user and silence for the model,
    which is the exact inverse of the contract. This already happened: `HttpError` was
    not on the list, and it is the most common failure.
    """
    try:
        _run()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 — see docstring
        try:
            log(f"unexpected failure ({type(exc).__name__}: {exc})")
            emit(unavailable_block("the hook", type(exc).__name__))
        except Exception:
            pass  # if even that fails, silence is the only path left


def _run() -> None:
    if os.environ.get("QCTX_RECALL_DISABLED") == "1" or os.environ.get("RECALL_DISABLED") == "1":
        return

    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    prompt = extract_prompt(data)
    if not prompt:
        # An EMPTY payload is a host that had nothing to say; a payload with keys but no
        # recognized prompt field is a host whose field name we do not know, and that
        # silently costs every prompt of every session. Only the second is worth alarming
        # about, and it has to alarm — otherwise a renamed field looks exactly like an
        # archive with no precedent.
        log(f"no prompt in the payload; keys={sorted(data.keys())}")
        if data:
            emit(unavailable_block("reading the prompt",
                                   f"no known prompt field in {sorted(data.keys())}"))
        return

    reason = query.skip_reason(prompt)
    if reason:
        log(f"skip ({reason}): {prompt[:60]!r}")
        return

    angles = query.angles(prompt)
    try:
        cfg = core.load()
        # The budget has to fit the hook timeout in hooks.json, and the arithmetic is not
        # one call per dependency: `recall` issues one vector search PER ANGLE, so the
        # Qdrant timeout is paid up to len(angles)+1 times (the existence check, then one
        # search each). The previous comment claimed 8+5+6=19s, true only for a single
        # angle; measured, 2 angles cost 24s and 3 cost 29s against a 20s deadline. Under a
        # slow-but-alive Qdrant — the common shared-infrastructure failure, and the exact
        # case the breaker exists for — the host killed the process mid-search and the
        # model got NOTHING, not even the unavailability warning. A SIGKILL is not an
        # exception, so the catch-all in main() cannot help.
        #
        # So the budget is divided, not repeated: the whole Qdrant allowance is split
        # across the calls that will actually be made.
        qdrant_calls = len(angles) + 1
        store = core.build_memory(cfg, timeouts={"embed": 8.0,
                                                 "qdrant": QDRANT_BUDGET / qdrant_calls,
                                                 "rerank": 6.0})
    except core.ConfigError as exc:
        # NOT silent. On this path the hook does nothing at all, for every prompt of every
        # session, while the memory skill tells the model a hook already guarantees the
        # floor. Configuration commonly comes from the environment, so a host started
        # without it — a desktop launcher, a systemd unit, a different shell — loses
        # long-term memory entirely. Of every degradation here, this is the one with the
        # largest blast radius, and it was the one that said nothing.
        log(f"incomplete config ({exc}) — no recall on this prompt")
        emit(unavailable_block("configuration", str(exc)))
        return

    # The breaker decides whether the cross-encoder takes part in this invocation.
    # Turning it off here, rather than inside the core, is what makes `recall` apply the
    # strict cut on its own: without the second gate, the permissive floor has nobody to
    # clean up after it. The decision is passed DOWN as `suppressed` — inferring it from
    # the absence of an error is what let the breaker report a degraded search as a
    # complete one.
    breaker = Breaker(STATE_DIR / "rerank-breaker", BREAKER_SECONDS)
    idle = breaker.is_open()
    suppressed = None
    if idle is not None:
        store.reranker = None
        suppressed = f"circuit breaker: the re-rank failed {idle:.0f}s ago"
        log(f"re-rank in breaker: failed {idle:.0f}s ago — strict dense cut")

    top_k = int(env("QCTX_RECALL_TOP_K", "RECALL_TOP_K", "20" if store.reranker else "8"))
    # MEMORY policy: the cross-encoder VETOES (a false positive pollutes the agent's
    # context) and the order among the approved ones is cosmetic, because all of them are
    # injected. Collapse detection is refused under a veto — see Policy.
    policy = core.Policy(dense_floor=DENSE_FLOOR, strict_floor=STRICT_FLOOR,
                           min_score=MIN_SCORE, max_results=MAX_MEMORIES,
                           veto=True, order_matters=False)
    t0 = time.monotonic()
    try:
        hits, outcome = store.recall(angles, policy, top_k, suppressed=suppressed)
    except core.EmbeddingError as exc:
        log(f"embeddings failed ({exc}) — no recall on this prompt")
        emit(unavailable_block("embeddings", type(exc).__name__))
        return
    except core.QdrantError as exc:
        log(f"Qdrant failed ({exc}) — no recall on this prompt")
        emit(unavailable_block("Qdrant", type(exc).__name__))
        return
    elapsed = time.monotonic() - t0

    if outcome.rerank_error:
        breaker.arm()
        log(f"re-rank failed ({outcome.rerank_error}) — breaker armed for {BREAKER_SECONDS:.0f}s")
    elif outcome.by_rerank:
        breaker.clear()

    # State is a NICETY — it only decides pointer-vs-full reinjection. Losing it must not
    # cost the search that already succeeded. Before this guard, an unwritable state dir
    # made the hook throw away results it was holding and tell the model the search had not
    # run: a safe direction with the wrong message.
    session = "".join(c if c.isalnum() or c in "-_" else "_"
                     for c in str(data.get("session_id") or "default"))
    state_path = None
    state: dict = {"round": 0, "seen": {}}
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_path = STATE_DIR / f"recall-{session}.json"
        state = st.load(state_path)
    except Exception as exc:
        log(f"state unavailable ({type(exc).__name__}) — proceeding without reinjection memory")
    round_no = st.next_round(state)
    seen_map = state.setdefault("seen", {})
    if not isinstance(seen_map, dict):
        # core.blocks.split_by_budget already degrades a non-dict `seen` to "nothing has
        # been seen" on its own — it will not raise or drop hits. But it does so on a
        # LOCAL substitute, on purpose: "the caller owns persistence" (see its docstring),
        # so it never reaches back into `state["seen"]` to fix it. Without this guard the
        # corruption would never heal: every round from here on would silently lose the
        # dedup memory again, forever, instead of just this one. Replacing it here means
        # the very next `st.save` below persists a clean `{}`, and dedup comes back next
        # round.
        seen_map = {}
        state["seen"] = seen_map

    if not hits:
        st.prune(state)
        # The empty line used to omit CE/collapse, so an empty round could not be told
        # apart afterwards: "the cross-encoder vetoed everything" and "there was no second
        # stage" and "the judgement was discarded" all looked identical in the log. Those
        # are the rounds most worth diagnosing.
        why = (f"CE={outcome.reranked} collapsed={outcome.collapsed} "
               f"dropped={outcome.dropped_above_floor}"
               + (f" suppressed={outcome.suppressed!r}" if outcome.suppressed else "")
               + (f" error={outcome.rerank_error!r}" if outcome.rerank_error else ""))
        log(f"round {round_no}: 0 above the cut (best {outcome.best_dense:.3f}) "
            f"in {elapsed:.1f}s | {len(angles)} angles | {why} | {prompt[:60]!r}")
        st.save(state_path, state)
        emit(empty_block(outcome, len(angles)))
        return

    full_hits, pointers = split_by_budget(hits, seen_map, round_no, BUDGET)

    pruned = st.prune(state)
    st.save(state_path, state)
    if round_no % 20 == 0:
        # A cheap, occasional sweep: once every 20 rounds is enough to keep the directory
        # from growing, and it does not pay for a `glob` on every prompt.
        dead = st.purge_dead(STATE_DIR)
        if dead:
            log(f"cleanup: {dead} dead session state(s) removed")
    scale = " (scale converted)" if outcome.scale_converted else ""
    log(f"round {round_no}: {len(full_hits)} injected + {len(pointers)} pointers "
        f"(out of {len(hits)} relevant / {outcome.candidates} candidates) in {elapsed:.1f}s | "
        f"{len(angles)} angles | CE={outcome.by_rerank}{scale} | {prompt[:60]!r}")

    emit(recall_block(full_hits, pointers, len(angles), outcome, BUDGET))


if __name__ == "__main__":
    main()
