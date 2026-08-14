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
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core  # noqa: E402
from core import query  # noqa: E402
from core.breaker import Breaker  # noqa: E402
from core.prompts import INSTRUCTIONS  # noqa: E402


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

#: Rounds before reinjecting a memory in full instead of the one-line pointer. The
#: context may have been compacted in the meantime.
REINJECT_AFTER = 8


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


def unavailable_block(stage: str, error: str) -> str:
    return (
        "[automatic recall — UNAVAILABLE for this prompt]\n"
        f"The long-term memory search was NOT executed: {stage} failed ({error}). "
        "This does NOT mean there is no precedent — it means the archive was not "
        "consulted. Do not claim anything is unprecedented or without history on the "
        "strength of this turn. If the subject might have precedent, try an explicit "
        "search; if that fails too, tell the user you are without memory instead of "
        "answering as though the archive were empty."
    )


def _degradation_note(outcome) -> str:
    """One line saying the judgement was PARTIAL, when it was.

    Without this the block asserted "there is no recorded precedent on this subject"
    even when the second stage had failed — that is, it presented the result of a
    degraded pipeline with the confidence of a complete one.
    """
    parts = []
    if outcome.rerank_error:
        parts.append(f"the re-rank did NOT run ({outcome.rerank_error[:80]}), so the order is "
                      "dense and the strict cut was reapplied")
    elif outcome.suppressed:
        parts.append(f"the re-rank was not used ({outcome.suppressed}), so only the dense "
                      "stage ran and the strict cut was reapplied")
    elif outcome.collapsed:
        parts.append(f"the re-rank scored everything at or near zero (best "
                      f"{outcome.best_rerank:.4f}) and its order was discarded; that happens "
                      "with a question and a document in different languages, and also when "
                      "nothing is relevant — the scores cannot tell those apart")
    # Only unjudged candidates ABOVE the strict floor are news. The pair ceiling always
    # cuts the lowest-scoring tail, so a tail below that floor could not have held
    # anything the single-stage mode would have returned, and announcing it on every
    # prompt is crying wolf — the warning loses its value exactly when it matters.
    if outcome.dropped_above_floor and len(outcome.scored) < MAX_MEMORIES:
        parts.append(f"{outcome.dropped_above_floor} candidate(s) that clear the dense floor "
                      f"went unjudged because of the pair ceiling, and the slots were not "
                      f"filled — there may be relevant memory outside")
    if not parts:
        return ""

    return "CAUTION, partial judgement: " + "; ".join(parts) + ".\n"


def empty_block(outcome, n_angles: int) -> str:
    # The conclusion is chosen by whether a degradation note EXISTS, not by re-testing
    # which degradation happened. The two used to be separate conditions, and they drifted:
    # the note warned about candidates left unjudged by the pair ceiling while the
    # conclusion still asserted flatly that no precedent exists. Both sentences went into
    # the same block, contradicting each other — and the flat claim is the exact failure
    # this hook exists to prevent, so it is the one that has to yield.
    #
    # Deriving one from the other was necessary and not sufficient: the note itself still
    # had to LEARN each new degradation, and it did not know about the circuit breaker.
    # With the breaker open there was no in-process error, so no note, so the flat claim
    # was earned by default — for the 300 seconds after every rerank failure, i.e.
    # exactly while the shared GPU is struggling. `Outcome.suppressed` closes that by
    # making the caller state the decision instead of leaving it to be inferred.
    note = _degradation_note(outcome)
    if note:
        conclusion = ("The archive was consulted but the judgement was PARTIAL, so this is "
                     "not evidence that no precedent exists — if the subject might have "
                     "history, run a targeted search.")
    else:
        conclusion = ("There is no recorded precedent on this subject — do not repeat this "
                     "generic search. If the work opens a specific sub-subject, then a "
                     "targeted search is worth it.")

    return (
        "[automatic recall — long-term memory]\n"
        f"Search executed from your prompt ({n_angles} semantic angles): no memory above "
        f"the relevance cutoff (best score {outcome.best_dense:.3f}).\n"
        + note + conclusion +
        " And consider whether the answer you are about to produce deserves to be saved at the end."
    )


def meta_line(meta: dict) -> str:
    fields = [meta.get(k) for k in ("type", "project", "connector", "area", "date")]

    return " · ".join(str(c) for c in fields if c)


def build_block(full_hits: list, pointers: list, n_angles: int, outcome) -> str:
    parts = [
        "[automatic recall — long-term memory]",
        f"This search was EXECUTED by the harness from your prompt ({n_angles} semantic "
        "angles, fused by highest score). What follows is knowledge from earlier sessions "
        "— read it BEFORE answering, investigating or proposing a design.",
    ]
    note = _degradation_note(outcome)
    if note:
        parts.append(note.rstrip())
    parts += ["", INSTRUCTIONS, ""]
    for i, h in enumerate(full_hits, 1):
        doc = h.document
        truncated = ""
        if len(doc) > MAX_PER_MEM:
            doc = doc[:MAX_PER_MEM]
            truncated = (f"\n[… truncated at {MAX_PER_MEM} chars — retrieve the rest by "
                     f"id {h.id} if the subject is central]")
        header = f"── {i}. {h.origin} {h.score:.3f}"
        meta = meta_line(h.metadata)
        if meta:
            header += f" · {meta}"
        header += f" · id {h.id}"
        parts += [header, doc + truncated, ""]

    if pointers:
        parts.append("Also relevant, not included in full (already injected in this "
                      "session, or outside this turn's context budget — retrieve them by "
                      "id if you need the text):")
        for p in pointers:
            summary = re.sub(r"\s+", " ", p.document)[:110]
            parts.append(f"- {p.id} (score {p.score:.3f}) — {summary}…")
        parts.append("")

    return "\n".join(parts).rstrip()


def load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"round": 0, "seen": {}}


def prune_state(state: dict) -> int:
    """Drops from `seen` whatever can no longer change any decision.

    An entry only matters while `round - seen < REINJECT_AFTER`: past that the memory
    comes back IN FULL anyway, so keeping it is just occupying space. Without pruning, a
    long session accumulates one entry per memory per round forever.
    """
    round_no = int(state.get("round", 0))
    seen_map = state.get("seen", {})
    old_entries = [mid for mid, r in seen_map.items()
              if not isinstance(r, int) or (round_no - r) >= REINJECT_AFTER]
    for mid in old_entries:
        seen_map.pop(mid, None)

    return len(old_entries)


def purge_dead_sessions(days: float = 7.0) -> int:
    """Deletes state for sessions untouched for days.

    Each session creates a file and nothing removed them: the directory grew forever. A
    session idle for a week is not coming back, and if it does the cost is starting with
    an empty `seen` — the worst effect is one memory reinjected once.
    """
    cutoff = time.time() - days * 86400
    deleted = 0
    try:
        for file_path in STATE_DIR.glob("recall-*.json"):
            if file_path.stat().st_mtime < cutoff:
                file_path.unlink()
                deleted += 1
    except Exception:
        pass

    return deleted


def save_state(path: Path | None, state: dict) -> None:
    """No path means state was unavailable this round; that is not an error."""
    if path is None:
        return
    try:
        path.write_text(json.dumps(state))
    except Exception:
        pass


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
        state = load_state(state_path)
    except Exception as exc:
        log(f"state unavailable ({type(exc).__name__}) — proceeding without reinjection memory")
    state["round"] = int(state.get("round", 0)) + 1
    seen_map = state.setdefault("seen", {})
    round_no = state["round"]

    if not hits:
        prune_state(state)
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
        save_state(state_path, state)
        emit(empty_block(outcome, len(angles)))
        return

    # Budget the context. A memory injected recently comes back as a one-line pointer:
    # repeating the whole document on every prompt about the same subject inflates the
    # context without adding anything, and the freed slot reveals MORE of the archive.
    full_hits, pointers = [], []
    budget = MAX_CHARS
    for h in hits:
        last_ts = seen_map.get(h.id)
        recent = isinstance(last_ts, int) and (round_no - last_ts) < REINJECT_AFTER
        cost = min(len(h.document), MAX_PER_MEM)
        if recent or len(full_hits) >= MAX_MEMORIES or cost > budget:
            pointers.append(h)
            continue
        full_hits.append(h)
        seen_map[h.id] = round_no
        budget -= cost

    pruned = prune_state(state)
    save_state(state_path, state)
    if round_no % 20 == 0:
        # A cheap, occasional sweep: once every 20 rounds is enough to keep the directory
        # from growing, and it does not pay for a `glob` on every prompt.
        dead = purge_dead_sessions()
        if dead:
            log(f"cleanup: {dead} dead session state(s) removed")
    scale = " (scale converted)" if outcome.scale_converted else ""
    log(f"round {round_no}: {len(full_hits)} injected + {len(pointers)} pointers "
        f"(out of {len(hits)} relevant / {outcome.candidates} candidates) in {elapsed:.1f}s | "
        f"{len(angles)} angles | CE={outcome.by_rerank}{scale} | {prompt[:60]!r}")

    emit(build_block(full_hits, pointers, len(angles), outcome))


if __name__ == "__main__":
    main()
