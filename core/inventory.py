"""Which documents are already in the archive — the one question the guard pays network for.

WHY THIS IS NOT IN `core/bigfile.py`. That module is PURE by design and proven so by
EXECUTION (`socket.socket.connect` patched to raise, and its whole test module still passes):
it decides from a `Budget` and a `stat`, and it must keep being the file anyone can read
without wondering what it talks to. This one talks to Qdrant. Mixing them would cost that
proof, and the proof is worth more than the import it saves.

WHY IT IS NOT IN EITHER ADAPTER EITHER, WHICH IS WHERE IT USED TO LIVE — TWICE. Both
`hooks/bigfile.py` and `hosts/hermes/bigfile.py` carried byte-identical copies of everything
below. Identical copies are the ones that diverge in silence: nothing announces the moment
they stop matching, and the deadline in here has ALREADY been fixed once, in a fix round, in
what was then the only copy. A test comparing the two would only report a divergence after it
existed. One owner reports it never.

WHEN IT MAY BE CALLED. Only after `decide(..., indexed_ids=None)` has already said it would
block. The common case is "small file, allow", and taxing it with a round trip it never needed
is the one thing the ordering exists to prevent — so the adapters import this module INSIDE
that rare branch, and a test on each host proves the common path never even imports it.

WHAT A FAILURE COSTS. Nothing but the better half of a message: "already indexed, search it"
degrades to "index it". Never the block, never the read.
"""
import threading

import core
from core.breaker import Breaker
from core.knobs import state_dir

#: Seconds allowed PER Qdrant call. A backstop, and explicitly not the thing that keeps this
#: step inside its budget — see `IDS_DEADLINE_S`.
QDRANT_TIMEOUT_S = 1.5

#: ONE HARD WALL-CLOCK DEADLINE for the whole step, and the only thing that bounds it. A
#: per-call timeout cannot: `list_docs("all")` is a structural minimum of six sequential round
#: trips (ensure tmp, sweep's ensure, sweep's delete, scroll tmp, ensure library, scroll
#: library), each bounded on its own and their SUM bounded by nothing. Measured against a
#: healthy-but-slow backend answering every endpoint correctly at 1.2 s per call, the
#: claude-code hook took 7.267 s against the `timeout: 5` declared in hooks.json — every
#: individual ceiling respected, the total blown anyway.
#:
#: The count of those round trips is knowledge that rots: `list_docs` may grow a call tomorrow
#: and nothing here would notice. A wall clock does not care how many terms the sum has. The
#: healthy path measures 0.10-0.12 s, so 2.0 s is ~20x headroom and never fires when Qdrant is
#: well. It is NOT relaxed for hermes' far larger hook timeout: the deadline exists so a guard
#: that runs before EVERY read cannot add latency the user feels, and the user's patience does
#: not scale with the host's configuration.
IDS_DEADLINE_S = 2.0

#: How long a failed (or too slow) lookup keeps this switched off. Not an env knob on purpose:
#: a knob here would have to be mirrored name-for-name on both hosts, and nothing about the
#: message quality is worth a second vocabulary.
BREAKER_SECONDS = 300.0

#: The one file this keeps, and the name is shared between hosts on purpose: what is being
#: circuit-broken is the QDRANT BACKEND, which both hosts talk to, so what one learns about it
#: holds for the other.
BREAKER_NAME = "bigfile-docs-breaker"


def indexed_ids(cfg) -> set | None:
    """Doc ids already in the archive, or None when finding out was not worth it.

    None is a complete answer, not a failure. Anything that raises — and anything still
    running when `IDS_DEADLINE_S` expires — degrades to None AND arms the breaker, so the next
    blocked read does not pay to rediscover it.

    ABANDONED, NOT CANCELLED. There is no way to interrupt a blocking socket read from
    outside, so the deadline is enforced by joining a DAEMON worker with a timeout: on expiry
    we stop waiting and walk away, and being a daemon is what lets the interpreter exit while
    it is still in flight. Checking the clock AFTER the work returned — which is what this did
    before a reviewer measured it — only ever bounds the NEXT invocation, never the one that
    is already over its budget.
    """
    breaker = Breaker(state_dir() / BREAKER_NAME, BREAKER_SECONDS)
    if breaker.is_open() is not None:
        return None
    done: dict = {}
    worker = threading.Thread(target=_collect_ids, args=(cfg, done), daemon=True)
    worker.start()
    worker.join(IDS_DEADLINE_S)
    if "ids" not in done:
        # Either still running at the deadline, or it raised. Both mean the same thing to the
        # read that is waiting on us, and both deserve the same cooldown.
        breaker.arm()

        return None
    breaker.clear()

    return done["ids"]


def _collect_ids(cfg, done: dict) -> None:
    """The round trip itself, on a worker thread. Reports only by landing a key in `done`.

    Swallows everything: this thread may still be running after `indexed_ids` has given up on
    it, and an exception raised then would print a traceback to stderr for a result nobody is
    listening for any more — and on hermes stderr is what the host quotes as the block message
    when stdout carries none.
    """
    try:
        index = core.DocIndex(core.build_qdrant(cfg, timeout=QDRANT_TIMEOUT_S),
                              # The embedder and reranker carry build_docs' own 60 s and 15 s
                              # timeouts. Harmless TODAY because `list_docs` touches neither —
                              # but they are two unbounded waits sitting inside a step that
                              # must finish in two seconds, and the day a listing grows an
                              # embedding call they become the trap. The wall clock above is
                              # what keeps that from being a regression.
                              core.build_embedder(cfg), core.build_reranker(cfg),
                              cfg.require_docs_collection(), cfg.require_library_collection(),
                              cfg.vector_size)
        # `doc_id` is the key `list_docs` publishes, confirmed against a live listing rather
        # than assumed: {'doc_id', 'scope', 'chunks', 'path', 'mode', 'indexed_at',
        # 'expires_at_ts', 'src_mtime', 'src_size', 'src_digest', 'ttl_seconds'}.
        done["ids"] = {d["doc_id"] for d in index.list_docs(scope="all")}
    except BaseException:      # noqa: BLE001 — a lost nicety may never cost the read
        pass
