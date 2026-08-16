"""Would reading this file cost more context than it is worth?

The criterion is RELATIVE to what remains, never the file's size alone: 171k tokens is
irrelevant in a freshly opened 1M window and fatal with 100k left. Both halves of the rule
come from the same measured incident — a 586 KB JSON read straight into context when a
search over 258 indexed chunks would have answered in ~6k tokens.

This module decides and nothing else. It does not learn the budget (the host adapters do),
it does not index (the model does, after reading the message), and it touches the network
never. That is what makes it testable with a fabricated Budget and no infrastructure.
"""
import os
from dataclasses import dataclass

#: Fraction of the window that must REMAIN after the read.
FLOOR_PCT = 0.20
#: Fraction of the free space a single file may take.
SHARE_PCT = 0.40
#: Same ratio hermes uses in `agent/context_breakdown.py::_chars_to_tokens`. Bytes are not
#: characters under UTF-8, but the error is far below the precision a threshold needs.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Budget:
    """What the host could tell us about the context right now.

    `exact` is not decoration: claude-code reports measured token usage, hermes can only
    sum message bodies. The message says which one it is, because a number that looks
    precise and is a guess is worse than a guess that admits it.
    """
    window: int
    used: int
    exact: bool


@dataclass(frozen=True)
class Verdict:
    block: bool
    reason: str
    cost: int
    free: int


#: Enough bytes to catch a NUL without paying for the file.
_SNIFF_BYTES = 8192

#: The literal the user types to force the read through. A marker and not a natural phrase
#: on purpose: "read it whole" false-positives easily ("read the whole paragraph"), and
#: `--full` is only ever typed deliberately.
ESCAPE_MARKER = "--full"


def is_indexable(path: str) -> bool:
    """Whether `docs_index` could do anything with this file.

    A NUL byte in the first few KB is the classic text/binary test, and it is enough: the
    question is not "is this valid UTF-8" but "would slicing it into chunks produce
    something searchable".
    """
    try:
        with open(path, "rb") as fh:
            return b"\x00" not in fh.read(_SNIFF_BYTES)
    except OSError:
        return False


def cost_of(path: str) -> int:
    """Tokens the file would cost, from its SIZE — the file is never read.

    Reading it to measure it would spend exactly what the guard exists to save.
    """
    try:
        return int(os.path.getsize(path) // CHARS_PER_TOKEN)
    except OSError:
        return 0


def decide(path: str, budget: Budget, *, indexed_ids: set | None = None,
           floor_pct: float = FLOOR_PCT, share_pct: float = SHARE_PCT) -> Verdict:
    """Block when either criterion fires, whichever comes first.

    `indexed_ids` is a set of doc ids ALREADY FETCHED by the caller — this function never
    reaches Qdrant itself. Knowing whether a file is indexed costs a round trip, and this
    guard runs before every file read; the common case is "small file, allow", so paying
    that round trip up front would tax the path that never needed it. The adapter fetches
    `indexed_ids` only after a first `decide(..., indexed_ids=None)` call has already said
    it would block — which is rare — and calls again just to enrich the message.
    """
    cost = cost_of(path)
    free = max(0, budget.window - budget.used)
    after = budget.used + cost

    if budget.window <= 0:
        # We could not learn the window. Allowing is mandatory: blocking here would mean
        # blocking on a guess, and the window is not derivable from disk on either host.
        return Verdict(False, "", cost, free)

    floor_hit = after > budget.window * (1 - floor_pct)
    share_hit = free > 0 and cost > free * share_pct
    if not (floor_hit or share_hit):
        return Verdict(False, "", cost, free)

    if not is_indexable(path):
        # Nothing to offer instead, so blocking would only take away an option.
        return Verdict(False, "", cost, free)

    about = "≈" if not budget.exact else ""
    if free:
        pct = int(round(cost / free * 100))
        left = f"{pct}% of the {about}{free:,} you have left"
    else:
        # free == 0: "100% of the 0 you have left" is grammatically odd and no test pins
        # it, but it is still confusing — there is no "0 you have left" to take a share of.
        # Say plainly that nothing remains instead of forcing a percentage out of it.
        left = "nothing left in the window"
    head = f"reading this file would cost {about}{cost:,} tokens, {left}"

    from core.docs import doc_id_for          # local: keeps the pure path import-light
    doc_id = doc_id_for(path)
    if indexed_ids and doc_id in indexed_ids:
        what = f"it is already indexed as {doc_id} — search it with docs_search"
    else:
        what = "index it with docs_index and search the parts that answer"

    reason = f"{head}. Instead, {what}. To read it anyway, put {ESCAPE_MARKER} in your request."

    return Verdict(True, reason, cost, free)
