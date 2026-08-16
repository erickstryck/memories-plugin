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

from core.chunk import is_probably_binary

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

    Reads a bounded sample and delegates the binary/text call to `chunk.is_probably_binary`
    — the same policy `docs_index` itself applies to the file — instead of re-deciding it
    here. A NUL byte alone is not enough: high-entropy content with no NUL decodes into a
    wall of U+FFFD replacement characters, which is exactly the measured incident that
    policy exists to catch (see `chunk.is_probably_binary`'s docstring). Duplicating a
    weaker check here would let this guard wave through a file `docs_index` then rejects.
    """
    try:
        with open(path, "rb") as fh:
            sample = fh.read(_SNIFF_BYTES)
    except OSError:
        return False
    return not is_probably_binary(sample.decode("utf-8", errors="replace"))


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

    if budget.window <= 0 or budget.used >= budget.window:
        # THE TWO WAYS OF NOT KNOWING THE WINDOW, and they sit together because they end
        # the same way: allow.
        #
        # `window <= 0` — we could not learn it at all. Blocking here would mean blocking
        # on a guess, and the window is not derivable from disk on either host.
        #
        # `used >= window` — we thought we knew it and the facts REFUTED us. A session
        # cannot consume more of the window than the window holds, so a `used` at or above
        # it does not describe a full session; it proves the number came from
        # `windows.MODEL_WINDOWS` guessing low for a variant that outgrew its bare name.
        # Measured before this rule existed: a real 1M session read as 200_000/989,479,
        # `free` collapsed to 0, and the guard denied a 4 KB file — fail open inverted into
        # fail closed. Erring large costs a sleeping guard; erring small costs a cage.
        return Verdict(False, "", cost, free)

    floor_hit = after > budget.window * (1 - floor_pct)
    share_hit = free > 0 and cost > free * share_pct
    if not (floor_hit or share_hit):
        return Verdict(False, "", cost, free)

    if not is_indexable(path):
        # Nothing to offer instead, so blocking would only take away an option.
        return Verdict(False, "", cost, free)

    about = "≈" if not budget.exact else ""
    # `free > 0` is guaranteed here, not hoped for: `used >= window` returned above, so
    # reaching this line means `used < window`. The branch that used to phrase `free == 0`
    # as "nothing left in the window" went with it — that state can no longer reach the
    # message, and a branch no call can enter is a branch that rots and lies about what
    # was verified.
    pct = int(round(cost / free * 100))
    head = (f"reading this file would cost {about}{cost:,} tokens, "
            f"{pct}% of the {about}{free:,} you have left")

    from core.docs import doc_id_for          # local: keeps the pure path import-light
    doc_id = doc_id_for(path)
    if indexed_ids and doc_id in indexed_ids:
        what = f"it is already indexed as {doc_id} — search it with docs_search"
    else:
        what = "index it with docs_index and search the parts that answer"

    reason = f"{head}. Instead, {what}. To read it anyway, put {ESCAPE_MARKER} in your request."

    return Verdict(True, reason, cost, free)
