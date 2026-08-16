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


def cost_of(path: str) -> int:
    """Tokens the file would cost, from its SIZE — the file is never read.

    Reading it to measure it would spend exactly what the guard exists to save.
    """
    try:
        return int(os.path.getsize(path) // CHARS_PER_TOKEN)
    except OSError:
        return 0


def decide(path: str, budget: Budget, *,
           floor_pct: float = FLOOR_PCT, share_pct: float = SHARE_PCT) -> Verdict:
    """Block when either criterion fires, whichever comes first."""
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

    about = "≈" if not budget.exact else ""
    pct = int(round(cost / free * 100)) if free else 100
    reason = (f"reading this file would cost {about}{cost:,} tokens, "
              f"{pct}% of the {about}{free:,} you have left")

    return Verdict(True, reason, cost, free)
