"""The two-stage retrieval pipeline, in ONE place.

There used to be three implementations of the same idea — memory, documents and the
diagnostic —, each with its own version of the rules. That already cost something
concrete: the re-rank scale normalization existed in one consumer and not the other,
and the second would have inherited the bug back the day it needed it.

The ALGORITHM is common; what differs between consumers is POLICY, and policy is a
parameter:

    memory     -> the re-rank VETOES. Precision matters more than reach: a false
                  positive enters the agent's context and pollutes it.
    documents  -> the re-rank ORDERS and does not veto. Whoever asks has already
                  chosen the document; silence is worse than imperfect order.

Two measured asymmetries shape the design, and ignoring either one degrades
silently:

1. If the second stage does NOT run, the first stage's permissive floor is left with
   nobody to clean up after it. Going back to the strict cut is mandatory — otherwise
   the mode WITH re-ranking ends up worse than the mode without, which is the
   opposite of the intent.

2. The cross-encoder COLLAPSES on a cross-lingual pair (measured: 0.2073 with the
   query in English against 0.0004 in Portuguese, on the same English document; the
   dense stage was indifferent, 0.475 against 0.460). When it collapses, its ORDER is
   noise too — so it is not enough to stop filtering, the collapse has to be detected
   and the dense order restored.
"""
from dataclasses import dataclass, field
from typing import Any, Callable

#: A best cross-encoder score below this means collapse, not irrelevance.
#: Wide separator: collapse measured at 4e-4, healthy cases at 0.21 and 0.53.
COLLAPSE_MAX = 0.01

#: Where each result's score came from. Whoever presents it has to distinguish:
#: dense order is NOT a relevance verdict, it is vector proximity.
CE = "CE"           # judged by the cross-encoder, above the cutoff
CE_WEAK = "CE?"    # judged, below the cutoff — delivered without veto, marked
DENSE = "dense"     # no cross-encoder judgement


def default_score(hit: Any) -> float:
    return float(hit.get("score") or 0.0)


@dataclass(frozen=True)
class Policy:
    """The decisions that vary per consumer.

    `veto` is the only genuinely structural one: with it the second stage may
    ELIMINATE candidates; without it, it only reorders. Both choices are defensible,
    but for opposite reasons, so stating them explicitly keeps someone from
    "unifying" the two by mistake later.
    """
    dense_floor: float          # first-stage floor when there is a second one
    strict_floor: float         # floor when the first stage is ALONE
    min_score: float            # second-stage cutoff
    max_results: int
    veto: bool = True
    detect_collapse: bool = True
    order_matters: bool = False
    """Whether the result's ORDER is the product, or only the set.

    Memory: the results ALL go into the context at once, so reordering changes
    nothing — the second stage only pays for itself when it selects or filters.

    Documents: the result is a list read top to bottom, and the order is exactly what
    is being bought. There the second stage is worth it even when everything fits and
    everything passes.

    Applying the same rule to both was a design mistake that only surfaced once the
    pipeline became one and got tests with a fake.
    """

    def floor_for(self, has_reranker: bool) -> float:
        """First-stage floor. It only relaxes when a second stage will clean up."""
        return self.dense_floor if has_reranker else self.strict_floor


@dataclass(frozen=True)
class Scored:
    """A result with its FINAL score and where that score came from."""
    item: Any
    score: float
    origin: str

    @property
    def is_weak(self) -> bool:
        return self.origin == CE_WEAK


@dataclass
class Outcome:
    """The chosen results and the TRAIL of how they were reached.

    The trail is not a logging luxury: the consumer needs to know whether the verdict
    came from the cross-encoder or is merely vector proximity, and needs to be able to
    tell the user that the order fell back to dense because the judgement collapsed.
    """
    scored: list[Scored] = field(default_factory=list)
    candidates: int = 0
    best_dense: float = 0.0
    best_rerank: float = 0.0
    reranked: bool = False
    collapsed: bool = False
    scale_converted: bool = False
    rerank_error: str | None = None
    dropped: int = 0
    """Candidates the second stage never even saw, because of the pair ceiling.

    It used to be computed by the client and read by nobody. A candidate discarded
    without judgement may include one that would have cleared the strict cut —
    whoever presents the results has to be able to say the list is not exhaustive."""

    @property
    def items(self) -> list[Any]:
        return [s.item for s in self.scored]

    @property
    def by_rerank(self) -> bool:
        return any(s.origin in (CE, CE_WEAK) for s in self.scored)


def fuse_by_id(batches: list[list[Any]], id_of: Callable[[Any], str],
               score_of: Callable[[Any], float] = default_score) -> list[Any]:
    """Fuses results from several vectors, keeping the HIGHEST score for each id.

    Different angles on the same question catch different records; a record that shows
    up in two angles should not be penalized by the worse of the two.
    """
    fused: dict[str, Any] = {}
    for batch in batches:
        for hit in batch:
            key = id_of(hit)
            current = fused.get(key)
            if current is None or score_of(hit) > score_of(current):
                fused[key] = hit

    return sorted(fused.values(), key=lambda h: -score_of(h))


def needs_rerank(candidates: list[Any], policy: Policy,
                 score_of: Callable[[Any], float] = default_score) -> bool:
    """The second stage has two jobs, and only the second one forces the call.

    SELECT: there are more candidates than slots.
    FILTER: some candidate sits in the permissive band, and only got in because
    somebody was going to judge it.

    With neither, reordering does not change what comes out — and the call is useless
    work paid for in latency, on a path fired at every user interaction.

    The EXCEPTION is when the order is the product (`order_matters`): there ORDERING
    is a third reason on its own, and more than one candidate is enough.
    """
    if not candidates:
        return False
    if policy.order_matters and len(candidates) > 1:
        return True
    if len(candidates) > policy.max_results:
        return True

    return any(score_of(c) < policy.strict_floor for c in candidates)


def two_stage(candidates: list[Any], query: str, reranker, policy: Policy,
              text_of: Callable[[Any], str],
              score_of: Callable[[Any], float] = default_score) -> Outcome:
    """Applies the second stage to candidates ALREADY retrieved by the first.

    It receives the candidates rather than fetching them: the first stage differs
    between consumers (one collection with several angles on the question, or several
    collections with one vector), and forcing both into the same mould would create
    parameters only one of them uses.

    Never mutates its input.
    """
    outcome = Outcome(candidates=len(candidates),
                   best_dense=score_of(candidates[0]) if candidates else 0.0)
    if not candidates:
        return outcome

    if reranker is None or not needs_rerank(candidates, policy, score_of):
        outcome.scored = _strict_cut(candidates, policy, score_of)

        return outcome

    pairs, info = reranker.rank(query, [text_of(c) for c in candidates])
    outcome.reranked = bool(info.get("ok"))
    outcome.scale_converted = bool(info.get("was_logit"))
    outcome.rerank_error = info.get("error")
    outcome.dropped = int(info.get("dropped") or 0)
    outcome.best_rerank = max((s for _, s in pairs), default=0.0)

    # With no judgement, the first stage's permissive floor has nobody to clean up.
    if not outcome.reranked:
        outcome.scored = _strict_cut(candidates, policy, score_of)

        return outcome

    # Collapse: the score is low because of a language mismatch, not because of
    # irrelevance. The cross-encoder's order is noise here too.
    if policy.detect_collapse and pairs and outcome.best_rerank < COLLAPSE_MAX:
        # A collapse is the judgement being DISCARDED, so the permissive floor is once
        # again left with nobody to clean up after it — exactly as when the re-rank
        # fails. Returning the dense order without reapplying the strict cut let
        # through candidates the mode WITHOUT re-ranking would never return, which is
        # the very defect this pipeline exists to prevent.
        outcome.collapsed = True
        outcome.scored = _strict_cut(candidates, policy, score_of)

        return outcome

    judged = [(candidates[i], s) for i, s in pairs if 0 <= i < len(candidates)]
    above = [Scored(c, s, CE) for c, s in judged if s >= policy.min_score]
    below = [Scored(c, s, CE_WEAK) for c, s in judged if s < policy.min_score]
    chosen = above if policy.veto else above + below
    outcome.scored = chosen[:policy.max_results]

    return outcome


def _strict_cut(candidates: list[Any], policy: Policy,
             score_of: Callable[[Any], float]) -> list[Scored]:
    """Reapplies the strict cut. Called whenever the second stage did not judge."""
    return [Scored(c, score_of(c), DENSE) for c in candidates
            if score_of(c) >= policy.strict_floor][:policy.max_results]
