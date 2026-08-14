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

#: A best cross-encoder score below this MAY mean collapse. It cannot mean it on its
#: own, and the original comment here claiming a "wide separator" was wrong: it
#: compared the collapse band only against RELEVANT cases (0.21 and 0.53) and never
#: against irrelevant ones. The same model on the same server scores a plainly
#: irrelevant document at 1.6e-05 — three orders of magnitude BELOW this threshold.
#: Measured consequence: 4 of 12 off-topic Portuguese prompts against a Portuguese
#: archive were classified as collapsed, with no language mismatch anywhere.
#:
#: So this threshold separates "the cross-encoder found nothing" from "the
#: cross-encoder found something", and nothing finer. Which of the two reasons
#: produced the silence is NOT decidable from the scores — see Policy.detect_collapse.
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
    detect_collapse: bool = False
    """Whether a crushed cross-encoder result should be DISCARDED for the dense order.

    Defaults to off, and it is REFUSED together with `veto` — see `__post_init__`.

    Treating a crushed result as collapse means throwing the judgement away and
    returning the dense candidates instead. That is right when the second stage only
    reorders: silence would be worse than imperfect order, and the discarded judgement
    was not going to remove anything anyway.

    It is wrong when the second stage VETOES. There, "every score is negligible" is
    also exactly what the cross-encoder says when nothing is relevant — and the scores
    cannot tell the two apart (COLLAPSE_MAX). So the only thing collapse detection can
    do under a veto is put back the candidates the cross-encoder just rejected, which
    is the precise opposite of why the veto exists.
    """
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

    def __post_init__(self):
        # Refused rather than documented, because the combination is not a tuning
        # mistake someone can measure their way out of: under a veto, collapse
        # detection can only ever restore what the veto removed. A future reliable
        # collapse detector (language ID, score distribution) would make this coherent
        # again — removing the guard should be a deliberate act, not an oversight.
        if self.veto and self.detect_collapse:
            raise ValueError(
                "veto=True with detect_collapse=True is incoherent: collapse detection "
                "discards the cross-encoder's verdict, and under a veto that can only "
                "re-add the candidates it just rejected. Pick one."
            )

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
    """Candidates the second stage never even saw, because of the pair ceiling."""
    dropped_above_floor: int = 0
    """Of those, how many would have survived on dense score alone.

    This is the number worth telling anyone about, and `dropped` is not. Candidates
    arrive sorted by dense score, so what the pair ceiling cuts is always the tail —
    and a tail entirely below `strict_floor` could not have contained anything the
    single-stage mode would have returned. Warning about it is warning about nothing.

    Measured before this existed: a real prompt produced 27 candidates with a best
    dense score of 0.510, i.e. every one of them below the 0.58 strict floor, and the
    injected block still announced "21 candidate(s) went unjudged … there may be
    relevant memory outside". Production runs 25-40 candidates against a ceiling of 12,
    so that warning fired on essentially every empty result — which is the crying-wolf
    failure the code that emits it argues against in its own comment."""
    suppressed: str | None = None
    """Why the second stage was not offered at all, when that was a DECISION.

    Distinct from `rerank_error`, which means it was offered and failed. The caller
    that skips the reranker on purpose — a circuit breaker holding it back after a
    recent failure, say — is the only one who knows; the pipeline sees an absent
    reranker and cannot tell that apart from a deployment that has none.

    That gap produced a real defect: with the breaker open, the trail carried no sign
    of degradation at all, so the hook reported an empty result as proof that no
    precedent exists — for the 300 seconds following every rerank failure, which is
    exactly when the shared GPU is struggling."""

    @property
    def items(self) -> list[Any]:
        return [s.item for s in self.scored]

    @property
    def by_rerank(self) -> bool:
        return any(s.origin in (CE, CE_WEAK) for s in self.scored)

    @property
    def partial(self) -> bool:
        """Whether anything stopped this from being a complete judgement.

        One place to ask, so a consumer cannot answer it by re-testing a subset of the
        reasons and quietly missing the one added last — which is how the breaker case
        escaped."""
        return bool(self.rerank_error or self.collapsed or self.suppressed
                    or self.dropped_above_floor)


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
    """The second stage has three jobs, and any one of them forces the call.

    VETO: the second stage may ELIMINATE any candidate, so under `veto` there is no
    such thing as a candidate that does not need judging. This was the original bug:
    the FILTER predicate below asks whether some candidate sits in the permissive band,
    which is the right question only when the second stage cannot remove anything.
    Measured: two candidates at 0.90 and 0.59, both clearing the strict floor and both
    fitting the slots, so the reranker was never called — and the cross-encoder would
    have scored the 0.59 one at 0.004. Adding an unrelated third candidate made the
    call happen and the bad one was vetoed. Whether a memory got judged depended on a
    candidate that had nothing to do with it.

    SELECT: there are more candidates than slots.

    FILTER: some candidate sits in the permissive band, and only got in because
    somebody was going to judge it.

    With none of them, reordering does not change what comes out — and the call is
    useless work paid for in latency, on a path fired at every user interaction.

    The EXCEPTION is when the order is the product (`order_matters`): there ORDERING
    is a further reason on its own, and more than one candidate is enough.
    """
    if not candidates:
        return False
    if policy.veto:
        return True
    if policy.order_matters and len(candidates) > 1:
        return True
    if len(candidates) > policy.max_results:
        return True

    return any(score_of(c) < policy.strict_floor for c in candidates)


def two_stage(candidates: list[Any], query: str, reranker, policy: Policy,
              text_of: Callable[[Any], str],
              score_of: Callable[[Any], float] = default_score,
              suppressed: str | None = None) -> Outcome:
    """Applies the second stage to candidates ALREADY retrieved by the first.

    It receives the candidates rather than fetching them: the first stage differs
    between consumers (one collection with several angles on the question, or several
    collections with one vector), and forcing both into the same mould would create
    parameters only one of them uses.

    `suppressed` is how a caller says "there IS a second stage, I chose not to use it
    this time, and here is why". Without it, a deliberately skipped reranker is
    indistinguishable from a deployment that has none, and the trail says the judgement
    was complete when it was not.

    Never mutates its input.
    """
    outcome = Outcome(candidates=len(candidates), suppressed=suppressed,
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
    # Candidates arrive dense-sorted, so the ceiling always cuts the tail: the dropped
    # ones are the last `dropped` entries, and only those clearing the strict floor
    # could have contained something the single-stage mode would have returned.
    if outcome.dropped:
        outcome.dropped_above_floor = sum(
            1 for c in candidates[-outcome.dropped:]
            if score_of(c) >= policy.strict_floor)
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
