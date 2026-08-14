"""Assembling the text this package injects, and deciding how much of it fits.

In `core` and not in an adapter, because every host injects the same four states and needs
the same budget discipline. When this lived in hooks/ it was reachable only by the
claude-code adapter, and a second host could only copy it.

THE FOUR STATES, and the contract that makes them worth distinguishing:

  populated    memories were found and are included, with the rules for using them
  empty        the archive was consulted and nothing cleared the cut
  partial      it was consulted, but the judgement was incomplete — NOT evidence of absence
  unavailable  the search did not run at all

The third is the one that took a real defect to get right. The conclusion is derived from
whether a degradation note EXISTS, never from re-testing which degradation happened: the
two were separate conditions once, they drifted, and a block went out saying both "there
may be relevant memory outside" and "there is no recorded precedent". A model that reads
the second goes on to call something unprecedented when nothing was exhaustively searched,
which is the exact failure this package exists to prevent.
"""
import re
from dataclasses import dataclass

from .prompts import INSTRUCTIONS


@dataclass(frozen=True)
class Budget:
    """How much context the injected block may spend.

    A host parameter, not a core one: what fits depends on the window the host gives us,
    so the adapter reads its own environment and hands the numbers in.
    """
    max_memories: int
    max_chars: int
    max_per_mem: int
    reinject_after: int


def split_by_budget(hits, seen, round_no, budget):
    """Split hits into (full, pointers) and MARK what went in full as seen.

    A memory injected recently comes back as a one-line pointer: repeating the whole
    document on every prompt about the same subject inflates the context without adding
    anything, and the slot it frees reveals MORE of the archive.

    Mutates `seen` on purpose — the caller owns persistence, and threading a returned copy
    through every adapter would be ceremony for the same effect. A non-dict `seen` (a
    hand-edited state file, a future format change) degrades to "nothing has been seen":
    everything is treated as fresh and nothing is marked, rather than raising. Losing the
    dedup memory costs one memory reinjected once — it must never cost the search that
    already ran.
    """
    if not isinstance(seen, dict):
        seen = {}
    full_hits, pointers = [], []
    remaining = budget.max_chars
    for h in hits:
        last_seen = seen.get(h.id)
        recent = isinstance(last_seen, int) and (round_no - last_seen) < budget.reinject_after
        cost = min(len(h.document), budget.max_per_mem)
        if recent or len(full_hits) >= budget.max_memories or cost > remaining:
            pointers.append(h)
            continue
        full_hits.append(h)
        seen[h.id] = round_no
        remaining -= cost

    return full_hits, pointers


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


def degradation_note(outcome, max_memories: int) -> str:
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
    if outcome.dropped_above_floor and len(outcome.scored) < max_memories:
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
    #
    # `empty_block` has no budget to read `max_memories` from. It does not need one: the
    # ceiling is derived from `outcome` itself, so the guard is vacuously true and no
    # caller can make it wrong.
    note = degradation_note(outcome, max_memories=len(outcome.scored) + 1)
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


def recall_block(full_hits: list, pointers: list, n_angles: int, outcome, budget: Budget) -> str:
    parts = [
        "[automatic recall — long-term memory]",
        f"This search was EXECUTED by the harness from your prompt ({n_angles} semantic "
        "angles, fused by highest score). What follows is knowledge from earlier sessions "
        "— read it BEFORE answering, investigating or proposing a design.",
    ]
    note = degradation_note(outcome, budget.max_memories)
    if note:
        parts.append(note.rstrip())
    parts += ["", INSTRUCTIONS, ""]
    for i, h in enumerate(full_hits, 1):
        doc = h.document
        truncated = ""
        if len(doc) > budget.max_per_mem:
            doc = doc[:budget.max_per_mem]
            truncated = (f"\n[… truncated at {budget.max_per_mem} chars — retrieve the rest by "
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
