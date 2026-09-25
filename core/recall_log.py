"""What a line of `recall.log` says, written by both hosts and read back by `qctx stats`.

ONE OWNER FOR THE FORMAT, because two hosts write the same file. The claude-code hook built
its lines inline; the hermes provider built none. Letting each host phrase its own would make
the two drift the first time either changed, and a summary that parses both would then have
to know two dialects of one event.

THE WORDING IS THE HOOK'S, kept on purpose: every `recall.log` already on disk was written by
it, and `parse` has to read those lines too. What changed is the host prefix, `[hermes]` or
`[claude-code]`; a line without one predates it and can only be the hook's.

`parse` returns None for anything it does not recognise and never raises, since it reads a
file other versions of this package also write.
"""
import re

from . import eventlog

HOSTS = ("claude-code", "hermes")

_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (?:\[([a-z-]+)\] )?(.+)$")
_ROUND = re.compile(r"^round \d+: (?:(\d+) injected \+ (\d+) pointers|0 above the cut)"
                    r".*? in ([\d.]+)s")
#: What a failure line starts with, per dependency. The older hook's wording is the same up to
#: the separator it used, so these prefixes read both.
_FAILURES = (("embeddings failed", "embeddings"), ("Qdrant failed", "qdrant"),
             ("re-rank failed", "rerank"), ("incomplete config", "config"),
             ("unexpected failure", "unexpected"))
_OTHER = (("skip (", "skip"), ("re-rank in breaker", "breaker"), ("cleanup:", "cleanup"),
          ("config:", "config"), ("no prompt in the payload", "no-prompt"),
          ("state unavailable", "state"))
_DEPENDENCY_WORDS = {"embeddings": "embeddings failed", "qdrant": "Qdrant failed",
                     "config": "incomplete config"}


def record(host: str, line: str) -> bool:
    """Appends one of the lines below to `recall.log`, prefixed with the host that wrote it."""
    return eventlog.write(eventlog.RECALL, f"[{host}] {line}")


def _quoted(prompt: str) -> str:
    return repr(prompt[:60])


def round_line(round_no: int, full: int, pointers: int, relevant: int, outcome, *,
               elapsed: float, angles: int, prompt: str) -> str:
    scale = " (scale converted)" if outcome.scale_converted else ""

    return (f"round {round_no}: {full} injected + {pointers} pointers "
            f"(out of {relevant} relevant / {outcome.candidates} candidates) in {elapsed:.1f}s | "
            f"{angles} angles | CE={outcome.by_rerank}{scale} | {_quoted(prompt)}")


def empty_line(round_no: int, outcome, *, elapsed: float, angles: int, prompt: str) -> str:
    """A round that found nothing above the cut. It carries everything that tells the empty
    rounds apart afterwards: "the cross-encoder vetoed everything", "there was no second stage"
    and "the judgement was discarded" are the rounds most worth diagnosing."""
    why = (f"CE={outcome.reranked} collapsed={outcome.collapsed} "
           f"dropped={outcome.dropped_above_floor}"
           + (f" suppressed={outcome.suppressed!r}" if outcome.suppressed else "")
           + (f" error={outcome.rerank_error!r}" if outcome.rerank_error else ""))

    return (f"round {round_no}: 0 above the cut (best {outcome.best_dense:.3f}) "
            f"in {elapsed:.1f}s | {angles} angles | {why} | {_quoted(prompt)}")


def failure_line(dependency: str, detail: str) -> str:
    """`dependency` is one of embeddings, qdrant, config, unexpected."""
    if dependency == "unexpected":
        return f"unexpected failure ({detail})"

    return f"{_DEPENDENCY_WORDS[dependency]} ({detail}), no recall on this prompt"


def rerank_failed_line(error: str, breaker_seconds: float) -> str:
    return f"re-rank failed ({error}), breaker armed for {breaker_seconds:.0f}s"


def breaker_line(idle_seconds: float) -> str:
    return f"re-rank in breaker: failed {idle_seconds:.0f}s ago, strict dense cut"


def skip_line(reason: str, prompt: str) -> str:
    return f"skip ({reason}): {_quoted(prompt)}"


def cleanup_line(removed: int) -> str:
    return f"cleanup: {removed} dead session state(s) removed"


def parse(line: str) -> dict | None:
    """`{ts, host, kind, ...}` for a line this module (or an earlier hook) wrote, else None.

    `kind` is one of: round, failure, skip, breaker, cleanup, config, no-prompt, state. A round
    carries `injected`, `pointers` and `elapsed`; a failure carries `dependency`.
    """
    m = _LINE.match(line.strip()) if isinstance(line, str) else None
    if not m:
        return None
    ts, host, msg = m.group(1), m.group(2) or "claude-code", m.group(3)
    out = {"ts": ts, "host": host}
    r = _ROUND.match(msg)
    if r:
        return {**out, "kind": "round", "injected": int(r.group(1) or 0),
                "pointers": int(r.group(2) or 0), "elapsed": float(r.group(3))}
    for prefix, dependency in _FAILURES:
        if msg.startswith(prefix):
            return {**out, "kind": "failure", "dependency": dependency}
    for prefix, kind in _OTHER:
        if msg.startswith(prefix):
            return {**out, "kind": kind}

    return None
