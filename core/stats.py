"""What `recall.log` and `daemon.log` add up to. The engine behind `qctx stats`.

It reads the two logs and nothing else, so it cannot disturb what it summarises and needs no
configuration: the Qdrant being down is one of the things someone runs this to find out. The
recall lines are parsed by `core.recall_log`, their one owner; the daemon's lines are few and
fixed, and are matched here.

Nothing here raises for what is on disk. A log another version wrote, or one that is half
garbage, yields a smaller summary, never a traceback.
"""
import math
import re
from pathlib import Path

from . import eventlog, recall_log

_DAEMON = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (\S+)(.*)$")
_RESULT = re.compile(r"result=(\w+)")
_VERSION = re.compile(r"version=(\S+)")


def summarize(state_dir: Path) -> dict:
    """`{"recall": {host: {...}}, "daemon": {...}}`. See `_recall` and `_daemon` for the keys."""
    return {"recall": _recall(_lines(state_dir / eventlog.RECALL)),
            "daemon": _daemon(_lines(state_dir / eventlog.DAEMON))}


def _lines(target: Path) -> list:
    try:
        return target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _recall(lines: list) -> dict:
    """Per host: rounds, with_memories, empty, skips, breaker, failures {dependency: n},
    p50/p95/max latency of the rounds, and the first and last timestamp seen."""
    hosts: dict = {}
    for line in lines:
        entry = recall_log.parse(line)
        if entry is None:
            continue
        h = hosts.setdefault(entry["host"], {"rounds": 0, "with_memories": 0, "empty": 0,
                                             "skips": 0, "breaker": 0, "failures": {},
                                             "_elapsed": [], "first": entry["ts"]})
        h["last"] = entry["ts"]
        kind = entry["kind"]
        if kind == "round":
            h["rounds"] += 1
            h["with_memories" if entry["injected"] or entry["pointers"] else "empty"] += 1
            h["_elapsed"].append(entry["elapsed"])
        elif kind == "failure":
            dep = entry["dependency"]
            h["failures"][dep] = h["failures"].get(dep, 0) + 1
        elif kind == "skip":
            h["skips"] += 1
        elif kind == "breaker":
            h["breaker"] += 1
    for h in hosts.values():
        elapsed = sorted(h.pop("_elapsed"))
        h.update(p50=_percentile(elapsed, 0.50), p95=_percentile(elapsed, 0.95),
                 max=elapsed[-1] if elapsed else None)

    return hosts


def _percentile(ordered: list, q: float):
    """Nearest-rank percentile, None for no data. Nearest rank and not interpolation because
    every value reported is then one that was actually measured."""
    if not ordered:
        return None

    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _daemon(lines: list) -> dict:
    """jobs {result: n}, enqueued, archive_reads, watcher_errors, last_start, version,
    last_error (the most recent failed job or watcher failure, as written)."""
    out = {"jobs": {}, "enqueued": 0, "archive_reads": 0, "watcher_errors": 0,
           "last_start": None, "version": None, "last_error": None}
    for line in lines:
        m = _DAEMON.match(line.strip())
        if not m:
            continue
        ts, event, rest = m.groups()
        if event == "start":
            version = _VERSION.search(rest)
            out.update(last_start=ts, version=version.group(1) if version else None)
        elif event == "job":
            result = _RESULT.search(rest)
            key = result.group(1) if result else "unknown"
            out["jobs"][key] = out["jobs"].get(key, 0) + 1
            if key == "failed":
                out["last_error"] = f"{ts}{rest}"
        elif event == "enqueue":
            out["enqueued"] += 1
        elif event == "sources":
            out["archive_reads"] += 1
        elif event == "watch" and rest.startswith(" failed"):
            out["watcher_errors"] += 1
            out["last_error"] = f"{ts} watch{rest}"

    return out
