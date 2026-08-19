"""What the daemon is doing, on disk, where every command can read it.

THERE IS NO PROTOCOL BETWEEN PROCESSES, and that is the design. A command writes a file; the
daemon reads it on its next cycle; `status` reads the same file. A socket would be more
immediate and would bring negotiation, a wire format and a new failure mode — a daemon that is
alive but not listening. For a queue that changes once a minute, the cost does not pay.

The consequence to hold onto: a file saying `running` under a daemon that died is LYING, and a
state that lies is worse than one that is absent — `status` would render stalled progress as
activity. `reap` is what turns that into an honest "interrupted".
"""
import json
import os
import time
from pathlib import Path

from .knobs import state_dir

PENDING = "pending"
RUNNING = "running"
DONE = "done"
CANCELLED = "cancelled"
FAILED = "failed"


def dir() -> Path:                                  # noqa: A001 — the name says what it holds
    return state_dir() / "jobs"


def enqueue(repo: str, kind: str, paths: list[str], total: int | None = None) -> dict:
    """Queues work for `repo`, REPLACING whatever job it had.

    Replacing rather than appending: asking again means asking about the state of the disk NOW,
    and an older list of paths is a description of a repository that has since changed.
    """
    job = {"repo": repo, "kind": kind, "paths": list(paths),
           "total": len(paths) if total is None else int(total),
           "done": 0, "current": "", "state": PENDING, "error": "",
           "cancel": False, "daemon_pid": 0, "queued_at": time.time()}
    _write(repo, job)

    return job


def load(repo: str) -> dict | None:
    try:
        return json.loads(_path(repo).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def all_jobs() -> list[dict]:
    out = []
    try:
        paths = sorted(dir().glob("*.json"))
    except OSError:
        return []
    for path in paths:
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue

    return out


def update(repo: str, **fields) -> dict | None:
    """Merges `fields` into the job. None when there is no job — it never CREATES one.

    Creating one here would let a late update resurrect a job that had already finished, which
    is exactly the kind of state that outlives the thing it describes.
    """
    job = load(repo)
    if job is None:
        return None
    job.update(fields)
    _write(repo, job)

    return job


def request_cancel(repo: str) -> bool:
    """Asks for `repo`'s job to stop. False when there is nothing to cancel.

    A flag on disk rather than a signal: the daemon reads it BETWEEN files, so it stops at a
    point where what is already indexed is consistent.
    """
    return update(repo, cancel=True) is not None


def cancel_requested(repo: str) -> bool:
    job = load(repo)

    return bool(job and job.get("cancel"))


def next_pending() -> dict | None:
    """The oldest pending job, or None. Skips anything already running: one job at a time."""
    pending = [j for j in all_jobs() if j.get("state") == PENDING]
    if any(j.get("state") == RUNNING for j in all_jobs()):
        return None

    return min(pending, key=lambda j: j.get("queued_at", 0)) if pending else None


def reap(pid_alive) -> list[str]:
    """Marks as interrupted every RUNNING job whose daemon is gone. Returns the repos touched.

    `pid_alive` is injected so a test can decide without spawning anything.
    """
    touched = []
    for job in all_jobs():
        if job.get("state") != RUNNING:
            continue
        if pid_alive(job.get("daemon_pid") or 0):
            continue
        update(job["repo"], state=FAILED,
               error="interrupted: the daemon running this job is gone")
        touched.append(job["repo"])

    return touched


def _path(repo: str) -> Path:
    return dir() / f"{_safe(repo)}.json"


def _write(repo: str, job: dict) -> None:
    try:
        dir().mkdir(parents=True, exist_ok=True)
        path = _path(repo)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(job, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        # State that cannot be written is state the reader will not find — the same answer as
        # no job at all, which every caller already handles.
        pass


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name or "default"))
