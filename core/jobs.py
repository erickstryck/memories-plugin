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
import uuid
from pathlib import Path

from . import names
from .errors import CoreError
from .knobs import state_dir

PENDING = "pending"
RUNNING = "running"
DONE = "done"
CANCELLED = "cancelled"
FAILED = "failed"


class JobError(CoreError):
    """Work could not be queued, or a cancellation could not be requested — the two calls
    whose whole job is to make a promise about the disk, so a write failure they swallowed
    would leave the caller believing something happened that never did. Raised by `enqueue`
    and `request_cancel`, deliberately; `update` tolerates the same kind of failure because a
    lost progress write is corrected by the next one, and there is no "next one" here."""


def dir() -> Path:                                  # noqa: A001 — the name says what it holds
    return state_dir() / "jobs"


def enqueue(repo: str, kind: str, paths: list[str], total: int | None = None) -> dict:
    """Queues work for `repo`, REPLACING whatever job it had.

    Replacing rather than appending: asking again means asking about the state of the disk NOW,
    and an older list of paths is a description of a repository that has since changed.

    Raises JobError if the job cannot be written to disk OR if a stale cancellation cannot be
    removed. The contract is BOTH halves: the job exists AND it starts clean. Unlike `update`
    and `request_cancel`, which tolerate write failures, a failed enqueue means work will never
    happen. An absent job is worse than a stale one: `status` could report it queued when it
    was not. And a job that starts cancelled is dead on arrival.
    """
    job = {"repo": repo, "kind": kind, "paths": list(paths),
           "total": len(paths) if total is None else int(total),
           "done": 0, "current": "", "state": PENDING, "error": "",
           "daemon_pid": 0, "queued_at": time.time(), "id": uuid.uuid4().hex}
    
    if not _write(repo, job):
        raise JobError(f"could not queue job for {repo}: state directory is unavailable")
    
    # Clear any stale cancel from a previous job. The job's contract is that it exists AND
    # starts clean. A cancel file that survives would kill it on arrival. Raise rather than
    # silently produce a dead job.
    if not _remove_cancel_file(repo):
        raise JobError(f"could not clear stale cancellation for {repo}: see {_cancel_path(repo)}")

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


def update(repo: str, only_if: str | None = None, **fields) -> dict | None:
    """Merges `fields` into the job. None when there is no job — it never CREATES one.

    Creating one here would let a late update resurrect a job that had already finished, which
    is exactly the kind of state that outlives the thing it describes.

    `only_if` IS THE JOB'S `id`, AND WITHOUT IT THIS FUNCTION WRITES TO WHATEVER JOB THE REPO
    HAS NOW — not the one the caller is holding. A job file is addressed by repository, so a
    daemon that started job A, and finished it after the user queued job B, stamped its result
    onto B. REPRODUCED: `state=done, done=0, total=2` with B's paths never indexed, and
    `status` reporting success. Passing `only_if` makes the write a compare-and-set: it lands
    only while the job on disk is still the one being described. A caller that legitimately
    means "whatever job is there" (a reaper cleaning up after a dead daemon) omits it.

    Tolerates write failures: losing one progress write is a stale number that the next write
    corrects. The cost of crashing is worse than the cost of stale progress.
    """
    job = load(repo)
    if job is None:
        return None
    if only_if is not None and job.get("id") != only_if:
        return None                     # superseded: this update describes a job that is gone
    job.update(fields)
    _write(repo, job)

    return job


def request_cancel(repo: str) -> bool:
    """Asks for `repo`'s job to stop. False when there is nothing to cancel OR the cancel
    file could not be written.

    A flag on disk rather than a signal: the daemon reads it BETWEEN files, so it stops at a
    point where what is already indexed is consistent.

    Creates a separate cancel file rather than a field in the job dict. The job dict is
    read-modify-write while progress updates run, and a cancel field would be silently
    overwritten when the daemon writes progress. No locking, no race: separate files mean the
    two writers never touch the same document.

    RAISES JobError WHEN THE FILE COULD NOT BE CREATED, the same answer `enqueue` gives for
    the same reason: the CLI's "cancel requested" is a claim about the disk, and a caller that
    swallows the write failure would print that claim while the job runs to completion,
    unaware it was never told to stop. `update`, by contrast, tolerates a lost write because a
    stale PROGRESS number is corrected by the next one — there is no "next one" for a cancel
    that silently failed to land; nothing will retry it.
    """
    if load(repo) is None:
        return False
    if not _create_cancel_file(repo):
        raise JobError(f"could not request cancellation for {repo}: see {_cancel_path(repo)}")

    return True


def cancel_requested(repo: str) -> bool:
    job = load(repo)
    return bool(job and _cancel_file_exists(repo))


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
    return dir() / f"{names.safe(repo)}.json"


def _cancel_path(repo: str) -> Path:
    return dir() / f"{names.safe(repo)}.cancel"


def _write(repo: str, job: dict) -> bool:
    """Writes the job to disk. Returns True on success, False on failure.

    Tolerates OSError silently: state that cannot be written is state the reader will not find —
    the same answer as no job at all, which every caller already handles. Only `enqueue` treats
    write failure as an error condition, because a queued job that was not written is worse than
    a missing job.
    """
    try:
        dir().mkdir(parents=True, exist_ok=True)
        path = _path(repo)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(job, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def _create_cancel_file(repo: str) -> bool:
    """Creates a cancel file for the given repo. Returns True on success, False on failure —
    checked by `request_cancel`, which must not report a cancel that did not land."""
    try:
        dir().mkdir(parents=True, exist_ok=True)
        _cancel_path(repo).touch()

        return True
    except OSError:
        return False


def _remove_cancel_file(repo: str) -> bool:
    """Removes a cancel file for the given repo. Returns True if the file is gone (including
    when it was never there — absence is success). Returns False only if the unlink failed.
    Absence means the job starts clean."""
    try:
        _cancel_path(repo).unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _cancel_file_exists(repo: str) -> bool:
    """Returns True if a cancel file exists for the given repo."""
    try:
        return _cancel_path(repo).exists()
    except OSError:
        return False



