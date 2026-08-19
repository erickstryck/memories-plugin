"""What the daemon actually runs. Kept apart so the daemon never has to know about Qdrant.

`daemon.run` takes a callable and knows nothing else about the work; this module builds it. The
separation is what lets every daemon test run with no network at all.

CANCELLATION IS CHECKED BETWEEN BATCHES, not inside one. A batch is already indexed or not, so
stopping between them leaves the archive consistent, and what was indexed STAYS — a partial
index answers questions about the part it has, and re-running skips whatever did not change.
"""
import os

from . import jobs, scan

#: How many files go to `add_files` at once. Small enough that progress moves visibly and a
#: cancel is honoured quickly; large enough not to pay the call overhead per file.
BATCH = 8


def work(cfg=None, index=None, batch: int = BATCH):
    """Returns the `work(job)` the daemon calls.

    `index` is injected by the tests; in production it is built from `cfg` on first use, so
    importing this module costs no connection.
    """
    def run_job(job: dict) -> None:
        target = index if index is not None else _build(cfg)
        repo = job["repo"]
        if job.get("kind") == "refresh":
            # `should_stop` gives `refresh` the SAME per-item cancel boundary the "index" path
            # already gets from its batch loop below. Without it, `refresh` walks the whole
            # repository in one call with no boundary to check between — a refresh queued by
            # the watcher and then cancelled mid-run would re-embed everything that changed
            # and STILL end up marked `cancelled` afterwards, claiming a cancellation for work
            # that fully ran.
            target.refresh(repo, should_stop=lambda: jobs.cancel_requested(repo))

            return
        paths = list(job.get("paths") or [])
        done = 0
        for start in range(0, len(paths), batch):
            if jobs.cancel_requested(repo):
                return
            chunk = paths[start:start + batch]
            target.add_files(repo, chunk)
            done += len(chunk)
            jobs.update(repo, done=done, current=chunk[-1])

    return run_job


def watcher(cfg=None, index=None):
    """Returns the `watch()` the daemon calls when no job is pending.

    WHY POLLING AND NOT `inotify`. Measured on 2026-08-18: stat over 2,000 files costs 16 ms, so
    a watch cycle is free — and `inotify` would mean either an external dependency, which this
    project refuses for a documented reason, or Linux-only code.

    WHY A CHANGE MUST BE SEEN TWICE. A file being written is a file that will change again in a
    moment; queueing on the first sighting reindexes on every keystroke of a long save. Seen
    twice with the same content, it is done being written.
    """
    seen: dict = {}
    new_memo: dict = {}

    def watch() -> None:
        target = index if index is not None else _build(cfg)
        for entry in target.list_repos():
            repo = entry["repo"]
            job = jobs.load(repo)
            if job and job.get("state") in (jobs.PENDING, jobs.RUNNING):
                # Already queued or running: a second job would only stack behind the first and
                # describe a disk that has moved on by the time it ran.
                continue
            new_paths = _new_tracked_paths(entry, target, new_memo)
            changed = set(target.changed_paths(repo)) | new_paths
            if not changed:
                seen.pop(repo, None)
                continue
            if seen.get(repo) == changed:
                if new_paths:
                    # A file just added to git was never indexed, so there is no digest for
                    # `refresh` to check — `add_files` is the one way a never-indexed file
                    # gets embedded at all, the same call the initial `add-all` job makes.
                    # Any PRE-EXISTING file that also changed this cycle is picked up as
                    # `refresh` on a LATER one, once `new_paths` is empty again: there is no
                    # job kind that does both, and inventing one here is not worth it for a
                    # combination that clears itself within one more cycle.
                    jobs.enqueue(repo, "index", sorted(new_paths))
                else:
                    jobs.enqueue(repo, "refresh", [])
                seen.pop(repo, None)
                continue
            seen[repo] = changed

    return watch


def _index_stamp(root: str):
    """`mtime_ns` of `<root>/.git/index`, or None when it cannot be read.

    This is the cheapest honest answer to "could the set of TRACKED files have moved?". Every
    way a file becomes tracked — `git add`, `git rm`, a checkout, a merge, a stash — writes the
    index, and nothing else this watcher cares about does. None means "no idea": a worktree or
    submodule where `.git` is a file, a permission problem, a path that is not a repository at
    all. Callers must treat None as "recompute", never as "unchanged", because guessing
    "unchanged" here is how a newly tracked file stays unindexed forever in silence.
    """
    try:
        return os.stat(os.path.join(root, ".git", "index")).st_mtime_ns
    except OSError:
        return None


def _new_tracked_paths(entry: dict, target, memo: dict | None = None) -> set:
    """Tracked, eligible files under any checkout of `entry` that the archive does not have a
    chunk for yet.

    WHY THIS EXISTS. `changed_paths` only walks paths the archive ALREADY has — a file added
    to git after the initial index (`git add newfile.py`, no commit needed) is invisible to
    it, so without this the watcher never sees it and it stays unindexed forever, silently.
    The spec's own step 1 for what belongs in the archive is `git ls-files`; this is that same
    source of truth, applied to the one case `changed_paths` structurally cannot cover.

    REUSES `scan.eligible` RATHER THAN A SECOND SELECTION RULE: it is the exact function the
    initial `add-all` job runs, so a file the watcher decides to index and a file `add-all`
    would have indexed are the SAME decision, made by one function — not two that could drift
    on a binary, a lockfile or the size ceiling.
    """
    repo = entry["repo"]
    roots = list(entry.get("checkouts") or [])
    stamps = tuple(_index_stamp(r) for r in roots)
    cached = memo.get(repo) if memo is not None else None
    if cached is not None and cached[0] == stamps and None not in stamps:
        eligible = cached[1]
    else:
        eligible = set()
        for root in roots:
            try:
                eligible.update(scan.eligible(root)["eligible"])
            except Exception:                         # noqa: BLE001 — one bad checkout root
                continue                              # must not blind the watcher to the rest
        if memo is not None:
            memo[repo] = (stamps, eligible)

    return {p for p in eligible if p not in target.indexed_paths(repo)}


def _build(cfg):
    import core

    return core.build_repos(cfg if cfg is not None else core.load())
