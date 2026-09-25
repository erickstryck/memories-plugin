"""What the daemon actually runs. Kept apart so the daemon never has to know about Qdrant.

`daemon.run` takes a callable and knows nothing else about the work; this module builds it. The
separation is what lets every daemon test run with no network at all.

CANCELLATION IS CHECKED BETWEEN BATCHES, not inside one. A batch is already indexed or not, so
stopping between them leaves the archive consistent, and what was indexed STAYS — a partial
index answers questions about the part it has, and re-running skips whatever did not change.
"""
import os
import subprocess
import time

from . import eventlog, jobs, quarantine, scan
from .breaker import Breaker
from .errors import CoreError, infrastructure_errors
from .knobs import state_dir

#: How long the watcher leaves a repository alone after its job died on infrastructure. The
#: outage this exists for lasts minutes (a shared embedding endpoint, a restarting Qdrant),
#: and the cost of not backing off is paid on the endpoint automatic recall shares: measured
#: 10 hits in 30 cycles against a refused connection, one every 15 s for the whole outage.
BREAKER_COOLDOWN_S = 60.0

#: How long the watcher trusts its in-memory copy of a repository's archive sources when
#: nothing it can see has moved. The registry and the job cover every write this plugin makes
#: (see `_sources_key`); this covers what they cannot see, a chunk deleted by hand or a restamp
#: done by `qctx repos refresh` outside the daemon. The worst case of expiring late is one
#: extra refresh, never a file missed for good.
SOURCES_TTL_S = 300.0


def _sources_key(entry: dict, job: dict | None) -> tuple:
    """What must stay the same for a repository's cached sources to still be the archive's.

    THE JOB, because every job the daemon runs rewrites chunks, and a finished job is exactly
    the moment a stale copy would re-queue the files it just indexed: that loop was measured
    once at one re-embed every 3 cycles (`TestTheWatcherDoesNotReindexForever`). The id catches
    a new job that happens to end in the same state as the last one.

    THE REGISTRY ENTRY, because `add_files` rewrites `indexed_at` and the counts whenever it
    writes a file, and it is also reached from outside the daemon: `qctx repos add`, the hermes
    `repos_add` tool, `qctx repos refresh`.
    """
    job = job or {}

    return (entry.get("indexed_at"), entry.get("files"), entry.get("chunks"),
            job.get("id"), job.get("state"))


def index_breaker() -> Breaker:
    """The backoff shared by every indexing path, on disk so it survives the process.

    A FILE AND NOT MEMORY, for the reason `core/breaker.py` gives: the daemon is one process
    but the CLI and both hosts spawn their own, and an endpoint that is down is down for all
    of them. `state_dir()` may be unwritable, which `Breaker` already treats as "no breaker
    this round" rather than an error — losing the backoff must never cost the indexing.
    """
    try:
        path = state_dir() / "index-breaker"
    except Exception:                                  # noqa: BLE001 — see the docstring
        path = None

    return Breaker(path, BREAKER_COOLDOWN_S)

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
        breaker = index_breaker()
        try:
            _run(target, job, repo)
        except infrastructure_errors():
            # ARMED, NOT SWALLOWED. The raise still reaches `daemon._run_one`, which marks the
            # job FAILED — an outage must stay visible. The breaker only stops the WATCHER from
            # queueing the same work again on the next cycle, which it otherwise did because
            # FAILED is neither PENDING nor RUNNING.
            breaker.arm()
            raise
        # A run that got through means the dependency is back; a stale breaker would keep the
        # repository idle for the rest of the cooldown for no reason.
        breaker.clear()

    def _run(target, job: dict, repo: str) -> None:
        if job.get("kind") == "refresh":
            # `should_stop` gives `refresh` the SAME per-item cancel boundary the "index" path
            # already gets from its batch loop below. Without it, `refresh` walks the whole
            # repository in one call with no boundary to check between — a refresh queued by
            # the watcher and then cancelled mid-run would re-embed everything that changed
            # and STILL end up marked `cancelled` afterwards, claiming a cancellation for work
            # that fully ran.
            report = target.refresh(repo, should_stop=lambda: jobs.cancel_requested(repo))
            # A file that changed and then fails to re-embed stays in `changed` forever, so it
            # is queued again on every cycle — the same loop the `index` path had, reached by
            # the other door. `missing` is deliberately NOT quarantined: a file that is gone
            # costs no embedding, and `refresh` reports it on purpose for the user to see.
            for item in report or ():
                if item.get("action") == "skipped":
                    quarantine.record(repo, item["path"], item.get("reason", "unindexable"))

            return
        # AN UNKNOWN KIND IS AN ERROR, NOT AN EMPTY BATCH. `index` used to be the fall-through,
        # so a job kind nobody implemented iterated an empty `paths`, did nothing, and was
        # stamped `done` with 0/0 — a silent success for work that never ran. `daemon._run_one`
        # turns this raise into a FAILED job the user can see, which is the honest report.
        if job.get("kind") not in ("index", None):
            raise CoreError(f"unknown job kind {job.get('kind')!r}")
        paths = list(job.get("paths") or [])
        done = 0
        for start in range(0, len(paths), batch):
            if jobs.cancel_requested(repo):
                return
            chunk = paths[start:start + batch]
            # THE RETURN VALUE IS THE POINT. `add_files` has always reported which paths it
            # skipped and why; discarding it here is what made the watcher queue the same
            # unindexable files every ~40 s forever, saturating the embedding endpoint that
            # automatic recall shares (measured: 0.04 s idle, 1.98 s during a batch, against a
            # 2.00 s ceiling).
            out = target.add_files(repo, chunk) or {}
            skipped = dict(out.get("skipped") or ())
            for path, why in skipped.items():
                quarantine.record(repo, path, why)
            # Anything that went in is released, so a file repaired between two runs stops
            # being held without anyone having to say so.
            quarantine.clear(repo, [p for p in chunk if p not in skipped])
            done += len(chunk)
            # Same compare-and-set as the state writes in `daemon._run_one`: progress for a
            # job that has been superseded must not be written onto its replacement.
            jobs.update(repo, only_if=job.get("id"), done=done, current=chunk[-1])

    return run_job


def watcher(cfg=None, index=None, clock=time.monotonic):
    """Returns the `watch()` the daemon calls when no job is pending.

    WHY POLLING AND NOT `inotify`. Measured on 2026-08-18: stat over 2,000 files costs 16 ms, so
    a watch cycle is free — and `inotify` would mean either an external dependency, which this
    project refuses for a documented reason, or Linux-only code.

    WHY A CHANGE MUST BE SEEN TWICE. A file being written is a file that will change again in a
    moment; queueing on the first sighting reindexes on every keystroke of a long save. Seen
    twice with the same content, it is done being written.

    WHY THE ARCHIVE'S SOURCES ARE KEPT IN MEMORY. The 16 ms above is the LOCAL half. The other
    half is knowing what the archive holds, and that is a scroll of every chunk of the
    repository: measured on 2026-09-25, 28,019 chunks, 110 pages and 11 MB for one repository,
    ~15 MB and ~150 HTTPS requests per cycle across five, ~8.6% CPU on a daemon with nothing to
    do. So each repository's sources are read once and reused until one of the things that can
    change them moves: see `_sources_key`. `clock` is injected for the tests, like `sleep` is
    for `daemon.run`.
    """
    seen: dict = {}
    new_memo: dict = {}
    sources_memo: dict = {}

    def sources_for(target, entry: dict, job) -> dict:
        repo = entry["repo"]
        key = _sources_key(entry, job)
        cached = sources_memo.get(repo)
        if cached is not None and cached[0] == key and clock() - cached[1] < SOURCES_TTL_S:
            return cached[2]
        started = time.monotonic()
        sources = target.indexed_sources(repo)
        sources_memo[repo] = (key, clock(), sources)
        # Logged when the read CHANGED something, not on every expiry: five repositories
        # re-read every 300 s would be ~1,440 lines a day pushing the job lines out of a log
        # that rotates at 256 KB. `eventlog` never raises.
        if cached is None or cached[0] != key or len(cached[2]) != len(sources):
            eventlog.write(eventlog.DAEMON, f"sources repo={repo} files={len(sources)} "
                                            f"in {time.monotonic() - started:.1f}s")

        return sources

    def watch() -> None:
        target = index if index is not None else _build(cfg)
        # Read once per cycle, not per repository: it is one file read, and every repo in this
        # loop is behind the same endpoint.
        breaker = index_breaker()
        entries = target.list_repos()
        # A repository that left the registry takes its copy with it: nothing else would ever
        # free it, and a re-registered name must start from what the archive says now.
        for gone in set(sources_memo) - {e["repo"] for e in entries}:
            sources_memo.pop(gone, None)
        for entry in entries:
            repo = entry["repo"]
            job = jobs.load(repo)
            if job and job.get("state") in (jobs.PENDING, jobs.RUNNING):
                # Already queued or running: a second job would only stack behind the first and
                # describe a disk that has moved on by the time it ran.
                continue
            # BACKED OFF AFTER AN INFRASTRUCTURE FAILURE. A job that died because the endpoint
            # was down lands FAILED, which is neither PENDING nor RUNNING — so the guard above
            # let the same work be queued again every cycle. Measured: 10 hits on a refused
            # connection in 30 cycles, on the endpoint automatic recall shares.
            if breaker.is_open() is not None:
                continue
            # ONE archive read answers both halves of the question ("what moved" and "what is
            # already indexed"), and it is paid only when the copy in memory went stale.
            sources = sources_for(target, entry, job)
            state = {"changed": target.changed_paths(repo, sources=sources),
                     "indexed": set(sources)}
            new_paths = _new_tracked_paths(entry, state["indexed"], new_memo)
            # Read ONCE per cycle, not per candidate: `held` reads the file, stats every
            # entry and may rewrite it. Inside `_new_tracked_paths`'s comprehension it ran
            # per eligible path — measured 110 ms for 2,000 paths against the 16 ms this
            # module's own docstring budgets for a whole cycle.
            held = quarantine.held(repo)
            # SUBTRACTED FROM `new_paths` ITSELF, not only from the union, because the two
            # lines below read `new_paths` RAW — once to pick the branch, once as the payload.
            # Filtering only the union left a held file queued forever AND, worse, kept
            # `new_paths` non-empty so the `refresh` branch was never reached: the genuinely
            # changed file was never repaired. Measured before this line, with one held file
            # beside one healthy one: 3 index jobs for the held file in 6 cycles, 0 refreshes.
            new_paths -= held
            # The union still needs the subtraction for the OTHER door: a file that WAS
            # indexed and then broke keeps its old chunks (`_write_one` raises before it
            # deletes), so `changed_paths` reports it forever. Measured: 3 refresh jobs in 6
            # cycles for one emptied file already on record.
            changed = (set(state["changed"]) | new_paths) - held
            # A VANISHED FILE IS NOT WORK FOR THE WATCHER. `changed_paths` reports a path it
            # cannot `stat` (correctly — `refresh` is where a missing file gets REPORTED, and
            # it deliberately keeps the chunks rather than deleting an archive nobody asked to
            # delete). But the report never changes, so acting on it queued a refresh every
            # other cycle forever, and a refresh read-and-SHAs every indexed path in the repo.
            # Measured before this line: 5 refresh jobs in 10 cycles for one deleted file.
            # The quarantine cannot cover this — `missing` is deliberately not held.
            changed = {path for path in changed if os.path.exists(path)}
            if not changed:
                seen.pop(repo, None)
                continue
            if seen.get(repo) == changed:
                kind = "index" if new_paths else "refresh"
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
                # After the enqueue, which raises when it cannot land: logged first, a failed
                # write read as a job that was queued.
                eventlog.write(eventlog.DAEMON, f"enqueue repo={repo} kind={kind} "
                                                f"paths={len(new_paths) or len(changed)}")
                seen.pop(repo, None)
                continue
            seen[repo] = changed

    return watch


def _index_stamp(root: str):
    """`mtime_ns` of the git index that governs `root`, or None when it cannot be found.

    This is the cheapest honest answer to "could the set of TRACKED files have moved?". Every
    way a file becomes tracked — `git add`, `git rm`, a checkout, a merge, a stash — writes the
    index, and nothing else this watcher cares about does.

    ASKS GIT WHERE THE INDEX IS INSTEAD OF ASSUMING `<root>/.git/index`. In a WORKTREE — a
    setup this plugin supports, see `core/bindings.py` — `.git` is a FILE holding a pointer,
    and the real index lives under `<main repo>/.git/worktrees/<name>/index`. The assumed path
    simply does not exist there, so the stamp was permanently None and the memo never hit: the
    watcher re-ran `git ls-files` and re-sniffed 8 KB of every tracked file on every cycle,
    forever, for exactly the users on the more advanced setup. `--absolute-git-dir` answers
    correctly for both layouts.

    None still means "no idea" — not a repository, git unavailable, a permission problem — and
    callers must treat it as "recompute", never as "unchanged": guessing "unchanged" here is
    how a newly tracked file stays unindexed forever, in silence.
    """
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse", "--absolute-git-dir"],
                             capture_output=True, timeout=30)
        if out.returncode != 0:
            return None
        git_dir = out.stdout.decode("utf-8", "replace").strip()
        if not git_dir:
            return None

        return os.stat(os.path.join(git_dir, "index")).st_mtime_ns
    except (OSError, subprocess.SubprocessError):
        return None


def _new_tracked_paths(entry: dict, indexed: set, memo: dict | None = None) -> set:
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

    THE EXPENSIVE HALF IS PAID ONLY FOR FILES THE ARCHIVE DOES NOT HAVE. `scan.eligible` opens
    and reads every tracked file to judge it (measured: 50.8 ms per 2,000 files against 3.0 ms
    to `stat` them, and 207 ms for a 10,380-file checkout — 13x the 16 ms this module budgets
    for a whole cycle). Nearly all of that is spent re-judging files that are already indexed
    and whose verdict is then thrown away on the last line. Subtracting `indexed` from the
    candidate list FIRST leaves the content judgement for the handful of genuinely new paths,
    which is what `core/scan.py` means when it says the sniff happens on the eligibility pass
    and never on the watcher's cycle.

    BUT A NARROWED ANSWER CANNOT BE CACHED AS IF IT WERE COMPLETE. The memo is keyed on the
    GIT index, which does not move when the ARCHIVE changes, so caching the already-narrowed
    set made a file the archive LOST invisible until someone happened to run `git add` — and
    that recovery is the whole reason this function exists (`poll` reports a file with no
    chunks as neither indexed nor changed). Measured: 0 index jobs in 10 cycles where the
    unmemoized path finds the file at once. So the memo also records WHAT IT NARROWED AGAINST,
    and a path that has since left `indexed` is re-judged — a set difference per cycle, and a
    read only for the handful of files that actually dropped out.
    """
    repo = entry["repo"]
    roots = list(entry.get("checkouts") or [])
    stamps = tuple(_index_stamp(r) for r in roots)
    cached = memo.get(repo) if memo is not None else None
    eligible: set = set()
    skipped: set = set()
    if cached is not None and cached[0] == stamps and None not in stamps:
        eligible, narrowed_against = cached[1], cached[2]
        # Paths the cold pass never looked at because the archive held them, and that the
        # archive no longer holds. They are the only candidates a warm memo can be missing.
        skipped = narrowed_against - indexed
    else:
        narrowed_against = set(indexed)
        for root in roots:
            try:
                eligible.update(scan.eligible(root, judge=lambda p: p not in indexed)["eligible"])
            except Exception:                         # noqa: BLE001 — one bad checkout root
                continue                              # must not blind the watcher to the rest

    if skipped:
        for root in roots:
            try:
                eligible.update(scan.eligible(root, judge=lambda p: p in skipped)["eligible"])
            except Exception:                         # noqa: BLE001 — as above
                continue
        narrowed_against = narrowed_against - skipped

    if memo is not None:
        memo[repo] = (stamps, eligible, narrowed_against)

    return {p for p in eligible if p not in indexed}


def _build(cfg):
    import core

    return core.build_repos(cfg if cfg is not None else core.load())
