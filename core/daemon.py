"""The background process: runs what is queued, watches what is indexed, and ends with its hosts.

WHY A DAEMON AT ALL. Indexing a real project is minutes of work — measured at ~0.13 s per file,
so 1,800 files is several minutes — and doing it in the caller's terminal means no progress, no
cancelling, and a frozen prompt. There is no prohibition on background processes in this project;
an earlier note about how the temporary archive's TTL expires was mistaken for one.

WHY IT ENDS WITH THE HOSTS. A process that outlives the tool that started it is a process the
user did not ask for and will not think to stop. Each cycle it checks the leases, and no living
lease means nobody is using it.

WHY `run` TAKES ITS WORKER AND ITS CYCLE COUNT. So the whole loop can be exercised in-process,
with no spawning and no network — the same choice `refresh_window(probe=...)` already makes.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import jobs, lease
from .errors import CoreError
from .knobs import state_dir

#: How long the loop sleeps between cycles. Short enough that a cancel or a file change is
#: noticed while the user is still looking at the screen; long enough to be free.
CYCLE_S = 5.0


class DaemonError(CoreError):
    """The daemon could not be started, or a start could not be confirmed. Raised only by
    `start`, and only after the claim it took is released — a caller that catches this must
    never see a stale claim left behind that a later `start()` would trip over."""


def path() -> Path:
    return state_dir() / "daemon.json"


def record() -> dict | None:
    """The running daemon, or None. A record whose process is gone reads as none.

    Same `(pid, starttime)` test the leases use: a recycled pid must not make a dead daemon look
    alive, or nothing would ever start one again.
    """
    try:
        entry = json.loads(path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict):
        return None
    if not lease.alive(entry):
        return None

    return entry


def start(spawn=None, argv: list[str] | None = None, sleep=time.sleep) -> dict:
    """Starts the daemon if none is running. `{"action": "started" | "already", "pid": int}`.

    CLAIMS THE RECORD FILE BEFORE SPAWNING, not after. `record()` then `write()` is a
    check-then-write: two commands invoked at the same moment can both read no daemon and both
    spawn one — the gap is between two of OUR statements, not something the OS arbitrates.
    `_claim()` instead creates `daemon.json` with `O_EXCL`, which the kernel guarantees only one
    caller can do for a given path; the loser gets `FileExistsError` and never spawns. This is
    the design's answer to "two daemons": exclusive creation decides who won, not a check.

    `spawn` is injected for tests; by default it launches a detached `qctx repos daemon run`.

    RAISES `DaemonError`, and RELEASES ITS OWN CLAIM, when either step after the claim fails:

    - `spawn` itself raises (the executable is missing, `fork` fails, ...) — nothing is
      running, so the claim is simply released and the reason is reported.
    - `spawn` SUCCEEDS but `_write_record` cannot persist the real pid — the trickier case,
      because a live process now exists that the claim does not correctly describe. Leaving
      the placeholder in place would misrepresent the CALLER as the daemon; releasing the
      claim without stopping the process would let it keep running untracked, and the next
      `start()` would then be free to spawn a second one — the one outcome the design calls
      impossible (`O_EXCL` only prevents two CONCURRENT claims, not a claim that never
      correctly recorded what it started). So the spawned process is killed and the claim is
      released: a start that could not be recorded is a failed start, not a silent orphan.

    Called from inside a long-lived host process (hermes' `initialize()`), not only from a
    short CLI invocation — so this cannot rely on "the caller will exit soon and the stale
    claim will self-heal", which is what an ephemeral CLI process gets for free.
    """
    if not _claim():
        existing = record()

        return {"action": "already", "pid": existing.get("pid", 0) if existing else 0}
    launch = spawn or _spawn
    command = argv or [sys.executable, _qctx_path(), "repos", "daemon", "run"]
    try:
        pid = launch(command)
    except OSError as exc:
        _release_claim()
        raise DaemonError(f"could not start the daemon: {exc}") from exc
    starttime = lease.process_start(pid) or ""
    if not _write_record({"pid": pid, "starttime": starttime, "started_at": time.time()}):
        if _stop_and_confirm({"pid": pid, "starttime": starttime}, sleep=sleep):
            _release_claim()
            raise DaemonError(f"the daemon started (pid {pid}) but its record could not be "
                              f"written — state directory unavailable; the process was stopped "
                              f"rather than left running untracked")
        raise DaemonError(f"the daemon started (pid {pid}) but its record could not be written "
                          f"AND the process did not stop when asked; the claim is being kept so "
                          f"nothing starts a second daemon on top of it — kill {pid} by hand")

    return {"action": "started", "pid": pid}


def _release_claim() -> None:
    try:
        path().unlink()
    except OSError:
        pass


def _stop_and_confirm(entry: dict, timeout_s: float = 2.0, poll_s: float = 0.05,
                      sleep=time.sleep) -> bool:
    """Signals the process named by `entry` and returns True only once it is CONFIRMED gone.

    WHY CONFIRMATION AND NOT A BARE `os.kill`. Releasing the claim is what lets the next
    `start()` spawn — so releasing it on an unconfirmed kill is exactly how a second daemon
    gets spawned on top of a first one that is still alive, which the spec calls impossible.
    A signal that raises (the process is already gone) and a signal that lands but is ignored
    look identical to the caller; only re-reading `(pid, starttime)` tells them apart, which is
    the same test the leases and `stop()` use. A timeout that runs out returns False and the
    caller KEEPS the claim: a stale claim blocks new daemons, while a released one over a live
    process multiplies them, and of those two failures only the first is recoverable by waiting.
    """
    try:
        os.kill(int(entry["pid"]), 15)
    except (OSError, ValueError, KeyError):
        pass                                          # already gone, or never startable
    deadline = time.monotonic() + timeout_s
    while lease.alive(entry):
        if time.monotonic() >= deadline:
            return False
        sleep(poll_s)

    return True


def _claim() -> bool:
    """Exclusively creates the record file. True only for the caller that wins the race.

    A `FileExistsError` means someone got there first — but "someone" might be a daemon that
    crashed without cleaning up, and `record()` already knows how to tell the two apart: it
    reads back as an entry only when the process it names is still alive. So on a collision we
    ask `record()`. An entry means the file is genuinely held — return False, not ours. `None`
    means the file is a corpse left by a dead daemon, and leaving it there would jam every future
    `start()` on one crash forever, so we remove it and retry the claim EXACTLY ONCE. We do not
    loop: a claim that fails twice means somebody else's `_claim()` won the newly-empty slot in
    the interval between our unlink and our retry, which is the correct outcome of the race, not
    a bug to spin around.
    """
    if _try_create():
        return True
    if record() is not None:
        return False
    try:
        path().unlink()
    except OSError:
        pass

    return _try_create()


def _try_create() -> bool:
    """Attempts the exclusive create. The placeholder written on success names THIS process —
    not the daemon `start()` is about to spawn — so a concurrent `_claim()` that collides with us
    while we are still between claiming and spawning reads back an alive entry (this process) and
    correctly backs off, instead of mistaking our in-progress claim for a stale one and tearing
    it out from under us. `_write_record` overwrites it with the real pid once spawning succeeds.
    """
    try:
        path().parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path(), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError:
        return False
    try:
        os.write(fd, json.dumps({"pid": os.getpid(),
                                 "starttime": lease.process_start(os.getpid()) or ""},
                                sort_keys=True).encode("utf-8"))
    finally:
        os.close(fd)

    return True


def stop(timeout_s: float = 2.0, poll_s: float = 0.05, sleep=time.sleep) -> bool:
    """Asks the running daemon to end, and WAITS for it to actually be gone before releasing
    the claim. False when there was none, or when the process did not die within `timeout_s`.

    UNLINKING RIGHT AFTER THE SIGNAL — an earlier version did exactly that — freed the claim
    while the process could still be mid-shutdown, so `stop()` immediately followed by
    `start()` (as `add-all` does) could spawn a SECOND live daemon before the first one had
    actually exited. Polling the same `(pid, starttime)` test the leases use closes that
    window: the claim is released only once the process is confirmed gone, comparing
    `starttime` too so a pid recycled during the wait is not mistaken for the daemon still
    running. A timeout that runs out KEEPS the claim rather than guessing — a stale "running"
    that turns out to be true is safer than a second daemon spawned onto a live first one.
    """
    entry = record()
    if not entry:
        return False
    try:
        os.kill(int(entry["pid"]), 15)
    except (OSError, ValueError, KeyError):
        return False
    deadline = time.monotonic() + timeout_s
    while lease.alive(entry):
        if time.monotonic() >= deadline:
            return False
        sleep(poll_s)
    try:
        path().unlink()
    except OSError:
        pass

    return True


def run(work, *, cycles: int | None = None, sleep=time.sleep, watch=None) -> str:
    """THE LOOP. Returns why it stopped.

    Each cycle, in this order and for these reasons:

      1. no living lease → exit. Checked FIRST so a daemon whose hosts are gone does not start
         one more job before noticing.
      2. reap jobs left `running` by a daemon that died, so `status` never shows stalled
         progress as activity.
      3. run one pending job, if any. One at a time: two workers on one repository duplicate
         the work without finishing sooner.
      4. otherwise watch, which is where changed files become new jobs.

    A worker that raises marks its job failed and the loop CONTINUES — one broken repository
    must not end the daemon for the others. `watch` gets the SAME survival guarantee: it has no
    single job to mark failed, but letting it propagate would end indexing for every repository
    being watched over one that could not enqueue, not just the one that failed.
    """
    seen = 0
    while cycles is None or seen < cycles:
        if not lease.live():
            return "no live lease"
        jobs.reap(lambda pid: lease.process_start(pid) is not None)
        job = jobs.next_pending()
        if job is not None:
            _run_one(job, work)
        elif watch is not None:
            try:
                watch()
            except Exception:                            # noqa: BLE001 — see the docstring
                # A watcher that cannot enqueue must not end the daemon for every OTHER
                # repository. There is no job here to mark failed, so the loop simply carries
                # on: the change is still on disk, the next cycle sees it again, and a file
                # whose reindex never happens keeps showing `[stale]` in search — which is
                # where the user actually notices, not in a daemon log nobody is watching.
                pass
        seen += 1
        if cycles is None or seen < cycles:
            sleep(CYCLE_S)

    return "cycles exhausted"


def _run_one(job: dict, work) -> None:
    repo = job["repo"]
    if jobs.cancel_requested(repo):
        jobs.update(repo, state=jobs.CANCELLED)

        return
    jobs.update(repo, state=jobs.RUNNING, daemon_pid=os.getpid(), error="")
    try:
        work(job)
    except Exception as exc:                        # noqa: BLE001 — see the docstring of `run`
        jobs.update(repo, state=jobs.FAILED, error=f"{type(exc).__name__}: {exc}"[:400])

        return
    if jobs.cancel_requested(repo):
        jobs.update(repo, state=jobs.CANCELLED)

        return
    jobs.update(repo, state=jobs.DONE, current="")


def _write_record(entry: dict) -> bool:
    """Writes the daemon record. Returns True on success, False on failure — checked by
    `start()`, which cannot afford to treat "wrote" and "did not" the same way `jobs._write`
    can, because the process behind a failed write is still running."""
    try:
        path().parent.mkdir(parents=True, exist_ok=True)
        tmp = path().with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(entry, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path())

        return True
    except OSError:
        return False


def _spawn(argv: list[str]) -> int:
    """Launches `argv` fully detached, so it survives the terminal that started it."""
    out = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           stdin=subprocess.DEVNULL, start_new_session=True)

    return out.pid


def _qctx_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                        "cli", "qctx.py")
