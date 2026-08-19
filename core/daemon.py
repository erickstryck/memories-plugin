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
from .knobs import state_dir

#: How long the loop sleeps between cycles. Short enough that a cancel or a file change is
#: noticed while the user is still looking at the screen; long enough to be free.
CYCLE_S = 5.0


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


def start(spawn=None, argv: list[str] | None = None) -> dict:
    """Starts the daemon if none is running. `{"action": "started" | "already", "pid": int}`.

    `spawn` is injected for tests; by default it launches a detached `qctx repos daemon run`.
    """
    existing = record()
    if existing:
        return {"action": "already", "pid": existing.get("pid", 0)}
    launch = spawn or _spawn
    command = argv or [sys.executable, _qctx_path(), "repos", "daemon", "run"]
    pid = launch(command)
    _write_record({"pid": pid, "starttime": lease.process_start(pid) or "",
                   "started_at": time.time()})

    return {"action": "started", "pid": pid}


def stop() -> bool:
    """Asks the running daemon to end. False when there was none."""
    entry = record()
    if not entry:
        return False
    try:
        os.kill(int(entry["pid"]), 15)
    except (OSError, ValueError, KeyError):
        return False
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
    must not end the daemon for the others.
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
            watch()
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


def _write_record(entry: dict) -> None:
    try:
        path().parent.mkdir(parents=True, exist_ok=True)
        tmp = path().with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(entry, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path())
    except OSError:
        pass


def _spawn(argv: list[str]) -> int:
    """Launches `argv` fully detached, so it survives the terminal that started it."""
    out = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           stdin=subprocess.DEVNULL, start_new_session=True)

    return out.pid


def _qctx_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                        "cli", "qctx.py")
