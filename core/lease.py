"""Who is still using the daemon.

A lease is a note saying "I am alive": the HOST process's pid and the moment it started. The
daemon checks the notes each cycle and exits when none is left. The user asked for exactly
this — quoted from the request that produced it, "o daemon deve ser morto quando o
claude ou hermes sai/morre": the daemon must be killed when claude or hermes exits or dies.

WHY THE TEST IS THE PROCESS AND NOT A MESSAGE. A host that exits cleanly could tell us; a host
killed with -9 cannot. Checking whether the process still exists covers both with one mechanism,
and the one it covers worse — a machine that lost power — leaves a stale file that the next
sweep removes anyway.

WHY `(pid, starttime)` AND NOT THE PID ALONE. The system reuses process numbers. With the pid
alone, an unrelated process that inherited the number would hold the daemon open forever, and
the failure would be invisible: nothing breaks, the daemon merely never exits. `starttime` is
field 22 of `/proc/<pid>/stat` and is stable for the life of a process.

PLATFORM: reading `/proc` is Linux. Where it is absent, `process_start` returns None and every
lease reads as dead, so the daemon exits rather than lingering — the safe direction. Adding a
portable process table would mean a dependency, which this project refuses for a reason.
"""
import json
import os
from pathlib import Path

from .knobs import state_dir
from .names import safe


def dir() -> Path:                                  # noqa: A001 — the name says what it holds
    return state_dir() / "leases"


def process_start(pid: int) -> str | None:
    """Field 22 of `/proc/<pid>/stat`, or None when the process is gone or unreadable.

    Parsed from the LAST `)` because the second field is the executable name in parentheses and
    may itself contain spaces or parentheses — splitting the whole line on whitespace is the
    classic way to read this file wrong.

    A ZOMBIE COUNTS AS GONE. A process that has exited but whose parent has not reaped it keeps
    a `/proc/<pid>/stat` with its original `starttime`, so reading the file alone cannot tell
    "still running" from "exited a while ago" — and every caller here means the first. That gap
    is reachable in ordinary use, not just in theory: the daemon is spawned as a child of the
    process that signals it, so unless that parent also waits, a daemon killed successfully
    reads as alive forever. State (field 3) is the only thing in the file that distinguishes
    them, so `Z` is reported the same way a missing file is.
    """
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8", errors="replace") as fh:
            after = fh.read().rsplit(")", 1)[1]
    except (OSError, ValueError, IndexError):
        return None
    fields = after.split()
    if fields and fields[0] == "Z":
        return None

    return fields[19] if len(fields) > 19 else None


def find_host_pid(names: tuple = ("claude", "hermes")) -> tuple[int, str] | None:
    """The nearest ancestor whose executable name is one of `names`, as `(pid, starttime)`.

    On claude-code a hook is a subprocess — measured: `python3 -> bash -> claude -> bash ->
    ptyxis` — so the host is up the tree. On hermes the provider runs INSIDE the host, and the
    caller passes `os.getpid()` instead of calling this.
    """
    pid = os.getpid()
    for _ in range(12):                             # deep enough for any real tree, bounded
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
            name = raw.split("(", 1)[1].rsplit(")", 1)[0]
            ppid = int(raw.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return None
        if name in names:
            start = process_start(pid)

            return (pid, start) if start else None
        if ppid <= 1:
            return None
        pid = ppid

    return None


def write(session_id: str, host: str, pid: int | None = None) -> dict:
    """Records that `session_id` on `host` is alive. Returns the entry it wrote.

    `pid` defaults to this process, which is right for hermes — the provider IS the host. On
    claude-code the caller resolves the host with `find_host_pid` first.
    """
    pid = os.getpid() if pid is None else int(pid)
    entry = {"session_id": session_id, "host": host, "pid": pid,
             "starttime": process_start(pid) or "", "written_at": _now()}
    try:
        dir().mkdir(parents=True, exist_ok=True)
        path = dir() / f"{safe(session_id)}.json"
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(entry, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        # A lease that cannot be written must not break a session start. The cost is a daemon
        # that exits sooner than it needed to, which is the safe direction.
        pass

    return entry


def alive(entry: dict) -> bool:
    """Whether the process this lease names is still the process it named."""
    if not isinstance(entry, dict):
        return False
    try:
        pid = int(entry.get("pid"))
    except (TypeError, ValueError):
        return False
    recorded = entry.get("starttime")
    if not recorded:
        return False

    return process_start(pid) == recorded


def live() -> list[dict]:
    """Every living lease, removing the files of the dead ones on the way through."""
    found = []
    try:
        paths = sorted(dir().glob("*.json"))
    except OSError:
        return []
    for path in paths:
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except OSError:
            # A transient read error (permission, I/O, EMFILE) is not a dead lease.
            # Skip it for this cycle but keep the file — the next cycle may succeed.
            continue
        except ValueError:
            # Corrupt JSON is unusable; removing it is right.
            entry = None
        if entry is not None and alive(entry):
            found.append(entry)
            continue
        try:
            path.unlink()
        except OSError:
            pass

    return found





def _now() -> float:
    import time

    return time.time()
