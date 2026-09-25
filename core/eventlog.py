"""Appending a timestamped line to a log in the state directory. One owner, three writers.

WHY A MODULE. The claude-code recall hook was the only thing that ever logged, with its file
handling inline. The hermes provider, which is the host used every day, logged nothing, and
the daemon's output went to /dev/null, so the two places where things actually go wrong left
no trace. Giving each of them its own copy of "open, rotate, append" would be three copies of
a rule this repo has already had to fix twice (the file mode, and rotation throwing it away).

WHAT IT PROMISES, and it is the rule `core/statefile.py` sets for state: writing NEVER raises
and answers whether it landed. A log is a convenience; losing a line must never cost a recall
or a daemon cycle.

WHY NOT `logging`. The stdlib logger is process-global configuration, and two of the three
writers run inside a host (hermes) that has its own. Its rotating handler also publishes numbered
files at the umask, which is the mode leak `statefile` exists to prevent.
"""
import os
import time
from pathlib import Path

from . import statefile
from .knobs import state_dir

RECALL = "recall.log"
DAEMON = "daemon.log"

#: Past this a log is cut to its newest half. The hook has used this ceiling since the log
#: existed, so an existing `recall.log` keeps the size it has always had.
MAX_BYTES = 256 * 1024


def path(name: str) -> Path:
    return state_dir() / name


def rotate(target: Path, max_bytes: int = MAX_BYTES) -> bool:
    """Halves `target` when it has grown past `max_bytes`. True when it was rotated.

    THE TAIL IS KEPT, NOT THE HEAD: what a reader wants from a rotated log is what happened
    most recently. The rewrite goes through `statefile`, because rotating creates a NEW file
    and a hand-rolled write takes the umask for it (0o664 measured, for a log that records what
    was recalled and when).
    """
    try:
        if not target.exists() or target.stat().st_size <= max_bytes:
            return False
        statefile.write_text(target, target.read_text(errors="replace")[-max_bytes // 2:])

        return True
    except OSError:
        return False


def write(name: str, line: str) -> bool:
    """Appends `line` to the log `name`, stamped with the local time. True when it landed.

    ONE LINE, ALWAYS. The message is flattened first, because some of what gets logged is user
    text (a prompt) and a newline in it would read back as a line of its own, which a summary
    would then count as an event that never happened.
    """
    flat = " ".join(str(line).split())
    try:
        target = path(name)
        statefile.ensure_dir(target.parent)
        rotate(target)
        # O_CREAT WITH A MODE, because `open(..., "a")` creates at the umask. The mode only
        # applies to a file being created; an existing one keeps its own, which is the same
        # "a directory you already chose is yours" rule `statefile.ensure_dir` follows.
        fd = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, f"{time.strftime('%Y-%m-%d %H:%M:%S')} {flat}\n".encode("utf-8"))
        finally:
            os.close(fd)

        return True
    except Exception:                                   # noqa: BLE001 (see the docstring)
        return False
