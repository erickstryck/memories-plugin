"""Files that could not be indexed, remembered so they are not tried forever.

WHY THIS EXISTS. A file that fails to index is a file the archive has no chunk for, so the next
poll reports it as missing and queues it again — forever. Measured on 2026-09-14: the same 22
files were re-queued every ~40 s indefinitely, and the load that put on the shared embedding
endpoint pushed automatic recall from 0.04 s to 1.98 s against its 2.00 s ceiling. The user saw
"[automatic recall — UNAVAILABLE]", which names nothing about indexing.

IT REMEMBERS A CONTENT, NOT A PATH. Every entry carries the `mtime` and `size` the file had when
it failed, and `held` returns only the paths that still match. A 0-byte file that later gains
content is therefore retried on its own, with no command to run — which is the state 20 of those
22 files were in. A quarantine keyed by path alone would need a human to clear it, and nothing
would ever tell them to.

THE REASON IS OPAQUE DATA. It is whatever the failing layer said, stored verbatim and shown to
the user. This module never enumerates reasons: a new failure mode — a new server error, a new
format — must not require a change here.

SAME IDIOM AS `jobs.py` AND `lease.py`: a JSON file per repository, written atomically, with
OSError tolerated. There is no protocol between processes; the daemon writes, the CLI reads.
"""
import json
import os
import time
from pathlib import Path

from . import names
from .knobs import state_dir


def dir() -> Path:                                  # noqa: A001 — the name says what it holds
    return state_dir() / "quarantine"


def load(repo: str) -> dict:
    """`{path: {reason, mtime, size, at}}`, or `{}` when there is nothing readable.

    Corrupt or unreadable state reads as EMPTY, never raises: holding nothing means retrying a
    file we could have skipped, which is exactly today's behaviour. The opposite failure —
    claiming a file is handled when nothing knows that — is the one this plugin refuses.
    """
    try:
        found = json.loads(_path(repo).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    return found if isinstance(found, dict) else {}


def record(repo: str, path: str, reason: str) -> None:
    """Remembers that `path` could not be indexed, with the content it had when it failed.

    A path that cannot be `stat`ed is NOT recorded: with no mtime/size there is nothing to
    compare later, so the entry could never be released and would be a permanent exclusion
    written by a transient error.
    """
    try:
        st = os.stat(path)
    except OSError:
        return
    entry = load(repo)
    entry[str(path)] = {"reason": str(reason), "mtime": st.st_mtime, "size": st.st_size,
                        "at": time.time()}
    _write(repo, entry)


def held(repo: str) -> set:
    """The paths whose content STILL matches what failed — the ones to skip.

    An entry whose file changed, or vanished, is not held and is dropped from disk on the way
    through: the next attempt is the point, and a stale entry that no longer describes anything
    is just noise in what the user reads.
    """
    entry = load(repo)
    still, changed = set(), False
    for path, meta in list(entry.items()):
        try:
            st = os.stat(path)
        except OSError:
            del entry[path]
            changed = True
            continue
        if st.st_mtime == meta.get("mtime") and st.st_size == meta.get("size"):
            still.add(path)
            continue
        del entry[path]
        changed = True
    if changed:
        _write(repo, entry)

    return still


def clear(repo: str, paths) -> None:
    """Forgets `paths`. Clearing something never held is not an error — the caller indexes a
    batch and releases all of it, without having to know which members had failed before."""
    entry = load(repo)
    removed = False
    for path in paths:
        if entry.pop(str(path), None) is not None:
            removed = True
    if removed:
        _write(repo, entry)


def _path(repo: str) -> Path:
    return dir() / f"{names.safe(repo)}.json"


def _write(repo: str, entry: dict) -> bool:
    """Writes atomically, the way `jobs._write` does. False on failure, never raises: state
    that cannot be written is state the reader will not find, which every caller handles."""
    try:
        dir().mkdir(parents=True, exist_ok=True)
        path = _path(repo)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(entry, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

        return True
    except OSError:
        return False
