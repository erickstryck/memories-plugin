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
import hashlib
import json
import os
import time
from pathlib import Path

from . import names
from .knobs import state_dir


#: The key under which a record file names its own repository. Prefixed so it can never
#: collide with a path: every real key is an absolute path, and `names.safe` is lossy, so the
#: FILENAME cannot be reversed into the name. Without this, a held file was invisible in
#: `repos status` for any repository that had no job row — see `repos_on_record`.
_REPO_KEY = "//repo"


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
    if not isinstance(found, dict):
        return {}

    # EVERY VALUE IS VALIDATED, not just the top level. A single malformed entry used to
    # raise AttributeError out of `held`, which runs inside the watcher — and `daemon.run`
    # swallows a raising watcher, so indexing stopped for EVERY repository on the machine,
    # silently and permanently. `repos status` raised on the same entry, so the one command
    # that could have diagnosed it was the one that died. Dropping the bad entry degrades to
    # "retry that file", which is this module's safe direction.
    #
    # `_REPO_KEY` is filtered out here so no caller ever sees the bookkeeping as a path: the
    # readers iterate this mapping expecting one entry per FILE, and a stray key would be
    # stat'ed, displayed and released like one.
    return {path: meta for path, meta in found.items()
            if path != _REPO_KEY and isinstance(meta, dict)}


def _digest(path: str) -> str | None:
    """SHA-1 of the file's bytes, or None when it cannot be read.

    WHY A HASH AND NOT JUST (mtime, size). This module's first paragraph promises a record
    keyed by CONTENT, and `docs/usage.md` repeats it to the user. Metadata is not content, and
    it was wrong in both directions: a `touch`, a `git checkout` that rewrites mtimes or a
    `cp -p` restore released a file whose bytes never changed — back into the very loop this
    module exists to break — while an in-place fix that preserved length and mtime left the
    repaired file held forever, with no command the user was told to run.

    IT IS AFFORDABLE HERE, which is why `repos._changed_paths` may NOT do the same thing. That
    one stats every TRACKED file every ~5 s; this one reads only the files already HELD, and a
    quarantine large enough for the hash to matter is a repository with a bigger problem.
    Measured on the 22-file case that motivated the feature, at 200 KB each: 0.02 ms to stat
    them against 2.08 ms to hash them, inside a 16 ms cycle budget.
    """
    try:
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                h.update(block)

        return h.hexdigest()
    except OSError:
        return None


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
                        "digest": _digest(path), "at": time.time()}
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
            # METADATA AGREES, BUT IT IS NOT THE PROMISE. An in-place fix that preserves both
            # — an editor that restores the mtime, a same-length correction — left the
            # repaired file held forever, and nothing told the user a command existed. When a
            # digest was recorded it decides; without one, metadata is all there is.
            recorded = meta.get("digest")
            if not recorded or recorded == _digest(path):
                still.add(path)
                continue
            del entry[path]
            changed = True
            continue
        # THE DIGEST OVERRULES THE METADATA, in the one direction metadata can be wrong about
        # content: mtime or size moved, but the bytes did not. A `touch`, a `git checkout` that
        # rewrites mtimes, a `cp -p` restore — each of those released a file that had not
        # changed at all, putting it back into the loop this module exists to break.
        #
        # ONLY WHEN A DIGEST WAS RECORDED. Entries written before this existed carry none, and
        # for them the metadata comparison above stays the whole answer — releasing every old
        # record in a batch the first time the new code runs would re-queue exactly the files
        # the quarantine was holding.
        recorded = meta.get("digest")
        if recorded and recorded == _digest(path):
            still.add(path)
            continue
        del entry[path]
        changed = True
    if changed:
        _write(repo, entry)

    return still


def clear(repo: str, paths) -> int:
    """Forgets `paths`, returning how many were dropped. Clearing something never held is not
    an error — the caller indexes a batch and releases all of it, without having to know which
    members had failed before."""
    return forget(repo, paths)


def forget(repo: str, paths=None) -> int:
    """Drops records on purpose, returning HOW MANY were dropped. `paths=None` drops them all.

    `None` IS THE ONLY WIPE. An empty list drops nothing, because "the caller named no paths"
    and "the caller wants everything gone" must not be the same value on a destructive call.
    A caller holding a possibly-empty list has to say `paths or None` deliberately, which is
    what `cmd_repos_quarantine_clear` does at its single point of translation.

    WHY A DELIBERATE RELEASE EXISTS AT ALL, when `held` already releases on a content change.
    That covers the file that was repaired; it cannot cover the file that was always fine and
    whose REASON went away — a server limit that was raised, a model that was swapped, an
    endpoint that was fixed. The content never changes in those cases, so nothing would ever
    let go, and the user would be left reading a `status` line with no way to act on it.

    IT RETURNS A COUNT BECAUSE THE CALLER HAS TO BE ABLE TO SAY WHAT HAPPENED. A command that
    prints "released" after releasing nothing is the class of small lie this project refuses
    everywhere else; `0` lets the caller say "there was nothing to release" instead. Which is
    why the count is what LANDED, not what was intended: `_write` returns False on OSError,
    and reporting the attempt would tell that same lie from the function written to refuse it.

    FORGETTING IS NOT AN EXEMPTION. It drops the record, and the next attempt decides afresh —
    a file that still cannot be indexed is simply held again. An allow-list would be a second
    policy living beside this one, and nobody asked for it.
    """
    entry = load(repo)
    wipe = paths is None
    if wipe:
        # A TOTAL DISCARD REACHES THE FILE EVEN WHEN IT READS AS EMPTY. `load` drops values it
        # cannot parse, so a wholly corrupt record looks like `{}` here — and stopping on that
        # would leave it on disk, hidden from `status` and refused by the only command offered
        # to repair it. Nothing is lost: unreadable entries hold no file anyone can act on.
        released, entry = len(entry), {}
    else:
        wanted = {str(p) for p in paths}
        kept = {path: meta for path, meta in entry.items() if path not in wanted}
        released, entry = len(entry) - len(kept), kept

    # Nothing on disk and nothing to drop: say so without writing. A wipe still falls through
    # when the file exists, so an unreadable record is thrown away rather than declined.
    if not released and not (wipe and _path(repo).exists()):
        return 0

    return released if _write(repo, entry) else 0


def repos_on_record() -> list[str]:
    """Every repository that HAS a quarantine file, by name.

    The caller cannot derive this itself: `names.safe` is lossy, so the filename does not
    reverse into a repository name. Reading the name back out of the record is what lets
    `repos status` show a held file for a SETTLED repository — one whose daemon has had
    nothing to do lately and therefore has no job row. That is exactly the repository whose
    files have been held longest, and it was the one place the count was invisible.

    THE STAMP IS NOT ASSUMED, for two reasons that both reach real users. A record written
    before the stamp existed has no key, and it never gains one on its own: `_write` is the
    only stamper and `held` rewrites only when it prunes, which a settled repository never
    does. Skipping those files would have left the very bug this function fixes in place for
    everyone who already had a quarantine. And a record touched by an OLDER build after this
    one drops the stamp again, so the gap reopens in a mixed-version fleet. The filename stem
    is the honest fallback: `names.safe` is lossy, so it is the name only when the name
    needed no escaping — but it beats naming nothing, and `load`/`held`/`forget` all work on
    such a record already.

    THE VALUE IS VALIDATED, like every value `load` did not write. `repos status` puts this
    list into a set; a dict or a list here raised `TypeError` out of the one command that
    could diagnose a bad record — which is the failure `load`'s own guard exists to prevent
    (see the comment there). A stamp that is not a name is ignored, not returned.
    """
    try:
        paths = sorted(dir().glob("*.json"))
    except OSError:
        return []
    found = []
    for path in paths:
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(entry, dict) or not entry:
            continue
        name = entry.get(_REPO_KEY)
        if not isinstance(name, str) or not name:
            name = path.stem
        found.append(name)

    return found


def _path(repo: str) -> Path:
    return dir() / f"{names.safe(repo)}.json"


def _write(repo: str, entry: dict) -> bool:
    """Writes atomically, the way `jobs._write` does. False on failure, never raises: state
    that cannot be written is state the reader will not find, which every caller handles.

    Stamps `_REPO_KEY` so the file can name its own repository — `load` filters it back out,
    so this is invisible to every reader. An EMPTY record is deleted rather than written as
    an empty object: a file that holds nothing should not make `repos_on_record` name a
    repository with nothing held.
    """
    try:
        dir().mkdir(parents=True, exist_ok=True)
        path = _path(repo)
        if not entry:
            path.unlink(missing_ok=True)

            return True
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({_REPO_KEY: repo, **entry}, indent=1, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, path)

        return True
    except OSError:
        return False
