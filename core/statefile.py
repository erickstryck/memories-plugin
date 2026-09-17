"""Publishing a JSON file so a reader never sees it half-written.

WHY A MODULE AND NOT A LINE IN EACH CALLER. Six places in `core/` write state this way, and
five of them had already learned to name the temporary after the process. The sixth,
`core/bindings.py`, kept a fixed `.tmp` — and `core/windowcache.py` names that gap in its own
comment without closing it, which is what a rule re-typed in six places eventually looks like.

TWO DIFFERENT PROTECTIONS, and it is worth not confusing them, because measuring told me I
had. The pid in the temporary's name stops two writers from sharing one file. What stopped the
crash `bindings` actually suffered is narrower: it held the file OPEN across the `json.dump`,
so when another process published first, the still-open descriptor was writing into a path
that no longer existed. Measured with six concurrent processes binding 40 checkouts each: 85
raised `FileNotFoundError` before, 0 after — and 0 as well with the pid deliberately removed
from the name, which is how I learned that write-then-replace, not the name, was carrying the
fix. Both are kept: the name because the race is real even when it currently costs nothing
observable, the single write because it is what the measurement credits.

WHAT IT DOES NOT PROMISE. `os.replace` makes PUBLICATION atomic — a reader sees the old file or
the new one, never a torn one. It does NOT make read-modify-write atomic: two processes that
both load, both edit and both write still end with the last writer's version, and the other's
edit is gone (measured: 21 of 240 entries surviving). Fixing THAT needs a lock, which is a
different change with a different cost, and pretending otherwise here would be the more
dangerous kind of wrong. The loss is why `jobs` writes one file per repository rather than one
file for all of them.

WHY IT RETURNS A BOOL AND NEVER RAISES. Every caller is writing state that exists to make the
next run cheaper: a lease, a queued job, a quarantine record, a cached window. None of them is
worth failing a user's command over, and each already had this same `except OSError: pass`
around its own copy. Returning the outcome lets a caller that DOES care — `jobs.enqueue`, where
losing the write means work that never happens — raise its own error with its own words.
"""
import json
import os
from pathlib import Path


def write_text(path, text: str, *, make_parents: bool = True) -> bool:
    """Writes `text` to `path` with the same mode `write_json` publishes. True when it landed.

    WHY THIS EXISTS BESIDE `write_json`. The mode is the part of the publish that every state
    file needs, and five writers -- a breaker stamp, a session's recall map, a checkpoint
    counter, a rotated log, the config file -- were each hand-rolling `Path.write_text` and
    taking whatever the umask gave them. MEASURED on the real machine: 105 of 107 files in
    the state directory were published 0o664 under the common 0o002 umask, and the two that
    were 0o600 were exactly the two that came through `write_json`. Among the group-readable
    ones are `recall-<session>.json`, which hold the ids of the memories injected into each
    session.

    IT IS ATOMIC FOR THE SAME REASON `write_json` IS, not because every caller needs it. A
    counter or a log has no torn-read consequence worth a rename -- but the staging is what
    makes the MODE a property of a file being created, and a file created fresh is the only
    one `os.open(..., 0o600)` can set a mode on. `O_CREAT | O_TRUNC` over an existing file
    keeps that file's permissions, which is precisely how the careful mode gets lost on the
    second write. Taking the rename as well costs one syscall and removes that whole class.

    NOT A `write_json` WRAPPER, AND NOT ITS PARENT. `write_json` serialises and this does not;
    routing one through the other would mean either a text writer that JSON-encodes its
    counter or a JSON writer whose payload arrives pre-encoded, and both hide what the caller
    asked for. They share the private publish below instead, which is where the mode and the
    staging actually live.
    """
    return _publish(path, text.encode("utf-8"), make_parents=make_parents)


def write_json(path, payload, *, make_parents: bool = True) -> bool:
    """Writes `payload` to `path` atomically. True when it landed.

    The temporary carries this process's pid, which is what makes concurrent writers safe.
    """
    return _publish(path, json.dumps(payload, indent=1, sort_keys=True).encode("utf-8"),
                    make_parents=make_parents)


def _publish(path, data: bytes, *, make_parents: bool = True) -> bool:
    """Stages `data` and renames it over `path`, owner-readable only. True when it landed.

    ONE OWNER FOR THE MODE AND THE RENAME, so a policy change is one edit rather than one per
    serialisation format. `write_json` and `write_text` differ only in how they turn a caller's
    value into bytes.
    """
    target = Path(path)
    tmp = None
    try:
        if make_parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(f"{target.suffix}.{os.getpid()}.tmp")
        # THE MODE IS SET HERE AND NOT LEFT TO THE UMASK. These files name pids and paths
        # this user controls; `os.replace` carries the TEMPORARY's mode onto the target, so
        # whatever the temporary was created with is what gets published.
        #
        # MEASURED, and it is why this is not cosmetic: `core/daemon.py::_claim` opens its
        # claim 0o600, and the daemon's first `record()` came straight back through here and
        # republished the same file 0o664 under the usual 0o002 umask. The careful mode on
        # the claim lasted exactly until the first save. Creating the temporary with 0o600
        # fixes every caller at once, which is the point of this module owning the write.
        #
        # O_EXCL, AND THE UNLINK BEFORE IT, BECAUSE THE MODE ONLY APPLIES TO A FILE BEING
        # CREATED. `O_CREAT | O_TRUNC` over a temporary left behind by an earlier crash of
        # this same pid — the name is deterministic, so that file is ours — reuses ITS
        # permissions, and `os.replace` then publishes those. Measured: a stale temporary at
        # 0o666 published the daemon's claim at 0o666, which is exactly the leak the comment
        # above says this closes. Removing it first and refusing to reuse it makes the mode
        # a property of what we publish rather than of what a crash left lying around.
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            _write_all(fd, data)
        finally:
            os.close(fd)
        os.replace(tmp, target)

        return True
    except OSError:
        # A temporary left behind is noise, not corruption: it carries our pid, so no other
        # writer will adopt it, and the next successful write publishes over the target
        # regardless. Removing it is courtesy, and failing to remove it must not become the
        # error that this function exists to avoid raising.
        #
        # THE TARGET IS NEVER TOUCHED ON THIS PATH, and that is the point of staging: a write
        # that failed leaves the PREVIOUS state published, which every reader here can still
        # use, rather than a half-written file that parses for nobody.
        try:
            if tmp is not None:
                os.unlink(tmp)
        except OSError:
            pass

        return False


def _write_all(fd: int, data: bytes) -> None:
    """Writes every byte of `data`, or raises. `os.write` alone does NOT promise this.

    A SHORT WRITE IS NOT AN ERROR, which is what makes it dangerous here. `os.write` may
    consume fewer bytes than it was given and return that count, raising nothing — so the
    `except OSError` above sees a completely successful call, `os.replace` publishes a
    truncated document, and `write_json` returns True. Every caller is then told the state
    landed: `jobs.enqueue`, whose docstring promises to raise rather than let "work that
    never happens" be reported as queued, raised nothing; `core/bindings.py::_save`, whose
    comment says the failure "IS RE-RAISED, and that is the contract these callers are built
    on", re-raised nothing.

    MEASURED with `RLIMIT_FSIZE` at 2048 bytes and SIGXFSZ ignored, against a 5012-byte
    payload: the single `os.write` this replaced returned 2048, `write_json` returned True,
    and reading the published file raised `Unterminated string starting at: line 2 column 7`.
    The version this module replaced used `Path.write_text`, which loops internally — so the
    move to `os.write` for the sake of the file mode is what introduced the defect, and this
    loop is what pays for that mode without giving up the guarantee.

    A `write` that returns 0 would spin forever, so it is treated as the failure it is: the
    file cannot take the bytes, and saying so is the honest answer.
    """
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError(f"the write stopped after {len(data) - len(view)} of "
                          f"{len(data)} bytes")
        view = view[written:]
