"""Which files of a repository go into the archive.

WHY `git ls-files` AND NOT A WALK OF THE DISK. It makes `.gitignore` respected BY DEFINITION
rather than by a reimplementation of ours — that file has negation, precedence and per-directory
rules, and getting them subtly wrong would index a `build/` nobody wanted, silently. It also
means `node_modules`, caches and artefacts need no special rule: they are not tracked.

The consequence, stated rather than discovered: a file that exists but was never `git add`ed is
NOT indexed. That is the same rule, seen from the other side.

WHY FOUR DISCARDS AND NOT NONE. The archive answers questions; a minified bundle and a lockfile
answer none, and they are large enough to dominate every search of that repository. A binary
would be refused later by `_read_source` anyway — discarding it here just avoids paying the read.
"""
import os
import subprocess

#: A file larger than this becomes hundreds of chunks and dominates the repository's archive.
MAX_FILE_BYTES = 1_048_576

#: A single line longer than this is machine-written: minified bundles, embedded data URIs.
MINIFIED_LINE_CHARS = 2_000

#: Generated, enormous, and nobody searches them by meaning.
LOCKFILES = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json",
    "poetry.lock", "Pipfile.lock", "Cargo.lock", "composer.lock", "Gemfile.lock",
    "go.sum", "flake.lock", "uv.lock",
})

#: Read once per file, and enough to decide both "binary" and "minified".
_SNIFF_BYTES = 8192


def tracked_files(root: str) -> list[str]:
    """The repository's tracked paths, relative to `root`. Empty when `root` is not a repo."""
    try:
        out = subprocess.run(["git", "-C", root, "ls-files", "-z"],
                             capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []

    return [name for name in out.stdout.decode("utf-8", "replace").split("\0") if name]


def eligible(root: str, max_bytes: int = MAX_FILE_BYTES) -> dict:
    """`{"tracked": int, "eligible": [absolute paths], "skipped": {reason: count}}`.

    Paths come back ABSOLUTE: `RepoIndex.add_files` takes paths and the daemon runs from a
    different working directory, so a relative one would resolve against the wrong place —
    silently, because a missing file is merely skipped and reported.

    Every discard reason is present even at zero, so no consumer has to guess whether a key
    exists.
    """
    names = tracked_files(root)
    skipped = {"binary": 0, "minified": 0, "lockfile": 0, "too_big": 0, "unreadable": 0}
    keep = []
    for name in names:
        path = os.path.join(root, name)
        if os.path.basename(name) in LOCKFILES:
            skipped["lockfile"] += 1
            continue
        try:
            size = os.stat(path).st_size
        except OSError:
            skipped["unreadable"] += 1
            continue
        if size > max_bytes:
            skipped["too_big"] += 1
            continue
        reason = _sniff(path)
        if reason:
            skipped[reason] += 1
            continue
        keep.append(os.path.abspath(path))

    return {"tracked": len(names), "eligible": sorted(keep), "skipped": skipped}


def _sniff(path: str) -> str | None:
    """`"binary"`, `"minified"`, `"unreadable"` or None — decided from one read of the head.

    A NUL byte is the same test `_read_source` uses, applied earlier so an image is not read in
    full only to be refused. The line length is measured on the same buffer, so a minified file
    costs no extra I/O.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(_SNIFF_BYTES)
    except OSError:
        return "unreadable"
    if b"\0" in head:
        return "binary"
    # A head with no newline at all is one very long line, which is the minified case; a head
    # WITH newlines is judged by its longest complete line.
    longest = max((len(part) for part in head.split(b"\n")[:-1]), default=len(head))
    if longest > MINIFIED_LINE_CHARS:
        return "minified"

    return None
