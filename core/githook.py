"""Installs a git `post-commit` hook that refreshes a repository archive after each commit.

WHY A GIT HOOK AND NOT A WATCHER. This plugin has no daemon and no long-running process, by
design — `core/docs.py` states the rule — and a filesystem watcher would need a dependency, which
inside a hook means a silent loss of functionality the day it is missing. A commit is the event
that actually matters for an indexed repository, git already fires one, and the hook costs nothing
while nothing is committed.

WHAT THE HOOK DOES, and this is the whole of it: reads the paths of the commit that just happened
(`git diff-tree`) and starts a detached `qctx repos add` for them. It does not commit, does not
stage, does not push, does not touch the working tree. The only thing it writes is the archive.

WHY IT RUNS DETACHED. Chosen by the user on 2026-08-18 over a synchronous version: a commit never
waits, whatever its size. The objection to it — that a failure becomes a log line nobody reads —
is answered by the archive itself rather than by the log: a file whose reindex failed comes back
marked `[stale]` in the next search, so the signal reappears at the point of use. The log is for
the reason, not for the alarm.

WHY IT CANNOT FAIL A COMMIT. `exit 0` unconditionally. A hook that can reject a commit is a hook
that gets deleted the first time it is wrong, and this one is not important enough to cost anybody
a commit.

WHY IT NEVER OVERWRITES. A `post-commit` may already belong to husky, pre-commit, or the user.
Writing over another tool's file to install a convenience is worse than not installing: the
install refuses and prints the line to add by hand.
"""
import os
import shutil
import stat
import subprocess

from .errors import CoreError

#: Written into the hook so a later install can recognise its own work — and only its own.
MARKER = "# installed by memories-plugin (qctx repos install-hook)"

TEMPLATE = """\
#!/bin/sh
{marker}
# Reindexes the files of this commit into the {repo!r} archive, detached, so the commit never
# waits. Never fails a commit: the last line is `exit 0` whatever happened above.
# `--root` because a repository's FIRST commit has no parent to diff against, and
# without it that commit lists nothing — the one commit where everything is new.
# Measured; on any later commit the flag changes nothing.
changed=$(git diff-tree --no-commit-id --name-only --diff-filter=ACMR -r --root HEAD 2>/dev/null)
[ -z "$changed" ] && exit 0
(
  cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)" 2>/dev/null || exit 0
  # shellcheck disable=SC2086
  {qctx} repos add {repo} $changed
) >> "{log}" 2>&1 &
exit 0
"""


class HookError(CoreError):
    """The hook could not be installed. Never raised for "it is already there"."""


def git_root(path: str) -> str:
    """The working copy `path` belongs to, or "" when it is not inside one."""
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""

    return out.stdout.strip() if out.returncode == 0 else ""


def qctx_command() -> str:
    """How the hook should invoke this plugin.

    An absolute path when `qctx` is on PATH, because a git hook does NOT inherit an interactive
    shell's PATH — it runs with whatever git was started with, and `qctx` alone would resolve on
    a terminal and silently not resolve from an editor's git integration.
    """
    found = shutil.which("qctx")

    return found or "qctx"


def hook_line(repo: str) -> str:
    """The one line to paste into an existing hook, when we refuse to write over it."""
    return (f'git diff-tree --no-commit-id --name-only --diff-filter=ACMR -r --root HEAD '
            f'| xargs -r {qctx_command()} repos add {repo} &')


def install(repo: str, path: str, force: bool = False) -> dict:
    """Writes the hook for `repo` into the working copy containing `path`.

    Returns `{"action": "installed" | "already" | "refused", ...}` and raises only when the
    request cannot be answered at all — not inside a repository, or the hooks directory is
    unwritable.
    """
    if not repo:
        raise HookError("a repository name is required")
    root = git_root(path)
    if not root:
        raise HookError(f"{path} is not inside a git working copy")
    hooks_dir = os.path.join(root, ".git", "hooks")
    hook = os.path.join(hooks_dir, "post-commit")
    log = os.path.join(os.path.expanduser("~"), ".memories-plugin", "state", "refresh.log")

    if os.path.exists(hook):
        existing = _read(hook)
        if MARKER not in existing:
            return {"action": "refused", "hook": hook, "root": root,
                    "line": hook_line(repo)}
        if not force and f"repos add {repo} " in existing:
            return {"action": "already", "hook": hook, "root": root}

    body = TEMPLATE.format(marker=MARKER, repo=repo, qctx=qctx_command(), log=log)
    try:
        os.makedirs(hooks_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(hook, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.chmod(hook, os.stat(hook).st_mode | stat.S_IXUSR | stat.S_IXGRP)
    except OSError as exc:
        raise HookError(f"the hook could not be written to {hook}: {exc}") from exc

    return {"action": "installed", "hook": hook, "root": root, "log": log}


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""
