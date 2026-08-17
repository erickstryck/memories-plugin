"""Which repository is THIS working copy, locally and without asking twice.

Identity here is DECLARED, not derived, and that is a user decision from 2026-08-17: he keeps
parallel clones of the same remote on purpose, and worktrees. Deriving identity from the
remote would merge working copies that can sit on different branches, so the plugin OFFERS a
choice and remembers the answer. This module prepares and remembers; the host adapter asks.

The binding is keyed by absolute path, so MOVING a checkout loses it. That is not a leak: the
next detection asks again, the remote match makes "join the existing repo" the default, and
the archive is reached again instead of orphaned. It heals.

State lives beside the other plugin state so both hosts share it, and a corrupt file reads as
"no bindings" rather than raising — a broken cache must not make the tool unusable.
"""
import json
import os
import re
import subprocess

from .knobs import state_dir

FILENAME = "repo-bindings.json"


def _path() -> str:
    """The binding file, under the ONE state directory this plugin has.

    `state_dir` is imported and not redefined: it already had an owner, and a third copy of
    "where does state live" is how the three of them start disagreeing.
    """
    root = state_dir()
    os.makedirs(root, exist_ok=True)

    return os.path.join(root, FILENAME)


def _load() -> dict:
    try:
        with open(_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}

    return data if isinstance(data, dict) else {}


def _save(data: dict) -> None:
    tmp = _path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    os.replace(tmp, _path())


def get(checkout: str) -> str | None:
    return _load().get(os.path.realpath(checkout)) or None


def bind(checkout: str, repo: str) -> None:
    data = _load()
    data[os.path.realpath(checkout)] = repo
    _save(data)


def forget_repo(repo: str) -> list[str]:
    """Unbinds every checkout of `repo` and names them, so the caller can say what changed."""
    data = _load()
    freed = [path for path, name in data.items() if name == repo]
    for path in freed:
        del data[path]
    if freed:
        _save(data)

    return freed


def git_root(path: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None

    return out.stdout.strip() or None if out.returncode == 0 else None


def remotes_of(root: str) -> list[str]:
    try:
        out = subprocess.run(["git", "-C", root, "remote", "-v"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    urls = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] not in urls:
            urls.append(parts[1])

    return urls


def normalize_remote(url: str) -> str:
    """`git@github.com:me/alpha.git` and `https://github.com/me/alpha` are the same repo.

    Without this the join offer misses and the user is asked to name a repository that is
    already registered under the other URL form.
    """
    u = (url or "").strip()
    u = re.sub(r"^[a-z+]+://", "", u)
    u = re.sub(r"^[^@/]+@", "", u)
    u = u.replace(":", "/", 1) if "/" not in u.split(":", 1)[0] else u
    u = re.sub(r"\.git$", "", u)

    return u.strip("/").lower()


def slug_for(name: str) -> str:
    """A stable id from a human name: the FILTER KEY, so it may never change afterwards."""
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower())

    return s.strip("-") or "repo"
