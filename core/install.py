"""Checks for the plumbing the wizard installs: the launcher, the PATH, and the file a
process with no shell reads.

DATA, not text — the same contract as `core/setup.py`, and it returns that module's
`Check` so one renderer serves both.

HOST-NEUTRAL, deliberately. `core/setup.py` states the rule and the reason; this module
obeys it. Which host is installed, and running its cutover, is the CLI's business, which
is where the two host names already appear in this repository.
"""
import filecmp
import shutil
from pathlib import Path

from .config import load
from .setup import COMMAND_PREFIX, Check

#: The command every skill and every page of documentation cites.
LAUNCHER_NAME = "qctx"

#: What a shell-less process needs to find in the FILE. The two keys are absent on
#: purpose: `save()` refuses them, so a config file is complete without them.
NO_SHELL_FIELDS = ("qdrant_url", "memory_collection")


def target_dir(env: dict) -> Path:
    """Where the launcher goes. One location, so the fix hint can name it."""
    return Path(env["HOME"]) / ".local" / "bin"


def launcher_check(root: Path, env: dict) -> Check:
    source = root / "bin" / LAUNCHER_NAME
    found = shutil.which(LAUNCHER_NAME, path=env.get("PATH", ""))
    if not found:
        return Check("launcher", False, f"{LAUNCHER_NAME} is not on PATH",
                     f"{COMMAND_PREFIX} install copies it to "
                     f"{target_dir(env) / LAUNCHER_NAME}")
    # Bytes, not a version marker: a number is one more thing to remember to bump, and
    # this one would go stale the same way the manifests' `version` did.
    if not filecmp.cmp(found, source, shallow=False):
        return Check("launcher", False, f"{found} differs from {source}",
                     f"{COMMAND_PREFIX} install refreshes the copy")

    return Check("launcher", True, str(found))


def path_check(env: dict) -> Check:
    wanted = target_dir(env)
    entries = [Path(p) for p in env.get("PATH", "").split(":") if p]
    if wanted in entries:
        return Check("PATH", True, f"{wanted} is on PATH")

    return Check("PATH", False, f"{wanted} is not on PATH",
                 'add `export PATH="$HOME/.local/bin:$PATH"` to your shell rc')


def no_shell_check(path: Path | None = None) -> Check:
    """What a process that inherits NO environment would read.

    `config show` mixes the file and the environment, so it prints a complete picture over
    an empty file. A background process started by systemd, by a gateway or by cron gets
    the file and nothing else, and the symptom of an empty file is not an error — it is an
    archive that looks simply empty.
    """
    cfg = load(path=path, env={})
    missing = [f for f in NO_SHELL_FIELDS if not getattr(cfg, f)]
    if not cfg.embed_url and not cfg.api_base_url:
        missing.append("embed_url or api_base_url")
    if missing:
        return Check("no-shell config", False,
                     f"the config FILE is missing: {', '.join(missing)}",
                     f"{COMMAND_PREFIX} config set … writes the file; exporting only sets "
                     f"it for your own shell")

    return Check("no-shell config", True, "the file alone is enough to run")


def plumbing(root: Path, env: dict, config_path: Path | None = None) -> list[Check]:
    return [launcher_check(root, env), path_check(env), no_shell_check(config_path)]
