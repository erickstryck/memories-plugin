"""Checks for the plumbing the wizard installs: the launcher, the PATH, the file a
process with no shell reads, and whether each key is set and in which spelling.

DATA, not text — the same contract as `core/setup.py`, and it returns that module's
`Check` so one renderer serves both.

HOST-NEUTRAL, deliberately. `core/setup.py` states the rule and the reason; this module
obeys it. Which host is installed, and running its cutover, is the CLI's business, which
is where the two host names already appear in this repository.
"""
import filecmp
import os
import re
import shutil
import tempfile
from pathlib import Path

from .config import ENV_ALIASES, SECRET_FIELDS, load
from .setup import COMMAND_PREFIX, Check

#: The command every skill and every page of documentation cites.
LAUNCHER_NAME = "qctx"

#: What a shell-less process needs to find in the FILE. The two keys are absent on
#: purpose: `save()` refuses them, so a config file is complete without them.
NO_SHELL_FIELDS = ("qdrant_url", "memory_collection")

#: Pass 1 — what blocks use. Asked outright, in this order.
REQUIRED_FIELDS = ("qdrant_url", "api_base_url", "qdrant_api_key", "api_key",
                   "memory_collection")

#: Pass 2 — everything else, with the current value shown and Enter keeping it.
#: `embed_url` and `rerank_url` are HERE and not derived away: with them empty, the config
#: builds them from `api_base_url`, and a setup that serves rerank on another port then
#: gets a URL that does not exist — measured on the author's own machine, embedding on
#: :8003 and rerank on :8004.
OPTIONAL_FIELDS = ("embed_url", "rerank_url", "embed_model", "rerank_model",
                   "docs_collection", "library_collection", "repos_collection",
                   "repos_registry_collection", "context_window")

#: Written from what the endpoint answered, never typed.
DETECTED_FIELDS = ("vector_size",)


def write_env_file(path: Path, values: dict) -> None:
    """Writes KEY=value lines into a credential file, replacing rather than appending.

    Appending is what a person does by hand, and it leaves two lines for one variable —
    whichever the loader reads last wins, which is a coin toss the operator cannot see.
    BOTH FORMS COUNT: this repository writes and reads `export NAME=` as well as bare
    `NAME=`, so matching only the bare one appended a duplicate for a variable that was
    already there. The prefix that was found is kept — rewriting `export NAME=` as
    `NAME=` would leave the variable set but no longer exported to child processes.

    The mode is set on the file DESCRIPTOR, never with a `chmod` on a path that is
    already holding the key: the file holds a plaintext credential, and `write_text` then
    `chmod` leaves it world-readable for as long as the two calls are apart. A file that
    already exists keeps the permissions its owner gave it — writing a key into somebody's
    file is not a licence to re-mode it — and a file being created gets 0600.

    ATOMIC REPLACE, never a truncation in place. `~/.secrets` may hold every credential
    the user has, and `os.open(O_WRONLY|O_CREAT|O_TRUNC)` then `write` truncates FIRST: a
    disk that filled up, or a process killed between the two calls, left the file empty.
    Everything else in this repository that rewrites a user's file protects it before
    touching it — both cutovers take a dated backup and refuse to proceed without one.
    Here the target is never opened for writing at all: a temporary file in the same
    directory (so the rename stays on one filesystem, where it is atomic) is written in
    full, and only then moved over the target. A failure at any point before that leaves
    the original exactly as it was, and takes the temporary with it — a directory of
    half-written files each holding a plaintext key would be worse than the truncation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text().splitlines() if path.exists() else []
    for name, value in values.items():
        pattern = re.compile(rf"^\s*(export\s+)?{re.escape(name)}=")
        for i, line in enumerate(lines):
            found = pattern.match(line)
            if found:
                lines[i] = f"{found.group(1) or ''}{name}={value}"
                break
        else:
            lines.append(f"{name}={value}")
    body = "\n".join(lines) + "\n"
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp")
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w") as handle:
            handle.write(body)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def read_env_names(path: Path) -> dict:
    """The variables a credential file SETS, and how long each value is.

    Names and lengths, never values — this feeds a report that gets pasted into issues.
    The `export NAME=` form is accepted because this package accepts it everywhere else.
    """
    try:
        body = path.read_text()
    except OSError:
        return {}
    found = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        value = value.strip().strip("'\"")
        if sep and value:
            found[name.strip()] = len(value)

    return found


def credentials_check(env: dict, files=()) -> list[Check]:
    """Whether each key is set, and in WHICH spelling — every one this package accepts.

    Checking fewer spellings than `core.config` reads would be worse than not checking:
    it would report a correctly configured machine as missing its key. So the spellings
    come from `ENV_ALIASES` and are never copied out by hand.

    NOT a blocker when absent. A store with no authentication is a legitimate install,
    and a FAIL here would report every one of them as broken. It is a WARNING — the
    "pending" the design asks for — because a key that lives nowhere durable is a key
    that is gone in the next process.

    `files` is passed IN, never guessed: which files on this machine exist for
    credentials is knowledge about the host, and this module does not have any.
    """
    in_files = {path: read_env_names(path) for path in files}
    checks = []
    for field in (f for f in REQUIRED_FIELDS if f in SECRET_FIELDS):
        aliases = ENV_ALIASES[field]
        where = []
        for name in aliases:
            if env.get(name):
                where.append(f"{name} ({len(env[name])} chars) in the environment")
            for path, names in in_files.items():
                if name in names:
                    where.append(f"{name} ({names[name]} chars) in {path}")
        if where:
            checks.append(Check(field, True, "; ".join(where)))
            continue
        spellings = ", ".join(aliases)
        checks.append(Check(
            field, False,
            f"not set in any spelling that is read: {spellings}",
            f"{COMMAND_PREFIX} install asks for it and writes it to a credential file; "
            f"`export {aliases[0]}=…` sets it for this shell only", warning=True))

    return checks


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


def plumbing(root: Path, env: dict, config_path: Path | None = None,
             credential_files=()) -> list[Check]:
    return [launcher_check(root, env), path_check(env), no_shell_check(config_path),
            *credentials_check(env, credential_files)]
