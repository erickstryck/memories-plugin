#!/usr/bin/env python3
"""qctx — the core's command-line interface.

This is how an agent with no MCP support (or any host at all) uses long-term memory
and the ephemeral document index: one process call, JSON or text on the output. All
the logic lives in `core/`; this file only translates arguments.

    qctx collections list
    qctx config show | set <key> <value>
    qctx memory store <text> [--type T] [--project P] [--json META]
    qctx memory find <question> [--limit N]
    qctx memory recall <question> [--limit N]
    qctx memory get|delete <id>
    qctx memory list [--limit N]
    qctx memory update <id> [--text T] [--json META]
    qctx docs index <path> [--ttl 24h]           temporary, expires
    qctx docs keep <path>                        library, permanent
    qctx docs search <question> [--scope all|tmp|library] [--doc-id ID]
    qctx docs list [--scope ...]
    qctx docs refresh [--scope library|tmp]      reindexes what changed on disk
    qctx docs drop <doc-id> [--scope ...] | --purge-tmp | --expired
    qctx repos register <name> [--label L]       declares a repository
    qctx repos list                              every indexed repository
    qctx repos search <question> [--repo R | --all] [--limit N]
    qctx repos add <repo> <path> [<path>...]     indexes exactly these files
                                                 (the repo must be registered first)
    qctx repos drop <repo> --yes                 permanent, no undo

Three archives, three lifecycles: MEMORY holds curated facts and never expires;
LIBRARY holds documents for reference and never expires; TEMPORARY holds a task's
documents and expires. Distinct collections, by configuration. REPOSITORIES are a
fourth: code chunks grouped by repo, with a registry of what exists, in two more
collections of their own — see `core/repos.py` for why they are not the library.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core  # noqa: E402
import core.docs  # noqa: E402
import core.install  # noqa: E402
import core.setup  # noqa: E402
from core.config import ConfigError  # noqa: E402


def as_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def output(obj, want_json: bool) -> None:
    """Prints the JSON form, or NOTHING when JSON was not asked for.

    The silent branch is deliberate but it is a landmine, so it is named here rather
    than discovered: every caller has to print the human form itself. A handler written
    as `output(x, args.json)` and nothing else prints nothing at all in the default
    mode.
    """
    if want_json:
        print(as_json(obj))


# ---- collections / config --------------------------------------------------

# One definition, in the core, imported here. It used to exist twice under two names
# (`NOISE` here, `NOISE_PREFIXES` in core.setup) with identical values — the kind of
# duplication where the two drift and only one of the two listings changes.
from core.setup import NOISE_PREFIXES as NOISE  # noqa: E402


def cmd_collections(args, cfg):
    q = core.build_qdrant(cfg)
    names = q.list_collections()
    # All five, like `Config._require_distinct`: a configured collection is exempt from the
    # noise-prefix hiding, and listing only three of them hid two collections this package
    # writes to from the command whose job is to show them.
    configured = {cfg.memory_collection, cfg.docs_collection, cfg.library_collection,
                  cfg.repos_collection, cfg.repos_registry_collection}
    if not args.all:
        visible = [n for n in names
                    if n in configured or not n.startswith(NOISE)]
    else:
        visible = list(names)
    hidden = len(names) - len(visible)
    lines = []
    for name in visible:
        info = q.collection_info(name) or {}
        size = info.get("size")
        role = ("memory" if name == cfg.memory_collection
                else "temporary" if name == cfg.docs_collection
                else "library" if name == cfg.library_collection
                else "repos" if name == cfg.repos_collection
                else "repo registry" if name == cfg.repos_registry_collection else "")
        # The registry is the one collection whose vector says nothing (`core/repos.py`
        # sizes it 1 on purpose), so judging it by the model's dimension would mark a
        # correct install INCOMPAT.
        wanted = core.repos.REGISTRY_VECTOR_SIZE if role == "repo registry" \
            else cfg.vector_size
        lines.append({
            "collection": name,
            "points": info.get("points"),
            "dim": size,
            "compatible": size == wanted,
            "role": role,
        })
    lines.sort(key=lambda l: (not l["role"], -(l["points"] or 0)))
    if args.json:
        output({"vector_size": cfg.vector_size, "hidden": hidden,
               "collections": lines}, True)

        return
    print(f"model {cfg.embed_model} uses dimension {cfg.vector_size}\n")
    print(f"{'collection':34} {'points':>8} {'dim':>6}  {'':10} role")
    for l in lines:
        mark = "ok" if l["compatible"] else "INCOMPAT."
        print(f"{l['collection']:34} {str(l['points']):>8} {str(l['dim']):>6}  {mark:10} {l['role']}")
    if hidden:
        print(f"\n({hidden} collection(s) from another system hidden — `--all` shows them)")
    print("\nto choose: qctx config set memory-collection|docs-collection|"
          "library-collection <name>")


def cmd_config_show(args, cfg):
    data = core.redacted(cfg)
    if args.json:
        output(data, True)

        return
    for k, v in data.items():
        print(f"  {k:20} {v}")


def cmd_config_set(args, cfg):
    key = args.key.replace("-", "_")
    value = int(args.value) if key == "vector_size" else args.value
    path = core.save({key: value})
    if args.json:
        output({"key": key, "value": value, "path": str(path)}, True)
    else:
        print(f"{key} = {value}  (written to {path})")
    # A warning, not an error: the collection may be created later. But an incompatible
    # dimension is a silent trap, so it is worth shouting at the moment of choice.
    if key in ("memory_collection", "docs_collection", "library_collection",
               "repos_collection", "repos_registry_collection"):
        # The registry's vector carries no meaning and is sized 1 (`core/repos.py`), so the
        # dimension it has to match is its own, not the model's.
        wanted = core.repos.REGISTRY_VECTOR_SIZE if key == "repos_registry_collection" \
            else cfg.vector_size
        try:
            q = core.build_qdrant(core.load())
            info = q.collection_info(value)
            if info is None:
                print(f"  (collection {value!r} does not exist yet — it will be created on first use)")
            elif info.get("size") not in (None, wanted):
                print(f"  CAUTION: {value!r} has dimension {info['size']}, "
                      f"incompatible with this role ({wanted})")
        except Exception:
            pass


def cmd_config_detect(args, cfg):
    """Finds the model's real dimension instead of trusting the number that was typed."""
    dim = core.build_embedder(cfg).detect_dimension()
    if dim == cfg.vector_size:
        print(f"{cfg.embed_model} returns {dim} dimensions — the config is already correct")

        return
    path = core.save({"vector_size": dim})
    print(f"{cfg.embed_model} returns {dim} dimensions (the config said {cfg.vector_size})")
    print(f"  vector_size updated in {path}")
    print("  check with `qctx collections list` which collections are still compatible")


def _render_check(c: dict) -> None:
    mark = "ok  " if c["ok"] else ("warn" if c["warning"] else "FAIL")
    print(f"  [{mark:5}] {c['name']:20} {c['detail']}")
    if not c["ok"] and c["fix_hint"]:
        print(f"            -> {c['fix_hint']}")


def cmd_setup(args, cfg):
    """Guided diagnostics. It does NOT block on stdin when there is no terminal.

    That is not a detail: this command exists to be called by an agent or a script too,
    and an `input()` waiting for an answer that never comes would hang the call until the
    timeout. With no TTY, the command diagnoses, prints the exact commands that are
    missing, and exits.
    """
    rel = core.setup.diagnose(cfg)
    if args.json:
        output(rel, True)

        return

    print("diagnostics:\n")
    for c in rel["checks"]:
        _render_check(c)

    if rel["ready"]:
        print("\nready to use.")
    else:
        print(f"\n{len(rel['blockers'])} item(s) block use — the commands above fix them.")

    interactive = sys.stdin.isatty() and not args.check
    if not interactive:
        if rel["memory_suggestions"] and not cfg.memory_collection:
            print("\ncandidate collections for memory (most populated first):")
            for i, s_ in enumerate(rel["memory_suggestions"], 1):
                print(f"  {i}. {s_['collection']:34} {s_['points']:>8} points")
            print("\nchoose with: qctx config set memory-collection <name>")
        if not sys.stdin.isatty():
            print("\n(no interactive terminal — nothing was changed)")

        return

    print("\n--- configure (Enter keeps the current value) ---")
    options = [s_["collection"] for s_ in rel["memory_suggestions"]]
    for i, s_ in enumerate(options, 1):
        print(f"  {i}. {s_}")
    choice = core.setup.choose_by_index(
        options, input(f"memory collection [{cfg.memory_collection or 'none'}]: "))
    if choice:
        core.save({"memory_collection": choice})
        print(f"  memory_collection = {choice}")
    for key, current in (("docs_collection", cfg.docs_collection),
                         ("library_collection", cfg.library_collection),
                         ("repos_collection", cfg.repos_collection),
                         ("repos_registry_collection", cfg.repos_registry_collection)):
        resp = input(f"{key} [{current}]: ").strip()
        if resp:
            core.save({key: resp})
            print(f"  {key} = {resp}")
    if rel["detected_dim"] and rel["detected_dim"] != cfg.vector_size:
        core.save({"vector_size": rel["detected_dim"]})
        print(f"  vector_size = {rel['detected_dim']} (detected from the endpoint)")
    print("\nrunning the diagnostics again:\n")
    for c in core.setup.diagnose(core.load())["checks"]:
        _render_check(c)


# ---- install ----------------------------------------------------------------

#: The host sections, and the variable each script reads to skip its own test suite.
#: The scripts stay the owners of host state: they already back up what they edit and
#: re-read it to confirm the write, and a second implementation of those checks here would
#: be a second source of truth that diverges at the first fix.
HOST_SECTIONS = (
    ("claude-code", "scripts/cutover.sh", "CUTOVER_SKIP_SUITE"),
    ("hermes", "scripts/hermes_cutover.sh", "HERMES_CUTOVER_SKIP_SUITE"),
)

#: The command that tells us a host is on this machine at all.
HOST_BINARIES = {"claude-code": "claude", "hermes": "hermes"}

#: How long an `--apply` cutover may take, in seconds.
#:
#: The scripts run the whole suite before they write anything — that is their own rule,
#: and they enforce it by refusing `--apply` while the skip variable is set. The suite
#: measured 852 s on 2026-08-20, against a value of 900: 48 seconds of margin for a
#: number that grows with every test added. And a `TimeoutExpired` here does not land
#: somewhere harmless — it fires while the script is rewriting settings.json, which is
#: the one moment there is nothing to gain by giving up. Well above the cost, therefore,
#: and the timeout that remains is handled rather than raised.
CUTOVER_APPLY_TIMEOUT = 3600

#: What each host needs typed, when the plugin is not installed there yet. Shown before it
#: is run: `--force` is the user agreeing to what hermes' scanner flagged (this tree ships
#: two scripts that edit host configuration, which is the cutovers' declared job), and
#: pointing `memory.provider` here REPLACES whatever provider is named, because hermes
#: activates exactly one.
HOST_INSTALL_COMMANDS = {
    "claude-code": (
        "claude plugin marketplace add erickstryck/memories-plugin",
        "claude plugin install memories-plugin@memories-plugin",
    ),
    "hermes": (
        "hermes plugins install erickstryck/memories-plugin --enable --force",
        "hermes config set memory.provider memories",
    ),
}


#: WHY each host's commands look the way they do. The design is explicit that these are
#: SHOWN before the question and not buried in a code comment: both are the user's
#: decision, not an implementation detail.
HOST_INSTALL_REASONS = {
    "claude-code": (
        "the marketplace is added first because `plugin install` resolves the name "
        "through it; without it there is nothing to install from.",
    ),
    "hermes": (
        "--force is needed because the scanner classifies this tree as `caution`: it "
        "ships two scripts that edit host configuration, which is the cutovers' "
        "declared job. The flag is your agreement to that, so it is shown here rather "
        "than hidden in the middle of the command.",
        "memory.provider REPLACES whatever provider is set — hermes activates exactly "
        "one of them.",
    ),
}


def hermes_memory_provider() -> str:
    """What hermes has `memory.provider` set to right now, or "" if it cannot be read.

    Read-only, and best effort: this exists so the wizard can say WHAT it is about to
    replace before it asks, and a host that will not answer must not stop the run.
    """
    try:
        done = subprocess.run(["hermes", "config", "get", "memory.provider"],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""

    return done.stdout.strip() if done.returncode == 0 else ""


def _explain_host_install(host: str) -> None:
    for reason in HOST_INSTALL_REASONS[host]:
        print(f"      why: {reason}")
    if host == "hermes":
        current = hermes_memory_provider()
        if current and current != "memories":
            print(f"      memory.provider is {current!r} today, and will be replaced")


def _host_dry_run(host: str, script: str, skip_var: str, root: Path) -> dict:
    """Runs a cutover in its report-only mode. Writes nothing; that is the script's
    contract with no `--apply`.

    The suite is skipped HERE and only here: both scripts run the full suite among their
    checks, which costs ~41s per host to draw a list. It runs once before any apply, and
    both scripts refuse to apply with the variable set.
    """
    env = dict(os.environ, **{skip_var: "1"})
    done = subprocess.run(["bash", str(root / script)], capture_output=True, text=True,
                          env=env, timeout=300)

    return {"host": host, "exit_code": done.returncode, "text": done.stdout + done.stderr}


def _host_sections(root: Path) -> list[dict]:
    sections = []
    for host, script, skip_var in HOST_SECTIONS:
        if not shutil.which(HOST_BINARIES[host]):
            sections.append({"host": host, "exit_code": 0,
                             "text": f"  ..    {HOST_BINARIES[host]} is not on PATH — "
                                     f"skipping this host\n"})
            continue
        sections.append(_host_dry_run(host, script, skip_var, root))

    return sections


def claude_install_path(home: Path) -> str | None:
    """The live install path, read from the harness' own record.

    Not guessable: the cache directory is named after the commit, and old ones stay behind
    — five of them on the machine this was measured on.
    """
    registry = home / ".claude" / "plugins" / "installed_plugins.json"
    try:
        data = json.loads(registry.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for entry in data.get("plugins", {}).get("memories-plugin@memories-plugin", []):
        if entry.get("installPath"):
            return entry["installPath"]

    return None


def hermes_install_path(env: dict) -> Path | None:
    """One level deep and no deeper: hermes' loader scans `$HERMES_HOME/plugins/<name>/`
    and never looks further down."""
    home = Path(env.get("HERMES_HOME") or Path(env["HOME"]) / ".hermes")
    candidate = home / "plugins" / "memories"

    return candidate if candidate.exists() else None


def credential_files(env: dict) -> list[Path]:
    """The files on THIS machine that exist to hold a credential.

    Here and not in `core/install.py`: that a hermes reads `$HERMES_HOME/.env` is
    knowledge about a host, and the core is not allowed any. The core takes the list and
    reports which spelling it found where.

    Only files that already exist are listed. The wizard creates a credential file when
    the user names one; a check that invented paths would report a file nobody has.
    """
    home = Path(env.get("HOME") or os.path.expanduser("~"))
    hermes_home = Path(env.get("HERMES_HOME") or home / ".hermes")

    return [p for p in (hermes_home / ".env", home / ".secrets") if p.is_file()]


def _plumbing(root: Path) -> list[dict]:
    """The plumbing checks, with this machine's credential files handed in."""
    env = dict(os.environ)

    return [asdict(c) for c in
            core.install.plumbing(root, env, credential_files=credential_files(env))]


def install_launcher(root: Path, env: dict) -> Path:
    """Copies the launcher onto PATH. A copy, not a symlink: on a machine with only
    claude-code the only stable thing to link to does not exist — the tree is a directory
    named after a commit, replaced by the next update."""
    target_dir = core.install.target_dir(env)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / core.install.LAUNCHER_NAME
    # UNLINK FIRST, always. The documented development install makes this path a symlink
    # into a checkout: copying onto it raises `SameFileError` when the link points at
    # THIS tree, and silently rewrites somebody else's `bin/qctx` when it points at
    # another one. Removing the name first turns both into a plain, correct copy.
    target.unlink(missing_ok=True)
    shutil.copyfile(root / "bin" / core.install.LAUNCHER_NAME, target)
    target.chmod(0o755)

    return target


#: What no wizard can do for you, printed at the end of a run that wrote something.
#: Both are one-time and both fail SILENTLY when skipped, which is why they are printed
#: rather than merely documented.
MANUAL_STEPS = (
    "hermes: approve the read guard once at a TTY — a hermes with no terminal skips the "
    "hook silently until then. `hermes hooks list` shows it allowed afterwards.",
    "claude-code: open a new terminal, or restart — hooks are read at start-up.",
)


def should_stop_before_hosts(report: dict) -> bool:
    """A blocker means the archive itself does not answer. Installing into a host on top
    of that produces a host wired to nothing, and a green report about it."""
    return not report.get("ready", False)


def host_plugin_path(host: str):
    """Where this host has the plugin installed, or None. The one probe, used twice."""
    if host == "claude-code":
        return claude_install_path(Path(os.path.expanduser("~")))

    return hermes_install_path(dict(os.environ))


def _host_install_group(args, root: Path) -> dict:
    """Per host that is on this machine: install the plugin if it is missing.

    The cutovers do not do this — they register hooks and verify state. Returns
    host -> whether the plugin is present afterwards, so the caller offers each
    host's cutover only when there is a plugin to cut over.

    "Afterwards" is RE-PROBED, never assumed. Recording success because the commands
    were run led to an `--apply` cutover over a host whose `plugins install` had exited
    non-zero — a cutover of nothing, reported as a cutover.
    """
    present_now = {}
    for host, _script, _skip in HOST_SECTIONS:
        binary = HOST_BINARIES[host]
        if not shutil.which(binary):
            continue
        present = host_plugin_path(host)
        if present:
            print(f"  ok    {host}: installed at {present}")
            present_now[host] = True
            continue
        print(f"\n  {host} is on this machine and the plugin is not installed there:")
        _explain_host_install(host)
        for command in HOST_INSTALL_COMMANDS[host]:
            print(f"      {command}")
        if not (args.yes or _ask("  run these? [y/N]: ").strip().lower() == "y"):
            present_now[host] = False
            continue
        for command in HOST_INSTALL_COMMANDS[host]:
            subprocess.run(command, shell=True, check=False)
        installed = host_plugin_path(host)
        present_now[host] = bool(installed)
        if installed:
            print(f"  ok    {host}: installed at {installed}")
        else:
            print(f"  FAIL  {host}: still not installed — the cutover is skipped, "
                  f"because there would be nothing to cut over to")

    return present_now


def _host_cutover_group(args, root: Path, present_now: dict) -> None:
    """Per host with a plugin: the dry-run plan, a confirmation of its own, and the
    same script with --apply when it is given.

    The plan is re-run after the install group, not reused from the report phase: a
    cutover shown before the plugin exists describes a host that no longer is one. The
    dry run skips the suite; the apply runs it once — the script's own rule, which it
    enforces by refusing `--apply` while the skip variable is set.
    """
    for host, script, skip_var in HOST_SECTIONS:
        if host not in present_now or not present_now[host]:
            continue
        print(f"\n{host} — cutover plan:\n")
        plan = _host_dry_run(host, script, skip_var, root)
        print(plan["text"].rstrip())
        if args.yes or _ask(f"\n  apply the {host} cutover? [y/N]: ").strip().lower() == "y":
            try:
                done = subprocess.run(["bash", str(root / script), "--apply"],
                                      capture_output=True, text=True,
                                      env=dict(os.environ),
                                      timeout=CUTOVER_APPLY_TIMEOUT)
            except subprocess.TimeoutExpired:
                print(f"  FAIL  the {host} cutover was killed after "
                      f"{CUTOVER_APPLY_TIMEOUT}s. It may have been interrupted "
                      f"mid-write — the script takes a dated backup of everything it "
                      f"edits, so check that before running it again.")
                continue
            print(done.stdout + done.stderr, end="")


def _interactive(args) -> bool:
    """Whether this run may ask questions and write.

    `QCTX_INSTALL_FORCE_TTY` is set by the tests, which have no terminal and still need
    the interactive path. A real TTY is the usual signal.
    """
    return bool(sys.stdin.isatty() or os.environ.get("QCTX_INSTALL_FORCE_TTY"))


def _ask(prompt: str) -> str:
    """`input()`, with end-of-input read as Enter.

    A closed stdin is not an error in this wizard, it is the answer "keep what is there".
    Without this, the run died with an uncaught `EOFError` the moment stdin ran out —
    and on the full path the launcher had already been copied into `~/.local/bin`, so
    the wizard failed half-done, which is the one outcome it exists to avoid.
    """
    try:
        return input(prompt)
    except EOFError:
        print()

        return ""


def _ask_number(prompt: str, current) -> int | None:
    """A whole number, or nothing. It RE-ASKS instead of raising.

    `int(entry)` on the last question threw an uncaught `ValueError` after up to
    fourteen answers had been typed and before `core.save` had written any of them: one
    typo cost the whole pass.
    """
    while True:
        entry = _ask(prompt).strip()
        if not entry:
            return None
        try:
            return int(entry)
        except ValueError:
            print(f"  ..    {entry!r} is not a whole number — type one, "
                  f"or press Enter to keep {current}")


def _read_secret(prompt: str) -> str:
    """Echo off when there is a terminal, plain read when there is not.

    `getpass` opens /dev/tty and falls back to stdin with a warning when it cannot — two
    different behaviours depending on where it runs, which is not something a test should
    have to guess at. Choosing explicitly keeps the piped case deterministic.
    """
    if sys.stdin.isatty():
        import getpass

        try:
            return getpass.getpass(prompt)
        except EOFError:
            return ""

    return _ask(prompt)


def _secret_status(field: str) -> str:
    """What the prompt shows beside a key: where it already lives, or MISSING.

    Reading only the process environment made a key that lives correctly in
    `~/.hermes/.env` show as MISSING and be re-asked on every run — and the wizard runs
    on every verification. That trains exactly the paste-a-secret-every-run reflex the
    design forbids; already configured has to be RECOGNISED, not re-asked.
    """
    from core import config as _config

    env = dict(os.environ)
    for name in _config.ENV_ALIASES[field]:
        if env.get(name):
            return f"already set as {name} in the environment"
    for path in credential_files(env):
        names = core.install.read_env_names(path)
        for name in _config.ENV_ALIASES[field]:
            if name in names:
                return f"already set as {name} in {path}"

    return "MISSING"


def _ask_config(cfg, interactive: bool = True, suggestions=()) -> None:
    """The two passes. Blockers first, then everything else with Enter keeping.

    With no terminal there is nobody to ask, so every value is kept. That is what `--yes`
    means here: the flag answers yes to the groups that ACT, and a configuration pass
    with no answers has nothing to act on. Writing the defaults over a working config
    because nobody was there to object would be worse than the crash this replaced.
    """
    from core import config as _config

    if not interactive:
        print("\n  ..    no terminal to ask at — every configuration value is kept")

        return

    patch, secrets = {}, {}
    print("\n--- required (Enter keeps the current value) ---")
    for field in core.install.REQUIRED_FIELDS:
        current = getattr(cfg, field)
        if field in _config.SECRET_FIELDS:
            entry = _read_secret(f"{field} [{_secret_status(field)}]: ").strip()
            if entry:
                secrets[_config.ENV_ALIASES[field][0]] = entry
            continue
        if field == "memory_collection" and suggestions:
            # The design asks for this collection to be offered "with the suggestions of
            # `suggest_collections`". They were computed for the report and then shown
            # nowhere, while `cmd_setup` printed the same list twenty lines away.
            print("\n  candidates (most populated first):")
            for i, option in enumerate(suggestions, 1):
                print(f"    {i}. {option['collection']:34} "
                      f"{option['points']:>8} points")
            choice = core.setup.choose_by_index(
                [option["collection"] for option in suggestions],
                _ask(f"{field} [{current or 'MISSING'}]: "))
            if choice:
                patch[field] = choice
            continue
        entry = _ask(f"{field} [{current or 'MISSING'}]: ").strip()
        if entry:
            patch[field] = entry

    print("\n--- everything else (Enter keeps) ---")
    for field in core.install.OPTIONAL_FIELDS:
        current = getattr(cfg, field)
        if isinstance(current, int):
            number = _ask_number(f"{field} [{current}]: ", current)
            if number is not None:
                patch[field] = number
            continue
        entry = _ask(f"{field} [{current}]: ").strip()
        if entry:
            patch[field] = entry

    if patch:
        core.save(patch)
        print(f"\n  wrote {len(patch)} setting(s) to the config file")
    if secrets:
        _store_secrets(secrets)


def _detect_vector_size() -> None:
    """The fifteenth field, and the only one the wizard must NOT ask for.

    A hand-typed dimension is a number that disagrees with the server the day the model
    changes, and the compatibility guard then blames the collection. `cmd_setup` has
    always written it from what the endpoint answered; the wizard declared the field in
    `DETECTED_FIELDS` and never wrote it, so the completeness proof the design demanded
    was green over a field nothing could set.

    An unreachable endpoint is not an error here. The run already reports it as a
    blocker, and raising at this point would throw away the fourteen answers just typed.
    """
    cfg = core.load()
    try:
        detected = core.build_embedder(cfg).detect_dimension()
    except core.CoreError as exc:
        print(f"  ..    vector_size not detected: {exc}")

        return
    if detected == cfg.vector_size:
        print(f"  ok    vector_size = {detected}, confirmed by the endpoint")

        return
    core.save({"vector_size": detected})
    print(f"  ok    vector_size = {detected}, detected from the endpoint "
          f"(the config said {cfg.vector_size})")


def _warn_if_readable(path: Path) -> None:
    """Says so when a credential file is readable by anyone but its owner.

    `write_env_file` deliberately does not re-mode a file somebody already had — but
    leaving that SILENT would be the other half of the same mistake.
    """
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        print(f"  ..    {path} is mode {mode:o} — readable beyond its owner; "
              f"`chmod 600 {path}` if that is not deliberate")


def _shell_credential_file(home: Path) -> Path | None:
    """The file the user names for their interactive shell, or None.

    `~/.secrets` is offered as the DEFAULT only when it already exists: the design says
    the wizard creates nothing on its own here. Naming a path is the consent to create
    it; an empty answer with no `~/.secrets` present is a no, and the caller then prints
    the `export` lines and leaves the key PENDING rather than claiming it is done.
    """
    default = home / ".secrets"
    shown = str(default) if default.is_file() else "none — Enter to skip"
    entry = _ask(f"  shell credential file [{shown}]: ").strip()
    if entry:
        return Path(os.path.expanduser(entry))

    return default if default.is_file() else None


def _store_secrets(secrets: dict) -> None:
    """The keys go to files that exist FOR credentials, and to no other file.

    Never echoes a value: this output gets pasted into issues and chats. And on a machine
    with no hermes there is no plugin-owned credential file at all — saying "done" there
    would be the same lie the README told.

    THE PROCESS ENVIRONMENT IS SET TOO, in addition to every file write and never instead
    of one. Everything after this point re-reads configuration through `core.load()`,
    which sees the config file plus this process's environment — and a key is in neither,
    because a secret never enters the config file. So a key typed thirty seconds ago was
    invisible to the `vector_size` probe and to the `diagnose` that decides whether the
    host steps may run: on a store with authentication turned on the wizard answered 401
    to itself and stopped, on exactly the fresh-machine flow it exists for.

    It changes nothing about durability, and the "PENDING" verdict below still belongs to
    the files alone: an environment variable is gone with this process, which is the whole
    reason a credential file is asked for.
    """
    home = Path(os.path.expanduser("~"))
    written_anywhere = False
    for name, value in secrets.items():
        os.environ[name] = value
    # `or`, not a default argument: `HERMES_HOME=""` makes `Path("")` — which is
    # `Path(".")`, whose `is_dir()` is True — so the empty value used to write a
    # plaintext key into whatever directory the wizard was run from. `hermes_install_path`
    # already spelled it this way.
    hermes_env = Path(os.environ.get("HERMES_HOME") or home / ".hermes") / ".env"
    if hermes_env.parent.is_dir():
        core.install.write_env_file(hermes_env, secrets)
        for name, value in secrets.items():
            print(f"  ok    {name} written to {hermes_env} (len {len(value)})")
        _warn_if_readable(hermes_env)
        written_anywhere = True

    shell_file = _shell_credential_file(home)
    if shell_file:
        core.install.write_env_file(shell_file, secrets)
        print(f"  ok    also written to {shell_file}")
        _warn_if_readable(shell_file)
        written_anywhere = True

    if not written_anywhere:
        print("  ..    no credential file on this machine — add these to your shell rc:")
        for name in secrets:
            print(f"          export {name}=…")
        print("  PENDING: a key that lives only in this shell is gone in the next one")


def merged_report(report: dict, plumbing: list[dict]) -> dict:
    """One verdict over BOTH halves of the picture.

    `ready` and `blockers` used to come from `core.setup.diagnose` alone, with the
    plumbing merged into `checks` and nowhere else. So a machine with a reachable Qdrant
    and no `qctx` on PATH answered `"ready": true` — and `ready` is the field a program
    reads, which makes it the one that must not be optimistic.

    The plumbing goes FIRST, matching the order the human rendering prints.
    """
    checks = plumbing + report["checks"]
    blockers = [c for c in checks if not c["ok"] and not c["warning"]]
    warnings = [c for c in checks if not c["ok"] and c["warning"]]

    return {**report, "checks": checks, "blockers": blockers, "warnings": warnings,
            "ready": not blockers}


def cmd_install(args, cfg):
    """The wizard: diagnose, then offer to fix, one group at a time.

    With `--check` it only reports. With no TTY and no `--yes` it also only reports — the
    same rule `cmd_setup` follows, and for the same reason: this command is called by
    agents and by scripts, and an `input()` waiting for an answer that never comes hangs
    the caller.
    """
    root = Path(__file__).resolve().parent.parent
    report = core.setup.diagnose(cfg)
    plumbing = _plumbing(root)
    hosts = [] if args.config_only else _host_sections(root)

    if args.json:
        output({**merged_report(report, plumbing), "hosts": hosts}, True)

        return

    print("plumbing:\n")
    for c in plumbing:
        _render_check(c)
    print("\nreachability and configuration:\n")
    for c in report["checks"]:
        _render_check(c)
    for section in hosts:
        print(f"\n{section['host']}:\n")
        print(section["text"].rstrip())

    if args.check:
        return
    if not _interactive(args) and not args.yes:
        print("\n(no interactive terminal — nothing was changed)")

        return

    if args.config_only:
        _ask_config(cfg, _interactive(args), report["memory_suggestions"])
        _detect_vector_size()

        return

    # 2. PATH — the launcher lands on PATH before the configuration, in the spec's
    # order. The rc line is printed, never written: the rc is the user's file.
    launcher = install_launcher(root, dict(os.environ))
    print(f"\n  ok    launcher at {launcher}")
    if not core.install.path_check(dict(os.environ)).ok:
        print(f"  ..    {core.install.target_dir(dict(os.environ))} is not on PATH — add:")
        print('        export PATH="$HOME/.local/bin:$PATH"')

    # 3. config — the two passes, then the one field that is answered by the endpoint.
    _ask_config(cfg, _interactive(args), report["memory_suggestions"])
    _detect_vector_size()

    # 4. hosts — but only if the archive itself answers. Installing into a host on top
    # of a Qdrant that does not answer produces a host wired to nothing.
    after = core.setup.diagnose(core.load())
    if should_stop_before_hosts(after):
        print("\nstopping here — these must answer before a host install means anything:")
        for c in after["blockers"]:
            _render_check(c)

        return

    print("\nhosts:\n")
    present_now = _host_install_group(args, root)
    _host_cutover_group(args, root, present_now)

    # 5. re-verify — including the no-shell check, which only a fresh read can prove.
    print("\nre-checking:\n")
    for c in _plumbing(root):
        _render_check(c)
    for c in core.setup.diagnose(core.load())["checks"]:
        _render_check(c)

    # 6. what is left, and only you can do it.
    print("\nwhat is left, and only you can do it:")
    for step in MANUAL_STEPS:
        print(f"  - {step}")


# ---- memory ----------------------------------------------------------------

def _metadata_from_args(args) -> dict:
    """Translates argparse into `core.metadata_from`, which owns the rule.

    The assembly itself — base object, then the named fields winning over it — lives in
    the core because the hermes `memory_store` tool assembles metadata the same way from
    a `metadata` object plus the same three names. While the rule lived here, the second
    host had nothing to call and would have carried its own copy of the precedence.
    """
    return core.metadata_from(
        getattr(args, "json_meta", None),
        **{field: getattr(args, field, None) for field in core.METADATA_FIELDS})


def cmd_memory_store(args, cfg):
    result = core.build_memory(cfg).store(args.text, _metadata_from_args(args))
    print(json.dumps(result, ensure_ascii=False) if args.json else f"stored id={result['id']}")


def cmd_memory_find(args, cfg):
    hits = core.build_memory(cfg).find(args.query, args.limit)
    if args.json:
        output(hits, True)

        return
    if not hits:
        print("no memory found")

        return
    for i, h in enumerate(hits, 1):
        print(f"{i}. {h['score']:.3f}  {h['id']}  {json.dumps(h['metadata'], ensure_ascii=False)}")
        print(f"   {h['document'][:400]}\n")


def cmd_memory_recall(args, cfg):
    store = core.build_memory(cfg)
    policy = core.Policy(dense_floor=args.dense_floor, strict_floor=args.strict_floor,
                           min_score=args.min_score, max_results=args.limit,
                           veto=True, order_matters=False)
    hits, outcome = store.recall([args.query], policy, args.top_k)
    if args.json:
        output({"info": core.outcome_payload(outcome),
                "hits": [h.__dict__ for h in hits]}, True)

        return
    if outcome.scale_converted:
        print("(logit scale detected and normalized to sigmoid)")
    if not hits:
        print(f"nothing above the cut (best dense {outcome.best_dense:.3f})")

        return
    for i, h in enumerate(hits[:args.limit], 1):
        print(f"{i}. {h.origin} {h.score:.3f} (dense {h.dense_score:.3f})  {h.id}")
        print(f"   {json.dumps(h.metadata, ensure_ascii=False)}")
        print(f"   {h.document[:600]}\n")


def cmd_memory_get(args, cfg):
    output(core.build_memory(cfg).get(args.id), True)


def cmd_memory_delete(args, cfg):
    output(core.build_memory(cfg).delete(args.id), True)


def cmd_memory_update(args, cfg):
    meta = _metadata_from_args(args) or None
    res = core.build_memory(cfg).update(args.id, args.text, meta)
    print(json.dumps(res, ensure_ascii=False) if args.json else f"{res['status']} id={res['id']}")


def cmd_memory_store_many(args, cfg):
    """A batch with ONE trip to the embeddings endpoint and all-or-nothing semantics.

    It existed in the core with no surface: without this, a checkpoint with N facts cost
    N processes and N embedding calls, and lost the atomicity the method was written to
    provide.
    """
    raw = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
    items = json.loads(raw)
    if not isinstance(items, list):
        print("error: expected a JSON array of {information, metadata?}", file=sys.stderr)
        raise SystemExit(2)
    res = core.build_memory(cfg).store_many(items)
    print(json.dumps(res, ensure_ascii=False) if args.json
          else f"stored {res['count']}: {' '.join(res['ids'])}")


def cmd_memory_search_collections(args, cfg):
    """READ-ONLY search across other systems' collections."""
    res = core.search_collections(core.build_qdrant(cfg), core.build_embedder(cfg),
                                  args.query, args.collections or None,
                                  cfg.vector_size, limit=args.limit)
    if args.json:
        output(res, True)

        return
    if res["skipped"]:
        for s_ in res["skipped"]:
            print(f"  (skipped {s_['collection']}: {s_['reason']})")
    for i, h in enumerate(res["results"], 1):
        print(f"{i}. [{h['collection']}] {h['score']:.3f}  {h['id']}")
        print(f"   {(h['document'] or str(h['payload']))[:300]}\n")


def cmd_memory_list(args, cfg):
    output(core.build_memory(cfg).list_page(args.limit), True)


# ---- docs ------------------------------------------------------------------

def _report_write(res: dict, as_json: bool) -> None:
    if as_json:
        output(res, True)

        return
    label = "kept in the library" if res["scope"] == "library" else "indexed (temporary)"
    print(f"{label}: {os.path.basename(res['path'])} -> doc_id={res['doc_id']}")
    print(f"  {res['lines']} lines, {res['chars']} chars -> {res['chunks']} chunks "
          f"(mode {res['mode']}, collection {res['collection']})")
    if res["expires_at"]:
        print(f"  expires at {res['expires_at']}")
    else:
        print("  no expiry — remove it with `qctx docs drop <doc-id> --scope library`")
    print(f"  search: qctx docs search \"<question>\" --doc-id {res['doc_id']}")


def cmd_docs_index(args, cfg):
    res = core.build_docs(cfg).index_file(args.path, core.parse_ttl(args.ttl), args.doc_id)
    _report_write(res, args.json)


def cmd_docs_keep(args, cfg):
    res = core.build_docs(cfg).keep_file(args.path, args.doc_id)
    _report_write(res, args.json)


def cmd_docs_refresh(args, cfg):
    report = core.build_docs(cfg).refresh(args.scope)
    if args.json:
        output(report, True)

        return
    if not report:
        print("nothing to check")

        return
    for r in report:
        mark = {"ok": "  ", "reindexed": "->", "missing": "!!"}.get(r["action"], "  ")
        print(f"{mark} {r['action']:11} {r['doc_id']}  {r['path']}")


def cmd_docs_search(args, cfg):
    hits, outcome = core.build_docs(cfg).search(args.query, args.scope, args.doc_id, args.limit)
    if args.json:
        output({"info": core.outcome_payload(outcome),
                "hits": [h.__dict__ for h in hits]}, True)

        return
    if not hits:
        print("no relevant chunk (or the index expired — see `qctx docs list`)")

        return
    if outcome.dropped_above_floor:
        # The same obligation recall.py honours: a candidate cut by the pair ceiling was
        # never judged, and one of them could have ranked first. Saying the list is not
        # exhaustive is the difference between a ranking and a verdict.
        print(f"({outcome.dropped_above_floor} candidate(s) above the dense floor went "
              f"unjudged — the pair ceiling cut them; narrow with --doc-id for a full pass)\n")
    if outcome.collapsed:
        print(f"(re-rank collapsed — best CE {outcome.best_rerank:.4f}, typical of a question "
              f"and a document in different languages; using DENSE order, which is "
              f"language-agnostic)\n")
    elif not outcome.reranked:
        print(f"(warning: the re-rank did not run — {outcome.rerank_error}; DENSE order, "
              f"not a verdict)\n")
    for i, h in enumerate(hits, 1):
        warning = f"  ⚠ {h.stale}" if h.stale else ""
        label = "library" if h.scope == "library" else "temporary"
        if h.mode == "locator":
            print(f"{i}. [{label}] {h.path}:{h.start_line}-{h.end_line}  "
                  f"({h.origin} {h.score:.3f}){warning}")
            print(f"   {' '.join(h.text.split())[:300]}…")
            print(f"   -> read lines {h.start_line}-{h.end_line} of the file for the current content")
        else:
            print(f"{i}. [{label}] {os.path.basename(h.path)}  "
                  f"({h.origin} {h.score:.3f}){warning}")
            print(f"   [SNAPSHOT from {h.indexed_at} — the source cannot be re-read by region]")
            print("   " + h.text.replace("\n", "\n   "))
        print()


def cmd_docs_list(args, cfg):
    docs = core.build_docs(cfg).list_docs(args.scope)
    if args.json:
        output(docs, True)

        return
    if not docs:
        print("nothing indexed")

        return
    import time
    print(f"{len(docs)} document(s):")
    for d in docs:
        if d["expires_at_ts"]:
            expiry = f"expires in {(d['expires_at_ts'] - time.time()) / 3600:5.1f}h"
        else:
            expiry = "permanent      "
        print(f"  [{d['scope']:7}] {d['doc_id']}  {d['chunks']:>4} chunks  {expiry}  "
              f"{d['mode']:9} {d['path']}")


def cmd_docs_drop(args, cfg):
    """Renders `DocIndex.drop_request`, which owns the decision.

    The three-way choice and the refusal when given none of the three live in the core,
    because the hermes `docs_drop` tool offers the same three and must refuse identically.
    """
    try:
        res = core.build_docs(cfg).drop_request(args.doc_id, args.scope,
                                               purge_tmp=args.purge_tmp,
                                               expired=args.expired)
    except core.DocsError as exc:
        # Exit 2 — "you called it wrong" — the same code the hand-written branch used, and
        # distinct from the 1 `main()` gives a genuine core failure. Only the no-target
        # refusal can land here: argparse's `choices` already rejects a bad `--scope`.
        print(f"{exc}", file=sys.stderr)
        raise SystemExit(2)
    if args.json:
        # Same three keys `docs drop <id> --json` has always printed. The purge and expired
        # branches gain JSON they never had: before, they printed human text and ignored the
        # flag entirely.
        output(res, True)

        return
    if res["status"] == "purged":
        print(f"temporary collection {res['collection']} removed (recreated on next use); "
              f"library untouched")
    elif res["status"] == "swept":
        print("expired entries removed from the temporary archive")
    else:
        print(f"doc_id {res['doc_id']} removed from {res['scope']}")


# ---- repositories ----------------------------------------------------------
#
# Every handler here RENDERS and decides nothing: the scope resolution, the confirmation and
# the translation of `--limit` into the search's two knobs all live in `core.repos`, because
# the hermes tools offer the same four operations and a decision taken in an adapter is a
# decision the two hosts are free to take differently.


def cmd_repos_list(args, cfg):
    out = core.build_repos(cfg).list_request()
    if args.json:
        output(out, True)

        return
    if not out["repos"] and not out["divergent"] and not out["emptied"]:
        # An empty screen reads as "the command is broken", especially right after installing —
        # this is the FIRST thing a new user runs, and the collections do not exist until
        # something is indexed. The same rule the search path already follows: say it.
        print("no repository is indexed yet — declare one with `repos register <name>`, "
              "then index files with `repos add <name> <path>...`")

        return
    claimed = {r["repo"]: r.get("chunks") or 0 for r in out["repos"]}
    for r in out["repos"]:
        # SHOWS THE LIVE CHUNK COUNT, NOT THE REGISTRY'S. The registry's `files`/`chunks` are
        # "as of the last `add_files` that wrote something", and the daemon calls that in
        # batches — so a 20-file `add-all` left this line printing "4 file(s)" and a later
        # one-file `refresh` left it printing 1, for a repository holding 20. The count was
        # never a size; this line was the one presenting it as one. `live_chunks` comes from the
        # facet the listing already asks for, and is None only when the facet was unavailable
        # and the fallback scroll ran — in which case say so rather than print a stale number.
        live = r.get("live_chunks")
        size = f"{live} chunk(s)" if live is not None else "size unknown (facet unavailable)"
        # "never indexed" and not a bare `?`: registering is not indexing, and the empty
        # stamp is a fact about this repo rather than a missing field.
        print(f"{r['repo']:<24} {r.get('label', ''):<24} "
              f"{len(r.get('checkouts') or [])} checkout(s)  {size}  "
              f"{r.get('indexed_at') or 'never indexed'}")
    for name in out["divergent"]:
        # Named out loud: it cannot be listed, so it cannot be dropped by name either.
        print(f"{name:<24} (chunks with no registry entry — run `repos drop {name}`)")
    for name in out["emptied"]:
        # The other direction, and a DIFFERENT repair: this one is listed, and what is wrong
        # is the count it is listed with.
        print(f"{name:<24} (its last indexing wrote {claimed.get(name, 0)} chunk(s) and the "
              f"archive has none — reindex it with `repos add`, or `repos drop {name}`)")


def cmd_repos_register(args, cfg):
    out = core.build_repos(cfg).register_request(args.repo, args.label)
    if args.json:
        output(out, True)

        return
    print(f"registered {out['repo']} ({out['label']}) — index files into it with "
          f"`qctx repos add {out['repo']} <path>...`")


def cmd_repos_search(args, cfg):
    out = core.build_repos(cfg).search_request(args.query, repo=args.repo,
                                               across=args.across, limit=args.limit)
    if args.json:
        # The core's conversion, not `default=str`: a `RepoHit` rendered as its repr is a
        # string nothing can address a file by. The hermes tool applies the same one.
        output(core.repos.search_payload(out), True)

        return
    for group in out["groups"]:
        print(f"\n=== {group['repo']}")
        for hit in group["hits"]:
            flag = f"  [{hit.stale}]" if hit.stale else ""
            print(f"  {hit.score:.3f}  {hit.path}:{hit.start_line}-{hit.end_line}{flag}")
    if out["note"]:
        # The loop above renders NOTHING when there are no groups, and an empty screen is
        # read as "there is nothing about this" — the one conclusion this feature must not
        # let anyone draw. The core writes the sentence; both hosts carry the same one.
        print(out["note"])
    if out["truncated"]:
        # Never a claim of absence: the groups beyond the ceiling were never judged.
        print("\n(more repositories matched than --limit allowed — raise it to see them)")


def cmd_repos_add(args, cfg):
    out = core.build_repos(cfg).add_files(args.repo, args.paths)
    if args.json:
        output(out, True)

        return
    print(f"{out['files']} file(s), {out['chunks']} chunk(s) under {out['repo']}")
    for path, why in out["skipped"]:
        print(f"  skipped {path}: {why}")


def cmd_repos_refresh(args, cfg):
    report = core.build_repos(cfg).refresh(args.repo)
    if args.json:
        output(report, True)

        return
    counts = {}
    for row in report:
        counts[row["action"]] = counts.get(row["action"], 0) + 1
        if row["action"] != "ok":
            print(f"{row['action']:<10} {row['path']}"
                  + (f"  ({row['reason']})" if row.get("reason") else ""))
    if not report:
        # The same rule the listing follows: an empty screen reads as a broken command.
        print(f"nothing is indexed under {args.repo!r}")

        return
    print(f"\n{counts.get('reindexed', 0)} reindexed, {counts.get('ok', 0)} unchanged, "
          f"{counts.get('missing', 0)} missing, {counts.get('skipped', 0)} skipped")


def cmd_repos_drop(args, cfg):
    out = core.build_repos(cfg).drop_request(args.repo, args.yes)
    if args.json:
        output(out, True)

        return
    if out["already_gone"]:
        # A different outcome, and it needs different words: no archive was deleted here —
        # it was already gone, and what this run removed is what outlived it. Printing
        # "dropped" would report a deletion that happened in some earlier, failed run.
        print(f"{out['repo']}: the archive was already gone; cleared "
              f"{len(out['unbound'])} stale binding(s)")
    else:
        print(f"dropped {out['repo']}; unbound {len(out['unbound'])} checkout(s)")


def cmd_repos_init(args, cfg):
    from core import bindings

    root = bindings.git_root(args.path or os.getcwd())
    if not root:
        raise SystemExit("not inside a git working copy — run this from a project, or pass --path")
    found = core.build_repos(cfg).candidates_for(root, bindings.remotes_of(root))
    found["root"] = root
    found["indexed"] = False
    if args.json:
        output(found, True)

        return
    if found["bound"]:
        print(f"already indexed as {found['bound']!r}")

        return
    if found["join"]:
        names = ", ".join(sorted(r["repo"] for r in found["join"]))
        print(f"this working copy shares a remote with: {names}")
    if found["taken"]:
        # Named rather than merged: two unrelated checkouts with the same directory name are not
        # the same repository, and deciding otherwise by an accident of naming is what the
        # declared-identity rule exists to refuse.
        print(f"the name {found['suggest']!r} already belongs to another repository")
    print(f"index this working copy as:  qctx repos add-all {found['suggest']}")


def cmd_repos_add_all(args, cfg):
    """Registers `args.repo` if needed, binds this checkout to it, and queues indexing.

    REGISTERS BEFORE ENQUEUEING. `RepoIndex.add_files` refuses to write chunks under a name
    the registry does not know — correctly, `list_repos` and `divergent_repos` both depend on
    the registry being authoritative — so without this, the FIRST batch this job ever ran
    would raise, and the job would land FAILED. This is the primary path the README, `repos
    init`'s own printed advice and the skill all send the user down, so it must actually work.

    CALLS `register`, NOT `register_request`, and WITH THE CHECKOUT. `register_request`'s own
    docstring says as much: it is for a caller that has only a name, and "register still
    accepts both [checkout and remotes]... for the caller that HAS them" — this command has
    `root` from `bindings.git_root` above. Recording it is not optional plumbing: the
    watcher's discovery of newly-tracked files (`core/indexer.py`'s `_new_tracked_paths`)
    walks a repo's registered `checkouts` to know what to run `git ls-files` against, and
    `register_request` never populates that list (every caller passes an empty checkout) — so
    calling it here would leave the registry entry without a root the watcher could ever scan.
    `register` is idempotent for a repeat call on the SAME checkout (accumulates without
    duplicating) and correctly extends the list for a SECOND checkout of an already-registered
    repo, so it is called every time rather than only when the name is brand new.

    IF REGISTRATION FAILS — a taken, non-slug name, most likely — IT RAISES, same as every
    other `CoreError` this CLI lets `main()` print and exit on. Nothing is enqueued: work that
    cannot be written into is work that would fail its first batch, and this command must not
    promise indexing it cannot deliver.

    BINDS THE CHECKOUT TO THE REPO NAME, closing the offer `repos init` makes: `candidates_for`
    reads `bindings.get(root)`, and until this call bound anything, that read was permanently
    fed by an empty write half — the offer to index never went away even after a successful
    index. This is the ONE place a checkout is accepted as "this is that repository", because
    it is the one command that turns "detected" into "working on it".
    """
    from core import bindings, daemon, jobs, lease, scan

    root = bindings.git_root(args.path or os.getcwd())
    if not root:
        raise SystemExit("not inside a git working copy — pass --path")
    index = core.build_repos(cfg)
    index.register(args.repo, args.repo, bindings.remotes_of(root), root)
    bindings.bind(root, args.repo)
    found = scan.eligible(root)
    dropped = ", ".join(f"{n} {k}" for k, n in sorted(found["skipped"].items()) if n)
    print(f"{found['tracked']} tracked → {len(found['eligible'])} eligible"
          + (f" ({dropped})" if dropped else ""))
    if not found["eligible"]:
        print("nothing to index")

        return
    jobs.enqueue(args.repo, "index", found["eligible"])
    if lease.live():
        # REPORTS THE QUEUE EVEN WHEN THE START FAILS. The enqueue above already SUCCEEDED —
        # it raises rather than lying, so reaching this line means the work is really on disk.
        # Letting `DaemonError` travel to the top from here would print only "could not start
        # the daemon" and leave the user believing nothing was queued, when in fact the job is
        # waiting and the next session that opens will drain it. A failure that reads as
        # "there is nothing" is the one outcome this project refuses.
        try:
            started = daemon.start()
            print(f"queued under {args.repo!r}; "
                  f"daemon {started['action']} (pid {started['pid']})")
        except daemon.DaemonError as exc:
            print(f"queued under {args.repo!r}, but the daemon could not be started: {exc}")
            print("the job is safe and the next claude or hermes session will run it")
    else:
        # A daemon spawned here would exit on its very own first cycle — `daemon.run` checks
        # `lease.live()` before touching anything else — so starting one now would only print
        # a pid that is already gone by the time `status` is read next; that is the exact
        # "queued... daemon started... daemon: not running, with nothing connecting the two"
        # failure the whole-branch review reproduced. The job IS queued for real, though: the
        # NEXT claude or hermes session to start writes a lease, and the lease write point in
        # both hosts now also calls `daemon.start()` (see `hooks/lease.py` and
        # `hosts/hermes/__init__.py`), so the queue drains automatically once a session opens.
        # Refusing to queue at all was the other option; this one was chosen because the work
        # is real and will run — just not this second — and saying so plainly is not a lie the
        # way printing "daemon started" would be.
        # DOES NOT OFFER `qctx repos daemon start` HERE. That command spawns `repos daemon
        # run`, whose first act is to check for a live lease and return immediately when there
        # is none — so from a bare terminal it prints "daemon started (pid N)" and `status`
        # then reports "daemon: not running". Suggesting it would hand the user, as the remedy,
        # the exact confusing failure this message exists to prevent.
        print(f"queued under {args.repo!r} — no claude or hermes session is open right now, "
              f"and the daemon only runs while one is. It will start automatically the next "
              f"time a session opens; open one to have it run now.")
    print("watch it with:  qctx repos status")


def cmd_repos_status(args, cfg):
    from core import daemon, jobs, lease

    # REAPS BEFORE READING, because `status` is the ONLY reader that runs with no guarantee a
    # daemon is alive to do it for itself — `reap` otherwise runs solely from inside
    # `daemon.run`'s own loop, on ITS OWN cycle. Without this, a daemon that died mid-job
    # leaves `jobs/<repo>.json` reading `running` forever: not stale data that self-corrects,
    # a LIE that gets worse the longer nobody looks, because the pid it names never comes back
    # to finish writing `done`. Same predicate `daemon.run` itself hands to `reap` — raw pid
    # liveness, not a comparison against the CURRENT daemon record, so a job whose daemon is
    # genuinely still running is never reaped out from under it just because some OTHER
    # process currently holds the daemon claim.
    jobs.reap(lambda pid: lease.process_start(pid) is not None)
    running = daemon.record()
    rows = jobs.all_jobs()
    if args.json:
        output({"daemon": running, "jobs": rows, "leases": lease.live()}, True)

        return
    # Said first and plainly: every number below was written by a process that may be gone, and
    # a reader who does not know that reads stalled progress as activity.
    print(f"daemon: {'running (pid %d)' % running['pid'] if running else 'not running'}")
    if not rows:
        print("no indexing jobs")

        return
    for job in rows:
        pct = f"{100 * job['done'] // job['total']}%" if job.get("total") else "—"
        line = f"  {job['repo']:<24} {job['state']:<10} {job['done']}/{job['total']} {pct}"
        print(line + (f"  {job['error']}" if job.get("error") else ""))


def cmd_repos_cancel(args, cfg):
    from core import jobs

    if jobs.request_cancel(args.repo):
        print(f"cancel requested for {args.repo!r} — what is already indexed stays")
    else:
        print(f"no job for {args.repo!r}")


def cmd_repos_daemon(args, cfg):
    from core import daemon, indexer

    if args.action == "stop":
        print("daemon stopped" if daemon.stop() else "no daemon was running")

        return
    if args.action == "start":
        out = daemon.start()
        print(f"daemon {out['action']} (pid {out['pid']})")

        return
    # `run` is what the detached process executes. It is a command rather than a flag so the
    # daemon is startable by hand when something needs to be watched directly.
    daemon.run(indexer.work(cfg), watch=indexer.watcher(cfg))


# ---- parser ----------------------------------------------------------------

def _propagate_json(parser: argparse.ArgumentParser) -> None:
    """Adds `--json` to EVERY subcommand, recursively.

    People type the flag at the end (`qctx memory find x --json`) and the documentation
    promised it worked, but it existed only on the top-level parser — the natural call
    failed with "unrecognized arguments". Walking the subparsers after they are built
    solves it in one place; repeating the definition across twenty `add_parser` calls
    would be the same duplication this project spent an afternoon removing.
    """
    for action in parser._subparsers._group_actions if parser._subparsers else []:
        for sub in getattr(action, "choices", {}).values():
            if not any(o == "--json" for a in sub._actions for o in a.option_strings):
                sub.add_argument("--json", action="store_true", help="JSON output")
            _propagate_json(sub)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="qctx", description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="JSON output")
    sub = ap.add_subparsers(dest="group", required=True)

    p = sub.add_parser("setup", help="guided configuration diagnostics")
    p.add_argument("--check", action="store_true",
                   help="diagnose only, never ask and never change anything")
    p.set_defaults(fn=cmd_setup)

    p = sub.add_parser("install", help="install and verify everything, step by step")
    p.add_argument("--check", action="store_true", help="report only; never writes")
    p.add_argument("--yes", action="store_true", help="answer yes to every group")
    p.add_argument("--config-only", action="store_true",
                   help="only the configuration pass; touch no host")
    p.set_defaults(fn=cmd_install)

    col = sub.add_parser("collections", help="inspect Qdrant collections")
    colsub = col.add_subparsers(dest="action", required=True)
    p = colsub.add_parser("list")
    p.add_argument("--all", action="store_true",
                   help="include collections from other systems (ws-*)")
    p.set_defaults(fn=cmd_collections)

    cfgp = sub.add_parser("config", help="view or change configuration")
    cfgsub = cfgp.add_subparsers(dest="action", required=True)
    cfgsub.add_parser("show").set_defaults(fn=cmd_config_show)
    cfgsub.add_parser("detect", help="detect the model dimension and store it").set_defaults(
        fn=cmd_config_detect)
    p = cfgsub.add_parser("set")
    p.add_argument("key")
    p.add_argument("value")
    p.set_defaults(fn=cmd_config_set)

    mem = sub.add_parser("memory", help="long-term semantic memory")
    memsub = mem.add_subparsers(dest="action", required=True)

    p = memsub.add_parser("store")
    p.add_argument("text")
    p.add_argument("--type")
    p.add_argument("--project")
    p.add_argument("--area")
    p.add_argument("--json-meta", dest="json_meta")
    p.set_defaults(fn=cmd_memory_store)

    p = memsub.add_parser("find", help="dense search (cheap, no re-rank)")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(fn=cmd_memory_find)

    p = memsub.add_parser("recall", help="search with re-rank (two gates)")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--dense-floor", type=float, default=0.45)
    p.add_argument("--strict-floor", type=float, default=0.58)
    p.add_argument("--min-score", type=float, default=0.10)
    p.add_argument("--top-k", type=int, default=20)
    p.set_defaults(fn=cmd_memory_recall)

    p = memsub.add_parser("get")
    p.add_argument("id")
    p.set_defaults(fn=cmd_memory_get)

    p = memsub.add_parser("delete")
    p.add_argument("id")
    p.set_defaults(fn=cmd_memory_delete)

    p = memsub.add_parser("update")
    p.add_argument("id")
    p.add_argument("--text")
    p.add_argument("--type")
    p.add_argument("--project")
    p.add_argument("--area")
    p.add_argument("--json-meta", dest="json_meta")
    p.set_defaults(fn=cmd_memory_update)

    p = memsub.add_parser("list")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(fn=cmd_memory_list)

    p = memsub.add_parser("store-many", help="a batch of facts, all-or-nothing")
    p.add_argument("file", nargs="?", default="-",
                   help="JSON file with [{information, metadata?}], or - for stdin")
    p.set_defaults(fn=cmd_memory_store_many)

    p = memsub.add_parser("search-collections", help="read-only search in other archives")
    p.add_argument("query")
    p.add_argument("--collections", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(fn=cmd_memory_search_collections)

    docs = sub.add_parser("docs", help="ephemeral index for long documents")
    docsub = docs.add_subparsers(dest="action", required=True)

    p = docsub.add_parser("index", help="index as TEMPORARY (with a TTL)")
    p.add_argument("path")
    p.add_argument("--ttl", default="24h", help="30m, 24h, 7d (default 24h)")
    p.add_argument("--doc-id", dest="doc_id", default=None)
    p.set_defaults(fn=cmd_docs_index)

    p = docsub.add_parser("keep", help="keep in the LIBRARY, with no expiry")
    p.add_argument("path")
    p.add_argument("--doc-id", dest="doc_id", default=None)
    p.set_defaults(fn=cmd_docs_keep)

    p = docsub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--scope", choices=core.docs.SCOPES, default="all")
    p.add_argument("--doc-id", dest="doc_id", default=None)
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(fn=cmd_docs_search)

    p = docsub.add_parser("list")
    p.add_argument("--scope", choices=core.docs.SCOPES, default="all")
    p.set_defaults(fn=cmd_docs_list)

    p = docsub.add_parser("refresh", help="reindex what changed on disk")
    p.add_argument("--scope", choices=("library", "tmp"), default="library")
    p.set_defaults(fn=cmd_docs_refresh)

    p = docsub.add_parser("drop")
    p.add_argument("doc_id", nargs="?", default=None)
    p.add_argument("--scope", choices=core.docs.SCOPES, default="all")
    p.add_argument("--purge-tmp", dest="purge_tmp", action="store_true",
                   help="delete the entire temporary collection (library untouched)")
    p.add_argument("--expired", action="store_true")
    p.set_defaults(fn=cmd_docs_drop)

    rep = sub.add_parser("repos", help="repository archives, grouped by repo")
    repsub = rep.add_subparsers(dest="repos_cmd", required=True)

    repsub.add_parser("list", help="every indexed repository").set_defaults(fn=cmd_repos_list)

    p = repsub.add_parser("search", help="search one repository, or every one")
    p.add_argument("query")
    p.add_argument("--repo", help="the repository to search (default: the one you are in)")
    p.add_argument("--all", action="store_true", dest="across",
                   help="search every indexed repository")
    p.add_argument("--limit", type=int, default=8)
    p.set_defaults(fn=cmd_repos_search)

    p = repsub.add_parser("register", help="declare a repository, by name")
    p.add_argument("repo", help="a slug: lower case, digits and hyphens (it is the filter "
                                "key, so it can never change)")
    p.add_argument("--label", help="a human name for it (default: the name itself)")
    p.set_defaults(fn=cmd_repos_register)

    p = repsub.add_parser("add", help="index the given files under a repository")
    p.add_argument("repo")
    p.add_argument("paths", nargs="+")
    p.set_defaults(fn=cmd_repos_add)

    p = repsub.add_parser("refresh", help="reindex the files that changed on disk")
    p.add_argument("repo")
    p.set_defaults(fn=cmd_repos_refresh)

    p = repsub.add_parser("init", help="detect the repository here and OFFER to index it — "
                                       "never indexes on its own")
    p.add_argument("--path", help="the working copy (default: the current directory)")
    p.set_defaults(fn=cmd_repos_init)

    p = repsub.add_parser("add-all", help="index a whole repository, in the background")
    p.add_argument("repo")
    p.add_argument("--path", help="the working copy (default: the current directory)")
    p.set_defaults(fn=cmd_repos_add_all)

    p = repsub.add_parser("status", help="what is being indexed, and whether the daemon is up")
    p.set_defaults(fn=cmd_repos_status)

    p = repsub.add_parser("cancel", help="stop indexing a repository; what is indexed stays")
    p.add_argument("repo")
    p.set_defaults(fn=cmd_repos_cancel)

    p = repsub.add_parser("daemon", help="the background indexer")
    p.add_argument("action", choices=["start", "stop", "run"])
    p.set_defaults(fn=cmd_repos_daemon)

    p = repsub.add_parser("drop", help="delete a repository archive, permanently")
    p.add_argument("repo")
    p.add_argument("--yes", action="store_true", help="skip the confirmation")
    p.set_defaults(fn=cmd_repos_drop)

    _propagate_json(ap)

    return ap


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.fn(args, core.load())
    except core.CoreError as exc:
        # The root of the hierarchy: a new core error is caught here the day it is born.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
