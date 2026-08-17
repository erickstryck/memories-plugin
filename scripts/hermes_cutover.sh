#!/usr/bin/env bash
# Install memories-plugin as the hermes-agent memory provider, in ONE atomic pass.
#
# hermes activates exactly ONE external memory provider (`MemoryManager.add_provider`
# rejects the second with a warning), so this REPLACES whatever `memory.provider` names —
# here a third-party `qdrant` provider holding 1423 points in its own collection. That
# provider is disabled by CONFIGURATION, never deleted, and its points are left untouched:
# they stay reachable read-only with
#     qctx memory search-collections "<topic>" --collections hermes_memory
#
# With no argument this is a DRY RUN and writes nothing. With --apply it applies, after
# taking a dated backup of config.yaml.
#
#     ./scripts/hermes_cutover.sh            # check + dry run
#     ./scripts/hermes_cutover.sh --apply    # apply
#
# Close other hermes sessions first: the provider is selected and loaded at agent init, so
# a running session keeps the old one either way.
#
# WHY THIS SCRIPT CHECKS THE ENVIRONMENT AND NOT ONLY THE FILES. An unavailable provider is
# never initialized, so nothing it could log from `initialize()` is ever reached. hermes
# 0.20.1 does warn (`agent/agent_init.py::_warn_memory_provider_unavailable`, which appends
# our `unavailable_reason()`), but that warning is a log line at session start, and the
# failure it cannot describe at all is the one this script exists for: the two API keys live
# ONLY in the environment — `core.save()` refuses to write them to config.json, and
# `is_available()` does not look at them. So a hermes started from a shell without them
# reports a perfectly healthy provider and then fails every single search. That diagnosis
# has to happen here, before the switch is flipped.
set -euo pipefail

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
CONFIG="$HERMES_HOME/config.yaml"
DOTENV="$HERMES_HOME/.env"
# FLAT, and measured rather than assumed: `plugins/memory/__init__.py` scans
# `$HERMES_HOME/plugins/<name>/` and resolves a name with `find_provider_dir`, which joins
# that directory with the provider name. A provider one level deeper — the layout the
# qdrant provider on this machine actually uses, `plugins/memory/qdrant/` — is NOT
# discovered (probed against the installed loader: `find_provider_dir` returned None for it
# and the path below for us). `tests/test_hermes_provider.py` holds that measurement.
PLUGINS="$HERMES_HOME/plugins"
LINK="$PLUGINS/memories"
TARGET="$ROOT/hosts/hermes"
ALLOWLIST="$HERMES_HOME/shell-hooks-allowlist.json"
# The big-file read guard, which is NOT installed by the symlink: the memory provider is
# discovered by directory scan, a shell hook is named by absolute command in config.yaml.
# DERIVED from $ROOT like every other path here — a path written into the script is the
# path of whoever wrote it, and this one has to survive a clone anywhere.
#
# The path is double-quoted inside the YAML value because hermes runs the command through
# `shlex.split(os.path.expanduser(command))` (`agent/shell_hooks.py`), so a checkout under a
# directory with a space in it would otherwise be split into two arguments. `hooks/hooks.json`
# quotes the sibling hook for the same reason.
GUARD="$TARGET/bigfile.py"
GUARD_CMD="python3 \"$GUARD\""
STAMP="$(date +%Y%m%d-%H%M%S)"

usage() {
  printf 'usage: %s [--apply] [--i-know-its-a-worktree]\n' "$0" >&2
  printf '  no argument: check the install and report what would change\n' >&2
  printf '  --apply:     apply it, after backing up config.yaml\n' >&2
  printf '  --i-know-its-a-worktree: allow --apply from a git worktree (see below)\n' >&2
}
APPLY=no
WORKTREE_OK=no
for arg in "$@"; do
  case "$arg" in
    --apply)                  APPLY=yes ;;
    --i-know-its-a-worktree)  WORKTREE_OK=yes ;;
    # Anything else is a usage error and NOT a dry run: a mistyped `--aply` that silently
    # became an ensaio, or a `-apply` that silently became an apply, is a worse outcome than
    # an error message.
    *)                        usage; exit 2 ;;
  esac
done

failed=0
say()  { printf '%s\n' "$*"; }
ok()   { say "  ok    $*"; }
fail() { say "  FAIL  $*"; failed=1; }
warn() { say "  WARN  $*"; }
note() { say "  ..    $*"; }

# ---------------------------------------------------------------------------- checks
say "=== checks ==="

[ -d "$HERMES_HOME" ] && ok "HERMES_HOME at $HERMES_HOME" \
                      || fail "no HERMES_HOME directory at $HERMES_HOME"
[ -f "$CONFIG" ] && ok "config.yaml found" || fail "no config.yaml at $CONFIG"
command -v python3 >/dev/null && ok "python3 available" || fail "python3 is required"

# The suite the script runs CONTAINS the tests that run the script, so its own tests set
# this. Announced as a WARN and never as an "ok": a skipped check must not read like a
# passing one.
if [ -n "${HERMES_CUTOVER_SKIP_SUITE:-}" ]; then
  # It gates the DRY RUN only. A stray export in a shell would otherwise apply the cutover
  # with the suite unverified, which is the one thing this variable must never buy.
  if [ "$APPLY" = yes ]; then
    fail "HERMES_CUTOVER_SKIP_SUITE is set — refusing to --apply with the suite unverified"
    say  "        unset it and run again; it exists for the script's own tests, which would"
    say  "        otherwise re-enter the suite that contains them"
  else
    warn "offline suite NOT run (HERMES_CUTOVER_SKIP_SUITE is set) — it was not verified"
  fi
elif python3 -m unittest discover -s "$ROOT/tests" >/dev/null 2>&1; then
  ok "offline suite passes"
else
  fail "the offline suite does not pass — do not flip the switch like this"
fi

# The provider's OWN verdict, through the same `is_available()`/`unavailable_reason()` pair
# hermes calls, rather than a re-implementation of the config rules that could drift from it.
#
# The SECOND question is the one an interactive shell hides: `core.load(env={})` resolves the
# same settings with the environment layer removed, i.e. what a hermes that inherits no shell
# — systemd, the gateway, cron — would see. Everything but the two API keys can live in
# config.json, so a gap here is fixable and worth naming.
probe_out="$(python3 - "$ROOT" <<'PY' 2>&1 || true
import sys
sys.path.insert(0, sys.argv[1])


def verdict(cfg):
    cfg.require_qdrant()
    cfg.resolved_embed_url()
    cfg.require_memory_collection()


def line(text):
    print(" ".join(str(text).split()) or "no reason given")


try:
    import core
    from hosts.hermes import MemoriesProvider
    p = MemoriesProvider()
    print("1" if p.is_available() else "0")
    line(p.unavailable_reason())
    try:
        verdict(core.load(env={}))          # the file alone, no shell
        print("1")
        line("")
    except Exception as exc:                # noqa: BLE001 - reported, never raised
        print("0")
        line(exc)
except Exception as exc:                    # noqa: BLE001 - reported, never raised
    # Four lines, because the reader below always reads four. Both reasons carry the real
    # exception: this is the one path where the operator has NOTHING else to go on — the
    # adapter did not even import, so no later check can say anything more specific. A
    # generic second line here would throw away the only clue in a script whose whole
    # purpose is to explain why memory is not working.
    detail = f"the adapter did not import: {exc}"
    print("0")
    line(detail)
    print("0")
    line(detail)
PY
)"
probe_ok="$(printf '%s\n' "$probe_out" | sed -n 1p)"
probe_reason="$(printf '%s\n' "$probe_out" | sed -n 2p)"
file_ok="$(printf '%s\n' "$probe_out" | sed -n 3p)"
file_reason="$(printf '%s\n' "$probe_out" | sed -n 4p)"
if [ "$probe_ok" = "1" ]; then
  ok "the plugin reports itself available"
else
  fail "the plugin reports itself UNAVAILABLE: $probe_reason"
fi

# ------------------------------------------------------------------- the environment
say ""
say "=== credentials (environment only — they are not in config.json) ==="

# EVERY spelling each key accepts, canonical first, taken from `ENV_ALIASES` in
# core/config.py rather than from memory — the Qdrant key has THREE (`QDRANT_API_KEY` is the
# upstream Qdrant name), and checking two of them turns a working environment into a FAIL
# that tells the operator to export what they already have. Only NAMES are printed, never
# values: this report gets pasted into issues and chats.
dotenv_has() {
  [ -f "$DOTENV" ] || return 1
  grep -qE "^[[:space:]]*(export[[:space:]]+)?$1=[^[:space:]]" "$DOTENV"
}
in_shell=""; in_dotenv=""; missing=""; dotenv_gap=""
check_key() {
  # $1 is a human label, $2 the canonical name, the rest its legacy aliases.
  local label="$1" canonical="$2" shell_hit="" dotenv_hit="" name
  shift 2
  # Spelled out rather than as `A || B && C`: that list is exempt from `set -e` only by a
  # rule about which command follows the final connector, which is not something the next
  # reader should have to know to be sure the script does not exit here.
  for name in "$canonical" "$@"; do
    # The name that is ACTUALLY set, not the canonical one — the operator debugging this
    # needs to know which of the three spellings the value came from.
    if [ -z "$shell_hit" ] && [ -n "${!name:-}" ]; then shell_hit="$name"; fi
    if [ -z "$dotenv_hit" ] && dotenv_has "$name"; then dotenv_hit="$name"; fi
  done
  if [ -n "$shell_hit" ]; then in_shell="$in_shell $shell_hit"; fi
  if [ -n "$dotenv_hit" ]; then in_dotenv="$in_dotenv $dotenv_hit"; fi
  local rest="$*"
  rest="${rest// /, }"
  # Per key, not by substring-matching the accumulated list: `QDRANT_SERVICE_API_KEY` does
  # not contain the string `QDRANT_API_KEY`, and a glob that assumed it did reported a gap
  # for a key that was right there.
  if [ -z "$dotenv_hit" ]; then
    dotenv_gap="$dotenv_gap
    $label — $canonical, or $rest"
  fi
  if [ -z "$shell_hit" ] && [ -z "$dotenv_hit" ]; then
    missing="$missing
    $label — $canonical, or $rest"
  fi
}
check_key "the Qdrant key"           QCTX_QDRANT_API_KEY QDRANT_SERVICE_API_KEY QDRANT_API_KEY
check_key "the embedding-server key" QCTX_API_KEY SERVER_API_KEY

if [ -n "$missing" ]; then
  fail "these keys are in neither this shell nor $DOTENV:$missing"
  say "        They cannot go into config.json — \`config set\` refuses a secret, because a"
  say "        plaintext key ends up in backups and in dotfile sync. Export them before"
  say "        starting hermes:    set -a; . ~/.secrets; set +a"
  say "        or, to cover the gateway too, add them to $DOTENV (chmod 600)."
else
  [ -n "$in_shell" ]  && ok "in this shell:${in_shell}"
  [ -n "$in_dotenv" ] && ok "in $DOTENV:${in_dotenv}"
  # hermes loads $HERMES_HOME/.env itself, with override=True, from run_agent.py and
  # cli.py (`hermes_cli/env_loader.py::load_hermes_dotenv`). A systemd/gateway hermes
  # inherits no interactive shell, so keys that live only in ~/.secrets reach a hermes you
  # launch by hand and NOT the gateway — the exact shape of hermes-agent#2765.
  if [ -n "$dotenv_gap" ]; then
    warn "not in $DOTENV, so a systemd/gateway hermes will not have it:$dotenv_gap"
  fi
fi

# The OTHER half of the same question, and the half the printed remedy above cannot fix. The
# keys are secrets and can only come from the environment; the URLs and the collection names
# are NOT, so they belong in config.json — which both hosts read with no shell involved. On
# this machine they are `export`ed from ~/.bashrc, so an interactive hermes has them and a
# gateway hermes does not, and telling that operator to put the KEYS in .env would leave them
# still memory-less with no idea why.
if [ "$file_ok" = "1" ]; then
  ok "every non-secret setting is in config.json — a hermes with no shell finds them"
else
  warn "with no shell environment the plugin is NOT configured: $file_reason"
  say  "        These are not secrets, so they belong in the config file both hosts read:"
  say  "            qctx config set qdrant-url \"<url>\"   ·   qctx config set embed-url \"<url>\""
  say  "            qctx config set memory-collection \"<name>\""
  say  "        Without that, only a hermes started from a shell that exports them has memory."
fi

# --------------------------------------------------------------------- configuration
say ""
say "=== configuration ==="

read_memory_key() { read_key memory "$1"; }

# $1 is the top-level block, $2 the key under it. `memory:` is not the only block this
# script has to read: the guard's fallback window is resolved from `model.default`.
read_key() {
  python3 - "$CONFIG" "$1" "$2" <<'PY' 2>/dev/null || true
import re, sys
try:
    src = open(sys.argv[1], encoding="utf-8").read()
except OSError:
    sys.exit(0)
block, want, inside, indent = sys.argv[2], sys.argv[3], False, None
#: The LAST direct child wins, not the first. That is what a YAML parser resolves a duplicate
#: key to (PyYAML keeps the last and does not complain), and hermes reads this file with
#: yaml.safe_load. Reading the first would let a file holding BOTH
#: `provider: memories` and `provider: qdrant` answer "memories" while hermes loads qdrant —
#: which is exactly the corruption the post-write verification exists to catch.
found = None
for line in src.splitlines():
    if re.match(r"^%s:\s*(#.*)?$" % re.escape(block), line):
        inside = True
        continue
    if inside:
        if line.strip() and not line[0].isspace():
            break
        m = re.match(r"^(\s+)([A-Za-z0-9_]+):\s*(.*?)\s*$", line)
        if not m:
            continue
        if indent is None:
            indent = m.group(1)
        if m.group(1) == indent and m.group(2) == want:
            found = m.group(3)
if found is not None:
    print(found)
PY
}

current="$(read_memory_key provider)"
if [ "$current" = "memories" ]; then
  ok "memory.provider is already selected: memories"
elif [ -n "$current" ]; then
  note "memory.provider: $current -> memories"
else
  note "memory.provider is unset -> memories"
fi

enabled="$(read_memory_key memory_enabled)"
# Deliberately NOT a requirement: `memory_enabled` gates only hermes' own file-backed
# MEMORY.md store (agent/agent_init.py), while the external provider block is gated on
# `memory.provider` alone. Reported so nobody reads its absence as the cause of a failure.
note "memory.memory_enabled: ${enabled:-<unset>} (gates hermes' built-in store, not this plugin)"

# ------------------------------------------------------------------ where it installs
say ""
say "=== install location ==="

if [ -L "$LINK" ]; then
  current_target="$(readlink "$LINK")"
  if [ "$current_target" = "$TARGET" ]; then
    ok "already installed: $LINK -> $TARGET"
  else
    note "$LINK points at $current_target — will repoint it at $TARGET"
  fi
elif [ -e "$LINK" ]; then
  fail "$LINK exists and is NOT a symlink — move it aside by hand"
else
  note "$LINK will be created -> $TARGET"
fi

# The symlink is only as durable as what it points at, and a git worktree is a temporary
# checkout: remove it and hermes is left with a dangling `plugins/memories`, which
# discovery skips — no provider, no error, no memory. Install from the main checkout.
#
# `--apply` REFUSES rather than warning, because the failure it prevents is the silent one:
# discovery skips a dangling `plugins/memories` and `load_memory_provider` returns None, and
# hermes 0.20.1 does not warn on the None case at all — `agent/agent_init.py:1784-1798` only
# warns when `_mp is not None`. So the operator gets a hermes with no memory and nothing
# anywhere says why. A WARN in the middle of a forty-line report is easy to scroll past.
case "$ROOT" in
  */.claude/worktrees/*|*/.git/worktrees/*)
    if [ "$APPLY" = yes ] && [ "$WORKTREE_OK" != yes ]; then
      fail "$ROOT is a git WORKTREE — refusing to --apply from it: the symlink dies when the"
      say  "        worktree is removed, and a dangling plugins/memories is skipped SILENTLY"
      say  "        (load_memory_provider returns None, and hermes does not warn on None)."
      say  "        Install from the main checkout, or pass --i-know-its-a-worktree."
    else
      warn "$ROOT is a git WORKTREE — the symlink dies when the worktree is removed."
      say  "        Install from the main checkout once this branch is merged."
    fi ;;
esac

# The two-plausible-locations lesson, and not a hypothetical one: the provider being
# replaced lives at $PLUGINS/memory/qdrant, a layout the memory loader never scans.
if [ -e "$PLUGINS/memory/memories" ]; then
  warn "$PLUGINS/memory/memories exists and is NOT discovered — the loader scans"
  say "        $PLUGINS/<name>/ only (one level, not two). Nothing reads that copy."
fi
if [ -n "${HERMES_PLUGIN_DIR:-}" ] && [ "$HERMES_PLUGIN_DIR" != "$PLUGINS" ]; then
  warn "HERMES_PLUGIN_DIR=$HERMES_PLUGIN_DIR is set, and the memory loader IGNORES it:"
  say "        it builds get_hermes_home()/plugins itself. Installing into it does nothing."
fi

# The marker the loader's cheap text scan looks for, before importing anything
# (`_is_memory_provider_dir`, first 8192 bytes). Without it the directory is not even
# considered a provider — and the failure looks like "the provider does not exist".
if head -c 8192 "$TARGET/__init__.py" 2>/dev/null \
   | grep -qE "register_memory_provider|MemoryProvider"; then
  ok "the adapter carries the string discovery greps for"
else
  fail "$TARGET/__init__.py has neither register_memory_provider nor MemoryProvider in"
  say "        its first 8192 bytes — discovery would skip the directory entirely"
fi

# Where the provider being REPLACED actually lives, which decides what is being replaced:
# a working provider, or one that never loaded. Only the two directory sources can be
# checked from here — a pip entry-point provider (`importlib.metadata`, group
# hermes.memory_providers) lives in hermes' own venv, so absence is reported as a note and
# never as a failure.
BUNDLED="$HERMES_HOME/hermes-agent/plugins/memory"
if [ -n "$current" ] && [ "$current" != memories ]; then
  if [ -d "$PLUGINS/$current" ] || [ -d "$BUNDLED/$current" ]; then
    note "the provider being replaced, $current, is where the loader looks"
  else
    note "the provider being replaced, $current, is NOT in $PLUGINS/$current"
    [ -d "$BUNDLED" ] && say "        nor in $BUNDLED/$current"
    if [ -d "$PLUGINS/memory/$current" ]; then
      say "        — its files are at $PLUGINS/memory/$current, one level too deep for"
      say "        discovery, so it is already inert. Its stored points are untouched."
    else
      say "        — it may be a pip entry-point provider in hermes' venv, which cannot be"
      say "        checked from here. Confirm with: hermes memory status"
    fi
  fi
fi

# ------------------------------------------------------------- the big-file read guard
say ""
say "=== the big-file read guard ==="

# The `hooks:` block, read the way hermes' own parser reads it. Measured in the installed
# v0.20.1: `agent/shell_hooks.py::_parse_hooks_block` opens with
# `if not isinstance(hooks_cfg, dict): return []` and then iterates `hooks_cfg.items()` as
# EVENT NAME -> LIST of entries. A block written as a sequence of `{event, command}` mappings
# — the shape that reads naturally and that this feature's plan specified — parses to zero
# hooks and logs NOTHING on that path. Which is why the shape is reported and never assumed.
read_hooks() {
  python3 - "$CONFIG" <<'PY' 2>/dev/null || true
import re, sys


def unquote(value):
    """Only a value quoted at BOTH ends. The command this script writes ends in a quote
    (`python3 "/path/bigfile.py"`) and a naive strip would eat it, so a registered guard
    would never compare equal to the one we would install."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]

    return value


try:
    src = open(sys.argv[1], encoding="utf-8").read()
except OSError:
    print("shape\tnone")
    raise SystemExit

lines = src.splitlines()
start = None
for i, line in enumerate(lines):
    if re.match(r"^hooks:\s*(#.*)?$", line):
        start = i                  # the LAST one wins, as PyYAML resolves a duplicate key
if start is None:
    print("shape\tnone")
    raise SystemExit

body = []
for line in lines[start + 1:]:
    if line.strip() and not line[0].isspace():
        break
    body.append(line)

first = next((l for l in body if l.strip()), "")
if re.match(r"^\s*-\s", first):
    print("shape\tseq")            # hermes reads no hook at all from this
    raise SystemExit
print("shape\tmap")

#: A DIRECT child of `hooks:` and nothing deeper. `hooks:` has reserved sub-keys that are
#: not events — `outbound` and `output_spill`, skipped by name in `_parse_hooks_block` —
#: and a `pre_tool_call:` under one of those is somebody else's configuration. Matching it
#: at any indent would report a guard where hermes reads none, and (in the rewriter below)
#: install one into a section nothing looks at.
child = len(first) - len(first.lstrip())
event = None
for i, line in enumerate(body):
    m = re.match(r"^(\s+)pre_tool_call:\s*(#.*)?$", line)
    if m and len(m.group(1)) == child:
        event = i
if event is None:
    # `pre_tool_call: <scalar>` is a different failure from absent: hermes logs
    # "hooks.%s must be a list of hook definitions" and registers nothing.
    scalar = any(len(m.group(1)) == child for m
                 in (re.match(r"^(\s+)pre_tool_call:\s*\S", l) for l in body) if m)
    print("event\tnotalist" if scalar else "event\tnone")
    raise SystemExit

indent = len(body[event]) - len(body[event].lstrip())
region = []
for line in body[event + 1:]:
    if line.strip() and len(line) - len(line.lstrip()) <= indent:
        break
    region.append(line)
if not any(re.match(r"^\s*-\s", l) for l in region if l.strip()):
    print("event\tnotalist" if any(l.strip() for l in region) else "event\tempty")
    raise SystemExit
print("event\tlist")

entries = []
for line in region:
    item = re.match(r"^\s*-\s*(.*)$", line)
    if item:
        entries.append({})
        rest = item.group(1)
    else:
        rest = line.strip()
    pair = re.match(r"^([A-Za-z0-9_]+):\s*(.*?)\s*$", rest)
    if entries and pair:
        entries[-1][pair.group(1)] = unquote(pair.group(2))
for entry in entries:
    print("entry\t%s\t%s" % (entry.get("matcher") or "", entry.get("command") or ""))
PY
}

# Which entry is OURS: any pre_tool_call hook running a `hosts/hermes/bigfile.py`, whatever
# checkout it points at — a guard pointing at a deleted worktree is the failure worth
# naming, and matching only on the exact command would report it as "not installed".
guard_state() {
  hooks_out="$(read_hooks)"
  hooks_shape="$(printf '%s\n' "$hooks_out" | awk -F'\t' '$1=="shape"{print $2}')"
  guard_entry="$(printf '%s\n' "$hooks_out" \
                 | awk -F'\t' '$1=="entry" && index($3, "hosts/hermes/bigfile.py"){print; exit}')"
  guard_matcher="$(printf '%s' "$guard_entry" | cut -f2)"
  guard_cmd="$(printf '%s' "$guard_entry" | cut -f3)"
}
guard_state

if [ "$hooks_shape" = seq ]; then
  warn "the hooks: block in $CONFIG is a LIST, and hermes reads no hook at all from it."
  say  "        _parse_hooks_block wants a MAPPING of event name to a list of entries and"
  say  "        returns [] for anything else, silently. Fix it by hand, in this shape:"
  say  "            hooks:"
  say  "              pre_tool_call:"
  say  "                - matcher: read_file"
  say  "                  command: $GUARD_CMD"
elif [ -z "$guard_entry" ]; then
  note "big-file guard: not registered -> pre_tool_call, matcher read_file, $GUARD"
elif [ "$guard_cmd" != "$GUARD_CMD" ]; then
  warn "the big-file guard is registered, but it runs $guard_cmd"
  say  "        This checkout would install $GUARD_CMD"
  say  "        Nothing here rewrites an existing hook entry — an entry you wrote is yours."
  say  "        Edit it by hand if that path is stale: a guard pointing at a removed"
  say  "        worktree fails open on every read and says nothing."
else
  ok "big-file guard registered: pre_tool_call -> $GUARD"
fi

if [ -n "$guard_entry" ] && [ "$hooks_shape" != seq ]; then
  # The matcher is not a detail, and this is measured: `write_file` and `patch` take a
  # `path` argument too, so a matcher-less pre_tool_call hook gets an opinion about WRITES,
  # which a read-cost guard has no business blocking. `ShellHookSpec.matches` compiles it as
  # a regex and falls back to literal equality, so `read_file` means exactly that tool.
  if [ -z "$guard_matcher" ]; then
    warn "the big-file guard is registered with no matcher — it will also see write_file and"
    say  "        patch, which take a \`path\` too, and it has no business blocking a WRITE."
  elif [ "$guard_matcher" != read_file ]; then
    warn "the big-file guard's matcher is '$guard_matcher', not read_file — it will fire on"
    say  "        every tool whose name that regex matches, write_file and patch included."
  fi

  # Registered is not wired: `register_from_config` skips any hook whose (event, command)
  # pair is missing from the allowlist unless a TTY approves it, and a gateway hermes has no
  # TTY. This script does NOT write that file — it records the user's consent, and forging
  # consent to run a command is not a thing an install script gets to do.
  if [ -f "$ALLOWLIST" ] && [ -n "$(python3 - "$ALLOWLIST" "$guard_cmd" <<'PY' 2>/dev/null || true
import json, sys
try:
    data = json.loads(open(sys.argv[1], encoding="utf-8").read())
except Exception:                      # noqa: BLE001 - absence and garbage read alike
    raise SystemExit
approvals = data.get("approvals") if isinstance(data, dict) else None
for entry in approvals or []:
    if (isinstance(entry, dict) and entry.get("event") == "pre_tool_call"
            and entry.get("command") == sys.argv[2]):
        print("yes")
        break
PY
)" ]; then
    ok "the guard command is approved in shell-hooks-allowlist.json"
  else
    warn "the guard command is NOT in $ALLOWLIST yet, so hermes will ask once at the TTY"
    say  "        before it runs. A hermes with no TTY (systemd, the gateway) skips the hook"
    say  "        instead, with only a log line: start one interactively once and approve it,"
    say  "        or set hooks_auto_accept: true / HERMES_ACCEPT_HOOKS=1."
  fi
fi

# WHETHER THE WINDOW IS DECLARED, and this is the half that decides whether the guard can
# do anything at all. `core/windows.py` holds CEILINGS per model NAME — the largest window
# any variant of that name can have — because the same bare name ships as a 200k and a 1M
# variant and nothing readable at hook time tells them apart. Erring large only makes the
# guard sleep; erring small would make it block on a guess, the one failure this feature
# must never produce. So: inert until configured is the accepted cost, and it is only
# acceptable while it is VISIBLE. A silently inert guard is indistinguishable from a working
# one, and the user finds out on the day it did not protect them.
model_default="$(read_key model default)"
window_out="$(python3 - "$ROOT" "$model_default" <<'PY' 2>/dev/null || true
import sys
sys.path.insert(0, sys.argv[1])
model = sys.argv[2] if len(sys.argv) > 2 else ""


def declared(**kwargs):
    """0 for anything unreadable: `load()` raises on a malformed number, and this line is a
    report, not a gate."""
    try:
        import core
        return int(getattr(core.load(**kwargs), "context_window", 0) or 0)
    except Exception:                  # noqa: BLE001 - reported as "not declared"
        return 0


try:
    from core.windows import MODEL_WINDOWS
except Exception:                      # noqa: BLE001
    MODEL_WINDOWS = {}
print(declared())
print(declared(env={}))
print(int(MODEL_WINDOWS.get(model.strip(), 0)))
print(", ".join(sorted(MODEL_WINDOWS)))
PY
)"
win_shell="$(printf '%s\n' "$window_out" | sed -n 1p)"
win_file="$(printf '%s\n' "$window_out" | sed -n 2p)"
win_ceiling="$(printf '%s\n' "$window_out" | sed -n 3p)"
win_table="$(printf '%s\n' "$window_out" | sed -n 4p)"

if [ "${win_shell:-0}" != 0 ]; then
  ok "context window declared: $win_shell tokens — the guard measures every read against it"
  if [ "${win_file:-0}" = 0 ]; then
    warn "…but only in this shell. It is not a secret, so put it where a systemd/gateway"
    say  "        hermes will find it:   qctx config set context-window $win_shell"
  fi
elif [ "${win_ceiling:-0}" != 0 ]; then
  warn "context_window is not declared, so the guard falls back to the ceiling for"
  say  "        ${model_default:-<unset>}: $win_ceiling tokens, the LARGEST window any variant of that name"
  say  "        can have. In a session smaller than that the guard stays nearly inert — it"
  say  "        thinks there is room that is not there. Declare the real size to sharpen it:"
  say  "            qctx config set context-window <tokens>   (or export QCTX_CONTEXT_WINDOW)"
else
  warn "context_window is not declared and no ceiling is known for model"
  say  "        '${model_default:-<unset>}' — the window resolves to UNKNOWN, and an unknown window"
  say  "        allows every read. The guard is installed and inert."
  say  "        The table knows: ${win_table:-<empty>}"
  say  "            qctx config set context-window <tokens>   (or export QCTX_CONTEXT_WINDOW)"
fi

if [ "$failed" -ne 0 ]; then
  say ""
  say "checks failed — nothing was changed."
  exit 1
fi

# --------------------------------------------------------------------------- dry run
say ""
say "=== what changes ==="
say "  $LINK -> $TARGET   (symlink; one source of truth)"
say "  $CONFIG: memory.provider: ${current:-<unset>} -> memories"
if [ -z "$guard_entry" ] && [ "$hooks_shape" != seq ]; then
  say "  $CONFIG: hooks.pre_tool_call += matcher read_file -> $GUARD"
fi
say ""
say "  UNCHANGED: every Qdrant collection, the 1423 points in hermes_memory, the"
say "  ${current:-previous} provider's own directory (disabled by configuration, not deleted),"
say "  ~/.secrets and the URLs in .bashrc. The old points stay reachable read-only:"
say "      qctx memory search-collections \"<topic>\" --collections hermes_memory"

if [ "$APPLY" != yes ]; then
  say ""
  say "DRY RUN. To apply: $0 --apply"
  exit 0
fi

# ----------------------------------------------------------------------------- apply
say ""
say "=== applying ==="
# `if`, not `cp ... && ok ...`: with `&&` a failed cp is the last command of its list and
# `set -e` would kill the script here with no message at all — the operator would see the
# banner above and nothing else. And a cutover with no backup does not proceed.
if cp "$CONFIG" "$CONFIG.bak-$STAMP"; then
  ok "backup $CONFIG.bak-$STAMP"
else
  fail "could not write $CONFIG.bak-$STAMP — refusing to change anything without a backup"
  exit 1
fi

# Reported like every neighbouring step instead of dying through `set -e` with no message.
if ! mkdir -p "$PLUGINS"; then
  fail "could not create $PLUGINS — the symlink has nowhere to go"
fi
if [ -L "$LINK" ] || [ ! -e "$LINK" ]; then
  if ln -sfn "$TARGET" "$LINK"; then
    ok "symlink in place"
  else
    fail "could not create $LINK (is $PLUGINS writable?)"
  fi
else
  fail "$LINK exists and is not a symlink — move it aside by hand"
fi

# INDEPENDENT of the step above: a failure there must not abort this one through `set -e`,
# or the cutover ends half done with no report of which half.
tmp="$(mktemp)"
if python3 - "$CONFIG" > "$tmp" <<'PY'
import re, sys

src = open(sys.argv[1], encoding="utf-8").read()
lines = src.splitlines(keepends=True)
out, inside, indent, done = [], False, None, False

for line in lines:
    if re.match(r"^memory:\s*(#.*)?$", line):
        inside = True
        out.append(line)
        continue
    if inside:
        if line.strip() and not line[0].isspace():
            # End of the block with no provider key of its own: insert one, at the
            # indentation the block's other keys use.
            if not done:
                out.append(f"{indent or '  '}provider: memories\n")
                done = True
            inside = False
        else:
            m = re.match(r"^(\s+)([A-Za-z0-9_]+):\s*(.*?)\s*$", line)
            if m:
                if indent is None:
                    indent = m.group(1)
                # Only a DIRECT child of `memory:` — a `provider:` nested deeper belongs to
                # something else (the real config.yaml has one under
                # auxiliary.memory_query_rewrite).
                if m.group(1) == indent and m.group(2) == "provider" and not done:
                    out.append(f"{indent}provider: memories\n")
                    done = True
                    continue
    out.append(line)

if inside and not done:                       # the block ran to the end of the file
    if out and not out[-1].endswith("\n"):
        out.append("\n")
    out.append(f"{indent or '  '}provider: memories\n")
    done = True

if not done:                                  # no `memory:` block at all
    if out and not out[-1].endswith("\n"):
        out.append("\n")
    out.append("memory:\n  provider: memories\n")

sys.stdout.write("".join(out))
PY
then
  if mv "$tmp" "$CONFIG"; then
    # RE-READ, with the same parser that reported the value before the change. The rewriter
    # exiting 0 and `mv` succeeding say the file was replaced, not that `memory.provider` is
    # now `memories`: a rewriter bug can emit a file where the old key survives (two
    # `provider:` lines in the block, the first one winning nothing) and every signal above
    # still looks like success. Printing ok for a state it had not verified is the defect
    # scripts/cutover.sh already paid for once.
    written="$(read_memory_key provider)"
    if [ "$written" = memories ]; then
      ok "memory.provider = memories"
    else
      fail "the rewrite did not take: memory.provider reads '${written:-<unset>}' in $CONFIG"
      say  "        Restore $CONFIG.bak-$STAMP and set the key by hand."
    fi
  else
    rm -f "$tmp"; fail "could not replace $CONFIG — set memory.provider by hand"
  fi
else
  rm -f "$tmp"; fail "could not rewrite config.yaml — set memory.provider by hand"
fi

# INDEPENDENT again, and re-read rather than trusting the state the checks section saw: the
# step above rewrote this same file.
guard_state
if [ "$hooks_shape" = seq ]; then
  fail "the hooks: block is a LIST, so the big-file guard was NOT registered"
  say  "        Rewriting that block into the mapping hermes parses would move every hook it"
  say  "        holds, so this refuses instead. Add the entry by hand:"
  say  "            hooks:"
  say  "              pre_tool_call:"
  say  "                - matcher: read_file"
  say  "                  command: $GUARD_CMD"
elif [ -n "$guard_entry" ] && [ "$guard_cmd" != "$GUARD_CMD" ]; then
  warn "the big-file guard already runs $guard_cmd — left exactly as it is"
else
  tmp="$(mktemp)"
  if python3 - "$CONFIG" "$GUARD_CMD" > "$tmp" <<'PY'
import re, sys

path, cmd = sys.argv[1], sys.argv[2]
src = open(path, encoding="utf-8").read()
lines = src.splitlines(keepends=True)


def bare(line):
    return line.rstrip("\n")


def entry(item_indent):
    """The entry hermes parses: an item of the list under an EVENT NAME key.

    `timeout: 5` is deliberate and not decoration — `DEFAULT_TIMEOUT_SECONDS` is 60 in the
    installed shell_hooks.py, and sixty seconds in front of every file read is a hang, not a
    guard. Five is what `hooks/hooks.json` gives the same hook on the other host.

    `fail_closed` is left at its default of false ON PURPOSE: this guard fails OPEN, and a
    crashed guard must never be what stops someone reading a file.
    """
    return (f"{item_indent}- matcher: read_file\n"
            f"{item_indent}  command: {cmd}\n"
            f"{item_indent}  timeout: 5\n")


def emit(text=None):
    sys.stdout.write(text if text is not None else "".join(lines))
    raise SystemExit


start = None
for i, line in enumerate(lines):
    if re.match(r"^hooks:\s*(#.*)?$", bare(line)):
        start = i                     # the LAST one, the one PyYAML would resolve to
if start is None:
    if lines and not lines[-1].endswith("\n"):
        lines.append("\n")
    lines.append("hooks:\n  pre_tool_call:\n" + entry("    "))
    emit()

end = start + 1
while end < len(lines):
    stripped = bare(lines[end])
    if stripped.strip() and not stripped[0].isspace():
        break
    end += 1
body = lines[start + 1:end]

# Two shapes this refuses to touch rather than repair, and the reason is the same for both:
# the repair would move or drop hooks the user wrote. Exit 3 is reported by the caller.
first = bare(next((l for l in body if l.strip()), ""))
if re.match(r"^\s*-\s", first):
    sys.exit(3)                       # `hooks:` is a sequence

# Only a DIRECT child of `hooks:` is an event list. `outbound:` and `output_spill:` are
# reserved sub-sections there, and a `pre_tool_call:` nested under one of them belongs to
# somebody else — appending to it would install the guard where hermes never looks and
# leave every later check reporting success.
child = re.match(r"^(\s*)", first).group(1)
event = None
for i, line in enumerate(body):
    m = re.match(r"^(\s+)pre_tool_call:\s*(#.*)?$", bare(line))
    if m and m.group(1) == child:
        event = i
    scalar = re.match(r"^(\s+)pre_tool_call:\s*\S", bare(line))
    if scalar and scalar.group(1) == child:
        sys.exit(3)                   # `pre_tool_call: <scalar>`
if event is None:
    lines[start + 1:start + 1] = [f"{child}pre_tool_call:\n" + entry(child + "  ")]
    emit()

indent = len(bare(body[event])) - len(bare(body[event]).lstrip())
region = []
for line in body[event + 1:]:
    stripped = bare(line)
    if stripped.strip() and len(stripped) - len(stripped.lstrip()) <= indent:
        break
    region.append(line)

items = [l for l in region if re.match(r"^\s*-\s", bare(l))]
if not items and any(l.strip() for l in region):
    sys.exit(3)                       # something under the key that is not a list
if any(cmd in bare(l) for l in region):
    emit(src)                         # already registered: idempotent, byte for byte

item_indent = re.match(r"^(\s*)-", bare(items[0])).group(1) if items else " " * (indent + 2)
at = start + 1 + event + 1 + len(region)
while region and not region[-1].strip():   # land inside the list, not after a blank line
    region.pop()
    at -= 1
lines[at:at] = [entry(item_indent)]
emit()
PY
  then
    if mv "$tmp" "$CONFIG"; then
      # Re-read, for the same reason `memory.provider` is re-read: the rewriter exiting 0
      # says a file was produced, not that hermes will find a hook in it.
      if [ "$(read_hooks | awk -F'\t' -v cmd="$GUARD_CMD" \
              '$1=="entry" && $3==cmd {print $2; exit}')" = read_file ]; then
        ok "big-file guard registered: pre_tool_call, matcher read_file -> $GUARD"
      else
        fail "the big-file guard did not take: no pre_tool_call entry runs $GUARD_CMD"
        say  "        Restore $CONFIG.bak-$STAMP and add the entry by hand."
      fi
    else
      rm -f "$tmp"; fail "could not replace $CONFIG — the big-file guard is not registered"
    fi
  else
    rm -f "$tmp"
    fail "could not register the big-file guard in $CONFIG"
    say  "        Its hooks: block is not a shape this can extend safely. By hand:"
    say  "            hooks:"
    say  "              pre_tool_call:"
    say  "                - matcher: read_file"
    say  "                  command: $GUARD_CMD"
  fi
fi

# The apply block sets `failed` on its own failure paths and NOTHING looked at it again:
# scripts/cutover.sh printed FAIL, then the success banner, then exited 0. A script whose
# job is to report state must not report success it did not establish.
if [ "$failed" -ne 0 ]; then
  say ""
  say "one or more steps FAILED — see above. The cutover is INCOMPLETE."
  say "  $CONFIG.bak-$STAMP restores the previous configuration, and the"
  say "  ${current:-previous} provider is still installed."
  exit 1
fi

say ""
say "=== now ==="
say "  1. Close every hermes session and start a NEW one from a fresh terminal, with the"
say "     credentials exported (set -a; . ~/.secrets; set +a)."
say "  2. Confirm the provider is the new one:"
say "       hermes memory status      # 'memories', available"
say "  3. The first file read of that new session asks you to approve the big-file guard"
say "     at the TTY. Approving records it in $ALLOWLIST;"
say "     until then hermes skips the hook, and a hermes with no TTY skips it silently."
say "  4. The old points remain reachable read-only, outside automatic recall:"
say "       qctx memory search-collections \"<topic>\" --collections hermes_memory"
say "  5. To go back: restore $CONFIG.bak-$STAMP and remove $LINK."

exit "$failed"
