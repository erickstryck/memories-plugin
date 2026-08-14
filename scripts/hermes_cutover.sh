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

read_memory_key() {
  python3 - "$CONFIG" "$1" <<'PY' 2>/dev/null || true
import re, sys
try:
    src = open(sys.argv[1], encoding="utf-8").read()
except OSError:
    sys.exit(0)
want, inside, indent = sys.argv[2], False, None
#: The LAST direct child wins, not the first. That is what a YAML parser resolves a duplicate
#: key to (PyYAML keeps the last and does not complain), and hermes reads this file with
#: yaml.safe_load. Reading the first would let a file holding BOTH
#: `provider: memories` and `provider: qdrant` answer "memories" while hermes loads qdrant —
#: which is exactly the corruption the post-write verification exists to catch.
found = None
for line in src.splitlines():
    if re.match(r"^memory:\s*(#.*)?$", line):
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
say "  3. The old points remain reachable read-only, outside automatic recall:"
say "       qctx memory search-collections \"<topic>\" --collections hermes_memory"
say "  4. To go back: restore $CONFIG.bak-$STAMP and remove $LINK."

exit "$failed"
