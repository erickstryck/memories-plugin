#!/usr/bin/env bash
# Cutover from the hand-made setup to the plugin, in ONE atomic pass.
#
# WHY ATOMIC: while the manual hooks in settings.json and the plugin's hooks coexist,
# BOTH fire and recall is injected twice into the same prompt — doubling the context
# cost without adding information. So enabling the plugin and removing the manual
# entries is a single move, not two.
#
# With no argument it does a DRY RUN: it shows exactly what would change and writes
# nothing. With --apply it applies, after taking a dated backup of both files.
#
#     ./scripts/cutover.sh              # dry run
#     ./scripts/cutover.sh --apply      # apply
#
# Close the other sessions before applying: the harness re-reads settings.json live,
# and a running session can lose its hooks in the middle of a turn.
set -euo pipefail

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETTINGS="$HOME/.claude/settings.json"
# DUAS localizações, e olhar só uma foi um defeito real: a config de MCP que o harness
# de fato carrega no escopo `user` vive em ~/.claude.json, não em ~/.mcp.json. Este
# script limpava o segundo, reportava "ok" para o servidor antigo e ele voltava a subir
# em toda sessão nova — o falso "ok" é pior que a checagem faltando, porque ele afirma
# que o trabalho foi feito.
MCP="$HOME/.mcp.json"
CLAUDE_JSON="$HOME/.claude.json"
APPLY="${1:-}"
STAMP="$(date +%Y%m%d-%H%M%S)"

failed=0
say()  { printf '%s\n' "$*"; }
ok()   { say "  ok    $*"; }
fail() { say "  FAIL  $*"; failed=1; }
note() { say "  ..    $*"; }

say "=== checks ==="

if command -v jq >/dev/null; then ok "jq available"; else fail "jq is required"; fi

if [ -f "$SETTINGS" ] && jq -e . "$SETTINGS" >/dev/null 2>&1; then
  ok "settings.json valid"
else
  fail "settings.json missing or invalid: $SETTINGS"
fi

if python3 "$ROOT/cli/qctx.py" setup --check </dev/null >/dev/null 2>&1; then
  ok "qctx answers"
else
  note "qctx setup --check reported something pending — run it and fix that before applying"
fi

if python3 -m unittest discover -s "$ROOT/tests" >/dev/null 2>&1; then
  ok "offline suite passes"
else
  fail "the offline suite does not pass — do not flip the switch like this"
fi

manual="$(jq -r '[.hooks.UserPromptSubmit[]?.hooks[]? | select(.command | test("memory-recall|remember-cadence")) | .command] | length' "$SETTINGS" 2>/dev/null || echo 0)"
if [ "$manual" -gt 0 ]; then
  note "$manual manual hook(s) to remove from settings.json"
else
  ok "no manual hooks in settings.json"
fi

mcp_in() { jq -e '.mcpServers["qdrant-memory"]' "$1" >/dev/null 2>&1; }
has_mcp=no; has_mcp_user=no
mcp_in "$MCP" && has_mcp=yes
mcp_in "$CLAUDE_JSON" && has_mcp_user=yes
if [ "$has_mcp" = yes ] || [ "$has_mcp_user" = yes ]; then
  [ "$has_mcp" = yes ]      && note "qdrant-memory to remove from .mcp.json"
  [ "$has_mcp_user" = yes ] && note "qdrant-memory to remove from .claude.json (user scope)"
else
  ok "no qdrant-memory MCP server in .mcp.json or .claude.json"
fi

# The old hand-made skills are auto-discovered from ~/.claude/skills — being absent from
# settings.json does NOT unregister them. Left in place, `remember` keeps telling the
# model to call mcp__qdrant-memory tools that no longer exist, and competes with the
# plugin's own memory skill.
retired=""
for s in remember doc-index; do
  [ -e "$HOME/.claude/skills/$s" ] && retired="$retired $s"
done
if [ -n "$retired" ]; then
  note "superseded skill(s) to retire from ~/.claude/skills:$retired"
else
  ok "no superseded skills in ~/.claude/skills"
fi

if [ "$failed" -ne 0 ]; then
  say ""
  say "checks failed — nothing was changed."
  exit 1
fi

say ""
say "=== what changes ==="
say "  settings.json:"
say "    - removes the manual UserPromptSubmit hooks (recall and checkpoint)"
say "    + registers the local marketplace $ROOT"
say "    + enables the memories-plugin plugin (which brings the same two hooks)"
say "  MCP server (memory becomes the CLI):"
say "    - removes qdrant-memory from .mcp.json and from .claude.json (user scope)"
say "  ~/.claude/skills:"
say "    - retires the hand-made remember/ and doc-index/, which the plugin supersedes"
say ""
say "  UNCHANGED: the Qdrant collection, the memories, ~/.secrets, the URLs in .bashrc."
say "  The old hook files in ~/.claude/hooks stay where they are, merely unregistered"
say "  (settings.json is what registers a hook). The skills are MOVED, not deleted,"
say "  because ~/.claude/skills is auto-discovered — leaving them there keeps them live."

if [ "$APPLY" != "--apply" ]; then
  say ""
  say "DRY RUN. To apply for real: $0 --apply"
  exit 0
fi

say ""
say "=== applying ==="
cp "$SETTINGS" "$SETTINGS.bak-$STAMP" && ok "backup $SETTINGS.bak-$STAMP"
[ -f "$MCP" ] && cp "$MCP" "$MCP.bak-$STAMP" && ok "backup $MCP.bak-$STAMP"

# IDEMPOTENT on purpose. The `// []` is not free defensiveness: without it the filter
# breaks with "Cannot iterate over null" on any settings.json without
# `.hooks.UserPromptSubmit` — which includes RUNNING THIS SCRIPT TWICE, because the
# first run deletes the key. Under `set -e` that aborted after the backups and before
# .mcp.json, leaving the cutover half done.
tmp="$(mktemp)"
jq --arg root "$ROOT" '
  # remove the manual hooks, then the groups left with no hooks at all
  .hooks = ((.hooks // {}) | .UserPromptSubmit = (
      ((.UserPromptSubmit // [])
       | map(.hooks |= map(select((.command // "") | test("memory-recall|remember-cadence") | not)))
       | map(select(((.hooks // []) | length) > 0)))
    ))
  | (if ((.hooks.UserPromptSubmit // []) | length) == 0
     then del(.hooks.UserPromptSubmit) else . end)
  # do not leave an empty `hooks` key behind: an empty stanza confuses whoever reads
  # the file later looking for what is registered
  | (if ((.hooks // {}) | length) == 0 then del(.hooks) else . end)
  | .extraKnownMarketplaces["memories-plugin"] = {source: {source: "directory", path: $root}}
  | .enabledPlugins["memories-plugin@memories-plugin"] = true
' "$SETTINGS" > "$tmp"
if jq -e . "$tmp" >/dev/null 2>&1; then
  mv "$tmp" "$SETTINGS"; ok "settings.json updated"
else
  rm -f "$tmp"; fail "the settings.json transform did not produce valid JSON — nothing was swapped"
fi

# INDEPENDENT steps: if one above fails, these still have to be able to run (or not run)
# on their own, rather than being aborted by `set -e` in the middle of the cutover.
if [ "$has_mcp" = yes ]; then
  tmp2="$(mktemp)"
  if jq 'del(.mcpServers["qdrant-memory"])' "$MCP" > "$tmp2" && jq -e . "$tmp2" >/dev/null 2>&1; then
    mv "$tmp2" "$MCP"; ok ".mcp.json updated"
  else
    rm -f "$tmp2"; fail ".mcp.json could not be updated — remove the server by hand"
  fi
fi

# `claude mcp remove` rather than editing .claude.json here: that file also holds project
# history and a running session rewrites it, so a hand-rolled read-modify-write can be
# clobbered mid-flight. The CLI is the supported path; the manual fallback is printed on
# failure rather than attempted.
if [ "$has_mcp_user" = yes ]; then
  cp "$CLAUDE_JSON" "$CLAUDE_JSON.bak-$STAMP" && ok "backup $CLAUDE_JSON.bak-$STAMP"
  if command -v claude >/dev/null && claude mcp remove qdrant-memory -s user >/dev/null 2>&1; then
    ok "qdrant-memory removed from user scope"
  else
    fail "could not remove it — run: claude mcp remove qdrant-memory -s user"
  fi
fi

# Moved, never deleted: they are the user's files, and a wrong call here is trivially
# reversible only if the originals still exist somewhere.
if [ -n "$retired" ]; then
  dest="$HOME/.claude/skills-retired-$STAMP"
  mkdir -p "$dest"
  for s in $retired; do
    mv "$HOME/.claude/skills/$s" "$dest/" && ok "retired $s -> $dest/"
  done
fi

say ""
say "=== now ==="
say "  1. Open a NEW terminal (or 'exec bash -l') and start claude from there."
say "  2. Confirm the old MCP server does NOT come back — it is read at session start,"
say "     so a running session keeps its process either way:"
say "       ps -eo args | grep qdrant_memory | grep -v grep"
say "  3. Check in the log that there is ONE round per prompt, not two:"
say "       tail -f \"\${QCTX_STATE_DIR:-\$HOME/.memories-plugin/state}/recall.log\""
say "  4. If anything goes wrong, the .bak-$STAMP backups restore the previous state."
