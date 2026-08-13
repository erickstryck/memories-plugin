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
MCP="$HOME/.mcp.json"
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

has_mcp="$(jq -r 'if .mcpServers["qdrant-memory"] then "yes" else "no" end' "$MCP" 2>/dev/null || echo no)"
if [ "$has_mcp" = "yes" ]; then
  note "qdrant-memory MCP server to remove from .mcp.json"
else
  ok "no qdrant-memory MCP server in .mcp.json"
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
say "  .mcp.json:"
say "    - removes the qdrant-memory server (memory becomes the CLI)"
say ""
say "  UNCHANGED: the Qdrant collection, the memories, ~/.secrets, the URLs in .bashrc."
say "  The old files in ~/.claude/hooks and ~/.claude/skills stay where they are,"
say "  merely unregistered — delete them after a few sessions if you like."

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

# An INDEPENDENT step: if the one above fails, this one still has to be able to run (or
# not run) on its own, rather than being aborted by `set -e` in the middle of the cutover.
if [ "$has_mcp" = "yes" ]; then
  tmp2="$(mktemp)"
  if jq 'del(.mcpServers["qdrant-memory"])' "$MCP" > "$tmp2" && jq -e . "$tmp2" >/dev/null 2>&1; then
    mv "$tmp2" "$MCP"; ok ".mcp.json updated"
  else
    rm -f "$tmp2"; fail ".mcp.json could not be updated — remove the server by hand"
  fi
fi

say ""
say "=== now ==="
say "  1. Open a NEW terminal (or 'exec bash -l') and start claude from there."
say "  2. Check in the log that there is ONE round per prompt, not two:"
say "       tail -f \"\${QCTX_STATE_DIR:-\$HOME/.memories-plugin/state}/recall.log\""
say "  3. If anything goes wrong, the .bak-$STAMP backups restore the previous state."
