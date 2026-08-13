#!/usr/bin/env bash
# Troca do setup feito à mão para o plugin, em UMA passada atômica.
#
# POR QUE ATÔMICA: enquanto os hooks manuais do settings.json e os hooks do plugin
# coexistirem, os DOIS disparam e o recall é injetado em dobro no mesmo prompt —
# duplicando o custo de contexto sem acrescentar informação. Então habilitar o
# plugin e remover as entradas manuais é um único movimento, não dois.
#
# Sem argumento, faz ENSAIO: mostra exatamente o que mudaria e não escreve nada.
# Com --apply, aplica, depois de fazer backup datado dos dois arquivos.
#
#     ./scripts/cutover.sh              # ensaio
#     ./scripts/cutover.sh --apply      # aplica
#
# Feche as outras sessões antes de aplicar: o harness relê o settings.json em
# tempo real, e uma sessão viva pode perder os hooks no meio de um turno.
set -euo pipefail

RAIZ="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETTINGS="$HOME/.claude/settings.json"
MCP="$HOME/.mcp.json"
APLICAR="${1:-}"
STAMP="$(date +%Y%m%d-%H%M%S)"

falhou=0
diga() { printf '%s\n' "$*"; }
ok()   { diga "  ok    $*"; }
erro() { diga "  FALHA $*"; falhou=1; }
nota() { diga "  ..    $*"; }

diga "=== verificações ==="

if command -v jq >/dev/null; then ok "jq disponível"; else erro "jq é necessário"; fi

if [ -f "$SETTINGS" ] && jq -e . "$SETTINGS" >/dev/null 2>&1; then
  ok "settings.json válido"
else
  erro "settings.json ausente ou inválido: $SETTINGS"
fi

if python3 "$RAIZ/cli/qctx.py" setup --check </dev/null >/dev/null 2>&1; then
  ok "qctx responde"
else
  nota "qctx setup --check reportou pendência — rode-o e resolva antes de aplicar"
fi

if python3 -m unittest discover -s "$RAIZ/tests" >/dev/null 2>&1; then
  ok "suíte offline passa"
else
  erro "a suíte offline não passa — não vire a chave assim"
fi

manuais="$(jq -r '[.hooks.UserPromptSubmit[]?.hooks[]? | select(.command | test("memory-recall|remember-cadence")) | .command] | length' "$SETTINGS" 2>/dev/null || echo 0)"
if [ "$manuais" -gt 0 ]; then
  nota "$manuais hook(s) manual(is) a remover do settings.json"
else
  ok "nenhum hook manual em settings.json"
fi

tem_mcp="$(jq -r 'if .mcpServers["qdrant-memory"] then "sim" else "nao" end' "$MCP" 2>/dev/null || echo nao)"
if [ "$tem_mcp" = "sim" ]; then
  nota "servidor MCP qdrant-memory a remover de .mcp.json"
else
  ok "nenhum servidor MCP qdrant-memory em .mcp.json"
fi

if [ "$falhou" -ne 0 ]; then
  diga ""
  diga "verificações falharam — nada foi alterado."
  exit 1
fi

diga ""
diga "=== o que muda ==="
diga "  settings.json:"
diga "    - remove os hooks manuais de UserPromptSubmit (recall e checkpoint)"
diga "    + registra o marketplace local $RAIZ"
diga "    + habilita o plugin memories-plugin (que traz os mesmos dois hooks)"
diga "  .mcp.json:"
diga "    - remove o servidor qdrant-memory (a memória passa a ser o CLI)"
diga ""
diga "  NÃO muda: a coleção no Qdrant, as memórias, ~/.secrets, as URLs no .bashrc."
diga "  Os arquivos antigos em ~/.claude/hooks e ~/.claude/skills ficam no lugar,"
diga "  apenas desregistrados — apague depois de algumas sessões, se quiser."

if [ "$APLICAR" != "--apply" ]; then
  diga ""
  diga "ENSAIO. Para aplicar de verdade: $0 --apply"
  exit 0
fi

diga ""
diga "=== aplicando ==="
cp "$SETTINGS" "$SETTINGS.bak-$STAMP" && ok "backup $SETTINGS.bak-$STAMP"
[ -f "$MCP" ] && cp "$MCP" "$MCP.bak-$STAMP" && ok "backup $MCP.bak-$STAMP"

tmp="$(mktemp)"
jq --arg raiz "$RAIZ" '
  # remove os hooks manuais, e depois os grupos que ficaram sem nenhum hook
  (.hooks.UserPromptSubmit) |= (
    map(.hooks |= map(select((.command // "") | test("memory-recall|remember-cadence") | not)))
    | map(select((.hooks | length) > 0))
  )
  | (if (.hooks.UserPromptSubmit | length) == 0 then del(.hooks.UserPromptSubmit) else . end)
  # não deixa a chave `hooks` vazia para trás: estrofe vazia confunde quem lê o
  # arquivo depois procurando o que está registrado
  | (if (.hooks | length) == 0 then del(.hooks) else . end)
  | .extraKnownMarketplaces["memories-plugin"] = {source: {source: "directory", path: $raiz}}
  | .enabledPlugins["memories-plugin@memories-plugin"] = true
' "$SETTINGS" > "$tmp"
jq -e . "$tmp" >/dev/null && mv "$tmp" "$SETTINGS" && ok "settings.json atualizado"

if [ "$tem_mcp" = "sim" ]; then
  tmp2="$(mktemp)"
  jq 'del(.mcpServers["qdrant-memory"])' "$MCP" > "$tmp2"
  jq -e . "$tmp2" >/dev/null && mv "$tmp2" "$MCP" && ok ".mcp.json atualizado"
fi

diga ""
diga "=== agora ==="
diga "  1. Abra um terminal NOVO (ou 'exec bash -l') e inicie o claude de lá."
diga "  2. Confira no log que há UM round por prompt, não dois:"
diga "       tail -f \"\${QCTX_STATE_DIR:-\$HOME/.memories-plugin/state}/recall.log\""
diga "  3. Se algo der errado, os backups .bak-$STAMP restauram o estado anterior."
