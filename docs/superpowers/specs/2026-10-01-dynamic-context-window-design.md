# Contexto do modelo descoberto dinamicamente, e limites do guard configuráveis

Data: 2026-10-01. Versão alvo: 1.3.0. Base: v1.2.0 (`4f018fc`).

## Problema

O guard de arquivos grandes (`core/bigfile.py`, chamado por `hooks/bigfile.py` no
claude-code e por `hosts/hermes/bigfile.py` no hermes) decide com dois números: o tamanho
do contexto do modelo e quanto dele já foi usado. O usado já é lido dinamicamente (o
transcript no claude-code, o banco de sessões no hermes). O tamanho do contexto não:
`core/windows.py:window_for` consulta primeiro o `context_window` do config, que vence
tudo, e depois uma tabela por nome exato de modelo.

Medido em 2026-10-01 na cópia instalada da v1.2.0:

| modelo, como o host grava | `window_for` com `context_window=0` |
|---|---|
| `claude-opus-5-5` (transcript do claude-code, e o modelo do hermes) | 0, o guard libera tudo |
| `claude-opus-5-5[1m]` | 0 |
| `claude-opus-5` | 1.000.000 |

Ou seja: com o config em 0 o guard está desligado nos dois hosts, e com o config fixo ele
ignora a troca de modelo. O `~/.claude/settings.json` do usuário tem `"model": "opus[1m]"`.

Os dois limites do guard (bloqueia com 20% ou menos do contexto livre, ou quando um arquivo
ocuparia mais de 40% do que resta) só se ajustam por variável de ambiente
(`QCTX_BIGFILE_FLOOR_PCT`, `QCTX_BIGFILE_SHARE_PCT`), lidas no import de cada adaptador.

## O que o usuário decidiu

1. O plugin descobre sozinho, nos dois hosts, o tamanho do contexto do modelo selecionado,
   de forma dinâmica. O `context_window` do config passa a ser o último recurso.
2. Nenhuma tabela fixa de modelos e nenhuma dedução pelo nome. O modelo selecionado e o
   tamanho do seu contexto são identificados quando a sessão começa e de novo quando o
   modelo é trocado. Isso reverte a parte "B" escolhida antes (dedução pelo modelo e tabela
   por família), por decisão do usuário: "não faz sentido ter gravado o modelo e o nome de
   modo fixo".
3. No claude-code, a ponte pela statusLine (valor exato, informado pelo próprio claude-code).
   No hermes, a função do próprio host.
4. Os dois limites do guard passam a ser ajustáveis por `qctx config set`, com 0,20 e 0,40
   como padrão.

## O que o host oferece (medido)

- **claude-code 2.1.282** (lido no binário com `grep -a -o`). O input de hook (função
  `ic`) traz `session_id`, `transcript_path`, `cwd`, `scratchpad_dir`, `prompt_id`,
  `permission_mode`, `agent_id`, `agent_type` e `effort`. Não traz o modelo nem o tamanho do
  contexto. O claude-code calcula `context_window: {context_window_size, used_percentage,
  remaining_percentage, total_input_tokens, current_usage}`, mas só entrega esse objeto ao
  comando de statusLine. O transcript grava o modelo sem o sufixo (`claude-opus-5-5`).
- **hermes**. O plugin roda dentro do processo do hermes, e
  `agent/model_metadata.py:get_model_context_length(model, base_url, api_key,
  config_context_length, provider, custom_providers)` é a função que o hermes usa para o
  próprio contexto.

## Design

### 1. Uma ordem, um dono

`core/windows.py` passa a ser o único dono da ordem, e cada host só fornece as fontes que
tem:

1. o valor que o host informou para o modelo selecionado agora: no hermes,
   `get_model_context_length`; no claude-code, o `context_window_size` gravado pela
   statusLine para esta sessão;
2. o valor que o endpoint informou, do cache que já existe (só hermes);
3. o `context_window` do config, se maior que 0;
4. 0, e o guard libera a leitura, como hoje.

A tabela `MODEL_WINDOWS` de `core/windows.py` é removida, e com ela o caminho por nome de
modelo.

Cada fonte é uma função pequena que devolve um número ou 0; a ordem é uma lista. Uma fonte
que falha (arquivo ausente, import do hermes indisponível) devolve 0 e a próxima responde.

### 2. Ponte pela statusLine (claude-code)

- Um comando novo do plugin, `hooks/statusline.py`, recebe no stdin o JSON que o claude-code
  manda para a statusLine. Grava `{context_window_size, model, at}` em
  `<state>/context-<session_id>.json`, por `core.statefile` (escrita atômica, 0600), e
  imprime uma linha curta, por exemplo `ctx 23% · 1M`.
- Nunca falha alto: qualquer erro imprime a linha mínima e sai 0, porque uma statusLine que
  quebra suja a tela do usuário.
- O `qctx install` oferece instalar a statusLine em `~/.claude/settings.json`. Se já houver
  uma, não a substitui sem perguntar. O `qctx install --check` diz se ela está instalada.
- O guard do claude-code lê o arquivo da sessão pelo `session_id` que já recebe no input do
  hook. Os arquivos de sessões mortas entram na varredura que já existe para os outros
  estados por sessão.

### 3. Quando a identificação acontece

- **Início da sessão e troca de modelo**, nos dois hosts:
  - claude-code: a statusLine roda ao abrir a sessão e de novo quando o modelo muda, e cada
    execução regrava `{model, context_window_size}`. O guard usa o último valor gravado para
    a sessão.
  - hermes: o tamanho é resolvido quando a sessão começa (`initialize`) e de novo quando o
    modelo da sessão muda. O guard compara o modelo atual com o último resolvido e só chama
    `get_model_context_length` quando ele mudou.
- A medir antes de implementar: em que momento a primeira statusLine roda (antes do primeiro
  tool call?) e se ela roda logo após um `/model`; no hermes, como o provider fica sabendo
  da troca de modelo.
- **Consequência que precisa ficar visível**: sem a statusLine instalada, o claude-code não
  informa o tamanho do contexto a nenhum processo externo. Nesse caso o guard cai no config
  e, sem ele, libera. Na v1.2.0 a tabela cobria `claude-opus-5` nessa situação; por isso o
  `qctx install` oferece a statusLine e o `qctx setup` avisa quando o guard está desligado.

### 4. Limites do guard no config

- Dois campos novos do `Config`: `bigfile_floor_pct` (padrão 0.20) e `bigfile_share_pct`
  (padrão 0.40), com os nomes de ambiente atuais como aliases (`QCTX_BIGFILE_FLOOR_PCT`,
  `BIGFILE_FLOOR_PCT` e os equivalentes de share). A precedência é a de todo campo:
  ambiente, arquivo, padrão.
- `core.config` passa a tratar campos `float` como trata os `int`: valor que não é número
  volta para o padrão e avisa pelo canal `note`; o `config set` recusa.
- Faixa válida: de 0 a 1. Zero desliga aquele critério (a semântica que `core/knobs.py`
  já documenta para eles). Fora da faixa, o `config set` recusa e o loader volta ao padrão
  com aviso.
- Os dois adaptadores deixam de ler essas variáveis no import e passam a usar o config, lido
  quando o guard roda (claude-code) ou com a configuração do provider (hermes).
- `docs/usage.md` documenta os dois campos na tabela config ↔ env e na seção do guard.

### 5. Visibilidade

- `qctx setup` mostra, por host, de onde veio o tamanho do contexto (host, endpoint,
  dedução, config ou nenhum) e avisa quando o guard está desligado por falta de informação,
  o que hoje nada avisa.
- A linha de decisão do guard registra a fonte do tamanho do contexto.

## Restrições

- KISS e S.O.L.I.D. sem perder completude: cada fonte com uma responsabilidade, a ordem num
  só lugar, nenhum host decide a ordem.
- Nada do que funciona na v1.2.0 pode degradar. Em particular: o guard continua liberando
  quando não sabe; um config ou um arquivo de estado ilegível custa a fonte, nunca a leitura.
- Nenhum travessão longo nas linhas adicionadas. Código, comentários e commits em inglês.
- Testes de comportamento, com fixtures tiradas do JSON real que o binário 2.1.282 monta
  para a statusLine.

## Testes e entrega

- TDD por fonte e pela ordem, nos dois hosts; um teste ponta a ponta com a statusLine real
  gravando e o guard real lendo.
- Regressão contra a v1.2.0: suíte completa, integração, probe no mesmo instante e a suíte
  com um config de desenvolvedor corrompido, comparando as falhas por nome.
- Code review com superpowers, com um revisor focado em compatibilidade.
- Entrega como v1.3.0: commit, tag, push e deploy (claude-code, hermes, daemon).

## Fora do escopo

- Calcular o contexto usado por conta própria: continua vindo do transcript e do banco de
  sessões.
- Mudar a política do guard (o que conta como arquivo grande): só os limites ficam
  configuráveis.
