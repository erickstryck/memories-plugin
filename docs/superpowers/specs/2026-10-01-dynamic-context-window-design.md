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
- **hermes**. O provider de memória roda dentro do processo do hermes, e
  `agent/model_metadata.py:get_model_context_length(model, base_url, api_key,
  config_context_length, provider, custom_providers)` é a função que o hermes usa para o
  próprio contexto. O guard do hermes, porém, é um shell hook (`hosts/hermes/bigfile.py`,
  um subprocesso a cada leitura), que não importa o hermes.

### Medido em 2026-10-02, rodando os dois hosts

Evidência em `~/.hermes/cache/scratch/window-measurements.md` e `wm-evidence/`.

- **claude-code 2.1.282, interativo sob PTY.** A statusLine roda quando o REPL monta, depois
  dos hooks `SessionStart` e antes de qualquer prompt (+0,37 s do launch); depois, ~323 ms
  após cada mensagem do assistente; e ~80 a 100 ms depois de um `/model`, já com o modelo e
  o tamanho novos (`claude-sonnet-5` 1.000.000, `claude-haiku-4-5-20251001` 200.000). No modo
  headless (`claude -p`) ela NUNCA roda. O payload real traz
  `model: {id: "claude-opus-5-5[1m]", display_name}` e
  `context_window: {context_window_size: 1000000, used_percentage, ...}`. `SessionStart` traz
  o modelo, mas não o tamanho; o evento novo `PreModelSwitch` traz `to_model`, mas não o
  tamanho. Um plugin não declara statusLine: ela é uma configuração do usuário.
- **hermes.** O provider não recebe o modelo nem evento de troca de modelo. O hook
  `pre_llm_call`, registrável pelo `register(ctx)` do provider, roda no começo de cada turno,
  antes de qualquer tool call, com `session_id` e o modelo já trocado, e é chamado sem a
  porta `has_hook` (o `pre_api_request` tem a porta, e registrá-lo faria toda requisição
  montar uma cópia sanitizada do payload). A sessão guarda `billing_provider` e
  `billing_base_url`, que o `/model` atualiza junto com `model`.
- **`get_model_context_length`**, rodado no venv do hermes: `claude-opus-5-5` com provider
  `anthropic` dá 1.000.000 (cache em disco do models.dev, 0,2 ms a quente, sem rede). Um
  modelo desconhecido dá 256.000, o `DEFAULT_FALLBACK_CONTEXT`, que pelo valor não se
  distingue de uma janela real de 256K. Com chave Anthropic que não é OAuth, a função faz um
  GET sem cache em toda chamada; com `api_key` vazia ela pula esse passo.

## Design

### 1. Uma ordem, um dono

`core/windows.py` passa a ser o único dono da ordem, e cada host só fornece as fontes que
tem:

1. o valor que o host informou para o modelo selecionado agora, publicado por sessão: no
   hermes, pelo provider, com `get_model_context_length`; no claude-code, pela statusLine,
   com `context_window_size`. Um valor que o host só chutou (o fallback de 256.000 do
   hermes) é publicado marcado como palpite e não conta como informado, porque um palpite
   pequeno demais inverte o guard;
2. o valor que o endpoint informou, do cache que já existe (só hermes);
3. o `context_window` do config, se maior que 0;
4. 0, e o guard libera a leitura, como hoje.

A tabela `MODEL_WINDOWS` de `core/windows.py` é removida, e com ela o caminho por nome de
modelo.

Cada fonte é uma função pequena que devolve um número ou 0; a ordem é uma lista. Uma fonte
que falha (arquivo ausente, import do hermes indisponível) devolve 0 e a próxima responde.

### 2. Ponte pela statusLine (claude-code)

- Um comando novo, `qctx statusline`, recebe no stdin o JSON que o claude-code manda para a
  statusLine, publica o registro da sessão e imprime uma linha curta, por exemplo
  `ctx 23% · 1M`. Pelo lançador estável `qctx` (0,08 s para subir) e não por um caminho
  dentro do plugin, porque o diretório do plugin muda a cada versão e a statusLine fica na
  configuração do usuário.
- O registro é um arquivo por sessão, `<state>/window-<session_id>.json` com
  `{model, window, source, guess, at}`, escrito e lido por um módulo só, `core/hostwindow.py`,
  com `core.statefile` (escrita atômica, 0600). Os dois hosts publicam por ele e o guard lê
  por ele.
- Nunca falha alto: qualquer erro imprime a linha mínima e sai 0, porque uma statusLine que
  quebra suja a tela do usuário.
- O `qctx install` instala a statusLine em `~/.claude/settings.json` pela seção do
  claude-code que ele já roda (`scripts/cutover.sh`, que já edita esse arquivo de forma
  atômica). Sem `--apply`, só relata. Se já houver uma statusLine que não é a do plugin, não
  a substitui: relata e segue, e o guard daquele host cai no config.
- O guard do claude-code lê o arquivo da sessão pelo `session_id` que já recebe no input do
  hook. Os arquivos de sessões mortas entram na varredura que já existe para os outros
  estados por sessão.

### 3. Quando a identificação acontece

- **Início da sessão e troca de modelo**, nos dois hosts:
  - claude-code: a statusLine roda ao abrir a sessão e de novo quando o modelo muda, e cada
    execução regrava `{model, context_window_size}`. O guard usa o último valor gravado para
    a sessão.
  - hermes: o provider registra `pre_llm_call`. A cada turno ele compara
    `(modelo, provider, base_url)` da sessão com o último que resolveu e, só quando mudou,
    chama `get_model_context_length` com os `custom_providers` do próprio hermes e
    `api_key` vazia, e publica o registro. O primeiro turno de uma sessão é o início dela;
    o turno depois de um `/model` é a troca.
- **Limites, medidos e aceitos**: `claude -p` não tem statusLine, então ali o guard cai no
  config e, sem ele, libera. No claude-code a statusLine descreve o modelo da conversa
  principal; um subagente em outro modelo usa o tamanho dela.
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
- Faixa válida: de 0 a 1, inclusive. Os extremos seguem a regra de `core/bigfile._blocks`:
  um floor de 0 só recusa a leitura que estoura a janela; um share de 1 só recusa a que não
  cabe no que está livre; um share de 0 recusa toda leitura que custa alguma coisa. Fora da
  faixa, o `config set` recusa e o loader volta ao padrão com aviso. (Corrigido em
  2026-10-02: a versão anterior dizia que zero desligava o critério, e isso é falso para os
  dois.)
- Os dois adaptadores deixam de ler essas variáveis no import e passam a usar o config, lido
  quando o guard roda (claude-code) ou com a configuração do provider (hermes).
- `docs/usage.md` documenta os dois campos na tabela config ↔ env e na seção do guard.

### 5. Visibilidade

- `qctx setup` diz se a statusLine do claude-code está instalada, mostra o último registro
  que cada host publicou e avisa quando o guard fica desligado por falta de informação, o
  que hoje nada avisa.
- A própria statusLine mostra o tamanho que o guard vai usar.
- Sem linha de log por decisão: o guard não escreve log hoje, e uma linha por leitura seria
  I/O no caminho que roda antes de toda leitura.

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
