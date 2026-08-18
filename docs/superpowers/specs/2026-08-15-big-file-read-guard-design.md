# Guarda de leitura de arquivo grande — design

**Data:** 2026-08-15
**Estado:** aprovado em brainstorming, pronto para plano de implementação

## O problema, medido

Em 2026-08-15, no primeiro teste real do embedding de documentos depois do cutover, o
usuário pediu ao hermes: *"analise o arquivo places.json e me diga se ele trata de entidades
de espaço"*. O arquivo tem **586 KB / 15.593 linhas / ~171k tokens estimados**. O hermes
chamou `read_file` direto, em 0,7s, e respondeu corretamente (3 buildings, 6 floors, 369
spaces — conferido). Só indexou quando o usuário pediu explicitamente: 258 chunks em 36,2s.

Custo comparado: **~171k tokens lendo** contra **~6k numa busca típica de 5 chunks**.

Nada no plugin impediu isso, e nada avisou. A skill `doc-index` (claude-code) e a descrição
da ferramenta `docs_index` (hermes) orientam, mas são conselho — o modelo decide, e aqui
decidiu ler. Três causas, e só a terceira é defeito:

1. A pergunta não bate com o gatilho: a regra exige arquivo grande **e** pergunta sobre uma
   parte. *"Se ele trata de X"* é pergunta sobre o todo, e ler é defensável.
2. A descrição lista "log, dump, transcript or report" — um snapshot JSON não está ali.
3. **O modelo não sabe o tamanho antes de ler.** Chama a ferramenta às cegas. As duas
   primeiras causas são consequência desta.

## O que se decidiu

Uma guarda que **bloqueia** a leitura quando ela custaria caro demais, com mensagem que diz
o que fazer em vez disso. Decisões do usuário, tomadas no brainstorming:

| Decisão | Escolha | Alternativas recusadas |
|---|---|---|
| Comportamento | **Bloqueia** e manda indexar | só avisar; bloquear só acima de limite duro |
| Escape | **Palavra no prompt** do usuário | 2ª tentativa passa; variável de ambiente |
| Critério | **Os dois**, o que disparar primeiro | só sobra final; só fatia do livre |
| Janela | **Tabela por modelo + config sobrescreve** | só config; só tabela |
| Escopo | **Só a leitura de arquivo** | + terminal com cat/head; + busca |

O critério é relativo ao contexto, não ao tamanho do arquivo — ideia do usuário, nas
palavras dele: *"se eu ler o arquivo x e a janela de contexto explodir então devo indexá-lo,
mas teria que ser possível ver quanto de contexto atual resta"*. Um arquivo de 171k é
irrelevante numa janela de 1M recém-aberta e fatal numa sessão com 100k livres.

## Viabilidade — tudo medido antes do desenho

O mecanismo existe nos **dois** hosts. Isto corrige uma afirmação errada feita antes na
mesma conversa (*"o hermes não expõe esse gancho"*), que vinha de eu ter lido a ABC
`MemoryProvider` — onde de fato não há gancho de ferramenta — e generalizado de um
subsistema para o produto.

**hermes** (`agent/shell_hooks.py`, `hermes_cli/hooks.py`, v0.20.1):
- eventos `pre_tool_call` e `post_tool_call`, com campo `matcher` para mirar uma ferramenta
- **exit code 2 bloqueia** (`_BLOCKING_EVENTS`); o fonte diz "Claude-Code / Cursor compatible"
- aceita `{"decision":"block","reason":…}` e `{"action":"block","message":…}`
- payload: `function_name`, `function_args`, `task_id`, `session_id`, `tool_call_id`,
  `turn_id`, `api_request_id`, `middleware_trace` — **sem dado de contexto**
- definições no bloco `hooks:` do `$HERMES_HOME/config.yaml`; consentimento em
  `~/.hermes/shell-hooks-allowlist.json`

**claude-code**: hooks `PreToolUse` com bloqueio, já usados por este plugin em
`UserPromptSubmit`.

### De onde sai o contexto usado

| | claude-code | hermes |
|---|---|---|
| Contexto atual | **exato** | **estimado** |
| Fonte | última linha `assistant` do transcript JSONL | `state.db` → `messages` com `active=1` |
| Cálculo | `input_tokens + cache_creation_input_tokens + cache_read_input_tokens` | soma de `(len(content)+3)//4` |
| Medido | 2 + 5.906 + 598.115 = **604.023** tokens | `token_count` é **NULL** em toda linha; `session_model_usage` é cumulativo por sessão, não contexto atual |

A razão `(len+3)//4` não é invenção: é literalmente `_chars_to_tokens` de
`agent/context_breakdown.py`, o que o próprio hermes usa quando não tem medição.

**Consequência aceita:** no hermes a estimativa **subestima**, porque não enxerga system
prompt, definições de ferramenta nem índice de skills — o `context_breakdown` soma essas
categorias porque roda em processo. A guarda dispara mais tarde no hermes. A mensagem
deve marcar o número com `≈`.

### De onde sai o prompt do usuário (para a palavra de escape)

| | claude-code | hermes |
|---|---|---|
| Fonte | transcript JSONL | `state.db` → `messages` |
| Filtro | `role=user` **e** `toolUseResult is None` **e** `userType=external` **e** `entrypoint=cli` | `role='user'` e `active=1`, mais recente |
| Medido | devolveu os prompts reais, separados de skills e injeções de hook | devolveu limpo, sem ruído |

### A janela máxima NÃO é legível de disco em nenhum host

- claude-code: o transcript traz o modelo (`claude-opus-5`), não a janela. **Armadilha real e
  imediata:** esta sessão é a variante de 1M; o mesmo nome de modelo em sessão normal tem
  200k. Tabela por nome erraria por 5×, em silêncio.
- hermes: `sessions.model` = `MiniMax-M2.7`; `model_config` tem `max_tokens` (saída, não
  janela); `provider_models_cache.json` não menciona janela; `context_length` vive só no
  compressor, em processo.

Daí a decisão: **tabela como TETO, config do usuário vence**.

> **EMENDA de 2026-08-16, depois da T4.** A decisão original dizia "tabela como palpite", e
> isso estava errado de um jeito que só apareceu quando o adaptador rodou de verdade: um
> palpite *pequeno demais* não enfraquece a guarda, ele a **inverte**. Medido no adaptador da
> T4, com esta máquina e sem override: `window_for('claude-opus-5')` = 200k numa sessão de 1M,
> `used` = 989.479, logo `free` = 0, e o hook real **negou um arquivo de 4 KB**. Fail-open
> virou fail-closed.
>
> A assimetria é o argumento inteiro: palpite grande demais só faz a guarda **dormir**;
> pequeno demais produz a única falha que esta feature não pode produzir. Então a tabela
> deixa de ser a janela *nominal* do modelo e passa a ser o **teto** — a maior janela que
> qualquer variante daquele nome pode ter (`claude-opus-5` → 1M) — e `used >= window` passa a
> significar "o palpite foi refutado", caindo para janela desconhecida.
>
> Custo aceito: quem roda uma sessão de 200k fica com guarda quase inerte até declarar
> `context_window`. Inerte até configurar é o modo de falha que este design já escolheu;
> jaula não é.

> **EMENDA de 2026-08-17.** A afirmação "a janela máxima NÃO é legível de disco em nenhum host"
> vale para o claude-code — os quatro caminhos foram medidos — e **deixou de valer para o
> hermes**: o endpoint que serve o modelo a informa por `/models`, e o adaptador passou a lê-la.
> A tabela de tetos continua sendo o degrau seguinte, não o primeiro. Ver
> `docs/superpowers/specs/2026-08-17-window-resolution-design.md`.

## Arquitetura

```
core/bigfile.py          decide. Não sabe o que é hook nem qual host chamou.
hooks/bigfile.py         adaptador claude-code: colhe Budget, traduz Verdict
hosts/hermes/bigfile.py  adaptador hermes: idem, sobre state.db
```

Mesma separação que fez o porte para o hermes custar um adaptador em vez de uma reescrita.

### O núcleo

```python
@dataclass(frozen=True)
class Budget:
    window: int    # tamanho da janela de contexto
    used:   int    # tokens em contexto agora
    exact:  bool   # medido (claude-code) ou estimado (hermes)

@dataclass(frozen=True)
class Verdict:
    block:  bool
    reason: str    # a mensagem que o MODELO lê
    cost:   int    # tokens que o arquivo custaria
    free:   int    # tokens livres

def cost_of(path: str) -> int
def decide(path: str, budget: Budget, cfg) -> Verdict
```

Regra, os dois critérios, o que disparar primeiro:

```
free  = window - used
cost  = cost_of(path)
after = used + cost

BLOQUEIA se  after > window * (1 - FLOOR_PCT)   (sobra final < 20%)
         ou  cost  > free   * SHARE_PCT         (arquivo > 40% do livre)
```

Constantes, ambas configuráveis: `FLOOR_PCT = 0.20` (fração da janela que tem de SOBRAR),
`SHARE_PCT = 0.40` (fração do livre que UM arquivo pode tomar).

Os nomes dizem o que a fração significa, e a fórmula faz a inversão explícita. A primeira
versão desta spec escrevia `window * FLOOR_PCT_INV`, um identificador que não era definido em
lugar nenhum — o leitor teria de adivinhar se `FLOOR_PCT` valia 0,20 ou 0,80. É a família de
defeito que este projeto já pagou uma vez, quando a spec dizia três `dirname` e o plano
escreveu dois: símbolo definido, valor divergente, e o implementador lê o plano.

Aplicado aos números reais desta sessão (604.023 usados de 1.000.000, arquivo de 171k):
`after = 775k` → sobra 22%, **não** bloqueia pelo primeiro critério; `cost/free = 171/396 =
43%` → **bloqueia** pelo segundo. Numa sessão com 850k usados, o primeiro também bloquearia.

### Três casos que o `decide` trata e não estavam nas escolhas

1. **Arquivo que não se indexa.** `docs_index` fatia texto. Binário → **libera**, porque
   mandar indexar seria conselho errado e bloquear sem alternativa é só travar.
2. **Arquivo já indexado.** O núcleo consulta o índice. Se já estiver lá, a mensagem muda de
   *"indexe"* para *"já indexado como `<doc_id>`, busque nele"* — a ação certa, e evita
   reindexar centenas de chunks à toa.
3. **Falha da própria guarda → libera.** Ver Modos de falha.

### Os adaptadores

Cada um responde duas perguntas e traduz o veredicto:

**claude-code** — lê a **cauda** do transcript (não o arquivo inteiro: o desta sessão tem
15,8 MB), extrai o último `usage` e o último turno real do usuário, resolve a janela pelo
modelo, chama `decide`, e em caso de bloqueio escreve JSON de bloqueio em **stdout do
protocolo de hook** conforme o contrato do `PreToolUse` — nunca texto solto, que corromperia
o protocolo.

**hermes** — script apontado pelo bloco `hooks:` com `matcher` na ferramenta de leitura.
Abre o `state.db` **somente-leitura, com timeout curto**, soma as mensagens ativas, lê o
último turno do usuário, chama `decide`, e devolve `{"decision":"block","reason":…}` com
**exit 2**.

### Divergências declaradas entre os dois adaptadores

`core/bigfile.py` decide igual para os dois. O que difere está nos adaptadores, e **cada
diferença abaixo é FORÇADA pelo host** — nenhuma é preferência de quem escreveu. Esta seção
mora no spec, e não num relatório de task, porque `.superpowers/sdd/` é gitignored: a tabela
de onde se compara os dois hosts não pode desaparecer no próximo clone.

A tabela não é decorativa: `tests/test_host_equivalence.py::TestEveryDivergenceOfTheGuardsIsDeclaredInTheSPEC`
DERIVA das duas fontes as constantes de módulo, as variáveis de ambiente lidas e as chaves de
`tool_input`, e exige que tudo que exista em um só lado esteja nomeado aqui. Acrescentar um teto
a um adaptador sem escrever a linha correspondente deixa a suíte vermelha.

| O quê | `hooks/bigfile.py` (claude-code) | `hosts/hermes/bigfile.py` | O que força |
|---|---|---|---|
| Fonte do orçamento | cauda do transcript JSONL, bloco `usage` MEDIDO | `state.db`, soma dos corpos de `messages` com `(len+3)//4` | medido: `messages.token_count` é NULL em 59 de 59 linhas do banco vivo, e `session_model_usage` é cumulativo, não o contexto atual |
| `Budget.exact` | `True` | `False` — a mensagem sai com `≈` | a estimativa SUBESTIMA (não vê system prompt, ferramentas, skills, nem as colunas irmãs `reasoning`/`tool_calls`/`api_content`); número que parece preciso e é palpite é pior que palpite que se declara |
| Protocolo de bloqueio | `{"hookSpecificOutput": {…"permissionDecision": "deny"…}}` em stdout, exit 0 | `{"decision":"block","reason":…}` em stdout **e** `BLOCK_EXIT_CODE` = 2 | contratos de host diferentes, ambos lidos do binário/fonte instalado; no hermes os dois juntos porque stdout truncado ainda bloqueia pelo código e código ilegível ainda bloqueia pela razão |
| Chave do caminho no payload | `tool_input.file_path` | `tool_input.path` | `agent/shell_hooks.py::_serialize_payload` renomeia `args`→`tool_input`, e `READ_FILE_SCHEMA` de `tools/file_tools.py` exige `path`. Ler a chave errada não é erro visível: o guarda não vê caminho nenhum e LIBERA tudo, calado |
| Filtro de turno real do usuário | `userType=external` + `entrypoint=cli` + ausência de `toolUseResult` | nenhum | medido: no hermes o resultado de ferramenta é `role='tool'` (21 linhas no banco vivo) e as 11 linhas `role='user'` são todas humanas. No claude-code resultado de ferramenta e texto de skill injetado carregam `role=user`, e sem o filtro qualquer saída de ferramenta contendo `--full` destrancaria o guarda |
| Como a leitura de contexto é limitada | `TAIL_BYTES` = 256 KB explícitos (o transcript desta sessão tinha 15,8 MB) | nenhum `LIMIT` no `select … where active=1`; apoia-se em o hermes virar `active=0` ao compactar | mesmo fim, mecanismos diferentes. NÃO é desempenho: medido a 100 MB de linhas `active=1` e ainda ~40 ms. É um invariante do host em que este arquivo se apoia sem verificar |
| Teto de UMA leitura (o que entra em `decide`) | só `read_lines` (2000, lido do binário v2.1.233) | `read_lines` (2000) **e** `read_bytes` = `READ_CHAR_CEILING` = 100.000 | `tools/file_tools.py:65` (`_DEFAULT_MAX_READ_CHARS`) trunca por CARACTERES; o `Read` do claude-code não tem esse teto legível. **Consequência aceita, e NÃO é um caso de canto — foi medida:** os dois hosts só precificam igual quando UMA leitura carrega no máximo 100.000 caracteres, ou seja arquivo abaixo de ~100 KB, ou linhas com média abaixo de ~50 bytes. Acima disso o hermes está preso em 25.000 tokens e o claude-code não: 4.000 linhas de 100 B (400 KB) dão 202.271 B contra 100.000 B — ou seja **50.567 tokens contra 25.000** — e 4.000 de 400 B dão 819.200 B contra 100.000 B, isto é **204.800 tokens contra 25.000**. Toda cifra desta linha é BYTES quando diz B e TOKENS quando diz tokens; misturar as duas foi um erro real deste documento, pego por um teste que derivava o número de `decide()` em vez de o copiar. Em quase todo arquivo grande o bastante para interessar à guarda os dois **podem decidir DIFERENTE**. É por isso que a equivalência se afirma como "mesmo `Budget` E mesmo CUSTO → mesmo veredito", nunca como "mesmo arquivo → mesmo veredito". Alinhar os tetos faria o teste passar mentindo sobre o host |
| Localização do estado do host | nada: `transcript_path` vem no payload | `QCTX_HERMES_STATE_DB`, senão `HERMES_HOME`, senão `~/.hermes` | o hermes não põe o banco no payload; o caminho tem de ser resolvido, e `hermes_constants.py::get_hermes_home` é a resolução que o próprio host usa |
| Timeout de I/O local | — | `SQLITE_TIMEOUT_S` = 0,5 s, com `mode=ro` | o hermes escreve no `state.db` AO VIVO e isto roda antes de cada leitura; um guarda que esperasse pelo lock seria uma interrupção auto-infligida. Do outro lado não há banco vivo nenhum |
| Bootstrap de `sys.path` | não precisa: hooks rodam como processo próprio | `REPO_ROOT` inserido em `sys.path` no topo do módulo | o loader do hermes pré-executa cada `*.py` irmão ANTES do `__init__.py`, registrando em `sys.modules` primeiro e engolindo a falha em `logger.debug`; sem o bootstrap o provider inteiro não carrega e a única pista é uma linha de debug |
| Knobs de sintonia (`QCTX_BIGFILE_FLOOR_PCT`, `QCTX_BIGFILE_SHARE_PCT`, `QCTX_BIGFILE_ESCAPE`) | idênticos | idênticos | é a promessa ao deployer, e a varredura de knobs de `tests/test_hermes_provider.py::KNOB_SOURCES` cobre os dois arquivos para que continue verdadeira |
| A decisão (`core.bigfile.decide`) | idêntica | idêntica | é o que a equivalência cobra, e a única camada em que ela é verdadeira |

### A palavra de escape

Marcador literal, default `--full`, configurável — `QCTX_BIGFILE_ESCAPE` (legado
`BIGFILE_ESCAPE`), lido pelo ADAPTADOR de cada host e passado para `decide`, que nunca tem
default próprio: o marcador aparece na MENSAGEM e na DETECÇÃO, e dois donos de um texto
configurável divergem — guarda que ensina uma palavra e recusa essa palavra. Valor em branco
cai no default; marcador de espaços casaria com toda mensagem. **Não** frase natural:
*"leia inteiro"* dispara falso positivo fácil (*"leia inteiro o parágrafo 3"*); `--full` só
se digita de propósito, e é configurável justamente porque existe domínio em que ele aparece
sozinho (quem trabalha numa CLI com flag `--full`).

Ganha o escopo certo de graça: o hook lê **o último turno do usuário**, então o `--full`
vale para aquele turno e evapora no próximo. Não existe estado a limpar, e não há como
deixar a guarda desligada por esquecimento — que era o defeito da opção "variável de
ambiente".

## Modos de falha

**A regra que domina: falha ABRE.** Isto é o inverso do hook de recall, e de propósito. Lá,
falhar custava a busca e o modelo precisava saber. Aqui, falhar fechado impediria o usuário
de ler arquivos sem dizer por quê — pior que o problema que a guarda resolve.

| Falha | Destino |
|---|---|
| transcript ausente ou ilegível | libera |
| `state.db` travado (o hermes escreve nele ao vivo) | libera, sem espera longa |
| janela desconhecida (modelo fora da tabela, sem config) | libera — nunca bloquear com base em palpite |
| `used >= window` (o palpite foi refutado pelos fatos) | libera — a janela real é maior do que a tabela supõe |
| `stat` do arquivo falha | libera; a leitura falhará com mensagem melhor |
| qualquer exceção inesperada | libera |

**Nunca escrever em stdout fora do protocolo.** No claude-code o stdout é o canal do hook.
Este é o defeito exato que a última revisão pegou no `env_num` do `recall.py`, onde
`print(file=None)` caía em stdout — aqui o custo seria o mesmo.

**Requisito de desempenho.** Isto roda antes de **cada** leitura de arquivo. Ler o transcript
inteiro seria inaceitável. Ler apenas a cauda (poucos KB), o suficiente para o último `usage`
e o último turno do usuário.

**O hook decide, não age.** Não indexa nada. Indexar é ação do modelo depois de ler a
mensagem. Responsabilidade única, e evita que uma leitura bloqueada dispare centenas de
chunks de embedding sem o usuário pedir.

> **PRECISÃO de 2026-08-17, medida.** "Não age" era forte demais como estava escrito, e a
> medição é melhor que a promessa. No caminho de BLOQUEIO — só nele — descobrir quem já está
> indexado passa por `DocIndex.list_docs`, que chama `ensure` e, no arquivo temporário,
> `sweep`. Medido nos dois hosts: `ensure_collection`, `ensure_payload_index` e
> `delete_by_filter`. **Zero embeddings e zero pontos escritos** — a catástrofe que esta
> cláusula existe para impedir continua impedida, e agora há teste que congela exatamente
> esse conjunto de operações.
>
> E o `sweep` não é efeito colateral indesejado: ele é o que torna a resposta CORRETA. Sem
> ele a guarda poderia dizer *"já indexado como X, busque nele"* sobre um documento que
> expirou — mandando o usuário buscar num acervo que não o tem mais. Conselho errado é
> exatamente o que os três casos do `decide` existem para evitar.
>
> Contornar isso exigiria uma listagem própria dentro da guarda, isto é, um SEGUNDO dono da
> pergunta "o que está indexado" — o defeito que a ruling F5 já corrigiu três vezes nesta
> feature. A redação é que estava imprecisa, não o código.

## Testes

- **`core/bigfile.py`** — tabela sobre `Budget` fabricado: os dois critérios, as fronteiras
  exatas, binário, já-indexado, e a diferença de mensagem entre exato e estimado. Sem rede,
  sem transcript, sem banco.
- **Cada adaptador** — transcript sintético e `state.db` sintético; afirma o `Budget`
  extraído, incluindo o filtro que separa turno real de injeção.
- **Equivalência** — mesmo arquivo e mesmo `Budget` produzem o mesmo `Verdict` nos dois
  adaptadores, no padrão de `tests/test_host_equivalence.py`.
- **Falha abre** — forçar cada linha da tabela de falhas e afirmar que **liberou**.
- **Sonda de mutação em toda guarda** — removê-la tem de deixar a suíte vermelha. Este repo
  já embarcou seis testes que passavam pelo motivo errado; a contagem de ocorrências da
  mutação é verificada antes de acreditar no resultado.

## SOLID

- **Responsabilidade única** — `decide` decide; o adaptador colhe; o protocolo traduz.
- **Aberto/fechado** — host novo é adaptador novo; o núcleo não muda.
- **Inversão de dependência** — `decide` recebe `Budget` e não sabe se veio de transcript, de
  SQLite ou de um teste. É o que torna o núcleo testável sem infraestrutura.

## Fora de escopo, deliberadamente

- **Terminal (`cat`, `head`)** — desviar pelo shell é escolha explícita de quem escreveu o
  comando, não descuido; e parsear linha de comando (pipes, redirecionamentos, `cat a b c`)
  traz falso positivo pior que o problema.
- **Busca que devolve muito (grep/glob)** — no `pre_tool_call` o hook vê os **argumentos**,
  não o resultado; só daria para adivinhar antes, ou mover para `post_tool_call`, que é tarde
  demais para bloquear.
- **Indexação de repositório, watcher, sub-coleções** — pedidos na mesma conversa, separados
  em spec própria por serem subsistemas independentes, um deles exigindo processo de fundo
  que o plugin hoje não tem (`core/docs.py:13` diz "no daemon", por decisão).
