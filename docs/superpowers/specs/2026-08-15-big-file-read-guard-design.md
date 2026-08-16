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
- definições no bloco `hooks:` do `~/.hermes/config.yaml`; consentimento em
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

Daí a decisão: **tabela como palpite, config do usuário vence**.

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

### A palavra de escape

Marcador literal, default `--full`, configurável. **Não** frase natural: *"leia inteiro"*
dispara falso positivo fácil (*"leia inteiro o parágrafo 3"*); `--full` só se digita de
propósito.

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
