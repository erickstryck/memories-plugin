# memories-plugin no hermes-agent — design

**Data:** 2026-08-14 · **Repo:** `memories-plugin` @ `2ceb111` · **Alvo:** hermes-agent v0.20.0 (2026.8.3)

> **Correção de 2026-08-14, depois de ler o install real.** A primeira versão desta spec foi
> escrita a partir do HEAD do GitHub, que está À FRENTE do que está instalado. Medido no
> install (`~/.hermes/hermes-agent/agent/memory_provider.py`, 357 linhas contra 404 no HEAD):
> `RecallStatus`, `recall_status()` e `unavailable_reason()` **não existem** na v0.20.0. O
> desenho sobrevive — ver §3.3 — mas as afirmações foram corrigidas.

## 1. Objetivo

Um plugin, dois hosts. O `memories-plugin` passa a ser importável pelo hermes-agent com as
**mesmas funções e a mesma configuração** que tem no claude-code. Nesta máquina os dois
agentes apontam para as mesmas coleções e contribuem um para o outro.

Não-objetivo: fork do hermes, repo separado, ou migrar dado existente.

## 2. Decisões

| # | decisão | motivo |
|---|---|---|
| D1 | Adaptador novo em `hosts/hermes/`, dentro deste repo | espelha `hooks/` (adaptador do claude-code); uma fonte de verdade |
| D2 | Instalação por symlink em `$HERMES_HOME/plugins/memories/` | fonte de primeira classe na descoberta do hermes; sem fork |
| D3 | Nome do provedor: `memories` | o diretório dá o nome; não colide com os 8 embutidos nem com o `qdrant` atual |
| D4 | Substitui o provedor `qdrant` atual | o hermes aceita **um** provedor externo por vez |
| D5 | Aponta para `claude_memory` / `memories_docs_library` / `memories_docs_tmp` | decisão do usuário: os dois agentes contribuem entre si |
| D6 | Os 1423 pontos de `hermes_memory` ficam onde estão | numa amostra de 200: 154 pedaços de doc e 46 turnos crus, não fatos curados; misturar é a poluição que a separação de acervos impede |
| D7 | Prosa injetada e montagem de bloco **movem para `core/`** | sem isso, equivalência vira copy-paste — ver §4 |
| D8 | `prefetch` bloqueante, sem thread de fundo | orçamento do hermes é 8s; o recall mede 0,5–1,7s em produção |
| D9 | Uma só config: `~/.config/memories-plugin/config.json` | `qctx setup` e `hermes memory setup` escrevem no mesmo arquivo |

## 3. O contrato do hermes, mapeado

O hermes expõe a ABC `MemoryProvider` (`agent/memory_provider.py`). O encaixe é quase 1:1.

| `MemoryProvider` | origem no plugin | nota |
|---|---|---|
| `prefetch(query) -> str` | bloco de recall | pipeline de dois portões, política de memória (veto) |
| `on_turn_start(n, msg)` | contador de cadência | roda todo turno e **fornece** o número |
| `system_prompt_block()` | `INSTRUCTIONS` | texto estático |
| `get_tool_schemas()` / `handle_tool_call()` | 15 das 20 operações (§3.1) | formato OpenAI function-calling; retorno é string JSON |
| `is_available()` | `core.load()` + guardas | sem rede: só checa config |
| `unavailable_reason()` · `recall_status()` | — | **não existem na v0.20.0**; implementados como forward-compat (§3.3) |
| `on_session_end(messages)` | **no-op deliberado** | ver §3.4 |
| `get_config_schema()` / `save_config()` | `core/config.py` | as 2 chaves de API marcadas `secret` → o hermes as manda para `.env` |
| `shutdown()` | — | no-op |

### 3.1 Quais operações viram ferramenta

| viram ferramenta (15) | ficam só na CLI (5) | motivo |
|---|---|---|
| `memory-store`, `memory-store-many`, `memory-find`, `memory-recall`, `memory-get`, `memory-update`, `memory-delete`, `memory-list`, `memory-search-collections` | `setup` | interativo, pede TTY; o hermes tem `hermes memory setup` |
| `docs-index`, `docs-keep`, `docs-search`, `docs-list`, `docs-refresh`, `docs-drop` | `config-set`, `config-detect`, `config-show`, `collections` | configuração é do operador, não do modelo; expor `config-set` como ferramenta deixaria o modelo apontar o acervo para outro lugar |

A CLI `qctx` continua disponível nos dois hosts para as 20 — a restrição é sobre o que o
**modelo** pode chamar sozinho.

### 3.2 A versão instalada e o que ela não tem

Medido no install, não no GitHub. A v0.20.0 expõe 19 métodos públicos e 4 abstratos
(`name`, `is_available`, `initialize`, `get_tool_schemas`). Tem `is_trivial_prompt`,
`queue_prefetch` e `sync_turn`. **Não** tem `RecallStatus`, `recall_status()` nem
`unavailable_reason()`.

Consequência de desenho: a visibilidade da degradação **não pode** depender de
`recall_status()`. Ela já não depende — a nota de degradação viaja DENTRO do texto que o
`prefetch` devolve, que é o mesmo mecanismo do claude-code. O indicador determinístico do
hermes seria um bônus, não o contrato.

### 3.3 Forward-compat: implementar o que ainda não é chamado

`recall_status()` e `unavailable_reason()` são implementados de qualquer forma. Numa v0.20.0
ninguém os chama, então são inertes; se o usuário atualizar o hermes, passam a funcionar sem
tocar no plugin. Custo: duas funções curtas. `RecallStatus` é importado com fallback local,
porque a classe não existe na versão instalada.

Herança da ABC segue o mesmo padrão: `try: from agent.memory_provider import MemoryProvider`
com fallback para `object`, porque a suíte de testes deste repo roda **sem** o hermes no path
(ele tem venv próprio). O registro usa `register(ctx)` — o caminho preferido do loader, que
não faz `issubclass` — então a herança é conveniência, não requisito.

### 3.4 `on_session_end` é no-op nesta versão

O claude-code não tem gancho de fim de sessão, e a exigência é equivalência. Implementar
extração de fim de sessão só no hermes criaria uma assimetria de comportamento entre os dois
hosts — exatamente o que este design existe para evitar. Fica registrado como oportunidade
conhecida, não como pendência.

**A cadência do checkpoint pega carona no `prefetch`.** `on_turn_start` roda todo turno mas
devolve `None`, então não injeta. Os únicos pontos de injeção são `prefetch` e
`system_prompt_block`. Logo: `on_turn_start` conta, e no turno N o `prefetch` devolve
`bloco de recall + procedimento de checkpoint`.

## 4. A consequência de "equivalente"

Hoje a prosa injetada e a montagem do bloco vivem em `hooks/`, que é o adaptador do
claude-code. Portar copiando faz os dois divergirem no primeiro conserto — este projeto já
pagou essa conta com três cópias do pipeline de dois estágios, onde a normalização de escala
do re-rank existia numa e não na outra.

**Move para `core/` (decisão), fica no adaptador (protocolo do host):**

| novo módulo em `core/` | conteúdo | sai de |
|---|---|---|
| `prompts.py` | `INSTRUCTIONS`, `CHECKPOINT_PROCEDURE` | `hooks/recall.py`, `hooks/checkpoint.py` |
| `blocks.py` | montar os 4 estados de bloco, orçamento de contexto, ponteiros, nota de degradação | `hooks/recall.py` |
| `session_state.py` | rodada, `seen`, poda, purga de sessão morta, cadência | `hooks/recall.py`, `hooks/checkpoint.py` |

| fica no adaptador | claude-code | hermes |
|---|---|---|
| ler o prompt | JSON no stdin | argumento de método |
| emitir | `print` de JSON | `return str` |
| diretório de estado | `QCTX_STATE_DIR` | `hermes_home` do `initialize()` |
| número do turno | contador em arquivo | `on_turn_start(turn_number)` |

`hooks/recall.py` cai de 501 linhas para uma casca fina. O núcleo continua sem saber que
host existe: recebe texto, devolve bloco.

## 5. Os 4 estados do bloco, iguais nos dois hosts

| estado | quando | o que afirma |
|---|---|---|
| memórias listadas | houve acerto | o acervo foi consultado |
| vazio + julgamento completo | nada passou do corte | não há precedente; não repita a busca |
| vazio + julgamento parcial | degradação (erro, colapso, suprimido, descarte acima do piso) | **não** é evidência de ausência |
| indisponível | busca não rodou | o acervo não foi consultado; não afirme ineditismo |

O quarto estado é o contrato central: falha em silêncio para o **usuário**, nunca para o
**modelo**. No hermes ele sai por `prefetch` devolvendo o bloco de indisponibilidade, e por
`unavailable_reason()` quando o provedor nem inicializa.

## 6. Filtro de prompt trivial

O hermes tem `is_trivial_prompt` própria, **só em inglês** (`yes|no|ok|thanks|hi|…`), aplicada
antes de chamar o provedor. A do plugin é portuguesa e composta (`TRIVIAL_WORDS` ∪ `STOPWORDS`).

Decisão: as duas rodam, em série. A do host corta o inglês; a do plugin corta o português que
a do host deixa passar — medido: `"ok, pode continuar"` não casa com a regex deles.

## 7. Configuração equivalente

Mesmo arquivo, mesmas chaves, mesmos aliases legados. `get_config_schema()` declara os campos
com `secret: True` nas duas chaves de API — o que casa com o `save()` do plugin, que **recusa**
gravar segredo em arquivo e manda exportar no ambiente.

| campo | onde vive |
|---|---|
| `qdrant_url`, `api_base_url`, `embed_url`, `rerank_url`, modelos, 3 coleções, `vector_size` | `config.json` |
| `qdrant_api_key`, `api_key` | `.env` (hermes) / ambiente (claude-code) |

## 8. Armadilha medida: symlink e `__file__`

Com o provedor instalado por symlink, `os.path.abspath(__file__)` devolve o caminho do
**symlink** (`$HERMES_HOME/plugins/…`), não o do repo. O padrão que `hooks/recall.py` usa hoje
para achar a raiz falharia.

| errado | certo |
|---|---|
| `dirname(dirname(abspath(__file__)))` | `dirname(dirname(dirname(realpath(__file__))))` |

Medido em probe: `abspath` → `/tmp/symprobe/home/plugins`; `realpath` → `/tmp/symprobe/repo`.

## 9. Testes

| teste | o que prova |
|---|---|
| contrato ABC | a classe satisfaz `MemoryProvider` (todos os abstratos implementados) |
| **equivalência de bloco** | os dois adaptadores, sobre o mesmo `Outcome`, produzem bloco **idêntico** |
| equivalência de config | os dois leem a mesma config e resolvem as mesmas coleções |
| 4 estados | cada estado sai correto pelo caminho do hermes |
| ferramentas | cada schema declarado é roteado por `handle_tool_call` e devolve JSON válido |
| cadência | o procedimento aparece no turno N e só nele |
| trivial | prompt trivial em português não gasta rede no caminho do hermes |
| symlink | o provedor carrega e acha `core/` quando instalado por symlink |

A equivalência é **testada, não afirmada**. Sem esse teste, D7 é uma intenção; com ele, é
contrato. Todo teste novo exige vermelho provado antes do verde.

## 10. Cutover nesta máquina

| passo | ação | reversível por |
|---|---|---|
| 1 | symlink `$HERMES_HOME/plugins/memories` → `hosts/hermes` | `rm` do symlink |
| 2 | `config.json`: já aponta para as 3 coleções | backup datado |
| 3 | `config.yaml` do hermes: `memory.provider: qdrant` → `memories` | backup datado |
| 4 | conferir: `hermes memory setup` lista `memories` como disponível | — |

Não muda: as coleções, os 1423 pontos, `~/.secrets`, as URLs do `.bashrc`, o diretório do
provedor `qdrant` (desativado por config, não apagado).

Ensaio antes de aplicar, como o `scripts/cutover.sh` já faz para o claude-code.

## 11. Fora de escopo

- Migrar ou converter os 1423 pontos de `hermes_memory`.
- Publicar no PyPI / entry point (`hermes_agent.memory_providers`) — o symlink resolve nesta
  máquina; o entry point é o caminho de distribuição, e fica para quando houver distribuição.
- Alterar o provedor `qdrant` de terceiro.
- Suporte a outros hosts além destes dois.
