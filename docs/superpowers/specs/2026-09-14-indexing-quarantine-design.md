# Quarentena de indexação — desenho

O watcher do daemon reenfileira indefinidamente arquivos que **nunca podem ser indexados**, e a
carga resultante no servidor de embeddings faz o *recall automático* estourar o próprio prazo.
O sintoma que o usuário vê não tem relação aparente com indexação:

> `[automatic recall — UNAVAILABLE for this prompt]`
> `The long-term memory search was NOT executed: EmbeddingError failed (network failure on`
> `http://…:8003/v1/embeddings: TimeoutError: timed out).`

Este documento desenha a correção. Diagnóstico completo medido em 2026-09-14, na máquina do
usuário, contra o acervo e a infra reais.

## O que foi medido

**A correlação.** O daemon (`python -m core.daemon`) subiu às 14:57:31,62. O primeiro
`EmbeddingError` é de 14:57:31,63 — **10 ms depois**, e zero ocorrências antes dele. Foram 20
blocos `UNAVAILABLE` entre 14:57 e 18:15, contados em `~/.hermes/state.db` (coluna `api_content`).

**O loop.** Os mesmos 22 arquivos voltam à fila a cada ~40 s, indefinidamente. Observado ao vivo
por três ciclos completos, lendo `state/jobs/*.json`.

**A saturação.** O endpoint de embeddings é compartilhado entre o indexador e o recall:

| condição | latência do embed |
|---|---|
| servidor ocioso | **0,04 s** |
| durante um batch do indexador | **1,98 s** |
| teto do recall no host hermes | **2,00 s** |

O recall estoura por dois centésimos de segundo. Amostrando o endpoint a cada 2 s durante 100 s:
picos de 1,0–1,3 s a cada ~40 s, **em fase com o ciclo do daemon**.

**O que foi descartado por medição:** rede, Tailscale, Qdrant e o reranker estão sãos.
`store.recall()` com os timeouts reais do host roda 8/8 vezes em ~0,7 s fora dos picos.

## Os quatro defeitos

São independentes. O **A** sozinho estanca o sangramento; os outros três são a causa de *por que
esses arquivos específicos* nunca entram.

### A. O watcher não tem memória de fracasso

`indexer.work()` chama `target.add_files(repo, chunk)` e **descarta o retorno**
(`core/indexer.py:45`). Esse retorno já carrega `skipped: [(path, motivo)]` — a informação existe
e é jogada fora. Sem ela, `poll()` reporta o arquivo como não indexado no ciclo seguinte, para
sempre.

Vale para os **dois** tipos de job: `index` e `refresh`. Um arquivo que mudou e falha ao
re-embeddar continua eternamente em `changed`.

**Por que os testes não pegaram:** `FakeIndex.add_files` (`tests/test_watch.py:72`) sempre tem
sucesso. O cenário "falhou, volta pra fila" nunca foi exercitado.

### B. O chunker viola o próprio teto

`HARD_MAX_CHARS = 6000` está documentado em `core/chunk.py:16` como invariante — *"o par (query,
chunk) tem que caber no reranker com folga"*. Mas `_window()` (`core/chunk.py:100`) acumula **por
linha** e nunca corta *dentro* de uma linha:

```
while cursor < end and accumulated < target:
    accumulated += len(lines[cursor])
```

Uma linha de 107.648 chars vira um chunk de 108.141 chars — 18× o teto. Medido no
`ibm-tririga-source/api.json`: 12 chunks, os três maiores com 108.141, 107.696 e 107.648 chars.

É isso que produz o HTTP 500:

```
input (83086 tokens) is too large to process (current batch size: 8192)
```

**O `EMBED_BATCH=32` NÃO tem defeito.** Medido: o limite de 8192 tokens do servidor é **por
item**, não por requisição — 32 itens de 2.400 chars (~25 mil tokens no total) passam em 4,17 s;
um único item de 10 mil tokens falha. Não mexer nele.

### C. A guarda de minificado é míope

`scan._sniff()` decide "minificado" lendo apenas os primeiros 8192 bytes (`core/scan.py:34`). No
`api.json` a primeira linha acima de 2.000 chars começa no **byte 16.687** — passa batido pela
guarda que existe exatamente para barrá-la. Maior linha nos primeiros 8 KB: 722 chars.

### D. `refresh` quebra com `AttributeError` em qualquer arquivo pulado

`add_files` devolve `skipped` como **tupla** `(path, motivo)` (`core/repos.py:191`), e o CLI lê
assim, corretamente (`cli/qctx.py:1318`). Mas `refresh` lê como **dict**:

```
out["skipped"][0].get("reason", "unreadable")      # core/repos.py:418
```

`'tuple' object has no attribute 'get'`. A exceção sobe até `daemon._run_one`, que marca o job
inteiro como `FAILED` — e **os arquivos restantes daquele refresh nunca são reindexados**. Defeito
latente hoje, mas a camada 1 depende de ler esse campo, então tem de ser corrigido junto.

## Por que esses 22 arquivos nunca entram

| quantos | por quê | defeito |
|---|---|---|
| 20 | têm **0 bytes** → `_write_one` levanta `nothing indexable (empty file, or whitespace only)` | A |
| 2 | `api.json` de cliente OpenAPI gerado, linha única de ~107 mil chars → HTTP 500 | A + B + C |

## O que se decidiu

| Decisão | Escolha | Alternativas recusadas |
|---|---|---|
| Onde mora a quarentena | **`core/quarantine.py`**, JSON por repo em `state_dir/quarantine/` | campo em `jobs.json`; coleção no Qdrant; dentro de `repos.py` |
| Quando reter | **Quando o conteúdo muda** (mtime/size ≠ do registrado) | nunca; backoff crescente por tempo |
| Visibilidade | **Contagem + motivos no `qctx repos status`** | silencioso; comando dedicado |
| Os dois `api.json` | **Saem do acervo** como minificados (camada 3) | fatiar e indexar |
| `EMBED_BATCH` | **Não mexer** — o limite do servidor é por item | reduzir para caber em 8192 tokens |

**Por que fora de `repos.py`:** a camada do acervo não deve conhecer estado local em disco.
`quarantine.py` segue o mesmo idioma de `jobs.py` e `lease.py` — escrita atômica via
`os.replace`, `OSError` tolerado, sem protocolo entre processos.

**Por que reter pelo conteúdo e não por tempo:** a quarentena descreve *um conteúdo*, não um
caminho. Um arquivo de 0 bytes que ganha conteúdo volta a ser indexado sozinho, sem comando
manual — que é exatamente o caso dos 20 arquivos vazios deste repositório. Backoff por tempo
traria de volta o custo periódico que esta spec existe para eliminar.

**Por que sem comando de limpeza:** se o reteste é automático, limpar à mão quase nunca é
preciso. YAGNI — cabe adicionar depois se a necessidade aparecer.

## Restrições de engenharia (valem para todas as camadas)

Regra permanente do usuário, dita em 2026-08-04: **KISS e S.O.L.I.D. em tudo**. Com o
qualificador de 2026-08-19, que pesa tanto quanto a regra: **simples não pode custar completo** —
*"não adianta ser simples se não cobrir toda a config necessária"*. As duas metades juntas, não
em tensão.

### Como aplicar aqui

**Escrever a versão direta primeiro.** Estrutura só entra quando um requisito **real e presente**
exigir, nunca por hipótese. Antes de entregar cada peça, comparar com como o código existente
resolve o mesmo problema — *"em quantas linhas?"*. Divergência grande é sinal de invenção, não de
rigor. Neste projeto essa regra já foi paga duas vezes: maquinário defensivo que **causou** o bug
que se estava depurando.

**O molde é `core/jobs.py` e `core/lease.py`** (237 e 160 linhas). `quarantine.py` deve ficar
nessa ordem de grandeza. Se passar disso, é sinal de que ganhou responsabilidade que não é dele.

**SOLID, aplicado ao que esta spec realmente pede:**

| Princípio | O que significa concretamente aqui |
|---|---|
| **S** — responsabilidade única | `quarantine.py` responde uma pergunta: *este conteúdo já falhou?* Não decide política de reteste, não fala com Qdrant, não formata saída. Quem decide é `indexer`; quem exibe é o CLI. |
| **O** — aberto/fechado | Um motivo novo de falha (formato novo, erro novo do servidor) não deve exigir mudança em `quarantine.py`: o motivo é **dado opaco**, string vinda de quem falhou, nunca um enum que a quarentena precise conhecer. |
| **L** — substituição | `FakeIndex` nos testes tem de honrar o mesmo contrato de `RepoIndex` — incluindo **falhar** como o real falha. É exatamente a violação disso (um fake que só sabe ter sucesso) que deixou o defeito A passar. |
| **I** — segregação de interface | `indexer` consome `held()`/`record()`/`clear()`. Não recebe o dicionário inteiro para filtrar por conta própria. |
| **D** — inversão de dependência | `quarantine` depende de `knobs.state_dir()`, a abstração que todo o resto usa, e de mais nada. Não importa `repos`, não importa `core`. A direção da dependência é `indexer → quarantine`, nunca o contrário. |

**Manutenibilidade — o padrão desta base de código, que deve ser seguido:**

- **Comentar a intenção, não a mecânica.** O padrão aqui é explicar *por que*, com a medição que
  motivou a escolha. Todo comportamento não óbvio ganha o número que o justifica, como
  `core/indexer.py` faz com os 16 ms do polling.
- **Falha tolerada não pode virar mentira.** `OSError` na quarentena degrada para "nada retido" —
  o comportamento de hoje — e nunca para "está tudo indexado".
- **Só stdlib.** Restrição real do projeto, documentada no README com o motivo: dependência
  faltando dentro de um hook vira perda silenciosa de funcionalidade.
- **Nomes que dizem o que a coisa é.** `held()` responde "o que está retido", não `get_paths()`.

### Como a completude é provada

Pelo padrão que o usuário aprovou em 2026-08-19: **não pela minha palavra de que auditei, mas por
uma asserção mecânica**. Cortar peça é bom; deixar buraco não — toda passada de corte vem seguida
de uma passada de cobertura.

Concretamente, nesta spec: os quatro defeitos (A–D) têm teste RED nomeado na tabela de testes
abaixo, e a verificação em produção é uma lista de quatro observações mensuráveis, não uma
impressão de que melhorou.

## Arquitetura

### Camada 1 — quarentena (obrigatória; estanca o sangramento sozinha)

Módulo novo `core/quarantine.py`, arquivo por repo em
`state_dir/quarantine/<repo>.json`:

```json
{
  "/caminho/absoluto/api.json": {
    "reason": "HTTP 500 on POST …: input (83086 tokens) is too large to process",
    "mtime": 1789420947.6,
    "size": 125434,
    "at": "2026-09-14T18:24:31+00:00"
  }
}
```

Funções, no idioma de `jobs.py`:

- `load(repo) -> dict` — `{}` quando não há arquivo ou está corrompido
- `record(repo, path, reason)` — grava com `mtime`/`size` lidos do disco no momento
- `held(repo) -> set[str]` — só os paths cujo `mtime`/`size` **ainda batem** com o registrado;
  um arquivo que mudou não está retido, e a entrada morta é descartada na mesma passagem
- `clear(repo, paths)` — remove entradas (usado quando um path é indexado com sucesso)

Fluxo de dados:

1. `indexer.work()` passa a **ler** o retorno de `add_files` e chamar `quarantine.record()` para
   cada `(path, motivo)` em `skipped`; em caso de sucesso, `quarantine.clear()` para os paths que
   entraram.
2. `indexer.watcher()` → `_new_tracked_paths()` subtrai `quarantine.held(repo)` do conjunto de
   candidatos.
3. Para o job `refresh`, a mesma leitura a partir do relatório que `refresh()` já devolve —
   **apenas** entradas com `action == "skipped"`. Toda entrada do relatório carrega `path`,
   então a leitura é direta.

   `action == "missing"` (arquivo apagado do disco) **não** entra na quarentena. São estados
   diferentes: "não pode ser indexado" versus "não está mais lá". O segundo já é tratado —
   `refresh` o reporta e mantém os chunks, deliberadamente, porque deleção neste plugin é
   explícita — e ele não gera custo de embedding nenhum, que é o problema desta spec. Retê-lo
   também apagaria o aviso que o usuário deveria continuar vendo.

O ponto 2 é o que fecha o loop: o arquivo deixa de ser candidato, então não é enfileirado, então
não custa embedding nenhum.

### Camada 2 — honrar o `HARD_MAX_CHARS`

`_window()` passa a fatiar uma linha mais longa que `target` em pedaços de até `target` chars.
Corrige a invariante para **qualquer** arquivo com linha longa legítima — CSV largo, tabela
markdown, JSON de uma linha — e não só para estes dois.

Os números de linha do chunk continuam válidos: os pedaços de uma linha fatiada reportam todos a
mesma linha de início e fim. `mode_for_suffix` já marca `.json` como `locator`, e ler a linha
inteira de volta é o comportamento correto para um consumidor.

### Camada 3 — `_sniff` olhar o arquivo todo

Trocar a leitura de 8 KB por uma varredura do maior comprimento de linha do arquivo inteiro. O
custo é ler arquivos já limitados a `MAX_FILE_BYTES` (1 MB), e só na varredura de elegibilidade —
não no ciclo do watcher, que compara `mtime`/`size`.

Com isso os dois `api.json` saem do acervo como `minified`, que é o que a regra sempre quis
dizer.

## Erros e degradação

Segue a regra já estabelecida no projeto — **falha aqui não abre**, mas também não mata:

- Um `OSError` ao ler ou escrever a quarentena é tolerado: o pior caso é reindexar um arquivo que
  teria sido retido, ou seja, o comportamento de hoje. Não pode derrubar um ciclo do daemon.
- JSON corrompido lê como `{}` (nada retido), nunca levanta.
- A quarentena **nunca** suprime um erro do usuário: o motivo aparece no `status`.
- Ela não é autoridade sobre o acervo. Se um path está em quarentena mas tem chunks no Qdrant
  (indexado antes de quebrar), os chunks ficam — deleção neste plugin é explícita.

## Testes

RED antes de verde nas quatro. O projeto é stdlib-only e a suíte é offline.

| # | Prova | Arquivo |
|---|---|---|
| A1 | um path que falha não é reenfileirado no ciclo seguinte | `tests/test_watch.py` |
| A2 | o mesmo, pelo caminho `refresh` | `tests/test_watch.py` |
| A3 | arquivo em quarentena volta a ser candidato quando `mtime`/`size` mudam | `tests/test_quarantine.py` (novo) |
| A4 | `OSError` na quarentena não derruba o ciclo | `tests/test_quarantine.py` |
| B1 | nenhum chunk excede `HARD_MAX_CHARS`, com linha de 100 mil chars | `tests/test_chunk.py` |
| C1 | linha longa **depois** do byte 8192 é detectada como minificada | `tests/test_scan.py` |
| D1 | `refresh` reporta um arquivo pulado em vez de levantar `AttributeError` | `tests/test_repos_refresh.py` |

`FakeIndex` (`tests/test_watch.py:35`) ganha a capacidade de **falhar** em paths escolhidos — é a
ausência disso que deixou o defeito A passar.

## Verificação em produção

Depois de aplicar, na máquina do usuário:

1. `state/jobs/*.json` deixa de rotacionar — os mesmos jobs param de voltar a `pending`.
2. Amostrar o embed a cada 2 s por ~2 min: sem picos acima de 0,5 s.
3. Nenhum bloco `UNAVAILABLE` novo em `~/.hermes/state.db` após o restart do daemon.
4. `qctx repos status` mostra os 22 arquivos retidos, com motivo.

## O que fica de fora

- **Comando para limpar a quarentena à mão.** O reteste automático cobre o caso real.
- **Mudar `EMBED_BATCH`.** Medido: não é o gargalo, o limite do servidor é por item.
- **Serializar indexador e recall**, ou dar prioridade ao recall no endpoint. Seria tratar o
  sintoma; com o loop corrigido, a carga de fundo some.
- **Remover os chunks já indexados** de arquivos que agora serão considerados minificados.
  Deleção é explícita neste plugin.
