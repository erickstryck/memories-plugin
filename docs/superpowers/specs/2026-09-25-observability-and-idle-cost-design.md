# Observabilidade e custo em repouso: desenho (v1.1.0)

O plugin funciona, mas duas coisas estão erradas e uma terceira é invisível:

1. O daemon consome CPU e rede o tempo todo, mesmo quando nada mudou no disco.
2. O daemon relê e refaz o SHA-1 de repositórios inteiros, em loop, por causa de arquivos cujo
   mtime mudou e o conteúdo não.
3. O uso diário (pelo hermes) não deixa rastro nenhum. Latência, falhas e breaker só aparecem no
   log do host claude-code, e o daemon não loga nada.

Medido em 2026-09-25, na máquina do usuário, contra o acervo real, com a v1.0.1 instalada.

## O que foi medido

**Custo em repouso do watcher.** O daemon (pid 108492) sem nenhum job: 2,57 s de CPU em 30 s
(~8,6%) e `rchar` de 829 MB em 4 minutos (~1,4 MB/s). A cada ciclo de 5 s, `watcher()` chama
`poll(repo)` para cada repo, e `poll` faz `scroll_all` de todos os chunks do repo:

| repo | chunks | páginas/ciclo | JSON/ciclo | tempo |
|---|---|---|---|---|
| awesome-cv3 | 28.019 | 110 | 11,0 MB | 3,4 s |
| core | 5.688 | 23 | 2,3 MB | 0,7 s |
| connector-sdk | 3.278 | 13 | 1,2 MB | 0,4 s |
| place-mappings-micro | 1.251 | 5 | 0,5 MB | 0,1 s |
| mapped-connector-skills | 445 | 2 | 0,2 MB | 0,05 s |

Total: ~153 requisições HTTPS (sem keep-alive) e ~15 MB por ciclo.

**Loop de refresh por mtime.** `changed_paths` (só mtime/tamanho) contra `source_changed`
(digest), por repo:

| repo | arquivos | mtime/tamanho diferente | conteúdo diferente | sumiram |
|---|---|---|---|---|
| awesome-cv3 | 10.759 | 567 | 406 | 153 |
| connector-sdk | 681 | 322 | 22 | 15 |
| core | 2.161 | 1 | 1 | 1 |

No connector-sdk, ~300 arquivos estão em `changed` com conteúdo idêntico. O `refresh` conclui
"ok" para eles e não regrava o mtime, então o watcher enfileira outro refresh a cada ~2 ciclos,
e cada refresh lê e faz hash de todos os 681 arquivos. Observado ao vivo: um refresh do
awesome-cv3 ficou `running` por mais de 1 minuto sem progresso visível.

**Lacunas de observabilidade.**
- `hosts/hermes/__init__.py` não chama nenhum log. A sessão hermes de 2026-09-25 10:40 não aparece
  no `recall.log`; a última linha é de uma sessão claude-code.
- O daemon é iniciado com stdout/stderr em `DEVNULL` (`core/daemon.py:727`), e as exceções do
  watcher somem num `pass` (`core/daemon.py:648-654`).
- Pelo `recall.log` atual (claude-code, 1.186 linhas): p50 1,1 s, p95 2,7 s, máx 7,3 s; 13
  falhas de Qdrant, 2 de embeddings, 4 de re-rank. Hoje só dá para chegar nesses números à mão.
- O claude-code estava rodando o commit `5b0f59b` (8 de setembro) e o hermes a `fc18389`.
  Os dois iniciam o daemon, então o código do daemon depende de quem o iniciou, e nada mostra isso.

## Os defeitos

Os três são independentes. O **A** sozinho zera o custo de rede em repouso.

### A. O watcher relê o acervo inteiro a cada ciclo

`core/indexer.py:152`: `state = target.poll(repo)`, e `poll` → `_indexed_sources` →
`scroll_all` (`core/repos.py:541-575`). A docstring de `changed_paths` orça o stat local
("16 ms para 2.000 arquivos") e não orça a releitura remota, que é o custo real.

O teste `test_a_cycle_reads_the_archive_exactly_once` (`tests/test_watch.py:275`) afirma que a
leitura "não pode ser memoizada", porque um memo que sobreviveu a um job re-enfileirou os mesmos
arquivos para sempre. **Isto é uma reversão explícita dessa decisão**, e ela só é segura com a
invalidação descrita abaixo, que cobre exatamente o caso que motivou a regra.

### B. O `refresh` não regrava o mtime quando o conteúdo confere

`core/repos.py:470-472`: `if reason is None: report.append({"action": "ok"})`. O mtime/tamanho
gravado continua o antigo, então `changed_paths` segue reportando o arquivo para sempre. O mesmo
acontece em `add_files` no ramo "inalterado" (`core/repos.py:206-214`).

### C. Nada registra o que acontece no hermes nem no daemon

Descrito acima. Consequência: o host usado no dia a dia é o único que não dá para diagnosticar.

## O que se decidiu

| Decisão | Recusadas |
|---|---|
| Cache do mapa `path → metadata` **no processo do daemon**, invalidado por (a) mudança no registro do repo (`indexed_at`/`files`/`chunks`), (b) mudança de `id` ou `state` do job do repo, (c) 300 s de idade | inotify (dependência externa ou código só para Linux); cache em arquivo (o daemon é o único leitor em loop); só TTL (deixaria o loop B de pé por 5 min após cada job) |
| O `refresh` e o `add_files` regravam só `src_mtime`/`src_size` (via `scroll` + `set_payload`, que já estão na porta) quando o digest confere | re-embeddar (caro e inútil); ignorar mtime no `changed_paths` (obrigaria a ler o conteúdo de todo arquivo a cada ciclo) |
| `core/eventlog.py`: dono único de "anexar uma linha com horário a um log de estado", com rotação a 256 KB e modo 0600 | `logging` da stdlib (handlers globais dentro de um host que tem os próprios; rotação por arquivos numerados foge ao padrão 0600 de `statefile`) |
| `core/recall_log.py`: dono único do FORMATO da linha de rodada e do parser dela; os dois hosts gravam o mesmo `recall.log`, com `[host]` em cada linha | um log por host (o breaker já é compartilhado por ser um fato da máquina; a latência também é) |
| `daemon.log` com start/exit, versão, jobs (duração/resultado), enfileiramentos, recargas do cache e erros do watcher, sem repetir o mesmo erro a cada ciclo | stderr do daemon para arquivo (captura traceback de terceiros sem formato; a rotação ficaria fora do nosso controle) |
| `daemon.json` ganha `version`; `qctx repos status` mostra a versão e avisa quando é diferente da do CLI | recusar iniciar/parar por diferença de versão (mudança de comportamento que ninguém pediu) |
| `qctx stats [--json]`: resumo dos dois logs. Fora das ferramentas do modelo | expor como ferramenta (descreve a operação do plugin para o operador, não responde nada sobre o acervo; mesmo motivo de `repos_status`) |

**Fora do escopo (YAGNI):** keep-alive HTTP; chmod automático de diretórios antigos
(`statefile.ensure_dir` respeita diretório existente por decisão registrada no próprio módulo);
métricas em formato Prometheus/OTel; logar os avisos de config do hermes no arquivo.

## Restrições de engenharia (valem para toda tarefa)

KISS e S.O.L.I.D., com a ressalva de mesmo peso: **simples não pode custar completo.**

- **S:** `eventlog` só escreve linha, `recall_log` só conhece o formato da rodada, `stats` só
  agrega. Nenhum deles decide política de recall ou de indexação.
- **O:** um terceiro host grava no `recall.log` chamando `recall_log.record(host, ...)`, sem
  editar nenhum desses módulos.
- **L:** o `FakeIndex` dos testes precisa expor `indexed_sources` e `changed_paths(repo,
  sources=)` com a mesma forma do `RepoIndex`; um fake que aceita menos do que o real é como o
  laço de reindexação passou pelo review da última vez.
- **I:** o watcher passa a depender de dois métodos pequenos (`indexed_sources`,
  `changed_paths`) em vez de `poll`, que misturava as duas perguntas numa só leitura.
- **D:** `indexer` e `daemon` escrevem via `eventlog`, não abrem arquivo; o relógio do cache é
  injetável (`clock=time.monotonic`), como já é o `sleep` do `daemon.run`.
- **Tamanho de referência:** `core/breaker.py` (67 linhas) e `core/jobs.py` (258). Um módulo novo
  que passe de ~150 linhas ganhou uma responsabilidade que não é dele.
- **Log nunca derruba nada:** toda escrita de log devolve bool e nunca levanta, a mesma regra de
  `statefile`. Perder uma linha é mais barato que perder um recall ou um ciclo.
- **Stdlib apenas**, como o resto do plugin.
- **Nada do que funciona hoje pode degradar** (pedido explícito do usuário). Provado por: suíte
  completa comparada com a linha de base por NOME de teste; suíte de integração antes e depois;
  e um roteiro ponta a ponta contra a infra real, rodado na v1.0.1 e na v1.1.0, comparando o que
  cada um devolve para as mesmas entradas.

## Arquitetura

### Watcher (`core/indexer.py`)

```
watch():
  para cada entry em list_repos():
    job = jobs.load(repo); pula se pending/running; pula se breaker aberto
    key = (entry.indexed_at, entry.files, entry.chunks, job.id, job.state)
    se memo[repo] ausente, ou key diferente, ou idade > SOURCES_TTL_S:
        sources = target.indexed_sources(repo)      # a única leitura do acervo
        memo[repo] = (key, agora, sources); loga "sources repo=… files=… secs=…"
    changed = target.changed_paths(repo, sources=sources)   # só stat local
    indexed = set(sources)
    … resto igual (quarentena, debounce, enqueue, que agora loga "enqueue …")
  descarta do memo os repos que saíram do registro
```

`RepoIndex._indexed_sources` vira público (`indexed_sources`) e `poll` é removido: seu único
chamador era o watcher, e a docstring dele ("o método para quem roda em loop") deixaria de ser
verdade.

Casos de invalidação e quem os cobre:
- job do daemon terminou (index/refresh/restamp) → muda `job.state`
- `qctx repos add/refresh` ou ferramenta do hermes → `add_files` muda `indexed_at`/contagens
- `repos drop` → o repo some de `list_repos`
- restamp feito por `qctx repos refresh` fora do daemon, ou perda de chunks por fora → TTL de 300 s
  (no pior caso, um refresh a mais; nunca um arquivo perdido para sempre)

### Restamp (`core/repos.py`)

`_metadata_drifted(st, md) -> bool`: a comparação de mtime/tamanho que hoje vive dentro de
`changed_paths`, extraída para ter um dono só. `refresh` e `add_files`, quando o digest confere e
`_metadata_drifted` diz sim, chamam `_restamp(path, st)`: `scroll` dos chunks com filtro `doc_id`,
`metadata.src_mtime/src_size` atualizados, `set_payload` por chunk. Sem embedding. Erro de
infraestrutura propaga, como em `add_files` (o job falha e o breaker de indexação arma).

### Logs

- `core/eventlog.py`: `RECALL = "recall.log"`, `DAEMON = "daemon.log"`, `path(name)`,
  `rotate(path, max_bytes)` (movido de `hooks/recall.py`), `write(name, line) -> bool`. Cria o
  arquivo com `os.open(..., 0o600)`, que hoje nasce com o umask.
- `core/recall_log.py`: `record(host, msg)`, `round_line(...)`, `empty_line(...)`,
  `parse(line) -> dict | None`. Linhas antigas sem `[host]` são lidas como `claude-code` (era o
  único host que escrevia).
- `hooks/recall.py`: `log()` passa a usar `recall_log.record("claude-code", …)`; as linhas de
  rodada vêm de `round_line`/`empty_line`. O texto das linhas não muda, só ganha o prefixo.
- `hosts/hermes/__init__.py`: `_prefetch` registra skip, breaker, falhas de embeddings/Qdrant,
  rodada (com latência), re-rank falho, limpeza; `prefetch` registra falha inesperada.
- `core/daemon.py`: `run` loga start (pid, versão) e exit (motivo); `_run_one` loga cada job
  (repo, tipo, resultado, segundos, erro); erro do watcher é logado quando muda, e a volta ao
  normal também.

### `qctx stats` (`core/stats.py` + `cli/qctx.py`)

`summarize(state_dir) -> dict`: por host, rodadas com/sem memória, p50/p95/máx, skips, falhas
por dependência, breaker; do daemon, jobs por resultado, enfileiramentos, erros do watcher,
último start e versão. O CLI imprime em texto ou `--json`. `stats` entra em
`NOT_FOR_THE_MODEL` com o motivo.

### Versão do daemon

`start()` grava `version` no `daemon.json` (o código do daemon é o mesmo do chamador: `_spawn`
põe a raiz do chamador no `PYTHONPATH`). `repos status` imprime
`daemon: running (pid N, version X)` e, quando X ≠ versão do CLI ou ausente, uma linha de aviso
com o comando para reiniciar.

## Erros e degradação

| Falha | Efeito |
|---|---|
| diretório de estado sem escrita | `eventlog.write` devolve False; recall e daemon seguem |
| log corrompido/binário | `parse` ignora a linha; `stats` conta só o que entende |
| `indexed_sources` falha (Qdrant fora) | igual a hoje: a exceção sai do `watch()`, o `run` segura, agora loga |
| `set_payload` falha no restamp | igual a `add_files`: infra propaga e o job falha, o breaker arma |

## Testes

TDD, RED visto antes de cada GREEN. Principais:
- watcher: 2 ciclos sem mudança = 1 leitura do acervo; releitura quando o job termina, quando o
  registro muda, depois do TTL; `TestTheWatcherDoesNotReindexForever` continua verde.
- restamp: arquivo com `touch` e conteúdo igual sai de `changed_paths` depois de um `refresh`,
  sem nenhum embedding; mesmo para `add_files`; arquivo com conteúdo novo continua sendo
  re-embeddado.
- eventlog: cria 0600, rotaciona, nunca levanta com diretório sem escrita.
- recall_log: `parse(round_line(...))` devolve os campos; linha antiga sem host vira claude-code.
- hermes: uma rodada gera linha `[hermes] round …` no `recall.log`; skip e falha também.
- daemon: job concluído e falho geram linhas; erro repetido do watcher gera uma linha, não N.
- status: versão exibida; aviso quando difere ou falta.
- stats: resumo correto a partir de um log sintético; `--json`.

## Verificação em produção

- CPU/rede do daemon em repouso: medir `rchar` e ticks de CPU por 60 s antes e depois (meta:
  rede em repouso ≈ 0 entre recargas; CPU abaixo de 1%).
- connector-sdk: depois de um refresh, `changed_paths` cai de ~322 para os que realmente mudaram.
- `qctx stats` mostra linhas `[hermes]` depois de um turno no hermes.
- `qctx repos status` mostra `version 1.1.0`.
