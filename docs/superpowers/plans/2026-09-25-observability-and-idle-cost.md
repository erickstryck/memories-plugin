# Observabilidade e custo em repouso: plano de implementação

> **Para agentes:** SUB-SKILL: superpowers:executing-plans (execução inline nesta sessão).
> Passos em checkbox (`- [ ]`).

**Objetivo:** zerar o custo de rede do daemon em repouso, fechar o loop de refresh por mtime, e
dar aos dois hosts e ao daemon um registro legível, com `qctx stats` por cima.

**Arquitetura:** cache por processo no watcher com invalidação por registro/job/TTL; restamp de
metadata sem embedding; dois módulos pequenos de log (`eventlog` escreve, `recall_log` formata)
consumidos pelos dois hosts e pelo daemon; `core/stats.py` agrega.

**Stack:** Python 3.11+ stdlib, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-25-observability-and-idle-cost-design.md`

## Restrições globais

- Stdlib apenas. Nenhuma dependência nova.
- KISS/S.O.L.I.D. conforme a seção "Restrições de engenharia" da spec; módulo novo ≤ ~150 linhas.
- Nenhuma escrita de log levanta exceção; devolve bool.
- Arquivos de estado e logs nascem 0600 (`statefile`).
- Código, comentários, mensagens de commit e nomes de teste em inglês; sem travessão (em dash) nas
  linhas adicionadas.
- Nada que funciona hoje pode degradar: a suíte completa é comparada por nome com a linha de base
  (`TMPDIR=/tmp`, porque sob `~/.hermes/cache/scratch` 4 testes falham por isolamento de HOME,
  pré-existentes e não relacionados).
- Um commit por tarefa; commit e push na `main` direto (o usuário é o dono e pediu).
- Sem `Co-Authored-By`.

Linha de base medida em `fc18389`: `TMPDIR=/tmp python3 -m unittest discover -s tests` → a medir
na Tarefa 0 e registrar os nomes que falham (esperado: nenhum).

---

### Tarefa 0: Linha de base e roteiro de regressão

Primeira porque tudo depois é julgado contra ela.

**Arquivos:** criar `scripts/regression_probe.py` (fora da suíte; lê o acervo real e grava só em
coleções descartáveis).

- [ ] Rodar a suíte com `TMPDIR=/tmp`, salvar a saída em
  `~/.hermes/cache/scratch/baseline-1.0.1.txt` e registrar a contagem e os nomes que falham.
- [ ] Rodar `QCTX_INTEGRATION=1 TMPDIR=/tmp python3 -m unittest tests.test_integration` e
  salvar a saída.
- [ ] Escrever `scripts/regression_probe.py`: para um conjunto fixo de consultas, imprime em JSON
  (a) os ids e a origem que `store.recall` devolve pela política do hook, (b) o bloco que o hook
  emitiria (`hooks/recall.py` como subprocesso com um payload fixo), (c) `repos search` em
  `connector-sdk` (ids dos grupos), (d) `docs list`, (e) `memory list --limit 5` (ids), (f) as
  22 ferramentas do provider hermes via o loader real. Rodar em `fc18389` num worktree e salvar
  em `probe-1.0.1.json`.
- [ ] Commit: `test: a regression probe that records what the live stack answers today`.

### Tarefa 1: O restamp fecha o loop de refresh por mtime (defeito B)

Antes do cache porque, sem ela, o cache ainda deixaria o watcher enfileirar refreshes inúteis.

**Arquivos:** modificar `core/repos.py` (`changed_paths`, `refresh`, `add_files`; novos
`_metadata_drifted`, `_restamp`); teste em `tests/test_repos_refresh.py`.

**Interfaces:** produz `RepoIndex._restamp(path: str, st: os.stat_result) -> int` (chunks
regravados) e `_metadata_drifted(st, md: dict) -> bool` (função de módulo).

- [ ] RED `TestATouchedFileIsRestampedNotReembedded`:
  - `test_after_a_refresh_a_touched_file_leaves_changed_paths`: indexa, `os.utime(path, +100 s)`,
    confere `changed_paths == [path]`, roda `refresh`, confere `changed_paths == []` e que o
    embedder não foi chamado de novo (contar chamadas do `FakeEmbedder`).
  - `test_add_files_over_a_touched_unchanged_file_restamps_it_too`.
  - `test_a_real_content_change_is_still_reembedded` (verde já no RED, prova o escopo).
  - `test_the_report_still_says_ok_for_a_restamped_file`.
- [ ] Rodar: falha em `changed_paths == []`.
- [ ] GREEN: extrair `_metadata_drifted` do corpo de `changed_paths`; em `refresh`, ramo
  `reason is None` e em `add_files`, ramo inalterado: `st = os.stat(path)`; se
  `_metadata_drifted(st, md)`: `self._restamp(path, st)`. `_restamp` faz `scroll_all` com filtro
  `doc_id = doc_id_for(path)`, copia a payload, ajusta `metadata.src_mtime/src_size`,
  `set_payload`.
- [ ] Suíte de `test_repos_refresh`, `test_repos`, `test_watch`, `test_memory_offline` verde.
- [ ] Commit: `fix: a file touched but not changed was refreshed on every other cycle, forever`.

### Tarefa 2: O watcher mantém as fontes em memória (defeito A)

**Arquivos:** `core/repos.py` (`indexed_sources` público, remover `poll` e `indexed_paths` se
ficarem sem chamador fora dos testes); `core/indexer.py` (`watcher`); `tests/test_watch.py`
(`FakeIndex` ganha `indexed_sources` e `changed_paths(repo, sources=None)`; os testes que contam
`indexed_calls` passam a contar `indexed_sources`); `tests/test_repos_refresh.py` (os usos de
`poll` passam a `indexed_sources`).

**Interfaces:** consome `indexed_sources(repo) -> dict[path, md]` e
`changed_paths(repo, sources=dict) -> list[str]`; produz
`indexer.watcher(cfg=None, index=None, clock=time.monotonic)` e `indexer.SOURCES_TTL_S = 300.0`.

- [ ] RED `TestTheWatcherDoesNotRereadAnUnchangedArchive`:
  - `test_two_quiet_cycles_read_the_archive_once`.
  - `test_a_finished_job_forces_a_reread` (muda `state` do job para `done`).
  - `test_a_registry_change_forces_a_reread` (muda `indexed_at` na entry).
  - `test_the_cache_expires_after_the_ttl` (clock injetado).
  - `test_a_repo_that_left_the_registry_is_dropped_from_the_cache`.
- [ ] Ajustar `test_a_cycle_reads_the_archive_exactly_once` e
  `test_a_second_cycle_over_an_untouched_git_index_repeats_neither_the_scan_nor_the_scroll`: o
  segundo ciclo passa a NÃO ler o acervo (essa é a reversão; a docstring diz por quê).
- [ ] Rodar: falha em "2 leituras, esperado 1".
- [ ] GREEN conforme a spec (seção Watcher).
- [ ] Rodar `test_watch`, `test_cli_daemon`, `test_daemon`, `test_repos_refresh`,
  `test_quarantine` verdes; `TestTheWatcherDoesNotReindexForever` em particular.
- [ ] Commit: `perf: the watcher re-read every repository's whole archive every five seconds`.

### Tarefa 3: `core/eventlog.py` e `core/recall_log.py`

**Arquivos:** criar os dois módulos e `tests/test_eventlog.py`; modificar `hooks/recall.py`
(`LOG`, `rotate`, `log`, as duas linhas de rodada); `tests/test_state_modes.py` (rotação passa a
ser `eventlog.rotate`).

**Interfaces:** produz
`eventlog.RECALL`, `eventlog.DAEMON`, `eventlog.MAX_BYTES = 256*1024`,
`eventlog.path(name) -> Path`, `eventlog.rotate(path, max_bytes=MAX_BYTES) -> bool`,
`eventlog.write(name, line) -> bool`;
`recall_log.record(host, msg) -> bool`,
`recall_log.round_line(round_no, full, pointers, relevant, candidates, elapsed, angles, by_rerank, scale_converted, prompt) -> str`,
`recall_log.empty_line(round_no, best_dense, elapsed, angles, outcome, prompt) -> str`,
`recall_log.parse(line) -> dict | None` com chaves `ts, host, kind, elapsed, injected, pointers,
dependency`.

- [ ] RED: `write` cria 0600, anexa com horário, rotaciona pela cauda, devolve False sem levantar
  num diretório 0500; `parse` de cada tipo de linha; linha antiga sem host vira `claude-code`;
  o hook grava `[claude-code] round 1: …`.
- [ ] GREEN.
- [ ] Rodar `test_eventlog`, `test_state_modes`, `test_hygiene_fixes`, `test_recall_block`,
  `test_host_equivalence` verdes.
- [ ] Commit: `refactor: one owner for writing a state log, and one for a recall round's line`.

### Tarefa 4: O hermes registra cada rodada

**Arquivos:** `hosts/hermes/__init__.py` (`prefetch`, `_prefetch`); teste em
`tests/test_hermes_provider.py`.

**Interfaces:** consome `recall_log.*`.

- [ ] RED `TestTheHermesHostLeavesARecord`: rodada com hit → linha `[hermes] round` com
  `injected`; sem hit → linha `0 above the cut`; skip → `skip`; falha do store → `failed`; breaker
  aberto → `re-rank in breaker`; `prefetch` com exceção inesperada → `unexpected failure`; log
  num diretório sem escrita não muda o bloco devolvido.
- [ ] GREEN: medir `time.monotonic()` em volta de `store.recall`; chamar `recall_log.record("hermes", …)`
  nos mesmos pontos do hook.
- [ ] Rodar `test_hermes_provider`, `test_host_equivalence`, `test_hermes_tools` verdes.
- [ ] Commit: `feat: the hermes host now leaves the same recall record the hook does`.

### Tarefa 5: O daemon registra o que faz, e diz sua versão

**Arquivos:** `core/daemon.py` (`start` grava `version`; `run` loga start/exit/erros do watcher;
`_run_one` loga cada job); `core/indexer.py` (loga enfileiramento e recarga); `cli/qctx.py`
(`cmd_repos_status`); testes em `tests/test_daemon.py`, `tests/test_cli_repos.py` ou arquivo
novo `tests/test_daemon_log.py`.

- [ ] RED: job ok/falho gera linha com repo, tipo, resultado, segundos; o mesmo erro do watcher
  em 3 ciclos gera 1 linha, e a recuperação gera 1 linha; `start()` com spawn injetado grava
  `version`; `repos status` imprime a versão e o aviso quando difere ou falta.
- [ ] GREEN.
- [ ] Rodar `test_daemon*`, `test_cli_daemon`, `test_cli_repos`, `test_watch` verdes.
- [ ] Commit: `feat: the daemon keeps a log and records the version it runs`.

### Tarefa 6: `qctx stats`

**Arquivos:** criar `core/stats.py` e `tests/test_stats.py`; `cli/qctx.py` (subcomando);
`tests/test_host_equivalence.py` (`NOT_FOR_THE_MODEL` ganha `stats`, com o motivo);
`docs/usage.md` (seção curta).

**Interfaces:** `stats.summarize(state_dir: Path) -> dict` com `recall: {host: {rounds, hits,
empty, skips, p50, p95, max, failures: {dep: n}, breaker}}` e `daemon: {jobs: {result: n},
enqueued, watcher_errors, last_start, version}`.

- [ ] RED: resumo de um log sintético; percentis; host antigo; linhas lixo ignoradas; CLI em
  texto e `--json`; `qctx stats --help` pelo entry point real.
- [ ] GREEN.
- [ ] Rodar `test_stats`, `test_readme_fidelity`, `test_host_equivalence` verdes.
- [ ] Commit: `feat: qctx stats summarises what both hosts and the daemon recorded`.

### Tarefa 7: Regressão e review

- [ ] Suíte completa com `TMPDIR=/tmp`; comparar nomes de falhas com a linha de base (tem que ser
  o mesmo conjunto; esperado vazio).
- [ ] Integração com `QCTX_INTEGRATION=1`, comparar com a da Tarefa 0.
- [ ] `scripts/regression_probe.py` na árvore nova; `diff` contra `probe-1.0.1.json` (tem que ser
  igual, salvo o prefixo de host nas linhas de log, que o probe não lê).
- [ ] Code review via superpowers:requesting-code-review; tratar achados com
  superpowers:receiving-code-review.

### Tarefa 8: Versão e deploy

- [ ] `core/version.py` e os três manifestos para `1.1.0`; suíte de versão verde.
- [ ] Commit `bump: 1.1.0`, tag `v1.1.0`, push `main` + tag; conferir `origin/main` pelo efeito.
- [ ] Claude: `claude plugin marketplace update memories-plugin`,
  `claude plugin update memories-plugin@memories-plugin`; conferir o SHA e `diff -rq` do cache.
- [ ] Hermes: `hermes plugins update memories`; conferir a revisão.
- [ ] Daemon: `qctx repos daemon stop && qctx repos daemon start`; `qctx repos status` mostra
  `version 1.1.0`.
- [ ] Medir CPU/`rchar` do daemon por 60 s em repouso; `qctx stats`.
