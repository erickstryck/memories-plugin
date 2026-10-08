# Contexto do modelo descoberto pelo host, e limites do guard no config: plano

> **Para agentes:** SUB-SKILL: superpowers:executing-plans (execução inline nesta sessão).
> Passos em checkbox (`- [ ]`).

**Objetivo:** o guard de arquivos grandes passa a usar o tamanho de contexto que o próprio host
informa para o modelo selecionado agora, nos dois hosts, com o `context_window` do config como
último recurso; e os dois limites do guard passam a ser ajustáveis por `qctx config set`.

**Arquitetura:** cada host PUBLICA um registro por sessão (`core/hostwindow.py`); o guard só LÊ.
No claude-code quem publica é a statusLine (`qctx statusline`), a única fonte que recebe o
tamanho; no hermes é o provider, num hook `pre_llm_call`, com a função do próprio hermes.
`core/windows.py` é o único dono da ordem: registro do host, cache do endpoint, config, 0. A
tabela de modelos sai.

**Stack:** Python 3.11+ stdlib, `unittest`, bash + jq no `scripts/cutover.sh`.

**Spec:** `docs/superpowers/specs/2026-10-01-dynamic-context-window-design.md`

## Restrições globais

- Stdlib apenas. Nenhuma dependência nova.
- KISS e S.O.L.I.D.: um módulo por responsabilidade (`hostwindow` persiste, `statusline`
  interpreta o payload do claude-code, `hosts/hermes/window.py` resolve no hermes, `windows`
  ordena). Módulo novo com até ~150 linhas.
- O guard, a statusLine e o hook do hermes NUNCA levantam exceção e nunca escrevem no stdout
  além do seu contrato. Na dúvida, liberam a leitura.
- Arquivos de estado nascem 0600 (`core.statefile`).
- O callback de `pre_llm_call` devolve `None`: o hermes injeta no prompt o que um callback
  desses devolve.
- Código, comentários, mensagens de commit e nomes de teste em inglês; nenhum travessão longo
  nas linhas adicionadas (o caractere U+2014; o grep do diff tem de dar 0).
- Nada que funciona na v1.2.0 degrada. As únicas mudanças de comportamento aceitas são as da
  spec: sem registro e sem config, `claude-opus-5` deixa de valer 1.000.000 e passa a liberar;
  e os relatórios dos cutovers mudam de texto.
- Ambiente dos testes: `TMPDIR=/tmp`, `unset QCTX_STATE_DIR`, e o `HOME` real do usuário
  (um revisor já deixou o do shell trocado).
- Um commit por tarefa, na `main`. Sem `Co-Authored-By`.

---

### Tarefa 0: Linha de base da v1.2.0

**Arquivos:** nenhum no repo; saídas em `~/.hermes/cache/scratch/`.

- [ ] `git worktree add /tmp/base-4f018fc 4f018fc`.
- [ ] Suíte nas duas árvores (`TMPDIR=/tmp python3 -m unittest discover -s tests`), salvar em
  `baseline-1.2.0.txt` e registrar contagem e nomes que falham (esperado: nenhum).
- [ ] Mesma suíte no cenário hostil (`env -u QCTX_CONFIG XDG_CONFIG_HOME=$H`, com
  `$H/mnemosine/config.json` contendo `{not json`): esperado só
  `test_the_qdrant_budget_knob_has_the_same_semantics_in_both_hosts`.
- [ ] Integração (`QCTX_INTEGRATION=1`) e `scripts/regression_probe.py --root /tmp/base-4f018fc`,
  salvar em `probe-1.2.0.json`.

### Tarefa 1: Os limites do guard viram configuração

Primeiro porque é independente e porque as tarefas seguintes mexem nos mesmos adaptadores.

**Arquivos:** `core/config.py`, `cli/qctx.py` (`cmd_config_set`), `hosts/hermes/__init__.py`
(schema das ferramentas de config, ~:784), `core/install.py` (`OPTIONAL_FIELDS`),
`hooks/bigfile.py`, `hosts/hermes/bigfile.py`, `core/knobs.py` (comentário),
`docs/usage.md`; testes em `tests/test_config_numbers.py`, `tests/test_bigfile_claude.py`,
`tests/test_bigfile_hermes.py`, `tests/test_cli_install.py`, `tests/test_hermes_tools.py`.

**Interfaces:** produz `Config.bigfile_floor_pct: float = 0.20`,
`Config.bigfile_share_pct: float = 0.40` e `core.config.fraction_fields() -> tuple`. Os
valores padrão vêm de `core.bigfile.FLOOR_PCT` e `SHARE_PCT`, um dono só para os números
(conferir que `core.bigfile` não importa `core.config`, para não criar ciclo).
`ENV_ALIASES`: `("QCTX_BIGFILE_FLOOR_PCT", "BIGFILE_FLOOR_PCT")` e
`("QCTX_BIGFILE_SHARE_PCT", "BIGFILE_SHARE_PCT")`.

- [ ] RED `TestTheGuardThresholdsAreConfigSettings` em `tests/test_config_numbers.py`:
  - `test_the_defaults_are_the_ones_the_guard_had`: env vazio e sem arquivo dão 0.20 e 0.40.
  - `test_the_file_sets_them` (0.15 no arquivo) e `test_the_environment_beats_the_file`.
  - `test_the_legacy_names_still_work` e `test_a_blank_canonical_name_falls_through`.
  - `test_a_non_number_falls_back_and_says_where` (`"banana"`, com `note` citando a fonte).
  - `test_outside_zero_to_one_falls_back_and_says_so` (`"1.5"`, `"-0.1"`).
  - `test_the_bounds_zero_and_one_are_kept` (os extremos têm significado em `_blocks`).
  - `test_they_are_fractions_not_whole_numbers`: estão em `fraction_fields()`, não em
    `numeric_fields()`.
- [ ] RED no CLI: `config set bigfile-floor-pct banana` e `1.5` são recusados sem tocar o
  arquivo; `0.15` grava o float. O mesmo formato de recusa que o campo inteiro já usa.
- [ ] RED nos adaptadores: com `QCTX_CONFIG` apontando um arquivo com
  `bigfile_floor_pct: 0.001`, o caso `FLOOR_ONLY` libera, nos dois hosts. Os testes que
  comparam os kwargs de `decide` com `adapter.FLOOR_PCT` passam a comparar com o config.
- [ ] RED no hermes: o schema das ferramentas de config tipa os dois campos como `"number"`.
- [ ] GREEN: em `load`, depois do laço dos inteiros, um laço sobre `_FRACTION_FIELDS`
  (derivado do dataclass por `float`): `float()` tolerante, faixa `0 <= v <= 1`, senão o
  padrão e um `note` (`"{name}={value!r}{where} is not a fraction between 0 and 1, using
  {default}"`). `cmd_config_set` recusa pelo mesmo critério. Os adaptadores apagam as
  constantes `FLOOR_PCT`/`SHARE_PCT` lidas no import e passam `cfg.bigfile_floor_pct` e
  `cfg.bigfile_share_pct` a `decide`. O wizard ganha os dois campos (ajustar as contagens de
  prompts em `tests/test_cli_install.py`).
- [ ] Docs: duas linhas na tabela config ↔ env de `docs/usage.md` e um parágrafo na seção do
  guard.
- [ ] Mutações: tirar a checagem de faixa, tirar o `float()` tolerante, voltar um adaptador
  para a constante do import. Cada uma tem de deixar um teste vermelho.
- [ ] Commit: `feat: the big-file guard thresholds are config settings both hosts read`.

### Tarefa 2: `core/hostwindow.py`, o registro que o host publica

**Arquivos:** criar `core/hostwindow.py`; modificar `core/session_state.py`
(`SESSION_FILE_PATTERNS`); teste em `tests/test_hostwindow.py`.

**Interfaces:**
- `HostWindow(NamedTuple)`: `model: str`, `window: int`, `source: str`, `guess: bool`,
  `at: float`.
- `publish(session_id: str, model: str, window: int, source: str, guess: bool = False) -> bool`:
  grava `<state_dir>/window-<names.safe(session_id)>.json` por `statefile.write_json`. Devolve
  `False`, sem gravar, para `session_id` vazio ou `window` que não seja inteiro positivo.
  Nunca levanta.
- `read(session_id: str) -> HostWindow | None`: `None` para ausente, ilegível ou com tipos
  errados. Nunca levanta.
- `newest(source: str) -> tuple[str, HostWindow] | None`: o registro mais recente daquela
  fonte, para o `qctx setup`.
- `PATTERN = "window-*.json"`, que entra em `SESSION_FILE_PATTERNS`.

- [ ] RED: ida e volta; arquivo 0600; sessão vazia e janela 0, negativa ou `"1M"` não gravam;
  JSON corrompido e `window: "x"` leem `None`; um id com `../` fica dentro do state dir;
  `purge_dead` apaga um `window-*.json` velho; `newest` escolhe pelo `at` e filtra pela fonte.
- [ ] GREEN e mutações (tirar a validação de `window`, tirar o `names.safe`, tirar o
  padrão de `SESSION_FILE_PATTERNS`).
- [ ] Commit: `feat: a per-session record of the context window a host reported`.

### Tarefa 3: `core/windows.py` sem tabela, com o registro do host primeiro

**Arquivos:** `core/windows.py`, `hooks/bigfile.py`, `hosts/hermes/bigfile.py`,
`core/bigfile.py` (comentário de ~:236); testes em `tests/test_windows.py`,
`tests/test_bigfile_claude.py`, `tests/test_bigfile_hermes.py`.

**Interfaces:** consome `hostwindow.read`. Produz
`window_for(model: str, cfg, endpoint: str = "", session_id: str = "") -> int`, na ordem:
(1) `hostwindow.read(session_id)` quando ele existe, não é palpite e tem `window > 0`;
(2) `windowcache.get(endpoint, model)` quando há endpoint; (3) `cfg.context_window > 0`;
(4) `0`. `MODEL_WINDOWS` é apagada. O docstring diz por que o config é o último e por que um
palpite não conta.

- [ ] RED `TestTheOrder` (substitui `TestWindowFor`, `TestTheTableHolds...` e `TestTheCascade`):
  - `test_the_host_record_beats_the_config` e `..._beats_the_endpoint_cache`;
  - `test_a_guess_is_skipped_for_the_cache_then_the_config_then_zero`;
  - `test_the_config_answers_when_the_host_published_nothing`;
  - `test_a_record_of_another_session_is_not_used`;
  - `test_a_model_name_alone_resolves_to_zero` (`claude-opus-5`, sem registro nem config);
  - manter `test_a_cached_value_is_used_even_when_STALE` e o teste sem rede.
- [ ] RED nos adaptadores: com um registro de 200.000 para a sessão do payload e um uso de
  150.000, um arquivo que custa ~40K tokens é negado; com um registro de 1.000.000, liberado;
  sem registro e sem config, liberado. Nos dois hosts, pelo `session_id` que o payload já
  traz.
- [ ] GREEN: os adaptadores passam `session_id=` a `window_for`.
- [ ] Mutações: inverter a ordem registro/config; aceitar palpite; ignorar o `session_id`.
- [ ] Commit: `feat: the guard takes the window the host reported, and the config last`.

### Tarefa 4: A ponte pela statusLine (claude-code)

**Arquivos:** criar `core/statusline.py` e `tests/fixtures/statusline-2.1.282.json` (o payload
real de `wm-evidence/statusline.log`, com caminhos trocados por `/tmp/x` e um `session_id`
fictício); modificar `cli/qctx.py` (subcomando e despacho antecipado); teste em
`tests/test_statusline.py`.

**Interfaces:**
- `render(payload: dict) -> str`: `ctx 23% · 1M`; sem `used_percentage`, `ctx · 1M`; sem
  janela, `ctx`. A janela em `k`/`M` (`200k`, `1M`, `1.5M`).
- `publish_from(payload: dict) -> bool`: `hostwindow.publish(session_id, model.id,
  context_window.context_window_size, source="claude-code")`.
- `main(stdin, stdout) -> int`: lê, publica, imprime uma linha, devolve 0. Nunca levanta.
- `install(settings: Path, command: str, apply: bool) -> tuple[str, str]`, com estado
  `installed`, `added`, `missing`, `foreign` ou `unreadable`. Só grava com `apply` e estado
  `missing`; grava num temporário no mesmo diretório e faz `os.replace`, preservando o modo
  do arquivo e todas as outras chaves.
- `is_ours(command: str) -> bool`: comando que termina em `qctx statusline`.
- CLI: `qctx statusline` (runtime) e `qctx statusline install [--apply] [--settings PATH]`.
  O runtime é despachado em `main()` ANTES de `core.load()`: um config corrompido não pode
  apagar a statusLine. O comando gravado é o caminho absoluto do lançador
  (`core.install.target_dir(env) / LAUNCHER_NAME`) seguido de ` statusline`.

- [ ] RED `TestTheStatusLinePublishesWhatClaudeCodeReported`: com a fixture, o registro tem
  `claude-opus-5-5[1m]` e 1.000.000; payload sem `context_window` não publica e ainda imprime;
  stdin lixo imprime e sai 0; `qctx statusline` como subprocesso com `QCTX_CONFIG` apontando
  `{not json` ainda publica e imprime.
- [ ] RED `TestInstallingTheStatusLine`: `missing` em dry run deixa o arquivo byte a byte
  igual; `apply` acrescenta só `statusLine` e preserva o resto e o modo; o nosso presente dá
  `installed`; um alheio dá `foreign` e fica intocado; JSON inválido dá `unreadable` e fica
  intocado.
- [ ] RED ponta a ponta: `qctx statusline` real com um payload de 200.000 para a sessão S,
  depois `hooks/bigfile.py` real para S, com um transcript de 150.000 usados: nega. Com o
  payload de 1.000.000: libera.
- [ ] GREEN, mutações (despacho depois do `load`; `install` gravando em dry run; `is_ours`
  aceitando qualquer comando) e medição: `qctx statusline < fixture` abaixo de 150 ms.
- [ ] Commit: `feat: a statusLine that hands claude-code's context window to the guard`.

### Tarefa 5: O cutover do claude-code instala a statusLine

**Arquivos:** `scripts/cutover.sh`; teste em `tests/test_cli_install.py` (o harness de
`hermetic_env(home, CUTOVER_SKIP_SUITE="1")` que já roda o script, ~:209).

- [ ] RED: o dry run relata a linha da statusLine (`..` com o comando que vai gravar, ou `ok`
  quando já está instalada, ou `..` dizendo que a alheia fica) e deixa o `settings.json`
  byte a byte igual.
- [ ] GREEN: na parte de checagens, `python3 "$ROOT/cli/qctx.py" statusline install
  --settings "$SETTINGS"`; na parte do `--apply`, depois do backup datado que o script já
  faz, o mesmo com `--apply`. O caminho do `--apply` fica coberto pelos testes de `install`
  da Tarefa 4, porque o script recusa `--apply` com a suíte pulada.
- [ ] Commit: `feat: the claude-code cutover installs the statusLine`.

### Tarefa 6: O provider do hermes publica o tamanho a cada troca de modelo

**Arquivos:** criar `hosts/hermes/window.py`; modificar `hosts/hermes/__init__.py`
(`register`), `scripts/hermes_cutover.sh` (seção da janela, ~:689-760); testes em
`tests/test_hermes_window.py` e `tests/test_hermes_cutover.py` (os três testes da tabela,
~:1132-1150).

**Interfaces:**
- `on_pre_llm_call(session_id: str = "", model: str = "", **_) -> None`. Lê
  `model_config` e `billing_provider` da sessão por `hosts.hermes.bigfile._rows` e
  `state_db_path` (só leitura, timeout curto) e obtém a rota com o
  `SessionDB.session_gateway_runtime` do hermes. Sem rota registrada, usa a do config,
  mas só para o modelo do config. Se `(model, provider, base_url)` é igual ao último que
  resolveu para aquela sessão, não faz nada. Senão, chama
  `agent.model_metadata.get_model_context_length(model, base_url=..., api_key=<a chave da
  entrada, nas rotas custom; vazia nos provedores conhecidos>, provider=...,
  config_context_length=<model.context_length do config do hermes, ou None>,
  custom_providers=get_compatible_custom_providers(load_config()))` e publica com
  `source="hermes"`. Contam como palpite o `DEFAULT_FALLBACK_CONTEXT` e a resposta de uma
  rota custom que declara chave mas não a tem; um valor fixado nunca é palpite. Nunca
  levanta e devolve `None`. Revisado em 2026-10-02 por medição e por decisão do usuário:
  com `api_key` vazia, a rota Eukrio dava 131.072 em vez de 524.288, e os `billing_*` não
  acompanham o `/model`.
- `register(ctx)` chama `_register_window_hook(ctx)`: `getattr(ctx, "register_hook", None)`;
  sem ele, nada; com falha, um `_note`.

- [ ] RED com um módulo falso `agent.model_metadata` em `sys.modules` e um `state.db` de
  fixture com as colunas `model_config` e `billing_provider`: o primeiro turno publica 1.000.000 sem palpite; o
  mesmo modelo de novo não chama a função; outro modelo chama e republica; o fallback
  publica com `guess=True`; a função levantando não publica nem propaga; sem o módulo do
  hermes, nada acontece; os kwargs passados são os da interface; o callback devolve `None`;
  `register` registra `pre_llm_call` num ctx que tem `register_hook` e não quebra num que
  não tem.
- [ ] RED ponta a ponta: o provider publica para a sessão S e o `hosts/hermes/bigfile.py`
  real, como subprocesso, decide com esse tamanho.
- [ ] RED no cutover: o relatório deixa de citar teto por nome de modelo e passa a dizer que
  o provider publica o tamanho por sessão e que o `context_window` só vale quando o hermes
  não sabe.
- [ ] GREEN e mutações (memo ignorado, palpite marcado como certo, chave repassada a um
  provedor conhecido, rota lida do `billing_provider`, chave ausente fora do palpite).
- [ ] Commit: `feat: the hermes provider publishes hermes' own context window per session`.

### Tarefa 7: Visibilidade e documentação

**Arquivos:** `core/setup.py` (`_check_context_window`), `docs/usage.md`; teste em
`tests/test_setup_context_window.py`.

- [ ] RED: com a statusLine instalada, `ok` e o último registro do claude-code (modelo,
  tamanho, idade); com `context_window` declarado e sem statusLine, `ok` dizendo que é o
  config; sem nenhum dos dois, aviso que nomeia `qctx install --apply` e
  `config set context-window`; o último registro do hermes aparece quando existe.
- [ ] GREEN.
- [ ] `docs/usage.md`: seção "How the guard learns the context window": a ordem, a statusLine,
  o hermes, `claude -p`, o subagente e o config por último.
- [ ] Commit: `feat: setup says where each host's context window comes from`.

### Tarefa 8: Regressão, review, release e deploy

- [ ] Suíte, cenário hostil, integração e probe na árvore nova e em `/tmp/base-4f018fc`, no
  mesmo instante; comparar as falhas por nome. As diferenças aceitas são as das Restrições.
- [ ] Ponta a ponta nas cópias instaladas depois do deploy, com state dir descartável: a
  statusLine real publica e o guard real lê, nos dois hosts.
- [ ] Code review com dois revisores em clones descartáveis (`/tmp/review3/{a,b}`):
  correção e design; compatibilidade com a v1.2.0. Corrigir, com teste vermelho antes.
- [ ] Bump 1.3.0 nos cinco arquivos de versão, commit `bump: 1.3.0`, tag `v1.3.0`, push.
- [ ] Deploy: plugin do claude-code, plugin do hermes (comparar os achados do scanner antes
  de qualquer contorno), reiniciar o daemon e conferir `version 1.3.0`.
- [ ] PERGUNTAR ao usuário antes de rodar `qctx install --apply` na máquina dele: grava a
  statusLine no `~/.claude/settings.json`.
- [ ] Limpar worktree e clones; atualizar a memória `3cdb8139`.
