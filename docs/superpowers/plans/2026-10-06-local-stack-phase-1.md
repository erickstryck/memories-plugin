# Stack local, fase 1: plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** o `qctx install` passa a subir sozinho o Qdrant e os dois llama-servers (embedding e
rerank) em containers, no Linux e no macOS, com Docker ou Podman, e o grupo `qctx stack
status|up|down|remove` cuida da stack depois.

**Architecture:** pacote novo `stack/`, só stdlib, com dependência num sentido só (`cli -> stack
-> core`). Módulos puros (catálogo, backends, compose, fetch, health, verify, state) atrás de
Protocols pequenos; dois casos de uso (`installer`, `lifecycle`) que falam só com Protocols; e
`stack/cli.py` como raiz de composição, que injeta subprocess, HTTP e terminal. O `cmd_install`
ganha uma etapa entre o launcher e a configuração.

**Tech Stack:** Python 3 stdlib (`subprocess`, `urllib`, `hashlib`, `socket`, `json`, `ast` nos
testes), `unittest`; Docker ou Podman com um provider de compose em tempo de execução.

**Spec:** `docs/superpowers/specs/2026-10-05-local-stack-design.md` (aprovada em `af124e8`,
revista em 2026-10-06 pelas verificações de abertura). Esta é a fase 1 da decisão 13.

## O que as verificações de abertura mediram

A spec manda o plano de cada fase começar pelas verificações dela. As da fase 1 rodaram antes
deste plano, numa máquina Linux com Podman 5.7.0 rootless, docker-compose v5.2.0 como provider
externo do Podman, podman-compose 1.6.0, duas Intel Arc B70 e uma RX 6900 XT, e o resultado está
na seção final da spec. Este plano já parte delas. As que mudaram o desenho:

| # | medição | consequência neste plano |
|---|---|---|
| M1 | um `containers.conf` com `[containers] devices` injeta `/dev/dri` em todo container; o perfil `cpu` sem `-dev` rodou na GPU (rerank 1,18 s, 204 MB) e com `-dev none` na CPU (7,06 s, 2,8 GB) | `server_command(role, None)` termina em `-dev none` (Task 2) |
| M2 | `api_base_url` sem `/v1` faz o plugin chamar `/embeddings`, que na `b11382` devolve lista crua e quebra o `Embedder` com `AttributeError` | `stack_urls` grava `/v1` e esvazia `embed_url` (Task 8) |
| M3 | o `podman info` disse `remoteSocket.exists: true` sem socket nenhum, e o `podman compose` (docker-compose) falhou | o socket é conferido por conexão; com ele parado e o `podman-compose` instalado, usa-se o `podman-compose` (Task 3) |
| M4 | o `podman-compose run` aloca TTY e devolve `\r\n` | a prova roda `compose run -T`, e o parser tolera `\r` (Tasks 5 e 10) |
| M5 | a linha `uma:` não sai com `--list-devices` | o padrão do menu é só a maior memória livre (Task 5) |
| M6 | os providers prefixam volume e nome de container com o projeto, cada um do seu jeito | `container_name: <projeto>-<serviço>` em todo serviço; o volume real é `<projeto>_memories-plugin-qdrant` (Task 6) |
| M7 | a primeira chamada custa 3 a 7 vezes a seguinte | a calibração aquece antes de medir (Task 8) |
| M8 | `--no-webui` é deprecado na `b11382` | `--no-ui` (Task 2) |

Medidas de referência, para os textos e os testes de integração: CPU com todas as threads, embed
de 6000 caracteres 0,97 s e rerank de 20 x 2400 7,0 s, com ~1,83 GB e ~2,78 GB de RSS; Intel B70
por Vulkan, 0,21 s e 1,07 s, com ~200 MB de RSS por container.

## Global Constraints

- Só stdlib. Nenhuma dependência nova em lugar nenhum, `plugin.yaml` incluído.
- `stack/` importa só `core` e stdlib. `core/`, `hooks/` e `hosts/` nunca importam `stack`;
  `stack` nunca importa `hooks`, `hosts` nem `cli`.
- Nada muda em `core/`, `hooks/`, `hosts/`, `scripts/` nem `bin/`.
- `StackError` herda de `core.errors.CoreError`, com `step` e `fix`.
- Todo subprocess tem timeout: 30 s para `info`/`version`, 60 s para `compose config`, 180 s
  para a prova de devices, 1800 s para `compose pull`, 600 s para `compose up -d`, 300 s para
  `down`, 60 s para `logs`.
- Imagens: `ghcr.io/ggml-org/llama.cpp:server-vulkan-b11382@sha256:431561ee79ee67b3980a02ff47ed9dc19496127643b75671d9789ef693ca57f9`
  e `docker.io/qdrant/qdrant:v1.19.2-unprivileged@sha256:efb96a9425a90d2d5a1a0a474156280df1892dcdf1af3e8515bd2589b1bfd88b`.
- Modelos: `gpustack/bge-m3-GGUF` @ `2d48f1737679ad900d5c26c5aad5410e9c70fdca`,
  `bge-m3-Q4_K_M.gguf`, 437778496 bytes, sha256
  `6d39681b26c61279ac1f82db35a04a05009e94c415b51c858ff571489a82fc06`, MIT;
  `gpustack/bge-reranker-v2-m3-GGUF` @ `3093af03b1a635e67b084b1d8c03c5f5e020fd05`,
  `bge-reranker-v2-m3-Q4_K_M.gguf`, 438376864 bytes, sha256
  `e186a244ed455b4ab66ec64339ce7427a6ae13f5c0b5e544de96e50f0f8b3673`, Apache-2.0. URL
  `https://huggingface.co/<repo>/resolve/<revisão>/<arquivo>`.
- Comando dos servidores: `-m /models/<arquivo> --host 0.0.0.0 --port 8080 --embedding|--reranking
  -c 8192 -b 8192 -ub 8192 --no-ui -dev <Vulkan<n>|none>`.
- Portas publicadas só em `127.0.0.1`: 6333 (Qdrant), 8003 (embed), 8004 (rerank); ocupada, a
  primeira livre a partir de `porta + 10000`.
- Diretório: `$QCTX_STACK_DIR`, senão `${XDG_DATA_HOME:-~/.local/share}/memories-plugin/stack`, com
  `models/`, `compose.yaml` e `stack.json`.
- Projeto compose `memories-plugin`; serviços `qdrant`, `embed`, `rerank`; volume
  `memories-plugin-qdrant`; `container_name` `<projeto>-<serviço>`.
- Comum a todo serviço: `restart: always`, `cap_drop: [ALL]`,
  `security_opt: [no-new-privileges:true]`; modelos `:ro` (`:ro,z` com SELinux ativo); no Qdrant,
  `QDRANT__TELEMETRY_DISABLED=true`.
- Config gravado: `qdrant_url = http://127.0.0.1:<q>`, `api_base_url = http://127.0.0.1:<e>/v1`,
  `embed_url = ""`, `rerank_url = http://127.0.0.1:<r>/v1/rerank`, `vector_size` detectado; nunca
  chave; só depois da verificação; diff antes, e `y/N` se um valor não vazio for substituído.
- `--yes` sozinho não provisiona. `--check` e `--json` nunca escrevem e nunca chamam runtime.
- Windows (inclusive WSL): a etapa não é oferecida na fase 1; diz isso e aponta `## Local models`
  do README.
- Toda saída de terminal em inglês, no estilo do wizard (`  ok    …`, `  ..    …`, `  FAIL  …`).
  Código, comentários, nomes de teste e mensagens de commit em inglês. Sem `Co-Authored-By`.
- Docs: nenhum travessão longo, `http://` só para localhost, nenhum download encadeado num
  shell ou num `python` por pipe, nenhum literal com forma de segredo, nenhum caminho do HOME real (o placeholder
  do repo é `/home/me`).
- Testes herméticos (`tests/isolation.py`), sem rede, sem runtime real; a integração com runtime
  só com `QCTX_STACK_IT=1`.
- Suíte: `TMPDIR=/tmp python3 -m unittest discover -s tests`. Linha de base em `af124e8` nesta
  máquina: 1931 testes, OK, 25 pulados. Com `TMPDIR` dentro do HOME, quatro testes falham por
  causa do ambiente, não do código: os três `test_the_environment_it_runs_in_is_assembled_not_inherited`
  e o `test_daemon_claim_gap...test_the_unlink_does_not_remove_a_file_that_replaced_the_one_it_judged`.

## Engenharia: o que KISS e SOLID pedem aqui

- **Molde de tamanho:** `core/jobs.py` (258 linhas) e `core/lease.py` (194). Módulo de `stack/`
  acima de ~300 linhas ganhou uma responsabilidade que não é dele; o `installer` é o único que
  pode passar disso, e então se divide em funções privadas por etapa, não em mais um módulo.
- **S:** os casos de uso não fazem `print` nem `input`; falam com `Prompter` e `Reporter`. O
  `render` é puro.
- **O:** um backend novo é uma classe e uma entrada em `BACKENDS`. Nem o `installer` nem o
  `render` mudam.
- **L:** `Docker`, `Podman` e o `FakeRuntime` dos testes passam pelo mesmo teste de contrato. O
  fake recusa o que o real recusa: provider ausente, `compose` que falha, socket parado.
- **I:** quem baixa não conhece compose; quem renderiza não conhece subprocess.
- **D:** `installer` e `lifecycle` recebem tudo por `Deps`; só `stack/cli.py` instancia as
  implementações concretas.
- **Simples não pode custar completo:** a cobertura da spec é provada por teste mecânico onde
  existe forma (a matriz de compatibilidade deriva de `runtimes()`, as fixtures se comparam com o
  `render`, os orçamentos se leem por AST), nunca por "eu conferi".

## Review Focus

1. **Engine que injeta GPU em todo container** (o `containers.conf` de M1): o perfil `cpu` tem de
   continuar CPU. Testes: `test_cpu_service_is_pinned_to_no_device` (Task 6) e
   `test_cpu_plan_pins_no_device_even_when_the_engine_injects_gpus` (Task 10).
2. **Config e ambiente que já apontam para outro lugar:** um `embed_url` antigo no arquivo vence o
   `api_base_url`, e `QDRANT_URL`/`SERVER_BASE_URL` no rc vencem o arquivo. Testes:
   `test_an_existing_embed_url_is_cleared_by_the_patch` (Task 10) e os de `env_overrides`
   (Task 8).
3. **Instalação interrompida:** `.part` deixado por um Ctrl-C e `stack.json` em fase `compose`.
   Rodar de novo retoma o download e não duplica containers. Testes:
   `test_a_part_file_resumes_from_its_size` (Task 7) e
   `test_an_interrupted_install_provisions_again` (Task 12).
4. **`stack.json` corrompido:** o `status` tem de dizer isso com a correção, e o `remove` tem de
   conseguir limpar. Testes: `test_a_corrupt_file_names_itself_and_the_remove_fix` (Task 9) e
   `test_remove_cleans_even_with_a_corrupt_state` (Task 11).
5. **Portas:** só uma das três ocupada, e a candidata `+10000` também ocupada. Testes:
   `test_only_the_busy_port_moves` e `test_the_next_free_port_is_taken_when_plus_10000_is_busy`
   (Task 10).

---

### Task 1: o pacote, o erro e as fronteiras

Primeiro porque todo módulo seguinte levanta `StackError` e mora sob as regras de importação que
esta tarefa trava.

**Files:**
- Create: `stack/__init__.py`
- Create: `tests/test_stack_boundaries.py`
- Modify: `tests/test_core_is_portable.py` (`FORBIDDEN`)
- Modify: `tests/test_installable_from_git.py` (classe nova no fim)

**Interfaces:**
- Produces: `stack.StackError(message: str, *, step: str, fix: str | None = None)`, subclasse de
  `core.errors.CoreError`, com `.step` e `.fix`; `str(exc) == f"{step}: {message}"`, mais
  `f" (fix: {fix})"` quando há `fix`.

- [ ] **Step 1: testes que falham**

`tests/test_stack_boundaries.py`, reusando `imported_packages` de `tests.test_core_is_portable`:

```python
def test_no_core_hooks_or_hosts_module_imports_stack(self):
    offenders = {str(p.relative_to(REPO)): "stack"
                 for d in ("core", "hooks", "hosts") for p in (REPO / d).rglob("*.py")
                 if "stack" in imported_packages(p)}
    self.assertEqual(offenders, {})

def test_stack_imports_no_host_and_no_cli(self):
    bad = {p.name: sorted(imported_packages(p) & {"hooks", "hosts", "cli", "agent"})
           for p in (REPO / "stack").glob("*.py")}
    self.assertEqual({k: v for k, v in bad.items() if v}, {})

def test_stack_imports_only_the_stdlib_core_and_itself(self):   # a restrição "só stdlib"
    allowed = set(sys.stdlib_module_names) | {"core", "stack"}
    extra = {p.name: sorted(imported_packages(p) - allowed) for p in (REPO / "stack").glob("*.py")}
    self.assertEqual({k: v for k, v in extra.items() if v}, {})

def test_the_walk_saw_the_stack_package(self):
    self.assertIn("__init__.py", [p.name for p in (REPO / "stack").glob("*.py")])

def test_stack_error_is_a_core_error_and_names_step_and_fix(self):
    exc = StackError("no runtime", step="runtime", fix="install Docker or Podman")
    self.assertIsInstance(exc, CoreError)
    self.assertEqual(str(exc), "runtime: no runtime (fix: install Docker or Podman)")
    self.assertEqual(str(StackError("x", step="s")), "s: x")
```

`tests/test_installable_from_git.py`, classe `TestNoTrackedFilePipesADownloadIntoAnInterpreter`:
`PIPE_TO_INTERPRETER = re.compile(r"\b(?:curl|wget)[ \t][^\n]*\|\s*(?:sudo\s+)?(?:(?:ba|z|da|k)?sh\b|python)", re.IGNORECASE)`;
`test_no_tracked_file_pipes_a_download_into_an_interpreter` varre `tracked_text_files()`;
`test_the_pattern_catches_the_shape` monta cada linha ofensiva por concatenação em tempo de
execução (`"cu" + "rl -fsSL https://x.example/i" + " | " + "sh"`), para o arquivo de teste não
carregar o padrão que o scanner do hermes classifica: curl em `sh`, wget em `bash`, curl em
`python3 -m json.tool` e curl em `tee x` e depois `sh`; e não casa curl em `sha256sum -c` nem em
`jq .`. Por que essa forma (medido no `plugin_guard` instalado em 2026-10-06, revisão da Task 1):
um curl encadeado em `python` é `critical` e BLOQUEIA o install; em shell é `high`, em qualquer
estágio do pipe e também em `ksh`; e o scanner ignora caixa e casa `python` como prefixo
(`pythonw`, `Python3`). A primeira versão desta regex, só
shell e só o primeiro estágio, deixava passar a forma que bloqueia. O `[ \t]` depois do nome da
ferramenta, como o `curl\s+` do scanner, impede a regex de casar o próprio texto: com `\b` no
lugar, a alternância casava a linha que a declara, neste plano e no teste (achado do fix round 1).

Em `tests/test_core_is_portable.py`: `FORBIDDEN = {"hooks", "hosts", "cli", "agent", "stack"}`.

- [ ] **Step 2:** `TMPDIR=/tmp python3 -m unittest tests.test_stack_boundaries -v`. Esperado:
  ERROR, `No module named 'stack'`.
- [ ] **Step 3:** `stack/__init__.py` com o docstring do pacote (fronteiras, por que fora de
  `core/`) e `StackError`.
- [ ] **Step 4:** os dois módulos de teste e `tests.test_core_is_portable` passam.
- [ ] **Step 5:** commit `feat(stack): package root, StackError and the import boundaries`.

### Task 2: o catálogo

**Files:**
- Create: `stack/catalog.py`
- Test: `tests/test_stack_catalog.py`

**Interfaces:**
- Consumes: `StackError`.
- Produces:
  - `LLAMA_IMAGE: str`, `QDRANT_IMAGE: str`, `IMAGES: dict[str, str]` (`"llama"`, `"qdrant"`),
    `IMAGE_ENV = {"llama": "QCTX_STACK_IMAGE_LLAMA", "qdrant": "QCTX_STACK_IMAGE_QDRANT"}`;
  - `@dataclass(frozen=True) class Model: role: str; repo: str; revision: str; filename: str;
    size: int; sha256: str; license: str` com `url() -> str`; `EMBED_MODEL`, `RERANK_MODEL`,
    `MODELS = (EMBED_MODEL, RERANK_MODEL)`, `MODELS_BYTES: int`;
  - `EMBED_DIM = 1024`, `CONTEXT = 8192`;
  - `PORTS = {"qdrant": 6333, "embed": 8003, "rerank": 8004}`,
    `CONTAINER_PORTS = {"qdrant": 6333, "embed": 8080, "rerank": 8080}`,
    `PORT_FALLBACK_OFFSET = 10000`;
  - `PROJECT = "memories-plugin"`, `SERVICES = ("qdrant", "embed", "rerank")`,
    `VOLUME = "memories-plugin-qdrant"`;
  - `server_command(role: str, device: str | None) -> list[str]`;
  - `qdrant_version(ref: str) -> str`;
  - `parse_image_flags(values: list[str]) -> dict[str, str]`;
  - `resolve_images(flags: dict[str, str], env: Mapping[str, str], base: dict[str, str] | None =
    None) -> dict[str, str]`.

- [ ] **Step 1: testes que falham.** O valor fica fixado de forma absoluta, numa tabela, para que
  mudar a constante seja o que fica vermelho:

```python
AS_SHIPPED = {
    "llama": "ghcr.io/ggml-org/llama.cpp:server-vulkan-b11382@sha256:4315...57f9",  # inteiro no teste
    "qdrant": "docker.io/qdrant/qdrant:v1.19.2-unprivileged@sha256:efb9...d88b",
}
def test_the_images_are_the_pinned_ones(self): self.assertEqual(catalog.IMAGES, AS_SHIPPED)
def test_the_models_are_the_pinned_ones(self): ...  # repo, revisão, arquivo, bytes, sha256, licença
def test_the_models_weigh_836_mib(self):
    self.assertEqual(catalog.MODELS_BYTES, 437778496 + 438376864)
    self.assertEqual(round(catalog.MODELS_BYTES / 2**20), 836)
def test_the_url_is_the_revision_pinned_resolve_url(self):
    self.assertEqual(catalog.EMBED_MODEL.url(), "https://huggingface.co/gpustack/bge-m3-GGUF/"
                     "resolve/2d48f1737679ad900d5c26c5aad5410e9c70fdca/bge-m3-Q4_K_M.gguf")
def test_the_cpu_command_declares_no_device(self):            # M1
    self.assertEqual(catalog.server_command("embed", None)[-2:], ["-dev", "none"])
def test_a_gpu_command_names_its_device(self):
    self.assertEqual(catalog.server_command("rerank", "Vulkan2")[-2:], ["-dev", "Vulkan2"])
def test_both_roles_carry_the_8192_batch_and_the_current_ui_flag(self):  # M8
    for role, switch in (("embed", "--embedding"), ("rerank", "--reranking")):
        cmd = catalog.server_command(role, None)
        self.assertIn(switch, cmd); self.assertIn("--no-ui", cmd); self.assertNotIn("--no-webui", cmd)
        for flag in ("-c", "-b", "-ub"): self.assertEqual(cmd[cmd.index(flag) + 1], "8192")
def test_qdrant_version_comes_from_the_tag(self):
    self.assertEqual(catalog.qdrant_version(catalog.QDRANT_IMAGE), "1.19.2")
def test_image_flags_parse_and_refuse(self):
    self.assertEqual(catalog.parse_image_flags(["qdrant=q:1"]), {"qdrant": "q:1"})
    for bad in (["qdrant"], ["dzn=x"], ["llama="]):
        with self.assertRaises(StackError): catalog.parse_image_flags(bad)
def test_flags_beat_env_beat_catalog(self):
    got = catalog.resolve_images({"llama": "f"}, {"QCTX_STACK_IMAGE_LLAMA": "e",
                                                 "QCTX_STACK_IMAGE_QDRANT": "  "})
    self.assertEqual(got, {"llama": "f", "qdrant": catalog.QDRANT_IMAGE})
```

- [ ] **Step 2:** `TMPDIR=/tmp python3 -m unittest tests.test_stack_catalog -v`; esperado
  `No module named 'stack.catalog'`.
- [ ] **Step 3:** implementar. O docstring do módulo leva o procedimento de bump da spec (llama.cpp
  pelo digest do índice; Qdrant só a minor seguinte; integração opt-in e
  `python3 tests/test_stack_compose.py --regen`). `qdrant_version` lê `v<x.y.z>` da tag e levanta
  `StackError(step="catalog")` sem versão. `--image` com papel fora de `IMAGES` levanta com
  `fix="--image llama=REF or --image qdrant=REF"`; o papel `llama-dzn` é da fase 3.
- [ ] **Step 4:** testes passam.
- [ ] **Step 5:** commit `feat(stack): the catalogue of images, models and server flags`.

### Task 3: runtimes e o provider de compose

Antes dos fatos do host porque `facts.py` usa o `Runner` daqui.

**Files:**
- Create: `stack/process.py` (`Completed`, `Runner`, `SubprocessRunner`)
- Create: `stack/engine.py` (o contrato dos runtimes e o que os dois engines compartilham)
- Create: `stack/docker.py` (`Docker`) e `stack/podman.py` (`Podman`)
- Create: `stack/runtimes.py` (a porta de entrada: `discover` e as reexportações)
- Create: `tests/stack_fakes.py` (`FakeRunner`, `FakeRuntime`)
- Test: `tests/test_stack_runtimes.py`

**Interfaces:**
- Direção: `process` <- `engine` <- `docker`, `podman` <- `runtimes`. O `stack/runtimes.py`
  reexporta `Completed`, `Runner`, `SubprocessRunner`, `EngineInfo`, `Provider`, `ProviderInfo`,
  `ContainerRuntime`, `Docker`, `Podman`, `normalize_arch`, `socket_alive` e `parse_size`, que as
  tarefas seguintes e os testes importam dele; dentro de `stack/`, `facts` e `backends` importam
  o contrato de `stack/process.py` e `stack/engine.py`, nunca da porta.
- Produces em `stack/process.py`:
  - `@dataclass(frozen=True) class Completed: returncode: int; stdout: str = ""; stderr: str = ""`,
    com `ok` (`returncode == 0`);
  - `class Runner(Protocol): def run(self, argv: list[str], *, timeout: float, stream: bool =
    False) -> Completed`; `class SubprocessRunner` (binário ausente vira `Completed(127, "",
    "<argv0>: not found")`; o comando roda como líder de uma sessão própria
    (`start_new_session`); `TimeoutExpired` vira `StackError(step="runtime")` com o comando e o
    tempo, e mata o grupo de processos inteiro (`os.killpg`), para o backend que o provider de
    compose sobe como filho também morrer; qualquer outra exceção na espera, o Ctrl-C acima de
    tudo, para o grupo inteiro antes de seguir: como o comando está noutra sessão, o Ctrl-C do
    terminal só chega ao Python, então o runner manda ao grupo o SIGINT que o terminal entregaria
    a um grupo em primeiro plano, espera até 5 s, manda SIGKILL ao que sobrar, colhe o processo,
    fecha os pipes e deixa a exceção seguir, nos dois modos; um segundo Ctrl-C na espera vai
    direto ao SIGKILL; `stream=True` herda stdout e stderr do terminal; um `OSError` no `exec`
    vira `StackError(step="runtime")`. Um `nvidia-smi` travado chega ao `collect` da Task 4 como
    o `StackError` do timeout, e o `collect` o lê como um que não lista placa nenhuma);
- Produces em `stack/engine.py`:
  - `normalize_arch(machine: str) -> str` (`x86_64`/`amd64` -> `"amd64"`, `aarch64`/`arm64` ->
    `"arm64"`, o resto em minúsculas);
  - `@dataclass(frozen=True) class EngineInfo: name: str; version: str; os: str; arch: str;
    rootless: bool; kernel: str = ""; vm: str | None = None; socket: str | None = None;
    memory_bytes: int | None = None` (a `memory_bytes` é o que os containers usam: no Docker o
    `MemTotal`, no Podman o `host.memTotal`; é a que a checagem de RAM lê no macOS);
  - `@dataclass(frozen=True) class Provider: argv: tuple[str, ...]; name: str; version: str`;
  - `@dataclass(frozen=True) class ProviderInfo: provider: Provider | None; problem: str | None =
    None; fix: str | None = None; note: str | None = None`;
  - `class ContainerRuntime(Protocol)`: `name: str`; `engine() -> EngineInfo | None`;
    `compose_provider() -> ProviderInfo`; `compose(provider: Provider, project: str, file: Path,
    *args: str, timeout: float, stream: bool = False) -> Completed`;
    `stats(names: list[str]) -> dict[str, int]`;
  - `socket_alive(path: str | None) -> bool` (connect AF_UNIX, timeout 2 s);
  - `parse_size(text: str) -> int | None`;
  - os auxiliares que os dois engines usam, sem sublinhado na frente porque cruzam módulos:
    `as_which`, `INFO_TIMEOUT`, `compose_argv`, `compose_version` (o token depois de `version` na
    linha que nomeia o compose, sem a vírgula do v1), `first_line` (a primeira linha não vazia da
    saída de uma ferramenta), `before_slash` e `stats_failed` (o `StackError` de um `stats` que
    falhou, com a primeira linha do stderr da ferramenta);
- Produces em `stack/docker.py` e `stack/podman.py`: `Docker(runner: Runner,
  which=shutil.which)`, `Podman(runner: Runner, which=shutil.which, host_system: str = "linux",
  socket_alive=socket_alive)`;
- Produces em `stack/runtimes.py`: `discover(runner: Runner, which=shutil.which, host_system:
  str = "linux") -> list[ContainerRuntime]` (Docker antes de Podman; só os que têm binário e cujo
  `engine()` responde);
- Produces em `tests/stack_fakes.py`: `FakeRunner(responses: dict[tuple[str, ...], Completed])`
  (casa pelo prefixo mais longo do argv e grava `calls`) e `FakeRuntime(name, engine,
  provider_info, list_devices: dict[str, str], fail: dict[str, Completed])` (grava cada `compose`
  em `calls`; para `run ... --list-devices` devolve a saída registrada para o arquivo de prova do
  perfil).

- [ ] **Step 1: testes que falham**, com saídas reais gravadas como constantes no teste, cada
  uma dizendo como foi obtida (o comando medido e a data, ou o arquivo e a tag do fonte lido): o
  JSON do `podman info` desta máquina, reduzido aos campos lidos; o `podman compose version` com
  o banner no stderr; um `docker info` de exemplo, rotulado como exemplo:
  - `test_contract_every_runtime_builds_the_same_compose_argv` (Docker, Podman e `FakeRuntime`:
    `[*provider.argv, "-p", "p", "-f", "/x/compose.yaml", "up", "-d"]`);
  - `test_podman_engine_reads_version_os_arch_rootless_and_socket`;
  - `test_docker_engine_reads_rootless_from_security_options`;
  - `test_podman_compose_behind_podman_needs_a_live_socket` (banner docker-compose,
    `socket_alive` falso, sem `podman-compose`: `provider is None`, `problem` nomeia o socket,
    `fix == "systemctl --user enable --now podman.socket"`);
  - `test_a_dead_socket_falls_back_to_podman_compose` (M3: `provider.argv == ("podman-compose",)`
    e `note` diz por quê);
  - `test_podman_info_saying_the_socket_exists_is_not_believed` (M3: `podman info` com
    `exists: true` e `socket_alive` falso dá o mesmo resultado do teste anterior);
  - `test_a_live_socket_keeps_podman_compose_wrapper` (`("podman", "compose")`);
  - `test_docker_prefers_the_plugin_then_the_standalone_binary`;
  - `test_socket_alive_connects_for_real` (um socket AF_UNIX escutando num tmp: `True`; caminho
    inexistente e `None`: `False`);
  - `test_parse_size` (`"1.831GB"` -> 1831000000, `"190.7MB"`, `"1.8GiB"` -> `int(1.8 * 2**30)`,
    `"512KiB"`, `"0B"`, `"--"` -> `None`);
  - `test_stats_reads_both_formats` (linhas JSON do Docker com `Name`/`MemUsage`; lista JSON do
    Podman com `name`/`mem_usage`; devolve bytes da parte antes de `" / "`);
  - `test_the_subprocess_runner_turns_a_timeout_into_a_stack_error` e
    `test_a_missing_binary_is_127`;
  - `test_a_ctrl_c_stops_the_whole_group_in_capture_mode` e `..._in_stream_mode`,
    `test_the_command_gets_the_sigint_first_and_stops_on_its_own` e
    `test_a_second_ctrl_c_cuts_the_grace_short` (de ponta a ponta: um processo intermediário de
    verdade, com o tratador padrão de SIGINT instalado e stderr em DEVNULL, roda
    `sh -c 'sleep 30 & echo $! > <arquivo>; wait'`; o SIGINT vai só para ele; o neto some dentro
    da espera mais 2 s, e o intermediário morre do Ctrl-C que segue; um comando que trata o SIGINT
    para sozinho bem antes dos 5 s; um segundo SIGINT encurta a espera. O tratador é instalado à
    mão porque um job em segundo plano de um shell não interativo herda o SIGINT IGNORADO, e aí o
    sinal nunca chega e o teste não prova nada);
  - `test_discover_skips_a_runtime_whose_engine_does_not_answer`.
- [ ] **Step 2:** rodar; esperado `No module named 'stack.runtimes'`.
- [ ] **Step 3:** implementar. `Podman.compose_provider`: `podman compose version`; o caminho do
  banner `Executing external compose provider "<p>"` decide: basename começando por
  `docker-compose` exige `socket_alive(engine.socket)`; com o banner desligado, a linha de versão
  do docker-compose no stdout decide, nas duas grafias e sem diferenciar maiúsculas
  (`Docker Compose version` do v2 em diante, `docker-compose version` no v1); parado,
  `podman-compose` no PATH vira o provider com `note`; sem ele, `problem` e `fix` (no macOS o fix
  é `podman machine start`).
  `podman compose version` falhando, tenta `podman-compose version`. No macOS, `engine().vm` vem
  do `Host.VMType` de `podman machine info` (em minúsculas), e `engine().socket` do
  `ConnectionInfo.PodmanSocket.Path` de `podman machine inspect <Host.CurrentMachine>`: o
  `remoteSocket` do `podman info` é o caminho dentro da VM, e o `machine inspect` não tem VMType
  (lido no código do Podman v5.7.0 e v6.0.0). No Docker, `docker info --format '{{json .}}'`
  (o atalho `json` não existe nos CLIs antigos), `OSType` em `engine().os` e `KernelVersion` em
  `engine().kernel`. O engine responde só pelo JSON, não pelo exit code: o CLI do Docker 23.0 a
  28.0 sai 0 com os campos zerados e `"SecurityOptions": null` quando o daemon não responde, e um
  atalho `docker` que é o shim do Podman imprime o info do Podman (sem `ServerVersion`); então
  `engine()` exige `ServerVersion` não vazio e `ServerErrors` vazio, e lê `SecurityOptions` com
  `or []` (senão dá `TypeError` e o `discover` cai, mesmo com o Podman funcionando). A versão do
  provider é a da linha que nomeia o compose: o
  `podman-compose version` daqui imprime `podman version 5.7.0` ANTES de
  `podman-compose version 1.6.0`.
- [ ] **Step 4:** testes passam.
- [ ] **Step 5:** commit `feat(stack): Docker and Podman behind one runtime contract`.

### Task 4: os fatos do host

**Files:**
- Create: `stack/facts.py`
- Test: `tests/test_stack_facts.py`

**Interfaces:**
- Consumes: `Runner`, `Completed`, `SubprocessRunner` (de `stack/process.py`), `normalize_arch`
  (de `stack/engine.py`) (Task 3).
- Produces:
  - `@dataclass(frozen=True) class Gpu: vendor: str; card: str` (o `card` é o endereço PCI);
  - `@dataclass(frozen=True) class NvidiaFacts: gpus: tuple[str, ...] = (); icd: bool = False;
    docker_hook: bool = False; cdi_hook: bool = False; cdi_spec: bool = False; ctk: bool = False`;
  - `@dataclass(frozen=True) class HostFacts: system: str; arch: str; wsl: bool;
    ram_bytes: int | None; disk_free_bytes: int | None; gpus: tuple[Gpu, ...];
    render_nodes: tuple[str, ...]; selinux: bool; nvidia: NvidiaFacts = NvidiaFacts()`;
  - `VENDORS = {"0x1002": "amd", "0x8086": "intel", "0x10de": "nvidia"}`;
  - `@dataclass class Probe: root: Path = Path("/"); system: Callable[[], str] =
    platform.system; machine: Callable[[], str] = platform.machine; which = shutil.which;
    runner: Runner = field(default_factory=SubprocessRunner);
    disk_usage: Callable[[Path], Any] = shutil.disk_usage` (o `runner` com um
    `SubprocessRunner` de verdade faz o `collect` de produção rodar o `nvidia-smi`; um teste
    injeta um falso; o `disk_usage` injetado mantém o teste do disco sem tocar no disco real);
  - `normalize_system(name: str) -> str` (`Linux` -> `"linux"`, `Darwin` -> `"macos"`,
    `Windows` -> `"windows"`);
  - `collect(probe: Probe, stack_dir: Path) -> HostFacts` (um `nvidia-smi` que trava, e o
    runner então levanta `StackError` no timeout, ou que falha, é um driver que não lista placa
    nenhuma: `NvidiaFacts.gpus == ()`, que o perfil nvidia lê como "install the NVIDIA driver";
    a detecção segue, porque a CPU continua sendo a reserva em todo host);
  - `port_free(port: int, host: str = "127.0.0.1") -> bool`;
  - `platform_of(facts: HostFacts, engine_kernel: str = "") -> str` (`"macos"`; `"windows"`
    quando o sistema é Windows, ou WSL, ou o kernel do engine contém `microsoft` (em
    minúsculas); senão `"linux"`).

- [ ] **Step 1: testes que falham**, com árvores `/sys` e `/proc` falsas sob um tmp passado como
  `Probe.root`:
  - `test_this_machines_shape_two_intel_one_amd_and_the_bmc_ignored` (os devices de vídeo do
    barramento PCI desta máquina, lidos de `/sys/bus/pci/devices/*`: dois `0x8086`, um `0x1002`,
    um `0x1a03` (BMC, ignorado) e as funções de áudio, classe `0x04`, também ignoradas; três
    `Gpu`, com o `card` sendo o endereço PCI);
  - `test_a_driverless_nvidia_is_on_the_bus_and_visible` (uma placa NVIDIA sem driver, que o
    `/sys/class/drm` não listaria, aparece no PCI) e `test_a_non_display_function_is_not_a_gpu`;
  - `test_render_nodes_are_listed`;
  - `test_wsl_is_read_from_the_kernel_release` (`/proc/sys/kernel/osrelease` com `microsoft`;
    o `/etc/os-release` é o da distro e não nomeia o WSL);
  - `test_ram_is_mem_available_on_linux` (`MemAvailable: 1000 kB` -> 1024000) e
    `test_ram_on_macos_is_not_the_host_figure` (no macOS o `ram_bytes` é `None`: o que os
    containers usam é o número do engine, não o `hw.memsize` do host);
  - `test_disk_is_measured_at_the_nearest_existing_ancestor` e os dois do walk, com
    `Probe.disk_usage` injetado (nenhum lê o disco de verdade);
  - `test_selinux_enforcing_is_read`;
  - `test_nvidia_readiness_has_its_parts` (nvidia-smi `-L` pelo `FakeRunner`; ICD em
    `usr/share/vulkan/icd.d/nvidia_icd.json`; `docker_hook`, `cdi_hook` e `ctk` pelo `which`;
    spec CDI em `etc/cdi/` no formato do teste do próprio nvidia-container-toolkit,
    `cmd/nvidia-ctk/cdi/generate/generate_test.go` na tag v1.18.0, com o `kind` que o comando
    grava por padrão, `nvidia.com/gpu`);
  - `test_a_cdi_file_that_is_not_utf8_is_read_as_bytes` (um arquivo que não é UTF-8 em
    `etc/cdi/`, sozinho, dá `False` sem exceção; ao lado do spec de verdade, `True`) e
    `test_a_hung_or_failing_nvidia_smi_lists_no_gpu` (um runner que levanta `StackError`, e um
    que sai com erro: `gpus == ()`, e os fatos de arquivo continuam lidos);
  - `test_the_default_probe_points_at_the_real_root_and_runner` (o `Probe()` padrão tem um
    `SubprocessRunner` de verdade, não `None`; os outros padrões são comparados por identidade,
    `platform.system`, `platform.machine`, `shutil.which` e `shutil.disk_usage`, sem chamar
    nenhum, para o teste não depender da máquina);
  - `test_normalize_and_platform_of`;
  - `test_port_free_says_no_for_a_bound_port`.
- [ ] **Step 2:** rodar; esperado `No module named 'stack.facts'`.
- [ ] **Step 3:** implementar.
- [ ] **Step 4:** passam.
- [ ] **Step 5:** commit `feat(stack): host facts from injected primitives`.

### Task 5: os perfis de backend

**Files:**
- Create: `stack/devices.py` (`Device`, `parse_devices`, `vendor_of`: a leitura do que o
  `--list-devices` de um container imprimiu)
- Create: `stack/backends.py` (importa os três de `stack/devices.py` e os reexporta)
- Test: `tests/test_stack_backends.py`

**Interfaces:**
- Consumes: `HostFacts` (Task 4), `EngineInfo` (de `stack/engine.py`, Task 3).
- Produces em `stack/devices.py` (reexportados pelo `stack/backends.py`):
  - `@dataclass(frozen=True) class Device: index: int; name: str; total_mib: int; free_mib: int`,
    com `id` (`f"Vulkan{index}"`) e `vendor` (`vendor_of(name)`);
  - `parse_devices(output: str) -> list[Device]` (tolera `\r`, ignora o que não casa, `(none)` dá
    lista vazia);
  - `vendor_of(name: str) -> str | None` (`NVIDIA` -> nvidia; `AMD` ou `RADV` -> amd; `Intel` ->
    intel; `Virtio`, `Venus` ou `Apple` -> apple; llvmpipe -> `None`; o device do dzn, na fase 3,
    embrulha o nome do adaptador, `Microsoft Direct3D12 (<adaptador>)`, então valem os mesmos
    tokens);
- Produces em `stack/backends.py`:
  - `READY, MISSING, RUNTIME, UNSUPPORTED = "ready", "missing", "runtime", "unsupported"`;
    `@dataclass(frozen=True) class Availability: state: str; reason: str = ""; fix: str | None =
    None; needs: str | None = None`;
  - `class Backend(Protocol)`: `id: str; experimental: bool; vendor: str | None;
    image_role: str | None`; `runtimes(platform: str) -> frozenset[str]`;
    `availability(platform: str, facts: HostFacts, engine: EngineInfo, runtime: str) ->
    Availability`; `service_patch(runtime: str, gpu_index: int | None) -> dict`;
    `devices_seen(output: str) -> list[Device]`;
  - `Cpu()`, `DriGpu(vendor: str)`, `NvidiaGpu()`, `AppleGpu()`;
    `BACKENDS: dict[str, Backend]` na ordem `cpu, amd, intel, nvidia, apple`;
    `GPU_PROFILES = ("amd", "intel", "nvidia", "apple")`;
  - `runtime_label(runtimes: frozenset[str]) -> str` (`"Docker and Podman"`, `"Podman only"`,
    `"Docker only"`, `"not available here"`);
  - `@dataclass(frozen=True) class Option: backend: str; device: Device | None;
    gpu_index: int | None; availability: Availability`;
  - `default_option(options: list[Option]) -> Option` (entre as `READY` com device e não
    experimentais, a de maior `free_mib`; nenhuma, a opção `cpu`).

- [ ] **Step 1: testes que falham:**
  - `test_parse_devices_on_this_machines_output` (as três linhas medidas: `Vulkan0` Intel 32656/
    29268, `Vulkan1` RADV 16368/4018, `Vulkan2` Intel 32656/29289);
  - `test_parse_devices_none_crlf_and_noise` (`(none)` -> `[]`; a mesma saída com `\r\n`; uma
    linha de log no meio é ignorada);
  - `test_vendor_of` (inclui `"NVIDIA GeForce RTX 4090"`, `"Virtio-GPU Venus (Apple M2 Pro)"` e
    o nome real de um dzn, `"Microsoft Direct3D12 (NVIDIA GeForce GTX 1080)"` -> nvidia, com a
    procedência: o `dzn_device.c` do Mesa na tag mesa-26.0.3 e o comentário de 2026-02-20 na
    issue 1215 do microsoft/wslg);
  - `test_the_compatibility_matrix_is_the_specs` (absoluto: Linux `cpu/amd/intel/nvidia` x
    `docker/podman`; macOS `cpu` x `docker/podman` e `apple` x `podman`; nada mais, nem a CPU
    no Windows, que a fase 1 recusa em vez de oferecer CPU);
  - `test_availability_table` (subTests: Dri sem GPU do vendor -> UNSUPPORTED; sem `renderD` ->
    MISSING com fix; Linux pronto -> READY; NVIDIA sem placa NVIDIA no host -> UNSUPPORTED;
    placa que o `nvidia-smi` não lista -> MISSING com o driver; NVIDIA sem ICD -> MISSING com
    `install the driver's GL/Vulkan package (libnvidia-gl-<version> on Ubuntu)` (no Podman,
    seguido do `CDI_GENERATE`); Docker sem hook -> MISSING (o hook de CDI só conta a partir do
    Docker 29.2); Podman sem CDI -> MISSING com
    `CDI_GENERATE = "nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml (as root)"`, ou com
    `install nvidia-container-toolkit, then CDI_GENERATE` quando o `nvidia-ctk` não está no
    PATH; a
    palavra do comando de root fica fora do código, porque o scanner do hermes a marca como
    high num módulo;
    Apple no Docker -> RUNTIME com `needs == "podman"`; Apple com VM
    `applehv`, no Podman 5 ou 6, -> MISSING com UMA fix única: gravar
    `provider = "libkrun"` na tabela `[machine]` de
    `~/.config/containers/containers.conf` (e tirar o `CONTAINERS_MACHINE_PROVIDER` exportado) e
    depois `podman machine init --now`; Mac Intel -> UNSUPPORTED;
    libkrun em arm64 -> READY, também com o Python sob Rosetta, porque vale o arch do
    engine; CPU fora de Linux e macOS -> UNSUPPORTED);
  - `test_service_patches` (Dri no Podman: `/dev/dri:/dev/dri`, a forma do exemplo da spec e
    das fixtures validadas nos providers, mais a anotação
    `run.oci.keep_original_groups: "1"`; no Docker, só o device; NVIDIA no Docker: o bloco
    `deploy` com `device_ids ["1"]` e `capabilities [gpu, compute, utility, graphics]`; no Podman:
    `devices ["nvidia.com/gpu=1"]`; NVIDIA sem índice -> `StackError`; Apple no Podman:
    `/dev/dri:/dev/dri` mais a mesma anotação (a máquina roda rootless por padrão); no Docker,
    só o device; Cpu: `{}`);
  - `test_devices_seen_filters_by_vendor`;
  - `test_default_is_the_most_free_memory_and_never_experimental` (M5: entre Intel 29268 e 29289
    e AMD 4018, `Vulkan2`; uma Apple READY nunca é o padrão; sem GPU, `cpu`);
  - `test_runtime_label`.
- [ ] **Step 2:** rodar; esperado `No module named 'stack.backends'`.
- [ ] **Step 3:** implementar. A anotação vai sempre que o runtime é Podman no `DriGpu`: no
  rootful ela é inofensiva e evita um ramo por rootless (ruling deste plano).
- [ ] **Step 4:** passam.
- [ ] **Step 5:** commit `feat(stack): backend profiles and where each one runs`.

### Task 6: o compose gerado e as 9 fixtures

**Files:**
- Create: `stack/compose.py`
- Create: `tests/fixtures/stack/{linux-docker-cpu,linux-docker-dri,linux-docker-nvidia,linux-podman-cpu,linux-podman-dri,linux-podman-nvidia,macos-docker-cpu,macos-podman-cpu,macos-podman-apple}.yaml`
- Test: `tests/test_stack_compose.py`

**Interfaces:**
- Consumes: `catalog`, `BACKENDS`.
- Produces:
  - `@dataclass(frozen=True) class Plan: platform: str; runtime: str; backend: str;
    device: str | None; gpu_index: int | None; ports: dict[str, int]; stack_dir: Path;
    images: dict[str, str]; selinux: bool = False; project: str = PROJECT`;
  - `render(plan: Plan) -> dict` (puro; recusa um `stack_dir` com `:` ou com caractere fora do
    Plano Multilíngue Básico, `StackError(step="compose")`, porque a sintaxe curta de volume
    `<host>:/models:ro` quebra nos dois providers); `emit(doc: dict) -> str`; `dump(plan: Plan)
    -> str`;
  - `container_name(project: str, service: str) -> str` (`f"{project}-{service}"`);
  - `volume_name(project: str) -> str` (`f"{project}_{VOLUME}"`, o nome real nos dois providers).

Regras do `render`: chaves de serviço na ordem `container_name, image, restart, command, ports,
volumes, environment, devices, deploy, annotations, cap_drop, security_opt`; Qdrant com o
volume `memories-plugin-qdrant:/qdrant/storage`; `embed` e `rerank` com
`server_command(role, plan.device)`, `127.0.0.1:<porta>:8080`,
`<stack_dir>/models:/models:ro` (`:ro,z` com SELinux) e o `service_patch` do backend; volumes de
topo `{VOLUME: {}}`. Regras do `emit`: bloco, indentação de 2; todo escalar por `json.dumps`; a
chave sai sem aspas quando casa `[A-Za-z_][A-Za-z0-9_]*` e não é palavra reservada do YAML 1.1
(`y`, `yes`, `no`, `true`, `false`, `on`, `off`, `null`, `~` e variações de caixa); mapa vazio
como `{}`.

Fixtures: device `"Vulkan0"` em dri, nvidia e apple; `gpu_index=0` na nvidia; backend `amd` nas
`dri`; portas padrão; imagens do catálogo; `stack_dir` `/home/me/.local/share/memories-plugin/stack`
no Linux e `/Users/me/.local/share/memories-plugin/stack` no macOS; sem SELinux.

- [ ] **Step 1: testes que falham:**
  - `test_each_fixture_is_what_render_produces` (9 subTests, texto igual);
  - `test_fixtures_exist_iff_the_runtime_serves_the_profile` (o conjunto de nomes de arquivo é
    derivado de `BACKENDS[b].runtimes(p)`, com `amd` e `intel` dobrados em `dri`; não existe
    `macos-docker-apple`);
  - `test_amd_and_intel_render_the_same_file`;
  - `test_every_scalar_is_json_quoted` (todo valor depois de `": "` ou `"- "` é um escalar JSON
    válido);
  - `test_reserved_and_dotted_keys_are_quoted` (`emit({"yes": 1, "run.oci.keep_original_groups":
    "1", "a_b": 2})` -> `"yes": 1`, `"run.oci.keep_original_groups": "1"`, `a_b: 2`);
  - `test_cpu_service_is_pinned_to_no_device` (M1: `-dev none` no comando e nenhuma chave
    `devices`);
  - `test_ports_bind_loopback_only`; `test_selinux_adds_the_z_label`;
  - `test_container_and_volume_names_carry_the_project` (M6).
  - `test_a_stack_dir_with_a_colon_is_refused` e
    `test_a_stack_dir_with_a_non_bmp_character_is_refused` (R2-9: `render` levanta
    `StackError(step="compose")` com a fix apontando para
    `QCTX_STACK_DIR`), e `test_an_accented_stack_dir_is_accepted` (acento dentro do BMP não
    recusa).
  Regeneração: `python3 tests/test_stack_compose.py --regen` reescreve as 9.
- [ ] **Step 2:** rodar; esperado `No module named 'stack.compose'`.
- [ ] **Step 3:** implementar e gerar as fixtures com `--regen`; ler as 9 uma vez, à mão, contra o
  exemplo da spec.
- [ ] **Step 4:** testes passam, e os providers aceitam as fixtures de verdade:
  `for f in tests/fixtures/stack/*.yaml; do docker-compose -p memories-plugin -f "$f" config -q &&
  podman-compose -p memories-plugin -f "$f" config >/dev/null || echo "FAIL $f"; done`. Esperado:
  nenhuma linha `FAIL`.
- [ ] **Step 5:** commit `feat(stack): the compose file, rendered from the catalogue and a plan`.

### Task 7: download dos modelos com retomada e barra

**Files:**
- Create: `stack/fetch.py`
- Test: `tests/test_stack_fetch.py`

**Interfaces:**
- Consumes: `Model`, `StackError`.
- Produces:
  - `class Transport(Protocol): def get(self, url: str, *, start: int = 0) -> Iterator[bytes]`
    (os bytes começam em `start`);
  - `UrllibTransport(timeout: float = 60.0, chunk: int = 1 << 20)` (cabeçalho `Range` com
    `start > 0`; resposta 200 a um pedido com Range descarta os primeiros `start` bytes; erro de
    rede vira `StackError(step="download")`);
  - `ProgressBar(total: int, label: str, *, tty: bool, write: Callable[[str], None],
    clock: Callable[[], float] = time.monotonic, start: int = 0, slices: int = 10)` com
    `update(done: int)` e `finish()`;
  - `fetch(model: Model, models_dir: Path, transport: Transport, *, bar: Callable[[int, int],
    ProgressBar], log: Callable[[str], None]) -> str` (`"present"`, `"downloaded"` ou
    `"resumed"`).

Linha da barra: `f"  {label}  {pct:3d}%  {done/2**20:.1f}/{total/2**20:.1f} MiB  {speed:.1f} MiB/s
ETA {eta}"`, com velocidade só sobre os bytes desta execução e `eta` em `m:ss` (`--` antes de
haver velocidade). Em TTY cada atualização reescreve com `"\r"` e só o `finish` quebra a linha;
fora dele, uma linha por fatia de 10%.

- [ ] **Step 1: testes que falham**, com `FakeTransport(content, cut_after=None,
  ignore_range=False)` e um `Model` de teste com o sha256 do conteúdo:
  - `test_a_fresh_download_lands_and_verifies`;
  - `test_a_part_file_resumes_from_its_size` (o transport recebe `start == len(part)`; resultado
    `"resumed"`; hash certo);
  - `test_a_server_that_ignores_range_still_resumes_correctly`;
  - `test_a_wrong_hash_deletes_the_part_and_fails` (`StackError` com `sha256` na mensagem; `.part`
    apagado);
  - `test_a_short_stream_keeps_the_part_and_says_run_again`;
  - `test_an_oversized_stream_deletes_the_part`;
  - `test_a_present_file_with_the_right_hash_is_skipped` (o transport nunca é chamado);
  - `test_a_present_file_with_the_wrong_hash_is_replaced`;
  - `test_tty_bar_rewrites_one_line`; `test_non_tty_bar_prints_one_line_per_slice` (10 linhas de
    0 a 100%);
  - `test_resumed_bar_starts_at_the_part_and_counts_speed_for_this_run_only` (relógio falso:
    porcentagem, MiB, MiB/s e ETA saem do byte certo).
- [ ] **Step 2:** rodar; esperado `No module named 'stack.fetch'`.
- [ ] **Step 3:** implementar. Hash do `.part` existente antes de continuar; `os.replace` só depois
  de tamanho e hash conferidos; Ctrl-C deixa o `.part`.
- [ ] **Step 4:** passam.
- [ ] **Step 5:** commit `feat(stack): resumable, verified model download with a progress bar`.

### Task 8: prontidão, verificação funcional, calibração e o ambiente

**Files:**
- Create: `stack/health.py`, `stack/verify.py`
- Test: `tests/test_stack_health.py`, `tests/test_stack_verify.py`

**Interfaces:**
- Consumes: `core.Config`, `core.config.load`, `core.setup.Check`, `core.setup.diagnose`,
  `core.config.ENV_ALIASES`, `core.chunk.HARD_MAX_CHARS`, `core.chunk.TARGET_CHARS`,
  `core.build_embedder`, `core.build_reranker`, `catalog.EMBED_DIM`.
- Produces (health): `READY_TIMEOUT_S = 600.0`; `http_status(url: str, *, timeout: float = 5.0)
  -> int | None`; `endpoints(ports: dict[str, int]) -> dict[str, str]`
  (`/readyz` no Qdrant, `/health` nos dois llama); `wait_ready(targets: dict[str, str], *,
  status=http_status, clock=time.monotonic, sleep=time.sleep, timeout: float = READY_TIMEOUT_S,
  interval: float = 1.0) -> dict[str, float]`.
- Produces (verify):
  - `stack_urls(ports: dict[str, int]) -> dict[str, str]` (M2: `qdrant_url`,
    `api_base_url` com `/v1`, `embed_url` vazio, `rerank_url` com `/v1/rerank`);
  - `stack_config(ports: dict[str, int], *, vector_size: int = EMBED_DIM) -> Config` (defaults
    sem arquivo e sem ambiente, mais `stack_urls` e `vector_size`);
  - `FUNCTIONAL_CHECKS = ("Qdrant", "Embedding", "Re-rank")`;
    `functional(cfg: Config, *, diagnose=core.setup.diagnose) -> tuple[list[Check], int | None]`;
    `functional_ok(checks: list[Check]) -> bool` (todos `ok`; aqui o aviso do Re-rank conta como
    falha);
  - `@dataclass(frozen=True) class Budget: host: str; embed_s: float; rerank_s: float`;
  - `CALIBRATION_RERANK_DOCS = 20`;
  - `calibrate(cfg: Config, budgets: list[Budget], *, embedder=None, reranker=None,
    clock=time.monotonic, memory: Callable[[], dict[str, int]] | None = None) ->
    tuple[list[Check], dict]` (aquece embed e rerank uma vez, M7; mede um embed de
    `HARD_MAX_CHARS` caracteres e um rerank de 20 documentos de `TARGET_CHARS`; um `Check` por
    host, `warning=True` quando estoura; nunca levanta por orçamento);
  - `env_overrides(env: Mapping[str, str], wanted: dict[str, str]) -> list[tuple[str, str]]`
    (para cada campo de `wanted`, o primeiro alias não vazio de `ENV_ALIASES[campo]` cujo valor
    difere do desejado, como `(nome, valor)`).

- [ ] **Step 1: testes que falham:**
  - health: `test_503_then_200_is_ready`; `test_never_ready_raises_naming_the_service_and_its_last_answer`;
    `test_no_answer_is_none_not_an_exception`;
  - verify: `test_stack_urls_carry_v1` (absoluto: `{"qdrant_url": "http://127.0.0.1:6333",
    "api_base_url": "http://127.0.0.1:8003/v1", "embed_url": "", "rerank_url":
    "http://127.0.0.1:8004/v1/rerank"}`); `test_stack_config_ignores_the_users_file_and_env`;
    `test_functional_keeps_only_the_three_checks`; `test_a_rerank_warning_fails_the_stack`;
    `test_calibration_warms_up_before_measuring` (o embedder falso vê a chamada de aquecimento
    antes da medida); `test_over_budget_is_a_named_warning_never_a_raise` (`"hermes"`,
    `"rerank"`, o medido e o orçamento na mensagem); `test_env_overrides` (subTests:
    `QDRANT_URL` diferente é nomeado; igual não; em branco é ignorado; o canônico vence o legado;
    `RECALL_EMBED_URL` com `embed_url` desejado vazio é nomeado);
  - `test_the_rerank_sample_is_the_recall_top_k`: lê `hooks/recall.py` por AST, acha a
    atribuição `TOP_K = env_num(...)` e compara o default `"20"` com
    `CALIBRATION_RERANK_DOCS`.
- [ ] **Step 2:** rodar; esperado `No module named 'stack.health'`.
- [ ] **Step 3:** implementar. O `functional` chama o `diagnose` inteiro e filtra por nome, como
  a spec manda.
- [ ] **Step 4:** passam.
- [ ] **Step 5:** commit `feat(stack): readiness, functional check, calibration and the env trap`.

### Task 9: o estado gravado

**Files:**
- Create: `stack/state.py`
- Test: `tests/test_stack_state.py`

**Interfaces:**
- Consumes: `core.statefile.write_json`, `compose.Plan`.
- Produces: `SCHEMA = 1`; `STATE_FILE = "stack.json"`, `COMPOSE_FILE = "compose.yaml"`,
  `MODELS_DIR = "models"`; `stack_dir(env: Mapping[str, str] = os.environ) -> Path`;
  `@dataclass class StackState: role: str; listen: str; platform: str; runtime: str;
  provider: list[str]; profile: str; device: str | None; gpu_index: int | None;
  ports: dict[str, int]; images: dict[str, str]; models: dict[str, str]; qdrant_version: str;
  selinux: bool; phase: str; created_at: str; updated_at: str; project: str = PROJECT;
  schema: int = SCHEMA`; fases `"compose"`, `"running"`, `"stopped"`;
  `load(directory: Path) -> StackState | None`; `save(directory: Path, state: StackState) ->
  None`; `plan_of(state: StackState, directory: Path) -> Plan`; `now() -> str` (ISO-8601 UTC).

- [ ] **Step 1: testes que falham:** `test_roundtrip`; `test_absent_is_none`;
  `test_a_corrupt_file_names_itself_and_the_remove_fix` (`StackError`, `step == "state"`,
  `fix == "qctx stack remove"`); `test_a_wrong_shape_is_corrupt_too` (`ports` como lista);
  `test_a_newer_schema_is_refused`; `test_the_file_is_owner_only` (0o600);
  `test_stack_dir_precedence` (`QCTX_STACK_DIR` > `XDG_DATA_HOME` > `HOME`; em branco conta como
  ausente); `test_plan_of_carries_every_field`.
- [ ] **Step 2:** rodar; esperado `No module named 'stack.state'`.
- [ ] **Step 3:** implementar. `save` que recebe `False` do `write_json` levanta
  `StackError(step="state")`: um estado que não pousou não pode ser relatado como gravado.
- [ ] **Step 4:** passam.
- [ ] **Step 5:** commit `feat(stack): stack.json, read tolerantly and written atomically`.

### Task 10: o caso de uso de provisionar

**Files:**
- Create: `stack/installer.py`
- Modify: `tests/stack_fakes.py` (`ScriptedPrompter`, `RecordingReporter`, `FakeConfigSink`,
  `FakeTransport`)
- Test: `tests/test_stack_installer.py`

**Interfaces:**
- Consumes: tudo das Tasks 2 a 9.
- Produces:
  - `class Prompter(Protocol)`: `ask(prompt: str) -> str`; `confirm(prompt: str, *,
    default: bool = False) -> bool`; `choose(title: str, lines: list[str], default: int) -> int`;
  - `class Reporter(Protocol)`: `step`, `ok`, `info`, `warn`, `fail` (cada um `(text: str) ->
    None`) e `bar(total: int, label: str, start: int) -> ProgressBar`;
  - `class ConfigSink(Protocol)`: `current_file() -> dict`; `save(patch: dict) -> None`;
    `effective() -> Config`;
  - `@dataclass class Request: profile: str | None = None; runtime: str | None = None;
    yes: bool = False; images: dict[str, str] = {}; ports: dict[str, int] = PORTS;
    project: str = PROJECT` (com `field(default_factory=...)`);
  - `@dataclass class Deps: runtimes: list[ContainerRuntime]; facts: HostFacts;
    prompter: Prompter; reporter: Reporter; config: ConfigSink; transport: Transport;
    stack_dir: Path; budgets: list[Budget]; env: Mapping[str, str];
    port_free: Callable[[int], bool]; status=http_status; diagnose=core.setup.diagnose;
    calibrate=verify.calibrate; memory: Callable[[ContainerRuntime, list[str]], dict[str, int]] |
    None = None; clock=time.monotonic; sleep=time.sleep`;
  - `provision(request: Request, deps: Deps) -> StackState | None` (`None` quando o usuário
    recusa o resumo);
  - `choose_ports(wanted: dict[str, int], port_free: Callable[[int], bool]) -> dict[str, int]`
    (a primeira livre em `range(p + 10000, p + 10100)`; nenhuma, `StackError(step="ports")`);
  - `build_options(platform: str, facts: HostFacts, engine: EngineInfo, runtime: str,
    proofs: dict[str, list[Device]], availability: dict[str, Availability]) -> list[Option]`;
  - `option_line(option: Option, platform: str) -> str`;
  - `config_patch(ports: dict[str, int], dim: int | None) -> dict`;
  - `config_diff(current: dict, patch: dict) -> list[tuple[str, str, str]]` (só o que muda);
  - `reboot_hint(runtime: str, platform: str) -> list[str]`.

Etapas, uma função privada cada, na ordem da spec ("O que ela faz, em ordem"):
1. runtime: os que têm provider; nenhum, `StackError(step="runtime")` com os `fix` coletados;
   `--runtime` que não responde, recusado; `apple` vai para o Podman sem perguntar, e com
   `--runtime docker` é recusado (`"apple runs on Podman only"`); dois e `--yes`, Docker; dois sem
   `--yes`, a pergunta, com cada linha dizendo o que aquele runtime atende aqui (perfis `READY` ou
   `MISSING` nele). A `note` do provider vai para o `reporter.info`.
2. plataforma (Windows recusado com o caminho do README), disponibilidade de cada backend,
   portas (`choose_ports`, cada porta movida relatada), disco (bloqueia abaixo de
   `MODELS_BYTES + 400 MiB` menos o que já está baixado), RAM (aviso abaixo de 6 GiB; o número
   vem do engine, `EngineInfo.memory_bytes`: é ele que os containers usam, e no macOS o host tem
   mais memória do que a VM da máquina do Podman).
3. `<stack>/models` criado; compose de cada perfil `READY` em `<stack>/probe/`; `compose config`
   em cada um.
4. `compose pull` com `stream=True` e 1800 s.
5. prova: `compose run -T --rm --no-deps embed --list-devices` (M4) por perfil de GPU `READY`; na
   NVIDIA, um render por índice do `nvidia-smi`; prova que falha deixa o perfil `MISSING` com o
   final do stderr.
6. menu: `build_options`; perfil `None` pergunta (padrão de `default_option`; escolher
   indisponível repete motivo e correção e pergunta de novo); `auto` pega o padrão sem perguntar;
   perfil explícito sem opção `READY` levanta `StackError(step="profile")` com motivo e correção.
7. resumo (modelos e MiB que faltam baixar, portas, diretório, `volume_name`) e
   `confirm("proceed? [y/N]")`, pulado com `yes`; recusado, devolve `None`.
8. `fetch` dos dois modelos com a barra do `reporter`.
9. `compose.yaml` do plano, `stack.json` em `compose`, `compose up -d`, `wait_ready`; falhando,
   `compose logs --tail 50 <serviço>` de cada um que não ficou pronto, e `StackError`.
10. `functional` (falha levanta `StackError(step="verify")` e o config fica intocado) e
    `calibrate` (avisos pelo `reporter.warn`, memória pelo `stats`).
11. `config_patch`, `config_diff` mostrado; valor não vazio substituído pede `confirm` (sim com
    `yes`); recusado, não grava e diz o `qctx config set` de cada campo; gravado, cada
    `env_overrides` vira `warn` com a correção "remove its export from your shell rc".
12. `stack.json` em `running`; `reboot_hint` no `reporter.info`. O diretório `probe/` é apagado.

- [ ] **Step 1: testes que falham** (todos com `FakeRuntime`, `FakeTransport`, `ScriptedPrompter`,
  `RecordingReporter` e `FakeConfigSink`):
  - `test_happy_path_cpu_writes_compose_state_and_config_in_order` (a ordem gravada no
    `FakeConfigSink` é depois do `functional`; `stack.json` em `running`; `compose.yaml ==
    dump(plan)`);
  - `test_happy_path_each_gpu_profile` (subTests amd, intel, nvidia com fakes, apple em macOS com
    Podman e libkrun);
  - `test_a_host_gpu_the_container_does_not_see_is_listed_unavailable`;
  - `test_explicit_profile_with_no_proved_gpu_stops_with_reason_and_fix`;
  - `test_an_option_the_runtime_does_not_serve_says_what_it_needs_and_asks_again` (macOS com
    Docker: a linha da Apple diz `Podman only`; escolhida, o reporter repete a correção e o menu
    volta);
  - `test_stack_apple_with_runtime_docker_is_refused`;
  - `test_two_runtimes_question_says_what_each_serves`; `test_yes_picks_docker`;
    `test_stack_apple_goes_to_podman_without_asking`;
  - `test_only_the_busy_port_moves` e `test_the_next_free_port_is_taken_when_plus_10000_is_busy`;
  - `test_env_trap_names_the_variable_and_its_value`;
  - `test_config_is_written_only_after_verification`;
  - `test_replacing_a_non_empty_value_asks_first`;
  - `test_an_existing_embed_url_is_cleared_by_the_patch`;
  - `test_auto_picks_the_proved_gpu_with_most_free_memory`; `test_auto_never_picks_apple`;
    `test_auto_without_a_proved_gpu_falls_back_to_cpu`;
  - `test_declining_the_summary_downloads_nothing`;
  - `test_cpu_plan_pins_no_device_even_when_the_engine_injects_gpus` (o `FakeRuntime` devolve as
    três GPUs até para o perfil `cpu`, como M1; o `compose.yaml` gravado leva `-dev none`);
  - `test_a_failed_up_shows_the_service_log_tail`;
  - `test_windows_is_refused_with_the_readme_path`;
  - `test_a_dead_podman_socket_uses_podman_compose_and_records_it` (o `provider` do `stack.json`
    é `["podman-compose"]`).
- [ ] **Step 2:** rodar; esperado `No module named 'stack.installer'`.
- [ ] **Step 3:** implementar.
- [ ] **Step 4:** passam.
- [ ] **Step 5:** commit `feat(stack): the provisioning use case`.

### Task 11: o ciclo de vida

**Files:**
- Create: `stack/lifecycle.py`
- Test: `tests/test_stack_lifecycle.py`

**Interfaces:**
- Consumes: `state`, `compose`, `health`, `verify.stack_urls`, `ContainerRuntime`, `Reporter`,
  `Prompter`, `ConfigSink`, `Runner`.
- Produces:
  - `@dataclass class LifeDeps: runtimes: list[ContainerRuntime]; reporter: Reporter;
    prompter: Prompter; config: ConfigSink; stack_dir: Path; runner: Runner;
    status=http_status; clock=time.monotonic; sleep=time.sleep`;
  - `status(deps: LifeDeps) -> tuple[dict, int]`: sem stack, `({"managed": False}, 0)`; com
    stack, `{"managed": True, "phase", "runtime", "provider", "profile", "device", "ports",
    "services": {nome: {"url", "status"}}, "healthy", "outdated_pins", "config_points_here",
    "boot"}` e saída 1 se não está saudável; estado corrompido,
    `{"managed": True, "error", "fix": "qctx stack remove"}` e 1;
  - `up(deps: LifeDeps, *, upgrade: bool = False, images: dict[str, str] | None = None) ->
    StackState`;
  - `down(deps: LifeDeps) -> StackState`;
  - `remove(deps: LifeDeps, *, purge_models: bool = False, purge_data: bool = False,
    yes: bool = False) -> list[str]`;
  - `check_minor(installed: str, target: str) -> None`;
  - `runtime_for(state: StackState, runtimes: list[ContainerRuntime]) ->
    tuple[ContainerRuntime, Provider]` (sempre o runtime e o provider gravados);
  - `boot_status(state: StackState, runner: Runner) -> str`.

- [ ] **Step 1: testes que falham:** `test_status_without_a_stack_exits_0`;
  `test_status_healthy_exits_0`; `test_one_endpoint_down_exits_1`;
  `test_outdated_pins_are_listed`; `test_config_points_here_is_reported_both_ways`;
  `test_status_on_a_corrupt_state_exits_1_with_the_fix`;
  `test_up_without_upgrade_repeats_the_recorded_images`;
  `test_up_upgrade_takes_the_catalogue_and_overrides_and_pulls`;
  `test_the_minor_guard_refuses_a_skip_and_allows_the_next` (1.19 -> 1.21 recusado com a minor
  intermediária na correção; 1.19 -> 1.20 aceito);
  `test_down_keeps_volume_and_models_and_marks_stopped`;
  `test_remove_deletes_files_and_keeps_models_and_data_by_default`;
  `test_purge_data_requires_the_project_name_or_yes` (resposta errada: nada apagado);
  `test_remove_never_touches_the_config_and_says_what_still_points_here`;
  `test_remove_cleans_even_with_a_corrupt_state` (tenta o `down` de cada runtime descoberto e
  apaga os arquivos);
  `test_a_recorded_provider_that_is_gone_is_an_error_with_a_fix`;
  `test_boot_status_for_podman_reads_the_unit_and_linger`.
- [ ] **Step 2:** rodar; esperado `No module named 'stack.lifecycle'`.
- [ ] **Step 3:** implementar. `--purge-data` usa `compose down -v`.
- [ ] **Step 4:** passam.
- [ ] **Step 5:** commit `feat(stack): status, up, down and remove`.

### Task 12: a ligação com o `qctx`

**Files:**
- Create: `stack/cli.py`
- Modify: `cli/qctx.py` (`build_parser`, `cmd_install`, `STACK_BUDGETS`, `_stack_section`)
- Modify: `tests/test_cli_install.py` (os construtores de `args` ganham `stack=None,
  runtime=None, image=[]`; o teste em processo de `--check` passa a substituir `_stack_section`
  como já substitui `_plumbing`)
- Test: `tests/test_stack_cli.py`

**Interfaces:**
- Consumes: `installer.provision`, `lifecycle.*`, `state.stack_dir`, `Budget`.
- Produces (`stack/cli.py`): `register(sub, *, ask: Callable[[str], str]) -> None` (grupo
  `stack` com `status`, `up [--upgrade] [--image ROLE=REF]...`, `down`,
  `remove [--purge-models] [--purge-data] [--yes]`); `add_install_flags(parser) -> None`
  (`--stack {auto,cpu,amd,intel,nvidia,apple}`, `--runtime {docker,podman}`, `--image ROLE=REF`
  com `append`); `install_step(args, report: dict, *, budgets: list[Budget], ask,
  interactive: bool) -> None`; `check_section(env=os.environ) -> dict` (lê estado e saúde, nunca
  runtime); `section_lines(section: dict) -> list[str]`; `TerminalPrompter`, `TerminalReporter`,
  `CoreConfigSink`. Nenhum import de `installer`, `lifecycle`, `runtimes` no nível do módulo: o
  `qctx statusline` roda a cada mensagem e monta o parser.
- Produces (`cli/qctx.py`): `STACK_BUDGETS = (Budget("claude-code", 8.0, 6.0), Budget("hermes",
  2.0, 2.0))`; `_stack_section() -> dict`.

`install_step`, pela seção "Quando aparece" da spec: plataforma Windows, uma linha e volta;
stack gerenciada em `running`, o resumo do `status`, e parada, `confirm` de religar (sim com
`--yes`), pins defasados, a dica de `qctx stack up --upgrade`; `stack.json` em outra fase, diz
que retoma e provisiona; sem stack: `--stack` provisiona; `--yes` sem `--stack` imprime
`qctx install --yes --stack auto` e volta; Qdrant ou Embedding entre os bloqueios do `report`,
explica (o que baixa, MiB, portas, diretório) e pergunta `y/N`; os endpoints respondendo, uma
linha dizendo que a stack local não é necessária. `StackError` sobe até o `main()`.

- [ ] **Step 1: testes que falham** (subprocess com `hermetic_env`, `QCTX_STACK_DIR` num tmp e um
  PATH com `docker` e `podman` falsos que só gravam a invocação num arquivo):
  - `test_stack_status_without_a_stack_exits_0` e `test_stack_status_json`;
  - `test_stack_help_lists_the_four_commands`; `test_install_help_shows_the_three_flags`;
  - `test_install_yes_alone_prints_the_auto_line_and_runs_no_runtime` (arquivo de invocações
    vazio);
  - `test_install_check_json_carries_the_stack_key` e
    `test_install_check_writes_nothing_in_the_stack_dir`;
  - `test_install_stack_with_no_runtime_exits_1_naming_the_fix`;
  - `test_an_interrupted_install_provisions_again` (em processo, `stack.json` em `compose` e
    `provision` substituído: é chamado);
  - `test_stack_budgets_match_the_hosts`: AST de `hooks/recall.py` (`EMBED_TIMEOUT_S`,
    `RERANK_TIMEOUT_S`) e de `hosts/hermes/__init__.py` (`HERMES_PREFETCH_BUDGET_S` e o divisor
    `4.0` da atribuição `share = ...`) contra `STACK_BUDGETS`;
  - `test_the_parser_does_not_import_the_heavy_modules`: subprocess que importa `cli/qctx.py`,
    monta o parser e confere que `stack.installer`, `stack.lifecycle` e `stack.runtimes` não
    estão em `sys.modules`.
- [ ] **Step 2:** rodar; esperado falha de import ou `invalid choice: 'stack'`.
- [ ] **Step 3:** implementar. O `cmd_install` mostra a seção `local stack:` no relatório de todo
  modo, põe a chave `stack` no `--json` e chama `install_step` logo depois do launcher. O
  `ready` não muda: a stack é opcional.
- [ ] **Step 4:** os testes novos e `tests.test_cli_install` passam.
- [ ] **Step 5:** commit `feat(stack): qctx stack, and the step in qctx install`.

### Task 13: documentação

**Files:**
- Modify: `README.md`, `docs/usage.md`, `docs/install.md`, `docs/architecture.md`
- Test: `tests/test_readme_fidelity.py` (classe nova)

**Interfaces:**
- Consumes: `BACKENDS`, `runtime_label`.

- [ ] **Step 1: teste que falha:** `TheCompatibilityTableIsTrue.test_install_doc_table_matches_runtimes`
  lê a tabela da seção nova do `docs/install.md` (`| platform | option | Docker | Podman | label |
  still needs |`) e confere cada linha de Linux e macOS contra `BACKENDS[...].runtimes(...)` e
  `runtime_label`; `test_every_backend_has_a_row` cobre o outro sentido.
- [ ] **Step 2:** rodar; esperado: a seção não existe.
- [ ] **Step 3:** escrever:
  - README, em `## Install, by OS`, depois da frase "The wizard is the same on every OS" e antes
    de `**Linux**`: uma linha dizendo que, com Docker ou Podman, o wizard sobe a infraestrutura
    sozinho com `qctx install --stack auto`;
  - README, em `## Local models`: `### Or let the wizard do it`, com `qctx install --stack auto` e
    o que ele faz em três linhas. O caminho manual fica como está (os dois defeitos dele são outra
    mudança, ver a spec);
  - `docs/usage.md`: o grupo `qctx stack` e as flags `--stack`, `--runtime` e `--image`;
  - `docs/install.md`: seção `## The local stack`: as etapas, o custo medido (294 MiB da imagem do
    llama.cpp, ~71 MiB do Qdrant, 836 MiB de modelos; RSS em CPU ~1,8 GB e ~2,8 GB; em GPU ~200
    MB por container), as portas, onde ficam os dados (diretório e `volume_name`), o reboot por
    runtime, a tabela de compatibilidade, o Windows na fase 3 e a armadilha do ambiente;
  - `docs/architecture.md`: `stack/` no Layout e a regra de fronteira.
- [ ] **Step 4:** `TMPDIR=/tmp python3 -m unittest tests.test_readme_fidelity
  tests.test_installable_from_git -v` passa (comandos citados existem, sem travessão longo,
  `http://` só localhost).
- [ ] **Step 5:** commit `docs: the local stack, its commands and where each option runs`.

### Task 14: integração opt-in, rodada nesta máquina

**Files:**
- Create: `tests/test_stack_integration.py`

**Interfaces:**
- Consumes: as implementações concretas de `stack/cli.py` e os casos de uso.

Pulada sem `QCTX_STACK_IT=1`. Projeto `memories-plugin-it`; portas de `QCTX_STACK_IT_PORTS`
(padrão `46333,48003,48004`); diretório num tmp de `/tmp`; config num tmp (`ConfigSink` sobre um
arquivo próprio); `QCTX_STACK_IT_MODELS=<dir>` semeia os GGUFs por hardlink ou cópia para não
baixar de novo; `QCTX_STACK_IT_RUNTIME` escolhe o runtime.

- [ ] **Step 1:** `test_cpu_profile_end_to_end`: `provision(profile="cpu", yes=True)` -> `running`;
  `status` saída 0; `down` -> `status` 1; `up` -> 0; `remove(purge_models=True,
  purge_data=True, yes=True)` -> nenhum container `memories-plugin-it-*` e nenhum volume
  `memories-plugin-it_memories-plugin-qdrant` no engine.
- [ ] **Step 2:** `test_gpu_profile_end_to_end` com `QCTX_STACK_IT_GPU=amd|intel|nvidia`: o perfil
  gravado é o pedido, o device começa com `Vulkan` e o rerank medido na calibração fica abaixo
  do da CPU medido no passo anterior (quando os dois rodam na mesma execução).
- [ ] **Step 3:** rodar aqui, nesta ordem, e anotar tempos e memória:
  `QCTX_STACK_IT=1 QCTX_STACK_IT_RUNTIME=podman TMPDIR=/tmp python3 -m unittest
  tests.test_stack_integration -v` (socket parado: provider `podman-compose`), depois com
  `QCTX_STACK_IT_GPU=intel`, depois de novo com o `podman.socket` do usuário iniciado só para o
  teste (provider docker-compose) e parado em seguida.
- [ ] **Step 4:** commit `test(stack): opt-in integration against a real runtime`.

### Task 15: verificação final

- [ ] **Step 1:** suíte inteira: `TMPDIR=/tmp python3 -m unittest discover -s tests`. Esperado:
  OK, com 1931 mais os testes novos; nenhuma falha nova contra a linha de base de `af124e8`.
- [ ] **Step 2:** o artefato, ponta a ponta, pelo `qctx` de verdade, num sandbox: `HOME` num
  scratch, `XDG_DATA_HOME` e `XDG_CONFIG_HOME` reais (o storage e o `containers.conf` do Podman),
  `QCTX_CONFIG` e `QCTX_STACK_DIR` num scratch, PATH sem `hermes` e `claude`, e as variáveis
  legadas do shell mantidas para exercitar a armadilha. `qctx install --yes --stack auto`:
  esperado GPU Intel escolhida, 8003 e 8004 movidas para 18003 e 18004 (ocupadas pela produção
  desta máquina), verificação passando, calibração dentro dos dois orçamentos, config gravado com
  `/v1` e os avisos de `QDRANT_URL` e `SERVER_BASE_URL`. Depois `qctx stack status` (0),
  `qctx stack down` (status 1), `qctx stack up` (0), `qctx stack remove --purge-models
  --purge-data --yes`, e uma rodada interativa com `--stack cpu` pelo
  `QCTX_INSTALL_FORCE_TTY` (a RSS medida confirma CPU: ~1,8 GB e ~2,8 GB).
- [ ] **Step 3:** o tempo do `qctx statusline` antes e depois (mediana de 20), para provar que o
  parser não ficou mais lento.
- [ ] **Step 4:** o scanner do hermes sobre um export limpo (`git archive HEAD | tar -x`), com
  `tools.plugin_guard.scan_plugin`: veredito `caution`, zero `critical`, e cada `high` novo lido.
- [ ] **Step 5:** a máquina como foi encontrada: nenhum container, volume ou rede `memories-plugin*`
  de teste, o `podman.socket` no estado de antes.

## Self-review

Rodado antes do commit deste plano:

- `grep -nE "TBD|TODO|FIXME|implement later|appropriate error|Similar to Task"` neste arquivo:
  limpo.
- Cobertura da spec, seção a seção da fase 1: catálogo (T2), detecção (T3, T4), perfis e
  compatibilidade (T5), compose e fixtures (T6), download e barra (T7), verificação e calibração
  (T8), `stack.json` (T9), a etapa e suas flags (T10, T12), ciclo de vida e guarda de minor (T11),
  erros (`StackError` em toda etapa, logs no `up` falho), testes (cada item da lista da spec tem
  dono), documentação (T13), integração opt-in (T14), verificações de abertura (feitas; T14 e
  T15 repetem o que precisa de código). Fora: modo servidor, chaves e `connect` (fase 2);
  Windows e `llama-dzn` (fase 3).
- Consistência de nomes entre tarefas: `server_command`, `Plan`, `dump`, `volume_name`,
  `container_name`, `stack_urls`, `Budget`, `Option`, `default_option`, `ProviderInfo`,
  `StackState`, `plan_of` aparecem com a mesma assinatura em quem produz e em quem consome.
- Valores contra a spec: imagens, digests, revisões, bytes, sha256, portas, `+10000`, `-c/-b/-ub
  8192`, 836 MiB, 600 s de prontidão, 1800 s de pull, orçamentos 8,0/6,0 e 2,0/2,0, `TOP_K` 20,
  6000 e 2400 caracteres, as 9 fixtures. As diferenças são as oito medições do topo, e a spec já
  foi corrigida nelas.
