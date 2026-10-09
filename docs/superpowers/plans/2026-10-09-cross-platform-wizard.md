# O wizard da stack em todas as plataformas: plano de implementação

> **Para quem executa:** SUB-SKILL OBRIGATÓRIO: use `superpowers:subagent-driven-development`
> (recomendado) ou `superpowers:executing-plans` para implementar tarefa por tarefa. Os
> passos usam caixinha (`- [ ]`) para rastrear.

**Meta:** fazer a jornada única de instalação (verifica runtime, verifica hardware,
mostra as opções, baixa no fim, termina com resumo) rodar no Windows via WSL2, sem
mudar a jornada das outras plataformas.

**Arquitetura:** a spine é a plataforma. O wizard já detecta WSL2 (`platform_of` já
devolve `windows` para ele; `facts.py:318`); o que muda é (a) a recusa de uma linha
passa a valer só para o Windows **nativo** (sem WSL2), (b) o backend de GPU do
Windows é o `dzn` (imagem própria, `/dev/dxg`) ao lado do `cpu` (imagem oficial,
como no Linux), (c) a imagem `llama-dzn` ganha pin no catálogo com fallback de build
local, (d) a verificação numérica do dzn roda quando o perfil dzn está no ar
(depois do download, antes de gravar o config), (e) a entrada `install.ps1` só
verifica (WSL2, runtime na distro, python3) e delega ao mesmo `install.sh` dentro da
distro, (f) um resumo final consolidado aparece em todas as plataformas. Nada dos 12
passos muda de assinatura.

**Stack:** Python 3 (stdlib), PowerShell 7 (`.ps1`), Docker/podman compose, GitHub
Actions (workflow da dzn).

**Spec:** `docs/superpowers/specs/2026-10-09-cross-platform-wizard-design.md`
(decisões 17 a 24) + a spec-mestra `docs/superpowers/specs/2026-10-05-local-stack-design.md`
(seção "Imagem própria e GPU no Windows", que define a dzn, o Dockerfile e a
verificação numérica).

**O que fecha nesta sessão (Linux, sem `pwsh`/`wsl`/GPU D3D12):** Tasks 1 a 8, 10 e
11 (a lógica do wizard, o resumo, o Dockerfile/workflow como artefato, a doc). A Task
9 fecha as funções puras **se** o `pwsh` instalar neste host; o corpo imperativo do
`.ps1` e a Task 12 (spikes) exigem máquina Windows+GPU de verdade. **Nada se finge:**
o que não roda aqui fica registrado como não-verificado, nunca como verde.

## Restrições globais

- Código, comentários, nomes de teste e mensagens de commit em **inglês**. Docs em
  **pt-BR** (a spec de 2026-10-09 é a referência de tom).
- Higiene de doc: sem em dash/en dash/hyphen não-ASCII; sem caminho do HOME real; sem
  identificador de máquina. Checar com
  `grep -nP '\x{2014}|\x{2013}|\x{2011}|\x{2012}|\x{2010}|\x{2015}|light-server|ts\.net|/home/' <doc>`.
- **Uma claim deve ser verdadeira.** Docstring, comentário ou mensagem que afirma algo
  falso é defeito. Comentários e docstrings que descrevem a recusa do Windows
  (`_windows_line`, `backends`, `catalog`) são atualizados **na mesma tarefa** que
  muda o comportamento.
- Um commit por preocupação independente. Mensagem via arquivo (`git commit -F`);
  nunca backtick em `-m`/`sed`.
- Suíte: `TMPDIR=/tmp /usr/bin/python3 -m unittest discover -s tests`. **O typo
  `TMPDIR=/timeout=0` gera 4 falsas falhas** (registrado): conferir o valor antes de
  rodar.
- Antes de um run focal, limpar `__pycache__`:
  `find stack tests core cli hosts -name __pycache__ -type d -prune -exec rm -rf {} +`.
- **NUNCA** rodar `rm -rf`/`find -delete` em alvo que não seja `__pycache__` ou o
  próprio `/tmp`/scratch. `.superpowers/` é gitignored e irrecuperável.
- Test-first: RED pelo motivo certo, depois GREEN; mutação para provar que cada
  defesa nova é coberta.
- Revisões em clones descartáveis próprios, nunca na árvore real.
- **Nunca** tocar no provider vivo (`~/.hermes/plugins/memories`) nem no Qdrant
  externo. As coleções `claude_memory`/`memories_docs_*`/`memories_repos*` são
  storage, nunca renomeadas.

## Foco de revisão

Seis classes de entrada que a spec implica mas nenhum teste da jornada exercita, na
ordem de probabilidade de morder um usuário real; cada linha ganha seu teste na
tarefa que dona o código:

1. **Windows nativo, sem WSL2** (`system()=="Windows"`, correndo o `qctx` à mão): o
   usuário espera a recusa de uma linha apontando para o manual, não um crash e não
   o menu. -> Task 1.
2. **WSL2 presente, mas nenhum runtime responde** (Docker Desktop desligado, ou
   integração WSL desmarcada, ou podman ausente): o usuário espera o abort
   `step="runtime"` nomedando `docker` e `podman`, não a recusa de plataforma. ->
   Task 1.
3. **WSL2 com runtime, sem GPU que o dzn liste** (`/dev/dxg` ausente, ou o
   `--list-devices` da prova do menu não lista adapter D3D12): o item `dzn` sai do
   menu com o motivo, e o `cpu` segue oferecido. -> Task 3 e Task 7.
4. **Pin da dzn não publicado no GHCR** (pacote privado ainda, ou primeiro uso): o
   install faz `docker build` do Dockerfile local e segue, sem ficar bloqueado. ->
   Task 4.
5. **RAM lida no WSL2**: o número do orçamento deve ser o da VM do **engine**
   (limite do Docker Desktop), não o da distro nem o do host Windows; e o resumo
   final nomeia isso. -> Task 5 e Task 8.
6. **Perfil dzn no ar, mas a verificação numérica falha** (a ordem das similaridades
   ou do rerank não bate entre o device D3D12 e o mesmo perfil sem device): o install
   aborta **antes de gravar o config** e oferece o CPU com o motivo (spec-mestra,
   "Se falhar, oferece CPU e diz por quê"). -> Task 6.

---

### Task 1: a plataforma WSL2 com runtime vira `windows`; a recusa de uma linha fica para o nativo

**Arquivos:**
- Modify: `stack/facts.py` (nova `is_native_windows`)
- Modify: `stack/cli.py` (o gate de uma linha em `install_step` + o comentário do ramo)
- Modify: `stack/installer.py` (`_check_platform`)
- Test: `tests/test_stack_facts.py`, `tests/test_stack_cli.py`, `tests/test_stack_installer.py`

**Interfaces:**
- Produz: `facts.is_native_windows(probe: Probe) -> bool` (True só quando
  `probe.system().strip().lower() == "windows"`, independentemente do kernel).
- O gate do `install_step` passa a usar `is_native_windows` (hoje
  `is_windows_host`); `_check_platform` aceita `("linux", "macos", "windows")`.
- `platform_of` **não muda** (já devolve `windows` para WSL2, `facts.py:318`).
- Consome: o existente `facts.Probe`.

- [ ] **Step 1: teste do `is_native_windows`**

```python
def test_is_native_windows_is_only_the_system_name(self):
    # system()=="Windows" (kernel sem "microsoft") -> True
    # system()=="Linux" + osrelease "microsoft" (WSL2) -> False
    # system()=="Linux" normal -> False
```

- [ ] **Step 2: rodar e confirmar FAIL** (`is_native_windows` não existe).

- [ ] **Step 3: implementar `is_native_windows` em `stack/facts.py`**

```python
def is_native_windows(probe: Probe) -> bool:
    """Native Windows (no WSL): the system name says Windows. The wizard runs here
    only through python, so there is no bash, no container runtime and no stack
    path: the one-line refusal is its answer. WSL2 reports system()=='Linux', so it
    is False here and is classified by `platform_of` (and by the runtime discovery)."""
    return probe.system().strip().lower() == "windows"
```

- [ ] **Step 4: trocar o gate em `stack/cli.py`** — de
  `if facts.is_windows_host(facts.Probe()):` para
  `if facts.is_native_windows(facts.Probe()):`, e atualizar o comentário do ramo
  (o texto atual afirma que a recusa vale para WSL incluído; isso fica falso).

- [ ] **Step 5: testar o gate** (no `tests/test_stack_cli.py`)
  - (Foco 1) Windows nativo -> imprime a linha de recusa e retorna.
  - (Foco 2) WSL2 + nenhum runtime -> o abort é `step="runtime"` nomedando docker
    e podman, não a recusa de plataforma.
  - WSL2 + runtime que responde -> o passo segue (não termina na linha de recusa).

- [ ] **Step 6: `_check_platform` aceita `windows`** em `stack/installer.py`
  (trocar `("linux", "macos")` por `("linux", "macos", "windows")`); atualizar a
  docstring (afirma "phase 3"); teste: plataforma `windows` devolve `"windows"` em
  vez de levantar.

- [ ] **Step 7: rodar a suíte focal** (`TMPDIR=/tmp`), confirmar GREEN.

- [ ] **Step 8: commit**

```bash
git commit -F <msg>   # feat(stack): WSL2 with a runtime is the windows platform; the one-liner is native-only
```

---

### Task 2: o backend `cpu` roda no Windows (imagem oficial, sem device)

**Arquivos:**
- Modify: `stack/backends.py` (`_PHASE1_PLATFORMS` -> `_SUPPORTED_PLATFORMS`; `Cpu`)
- Test: `tests/test_stack_backends.py`

**Interfaces:**
- Produz: `Cpu.runtimes("windows") -> frozenset({"docker","podman"})`;
  `Cpu.availability("windows", ...) -> READY`; `Cpu.service_patch` continua `{}`
  (a imagem oficial, `-dev none` — o `cpu` no WSL2 não usa a dzn, spec de
  2026-10-09, risco registrado).
- A constante vira `_SUPPORTED_PLATFORMS = ("linux", "macos", "windows")`; só a
  `Cpu` a consome (verificado: os demais backends comparam a plataforma
  diretamente).
- Consome: nada de novo.

- [ ] **Step 1: teste** — `Cpu().runtimes("windows")` == {docker,podman};
  `Cpu().availability("windows", facts, engine, "docker")` tem state `READY`;
  `Cpu().service_patch("docker", None)` == {}; e `runtimes("linux")`/`"macos"`
  inalterados.

- [ ] **Step 2: rodar e confirmar FAIL** (hoje `runtimes("windows")` == frozenset()).

- [ ] **Step 3: implementar** — renomear a constante, adicionar `windows`,
  atualizar o comentário (a recusa de uma linha agora mora no
  `is_native_windows`).

- [ ] **Step 4: rodar a suíte focal**, confirmar GREEN.

- [ ] **Step 5: commit**

```bash
git commit -F <msg>   # feat(stack): the cpu profile runs on windows (official image, no device)
```

---

### Task 3: o backend `dzn` (GPU no Windows, imagem própria, `/dev/dxg`)

**Arquivos:**
- Modify: `stack/backends.py` (nova classe `DznGpu`, entrada em `BACKENDS` e `GPU_PROFILES`)
- Test: `tests/test_stack_backends.py`

**Interfaces:**
- Produz: `class DznGpu` com `id="dzn"`, `experimental=False`, `vendor=None`,
  `image_role="llama-dzn"`:
  - `runtimes("windows") -> frozenset({"docker"})`; as outras plataformas,
    frozenset();
  - `availability(platform, facts, engine, runtime)`:
    plataforma != windows -> `UNSUPPORTED` ("dzn runs on windows only");
    runtime != docker -> `RUNTIME` com `needs="docker"`;
    senão `READY` (a presença da GPU é provada no passo de prova, não aqui);
  - `service_patch(runtime, gpu_index) -> {"devices": ["/dev/dxg"],
    "volumes": ["/usr/lib/wsl:/usr/lib/wsl"],
    "environment": ["LD_LIBRARY_PATH=/usr/lib/wsl/lib"]}`;
  - `devices_seen(output) -> [d for d in parse_devices(output)
    if "Microsoft Direct3D12" in d.name]` (o adapter real; `llvmpipe` não conta).
- Consome: `parse_devices` (re-exportado por `backends`), `Availability`,
  `READY/RUNTIME/UNSUPPORTED`.

- [ ] **Step 1: testes** (Foco 3)
  - `DznGpu().runtimes("windows")` == {docker}; `runtimes("linux")` == frozenset().
  - `availability` nos três ramos acima.
  - `service_patch("docker", None)` devolve exatamente o dict com `/dev/dxg`,
    `/usr/lib/wsl` e o `LD_LIBRARY_PATH`.
  - `devices_seen` com uma linha `ggml_vulkan ... uma: Vulkan1: Microsoft Direct3D12 (NVIDIA GeForce RTX 4090) ...`
    devolve 1 device; com `llvmpipe (LLVM 15)` devolve 0; o `vendor_of` do nome
    resolve `nvidia` do adapter (token `NVIDIA` entre parênteses).

- [ ] **Step 2: rodar e confirmar FAIL** (classe não existe).

- [ ] **Step 3: implementar** `DznGpu` e registrar em `BACKENDS["dzn"]`;
  `dzn` entra no fim da tupla `GPU_PROFILES` (a ordem do menu: `cpu` primeiro,
  depois as GPUs).

- [ ] **Step 4: rodar a suíte focal**, confirmar GREEN.

- [ ] **Step 5: mutação** — apagar o filtro `"Microsoft Direct3D12" in d.name` de
  `devices_seen`; o teste do llvmpipe deve falhar (prova que o filtro é coberto).

- [ ] **Step 6: commit**

```bash
git commit -F <msg>   # feat(stack): the dzn gpu backend for windows (own image, /dev/dxg)
```

---

### Task 4: a imagem `llama-dzn` no catálogo, o `image_role` consumido, e o fallback de build local

**Arquivos:**
- Modify: `stack/catalog.py` (`LLAMA_DZN_IMAGE`, `IMAGES`, `IMAGE_ENV`)
- Modify: `stack/compose.py` (`_server` consome `image_role`)
- Modify: `stack/installer.py` (o passo de pull: atingibilidade do pin + build local)
- Test: `tests/test_stack_catalog.py`, `tests/test_stack_compose.py`, `tests/test_stack_installer.py`

**Interfaces:**
- Produz: `catalog.LLAMA_DZN_IMAGE = "ghcr.io/erickstryck/llama-dzn:b11382-mesa26.0.3"`
  (tag até o primeiro publish; o digest entra pelo bump, como o da llama, e a
  docstring do catálogo diz exatamente isso);
  `catalog.IMAGES` ganha `"llama-dzn": LLAMA_DZN_IMAGE`;
  `catalog.IMAGE_ENV` ganha `"llama-dzn": "QCTX_STACK_IMAGE_LLAMA_DZN"`.
- `_server` usa `plan.images[BACKENDS[plan.backend].image_role or "llama"]` (hoje
  hardcode `plan.images["llama"]`).
- Uma função pura no installer, `dzn_image_ref(reachable: bool, runner, ref) ->
  (ref, built: bool)`: `reachable` False -> roda
  `docker build -t <ref> images/llama-dzn/` e devolve `(ref, True)`.
- Efeito colateral **intencional** (não "corrigir"): com `llama-dzn` em `IMAGES`,
  todo install antigo (sem o role em `stack.json`) mostra a dica de upgrade
  `stack up --upgrade` uma vez, pela primeira vez que o catálogo moveu um pin
  além do gravado. É o comportamento documentado do `outdated_pins`.
- Consome: `BACKENDS[...].image_role`, `catalog.resolve_images`, o runner do compose.

- [ ] **Step 1: teste do catálogo** — `"llama-dzn" in IMAGES`;
  `parse_image_flags(["llama-dzn=REF"])` aceita o novo role;
  `resolve_images` devolve o ref da dzn pelo env `QCTX_STACK_IMAGE_LLAMA_DZN` e
  pela flag, com a precedência flag > env > catálogo.

- [ ] **Step 2: teste do compose** — um `Plan` com `backend="dzn"` renderiza o
  service com `image == plan.images["llama-dzn"]`; um `Plan` com `backend="cpu"`
  continua com `plan.images["llama"]`.

- [ ] **Step 3: teste do fallback** (Foco 4) — `dzn_image_ref(False, runner, ref)`
  com um runner fake registra o `docker build` e devolve `(ref, True)`;
  `dzn_image_ref(True, runner, ref)` não chama o runner.

- [ ] **Step 4: rodar e confirmar FAIL.**

- [ ] **Step 5: implementar** o pin no catálogo, o consumo de `image_role` em
  `_server`, e o passo de pull que confere a atingibilidade do pin (manifest GET)
  e, indisponível, delega a `dzn_image_ref` e reporta que usou o build local.

- [ ] **Step 6: rodar a suíte focal**, confirmar GREEN.

- [ ] **Step 7: mutação** — reverter `_server` para `plan.images["llama"]` fixo; o
  teste do compose (dzn usa o ref da dzn) deve falhar.

- [ ] **Step 8: commit**

```bash
git commit -F <msg>   # feat(stack): the llama-dzn image pin, the image_role seam, and the local-build fallback
```

---

### Task 5: a RAM no WSL2 vem do engine (VM do Docker Desktop), não do `/proc` da distro

**Arquivos:**
- Modify: `stack/installer.py` (`_check_disk_and_ram`)
- Test: `tests/test_stack_installer.py`

**Interfaces:**
- Modifica a escolha do número de RAM. Hoje:
  `ram = engine.memory_bytes if facts.system == "macos" else facts.ram_bytes`.
  Novo: `ram = engine.memory_bytes if (facts.system == "macos" or platform == "windows")
  else facts.ram_bytes`, onde `platform = ctx.platform` (**não** `facts.system`,
  que no WSL2 é `linux`). `ctx.platform` é definido antes (`provision` roda
  `_check_platform` antes de `_check_disk_and_ram`).
- Consome: `EngineInfo.memory_bytes` (o docker publica `MemTotal`, o podman
  `memTotal` — verificado em `stack/docker.py:59` / `stack/podman.py:75`).

- [ ] **Step 1: teste** (Foco 5) — facts.system=="linux" + `ctx.platform`=="windows"
  + `engine.memory_bytes` pequeno + `facts.ram_bytes` grande: o warn de RAM usa o
  número do **engine** (o da VM), não o `MemAvailable` da distro. E
  platform=="linux": continua usando `facts.ram_bytes` (regressão).

- [ ] **Step 2: rodar e confirmar FAIL.**

- [ ] **Step 3: implementar** a troca da condição para a plataforma.

- [ ] **Step 4: rodar a suíte focal**, confirmar GREEN.

- [ ] **Step 5: commit**

```bash
git commit -F <msg>   # fix(stack): the WSL2 ram figure is the engine's VM, not the distro /proc
```

---

### Task 6: a verificação numérica do dzn (ordem das similaridades e do rerank)

**Arquivos:**
- Modify: `stack/verify.py` (nova `numerical_compare` + o corpus reutilizado da calibração)
- Modify: `stack/installer.py` (novo passo `_verify_numerical`, chamado quando o
  backend é `dzn`, depois do `_verify_functional` e antes de gravar o config)
- Test: `tests/test_stack_verify.py`, `tests/test_stack_installer.py`

**Interfaces:**
- Produz: `verify.numerical_compare(gpu: dict, cpu: dict) -> tuple[bool, str]`,
  função pura. Cada entrada é `{"sim": list[float], "rank": list[int]}` para o
  **corpus fixo da calibração** (dono único, já em `verify.py`; não inventar textos
  novos). As três checagens da spec-mestra:
  1. a **ordem** das similaridades bate (a ordem dos índices por `sim` é a mesma);
  2. o **desvio** `max |gpu.sim[i] - cpu.sim[i]|` (índice a índice) é menor que o
     **menor intervalo** entre similaridades adjacentes na ordem (o desvio é
     pequeno **frente aos intervalos que decidem os cortes**, não um limiar
     absoluto de cosseno);
  3. a **ordem** do rerank bate (`rank` igual).
  `reason` nomeia qual das três falhou (ou `""` se passou).
- O passo `_verify_numerical(ctx, plan)`: só roda quando `plan.backend == "dzn"`.
  O lado `cpu` é um container **descartável** do mesmo perfil sem device (a dzn
  roda CPU, é a oficial + o driver): `compose run` no probe file sem os mounts de
  WSL2 (mesmo mecanismo dos probes do `_prove`), num porto efêmero, consultado com
  o corpus e derrubado. O lado `gpu` é a stack no ar (os portos do `plan`).
  Falhou -> `StackError` com `step="verify"` e `fix` dizendo o motivo e
  `re-run the install with --stack cpu`; nada é gravado (o config vem depois).
  Passou -> uma linha `ok` com o desvio máximo medido.
- Consome: `verify.calibrate`'s corpus, o runner do compose, o http existente.

- [ ] **Step 1: testes puros da `numerical_compare`** (Foco 6)
  - ordem igual + desvio menor que o menor intervalo -> `(True, "")`.
  - dois textos adjacentes trocados de ordem -> `(False, reason nomeda "order")`.
  - mesma ordem, mas um desvio maior que o menor intervalo -> `(False, reason
    nomeda o desvio e o intervalo).
  - `sim` idêntico, `rank` com dois índices trocados -> `(False, reason nomeda
    "rerank")`.
  - corpus com dois valores iguais (intervalo zero): o menor intervalo é 0 e
    qualquer desvio > 0 falha (o caso degenerado é conservador).

- [ ] **Step 2: teste do passo no installer**
  - `plan.backend == "dzn"` + `numerical_compare` injetada falhando ->
    `StackError` `step="verify"`, `fix` contendo `--stack cpu`; o config **não é
    gravado** (o `_save_config` não é chamado).
  - `plan.backend == "cpu"` -> o `numerical_compare` injetado (espionado) **nunca
    é chamado**.
  - `plan.backend == "dzn"` + passando -> o install termina, config gravado.

- [ ] **Step 3: rodar e confirmar FAIL.**

- [ ] **Step 4: implementar** `numerical_compare` em `verify.py` e o passo
  `_verify_numerical` no installer (incluindo o probe descartável sem device).

- [ ] **Step 5: rodar a suíte focal**, confirmar GREEN.

- [ ] **Step 6: mutação** — trocar a checagem 2 por um limiar absoluto de cosseno
  (ex.: `> 0.1`); o teste "desvio maior que o menor intervalo" deve continuar
  passando **e** um novo caso com desvio 0.05 e intervalo 0.02 (absoluto < 0.1,
  relativo > intervalo) deve falhar — prova que a checagem é relativa, como a
  spec-mestra exige.

- [ ] **Step 7: commit**

```bash
git commit -F <msg>   # feat(stack): the dzn numerical check (similarity and rerank order vs the no-device profile)
```

---

### Task 7: os fixtures do Windows no compose

**Arquivos:**
- Create (gerados): `tests/fixtures/stack/windows-docker-cpu.yaml`,
  `tests/fixtures/stack/windows-docker-dzn.yaml`,
  `tests/fixtures/stack/windows-podman-cpu.yaml`
- Modify: `tests/test_stack_compose.py` (a matriz ganha a linha windows)
- Test: idem

**Interfaces:**
- Consome: o `render`/`dump` do compose, os backends `cpu`/`dzn` (Tasks 2/3), o
  pin da dzn (Task 4), o `Plan`.
- A prova do dzn **reusa** o `_prove` existente (o probe file é o compose do
  Windows, com a imagem dzn e `/dev/dxg`): nenhum novo código de prova — a fixture
  muda.

- [ ] **Step 1: adicionar a linha windows à matriz** do `test_stack_compose`
  (docker-cpu, docker-dzn, podman-cpu) e rodar
  `python3 tests/test_stack_compose.py --regen` para gerar os fixtures.

- [ ] **Step 2: conferir `windows-docker-dzn.yaml`** — o service `embed` (e
  `rerank`) carrega `image: ghcr.io/erickstryck/llama-dzn:b11382-mesa26.0.3`,
  `devices: [/dev/dxg]`, `volumes: [/usr/lib/wsl:/usr/lib/wsl]`,
  `environment: [LD_LIBRARY_PATH=/usr/lib/wsl/lib]`, e a porta
  `127.0.0.1:<p>:8080`. `windows-docker-cpu.yaml` é idêntico a
  `linux-docker-cpu.yaml` (imagem oficial, sem device); `windows-podman-cpu.yaml`
  é idêntico a `windows-docker-cpu.yaml` (o patch do cpu é vazio, verificado: os
  fixtures linux docker/podman cpu são byte a byte iguais).

- [ ] **Step 3: rodar a suíte focal** (`--regen` não deve mais mudar nada; o
  golden é estável).

- [ ] **Step 4: commit**

```bash
git commit -F <msg>   # test(stack): the windows compose fixtures (dzn and cpu)
```

---

### Task 8: o resumo final, em todas as plataformas

**Arquivos:**
- Create: `stack/summary.py`
- Modify: `stack/installer.py` (`_finish` chama o resumo)
- Test: `tests/test_stack_summary.py`

**Interfaces:**
- Produz: `summary.render_summary(saved_config, plan: compose.Plan) -> list[str]`
  (função pura; devolve as linhas; o `_finish` as imprime). Bloco:
  `ok stack: running (<runtime>, <backend>)`; as três URLs lidas do config salvo
  (`qdrant_url`, `api_base_url`, `rerank_url`); a api-key:
  `saved_config.qdrant_api_key` vazio -> `api-key (nenhuma: Qdrant local sem
  chave)`, senão a chave; e, quando `plan.platform == "windows"`, a linha que
  nomeia que a RAM medida é a da VM do engine, não do host Windows (Foco 5).
- `_finish` **re-lê o config do disco** antes de renderizar (o resumo mostra o
  que está no disco, não o que a memória do processo acha): quando o usuário
  recusou a gravação, o resumo mostra o config antigo e as URLs do `plan`.
- Consome: o `core.config` salvo (os campos citados existem: `core/config.py:99-104`),
  o `Plan`.

- [ ] **Step 1: testes**
  - Sem api-key: a linha é `(nenhuma: Qdrant local sem chave)`.
  - Com api-key: a linha mostra a chave do config.
  - As três URLs batem com os campos do config/plan.
  - `platform == "windows"`: a linha da VM do engine aparece; `platform ==
    "linux"`: não aparece.
  - O resumo lê o **config passado** (não a memória): trocar o objeto troca a saída.

- [ ] **Step 2: rodar e confirmar FAIL** (módulo não existe).

- [ ] **Step 3: implementar** `stack/summary.py` e o chamamento em `_finish`
  (antes das linhas de `reboot_hint`, que se mantêm).

- [ ] **Step 4: rodar a suíte focal**, confirmar GREEN.

- [ ] **Step 5: commit**

```bash
git commit -F <msg>   # feat(stack): the final summary (what was done, the URLs, the default api key)
```

---

### Task 9: a entrada `install.ps1` (verifica WSL2/runtime/python3, delega; `-Command`)

**Arquivos:**
- Create: `scripts/install.ps1`
- Create: `tests/wsl_gateway.Tests.ps1`
- Test: idem (roda com `pwsh`)

**Interfaces:**
- O `.ps1` não decide nada que não seja puro e testável. Funções puras:
  - `ConvertTo-DistroTable([string]$wslListView) -> [pscustomobject[]]` (parseia
    `wsl -l -v`: nome + estado).
  - `Select-Distro($table, [string]$explicit) -> [string]` (o `$explicit` vence;
    senão o primeiro estado `Running` ou `Stopped`, **nunca** `Not Installed`;
    nenhum, devolve `""`).
  - `Test-Gateway([bool]$wslPresent, [string]$distro, [bool]$dockerOk,
    [bool]$podmanOk, [bool]$python3Ok) -> [pscustomobject]` com `.Action`
    (`"abort"` | `"install-python"` | `"delegate"`) e `.Message`/`.Fix`.
  - `Resolve-Command([string]$command) -> [string[]]` (decisão 24b): mapeia
    `status`/`up`/`down`/`remove` para `qctx stack <cmd>` e recusa qualquer outra
    entrada (devolve `$null`).
- O corpo imperativo: `where.exe wsl` (ausente -> abort com `wsl --install`,
  admin + reboot, **não** auto-instala); `wsl -l -v` -> parse -> select; sondas na
  distro (`wsl -d <d> -- sh -lc 'command -v docker >/dev/null 2>&1 && docker info
  >/dev/null 2>&1'` e análogo para podman e python3); `Test-Gateway`; e então:
  abort (dependência + o comando que instala + o aviso de que o install não
  instala por ele), ou `install-python` (apt com o sim), ou o delegue:
  - sem `-Command`: `wsl -d <d> -- bash <clone>/scripts/install.sh <args…>`;
  - com `-Command`: `wsl -d <d> -- bash -lc 'qctx stack <cmd>'` (o launcher na
    home da distro, decisão 24).
  O exit code do processo na distro é o exit code do script (Foco 1/2).

- [ ] **Step 1: tentar instalar `pwsh` neste host** (única vez) para rodar os
  testes: `apt-get install -y powershell` (ou o canal Microsoft); confirmar
  `pwsh --version`. **Fallback honesto:** se não instalar, a Task 9 para nos
  passos de teste escrito, e o commit registra que a verificação PowerShell ficou
  pendente do CI Windows — **não** rotular de verde o que não rodou.

- [ ] **Step 2: escrever `tests/wsl_gateway.Tests.ps1`** cobrindo (com strings de
  fixture, sem chamar wsl/docker de verdade):
  - `ConvertTo-DistroTable` com a saída real de `wsl -l -v` (vários distros, um
    `Not Installed`) -> as linhas certas.
  - `Select-Distro`: explícito vence; sem explícito, o primeiro `Running`/`Stopped`;
    só `Not Installed` -> `""`.
  - `Test-Gateway`: wsl ausente -> abort com `wsl --install`; wsl ok + sem docker e
    sem podman -> abort nomedando as causas; python3 ausente -> `install-python`;
    tudo ok -> `delegate`.
  - `Resolve-Command`: os quatro comandos -> `qctx stack <cmd>`; `"install"` ->
    `$null`.

- [ ] **Step 3: rodar os testes e confirmar FAIL** (funções ainda não existem).

- [ ] **Step 4: escrever `scripts/install.ps1`** com as funções puras no topo e o
  corpo imperativo depois.

- [ ] **Step 5: rodar os testes PowerShell**, confirmar GREEN.

- [ ] **Step 6: revisar o exit code** — o script propaga o exit do processo na
  distro (o abort do wizard não vira erro genérico do PowerShell).

- [ ] **Step 7: commit**

```bash
git commit -F <msg>   # feat(stack): the windows entry (install.ps1 verifies WSL2/runtime/python3 and delegates)
```

---

### Task 10: o Dockerfile da dzn + o workflow de publicação

**Arquivos:**
- Create: `images/llama-dzn/Dockerfile`
- Create: `.github/workflows/llama-dzn.yml`

**Interfaces:**
- O Dockerfile segue a spec-mestra ("A imagem"): estágio 1 build (Ubuntu 26.04,
  `meson`, `ninja`, `python3-mako`, `directx-headers-dev`; tarball do Mesa 26.0.3
  com **sha256 conferido**; compila só o dzn, `-Dvulkan-drivers=microsoft-experimental`,
  demais drivers/GL/LLVM desligados). Estágio 2: `FROM server-vulkan-b11382` pelo
  digest (o de `catalog.LLAMA_IMAGE`) + `libvulkan_dzn.so` + manifesto ICD do dzn.
  Só amd64.
- O workflow: disparo manual (build, digest da base, versão do Mesa); em PR que
  mexa no Dockerfile, só build, sem push; publica
  `ghcr.io/erickstryck/llama-dzn:b11382-mesa26.0.3` com atestado de proveniência;
  o digest publicado entra no catálogo pelo bump (Task 4). O pacote novo no GHCR
  nasce privado: torná-lo público é um passo manual, uma vez (o fallback de build
  local cobre o intervalo).

- [ ] **Step 1: obter o sha256 real** do tarball do Mesa 26.0.3 (baixar o tarball
  que o Dockerfile vai usar e `sha256sum`); registrar o valor no Dockerfile como
  `ARG MESA_SHA256=<valor real>`. **Nunca inventar o sha256.**

- [ ] **Step 2: escrever `images/llama-dzn/Dockerfile`** com os dois estágios e os
  pins (Mesa 26.0.3 + sha256 real, digest da base do catálogo).

- [ ] **Step 3: escrever `.github/workflows/llama-dzn.yml`** (disparo manual + PR
  build-only + publish com proveniência).

- [ ] **Step 4: validar a sintaxe** do workflow (`actionlint` se disponível, senão
  revisão manual) e a estrutura do Dockerfile. **Não** dá para buildar aqui
  (falta `directx-headers`; o alvo é o dzn do WSL2): registrar no commit que o
  build é gate do CI + do spike (Task 12).

- [ ] **Step 5: commit**

```bash
git commit -F <msg>   # feat(stack): the llama-dzn image (Dockerfile) and its publish workflow
```

---

### Task 11: a documentação (README, install.md)

**Arquivos:**
- Modify: `README.md` (bloco do Windows na instalação; a seção "Local models"
  mantém o papel de caminho sem plugin)
- Modify: `docs/install.md` (a seção standalone passa a dizer que o Windows entra
  pela jornada normal e remove "não está no caminho, nesta versão")

**Interfaces:**
- Consome: as decisões 17-24 da spec de 2026-10-09.
- Higiene: sem em dash, sem HOME real, sem identificador de máquina (o grep das
  restrições globais).

- [ ] **Step 1: README** — bloco "Windows": `.\install.ps1` (e a forma de primeira
  execução `powershell -ExecutionPolicy Bypass -File .\install.ps1`, que evita a
  recusa de política), as três verificações, que quem instala WSL2/Docker é o
  usuário, e `-Command status` para o dia a dia; a jornada de seis passos é a mesma.

- [ ] **Step 2: install.md** — atualizar a seção standalone (a escrita em
  `ad2bed7`) para refletir a reversão: o Windows roda pela jornada normal via
  WSL2; o manual "Local models" fica para quem não quer o plugin.

- [ ] **Step 3: checar higiene** (grep das restrições globais) e consistência com
  a spec de 2026-10-09 (especialmente: a recusa de uma linha agora vale só para o
  Windows nativo sem WSL2).

- [ ] **Step 4: rodar `tests/test_readme_fidelity`** (executa os comandos citados
  no README) e confirmar GREEN.

- [ ] **Step 5: commit**

```bash
git commit -F <msg>   # docs(stack): the windows entry in the README, and the standalone section reflects the reversal
```

---

### Task 12: os spikes de Windows (verificação externa; exige máquina Windows+GPU)

**Arquivos:** nenhum código; o registro vai para o ledger
(`.superpowers/sdd/.../progress.md`) e a memória.

- [ ] **Step 1 (Foco 3/6):** numa máquina Windows+GPU com Docker Desktop, rodar a
  jornada de ponta a ponta: `.\install.ps1 --stack auto`; confirmar que o menu
  lista o `dzn` **se e só se** a prova (`--list-devices` com `/dev/dxg`) lista o
  adapter D3D12; e a **verificação numérica** (Task 6) passando contra o mesmo
  perfil sem device.
- [ ] **Step 2 (Foco 2):** confirmar que o socket do Docker Desktop responde
  `docker info` dentro da distro WSL2 (a premissa do passo 2 do `.ps1`).
- [ ] **Step 3:** registrar o resultado (passou / o item dzn sai do menu com o
  motivo). **Não executável nesta sessão** (Linux sem WSL2/GPU D3D12): é a gate de
  aceite da jornada no Windows, e fica marcada como tal, nunca como feito.

---

## Ordem e dependências

- 1 → 2 → 3 (a plataforma antes dos backends; o `dzn` depois do `cpu`).
- 4 depende de 3 (o `image_role` consumido pelo `dzn`).
- 5 depende de 1 (`ctx.platform == "windows"` precisa ser alcançável).
- 6 depende de 3 e 4 (o perfil dzn e a imagem dele).
- 7 depende de 2, 3 e 4 (os fixtures renderizam o dzn e o cpu no windows).
- 8 é independente do resto, mas vem depois de 5 (a linha da VM do engine).
- 9 depende de 1-8 (o `.ps1` delega a um `install.sh` que já faz a jornada).
- 10 depende de 4 (o digest da base no estágio 2 vem do catálogo).
- 11 depende de 9 e 10 (a doc descreve o que existe).
- 12 é a gate externa final.

Fecháveis nesta sessão: 1, 2, 3, 4, 5, 6, 7, 8, 10, 11. A 9 fecha as funções puras
se o `pwsh` instalar; o corpo e a 12 exigem Windows.

Sequência de commits: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12.
