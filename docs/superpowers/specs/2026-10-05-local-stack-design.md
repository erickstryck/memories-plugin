# Stack de memória provisionada pelo `qctx` (local, servidor e GPU no Windows): desenho

Data: 2026-10-05, revista em 2026-10-06. Repo: `memories-plugin` @ `05ff84b` (v1.3.0).

Estado: as decisões da tabela "O que se decidiu" foram tomadas pelo usuário no brainstorming de
2026-10-05 e 2026-10-06. O que está marcado **[a validar]** é proposta minha, escrita a pedido do
usuário para ser validada direto neste documento, sem a revisão seção por seção que normalmente
viria antes. A spec foi aprovada em 2026-10-06 (`af124e8`).

Revisão de 2026-10-06, depois da aprovação: as verificações de abertura da fase 1 rodaram numa
máquina com GPU (ver "Resultado das verificações de abertura da fase 1", no fim). Sete detalhes
marcados **[a validar]** mudaram por medição, e cada um está corrigido no lugar onde a spec o
afirmava. Nenhuma decisão da tabela mudou.

## O problema

Hoje o plugin exige três endpoints de pé antes de qualquer coisa: um Qdrant, um servidor de
embedding e um de rerank. O README ensina a subi-los à mão, com três `docker run` e dois downloads
de GGUF. O wizard (`qctx install`) checa os três e, se não respondem, para em "stopping here".

A spec do wizard (2026-08-19) registrou isso como limite deliberado: "O que o wizard não faz, e
diz: subir um Qdrant ou um servidor de embedding". **Esta spec reverte essa linha, a pedido do
usuário.** Com Docker ou Podman instalado, o `qctx install` passa a oferecer subir os três
sozinho: detecta o SO e o hardware, deixa o usuário escolher onde rodar os modelos, baixa os
modelos e deixa o plugin configurado.

O pedido original falava em "uma adição, não uma edição dos fluxos existentes", e na rodada de
perguntas o usuário escolheu integrar como etapa do `qctx install`. As duas coisas convivem
assim: o subsistema é novo e isolado (pacote `stack/`), e o wizard ganha uma etapa que o chama.
As etapas que já existem não mudam de comportamento.

Em 2026-10-06 entraram mais duas necessidades. A primeira é GPU no Windows, onde a imagem oficial
não alcança a GPU de dentro do container. A segunda é instalar só a stack numa máquina à parte,
acessível pela rede, e apontar para ela o plugin das outras máquinas. É o que o usuário já faz à
mão hoje: embedding e rerank num host da tailnet, e o Qdrant atrás de um proxy HTTPS.

## O que se decidiu

| # | decisão | escolha | recusado |
|---|---|---|---|
| 1 | onde entra | etapa nova dentro do `qctx install`; a lógica num subsistema isolado | grupo `qctx stack` separado sem tocar no wizard; grupo separado com dicas nos fluxos atuais |
| 2 | Apple GPU | experimental, com verificação em runtime: só com Podman de provider libkrun, imagem oficial `server-vulkan` arm64, e só se `llama-server --list-devices` no container listar a GPU; senão, CPU | fora da v1; Docker Model Runner |
| 3 | versões | fixadas num catálogo único (llama.cpp `b11382`, Qdrant `v1.19.2`), com procedimento de bump e override | `qdrant:latest`; fixar só o llama.cpp |
| 4 | backends | **Vulkan em tudo**: CPU, Radeon, Intel, NVIDIA e Apple (experimental). Uma imagem do llama.cpp por plataforma: a oficial `server-vulkan` e, no Windows, a própria (decisões 8 a 10) | CUDA para a NVIDIA, no Linux e no Windows (escolha anterior, revertida); ROCm e SYCL (ficam para depois, o catálogo permite) |
| 5 | ciclo de vida | `qctx stack status\|up\|down\|remove` | só `status`; tudo dentro do `qctx install` |
| 6 | abordagem | compose **gerado em Python** a partir do catálogo e de perfis de backend, gravado no diretório de dados, com fixtures golden por combinação | YAML estático com overlays; `run` direto sem compose |
| 7 | spike prático | **recusado** (a máquina de teste roda produção nas GPUs) | as medições viram etapa do próprio install e tarefas de abertura do plano |
| 8 | GPU no Windows | **imagem própria**: a oficial do llama.cpp mais o driver dzn do Mesa (Vulkan sobre D3D12) | Windows só em CPU; llama.cpp nativo fora de container; imagem CUDA |
| 9 | onde vale a imagem própria | **só no Windows**; Linux e macOS seguem com a oficial | em todas as plataformas |
| 10 | build da imagem própria | **GitHub Actions deste repositório**, publicando em `ghcr.io/erickstryck`, com versão e digest fixados no catálogo | build e push à mão; build na máquina do usuário |
| 11 | stack em outra máquina | **escuta num IP escolhido**, com chave de API obrigatória e HTTP; o HTTPS fica com o usuário, com receita documentada (tailscale serve ou proxy próprio) | o `qctx` configurar o tailscale serve; HTTPS embutido com Caddy |
| 12 | apontar o plugin | **`qctx stack connect <host>`**, e a opção "stack remota" no `qctx install`, que chama o mesmo caso de uso | imprimir os comandos para colar no cliente |
| 13 | organização | **uma spec, três fases, um plano por fase** | um plano só; três specs |
| 14 | Windows entre as fases | **o Windows só entra na fase 3, já com GPU pela imagem própria**; nas fases 1 e 2 a etapa não é oferecida nele | Windows em CPU nas fases 1 e 2; trazer a imagem própria para a fase 1 |
| 15 | compatibilidade de runtime | **toda opção diz onde funciona: Docker, Podman ou os dois**, no menu, na pergunta de runtime e na documentação. O menu lista também a opção que o runtime em uso não atende, como indisponível, com o runtime que ela exige e a correção. No macOS: CPU (Docker e Podman) e GPU Apple experimental (só Podman) | mostrar só o estado de cada opção, sem dizer o runtime; esconder o que o runtime em uso não atende |
| 16 | progresso da instalação | **barra por operação**: a baixa dos modelos (a parte longa, ~836 MiB, e a que o `qctx` controla de ponta a ponta) desenha barra com porcentagem, MiB, velocidade e ETA; o pull de imagens usa a barra nativa do provider de compose, que em TTY já desenha a dele | barra única cobrindo a instalação inteira, re-parseando o progresso do pull do compose |

A decisão 4 foi revista duas vezes no mesmo dia. A pergunta "daria para focar em vulkan em todas
as plataformas?" tirou a imagem de CPU separada: o Vulkan passou a cobrir CPU, Radeon, Intel e
Apple, com CUDA para a NVIDIA. Depois, com a pergunta "o vulkan também roda na nvidia certo?", o
usuário escolheu padronizar: Vulkan também na NVIDIA, e a imagem CUDA saiu. Isso reverte a
alternativa "só Vulkan, sem CUDA", que antes constava como recusada. O custo está em "Por que
Vulkan em tudo, e o que custa".

Em 2026-10-06, a pergunta "o vulkan não funciona no windows?" mostrou que ele funciona fora de
container e que, dentro dele, depende do driver dzn, que a imagem oficial não traz. Entre deixar o
Windows em CPU, rodar o llama.cpp nativo ou construir uma imagem com o dzn, o usuário escolheu a
imagem própria (decisões 8 a 10). Na mesma rodada vieram o modo servidor e o connect (11 e 12) e a
divisão em fases (13).

A decisão 14 corrigiu um efeito da divisão em fases. Com a imagem própria só na fase 3, as fases 1
e 2 entregariam o Windows em CPU, e o usuário recusou: no Windows, o desenho é GPU. A CPU fica ali
só como reserva, como no Linux, para máquina sem GPU ou GPU que não passa na verificação.

A decisão 15 veio do fluxo simulado. Ao ver o menu, o usuário pediu que o macOS mostre CPU e GPU
(experimental), dizendo que a GPU exige o Podman, e que toda opção diga se funciona no Docker, no
Podman ou nos dois.

A decisão 16, pedida depois do fluxo simulado, é progresso visível: durante a instalação, a baixa
dos modelos desenha uma barra com porcentagem, MiB, velocidade e ETA. O desenho é por operação,
não uma barra única sobre a instalação inteira: a baixa dos modelos é o trecho longo (~836 MiB) e
o `qctx` o controla de ponta a ponta, enquanto o pull das imagens já desenha a própria barra no
provider de compose, e re-parseá-la por runtime seria mais frágil do que deixá-la passar.

## Fases de entrega

Uma spec, três fases, cada uma com o seu plano (decisão 13). A fase 1 vem primeiro; a 2 e a 3
dependem só dela, e não uma da outra. O Windows entra inteiro na fase 3, já com GPU (decisão 14).

| fase | escopo | aceite |
|---|---|---|
| 1. stack local | o pacote `stack/`, a etapa no `qctx install` e `qctx stack status\|up\|down\|remove`, no Linux e no macOS, com a imagem oficial. No Windows, a etapa ainda não é oferecida | as 9 fixtures da fase; a integração opt-in no perfil `cpu`; as verificações de abertura da fase 1 |
| 2. modo servidor e connect | `qctx stack install --listen`, as chaves, `qctx stack keys`, `qctx stack connect` e a opção "stack remota" no wizard; receitas de HTTPS na documentação. Servidor Linux | uma máquina Linux servindo a stack num IP da tailnet ou da LAN, e outra conectada por `qctx stack connect`, com a verificação passando |
| 3. Windows, com GPU pela imagem própria | o suporte ao Windows inteiro: Dockerfile, workflow, publicação no GHCR, pin no catálogo, os perfis do Windows e a verificação numérica | o workflow publica a imagem; numa máquina Windows real, o `--list-devices` mostra a GPU pelo dzn e a verificação numérica passa |

## Viabilidade, passo a passo

Pesquisa documental de 2026-10-05: registries, código-fonte do llama.cpp na tag `b11382`, lista
oficial de pacotes do Ubuntu e issues dos projetos. Nada foi executado em container (decisão 7).

| passo do pedido | veredito | evidência |
|---|---|---|
| o plugin instala como hoje | sim | nenhuma etapa existente muda de comportamento |
| o `qctx` instala as dependências | sim | etapa nova no `qctx install` e grupo `qctx stack` |
| "só precisa de Docker ou Podman" | sim para CPU; com ressalvas | precisa de um provider de compose: o Docker traz o seu; o Podman precisa de docker-compose ou podman-compose, e o docker-compose fala com o socket da API do Podman, que tem de estar ativo. GPU NVIDIA exige, no host, o driver com o componente Vulkan e o NVIDIA Container Toolkit (CDI no Podman). Apple GPU exige Podman com provider libkrun. O `qctx` detecta e explica cada um; não instala nenhum |
| identificar SO e hardware | sim, com stdlib | `platform`, o barramento PCI `/sys/bus/pci/devices/*` (classe `0x03` e vendor 0x1002 AMD, 0x8086 Intel, 0x10de NVIDIA), `/dev/dri` (os nós de render), `nvidia-smi`, `podman machine info` e `podman machine inspect`. Lê o PCI, não o `/sys/class/drm`, porque um `cardN` só aparece lá quando um driver já está anexado, e uma placa sem driver (o caso do NVIDIA sem driver) ficaria invisível. A detecção é só pista: a prova é o próprio container listar a GPU |
| o usuário escolhe onde rodar | sim | menu com o estado de cada opção e onde ela funciona: Docker, Podman ou os dois (ver "Compatibilidade", em "Perfis de backend") |
| compose específico por SO e hardware | sim, e necessário | NVIDIA: o Docker usa `deploy.resources.reservations.devices`, e o `podman compose` ignora esse bloco e exige CDI (podman #28309, #28436). AMD e Intel precisam de `/dev/dri`. Apple, via krunkit. No macOS com Docker Desktop, nenhuma GPU chega ao container; a GPU Apple exige o Podman. No Windows, o Vulkan da imagem oficial não chega à GPU, e por isso o Windows usa a imagem própria (fase 3) |
| Qdrant sempre em CPU | sim | imagem multi-arch (amd64, arm64), 71 MiB |
| o `qctx` baixa os modelos | sim, com stdlib | o Hugging Face devolve sha256 (`x-linked-etag`, `lfs.oid`) e tamanho, e aceita Range: retomada e verificação sem dependência |
| configurar para o plugin | sim | `core.save`, com as mesmas chaves que o README manda gravar à mão. Armadilha: variável de ambiente legada vence o arquivo (`core/config.py`); a etapa detecta e avisa |
| instalar só a stack numa máquina à parte | sim | `qctx stack install` precisa só de Python 3, de um clone do repositório e de Docker ou Podman; não toca em host nem no config do plugin |
| apontar o plugin para ela pela rede | sim, sem mudar o cliente | o cliente já manda `Authorization: Bearer` ao embedding e ao rerank (`core/http.py:66`) e `api-key` ao Qdrant (`core/qdrant.py:44`). É o que o llama-server (`--api-key-file`, `common/arg.cpp:3503` na `b11382`) e o Qdrant (`service.api_key`) esperam |
| GPU no Windows em container | sim, com imagem própria; não verificado com o llama.cpp | o dzn compila do Mesa 26.0.3 (`microsoft-experimental` em `meson.options`), e o `directx-headers-dev 1.619.1-1` existe no Ubuntu 26.04. Um container no WSL2 já listou uma GPU pelo dzn (microsoft/wslg #1215, comentário de 2026-02-20) |

### Por que Vulkan em tudo, e o que custa

- **A imagem Vulkan também é a de CPU.** No `ggml-vulkan.cpp` da `b11382` (bloco "Default to using
  all dedicated GPUs", idêntico ao do master), o llama.cpp usa todas as GPUs discretas e
  integradas que tenham `storageBuffer16BitAccess` (Vulkan 1.1). Sem nenhuma, cai para o primeiro
  device que não seja CPU, que é por onde entra uma GPU virtual como o Venus. Se só houver device
  de CPU (llvmpipe), não usa Vulkan e roda no backend de CPU. Esse backend é compilado com as
  mesmas flags nas duas imagens (`-DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON
  -DGGML_CPU_ALL_VARIANTS=ON`, em amd64 e arm64; `.devops/cpu.Dockerfile` e
  `.devops/vulkan.Dockerfile`). Os tamanhos também batem: `server` tem 296 MiB e `server-vulkan`
  tem 294 MiB.
- **Os drivers de AMD, Intel e Apple já estão na imagem.** A `server-vulkan` é Ubuntu 26.04 com
  `mesa-vulkan-drivers 26.0.3`, que traz RADV (AMD), ANV e hasvk (Intel), lavapipe, NVK, asahi e
  virtio (Venus, o driver do krunkit no Mac). Isso foi conferido na lista oficial de arquivos do
  pacote.
- **Na NVIDIA, o driver Vulkan vem do host.** O NVK da imagem só funciona com o driver nouveau do
  kernel, e não com o driver proprietário. Com o proprietário, quem injeta o ICD
  (`nvidia_icd.json`) e as bibliotecas do host é o NVIDIA Container Toolkit, sob três condições:
  - no modo legado, a capacidade `graphics`. A documentação do toolkit diz "graphics: required
    for rendering OpenGL, EGL, and Vulkan applications", e sem a variável o padrão é só
    `utility,compute`;
  - no CDI, o spec gerado por `nvidia-ctk cdi generate` tem de conter o ICD. Há relato dele
    sumindo (nvidia-container-toolkit #767), e o CDI é o caminho do Podman;
  - o host tem de ter o componente Vulkan do driver. No Ubuntu ele está no
    `libnvidia-gl-<versão>` (`/usr/share/vulkan/icd.d/nvidia_icd.json`), e a instalação headless
    não o traz: `nvidia-headless-580-server` depende só de pacotes de compute (conferido em
    packages.ubuntu.com).

  O backend Vulkan do llama.cpp trata a NVIDIA como caminho de primeira classe: usa
  `VK_NV_cooperative_matrix2` em matmul e flash attention, e tem ajustes só para esse vendor
  (`ggml-vulkan.cpp` na `b11382`). O desempenho contra o CUDA, nestes dois modelos, não foi medido.
- **O que se ganha.** Uma imagem do llama.cpp, de 294 MiB, em vez de duas, com a CUDA de 2,4 GiB:
  uma imagem a menos para fixar e atualizar a cada bump. Como a imagem é a mesma em todos os
  perfis, a etapa pode puxá-la e provar as GPUs antes do menu (ver "O que ela faz, em ordem"). E
  some a exigência de um driver novo o bastante para a versão do CUDA da imagem: o ICD é o do
  driver que já está no host.
- **No Windows, a imagem oficial não basta.** Fora de container, o Vulkan funciona no Windows com
  qualquer GPU, e o llama.cpp publica `llama-b11382-bin-win-vulkan-x64.zip`. Mas os containers
  rodam numa VM WSL2, onde a GPU chega via D3D12 (`/dev/dxg`), e o Vulkan só a alcança pelo
  driver dzn do Mesa (Vulkan sobre D3D12). O Mesa da imagem oficial não traz o dzn (reconferido
  em 2026-10-06: asahi, gfxstream, intel, intel_hasvk, lvp, nouveau, radeon e virtio). Com um
  Mesa que o traz, funciona: um comentário de 2026-02-20 na microsoft/wslg #1215 mostra um
  container no WSL2 listando uma GTX 1080 pelo dzn, com `--gpus all` e `/usr/lib/wsl` montado. O
  próprio dzn avisa "not a conformant Vulkan implementation, testing use only". Por isso o
  Windows usa a imagem própria, com uma verificação numérica contra a CPU, e só entra na fase 3,
  já com GPU (ver "Imagem própria e GPU no Windows").

## Arquitetura **[a validar]**

### Onde mora

Pacote novo `stack/` na raiz, ao lado de `core/`, `cli/`, `hooks/` e `hosts/`, só com stdlib. As
dependências apontam num sentido só: `cli -> stack -> core`.

- `core/`, `hooks/` e `hosts/` nunca importam `stack`. Os hooks carregam `core` a cada prompt e
  não devem nem carregar código de subprocess, containers e downloads.
- `stack/` não importa `hooks/`, `hosts/` nem `cli/`, e não sabe que hosts existem. Os orçamentos
  de latência de cada host chegam por parâmetro, vindos do `cli`, que já nomeia os hosts.
- Fora de `core/` porque `core/` é o núcleo portável que não conhece a máquina. A stack é
  infraestrutura da máquina.

### Módulos

| módulo | responsabilidade única |
|---|---|
| `stack/catalog.py` | só dados: imagens (tag + digest do índice), GGUFs (repo, revisão, arquivo, bytes, sha256), flags por papel, portas e nomes padrão. O procedimento de bump vive aqui |
| `stack/process.py` | `Completed`, `Runner` e `SubprocessRunner`: o runner que roda cada comando como processo filho, com timeout que mata o grupo de processos inteiro e Ctrl-C que para esse grupo antes de seguir (ver "Detecção") |
| `stack/engine.py` | o contrato dos runtimes (`EngineInfo`, `Provider`, `ProviderInfo` e o Protocol `ContainerRuntime`) e o que os dois engines compartilham: `normalize_arch`, `socket_alive`, `parse_size` e os auxiliares de compose (o argv, o token de versão, a primeira linha da saída de uma ferramenta, a parte antes de ` / `) |
| `stack/docker.py` | `Docker` atrás do contrato: engine (com a memória que os containers usam), provider de compose, compose, stats |
| `stack/podman.py` | `Podman` atrás do contrato: a escolha do provider de compose, o socket da API e a máquina do macOS |
| `stack/runtimes.py` | a porta de entrada: `discover`, a descoberta do que responde, e a reexportação dos nomes que as tarefas anteriores e os testes importam dele |
| `stack/facts.py` | `HostFacts`: SO, arch, WSL, RAM (no Linux; `None` no macOS, que lê o número do engine), disco, GPUs pelo barramento PCI, `/dev/dri`, SELinux, portas ocupadas. Coletado por primitivas injetadas |
| `stack/devices.py` | o que o `--list-devices` de um container imprimiu: `Device`, `parse_devices` e `vendor_of` |
| `stack/backends.py` | perfis `Cpu`, `DriGpu` (AMD e Intel), `NvidiaGpu` e `AppleGpu` atrás do Protocol `Backend`; registro `BACKENDS`; a compatibilidade de cada perfil com Docker e Podman, por plataforma. Lê os devices pelo `stack/devices.py` e os reexporta |
| `stack/compose.py` | `render(plan) -> dict`, puro, e o emissor YAML |
| `stack/fetch.py` | download com retomada e sha256, com transporte HTTP injetado; a barra de progresso (decisão 16) é dele: porcentagem, MiB baixado/total, velocidade e ETA, com reescrita da linha em TTY e uma linha por fatia fora dele |
| `stack/health.py` | espera de prontidão, com relógio injetado |
| `stack/verify.py` | verificação funcional e calibração de latência; devolve `core.setup.Check` |
| `stack/state.py` | `StackState`, gravado atômico por `core.statefile.write_json` |
| `stack/installer.py` | caso de uso: provisionar, no papel local (etapa do wizard) ou servidor (`qctx stack install`, fase 2) |
| `stack/lifecycle.py` | casos de uso: status, up, down, remove |
| `stack/exposure.py` (fase 2) | endereços candidatos para escutar e a classe de cada um (loopback, tailnet, privado, público); a regra "fora do loopback, sem chave não sobe" |
| `stack/keys.py` (fase 2) | gera, grava e roda as chaves da stack; nunca as escreve no compose, no `stack.json` ou no config |
| `stack/connect.py` (fase 2) | caso de uso: apontar o plugin para uma stack remota |
| `stack/cli.py` | apresentação e raiz de composição: registra `qctx stack`, expõe `install_step()` para o `cmd_install`, injeta as implementações concretas. Recebe do `cli/qctx.py` a gravação de segredos que o wizard já usa, para não importar `cli` |

Dentro do pacote, as dependências também apontam num sentido só: `process` <- `engine` <-
`docker` e `podman` <- `runtimes`. O `facts` e o `backends` importam o contrato de `process` e de
`engine`, nunca da porta `runtimes`, que existe para quem já importava dela. Um engine novo é um
módulo e uma classe; um módulo que passa de ~300 linhas (regra do plano) ganhou uma
responsabilidade que não é dele.

### Contratos **[a validar]**

Esboço das assinaturas, para fixar as fronteiras (não é código final):

```python
class Runner(Protocol):          # subprocess, com timeout sempre
    def run(self, argv: list[str], *, timeout: float, stream: bool = False) -> Completed: ...

class Transport(Protocol):       # HTTP GET em pedaços, a partir de um byte
    def get(self, url: str, *, start: int = 0) -> Iterator[bytes]: ...

class ContainerRuntime(Protocol):
    name: str                                   # "docker" | "podman"
    def engine(self) -> EngineInfo: ...         # versão, os/arch do engine, rootless, VM (libkrun, wsl...), a memória que os containers usam (memTotal/MemTotal)
    def compose_provider(self) -> ProviderInfo: ...
    def compose(self, project: str, file: Path, *args: str, stream: bool = False) -> Completed: ...
    def stats(self, project: str) -> dict[str, int]: ...   # bytes de memória por serviço

class Backend(Protocol):
    id: str                                     # "cpu" | "amd" | "intel" | "nvidia" | "apple"
    experimental: bool
    image_role: str | None                      # None: a imagem da plataforma (ver "Catálogo")
    def runtimes(self, platform: str) -> frozenset[str]: ...  # onde funciona: docker, podman ou os dois
    def availability(self, facts: HostFacts, engine: EngineInfo) -> Availability: ...
    def service_patch(self, runtime: str, facts: HostFacts, device: str | None) -> dict: ...
    def devices_seen(self, list_devices_output: str) -> list[Device]: ...

class Prompter(Protocol): ...    # ask, confirm, choose, secret (sem eco)
class Reporter(Protocol): ...    # step, ok, warn, fail, progress
class ConfigSink(Protocol): ...  # current() -> Config; save(patch)
class SecretSink(Protocol): ...  # store({ENV_NAME: valor}) -> Path; a implementação é a do wizard
```

### SOLID, concretamente

- **S**: cada módulo tem um motivo para mudar. Os casos de uso não fazem `print` nem `input`; falam
  com `Prompter` e `Reporter`.
- **O**: um backend novo (ROCm, SYCL, ou CUDA para a NVIDIA no Windows) é uma classe, uma entrada
  em `BACKENDS` e, se usar outra imagem, uma entrada no catálogo. Nem o installer nem o
  renderizador mudam.
- **L**: `Docker`, `Podman` e o `FakeRuntime` dos testes passam pelo mesmo teste de contrato.
- **I**: Protocols pequenos. Quem baixa modelo não conhece compose, e quem renderiza não conhece
  subprocess.
- **D**: `installer` e `lifecycle` dependem dos Protocols; `stack/cli.py` injeta as implementações
  concretas.

### Código existente que muda

| arquivo | mudança |
|---|---|
| `cli/qctx.py` | registra o grupo `stack`, passando a gravação de segredos do wizard; flags `--stack` e `--image` no `install`; chama `stack.cli.install_step()` entre o launcher e a configuração; inclui a seção `stack` no `--check` e no `--json`. Na fase 2, a etapa ganha o ramo "stack remota" |
| `images/llama-dzn/Dockerfile` (novo, fase 3) | a imagem própria |
| `.github/workflows/llama-dzn.yml` (novo, fase 3) | build e publicação da imagem própria |
| `tests/test_core_is_portable.py` | `FORBIDDEN` ganha `stack` |
| `tests/test_installable_from_git.py` | passa a barrar o padrão "download encadeado num shell por pipe", que o scanner do hermes classifica como crítico |
| `README.md`, `docs/*.md` | ver "Documentação" |

Nada muda em `core/`, `hooks/`, `hosts/`, `scripts/` ou `bin/`. `StackError` herda de `CoreError`,
então o `main()` atual já imprime o erro.

## Catálogo

### Imagens

| papel | referência | tamanho comprimido |
|---|---|---|
| llama | `ghcr.io/ggml-org/llama.cpp:server-vulkan-b11382@sha256:431561ee79ee67b3980a02ff47ed9dc19496127643b75671d9789ef693ca57f9` | 294 MiB amd64, 290 MiB arm64 |
| qdrant | `docker.io/qdrant/qdrant:v1.19.2-unprivileged@sha256:efb96a9425a90d2d5a1a0a474156280df1892dcdf1af3e8515bd2589b1bfd88b` | ~71 MiB |
| llama-dzn (fase 3) | `ghcr.io/erickstryck/llama-dzn:b11382-mesa26.0.3`, com o digest do primeiro build | a oficial mais o dzn; só amd64 |

O digest é o do **índice multi-arch**, então a mesma referência serve em amd64 e arm64. A variante
`-unprivileged` do Qdrant (roda como não-root) é proposta minha **[a validar]**.

A imagem do llama sai da plataforma do engine, numa regra só, no catálogo: com o engine no WSL2
(Windows, fase 3), a `llama-dzn` em todos os perfis; nas demais, a oficial. Um backend só declara
`image_role` se precisar de outra imagem.

A `llama-dzn` compila o dzn do Mesa 26.0.3: `https://archive.mesa3d.org/mesa-26.0.3.tar.xz`,
41,8 MiB, sha256 `ddb7443d328e89aa45b4b6b80f077bf937f099daeca8ba48cabe32aab769e134`, com a
assinatura `.sig` publicada ao lado.

### Modelos

| papel | repo @ revisão | arquivo | bytes | sha256 | licença |
|---|---|---|---|---|---|
| embedding | `gpustack/bge-m3-GGUF` @ `2d48f1737679ad900d5c26c5aad5410e9c70fdca` | `bge-m3-Q4_K_M.gguf` | 437778496 | `6d39681b26c61279ac1f82db35a04a05009e94c415b51c858ff571489a82fc06` | MIT |
| rerank | `gpustack/bge-reranker-v2-m3-GGUF` @ `3093af03b1a635e67b084b1d8c03c5f5e020fd05` | `bge-reranker-v2-m3-Q4_K_M.gguf` | 438376864 | `e186a244ed455b4ab66ec64339ce7427a6ae13f5c0b5e544de96e50f0f8b3673` | Apache-2.0 |

URL: `https://huggingface.co/<repo>/resolve/<revisão>/<arquivo>`. Com a revisão fixa, o arquivo
é imutável.

### Flags dos servidores **[a validar]**

```
embed:  -m /models/bge-m3-Q4_K_M.gguf --host 0.0.0.0 --port 8080 --embedding -c 8192 -b 8192 -ub 8192 --no-ui -dev <device|none>
rerank: -m /models/bge-reranker-v2-m3-Q4_K_M.gguf --host 0.0.0.0 --port 8080 --reranking -c 8192 -b 8192 -ub 8192 --no-ui -dev <device|none>
```

**`-dev none` no perfil `cpu`, sempre** (medido em 2026-10-06). Um engine pode injetar GPUs em todo
container sem que o compose peça: o `containers.conf` do Podman aceita `[containers] devices`, e na
máquina de teste ele trazia `/dev/dri`. Ali, o container do perfil `cpu` listou as três GPUs, e o
llama.cpp usou a GPU por padrão (rerank de 20 x 2400 em 1,18 s, 204 MB de RSS). Com `-dev none`, o
mesmo container ficou na CPU (7,06 s, 2,8 GB), igual a uma máquina sem essa configuração. Então o
perfil `cpu` não depende da ausência de device: ele a declara.

**`--no-ui`, e não `--no-webui`**: na `b11382`, o log avisa "Use --ui/--no-ui (or deprecated
--webui/--no-webui)". As duas formas desligam a UI; fica a que não é deprecada.

**`-ub 8192` vai nos dois, não só no rerank.** No código da `b11382`, com embeddings ligados o
servidor força `n_batch = n_ubatch` (`tools/server/server.cpp:152-157`, `common/common.cpp:1288-1293`),
e o `ubatch` padrão é 512. Para modelo não-causal, entrada maior que o `ubatch` é recusada com
"input (N tokens) is too large to process" (`tools/server/server-context.cpp:3330-3333`). O
`--reranking` liga o mesmo modo (`common/arg.cpp:3484-3488`). O chunk alvo do plugin tem 2400
caracteres, ~800 tokens (`core/chunk.py:15`), e o teto é de 6000: com 512, o chunk típico seria
recusado.

`--host 0.0.0.0` vale dentro do container, onde o mapeamento de porta exige. Quem limita a
exposição é a publicação só em `127.0.0.1`.

### Portas, caminhos e nomes **[a validar]**

- Portas: `127.0.0.1:6333` (Qdrant HTTP), `:8003` (embed), `:8004` (rerank), as mesmas do README.
  Se uma estiver ocupada, a etapa propõe a primeira livre a partir de `porta + 10000` e mostra a
  escolha antes de seguir.
- Diretório: `$QCTX_STACK_DIR`, ou `${XDG_DATA_HOME:-~/.local/share}/memories-plugin/stack`, com
  `models/`, `compose.yaml` e `stack.json`. Fica sob `$HOME`, que o Docker Desktop e a VM do
  Podman compartilham por padrão. A etapa recusa, com um `StackError` que aponta para
  `QCTX_STACK_DIR`, qualquer diretório que não seja absoluto ou que leve `:` (quebra a
  sintaxe curta de volume nos dois providers), `$` (os dois providers interpolam `$VAR` e
  `${VAR}` no caminho: `/x/$HOME/stack` resolve para a home real, saindo 0 sem aviso), ou um
  surrogato ou caractere fora do Plano Multilíngue Básico (um emoji: um provider rejeita o
  arquivo, o outro o corrompe). Recusa em vez de mudar para a sintaxe longa, porque `$$` não
  é portável: medido em 2026-10-07, `a$$b` ficou `a$$b` no docker-compose e virou `a$b` no
  podman-compose. Acento dentro do BMP funciona nos dois e é aceito (medido em 2026-10-06 nos
  dois providers com arquivos descartáveis).
- Projeto compose `memories-plugin`, serviços `qdrant`, `embed` e `rerank`, volume nomeado
  `memories-plugin-qdrant`. Os dois providers prefixam o volume com o projeto, então o nome real
  no engine é `memories-plugin_memories-plugin-qdrant` (medido no docker-compose e no
  podman-compose). É o prefixo que impede a integração opt-in, que usa outro projeto, de tocar no
  acervo da stack de verdade.
- Cada serviço leva `container_name: <projeto>-<serviço>`. Sem isso, o docker-compose nomeia
  `<projeto>-<serviço>-1` e o podman-compose `<projeto>_<serviço>_1`, e o `stats`, os logs e o
  `status` teriam de saber qual provider criou o container.

### Bump e override **[a validar]**

Bump, a cada release que mexer no catálogo:

1. llama.cpp: escolher a build nova e ler o digest do índice de `server-vulkan-bNNNNN` (GET do
   manifest com Accept de índice OCI; o digest vem no cabeçalho `docker-content-digest`).
2. llama-dzn (fase 3): rodar o workflow com a build nova, apontando para o digest do passo 1, e
   copiar para o catálogo o digest que ele publica.
3. Qdrant: **só a minor seguinte**, nunca pular. A compatibilidade de storage só é garantida
   entre minors consecutivas.
4. Rodar a integração opt-in (ver "Testes") e regenerar as fixtures golden.

Override: `--image ROLE=REF` no `qctx install --stack` e no `qctx stack up --upgrade`, com
`ROLE` igual a `llama`, `llama-dzn` ou `qdrant`; ou as variáveis
`QCTX_STACK_IMAGE_LLAMA|LLAMA_DZN|QDRANT`. O que foi usado fica gravado no `stack.json`, e `qctx stack
up` sem `--upgrade` repete exatamente o que está lá: **nada atualiza sozinho**.

## Perfis de backend e o compose de cada combinação **[a validar nos detalhes]**

Na mesma plataforma, todos os perfis usam a mesma imagem; o que muda é como a GPU chega ao
container.

| perfil | o serviço recebe | oferecido quando | prova |
|---|---|---|---|
| `cpu` | nenhum device | sempre | `/health` 200 |
| `amd`, `intel` | `devices: /dev/dri`; no Podman, também a anotação `run.oci.keep_original_groups: "1"` (exigida rootless, inócua rootful, por isso sem ramo) | Linux nativo com GPU do vendor no barramento PCI (classe `0x03`, vendor conhecido) e `/dev/dri/renderD*` | `--list-devices` lista um `Vulkan<n>` do vendor |
| `nvidia` | Docker: `deploy.resources.reservations.devices` com driver `nvidia`, `device_ids` e `capabilities: [gpu, compute, utility, graphics]`; Podman: `devices: nvidia.com/gpu=<i>` (CDI) | Linux nativo com NVIDIA visível (`nvidia-smi`), o ICD Vulkan da NVIDIA no host (`nvidia_icd.json` em `/usr/share/vulkan/icd.d` ou `/etc/vulkan/icd.d`) e o toolkit pronto (Docker: `nvidia-container-runtime-hook` no PATH, ou `nvidia-cdi-hook` a partir do Docker 29.2; Podman: spec CDI em `/etc/cdi` ou `/var/run/cdi`, gerado por `nvidia-ctk`) | `--list-devices` lista um `Vulkan<n>` NVIDIA |
| `apple` (experimental) | `devices: /dev/dri` dentro da VM; no Podman, também a anotação `run.oci.keep_original_groups: "1"` | macOS arm64 com `podman machine` em VMType libkrun | `--list-devices` lista o Venus |
| `amd`, `intel`, `nvidia` no Windows (fase 3) | imagem `llama-dzn`; `devices: /dev/dxg`; `/usr/lib/wsl:/usr/lib/wsl:ro`; `LD_LIBRARY_PATH=/usr/lib/wsl/lib` | engine no WSL2 (Docker Desktop ou `podman machine` com WSL) e GPU do vendor no host | `--list-devices` lista um `Vulkan<n>` `Microsoft Direct3D12 (<GPU do vendor>)`, e a verificação numérica passa |

**Compatibilidade: Docker, Podman ou os dois.** Toda opção diz onde funciona (decisão 15). A fonte
é uma só, `Backend.runtimes(plataforma)`: dela saem o rótulo do menu, a pergunta de runtime, as
fixtures e a tabela do `docs/install.md`, e um teste as mantém de acordo.

| plataforma | opção | Docker | Podman | rótulo no menu | exige ainda |
|---|---|---|---|---|---|
| Linux | CPU | sim | sim | Docker e Podman | nada |
| Linux | GPU AMD ou Intel | sim | sim | Docker e Podman | `/dev/dri` acessível; no Podman rootless, os grupos do usuário preservados |
| Linux | GPU NVIDIA | sim | sim | Docker e Podman | o ICD Vulkan da NVIDIA no host e o NVIDIA Container Toolkit: o hook no Docker, o spec CDI no Podman |
| macOS | CPU | sim | sim | Docker e Podman | nada |
| macOS, Apple Silicon | GPU Apple (experimental) | não: o Docker Desktop não entrega GPU a container | sim | só Podman | máquina com provider libkrun |
| Windows (fase 3) | GPU AMD, Intel ou NVIDIA | sim, com o backend WSL2 | sim, com máquina WSL | Docker e Podman | a imagem própria e a verificação numérica |
| Windows (fase 3) | CPU, como reserva | sim | sim | Docker e Podman | nada |

Só o `apple` é de um runtime só. No Podman 6 (2026-06-24) em diante, libkrun é o provider padrão
no macOS, e no Podman 5 o padrão é applehv. A troca de provider precisa FICAR GRAVADA na
configuração, não ser um comando de uma vez só: o `podman machine info` (o `Host.VMType`) e o
`podman machine start` leem o provider CONFIGURADO (a tabela `[machine]` de
`~/.config/containers/containers.conf` ou a variável `CONTAINERS_MACHINE_PROVIDER` do ambiente de
cada comando). Um `CONTAINERS_MACHINE_PROVIDER=libkrun podman machine init` deixa o próximo
`machine info` dizendo applehv de novo, e o `machine start` nem acha a máquina nova; e no Podman 6
o applehv só aparece quando a config ou o ambiente o fixam, o que a flag `--provider libkrun` no
`init` não muda. A correção durável, igual para os dois, é: gravar `provider = "libkrun"` na
tabela `[machine]` (e tirar um `CONTAINERS_MACHINE_PROVIDER` exportado) e depois
`podman machine init --now`. Isso foi lido no fonte do Podman nas tags `v5.7.0` e `v6.0.0`
(`pkg/machine/provider/platform_darwin.go`, `cmd/podman/machine/info.go` e `start.go`). O
instalador oficial (`.pkg`) traz o krunkit: a `v6.1.3` empacota a 1.3.2
(`contrib/pkginstaller/Makefile`).

**NVIDIA no Docker.** O daemon só aceita pedido de GPU NVIDIA se achar um dos dois hooks do
toolkit no PATH, e tenta o CDI antes do modo legado (`daemon/devices_nvidia_linux.go` do moby).
O hook legado (`nvidia-container-runtime-hook`) vale em qualquer versão do Docker; o hook de CDI
(`nvidia-cdi-hook`) o daemon reconhece só a partir da 29.2 (lido no fonte do moby nas tags
`v28.3.2`, `v28.5.2`, `docker-v29.1.0` e `docker-v29.2.0`), e uma versão que não dá para ler não
conta o hook de CDI. No modo legado, as capacidades do pedido além de `gpu` viram o
`NVIDIA_DRIVER_CAPABILITIES`, e o daemon só define essa variável quando o pedido lista alguma.
Por isso o `graphics` vai explícito: sem ele, o ICD não entra. No CDI, vale o que o spec
gerado contém.

**GPU no Windows (fase 3).** O `/dev/dxg` é o device genérico do GPU-PV, o mesmo para os três
vendors, e o dzn chega à GPU pelas bibliotecas D3D12 do Windows, montadas de `/usr/lib/wsl`. A
receita que funcionou (wslg #1215) usava `--gpus all`, que no Docker Desktop monta esse device
para a NVIDIA. Se o `/dev/dxg` direto não bastar, o recurso é esse, e isso é verificação de
abertura da fase 3.

Comum a todos:

- Portas publicadas só em `127.0.0.1`.
- `cap_drop: [ALL]` e `security_opt: [no-new-privileges:true]`.
- Modelos montados `:ro`, com `z` só quando o SELinux está ativo.
- `restart: always`. No Podman 4.9.3, o `podman-restart.service` só religa containers com
  `restart-policy=always` (lido na própria unidade). No Docker, `always` tem o mesmo efeito.
- No Qdrant, `QDRANT__TELEMETRY_DISABLED=true`.

**Escolha da GPU.** O menu lista cada GPU que o `--list-devices` mostrou, com nome, memória livre
e onde a opção funciona (ver "Compatibilidade"). O formato é
`Vulkan<n>: <nome> (<total> MiB, <livre> MiB free)`, ou `(none)` sem GPU (`common/arg.cpp:1141`
na `b11382`). O padrão é a GPU provada de maior memória livre; opção experimental nunca é o
padrão. A versão anterior desta seção punha a dedicada antes da integrada, lendo o tipo da linha
`ggml_vulkan: <n> = … | uma: <0|1> | …`. Medido em 2026-10-06: essa linha não sai com
`--list-devices`, nem com `-lv 4`, então só a memória livre decide e o menu não afirma o tipo. Em
AMD e Intel, a escolha vira `-dev Vulkan<n>`. Na NVIDIA, a GPU entra no container pelo índice do
`nvidia-smi` (`device_ids` no Docker, `nvidia.com/gpu=<i>` no CDI, os dois na ordem do NVML), e o
`-dev` sai do `--list-devices` desse container.

**O emissor YAML.** É de bloco, e todo valor sai por `json.dumps`. Isso evita as armadilhas do
YAML 1.1 (`"8003:8080"` lido como sexagesimal, `yes`/`no` lidos como booleano) e continua legível.
As chaves saem sem aspas quando casam com `[A-Za-z_][A-Za-z0-9_]*` e não são palavra reservada do
YAML 1.1; as demais saem por `json.dumps`.

Exemplo renderizado, Linux + Podman + `amd` (digests encurtados aqui; o arquivo leva os inteiros):

```yaml
services:
  qdrant:
    container_name: "memories-plugin-qdrant"
    image: "docker.io/qdrant/qdrant:v1.19.2-unprivileged@sha256:efb96a94…"
    restart: "always"
    ports:
      - "127.0.0.1:6333:6333"
    volumes:
      - "memories-plugin-qdrant:/qdrant/storage"
    environment:
      QDRANT__TELEMETRY_DISABLED: "true"
    cap_drop:
      - "ALL"
    security_opt:
      - "no-new-privileges:true"
  embed:
    container_name: "memories-plugin-embed"
    image: "ghcr.io/ggml-org/llama.cpp:server-vulkan-b11382@sha256:431561ee…"
    restart: "always"
    command:
      - "-m"
      - "/models/bge-m3-Q4_K_M.gguf"
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "8080"
      - "--embedding"
      - "-c"
      - "8192"
      - "-b"
      - "8192"
      - "-ub"
      - "8192"
      - "--no-ui"
      - "-dev"
      - "Vulkan0"
    ports:
      - "127.0.0.1:8003:8080"
    volumes:
      - "<stack>/models:/models:ro"
    devices:
      - "/dev/dri:/dev/dri"
    annotations:
      "run.oci.keep_original_groups": "1"
    cap_drop:
      - "ALL"
    security_opt:
      - "no-new-privileges:true"
  rerank:
    # igual ao embed, com --reranking, o outro GGUF e 127.0.0.1:8004:8080
volumes:
  "memories-plugin-qdrant": {}
```

Fixtures golden, uma por renderização distinta. Na fase 1, nove: `linux-docker-{cpu,dri,nvidia}`,
`linux-podman-{cpu,dri,nvidia}`, `macos-docker-cpu` e `macos-podman-{cpu,apple}`. A `dri` cobre
`amd` e `intel`, que só diferem no `-dev`. As fixtures seguem a tabela de "Compatibilidade": não
existe `macos-docker-apple`. A fase 2 acrescenta `linux-{docker,podman}-cpu-server`,
com IP de escuta e chaves; a fase 3, `windows-{docker,podman}-{cpu,dxg}`, todas com a
`llama-dzn`. Ao fim, 15 arquivos em `tests/fixtures/stack/`, e são eles o "compose específico por
SO e hardware" visível no repositório.

## A etapa no `qctx install` **[a validar]**

### Onde entra

A ordem atual do `cmd_install` é: launcher, configuração, `vector_size`, hosts, re-check e passos
manuais. A etapa nova entra **entre o launcher e a configuração**, para que a passada de
configuração já mostre os endereços novos como valor atual.

### Quando aparece

- Existe uma stack gerenciada (`stack.json`): mostra o estado. Se ela está parada, oferece
  religá-la (`y/N`). Se o catálogo tem pins mais novos, aponta `qctx stack up --upgrade`.
- Não existe, e o `diagnose` atual acusa bloqueio no Qdrant ou no embedding: explica o que faria
  (o que baixa, quanto pesa, portas, onde ficam os dados) e pergunta `y/N`. A partir da fase 2, a
  pergunta tem três saídas: provisionar aqui, conectar a uma stack remota (ver "Conectar o plugin
  a uma stack remota") ou pular.
- Os endpoints respondem: uma linha dizendo que a stack local não é necessária, e segue.
- `--stack <perfil>` força a etapa em qualquer caso.

### O que ela faz, em ordem

Como todos os perfis usam a mesma imagem do llama.cpp, o pull não depende da escolha. Por isso a
etapa puxa as imagens e prova as GPUs antes do menu, e o menu só deixa escolher o que o container
de fato enxergou. O consentimento para baixar vem da pergunta de entrada, ou do próprio `--stack`
(ver "Quando aparece").

1. Detecta os runtimes que respondem e o provider de compose de cada um. Com Podman +
   docker-compose, confere também o socket da API.
2. Detecta o hardware e calcula, pelo lado do host, a disponibilidade de cada perfil em cada
   runtime: pronto para provar, falta X (com a correção), não atendido por este runtime (com o
   runtime que ele exige) ou não suportado aqui (com o motivo). Confere portas e disco; a RAM
   confere pelo número do engine (o `memTotal` do Podman, o `MemTotal` do Docker), porque é ele
   que os containers usam: no macOS o host tem mais memória, mas os containers rodam na VM da
   máquina do Podman, que por padrão tem 2 GiB. Se
   os dois runtimes respondem, pergunta qual usar, e cada linha diz o que ele atende nesta
   máquina. Não pergunta com `--runtime`, nem quando o `--stack` pedido só é atendido por um
   runtime (o `apple`, só pelo Podman). Com `--yes`, e como padrão da pergunta, fica o Docker
   **[a validar]**: ele religa a stack com o daemon, sem passo extra, tem menos verificações em
   aberto no caminho de GPU, e opção experimental não decide o padrão.
3. Renderiza o compose de cada perfil pronto e roda `compose config` em cada um, para que o
   próprio provider valide o arquivo.
4. `compose pull`, com o progresso do provider repassado ao terminal: 294 MiB do llama.cpp e
   ~71 MiB do Qdrant.
5. Prova cada perfil de GPU pronto com `compose run -T --rm --no-deps embed --list-devices`. Usa
   a mesma definição de serviço que vai rodar, então o que o container vê aqui é o que verá
   depois (medido: a mesma lista que um `compose exec` no serviço rodando). O `-T` porque o
   podman-compose aloca TTY por padrão no `run`, e a saída vinha com `\r\n`.
6. Menu: uma linha por opção que a plataforma tem para o hardware desta máquina: `cpu`, cada GPU
   provada e cada GPU que não pôde ser usada. Toda linha diz onde a opção funciona (Docker e
   Podman, ou só um deles, pela tabela de "Compatibilidade") e o estado aqui: disponível, com
   nome e memória livre, ou indisponível, com o motivo e a correção. A GPU que o host tem e o
   container não viu aparece com o provável motivo. A opção que o runtime em uso não atende
   aparece com o runtime que ela exige: no macOS com Docker Desktop, a GPU Apple aparece como
   "só Podman". Escolher uma opção indisponível repete a correção e volta ao menu; a etapa não
   instala runtime. O padrão segue "Escolha da GPU". Com `--stack <perfil>`, o menu se reduz às
   GPUs daquele perfil, e com `--yes` vale o padrão sem perguntar. Se o perfil pedido
   explicitamente não tem nenhuma GPU provada, ou não é atendido pelo runtime em uso, a etapa
   para com o motivo e a correção, em vez de cair para CPU sem avisar. Só o `auto` cai para
   `cpu`.
7. Resumo e confirmação: modelos a baixar (836 MiB), portas, diretório e volume.
8. Baixa os modelos (retomada + sha256), desenhando a barra de progresso: porcentagem, MiB
   baixado/total, velocidade e ETA (decisão 16). Com `--yes` ou fora de TTY, a barra vira uma
   linha por fatia, para o log não se encher de reescrita de linha.
9. Grava o compose do perfil escolhido, roda `compose up -d` e espera pela prontidão.
10. Verificação funcional e calibração (ver abaixo).
11. Grava o config e checa a armadilha do ambiente (ver abaixo).
12. Grava `stack.json` com fase `running` e imprime como a stack volta depois de um reboot neste
    runtime.

Depois disso, o wizard continua como hoje: passada de configuração (agora com os endereços da
stack como valor atual), detecção de `vector_size`, hosts e re-check.

### Flags

| invocação | comportamento |
|---|---|
| `qctx install` com terminal | oferece a etapa quando cabe e pergunta |
| `qctx install --stack auto\|cpu\|amd\|intel\|nvidia\|apple` | entra direto na etapa, com o perfil pré-escolhido |
| `qctx install --runtime docker\|podman` | escolhe o runtime quando os dois respondem. Perfil que esse runtime não atende é recusado na hora, com o runtime que ele exige: `--stack apple --runtime docker` não passa |
| `qctx install --yes` | **não** provisiona: baixar gigabytes não é um "sim" implícito. Imprime a linha com `--stack auto` |
| `qctx install --yes --stack auto` | provisiona sem perguntar. `auto` é a GPU verificada que o menu ofereceria como padrão (ver "Escolha da GPU"); sem GPU verificada, `cpu`; nunca `apple` |
| `qctx install --check` / `--json` | só relata o estado da stack gerenciada (saúde, pins defasados); nunca escreve, nunca puxa imagem. O JSON ganha a chave `stack` |
| `qctx install --config-only` | sem a etapa |
| sem terminal e sem `--yes` | só relata, a regra atual |

### O que grava no config

Grava `qdrant_url = http://127.0.0.1:<q>`, `api_base_url = http://127.0.0.1:<e>/v1`, `rerank_url =
http://127.0.0.1:<r>/v1/rerank`, `embed_url` vazio e `vector_size` detectado do endpoint.
`embed_model` e `rerank_model` só são gravados se diferirem dos nomes do catálogo, que hoje são os
defaults. Nenhuma chave: a stack local não tem autenticação.

**O `/v1` é obrigatório** (medido em 2026-10-06). A versão anterior desta seção dizia
`api_base_url = http://127.0.0.1:<e>`, como o README manda à mão. Com isso o plugin chama
`/embeddings`, e na `b11382` essa rota não é a da OpenAI: devolve uma lista crua, e o `Embedder`
quebra com `AttributeError: 'list' object has no attribute 'get'`. Só `/v1/embeddings` devolve o
formato que o plugin lê (dimensão 1024 detectada). O rerank responde igual em `/rerank` e em
`/v1/rerank`; fica o `/v1`, o mesmo layout do `connect` da fase 2. O caminho manual do README tem
o mesmo defeito, e a correção dele fica fora desta spec, como o `-ub` do embed (ver "Achado à
parte").

**O `embed_url` é esvaziado** porque ele vence o `api_base_url` (`resolved_embed_url` em
`core/config.py`): um `embed_url` antigo no arquivo continuaria mandando o embedding para o
endereço de antes, com o resto apontando para a stack.

Antes de gravar, mostra o diff. Se algum valor não vazio vai ser substituído, pergunta `y/N`
(`--yes` responde sim). O config só é gravado **depois** da verificação passar.

**A armadilha do ambiente.** O ambiente vence o arquivo (`core/config.py`, `ENV_ALIASES`). Uma
`QDRANT_URL` ou `RECALL_EMBED_URL` legada no rc do shell continua mandando os processos para o
endereço antigo, com o arquivo certo. Depois de gravar, a etapa resolve os endpoints efetivos
(`core.load()`) e compara com os da stack. Para cada diferença, nomeia a variável que vence e o
valor dela (endereço não é segredo), e diz para tirar o `export` do rc. Não edita o rc: o rc é do
usuário, como o wizard já faz com o PATH.

## Detecção **[a validar]**

- **Comandos**: todo comando da etapa (`info`, `version`, `compose`, `nvidia-smi`) roda com
  timeout, como processo filho que lidera uma sessão própria, para que o timeout mate o grupo de
  processos inteiro: o provider de compose roda o backend como filho dele. Por estar noutra
  sessão, o Ctrl-C do terminal chega só ao Python; então o runner repassa um SIGINT ao grupo do
  comando (o que o terminal teria entregado a um grupo em primeiro plano), espera até 5 s, manda
  SIGKILL ao que sobrar e só então deixa o Ctrl-C seguir. Um segundo Ctrl-C durante a espera vai
  direto ao SIGKILL. Sem isso, o comando interrompido seguia rodando, solto (medido em 2026-10-07
  com o runner anterior: o neto do comando sobreviveu ao Ctrl-C do processo que o rodava).
- **Runtimes**: `docker info` e `podman info` em JSON, com timeout. Deles saem versão, os/arch do
  engine (é o que vale para a imagem e os devices, porque no macOS e no Windows o engine roda numa
  VM), kernel, rootless e socket. No Docker, `--format '{{json .}}'`, que toda versão aceita: o
  atalho `--format json` é recente, e um CLI antigo imprime a palavra `json` no lugar do JSON. O
  sistema do engine é o `OSType`; o `OperatingSystem` é um rótulo, como "Docker Desktop".
- **Provider de compose**: `docker compose version`, senão `docker-compose version`;
  `podman compose version`, senão `podman-compose version`. O `podman compose` diz no stderr qual
  provider externo executa (`Executing external compose provider "<caminho>"`); com esse aviso
  desligado, o docker-compose se reconhece pela linha de versão no stdout, nas duas grafias
  (`Docker Compose version` do v2 em diante, `docker-compose version` no v1). Com o
  docker-compose atrás do Podman, o socket da API precisa responder, e a etapa confere
  **conectando nele**, não pelo `podman info`: no Podman 5.7.0, o `remoteSocket.exists` do
  `podman info` veio `true` com o socket inexistente, e o `podman compose` então falhou com
  "failed to connect to the docker API" (medido em 2026-10-06). Com o socket parado e o
  `podman-compose` instalado, a etapa usa o `podman-compose`, que não precisa de socket, e diz
  isso numa linha. Sem nenhum dos dois, a correção no Linux é habilitar o `podman.socket` do
  usuário. O provider usado fica gravado no `stack.json`, mas como informação
  (status e texto de erro): o ciclo de vida o **re-deriva do runtime gravado**,
  não o lê do arquivo. Isso é seguro porque a razão para gravá-lo já foi
  neutralizada: os dois providers rotulam os containers de forma diferente só
  no nome DEFAULT, e cada serviço leva um `container_name` explícito (M6),
  então um `down`/`up`/`remove` que segue um restart endereça os mesmos
  containers e o mesmo volume quem quer que os tenha criado. Re-derivar também
  permanece certo quando o estado do socket muda: com o socket da API parado, o
  `podman compose` (o wrapper do docker-compose) precisa ceder ao
  `podman-compose` standalone, e o nome gravado apontaria para o errado.
- **Hardware, no Linux nativo**: o barramento PCI `/sys/bus/pci/devices/*` (classe `0x03`, os
  vendors 0x1002 AMD, 0x8086 Intel e 0x10de NVIDIA; o ASPEED de BMC, 0x1a03, e as funções de
  áudio, classe `0x04`, são ignorados), `/dev/dri/renderD*` e `nvidia-smi -L`. Lê o PCI, não o
  `/sys/class/drm`, porque um `cardN` só aparece lá com um driver anexado, e a placa sem driver
  (o caso do NVIDIA sem driver) ficaria invisível. A prontidão da NVIDIA tem duas partes, cada
  uma com a correção nomeada quando falta:
  - o toolkit: no Docker, o `nvidia-container-runtime-hook` no PATH (qualquer versão) ou o
    `nvidia-cdi-hook` a partir do Docker 29.2; no Podman, o spec CDI, gerado por
    `nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`, como root (sem `--output`, o
    comando só imprime o spec e não grava nada); quando o toolkit não está instalado, a
    correção diz primeiro para instalá-lo;
  - o ICD Vulkan da NVIDIA no host. Sem ele, a correção é instalar o componente GL/Vulkan do
    driver (no Ubuntu, `libnvidia-gl-<versão>`) e, no Podman, regenerar o spec CDI.

  Sem placa NVIDIA no host, o perfil não é oferecido. Com a placa no barramento e o `nvidia-smi`
  sem listá-la (nouveau, ou driver nenhum), a correção é o driver da NVIDIA. Um `nvidia-smi` que
  trava (um driver quebrado pode travá-lo, e o runner desiste no timeout de 30 s) ou que falha
  conta como um que não lista placa nenhuma: a detecção segue, e a CPU continua no menu.

  A prova continua sendo o `--list-devices` do container.
- **Windows (WSL2, com Docker Desktop ou Podman)**: entra na fase 3, já com os perfis de GPU da
  imagem própria (ver "Imagem própria e GPU no Windows"). Antes disso, a etapa diz que o Windows
  ainda não é suportado e aponta o caminho manual do README, em vez de oferecer CPU. O WSL é
  reconhecido pelo kernel: `microsoft` no `/proc/sys/kernel/osrelease` do host (WSL2
  `...-microsoft-standard-WSL2`, WSL1 `...-Microsoft`) ou no kernel do engine (`KernelVersion` do
  `docker info`, `host.kernel` do `podman info`). O `/etc/os-release` é o da distro e não diz
  nada sobre o WSL.
- **macOS**: o `apple` só existe em Apple Silicon e só no Podman com máquina libkrun, conferida
  pelo `Host.VMType` do `podman machine info`. O `podman machine inspect` não traz o tipo da VM
  (lido no código do Podman: o `InspectInfo` das tags `v5.7.0` e `v6.0.0` não tem esse campo);
  dele sai o socket da API do lado do host, `ConnectionInfo.PodmanSocket.Path`, da máquina que o
  `machine info` dá em `Host.CurrentMachine`. É esse socket que o `podman compose` passa ao
  docker-compose no Mac (`cmd/podman/compose.go`); o `remoteSocket` do `podman info` é o caminho
  dentro da VM. No Docker Desktop, ele aparece no menu como "só Podman"; com máquina applehv,
  indisponível, com uma correção só, a mesma no Podman 5 e no 6: gravar o provider libkrun no
  `containers.conf` e recriar a máquina (ver "Compatibilidade"). Em Mac com Intel não há caminho
  de GPU para container: o menu mostra a CPU e diz por quê.
- **Outros**: disco livre antes de baixar; a RAM é o número do engine (o que os containers
  usam, não o do host, que no macOS é maior); portas por `bind` em `127.0.0.1`; SELinux por
  `/sys/fs/selinux/enforce`.

## Download dos modelos **[a validar]**

- Baixa para `<arquivo>.part`, retoma por Range, calcula o sha256 enquanto baixa e confere bytes e
  hash antes do rename atômico.
- Arquivo já presente com o hash certo é pulado: a etapa é idempotente.
- Hash errado: apaga o `.part` e falha dizendo isso.
- Segue os redirects do Hugging Face para a CDN. A integridade vem do hash, não do host.
- Progresso pelo `Reporter` (MiB, %, MiB/s). Ctrl-C deixa o `.part`, e rodar de novo retoma.

## Verificação e calibração **[a validar]**

É aqui que entram as medições do spike recusado, feitas na máquina do usuário.

1. **Prontidão**: Qdrant `/readyz` 200; llama-server `/health`, que devolve 503 enquanto carrega e
   200 quando pronto. Timeout de 10 minutos.
2. **Funcional**: reusa `core.setup.diagnose()` com um `Config` montado a partir do plano e
   considera só os checks de Qdrant, Embedding (dimensão igual a `vector_size`) e Re-rank (Paris
   em primeiro). A coleção de memória é a etapa seguinte do wizard. Aqui o Re-rank falho
   bloqueia, ao contrário do `diagnose`, onde ele é aviso: no `diagnose` o rerank é opcional
   porque pode não existir, e aqui ele é um dos três serviços que a etapa acabou de subir.
3. **Calibração**, comparada com o orçamento de cada host, depois de uma chamada de aquecimento
   em cada servidor (medido em 2026-10-06: a primeira chamada custa bem mais que as seguintes,
   1,39 s contra 0,21 s no embed e 2,93 s contra 1,07 s no rerank, na mesma GPU, e é o regime
   quente que o recall de cada host vive):
   - embed de um texto de 6000 caracteres (o teto do chunk), que também prova o `-ub 8192`;
   - rerank de 20 pares (o `TOP_K` do recall com rerank, `hooks/recall.py:135`) de 2400
     caracteres;
   - memória dos containers, pelo `stats` do runtime.

   Os orçamentos chegam do `cli`:
   - claude-code: embed 8,0 s e rerank 6,0 s (`hooks/recall.py:147-148`);
   - hermes: 2,0 s cada (`HERMES_PREFETCH_BUDGET_S / 4`, `hosts/hermes/__init__.py:63,682-688`).

   Um teste lê os dois arquivos por AST para os números não divergirem. O resultado é, por host,
   ok ou aviso. Exemplo de aviso: "no hermes, o rerank levou 4,1 s contra 2,0 s; o recall vai
   pular o rerank, e um perfil de GPU resolve". Nunca bloqueia.

## Ciclo de vida: `qctx stack` **[a validar]**

| comando | faz |
|---|---|
| `qctx stack status [--json]` | lê o `stack.json`; saúde dos três por sonda nos endpoints (sem tocar num runtime: o status é barato e read-only, e a sonda dos endpoints é o sinal autoritativo de up/down); pins contra o catálogo; se o config aponta para a stack; se a stack volta no boot; e, se a stack está para baixo, aponta `qctx stack up`. Sai 1 se a stack gerenciada não está saudável. Sem stack gerenciada, diz isso e sai 0 |
| `qctx stack up [--upgrade] [--image ROLE=REF]` | re-renderiza a partir do `stack.json` (ou do catálogo, com `--upgrade`), roda `compose up -d` e espera a prontidão |
| `qctx stack down` | `compose down`, mantendo volume e modelos |
| `qctx stack remove [--purge-models] [--purge-data]` | `compose down` e apaga `compose.yaml` e `stack.json`. `--purge-models` apaga os GGUFs. `--purge-data` apaga o volume do Qdrant, ou seja o acervo, e exige digitar o nome do projeto (ou `--yes`). Nunca mexe no config: diz o que ficou apontando para `127.0.0.1` |
| `qctx stack install [--listen ADDR] [--keys] [--profile PERFIL] [--runtime docker\|podman] [--yes]` (fase 2) | provisiona só a stack, sem host e sem tocar no config do plugin (ver "Modo servidor") |
| `qctx stack keys show\|rotate` (fase 2) | `show` reimprime as chaves; `rotate` gera novas e recria os containers |
| `qctx stack connect <host>\|<https-base> [--ports Q,E,R] [--yes]` (fase 2) | aponta o plugin desta máquina para uma stack remota (ver "Conectar o plugin a uma stack remota") |

O `stack.json` guarda: versão do esquema, papel (local ou servidor), endereço de escuta, runtime,
provider, perfil, device, portas, referências de imagem em uso, arquivos e hashes dos modelos,
versão do Qdrant instalada, fase e datas.

**Guarda de minor do Qdrant.** `up --upgrade` recusa quando a minor do catálogo passa de `minor
instalada + 1`, dizendo qual minor intermediária usar e como (override).

**Reboot.**
- Docker: `restart: always` mais o daemon no boot (no Docker Desktop, "start at login").
- Podman no Linux: imprime, sem executar, os comandos que habilitam o `podman-restart.service` do
  usuário e o linger da sessão.
- Podman no macOS e no Windows: `podman machine start`.

Em todos os casos, `qctx stack status` acusa a stack parada e aponta `qctx stack up`.

## Modo servidor (fase 2) **[a validar]**

Para servir a stack a outras máquinas, sem o plugin nem host nenhum nesta:

```
git clone https://github.com/erickstryck/memories-plugin
memories-plugin/bin/qctx stack install --listen <ip>
```

É o mesmo caso de uso da etapa local, com outro papel: escuta no endereço escolhido, exige chaves
e não grava o config do plugin. Precisa de Python 3 e de Docker ou Podman. Na fase 2, só servidor
Linux.

### Endereço de escuta

- Os candidatos vêm de `ip -j addr`, classificados em loopback, tailnet (`100.64.0.0/10`),
  privado (RFC 1918) e público.
- Sem `--listen`, a etapa sugere o IP da tailnet, se houver, e senão pergunta. `0.0.0.0` e IP
  público só com `--listen` explícito, e com aviso.
- Fora do loopback, a chave é obrigatória: sem chave, a stack não sobe. No loopback, `--keys`
  liga as chaves para quem vai pôr um proxy na frente.
- As três portas são publicadas só nesse endereço, e é ele a fronteira, não o firewall: com Docker
  rootful, porta publicada passa por fora do ufw, porque o Docker faz o DNAT antes da cadeia
  INPUT. O `qctx` não mexe em firewall; imprime, como sugestão, regras de ufw ou firewalld
  restritas à rede escolhida.

### Chaves

- São duas, geradas com `secrets.token_urlsafe(32)`: uma do Qdrant e uma dividida pelos dois
  llama-servers, porque o plugin tem uma chave só para embedding e rerank (`api_key` em
  `core/config.py`).
- O Qdrant recebe a dele por `QDRANT__SERVICE__API_KEY`, num `env_file`; os llama-servers, por
  `--api-key-file`, montado `:ro`. O `/health` do llama-server continua público (só `/health` e
  `/v1/health`, `tools/server/server-http.cpp:251-254` na `b11382`), e a checagem de prontidão do
  Qdrant manda a chave.
- Ficam em `<stack>/secrets/`, com o diretório em 0700. Nunca vão para o `compose.yaml`, o
  `stack.json` ou o config.
- `qctx stack keys show` reimprime; `qctx stack keys rotate` gera novas e recria os containers com
  `up -d --force-recreate`, porque o ambiente só é relido na recriação.
- No fim da instalação, a etapa imprime as chaves uma vez e a linha de `qctx stack connect` para
  as outras máquinas.

### HTTPS, por conta do usuário

A documentação traz duas receitas, testadas antes de entrarem nela. Nas duas, a stack escuta em
`127.0.0.1` com `--keys`, e o proxy publica um endereço só, com os caminhos `/qdrant`, `/embed` e
`/rerank`:

- `tailscale serve`, com o certificado automático da tailnet;
- um proxy próprio (nginx, Caddy) que já termine o HTTPS da máquina.

O `qctx` não configura nenhum dos dois (decisão 11).

## Conectar o plugin a uma stack remota (fase 2) **[a validar]**

`qctx stack connect` aponta o plugin desta máquina para uma stack remota, e a etapa do
`qctx install` chama o mesmo caso de uso pela opção "stack remota".

| alvo | o que grava |
|---|---|
| `<host>` | `qdrant_url`, `api_base_url` e `rerank_url` em HTTP, nas portas 6333, 8003 (`/v1`) e 8004 (`/v1/rerank`) |
| `https://<base>` | `<base>/qdrant`, `<base>/embed/v1` e `<base>/rerank/v1/rerank`, o layout das receitas de HTTPS |

- `--ports Q,E,R` troca as portas. O `rerank_url` vai sempre explícito: derivado do
  `api_base_url`, cairia na porta do embedding (`resolved_rerank_url` em `core/config.py`).
- Pede as duas chaves sem eco, pelo mesmo caminho do wizard, e as grava onde o wizard grava
  credenciais (`~/.hermes/.env` ou `~/.secrets`), como `QCTX_QDRANT_API_KEY` e `QCTX_API_KEY`.
  Nunca as aceita como argumento, que ficaria no histórico do shell e no `ps`. Com `--yes`, lê as
  duas do ambiente.
- Testa antes de gravar: `core.setup.diagnose()` com o config novo, com Qdrant, embedding
  (dimensão igual ao `vector_size`) e rerank. Mostra o diff, grava só se passar e checa a
  armadilha do ambiente, como a etapa local.
- Para HTTP fora do loopback, avisa que as chaves trafegam em claro, a menos que a rede cifre,
  como a tailnet ou uma VPN.

## Imagem própria e GPU no Windows (fase 3) **[a validar]**

### A imagem

`images/llama-dzn/Dockerfile`, em dois estágios:

1. build: Ubuntu 26.04, a mesma base da oficial, com `meson`, `ninja`, `python3-mako` e
   `directx-headers-dev`. Baixa o tarball do Mesa 26.0.3 e confere o sha256 do catálogo. Compila
   só o dzn, com `-Dvulkan-drivers=microsoft-experimental` e os demais drivers, plataformas, GL e
   LLVM desligados.
2. final: `FROM` a `server-vulkan-b11382` pelo digest, mais `libvulkan_dzn.so` e o manifesto ICD
   do dzn. O llama.cpp continua o da imagem oficial.

Só amd64. A versão do Mesa acompanha a da imagem oficial, para o dzn não divergir dos outros
drivers.

### Build e publicação

- `.github/workflows/llama-dzn.yml`, com disparo manual e três entradas: a build do llama.cpp, o
  digest da base e a versão do Mesa. Em PR que mexa no Dockerfile, só o build, sem push.
- Publica `ghcr.io/erickstryck/llama-dzn:b11382-mesa26.0.3` com atestado de proveniência
  (`actions/attest-build-provenance`). O digest publicado entra no catálogo pelo bump.
- Pacote novo no GHCR nasce privado. Torná-lo público é um passo manual, feito uma vez, para o
  pull anônimo funcionar.

### No Windows

- Com o engine no WSL2 (Docker Desktop ou `podman machine` com WSL), todos os perfis usam a
  `llama-dzn`. O `cpu` existe ali só como reserva, como no Linux: máquina sem GPU, ou GPU que não
  passa na verificação numérica. Sem `/dev/dxg`, ele roda no backend de CPU.
- Os perfis `amd`, `intel` e `nvidia` recebem `/dev/dxg`, `/usr/lib/wsl` e `LD_LIBRARY_PATH` (ver
  "Perfis de backend"). A GPU aparece no `--list-devices` como `Microsoft Direct3D12 (<GPU>)`, e
  o vendor sai do nome entre parênteses.

### Verificação numérica

O dzn se declara não conforme, então a etapa não confia só no "funcionou". Embeda textos fixos no
perfil escolhido e na CPU (a mesma imagem, sem device) e compara:

- a ordem das similaridades entre os textos tem de ser a mesma;
- o desvio tem de ser pequeno frente aos intervalos que decidem os cortes, e não comparado com um
  limiar absoluto de cosseno;
- a ordem do rerank de pares fixos também tem de bater.

Se falhar, oferece CPU e diz por quê.

## Erros **[a validar]**

- `StackError(CoreError)`, com a etapa e a correção. O `main()` atual já imprime.
- Todo subprocess tem timeout. O do `pull` é de 30 minutos, com o progresso visível. Na falha,
  mostra o final do log do serviço (`compose logs --tail 50 <serviço>`).
- `stack.json` registra a última etapa concluída. Rodar de novo retoma: o download continua e o
  compose é idempotente.
- O config nunca fica pela metade: só é gravado depois da verificação.

## Testes **[a validar]**

Offline e herméticos, como o resto da suíte (`tests/isolation.py`):

- `facts`: árvores `/sys` falsas e saídas reais de `nvidia-smi`, `docker info`, `podman info`,
  `podman machine info` e `podman machine inspect`, por plataforma.
- `runtimes`: teste de contrato. `Docker`, `Podman` e `Fake` produzem o mesmo formato de argv e
  interpretam as saídas de versão e info.
- `process`: o timeout e o Ctrl-C param o grupo de processos inteiro, provados com processos de
  verdade (o Ctrl-C, por um processo intermediário com o tratador padrão de SIGINT, que é quem
  recebe o sinal).
- `backends`: a matriz de disponibilidade. `devices_seen` sobre amostras de `--list-devices` no
  formato da `b11382` (RADV, ANV, NVIDIA, Venus e `(none)`), e o desempate pelo `uma:`. A matriz
  de compatibilidade: há fixture para plataforma, runtime e perfil se e só se `runtimes()` inclui
  o runtime, e o rótulo do menu sai dela.
- `compose`: as fixtures golden de cada fase (9, 11 e 15). O emissor: todo valor sai por
  `json.dumps`, e as chaves seguem a regra.
- `fetch`: transporte falso. Retoma do `.part`; hash errado apaga e falha; tamanho errado falha;
  arquivo certo é pulado. A barra de progresso é testada com transporte falso e detecção de TTY
  injetável: os dois modos (reescrita em TTY, linha por fatia fora dele), a retomada continuando
  a barra de onde parou, e os campos (porcentagem, MiB, velocidade, ETA) saindo do byte correto.
- `health`: relógio falso. 503 seguido de 200; timeout.
- `installer`, com fakes:
  - o caminho feliz de cada perfil;
  - GPU do host que o container não vê: aparece indisponível, com o motivo; com `--stack`
    explícito, a etapa para;
  - opção que o runtime em uso não atende: aparece com o runtime que exige e a correção; escolhida,
    volta ao menu; `--stack apple --runtime docker` é recusado;
  - os dois runtimes: a pergunta diz o que cada um atende; `--yes` fica com o Docker; o
    `--stack apple` vai para o Podman sem perguntar;
  - porta ocupada: propõe outra;
  - aviso da armadilha do ambiente;
  - config só depois da verificação;
  - `--yes` sem `--stack` não provisiona;
  - `--check` não escreve (mtimes inalterados).
- `lifecycle`: códigos de saída do `status`; `up`, `down` e `remove`; a guarda de minor; a
  confirmação do `--purge-data`.
- Arquitetura: `core`, `hooks` e `hosts` não importam `stack`; `stack` não importa `hooks`,
  `hosts` nem `cli`.
- Orçamentos: AST de `hooks/recall.py` e de `hosts/hermes/__init__.py`.
- Higiene do scanner: nenhum arquivo rastreado com download encadeado num shell por pipe.
- Fase 2:
  - `exposure`: a classe de cada endereço; `0.0.0.0` e IP público só explícitos; fora do
    loopback, sem chave não sobe;
  - `keys`: geração, permissões, `rotate` recria; a chave nunca no compose, no `stack.json` ou no
    config;
  - `connect`: os endereços de `<host>` e de `https://<base>`; `--ports`; config gravado só depois
    do `diagnose`; chaves pelo `SecretSink`, nunca por argumento; o aviso de HTTP fora do
    loopback;
  - o ramo "stack remota" do wizard chama o mesmo caso de uso.
- Fase 3:
  - `DxgGpu`: disponível só com o engine no WSL2; `devices_seen` sobre `Microsoft Direct3D12 (…)`;
  - a regra de imagem por plataforma;
  - a verificação numérica, com vetores falsos que passam e que falham;
  - o Dockerfile confere o sha256 do Mesa.

Integração opt-in, pulada por padrão: `QCTX_STACK_IT=1` roda o perfil `cpu` com o runtime real, em
portas alternativas, e faz `up`, verificação, calibração, `down` e `remove`. Com
`QCTX_STACK_IT_GPU=amd|intel|nvidia`, roda também o perfil de GPU. A fase 2 acrescenta
`QCTX_STACK_IT_LISTEN=<ip>`, que sobe a stack em modo servidor e conecta a ela pelo próprio
`qctx stack connect`.

## Documentação **[a validar]**

- `README.md`:
  - em "## Install, by OS", antes dos blocos por SO, uma linha dizendo que o wizard sobe a
    infraestrutura sozinho com Docker ou Podman;
  - em "## Local models", uma subseção "### Or let the wizard do it", com `qctx install --stack
    auto`;
  - o caminho manual fica, e os testes de fidelidade continuam o fixando.
- `docs/usage.md`: o grupo `qctx stack` e as flags `--stack` e `--runtime`.
- `docs/install.md`: o que a etapa faz e custa (tamanhos, portas, onde ficam os dados, reboot) e
  a tabela de compatibilidade, opção a opção: Docker, Podman ou os dois. Um teste, no estilo do
  `tests/test_readme_fidelity.py`, a confere contra `runtimes()`.
- `docs/architecture.md`: `stack/` no Layout e uma linha na fronteira.
- Fase 2: `docs/install.md` ganha "Stack em outra máquina", com `qctx stack install`,
  `qctx stack connect` e as duas receitas de HTTPS. Os exemplos usam `<host>` sem esquema, ou
  HTTPS, porque os testes só aceitam endereço `http://` em localhost.
- Fase 3: `docs/install.md` diz o que muda no Windows e de onde vem a imagem própria.
- Regras que os testes já impõem e continuam valendo:
  - todo comando `qctx` citado existe;
  - nenhum travessão longo;
  - endereço `http://` só localhost;
  - e, pelo scanner do hermes, nunca o download encadeado num shell.

## Verificações de abertura do plano

O spike foi recusado, então o plano de cada fase começa pelas verificações dela, numa máquina que
o usuário autorizar e pela integração opt-in.

Fase 1:

1. O YAML gerado é aceito por `docker compose`, por `podman compose` (com o docker-compose de
   provider) e por `podman-compose`: `compose config` nas 9 fixtures da fase; as fases 2 e 3
   repetem com as delas.
2. Perfil `cpu` com a `server-vulkan`:
   - o llvmpipe é ignorado;
   - o embed de 6000 caracteres e o rerank de 20 x 2400 passam com `-ub 8192`;
   - latência e memória com 4, 8 e todas as threads, para calibrar o texto dos avisos.
3. Podman rootless com `/dev/dri`: a anotação `run.oci.keep_original_groups` chega ao crun pela API
   compatível? Se não chegar, a alternativa é `group_add: keep-groups` no podman-compose, ou
   documentar o acesso por ACL de sessão.
4. `restart: always` com o `podman-restart.service` traz a stack de volta depois de um reboot.
5. Qdrant `-unprivileged` com volume nomeado funciona no Podman rootless e no Docker.
6. `compose run --rm --no-deps embed --list-devices` dá a mesma visão de devices que o serviço.
7. A linha `ggml_vulkan: <n> = … | uma: …` sai junto com o `--list-devices`? Se não sair, o
   desempate entre dedicada e integrada cai para só a memória livre, e o texto do menu muda.

Sem hardware do projeto para provar: `nvidia` fica "verificado só por documentação" (o ICD
chegando ao container, no Docker e no CDI do Podman); `apple` fica experimental, e a correção
que o menu mostra para ele (o provider libkrun no Podman 5 e no 6) foi lida no código do Podman,
não executada.

Fase 2:

1. O llama-server lê o arquivo de chave montado `:ro` com `cap_drop: [ALL]`, no Docker rootful e
   no Podman rootless. O root do container sem `CAP_DAC_OVERRIDE` não lê arquivo 0600 de outro
   dono; se for o caso, o arquivo fica legível e a proteção fica no diretório 0700.
2. A `-unprivileged` do Qdrant aceita a chave pelo `env_file`, e a checagem de prontidão funciona
   com ela.
3. Escutar num IP da tailnet funciona no Podman rootless e no Docker.
4. As duas receitas de HTTPS, de ponta a ponta, com `qctx stack connect https://…`.

Fase 3, numa máquina Windows que o usuário fornecer:

1. O workflow compila o dzn e publica a imagem.
2. No Docker Desktop, o `/dev/dxg` e o `/usr/lib/wsl` chegam ao container, com cada vendor que o
   hardware disponível permitir; sem NVIDIA, sem `--gpus`.
3. O backend Vulkan do llama.cpp aceita o device do dzn (o requisito é
   `storageBuffer16BitAccess`), e embedding e rerank passam na verificação numérica.
4. A latência, contra a CPU da mesma máquina.
5. O mesmo com `podman machine` em WSL.

## Riscos

- As imagens de GPU do llama.cpp não são testadas pelo CI além do build (`docs/docker.md`
  oficial).
- Em CPU, o orçamento de 2,0 s do rerank no hermes provavelmente estoura, e o recall daquele host
  roda sem rerank. A calibração diz isso na hora, e um perfil de GPU resolve.
- A memória exigida por `-c/-b/-ub 8192` não foi medida. A calibração mede e reporta.
- Os providers de compose diferem. Mitigação: só o subconjunto comum e `compose config` antes do
  `up`.
- O acesso por grupo ao `/dev/dri` no Podman rootless é provado pelo `--list-devices`, não
  presumido.
- O caminho `apple` depende do krunkit/Venus com o Mesa 26.0.3 sem o patch que o guia de 2025
  usava. Não verificado.
- NVIDIA via Vulkan depende do componente Vulkan do driver no host e da injeção do ICD pelo
  toolkit (#767 no CDI). O `--list-devices` prova, e a falha nomeia a correção. O desempenho do
  Vulkan contra o CUDA na NVIDIA, nestes modelos, não foi medido.
- O Windows só chega na fase 3; até lá, quem usa Windows segue pelo caminho manual do README.
- Fase 2: a chave em HTTP trafega em claro fora de rede cifrada, e a etapa avisa. Com Docker
  rootful, porta publicada passa por fora do ufw, e a fronteira real é o IP de escuta. A chave do
  Qdrant aparece no `inspect` para quem tem acesso ao engine, o que no Docker rootful equivale a
  root.
- Fase 3: o dzn é não conforme, e um erro numérico estragaria vetores em silêncio; a defesa é a
  verificação numérica. A imagem passa a depender da conta do GitHub do projeto: o digest fixado
  e o atestado de proveniência limitam o estrago, não o evitam. Não está verificado que o Docker
  Desktop entrega o `/dev/dxg` ao container sem NVIDIA. E tudo depende de uma máquina Windows que
  o usuário forneça.
- A armadilha do ambiente: detectada e avisada, não corrigida.
- Disco: 294 MiB da imagem do llama.cpp, ~71 MiB do Qdrant, 836 MiB de modelos, mais o acervo.
- A stack local não tem autenticação: só `127.0.0.1`, mas outro usuário local da mesma máquina
  alcança as portas. Fica documentado.
- O scanner do hermes bloqueia a instalação com um único padrão crítico. Arquivo novo não pode
  trazer nenhum, e o teste de higiene cobre o mais provável.

## Fora de escopo, nas três fases

- Imagens ROCm, SYCL e CUDA (o catálogo permite depois).
- llama.cpp nativo fora de container, em qualquer plataforma.
- Stack parcial (só os modelos, ou só o Qdrant).
- Escolha de quantização (o catálogo tem só Q4_K_M).
- Migrar um acervo existente para o Qdrant local.
- Autenticação na stack local da etapa do wizard; as chaves são do modo servidor.
- HTTPS embutido na stack, e o `qctx` configurar `tailscale serve`, proxy ou firewall.
- Servidor macOS ou Windows na fase 2, e Windows em ARM na fase 3.
- Instalar Docker, Podman, driver ou toolkit; criar unidades do systemd (são impressas, não
  executadas).
- Docker Model Runner.

## Achado à parte, fora desta spec

O comando de embedding do README (`## Local models`, passo 3) sobe o servidor só com
`--embedding`, ou seja com o `ubatch` padrão de 512. Pelo código da `b11382` citado em "Flags dos
servidores", entrada acima de 512 tokens é recusada, e o chunk típico do plugin tem ~800. Medido em
2026-10-06: com esse comando, um chunk de 2400 caracteres volta HTTP 500 "input (568 tokens) is too
large to process. increase the physical batch size (current batch size: 512)", e um de 1200 passa.
A correção é independente desta spec: acrescentar `-c 8192 -b 8192 -ub 8192` ao comando de embed e
fixá-lo no teste de fidelidade, como as flags do rerank já são fixadas.

O mesmo caminho manual tem um segundo defeito, medido no mesmo dia: `qctx config set api-base-url
http://127.0.0.1:8003` faz o plugin chamar `/embeddings`, que na `b11382` devolve uma lista crua e
quebra o `Embedder` (ver "O que grava no config"). A correção do README é `http://127.0.0.1:8003/v1`.
E o `Embedder` levanta `AttributeError`, que não é `CoreError`, para uma resposta nesse formato; o
`diagnose` não o captura. As três correções ficam para depois, com o usuário.

## Resultado das verificações de abertura da fase 1

Medido em 2026-10-06 numa máquina Linux com Podman 5.7.0 rootless (crun 1.21), o docker-compose
v5.2.0 como provider externo do Podman e o podman-compose 1.6.0, sem Docker engine, com duas Intel
Arc B70 e uma Radeon RX 6900 XT. As fixtures da medição vieram de um renderizador descartável no
formato desta spec.

| # | verificação | resultado |
|---|---|---|
| 1 | YAML aceito pelos providers | 9 de 9 em `docker-compose config -q`, `podman compose config` e `podman-compose config`. O Docker engine não estava na máquina |
| 2 | perfil `cpu` com a `server-vulkan` | sem `/dev/dri`, o `--list-devices` mostra `(none)`: o llvmpipe é ignorado. Embed de 6000 caracteres e rerank de 20 x 2400 passam com `-ub 8192`. Quente, com 4, 8 e todas as threads: embed 4,69, 2,61 e 0,97 s; rerank 35,8, 19,3 e 7,0 s. Memória: embed ~1,83 GB, rerank ~2,78 GB |
| 3 | anotação `run.oci.keep_original_groups` no Podman rootless | chega ao crun pela API compatível (docker-compose sobre a API 1.41 do Podman) e pelo podman-compose: os grupos vão de `0(root)` para `0(root)` mais dez `65534`. Os nós `renderD*` daquela máquina são 0666, então a recusa de permissão sem a anotação não pôde ser reproduzida |
| 4 | `restart: always` e o `podman-restart.service` | os dois providers criam o container com `RestartPolicy=always`, e o comando do `ExecStart` da unidade (`podman start --all --filter restart-policy=always`) religou um container parado. O reboot em si não foi executado |
| 5 | Qdrant `-unprivileged` com volume nomeado | funciona no Podman rootless: `/readyz` 200 em ~1 s, os dados sobrevivem a `down` e `up`, roda como uid 1000. O Docker não foi testado |
| 6 | `compose run` vê o mesmo que o serviço | sim: `compose run -T --rm --no-deps embed --list-devices` e `compose exec embed /app/llama-server --list-devices` no serviço rodando dão a mesma lista |
| 7 | a linha `uma:` sai com o `--list-devices` | não, nem com `-lv 4`. O desempate fica só pela memória livre |

Medido também, com a GPU Intel (`-dev Vulkan0`): embed de 6000 caracteres em 0,21 s e rerank de
20 x 2400 em 1,07 s, quentes (1,39 s e 2,93 s na primeira chamada), ~200 MB de RSS por container e
~945 MB de VRAM para os dois modelos. Com esses números, o rerank em CPU estoura os orçamentos dos
dois hosts (7,0 s contra 6,0 s e 2,0 s), e na GPU cabe nos dois.

O que a medição mudou nesta spec, cada item corrigido no seu lugar: `-dev none` no perfil `cpu` e
`--no-ui` ("Flags dos servidores"); `/v1` no `api_base_url` e no `rerank_url`, e o `embed_url`
esvaziado ("O que grava no config"); o socket conferido por conexão, com o podman-compose de
reserva ("Detecção"); o `-T` no `compose run` ("O que ela faz, em ordem"); o desempate só pela
memória livre ("Escolha da GPU"); o `container_name` e o nome real do volume ("Portas, caminhos e
nomes"); o aquecimento antes da calibração ("Verificação e calibração").
