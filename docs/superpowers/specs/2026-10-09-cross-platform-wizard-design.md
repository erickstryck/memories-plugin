# O wizard da stack em todas as plataformas: desenho

Uma spec de acréscimo à [stack de memória provisionada pelo `qctx`](2026-10-05-local-stack-design.md),
que ela governa onde não houver colisão. Registra a reversão da decisão 14: o Windows
entra nesta versão, via WSL2, com a mesma jornada das outras plataformas.

## O que se decidiu

Em 2026-10-09, ao perguntar como instalar a stack numa máquina que só serve a stack
(uma máquina Windows com Docker Desktop, sem `hermes` nem `claude`), o usuário fixou a
experiência: clonar o repo, rodar um install, e daí tudo automático: avaliar se existe
`docker` e `podman` (e o WSL2, no Windows), verificar o hardware, mostrar as opções de
acordo com o hardware, instalar com barra de progresso ao selecionar, e terminar com o
resumo do que foi feito, as URLs e a api-key padrão quando houver. Pediu também que a
jornada **não difira entre plataformas**, e confirmou que as dependências pesadas
(WSL2, Docker) são **só verificadas e reportadas**: quem as instala é o usuário.

Em 2026-10-09, a decisão 14 (o Windows só na fase 3, com GPU) é **revertida**: o Windows
entra nesta versão. O que dela sobrevive: a GPU no Windows passa pela imagem própria dzn
(a oficial não chega à GPU), e o `cpu` continua a reserva para máquina sem GPU ou GPU
que não passa na verificação.

| # | decisão | texto | alternativa recusada |
|---|---|---|---|
| 17 | jornada única | a sequência de passos é a mesma em Linux, macOS e Windows; a plataforma muda só a entrada e o que existe por baixo (WSL2), nunca as perguntas nem a ordem | uma jornada com perguntas específicas do Windows (ex.: escolher CPU ou GPU antes do menu) |
| 18 | dependências só verificadas | o install verifica WSL2 e Docker/podman e, ausentes, aborta nomedando a dependência, o comando que instala e para; nunca tenta instalá-las | o install auto-instala WSL2 (`wsl --install`) ou o Docker Desktop |
| 19 | entrada do Windows | `scripts/install.ps1`, em PowerShell: verifica e delega ao `install.sh` dentro da distro (`wsl -d <distro> -- bash ...`); instala apenas o `python3` da distro (`apt`, com consentimento) quando falta | wizard nativo em Windows (portar fatos e compose); exigir que o usuário abra a distro à mão |
| 20 | WSL2 com runtime é Windows | um host WSL2 cujo WSL2 responde a `docker` (ou a `podman`) devolve a plataforma `windows`; a recusa de uma linha fica para Windows sem WSL2 | continuar recusando qualquer WSL2 (o estado da fase 1); ou aceitar WSL2 sem runtime e deixar o passo de runtime abortar |
| 21 | GPU no Windows pelo dzn | os perfis de GPU rodam a imagem própria dzn com `/dev/dxg` e `/usr/lib/wsl` montados; o menu os oferece **somente quando a prova passa**, como já acontece no Linux com a imagem oficial | oferecer o item GPU "na fé" e deixar o compose falhar; ou esconder GPU no Windows |
| 22 | a imagem dzn vive no repo | `images/llama-dzn/Dockerfile` + workflow que publica no GHCR + pin por digest no catálogo; se o pin não estiver publicado ou atingível, o wizard faz `docker build` do Dockerfile (o repo está clonado) e segue | exigir o publish como pré-requisito; ou assumir que o build local é o único caminho |
| 23 | o resumo final | o passo de término mostra, em todas as plataformas: o que foi feito, as URLs do Qdrant, do embedding e do rerank, a api-key padrão quando houver (o Qdrant local não tem chave; se o usuário configurou uma, ela aparece), e o que fazer após o reboot | terminar no "stack.json: running" e no hint de reboot, como está hoje |
| 24 | onde o estado mora, no Windows | `stack.json` e o config ficam na home da distro WSL2; os containers ficam no Docker Desktop; `qctx stack status\|up\|down\|remove` funcionam de dentro da distro, e o `install.ps1 -Command status` é o atalho do lado Windows | duplicar o estado no perfil Windows (duas fontes para o mesmo stack) |

## A jornada, passo a passo

A sequência, **idêntica** nas três plataformas:

| passo | o que o sistema faz | onde está |
|---|---|---|
| 1. clona e roda o install | Linux/macOS: `./scripts/install.sh`. Windows: `.\install.ps1`. A porta de entrada não decide nada além do que é específico da plataforma; depois do `exec`, tudo é o mesmo Python | `install.sh` existe; `install.ps1` é novo |
| 2. existe `docker` e `podman`? | descobre os runtimes que **responderem** `info` (Docker antes de Podman; um binário presente mas mudo é pulado, não erro). Nenhum responde, **aborta** nomedando cada um dos dois e a correção | já existe (o `discover` e a recusa `step="runtime"`); no Windows o WSL2 é verificado **antes**, na entrada |
| 3. verifica o hardware | RAM, disco, GPUs pelo barramento PCI e render nodes; no WSL2 a RAM e o disco que os containers têm ficam na VM do **engine** (docker-desktop), não na da distro: cada distro WSL2 é uma VM separada, e o número sai do próprio engine (`TotalMemory` do `docker info`, o limite da `podman machine`), como o macOS já faz pela memória | o caminho do engine já existe; o WSL2 lê o mesmo número do engine, sem `/proc` da distro |
| 4. mostra as opções | o menu lista o perfil `cpu` e um item por GPU **provada** (a prova é `compose run ... --list-devices` na definição que de fato vai rodar); o que não pode ser oferecido aparece marcado indisponível, com o motivo e a correção | já existe (o "menu diz por quê") |
| 5. instala com barra | pull das imagens (barra nativa do provider) e download dos modelos, 836 MiB, com porcentagem, MiB, velocidade e ETA; só começa **depois** do resumo confirmado | já existe (decisão 16) |
| 6. resumo final | o que foi feito, as três URLs, a api-key padrão quando houver, e o retorno após o reboot | **novo** (decisão 23) |

O passo 2 no Windows, na entrada (`install.ps1`), antes de qualquer outra coisa:

1. WSL2: nenhum distro WSL2 ativo -> aborta com `wsl --install` (admin + reboot,
   por isso o script não tenta).
2. Runtime na distro: o socket do Docker Desktop é exposto dentro da distro; se nem
   `docker` responde nem `podman` existe, aborta nomedando as três causas possíveis
   (Docker Desktop desligado, a integração WSL desmarcada, podman não instalado).
3. `python3` na distro: se falta, `apt install python3` **com o sim do usuário**.

Depois, o `install.ps1` executa o `install.sh` da cópia clonada dentro da distro e não
faz mais nada: a partir do `exec`, é a jornada da tabela, igual.

## O que muda no código

### A plataforma

`facts.platform_of` passa a tratar WSL2 com runtime que responde como `windows`
(decisão 20): hoje o WSL2 caía no ramo de recusa antes de o runtime ser descoberto.
A ordem nova do passo de plataforma: sem WSL2 no Windows, a recusa de uma linha não
muda; WSL2 presente, o `discover` decide (nenhum runtime, a recusa `step="runtime"`;
há runtime, a plataforma é `windows`). A recusa de uma linha do `install` (`_windows_line`)
sobra para o caso que a entrada `.ps1` não cobre: quem rode o `qctx` a mão num Windows
sem WSL2.

O compose ganha o fixture do Windows (`windows-dzn`), com os mounts de WSL2
(`/dev/dxg`, `/usr/lib/wsl`, o `LD_LIBRARY_PATH`), ao lado dos fixtures de Linux e
macOS. O vendor da GPU no WSL2 sai do nome que o dzn imprime,
`Microsoft Direct3D12 (<GPU>)`: o que está entre parênteses é o nome, e o vendor é
lido dele, como o dzn se declara na spec-mestra.

### A prova de GPU e o menu

Não há caminho novo de prova: o `_prove` já roda o perfil na definição que vai rodar
(`compose run -T --rm --no-deps embed --list-devices` no probe file). No Windows, o
probe file é o compose do Windows, com a imagem dzn: o que o container vê na prova é
o que verá depois. Falhou a prova, o perfil desce para indisponível com o rabo do
stderr, como já acontece; GPU que não chega ao container, o motivo diz por quê.

A **verificação numérica** do dzn é a da spec-mestra, e vale para o item GPU do
menu no Windows: embedar textos fixos no perfil dzn e no mesmo perfil sem device,
e comparar a **ordem** das similaridades (tem de bater), o **desvio** frente aos
intervalos que decidem os cortes (não um limiar absoluto de cosseno) e a **ordem**
do rerank de pares fixos. Falhou, o item sai do menu com o motivo. O perfil `cpu`
no WSL2 não usa o dzn (roda a imagem oficial, sem device, como no Linux): ele
passa pela verificação funcional que já existe, sem a numérica.

### A imagem dzn

`images/llama-dzn/Dockerfile`, em dois estágios, **como a spec-mestra descreve**
(base Ubuntu 26.04, dzn do Mesa 26.0.3 compilado só o driver
`microsoft-experimental`, final `FROM` a `server-vulkan-b11382` pelo digest).
Só amd64.

- Publicação: `.github/workflows/llama-dzn.yml`, disparo manual (build, digest da
  base, versão do Mesa); em PR que mexa no Dockerfile, só o build. Publica
  `ghcr.io/erickstryck/llama-dzn:b11382-mesa26.0.3` com atestado de proveniência.
  O pacote novo no GHCR nasce privado: torná-lo público é um passo manual, uma vez.
- Pin: o digest publicado entra no **catálogo**, dono único (mesmo mecanismo
  `QCTX_STACK_IMAGE_LLAMA`/`QDRANT` de override), e o fixture do Windows referencia
  o pin, nunca o tag nu.
- Fallback (decisão 22): antes do pull, o passo confere se o pin está atingível;
  não está, faz `docker build -t <pin-local> images/llama-dzn/` a partir do repo
  clonado, reporta que usou o build local e segue. O install nunca fica bloqueado
  por publicação.

### O resumo final

Um módulo novo, `stack/summary.py`, que lê o `Plan` e o config e imprime o bloco:

```
  ok    stack: running (docker, cpu)
        qdrant   http://127.0.0.1:6333
        embed    http://127.0.0.1:8003/v1
        rerank   http://127.0.0.1:8004/v1/rerank
        api-key  (nenhuma: Qdrant local sem chave)   <- ou a chave, quando houver
  ..    após o reboot: o Docker Desktop volta sozinho (config do Desktop); os
        containers voltam com `qctx stack up`
```

O módulo não conhece plataforma (ele recebe o que foi gravado), e o passo de término
o chama depois de gravar o config: o resumo mostra o que está **no disco**, não o que
a memória do processo acha.

## A entrada: `scripts/install.ps1`

Regras, na ordem:

1. Não decide nada da jornada: verifica (WSL2, runtime na distro, `python3`), e
   delega. O `install.sh` continua sendo a única porta de decisão.
2. As três verificações do passo 2 da seção anterior, cada uma com seu abort
   (dependência, comando que instala, e que o install não vai instalar por ele).
3. Escolhe a distro: o `-Distro <nome>` explicita; sem ele, o primeiro distro WSL2
   na ordem de `wsl -l -v` que estiver `Running`/`Stopped` (não `Not Installed`).
   Sem nenhum, aborta (regra 1).
4. `.\install.ps1 --stack auto` (ou qualquer arg do wizard) passa tudo ao
   `install.sh`; `.\install.ps1 -Command status|up|down|remove` roda
   `qctx stack <cmd>` na distro (decisão 24), sem re-disparar o wizard.
5. O exit code do processo dentro da distro é o exit code do script: o abort do
   wizard (sem runtime, sem disco, recusa do resumo) não vira "terminou com erro
   genérico" do PowerShell.

O script é **testável sem Windows**: as três verificações e a escolha de distro são
funções puras sobre a saída dos comandos (mesma regra do `install.sh`: nada de
decisão no shell que a suíte não veja). A suíte injeta as saídas de `wsl -l -v` e dos
`docker info`/`podman info` da distro e confere os aborts e o comando delegado.

## S.O.L.I.D

- **S**: a coleta de fatos por plataforma (Linux, macOS, WSL2), a prova de perfil
  (imagem oficial, dzn), e a descoberta de runtime (com o WSL2 na frente, no
  Windows) são unidades com uma responsabilidade cada; a jornada de 12 passos não
  cresce.
- **O**: o fixture do Windows + o pin do dzn no catálogo + a entrada `.ps1` são a
  extensão; nenhum passo dos 12 muda de assinatura. Uma plataforma nova entra pelos
  mesmos três pontos.
- **L**: quem consome a plataforma (`platform_of`) recebe o valor novo `windows` e
  só o que o passo precisa dele (a chave do fixture); quem prova não sabe se a
  imagem é a oficial ou a dzn (o probe file já resolve).
- **I**: a prova dzn depende do mesmo protocolo que a prova de hoje (o runner do
  compose + o parse do `--list-devices`); não nasce protocolo novo para um caso.
- **D**: o pin do dzn tem dono único no catálogo (a lição do `PORT_FALLBACK_OFFSET`);
  o texto do resumo final mora em `summary.py`, não espalhado nos passos.

## Testes

- **Unidade**: `platform_of` com WSL2 (com runtime, sem runtime, com e sem
  `microsoft` no kernel); a escolha de distro e os três aborts do `.ps1` com saídas
  injetadas; o fallback do pin (atingível, não publicado, build local); o resumo
  final com e sem api-key; o vendor do nome `Microsoft Direct3D12 (NVIDIA ...)`.
- **Fixture**: o compose do Windows no conjunto dos fixtures (o mesmo esquema dos
  de Linux/macOS), e a prova dzn contra ele (o `_prove` não muda: a fixture muda).
- **Spikes, antes do código**: (1) o dzn num Docker Desktop real: o container com
  `/dev/dxg` lista a GPU no `--list-devices`, e a verificação numérica passa;
  (2) o socket do Docker Desktop respondendo `docker info` dentro de uma distro
  WSL2 (a premissa do passo 2). Se o spike 1 falhar, o item GPU sai do menu com o
  motivo e o escopo segue com o CPU (a jornada não muda; o menu já foi desenhado
  para isso).
- **Integração opt-in**: o perfil `windows` na suíte de integração (mesma mecânica
  do `QCTX_STACK_IT`), executável só onde houver WSL2 + GPU que passe no dzn.

## Documentação

- `README.md`: a instalação ganha o bloco do Windows (o `.ps1`, as três verificações,
  que quem instala WSL2/Docker é o usuário) e a seção "Local models" do manual fica
  com o papel que já tem: o caminho sem o plugin.
- `install.md`: a seção "uma máquina só de stack" (a do standalone) passa a dizer
  que Windows entra pela jornada normal, e remove a afirmação de que o Windows "não
  está no caminho, nesta versão".
- A spec-mestra: a linha da decisão 14 e a do roadmap ganham a marca de reversão
  apontando para esta spec.

## Fora de escopo

- Servir **outra** máquina com autenticação (`listen`, chaves, `connect`): segue
  como a fase 2 da spec-mestra, e vale para todas as plataformas.
- Windows **nativo** sem WSL2: impossível para containers; não existe esse alvo.
- GPU NVIDIA por CDI: o dzn cobre GPU em geral via D3D12; uma NVIDIA que o dzn não
  lista sai do menu com o motivo.

## Riscos

- O dzn no Docker Desktop/WSL2 é **documental** até o spike: a evidência de origem
  é um comentário de 2026-02 na issue do WSLG. Mitigação: o spike abre o plano, e
  a falha dele tem caminho (CPU no menu, com o motivo).
- A RAM que o passo 3 mede no WSL2 é a da **VM do engine** (o limite do Docker
  Desktop), não a do host Windows nem a da distro: é o número que importa para o
  orçamento dos containers, mas o resumo deve nomear isso para não confundir.
- A spec-mestra diz que **todos os perfis no WSL2** usam a dzn (ela cobre o CPU,
  pois é a oficial + o driver). Esta spec diverge de propósito, por ser posterior:
  o `cpu` roda a imagem oficial sem device (como no Linux), e a dzn entra nos
  perfis de GPU. Se o teste mostrar que a dzn roda o CPU melhor do que a oficial
  no WSL2, a converte-se de volta para "todos pela dzn" sem tocar na jornada.
- O pin do dzn depende do pacote no GHCR ser público (um passo manual, uma vez): o
  fallback de build local cobre o intervalo em que ele não está.
