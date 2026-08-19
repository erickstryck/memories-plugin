# Wizard de instalação e verificação — design

**Data:** 2026-08-19 · **Repo:** `memories-plugin` @ `b8008f7` · **Estado:** aprovado em
brainstorming, pronto para plano de implementação

## O problema, medido

O README manda configurar tudo com `qctx` — nos passos 2, 3 e 4 da própria tabela de
instalação, e em outras 30 linhas. Numa máquina nova instalada pelo caminho oficial,
`qctx` **não existe**.

A causa é uma só, medida em 2026-08-19 nesta máquina:

- `bin/qctx` só chega ao PATH por um `ln -s` **manual**, documentado apenas no fluxo de
  `git clone`. Nenhum dos dois caminhos oficiais o executa.
- `hermes plugins install …` clona para `~/.hermes/plugins/memories/`. `bin/qctx` está lá,
  sem link nenhum.
- `claude plugin install …` **copia por SHA** para
  `~/.claude/plugins/cache/memories-plugin/memories-plugin/<SHA>/`. Nesta máquina há
  **cinco** pastas de SHA, e `installed_plugins.json` aponta para uma. O caminho muda a cada
  `plugin update` — um symlink apontando para ele morre no update seguinte, em silêncio.

E a documentação envelheceu junto. Medido no mesmo dia:

| o README afirma | o repositório tem |
|---|---|
| "three hooks" | **4** — recall, checkpoint, lease, bigfile (`hooks/hooks.json`) |
| "1140 collected, 1123 offline" | **1169 testes, 1150 offline** (19 pulados) |
| "20 tools" e "22 tools", no mesmo arquivo | um número só, e nenhum dos dois é conferido |

Duas coisas diferentes, então: uma instalação que exige passos que o README não dá, e um
README que não bate com o código. O wizard resolve a primeira; um teste resolve a segunda.

## O que se decidiu

Decisões do usuário no brainstorming de 2026-08-19:

| Decisão | Escolha | Recusado |
|---|---|---|
| Entrada | **bash fino + cérebro em Python** | só subcomando `qctx`; um wizard por host |
| PATH | **launcher que se resolve sozinho** | symlink para caminho estável; symlink por SHA |
| Alcance | **executa, perguntando por etapa** | só imprimir o comando; um OK único para tudo |
| Segredos | **pergunta e escreve nos arquivos de credencial** | só nomear o que falta; só o `.env` |
| README | **corrige e trava com teste** | só corrigir à mão |

Princípio que atravessa o desenho, nas palavras do usuário: *"simples é sempre melhor"*.
Ele já cortou duas peças que eu tinha proposto — ver "O que foi cortado".

## Arquitetura

| peça | papel | novo? |
|---|---|---|
| `scripts/install.sh` | localiza a árvore e o `python3`, faz `exec`. **Zero lógica** | sim |
| `core/install.py` | os checks de encanamento, como **dados** | sim |
| `cli/qctx.py: cmd_install` | renderiza, pergunta, executa, delega aos cutovers | sim |
| `bin/qctx` | ganha o resolvedor; o mesmo arquivo serve na árvore e como cópia no PATH | +~30 linhas |
| `core/setup.py` | ganha o check de aviso de `context_window`, que hoje ninguém faz | +~10 linhas |
| `scripts/cutover.sh`, `scripts/hermes_cutover.sh` | **inalterados** | não |

**A decisão estrutural é o que NÃO se escreve.** O estado de cada host continua sendo
assunto exclusivo dos dois cutovers, que já existem, já fazem backup datado, já reconferem a
escrita e já falam o vocabulário `ok / FAIL / WARN`. O wizard **roda o cutover em dry-run**
para montar a seção daquele host e **chama o mesmo script com `--apply`** quando o usuário
confirma. Reimplementar 975 linhas de verificação de host em Python criaria duas fontes de
verdade que divergem no primeiro conserto — conta que este repo já pagou com três cópias do
pipeline de dois estágios.

### A fronteira: `core/` não conhece host

`core/setup.py` declara a regra e o motivo (a prosa em torno de `COMMAND_PREFIX`).
`core/install.py` obedece: só encanamento — PATH, launcher, config legível sem shell.
Detecção de host e chamada dos cutovers ficam em `cli/qctx.py`, que já nomeia os dois hosts
na sua própria saída (`cli/qctx.py:708,726`).

### O resolvedor, dentro do próprio `bin/qctx`

Um arquivo só, servindo os dois papéis. Ordem, a primeira que existir vence:

```
$QCTX_HOME                          override explícito (e o que os testes usam)
a árvore do próprio script          resolvendo symlinks, como já faz hoje
~/.hermes/plugins/memories          caminho estável, quando há hermes
installPath de installed_plugins.json    o SHA vivo do claude, lido na hora
```

Quem instalou por symlink — o caso de desenvolvimento — cai no segundo e nada muda. A cópia
em `~/.local/bin` cai no terceiro ou no quarto. `plugin update` troca o SHA e o quarto caso
resolve sozinho na chamada seguinte: nada para relinkar, nada que envelheça calado.

**Sem caminho de desenvolvimento embutido.** Uma entrada `~/dev/memories-plugin` na ordem
resolveria o caso do autor e seria lixo no repositório público de todo mundo. `QCTX_HOME`
cobre o mesmo caso, explicitamente.

O launcher é **copiado** para `~/.local/bin/qctx`, então pode envelhecer em relação à
árvore. O check é uma comparação de bytes (`filecmp.cmp`) contra `<root>/bin/qctx`, e a
correção é copiar de novo. Sem número de versão para alguém esquecer de subir — a mesma
razão pela qual `plugin.yaml` não declara `version`.

### `scripts/install.sh`

A única coisa que pode rodar antes de `qctx` existir, e por isso a única em bash:

```bash
bash ~/.hermes/plugins/memories/scripts/install.sh          # instalado pelo hermes
bash ~/.claude/plugins/cache/…/<SHA>/scripts/install.sh     # instalado pelo claude
./scripts/install.sh                                        # clonado
```

Resolve symlinks para achar a raiz (a rotina que `bin/qctx` já tem), confere que há
`python3`, e faz `exec python3 "$root/cli/qctx.py" install "$@"`. Nada além disso: toda
decisão está do outro lado do `exec`, onde a suíte offline alcança.

## O que `qctx install --check` reporta

Quatro seções, e só a segunda é código novo:

1. **alcance e configuração** — `core.setup.diagnose()`, que já existe quase inteiro:
   Qdrant, embedding com a dimensão real, rerank com a escala, as cinco coleções — mais o
   aviso de `context_window` que esta spec acrescenta a ele.
2. **encanamento** — `core/install.py`: `qctx` no PATH e resolvendo para esta árvore;
   `~/.local/bin` no PATH; launcher idêntico ao da árvore; e o **teste sem shell promovido a
   check de primeira classe** (`core.load(env={})` — hoje o README ensina isso como conselho
   manual com `env -i`, e é a falha silenciosa que ele mesmo diz que as pessoas cometem).
3. **claude-code** — `scripts/cutover.sh` em dry-run, se houver `claude` no PATH.
4. **hermes** — `scripts/hermes_cutover.sh` em dry-run, se houver `hermes` no PATH.

`--check` nunca escreve. `--json` devolve 1 e 2 estruturados e 3 e 4 como
`{exit_code, text}` — a saída dos scripts é deles, não vale reformatar.

**A suíte não roda na fase de plano.** Os dois cutovers rodam `unittest discover` nos seus
checks: 41s medidos, por host, só para exibir uma lista. O wizard pede para pular na fase de
plano e a roda **uma vez antes de qualquer `--apply`** — que é exatamente o que
`hermes_cutover.sh` já exige, ao recusar `--apply` com a suíte não verificada.

## A ordem do wizard

1. **bash**: acha a árvore e o `python3`, entrega para o Python.
2. **PATH**: instala ou atualiza `~/.local/bin/qctx`. Se o diretório não estiver no PATH,
   imprime a linha exata para o rc do shell — e **não escreve** no rc.
3. **config**: as duas passadas descritas abaixo — o que bloqueia primeiro, todo o resto
   depois com Enter mantendo. Chaves com eco desligado. `vector_size` detectado.
4. **hosts**: detecta `claude` e `hermes`; por host, instala se faltar (abaixo) e então
   plano em dry-run → confirmação → apply.
5. **reverifica**, incluindo o check sem shell.
6. **imprime o que só um humano pode fazer**: aprovar o shell hook do hermes num TTY,
   reiniciar o claude para os hooks entrarem.

Confirmação **por grupo**, não por comando e não uma só para tudo: escrever uma URL e
substituir o provedor de memória do hermes não merecem o mesmo "sim". `--yes` responde sim a
todos, para script. Sem TTY e sem `--yes`, o comando diagnostica e sai — a mesma regra que
`cmd_setup` já segue, e pelo mesmo motivo: um `input()` esperando resposta que não vem
pendura a chamada de um agente.

### O que se pergunta, e a prova de que nada falta

Simples não pode custar completo. `Config` tem **15 campos** e o wizard tem de alcançar
todos, então a pergunta não é quais perguntar, é em que ordem — e como provar depois que
nenhum ficou de fora.

**Duas passadas.** A primeira pede só o que bloqueia (`diagnose` chama de *blocker*); a
segunda percorre **todo o resto** mostrando o valor atual ou o default, com Enter mantendo.
Uma passada linear com Enter é simples de atravessar e não deixa campo inalcançável.

| passada | campos | como |
|---|---|---|
| 1 — bloqueia | `qdrant_url`, `api_base_url`, as duas chaves, `memory_collection` | pergunta direto; a coleção com as sugestões de `suggest_collections` |
| 2 — Enter mantém | `embed_url`, `rerank_url`, `embed_model`, `rerank_model`, as outras quatro coleções, `context_window` | valor atual entre colchetes |
| detectado | `vector_size` | escrito do que o endpoint respondeu, como `cmd_setup` já faz |

Três descobertas da auditoria, todas medidas, e nenhuma delas estava coberta pelos passos
que eu tinha escrito:

- **`rerank_url` derivado dá endereço errado nesta máquina.** Quando vazio,
  `resolved_rerank_url()` monta `api_base_url + /rerank`. A configuração real do usuário tem
  embedding em `:8003/v1` e rerank em `:8004/v1/rerank` — portas diferentes. Um wizard que
  pergunta só `api_base_url` produz um rerank quebrado e um check amarelo que ninguém pediu.
  `embed_url` tem a mesma forma e o mesmo risco.
- **`context_window` não é checado por ninguém.** `diagnose()` não olha para ele, e a guarda
  de arquivo grande fica **inerte** sem ele para modelo fora de `MODEL_WINDOWS` — que
  resolve para janela 0, e 0 libera tudo (medido em 2026-08-17; o `MiniMax-M2.7` do hermes
  cai exatamente aí). Hoje só `hermes_cutover.sh` reporta, então numa máquina só-claude nada
  avisa. **Entra como check de aviso em `core.setup.diagnose`**, não só no wizard: quem roda
  `qctx setup` merece o mesmo aviso, e é uma linha no lugar certo em vez de duas no errado.
- **`embed_model` e `rerank_model` têm default** (`bge-m3`, `bge-reranker-v2-m3`) que não é
  garantia. Servidor com outro modelo faz o check de embedding falhar dizendo que o modelo
  não responde — com a correção a um Enter de distância, na segunda passada.

**As duas chaves são perguntadas como todo o resto.** O que muda nelas é o **destino**, não
se o wizard pergunta: os outros treze campos vão para o `config.json`, as duas chaves vão
para os arquivos de credencial descritos abaixo. `SECRET_FIELDS` é uma regra sobre onde se
escreve, nunca sobre o que se pergunta.

**A prova.** Um teste compara a lista de campos que o wizard percorre com **todo** o
`config.DEFAULTS` — as quinze, chaves incluídas — e falha se sobrar de qualquer lado. Um
segundo teste, curto, verifica o destino: depois de um apply completo, nenhuma das duas
chaves aparece no `config.json`, em nenhuma grafia. Assim a completude e o sigilo são duas
asserções separadas, e nenhuma das duas pode ser satisfeita esquecendo a outra.

Campo novo em `Config` quebra o primeiro teste até ser colocado numa das passadas — é a
mesma lição que `_check_collections` já aprendeu na marra ("ALL FIVE, and the count is the
point": ele enumerava três de cinco e reportava `ready` numa configuração em que o acervo de
repositórios estava por cima do de memória).

O que o wizard **não** faz, e diz: subir um Qdrant ou um servidor de embedding. Sem os dois
não há o que configurar, e é o passo 1 da tabela do README — o wizard checa, nomeia o que
não respondeu e para ali, em vez de perguntar quinze coisas que não vão funcionar.

### Instalar no host, que nenhum cutover faz

Os dois cutovers registram hooks e conferem configuração; **instalar o plugin no host não é
trabalho deles**. Então essa peça é do wizard, em `cli/qctx.py`, e é curta — checar e, se
faltar, oferecer o comando exato para rodar:

| host | como se sabe que está instalado | o que o wizard oferece rodar |
|---|---|---|
| claude-code | `memories-plugin@memories-plugin` em `installed_plugins.json` | `claude plugin marketplace add …` + `claude plugin install …@…` |
| hermes | `$HERMES_HOME/plugins/memories` existe (dir ou symlink) e `hermes config get memory.provider` responde `memories` | `hermes plugins install … --enable --force` + `hermes config set memory.provider memories` |

Duas coisas que o wizard **explica antes de perguntar**, porque as duas são decisão do
usuário e não detalhe:

- o `--force` do hermes existe porque o scanner classifica esta árvore como `caution` — ela
  traz dois scripts que editam configuração do host, que é o trabalho declarado dos
  cutovers. O wizard mostra esse motivo e o comando; não esconde o `--force` no meio.
- apontar `memory.provider` para `memories` **substitui** o provedor externo que estiver
  ali, porque o hermes ativa exatamente um. O wizard mostra qual é o valor atual antes de
  perguntar.

Se o host não estiver no PATH, sua seção inteira é pulada com uma linha dizendo isso — uma
máquina que só tem um dos dois hosts é o caso normal, não uma falha.

### Os segredos

As duas chaves nunca entram no `config.json`; `core.save()` recusa, e continua recusando. O
wizard lê com eco desligado (`getpass`) e escreve **só** em arquivo que existe para
credencial:

- `~/.hermes/.env` (`chmod 600`) — o único jeito de um hermes de systemd, gateway ou cron
  ter chave, porque ele não herda shell nenhum.
- para o shell interativo, um arquivo que o usuário nomeia; default `~/.secrets` **se já
  existir**. Não cria nada por conta própria.

Nunca ecoa o valor: confirma por nome da variável e comprimento. No `--check`, diz se a
chave está presente e em qual grafia — todas as que o core aceita, três para a do Qdrant,
porque checar menos grafias do que o core aceita seria pior do que não checar.

**Numa máquina só-claude não existe arquivo de credencial do plugin**, e isto tem de ser
dito em vez de resolvido por baixo: se `~/.secrets` (ou o que o usuário nomear) não existir
e ele não quiser criar, o wizard imprime as duas linhas de `export` e deixa o check
**pendente**, não `ok`. Chave que só vive na sessão de shell corrente é chave que some no
próximo terminal, e um wizard que dissesse "pronto" ali estaria mentindo do mesmo jeito que
o README que originou este trabalho.

Já configurado é reconhecido, não re-perguntado: chave presente aparece como
`[already set in ~/.hermes/.env]` e Enter mantém. Trocar exige digitar por cima, de
propósito — o wizard roda de novo a cada verificação, e um fluxo que pede as chaves toda vez
treina a pessoa a colar segredo no terminal por reflexo.

## O README

A seção de instalação passa a ter o wizard na frente e o caminho manual embaixo — mantido,
não apagado: é o que se usa quando o wizard falha, e é o que documenta o que ele faz. Os
números medidos entram corrigidos.

### O teste de fidelidade

O que ataca a causa em vez do sintoma, e o que mantém isto verdade daqui a um mês. Um teste
offline, pequeno:

- extrai todo `qctx <subcomando> <sub-subcomando>` citado no README e exige que cada um
  exista no argparse de `cli/qctx.py`;
- confere os contadores que o README afirma contra a fonte: hooks em `hooks/hooks.json`,
  skills em `skills/*/SKILL.md`.

Só isso. Não valida prosa, não valida exemplo de saída, não conta testes — um teste que
afirma o número de testes é um teste que quebra a cada teste novo, e o remédio vira ruído.

## O que foi cortado, por simplicidade

| cortado | por quê |
|---|---|
| Marcador `# launcher-version: N` | `filecmp` responde a mesma pergunta sem número para manter |
| `~/dev/memories-plugin` no resolvedor | caminho pessoal em repo público; `QCTX_HOME` cobre |
| Reimplementar os checks de host em Python | duplicaria 975 linhas já testadas em campo |
| Contador de testes no teste de fidelidade | quebraria a cada teste novo |

## Testes

Offline, com `HOME` temporário, sem rede e sem dependência — como todo o resto da suíte:

- **resolvedor**: cada uma das quatro posições da ordem, e a precedência entre elas, com
  árvores falsas; `QCTX_HOME` vencendo as outras três.
- **launcher velho**: cópia diferente da árvore é detectada; idêntica não gera check.
- **plano**: as quatro combinações de host presente (nenhum, só claude, só hermes, os dois).
- **completude**: os campos que o wizard percorre são exatamente `DEFAULTS`, as duas chaves
  incluídas; sobrar de qualquer lado falha.
- **`context_window`**: `diagnose()` avisa quando não está declarado e cala quando está.
- **segredo**: depois de um apply completo, `config.json` não contém nenhuma das duas
  chaves, em nenhuma grafia.
- **`--check` não escreve**: mtime da árvore de config e de `~/.local/bin` inalterados.
- **fidelidade do README**: o teste descrito acima.

## Riscos

- **O wizard roda comandos de host.** É o que foi pedido, e a mitigação é a confirmação por
  grupo mais o `--check` que nunca escreve. Os cutovers, que fazem as escritas de peso, já
  tiram backup datado e reconferem o resultado.
- **`scripts/install.sh` é mais uma porta de entrada.** Aceito porque é a única que funciona
  antes de `qctx` existir, e porque ela não decide nada: quem decide está do outro lado de
  um `exec` coberto por testes.
- **O launcher copiado diverge da árvore.** Coberto pelo check de bytes, que é a primeira
  coisa que `--check` reporta.
