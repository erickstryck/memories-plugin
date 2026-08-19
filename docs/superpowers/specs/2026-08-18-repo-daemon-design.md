# Indexação de repositório em segundo plano — desenho

Sub-projetos **B, C, D e E** de cinco. O **A** (modelo de dados, sub-coleções por repo) foi
entregue em 2026-08-17; estes quatro são o resto do pedido original, e o usuário pediu em
2026-08-18 que fossem revisados e seguidos juntos.

Nas palavras dele, o requisito inteiro: *"identifica o projeto no git, dá a opção de indexar,
se disser que sim, será executada a indexação assíncrona, cancelável e com um jeito fácil de
acompanhar o status, se o projeto sofrer atualizações então a indexação ocorre em segundo
plano atualizando os chunks relacionados às mudanças"*.

## O que motivou a revisão

`repos add` é **síncrono** e só aceita caminhos explícitos. Medido em 2026-08-18: 0,19 s para
um arquivo, 0,51 s para três — ou seja ~0,13 s por arquivo. Um projeto de 1.800 arquivos
elegíveis trava o terminal por vários minutos, sem progresso e sem cancelamento. Nas palavras
do usuário: *"inadmissível ter isso sem ser assíncrono"*.

## CORREÇÃO DE REGISTRO: não existe proibição de daemon neste projeto

A decomposição de 2026-08-17 adiou o sub-projeto D com a justificativa de que *"D contraria o
'no daemon' declarado em `core/docs.py`"*. **Isso estava errado, e o erro foi meu.** A frase
naquele arquivo é:

> `Qdrant has no native expiry, so the TTL is exactly this: a delete-by-filter on each call,`
> **`no daemon`**`.`

Ela descreve **como o TTL de documentos temporários expira** — uma nota sobre um mecanismo, não
uma regra de arquitetura. Generalizei uma linha sobre expiração numa proibição de projeto e usei
essa proibição inventada para adiar duas vezes justamente a parte que o usuário mais precisava.
Ele corrigiu: *"pode usar daemon, não sei de onde tirou isso que daemon é proibido"*.

**A restrição que É real, e continua valendo: só stdlib.** Está no README, com o motivo —
dependência faltando dentro de um hook vira perda silenciosa de funcionalidade. Daemon pode;
dependência externa não. É por isso que a vigilância aqui é por *polling* e não por `inotify`:
medido, varrer o `mtime` de 2.000 arquivos custa **16 ms**, o que torna a dependência
desnecessária e o código independente de plataforma.

## O que se decidiu

| Decisão | Escolha | Alternativas recusadas |
|---|---|---|
| Execução | **Um daemon por usuário**, iniciado sob demanda | processo por repo; sem processo (síncrono); systemd |
| Ciclo de vida | **Morre quando o último host morre** | vive indefinidamente; encerra por ociosidade |
| Como detecta a morte do host | **Lease com `(pid, starttime)`**, verificado a cada ciclo | sinal na saída; timeout de heartbeat sozinho |
| Onde o lease é escrito | **Ponto próprio, uma vez por sessão** | pendurado no hook de recall |
| Vigilância | **Polling de `git ls-files` + `mtime`** | `inotify` (dependência / só Linux); só hooks de git |
| Quais arquivos entram | **`git ls-files`** menos binário, minificado, lockfile e acima do teto | varrer o disco; globs manuais |
| Gatilho de atualização | **Só o daemon** | git hooks (removidos — ver "O que sai") |
| Cancelamento | **Mantém o que já indexou** | desfazer o parcial |

## Arquitetura

Um processo, `qctx repos daemon`, que faz duas coisas e nada mais:

1. **Executa trabalhos** de uma fila — indexação inicial de um repo, ou reindexação de arquivos
   que mudaram.
2. **Vigia** os repos indexados e enfileira trabalho quando algo muda.

Todo o estado vive em `~/.memories-plugin/state/`, em JSON, e é isso que torna o `status`
trivial: ele lê arquivo, não conversa com o daemon.

```
state/
  daemon.json          pid, starttime, iniciado_em
  leases/<sessao>.json host, pid, starttime, iniciado_em
  jobs/<repo>.json     tipo, total, feitos, arquivo_atual, estado, cancelar
```

**Não há protocolo entre processos.** Comandos escrevem arquivos; o daemon lê no ciclo
seguinte. Um socket ou pipe seria mais imediato e traria negociação, formato de fio e um modo
de falha novo (o daemon vivo mas surdo) — o custo não se paga para uma fila que muda uma vez por
minuto.

## Ciclo de vida: o daemon morre com o host

Requisito do usuário: *"o daemon deve ser morto quando o claude ou hermes sai/morre"*.

**Um lease é um bilhete de "estou vivo"**, escrito uma vez por sessão, contendo o **pid do
processo do host** e o **`starttime`** dele. O daemon, a cada ciclo, confere cada lease:

- `os.kill(pid, 0)` falha → o processo morreu → lease removido;
- o pid existe mas o `starttime` mudou → é **outro** processo que reusou o número → lease
  removido;
- nenhum lease vivo → **o daemon encerra**.

O teste é a existência do processo, não um aviso do host. Por isso cobre igualmente a saída
limpa e o `kill -9`. A latência é um ciclo.

**Como cada host descobre o próprio pid** — medido em 2026-08-18:

| host | mecanismo |
|---|---|
| hermes | o provedor roda **dentro** do processo do hermes: `os.getpid()` já é o host |
| claude-code | o hook é subprocesso; sobe a árvore em `/proc` até achar o executável `claude` |

Medido no claude-code: `python3 → bash → claude (pid 1557842) → bash → ptyxis`. Se a subida não
encontrar o host — outro sistema operacional, árvore inesperada —, o lease grava o pid que
achou e **registra que a resolução foi aproximada**; o pior caso é o daemon sobreviver ao host e
ser encerrado no `status` seguinte, nunca o contrário.

**Onde o lease é escrito, e por que não no recall:** um ponto próprio em cada host — um hook
`SessionStart` no claude-code, `initialize()` no hermes. O hook de recall roda a cada prompt e
seria conveniente, mas recall é busca de memória e lease é sinal de vida: acoplá-los faria
desligar o recall encerrar o daemon com o host vivo. Decisão do usuário em 2026-08-18.

## B — o que entra, e a varredura

A fonte é **`git ls-files`**. Isso respeita o `.gitignore` por definição, e `node_modules`,
build e cache ficam de fora sem regra especial. Um arquivo não versionado não é indexado —
quase sempre o certo, e sempre explicável.

Sobre essa lista, quatro descartes:

| descarte | por quê |
|---|---|
| binário | `_read_source` já detecta e recusa; aqui é antecipado para não pagar a leitura |
| minificado | uma linha de 40 KB não responde pergunta nenhuma e enche o acervo |
| lockfile | `package-lock.json` e afins: enorme, gerado, sem conteúdo que alguém procure |
| acima do teto | um arquivo de 5 MB vira centenas de chunks e domina toda busca daquele repo |

O funil é **relatado antes de começar**, porque uma contagem que aparece só no fim é uma
contagem que ninguém usa para decidir:

```
12.403 rastreados → 1.847 elegíveis (214 binários, 3 lockfiles, 12 acima de 1 MB)
```

## C — detecção e consentimento

`candidates_for(root, remotes)` **já existe** no core desde o sub-projeto A e **nenhum host a
chama** — é código morto hoje, e este sub-projeto é o consumidor que faltava.

Ela responde: é repo novo, junta-se a um já indexado (mesmo remote), ou o nome sugerido já está
tomado por um repo não relacionado. Apresentar é do host:

- **CLI** — `qctx repos init` mostra o que achou e pergunta. Sem TTY, imprime o comando que
  faria e não escreve nada, seguindo o que `setup` já faz.
- **hermes e claude-code** — a skill instrui o modelo a oferecer quando o diretório é um repo
  git não indexado. A ferramenta que ele chama devolve os candidatos; **quem decide é o
  usuário**, e o modelo apresenta.

**Indexar nunca começa sem aceite.** O que muda depois do aceite é que a *atualização* passa a
ser automática — o consentimento é dado uma vez, para o repo, e não a cada escrita. Isso
**reverte** a decisão do sub-projeto A (*"escrever no acervo é sempre pedido"*), e a reversão é
o pedido do usuário: *"se o projeto sofrer atualizações então a indexação ocorre em segundo
plano"*.

## D — execução, status e cancelamento

```bash
qctx repos init                  # detecta, oferece, e ao aceitar enfileira a indexação
qctx repos status                # o que roda, o que é vigiado, e se o daemon está vivo
qctx repos cancel <repo>         # cancela o trabalho daquele repo
qctx repos daemon stop           # encerra o daemon e para de vigiar tudo
```

O `status` lê arquivos de estado, então funciona mesmo com o daemon morto — e nesse caso
**diz isso**, em vez de mostrar um progresso congelado que parece atividade.

**Cancelar mantém o que já foi indexado.** Um índice parcial responde perguntas sobre a parte
que tem; jogar fora obrigaria a recomeçar do zero. E retomar é barato: um novo trabalho pula
os arquivos cujo digest não mudou, comparação que `refresh` já implementa.

**Um trabalho por vez.** Dois processos indexando o mesmo repo se sobrescreveriam sem ganho —
os pontos têm id determinístico, então não corrompem, mas o trabalho seria duplicado. A fila é
serial, e é o suficiente para a escala real.

**Se o daemon morrer no meio**, o `jobs/<repo>.json` fica com estado `running` e um pid que não
existe mais. O comando seguinte detecta isso pelo mesmo teste de `(pid, starttime)`, marca o
trabalho como interrompido e oferece retomar. Um estado que mente é pior que um estado ausente.

## E — vigilância

O daemon percorre cada repo indexado a cada `WATCH_INTERVAL_S` (padrão 5 s):

1. `git ls-files` — a mesma fonte da indexação, então o que é vigiado é exatamente o que foi
   indexado;
2. `stat` em cada arquivo, comparando `mtime` e tamanho com o que está no acervo;
3. mudou → entra na fila de reindexação.

Medido: 16 ms para 2.000 arquivos. Um repo grande custa milissegundos por ciclo, e o resto do
tempo o processo dorme.

**Debounce de um ciclo.** Um arquivo que acabou de mudar só é reindexado quando estiver estável
por um ciclo — sem isso, salvar durante a edição dispara reindexação a cada save.

**Isso pega edição não commitada**, que é o que os hooks de git não pegavam. Arquivo novo ainda
não adicionado ao git não aparece, porque `git ls-files` não o lista — coerente com a regra de
que só entra o que é versionado.

## O que sai

`core/githook.py`, o comando `repos install-hook` e `tests/test_githook.py` **são removidos**.
Foram construídos em 2026-08-18, algumas horas antes desta spec, para dar o gatilho de
atualização sem daemon. O daemon cobre tudo que eles cobriam e mais, e manter os dois seria
duas coisas indexando o mesmo arquivo. Decisão do usuário: *"pode tirar o processo redundante,
use um só o daemon"*.

`repos refresh` **fica**: é a operação que o daemon executa, e continua valendo à mão.

## Falhas, e o que o usuário vê

| falha | comportamento |
|---|---|
| Qdrant fora do ar | o trabalho falha, `status` mostra o erro, os arquivos ficam `[stale]` na busca |
| um arquivo ilegível | pulado e reportado; o lote continua (regra que `add_files` já segue) |
| daemon não sobe | o comando diz por quê e não finge que enfileirou |
| daemon morre no meio | trabalho marcado interrompido, retomável |
| host morre | daemon encerra no ciclo seguinte |
| dois daemons | impossível: criação exclusiva do `daemon.json` (`O_EXCL`) decide quem venceu |

**Nenhuma falha aqui abre para "não há nada".** É a mesma regra do sub-projeto A: busca que
falha e devolve vazio é indistinguível de ausência real, e o modelo concluiria ausência a partir
de infraestrutura fora do ar.

## Fora de escopo

- **Arquivo novo não versionado.** `git ls-files` não o lista; ele entra quando for adicionado
  ao git. Consistente com a regra de que só entra o que é versionado.
- **Vigiar repo que não é git.** A detecção, a lista de arquivos e o `.gitignore` vêm todos do
  git; um diretório qualquer precisaria de outra política de seleção.
- **Paralelizar a indexação.** Um trabalho por vez. O gargalo medido é a rede do embedding, não
  a CPU, e paralelismo aqui pede controle de taxa que ninguém pediu.
- **Reindexação por mudança de conteúdo do chunker.** Trocar `TARGET_CHARS` não reindexa nada
  automaticamente; isso é `repos refresh` à mão.

## Testes

- **Ciclo de vida:** daemon encerra quando o último lease morre; sobrevive enquanto um vive;
  pid reciclado não segura o daemon (`starttime` diferente).
- **Cancelamento:** o parcial permanece indexado; retomar pula o que não mudou.
- **Varredura:** `.gitignore` respeitado via `git ls-files`; cada um dos quatro descartes,
  isolado.
- **Vigilância:** arquivo alterado é reindexado; arquivo estável não; debounce segura um save
  em andamento.
- **Estado:** `status` com daemon morto diz que está morto, em vez de mostrar progresso parado.
- **Equivalência:** as mesmas operações nos dois hosts, exceto as que são do operador.
- **Nenhum teste sobe daemon de verdade contra o Qdrant real.** O executor de trabalho é
  injetável, como `probe` em `refresh_window`.
