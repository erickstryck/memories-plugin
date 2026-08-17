# Sub-coleções por repositório — desenho

Sub-projeto **A** de cinco. O pedido original do usuário (indexar um repositório inteiro,
em segundo plano, cancelável, com watcher, status visível e sub-coleções por repo) são
**sete subsistemas**, e foi decomposto em 2026-08-17 antes de qualquer desenho. Esta spec é
só o primeiro: **o modelo de dados e seu contrato**. Os outros quatro estão em "Fora de
escopo" com o motivo de cada um.

## O problema

O acervo de documentos de hoje indexa **um arquivo por vez**, com TTL, para responder uma
pergunta sobre um arquivo grande sem pagar o arquivo inteiro em contexto. Não há como
perguntar *"em qual dos meus projetos isto aparece"*, e não há acervo cujo ciclo de vida
seja "permanente até eu apagar à mão".

Nas palavras do usuário, o que a divisão por repo compra: *"o modelo pode ter uma
observabilidade entre projetos, podendo pesquisar coisas chaves antes de varrer tudo para
encontrar algo específico"*.

## O que se decidiu

| Decisão | Escolha | Alternativas recusadas |
|---|---|---|
| Modelo físico | **Uma coleção + campo `repo` indexado + registro à parte** | uma coleção do Qdrant por repo |
| Identidade do repo | **Declarada na indexação** (novo, ou junta-se a um existente) | só remote; só caminho; remote com caminho como atributo |
| Escopo padrão da busca | **O repo do diretório atual**, o resto a um passo | todos por padrão; dois estágios; sempre explícito |
| Onde mora | **Acervo próprio**, permanente, fora do `scope=all` | dentro do `library` |
| Deleção | **Manual**, chunks antes do registro | drop de coleção; deleção automática |

### Por que o modelo físico não é uma coleção por repo

O Qdrant **não tem coleção aninhada**: o desenho que o usuário descreveu
(`repository → repoA, repoB`) não existe como hierarquia física, e existe como filtro.

Uma coleção por repo daria isolamento real e deleção instantânea por drop. Perde em duas
frentes, e a primeira é medida nesta mesma base de código: busca entre projetos custaria
**N consultas sequenciais**, uma por repo — que é exatamente o defeito que a guarda de
arquivo grande pagou em 2026-08-16, onde seis idas ao Qdrant, cada uma dentro do seu
timeout, somaram 7,2 s contra um limite de hook de 5 s. Ali era o caminho raro; aqui seria
o caminho quente. A segunda: N grafos HNSW pequenos recuperam pior que um grande quando a
pergunta é "onde, em qualquer projeto, isto aparece" — otimizaria contra o caso de uso que
justifica a feature.

## Viabilidade — medido antes do desenho

**A porta de vetores NÃO tem `facet`, `distinct` nem `count`** (`core/ports.py`, lido em
2026-08-17: `ensure_collection`, `ensure_payload_index`, `collection_info`,
`list_collections`, `delete_collection`, `upsert`, `search`, `get_point`, `set_payload`,
`delete_points`, `delete_by_filter`, `scroll`, `scroll_all`). Logo responder *"quais repos
eu tenho"* varrendo os chunks seria `scroll_all` sobre TODOS os pontos — dezenas de milhares
de chunks lidos para uma pergunta de metadado. **É isto que torna o registro obrigatório, e
não uma conveniência.**

**UMA porta nova é necessária, e ela vem de uma pergunta do usuário** (2026-08-17): *"se eu
quiser verificar se algum repo menciona sobre o assunto x, ele consegue fazer essa
travessia?"*. Filtro por keyword já é usado (`{"key": "doc_id", "match": {"value": ...}}` em
`core/docs.py:343`), `search` aceita filtro, `scroll_all` aceita filtro, e
`ensure_payload_index(name, field, schema)` existe — mas nada disso responde àquela pergunta,
pelo motivo da seção seguinte. Falta `search_groups`.

**`group_by` do Qdrant existe e foi MEDIDO neste deployment** (versão 1.18.2, medida em
`GET /`): `POST /collections/<c>/points/search/groups` com `group_by`, `limit` (número de
GRUPOS) e `group_size` (hits por grupo) devolveu 4 grupos distintos numa consulta contra
`memories_docs_tmp`, agrupando por `doc_id`. Capacidade confirmada por efeito, não por número
de versão.

**A instância já é povoada:** ~85 coleções e ~2 milhões de pontos (medido). Acrescentar duas
coleções é irrelevante; acrescentar uma por repositório entraria numa instância já cheia.

**Metade da detecção de desatualizado já existe.** `core/docs.py::source_changed` compara
por **digest** e não por metadado, e o docstring registra por quê: `cp -p`, `rsync --times`
e restore de backup preservam mtime, e uma edição de um caractere preserva o tamanho. A
tolerância de float foi calibrada por medição (1786646270.9956777 gravado contra
...75 em disco). O watcher do sub-projeto E precisa do **gatilho**, não da detecção.

**A semântica de permanência já existe.** O acervo `library` é sem TTL, nunca varrido e
fora do alcance de qualquer comando de limpeza. Repos herdam esse ciclo de vida — em
coleção própria, pelo motivo da seção seguinte.

**Escala.** Este repositório: 59 arquivos rastreados, ~930 KB de conteúdo, ~465 chunks de
2000. Um monorepo de 10 mil arquivos vira dezenas de milhares de chamadas de embedding — o
que é problema do sub-projeto B, e é por isso que ele é separado.

**Identidade não pode ser só o remote, e isso é medido no disco do usuário.** Ele mantém
clones paralelos do MESMO remote de propósito (`awesome-cv3` e `temp-awesome-cv3`; a
memória do trabalho de conectores registra clones `temp`/`temp-2` para fluxos paralelos), e
worktrees em caminhos distintos. Identidade por remote fundiria checkouts que podem estar
em branches diferentes.

## Arquitetura

```
core/repos.py    RepoIndex: o acervo, o registro, e as operações sobre eles
core/docs.py     REUSADO, não copiado: chunking, content_digest, source_changed, doc_id_for
core/ports.py    ganha `search_groups` — a travessia "quais repos mencionam x"
cli/qctx.py      verbos `repos` — invólucros finos sobre RepoIndex
```

**A escrita em A recebe a lista de arquivos; não a descobre.** `RepoIndex.add_files(repo,
paths)` indexa exatamente os caminhos que lhe são dados, e há um verbo de CLI que o expõe.
Quem DECIDE quais arquivos entram — percorrer o repo, respeitar `.gitignore`, pular
minificado e binário — é o sub-projeto B. Essa fronteira é o que torna A testável ponta a
ponta com meia dúzia de arquivos, sem depender do pipeline em massa.

`RepoIndex` é módulo novo e **não** uma extensão do `DocIndex`. O `DocIndex` já é dono de
dois acervos com semântica de TTL; repos têm semântica diferente (permanente, agrupado, com
registro). Crescer o `DocIndex` seria o sinal de arquivo fazendo coisa demais.

### O acervo

Coleção `memories_repos`, campo de config `repos_collection`
(`QCTX_REPOS_COLLECTION`). Cada ponto é um chunk com `repo` no payload, e há índice de
payload de keyword em `repo`.

**Por que não dentro do `library`.** O argumento é do próprio `core/docs.py`: *"a file chunk
competing with a curated fact in a recall search wins on volume and drowns the archive that
matters most"* — foi por isso que documentos não moram na coleção de memória. Um nível
abaixo é o mesmo caso: `library` é documento escolhido a dedo, e indexação de repo são
dezenas de milhares de chunks automáticos.

**`scope=all` NÃO passa a incluir repos.** Se incluísse, todo `qctx docs search --scope all`
que já funciona seria inundado — regressão de comportamento já entregue. Repos exigem
escopo explícito.

### O registro

Coleção `memories_repos_registry` (`QCTX_REPOS_REGISTRY_COLLECTION`), um item por repo:
id estável, rótulo da listagem, remotes normalizados já vistos, checkouts conhecidos,
contagem de arquivos e chunks, e quando foi indexado.

**Por que coleção separada e não uma flag `kind` na mesma.** Se o registro morasse junto dos
chunks, TODA busca precisaria filtrá-lo para sempre, e um filtro esquecido uma vez faz linha
de registro aparecer como resultado. A review de branco de 2026-08-17 mediu duas guardas
desta feature que nenhum teste segurava — guarda que só existe por vigilância acaba deletada
por limpeza que passa no CI. Coleção separada torna o problema inexistente em vez de
vigiado.

Custo aceito: é uma quinta coleção, e é um key-value que por acaso mora no Qdrant — nunca
buscada por vetor, só percorrida. O invariante do `core/config.py` que garante coleções
distintas passa a valer para cinco.

### O status é uma junção de três fontes, não um campo

| Estado | Onde existe de verdade | Vida |
|---|---|---|
| **indexado** | no acervo — fato sobre o Qdrant | durável, compartilhado |
| **indexando** | nesta máquina — um job rodando | efêmero, local |
| **desativado** | preferência para ESTE checkout | durável, local |

Forçá-los num store só seria errado. Consequência boa: o registro nunca mente "indexando"
depois de a máquina ser reiniciada no meio, porque não é dono dessa informação. A superfície
que junta os três é do sub-projeto B; A entrega apenas a parte "indexado".

### Identidade e vínculo

Na indexação: detecta a raiz do git, lê os remotes, procura no registro por remote
normalizado. Casou, oferece **juntar-se a `<rótulo>`** como default; não casou, oferece repo
novo com o nome do diretório. `RepoIndex.candidates_for(path)` devolve essa escolha, e quem
a APRESENTA é o sub-projeto C — A não tem superfície interativa.

**O id nasce do nome escolhido, e a colisão é erro e não fusão.** Para um repo novo, o id é
um slug derivado do nome (default: o nome do diretório) e tem de ser único no registro; se
já existir, a escolha é recusada com o conflito nomeado, para o usuário decidir entre outro
nome e juntar-se ao existente. Fundir por colisão de slug seria decidir identidade por
acidente de nomenclatura, que é justamente o que a decisão de identidade declarada rejeita.
O id é a chave de filtro e nunca muda; o rótulo é texto de listagem e pode ser reescrito.

O vínculo *caminho absoluto → repo* fica no estado local compartilhado entre os dois hosts
(`~/.memories-plugin/state/`, com `QCTX_STATE_DIR` sobrescrevendo), para a pergunta não se
repetir.

Duas consequências, e são onde isto normalmente apodrece:

- **Mover um checkout** perde o vínculo (a chave é o caminho), a detecção pergunta de novo,
  e o casamento por remote faz "juntar" ser o default — então CICATRIZA em vez de orfanar.
- **Deletar um repo pela listagem invalida os vínculos locais.** Sem isso um checkout
  continua afirmando pertencer a um repo que não existe, e a próxima indexação escreve em
  repo fantasma.

### O contrato de busca

`qctx repos search <query>` busca no repo do diretório atual, resolvido por raiz do git →
vínculo local → id. `--repo <id>` aponta outro; `--all` atravessa todos.

Fora de um repo indexado, o comando **falha dizendo o remédio** (nomeie um repo ou use
`--all`) em vez de buscar em tudo: buscar tudo em silêncio é o ruído que a decisão de escopo
recusou.

**`--all` agrupa NO SERVIDOR, e a versão anterior desta spec estava errada aqui.** Ela dizia
que bastava agrupar os resultados no cliente, porque cada hit carrega o `repo`. Isso responde
"quais repos estão no top-K", que não é a pergunta: busca vetorial devolve os K melhores por
similaridade, então um repo com cinquenta chunks parecidos ocupa os dez primeiros lugares e um
repo com UMA menção real desaparece. Para *"algum repo fala de x?"* é a pergunta errada
respondida com confiança.

O correto é `search_groups` com `group_by: "repo"`, `limit` = quantos repos o registro conhece,
`group_size` pequeno (3). Os grupos se formam por VALOR DISTINTO e não por posição no ranking
global, então o repo de menção única aparece como grupo próprio. Uma consulta só — melhor que
os dois estágios e melhor que uma coleção por repo, que precisaria de N.

**Sinergia com o registro:** é ele que diz quantos repos existem, logo quantos grupos pedir.
Sem registro, esse número seria um chute.

**O limite honesto:** o agrupamento é sobre o que a busca alcança, não uma varredura
exaustiva. Um repo cujo melhor chunk fique abaixo do horizonte de score pode não aparecer.
Então a saída de `--all` nunca afirma ausência: ela diz "nada acima do corte", que é a mesma
distinção que o hook de recall já faz entre "consultado, nada relevante" e "indisponível".
Afirmar "nenhum repo menciona x" a partir de um top-K é a versão desta feature do erro que
aquele hook existe para não cometer.

Herda do `docs search` o retorno da **localização** em vez de um retrato do conteúdo, para o
consumidor ler o arquivo atual — o que também faz um chunk velho degradar para "arquivo
mudou" em vez de mentir.

### Deleção

`qctx repos list` para ver, `qctx repos drop <id>` para apagar, com confirmação. Apaga três
coisas: os chunks, o item do registro, e os vínculos locais.

**A ordem é chunks primeiro, registro por último, e não é arbitrária.** Registro primeiro
com falha nos chunks deixa chunks invisíveis à listagem que ainda competem em `--all` —
lixo inalcançável. Chunks primeiro com falha no registro deixa um item apontando para zero
chunks: **visível, e uma segunda execução termina o serviço.** É o mesmo princípio da
releitura pós-escrita que o script de cutover defende.

`drop` não pode alcançar `tmp`, `library` nem a memória.

## Modos de falha

**Aqui falha NÃO abre — e é o oposto da guarda de arquivo grande, de propósito.** Lá,
bloquear com número duvidoso impedia o usuário de ler arquivos, então falhar tinha de
liberar. Aqui o risco é o inverso: uma busca que falha e devolve **vazio** é indistinguível
de "não há nada", e o modelo concluiria ausência a partir de infraestrutura fora do ar. É a
mesma distinção que o hook de recall já faz entre "consultado, nada acima do corte" e
"indisponível".

| Falha | Destino |
|---|---|
| Qdrant inalcançável numa busca | ERRO explícito, nunca resultado vazio |
| registro tem repo, acervo não tem chunk | listagem marca a divergência; `drop` termina o serviço |
| acervo tem chunks, registro não tem item | **detectável por teste** (ver abaixo); repo inalcançável pela listagem |
| vínculo local aponta para repo apagado | vínculo invalidado no `drop`; se sobrar, a detecção trata como não vinculado |
| checkout movido | vínculo perdido, cicatriza por casamento de remote |
| diretório atual não está em repo indexado | erro com o remédio nomeado |
| repo sem remote nenhum | permitido: identidade é declarada, e o default vira o nome do diretório |

## Testes

- **`RepoIndex` sobre fakes**, sem rede: agrupamento por `repo`, filtro escopado, busca
  ampla, listagem pelo registro, deleção na ordem, e que `drop` não alcança os outros três
  acervos.
- **Divergência registro × chunks** — teste dedicado, e ele é o preço de honestidade do
  desenho: o registro é uma SEGUNDA fonte de verdade sobre quais repos existem, ao lado do
  campo `repo` nos chunks. É a mesma classe da ruling F5 desta base (um dono por
  invariante), e desta vez é o desenho que a introduz. A defesa é declarar dono com precisão — o
  registro é autoritativo sobre QUAIS REPOS EXISTEM, seus rótulos e seus vínculos; os chunks
  são autoritativos sobre CONTEÚDO, e o campo `repo` neles é derivado do registro, nunca a
  fonte dele — e testar a divergência. Sem esse teste é só uma cópia esperando divergir.
- **Equivalência entre hosts**, no padrão de `tests/test_host_equivalence.py`: as mesmas
  operações existem e decidem igual nos dois hosts.
- **`scope=all` não alcança repos** — teste, porque é regressão de funcionalidade entregue.
- **O repo de menção única não é ofuscado** — teste dedicado, com fixture em que um repo tem
  muitos chunks fortes e outro tem UM fraco: o agrupamento tem de devolver os dois. É a
  regressão exata que a versão anterior desta spec teria embarcado, e um teste que use fixture
  equilibrada não a pegaria.
- **Sonda de mutação em toda guarda**: removê-la tem de deixar a suíte vermelha, com a
  contagem de ocorrências verificada antes de substituir e o ESCOPO de cada contagem
  declarado (módulo isolado ou suíte inteira). As duas regras nasceram de defeitos reais
  medidos em 2026-08-16/17.

## SOLID

- **Responsabilidade única** — `RepoIndex` é dono do acervo de repos e do registro; o
  `DocIndex` continua dono dos dois acervos com TTL; o CLI só traduz argumentos.
- **Aberto/fechado** — o campo `repo` e o índice de payload permitem novos eixos de filtro
  (branch, linguagem) sem tocar quem já consome.
- **Substituição** — `RepoIndex` fala com as mesmas portas; qualquer store que as satisfaça
  serve, e é isso que os testes com fakes exploram.
- **Segregação de interface** — um método novo em `VectorStore` (`search_groups`), e ele é
  exigido porque a travessia entre repos não se expressa com os que existem. O fake dos testes
  o implementa, e é o mesmo contrato que o Qdrant já oferece — não estamos inventando
  capacidade, estamos expondo uma que o servidor tem e a porta escondia.
- **Inversão de dependência** — depende de `ports.VectorStore` e `ports.EmbeddingModel`,
  nunca do cliente HTTP.
- **Reusar, não copiar** — chunking, `content_digest`, `source_changed` e `doc_id_for` são
  CHAMADOS. Esta base pagou quatro rodadas para aprender que cópia idêntica é a que diverge
  em silêncio.

## Fora de escopo, deliberadamente

- **Quais arquivos entram** (respeitar `.gitignore`, pular lockfile, minificado, binário) e
  **percorrer o repo inteiro** — sub-projeto **B**, junto da superfície de status que junta
  as três fontes. É onde a escala vira problema.
- **Detecção de git e o consentimento** nos dois hosts — sub-projeto **C**. A entrega
  `candidates_for(path)`; C apresenta.
- **Execução em segundo plano e cancelamento** — sub-projeto **D**. Hoje o plugin não tem
  nenhum processo longo (medido: nenhum `Popen`, nenhuma thread fora do worker em-processo da
  guarda), e o `core/docs.py` declara "no daemon" como decisão. Introduzir processo de fundo
  é a decisão mais cara do conjunto e merece spec própria.
- **Watcher** — sub-projeto **E**, e depende do D. A detecção já existe
  (`source_changed`); falta o gatilho, e um watcher pede processo longo e normalmente
  dependência externa, que este projeto recusa por decisão (só stdlib, porque dependência
  faltando dentro de um hook vira perda silenciosa de funcionalidade).
- **Reindexação automática** de qualquer natureza. Em A, escrever no acervo é sempre pedido.
