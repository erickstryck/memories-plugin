# memories-plugin

Memória semântica de longo prazo e índice de documentos sobre [Qdrant](https://qdrant.tech),
para agentes. Núcleo em Python puro (só stdlib), com adaptadores finos por host.

O problema que resolve é duplo:

- **Contexto que se perde.** Decisão tomada, armadilha já paga, comportamento já
  medido — tudo isso evapora no fim da sessão e é redescoberto do zero na
  seguinte, às vezes contradizendo o que já se sabia.
- **Contexto que não cabe.** Responder uma pergunta sobre 40 linhas de um arquivo
  de 8.000 não deveria custar o arquivo inteiro. Aqui o arquivo é lido, fatiado e
  indexado **fora** do contexto do agente; a busca devolve só os trechos que
  respondem.

## Três acervos, três ciclos de vida

A separação é estrutural — coleções distintas, configuráveis —, não convenção:

| acervo | o que guarda | expira |
|---|---|---|
| **memória** | fato atômico curado (decisão, preferência, comportamento medido) | não |
| **biblioteca** | documento inteiro guardado para consulta | não |
| **temporário** | documento aberto para uma tarefa | sim, TTL |

Por que não juntar tudo numa coleção: um documento longo vira dezenas de trechos
verbosos. Misturados aos fatos curados, eles vencem por volume em toda busca e
afundam justamente o acervo que mais importa. E o temporário é destrutível por
construção (existe comando que apaga a coleção inteira), então acervo permanente
não pode morar lá. A configuração recusa apontar dois papéis para a mesma coleção.

## Como a busca funciona

Dois estágios, com papéis distintos:

1. **Denso** (`bge-m3` ou outro embedder): varre o acervo inteiro por similaridade
   de vetor. Barato, aproximado, e **praticamente indiferente ao idioma** — uma
   pergunta em português encontra documento em inglês (medido: 0.460 contra 0.475
   para a mesma pergunta nas duas línguas).
2. **Cross-encoder** (`bge-reranker-v2-m3` ou outro): lê pergunta e trecho na
   MESMA passada, com atenção cruzada. Julga muito melhor, e por isso não tem
   vetor pré-computável — é um forward por par, custo linear no total de tokens.

Duas descobertas medidas moldaram o desenho, e ambas são falhas silenciosas se
ignoradas:

**A escala do re-rank depende do servidor.** O mesmo modelo devolve sigmoid (0..1)
num servidor e logit cru noutro — o mesmo documento irrelevante deu `1.6e-05` e
`-11.04`, sendo o segundo exatamente `logit(1.6e-05)`. Um corte calibrado numa
escala é inócuo na outra. O núcleo detecta pela faixa e normaliza, então o número
calibrado continua válido em qualquer servidor.

**O cross-encoder colapsa em par cross-lingual.** A mesma pergunta sobre o mesmo
documento em inglês: `0.2073` em inglês, `0.0004` em português — 500x. Ele casa
idioma, não só semântica. Consequências no desenho:

- Na busca de **documento**, o re-rank **ordena mas não veta**: quem pergunta já
  escolheu o documento, e silêncio é pior que ordem imperfeita. Colapso é
  detectado (melhor score abaixo de `0.01`) e a ordem densa assume, com aviso.
- Na busca de **memória**, o re-rank mantém o veto: ali precisão importa mais que
  alcance, e um falso positivo polui o contexto do agente.

## Instalação

```bash
git clone git@github.com:erickstryck/memories-plugin.git
cd memories-plugin
python3 -m unittest discover -s tests    # 137 testes, sem rede, sem dependência
ln -s "$PWD/bin/qctx" ~/.local/bin/qctx  # para `qctx` funcionar de qualquer lugar
```

Não há `pip install`: o núcleo usa só a biblioteca padrão. Isso é deliberado —
este código roda dentro de hooks disparados a cada interação, e uma dependência
faltando transformaria falha de ambiente em perda silenciosa de funcionalidade.

### Como plugin de Claude Code

O repositório é ao mesmo tempo um plugin e um marketplace de um só plugin:

```bash
claude plugin marketplace add ~/dev/memories-plugin
claude plugin install memories-plugin@memories-plugin
```

Habilitar o plugin registra **dois hooks** de `UserPromptSubmit` (recall a cada
prompt, checkpoint a cada N) e **duas skills** (`memory`, `doc-index`). Os hooks
usam `${CLAUDE_PLUGIN_ROOT}`, então não há caminho fixo para manter.

**Se você já tinha hooks equivalentes registrados à mão no `settings.json`,
remova-os na MESMA passada.** Os dois conjuntos disparam juntos e o recall é
injetado em dobro no mesmo prompt — que é pior que não ter nenhum, porque duplica
o custo de contexto sem acrescentar informação. Confira pelo log: um round por
prompt, não dois.

## Configuração

Comece pelo diagnóstico guiado:

```bash
python3 cli/qctx.py setup
```

Ele verifica Qdrant, endpoint de embedding (detectando a dimensão real do modelo),
endpoint de re-rank (inclusive a escala em que ele responde) e as três coleções —
e imprime, para cada item que falta, o comando exato que resolve. Num terminal
interativo ele pergunta e grava; **sem TTY ele nunca bloqueia**, apenas relata.
Isso é deliberado: o comando também é chamado por agente e por script, e um prompt
esperando resposta que nunca vem penduraria a chamada.

`--check` força o modo somente-diagnóstico; `--json` devolve o retrato completo
para consumo por programa.

Precedência: **variável de ambiente > arquivo > default**. O arquivo vive em
`~/.config/memories-plugin/config.json`.

```bash
python3 cli/qctx.py collections list           # o que existe no Qdrant, com dimensão
python3 cli/qctx.py config set memory-collection minhas_memorias
python3 cli/qctx.py config show
```

`collections list` marca cada coleção como compatível ou não com a dimensão do
modelo configurado. Gravar num acervo de outra dimensão é recusado: passaria e
degradaria a busca sem nenhum erro aparecer.

Variáveis reconhecidas (canônica primeiro, aliases legados aceitos):

| config | ambiente |
|---|---|
| `qdrant_url` | `QCTX_QDRANT_URL`, `QDRANT_URL` |
| `qdrant_api_key` | `QCTX_QDRANT_API_KEY`, `QDRANT_SERVICE_API_KEY` |
| `api_base_url` | `QCTX_API_BASE_URL`, `SERVER_BASE_URL` |
| `api_key` | `QCTX_API_KEY`, `SERVER_API_KEY` |
| `embed_url` | `QCTX_EMBED_URL`, `RECALL_EMBED_URL` |
| `rerank_url` | `QCTX_RERANK_URL`, `RECALL_RERANK_URL` |
| `embed_model` | `QCTX_EMBED_MODEL`, `EMBEDDING_MODEL` |
| `rerank_model` | `QCTX_RERANK_MODEL`, `RECALL_RERANK_MODEL` |
| `memory_collection` | `QCTX_MEMORY_COLLECTION`, `COLLECTION_NAME` |
| `docs_collection` | `QCTX_DOCS_COLLECTION` |
| `library_collection` | `QCTX_LIBRARY_COLLECTION` |

`memory_collection` nasce **vazia** de propósito: sem escolha explícita o CLI se
recusa a operar, para que não exista caminho acidental de escrita num acervo
errado.

## Uso

```bash
qctx() { python3 cli/qctx.py "$@"; }

# memória
qctx memory store "o poll do conector X trunca em 100 itens" --type reference
qctx memory find "paginação do poll"          # denso, barato
qctx memory recall "paginação do poll"        # dois estágios, com re-rank
qctx memory update <id> --text "..." ; qctx memory delete <id>

# documentos
qctx docs index ./relatorio-gigante.md --ttl 24h    # temporário
qctx docs keep ./manual-da-api.md                   # biblioteca, permanente
qctx docs search "como autenticar?" --scope all --limit 5
qctx docs list
qctx docs refresh --scope library                   # reindexa o que mudou no disco
qctx docs drop <doc-id> --scope library
qctx docs drop --purge-tmp                          # apaga só o temporário
```

Para arquivo de texto, a busca devolve **`caminho:linhas`** e um trecho curto, em
vez do conteúdo inteiro: quem consome relê a região exata e trabalha sobre o
conteúdo **atual**, sem risco de operar sobre uma foto vencida. Para origem não
relegível por região, devolve o texto com a data da indexação e um aviso. Em todos
os casos, se o arquivo mudou desde a indexação, o resultado vem marcado.

## Estrutura

```
core/       núcleo portável — nenhuma referência a host ou agente
  config.py   precedência de configuração, guardas de coleção
  qdrant.py   cliente mínimo (stdlib)
  models.py   embedder e cross-encoder, com normalização de escala
  chunk.py    fatiamento por fronteira estrutural
  memory.py   CRUD de memórias + recall de dois estágios
  docs.py     índice de documentos, TTL, obsolescência
cli/        interface de linha de comando sobre o núcleo
tests/      137 testes offline + 17 de integração
```

## Desenho

O núcleo depende de CONTRATOS (`core/ports.py`, `typing.Protocol`) e não de
implementações: `VectorStore`, `EmbeddingModel`, `RerankModel`. Trocar Qdrant por
outro banco vetorial, ou o endpoint de embedding por uma biblioteca local, é
escrever um adaptador — nenhum arquivo de regra muda.

São `Protocol` e não classe base abstrata de propósito: tipagem estrutural, sem
herança e sem custo em tempo de execução. O ganho concreto é teste — o pipeline de
recuperação, que é a lógica mais delicada do pacote, roda com dublês em
milissegundos. Antes só dava para exercitá-lo com infra real, o que significa que
ninguém o rodava enquanto editava.

O pipeline de dois estágios vive em UM lugar (`core/retrieval.py`) e as diferenças
entre consumidores são POLÍTICA, não código duplicado:

| | memória | documentos |
|---|---|---|
| o re-rank pode eliminar? | sim, veta | não, só ordena |
| a ordem importa? | não — tudo é injetado junto | sim — é uma lista lida de cima para baixo |
| por quê | falso positivo polui o contexto do agente | quem pergunta já escolheu o documento; silêncio é pior que ordem imperfeita |

Antes existiam três implementações da mesma ideia, e isso já custou: a normalização
de escala do re-rank existiu num consumidor e não no outro.

## Portabilidade

`core/` não conhece o chamador. Um host novo é um adaptador fino:

- **Como biblioteca:** `import core` e monte com `build_memory(cfg)` /
  `build_docs(cfg)`.
- **Como processo:** chame `cli/qctx.py` e leia o JSON de `--json`.

O fatiamento acontece **dentro** deste processo, então o documento nunca passa
pelo contexto de quem pergunta — é isso que torna viável indexar um arquivo de
30 mil caracteres para responder com três trechos.

## Os hooks

`recall.py` roda a cada prompt do usuário, ANTES de o modelo ver o texto: monta
três ângulos da pergunta numa única chamada de embeddings, funde os resultados por
id pelo maior score, aplica os dois portões e injeta os documentos com as regras de
uso. Memória já injetada há pouco volta como ponteiro de uma linha, e a vaga
liberada revela mais do acervo.

Falha em silêncio para o **usuário** e nunca para o **modelo**: se a busca não
roda, o prompt segue normalmente, mas o bloco injetado diz explicitamente que o
acervo não foi consultado. Sem esse aviso, ausência de resultado é indistinguível
de "não há precedente", e é assim que se afirma que algo é inédito sem ninguém ter
olhado.

`checkpoint.py` injeta o procedimento completo de gravação a cada N interações. O
texto é autossuficiente de propósito: lembrete de uma linha produz memória vaga,
duplicada e sem metadata, e o custo aparece meses depois.

## Estado

Pronto e testado: núcleo, CLI, os três acervos, diagnóstico guiado, os dois hooks,
as duas skills e o manifesto de plugin. 137 testes offline e 17 de integração
contra Qdrant e modelos reais.
