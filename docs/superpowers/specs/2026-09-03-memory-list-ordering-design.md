# Listagem de memória ordenada por recência: design

**Data:** 2026-09-03 · **Repo:** `memories-plugin` @ `190e682` · **Estado:** aprovado em
brainstorming, pronto para plano de implementação

## O problema, medido

A ferramenta `memory_list` promete recência e entrega ordem de id.

A descrição que o modelo lê, em `hosts/hermes/tools.py:556`, diz:

> "Page through the archive without a query, **newest page first**."

O caminho do código não ordena por nada. Medido em 2026-09-03 lendo a árvore:

| passo | arquivo | o que faz |
|---|---|---|
| `_memory_list` | `hosts/hermes/tools.py:312` | chama `list_page(limit)` |
| `MemoryStore.list_page` | `core/memory.py:374` | chama `self.q.scroll(collection, limit, offset)` |
| `Qdrant.scroll` | `core/qdrant.py:168` | POST em `/points/scroll` com `limit`, `with_vector`, `with_payload`, `offset` e `filter` |

`grep -rn "order_by" --include=*.py .` no repositório inteiro: **zero ocorrências**. O scroll
do Qdrant sem `order_by` percorre por id, e os ids são `uuid4` (`core/memory.py:168`), então a
ordem é arbitrária e estável.

**Sintoma observado**, e é o que motivou o pedido: `memory_list(limit=4)` numa sessão que
acabara de gravar 3 registros devolveu quatro registros de agosto (`000300fe`, `00295a2c`,
`00bf3c21`, `0137aa57`) e nenhum daquele dia. Quem usa a listagem para conferir se a gravação
entrou conclui "não gravou" sobre um registro que está lá.

São duas superfícies, e as duas herdam o defeito porque as duas passam por `list_page`:
`memory_list` no hermes (`hosts/hermes/tools.py:312`) e `qctx memory list` na CLI
(`cli/qctx.py:1089`). A segunda é também o caminho do claude-code, que chega à listagem pela
CLI e não por uma tool própria (`skills/memory/SKILL.md:162`).

## O que se mediu antes de desenhar

Contra o Qdrant real do usuário (**1.18.2**), em coleções descartáveis `zz_probe_orderby` e
`zz_probe_orderby2`, criadas e apagadas no mesmo script. Cada linha abaixo é efeito
observado, não leitura de documentação.

| # | pergunta | resposta medida |
|---|---|---|
| 1 | `order_by` sem índice? | **400**: `No range index for order_by key: updated_at. Please create one to use order_by` |
| 2 | índice resolve? | `PUT /collections/<c>/index {"field_name":"updated_at","field_schema":"datetime"}` devolve `completed`, e o scroll ordenado passa a funcionar |
| 3 | é retroativo? | **Sim.** Pontos gravados ANTES do índice saíram ordenados. Nenhuma migração, nenhuma reindexação |
| 4 | `order_by` + `offset`? | **400**: `Cannot use an offset when using order_by. The alternative for paging is to use order_by.start_from and a filter to exclude the IDs that you've already seen` |
| 5 | `next_page_offset` com `order_by`? | **Sempre `null`**, mesmo havendo páginas seguintes. O cursor do servidor deixa de existir |
| 6 | paginação por valor funciona com empate? | **Sim.** 4 pontos com o mesmo timestamp: página 1 deu `newest, tie-1, tie-3`; página 2 com `start_from` no valor empatado mais `must_not.has_id` dos vistos deu `tie-2, tie-4, oldest`. Sem pular, sem repetir |
| 7 | registro SEM o campo? | **Some da listagem ordenada.** Não aparece nem no fim |
| 8 | dá para contar os sem campo sem varrer? | **Sim.** `POST /points/count` com `filter.must.is_empty.key = updated_at` |

E o estado do acervo real, por leitura pura (nada escrito) da coleção `claude_memory`:

| medida | valor |
|---|---|
| pontos | **744** |
| sem `updated_at` | **0** |
| sem `created_at` | **0** |
| índices de payload hoje | **nenhum** (`payload_schema` vazio) |
| faixa de `updated_at` | 2026-07-23T13:12 a 2026-09-03T19:41 |

Três consequências que fecham decisões:

1. Como **nenhum** registro carece de `updated_at`, ligar a ordenação não esconde nada no
   acervo do usuário. O caso do registro sem data é possibilidade teórica (payload legado do
   MCP antigo), não realidade atual, mas continua precisando de tratamento, porque o conjunto
   vazio de hoje não é garantia de amanhã.
2. A medição **1** derruba a conclusão da investigação anterior, que dizia "ordenar exige
   índice, logo o conserto barato é corrigir a descrição". Exigir é verdade; ser caro, não. O
   índice é uma chamada, e é retroativo.
3. As medições **4** e **5** significam que isto **não é** acrescentar um parâmetro: o modelo
   de paginação muda de cursor-do-servidor para cursor-do-cliente. É a razão de o trabalho ser
   architectural e não bounded.

## Decisão

Implementar a ordenação de verdade. Decisão do usuário em 2026-09-03: *"faça o 2, pois isso é
relevante e importante de se ter"*, recusando explicitamente a alternativa de só corrigir a
descrição.

| Decisão | Escolha | Recusado, e por quê |
|---|---|---|
| Chave de ordenação | `updated_at` | `created_at`: quem lista quer ver o que mudou por último, e uma correção é uma mudança |
| Onde nasce o índice | na **criação da coleção** (escrita) e, para coleção preexistente, **sob demanda na primeira listagem** | criar em toda escrita: medido, custa ~25 ms de `PUT /index?wait=true` por `store` e leva uma escrita de 45 ms para 74 ms, sem ganho, porque o índice ou já existe ou a listagem o cria |
| Paginação | cursor opaco de valor mais ids vistos, e o cursor **declara seu modo** | `offset` de id: o servidor recusa junto com `order_by` (medição 4) |
| Falha do caminho ordenado | degradar a **ordem** dizendo que degradou, **mantendo a paginação** | falhar a operação inteira; degradar em silêncio; ou degradar a paginação junto, que reduz o acervo a uma página |
| Quais falhas podem ser absorvidas | **somente 400**, em lista de permissão | lista de exclusão: engole todo 4xx imprevisto, inclusive 408 e 409, e vira página aparentemente boa |
| Ordem de `recall`/`find` | **intocada** | ordenar busca por similaridade por data destrói o propósito dela |
| Índice em `created_at` | **não criar** | YAGNI: nada lista por ele |

Sobre a linha do índice, porque a primeira versão deste documento se contradizia: `require_existing`
garante que uma leitura nunca **cria a coleção**, e isso continua valendo. Criar um índice de
payload numa coleção que já existe é outra coisa: é ajuste de esquema, não criação de acervo, e
sem ele o acervo de 744 registros do usuário (que não tinha índice nenhum) nunca ordenaria, já que
o índice só nasceria numa coleção nova. Então a leitura pode criar índice e não pode criar coleção.

## Arquitetura

Quatro responsabilidades, quatro lugares, seguindo a fronteira que o pacote já tem
(`core/ports.py` declara contratos; `core/qdrant.py` traduz HTTP e não decide nada;
`core/memory.py` tem a regra de negócio; `hosts/` e `cli/` são só protocolo de superfície).

```
hosts/hermes/tools.py ─┐
cli/qctx.py           ─┴─► MemoryStore.list_page ──► core.paging (regra do cursor, pura)
                                    │
                                    └──► ports.VectorStore.scroll ──► Qdrant.scroll ──► HTTP
```

### `core/paging.py`, NOVO, puro, sem rede

Dono único da regra de cursor por valor. Nada nele conhece Qdrant, HTTP ou memória; ele
recebe e devolve dados. É o que torna a regra testável sem servidor e sem fake.

Responsabilidades, e só estas:

- codificar e decodificar um cursor opaco carregando `{value, seen_ids}`;
- montar o `order_by` e o `filter` da próxima página a partir de um cursor;
- derivar o cursor seguinte a partir dos pontos que voltaram, inclusive a regra de empate:
  quando o último ponto da página divide o timestamp com outros, os ids já vistos **naquele
  valor** viajam no cursor para a página seguinte não os repetir (medição 6);
- decidir que não há página seguinte.

O cursor é uma string opaca para quem consome. O formato interno é detalhe de implementação
deste módulo. Nenhum outro arquivo pode montar ou ler um cursor por conta própria, e é isso
que impede a regra de empate de se espalhar.

### `core/qdrant.py`: `scroll` ganha `order_by`

Um parâmetro opcional, repassado ao corpo do POST. Zero regra, como o resto do adaptador.
Quando `order_by` é `None` o corpo sai idêntico ao de hoje, e nenhum chamador existente muda
de comportamento.

### `core/ports.py`: o Protocol acompanha

`VectorStore.scroll` ganha o mesmo parâmetro opcional. **Nenhuma porta nova**: a operação
continua sendo "percorrer a coleção", e um `VectorStore` alternativo que não saiba ordenar
levanta, que é o caminho de degradação já previsto abaixo.

### `core/memory.py`: orquestração e garantia do índice

`ensure()` (caminho de escrita) passa a garantir o índice `datetime` em `updated_at`, do jeito
que `core/docs.py:172-177` e `core/repos.py:121-126` já fazem com os índices deles. O padrão
existe no repositório e é seguido, não reinventado. `ensure_payload_index` já engole a própria
falha por desenho (`core/qdrant.py:107`), então a garantia nunca derruba uma escrita.

`list_page(limit, offset=None)` passa a:

1. pedir ao `paging` o `order_by` e o `filter` da página;
2. chamar `scroll`;
3. montar as memórias como hoje;
4. devolver, além de `count`, `memories` e `next_offset` (agora o cursor opaco), um campo novo
   **`order`** dizendo o que de fato aconteceu.

### Degradação honesta

É o ponto que mais importa, porque o defeito que estamos consertando é exatamente uma promessa
que o retorno não cumpria. A regra: **o retorno sempre diz o que aconteceu**.

| situação | o que `list_page` faz | `order` |
|---|---|---|
| ordenado funciona | devolve a página ordenada | `"updated_at_desc"` |
| falta o índice | cria o índice, tenta **uma** vez mais; se funcionar, segue | `"updated_at_desc"` |
| ordenado falha mesmo assim | cai para o scroll não ordenado, **que continua paginando** | `"unordered"` mais `warning` no corpo |

A decisão de tentar de novo é tomada por **status HTTP**, nunca por substring da mensagem de
erro. É a regra que `_is_absent` já documenta em `core/qdrant.py:19-34`, com o motivo medido:
um proxy ecoa o status de upstream no corpo e engana quem lê texto.

Absorver é permitido para **400 e só 400**, em lista de permissão e não de exclusão. Uma lista
de exclusão engoliria todo 4xx imprevisto, incluindo 408 (timeout) e 409, e transformaria "o
acervo não respondeu" numa página aparentemente bem-sucedida.

**Degradar a ORDEM não degrada a PAGINAÇÃO.** Um scroll não ordenado mantém o cursor do próprio
servidor, então a queda custa a ordem e nada mais. Forçar `next_offset: null` na degradação faria
um acervo de 749 registros responder como se terminasse na primeira página, enquanto a descrição
da ferramenta diz que `next_offset` nulo significa que não há mais nada. Por isso o cursor
**carrega o modo a que pertence**, e um cursor de um modo nunca é entregue ao outro: os dois
guardam coisas diferentes em `value` (um instante contra um id de ponto).

**O aviso cita o servidor e não diagnostica.** A primeira versão afirmava "o índice de payload
está faltando e não pôde ser criado" para todo 400, uma causa que nenhuma linha do caminho
verificou; filtro malformado recebia a mesma frase. Afirmar causa não verificada no campo criado
para impedir a ferramenta de afirmar coisas não verificadas seria o defeito original reencarnado.

### Empate é comparado como INSTANTE, não como texto

O índice `datetime` faz o servidor comparar instantes. `2026-09-01T00:00:00Z` e
`2026-09-01T00:00:00+00:00` são um instante e duas strings. Enquanto o empate foi detectado com
`==` sobre o texto cru, o gêmeo do registro de fronteira nunca era excluído: medido contra o
servidor real, um acervo de 6 registros devolveu 50 linhas, repetiu um deles 18 vezes, nunca
alcançou um sétimo e nunca terminou. O valor que volta ao servidor, porém, é o que ele
escreveu, não a normalização: posicionar `start_from` com um texto que o servidor nunca gravou
é outra forma de errar.

### Cursor é validado no VALOR, não só na forma

Um cursor sintaticamente válido carregando valor de outro acervo era aceito, ia como
`order_by.start_from`, tomava 400 do servidor ("Format error in JSON body") e era absorvido como
"não consigo ordenar", devolvendo a página 1 rotulada como página posterior, culpando um índice
que existia. É exatamente o modo de falha que a tabela de modos de falha proíbe por escrito, e
por isso `decode_cursor` exige que um cursor ordenado carregue um timestamp parseável.

### Superfície: o cursor precisa ser utilizável

Hoje `list_page` já devolve `next_offset` e **nenhuma** superfície aceita um `offset` de
volta, então o campo é decorativo (verificado: `grep -rn next_offset` só encontra as duas
linhas que o produzem em `core/memory.py`). Se ele passa a ser um cursor de verdade e continua
sem entrada, o usuário fica sem página 2.

- `memory_list` ganha o parâmetro `offset` (string de cursor);
- `qctx memory list` ganha `--offset`;
- a descrição da tool passa a descrever o comportamento real, incluindo o que acontece quando
  a ordenação não está disponível.

### Testes: o fake tem que ser tão pobre quanto o real

`FakeVectorStore.scroll` (`tests/fakes.py:112`) passa a honrar `order_by` **de verdade** e a
**recusar** quando não existe índice no campo, reproduzindo a medição 1. Sem isso o caminho de
fallback nunca é exercitado offline, e um fake mais generoso que a produção esconde exatamente
o bug que ele deveria pegar. O arquivo já defende esse princípio para `payload_fields` em
`tests/fakes.py:122-127`; aqui é o mesmo princípio.

Camadas de prova:

| camada | prova o quê | precisa de rede? |
|---|---|---|
| `tests/test_paging.py` (novo) | regra de cursor: empate, última página, cursor inválido | não |
| `tests/test_memory_offline.py` | `list_page` ordenado, `list_page` degradando, `ensure` criando o índice | não |
| `tests/test_hermes_tools.py` | `memory_list` aceita e devolve cursor | não |
| `tests/test_integration.py` | contra o Qdrant real: a primeira página é a mais recente | sim (já tem gate) |

## SOLID, aplicado onde muda decisão

Não como etiqueta. Cada linha abaixo mudou o desenho.

- **Responsabilidade única.** A regra de cursor não cabe em `memory.py`: ali ela ficaria
  misturada com embedding, validação de texto em branco e política de recall. Daí
  `core/paging.py` existir. E o adaptador Qdrant continua sem regra nenhuma.
- **Aberto/fechado.** `scroll` ganha um parâmetro opcional e nenhum chamador existente muda.
  `list_page` ganha um campo no retorno e quem lê `memories` continua lendo `memories`.
- **Substituição de Liskov.** O `FakeVectorStore` passa a falhar onde o Qdrant falha. Um fake
  que aceita o que a produção recusa não é substituto, é um ambiente diferente.
- **Segregação de interface.** Nenhuma porta nova. `order_by` entra na operação que já é
  "percorrer a coleção", em vez de criar uma `OrderedScroll` que só o Qdrant implementaria.
- **Inversão de dependência.** `list_page` continua falando com `ports.VectorStore`, e
  `paging` não conhece store nenhum. Por isso a regra mais delicada é testável sem rede.

## Fora de escopo

- Reordenar `recall` e `find`.
- Índice em `created_at`.
- Ordenar a listagem de documentos (`docs list`) ou de repositórios. Mesma técnica, outro
  acervo, outro pedido.
- Migrar ou reescrever qualquer payload. O índice é retroativo (medição 3); nada é tocado.

## Modos de falha, e o que acontece em cada um

| falha | comportamento |
|---|---|
| coleção não existe | `require_existing` levanta, como hoje. Uma leitura não cria coleção |
| índice não existe | criado com a coleção quando ela nasce; numa coleção preexistente, criado sob demanda na primeira listagem ordenada, que toma um 400 e tenta de novo |
| criar índice é proibido (permissão) | `ensure_payload_index` engole a falha; a listagem degrada a ordem, **diz** e continua paginando |
| servidor sem suporte a `order_by` | mesma degradação, mesmo aviso |
| cursor corrompido, de outro acervo, ou de formato antigo | erro claro do `paging`, sem silêncio e sem devolver a página 1 fingindo ser a página 2. Vale para cursor ilegível, para cursor cujo valor não é um `updated_at` e para cursor de versão anterior |
| cursor de um modo entregue ao outro | recusado: o cursor declara se é ordenado, e continuar um caminho não ordenado não pede ordenação |
| falha que não é 400 (404, 403, 408, 429, 5xx, timeout, sem status) | **sobe como falha**, nunca vira página. Um acervo inalcançável relatado como listagem bem-sucedida é a pior mentira possível aqui |
| 400 que não é sobre ordenação | degrada, e o aviso cita o texto que o servidor devolveu, sem diagnosticar causa. Não é alcançável pela API pública hoje: o único filtro que sai daqui é o `must_not/has_id` que o `paging` monta |
| registro sem `updated_at` | invisível na listagem ordenada (medição 7). Hoje são **zero** (744 de 744 têm o campo); o aviso de degradação não cobre esse caso, e a listagem ordenada não mente sobre ele: ela ordena o que tem data |
| empate de timestamp em grafias diferentes | tratado como um só instante. Comparar o texto cru fazia a caminhada repetir e nunca terminar |

## Restrições do repositório que este trabalho tem de respeitar

Medidas na árvore, porque cada uma reprova a suíte se ignorada:

| restrição | onde é imposta |
|---|---|
| a contagem de tools citada nos docs tem de bater com `len(tools.SCHEMAS)` | `tests/test_readme_fidelity.py:105-107`. Este trabalho **não** cria tool nova, então a contagem não muda |
| todo comando `qctx` citado nos docs tem de existir no parser | `tests/test_readme_fidelity.py:61` |
| travessão `—` é proibido em `README.md`, `docs/usage.md`, `docs/install.md` e `docs/architecture.md` | `tests/test_readme_fidelity.py:251` |
| a superfície de leitura de `MemoryStore` é uma lista explícita | `tests/test_host_equivalence.py:2095` (`READING_METHODS`). `list_page` já está lá e continua |

## O que fica sem prova automática

Honestidade sobre o alcance dos testes:

- que o Qdrant **de produção do usuário** aceite a criação do índice sob a chave de API dele.
  Provado na medição de hoje contra `zz_probe_*`, mas a coleção `claude_memory` ainda não tem
  índice nenhum; a primeira execução real é a prova.
- que nenhuma outra ferramenta dependa da ordem de id que a listagem tinha. Verificado por
  leitura: `next_offset` não é consumido por ninguém no repositório.
